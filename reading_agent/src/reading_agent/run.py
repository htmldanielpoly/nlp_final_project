"""Run one (dataset, fold) experiment: agent (all modes) + baselines. Appends JSON lines to results/.

    python -m reading_agent.run --dataset MUL_TEXT_WOMEN_CLOTHING_REVIEW --fold 0
"""
import argparse
import copy
import hashlib
import json
import math
import os
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .baselines import cb_predict, fit_catboost, frozen_frame
from .baseline_cache import fit_or_load
from .data import load_dataset, get_split
from .embed import release_model
from .features import FeaturePipeline, whole_text_embeddings
from .estimator import ReadingEstimator
from .model import SoftAttentionReader
from .train import (Config, build_model, predict, sample_loss, to_batch, train_policy_gain, train_policy_imitation,
                    train_warmup)

RESULTS = Path(__file__).resolve().parents[2] / "results"


def score(task, y, out):
    """AUROC (binary / macro one-vs-rest) or R^2. `out` = logits/probs (cls) or predictions (reg)."""
    out = np.asarray(out)
    if task == "regression":
        return float(r2_score(y, out.reshape(-1)))
    if out.ndim == 2 and not np.allclose(out.sum(1), 1, atol=1e-3):
        out = torch.softmax(torch.as_tensor(out), 1).numpy()
    if task == "binary":
        return float(roc_auc_score(y, out[:, 1]))
    # macro OvR over the classes present in y (an absent class would make sklearn return NaN)
    present = np.unique(y)
    if len(present) == 2:
        return float(roc_auc_score(y == present[1], out[:, present[1]] / out[:, present].sum(1).clip(1e-12)))
    sub = out[:, present]
    sub = sub / sub.sum(1, keepdims=True).clip(1e-12)
    return float(roc_auc_score(y, sub, multi_class="ovr", average="macro", labels=present))


def calibrate_lam(pol, prep, target, lo=-0.5, hi=1.0, iters=14):
    """Bisection for the cost lam at which the gain policy reads `target` of the items on `prep`.
    The read fraction is non-increasing in lam (greedy prefix), so widen the bracket until it contains the target."""
    if not 0 <= target <= 1:
        raise ValueError("target budget must lie between zero and one")
    total = float(prep.item_mask.sum())
    if total == 0:
        return 0.0
    def frac_at(l):
        return float(predict(pol, prep, "gain", lam=l)[1].float().sum()) / total
    for _ in range(20):
        if frac_at(hi) <= target:
            break
        lo, hi = hi, hi * 2
    for _ in range(20):
        if frac_at(lo) >= target:
            break
        hi, lo = lo, lo * 2
    for _ in range(iters):
        mid = (lo + hi) / 2
        frac = frac_at(mid)
        lo, hi = (lo, mid) if frac < target else (mid, hi)  # reads fewer than target -> lower the cost
    return (lo + hi) / 2


def fixed_budget_counts(mask, target, seed=0):
    """Allocate an independently specified aggregate budget using row capacities only."""
    if not 0 <= target <= 1:
        raise ValueError("target budget must lie between zero and one")
    desired = target * np.asarray(mask).sum(1)
    counts = np.floor(desired).astype(np.int64)
    remaining = int(np.floor(desired.sum() + 0.5)) - int(counts.sum())
    fractional = desired - counts
    eligible = np.flatnonzero(fractional > 0)
    if remaining:
        rng = np.random.RandomState(seed)
        selected = rng.choice(eligible, size=remaining, replace=False,
                              p=fractional[eligible] / fractional[eligible].sum())
        counts[selected] += 1
    return counts


def train_soft(pipe, tr, va, task, n_out, cfg, log):
    kw = dict(n_num=pipe.n_num_in, cat_cards=pipe.cat_cardinalities, col_name_emb=pipe.col_name_emb,
              field_name_emb=pipe.field_name_emb, n_out=n_out, d=cfg.d, layers=cfg.layers, cell=cfg.cell,
              dropout=cfg.dropout, max_items=pipe.max_items)
    torch.manual_seed(cfg.seed)
    m = SoftAttentionReader(kw)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best, best_state, bad = math.inf, None, 0
    for ep in range(cfg.epochs):
        m.train()
        perm = np.random.RandomState(ep).permutation(len(tr))
        for s in range(0, len(tr), cfg.batch_size):
            b = to_batch(tr, perm[s:s + cfg.batch_size])
            loss = sample_loss(m(b)[0], b["y"], task).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        m.eval()
        with torch.no_grad():
            b = to_batch(va)
            vl = sample_loss(m(b)[0], b["y"], task).mean().item()
        if vl < best - 1e-4:
            best, best_state, bad = vl, copy.deepcopy(m.state_dict()), 0
        else:
            bad += 1
        if bad >= cfg.patience:
            break
    m.load_state_dict(best_state)
    log(f"  soft-attention best val {best:.4f} (ep {ep})")
    return m


def agent_feats(out, task):
    """Agent outputs as stacking features: class probabilities (drop one column for binary) or the regression value."""
    out = torch.as_tensor(out)
    if task == "regression":
        return out.reshape(-1, 1).numpy()
    p = torch.softmax(out, 1).numpy()
    return p[:, 1:] if task == "binary" else p


def to_prob(out, task):
    out = torch.as_tensor(out)
    return out.reshape(-1).numpy() if task == "regression" else torch.softmax(out, 1).numpy()


def crossfit_agent(pipe, P_tr, P_te, y_tr, task, n_out, cfg, n_splits=3, seed=0, log=print):
    """Out-of-fold agent predictions on the train split (no leakage into the stacker) + averaged test predictions."""
    from sklearn.model_selection import KFold, StratifiedKFold
    kf = (StratifiedKFold if task != "regression" and np.bincount(y_tr).min() >= n_splits else KFold)(
        n_splits=n_splits, shuffle=True, random_state=seed)
    oof = {m: None for m in ("all", "none")}
    te = {m: 0 for m in ("all", "none")}
    for k, (a, b) in enumerate(kf.split(np.zeros(len(y_tr)), y_tr if task != "regression" else None)):
        a_in, a_va = train_test_split(a, test_size=0.15, random_state=seed + k)
        m = build_model(pipe, n_out, cfg)
        train_warmup(m, P_tr.subset(a_in), P_tr.subset(a_va), task, cfg, log=lambda s: None)
        for mode in ("all", "none"):
            f_b = agent_feats(predict(m, P_tr.subset(b), mode)[0], task)
            if oof[mode] is None:
                oof[mode] = np.zeros((len(y_tr), f_b.shape[1]), np.float32)
            oof[mode][b] = f_b
            te[mode] = te[mode] + agent_feats(predict(m, P_te, mode)[0], task) / n_splits
        log(f"  cross-fit fold {k} done")
    return oof, te


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--item_mode", default="chunk")
    ap.add_argument("--max_items", type=int, default=8)
    ap.add_argument("--cell", default="gru")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lams", default="-1,0,0.01,0.03,0.06,0.1,0.2,0.4")
    ap.add_argument("--skip", default="", help="comma list: catboost,soft,hybrid,policy,agent (agent = baselines only)")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--key_modes", default="content,meta")
    ap.add_argument("--extra", default="", help="comma list: imitation")
    ap.add_argument("--seeds", type=int, default=1, help="agent seeds averaged for read-all / blend")
    ap.add_argument("--gain_data", default="val", help="val | trainval (data used to fit the gain head)")
    ap.add_argument("--gain_loss", default="mse", help="mse | huber")
    ap.add_argument("--gain_epochs", type=int, default=0)
    ap.add_argument("--sched", default="none", help="none | plateau (warm-up lr schedule)")
    ap.add_argument("--budgets", default="0.1,0.25,0.5,0.75", help="target read fractions (lam tuned on val)")
    ap.add_argument("--field_meta", type=int, default=0, help="1: give the table encoder [present, log length] per text field")
    ap.add_argument("--cap_fields", type=int, default=1, help="1: max_items >= number of text fields")
    ap.add_argument("--protocol", default="current", help="current (3-fold x2, HF master) | paper (paper_version holdouts, folds 0-4)")
    ap.add_argument("--data_revision", default=None, help="pin a Hugging Face dataset revision")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--min_epochs", type=int, default=0)
    ap.add_argument("--early_stop_ema", type=float, default=0.0, help="EMA retention for patience; checkpoint still uses raw validation loss")
    ap.add_argument("--controls", type=int, choices=[0, 1], default=0, help="add fixed-budget random and natural-order controls")
    ap.add_argument("--save_models", type=int, choices=[0, 1], default=1, help="save reusable reader/policy and blend inference artifacts")
    ap.add_argument("--warmup_only", type=int, choices=[0, 1], default=0, help="1: stop after the warm-up (diagnostic; history saved, no _done row)")
    ap.add_argument("--baseline_cache", type=int, choices=[0, 1], default=1, help="reuse completed CatBoost stages only when actual training inputs/settings match")
    ap.add_argument("--threads", type=int, default=0, help="Torch CPU threads; 0 preserves automatic setting, 1 useful for reproducibility diagnostics")
    ap.add_argument("--deterministic", type=int, choices=[0, 1], default=0, help="request Torch deterministic algorithms (opt-in; keep a separate experimental tag)")
    args = ap.parse_args(argv)
    if args.seeds < 1 or args.max_items < 1 or args.gain_epochs < 0 or args.threads < 0:
        ap.error("seeds/max_items must be positive and gain_epochs nonnegative")
    if args.key_modes and any(k not in {"content", "meta"} for k in args.key_modes.split(",")):
        ap.error("key_modes must contain only content or meta")
    cfg = Config(cell=args.cell, layers=args.layers, sched=args.sched, epochs=args.epochs,
                 patience=args.patience, min_epochs=args.min_epochs, early_stop_ema=args.early_stop_ema)
    skip = set(filter(None, args.skip.split(",")))
    torch.set_num_threads(args.threads or os.cpu_count() or 4)
    torch.use_deterministic_algorithms(bool(args.deterministic))
    RESULTS.mkdir(exist_ok=True)
    out_path = RESULTS / f"{args.tag}.jsonl"
    t0 = time.time()
    stage_times = {}

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    ds = load_dataset(args.dataset, revision=args.data_revision)
    tr_idx, te_idx = get_split(ds.y, ds.is_cls, args.fold, args.protocol)
    x_tr, x_te = ds.x.iloc[tr_idx].reset_index(drop=True), ds.x.iloc[te_idx].reset_index(drop=True)
    y_tr_raw, y_te_raw = ds.y.iloc[tr_idx].to_numpy(), ds.y.iloc[te_idx].to_numpy()
    ds.x = ds.y = None  # free the full table (several GB on the big datasets)
    if ds.is_cls:
        classes = np.unique(y_tr_raw)
        cmap = {c: i for i, c in enumerate(classes)}
        y_tr = np.array([cmap[v] for v in y_tr_raw])
        keep = np.array([v in cmap for v in y_te_raw])  # drop test rows of unseen classes (rare)
        x_te, y_te = x_te[keep].reset_index(drop=True), np.array([cmap[v] for v in y_te_raw[keep]])
        if not keep.all():
            print(f"WARNING: dropped {int((~keep).sum())} test rows of classes unseen in train", flush=True)
        te_idx_kept = te_idx[keep]
        n_out = len(classes)
    else:
        mu, target_sd = y_tr_raw.astype(float).mean(), y_tr_raw.astype(float).std() + 1e-9
        y_tr, y_te = (y_tr_raw.astype(float) - mu) / target_sd, (y_te_raw.astype(float) - mu) / target_sd
        te_idx_kept = te_idx
        n_out = 1
    # inner validation split (early stopping, policy training)
    strat = y_tr if ds.is_cls and np.bincount(y_tr).min() >= 2 else None
    i_in, i_va = train_test_split(np.arange(len(y_tr)), test_size=0.2, random_state=args.fold, stratify=strat)
    log(f"{args.dataset} fold {args.fold}: train {len(i_in)} val {len(i_va)} test {len(y_te)} | "
        f"num {len(ds.num_cols)} cat {len(ds.cat_cols)} text {len(ds.text_cols)}")

    rows = []
    run_id = f"{args.dataset}_f{args.fold}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    def fingerprint(frame, target):
        values = pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()
        labels = pd.util.hash_pandas_object(pd.Series(target), index=False).to_numpy().tobytes()
        return hashlib.sha256(values + labels).hexdigest()
    provenance = dict(data_source=ds.source, data_revision=ds.source_revision,
                      train_fingerprint=fingerprint(x_tr, y_tr_raw),
                      test_fingerprint=fingerprint(x_te, y_te_raw[keep] if ds.is_cls else y_te_raw),
                      leaderboard_pairing="unverified_data", dropped_test_rows=int(len(te_idx)-len(te_idx_kept)))
    base = dict(dataset=args.dataset, fold=args.fold, task=ds.task, tag=args.tag, run_id=run_id, protocol=args.protocol, field_meta=args.field_meta, sched=args.sched, item_mode=args.item_mode,
                cell=args.cell, layers=args.layers, args=vars(args), config=asdict(cfg), provenance=provenance,
                implementation_version="2026-10-16-controls")
    base["runtime"] = dict(torch_threads=torch.get_num_threads(), deterministic=torch.are_deterministic_algorithms_enabled(),
                            torch_version=torch.__version__, numpy_version=np.__version__)

    preds = {}  # test predictions per method, saved with the _done row (recompute metrics/paired tests offline)

    def record(method, s, pred=None, prediction_orders=None, reads_per_row=None, **kw):
        pred_key = method + (f"@{kw['budget']:g}" if "budget" in kw else f"@{kw['lam']:g}" if "lam" in kw else "")
        if pred is not None:
            preds[pred_key] = np.asarray(pred, np.float32)
            kw["prediction_key"] = pred_key
        if prediction_orders is not None:
            keys = []
            for order, prediction in enumerate(prediction_orders):
                key = f"{pred_key}__order_{order}"
                preds[key] = np.asarray(prediction, np.float32)
                keys.append(key)
            kw.update(prediction_keys=keys, prediction_aggregation="mean_metric_over_orders")
        if reads_per_row is not None:
            preds[f"{pred_key}__reads"] = np.asarray(reads_per_row, np.int64)
        r = dict(base, method=method, score=s, **kw)
        rows.append(r)
        with open(out_path, "a") as f:
            f.write(json.dumps(r) + "\n")
        log(f"  >> {method:28s} {s:.4f} {kw if kw else ''}")

    def save_preds():
        d = RESULTS / "preds" / args.tag
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / f"{run_id}.npz", y_te=np.asarray(y_te), te_idx=te_idx_kept, **preds)
        (d / f"{run_id}.json").write_text(json.dumps(dict(base, classes=classes.tolist() if ds.is_cls else None,
               target_mean=0.0 if ds.is_cls else mu, target_scale=1.0 if ds.is_cls else target_sd)), encoding="utf-8")

    # ---------------- features
    ns = f"e5_{args.dataset}"
    # every text field gets at least one slot (RT has 13 fields; a cap of 8 hid 5 of them, exp #10)
    max_items = max(args.max_items, len(ds.text_cols)) if args.cap_fields else args.max_items
    base["effective_max_items"] = max_items
    pipe = FeaturePipeline(ds.num_cols, ds.cat_cols, ds.text_cols, namespace=ns, item_mode=args.item_mode,
                           max_items=max_items, field_meta=bool(args.field_meta)).fit(x_tr)
    P_tr = pipe.transform(x_tr, y_tr)
    P_te = pipe.transform(x_te, y_te)
    P_in, P_va = P_tr.subset(i_in), P_tr.subset(i_va)
    nv = P_tr.item_mask.sum(1)
    log(f"items/row mean {nv.mean():.2f} max {nv.max():.0f}; features ready")

    # ---------------- Joint Frozen (CatBoost, PCA-30 per column)
    if "catboost" not in skip:
        E_tr = whole_text_embeddings(x_tr, ds.text_cols, ns)
        E_te = whole_text_embeddings(x_te, ds.text_cols, ns)
        F_tr, pcas = frozen_frame(x_tr, ds.num_cols, ds.cat_cols, E_tr)
        F_te, _ = frozen_frame(x_te, ds.num_cols, ds.cat_cols, E_te, pcas)
        del E_tr, E_te
        stage_start = time.time()
        cache_dir = RESULTS / "baseline_cache" / args.dataset / f"fold_{args.fold}"
        cb, cache_status = fit_or_load(F_tr.iloc[i_in], y_tr[i_in], F_tr.iloc[i_va], y_tr[i_va], ds.task,
                                      cache_dir / "frozen", enabled=bool(args.baseline_cache))
        stage_times["catboost_frozen_seconds"] = time.time() - stage_start
        log(f"CatBoost Frozen cache={cache_status}")
        cb_frozen = cb
        cb_va, cb_te = cb_predict(cb, F_tr.iloc[i_va], ds.task), cb_predict(cb, F_te, ds.task)
        record("catboost_frozen", score(ds.task, y_te, cb_te), pred=cb_te, baseline_cache_status=cache_status)
        Ft_tr, Ft_te = F_tr[[c for c in F_tr if not c.startswith("txt")]], F_te[[c for c in F_te if not c.startswith("txt")]]
        if Ft_tr.shape[1]:
            stage_start = time.time()
            cb, cache_status = fit_or_load(Ft_tr.iloc[i_in], y_tr[i_in], Ft_tr.iloc[i_va], y_tr[i_va], ds.task,
                                          cache_dir / "tab_only", enabled=bool(args.baseline_cache))
            stage_times["catboost_tab_only_seconds"] = time.time() - stage_start
            log(f"CatBoost tab-only cache={cache_status}")
            cbt_te = cb_predict(cb, Ft_te, ds.task)
            record("catboost_tab_only", score(ds.task, y_te, cbt_te), pred=cbt_te, baseline_cache_status=cache_status)

    release_model()
    # Embedding initialization may set its own CPU thread count. Restore the
    # explicitly requested reader setting before training.
    torch.set_num_threads(args.threads or os.cpu_count() or 4)
    base["runtime"]["torch_threads"] = torch.get_num_threads()
    if "agent" in skip:
        save_preds()
        record("_done", 0.0, minutes=(time.time() - t0) / 60, stage_times=stage_times)
        return rows
    # ---------------- reading agent
    torch.manual_seed(0)
    model = build_model(pipe, n_out, cfg)
    stage_start = time.time()
    best = train_warmup(model, P_in, P_va, ds.task, cfg, log)
    stage_times["warmup_seconds"] = time.time() - stage_start
    base["warmup_best_epoch"] = model.best_epoch
    base["warmup_best_val_loss"] = best
    history_dir = RESULTS / "history" / args.tag
    history_dir.mkdir(parents=True, exist_ok=True)
    (history_dir / f"{run_id}.json").write_text(json.dumps(model.training_history, indent=2), encoding="utf-8")
    (history_dir / f"{run_id}.meta.json").write_text(json.dumps(dict(config=asdict(cfg), early_stopped=model.early_stopped,
               ran_epochs=len(model.training_history), provenance=provenance, runtime=base["runtime"]), indent=2), encoding="utf-8")
    log(f"warm-up done, best val loss {best:.4f}")
    if args.warmup_only:
        return rows
    out_none, _ = predict(model, P_te, "none")
    record("agent_table_only", score(ds.task, y_te, out_none), pred=to_prob(out_none, ds.task))
    out_all, n_all = predict(model, P_te, "all")
    record("agent_read_all", score(ds.task, y_te, out_all), pred=to_prob(out_all, ds.task), reads=float(n_all.float().mean()))
    ens_va = to_prob(predict(model, P_va, "all")[0], ds.task)
    ens_te = to_prob(out_all, ds.task)
    ensemble_models = []
    for sd in range(1, args.seeds):
        cfg_s = copy.copy(cfg)
        cfg_s.seed = sd
        torch.manual_seed(sd)
        m2 = build_model(pipe, n_out, cfg_s)
        train_warmup(m2, P_in, P_va, ds.task, cfg_s, log=lambda m: None)
        ensemble_models.append(m2)
        ens_va = ens_va + to_prob(predict(m2, P_va, "all")[0], ds.task)
        ens_te = ens_te + to_prob(predict(m2, P_te, "all")[0], ds.task)
    ens_va, ens_te = ens_va / args.seeds, ens_te / args.seeds
    if args.seeds > 1:
        record(f"agent_read_all_ens{args.seeds}", score(ds.task, y_te, ens_te), pred=ens_te)

    if "soft" not in skip:
        soft = train_soft(pipe, P_in, P_va, ds.task, n_out, cfg, log)
        with torch.no_grad():
            soft_out = soft(to_batch(P_te))[0]
            record("soft_attention_all", score(ds.task, y_te, soft_out), pred=to_prob(soft_out, ds.task),
                   reads=float(nv.mean()))

    if "catboost" not in skip:
        # validation-tuned convex blend of CatBoost (Joint Frozen) and the agent (read-all)
        ag_va, ag_te = ens_va, ens_te
        ws = np.linspace(0, 1, 21)
        vs = [score(ds.task, y_tr[i_va], w * ag_va + (1 - w) * cb_va) for w in ws]
        w = float(ws[int(np.argmax(vs))])
        record("blend_catboost_agent", score(ds.task, y_te, w * ag_te + (1 - w) * cb_te), pred=w * ag_te + (1 - w) * cb_te, w=w)

    if "hybrid" not in skip and "catboost" not in skip:
        oof, te_f = crossfit_agent(pipe, P_tr, P_te, y_tr, ds.task, n_out, cfg, seed=args.fold, log=log)
        H_tr, H_te = F_tr.copy(), F_te.copy()
        for mode in ("all", "none"):
            for j in range(oof[mode].shape[1]):
                H_tr[f"agt_{mode}_{j}"] = oof[mode][:, j]
                H_te[f"agt_{mode}_{j}"] = te_f[mode][:, j]
        cb = fit_catboost(H_tr.iloc[i_in], y_tr[i_in], H_tr.iloc[i_va], y_tr[i_va], ds.task)
        hybrid_out = cb_predict(cb, H_te, ds.task)
        record("hybrid_catboost", score(ds.task, y_te, hybrid_out), pred=hybrid_out)

    artifact_root = RESULTS / "models" / args.tag / run_id
    def artifact(reader, thresholds=None, models=None):
        return ReadingEstimator.from_trained(pipe, reader, cfg, ds.task, classes=classes if ds.is_cls else None,
                 target_mean=0.0 if ds.is_cls else mu, target_scale=1.0 if ds.is_cls else target_sd,
                 thresholds=thresholds, metadata=base, ensemble_models=models)
    if args.save_models:
        saved = artifact(model, models=ensemble_models)
        if "catboost" not in skip:
            saved.with_catboost(cb_frozen, pcas, w)
        saved.save(artifact_root / "reader")

    if args.controls:
        for target in [float(v) for v in args.budgets.split(",") if v]:
            counts = fixed_budget_counts(P_te.item_mask, target)
            order_outputs = [predict(model, P_te, "random", budget=counts, seed=rs)[0] for rs in range(3)]
            order_scores = [score(ds.task, y_te, out) for out in order_outputs]
            fraction = float(counts.sum() / max(1, P_te.item_mask.sum()))
            record("random_fixed", float(np.mean(order_scores)), prediction_orders=[to_prob(out, ds.task) for out in order_outputs],
                   reads_per_row=counts, budget=target, frac=fraction, seed_scores=order_scores)
            natural_out, _ = predict(model, P_te, "natural", budget=counts)
            record("natural_fixed", score(ds.task, y_te, natural_out), pred=to_prob(natural_out, ds.task),
                   reads_per_row=counts, budget=target, frac=fraction)

    lams = [float(v) for v in args.lams.split(",")]
    if "policy" not in skip:
        for key_mode in args.key_modes.split(","):
            stage_start = time.time()
            thresholds = {}
            pol = copy.deepcopy(model)
            pol.key_mode = key_mode
            gdata = P_va if args.gain_data == "val" else P_tr
            train_policy_gain(pol, gdata, ds.task, cfg, loss_kind=args.gain_loss,
                              epochs=args.gain_epochs, log=lambda m: None)
            for lam in lams:
                out_p, n_p = predict(pol, P_te, "gain", lam=lam)
                frac = float(n_p.float().sum() / max(1, P_te.item_mask.sum()))
                record(f"gain_{key_mode}_{args.gain_data}_{args.gain_loss}", score(ds.task, y_te, out_p), pred=to_prob(out_p, ds.task), reads_per_row=n_p.numpy(), lam=lam, reads=float(n_p.float().mean()), frac=frac)
                out_r, _ = predict(model, P_te, "random", budget=n_p.numpy())
                record(f"random_budget_{key_mode}", score(ds.task, y_te, out_r), pred=to_prob(out_r, ds.task), lam=lam, frac=frac)
            # budget-targeted curve: per target fraction of items read, pick lam on val (label-free, uses g only)
            for target in [float(v) for v in args.budgets.split(",") if v]:
                lam = calibrate_lam(pol, P_va, target)
                thresholds[f"{target:g}"] = lam
                out_p, n_p = predict(pol, P_te, "gain", lam=lam)
                frac = float(n_p.float().sum() / max(1, P_te.item_mask.sum()))
                record(f"gainB_{key_mode}", score(ds.task, y_te, out_p), pred=to_prob(out_p, ds.task), reads_per_row=n_p.numpy(), budget=target, lam=lam, frac=frac)
                # matched random reader (same per-row item count), averaged over 3 orders
                outs_r = [predict(model, P_te, "random", budget=n_p.numpy(), seed=rs)[0] for rs in range(3)]
                sc_r = [score(ds.task, y_te, o) for o in outs_r]
                record(f"randomB_{key_mode}", float(np.mean(sc_r)), prediction_orders=[to_prob(out, ds.task) for out in outs_r], reads_per_row=n_p.numpy(), budget=target,
                       frac=frac, seed_scores=sc_r)
                if args.controls:
                    # same total reads as the policy, spread by row capacity: separates item choice
                    # (gainB vs randomB) from the policy's allocation of reads across rows (randomB vs randomAgg)
                    agg = fixed_budget_counts(P_te.item_mask, frac)
                    outs_a = [predict(model, P_te, "random", budget=agg, seed=rs)[0] for rs in range(3)]
                    sc_a = [score(ds.task, y_te, o) for o in outs_a]
                    agg_frac = float(agg.sum() / max(1, P_te.item_mask.sum()))
                    record(f"randomAgg_{key_mode}", float(np.mean(sc_a)), prediction_orders=[to_prob(out, ds.task) for out in outs_a],
                           reads_per_row=agg, budget=target, frac=agg_frac, seed_scores=sc_a)
                    out_n, _ = predict(model, P_te, "natural", budget=agg)
                    record(f"naturalAgg_{key_mode}", score(ds.task, y_te, out_n), pred=to_prob(out_n, ds.task),
                           reads_per_row=agg, budget=target, frac=agg_frac)
            if args.save_models:
                artifact(pol, thresholds).save(artifact_root / f"gain_{key_mode}")
            stage_times[f"policy_{key_mode}_train_and_eval_seconds"] = time.time() - stage_start
    if "imitation" in args.extra:
        for lam in lams:
            pol = copy.deepcopy(model)
            train_policy_imitation(pol, P_va, ds.task, lam, cfg, log=lambda m: None)
            out_p, n_p = predict(pol, P_te, "policy")
            frac = float(n_p.float().sum() / P_te.item_mask.sum())
            record("imitation_policy", score(ds.task, y_te, out_p), lam=lam, reads=float(n_p.float().mean()), frac=frac)
            out_o, n_o = predict(model, P_te, "oracle", lam=lam, task=ds.task)
            record("label_oracle", score(ds.task, y_te, out_o), lam=lam, reads=float(n_o.float().mean()))
    save_preds()
    record("_done", 0.0, minutes=(time.time() - t0) / 60, stage_times=stage_times)
    log("done")
    return rows


if __name__ == "__main__":
    main()
