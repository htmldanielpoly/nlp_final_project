"""Stage A: cross-fitted reader signals stacked into strong tabular heads.

Two steps per (dataset, fold), on the paper protocol and the v1 rows (pinned revisions):

    python -m reading_agent2.stack features --dataset REG_TEXT_ROTTEN_TOMATOES --fold 0   # CPU
    python -m reading_agent2.stack heads    --dataset REG_TEXT_ROTTEN_TOMATOES --fold 0   # GPU

`features` builds the Joint-Frozen frame (structured columns + PCA-30 of each field's
whole-text E5 embedding, as in v1) and K-fold cross-fitted reader predictions: each
training row's reader features come from a reader that never saw that row, and test
rows get the mean over the K readers. Nothing here sees the outer test labels.

`heads` fits each tabular head twice on identical rows: on the frozen frame alone
(the head's own Joint-Frozen baseline) and with the reader features appended (the
stack), so their difference is what the reader adds.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from reading_agent.baselines import frozen_frame
from reading_agent.data import get_split, load_dataset
from reading_agent.embed import release_model
from reading_agent.features import FeaturePipeline, whole_text_embeddings
from reading_agent.run import score
from reading_agent.train import Config, build_model, predict, train_warmup

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "results" / "stack"
VERSION = "2026-10-23-stackA"


def prepare(dataset, fold):
    """Split, label mapping and target scaling exactly as v1's run.py (paper protocol)."""
    ds = load_dataset(dataset)
    tr_idx, te_idx = get_split(ds.y, ds.is_cls, fold, "paper")
    x_tr, x_te = ds.x.iloc[tr_idx].reset_index(drop=True), ds.x.iloc[te_idx].reset_index(drop=True)
    y_tr_raw, y_te_raw = ds.y.iloc[tr_idx].to_numpy(), ds.y.iloc[te_idx].to_numpy()
    info = dict(dataset=dataset, fold=fold, task=ds.task, revision=ds.source_revision,
                num_cols=ds.num_cols, cat_cols=ds.cat_cols, text_cols=ds.text_cols)
    if ds.is_cls:
        classes = np.unique(y_tr_raw)
        cmap = {c: i for i, c in enumerate(classes)}
        y_tr = np.array([cmap[v] for v in y_tr_raw])
        keep = np.array([v in cmap for v in y_te_raw])
        x_te, y_te = x_te[keep].reset_index(drop=True), np.array([cmap[v] for v in y_te_raw[keep]])
        te_idx, n_out = te_idx[keep], len(classes)
        info.update(classes=[str(c) for c in classes], target_mean=0.0, target_scale=1.0)
    else:
        mu, sd = y_tr_raw.astype(float).mean(), y_tr_raw.astype(float).std() + 1e-9
        y_tr, y_te = (y_tr_raw.astype(float) - mu) / sd, (y_te_raw.astype(float) - mu) / sd
        n_out = 1
        info.update(classes=None, target_mean=float(mu), target_scale=float(sd))
    strat = y_tr if ds.is_cls and np.bincount(y_tr).min() >= 2 else None
    i_in, i_va = train_test_split(np.arange(len(y_tr)), test_size=0.2, random_state=fold, stratify=strat)
    return ds, x_tr, x_te, y_tr, y_te, te_idx, i_in, i_va, n_out, info


def reader_feats(out, task):
    """Reader output as features: class probabilities (binary: P(class 1)) or the regression value."""
    out = torch.as_tensor(out)
    if task == "regression":
        return out.reshape(-1, 1).numpy()
    p = torch.softmax(out, 1).numpy()
    return p[:, 1:] if task == "binary" else p


def crossfit(pipe, P_tr, P_te, y_tr, task, n_out, cfg, n_splits, seed, log):
    """K-fold out-of-fold reader features on the training rows, fold-averaged features on the test rows."""
    use_strat = task != "regression" and np.bincount(y_tr).min() >= n_splits
    kf = (StratifiedKFold if use_strat else KFold)(n_splits=n_splits, shuffle=True, random_state=seed)
    modes = ("all", "none")
    oof, te = {m: None for m in modes}, {m: 0.0 for m in modes}
    for k, (a, b) in enumerate(kf.split(np.zeros(len(y_tr)), y_tr if use_strat else None)):
        t = time.time()
        a_in, a_va = train_test_split(a, test_size=0.15, random_state=seed + k)
        torch.manual_seed(seed + k)
        m = build_model(pipe, n_out, cfg)
        train_warmup(m, P_tr.subset(a_in), P_tr.subset(a_va), task, cfg, log=lambda s: None)
        for mode in modes:
            f_b = reader_feats(predict(m, P_tr.subset(b), mode)[0], task)
            if oof[mode] is None:
                oof[mode] = np.zeros((len(y_tr), f_b.shape[1]), np.float32)
            oof[mode][b] = f_b
            te[mode] = te[mode] + reader_feats(predict(m, P_te, mode)[0], task) / n_splits
        log(f"  cross-fit {k + 1}/{n_splits}: best epoch {m.best_epoch}, {time.time() - t:.0f}s")
    return oof, te


def features(dataset, fold, n_splits=3, epochs=100, patience=15, threads=2):
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    torch.set_num_threads(threads)
    ds, x_tr, x_te, y_tr, y_te, te_idx, i_in, i_va, n_out, info = prepare(dataset, fold)
    ns = f"e5_{dataset}"
    max_items = max(8, len(ds.text_cols))
    pipe = FeaturePipeline(ds.num_cols, ds.cat_cols, ds.text_cols, namespace=ns, max_items=max_items).fit(x_tr)
    P_tr, P_te = pipe.transform(x_tr, y_tr), pipe.transform(x_te, y_te)
    E_tr, E_te = whole_text_embeddings(x_tr, ds.text_cols, ns), whole_text_embeddings(x_te, ds.text_cols, ns)
    F_tr, pcas = frozen_frame(x_tr, ds.num_cols, ds.cat_cols, E_tr)
    F_te, _ = frozen_frame(x_te, ds.num_cols, ds.cat_cols, E_te, pcas)
    release_model()
    torch.set_num_threads(threads)
    log(f"{dataset} fold {fold}: train {len(y_tr)} test {len(y_te)}, frozen frame {F_tr.shape[1]} cols")
    cfg = Config(epochs=epochs, patience=patience)
    oof, te = crossfit(pipe, P_tr, P_te, y_tr, ds.task, n_out, cfg, n_splits, seed=fold, log=log)
    out = WORK / f"{dataset}_f{fold}"
    out.mkdir(parents=True, exist_ok=True)
    F_tr.to_parquet(out / "F_tr.parquet")
    F_te.to_parquet(out / "F_te.parquet")
    np.savez_compressed(out / "arrays.npz", y_tr=y_tr, y_te=y_te, te_idx=te_idx, i_in=i_in, i_va=i_va,
                        oof_all=oof["all"], oof_none=oof["none"], te_all=te["all"], te_none=te["none"])
    info.update(n_splits=n_splits, epochs=epochs, patience=patience, version=VERSION, minutes=(time.time() - t0) / 60,
                reader_oof_score=score(ds.task, y_tr, oof["all"] if ds.task == "regression" else _full_probs(oof["all"], ds.task)),
                reader_test_score=score(ds.task, y_te, te["all"] if ds.task == "regression" else _full_probs(te["all"], ds.task)))
    (out / "info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    log(f"saved {out} | reader OOF {info['reader_oof_score']:.4f} test {info['reader_test_score']:.4f}")


def _full_probs(f, task):
    """Undo reader_feats' column drop for binary tasks so `score` sees one column per class."""
    return np.c_[1 - f, f] if task == "binary" else f


def design(F, extra, cat_prefix="cat_"):
    """Head input: frozen frame plus optional reader features; returns (frame, categorical column names)."""
    X = F.copy()
    if extra is not None:
        for j in range(extra.shape[1]):
            X[f"reader_{j}"] = extra[:, j]
    cats = [c for c in X.columns if c.startswith(cat_prefix)]
    return X, cats


def impute(X_tr, X_te, cats):
    """TabM needs complete numeric columns: training-row medians plus a missingness flag per affected column."""
    X_tr, X_te = X_tr.copy(), X_te.copy()
    for c in X_tr.columns:
        if c in cats or not (X_tr[c].isna().any() or X_te[c].isna().any()):
            continue
        med = X_tr[c].median()
        med = 0.0 if pd.isna(med) else med
        X_tr[f"{c}_missing"], X_te[f"{c}_missing"] = X_tr[c].isna().astype(float), X_te[c].isna().astype(float)
        X_tr[c], X_te[c] = X_tr[c].fillna(med), X_te[c].fillna(med)
    return X_tr, X_te


def fit_tabm(X_tr, y_tr, X_va, y_va, cats, task, seed):
    from pytabkit import TabM_D_Classifier, TabM_D_Regressor
    cls = TabM_D_Regressor if task == "regression" else TabM_D_Classifier
    m = cls(device="cuda" if torch.cuda.is_available() else "cpu", random_state=seed, verbosity=0)
    m.fit(X_tr, y_tr, X_val=X_va, y_val=y_va, cat_col_names=cats)
    return m


def tabm_predict(m, X, task):
    return m.predict(X) if task == "regression" else m.predict_proba(X)


def fit_tabpfn(X_tr, y_tr, cats, task, seed):
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    from tabpfn.constants import ModelVersion
    cls = TabPFNRegressor if task == "regression" else TabPFNClassifier
    m = cls.create_default_for_version(ModelVersion.V2_5, device="cuda" if torch.cuda.is_available() else "cpu",
                                       ignore_pretraining_limits=True, random_state=seed)
    m.fit(X_tr, y_tr)
    return m


def tabpfn_predict(m, X, task):
    """TabPFN predictions. On GPU OOM (memory held by another job) first wait and retry; then predict in row
    chunks, halving down to 1/8 of the rows (at least 512). Each test row is conditioned only on the context, so
    chunking does not change the predictions, but every chunk re-encodes the context, hence the floor."""
    pred = m.predict if task == "regression" else m.predict_proba
    chunk, floor, waits = len(X), min(len(X), max(512, len(X) // 8)), 0
    for _ in range(20):
        try:
            return np.concatenate([np.asarray(pred(X.iloc[i:i + chunk])) for i in range(0, len(X), chunk)])
        except Exception as e:  # tabpfn.errors.TabPFNCUDAOutOfMemoryError / torch.OutOfMemoryError
            if "out of memory" not in str(e).lower():
                raise
            torch.cuda.empty_cache()
            if waits < 3 or chunk <= floor:
                waits += 1
                time.sleep(90)
            else:
                chunk = max(floor, chunk // 2)
            print(f"  TabPFN OOM; retrying with chunk {chunk}", flush=True)
    raise RuntimeError("TabPFN kept running out of GPU memory")


def _ordinal(X_tr, X_te, cats):
    """TabPFN needs numeric input: ordinal-encode categoricals with the training-row vocabulary."""
    X_tr, X_te = X_tr.copy(), X_te.copy()
    for c in cats:
        vocab = {v: i for i, v in enumerate(pd.unique(X_tr[c]))}
        X_tr[c] = X_tr[c].map(vocab).astype(float)
        X_te[c] = X_te[c].map(vocab).astype(float)  # unseen categories become NaN
    return X_tr, X_te


PRIMARY = "blend_mean@stackB"  # fixed before the fold-0 results: TabLoRA reader blended with the mean of the heads


def heads(dataset, fold, which=("tabm", "tabpfn"), tag="v2", features=("stack", "stackB"), tabpfn_stack=False):
    """Fit each head once on the Joint-Frozen frame, then combine it with every available reader-feature set.

    For each feature set `feat` (stage A `stack`, stage B `stackB`, ...):
      reader_crossfit_all@feat   the cross-fitted reader alone (test rows: mean of the K readers)
      {head}_blend@feat          convex blend of the frozen head and that reader, weight tuned on the inner
                                 validation rows (reader predictions there are out-of-fold)
      {head}_stack@feat          head refit with the reader features appended (TabPFN only if tabpfn_stack)
      blend_mean@feat            equal-weight mean of the heads' blends (no further tuning)
    The frozen frame always comes from stage A's folder.
    """
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    src = WORK / f"{dataset}_f{fold}"
    info = json.loads((src / "info.json").read_text(encoding="utf-8"))
    base = np.load(src / "arrays.npz")
    F_tr, F_te = pd.read_parquet(src / "F_tr.parquet"), pd.read_parquet(src / "F_te.parquet")
    task, y_tr, y_te, i_in, i_va = info["task"], base["y_tr"], base["y_te"], base["i_in"], base["i_va"]
    full = lambda f: f if task == "regression" else _full_probs(f, task)
    readers = {}
    for feat in features:
        path = ROOT / "results" / feat / f"{dataset}_f{fold}" / "arrays.npz"
        if not path.exists():
            log(f"  no {feat} features yet; skipped")
            continue
        a = np.load(path)
        assert np.array_equal(a["y_tr"], y_tr) and np.array_equal(a["y_te"], y_te), f"{feat} rows differ"
        readers[feat] = dict(tr=np.c_[a["oof_all"], a["oof_none"]], te=np.c_[a["te_all"], a["te_none"]],
                             test=full(a["te_all"]), val=full(a["oof_all"][i_va]))
    n_classes = len(info["classes"] or [])
    results, preds, fitted, weights = [], {}, {}, {}

    def record(method, out, **kw):
        s = score(task, y_te, out)
        preds[method] = np.asarray(out, np.float32)
        results.append(dict(dataset=dataset, fold=fold, task=task, tag=tag, method=method, score=s, version=VERSION,
                            revision=info["revision"], **kw))
        log(f"  >> {method:28s} {s:.4f}")

    def fit_predict(head, ex_tr, ex_te, want_val):
        X_tr, cats = design(F_tr, ex_tr)
        X_te, _ = design(F_te, ex_te)
        if head == "tabm":
            Xi_tr, Xi_te = impute(X_tr, X_te, cats)
            m = fit_tabm(Xi_tr.iloc[i_in], y_tr[i_in], Xi_tr.iloc[i_va], y_tr[i_va], cats, task, seed=fold)
            if ex_tr is None:
                fitted[head] = m
            return tabm_predict(m, Xi_te, task), tabm_predict(m, Xi_tr.iloc[i_va], task)
        Xo_tr, Xo_te = _ordinal(X_tr, X_te, cats)
        out = tabpfn_predict(fit_tabpfn(Xo_tr, y_tr, cats, task, seed=fold), Xo_te, task)
        val = None
        if want_val:  # TabPFN conditions on all training rows; validation predictions need an inner fit
            val = tabpfn_predict(fit_tabpfn(Xo_tr.iloc[i_in], y_tr[i_in], cats, task, seed=fold), Xo_tr.iloc[i_va], task)
        return out, val

    for feat, r in readers.items():
        record(f"reader_crossfit_all@{feat}", r["test"])
    blends = {feat: [] for feat in readers}
    for head in which:
        if head == "tabpfn" and task != "regression" and n_classes > 10:
            log(f"  TabPFN skipped: {n_classes} classes > 10 (as in MulTaBench)")
            continue
        t = time.time()
        out, val = fit_predict(head, None, None, want_val=True)
        record(f"{head}_frozen", out, seconds=time.time() - t)
        for feat, r in readers.items():
            if np.asarray(val).size != r["val"].size:
                log(f"  {head}_blend@{feat} skipped: output columns differ (class missing in a split)")
                continue
            ws = np.linspace(0, 1, 21)
            vs = [score(task, y_tr[i_va], w * np.asarray(val).reshape(r["val"].shape) + (1 - w) * r["val"]) for w in ws]
            w = float(ws[int(np.argmax(vs))])
            record(f"{head}_blend@{feat}", w * np.asarray(out).reshape(r["test"].shape) + (1 - w) * r["test"], w=w)
            weights.setdefault(feat, {})[head] = w
            blends[feat].append(f"{head}_blend@{feat}")
        if head == "tabm" or tabpfn_stack:
            for feat, r in readers.items():
                t = time.time()
                out_s, _ = fit_predict(head, r["tr"], r["te"], want_val=False)
                record(f"{head}_stack@{feat}", out_s, seconds=time.time() - t)
        torch.cuda.empty_cache()
    for feat, members in blends.items():
        if members:
            record(f"blend_mean@{feat}", np.mean([preds[k] for k in members], axis=0), members=members)
    out_dir = ROOT / "results"
    with open(out_dir / f"{tag}.jsonl", "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    pdir = out_dir / "preds" / tag
    pdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pdir / f"{dataset}_f{fold}.npz", y_te=y_te, te_idx=base["te_idx"], **preds)
    # Inference artifacts: the fitted Joint-Frozen TabM and the blend weights (TabPFN-2.5 is refit on the
    # stored training frame at inference, which is how it conditions on training rows).
    mdir = out_dir / "heads" / tag / f"{dataset}_f{fold}"
    mdir.mkdir(parents=True, exist_ok=True)
    for head, m in fitted.items():  # torch.save so CUDA-trained heads load on CPU with map_location
        torch.save(m, mdir / f"{head}.pt", pickle_protocol=4)
    (mdir / "blend.json").write_text(json.dumps({"weights": weights, "heads": [h for h in which if any(h in v for v in weights.values())],
                                                 "version": VERSION}, indent=1), encoding="utf-8")
    log("done")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("features", "heads"):
        p = sub.add_parser(name)
        p.add_argument("--dataset", required=True)
        p.add_argument("--fold", type=int, default=0)
        if name == "features":
            p.add_argument("--n_splits", type=int, default=3)
            p.add_argument("--epochs", type=int, default=100)
            p.add_argument("--patience", type=int, default=15)
            p.add_argument("--threads", type=int, default=2)
        else:
            p.add_argument("--heads", default="tabm,tabpfn")
            p.add_argument("--tag", default="v2")
            p.add_argument("--features", default="stack,stackB", help="reader-feature folders: stack (A), stackB (TabLoRA), ...")
            p.add_argument("--tabpfn_stack", type=int, default=0)
    args = ap.parse_args(argv)
    if args.cmd == "features":
        features(args.dataset, args.fold, args.n_splits, args.epochs, args.patience, args.threads)
    else:
        heads(args.dataset, args.fold, tuple(args.heads.split(",")), args.tag, tuple(args.features.split(",")),
              bool(args.tabpfn_stack))


if __name__ == "__main__":
    main()
