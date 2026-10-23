"""Stage B: a table-conditioned, target-aware item encoder trained through the reader (TabLoRA).

E5's last three layers get LoRA adapters (r=16, alpha=32, the same layers and rank as
MulTaBench's Joint TAR), but the low-rank update is gated per row by the table encoding:

    dW(x_row) = B diag(1 + tanh(G h_tab(x_row))) A,     G initialised at zero,

so the adapter starts as plain LoRA and learns how a row's structured columns should
change the reading of its text. The encoder is trained through the reader's prefix loss
(no separate text-only head), K-fold cross-fitted exactly like stage A, and its reader
features go to the same tabular heads (`stack heads --features stackB`).

The hidden states entering layer 9 are cached once per (dataset, fold), so training only
runs the top three layers. Run on a GPU:

    python -m reading_agent2.tabcond --dataset REG_TEXT_VIDEO_GAMES_SALES --fold 0 [--gate 0]
"""
import argparse
import copy
import json
import math
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from reading_agent.data import get_split  # noqa: F401  (prepare() below uses the v1 split via stack.prepare)
from reading_agent.embed import MODEL_NAME, MODEL_REVISION
from reading_agent.features import FeaturePipeline
from reading_agent.model import ReadingAgent
from reading_agent.run import score
from reading_agent.train import Config, natural_order, sample_loss

from .stack import ROOT, _full_probs, prepare, reader_feats

TUNED = ["attention.self.query", "attention.self.key", "attention.self.value",
         "attention.output.dense", "intermediate.dense", "output.dense"]
N_TOP = 3
MAX_TOKENS = 128
VERSION = "2026-10-24-tablora"


# ---------------------------------------------------------------- encoder
class LoRALinear(nn.Module):
    """Frozen linear layer plus a low-rank update whose rank space can be gated per sequence."""

    def __init__(self, base, r=16, alpha=32, dropout=0.1):
        super().__init__()
        self.base, self.scale, self.drop = base, alpha / r, nn.Dropout(dropout)
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.gate = None  # (N, r) per-sequence gate, set by TopE5 before each forward

    def forward(self, x):
        z = self.drop(x) @ self.A.to(x.dtype).t()
        if self.gate is not None:
            z = z * self.gate.to(x.dtype).unsqueeze(1)
        return self.base(x) + (z @ self.B.to(x.dtype).t()) * self.scale


class TopE5(nn.Module):
    """E5's last N_TOP layers with (table-gated) LoRA, mean pooling and L2 normalisation."""

    def __init__(self, e5, d_tab, r=16, alpha=32, gate=True):
        super().__init__()
        self.layers = e5.encoder.layer[-N_TOP:]
        for p in self.layers.parameters():
            p.requires_grad_(False)
        self.loras = []
        for layer in self.layers:
            for name in TUNED:
                parent_name, attr = name.rsplit(".", 1)
                parent = layer.get_submodule(parent_name)
                lora = LoRALinear(getattr(parent, attr), r, alpha)
                setattr(parent, attr, lora)
                self.loras.append(lora)
        self.lora_params = nn.ModuleList(self.loras)
        self.r, self.use_gate = r, gate
        self.gate_proj = nn.Linear(d_tab, len(self.loras) * r)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)
        self.adapt = False  # False: no LoRA/gate (phase 1 and the static cache check)

    def forward(self, states, mask, h_tab_seq=None):
        """states (N, L, 384) layer-9 inputs, mask (N, L); h_tab_seq (N, d_tab) for the gate."""
        gates = None
        if self.adapt and self.use_gate and h_tab_seq is not None:
            gates = 1 + torch.tanh(self.gate_proj(h_tab_seq)).view(-1, len(self.loras), self.r)
        for i, lora in enumerate(self.loras):
            lora.gate = None if gates is None else gates[:, i]
            lora.B.requires_grad_(self.adapt)
            lora.A.requires_grad_(self.adapt)
        if not self.adapt:  # exact frozen E5: zero the update by skipping it
            saved = [l.scale for l in self.loras]
            for l in self.loras:
                l.scale = 0.0
        ext = (1.0 - mask[:, None, None, :].to(states.dtype)) * torch.finfo(states.dtype).min
        h = states
        for layer in self.layers:
            h = layer(h, attention_mask=ext)
            h = h[0] if isinstance(h, tuple) else h
        if not self.adapt:
            for l, s in zip(self.loras, saved):
                l.scale = s
        m = mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return F.normalize(pooled.float(), dim=-1)


class ItemStates:
    """Layer-9 input states of every distinct item string, stored flat (sum of lengths, 384) fp16."""

    def __init__(self, texts, device, batch_size=256):
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
        self.e5 = AutoModel.from_pretrained(MODEL_NAME, revision=MODEL_REVISION).to(device).half().eval()
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        chunks, lengths = [None] * len(texts), np.zeros(len(texts), np.int64)
        with torch.no_grad():
            for s in range(0, len(order), batch_size):
                ids = order[s:s + batch_size]
                enc = tok([texts[i] for i in ids], max_length=MAX_TOKENS, truncation=True, padding=True,
                          return_tensors="pt").to(device)
                hs = self.e5(**enc, output_hidden_states=True).hidden_states[-N_TOP - 1]  # input to layer 12-N_TOP
                n = enc["attention_mask"].sum(1)
                for j, i in enumerate(ids):
                    chunks[i] = hs[j, : n[j]].cpu()
                    lengths[i] = int(n[j])
        self.lengths = torch.as_tensor(lengths)
        self.offsets = torch.as_tensor(np.r_[0, np.cumsum(lengths)[:-1]])
        flat = torch.cat(chunks)
        on_gpu = flat.numel() * 2 < 4 * 1024 ** 3
        self.flat = flat.to(device) if on_gpu else flat.pin_memory()
        self.device = device

    def batch(self, ids):
        """Padded (N, L, 384) states and (N, L) mask for item ids (LongTensor on any device)."""
        ids = ids.cpu()
        lens = self.lengths[ids]
        L = int(lens.max())
        pos = torch.arange(L)
        mask = pos.unsqueeze(0) < lens.unsqueeze(1)
        idx = (self.offsets[ids].unsqueeze(1) + pos.unsqueeze(0).clamp(max=L - 1)).clamp(max=len(self.flat) - 1)
        idx = torch.where(mask, idx, torch.zeros_like(idx))
        states = self.flat[idx.to(self.flat.device)].to(self.device, non_blocking=True)
        return states, mask.to(self.device)


# ---------------------------------------------------------------- reader with the encoder inside
class TabCondReader(ReadingAgent):
    """The v1 reader whose item vectors come from TopE5 (static or adapted) instead of the frozen cache."""

    def attach(self, top, states):
        self.top, self.states = top, states

    def table_encoding(self, b):
        B = b["item_mask"].shape[0]
        toks = []
        names = self.name_proj(self.col_name_emb) if self.col_name_emb.numel() else None
        if self.n_num:
            toks.append(names[: self.n_num] + b["x_num"].unsqueeze(-1) * self.num_w + b["m_num"].unsqueeze(-1) * self.num_miss)
        if self.n_cat:
            toks.append(names[self.n_num:] + self.cat_emb(b["x_cat"] + self.cat_off))
        return self.tab_norm(self.tab_mlp(torch.cat(toks, 1).flatten(1))) if toks else self.tab_null.expand(B, -1)

    def encode(self, b):
        h_tab = self.table_encoding(b)
        if self.top.adapt:
            valid = b["item_mask"] > 0
            raw = torch.zeros(*valid.shape, 384, device=h_tab.device)
            if valid.any():
                rows = torch.arange(valid.shape[0], device=h_tab.device).unsqueeze(1).expand_as(valid)[valid]
                st, mk = self.states.batch(b["item_ids"][valid])
                with torch.autocast("cuda", dtype=torch.float16, enabled=st.is_cuda):
                    raw[valid] = self.top(st, mk, h_tab[rows]).to(raw.dtype)
        else:
            raw = b["items"].float()
        fields = self.name_proj(self.field_name_emb)[b["item_field"]]
        items = self.item_norm(self.item_proj(raw) + fields + self.pos_emb(b["item_pos"]))
        return h_tab, items


# ---------------------------------------------------------------- data
def item_layout(pipe, df):
    """Per-row item strings in the reader's slot order (as FeaturePipeline.transform builds them)."""
    return [[it[0] for it in pipe._row_items(list(r))] for r in df[pipe.text_cols].itertuples(index=False, name=None)]


def to_batch(P, ids, idx, device):
    b = {k: torch.from_numpy(np.ascontiguousarray(getattr(P, k)[idx])).to(device)
         for k in ("x_num", "m_num", "x_cat", "item_field", "item_pos", "item_mask")}
    b["items"] = torch.from_numpy(P.items[idx].astype(np.float32)).to(device)
    b["item_ids"] = torch.from_numpy(ids[idx]).to(device)
    b["y"] = torch.from_numpy(np.asarray(P.y[idx])).to(device)
    return b


def prefix_loss(model, b, task, full_weight=0.5, gen=None):
    mask = b["item_mask"]
    n_valid = mask.sum(1).long()
    order = (torch.rand(mask.shape, generator=gen).to(mask.device) + (1 - mask) * 10).argsort(1)
    preds = model.forward_order(b, order, n_valid)
    K1 = preds.shape[0]
    losses = torch.stack([sample_loss(preds[t].float(), b["y"], task) for t in range(K1)])
    valid = (torch.arange(K1, device=mask.device).unsqueeze(1) <= n_valid.unsqueeze(0)).float()
    mean_prefix = (losses * valid).sum(0) / valid.sum(0)
    full = losses.gather(0, n_valid.unsqueeze(0)).squeeze(0)
    return ((1 - full_weight) * mean_prefix + full_weight * full).mean()


@torch.no_grad()
def outputs(model, P, ids, idx, task, device, bs=128):
    """(read-all output, table-only output) for rows idx, natural order."""
    model.eval()
    alls, nones = [], []
    for s in range(0, len(idx), bs):
        b = to_batch(P, ids, idx[s:s + bs], device)
        n_valid = b["item_mask"].sum(1).long()
        preds = model.forward_order(b, natural_order(b["item_mask"].cpu()).to(device), n_valid).float()
        alls.append(preds.gather(0, n_valid.view(1, -1, 1).expand(1, -1, preds.shape[-1])).squeeze(0).cpu())
        nones.append(preds[0].cpu())
    return torch.cat(alls), torch.cat(nones)


def val_loss(model, P, ids, idx, task, device):
    out, _ = outputs(model, P, ids, idx, task, device)
    return float(sample_loss(out, torch.as_tensor(np.asarray(P.y[idx])), task).mean())


def fit(model, P, ids, tr, va, task, device, epochs, patience, groups, log, seed, amp, verbose=False, batch_rows=256):
    """groups: AdamW parameter groups [{"params": [...], "lr": ...}, ...]."""
    params = [p for g in groups for p in g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    gen = torch.Generator().manual_seed(seed)
    best, best_state, bad, best_ep = val_loss(model, P, ids, va, task, device), None, 0, -1
    best_state = copy.deepcopy({k: v.detach().clone() for k, v in model.state_dict().items()})
    for ep in range(epochs):
        model.train()
        perm = tr[torch.randperm(len(tr), generator=gen).numpy()]
        tot = 0.0
        for s in range(0, len(perm), batch_rows):
            b = to_batch(P, ids, perm[s:s + batch_rows], device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss = prefix_loss(model, b, task, gen=gen)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            tot += float(loss) * len(b["y"])
        vl = val_loss(model, P, ids, va, task, device)
        if verbose:
            log(f"      ep {ep} train {tot / len(perm):.4f} val {vl:.4f}")
        if vl < best - 1e-4:
            best, bad, best_ep = vl, 0, ep
            best_state = copy.deepcopy({k: v.detach().clone() for k, v in model.state_dict().items()})
        else:
            bad += 1
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    log(f"    best val {best:.4f} at epoch {best_ep} of {ep + 1}")
    return best


def adapted_state(model):
    """Reader, LoRA and gate parameters; the frozen E5 layers are rebuilt from the pinned checkpoint."""
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if not k.startswith("top.layers.") and ".base." not in k}


# ---------------------------------------------------------------- one cross-fitted run
def run(dataset, fold, gate=True, n_splits=3, adapt_epochs=10, adapt_patience=3, lora_lr=3e-4, tag="stackB",
        reader_lr=1e-4, verbose=False, max_folds=None):
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    device = "cuda"
    ds, x_tr, x_te, y_tr, y_te, te_idx, i_in, i_va, n_out, info = prepare(dataset, fold)
    task = ds.task
    pipe = FeaturePipeline(ds.num_cols, ds.cat_cols, ds.text_cols, namespace=f"e5_{dataset}",
                           max_items=max(8, len(ds.text_cols))).fit(x_tr)
    P_tr, P_te = pipe.transform(x_tr, y_tr), pipe.transform(x_te, y_te)
    lay_tr, lay_te = item_layout(pipe, x_tr), item_layout(pipe, x_te)
    texts = sorted({t for r in lay_tr + lay_te for t in r})
    tid = {t: i for i, t in enumerate(texts)}
    def id_matrix(layout, K):
        m = np.zeros((len(layout), K), np.int64)
        for i, r in enumerate(layout):
            m[i, : len(r)] = [tid[t] for t in r]
        return m
    K = pipe.max_items
    ids_tr, ids_te = id_matrix(lay_tr, K), id_matrix(lay_te, K)
    states = ItemStates(texts, device)
    log(f"{dataset} fold {fold}: {len(texts)} item strings cached ({states.flat.numel() * 2 / 1e9:.2f} GB), gate={gate}")
    # static item vectors from the same truncated states (phase 1 input; equals frozen E5 up to truncation)
    e5 = states.e5
    probe = TopE5(copy.deepcopy(e5), d_tab=128, gate=gate).to(device)
    static = np.zeros((len(texts), 384), np.float32)
    with torch.no_grad():
        for s in range(0, len(texts), 512):
            ids = torch.arange(s, min(s + 512, len(texts)))
            st, mk = states.batch(ids)
            static[s:s + len(ids)] = probe(st, mk).cpu().numpy()
    cache_vec = np.asarray(P_tr.items[P_tr.item_mask > 0], np.float32)
    agree = float((static[ids_tr[P_tr.item_mask > 0]] * cache_vec).sum(1).mean())
    log(f"static vectors vs v1 cache: mean cosine {agree:.4f} (1.0 except for items over {MAX_TOKENS} tokens)")
    del probe
    P_tr.items = static[ids_tr].astype(np.float16) * P_tr.item_mask[..., None]
    P_te.items = static[ids_te].astype(np.float16) * P_te.item_mask[..., None]

    cfg = Config(epochs=100, patience=15)
    wdir = ROOT / "results" / tag / f"{dataset}_f{fold}" / "weights"
    if max_folds is None:
        wdir.mkdir(parents=True, exist_ok=True)
        with (wdir / "pipe.pkl").open("wb") as f:
            pickle.dump(pipe, f, protocol=4)
    use_strat = task != "regression" and np.bincount(y_tr).min() >= n_splits
    kf = (StratifiedKFold if use_strat else KFold)(n_splits=n_splits, shuffle=True, random_state=fold)
    oof, te = {"all": None, "none": None}, {"all": 0.0, "none": 0.0}
    te_rows = np.arange(len(y_te))
    gate_norms = []
    for k, (a, b_idx) in enumerate(kf.split(np.zeros(len(y_tr)), y_tr if use_strat else None)):
        if max_folds is not None and k >= max_folds:  # diagnostics only: OOF features stay incomplete
            break
        a_in, a_va = train_test_split(a, test_size=0.15, random_state=fold + k)
        torch.manual_seed(fold + k)
        model = TabCondReader(n_num=pipe.n_num_in, cat_cards=pipe.cat_cardinalities, col_name_emb=pipe.col_name_emb,
                              field_name_emb=pipe.field_name_emb, n_out=n_out, d=cfg.d, layers=cfg.layers,
                              cell=cfg.cell, dropout=cfg.dropout, max_items=K)
        top = TopE5(copy.deepcopy(e5), d_tab=cfg.d, gate=gate)
        model.attach(top, states)
        model.to(device)
        log(f"  fold {k + 1}/{n_splits}: phase 1 (static items)")
        reader_params = [p for n, p in model.named_parameters() if not n.startswith("top.")]
        fit(model, P_tr, ids_tr, a_in, a_va, task, device, cfg.epochs, cfg.patience,
            [{"params": reader_params, "lr": cfg.lr}], log, seed=fold + k, amp=False)
        log(f"  fold {k + 1}/{n_splits}: phase 2 (TabLoRA{'' if gate else ' without gate'})")
        top.adapt = True
        adapt = [p for l in top.loras for p in (l.A, l.B)] + (list(top.gate_proj.parameters()) if gate else [])
        groups = [{"params": adapt, "lr": lora_lr}]
        if reader_lr > 0:
            groups.append({"params": reader_params, "lr": reader_lr})
        else:
            for p in reader_params:
                p.requires_grad_(False)
        fit(model, P_tr, ids_tr, a_in, a_va, task, device, adapt_epochs, adapt_patience, groups, log,
            seed=100 + fold + k, amp=True, verbose=verbose, batch_rows=64)
        if gate:
            gate_norms.append(float(top.gate_proj.weight.detach().norm()))
        for mode_out, rows, P, ids in (("oof", b_idx, P_tr, ids_tr), ("te", te_rows, P_te, ids_te)):
            out_all, out_none = outputs(model, P, ids, rows, task, device)
            f_all, f_none = reader_feats(out_all, task), reader_feats(out_none, task)
            if mode_out == "oof":
                for key, f in (("all", f_all), ("none", f_none)):
                    if oof[key] is None:
                        oof[key] = np.zeros((len(y_tr), f.shape[1]), np.float32)
                    oof[key][b_idx] = f
            else:
                te["all"] = te["all"] + f_all / n_splits
                te["none"] = te["none"] + f_none / n_splits
        if max_folds is None:
            torch.save(adapted_state(model), wdir / f"reader_{k}.pt")
        del model, top
        torch.cuda.empty_cache()
    out = ROOT / "results" / tag / f"{dataset}_f{fold}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "arrays.npz", y_tr=y_tr, y_te=y_te, te_idx=te_idx, i_in=i_in, i_va=i_va,
                        oof_all=oof["all"], oof_none=oof["none"], te_all=te["all"], te_none=te["none"])
    rs = lambda f, y: score(task, y, f if task == "regression" else _full_probs(f, task))
    if max_folds is not None:
        log("diagnostic run (max_folds): nothing saved")
        return
    info.update(n_out=n_out, max_items=K, max_tokens=MAX_TOKENS, config=dict(d=cfg.d, layers=cfg.layers, cell=cfg.cell, dropout=cfg.dropout),
                cv_seed=fold, n_splits=n_splits, gate=gate, adapt_epochs=adapt_epochs, lora_lr=lora_lr, reader_lr=reader_lr, version=VERSION,
                static_cosine=agree, gate_weight_norms=gate_norms, minutes=(time.time() - t0) / 60,
                reader_oof_score=rs(oof["all"], y_tr), reader_test_score=rs(te["all"], y_te))
    (out / "info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    log(f"saved {out} | reader OOF {info['reader_oof_score']:.4f} test {info['reader_test_score']:.4f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--gate", type=int, choices=[0, 1], default=1)
    ap.add_argument("--n_splits", type=int, default=3)
    ap.add_argument("--adapt_epochs", type=int, default=10)
    ap.add_argument("--lora_lr", type=float, default=1e-3)
    ap.add_argument("--reader_lr", type=float, default=0.0, help="phase-2 reader lr (0 freezes the reader)")
    ap.add_argument("--verbose", type=int, default=0)
    ap.add_argument("--max_folds", type=int, default=None, help="diagnostic: stop after this many cross-fit folds")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args(argv)
    tag = args.tag or ("stackB" if args.gate else "stackB_nogate")
    run(args.dataset, args.fold, bool(args.gate), args.n_splits, args.adapt_epochs, lora_lr=args.lora_lr, tag=tag,
        reader_lr=args.reader_lr, verbose=bool(args.verbose), max_folds=args.max_folds)


if __name__ == "__main__":
    main()
