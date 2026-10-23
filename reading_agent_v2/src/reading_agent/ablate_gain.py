"""Ablate how the gain head is trained, on one (dataset, fold), sharing one warmed-up agent.

Reports, per variant, the advantage over a random reader at matched budgets, averaged over the λ grid.
    python -m reading_agent.ablate_gain --dataset MUL_TEXT_WOMEN_CLOTHING_REVIEW --fold 0
"""
import argparse
import copy
import json
import time

import numpy as np
import torch

from .run import RESULTS, score
from .train import Config, build_model, predict, train_policy_gain, train_warmup
from .data import get_split, load_dataset
from .features import FeaturePipeline
from sklearn.model_selection import train_test_split

VARIANTS = [
    dict(name="val_mse", data="val", loss="mse", epochs=15),
    dict(name="val_huber", data="val", loss="huber", epochs=15),
    dict(name="trainval_mse", data="trainval", loss="mse", epochs=15),
    dict(name="trainval_huber", data="trainval", loss="huber", epochs=15),
    dict(name="val_mse_40ep", data="val", loss="mse", epochs=40),
]
LAMS = [0.0, 0.01, 0.03, 0.06, 0.1, 0.2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--variants", default="")
    a = ap.parse_args()
    t0 = time.time()
    ds = load_dataset(a.dataset)
    tr, te = get_split(ds.y, ds.is_cls, a.fold)
    x_tr, x_te = ds.x.iloc[tr].reset_index(drop=True), ds.x.iloc[te].reset_index(drop=True)
    y_tr_raw, y_te_raw = ds.y.iloc[tr].to_numpy(), ds.y.iloc[te].to_numpy()
    if ds.is_cls:
        cl = {c: i for i, c in enumerate(np.unique(y_tr_raw))}
        keep = np.array([v in cl for v in y_te_raw])
        x_te = x_te[keep].reset_index(drop=True)
        y_tr, y_te = np.array([cl[v] for v in y_tr_raw]), np.array([cl[v] for v in y_te_raw[keep]])
        n_out = len(cl)
    else:
        mu, sd = y_tr_raw.mean(), y_tr_raw.std() + 1e-9
        y_tr, y_te, n_out = (y_tr_raw - mu) / sd, (y_te_raw - mu) / sd, 1
    strat = y_tr if ds.is_cls and np.bincount(y_tr).min() >= 2 else None
    i_in, i_va = train_test_split(np.arange(len(y_tr)), test_size=0.2, random_state=a.fold, stratify=strat)
    pipe = FeaturePipeline(ds.num_cols, ds.cat_cols, ds.text_cols, namespace=f"e5_{a.dataset}").fit(x_tr)
    P_tr, P_te = pipe.transform(x_tr, y_tr), pipe.transform(x_te, y_te)
    P_in, P_va = P_tr.subset(i_in), P_tr.subset(i_va)
    cfg = Config()
    torch.manual_seed(0)
    model = build_model(pipe, n_out, cfg)
    train_warmup(model, P_in, P_va, ds.task, cfg, log=lambda m: None)
    full = score(ds.task, y_te, predict(model, P_te, "all")[0])
    print(f"[{time.time()-t0:.0f}s] read-all {full:.4f}", flush=True)
    sel = [v for v in VARIANTS if not a.variants or v["name"] in a.variants.split(",")]
    out = []
    for v in sel:
        for key_mode in ("content", "meta"):
            pol = copy.deepcopy(model)
            pol.key_mode = key_mode
            data = P_va if v["data"] == "val" else P_tr
            train_policy_gain(pol, data, ds.task, cfg, loss_kind=v["loss"], epochs=v["epochs"], log=lambda m: None)
            diffs, curve = [], []
            for lam in LAMS:
                o, n = predict(pol, P_te, "gain", lam=lam)
                frac = float(n.float().sum() / P_te.item_mask.sum())
                s_g = score(ds.task, y_te, o)
                s_r = np.mean([score(ds.task, y_te, predict(model, P_te, "random", budget=n.numpy(), seed=s)[0])
                               for s in range(3)])
                diffs.append(s_g - s_r)
                curve.append((round(frac, 3), round(s_g, 4), round(float(s_r), 4)))
            r = dict(dataset=a.dataset, fold=a.fold, variant=v["name"], key_mode=key_mode,
                     mean_adv=float(np.mean(diffs)), curve=curve, read_all=full)
            out.append(r)
            with open(RESULTS / "ablate_gain.jsonl", "a") as f:
                f.write(json.dumps(r) + "\n")
            print(f"[{time.time()-t0:.0f}s] {v['name']:16s} {key_mode:7s} mean adv vs random {np.mean(diffs):+.4f}  "
                  f"{curve}", flush=True)


if __name__ == "__main__":
    main()
