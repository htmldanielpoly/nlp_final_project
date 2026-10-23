"""Embed every text item a dataset needs for folds 0..4 (chunks + whole fields) into its E5 store.

    python -m reading_agent.precompute [--protocol paper] [--dry] [DATASET ...]
(current protocol: folds 0-2; paper protocol: folds 0-4. --dry only counts the texts still to embed.)
"""
import argparse
import time

import numpy as np

from .data import TEXT_DATASETS, get_split, load_dataset
from .embed import EmbeddingStore
from .features import FeaturePipeline, _is_missing


def precompute(name, folds=range(3), max_items=8, protocol="current", dry=False):
    t = time.time()
    ds = load_dataset(name)
    idx = np.unique(np.concatenate([np.concatenate(get_split(ds.y, ds.is_cls, f, protocol)) for f in folds]))
    x = ds.x.iloc[idx]
    pipe = FeaturePipeline(ds.num_cols, ds.cat_cols, ds.text_cols, namespace=f"e5_{name}",
                           max_items=max(max_items, len(ds.text_cols)))  # same cap as run.py (cap_fields)
    texts = set()
    for r in x[ds.text_cols].itertuples(index=False, name=None):
        texts.update(it[0] for it in pipe._row_items(list(r)))
        for c, v in zip(ds.text_cols, r):
            texts.add(f"passage: {c}: {'' if _is_missing(v) else str(v).strip()}")
    store = EmbeddingStore(f"e5_{name}")
    todo = [s for s in texts if s not in store.index]
    print(f"{name}: {len(idx)} rows, {len(texts)} unique texts, {len(todo)} to embed", flush=True)
    if dry:
        return
    store.add(todo)
    print(f"{name}: done in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--protocol", default="current")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    for n in a.names or list(TEXT_DATASETS):
        precompute(n, folds=range(5) if a.protocol == "paper" else range(3), protocol=a.protocol, dry=a.dry)
