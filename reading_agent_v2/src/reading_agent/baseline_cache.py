"""Persist expensive CatBoost stages independently of reader training.

Reuse requires matching actual train/validation features, labels, model settings
and CatBoost version. A manifest is committed last; incomplete writes are ignored.
This does not cache or reuse any reader/policy fit across folds.
"""
import hashlib
import json
import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import fit_catboost


def file_digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def baseline_identity(train, y_train, val, y_val, task, *, seed=0, iterations=3000):
    import catboost
    digest = hashlib.sha256()
    for frame, y in [(train, y_train), (val, y_val)]:
        schema = [(str(c), str(dtype)) for c, dtype in zip(frame.columns, frame.dtypes)]
        digest.update(json.dumps(schema).encode())
        digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
        digest.update(pd.util.hash_pandas_object(pd.Series(np.asarray(y)), index=False).to_numpy().tobytes())
    recipe = dict(format_version=1, task=task, seed=seed, iterations=iterations,
                  catboost_version=catboost.__version__, learning_rate=0.05, depth=6,
                  od_wait=200, thread_count=4, implementation="catboost-stage-v1")
    digest.update(json.dumps(recipe, sort_keys=True).encode())
    return digest.hexdigest(), recipe


def fit_or_load(train, y_train, val, y_val, task, directory, *, seed=0, iterations=3000,
                enabled=True):
    if not enabled:
        return fit_catboost(train, y_train, val, y_val, task, seed=seed, iterations=iterations), "disabled"
    identity, recipe = baseline_identity(train, y_train, val, y_val, task, seed=seed, iterations=iterations)
    target = Path(directory) / identity
    manifest_path, model_path = target / "manifest.json", target / "catboost.cbm"
    if manifest_path.exists() and model_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("identity") != identity or manifest.get("recipe") != recipe:
            raise ValueError(f"Baseline cache manifest mismatch at {target}")
        actual = file_digest(model_path)
        if actual != manifest.get("model_sha256"):
            raise ValueError(f"Baseline cache model checksum mismatch at {target}")
        from catboost import CatBoostClassifier, CatBoostRegressor
        model = (CatBoostRegressor if task == "regression" else CatBoostClassifier)()
        model.load_model(str(model_path))
        return model, "hit"
    model = fit_catboost(train, y_train, val, y_val, task, seed=seed, iterations=iterations)
    target.mkdir(parents=True, exist_ok=True)
    suffix = uuid.uuid4().hex
    temporary_model = target / f"catboost.{suffix}.tmp"
    temporary_manifest = target / f"manifest.{suffix}.tmp"
    model.save_model(str(temporary_model))
    manifest = dict(identity=identity, recipe=recipe,
                    model_sha256=file_digest(temporary_model))
    temporary_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temporary_model, model_path)
    os.replace(temporary_manifest, manifest_path)
    return model, "miss"
