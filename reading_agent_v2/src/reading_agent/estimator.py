"""Generic DataFrame fit/predict and persisted inference for the reading agent.

Artifacts contain trained reader weights, fitted feature transforms, label/target
mapping, thresholds, and optionally the whole-field CatBoost baseline and blend.
They never contain training labels or require a benchmark dataset download.
"""
import argparse
import json
import pickle
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from .baselines import cb_predict, frozen_frame
from .embed import MODEL_NAME, release_model
from .features import FeaturePipeline, whole_text_embeddings
from .train import Config, build_model, predict as run_predict, train_policy_gain, train_warmup


class ReadingEstimator:
    """A reader for a declared numeric/categorical/text schema.

    ``fit`` trains one reader and gain head on a generic DataFrame. ``from_trained``
    wraps an existing experiment without retraining. Predictions preserve the raw
    regression scale or original class labels; predict_proba uses classes_ order.
    Budget selection is a cached-evidence simulation, not deferred E5 encoding.
    """

    def __init__(self, num_cols, cat_cols, text_cols, *, task="binary", config=None,
                 namespace="generic", max_items=8, field_meta=False, key_mode="meta"):
        if task not in {"binary", "multiclass", "regression"}:
            raise ValueError("task must be binary, multiclass or regression")
        if key_mode not in {"meta", "content"}:
            raise ValueError("key_mode must be meta or content")
        self.config = config or Config()
        self.task, self.key_mode = task, key_mode
        cols = list(num_cols) + list(cat_cols) + list(text_cols)
        if len(set(cols)) != len(cols):
            raise ValueError("Each column must occur in exactly one schema group")
        if max_items < 1:
            raise ValueError("max_items must be positive")
        self.pipe = FeaturePipeline(list(num_cols), list(cat_cols), list(text_cols), namespace,
                                    max_items=max(max_items, len(text_cols)), field_meta=field_meta)
        self.thresholds = {}
        self.classes_ = None
        self.target_mean, self.target_scale = 0.0, 1.0
        self.models = []
        self.catboost = None
        self.pcas = None
        self.blend_weight = None
        self.metadata = {}

    def _validate_frame(self, frame):
        if not isinstance(frame, pd.DataFrame) or frame.columns.has_duplicates:
            raise ValueError("Input must be a DataFrame with unique column names")
        required = self.pipe.num_cols + self.pipe.cat_cols + self.pipe.text_cols
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

    def fit(self, frame, y, *, val_fraction=0.2, train_gain=True, log=print):
        self._validate_frame(frame)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(frame) or len(y) < 4:
            raise ValueError("A one-dimensional target with at least four matching rows is required")
        if not 0 < val_fraction < 1:
            raise ValueError("val_fraction must be between zero and one")
        if pd.isna(y).any():
            raise ValueError("Targets cannot be missing")
        if self.task != "regression":
            self.classes_, encoded = np.unique(y, return_inverse=True)
            if len(self.classes_) < 2 or (self.task == "binary" and len(self.classes_) != 2):
                raise ValueError("Classification target does not match the declared task")
            n_out = len(self.classes_)
            strat = encoded if np.bincount(encoded).min() >= 2 else None
        else:
            encoded = y.astype(float)
            if not np.isfinite(encoded).all():
                raise ValueError("Regression targets must be finite")
            n_out, strat = 1, None
        inner, val = train_test_split(np.arange(len(y)), test_size=val_fraction,
                                      random_state=self.config.seed, stratify=strat)
        if self.task == "regression":
            self.target_mean = float(encoded[inner].mean())
            self.target_scale = float(encoded[inner].std()) + 1e-9
            encoded = (encoded - self.target_mean) / self.target_scale
        # Fitted numeric/category transforms see predictor-training rows only.
        self.pipe.fit(frame.iloc[inner])
        prepared = self.pipe.transform(frame, encoded)
        release_model()
        torch.manual_seed(self.config.seed)
        model = build_model(self.pipe, n_out, self.config)
        train_warmup(model, prepared.subset(inner), prepared.subset(val), self.task, self.config, log)
        model.key_mode = self.key_mode
        if train_gain:
            train_policy_gain(model, prepared.subset(val), self.task, self.config, log=log)
        self.models = [model.eval()]
        self.default_mode = "gain" if train_gain else "all"
        self.metadata = {"training_rows": len(inner), "policy_training_rows": len(val) if train_gain else 0,
                         "validation_roles": "early stopping and policy fit", "encoder": MODEL_NAME}
        return self

    @classmethod
    def from_trained(cls, pipe, model, config, task, *, classes=None, target_mean=0.0,
                     target_scale=1.0, thresholds=None, metadata=None, ensemble_models=None):
        obj = cls(pipe.num_cols, pipe.cat_cols, pipe.text_cols, task=task, config=config,
                  namespace=pipe.namespace, max_items=pipe.max_items, key_mode=model.key_mode)
        obj.pipe = pipe
        obj.models = [model] + list(ensemble_models or [])
        obj.classes_ = None if classes is None else np.asarray(classes)
        obj.target_mean, obj.target_scale = float(target_mean), float(target_scale)
        obj.thresholds = {str(k): float(v) for k, v in (thresholds or {}).items()}
        obj.metadata = metadata or {}
        obj.default_mode = "all"
        return obj

    def with_catboost(self, model, pcas, blend_weight):
        if not 0 <= blend_weight <= 1:
            raise ValueError("Blend weight must lie between zero and one")
        self.catboost, self.pcas, self.blend_weight = model, pcas, float(blend_weight)
        return self

    def predict_outputs(self, frame, *, mode=None, lam=None, budget=None):
        """Return probabilities or raw-scale regression values, plus per-row reads.

        ``budget`` names a saved calibrated gain threshold, e.g. 0.1, not a hard
        per-row cap. ``all``/``blend`` average any saved reader ensemble; ``gain``
        uses the first reader, matching the experiment runner's policy semantics.
        """
        self._validate_frame(frame)
        if not self.models:
            raise RuntimeError("Call fit or load before prediction")
        mode = mode or self.default_mode
        if mode not in {"none", "all", "gain", "blend", "catboost"}:
            raise ValueError("Unsupported inference mode")
        if budget is not None:
            if mode != "gain" or lam is not None:
                raise ValueError("Use a calibrated budget only with gain mode and without lam")
            key = f"{float(budget):g}"
            if key not in self.thresholds:
                raise ValueError(f"Budget {key} is not calibrated in this artifact")
            lam = self.thresholds[key]
        if not len(frame):
            shape = (0,) if self.task == "regression" else (0, len(self.classes_))
            return np.empty(shape, dtype=np.float32), np.empty(0, dtype=np.int64)
        prepared = self.pipe.transform(frame)
        preds, reads = [], None
        if mode != "catboost":
            active_models = self.models if mode in {"all", "blend"} else self.models[:1]
            for model in active_models:
                raw, reads = run_predict(model, prepared, "all" if mode == "blend" else mode,
                                         lam=0.0 if lam is None else lam)
                preds.append(raw.reshape(-1).numpy() if self.task == "regression"
                             else torch.softmax(raw, 1).numpy())
            outputs = np.mean(preds, axis=0)
        if mode in {"blend", "catboost"}:
            if self.catboost is None:
                raise ValueError("Artifact contains no CatBoost baseline")
            emb = whole_text_embeddings(frame, self.pipe.text_cols, self.pipe.namespace)
            features, _ = frozen_frame(frame, self.pipe.num_cols, self.pipe.cat_cols, emb, self.pcas)
            cb = cb_predict(self.catboost, features, self.task)
            outputs = cb if mode == "catboost" else self.blend_weight * outputs + (1 - self.blend_weight) * cb
            reads = torch.as_tensor(prepared.item_mask.sum(1), dtype=torch.int64)
        if self.task == "regression":
            outputs = outputs * self.target_scale + self.target_mean
        return outputs, reads.numpy()

    def predict_proba(self, frame, **kwargs):
        if self.task == "regression":
            raise ValueError("Regression has no class probabilities")
        return self.predict_outputs(frame, **kwargs)[0]

    def predict(self, frame, **kwargs):
        outputs, _ = self.predict_outputs(frame, **kwargs)
        return outputs if self.task == "regression" else self.classes_[outputs.argmax(1)]

    def save(self, directory):
        if not self.models:
            raise RuntimeError("Cannot save an unfitted estimator")
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        manifest = {"format_version": 1, "task": self.task, "config": asdict(self.config),
                    "key_mode": self.key_mode, "classes": None if self.classes_ is None else self.classes_.tolist(),
                    "target_mean": self.target_mean, "target_scale": self.target_scale,
                    "thresholds": self.thresholds, "default_mode": self.default_mode,
                    "n_models": len(self.models), "n_out": self.models[0].predictor[-1].out_features,
                    "encoder": MODEL_NAME, "metadata": self.metadata,
                    "has_catboost": self.catboost is not None, "blend_weight": self.blend_weight}
        with (path / "preprocessing.pkl").open("wb") as f:
            pickle.dump({"pipe": self.pipe, "pcas": self.pcas}, f, protocol=4)
        for i, model in enumerate(self.models):
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path / f"reader_{i}.pt")
        if self.catboost is not None:
            self.catboost.save_model(str(path / "catboost.cbm"))
        # Manifest last: a directory without this file is not a complete artifact.
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, directory):
        path = Path(directory)
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if manifest["format_version"] != 1:
            raise ValueError("Unsupported artifact format")
        with (path / "preprocessing.pkl").open("rb") as f:
            prep = pickle.load(f)
        pipe, cfg = prep["pipe"], Config(**manifest["config"])
        obj = cls(pipe.num_cols, pipe.cat_cols, pipe.text_cols, task=manifest["task"], config=cfg,
                  namespace=pipe.namespace, max_items=pipe.max_items, key_mode=manifest["key_mode"])
        obj.pipe, obj.pcas = pipe, prep["pcas"]
        obj.classes_ = None if manifest["classes"] is None else np.asarray(manifest["classes"])
        obj.target_mean, obj.target_scale = manifest["target_mean"], manifest["target_scale"]
        obj.thresholds, obj.metadata = manifest["thresholds"], manifest["metadata"]
        obj.default_mode, obj.blend_weight = manifest["default_mode"], manifest["blend_weight"]
        for i in range(manifest["n_models"]):
            model = build_model(pipe, manifest["n_out"], cfg)
            model.load_state_dict(torch.load(path / f"reader_{i}.pt", map_location="cpu", weights_only=True))
            model.key_mode = obj.key_mode
            obj.models.append(model.eval())
        if manifest["has_catboost"]:
            from catboost import CatBoostClassifier, CatBoostRegressor
            obj.catboost = (CatBoostRegressor if obj.task == "regression" else CatBoostClassifier)()
            obj.catboost.load_model(str(path / "catboost.cbm"))
        return obj


def main(argv=None):
    ap = argparse.ArgumentParser(description="Predict CSV/Parquet rows from saved reading-agent weights")
    ap.add_argument("--artifact", help="local artifact directory (or use --repo/--dataset/--fold)")
    ap.add_argument("--repo", help="Hugging Face model repo; downloads one run instead of --artifact")
    ap.add_argument("--dataset")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--part", default="reader", choices=["reader", "gain_content", "gain_meta"])
    ap.add_argument("--revision")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--mode", choices=["all", "none", "gain", "blend", "catboost"])
    ap.add_argument("--lam", type=float)
    ap.add_argument("--budget", type=float)
    args = ap.parse_args(argv)
    if args.repo:
        if not args.dataset:
            ap.error("--repo requires --dataset")
        from .hub import fetch
        args.artifact = fetch(args.repo, args.dataset, args.fold, args.revision) / args.part
    elif not args.artifact:
        ap.error("give --artifact or --repo")
    estimator = ReadingEstimator.load(args.artifact)
    frame = pd.read_parquet(args.input) if Path(args.input).suffix == ".parquet" else pd.read_csv(args.input)
    out, reads = estimator.predict_outputs(frame, mode=args.mode, lam=args.lam, budget=args.budget)
    result = pd.DataFrame({"prediction": out if estimator.task == "regression" else estimator.classes_[out.argmax(1)],
                           "reads": reads})
    if estimator.task != "regression":
        for i, label in enumerate(estimator.classes_):
            result[f"probability_{label}"] = out[:, i]
    result.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
