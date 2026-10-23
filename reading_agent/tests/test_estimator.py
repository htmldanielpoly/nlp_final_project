import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from reading_agent.estimator import ReadingEstimator
from reading_agent.train import Config


def vectors(texts):
    return np.array([np.random.RandomState(int.from_bytes(hashlib.sha256(t.encode()).digest()[:4], "little"))
                     .normal(size=384) for t in texts], dtype=np.float32).reshape(-1, 384)


class EstimatorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.patchers = [patch("reading_agent.features.embed", side_effect=lambda texts, **kwargs: vectors(texts)),
                         patch("reading_agent.features.EmbeddingStore.get", side_effect=vectors)]
        for p in self.patchers:
            p.start()
            self.addCleanup(p.stop)
        self.frame = pd.DataFrame({"number": np.arange(30, dtype=float),
                                   "category": pd.Categorical(["a", "b", "c"] * 10),
                                   "text": [f"example {i}" for i in range(30)]})
        self.cfg = Config(d=8, layers=1, epochs=2, patience=2, policy_epochs=1, batch_size=16)

    def estimator(self, task):
        return ReadingEstimator(["number"], ["category"], ["text"], task=task, config=self.cfg, max_items=2)

    def test_binary_and_multiclass_round_trip(self):
        for task, labels in [("binary", ["no", "yes"] * 15), ("multiclass", ["a", "b", "c"] * 10)]:
            with self.subTest(task=task), tempfile.TemporaryDirectory() as temp:
                fitted = self.estimator(task).fit(self.frame, labels, log=lambda _: None)
                before, reads = fitted.predict_outputs(self.frame, mode="gain", lam=0)
                fitted.save(temp)
                loaded = ReadingEstimator.load(temp)
                after, loaded_reads = loaded.predict_outputs(self.frame, mode="gain", lam=0)
                np.testing.assert_allclose(before, after, rtol=0, atol=0)
                np.testing.assert_array_equal(reads, loaded_reads)
                np.testing.assert_array_equal(fitted.predict(self.frame), loaded.predict(self.frame))
                np.testing.assert_allclose(after.sum(1), 1, atol=1e-6)

    def test_regression_raw_scale_and_round_trip(self):
        fitted = self.estimator("regression").fit(self.frame, 1000 + 100 * np.arange(30), log=lambda _: None)
        with tempfile.TemporaryDirectory() as temp:
            expected = fitted.predict(self.frame, mode="all")
            fitted.save(temp)
            actual = ReadingEstimator.load(temp).predict(self.frame, mode="all")
        np.testing.assert_allclose(expected, actual, rtol=0, atol=0)
        self.assertGreater(abs(float(actual.mean())), 100)

    def test_budget_and_schema_validation(self):
        fitted = self.estimator("binary").fit(self.frame, [0, 1] * 15, log=lambda _: None)
        with self.assertRaisesRegex(ValueError, "not calibrated"):
            fitted.predict_outputs(self.frame, mode="gain", budget=0.1)
        fitted.thresholds["0.1"] = 0.02
        a = fitted.predict_outputs(self.frame, mode="gain", budget=0.1)
        b = fitted.predict_outputs(self.frame, mode="gain", lam=0.02)
        np.testing.assert_array_equal(a[0], b[0])
        with self.assertRaisesRegex(ValueError, "Missing required"):
            fitted.predict(self.frame.drop(columns="number"))
        self.assertEqual(fitted.predict_proba(self.frame.iloc[:0]).shape, (0, 2))

    def test_invalid_target(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            self.estimator("regression").fit(self.frame, [np.inf] * 30)
        with self.assertRaisesRegex(ValueError, "declared task"):
            self.estimator("binary").fit(self.frame, [0, 1, 2] * 10)

    def test_fixed_control_capacity_and_aggregate_budget(self):
        from reading_agent.run import fixed_budget_counts
        capacities = np.array([0, 1, 2, 3, 4])
        mask = np.arange(4)[None, :] < capacities[:, None]
        for target in [0, 0.1, 0.25, 0.5, 0.75, 1]:
            counts = fixed_budget_counts(mask, target)
            self.assertEqual(int(counts.sum()), int(np.floor(target * capacities.sum() + 0.5)))
            self.assertTrue((counts >= 0).all() and (counts <= capacities).all())
        with self.assertRaises(ValueError):
            fixed_budget_counts(mask, 1.1)

    def test_full_experiment_artifacts_and_all_random_orders(self):
        from reading_agent import run
        from reading_agent.baselines import fit_catboost
        from reading_agent.data import Dataset
        target = np.array([0, 1, 2] * 10)
        dataset = Dataset("synthetic", self.frame.copy(), pd.Series(target), "multiclass",
                          ["number"], ["category"], ["text"], source_revision="synthetic-v1")
        with tempfile.TemporaryDirectory() as temp, patch.object(run, "RESULTS", Path(temp)), \
             patch.object(run, "load_dataset", return_value=dataset), \
             patch("reading_agent.baseline_cache.fit_catboost", side_effect=lambda *a, **kw: fit_catboost(*a, **{**kw, "iterations": 5})), \
             patch.object(run.torch, "set_num_threads"):
            rows = run.main(["--dataset", "synthetic", "--protocol", "current", "--tag", "smoke",
                             "--skip", "soft,hybrid", "--epochs", "2", "--patience", "2",
                             "--gain_epochs", "1", "--budgets", "0.25", "--lams=-1", "--controls", "1"])
            self.assertEqual(rows[-1]["method"], "_done")
            from reading_agent.verify_predictions import verify
            verification = verify([Path(temp) / "smoke.jsonl"], Path(temp) / "preds")
            self.assertEqual(verification["mismatches"], [])
            self.assertEqual(verification["incomplete"], [])
            run_id = rows[-1]["run_id"]
            with np.load(Path(temp) / "preds" / "smoke" / (run_id + ".npz")) as saved:
                for row in rows:
                    if row.get("prediction_keys"):
                        scores = [run.score(row["task"], saved["y_te"], saved[k]) for k in row["prediction_keys"]]
                        self.assertAlmostEqual(row["score"], float(np.mean(scores)), places=6)
                raw = saved["blend_catboost_agent"].copy()
                indices = saved["te_idx"].copy()
            artifact = ReadingEstimator.load(Path(temp) / "models" / "smoke" / run_id / "reader")
            restored = artifact.predict_proba(self.frame.iloc[indices], mode="blend")
            np.testing.assert_allclose(raw, restored, atol=2e-7, rtol=0)
            policy = ReadingEstimator.load(Path(temp) / "models" / "smoke" / run_id / "gain_meta")
            self.assertIn("0.25", policy.thresholds)
            sidecar = json.loads((Path(temp) / "preds" / "smoke" / (run_id + ".json")).read_text())
            self.assertEqual(sidecar["provenance"]["data_revision"], "synthetic-v1")

    def test_regression_runner_preserves_ensemble_scale(self):
        from reading_agent import run
        from reading_agent.data import Dataset
        raw_y = 1000 + 10 * np.arange(30)
        ds = Dataset("synthetic", self.frame.copy(), pd.Series(raw_y), "regression",
                     ["number"], ["category"], ["text"])
        with tempfile.TemporaryDirectory() as temp, patch.object(run, "RESULTS", Path(temp)), \
             patch.object(run, "load_dataset", return_value=ds), patch.object(run.torch, "set_num_threads"):
            rows = run.main(["--dataset", "synthetic", "--tag", "regression", "--seeds", "2",
                             "--skip", "catboost,soft,hybrid,policy", "--epochs", "1"])
            run_id = rows[-1]["run_id"]
            prefix = Path(temp) / "preds" / "regression" / run_id
            metadata = json.loads(prefix.with_suffix(".json").read_text())
            with np.load(prefix.with_suffix(".npz")) as archive:
                expected = archive["agent_read_all_ens2"] * metadata["target_scale"] + metadata["target_mean"]
                indices = archive["te_idx"].copy()
            loaded = ReadingEstimator.load(Path(temp) / "models" / "regression" / run_id / "reader")
            actual = loaded.predict(self.frame.iloc[indices], mode="all")
            np.testing.assert_allclose(expected, actual, atol=1e-4, rtol=0)
            self.assertGreater(metadata["target_scale"], 2)


if __name__ == "__main__":
    unittest.main()
