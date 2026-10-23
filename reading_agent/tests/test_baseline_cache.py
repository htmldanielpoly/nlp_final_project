import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from reading_agent.baseline_cache import baseline_identity, fit_or_load
from reading_agent.baselines import cb_predict


class BaselineCacheTests(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame({"num_x": np.arange(24, dtype=float)})
        self.val = pd.DataFrame({"num_x": np.arange(24, 32, dtype=float)})
        self.y, self.yv = np.arange(24) % 2, np.arange(8) % 2

    def test_real_model_roundtrip_skips_fit(self):
        with tempfile.TemporaryDirectory() as temp:
            model, status = fit_or_load(self.train, self.y, self.val, self.yv, "binary", temp, iterations=5)
            self.assertEqual(status, "miss")
            with patch("reading_agent.baseline_cache.fit_catboost", side_effect=AssertionError("fit called on cache hit")):
                restored, status = fit_or_load(self.train, self.y, self.val, self.yv, "binary", temp, iterations=5)
            self.assertEqual(status, "hit")
            np.testing.assert_array_equal(cb_predict(model, self.val, "binary"), cb_predict(restored, self.val, "binary"))

    def test_any_training_inputs_or_recipe_change_invalidate(self):
        original = baseline_identity(self.train, self.y, self.val, self.yv, "binary", iterations=5)[0]
        changed = self.train.copy()
        changed.iloc[0, 0] += 0.1
        cases = [(changed, self.y, self.val, self.yv, "binary", 0, 5),
                 (self.train, 1 - self.y, self.val, self.yv, "binary", 0, 5),
                 (self.train, self.y, self.val, 1 - self.yv, "binary", 0, 5),
                 (self.train, self.y, self.val, self.yv, "binary", 1, 5),
                 (self.train, self.y, self.val, self.yv, "binary", 0, 6)]
        for tr, y, va, yv, task, seed, iterations in cases:
            self.assertNotEqual(original, baseline_identity(tr, y, va, yv, task, seed=seed, iterations=iterations)[0])

    def test_corrupt_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            fit_or_load(self.train, self.y, self.val, self.yv, "binary", temp, iterations=5)
            model_path = next(Path(temp).glob("*/catboost.cbm"))
            model_path.write_bytes(b"damaged")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                fit_or_load(self.train, self.y, self.val, self.yv, "binary", temp, iterations=5)

    def test_model_without_manifest_is_not_a_completed_stage(self):
        with tempfile.TemporaryDirectory() as temp:
            fit_or_load(self.train, self.y, self.val, self.yv, "binary", temp, iterations=5)
            next(Path(temp).glob("*/manifest.json")).unlink()
            _, status = fit_or_load(self.train, self.y, self.val, self.yv, "binary", temp, iterations=5)
            self.assertEqual(status, "miss")


if __name__ == "__main__":
    unittest.main()
