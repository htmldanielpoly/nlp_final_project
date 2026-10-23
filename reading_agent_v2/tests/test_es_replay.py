"""Offline replay of warm-up stopping rules."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from reading_agent.es_replay import RULES, cli, longest_stall, main, replay
from reading_agent.train import Config, train_warmup


class ReplayTest(unittest.TestCase):
    def test_patience_stops_after_stall_and_keeps_raw_best(self):
        val = [1.0, 0.8, 0.7] + [0.75] * 10 + [0.5]
        self.assertEqual(replay(val, 60, 6, 0.0, 0), (9, 0.7, 2))
        self.assertEqual(replay(val, 60, 12, 0.0, 0), (14, 0.5, 13))

    def test_min_epochs_delays_stop(self):
        self.assertEqual(replay([1.0] + [1.1] * 40, 60, 6, 0.0, 30)[0], 30)

    def test_replay_matches_training_for_all_nine_rules(self):
        # Include delayed recovery, sub-threshold raw records, a noisy stretch and a cap.
        val = ([1.0, 0.9, 0.89996, 0.89992, 0.89988] + [0.91] * 9
               + [0.8, 0.82, 0.79, 0.81] + [0.84] * 25
               + [0.7, 0.72, 0.71] + [0.73] * 65)
        for name, epochs, patience, ema, min_epochs in RULES:
            with self.subTest(rule=name):
                model = nn.Linear(1, 1)
                cfg = Config(epochs=epochs, patience=patience, early_stop_ema=ema,
                             min_epochs=min_epochs, batch_size=1)
                data = [None]
                batch = {"item_mask": torch.ones(1, 1), "y": torch.zeros(1)}
                with patch("reading_agent.train.to_batch", return_value=batch), \
                        patch("reading_agent.train.prefix_loss",
                              side_effect=lambda m, *_: (m.weight.square().mean(), None)), \
                        patch("reading_agent.train.eval_full_loss", side_effect=val):
                    actual = train_warmup(model, data, data, "binary", cfg, log=lambda *_: None)
                self.assertEqual(replay(val, epochs, patience, ema, min_epochs),
                                 (len(model.training_history), actual, model.best_epoch))

    def test_plateau_extra_patience_matches_training(self):
        val = [1.0] + [1.1] * 20
        model, cfg = nn.Linear(1, 1), Config(epochs=20, patience=2, sched="plateau")
        with patch("reading_agent.train.to_batch",
                   return_value={"item_mask": torch.ones(1, 1), "y": torch.zeros(1)}), \
                patch("reading_agent.train.prefix_loss",
                      side_effect=lambda m, *_: (m.weight.square().mean(), None)), \
                patch("reading_agent.train.eval_full_loss", side_effect=val):
            best = train_warmup(model, [None], [None], "binary", cfg, log=lambda *_: None)
        self.assertEqual(replay(val, 20, 2, 0.0, 0, sched="plateau"),
                         (len(model.training_history), best, model.best_epoch))
        self.assertEqual(len(model.training_history), 7)

    def test_invalid_curve_and_rule_parameters(self):
        for val in ([], [float("nan")], [float("inf")], [-float("inf")], ["0.4"], [True]):
            with self.subTest(curve=val), self.assertRaises(ValueError):
                replay(val, 60, 6, 0.0, 0)
        invalid = [(0, 6, 0, 0), (60, 0, 0, 0), (60, True, 0, 0),
                   (60, 6, 1, 0), (60, 6, float("nan"), 0),
                   (60, 6, 0, -1), (60, 6, 0, 61), (1.5, 6, 0, 0)]
        for rule in invalid:
            with self.subTest(rule=rule), self.assertRaises(ValueError):
                replay([1.0], *rule)
        with self.assertRaises(ValueError):
            replay([1.0], 60, 6, 0, 0, sched="bad")

    def test_best_raw_updates_without_patience_reset(self):
        self.assertEqual(replay([1.0, 0.99996, 0.5], 3, 1, 0.0, 0), (2, 0.99996, 1))

    def test_longest_stall_uses_threshold_baseline_not_exact_raw_minimum(self):
        val = [1.0, 0.99996, 0.99992, 0.99988, 0.99984]
        # Every value is a new exact raw minimum, but two epochs precede a >1e-4 reset.
        self.assertEqual(longest_stall(val, 4), 2)
        self.assertEqual(longest_stall(val, 1), 1)
        for upto in (-1, len(val), 1.2, True):
            with self.subTest(upto=upto), self.assertRaises(ValueError):
                longest_stall(val, upto)

    @staticmethod
    def write_history(path, val, envelope=None):
        rows = [{"epoch": ep, "val_loss": loss} for ep, loss in enumerate(val)]
        path.write_text(json.dumps(rows if envelope is None else {"history": rows, **envelope}),
                        encoding="utf-8")

    def test_main_flags_unknown_and_truncated_curves_without_claiming_global_best(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "run.json"
            self.write_history(path, [1.0, 0.8])
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                main([path])
            output = buf.getvalue()
            self.assertIn("best observed val", output)
            self.assertIn("source metadata: unavailable", output)
            self.assertIn("early_stopped=unknown", output)
            self.assertIn("source coverage: unknown", output)
            self.assertIn("truncated: needs up to 60 epochs", output)
            self.assertIn("not a global best", output)
            self.assertNotIn("oracle best", output)

    def test_sidecar_metadata_and_source_truncation_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "run.json"
            self.write_history(path, [1.0, 0.8])
            sidecar = path.with_suffix(".meta.json")
            sidecar.write_text(json.dumps({"config": {"epochs": 100, "sched": "none"},
                                           "early_stopped": True}), encoding="utf-8")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                main(path)  # A single path also works programmatically.
            output = buf.getvalue()
            self.assertIn(str(sidecar), output)
            self.assertIn("early_stopped=True", output)
            self.assertIn("truncated before configured epoch cap (2/100)", output)

    def test_model_manifest_metadata_is_discovered(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "history" / "test_tag" / "run.json"
            path.parent.mkdir(parents=True)
            self.write_history(path, [1.0, 0.8])
            manifest = root / "models" / "test_tag" / "run" / "gain_content" / "manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({"config": {"epochs": 2, "sched": "none"},
                                            "metadata": {"early_stopped": False}}), encoding="utf-8")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                main([path])
            self.assertIn(str(manifest), buf.getvalue())
            self.assertIn("reached configured epoch cap (2/2)", buf.getvalue())
            self.assertIn("early_stopped=False", buf.getvalue())

    def test_cli_expands_globs_and_rejects_missing_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "curve.json"
            self.write_history(path, [1.0, 0.8], envelope={"config": {"epochs": 2}})
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                cli([str(Path(folder) / "*.json")])
            self.assertIn("curve.json", buf.getvalue())
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                cli([str(Path(folder) / "missing*.json")])
            self.assertEqual(caught.exception.code, 2)

    def test_main_rejects_empty_or_malformed_histories(self):
        with self.assertRaises(ValueError):
            main([])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.json"
            for payload in ([], {}, [{"val_loss": float("nan")}],
                            [{"epoch": 1, "val_loss": 1.0}], [{"not_loss": 1.0}]):
                with self.subTest(payload=payload):
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        main([path])


if __name__ == "__main__":
    unittest.main()
