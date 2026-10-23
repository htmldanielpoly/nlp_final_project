"""Small deterministic checks for warm-up stopping and gain-head gradients."""
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from reading_agent.features import Prepared
from reading_agent.model import ReadingAgent
from reading_agent.train import Config, eval_full_loss, train_policy_gain, train_warmup


def prepared(n=6, k=3):
    rng = np.random.default_rng(42)
    return Prepared(
        x_num=rng.normal(size=(n, 1)).astype(np.float32),
        m_num=np.zeros((n, 1), np.float32),
        x_cat=np.zeros((n, 0), np.int64),
        items=rng.normal(size=(n, k, 384)).astype(np.float32),
        item_field=np.tile(np.arange(k, dtype=np.int64), (n, 1)),
        item_pos=np.tile(np.arange(k, dtype=np.int64), (n, 1)),
        item_mask=np.ones((n, k), np.float32),
        y=np.arange(n, dtype=np.int64) % 2,
    )


class ScalarModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))


class TrainingTests(unittest.TestCase):
    def scripted_warmup(self, losses, **config):
        """Force noisy validation while actual optimizer steps make checkpoints distinct."""
        model, data = ScalarModel(), prepared(n=2)
        checkpoints = []
        iterator = iter(losses)

        def evaluate(m, *_):
            checkpoints.append(m.weight.detach().clone())
            return next(iterator)

        cfg = Config(epochs=len(losses), batch_size=2, **config)
        with patch("reading_agent.train.prefix_loss", side_effect=lambda m, *_: (m.weight.square(), None)), \
                patch("reading_agent.train.eval_full_loss", side_effect=evaluate):
            best = train_warmup(model, data, data, "binary", cfg, log=lambda *_: None)
        return model, best, checkpoints

    def test_config_invalid_values(self):
        invalid = [dict(epochs=0), dict(epochs=1.5), dict(patience=0), dict(patience=True),
                   dict(min_epochs=-1), dict(min_epochs=61), dict(min_epochs=1.5),
                   dict(early_stop_ema=1), dict(early_stop_ema=-0.1),
                   dict(early_stop_ema=float("nan")), dict(early_stop_ema=float("inf")),
                   dict(batch_size=0), dict(lr=0), dict(full_weight=1.1), dict(sched="invalid")]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Config(**kwargs)

    def test_raw_default_patience_and_checkpoint(self):
        model, best, weights = self.scripted_warmup([1.0, 0.7, 0.8, 0.9, 0.6], patience=2)
        self.assertEqual(len(model.training_history), 4)
        self.assertTrue(model.early_stopped)
        self.assertEqual(model.best_epoch, 1)
        self.assertEqual(best, 0.7)
        torch.testing.assert_close(model.weight, weights[1])
        for row in model.training_history:
            self.assertEqual(row["stop_loss"], row["val_loss"])

    def test_min_epochs_is_completed_epoch_floor(self):
        model, best, _ = self.scripted_warmup([1.0, 1.1, 1.2, 1.3, 0.5],
                                            patience=1, min_epochs=4)
        self.assertEqual(len(model.training_history), 4)
        self.assertEqual(best, 1.0)
        self.assertEqual(model.best_epoch, 0)

    def test_ema_stopping_restores_best_raw_checkpoint(self):
        # EMA keeps improving through epoch 3 even though the raw minimum was epoch 1.
        model, best, weights = self.scripted_warmup([1.0, 0.2, 0.8, 0.6, 1.4, 1.4],
                                                  patience=1, early_stop_ema=0.8)
        self.assertEqual(len(model.training_history), 5)
        self.assertEqual(model.best_epoch, 1)
        self.assertEqual(best, 0.2)
        self.assertEqual(model.best_val_loss, best)
        torch.testing.assert_close(model.weight, weights[1])
        self.assertAlmostEqual(model.training_history[1]["stop_loss"], 0.84)
        self.assertGreater(model.training_history[3]["val_loss"], best)

    def test_small_raw_improvement_can_select_checkpoint_without_resetting_patience(self):
        model, best, weights = self.scripted_warmup([1.0, 0.99995, 1.1], patience=1)
        self.assertEqual(len(model.training_history), 2)
        self.assertEqual(best, 0.99995)
        self.assertEqual(model.best_epoch, 1)
        torch.testing.assert_close(model.weight, weights[1])

    def test_config_mutation_checked_at_train_entry(self):
        cfg = Config()
        cfg.epochs = 0
        with self.assertRaisesRegex(ValueError, "epochs"):
            train_warmup(ScalarModel(), prepared(), prepared(), "binary", cfg, log=lambda *_: None)

    def test_empty_data_and_invalid_validation_batch_size(self):
        with self.assertRaisesRegex(ValueError, "both contain rows"):
            train_warmup(ScalarModel(), prepared(n=0), prepared(), "binary", Config())
        with self.assertRaisesRegex(ValueError, "at least one row"):
            eval_full_loss(ScalarModel(), prepared(n=0), "binary")
        with self.assertRaisesRegex(ValueError, "batch size"):
            eval_full_loss(ScalarModel(), prepared(), "binary", bs=0)

    def test_nonfinite_validation_fails_immediately(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, "non-finite validation"):
                self.scripted_warmup([value], patience=1)

    def test_nonfinite_training_fails_before_validation(self):
        model, data = ScalarModel(), prepared()
        with patch("reading_agent.train.prefix_loss", return_value=(torch.tensor(float("nan")), None)), \
                patch("reading_agent.train.eval_full_loss") as evaluate:
            with self.assertRaisesRegex(RuntimeError, "non-finite training"):
                train_warmup(model, data, data, "binary", Config(epochs=1), log=lambda *_: None)
            evaluate.assert_not_called()

    def test_real_validation_rejects_nan_targets(self):
        data = prepared()
        data.y = np.full(len(data), np.nan, dtype=np.float32)
        model = self.make_reader(n_out=1)
        with self.assertRaisesRegex(RuntimeError, "non-finite per-row"):
            eval_full_loss(model, data, "regression")

    @staticmethod
    def make_reader(n_out=2):
        torch.manual_seed(7)
        rng = np.random.default_rng(7)
        return ReadingAgent(n_num=1, cat_cards=[],
                            col_name_emb=rng.normal(size=(1, 384)).astype(np.float32),
                            field_name_emb=rng.normal(size=(3, 384)).astype(np.float32),
                            n_out=n_out, d=8, layers=1, dropout=0, max_items=3)

    def test_real_synthetic_warmup_records_finite_history(self):
        model = self.make_reader()
        data = prepared()
        result = train_warmup(model, data, data, "binary",
                              Config(epochs=2, min_epochs=2, batch_size=3), log=lambda *_: None)
        self.assertTrue(np.isfinite(result))
        self.assertEqual(len(model.training_history), 2)
        self.assertEqual(result, min(row["val_loss"] for row in model.training_history))
        self.assertEqual(model.best_val_loss, result)
        self.assertFalse(model.training)

    def test_meta_key_receives_nonzero_update_while_reader_is_frozen(self):
        model = self.make_reader()
        model.key_mode = "meta"
        before = {k: p.detach().clone() for k, p in model.named_parameters()}
        train_policy_gain(model, prepared(), "binary",
                          Config(batch_size=3, policy_epochs=2, policy_lr=0.01), log=lambda *_: None)
        delta = (model.meta_key.weight.detach() - before["meta_key.weight"]).abs().sum().item()
        self.assertGreater(delta, 0)
        for name, param in model.named_parameters():
            if not name.startswith(("gain.", "meta_key.")):
                torch.testing.assert_close(param, before[name])
        self.assertTrue(all(p.requires_grad for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
