import unittest

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from reading_agent2.estimator2 import _tabm_frame
from reading_agent2.paper_assets2 import fmt_signed
from reading_agent2.tabcond import LoRALinear, adapted_state


class TabLoRATest(unittest.TestCase):
    def test_lora_starts_as_the_frozen_layer_and_gate_zero_is_plain_lora(self):
        torch.manual_seed(0)
        base = nn.Linear(6, 5)
        lora = LoRALinear(base, r=4, alpha=8, dropout=0.0)
        x = torch.randn(3, 7, 6)
        self.assertTrue(torch.allclose(lora(x), base(x)))  # B is zero-initialised
        nn.init.normal_(lora.B)
        plain = lora(x)
        lora.gate = torch.ones(3, 4)  # 1 + tanh(0): a zero-initialised gate projection
        self.assertTrue(torch.allclose(lora(x), plain))
        lora.gate = torch.zeros(3, 4)  # a closed gate removes the update for that row
        self.assertTrue(torch.allclose(lora(x), base(x)))

    def test_adapted_state_keeps_reader_lora_and_gate_but_not_frozen_e5(self):
        class Top(nn.Module):
            def __init__(self):
                super().__init__()
                lora = LoRALinear(nn.Linear(4, 4), r=2, alpha=4)
                self.layers = nn.ModuleList([nn.ModuleDict({"q": lora, "norm": nn.LayerNorm(4)})])
                self.lora_params = nn.ModuleList([lora])
                self.gate_proj = nn.Linear(3, 2)

        model = nn.Module()
        model.top, model.head = Top(), nn.Linear(4, 1)
        self.assertEqual(set(adapted_state(model)),
                         {"top.lora_params.0.A", "top.lora_params.0.B", "top.gate_proj.weight", "top.gate_proj.bias",
                          "head.weight", "head.bias"})


class HeadInputTest(unittest.TestCase):
    def test_tabm_frame_flags_training_columns_and_fills_every_numeric_gap(self):
        F = pd.DataFrame({"num_a": [1.0, np.nan], "num_b": [np.nan, 2.0], "cat_c": ["x", "y"]})
        X, cats = _tabm_frame(F, flags=["num_a"], medians={"num_a": 5.0, "num_b": 7.0})
        self.assertEqual(cats, ["cat_c"])
        self.assertEqual(list(X.columns), ["num_a", "num_b", "cat_c", "num_a_missing"])
        self.assertEqual(X.num_a.tolist(), [1.0, 5.0])
        self.assertEqual(X.num_a_missing.tolist(), [0.0, 1.0])
        self.assertEqual(X.num_b.tolist(), [7.0, 2.0])  # gap unseen in training: filled, not flagged


class PaperAssetsTest(unittest.TestCase):
    def test_signed_differences_never_print_negative_zero(self):
        self.assertEqual(fmt_signed(-0.0002), "0.000")
        self.assertEqual(fmt_signed(0.0126), "+0.013")
        self.assertEqual(fmt_signed(-0.0126), "$-$0.013")


if __name__ == "__main__":
    unittest.main()
