import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from reading_agent.budget_report import report as budget_report
from reading_agent.compare import load_results
from reading_agent.paper_assets import (CLS, budget_figure, budget_table, macros, main_table, paired,
                                        per_dataset, per_dataset_table, wide_scores)
from reading_agent.summarize import budget_comparison

LETTERS = "ABCDEFG"


def synthetic(temp):
    """7 datasets per task family, 2 folds; the blend beats CatBoost Frozen everywhere by 0.01."""
    rows, lb = [], []
    for prefix, task in (("BIN", "binary"), ("REG", "regression")):
        for i, letter in enumerate(LETTERS):
            dataset = f"{prefix}_D{letter}"
            for fold in (0, 1):
                base = dict(task=task, dataset=dataset, fold=fold, run_id=f"{dataset}{fold}", protocol="paper", tag="t")
                cb = .7 + .01 * i + .001 * fold
                for method, score in (("catboost_frozen", cb), ("catboost_tab_only", cb - .1), ("agent_table_only", cb - .2),
                                      ("agent_read_all", cb - .005), ("blend_catboost_agent", cb + .01)):
                    rows.append(dict(base, method=method, score=score))
                for key in ("content", "meta"):
                    for budget in (.1, .25):
                        rows.append(dict(base, method=f"gainB_{key}", score=cb - .05, budget=budget, frac=budget))
                        rows.append(dict(base, method=f"randomB_{key}", score=cb - .08, budget=budget, frac=budget))
                rows.append(dict(base, method="_done", score=0))
                for model in ("CatBoost", "TabM"):
                    lb.append(dict(dataset=dataset, fold=fold, model=model, cond="frozen", test_score=cb - .002))
                    lb.append(dict(dataset=dataset, fold=fold, model=model, cond="tar", test_score=cb + .02))
    path = Path(temp) / "t.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return load_results([path]), pd.DataFrame(lb)


class PaperAssetsTests(unittest.TestCase):
    def test_tables_macros_and_figure(self):
        with tempfile.TemporaryDirectory() as temp:
            df, lb = synthetic(temp)
            wide = wide_scores(df, lb)
            means, stds, counts = per_dataset(wide)
            self.assertTrue((counts == 2).all())
            # folds are averaged within a dataset: n is datasets, not dataset-folds
            stats = paired(wide, "blend_catboost_agent", "catboost_frozen", CLS)
            self.assertEqual((stats["n_datasets"], stats["wins"]), (7, 7))
            self.assertLess(stats["pvalue"], .05)
            self.assertEqual(paired(wide, "blend_catboost_agent", "REF_CatBoost_tar", CLS)["wins"], 0)

            table = main_table(wide, means)
            self.assertIn("CatBoost, Joint TAR", table)
            self.assertIn("7/0/0$^{*}$", table)  # local baseline: significance mark
            self.assertIn("0/0/7 \\\\", table)  # external reference: no mark
            self.assertNotIn("0/0/7$^{*}$", table)
            self.assertIn("--", table)  # TabPFN references are absent

            brows = budget_report(df, n_resamples=50)
            budget = budget_table(brows)
            self.assertIn("+0.030", budget)
            self.assertIn("7/0/0", budget)

            text = macros(df, wide, means, counts, brows)
            self.assertIn("\\newcommand{\\nFolds}{2}", text)
            self.assertIn("\\newcommand{\\blendVsLocalCBClsDelta}{+0.010}", text)
            self.assertIn("\\newcommand{\\sBinDaReadAll}{", text)

            self.assertIn("\\pm", per_dataset_table(means, stds, counts))
            out = Path(temp) / "fig.pdf"
            budget_figure(budget_comparison(df), means, out)
            self.assertGreater(out.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
