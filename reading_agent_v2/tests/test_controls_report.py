import unittest

import numpy as np
import pandas as pd

from reading_agent.controls_report import fieldmeta_compare, fieldmeta_latex, fieldmeta_macros, latex, macros, report
from reading_agent.paper_assets import CLS, REG


def synthetic():
    """7 datasets x 2 folds per family: gain > randomB > randomAgg by fixed margins; natural ties random."""
    rows = []
    for task, prefix in ((CLS, "BIN"), (REG, "REG")):
        for i in range(7):
            for fold in (0, 1):
                row = {"dataset": f"{prefix}_{i}", "task_group": task, "fold": fold}
                for key in ("content", "meta"):
                    for b in (0.1, 0.5):
                        base = 0.7 + 0.01 * i + 0.001 * fold
                        row[f"randomAgg_{key}@{b:g}"] = base
                        row[f"naturalAgg_{key}@{b:g}"] = base
                        row[f"randomB_{key}@{b:g}"] = base + 0.01
                        row[f"gainB_{key}@{b:g}"] = base + 0.03
                rows.append(row)
    return pd.DataFrame(rows).set_index(["dataset", "task_group", "fold"])


class ControlsReportTest(unittest.TestCase):
    def test_decomposition_and_holm(self):
        rep = report(synthetic())
        self.assertEqual(len(rep), 2 * 4 * 2 * 2)  # task x comparison x key x budget
        get = lambda name: rep[(rep.comparison == name) & (rep.task == CLS)]
        np.testing.assert_allclose(get("total").mean_delta, 0.03)
        np.testing.assert_allclose(get("choice").mean_delta, 0.02)
        np.testing.assert_allclose(get("alloc").mean_delta, 0.01)
        self.assertTrue((get("total").wins == 7).all())
        self.assertTrue((rep.n_datasets == 7).all())  # folds averaged first, never counted as n
        self.assertTrue((rep.holm_pvalue >= rep.pvalue - 1e-12).all())

    def test_outputs_render(self):
        wide = synthetic()
        rep = report(wide)
        table = latex(rep)
        self.assertIn("+0.030 (7/7)", table)
        self.assertEqual(table.count("\\\\\n"), 2 + 8)
        tex = macros(rep, wide)
        self.assertIn("\\newcommand{\\ctlTotalContentTenRegDelta}{+0.030}", tex)
        self.assertIn("\\newcommand{\\ctlFolds}{2}", tex)

    def test_fieldmeta_compare_matches_folds(self):
        base = synthetic()
        base["agent_table_only"], base["agent_read_all"] = 0.6, 0.8
        fm = base.copy()
        fm["agent_table_only"] = 0.65
        for key in ("content", "meta"):
            for b in (0.1, 0.25):
                base[f"gainB_{key}@{b:g}"] = base[f"randomB_{key}@0.1"] + 0.03
                base[f"randomB_{key}@{b:g}"] = base[f"randomB_{key}@0.1"]
                fm[f"gainB_{key}@{b:g}"] = base[f"randomB_{key}@0.1"] + 0.01
                fm[f"randomB_{key}@{b:g}"] = base[f"randomB_{key}@0.1"]
        fm = fm.drop(index="BIN_0", level="dataset")  # unmatched datasets are ignored
        fmr = fieldmeta_compare(base, fm)
        get = lambda metric, task: fmr[(fmr.metric == metric) & (fmr.task == task)].iloc[0]
        self.assertEqual(get("table", CLS).n_datasets, 6)
        self.assertAlmostEqual(get("table", CLS).mean_delta, 0.05)
        self.assertAlmostEqual(get("adv_content@0.1", REG).mean_delta, -0.02)
        self.assertIn("\\newcommand{\\fmAdvContentTenRegDelta}{$-$0.020}", fieldmeta_macros(fmr))
        self.assertEqual(fieldmeta_latex(fmr).count("\\\\\n"), 2 + 6)  # two header rows + six metrics


if __name__ == "__main__":
    unittest.main()
