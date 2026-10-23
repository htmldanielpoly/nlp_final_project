import unittest

import numpy as np

from reading_agent.budget_report import bootstrap_interval, holm_adjust, report
from reading_agent.compare import load_results
from pathlib import Path
import json
import tempfile


class BudgetReportTests(unittest.TestCase):
    def test_holm_order_and_monotonicity(self):
        np.testing.assert_allclose(holm_adjust([.02, .001, .04, np.nan]), [.04, .003, .04, np.nan])
        np.testing.assert_allclose(holm_adjust([.5, .6]), [1, 1])

    def test_bootstrap_constant_and_reproducible(self):
        self.assertEqual(bootstrap_interval([.2, .2], 100), [.2, .2])
        self.assertEqual(bootstrap_interval([0, .1, .2], 100), bootstrap_interval([0, .1, .2], 100))
        self.assertIsNone(bootstrap_interval([1], 100))

    def test_tasks_fold_unit_and_budget_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            rows = []
            for task, dataset in [("binary", "BIN_X"), ("regression", "REG_X")]:
                for fold in [0, 1]:
                    base = dict(task=task,dataset=dataset,fold=fold,run_id=f"{dataset}{fold}",protocol="paper",tag="x")
                    for method, score in [("gainB_meta", .8), ("randomB_meta", .6)]:
                        rows.append(dict(base,method=method,score=score,budget=.1,frac=0))
                    rows.append(dict(base,method="_done",score=0))
            path = Path(temp)/"r.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows))
            output = report(load_results([path]), 100)
            self.assertEqual(len(output), 2)
            for row in output:
                self.assertEqual(row["n_datasets"], 1)
                self.assertEqual(row["datasets_over_5pct_budget_error"], 1)
                self.assertIsNone(row["bootstrap_mean_ci95"])


if __name__ == "__main__":
    unittest.main()
