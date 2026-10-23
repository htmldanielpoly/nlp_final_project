"""Analysis correctness tests; synthetic scores only, no models/data downloads."""
import contextlib
import importlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from reading_agent.compare import (GROUP, compare_group, load_results,
                                   method_scores, paired_dataset_deltas,
                                   select_results, summarize_deltas)
from reading_agent.eval_paper import BUGGY_DATASET, evaluation_results
from reading_agent.summarize import budget_comparison, summarize_group


def row(run="r1", dataset="BIN_TEXT_TEST", fold=0, method="agent_read_all", score=.8,
        protocol="paper", tag="test", **extra):
    return dict(dataset=dataset, fold=fold, method=method, score=score,
                protocol=protocol, tag=tag, run_id=run, task="binary", **extra)


def done(run):
    return {"method": "_done", "run_id": run}


class AnalysisTests(unittest.TestCase):
    def test_floating_point_noise_is_a_tie(self):
        from reading_agent.compare import summarize_deltas
        result = summarize_deltas(pd.DataFrame({"delta": [1.11e-16, -1e-12, 0.01]}))
        self.assertEqual(result["wins"], 1)
        self.assertEqual(result["ties"], 2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, name, rows):
        path = self.root / name
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        return path

    def test_completion_is_file_scoped_and_empty_partial_run_is_safe(self):
        partial = self.write("partial.jsonl", [row("collision")])
        complete = self.write("complete.jsonl", [row("collision"), done("collision")])
        df = load_results([partial, complete])
        self.assertEqual(len(df), 1)
        self.assertEqual(df.attrs["excluded_partial_rows"], 1)
        self.assertEqual(df.source_path.iloc[0], str(complete.resolve()))
        empty = load_results([partial])
        self.assertTrue(empty.empty)
        self.assertEqual(empty.attrs["excluded_partial_rows"], 1)
        self.assertTrue(method_scores(empty).empty)

    def test_legacy_rows_and_budget_naming_are_retained(self):
        legacy = row(method="gainB_meta", budget=.1, lam=.023)
        legacy.pop("run_id")
        legacy.pop("protocol")
        df = load_results([self.write("legacy.jsonl", [legacy])])
        self.assertEqual(df.method.iloc[0], "gainB_meta@0.1")
        self.assertEqual(df.protocol.iloc[0], "current")
        self.assertEqual(df.completion.iloc[0], "legacy-unverified")

    def test_different_configs_and_protocols_are_never_silently_averaged(self):
        rows = [row("a", args={"field_meta": 0}), done("a"),
                row("b", args={"field_meta": 1}), done("b"),
                row("c", args={"field_meta": 1}, protocol="current"), done("c")]
        df = load_results([self.write("mixed.jsonl", rows)])
        self.assertEqual(len(df[GROUP].drop_duplicates()), 3)
        with self.assertRaisesRegex(ValueError, "one protocol/configuration"):
            method_scores(df)
        for _, group in df.groupby(GROUP):
            self.assertEqual(len(method_scores(group)), 1)

    def test_repeated_runs_require_selection(self):
        rows = [row("a", tag="old"), done("a"), row("b", tag="new", score=.9), done("b")]
        df = load_results([self.write("repeat.jsonl", rows)])
        with self.assertRaisesRegex(ValueError, "multiple score rows"):
            method_scores(df)
        chosen = method_scores(select_results(df, tag="new"))
        self.assertEqual(chosen.score.iloc[0], .9)
        self.assertEqual(len(select_results(df, run_id="a")), 1)

    def test_dataset_is_statistical_unit_and_finite_pairs_only(self):
        pairs = pd.DataFrame({"dataset": ["a", "a", "b", "b", "c"],
                              "task_group": ["classification (AUROC)"] * 5,
                              "fold": [0, 1, 0, 1, 0], "score": [.8, .6, .7, float("nan"), float("inf")],
                              "test_score": [.5, .5, .6, .5, .5]})
        deltas = paired_dataset_deltas(pairs)
        self.assertEqual(len(deltas), 2)
        self.assertAlmostEqual(deltas.set_index("dataset").loc["a", "delta"], .2)
        self.assertEqual(deltas.set_index("dataset").loc["a", "folds"], 2)
        stats = summarize_deltas(deltas)
        self.assertEqual(stats["n_datasets"], 2)
        self.assertAlmostEqual(stats["mean_delta"], .15)
        self.assertTrue(math.isnan(stats["pvalue"]))

    def test_zero_and_nan_differences_are_safe(self):
        zeros = pd.DataFrame({"delta": [0.] * 6 + [float("nan")]})
        stats = summarize_deltas(zeros)
        self.assertEqual(stats["n_datasets"], 6)
        self.assertEqual(stats["pvalue"], 1.)
        self.assertEqual(stats["ties"], 6)
        empty = summarize_deltas(pd.DataFrame({"delta": [float("nan"), float("inf")]}))
        self.assertEqual(empty["n_datasets"], 0)
        self.assertTrue(math.isnan(empty["mean_delta"]))

    def test_mixed_metrics_cannot_be_pooled(self):
        mixed = pd.DataFrame({"delta": [.1, .2], "task_group": ["classification (AUROC)", "regression (R2)"]})
        with self.assertRaisesRegex(ValueError, "separately"):
            summarize_deltas(mixed)

    def test_external_current_protocol_pvalues_are_suppressed(self):
        rows, refs = [], []
        for i in range(6):
            dataset, run = f"BIN_TEXT_TEST{i}", f"r{i}"
            rows += [row(run, dataset=dataset, protocol="current"), done(run)]
            refs.append(dict(dataset=dataset, fold=0, model="CatBoost", cond="frozen", test_score=.7))
        df = load_results([self.write("current.jsonl", rows)])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            compare_group(df, pd.DataFrame(refs), ["agent_read_all"])
        self.assertIn("p-values suppressed", output.getvalue())
        self.assertIn("p=not reported", output.getvalue())
        self.assertIn("datasets=6", output.getvalue())
        self.assertIn("REF_CatBoost_frozen", output.getvalue())

    def test_budget_pairing_needs_same_run_and_handles_missing_frac(self):
        rows = [row("a", method="gainB_meta", budget=.1, score=.8), done("a"),
                row("b", method="randomB_meta", budget=.1, score=.7), done("b")]
        unpaired = budget_comparison(load_results([self.write("unpaired.jsonl", rows)]))
        self.assertTrue(unpaired.adv.isna().all())
        rows = [row("a", method="gainB_meta", budget=.1, score=.8),
                row("a", method="randomB_meta", budget=.1, score=.7), done("a")]
        paired = budget_comparison(load_results([self.write("paired.jsonl", rows)]))
        self.assertAlmostEqual(paired.adv.iloc[0], .1)
        self.assertTrue(math.isnan(paired.frac.iloc[0]))

    def test_partial_budget_and_by_field_summary_does_not_crash(self):
        rows = [row("a", method="gainB_content", budget=.1, field_meta=1), done("a")]
        part = load_results([self.write("budget_only.jsonl", rows)])
        with contextlib.redirect_stdout(io.StringIO()):
            acc, budgets = summarize_group(part, by="field_meta")
        self.assertTrue(acc.empty)
        self.assertTrue(budgets.random.isna().all())

    def test_historical_default_replaces_only_buggy_rt_explicit_paths_do_not(self):
        self.write("paper_f0.jsonl", [row("old", dataset=BUGGY_DATASET, tag="paper_f0"), done("old"),
                                      row("other", dataset="BIN_TEXT_OTHER", tag="paper_f0"), done("other")])
        cap = self.write("paper_f0_cap.jsonl", [row("fixed", dataset=BUGGY_DATASET, tag="paper_f0_cap", score=.9), done("fixed")])
        default = evaluation_results(results_dir=self.root)
        self.assertEqual(len(default), 2)
        self.assertEqual(default[default.dataset == BUGGY_DATASET].run_id.tolist(), ["fixed"])
        explicit = evaluation_results([self.root / "paper_f0.jsonl"])
        self.assertEqual(explicit[explicit.dataset == BUGGY_DATASET].run_id.tolist(), ["old"])
        self.assertEqual(evaluation_results(tag="paper_f0_cap", results_dir=self.root).run_id.tolist(), ["fixed"])

    def test_eval_module_import_has_no_result_file_side_effects(self):
        import reading_agent.eval_paper as module
        with patch("builtins.open", side_effect=AssertionError("unexpected file access")):
            importlib.reload(module)


if __name__ == "__main__":
    unittest.main()
