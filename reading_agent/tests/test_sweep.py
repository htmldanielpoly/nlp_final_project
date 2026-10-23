"""Resume/planning tests without embedding loads, data downloads or training."""
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reading_agent import sweep

PROJECT = sweep.ROOT
A = "MUL_TEXT_PRODUCT_SENTIMENT"
B = "REG_TEXT_BOOK_PRICE"
SPOTIFY = "MUL_TEXT_SPOTIFY_GENRES"
FLAGS = ["--protocol", "paper", "--epochs", "100", "--patience", "15", "--skip", "soft,hybrid", "--lams=-1"]


class SweepTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.requested = sweep.run_arguments(FLAGS)

    def cache_file(self, name):
        path = self.cache / f"e5_{name}.pkl"
        path.write_bytes(b"not a pickle; only its presence is checked")
        return path

    def record(self, name=A, fold=0, method="_done", run="r", tag="test", **updates):
        args = dict(self.requested, dataset=name, fold=fold, tag=tag, save_models=1)
        args.update(updates)
        return {"dataset": name, "fold": fold, "tag": tag, "run_id": run,
                "protocol": args["protocol"], "args": args, "method": method,
                "provenance": {"data_source": "huggingface", "data_revision": "sha",
                               "train_fingerprint": "train", "test_fingerprint": "test"},
                "implementation_version": "test"}

    def results(self, rows, tag="test", tail=""):
        directory = self.root / "results"
        directory.mkdir(exist_ok=True)
        path = directory / f"{tag}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows) + tail, encoding="utf-8")
        return path

    def plan(self, names=(A,), folds=(0,), last=()):
        return sweep.build_plan("test", names, folds, self.requested, last,
                                root=self.root, cache_dir=self.cache)

    def test_partial_does_not_satisfy_completion(self):
        self.cache_file(A)
        self.results([self.record(method="agent_read_all")])
        plan = self.plan()
        self.assertEqual(plan.complete, [])
        self.assertEqual(plan.pending, [(A, 0)])
        self.assertIn((A, 0), plan.partial)
        self.assertEqual(plan.errors, [])

    def test_resume_requires_matching_recipe_not_only_dataset_fold(self):
        self.results([self.record(epochs=60)])
        plan = self.plan()
        self.assertEqual(plan.complete, [])
        self.assertTrue(any("epochs" in error for error in plan.errors))
        with patch.object(sweep.subprocess, "call") as call:
            self.assertEqual(sweep.execute_plan(plan, "test", FLAGS, self.requested, self.root), 2)
        call.assert_not_called()
        self.assertFalse((self.root / "logs").exists())

    def test_unrequested_fold_with_different_recipe_blocks_tag_mix(self):
        self.cache_file(A)
        self.results([self.record(fold=0, patience=6)])
        plan = self.plan(folds=(1, 2))
        self.assertTrue(plan.errors)
        self.assertEqual(plan.pending, [(A, 1), (A, 2)])

    def test_matching_complete_does_not_require_cache_to_skip(self):
        self.results([self.record()])
        plan = self.plan()
        self.assertEqual(plan.complete, [(A, 0)])
        self.assertEqual(plan.pending, [])
        self.assertEqual(plan.missing_caches, [])

    def test_diagnostic_markers_and_warmup_arguments_are_not_full_done(self):
        for marker in ("warmup_only_done", "_warmup_only_done", "_done"):
            with self.subTest(marker=marker):
                self.results([self.record(method=marker, warmup_only=1)])
                plan = self.plan()
                self.assertEqual(plan.complete, [])
                self.assertTrue(any("warmup_only" in error for error in plan.errors))
        self.results([self.record(method="warmup_only_done")])
        self.assertEqual(self.plan().complete, [])

    def test_missing_recipe_cannot_be_assumed_to_match(self):
        record = self.record()
        record.pop("args")
        self.results([record])
        self.assertTrue(any("recipe missing" in error for error in self.plan().errors))

    def test_protocol_and_archive_tag_consistency_are_guarded(self):
        record = self.record()
        record["protocol"] = "current"
        record["tag"] = "other-tag"
        self.results([record])
        plan = self.plan()
        errors = plan.errors
        self.assertEqual(plan.complete, [])
        self.assertTrue(any("protocol" in error for error in errors))
        self.assertTrue(any("tag" in error for error in errors))

    def test_operational_flags_ignored_set_order_canonicalized(self):
        self.results([self.record(save_models=0, baseline_cache=0, skip="hybrid,soft", key_modes="meta,content")])
        plan = self.plan()
        self.assertEqual(plan.complete, [(A, 0)])
        self.assertEqual(plan.errors, [])

    def test_missing_provenance_warns_instead_of_claiming_identity(self):
        record = self.record()
        record.pop("provenance")
        self.results([record])
        plan = self.plan()
        self.assertEqual(plan.complete, [(A, 0)])
        self.assertTrue(any("provenance" in warning for warning in plan.warnings))

    def test_only_known_historical_zero_defaults_are_compatible(self):
        record = self.record()
        record["args"].pop("threads")
        record["args"].pop("deterministic")
        self.results([record])
        plan = self.plan()
        self.assertEqual(plan.complete, [(A, 0)])
        self.assertEqual(plan.errors, [])
        self.assertTrue(any("zero defaults" in warning for warning in plan.warnings))
        for flag in ("threads", "deterministic"):
            with self.subTest(flag=flag):
                requested = dict(self.requested, **{flag: 1})
                blocked = sweep.build_plan("test", [A], [0], requested,
                                           root=self.root, cache_dir=self.cache)
                self.assertEqual(blocked.complete, [])
                self.assertTrue(any(flag in error for error in blocked.errors))
        record["args"].pop("epochs")
        self.results([record])
        self.assertTrue(any("epochs=unrecorded" in error for error in self.plan().errors))

    def test_spotify_last_in_each_fold_and_missing_cache_explicit(self):
        self.cache_file(A)
        self.cache_file(SPOTIFY)
        plan = self.plan(names=(SPOTIFY, A, B), folds=(1, 2), last=(SPOTIFY,))
        self.assertEqual(plan.pending, [(A, 1), (B, 1), (SPOTIFY, 1), (A, 2), (B, 2), (SPOTIFY, 2)])
        self.assertEqual(plan.missing_caches, [B])
        with patch.object(sweep.subprocess, "call") as call:
            self.assertEqual(sweep.execute_plan(plan, "test", FLAGS, self.requested, self.root), 2)
        call.assert_not_called()

    def test_empty_cache_is_missing_and_final_truncated_line_is_not_done(self):
        self.cache_file(A).write_bytes(b"")
        self.results([self.record(method="agent_read_all")], tail='{"method": "_done"')
        plan = self.plan()
        self.assertEqual(plan.missing_caches, [A])
        self.assertEqual(plan.complete, [])
        self.assertTrue(any("truncated" in warning for warning in plan.warnings))

    def test_dry_run_preserves_separator_and_never_launches_or_creates_logs(self):
        cached = self.cache_file(A)
        with patch.object(sweep, "ROOT", self.root), patch.object(sweep, "CACHE_DIR", self.cache), \
                patch.object(sweep, "dataset_names", return_value=[A]), \
                patch.object(sweep, "run_arguments", wraps=lambda rest, root: sweep_run_arguments(rest)) as parsed, \
                patch.object(sweep.subprocess, "call") as call, contextlib.redirect_stdout(io.StringIO()) as out:
            code = sweep.main(["--tag", "test", "--folds", "1", "--dry-run", "--", *FLAGS])
        self.assertEqual(code, 0)
        parsed.assert_called_once_with(FLAGS, self.root)
        call.assert_not_called()
        self.assertFalse((self.root / "logs").exists())
        self.assertIn("presence/size only", out.getvalue())
        self.assertEqual(cached.read_bytes(), b"not a pickle; only its presence is checked")

    def test_dry_run_missing_cache_returns_nonzero(self):
        with patch.object(sweep, "ROOT", self.root), patch.object(sweep, "CACHE_DIR", self.cache), \
                patch.object(sweep, "dataset_names", return_value=[A]), \
                patch.object(sweep, "run_arguments", wraps=lambda rest, root: sweep_run_arguments(rest)), \
                patch.object(sweep.subprocess, "call") as call, contextlib.redirect_stdout(io.StringIO()) as out:
            code = sweep.main(["--tag", "test", "--folds", "1", "--dry-run", "--", *FLAGS])
        self.assertEqual(code, 2)
        call.assert_not_called()
        self.assertFalse((self.root / "logs").exists())
        self.assertIn("missing-cache", out.getvalue())

    def test_failed_children_nonzero_summary_and_previous_logs_retained(self):
        self.cache_file(A)
        logs = self.root / "logs"
        logs.mkdir()
        old = logs / "test_previous_crash.log"
        old.write_text("old crash", encoding="utf-8")
        with patch.object(sweep.subprocess, "call", return_value=7) as call, contextlib.redirect_stdout(io.StringIO()) as out:
            code = sweep.execute_plan(self.plan(), "test", FLAGS, self.requested, self.root)
        self.assertEqual(code, 1)
        self.assertIn("failed=1", out.getvalue())
        self.assertEqual(old.read_text(), "old crash")
        self.assertEqual(len(list(logs.glob("*.log"))), 2)
        self.assertEqual(call.call_args.args[0][-len(FLAGS):], FLAGS)

    def test_zero_exit_without_full_completion_is_failure(self):
        self.cache_file(A)
        with patch.object(sweep.subprocess, "call", return_value=0), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(sweep.execute_plan(self.plan(), "test", FLAGS, self.requested, self.root), 1)

    def test_completed_child_is_success_and_receives_exact_command(self):
        self.cache_file(A)
        def finish(command, **kwargs):
            self.results([self.record()])
            return 0
        with patch.object(sweep.subprocess, "call", side_effect=finish) as call, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(sweep.execute_plan(self.plan(), "test", FLAGS, self.requested, self.root), 0)
        self.assertEqual(call.call_args.args[0][:3], [sys.executable, "-m", "reading_agent.run"])

    def test_import_and_recipe_parser_do_not_import_ml_modules(self):
        # Isolated lightweight interpreter; no training code is executed.
        command = [sys.executable, "-c", "import sys; from reading_agent import sweep; sweep.run_arguments(['--protocol','paper']); assert not any(k in sys.modules for k in ('torch', 'reading_agent.run', 'reading_agent.data', 'reading_agent.embed'))"]
        checked = subprocess.run(command, cwd=PROJECT, capture_output=True, text=True)
        self.assertEqual(checked.returncode, 0, checked.stderr)


# Keep the original safe parser for the main() test's patched wrapper.
sweep_run_arguments = sweep.run_arguments


if __name__ == "__main__":
    unittest.main()
