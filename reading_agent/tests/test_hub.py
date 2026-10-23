import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from reading_agent import hub


class HubIndexTest(unittest.TestCase):
    def test_index_lists_completed_runs_with_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [dict(run_id="A_f0_1", dataset="A", fold=0, task="binary", protocol="paper", method="agent_read_all", score=0.8),
                    dict(run_id="A_f0_1", dataset="A", fold=0, task="binary", protocol="paper", method="gainB_meta", budget=0.1, score=0.7),
                    dict(run_id="A_f0_1", dataset="A", fold=0, task="binary", protocol="paper", method="gainB_meta", budget=0.5, score=0.75),
                    dict(run_id="A_f0_1", dataset="A", fold=0, task="binary", protocol="paper", method="_done", score=0.0),
                    dict(run_id="B_f0_2", dataset="B", fold=0, task="regression", protocol="paper", method="agent_read_all", score=0.5)]
            (root / "t.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            for run, sub in (("A_f0_1", "reader"), ("A_f0_1", "gain_meta"), ("B_f0_2", "reader")):
                (root / "models" / "t" / run / sub).mkdir(parents=True)
            meta = {"metadata": {"implementation_version": "v", "provenance": {
                "data_revision": "abc", "train_fingerprint": "tr", "test_fingerprint": "te"}}}
            (root / "models" / "t" / "A_f0_1" / "reader" / "manifest.json").write_text(json.dumps(meta), encoding="utf-8")
            with mock.patch.object(hub, "RESULTS", root):
                path, index = hub.write_index("t")
            self.assertEqual(list(index["runs"]), ["A_f0_1"])  # B has no _done row
            entry = index["runs"]["A_f0_1"]
            self.assertEqual(entry["artifacts"], ["gain_meta", "reader"])
            self.assertEqual((entry["data_revision"], entry["train_fingerprint"], entry["test_fingerprint"],
                              entry["implementation_version"]), ("abc", "tr", "te", "v"))
            self.assertEqual(entry["test_scores"], {"agent_read_all": 0.8, "gainB_meta@0.1": 0.7})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), index)


if __name__ == "__main__":
    unittest.main()
