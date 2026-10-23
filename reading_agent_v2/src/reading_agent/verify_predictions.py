"""Recompute completed run scores from saved per-row predictions.

Random baselines average each order's metric, not their probabilities. Historical
archives that stored only order zero are explicitly reported as incomplete.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .run import RESULTS, score


def verify(paths, predictions_dir=None, tolerance=1e-5):
    checked, incomplete, mismatches = 0, [], []
    for path in paths:
        records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        done = {r.get("run_id") for r in records if r.get("method") == "_done"}
        archives = {}
        try:
            for row in records:
                if not row.get("run_id"):
                    incomplete.append((None, row.get("method"), "legacy row has no run ID"))
                    continue
                if row.get("run_id") not in done or row.get("method") == "_done":
                    continue
                run_id = row["run_id"]
                if run_id not in archives:
                    archive_path = (Path(predictions_dir) if predictions_dir else RESULTS / "preds") / row["tag"] / f"{run_id}.npz"
                    archives[run_id] = np.load(archive_path)
                data = archives[run_id]
                keys = row.get("prediction_keys")
                if not keys:
                    if row.get("seed_scores") and len(row["seed_scores"]) > 1:
                        incomplete.append((run_id, row["method"], "historical archive saved only one random order"))
                        continue
                    key = row.get("prediction_key") or row["method"] + (f"@{row['budget']:g}" if "budget" in row else "")
                    if key not in data:
                        incomplete.append((run_id, row["method"], "no stored prediction array"))
                        continue
                    keys = [key]
                actual = float(np.mean([score(row["task"], data["y_te"], data[key]) for key in keys]))
                checked += 1
                if not np.isfinite(actual) or not np.isclose(actual, row["score"], atol=tolerance, rtol=0):
                    mismatches.append(dict(run_id=run_id, method=row["method"], recorded=row["score"], recomputed=actual))
        finally:
            for data in archives.values():
                data.close()
    return dict(checked=checked, incomplete=incomplete, mismatches=mismatches)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", nargs="+")
    ap.add_argument("--predictions-dir", type=Path)
    args = ap.parse_args(argv)
    result = verify(args.results, args.predictions_dir)
    print(json.dumps(result, indent=2))
    if result["mismatches"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
