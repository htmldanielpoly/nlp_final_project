"""Check that a saved run artifact reproduces its stored test predictions.

Raw benchmark rows go through a CSV round trip (as with the inference CLI), then
the loaded artifact predicts them. Stored regression predictions are in
standardised target units; the artifact returns original units, so they are
compared after undoing the saved target scaling.
"""
import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_dataset
from .estimator import ReadingEstimator
from .run import RESULTS

CHECKS = [("reader", "all", None, "agent_read_all"), ("reader", "blend", None, "blend_catboost_agent"),
          ("reader", "catboost", None, "catboost_frozen")]
BUDGET_CHECKS = [("gain_content", "gainB_content"), ("gain_meta", "gainB_meta")]


def check(tag, run_id, rows=300, budgets=(0.1, 0.5), tolerance=1e-4, root=None):
    """`root` overrides the local artifact directory, e.g. a run downloaded from the Hub."""
    archive = np.load(RESULTS / "preds" / tag / f"{run_id}.npz")
    root = Path(root) if root else RESULTS / "models" / tag / run_id
    dataset = run_id.split("_f")[0]
    idx = archive["te_idx"][:rows]
    with tempfile.TemporaryDirectory() as tmp:
        csv = Path(tmp) / "rows.csv"
        load_dataset(dataset).x.iloc[idx].to_csv(csv, index=False)
        frame = pd.read_csv(csv)
    checks = list(CHECKS) + [(sub, "gain", b, f"{key}@{b:g}") for sub, key in BUDGET_CHECKS for b in budgets]
    loaded, failures = {}, []
    for sub, mode, budget, key in checks:
        if not (root / sub).exists() or key not in archive.files:
            print(f"{sub:12s} {mode:8s} {key}: skipped (not saved)")
            continue
        est = loaded.setdefault(sub, ReadingEstimator.load(root / sub))
        out, reads = est.predict_outputs(frame, mode=mode, budget=budget)
        ref = archive[key][:rows]
        if est.task == "regression":
            out = (out - est.target_mean) / est.target_scale
        diff = float(np.abs(out.reshape(ref.shape) - ref).max())
        reads_ok = f"{key}__reads" not in archive.files or bool(np.array_equal(reads, archive[f"{key}__reads"][:rows]))
        ok = diff <= tolerance and reads_ok
        failures += [] if ok else [key]
        print(f"{sub:12s} {mode:8s} {key:24s} max|diff|={diff:.2e} reads match={reads_ok} {'OK' if ok else 'MISMATCH'}")
    return failures


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="paper_v2")
    ap.add_argument("--run_id", required=True, help="e.g. MUL_TEXT_WOMEN_CLOTHING_REVIEW_f0_1791988577_1772912b")
    ap.add_argument("--rows", type=int, default=300)
    ap.add_argument("--repo", help="check the copy published in this Hugging Face repo instead of the local one")
    args = ap.parse_args(argv)
    root = None
    if args.repo:
        from .hub import fetch
        dataset, rest = args.run_id.split("_f", 1)
        root = fetch(args.repo, dataset, int(rest.split("_")[0]))
        if root.name != args.run_id:
            raise SystemExit(f"Hub run {root.name} is not {args.run_id}")
        print(f"checking Hub copy {root}")
    failures = check(args.tag, args.run_id, args.rows, root=root)
    print("all checks passed" if not failures else f"mismatches: {failures}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
