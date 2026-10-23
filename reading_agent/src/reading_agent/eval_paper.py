"""Paper-oriented evaluation with explicit input files or a result tag.

With no inputs, retain the historical paper_f0 + paper_f0_cap selection, excluding
only the original buggy Rotten Tomatoes run. Explicit inputs have no exclusions.
Importing this module never reads result files or executes an evaluation.
"""
import argparse
from pathlib import Path

from .compare import (DEFAULT_LB, GROUP, compare_group, load_leaderboard,
                      load_results, report_coverage, select_results)
from .summarize import summarize_group

RESULTS = Path(__file__).resolve().parents[2] / "results"
DEFAULT_FILES = ("paper_f0.jsonl", "paper_f0_cap.jsonl")
BUGGY_DATASET = "REG_TEXT_ROTTEN_TOMATOES"


def evaluation_results(paths=None, tag="", results_dir=RESULTS, run_id=""):
    """Load an explicit selection or the backwards-compatible historical bundle."""
    default = not paths and not tag
    if not paths:
        paths = [Path(results_dir) / f"{t}.jsonl" for t in tag.split(",")] if tag else [Path(results_dir) / name for name in DEFAULT_FILES]
    df = select_results(load_results(paths), tag, run_id)
    if default:
        buggy = (df.dataset == BUGGY_DATASET) & (df.source_path.map(lambda p: Path(p).name) == DEFAULT_FILES[0])
        attrs = dict(df.attrs)
        df = df[~buggy].copy()
        df.attrs.update(attrs)
        df.attrs["historical_exclusion_rows"] = int(buggy.sum())
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", nargs="*")
    ap.add_argument("--tag", default="", help="comma-separated tags; resolves results/<tag>.jsonl when no paths given")
    ap.add_argument("--results-dir", type=Path, default=RESULTS)
    ap.add_argument("--run-id", default="")
    ap.add_argument("--leaderboard", default=DEFAULT_LB)
    args = ap.parse_args(argv)
    try:
        df = evaluation_results(args.results, args.tag, args.results_dir, args.run_id)
        if "historical_exclusion_rows" in df.attrs:
            print(f"Historical default: excluded {df.attrs['historical_exclusion_rows']} buggy paper_f0 Rotten Tomatoes rows; retained paper_f0_cap.")
        report_coverage(df)
        if df.empty:
            return
        lb = load_leaderboard(args.leaderboard)
        for _, part in df.groupby(GROUP):
            summarize_group(part)
            compare_group(part, lb, ["agent_read_all", "catboost_frozen", "blend_catboost_agent"])
    except ValueError as error:
        ap.error(str(error))


if __name__ == "__main__":
    main()
