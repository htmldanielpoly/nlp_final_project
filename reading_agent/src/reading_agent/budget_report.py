"""Dataset-level budget effects, actual read fractions, and exploratory uncertainty.

Each task/config is kept separate. Holm adjustment covers the key×budget family
within that task/config. Bootstrap intervals resample dataset mean differences;
they do not measure training-seed uncertainty or establish true encoding savings.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .compare import GROUP, load_results, method_scores, paired_dataset_deltas, summarize_deltas
from .summarize import budget_comparison

RESULTS = Path(__file__).resolve().parents[2] / "results"


def holm_adjust(values):
    values = np.asarray(values, dtype=float)
    result = np.full(len(values), np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    ordered = finite[np.argsort(values[finite])]
    previous = 0.0
    for rank, index in enumerate(ordered):
        previous = max(previous, min(1.0, (len(ordered) - rank) * values[index]))
        result[index] = previous
    return result


def bootstrap_interval(values, n_resamples=5000, seed=42):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    if len(values) < 2:
        return None
    rng = np.random.RandomState(seed)
    draws = rng.choice(values, size=(n_resamples, len(values)), replace=True).mean(1)
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def report(results, n_resamples=5000):
    rows = []
    for (protocol, config), part in results.groupby(GROUP):
        method_scores(part)  # reject ambiguous/repeated runs before summarizing
        budgets = budget_comparison(part)
        for task, family in budgets.groupby("task_group"):
            effects = []
            for (key, budget), pairs in family.groupby(["key", "budget"]):
                deltas = paired_dataset_deltas(pairs, "gain", "random")
                stats = summarize_deltas(deltas)
                actual = pairs.groupby("dataset").frac.mean()
                finite = actual[np.isfinite(actual)]
                effects.append(dict(protocol=protocol, config=config, task=task, key=key, nominal_budget=float(budget),
                     **stats, bootstrap_mean_ci95=bootstrap_interval(deltas.delta, n_resamples),
                     actual_fraction_mean=float(finite.mean()) if len(finite) else None,
                     actual_fraction_min=float(finite.min()) if len(finite) else None,
                     actual_fraction_max=float(finite.max()) if len(finite) else None,
                     datasets_over_5pct_budget_error=int(((finite-budget).abs() > .05).sum()),
                     budget_mismatch_datasets=finite.index[(finite-budget).abs() > .05].tolist(),
                     uncertainty="exploratory dataset bootstrap; one-fold results omit training-seed variability"))
            adjusted = holm_adjust([e["pvalue"] for e in effects])
            for effect, p in zip(effects, adjusted):
                effect["holm_pvalue"] = float(p)
                effect["holm_family_size"] = len(effects)
                rows.append(effect)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", nargs="*")
    ap.add_argument("--tag")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args(argv)
    paths = args.results or ([RESULTS / f"{args.tag}.jsonl"] if args.tag else [])
    if not paths:
        ap.error("provide result files or --tag")
    rows = report(load_results(paths))
    # Strict JSON: unavailable inferential p-values are null, never NaN.
    clean = [{k: None if isinstance(v, float) and not np.isfinite(v) else v for k, v in r.items()} for r in rows]
    rendered = json.dumps(clean, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
