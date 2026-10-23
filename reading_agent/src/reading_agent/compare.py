"""Compare completed experiments with external paper-era leaderboard references.

Paper split logic alone does not verify identical data or test rows. External
comparisons remain references; paper fold-matched statistics are exploratory.
Current-protocol comparisons never receive inferential p-values.
"""
import argparse
import glob
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# Bundled copy of the MulTaBench leaderboard CSVs (source commit in reference/.../SOURCE.md).
DEFAULT_LB = os.environ.get("MULTABENCH_LB", str(Path(__file__).resolve().parents[2] / "reference" / "multabench_leaderboard_text"))
GROUP = ["protocol", "analysis_config"]
SCORE_KEY = ["dataset", "fold", "method"]
_CONTROL_ARGS = {"dataset", "fold", "tag", "skip", "key_modes", "lams", "budgets", "extra", "protocol", "save_models", "controls", "data_revision", "baseline_cache"}
_CONFIG_FIELDS = ("field_meta", "sched", "item_mode", "cell", "layers")


def load_leaderboard(path=DEFAULT_LB):
    files = sorted(glob.glob(os.path.join(path, "*.csv")))
    if not files:
        raise ValueError(f"no leaderboard CSVs in {path}")
    lb = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    lb["model"] = lb.model.str.split().str[0]
    lb["cond"] = lb.multimodal_state.map({"all": "frozen", "ft": "tar"})
    lb = lb[lb.cond.notna()][["dataset", "fold", "model", "cond", "test_score"]]
    lb["test_score"] = pd.to_numeric(lb.test_score, errors="coerce")
    key = ["dataset", "fold", "model", "cond"]
    lb = lb.drop_duplicates()
    if lb.duplicated(key).any():
        raise ValueError("conflicting leaderboard scores for the same dataset/fold/model/condition")
    return lb


def _present(value):
    return value is not None and not (isinstance(value, float) and np.isnan(value))


def _configuration(row):
    args = row.get("args")
    config = {k: v for k, v in args.items() if k not in _CONTROL_ARGS} if isinstance(args, dict) else {"args": "unrecorded"}
    for name in _CONFIG_FIELDS:
        value = row.get(name)
        if _present(value):
            config.setdefault(name, value)
    if row.get("implementation_version"):
        config["implementation_version"] = row["implementation_version"]
    # These opt-in flags did not exist in historical runs; their old behavior
    # was automatic CPU threads without an explicit deterministic request.
    config.setdefault("threads", 0)
    config.setdefault("deterministic", 0)
    detail = json.dumps(config, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(detail.encode()).hexdigest()[:10], detail


def task_group(row):
    task = row.get("task")
    if task == "regression" or str(row.get("dataset", "")).startswith("REG_"):
        return "regression (R2)"
    if task in {"binary", "multiclass"} or str(row.get("dataset", "")).startswith(("BIN_", "MUL_")):
        return "classification (AUROC)"
    return "unknown task"


def load_results(paths):
    """Preserve the DataFrame API and method@budget/lam naming.

    Completion is scoped to each input file. Legacy rows without run_id remain
    visible but explicitly unverified. No repeated scores are averaged.
    """
    rows, excluded = [], 0
    for path in paths:
        source = str(Path(path).resolve())
        with open(path, encoding="utf-8") as handle:
            file_rows = [json.loads(line) for line in handle if line.strip()]
        finished = {r.get("run_id") for r in file_rows if r.get("method") == "_done" and _present(r.get("run_id"))}
        for row in file_rows:
            if row.get("method") == "_done":
                continue
            run_id = row.get("run_id")
            if _present(run_id) and run_id not in finished:
                excluded += 1
                continue
            row = dict(row)
            row["source_path"] = source
            row["completion"] = "finished" if _present(run_id) else "legacy-unverified"
            row["protocol"] = row.get("protocol") or "current"
            row["analysis_run"] = f"{source}::{run_id}" if _present(run_id) else f"{source}::legacy::{row.get('tag', '')}::{row.get('dataset')}::{row.get('fold')}"
            row["analysis_config"], row["analysis_config_detail"] = _configuration(row)
            row["task_group"] = task_group(row)
            method = str(row.get("method", ""))
            budget, lam = row.get("budget"), row.get("lam")
            if _present(budget):
                method += f"@{float(budget):g}"
            elif _present(lam):
                method += f"@{float(lam):g}"
            row["method"] = method
            rows.append(row)
    columns = SCORE_KEY + GROUP + ["score", "task_group", "source_path", "analysis_run", "analysis_config_detail", "completion", "tag", "run_id"]
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)
    df["score"] = pd.to_numeric(df.score, errors="coerce")
    # Ensure optional legacy columns exist for downstream selection/diagnostics.
    for name in ("tag", "run_id"):
        if name not in df:
            df[name] = None
    df.attrs["excluded_partial_rows"] = excluded
    return df


def add_selection_args(parser):
    parser.add_argument("--tag", default="", help="comma-separated tags to include")
    parser.add_argument("--run-id", default="", help="comma-separated completed run IDs to include")


def select_results(df, tag="", run_id=""):
    attrs = dict(df.attrs)
    if tag:
        df = df[df.tag.isin(tag.split(","))]
    if run_id:
        df = df[df.run_id.isin(run_id.split(","))]
    df = df.copy()
    df.attrs.update(attrs)
    return df


def report_coverage(df):
    print(f"Selected score rows: {len(df)}; excluded partial rows: {df.attrs.get('excluded_partial_rows', 0)}")
    if df.empty:
        print("no completed results selected")
        return
    print("Protocol / configuration / task coverage (no mixed-config averaging):")
    coverage = df.groupby(GROUP + ["task_group"]).agg(datasets=("dataset", "nunique"), runs=("analysis_run", "nunique"), scores=("score", "size"))
    print(coverage.to_string())
    finite = np.isfinite(df.score.to_numpy(dtype=float))
    if not finite.all():
        print(f"Non-finite scores: {int((~finite).sum())}; excluded from difference summaries.")
    legacy = int((df.completion == "legacy-unverified").sum())
    if legacy:
        print(f"Legacy rows without completion markers: {legacy}; completeness is unverified.")
    for config, part in df.groupby("analysis_config"):
        print(f"Config {config}: {part.analysis_config_detail.iloc[0]}")
    print("Fold coverage:")
    print(df.groupby(GROUP + ["dataset"]).fold.agg(lambda v: ",".join(str(x) for x in sorted(v.unique()))).to_string())
    from .data import TEXT_DATASETS
    known, present = set(TEXT_DATASETS), set(df.dataset)
    if present <= known:
        print(f"Text benchmark coverage: {len(present)}/{len(known)} datasets")
        if known - present:
            print("Missing datasets: " + ", ".join(sorted(known - present)))


def method_scores(df):
    """One score per dataset/fold/method inside one config/protocol."""
    if len(df[GROUP].drop_duplicates()) > 1:
        raise ValueError("select one protocol/configuration before computing method scores")
    if df.duplicated(SCORE_KEY).any():
        examples = df.loc[df.duplicated(SCORE_KEY, keep=False), SCORE_KEY + ["run_id", "tag"]]
        raise ValueError("multiple score rows per dataset/fold/method; select --tag or --run-id:\n" + examples.to_string(index=False))
    return df[SCORE_KEY + ["score", "task_group"]].copy()


def paired_dataset_deltas(pairs, left="score", right="test_score"):
    """Average finite within-fold differences per dataset, never count folds as n."""
    if pairs.empty:
        return pd.DataFrame(columns=["dataset", "task_group", "delta", "folds"])
    finite = np.isfinite(pairs[left].to_numpy(dtype=float)) & np.isfinite(pairs[right].to_numpy(dtype=float))
    part = pairs.loc[finite].copy()
    part["delta"] = part[left] - part[right]
    return part.groupby(["dataset", "task_group"], as_index=False).agg(delta=("delta", "mean"), folds=("fold", "nunique"))


def summarize_deltas(deltas, allow_inference=True, tie_tolerance=1e-8):
    """Summarize one task family; all-zero differences are a defined null result."""
    if "task_group" in deltas and deltas.task_group.nunique() > 1:
        raise ValueError("classification and regression must be summarized separately")
    values = pd.to_numeric(deltas.get("delta", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    values = np.where(np.abs(values) <= tie_tolerance, 0.0, values)
    n, p = len(values), float("nan")
    if allow_inference and n >= 6:
        p = 1.0 if np.all(values == 0) else float(wilcoxon(values).pvalue)
    return {"n_datasets": n, "mean_delta": float(values.mean()) if n else float("nan"), "wins": int((values > 0).sum()), "ties": int((values == 0).sum()), "pvalue": p}


def print_delta_summary(label, deltas, allow_inference=True, exploratory=False):
    for task, part in deltas.groupby("task_group"):
        stats = summarize_deltas(part, allow_inference=allow_inference)
        p = f"{stats['pvalue']:.3g}" if np.isfinite(stats["pvalue"]) else "not reported"
        qualifier = "exploratory Wilcoxon" if exploratory else "Wilcoxon"
        print(f"{label} | {task}: datasets={stats['n_datasets']} mean delta={stats['mean_delta']:+.4f} "
              f"wins={stats['wins']}/{stats['n_datasets']} ties={stats['ties']}; {qualifier} p={p}")


def compare_group(part, lb, methods=None):
    ours = method_scores(part)
    protocol, config = part.protocol.iloc[0], part.analysis_config.iloc[0]
    print(f"\n== protocol={protocol}, config={config}")
    print("External LB values are references; identical data and test-row provenance have not been verified.")
    if protocol != "paper":
        print("Different outer split protocol: external inferential p-values suppressed.")
    else:
        print("Paper split logic only: fold-matched deltas/p-values are exploratory, not verified paired benchmark evidence.")
    chosen = methods or [m for m in sorted(ours.method.unique()) if m.startswith(("agent_read_all", "soft", "hybrid", "blend", "catboost_frozen"))]
    table = ours[ours.method.isin(chosen)].pivot(index=["dataset", "task_group", "fold"], columns="method", values="score")
    references = lb[lb.model.isin(["CatBoost", "TabPFN-v2p5", "TabM"])].copy()
    references["reference"] = "REF_" + references.model + "_" + references.cond
    reference_table = references.pivot(index=["dataset", "fold"], columns="reference", values="test_score")
    print(table.join(reference_table, on=["dataset", "fold"]).round(4).to_string())
    for method in chosen:
        mine = ours[ours.method == method].drop(columns="method")
        for (model, cond), ref in lb.groupby(["model", "cond"]):
            if model not in {"CatBoost", "TabPFN-v2p5", "TabM"}:
                continue
            pairs = mine.merge(ref, on=["dataset", "fold"], validate="one_to_one")
            if pairs.empty:
                continue
            deltas = paired_dataset_deltas(pairs)
            print_delta_summary(f"{method} vs REF_{model}_{cond}", deltas, allow_inference=protocol == "paper", exploratory=True)
            missing = len(mine) - len(pairs)
            if missing:
                print(f"  Missing external reference for {missing}/{len(mine)} dataset/fold scores.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", nargs="+")
    ap.add_argument("--leaderboard", default=DEFAULT_LB)
    ap.add_argument("--methods", default="")
    add_selection_args(ap)
    args = ap.parse_args(argv)
    try:
        res = select_results(load_results(args.results), args.tag, args.run_id)
        report_coverage(res)
        if res.empty:
            return
        lb = load_leaderboard(args.leaderboard)
        pd.set_option("display.width", 250)
        for _, part in res.groupby(GROUP):
            compare_group(part, lb, [m for m in args.methods.split(",") if m])
    except ValueError as error:
        ap.error(str(error))


if __name__ == "__main__":
    main()
