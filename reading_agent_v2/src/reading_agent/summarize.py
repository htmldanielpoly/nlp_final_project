"""Summarize completed runs, preserving protocols, configurations and task families."""
import argparse

import pandas as pd

from .compare import (GROUP, add_selection_args, load_results, method_scores,
                      paired_dataset_deltas, print_delta_summary, report_coverage,
                      select_results)

KEY = ["catboost_frozen", "catboost_tab_only", "agent_table_only", "agent_read_all", "blend_catboost_agent"]


def budget_comparison(part):
    """Match learned/random budget rows within the same completed run."""
    columns = ["dataset", "fold", "task_group", "key", "budget", "gain", "random", "frac", "adv"]
    if "budget" not in part:
        return pd.DataFrame(columns=columns)
    b = part[part.budget.notna()].copy()
    b["kind"] = b.method.str.split("@").str[0]
    b["policy"] = b.kind.str.extract(r"^(gainB|randomB)_")[0]
    b["key"] = b.kind.str.extract(r"_(content|meta)$")[0]
    b = b[b.policy.notna() & b.key.notna()]
    if b.empty:
        return pd.DataFrame(columns=columns)
    index = ["dataset", "fold", "task_group", "analysis_run", "key", "budget"]
    if b.duplicated(index + ["policy"]).any():
        raise ValueError("duplicate policy scores within a run; no hidden budget averaging is allowed")
    scores = b.pivot(index=index, columns="policy", values="score").reindex(columns=["gainB", "randomB"])
    out = scores.rename(columns={"gainB": "gain", "randomB": "random"})
    if "frac" in b:
        out["frac"] = b[b.policy == "gainB"].set_index(index).frac
    else:
        out["frac"] = float("nan")
    out["adv"] = out.gain - out.random
    return out.reset_index()


def summarize_group(part, by=""):
    scores = method_scores(part)
    print(f"\n== protocol={part.protocol.iloc[0]}, config={part.analysis_config.iloc[0]}")
    index = ["dataset", "task_group", "fold"]
    if by:
        scores = scores.merge(part[["dataset", "fold", "method", by]], on=["dataset", "fold", "method"], validate="one_to_one")
        if by not in index:
            index += [by]
    acc = scores[scores.method.isin(KEY)].pivot(index=index, columns="method", values="score")
    print("Scores:\n" + acc.reindex(columns=[k for k in KEY if k in acc]).round(4).to_string())
    budgets = budget_comparison(part)
    if not budgets.empty:
        print("\nBudget curves (gain/random matched within run; frac is actual test read fraction):")
        print(budgets.drop(columns="analysis_run").round(4).to_string(index=False))
        print("\nDataset-level advantages, separately by task/key/budget:")
        for (key, budget), pairs in budgets.groupby(["key", "budget"]):
            deltas = paired_dataset_deltas(pairs, left="gain", right="random")
            print_delta_summary(f"gain - random ({key}, budget={budget:g})", deltas)
    return acc, budgets


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", nargs="+")
    ap.add_argument("--by", default="", help="additional displayed column, e.g. field_meta")
    add_selection_args(ap)
    args = ap.parse_args(argv)
    try:
        df = select_results(load_results(args.results), args.tag, args.run_id)
        if args.by:
            df[args.by] = df[args.by].fillna("unset") if args.by in df else "unset"
        report_coverage(df)
        pd.set_option("display.width", 250)
        for _, part in df.groupby(GROUP):
            summarize_group(part, args.by)
    except ValueError as error:
        ap.error(str(error))


if __name__ == "__main__":
    main()
