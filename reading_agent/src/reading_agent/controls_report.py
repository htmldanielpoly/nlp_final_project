"""Decompose the gain policy's advantage over random reading using the `--controls 1` runs.

At each nominal budget the run records, on the same trained reader:
  gainB      learned policy (lambda calibrated on validation)
  randomB    random items, the policy's per-row read counts
  randomAgg  random items, the policy's *total* reads spread by row capacity
  naturalAgg natural field/position order, same counts as randomAgg
so gainB - randomAgg = total selection effect, gainB - randomB = item choice,
randomB - randomAgg = allocation of reads across rows, naturalAgg - randomAgg =
value of the fixed field order. Folds are averaged per dataset; Wilcoxon over
datasets, Holm-adjusted over the 8 key x budget cells of each comparison/task.

    python -m reading_agent.controls_report --tag controls_v2 [--fieldmeta_tag fieldmeta_v1] [--out ../paper/generated]

With --fieldmeta_tag it also compares the same recipe with field presence/log length
in the table (`--field_meta 1`) on matched dataset/folds: table-only and read-all
scores, and the gain-minus-random advantage at 10/25%.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .budget_report import holm_adjust
from .compare import method_scores, summarize_deltas
from .eval_paper import evaluation_results
from .paper_assets import CLS, OUT, REG, fmt_p, fmt_signed

COMPARISONS = [  # (name, left, right, label)
    ("total", "gainB", "randomAgg", "gain $-$ random (same total reads)"),
    ("choice", "gainB", "randomB", "gain $-$ random (same per-row reads)"),
    ("alloc", "randomB", "randomAgg", "policy row allocation $-$ uniform"),
    ("natural", "gainB", "naturalAgg", "gain $-$ natural field order"),
]


def scores(tag, results_dir=None):
    kw = {"results_dir": results_dir} if results_dir else {}
    df = evaluation_results(tag=tag, **kw)
    s = method_scores(df)
    return s.pivot(index=["dataset", "task_group", "fold"], columns="method", values="score"), df


def fractions(df):
    f = df[df.method.str.startswith(("gainB_", "randomAgg_"))].copy()
    stem = f.method.str.split("@").str[0].str.split("_")
    f["kind"], f["key"] = stem.str[0], stem.str[1]
    return f.groupby(["task_group", "key", "budget", "kind"]).frac.mean()


def report(wide):
    rows = []
    budgets = sorted({float(c.split("@")[1]) for c in wide if c.startswith("gainB_")})
    for task in (CLS, REG):
        if task not in wide.index.get_level_values("task_group"):
            continue
        part = wide.xs(task, level="task_group")
        for name, left, right, label in COMPARISONS:
            family = []
            for key in ("content", "meta"):
                for b in budgets:
                    lc, rc = f"{left}_{key}@{b:g}", f"{right}_{key}@{b:g}"
                    if lc not in part or rc not in part:
                        continue
                    d = part[[lc, rc]].dropna()
                    deltas = (d[lc] - d[rc]).groupby(level="dataset").mean().rename("delta").reset_index()
                    st = summarize_deltas(deltas)
                    family.append(dict(task=task, comparison=name, label=label, key=key, budget=b,
                                       folds=int(d.groupby(level="dataset").size().min()), **st))
            for r, p in zip(family, holm_adjust([r["pvalue"] for r in family])):
                r["holm_pvalue"] = float(p)
            rows += family
    return pd.DataFrame(rows)


def latex(rep, budgets=(0.1, 0.5)):
    """One row per comparison x key; columns: budgets per task family (dataset-mean delta, wins)."""
    head = ("\\begin{tabular}{@{}ll" + "cc" * len(budgets) + "@{}}\n\\toprule\n & & "
            + " & ".join(f"\\multicolumn{{{len(budgets)}}}{{c}}{{{t}}}" for t in ("Classification ($\\Delta$AUROC)", "Regression ($\\Delta R^2$)"))
            + " \\\\\n" + "".join(f"\\cmidrule(lr){{{3 + i * len(budgets)}-{2 + (i + 1) * len(budgets)}}}" for i in range(2))
            + "\nComparison & Keys & " + " & ".join(f"{int(b * 100)}\\%" for b in tuple(budgets) * 2) + " \\\\\n\\midrule\n")
    lines = []
    for name, _, _, label in COMPARISONS:
        for key, kl in (("content", "content"), ("meta", "metadata")):
            cells = [label if key == "content" else "", kl]
            for task in (CLS, REG):
                for b in budgets:
                    r = rep[(rep.comparison == name) & (rep.key == key) & (rep.task == task) & np.isclose(rep.budget, b)]
                    if r.empty:
                        cells.append("--")
                        continue
                    r = r.iloc[0]
                    star = "$^{*}$" if np.isfinite(r.holm_pvalue) and r.holm_pvalue < 0.05 else ""
                    cells.append(f"{fmt_signed(r.mean_delta)} ({r.wins}/{r.n_datasets}){star}")
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    lines.pop()
    return head + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"


def macros(rep, wide):
    out = {"ctlFolds": int(wide.groupby(level=["dataset"]).size().min()),
           "ctlDatasets": wide.index.get_level_values("dataset").nunique()}
    for task, tk in ((CLS, "Cls"), (REG, "Reg")):
        for _, r in rep[rep.task == task].iterrows():
            if not np.isclose(r.budget, 0.1):
                continue
            stem = f"ctl{r.comparison.title()}{r.key.title()}Ten{tk}"
            out[stem + "Delta"] = fmt_signed(r.mean_delta)
            out[stem + "Wins"] = int(r.wins)
            out[stem + "N"] = int(r.n_datasets)
            out[stem + "P"] = fmt_p(r.holm_pvalue)
    return "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in out.items())


FM_ROWS = [("table", "Reader, table only"), ("readall", "Reader, read all"),
           ("adv_content@0.1", "gain $-$ random, content @10\\%"), ("adv_content@0.25", "gain $-$ random, content @25\\%"),
           ("adv_meta@0.1", "gain $-$ random, metadata @10\\%"), ("adv_meta@0.25", "gain $-$ random, metadata @25\\%")]


def fm_metrics(wide):
    out = pd.DataFrame({"table": wide["agent_table_only"], "readall": wide["agent_read_all"]})
    for key in ("content", "meta"):
        for b in (0.1, 0.25):
            out[f"adv_{key}@{b:g}"] = wide[f"gainB_{key}@{b:g}"] - wide[f"randomB_{key}@{b:g}"]
    return out


def fieldmeta_compare(base_wide, fm_wide):
    """Same recipe with vs without [present, log length] per text field in the table (matched folds)."""
    base, fm = fm_metrics(base_wide), fm_metrics(fm_wide)
    common = base.index.intersection(fm.index)
    base, fm = base.loc[common], fm.loc[common]
    rows = []
    for task in (CLS, REG):
        if task not in common.get_level_values("task_group"):
            continue
        b, f = base.xs(task, level="task_group"), fm.xs(task, level="task_group")
        for metric, label in FM_ROWS:
            bd, fd = b[metric].groupby(level="dataset").mean(), f[metric].groupby(level="dataset").mean()
            st = summarize_deltas((fd - bd).rename("delta").reset_index())
            rows.append(dict(task=task, metric=metric, label=label, without=bd.mean(), with_meta=fd.mean(), **st))
    return pd.DataFrame(rows)


def fieldmeta_latex(fmr):
    head = ("\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n & \\multicolumn{3}{c}{Classification (AUROC)} & \\multicolumn{3}{c}{Regression ($R^2$)} \\\\\n"
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n & without & with & $\\Delta$ (W/$n$) & without & with & $\\Delta$ (W/$n$) \\\\\n\\midrule\n")
    lines = []
    for metric, label in FM_ROWS:
        cells = [label]
        for task in (CLS, REG):
            r = fmr[(fmr.metric == metric) & (fmr.task == task)]
            if r.empty:
                cells += ["--"] * 3
                continue
            r = r.iloc[0]
            star = "$^{*}$" if np.isfinite(r.pvalue) and r.pvalue < 0.05 else ""
            cells += [fmt_signed(r.without) if metric.startswith("adv") else f"{r.without:.3f}",
                      fmt_signed(r.with_meta) if metric.startswith("adv") else f"{r.with_meta:.3f}",
                      f"{fmt_signed(r.mean_delta)} ({r.wins}/{r.n_datasets}){star}"]
        lines.append(" & ".join(cells) + " \\\\")
    return head + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"


def fieldmeta_macros(fmr):
    out = {}
    for task, tk in ((CLS, "Cls"), (REG, "Reg")):
        for _, r in fmr[fmr.task == task].iterrows():
            name, _, budget = r.metric.partition("@")
            stem = "fm" + "".join(part.title() for part in name.split("_")) + {"0.1": "Ten", "0.25": "Quarter", "": ""}[budget] + tk
            out[stem + "Without"] = fmt_signed(r.without) if r.metric.startswith("adv") else f"{r.without:.3f}"
            out[stem + "With"] = fmt_signed(r.with_meta) if r.metric.startswith("adv") else f"{r.with_meta:.3f}"
            out[stem + "Delta"] = fmt_signed(r.mean_delta)
            out[stem + "Wins"] = int(r.wins)
            out[stem + "N"] = int(r.n_datasets)
            out[stem + "P"] = fmt_p(r.pvalue)
    return "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in out.items())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="controls_v2")
    ap.add_argument("--fieldmeta_tag", default=None, help="e.g. fieldmeta_v1: same recipe with --field_meta 1")
    ap.add_argument("--out", type=Path, default=None, help="write tab_controls.tex and controls.tex here")
    args = ap.parse_args(argv)
    wide, df = scores(args.tag)
    rep = report(wide)
    folds = wide.groupby(level="dataset").size()
    print(f"tag {args.tag}: {len(folds)} datasets, folds per dataset {folds.min()}-{folds.max()}")
    pd.set_option("display.width", 200)
    for task in (CLS, REG):
        part = rep[rep.task == task]
        if part.empty:
            continue
        print(f"\n== {task}")
        print(part[["comparison", "key", "budget", "n_datasets", "mean_delta", "wins", "ties", "pvalue", "holm_pvalue"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nmean achieved read fraction (gainB vs randomAgg):")
    print(fractions(df).unstack("kind").to_string(float_format=lambda v: f"{v:.3f}"))
    tex = macros(rep, wide)
    if args.fieldmeta_tag:
        fmr = fieldmeta_compare(wide, scores(args.fieldmeta_tag)[0])
        print(f"\n== field metadata in the table ({args.fieldmeta_tag} vs {args.tag}, matched folds; with - without)")
        print(fmr[["task", "metric", "n_datasets", "without", "with_meta", "mean_delta", "wins", "ties", "pvalue"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        tex += fieldmeta_macros(fmr)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "tab_controls.tex").write_text(latex(rep), encoding="utf-8")
        if args.fieldmeta_tag:
            (args.out / "tab_fieldmeta.tex").write_text(fieldmeta_latex(fmr), encoding="utf-8")
        (args.out / "controls.tex").write_text(tex, encoding="utf-8")
        print(f"wrote tab_controls.tex{', tab_fieldmeta.tex' if args.fieldmeta_tag else ''} and controls.tex to {args.out}")


if __name__ == "__main__":
    main()
