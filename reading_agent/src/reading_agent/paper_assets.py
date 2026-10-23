"""Generate the paper's tables, figure and number macros from completed results.

Everything is derived from one tag/configuration (default `paper_v2`). Folds are
averaged within each dataset before any cross-dataset summary or Wilcoxon test.
External MulTaBench leaderboard values are matched on the folds we completed.

    python -m reading_agent.paper_assets --tag paper_v2 --out ../paper/generated
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .budget_report import report as budget_report
from .compare import DEFAULT_LB, GROUP, load_leaderboard, method_scores, summarize_deltas
from .eval_paper import evaluation_results
from .summarize import budget_comparison

OUT = Path(__file__).resolve().parents[3] / "paper" / "generated"
CLS, REG = "classification (AUROC)", "regression (R2)"
SHORT = {
    "BIN_TEXT_FAKE_JOB_POSTING": "Fake Job Posting", "BIN_TEXT_JIGSAW_TOXICITY": "Jigsaw Toxicity",
    "BIN_TEXT_KICKSTARTER_FUNDING": "Kickstarter", "MUL_TEXT_DATA_SCIENTIST_SALARY": "DS Salary",
    "MUL_TEXT_MICHELIN_RESTAURANTS": "Michelin", "MUL_TEXT_PRODUCT_SENTIMENT": "Product Sentiment",
    "MUL_TEXT_SPOTIFY_GENRES": "Spotify Genres", "MUL_TEXT_US_ACCIDENTS": "US Accidents",
    "MUL_TEXT_WINE_REVIEW": "Wine Review", "MUL_TEXT_WOMEN_CLOTHING_REVIEW": "Women Clothing",
    "REG_TEXT_BABIES_PRICES": "Babies Prices", "REG_TEXT_BOOK_PRICE": "Book Price",
    "REG_TEXT_BOOK_READABILITY": "Book Readability", "REG_TEXT_MERCARI_MARKETPLACE": "Mercari",
    "REG_TEXT_MONTGOMERY_SALARIES": "Montgomery Sal.", "REG_TEXT_SCIMAGOJR_IMPACT": "SCImago Impact",
    "REG_TEXT_ROTTEN_TOMATOES": "Rotten Tomatoes", "REG_TEXT_VANCOUVER_SALARIES": "Vancouver Sal.",
    "REG_TEXT_VIDEO_GAMES_SALES": "Video Games", "REG_TEXT_ZOMATO_RESTAURANTS": "Zomato",
}
OURS = [  # (method key, label)
    ("agent_table_only", "Reader, table only (0 reads)"),
    ("agent_read_all", "Reader, read all"),
    ("blend_catboost_agent", "\\textbf{Blend}: CatBoost Frozen + reader"),
]
LOCAL = [("catboost_tab_only", "CatBoost, tabular only"), ("catboost_frozen", "CatBoost, Joint Frozen")]
REF_MODELS = [("CatBoost", "CatBoost"), ("LightGBM", "LightGBM"), ("TabM", "TabM"), ("TabPFN-v2p5", "TabPFN-2.5")]
PRIMARY = "blend_catboost_agent"


def load(tag, results_dir=None):
    kw = {"results_dir": results_dir} if results_dir else {}
    df = evaluation_results(tag=tag, **kw)
    groups = df[GROUP].drop_duplicates()
    if len(groups) != 1:
        raise ValueError(f"expected one protocol/configuration, found {len(groups)}")
    return df


def wide_scores(df, lb):
    """dataset x fold x method score table; external references join as REF_<model>_<cond>."""
    ours = method_scores(df)
    wide = ours.pivot(index=["dataset", "task_group", "fold"], columns="method", values="score")
    ref = lb.copy()
    ref["method"] = "REF_" + ref.model + "_" + ref.cond
    ref = ref.pivot(index=["dataset", "fold"], columns="method", values="test_score")
    return wide.join(ref, on=["dataset", "fold"])


def per_dataset(wide):
    """Fold mean per dataset (only folds we completed), plus fold std and count."""
    g = wide.groupby(level=["dataset", "task_group"])
    return g.mean(), g.std(ddof=1), g.size()


def paired(wide, left, right, task):
    part = wide.xs(task, level="task_group")
    if left not in part or right not in part:
        return None
    d = part[[left, right]].dropna()
    if d.empty:
        return None
    deltas = (d[left] - d[right]).groupby(level="dataset").mean().rename("delta").reset_index()
    deltas["task_group"] = task
    return summarize_deltas(deltas)


def fmt(v, nd=3):
    return "--" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def fmt_signed(v, nd=3):
    return "--" if v is None or not np.isfinite(v) else f"{v:+.{nd}f}".replace("-", "$-$")


def fmt_p(p):
    if p is None or not np.isfinite(p):
        return "--"
    return "$<$0.001" if p < 1e-3 else f"{p:.3f}"


def record_wl(stats, star=True):
    if stats is None:
        return "--"
    losses = stats["n_datasets"] - stats["wins"] - stats["ties"]
    mark = "$^{*}$" if star and np.isfinite(stats["pvalue"]) and stats["pvalue"] < 0.05 else ""
    return f"{stats['wins']}/{stats['ties']}/{losses}{mark}"


def main_table(wide, means):
    """Mean score per task family; W/T/L of the blend against each row (dataset-level, folds averaged)."""
    rows = []
    def add(key, label, group):
        cells = [label]
        for task in (CLS, REG):
            m = means.xs(task, level="task_group")
            if key not in m:
                cells += ["--", "--"]
                continue
            col = m[key].dropna()
            n = len(col)
            # Partial coverage: also give the Blend's mean on the same datasets, so means are comparable.
            suffix = f"$^{{\\dagger{n}}}$\\,[{fmt(m.loc[col.index, PRIMARY].mean())}]" if 0 < n < len(m) else ""
            cells.append(fmt(col.mean()) + suffix if n else "--")
            # External references have unverified row identity: report W/T/L without a significance mark.
            cells.append("" if key == PRIMARY else record_wl(paired(wide, PRIMARY, key, task),
                                                             star=not key.startswith("REF_")))
        rows.append((group, " & ".join(cells) + " \\\\"))
    for k, lbl in OURS:
        add(k, lbl, "ours")
    for k, lbl in LOCAL:
        add(k, lbl, "local")
    for cond, cond_lbl in (("frozen", "Joint Frozen"), ("tar", "Joint TAR")):
        for model, name in REF_MODELS:
            add(f"REF_{model}_{cond}", f"{name}, {cond_lbl}", cond)
    head = ("\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n"
            " & \\multicolumn{2}{c}{Classification (10, AUROC)} & \\multicolumn{2}{c}{Regression (10, $R^2$)} \\\\\n"
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
            "Method & mean & Blend W/T/L & mean & Blend W/T/L \\\\\n\\midrule\n")
    blocks = {"ours": "\\multicolumn{5}{@{}l}{\\emph{Ours}} \\\\\n",
              "local": "\\midrule\n\\multicolumn{5}{@{}l}{\\emph{Local baselines (identical rows)}} \\\\\n",
              "frozen": "\\midrule\n\\multicolumn{5}{@{}l}{\\emph{MulTaBench reference, Joint Frozen (E5)}} \\\\\n",
              "tar": "\\midrule\n\\multicolumn{5}{@{}l}{\\emph{MulTaBench reference, Joint TAR (LoRA-tuned E5)}} \\\\\n"}
    body, seen = "", set()
    for group, line in rows:
        if group not in seen:
            body += blocks[group]
            seen.add(group)
        body += line + "\n"
    return head + body + "\\bottomrule\n\\end{tabular}\n"


def budget_table(rows):
    """Gain policy minus matched random reader, dataset-level, per key and nominal budget."""
    df = pd.DataFrame(rows)
    lines = []
    for key, key_lbl in (("content", "content"), ("meta", "metadata")):
        for budget in sorted(df.nominal_budget.unique()):
            cells = [key_lbl, f"{int(round(budget * 100))}\\%"]
            for task in (CLS, REG):
                r = df[(df.key == key) & (df.nominal_budget == budget) & (df.task == task)]
                if r.empty:
                    cells += ["--"] * 3
                    continue
                r = r.iloc[0]
                losses = r.n_datasets - r.wins - r.ties
                cells += [f"{100 * r.actual_fraction_mean:.0f}\\%", fmt_signed(r.mean_delta),
                          f"{r.wins}/{r.ties}/{losses}", fmt_p(r.holm_pvalue)]
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    lines.pop()
    head = ("\\begin{tabular}{@{}llcccccccc@{}}\n\\toprule\n"
            " & & \\multicolumn{4}{c}{Classification ($\\Delta$AUROC)} & \\multicolumn{4}{c}{Regression ($\\Delta R^2$)} \\\\\n"
            "\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}\n"
            "Keys & Budget & read & $\\Delta$ & W/T/L & $p_{\\text{Holm}}$ & read & $\\Delta$ & W/T/L & $p_{\\text{Holm}}$ \\\\\n\\midrule\n")
    return head + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"


def per_dataset_table(means, stds, counts):
    cols = [("catboost_tab_only", "CB tab"), ("catboost_frozen", "CB Frozen"), ("agent_table_only", "R: table"),
            ("gainB_content@0.25", "R: gain@25\\%"), ("agent_read_all", "R: all"), ("blend_catboost_agent", "Blend"),
            ("REF_CatBoost_frozen", "LB CB Frozen"), ("REF_CatBoost_tar", "LB CB TAR"), ("BEST_tar", "LB best TAR")]
    tar_cols = [c for c in means if c.startswith("REF_") and c.endswith("_tar")]
    means = means.copy()
    means["BEST_tar"] = means[tar_cols].max(axis=1)
    lines = []
    for task, title in ((CLS, "Classification (AUROC)"), (REG, "Regression ($R^2$)")):
        lines.append(f"\\midrule\n\\multicolumn{{{len(cols) + 2}}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\")
        m, s = means.xs(task, level="task_group"), stds.xs(task, level="task_group")
        n = counts.xs(task, level="task_group")
        for ds in m.index:
            vals = [m.loc[ds, k] if k in m else np.nan for k, _ in cols]
            ours = [vals[i] for i, (k, _) in enumerate(cols) if not k.startswith(("REF_", "BEST_"))]
            best = np.nanmax(ours)
            cells = [SHORT.get(ds, ds), str(int(n.loc[ds]))]
            for (k, _), v in zip(cols, vals):
                txt = fmt(v)
                if not k.startswith(("REF_", "BEST_")) and np.isfinite(v) and np.isclose(v, best):
                    txt = f"\\textbf{{{txt}}}"
                if k == "blend_catboost_agent" and k in s and np.isfinite(s.loc[ds, k]):
                    txt += f"\\tiny{{$\\pm${s.loc[ds, k]:.3f}}}"
                cells.append(txt)
            lines.append(" & ".join(cells) + " \\\\")
    head = ("\\begin{tabular}{@{}lc" + "c" * len(cols) + "@{}}\n\\toprule\n"
            "Dataset & folds & " + " & ".join(lbl for _, lbl in cols) + " \\\\\n")
    return head + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"


def budget_figure(budgets, means, path):
    """Mean score relative to read-all versus actual read fraction (dataset means; folds averaged)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    b = budgets.groupby(["dataset", "task_group", "key", "budget"])[["gain", "random", "frac"]].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.1))
    colors = {"content": "#2a6fbb", "meta": "#c0662a"}
    for ax, task, ylabel in zip(axes, (CLS, REG), ("AUROC $-$ read-all", "$R^2$ $-$ read-all")):
        m = means.xs(task, level="task_group")
        ref, base = m["agent_read_all"], m["agent_table_only"]
        zero = float((base - ref).mean())
        for key, lbl in (("content", "content keys"), ("meta", "metadata keys")):
            part = b[(b.task_group == task) & (b.key == key)]
            agg = part.assign(g=part.gain - part.dataset.map(ref), r=part.random - part.dataset.map(ref)) \
                      .groupby("budget")[["g", "r", "frac"]].mean()
            # Curves stop at the last measured budget: read-all is measured in natural order only,
            # and the order-sensitive reader need not reach the same score in policy or random order.
            x = np.r_[0.0, agg.frac.to_numpy()]
            ax.plot(x, np.r_[zero, agg.g.to_numpy()], "-o", ms=3, color=colors[key], label=f"gain, {lbl}")
            ax.plot(x, np.r_[zero, agg.r.to_numpy()], "--", marker="s", ms=2.5, color=colors[key], alpha=.55,
                    label=f"random, matched to {key}")
        ax.plot([1.0], [0.0], "*", ms=6, color="black", label="read all (natural order)")
        ax.axhline(0, color="#777", lw=.6)
        ax.set_xlabel("fraction of text items read")
        ax.set_ylabel(ylabel)
        ax.set_title("Classification (10 datasets)" if task == CLS else "Regression (10 datasets)", fontsize=8)
        ax.grid(alpha=.25, lw=.5)
    axes[0].legend(fontsize=6, frameon=False, loc="lower right")
    fig.tight_layout(pad=.4)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def recovery_macros(means, min_text_effect=0.02):
    """Scale-free view of selective reading, pooled over both families.

    Recovered share at budget b: (score@b - table only) / (read all - table only), for
    datasets where reading everything helps by at least `min_text_effect`; medians over
    datasets for the gain policy and its matched random reader, with a Wilcoxon test
    on their per-dataset difference. Also Spearman correlation, per family, between the
    gain-minus-random advantage at 10% (content keys) and the text effect itself.
    """
    from scipy.stats import spearmanr, wilcoxon
    out = {}
    effect = means["agent_read_all"] - means["agent_table_only"]
    keep = effect >= min_text_effect
    out["recN"] = int(keep.sum())
    pvals = []
    for key, kn in (("content", "Content"), ("meta", "Meta")):
        for b, bn in ((0.1, "Ten"), (0.25, "Quarter")):
            g = (means[f"gainB_{key}@{b:g}"] - means["agent_table_only"])[keep] / effect[keep]
            r = (means[f"randomB_{key}@{b:g}"] - means["agent_table_only"])[keep] / effect[keep]
            out[f"rec{kn}{bn}"] = f"{100 * g.median():.0f}\\%"
            out[f"recRandom{kn}{bn}"] = f"{100 * r.median():.0f}\\%"
            pvals.append(float(wilcoxon(g - r).pvalue))
    out["recMaxP"] = fmt_p(max(pvals))
    for task, tk in ((CLS, "Cls"), (REG, "Reg")):
        m = means.xs(task, level="task_group")
        rho, p = spearmanr(m["gainB_content@0.1"] - m["randomB_content@0.1"], m["agent_read_all"] - m["agent_table_only"])
        out[f"rhoTextEffect{tk}"] = f"{rho:.2f}"
        out[f"rhoTextEffect{tk}P"] = fmt_p(p)
    return out


def macros(df, wide, means, counts, brows):
    out = {}
    out["nRuns"] = df.analysis_run.nunique()
    folds = sorted(df.fold.unique())
    out["nFolds"] = len(folds)
    out["minFolds"] = int(counts.min())
    out["maxFolds"] = int(counts.max())
    for task, tk in ((CLS, "Cls"), (REG, "Reg")):
        m = means.xs(task, level="task_group")
        for key, name in (("blend_catboost_agent", "Blend"), ("agent_read_all", "ReadAll"),
                          ("catboost_frozen", "CBFrozen"), ("agent_table_only", "TableOnly")):
            out[f"mean{name}{tk}"] = fmt(m[key].mean())
        for ref, name in (("catboost_frozen", "LocalCB"), ("REF_CatBoost_frozen", "LbCBFrozen"),
                          ("REF_CatBoost_tar", "LbCBTar"), ("agent_read_all", "ReadAll")):
            st = paired(wide, PRIMARY, ref, task)
            if st is None:
                continue
            out[f"blendVs{name}{tk}Delta"] = fmt_signed(st["mean_delta"])
            out[f"blendVs{name}{tk}Wins"] = st["wins"]
            out[f"blendVs{name}{tk}P"] = fmt_p(st["pvalue"])
        for ref, name in (("REF_CatBoost_frozen", "LbCBFrozen"), ("REF_CatBoost_tar", "LbCBTar"), ("catboost_frozen", "LocalCB")):
            st = paired(wide, "agent_read_all", ref, task)
            if st is not None:
                out[f"readAllVs{name}{tk}Delta"] = fmt_signed(st["mean_delta"])
                out[f"readAllVs{name}{tk}Wins"] = st["wins"]
        b = pd.DataFrame(brows)
        for key, kn in (("content", "Content"), ("meta", "Meta")):
            for budget in (0.1, 0.25):
                r = b[(b.task == task) & (b.key == key) & np.isclose(b.nominal_budget, budget)]
                if r.empty:
                    continue
                r = r.iloc[0]
                bn = "Ten" if budget == 0.1 else "Quarter"
                out[f"gain{kn}{bn}{tk}Delta"] = fmt_signed(r.mean_delta)
                out[f"gain{kn}{bn}{tk}Wins"] = int(r.wins)
                out[f"gain{kn}{bn}{tk}P"] = fmt_p(r.holm_pvalue)
    out.update(recovery_macros(means))
    # per-dataset fold means, e.g. \sVideoGamesReadAll
    tar_cols = [c for c in means if c.startswith("REF_") and c.endswith("_tar")]
    frozen_cols = [c for c in means if c.startswith("REF_") and c.endswith("_frozen")]
    for (ds, _), row in means.iterrows():
        name = "".join(ch for ch in SHORT.get(ds, ds).title() if ch.isalpha())
        for key, suffix in (("agent_read_all", "ReadAll"), ("catboost_frozen", "CBFrozen"), ("blend_catboost_agent", "Blend"),
                            ("agent_table_only", "TableOnly"), ("catboost_tab_only", "CBTab"),
                            ("REF_CatBoost_tar", "LbCBTar"), ("REF_TabM_frozen", "LbTabMFrozen")):
            if key in row:
                out[f"s{name}{suffix}"] = fmt(row[key])
        out[f"s{name}LbBestTar"] = fmt(row[tar_cols].max())
        out[f"s{name}LbBestFrozen"] = fmt(row[frozen_cols].max())
    return "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in out.items())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="paper_v2")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--leaderboard", default=DEFAULT_LB)
    args = ap.parse_args(argv)
    df = load(args.tag)
    lb = load_leaderboard(args.leaderboard)
    wide = wide_scores(df, lb)
    means, stds, counts = per_dataset(wide)
    brows = budget_report(df)
    budgets = budget_comparison(df)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "tab_main.tex").write_text(main_table(wide, means), encoding="utf-8")
    (args.out / "tab_budget.tex").write_text(budget_table(brows), encoding="utf-8")
    (args.out / "tab_per_dataset.tex").write_text(per_dataset_table(means, stds, counts), encoding="utf-8")
    (args.out / "numbers.tex").write_text(macros(df, wide, means, counts, brows), encoding="utf-8")
    budget_figure(budgets, means, args.out / "fig_budget.pdf")
    print(f"wrote paper assets for {df.analysis_run.nunique()} runs, folds {sorted(df.fold.unique())} -> {args.out}")


if __name__ == "__main__":
    main()
