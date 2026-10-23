"""Tables and number macros for the v2 paper (results/v2.jsonl + the v1 paper_v2 runs on identical rows).

Folds are averaged within each dataset before any cross-dataset summary; Wilcoxon signed-rank
tests run over datasets, separately for AUROC and R^2 (as in v1). v1 rows and v2 rows are the
same test rows (checked by `--check_rows`). External leaderboard rows get W/T/L but no test.

    python -m reading_agent2.paper_assets2 --out paper/generated
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reading_agent.compare import load_leaderboard, method_scores, summarize_deltas, task_group
from reading_agent.eval_paper import evaluation_results
from reading_agent.paper_assets import CLS, REG, SHORT, fmt, fmt_p, record_wl

from .stack import PRIMARY

ROOT = Path(__file__).resolve().parents[2]
V1_RESULTS = ROOT.parent / "reading_agent" / "results"
V1 = {"agent_read_all": "v1_read_all", "blend_catboost_agent": "v1_blend", "catboost_frozen": "cb_frozen",
      "catboost_tab_only": "cb_tab_only"}
REF_MODELS = [("CatBoost", "CatBoost"), ("LightGBM", "LightGBM"), ("TabM", "TabM"), ("TabPFN-v2p5", "TabPFN-2.5")]


def fmt_signed(v, nd=3):
    """Signed difference; a value that rounds to zero prints as 0.000, never as -0.000."""
    if v is None or not np.isfinite(v):
        return "--"
    v = round(float(v), nd) + 0.0
    return ("0." + "0" * nd) if v == 0 else f"{v:+.{nd}f}".replace("-", "$-$")


def load_v2(path):
    df = pd.read_json(path, lines=True)
    df = df.drop_duplicates(["dataset", "fold", "method"], keep="last")
    df["task_group"] = df.apply(task_group, axis=1)
    return df[["dataset", "task_group", "fold", "method", "score"]]


def load_v1():
    df = method_scores(evaluation_results(tag="paper_v2", results_dir=V1_RESULTS))
    df = df[df.method.isin(V1)].copy()
    df["method"] = df.method.map(V1)
    return df


def load_info(tag):
    """Per-run info.json of a reader-feature stage (reader test score, runtime, gate norms)."""
    rows = []
    for p in sorted((ROOT / "results" / tag).glob("*/info.json")):
        i = json.loads(p.read_text(encoding="utf-8"))
        rows.append(dict(dataset=i["dataset"], fold=i["fold"], task=i["task"], score=i["reader_test_score"],
                         minutes=i.get("minutes"), gate_norm=np.mean(i["gate_weight_norms"]) if i.get("gate_weight_norms") else np.nan))
    df = pd.DataFrame(rows)
    if len(df):
        df["task_group"] = df.apply(task_group, axis=1)
    return df


def build_wide(v2, v1, lb, nogate):
    long = pd.concat([v2, v1], ignore_index=True)
    if len(nogate):
        long = pd.concat([long, nogate.assign(method="reader_nogate@stackB")[long.columns]], ignore_index=True)
    wide = long.pivot_table(index=["dataset", "task_group", "fold"], columns="method", values="score")
    ref = lb.copy()
    ref["method"] = "REF_" + ref.model + "_" + ref.cond
    ref = ref.pivot(index=["dataset", "fold"], columns="method", values="test_score")
    wide = wide.join(ref, on=["dataset", "fold"])
    tar = [f"REF_{m}_tar" for m, _ in REF_MODELS if f"REF_{m}_tar" in wide]
    wide["REF_best_tar"] = wide[tar].max(axis=1)
    return wide


def paired(wide, left, right, task, folds=None):
    part = wide.xs(task, level="task_group")
    if folds is not None:
        part = part[part.index.get_level_values("fold").isin(folds)]
    if left not in part or right not in part:
        return None
    d = part[[left, right]].dropna()
    if d.empty:
        return None
    deltas = (d[left] - d[right]).groupby(level="dataset").mean().rename("delta").reset_index()
    deltas["task_group"] = task
    return summarize_deltas(deltas)


ROWS = [  # (group, key, label)
    ("ours", "reader_crossfit_all@stack", "Reader, frozen E5 items"),
    ("ours", "reader_crossfit_all@stackB", "Reader, TabLoRA items"),
    ("ours", "tabm_blend@stackB", "TabM Frozen + TabLoRA reader"),
    ("ours", "tabpfn_blend@stackB", "TabPFN-2.5 Frozen + TabLoRA reader"),
    ("ours", PRIMARY, "\\textbf{Ours}: mean of the two blends"),
    ("local", "v1_blend", "v1 Blend: CatBoost Frozen + reader"),
    ("local", "cb_frozen", "CatBoost, Joint Frozen"),
    ("local", "tabm_frozen", "TabM, Joint Frozen"),
    ("local", "tabpfn_frozen", "TabPFN-2.5, Joint Frozen"),
] + [("tar", f"REF_{m}_tar", f"{name}, Joint TAR") for m, name in REF_MODELS]
BLOCKS = {"ours": "\\multicolumn{5}{@{}l}{\\emph{Ours (cross-fitted readers, local heads)}} \\\\\n",
          "local": "\\midrule\n\\multicolumn{5}{@{}l}{\\emph{Local baselines (identical rows)}} \\\\\n",
          "tar": "\\midrule\n\\multicolumn{5}{@{}l}{\\emph{MulTaBench reference, Joint TAR (LoRA-tuned E5)}} \\\\\n"}


def main_table(wide, means):
    body, seen = "", set()
    for group, key, label in ROWS:
        cells = [label]
        for task in (CLS, REG):
            m = means.xs(task, level="task_group")
            col = m[key].dropna() if key in m else pd.Series(dtype=float)
            n = len(col)
            suffix = f"$^{{\\dagger{n}}}$\\,[{fmt(m.loc[col.index, PRIMARY].mean())}]" if 0 < n < len(m) else ""
            cells.append(fmt(col.mean()) + suffix if n else "--")
            cells.append("" if key == PRIMARY else record_wl(paired(wide, PRIMARY, key, task), star=not key.startswith("REF_")))
        if group not in seen:
            body += BLOCKS[group]
            seen.add(group)
        body += " & ".join(cells) + " \\\\\n"
    head = ("\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n"
            " & \\multicolumn{2}{c}{Classification (10, AUROC)} & \\multicolumn{2}{c}{Regression (10, $R^2$)} \\\\\n"
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
            "Method & mean & Ours W/T/L & mean & Ours W/T/L \\\\\n\\midrule\n")
    return head + body + "\\bottomrule\n\\end{tabular}\n"


ABLATIONS = [  # (label, left, right, folds)
    ("TabLoRA vs.\\ frozen-E5 items, reader alone", "reader_crossfit_all@stackB", "reader_crossfit_all@stack", None),
    ("TabLoRA vs.\\ frozen-E5 items, final system", PRIMARY, "blend_mean@stack", None),
    ("Table gate vs.\\ plain LoRA, reader (fold 0)", "reader_crossfit_all@stackB", "reader_nogate@stackB", (0,)),
    ("Adding the reader to TabM Frozen", "tabm_blend@stackB", "tabm_frozen", None),
    ("Adding the reader to TabPFN-2.5 Frozen", "tabpfn_blend@stackB", "tabpfn_frozen", None),
    ("Blend vs.\\ stacking (TabM)", "tabm_blend@stackB", "tabm_stack@stackB", None),
    ("Two heads vs.\\ TabM only", PRIMARY, "tabm_blend@stackB", None),
]


def cell(stats):
    if stats is None or stats["n_datasets"] == 0:
        return ["--", "--", "--"]
    return [fmt_signed(stats["mean_delta"]), f"{stats['wins']}/{stats['n_datasets']}", fmt_p(stats["pvalue"])]


def ablation_table(wide):
    lines = []
    for label, left, right, folds in ABLATIONS:
        lines.append(" & ".join([label] + cell(paired(wide, left, right, CLS, folds)) + cell(paired(wide, left, right, REG, folds))) + " \\\\")
    head = ("\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n"
            " & \\multicolumn{3}{c}{Classification (AUROC)} & \\multicolumn{3}{c}{Regression ($R^2$)} \\\\\n"
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
            "Comparison (left $-$ right) & $\\Delta$ & wins & $p$ & $\\Delta$ & wins & $p$ \\\\\n\\midrule\n")
    return head + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"


PER_COLS = [("v1_blend", "v1"), ("tabm_frozen", "TabM"), ("tabpfn_frozen", "PFN"), ("reader_crossfit_all@stack", "R"),
            ("reader_crossfit_all@stackB", "R$_{\\text{LoRA}}$"), (PRIMARY, "Ours"), ("REF_TabM_tar", "TabM"), ("REF_best_tar", "best")]


def per_dataset_table(means, stds):
    lines = []
    for task in (CLS, REG):
        m, s = means.xs(task, level="task_group"), stds.xs(task, level="task_group")
        lines.append(f"\\midrule\n\\multicolumn{{{len(PER_COLS) + 1}}}{{@{{}}l}}{{\\emph{{{'Classification (AUROC)' if task == CLS else 'Regression ($R^2$)'}}}}} \\\\")
        for ds in m.index:
            ours = [k for k, _ in PER_COLS if not k.startswith("REF_")]
            best = m.loc[ds, [k for k in ours if k in m]].max()
            cells = [SHORT.get(ds, ds)]
            for k, _ in PER_COLS:
                v = m.loc[ds, k] if k in m else np.nan
                txt = fmt(v)
                if k == PRIMARY and np.isfinite(s.loc[ds, k]):
                    txt += f"\\,$\\pm$\\,{s.loc[ds, k]:.3f}"
                if not k.startswith("REF_") and np.isfinite(v) and np.isclose(v, best):
                    txt = f"\\textbf{{{txt}}}"
                cells.append(txt)
            lines.append(" & ".join(cells) + " \\\\")
    head = ("\\begin{tabular}{@{}l" + "c" * len(PER_COLS) + "@{}}\n\\toprule\n"
            " & \\multicolumn{6}{c}{Ours and local baselines (identical rows)} & \\multicolumn{2}{c}{LB Joint TAR} \\\\\n"
            "\\cmidrule(lr){2-7}\\cmidrule(lr){8-9}\n"
            "Dataset & " + " & ".join(lbl for _, lbl in PER_COLS) + " \\\\\n")
    return head + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"


def camel(s):
    return "".join(w[:1].upper() + w[1:] for w in "".join(c if c.isalpha() else " " for c in s).split())


def macros(wide, means, info_b):
    out = []
    def put(name, value):
        out.append(f"\\newcommand{{\\{name}}}{{{value}}}")
    def delta(prefix, left, right, folds=None):
        for task, fam in ((CLS, "Cls"), (REG, "Reg")):
            st = paired(wide, left, right, task, folds)
            if st is None or st["n_datasets"] == 0:
                continue
            put(f"{prefix}{fam}Delta", fmt_signed(st["mean_delta"]))
            put(f"{prefix}{fam}Wins", f"{st['wins']}")
            put(f"{prefix}{fam}N", f"{st['n_datasets']}")
            put(f"{prefix}{fam}P", fmt_p(st["pvalue"]))
    for task, fam in ((CLS, "Cls"), (REG, "Reg")):
        m = means.xs(task, level="task_group")
        for key, name in ((PRIMARY, "ours"), ("v1_blend", "voneBlend"), ("REF_TabM_tar", "tabmTar"),
                          ("REF_CatBoost_tar", "cbTar"), ("REF_best_tar", "bestTar")):
            if key in m:
                put(f"{name}{fam}", fmt(m[key].mean()))
    delta("oursVsVone", PRIMARY, "v1_blend")
    delta("noPilotVsVone", PRIMARY, "v1_blend", (1, 2, 3, 4))  # without fold 0, which the pilots used
    delta("noPilotVsTabM", PRIMARY, "tabm_frozen", (1, 2, 3, 4))
    delta("oursVsCB", PRIMARY, "cb_frozen")
    delta("oursVsTabM", PRIMARY, "tabm_frozen")
    delta("oursVsPFN", PRIMARY, "tabpfn_frozen")
    for m, name in REF_MODELS:
        delta(f"oursVsLb{camel(name)}Tar", PRIMARY, f"REF_{m}_tar")
    delta("oursVsLbBestTar", PRIMARY, "REF_best_tar")
    delta("loraReader", "reader_crossfit_all@stackB", "reader_crossfit_all@stack")
    delta("loraSystem", PRIMARY, "blend_mean@stack")
    delta("gate", "reader_crossfit_all@stackB", "reader_nogate@stackB", (0,))
    delta("readerAddsTabM", "tabm_blend@stackB", "tabm_frozen")
    delta("readerAddsPFN", "tabpfn_blend@stackB", "tabpfn_frozen")
    delta("blendVsStack", "tabm_blend@stackB", "tabm_stack@stackB")
    delta("twoHeads", PRIMARY, "tabm_blend@stackB")
    per = means.droplevel("task_group")
    for ds, short in SHORT.items():
        if ds not in per.index:
            continue
        for key, name in ((PRIMARY, "Ours"), ("reader_crossfit_all@stack", "R"), ("reader_crossfit_all@stackB", "RLora"),
                          ("v1_blend", "Vone"), ("REF_best_tar", "BestTar"), ("tabm_frozen", "TabM")):
            if key in per and np.isfinite(per.loc[ds, key]):
                put(f"s{camel(short)}{name}", fmt(per.loc[ds, key]))
    if len(info_b):
        put("loraMinutes", f"{info_b.minutes.mean():.0f}")
        put("loraRuns", f"{len(info_b)}")
    folds = wide.index.get_level_values("fold")
    put("vtwoFolds", f"{folds.nunique()}")
    put("vtwoRuns", f"{wide[PRIMARY].notna().sum()}")
    return "\n".join(out) + "\n"


def check_rows(v1_dir=V1_RESULTS / "preds" / "paper_v2", v2_dir=ROOT / "results" / "preds" / "v2"):
    """v2 and v1 must score the same test rows with the same labels."""
    bad = []
    for p in sorted(v2_dir.glob("*.npz")):
        ds, fold = p.stem.rsplit("_f", 1)
        hits = sorted(v1_dir.glob(f"{ds}_f{fold}_*.npz"))
        a, b = np.load(p), np.load(hits[0]) if hits else None
        if b is None or not (np.array_equal(a["te_idx"], b["te_idx"]) and np.allclose(a["y_te"], b["y_te"])):
            bad.append(p.stem)
    print(f"row check: {len(list(v2_dir.glob('*.npz'))) - len(bad)} runs identical to v1, mismatches: {bad}")
    return bad


CSV_COLS = {PRIMARY: "ours_v2", "tabm_blend@stackB": "tabm_frozen_plus_tablora_reader",
            "tabpfn_blend@stackB": "tabpfn25_frozen_plus_tablora_reader", "reader_crossfit_all@stackB": "tablora_reader",
            "reader_crossfit_all@stack": "frozen_item_reader", "v1_blend": "v1_blend", "cb_frozen": "catboost_frozen_local",
            "tabm_frozen": "tabm_frozen_local", "tabpfn_frozen": "tabpfn25_frozen_local",
            **{f"REF_{m}_tar": f"lb_{m.lower().replace('-', '').replace('.', '')}_tar" for m, _ in REF_MODELS}}


def write_csv(wide, means, out):
    """Evaluation report: per (dataset, fold) and fold-mean per dataset (AUROC or R^2; lb_* = leaderboard)."""
    out.mkdir(parents=True, exist_ok=True)
    cols = [c for c in CSV_COLS if c in wide]
    for frame, name in ((wide, "scores_per_fold.csv"), (means, "scores_per_dataset.csv")):
        t = frame[cols].rename(columns=CSV_COLS).reset_index()
        t.insert(2, "metric", np.where(t.task_group == CLS, "AUROC", "R2"))
        t.drop(columns="task_group").round(4).to_csv(out / name, index=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(ROOT / "results" / "v2.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "paper" / "generated"))
    ap.add_argument("--check_rows", type=int, default=1)
    ap.add_argument("--csv_out", default=str(ROOT / "evaluation"))
    args = ap.parse_args(argv)
    if args.check_rows and check_rows():
        raise SystemExit("v2 rows differ from v1")
    info_b, info_ng = load_info("stackB"), load_info("stackB_nogate")
    nogate = info_ng[["dataset", "task_group", "fold", "score"]] if len(info_ng) else info_ng
    wide = build_wide(load_v2(args.results), load_v1(), load_leaderboard(), nogate)
    wide = wide[wide[PRIMARY].notna()]  # only (dataset, fold) pairs the v2 system finished
    g = wide.groupby(level=["dataset", "task_group"])
    means, stds = g.mean(), g.std(ddof=1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tab_main2.tex").write_text(main_table(wide, means), encoding="utf-8")
    (out / "tab_ablation2.tex").write_text(ablation_table(wide), encoding="utf-8")
    (out / "tab_per_dataset2.tex").write_text(per_dataset_table(means, stds), encoding="utf-8")
    (out / "numbers2.tex").write_text(macros(wide, means, info_b), encoding="utf-8")
    write_csv(wide, means, Path(args.csv_out))
    print(f"{len(wide)} dataset-folds ({wide.index.get_level_values('dataset').nunique()} datasets) -> {out}")
    for task in (CLS, REG):
        m = means.xs(task, level="task_group")
        print(task, {k: round(m[k].mean(), 4) for k in (PRIMARY, "v1_blend", "REF_TabM_tar", "REF_best_tar") if k in m})


if __name__ == "__main__":
    main()
