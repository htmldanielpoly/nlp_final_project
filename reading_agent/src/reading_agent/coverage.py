"""Text coverage of the item representation, per dataset (paper protocol, fold-0 test rows).

Reports text fields, candidate chunks per row before the item cap, items kept, the
share of rows whose chunks exceed the cap, and the share of words inside kept
items. Chunking only; no embeddings are loaded. A field with more than
`max_chunks` chunks has its tail merged into the last chunk, which E5 truncates at
512 tokens, so word coverage is an upper bound on what the encoder sees.

    python -m reading_agent.coverage [--out results/coverage.csv] [--tex ../paper/generated/tab_coverage.tex]
    python -m reading_agent.coverage --from_csv results/coverage.csv --tex ../paper/generated/tab_coverage.tex
"""
import argparse
import gc
import pathlib

import numpy as np
import pandas as pd

from .chunking import chunk_text
from .data import TEXT_DATASETS, get_split, load_dataset
from .features import FeaturePipeline, _is_missing


def dataset_coverage(name, fold=0):
    ds = load_dataset(name)
    _, te = get_split(ds.y, ds.task != "regression", fold, "paper")
    texts = ds.x.iloc[te][ds.text_cols]
    del ds
    gc.collect()
    pipe = FeaturePipeline([], [], list(texts.columns), name, max_items=max(8, texts.shape[1]))
    cand, kept, words, kept_words = [], [], [], []
    for row in texts.itertuples(index=False):
        n_cand = sum(len(chunk_text(t, pipe.max_words, pipe.max_chunks)) for t in row if not _is_missing(t))
        items = pipe._row_items(list(row))
        cand.append(n_cand)
        kept.append(len(items))
        words.append(sum(len(str(t).split()) for t in row if not _is_missing(t)))
        kept_words.append(sum(len(text.split(": ", 2)[2].split()) for text, _, _ in items))
    cand, kept, words, kept_words = map(np.asarray, (cand, kept, words, kept_words))
    return {"dataset": name, "rows": len(texts), "text_fields": texts.shape[1], "K": pipe.max_items,
            "words_per_row": words.mean(), "chunks_per_row": cand.mean(), "items_per_row": kept.mean(),
            "rows_over_cap": float((cand > pipe.max_items).mean()),
            "word_coverage": float(kept_words.sum() / max(1, words.sum()))}


def latex(df):
    from .paper_assets import SHORT
    lines = []
    for prefix, title in (("BIN|MUL", "Classification"), ("REG", "Regression")):
        part = df[df.dataset.str.match(f"^({prefix})_")]
        lines.append("\\midrule\n\\multicolumn{7}{@{}l}{\\emph{" + title + "}} \\\\")
        for r in part.itertuples():
            lines.append(f"{SHORT.get(r.dataset, r.dataset)} & {r.text_fields} & {r.K} & {r.words_per_row:.0f} & "
                         f"{r.items_per_row:.1f} & {100 * r.rows_over_cap:.0f}\\% & {100 * r.word_coverage:.0f}\\% \\\\")
    head = ("\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n"
            "Dataset & text fields & $K$ & words/row & items/row & rows over $K$ & words kept \\\\\n")
    return head + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="results/coverage.csv")
    ap.add_argument("--datasets", default="")
    ap.add_argument("--tex", default=None, help="also write a LaTeX table")
    ap.add_argument("--from_csv", default=None, help="only render --tex from an existing CSV")
    args = ap.parse_args(argv)
    if args.from_csv:
        pathlib.Path(args.tex).write_text(latex(pd.read_csv(args.from_csv)), encoding="utf-8")
        return
    names = [n for n in args.datasets.split(",") if n] or sorted(TEXT_DATASETS)
    rows = []
    for name in names:
        rows.append(dataset_coverage(name))
        print(pd.DataFrame(rows[-1:]).to_string(index=False, header=len(rows) == 1, float_format=lambda v: f"{v:.2f}"), flush=True)
        gc.collect()
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {args.out}")
    if args.tex:
        pathlib.Path(args.tex).write_text(latex(pd.DataFrame(rows)), encoding="utf-8")


if __name__ == "__main__":
    main()
