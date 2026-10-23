# Reading Agent v2: table-conditioned text reading (TabLoRA)

Code for *Reading What Matters: Table-Conditioned Text Reading for Text-Tabular
Prediction* (NLP 097215 final project, Track 1, Technion, 2026; paper in `paper/`).
v2 extends the v1 reading agent (`../reading_agent`, unchanged) with

- **TabLoRA**: LoRA adapters (r=16, alpha=32) on the query, key, value, attention-output and
  both feed-forward maps of E5-small-v2's last three layers, whose rank space is gated per row
  by the reader's table encoding, `W x + (alpha/r) B diag(1 + tanh(G h_tab)) A x`, with `B` and
  `G` zero-initialised. It is trained through the anytime reader's prefix loss.
- **Cross-fitted reader signals**: three readers per split, each trained on two thirds of the
  training rows, so every training row has an out-of-fold reader prediction.
- **Heads**: Joint-Frozen TabM and TabPFN-2.5 (structured columns + PCA-30 of each field's E5
  embedding), each blended with the reader using a weight tuned on held-out training rows.
  The prediction is the mean of the two blends (`blend_mean@stackB`).

Package `src/reading_agent` is a copy of v1 (also needed to load v1 artifacts); `src/reading_agent2`
holds the v2 code.

| Module | Role |
|---|---|
| `reading_agent2/stack.py` | `features`: Joint-Frozen frame and frozen-item cross-fitted readers (stage A); `heads`: TabM / TabPFN-2.5, blends, scores |
| `reading_agent2/tabcond.py` | TabLoRA: cached layer-10 inputs, gated LoRA, two-phase cross-fitted training (stage B) |
| `reading_agent2/estimator2.py` | `TabLoRAStack` (fit / predict / save / load on any DataFrame), `export`, `check`, `publish`, inference CLI |
| `reading_agent2/paper_assets2.py` | the paper's v2 tables and number macros |
| `reading_agent2/revisions.py` | pinned MulTaBench dataset revisions (the rows of the v1 runs) |

## Install

```bash
uv sync --locked          # Python 3.11; Linux installs CUDA 12.6 torch wheels
```

TabPFN-2.5 downloads its weights on first use and needs a Prior Labs licence token in
`TABPFN_TOKEN`. Training needs a CUDA GPU (runs used one T4, 16 GB); inference also runs on CPU.

## Inference mode

Download one benchmark run from the Hub and predict a CSV with the dataset's columns:

```bash
PYTHONPATH=src python -m reading_agent2.estimator2 predict --repo htmldaniel/reading-agent-v2-multabench \
    --dataset REG_TEXT_BOOK_PRICE --fold 0 --input rows.csv --output preds.csv
```

or, from Python, `TabLoRAStack.load(dir).predict(frame)`. Classification output has the predicted
label and one probability per class; regression output is in the target's units.
`python -m reading_agent2.estimator2 check --dataset <D> --fold <k> [--repo <repo>]` replays a
run's stored test predictions from raw rows.

## End-to-end training mode

```bash
bash scripts/train_all.sh                      # all 20 datasets x 5 folds, then export
python -m reading_agent2.estimator2 publish --repo <user>/<repo>
```

`train_all.sh` runs, per dataset and fold: `stack features` (frozen frame), `tabcond` (three
TabLoRA readers), `stack heads` (TabM, TabPFN-2.5, blend weights, test scores in
`results/v2.jsonl`) and `export` (artifact in `results/models/v2/<dataset>_f<fold>/`). For a new
text-tabular dataset, use the estimator directly:

```python
from reading_agent2.estimator2 import TabLoRAStack
model = TabLoRAStack(num_cols, cat_cols, text_cols, task="regression", namespace="e5_mydata").fit(train_df, y)
model.save("my_artifact"); model.predict(test_df)
```

## Evaluation protocol

All 20 MulTaBench text datasets, paper protocol (random holdout, seed 42+fold, test <= 2,000 and
train <= 10,000 rows), folds 0-4, on the dataset revisions of the v1 runs (`revisions.py`;
`paper_assets2 --check_rows` confirms that v1 and v2 score identical test rows). Folds are
averaged per dataset; two-sided Wilcoxon signed-rank tests run over datasets, separately for
AUROC and R². External MulTaBench leaderboard values are references without significance tests
(row identity unverified).

```bash
python -m reading_agent2.paper_assets2         # tables + macros -> paper/generated
bash scripts/build_paper.sh                    # PDF, checks the 4-page limit
python -m unittest discover -s tests -v
```

## Results

See `paper/main.pdf` (Tables 1-2, Appendix A) and `MODEL_CARD.md`.
