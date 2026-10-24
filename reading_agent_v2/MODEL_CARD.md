---
library_name: pytorch
license: mit
tags:
- tabular-classification
- tabular-regression
- text-tabular
- multabench
- lora
base_model: intfloat/e5-small-v2
---

# Reading Agent v2: table-conditioned text reading with TabLoRA (MulTaBench)

Trained models for *Reading What Matters: Table-Conditioned Text Reading for Text-Tabular
Prediction* (NLP 097215 final project, Track 1, Technion, 2026).
Code: <https://github.com/htmldanielpoly/nlp_final_project> (folder `reading_agent_v2`);
a source snapshot is in this repo under `code/`.

For each row, a table encoder initialises a recurrent reader that consumes field-tagged
sentence items of the row's text fields. Items are encoded by `intfloat/e5-small-v2`
(revision `ffb93f3bd4047442299a41ebb6fa998a38507c52`) whose last three layers carry
**TabLoRA** adapters: LoRA (r=16, alpha=32) on query, key, value, attention output and both
feed-forward maps, with the rank space gated per row by the table encoding,
`W x + (alpha/r) B diag(1 + tanh(G h_tab)) A x`. Three readers are cross-fitted per split.
The prediction is the mean of two validation-tuned blends of the reader with Joint-Frozen
**TabM** and **TabPFN-2.5** (structured columns + PCA-30 of each field's whole-text E5
embedding); beyond 10 classes only the TabM blend is used.

## Contents

One folder per run: 20 MulTaBench text datasets x 5 paper holdout folds (`<dataset>_f<fold>`).

```text
index.json            per run: task, blend weights, data revision, test-row hash, test score
<run>/manifest.json   task, classes, target scaling, blend weights, TabM input columns, metadata
<run>/preprocessing.pkl  fitted feature pipeline and the frozen frame's PCAs (pickle: load only trusted files)
<run>/reader_{0,1,2}.pt  cross-fitted readers: reader, LoRA and gate parameters (frozen E5 is rebuilt from the checkpoint)
<run>/tabm.pt         Joint-Frozen TabM (pytabkit)
<run>/context.parquet TabPFN-2.5 context: the frozen-frame training rows and their (encoded) targets
code/                 source snapshot (package, tests, scripts, pyproject.toml, uv.lock)
```

TabPFN-2.5 is a pretrained in-context learner, so its "fitted model" is its context; it is
re-conditioned on `context.parquet` at inference. Its weights are downloaded from Prior Labs and
need a licence token (`TABPFN_TOKEN`).

## Inference mode

```bash
uv sync --locked      # in code/ or in the GitHub folder reading_agent_v2
PYTHONPATH=src python -m reading_agent2.estimator2 predict --repo htmldaniel/reading-agent-v2-multabench \
    --dataset REG_TEXT_BOOK_PRICE --fold 0 --input rows.csv --output preds.csv
```

The input needs the dataset's feature columns in the MulTaBench representation.
Classification output: predicted label and one probability per class; regression output: the
target in its original units. GPU recommended (CPU works but is slow).
`python -m reading_agent2.estimator2 check --dataset <D> --fold <k> --repo htmldaniel/reading-agent-v2-multabench`
replays a run's stored test predictions from raw rows.

## End-to-end training mode

```bash
bash scripts/train_all.sh     # E5 cache, frozen frame, TabLoRA readers, heads, export (one CUDA GPU)
```

Data: MulTaBench Hugging Face datasets at the revisions pinned in `reading_agent2/revisions.py`
(recorded per run in `index.json`). GPU training is not bit-identical across runs; retrained
weights reproduce the scores closely, not exactly.

## Evaluation

MulTaBench paper protocol: 20 text datasets x 5 random holdout splits (seed 42+fold), dataset means
over folds, then means over the 10 classification (AUROC) and 10 regression (R^2) datasets.
Local rows are run on exactly the test rows of these artifacts; p: two-sided Wilcoxon signed-rank
test of Ours against that row over datasets.

| Method | AUROC | Ours wins (p) | R^2 | Ours wins (p) |
|---|---|---|---|---|
| **Ours** (mean of the two blends, this repo) | **0.913** | | **0.738** | |
| TabM Frozen + TabLoRA reader | 0.910 | 8/10 (0.008) | 0.724 | 9/10 (0.004) |
| TabPFN-2.5 Frozen + TabLoRA reader | 0.897 (8 datasets) | 4/8 | 0.734 | 4/10 |
| TabLoRA reader alone | 0.903 | 10/10 (0.002) | 0.672 | 10/10 (0.002) |
| v1 Blend (CatBoost Frozen + frozen-item reader) | 0.900 | 9/10 (0.004) | 0.702 | 10/10 (0.002) |
| CatBoost, Joint Frozen (local) | 0.887 | 10/10 (0.002) | 0.684 | 10/10 (0.002) |
| TabM, Joint Frozen (local) | 0.895 | 10/10 (0.002) | 0.716 | 10/10 (0.002) |
| TabPFN-2.5, Joint Frozen (local) | 0.884 (8 datasets) | 6/8 (0.148) | 0.729 | 5/10 (0.375) |
| CatBoost, Joint TAR (leaderboard) | 0.907 | 9/10 | 0.716 | 9/10 |
| TabM, Joint TAR (leaderboard) | 0.911 | 5/10 | 0.728 | 9/10 |
| TabPFN-2.5, Joint TAR (leaderboard) | 0.900 (8 datasets) | 3/8 | 0.732 | 4/10 |

TabPFN-2.5 is not used beyond 10 classes (Spotify, Wine). Per-dataset and per-fold scores:
`code/evaluation/scores_per_dataset.csv`, `code/evaluation/scores_per_fold.csv`; the per-run test
score is also in `index.json`.

External leaderboard values use the same split seeds, but row identity with the original data
is not verified, so no significance test is claimed against them.

## Limitations

- Per-dataset models: each artifact is trained for one dataset and split; for new data use
  `TabLoRAStack.fit` (see the GitHub README).
- TabPFN-2.5 needs its training rows (stored here as the frozen-frame context) at inference.
- No test leakage: every reader, adapter, preprocessing step, head and blend weight is fit per
  dataset and split on that split's training rows only. Pretrained parts: E5-small-v2 (generic
  web text) and TabPFN-2.5, whose default *classifier* checkpoint is additionally fine-tuned on 43
  real datasets (TabPFN-2.5 report, App. C.1), none of them a MulTaBench dataset by name.
- The per-row table gate of TabLoRA was level with plain LoRA in our fold-0 ablation; the gain
  over frozen items comes from the adapters.
- Items are truncated at 128 tokens for TabLoRA, and at most `K = max(8, #fields)` items per row
  are kept, so very long texts are only partly read.
