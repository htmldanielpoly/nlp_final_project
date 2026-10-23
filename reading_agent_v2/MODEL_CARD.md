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

RESULTS_TABLE

External leaderboard values use the same split seeds, but row identity with the original data
is not verified, so no significance test is claimed against them.

## Limitations

- Per-dataset models: each artifact is trained for one dataset and split; for new data use
  `TabLoRAStack.fit` (see the GitHub README).
- TabPFN-2.5 needs its training rows (stored here as the frozen-frame context) at inference.
- Items are truncated at 128 tokens for TabLoRA, and at most `K = max(8, #fields)` items per row
  are kept, so very long texts are only partly read.
