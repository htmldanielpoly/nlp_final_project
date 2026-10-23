---
library_name: pytorch
license: mit
tags:
- tabular-classification
- tabular-regression
- text-tabular
- multabench
- selective-reading
base_model: intfloat/e5-small-v2
---

# Reading Agent: table-conditioned selective text reading (MulTaBench)

Trained weights for *Reading What Matters: Table-Conditioned Selective Text Reading
for Text-Tabular Prediction* (NLP 097215 final project, Track 1, Technion, 2026).
Code: <https://github.com/htmldanielpoly/nlp_final_project> (commit `ba1ab69` or later);
the source snapshot uploaded with these weights is in this repo under `code/`.

Each row's free-text fields are split into sentence items and embedded with the
frozen `intfloat/e5-small-v2` encoder (revision `ffb93f3bd4047442299a41ebb6fa998a38507c52`).
A table encoder initialises a 2-layer GRU that reads items one at a time and can
predict after any number of reads. An expected-gain head (DIME-style) predicts how
much the next item would reduce the predictor's loss, and the reader reads greedily
while the predicted gain exceeds a threshold calibrated to a target read budget.
Gain heads use either *content* keys (the item embedding) or *metadata* keys
(field and position only). The final full-budget model blends the reader with a
CatBoost Joint-Frozen model, with the blend weight tuned on validation data.

## Contents

One folder per run: 20 MulTaBench text datasets x 5 paper holdout folds (100 runs).

```text
index.json                  dataset, fold, task, test scores, data revision and split fingerprints of every run
<run_id>/reader/            reader + fitted preprocessing + CatBoost Frozen + blend weight
<run_id>/gain_content/      reader with content-key gain head and calibrated budget thresholds
<run_id>/gain_meta/         reader with metadata-key gain head and calibrated budget thresholds
code/                       source snapshot (package, tests, pyproject.toml, uv.lock, leaderboard references)
```

Each folder holds `manifest.json`, `preprocessing.pkl` (fitted transforms, class
mapping, target scaling), `reader_0.pt` and, for `reader/`, `catboost.cbm`.
`preprocessing.pkl` is a pickle: load only artifacts you trust.

## Inference mode

Install the project (`uv sync --locked` in `code/` or in the GitHub repository) and run
from its directory with `PYTHONPATH=src`. The estimator downloads one run from this
repository and predicts:

```bash
python -m reading_agent.estimator --repo htmldaniel/reading-agent-multabench \
    --dataset REG_TEXT_ROTTEN_TOMATOES --fold 0 --part reader \
    --input rows.csv --output preds.csv --mode blend
python -m reading_agent.estimator --repo htmldaniel/reading-agent-multabench \
    --dataset REG_TEXT_ROTTEN_TOMATOES --fold 0 --part gain_meta \
    --input rows.csv --output preds.csv --mode gain --budget 0.1
```

Modes are `all` (read every item), `none` (table only), `gain` (learned policy,
`--budget` selects a calibrated threshold: 0.1, 0.25, 0.5 or 0.75), `catboost`
and `blend`. The input must contain the dataset's feature columns in the benchmark
representation. Regression outputs are in original target units. Classification
outputs give predicted labels and per-class probabilities.
`python -m reading_agent.check_artifact --tag paper_v2 --run_id <run_id>` replays a
run's stored test predictions from raw rows.

## End-to-end training mode

The weights were produced by this command (100 runs, tag `paper_v2`):

```bash
python -m reading_agent.sweep --tag paper_v2 --folds 0,1,2,3,4 -- \
    --protocol paper --skip soft,hybrid --lams=-1 --epochs 100 --patience 15
```

Data comes from the MulTaBench Hugging Face datasets. The runs did not pin a dataset
revision on the command line; each run's resolved revision and train/test fingerprints
are recorded in `index.json` and its manifests. **The MulTaBench datasets have been
updated since these runs**, so reproduce a run with its recorded revision, e.g.
`python -m reading_agent.run --dataset <DATASET> --fold <k> --data_revision <data_revision from index.json>`
plus the recipe flags above. Training ran on CPU; GPU or different thread counts give
numerically close but not bit-identical weights.

## Evaluation

MulTaBench text datasets, paper protocol (folds 0-4). Metrics are AUROC for
classification (macro one-vs-rest for multiclass) and R² for regression. Folds are
averaged per dataset, and the Wilcoxon signed-rank test runs across datasets.

| | Classification (10 datasets, AUROC) | Regression (10 datasets, R²) |
|---|---|---|
| Blend (reader + CatBoost Frozen) | 0.900 | 0.702 |
| Reader, read all | 0.893 | 0.640 |
| CatBoost Joint Frozen (local, same rows) | 0.887 | 0.684 |
| Blend vs local CatBoost Frozen | +0.012, 10/10 wins, p=0.002 | +0.017, 8/10, p=0.008 |
| Blend vs CatBoost Joint TAR (leaderboard reference) | -0.008, 3/10 wins | -0.014, 2/10 wins |
| Gain policy vs matched random reader, nominal 10% of items, content keys | +0.026 | +0.051 |

External leaderboard values are approximate references: they use the same split
seeds, but row identity with the original data is not verified, so no significance
test is claimed against them. TabPFN-2.5 references cover only 8 classification
datasets; on those, the Blend averages 0.881 against TabPFN-2.5 TAR 0.900.

The learned policy beats a random reader with the same per-row read counts at
every tested budget, for both key types. The blend improves on Joint Frozen but
does **not** reach the Joint TAR references.

## Limitations

- **Selection, not lazy encoding.** All E5 embeddings are computed before
  selection, and content keys inspect every candidate item. The read fraction is a
  share of retained items, not of text or of measured compute.
- **Stronger external baselines.** TabM and TabPFN-2.5 baselines are stronger than these models.
- **Per-dataset models.** Each model is trained for one dataset and fold. They are
  not general-purpose models for new schemas; for those, retrain with the
  project's `ReadingEstimator.fit`.
