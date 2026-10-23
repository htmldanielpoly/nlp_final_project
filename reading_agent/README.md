# Reading Agent

Tabular-conditioned selection of text evidence for classification and regression.
Frozen E5-small-v2 embeddings feed an anytime GRU predictor; a supervised gain head
selects evidence using content or field/position metadata. This is adapted from
dynamic feature selection (DIME, ICLR 2024), rather than an RL-trained policy.

The submitted paper is `../paper/main.pdf` (source `../paper/main.tex`).
`docs/architecture.pdf` is an internal design log from day 3 of the project;
where it disagrees with the paper (protocol, results, coverage), the paper is current.

Trained weights (all 100 runs): <https://huggingface.co/htmldaniel/reading-agent-multabench>.

## Submission quick reference

```powershell
# Inference mode: download one run from Hugging Face and predict
python -m reading_agent.estimator --repo htmldaniel/reading-agent-multabench --dataset REG_TEXT_ROTTEN_TOMATOES --fold 0 --part reader --input rows.csv --output preds.csv --mode blend

# End-to-end training mode: the exact recipe that produced the uploaded weights (tag paper_v2)
python -m reading_agent.sweep --tag paper_v2 --folds 0,1,2,3,4 -- --protocol paper --skip soft,hybrid --lams=-1 --epochs 100 --patience 15

# Paper tables and figure from the completed runs
python -m reading_agent.paper_assets --tag paper_v2
```

The MulTaBench leaderboard references used by the paper are bundled in
`reference/multabench_leaderboard_text/` (source commit in its `SOURCE.md`).

## Install and verify

Python 3.11 or later is required. The project dependencies are pinned by `uv.lock`;
a clean `uv sync --locked` plus the test suite was verified on Windows (CPU) on 2026-10-16.

```powershell
uv sync --locked
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = 'src'
uv run --no-sync python -m unittest discover -s tests -v
```

For the existing external environment on this machine, substitute its Python
executable for `uv run --no-sync python`. `PYTHONPATH=src` also avoids editable-path
encoding problems on Windows. Run commands from the project directory.

## Benchmark experiments

```powershell
python -m reading_agent.run --dataset MUL_TEXT_WOMEN_CLOTHING_REVIEW --protocol paper --fold 0 --tag paper_review --skip soft,hybrid --lams=-1 --controls 1
python -m reading_agent.sweep --tag paper_review --folds 0 -- --protocol paper --skip soft,hybrid --lams=-1 --controls 1
python -m reading_agent.eval_paper --tag paper_review
python -m reading_agent.verify_predictions results/paper_review.jsonl
```

`paper` ports the paper's five holdout seeds (folds 0–4); `current` uses repeated
three-fold splits (folds 0–5). Data still comes from Hugging Face typed Parquet.
**Paper split rules alone do not prove identical rows to the old Kaggle CSVs.**
External leaderboard values remain references until original data identity is
verified. Local baselines share this runner's exact rows. Analysis separates
classification AUROC from regression R² and averages folds within datasets.

Use a new tag when changing the recipe. The sweep resumes completed dataset/fold
pairs only when their recorded recipe matches. Mismatches block the sweep rather
than silently mixing settings. Missing/empty caches and unsuccessful child runs
produce a nonzero status. Logs use unique filenames so interrupted logs survive.
Run one heavy job at a time on a memory-constrained laptop.

The current main recipe uses `--epochs 100 --patience 15` (code defaults remain
60/6). Inspect the remaining plan without starting training:

```powershell
python -m reading_agent.sweep --tag paper_v2 --folds 0,1,2,3,4 --last-datasets MUL_TEXT_SPOTIFY_GENRES --dry-run -- --protocol paper --skip soft,hybrid --lams=-1 --epochs 100 --patience 15
```

Remove `--dry-run` to execute that plan. Completed CatBoost stages are cached by
actual train/validation inputs, model settings and library version, independently
of later reader/policy stages. An interruption after a completed baseline can
reuse it (`--baseline_cache 0` disables this). New outer seeds still require new
baseline fits; old artifacts are not imported into this cache automatically.

`--data_revision <HF commit>` pins a dataset release. Each new run records its
resolved revision, train/test fingerprints, effective item cap, configuration and
implementation version. The frozen encoder is pinned to the
`intfloat/e5-small-v2` snapshot used for every result
(`ffb93f3bd4047442299a41ebb6fa998a38507c52`; override with `READING_AGENT_E5_REVISION`).

## Noisy early stopping

The existing defaults remain epochs=60 and patience=6. These controls are available
for inner-validation experiments, without changing historical results:

```powershell
python -m reading_agent.run --dataset REG_TEXT_ROTTEN_TOMATOES --protocol paper --tag warmup_dev --skip soft,hybrid,policy,catboost --epochs 120 --patience 15 --min_epochs 30 --early_stop_ema 0.8
```

EMA retention 0.8 means 80% previous EMA plus 20% current validation loss. It affects
stopping patience; the restored checkpoint always has the lowest **raw** validation
loss. Training history is saved under `results/history/<tag>/`. Select the recipe
using inner validation on several datasets, not fold-0 test improvements. These
settings are candidates, not a validated better recipe.

For a separate reproducibility diagnostic, use `--threads 1 --deterministic 1`
under a new tag. The main recipe keeps automatic CPU threading and the existing
determinism setting. Runtime versions/thread count and history metadata are saved.
Offline `es_replay` reports the best observed curve loss and warns about truncated
or unverified coverage; replay is not proof of the best attainable training loss.

## Saved inference artifacts

New agent runs save artifacts by default (`--save_models 0` disables this):

```text
results/models/<tag>/<run_id>/reader/
results/models/<tag>/<run_id>/gain_content/
results/models/<tag>/<run_id>/gain_meta/
```

The reader artifact includes fitted feature transforms, class mapping or target
scale, reader/ensemble weights and the CatBoost/PCA/blend state when enabled. Gain
artifacts include trained gain weights and calibrated budget thresholds. Existing
historical runs cannot gain checkpoints retroactively; they must be retrained.
Artifacts are local trusted model packages; their preprocessing uses pickle.

```powershell
python -m reading_agent.estimator --artifact results/models/<tag>/<run_id>/reader --input new_rows.csv --output predictions.csv --mode blend
python -m reading_agent.estimator --artifact results/models/<tag>/<run_id>/gain_meta --input new_rows.csv --output predictions.csv --mode gain --budget 0.1
```

`--budget` selects a saved nominal-budget threshold, not a per-row hard limit.
Regression outputs return to original target units. Classification outputs include
original-label predictions and probabilities in the saved class order. CSV input
must contain the fitted columns; additional columns are ignored. Parquet is also
supported. Numeric columns must use the fitted numeric representation (including
ordinal date preprocessing for benchmark date columns). Loading the artifact does not download the benchmark dataset. E5 and
uncached text embeddings may still need to download/compute normally.

To check that a saved artifact reproduces its run's stored test predictions from
raw rows (CSV round trip, then every inference mode and two calibrated budgets):

```powershell
python -m reading_agent.check_artifact --tag paper_v2 --run_id MUL_TEXT_WOMEN_CLOTHING_REVIEW_f0_1791988577_1772912b
```

This passed (max |difference| below 1e-6, identical per-row read counts) for
binary, multiclass and regression runs (Women Clothing, Fake Job Posting,
Kickstarter, DS Salary, Rotten Tomatoes, Vancouver Salaries).

### Hugging Face Hub

`reading_agent.hub` mirrors `results/models/<tag>/` to a Hub model repo, together with
`index.json` (dataset, fold, task, test scores, data revision and split fingerprints per
run), `MODEL_CARD.md` as its README and a `code/` source snapshot. The published weights
are at `htmldaniel/reading-agent-multabench`:

```powershell
hf auth login                                            # once, with a write token
python -m reading_agent.hub upload --tag paper_v2 --repo htmldaniel/reading-agent-multabench
python -m reading_agent.hub fetch --repo htmldaniel/reading-agent-multabench --dataset REG_TEXT_ROTTEN_TOMATOES --fold 0
```

`fetch` downloads one run and prints its local folder; pass its `reader`,
`gain_content` or `gain_meta` subfolder to `reading_agent.estimator --artifact`, or let
the estimator download it directly with `--repo/--dataset/--fold/--part`.

## Generic DataFrame training

```python
from reading_agent.estimator import ReadingEstimator
from reading_agent.train import Config

model = ReadingEstimator(
    num_cols=['age'], cat_cols=['region'], text_cols=['description'],
    task='binary', namespace='my_dataset', key_mode='meta',
    config=Config(epochs=60, patience=12),
)
model.fit(training_frame, training_labels)
model.save('saved_reader')
restored = ReadingEstimator.load('saved_reader')
probabilities = restored.predict_proba(new_frame, mode='all')
labels = restored.predict(new_frame, mode='gain', lam=0.02)
```

Declare each column's role explicitly. Generic fitting trains preprocessing only on
the predictor-training partition. It currently reuses the inner holdout for
early stopping and gain training; a separate final test set remains necessary.
The benchmark runner retains its historical outer-train preprocessing convention.

## Predictions and controls

Completed runs save test indices, targets and per-method predictions to
`results/preds/<tag>/<run_id>.npz`, with labels/target scaling and provenance in a
JSON sidecar. Random controls save every order separately because their reported
score is the mean of metrics across orders, **not** the metric of averaged predictions.
The verifier flags historical multi-order archives as incomplete rather than
treating stored order zero as the reported mean.

`--controls 1` adds random and natural-field-order readers at an independently
allocated aggregate budget (`random_fixed`, `natural_fixed` at each nominal budget)
and, per key mode, at the gain policy's *achieved* total budget spread by row
capacity (`randomAgg_*`, `naturalAgg_*`). The original per-row-matched random reader
(`randomB_*`) inherits the policy's per-row stopping counts, so together they
decompose the policy's advantage into item choice (`gainB - randomB`) and the
allocation of reads across rows (`randomB - randomAgg`):

```powershell
python -m reading_agent.controls_report --tag controls_v2 --out ../paper/generated
```

The controls sweep reuses the main recipe without CatBoost or saved models:
`sweep --tag controls_v2 --folds 0,1,2,3,4 -- --skip soft,hybrid,catboost --lams=-1 --protocol paper --epochs 100 --patience 15 --controls 1 --save_models 0`.

The reported read fraction measures selected evidence among retained items.
Every E5 embedding is still prepared in advance, and content selection inspects
all candidates. Metadata keys avoid inspecting unread content **for selection**,
but the implementation is not yet lazy encoding. Do not claim measured text
encoding savings from these results. Full-text CatBoost blends also pay for all
text embeddings.

```powershell
python -m reading_agent.budget_report --tag paper_v2 --output results/paper_v2_budget_report.json
```

This reports achieved fractions, dataset bootstrap intervals and Holm-adjusted
exploratory p-values across key/budget comparisons within each task/configuration.
The intervals resample datasets; they do not capture training-seed variation.
Positive average deltas at every budget do not establish significance everywhere.

## Paper tables and figures

The paper (`../paper/main.tex`) takes every number from generated files:

```powershell
python -m reading_agent.paper_assets --tag paper_v2
```

This writes `../paper/generated/`: `numbers.tex` (LaTeX macros), the main
accuracy table, the budget table, the per-dataset appendix table and
`fig_budget.pdf`. Folds are averaged within each dataset before any Wilcoxon test;
external leaderboard references are matched on the folds that were completed.
`numbers.tex` also holds the scale-free recovery statistics: the share of the read-all
gain over table-only that is recovered at 10%/25% of the items, by policy and by random.

Two appendix inputs come from separate commands:

```powershell
python -m reading_agent.coverage --out results/coverage.csv --tex ../paper/generated/tab_coverage.tex
python -m reading_agent.controls_report --tag controls_v2 --fieldmeta_tag fieldmeta_v1 --out ../paper/generated
```

`main.tex` includes the controls appendix only when `tab_controls.tex` exists. The
paper is compiled with `tectonic -X compile main.tex` (no other TeX installation is
needed). Re-check the four-page limit after each regeneration: the References
heading must start page 5.

## Status

`paper_v2` is complete: 20 datasets x 5 paper folds (100 runs), all 2500 stored
scores re-verified from predictions. `controls_v2` is complete (100/100 runs; the
paper's controls appendix uses all five folds) and `fieldmeta_v1` (fold 0, 20 runs) is
complete. The saved artifacts are published on Hugging Face (see `MODEL_CARD.md`).
Original-data identity for the external leaderboard references remains unverified,
so they are reported as references without significance tests. The clean-install
check passed on 2026-10-16.

To rebuild the paper and check the page limit: `bash scripts/build_paper.sh`
(Git Bash; it compiles a copy outside the project path with tectonic).
