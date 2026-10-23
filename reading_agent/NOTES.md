# Reading Agent — design log & experiment notes

Running log of approaches, decisions and results. Newest entries at the bottom of each section.

---
## ▶ START HERE (2026-10-17, day 9). Read this block first
Deadline **2026-10-26**. ALL experiments are done; what is left is packaging.

**Where things stand**
- `paper_v2` main sweep: **100/100 runs**, all scores verified. The paper uses it via `paper_assets --tag paper_v2`.
- `fieldmeta_v1` fold 0: 20/20 runs, in the paper (Appendix C, Table 6, captioned fold 0).
- `controls_v2`: **100/100 runs** (resumed 10-17 14:51, the last 5 fold-4 runs all exit 0, `ALL DONE rc=0` at 15:22). `verify_predictions`: 4600 scores, 0 incomplete, 0 mismatches.
  - `controls_report` regenerated into `../paper/generated` with 5 folds (`\ctlFolds`=5). @10% same-total-reads: content +0.053 AUROC / +0.122 R² (10/10, Holm p=0.016 both); metadata cls +0.020 (8/10, p=0.059, lost its star vs fold 0), reg +0.086 (10/10). Allocation = choice on cls (+0.027 each), larger on reg (+0.071 vs +0.051); natural order ≈ random.
  - Scheduled task `reading_agent_controls` **unregistered**. No scheduled tasks remain.
- Paper (day 9 edits): §4 sentence and Appendix C title/Table 5 caption now say five folds; proofread fixes ("the content policy beats", "spread over rows by item count", text plus signs, Fig. 1 $s_t$ label, removed TODO comment). Hardcoded per-dataset advantages (RT +0.14, Vancouver +0.11, DSS +0.085) re-checked against paper_v2. Builds to exactly 4 pages; `../paper/main.pdf` is current.
- Tests: 74, all pass. No git commits yet.

**Next session, in order**
1. ~~Finish controls~~ DONE 10-17.
2. ~~Switch appendix to 5 folds~~ DONE 10-17.
3. Hugging Face upload: the user runs `! hf auth login` (write token) and picks a repo name. Then `$CPU -m reading_agent.hub upload --tag paper_v2 --repo <user>/<name>` (2.3 GB). Put the link in the README and paper if wanted.
4. First git commit: ask the user first.
5. Final proofread of `../paper/main.pdf`, then the user submits on Moodle by 10-26.

**Tools that survive the session** (the old scratchpad copies are gone)
- Paper build: `bash scripts/build_paper.sh`.
  - Tectonic is at `C:\Users\riden\tools\tectonic\tectonic.exe`.
  - The build dir is `C:\Users\riden\tools\paper_build` (outside the Hebrew path, which tectonic cannot handle).
  - fitz comes from the system Python 3.11.
- Explainer pages for the user (claude.ai artifacts, private):
  - roadmap: https://claude.ai/artifact/ACsvbcS1srR8wmHreuJ1rG
  - "what is new in the method": https://claude.ai/artifact/FhP99qfLGDJaT1dfM63Ged
  - The roadmap shows controls f1 at 13/20; it is stale and can be republished.

---
## START HERE (2026-10-14), kept for history. Superseded by the block above
Deadline **2026-10-26** (12 days left). Current assessment: `../review/PROGRESS_REVIEW_2026-10-14.md`. Keep the 100-epoch/patience-15 recipe fixed for remaining seeds; reserve October 22–25 for paper and packaging.

**State**
- **Protocol = PAPER** (the user decided on 10-13): `--protocol paper`, folds 0–4. The split algorithm matches the official paper implementation on 40/40 synthetic cases. Exact current-HF versus original-Kaggle row identity remains unverified; fold-score correlations alone do not prove it. External Frozen/TAR values remain provisional references until the original ordered data is checked.
- **E5 cache complete for the paper protocol, all 20 datasets** (`precompute --protocol paper --dry` → 0 to embed, except 4 DSS texts that the run embeds on CPU). The GPU is not needed any more.
- **Fold-0 results for all 20 datasets**, exp #14, tag `paper_v2`; one recorded configuration, all 500 saved scores verified, all reader/policy artifacts present.
  - Tags `paper_f0` and `paper_f0_cap` are historical experiments, not the current main result.
  - Current evaluation: `$CPU -m reading_agent.eval_paper --tag paper_v2`.
  - Headline:
    - **Gain − matched random at nominal 10%:** content +0.0270 AUROC (8 wins/1 tie/1 loss), +0.0532 R² (9/0/1); metadata +0.0100 AUROC (7/2/1), +0.0305 R² (9/0/1). Positive means at every tested budget do not imply universal wins/significance. Actual DSS metadata @10% reads zero; Product Sentiment metadata @50/75% reads all items. See `budget_report` for achieved fractions and Holm-adjusted exploratory comparisons.
    - **Blend vs local CB Frozen:** +0.0103 AUROC (10/10 wins), +0.0190 R² (8 wins/1 tie/1 loss). TAR and stronger external references remain ahead on average; external data pairing is still provisional.
    - Spotify blend 0.968 exceeds all stored fold-0 references provisionally. Video Games read-all 0.544 exceeds local CB and CB TAR, **not** TabM Frozen 0.563 or LightGBM TAR 0.559. Jigsaw read-all 0.840 exceeds local CB 0.784 but remains below TAR.
- Full seed coverage is 20/100 combinations, with 80 remaining. The full fold took **297.7 minutes (~4 h 58)**; Spotify was 141.1 minutes, approximately 97% of that in CatBoost. Remaining seeds are roughly 20 hours at observed throughput, excluding interruptions.

**2026-10-16 (day 8, the user was away for 4 h and asked me to keep refining):**
- **Controls job RUNNING since 13:28** as the scheduled task `reading_agent_controls`, which calls `C:\Users\riden\run_controls_v3.sh`.
  - Stages, in order:
    1. `controls_v2` fold 0, the main recipe plus `--controls 1 --skip soft,hybrid,catboost --save_models 0`;
    2. `fieldmeta_v1` fold 0, the same with `--field_meta 1` and no controls (the metadata presence/length ablation on all 20 datasets);
    3. `controls_v2` folds 1–4.
  - Wine and Spotify run last in each stage. A run takes ~7–10 min, so each fold-0 stage ≈ 2.5–3 h.
  - Logs: `logs/sweep_controls_v2.log` (stage markers `FOLD0 DONE`, `FIELDMETA DONE`, `ALL DONE`) and `logs/sweep_fieldmeta_v1.log`.
  - Remove afterwards with `Unregister-ScheduledTask reading_agent_controls`. The sweeps skip completed pairs, so `Start-ScheduledTask reading_agent_controls` resumes the job.
  - Stage 3 can be stopped once enough folds are in.
  - Progress at 13:54: 6/20 controls runs done. Expected: fold 0 ≈ 15:45, fieldmeta ≈ 18:15.
  - The Claude Code reaper killed my background *wait loop* (low memory while idle). The scheduled task was unaffected.
  - Paper wiring: `main.tex` auto-includes `generated/controls.tex` and an Appendix C ("Reading controls") with `tab_controls.tex` / `tab_fieldmeta.tex` whenever those files exist. Test-compiled in a scratch copy with interim data: the main text stays at 4 pages.
  - TODO when the data is in:
    1. Run `controls_report --out ../paper/generated` (with `--fieldmeta_tag`).
    2. Rewrite the "random reader gets the policy's per-row read counts" sentence and the one-fold FJP field-meta sentence in §4 using the `\ctl*` / `\fm*` macros.
    3. Rebuild and re-check the 4 pages.
  - **15:19: controls_v2 fold 0 DONE**: 20/20 runs, 0 failures; `verify_predictions` checked 920 scores, 0 mismatches.
    - Gain − random with the same total reads spread by row capacity, @10%:
      - content: +0.054 AUROC (10/10), +0.144 R² (10/10);
      - metadata: +0.021 / +0.100;
      - all Holm p = 0.016.
    - Allocation effect (randomB − randomAgg) @10%: +0.026 AUROC, +0.093 R² (10/10). It fades by 50–75%; metadata @75% on regression is −0.017 (2/10).
    - Stored (natural) field order ≈ random: gain − natural +0.045 AUROC / +0.145 R² @10%.
  - Paper updated: one sentence in §4 Selective reading (macros `\ctlTotalContentTen*`) plus Appendix C (`tab_controls.tex`). Still 4 pages.
    - To fit it, I trimmed the Fig. 2 caption (figure width now 0.64), the imitation parenthetical, the metadata sentence, the caveat sentence and the first conclusion sentence.
  - Also new this afternoon: "Where reading helps" now leads with scale-free recovery statistics generated by `paper_assets.recovery_macros`.
    - At 10%, content recovers a median 63% of the read-all gain over table-only, vs 39% for random; metadata 45% vs 24%; 19 datasets, p<0.001.
    - The advantage correlates with the text effect: Spearman ρ=0.75 (p=0.013) on regression, 0.50 (p=0.138) on classification, not with length.
  - **17:06: fieldmeta_v1 fold 0 DONE**: 20/20 runs; 440 scores verified, 0 mismatches. With vs without field presence/log length, matched to controls_v2 fold 0:
    - Table-only: +0.025 AUROC (9/10, p=0.014), +0.037 R² (8/10, p=0.020).
    - Read-all: unchanged (−0.002 / −0.026, n.s.).
    - Content advantage @10%: classification +0.028 → +0.017 (p=0.012); regression +0.051 → +0.064 (n.s.).
    - Metadata advantage @25% on regression: +0.040 → +0.023 (p=0.014).
  - Paper: the one-fold FJP sentence is replaced with these `\fm*` macros, and Appendix C holds Table 6 (`tab_fieldmeta.tex`). Appendix float pages are now top-aligned. Still 4 pages; `paper/main.pdf` is updated.
  - Stage 3 (controls_v2 folds 1–4) started 17:06, ~8–9 h.
    - If it finishes, rerun `controls_report ... --out ../paper/generated`, then change "fold 0" in the §4 sentence and the Appendix C captions to the actual folds.
    - Otherwise stop it with `Stop-ScheduledTask reading_agent_controls` and kill the `reading_agent.sweep` / `reading_agent.run` Python processes. Fold 0 is already in the paper.
  - New in `run.py` (`--controls 1` only): `randomAgg_<key>` / `naturalAgg_<key>` = the gain policy's *achieved total* reads spread by row capacity. This decomposes gain − random into item choice (gainB − randomB) and per-row allocation (randomB − randomAgg). `implementation_version` = `2026-10-16-controls`.
  - `controls_v1` (Product Sentiment only) is a superseded smoke test without the Agg controls.
  - Analysis: `python -m reading_agent.controls_report --tag controls_v2 --fieldmeta_tag fieldmeta_v1 --out ../paper/generated` writes `tab_controls.tex`, `tab_fieldmeta.tex` and `controls.tex`.
    - Macros `\ctl<Total|Choice|Alloc|Natural><Content|Meta>Ten<Cls|Reg>{Delta,Wins,N,P}`.
    - Macros `\fm<Table|Readall|AdvContentTen|...><Cls|Reg>{Without,With,Delta,Wins,N,P}`.
- **Saved-artifact inference verified.** `python -m reading_agent.check_artifact --run_id <id>` does a CSV round trip of raw rows, then runs every mode and two budgets.
  - It reproduces the stored test predictions to <1e-6 with identical read counts on WCR, FJP, Kickstarter, DSS, RT and Vancouver.
  - Stored regression predictions are in standardised units; the CLI returns original units.
- **HF upload prepared, NOT uploaded.**
  - `reading_agent.hub index|upload|fetch` and `MODEL_CARD.md`.
  - `results/models/paper_v2/index.json` lists 100 runs.
  - To upload: the user runs `hf auth login`, then `python -m reading_agent.hub upload --tag paper_v2 --repo <user>/<name>` (2.3 GB, `upload_large_folder`).
- **E5 pinned** to snapshot `ffb93f3bd4047442299a41ebb6fa998a38507c52`, the only cached one, in `embed.py` (env override `READING_AGENT_E5_REVISION`).
- `matplotlib` is now a direct dependency; re-locking changed no versions.
- **Clean install PASSED:** in a scratch copy outside the Hebrew path, with an empty uv cache, `uv sync --locked` succeeded (~3 min to download) and all tests passed.
- New `reading_agent.coverage` writes `results/coverage.csv` and `../paper/generated/tab_coverage.tex` (now paper Appendix B).
  - Items keep every word on 15/20 datasets.
  - The cap binds on Zomato (1,909 words/row, 6% of words kept), Book Price (77%), FJP (89%), Book Readability (90%) and Michelin (94%).
  - The paper's Zomato sentence now cites this instead of "the MLP table encoder underfits".
- Tests: 74, all pass (`test_controls_report` added).

**2026-10-15 (day 7): folds 1–4 RUNNING since 14:03 as the Windows scheduled task `reading_agent_paper_v2`** (same tag `paper_v2`, same recipe).
- The task has no time limit and is battery-safe. It runs a hidden PowerShell that calls `C:\Users\riden\run_folds_paper_v2.sh`.
- Order: the 19 non-Spotify datasets × folds 1–4 (~10 h), then Spotify folds 1–4 (~9.5 h).
- Progress log: `logs/sweep_paper_v2_f1-4.log` (it ends with `ALL DONE`).
- Resume: `Start-ScheduledTask reading_agent_paper_v2`. The sweep skips done pairs.
- Remove afterwards: `Unregister-ScheduledTask reading_agent_paper_v2`.
- The first attempt at 12:40 used a WMI launch (`Win32_Process Create`) and was **killed after 5 min**. WMI children inherit the WmiPrvSE job, whose quota is `MemoryPerHost` = 512 MB. Never launch heavy jobs that way.
- **10-16 07:53: SWEEP COMPLETE.** `paper_v2` has 100/100 runs (20 datasets × 5 folds). `verify_predictions` checks all 2500 scores: 0 incomplete, 0 mismatches.
  - The scheduled task has been removed.
  - Paper regenerated with all folds: still exactly 4 pages; PDF at `../paper/main.pdf`.
  - Blend vs CatBoost TAR is now significantly worse on BOTH families (cls −0.008, p = 0.049; reg −0.014, p = 0.027).
- **10-16 00:02: the 19 non-Spotify datasets × folds 1–4 are DONE**: 76/76 succeeded and 0 failed. `verify_predictions` checks all 2400 scores with no mismatches.
  - Spotify folds 1–4 started 00:02 (~9.5 h).
  - The paper was regenerated with 5 folds and its wording updated (abstract cites the local CatBoost comparison).
  - All 16 gain-vs-random comparisons are Holm-significant: p ≤ 0.031 for classification, ≤ 0.018 for regression.
  - Blend vs LB CatBoost Frozen on regression: 9/10 wins but p = 0.064.
  - Blend vs CatBoost TAR on regression: −0.014, significant.
  - The main text is still exactly 4 pages.
- **Paper drafted (10-15)** in `../paper/main.tex` (NeurIPS 2025 style, `[preprint]`). Main text is exactly 4 pages; References start at the top of p.5; the appendix is the per-dataset table.
  - All numbers are generated: `$CPU -m reading_agent.paper_assets --tag paper_v2` writes `../paper/generated/{numbers.tex,tab_main.tex,tab_budget.tex,tab_per_dataset.tex,fig_budget.pdf}`.
  - The text uses macros (`\blendVsLocalCBClsDelta`, `\sVideoGamesReadAll`, …), so regenerate after the folds finish, then re-read the sentences marked `% CHECK-AFTER-FOLDS` and `% TODO(after folds 1-4)` (wording such as "wins"/"loses" may flip).
  - Compile: no local TeX. Use tectonic (`tectonic -X compile main.tex -Z continue-on-errors --keep-logs`) on a copy outside the Hebrew path. Re-check the 4-page limit after every regeneration.
  - Open items: the author name is a placeholder (ask the user). Table 1 dropped the gain@25% rows for space (they are in Fig. 2/Table 2).

**Next steps, in order**
1. ~~Early-stopping fix~~ **DONE 10-14 (exp #13):** use `--epochs 100 --patience 15` (selected on inner-val loss for RT/FJP/BP via `es_replay`). Defaults in code are unchanged, so always pass the flags.
2. Optional, cheap: BP and Babies (price targets) — try a log1p / clipped training target if read-all stays far below CB (see risks).
3. ~~Fold-0 sweep~~ **DONE 10-14 (exp #14, tag `paper_v2`), 20/20 (Spotify rerun done).** A full fold takes ≈ 2.5 h plus ≈ 2.3 h for Spotify's CatBoost, so folds 1–4 take ≈ 19 h. Run Spotify LAST in each fold, or in a separate run.
4. Folds 1–4 under the same recipe/tag. First use the recipe-guarded `sweep --dry-run`; put Spotify last via `--last-datasets MUL_TEXT_SPOTIFY_GENRES`. No run was launched by the October 14 review.
5. Deliverables (review §Requirements):
   - 4-page NeurIPS paper; cite DIME (ICLR 2024) as prior work for expected-gain acquisition.
   - README now documents training, analysis, inference and limitations.
   - New runs save reader/gain/optional CatBoost-blend artifacts by default, with preprocessing, class/target mapping and calibrated thresholds. Hugging Face upload remains pending; historical weights cannot be recovered from prediction files.
   - Generic `ReadingEstimator.fit/predict/save/load` and CSV/Parquet inference CLI are implemented and synthetic-tested.
   - Clean-install check.
   - First git commit: the repo has no commits yet; ask the user first.

**Recipe (current defaults):**
- `--gain_data val --gain_loss mse`, field_meta 0, sched none, budgets 0.1/0.25/0.5/0.75 (λ bisected on val, bracket auto-widens);
- `--cap_fields 1` (max_items ≥ #text fields);
- key modes content + meta; randomB averaged over 3 orders;
- always pass `--skip soft,hybrid --lams=-1 --protocol paper --epochs 100 --patience 15`.

**Commands** (Git Bash, from `final project/reading_agent`):
```bash
export PYTHONUTF8=1 PYTHONPATH=src HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONWARNINGS=ignore HF_HUB_OFFLINE=1 TQDM_DISABLE=1
CPU=/c/Users/riden/venvs/reading_agent/Scripts/python.exe      # training / eval
GPU=/c/Users/riden/venvs/embed_gpu/Scripts/python.exe          # E5 embedding only (not needed now)
# one run (results/<tag>.jsonl, logs to stdout, predictions to results/preds/<tag>/<run_id>.npz)
$CPU -m reading_agent.run --dataset REG_TEXT_ROTTEN_TOMATOES --fold 0 --protocol paper --skip soft,hybrid --lams=-1 --tag dev
# sweep (resumes: skips (dataset, fold) pairs that have a _done row; per-run logs in logs/<tag>_<dataset>_f<fold>.log)
$CPU -m reading_agent.sweep --tag paper_v2 --folds 0 [--datasets A,B] -- --skip soft,hybrid --lams=-1 --protocol paper --epochs 100 --patience 15 > logs/sweep_paper_v2.log 2>&1
# analysis
$CPU -m reading_agent.summarize results/paper_v2.jsonl     # accuracy + gain-vs-random per budget
$CPU -m reading_agent.compare results/paper_v2.jsonl       # external provisional references
$CPU -m reading_agent.eval_paper --tag paper_v2             # current main result
$CPU -m reading_agent.budget_report --tag paper_v2          # actual budgets + uncertainty/multiplicity
$CPU -m reading_agent.precompute --protocol paper --dry    # check the E5 cache
```

**Hardware rules**
- 16 GB laptop. The Claude Code low-memory reaper killed background jobs **7×** (10-08 to 10-13), even with Chrome mostly closed and only ~1–1.5 GB used by our process.
  - Ask the user to launch Claude Code with `CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1`.
  - **ONE heavy job at a time.**
  - **After a kill, ask the user before restarting.**
- Timing: small datasets take ~4–5 min per fold; RT with 13 items/row takes ~10+ min. The big datasets are untimed for training.
- Monitor tip: the precompute logs contain `
` progress bars. Use `TQDM_DISABLE=1`, or `grep -o` in Monitor filters, and never `grep -A` on that log.

**Still-open review items:**
- read fraction ≠ compute saved, so report retained-item fraction, token coverage and wall-clock separately;
- add fixed-budget random and static field-ranking baselines;
- dataset-level aggregation (average the folds per dataset first; don't average AUROC and R² deltas naively);
- separate calibration data for the gain head (val is used for early stopping, gain fit, blend and λ);
- metadata control (exp #6);
- pin E5 / HF data revisions.

**Review improvements (2026-10-13):**
- Split port checked against original code: 40/40 cases. See `../review/PROGRESS_AUDIT_2026-10-13.md`.
- Future archives save all random-order arrays and per-row counts; old random archives only saved order 0 and cannot reproduce their reported mean for per-row resampling. `verify_predictions` flags them as incomplete.
- Analysis now separates task metrics and configurations, averages folds within each dataset, treats numerical near-zero deltas as ties, and accepts explicit files/tags. `eval_paper` has no import-time side effects.
- Optional `--controls 1` adds independent aggregate-budget random and natural-field-order baselines.
- New rows save resolved HF revision, train/test content fingerprints, effective cap, implementation version, actual training configuration and warm-up history.
- Standalone artifacts: `results/models/<tag>/<run_id>/reader`, `gain_content`, `gain_meta`; generic API/CLI in `estimator.py`. Preserve a new tag for these changes.
- Tests cover analysis, gradient/stopping behavior, classifier/regressor persistence, saved random-order metrics, blend reload and regression ensemble scaling. No heavy benchmark job was launched by this review.

**Review additions (2026-10-14):**
- `sweep` now guards recipe consistency and diagnostic/partial completion, preserves crash logs, checks missing caches explicitly, supports dry-run/Spotify-last ordering, and returns failure status when jobs fail.
- Completed CatBoost stages are independently cached with actual training/validation feature+label hashes, model settings, library version and model checksum; future interruptions after a completed fit can reuse it. This does not remove initial fits on new seeds or import old artifacts automatically. `--baseline_cache 0` disables it.
- Optional `--threads 1 --deterministic 1` enables a separate reproducibility diagnostic; keep those experiments under a new tag. Main recipe defaults are unchanged. Runtime details and history metadata are recorded.
- Replay now validates curves/configs and reports unknown/truncated source coverage; `best observed` is not a global optimum.
- Budget report adds dataset bootstrap intervals and Holm adjustment over the eight key/budget comparisons per task; these remain exploratory and do not include training-seed uncertainty.
- All 20 main runs had `controls=0`. Independent-budget/static controls and the metadata-presence/length ablation remain necessary on representative datasets, using a separate tag so they do not block or mix the main sweep.

**Untested:** seeds 1–4; independent controls/metadata ablations on the locked recipe; `--gain_data trainval`; the `current` protocol with the new flags. Book Price still stopped early in its actual rerun, so the longer-patience intervention is not a universal undertraining fix.

## ARCHIVE: day-3 handoff + day-4/5 notes (superseded by START HERE above)

**State in one paragraph.**
- **All 20 datasets are embedded** (cache 1.3 GB). The precompute is finished, and the GPU isn't needed any more.
- The method is the reading agent with an **expected-gain policy** (§2b). It has been run on women-clothing (WCR) and fake-job (FJP), fold 0 only (exp #5–8). The gain policy beats random at tight budgets on both:
  - WCR: +0.013 @10% read, +0.012 @25%.
  - FJP: +0.015–0.024 at ≤50% read.
- The CatBoost+agent blend is ≈ CatBoost Frozen or slightly above (WCR 0.905 vs 0.902; FJP 0.915–0.920 vs 0.915).
- Agent read-all lags CatBoost on FJP (≈0.86 vs 0.915). The cause is the learner, not chunking (exp #7).
- **The 20-dataset sweep has never completed.** The first attempt (`--tag main2`, fold 0) was reaper-killed in its first dataset.
- Day 3 code changes are in §4 (fixes from a subagent audit, `--budgets`, `--field_meta`, `--sched`, `summarize.py`). **Results from before 10-12 used a buggy gain loss.**

**Architecture write-up:** `docs/architecture.tex` (+ compiled `docs/architecture.pdf`, 5 pages, updated on day 3 with an iteration-log section). No local TeX, so compile with a standalone `tectonic`: download the Windows zip into the scratchpad, then `tectonic -X compile architecture.tex`. Or use Overleaf. Edit the .tex with the Edit tool. Bash heredocs mangle backslashes.

**Recipe (locked on day 3):** defaults: `--gain_data val --gain_loss mse`, field_meta 0, sched none, budgets 0.1/0.25/0.5/0.75 (λ bisected on val). The λ grid is not needed, so pass `--lams=-1`.

**Hardware rules (learned the hard way)**
- 16 GB laptop. The Claude Code low-memory reaper has killed background jobs **4×**, the last during the sweep with only ~4 GB free (Chrome ~2.4 GB).
- **Run ONE heavy job at a time.** After a reaper kill, **ask the user before restarting**.
- Suggest closing Chrome before long runs, or launching Claude Code with `CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1`.
- Per-run time: ~4–5 min per dataset/fold (FJP, WCR). The big datasets are not yet timed.

**Exact commands (Git Bash, from `final project/reading_agent`):**
```bash
export PYTHONUTF8=1 PYTHONPATH=src HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONWARNINGS=ignore HF_HUB_OFFLINE=1
CPU=/c/Users/riden/venvs/reading_agent/Scripts/python.exe  # project env, training/eval
# 1) fold-0 sweep over all 20 (resumes: skips (dataset, fold) pairs with a _done row; partial rows are ignored)
$CPU -m reading_agent.sweep --tag main2 --folds 0 -- --skip soft,hybrid --lams=-1 > logs/sweep_main2_f0.log 2>&1
# 2) quick tables / leaderboard comparison
$CPU -m reading_agent.summarize results/main2.jsonl          # accuracy + gain-vs-random per budget
$CPU -m reading_agent.compare results/main2.jsonl            # vs official leaderboard + Wilcoxon
# 3) then folds 1,2
$CPU -m reading_agent.sweep --tag main2 --folds 1,2 -- --skip soft,hybrid --lams=-1 > logs/sweep_main2_f12.log 2>&1
```

**Next steps, in priority order**
1. **Fold-0 sweep, all 20 datasets** (`--tag main2`). It's the first run of regression and multiclass tasks and of the big datasets, so check the first few logs for crashes or RAM spikes. Rough estimate: 20 × ~5 min, but the big datasets will be slower.
2. `summarize` + `compare`. Look at: gain adv vs random per budget across datasets; blend vs CB Frozen; agent vs LB Frozen/TAR; regression behaviour (heavy-tailed targets, see risks).
3. Folds 1–2. Then, if time allows, the gain ablation (trainval vs val; now valid after the loss fix).
4. Paper (NeurIPS, 4 pages). Framing: tabular-conditioned active text reading. Headline results are the budget curves, the blend vs Frozen, an honest comparison with TAR, and the metadata finding (exp #6).

**Day 4 (10-13) external review: action items**
- FIXED: `meta_key` was computed under `torch.no_grad()` in `train_gain`, so it never trained. Every `--key_mode meta` result before 10-13 is invalid and must be rerun.
- TODO before any leaderboard claim: check that the MulTaBench leaderboard CSVs use the same protocol and splits as our folds (repeated 3-fold vs the paper's 5-seed holdout). Otherwise report the comparison as indicative only.
- Framing: "read fraction" = evidence used, not compute saved. Content keys already need E5 on every item. A real-savings claim needs meta/lexical keys plus deferred encoding.
- Cite DIME (ICLR 2024) as the prior for expected-loss-reduction acquisition. Our contribution is the text-tabular adaptation and analysis.
- Candidate single extension: masked table-query attention over acquired items (order-invariant reader).

**Day 5 (10-13): protocol check DONE, see exp #9**
- Verified the meta_key fix: on a synthetic check its weights now change (Δ=26.1, before 0.0).
- The leaderboard CSVs are **paper_version** results: folds 0–4, Kaggle-CSV data, holdout splits with SEED=42+fold, 10K train + min(10%, 2K) test.
  - They are NOT our current-protocol folds. The old claim in §3 ("identical to the leaderboard's folds") was wrong.
- Ported the paper split as `get_split(..., protocol="paper")`. Validated: CB Frozen on paper folds tracks the LB fold-for-fold.
- New flags:
  - `run.py --protocol paper|current`;
  - `--skip agent` (baselines only);
  - `precompute --protocol paper [--dry]`;
  - sweep passes `-- --protocol paper` through.
- Rows now store `protocol` and `args`. Test predictions are saved to `results/preds/<tag>/<run_id>.npz`.
- randomB is averaged over 3 orders.
- λ bisection widens its bracket.
- `compare` names budget rows `@budget` (λ is metadata), warns on current-protocol rows, and includes blend and CB in the paired tests.
- (DONE later on day 5, all 20 embedded) Paper-protocol embedding was missing (texts to embed): jigsaw 68K, kickstarter 79K, spotify 39K, us-accidents 44K, wine 33K, mercari 63K, scimagojr 12K, zomato 11K, vancouver 1K, michelin 1K, ds-salary/video-games a few. The other datasets are fully cached.

**Untested code paths**
- Regression (R²) tasks and multiclass AUROC with the fixed scoring.
- The big datasets (Mercari, US accidents, Jigsaw, Spotify, Kickstarter): RAM when loading the parquet.
- `--gain_data trainval`.

**Open questions / risks to keep in mind**
- Heavy-tailed regression targets (prices, salaries) with MSE on standardised y could dominate warm-up and gain targets. If R² is bad, try clipping or a log transform of the training target.
- FJP is noisy (±0.01 between near-identical runs), so judge tweaks across datasets and folds, not on one fold.
- Exp #6: when the table encoder knows field presence/length, the policy's edge over random vanishes on FJP. Part of the "value of information" is metadata. Report this honestly.
- The target-aware bar is Joint TAR. On FJP, LB TabPFN-v2.5 Frozen = 0.950 and TAR ≈ 0.93, both above our blend (0.92).
---

## 1. Problem
NLP 097215 final project, Track 1 (Algorithm). Text-tabular prediction on MulTaBench
(AUROC for classification, R² for regression). Deadline 2026-10-26.
The agent is trained on the target, so the relevant bar is most likely **Joint TAR**.
We also report against Joint Frozen.

## 2. Method — ORIGINAL PLAN (2026-10-08, kept for history; see §2b for what is actually implemented)
"Reading agent": an MDP over which text pieces to read, given the table.

- **Items.** Each text field is split into chunks (sentences merged up to ~64 tokens, K ≤ 8 per row).
  Every chunk is embedded once with frozen E5-small-v2 ("passage: " prefix) and cached.
- **Tabular encoder (ANN).** Each column becomes `col_emb(E5("query: <col name>")) + value_enc`.
  An MLP over these gives h0, used to initialise **both** GRU layers.
  This follows "the encoder provides the initial hidden state" (L9 p18).
- **State.** A 2-layer stacked GRU over the chunks read so far, with 0.2 inter-layer dropout (L8 p32–34).
  Ablation: plain RNN vs GRU vs LSTM.
- **Policy.** Pointer attention `e_i = (W_q h_t)ᵀ(W_k c_i)/√d`, read chunks masked to −∞,
  plus a learned STOP key (L9 p43, p60, p74).
- **Predictor.** An MLP over [h_t ; mean/max pool of the chunks read] (L6 p67), so it can predict after any number of reads.
- **Training.**
  1. Supervised warm-up over random reading prefixes: "sum of losses over time", teacher forcing.
  2. Imitation learning from a greedy oracle (a* = the chunk that most reduces the loss, or STOP if no gain beats λ),
     with DAgger rollouts to fix exposure bias (L11 p50; transition-parser oracles, T8 p6; MEMM local classifiers, L3).
  3. Optional REINFORCE / PPO fine-tune with a KL penalty to the imitation policy
     (RLHF structure, L12 p55: pretrain → SFT → RL).
- **λ** = cost per chunk read, i.e. a length penalty (L11 p38–39). Sweeping it gives accuracy-vs-budget curves.
- **Hybrid.** Out-of-fold agent features (ŷ, PCA-30 of h_T, number of chunks read) are added to CatBoost/TabPFN.

## 2b. Method as actually implemented (current, end of day 1). Supersedes §2 where they differ
- **Items** are (text column, chunk) pairs, verbalised as `"passage: <col>: <chunk>"` (same format as MulTaBench).
  - Short fields give a single item; long fields are split into sentence chunks of ≤ 48 words.
  - At most 8 items per row, allocated round-robin across fields.
  - Embedded once with frozen E5-small-v2 and cached.
- **Table encoder:** numeric (quantile-normal × learned vector + missing vector) and categorical (embedding) tokens, each plus the projected E5 embedding of the column name. They are flattened, passed through an MLP and a LayerNorm to give h_tab, and h_tab initialises both layers of the 2-layer GRU via tanh(Linear).
- **Predictor:** an MLP over [GRU top state ; mean of the items read ; h_tab], so it can predict after any number of reads.
- **Phase A warm-up:** random reading orders. Loss = 0.5·(mean loss over all prefixes) + 0.5·(full-read loss). AdamW 1e-3, clip 1.0, early stopping on val loss of the full read.
- **Phase B, expected-gain policy (the current method):**
  - The gain head g(s_t, key_i) regresses the observed one-step loss reduction (cur_loss − loss_after_reading_i) on states from an ε-greedy roll-in.
  - The predictor is frozen during this phase.
  - It is fit on the 20% inner-validation split, because the predictor's in-sample losses are biased low.
  - At inference it reads argmax g while max g ≥ λ.
  - The keys are "content" (the item's own embedding) or "meta" (field + position only, a true acquisition setting).
- **Not used any more:** the pointer/STOP imitation policy (exp #1, kept only behind `--extra imitation`) and REINFORCE (never implemented). Seed ensembles were tried and gave no gain.
- **Evaluation modes:** none (table only), all, gain@λ, random at a matched per-row budget, label-oracle (diagnostic only).
- **Baselines in `run.py`:**
  - CatBoost Joint-Frozen (num + cat + PCA-30 per text column of whole-field E5);
  - CatBoost tab-only;
  - soft attention over all items;
  - hybrid OOF stacking;
  - **val-tuned convex blend** of CatBoost and the agent.
  - The official leaderboard numbers (all 5 learners, Frozen and TAR) come via `compare.py`.

### Required baselines and ablations
- Table only (STOP at t=0)
- Read all, with mean-pooled E5
- **Soft attention over all chunks.** The course's answer to the bottleneck problem (L9 p31–44). This is the main competitor.
- Random policy at the same budget vs the learned policy (the key MDP ablation)
- Imitation only vs imitation + RL; greedy vs beam decoding; exhaustive-subset oracle upper bound

### Known risks (from the course-material review)
- Text in MulTaBench may be short. If most fields produce a single chunk, switch the actions to "which **field** to read".
  Text lengths must be measured first.
- Baselines use PCA-30 of the embeddings (lecture p57). Results are reported as normalized 0–1 scores averaged over datasets.
- Small datasets make RL high-variance, so the plan is imitation-first.

## 3. Environment
- Laptop: i5-11300H (4 cores), 16 GB RAM, MX450 2 GB.
- Project venv (CPU torch 2.14, pinned via `uv.lock`): `C:\Users\riden\venvs\reading_agent`. Create it with `UV_PROJECT_ENVIRONMENT=... uv sync`.
- GPU embedding venv (torch 2.11+cu128, not part of the submission): `C:\Users\riden\venvs\embed_gpu`. It runs about 2× faster than the CPU on long texts.
- MulTaBench clone (for leaderboard CSVs only): `C:\Users\riden\code\MulTaBench`. Data comes from the HF org `multabench` (cached in `~/.cache/huggingface`).
- The splits are vendored in `data.py`. `current` = repeated 3-fold × 2, seed 42, train capped at 10K and test at 5K (HF master). `paper` = paper_version holdouts (folds 0–4), the protocol of the LB CSVs (exp #9).
- The E5 cache lives outside OneDrive: `~/.cache/reading_agent/`.

## 4. Experiment log
| # | date | change | datasets | result | takeaway |
|---|------|--------|----------|--------|----------|
| 0 | 10-08 | first end-to-end run (women-clothing, fold 0) | WCR f0 | **killed: system low on memory** during CatBoost stage, after 15 min of embedding | E5 cache saved (~41 MB), so the rerun skips embedding. Mitigations: embed batch 32 + inference_mode, free E5 weights before training, CatBoost 4 threads / 3 GB RAM limit |
| 1 | 10-08 | v1: warm-up agent + imitation policy (greedy label oracle, DAgger) | WCR f0 | CatBoost frozen **0.9024** (LB CatBoost frozen 0.9028, so the reproduction checks out); CB tab-only 0.7825; agent table-only 0.7826; **agent read-all 0.8990**; soft-attn 0.8976; hybrid (OOF stacking) 0.9032. Imitation policy ≈ random at the same budget at every λ (e.g. λ=0: 0.8907 vs 0.8922; λ=0.1: 0.8806 vs 0.8788). Label oracle 0.966 with only 1.5 reads | (a) The GRU agent ≈ soft attention, so the bottleneck isn't hurting. (b) **Imitating a label-peeking oracle fails**: its STOP and choice depend on y, which the policy can't infer, and its 0.966 is cherry-picking rather than a real upper bound. So switch to learning E[Δloss given state, item] by regression (label-free value of information, cf. Covert et al. 2023). (c) Stacking adds ~0. Try a val-tuned blend |
| 2 | 10-08 | v2: **expected-gain policy** (regress observed Δloss per item, ε-greedy roll-in), content vs meta keys; val-tuned blend | WCR f0 | read-all 0.8987; **blend CB+agent 0.9046 (w=0.4)**, +0.0022 over CB frozen. Gain policy beats random at every budget. Content keys: 91% items read → 0.8986 (no loss) vs random 0.8972; 81% → 0.8978 vs 0.8957; 64% → 0.8930 vs 0.8900. Meta keys (no peeking at content): 78% → 0.8961 vs 0.8940 | **The value-of-information policy works where imitation failed.** The gap grows as the budget shrinks. λ grid too fine (gains are small in nats); use 0.02–0.3. The blend is the best accuracy so far, but still < CB TAR (~0.909) |
| 3 | 10-08 | v3: 3-seed agent ensemble; λ grid 0–0.4 | WCR f0 | ens3 read-all 0.8969 (< single 0.8990); blend 0.9037. Gain (content) vs random: 83% → 0.8930/0.8936; 45% → 0.8805/0.8767; **28% → 0.8738/0.8611; 11% → 0.8545/0.8391**. Meta keys ≈ random at every budget | Seed ensembling doesn't help (seed noise ±0.002), so drop it. **Gain policy is unstable at near-full budgets** (v2 showed an advantage at 80–90%, v3 did not; only the init changed) but gives a robust +0.013–0.015 at tight budgets. The gain head is fit on only 2K val rows with noisy targets, so next ablate fitting data (val vs train+val), Huber loss, epochs (`ablate_gain.py`, 3 random seeds per budget) |
| 4 | 10-08 | gain-head ablation (`ablate_gain.py`; advantage over random averaged over 6 budgets × 3 random seeds) | WCR f0 | **partial, killed (low RAM)**. val+MSE: content **+0.0056** (+0.014 at 28%, +0.016 at 11%), meta +0.0009; val+Huber content +0.0019 | Huber hurts (the large gains carry the signal). Content keys ≫ meta keys. trainval / 40-epoch variants not run yet |
| – | 10-10 | precompute day 2 | 10 datasets | Batch 1 (book-price rest, book-readability, rotten-tomatoes, montgomery, DS-salary): 27 min. Batch 2 (video-games 1 min, michelin 27 min, scimagojr 10 min, **zomato 69 min**, vancouver 0.5 min): 107 min. No errors, no reaper kills; GPU 83–88°C. Cache 901 MB for 14 datasets | Large datasets take far longer than texts/row suggests: length-sorted chunks reach ~8 min per 4K texts at the long tail. Run one big dataset per command |
| 5 | 10-12 | first run off women-clothing: fake-job fold 0 (`--tag main`, pre-fix code) | FJP f0 | CB frozen **0.9153**, CB tab-only 0.6689, agent table-only 0.6953, **agent read-all 0.8673** (5.4 items/row), **blend 0.9186 (w=0.4)**. Gain vs random: 76% → 0.8601/0.8587; **50% → 0.8569/0.8436; 17% → 0.8109/0.7933**; 2% → 0.7191/0.7089. λ ≥ 0.1 reads ~nothing | **Gain policy generalises** (+0.013 at 50%, +0.018 at 17%). Blend +0.003 over Frozen. But the agent alone lags CB by 0.05 here (vs 0.003 on WCR). Hypothesis: the agent never sees text-field missingness or length (CB does, via the embedding of the empty string), and round-robin truncation to 8 items drops the ends of long fields. The λ grid is badly scaled per dataset, so switch to budget-targeted λ |
| – | 10-12 | code fixes after a static audit (subagent) | – | (1) gain-head loss was divided by a *running* count across batches (lr decayed within each epoch; confounded the trainval-vs-val ablation), now per-batch mean; (2) multiclass AUROC was NaN if a class is absent from y, now macro over present classes; (3) CatBoost `class_names` fixed to all classes; (4) `run_id` per row, and `compare` drops rows of runs without `_done`; (5) `ds.x` freed after the split (RAM on big datasets); (6) `sweep` skips datasets that aren't embedded; (7) warm-up raises if val loss is never finite | Results from before 10-12 used the old gain loss; they are not directly comparable |
| – | 10-12 | new options | – | `--field_meta 1`: per text field, [present ±1, standardised log word count] as extra numeric tokens in the table encoder (metadata, known before reading). `--budgets 0.1,0.25,0.5,0.75` (default on): λ bisected on val to hit each target read fraction → methods `gainB_<key>` / `randomB_<key>` | to test: fake-job with/without field_meta |
| – | 10-12 | precompute | wine | WINE_REVIEW embedded in 11 min (53 MB), GPU 91°C. **15/20 cached** | |
| 6 | 10-12 | A/B `--field_meta` 0 vs 1, fixed gain loss, budget curves (`--tag fm`) | FJP f0 | fm=0: table-only 0.695, read-all 0.867, blend 0.9186; gainB−randomB (content) **+0.024 @10%, +0.015 @25%, +0.017 @50%**, +0.004 @75% (mean +0.015); meta keys +0.005. fm=1: **table-only 0.761**, read-all 0.858, blend 0.9195; gain advantage **≈ −0.003** at every budget (content and meta), although its absolute low-budget AUROC is higher (0.810 vs 0.789 @10%) | Field presence/length is very predictive (+0.066 table-only). **Much of the gain policy's edge over random on FJP is "read the fields that exist/are long"**, information that field_meta hands to the table encoder for free. Read-all gap to CB (~0.05) is NOT explained by missingness. Don't adopt field_meta yet (single seed). Next: diagnose the gap with `--item_mode field` (whole-field items = the CB strings, cached) |
| 7 | 10-12 | diagnostic: `--item_mode field` (one item per whole field = exactly the strings CB embeds) | FJP f0 | read-all **0.8625** (2.2 items/row) vs chunks 0.867; CB frozen 0.915; LB TabM frozen 0.940 | **The read-all gap is not chunking/truncation.** Given the same embeddings, the agent's learner underperforms. Warm-up val loss is very noisy (0.124↔0.159 between epochs) while train loss keeps falling, so early stopping (patience 6, fixed lr 1e-3) stops at a random point. Fix to try: `--sched plateau` (halve lr when val stalls 2 epochs, patience +4) |
| 8 | 10-12 | A/B `--sched plateau` (`--tag sched`) | FJP f0, WCR f0 | FJP plateau: read-all 0.8542 (vs 0.8673 none), blend 0.9146, gain adv +0.002. WCR none: read-all 0.8990, blend 0.9047, gain adv content **+0.013 @10%, +0.012 @25%**, +0.006 @50%, +0.001 @75%; meta ≈ 0. WCR plateau: read-all 0.8973, similar curve | **Plateau rejected** (no gain on either). **FJP is very noisy** run to run (read-all 0.854–0.867, gain adv +0.002…+0.015 between near-identical configs), so single-fold FJP comparisons are not conclusive. WCR is stable and confirms the gain policy with the fixed loss. Recipe locked: field_meta 0, sched none, budgets 0.1/0.25/0.5/0.75. Next: fold-0 sweep over the 15 cached datasets (`--tag main2`) |
| – | 10-12 | sweep `--tag main2 --folds 0` | – | **Killed by the low-memory reaper (4th time)** at ~4.5 min into the first dataset (FJP), while the session was idle. No finished runs; partial rows in results/main2.jsonl are ignored by compare/summarize (no `_done`). Python used ~0.7 GB, so the pressure came from the rest of the system | Ask the user before restarting. Rerunning the same sweep command resumes (only pairs with `_done` are skipped) |
| – | 10-12 | precompute | kickstarter | KICKSTARTER_FUNDING embedded in 17 min (100 MB), GPU ≤ 91°C. **16/20 cached**. US_ACCIDENTS: 14 min (64 MB), GPU 89°C, **17/20 cached**. MERCARI: 15 min (80 MB), **18/20 cached**. JIGSAW: 24 min (93 MB), **19/20 cached**. SPOTIFY: 5 min (58 MB). **All 20 cached (1.3 GB); precompute finished** | |
| 9 | 10-13 | protocol check: CatBoost Frozen only on **paper-protocol** splits (`--protocol paper --skip agent --tag paper_cb`), folds 0–4 | FJP, WCR, RT | ours vs LB CatBoost Frozen, mean (fold-score correlation): WCR 0.9032 vs 0.9028 (r=+0.93); RT R² 0.4933 vs 0.5085 (r=+0.98); FJP 0.8927 vs 0.9058 (r=+0.53, ~60 positives per test fold, so noisy) | **The split port reproduces the paper's test rows** (fold patterns match). The ~0.015 level gaps on RT/FJP come from our CB setup (20% inner val, date→ordinal, HF typed data), not the rows. The paper protocol enables paired comparisons with all LB learners, Frozen and TAR, over 5 folds |
| 10 | 10-13 | **first agent sweep, paper protocol**, fold 0 (`--tag paper_f0`, `--skip soft,hybrid --lams=-1 --protocol paper`) | 8 datasets (DS-salary killed by the reaper, 5th kill; video-games not reached) | read-all vs our CB Frozen: BR **0.804** vs 0.777, PS **0.903** vs 0.893, WCR 0.902 vs 0.905, FJP 0.906 vs 0.910, Montgomery 0.968 vs 0.985, Babies 0.810 vs 0.875, Book-price 0.429 vs 0.552, **RT 0.215 vs 0.507** (= agent table-only). Blend ≥ CB on all 8 (+0.006 mean vs LB CB Frozen, 5/8 wins; −0.007 vs LB CB TAR). TabPFN-v2.5 beats everything 8/8. Gain−random mean advantage (content/meta): FJP +0.040/+0.027, Babies +0.026/+0.044, BR +0.012/+0.013, WCR +0.011/+0.003, BP +0.008/+0.009, PS/RT/Montgomery ≈0 | Regression path works, and **meta keys now work** after the gradient fix (≈ content on 4 datasets). **RT failure = item cap:** 13 text fields but max 8 items, so ≥5 fields are never seen; warm-up val loss stuck at 0.90. Book-price is also near the cap. PS has 1 item/row, so policy ≡ random (expected). Next: make the cap ≥ #text fields (e.g. `max_items = max(8, n_fields)`, or whole-field items for short fields) and rerun RT/BP |
| 11 | 10-13 | resume paper_f0 (DS-salary, video-games) + RT with the cap fix (`--tag paper_f0_cap`) | DSS, VG, RT | DSS: read-all **0.825** vs CB 0.817 (LB CB TAR 0.824, TabPFN-v2.5 0.833), blend 0.826, gain adv **+0.060** (+0.083 @10%). **VG: read-all R² 0.544** vs CB 0.399; beats every LB entry (best LB = CB TAR 0.500). Gain adv −0.008. RT with the cap: warm-up val loss 0.55 (was 0.90 with cap 8), but the run was **killed by the reaper (6th kill)** at warm-up ep 50, ~9 s/epoch with 12.8 items/row, so no test score. Over the 10 paper_f0 datasets (RT still the buggy row): blend vs LB CB Frozen **+0.016, 7/10 wins (p=0.06)**; vs LB CB TAR −0.005, 4/10; read-all vs LB CB Frozen −0.033, 5/10 | The cap fix works on RT (training is no longer stuck). The agent is strongest on text-heavy regression (VG, BR) and multi-field DSS. Rerun RT with `--tag paper_f0_cap`. Mixed AUROC/R² Wilcoxon on 1 fold is only indicative |
| 13 | 10-14 | **warm-up early-stopping selection** (`run --warmup_only 1 --epochs 100 --patience 100 --tag es_curve`, then `python -m reading_agent.es_replay results/history/es_curve/*.json`): one full 100-epoch curve per dataset, every stopping rule replayed offline on it; selected on **inner-val loss only** (no test scores computed) | RT, FJP, BP fold 0 | Best val on curve / longest stall before it: RT 0.482 @ep74 / 5, FJP 0.1045 @ep53 / 8, BP 0.588 @ep97 / 19. Gap to curve best: current p6+60ep: RT +0.046 (60-ep cap binds; exp #12 run hit a 6-stall and stopped at ep 24), FJP +0.0018, BP +0.047; p12+80ep: 0, 0, +0.032; **p15+100ep: 0, 0, +0.0068**; p10+EMA0.7+100ep: 0, 0, +0.0068; EMA0.5 / min_epochs alone do not help. Cost: 100 ep ≈ 6.5 min RT, 6 min FJP, 2.7 min BP | **New recipe: `--epochs 100 --patience 15`** (best on all 3; p10+EMA0.7 ties but adds a knob). Patience 6 sits at the stall length of these curves, so it is fragile; CPU multithread nondeterminism makes two identical RT runs diverge after ep 15. BP is still improving at ep 97 with very noisy val, so part of its read-all gap to CB may be undertraining. One curve per dataset, so indicative |
| 14 | 10-14 | **fold-0 sweep, all datasets, final recipe** (`--tag paper_v2`, `--epochs 100 --patience 15`, otherwise as recipe) | 19/20 (Spotify killed by the reaper during CatBoost, 7th+ kill; everything else exit 0). Total ≈ 2.5 h: the paper protocol caps splits at ~8k/2k/2k rows, so big datasets take 4–12 min (wine 32 min) | **Gain − matched random** (dataset-level, Wilcoxon): content @10% cls +0.029 AUROC 7/9 (p=.016), reg +0.053 R² 9/10 (p=.006); meta @10% cls +0.011 6/9 (p=.047), reg +0.031 9/10 (p=.004); positive at every budget. **Blend vs LB CB Frozen:** cls +0.014 9/9 (p=.004), reg +0.014 8/10 (p=.037). Blend vs LB CB TAR: cls −0.009 2/9, reg −0.008 3/10. Read-all vs LB CB Frozen: cls +0.007 5/9, reg −0.037 2/10. Standouts: jigsaw read-all 0.840 vs CB 0.785, VG 0.544 vs 0.399 (beats all LB), BR 0.804. Weak: Zomato read-all 0.59 vs CB 0.82 (blend w=0), BP 0.43, Mercari 0.37 | Early-stop fix helped Babies (read-all 0.810→0.849) and RT (blend 0.529). Headline claim holds on 19 datasets: the gain policy beats random at all budgets in both key modes, and blend > CB Frozen. Gap to TAR/TabM/TabPFN remains. BP early-stopped at ep 39 again (stall 15, nondeterminism). DSS meta@10%: λ calibration gave 0 reads (frac 0), so it is counted as a tie. **Spotify rerun 10-14 (same tag): done**: CB Frozen 0.9615, read-all 0.9569, blend **0.9680** (beats every LB entry; LB CB TAR 0.9650), gain−random @10% content +0.006, meta +0.002. Took **141 min, almost all in the two CatBoost fits** (114 classes × 3000 iters, MultiClass); the agent itself takes ~2 min. With 20/20: blend vs LB CB Frozen cls 10/10 (+0.013, p=.002); content @10% cls +0.027 8/10 (p=.008) |
| 15 | 10-15/16 | **folds 1–4, same recipe** (`--tag paper_v2`, scheduled task `reading_agent_paper_v2`) | 20 × folds 1–4 | 80/80 exit 0; paper_v2 = 100/100, 2500 scores verified. Gain > random Holm-significant at all 16 key/budget/family cells; blend > local CB Frozen +0.012 AUROC / +0.017 R² (both p<0.01); blend < CB TAR on both families | main result of the paper |
| 16 | 10-16/17 | **reading controls** (`--tag controls_v2`, `--controls 1 --skip soft,hybrid,catboost --save_models 0`) | 20 × folds 0–4 | 100/100 (fold 4 finished 10-17 after a user-requested stop); 4600 scores verified. @10% gain − random(same total reads): content +0.053 AUROC / +0.122 R² (10/10, p=0.016); row allocation ≈ item choice on cls, larger on reg; natural field order ≈ random | paper §4 + Appendix C Table 5 (5 folds) |
| 17 | 10-16 | **field metadata in table** (`--tag fieldmeta_v1`, `--field_meta 1`) | 20 × fold 0 | table-only +0.025 AUROC / +0.037 R² (sig.); read-all unchanged; content advantage @10% cls +0.028→+0.017 (sig.), reg +0.051→+0.064 (n.s.) | paper §4 + Appendix C Table 6 (fold 0) |
| – | 10-08 | infra | – | CPU embedding too slow for long fields (~4.5 s/batch). Built a separate CUDA env (`C:\Users\riden\venvs\embed_gpu`, torch 2.11+cu128) for the MX450: ~2.3 s/batch at 512 tokens (~2× faster); short texts are much faster | **Run only one heavy job at a time.** GPU precompute + CPU training together triggered the low-memory reaper. Embedding store now saves every 4K texts |

## 5. Implementation status (2026-10-08)
- `src/reading_agent/`: data (vendored MulTaBench splits, HF loader), chunking, embed (persistent E5 store), features, model (agent + soft-attention baseline), train (warm-up, DAgger imitation, inference modes), baselines (CatBoost Joint-Frozen, PCA-30), run (per dataset/fold), compare (vs leaderboard, Wilcoxon), precompute (embed all folds).
- Leaderboard facts: the published CSVs report AUROC (macro OvR for multiclass) and R², folds 0–4, for CatBoost/LightGBM/TabM/TabPFN-v2/v2.5 under `all` (Frozen) and `ft` (TAR). TAR > Frozen mostly by < 0.01, except Jigsaw (+0.1).
- Run with `PYTHONUTF8=1 PYTHONPATH=src` (the editable-install .pth breaks on the Hebrew path).
- Venv lives outside OneDrive: `C:\Users\riden\venvs\reading_agent`.

## 6. End of day 1 (2026-10-08): status and next steps (summarised in START HERE at the top)
**Where we are**
- The full pipeline works end to end on women-clothing fold 0. The CatBoost Joint-Frozen reproduction matches the leaderboard (0.9024 vs 0.9028).
- Accuracy: agent read-all ≈ 0.899, soft attention ≈ 0.898, CatBoost frozen 0.902, **val-tuned blend 0.904–0.905**. CatBoost TAR (LB) ≈ 0.909. Beating TAR on raw accuracy looks unlikely.
- The method's real result is the **expected-gain reading policy**: at the same reading budget it beats a random reader, by +0.014–0.016 AUROC when reading 11–28% of the items. Imitating a label oracle does *not* work (documented negative result).
- Precompute was **stopped early (laptop overheating)**. Embedded: women-clothing, fake-job, product-sentiment, babies-prices (complete); book-price partial (~1/2). The other 15 datasets are still to do. Rerunning the same precompute command resumes and skips cached texts.
- Overheating: run the GPU precompute in shorter sessions, a few datasets at a time.

**Next steps (in order, one heavy job at a time)**
1. Let the precompute finish (check `logs/precompute.log`; rerun the same command to resume).
2. Finish the gain ablation: `python -m reading_agent.ablate_gain --dataset MUL_TEXT_WOMEN_CLOTHING_REVIEW --fold 0 --variants trainval_mse,val_mse_40ep`, then fix the gain-head recipe.
3. Run the sweep over all 20 datasets, folds 0–2: `python -m reading_agent.sweep --tag main --folds 0,1,2 -- --skip soft,hybrid`. Then `python -m reading_agent.compare results/main.jsonl`.
4. Paper framing: "tabular-conditioned active text reading". Headline results:
   (a) accuracy-vs-budget curves against random and against reading everything;
   (b) the blend vs Joint Frozen (Wilcoxon over dataset × fold);
   (c) honest comparison to TAR.
   Also discuss: imitation-oracle failure, meta vs content keys, GRU vs soft attention.
- Commands need `PYTHONUTF8=1 PYTHONPATH=src`. Use the CPU venv for training and the `embed_gpu` venv for embedding.

**Day 5 (10-13) late: item-cap fix (code ready, not yet run).**
- `run.py --cap_fields 1` (the default) sets `max_items = max(8, #text fields)`.
- `FeaturePipeline.max_chunks` (per-field chunk cap) is now separate from `max_items` and fixed at 8, so the cached chunk strings stay valid. Precompute uses the same cap.
- Affects only RT (13 fields; 46 new texts) and Scimagojr (10 fields).
- Book-price has 7 fields, so its gap to CB (0.43 vs 0.55) is NOT the cap. It's the learner (heavy-tailed price target? see risks).
- To resume: the same paper_f0 sweep command picks up DS-salary + video-games. RT needs a new tag (e.g. `paper_f0_cap`) or deleting its paper_f0 rows.
- **Exp #12 (10-13): RT rerun with the cap fix** (`paper_f0_cap`): read-all R² **0.394** (was 0.215 with cap 8), CB 0.507, blend w=0 → 0.507.
  - Warm-up early-stopped at ep 24 (best val 0.676). The killed run had reached val 0.547 by ep 48 and was still improving.
  - The val loss is very noisy (0.68↔0.82 between epochs) and still trending down. Patience 6 stops too early, the same symptom as exp #7 (FJP).
  - Candidate fix: patience 12–15 (+ more epochs) or an EMA of val loss for early stopping. Choose it on **inner val loss** across several datasets, not on fold-0 test.
- **Paper-protocol embedding (10-13):** michelin, vancouver, zomato (11 min), scimagojr, wine (7.5 min), spotify (3 min), us-accidents (3.5 min) and mercari (9 min) are done. **Jigsaw was killed by the reaper (7th kill) mid-run** (the store saves every 4K texts, so a rerun resumes). Kickstarter is not started.
  - Resume: `$GPU -m reading_agent.precompute --protocol paper BIN_TEXT_JIGSAW_TOXICITY` and then `BIN_TEXT_KICKSTARTER_FUNDING` (GPU = embed_gpu venv; set TQDM_DISABLE=1 for clean logs).
