#!/bin/bash
# Finish the v2 sweep on the VM in GPU-safe phases (replaces finish_v2.sh: three TabLoRA trainers next to a
# TabPFN head overflow the 16 GB T4, so training and heads no longer overlap):
#  1. training only (3 workers, claim-aware via the log file): every gated run without saved weights, then the
#     gate ablation (fold 0, plain LoRA, results/stackB_nogate);
#  2. heads only (2 loops): every run without a primary score in results/v2.jsonl;
#  3. retry anything that failed (1 trainer, 1 heads loop), export every artifact and replay a sample.
# Usage: bash scripts/finish_v3.sh [pid of a still-running trainer ...]
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
. ~/.tabpfn_env
mkdir -p logs/export logs/heads logs/heads_claims logs/stackB_nogate logs/stackB_failed
DATASETS=$($PY -c "from reading_agent.data import TEXT_DATASETS; print(' '.join(TEXT_DATASETS))")
has_heads() { grep -q "\"dataset\": \"$1\", \"fold\": $2," results/v2.jsonl; }

release_failed() {  # crashed runs give up their claim (log file) so a worker can retry them
  for f in 0 1 2 3 4; do
    for d in $DATASETS; do
      run=${d}_f${f}
      [ -f results/stackB/$run/weights/reader_2.pt ] && continue
      grep -q Traceback logs/stackB/$run.log 2> /dev/null && mv logs/stackB/$run.log logs/stackB_failed/$run.$(date +%H%M).log
    done
  done
  for d in $DATASETS; do
    [ -f results/stackB_nogate/${d}_f0/info.json ] && continue
    grep -q Traceback logs/stackB_nogate/${d}_f0.log 2> /dev/null \
      && mv logs/stackB_nogate/${d}_f0.log logs/stackB_failed/nogate_${d}_f0.$(date +%H%M).log
  done
}

train() {
  for f in 0 1 2 3 4; do
    for d in $DATASETS; do
      [ -f results/stackB/${d}_f${f}/weights/reader_2.pt ] && continue
      [ -f logs/stackB/${d}_f${f}.log ] && continue
      $PY -m reading_agent2.tabcond --dataset $d --fold $f --gate 1 --tag stackB > logs/stackB/${d}_f${f}.log 2>&1
    done
  done
  for d in $DATASETS; do
    [ -f results/stackB_nogate/${d}_f0/info.json ] && continue
    [ -f logs/stackB_nogate/${d}_f0.log ] && continue
    $PY -m reading_agent2.tabcond --dataset $d --fold 0 --gate 0 --tag stackB_nogate > logs/stackB_nogate/${d}_f0.log 2>&1
  done
}

heads_loop() {
  for f in 0 1 2 3 4; do
    for d in $DATASETS; do
      run=${d}_f${f}
      has_heads $d $f && continue
      { [ -f results/stackB/$run/info.json ] && [ -f results/stackB/$run/weights/reader_2.pt ]; } || continue
      mkdir logs/heads_claims/$run 2> /dev/null || continue  # atomic claim
      $PY -m reading_agent2.stack heads --dataset $d --fold $f --tag v2 > logs/heads/$run.log 2>&1 \
        && echo "heads done $run" || echo "heads FAILED $run"
    done
  done
}

release_heads() { for r in logs/heads_claims/*/; do r=$(basename $r); has_heads ${r%_f*} ${r##*_f} || rmdir logs/heads_claims/$r; done; }

echo "$(date +%H:%M) phase 1: training"
release_failed
TS=""
for pid in "$@"; do  # a new worker starts when an inherited trainer exits
  ( while kill -0 $pid 2> /dev/null; do sleep 30; done; train ) & TS="$TS $!"
done
for i in $(seq $((3 - $#))); do train & TS="$TS $!"; done  # 3 trainers in all
wait $TS
echo "$(date +%H:%M) phase 2: heads"
release_heads
heads_loop & H1=$!
sleep 20
heads_loop & H2=$!
wait $H1 $H2

echo "$(date +%H:%M) phase 3: retries"
release_failed
train
release_heads
heads_loop

n_w=$(ls results/stackB/*/weights/reader_2.pt | wc -l)
n_h=$(grep -c '"method": "blend_mean@stackB"' results/v2.jsonl)
n_g=$(ls results/stackB_nogate/*/info.json | wc -l)
echo "$(date +%H:%M) complete: stage-B weights $n_w/100, primary scores $n_h/100, gate ablation $n_g/20"

for f in 0 1 2 3 4; do
  for d in $DATASETS; do
    [ -f results/models/v2/${d}_f${f}/manifest.json ] && continue
    $PY -m reading_agent2.estimator2 export --dataset $d --folds $f > logs/export/${d}_f${f}.log 2>&1 \
      || echo "export FAILED $d $f"
  done
done
echo "$(date +%H:%M) exports: $(ls results/models/v2/*/manifest.json | wc -l)"
for run in BIN_TEXT_FAKE_JOB_POSTING:1 MUL_TEXT_WINE_REVIEW:2 REG_TEXT_BOOK_PRICE:3 MUL_TEXT_US_ACCIDENTS:4; do
  $PY -m reading_agent2.estimator2 check --dataset ${run%:*} --fold ${run#*:}
done
echo "$(date +%H:%M) finish_v3 finished"
