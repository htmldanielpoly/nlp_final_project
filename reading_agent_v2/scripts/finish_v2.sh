#!/bin/bash
# Finish the v2 sweep on the VM (takes over from post_stackB.sh after its step 0, which moved the gated runs
# without saved weights to results/stackB_prepatch and listed them in logs/rerun_list.txt):
#  - trainers (3 GPU workers, claim-aware via the log file): re-train the listed runs, then the gate ablation
#    (fold 0, plain LoRA, results/stackB_nogate);
#  - heads: each run's heads are fitted as soon as its stage-B weights exist (one loop while training, two after);
#  - finally retry anything that crashed, export every artifact and replay a sample.
# Usage: bash scripts/finish_v2.sh [pid of a still-running trainer ...]
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
. ~/.tabpfn_env
mkdir -p logs/export logs/heads logs/heads_claims logs/stackB_nogate logs/stackB_failed
rm -f logs/training_over
DATASETS=$($PY -c "from reading_agent.data import TEXT_DATASETS; print(' '.join(TEXT_DATASETS))")
LIST=logs/rerun_list.txt
has_heads() { grep -q "\"dataset\": \"$1\", \"fold\": $2," results/v2.jsonl; }

train() {
  while read d f; do
    [ -f results/stackB/${d}_f${f}/weights/reader_2.pt ] && continue
    [ -f logs/stackB/${d}_f${f}.log ] && continue
    $PY -m reading_agent2.tabcond --dataset $d --fold $f --gate 1 --tag stackB > logs/stackB/${d}_f${f}.log 2>&1
  done < $LIST
  for d in $DATASETS; do
    [ -f results/stackB_nogate/${d}_f0/info.json ] && continue
    [ -f logs/stackB_nogate/${d}_f0.log ] && continue
    $PY -m reading_agent2.tabcond --dataset $d --fold 0 --gate 0 --tag stackB_nogate > logs/stackB_nogate/${d}_f0.log 2>&1
  done
}

heads_loop() {
  while :; do
    progress=0; pending=0
    for f in 0 1 2 3 4; do
      for d in $DATASETS; do
        run=${d}_f${f}
        has_heads $d $f && continue
        pending=1
        { [ -f results/stackB/$run/info.json ] && [ -f results/stackB/$run/weights/reader_2.pt ]; } || continue
        mkdir logs/heads_claims/$run 2> /dev/null || continue  # atomic claim
        $PY -m reading_agent2.stack heads --dataset $d --fold $f --tag v2 > logs/heads/$run.log 2>&1 \
          && echo "heads done $run" || echo "heads FAILED $run"
        progress=1
      done
    done
    [ $pending = 0 ] && return
    if [ $progress = 0 ]; then
      [ -f logs/training_over ] && return
      sleep 60
    fi
  done
}

# heads of the listed runs came from readers without saved weights: drop them (backup kept)
[ -f results/v2_prepatch.jsonl ] || cp results/v2.jsonl results/v2_prepatch.jsonl
while read d f; do
  grep -v "\"dataset\": \"$d\", \"fold\": $f," results/v2.jsonl > results/v2.tmp && mv results/v2.tmp results/v2.jsonl
done < $LIST
echo "$(date +%H:%M) training $(wc -l < $LIST) gated runs + gate ablation; heads in parallel"
heads_loop & H1=$!
train & T=$!
TS="$T"
for pid in "$@"; do  # a new worker starts when an inherited trainer exits
  ( while kill -0 $pid 2> /dev/null; do sleep 30; done; train ) & TS="$TS $!"
done
wait $TS
touch logs/training_over
echo "$(date +%H:%M) training finished"
heads_loop & H2=$!
wait $H1 $H2

echo "$(date +%H:%M) retrying crashed runs"
rm -f logs/training_over
for f in 0 1 2 3 4; do
  for d in $DATASETS; do
    run=${d}_f${f}
    [ -f results/stackB/$run/weights/reader_2.pt ] || { mv logs/stackB/$run.log logs/stackB_failed/ 2> /dev/null; echo "retry $run"; }
    has_heads $d $f || rmdir logs/heads_claims/$run 2> /dev/null
  done
done
for d in $DATASETS; do
  [ -f results/stackB_nogate/${d}_f0/info.json ] || mv logs/stackB_nogate/${d}_f0.log logs/stackB_failed/nogate_${d}_f0.log 2> /dev/null
done
train
touch logs/training_over
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
echo "$(date +%H:%M) finish_v2 finished"
