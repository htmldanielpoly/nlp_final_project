#!/bin/bash
# Last part of the v2 sweep on the VM (takes over from finish_v3.sh during its heads phase):
#  - exports run on the CPU (no GPU memory) as soon as a run's heads exist;
#  - once the running heads loops are done: the runs that crashed with GPU OOM are re-trained with 2 workers
#    (3 overflow the T4 on the long-text datasets), then 1 worker for anything still missing;
#  - heads of those runs (2 loops), then a replay check of a sample of artifacts.
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
. ~/.tabpfn_env
mkdir -p logs/export logs/heads logs/heads_claims logs/stackB_nogate logs/stackB_failed results/models/v2
DATASETS=$($PY -c "from reading_agent.data import TEXT_DATASETS; print(' '.join(TEXT_DATASETS))")
has_heads() { grep -q "\"dataset\": \"$1\", \"fold\": $2," results/v2.jsonl; }
eval "$(sed -n '/^release_failed() {/,/^}/p; /^train() {/,/^}/p; /^heads_loop() {/,/^}/p; /^release_heads() {/p' scripts/finish_v3.sh)"

export_loop() {  # CPU only; until every run is exported or training and heads are over
  while :; do
    left=0
    for f in 0 1 2 3 4; do
      for d in $DATASETS; do
        run=${d}_f${f}
        [ -f results/models/v2/$run/manifest.json ] && continue
        left=1
        [ -f results/heads/v2/$run/blend.json ] || continue  # written last by the heads
        [ -f logs/heads_claims/$run.exporting ] && continue
        CUDA_VISIBLE_DEVICES="" $PY -m reading_agent2.estimator2 export --dataset $d --folds $f > logs/export/$run.log 2>&1 \
          && echo "exported $run" || { echo "export FAILED $run"; touch logs/heads_claims/$run.exporting; }
      done
    done
    [ $left = 0 ] && return
    [ -f logs/gpu_over ] && return
    sleep 120
  done
}

rm -f logs/gpu_over
export_loop & E=$!
echo "$(date +%H:%M) waiting for the running heads loops"
while pgrep -f "^bash scripts/finish_v[3]" > /dev/null || pgrep -f "reading_agent2.stack head[s]" > /dev/null; do sleep 30; done
echo "$(date +%H:%M) re-training crashed runs (2 workers)"
release_failed
train & T1=$!
sleep 60
train & T2=$!
wait $T1 $T2
release_failed
echo "$(date +%H:%M) re-training anything still missing (1 worker)"
train
echo "$(date +%H:%M) heads"
release_heads
heads_loop & H1=$!
sleep 20
heads_loop & H2=$!
wait $H1 $H2
release_heads
heads_loop
touch logs/gpu_over
n_w=$(ls results/stackB/*/weights/reader_2.pt | wc -l)
n_h=$(grep -c '"method": "blend_mean@stackB"' results/v2.jsonl)
n_g=$(ls results/stackB_nogate/*/info.json | wc -l)
echo "$(date +%H:%M) complete: stage-B weights $n_w/100, primary scores $n_h/100, gate ablation $n_g/20"
wait $E
export_loop  # one more pass (logs/gpu_over is still set)
rm -f logs/gpu_over
echo "$(date +%H:%M) exports: $(ls results/models/v2/*/manifest.json | wc -l)"
for run in BIN_TEXT_FAKE_JOB_POSTING:1 MUL_TEXT_WINE_REVIEW:2 REG_TEXT_BOOK_PRICE:3 MUL_TEXT_US_ACCIDENTS:4; do
  $PY -m reading_agent2.estimator2 check --dataset ${run%:*} --fold ${run#*:}
done
echo "$(date +%H:%M) finish_v4 finished"
