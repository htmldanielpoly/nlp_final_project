#!/bin/bash
# After the gated stage-B sweep (run on the VM):
#  1. re-run the stage-B runs that finished before weight saving was added (old outputs moved to
#     results/stackB_prepatch, same code and recipe otherwise),
#  2. once the heads loop is idle, re-run their heads so every reported score comes from saved weights,
#  3. the gate ablation (fold 0, plain LoRA).
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
until [ "$(ls results/stackB/*/info.json 2>/dev/null | wc -l)" -ge 100 ] && ! pgrep -f "tabcon[d].*--tag stackB$" > /dev/null; do
  sleep 120
done
mkdir -p results/stackB_prepatch logs/stackB_prepatch
LIST=logs/rerun_list.txt; : > $LIST
for dir in results/stackB/*/; do
  run=$(basename "$dir")
  [ -f "$dir/weights/reader_2.pt" ] && continue
  mv "$dir" results/stackB_prepatch/ && mv "logs/stackB/$run.log" logs/stackB_prepatch/ 2>/dev/null
  echo "${run%_f*} ${run##*_f}" >> $LIST
done
echo "step 1: re-running $(wc -l < $LIST) stage-B runs"
job() { .venv/bin/python -m reading_agent2.tabcond --dataset $1 --fold $2 --gate 1 --tag stackB > logs/stackB/$1_f$2.log 2>&1; }
export -f job
xargs -P 2 -L 1 bash -c 'job $0 $1' < $LIST
echo "step 2: heads (waiting for the heads loop to finish)"
while pgrep -f "run_head[s]" > /dev/null; do sleep 60; done
. ~/.tabpfn_env
cp results/v2.jsonl results/v2_prepatch.jsonl
while read d f; do
  grep -v "\"dataset\": \"$d\", \"fold\": $f," results/v2.jsonl > results/v2.tmp && mv results/v2.tmp results/v2.jsonl
  $PY -m reading_agent2.stack heads --dataset $d --fold $f --tag v2 > logs/heads/${d}_f${f}.log 2>&1 \
    && echo "heads done $d $f" || echo "heads FAILED $d $f"
done < $LIST
echo "step 3: gate ablation"
bash scripts/run_stackB.sh "0" 2 0
echo "post_stackB finished"
