#!/bin/bash
# Stage A on the VM: cross-fitted reader features (CPU workers) + heads (GPU), fold 0 first.
# Usage: bash scripts/run_stackA.sh "0 1 2 3 4" 3 tabm v2_stackA
cd "$(dirname "$0")/.." || exit 1
FOLDS=${1:-"0 1 2 3 4"}; WORKERS=${2:-3}; HEADS=${3:-tabm}; TAG=${4:-v2_stackA}
export PYTHONPATH=src PY=.venv/bin/python
mkdir -p logs/stackA
DATASETS=$($PY -c "from reading_agent.data import TEXT_DATASETS; print(' '.join(TEXT_DATASETS))")
QUEUE=logs/stackA/queue.txt; : > $QUEUE
for f in $FOLDS; do for d in $DATASETS; do echo "$d $f" >> $QUEUE; done; done
feature_job() {
  d=$1; f=$2; out=results/stack/${d}_f${f}/info.json
  [ -f "$out" ] && exit 0
  $PY -m reading_agent2.stack features --dataset $d --fold $f --threads 2 > logs/stackA/feat_${d}_f${f}.log 2>&1
}
export -f feature_job
# GPU loop: fit heads for every finished feature set, in queue order, until all are done
(
  while read d f; do
    while [ ! -f results/stack/${d}_f${f}/info.json ]; do
      pgrep -f "stack features" > /dev/null || [ -f results/stack/${d}_f${f}/info.json ] || { echo "missing features $d $f"; break; }
      sleep 20
    done
    [ -f results/stack/${d}_f${f}/info.json ] || continue
    grep -q "\"dataset\": \"$d\", \"fold\": $f," results/$TAG.jsonl 2>/dev/null && continue
    $PY -m reading_agent2.stack heads --dataset $d --fold $f --heads $HEADS --tag $TAG > logs/stackA/heads_${d}_f${f}.log 2>&1
  done < $QUEUE
  echo "heads loop finished"
) > logs/stackA/heads_loop.log 2>&1 &
xargs -P $WORKERS -L 1 bash -c 'feature_job $0 $1' < $QUEUE
wait
echo "stage A finished"
