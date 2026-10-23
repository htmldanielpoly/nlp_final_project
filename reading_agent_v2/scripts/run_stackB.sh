#!/bin/bash
# Stage B on the VM (GPU): TabLoRA cross-fitted reader features, fold by fold, then TabM heads on them.
# Usage: bash scripts/run_stackB.sh "0 1 2 3 4" 2 1   (folds, parallel GPU workers, gate)
cd "$(dirname "$0")/.." || exit 1
FOLDS=${1:-"0 1 2 3 4"}; WORKERS=${2:-2}; GATE=${3:-1}
TAG=$([ "$GATE" = 1 ] && echo stackB || echo stackB_nogate)
export PYTHONPATH=src PY=.venv/bin/python GATE TAG PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs/$TAG
DATASETS=$($PY -c "from reading_agent.data import TEXT_DATASETS; print(' '.join(TEXT_DATASETS))")
QUEUE=logs/$TAG/queue.txt; : > $QUEUE
for f in $FOLDS; do for d in $DATASETS; do echo "$d $f" >> $QUEUE; done; done
job() {
  d=$1; f=$2
  [ -f results/$TAG/${d}_f${f}/info.json ] || \
    $PY -m reading_agent2.tabcond --dataset $d --fold $f --gate $GATE --tag $TAG > logs/$TAG/${d}_f${f}.log 2>&1
}
export -f job
xargs -P $WORKERS -L 1 bash -c 'job $0 $1' < $QUEUE
echo "stage B ($TAG) finished"
