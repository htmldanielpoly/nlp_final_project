#!/bin/bash
# Second stage-B GPU worker: same jobs as run_stackB.sh in reverse fold order; skips (dataset, fold) pairs
# that are finished or already claimed (log file exists). Usage: bash scripts/run_stackB_rev.sh "4 3 2 1"
cd "$(dirname "$0")/.." || exit 1
FOLDS=${1:-"4 3 2 1"}; TAG=stackB
export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
DATASETS=$($PY -c "from reading_agent.data import TEXT_DATASETS; print(' '.join(TEXT_DATASETS))")
for f in $FOLDS; do
  for d in $DATASETS; do
    [ -f results/$TAG/${d}_f${f}/info.json ] && continue
    [ -f logs/$TAG/${d}_f${f}.log ] && continue
    $PY -m reading_agent2.tabcond --dataset $d --fold $f --gate 1 --tag $TAG > logs/$TAG/${d}_f${f}.log 2>&1
  done
done
echo "reverse worker finished"
