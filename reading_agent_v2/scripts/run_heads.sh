#!/bin/bash
# Heads (GPU): for each (dataset, fold), once stage A and stage B features exist, fit TabM + TabPFN once
# and combine them with both readers. Fold 0 first. Usage: bash scripts/run_heads.sh "0 1 2 3 4" v2
cd "$(dirname "$0")/.." || exit 1
FOLDS=${1:-"0 1 2 3 4"}; TAG=${2:-v2}
. ~/.tabpfn_env
export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
mkdir -p logs/heads
DATASETS=$($PY -c "from reading_agent.data import TEXT_DATASETS; print(' '.join(TEXT_DATASETS))")
for f in $FOLDS; do
  for d in $DATASETS; do
    while [ ! -f results/stack/${d}_f${f}/info.json ] || [ ! -f results/stackB/${d}_f${f}/info.json ]; do
      grep -q Traceback logs/stackB/${d}_f${f}.log 2>/dev/null && break   # crashed: post_stackB re-runs it
      if ! pgrep -f "stack feature[s]|tabcon[d]" > /dev/null; then break; fi
      sleep 30
    done
    if [ ! -f results/stack/${d}_f${f}/info.json ] || [ ! -f results/stackB/${d}_f${f}/info.json ]; then
      echo "missing features for $d $f; skipped"; continue
    fi
    # stage-B runs without saved weights are re-trained by post_stackB, which then fits their heads
    [ -f results/stackB/${d}_f${f}/weights/reader_2.pt ] || { echo "no weights for $d $f; deferred"; continue; }
    grep -q "\"dataset\": \"$d\", \"fold\": $f," results/$TAG.jsonl 2>/dev/null && continue
    $PY -m reading_agent2.stack heads --dataset $d --fold $f --tag $TAG > logs/heads/${d}_f${f}.log 2>&1 \
      && echo "done $d $f" || echo "FAILED $d $f"
  done
done
echo "heads finished"
