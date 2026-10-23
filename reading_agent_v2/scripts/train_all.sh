#!/bin/bash
# End-to-end training mode: reproduces every published v2 artifact from scratch on one CUDA GPU.
# Usage: bash scripts/train_all.sh ["DATASET ..."] ["0 1 2 3 4"]
# Needs TABPFN_TOKEN (Prior Labs licence for the TabPFN-2.5 weights) in the environment.
# The benchmark runs used the same steps with several workers in parallel
# (scripts/run_stackA.sh, run_stackB.sh, run_heads.sh); the order below is what matters.
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-.venv/bin/python}
DATASETS=${1:-$($PY -c "from reading_agent.data import TEXT_DATASETS; print(' '.join(TEXT_DATASETS))")}
FOLDS=${2:-"0 1 2 3 4"}
set -e
$PY -m reading_agent.precompute --protocol paper $DATASETS        # frozen E5 cache (items + whole fields)
for f in $FOLDS; do
  for d in $DATASETS; do
    $PY -m reading_agent2.stack features --dataset $d --fold $f     # Joint-Frozen frame (+ frozen-item readers, for the ablation)
    $PY -m reading_agent2.tabcond --dataset $d --fold $f --gate 1   # 3 cross-fitted TabLoRA readers -> results/stackB
    $PY -m reading_agent2.stack heads --dataset $d --fold $f --tag v2   # TabM + TabPFN-2.5, blend weights, test scores
    $PY -c "from reading_agent2.estimator2 import export; export('$d', $f)"   # artifact -> results/models/v2
  done
done
