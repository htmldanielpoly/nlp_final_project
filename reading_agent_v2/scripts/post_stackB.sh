#!/bin/bash
# After the gated stage-B sweep (run on the VM):
#  1. once no stage-B worker is left, (re-)run every gated run without complete outputs: runs that
#     finished before weight saving was added (old outputs moved to results/stackB_prepatch, same
#     code and recipe otherwise) and runs that crashed,
#  2. once the heads loop is idle, re-run their heads so every reported score comes from saved weights,
#  3. the gate ablation (fold 0, plain LoRA).
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
while pgrep -f "run_stack[B]" > /dev/null || pgrep -f "tabcon[d].*--tag stackB$" > /dev/null; do
  sleep 120
done
mkdir -p results/stackB_prepatch logs/stackB_prepatch logs/export
LIST=logs/rerun_list.txt; : > $LIST
DATASETS=$($PY -c "from reading_agent.data import TEXT_DATASETS; print(' '.join(TEXT_DATASETS))")
for f in 0 1 2 3 4; do
  for d in $DATASETS; do
    run=${d}_f${f}
    [ -f results/stackB/$run/info.json ] && [ -f results/stackB/$run/weights/reader_2.pt ] && continue
    [ -d results/stackB/$run ] && rm -rf results/stackB_prepatch/$run && mv results/stackB/$run results/stackB_prepatch/
    [ -f logs/stackB/$run.log ] && mv logs/stackB/$run.log logs/stackB_prepatch/
    echo "$d $f" >> $LIST
  done
done
echo "step 1: re-running $(wc -l < $LIST) stage-B runs"
job() { [ -f results/stackB/$1_f$2/weights/reader_2.pt ] && return
        .venv/bin/python -m reading_agent2.tabcond --dataset $1 --fold $2 --gate 1 --tag stackB > logs/stackB/$1_f$2.log 2>&1; }
export -f job
xargs -P 2 -L 1 bash -c 'job $0 $1' < $LIST
xargs -P 1 -L 1 bash -c 'job $0 $1' < $LIST   # retry anything that crashed (e.g. GPU OOM)
echo "step 2: heads (waiting for the heads loop to finish)"
while pgrep -f "run_head[s]" > /dev/null; do sleep 60; done
. ~/.tabpfn_env
cp results/v2.jsonl results/v2_prepatch.jsonl
while read d f; do
  grep -v "\"dataset\": \"$d\", \"fold\": $f," results/v2.jsonl > results/v2.tmp && mv results/v2.tmp results/v2.jsonl
  $PY -m reading_agent2.stack heads --dataset $d --fold $f --tag v2 > logs/heads/${d}_f${f}.log 2>&1 \
    && echo "heads done $d $f" || echo "heads FAILED $d $f"
done < $LIST
echo "step 2b: heads of any run still missing from results/v2.jsonl (e.g. an earlier TabPFN OOM)"
bash scripts/run_heads.sh "0 1 2 3 4" v2 | grep -v "^done"
missing=$(for r in results/stackB/*_f*/; do [ -f $r/weights/reader_2.pt ] || echo $r; done)
[ -n "$missing" ] && { echo "still incomplete: $missing"; exit 1; }
echo "step 3: gate ablation, and exports of all runs"
bash scripts/run_stackB.sh "0" 2 0 &
for f in 0 1 2 3 4; do
  for d in $DATASETS; do
    [ -f results/models/v2/${d}_f${f}/manifest.json ] && continue
    $PY -m reading_agent2.estimator2 export --dataset $d --folds $f > logs/export/${d}_f${f}.log 2>&1       || echo "export FAILED $d $f"
  done
done
echo "exports: $(ls results/models/v2/*/manifest.json | wc -l)"
for run in BIN_TEXT_FAKE_JOB_POSTING:1 MUL_TEXT_WINE_REVIEW:2 REG_TEXT_BOOK_PRICE:3 MUL_TEXT_US_ACCIDENTS:4; do
  $PY -m reading_agent2.estimator2 check --dataset ${run%:*} --fold ${run#*:}
done
wait
echo "post_stackB finished"
