#!/bin/bash
# NAVSIM-style evaluation of one drive_v4 split (SPLIT=val, test, val_twins or test_twins): records (system python,
# the build env), then scores and tables (navsim env). The plans to score go in $WORK/plans (Task_specific_JDM docs/TASK_ENCODER_EVAL.md,
# Plans); without any, only the references are scored.
set -e
cd "$(dirname "$0")/.."
SPLIT=${SPLIT:-val}
DATA=${DATA:-/root/projects/drive_v4/$SPLIT}
WORK=${WORK:-/root/projects/drive_v4_bench/$SPLIT}
PY=/root/miniconda3/envs/navsim/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "$WORK/plans"
python3 -m bench.records --data "$DATA" --out "$WORK/records" --workers ${WORKERS:-64}
$PY -m bench.score --plans "$WORK/plans" --records "$WORK/records" --data "$DATA" --out "$WORK/scores.npz" --workers ${WORKERS:-64}
NAMES=(); [ -f "$WORK/plans/names.json" ] && NAMES=(--names "$WORK/plans/names.json")
$PY -m bench.report --scores "$WORK/scores.npz" --data "$DATA" --out "$WORK/navsim_eval.md" "${NAMES[@]}"
