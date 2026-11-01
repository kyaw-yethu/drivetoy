#!/bin/bash
# NAVSIM-style evaluation of drive_v3 val: records (system python, the build env), then scores and
# tables (navsim env). Plans come from Task_specific_JDM_waypoint scripts/task_encoder_eval/export_plans.py.
set -e
cd "$(dirname "$0")/.."
DATA=${DATA:-/root/projects/drive_v3/val}
WORK=${WORK:-/root/projects/drive_v3_bench/val}
PY=/root/miniconda3/envs/navsim/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
python3 -m bench.records --data "$DATA" --out "$WORK/records" --workers ${WORKERS:-96}
$PY -m bench.score --plans "$WORK/plans" --records "$WORK/records" --data "$DATA" --out "$WORK/scores.npz" --workers ${WORKERS:-96}
$PY -m bench.report --scores "$WORK/scores.npz" --data "$DATA" --out "$WORK/navsim_eval.md" --names "$WORK/plans/names.json"
