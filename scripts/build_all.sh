#!/bin/bash
# Build all five DriveToy v4 splits from plans/ into OUT (default /root/projects/drive_v4) and write OUT/report.md.
# Each split is simulated on the CPUs while the one before it renders. Only one split renders at a time: each
# Blender process takes about 1.8 GB of a 24 GB GPU, and two splits at once run the GPUs out of memory.
# Rerunning resumes where it stopped. Progress: OUT/build.log and OUT/<split>/logs/status.log; marker OUT/build.done.
#
#     bash scripts/build_all.sh [OUT]        # GPUS=0,1,2,3 by default
cd "$(dirname "$0")/.." || exit 1
OUT=${1:-/root/projects/drive_v4}
GPUS=${GPUS:-0,1,2,3}
mkdir -p "$OUT"
rm -f "$OUT/build.done"
log() { echo "$(TZ=UTC-9 date '+%F %T') KST $*" >> "$OUT/build.log"; }
fail() { log "FAILED: $*"; exit 1; }

rendering="" pid=""
for split in val_twins test_twins train val test; do
  mkdir -p "$OUT/$split/logs"
  log "$split: simulating"
  python3 -u scripts/build.py --split "$split" --out "$OUT" --no-render >> "$OUT/$split/logs/build.out" 2>&1 \
    || fail "$split simulation"
  if [ -n "$pid" ]; then wait "$pid" || fail "$rendering render"; fi
  log "$split: rendering"
  python3 -u scripts/build.py --split "$split" --out "$OUT" --gpus "$GPUS" >> "$OUT/$split/logs/build.out" 2>&1 &
  rendering=$split pid=$!
done
wait "$pid" || fail "$rendering render"
python3 scripts/report.py "$OUT" > "$OUT/report.md" || fail report
log "done"
touch "$OUT/build.done"
