#!/bin/bash
# Build DriveToy v3: val, test and train, then the checks (val re-simulated) and the GIFs.
#   scripts/run_all.sh [out_dir]    (default /root/projects/drive_v3; status lines in <out_dir>/logs/status.log)
cd "$(dirname "$0")/.." || exit 1
export LD_LIBRARY_PATH=/root/opt/cuda525:$LD_LIBRARY_PATH   # Blender's CUDA libraries on yt-workspace
D=${1:-/root/projects/drive_v3}
L=$D/logs
mkdir -p $L
st() { echo "$(date -u '+%F %T') $*" >> $L/status.log; }
for s in val test train; do
  st "build $s"
  python3 scripts/build_drivetoy_v3.py --split $s --out $D --gpus 0,1,2,3 > $L/build_$s.log 2>&1 || { st "build $s FAILED"; exit 1; }
done
st "checks"
python3 scripts/v3_checks.py $D/val --resim > $L/checks_val.log 2>&1 || st "checks val FAILED"
python3 scripts/v3_checks.py $D/test > $L/checks_test.log 2>&1 || st "checks test FAILED"
python3 scripts/v3_checks.py $D/train > $L/checks_train.log 2>&1 || st "checks train FAILED"
st "viz"
python3 scripts/v2s_viz.py $D/val --out $D/viz --suffix v3 > $L/viz.log 2>&1 || st "viz FAILED"
st "DONE build"
