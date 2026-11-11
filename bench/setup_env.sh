#!/bin/bash
# One-time setup for bench/: the CPU conda env "navsim" (NAVSIM v2 at commit 0a380a9, nuplan-devkit v1.2, CPU torch,
# the package versions of bench/requirements-navsim.txt) and /root/opt/drivetoy_v4_build, the code that built
# drive_v4 (bench/code.py). Run from anywhere: bash bench/setup_env.sh
set -e
REPO=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p /root/opt && cd /root/opt
if [ ! -x /root/miniconda3/bin/conda ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
  bash miniconda.sh -b -p /root/miniconda3 && rm miniconda.sh
fi
C=/root/miniconda3/bin/conda
$C env list | grep -q '^navsim ' || $C create -y -n navsim --override-channels -c conda-forge python=3.9
[ -d navsim ] || { git clone -q https://github.com/autonomousvision/navsim.git && git -C navsim checkout -q 0a380a9063d7162ec93d0f51e9990ebac585f720; }
[ -d nuplan-devkit ] || git clone -q --branch nuplan-devkit-v1.2 https://github.com/motional/nuplan-devkit.git
P=/root/miniconda3/envs/navsim/bin/pip
$P install -q torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
$P install -q -r "$REPO/bench/requirements-navsim.txt"
$P install -q --no-deps -e nuplan-devkit
$P install -q --no-deps -e navsim
$P cache purge > /dev/null 2>&1 || true
# drivetoy/, scripts/build.py and plans/ of commit c6d3763, which built every drive_v4 split
B=/root/opt/drivetoy_v4_build
[ -d $B ] || { mkdir -p $B && git -C "$REPO" archive c6d3763 drivetoy scripts/build.py plans | tar -x -C $B; }
echo done
