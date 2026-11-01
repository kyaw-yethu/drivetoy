#!/bin/bash
# CPU conda env "navsim" for bench/: NAVSIM v2 and nuplan-devkit v1.2 under /root/opt, CPU torch
# (imported by navsim but unused here). Run once; the drivetoy code that built a split goes in
# /root/opt/drivetoy_v3_build (see bench/code.py).
set -e
mkdir -p /root/opt && cd /root/opt
if [ ! -x /root/miniconda3/bin/conda ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
  bash miniconda.sh -b -p /root/miniconda3 && rm miniconda.sh
fi
C=/root/miniconda3/bin/conda
$C env list | grep -q '^navsim ' || $C create -y -n navsim --override-channels -c conda-forge python=3.9
[ -d navsim ] || git clone -q https://github.com/autonomousvision/navsim.git
[ -d nuplan-devkit ] || git clone -q --branch nuplan-devkit-v1.2 https://github.com/motional/nuplan-devkit.git
P=/root/miniconda3/envs/navsim/bin/pip
$P install -q "numpy<2" "shapely>=2" scipy pandas pyquaternion rtree geopandas tqdm hydra-core omegaconf pytest \
  aiofiles opencv-python-headless pillow matplotlib rasterio aioboto3 retry requests
$P install -q torch --index-url https://download.pytorch.org/whl/cpu
$P install -q --no-deps -e nuplan-devkit
$P install -q --no-deps -e navsim
$P cache purge > /dev/null 2>&1 || true
echo done
