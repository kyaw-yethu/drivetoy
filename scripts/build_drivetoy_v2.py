#!/usr/bin/env python
"""Build a DriveToy v2 split (DATASETv2_plan.md): simulate with target labels, render, assemble.

    python scripts/build_drivetoy_v2.py --split pilot
    python scripts/build_drivetoy_v2.py --split train

Splits (clips, simulator seed base, futures per clip): pilot 500 / 30,000,000 / 1,
train 10,000 / 0 / 1, val 1,500 / 20,000,000 / 1, test 1,500 / 10,000,000 / 2 (branching at
frame 4, so 3,000 samples). A build resumes where it stopped.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["LD_LIBRARY_PATH"] = "/root/opt/cuda525:" + os.environ.get("LD_LIBRARY_PATH", "")   # Blender finds the GPUs only so

from f_toy.datasets.drivetoy.builder import build_to_disk
from f_toy.datasets.drivetoy.config import DRIVETOY_DEFAULT_CFG

SPLITS = dict(pilot=(500, 30_000_000, 1), train=(10_000, 0, 1), val=(1_500, 20_000_000, 1), test=(1_500, 10_000_000, 2),
              smoke=(8, 40_000_000, 1), pilot2=(500, 50_000_000, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=list(SPLITS))
    ap.add_argument("--out", default="/root/projects/drive_v2s")
    ap.add_argument("--forks", type=int, default=None, help="futures per sample (default DT_FORKS)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--gpus", default="auto")
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()
    n, seed, k = SPLITS[args.split]
    cfg = dict(DRIVETOY_DEFAULT_CFG)
    cfg.update(NUM_SAMPLES=n * k, SEED=seed, DT_BRANCHES=k, DT_RENDER_GPUS=args.gpus, DT_WRITE_VIDEOS=args.split != "train")
    if args.forks:
        cfg["DT_FORKS"] = args.forks
    build_to_disk(cfg, Path(args.out) / args.split, workers=args.workers, render=not args.no_render)


if __name__ == "__main__":
    main()
