#!/usr/bin/env python
"""Simulate a few DriveToy clips and render them with Blender, for checking
the look by eye: writes per-clip ``<out>/<tag>.mp4`` (ego view next to the
top-down debug view) and ``<out>/sheet.png`` (frames 0, 8, 16, 24, 31 of
every clip).

    python scripts/drivetoy_preview.py --out analysis/drivetoy_preview --n 2 --gpu 3
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f_toy.datasets.drivetoy.config import DRIVETOY_DEFAULT_CFG
from f_toy.datasets.drivetoy.debug_view import render_topdown
from f_toy.datasets.drivetoy.export import write_scene
from f_toy.datasets.drivetoy.scenarios import simulate

BLENDER = os.environ.get("BLENDER", "/root/opt/blender/blender")
SCRIPT = Path(__file__).resolve().parent.parent / "f_toy/datasets/drivetoy/blender/render_clip.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1, help="clips per template")
    ap.add_argument("--templates", default="signal4,stop4,multilane,residential")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--img", type=int, default=128)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    work = Path("/dev/shm/drivetoy_preview")
    work.mkdir(parents=True, exist_ok=True)
    cfg = dict(DRIVETOY_DEFAULT_CFG, IMG=args.img)
    clips = []
    for t in args.templates.split(","):
        for k in range(args.n):
            seed = args.seed + 1000 * k + hash(t) % 997
            res = simulate(seed, cfg, template=t)
            tag = f"{t}_{seed}"
            write_scene(work / f"{tag}.json", res, 0, cfg, seed)
            clips.append((tag, res))
    jobs = work / "jobs.txt"
    jobs.write_text("".join(f"{work / (tag + '.json')}\t{work / (tag + '.npz')}\n" for tag, _ in clips))
    for tag, _ in clips:
        if (work / f"{tag}.npz").exists():
            os.remove(work / f"{tag}.npz")
    t0 = time.time()
    subprocess.run([BLENDER, "-b", "--python", str(SCRIPT), "--", "--jobs", str(jobs), "--gpu", str(args.gpu),
                    "--samples", str(args.samples)], check=True, stdout=open(work / "blender.log", "w"),
                   stderr=subprocess.STDOUT)
    print(f"rendered {len(clips)} clips in {time.time() - t0:.0f}s")
    rows = []
    for tag, res in clips:
        r = np.load(work / f"{tag}.npz")
        video = r["video"]
        T = video.shape[0]
        S = 384
        vw = cv2.VideoWriter(str(out / f"{tag}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 5, (2 * S, S))
        for f in range(T):
            ego = cv2.resize(video[f][:, :, ::-1], (S, S), interpolation=cv2.INTER_NEAREST)
            top = render_topdown(res["map"], res["branches"][0], f, size=S, span=80)
            vw.write(np.concatenate([ego, top], 1))
        vw.release()
        rows.append(np.concatenate([cv2.resize(video[f][:, :, ::-1], (256, 256), interpolation=cv2.INTER_NEAREST)
                                    for f in (0, 8, 16, 24, 31)], 1))
    cv2.imwrite(str(out / "sheet.png"), np.concatenate(rows, 0))
    print(f"wrote {out}/sheet.png and {len(clips)} mp4s")


if __name__ == "__main__":
    main()
