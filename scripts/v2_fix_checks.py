#!/usr/bin/env python
"""Dataset-wide checks of DATASETv2_revision.md fixes 1 and 4 on a built split.

- Fix 4: share of each vehicle's footprint over a sidewalk polygon, every vehicle and frame
  (parked vehicles included), from the stored labels and scene.
- Fix 1: for every merge clip, pixels of the lane-ends sign per frame, from the render's
  object index (sign i has index 3000 + i).

    python scripts/v2_fix_checks.py /root/projects/drive_v2/val
"""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from matplotlib.path import Path as MP

ROOT = None


def one(n):
    sc = json.loads((ROOT / "scenes" / f"{n:06d}.json").read_text())
    a = np.load(ROOT / "labels" / f"{n:06d}.npz")
    sw = [MP(np.asarray(s["poly"])) for s in sc["surfaces"] if s["kind"] == "sidewalk"]
    ag = a["agents"]                       # (T, A, 10): type, x, y, h, v, acc, L, W, H, alive
    worst, frames = 0.0, 0
    for f in range(ag.shape[0]):
        for i in np.flatnonzero((ag[f, :, 9] > 0) & (ag[f, :, 0] >= 1) & (ag[f, :, 0] <= 4)):
            _, x, y, h, _, _, L, W = ag[f, i, :8]
            gx, gy = np.meshgrid(np.linspace(-L / 2, L / 2, 9), np.linspace(-W / 2, W / 2, 5))
            c, s = np.cos(h), np.sin(h)
            pts = np.stack([x + c * gx.ravel() - s * gy.ravel(), y + s * gx.ravel() + c * gy.ravel()], 1)
            share = float(np.mean(np.any([p.contains_points(pts) for p in sw], 0))) if sw else 0.0
            if share > 0.1:
                frames += 1
            worst = max(worst, share)
    sign = None
    if int(a["diversity"][3]) == META["diversity"]["D"].index("merge"):
        si = [k for k, s_ in enumerate(sc["signs"]) if s_["kind"] == "lane_ends"]
        ix = np.load(ROOT / "render" / f"{n:06d}.npz")["index"]
        sign = [int((ix[f] == 3000 + si[0]).sum()) for f in range(ix.shape[0])] if si else None
    return n, sc["template"], worst, frames, sign


def main():
    global ROOT, META
    ROOT = Path(sys.argv[1])
    META = json.loads((ROOT / "meta.json").read_text())
    N = META["num_samples"]
    with Pool(120, initializer=lambda: None) as p:
        res = p.map(one, range(N), chunksize=8)
    out = dict(n=N, fix4={}, fix1={})
    for t in sorted({r[1] for r in res}):
        R = [r for r in res if r[1] == t]
        out["fix4"][t] = dict(clips=len(R), clips_any_vehicle_over_10pct=sum(r[3] > 0 for r in R),
                              vehicle_frames_over_10pct=sum(r[3] for r in R), worst_share=round(max(r[2] for r in R), 3))
    S = [r[4] for r in res if r[4] is not None]
    if S:
        S = np.array(S)
        out["fix1"] = dict(merge_clips=len(S), sign_visible_f0=round(float((S[:, 0] > 0).mean()), 3),
                           sign_px_f0_median=float(np.median(S[:, 0])), sign_px_f5_median=float(np.median(S[:, 5])),
                           sign_visible_any_0_31=round(float((S[:, :32].max(1) > 0).mean()), 3))
    (ROOT / "fix_checks.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


META = None
if __name__ == "__main__":
    main()
