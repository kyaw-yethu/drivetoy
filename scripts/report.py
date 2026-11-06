#!/usr/bin/env python
"""Summary of built DriveToy v4 splits (docs/drivetoy-v4-design.md) as markdown: decision-frame shares, situations,
whether every change starts on the frame after the decision, the hardest braking after it, and the light, cause,
response and path labels against drive_v3's shares.

    python scripts/report.py /root/projects/drive_v4 [--splits train val test val_twins test_twins]
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from drivetoy.config import DRIVETOY_DEFAULT_CFG

ASPECTS = "ABCD"


def report(root, split):
    d = root / split
    a = np.load(d / "annotations.npz")
    meta = json.loads((d / "meta.json").read_text())
    fams = meta["families"]
    n = len(a["family"])
    fd = a["decision_frame"].astype(int) if "decision_frame" in a else np.full(n, 31)
    fam = [fams[i] for i in a["family"]]
    ego = a["ego"].astype(float)
    print(f"\n## {split}: {n} clips\n")
    bins = Counter("none" if f < 0 else (str(f) if f in (7, 15, 23, 31) else "other") for f in fd)
    print("| Decision frame | Clips | Share |\n|---|---|---|")
    for k in ("7", "15", "23", "31", "other", "none"):
        if bins.get(k):
            print(f"| {k} | {bins[k]} | {bins[k] / n:.1%} |")
    onset = a["onset"][:, 0].astype(int)
    change = a["change"][:, 0].astype(bool)
    has = fd >= 0
    exact = (onset == fd + 1) & change & has
    brake = np.zeros(n)
    for i in np.flatnonzero(has):
        v = ego[i, fd[i]:fd[i] + 21, 3]
        brake[i] = float((-np.diff(v) * 5.0).max()) if len(v) > 1 else 0.0
    print(f"\nOf {int((change & has).sum())} clips whose ground truth carries out the change, "
          f"{int(exact.sum())} start it on the frame after the decision. Hardest braking within 4 s of the decision: "
          f"median {np.median(brake[has]):.1f}, 99th percentile {np.percentile(brake[has], 99):.1f}, "
          f"max {brake[has].max():.1f} m/s^2.\n")
    print("| Situation | Clips | Ground truth changes | Carries on | No decision |\n|---|---|---|---|---|")
    for name, c in sorted(Counter(fam).items(), key=lambda kv: -kv[1]):
        idx = [i for i in range(n) if fam[i] == name]
        ch = sum(bool(change[i]) and fd[i] >= 0 for i in idx)
        nd = sum(fd[i] < 0 for i in idx)
        print(f"| {name} | {c} | {ch} | {c - ch - nd} | {nd} |")
    labels = {k: [] for k in ASPECTS}
    for i in range(n):
        lab = np.load(d / "labels" / f"{i:06d}.npz", allow_pickle=True)
        div = lab["diversity"]
        for j, k in enumerate(ASPECTS):
            labels[k].append(int(div[j]))
    resp, path = Counter(), Counter()
    for i in range(n):   # the response and path over all 64 frames, from the ego's own motion
        e = ego[i]
        v, h = e[:, 3], np.unwrap(e[:, 2])
        turn = h[-1] - h[0]
        c, sn = np.cos(h[0]), np.sin(h[0])
        lat = -sn * (e[:, 0] - e[0, 0]) + c * (e[:, 1] - e[0, 1])
        if turn > 0.8:
            path["left turn"] += 1
        elif turn < -0.8:
            path["right turn"] += 1
        elif abs(lat[-1]) > 2.5:
            path["lane change or merge"] += 1
        elif np.abs(lat).max() > 1.0:
            path["nudge around"] += 1
        else:
            path["straight"] += 1
        low = v < 0.3
        if low[0] and not low[-1]:
            resp["pull away"] += 1
        elif not low[0] and low.any() and not low[-1]:
            resp["stop and go"] += 1
        elif not low[0] and low[-1]:
            resp["stop"] += 1
        elif low.all():
            resp["wait"] += 1
        elif v.min() < v[0] - 1.5:
            resp["slow down"] += 1
        else:
            resp["cruise or follow"] += 1
    print("- response over frames 0-63: " + ", ".join(f"{k} {c / n:.0%}" for k, c in resp.most_common()))
    print("- path over frames 0-63: " + ", ".join(f"{k} {c / n:.0%}" for k, c in path.most_common()))
    print("- drive_v3 labels over frames 0-31 (share in drive_v3 in brackets):")
    shares = DRIVETOY_DEFAULT_CFG["DT_SHARES"]
    for j, k in enumerate(ASPECTS):
        names = list(shares[k])
        cnt = Counter(labels[k])
        row = ", ".join(f"{nm} {cnt.get(t, 0) / n:.0%} ({shares[k][nm]:.0%})" for t, nm in enumerate(names))
        print(f"- {k}: {row}; unlabelled {cnt.get(-1, 0) / n:.0%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test", "val_twins", "test_twins"])
    args = ap.parse_args()
    root = Path(args.root)
    print("# DriveToy v4 build report")
    for split in args.splits:
        if (root / split / "annotations.npz").exists():
            report(root, split)


if __name__ == "__main__":
    main()
