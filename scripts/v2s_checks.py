#!/usr/bin/env python
"""Checks of a stochastic build (README, "Labels and checks"):

1. shares of aspects A to E against their targets;
2. pending samples whose futures show both outcomes of the pending choice (the pilot passes at
   70%), and the ego's spread at frame 63 across the futures, per E label;
3. the realised share of the assertive outcome on the futures against the drawn odds, binned;
4. test sibling pairs whose ego trajectories differ by more than 2 m in frames 5 to 31;
5. with --resim: every sample re-simulated from its sim_seed and target must equal its labels,
   and rules.check must find no violations.

    python scripts/v2s_checks.py /root/projects/drive_v2s/pilot [--resim]

Writes <split>/checks.json.
"""
import argparse
import json
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from f_toy.datasets.drivetoy.choices import CHOICE_KINDS
from f_toy.datasets.drivetoy.diversity import ASPECTS
from f_toy.datasets.drivetoy.rules import check
from f_toy.datasets.drivetoy.scenarios import VARIANTS, simulate

ROOT = META = None
T0, T1 = 31, 40


def pending_choice(dec):
    """(slot, kind, p) of the first choice concerning the ego in frames T0+1 to T1 with odds in [0.1, 0.9]."""
    d = dec[(dec[:, 0] > T0) & (dec[:, 0] <= T1) & (dec[:, 6] > 0) & (dec[:, 4] >= 0.1) & (dec[:, 4] <= 0.9)]
    return None if not len(d) else (int(d[0, 1]), int(d[0, 2]), float(d[0, 4]))


def resim(n):
    a = np.load(ROOT / "labels" / f"{n:06d}.npz")
    cfg = META["cfg"]
    tgt = {k: (ASPECTS[k][c] if c >= 0 else None) for k, c in zip("ABCDE", a["target"])}
    tgt["template"] = META["templates"][int(a["template"])]
    tgt["V"] = VARIANTS[int(a["variant"])] if int(a["variant"]) >= 0 else None
    r = simulate(int(a["sim_seed"]), cfg, target=tgt)
    rec = r["branches"][int(a["branch_id"])]
    viol = [(int(f), k, int(ag), str(x)) for f, k, ag, x in check(rec, r["map"], cfg)]
    return n, bool(np.array_equal(rec["agents"], a["agents"])), viol


def main():
    global ROOT, META
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--resim", action="store_true")
    ap.add_argument("--workers", type=int, default=120)
    args = ap.parse_args()
    ROOT = Path(args.root)
    META = json.loads((ROOT / "meta.json").read_text())
    ann = np.load(ROOT / "annotations.npz")
    fs = np.load(ROOT / "future_set.npz")
    div, dec, fdec, fego = ann["diversity"], ann["decisions"], fs["decisions"], fs["ego"]
    N = len(div)
    shares = META["cfg"]["DT_SHARES"]
    out = dict(n=N, shares={})
    for j, k in enumerate("ABCDE"):
        c = Counter(ASPECTS[k][x] for x in div[:, j])
        out["shares"][k] = {l: dict(target=shares[k].get(l), built=round(c[l] / N, 4)) for l in ASPECTS[k]}

    # pending choices on the futures
    both, odds, kinds = [], [], Counter()
    for n in range(N):
        if ASPECTS["E"][div[n, 4]] != "pending":
            continue
        pc = pending_choice(dec[n])
        if pc is None:
            continue
        slot, kind, p = pc
        kinds[CHOICE_KINDS[kind]] += 1
        outs = []
        for f in fdec[n]:
            m = f[(f[:, 1] == slot) & (f[:, 2] == kind) & (f[:, 0] > T0)]
            if len(m):
                outs.append(int(m[0, 5]))
        if outs:
            both.append(len(set(outs)) > 1)
            odds.append((p, float(np.mean(outs[1:])) if len(outs) > 1 else np.nan, len(outs)))
    out["pending"] = dict(samples=int(sum(ASPECTS["E"][x] == "pending" for x in div[:, 4])), with_choice=len(both),
                          both_outcomes=round(float(np.mean(both)), 3) if both else None, kinds=dict(kinds),
                          passes=bool(both) and float(np.mean(both)) >= 0.7)
    bins = [(0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9)]
    out["odds"] = {f"{lo:.1f}-{hi:.1f}": dict(n=sum(lo <= p < hi for p, _, _ in odds),
                                             mean_p=round(float(np.nanmean([p for p, _, _ in odds if lo <= p < hi] or [np.nan])), 3),
                                             realised=round(float(np.nanmean([r for p, r, _ in odds if lo <= p < hi] or [np.nan])), 3))
                   for lo, hi in bins}
    spread = np.max(np.linalg.norm(fego[:, :, -1, :2] - fego[:, :1, -1, :2], axis=-1), axis=1)
    out["spread63"] = {l: dict(n=int((div[:, 4] == i).sum()), over_2m=round(float((spread[div[:, 4] == i] > 2).mean()), 3)
                               if (div[:, 4] == i).any() else None, median_m=round(float(np.median(spread[div[:, 4] == i])), 2)
                               if (div[:, 4] == i).any() else None)
                       for i, l in enumerate(ASPECTS["E"])}
    K = META["branches"]
    if K > 1:
        ego = ann["ego"]
        d = [float(np.max(np.linalg.norm(ego[i, 5:32, :2] - ego[i + 1, 5:32, :2], axis=-1))) for i in range(0, N, K)]
        out["siblings_ego_over_2m"] = round(float(np.mean(np.array(d) > 2.0)), 3)
    if args.resim:
        with Pool(args.workers) as p:
            res = p.map(resim, range(N), chunksize=2)
        out["resim_equal"] = sum(r[1] for r in res)
        viol = [(n, v) for n, _, vs in res for v in vs]
        out["violations"] = len(viol)
        out["violation_rules"] = dict(Counter(v[1] for _, v in viol))
        out["violation_examples"] = [(n, v) for n, v in viol[:20]]
    (ROOT / "checks.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
