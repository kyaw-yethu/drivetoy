#!/usr/bin/env python
"""Difficulty / content check of a built DriveToy dataset (docs/DRIVETOY.md):
what happens in the frames a model has to generate, how visible the cues
are, and how different the branches of one context look.

    python scripts/drivetoy_difficulty.py --dataset-dir data/drivetoy/dt_1k_k2 [--rules]
"""
import argparse
import collections
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f_toy.datasets.drivetoy.scenarios import DECISION_EVENTS, _in_view
from f_toy.datasets.drivetoy.sim import EVENTS, RULE_STATES

ROOT = None


def _rules_one(task):
    from f_toy.datasets.drivetoy.rules import check
    from f_toy.datasets.drivetoy.scenarios import simulate
    clip, seed, cfg = task
    r = simulate(seed + clip, cfg)
    return clip, [[(int(f), rule, int(a)) for f, rule, a, _ in check(b, r["map"], cfg)] for b in r["branches"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--rules", action="store_true", help="re-simulate every clip and run rules.check on it")
    args = ap.parse_args()
    root = Path(args.dataset_dir)
    meta = json.loads((root / "meta.json").read_text())
    a = dict(np.load(root / "annotations.npz"))     # materialise once: npz members re-read on every access
    videos = np.load(root / "videos.npy", mmap_mode="r")
    cfg = meta["cfg"]
    N, T, K = meta["num_samples"], meta["frames"], meta["branches"]
    lo = meta["branch_frame"] + 1
    tmpl = a["template"]
    names = meta["templates"]
    out = {}
    print(f"{N} samples, {N // K} contexts x {K} branches; sim_ok {a['sim_ok'].mean():.3f}; "
          f"templates {dict(zip(names, np.bincount(tmpl, minlength=len(names)).tolist()))}")

    # ---- decision events in view, in the generated frames ----
    per = collections.defaultdict(list)
    kinds = collections.defaultdict(collections.Counter)
    for n in range(N):
        cnt = 0
        for f, typ, ag, b, val in a["events"][n][:a["num_events"][n]]:
            f, typ, ag = int(f), int(typ), int(ag)
            if not (lo <= f < T) or typ not in DECISION_EVENTS:
                continue
            ego = a["ego"][n, f]
            if typ == EVENTS.index("SIGNAL") or (ag >= 0 and a["agents"][n, f, ag, 9] > 0 and _in_view(ego, a["agents"][n, f, ag, 1:3], cfg)):
                cnt += 1
                name = EVENTS[typ] + ("_" + RULE_STATES[int(val)] if EVENTS[typ] == "YIELD" else "")
                kinds[names[tmpl[n]]][name] += 1
        per[names[tmpl[n]]].append(cnt)
    print(f"\ndecision events in the ego's view, frames {lo}-{T - 1} (per clip):")
    for t, c in per.items():
        c = np.array(c)
        top = ", ".join(f"{k} {v / len(c):.2f}" for k, v in kinds[t].most_common(6))
        print(f"  {t:12s} mean {c.mean():.1f}  >=1 {np.mean(c >= 1):.2f}  >=3 {np.mean(c >= 3):.2f}   {top}")
        out[f"events_{t}"] = dict(mean=float(c.mean()), ge1=float(np.mean(c >= 1)), ge3=float(np.mean(c >= 3)))

    # ---- the ego itself ----
    v = a["ego"][:, lo:T, 3]
    rs = a["rule_state"][:, lo:T, 0]
    print(f"\nego in the generated frames: speed {v.mean():.1f} m/s (p10 {np.percentile(v, 10):.1f}, p90 {np.percentile(v, 90):.1f}); "
          f"stopped {np.mean(v < 0.3):.2f} of frames; clips with a stop {np.mean((v < 0.3).any(1)):.2f}; "
          f"speed change >3 m/s {np.mean(v.max(1) - v.min(1) > 3):.2f}")
    share = collections.Counter(RULE_STATES[int(x)] for x in rs.ravel())
    print("ego binding rule:", ", ".join(f"{k} {c / rs.size:.2f}" for k, c in share.most_common()))

    # ---- cues ----
    cv = a["cue_visibility"]
    print("\ncue visibility (share of clips with >= 4 px in some generated frame | median px when visible):")
    for c, name in enumerate(meta["cues"]):
        x = cv[:, lo:T, c]
        vis = x[x >= 4]
        row = "  ".join(f"{t} {np.mean((x[tmpl == i] >= 4).any(1)):.2f}" for i, t in enumerate(names) if (tmpl == i).any())
        print(f"  {name:14s} all {np.mean((x >= 4).any(1)):.2f} | {np.median(vis) if len(vis) else 0:5.0f} px   ({row})")

    # ---- branches ----
    if K > 1:
        d = []
        for i in range(N // K):
            v0 = videos[i * K].astype(np.int16)
            for b in range(1, K):
                diff = np.abs(videos[i * K + b].astype(np.int16) - v0).mean(axis=(0, 2, 3))   # per frame
                d.append(diff)
        d = np.array(d)
        print(f"\nbranches: mean |pixel diff| per frame vs branch 0 — frames 0-{lo - 1}: {d[:, :lo].max():.3f} (must be 0); "
              f"frame 15: {d[:, 15].mean():.2f}, frame 31: {d[:, 31].mean():.2f}; "
              f"pairs visibly different by frame 31 (>1.0): {np.mean(d[:, 31] > 1.0):.2f}")
        out["branch_diff_31"] = float(d[:, 31].mean())

    if args.rules:
        sim_cfg = dict(cfg, DT_VIOLATION=None)
        tasks = [(i, meta["seed"], sim_cfg) for i in range(N // K)]
        viol = collections.Counter()
        with ProcessPoolExecutor(64) as ex:
            for clip, per_branch in ex.map(_rules_one, tasks, chunksize=4):
                for fl in per_branch:
                    for f, rule, ag in fl:
                        viol[rule] += 1
        print(f"\nrules.check over all {N} samples: {dict(viol) or 'no violations'}")
        out["violations"] = dict(viol)
    (root / "difficulty.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
