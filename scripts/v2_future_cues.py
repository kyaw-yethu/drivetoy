#!/usr/bin/env python
"""How much of the ego's future (frames 32-63) is cued by the visible frames 0-31.

Per sample, from the stored labels and the render's object index (agent slot a has index a+1):
- turn: the ego's heading changes by more than 45 deg in frames 32-63; cued if its indicator in
  that direction was on in some frame 0-31;
- lane change: the ego starts a lane change in frames 32-63; cued if its indicator was on by 31;
- binding agents: frames 32-63 in which the ego follows or yields to an agent (rule_state with
  yield_to >= 0), split by whether that agent was visible (>= 1 px) in some frame 0-31.

    python scripts/v2_future_cues.py /root/projects/drive_v2/val
"""
import json, sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from f_toy.datasets.drivetoy.sim import RULE_STATES

ROOT = None
T0 = 31


def one(n):
    a = np.load(ROOT / "labels" / f"{n:06d}.npz")
    ix = np.load(ROOT / "render" / f"{n:06d}.npz")["index"]
    ego, ind, rs, yt, lc = a["ego"], a["indicator"][:, 0], a["rule_state"][:, 0], a["yield_to"][:, 0], a["lc"][:, 0]
    h = np.unwrap(ego[:, 2])
    res = dict(turn=None, lc=None)
    # turn in the future: heading change over frames 32-63 beyond 45 deg
    dh = np.degrees(h[63] - h[T0])
    if abs(dh) > 45:
        want = 1 if dh > 0 else 2
        res["turn"] = bool((ind[: T0 + 1] == want).any())
    # lane change starting in the future
    mv = np.flatnonzero(lc > 0)
    starts = [f for f in mv if f > T0 and lc[f - 1] == 0]
    if starts:
        res["lc"] = bool((ind[: T0 + 1] > 0).any())
    seen = set(int(i) - 1 for i in np.unique(ix) if 1 <= i < 1000)
    bound = Counter()
    for f in range(T0 + 1, 64):
        y = int(yt[f])
        if y >= 0 and y != 0:
            bound[(RULE_STATES[int(rs[f])], y in seen)] += 1
    res["bound"] = dict(bound)
    res["unseen_clip"] = sum(c for (s, v), c in bound.items() if not v) >= 4
    return n, res


def main():
    global ROOT
    ROOT = Path(sys.argv[1])
    N = json.loads((ROOT / "meta.json").read_text())["num_samples"]
    with Pool(120) as p:
        res = dict(p.map(one, range(N), chunksize=8))
    turns = [r["turn"] for r in res.values() if r["turn"] is not None]
    lcs = [r["lc"] for r in res.values() if r["lc"] is not None]
    by = Counter()
    for r in res.values():
        for k, c in r["bound"].items():
            by[k] += c
    states = sorted({s for s, _ in by})
    out = dict(n=N,
               future_turns=dict(samples=len(turns), cued=sum(turns)),
               future_lane_changes=dict(samples=len(lcs), cued=sum(lcs)),
               bound_frames={s: dict(seen=by[(s, True)], unseen=by[(s, False)]) for s in states},
               samples_bound_4plus_frames_to_unseen=sum(r["unseen_clip"] for r in res.values()),
               examples_unseen=[n for n, r in res.items() if r["unseen_clip"]][:15])
    (ROOT / "future_cues.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
