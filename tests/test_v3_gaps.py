"""Issue 3: oncoming gaps are random, 2 s plus an exponential tail with a mean of 4 to 8 s.
python3 tests/test_v3_gaps.py"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from f_toy.datasets.drivetoy.config import DRIVETOY_DEFAULT_CFG as CFG
from f_toy.datasets.drivetoy.roads import build_signal4
from f_toy.datasets.drivetoy.scenarios import COMBOS, simulate
from f_toy.datasets.drivetoy.sim import EV, Sim

V3 = dict(CFG, DT_FORKS=1)


def test_headway_law():
    rng = np.random.RandomState(0)
    m = build_signal4(rng, V3)
    sim = Sim(m, V3, rng)
    lid = m.spawn_lanes[1]
    sim.gap_lanes = {lid: 6.0}
    h = np.array([sim.headway(lid) for _ in range(5000)])
    assert h.min() >= 2.0 and abs(h.mean() - 8.0) < 0.3, (h.min(), h.mean())


def _arrivals(job):
    """Frame gaps between oncoming vehicles from the opposite approach's straight-and-right lane
    crossing their stop line, in one clip."""
    seed, row = job
    A, B, tmpl, D, C, E, V = row
    res = simulate(4000 + seed, V3, target=dict(A=A, B=B, C=C, D=D, E=E, V=V, template=tmpl))
    if not res["ok"]:
        return None
    m, rec = res["map"], res["branches"][0]
    leg = next(m.lanes[int(l)].leg for l in rec["extra"][:, 0, 0] if l >= 0 and m.lanes[int(l)].kind == "in")
    opp = [l.id for l in m.lanes if l.kind == "in" and l.leg == (leg + 2) % 4 and l.lane_index == 1]
    ev = rec["events"]
    frames = sorted(int(e[0]) for e in ev if e[1] == EV["ENTER"] and e[0] >= 1
                    and int(rec["extra"][int(e[0]) - 1, int(e[2]), 0]) in opp)
    return np.diff(frames).tolist()


def test_oncoming_arrivals():
    rows = [r for r in COMBOS if r[2] == "signal4" and (r[0] == "permissive left" or r[6] == "left_gap")]
    with Pool(32) as p:
        out = [o for o in p.map(_arrivals, [(s, r) for s in range(10) for r in rows]) if o is not None]
    gaps = [g for o in out for g in o]
    short = float(np.mean(np.array(gaps) < 15))   # below the 3 s gap floor
    print(f"clips {len(out)}; oncoming line crossings {len(gaps) + len(out)}; shortest gaps (frames) {sorted(gaps)[:8]}; "
          f"share under 3 s {short:.2f}")
    assert len(gaps) >= 20 and min(gaps) >= 9 and short <= 0.35   # 1.8 s at 5 Hz; most gaps are usable


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
