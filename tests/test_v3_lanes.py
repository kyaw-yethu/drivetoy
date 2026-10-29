"""Issue 4: the signal4 approach line is solid all the way, the left-only lane carries about 25% of
its approach, and the arrow phase is unchanged. python3 tests/test_v3_lanes.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from f_toy.datasets.drivetoy.config import DRIVETOY_DEFAULT_CFG as CFG, V3_OVERRIDES
from f_toy.datasets.drivetoy.roads import LANE_W, _rot, build_signal4
from f_toy.datasets.drivetoy.sim import Sim, Signals

V3 = dict(CFG, **V3_OVERRIDES)


def _approach_lines(m, style):
    """Per leg, the markings of ``style`` on the line between its two approach lanes (local x = LANE_W)."""
    out = {k: [] for k in range(4)}
    for mk in m.markings:
        if mk["style"] != style:
            continue
        for k in range(4):
            loc = (_rot(k).T @ np.asarray(mk["pts"], float).T).T
            if np.allclose(loc[:, 0], LANE_W, atol=1e-6) and (loc[:, 1] <= -m.intersection["stop"] + 1e-6).all():
                out[k].append(loc[:, 1])
    return out


def test_solid_approach():
    m = build_signal4(np.random.RandomState(0), V3)
    assert all(not v for v in _approach_lines(m, "dashed_white").values())
    for k, lines in _approach_lines(m, "solid_white").items():
        assert len(lines) == 1, (k, lines)
        assert np.isclose(lines[0].min(), -200.0) and np.isclose(lines[0].max(), -m.intersection["stop"]), (k, lines)


def test_left_share():
    placed, rate = [0, 0], []
    for seed in range(200):
        rng = np.random.RandomState(seed)
        m = build_signal4(rng, V3)
        sim = Sim(m, V3, rng)
        sim.populate(0.5, 0.05)   # at 1.0 the 64 agent slots run out before the last lanes are filled
        inner = {l.id for l in m.lanes if l.kind == "in" and l.lane_index == 0}
        outer = {l.id for l in m.lanes if l.kind == "in" and l.lane_index == 1}
        for u in sim.vehicles:
            placed[0] += u.route[0] in inner
            placed[1] += u.route[0] in inner | outer
        rate.append(sum(sim.inflow[i] for i in inner) / sum(sim.inflow[i] for i in inner | outer))
    share, mean_rate = placed[0] / placed[1], float(np.mean(rate))
    print(f"inner lane: share of placed vehicles {share:.3f}, share of inflow {mean_rate:.3f}")
    assert 0.20 <= share <= 0.30 and 0.20 <= mean_rate <= 0.30


def test_arrow_unchanged():
    assert V3["SIG_ARROW_P"] == 0.5 and V3["SIG_ARROW"] == (3.0, 5.0)
    m = build_signal4(np.random.RandomState(1), V3, protected=True)
    seq = Signals(m, V3, np.random.RandomState(1), 0.0).seq
    for g in m.groups:
        assert seq.index((g, "A")) < seq.index((g, "G"))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
