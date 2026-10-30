"""Fixes after the v3 pilot and review: acceleration-aware oncoming times, E chosen without regard
to future 0's drawn route.
python3 tests/test_v3_fixes.py"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from f_toy.datasets.drivetoy import sim
from f_toy.datasets.drivetoy.config import DRIVETOY_DEFAULT_CFG as CFG, V3_OVERRIDES
from f_toy.datasets.drivetoy.scenarios import COMBOS, simulate

V3 = dict(CFG, **V3_OVERRIDES)


def test_time_to_cover():
    assert abs(sim._time_to_cover(8.0, 0.0, 1.0, 10.0) - 4.0) < 1e-9     # 8 = t^2 / 2
    assert abs(sim._time_to_cover(60.0, 0.0, 1.0, 10.0) - 11.0) < 1e-9   # 10 s to 10 m/s (50 m), then 1 s
    assert abs(sim._time_to_cover(30.0, 15.0, 1.0, 10.0) - 2.0) < 1e-9   # already faster than vmax


def _route_job(seed):
    row = [r for r in COMBOS if r[6] == "rtor" and r[5] == "pending"][seed % 2]
    A, B, tmpl, D, C, E, V = row
    res = simulate(9000 + seed, dict(V3, DT_FORKS=4), target=dict(A=A, B=B, C=C, D=D, E=E, V=V, template=tmpl))
    if not res["ok"] or res["route_draws"][0] is None:
        return None
    return list(res["route_draws"][0])


def test_future0_route_unbiased():
    """Selection on E must not favour the move drawn for future 0 (review item 2)."""
    with Pool(32) as p:
        out = [o for o in p.map(_route_job, range(64)) if o is not None]
    f0 = np.mean([o[0] == "R" for o in out])
    rest = np.mean([x == "R" for o in out for x in o[1:]])
    print(f"samples {len(out)}; right turns: future 0 {f0:.2f}, futures 1-3 {rest:.2f}")
    assert len(out) >= 20 and f0 <= rest + 0.15


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
