"""Issue 6 (D2, D4): where the ego's lane allows several moves and it has not begun to turn by
frame 31, each future draws its own route with equal odds; nothing before frame 32 shows it.
python3 tests/test_v3_routes.py"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from f_toy.datasets.drivetoy.config import DRIVETOY_DEFAULT_CFG as CFG, V3_OVERRIDES
from f_toy.datasets.drivetoy.rules import check
from f_toy.datasets.drivetoy.scenarios import COMBOS, route_draws, simulate

V3 = dict(CFG, **V3_OVERRIDES, DT_FORKS=16, DT_KEEP_FORK_RECS=True)
T0 = CFG["DT_CUE_FRAME"]


def test_route_draws():
    rng = np.random.RandomState(0)
    for moves in (["S", "R"], ["L", "S", "R"]):
        d = np.array([route_draws(moves, 16, rng) for _ in range(400)]).ravel()
        for mv in moves:
            assert abs(np.mean(d == mv) - 1 / len(moves)) < 0.05, (moves, mv, np.mean(d == mv))


def _turn(rec, m):
    """The move of the ego's final route (the turn of its connector)."""
    return next((m.lanes[int(l)].turn for l in rec["routes"][0] if l >= 0 and m.lanes[int(l)].kind == "conn"), None)


def _job(job):
    seed, row = job
    A, B, tmpl, D, C, E, V = row
    res = simulate(5000 + seed, V3, target=dict(A=A, B=B, C=C, D=D, E=E, V=V, template=tmpl))
    if not res["ok"]:
        return None
    m, recs, draws = res["map"], res["fork_recs"][0], res["route_draws"][0]
    if draws is None:
        return "not eligible"
    turns = {_turn(r, m) for r in recs}
    assert [_turn(r, m) for r in recs] == list(draws)
    lamps = int(recs[0]["lamps"][:T0 + 1, 0, 1:3].sum())
    decel = min(float(np.min(np.diff(r["ego"][:, 3]))) * CFG["DT_FPS"] for r in recs)
    viol = [v for r in recs for v in check(r, m, V3)]
    return turns, lamps, decel, viol


def test_routes_per_future():
    rows = [r for r in COMBOS if r[2] in ("stop4", "signal4") and r[3] == "straight"]
    with Pool(32) as p:
        out = p.map(_job, [(s, r) for s in range(6) for r in rows])
    elig = [o for o in out if isinstance(o, tuple)]
    print(f"built {sum(o is not None for o in out)}/{len(out)}, routes drawn per future: {len(elig)}; "
          f"futures with 2+ moves: {sum(len(o[0]) >= 2 for o in elig)}; indicator frames before {T0 + 1}: "
          f"{sum(o[1] for o in elig)}; hardest braking {min(o[2] for o in elig):.1f} m/s2; violations {sum(len(o[3]) for o in elig)}")
    assert len(elig) >= 10
    assert all(len(o[0]) >= 2 for o in elig)
    assert all(o[1] == 0 for o in elig)
    assert all(o[2] >= -CFG["DT_B_MAX"] - 0.5 for o in elig)
    assert not [v for o in elig for v in o[3]], [v for o in elig for v in o[3]][:12]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
