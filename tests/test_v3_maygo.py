"""Issue 7 (D3): the may_go label, per frame, in the sample's own record and in every future.
python3 tests/test_v3_maygo.py"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from f_toy.datasets.drivetoy.choices import cautious_gap
from f_toy.datasets.drivetoy.config import DRIVETOY_DEFAULT_CFG as CFG
from f_toy.datasets.drivetoy.scenarios import COMBOS, simulate
from f_toy.datasets.drivetoy.sim import BALL_R, EV, MAY_GO_REASONS, RS

V3 = dict(CFG, DT_FORKS=4, DT_KEEP_FORK_RECS=True)
LIGHT = MAY_GO_REASONS.index("light")


def test_cautious_gap():
    assert abs(cautious_gap(CFG) - 5.269) < 1e-3


def _target(row):
    A, B, tmpl, D, C, E, V = row
    return dict(A=A, B=B, C=C, D=D, E=E, V=V, template=tmpl)


def _job(job):
    seed, row = job
    res = simulate(7000 + seed, V3, target=_target(row))
    if not res["ok"]:
        return None
    m, bad, n_go = res["map"], [], 0
    for k, rec in enumerate(res["fork_recs"][0]):
        mg, why = rec["may_go"], rec["may_go_reason"]
        assert mg.shape == why.shape == (rec["agents"].shape[0],)
        enter = min([int(e[0]) for e in rec["events"] if e[1] == EV["ENTER"] and e[2] == 0] + [10 ** 6])
        lane0 = next((m.lanes[int(l)] for l in rec["extra"][:, 0, 0] if l >= 0 and m.lanes[int(l)].kind == "in"), None)
        turn = next((m.lanes[int(l)].turn for l in rec["routes"][0] if l >= 0 and m.lanes[int(l)].kind == "conn"), None)
        for f in range(rec["agents"].shape[0]):
            n_go += int(mg[f] == 1)
            if lane0 is not None and lane0.stop and mg[f] == -1:
                bad.append((k, f, "undefined although the route has a stop line", f > enter))
            if lane0 is not None and lane0.stop and lane0.stop["control"] == "signal":
                ball, arrow = rec["signals"][f, m.groups.index(lane0.stop["group"]), :2]
                if int(ball) == BALL_R and not (arrow > 0 and turn == "L") and (mg[f], why[f]) != (0, LIGHT):
                    bad.append((k, f, "red but", int(mg[f]), MAY_GO_REASONS[int(why[f])]))
            if mg[f] == 1 and int(rec["extra"][f, 0, 3]) == RS["YIELD_PED"]:
                bad.append((k, f, "may go while yielding to a pedestrian"))
    return bad, n_go


def test_maygo_consistent():
    rows = [r for r in COMBOS if r[2] in ("signal4", "stop4")]
    jobs = [(s, rows[(7 * s + i) % len(rows)]) for s in range(8) for i in range(6)]
    with Pool(32) as p:
        out = [o for o in p.map(_job, jobs) if o is not None]
    bad = [b for o in out for b in o[0]]
    print(f"clips {len(out)}; frames with may_go = 1: {sum(o[1] for o in out)}; problems {len(bad)}")
    assert len(out) >= 30 and sum(o[1] for o in out) > 0
    assert not bad, bad[:8]


def test_maygo_deterministic():
    row = next(r for r in COMBOS if r[2] == "stop4")
    a, b = (simulate(7100, V3, target=_target(row)) for _ in range(2))
    assert a["ok"] and b["ok"]
    for ra, rb in zip(a["fork_recs"][0], b["fork_recs"][0]):
        for key in ("agents", "decisions", "may_go", "may_go_reason"):
            assert np.array_equal(ra[key], rb[key]), key
    for fa, fb in zip(a["forks"][0], b["forks"][0]):
        assert fa["may_go"].shape == (32,)
        assert np.array_equal(fa["may_go"], fb["may_go"]) and np.array_equal(fa["ego"], fb["ego"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
