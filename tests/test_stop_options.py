"""H14, H39, H40, S7: at a pending stop-sign tie each future draws one of the ego's options open at
frame 31 (letting the others go, or a move whose path no moving car crosses); a turn holder that is
itself held never freezes the junction; an ego that lets the others go still gets going.
python3 tests/test_stop_options.py"""
import numpy as np

from helpers import CFG, ego_rows, main, rows, run
from drivetoy.rules import check
from drivetoy.sim import RS

TIE = rows(V="near_tie")
T0 = 31
MOVE = {1: "L", 2: "S", 3: "R"}


def _cross(m, a, b):
    return (a.id, b.id) in m.conflicts or (b.id, a.id) in m.conflicts or (a.next and b.next and a.next[0] == b.next[0])


def _moving_paths(r, m):
    """Connectors of the vehicles moving at the junction at frame 31: on a connector, or near their
    stop line on another approach at 0.3 m/s or more (their path from their final route)."""
    ex, ag = r["extra"], r["agents"]
    ego_ln = m.lanes[int(ex[T0, 0, 0])]
    out = []
    for a in range(1, ag.shape[1]):
        lid = int(ex[T0, a, 0])
        if ag[T0, a, 9] <= 0 or lid < 0 or not 1 <= ag[T0, a, 0] <= 4:
            continue
        ln = m.lanes[lid]
        if ln.kind == "conn":
            out.append(ln)
        elif ln.kind == "in" and ln.stop is not None and ln.leg != ego_ln.leg and ag[T0, a, 4] >= 0.3 \
                and ln.stop["s"] - (ex[T0, a, 1] + ag[T0, a, 6] / 2) < 15.0 \
                and any(ag[f, a, 4] < 0.3 and int(ex[f, a, 0]) == lid for f in range(T0)):   # under way after its stop
            route = [x for x in r["routes"][a] if x >= 0]
            nxt = [m.lanes[x] for x in route if m.lanes[x].kind == "conn"]
            if nxt:
                out.append(nxt[0])
    return out


def _open(r, m):
    ego_ln = m.lanes[int(r["extra"][T0, 0, 0])]
    paths = _moving_paths(r, m)
    return {"yield"} | {m.lanes[c].turn for c in ego_ln.next if not any(_cross(m, m.lanes[c], p) for p in paths)}


def _taken(r):
    g = ego_rows(r, "stop_wave")
    g = g[g[:, 0] > T0]
    if not len(g):
        return None
    return MOVE.get(int(r["ego_turn"])) if g[0, 5] == 1 else "yield"


def test_options_are_open_ones():
    n = 0
    for s in run(TIE, range(20), forks=16):
        m = s["res"]["map"]
        own = s["res"]["fork_recs"][0][0]
        if _taken(own) is None:
            continue
        n += 1
        opts = _open(own, m)
        for k, r in enumerate(s["res"]["fork_recs"][0]):
            assert _taken(r) in opts, (s["seed"], k, _taken(r), opts)
            assert not check(r, m, CFG), (s["seed"], k, check(r, m, CFG)[:2])
    assert n >= 10, n


def test_go_first_is_carried_out():   # the drawn move happens: the tied cars wait for the ego
    went = []
    for s in run(TIE, range(20), forks=8):
        for r in s["res"]["fork_recs"][0]:
            t = _taken(r)
            if t not in (None, "yield"):
                e = r["ego"]
                went.append(float(np.sum(np.hypot(*np.diff(e[T0:57, :2], axis=0).T))) > 5.0)
    print(f"  go-first futures moving 5 m by frame 56: {np.mean(went):.2f} of {len(went)}", flush=True)
    assert went and np.mean(went) >= 0.9


def test_all_four_when_nobody_moves():
    hits = []
    for s in run(TIE, range(20), forks=16):
        own = s["res"]["fork_recs"][0][0]
        g = ego_rows(own, "stop_wave")
        g = g[g[:, 0] > T0]
        if len(g) and int(g[0, 3]) == 4:   # four options open at frame 31 (the row's cue)
            hits.append(len({_taken(r) for r in s["res"]["fork_recs"][0]}) == 4)
    print(f"  all four options among 16 futures: {np.mean(hits):.2f} of {len(hits)}", flush=True)
    assert hits and np.mean(hits) >= 0.8


def _path_at(r, m, a, f):
    """The connector vehicle a takes, as others can judge it at frame f: its own if it is on one,
    else the one its indicator shows (none: straight), as the simulator reads an open route."""
    lid = int(r["extra"][f, a, 0])
    if lid < 0:
        return None
    ln = m.lanes[lid]
    if ln.kind == "conn":
        return ln
    want = {1: "L", 2: "R"}.get(int(r["extra"][f, a, 2]), "S")
    return next((m.lanes[c] for c in ln.next if m.lanes[c].turn == want), None)


def _frozen_turn(r, m):
    """A vehicle standing at its stop sign for 3 s behind a turn holder that stands, held by a
    pedestrian or a conflict, although their paths do not cross (its own full stop of 0.8 s, the
    holder held for 1 s and a start delay come first)."""
    ex, ag = r["extra"], r["agents"]
    lets_go = _taken(r) == "yield"   # an ego that lets the others go waits for them by its choice
    for a in range(1 if lets_go else 0, ag.shape[1]):
        run_ = 0
        for f in range(ag.shape[0]):
            h = int(ex[f, a, 4])
            if ag[f, a, 9] > 0 and int(ex[f, a, 3]) == RS["STOP_SIGN"] and ag[f, a, 4] < 0.3 and h >= 0 \
                    and ag[f, h, 4] < 0.3 and int(ex[f, h, 3]) in (RS["YIELD_PED"], RS["YIELD_CONFLICT"]):
                ra, rh = _path_at(r, m, a, f), _path_at(r, m, h, f)
                run_ = 0 if (ra is None or rh is None or _cross(m, ra, rh)) else run_ + 1
                if run_ >= 15:
                    return a, f
            else:
                run_ = 0
    return None


def test_no_stop_sign_freeze():
    for s in run(rows(template="stop4"), range(8)):
        for k, r in enumerate(s["res"]["fork_recs"][0]):
            assert _frozen_turn(r, s["res"]["map"]) is None, (s["row"], s["seed"], k, _frozen_turn(r, s["res"]["map"]))


def test_yield_then_go():   # S7
    started = []
    for s in run(TIE, range(20), forks=8):
        for r in s["res"]["fork_recs"][0]:
            if _taken(r) == "yield":
                e = r["ego"]
                started.append(float(np.sum(np.hypot(*np.diff(e[T0:, :2], axis=0).T))) > 3.0)
    print(f"  egos letting the others go that start by frame 63: {np.mean(started):.2f} of {len(started)}", flush=True)
    assert started and np.mean(started) >= 0.25   # soft (S7): letting one or two cars go first often takes the whole future


if __name__ == "__main__":
    main(globals())
