"""H29, H38, H47: the ego signals for a lane-end merge only once a lane-ends sign covers 25 px of its
view; it then merges ahead of or behind the car alongside, futures of the two outcomes differ, and
every merge is done before the lane ends.
python3 tests/test_lane_end.py"""
import numpy as np

from helpers import CFG, ego_rows, events, main, rows, run
from drivetoy.rules import check

LE = rows(B="lane end")
F_PX = (CFG["IMG"] / 2) / np.tan(np.deg2rad(CFG["CAM_FOV"] / 2))


def _sign_px(r, m, f):
    """Largest pixel area of a lane-ends sign ahead of the ego at frame f (a diamond 0.8 size m across)."""
    p, h = r["ego"][f, :2], r["ego"][f, 2]
    fwd = np.array([np.cos(h), np.sin(h)])
    best = 0.0
    for s in m.signs:
        if s["kind"] == "lane_ends":
            d = float((np.asarray(s["pos"][:2]) - p) @ fwd)
            if d > 0.5:
                best = max(best, 0.5 * (0.8 * s.get("size", 1.0) * F_PX / d) ** 2)
    return best


def _end_dist(r, m, f):
    lid = int(r["extra"][f, 0, 0])
    ln = m.lanes[lid] if lid >= 0 else None
    return np.inf if ln is None or not ln.closed_end else ln.length - float(r["extra"][f, 0, 1])


def test_sign_legible_before_indicator():
    for s in run(LE, range(16)):
        m = s["res"]["map"]
        for k, r in enumerate(s["res"]["fork_recs"][0]):
            for e in events(r, "TURN_ON", 0):
                f = int(e[0])   # the event is logged at the end of the frame in which it fires
                if np.isfinite(_end_dist(r, m, f)):   # signalling out of the lane that ends
                    assert _sign_px(r, m, f) >= 25 or _end_dist(r, m, f) < 50, (s["row"], s["seed"], k, f, _sign_px(r, m, f))


def test_ahead_behind_differ():
    n = ok = 0
    for s in run(rows(V="merge"), range(30), forks=8):
        pos = {}
        for r in s["res"]["fork_recs"][0]:
            g = ego_rows(r, "merge")
            g = g[g[:, 0] > 31]
            if len(g):
                pos.setdefault(int(g[0, 5]), []).append(r["ego"][63, :2])
        if len(pos) == 2:
            n += 1
            ok += min(np.hypot(*(a - b)) for a in pos[0] for b in pos[1]) >= 2.0
    print(f"  merge futures of both outcomes apart by 2 m: {ok}/{n}", flush=True)
    assert n and ok / n >= 0.9, (ok, n)


def test_merges_complete():
    for s in run(LE, range(16)):
        m = s["res"]["map"]
        for k, r in enumerate(s["res"]["fork_recs"][0]):
            assert not np.isfinite(_end_dist(r, m, 63)), (s["row"], s["seed"], k, "still in the closing lane")
            assert not check(r, m, CFG), (s["row"], s["seed"], k, check(r, m, CFG)[:3])


if __name__ == "__main__":
    main(globals())
