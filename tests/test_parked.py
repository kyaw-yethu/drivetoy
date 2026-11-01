"""H15, H27: going around a parked car is never drawn, and the ego never stands behind one while no
moving oncoming vehicle is near.
python3 tests/test_parked.py"""
import numpy as np

from helpers import ego_rows, main, rows, run
from drivetoy.sim import RS

PARK = rows(B="parked obstacle") + rows(D="nudge around")


PEND = rows(V="nudge", E="pending")


def test_pending_parked_car():   # the ego closes up behind it around frame 31; before or after the oncoming car is drawn
    assert PEND, "no pending parked-car row"
    hit = []
    for s in run(PEND, range(16)):
        g = ego_rows(s["res"]["branches"][0], "nudge")
        hit.append(bool(len(g)) and 32 <= g[0, 0] <= 40 and 0.1 <= g[0, 4] <= 0.9)
    print(f"  pending parked-car choices in frames 32 to 40: {np.mean(hit):.2f} of {len(hit)}", flush=True)
    assert np.mean(hit) >= 0.5


def test_go_round_now_is_carried_out():
    for s in run(PEND, range(12), forks=8):
        for k, r in enumerate(s["res"]["fork_recs"][0]):
            g = ego_rows(r, "nudge")
            g = g[g[:, 0] > 31]
            if len(g) and g[0, 5] == 1 and g[0, 0] + 20 < r["agents"].shape[0]:
                f = int(g[0, 0])
                e = r["ego"]
                h = e[max(f - 2, 0), 2]
                lat = np.abs((e[f:f + 21, :2] - e[f, :2]) @ np.array([-np.sin(h), np.cos(h)])).max()
                assert lat >= 1.0, (s["seed"], k, f, lat)


CLEAR = rows(V="nudge_clear")


def test_clear_road_goes_round_in_every_future():   # H62: nobody coming, so no choice and no closing up
    assert CLEAR, "no parked-car row with a clear road"
    res = run(CLEAR, range(16), forks=8)
    assert len(res) >= 12, len(res)
    for s in res:
        for k, r in enumerate(s["res"]["fork_recs"][0]):
            assert not len(ego_rows(r, "nudge")), (s["row"], s["seed"], k)
            e = r["ego"]
            h = e[20, 2]
            lat = (e[:, :2] - e[20, :2]) @ np.array([-np.sin(h), np.cos(h)])
            assert np.abs(lat[31:]).max() >= 1.0 and abs(lat[-1]) < 0.5, (s["row"], s["seed"], k, lat[31:].round(2))
            assert e[25:, 3].min() >= 4.0, (s["row"], s["seed"], k, e[25:, 3].min())


def _stuck_behind_parked(r):
    """10 frames in a row with the ego standing behind a parked car it follows while no oncoming
    vehicle that moves is within 80 m."""
    ag, ex, e = r["agents"], r["extra"], r["ego"]
    run_ = 0
    for f in range(ag.shape[0]):
        y = int(ex[f, 0, 4])
        behind = e[f, 3] < 0.3 and int(ex[f, 0, 3]) == RS["FOLLOW"] and y > 0 and int(ex[f, y, 0]) == -1 \
            and 1 <= ag[f, y, 0] <= 4
        if behind:
            near = [u for u in range(1, ag.shape[1]) if ag[f, u, 9] > 0 and 1 <= ag[f, u, 0] <= 4 and ag[f, u, 4] >= 0.5
                    and np.cos(ag[f, u, 3] - e[f, 2]) < -0.5 and np.hypot(*(ag[f, u, 1:3] - e[f, :2])) < 80.0]
            run_ = 0 if near else run_ + 1
            if run_ >= 10:
                return True
        else:
            run_ = 0
    return False


def test_never_stuck_behind_parked_car():
    for s in run(PARK, range(16)):
        for k, r in enumerate(s["res"]["fork_recs"][0]):
            assert not _stuck_behind_parked(r), (s["row"], s["seed"], k)


if __name__ == "__main__":
    main(globals())
