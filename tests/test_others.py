"""H31, H42: what other road users do that the ego reacts to after frame 31 has started by frame 31,
their choices that concern the ego are made by then, and pedestrians or mergers held for the ego
never leave anyone waiting.
python3 tests/test_others.py"""
import numpy as np

from helpers import COMBOS, events, main, rows, run
from drivetoy.sim import P_CROSS, RS

T0 = 31


def _late_ped_yield(r):
    ex = r["extra"]
    for f in range(T0 + 1, ex.shape[0]):
        p = int(ex[f, 0, 4])
        if int(ex[f, 0, 3]) == RS["YIELD_PED"] and (p < 0 or int(ex[T0, p, 5]) // 10 != P_CROSS):
            return f
    return None


def _late_lane_change_ahead(r):
    ag, ex, e = r["agents"], r["extra"], r["ego"]
    for ev in events(r, "LC_START"):
        f, a = int(ev[0]), int(ev[2])
        if a == 0 or f <= T0 or f >= ag.shape[0] - 1:
            continue
        g = min(f + 1, ag.shape[0] - 1)
        fwd = np.array([np.cos(e[f, 2]), np.sin(e[f, 2])])
        d = float((ag[f, a, 1:3] - e[f, :2]) @ fwd)
        reach = max(10.0, 2.0 * e[f, 3]) + (ag[f, a, 6] + ag[f, 0, 6]) / 2   # where a cut-in binds
        if int(ex[g, a, 0]) == int(ex[g, 0, 0]) and -(ag[f, a, 6] + ag[f, 0, 6]) / 2 < d < reach:
            return f, a
    return None


def _mutual_wait(r):
    """15 frames in a row with the ego standing for a pedestrian who stands too."""
    ex, ag = r["extra"], r["agents"]
    run_ = 0
    for f in range(ex.shape[0]):
        p = int(ex[f, 0, 4])
        if r["ego"][f, 3] < 0.3 and int(ex[f, 0, 3]) == RS["YIELD_PED"] and p >= 0 and ag[f, p, 4] < 0.1:
            run_ += 1
            if run_ >= 15:
                return True
        else:
            run_ = 0
    return False


def _stands_at_lane_end(r, m):
    """A vehicle other than the ego standing for more than 5 s in a lane that ends after frame 31 while it
    yields to the ego (held for it): beside an ego crawling in a queue it waits a few seconds, a zipper,
    but not longer. Waits for other cars in a jam are traffic, not the ego's doing."""
    ex, ag = r["extra"], r["agents"]
    for a in range(1, ag.shape[1]):
        run_ = 0
        for f in range(T0 + 1, ag.shape[0]):
            lid = int(ex[f, a, 0])
            if ag[f, a, 9] > 0 and lid >= 0 and m.lanes[lid].closed_end and ag[f, a, 4] < 0.3 and int(ex[f, a, 4]) == 0:
                run_ += 1
                if run_ > 25:
                    return a
            else:
                run_ = 0
    return None


PEDS = rows(template="residential") + rows(B="pedestrian")


def test_peds_cross_on_screen():
    for s in run(PEDS, range(6)):
        for k, r in enumerate(s["res"]["fork_recs"][0]):
            assert _late_ped_yield(r) is None, (s["row"], s["seed"], k, _late_ped_yield(r))


def test_held_pedestrians_release():
    for s in run(PEDS, range(6)):
        for k, r in enumerate(s["res"]["fork_recs"][0]):
            assert not _mutual_wait(r), (s["row"], s["seed"], k)


def test_no_late_cut_ins():
    for s in run(rows(template="multilane"), range(6)):
        for k, r in enumerate(s["res"]["fork_recs"][0]):
            assert _late_lane_change_ahead(r) is None, (s["row"], s["seed"], k, _late_lane_change_ahead(r))


def test_held_mergers_merge_behind():
    for s in run(rows(B="lane end"), range(10)):
        for k, r in enumerate(s["res"]["fork_recs"][0]):
            assert _stands_at_lane_end(r, s["res"]["map"]) is None, (s["row"], s["seed"], k)


def test_only_ego_pending():
    for s in run(COMBOS, range(1)):
        for k, r in enumerate(s["res"]["fork_recs"][0]):
            d = r["decisions"]
            bad = d[(d[:, 0] > T0) & (d[:, 1] != 0) & (d[:, 6] > 0) & (d[:, 4] >= 0.1) & (d[:, 4] <= 0.9)]
            assert not len(bad), (s["row"], s["seed"], k, bad[:2].tolist())


if __name__ == "__main__":
    main(globals())
