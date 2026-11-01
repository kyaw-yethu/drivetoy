"""H23, H25: only vehicles actually coming across the ego's path hold it, and a declined gap ends
when the vehicle bounding it passes or stops.
python3 tests/test_oncoming.py"""
import numpy as np

from helpers import CFG, events, main, rows, run
from drivetoy.choices import CHOICE_KINDS
from drivetoy.roads import build_map
from drivetoy.rules import check
from drivetoy.scenarios import _dur, _in_lanes, _insert_ego, _pack, _s_for_line
from drivetoy.sim import EV, RS, Sim

LEFT_GAP = CHOICE_KINDS.index("left_gap")


def scene_left_turn(seed=3, oncoming_v=0.0, oncoming_d=1.0, exit_blocked=False, park_d=None, unblock_at=None,
                    frames=40):
    """A signal4 junction on a green ball for the ego's group, nobody else on the map: the ego (style 0,
    the most cautious) stands 1 m before its line in the inner lane, turning left, and one car is
    on the opposite through lane oncoming_d metres before its line. exit_blocked parks a car at the
    start of that car's straight exit, so it stands at its line; park_d parks one on its approach
    park_d metres before its line, so it stops behind it. unblock_at removes the parked car at that
    frame. Returns the sim after ``frames`` frames."""
    rng = np.random.RandomState(seed)
    cfg = dict(CFG)
    m = build_map("signal4", rng, cfg, protected=False)
    sim = Sim(m, cfg, rng)
    sim.next_slot = 1
    ego_ln = _in_lanes(m, 0)[0]
    opp = _in_lanes(m, 2)[1]
    own = ego_ln.stop["group"]
    sim.signals.schedule(own, "G", -6.0, sim.t, _dur(rng, cfg, {(own, "G"): 60.0}))
    while sim.frame < -1:
        sim.step_frame()
    plan = dict(lane=ego_ln, turn="L", s=_s_for_line(ego_ln, 1.0), v=0.0, cap=True, clear_ahead=0.0, post=[],
                D="left turn", extra=dict(style=0.0))
    _insert_ego(sim, plan, rng)
    car = sim.add_vehicle(opp.id, _s_for_line(opp, oncoming_d), v=oncoming_v, kind="car")
    car.turn_w = {"L": 1e-9, "S": 1.0, "R": 1e-9}
    parked = None
    if exit_blocked:
        c = next(m.lanes[x] for x in opp.next if m.lanes[x].turn == "S")
        parked = sim.add_parked(c.next[0], 4.0, 0.0, kind="car")
    elif park_d is not None:
        parked = sim.add_parked(opp.id, _s_for_line(opp, park_d), 0.0, kind="car")
    for f in range(frames):
        if unblock_at is not None and f == unblock_at and parked in sim.vehicles:
            sim.vehicles.remove(parked)
        sim.step_frame()
    return sim


def entered(sim, slot=0):
    return any(e[1] == EV["ENTER"] and e[2] == slot for e in sim.events)


def test_standing_oncoming_car_is_not_traffic():
    for seed in range(4):
        sim = scene_left_turn(seed=seed, exit_blocked=True, frames=15)
        assert entered(sim), seed


def test_queue_starts_during_turn():
    for seed in range(4):
        sim = scene_left_turn(seed=seed, exit_blocked=True, unblock_at=5, frames=60)   # the queue starts as the ego turns
        assert entered(sim), seed
        assert not check(_pack(sim), sim.m, sim.cfg), (seed, check(_pack(sim), sim.m, sim.cfg)[:3])


def test_memo_cleared():   # an oncoming car that brakes and stops behind a parked car stops holding the ego
    for seed in range(4):
        sim = scene_left_turn(seed=seed, oncoming_v=11.0, oncoming_d=45.0, park_d=12.0, frames=40)
        assert entered(sim), seed


def _stands_for_standing(rec):
    """10 frames in a row with the ego standing, yielding to oncoming traffic, and the vehicle it
    yields to standing too."""
    run_ = 0
    for f in range(rec["agents"].shape[0]):
        y = int(rec["extra"][f, 0, 4])
        if rec["ego"][f, 3] < 0.3 and int(rec["extra"][f, 0, 3]) == RS["YIELD_ONCOMING"] and y >= 0 \
                and rec["agents"][f, y, 4] < 0.5:
            run_ += 1
            if run_ >= 10:
                return True
        else:
            run_ = 0
    return False


def test_no_wait_for_standing_traffic():
    for s in run(rows(V="left_gap") + rows(A="permissive left"), range(12)):
        for r in s["res"]["fork_recs"][0]:
            assert not _stands_for_standing(r), (s["row"], s["seed"])


if __name__ == "__main__":
    main(globals())
