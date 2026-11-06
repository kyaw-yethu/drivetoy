"""Decision scenes of DriveToy v4 (docs/drivetoy-v4-design.md): counterfactual twin pairs, single clips and,
for train, one clip of either outcome with its decision on any frame.

The decision frame FD (31 for val and test) is a module setting, ``set_decision_frame(fd)``: every scene
times its cue and places the ego relative to it, frames are still forked and labelled at T0 = 31.

A pair is one scene simulated twice. Both twins share the simulator state and its random stream up to
the split frame; then they differ in a single thing (a light's phase or a pedestrian's choice at the curb).
``simulate_pair(seed, cfg, name)`` returns both twins as two
branches (0: the change twin, 1: the carry-on twin).

A single clip shows a cue ahead that calls for a change right after frame 31: a car signalling into the
ego's lane, a red light, a stop sign, the school zone, a parked car or the end of the ego's lane; or, at an
all-way stop, cars under way that leave some of the ego's moves open. ``simulate_single(seed, cfg, name)``
returns it with one branch.

The ego holds its speed (or stands) until the cue is clearly visible, and frame 31 is the last frame before
it changes what it does. Each scene places the ego so that its cue reaches a set size on screen by frame 31
(LAMP_PX and the like), slow enough to stop from there at 2.5 to 3.5 m/s^2, and a dry run then shifts the
scene's timing until the change starts on frame 32. Both functions return ok=False with the reasons
attempts failed when no attempt passes the checks.

Two earlier states of this code built part of DriveToy v4, and ``set_stage`` rebuilds their clips exactly
(plans/<split>.json names each clip's stage). "twins" built val_twins and test_twins: it aimed in at most five
undamped steps, checked a pair's futures without the scene's own settings, and let a red_green ego that had
arrived at the line rest as it came. "train1" built train's first pass. Neither cleared the oncoming lane
beside a parked car, nor had an ego with a set braking stop steadily behind a parked car ahead. "final" built most
of the rest. "clean", for new clips and the ones rebuilt after them, keeps what happens after frame 31
predictable from frames 0 to 31: an ego that stood goes straight or keeps standing (right turns on red are
right_red's), a carry-on ego at a red or waiting to turn left stays put to the clip's end, and a clip whose
speed dips on the way away, or drops while it yields, is rejected, since the ego then yields to something out
of its way. A left turn across traffic is taken only with the gap clear at the decision: no car moving toward
the ego within GAP_M but the last one passing. Outside RESTARTS, where the cause of a late start shows by frame
31, an ego that has held steady does not change again from frame 34 on.
"""
import copy

import numpy as np

from .checks import didnt_go
from .diversity import clip_labels, cue_problems, ego_approach
from .roads import build_map
from .rules import check as rules_check
from .scenarios import (EGO_HALF, STILL_M, _add_ahead, _clear, _conflict_lane, _dur, _fork_seed, _future, _in_lanes, _insert_ego,
                        _pack, _s_for_line, _timed_cars, _u, moved, route_draws)
from .sim import BALL_G, BALL_R, BALL_Y, RS, VEH_KINDS, Sim

STAYS, STARTS, STOPS, KEEPS = range(4)
OUTCOMES = ("stays", "starts", "stops", "keeps moving")
T0, H = 31, 20               # T0: last visible frame, where futures fork; H: frames of the 4 s outcome
FD = 31                      # the decision frame: the ego's change starts on FD + 1 (set_decision_frame)


def set_decision_frame(fd):
    """Put every scene's decision on frame fd (0 to 31). A scene then runs as it does for frame 31, started
    31 - fd frames earlier: a longer warm-up, the ego placed that much earlier, and its clock times moved by _t;
    the clip shows the decision at frame fd and keeps frames 32 to 63 as its one ground truth."""
    global FD
    FD = int(fd)


STAGES = ("clean", "final", "train1", "twins")
STAGE = "clean"


def set_stage(stage):
    """Build as one of STAGES (see the module docstring)."""
    global STAGE
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    STAGE = stage


def _t(x):
    """A scene time written for a decision on frame 31, moved to the current decision frame."""
    return x + (FD - 31) / 5.0


def _shift():
    """Frames by which a scene starts earlier than its frame-31 version."""
    return 31 - FD


def _insert_frame(plan):
    """The frame the ego is placed at: the plan's own (counted from the warm-up's start) or -1, moved earlier."""
    return plan["insert"] if "insert" in plan else -1 - _shift()


def _go_now():
    return FD + 2                # all-way stop: a future goes at once if the ego moves by FD + 2
V_MOVING, V_STOP, START_M = 0.5, 0.3, 2.0
V_TOL, H_TOL = 0.02, 0.004   # a change of speed (m/s) or heading (rad) per frame: 0.1 m/s^2, 0.02 rad/s
AIM_TRIES = 8
YIELDS = (RS["YIELD_ONCOMING"], RS["YIELD_PED"], RS["YIELD_CONFLICT"])
GAP_M = 60.0                 # about 5.5 s of oncoming traffic: a left turn wants this much road clear
LEFT_GAPS = dict(left_go=1, left_free=0)   # oncoming cars allowed within GAP_M at the decision
SAME_EGO = 1e-4              # the twins' ego poses and speeds over frames 0 to 31
MIN_AGREE = 12               # of 16 futures carrying out (or not) the clip's change
SLOW_BY, SIDE_M = 1.5, 1.0
MAX_BRAKE = 4.0              # m/s^2: the hardest braking a clip's ground truth may show within 4 s of the decision
HALF_FOV, VIEW_M = np.deg2rad(35.0), 70.0
# A cue is clearly visible at frame 31 at these sizes (pixels of the 128 x 128 view): lit lamps of the two
# far-side heads, a pedestrian, a stop sign, a car seen from behind, a lane-ends sign. The scenes place the
# ego from each cue's size; the build measures the sizes on the render.
LAMP_PX, PED_PX, SIGN_PX, CAR_PX, LANE_SIGN_PX = 12, 20, 20, 30, 60
F_PX = 64 / np.tan(np.deg2rad(35.0))   # focal length of the 70 degree, 128 px camera
CAM_BACK = 1.75                        # the camera sits this far behind the ego's front (m)
LAMP_C = 17000.0                       # lit lamps of the heads facing the ego cover LAMP_C / d^2 px at d m (measured)
CAR_FILL = dict(car=0.6, suv=0.7, truck=0.85)   # share of its width x height a car's body covers from behind (measured)
HEAD_PAST_LINE = 27.5                  # signal4: stop line to the far-side heads (m), with DT_HEADS_CORNER
STOP_SIGN_A = 0.56                     # stop sign seen face on, with its pole below it (m^2)
PED_ADULT_A = 0.5 * 1.75 * 0.55        # an adult's silhouette (m^2)
STOP_CASES = ("none", "west_right", "north_straight", "east_left", "cross_both")
# the ego's moves open at frame 32 in each case (the user's rule, 2026-11-05): no car, or the car on its left
# turning right: all three; the oncoming car going straight: straight and right; the car on its right turning
# left: right; the cars on both sides going straight: none
OPEN_MOVES = dict(none="LSR", west_right="LSR", north_straight="SR", east_left="R", cross_both="")
# the cars under way in each case: (side as the ego sees it, move, time it moves off, s), each in view at frame 31
STOP_CARS = dict(none=[], west_right=[("W", "R", 2.5)], north_straight=[("N", "S", 3.6)],
                 east_left=[("E", "L", 3.6)], cross_both=[("W", "S", 2.8), ("E", "S", 3.0)])


class Skip(Exception):
    """The scene could not be set up in this attempt."""


# ---------------------------------------------------------------------------
# outcomes and checks
# ---------------------------------------------------------------------------

def _ahead(rec, t0=None, h=H):
    """The ego's positions over the h frames after t0 (default FD) in its frame at t0, its speed per step, and
    its speed at t0."""
    t0 = FD if t0 is None else t0
    ego = np.asarray(rec["ego"], np.float64)
    x0, y0, h0, v0 = ego[t0]
    d = ego[t0 + 1:t0 + 1 + h, :2] - (x0, y0)
    c, s = np.cos(h0), np.sin(h0)
    xy = np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], -1)
    speed = np.linalg.norm(np.diff(np.concatenate([np.zeros((1, 2)), xy]), axis=0), axis=1) * 5.0
    return xy, speed, float(v0)


def outcome(rec, t0=None):
    """The ego's 4 s outcome after t0 (default FD): stays / starts (moves more than 2 m) when stopped then
    (below 0.5 m/s), else stops (falls below 0.3 m/s) / keeps moving, as in ego_status_test.py."""
    xy, speed, v0 = _ahead(rec, t0)
    if v0 >= V_MOVING:
        return STOPS if speed.min() < V_STOP else KEEPS
    return STARTS if speed.sum() / 5.0 > START_M else STAYS


def changes(rec, kind):
    """Whether the ego carries out a change within 4 s: "starts" (the 4 s outcome), "slows" (its speed falls
    by at least 1.5 m/s), "slows_or_moves" (that, or it moves at least 1 m sideways off its frame-31 line) or
    "goes_now" (it moves off by frame 33)."""
    if kind == "starts":
        return outcome(rec) == STARTS
    if kind == "starts_straight":   # a right turn on red is the carry-on twin's alternative, not its change
        h = np.asarray(rec["ego"], np.float64)[:, 2]
        return outcome(rec) == STARTS and abs(float(np.angle(np.exp(1j * (h[-1] - h[FD]))))) < 0.5
    if kind == "goes_now":
        f = onset(rec, last=len(rec["ego"]) - 1)
        return f is not None and f <= _go_now()
    xy, speed, v0 = _ahead(rec)
    slows = speed.min() <= v0 - SLOW_BY
    return bool(slows) if kind == "slows" else bool(slows or np.abs(xy[:, 1]).max() >= SIDE_M)


def _window():
    """First frame of the window that must show the ego holding its motion: FD - 7 (0 at the earliest)."""
    return max(FD - 7, 0)


def onset(rec, last=None):
    """The first frame after the window start (FD - 7) up to ``last`` whose ego speed or heading differs from
    the frame before by more than 0.1 m/s^2 or 0.02 rad/s (a standing ego: any movement), or None."""
    last = FD + H if last is None else last
    w = _window()
    e = np.asarray(rec["ego"], np.float64)
    v = e[w:last + 1, 3]
    dv = np.abs(np.diff(v))
    tol = np.where(v[:-1] < 0.05, 1e-3, V_TOL)
    dh = np.abs(np.angle(np.exp(1j * np.diff(e[w:last + 1, 2]))))
    f = np.flatnonzero((dv > tol) | (dh > H_TOL))
    return int(f[0]) + w + 1 if len(f) else None


def in_view(rec, slot, f=None):
    """Whether agent ``slot`` is within 70 m and the camera's 70 degree field of view at frame f (default FD)."""
    f = FD if f is None else f
    a = rec["agents"][f, slot]
    if a[9] <= 0:
        return False
    ex, ey, eh = rec["ego"][f, :3]
    d = a[1:3] - (ex, ey)
    ang = np.angle(np.exp(1j * (np.arctan2(d[1], d[0]) - eh)))
    return bool(np.hypot(*d) <= VIEW_M and abs(ang) <= HALF_FOV)


def _light(rec, m, f=None):
    """The ball of the ego's own signal at frame f (default FD), or None away from a signal."""
    f = FD if f is None else f
    gi = ego_approach(rec, m)
    return None if gi is None else int(rec["signals"][f, gi, 0])


def _problems(pk, m, cfg, cued_slot=None, allow_r0=False, wait_until=None):
    """The drive_v3 checks of a future; a cued car's own move into the ego's lane is allowed, the ego's
    speeding is allowed where the scene has it slow down only on entering a lower limit (allow_r0), and
    standing up to frame ``wait_until`` (default FD + 2: a start-up delay or hold ending at the decision, and
    the ego's first frames under way) is not a "did not go"."""
    wait_until = FD + 2 if wait_until is None else wait_until
    if pk["ran_out"]:
        return "ran out"
    cp = [p for p in cue_problems(pk, m, cfg) if not (cued_slot is not None and p[1] == "lane change"
                                                       and p[2].startswith(f"vehicle {cued_slot} "))]
    if cp:
        return "cue problem: " + cp[0][1]
    if not cfg.get("DT_VIOLATION"):
        broken = [r for r in rules_check(pk, m, cfg) if not (allow_r0 and r[1] == "R0" and r[2] == 0)]
        if broken:
            return "rule broken: " + str(broken[0][1])
    if [f for f in didnt_go(pk, m, cfg) if f > wait_until]:
        return "did not go"
    return None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _speed_for(d, b, margin=1.0, react=0.2):
    """The speed from which a steady stop at b m/s^2, begun ``react`` s later, ends ``margin`` m short of a
    point d m ahead of the front."""
    return float(-react * b + np.sqrt((react * b) ** 2 + 2 * b * max(d - margin, 0.5)))


def _lamp_line(rng):
    """Front-to-line distance at which the lit lamps of the far-side heads cover LAMP_PX px or a little more."""
    return float(np.sqrt(LAMP_C / LAMP_PX) - CAM_BACK - HEAD_PAST_LINE - _u(rng, 0.0, 1.0))


def _delay_signal(extra):
    """A change made before frame 0: the controller's current phase lasts ``extra`` s longer, which
    postpones every change after it. No random draw is used."""
    def apply(s, r):
        s.signals.start += extra
    return apply


def _steady(u):
    """No slow wander of a driver's desired speed along the route."""
    d = list(u.drift)
    d[3] = 0.0
    u.drift = d


def _cruise(s, e, r):
    """The ego cruises at its desired speed, without the slow wander of that speed along the route."""
    _steady(e)
    e.v = s._v0(e)


def _rest_at_line(s, e, r):
    """A standing ego rests where it stands: its stopping distance before the line is its distance now."""
    e.stop_dist = max(float(e.lanes[0].stop["s"] - e.front), 0.3)


def _signal_setup(sim, rng, lane=1):
    m, cfg = sim.m, sim.cfg
    k = sim.ego_leg
    sim.gap_lanes[_in_lanes(m, (k + 2) % 4)[1].id] = _u(rng, *cfg["DT_ONCOMING_MEAN"])
    ln = _in_lanes(m, k)[lane]                   # 1: the straight-and-right lane, 0: the left-turn lane
    own = ln.stop["group"]
    other = [g for g in m.groups if g != own][0]
    return m, cfg, ln, own, other


def _residential(sim, rng):
    m = sim.m
    j = int(rng.randint(2))
    return m, m.lanes[m.spawn_lanes[j]], m.lanes[m.spawn_lanes[1 - j]]


def _northbound(m):
    return sorted((m.lanes[i] for i in m.spawn_lanes if m.lanes[i].pts[-1, 1] > m.lanes[i].pts[0, 1]),
                  key=lambda l: l.lane_index)


# ---------------------------------------------------------------------------
# twin scenes: scene(sim, rng, dt, variant) -> (ego plan, twin), where dt (s) moves the cue's timing earlier
# and twin = dict(split frame, apply=[change twin, carry-on twin], cue = slot to see at frame 31 in twin
# cue_twin, light = the ego's ball at frame 31 in the change and carry-on twins)
# ---------------------------------------------------------------------------

def _red_green(sim, rng, dt, variant=None):
    """First in line at a red since frame 20 or earlier; its green comes just before frame 31."""
    m, cfg, ln, own, other = _signal_setup(sim, rng)
    t0 = sim.t
    t_on = _t(_u(rng, 5.6, 6.2)) - dt
    plan = dict(lane=ln, turn="S", cap=True, v=0.0, clear_ahead=300.0, post=[], D="straight",
                extra=dict(start_delay=_u(rng, *cfg["DT_START_DELAY"])))
    if rng.rand() < 0.5:
        d_line = _u(rng, 0.8, 2.0)                           # stopped at its line all clip
        plan["post"].append(_rest_at_line)
    else:
        v_a, t_stop = _u(rng, 6.0, 9.0), _u(rng, 2.4, 3.6)   # arrives and stops by about frame 20
        d_line, plan["v"] = v_a * t_stop / 2.0, v_a
        if STAGE != "twins":
            plan["extra"]["settle"] = True                   # and rests there
    plan["s"] = _s_for_line(ln, d_line)
    sim.populate(_u(rng, 0.3, 1.0), ped_rate=_u(rng, 0.02, 0.12))
    sim.signals.schedule(own, "G", t_on, t0, _dur(rng, cfg, {(own, "G"): 10.0 + _u(rng, 0, 6)}))
    return plan, dict(split=-1, apply=[None, _delay_signal(15.0)], light=(BALL_G, BALL_R))


def _queue(sim, rng, dt, variant=None):
    """Stopped behind a car at a red; the green comes so that the car ahead has just moved off by frame 31."""
    m, cfg, ln, own, other = _signal_setup(sim, rng)
    t0 = sim.t
    t_on = _t(_u(rng, 5.0, 5.8)) - dt
    plan = dict(lane=ln, turn="S", cap=True, v=0.0, clear_ahead=300.0, post=[], D="straight",
                extra=dict(T=_u(rng, 1.0, 1.3), s0=_u(rng, 1.5, 2.0),   # follows within 10 m, as in drive_v3
                           start_delay=_u(rng, *cfg["DT_START_DELAY"])))
    plan["s"] = _s_for_line(ln, _u(rng, 0.8, 2.0) + _u(rng, 6.5, 9.0))
    lead_delay = _u(rng, *cfg["DT_START_DELAY"])

    def lead(s, e, r):
        u = _add_ahead(s, r, ln, e.s, max(ln.stop["s"] - e.front - 1.5 - 4.5, 1.5), v=0.0, kind="car", route="S")
        if u is None:
            raise Skip("no room for the car ahead")
        u.start_delay = lead_delay
        e.s0 = max(float(u.s - u.L / 2 - e.front), 1.0)   # it rests at the gap it stands at: no creeping up
    plan["post"].append(lead)
    sim.populate(_u(rng, 0.3, 1.0), ped_rate=_u(rng, 0.02, 0.12))
    sim.signals.schedule(own, "G", t_on, t0, _dur(rng, cfg, {(own, "G"): 10.0 + _u(rng, 0, 6)}))
    return plan, dict(split=-1, apply=[None, _delay_signal(15.0)], light=(BALL_G, BALL_R))


def _green_yellow(sim, rng, dt, variant=None):
    """Cruising slowly toward a green on a quiet junction. The yellow comes just before frame 31, when the
    lamps first show clearly (front 7.4 to 8.4 m from its line), and the ego, whose stop then needs under
    3 m/s^2 (drive_v3's yellow rule stops it), brakes steadily from frame 32."""
    m, cfg, ln, own, other = _signal_setup(sim, rng)
    t0 = sim.t
    d31 = _lamp_line(rng)
    v_e = _speed_for(d31, _u(rng, 2.4, 2.8))
    t_y = _t(_u(rng, 5.6, 5.9)) - dt
    plan = dict(lane=ln, turn="S", cap=False, v=v_e, clear_ahead=300.0, post=[_cruise], D="straight",
                extra=dict(v0frac=v_e / ln.limit, brake_at=0.1, stop_dist=2.0, s0=1.0), draw_route=False)
    plan["s"] = _s_for_line(ln, d31 + v_e * (T0 + 1) / cfg["DT_FPS"])
    sim.populate(_u(rng, 0.0, 0.15), ped_rate=_u(rng, 0.0, 0.05))
    sim.signals.schedule(own, "Y", t_y, t0, _dur(rng, cfg, {}))
    return plan, dict(split=-1, apply=[None, _delay_signal(15.0)], light=(BALL_Y, BALL_G))


def _left_gap(sim, rng, dt, variant=None):
    """Waiting at its line in the left-turn lane under protected-permissive heads, while oncoming traffic
    flows at about 2 s headways. In the change twin the left arrow comes on just before frame 31, with the
    oncoming cars stopped at their red; in the carry-on twin the green ball came at frame 0 and the
    oncoming cars are already crossing the junction."""
    m, cfg, ln, own, other = _signal_setup(sim, rng, lane=0)
    k = sim.ego_leg
    opp = _in_lanes(m, (k + 2) % 4)
    sim.gap_lanes[opp[1].id] = 0.05                   # oncoming through traffic every 2 to 2.2 s
    sim.gap_lanes[opp[0].id] = 60.0                   # and next to nobody in their left-turn lane
    t0 = sim.t
    sd = _u(rng, 0.6, 0.75)                           # the ego's start-up delay: an oncoming car moving off
    t_a = _t(_u(rng, 5.4, 5.6)) - dt                  # with the green at frame 0 is under way before it could go
    sim.signals.schedule(own, "A", t_a, t0, _dur(rng, cfg, {(own, "A"): 7.0}))
    plan = dict(lane=ln, turn="L", cap=True, v=0.0, clear_ahead=300.0, post=[_rest_at_line], D="left turn",
                extra=dict(start_delay=sd), insert=1 - cfg["DT_WARMUP"])
    plan["s"] = _s_for_line(ln, _u(rng, 0.8, 2.0))

    def setup(s, e, r):
        _clear(s, opp[0].id, 0.0, opp[0].length)
        s.peds = [p for p in s.peds if p.cw != (k + 3) % 4 or p.state == 4]   # nobody on the crosswalk the turn passes
    plan["post"].append(setup)
    sim.populate(_u(rng, 0.3, 0.6), ped_rate=_u(rng, 0.0, 0.05))

    def permissive(s, r):
        t_g = s.t + cfg["SIG_YELLOW"] + cfg["SIG_ALL_RED"] + 0.1
        s.signals.schedule(own, "G", t_g, s.t, _dur(r, cfg, {(own, "A"): 0.0, (own, "G"): 40.0}))
        for u in s.vehicles:
            if u.route[u.where()[0]] == opp[1].id:
                u.start_delay = 0.0

    def cue(s):   # the nearest oncoming car under way past its line
        best = None
        for u in s.vehicles:
            ui, _ = u.where()
            ul = u.lanes[ui]
            if u.entered and u.v > 1.0 and (ul.leg == (k + 2) % 4 if ul.kind == "conn" else ul.kind == "out"
                                            and ul.leg == k):
                d = float(np.hypot(*(u.pose()[0] - s.ego.pose()[0])))
                if best is None or d < best[0]:
                    best = (d, u.slot)
        return None if best is None else best[1]
    return plan, dict(split=2 - cfg["DT_WARMUP"], apply=[None, permissive], light=(BALL_R, BALL_G), cue=cue, cue_twin=1)


def _ped_step(sim, rng, dt, variant=None):
    """Cruising slowly toward a crosswalk as an adult reaches the curb; in the change twin they step out
    before frame 31, in the carry-on twin they walk on. At frame 31 they are about 13 m ahead of the camera
    (about 20 px), and the ego, noticing them 0.6 to 1 s after they step out, stops steadily 1 m short of
    the crosswalk at 2.5 to 3.5 m/s^2."""
    m, ln, opp = _residential(sim, rng)
    cfg = sim.cfg
    cw0, cw1, cw_id = ln.crosswalks[0]
    d_cam = np.sqrt(PED_ADULT_A * F_PX ** 2 / PED_PX) * _u(rng, 0.9, 0.97)   # camera to the crosswalk's middle
    d31 = d_cam - CAM_BACK - (cw1 - cw0) / 2                                 # front to its near edge
    b = _u(rng, 2.5, 3.5)
    v_e = _speed_for(d31, b, margin=1.0)
    t_c = _t(_u(rng, 5.1, 5.4)) - dt
    limit = ln.limit_at(cw0) if variant == 1 else ln.limit   # variant 1: inside the school zone all clip
    plan = dict(lane=ln, cap=False, v=v_e, clear_ahead=300.0, post=[], turn=None, D="straight",
                extra=dict(v0frac=v_e / limit, brake_at=b))
    plan["s"] = cw0 - EGO_HALF - d31 - v_e * (T0 + 1) / cfg["DT_FPS"]
    if variant == 1 and min(z[0] for z in ln.zones) > plan["s"] - EGO_HALF:
        raise Skip("starts outside the school zone")
    if plan["s"] < 5.0:
        raise Skip("street too short")
    paths = [p for p in m.ped_paths if p["cw"] == cw_id]
    path = paths[int(rng.randint(len(paths)))]
    react = _u(rng, 0.6, 1.0)

    def ped(s, e, r):
        s.peds = [p for p in s.peds if p.cw != cw_id]
        p = s.add_ped(path, kind="adult")
        if p is None:
            raise Skip("no pedestrian")
        p.s = max(0.0, p.cum[-1] - p.speed * (t_c - s.t))
        p.heading = p.path_heading()
        p.react = react
        s.twin_ped = p
    plan["post"] += [ped, _cruise]
    sim.populate(_u(rng, 0.2, 0.5), ped_rate=_u(rng, 0.0, 0.03))

    def force(cross):
        def apply(s, r):
            s.twin_ped.force_cross = cross
        return apply
    return plan, dict(split=-1, apply=[force(True), force(False)], cue=lambda s: s.twin_ped.slot, cue_twin=0)


def _overtake(sim, rng, dt, variant=None):
    """Following a slow truck or bus in the middle lane at its speed, held up long enough to want to pass,
    with a car in the left lane level with the gap (its rear 3 to 6 m ahead of the ego's front). In the change
    twin that car speeds up from about frame 15, and once its rear is 10 to 14 m clear of the ego's front the
    ego pulls out behind it, from frame 32; in the carry-on twin it stays alongside and the ego keeps
    following."""
    m, cfg = sim.m, sim.cfg
    nb = _northbound(m)
    ln, tgt = nb[1], nb[0]
    solid = nb[0].solid_right[0]
    for _ in range(50):
        s0 = _u(rng, 30.0, 240.0)
        if s0 + 230.0 < solid[0] or (s0 - 10.0 > solid[1] and s0 + 230.0 < nb[0].length):
            break
    else:
        raise Skip("no dashed stretch")
    v_l = ln.limit * _u(rng, *cfg["DT_OVERTAKE_V0_FRAC"])
    v0frac = _u(rng, *cfg["DT_OVERTAKE_EGO_FRAC"])
    T, s_min = 1.0, 1.5
    gap = (s_min + v_l * T) / np.sqrt(1 - (v_l / (ln.limit * v0frac)) ** 4)   # where it follows at a steady speed
    kind = ("truck", "bus")[int(rng.randint(2))]
    t_go = _t(_u(rng, 3.0, 3.4)) - dt
    plan = dict(lane=ln, cap=False, v=v_l, clear_ahead=150.0, post=[], turn=None, D="lane change", s=s0,
                extra=dict(v0frac=v0frac, overtaker=True, patience=0.5, T=T, s0=s_min, hold_gap_s=1.6,
                           lc_quick=True, overtake_wait=True, overtake_gap=_u(rng, 10.0, 14.0)))

    def setup(s, e, r):
        _clear(s, tgt.id, e.s - 150.0, e.s + 150.0)
        _clear(s, nb[2].id, e.s - 100.0, e.s + 150.0)
        lead = _add_ahead(s, r, ln, e.s, gap, v=v_l, v0frac=v_l / ln.limit, kind=kind)
        if lead is None:
            raise Skip("no room for the slow lead")
        nbr = _add_ahead(s, r, tgt, e.s, _u(r, 3.0, 6.0), v=v_l, v0frac=v_l / tgt.limit, kind="car")
        if nbr is None:
            raise Skip("no room for the car alongside")
        for u in (lead, nbr, e):
            _steady(u)
        e.v = v_l
        e.behind_since = s.t - 5.0                 # held up behind it well before frame 0
        s.twin_nbr = nbr
    plan["post"].append(setup)
    sim.populate(_u(rng, 0.1, 0.3), ped_rate=0.0)

    def moves_on(s, r):
        s.twin_nbr.v0_at = (t_go, 0.9)             # it speeds up to 0.9 of the limit
    return plan, dict(split=-1, apply=[moves_on, None], cue=lambda s: s.twin_nbr.slot, cue_twin=1)


# ---------------------------------------------------------------------------
# single scenes: scene(sim, rng, dt, variant) -> (ego plan, single), where dt (s) moves the change earlier
# and single = dict(change kind, cue = slots to see at frame 31 or None, cued = slot of a car allowed to move
# into the ego's lane after frame 31, light = the ego's ball at frame 31)
# ---------------------------------------------------------------------------

def _red_ahead(sim, rng, dt, variant=None):
    """Cruising slowly toward a red that stays red; it brakes steadily at 2.5 to 3.5 m/s^2 from frame 32,
    when the lamps first show clearly (front 7.4 to 8.4 m from its line)."""
    m, cfg, ln, own, other = _signal_setup(sim, rng)
    t0 = sim.t
    sim.signals.schedule(other, "G", _t(_u(rng, -9.0, -6.0)), t0, _dur(rng, cfg, {(other, "G"): 40.0}))
    d31, b = _lamp_line(rng), _u(rng, 2.5, 3.5)
    v_e = _speed_for(d31, b)
    plan = dict(lane=ln, turn="S", cap=False, v=v_e, clear_ahead=300.0, post=[_cruise], D="straight",
                extra=dict(v0frac=v_e / ln.limit, brake_at=b, stop_dist=2.0, s0=1.0), draw_route=False)
    plan["s"] = _s_for_line(ln, d31 + v_e * ((T0 + 1) / cfg["DT_FPS"] - dt))
    sim.populate(_u(rng, 0.0, 0.15), ped_rate=_u(rng, 0.0, 0.05))
    return plan, dict(kind="slows", light=BALL_R)


def _stop_ahead(sim, rng, dt, variant=None):
    """Cruising slowly toward its stop sign; it brakes steadily at 2.5 to 3.5 m/s^2 from frame 32, when the
    sign first shows clearly (about 15 m from the camera, 20 px)."""
    m, cfg = sim.m, sim.cfg
    ln = _in_lanes(m, sim.ego_leg)[0]
    d31 = np.sqrt(STOP_SIGN_A * F_PX ** 2 / SIGN_PX) * _u(rng, 0.93, 1.0) - CAM_BACK   # the sign stands at the line
    b = _u(rng, 2.5, 3.5)
    v_e = _speed_for(d31, b)
    plan = dict(lane=ln, turn="S", cap=False, v=v_e, clear_ahead=300.0, post=[_cruise], D="straight",
                extra=dict(v0frac=v_e / ln.limit, brake_at=b, stop_dist=2.0, s0=1.0, creep=False), draw_route=False)
    plan["s"] = _s_for_line(ln, d31 + v_e * ((T0 + 1) / cfg["DT_FPS"] - dt))
    sim.populate(_u(rng, 0.0, 0.15), ped_rate=_u(rng, 0.0, 0.05))
    return plan, dict(kind="slows")


def _zone_ahead(sim, rng, dt, variant=None):
    """Cruising toward the 30 km/h school zone, holding its speed until its front is 0.5 to 2 m from the
    red pavement, then slowing at 2.5 to 3 m/s^2 from frame 32."""
    m, ln, opp = _residential(sim, rng)
    if not ln.zones:
        raise Skip("no school zone")
    cfg = sim.cfg
    v_e = 50 / 3.6 * 0.92
    z0 = min(z[0] for z in ln.zones)
    zl = _u(rng, 0.5, 2.0)
    plan = dict(lane=ln, cap=False, v=v_e, clear_ahead=300.0, post=[_cruise], turn=None, D="straight",
                extra=dict(v0frac=0.92, zone_late=zl, zone_brake=_u(rng, 2.5, 3.0)))
    plan["s"] = z0 - EGO_HALF - zl - v_e * ((T0 + 1) / cfg["DT_FPS"] - dt)
    sim.populate(_u(rng, 0.1, 0.4), ped_rate=_u(rng, 0.0, 0.03))
    return plan, dict(kind="slows", allow_r0=True)


def _parked_ahead(sim, rng, dt, variant=None):
    """Cruising at about 30 km/h down the residential street toward a car double-parked in its lane, with the
    oncoming lane clear; it holds its speed until the parked car first shows clearly (about 30 px at frame 31,
    18 to 32 m ahead), then steers round it from frame 32 and back over as long a stretch, so the swerve stays
    under about 2 m/s^2 sideways."""
    m, ln, opp = _residential(sim, rng)
    cfg = sim.cfg
    cw0, cw1, _ = ln.crosswalks[0]
    v_e = 30 / 3.6 * _u(rng, 0.95, 1.0)
    kind = ["car", "suv", "truck"][int(rng.choice(3, p=[0.5, 0.2, 0.3]))]
    L, W, Hh = VEH_KINDS[kind][1:4]
    gap31 = float(np.clip(np.sqrt(CAR_FILL[kind] * W * Hh * F_PX ** 2 / CAR_PX) * _u(rng, 0.93, 1.0) - CAM_BACK,
                          18.0, 32.0))   # front to its rear
    for _ in range(50):
        s_p = _u(rng, 130.0, 230.0)
        if min(abs(s_p - cw0), abs(s_p - cw1)) > 25.0:
            break
    plan = dict(lane=ln, cap=False, v=v_e, clear_ahead=300.0, post=[], turn=None, D="straight",
                extra=dict(v0frac=v_e / ln.limit, parked_late=gap31 - 0.5 * v_e / cfg["DT_FPS"]))
    plan["s"] = s_p - L / 2 - EGO_HALF - gap31 - v_e * ((T0 + 1) / cfg["DT_FPS"] - dt)
    if plan["s"] < 5.0:
        raise Skip("street too short")

    t_on = None
    if STAGE in ("final", "clean"):
        sim.gap_lanes[opp.id] = 60.0                   # next to no oncoming traffic but the one placed below
        if variant == 1:   # an oncoming car reaches the parked car 2 to 3 s after the decision: the ego stops behind it
            plan["extra"].update(brake_at=3.5, s0=_u(rng, 4.0, 6.0))
        t_on = _t(_u(rng, 8.4, 9.2)) - dt

    def parked(s, e, r):
        _clear(s, ln.id, s_p - 15.0, s_p + 15.0)
        s_o = opp.length - s_p                         # the oncoming lane runs the other way
        _clear(s, opp.id, s_o - 160.0, s_o + 40.0)
        if variant == 1:
            n0 = {u.slot for u in s.vehicles}
            _timed_cars(s, r, opp, s_o, [(t_on, opp.limit * _u(r, 0.7, 0.85))])
            if len({u.slot for u in s.vehicles} - n0) != 1:
                raise Skip("no room for the oncoming car")
        u = s.add_parked(ln.id, s_p, -_u(r, 0.5, 0.95), kind=kind)
        if u is None:
            raise Skip("no room for the parked car")
        s.cue_slot = u.slot
    plan["post"] += [parked, _cruise]
    sim.populate(_u(rng, 0.2, 0.5), ped_rate=_u(rng, 0.0, 0.03))
    return plan, dict(kind="slows_or_moves", cue=lambda s: [s.cue_slot])


def _cut_in(sim, rng, dt, variant=None):
    """Cruising at 50 km/h behind a slightly slower car in the next lane that signals from about frame 20
    and moves into the ego's lane just after frame 31, its rear 13 to 17 m ahead of the ego's front; the
    ego slows from frame 32."""
    m, cfg = sim.m, sim.cfg
    nb = _northbound(m)
    j = int(rng.randint(2))
    ln, src = nb[j], nb[1 - j]
    if variant == 2:
        ln, src = nb[1], nb[2]
    v_e = 50 / 3.6
    plan = dict(lane=ln, cap=False, v=v_e, clear_ahead=200.0, post=[], turn=None, D="straight",
                extra=dict(v0frac=v_e / ln.limit, T=1.0, s0=1.5, a=1.5, b=2.5))
    solid = nb[0].solid_right[0]
    for _ in range(50):
        s = _u(rng, 30.0, 220.0)
        if s + 200.0 < solid[0] or (s - 10.0 > solid[1] and s + 200.0 < nb[0].length):
            break
    plan["s"] = s
    gap31, f_c = _u(rng, 13.0, 17.0), _u(rng, 0.9, 0.95)
    if variant == 1:   # far ahead and no slower: nothing to slow for
        gap31, f_c = _u(rng, 28.0, 40.0), _u(rng, 1.0, 1.08)
    if variant == 2:   # the merger is 35 to 50 m from its lane's end at the decision
        plan["s"] = src.length - _u(rng, 35.0, 50.0) - gap31 - 4.5 - EGO_HALF - v_e * (T0 + 1) / cfg["DT_FPS"]
        if plan["s"] < 5.0:
            raise Skip("road too short")
    sig_at, start = _t(_u(rng, 3.8, 5.0)) - dt, _t(_u(rng, 6.0, 6.3)) - dt

    def cutin(s, e, r):
        _clear(s, src.id, e.s - 15.0, e.s + 90.0)
        v_c = e.v * f_c
        u = _add_ahead(s, r, src, e.s, gap31 + (e.v - v_c) * (T0 + 1) / cfg["DT_FPS"], v=v_c, kind="car",
                       v0frac=v_c / ln.limit)
        if u is None:
            raise Skip("no room for the cut-in")
        _steady(u)
        if variant == 2:
            u.end_near = 10.0   # it brakes for its lane's end only when it has not merged by 10 m before it
        side = "right" if src.lane_index < ln.lane_index else "left"
        u.lc = dict(side=side, target=ln.id, mandatory=False, moving=False, forced=True, cued=True,
                    lead=start - sig_at, dur=_u(r, 2.5, 3.5), t_sig=sig_at, sig_at=sig_at, follower_decel=3.5)
        s.cue_slot = u.slot
    plan["post"] += [cutin, _cruise]
    sim.populate(_u(rng, 0.1, 0.3), ped_rate=0.0)
    return plan, dict(kind="slows", cue=lambda s: [s.cue_slot], cued=lambda s: s.cue_slot, carry=variant == 1)


def _lane_end(sim, rng, dt, variant=None):
    """Cruising in the closing lane toward its 3.2 m lane-ends sign, with the cone taper starting 8 m past it
    and the lane to its left clear. It signals 1.2 to 1.4 s before it moves over (R7 checks 1 s), which it does
    from frame 32 with the sign 13 to 15 m ahead of the camera, well over LANE_SIGN_PX."""
    m, cfg = sim.m, sim.cfg
    nb = _northbound(m)
    ln = nb[2]
    v_e = _u(rng, 50.0, 56.0) / 3.6
    sign = next(s for s in m.signs if s["kind"] == "lane_ends")
    lead = _u(rng, 1.2, 1.4)
    d_sig = _u(rng, 13.0, 15.0) + 0.5 + v_e * lead   # ego centre to the sign when it signals (the move starts a frame after the lead)
    px_sig = 0.5 * (0.8 * sign.get("size", 1.0) * F_PX / d_sig) ** 2      # what the sign covers then
    plan = dict(lane=ln, cap=False, v=v_e, clear_ahead=300.0, post=[], turn=None, D="merge",
                extra=dict(v0frac=v_e / ln.limit, sign_px=0.999 * px_sig, end_near=10.0, merge_lead=(lead, lead),
                           merge_dur=_u(rng, 2.5, 3.0)))
    t_sig = (T0 + 1) / cfg["DT_FPS"] - lead - dt
    plan["s"] = (sign["pos"][1] - ln.pts[0, 1]) - d_sig - v_e * (t_sig + 0.2)

    def clear(s, e, r):
        _clear(s, nb[1].id, e.s - 150.0, e.s + 150.0)
        _clear(s, ln.id, e.s + 3.0, ln.length)
    plan["post"] += [clear, _cruise]
    sim.populate(_u(rng, 0.1, 0.3), ped_rate=0.0)
    return plan, dict(kind="slows_or_moves", cue=None)


def _approach_car(s, leg, d_line, v, b, route, t_go):
    """A car on leg's approach whose front is d_line m before its stop line at speed v; it stops steadily at
    b m/s^2 at its line and, its turn come, moves off with ``route`` at time t_go."""
    cl = _in_lanes(s.m, leg)[0]
    _clear(s, cl.id, cl.stop["s"] - d_line - 40.0, cl.stop["s"])
    u = s.add_vehicle(cl.id, _s_for_line(cl, d_line), v=v, kind="car")
    if u is None:
        raise Skip("no room at an approach")
    _steady(u)
    u.v0frac, u.a = 0.85, 2.0
    u.brake_at, u.stop_dist, u.s0, u.creep, u.start_delay, u.hold_until = b, 2.0, 1.0, False, 0.2, t_go
    u.turn_w = {x: float(x == route) + 1e-9 for x in "LSR"}
    return u


def _stop_sign(sim, rng, dt, variant=None):
    """Arriving at an all-way stop as one or two cars that stopped before it are under way: none, the car on
    its left turning right, the oncoming car going straight, the car on its right turning left, or the cars
    on both sides going straight. The ego comes to rest by frame 22 and is held to frame 31; its move is
    drawn per future, and a future goes at once where its move crosses no car under way and waits where it
    does."""
    m, cfg = sim.m, sim.cfg
    k = sim.ego_leg
    ln = _in_lanes(m, k)[0]
    case = STOP_CASES[variant if variant is not None else int(rng.randint(len(STOP_CASES)))]
    fixed = int(cfg.get("DT_FORKS", 1)) == 1   # one ground truth (train): the ego's move is drawn now, per clip
    move = "LSR"[int(rng.randint(3))] if fixed else None
    west, north, east = (k + 3) % 4, (k + 2) % 4, (k + 1) % 4
    cars = [((west, north, east)["WNE".index(side)], mv, t_go) for side, mv, t_go in STOP_CARS[case]]
    v_e, b_e = _u(rng, 4.5, 5.5), _u(rng, 2.0, 2.5)
    t_stop = _t(_u(rng, 3.6, 4.2))                  # the ego is at rest by FD - 10
    t_ins = sim.t + (-1 - _shift() - sim.frame) / cfg["DT_FPS"]
    plan = dict(lane=ln, turn=move or "S", cap=False, v=v_e, clear_ahead=300.0, D="straight", draw_route=not fixed,
                extra=dict(v0frac=v_e / ln.limit, brake_at=b_e, stop_dist=2.0, s0=1.0, creep=False,
                           start_delay=0.2, hold_until=(FD + 0.5) / cfg["DT_FPS"]))
    cruise = t_stop - t_ins - v_e / b_e
    if cruise < 0.2:
        raise Skip("ego too close")
    plan["s"] = _s_for_line(ln, v_e * cruise + v_e ** 2 / (2 * b_e))
    draws = [(leg, mv, _t(t_go + _u(rng, -0.2, 0.2)), 0.0) for leg, mv, t_go in cars]

    def others(s, e, r):
        for leg in (west, north, east):
            cl = _in_lanes(s.m, leg)[0]
            _clear(s, cl.id, cl.stop["s"] - 60.0, cl.stop["s"])
        s.cue_slots = []
        for leg, mv, t_go, _ in draws:
            t_rest = t_go - 1.3                         # at rest 1.3 s before it moves off (full stop: 0.8 s)
            b = 3.0
            v = min(6.0, 0.8 * b * (t_rest - t_ins))
            d = v * (t_rest - t_ins - v / b) + v ** 2 / (2 * b) + 1.0
            s.cue_slots.append(_approach_car(s, leg, d, v, b, mv, t_go).slot)
        # where drive_v3's small junction has two paths pass within 2.8 m but the other car is through
        # before the ego gets there, the ego may go: its left beside a right turn from its left, its right
        # beside a left turn from its right
        e.compatible = {("L", c) for c, (leg, mv, _, _) in zip(s.cue_slots, draws) if (leg, mv) == (west, "R")} | \
                       {("R", c) for c, (leg, mv, _, _) in zip(s.cue_slots, draws) if (leg, mv) == (east, "L")}
    plan["post"] = [others, _cruise]
    sim.populate(_u(rng, 0.0, 0.1), ped_rate=_u(rng, 0.0, 0.03))
    return plan, dict(kind="goes_now", cue=lambda s: list(s.cue_slots), case=case, regrant=True, move=move)


# ---------------------------------------------------------------------------
# Scenes of train and the train-like part of val and test only (docs/drivetoy-v4-design.md): clips without a decision, a
# lead that brakes, a yellow that turns red ahead, a yellow too late to stop for, a permissive left turn
# taken once the last oncoming car has passed
# ---------------------------------------------------------------------------

def _no_peds(s, e, r):
    s.peds = []


def _cruise_junction(sim, rng, dt, variant=None):
    """Cruising straight through a signalised junction on a green that stays green; nobody crosses."""
    m, cfg, ln, own, other = _signal_setup(sim, rng)
    t0 = sim.t
    sim.signals.schedule(own, "G", _u(rng, -8.0, -2.0), t0, _dur(rng, cfg, {(own, "G"): 40.0}))
    v_e = ln.limit * _u(rng, 0.8, 0.95)
    plan = dict(lane=ln, turn="S", cap=False, v=v_e, clear_ahead=300.0, post=[_cruise, _no_peds], D="straight",
                extra=dict(v0frac=v_e / ln.limit), draw_route=False)
    plan["s"] = _s_for_line(ln, _u(rng, 20.0, 90.0))
    sim.populate(_u(rng, 0.2, 0.6), ped_rate=0.0)
    return plan, dict(kind="none")


def _cruise_road(sim, rng, dt, variant=None):
    """Cruising in the left or middle lane of the multilane road with the lane clear ahead."""
    m, cfg = sim.m, sim.cfg
    nb = _northbound(m)
    ln = nb[int(rng.randint(2))]
    v_e = ln.limit * _u(rng, 0.8, 0.95)
    plan = dict(lane=ln, cap=False, v=v_e, clear_ahead=300.0, post=[_cruise], turn=None, D="straight",
                s=_u(rng, 10.0, 60.0), extra=dict(v0frac=v_e / ln.limit))
    sim.populate(_u(rng, 0.1, 0.4), ped_rate=0.0)
    return plan, dict(kind="none")


def _cruise_street(sim, rng, dt, variant=None):
    """Driving down the residential street (no school zone) with nobody about to cross."""
    m, ln, opp = _residential(sim, rng)
    v_e = ln.limit * _u(rng, 0.7, 0.85)
    plan = dict(lane=ln, cap=False, v=v_e, clear_ahead=300.0, post=[_cruise, _no_peds], turn=None, D="straight",
                s=_u(rng, 10.0, 80.0), extra=dict(v0frac=v_e / ln.limit))
    sim.populate(_u(rng, 0.1, 0.4), ped_rate=0.0)
    return plan, dict(kind="none")


def _follow_setup(sim, rng, ln, v_l, kind=None, brake=None):
    """The ego follows a lead at v_l at the gap where both hold their speed; with ``brake`` = (t, fraction,
    decel) the lead slows at time t to that fraction of its speed at no more than decel m/s^2."""
    v0frac = _u(rng, 0.85, 0.95)
    T, s_min = _u(rng, 1.1, 1.5), _u(rng, 1.5, 2.5)
    gap = (s_min + v_l * T) / np.sqrt(max(1 - (v_l / (ln.limit * v0frac)) ** 4, 0.2))
    kind = kind or ("car", "suv", "truck")[int(rng.choice(3, p=[0.5, 0.3, 0.2]))]

    def lead(s, e, r):
        u = _add_ahead(s, r, ln, e.s, gap, v=v_l, v0frac=v_l / ln.limit, kind=kind)
        if u is None:
            raise Skip("no room for the lead")
        _steady(u)
        _steady(e)
        e.v = v_l
        if brake is not None:
            u.v0_at, u.zone_brake = (brake[0], brake[1] * v_l / ln.limit), brake[2]
        s.cue_slot = u.slot
    return dict(v0frac=v0frac, T=T, s0=s_min), lead


def _follow_road(sim, rng, dt, variant=None):
    """Following a slower car at a steady gap in the left or middle lane of the multilane road."""
    m, cfg = sim.m, sim.cfg
    nb = _northbound(m)
    ln = nb[int(rng.randint(2))]
    extra, lead = _follow_setup(sim, rng, ln, ln.limit * _u(rng, 0.55, 0.8))
    plan = dict(lane=ln, cap=False, v=None, clear_ahead=300.0, post=[lead], turn=None, D="straight",
                s=_u(rng, 10.0, 60.0), extra=extra)
    sim.populate(_u(rng, 0.1, 0.3), ped_rate=0.0)
    return plan, dict(kind="none")


def _follow_street(sim, rng, dt, variant=None):
    """Following a slower car at a steady gap down the residential street; nobody about to cross."""
    m, ln, opp = _residential(sim, rng)
    extra, lead = _follow_setup(sim, rng, ln, ln.limit * _u(rng, 0.55, 0.75), kind="car")
    plan = dict(lane=ln, cap=False, v=None, clear_ahead=300.0, post=[lead, _no_peds], turn=None, D="straight",
                s=_u(rng, 10.0, 80.0), extra=extra)
    sim.populate(_u(rng, 0.1, 0.3), ped_rate=0.0)
    return plan, dict(kind="none")


def _lead_brakes_road(sim, rng, dt, variant=None):
    """Following a car at a steady gap on the multilane road; it brakes (its brake lamps on) to 30 to 60% of
    its speed at 2 to 3 m/s^2 just before frame 31, and the ego slows from frame 32."""
    m, cfg = sim.m, sim.cfg
    nb = _northbound(m)
    ln = nb[int(rng.randint(2))]
    brake = (_t(_u(rng, 5.9, 6.1)) - dt, _u(rng, 0.3, 0.6), _u(rng, 2.0, 3.0))
    extra, lead = _follow_setup(sim, rng, ln, ln.limit * _u(rng, 0.55, 0.8), brake=brake)
    plan = dict(lane=ln, cap=False, v=None, clear_ahead=300.0, post=[lead], turn=None, D="straight",
                s=_u(rng, 10.0, 60.0), extra=extra)
    sim.populate(_u(rng, 0.1, 0.3), ped_rate=0.0)
    return plan, dict(kind="slows", cue=lambda s: [s.cue_slot])


def _lead_brakes_street(sim, rng, dt, variant=None):
    """Following a car at a steady gap down the residential street; it brakes to 20 to 50% of its speed at
    2 to 3 m/s^2 just before frame 31, and the ego slows from frame 32."""
    m, ln, opp = _residential(sim, rng)
    brake = (_t(_u(rng, 5.9, 6.1)) - dt, _u(rng, 0.2, 0.5), _u(rng, 2.0, 3.0))
    extra, lead = _follow_setup(sim, rng, ln, ln.limit * _u(rng, 0.55, 0.75), kind="car", brake=brake)
    plan = dict(lane=ln, cap=False, v=None, clear_ahead=300.0, post=[lead, _no_peds], turn=None, D="straight",
                s=_u(rng, 10.0, 80.0), extra=extra)
    sim.populate(_u(rng, 0.1, 0.3), ped_rate=0.0)
    return plan, dict(kind="slows", cue=lambda s: [s.cue_slot])


def _yellow_red(sim, rng, dt, variant=None):
    """Cruising slowly toward a light that is yellow at the clip's start and turns red within it; the ego
    brakes steadily at 2.5 to 3.5 m/s^2 from frame 32, when the lamps first show clearly."""
    m, cfg, ln, own, other = _signal_setup(sim, rng)
    t0 = sim.t
    sim.signals.schedule(own, "Y", _u(rng, -2.6, -0.6), t0, _dur(rng, cfg, {(other, "G"): 40.0}))
    d31, b = _lamp_line(rng), _u(rng, 2.5, 3.5)
    v_e = _speed_for(d31, b)
    plan = dict(lane=ln, turn="S", cap=False, v=v_e, clear_ahead=300.0, post=[_cruise], D="straight",
                extra=dict(v0frac=v_e / ln.limit, brake_at=b, stop_dist=2.0, s0=1.0), draw_route=False)
    plan["s"] = _s_for_line(ln, d31 + v_e * ((T0 + 1) / cfg["DT_FPS"] - dt))
    sim.populate(_u(rng, 0.0, 0.15), ped_rate=_u(rng, 0.0, 0.05))
    return plan, dict(kind="slows")


def _yellow_go(sim, rng, dt, variant=None):
    """Cruising slowly toward a green; it turns yellow just before frame 31 when the ego is 3 to 5 m from its
    line, too close to stop (drive_v3's yellow rule: it would need over 3 m/s^2), so the ego drives on."""
    m, cfg, ln, own, other = _signal_setup(sim, rng)
    t0 = sim.t
    v_e = _u(rng, 6.0, 7.0)
    d_y = _u(rng, 3.0, 5.0)                       # front to line when the light turns
    t_y = _t(_u(rng, 5.6, 5.9))
    plan = dict(lane=ln, turn="S", cap=False, v=v_e, clear_ahead=300.0, post=[_cruise, _no_peds], D="straight",
                extra=dict(v0frac=v_e / ln.limit, stop_dist=2.0, s0=1.0), draw_route=False)
    plan["s"] = _s_for_line(ln, d_y + v_e * (t_y + 0.2 + _shift() / cfg["DT_FPS"]))
    sim.populate(_u(rng, 0.0, 0.15), ped_rate=0.0)
    sim.signals.schedule(own, "Y", t_y, t0, _dur(rng, cfg, {}))
    return plan, dict(kind="slows", carry=True, light=BALL_Y)


def _left_go(sim, rng, dt, variant=None):
    """Waiting at its line to turn left on a green ball (no arrow) while two oncoming cars pass; the last
    one passes just before frame 31, leaving the junction clear, and the ego turns from frame 32."""
    m, cfg, ln, own, other = _signal_setup(sim, rng, lane=0)
    k = sim.ego_leg
    opp = _in_lanes(m, (k + 2) % 4)
    for o in opp:
        sim.gap_lanes[o.id] = 60.0                    # no other oncoming arrivals to speak of
    t0 = sim.t
    sim.signals.schedule(own, "G", _t(_u(rng, -6.0, -2.0)), t0, _dur(rng, cfg, {(own, "G"): 40.0}))
    plan = dict(lane=ln, turn="L", cap=True, v=0.0, clear_ahead=300.0, post=[_rest_at_line], D="left turn",
                extra=dict(start_delay=_u(rng, *cfg["DT_START_DELAY"])))
    plan["s"] = _s_for_line(ln, _u(rng, 0.8, 2.0))
    v_on = 50 / 3.6 * _u(rng, 0.7, 0.9)
    t_last = _t(_u(rng, 4.8, 5.6)) - dt               # the last oncoming front reaches the conflict then
    times = [(t_last - _u(rng, 1.8, 3.0), v_on), (t_last, v_on)]

    def oncoming(s, e, r):
        cl, s_conf = _conflict_lane(s.m, ln, "L", [(k + 2) % 4], "S")
        if cl is None:
            raise Skip("no oncoming lane")
        for o in opp:
            _clear(s, o.id, 0.0, o.length)
        s.peds = [p for p in s.peds if p.cw != (k + 3) % 4 or p.state == 4]   # nobody on the crosswalk the turn passes
        n0 = {u.slot for u in s.vehicles}
        _timed_cars(s, r, cl, s_conf, times, route="S")
        s.cue_slots = [u.slot for u in s.vehicles if u.slot not in n0]
        if len(s.cue_slots) < 2:
            raise Skip("no room for the oncoming cars")
    plan["post"].append(oncoming)
    sim.populate(_u(rng, 0.2, 0.5), ped_rate=_u(rng, 0.0, 0.05))
    return plan, dict(kind="starts", light=BALL_G)


def _right_green(sim, rng, dt, variant=None):
    """Cruising toward a green in the straight-and-right lane, about to turn right; nobody on the crosswalks.
    It holds its speed until the turn is 14 to 18 m off, then slows for it at 2 to 3 m/s^2 from frame 32."""
    m, cfg, ln, own, other = _signal_setup(sim, rng)
    t0 = sim.t
    sim.signals.schedule(own, "G", _t(_u(rng, -6.0, -2.0)), t0, _dur(rng, cfg, {(own, "G"): 40.0}))
    v_e = ln.limit * _u(rng, 0.65, 0.8)
    cl = _u(rng, 14.0, 18.0)
    plan = dict(lane=ln, turn="R", cap=False, v=v_e, clear_ahead=300.0, post=[_cruise, _no_peds], D="right turn",
                extra=dict(v0frac=v_e / ln.limit, curve_late=cl, zone_brake=_u(rng, 2.0, 3.0)), draw_route=False)
    plan["s"] = _s_for_line(ln, cl + 2.0 + v_e * ((T0 + 1) / cfg["DT_FPS"] - dt))
    sim.populate(_u(rng, 0.1, 0.4), ped_rate=0.0)
    return plan, dict(kind="slows", light=BALL_G)


def _left_free(sim, rng, dt, variant=None):
    """Cruising toward a green ball in the left-turn lane with no oncoming traffic, about to turn left; it
    holds its speed until the turn is 12 to 16 m off, then slows for it at 2 to 3 m/s^2 from frame 32."""
    m, cfg, ln, own, other = _signal_setup(sim, rng, lane=0)
    k = sim.ego_leg
    opp = _in_lanes(m, (k + 2) % 4)
    for o in opp:
        sim.gap_lanes[o.id] = 60.0
    t0 = sim.t
    sim.signals.schedule(own, "G", _t(_u(rng, -6.0, -2.0)), t0, _dur(rng, cfg, {(own, "G"): 40.0}))
    v_e = ln.limit * _u(rng, 0.6, 0.75)
    cl = _u(rng, 12.0, 16.0)
    plan = dict(lane=ln, turn="L", cap=False, v=v_e, clear_ahead=300.0, post=[_cruise, _no_peds], D="left turn",
                extra=dict(v0frac=v_e / ln.limit, curve_late=cl, zone_brake=_u(rng, 2.0, 3.0)))
    plan["s"] = _s_for_line(ln, cl + 2.0 + v_e * ((T0 + 1) / cfg["DT_FPS"] - dt))

    def clear(s, e, r):
        for o in opp:
            _clear(s, o.id, 0.0, o.length)
    plan["post"].append(clear)
    sim.populate(_u(rng, 0.1, 0.4), ped_rate=0.0)
    return plan, dict(kind="slows", light=BALL_G)


def _right_red(sim, rng, dt, variant=None):
    """Standing at a red in the straight-and-right lane, about to turn right, while a car crosses from its left;
    once it has passed and cleared the lane the ego turns into, just before frame 31, the ego turns right on red
    from frame 32 (it takes any gap of 5 s or more)."""
    m, cfg, ln, own, other = _signal_setup(sim, rng)
    k = sim.ego_leg
    t0 = sim.t
    sim.signals.schedule(other, "G", _t(_u(rng, -9.0, -6.0)), t0, _dur(rng, cfg, {(other, "G"): 40.0}))
    plan = dict(lane=ln, turn="R", cap=True, v=0.0, clear_ahead=300.0, post=[_rest_at_line, _no_peds], D="right turn",
                extra=dict(start_delay=_u(rng, *cfg["DT_START_DELAY"])), draw_route=False)
    plan["s"] = _s_for_line(ln, _u(rng, 0.8, 2.0))
    v_x = 50 / 3.6 * _u(rng, 0.7, 0.9)
    t_last = _t(_u(rng, 3.4, 3.8)) - dt               # the exit clears about 2.6 s after it reaches the conflict
    times = [(t_last, v_x)]
    left_leg = (k + 3) % 4

    def cross(s, e, r):
        cl, s_conf = _conflict_lane(s.m, ln, "R", [left_leg], "S")
        if cl is None:
            raise Skip("no cross lane")
        for c in _in_lanes(s.m, left_leg):
            _clear(s, c.id, 0.0, c.length)
            s.gap_lanes[c.id] = 60.0
        for o in s.m.lanes:   # and the lanes beside them, so the timed cars have room
            if o.kind == "out" and o.leg == left_leg:
                _clear(s, o.id, 0.0, o.length)
        n0 = {u.slot for u in s.vehicles}
        _timed_cars(s, r, cl, s_conf, times, route="S")
        s.cue_slots = [u.slot for u in s.vehicles if u.slot not in n0]
        if len(s.cue_slots) < len(times):
            raise Skip("no room for the cross traffic")
    plan["post"].append(cross)
    sim.populate(_u(rng, 0.2, 0.5), ped_rate=0.0)
    return plan, dict(kind="starts", light=BALL_R)


def _stop_lead(sim, rng, dt, variant=None):
    """Following a car toward an all-way stop; it brakes steadily to its stop at the line just before frame 31
    (its brake lamps on), and the ego slows behind it from frame 32."""
    m, cfg = sim.m, sim.cfg
    ln = _in_lanes(m, sim.ego_leg)[0]
    v_l = ln.limit * _u(rng, 0.7, 0.85)
    b = _u(rng, 2.2, 3.0)
    extra, lead = _follow_setup(sim, rng, ln, v_l, kind="car")
    d_trig = v_l ** 2 / (2 * b) + 1.0                    # the lead's front to the line when it starts to brake

    def lead_stop(s, e, r):
        lead(s, e, r)
        u = next(x for x in s.vehicles if x.slot == s.cue_slot)
        u.brake_at, u.stop_dist, u.s0, u.creep = b, 2.0, 1.0, False
        u.turn_w = {x: float(x == "S") + 1e-9 for x in "LSR"}
    plan = dict(lane=ln, turn="S", cap=False, v=None, clear_ahead=300.0, post=[lead_stop, _no_peds], D="straight",
                extra=extra, draw_route=False)
    gap = 1.5 + v_l * extra["T"]                        # roughly where the ego follows
    plan["s"] = _s_for_line(ln, d_trig + 4.5 + gap + v_l * ((T0 + 1) / cfg["DT_FPS"] - dt))
    if plan["s"] < 5.0:
        raise Skip("approach too short")
    sim.populate(_u(rng, 0.0, 0.1), ped_rate=0.0)
    return plan, dict(kind="slows", cue=lambda s: [s.cue_slot])


def _yellow_lead(sim, rng, dt, variant=None):
    """Following a car toward a green; it turns yellow and the car ahead, able to stop, brakes steadily to its
    line just before frame 31 (brake lamps on), and the ego slows behind it from frame 32."""
    m, cfg, ln, own, other = _signal_setup(sim, rng)
    t0 = sim.t
    v_l = _u(rng, 7.5, 9.0)
    b = _u(rng, 2.2, 2.8)
    extra, lead = _follow_setup(sim, rng, ln, v_l, kind="car")
    d_trig = v_l ** 2 / (2 * b) + 1.0                    # the lead's front to the line when it starts to brake
    t_y = _t(_u(rng, 5.6, 5.9)) - dt                      # the yellow; the lead's reaction follows

    def lead_stop(s, e, r):
        lead(s, e, r)
        u = next(x for x in s.vehicles if x.slot == s.cue_slot)
        u.brake_at, u.stop_dist, u.s0 = 0.1, 2.0, 1.0
        u.turn_w = {x: float(x == "S") + 1e-9 for x in "LSR"}
    plan = dict(lane=ln, turn="S", cap=False, v=None, clear_ahead=300.0, post=[lead_stop, _no_peds], D="straight",
                extra=dict(extra, stop_dist=2.0), draw_route=False)
    gap = 1.5 + v_l * extra["T"]
    plan["s"] = _s_for_line(ln, d_trig + 4.5 + gap + v_l * (T0 + 1) / cfg["DT_FPS"])
    sim.populate(_u(rng, 0.0, 0.1), ped_rate=0.0)
    sim.signals.schedule(own, "Y", t_y, t0, _dur(rng, cfg, {}))
    return plan, dict(kind="slows", cue=lambda s: [s.cue_slot])


# change: what the change twin's futures do and the carry-on twin's do not ("starts": the 4 s outcome;
# "slows": the speed falls by at least 1.5 m/s within 4 s)
TWINS = {
    "red_green": dict(template="signal4", scene=_red_green, change="starts_straight", group="signals"),
    "green_yellow": dict(template="signal4", scene=_green_yellow, change="slows", group="signals"),
    "queue": dict(template="signal4", scene=_queue, change="starts", group="other vehicles"),
    "left_gap": dict(template="signal4", scene=_left_gap, change="starts", group="signals",
                     map_opts=dict(protected=True)),
    "ped_step": dict(template="residential", scene=_ped_step, change="slows", group="pedestrians",
                     map_opts=dict(school_zone=False)),
    "overtake": dict(template="multilane", scene=_overtake, change="slows_or_moves", group="other vehicles"),
}
SINGLES = {
    "cut_in": dict(template="multilane", scene=_cut_in, group="other vehicles", aim=True),
    "red_ahead": dict(template="signal4", scene=_red_ahead, group="signals", aim=True),
    "stop_ahead": dict(template="stop4", scene=_stop_ahead, group="stop signs", aim=True),
    "zone_ahead": dict(template="residential", scene=_zone_ahead, group="signs", aim=True,
                       map_opts=dict(school_zone=True)),
    "parked_ahead": dict(template="residential", scene=_parked_ahead, group="other vehicles", aim=True,
                         map_opts=dict(school_zone=False)),
    "lane_end": dict(template="multilane", scene=_lane_end, group="signs", aim=True,
                     map_opts=dict(sign_before=8.0, sign_size=4.0)),
    "stop_sign": dict(template="stop4", scene=_stop_sign, group="stop signs", aim=False, variants=STOP_CASES),
    # train and the train-like part of val and test only (the twin splits keep the 13 situations above)
    "lead_brakes_road": dict(template="multilane", scene=_lead_brakes_road, group="other vehicles", aim=True),
    "lead_brakes_street": dict(template="residential", scene=_lead_brakes_street, group="other vehicles", aim=True,
                               map_opts=dict(school_zone=False)),
    "yellow_red": dict(template="signal4", scene=_yellow_red, group="signals", aim=True),
    "yellow_go": dict(template="signal4", scene=_yellow_go, group="signals", aim=False),
    "left_go": dict(template="signal4", scene=_left_go, group="signals", aim=True, map_opts=dict(protected=False)),
    "cruise_junction": dict(template="signal4", scene=_cruise_junction, group="no decision", aim=False),
    "cruise_road": dict(template="multilane", scene=_cruise_road, group="no decision", aim=False),
    "cruise_street": dict(template="residential", scene=_cruise_street, group="no decision", aim=False,
                          map_opts=dict(school_zone=False)),
    "follow_road": dict(template="multilane", scene=_follow_road, group="no decision", aim=False),
    "follow_street": dict(template="residential", scene=_follow_street, group="no decision", aim=False,
                          map_opts=dict(school_zone=False)),
    # added after the first train build (kept last so earlier family indices hold)
    "right_green": dict(template="signal4", scene=_right_green, group="signals", aim=True),
    "left_free": dict(template="signal4", scene=_left_free, group="signals", aim=True, map_opts=dict(protected=False)),
    "right_red": dict(template="signal4", scene=_right_red, group="signals", aim=True),
    "stop_lead": dict(template="stop4", scene=_stop_lead, group="other vehicles", aim=True),
    "cut_in_far": dict(template="multilane", scene=_cut_in, group="other vehicles", aim=False, fixed_variant=1),
    "merge_in": dict(template="multilane", scene=_cut_in, group="other vehicles", aim=True, fixed_variant=2),
    "parked_wait": dict(template="residential", scene=_parked_ahead, group="other vehicles", aim=True,
                        map_opts=dict(school_zone=False), fixed_variant=1),
    "yellow_lead": dict(template="signal4", scene=_yellow_lead, group="signals", aim=True),
}
FAMILIES = list(TWINS) + list(SINGLES)
TWIN_SPLIT = FAMILIES[:13]   # the situations of val_twins and test_twins
WAIT_TWINS = ("red_green", "queue", "left_gap")   # their carry-on ego stays put to the clip's end
# the ego may start again late here: what it waits for (a crossing pedestrian, a stop, cars under way) shows by frame 31
RESTARTS = ("ped_step", "ped_zone", "parked_wait", "parked_ahead", "stop_sign", "stop_ahead", "stop_lead")
assert TWIN_SPLIT[6:] == ["cut_in", "red_ahead", "stop_ahead", "zone_ahead", "parked_ahead", "lane_end", "stop_sign"]
# twin situations added after the first train build go last, so earlier family indices hold
TWINS["ped_zone"] = dict(template="residential", scene=_ped_step, change="slows", group="pedestrians",
                         map_opts=dict(school_zone=True, zone_before=100.0), fixed_variant=1)
FAMILIES.append("ped_zone")


# ---------------------------------------------------------------------------
# simulation
# ---------------------------------------------------------------------------

def _fork_moves(s, rec, m, seed, attempt, b, K, balanced=False):
    """The ego's move in each of the K futures, or None when it is fixed. In a lane with several moves the
    futures draw it with equal odds (balanced: as evenly as K allows), except that an ego which stood at a
    red that has turned green goes straight: a driver turning right would already have turned on red."""
    ego = s.ego
    if ego is None or not ego.route_pending or ego.entered:
        return None
    i, _ = ego.where()
    moves = [m.lanes[x].turn for x in ego.lanes[i].next]
    if len(moves) < 2:
        return None
    gi = ego_approach(rec, m)
    if gi is not None and "S" in moves:
        ball, v = rec["signals"][:T0 + 1, gi, 0].astype(int), rec["ego"][:T0 + 1, 3]
        if ball[T0] == BALL_G and ((ball == BALL_R) & (v < V_STOP)).any():
            return ["S"] * K
        if STAGE == "clean" and ball[T0] in (BALL_R, BALL_G) and (v < V_STOP).any():
            return ["S"] * K
    r = np.random.RandomState(_fork_seed(seed, attempt, b, 99))
    if balanced and K >= len(moves):
        return [moves[i] for i in r.permutation(np.arange(K) % len(moves))]
    return route_draws(moves, K, r)


def _first_difference(a, b):
    """The first frame (0 to 31) on which the twins differ on screen, or -1."""
    diff = np.zeros(T0 + 1, bool)
    for key in ("agents", "signals", "lamps"):
        x, y = a[key][:T0 + 1].astype(np.float64), b[key][:T0 + 1].astype(np.float64)
        diff |= np.abs(x - y).reshape(T0 + 1, -1).max(1) > 1e-6
    f = np.flatnonzero(diff)
    return int(f[0]) if len(f) else -1


def _run_future(s, m, cfg, rng_seed, move, cued_slot, check=True, spec=None):
    fk = copy.deepcopy(s, memo={id(m): m})
    if rng_seed is not None:
        fk.rng = np.random.RandomState(rng_seed)
    spec = spec or {}
    if move is not None:
        fk.set_ego_route(move)
        if spec.get("regrant"):   # a let-through earned on the provisional move does not carry over
            fk.ego.granted = False
    total = cfg["FRAMES"] + cfg["FRAMES_FUTURE"]
    while fk.frame < total - 1:
        fk.step_frame()
    pk = _pack(fk)
    pk["colors"] = {u.slot: u.color for u in fk.vehicles + fk.peds + fk.gone}   # for the renderer (export.scene_dict)
    return pk, (_problems(pk, m, cfg, cued_slot, spec.get("allow_r0", False), spec.get("wait_until"))
                if check else None)


def _setup(rng, cfg, fam, dt, variant=None):
    variant = fam.get("fixed_variant", variant)
    """Map, simulator and ego, plus the scene's spec. The warm-up is longer by _shift() frames and the ego is
    placed that much earlier (at the plan's frame, counted from the warm-up's start, or at -1 - _shift())."""
    leg = int(rng.randint(4))
    m = build_map(fam["template"], rng, cfg, **fam.get("map_opts", {}))
    sim = Sim(m, dict(cfg, DT_WARMUP=cfg["DT_WARMUP"] + _shift()), rng)
    sim.parked_stop = STAGE in ("final", "clean")
    sim.next_slot = 1
    sim.ego_leg = leg
    plan, spec = fam["scene"](sim, rng, dt, variant)
    sim.insert_frame = _insert_frame(plan)
    if not -sim.cfg["DT_WARMUP"] < sim.insert_frame <= -1:
        raise Skip("cannot place the ego early or late enough")
    while sim.frame < sim.insert_frame:
        sim.step_frame()
    _insert_ego(sim, plan, rng)
    return m, sim, spec


def _aim(build, rng, cfg):
    """Build with the random state reset each time, shifting the scene's timing until the ego's speed or
    heading first changes on frame FD + 1: FD is the last frame before it. Returns (built, why, dt)."""
    state = rng.get_state()
    dt, last, scale = 0.0, None, 1.0
    damped = STAGE != "twins"                          # stage "twins" aimed in at most five undamped steps
    for _ in range(AIM_TRIES if damped else 5):
        rng.set_state(state)
        built, f = build(dt)
        if f == FD + 1:
            return built, None, dt
        if f is None:
            dt += 2.0                                  # no change by the clip's end: 2 s earlier
        else:
            if damped and last is not None and (f - FD - 1) * (last - FD - 1) < 0:
                scale *= 0.5                           # overshot: smaller steps
            dt += scale * (f - (FD + 1)) / cfg["DT_FPS"]
        last = f
    return None, f"change not on frame {FD + 1} (last at {last})", dt


def _futures(sims, pre, m, cfg, seed, attempt, K, cued_slot, first_only=False, start=0, spec=None, balanced=False):
    """Futures k = start.. of each simulator, forked at frame 31; future 0 continues its own random stream."""
    moves = [_fork_moves(s, p, m, seed, attempt, b, K, balanced) for b, (s, p) in enumerate(zip(sims, pre))]
    out = [[] for _ in sims]
    for b, s in enumerate(sims):
        for k in range(start, 1 if first_only else K):
            pk, why = _run_future(s, m, cfg, None if k == 0 else _fork_seed(seed, attempt, b, k),
                                  None if moves[b] is None else moves[b][k], cued_slot, spec=spec)
            if why:
                return None, moves, why
            out[b].append(pk)
    return out, moves, None


def _brakes_ok(rec):
    """Whether the ego brakes at most MAX_BRAKE over the 4 s after the decision."""
    v = np.asarray(rec["ego"], np.float64)[FD:FD + H + 1, 3]
    return len(v) < 2 or float((-np.diff(v) * 5.0).max()) <= MAX_BRAKE


def _gap_clear(rec, allowed):
    """Whether at most ``allowed`` cars come toward the ego within GAP_M at the decision."""
    e, a = np.asarray(rec["ego"][FD], np.float64), np.asarray(rec["agents"][FD], np.float64)[1:]
    h = e[2]
    ahead = (a[:, 1:3] - e[:2]) @ np.array([np.cos(h), np.sin(h)])
    toward = np.abs(np.angle(np.exp(1j * (a[:, 3] - h - np.pi)))) < 0.6
    return int(((a[:, 9] > 0) & toward & (a[:, 4] > 1.0) & (ahead > 0) & (ahead < GAP_M)).sum()) <= allowed


def _stands(rec):
    """Whether the ego stays put from frame 31 to the clip's end."""
    e = np.asarray(rec["ego"], np.float64)
    return moved(e) < STILL_M and e[31:, 3].max() < V_STOP


def _dips(rec):
    """Whether, once the ego's speed has risen by 3 km/h, it drops by 2 while the ego yields, or dips by 2
    without stopping and rises by 3 again."""
    x = np.asarray(rec["ego"], np.float64)[:, 3] * 3.6
    state = np.asarray(rec["extra"])[:, 0, 3].astype(int)
    low, peak, first = x[0], None, True
    for f in range(1, len(x)):
        if peak is None:
            low = min(low, x[f])
            peak = x[f] if x[f] >= low + 3.0 else None
            continue
        peak = max(peak, x[f])
        if x[f] <= peak - 2.0:
            if state[f] in YIELDS:
                return True
            if first:
                first = False
                dip = f + int(np.argmin(x[f:]))
                if x[dip] > 0.5 and x[dip:].max() >= x[dip] + 3.0:
                    return True
    return False


def _late_change(rec):
    """Whether an ego that has held steady for six frames changes from frame 34 on."""
    e = np.asarray(rec["ego"], np.float64)
    v, h = e[:, 3], np.unwrap(e[:, 2])
    steady = (np.abs(np.diff(v)) <= V_TOL) & (np.abs(np.diff(h)) <= H_TOL)
    for t in range(33, 60):
        if steady[t - 6:t].all() and not steady[t]:
            w = min(t + 12, len(v) - 1)
            if np.abs(v[t:w + 1] - v[t]).max() >= 1.0 or np.abs(h[t:w + 1] - h[t]).max() >= 0.15:
                return True
    return False


def _holds(pre):
    """Whether the ego's speed and heading hold over the window FD - 7 to FD."""
    e = pre["ego"][_window():FD + 1]
    return np.abs(np.diff(e[:, 3])).max() <= V_TOL and np.abs(np.angle(np.exp(1j * np.diff(e[:, 2])))).max() <= H_TOL


def _split_frame(sim, twin):
    return sim.insert_frame if twin["split"] == -1 else twin["split"]


def _build_pair(rng, cfg, fam, seed, attempt, dt):
    m, sim, twin = _setup(rng, cfg, fam, dt)
    while sim.frame < _split_frame(sim, twin):
        sim.step_frame()
    sims = [copy.deepcopy(sim, memo={id(m): m}) for _ in range(2)]
    for b, fn in enumerate(twin["apply"]):
        if fn is not None:
            fn(sims[b], np.random.RandomState(_fork_seed(seed, attempt, b, 300)))
    for s in sims:
        while s.frame < T0:
            s.step_frame()
    pre = [_pack(s) for s in sims]
    moves0 = _fork_moves(sims[0], pre[0], m, seed, attempt, 0, 1)
    rec0, _ = _run_future(sims[0], m, cfg, None, None if moves0 is None else moves0[0], None, check=False)
    return (m, sims, pre, twin), onset(rec0, last=len(rec0["ego"]) - 1)


def _attempt_pair(rng, cfg, name, seed, attempt, K, variant=None):
    fam = TWINS[name]
    try:
        built, why, _ = _aim(lambda dt: _build_pair(rng, cfg, fam, seed, attempt, dt), rng, cfg)
    except Skip as e:
        return None, f"setup: {e}"
    if built is None:
        return None, why
    m, sims, pre, twin = built
    if np.abs(pre[0]["ego"] - pre[1]["ego"]).max() > SAME_EGO:
        return None, "ego differs before frame 32"
    if twin.get("light") is not None and tuple(_light(p, m) for p in pre) != twin["light"]:
        return None, "light at the decision not as planned"
    if twin.get("cue") is not None:
        c = twin["cue_twin"]
        slot = twin["cue"](sims[c])
        if slot is None or not in_view(pre[c], slot):
            return None, "cue not in view at the decision"
    fspec = None if STAGE == "twins" else twin
    first, moves, why = _futures(sims, pre, m, cfg, seed, attempt, K, None, first_only=True, spec=fspec)
    if why:
        return None, why
    if not changes(first[0][0], fam["change"]) or changes(first[1][0], fam["change"]):
        return None, f"future 0 of the twins: {OUTCOMES[outcome(first[0][0])]} / {OUTCOMES[outcome(first[1][0])]}"
    if STAGE == "clean" and (_dips(first[0][0]) or _dips(first[1][0])):
        return None, "speed dips on the way away"
    if STAGE == "clean" and name not in RESTARTS and (_late_change(first[0][0]) or _late_change(first[1][0])):
        return None, "change after frame 33"
    rest, _, why = _futures(sims, pre, m, cfg, seed, attempt, K, None, start=1, spec=fspec)
    if why:
        return None, why
    recs = [first[b] + rest[b] for b in range(2)]
    if STAGE == "clean" and name in WAIT_TWINS and not all(_stands(r) for r in recs[1]):
        return None, "carry-on ego moves after the decision"
    oc = [np.array([outcome(r) for r in rs]) for rs in recs]
    ch = [np.array([changes(r, fam["change"]) for r in rs]) for rs in recs]
    agree = [int(ch[0].sum()), int((~ch[1]).sum())]
    if min(agree) < min(MIN_AGREE, K):
        return None, f"futures agree {agree[0]} / {agree[1]}"
    return dict(map=m, template=fam["template"], branches=[rs[0] for rs in recs],
                forks=[[_future(r, T0) for r in rs] for rs in recs],
                fork_recs=recs if cfg.get("DT_KEEP_FORK_RECS") else None, ok=True,
                labels=[clip_labels(rs[0], m, cfg) for rs in recs], route_draws=moves,
                target=dict(template=fam["template"]), family=name, cue_frame=_first_difference(pre[0], pre[1]),
                outcomes=oc, change=ch, onsets=[np.array([onset(r, last=len(r["ego"]) - 1) or -1 for r in rs]) for rs in recs],
                colors=[rs[0]["colors"] for rs in recs], variant=-1, open_moves="", decision_frame=FD,
                members=[0, 1]), None


def _build_member(rng, cfg, fam, seed, attempt, dt, b):
    """One member of a twin situation (b = 0 change, 1 carry-on), its own twin's change applied at the split."""
    m, sim, twin = _setup(rng, cfg, fam, dt)
    while sim.frame < _split_frame(sim, twin):
        sim.step_frame()
    fn = twin["apply"][b]
    if fn is not None:
        fn(sim, np.random.RandomState(_fork_seed(seed, attempt, b, 300)))
    while sim.frame < T0:
        sim.step_frame()
    pre = _pack(sim)
    moves0 = _fork_moves(sim, pre, m, seed, attempt, b, 1)
    rec0, _ = _run_future(sim, m, cfg, None, None if moves0 is None else moves0[0], None, check=False)
    return (m, sim, pre, twin), onset(rec0, last=len(rec0["ego"]) - 1)


def _attempt_member(rng, cfg, name, seed, attempt, K, b):
    """A clip of one member of a twin situation: the timing is aimed on the change member, and the carry-on
    member is the same scene with its own twin's change instead."""
    fam = TWINS[name]
    state = rng.get_state()
    try:
        built, why, dt = _aim(lambda dt: _build_member(rng, cfg, fam, seed, attempt, dt, 0), rng, cfg)
        if built is not None and b == 1:
            rng.set_state(state)
            built, _ = _build_member(rng, cfg, fam, seed, attempt, dt, 1)
    except Skip as e:
        return None, f"setup: {e}"
    if built is None:
        return None, why
    m, sim, pre, twin = built
    if not _holds(pre):
        return None, "ego speed or heading changes before the decision"
    if twin.get("light") is not None and _light(pre, m) != twin["light"][b]:
        return None, "light at the decision not as planned"
    if twin.get("cue") is not None and twin["cue_twin"] == b:
        slot = twin["cue"](sim)
        if slot is None or not in_view(pre, slot):
            return None, "cue not in view at the decision"
    out, moves, why = _futures([sim], [pre], m, cfg, seed, attempt, K, None, spec=twin)
    if why:
        return None, why
    recs = out[0]
    if not _brakes_ok(recs[0]):
        return None, "brakes too hard"
    if STAGE == "clean" and _dips(recs[0]):
        return None, "speed dips on the way away"
    if STAGE == "clean" and name not in RESTARTS and _late_change(recs[0]):
        return None, "change after frame 33"
    if STAGE == "clean" and b == 1 and name in WAIT_TWINS and not all(_stands(r) for r in recs):
        return None, "carry-on ego moves after the decision"
    ch = np.array([changes(r, fam["change"]) for r in recs])
    want = ch if b == 0 else ~ch
    if not want[0] or int(want.sum()) < min(MIN_AGREE, K):
        return None, f"futures carry out the {'change' if b == 0 else 'carry-on'} in {int(want.sum())} of {K}"
    return dict(map=m, template=fam["template"], branches=[recs[0]], forks=[[_future(r, T0) for r in recs]],
                fork_recs=[recs] if cfg.get("DT_KEEP_FORK_RECS") else None, ok=True,
                labels=[clip_labels(recs[0], m, cfg)], route_draws=moves, target=dict(template=fam["template"]),
                family=name, cue_frame=-1, outcomes=[np.array([outcome(r) for r in recs])], change=[ch],
                onsets=[np.array([onset(r, last=len(r["ego"]) - 1) or -1 for r in recs])], colors=[recs[0]["colors"]],
                variant=-1, open_moves="", decision_frame=FD, members=[b]), None


def _build_single(rng, cfg, fam, seed, attempt, dt, variant=None):
    m, sim, spec = _setup(rng, cfg, fam, dt, variant)
    cued = spec["cued"](sim) if spec.get("cued") else None
    while sim.frame < T0:
        sim.step_frame()
    pre = _pack(sim)
    moves0 = _fork_moves(sim, pre, m, seed, attempt, 0, 1)
    rec0, _ = _run_future(sim, m, cfg, None, None if moves0 is None else moves0[0], cued, check=False)
    return (m, sim, spec, pre, cued), onset(rec0, last=len(rec0["ego"]) - 1)


def _attempt_single(rng, cfg, name, seed, attempt, K, variant=None):
    fam = SINGLES[name]
    try:
        if fam["aim"]:
            built, why, _ = _aim(lambda dt: _build_single(rng, cfg, fam, seed, attempt, dt, variant), rng, cfg)
        else:
            built, why = _build_single(rng, cfg, fam, seed, attempt, 0.0, variant)[0], None
    except Skip as e:
        return None, f"setup: {e}"
    if built is None:
        return None, why
    m, sim, spec, pre, cued = built
    if spec["kind"] == "none":   # a clip without a decision: the ego holds its motion through all 64 frames
        return _attempt_steady(m, sim, spec, pre, cfg, seed, attempt, K, name, fam, variant)
    if not _holds(pre):
        return None, "ego speed or heading changes before the decision"
    if spec.get("light") is not None and _light(pre, m) != spec["light"]:
        return None, "light at the decision not as planned"
    if spec.get("cue") is not None and not all(in_view(pre, c) for c in spec["cue"](sim)):
        return None, "cue not in view at the decision"
    balanced = spec["kind"] == "goes_now"
    expect = not spec.get("carry", False)   # a carry-on clip: the right decision is to keep going
    first, moves, why = _futures([sim], [pre], m, cfg, seed, attempt, K, cued, first_only=True, spec=spec,
                                 balanced=balanced)
    if why:
        return None, why
    if not balanced and changes(first[0][0], spec["kind"]) != expect:
        return None, "future 0 " + ("carries on" if expect else "changes")
    rest, _, why = _futures([sim], [pre], m, cfg, seed, attempt, K, cued, start=1, spec=spec, balanced=balanced)
    if why:
        return None, why
    recs = first[0] + rest[0]
    if not _brakes_ok(recs[0]):
        return None, "brakes too hard"
    if STAGE == "clean" and _dips(recs[0]):
        return None, "speed dips on the way away"
    if STAGE == "clean" and name not in RESTARTS and _late_change(recs[0]):
        return None, "change after frame 33"
    if STAGE == "clean" and name in LEFT_GAPS and not _gap_clear(recs[0], LEFT_GAPS[name]):
        return None, "gap not clear at the decision"
    ch = np.array([changes(r, spec["kind"]) for r in recs])
    open_moves = ""
    if balanced:   # every future goes at once exactly where its move is open (OPEN_MOVES)
        opened = set(OPEN_MOVES[spec["case"]])
        drawn = moves[0] if moves[0] is not None else ([spec["move"]] * len(ch) if spec.get("move") else None)
        if drawn is None or any(c != (mv in opened) for c, mv in zip(ch, drawn)):
            return None, f"futures do not follow the open moves {sorted(opened)}"
        open_moves = "".join(sorted(opened))
    elif int((ch == expect).sum()) < min(MIN_AGREE, K):
        return None, f"futures {'change' if expect else 'carry on'} {int((ch == expect).sum())}"
    return dict(map=m, template=fam["template"], branches=[recs[0]], forks=[[_future(r, T0) for r in recs]],
                fork_recs=[recs] if cfg.get("DT_KEEP_FORK_RECS") else None, ok=True,
                labels=[clip_labels(recs[0], m, cfg)],
                route_draws=[[spec["move"]] * len(recs)] if (moves[0] is None and spec.get("move")) else moves,
                target=dict(template=fam["template"]),
                family=name, cue_frame=-1, outcomes=[np.array([outcome(r) for r in recs])], change=[ch],
                onsets=[np.array([onset(r, last=len(r["ego"]) - 1) or -1 for r in recs])], colors=[recs[0]["colors"]],
                variant=-1 if variant is None else int(variant), open_moves=open_moves, decision_frame=FD,
                members=[0 if expect else 1]), None


def _attempt_steady(m, sim, spec, pre, cfg, seed, attempt, K, name, fam, variant):
    """A clip without a major decision: its speed and heading hold (within the tolerances) over frames 0 to 63."""
    out, moves, why = _futures([sim], [pre], m, cfg, seed, attempt, K, None, spec=spec)
    if why:
        return None, why
    recs = out[0]
    e = np.asarray(recs[0]["ego"], np.float64)
    if np.abs(np.diff(e[:, 3])).max() > 2 * V_TOL or np.abs(np.angle(np.exp(1j * np.diff(e[:, 2])))).max() > 2 * H_TOL:
        return None, "the ego changes what it does"
    return dict(map=m, template=fam["template"], branches=[recs[0]], forks=[[_future(r, T0) for r in recs]],
                fork_recs=[recs] if cfg.get("DT_KEEP_FORK_RECS") else None, ok=True,
                labels=[clip_labels(recs[0], m, cfg)], route_draws=moves, target=dict(template=fam["template"]),
                family=name, cue_frame=-1, outcomes=[np.array([outcome(r, T0) for r in recs])],
                change=[np.zeros(len(recs), bool)], onsets=[np.full(len(recs), -1)], colors=[recs[0]["colors"]],
                variant=-1 if variant is None else int(variant), open_moves="", decision_frame=-1, members=[0]), None


def simulate_pair(seed, cfg, name, tries=None, variant=None):
    """Both twins of one pair of ``name`` (see TWINS), decision on frame FD, or dict(ok=False, reasons=...)."""
    return _simulate(_attempt_pair, seed, cfg, name, tries, variant)


def simulate_single(seed, cfg, name, tries=None, variant=None):
    """One single clip of ``name`` (see SINGLES), or dict(ok=False, reasons=...). ``variant`` picks a
    scene's case where it has several (the all-way stop's STOP_CASES)."""
    return _simulate(_attempt_single, seed, cfg, name, tries, variant)


def simulate_clip(seed, cfg, name, fd, member=0, variant=None, tries=None):
    """One clip of ``name`` with its decision on frame fd (None: a clip without one): a twin situation's
    change (member 0) or carry-on (member 1), or a single situation, with cfg["DT_FORKS"] futures forked at
    frame 31 (one for train). The format of simulate_single, or dict(ok=False, reasons=...)."""
    set_decision_frame(31 if fd is None else fd)
    if name in SINGLES:
        return _simulate(_attempt_single, seed, cfg, name, tries, variant)
    return _simulate(lambda rng, c, n, s, a, K, v: _attempt_member(rng, c, n, s, a, K, member), seed, cfg, name,
                     tries, variant)


def _simulate(fn, seed, cfg, name, tries, variant=None):
    rng = np.random.RandomState(seed)
    K = int(cfg.get("DT_FORKS", 1))
    n = tries or cfg["DT_TRIES"]
    reasons = {}
    for attempt in range(n):
        res, why = fn(rng, cfg, name, seed, attempt, K, variant)
        if res is not None:
            res.update(tries=attempt + 1, reasons=reasons)
            return res
        reasons[why] = reasons.get(why, 0) + 1
    return dict(ok=False, map=None, tries=n, reasons=reasons, family=name)
