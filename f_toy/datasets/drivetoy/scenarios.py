"""Scene sampling and branched simulation (docs/DRIVETOY.md §4–§5, DATASETv2_plan.md).

``simulate(seed, cfg, target=...)`` (v2) builds a scene for a target combination of the four
diversity labels (``diversity.py``): a recipe picks the template, places the ego so that its
episode falls in frames 5 to 31, schedules the signal and adds the agents the cause needs. The
ego's route (and so its indicator) is decided before frame 0. Every attempt runs one shared past
up to ``DT_BRANCH_FRAME``, then ``DT_BRANCHES`` futures to the last visible frame, which are
labelled. An attempt counts only if every branch has the target light label (A) with the ego
carrying out its response; among those the one matching most of B, then C, then D wins. The
winner is run to the end of the label-only frames and must keep the ego on the map and pass
``diversity.cue_problems``.

``simulate(seed, cfg)`` without a target is the v1 sampler.
"""
import copy

import numpy as np

from .diversity import clip_labels, cue_problems, overtake_problems
from .roads import build_map
from .choices import DEC_COLS, DRIVER_COLS, end_label
from .sim import EV, P_CROSS, P_WAIT, RS, Sim, VEH_KINDS

DECISION_EVENTS = {EV[n] for n in ("SIGNAL", "TURN_ON", "LC_START", "STOP", "GO", "YIELD", "PED_DECIDE", "PED_CROSS")}

# ---------------------------------------------------------------------------
# Target combinations the recipes can build: (A, B, template, D, C)
# ---------------------------------------------------------------------------
COMBOS = []


def _add(A, B, tmpl, Ds, Cs, E="decided", V=None):
    """E is the planned state of the ego's major choices at frame 31 (the sampler's prior; the label
    comes from the decision log). V names a recipe variant that sets up a choice's cue."""
    COMBOS.extend((A, B, tmpl, D, C, E, V) for D in Ds for C in Cs)


_add("steady green", "none", "signal4", ["straight"], ["cruise"], "none")
_add("steady green", "none", "signal4", ["right turn"], ["cruise", "slow down"], "none")
_add("steady green", "lead vehicle", "signal4", ["straight"], ["follow", "slow down"], "none")
_add("steady green", "pedestrian", "signal4", ["straight"], ["stop", "slow down"])   # turns right after 31
_add("steady green", "oncoming traffic", "signal4", ["straight"], ["wait", "stop"], "pending", "left_gap")   # turns left after 31
_add("steady red", "signal", "signal4", ["straight"], ["stop", "wait", "slow down"], "none")
_add("steady red", "signal", "signal4", ["straight"], ["stop", "wait"], "pending", "rtor")   # may turn right on red after 31
_add("steady red", "lead vehicle", "signal4", ["straight"], ["stop", "wait"], "none")
_add("red to green", "signal", "signal4", ["straight"], ["pull away", "stop and go", "slow down"], "none")
_add("red to green", "none", "signal4", ["right turn"], ["pull away", "slow down"], "none")   # held less than frames 5-31
_add("red to green", "lead vehicle", "signal4", ["straight"], ["pull away"], "none")
_add("red to green", "pedestrian", "signal4", ["straight"], ["pull away"])   # right turn held by walkers
_add("green to yellow", "signal", "signal4", ["straight"], ["slow down"])   # the full stop comes after 31
_add("green to yellow", "none", "signal4", ["straight"], ["cruise"])
_add("green to yellow", "none", "signal4", ["straight"], ["cruise"], "pending", "yellow")   # yellow at frame 30 or 31
_add("green to yellow", "lead vehicle", "signal4", ["straight"], ["slow down"])
_add("yellow to red", "signal", "signal4", ["straight"], ["stop", "slow down"], "none")
_add("arrow", "none", "signal4", ["left turn"], ["pull away", "slow down"], "none")
_add("arrow", "signal", "signal4", ["straight"], ["pull away"], "none")          # turns after frame 31
_add("permissive left", "oncoming traffic", "signal4", ["straight"], ["stop and go", "slow down", "pull away"])
_add("permissive left", "none", "signal4", ["left turn"], ["slow down"])
_add("no light", "stop sign", "stop4", ["straight"], ["stop and go", "stop"], "none")
_add("no light", "stop sign", "stop4", ["straight"], ["stop"], "pending", "near_tie")   # two cars stop at once
_add("no light", "none", "stop4", ["left turn", "right turn"], ["pull away"], "none")   # its stop is almost over at frame 0
_add("no light", "crossing traffic", "stop4", ["straight"], ["wait", "stop"])
_add("no light", "lead vehicle", "stop4", ["straight"], ["slow down", "stop"], "none")
_add("no light", "lane end", "multilane", ["merge", "straight"], ["slow down"])
_add("no light", "lane end", "multilane", ["straight"], ["cruise", "slow down"], "pending", "merge")   # signalling at 31
_add("no light", "cut-in vehicle", "multilane", ["straight"], ["slow down", "cruise"])
_add("no light", "none", "multilane", ["straight"], ["cruise"], "pending", "cutin")   # a car signalling beside it at 31
_add("no light", "lead vehicle", "multilane", ["straight"], ["follow", "slow down"], "none")
_add("no light", "lead vehicle", "multilane", ["lane change"], ["cruise"], "decided", "overtake")   # pulls out from behind it
_add("no light", "lead vehicle", "multilane", ["straight"], ["slow down"], "pending", "overtake")   # held up at 31
_add("no light", "none", "multilane", ["straight"], ["cruise"], "none")
_add("no light", "pedestrian", "residential", ["straight"], ["stop", "slow down"])
_add("no light", "speed limit", "residential", ["straight"], ["cruise", "slow down"], "pending", "ped")   # someone nears the curb (school zone)
_add("no light", "parked obstacle", "residential", ["straight"], ["stop", "slow down"])   # goes around after 31
_add("no light", "parked obstacle", "residential", ["straight"], ["stop", "wait"], "pending", "nudge")   # a gap after 31
_add("no light", "parked obstacle", "residential", ["nudge around"], ["cruise", "slow down"])
_add("no light", "none", "residential", ["straight"], ["cruise"], "none")
_add("no light", "speed limit", "residential", ["straight"], ["slow down"], "none")   # slows for the school zone
_add("no light", "lead vehicle", "residential", ["straight"], ["follow"], "none")

VARIANTS = ["left_gap", "rtor", "yellow", "near_tie", "merge", "cutin", "overtake", "ped", "nudge"]

TURN = {"left turn": "L", "right turn": "R"}
EGO_HALF = VEH_KINDS["car"][1] / 2


PLAN_W = dict(B=1.0, C=0.8, D=0.8, E=0.6)   # B before C and D, as the plan orders them; E last


def plan_targets(a_list, counts, total, shares, rng, pending_min=0.2):
    """Targets for a batch whose light labels are fixed (``a_list``). Each clip draws one buildable
    combination with that light label, weighted by how far its B, C, D and E labels are still below
    their shares of ``total`` clips (relative deficit, B weighted most, E least unless pending has
    fallen below ``pending_min``); ``counts`` (aspect -> label -> clips so far) is updated in place,
    so a batch spreads over the scarce labels."""
    out = []
    n = sum(counts["A"].values())
    w_e = PLAN_W["E"] if n == 0 or counts["E"].get("pending", 0) / n >= pending_min else 2.0
    for A in a_list:
        rows = [c for c in COMBOS if c[0] == A]
        w = np.zeros(len(rows))
        for r, (_, B, _, D, C, E, _) in enumerate(rows):
            for key, l, wk in (("B", B, PLAN_W["B"]), ("C", C, PLAN_W["C"]), ("D", D, PLAN_W["D"]), ("E", E, w_e)):
                q = shares[key][l] * total
                w[r] += wk * max(q - counts[key].get(l, 0), 0.0) / max(q, 1.0)
        w = (w + 0.05) ** 3
        A, B, tmpl, D, C, E, V = rows[int(rng.choice(len(rows), p=w / w.sum()))]
        for key, l in (("A", A), ("B", B), ("C", C), ("D", D), ("E", E)):
            counts[key][l] = counts[key].get(l, 0) + 1
        out.append(dict(A=A, B=B, C=C, D=D, E=E, V=V, template=tmpl))
    return out


# ---------------------------------------------------------------------------
# helpers shared by the recipes
# ---------------------------------------------------------------------------

def _u(rng, a, b):
    return float(rng.uniform(a, b))


def _in_lanes(m, leg):
    return sorted((m.lanes[i] for i in m.spawn_lanes if m.lanes[i].leg == leg), key=lambda l: l.lane_index)


def _s_for_line(ln, d_line, L=2 * EGO_HALF):
    """Centre arclength of a vehicle of length L whose front is d_line before the stop line."""
    return ln.stop["s"] - d_line - L / 2


def _clear(sim, lane_id, s0, s1, keep=()):
    """Remove vehicles whose centre is in [s0, s1] on lane_id (and their lane changes into it)."""
    out = []
    for u in sim.vehicles:
        if u in keep or u.ego:
            out.append(u)
            continue
        i, sl = u.where()
        on = u.route[i] == lane_id or (u.lc is not None and u.lc.get("target") == lane_id)
        if on and s0 <= sl <= s1:
            continue
        out.append(u)
    sim.vehicles = out


def _room(sim, lane_id, s, L, v):
    """No vehicle on lane_id within a speed-dependent gap of a new one at s (R6 stays clear);
    the ego is left out: its speed is capped to its leader after the scene is set up."""
    for o in sim.vehicles:
        if o.ego:
            continue
        i, so = o.where()
        if o.route[i] == lane_id and abs(so - s) - (o.L + L) / 2 < max(2.0, 0.6 * max(v, o.v)) + 1.0:
            return False
    return True


def _add(sim, lane_id, s, v, kind=None):
    """add_vehicle, never faster than the vehicle's own desired speed there (zones included)."""
    if not _room(sim, lane_id, s, 4.8, v):
        return None
    u = sim.add_vehicle(lane_id, s, v=v, kind=kind)
    if u is not None:
        u.v = min(u.v, sim._v0(u))
        u.stopped = u.v < 0.3
    return u


def _lead_frac(rng, cfg, kind):
    """v3: a leader's desired speed as a fraction of the limit, from its kind's range."""
    return _u(rng, *(cfg["DT_V0_FRAC_HEAVY"] if VEH_KINDS[kind][0] >= 3 else cfg["DT_V0_FRAC"]))


def _add_ahead(sim, rng, ln, s, gap, v, kind=None, v0frac=None, route=None, cap_line=True):
    """A vehicle on ln whose rear is ``gap`` metres ahead of a front bumper at s + EGO_HALF."""
    kind = kind or ("car" if rng.rand() < 0.6 else ("suv" if rng.rand() < 0.6 else "truck"))
    L = VEH_KINDS[kind][1]
    sc = s + EGO_HALF + gap + L / 2
    if ln.stop is not None and sc + L / 2 > ln.stop["s"] - 0.5:
        return None
    if not _room(sim, ln.id, sc, L, v):
        return None
    u = sim.add_vehicle(ln.id, sc, v=v, kind=kind)
    if u is not None and v0frac is not None:
        u.v0frac = v0frac
    if u is not None and route is not None:
        u.turn_w = {t: float(t == route) + 1e-9 for t in "LSR"}
    if u is not None:
        u.v = min(u.v, sim._v0(u))
        if ln.stop is not None and cap_line:   # able to stop at its line, like the initial traffic
            u.v = min(u.v, float(np.sqrt(2 * 2.0 * max(ln.stop["s"] - u.front - 2.0, 0.0))))
        u.stopped = u.v < 0.3
    return u


def _add_ped(sim, rng, path, t_curb, cross=True):
    """A pedestrian on ``path`` that reaches the curb t_curb seconds after frame 0 and then
    crosses (its intent is set: it is decided before frame 31 and visible as it turns)."""
    p = sim.add_ped(path)
    if p is None:
        return None
    p.s = max(0.0, p.cum[-1] - p.speed * (t_curb - sim.t))
    p.heading = p.path_heading()
    if not cross:
        p.force_cross = False   # otherwise its choice at the curb is its own
    return p


def _dur(rng, cfg, over):
    def d(ph):
        if ph in over:
            return over[ph]
        if ph[1] == "G":
            return _u(rng, *cfg["SIG_GREEN_WALK"]) + cfg["SIG_FLASH"]
        if ph[1] == "A":
            return _u(rng, *cfg["SIG_ARROW"])
        return cfg["SIG_YELLOW"] if ph[1] == "Y" else cfg["SIG_ALL_RED"]
    return d


def _stream(sim, rng, ln, s_conf, t_end, gap, v, route=None, v_gap=None):
    """Cars on lane ``ln`` whose fronts reach arclength ``s_conf`` every 1.6 to 2.4 s up to t_end
    seconds after frame 0, then one more ``gap`` seconds later (at ``v_gap``): a stream that keeps
    a waiting driver waiting until t_end, then a gap it may or may not take."""
    times, t = [], t_end
    while t > 1.2:
        times.append((t, v))
        t -= sim.headway(ln.id) if sim.cfg.get("DT_V3") else _u(rng, 1.6, 2.4)   # v3: Issue 3 gaps
    times.append((t_end + gap, v_gap or v))
    cws = {cw for _, _, cw in ln.crosswalks}
    sim.peds = [p for p in sim.peds if p.cw not in cws]   # nobody on its crosswalks: the stream keeps its timing
    for tk, vk in times:
        s_c = s_conf - vk * (tk - sim.t) - 2.25    # centre of a 4.5 m car whose front gets there at tk
        if s_c < 3.0 or s_c > ln.length - 8.0 or any(np.hypot(*(u.pose()[0] - ln.pos(s_c))) < 8.0 for u in sim.vehicles):
            continue
        lim = min(ln.limit_at(x) for x in np.linspace(s_c, s_c + 3.0 * vk, 8))   # within any zone just ahead
        u = sim.add_vehicle(ln.id, s_c, v=min(vk, 0.95 * lim), kind="car")
        if u is None:
            continue
        if route is not None:
            u.turn_w = {x: float(x == route) + 1e-9 for x in "LSR"}
        u.v0frac = min(0.97, vk / ln.limit + 0.05)
        u.T, u.s0 = _u(rng, 1.0, 1.3), _u(rng, 1.5, 2.0)
        if sim.cfg.get("DT_V3"):
            u.T = max(u.T, sim.cfg["DT_ONCOMING_MIN"])


def _conflict_lane(m, ego_ln, ego_turn, legs, route="S"):
    """The first in-lane of ``legs`` (outer lanes first) whose ``route`` movement crosses or joins
    the ego's, with the arclength along it where that conflict starts."""
    c_e = next(m.lanes[c] for c in ego_ln.next if m.lanes[c].turn == ego_turn)
    for leg in legs:
        for cl in _in_lanes(m, leg)[::-1]:
            for c in cl.next:
                z = m.conflicts.get((c, c_e.id))
                if m.lanes[c].turn == route and z is not None:
                    return cl, cl.length + z[0]
    return None, None


def _stream_to(sim, rng, ego_ln, ego_turn, legs, t_end, gap, route="S", clear_cw=None):
    """A stream (``_stream``) of ``route`` traffic from the first approach of ``legs`` that crosses
    or joins the ego's movement, timed at that conflict."""
    cl, s_conf = _conflict_lane(sim.m, ego_ln, ego_turn, legs, route)
    if cl is None:
        return
    _clear(sim, cl.id, 0.0, cl.length)
    _stream(sim, rng, cl, s_conf, t_end, gap, v=cl.limit * _u(rng, 0.7, 0.85), route=route)
    if clear_cw is not None:   # nobody on the crosswalk the ego's turn passes
        sim.peds = [p for p in sim.peds if p.cw != clear_cw or p.state == 4]


# ---------------------------------------------------------------------------
# recipes: each returns the ego plan; the map, the population and the signal are set here too
# ---------------------------------------------------------------------------

def _recipe_signal4(sim, rng, tg):
    m, cfg = sim.m, sim.cfg
    A, B, C, D, V = tg["A"], tg["B"], tg["C"], tg["D"], tg.get("V")
    k = sim.ego_leg
    lanes = _in_lanes(m, k)
    if cfg.get("DT_V3"):   # v3 (Issue 3): the opposite straight-and-right lane brings random gaps
        sim.gap_lanes[_in_lanes(m, (k + 2) % 4)[1].id] = _u(rng, *cfg["DT_ONCOMING_MEAN"])
    inner = D == "left turn" or A in ("arrow", "permissive left") or V == "left_gap" or \
        (A == "steady red" and V != "rtor" and rng.rand() < 0.3)
    ln = lanes[0] if inner else lanes[1]
    turn = TURN.get(D, "L" if inner else ("S" if A in ("red to green", "green to yellow", "steady green") or rng.rand() < 0.6 else "R"))
    if B == "pedestrian" or V == "rtor":
        turn = "R"
    own = ln.stop["group"]
    other = [g for g in m.groups if g != own][0]
    t0 = sim.t
    v0 = 50 / 3.6 * 0.92
    plan = dict(lane=ln, turn=turn, cap=True, v=None, clear_ahead=0.0, post=[])
    density = _u(rng, 0.3, 1.0)
    stopped_at_line = False
    over = {}
    sched = None
    if A == "steady green" and V == "left_gap":
        # waits to turn left while oncoming cars pass until about frame 32, then a gap it may take
        ts = _u(rng, -6.0, -2.0)
        over[(own, "G")] = 20.0
        sched = (own, "G", ts)
        d_line = _u(rng, 0.8, 2.0) if C == "wait" else _u(rng, 12.0, 22.0)
        stopped_at_line = C == "wait"
        plan["post"].append(lambda s, e, r, ln=ln: _stream_to(s, r, ln, "L", [(k + 2) % 4], _u(r, 5.7, 6.6),
                                                              _u(r, 3.2, 4.8), clear_cw=(k + 3) % 4))
    elif A == "steady green":
        ts = _u(rng, -6.0, -1.0)
        t_c = None
        if B == "pedestrian":
            d_line = _u(rng, 6.0, 22.0)
            t_c = _u(rng, 0.3, 2.5)
            over[(own, "G")] = max(9.0 - ts + _u(rng, 1, 4), t_c + 9.5 - ts)
        else:
            d_line = _u(rng, 15.0, 60.0) if D == "straight" else _u(rng, 8.0, 35.0)
            over[(own, "G")] = 9.0 - ts + _u(rng, 1, 4)
        sched = (own, "G", ts)
        plan["cap"] = False
        if t_c is not None:
            out_leg = (k + 1) % 4
            paths = [p for p in m.ped_paths if p["cw"] == out_leg]
            path = paths[int(rng.randint(len(paths)))]
            plan["post"].append(lambda s, e, r, path=path, t_c=t_c: _add_ped(s, r, path, t_c))
    elif A == "steady red":
        ts = _u(rng, -6.0, -2.0)
        over[(other, "G")] = 7.5 - ts + _u(rng, 0, 4)
        sched = (other, "G", ts)
        d_line = dict(stop=_u(rng, 20.0, 55.0), wait=_u(rng, 0.8, 2.0), **{"slow down": _u(rng, 60.0, 90.0)}).get(C, 30.0)
        stopped_at_line = C == "wait"
        if V == "rtor":   # right turn on red: cross traffic until t_end, then a gap it may take
            over[(other, "G")] = 25.0
            if D == "right turn":
                d_line = _u(rng, 0.8, 2.0)
                stopped_at_line, t_end = True, _u(rng, 0.0, 0.8)
                plan["extra"] = dict(style=_u(rng, 0.6, 1.0))   # a quick start, so the turn ends inside the clip
            else:
                d_line = _u(rng, 0.8, 2.0) if C == "wait" else _u(rng, 10.0, 20.0)
                t_end = _u(rng, 5.7, 6.6)
            plan["post"].append(lambda s, e, r, ln=ln, t_end=t_end: _stream_to(
                s, r, ln, "R", [(k + 3) % 4, (k + 1) % 4], t_end, _u(r, 4.5, 7.5), clear_cw=(k + 1) % 4))
    elif A == "red to green":
        if D == "right turn":
            t_g = _u(rng, 0.4, 1.5) if C == "pull away" else _u(rng, 0.5, 2.0)
            d_line = _u(rng, 0.8, 2.0) if C == "pull away" else _u(rng, 12.0, 25.0)
        elif B == "pedestrian":
            t_g = _u(rng, 0.3, 2.0)
            d_line = _u(rng, 0.8, 2.0)
        else:
            t_g = {"pull away": _u(rng, 2.4, 4.0), "stop and go": _u(rng, 4.0, 5.5), "slow down": _u(rng, 1.0, 3.0)}.get(C, 2.0)
            d_line = {"pull away": _u(rng, 0.8, 2.0), "stop and go": _u(rng, 12.0, 25.0), "slow down": _u(rng, 25.0, 50.0)}.get(C, 20.0)
        over[(own, "G")] = 10.0 + _u(rng, 0, 6)
        sched = (own, "G", t_g)
        stopped_at_line = C in ("pull away", "wait")
        if B == "pedestrian":   # walkers on the crosswalk the right turn passes, waiting for WALK
            paths = [p for p in m.ped_paths if p["cw"] == (k + 1) % 4]
            for path in [paths[i] for i in rng.permutation(len(paths))[:1 + int(rng.rand() < 0.5)]]:
                plan["post"].append(lambda s, e, r, path=path, t=t_g - _u(rng, 0.0, 2.0): _add_ped(s, r, path, t))
    elif A == "green to yellow" and V == "yellow":
        # the light turns yellow at frame 30 or 31, where stopping or going is a choice
        t_y = _u(rng, 6.0, 6.18)
        sched = (own, "Y", t_y)
        v_e = v0 * _u(rng, 0.97, 1.0)
        d_line = v_e ** 2 / (2 * _u(rng, 2.3, 3.8)) + 0.5 + v_e * t_y
        plan["cap"], plan["v"], plan["extra"] = False, v_e, dict(v0frac=0.92)
    elif A == "green to yellow":
        t_y = {"stop": _u(rng, 0.5, 1.2), "slow down": _u(rng, 1.0, 3.0)}.get(C, _u(rng, 1.0, 4.0))
        sched = (own, "Y", t_y)
        d_y = {"stop": _u(rng, 26.0, 30.0), "slow down": _u(rng, 30.0, 45.0)}.get(C, _u(rng, 2.0, 14.0))
        d_line = d_y + v0 * t_y
        plan["cap"] = False
        plan["v"] = v0 * _u(rng, 0.95, 1.0)
    elif A == "yellow to red":
        sched = (own, "Y", _u(rng, -2.6, -0.4))
        d_line = _u(rng, 15.0, 45.0) if C == "stop" else _u(rng, 50.0, 80.0)
    elif A == "arrow":
        # B none: turns at once; B signal / lead vehicle: waits, enters late in the clip, turns after 31
        t_a = _u(rng, -1.0, 1.5) if C == "slow down" else \
            {"signal": _u(rng, 3.8, 5.0), "lead vehicle": _u(rng, 2.5, 3.5)}.get(B, _u(rng, 0.3, 1.2))
        over[(own, "A")] = _u(rng, 4.5, 6.5)
        sched = (own, "A", t_a)
        d_line = dict(**{"pull away": _u(rng, 0.8, 2.0), "stop and go": _u(rng, 12.0, 30.0), "slow down": _u(rng, 15.0, 35.0)}).get(C, 15.0)
        stopped_at_line = C == "pull away"
        plan["cap"] = C != "slow down"
    else:   # permissive left: oncoming traffic close enough to wait for, then a gap
        ts = _u(rng, -6.0, 0.5)
        over[(own, "G")] = 9.0 - ts + _u(rng, 1, 4)
        sched = (own, "G", ts)
        stopped_at_line = C == "pull away"
        density = _u(rng, 0.3, 0.6)
        opp = _in_lanes(m, (k + 2) % 4)
        if B == "none":   # rolls through a gap: the turn is done by frame 31
            d_line = _u(rng, 8.0, 25.0)
            n_on, d_first = 0, 0.0
        else:             # waits for one or two oncoming cars, enters after them
            d_line = {"stop and go": _u(rng, 4.0, 10.0), "slow down": _u(rng, 18.0, 30.0), "pull away": _u(rng, 0.8, 2.0)}.get(C, 8.0)
            n_on, d_first = (1, _u(rng, 3.0, 12.0)) if C == "slow down" else (2, _u(rng, 8.0, 20.0))

        def oncoming(s, e, r, n_on=n_on, d=d_first):
            for o in opp:
                _clear(s, o.id, 0.0, o.stop["s"])
            for _ in range(n_on):
                v_on = 50 / 3.6 * _u(r, 0.7, 0.95)
                _add_ahead(s, r, opp[1], _s_for_line(opp[1], d) - EGO_HALF - 1.0, 1.0, v=v_on, route="S",
                           cap_line=False)   # it has the green
                d += v_on * s.headway(opp[1].id) if s.cfg.get("DT_V3") else _u(r, 10.0, 18.0)
            out_cw = (k + 3) % 4      # nobody on the crosswalk the turn passes
            s.peds = [p for p in s.peds if p.cw != out_cw or p.state == 4]
        plan["post"].append(oncoming)
    if stopped_at_line:
        plan["v"] = 0.0
    plan["s"] = _s_for_line(ln, d_line)
    if B in ("signal", "none", "oncoming traffic"):
        plan["clear_ahead"] = 300.0
    elif B == "lead vehicle":
        plan["clear_ahead"] = 300.0
        if stopped_at_line or C == "wait":   # queued behind a stopped leader at the line
            plan["s"] = _s_for_line(ln, d_line + _u(rng, 6.5, 9.0))
            plan["post"].append(lambda s, e, r, ln=ln: _add_ahead(s, r, ln, e.s, max(ln.stop["s"] - e.front - 1.5 - 4.5, 1.5), v=0.0, kind="car", route=turn))
        else:
            if C == "follow" and cfg.get("DT_V3"):   # v3: a leader at its normal speed, 1.0 to 1.8 s ahead
                kind = ("car", "suv", "truck")[int(rng.choice(3, p=[0.5, 0.3, 0.2]))]
                f_l = _lead_frac(rng, cfg, kind)
                v_f = ln.limit * f_l
                plan["v"], plan["cap"] = v_f * _u(rng, 1.0, 1.05), False
                plan.setdefault("extra", {})["v0frac"] = min(0.97, f_l + _u(rng, 0.05, 0.12))
                plan["post"].append(lambda s, e, r, ln=ln, v=v_f, k=kind, f=f_l: _add_ahead(
                    s, r, ln, e.s, v * _u(r, 1.0, 1.8), v=v, kind=k, v0frac=f, route=turn))
            elif C == "follow":   # a slow vehicle close ahead, followed within 7 m
                v_f = _u(rng, 2.5, 4.0)
                plan["v"], plan["cap"] = v_f * _u(rng, 1.0, 1.1), False
                plan["post"].append(lambda s, e, r, ln=ln, v=v_f: _add_ahead(s, r, ln, e.s, _u(r, 4.0, 6.0), v=v,
                                                                             v0frac=v / ln.limit, route=turn))
            else:
                plan["post"].append(lambda s, e, r, ln=ln: _add_ahead(s, r, ln, e.s, _u(r, 5.0, 9.0), v=max(e.v * _u(r, 0.7, 0.9), 0.0), route=turn))
            plan.setdefault("extra", {}).update(T=_u(rng, 1.0, 1.3), s0=_u(rng, 1.5, 2.0))   # follows within 10 m
    sim.populate(density, ped_rate=_u(rng, 0.02, 0.12))
    g, ph, ts = sched
    sim.signals.schedule(g, ph, ts, t0, _dur(rng, cfg, over))
    return plan


def _recipe_stop4(sim, rng, tg):
    m = sim.m
    B, C, D, V = tg["B"], tg["C"], tg["D"], tg.get("V")
    k = sim.ego_leg
    ln = _in_lanes(m, k)[0]
    plan = dict(lane=ln, turn=TURN.get(D, "S"), cap=True, v=None, clear_ahead=0.0, post=[])
    if D in TURN:   # a brisk start, so the turn is done by frame 31
        plan["extra"] = dict(a=_u(rng, 1.8, 2.0))
    density = _u(rng, 0.15, 0.35)
    if V == "near_tie":   # a car on a cross street stops at its line about when the ego does
        d_line = _u(rng, 16.0, 24.0)
        plan["v"] = _u(rng, 5.0, 7.0)
        plan["clear_ahead"] = 300.0

        def twin(s, e, r, k=k, d=d_line):
            cl = _in_lanes(s.m, int(r.choice([(k + 1) % 4, (k + 3) % 4])))[0]
            _clear(s, cl.id, 0.0, cl.length)
            u = s.add_vehicle(cl.id, _s_for_line(cl, d + _u(r, -0.8, 0.8)), v=e.v, kind="car")
            if u is not None:   # it drives like the ego, so both come to their stop together
                for key in ("T", "s0", "a", "b", "stop_dist", "v0frac"):
                    setattr(u, key, getattr(e, key))
                u.turn_w = {x: float(x == "S") + 1e-9 for x in "LSR"}
        plan["post"].append(twin)
    elif B == "none":   # stopped at the line for a while already: it goes as soon as frame 0
        d_line = _u(rng, 0.5, 1.5)
        plan["clear_ahead"] = 300.0

        def stopped(s, e, r):
            e.stop_start = e.zone_t = s.t - _u(r, 0.8, 1.6)
            e.best_stop = s.t - e.stop_start
            e.arrival = e.stop_start if e.best_stop >= 1.0 else None
        plan["post"].append(stopped)
    elif B == "stop sign":
        if D != "straight":   # a brisk driver, so the turn after the stop ends inside the clip
            d_line = _u(rng, 4.0, 6.0)
            plan["v"] = _u(rng, 3.5, 4.5)
            plan["extra"] = dict(a=_u(rng, 1.8, 2.0), style=_u(rng, 0.8, 1.0), stop_dist=_u(rng, 0.5, 1.0), creep=False)
        else:
            d_line = _u(rng, 20.0, 30.0) if C == "stop" else _u(rng, 6.0, 18.0)
        plan["clear_ahead"] = 300.0
    elif B == "crossing traffic":
        density = _u(rng, 0.5, 0.9)
        d_line = _u(rng, 0.5, 3.0) if C == "wait" else _u(rng, 5.0, 12.0)
        plan["clear_ahead"] = 300.0

        def cross(s, e, r, k=k):
            for leg in r.permutation([(k + 1) % 4, (k + 3) % 4])[:int(r.randint(1, 3))]:
                cl = _in_lanes(s.m, int(leg))[0]
                waiting = r.rand() < 0.6
                d = _u(r, 0.5, 2.0) if waiting else _u(r, 5.0, 9.0)
                _clear(s, cl.id, cl.stop["s"] - d - 12.0, cl.stop["s"])
                u = s.add_vehicle(cl.id, _s_for_line(cl, d), v=0.0 if waiting else float(np.sqrt(2 * 1.5 * (d - 2.0))))
                if u is not None and waiting:   # it stopped at its line before the ego got there
                    u.arrival = s.t - _u(r, 0.0, 1.5)
                    u.stop_start, u.zone_t, u.best_stop = u.arrival, u.arrival, 1.0
        plan["post"].append(cross)
    else:   # lead vehicle
        d_line = _u(rng, 18.0, 32.0)
        plan["clear_ahead"] = 300.0
        plan["post"].append(lambda s, e, r, ln=ln: _add_ahead(s, r, ln, e.s, _u(r, 4.0, 10.0), v=e.v * _u(r, 0.8, 1.0)))
    if C == "pull away" or (C == "wait" and d_line < 3.5):
        plan["v"] = 0.0
    plan["s"] = _s_for_line(ln, d_line)
    sim.populate(density, ped_rate=_u(rng, 0.0, 0.08))
    return plan


def _recipe_multilane(sim, rng, tg):
    m = sim.m
    B, C, D, V = tg["B"], tg["C"], tg["D"], tg.get("V")
    pend = tg.get("E") == "pending"
    nb = sorted((m.lanes[i] for i in m.spawn_lanes if m.lanes[i].pts[-1, 1] > m.lanes[i].pts[0, 1]),
                key=lambda l: l.lane_index)
    v0 = 70 / 3.6 * 0.92
    solid = nb[0].solid_right[0]
    plan = dict(cap=False, v=None, clear_ahead=0.0, post=[], turn=None)
    ego_frac, ego_extra = 0.97, {}

    def free_s():   # a stretch of lanes 0 and 1 whose dividing line stays dashed for the manoeuvre
        for _ in range(50):
            s = _u(rng, 30.0, 240.0)
            if s + 140.0 < solid[0] or s - 10.0 > solid[1]:
                return s
        return 30.0
    density = _u(rng, 0.3, 0.8)
    if V == "merge":   # signalling out of a lane that ends; the gap it takes comes after frame 31
        ln = nb[2]
        plan["s"] = ln.length - _u(rng, 150.0, 185.0)
        plan["v"] = v0 * _u(rng, 0.75, 0.9)
        density = _u(rng, 0.6, 0.9)
        plan["clear_ahead"] = 300.0
    elif B == "lane end" and D == "merge":
        ln = nb[2]
        plan["s"] = ln.length - _u(rng, 60.0, 80.0)   # the cone taper is in view from the start
        plan["v"] = v0 * _u(rng, 0.8, 1.0)
        density = _u(rng, 0.3, 0.6)
        plan["clear_ahead"] = 300.0

        ego_extra = dict(style=_u(rng, 0.75, 1.0))

        def merge(s, e, r):   # already signalling at frame 0; which gap it takes is its choice
            _clear(s, nb[1].id, e.s - 50.0, e.s + 40.0)
            t_sig = s.t - _u(r, 0.5, 1.5)
            e.lc = dict(side="left", target=nb[1].id, t_sig=t_sig, mandatory=True, moving=False, lead=1.6,
                        dur=_u(r, 2.5, 5.0))
            e.indicator, e.indicator_on_t = 1, t_sig
            s._event("TURN_ON", 0, -1, 1)
        plan["post"].append(merge)
    elif B == "lane end":
        ln = nb[1]
        plan["s"] = nb[2].length - _u(rng, 60.0, 110.0)
        plan["clear_ahead"] = 60.0

        def merger(s, e, r):
            u = _add_ahead(s, r, nb[2], e.s, _u(r, -4.0, 8.0), v=e.v * _u(r, 0.85, 1.0), kind="car")
            if u is not None:
                _clear(s, nb[2].id, u.s + 1.0, u.s + 40.0)
        plan["post"].append(merger)
    else:
        j = int(rng.randint(2))
        ln = nb[j]
        plan["s"] = free_s()
        if B == "cut-in vehicle" or V == "cutin":
            # a faster car signals into the ego's lane; it moves over 0.6 to 1.2 s ahead of the ego
            plan["clear_ahead"] = 90.0
            plan["v"], ego_frac = 70 / 3.6 * 0.85, 0.85
            src = nb[1 - j]

            def cutin(s, e, r, src=src, ln=ln, pending=V == "cutin"):
                _clear(s, src.id, e.s - 15.0, e.s + 60.0)
                u = _add_ahead(s, r, src, e.s, _u(r, -2.0, 3.0) if pending else _u(r, 2.0, 8.0),
                               v=e.v * (_u(r, 1.05, 1.12) if pending else _u(r, 1.08, 1.18)), kind="car", v0frac=0.96)
                if u is None:
                    return
                side = "right" if src.lane_index < ln.lane_index else "left"
                lc = dict(side=side, target=ln.id, mandatory=False, moving=False, forced=True, cutin=True,
                          lead=_u(r, 1.5, 2.0), dur=_u(r, 2.5, 3.5))
                if pending:   # starts signalling 4.6 to 5.4 s into the clip, so it may move over after frame 31
                    lc.update(t_sig=None, sig_at=_u(r, 4.6, 5.4))
                    u.lc = lc
                else:
                    lc["t_sig"] = s.t - _u(r, 0.0, 1.0)
                    u.lc = lc
                    u.indicator, u.indicator_on_t = (1 if side == "left" else 2), lc["t_sig"]
                    s._event("TURN_ON", u.slot, -1, u.indicator)
            plan["post"].append(cutin)
        elif B == "lead vehicle" and (D == "lane change" or V == "overtake"):
            # a slow vehicle the ego follows within 10 m; pulling out from behind it is its choice
            plan["clear_ahead"] = 150.0
            v3 = sim.cfg.get("DT_V3")
            v_l = 70 / 3.6 * (_u(rng, *sim.cfg["DT_OVERTAKE_V0_FRAC"]) if v3 else _u(rng, 0.18, 0.26))
            kind = ["truck", "bus", "car"][int(rng.choice(3, p=[0.45, 0.35, 0.2]))]
            if v3:   # v3 (D1): slower than the ego, above 60% of the limit; the ego closes in to within 2 s
                ego_frac = _u(rng, *sim.cfg["DT_OVERTAKE_EGO_FRAC"])
            if pend:   # it catches up, slows, and is held up when the clip ends
                plan["v"], gap0, patience = v0 * _u(rng, 0.85, 1.0), _u(rng, 30.0, 45.0), _u(rng, 1.5, 2.5)
                if v3:
                    gap0 = v_l * _u(rng, 1.4, 1.8) + (70 / 3.6 * ego_frac - v_l) * _u(rng, 2.0, 3.5)
            else:      # already behind it for a while, pulls out within the clip
                plan["v"], gap0, patience = v_l * _u(rng, 1.0, 1.1), _u(rng, 5.5, 7.5), _u(rng, 0.8, 1.5)
                if v3:
                    gap0 = v_l * _u(rng, 1.1, 1.6)
            ego_extra = dict(overtaker=True, patience=patience, T=_u(rng, 1.0, 1.2), s0=_u(rng, 1.5, 2.0), lc_quick=not pend)
            if not pend:
                plan["post"].append(lambda s, e, r: setattr(e, "behind_since", s.t - e.patience - _u(r, 0.0, 0.3)))
            tgt = nb[1 - j]
            plan["post"].append(lambda s, e, r, ln=ln, g=gap0, v=v_l, kind=kind: _add_ahead(
                s, r, ln, e.s, g, v=v, v0frac=v / (70 / 3.6), kind=kind))

            def traffic(s, e, r, tgt=tgt, pending=pend):   # in the lane it pulls out into
                _clear(s, tgt.id, e.s - 150.0, e.s + 150.0)
                back = _u(r, 30.0, 70.0) if pending else _u(r, 70.0, 140.0)
                for _ in range(int(r.randint(1, 3)) if pending else int(r.rand() < 0.5)):
                    _add(s, tgt.id, e.s - back, v=70 / 3.6 * _u(r, 0.8, 0.92))
                    back += _u(r, 30.0, 60.0)
            plan["post"].append(traffic)
        elif B == "lead vehicle":   # a slow vehicle close ahead: followed within 10 m
            plan["clear_ahead"] = 120.0
            if sim.cfg.get("DT_V3"):   # v3: a leader at its normal speed (a truck or bus to slow down for)
                kind = ("truck", "bus")[int(rng.randint(2))] if C == "slow down" else ("car", "truck", "bus")[int(rng.randint(3))]
                f_l = _lead_frac(rng, sim.cfg, kind)
                v_l = 70 / 3.6 * f_l
                if C == "slow down":
                    plan["v"], gap0 = v0 * _u(rng, 0.95, 1.0), _u(rng, 40.0, 60.0)
                else:
                    plan["v"], gap0 = v_l * _u(rng, 1.0, 1.05), v_l * _u(rng, 1.0, 1.8)
                ego_frac = 0.97 if C == "slow down" else min(0.97, f_l + _u(rng, 0.05, 0.12))
                ego_extra = dict(T=_u(rng, 1.0, 1.2), s0=_u(rng, 1.5, 2.0))
                plan["post"].append(lambda s, e, r, ln=ln, v=v_l, g=gap0, k=kind, f=f_l: _add_ahead(
                    s, r, ln, e.s, g, v=v, v0frac=f, kind=k))
            else:
                v_l = 70 / 3.6 * (_u(rng, 0.13, 0.2) if C == "follow" else _u(rng, 0.18, 0.3))
                if C == "slow down":
                    plan["v"], gap0 = v0 * _u(rng, 0.8, 1.0), _u(rng, 40.0, 60.0)
                else:
                    plan["v"], gap0 = v_l * _u(rng, 1.0, 1.15), _u(rng, 5.0, 8.0)
                ego_extra = dict(T=_u(rng, 1.0, 1.2), s0=_u(rng, 1.5, 2.0))
                plan["post"].append(lambda s, e, r, ln=ln, v=v_l, g=gap0: _add_ahead(
                    s, r, ln, e.s, g, v=v, v0frac=v / (70 / 3.6), kind=["car", "truck", "bus"][int(r.randint(3))]))
        else:   # none: free driving at the limit on an open stretch
            plan["clear_ahead"] = 300.0
            density = _u(rng, 0.15, 0.35)
    plan["lane"] = ln
    plan["extra"] = dict(v0frac=ego_frac, **ego_extra)
    sim.populate(density, ped_rate=0.0)
    return plan


def _recipe_residential(sim, rng, tg):
    m = sim.m
    B, C, D, V = tg["B"], tg["C"], tg["D"], tg.get("V")
    j = int(rng.randint(2))
    ln = m.lanes[m.spawn_lanes[j]]
    opp = m.lanes[m.spawn_lanes[1 - j]]
    cw0, cw1, cw_id = ln.crosswalks[0]
    plan = dict(lane=ln, cap=False, v=None, clear_ahead=0.0, post=[], turn=None)
    density = _u(rng, 0.3, 0.8)
    if V == "ped":   # someone reaches the curb ahead just after frame 31; crossing is their choice
        t_c, tau = _u(rng, 6.35, 6.9), _u(rng, 1.9, 4.1)
        v_e = min(50 / 3.6 * 0.9, (cw0 - EGO_HALF - 15.0) / (t_c + tau))
        plan["s"] = cw0 - EGO_HALF - v_e * (t_c + tau)
        plan["v"], plan["extra"] = v_e, dict(v0frac=min(0.97, v_e / (50 / 3.6) + 0.02))
        plan["clear_ahead"] = 300.0
        paths = [p for p in m.ped_paths if p["cw"] == cw_id]
        path = paths[int(rng.randint(len(paths)))]

        def ped(s, e, r, path=path, t_c=t_c):
            s.peds = [p for p in s.peds if p.cw != cw_id]
            _add_ped(s, r, path, t_c)
        plan["post"].append(ped)
    elif B == "pedestrian":
        plan["s"] = cw0 - _u(rng, 25.0, 60.0)
        plan["clear_ahead"] = 300.0
        t_c = _u(rng, 0.6, 3.0)
        paths = [p for p in m.ped_paths if p["cw"] == cw_id]
        path = paths[int(rng.randint(len(paths)))]
        plan["post"].append(lambda s, e, r, path=path, t_c=t_c: _add_ped(s, r, path, t_c))
    elif B == "speed limit":
        # front 55 to 70 m before the school-zone sign: the ego starts slowing for the 30 km/h
        # zone around frame 5 and has slowed by more than 3 m/s by frame 31
        z0 = min(z[0] for z in ln.zones)
        plan["s"] = z0 - EGO_HALF - _u(rng, 55.0, 70.0)
        plan["clear_ahead"] = 300.0
    else:
        for _ in range(50):
            s = _u(rng, 20.0, 200.0)
            s_p = s + EGO_HALF + (_u(rng, 6.0, 9.0) if C == "wait" else _u(rng, 15.0, 35.0))
            if (D != "nudge around" and B != "parked obstacle") or min(abs(s_p - cw0), abs(s_p - cw1)) > 18.0:
                break
        plan["s"] = s
        plan["clear_ahead"] = 300.0
        if C == "wait":
            plan["v"] = 0.0
        if D == "nudge around" or B == "parked obstacle":
            wait = B == "parked obstacle" and D != "nudge around"   # waits behind it; a nudge goes round

            def parked(sm, e, r, s_p=s_p, wait=wait, pending=V == "nudge"):
                _clear(sm, ln.id, s_p - 15.0, s_p + 15.0)
                kind = ["car", "suv", "truck"][int(r.choice(3, p=[0.5, 0.2, 0.3]))]
                sm.add_parked(ln.id, s_p, -_u(r, 0.5, 0.95), kind=kind)
                s_o = opp.length - s_p      # the parked car on the oncoming lane
                if pending:   # oncoming cars until about frame 32, then a slower one it may go in front of
                    _clear(sm, opp.id, 0.0, s_o + 10.0)
                    _stream(sm, r, opp, s_o, _u(r, 5.7, 6.6), _u(r, 14.0, 17.0), v=50 / 3.6 * _u(r, 0.75, 0.9),
                            v_gap=_u(r, 5.0, 8.0))
                elif wait:
                    _clear(sm, opp.id, s_o - 150.0, s_o + 10.0)
                    t_arr = (s_p - e.s) / max(e.v, 3.0)
                    d = 13.0 * _u(r, max(t_arr - 0.5, 0.5), t_arr + 1.5)
                    for _ in range(1 + int(r.rand() < 0.3)):
                        _add(sm, opp.id, s_o - d, v=50 / 3.6 * _u(r, 0.75, 0.95))
                        d += _u(r, 15.0, 25.0)
                else:
                    _clear(sm, opp.id, s_o - 160.0, s_o + 15.0)
            plan["post"].append(parked)
        if B == "lead vehicle" and sim.cfg.get("DT_V3"):   # v3: a car at its normal speed, 1.0 to 1.8 s ahead
            f_l = _lead_frac(rng, sim.cfg, "car")
            v_l = 50 / 3.6 * f_l
            plan["v"] = v_l * _u(rng, 1.0, 1.05)
            plan["extra"] = dict(T=_u(rng, 1.0, 1.2), s0=_u(rng, 1.5, 2.0), v0frac=min(0.97, f_l + _u(rng, 0.05, 0.12)))
            plan["post"].append(lambda sm, e, r, v=v_l, f=f_l: _add_ahead(sm, r, ln, e.s, v * _u(r, 1.0, 1.8), v=v,
                                                                       kind="car", v0frac=f))
        elif B == "lead vehicle":   # a slow car close ahead: followed within 10 m
            v_l = 50 / 3.6 * (_u(rng, 0.18, 0.28) if C == "follow" else _u(rng, 0.25, 0.4))
            plan["v"] = v_l * _u(rng, 1.0, 1.15)
            plan["extra"] = dict(T=_u(rng, 1.0, 1.2), s0=_u(rng, 1.5, 2.0))
            plan["post"].append(lambda sm, e, r, v=v_l: _add_ahead(sm, r, ln, e.s, _u(r, 5.0, 8.0), v=v, v0frac=v / (50 / 3.6)))
        elif B == "none" and D != "nudge around":
            plan["v"] = 50 / 3.6 * _u(rng, 0.25, 0.5)
    sim.populate(density, ped_rate=_u(rng, 0.0, 0.08) if B != "pedestrian" else _u(rng, 0.0, 0.05))
    return plan


RECIPES = dict(signal4=_recipe_signal4, stop4=_recipe_stop4, multilane=_recipe_multilane, residential=_recipe_residential)


def _insert_ego_v2(sim, plan, rng):
    cfg, ln, s = sim.cfg, plan["lane"], plan["s"]
    keep = []
    for u in sim.vehicles:
        i, us = u.where()
        if u.route[i] == ln.id or (u.lc is not None and u.lc.get("from") == ln.id):
            if abs(us - s) < u.L / 2 + EGO_HALF + 6.0:
                continue
            if s < us < s + plan["clear_ahead"]:
                continue
        keep.append(u)
    sim.vehicles = keep
    ego = sim.add_vehicle(ln.id, s, kind="car", ego=True, slot=0)
    ego.mobil_thresh = np.inf          # no discretionary lane change but the planned one
    for k, v in plan.get("extra", {}).items():
        setattr(ego, k, v)
    if ln.stop is not None:            # route and indicator decided before frame 0
        ego.turn_w = {t: float(t == plan["turn"]) + 1e-9 for t in "LSR"}
        sim._decide_route(ego, ln, indicate=False)   # the indicator comes on at its turn_dist
        if sim.cfg.get("DT_V3") and len(ln.next) > 1 and plan.get("D") not in ("left turn", "right turn"):
            ego.route_pending, ego.ind_pending = True, False   # v3 (Issue 6): its move is drawn per future
    v = sim._v0(ego) * rng.uniform(0.8, 1.0) if plan["v"] is None else plan["v"]
    index = sim._lane_index()
    lead, gap = sim._leader(ego, index)
    if lead is not None:
        v = min(v, lead.v + 0.5 * max(gap - 3.0, 0.0), max(gap - 2.0, 0.0) / 1.2)
    if ln.stop is not None and plan["cap"]:
        v = min(v, float(np.sqrt(2 * 2.0 * max(ln.stop["s"] - ego.front - 2.0, 0.0))))
    ego.v = max(v, 0.0)
    ego.stopped = ego.v < 0.3
    sim.vehicles = [u for u in sim.vehicles if u is ego or u.route[u.where()[0]] != ln.id
                    or u.where()[1] > s or s - u.where()[1] - (u.L + ego.L) / 2 > 2.0 + 1.2 * u.v]
    if ego.v > 1.0:   # nobody on a crosswalk the new ego could not stop for
        for j in range(len(ego.lanes)):
            for s0, s1, cw in ego.lanes[j].crosswalks:
                d = ego.off[j] + s0 - ego.front
                if -ego.L - (s1 - s0) < d < ego.v ** 2 / (2 * 3.0) + 3.0:
                    sim.peds = [p for p in sim.peds if not (p.cw == cw and p.state in (P_CROSS, P_WAIT))]
    for fn in plan["post"]:
        fn(sim, ego, rng)
    lead, gap = sim._leader(ego, sim._lane_index())
    if lead is not None:   # a leader added just now: keep clear of R6 as above
        ego.v = max(0.0, min(ego.v, lead.v + 0.5 * max(gap - 3.0, 0.0), max(gap - 2.0, 0.0) / 1.2))
        ego.stopped = ego.v < 0.3
    return ego


def _score(lab, tg):
    return 4 * (lab["B"] == tg["B"]) + 2 * (lab["C"] == tg["C"]) + (lab["D"] == tg["D"])


def _attempt(rng, cfg, tg):
    tmpl = tg["template"]
    opts = {}
    leg = int(rng.randint(4))
    if tmpl == "signal4":
        opts["protected"] = True if tg["A"] == "arrow" else (False if tg["A"] == "permissive left" else None)
    elif tmpl == "stop4" and tg["B"] == "pedestrian":
        opts["force_cw"] = (leg,)
    m = build_map(tmpl, rng, cfg, **opts)
    sim = Sim(m, cfg, rng)
    sim.next_slot = 1                  # slot 0 stays the ego's
    sim.ego_leg = leg
    plan = RECIPES[tmpl](sim, rng, tg)
    plan.setdefault("D", tg["D"])
    while sim.frame < -1:              # warm-up without the ego
        sim.step_frame()
    _insert_ego_v2(sim, plan, rng)
    while sim.frame < cfg["DT_BRANCH_FRAME"]:
        sim.step_frame()
    branches = []
    for b in range(cfg["DT_BRANCHES"]):
        br = copy.deepcopy(sim, memo={id(m): m})
        br.rng = np.random.RandomState(rng.randint(2 ** 31 - 1))
        while br.frame < cfg["DT_CUE_FRAME"]:
            br.step_frame()
        branches.append(br)
    return m, branches


def route_draws(moves, K, rng):
    """v3 (Issue 6, D2): the moves of K futures, drawn independently with equal odds."""
    return [moves[int(i)] for i in rng.randint(len(moves), size=K)]


def _route_futures(br, seed, attempt, b, K):
    """v3: the moves of the K futures of a branch whose ego has not begun its move by the fork, or None."""
    ego = br.ego
    if ego is None or not ego.route_pending or ego.entered:
        return None
    i, _ = ego.where()
    moves = [br.m.lanes[x].turn for x in ego.lanes[i].next]
    return route_draws(moves, K, np.random.RandomState(_fork_seed(seed, attempt, b, 99)))


def _fork_seed(seed, attempt, b, k):
    return (int(seed) * 1_000_003 + attempt * 7_919 + b * 104_729 + k * 31) % (2 ** 31 - 1)


def _future(pk, T0):
    """What a future set keeps of a packed record: frames T0+1 on, and the choices made in them."""
    ag, ex, dec = pk["agents"][T0 + 1:], pk["extra"][T0 + 1:], pk["decisions"]
    fu = dict(ego=pk["ego"][T0 + 1:], agents=ag[..., [1, 2, 3, 4, 9]].astype(np.float16),
                lamps=np.packbits(pk["lamps"][T0 + 1:].astype(bool), axis=-1),
                signals=pk["signals"][T0 + 1:, :, :2].astype(np.int8),
                rule_state=ex[:, 0, 3].astype(np.int8), yield_to=ex[:, 0, 4].astype(np.int8),
                decisions=dec[dec[:, 0] > T0])
    if "may_go" in pk:   # v3 (Issue 7)
        fu["may_go"], fu["may_go_reason"] = pk["may_go"][T0 + 1:], pk["may_go_reason"][T0 + 1:]
    return fu


def simulate(seed, cfg, template=None, target=None):
    """Returns dict(map, template, branches=[record...], forks=[[future...] per branch], tries, ok,
    labels=[per branch], target). Each branch gets DT_FORKS futures forked at DT_CUE_FRAME with
    fresh draws; future 0 is its own continuation. An attempt is chosen on the labels of frames 0
    to 31 (A to D), on E (when the ego's choices fall, not how they come out) and on checks run
    over all of its futures together, never on how the branch's own future turned out."""
    if target is None:
        return simulate_v1(seed, cfg, template)
    rng = np.random.RandomState(seed)
    T0 = cfg["DT_CUE_FRAME"]
    total = cfg["FRAMES"] + cfg["FRAMES_FUTURE"]
    K = int(cfg.get("DT_FORKS", 1))
    pend = target.get("E") == "pending"   # its response to the light comes after frame 31
    best, best_key = None, -1
    for attempt in range(cfg["DT_TRIES"]):
        m, brs = _attempt(rng, cfg, target)
        pks = [_pack(br) for br in brs]
        labs = [clip_labels(pk, m, cfg) for pk in pks]
        if not all(l["A"] == target["A"] and (l["a_ok"] or pend) and l["D"] is not None for l in labs):
            continue
        if cfg.get("DT_OVERTAKE_CHECK") and any(l["D"] == "lane change" and overtake_problems(pk, T0 + 1, cfg)
                                                for l, pk in zip(labs, pks)):
            continue   # an overtake must start from following its leader within 10 m
        sc = _score(labs[0], target)
        if sc + 2 <= best_key:
            continue
        fin = [copy.deepcopy(br, memo={id(m): m}) for br in brs]
        routes = [_route_futures(br, seed, attempt, b, K) for b, br in enumerate(brs)]
        for br, rt in zip(fin, routes):
            if rt is not None:
                br.set_ego_route(rt[0])
        for br in fin:
            while br.frame < total - 1:
                br.step_frame()
        recs = [_pack(br) for br in fin]
        for l, r in zip(labs, recs):
            l["E"] = end_label(r["decisions"], T0, T0 + 9)
        sc += 2 * (labs[0]["E"] == target.get("E", labs[0]["E"]))
        if sc <= best_key or not all(not r["ran_out"] and not cue_problems(r, m, cfg) for r in recs):
            continue
        forks, full, ok = [], [], True
        for b, br in enumerate(brs):
            fb, fr = [_future(recs[b], T0)], [recs[b]]
            for k in range(1, K):
                fk = copy.deepcopy(br, memo={id(m): m})
                fk.rng = np.random.RandomState(_fork_seed(seed, attempt, b, k))
                if routes[b] is not None:
                    fk.set_ego_route(routes[b][k])
                while fk.frame < total - 1:
                    fk.step_frame()
                pk = _pack(fk)
                if pk["ran_out"] or cue_problems(pk, m, cfg):
                    ok = False
                    break
                fb.append(_future(pk, T0))
                if cfg.get("DT_KEEP_FORK_RECS"):   # full records of the futures, for visualisation
                    fr.append(pk)
            if not ok:
                break
            forks.append(fb)
            full.append(fr)
        if ok:
            best, best_key = (m, fin, recs, labs, attempt, forks, full, routes), sc
        if best_key == 9 or (attempt >= 9 and best_key >= 8) or (attempt >= 19 and best_key >= 6):
            break
    if best is None:
        return dict(map=None, ok=False, tries=cfg["DT_TRIES"], target=target)
    m, fin, recs, labs, attempt, forks, full, routes = best
    return dict(map=m, template=target["template"], branches=recs, forks=forks, fork_recs=full, tries=attempt + 1, ok=True, labels=labs,
                route_draws=routes,
                target=target, colors={u.slot: u.color for u in fin[0].vehicles + fin[0].peds + fin[0].gone})


# ---------------------------------------------------------------------------
# v1 sampler (docs/DRIVETOY.md §5), kept so v1 clips re-simulate unchanged
# ---------------------------------------------------------------------------

def _plan_ego(sim, rng):
    """Choose the ego's lane and where it will be when the clip starts (DT_EGO_*). The ego
    itself is inserted only at the end of the warm-up (``_insert_ego``): placed earlier it
    would make its own route decision — 30 m before the stop line — before the branch frame,
    and the branches of a context would all show the same ego trajectory."""
    m, cfg = sim.m, sim.cfg
    t = m.template
    plan = dict(extra={})
    if t in ("signal4", "stop4"):
        legs = sorted({m.lanes[i].leg for i in m.spawn_lanes})
        leg = legs[int(rng.randint(len(legs)))]
        lanes = sorted((m.lanes[i] for i in m.spawn_lanes if m.lanes[i].leg == leg), key=lambda l: l.lane_index)
        ln = lanes[0] if (len(lanes) == 1 or rng.rand() < cfg["DT_EGO_INNER_P"]) else lanes[-1]
        plan["s"] = ln.stop["s"] - rng.uniform(*cfg["DT_EGO_LINE_DIST"][t])
        plan["extra"]["turn_w"] = dict(cfg["DT_EGO_TURN"])
        plan["cw"] = leg if any(c["id"] == leg for c in m.crosswalks) else None
    elif t == "multilane":
        nb = sorted((m.lanes[i] for i in m.spawn_lanes if m.lanes[i].pts[-1, 1] > m.lanes[i].pts[0, 1]),
                    key=lambda l: l.lane_index)
        if rng.rand() < cfg["DT_EGO_DROP_P"]:
            ln = nb[-1]
            plan["s"] = ln.length - rng.uniform(125.0, 200.0)   # mandatory merge signals 120 m before the end
        else:
            ln = nb[int(rng.randint(len(nb) - 1))]
            plan["s"] = rng.uniform(120.0, 260.0)
        plan["extra"]["v0frac"] = 0.97
        plan["cw"] = None
    else:
        ln = m.lanes[m.spawn_lanes[int(rng.randint(2))]]
        plan["s"] = ln.crosswalks[0][0] - rng.uniform(*cfg["DT_EGO_CW_DIST"])
        plan["cw"] = 0
    plan["lane"] = ln
    sim.next_slot = max(sim.next_slot, 1)     # slot 0 stays the ego's
    return plan


def _insert_ego(sim, plan, rng):
    """Put the ego into the warmed-up traffic at slot 0, with a safe speed, and give the
    crosswalk it is approaching pedestrians that reach the curb just after the branch frame."""
    cfg, ln, s = sim.cfg, plan["lane"], plan["s"]
    keep = []
    for u in sim.vehicles:
        if u.route[u.where()[0]] == ln.id or (u.lc is not None and u.lc.get("from") == ln.id):
            us = u.where()[1]
            if abs(us - s) < u.L / 2 + 2.25 + 6.0:
                continue                                  # no room: make room
            if s < us < s + 60.0 and rng.rand() < cfg["DT_EGO_CLEAR_P"]:
                continue                                  # thin the queue ahead
        keep.append(u)
    sim.vehicles = keep
    ego = sim.add_vehicle(ln.id, s, kind="car", ego=True, slot=0)
    for k, v in plan["extra"].items():
        setattr(ego, k, v)
    v = sim._v0(ego) * rng.uniform(0.8, 1.0)
    index = sim._lane_index()
    lead, gap = sim._leader(ego, index)
    if lead is not None:   # keep well clear of R6 (gap >= max(1 m, 0.5 s * v)) and closing slowly
        v = min(v, lead.v + 0.5 * max(gap - 3.0, 0.0), max(gap - 2.0, 0.0) / 1.2)
    if ln.stop is not None:
        v = min(v, float(np.sqrt(2 * 2.0 * max(ln.stop["s"] - ego.front - 2.0, 0.0))))
    ego.v = v
    ego.stopped = v < 0.3
    # nobody may be following the new ego too closely either
    sim.vehicles = [u for u in sim.vehicles if u is ego or u.route[u.where()[0]] != ln.id
                    or u.where()[1] > s or s - u.where()[1] - (u.L + ego.L) / 2 > 2.0 + 1.2 * u.v]
    if plan["cw"] is not None and rng.rand() < cfg["DT_EGO_PED_P"]:
        paths = [p for p in sim.m.ped_paths if p["cw"] == plan["cw"]]
        for path in [paths[i] for i in rng.permutation(len(paths))[:1 + int(rng.rand() < 0.4)]]:
            ped = sim.add_ped(path)
            if ped is not None:
                L = ped.cum[-1]
                ped.s = max(0.0, L - ped.speed * rng.uniform(1.4, 3.6))   # curb reached 1.2-3.4 s into the clip
                ped.heading = ped.path_heading()
    return ego


def _bias_signal(sim, rng, ego_lane):
    """signal4: start the controller so the ego's own light does something within the clip."""
    sig, cfg = sim.signals, sim.cfg
    if sig is None:
        return
    own = ego_lane.stop["group"]
    other = [g for g in sig.groups if g != own][0]
    names = list(cfg["DT_EGO_SIGNAL"])
    w = np.array([cfg["DT_EGO_SIGNAL"][n] for n in names], float)
    pick = names[int(rng.choice(len(names), p=w / w.sum()))]
    lo, hi = cfg["SIG_GREEN_WALK"]
    if pick == "own_green":
        sig.set_phase(own, "G", rng.uniform(0, lo), sim.t)
    elif pick == "green_soon":
        k = rng.randint(3)
        if k == 0:     # the cross street is flashing: its yellow comes within the warm-up
            el = rng.uniform(lo, hi)
            sig.set_phase(other, "G", el, sim.t, walk=el - rng.uniform(0, cfg["SIG_FLASH"]))
        elif k == 1:
            sig.set_phase(other, "Y", rng.uniform(0, cfg["SIG_YELLOW"]), sim.t)
        else:
            sig.set_phase(other, "AR", rng.uniform(0, cfg["SIG_ALL_RED"]), sim.t)


def _in_view(ego, pos, cfg):
    d = np.asarray(pos[:2]) - ego[:2]
    dist = np.hypot(*d)
    if dist < 1e-3:
        return True
    ang = np.angle(np.exp(1j * (np.arctan2(d[1], d[0]) - ego[2])))
    return dist <= cfg["DT_VIEW_RANGE"] and abs(ang) <= np.deg2rad(cfg["DT_VIEW_HALF_ANGLE"])


def count_view_events(rec, m, cfg):
    lo, hi = cfg["DT_BRANCH_FRAME"] + 1, cfg["FRAMES"] - 1
    heads = [h["pos"] for h in m.signal_heads if h["kind"] == "veh"]
    n = 0
    for f, typ, a, b, val in rec["events"]:
        f, typ, a = int(f), int(typ), int(a)
        if not (lo <= f <= hi) or typ not in DECISION_EVENTS:
            continue
        ego = rec["ego"][f]
        if typ == EV["SIGNAL"]:
            if any(_in_view(ego, p, cfg) for p in heads):
                n += 1
        elif a >= 0 and rec["agents"][f, a, 9] > 0 and _in_view(ego, rec["agents"][f, a, 1:3], cfg):
            n += 1
    return n


def _pack(sim):
    fr = sim.frames
    events = [e for e in sim.events if e[0] >= 0]
    pk = dict(
        agents=np.stack([f["agents"] for f in fr]), lamps=np.stack([f["lamps"] for f in fr]),
        extra=np.stack([f["extra"] for f in fr]), signals=np.stack([f["signals"] for f in fr]),
        lc=np.stack([f["lc"] for f in fr]), ego=np.stack([f["ego"] for f in fr]),
        events=np.array(events, np.float32).reshape(-1, 5), ran_out=sim.ego_ran_out,
        decisions=np.array(sim.decisions, np.float32).reshape(-1, len(DEC_COLS)), drivers=_drivers(sim),
        routes=_routes(sim))
    if fr and "may_go" in fr[0]:   # v3 (Issue 7)
        mg = np.stack([f["may_go"] for f in fr])
        pk["may_go"], pk["may_go_reason"] = mg[:, 0], mg[:, 1]
    return pk


def _drivers(sim):
    """Per-slot driver values (choices.DRIVER_COLS; pedestrians: style only) and, in the last
    column, whether a pedestrian was cued at DT_CUE_FRAME."""
    out = np.zeros((sim.max_agents, len(DRIVER_COLS) + 1), np.float32)
    for u in sim.vehicles + sim.peds + sim.gone:
        if hasattr(u, "T"):
            out[u.slot, :len(DRIVER_COLS)] = [getattr(u, k) for k in DRIVER_COLS]
        else:
            out[u.slot, 0], out[u.slot, -1] = u.style, float(u.cued)
    return out


def _routes(sim):
    """Final route (lane ids, -1 padded) of every vehicle still on the map at the end."""
    r = np.full((sim.max_agents, 4), -1, np.int32)
    for u in sim.vehicles:
        r[u.slot, :len(u.route[-4:])] = u.route[-4:]
    return r


def simulate_v1(seed, cfg, template=None):
    """Returns dict(map, template, branches=[record...], tries)."""
    rng = np.random.RandomState(seed)
    names = list(cfg["DT_TEMPLATES"])
    w = np.array([cfg["DT_TEMPLATES"][n] for n in names], float)
    total = cfg["FRAMES"] + cfg["FRAMES_FUTURE"]
    for attempt in range(cfg["DT_RESAMPLE_TRIES"]):
        tmpl = template or names[int(rng.choice(len(names), p=w / w.sum()))]
        m = build_map(tmpl, rng, cfg)
        sim = Sim(m, cfg, rng)
        plan = _plan_ego(sim, rng)
        density = rng.uniform(*cfg["DT_STOP4_DENSITY"]) if tmpl == "stop4" else rng.uniform(0.3, 1.0)
        sim.populate(density, ped_rate=rng.uniform(0.05, 0.25) if m.ped_paths else 0.0)
        _bias_signal(sim, rng, plan["lane"])
        while sim.frame < -1:                      # warm-up without the ego
            sim.step_frame()
        _insert_ego(sim, plan, rng)
        while sim.frame < cfg["DT_BRANCH_FRAME"]:
            sim.step_frame()
        branches = []
        for b in range(cfg["DT_BRANCHES"]):
            br = copy.deepcopy(sim, memo={id(m): m})
            br.rng = np.random.RandomState(rng.randint(2 ** 31 - 1))
            while br.frame < total - 1:
                br.step_frame()
            branches.append(_pack(br))
        ok = all(not r["ran_out"] and count_view_events(r, m, cfg) >= cfg["DT_MIN_EVENTS"] for r in branches)
        if ok or attempt == cfg["DT_RESAMPLE_TRIES"] - 1:
            return dict(map=m, template=tmpl, branches=branches, tries=attempt + 1, ok=ok,
                        colors={u.slot: u.color for u in sim.vehicles + sim.peds})
