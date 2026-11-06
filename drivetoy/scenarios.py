"""Scene helpers for the decision scenes (decision.py): placing the ego and timed cars in a simulation,
clearing lanes, forking futures and packing a simulation's record for the labels.
"""
import numpy as np

from .choices import DEC_COLS, DRIVER_COLS
from .sim import P_CROSS, P_WAIT, VEH_KINDS

# the simulator's scene variants, listed in meta.json ("variants")
VARIANTS = ["left_gap", "rtor", "yellow", "near_tie", "merge", "cutin", "overtake", "ped", "nudge", "nudge_clear"]

EGO_HALF = VEH_KINDS["car"][1] / 2
EGO_TURNS = ["none", "left", "straight", "right", "wait"]   # the label ego_turn
STILL_M = 1.0                # the ego "waits" if it covers less than this from frame 31 to the clip's end


def moved(ego, t0=31):
    """Metres the ego covers from frame t0 on."""
    d = np.diff(np.asarray(ego)[t0:, :2], axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


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


def _timed_cars(sim, rng, ln, s_conf, times, route=None):
    """Cars on lane ``ln`` whose fronts reach arclength ``s_conf`` at the (time, speed) pairs given."""
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
        u.v0frac = max(u.v0frac, sim.cfg["DT_V0_FRAC"][0])   # a normal desired speed; a zone it drives through slows it
        u.T, u.s0 = _u(rng, 1.0, 1.3), _u(rng, 1.5, 2.0)
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


def _insert_ego(sim, plan, rng):
    ln, s = plan["lane"], plan["s"]
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
        if len(ln.next) > 1 and plan.get("D") not in ("left turn", "right turn") and plan.get("draw_route", True):
            ego.route_pending, ego.ind_pending = True, False   # its move is drawn per future
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


def route_draws(moves, K, rng):
    """The moves of K futures, drawn independently with equal odds."""
    return [moves[int(i)] for i in rng.randint(len(moves), size=K)]


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
    if "may_go" in pk:
        fu["may_go"], fu["may_go_reason"] = pk["may_go"][T0 + 1:], pk["may_go_reason"][T0 + 1:]
        fu["ego_turn"] = pk["ego_turn"]
    return fu


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
    if fr and "may_go" in fr[0]:
        mg = np.stack([f["may_go"] for f in fr])
        pk["may_go"], pk["may_go_reason"] = mg[:, 0], mg[:, 1]
        pk["ego_turn"] = np.int8({"L": 1, "S": 2, "R": 3}.get(getattr(sim.ego, "turn", None), 0))   # its final move
        if len(fr) > 40 and moved(pk["ego"]) < STILL_M:
            pk["ego_turn"] = np.int8(EGO_TURNS.index("wait"))
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
