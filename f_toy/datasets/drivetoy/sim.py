"""DriveToy traffic simulator — the rulebook of docs/DRIVETOY.md §2 as agent
policies on a ``roads.RoadMap``.

Vehicles follow IDM along a route of lanes; every rule is an extra IDM
obstacle (a virtual stopped car at the stop line, at the crosswalk, ...),
and the binding obstacle is recorded as the vehicle's ``rule_state``.
Pedestrians walk polylines and decide at the curb.

Branching (§5): hidden variables — a vehicle's turn, a pedestrian's intent
and start delay, the green's length, a driver's politeness, new arrivals —
are drawn from ``self.rng`` at the moment they are first needed, never
earlier. ``simulate`` runs one shared past up to ``DT_BRANCH_FRAME``, deep-
copies the simulator and swaps in a fresh RNG per branch, so every branch is
byte-identical up to that frame and differs only afterwards.
"""
import copy

import numpy as np

from .choices import CHOICE_KINDS, biased, cautious_gap, choice_p, driver_values, style_c50
from .roads import LANE_W, build_map

# kind -> (type code, length, width, height, weight)
VEH_KINDS = dict(car=(1, 4.5, 1.8, 1.5, 0.55), suv=(2, 4.8, 1.9, 1.8, 0.25),
                 truck=(3, 7.5, 2.4, 3.0, 0.12), bus=(4, 11.0, 2.5, 3.2, 0.08))
PED_KINDS = dict(adult=(5, 0.5, 0.5, 1.75, 0.8), child=(6, 0.4, 0.4, 1.2, 0.2))
TYPE_NAMES = ["none", "car", "suv", "truck", "bus", "ped_adult", "ped_child"]

RULE_STATES = ["GO", "FOLLOW", "STOP_SIGNAL", "STOP_SIGN", "YIELD_ONCOMING", "YIELD_PED",
               "YIELD_CONFLICT", "MERGE", "SLOW", "PED_WALK", "PED_WAIT"]
RS = {n: i for i, n in enumerate(RULE_STATES)}
# v3 (Issue 7): why an ego stopped at its line could not enter now (index 0: it could, or no line)
MAY_GO_REASONS = ["", "light", "oncoming", "pedestrian", "conflict", "stop_order"]

EVENTS = ["SIGNAL", "TURN_ON", "LC_START", "STOP", "GO", "YIELD", "ENTER", "PED_DECIDE", "PED_CROSS",
          "LET_IN", "INDICATOR_OFF", "VIOLATE", "WAVE", "CANCEL"]

# Violation sets (docs/DRIVETOY.md §9): DT_VIOLATION switches one rule off for
# every agent after DT_BRANCH_FRAME, so the context stays lawful. Each illegal
# act is logged as a VIOLATE event whose value indexes VIOLATION_RULES — the
# ground truth the rule checker is scored against.
VIOLATIONS = dict(red="R1", left_gap="R3", no_stop="R4", ped="R5", tailgate="R6", solid="R7", no_signal="R7",
                  speed="R0", jaywalk="P1", ped_dart="P2")
VIOLATION_RULES = ["R0", "R1", "R3", "R4", "R5", "R6", "R7", "P1", "P2"]
EV = {n: i for i, n in enumerate(EVENTS)}

# signal codes
BALL_R, BALL_Y, BALL_G = 0, 1, 2
PED_DONT, PED_WALK, PED_FLASH = 0, 1, 2
# pedestrian states / intents
P_APPROACH, P_WAIT, P_CROSS, P_EXIT = 1, 2, 3, 4
I_UNKNOWN, I_CROSS, I_WAIT, I_PASS = 0, 1, 2, 3

TURN_WEIGHTS = {"L": 0.3, "S": 0.45, "R": 0.25}


def _u(rng, r):
    return float(rng.uniform(r[0], r[1]))


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

class Signals:
    """Two-group (NS / EW) controller: [arrow] -> green (walk + flash) ->
    yellow -> all-red, per group. The walk part of each green is the hidden
    variable; it is drawn when the green reaches its minimum length."""

    def __init__(self, m, cfg, rng, t0):
        self.cfg = cfg
        self.groups = list(m.groups)
        self.seq = []
        for g in self.groups:
            if m.protected_left:
                self.seq.append((g, "A"))
            self.seq += [(g, "G"), (g, "Y"), (g, "AR")]
        self.i = int(rng.randint(len(self.seq)))
        g, ph = self.seq[self.i]
        self.walk = None
        self.arrow_dur = None
        self.queue = []       # v2: planned [(phase, duration)] of the phases to come
        self.frozen = None    # v2: time at which the controller is held (cued signal timing)
        if ph == "G":
            self.start = t0 - rng.uniform(0, cfg["SIG_GREEN_WALK"][0])
        elif ph == "A":
            self.arrow_dur = _u(rng, cfg["SIG_ARROW"])
            self.start = t0 - rng.uniform(0, self.arrow_dur)
        else:
            self.start = t0 - rng.uniform(0, self._fixed(ph))

    def set_phase(self, group, phase, elapsed, t0, walk=None, rng=None):
        """Start the controller in ``phase`` of ``group``, ``elapsed`` seconds in (scene setup)."""
        self.i = self.seq.index((group, phase))
        self.start = t0 - elapsed
        self.walk = walk
        if phase == "A":
            self.arrow_dur = elapsed + _u(rng, (0.5, 2.0))

    def schedule(self, group, phase, t_start, t0, dur):
        """v2: make (group, phase) start at time t_start. The cycle is walked back to the
        warm-up start t0; dur(phase) gives the length of every planned phase (a green's
        includes its flashing don't-walk). The target phase is planned too."""
        j = self.seq.index((group, phase))
        chain = [(self.seq[j], dur(self.seq[j]))]
        start = t_start
        while start > t0:
            j = (j - 1) % len(self.seq)
            d = dur(self.seq[j])
            chain.insert(0, (self.seq[j], d))
            start -= d
        (g, ph), d0 = chain[0]
        self.i, self.start = self.seq.index((g, ph)), start
        self.walk = d0 - self.cfg["SIG_FLASH"] if ph == "G" else None
        if ph == "A":
            self.arrow_dur = d0
        self.queue = chain[1:]

    def unfreeze(self, t):
        self.start += t - self.frozen
        self.frozen = None

    def _fixed(self, ph):
        return self.cfg["SIG_YELLOW"] if ph == "Y" else self.cfg["SIG_ALL_RED"]

    def end(self):
        g, ph = self.seq[self.i]
        if ph == "G":
            return None if self.walk is None else self.start + self.walk + self.cfg["SIG_FLASH"]
        if ph == "A":
            return self.start + self.arrow_dur
        return self.start + self._fixed(ph)

    def update(self, t, rng):
        """Advance to time t; returns True when the phase changed."""
        if self.frozen is not None:
            return False
        changed = False
        while True:
            g, ph = self.seq[self.i]
            if ph == "G" and self.walk is None and t >= self.start + self.cfg["SIG_GREEN_WALK"][0]:
                lo, hi = self.cfg["SIG_GREEN_WALK"]
                self.walk = lo + float(rng.uniform(0, hi - lo))
            e = self.end()
            if e is None or t < e:
                return changed
            self.i = (self.i + 1) % len(self.seq)
            self.start = e
            self.walk = None
            if self.queue and self.queue[0][0] == self.seq[self.i]:
                d = self.queue.pop(0)[1]
                if self.seq[self.i][1] == "G":
                    self.walk = d - self.cfg["SIG_FLASH"]
                elif self.seq[self.i][1] == "A":
                    self.arrow_dur = d
            elif self.seq[self.i][1] == "A":
                self.arrow_dur = _u(rng, self.cfg["SIG_ARROW"])
            changed = True

    def state(self, group, t):
        """(ball, arrow_green, ped, countdown) for a vehicle group / its parallel peds."""
        if self.frozen is not None:
            t = min(t, self.frozen)
        g, ph = self.seq[self.i]
        if group != g:
            return BALL_R, False, PED_DONT, -1
        if ph == "A":
            return BALL_R, True, PED_DONT, -1
        if ph == "Y":
            return BALL_Y, False, PED_DONT, -1
        if ph == "AR":
            return BALL_R, False, PED_DONT, -1
        if self.walk is None or t < self.start + self.walk:
            return BALL_G, False, PED_WALK, -1
        cd = int(np.ceil(self.start + self.walk + self.cfg["SIG_FLASH"] - t - 1e-9))
        return BALL_G, False, PED_FLASH, cd


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

class Vehicle:
    def __init__(self, slot, kind, rng, cfg, route, s, v, ego=False):
        code, L, W, H, _ = VEH_KINDS[kind]
        self.slot, self.kind, self.type = slot, kind, code
        self.L, self.W, self.H = L, W, H
        self.ego = ego
        self.route = list(route)
        self.s, self.v, self.acc = s, v, 0.0
        self.style = float(rng.rand())         # cautious 0 to assertive 1 (choices.py)
        dv = driver_values(rng, cfg, self.style, heavy=code >= 3)
        self.T, self.s0, self.b = dv["T"], dv["s0"], dv["b"]
        self.a = dv["a"] * (0.7 if code >= 3 else 1.0)
        self.v0frac, self.a_lat, self.turn_dist = dv["v0frac"], dv["a_lat"], dv["turn_dist"]
        self.d_pref, self.d_turn, self.drift, self.stop_dist = dv["d_pref"], dv["d_turn"], dv["drift"], dv["stop_dist"]
        self.creep = bool(rng.rand() < self.style)   # creeps toward the line while waiting at a stop sign
        self.patience = biased(rng, cfg["DT_PATIENCE"], self.style, cfg["DT_STYLE_CORR"], up=False)
        self.choices = {}          # kind -> (key, outcome) of its latest major choice
        self.go_at = None          # start-up delay: when a stopped vehicle may move off
        self.commit = None         # v3: the "go" choice it committed to (left_gap, rtor, stop_wave)
        self.route_pending = False # v3: the ego's move is drawn per future at the fork (Issue 6)
        self.late_route = False    # v3: its move was drawn at the fork; the indicator follows turn_dist only
        self.rtor_stop, self.rtor_ready, self.rtor_on = 0.0, False, False   # right turn on red
        self.behind_since = None   # overtaker: since when it is held up behind a slow leader
        self.ind_pending = False   # turn decided, indicator not on yet (it comes on at turn_dist)
        self.blink_phase = int(rng.randint(cfg["L2_BLINK_FRAMES"]))
        self.color = int(rng.randint(12))
        self.route_open = True     # route beyond the current lane not decided yet
        self.turn = None
        self.indicator = 0         # 0 off, 1 left, 2 right
        self.indicator_on_t = None
        self.hazard = False
        self.stop_time = 0.0
        self.stop_start = None
        self.arrival = None        # stop-sign arrival time
        self.granted = False       # allowed through the stop line (latched)
        self.entered = False
        self.enter_time = None
        self.yellow = None         # 'go' | 'stop' decision for the current yellow
        self.yellow_tag = None
        self.lc = None
        self.d = 0.0               # lateral offset from the route's lane centre (+ left)
        self.let_in = False
        self.polite = None
        self.next_mobil = 0.0
        self.state = RS["GO"]
        self.yield_to = -1
        self.alive = True
        self.stopped = v < 0.3
        self.parked = False        # v2: double-parked, hazard lamps on, never moves
        self.nudge = None          # v2: planned sideways detour around a parked vehicle
        self.dslope = 0.0

    def set_route(self, m, route):
        self.route = list(route)
        self.lanes = [m.lanes[i] for i in self.route]
        self.off = np.concatenate([[0.0], np.cumsum([l.length for l in self.lanes])])

    def where(self, s=None):
        s = self.s if s is None else s
        i = int(np.clip(np.searchsorted(self.off, s, side="right") - 1, 0, len(self.lanes) - 1))
        return i, s - self.off[i]

    def v0f(self):
        """Desired fraction of the limit, wandering slowly along the route (capped below the limit)."""
        amp, per, ph = self.drift[3:]
        return min(self.v0frac * (1.0 + amp * np.sin(2 * np.pi * self.s / per + ph)), max(self.v0frac, 0.99))

    @property
    def front(self):
        return self.s + self.L / 2

    @property
    def rear(self):
        return self.s - self.L / 2

    def route_pos(self, s):
        """Point on the route at arclength s (extrapolated straight past either end)."""
        if s < 0:
            ln = self.lanes[0]
            h = ln.head[0]
            return ln.pts[0] + s * np.array([np.cos(h), np.sin(h)])
        if s > self.off[-1]:
            ln = self.lanes[-1]
            h = ln.head[-1]
            return ln.pts[-1] + (s - self.off[-1]) * np.array([np.cos(h), np.sin(h)])
        i, sl = self.where(s)
        return self.lanes[i].pos(sl)

    def pose(self):
        """Centre and heading of the body with both axles on the path (no
        rear swing-out in turns), shifted sideways by the lane-change offset."""
        e = 0.35 * self.L
        pf, pr = self.route_pos(self.s + e), self.route_pos(self.s - e)
        p = (pf + pr) / 2
        h = float(np.arctan2(pf[1] - pr[1], pf[0] - pr[0]))
        lat = self.d
        if not self.parked:   # its own line in the lane: a preferred offset, a slow drift, tight or wide turns
            amp, per, ph = self.drift[:3]
            lat += self.d_pref + amp * np.sin(2 * np.pi * self.s / per + ph)
            i, sl = self.where()
            ln = self.lanes[i]
            if ln.kind == "conn" and ln.turn in ("L", "R"):
                lat += self.d_turn * np.sin(np.pi * min(max(sl / max(ln.length, 1.0), 0.0), 1.0))
        if lat:
            p = p + lat * np.array([-np.sin(h), np.cos(h)])
        if self.dslope:
            h = h + np.arctan(self.dslope)
        if self.lc is not None and self.lc.get("moving"):
            tau = self.lc["tau"]
            h = h + np.arctan(-self.lc["d0"] * (6 * tau - 6 * tau ** 2) / self.lc["D"])
        return p, h


class Pedestrian:
    def __init__(self, slot, kind, rng, cfg, path, s=0.0):
        code, L, W, H, _ = PED_KINDS[kind]
        self.slot, self.kind, self.type = slot, kind, code
        self.L, self.W, self.H = L, W, H
        self.path = path            # m.ped_paths entry
        self.pts = np.asarray(path["approach"], float)
        self._seg()
        self.s = s
        self.speed = _u(rng, cfg["PED_SPEED"]) * (0.85 if code == 6 else 1.0)
        self.v = self.speed
        self.state = P_APPROACH
        self.intent = I_UNKNOWN
        self.delay_until = None
        self.heading = 0.0
        self.color = int(rng.randint(12))
        self.alive = True
        self.cw = path["cw"]
        self.cross_start = None
        self.style = float(rng.rand())
        self.choices = {}
        self.cued = False          # visible at the curb by DT_CUE_FRAME: may still decide after it

    def _seg(self):
        d = np.diff(self.pts, axis=0)
        self.cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(d, axis=1))])

    def set_path(self, pts):
        self.pts = np.asarray(pts, float)
        self._seg()
        self.s = 0.0

    def pos(self):
        s = np.clip(self.s, 0, self.cum[-1])
        return np.array([np.interp(s, self.cum, self.pts[:, 0]), np.interp(s, self.cum, self.pts[:, 1])])

    def path_heading(self):
        i = int(np.clip(np.searchsorted(self.cum, self.s, side="right") - 1, 0, len(self.pts) - 2))
        d = self.pts[i + 1] - self.pts[i]
        return float(np.arctan2(d[1], d[0]))


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class Sim:
    def __init__(self, m, cfg, rng):
        self.m, self.cfg, self.rng = m, cfg, rng
        self.fps = cfg["DT_FPS"]
        self.sub = cfg["DT_SUBSTEPS"]
        self.dt = 1.0 / self.fps / self.sub
        self.frame = -cfg["DT_WARMUP"]
        self.t = self.frame / self.fps
        self.max_agents = cfg["DT_MAX_AGENTS"]
        self.next_slot = 0
        self.vehicles, self.peds = [], []
        self.events = []
        self.frames = []
        self.signals = Signals(m, cfg, rng, self.t) if m.groups else None
        self.ego = None
        self.ego_ran_out = False
        self.inflow = {}
        self.ped_inflow = 0.0
        self.gap_lanes = {}        # v3: lane id -> mean of its exponential headway tail (Issue 3)
        self.stream_mu = {}        # v3: the same for lanes fed by a recipe's stream only
        self.next_arrival = {}     # v3: next spawn time on a gap lane
        self.violation = cfg.get("DT_VIOLATION")
        self.viol_from = cfg["DT_BRANCH_FRAME"] + 1
        self.decisions = []        # [frame, slot, kind, cue, p, outcome, ego_rel] (choices.DEC_COLS)
        self.gone = []             # agents that left the map (their per-driver values are still labelled)

    def _viol(self, kind):
        return self.violation == kind and self.frame >= self.viol_from

    def _log_violation(self, a, rule, b=-1):
        self.events.append((self.frame + 1, EV["VIOLATE"], a, b, float(VIOLATION_RULES.index(rule))))

    def _driver_violation(self, veh):
        """Per-driver parameters of the 'tailgate' and 'speed' sets."""
        if self._viol("tailgate") and not getattr(veh, "_tailgating", False):
            veh.T, veh.s0, veh._tailgating = 0.3, 0.5, True
        if self._viol("speed") and veh.v0frac < 1.2:
            veh.v0frac = 1.3

    # ----- major choices (choices.py) ------------------------------------
    def decide(self, u, kind, cue, key=None, other=None):
        """Agent u's choice of ``kind``: True (the assertive outcome) with p = sigmoid of the cue.
        Logged in ``self.decisions``; with ``key`` the outcome holds while the same key comes back
        (one choice per gap). ``other`` is the agent bounding the gap, if any."""
        memo = u.choices.get(kind)
        if key is not None and memo is not None and memo[0] == key:
            return memo[1]
        c50, w = style_c50(self.cfg, kind, u.style)
        p = choice_p(float(cue), c50, w)
        out = bool(self.rng.rand() < p)
        self.decisions.append((self.frame + 1, u.slot, CHOICE_KINDS.index(kind), float(cue), p, out,
                               self._ego_rel(u, kind, other)))
        u.choices[kind] = (key, out)
        return out

    def _ego_rel(self, u, kind, other=None):
        """Whether a choice concerns the ego: its own, one about a gap it bounds, its leader's, a
        pedestrian at a crosswalk ahead of it, or a lane change into its lane near it."""
        ego = self.ego
        if ego is None:
            return False
        if u is ego or other is ego:
            return True
        if kind == "ped_cross":
            return self._ego_will_pass(u.cw)
        if ego.yield_to == u.slot:
            return True
        i, sl = ego.where()
        if u.lc is not None and u.lc.get("target") == ego.route[i]:
            ui, usl = u.where()
            return abs(usl - sl) < 40.0
        lead, gap = self._leader(ego, self._lane_index())
        return lead is u and gap < 40.0

    def _in_view(self, pos):
        """Inside the ego camera's view range (the diversity check's definition)."""
        ep, eh = self.ego.pose()
        d = np.asarray(pos[:2]) - ep
        dist = float(np.hypot(*d))
        if dist < 1e-3:
            return True
        ang = np.angle(np.exp(1j * (np.arctan2(d[1], d[0]) - eh)))
        return dist <= self.cfg["DT_VIEW_RANGE"] and abs(ang) <= np.deg2rad(self.cfg["DT_VIEW_HALF_ANGLE"])

    def _event_value(self, veh, key, up=False):
        """A per-event value (README, "Minor variation"), leaning to the driver's style."""
        return biased(self.rng, self.cfg[key], veh.style, self.cfg["DT_STYLE_CORR"], up=up)

    # ----- setup -------------------------------------------------------
    def _slot(self):
        if self.next_slot >= self.max_agents:
            return None
        self.next_slot += 1
        return self.next_slot - 1

    def _kind(self, rng, kinds):
        names = list(kinds)
        w = np.array([kinds[n][4] for n in names])
        return names[int(rng.choice(len(names), p=w / w.sum()))]

    def add_vehicle(self, lane_id, s, v=None, kind=None, ego=False, slot=None):
        slot = self._slot() if slot is None else slot
        if slot is None:
            return None
        rng = self.rng
        kind = kind or self._kind(rng, VEH_KINDS)
        veh = Vehicle(slot, kind, rng, self.cfg, [lane_id], s, 0.0, ego=ego)
        veh.set_route(self.m, [lane_id])
        if v is None:
            v = self._v0(veh) * rng.uniform(0.7, 1.0)
        veh.v = v
        veh.stopped = v < 0.3
        self._driver_violation(veh)
        self.vehicles.append(veh)
        if ego:
            self.ego = veh
        return veh

    def add_parked(self, lane_id, s, d, kind=None):
        """v2: a double-parked vehicle at arclength s of a lane, d metres left of its centre."""
        veh = self.add_vehicle(lane_id, s, v=0.0, kind=kind)
        if veh is not None:
            veh.parked, veh.hazard, veh.d, veh.stopped = True, True, float(d), True
        return veh

    def add_ped(self, path, s=0.0, kind=None):
        slot = self._slot()
        if slot is None:
            return None
        kind = kind or self._kind(self.rng, PED_KINDS)
        p = Pedestrian(slot, kind, self.rng, self.cfg, path, s)
        p.heading = p.path_heading()
        self.peds.append(p)
        return p

    def populate(self, density, ped_rate):
        """Initial traffic on every spawn lane plus Poisson inflow rates."""
        rng = self.rng
        for lid in self.m.spawn_lanes:
            ln = self.m.lanes[lid]
            end = ln.length - (8.0 if ln.stop else 3.0)
            if self.cfg.get("DT_V3") and lid not in self.gap_lanes:
                # v3: the first car within one usual gap of the lane start, so a sparse lane keeps its share
                s = 5.0 + rng.rand() * rng.uniform(12, 70) / max(density, 0.15) / ln.inflow_w
            else:
                s = rng.uniform(5, 30)
            while s < end:
                if not self._blocked(lid, s, 4.0):
                    veh = self.add_vehicle(lid, s)
                    if veh is None:
                        break
                    if ln.stop is not None:   # able to stop at the line if it has to
                        d = max(ln.stop["s"] - veh.front - 2.0, 0.0)
                        veh.v = min(veh.v, np.sqrt(2 * 2.0 * d))
                    if ln.closed_end:         # ... or before the end of a closing lane (no RNG draw:
                        d = max(ln.length - veh.front - 3.0, 0.0)   # other scenes stay reproducible)
                        veh.v = min(veh.v, np.sqrt(2 * 2.0 * d))
                    if lid in self.gap_lanes:
                        veh.T = max(veh.T, self.cfg["DT_ONCOMING_MIN"])
                if lid in self.gap_lanes:     # v3: spaced by the oncoming headway law
                    s += ln.limit * 0.85 * self.headway(lid)
                else:
                    s += rng.uniform(12, 70) / max(density, 0.15) / ln.inflow_w
            # front to back: a follower must be able to stop behind its leader even if the leader
            # brakes as hard as it can (no RNG draw: scenes where this does not bind stay unchanged)
            on_lane = sorted((u for u in self.vehicles if u.route[0] == lid), key=lambda u: -u.s)
            for lead, u in zip(on_lane, on_lane[1:]):
                gap = lead.s - u.s - (lead.L + u.L) / 2 - 1.0
                u.v = min(u.v, float(np.sqrt(lead.v ** 2 + 2 * self.cfg["DT_B_MAX"] * max(gap, 0.0))))
            self.inflow[lid] = density * rng.uniform(0.08, 0.18) * ln.inflow_w   # vehicles / s
        self.ped_inflow = ped_rate
        for path in self.m.ped_paths:
            if rng.rand() < min(1.0, ped_rate * 6):
                self.add_ped(path, s=rng.uniform(0, _pathlen(path["approach"])))

    def set_ego_route(self, turn):
        """v3 (Issue 6): give the ego, still before its line, the move ``turn`` drawn for this future."""
        veh = self.ego
        i, _ = veh.where()
        ln = veh.lanes[i]
        c = next(self.m.lanes[x] for x in ln.next if self.m.lanes[x].turn == turn)
        veh.set_route(self.m, veh.route[:veh.route.index(ln.id) + 1] + [c.id] + c.next)
        veh.turn = c.turn
        veh.route_pending, veh.late_route = False, True
        veh.ind_pending = c.turn in ("L", "R")

    def headway(self, lane_id):
        """v3 (Issue 3): an oncoming headway, DT_ONCOMING_MIN plus an exponential tail whose mean is
        the lane's (``gap_lanes``) or one drawn once for the lane from DT_ONCOMING_MEAN."""
        mu = self.gap_lanes.get(lane_id)
        if mu is None:
            mu = self.stream_mu.get(lane_id)
            if mu is None:
                mu = self.stream_mu[lane_id] = _u(self.rng, self.cfg["DT_ONCOMING_MEAN"])
        return self.cfg["DT_ONCOMING_MIN"] + float(self.rng.exponential(mu))

    def _blocked(self, lid, s, margin):
        for u in self.vehicles:
            if u.route[0] == lid and abs(u.s - s) < (u.L + margin + 5.0):
                return True
        return False

    # ----- helpers -------------------------------------------------------
    def _v0(self, veh):
        """Desired speed: limit x v0frac, anticipating curves and lower-limit zones ahead."""
        i, sl = veh.where()
        a_lat, acc, frac = veh.a_lat, 1.5, veh.v0f()
        best = np.inf
        start = -sl - veh.L / 2          # route distance from the front to the start of lane j
        for j in range(i, len(veh.lanes)):
            ln = veh.lanes[j]
            dist = start + ln.s
            mask = (dist >= -veh.L) & (dist <= 80.0)
            if mask.any():
                d = dist[mask]
                v_curve = np.sqrt(a_lat / np.maximum(ln.curv[mask], 1e-4))
                v_lim = ln.limits()[mask] * frac
                # reach a lower limit 8 m before its sign, so IDM's lag never shows as speeding
                best = min(best, float(np.min(np.sqrt(v_curve ** 2 + 2 * acc * np.maximum(d, 0.0)))),
                           float(np.min(np.sqrt(v_lim ** 2 + 2 * acc * np.maximum(d - 8.0, 0.0)))))
            start += ln.length
            if start > 80.0:
                break
        if veh.route_pending and veh.lanes[i].kind == "in":
            # v3 (D4): an ego whose move is still open slows for the tightest move its lane allows
            ln_i = veh.lanes[i]
            base = -sl - veh.L / 2 + ln_i.length
            for c in (self.m.lanes[x] for x in ln_i.next):
                dist = base + c.s
                mask = (dist >= -veh.L) & (dist <= 80.0)
                if mask.any():
                    v_curve = np.sqrt(a_lat / np.maximum(c.curv[mask], 1e-4))
                    best = min(best, float(np.min(np.sqrt(v_curve ** 2 + 2 * acc * np.maximum(dist[mask], 0.0)))))
        return best

    def _idm(self, veh, v0, gap, vl):
        v = veh.v
        ss = veh.s0 + max(0.0, v * veh.T + v * (v - vl) / (2 * np.sqrt(veh.a * veh.b)))
        return veh.a * (1 - (v / max(v0, 0.1)) ** 4 - (ss / max(gap, 0.1)) ** 2)

    def _free(self, veh, v0):
        return veh.a * (1 - (veh.v / max(v0, 0.1)) ** 4)

    def _lane_index(self):
        """lane id -> [(veh, s_local)] including both lanes of a moving lane change."""
        idx = {}
        for u in self.vehicles:
            i, sl = u.where()
            idx.setdefault(u.route[i], []).append((u, sl))
            ln = u.lanes[i]
            if ln.kind == "conn" and sl - u.L / 2 < 12.0:
                # still in the mouth of the junction: also a leader for whoever follows on the
                # incoming lane, and for whoever is on a sibling connector (they overlap there)
                prev = self.m.lanes[ln.prev]
                idx.setdefault(prev.id, []).append((u, prev.length + sl))
                for sib in prev.next:
                    if sib != ln.id:
                        idx.setdefault(sib, []).append((u, sl))
            if u.lc is not None and u.lc.get("moving"):
                idx.setdefault(u.lc["from"], []).append((u, sl))
            # junction occupancy, keyed ("occ", connector): vehicles on the connector, and those
            # whose front is already over the stop line while the body is still on the incoming
            # lane (a bus is 11 m long) — at a negative arclength along that connector
            if ln.kind == "conn":
                idx.setdefault(("occ", ln.id), []).append((u, sl))
            elif ln.kind == "in" and u.entered and not u.route_open and i + 1 < len(u.lanes):
                idx.setdefault(("occ", u.route[i + 1]), []).append((u, sl - ln.length))
        return idx

    def _leader(self, veh, index):
        """Nearest vehicle ahead along ``veh``'s route: (vehicle, bumper gap) or (None, inf)."""
        i, sl = veh.where()
        base = -sl
        for j in range(i, len(veh.route)):
            best, bgap = None, np.inf
            for u, us in index.get(veh.route[j], ()):
                if u is veh or (j == i and us <= sl) or (veh.nudge is not None and u.slot == veh.nudge["slot"]):
                    continue
                gap = base + us - (veh.L + u.L) / 2
                if gap < bgap:
                    best, bgap = u, gap
            if best is not None:
                return best, bgap
            base += veh.lanes[j].length
            if base > 120.0:
                break
        return None, np.inf

    def _follower(self, lane_id, s_local, index, exclude=()):
        best, bgap = None, np.inf
        for u, us in index.get(lane_id, ()):
            if u in exclude or us >= s_local:
                continue
            if s_local - us < bgap:
                best, bgap = u, s_local - us
        return best, bgap

    def _decide_route(self, veh, ln, indicate=True):
        moves = [self.m.lanes[c] for c in ln.next]
        if veh.L > 7.0 and len(moves) > 1:   # trucks and buses would off-track over the corner
            moves = [c for c in moves if c.turn != "R"]
        weights = getattr(veh, "turn_w", None) or TURN_WEIGHTS
        w = np.array([weights[c.turn] for c in moves])
        c = moves[int(self.rng.choice(len(moves), p=w / w.sum()))]
        veh.set_route(self.m, veh.route[:veh.route.index(ln.id) + 1] + [c.id] + c.next)
        veh.route_open = False
        veh.turn = c.turn
        if c.turn in ("L", "R") and not indicate:
            veh.ind_pending = True
        elif c.turn in ("L", "R"):
            veh.indicator = 1 if c.turn == "L" else 2
            veh.indicator_on_t = self.t
            self._event("TURN_ON", veh.slot, -1, veh.indicator)

    def _event(self, name, a, b=-1, value=0.0):
        self.events.append((self.frame + 1, EV[name], a, b, float(value)))

    # ----- per-step policy ------------------------------------------------
    def _vehicle_acc(self, veh, index, ped_block, ped_entering):
        cfg = self.cfg
        if veh.parked:
            return 0.0, RS["GO"], -1
        i, sl = veh.where()
        ln = veh.lanes[i]
        v0 = self._v0(veh)
        cands = [(self._free(veh, v0), RS["GO"] if v0 >= ln.limit_at(sl) * veh.v0f() - 0.3 else RS["SLOW"], -1)]
        if veh.v > v0 + 0.05:   # IDM converges slowly from above; track the target firmly
            cands.append((-2.0 * (veh.v - v0), RS["SLOW"], -1))
        lead, gap = self._leader(veh, index)
        if lead is not None and lead.parked and ln.kind == "road" and veh.nudge is None:
            if self._plan_nudge(veh, lead, gap):
                lead, gap = self._leader(veh, index)
            elif gap < 45.0:   # wait well behind it, so the detour has room to start
                gap = gap - 5.0
        if lead is not None:
            cands.append((self._idm(veh, v0, gap, lead.v), RS["FOLLOW"], lead.slot))
        if veh.lc is not None and veh.lc.get("moving") and veh.lc["tau"] < 0.6:
            # half-way over: the old lane's leader still matters
            best = None
            for u, us in index.get(veh.lc["from"], ()):
                if u is not veh and us > sl and (best is None or us < best[1]):
                    best = (u, us)
            if best is not None:
                g = best[1] - sl - (veh.L + best[0].L) / 2
                cands.append((self._idm(veh, v0, g, best[0].v), RS["FOLLOW"], best[0].slot))

        # ---- junction approach ----
        if ln.kind == "in" and ln.stop is not None and not veh.entered:
            to_line = ln.stop["s"] - (sl + veh.L / 2)
            if veh.route_open and to_line <= max(veh.turn_dist, 2.5 * veh.v):
                self._decide_route(veh, ln)
                i, sl = veh.where()
            if veh.ind_pending and (to_line <= max(veh.turn_dist, 2.5 * veh.v)
                                    or (self.frame >= cfg["DT_CUE_FRAME"] - 1 and not veh.late_route)):
                veh.ind_pending = False   # the turn shows by the last visible frame at the latest
                veh.indicator, veh.indicator_on_t = (1 if veh.turn == "L" else 2), self.t
                self._event("TURN_ON", veh.slot, -1, veh.indicator)
            d_line = veh.off[i] + ln.stop["s"] - veh.front
            if not veh.route_open:
                ok, why, other = self._may_enter(veh, ln, d_line, index, ped_entering, ped_block)
                if veh.route_pending and veh.granted:   # v3: let through before the fork, its move is final
                    veh.route_pending = False
                    veh.ind_pending = veh.turn in ("L", "R")
                if not ok:
                    margin = veh.stop_dist - veh.s0          # it comes to rest stop_dist before the line
                    if why == RS["STOP_SIGN"] and veh.creep and veh.arrival is not None:
                        margin = 0.5 - veh.s0                # after its full stop it creeps up to the line
                    cands.append((self._idm_stop(veh, v0, d_line - margin), why, other))
                if ln.stop["control"] == "signal" and veh.turn == "L" and not veh.granted \
                        and not self.signals.state(ln.stop["group"], self.t)[1]:
                    # a permissive left approaches slowly enough to still yield at the line
                    v_cap = float(np.sqrt(4.0 ** 2 + 2 * 1.5 * max(d_line - 1.0, 0.0)))
                    cands.append((self._free(veh, min(v0, v_cap)), RS["SLOW"], -1))
                    if veh.v > v_cap:
                        cands.append((-2.0 * (veh.v - v_cap), RS["SLOW"], -1))
        # ---- inside / after the line: crosswalks and conflict zones ahead ----
        for j in range(i, min(i + 2, len(veh.lanes))):
            lj = veh.lanes[j]
            for s0, s1, cw in lj.crosswalks:
                d_cw = veh.off[j] + s0 - veh.front
                if d_cw < -0.5 or d_cw > 40.0:
                    continue
                if veh.off[j] + s1 < veh.rear:
                    continue
                if (cw in ped_block or cw in ped_entering) and not self._viol("ped"):
                    if veh.v ** 2 / (2 * max(d_cw - 0.5, 0.05)) <= cfg["DT_B_MAX"] and d_cw > 0:
                        cands.append((self._idm_stop(veh, v0, d_cw - 1.0), RS["YIELD_PED"],
                                      (ped_block.get(cw) or ped_entering.get(cw))))
            if lj.kind == "conn" and veh.entered:
                for (a, b), (za, zb) in self.m.conflicts.items():
                    if a != lj.id:
                        continue
                    d_z = veh.off[j] + za - veh.front
                    if d_z < 0 or d_z > 25:
                        continue
                    for u, us in index.get(("occ", b), ()):
                        zb0, zb1 = self.m.conflicts[(b, a)]
                        # priority to whoever entered the junction first, until it has cleared the zone
                        if us - u.L / 2 <= zb1 + 0.5 and u.enter_time is not None \
                                and u.enter_time < veh.enter_time and \
                                veh.v ** 2 / (2 * max(d_z - 0.5, 0.05)) <= cfg["DT_B_MAX"]:
                            cands.append((self._idm_stop(veh, v0, d_z - 1.0), RS["YIELD_CONFLICT"], u.slot))
        # ---- lane drop: stop before the end if the merge has not happened ----
        if ln.closed_end:
            d_end = veh.off[i] + ln.length - veh.front
            cands.append((self._idm_stop(veh, v0, d_end - 2.0), RS["MERGE"], -1))
        # zipper: let one signalling merger in
        if not veh.let_in and ln.kind == "road":
            m = self._merger_for(veh, ln, sl, index)
            if m is not None:
                mi, msl = m.where()
                gap = msl - sl - (veh.L + m.L) / 2
                cands.append((self._idm(veh, v0, gap, m.v), RS["MERGE"], m.slot))

        acc, why, other = min(cands, key=lambda c: c[0])
        return float(np.clip(acc, -cfg["DT_B_MAX"], veh.a)), why, other

    def _idm_stop(self, veh, v0, gap):
        return self._idm(veh, v0, gap, 0.0)

    def _plan_nudge(self, veh, p, gap):
        """v2: a short detour around the double-parked ``p`` ahead, back into the same lane.
        Where it crosses the centre line it waits for a gap in the oncoming traffic."""
        req = p.d + (p.W + veh.W) / 2 + 0.45          # left offset that clears it by 0.45 m
        if req <= 0.0 or req > 2.4 or gap > 40.0:
            return False
        s_c = veh.s + veh.L / 2 + gap + p.L / 2        # its centre, on veh's route
        if req > LANE_W / 2 - veh.W / 2:
            pp, ph = p.pose()
            fwd = np.array([np.cos(ph), np.sin(ph)])
            t_need = (gap + p.L + veh.L + 12.0) / max(veh.v, 2.5) + 2.0
            spare, who = np.inf, None
            for u in self.vehicles:
                if u is veh or u.parked:
                    continue
                up, uh = u.pose()
                if np.cos(uh - ph) > -0.5:
                    continue
                rel = up - pp
                ahead, lat = float(rel @ fwd), float(rel @ np.array([-fwd[1], fwd[0]]))
                if abs(lat) > 6.0 or ahead < -(p.L / 2 + u.L / 2 + 1.0):
                    continue
                s_u = (ahead - p.L / 2 - u.L / 2) / max(u.v, 0.5) - t_need
                if s_u < spare:
                    spare, who = s_u, u
            # going round in front of oncoming traffic is a choice whose odds rise with the spare time
            if who is not None and (spare < 0.0 or (spare < 10.0 and not self.decide(veh, "nudge", spare, key=who.slot, other=who))):
                return False
        s1 = s_c - p.L / 2 - veh.L / 2 - 0.5           # fully out when its front is 0.5 m short of p
        s0 = max(veh.s, s1 - max(8.0, 1.2 * veh.v))
        s2 = s_c + p.L / 2 + veh.L / 2 + 0.5           # back once its rear is 0.5 m past p
        veh.nudge = dict(slot=p.slot, s=(s0, max(s1, s0 + 3.0), s2, s2 + max(8.0, 1.2 * veh.v)), req=req)
        return True

    @staticmethod
    def _nudge_profile(nd, s):
        """(offset, d offset / ds) of a nudge at route arclength s: smoothstep out, hold, back."""
        s0, s1, s2, s3 = nd["s"]
        if s <= s0 or s >= s3:
            return 0.0, 0.0
        if s < s1:
            u, k, sgn = (s - s0) / (s1 - s0), 1.0 / (s1 - s0), 1.0
        elif s <= s2:
            return nd["req"], 0.0
        else:
            u, k, sgn = 1.0 - (s - s2) / (s3 - s2), 1.0 / (s3 - s2), -1.0
        return nd["req"] * (3 * u ** 2 - 2 * u ** 3), sgn * nd["req"] * (6 * u - 6 * u ** 2) * k

    def _merger_for(self, veh, ln, sl, index):
        """A mandatory merger signalling into ``ln`` from its right neighbour,
        alongside or ahead of ``veh`` — zipper: each vehicle lets one in."""
        if ln.right is None:
            return None
        best = None
        for u, us in index.get(ln.right, ()):
            if u.lc is None or u.lc.get("moving") or u.lc["target"] != ln.id or not u.lc.get("mandatory"):
                continue
            if us + u.L / 2 >= sl + veh.L / 2 - 2.0 and us - sl < 30.0:
                if best is None or us < best[1]:
                    best = (u, us)
        return None if best is None else best[0]

    def _may_enter(self, veh, ln, d_line, index, ped_entering, ped_block=None):
        """Whether ``veh`` may pass its stop line now: (ok, rule_state, other)."""
        cfg = self.cfg
        conn = veh.lanes[veh.route.index(ln.id) + 1]
        can_stop = veh.v ** 2 / (2 * max(d_line - 0.5, 0.05)) <= cfg["R1_STOP_DECEL"]
        if ln.stop["control"] == "signal":
            ball, arrow, _, _ = self.signals.state(ln.stop["group"], self.t)
            if ball == BALL_Y:
                tag = self.signals.start
                if veh.yellow_tag != tag:   # onset: the driver chooses after its reaction time
                    veh.yellow_tag = tag
                    d_on = d_line + veh.v * max(self.t - tag, 0.0)   # where it was when the light turned
                    veh.yellow_need = veh.v ** 2 / (2 * max(d_on - 0.5, 0.05))
                    veh.yellow = "react"
                    veh.yellow_at = self.t + self._event_value(veh, "DT_YELLOW_REACT")
                if veh.yellow == "react" and self.t >= veh.yellow_at:
                    veh.yellow = self._yellow_choice(veh, d_line)
                    if veh.yellow == "stop":
                        veh.granted = False
                elif veh.yellow == "go" and not veh.granted and (can_stop or not self._reaches_line(veh, d_line)) \
                        and veh.v ** 2 / (2 * max(d_line - 0.5, 0.05)) <= cfg["DT_B_MAX"]:
                    veh.yellow = "stop"   # slowed down before the line (e.g. behind a car): it stays
                if veh.yellow == "stop":
                    if not self._viol("red"):
                        return False, RS["STOP_SIGNAL"], -1
            elif ball == BALL_R and not (arrow and conn.turn == "L"):
                if not self._viol("red") and not self._rtor_go(veh, conn):
                    veh.granted = False
                    return False, RS["STOP_SIGNAL"], -1
        if veh.granted:
            return True, RS["GO"], -1
        if ln.stop["control"] == "signal":
            if conn.turn == "L" and not arrow:
                ok, other = self._oncoming_clear(veh, ln, conn, index)
                if not ok:
                    if not self._viol("left_gap"):
                        return False, RS["YIELD_ONCOMING"], other
        else:   # all-way stop
            first = self._stop_order_first() if veh.arrival is not None else None
            if veh.arrival is None or first is not veh:
                if not self._viol("no_stop"):
                    if first is None and cfg.get("DT_V2"):
                        # v2: still arriving: name the vehicle on another approach that already
                        # waits with precedence, so yielding to crossing traffic is recorded
                        leg = ln.leg
                        w = [u for u in self.vehicles if u.arrival is not None and not u.entered
                             and u.lanes[u.where()[0]].leg != leg]
                        first = min(w, key=lambda u: u.arrival) if w else None
                    return False, RS["STOP_SIGN"], -1 if first is None else first.slot
        if not self._viol("ped"):
            for s0, s1, cw in conn.crosswalks:
                if cw in ped_entering:
                    return False, RS["YIELD_PED"], ped_entering[cw]
        if veh.route_pending:   # v3 (D4): it enters only if every move its lane allows could go now
            for c in (self.m.lanes[x] for x in ln.next):
                for s0, s1, cw in c.crosswalks:
                    p = (ped_block or {}).get(cw, ped_entering.get(cw))
                    if p is not None:
                        return False, RS["YIELD_PED"], p
                if c.turn == "L" and ln.stop["control"] == "signal" and not self.signals.state(ln.stop["group"], self.t)[1]:
                    gap, who, inside = self._oncoming_gap(ln, c)
                    if inside is not None or (who is not None and gap < self.cfg["DT_GAP_FLOOR"]):
                        return False, RS["YIELD_ONCOMING"], (inside or who).slot
        if self.cfg.get("DT_V3") and veh.commit in ("left_gap", "rtor"):
            occ = self._conflict_occupied(veh, conn)
            if occ >= 0:
                self._cancel(veh, occ)
                return False, RS["YIELD_CONFLICT"], occ
            blocker = None
        else:
            blocker = None if (self._viol("left_gap") and conn.turn == "L") else self._conflict_blocker(veh, conn, index)
        if blocker is not None:
            return False, RS["YIELD_CONFLICT"], blocker.slot
        out = self.m.lanes[conn.next[0]]
        for u, us in index.get(out.id, ()):
            if us - u.L / 2 < veh.L + 2.0:
                return False, RS["YIELD_CONFLICT"], u.slot
        # latch only once the vehicle can no longer stop comfortably (or is at the line);
        # a permissive left keeps checking the oncoming traffic until it could not stop at all
        if conn.turn == "L" and ln.stop["control"] == "signal" and not arrow:
            if veh.v ** 2 / (2 * max(d_line - 0.5, 0.05)) > 0.8 * cfg["DT_B_MAX"]:
                veh.granted = True
        elif not can_stop or d_line < 3.0:
            veh.granted = True
        return True, RS["GO"], -1

    def _oncoming_gap(self, ln, conn):
        """R3's scan of oncoming through and right traffic: (time of the nearest one to the conflict
        with ``conn``, that vehicle, None), or (inf, None, u) for a vehicle u already on its connector
        short of the conflict's end. Draws nothing."""
        ko = (ln.leg + 2) % 4
        best, who = np.inf, None
        for u in self.vehicles:
            ui, usl = u.where()
            ul = u.lanes[ui]
            if ul.leg != ko or ul.kind not in ("in", "conn"):
                continue
            uconn = u.lanes[ui] if ul.kind == "conn" else (u.lanes[ui + 1] if not u.route_open else None)
            if uconn is not None and uconn.turn == "L":
                continue
            if uconn is None and all(self.m.lanes[c].turn == "L" for c in ul.next):
                continue
            if ul.kind == "conn":
                z = self.m.conflicts.get((uconn.id, conn.id))
                if z is not None and usl - u.L / 2 <= z[1] + 0.5:
                    return np.inf, None, u
                continue
            if u.yellow == "stop" and self.signals.state(ul.stop["group"], self.t)[0] == BALL_Y:
                continue
            if self.signals.state(ul.stop["group"], self.t)[0] == BALL_R:
                continue
            z = self.m.conflicts.get((uconn.id, conn.id)) if uconn is not None else None
            dist = (ul.stop["s"] - (usl + u.L / 2)) + (z[0] if z is not None else 6.0)
            tta = dist / max(u.v, 1.0)
            if tta < min(best, 12.0):
                best, who = tta, u
        return best, who, None

    def _oncoming_clear(self, veh, ln, conn, index):
        """R3: oncoming through or right traffic. Nobody inside the conflict and nobody within
        DT_GAP_FLOOR s of it; then taking the gap is a choice whose odds rise with the nearest
        oncoming vehicle's time to the conflict (one choice per gap, keyed by that vehicle)."""
        if self.cfg.get("DT_V3") and veh.commit == "left_gap":
            occ = self._conflict_occupied(veh, conn)
            if occ < 0:   # ... or an oncoming car arrives within R3's check while the turner is short of its line
                tau, who, inside = self._oncoming_gap(ln, conn)
                if inside is not None or (who is not None and tau < self.cfg["R3_GAP_CHECK"] + 0.3):
                    occ = (inside or who).slot
            if occ < 0:
                return True, -1
            self._cancel(veh, occ)
            return False, occ
        best, who, inside = self._oncoming_gap(ln, conn)
        if inside is not None:
            return False, inside.slot
        if who is None:
            return True, -1
        if self.cfg.get("DT_V3"):
            # v3: the gap is judged at the line, as the time the oncoming car still needs once the
            # turner gets there, so a committed turner crosses its line with that gap left (R3)
            i = veh.route.index(ln.id)
            d = max(veh.off[i] + ln.stop["s"] - veh.front, 0.0)
            t_reach = d / veh.v if veh.v >= 1.0 else \
                float(np.mean(self.cfg["DT_START_DELAY"])) + float(np.sqrt(2 * d / max(veh.a, 0.5)))
            best -= t_reach
            if best >= self.cfg["DT_GAP_FLOOR"] and d > 3.5:
                return True, -1   # still approaching: the choice is made at the line
        if best < self.cfg["DT_GAP_FLOOR"]:
            return False, who.slot
        go = self.decide(veh, "left_gap", best, key=who.slot, other=who)
        if go and self.cfg.get("DT_V3"):
            veh.commit = "left_gap"
        return go, who.slot

    def _conflict_occupied(self, veh, conn):
        """v3: the gap has truly closed: slot of a vehicle whose body lies inside a conflict area of
        ``conn``, or that can no longer stop before it at R1_STOP_DECEL; -1 if there is none. This
        is the one thing that cancels a committed go."""
        for u in self.vehicles:
            if u is veh or u.parked:
                continue
            ui, usl = u.where()
            ul = u.lanes[ui]
            if ul.kind == "conn":
                uc, us = ul, usl
            elif ul.kind == "in" and not u.route_open and ui + 1 < len(u.lanes) and u.lanes[ui + 1].kind == "conn":
                uc, us = u.lanes[ui + 1], usl - ul.length   # still on its approach: the connector ahead
            else:
                continue
            z = self.m.conflicts.get((uc.id, conn.id))
            if z is None or us - u.L / 2 > z[1]:
                continue
            ahead = z[0] - (us + u.L / 2)
            if ahead <= 0.0 or u.v ** 2 / (2 * max(ahead - 0.5, 0.05)) > self.cfg["R1_STOP_DECEL"]:
                return u.slot
        return -1

    def _cancel(self, veh, other):
        """v3: drop a committed go (the gap has closed); the next gap is a new choice."""
        veh.choices.pop(veh.commit, None)
        veh.commit = None
        self._event("CANCEL", veh.slot, other)

    # ----- stochastic choices at lines, gaps and crosswalks ------------------
    def _yellow_choice(self, veh, d_line):
        """Stop or go at a yellow, after the reaction time. The cue is the braking a stop needed at
        onset: below DT_YELLOW_BAND it stops, above it goes; in between it chooses, as long as it
        still reaches the line before red and could still stop without braking hard."""
        cfg = self.cfg
        lo, hi = cfg["DT_YELLOW_BAND"]
        need, need_now = veh.yellow_need, veh.v ** 2 / (2 * max(d_line - 0.5, 0.05))
        end = self.signals.end()
        left = (end - self.t) if end is not None else cfg["SIG_YELLOW"]
        reach = d_line <= max(veh.v, 0.1) * (left - 0.2)
        if need <= lo or (not reach and need_now <= cfg["DT_B_MAX"]):
            return "stop"
        if need >= hi or need_now > 0.85 * cfg["DT_B_MAX"] or not reach:
            return "go"
        return "go" if self.decide(veh, "yellow", need, key=veh.yellow_tag) else "stop"

    def _reaches_line(self, veh, d_line):
        """Whether veh reaches its line, at its speed, before the yellow ends."""
        end = self.signals.end()
        left = (end - self.t) if end is not None else self.cfg["SIG_YELLOW"]
        return d_line <= max(veh.v, 0.1) * (left - 0.2)

    def _rtor_go(self, veh, conn):
        """Right turn on red (DT_RTOR): after a full stop in the stop zone and with no cross
        traffic within DT_GAP_FLOOR s of the conflict, going is a choice whose odds rise with the gap."""
        if not self.cfg.get("DT_RTOR") or conn.turn != "R":
            return False
        if self.cfg.get("DT_V3") and veh.commit == "rtor":
            occ = self._conflict_occupied(veh, conn)
            if occ < 0:
                return True
            self._cancel(veh, occ)
            veh.rtor_on = False
            return False
        if veh.rtor_on and veh.granted:
            return True
        if not veh.rtor_ready:
            return False
        tta, who = self._cross_tta(veh, conn)
        if tta < self.cfg["DT_GAP_FLOOR"]:
            veh.rtor_on = False
        elif who is None or tta >= 12.0:
            veh.rtor_on = True
        else:
            veh.rtor_on = self.decide(veh, "rtor", tta, key=who.slot, other=who)
            if veh.rtor_on and self.cfg.get("DT_V3"):
                veh.commit = "rtor"
        return veh.rtor_on

    def _cross_tta(self, veh, conn):
        """Time until the next vehicle from another approach reaches a conflict with ``conn`` (0 if
        one is in it), and that vehicle; vehicles held by their red light do not count."""
        best, who = np.inf, None
        for u in self.vehicles:
            if u is veh or u.parked:
                continue
            ui, usl = u.where()
            ul = u.lanes[ui]
            if ul.leg == conn.leg:
                continue
            if ul.kind == "conn":
                z = self.m.conflicts.get((ul.id, conn.id))
                if z is None or usl - u.L / 2 > z[1] + 0.5:
                    continue
                tta = max(z[0] - (usl + u.L / 2), 0.0) / max(u.v, 1.0)
            elif ul.kind == "in" and ul.stop is not None and not u.entered and u.v >= 0.5:
                cs = [self.m.lanes[c] for c in ul.next] if u.route_open else u.lanes[ui + 1:ui + 2]
                zs = [(c, self.m.conflicts[(c.id, conn.id)][0]) for c in cs if (c.id, conn.id) in self.m.conflicts]
                if ul.stop["control"] == "signal":
                    ball, arrow = self.signals.state(ul.stop["group"], self.t)[:2]
                    zs = [(c, z) for c, z in zs if ball != BALL_R or (arrow and c.turn == "L")]
                if not zs:
                    continue
                tta = (ul.stop["s"] - (usl + u.L / 2) + min(z for _, z in zs)) / max(u.v, 1.0)
            else:
                continue
            if tta < best:
                best, who = tta, u
        return best, who

    def _ped_tta(self, p):
        """The nearest approaching vehicle's time to pedestrian p's crosswalk (inf if none within 80 m)."""
        best = np.inf
        for u in self.vehicles:
            if u.parked:
                continue
            i, sl = u.where()
            for j in range(i, min(i + 3, len(u.lanes))):
                for s0, s1, cw in u.lanes[j].crosswalks:
                    d = u.off[j] + s0 - u.front
                    if cw == p.cw and 0.0 < d < 80.0:
                        best = min(best, d / max(u.v, 0.5))
        return best

    def _lc_gap(self, veh, tgt, sl, index, follower_only=False):
        """Smaller time gap a lane change into ``tgt`` would leave (the follower's headway behind
        veh there, veh's behind the leader there), and the vehicle bounding it."""
        best, who = np.inf, None
        f, fg = self._follower(tgt.id, sl, index, exclude=(veh,))
        if f is not None:
            best, who = (fg - (veh.L + f.L) / 2) / max(f.v, 1.0), f
        if not follower_only:
            lead, lgap = None, np.inf
            for u, us in index.get(tgt.id, ()):
                if u is not veh and us >= sl and us - sl < lgap:
                    lead, lgap = u, us - sl
            if lead is not None:
                t = (lgap - (veh.L + lead.L) / 2) / max(veh.v, 1.0)
                if t < best:
                    best, who = t, lead
        return best, who

    def _ego_overtake(self, veh, ln, sl, index):
        """The ego held up behind a slow leader (below 0.65 of its desired speed, within
        DT_LEAD_GAP + 2 m) for its patience looks for a gap in the next lane once a second;
        pulling out is a choice whose odds rise with the time gap it would leave there. After
        the last visible frame it pulls out only if it was held up by then."""
        cfg = self.cfg
        lead, gap = self._leader(veh, index)
        # v3 (D1): a lead below 0.85 of its desired speed, within 10 m or 2 s, holds it up
        reach = max(cfg["DT_LEAD_GAP"], cfg["DT_FOLLOW_GAP_S"] * veh.v) if cfg.get("DT_V3") else cfg["DT_LEAD_GAP"]
        slow = 0.85 if cfg.get("DT_V3") else 0.65
        if lead is None or lead.parked or gap > reach + 2.0 or lead.v > slow * self._v0(veh):
            veh.behind_since = None
            return
        if veh.behind_since is None:
            veh.behind_since = self.t
        if self.t - veh.behind_since < veh.patience:
            return
        if self.frame >= cfg["DT_CUE_FRAME"] and veh.behind_since * self.fps > cfg["DT_CUE_FRAME"]:
            return
        for side in ("left", "right"):
            tid = ln.left if side == "left" else ln.right
            if tid is None or self.m.lanes[tid].closed_end or not self._line_ok(ln, side, sl, veh):
                continue
            tgt = self.m.lanes[tid]
            if not self._lc_safe(veh, tgt, sl, index):
                continue
            tau, who = self._lc_gap(veh, tgt, sl, index)
            if who is None or tau >= 12.0 or self.decide(veh, "overtake", tau, key=(who.slot, int(self.t)), other=who):
                quick = getattr(veh, "lc_quick", False)   # an overtake that should end inside the clip
                veh.lc = dict(side=side, target=tid, t_sig=self.t, mandatory=False, moving=False, forced=True,
                              lead=_u(self.rng, (1.5, 2.0)) if quick else self._event_value(veh, "DT_LC_LEAD"),
                              dur=_u(self.rng, (2.5, 3.0)) if quick else self._event_value(veh, "DT_LC_DUR"))
                veh.indicator, veh.indicator_on_t = (1 if side == "left" else 2), self.t
                self._event("TURN_ON", veh.slot, -1, veh.indicator)
            return

    def _conflict_blocker(self, veh, conn, index):
        for u in self.vehicles:
            if u is veh:
                continue
            ui, usl = u.where()
            ul = u.lanes[ui]
            uc = ul if ul.kind == "conn" else None
            us = usl
            if ul.kind == "in" and u.entered and not u.route_open and ui + 1 < len(u.lanes):
                uc, us = u.lanes[ui + 1], usl - ul.length   # front over the line, body not yet
            if uc is not None:
                z = self.m.conflicts.get((uc.id, conn.id))
                if z is not None and us - u.L / 2 <= z[1] + 0.5:
                    return u
            elif u.granted and not u.entered and not u.route_open:
                uc = u.lanes[ui + 1] if ui + 1 < len(u.lanes) else None
                if uc is not None and (uc.id, conn.id) in self.m.conflicts:
                    return u
        return None

    def _stop_order(self):
        """R4: earliest arrival; ties within R4_TIE go to the vehicle with nobody on its right."""
        waiting = [u for u in self.vehicles if u.arrival is not None and not u.entered]
        if not waiting:
            return None
        a0 = min(u.arrival for u in waiting)
        tie = [u for u in waiting if u.arrival <= a0 + self.cfg["R4_TIE"]]
        if len(tie) == 1:
            return tie[0]
        legs = {u.lanes[u.where()[0]].leg: u for u in tie}
        free = [u for leg, u in legs.items() if (leg + 1) % 4 not in legs]
        if free:
            return min(free, key=lambda u: (u.arrival, u.slot))
        for leg in (2, 1, 0, 3):   # N, E, S, W
            if leg in legs:
                return legs[leg]
        return tie[0]

    def _stop_order_first(self):
        """The vehicle whose turn it is at the all-way stop. It may wave through a car that came
        to its full stop at most DT_WAVE_WINDOW s later on another approach: going in turn is a
        choice whose odds rise with how much later that car arrived (a WAVE event otherwise)."""
        if self.cfg.get("DT_V3"):   # a car already let through, then one that chose to go in turn, keeps its turn
            for u in self.vehicles:
                if u.arrival is not None and u.granted and not u.entered:
                    return u
            for u in self.vehicles:
                if u.commit == "stop_wave" and not u.entered:
                    return u
        first = self._stop_order()
        if first is None or first.arrival is None:
            return first
        leg = first.lanes[first.where()[0]].leg
        near = [u for u in self.vehicles if u is not first and u.arrival is not None and not u.entered
                and u.lanes[u.where()[0]].leg != leg and u.arrival - first.arrival <= self.cfg["DT_WAVE_WINDOW"]]
        if not near:
            return first
        other = min(near, key=lambda u: (u.arrival, u.slot))
        if self.decide(first, "stop_wave", max(other.arrival - first.arrival, 0.0), key=other.slot, other=other):
            if self.cfg.get("DT_V3"):
                first.commit = "stop_wave"
            return first
        if getattr(first, "waved", None) != other.slot:
            first.waved = other.slot
            self._event("WAVE", first.slot, other.slot)
        return other

    # ----- lane changes (MOBIL) -------------------------------------------
    def _lane_changes(self, index):
        cfg = self.cfg
        for veh in self.vehicles:
            if veh.parked:
                continue
            i, sl = veh.where()
            ln = veh.lanes[i]
            if ln.kind != "road" or (ln.left is None and ln.right is None):
                continue
            if veh.lc is None:
                if ln.closed_end and ln.left is not None and ln.length - sl < 120.0:
                    if not self._line_ok(ln, "left", sl, veh):
                        continue
                    veh.lc = dict(side="left", target=ln.left, t_sig=self.t, mandatory=True, moving=False,
                                  lead=self._event_value(veh, "DT_LC_LEAD"), dur=self._event_value(veh, "DT_LC_DUR"))
                    veh.indicator, veh.indicator_on_t = 1, self.t
                    self._event("TURN_ON", veh.slot, -1, 1)
                    continue
                if self.t < veh.next_mobil:
                    continue
                veh.next_mobil = self.t + 1.0
                if veh.ego and getattr(veh, "overtaker", False):
                    self._ego_overtake(veh, ln, sl, index)
                    continue
                if not veh.ego and self.ego is not None and self.frame >= cfg["DT_CUE_FRAME"] \
                        and np.hypot(*(veh.pose()[0] - self.ego.pose()[0])) < 60.0:
                    continue   # no lane change near the ego that frames 0 to 31 did not show
                if veh.ego:
                    # the ego's wish to change lanes is a hidden variable of the branch: no
                    # discretionary change in the context, drawn at its first chance after it
                    if self.frame < self.cfg["DT_BRANCH_FRAME"]:
                        continue
                    if not hasattr(veh, "mobil_thresh"):
                        veh.mobil_thresh = (cfg["DT_EGO_MOBIL_THRESH"] if self.rng.rand() < cfg["DT_EGO_LC_P"]
                                            else np.inf)
                if veh.polite is None:
                    veh.polite = _u(self.rng, cfg["DT_MOBIL_POLITE"])
                best = None
                for side in ("left", "right"):
                    tgt = ln.left if side == "left" else ln.right
                    if tgt is None or self.m.lanes[tgt].closed_end or not self._line_ok(ln, side, sl, veh):
                        continue
                    gain = self._mobil_gain(veh, ln, self.m.lanes[tgt], sl, index)
                    bias = 0.1 if side == "right" else -0.1
                    thresh = getattr(veh, "mobil_thresh", cfg["DT_MOBIL_THRESH"])
                    if gain is not None and gain + bias > thresh and (best is None or gain > best[0]):
                        best = (gain, side, tgt)
                if best is not None and (veh.ego or self.rng.rand() < cfg["DT_LC_WILL"]):   # the ego's wish is drawn above
                    # drivers take a worthwhile gap only sometimes: lane changes stay a hidden,
                    # per-branch decision rather than a deterministic function of the context
                    veh.lc = dict(side=best[1], target=best[2], t_sig=self.t, mandatory=False, moving=False,
                                  lead=self._event_value(veh, "DT_LC_LEAD"), dur=self._event_value(veh, "DT_LC_DUR"))
                    veh.indicator, veh.indicator_on_t = (1 if best[1] == "left" else 2), self.t
                    self._event("TURN_ON", veh.slot, -1, veh.indicator)
            elif not veh.lc["moving"]:
                if veh.lc.get("sig_at") is not None:   # a planned lane change: indicator on from sig_at
                    if self.t < veh.lc["sig_at"]:
                        continue
                    veh.lc["t_sig"], veh.lc["sig_at"] = veh.lc["sig_at"], None
                    veh.indicator, veh.indicator_on_t = (1 if veh.lc["side"] == "left" else 2), veh.lc["t_sig"]
                    self._event("TURN_ON", veh.slot, -1, veh.indicator)
                waited = self.t - veh.lc["t_sig"]
                lead_t = veh.lc.get("lead", cfg["R7_SIGNAL_POLICY"])
                if waited < (0.0 if self._viol("no_signal") else lead_t):
                    continue
                tgt = self.m.lanes[veh.lc["target"]]
                ok = self._line_ok(ln, veh.lc["side"], sl, veh) and self._lc_safe(veh, tgt, sl, index)
                if ok and veh.lc["mandatory"]:
                    # which safe gap a merger takes is a choice whose odds rise with the gap (and as the lane runs out)
                    if self.t < veh.lc.get("next_try", 0.0):
                        ok = False
                    else:
                        tau, who = self._lc_gap(veh, tgt, sl, index)
                        cue = min(tau, 8.0) + 2.0 * max(0.0, 1.0 - (ln.length - sl) / 60.0)
                        key = (who.slot if who is not None else -1, int(self.t // 1.5))
                        if not self.decide(veh, "merge", cue, key=key, other=who):
                            veh.lc["next_try"] = self.t + 0.6
                            ok = False
                elif ok and veh.lc.get("cutin"):
                    # signalling into the lane ahead of another car: cutting in is a choice whose
                    # odds rise with the time gap it leaves to the car behind
                    tau, who = self._lc_gap(veh, tgt, sl, index, follower_only=True)
                    if who is not None and not self.decide(veh, "cutin", min(tau, 5.0), key=(who.slot, int(self.t)), other=who):
                        ok = False
                if ok:
                    if self._viol("no_signal") and self.t - veh.indicator_on_t < cfg["R7_SIGNAL_CHECK"] - 0.25:
                        self._log_violation(veh.slot, "R7")
                    # lateral progress is tied to distance travelled, so a slow merge stays physical
                    veh.lc.update(moving=True, t_move=self.t, tau=0.0, s_start=veh.s,
                                  D=max(15.0, veh.v * veh.lc.get("dur", cfg["DT_LC_DURATION"])),
                                  d0=(-LANE_W if veh.lc["side"] == "left" else LANE_W))
                    veh.lc["from"] = ln.id
                    veh.set_route(self.m, [tgt.id])
                    veh.d = veh.lc["d0"]
                    self._event("LC_START", veh.slot, -1, 1 if veh.lc["side"] == "left" else 2)
                    # whoever let it in
                    f, _ = self._follower(tgt.id, sl, index, exclude=(veh,))
                    if f is not None and veh.lc.get("mandatory") and not f.let_in:
                        f.let_in = True
                        self._event("LET_IN", f.slot, veh.slot)
                elif not veh.lc["mandatory"] and (not veh.lc.get("forced") or veh.lc.get("cutin")) \
                        and waited > lead_t + (5.0 if veh.lc.get("cutin") else 3.0):   # a cut-in that never came: it stays
                    veh.lc = None
                    veh.indicator = 0
                    self._event("INDICATOR_OFF", veh.slot)

    def _line_ok(self, ln, side, sl, veh):
        """Lane change allowed: the line on ``side`` is dashed over the whole manoeuvre."""
        if self._viol("solid"):
            return True
        lc = veh.lc or {}
        horizon = sl + max(veh.v, 3.0) * (lc.get("dur", self.cfg["DT_LC_DURATION"]) + lc.get("lead", self.cfg["R7_SIGNAL_POLICY"])) + veh.L
        for s0, s1 in (ln.solid_left if side == "left" else ln.solid_right):
            if s1 >= sl - veh.L and s0 <= horizon:
                return False
        return True

    def _lc_safe(self, veh, tgt, sl, index):
        cfg = self.cfg
        f, fg = self._follower(tgt.id, sl, index, exclude=(veh,))
        lead = None
        lgap = np.inf
        for u, us in index.get(tgt.id, ()):
            if u is veh or us < sl:
                continue
            if us - sl < lgap:
                lead, lgap = u, us - sl
        if lead is not None:
            g = lgap - (veh.L + lead.L) / 2
            if g < 1.0 or self._idm(veh, self._v0(veh), g, lead.v) < -cfg["R7_FOLLOWER_DECEL_POLICY"]:
                return False
        if f is not None:
            g = fg - (veh.L + f.L) / 2
            if g < 1.0 or self._idm(f, self._v0(f), g, veh.v) < -cfg["R7_FOLLOWER_DECEL_POLICY"]:
                return False
        return True

    def _mobil_gain(self, veh, ln, tgt, sl, index):
        cfg = self.cfg
        v0 = self._v0(veh)
        lead, gap = self._leader(veh, index)
        a_old = self._idm(veh, v0, gap, lead.v) if lead is not None else self._free(veh, v0)
        nl, ngap = None, np.inf
        for u, us in index.get(tgt.id, ()):
            if u is veh or us < sl:
                continue
            if us - sl < ngap:
                nl, ngap = u, us - sl
        if nl is not None:
            g = ngap - (veh.L + nl.L) / 2
            if g < 1.0:
                return None
            a_new = self._idm(veh, v0, g, nl.v)
        else:
            a_new = self._free(veh, v0)
        f, fg = self._follower(tgt.id, sl, index, exclude=(veh,))
        dpol = 0.0
        if f is not None:
            g = fg - (veh.L + f.L) / 2
            if g < 1.0:
                return None
            fv0 = self._v0(f)
            a_f_new = self._idm(f, fv0, g, veh.v)
            if a_f_new < -cfg["R7_FOLLOWER_DECEL_POLICY"]:
                return None
            a_f_old = self._idm(f, fv0, ngap + fg - (f.L + nl.L) / 2, nl.v) if nl is not None else self._free(f, fv0)
            dpol = a_f_new - a_f_old
        return a_new - a_old + veh.polite * dpol

    # ----- ground truth for the violation sets ------------------------------
    # These follow the rules' *check* definitions (docs/DRIVETOY.md §2), with a
    # margin, not the stricter policy: an act counts only when it is clearly
    # illegal, so the checker's recall measures misses, not boundary noise.
    def _truth_at_line(self, veh, ln):
        cfg = self.cfg
        conn = veh.lanes[veh.route.index(ln.id) + 1] if ln.id in veh.route[:-1] else None
        if ln.stop["control"] == "signal":
            ball, arrow, _, _ = self.signals.state(ln.stop["group"], self.t)
            prot = arrow and conn is not None and conn.turn == "L"
            if self.violation == "red":
                if ball == BALL_R and not prot:
                    self._log_violation(veh.slot, "R1")
                elif ball == BALL_Y and getattr(veh, "yellow_need", 99) <= cfg["R1_STOP_DECEL_CHECK"] - 0.3:
                    self._log_violation(veh.slot, "R1")
            elif self.violation == "left_gap" and conn is not None and conn.turn == "L" and not prot and ball != BALL_R:
                if self._oncoming_min_tta(ln, conn) < cfg["R3_GAP_CHECK"] - 0.3:
                    self._log_violation(veh.slot, "R3")
        elif self.violation == "no_stop" and getattr(veh, "zone_t", None) is not None and veh.zone_t >= 1.0:
            # observable only if the whole stop-zone approach is inside the record (1 s margin)
            if veh.best_stop < 0.5:
                self._log_violation(veh.slot, "R4")
            elif veh.arrival is not None:
                # both stops must begin inside the record (0.4 s margin) to be observable
                earlier = [u for u in self.vehicles if u is not veh and u.arrival is not None and not u.entered
                           and getattr(u, "zone_t", -1.0) >= 0.4 and u.arrival < veh.arrival - cfg["R4_TIE"] - 0.4]
                if earlier:
                    self._log_violation(veh.slot, "R4", earlier[0].slot)

    def _oncoming_min_tta(self, ln, conn):
        ko = (ln.leg + 2) % 4
        best = np.inf
        for u in self.vehicles:
            ui, usl = u.where()
            ul = u.lanes[ui]
            if ul.leg != ko:
                continue
            front, rear = usl + u.L / 2, usl - u.L / 2
            if ul.kind == "conn":
                z = self.m.conflicts.get((ul.id, conn.id))
                if ul.turn == "L" or z is None:
                    continue
                if rear <= z[1] and front >= z[0]:
                    best = 0.0
                elif front < z[0]:
                    best = min(best, (z[0] - front) / max(u.v, 1.0))
            elif ul.kind == "in" and u.v >= 0.5:
                zs = [self.m.conflicts[(c, conn.id)][0] for c in ul.next
                      if (c, conn.id) in self.m.conflicts and self.m.lanes[c].turn != "L"]
                if zs:
                    best = min(best, (ul.stop["s"] - front + min(zs)) / max(u.v, 1.0))
        return best

    def _moving_on_crosswalk(self, cw_id):
        """A vehicle body over crosswalk ``cw_id`` moving faster than 1.5 m/s (rules.py's P2, with margin)."""
        for u in self.vehicles:
            i, sl = u.where()
            for j in range(max(i - 1, 0), min(i + 2, len(u.lanes))):
                for s0, s1, cw in u.lanes[j].crosswalks:
                    m = 0.5 + 0.25 * u.v      # on it a frame before and after too (rules.py judges f-1 and f)
                    if cw == cw_id and u.front > u.off[j] + s0 + m and u.rear < u.off[j] + s1 - m and u.v > 1.5:
                        return True
        return False

    def _max_need_to_stop(self, cw_id):
        """Largest deceleration an approaching vehicle would need to stop before crosswalk ``cw_id``."""
        need = 0.0
        for u in self.vehicles:
            i, sl = u.where()
            for j in range(i, min(i + 3, len(u.lanes))):
                for s0, s1, cw in u.lanes[j].crosswalks:
                    d = u.off[j] + s0 - u.front
                    if cw == cw_id and 0.0 < d < 60.0:
                        need = max(need, u.v ** 2 / (2 * max(d, 0.5)))   # rules.py's P2 definition
        return need

    # ----- pedestrians ----------------------------------------------------
    def _ped_block(self):
        """crosswalk id -> ped slot, for peds on the crosswalk (block) and
        peds about to step onto it (entering)."""
        block, entering = {}, {}
        for p in self.peds:
            if p.state == P_CROSS:
                block.setdefault(p.cw, p.slot)
            elif p.state == P_WAIT and p.intent == I_CROSS and self._ped_signal_ok(p) and not self._viol("ped_dart"):
                entering.setdefault(p.cw, p.slot)
        return block, entering

    def _ped_signal_ok(self, p):
        cw = self.m.crosswalks[[c["id"] for c in self.m.crosswalks].index(p.cw)]
        if cw["group"] is None:
            return True
        return self.signals.state(cw["group"], self.t)[2] == PED_WALK

    def _ped_gap_ok(self, p):
        """P2: no vehicle whose path crosses this crosswalk would need more than P2_DECEL_POLICY to stop."""
        for u in self.vehicles:
            i, sl = u.where()
            for j in range(max(i - 1, 0), min(i + 3, len(u.lanes))):   # i-1: the rear may still be over it
                for s0, s1, cw in u.lanes[j].crosswalks:
                    if cw != p.cw:
                        continue
                    d = u.off[j] + s0 - u.front
                    if u.off[j] + s1 < u.rear:
                        continue
                    if d < 0.5:
                        return False
                    if u.v ** 2 / (2 * max(d - 1.0, 0.05)) > self.cfg["P2_DECEL_POLICY"] and d < 60:
                        return False
            if u.route_open:   # undecided vehicles: assume any movement
                ln = u.lanes[i]
                for c in ln.next:
                    for s0, s1, cw in self.m.lanes[c].crosswalks:
                        if cw == p.cw:
                            d = u.off[i] + ln.length + s0 - u.front
                            if u.v ** 2 / (2 * max(d - 1.0, 0.05)) > self.cfg["P2_DECEL_POLICY"] and d < 60:
                                return False
        return True

    def _step_peds(self, dt):
        cfg = self.cfg
        for p in self.peds:
            if p.state in (P_APPROACH, P_EXIT, P_CROSS):
                p.s += p.speed * dt
                p.v = p.speed
                p.heading = p.path_heading()
                if p.s >= p.cum[-1]:
                    if p.state == P_APPROACH:
                        p.state = P_WAIT
                        p.v = 0.0
                        self._ped_decide(p)
                    elif p.state == P_CROSS:
                        p.state = P_EXIT
                        p.set_path(p.path["exit_other"])
                        p.heading = p.path_heading()
                    else:
                        p.alive = False
            elif p.state == P_WAIT:
                p.v = 0.0
                if p.intent == I_UNKNOWN and p.delay_until is not None and self.t >= p.delay_until:
                    self._ped_decide(p)   # after a pause at the curb
                sig_ok = self._ped_signal_ok(p) or self._viol("jaywalk")
                gap_ok = self._viol("ped_dart") or self._ped_gap_ok(p)
                if p.intent == I_CROSS and self.t >= p.delay_until and sig_ok and gap_ok:
                    if self._viol("jaywalk") and not self._ped_signal_ok(p):
                        self._log_violation(p.slot, "P1")
                    if self._viol("ped_dart") and (self._max_need_to_stop(p.cw) > self.cfg["P2_DECEL_CHECK"] + 0.5
                                                   or self._moving_on_crosswalk(p.cw)):
                        self._log_violation(p.slot, "P2")
                    p.state = P_CROSS
                    p.set_path(np.stack([p.path["curb"], p.path["other_curb"]]))
                    p.cross_start = self.t
                    self._event("PED_CROSS", p.slot, p.cw)
                elif p.intent == I_PASS and self.t >= p.delay_until:
                    p.state = P_EXIT
                    p.set_path(p.path["exit_far"])
        self.gone += [p for p in self.peds if not p.alive]
        self.peds = [p for p in self.peds if p.alive]

    def _ego_will_pass(self, cw_id):
        """Whether crosswalk ``cw_id`` is still ahead on (any possible) route of the ego."""
        ego = self.ego
        if ego is None:
            return False
        i, _ = ego.where()
        for j in range(i, len(ego.lanes)):
            for s0, s1, cw in ego.lanes[j].crosswalks:
                if cw == cw_id and ego.off[j] + s1 > ego.rear:
                    return True
        if ego.route_open:
            for c in ego.lanes[-1].next:
                if any(cw == cw_id for _, _, cw in self.m.lanes[c].crosswalks):
                    return True
        return False

    def _ped_decide(self, p):
        """At the curb: cross, or wait (pass by in residential streets), with odds that rise with
        the nearest vehicle's time to the crosswalk; an undecided pedestrian may pause and decide
        after it. From DT_CUE_FRAME on, only a pedestrian visible at the curb by then (``cued``)
        may still step out in front of the ego."""
        cfg = self.cfg
        rng = self.rng
        cue = min(self._ped_tta(p), 10.0)
        forced = getattr(p, "force_cross", None)
        if forced is None and not getattr(p, "hesitated", False):
            c50, w = style_c50(cfg, "ped_cross", p.style)
            pr = choice_p(cue, c50, w)
            if rng.rand() < cfg["DT_PED_HESITATE"] * 4 * pr * (1 - pr):
                p.hesitated, p.intent = True, I_UNKNOWN
                p.delay_until = self.t + _u(rng, cfg["DT_HESITATE"])
                return
        delay = _u(rng, cfg["PED_DELAY"])
        if forced is not None:
            cross = forced
        elif cfg.get("DT_V2") and self.frame >= cfg["DT_CUE_FRAME"] and self._ego_will_pass(p.cw) and not p.cued:
            cross = False
        else:
            cross = self.decide(p, "ped_cross", cue)
        if cross:
            p.intent = I_CROSS
            d = p.path["other_curb"] - p.path["curb"]
            p.heading = float(np.arctan2(d[1], d[0]))      # turns to face the road
            p.delay_until = self.t + (0.0 if self._viol("ped_dart") else 0.6 + delay)
        else:
            p.intent = I_PASS if self.m.template == "residential" else I_WAIT
            p.delay_until = self.t + delay + (0.0 if p.intent == I_PASS else 1e9)
        self._event("PED_DECIDE", p.slot, p.cw, p.intent)

    # ----- integration ----------------------------------------------------
    def step_frame(self):
        if self.violation in ("tailgate", "speed"):
            for u in self.vehicles:
                self._driver_violation(u)
        v_start = {u.slot: u.v for u in self.vehicles}
        for _ in range(self.sub):
            self._substep()
        self.frame += 1
        self.t = self.frame / self.fps
        for u in self.vehicles:
            u.acc = (u.v - v_start.get(u.slot, u.v)) * self.fps
        self._spawn()
        if self.frame == self.cfg["DT_CUE_FRAME"] and self.ego is not None:
            for p in self.peds:   # at the curb, or a few steps from it, and in view: its choice is cued
                if (p.state == P_WAIT or (p.state == P_APPROACH and p.cum[-1] - p.s <= 6.0)) and self._in_view(p.pos()):
                    p.cued = True
        self._record()

    def _substep(self):
        cfg = self.cfg
        dt = self.dt
        if self.signals is not None:
            self._cue_signals()
        if self.signals is not None and self.signals.update(self.t, self.rng):
            g, ph = self.signals.seq[self.signals.i]
            self._event("SIGNAL", -1, -1, self.signals.i)
        index = self._lane_index()
        self._lane_changes(index)
        index = self._lane_index()
        block, entering = self._ped_block()
        accs = []
        for veh in self.vehicles:
            acc, why, other = self._vehicle_acc(veh, index, block, entering)
            if why != veh.state and why in (RS["YIELD_PED"], RS["YIELD_ONCOMING"], RS["YIELD_CONFLICT"], RS["MERGE"]):
                self._event("YIELD", veh.slot, other if other is not None else -1, why)
            veh.state, veh.yield_to = why, (other if other is not None else -1)
            accs.append(acc)
        for veh, acc in zip(self.vehicles, accs):
            front0 = veh.front
            if veh.stopped and acc > 0.05 and not veh.parked:   # start-up delay once it may move off
                if veh.go_at is None:
                    veh.go_at = self.t + self._event_value(veh, "DT_START_DELAY")
                if self.t < veh.go_at:
                    acc = 0.0
            elif not veh.stopped or acc <= 0.0:
                veh.go_at = None
            v_new = max(0.0, veh.v + acc * dt)
            veh.s += 0.5 * (veh.v + v_new) * dt
            veh.v = v_new
            if veh.nudge is not None:
                veh.d, veh.dslope = self._nudge_profile(veh.nudge, veh.s)
                if veh.s >= veh.nudge["s"][3]:
                    veh.nudge, veh.d, veh.dslope = None, 0.0, 0.0
            if self._viol("ped") and block:
                for j, lj in enumerate(veh.lanes):
                    for s0, s1, cw in lj.crosswalks:
                        if cw in block and front0 < veh.off[j] + s0 <= veh.front:
                            path = lj.pos(np.linspace(s0, s1, 8))
                            for p in self.peds:
                                if p.state == P_CROSS and p.cw == cw and \
                                        np.min(np.linalg.norm(path - p.pos(), axis=1)) <= cfg["R5_LATERAL_CHECK"] - 0.3:
                                    self._log_violation(veh.slot, "R5", p.slot)
                                    break
            i, sl = veh.where()
            ln = veh.lanes[i]
            if ln.kind == "in" and ln.stop is not None and ln.stop["control"] == "signal" and not veh.entered \
                    and veh.turn == "R":   # right turn on red: a full stop in the stop zone first
                if veh.off[i] + ln.stop["s"] - veh.front <= cfg["R4_STOP_ZONE"] and veh.v < cfg["R4_STOP_V"]:
                    veh.rtor_stop += dt
                    veh.rtor_ready = veh.rtor_ready or veh.rtor_stop >= cfg["R4_STOP_TIME"]
                elif not veh.rtor_ready:
                    veh.rtor_stop = 0.0
            # stop-sign bookkeeping
            if ln.kind == "in" and ln.stop is not None and ln.stop["control"] == "stop" and not veh.entered:
                d_line = veh.off[i] + ln.stop["s"] - veh.front
                if d_line <= cfg["R4_STOP_ZONE"] and getattr(veh, "zone_t", None) is None:
                    veh.zone_t, veh.best_stop = self.t, 0.0
                if d_line <= cfg["R4_STOP_ZONE"] and veh.v < cfg["R4_STOP_V"]:
                    if veh.stop_start is None:
                        veh.stop_start = self.t
                    veh.best_stop = max(veh.best_stop, self.t + dt - veh.stop_start)
                    if veh.arrival is None and self.t + dt - veh.stop_start >= cfg["R4_STOP_TIME"]:
                        veh.arrival = veh.stop_start
                elif veh.arrival is None:
                    veh.stop_start = None
            # crossing the stop line
            if not veh.entered and ln.stop is not None and ln.kind == "in":
                if veh.front >= veh.off[i] + ln.stop["s"]:
                    veh.entered, veh.enter_time = True, self.t
                    veh.commit = None
                    if veh.route_pending:
                        veh.route_pending, veh.ind_pending = False, veh.turn in ("L", "R")
                    self._event("ENTER", veh.slot, -1, {"L": 1, "S": 2, "R": 3}.get(veh.turn, 0))
                    if self.violation in ("red", "left_gap", "no_stop") and self.frame >= self.viol_from:
                        self._truth_at_line(veh, ln)
            elif not veh.entered and i > 0 and veh.lanes[i - 1].stop is not None:
                veh.entered, veh.enter_time = True, self.t
            # indicator off after the turn / lane change
            if veh.turn in ("L", "R") and ln.kind == "out" and veh.indicator:
                veh.indicator = 0
                self._event("INDICATOR_OFF", veh.slot)
            if veh.lc is not None and veh.lc.get("moving"):
                tau = min(1.0, (veh.s - veh.lc["s_start"]) / veh.lc["D"])
                if self._viol("solid") and veh.lc["tau"] < 0.5 <= tau:   # the body centre crosses the line here
                    old = self.m.lanes[veh.lc["from"]]
                    solid = old.solid_left if veh.lc["side"] == "left" else old.solid_right
                    if any(s0 + 1.0 <= veh.s <= s1 - 1.0 for s0, s1 in solid):
                        self._log_violation(veh.slot, "R7")
                veh.lc["tau"] = tau
                veh.d = veh.lc["d0"] * (1 - (3 * tau ** 2 - 2 * tau ** 3))
                if tau >= 1.0:
                    veh.d = 0.0
                    veh.lc = None
                    veh.next_mobil = max(veh.next_mobil, self.t + 2.0)   # settle before another change
                    veh.indicator = 0
                    self._event("INDICATOR_OFF", veh.slot)
            stopped = veh.v < 0.3
            if stopped and not veh.stopped:
                self._event("STOP", veh.slot, veh.yield_to, veh.state)
            elif not stopped and veh.stopped:
                self._event("GO", veh.slot)
            veh.stopped = stopped
            if veh.s >= veh.off[-1] - 1.0 and not veh.lanes[-1].next and not veh.lanes[-1].closed_end:
                if veh.ego:
                    veh.s, veh.v = veh.off[-1] - 1.0, 0.0
                    self.ego_ran_out = True
                else:
                    veh.alive = False
        self.gone += [u for u in self.vehicles if not (u.alive or u.ego)]
        self.vehicles = [u for u in self.vehicles if u.alive or u.ego]
        self._step_peds(dt)
        self.t += dt

    def _cue_signals(self):
        """v2 cued timing: from DT_CUE_FRAME on, the ego's own light may change only out of a
        yellow it showed by then (yellow to red). Otherwise the controller holds its state until
        the ego has crossed its stop line, and runs on from there."""
        sig, ego, cfg = self.signals, self.ego, self.cfg
        if not cfg.get("DT_V2") or ego is None or self.frame < cfg["DT_CUE_FRAME"]:
            return
        app = ego.lanes[0]
        governed = app.kind == "in" and app.stop is not None and app.stop["control"] == "signal" and not ego.entered
        if not governed:
            if sig.frozen is not None:
                sig.unfreeze(self.t)
            return
        if sig.frozen is None and sig.state(app.stop["group"], self.t)[0] != BALL_Y:
            sig.frozen = self.t

    def _spawn(self):
        rng = self.rng
        for lid, rate in self.inflow.items():
            renewal = lid in self.gap_lanes   # v3: arrivals with the oncoming headway law
            if renewal:
                if self.t < self.next_arrival.setdefault(lid, self.t + self.headway(lid)):
                    continue
            elif not rng.rand() < rate / self.fps:
                continue
            ln = self.m.lanes[lid]
            ok = True
            for u, us in self._lane_index().get(lid, ()):
                if us - u.L / 2 < 25.0:
                    ok = False
            if ok:
                veh = self.add_vehicle(lid, 3.0)
                if veh is not None:
                    lead, gap = self._leader(veh, self._lane_index())
                    if lead is not None:
                        veh.v = min(veh.v, lead.v + 0.1 * gap)
                    if renewal:
                        veh.T = max(veh.T, self.cfg["DT_ONCOMING_MIN"])
                        self.next_arrival[lid] = self.t + self.headway(lid)
        if self.ped_inflow > 0 and self.m.ped_paths and rng.rand() < self.ped_inflow / self.fps:
            path = self.m.ped_paths[int(rng.randint(len(self.m.ped_paths)))]
            self.add_ped(path, 0.0)

    # ----- recording --------------------------------------------------------
    def _record(self):
        if self.frame < 0:
            return
        A = self.max_agents
        cfg = self.cfg
        ag = np.zeros((A, 10), np.float32)
        lamps = np.zeros((A, 4), np.uint8)
        extra = np.full((A, 6), -1.0, np.float32)   # lane, s_local, indicator, rule_state, yield_to, ped_state|intent
        for u in self.vehicles:
            p, h = u.pose()
            ag[u.slot] = [u.type, p[0], p[1], h, u.v, u.acc, u.L, u.W, u.H, 1.0]
            blink_on = ((self.frame + u.blink_phase) % cfg["L2_BLINK_FRAMES"]) < cfg["L2_BLINK_FRAMES"] // 2
            brake = (u.acc < -cfg["L1_BRAKE_DECEL"] or u.v < cfg["R4_STOP_V"]) and not u.parked
            haz = u.hazard and blink_on
            lamps[u.slot] = [brake, (u.indicator == 1 and blink_on) or haz, (u.indicator == 2 and blink_on) or haz, haz]
            i, sl = u.where()
            extra[u.slot] = [-1 if u.parked else u.route[i], sl, u.indicator, u.state, u.yield_to, -1]
        for p in self.peds:
            pos = p.pos()
            ag[p.slot] = [p.type, pos[0], pos[1], p.heading, p.v, 0.0, p.L, p.W, p.H, 1.0]
            extra[p.slot] = [-1, p.s, 0, RS["PED_WALK"] if p.v > 0 else RS["PED_WAIT"], -1, p.state * 10 + p.intent]
        sig = np.zeros((2, 4), np.float32)
        if self.signals is not None:
            for gi, g in enumerate(self.signals.groups):
                ball, arrow, ped, cd = self.signals.state(g, self.t)
                sig[gi] = [ball, arrow, ped, cd]
        ego = self.ego
        ep, eh = ego.pose()
        lc_moving = np.zeros(A, np.int8)
        for u in self.vehicles:
            if u.lc is not None and u.lc.get("moving"):
                lc_moving[u.slot] = 1 if u.lc["side"] == "left" else 2
        self.frames.append(dict(agents=ag, lamps=lamps, extra=extra, signals=sig, lc=lc_moving,
                                ego=np.array([ep[0], ep[1], eh, ego.v], np.float32)))
        if self.cfg.get("DT_V3"):
            self.frames[-1]["may_go"] = np.array(self.may_go_probe(), np.int8)

    def may_go_probe(self):
        """v3 (Issue 7, D3): whether an ego stopped at its stop line on its own route could enter
        now: (1, 0), (0, index into MAY_GO_REASONS), or (-1, 0) when it has no stop line ahead or has
        entered. An open route needs every move its lane allows. Reads state only; draws nothing."""
        ego = self.ego
        if ego is None or ego.entered:
            return -1, 0
        i, _ = ego.where()
        ln = ego.lanes[i]
        if ln.kind != "in" or ln.stop is None or i + 1 >= len(ego.lanes):
            return -1, 0
        conns = [self.m.lanes[x] for x in ln.next] if ego.route_pending else [ego.lanes[i + 1]]
        signal = ln.stop["control"] == "signal"
        arrow = False
        if signal:
            ball, arrow, _, _ = self.signals.state(ln.stop["group"], self.t)
            if any(not (ball == BALL_G or (arrow and c.turn == "L")) for c in conns):
                return 0, MAY_GO_REASONS.index("light")
        block, entering = self._ped_block()
        if any(cw in block or cw in entering for c in conns for _, _, cw in c.crosswalks):
            return 0, MAY_GO_REASONS.index("pedestrian")
        if signal:   # the gap the most cautious driver takes, judged from a standstill at the line
            need = cautious_gap(self.cfg) + float(np.mean(self.cfg["DT_START_DELAY"]))
            for c in conns:
                if c.turn == "L" and not arrow:
                    gap, who, inside = self._oncoming_gap(ln, c)
                    if inside is not None or (who is not None and gap < need):
                        return 0, MAY_GO_REASONS.index("oncoming")
        for u in self.vehicles:
            if u is ego or u.parked:
                continue
            ui, usl = u.where()
            ul = u.lanes[ui]
            uc, us = (ul, usl) if ul.kind == "conn" else (None, None)
            if ul.kind == "in" and u.entered and not u.route_open and ui + 1 < len(u.lanes):
                uc, us = u.lanes[ui + 1], usl - ul.length   # front over the line, centre not yet
            if uc is None:
                continue
            for c in conns:
                z = self.m.conflicts.get((uc.id, c.id))
                if z is not None and us - u.L / 2 <= z[1]:
                    return 0, MAY_GO_REASONS.index("conflict")
        if not signal:   # D3: no car waiting at another approach is moving
            for u in self.vehicles:
                if u is ego or u.entered or u.arrival is None:
                    continue
                ul = u.lanes[u.where()[0]]
                if ul.kind == "in" and ul.leg != ln.leg and u.v >= self.cfg["R4_STOP_V"]:
                    return 0, MAY_GO_REASONS.index("stop_order")
        return 1, 0

    def routes(self):
        """Final route of every vehicle slot (lane ids, -1 padded) and its lane-change history is in frames."""
        out = {}
        for u in self.vehicles:
            out[u.slot] = list(u.route)
        return out


def _pathlen(pts):
    pts = np.asarray(pts)
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
