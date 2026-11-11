"""Traffic rule compliance (TRC) of a plan in one future (Task_specific_JDM docs/TASK_ENCODER_EVAL.md).

The plan replaces the ego in the record of that future from frame 32 on: positions and headings are the
plan's own (plan_states), speeds come from its positions, and lanes and arclengths from matching them onto the
lane graph.
The first eight rules are those of drivetoy rules.check that the ego itself breaks; the ninth,
"did not go when the way was clear", reads the may_go label of the future.
"""
import numpy as np

EGO, LANE_HALF, AHEAD, ROUTE_SLACK = 0, 1.75, 10.0, 0.5
# Speeding tolerance inside a school zone (m/s above its 30 km/h): the simulator's egos brake into the zone and
# enter it up to 3.569 m/s too fast, in every drive_v4 split, so 3.6 lets every recorded future pass. Outside
# school zones the tolerance stays R0_SPEED_TOL (0.5 m/s).
ZONE_TOL = 3.6
RULES = ('speeding', 'red light', 'yellow', 'left turn', 'no full stop', 'stop order', 'pedestrian',
         'too close', 'did not go')
_NAMES = {'R0': 'speeding', 'R3': 'left turn', 'R5': 'pedestrian', 'R6': 'too close'}


def plan_states(path, ego, fps):
    """World centre poses and speeds (F, 4: x, y, heading, speed) of a plan (F, 3) in the ego frame at frame o, from
    the ego's state at o (an agents row: type, x, y, heading, speed, ...). The simulator advances a position by the
    mean of the speeds at both ends of a step, so a central difference of the positions recovers each frame's speed;
    the last frame takes the backward difference."""
    x0, y0, h0 = (float(ego[i]) for i in (1, 2, 3))
    c, s = np.cos(h0), np.sin(h0)
    p = np.asarray(path, np.float64)
    xy = np.stack([x0 + c * p[:, 0] - s * p[:, 1], y0 + s * p[:, 0] + c * p[:, 1]], -1)
    q = np.vstack([[x0, y0], xy])
    v = np.empty(len(p))
    v[:-1] = np.linalg.norm(q[2:] - q[:-2], axis=1) * fps / 2
    v[-1] = np.linalg.norm(q[-1] - q[-2]) * fps
    return np.column_stack([xy, np.angle(np.exp(1j * (h0 + p[:, 2]))), v])


class Matcher:
    """Map matching of a path onto the lane graph, preferring lanes reachable from the previous one."""
    def __init__(self, m):
        self.lanes = m.lanes
        self.pts = np.concatenate([ln.pts for ln in m.lanes])
        self.head = np.concatenate([ln.head for ln in m.lanes])
        self.start = np.cumsum([0] + [len(ln.pts) for ln in m.lanes])

    def related(self, k):
        ln = self.lanes[k]
        out = {k, *ln.next}
        for n in ln.next:
            out.update(self.lanes[n].next)
        out.update(x for x in (ln.left, ln.right) if x is not None)
        if ln.prev is not None:
            out.update(self.lanes[getattr(ln.prev, 'id', ln.prev)].next)
        return out

    def project(self, p, h):
        """lane id -> (distance to its centreline, arclength, beyond its ends)."""
        d = np.linalg.norm(self.pts - p, axis=1)
        d[np.abs(np.angle(np.exp(1j * (self.head - h)))) > np.pi / 3] = np.inf
        out = {}
        for k, ln in enumerate(self.lanes):
            seg = d[self.start[k]:self.start[k + 1]]
            i = int(seg.argmin())
            if not np.isfinite(seg[i]) or seg[i] > 4.0:
                continue
            s = ln.s[i] + (p - ln.pts[i]) @ np.array([np.cos(ln.head[i]), np.sin(ln.head[i])])
            out[k] = (float(seg[i]), float(np.clip(s, 0, ln.length)), s > ln.length or s < 0)
        return out

    def _greedy(self, P, H, prev):
        lanes, ss = [], []
        for p, h in zip(P, H):
            c = self.project(p, h)
            pref = self.related(prev) if prev >= 0 else set()
            cand = {k: v for k, v in c.items() if k in pref}
            if not cand:
                cand = {k: v for k, v in c.items() if v[0] < LANE_HALF + 0.75}
            if not cand:
                lanes.append(-1)
                ss.append(0.0)
                continue
            k = min(cand, key=lambda x: cand[x][0])
            if prev in cand and not cand[prev][2] and cand[prev][0] <= cand[k][0] + 0.5:
                k = prev   # hysteresis against flicker between parallel lanes
            lanes.append(k)
            ss.append(cand[k][1])
            prev = k
        return np.array(lanes), np.array(ss)

    def _fit(self, k, Q):
        """Distance of each point to lane k's centreline, and its arclength there."""
        ln = self.lanes[k]
        pts = np.asarray(ln.pts, np.float64)
        d = np.linalg.norm(Q[:, None] - pts[None], axis=-1)
        i = d.argmin(1)
        h = np.asarray(ln.head)[i]
        s = np.asarray(ln.s)[i] + ((Q - pts[i]) * np.stack([np.cos(h), np.sin(h)], -1)).sum(-1)
        return d[np.arange(len(Q)), i], np.clip(s, 0, ln.length)

    def _siblings(self, k):
        """The connectors that leave the same lane as connector k, or that lane's connectors for an incoming lane."""
        ln = self.lanes[k]
        if ln.kind == 'conn':
            if ln.prev is None:
                return [k]
            ln = self.lanes[getattr(ln.prev, 'id', ln.prev)]
        return [n for n in ln.next if self.lanes[n].kind == 'conn']

    def match(self, P, H, prev, route=None):
        """Lane and arclength of each pose, matched frame by frame, then each stretch of frames on junction
        connectors moved as a whole onto the connector that fits it best: where a right turn and straight through
        leave the stop line together, the nearest connector of the first frames is often the wrong one. A connector
        on ``route`` (lane ids of the future's route) wins if it fits within ROUTE_SLACK m of the best, as when the
        ego only creeps into the junction within the plan."""
        lanes, ss = self._greedy(P, H, prev)
        P = np.asarray(P, np.float64)
        conn = np.array([k >= 0 and self.lanes[k].kind == 'conn' for k in lanes])
        f = 0
        while f < len(lanes):
            if not conn[f]:
                f += 1
                continue
            g = f
            while g < len(lanes) and conn[g]:
                g += 1
            before = lanes[f - 1] if f > 0 else prev
            sib = self._siblings(before if before >= 0 else lanes[f])
            if len(sib) > 1:
                fits = {k: self._fit(k, P[f:g]) for k in sib}
                best = min(fits, key=lambda k: fits[k][0].mean())
                on = [k for k in sib if route is not None and k in route]
                if on:
                    r = min(on, key=lambda k: fits[k][0].mean())
                    if fits[r][0].mean() <= fits[best][0].mean() + ROUTE_SLACK:
                        best = r
                lanes[f:g], ss[f:g] = best, fits[best][1]
            f = g
        return lanes, ss


def splice(rec, k, plan, lanes, ss, matcher, fps, o):
    """Record of future k up to the end of the plan, with the ego from frame o + 1 replaced by the
    plan (F, 4: x, y, heading, speed)."""
    T = o + 1 + len(plan)
    r = {key: rec[key][k, :T].copy() for key in ('agents', 'extra', 'signals', 'lc')}
    r['routes'] = rec['routes'][k]
    w = np.arange(o + 1, T)
    r['agents'][w, EGO, 1:5] = plan
    r['agents'][w, EGO, 5] = np.diff(np.r_[r['agents'][o, EGO, 4], plan[:, 3]]) * fps
    r['extra'][w, EGO, 0] = lanes
    r['extra'][w, EGO, 1] = ss
    r['lc'][w, EGO] = 0
    seq = np.r_[int(r['extra'][o, EGO, 0]), lanes]
    for i in range(1, len(seq)):
        a, b = seq[i - 1], seq[i]
        if a < 0 or b < 0 or a == b:
            continue
        side = 1 if matcher.lanes[a].left == b else 2 if matcher.lanes[a].right == b else 0
        if side:
            r['lc'][max(o + 1, o + i - 3):min(T, o + i + 4), EGO] = side
    return r


def attribute(v, lo):
    """TRC rule of a rules.check entry, or None: only the ego, only from frame lo, no collisions,
    lane changes or pedestrian rules."""
    f, rule, a, det = v
    if f < lo or a != EGO:
        return None
    if rule == 'R1':
        return 'red light' if det == 'red' else 'yellow'
    if rule == 'R4':
        return 'stop order' if str(det).startswith('order') else 'no full stop'
    return _NAMES.get(rule)


def did_not_go(r, m, cfg, may_go, o):
    """True if the planned ego stands (below R4_STOP_V) at or before its stop line for 1 s while
    may_go is 1 and no stopped vehicle stands within 10 m ahead on its route. At a stop sign the
    second only starts once its full stop is done, and frames are skipped while a vehicle that
    stopped before it still waits."""
    from drivetoy.rules import _arrival, _route_of, _veh_slots
    fps, vstop = cfg['DT_FPS'], cfg['R4_STOP_V']
    need = int(round(cfg['R4_STOP_TIME'] * fps)) - 1
    ag, lanes = r['agents'], m.lanes
    lane_of, s_loc = r['extra'][..., 0].astype(int), r['extra'][..., 1]

    def front_line_dist(f, a):
        ln = lanes[lane_of[f, a]] if lane_of[f, a] >= 0 else None
        if ln is None or ln.kind != 'in' or ln.stop is None:
            return None, None
        return ln.stop['s'] - (s_loc[f, a] + ag[f, a, 6] / 2), ln

    route = _route_of(r, EGO, lane_of, m)

    def along(f, a, end):
        """Front (end=1) or rear (end=-1) of vehicle a along the ego route, None off the route."""
        if lane_of[f, a] not in route['ids']:
            return None
        return route['off'][route['ids'].index(lane_of[f, a])] + s_loc[f, a] + end * ag[f, a, 6] / 2

    others = [b for b in _veh_slots(r) if b != EGO]

    def blocked(f):
        front = along(f, EGO, 1)
        for b in others:
            if ag[f, b, 9] <= 0 or ag[f, b, 4] >= vstop:
                continue
            rear = along(f, b, -1)
            if front is not None and rear is not None and 0 <= rear - front <= AHEAD:
                return True
        return False

    def waits_first(f, ln, arr):
        for b in others:
            if ag[f, b, 9] <= 0:
                continue
            db, lb = front_line_dist(f, b)
            if lb is None or lb.leg == ln.leg or db <= 0 or lb.stop['control'] == 'signal':
                continue
            arr_b = _arrival(r, b, f, lb, front_line_dist, cfg, 0)
            if arr_b is not None and (arr is None or arr_b < arr):
                return True
        return False

    stood = 0   # standing frames in a row, counted from the record before the plan
    for g in range(o, -1, -1):
        if ag[g, EGO, 4] >= vstop:
            break
        stood += 1
    run = 0
    for f in range(o + 1, len(ag)):
        d, ln = front_line_dist(f, EGO)
        if ag[f, EGO, 4] >= vstop or ln is None or d < -0.5:
            stood = run = 0
            continue
        stood += 1
        if may_go[f] != 1 or blocked(f):
            run = 0
            continue
        if ln.stop['control'] != 'signal':
            if stood <= need:
                continue   # the full stop itself
            if waits_first(f, ln, _arrival(r, EGO, f, ln, front_line_dist, cfg, 0)):
                continue
        run += 1
        if run >= fps:
            return True
    return False


def rule_hits(rec, k, plan, lanes, ss, matcher, m, cfg, o):
    """(9,) bool: the TRC rules the plan (F, 4) breaks in future k of the records."""
    from .rules_ego import check_ego
    r = splice(rec, k, plan, lanes, ss, matcher, cfg['DT_FPS'], o)
    hits = {attribute(v, o + 1) for v in check_ego(r, m, cfg, zone_tol=ZONE_TOL)} - {None}
    if did_not_go(r, m, cfg, rec['may_go'][k], o):
        hits.add('did not go')
    return np.array([x in hits for x in RULES])
