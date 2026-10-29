"""Road maps for DriveToy's four templates (docs/DRIVETOY.md §4).

A map is a lane graph plus everything the renderer and the rule checker need
to know about the static scene. Coordinates are metres, x east, y north;
traffic keeps right. Every lane is a polyline resampled at 0.5 m with its own
arclength ``s``; a vehicle drives a *route* (a list of lane ids) and lives at
an arclength along it. Parallel lanes of the same road share their ``s``
parametrisation, so a lane change keeps ``s``.

The four-way templates are built for one canonical approach (from the south,
heading north) and rotated by k*90 deg: leg 0 = S, 1 = E, 2 = N, 3 = W. A vehicle
from leg k heads (pi/2 + k pi/2); the leg on its right is (k+1) % 4.
"""
import numpy as np

LANE_W = 3.5
STEP = 0.5
CW_W = 4.0     # crosswalk width (along the road)
SW_W = 3.0     # sidewalk width


def _resample(pts, step=STEP):
    pts = np.asarray(pts, dtype=np.float64)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(2, int(np.ceil(s[-1] / step)) + 1)
    si = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(si, s, pts[:, 0]), np.interp(si, s, pts[:, 1])], 1)


def _bezier(p0, h0, p1, h1, n=60):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d0 = np.array([np.cos(h0), np.sin(h0)])
    d1 = np.array([np.cos(h1), np.sin(h1)])
    dist = np.linalg.norm(p1 - p0)
    if abs(np.cos(h1 - h0)) > 0.999:          # straight through
        return np.stack([p0, p1])
    # Kappa for a quarter circle; the two legs of a turn are not equal, so
    # scale each control arm by its own projected distance.
    k = 0.5523
    a = abs(np.dot(p1 - p0, d0))
    b = abs(np.dot(p1 - p0, d1))
    c0, c1 = p0 + k * a * d0, p1 - k * b * d1
    t = np.linspace(0, 1, n)[:, None]
    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * c0 + 3 * (1 - t) * t ** 2 * c1 + t ** 3 * p1 if dist > 0 else np.stack([p0, p1])


def _fillet(p0, h0, p1, h1, r, n=24):
    """Straight from p0, a circular arc of radius r, straight into p1: the two lines meet at a
    corner, and the arc touches each line r from it (a 90 deg turn). Used for right turns, where
    the arc is concentric with the curb so the turn keeps half a lane from it."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d0 = np.array([np.cos(h0), np.sin(h0)])
    d1 = np.array([np.cos(h1), np.sin(h1)])
    a = np.cross(p1 - p0, d1) / np.cross(d0, d1)   # p0 to the corner
    corner = p0 + a * d0
    t0, t1 = corner - r * d0, corner + r * d1
    side = np.sign(np.cross(d0, d1))                  # +1 left turn, -1 right turn
    c = t0 + side * r * np.array([-d0[1], d0[0]])
    a0 = np.arctan2(*(t0 - c)[::-1])
    a1 = a0 + side * (np.pi / 2)
    return np.vstack([p0, _arc(c, r, a0, a1, n), p1])


def _rot(k):
    c, s = np.cos(k * np.pi / 2), np.sin(k * np.pi / 2)
    return np.array([[c, -s], [s, c]])


class Lane:
    def __init__(self, lid, pts, kind, limit):
        self.id = lid
        self.pts = _resample(pts)
        d = np.diff(self.pts, axis=0)
        seg = np.linalg.norm(d, axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(self.s[-1])
        h = np.arctan2(d[:, 1], d[:, 0])
        self.head = np.concatenate([h, h[-1:]])
        dh = np.angle(np.exp(1j * np.diff(self.head)))
        curv = np.abs(dh) / np.maximum(seg, 1e-6)
        self.curv = np.concatenate([curv, [0.0]])
        self.kind = kind            # 'in' | 'out' | 'conn' | 'road'
        self.limit = limit          # m/s
        self.zones = []             # [(s0, s1, limit)] lower limits (school zone)
        self.next = []
        self.left = self.right = None
        self.solid_left = []        # [(s0, s1)] where the left boundary may not be crossed
        self.solid_right = []
        self.stop = None            # dict(s, control 'signal'|'stop', leg, group)
        self.crosswalks = []        # [(s0, s1, cw_id)]
        self.turn = None            # 'L' | 'S' | 'R' (connectors)
        self.leg = None             # approach leg for 'in' lanes and connectors
        self.out_leg = None
        self.prev = None            # connectors: the incoming lane they leave from
        self.closed_end = False     # lane drop: the lane ends, vehicles must leave it
        self.lane_index = 0         # 0 = innermost
        self.inflow_w = 1.0         # v3: weight of this lane in its approach's traffic (Issue 4)

    def pos(self, s):
        s = np.clip(s, 0.0, self.length)
        return np.stack([np.interp(s, self.s, self.pts[:, 0]), np.interp(s, self.s, self.pts[:, 1])], -1)

    def heading(self, s):
        i = np.clip(np.searchsorted(self.s, s) - 1, 0, len(self.s) - 1)
        return self.head[i]

    def limit_at(self, s):
        v = self.limit
        for s0, s1, lim in self.zones:
            if s0 <= s <= s1:
                v = min(v, lim)
        return v

    def limits(self):
        """Speed limit at every sample (zones applied); cached — zones are set right after construction."""
        if getattr(self, "_lim", None) is None:
            self._lim = np.array([self.limit_at(x) for x in self.s])
        return self._lim

    def solid(self, side, s):
        return any(a <= s <= b for a, b in (self.solid_left if side == "left" else self.solid_right))


class RoadMap:
    def __init__(self, template):
        self.template = template
        self.lanes = []
        self.crosswalks = []   # dict(id, poly(4,2), ends(2,2) curb points, axis unit, signal group or None, lanes)
        self.signal_heads = [] # dict(kind 'veh'|'ped', leg, group, pos(3), heading, countdown)
        self.signs = []        # dict(kind, pos(2), heading, value)
        self.markings = []     # dict(style, pts(N,2))
        self.surfaces = []     # dict(kind 'road'|'sidewalk'|'school'|'island', poly(N,2))
        self.buildings = []    # dict(center(2), size(3), heading)
        self.statics = []      # dict(kind 'parked'|'cone', pos(2), heading, size(3), hazard)
        self.ped_paths = []    # dict(id, pts(N,2), curb_index, cw, far_pts) — see sim.py
        self.conflicts = {}    # (lane_a, lane_b) -> (sa0, sa1) zone on lane a
        self.groups = []       # signal groups ('NS', 'EW') for signalised maps
        self.intersection = None  # dict(center, half) for four-way maps
        self.spawn_lanes = []  # lanes where traffic enters the map

    def add_lane(self, pts, kind, limit):
        ln = Lane(len(self.lanes), pts, kind, limit)
        self.lanes.append(ln)
        return ln

    def finish(self):
        """Conflict zones between connectors of different approaches (and of
        merging lanes) — where two vehicle bodies would overlap."""
        conns = [l for l in self.lanes if l.kind == "conn"]
        for a in conns:
            for b in conns:
                if a is b or a.leg == b.leg:
                    continue
                d = np.linalg.norm(a.pts[:, None, :] - b.pts[None, :, :], axis=-1)
                near = np.where(d.min(1) < 2.8)[0]
                if len(near):
                    self.conflicts[(a.id, b.id)] = (float(a.s[near[0]]), float(a.s[near[-1]]))
        return self


# ---------------------------------------------------------------------------
# Four-way intersections (signal4, stop4)
# ---------------------------------------------------------------------------

def _arc(c, r, a0, a1, n=12):
    a = np.linspace(a0, a1, n)
    return np.stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a)], 1)


def _four_way(rng, template, n_lanes, limit, signalised, crosswalk_p=1.0, force_cw=(), heads_corner=False,
              v3=False, left_share=0.25):
    """Legs start ``inner`` = half + rc from the centre, where ``rc`` is the
    curb radius of the rounded corners; the crosswalk sits just outside the
    corner fillet and the stop line 1 m before it. Legs in ``force_cw`` always
    get a crosswalk (the draw is still made, so the rest of the map is unchanged)."""
    m = RoadMap(template)
    L = 200.0
    half = n_lanes * LANE_W                    # road half-width (one direction)
    rc = 6.0 if n_lanes > 1 else 4.0
    inner = half + rc
    has_cw = [rng.rand() < crosswalk_p or k in force_cw for k in range(4)]
    y_cw0, y_cw1 = inner, inner + CW_W
    y_stop = y_cw1 + 1.0
    y_out = y_cw1 + 0.5
    m.intersection = dict(center=np.zeros(2), half=inner, stop=y_stop)
    m.groups = ["NS", "EW"] if signalised else []

    inc, out = {}, {}
    for k in range(4):
        R = _rot(k)
        for j in range(n_lanes):
            x = (j + 0.5) * LANE_W
            ln = m.add_lane((R @ np.array([[x, -L], [x, -y_stop]]).T).T, "in", limit)
            ln.leg, ln.lane_index = k, j
            ln.stop = dict(s=ln.length, control="signal" if signalised else "stop", leg=k,
                           group=("NS" if k % 2 == 0 else "EW") if signalised else None)
            inc[k, j] = ln
            m.spawn_lanes.append(ln.id)
            xo = -(j + 0.5) * LANE_W
            lo = m.add_lane((R @ np.array([[xo, -y_out], [xo, -L]]).T).T, "out", limit)
            lo.leg, lo.lane_index = k, j
            out[k, j] = lo
        for j in range(n_lanes - 1):
            inc[k, j].right, inc[k, j + 1].left = inc[k, j + 1].id, inc[k, j].id
            out[k, j].right, out[k, j + 1].left = out[k, j + 1].id, out[k, j].id
            # no lane changes on the approach (lanes are chosen by movement)
            inc[k, j].solid_right = [(0.0, inc[k, j].length)]
            inc[k, j + 1].solid_left = [(0.0, inc[k, j + 1].length)]
        if v3 and n_lanes == 2:   # v3: the left-only lane carries left_share of the approach
            inc[k, 0].inflow_w, inc[k, 1].inflow_w = 2 * left_share, 2 * (1 - left_share)

    movements = {}  # (k, j) -> [(turn, leg_out, j_out)]
    for k in range(4):
        if n_lanes == 1:
            movements[k, 0] = [("L", (k + 3) % 4, 0), ("S", (k + 2) % 4, 0), ("R", (k + 1) % 4, 0)]
        else:
            movements[k, 0] = [("L", (k + 3) % 4, 0)]
            movements[k, 1] = [("S", (k + 2) % 4, 1), ("R", (k + 1) % 4, 1)]
    for (k, j), moves in movements.items():
        a = inc[k, j]
        for turn, ko, jo in moves:
            b = out[ko, jo]
            if turn == "R":    # concentric with the corner curb, half a lane from it
                pts = _fillet(a.pts[-1], a.head[-1], b.pts[0], b.head[0], rc + LANE_W / 2)
            else:
                pts = _bezier(a.pts[-1], a.head[-1], b.pts[0], b.head[0])
            c = m.add_lane(pts, "conn", limit)
            c.turn, c.leg, c.out_leg = turn, k, ko
            c.next = [b.id]
            c.prev = a.id
            a.next.append(c.id)
            for kk in (k, ko):   # near and far crosswalks the connector passes
                if has_cw[kk]:
                    inside = _points_in_quad(c.pts, _cw_poly(kk, half, y_cw0, y_cw1))
                    if inside.any():
                        idx = np.where(inside)[0]
                        c.crosswalks.append((float(c.s[idx[0]]), float(c.s[idx[-1]]), kk))

    # junction surface: square with the four corners rounded off (CCW: SE, NE, NW, SW)
    junction = []
    for q in range(4):
        R = _rot(q)
        cc = np.array([inner, -inner])
        junction.append((R @ _arc(cc, rc, np.pi, np.pi / 2).T).T)
        sw = np.vstack([_arc(cc, rc, np.pi, np.pi / 2), _arc(cc, rc - SW_W, np.pi / 2, np.pi)])
        m.surfaces.append(dict(kind="sidewalk", poly=(R @ sw.T).T))
    m.surfaces.append(dict(kind="road", poly=np.vstack(junction)))

    for k in range(4):
        R = _rot(k)
        hd_in = np.pi / 2 + k * np.pi / 2
        face = hd_in + np.pi                       # a sign facing approaching traffic
        if has_cw[k]:
            poly = _cw_poly(k, half, y_cw0, y_cw1)
            ym = -(y_cw0 + y_cw1) / 2
            ends = (R @ np.array([[-half - 0.5, ym], [half + 0.5, ym]]).T).T
            m.crosswalks.append(dict(id=k, poly=poly, ends=ends, leg=k,
                                     group=("EW" if k % 2 == 0 else "NS") if signalised else None))
            m.markings.append(dict(style="crosswalk", pts=poly))
        for j in range(n_lanes):
            ln = inc[k, j]
            p, hd = ln.pts[-1], ln.head[-1]
            nrm = np.array([np.cos(hd - np.pi / 2), np.sin(hd - np.pi / 2)])
            m.markings.append(dict(style="stop_line", pts=np.stack([p - nrm * LANE_W / 2, p + nrm * LANE_W / 2])))
            if n_lanes > 1:
                m.markings.append(dict(style="arrow_L" if j == 0 else "arrow_SR",
                                       pts=np.stack([ln.pos(ln.length - 9.0), ln.pos(ln.length - 4.0)])))
        m.markings.append(dict(style="double_yellow", pts=(R @ np.array([[0.0, -L], [0.0, -y_stop]]).T).T))
        for j in range(1, n_lanes):
            if v3:   # v3: solid along the whole approach, like the ban on lane changes
                m.markings.append(dict(style="solid_white", pts=(R @ np.array([[j * LANE_W, -L], [j * LANE_W, -y_stop]]).T).T))
            else:
                m.markings.append(dict(style="solid_white", pts=(R @ np.array([[j * LANE_W, -y_stop - 30], [j * LANE_W, -y_stop]]).T).T))
                m.markings.append(dict(style="dashed_white", pts=(R @ np.array([[j * LANE_W, -L], [j * LANE_W, -y_stop - 30]]).T).T))
            m.markings.append(dict(style="dashed_white", pts=(R @ np.array([[-j * LANE_W, -y_out], [-j * LANE_W, -L]]).T).T))
        for side in (-1, 1):
            m.markings.append(dict(style="edge", pts=(R @ np.array([[side * half, -L], [side * half, -inner]]).T).T))
            x0, x1 = side * half, side * (half + SW_W)
            m.surfaces.append(dict(kind="sidewalk", poly=(R @ np.array([[x0, -L], [x1, -L], [x1, -inner], [x0, -inner]]).T).T))
        m.surfaces.append(dict(kind="road", poly=(R @ np.array([[-half, -L], [half, -L], [half, -inner], [-half, -inner]]).T).T))
        if signalised:
            g = "NS" if k % 2 == 0 else "EW"
            # far-side pole past the far-right corner; with heads_corner each head also gets
            # pos_near, the pole on the nose of that corner 1 m in from the curb, which brings the
            # heads 4.5 m nearer the approach (2 lanes). export.py uses it for the ego's approach only.
            pole = np.array([half + 1.0, inner + 1.5])
            pole_n = np.full(2, inner - (rc - 1.0) / np.sqrt(2))
            for j in range(n_lanes):   # far-side mast-arm heads over each lane
                pos, pn = R @ np.array([(j + 0.5) * LANE_W, inner + 1.0]), R @ np.array([(j + 0.5) * LANE_W, pole_n[1]])
                m.signal_heads.append(dict(kind="veh", leg=k, group=g, lane=j, pos=np.array([pos[0], pos[1], 6.6]), heading=face))
                if heads_corner:
                    m.signal_heads[-1]["pos_near"] = np.array([pn[0], pn[1], 6.6])
            pos, pn = R @ pole, R @ pole_n   # far-right pole head
            m.signal_heads.append(dict(kind="veh_pole", leg=k, group=g, lane=None, pos=np.array([pos[0], pos[1], 3.0]), heading=face))
            if heads_corner:
                m.signal_heads[-1]["pos_near"] = np.array([pn[0], pn[1], 3.0])
            # v2 near-side head on a pole at the near-right corner, on the junction side of the
            # crosswalk: at the stop line itself it would sit outside the 70 deg view of the first car
            pos = R @ np.array([half + 0.8, -(y_cw0 - 0.5)])
            m.signal_heads.append(dict(kind="veh_near", leg=k, group=g, lane=None, pos=np.array([pos[0], pos[1], 3.0]), heading=face))
            if has_cw[k]:
                gp = "EW" if k % 2 == 0 else "NS"
                ends = m.crosswalks[-1]["ends"]
                for e, other in ((ends[0], ends[1]), (ends[1], ends[0])):
                    d = other - e
                    m.signal_heads.append(dict(kind="ped", leg=k, group=gp, lane=None,
                                               pos=np.array([e[0], e[1], 2.6]) + np.r_[-0.4 * d / np.linalg.norm(d), 0.0],
                                               heading=float(np.arctan2(-d[1], -d[0]))))
        else:
            pos = R @ np.array([half + 0.8, -y_stop])
            m.signs.append(dict(kind="stop", pos=pos, heading=face))
        m.signs.append(dict(kind="speed", pos=R @ np.array([half + 1.0, -L + 40.0]), heading=face, value=round(limit * 3.6)))
    _ped_paths_four_way(m, half, inner, L)
    _buildings_around(m, rng, half + SW_W + 1.0, L, inner + 2.0)
    return m.finish()


def _cw_poly(k, half, y0, y1):
    R = _rot(k)
    return (R @ np.array([[-half, -y1], [half, -y1], [half, -y0], [-half, -y0]]).T).T


def _points_in_quad(pts, quad):
    """Convex quad (CCW or CW) containment."""
    inside = np.ones(len(pts), bool)
    sign = None
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        cr = (b[0] - a[0]) * (pts[:, 1] - a[1]) - (b[1] - a[1]) * (pts[:, 0] - a[0])
        if sign is None:
            sign = np.sign(np.sum(np.sign(cr))) or 1.0
        inside &= cr * sign >= -1e-9
    return inside


def _ped_paths_four_way(m, half, inner, L):
    """For every crosswalk, one approach path per curb (walking in along the
    leg's sidewalk) ending at the curb point. sim.py extends a path across
    the crosswalk or back along the sidewalk once the pedestrian's intent is
    drawn."""
    xs = half + SW_W / 2
    for cw in m.crosswalks:
        R = _rot(cw["leg"])
        ym = -(inner + CW_W / 2)
        for side in (-1, 1):
            curb = R @ np.array([side * (half + 0.5), ym])
            m.ped_paths.append(dict(
                cw=cw["id"], side=side, curb=curb,
                approach=(R @ np.array([[side * xs, ym - 45.0], [side * xs, ym], [side * (half + 0.5), ym]]).T).T,
                other_curb=R @ np.array([-side * (half + 0.5), ym]),
                exit_far=(R @ np.array([[side * xs, ym], [side * xs, -L + 20.0]]).T).T,
                exit_other=(R @ np.array([[-side * xs, ym], [-side * xs, -L + 20.0]]).T).T))


def _buildings_around(m, rng, inner, L, corner):
    for q in range(4):
        R = _rot(q)
        y = -corner - rng.uniform(0, 4)
        while y > -L + 10:
            depth = rng.uniform(8, 16)
            width = rng.uniform(8, 20)
            height = rng.uniform(5, 18)
            for side in (-1, 1):
                if rng.rand() < 0.8:
                    c = R @ np.array([side * (inner + rng.uniform(1, 4) + depth / 2), y - width / 2])
                    m.buildings.append(dict(center=c, size=np.array([depth, width, height]), heading=q * np.pi / 2))
            y -= width + rng.uniform(1, 6)


def build_signal4(rng, cfg, protected=None, force_cw=()):
    m = _four_way(rng, "signal4", 2, 50 / 3.6, True, force_cw=force_cw, heads_corner=cfg.get("DT_HEADS_CORNER", False),
                  v3=bool(cfg.get("DT_V3")), left_share=cfg.get("DT_LEFT_SHARE", 0.25))
    m.protected_left = rng.rand() < cfg["SIG_ARROW_P"]
    if protected is not None:
        m.protected_left = bool(protected)
    m.countdown = rng.rand() < cfg["SIG_COUNTDOWN_P"]
    if not cfg.get("DT_NEAR_HEADS", True):   # dropped after the draws, so the rest of the map is unchanged
        m.signal_heads = [h for h in m.signal_heads if h["kind"] != "veh_near"]
    for h in m.signal_heads:
        h["countdown"] = m.countdown and h["kind"] == "ped"
        h["arrow"] = m.protected_left and ((h["kind"] == "veh" and h["lane"] == 0) or h["kind"] == "veh_near")
    return m


def build_stop4(rng, cfg, force_cw=()):
    m = _four_way(rng, "stop4", 1, 40 / 3.6, False, crosswalk_p=0.6, force_cw=force_cw)
    m.protected_left = False
    m.countdown = False
    return m


# ---------------------------------------------------------------------------
# Straight roads (multilane, residential)
# ---------------------------------------------------------------------------

def build_multilane(rng, cfg, **_):
    """Northbound 3 lanes (lane 2, the outermost, closes at a cone taper),
    southbound 2. A solid stretch forbids lane changes between lanes 0 and 1."""
    m = RoadMap("multilane")
    limit = 70 / 3.6
    L0, L1 = -200.0, 300.0
    nb = []
    drop = rng.uniform(90.0, 160.0)                # where lane 2 ends (y)
    solid0 = rng.uniform(-60.0, 40.0)
    solid1 = solid0 + rng.uniform(30.0, 60.0)
    for j in range(3):
        x = (j + 0.5) * LANE_W
        y_end = drop if j == 2 else L1
        ln = m.add_lane([[x, L0], [x, y_end]], "road", limit)
        ln.lane_index = j
        nb.append(ln)
        m.spawn_lanes.append(ln.id)
    nb[2].closed_end = True
    for j in range(2):
        nb[j].right, nb[j + 1].left = nb[j + 1].id, nb[j].id
    nb[0].solid_right = [(solid0 - L0, solid1 - L0)]
    nb[1].solid_left = [(solid0 - L0, solid1 - L0)]
    sb = []
    for j in range(2):
        x = -(j + 0.5) * LANE_W
        ln = m.add_lane([[x, L1], [x, L0]], "road", limit)
        ln.lane_index = j
        sb.append(ln)
        m.spawn_lanes.append(ln.id)
    sb[0].right, sb[1].left = sb[1].id, sb[0].id
    m.markings.append(dict(style="double_yellow", pts=np.array([[0.0, L0], [0.0, L1]])))
    m.markings.append(dict(style="dashed_white", pts=np.array([[LANE_W, L0], [LANE_W, solid0]])))
    m.markings.append(dict(style="solid_white", pts=np.array([[LANE_W, solid0], [LANE_W, solid1]])))
    m.markings.append(dict(style="dashed_white", pts=np.array([[LANE_W, solid1], [LANE_W, L1]])))
    m.markings.append(dict(style="dashed_white", pts=np.array([[2 * LANE_W, L0], [2 * LANE_W, drop]])))
    m.markings.append(dict(style="dashed_white", pts=np.array([[-LANE_W, L1], [-LANE_W, L0]])))
    m.markings.append(dict(style="edge", pts=np.array([[3 * LANE_W, L0], [3 * LANE_W, drop], [2 * LANE_W, drop + 25.0], [2 * LANE_W, L1]])))
    m.markings.append(dict(style="edge", pts=np.array([[-2 * LANE_W, L0], [-2 * LANE_W, L1]])))
    for y in (drop - 60.0, drop - 30.0):
        m.markings.append(dict(style="arrow_merge_left", pts=np.array([[2.5 * LANE_W, y - 4], [2.5 * LANE_W, y]])))
    # v2: twice the size, 45 m before the end, so it is still ahead of an ego that starts 60-80 m out
    m.signs.append(dict(kind="lane_ends", pos=np.array([3 * LANE_W + 1.0, drop - 45.0]), heading=-np.pi / 2, size=2.0))
    m.signs.append(dict(kind="speed", pos=np.array([3 * LANE_W + 1.0, L0 + 30.0]), heading=-np.pi / 2, value=70))
    for i in range(10):   # cone taper closing lane 2
        t = i / 9
        m.statics.append(dict(kind="cone", pos=np.array([3 * LANE_W - 0.5 - t * (LANE_W - 0.6), drop - 2 + t * 25.0]),
                              heading=0.0, size=np.array([0.4, 0.4, 0.7]), hazard=False))
    m.surfaces.append(dict(kind="road", poly=np.array([[-2 * LANE_W, L0], [3 * LANE_W, L0], [3 * LANE_W, drop],
                                                       [2 * LANE_W, drop + 25.0], [2 * LANE_W, L1], [-2 * LANE_W, L1]])))
    for x0, x1 in ((3 * LANE_W, 3 * LANE_W + SW_W), (-2 * LANE_W - SW_W, -2 * LANE_W)):
        m.surfaces.append(dict(kind="sidewalk", poly=np.array([[x0, L0], [x1, L0], [x1, L1], [x0, L1]])))
    m.drop_y = drop
    m.protected_left = m.countdown = False
    _buildings_line(m, rng, 3 * LANE_W + SW_W + 2.0, -2 * LANE_W - SW_W - 2.0, L0, L1)
    return m.finish()


def build_residential(rng, cfg, **_):
    """Two-lane street with parking lanes on both sides, an unsignalised
    crosswalk in a school zone (red pavement, 30 km/h) and parked cars that
    hide pedestrians stepping out to it."""
    m = RoadMap("residential")
    limit = 50 / 3.6
    school = 30 / 3.6
    L0, L1 = -200.0, 200.0
    PK = 2.4
    half = LANE_W + PK
    y_cw = rng.uniform(10.0, 60.0)
    zone = (y_cw - 60.0, y_cw + 30.0)
    nb = m.add_lane([[LANE_W / 2, L0], [LANE_W / 2, L1]], "road", limit)
    sb = m.add_lane([[-LANE_W / 2, L1], [-LANE_W / 2, L0]], "road", limit)
    nb.zones = [(zone[0] - L0, zone[1] - L0, school)]
    sb.zones = [(L1 - zone[1], L1 - zone[0], school)]
    m.spawn_lanes += [nb.id, sb.id]
    cw_poly = np.array([[-half, y_cw - CW_W / 2], [half, y_cw - CW_W / 2], [half, y_cw + CW_W / 2], [-half, y_cw + CW_W / 2]])
    m.crosswalks.append(dict(id=0, poly=cw_poly, ends=np.array([[-half - 0.6, y_cw], [half + 0.6, y_cw]]), leg=0, group=None))
    nb.crosswalks.append((y_cw - CW_W / 2 - L0, y_cw + CW_W / 2 - L0, 0))
    sb.crosswalks.append((L1 - (y_cw + CW_W / 2), L1 - (y_cw - CW_W / 2), 0))
    m.markings.append(dict(style="crosswalk", pts=cw_poly))
    m.markings.append(dict(style="double_yellow", pts=np.array([[0.0, L0], [0.0, y_cw - CW_W / 2 - 1]])))
    m.markings.append(dict(style="double_yellow", pts=np.array([[0.0, y_cw + CW_W / 2 + 1], [0.0, L1]])))
    for side in (-1, 1):
        m.markings.append(dict(style="edge", pts=np.array([[side * LANE_W, L0], [side * LANE_W, L1]])))
        m.markings.append(dict(style="edge", pts=np.array([[side * half, L0], [side * half, L1]])))
    m.surfaces.append(dict(kind="road", poly=np.array([[-half, L0], [half, L0], [half, L1], [-half, L1]])))
    if cfg.get("DT_V3"):
        # v3 (Issue 8): red pavement on both lanes along the whole zone, and a filled blue triangle
        # (2.0 m by 4 m) on each lane 2 m into its zone, apex toward the approaching driver
        m.surfaces.append(dict(kind="school", poly=np.array([[-LANE_W, zone[0]], [LANE_W, zone[0]], [LANE_W, zone[1]], [-LANE_W, zone[1]]])))
        for sgn, y_in in ((1.0, zone[0]), (-1.0, zone[1])):
            xc = sgn * LANE_W / 2
            m.markings.append(dict(style="school_triangle", pts=np.array(
                [[xc, y_in + sgn * 2.0], [xc - 1.0, y_in + sgn * 6.0], [xc + 1.0, y_in + sgn * 6.0]])))
    else:
        m.surfaces.append(dict(kind="school", poly=np.array([[0.0, zone[0]], [LANE_W, zone[0]], [LANE_W, zone[0] + 15], [0.0, zone[0] + 15]])))
        m.surfaces.append(dict(kind="school", poly=np.array([[-LANE_W, zone[1] - 15], [0.0, zone[1] - 15], [0.0, zone[1]], [-LANE_W, zone[1]]])))
    for x0, x1 in ((half, half + SW_W), (-half - SW_W, -half)):
        m.surfaces.append(dict(kind="sidewalk", poly=np.array([[x0, L0], [x1, L0], [x1, L1], [x0, L1]])))
    m.signs.append(dict(kind="school_zone", pos=np.array([half + 1.0, zone[0]]), heading=-np.pi / 2, value=30))
    m.signs.append(dict(kind="school_zone", pos=np.array([-half - 1.0, zone[1]]), heading=np.pi / 2, value=30))
    m.signs.append(dict(kind="crosswalk", pos=np.array([half + 1.0, y_cw - CW_W / 2 - 1]), heading=-np.pi / 2))
    m.signs.append(dict(kind="crosswalk", pos=np.array([-half - 1.0, y_cw + CW_W / 2 + 1]), heading=np.pi / 2))
    m.signs.append(dict(kind="speed", pos=np.array([half + 1.0, L0 + 20.0]), heading=-np.pi / 2, value=50))
    # parked cars, leaving the crosswalk and 6 m either side clear
    for side in (-1, 1):
        y = L0 + rng.uniform(0, 8)
        while y < L1 - 5:
            ln = rng.uniform(4.2, 5.2)
            if abs(y + ln / 2 - y_cw) > CW_W / 2 + 4.0 + ln / 2 and rng.rand() < 0.75:
                hd = np.pi / 2 if side > 0 else -np.pi / 2
                m.statics.append(dict(kind="parked", pos=np.array([side * (LANE_W + PK / 2), y + ln / 2]), heading=hd,
                                      size=np.array([ln, 1.8, rng.uniform(1.4, 1.9)]), hazard=False))
            y += ln + rng.uniform(0.8, 3.0)
    m.crosswalk_y = y_cw
    xs = half + SW_W / 2
    for side in (-1, 1):
        for d in (-1, 1):
            curb = np.array([side * (half + 0.6), y_cw])
            m.ped_paths.append(dict(cw=0, side=side, approach=np.array([[side * xs, y_cw + d * 45.0], [side * xs, y_cw], curb]),
                                    curb=curb, other_curb=np.array([-side * (half + 0.6), y_cw]),
                                    exit_far=np.array([[side * xs, y_cw], [side * xs, y_cw - d * 60.0]]),
                                    exit_other=np.array([[-side * xs, y_cw], [-side * xs, y_cw + d * 60.0]])))
    m.protected_left = m.countdown = False
    _buildings_line(m, rng, half + SW_W + 2.0, -half - SW_W - 2.0, L0, L1)
    return m.finish()


def _buildings_line(m, rng, x_right, x_left, y0, y1):
    for side, x in ((1, x_right), (-1, x_left)):
        y = y0 + rng.uniform(0, 5)
        while y < y1 - 5:
            depth, width, height = rng.uniform(8, 14), rng.uniform(8, 18), rng.uniform(4, 14)
            if rng.rand() < 0.8:
                m.buildings.append(dict(center=np.array([x + side * (depth / 2 + rng.uniform(0, 3)), y + width / 2]),
                                        size=np.array([depth, width, height]), heading=0.0))
            y += width + rng.uniform(1, 6)


BUILDERS = dict(signal4=build_signal4, stop4=build_stop4, multilane=build_multilane, residential=build_residential)


def build_map(template, rng, cfg, **opts):
    return BUILDERS[template](rng, cfg, **opts)
