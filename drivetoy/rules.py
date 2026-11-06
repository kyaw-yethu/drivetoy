"""Rule checker (the "Check" side of the simulator's rulebook).

Works only from what a clip records — per-frame agent states, lamps,
signal states, lane ids — plus the static map, never from the simulator's
internal decisions, so it can later score any source of trajectories (the
violation sets, or states read back from generated video). Each check is
deliberately looser than the policy it mirrors.

``check(rec, m, cfg)`` returns a list of (frame, rule, agent, detail).
"""
import numpy as np

from .sim import BALL_G, BALL_R, BALL_Y, P_CROSS, PED_WALK, RS

VEH_TYPES = (1, 2, 3, 4)
LANE_HALF = 1.75


def _veh_slots(rec):
    t = rec["agents"][..., 0]
    return [a for a in range(t.shape[1]) if np.isin(t[:, a], VEH_TYPES).any()]


def _ped_slots(rec):
    t = rec["agents"][..., 0]
    return [a for a in range(t.shape[1]) if np.isin(t[:, a], (5, 6)).any()]


def _box(ag):
    x, y, h, L, W = ag[1], ag[2], ag[3], ag[6], ag[7]
    c, s = np.cos(h), np.sin(h)
    ex, ey = np.array([c, s]) * L / 2, np.array([-s, c]) * W / 2
    p = np.array([x, y])
    return np.stack([p + ex + ey, p - ex + ey, p - ex - ey, p + ex - ey])


def _overlap(b1, b2):
    for poly in (b1, b2):
        for i in range(4):
            e = poly[(i + 1) % 4] - poly[i]
            n = np.array([-e[1], e[0]])
            p1, p2 = b1 @ n, b2 @ n
            if p1.max() < p2.min() or p2.max() < p1.min():
                return False
    return True


def _group_index(m, group):
    return m.groups.index(group)


def check(rec, m, cfg, start=0):
    fps = cfg["DT_FPS"]
    ag, ex, sig = rec["agents"], rec["extra"], rec["signals"]
    T = ag.shape[0]
    out = []
    vehs, peds = _veh_slots(rec), _ped_slots(rec)
    lanes = m.lanes
    lane_of = ex[..., 0].astype(int)
    s_loc = ex[..., 1]

    def valid(f, a):
        return ag[f, a, 9] > 0

    def front_line_dist(f, a):
        """Distance from the front bumper to the stop line of the 'in' lane the vehicle is on (None if not on one)."""
        ln = lanes[lane_of[f, a]] if lane_of[f, a] >= 0 else None
        if ln is None or ln.kind != "in" or ln.stop is None:
            return None, None
        return ln.stop["s"] - (s_loc[f, a] + ag[f, a, 6] / 2), ln

    # ---- R0 speed ----
    for a in vehs:
        for f in range(start, T):
            if not valid(f, a) or lane_of[f, a] < 0:
                continue
            lim = lanes[lane_of[f, a]].limit_at(s_loc[f, a])
            if ag[f, a, 4] > lim + cfg["R0_SPEED_TOL"]:
                out.append((f, "R0", a, float(ag[f, a, 4] - lim)))
                break

    # ---- line crossings: R1 (signal), R3 (permissive left), R4 (stop sign) ----
    for a in vehs:
        for f in range(max(start, 1), T):
            if not (valid(f, a) and valid(f - 1, a)):
                continue
            d0, ln = front_line_dist(f - 1, a)
            if ln is None:
                continue
            d1, ln1 = front_line_dist(f, a)
            crossed = (ln1 is not ln) or (d1 is not None and d1 <= 0 < d0)
            if not crossed or d0 <= 0:
                continue
            conn = None   # the movement: first connector the vehicle drives onto from here on
            for g in range(f, T):
                if valid(g, a) and lane_of[g, a] >= 0 and lanes[lane_of[g, a]].kind == "conn":
                    conn = lanes[lane_of[g, a]]
                    break
            if conn is None and "routes" in rec:   # turns after the record ends
                cs = [lanes[i] for i in rec["routes"][a] if i >= 0 and lanes[i].kind == "conn"]
                conn = cs[0] if cs else None
            if ln.stop["control"] == "signal":
                gi = _group_index(m, ln.stop["group"])
                ball, arrow = sig[f, gi, 0], sig[f, gi, 1]
                prot_left = arrow > 0 and conn is not None and conn.turn == "L"
                best_stop, from_start = _zone_stop(ag, a, f, ln, front_line_dist, cfg, start)
                rtor = cfg.get("DT_RTOR") and conn is not None and conn.turn == "R" and \
                    (best_stop >= int(round(cfg["R4_STOP_TIME"] * fps)) - 1 or from_start)
                if ball == BALL_R and not prot_left and not rtor:   # a right turn on red after a full stop is lawful
                    out.append((f, "R1", a, "red"))
                elif ball == BALL_Y:
                    f0 = f
                    while f0 > start and sig[f0 - 1, gi, 0] == BALL_Y:
                        f0 -= 1
                    if f0 > start:   # onset inside the record
                        d_on, _ = front_line_dist(f0, a)
                        if d_on is not None and d_on > 0:
                            need = ag[f0, a, 4] ** 2 / (2 * max(d_on - 0.5, 0.05))
                            if need <= cfg["R1_STOP_DECEL_CHECK"]:
                                out.append((f, "R1", a, f"yellow need={need:.2f}"))
                if conn is not None and conn.turn == "L" and not prot_left and ball != BALL_R:
                    # the line is crossed between frames f-1 and f: take the worse of the two
                    tta = min(_oncoming_tta(rec, m, g, a, conn, ln, lanes, lane_of, s_loc) for g in (f - 1, f))
                    if tta < cfg["R3_GAP_CHECK"]:
                        out.append((f, "R3", a, f"tta={tta:.2f}"))
            else:   # all-way stop: full stop, then order
                need = int(round(cfg["R4_STOP_TIME"] * fps)) - 1
                stopped, g = 0, f - 1
                best = 0
                while g >= start and valid(g, a):   # back through the stop zone only
                    dg, lg = front_line_dist(g, a)
                    if lg is not ln or dg > cfg["R4_STOP_ZONE"] + 1.0:
                        break
                    if ag[g, a, 4] < cfg["R4_STOP_V"]:
                        stopped += 1
                        best = max(best, stopped)
                    else:
                        stopped = 0
                    g -= 1
                if best < need and g >= start:   # only judge when its time in the zone is inside the record
                    out.append((f, "R4", a, "no full stop"))
                arr_a = _arrival(rec, a, f, ln, front_line_dist, cfg, start)
                if arr_a is not None:
                    for b in vehs:
                        if b == a or not valid(f, b):
                            continue
                        db, lb = front_line_dist(f, b)
                        if lb is None or lb.leg == ln.leg or db is None or db <= 0:
                            continue
                        arr_b = _arrival(rec, b, f, lb, front_line_dist, cfg, start)
                        # an earlier arrival held by someone else (H14), or already under way, is not waiting
                        held = int(ex[f, b, 3]) in (RS["YIELD_PED"], RS["YIELD_CONFLICT"]) or ag[f, b, 4] >= cfg["R4_STOP_V"]
                        if arr_b is not None and arr_b < arr_a - cfg["R4_TIE"] * fps - 1 and not held:
                            out.append((f, "R4", a, f"order: {b} arrived first"))

    # ---- R5 crosswalk yield / P1 / P2 ----
    cw_by_id = {c["id"]: c for c in m.crosswalks}
    ped_state = (ex[..., 5] // 10).astype(int)
    routes = {a: _route_of(rec, a, lane_of, m) for a in vehs}

    def front_route(f, a):
        """Front bumper as an arclength along the vehicle's route (None if unknown)."""
        r = routes[a]
        if lane_of[f, a] < 0 or lane_of[f, a] not in r["ids"]:
            return None
        return r["off"][r["ids"].index(lane_of[f, a])] + s_loc[f, a] + ag[f, a, 6] / 2

    for a in vehs:
        r = routes[a]
        for f in range(max(start, 1), T):
            if not (valid(f, a) and valid(f - 1, a)):
                continue
            fr0, fr1 = front_route(f - 1, a), front_route(f, a)
            if fr0 is None or fr1 is None:
                continue
            for ln, off in zip(r["lanes"], r["off"]):
                for s0, s1, cw in ln.crosswalks:
                    if fr0 < off + s0 <= fr1:
                        p_path = ln.pos(np.linspace(s0, s1, 8))
                        for p in peds:
                            if valid(f, p) and ped_state[f, p] == P_CROSS:
                                pp = ag[f, p, 1:3]
                                if _on_cw(pp, cw_by_id[cw]) and \
                                        np.min(np.linalg.norm(p_path - pp, axis=1)) <= cfg["R5_LATERAL_CHECK"]:
                                    out.append((f, "R5", a, f"ped {p}"))
    for p in peds:
        for f in range(max(start, 1), T):
            if not (valid(f, p) and valid(f - 1, p)):
                continue
            if ped_state[f, p] == P_CROSS and ped_state[f - 1, p] != P_CROSS:
                cw = _nearest_cw(ag[f, p, 1:3], m)
                if cw is None:
                    continue
                if cw["group"] is not None:
                    gi = _group_index(m, cw["group"])
                    if sig[f - 1, gi, 2] != PED_WALK and sig[f, gi, 2] != PED_WALK:
                        out.append((f, "P1", p, "start on don't walk"))
                # P2: an approaching vehicle could not stop (on f-1 or f: the step happened in
                # between), or a vehicle was moving over the crosswalk on both f-1 and f — so also
                # at the moment of the step
                for a in vehs:
                    r = routes[a]
                    on = []
                    for g in (f - 1, f):
                        fr = front_route(g, a) if valid(g, a) else None
                        hit = False
                        for ln, off in zip(r["lanes"], r["off"]):
                            for s0, s1, c in ln.crosswalks:
                                if c != cw["id"] or fr is None:
                                    continue
                                d = off + s0 - fr
                                if 0 < d < 60 and ag[g, a, 4] ** 2 / (2 * max(d, 0.5)) > cfg["P2_DECEL_CHECK"]:
                                    out.append((f, "P2", p, f"veh {a} d={d:.1f} (frame {g})"))
                                if d <= 0 and fr - ag[g, a, 6] < off + s1 and ag[g, a, 4] > 1.0:
                                    hit = True
                        on.append(hit)
                    if all(on):
                        out.append((f, "P2", p, f"veh {a} moving on the crosswalk"))

    # ---- R6 headway and collisions ----
    for f in range(start, T):
        live = [a for a in vehs if valid(f, a)]
        boxes = {a: _box(ag[f, a]) for a in live}
        for i, a in enumerate(live):
            for b in live[i + 1:]:
                if np.hypot(*(ag[f, a, 1:3] - ag[f, b, 1:3])) < (ag[f, a, 6] + ag[f, b, 6]) / 2 + 1 and _overlap(boxes[a], boxes[b]):
                    out.append((f, "COLLISION", a, f"veh {b}"))
        for a in live:
            if lane_of[f, a] < 0:
                continue
            for b in live:
                if b == a or lane_of[f, b] != lane_of[f, a] or rec["lc"][f, a] or rec["lc"][f, b]:
                    continue
                gap = s_loc[f, b] - s_loc[f, a] - (ag[f, a, 6] + ag[f, b, 6]) / 2
                if 0 <= s_loc[f, b] - s_loc[f, a] and gap < max(cfg["R6_GAP_MIN"], cfg["R6_HEADWAY_CHECK"] * ag[f, a, 4]) \
                        and ag[f, a, 4] > ag[f, b, 4] + 0.5:
                    out.append((f, "R6", a, f"gap={gap:.1f} to {b}"))
        for p in peds:
            if not valid(f, p):
                continue
            for a in live:
                if np.hypot(*(ag[f, a, 1:3] - ag[f, p, 1:3])) < ag[f, a, 6] / 2 + 0.5:
                    if _point_in_box(ag[f, p, 1:3], boxes[a], 0.25):
                        out.append((f, "COLLISION", a, f"ped {p}"))

    # ---- R7 lane changes ----
    # A lane change starts on the frame its lane id switches to a parallel lane while the
    # lc flag is set (two changes can follow back to back). The line is crossed where the
    # body centre comes within half a lane width of the new lane's centreline; the
    # boundary must not be solid there.
    ind = ex[..., 2]
    for a in vehs:
        for f in range(max(start, 1), T):
            if not (rec["lc"][f, a] and valid(f - 1, a) and lane_of[f, a] != lane_of[f - 1, a]
                    and lane_of[f - 1, a] >= 0):
                continue
            side = int(rec["lc"][f, a])
            old, new = lanes[lane_of[f - 1, a]], lanes[lane_of[f, a]]
            if lanes[lane_of[f - 1, a]].kind != "road":
                continue
            n_on, g = 0, f - 1
            while g >= start and ind[g, a] == side and (g == f - 1 or lane_of[g, a] == lane_of[f - 1, a]):
                n_on += 1
                g -= 1
            if n_on / fps < cfg["R7_SIGNAL_CHECK"] and g >= start:
                out.append((f, "R7", a, f"indicator {n_on / fps:.1f}s"))
            solid = old.solid_left if side == 1 else old.solid_right
            if not solid:
                continue
            prev_d, prev_s = None, None
            for g in range(f, T):
                if not valid(g, a) or lane_of[g, a] != new.id:
                    break
                d = float(np.linalg.norm(ag[g, a, 1:3] - new.pos(s_loc[g, a])))
                if d <= LANE_HALF:
                    s_cross = s_loc[g, a] if prev_d is None else \
                        prev_s + (prev_d - LANE_HALF) / max(prev_d - d, 1e-6) * (s_loc[g, a] - prev_s)
                    if any(s0 <= s_cross <= s1 for s0, s1 in solid):
                        out.append((g, "R7", a, f"solid line at s={s_cross:.1f}"))
                    break
                prev_d, prev_s = d, s_loc[g, a]
    return out


def _zone_stop(ag, a, f, ln, front_line_dist, cfg, start):
    """Longest run of frames before f in which vehicle a stood (v < R4_STOP_V) in the stop zone of
    ``ln``, and whether it was in the zone at the record's first frame (its stop may lie before it)."""
    stopped, best, g = 0, 0, f - 1
    while g >= start and ag[g, a, 9] > 0:
        dg, lg = front_line_dist(g, a)
        if lg is not ln or dg > cfg["R4_STOP_ZONE"] + 1.0:
            break
        stopped = stopped + 1 if ag[g, a, 4] < cfg["R4_STOP_V"] else 0
        best = max(best, stopped)
        g -= 1
    return best, g < start


def _route_of(rec, a, lane_of, m):
    """The lanes a vehicle drives, in order: the lanes it is seen on, then the rest of
    its final route. Lane changes replace a lane by its neighbour at the same arclength,
    so only the first of two parallel lanes enters the offsets (neither has a crosswalk)."""
    ids = []
    for lid in list(lane_of[:, a]) + [int(x) for x in rec.get("routes", np.full((lane_of.shape[1], 1), -1))[a]]:
        lid = int(lid)
        if lid >= 0 and lid not in ids:
            ids.append(lid)
    lanes = [m.lanes[i] for i in ids]
    return dict(ids=ids, lanes=lanes, off=list(np.concatenate([[0.0], np.cumsum([l.length for l in lanes])])[:-1]))


def _arrival(rec, a, f, ln, front_line_dist, cfg, start):
    """When the vehicle arrived at the stop line of ``ln``: the first frame of its earliest
    full stop (>= R4_STOP_TIME, v < R4_STOP_V) inside the stop zone during this approach,
    looking back from frame f (the earliest stop of any length if it made no full stop).
    None unless the vehicle's entry into the stop zone is inside the record: after a full
    stop a queued car may creep inside the zone, so an earlier stop can hide before it."""
    ag = rec["agents"]
    need = int(round(cfg["R4_STOP_TIME"] * cfg["DT_FPS"])) - 1
    runs, run_end = [], None          # (first frame, length), latest first
    g = f
    while g >= start and ag[g, a, 9] > 0:
        dg, lg = front_line_dist(g, a)
        if lg is not ln or dg > cfg["R4_STOP_ZONE"] + 1.0:
            break
        stopped = ag[g, a, 4] < cfg["R4_STOP_V"]
        if stopped and run_end is None:
            run_end = g
        if not stopped and run_end is not None:
            runs.append((g + 1, run_end - g))
            run_end = None
        g -= 1
    if g < start:
        return None                   # zone entry not seen
    if run_end is not None:
        runs.append((g + 1, run_end - g))
    if not runs:
        return None
    full = [r for r in runs if r[1] >= need]
    return (full or runs)[-1][0]


def _oncoming_tta(rec, m, f, a, conn, ln, lanes, lane_of, s_loc):
    ag = rec["agents"]
    ko = (ln.leg + 2) % 4
    best = np.inf
    for b in range(ag.shape[1]):
        if b == a or ag[f, b, 9] <= 0 or ag[f, b, 0] not in VEH_TYPES or lane_of[f, b] < 0:
            continue
        lb = lanes[lane_of[f, b]]
        if lb.leg != ko or lb.kind not in ("in", "conn"):
            continue
        if lb.kind == "conn":
            if lb.turn == "L":
                continue
            z = m.conflicts.get((lb.id, conn.id))
            if z is not None and s_loc[f, b] - ag[f, b, 6] / 2 <= z[1] and s_loc[f, b] + ag[f, b, 6] / 2 >= z[0]:
                best = min(best, 0.0)
            elif z is not None and s_loc[f, b] + ag[f, b, 6] / 2 < z[0]:
                best = min(best, (z[0] - s_loc[f, b] - ag[f, b, 6] / 2) / max(ag[f, b, 4], 1.0))
            continue
        if all(lanes[c].turn == "L" for c in lb.next):
            continue
        d = lb.stop["s"] - (s_loc[f, b] + ag[f, b, 6] / 2)
        zs = [m.conflicts[(c, conn.id)][0] for c in lb.next if (c, conn.id) in m.conflicts and lanes[c].turn != "L"]
        if not zs or ag[f, b, 4] < 0.5:
            continue
        best = min(best, (d + min(zs)) / max(ag[f, b, 4], 1.0))
    return best


def _on_cw(p, cw):
    from .roads import _points_in_quad
    return bool(_points_in_quad(np.asarray(p)[None], cw["poly"])[0])


def _nearest_cw(p, m):
    for cw in m.crosswalks:
        if _on_cw(p, cw):
            return cw
    best, bd = None, 3.0
    for cw in m.crosswalks:
        d = np.min(np.linalg.norm(cw["ends"] - p, axis=1))
        if d < bd:
            best, bd = cw, d
    return best


def _point_in_box(p, box, margin):
    c = box.mean(0)
    ex, ey = (box[0] - box[1]) / 2, (box[0] - box[3]) / 2
    d = p - c
    return abs(d @ ex) / (ex @ ex) <= 1 + margin / np.linalg.norm(ex) and abs(d @ ey) / (ey @ ey) <= 1 + margin / np.linalg.norm(ey)
