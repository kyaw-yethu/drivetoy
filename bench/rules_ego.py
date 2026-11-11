"""drivetoy rules.check restricted to the ego (slot 0), for speed.

The checks and thresholds are those of rules.check, run on the same record, but only the rules the
ego itself can break as TRC counts them (R0, R1, R3, R4, R5, R6 as the follower) are looked for, so
other road users enter only as the context of an ego check. bench/tests/test_rules_ego.py checks
that it returns the ego entries of rules.check.
"""
import numpy as np

EGO = 0


def check_ego(rec, m, cfg, start=0, zone_tol=None):
    """rules.check entries of the ego. ``zone_tol`` (m/s), if given, replaces R0_SPEED_TOL inside school zones."""
    from drivetoy.rules import (BALL_R, BALL_Y, P_CROSS, RS, _arrival, _group_index, _on_cw, _oncoming_tta,
                                _ped_slots, _route_of, _veh_slots, _zone_stop)
    fps = cfg["DT_FPS"]
    ag, ex, sig = rec["agents"], rec["extra"], rec["signals"]
    T = ag.shape[0]
    out = []
    vehs, peds = _veh_slots(rec), _ped_slots(rec)
    if EGO not in vehs:
        return out
    lanes = m.lanes
    lane_of = ex[..., 0].astype(int)
    s_loc = ex[..., 1]
    a = EGO

    def valid(f, b):
        return ag[f, b, 9] > 0

    def front_line_dist(f, b):
        ln = lanes[lane_of[f, b]] if lane_of[f, b] >= 0 else None
        if ln is None or ln.kind != "in" or ln.stop is None:
            return None, None
        return ln.stop["s"] - (s_loc[f, b] + ag[f, b, 6] / 2), ln

    # ---- R0 speed ----
    for f in range(start, T):
        if not valid(f, a) or lane_of[f, a] < 0:
            continue
        ln = lanes[lane_of[f, a]]
        lim = ln.limit_at(s_loc[f, a])
        tol = cfg["R0_SPEED_TOL"]
        if zone_tol is not None and any(s0 <= s_loc[f, a] <= s1 for s0, s1, _ in ln.zones):
            tol = zone_tol
        if ag[f, a, 4] > lim + tol:
            out.append((f, "R0", a, float(ag[f, a, 4] - lim)))
            break

    # ---- line crossings: R1 (signal), R3 (permissive left), R4 (stop sign) ----
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
        conn = None
        for g in range(f, T):
            if valid(g, a) and lane_of[g, a] >= 0 and lanes[lane_of[g, a]].kind == "conn":
                conn = lanes[lane_of[g, a]]
                break
        if conn is None and "routes" in rec:
            cs = [lanes[i] for i in rec["routes"][a] if i >= 0 and lanes[i].kind == "conn"]
            conn = cs[0] if cs else None
        if ln.stop["control"] == "signal":
            gi = _group_index(m, ln.stop["group"])
            ball, arrow = sig[f, gi, 0], sig[f, gi, 1]
            prot_left = arrow > 0 and conn is not None and conn.turn == "L"
            best_stop, from_start = _zone_stop(ag, a, f, ln, front_line_dist, cfg, start)
            rtor = cfg.get("DT_RTOR") and conn is not None and conn.turn == "R" and \
                (best_stop >= int(round(cfg["R4_STOP_TIME"] * fps)) - 1 or from_start)
            if ball == BALL_R and not prot_left and not rtor:
                out.append((f, "R1", a, "red"))
            elif ball == BALL_Y:
                f0 = f
                while f0 > start and sig[f0 - 1, gi, 0] == BALL_Y:
                    f0 -= 1
                if f0 > start:
                    d_on, _ = front_line_dist(f0, a)
                    if d_on is not None and d_on > 0:
                        need = ag[f0, a, 4] ** 2 / (2 * max(d_on - 0.5, 0.05))
                        if need <= cfg["R1_STOP_DECEL_CHECK"]:
                            out.append((f, "R1", a, f"yellow need={need:.2f}"))
            if conn is not None and conn.turn == "L" and not prot_left and ball != BALL_R:
                tta = min(_oncoming_tta(rec, m, g, a, conn, ln, lanes, lane_of, s_loc) for g in (f - 1, f))
                if tta < cfg["R3_GAP_CHECK"]:
                    out.append((f, "R3", a, f"tta={tta:.2f}"))
        else:
            need = int(round(cfg["R4_STOP_TIME"] * fps)) - 1
            stopped, g = 0, f - 1
            best = 0
            while g >= start and valid(g, a):
                dg, lg = front_line_dist(g, a)
                if lg is not ln or dg > cfg["R4_STOP_ZONE"] + 1.0:
                    break
                if ag[g, a, 4] < cfg["R4_STOP_V"]:
                    stopped += 1
                    best = max(best, stopped)
                else:
                    stopped = 0
                g -= 1
            if best < need and g >= start:
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

    # ---- R5 crosswalk yield ----
    cw_by_id = {c["id"]: c for c in m.crosswalks}
    ped_state = (ex[..., 5] // 10).astype(int)
    r = _route_of(rec, a, lane_of, m)

    def front_route(f):
        if lane_of[f, a] < 0 or lane_of[f, a] not in r["ids"]:
            return None
        return r["off"][r["ids"].index(lane_of[f, a])] + s_loc[f, a] + ag[f, a, 6] / 2

    for f in range(max(start, 1), T):
        if not (valid(f, a) and valid(f - 1, a)):
            continue
        fr0, fr1 = front_route(f - 1), front_route(f)
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

    # ---- R6 headway, the ego as the follower ----
    for f in range(start, T):
        if not valid(f, a) or lane_of[f, a] < 0:
            continue
        for b in vehs:
            if b == a or not valid(f, b) or lane_of[f, b] != lane_of[f, a] or rec["lc"][f, a] or rec["lc"][f, b]:
                continue
            gap = s_loc[f, b] - s_loc[f, a] - (ag[f, a, 6] + ag[f, b, 6]) / 2
            if 0 <= s_loc[f, b] - s_loc[f, a] and gap < max(cfg["R6_GAP_MIN"], cfg["R6_HEADWAY_CHECK"] * ag[f, a, 4]) \
                    and ag[f, a, 4] > ag[f, b, 4] + 0.5:
                out.append((f, "R6", a, f"gap={gap:.1f} to {b}"))
    return out
