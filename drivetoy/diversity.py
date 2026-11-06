"""DriveToy diversity labels (README, "Diversity"). Every clip gets exactly one
label per aspect, computed from its labels of frames 0 to 31 alone:

A  traffic light   the ego's governing light (its approach group) over the clip
B  cause           what binds the ego in frames 5 to 31 (from rule_state and yield_to)
C  longitudinal    what the ego's speed does
D  lateral         what the ego's path does

``clip_labels(rec, m, cfg)`` returns dict(A, B, C, D, a_ok, info). ``a_ok`` says whether the
ego carries out the typical response to its light inside the clip (the plan's first check).
``cue_problems(rec, m, cfg)`` lists the plan's second check: the ego's light changing after the
last visible frame without a yellow shown by then, and an ego reaction after it that depends on
a pedestrian who was still undecided at the last visible frame.
"""
import numpy as np

from .config import DRIVETOY_DEFAULT_CFG as _CFG
from .choices import CHOICE_KINDS
from .sim import BALL_G, BALL_R, BALL_Y, EV, P_CROSS, RS

A_LABELS = ["no light", "steady green", "steady red", "red to green", "green to yellow", "yellow to red", "arrow",
            "permissive left"]
B_LABELS = ["none", "lead vehicle", "signal", "stop sign", "oncoming traffic", "crossing traffic", "pedestrian",
            "lane end", "speed limit", "cut-in vehicle", "parked obstacle"]
C_LABELS = ["cruise", "follow", "slow down", "stop", "wait", "pull away", "stop and go"]
D_LABELS = ["straight", "left turn", "right turn", "lane change", "merge", "nudge around"]
E_LABELS = ["pending", "decided", "none"]   # choices.end_label: the ego's major choices at frame 31
ASPECTS = dict(A=A_LABELS, B=B_LABELS, C=C_LABELS, D=D_LABELS, E=E_LABELS)

STOP_V = 0.3          # stopped (as R4_STOP_V)
MOVE_V = 2.0          # clearly moving
BIND_MIN = 4          # frames (0.8 s) a cause must bind to beat free driving
CLEAR_AFTER = 10      # frames after the ego enters the junction that still count for its light (clears it)


def _runs(x):
    """Values of x with consecutive repeats collapsed."""
    x = list(x)
    return [v for i, v in enumerate(x) if i == 0 or v != x[i - 1]]


def ego_enter(rec, T):
    """(frame, turn code 1 L / 2 S / 3 R) of the ego crossing its stop line in frames < T, or (None, 0)."""
    for f, typ, a, _, val in rec["events"]:
        if int(typ) == EV["ENTER"] and int(a) == 0 and 0 <= int(f) < T:
            return int(f), int(val)
    return None, 0


def ego_approach(rec, m):
    """The signal group index that governs the ego at frame 0, or None."""
    lane0 = int(rec["extra"][0, 0, 0])
    if lane0 < 0 or not m.groups:
        return None
    ln = m.lanes[lane0]
    if ln.kind != "in" or ln.stop is None or ln.stop["control"] != "signal":
        return None
    return m.groups.index(ln.stop["group"])


def light_label(rec, m, T):
    gi = ego_approach(rec, m)
    if gi is None:
        return "no light", True
    sig = rec["signals"]
    v = rec["ego"][:T, 3]
    ent, turn = ego_enter(rec, T)
    end = T - 1 if ent is None else min(T - 1, ent + CLEAR_AFTER)
    ball, arrow = sig[:end + 1, gi, 0].astype(int), sig[:end + 1, gi, 1] > 0
    if ent is not None and turn == 1:
        if arrow[ent] or arrow[max(ent - 1, 0)]:
            return "arrow", True
        if ball[ent] == BALL_G:
            return "permissive left", True
    seq = _runs(ball)
    pairs = list(zip(seq, seq[1:]))
    stopped = (v < STOP_V).any()
    if (BALL_R, BALL_G) in pairs:
        t_g = next(f for f in range(1, end + 1) if ball[f] == BALL_G and ball[f - 1] == BALL_R)
        return "red to green", bool(ent is not None or (v[t_g:] > MOVE_V).any())
    if (BALL_G, BALL_Y) in pairs or (seq[0] == BALL_Y and (BALL_Y, BALL_R) in pairs):
        t_y = next((f for f in range(1, end + 1) if ball[f] == BALL_Y and ball[f - 1] == BALL_G), 0)
        braking = float(np.max(v[t_y:]) - v[-1]) > 3.0       # stopping, if not yet stopped
        return ("green to yellow" if (BALL_G, BALL_Y) in pairs else "yellow to red"), \
            bool(stopped or braking or ent is not None)
    if seq == [BALL_G]:
        return "steady green", bool(ent is not None or (v > MOVE_V).mean() > 0.5)
    if seq == [BALL_R]:
        return "steady red", bool(stopped or v[-1] < v[0] - 3.0)
    return None, False


def _parked_slots(rec):
    ag, lane = rec["agents"], rec["extra"][..., 0]
    veh = (ag[..., 9] > 0) & (ag[..., 0] >= 1) & (ag[..., 0] <= 4)
    return {a for a in range(1, ag.shape[1]) if veh[:, a].any() and (lane[veh[:, a], a] < 0).all()}


def _cut_ins(rec, T):
    """slot -> frame of a lane change into the ego's lane that starts in frames < T."""
    lane = rec["extra"][..., 0].astype(int)
    out = {}
    for f, typ, a, _, _ in rec["events"]:
        f, a = int(f), int(a)
        if int(typ) == EV["LC_START"] and a > 0 and 0 <= f < T and lane[f, a] == lane[f, 0] >= 0:
            out.setdefault(a, f)
    return out


LEAD_GAP = _CFG["DT_LEAD_GAP"]         # causes "lead vehicle", "parked obstacle": within 10 m
CUTIN_GAP_S = _CFG["DT_CUTIN_GAP_S"]   # cause "cut-in vehicle": within a 2 s time gap


def follow_binds(rec, f, y, gap_m, gap_s=0.0):
    """The simulator's FOLLOW names any vehicle ahead in the lane, however far. It counts as the
    ego following that vehicle only while the bumper gap is at most ``gap_m`` metres, or
    ``gap_s`` seconds at the ego's speed."""
    if y <= 0:
        return False
    ag = rec["agents"]
    gap = float(np.hypot(*(ag[f, y, 1:3] - ag[f, 0, 1:3]))) - (ag[f, 0, 6] + ag[f, y, 6]) / 2
    return gap <= max(gap_m, gap_s * float(rec["ego"][f, 3]))


def follow_gaps(cfg=None):
    """(metres, seconds) within which the speed label "follow" binds: a bumper gap of at most
    DT_FOLLOW_GAP_S (2 s) at the ego's speed."""
    return 0.0, float((cfg or _CFG)["DT_FOLLOW_GAP_S"])


def lead_gaps(cfg=None):
    """(metres, seconds) for the causes "lead vehicle", "parked obstacle" and the overtake check:
    10 m or 2 s, whichever is larger."""
    return LEAD_GAP, float((cfg or _CFG)["DT_FOLLOW_GAP_S"])


def _limit_binds(ln, s, state, ahead=80.0):
    """A lower limit (the school zone) holds the ego below the lane's normal limit: it is inside
    the zone, or slowing (SLOW) for one that starts within ``ahead`` metres."""
    if ln.limit_at(s) < ln.limit - 0.1:
        return True
    return state == RS["SLOW"] and any(lim < ln.limit - 0.1 and 0.0 <= s0 - s <= ahead for s0, _, lim in ln.zones)


def _nudge_frames(rec, m, T, parked, off_min=0.3, reach=25.0):
    """Frames in which the ego is displaced sideways around a parked car: off its lane centre by
    more than ``off_min`` with a parked vehicle within ``reach`` metres."""
    if not parked:
        return set()
    off, ag, ego = lateral_offset(rec, m, T), rec["agents"], rec["ego"]
    off = off - np.median(off)   # relative to the driver's own line in the lane
    out = set()
    for f in range(T):
        if abs(off[f]) <= off_min:
            continue
        if any(np.hypot(*(ag[f, a, 1:3] - ego[f, :2])) <= reach for a in parked if ag[f, a, 9] > 0):
            out.add(f)
    return out


def cause_label(rec, m, T, lo=5, cfg=None):
    ex = rec["extra"]
    st, yt = ex[lo:T, 0, 3].astype(int), ex[lo:T, 0, 4].astype(int)
    lane, s_loc = ex[lo:T, 0, 0].astype(int), ex[lo:T, 0, 1]
    v = rec["ego"][lo:T, 3]
    parked, cut = _parked_slots(rec), _cut_ins(rec, T)
    around = _nudge_frames(rec, m, T, parked)
    bind, free = [], []
    for k in range(len(st)):
        f = lo + k
        s, y = st[k], yt[k]
        if s == RS["FOLLOW"] and y in cut and cut[y] <= f and follow_binds(rec, f, y, LEAD_GAP, CUTIN_GAP_S):
            bind.append("cut-in vehicle")
        elif s == RS["FOLLOW"] and follow_binds(rec, f, y, *lead_gaps(cfg)):
            bind.append("parked obstacle" if y in parked else "lead vehicle")
        elif s == RS["STOP_SIGNAL"]:
            bind.append("signal")
        elif s == RS["STOP_SIGN"]:
            # crossing traffic only while the vehicle it waits for is roughly perpendicular
            cross = y >= 0 and abs(np.cos(rec["agents"][f, y, 3] - rec["ego"][f, 2])) < 0.5
            bind.append("crossing traffic" if cross else "stop sign")
        elif s == RS["YIELD_ONCOMING"]:
            bind.append("oncoming traffic")
        elif s == RS["YIELD_CONFLICT"]:
            bind.append("crossing traffic")
        elif s == RS["YIELD_PED"]:
            bind.append("pedestrian")
        elif s == RS["MERGE"]:
            bind.append("lane end")
        elif f in around:
            bind.append("parked obstacle")
        elif lane[k] >= 0 and _limit_binds(m.lanes[lane[k]], s_loc[k], s):
            bind.append("speed limit")
        else:   # GO / SLOW: free driving
            free.append("none")
    if bind:
        names, counts = np.unique(bind, return_counts=True)
        if counts.max() >= BIND_MIN:
            return str(names[np.argmax(counts)])
    if not free:
        return "none"
    names, counts = np.unique(free, return_counts=True)
    return str(names[np.argmax(counts)])


def speed_label(rec, T, cfg=None):
    v = rec["ego"][:T, 3]
    stopped = v < STOP_V
    if not stopped.any():
        if max(float(np.max(v[:t + 1]) - v[t]) for t in range(T)) > 3.0:
            return "slow down"
        follow = sum(int(rec["extra"][f, 0, 3]) == RS["FOLLOW"] and follow_binds(rec, f, int(rec["extra"][f, 0, 4]), *follow_gaps(cfg))
                     for f in range(T))
        return "follow" if follow >= T // 2 else "cruise"
    first, last = int(np.argmax(stopped)), T - 1 - int(np.argmax(stopped[::-1]))
    pre, post = bool((v[:first] > MOVE_V).any()), bool((v[last + 1:] > MOVE_V).any())
    n = int(stopped.sum())
    if pre and post:
        return "stop and go"
    if post:
        return "pull away" if n < 24 else "wait"
    if pre:
        return "stop" if n < 24 else "wait"
    return "wait"


def lateral_offset(rec, m, T):
    """Signed sideways offset (+ left) of the ego from the centreline of the lane it is on."""
    ex, ego = rec["extra"], rec["ego"]
    out = np.zeros(T)
    for f in range(T):
        lid = int(ex[f, 0, 0])
        if lid < 0:
            continue
        ln = m.lanes[lid]
        c = ln.pos(ex[f, 0, 1])
        h = ln.heading(ex[f, 0, 1])
        out[f] = float(np.dot(ego[f, :2] - c, [-np.sin(h), np.cos(h)]))
    return out


def path_label(rec, m, T):
    h = np.unwrap(rec["ego"][:T, 2])
    dh = float(np.degrees(h[-1] - h[0]))
    if dh > 45:
        return "left turn"
    if dh < -45:
        return "right turn"
    lc = rec["lc"][:T, 0]
    if lc.any():
        t0 = int(np.argmax(lc > 0))
        done = (lc[t0:] == 0).any()
        if t0 >= 1 and done:
            src = int(rec["extra"][t0 - 1, 0, 0])
            return "merge" if src >= 0 and m.lanes[src].closed_end else "lane change"
    else:
        d = lateral_offset(rec, m, T)
        d = d - np.median(d)     # relative to the driver's own line in the lane
        k = int(np.argmax(np.abs(d)))
        if 0.5 <= abs(d[k]) <= 2.5 and (np.abs(d[k:]) < 0.3).any() and abs(dh) < 15:
            return "nudge around"
    if abs(dh) < 15:
        return "straight"
    return None   # a turn still under way at the last visible frame


def clip_labels(rec, m, cfg):
    T = cfg["DT_CUE_FRAME"] + 1
    a, a_ok = light_label(rec, m, T)
    d = path_label(rec, m, T)
    b = "lane end" if d == "merge" else cause_label(rec, m, T, cfg=cfg)   # a merge is made because its lane ends
    return dict(A=a, B=b, C=speed_label(rec, T, cfg), D=d, a_ok=a_ok)


def cue_problems(rec, m, cfg):
    """[(frame, kind, detail)] for the plan's second check (label-only frames only)."""
    T0 = cfg["DT_CUE_FRAME"]
    out = []
    gi = ego_approach(rec, m)
    n = rec["signals"].shape[0]
    if gi is not None:
        ent, _ = ego_enter(rec, n)
        sig = rec["signals"][:, gi, :2].astype(int)
        for f in range(T0 + 1, n if ent is None else min(ent, n)):
            if (sig[f] != sig[f - 1]).any():
                cued = sig[T0, 0] == BALL_Y and sig[f - 1, 0] == BALL_Y and sig[f, 0] == BALL_R
                if not cued:
                    out.append((f, "light", f"{sig[f - 1].tolist()} -> {sig[f].tolist()}"))
    ex = rec["extra"]
    state = (ex[:, :, 5] // 10).astype(int)
    for f in range(T0 + 1, n):   # H31: a pedestrian the ego yields to is already crossing at frame 31
        if int(ex[f, 0, 3]) == RS["YIELD_PED"]:
            p = int(ex[f, 0, 4])
            if p < 0 or state[T0, p] != P_CROSS:
                out.append((f, "pedestrian", f"yields to {p}, not crossing at frame {T0}"))
    lane = ex[..., 0].astype(int)
    ag, ego = rec["agents"], rec["ego"]
    for fe, typ, a, _, _ in rec["events"]:   # H31: nobody cuts in ahead of the ego after frame 31
        fe, a = int(fe), int(a)
        if int(typ) == EV["LC_START"] and a > 0 and T0 < fe < n and lane[fe, a] == lane[fe, 0] >= 0:
            d = float((ag[fe, a, 1:3] - ego[fe, :2]) @ np.array([np.cos(ego[fe, 2]), np.sin(ego[fe, 2])]))
            reach = max(LEAD_GAP, CUTIN_GAP_S * ego[fe, 3]) + (ag[fe, a, 6] + ag[fe, 0, 6]) / 2
            if -(ag[fe, a, 6] + ag[fe, 0, 6]) / 2 < d < reach:
                out.append((fe, "lane change", f"vehicle {a} moves into the ego's lane ahead of it after frame {T0}"))
    dec = rec["decisions"]   # H31, H42: no other road user draws a choice that concerns the ego after frame 31
    for row in dec[(dec[:, 0] > T0) & (dec[:, 1] != 0) & (dec[:, 6] > 0) & (dec[:, 4] >= 0.1) & (dec[:, 4] <= 0.9)]:
        out.append((int(row[0]), "choice", f"agent {int(row[1])} chooses {CHOICE_KINDS[int(row[2])]} after frame {T0}"))
    return out
