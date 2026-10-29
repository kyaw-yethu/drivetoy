"""DriveToy rule episodes that start and finish inside the 32 rendered frames of drive_10k_f32/train,
carried out by the ego or by an agent in the camera's view at the episode's key frame.
Writes coverage stats (rules_stats.json) and one 2x2 GIF per rule (ego view + top-down map)."""
import sys, json
import numpy as np
from multiprocessing import Pool
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/root/projects/Task_specific_JDM")
from f_toy.datasets.drivetoy import sim

ROOT = "/root/projects/drive_10k_f32/train"
OUT = "/root/projects/drive_10k_f32_viz"
TEMPLATES = ["signal4", "stop4", "multilane", "residential"]
LIMIT = np.array([50, 40, 70, 50]) / 3.6
RS = {n: i for i, n in enumerate(sim.RULE_STATES)}
INV = {i: n for n, i in RS.items()}
EV = {n: i for i, n in enumerate(sim.EVENTS)}
PED_NAMES = {0: "-", 1: "approach", 2: "wait", 3: "cross", 4: "exit"}
T, A, BIG = 32, 64, 99
rng = np.random.default_rng(0)

vid = np.load(f"{ROOT}/videos.npy", mmap_mode="r")
a = np.load(f"{ROOT}/annotations.npz")
tpl = a["template"]
N = len(tpl)
ag = a["agents"][:, :T]
rs = a["rule_state"][:, :T]
yt = a["yield_to"][:, :T].astype(int)
ped = a["ped_state"][:, :T] // 10
lc, ind, sig = a["lc"][:, :T], a["indicator"][:, :T], a["signals"][:, :T]
ego = a["ego"][:, :T]
ev, nev = a["events"], a["num_events"]
typ = ag[..., 0].astype(int)
valid = ag[..., 9] > 0
is_veh = valid & (typ >= 1) & (typ <= 4)
is_ped = valid & (typ >= 5)
v, hd = ag[..., 4], ag[..., 3]
tt = np.arange(T)[None, :, None]

# in view: ahead of the camera, within 70 m and the 70 deg horizontal FOV (occlusion ignored)
dx, dy = ag[..., 1] - ego[..., None, 0], ag[..., 2] - ego[..., None, 1]
c, s = np.cos(ego[..., 2])[..., None], np.sin(ego[..., 2])[..., None]
fwd, lat = c * dx + s * dy, -s * dx + c * dy
view = (fwd > 1.0) & (np.hypot(fwd, lat) < 70) & (np.abs(np.arctan2(lat, fwd)) < np.radians(35))
view[..., 0] = False


def first(m):
    """First frame where m holds, per (clip, agent); BIG if never."""
    return np.where(m.any(1), m.argmax(1), BIG)


def run_ge(m, k):
    """True at frame t if m held for frames t-k+1..t."""
    cs = np.concatenate([np.zeros_like(m[:, :1], int), np.cumsum(m, 1)], 1)
    out = np.zeros_like(m)
    out[:, k - 1:] = (cs[:, k:] - cs[:, :-k]) == k
    return out


def at(x, t):
    """x (N,T,A) at frame t (N,A)."""
    return np.take_along_axis(x, np.clip(t, 0, T - 1)[:, None, :], 1)[:, 0]


def of_slot(x, slot):
    """x (N,T,A) of the agent named by slot (N,T,A) -> (N,T,A)."""
    return np.take_along_axis(x, np.clip(slot, 0, A - 1), 2)


enter_t = np.full((N, A), BIG)
enter_turn = np.zeros((N, A), int)
for i in range(N):
    for f, ty, k, _, val in ev[i, :nev[i]]:
        f, k = int(f), int(k)
        if ty == EV["ENTER"] and 0 <= f < T and k >= 0 and enter_t[i, k] == BIG:
            enter_t[i, k], enter_turn[i, k] = f, int(val)

ep, key, partner = {}, {}, {}
none = np.full((N, A), -1)

# R0: the ego holds 85-100% of the posted limit for >= 10 frames (2 s)
lim = (ego[..., 3] >= 0.85 * LIMIT[tpl][:, None])[..., None]
e0 = np.zeros((N, A), bool)
e0[:, 0] = run_ge(lim, 10)[..., 0].any(1)
ep["R0"], key["R0"], partner["R0"] = e0, np.where(e0, first(run_ge(lim, 10)).repeat(A, 1), BIG), none

# R1: moving (> 3 m/s), then a full stop (< 0.3 m/s) held by the signal
stop_sig = is_veh & (rs == RS["STOP_SIGNAL"]) & (v < 0.3)
t1 = first(stop_sig)
ep["R1"] = (t1 < BIG) & ((v > 3) & (tt < t1[:, None])).any(1)
key["R1"], partner["R1"] = t1, none

# R2: stopped by the signal, then released: no longer held and moving > 2 m/s
t2 = first(is_veh & (tt > t1[:, None]) & (rs != RS["STOP_SIGNAL"]) & (v > 2))
ep["R2"], key["R2"], partner["R2"] = (t1 < BIG) & (t2 < BIG), t2, none

# R3: held back by oncoming traffic, then enters the junction turning left and has turned >= 30 deg by its last frame
ty_ = first(is_veh & (rs == RS["YIELD_ONCOMING"]))
lastv = T - 1 - np.argmax(valid[:, ::-1], 1)
dh = np.degrees(np.angle(np.exp(1j * (at(hd, lastv) - at(hd, enter_t)))))
ep["R3"] = (ty_ < enter_t) & (enter_t < T) & (enter_turn == 1) & (dh > 30)
key["R3"], partner["R3"] = enter_t, at(yt, ty_)

# R4: full stop at the stop sign for >= 5 frames (1 s), then enters the junction
ts4 = first(run_ge(is_veh & (rs == RS["STOP_SIGN"]) & (v < 0.3), 5))
ep["R4"], key["R4"], partner["R4"] = (ts4 < enter_t) & (enter_t < T), enter_t, none

# R4b: waits (< 1 m/s) for a crossing vehicle with right of way, then enters the junction
tcf = first(is_veh & (rs == RS["YIELD_CONFLICT"]) & (v < 1))
ep["R4b"], key["R4b"], partner["R4b"] = (tcf < enter_t) & (enter_t < T), enter_t, at(yt, tcf)

# R5: waits (< 1 m/s) while the pedestrian it yields to is on the crosswalk
yp = is_veh & (rs == RS["YIELD_PED"]) & (v < 1) & (yt >= 0) & (of_slot(ped, yt) == 3)
t5 = first(yp)
ep["R5"], key["R5"], partner["R5"] = t5 < BIG, t5, at(yt, t5)

# R6: follows a leader in view for >= 10 frames while the leader's speed varies by >= 3 m/s
fol = is_veh & (rs == RS["FOLLOW"]) & (yt >= 0) & of_slot(view, yt)
lv = np.where(fol, of_slot(v, yt), np.nan)
with np.errstate(all="ignore"):
    lrange = np.nan_to_num(np.nanmax(lv, 1) - np.nanmin(lv, 1))
t6 = first(fol)
ep["R6"], key["R6"], partner["R6"] = (fol.sum(1) >= 10) & (lrange >= 3), t6, at(yt, t6)

# R7 / R8: a lane change that starts and finishes inside the clip, indicator on >= 4 frames before it;
# R8 when the vehicle was merging out of a closing lane before it started
lcon = is_veh & (lc > 0)
tls = first(lcon)
tle = first(is_veh & (lc == 0) & (tt > tls[:, None]))
done = (tls >= 1) & (tls < BIG) & (tle < BIG)
signalled = ((ind > 0) & (tt < tls[:, None])).sum(1) >= 4
merging = ((rs == RS["MERGE"]) & (tt <= tls[:, None])).any(1)
ep["R7"], key["R7"], partner["R7"] = done & signalled & ~merging, tls, none
ep["R8"], key["R8"], partner["R8"] = done & merging, tls, none

# P1 / P2: a pedestrian waits at the curb and then starts crossing (signal4 = on WALK; elsewhere unsignalised)
wait = is_ped & (ped == 2)
tw = first(wait)
tcr = first(is_ped & (ped == 3) & (tt > tw[:, None]))
cross = (tw < BIG) & (tcr < BIG)
ep["P1"], key["P1"], partner["P1"] = cross & (tpl == 0)[:, None], tcr, none
ep["P2"], key["P2"], partner["P2"] = cross & (tpl != 0)[:, None], tcr, none
# P1b: a pedestrian in view waits at a signalised curb for >= 15 frames (3 s)
wv = wait & view
ep["P1b"], key["P1b"], partner["P1b"] = (wv.sum(1) >= 15) & (tpl == 0)[:, None], first(wv), none

DEF = {
    "R0": "ego holds 85-100% of the posted limit for >= 10 frames (school zone ignored)",
    "R1": "moving > 3 m/s, then a full stop held by a red/yellow signal",
    "R2": "stopped by the signal, then released and moving > 2 m/s",
    "R3": "held back by oncoming traffic, then enters turning left and turns >= 30 deg",
    "R4": "full stop >= 1 s at the stop sign, then enters the junction",
    "R4b": "waits (< 1 m/s) for a crossing vehicle with right of way, then enters",
    "R5": "waits (< 1 m/s) while the pedestrian it yields to is on the crosswalk",
    "R6": "follows a leader in view >= 10 frames while the leader's speed varies >= 3 m/s",
    "R7": "lane change starts and finishes in the clip, indicator >= 4 frames before",
    "R8": "lane change out of a closing lane (merge) starts and finishes in the clip",
    "P1": "signal4: pedestrian waits at the curb, then starts crossing",
    "P1b": "signal4: pedestrian in view waits at the curb >= 15 frames",
    "P2": "stop4 / residential: pedestrian waits at an unsignalised curb, then starts crossing",
}

stats, picks = {}, {}
for r in DEF:
    e, k = ep[r], key[r]
    vis = e & at(view, k)          # other agents: in view at the key frame
    vis[:, 0] = False
    if r == "R5":                   # the pedestrian must be in view too
        vis &= np.take_along_axis(at(view, k), np.clip(partner[r], 0, A - 1), 1)
    ego_ok = e[:, 0]
    row = {"def": DEF[r]}
    if r[0] == "R":
        row["ego_clips"] = int(ego_ok.sum())
    row["inview_clips"] = int(vis.any(1).sum())
    row["either_by_template"] = {t: round(float((ego_ok | vis.any(1))[tpl == j].mean()), 3) for j, t in enumerate(TEMPLATES)}
    stats[r] = row
    # prefer the ego carrying out the episode; fill up with in-view agents
    pool_e = np.flatnonzero(ego_ok)
    pe = list(rng.choice(pool_e, min(4, len(pool_e)), replace=False)) if r[0] == "R" else []
    pool_o = np.setdiff1d(np.flatnonzero(vis.any(1)), pe)
    po = list(rng.choice(pool_o, min(4 - len(pe), len(pool_o)), replace=False))
    sel = [(int(i), 0) for i in pe] + [(int(i), int(np.flatnonzero(vis[i])[np.argmin(k[i, vis[i]])])) for i in po]
    picks[r] = [(i, s_, int(key[r][i, s_]), int(partner[r][i, s_])) for i, s_ in sel]

# ---------------- GIFs ----------------
MAP, SPAN, SCALE, CAP = 256, 80.0, 2, 44
META = json.load(open(f"{ROOT}/meta.json"))
try:
    font = ImageFont.load_default(size=12)
except TypeError:
    font = ImageFont.load_default()
BALL = {0: "R", 1: "Y", 2: "G"}


def sim_top(i):
    """Re-simulate clip i (deterministic, equals the stored labels) and draw the top-down map per frame."""
    from f_toy.datasets.drivetoy.scenarios import simulate
    from f_toy.datasets.drivetoy.debug_view import render_topdown
    r = simulate(META["seed"] + int(i), META["cfg"])
    return i, np.stack([render_topdown(r["map"], r["branches"][0], f, size=MAP, span=SPAN)[:, :, ::-1] for f in range(T)])


def to_map(i, t, xy):
    """World xy -> top-down pixel, the transform render_topdown uses (ego-centred, heading up)."""
    ang = np.pi / 2 - ego[i, t, 2]
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    q = (xy - ego[i, t, :2]) @ R.T - np.array([0.0, SPAN * 0.3])
    return MAP / 2 + q[0] * MAP / SPAN, MAP / 2 - q[1] * MAP / SPAN


def who(i, t, k):
    if k < 0 or not valid[i, t, k]:
        return ""
    if typ[i, t, k] >= 5:
        return f"ped{k} {PED_NAMES.get(int(ped[i, t, k]), '?')}"
    return f"{'ego' if k == 0 else 'veh' + str(k)} {INV[int(rs[i, t, k])]} {v[i, t, k]:.1f}m/s"


def panel(i, t, k, kf, pk, top):
    img = Image.fromarray(np.ascontiguousarray(vid[i, :, t].transpose(1, 2, 0))).resize((128 * SCALE,) * 2, Image.NEAREST)
    cv = Image.new("RGB", (img.width + MAP + 2, img.height + CAP), (20, 20, 20))
    cv.paste(img, (0, CAP))
    cv.paste(Image.fromarray(np.ascontiguousarray(top[t])), (img.width + 2, CAP))
    d = ImageDraw.Draw(cv)
    for slot, col in ((k, (255, 230, 0)), (pk, (0, 230, 255))):
        if slot >= 0 and valid[i, t, slot]:
            u, w = to_map(i, t, ag[i, t, slot, 1:3])
            u, w = u + img.width + 2, w + CAP
            d.ellipse([u - 8, w - 8, u + 8, w + 8], outline=col, width=2)
    l1 = f"#{i} {TEMPLATES[tpl[i]]} f{t:02d}  key f{kf:02d}" + ("  <== key" if t == kf else "")
    if tpl[i] == 0:
        l1 += f"  NS {BALL[int(sig[i, t, 0, 0])]} EW {BALL[int(sig[i, t, 1, 0])]}"
    l2 = "yellow: " + who(i, t, k)
    l3 = ("cyan: " + who(i, t, pk)) if pk >= 0 else ""
    for n_, (txt, col) in enumerate([(l1, (255, 255, 255)), (l2, (255, 230, 0)), (l3, (0, 230, 255))]):
        d.text((4, 2 + 13 * n_), txt, fill=col, font=font)
    return cv


ids = sorted({i for p in picks.values() for i, *_ in p})
with Pool(16) as p:
    tops = dict(p.map(sim_top, ids))
for r, p in picks.items():
    frames = []
    for t in range(T):
        ps = [panel(i, t, k, kf, pk, tops[i]) for i, k, kf, pk in p]
        w, h = ps[0].size
        g = Image.new("RGB", (2 * w + 6, 2 * h + 6))
        for n_, q in enumerate(ps):
            g.paste(q, ((n_ % 2) * (w + 6), (n_ // 2) * (h + 6)))
        frames.append(g)
    frames[0].save(f"{OUT}/rule_{r}.gif", save_all=True, append_images=frames[1:], duration=200, loop=0, optimize=True)

json.dump({"stats": stats, "picks (clip, slot, key frame, partner slot)": picks}, open(f"{OUT}/rules_stats.json", "w"), indent=1)
print(json.dumps(stats, indent=1))
print(json.dumps(picks))
