#!/usr/bin/env python
"""GIFs of a twin split, val_twins or test_twins (docs/drivetoy-v4-design.md), one per situation: the camera view of frames 0 to 31,
upscaled, then frame 31 held, alternately plain and with the cue marked in magenta (twins: the pixels where the
two twins differ) plus a zoomed inset of a small cue. Twins sit side by side, change twin left and carry-on twin
right. Under each view is the ego's speed: frames 0-31 as the clip plays, then all 16 futures in yellow. Each
sample's cue size, speed and braking are also printed as a table row.

    python scripts/twin_gifs.py <split dir> --out <dir> [--per-family 2] [--count stop_sign=5] [--tag v1]
                                    [--only name ...]
"""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

UP = 3
S = 128 * UP
LINE = 34
PLOT_H = 140
INSET = 150
PIX = 24            # twin pixels differ when one channel differs by more than this
MARK = (255, 0, 255)
GO_NOW = 33
NAME = {"red_green": "Red turns green", "green_yellow": "Green turns yellow", "queue": "Queue pulls away",
        "left_gap": "Left turn: arrow, or oncoming cars", "ped_step": "Pedestrian steps out",
        "cut_in": "Cut-in", "red_ahead": "Red light ahead", "stop_ahead": "Stop sign ahead",
        "zone_ahead": "School zone ahead", "parked_ahead": "Parked car ahead", "lane_end": "Lane ends ahead",
        "stop_sign": "All-way stop, other cars under way", "overtake": "Overtaking a slow lead"}
VERB = {"red_green": "start", "green_yellow": "slow down", "queue": "start", "left_gap": "start",
        "ped_step": "slow down", "cut_in": "slow down", "red_ahead": "slow down", "stop_ahead": "slow down",
        "zone_ahead": "slow down", "parked_ahead": "slow or steer", "lane_end": "move over or slow",
        "stop_sign": "go at once", "overtake": "pull out"}
WHAT = {"cut_in": "the indicating car", "red_ahead": "lit lamps", "stop_ahead": "the stop sign",
        "zone_ahead": "the red pavement", "parked_ahead": "the parked car", "lane_end": "the lane-ends sign",
        "stop_sign": "cars under way"}
ROLE = {"left_gap": ("arrow", "green ball"), "green_yellow": ("turns yellow", "stays green"),
        "red_green": ("turns green", "stays red"), "queue": ("turns green", "stays red"),
        "ped_step": ("steps out", "walks on"), "overtake": ("car beside moves on", "car beside stays")}
CASE = {"none": "no other car", "west_right": "car on the left turns right",
        "north_straight": "oncoming car goes straight", "east_left": "car on the right turns left",
        "cross_both": "cars on both sides go straight"}
MOVE = {0: "left", 1: "straight", 2: "right"}


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


FT, FL, FS = font(34), font(27), font(21)


def write_gif(frames, path, fps=4):
    with tempfile.TemporaryDirectory() as tmp:
        mp4 = Path(tmp) / "v.mp4"
        frames = [np.asarray(fr)[: fr.height // 2 * 2, : fr.width // 2 * 2] for fr in frames]
        imageio.mimwrite(mp4, frames, fps=fps, codec="libx264", quality=None,
                         output_params=["-crf", "6", "-pix_fmt", "yuv444p"], macro_block_size=1)
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(mp4), "-filter_complex",
                        "split[a][b];[a]palettegen=stats_mode=single[p];[b][p]paletteuse=new=1:dither=bayer:bayer_scale=4",
                        "-loop", "0", str(path)], check=True)
    print(path, f"{path.stat().st_size / 1e6:.1f} MB", flush=True)


def _render(root, n):
    with np.load(root / "render" / f"{n:06d}.npz") as r:
        return r["video"], r["index"].astype(np.int32)


def _lit(sig_g, k):
    ball, arrow = int(sig_g[0]), sig_g[1] > 0
    return arrow if k == 3 else ball == (0, 1, 2)[k]


def _zone_mask(sc, f):
    """The school-zone pavement projected into the camera at frame f (ignoring what covers it)."""
    x, y, h, _ = sc["ego"][f]
    k = sc["camera"]
    C = np.array([x + k["forward"] * np.cos(h), y + k["forward"] * np.sin(h), k["height"]])
    p = np.radians(k["pitch"])
    fwd = np.array([np.cos(p) * np.cos(h), np.cos(p) * np.sin(h), np.sin(p)])
    r = np.array([np.sin(h), -np.cos(h), 0.0])
    u = np.cross(r, fwd)
    F = 64 / np.tan(np.radians(k["fov"] / 2))
    img = Image.new("L", (128, 128), 0)
    dr = ImageDraw.Draw(img)
    for srf in sc["surfaces"]:
        if srf.get("kind") != "school":
            continue
        P = np.c_[np.asarray(srf["poly"], float), np.full(len(srf["poly"]), 0.006)]
        Q = []
        for i in range(len(P)):   # clip at 0.5 m in front of the camera
            A, B = P[i], P[(i + 1) % len(P)]
            za, zb = (A - C) @ fwd, (B - C) @ fwd
            if za >= 0.5:
                Q.append(A)
            if (za >= 0.5) != (zb >= 0.5):
                Q.append(A + (0.5 - za) / (zb - za) * (B - A))
        if len(Q) < 3:
            continue
        d = np.array(Q) - C
        z = d @ fwd
        dr.polygon([(64 + F * (q @ r) / zz, 64 - F * (q @ u) / zz) for q, zz in zip(d, z)], fill=1)
    return np.asarray(img, bool)


def _facing(items, eh):
    """Indices of signs or signal heads whose face points back at an ego heading eh."""
    return [i for i, h in enumerate(items) if np.cos(h["heading"] - (eh + np.pi)) > 0.7]


def _cue_mask(name, n, a, root, index, f=31):
    """The single clip's cue at frame f, from the render's object index (signs and heads facing the ego)."""
    ix = index[f]
    lab = np.load(root / "labels" / f"{n:06d}.npz")
    sc = json.loads((root / "scenes" / f"{n:06d}.json").read_text())
    eh = a["ego"][n, f, 2]
    if name == "red_ahead":
        info, sig = lab["head_info"], lab["signals"]
        hs = [h for h in _facing(sc["heads"], eh) if sc["heads"][h]["kind"].startswith("veh")]
        return np.isin(ix, [5000 + h * 4 + k for h in hs for k in range(4) if _lit(sig[f, info[h, 1]], k)])
    if name == "stop_ahead":
        return np.isin(ix, [3000 + i for i in _facing(sc["signs"], eh) if sc["signs"][i]["kind"] == "stop"])
    if name in ("zone_ahead", "lane_end"):
        if name == "zone_ahead":
            return _zone_mask(sc, f) & (ix == 0)
        return np.isin(ix, [3000 + i for i, s in enumerate(sc["signs"]) if s["kind"] == "lane_ends"])
    ag = a["agents"][n]
    if name == "stop_sign":   # cars under way near the ego
        slots = [s for s in range(1, ag.shape[1]) if ag[f, s, 9] > 0 and ag[f, s, 0] < 5 and ag[f, s, 4] > 0.5
                 and np.hypot(*(ag[f, s, 1:3] - a["ego"][n, f, :2])) < 35.0]
        return np.isin(ix, [s + 1 for s in slots] + [1000 + s * 4 + k for s in slots for k in range(4)])
    if name == "cut_in":
        slots = list(np.flatnonzero(a["indicator"][n, f] != 0))
    else:   # parked_ahead: cars that have not moved since frame 0
        slots = [s for s in range(ag.shape[1]) if ag[f, s, 9] > 0 and ag[f, s, 0] < 5
                 and np.abs(ag[:f + 1, s, 1:3] - ag[f, s, 1:3]).max() < 0.05]
    if not slots:
        return np.zeros_like(ix, bool)
    best = max(slots, key=lambda s: int((ix == s + 1).sum()))
    return np.isin(ix, [best + 1] + [1000 + best * 4 + k for k in range(4)])


def _outline(mask):
    m = np.pad(mask, 1)
    grown = np.zeros_like(m)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            grown |= np.roll(np.roll(m, dy, 0), dx, 1)
    return (grown & ~m)[1:-1, 1:-1]


def _view(frame, mask):
    """The upscaled frame; with a mask, a box around a small cue plus a zoomed inset, or outlines of a large one."""
    im = Image.fromarray(np.ascontiguousarray(frame)).resize((S, S), Image.NEAREST)
    if mask is None or not mask.any():
        return im
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    if rows[-1] - rows[0] >= 40 or cols[-1] - cols[0] >= 40:
        arr = np.asarray(im).copy()
        arr[np.kron(_outline(mask), np.ones((UP, UP), bool))] = MARK
        return Image.fromarray(arr)
    p = 5
    r0, r1 = max(rows[0] - p, 0), min(rows[-1] + 1 + p, 128)
    c0, c1 = max(cols[0] - p, 0), min(cols[-1] + 1 + p, 128)
    d = ImageDraw.Draw(im)
    d.rectangle([c0 * UP, r0 * UP, c1 * UP - 1, r1 * UP - 1], outline=MARK, width=3)
    crop = Image.fromarray(np.ascontiguousarray(frame[r0:r1, c0:c1]))
    z = max(1, INSET // max(r1 - r0, c1 - c0))
    crop = crop.resize(((c1 - c0) * z, (r1 - r0) * z), Image.NEAREST)
    x = S - crop.width - 8 if (c0 + c1) / 2 < 64 else 8          # the corner away from the cue
    im.paste(crop, (x, 8))
    d.rectangle([x - 3, 5, x + crop.width + 2, 8 + crop.height + 2], outline=MARK, width=3)
    d.text((x + 4, 8 + crop.height + 6), f"{z}x zoom", fill=MARK, font=FS)
    return im


def _plot(d, top, past, fut, f):
    x0, x1, y0, y1 = 46, S - 10, top + 12, top + PLOT_H - 34
    vmax = max(20.0, float(max(past.max(), fut.max() if fut is not None else 0))) * 1.1
    X = lambda t: x0 + (x1 - x0) * t / 63
    Y = lambda v: y1 - (y1 - y0) * v / vmax
    d.line([(x0, y1), (x1, y1)], fill=(100, 100, 100), width=1)
    d.line([(x0, y0), (x0, y1)], fill=(100, 100, 100), width=1)
    d.line([(X(31.5), y0), (X(31.5), y1)], fill=(255, 140, 140), width=2)
    d.text((4, y0 - 10), f"{vmax:.0f}", fill=(170, 170, 170), font=FS)
    d.text((14, y1 - 12), "0", fill=(170, 170, 170), font=FS)
    d.text((x0, y1 + 4), "ego speed (km/h), frames 0-63", fill=(170, 170, 170), font=FS)
    if fut is not None:
        for k in range(fut.shape[0]):
            d.line([(X(31), Y(past[31]))] + [(X(32 + t), Y(v)) for t, v in enumerate(fut[k])], fill=(255, 220, 60),
                   width=2)
    pts = [(X(t), Y(past[t])) for t in range(f + 1)]
    d.line(pts if len(pts) > 1 else pts * 2, fill=(255, 255, 255), width=3)


def panel(frame, lines, past, fut, f, mask):
    head = len(lines) * LINE + 8
    g = Image.new("RGB", (S, head + S + PLOT_H), (25, 25, 25))
    d = ImageDraw.Draw(g)
    for j, (txt, col) in enumerate(lines):
        d.text((6, 4 + j * LINE), txt, fill=col, font=FL)
    g.paste(_view(frame, mask), (0, head))
    _plot(d, head + S, past, fut, f)
    return g


def _cue_gap(name, n, a, mask_slot, f=31):
    """Bumper gap (m) from the ego to the cue car at frame f."""
    ag = a["agents"][n, f, mask_slot]
    return float(np.hypot(*(ag[1:3] - a["ego"][n, f, :2])) - (ag[6] + 4.5) / 2)


def _cue_slot(name, n, a, index, f=31):
    """The cut-in or parked car the cue mask marks."""
    ix, ag = index[f], a["agents"][n]
    if name == "cut_in":
        slots = list(np.flatnonzero(a["indicator"][n, f] != 0))
    else:
        slots = [s for s in range(ag.shape[1]) if ag[f, s, 9] > 0 and ag[f, s, 0] < 5
                 and np.abs(ag[:f + 1, s, 1:3] - ag[f, s, 1:3]).max() < 0.05]
    return max(slots, key=lambda s: int((ix == s + 1).sum())) if slots else None


def _swerve(a, fego, m):
    """The hardest sideways acceleration (m/s^2) in the sample's 16 futures: speed times turn rate."""
    h = np.concatenate([np.full((fego.shape[1], 1), a["ego"][m, 31, 2]), fego[m, :, :, 2]], 1)
    rate = np.abs(np.angle(np.exp(1j * np.diff(h, axis=1)))) * 5.0
    return float((rate * fego[m, :, :, 3]).max())


def _braking(a, fego, m):
    """The hardest braking (m/s^2) in the sample's 16 futures."""
    v = np.concatenate([np.full((fego.shape[1], 1), a["ego"][m, 31, 3]), fego[m, :, :, 3]], 1)
    return float(max(0.0, -np.diff(v, axis=1).min() * 5.0))


def _stop_line(a, m):
    """Which moves go at once and which wait, from the futures' moves and onsets."""
    mv, on = a["future_move"][m], a["onset"][m]
    go = sorted({int(x) for x, o in zip(mv, on) if 0 <= o <= GO_NOW})
    wait = sorted({int(x) for x in mv if x >= 0} - set(go))
    return ("go at once: " + (", ".join(MOVE[x] for x in go) or "none"),
            "wait: " + (", ".join(MOVE[x] for x in wait) or "none"))


def main(args):
    root = Path(args.split)
    a = np.load(root / "annotations.npz")
    fego = np.load(root / "future_set.npz")["ego"]
    meta = json.loads((root / "meta.json").read_text())
    fams = meta["families"]
    cases = meta.get("stop_cases", [])
    fam = [fams[i] for i in a["family"]]
    twin = a["twin"]
    counts = {k: int(v) for k, v in (c.split("=") for c in args.count)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print("| Sample | Situation | Role or case | Speed at 31 (km/h) | Cue at 31 (px) | Twins differ (px) | Hardest braking (m/s^2) | Futures |")
    print("|---|---|---|---|---|---|---|---|")
    for name in fams:
        if args.only and name not in args.only:
            continue
        firsts = [n for n in range(len(fam)) if fam[n] == name and twin[n] in (0, -1)][: counts.get(name, args.per_family)]
        if not firsts:
            continue
        views = []
        for n in firsts:
            members = [n, n + 1] if twin[n] == 0 else [n]
            rend = {m: _render(root, m) for m in members}
            diff = None
            if len(members) == 2:
                diff = np.abs(rend[n][0][31].astype(int) - rend[n + 1][0][31].astype(int)).max(-1) > PIX
                masks = {m: diff for m in members}
            else:
                masks = {n: _cue_mask(name, n, a, root, rend[n][1])}
            for i, m in enumerate(members):
                br = _braking(a, fego, m)
                v31 = a["ego"][m, 31, 3] * 3.6
                if len(members) == 2:
                    role = f"{'change' if i == 0 else 'carry-on'} twin: {ROLE.get(name, ('', ''))[i]}"
                    cue = f"twins differ: {int(diff.sum())} px at frame 31"
                    fut_line = f"{int(a['change'][m].sum())} of 16 futures {VERB[name]}"
                    row = (role, "", int(diff.sum()))
                elif name == "stop_sign":
                    role = CASE.get(cases[a["variant"][m]] if cases else "", "")
                    cue = f"cars under way: {int(masks[m].sum())} px at frame 31"
                    fut_line = " | ".join(_stop_line(a, m))
                    row = (role, int(masks[m].sum()), "")
                else:
                    role = "single clip"
                    cue = f"cue: {int(masks[m].sum())} px at frame 31"
                    if name in ("cut_in", "parked_ahead"):
                        slot = _cue_slot(name, m, a, rend[m][1])
                        if slot is not None:
                            cue += f", {_cue_gap(name, m, a, slot):.0f} m ahead"
                    fut_line = f"{int(a['change'][m].sum())} of 16 futures {VERB[name]}"
                    row = (role, int(masks[m].sum()), "")
                print(f"| {m} | {name} | {row[0]} | {v31:.0f} | {row[1]} | {row[2]} | {br:.1f} | {fut_line} |")
                motion = (f"swerve up to {_swerve(a, fego, m):.1f} m/s2 sideways" if name == "parked_ahead"
                          else f"braking up to {br:.1f} m/s2")
                lines = [(f"sample {m}, {role}", (255, 255, 255)),
                         (fut_line, (255, 220, 60)),
                         (cue, (120, 220, 255)),
                         (f"speed at 31: {v31:.0f} km/h, {motion}", (200, 200, 200))]
                views.append((rend[m][0], lines, a["ego"][m, :32, 3] * 3.6, fego[m, :, :, 3] * 3.6, masks[m]))
        sub = ("change twin left, carry-on twin right; same ego frames" if twin[firsts[0]] == 0
               else f"single clips; marked: {WHAT[name]}")
        frames = []
        for f, marked in [(f, False) for f in range(32)] + [(31, j // 4 % 2 == 1) for j in range(20)]:
            held = len(frames) >= 32
            tiles = []
            for video, lines, past, fut, mask in views:
                tag = ("frame 31 held, cue marked" if marked else "frame 31 held, futures in yellow") if held \
                    else f"frame {f}"
                tiles.append(panel(video[f], lines + [(tag, (255, 140, 140) if f >= 29 else (180, 180, 180))],
                                   past, fut if held else None, f, mask if marked else None))
            w, h = tiles[0].size
            cols, title_h = 2, 2 * 40 + 12
            n_rows = -(-len(tiles) // cols)
            g = Image.new("RGB", (cols * (w + 12), title_h + n_rows * (h + 12)), (10, 10, 10))
            d = ImageDraw.Draw(g)
            d.text((8, 6), f"{NAME[name]} ({root.name})", fill=(255, 255, 255), font=FT)
            d.text((8, 48), sub, fill=(200, 200, 200), font=FL)
            for i, t in enumerate(tiles):
                g.paste(t, ((i % cols) * (w + 12), title_h + (i // cols) * (h + 12)))
            frames.append(g)
        write_gif(frames, out / f"{name}_{args.tag}.gif")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("split")
    p.add_argument("--out", required=True)
    p.add_argument("--per-family", type=int, default=2)
    p.add_argument("--count", nargs="*", default=[], help="name=n: samples of one situation")
    p.add_argument("--tag", default="v1")
    p.add_argument("--only", nargs="*")
    main(p.parse_args())
