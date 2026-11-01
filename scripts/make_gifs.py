#!/usr/bin/env python
"""GIFs of a stochastic build, each tile the ego view beside the top-down map of the same frame
(heading up, ego in white), from a re-simulation of the clip with its stored sim_seed and target:

- diversity_<aspect>_v3.gif, aspects A to E: one clip per label;
- pending_v3.gif: one pending clip per kind of choice, running on to frame 40 on the map
  (the ego view holds frame 31) to show the choice being made;
- forks_v3.gif: pending clips with four of their futures forked at frame 31, side by side,
  frames 0 to 63: identical up to frame 31, apart after it.

With --showcase it writes instead forks_<category>_v3.gif, the forks layout with larger text for
four clips of each SHOWCASE category (docs/forked-futures.md).

    python scripts/make_gifs.py /root/projects/drive_v3/val --out /root/projects/drive_v3/viz
    python scripts/make_gifs.py /root/projects/drive_v3/val --out assets --showcase
"""
import argparse
import json
import subprocess
import sys
import tempfile
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from drivetoy.choices import CHOICE_KINDS
from drivetoy.debug_view import render_topdown
from drivetoy.diversity import ASPECTS
from drivetoy.scenarios import VARIANTS, simulate

NAMES = dict(A="light", B="cause", C="speed", D="path", E="choice")
OUTCOME = dict(yellow=("stops", "goes"), left_gap=("waits", "turns"), stop_wave=("lets the others go", "goes first"),
               rtor=("waits for green", "turns on red"), overtake=("stays behind", "pulls out"),
               nudge=("waits for oncoming", "goes round now"), merge=("waits", "merges"), cutin=("stays", "cuts in"),
               ped_cross=("waits", "crosses"))
KIND = dict(yellow="yellow onset", left_gap="permissive left", stop_wave="stop-sign tie", rtor="right turn on red",
            overtake="overtake", nudge="parked car", merge="lane-end merge", cutin="cut-in", ped_cross="pedestrian")
MOVES = ["no move", "left", "straight", "right"]   # ego_turn
SHOWCASE = [   # name, the ego's pending choice in each of its four clips, or "B:<cause>" for a clip whose
    # cause is acted out on screen; None: the ego's route drawn per future
    ("yellow", ["yellow"] * 4),
    ("signal_turns", ["left_gap", "left_gap", "rtor", "rtor"]),
    ("stop_sign", ["stop_wave"] * 4),
    ("lanes", ["overtake", "overtake", "merge", "merge"]),
    ("road_users", ["B:pedestrian", "B:cut-in vehicle", "nudge", "nudge"]),
    ("route", None),
]
STOP_TEXT = ["lets the others go", "goes first, left", "goes first, straight", "goes first, right"]
OUT_COLORS = [(255, 150, 90), (110, 220, 120), (120, 180, 255), (230, 140, 230)]   # per outcome or move
ROOT = META = None
S = 256
T0, T1 = 31, 40


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


FONT, FONT_L = _font(13), _font(20)


def resim(n, forks=False):
    """(map, own record, fork records or None) of sample n, re-simulated."""
    a = np.load(ROOT / "labels" / f"{n:06d}.npz")
    tgt = {k: (ASPECTS[k][c] if c >= 0 else None) for k, c in zip("ABCDE", a["target"])}
    tgt["template"] = META["templates"][int(a["template"])]
    tgt["V"] = VARIANTS[int(a["variant"])] if int(a["variant"]) >= 0 else None
    cfg = dict(META["cfg"], DT_KEEP_FORK_RECS=forks)
    r = simulate(int(a["sim_seed"]), cfg, target=tgt)
    b = int(a["branch_id"])
    return r["map"], r["branches"][b], (r["fork_recs"][b] if forks else None)


def topdown(m, rec, frames):
    return [render_topdown(m, rec, f, size=S, span=80.0)[:, :, ::-1] for f in frames]


def job_plain(n):
    m, rec, _ = resim(n)
    return n, np.stack(topdown(m, rec, range(T1 + 1)))


def pending_choice(d):
    """The ego's first choice pending at frame 31 (made in frames 32 to 40, drawn with odds 0.1 to
    0.9, or its determined response to a yellow), or None."""
    d = d[(d[:, 0] > T0) & (d[:, 0] <= T1) & (d[:, 1] == 0) & (d[:, 6] > 0)
          & (((d[:, 4] >= 0.1) & (d[:, 4] <= 0.9)) | (d[:, 2] == CHOICE_KINDS.index("yellow")))]
    return d[0] if len(d) else None


def pick_futures(outs, k=4):
    """Future 0 first, then the others spread over the outcomes, two of each where possible."""
    pick, rest = [0], list(range(1, len(outs)))
    while len(pick) < min(k, len(outs)):
        shown = Counter(outs[p] for p in pick)
        j = min(rest, key=lambda j: (shown[outs[j]], j))
        pick.append(j)
        rest.remove(j)
    return pick


def job_forks(args):
    """Sample n's futures: their outcomes of the choice (slot, kind), or the outcomes given, and the
    maps of four of them."""
    n, slot, kind, outs = args
    m, rec, frs = resim(n, forks=True)
    if outs is None:
        outs = []
        for fr in frs:
            d = fr["decisions"]
            hit = d[(d[:, 1] == slot) & (d[:, 2] == kind) & (d[:, 0] > T0)]
            o = int(hit[0, 5]) if len(hit) else -1
            if o == 1 and CHOICE_KINDS[kind] == "stop_wave":
                o = int(fr["ego_turn"])   # goes first: 1 left, 2 straight, 3 right
            outs.append(o)
    pick = pick_futures(outs)
    return n, pick, outs, {k: np.stack(topdown(m, frs[k], range(64))) for k in pick}


def choice_title(kind, outs):
    n = len(outs)
    if kind == "yellow":
        if len(set(outs)) == 1:
            return f"{KIND[kind]}: {OUTCOME[kind][outs[0]]} in all {n} futures"
        return f"{KIND[kind]}: goes in {outs.count(1)} of {n} futures, stops in {outs.count(0)}"
    if kind == "stop_wave":
        c = Counter(outs)
        return f"{KIND[kind]} ({n} futures): others first {c[0]}, left {c[1]}, straight {c[2]}, right {c[3]}"
    return f"{KIND[kind]}: {OUTCOME[kind][1]} in {outs.count(1)} of {n} futures, {OUTCOME[kind][0]} in {outs.count(0)}"


def tile_text(kind, o):
    if o < 0:
        return "no choice"
    return STOP_TEXT[o] if kind == "stop_wave" else OUTCOME[kind][o]


def route_title(outs):
    parts = [f"{MOVES[t]} in {c}" for t, c in sorted(Counter(outs).items())]
    return f"ego's route: {parts[0]} of {len(outs)} futures, " + ", ".join(parts[1:])


def frame_of(vid, n, f):
    im = Image.fromarray(np.ascontiguousarray(vid[n, :, min(f, 31)].transpose(1, 2, 0))).resize((S, S), Image.NEAREST)
    if f > 31:   # label-only frames: the ego view holds its last frame
        im = Image.blend(im, Image.new("RGB", im.size, (0, 0, 0)), 0.5)
    return im


def write_gif(frames, path, fps=5):
    """Through ffmpeg with a palette per frame, so every tile keeps its colours."""
    with tempfile.TemporaryDirectory() as tmp:
        mp4 = Path(tmp) / "v.mp4"
        frames = [np.asarray(fr)[: fr.height // 2 * 2, : fr.width // 2 * 2] for fr in frames]
        imageio.mimwrite(mp4, frames, fps=fps, codec="libx264", quality=None,
                         output_params=["-crf", "6", "-pix_fmt", "yuv444p"], macro_block_size=1)
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(mp4), "-filter_complex",
                        "split[a][b];[a]palettegen=stats_mode=single[p];[b][p]paletteuse=new=1:dither=bayer:bayer_scale=4",
                        "-loop", "0", str(path)], check=True)
    print(path, f"{path.stat().st_size / 1e6:.1f} MB", flush=True)


def grid(tiles, cols, n_frames, cap_lines, draw_tile):
    W, CAP = 2 * S + 2, 16 * cap_lines + 4
    rows = -(-len(tiles) // cols)
    out = []
    for f in range(n_frames):
        g = Image.new("RGB", (cols * W + (cols - 1) * 6, rows * (S + CAP) + (rows - 1) * 6), (25, 25, 25))
        d = ImageDraw.Draw(g)
        for t, tile in enumerate(tiles):
            x, y = (t % cols) * (W + 6), (t // cols) * (S + CAP + 6)
            draw_tile(g, d, x, y, CAP, tile, f)
        out.append(g)
    return out


def forks_gif(vid, rows, path, font, line):
    """One row per (n, title, tiles, tops): the ego view, then the maps of four futures forked at
    frame 31, frames 0 to 63; tiles holds (future, outcome, outcome text)."""
    cap = 3 * line + 4
    frames = []
    for f in range(64):
        g = Image.new("RGB", (5 * S + 4 * 4, len(rows) * (S + cap) + (len(rows) - 1) * 8), (25, 25, 25))
        d = ImageDraw.Draw(g)
        for r, (n, title, tiles, tops) in enumerate(rows):
            y = r * (S + cap + 8)
            d.text((4, y + 2), f"{title}   |   frame {f}"
                   + ("  (label-only; the futures forked at frame 31)" if f > 31 else "  (shared by every future)"),
                   fill=(255, 220, 60), font=font)
            d.text((4, y + 2 + line), f"ego view, sample {n}", fill=(200, 200, 200), font=font)
            g.paste(frame_of(vid, n, f), (0, y + cap))
            for c, (k, o, txt) in enumerate(tiles):
                x = (c + 1) * (S + 4)
                g.paste(Image.fromarray(np.ascontiguousarray(tops[k][f])), (x, y + cap))
                d.text((x + 4, y + 2 + line), f"future {k}" + (" (label)" if k == 0 else ""), fill=(200, 200, 200), font=font)
                d.text((x + 4, y + 2 + 2 * line), txt, fill=OUT_COLORS[o] if o >= 0 else (200, 200, 200), font=font)
        frames.append(g)
    write_gif(frames + [frames[-1]] * 5, path, fps=4)


def showcase(ann, vid, out, rng, suffix):
    """forks_<name>_<suffix>.gif for each SHOWCASE category, four clips each, with larger text."""
    div, br, dec = np.asarray(ann["diversity"]), np.asarray(ann["branch_id"]), np.asarray(ann["decisions"])
    turns = np.asarray(np.load(ROOT / "future_set.npz", mmap_mode="r")["ego_turn"])
    B, D = META["diversity"]["B"], META["diversity"]["D"]
    cands = {}   # kind -> (n, slot, kind, None) of the samples whose first pending choice is of that kind
    for n in rng.permutation(np.flatnonzero(br == 0)):
        d = pending_choice(dec[n])
        if d is not None:
            cands.setdefault(CHOICE_KINDS[int(d[2])], []).append((int(n), int(d[1]), int(d[2]), None))
    decided = {}   # cause -> a clip in which it is acted out on screen: every future goes on alike
    for want in [k[2:] for _, ks in SHOWCASE if isinstance(ks, list) for k in ks if k.startswith("B:")]:
        pool = [int(n) for n in rng.permutation(np.flatnonzero((br == 0) & (div[:, 1] == B.index(want)) & (div[:, 4] != 0)
                                                             & ~np.asarray(ann["route_drawn"], bool)))]   # its route not drawn per future
        if pool:
            decided[want] = pool[0]
    routed = [int(n) for n in rng.permutation(np.flatnonzero(np.asarray(ann["route_drawn"]) & (br == 0)))]
    moves = {n: len(set(turns[n].tolist())) for n in routed}
    route = [n for n in routed if moves[n] == 3][:2] + [n for n in routed if moves[n] == 2][:2]
    jobs = [(n, -1, -1, [int(t) for t in turns[n]]) for n in route] + \
        [(n, -2, -1, [-1] * turns.shape[1]) for n in decided.values()]
    for _, kinds in SHOWCASE:
        if isinstance(kinds, list):
            for kind, c in Counter(k for k in kinds if not k.startswith("B:")).items():
                jobs += cands.get(kind, [])[:3 * c]   # spares
    with Pool(64) as p:
        res = dict(zip([(j[0], j[1] if j[1] < 0 else j[2]) for j in jobs], p.map(job_forks, jobs)))
    for name, kinds in SHOWCASE:
        rows = []
        if kinds is None:
            for n in route:
                _, pick, outs, tops = res[(n, -1)]
                rows.append((n, route_title(outs), [(k, outs[k], MOVES[outs[k]]) for k in pick], tops))
        else:
            used, seen = set(), Counter()
            for kind in kinds:
                if kind.startswith("B:"):
                    if kind[2:] in decided:
                        n = decided[kind[2:]]
                        _, pick, outs, tops = res[(n, -2)]
                        rows.append((n, f"{kind[2:]}, acted out on screen: the futures go on alike",
                                     [(k, -1, f"future {k}") for k in pick], tops))
                    continue
                pool = [j for j in cands.get(kind, [])[:3 * kinds.count(kind)] if j[0] not in used]
                if kind == "yellow":   # as many that stop as that go
                    pool.sort(key=lambda j: seen[res[(j[0], j[2])][2][0]])
                else:                  # futures with two or more outcomes first
                    pool.sort(key=lambda j: len({o for o in res[(j[0], j[2])][2] if o >= 0}) < 2)
                if not pool:
                    continue
                n, _, k, _ = pool[0]
                used.add(n)
                _, pick, outs, tops = res[(n, k)]
                seen[outs[0]] += 1
                rows.append((n, choice_title(kind, outs), [(f, outs[f], tile_text(kind, outs[f])) for f in pick], tops))
        forks_gif(vid, rows, out / f"forks_{name}_{suffix}.gif", FONT_L, 26)


def main():
    global ROOT, META
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--forks", type=int, default=4, help="pending clips in the forks GIF")
    ap.add_argument("--suffix", default="v3", help="file name suffix")
    ap.add_argument("--showcase", action="store_true", help="write only the forks_<category> GIFs")
    args = ap.parse_args()
    ROOT, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    META = json.loads((ROOT / "meta.json").read_text())
    ann = np.load(ROOT / "annotations.npz", mmap_mode="r")
    vid = np.load(ROOT / "videos.npy", mmap_mode="r")
    rng = np.random.default_rng(args.seed)
    if args.showcase:
        return showcase(ann, vid, out, rng, args.suffix)
    div, br, dec = np.asarray(ann["diversity"]), np.asarray(ann["branch_id"]), np.asarray(ann["decisions"])

    # diversity: one clip per label of each aspect
    picks = {}
    for i, key in enumerate("ABCDE"):
        picks[key] = []
        for j, lab in enumerate(META["diversity"][key]):
            pool = np.flatnonzero((div[:, i] == j) & (br == 0))
            if len(pool):
                picks[key].append((lab, int(rng.choice(pool))))
    # pending: one clip per kind of pending choice
    pend = {}
    for n in rng.permutation(np.flatnonzero((div[:, 4] == 0) & (br == 0))):
        d = pending_choice(dec[n])
        if d is not None and CHOICE_KINDS[int(d[2])] not in pend:
            pend[CHOICE_KINDS[int(d[2])]] = (int(n), int(d[1]), int(d[2]), float(d[0]), float(d[4]), float(d[3]))
    with Pool(64) as p:
        tops = dict(p.map(job_plain, sorted({n for v in picks.values() for _, n in v} | {v[0] for v in pend.values()})))
        fork_jobs = [(v[0], v[1], v[2], None) for v in list(pend.values())[:args.forks]]
        forks = p.map(job_forks, fork_jobs)

    for i, key in enumerate("ABCDE"):
        def draw(g, d, x, y, CAP, tile, f, key=key):
            lab, n = tile
            g.paste(frame_of(vid, n, f), (x, y + CAP))
            g.paste(Image.fromarray(np.ascontiguousarray(tops[n][f])), (x + S + 2, y + CAP))
            rest = " / ".join(META["diversity"][k][div[n, q]] for q, k in enumerate("ABCDE") if k != key)
            d.text((x + 4, y + 2), f"{NAMES[key]}: {lab}", fill=(255, 220, 60), font=FONT)
            d.text((x + 4, y + 18), rest, fill=(200, 200, 200), font=FONT)
        write_gif(grid(picks[key], 3, 32, 2, draw), out / f"diversity_{key}_{NAMES[key]}_{args.suffix}.gif")

    def draw_p(g, d, x, y, CAP, tile, f):
        kind, (n, slot, k, fr, pr, cue) = tile
        g.paste(frame_of(vid, n, f), (x, y + CAP))
        g.paste(Image.fromarray(np.ascontiguousarray(tops[n][f])), (x + S + 2, y + CAP))
        d.text((x + 4, y + 2), f"pending: {kind} (agent {slot}), chosen at frame {int(fr)}", fill=(255, 220, 60), font=FONT)
        d.text((x + 4, y + 18), f"cue {cue:.2f}, p({OUTCOME[kind][1]}) = {pr:.2f}" + ("   [label-only frame]" if f > 31 else ""),
               fill=(255, 120, 120) if f > 31 else (200, 200, 200), font=FONT)
    write_gif(grid(list(pend.items()), 3, T1 + 1, 2, draw_p), out / f"pending_{args.suffix}.gif")

    # forks: per pending clip, the ego view and four futures forked at frame 31
    rows = []
    for (n, _, k, _), (_, pick, outs, tp) in zip(fork_jobs, forks):
        kind = CHOICE_KINDS[k]
        rows.append((n, choice_title(kind, outs), [(f, outs[f], tile_text(kind, outs[f]) if outs[f] >= 0 else "") for f in pick], tp))
    forks_gif(vid, rows, out / f"forks_{args.suffix}.gif", FONT, 16)


if __name__ == "__main__":
    main()
