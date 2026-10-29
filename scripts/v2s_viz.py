#!/usr/bin/env python
"""GIFs of a stochastic build, each tile the ego view beside the top-down map of the same frame
(heading up, ego in white), from a re-simulation of the clip with its stored sim_seed and target:

- diversity_<aspect>_v2s.gif, aspects A to E: one clip per label;
- pending_v2s.gif: one pending clip per kind of choice, running on to frame 40 on the map
  (the ego view holds frame 31) to show the choice being made;
- forks_v2s.gif: pending clips with four of their futures forked at frame 31, side by side,
  frames 0 to 63: identical up to frame 31, apart after it.

    python scripts/v2s_viz.py /root/projects/drive_v2s/val --out /root/projects/drive_v2s/viz
"""
import argparse
import json
import subprocess
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from f_toy.datasets.drivetoy.choices import CHOICE_KINDS
from f_toy.datasets.drivetoy.debug_view import render_topdown
from f_toy.datasets.drivetoy.diversity import ASPECTS
from f_toy.datasets.drivetoy.scenarios import VARIANTS, simulate

NAMES = dict(A="light", B="cause", C="speed", D="path", E="choice")
OUTCOME = dict(yellow=("stop", "go"), left_gap=("waits", "turns"), stop_wave=("waves through", "goes first"),
               rtor=("waits for green", "turns on red"), overtake=("stays behind", "pulls out"),
               nudge=("waits", "goes around"), merge=("waits", "merges"), cutin=("stays", "cuts in"),
               ped_cross=("waits", "crosses"))
ROOT = META = None
S = 256
T0, T1 = 31, 40
try:
    FONT = ImageFont.load_default(size=13)
except TypeError:
    FONT = ImageFont.load_default()


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


def job_forks(args):
    n, slot, kind = args
    m, rec, frs = resim(n, forks=True)
    outs = []
    for k, fr in enumerate(frs):
        d = fr["decisions"]
        hit = d[(d[:, 1] == slot) & (d[:, 2] == kind) & (d[:, 0] > T0)]
        outs.append(int(hit[0, 5]) if len(hit) else -1)
    # two futures of each outcome where possible, future 0 first
    pick = [0] + [k for k in range(1, len(frs)) if outs[k] != outs[0]][:2]
    pick += [k for k in range(1, len(frs)) if k not in pick][:4 - len(pick)]
    tops = {k: np.stack(topdown(m, frs[k], range(64))) for k in pick}
    return n, pick, [outs[k] for k in pick], outs, tops


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


def main():
    global ROOT, META
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--forks", type=int, default=4, help="pending clips in forks_v2s.gif")
    ap.add_argument("--suffix", default="v2s", help="file name suffix, e.g. v3")
    args = ap.parse_args()
    ROOT, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    META = json.loads((ROOT / "meta.json").read_text())
    ann = np.load(ROOT / "annotations.npz", mmap_mode="r")
    vid = np.load(ROOT / "videos.npy", mmap_mode="r")
    div, br, dec = np.asarray(ann["diversity"]), np.asarray(ann["branch_id"]), np.asarray(ann["decisions"])
    rng = np.random.default_rng(args.seed)

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
        d = dec[n]
        d = d[(d[:, 0] > T0) & (d[:, 0] <= T1) & (d[:, 6] > 0) & (d[:, 4] >= 0.1) & (d[:, 4] <= 0.9)]
        if len(d) and CHOICE_KINDS[int(d[0, 2])] not in pend:
            pend[CHOICE_KINDS[int(d[0, 2])]] = (int(n), int(d[0, 1]), int(d[0, 2]), float(d[0, 0]), float(d[0, 4]), float(d[0, 3]))
    with Pool(64) as p:
        tops = dict(p.map(job_plain, sorted({n for v in picks.values() for _, n in v} | {v[0] for v in pend.values()})))
        fork_jobs = [(v[0], v[1], v[2]) for v in list(pend.values())[:args.forks]]
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
    W, CAP = S, 34
    frames = []
    for f in range(64):
        g = Image.new("RGB", (5 * W + 4 * 4, len(forks) * (S + CAP) + (len(forks) - 1) * 8), (25, 25, 25))
        d = ImageDraw.Draw(g)
        for r, (n, pick, po, outs, tp) in enumerate(forks):
            y = r * (S + CAP + 8)
            kind = CHOICE_KINDS[[j for j in fork_jobs if j[0] == n][0][2]]
            n1, n0 = sum(o == 1 for o in outs), sum(o == 0 for o in outs)
            d.text((4, y + 2), f"{kind}: {OUTCOME[kind][1]} in {n1} of {len(outs)} futures, {OUTCOME[kind][0]} in {n0}   |   frame {f}"
                   + ("  (label-only; the futures forked at frame 31)" if f > 31 else "  (shared by every future)"),
                   fill=(255, 220, 60), font=FONT)
            d.text((4, y + 18), "ego view, frames 0-31", fill=(200, 200, 200), font=FONT)
            g.paste(frame_of(vid, n, f), (0, y + CAP))
            for c, (k, o) in enumerate(zip(pick, po)):
                xx = (c + 1) * (W + 4)
                g.paste(Image.fromarray(np.ascontiguousarray(tp[k][f])), (xx, y + CAP))
                lab = f"future {k}" + (" (label)" if k == 0 else "") + (f": {OUTCOME[kind][o]}" if o >= 0 else "")
                d.text((xx + 4, y + 18), lab, fill=(255, 120, 120) if f > 31 else (200, 200, 200), font=FONT)
        frames.append(g)
    write_gif(frames + [frames[-1]] * 5, out / f"forks_{args.suffix}.gif", fps=4)


if __name__ == "__main__":
    main()
