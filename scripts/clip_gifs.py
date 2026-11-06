#!/usr/bin/env python
"""GIFs of DriveToy v4 clips (docs/drivetoy-v4-design.md): a grid of clips, each playing frames 0 to 31 with its
decision frame marked (the frame label turns red there and the clip pauses on it), and under each the ego's
speed over frames 0 to 63 with the decision frame as a red line.

    python scripts/clip_gifs.py <split dir> --out <dir> --tag v1 [--per-gif 8] [--families name ...] [--per-family k]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from twin_gifs import FL, FS, FT, S, UP, write_gif

PLOT_H, LINE = 150, 32
NICE = {0: "change", 1: "carry-on"}


def _plot(d, top, v, fd, f):
    x0, x1, y0, y1 = 46, S - 10, top + 12, top + PLOT_H - 36
    vmax = max(20.0, float(v.max())) * 1.1
    X = lambda t: x0 + (x1 - x0) * t / 63
    Y = lambda s: y1 - (y1 - y0) * s / vmax
    d.line([(x0, y1), (x1, y1)], fill=(100, 100, 100), width=1)
    d.line([(x0, y0), (x0, y1)], fill=(100, 100, 100), width=1)
    d.line([(X(31.5), y0), (X(31.5), y1)], fill=(120, 120, 120), width=1)
    if fd >= 0:
        d.line([(X(fd), y0), (X(fd), y1)], fill=(255, 80, 80), width=3)
    d.text((4, y0 - 10), f"{vmax:.0f}", fill=(170, 170, 170), font=FS)
    d.text((14, y1 - 12), "0", fill=(170, 170, 170), font=FS)
    d.text((x0, y1 + 4), "ego km/h, frames 0-63 (red: decision)", fill=(170, 170, 170), font=FS)
    d.line([(X(t), Y(v[t])) for t in range(32, 64)], fill=(255, 220, 60), width=2)
    pts = [(X(t), Y(v[t])) for t in range(f + 1)]
    d.line(pts if len(pts) > 1 else pts * 2, fill=(255, 255, 255), width=3)


def main(args):
    root = Path(args.split)
    a = np.load(root / "annotations.npz")
    fams = json.loads((root / "meta.json").read_text())["families"]
    fam = [fams[i] for i in a["family"]]
    fd = a["decision_frame"].astype(int)
    order = [n for n in range(len(fam)) if not args.families or fam[n] in args.families]
    if args.per_family:   # k clips per situation, distinct decision frames first
        rng = np.random.RandomState(args.seed)
        pick = []
        for name in sorted(set(fam[n] for n in order), key=fams.index):
            ids = [n for n in order if fam[n] == name]
            rng.shuffle(ids)
            chosen, seen = [], set()
            for n in ids:
                if len(chosen) < args.per_family and fd[n] not in seen:
                    chosen.append(n)
                    seen.add(fd[n])
            chosen += [n for n in ids if n not in chosen][:args.per_family - len(chosen)]
            pick += sorted(chosen)
        order = pick
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for g in range(0, len(order), args.per_gif):
        clips = order[g:g + args.per_gif]
        vids = {n: np.load(root / "render" / f"{n:06d}.npz")["video"] for n in clips}
        seqs = {}
        for n in clips:   # frames 0 to 31, pausing six steps on the decision frame
            seq = []
            for f in range(32):
                seq += [f] * (7 if f == fd[n] else 1)
            seqs[n] = seq
        frames = []
        for step in range(max(len(q) for q in seqs.values()) + 8):
            tiles = []
            for n in clips:
                d_n = fd[n]
                f = seqs[n][min(step, len(seqs[n]) - 1)]
                lines = [(f"sample {n}: {fam[n]}" + (f" ({NICE.get(int(a['twin'][n]), '')})" if a["twin"][n] >= 0 else ""),
                          (255, 255, 255)),
                         ("no decision" if d_n < 0 else f"decision at frame {d_n}, change from {d_n + 1}",
                          (255, 220, 60)),
                         (f"frame {f}", (255, 80, 80) if d_n >= 0 and f == d_n else (190, 190, 190))]
                head = len(lines) * LINE + 8
                t = Image.new("RGB", (S, head + S + PLOT_H), (25, 25, 25))
                d = ImageDraw.Draw(t)
                for j, (txt, col) in enumerate(lines):
                    d.text((6, 4 + j * LINE), txt, fill=col, font=FL)
                im = Image.fromarray(vids[n][f]).resize((S, S), Image.NEAREST)
                if d_n >= 0 and f == d_n:
                    ImageDraw.Draw(im).rectangle([0, 0, S - 1, S - 1], outline=(255, 80, 80), width=6)
                t.paste(im, (0, head))
                _plot(d, head + S, a["ego"][n, :, 3] * 3.6, d_n, f)
                tiles.append(t)
            w, h = tiles[0].size
            cols = 4
            rows = -(-len(tiles) // cols)
            G = Image.new("RGB", (cols * (w + 10), 60 + rows * (h + 10)), (10, 10, 10))
            dd = ImageDraw.Draw(G)
            dd.text((8, 10), f"DriveToy v4 ({root.name}): each clip pauses on its decision frame (red)", fill=(255, 255, 255), font=FT)
            for i, t in enumerate(tiles):
                G.paste(t, ((i % cols) * (w + 10), 60 + (i // cols) * (h + 10)))
            frames.append(G)
        write_gif(frames, out / f"v4_{root.name}_{g // args.per_gif:02d}_{args.tag}.gif")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("split")
    p.add_argument("--out", required=True)
    p.add_argument("--tag", default="v1")
    p.add_argument("--per-gif", type=int, default=8)
    p.add_argument("--families", nargs="*")
    p.add_argument("--per-family", type=int, default=0, help="k clips per situation (0: all)")
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
