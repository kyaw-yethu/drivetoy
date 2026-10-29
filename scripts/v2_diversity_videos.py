#!/usr/bin/env python
"""One MP4 per diversity aspect (A light, B cause, C speed profile, D path): a grid with one
clip per label, drawn at random from a split, each tile captioned with the clip's labels. Each
tile shows the ego view next to the top-down map of the same frame (heading up, ego in white),
drawn from a re-simulation of the clip with its stored sim_seed and target.

    python scripts/v2_diversity_videos.py /root/projects/drive_v2/val --out /root/projects/drive_v2/viz
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f_toy.datasets.drivetoy.debug_view import render_topdown
from f_toy.datasets.drivetoy.diversity import ASPECTS
from f_toy.datasets.drivetoy.scenarios import simulate

NAMES = dict(A="light", B="cause", C="speed", D="path")
ROOT, META, SIZE = None, None, 256


def topdown(n):
    """Top-down frames (T, SIZE, SIZE, 3) RGB of sample n, re-simulated."""
    a = np.load(ROOT / "labels" / f"{n:06d}.npz")
    tgt = {k: ASPECTS[k][c] for k, c in zip("ABCD", a["target"])}
    tgt["template"] = META["templates"][int(a["template"])]
    r = simulate(int(a["sim_seed"]), META["cfg"], target=tgt)
    rec = r["branches"][int(a["branch_id"])]
    return n, np.stack([render_topdown(r["map"], rec, f, size=SIZE, span=80.0)[:, :, ::-1]
                        for f in range(META["frames"])])


def main():
    global ROOT, META, SIZE
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--suffix", default="", help="appended to each file name, e.g. _rev")
    args = ap.parse_args()
    ROOT, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    META = meta = json.loads((ROOT / "meta.json").read_text())
    SIZE = S = 128 * args.scale
    ann = np.load(ROOT / "annotations.npz", mmap_mode="r")
    vid = np.load(ROOT / "videos.npy", mmap_mode="r")
    div, br = np.asarray(ann["diversity"]), np.asarray(ann["branch_id"])
    rng = np.random.default_rng(args.seed)
    try:
        font = ImageFont.load_default(size=13)
    except TypeError:
        font = ImageFont.load_default()
    W, CAP = 2 * S + 2, 36          # ego view | map, caption above
    picks = {}
    for i, key in enumerate("ABCD"):
        picks[key] = []
        for j, lab in enumerate(meta["diversity"][key]):
            pool = np.flatnonzero((div[:, i] == j) & (br == 0))
            if len(pool):
                picks[key].append((lab, int(rng.choice(pool))))
    with Pool(64) as p:
        maps = dict(p.map(topdown, sorted({n for v in picks.values() for _, n in v})))
    for i, key in enumerate("ABCD"):
        cols = min(args.cols, len(picks[key]))
        rows = -(-len(picks[key]) // cols)
        frames = []
        for f in range(vid.shape[2]):
            g = Image.new("RGB", (cols * W + (cols - 1) * 6, rows * (S + CAP) + (rows - 1) * 6), (25, 25, 25))
            d = ImageDraw.Draw(g)
            for t, (lab, n) in enumerate(picks[key]):
                x, y = (t % cols) * (W + 6), (t // cols) * (S + CAP + 6)
                im = Image.fromarray(np.ascontiguousarray(vid[n, :, f].transpose(1, 2, 0))).resize((S, S), Image.NEAREST)
                g.paste(im, (x, y + CAP))
                g.paste(Image.fromarray(np.ascontiguousarray(maps[n][f])), (x + S + 2, y + CAP))
                rest = " / ".join(meta["diversity"][k][div[n, q]] for q, k in enumerate("ABCD") if k != key)
                d.text((x + 4, y + 2), f"{NAMES[key]}: {lab}", fill=(255, 220, 60), font=font)
                d.text((x + 4, y + 18), rest, fill=(200, 200, 200), font=font)
            frames.append(g)
        # MP4, not GIF: a 256-colour GIF palette shared by the whole grid turns the trees brown
        path = out / f"diversity_{key}_{NAMES[key]}{args.suffix}.mp4"
        frames = [np.asarray(fr)[: fr.height // 2 * 2, : fr.width // 2 * 2] for fr in frames]
        imageio.mimwrite(path, frames, fps=5, codec="libx264", quality=None,
                         output_params=["-crf", "8", "-pix_fmt", "yuv420p"], macro_block_size=1)
        print(path, len(picks[key]), "labels", f"{path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
