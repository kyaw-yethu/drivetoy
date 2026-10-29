#!/usr/bin/env python
"""A grid video (MP4 and GIF) of example clips for chosen labels: one row per label, ``--n``
random clips each, every tile the ego view next to the re-simulated top-down map.

    python scripts/v2_label_videos.py /root/projects/drive_v2/val --out viz/cutin_lanechange.mp4 \
        --label "B=cut-in vehicle" --label "D=lane change" --n 3
"""
import argparse, json, subprocess, sys
from multiprocessing import Pool
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from f_toy.datasets.drivetoy.debug_view import render_topdown
from f_toy.datasets.drivetoy.diversity import ASPECTS
from f_toy.datasets.drivetoy.scenarios import simulate

ROOT, META, S = None, None, 256


def topdown(n):
    a = np.load(ROOT / "labels" / f"{n:06d}.npz")
    tgt = {k: ASPECTS[k][c] for k, c in zip("ABCD", a["target"])}
    tgt["template"] = META["templates"][int(a["template"])]
    r = simulate(int(a["sim_seed"]), META["cfg"], target=tgt)
    rec = r["branches"][int(a["branch_id"])]
    return n, np.stack([render_topdown(r["map"], rec, f, size=S, span=80.0)[:, :, ::-1] for f in range(META["frames"])])


def main():
    global ROOT, META
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", action="append", required=True, help='aspect=label, e.g. "D=lane change"')
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ROOT = Path(args.root)
    META = json.loads((ROOT / "meta.json").read_text())
    ann = np.load(ROOT / "annotations.npz")
    div, br = ann["diversity"], ann["branch_id"]
    rng = np.random.default_rng(args.seed)
    rows = []
    for spec in args.label:
        k, lab = spec.split("=", 1)
        pool = np.flatnonzero((div[:, "ABCD".index(k)] == ASPECTS[k].index(lab)) & (br == 0))
        rows.append((f"{k}: {lab}", [int(x) for x in rng.choice(pool, min(args.n, len(pool)), replace=False)]))
    with Pool(32) as p:
        maps = dict(p.map(topdown, sorted({n for _, ns in rows for n in ns})))
    vid = np.load(ROOT / "videos.npy", mmap_mode="r")
    try:
        font = ImageFont.load_default(size=13)
    except TypeError:
        font = ImageFont.load_default()
    W, CAP, cols = 2 * S + 2, 36, max(len(ns) for _, ns in rows)
    frames = []
    for f in range(vid.shape[2]):
        g = Image.new("RGB", (cols * W + (cols - 1) * 6, len(rows) * (S + CAP) + (len(rows) - 1) * 6), (25, 25, 25))
        d = ImageDraw.Draw(g)
        for r, (title, ns) in enumerate(rows):
            for c, n in enumerate(ns):
                x, y = c * (W + 6), r * (S + CAP + 6)
                g.paste(Image.fromarray(np.ascontiguousarray(vid[n, :, f].transpose(1, 2, 0))).resize((S, S), Image.NEAREST), (x, y + CAP))
                g.paste(Image.fromarray(np.ascontiguousarray(maps[n][f])), (x + S + 2, y + CAP))
                d.text((x + 4, y + 2), f"{title}   (val {n})", fill=(255, 220, 60), font=font)
                d.text((x + 4, y + 18), " / ".join(ASPECTS[k][div[n, q]] for q, k in enumerate("ABCD")), fill=(200, 200, 200), font=font)
        frames.append(np.asarray(g)[: g.height // 2 * 2, : g.width // 2 * 2])
    out = Path(args.out)
    imageio.mimwrite(out, frames, fps=5, codec="libx264", quality=None, output_params=["-crf", "8", "-pix_fmt", "yuv420p"], macro_block_size=1)
    import imageio_ffmpeg
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", "-i", str(out), "-vf",
                    "fps=5,split[a][b];[a]palettegen=max_colors=256:stats_mode=full[p];[b][p]paletteuse=dither=sierra2_4a",
                    "-loop", "0", str(out.with_suffix(".gif"))], check=True)
    print(out, out.with_suffix(".gif"), [(t, ns) for t, ns in rows])


if __name__ == "__main__":
    main()
