#!/usr/bin/env python
"""Before/after video of one val sample: the stored clip (old code) above a re-simulation and
re-render with the current code. Each row is the ego view, a top-down map and a close-up map.
The old map comes from re-simulating with the code at --old (a git worktree of the old commit);
the new row uses the current code and the current config defaults.

    python scripts/v2_before_after.py 1108 --old /root/dt_old --title "right turn" --out viz/right_turn_before_after.mp4
"""
import argparse, json, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent.parent
ROOT = Path("/root/projects/drive_v2/val")
META = json.loads((ROOT / "meta.json").read_text())

MAPS = """
import sys, json, numpy as np
sys.path.insert(0, sys.argv[1])
from f_toy.datasets.drivetoy.debug_view import render_topdown
from f_toy.datasets.drivetoy.diversity import ASPECTS
from f_toy.datasets.drivetoy.scenarios import simulate
from f_toy.datasets.drivetoy.export import scene_dict
meta = json.load(open(sys.argv[2] + "/meta.json")); n = int(sys.argv[3])
if sys.argv[5] == "new":   # the current code also takes the current config defaults
    from f_toy.datasets.drivetoy.config import DRIVETOY_DEFAULT_CFG
    meta["cfg"] = {**meta["cfg"], **DRIVETOY_DEFAULT_CFG}
a = np.load(sys.argv[2] + f"/labels/{n:06d}.npz")
tgt = {k: ASPECTS[k][c] for k, c in zip("ABCD", a["target"])}; tgt["template"] = meta["templates"][int(a["template"])]
r = simulate(int(a["sim_seed"]), meta["cfg"], target=tgt); b = int(a["branch_id"]); rec = r["branches"][b]
T = meta["frames"]
far = np.stack([render_topdown(r["map"], rec, f, size=256, span=80.0)[:, :, ::-1] for f in range(T)])
near = np.stack([render_topdown(r["map"], rec, f, size=256, span=30.0)[:, :, ::-1] for f in range(T)])
# ego body clearance to the sidewalk polygons (negative = overlap)
from matplotlib.path import Path as MP
sw = [MP(np.asarray(s["poly"])) for s in r["map"].surfaces if s["kind"] == "sidewalk"]
L, W = 4.5, 1.8
gx, gy = np.meshgrid(np.linspace(-L / 2, L / 2, 19), np.linspace(-W / 2, W / 2, 9))
body = np.stack([gx.ravel(), gy.ravel()], 1)
over = []
for f in range(T):
    x, y, h = rec["ego"][f, :3]
    R = np.array([[np.cos(h), -np.sin(h)], [np.sin(h), np.cos(h)]])
    pts = body @ R.T + [x, y]
    over.append(float(np.mean(np.any([p.contains_points(pts) for p in sw], 0))))
sc = scene_dict(r, b, meta["cfg"], int(a["sim_seed"]))
np.savez(sys.argv[4], far=far, near=near, over=np.array(over), labels=json.dumps(r["labels"][b]))
json.dump(sc, open(sys.argv[4] + ".scene.json", "w"))
"""


def maps(code, n, out, which):
    subprocess.run([sys.executable, "-c", MAPS, str(code), str(ROOT), str(n), out, which], check=True, cwd=code)
    z = np.load(out)
    return z, json.load(open(out + ".scene.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("--old", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from PIL import Image, ImageDraw, ImageFont
    import imageio.v2 as imageio
    d = Path(tempfile.mkdtemp())
    old, _ = maps(args.old, args.n, str(d / "old.npz"), "old")
    new, sc = maps(HERE, args.n, str(d / "new.npz"), "new")
    (d / "scene.json").write_text(json.dumps(sc)); (d / "jobs.txt").write_text(f"{d}/scene.json\t{d}/new_r.npz\n")
    env = dict(os.environ, LD_LIBRARY_PATH="/root/opt/cuda525:" + os.environ.get("LD_LIBRARY_PATH", ""))
    subprocess.run(["/root/opt/blender/blender", "-b", "--python", str(HERE / "f_toy/datasets/drivetoy/blender/render_clip.py"),
                    "--", "--jobs", str(d / "jobs.txt"), "--gpu", "0"], env=env, cwd=HERE,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    new_vid = np.load(d / "new_r.npz")["video"]
    old_vid = np.load(ROOT / "videos.npy", mmap_mode="r")[args.n].transpose(1, 2, 3, 0)
    for name, z in (("before", old), ("after", new)):
        print(name, json.loads(str(z["labels"])), "max share of ego body over the sidewalk %.0f%%" % (100 * z["over"].max()))
    try:
        font = ImageFont.load_default(size=13)
    except TypeError:
        font = ImageFont.load_default()
    S, CAP = 256, 22
    frames = []
    for f in range(len(new_vid)):
        g = Image.new("RGB", (3 * S + 4, 2 * (S + CAP) + 6), (25, 25, 25)); dr = ImageDraw.Draw(g)
        for row, (name, vid, z) in enumerate((("before", old_vid, old), ("after", new_vid, new))):
            y = row * (S + CAP + 6)
            dr.text((4, y + 4), f"{name}: {args.title}   ego body over sidewalk {100 * z['over'][f]:.0f}%   f={f}",
                    fill=(255, 220, 60) if row == 0 else (120, 220, 255), font=font)
            g.paste(Image.fromarray(np.ascontiguousarray(vid[f])).resize((S, S), Image.NEAREST), (0, y + CAP))
            g.paste(Image.fromarray(np.ascontiguousarray(z["far"][f])), (S + 2, y + CAP))
            g.paste(Image.fromarray(np.ascontiguousarray(z["near"][f])), (2 * S + 4, y + CAP))
        frames.append(np.asarray(g))
    frames = [fr[: fr.shape[0] // 2 * 2, : fr.shape[1] // 2 * 2] for fr in frames]
    imageio.mimwrite(args.out, frames, fps=5, codec="libx264", quality=None,
                     output_params=["-crf", "8", "-pix_fmt", "yuv420p"], macro_block_size=1)
    Image.fromarray(np.concatenate([frames[f] for f in (10, 20, 28)], 0)).save(str(Path(args.out).with_suffix(".png")))
    print(args.out)


if __name__ == "__main__":
    main()
