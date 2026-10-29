import json, os, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, "/root/projects/drivetoy_v2")
from f_toy.datasets.drivetoy.debug_view import render_topdown
from f_toy.datasets.drivetoy.diversity import ASPECTS
from f_toy.datasets.drivetoy.export import scene_dict
from f_toy.datasets.drivetoy.scenarios import simulate
ROOT = Path("/root/projects/drive_v2/val"); OUT = Path("/root/projects/drive_v2/viz")
meta = json.loads((ROOT / "meta.json").read_text())
ann = np.load(ROOT / "annotations.npz", mmap_mode="r"); div, br = np.asarray(ann["diversity"]), np.asarray(ann["branch_id"])
rng = np.random.default_rng(0); picks = {}
for i, key in enumerate("ABCD"):            # the same draws as v2_diversity_gifs.py --seed 0
    picks[key] = []
    for j, lab in enumerate(meta["diversity"][key]):
        pool = np.flatnonzero((div[:, i] == j) & (br == 0))
        if len(pool): picks[key].append((lab, int(rng.choice(pool))))
t = [l for l, _ in picks["B"]].index("lane end"); n = picks["B"][t][1]
a = np.load(ROOT / "labels" / f"{n:06d}.npz")
tgt = {k: ASPECTS[k][c] for k, c in zip("ABCD", a["target"])}; tgt["template"] = meta["templates"][int(a["template"])]
print("sample", n, "seed", int(a["sim_seed"]), "target", tgt)
r = simulate(int(a["sim_seed"]), meta["cfg"], target=tgt)
lab = r["labels"][0]; print("new labels", lab)
sc = scene_dict(r, 0, meta["cfg"], int(a["sim_seed"]))
d = Path("/tmp/lane_end"); d.mkdir(exist_ok=True)
(d / "scene.json").write_text(json.dumps(sc)); (d / "jobs.txt").write_text(f"{d}/scene.json\t{d}/new.npz\n")
if (d / "new.npz").exists(): os.remove(d / "new.npz")
env = dict(os.environ, LD_LIBRARY_PATH="/root/opt/cuda525")
subprocess.run(["/root/opt/blender/blender", "-b", "--python", "f_toy/datasets/drivetoy/blender/render_clip.py", "--", "--jobs",
                str(d / "jobs.txt"), "--gpu", "0"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
rv = np.load(d / "new.npz"); video, index = rv["video"], rv["index"]
si = [k for k, s_ in enumerate(sc["signs"]) if s_["kind"] == "lane_ends"][0]
sign_px = [int((index[f] == 3000 + si).sum()) for f in range(32)]
print("lane-ends sign pixels per frame", sign_px)
rec = r["branches"][0]
S, W, CAP = 256, 514, 36
old = Image.open(OUT / "diversity_B_cause.gif")
x0, y0 = (t % 3) * (W + 6), (t // 3) * (S + CAP + 6)
try: font = ImageFont.load_default(size=13)
except TypeError: font = ImageFont.load_default()
frames = []
for f in range(32):
    old.seek(f)
    g = Image.new("RGB", (W, 2 * (S + CAP) + 8), (25, 25, 25))
    g.paste(old.convert("RGB").crop((x0, y0, x0 + W, y0 + CAP + S)), (0, 0))
    y = S + CAP + 8; dr = ImageDraw.Draw(g)
    dr.text((4, y + 2), "after: cones nearer, sign 2x and 45 m out", fill=(120, 220, 255), font=font)
    dr.text((4, y + 18), " / ".join(lab[k] for k in "ABCD"), fill=(200, 200, 200), font=font)
    g.paste(Image.fromarray(video[f]).resize((S, S), Image.NEAREST), (0, y + CAP))
    g.paste(Image.fromarray(np.ascontiguousarray(render_topdown(r["map"], rec, f, size=S, span=80.0)[:, :, ::-1])), (S + 2, y + CAP))
    frames.append(g)
p = OUT / "lane_end_before_after.gif"
frames[0].save(p, save_all=True, append_images=frames[1:], duration=200, loop=0, optimize=True)
Image.fromarray(np.concatenate([np.asarray(frames[f]) for f in (0, 8, 16)], 1)).save("/tmp/lane_end/peek.png")
print(p)
