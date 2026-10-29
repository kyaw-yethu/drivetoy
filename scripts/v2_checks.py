#!/usr/bin/env python
"""The checks of DATASETv2_plan.md ("Checks before the full build") on a built split:

1. shares of the four diversity aspects (A within 2 points of its share; the ego carries out
   its light's response, re-derived from a re-simulation), and B, C, D against their shares;
2. no light change in frames 32-63 without a yellow by frame 31, no ego reaction in frames
   32-63 to a pedestrian still undecided at frame 31 (``diversity.cue_problems``);
3. size of a lit lamp of the ego's own mast-arm heads, from the render's object index;
4. ``rules.check`` finds no violations;
5. the v1 VAE (vae_100e18b) keeps the signal colour of lit lamps on the ego's mast-arm heads.

Every clip is re-simulated from its stored sim_seed and target; the re-simulation must equal the
stored labels. Writes <split>/checks.json.

    python scripts/v2_checks.py /root/projects/drive_v2/pilot [--vae /root/projects/drivetoy_checkpoint/vae_100e18b]
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f_toy.datasets.drivetoy.builder import HEAD_KINDS, lamp_lit
from f_toy.datasets.drivetoy.diversity import ASPECTS, clip_labels, cue_problems
from f_toy.datasets.drivetoy.rules import check
from f_toy.datasets.drivetoy.scenarios import simulate

ROOT = None
META = None


def resim(n):
    a = np.load(ROOT / "labels" / f"{n:06d}.npz")
    cfg = META["cfg"]
    tgt = {k: ASPECTS[k][c] for k, c in zip("ABCD", a["target"])}
    tgt["template"] = META["templates"][int(a["template"])]
    r = simulate(int(a["sim_seed"]), cfg, target=tgt)
    b = int(a["branch_id"])
    rec = r["branches"][b]
    same = bool(np.array_equal(rec["agents"], a["agents"]))
    lab = clip_labels({k: rec[k] for k in rec}, r["map"], cfg)
    m = r["map"]
    lane0 = int(rec["extra"][0, 0, 0])
    ln = m.lanes[lane0]
    leg = ln.leg if ln.kind == "in" else None
    # distance of the ego's front bumper to its stop line, per visible frame (None once past it)
    dist = []
    for f in range(cfg["FRAMES"]):
        lid = int(rec["extra"][f, 0, 0])
        L = m.lanes[lid] if lid >= 0 else None
        dist.append(float(L.stop["s"] - rec["extra"][f, 0, 1] - 2.25) if L is not None and L.kind == "in" and L.stop is not None else None)
    return n, dict(same=same, A=lab["A"], a_ok=lab["a_ok"], stored=[ASPECTS[k][c] for k, c in zip("ABCD", a["diversity"])],
                   labels=[lab[k] for k in "ABCD"], cue=[str(p) for p in cue_problems(rec, m, cfg)],
                   viol=[str(v) for v in check(rec, m, cfg)], leg=leg, dist=dist, template=tgt["template"])


def lamp_stats(n, info, sims):
    """Per visible frame: (distance to the line, bbox span in px of the lit lamp of the ego's own mast-arm heads)."""
    s = sims[n]
    if s["leg"] is None or s["template"] != "signal4":
        return []
    a = np.load(ROOT / "labels" / f"{n:06d}.npz")
    ix = np.load(ROOT / "render" / f"{n:06d}.npz")["index"]
    hi = [h for h in range(len(info)) if info[h, 0] == HEAD_KINDS.index("veh") and info[h, 2] == s["leg"]]
    out = []
    for f in range(ix.shape[0]):
        if s["dist"][f] is None:
            continue
        span = 0
        for h in hi:
            for k in range(4):
                if lamp_lit(a["signals"][f, info[h, 1]], k):
                    ys, xs = np.nonzero(ix[f] == 5000 + 4 * h + k)
                    if len(ys):
                        span = max(span, int(max(ys.max() - ys.min(), xs.max() - xs.min()) + 1))
        out.append((s["dist"][f], span))
    return out


def vae_colour(paths, ns, vae_dir, sims):
    """Mean colour of lit lamp pixels on the ego's mast-arm heads in the input and in the VAE reconstruction,
    classified as red / yellow / green by nearest reference hue."""
    import torch
    for k in [k for k in sys.modules if k == "f_toy" or k.startswith("f_toy.")]:
        del sys.modules[k]          # the v1 VAE loader lives in Task_specific_JDM's f_toy (imported, not edited)
    sys.path.insert(0, "/root/projects/Task_specific_JDM")
    from f_toy.evaluation.checkpoints import load_vae
    vae, _ = load_vae(vae_dir, "best", "cuda")
    vid = np.load(ROOT / "videos.npy", mmap_mode="r")
    res = dict(n=0, same=0, by_state={})
    for n in ns:
        a = np.load(ROOT / "labels" / f"{n:06d}.npz")
        ix = np.load(ROOT / "render" / f"{n:06d}.npz")["index"]
        info = a["head_info"]
        s = sims[n]
        hi = [h for h in range(len(info)) if info[h, 0] == HEAD_KINDS.index("veh") and info[h, 2] == s["leg"]]
        x = torch.from_numpy(np.asarray(vid[n], np.float32) / 255.0)[None].cuda()
        with torch.no_grad():
            mu, _ = vae.encode(x)
            rec = vae.decode(mu).clamp(0, 1)[0].cpu().numpy()
        org = np.asarray(vid[n], np.float32) / 255.0
        for f in range(org.shape[1]):
            for h in hi:
                for k in range(3):
                    if not lamp_lit(a["signals"][f, info[h, 1]], k):
                        continue
                    mask = ix[f] == 5000 + 4 * h + k
                    if mask.sum() < 4:
                        continue
                    c0, c1 = org[:, f][:, mask].mean(1), rec[:, f][:, mask].mean(1)
                    k0, k1 = _hue_class(c0), _hue_class(c1)
                    name = ["red", "yellow", "green"][k]
                    d = res["by_state"].setdefault(name, dict(n=0, input_ok=0, recon_ok=0, recon_eq_input=0))
                    d["n"] += 1
                    d["input_ok"] += int(k0 == k)
                    d["recon_ok"] += int(k1 == k)
                    d["recon_eq_input"] += int(k1 == k0)
    return res


def _hue_class(rgb):
    r, g, b = rgb
    # red: r >> g; yellow: r and g both high; green: g > r
    if g > r * 1.05:
        return 2
    return 1 if g > 0.55 * r else 0


def main():
    global ROOT, META
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--workers", type=int, default=120)
    ap.add_argument("--vae", default="/root/projects/drivetoy_checkpoint/vae_100e18b")
    args = ap.parse_args()
    ROOT = Path(args.root)
    META = json.loads((ROOT / "meta.json").read_text())
    N = META["num_samples"]
    with Pool(args.workers) as p:
        sims = dict(p.map(resim, range(N)))
    out = {}
    # 1. shares
    ann = np.load(ROOT / "annotations.npz")
    shares = META["cfg"]["DT_SHARES"]
    first = [n for n in range(N) if int(ann["branch_id"][n]) == 0]
    tab = {}
    for i, k in enumerate("ABCD"):
        codes = ann["diversity"][first, i]
        tab[k] = {lab: dict(share=shares[k][lab], got=round(float((codes == j).mean()), 4))
                  for j, lab in enumerate(ASPECTS[k])}
    out["shares"] = tab
    out["A_max_abs_dev_pp"] = round(100 * max(abs(v["got"] - v["share"]) for v in tab["A"].values()), 2)
    out["resim_equal"] = sum(s["same"] for s in sims.values())
    out["labels_equal"] = sum(s["labels"] == s["stored"] for s in sims.values())
    out["a_ok"] = sum(bool(s["a_ok"]) for s in sims.values())
    out["n"] = N
    # 2. cued futures
    out["cue_problem_clips"] = sum(bool(s["cue"]) for s in sims.values())
    out["cue_examples"] = [(n, s["cue"][:2]) for n, s in sims.items() if s["cue"]][:5]
    # 4. rules
    out["violation_clips"] = sum(bool(s["viol"]) for s in sims.values())
    out["violation_examples"] = [(n, s["viol"][:2]) for n, s in sims.items() if s["viol"]][:8]
    # 3. lamp size on the ego's mast-arm heads
    pairs = []
    for n in range(N):
        info = np.load(ROOT / "labels" / f"{n:06d}.npz")["head_info"]
        pairs += lamp_stats(n, info, sims)
    pairs = np.array(pairs, float).reshape(-1, 2)
    bins = [(0, 3), (3, 10), (10, 20), (20, 35), (35, 60)]
    out["lamp_span_px"] = {}
    for lo, hi in bins:
        sel = (pairs[:, 0] >= lo) & (pairs[:, 0] < hi)
        sp = pairs[sel, 1]
        vis = sp[sp > 0]
        out["lamp_span_px"][f"{lo}-{hi} m"] = dict(frames=int(sel.sum()), in_view=round(float((sp > 0).mean()), 3) if len(sp) else None,
                                                     median=float(np.median(vis)) if len(vis) else None,
                                                     p10=float(np.percentile(vis, 10)) if len(vis) else None,
                                                     share_ge3=round(float((vis >= 3).mean()), 3) if len(vis) else None)
    # 5. VAE colour
    sig = [n for n, s in sims.items() if s["template"] == "signal4" and s["leg"] is not None]
    out["vae_signal_colour"] = vae_colour(None, sig, args.vae, sims)
    (ROOT / "checks.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main()
