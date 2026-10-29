#!/usr/bin/env python
"""How much the ego's frames 32-63 can still vary once frames 0-31 are fixed.

Every sample is re-simulated from its stored seed and target up to frame 31 (DT_CUE_FRAME),
then forked K ways with a fresh random stream and run to frame 63, in two modes:

- v2:   the build's rules after frame 31 (cued signal timing, decided pedestrians);
- free: those two rules off after frame 31, and the signal timing the video does not show
        (the WALK part of a green, an arrow's length, the planned later phases) drawn again
        per fork, as DriveToy v3 does for its set labels.

Per template, the share of samples with some fork more than 2 m / 10 m from fork 0 at frame
63 (DriveToy v3's measure) and at frame 43, and how far a constant-speed guess from frame 31
lands from the distance the ego drives in frames 32-63.

Kept clips re-simulate only with the code they were built with (--old, commit 71e9a11); the
lane-change clips rebuilt by the partial update use this checkout.

    python analysis/v2_fork_spread.py /root/projects/drive_v2/val --old /root/dt_prev --k 16
"""
import argparse
import copy
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
ROOT = META = S = ASPECTS = DEFAULT = None
K = 16
NEW = False


def init(code, root, k, new):
    global ROOT, META, S, ASPECTS, DEFAULT, K, NEW
    sys.path.insert(0, code)
    from f_toy.datasets.drivetoy import scenarios
    from f_toy.datasets.drivetoy.config import DRIVETOY_DEFAULT_CFG
    from f_toy.datasets.drivetoy.diversity import ASPECTS as AS
    from f_toy.datasets.drivetoy.diversity import ego_approach as ea
    from f_toy.datasets.drivetoy.sim import Signals as SG
    S, ASPECTS, DEFAULT = scenarios, AS, DRIVETOY_DEFAULT_CFG
    globals().update(ego_approach=ea, Signals=SG)
    ROOT, K, NEW = Path(root), k, new
    META = json.loads((ROOT / "meta.json").read_text())
    S._CAP = []
    orig = S._attempt

    def attempt(rng, cfg, tg):   # keep every attempt's branches at frame 31
        m, brs = orig(rng, cfg, tg)
        S._CAP.append((m, brs))
        return m, brs
    S._attempt = attempt


def posterior_signals(br31, rec, m, cfg, k, seed, tries=60000):
    """k controllers that show the ego exactly the light it saw in frames 0-31, drawn from the
    standard cycle (random phase, warm-up from t = -60 s): the phase timing the video does not
    show is drawn again. None when the ego has no light."""
    gi = ego_approach(rec, m)
    if gi is None or br31.signals is None:
        return None
    g = br31.signals.groups[gi]
    fps, T0 = cfg["DT_FPS"], cfg["DT_CUE_FRAME"]
    seen = [tuple(int(x) for x in rec["signals"][f, gi, :2]) for f in range(T0 + 1)]
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(tries):
        c = Signals(m, cfg, rng, -60.0)
        ok = True
        for f in range(T0 + 1):
            c.update(f / fps, rng)
            b, a = c.state(g, f / fps)[:2]
            if (int(b), int(a)) != seen[f]:
                ok = False
                break
        if ok:
            out.append(c)
            if len(out) == k:
                break
    return out


def travelled(xy):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])


def one(n):
    a = np.load(ROOT / "labels" / f"{n:06d}.npz")
    cfg = {**DEFAULT, **META["cfg"]} if NEW else META["cfg"]
    tgt = {k: ASPECTS[k][c] for k, c in zip("ABCD", a["target"])}
    tgt["template"] = META["templates"][int(a["template"])]
    S._CAP.clear()
    r = S.simulate(int(a["sim_seed"]), cfg, target=tgt)
    if not r["ok"]:
        return n, None
    m, brs = S._CAP[r["tries"] - 1]
    br31 = brs[int(a["branch_id"])]
    T0 = cfg["DT_CUE_FRAME"]
    total = cfg["FRAMES"] + cfg["FRAMES_FUTURE"]
    assert br31.frame == T0 and m is r["map"]

    post = posterior_signals(br31, S._pack(br31), m, cfg, K, 7 + n)

    def run(seed, free, sig=None):
        br = copy.deepcopy(br31, memo={id(m): m})
        if seed is not None:
            br.rng = np.random.RandomState(seed)
        if free:
            br.cfg = dict(br.cfg, DT_V2=False)
        if sig is not None:
            br.signals = copy.deepcopy(sig, memo={id(m): m})
        while br.frame < total - 1:
            br.step_frame()
        pk = S._pack(br)
        return pk["agents"][T0:], pk["signals"][T0:]   # agents [type, x, y, h, v, ..., alive], frames 31-63

    own = run(None, False)[0]
    same = bool(np.array_equal(own, a["agents"][T0:]))
    nohold = run(None, True)[0][:, 0]
    gi = ego_approach(S._pack(br31), m)
    ag = a["agents"][T0]
    near = [j for j in range(1, ag.shape[0]) if ag[j, 9] > 0 and np.linalg.norm(ag[j, 1:3] - ag[0, 1:3]) < 50.0]
    out = dict(n=n, template=tgt["template"], A=ASPECTS["A"][int(a["diversity"][0])],
               B=ASPECTS["B"][int(a["diversity"][1])], same=same, v31=float(a["agents"][T0, 0, 4]),
               label_drive=float(travelled(a["agents"][T0:, 0, 1:3])[-1]),
               nohold63=float(np.linalg.norm(nohold[-1, 1:3] - a["agents"][-1, 0, 1:3])),
               n_post=None if post is None else len(post))
    lab = a["agents"][T0:, 0, 1:3]
    for mode in ("v2", "free"):
        sigs = post if (mode == "free" and post) else [None] * K
        runs = [run(10_000 * n + k, mode == "free", sigs[k % len(sigs)]) for k in range(K)]
        agents = np.stack([r_[0] for r_ in runs])                              # (K, 33, A, 10), frames 31-63
        xy = agents[:, :, 0, 1:3]                                              # ego (K, 33, 2)
        light = None if gi is None else np.stack([r_[1][:, gi, :2] for r_ in runs])
        dn = 0.0
        if near:   # agents near the ego at frame 31 that are still there at 63 in every fork
            keep = [j for j in near if (agents[:, -1, j, 9] > 0).all()]
            if keep:
                dn = float(np.linalg.norm(agents[:, -1, keep, 1:3] - agents[0, -1, keep, 1:3], axis=-1).max())
        out[mode] = dict(d63=np.linalg.norm(xy[:, -1] - xy[0, -1], axis=1).tolist(),
                         d43=np.linalg.norm(xy[:, 12] - xy[0, 12], axis=1).tolist(),
                         lab63=np.linalg.norm(xy[:, -1] - lab[-1], axis=1).tolist(),
                         lab43=np.linalg.norm(xy[:, 12] - lab[12], axis=1).tolist(),
                         drive=[float(travelled(e)[-1]) for e in xy],
                         light_differs=None if light is None else bool((light != light[0]).any()),
                         light_vs_label=None if light is None else bool((light != a["signals"][T0:, gi, :2]).any()),
                         near63=dn)
    return n, out


def summary(res):
    def block(rows):
        if not rows:
            return {}
        d = dict(n=len(rows))
        for mode in ("v2", "free"):
            d63 = np.array([max(r[mode]["d63"]) for r in rows])
            d43 = np.array([max(r[mode]["d43"]) for r in rows])
            l63 = np.array([max(r[mode]["lab63"]) for r in rows])
            l43 = np.array([max(r[mode]["lab43"]) for r in rows])
            sd = np.array([np.std(r[mode]["drive"]) for r in rows])
            d[mode] = {"f63>2m": round(float((d63 > 2).mean()), 3), "f63>10m": round(float((d63 > 10).mean()), 3),
                       "f43>2m": round(float((d43 > 2).mean()), 3),
                       "label63>2m": round(float((l63 > 2).mean()), 3), "label43>2m": round(float((l43 > 2).mean()), 3),
                       "drive_std_mean": round(float(sd.mean()), 2)}
            lt = [r[mode]["light_differs"] for r in rows if r[mode]["light_differs"] is not None]
            lv = [r[mode]["light_vs_label"] for r in rows if r[mode]["light_vs_label"] is not None]
            d[mode]["ego_light_differs"] = round(float(np.mean(lt)), 3) if lt else None
            d[mode]["ego_light_vs_label"] = round(float(np.mean(lv)), 3) if lv else None
            d[mode]["near_agent63>2m"] = round(float(np.mean([r[mode]["near63"] > 2 for r in rows])), 3)
        d["nohold63>2m"] = round(float(np.mean([r["nohold63"] > 2 for r in rows])), 3)
        post = [r["n_post"] for r in rows if r["n_post"] is not None]
        d["no_posterior"] = int(sum(p == 0 for p in post))
        return d
    rows = [r for r in res.values() if r is not None]
    out = dict(n=len(res), resim_ok=len(rows), resim_equal=sum(r["same"] for r in rows), all=block(rows))
    for key in ("template", "A", "B"):
        out[key] = {v: block([r for r in rows if r[key] == v]) for v in sorted({r[key] for r in rows})}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--old", required=True, help="git worktree of the commit the kept clips were built with")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--n", type=int, default=0, help="first n samples only (0: all)")
    ap.add_argument("--workers", type=int, default=120)
    ap.add_argument("--out", default=str(HERE / "analysis" / "v2_fork_spread"))
    args = ap.parse_args()
    root = Path(args.root)
    meta = json.loads((root / "meta.json").read_text())
    N = args.n or meta["num_samples"]
    lc = list(meta["cfg"]["DT_SHARES"]["D"]).index("lane change")
    is_lc = {n: int(np.load(root / "labels" / f"{n:06d}.npz")["target"][3]) == lc for n in range(N)}
    ctx = mp.get_context("fork")
    res = {}
    for code, ids, new in ((args.old, [n for n in range(N) if not is_lc[n]], False),
                           (str(HERE), [n for n in range(N) if is_lc[n]], True)):
        with ctx.Pool(args.workers, initializer=init, initargs=(code, str(root), args.k, new)) as p:
            for n, r in p.imap_unordered(one, ids, chunksize=2):
                res[n] = r
    out = Path(args.out + f"_{root.name}")
    (out.with_suffix(".samples.json")).write_text(json.dumps([res[n] for n in sorted(res)]))
    s = summary(res)
    out.with_suffix(".json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
