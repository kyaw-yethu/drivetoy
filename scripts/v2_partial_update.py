#!/usr/bin/env python
"""Partial update of a built split for DATASETv2_revision.md fixes 7 and 8, without a rebuild.

- Fix 8 (follow label): every clip that is kept is re-simulated with the code it was built with
  (``--old``, a git worktree of that commit; the re-simulation must equal the stored agents) and
  relabelled with the current ``diversity.py``. Only its diversity labels change.
- Fix 7 (overtake): every clip whose target path is "lane change" is simulated again with the
  current code (same clip id, target and seeds) and rendered again.

Then videos.npy is updated in place for the replaced samples, and annotations.npz and meta.json
are rewritten.

    python scripts/v2_partial_update.py /root/projects/drive_v2/val --old /root/dt_prev
"""
import argparse, importlib.util, json, os, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent.parent

RELABEL = r"""
import importlib.util, json, sys
from multiprocessing import Pool
import numpy as np
old, new_div, root, ids_file, out_file = sys.argv[1:6]
sys.path.insert(0, old)
from f_toy.datasets.drivetoy.diversity import ASPECTS
from f_toy.datasets.drivetoy.scenarios import simulate
spec = importlib.util.spec_from_file_location("f_toy.datasets.drivetoy.diversity_new", new_div)
new = importlib.util.module_from_spec(spec); spec.loader.exec_module(new)
meta = json.load(open(root + "/meta.json")); cfg = meta["cfg"]; K = meta["branches"]
def one(i):
    a0 = np.load(root + f"/labels/{i * K:06d}.npz")
    tgt = {k: ASPECTS[k][c] for k, c in zip("ABCD", a0["target"])}; tgt["template"] = meta["templates"][int(a0["template"])]
    r = simulate(int(a0["sim_seed"]), cfg, target=tgt)
    res = []
    for b in range(K):
        a = np.load(root + f"/labels/{i * K + b:06d}.npz")
        rec = r["branches"][b]
        same = bool(np.array_equal(rec["agents"], a["agents"]))
        lab = new.clip_labels(rec, r["map"], cfg)
        res.append((i * K + b, same, [ASPECTS[k].index(lab[k]) for k in "ABCD"], [int(x) for x in a["diversity"]]))
    return res
ids = json.load(open(ids_file))
with Pool(120) as p:
    out = [x for rs in p.map(one, ids, chunksize=4) for x in rs]
json.dump(out, open(out_file, "w"))
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--old", required=True, help="git worktree of the commit the split was built with")
    ap.add_argument("--workers", type=int, default=120)
    args = ap.parse_args()
    root = Path(args.root)
    sys.path.insert(0, str(HERE))
    from f_toy.datasets.drivetoy.builder import _sim_one, assemble, render_all, _free_gpus
    from f_toy.datasets.drivetoy.config import DRIVETOY_DEFAULT_CFG
    from f_toy.datasets.drivetoy.diversity import ASPECTS
    meta = json.loads((root / "meta.json").read_text())
    cfg = {**DRIVETOY_DEFAULT_CFG, **meta["cfg"], "DT_OVERTAKE_CHECK": True, "NUM_SAMPLES": meta["num_samples"]}
    K, N = meta["branches"], meta["num_samples"]
    n_clips = N // K
    LC = ASPECTS["D"].index("lane change")
    tg = {i: np.load(root / "labels" / f"{i * K:06d}.npz")["target"] for i in range(n_clips)}
    redo = [i for i in range(n_clips) if int(tg[i][3]) == LC]
    keep = [i for i in range(n_clips) if int(tg[i][3]) != LC]
    print(f"{root.name}: {len(keep)} clips relabelled, {len(redo)} lane-change clips rebuilt", flush=True)

    # fix 8: relabel the kept clips (re-simulated with the old code)
    t0 = time.time()
    ids_file, out_file = root / "_relabel_ids.json", root / "_relabel_out.json"
    ids_file.write_text(json.dumps(keep))
    subprocess.run([sys.executable, "-c", RELABEL, args.old, str(HERE / "f_toy/datasets/drivetoy/diversity.py"),
                    str(root), str(ids_file), str(out_file)], check=True, cwd=args.old)
    rel = json.loads(out_file.read_text())
    bad = [n for n, same, _, _ in rel if not same]
    if bad:
        raise RuntimeError(f"{len(bad)} kept samples do not re-simulate to the stored agents: {bad[:10]}")
    changed = 0
    for n, _, new, old in rel:
        if new != old:
            p = root / "labels" / f"{n:06d}.npz"
            d = dict(np.load(p))
            d["diversity"] = np.array(new, np.int8)
            np.savez(p, **d)
            changed += 1
    ids_file.unlink(); out_file.unlink()
    print(f"  relabelled: {changed} of {len(rel)} samples changed ({time.time() - t0:.0f}s)", flush=True)

    # fix 7: rebuild the lane-change clips
    t0 = time.time()
    samples = [i * K + b for i in redo for b in range(K)]
    for n in samples:
        for sub, ext in (("labels", "npz"), ("scenes", "json"), ("render", "npz")):
            f = root / sub / f"{n:06d}.{ext}"
            if f.exists():
                f.unlink()
    targets = {i: {**{k: ASPECTS[k][int(c)] for k, c in zip("ABCD", tg[i])},
                   "template": meta["templates"][int(np.load(root.parent / root.name / "annotations.npz")["template"][i * K])]}
               for i in redo}
    failed = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, lab in ex.map(_sim_one, [(i, cfg, root, targets[i]) for i in redo], chunksize=1):
            if lab is None:
                failed.append(i)
    if failed:
        raise RuntimeError(f"{len(failed)} lane-change clips could not be built: {failed[:10]}")
    print(f"  simulated {len(redo)} clips ({time.time() - t0:.0f}s)", flush=True)
    gpus = _free_gpus()
    render_all(root, N, gpus, cfg["DT_RENDER_PROCS"], cfg["DT_RENDER_SAMPLES"])
    print(f"  rendered {len(samples)} samples ({time.time() - t0:.0f}s)", flush=True)
    assemble(root, cfg, only=samples)
    print(f"  assembled {root} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
