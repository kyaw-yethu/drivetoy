#!/usr/bin/env python
"""Rebuild a DriveToy v4 split from its plan and compare it with the stored data.

    python scripts/check_reproduce.py --split train                   # every clip of train
    python scripts/check_reproduce.py --split val_twins --sample 130   # 130 jobs spread over the situations
    python scripts/check_reproduce.py --split val --render 8           # also re-render 8 samples

Every job of the plan (a twin pair is one job) is simulated again exactly as scripts/build.py does, into a
scratch folder, and the labels, scene, map and futures of its samples are compared with the stored ones, along
with meta.json (apart from the render settings DT_RENDER_GPUS and DT_RENDER_PROCS). ``--render k`` re-renders
k stored scenes on one GPU and compares the frames. Mismatches go to <data>/<split>/logs/reproduce.log.
"""
import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import build  # noqa: E402
from drivetoy.builder import BLENDER, RENDER_SCRIPT, base_meta  # noqa: E402

RENDER_SETTINGS = ("DT_RENDER_GPUS", "DT_RENDER_PROCS")


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool(np.array_equal(a, b, equal_nan=a.dtype.kind == "f"))


def _check(task):
    split, job, samples, data, tmp, cfg, base = task
    t0 = time.time()
    work = tmp / f"job{job[0]:06d}"
    for sub in ("scenes", "labels", "render", "futures", "maps"):
        (work / sub).mkdir(parents=True, exist_ok=True)
    run = build._run_twin if split in build.TWIN_SPLITS else build._run_clip
    ok = run((*job, cfg, work, base))[1]
    problems, futures = [], {}
    if not ok:
        problems.append((samples[0], "build failed"))
    else:
        for n in samples:
            with np.load(data / "labels" / f"{n:06d}.npz") as a, np.load(work / "labels" / f"{n:06d}.npz") as b:
                for k in sorted(set(a.files) ^ set(b.files)):
                    problems.append((n, f"label key {k} only in {'stored' if k in a.files else 'rebuilt'}"))
                for k in sorted(set(a.files) & set(b.files)):
                    if not _same(a[k], b[k]):
                        problems.append((n, f"label {k}"))
            if json.loads((data / "scenes" / f"{n:06d}.json").read_text()) != \
                    json.loads((work / "scenes" / f"{n:06d}.json").read_text()):
                problems.append((n, "scene"))
            with gzip.open(data / "maps" / f"{n:06d}.json.gz") as fa, gzip.open(work / "maps" / f"{n:06d}.json.gz") as fb:
                if json.load(fa) != json.load(fb):
                    problems.append((n, "map"))
            with np.load(work / "futures" / f"{n:06d}.npz") as z:
                futures[n] = {k: z[k] for k in z.files}
    shutil.rmtree(work, ignore_errors=True)
    return job[0], samples, problems, futures, time.time() - t0


def _pick(todo, k):
    """k jobs spread over the situations, in job order."""
    rng = np.random.RandomState(0)
    groups = {}
    for t in todo:
        groups.setdefault(t[2], []).append(t)
    for g in groups.values():
        rng.shuffle(g)
    pick, i = [], 0
    while len(pick) < min(k, len(todo)):
        for g in groups.values():
            if i < len(g) and len(pick) < k:
                pick.append(g[i])
        i += 1
    return sorted(pick, key=lambda t: t[0][0])


def check_meta(split, data, jobs, cfg):
    stored = json.loads((data / "meta.json").read_text())
    want = json.loads(json.dumps(dict(base_meta(cfg, build.n_samples(split, jobs)), **build.split_meta(split)),
                                 default=list))
    for m in (stored, want):
        for k in RENDER_SETTINGS:
            m.get("cfg", {}).pop(k, None)
    return [k for k in sorted(set(stored) | set(want)) if stored.get(k) != want.get(k)]


def check_renders(data, tmp, samples, gpu, spp):
    out = tmp / "render"
    out.mkdir(parents=True, exist_ok=True)
    jf = tmp / "jobs.txt"
    jf.write_text("".join(f"{data / 'scenes' / f'{n:06d}.json'}\t{out / f'{n:06d}.npz'}\n" for n in samples))
    env = dict(os.environ, LD_LIBRARY_PATH="/root/opt/cuda525:" + os.environ.get("LD_LIBRARY_PATH", ""))
    subprocess.run([BLENDER, "-b", "--python", str(RENDER_SCRIPT), "--", "--jobs", str(jf), "--gpu", str(gpu),
                    "--samples", str(spp)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env, check=True)
    res = []
    for n in samples:
        with np.load(data / "render" / f"{n:06d}.npz") as a, np.load(out / f"{n:06d}.npz") as b:
            dv = np.abs(a["video"].astype(int) - b["video"].astype(int))
            res.append((n, int(dv.max()), float(dv.mean()), bool(np.array_equal(a["index"], b["index"]))))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=list(build.SPLITS))
    ap.add_argument("--data", default="/root/projects/drive_v4")
    ap.add_argument("--sample", type=int, default=0, help="check this many jobs, spread over the situations")
    ap.add_argument("--jobs", type=int, nargs="*", default=None, help="check these jobs only")
    ap.add_argument("--render", type=int, default=0, help="re-render this many samples and compare the frames")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--tmp", default="/root/scratch/reproduce")
    args = ap.parse_args()
    data = Path(args.data) / args.split
    tmp = Path(args.tmp) / args.split
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    jobs = build.load_plan(args.split)
    cfg = build.split_cfg(args.split, build.n_samples(args.split, jobs))
    base = build.SPLITS[args.split]
    twins = args.split in build.TWIN_SPLITS
    todo = [(j, [j[3], j[3] + 1] if j[1] == "pair" else [j[3]], j[2]) if twins else (j, [j[0]], j[1]) for j in jobs]
    meta_diff = check_meta(args.split, data, jobs, cfg)
    print(f"{args.split}: meta.json " + ("matches" if not meta_diff else f"differs in {meta_diff}"), flush=True)
    if args.jobs is not None:
        todo = [t for t in todo if t[0][0] in set(args.jobs)]
    elif args.sample:
        todo = _pick(todo, args.sample)
    stored = dict(np.load(data / "future_set.npz"))
    log = data / "logs" / "reproduce.log"
    n_samples = sum(len(t[1]) for t in todo)
    print(f"{args.split}: re-simulating {len(todo)} jobs ({n_samples} samples)", flush=True)
    bad, kinds, done, t0 = {}, {}, 0, time.time()
    with ProcessPoolExecutor(max_workers=args.workers or max(os.cpu_count() - 8, 1)) as ex, open(log, "w") as lf:
        futs = [ex.submit(_check, (args.split, job, samples, data, tmp, cfg, base)) for job, samples, _ in todo]
        for fu in as_completed(futs):
            j, samples, problems, futures, sec = fu.result()
            for n, f in futures.items():
                diff = [k for k in f if k not in stored or not _same(stored[k][n], f[k])]
                if diff:
                    problems.append((n, f"futures {','.join(diff)}"))
            for n, what in problems:
                bad.setdefault(n, []).append(what)
                kinds[what] = kinds.get(what, 0) + 1
                lf.write(f"sample {n}: {what}\n")
            lf.flush()
            done += 1
            if done % 100 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)} jobs in {time.time() - t0:.0f}s, {len(bad)} samples differ", flush=True)
    print(f"{args.split}: {n_samples - len(bad)} of {n_samples} samples identical"
          + (f"; differences: {json.dumps(dict(sorted(kinds.items(), key=lambda kv: -kv[1])))}" if bad else ""),
          flush=True)
    if args.render:
        all_samples = [n for t in todo for n in t[1]]
        pick = [all_samples[int(i)] for i in np.linspace(0, len(all_samples) - 1, min(args.render, len(all_samples)))]
        res = check_renders(data, tmp, pick, args.gpu, cfg["DT_RENDER_SAMPLES"])
        same = sum(r[1] == 0 and r[3] for r in res)
        print(f"{args.split}: {same} of {len(res)} re-rendered samples identical; max pixel difference "
              f"{max(r[1] for r in res)}, mean {np.mean([r[2] for r in res]):.4f}; index passes equal: "
              f"{sum(r[3] for r in res)}", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
