"""Build a DriveToy dataset to disk (docs/DRIVETOY.md): simulate → render →
assemble, the same envelope as CubeToy (``videos.npy`` memmap, ``annotations.npz``,
``meta.json``).

1. **simulate** (CPU pool), in batches of ``DT_BATCH`` clips. Every clip gets a target
   combination of the four diversity labels: the light label (A) from an exact allocation of
   its shares over the split, then B, (template, D) and C weighted by how far each label still
   is below its share, counting the labels the earlier batches actually reached. Clip i is
   ``scenarios.simulate(sim_seed, cfg, target)``; each of its ``DT_BRANCHES`` futures becomes
   one sample, ``n = i*K + b``, with its scene in ``scenes/<n>.json`` and its labels in
   ``labels/<n>.npz``. A clip whose target cannot be built is retried with a new seed, and
   then with another target of the same light label.
2. **render** (Blender, Cycles/CUDA): ``DT_RENDER_PROCS`` processes per GPU over
   ``render/<n>.npz``; finished samples are skipped, so an interrupted build resumes here.
3. **assemble**: stack labels, compute ``cue_visibility`` from the index pass, write the memmap.

The simulator is deterministic: a clip is rebuilt with ``simulate(sim_seed, cfg, target)``
from its stored ``sim_seed`` and ``target``.
"""
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .choices import CHOICE_KINDS, DEC_COLS, DRIVER_COLS, MAX_DECISIONS
from .config import DRIVETOY_DEFAULT_CFG
from .diversity import ASPECTS
from .export import scene_dict
from .map_io import save_map
from .scenarios import COMBOS, VARIANTS, plan_targets, simulate
from .sim import MAY_GO_REASONS

BLENDER = os.environ.get("BLENDER", "/root/opt/blender/blender")
RENDER_SCRIPT = Path(__file__).resolve().parent / "blender" / "render_clip.py"
TEMPLATES = ["signal4", "stop4", "multilane", "residential"]
CUES = ["veh_signal", "ped_signal", "stop_sign", "other_sign", "crosswalk", "brake_lit", "indicator_lit",
        "vehicle", "pedestrian", "signal_lit"]
HEAD_KINDS = ["veh", "veh_pole", "veh_near", "ped"]
RETRY_SEED = 1_000_003


def _code(key, label):
    return ASPECTS[key].index(label) if label in ASPECTS[key] else -1


def _sim_one(task):
    i, cfg, out, target = task
    K = cfg["DT_BRANCHES"]
    res, tgt = None, dict(target)
    rng = np.random.RandomState(cfg["SEED"] + i + 77)
    for r in range(10):
        if r >= 2:   # the combination did not come out: another one with the same light label
            rows = [c for c in COMBOS if c[0] == target["A"]]
            A, B, tmpl, D, C, E, V = rows[int(rng.randint(len(rows)))]
            tgt = dict(A=A, B=B, C=C, D=D, E=E, V=V, template=tmpl)
        seed = cfg["SEED"] + i + r * RETRY_SEED
        res = simulate(seed, cfg, target=tgt)
        if res["ok"]:
            break
    if not res["ok"]:
        return i, None
    E = cfg["DT_MAX_EVENTS"]
    for b, rec in enumerate(res["branches"]):
        n = i * K + b
        sc = scene_dict(res, b, cfg, seed)
        with open(out / "scenes" / f"{n:06d}.json", "w") as f:
            json.dump(sc, f)
        ev = rec["events"][:E]
        events = np.full((E, 5), -1, np.float32)
        events[:len(ev)] = ev
        ex = rec["extra"]
        lab = res["labels"][b]
        dec = rec["decisions"][np.argsort(-rec["decisions"][:, 6], kind="stable")][:MAX_DECISIONS]   # the ego's first
        decisions = np.full((MAX_DECISIONS, len(DEC_COLS)), -1, np.float32)
        decisions[:len(dec)] = dec
        _save_futures(out / "futures" / f"{n:06d}.npz", res["forks"][b])
        # scoring labels: may_go per frame, the ego's move, whether the route was drawn per future, the map
        score = dict(may_go=rec["may_go"], may_go_reason=rec["may_go_reason"], ego_turn=rec["ego_turn"],
                     route_drawn=np.bool_(res["route_draws"][b] is not None))
        save_map(out / "maps" / f"{n:06d}.json.gz", res["map"])
        np.savez(out / "labels" / f"{n:06d}.npz",
                 decisions=decisions, num_decisions=np.int16(len(dec)), drivers=rec["drivers"],
                 variant=np.int8(VARIANTS.index(tgt["V"]) if tgt.get("V") else -1),
                 agents=rec["agents"], lamps=rec["lamps"], signals=rec["signals"], ego=rec["ego"],
                 lane=ex[..., 0].astype(np.int16), s_local=ex[..., 1], indicator=ex[..., 2].astype(np.int8),
                 rule_state=ex[..., 3].astype(np.int8), yield_to=ex[..., 4].astype(np.int8),
                 ped_state=ex[..., 5].astype(np.int8), lc=rec["lc"], routes=rec["routes"],
                 events=events, num_events=np.int32(min(len(rec["events"]), E)),
                 clip_id=np.int32(i), branch_id=np.int8(b), template=np.int8(TEMPLATES.index(res["template"])),
                 sim_ok=np.bool_(res["ok"]), sim_seed=np.int64(seed), tries=np.int16(res["tries"]),
                 diversity=np.array([_code(k, lab[k]) for k in "ABCDE"], np.int8),
                 target=np.array([_code(k, tgt.get(k)) for k in "ABCDE"], np.int8),
                 camera=np.array([sc["camera"]["height"], sc["camera"]["pitch"], sc["camera"]["fov"]], np.float32),
                 head_info=np.array([[HEAD_KINDS.index(h["kind"]), sc["groups"].index(h["group"]) if h["group"] in sc["groups"] else -1,
                                      h["leg"]] for h in sc["heads"]] or np.zeros((0, 3)), np.int8).reshape(-1, 3),
                 signs=np.array([s["kind"] == "stop" for s in sc["signs"]], bool), **score)
    return i, [res["labels"][0][k] for k in "ABCDE"]


MAX_FUTURE_DECISIONS = 24


def _save_futures(path, futs):
    """A sample's DT_FORKS futures of frames 32 to 63 (future 0 is its own), stacked; per future
    the choices made in them, the ego's first."""
    dec = np.full((len(futs), MAX_FUTURE_DECISIONS, len(DEC_COLS)), -1, np.float32)
    for k, f in enumerate(futs):
        d = f["decisions"][np.argsort(-f["decisions"][:, 6], kind="stable")][:MAX_FUTURE_DECISIONS]
        dec[k, :len(d)] = d
    np.savez(path, decisions=dec, **{key: np.stack([f[key] for f in futs])
                                     for key in ("ego", "agents", "lamps", "signals", "rule_state", "yield_to")
                                     + tuple(k for k in ("may_go", "may_go_reason", "ego_turn") if k in futs[0])})


def _assemble_futures(out, N):
    """Stack futures/<n>.npz into future_set.npz and remove them."""
    fdir = out / "futures"
    if not (fdir / f"{N - 1:06d}.npz").exists():
        return
    fs = None
    for n in range(N):
        f = np.load(fdir / f"{n:06d}.npz")
        if fs is None:
            fs = {k: np.zeros((N,) + f[k].shape, f[k].dtype) for k in f.files}
        for k in f.files:
            fs[k][n] = f[k]
    np.savez(out / "future_set.npz", **fs)
    shutil.rmtree(fdir)


def _free_gpus():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return [int(l.split(",")[0]) for l in out.strip().splitlines() if int(l.split(",")[1]) < 1000]


def render_all(out, n_samples, gpus, procs, samples):
    todo = [n for n in range(n_samples) if not (out / "render" / f"{n:06d}.npz").exists()]
    if not todo:
        return
    slots = [(g, p) for g in gpus for p in range(procs)]
    running = []
    for k, (g, p) in enumerate(slots):
        mine = todo[k::len(slots)]
        if not mine:
            continue
        jf = out / "render" / f"jobs_{g}_{p}.txt"
        jf.write_text("".join(f"{out / 'scenes' / f'{n:06d}.json'}\t{out / 'render' / f'{n:06d}.npz'}\n" for n in mine))
        log = open(out / "render" / f"log_{g}_{p}.txt", "w")
        running.append(subprocess.Popen([BLENDER, "-b", "--python", str(RENDER_SCRIPT), "--", "--jobs", str(jf),
                                         "--gpu", str(g), "--samples", str(samples)], stdout=log, stderr=subprocess.STDOUT))
    t0 = time.time()
    while any(r.poll() is None for r in running):
        time.sleep(30)
        done = sum(1 for n in todo if (out / "render" / f"{n:06d}.npz").exists())
        rate = done / max(time.time() - t0, 1)
        print(f"  rendered {done}/{len(todo)}  ({rate:.2f}/s, eta {(len(todo) - done) / max(rate, 1e-6) / 3600:.1f} h)", flush=True)
    bad = [r.returncode for r in running if r.returncode != 0]
    missing = [n for n in todo if not (out / "render" / f"{n:06d}.npz").exists()]
    if bad or missing:
        raise RuntimeError(f"{len(bad)} Blender workers failed, {len(missing)} renders missing; see {out}/render/log_*.txt")


def cue_counts(index, lab):
    """(T, len(CUES)) visible-pixel counts per cue from the render's object index."""
    T = index.shape[0]
    out = np.zeros((T, len(CUES)), np.int32)
    info, signs = lab["head_info"], lab["signs"]
    lamps, sig = lab["lamps"], lab["signals"]
    for f in range(T):
        ix = index[f].astype(np.int32)
        ids, cnt = np.unique(ix, return_counts=True)
        for i, c in zip(ids, cnt):
            if 1 <= i < 1000:
                typ = lab["agents"][f, i - 1, 0]
                out[f, 7 if typ < 5 else 8] += c
            elif 1000 <= i < 2000:
                slot, k = divmod(i - 1000, 4)
                if lamps[f, slot, k]:
                    out[f, 5 if k == 0 else 6] += c
            elif 2000 <= i < 3000 and i - 2000 < len(info):
                out[f, 1 if info[i - 2000, 0] == 3 else 0] += c
            elif 3000 <= i < 4000 and i - 3000 < len(signs):
                out[f, 2 if signs[i - 3000] else 3] += c
            elif 4000 <= i < 5000:
                out[f, 4] += c
            elif 5000 <= i < 6000:
                h, k = divmod(i - 5000, 4)
                out[f, 0] += c
                if h < len(info) and lamp_lit(sig[f, info[h, 1]], k):
                    out[f, 9] += c
    return out


def lamp_lit(sig_g, k):
    """Whether lamp k (0 red, 1 yellow, 2 green, 3 arrow) of a head shows, for its group's state."""
    ball, arrow = int(sig_g[0]), sig_g[1] > 0
    return arrow if k == 3 else ball == (0, 1, 2)[k]


def _counts(out, done):
    counts = {k: {} for k in "ABCDE"}
    for lab in done.values():
        for k, l in zip("ABCDE", lab):
            counts[k][l] = counts[k].get(l, 0) + 1
    return counts


def _a_allocation(n, shares, seed):
    """Exactly round(share * n) light targets per label (largest remainder), in shuffled order."""
    names = list(shares)
    raw = np.array([shares[k] * n for k in names])
    k = np.floor(raw).astype(int)
    k[np.argsort(-(raw - k))[:n - k.sum()]] += 1
    a = [nm for nm, c in zip(names, k) for _ in range(c)]
    np.random.RandomState(seed).shuffle(a)
    return a


def simulate_split(cfg, out, n_clips, workers=None):
    """Simulate every clip of a split (resumable); returns {clip: [A, B, C, D] of branch 0}."""
    K = cfg["DT_BRANCHES"]
    shares = cfg["DT_SHARES"]
    a_all = _a_allocation(n_clips, shares["A"], cfg["SEED"] + 4242)
    done = {}
    for i in range(n_clips):
        f = out / "labels" / f"{(i * K + K - 1):06d}.npz"
        if f.exists():
            d = np.load(out / "labels" / f"{i * K:06d}.npz")["diversity"]
            done[i] = [ASPECTS[k][c] for k, c in zip("ABCDE", d)]
    t0 = time.time()
    failed = []
    B = cfg["DT_BATCH"]
    with ProcessPoolExecutor(max_workers=workers or os.cpu_count()) as ex:
        for b0 in range(0, n_clips, B):
            ids = [i for i in range(b0, min(b0 + B, n_clips)) if i not in done]
            if not ids:
                continue
            if cfg.get("DT_TARGETS"):   # a split built from a fixed list of targets (the preview)
                tmap = {i: cfg["DT_TARGETS"][i] for i in ids}
            else:
                counts = _counts(out, {i: v for i, v in done.items() if i < b0})
                # B, C, D in proportion to the clips planned so far (batches before b0 plus this one)
                targets = plan_targets([a_all[i] for i in range(b0, min(b0 + B, n_clips))], counts, n_clips, shares,
                                       np.random.RandomState(cfg["SEED"] + b0 + 999), pending_min=cfg["DT_PENDING_MIN"])
                tmap = {i: targets[i - b0] for i in ids}
            for i, lab in ex.map(_sim_one, [(i, cfg, out, tmap[i]) for i in ids], chunksize=1):
                if lab is None:
                    failed.append(i)
                else:
                    done[i] = lab
            print(f"  simulated {len(done)}/{n_clips} clips ({time.time() - t0:.0f}s, {len(failed)} failed)", flush=True)
    return done, failed


def build_to_disk(cfg, out_dir, workers=None, render=True):
    out = Path(out_dir)
    for d in ("scenes", "labels", "render", "futures", "maps"):
        (out / d).mkdir(parents=True, exist_ok=True)
    K, T, IMG = cfg["DT_BRANCHES"], cfg["FRAMES"], cfg["IMG"]
    n_clips = -(-cfg["NUM_SAMPLES"] // K)
    N = n_clips * K                     # every clip contributes all K branches
    t0 = time.time()
    done, failed = simulate_split(cfg, out, n_clips, workers)
    print(f"simulated {len(done)} clips in {time.time() - t0:.0f}s; failed: {failed}", flush=True)
    if failed:
        raise RuntimeError(f"{len(failed)} clips could not be built: {failed[:20]}")
    if not render:
        return
    gpus = [int(g) for g in str(cfg["DT_RENDER_GPUS"]).split(",")] if cfg["DT_RENDER_GPUS"] != "auto" else _free_gpus()
    if not gpus:
        raise RuntimeError("no idle GPU to render on (all have >1 GB in use); set DT_RENDER_GPUS")
    print(f"rendering on GPUs {gpus}, {cfg['DT_RENDER_PROCS']} Blender processes each", flush=True)
    render_all(out, N, gpus, cfg["DT_RENDER_PROCS"], cfg["DT_RENDER_SAMPLES"])
    assemble(out, cfg)
    print(f"dataset built to {out} in {time.time() - t0:.0f}s ({N} samples)")


def assemble(out, cfg, only=None):
    """Write videos.npy, annotations.npz and meta.json from labels/ and render/. With ``only``
    (sample ids), videos.npy is updated in place for those samples and the rest is kept."""
    out = Path(out)
    K, T, IMG = cfg["DT_BRANCHES"], cfg["FRAMES"], cfg["IMG"]
    N = -(-cfg["NUM_SAMPLES"] // K) * K
    if only is not None:
        only = set(int(n) for n in only)
    videos = None
    if cfg.get("DT_WRITE_VIDEOS", True):   # the train split keeps its frames in render/ only
        videos = (np.lib.format.open_memmap(out / "videos.npy", mode="w+", dtype=np.uint8, shape=(N, 3, T, IMG, IMG))
                  if only is None else np.load(out / "videos.npy", mmap_mode="r+"))
    keys = None
    ann = {}
    for n in range(N):
        lab = dict(np.load(out / "labels" / f"{n:06d}.npz"))
        r = np.load(out / "render" / f"{n:06d}.npz")
        if videos is not None and (only is None or n in only):
            videos[n] = r["video"].transpose(3, 0, 1, 2)
        lab["cue_visibility"] = cue_counts(r["index"], lab)
        lab.pop("head_info")
        lab.pop("signs")
        if keys is None:
            keys = list(lab)
            ann = {k: np.zeros((N,) + np.shape(lab[k]), np.asarray(lab[k]).dtype) for k in keys}
        for k in keys:
            ann[k][n] = lab[k]
    # Branches share their simulator state up to DT_BRANCH_FRAME byte for byte, but Cycles on
    # different GPUs/processes adds a little noise (mean |diff| <= 0.03 / 255): give every branch
    # branch 0's rendered context frames so the shared past is pixel-identical too.
    if videos is not None:
        for n in range(N):
            b = n % K
            if b and (only is None or n in only):
                videos[n, :, :cfg["DT_BRANCH_FRAME"] + 1] = videos[n - b, :, :cfg["DT_BRANCH_FRAME"] + 1]
        videos.flush()
        del videos
    np.savez(out / "annotations.npz", **ann)
    _assemble_futures(out, N)
    meta = dict(num_samples=N, frames=T, img_size=IMG, frames_future=cfg["FRAMES_FUTURE"], seed=cfg["SEED"],
                branches=K, templates=TEMPLATES, cues=CUES, fps=cfg["DT_FPS"], branch_frame=cfg["DT_BRANCH_FRAME"],
                diversity={k: v for k, v in ASPECTS.items()}, forks=cfg.get("DT_FORKS", 1), choices=CHOICE_KINDS,
                decision_cols=DEC_COLS, driver_cols=DRIVER_COLS + ["cued"], variants=VARIANTS,
                cfg={k: cfg[k] for k in DRIVETOY_DEFAULT_CFG})
    meta.update(version=3, may_go_reasons=MAY_GO_REASONS)
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=list))
