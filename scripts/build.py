#!/usr/bin/env python
"""Build a DriveToy v4 split (docs/drivetoy-v4-design.md) from its plan, plans/<split>.json.

    python scripts/build.py --split train         # 10,000 clips, one ground truth each
    python scripts/build.py --split val           # 1,500 clips with the decision on frame 31, 16 futures aside
    python scripts/build.py --split val_twins     # 1,300 jobs: twin pairs and single clips, 16 futures aside

In train, val and test a plan entry is one clip: [job, situation, member (0 change, 1 carry-on), decision
frame (None: no decision), variant, stage]. A clip whose situation fails every attempt at its frame is retried
with new seeds and then with the FALLBACK situations at the same frame. In val_twins and test_twins an entry
is a twin pair (samples 2i and 2i+1, change twin first) or a single clip: [job, "pair" or "single", situation,
first sample, variant, stage]. The stage is the code state that built the entry (drivetoy/decision.py,
set_stage), so every split rebuilds exactly. ``--new-plan`` deals a fresh plan instead (plan, twin_plan).

The build simulates the clips on the CPUs, renders them on the GPUs and stacks them into annotations.npz,
future_set.npz and, except for train, videos.npy. It resumes where it stopped; status lines go to
<out>/<split>/logs/status.log. Render one split at a time: each Blender process takes about 1.8 GB of a 24 GB
GPU.
"""
import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["LD_LIBRARY_PATH"] = "/root/opt/cuda525:" + os.environ.get("LD_LIBRARY_PATH", "")   # Blender finds the GPUs only so

from drivetoy import decision as dec  # noqa: E402
from drivetoy.builder import (HEAD_KINDS, TEMPLATES, _code, _free_gpus, _save_futures, assemble,  # noqa: E402
                              render_all)
from drivetoy.choices import DEC_COLS, MAX_DECISIONS, end_label  # noqa: E402
from drivetoy.config import DRIVETOY_DEFAULT_CFG  # noqa: E402
from drivetoy.export import scene_dict  # noqa: E402
from drivetoy.map_io import save_map  # noqa: E402

SPLITS = dict(train=100_000_000, val=120_000_000, test=130_000_000, val_twins=160_000_000, test_twins=170_000_000)
TWIN_SPLITS = ("val_twins", "test_twins")
PLANS = ROOT / "plans"
RETRY_SEED = 1_000_003
RETRIES, TRIES = 6, 8                 # train, val and test: seeds per clip, attempts per seed
TWIN_RETRIES, TWIN_TRIES = 12, 40     # twin splits: seeds per job, attempts per seed
PAIRS = SINGLES = 100                 # twin splits: pairs per twin situation, clips per single situation

FRAME_SHARES = {7: 0.16, 15: 0.17, 23: 0.17, 31: 0.23, "other": 0.22, None: 0.05}
OTHER_FRAMES = [f for f in range(1, 31) if f not in (7, 15, 23)]
# (situation, member, weight): member 0 is a twin situation's change, 1 its carry-on; singles use 0
# turning situations weigh more, toward drive_v3's path shares (left 18%, right 16%) over the full 64 frames
CATALOGUE = [
    ("red_green", 0, 5), ("red_green", 1, 3), ("green_yellow", 0, 4), ("green_yellow", 1, 2), ("queue", 0, 4),
    ("queue", 1, 1), ("left_gap", 0, 5), ("left_gap", 1, 3), ("red_ahead", 0, 4), ("yellow_red", 0, 4),
    ("yellow_go", 0, 3), ("left_go", 0, 6), ("right_green", 0, 7), ("left_free", 0, 5), ("right_red", 0, 6),
    ("ped_step", 0, 5), ("ped_step", 1, 3), ("overtake", 0, 4), ("overtake", 1, 2), ("cut_in", 0, 4),
    ("cut_in_far", 0, 2), ("merge_in", 0, 3), ("stop_ahead", 0, 4), ("stop_sign", 0, 7), ("stop_lead", 0, 3),
    ("zone_ahead", 0, 4), ("parked_ahead", 0, 4), ("lane_end", 0, 4), ("lead_brakes_road", 0, 4),
    ("lead_brakes_street", 0, 3), ("parked_wait", 0, 2), ("yellow_lead", 0, 3), ("ped_zone", 0, 3), ("ped_zone", 1, 2),
]
FALLBACK = ["red_ahead", "zone_ahead", "stop_ahead", "cut_in"]   # situations that build at any decision frame
STEADY = ["cruise_junction", "cruise_road", "cruise_street", "follow_road", "follow_street"]


def plan(n, split, seed):
    """[(job, situation, member, decision frame or None, variant)] for n clips."""
    rng = np.random.RandomState(seed)
    shares = FRAME_SHARES if split == "train" else {31: 0.95, None: 0.05}
    keys = list(shares)
    counts = {k: int(round(shares[k] * n)) for k in keys}
    counts[keys[0]] += n - sum(counts.values())
    frames = []
    for k in keys:
        frames += [k] * counts[k]
    frames = [int(rng.choice(OTHER_FRAMES)) if f == "other" else f for f in frames]
    rng.shuffle(frames)
    n_dec = sum(f is not None for f in frames)
    w = np.array([c[2] for c in CATALOGUE], float)
    quota = np.floor(w / w.sum() * n_dec).astype(int)
    quota[np.argsort(-(w / w.sum() * n_dec - quota))[: n_dec - quota.sum()]] += 1
    deck = [i for i, q in enumerate(quota) for _ in range(q)]
    rng.shuffle(deck)
    steady = [STEADY[i % len(STEADY)] for i in range(n - n_dec)]
    rng.shuffle(steady)
    jobs, d, s, stop_case = [], 0, 0, 0
    for j, f in enumerate(frames):
        if f is None:
            jobs.append((j, steady[s], 0, None, None))
            s += 1
            continue
        name, member, _ = CATALOGUE[deck[d]]
        d += 1
        var = None
        if name == "stop_sign":
            var, stop_case = stop_case % len(dec.STOP_CASES), stop_case + 1
        jobs.append((j, name, member, f, var))
    return jobs


def twin_plan(n_pairs, n_singles, counts=None, only=None):
    """[(job, kind, situation, first sample, variant)] in sample order: pairs of every twin situation, then
    single clips; a situation with several cases (the all-way stop) cycles through them. ``counts`` overrides
    the number of pairs or clips of a situation, and ``only`` keeps those situations."""
    counts = counts or {}
    if only:
        counts = {f: (counts.get(f, n_pairs if f in dec.TWINS else n_singles) if f in only else 0)
                  for f in dec.FAMILIES}
    out, n = [], 0
    for f in dec.TWINS:
        for _ in range(counts.get(f, n_pairs)):
            out.append((len(out), "pair", f, n, None))
            n += 2
    for f in dec.SINGLES:
        nv = len(dec.SINGLES[f].get("variants", ()))
        for i in range(counts.get(f, n_singles)):
            out.append((len(out), "single", f, n, i % nv if nv else None))
            n += 1
    return out, n


def load_plan(split, path=None):
    """A split's plan: plans/<split>.json, or the file at path."""
    return [tuple(j) for j in json.loads(Path(path or PLANS / f"{split}.json").read_text())]


def n_samples(split, jobs):
    return sum(2 if j[1] == "pair" else 1 for j in jobs) if split in TWIN_SPLITS else len(jobs)


def split_cfg(split, n, gpus="auto"):
    cfg = dict(DRIVETOY_DEFAULT_CFG)
    cfg.update(NUM_SAMPLES=n, SEED=SPLITS[split], DT_BRANCHES=1, DT_RENDER_GPUS=gpus)
    if split in TWIN_SPLITS:
        cfg.update(DT_WRITE_VIDEOS=True)
    else:
        cfg.update(DT_WRITE_VIDEOS=split != "train", DT_FORKS=1 if split == "train" else 16)
    return cfg


def split_meta(split, counts=None):
    """What a split adds to the meta.json that assemble() writes."""
    if split in TWIN_SPLITS:
        fams = dec.TWIN_SPLIT
        specs = {**dec.TWINS, **dec.SINGLES}
        return dict(families=fams, family_groups={f: specs[f]["group"] for f in fams},
                    twins=[f for f in dec.TWINS if f in fams], singles=[f for f in dec.SINGLES if f in fams],
                    pairs_per_family=PAIRS, singles_per_family=SINGLES, counts=counts or {},
                    outcomes=list(dec.OUTCOMES), stop_cases=list(dec.STOP_CASES), visible_px=dict(
                        lamps=dec.LAMP_PX, pedestrian=dec.PED_PX, stop_sign=dec.SIGN_PX, car=dec.CAR_PX,
                        lane_ends_sign=dec.LANE_SIGN_PX))
    return dict(families=dec.FAMILIES, frame_shares={str(k): v for k, v in FRAME_SHARES.items()}
                if split == "train" else {"31": 0.95, "None": 0.05}, catalogue=CATALOGUE, steady=STEADY,
                stop_cases=list(dec.STOP_CASES), outcomes=list(dec.OUTCOMES), forks=split_cfg(split, 1)["DT_FORKS"])


def _readable(f):
    try:
        with np.load(f) as z:
            return z["video"].shape[0] == 32 and z["index"].shape[0] == 32
    except Exception:
        return False


def _drop_broken_renders(out):
    """Delete render files that do not load (a Blender process killed while writing), so they are redone."""
    files = sorted((out / "render").glob("[0-9]*.npz"))
    with ProcessPoolExecutor(16) as ex:
        bad = [f for f, ok in zip(files, ex.map(_readable, files, chunksize=64)) if not ok]
    for f in bad:
        f.unlink()
    return len(bad)


def _save(out, n, res, b, cfg, seed, pair, twin, frame=True):
    rec = res["branches"][b]
    sc = scene_dict(dict(res, colors=res["colors"][b]), b, cfg, seed)   # each twin its own agents' colours
    with open(out / "scenes" / f"{n:06d}.json", "w") as f:
        json.dump(sc, f)
    E = cfg["DT_MAX_EVENTS"]
    ev = rec["events"][:E]
    events = np.full((E, 5), -1, np.float32)
    events[:len(ev)] = ev
    ex = rec["extra"]
    lab = dict(res["labels"][b], E=end_label(rec["decisions"], dec.T0, dec.T0 + 9))
    draws = res["route_draws"][b]
    d = rec["decisions"][np.argsort(-rec["decisions"][:, 6], kind="stable")][:MAX_DECISIONS]   # the ego's first
    decisions = np.full((MAX_DECISIONS, len(DEC_COLS)), -1, np.float32)
    decisions[:len(d)] = d
    _save_futures(out / "futures" / f"{n:06d}.npz", res["forks"][b])
    save_map(out / "maps" / f"{n:06d}.json.gz", res["map"])
    fields = dict(decisions=decisions, num_decisions=np.int16(len(d)), drivers=rec["drivers"], variant=np.int8(res["variant"]),
                  open_moves=np.array([mv in res["open_moves"] for mv in "LSR"], bool),
                  future_move=np.array(["LSR".index(x) for x in draws] if draws is not None else [-1] * len(res["forks"][b]), np.int8),
                  agents=rec["agents"], lamps=rec["lamps"], signals=rec["signals"], ego=rec["ego"],
                  lane=ex[..., 0].astype(np.int16), s_local=ex[..., 1], indicator=ex[..., 2].astype(np.int8),
                  rule_state=ex[..., 3].astype(np.int8), yield_to=ex[..., 4].astype(np.int8),
                  ped_state=ex[..., 5].astype(np.int8), lc=rec["lc"], routes=rec["routes"],
                  events=events, num_events=np.int32(min(len(rec["events"]), E)),
                  clip_id=np.int32(n), branch_id=np.int8(0), template=np.int8(TEMPLATES.index(res["template"])),
                  sim_ok=np.bool_(True), sim_seed=np.int64(seed), tries=np.int16(res["tries"]),
                  diversity=np.array([_code(k, lab.get(k)) for k in "ABCDE"], np.int8),
                  target=np.full(5, -1, np.int8),
                  camera=np.array([sc["camera"]["height"], sc["camera"]["pitch"], sc["camera"]["fov"]], np.float32),
                  head_info=np.array([[HEAD_KINDS.index(h["kind"]), sc["groups"].index(h["group"]) if h["group"] in sc["groups"] else -1,
                                       h["leg"]] for h in sc["heads"]] or np.zeros((0, 3)), np.int8).reshape(-1, 3),
                  signs=np.array([s["kind"] == "stop" for s in sc["signs"]], bool),
                  may_go=rec["may_go"], may_go_reason=rec["may_go_reason"], ego_turn=rec["ego_turn"],
                  route_drawn=np.bool_(res["route_draws"][b] is not None),
                  pair=np.int32(pair), twin=np.int8(twin), family=np.int8(dec.FAMILIES.index(res["family"])),
                  cue_frame=np.int8(res["cue_frame"]), outcome=res["outcomes"][b].astype(np.int8),
                  change=res["change"][b].astype(bool), onset=res["onsets"][b].astype(np.int8))
    if frame:              # the twin splits leave the decision frame out: always 31 there
        fields["decision_frame"] = np.int8(res.get("decision_frame", 31))
    np.savez(out / "labels" / f"{n:06d}.npz", **fields)


def _merge_futures(out):
    """assemble() stacks futures/ into future_set.npz only on a full build; after samples are rebuilt into a
    finished split, put their futures into the existing future_set.npz instead."""
    fdir = out / "futures"
    if not fdir.exists() or not (out / "future_set.npz").exists():
        return
    fs = dict(np.load(out / "future_set.npz"))
    for f in sorted(fdir.glob("*.npz")):
        new = np.load(f)
        for k in new.files:
            fs[k][int(f.stem)] = new[k]
    np.savez(out / "future_set.npz", **fs)
    shutil.rmtree(fdir)


def _run_clip(task):
    j, name, member, fd, var, stage, cfg, out, base = task
    dec.set_stage(stage)
    t0 = time.time()
    reasons, tried = {}, []
    names = [name] + [x for x in FALLBACK if x != name] if fd is not None else [name]
    for alt, nm in enumerate(names[:4]):
        mb = member if alt == 0 else 0
        v = var if nm == "stop_sign" and alt == 0 else (int(j % len(dec.STOP_CASES)) if nm == "stop_sign" else None)
        for r in range(RETRIES if alt == 0 else 2):
            seed = base + j * 101 + r * RETRY_SEED + alt * 7_777_777
            res = dec.simulate_clip(seed, cfg, nm, fd, member=mb, variant=v, tries=TRIES)
            if res["ok"]:
                _save(out, j, res, 0, cfg, seed, pair=-1, twin=res["members"][0])
                return j, True, time.time() - t0, reasons, nm
            for k, val in res["reasons"].items():
                reasons[f"{nm}: {k}"] = reasons.get(f"{nm}: {k}", 0) + val
        tried.append(nm)
    return j, False, time.time() - t0, reasons, None


def _run_twin(task):
    j, kind, fam, n, variant, stage, cfg, out, base = task
    dec.set_stage(stage)
    dec.set_decision_frame(31)
    t0 = time.time()
    fn = dec.simulate_pair if kind == "pair" else dec.simulate_single
    reasons = {}
    for r in range(TWIN_RETRIES):
        seed = base + j * 101 + r * RETRY_SEED
        res = fn(seed, cfg, fam, tries=TWIN_TRIES, variant=variant)
        if res["ok"]:
            break
        for k, v in res["reasons"].items():
            reasons[k] = reasons.get(k, 0) + v
    else:
        return j, False, time.time() - t0, reasons
    if kind == "pair":
        for b in range(2):
            _save(out, n + b, res, b, cfg, seed, pair=n // 2, twin=b, frame=stage == "final")
    else:
        _save(out, n, res, 0, cfg, seed, pair=-1, twin=-1, frame=stage == "final")
    return j, True, time.time() - t0, reasons

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=list(SPLITS))
    ap.add_argument("--out", default="/root/projects/drive_v4")
    ap.add_argument("--plan", default=None, help="plan file (default plans/<split>.json)")
    ap.add_argument("--new-plan", action="store_true", help="deal a fresh plan instead of reading one")
    ap.add_argument("--n", type=int, default=None, help="--new-plan: clips (train 10,000, val and test 1,500)")
    ap.add_argument("--pairs", type=int, default=PAIRS, help="--new-plan, twin splits: pairs per twin situation")
    ap.add_argument("--singles", type=int, default=SINGLES, help="--new-plan, twin splits: clips per single situation")
    ap.add_argument("--count", nargs="*", default=[], help="--new-plan, twin splits: name=n for one situation")
    ap.add_argument("--only", nargs="*", default=None, help="--new-plan, twin splits: only these situations")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--gpus", default="auto")
    ap.add_argument("--no-render", action="store_true", help="simulate only")
    args = ap.parse_args()
    twins = args.split in TWIN_SPLITS
    out = Path(args.out) / args.split
    for sub in ("scenes", "labels", "render", "futures", "maps", "logs"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    status = out / "logs" / "status.log"

    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(time.time() + 9 * 3600))} KST {msg}"
        print(line, flush=True)
        with open(status, "a") as f:
            f.write(line + "\n")

    base = SPLITS[args.split]
    counts = {k: int(v) for k, v in (c.split("=") for c in args.count)}
    if args.new_plan and twins:
        jobs = [(*j, "clean") for j in twin_plan(args.pairs, args.singles, counts, args.only or dec.TWIN_SPLIT)[0]]
    elif args.new_plan:
        jobs = [(*j, "clean") for j in plan(args.n or (10_000 if args.split == "train" else 1_500), args.split, base + 17)]
    else:
        jobs = load_plan(args.split, args.plan)
    (out / "logs" / "plan.json").write_text(json.dumps(jobs))
    n = n_samples(args.split, jobs)
    cfg = split_cfg(args.split, n, args.gpus)
    if twins:
        left = [t for t in jobs if not (out / "labels" / f"{t[3] + (1 if t[1] == 'pair' else 0):06d}.npz").exists()]
    else:
        left = [t for t in jobs if not (out / "labels" / f"{t[0]:06d}.npz").exists()]
    log(f"start {args.split}: {len(jobs)} jobs ({n} samples), {len(left)} to simulate")
    t0, failed, done, built = time.time(), [], 0, {}
    with ProcessPoolExecutor(max_workers=args.workers or os.cpu_count()) as ex:
        for res in ex.map(_run_twin if twins else _run_clip, [(*t, cfg, out, base) for t in left], chunksize=1):
            j, ok, reasons = res[0], res[1], res[3]
            done += 1
            if ok and not twins:
                built[res[4]] = built.get(res[4], 0) + 1
            if not ok:
                failed.append(j)
                log(f"job {j} ({jobs[j][2] if twins else jobs[j][1]}) failed: "
                    f"{sorted(reasons.items(), key=lambda kv: -kv[1])[:3]}")
            if done % 250 == 0 or done == len(left):
                log(f"simulated {done}/{len(left)} in {time.time() - t0:.0f}s, {len(failed)} failed")
    if built:
        log(f"situations built: {json.dumps(dict(sorted(built.items())))}")
    if failed:
        log(f"FAILED: {len(failed)} jobs could not be built")
        raise SystemExit(1)
    if args.no_render:
        log("simulation done (no render)")
        return
    gpus = [int(g) for g in str(args.gpus).split(",")] if args.gpus != "auto" else _free_gpus()
    if not gpus:
        log("FAILED: no idle GPU to render on")
        raise SystemExit(1)
    broken = _drop_broken_renders(out)
    log(f"rendering on GPUs {gpus}" + (f" ({broken} unreadable renders redone)" if broken else ""))
    render_all(out, n, gpus, cfg["DT_RENDER_PROCS"], cfg["DT_RENDER_SAMPLES"])
    log("render done; assembling")
    assemble(out, cfg)
    _merge_futures(out)
    meta = json.loads((out / "meta.json").read_text())
    meta.update(split_meta(args.split, counts))
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=list))
    log(f"DONE {n} samples")


if __name__ == "__main__":
    main()
