"""Full records of each sample's 16 futures, re-simulated from its plan entry and seed.

The split keeps only part of each future (``future_set.npz``), while scoring a plan needs every road user's lane,
arclength and size, and the signals and lane changes of each future. They come from running the simulator again
with ``DT_KEEP_FORK_RECS`` on the code that built the split, as scripts/build.py ran the sample's plan entry, with
the seed and situation in its labels. Each sample is checked against its labels and ``future_set.npz``.

    python3 -m bench.records --data /root/projects/drive_v4/val --out /root/projects/drive_v4_bench/val/records
"""
import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from . import code

KEYS = ('agents', 'extra', 'signals', 'lc', 'routes', 'may_go', 'may_go_reason')
# future_set keys compared with the re-simulation (decisions is stored sorted and padded, and the bench skips it)
CHECKED = ('ego', 'agents', 'lamps', 'signals', 'rule_state', 'yield_to', 'may_go', 'may_go_reason', 'ego_turn')
G = {}


def samples_of(entry, twins):
    """Sample numbers of one plan entry: a clip, a twin pair (two samples) or a single clip of a twin split."""
    if not twins:
        return [entry[0]]
    _, kind, _, n, _, _ = entry
    return [n, n + 1] if kind == 'pair' else [n]


def resimulate(entry, split, data, meta):
    """[(n, its 16 full records, future 0 the logged one, or None if the simulation failed)] of one plan entry."""
    from drivetoy import decision as dec
    b = code.build(G['root'])
    cfg = dict(meta['cfg'], DT_KEEP_FORK_RECS=True)
    twins = split in b.TWIN_SPLITS
    ns = samples_of(entry, twins)
    lab = np.load(Path(data) / 'labels' / f'{ns[0]:06d}.npz')
    seed = int(lab['sim_seed'])
    if twins:
        _, kind, fam, _, variant, stage = entry
        dec.set_stage(stage)
        dec.set_decision_frame(31)
        fn = dec.simulate_pair if kind == 'pair' else dec.simulate_single
        res = fn(seed, cfg, fam, tries=b.TWIN_TRIES, variant=variant)
    else:
        _, name, member, fd, _, stage = entry
        built = dec.FAMILIES[int(lab['family'])]       # build.FALLBACK may have replaced the planned situation
        dec.set_stage(stage)
        res = dec.simulate_clip(seed, cfg, built, fd, member=member if built == name else 0,
                                variant=int(lab['variant']) if int(lab['variant']) >= 0 else None, tries=b.TRIES)
    if not res.get('ok'):
        return [(n, None) for n in ns]
    return list(zip(ns, res['fork_recs']))


def _one(entry):
    from drivetoy import decision as dec
    from drivetoy.scenarios import _future
    t0, out = time.time(), []
    for n, recs in resimulate(entry, G['split'], G['data'], G['meta']):
        if recs is None:
            out.append((n, False, None))
            continue
        lab = np.load(G['data'] / 'labels' / f'{n:06d}.npz')
        same = len(recs) == 16 and bool(np.array_equal(recs[0]['agents'], lab['agents']))
        np.savez_compressed(G['out'] / f'{n:06d}.npz', **{k: np.stack([r[k] for r in recs]) for k in KEYS})
        futs = [_future(r, dec.T0) for r in recs]
        out.append((n, same, {k: np.stack([f[k] for f in futs]) for k in CHECKED}))
    return out, time.time() - t0


def _init(root, split, data, out, meta):
    code.use(root)
    G.update(root=root, split=split, data=Path(data), out=Path(out), meta=meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--code', default=code.DEFAULT)
    ap.add_argument('--data', required=True, help='a drive_v4 split folder; its name is the split')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=None, help='only the first n samples')
    ap.add_argument('--workers', type=int, default=64)
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((data / 'meta.json').read_text())
    split = data.name
    b = code.build(args.code)
    N = args.n or meta['num_samples']
    entries = [e for e in b.load_plan(split) if any(n < N for n in samples_of(e, split in b.TWIN_SPLITS))]
    todo = [e for e in entries if not all((out / f'{n:06d}.npz').exists() for n in samples_of(e, split in b.TWIN_SPLITS))]
    if not todo:
        print(f'all {N} records exist; see records.json')
        return
    with np.load(data / 'future_set.npz') as z:
        fs = {k: z[k] for k in CHECKED}
    bad, failed, t0, done = [], [], time.time(), 0
    with Pool(args.workers, initializer=_init, initargs=(args.code, split, data, out, meta)) as pool:
        for i, (res, dt) in enumerate(pool.imap_unordered(_one, todo)):
            for n, same, futs in res:
                done += 1
                if futs is None:
                    failed.append(n)
                elif not (same and all(np.array_equal(futs[k], fs[k][n]) for k in CHECKED)):
                    bad.append(n)
            if i % 100 == 0:
                print(f'{i + 1}/{len(todo)} entries, {dt:.0f}s for the last, elapsed {time.time() - t0:.0f}s', flush=True)
    report = dict(split=split, n=N, done=done, mismatch=sorted(bad), failed=sorted(failed), code=str(args.code),
                  seconds=round(time.time() - t0))
    (out / 'records.json').write_text(json.dumps(report, indent=1))
    print(report)


if __name__ == '__main__':
    main()
