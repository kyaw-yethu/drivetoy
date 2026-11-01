"""Full records of each sample's 16 futures, re-simulated from its seed and target.

The split keeps only part of each future (``future_set.npz``), while scoring a plan needs every road
user's lane, arclength and size, and the signals and lane changes of each future. They come from
running the simulator again with ``DT_KEEP_FORK_RECS`` on the code that built the split, and each
sample is checked against its labels and ``future_set.npz``.

    python3 -m bench.records --data /root/projects/drive_v3/val --out /root/projects/drive_v3_bench/val/records
"""
import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from . import code

KEYS = ('agents', 'extra', 'signals', 'lc', 'routes', 'may_go', 'may_go_reason')
DATA = OUT = META = None


def resimulate(n, data, meta):
    """The 16 full records of sample n (future 0 is the logged one) and its map."""
    from drivetoy.diversity import ASPECTS
    from drivetoy.scenarios import VARIANTS, simulate
    lab = np.load(Path(data) / 'labels' / f'{n:06d}.npz')
    cfg = dict(meta['cfg'], DT_KEEP_FORK_RECS=True)
    tgt = {k: (ASPECTS[k][c] if c >= 0 else None) for k, c in zip('ABCDE', lab['target'])}
    tgt['template'] = meta['templates'][int(lab['template'])]
    tgt['V'] = VARIANTS[int(lab['variant'])] if int(lab['variant']) >= 0 else None
    sim = simulate(int(lab['sim_seed']), cfg, target=tgt)
    recs = sim['fork_recs'][int(lab['branch_id'])]
    same = len(recs) == 16 and bool(np.array_equal(recs[0]['agents'], lab['agents']))
    return recs, sim['map'], same


def _one(n):
    t0 = time.time()
    recs, _, same = resimulate(n, DATA, META)
    out = {k: np.stack([r[k] for r in recs]) for k in KEYS}
    np.savez_compressed(OUT / f'{n:06d}.npz', **out)
    T0 = META['frames']
    short = out['agents'][:, T0:][..., [1, 2, 3, 4, 9]].astype(np.float16)
    return n, same, short, out['may_go'][:, T0:], time.time() - t0


def _init(root, data, out, meta):
    global DATA, OUT, META
    code.use(root)
    DATA, OUT, META = data, out, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--code', default=code.DEFAULT)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=None)
    ap.add_argument('--workers', type=int, default=96)
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((data / 'meta.json').read_text())
    N = args.n or meta['num_samples']
    todo = [n for n in range(N) if not (out / f'{n:06d}.npz').exists()]
    if not todo:
        print(f'all {N} records exist; see records.json')
        return
    with np.load(data / 'future_set.npz') as z:
        fs_agents, fs_may_go = z['agents'], z['may_go']
    bad, t0 = [], time.time()
    with Pool(args.workers, initializer=_init, initargs=(args.code, data, out, meta)) as pool:
        for i, (n, same, short, may_go, dt) in enumerate(pool.imap_unordered(_one, todo)):
            if not (same and np.array_equal(short, fs_agents[n]) and np.array_equal(may_go, fs_may_go[n])):
                bad.append(n)
            if i % 100 == 0:
                print(f'{i + 1}/{len(todo)} sample {n} {dt:.0f}s elapsed {time.time() - t0:.0f}s', flush=True)
    report = dict(n=N, done=len(todo), mismatch=sorted(bad), code=str(args.code), seconds=round(time.time() - t0))
    (out / 'records.json').write_text(json.dumps(report, indent=1))
    print(report)


if __name__ == '__main__':
    main()
