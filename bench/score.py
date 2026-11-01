"""NAVSIM-style scores of plans on DriveToy (docs/task_encoder_eval.md).

Every plan of every source and the reference plans are tracked once from the ego state at frame 31
(the earlier plans, for EC, from frame 29) and then scored in each of the 16 futures: NC, DAC, TTC
and HC by the NAVSIM scorer, TRC by bench.trc, EC against the earlier plan of the same source.
Within each future, a term (for TRC, a rule) that the own ego of the future fails after the same
conversion counts as passed, which is the NAVSIM human filter.

A plans file holds paths (N, M, 20, 3), weights (N, M) with the plan first and zero for unused
slots, and prev (N, 20, 3), the plan made at frame 29; poses are (x forward, y left, heading) of the
ego centre in the ego frame at the frame of the plan.

    python -m bench.score --plans DIR --records DIR --data DIR --out FILE
"""
import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from . import code

TERMS = ('NC', 'DAC', 'TRC', 'TTC', 'HC', 'EC')
REFS = ('gt_rec', 'cv', 'stop')
FUTURES, F = 16, 20
G = {}


def episode_score(t):
    """NC * DAC * TRC * (5 TTC + 2 HC + 2 EC) / 9 over the last axis of (..., 6) terms."""
    return t[..., 0] * t[..., 1] * t[..., 2] * (5 * t[..., 3] + 2 * t[..., 4] + 2 * t[..., 5]) / 9


def relative(ego, o):
    """Recorded centre poses of frames o+1 .. o+20 in the ego frame at frame o: (20, 3)."""
    x, y, h = ego[o, 1:4]
    d = ego[o + 1:o + 1 + F, 1:3] - (x, y)
    c, s = np.cos(h), np.sin(h)
    a = ego[o + 1:o + 1 + F, 3] - h
    return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], np.arctan2(np.sin(a), np.cos(a))], -1)


def constant(v, decel=0.0):
    """Straight path at speed v, braking at decel to a standstill: (20, 3)."""
    t = np.arange(1, F + 1) / 5.0
    s = v * t - decel * t ** 2 / 2 if decel <= 0 else np.where(t < v / decel, v * t - decel * t ** 2 / 2, v ** 2 / (2 * decel))
    return np.stack([s, 0 * t, 0 * t], -1)


class Paths:
    """Unique paths of one origin, so that a path shared by two sources is tracked and scored once."""
    def __init__(self):
        self.paths, self.keys = [], {}

    def add(self, p):
        key = np.round(np.asarray(p, np.float64), 4).tobytes()
        if key not in self.keys:
            self.keys[key] = len(self.paths)
            self.paths.append(np.asarray(p, np.float64))
        return self.keys[key]


def score_sample(n):
    """Terms of every source in sample n: dict source -> arrays (see main)."""
    from drivetoy.map_io import load_map
    from drivetoy.rules import _route_of
    from . import navsim_io as nv, trc
    rec = dict(np.load(G['records'] / f'{n:06d}.npz'))
    m = load_map(G['data'] / 'maps' / f'{n:06d}.json.gz')
    cfg, ag0 = G['cfg'], rec['agents'][0]
    vp = nv.vehicle_parameters(*ag0[nv.T0, nv.EGO, 6:9])
    sim = nv.simulator()
    init_now, init_prev = nv.ego_states(ag0, [nv.T0, nv.PREV], vp)
    now, prev, src = Paths(), Paths(), {}
    for name, z in G['plans'].items():
        w = z['weights'][n]
        keep = np.flatnonzero(w > 0)
        src[name] = ([now.add(p) for p in z['paths'][n][keep]], w[keep], prev.add(z['prev'][n]))
    ego0 = ag0[:, nv.EGO]
    v_now, v_prev = float(ego0[nv.T0, 4]), float(ego0[nv.PREV, 4])
    one = np.ones(1)
    src['gt_rec'] = ([now.add(relative(ego0, nv.T0))], one, prev.add(relative(ego0, nv.PREV)))
    src['cv'] = ([now.add(constant(v_now))], one, prev.add(constant(v_prev)))
    src['stop'] = ([now.add(constant(v_now, 3.0))], one, prev.add(constant(v_prev, 3.0)))
    selfs = [(now.add(relative(rec['agents'][k][:, nv.EGO], nv.T0)), prev.add(relative(rec['agents'][k][:, nv.EGO], nv.PREV)))
             for k in range(FUTURES)]

    S_now = nv.track(np.stack(now.paths), init_now, sim)
    S_prev = nv.track(np.stack(prev.paths), init_prev, sim)
    C = nv.centres(S_now, vp)
    matcher = trc.Matcher(m)
    lane0 = int(rec['extra'][0, nv.T0, nv.EGO, 0])
    matched = [matcher.match(c[:, :2], c[:, 2], lane0) for c in C]
    pairs = sorted({(i, s[2]) for s in src.values() for i in s[0]} | set(selfs))
    ec = dict(zip(pairs, nv.extended_comfort(S_now[[i for i, _ in pairs]], S_prev[[j for _, j in pairs]])))

    dmap, past = nv.drivable_map(m), nv.past_trajectory(ag0, vp)
    route = _route_of({'routes': rec['routes'][0]}, nv.EGO, rec['extra'][0][..., 0].astype(int), m)['ids']
    line, route_ids = nv.centreline(m, route), [f'lane_{i}' for i in route]
    shared = sorted({i for s in src.values() for i in s[0]})
    P = len(now.paths)
    nav = np.ones((FUTURES, P, 4))
    hits = np.zeros((FUTURES, P, len(trc.RULES)), bool)
    for k in range(FUTURES):
        idx = sorted(set(shared) | {selfs[k][0]})
        obs = nv.observation(nv.detections(rec['agents'][k], m.statics))
        nav[k, idx] = nv.score(S_now[idx], obs, line, route_ids, dmap, past, vp)
        for i in idx:
            hits[k, i] = trc.rule_hits(rec, k, C[i], *matched[i], matcher, m, cfg, nv.T0)

    def raw(k, i, j):
        return np.array([nav[k, i, 0], nav[k, i, 1], 1.0 - hits[k, i].any(), nav[k, i, 2], nav[k, i, 3], ec[(i, j)]])

    out = {}
    gt = np.stack([raw(k, *selfs[k]) for k in range(FUTURES)])
    gt_hits = np.stack([hits[k, selfs[k][0]] for k in range(FUTURES)])
    out['gt_self'] = dict(raw=gt[:, None], rules=gt_hits[:, None], weights=one)
    for name, (ids, w, j) in src.items():
        r = np.stack([[raw(k, i, j) for i in ids] for k in range(FUTURES)])          # (16, M, 6)
        h = hits[:, ids]                                                              # (16, M, 9)
        hf = h & ~gt_hits[:, None]
        t = np.where(gt[:, None] < 1, 1.0, r)
        t[..., 2] = 1.0 - hf.any(-1)
        out[name] = dict(raw=r, terms=t, rules=hf, weights=w)
    return out


def _init(root, records, data, plans, cfg):
    code.use(root)
    G.update(records=Path(records), data=Path(data), plans=plans, cfg=cfg)


def _run(n):
    t0 = time.time()
    return n, score_sample(n), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--code', default=code.DEFAULT)
    ap.add_argument('--plans', required=True)
    ap.add_argument('--records', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=None)
    ap.add_argument('--workers', type=int, default=96)
    args = ap.parse_args()
    data = Path(args.data)
    meta = json.loads((data / 'meta.json').read_text())
    plans = {}
    for p in sorted(Path(args.plans).glob('*.npz')):
        with np.load(p) as z:
            plans[p.stem] = {k: z[k] for k in ('paths', 'weights', 'prev')}
    N = args.n or meta['num_samples']
    res, t0 = [None] * N, time.time()
    with Pool(args.workers, initializer=_init, initargs=(args.code, args.records, data, plans, meta['cfg'])) as pool:
        for i, (n, r, dt) in enumerate(pool.imap_unordered(_run, range(N))):
            res[n] = r
            if i % 50 == 0:
                print(f'{i + 1}/{N} sample {n} {dt:.0f}s elapsed {time.time() - t0:.0f}s', flush=True)
    arrays = {}
    for name in res[0]:
        M = max(len(r[name]['weights']) for r in res)
        for key, fill, dtype in (('raw', np.nan, np.float32), ('terms', np.nan, np.float32),
                                 ('rules', False, bool), ('weights', 0.0, np.float32)):
            if key not in res[0][name]:
                continue
            a0 = res[0][name][key]
            full = np.full((N, M) if key == 'weights' else (N, FUTURES, M) + a0.shape[2:], fill, dtype)
            for n, r in enumerate(res):
                a = r[name][key]
                if key == 'weights':
                    full[n, :len(a)] = a
                else:
                    full[n, :, :a.shape[1]] = a
            arrays[f'{name}__{key}'] = full
    ann = np.load(data / 'annotations.npz')
    np.savez_compressed(args.out, diversity=ann['diversity'][:N], template=ann['template'][:N],
                        terms=np.array(TERMS), **arrays)
    print(f'wrote {args.out} in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
