"""check_ego returns exactly the ego entries of rules.check that TRC can count, for every plan of
every source and every future of a few val samples. Run in the navsim env from the repo root:

    python bench/tests/test_rules_ego.py [sample ...]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np

from bench import code, score

BENCH = Path('/root/projects/drive_v3_bench/val')
DATA = Path('/root/projects/drive_v3/val')
TRC = ('R0', 'R1', 'R3', 'R4', 'R5', 'R6')


def main(samples):
    code.use()
    from drivetoy.map_io import load_map
    from drivetoy.rules import check
    from bench import navsim_io as nv, trc
    from bench.rules_ego import check_ego
    cfg = json.loads((DATA / 'meta.json').read_text())['cfg']
    plans = {}
    for p in sorted((BENCH / 'plans').glob('*.npz')):
        with np.load(p) as z:
            plans[p.stem] = z['paths']
    pairs = hits = 0
    for n in samples:
        rec = dict(np.load(BENCH / 'records' / f'{n:06d}.npz'))
        m = load_map(DATA / 'maps' / f'{n:06d}.json.gz')
        ag0 = rec['agents'][0]
        vp = nv.vehicle_parameters(*ag0[nv.T0, 0, 6:9])
        init = nv.ego_states(ag0, [nv.T0], vp)[0]
        paths = score.Paths()
        for p in plans.values():
            for q in p[n]:
                paths.add(q)
        for k in range(16):
            paths.add(score.relative(rec['agents'][k][:, 0], nv.T0))
        paths.add(score.constant(float(ag0[nv.T0, 0, 4])))
        C = nv.centres(nv.track(np.stack(paths.paths), init, nv.simulator()), vp)
        matcher = trc.Matcher(m)
        lane0 = int(rec['extra'][0, nv.T0, 0, 0])
        for c in C:
            lanes, ss = matcher.match(c[:, :2], c[:, 2], lane0)
            for k in range(16):
                r = trc.splice(rec, k, c, lanes, ss, matcher, cfg['DT_FPS'], nv.T0)
                full = sorted(v for v in check(r, m, cfg) if v[2] == 0 and v[1] in TRC)
                ego = sorted(check_ego(r, m, cfg))
                assert full == ego, (n, k, full, ego)
                pairs += 1
                hits += bool(ego)
        print(f'sample {n}: {len(paths.paths)} paths x 16 futures agree', flush=True)
    print(f'ok: {pairs} plan-future pairs, {hits} with ego violations')


if __name__ == '__main__':
    main([int(x) for x in sys.argv[1:]] or [1, 2, 3])
