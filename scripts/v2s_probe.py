#!/usr/bin/env python
"""Quick look at the stochastic choices on a few simulated targets: how often each kind of
choice fires (all agents and the ego's), the odds it was drawn with, and rule violations.

    python scripts/v2s_probe.py --n 40 [--workers 40]
"""
import argparse
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from f_toy.datasets.drivetoy.choices import CHOICE_KINDS, end_label
from f_toy.datasets.drivetoy.config import DRIVETOY_DEFAULT_CFG
from f_toy.datasets.drivetoy.diversity import ASPECTS
from f_toy.datasets.drivetoy.rules import check
from f_toy.datasets.drivetoy.scenarios import COMBOS, simulate

CFG = dict(DRIVETOY_DEFAULT_CFG)


def one(i):
    rng = np.random.RandomState(i)
    row = COMBOS[int(rng.randint(len(COMBOS)))]
    A, B, tmpl, D, C = row[:5]
    tg = dict(A=A, B=B, C=C, D=D, template=tmpl)
    if len(row) > 5:
        tg["E"] = row[5]
    r = simulate(90_000_000 + i, CFG, target=tg)
    if not r["ok"]:
        return i, tg, None
    rec = r["branches"][0]
    viol = check(rec, r["map"], CFG)
    return i, tg, dict(dec=rec["decisions"], viol=[(int(f), k, int(a), str(x)) for f, k, a, x in viol],
                       labels=r["labels"][0], E=end_label(rec["decisions"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--workers", type=int, default=40)
    args = ap.parse_args()
    with Pool(args.workers) as p:
        res = p.map(one, range(args.n), chunksize=1)
    ok = [r for r in res if r[2] is not None]
    print(f"{len(ok)}/{args.n} targets built")
    kinds, ego_kinds, ps, viol, E = Counter(), Counter(), [], Counter(), Counter()
    for i, tg, r in ok:
        d = r["dec"]
        for row in d:
            kinds[CHOICE_KINDS[int(row[2])]] += 1
            if row[6] > 0:
                ego_kinds[CHOICE_KINDS[int(row[2])]] += 1
                ps.append(row[4])
        for f, k, a, x in r["viol"]:
            viol[k] += 1
            print("  violation", i, tg, f, k, a, x)
        E[r["E"]] += 1
    print("choices (all agents):", dict(kinds))
    print("choices concerning the ego:", dict(ego_kinds))
    if ps:
        print("their odds: deciles", np.round(np.percentile(ps, np.arange(0, 101, 10)), 2).tolist())
    print("E:", dict(E), " violations:", dict(viol))


if __name__ == "__main__":
    main()
