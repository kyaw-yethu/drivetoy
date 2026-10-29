#!/usr/bin/env python
"""Does the DriveToy rule checker catch real violations? (docs/DRIVETOY.md §9)

For every violation set the simulator switches one rule off after
DT_BRANCH_FRAME and logs each illegal act as a VIOLATE event (sim.py).
The checker (rules.py) never sees those logs; it is scored against them:

  act recall  — logged acts the checker flags (same rule, same agent, within ±3 frames)
  clip rate   — clips flagged for the set's rule (for 'tailgate' and 'speed',
                which have no per-act log, this is the only number)
  lawful rate — the same rule flagged on the lawful clips with the same seeds

    python scripts/drivetoy_violation_check.py --n 300 --out analysis/drivetoy_violations
"""
import argparse
import collections
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f_toy.datasets.drivetoy.config import DRIVETOY_DEFAULT_CFG
from f_toy.datasets.drivetoy.rules import check
from f_toy.datasets.drivetoy.scenarios import simulate
from f_toy.datasets.drivetoy.sim import EV, VIOLATION_RULES, VIOLATIONS

SETS = dict(red=["signal4"], left_gap=["signal4"], no_stop=["stop4"], ped=["signal4", "stop4", "residential"],
            tailgate=["multilane", "signal4"], solid=["multilane"], no_signal=["multilane"], speed=["multilane", "residential"],
            jaywalk=["signal4"], ped_dart=["stop4", "residential"])


def one(task):
    kind, seed, templates = task
    tmpl = templates[seed % len(templates)]
    out = {}
    for v in (kind, None):
        cfg = dict(DRIVETOY_DEFAULT_CFG, DT_VIOLATION=v)
        r = simulate(seed, cfg, template=tmpl)
        rec = r["branches"][0]
        flags = check(rec, r["map"], cfg)
        acts = [(int(f), int(a), VIOLATION_RULES[int(val)]) for f, t, a, b, val in rec["events"]
                if int(t) == EV["VIOLATE"] and 0 <= f < rec["agents"].shape[0]]
        out["viol" if v else "lawful"] = dict(flags=[(int(f), r_, int(a)) for f, r_, a, _ in flags], acts=acts)
    return kind, seed, tmpl, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()
    tasks = [(k, 10_000 + i, SETS[k]) for k in args.sets.split(",") for i in range(args.n)]
    res = collections.defaultdict(list)
    with ProcessPoolExecutor(args.workers) as ex:
        for kind, seed, tmpl, out in ex.map(one, tasks, chunksize=2):
            res[kind].append((seed, tmpl, out))
    table = {}
    print(f"{'set':10s} {'rule':5s} {'acts':>6s} {'recall':>7s} {'clips w/ act':>12s} {'clip rate':>9s} {'lawful':>7s}  other rules flagged (clips)")
    for kind, rows in res.items():
        rule = VIOLATIONS[kind]
        n_acts = hit = 0
        clips_act = clips_flag = lawful_flag = 0
        other = collections.Counter()
        misses = []
        for seed, tmpl, out in rows:
            v, l = out["viol"], out["lawful"]
            flagged = [(f, a) for f, r_, a in v["flags"] if r_ == rule]
            for f, a, r_ in v["acts"]:
                n_acts += 1
                if any(fa == a and abs(ff - f) <= 3 for ff, fa in flagged):
                    hit += 1
                elif len(misses) < 5:
                    misses.append((seed, tmpl, f, a))
            clips_act += bool(v["acts"])
            clips_flag += bool(flagged)
            lawful_flag += any(r_ == rule for _, r_, _ in l["flags"])
            for r_ in {r_ for _, r_, _ in v["flags"] if r_ != rule}:
                other[r_] += 1
        n = len(rows)
        rec_s = f"{hit / n_acts:.2f}" if n_acts else "  -"
        print(f"{kind:10s} {rule:5s} {n_acts:6d} {rec_s:>7s} {clips_act / n:12.2f} {clips_flag / n:9.2f} {lawful_flag / n:7.2f}  "
              + ", ".join(f"{k} {c / n:.2f}" for k, c in other.most_common()))
        if misses:
            print(f"{'':16s}missed (seed, template, frame, agent): {misses}")
        table[kind] = dict(rule=rule, clips=n, acts=n_acts, recall=hit / n_acts if n_acts else None,
                           clips_with_act=clips_act / n, clip_rate=clips_flag / n, lawful_rate=lawful_flag / n,
                           other={k: c / n for k, c in other.items()}, misses=misses)
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "violation_check.json").write_text(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
