#!/usr/bin/env python
"""Simulate every buildable target combination of DriveToy v2 a few times (no rendering) and
report, per combination, how often each aspect comes out as targeted, the attempts used and
the time taken. Also runs rules.check on every clip.

    python scripts/v2_sim_probe.py --n 3 --workers 100 [--only signal4]
"""
import argparse
import json
import sys
import time
import traceback
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f_toy.datasets.drivetoy.config import DRIVETOY_DEFAULT_CFG
from f_toy.datasets.drivetoy.rules import check
from f_toy.datasets.drivetoy.scenarios import COMBOS, simulate

CFG = dict(DRIVETOY_DEFAULT_CFG)


def run(task):
    k, seed = task
    A, B, tmpl, D, C = COMBOS[k]
    tg = dict(A=A, B=B, C=C, D=D, template=tmpl)
    t0 = time.time()
    try:
        r = simulate(seed, CFG, target=tg)
    except Exception:
        return k, dict(err=traceback.format_exc())
    out = dict(ok=r["ok"], tries=r["tries"], sec=time.time() - t0)
    if r["ok"]:
        lab = r["labels"][0]
        out.update({x: lab[x] for x in "ABCD"})
        out["viol"] = [str(v) for v in check(dict(r["branches"][0], extra=r["branches"][0]["extra"]), r["map"], CFG)][:3]
    return k, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--workers", type=int, default=100)
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="analysis/v2_sim_probe.json")
    args = ap.parse_args()
    ks = [k for k, c in enumerate(COMBOS) if not args.only or args.only in c]
    tasks = [(k, 1000 * k + j) for k in ks for j in range(args.n)]
    res = defaultdict(list)
    t0 = time.time()
    with Pool(args.workers) as p:
        for k, o in p.imap_unordered(run, tasks):
            res[k].append(o)
    print(f"{len(tasks)} clips in {time.time() - t0:.0f}s")
    for k in ks:
        A, B, tmpl, D, C = COMBOS[k]
        rs = res[k]
        errs = [r for r in rs if "err" in r]
        if errs:
            print(f"ERR {COMBOS[k]}\n{errs[0]['err']}")
            continue
        ok = [r for r in rs if r["ok"]]
        hit = {x: sum(r[x] == t for r in ok) for x, t in zip("BCD", (B, C, D))}
        tries = sum(r["tries"] for r in ok) / max(len(ok), 1)
        sec = sum(r["sec"] for r in rs) / len(rs)
        got = "; ".join(f"{r['B']}/{r['C']}/{r['D']}" for r in ok if (r["B"], r["C"], r["D"]) != (B, C, D))
        viol = [v for r in ok for v in r["viol"]]
        print(f"{A:15s} {B:16s} {tmpl:11s} {D:12s} {C:11s} ok {len(ok)}/{len(rs)} B{hit['B']} C{hit['C']} D{hit['D']} "
              f"tries {tries:4.1f} {sec:5.1f}s  {got}" + (f"  VIOL {viol[:2]}" if viol else ""))
    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps({str(COMBOS[k]): v for k, v in res.items()}, indent=1, default=str))


if __name__ == "__main__":
    main()
