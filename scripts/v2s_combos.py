#!/usr/bin/env python
"""How each target combination comes out: per row of COMBOS, the share of seeds whose clip
was built, matched its B, C, D and E targets, and (pending rows) showed both outcomes of the
pending choice among its futures; plus rule violations.

    python scripts/v2s_combos.py --seeds 4 --forks 8 [--only pending]
"""
import argparse
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from f_toy.datasets.drivetoy.choices import CHOICE_KINDS
from f_toy.datasets.drivetoy.config import DRIVETOY_DEFAULT_CFG
from f_toy.datasets.drivetoy.rules import check
from f_toy.datasets.drivetoy.scenarios import COMBOS, simulate

CFG = dict(DRIVETOY_DEFAULT_CFG)


def pending_outcomes(rec, futs, T0=31, T1=40):
    """Outcomes, across the futures, of the first choice concerning the ego in frames T0+1 to T1."""
    d = rec["decisions"]
    rows = d[(d[:, 0] > T0) & (d[:, 0] <= T1) & (d[:, 6] > 0) & (d[:, 4] >= 0.1) & (d[:, 4] <= 0.9)]
    if not len(rows):
        return None, None
    slot, kind = int(rows[0, 1]), int(rows[0, 2])
    outs = []
    for f in futs:
        fd = f["decisions"]
        m = fd[(fd[:, 1] == slot) & (fd[:, 2] == kind)]
        if len(m):
            outs.append(int(m[0, 5]))
    return CHOICE_KINDS[kind], outs


def one(job):
    r, seed = job
    A, B, tmpl, D, C, E, V = COMBOS[r]
    tg = dict(A=A, B=B, C=C, D=D, E=E, V=V, template=tmpl)
    res = simulate(70_000_000 + 1000 * r + seed, CFG, target=tg)
    if not res["ok"]:
        return r, None
    lab, rec = res["labels"][0], res["branches"][0]
    out = dict(B=lab["B"] == B, C=lab["C"] == C, D=lab["D"] == D, E=lab["E"] == E, got=(lab["B"], lab["C"], lab["D"], lab["E"]),
               viol=len(check(rec, res["map"], CFG)), tries=res["tries"])
    if E == "pending":
        out["kind"], out["outs"] = pending_outcomes(rec, res["forks"][0])
    return r, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--forks", type=int, default=8)
    ap.add_argument("--only", default=None, help="pending: only the pending rows")
    ap.add_argument("--workers", type=int, default=120)
    args = ap.parse_args()
    CFG["DT_FORKS"] = args.forks
    rows = [r for r, c in enumerate(COMBOS) if args.only != "pending" or c[5] == "pending"]
    jobs = [(r, s) for r in rows for s in range(args.seeds)]
    with Pool(args.workers) as p:
        res = p.map(one, jobs, chunksize=1)
    by = defaultdict(list)
    for r, o in res:
        by[r].append(o)
    tot = defaultdict(float)
    for r in rows:
        os_ = [o for o in by[r] if o is not None]
        c = COMBOS[r]
        line = f"{r:2d} {c[0][:12]:12s} {c[1][:14]:14s} {c[2][:6]:6s} {c[3][:11]:11s} {c[4][:10]:10s} {c[5][:7]:7s} {str(c[6])[:8]:8s}"
        if not os_:
            print(line, " none built")
            continue
        m = {k: np.mean([o[k] for o in os_]) for k in "BCDE"}
        both = ""
        if c[5] == "pending":
            outs = [o.get("outs") for o in os_]
            both = " both:" + "".join("y" if o and len(set(o)) > 1 else ("-" if o is None else "n") for o in outs)
            both += " " + ",".join(sorted({str(o.get("kind")) for o in os_}))
        print(line, f" built {len(os_)}/{len(by[r])}  B {m['B']:.2f} C {m['C']:.2f} D {m['D']:.2f} E {m['E']:.2f}"
              f"  tries {np.mean([o['tries'] for o in os_]):.1f} viol {sum(o['viol'] for o in os_)}{both}")
        miss = [o["got"] for o in os_ if not (o["B"] and o["C"] and o["D"] and o["E"])]
        if miss:
            print("      got:", miss[:3])


if __name__ == "__main__":
    main()
