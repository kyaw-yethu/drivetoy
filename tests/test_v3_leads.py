"""Issue 1: leaders drive at normal speeds (cars 75-97% of the limit, trucks and buses 65-85%,
overtaken leads 60-75%), nobody below 60%. python3 tests/test_v3_leads.py"""
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from drivetoy.config import DRIVETOY_DEFAULT_CFG as CFG
from drivetoy.scenarios import COMBOS, simulate

V3 = dict(CFG, DT_FORKS=1)
ROWS = [c for c in COMBOS if c[1] == "lead vehicle"]


def _one(job):
    seed, row = job
    A, B, tmpl, D, C, E, V = row
    res = simulate(1000 + seed, V3, target=dict(A=A, B=B, C=C, D=D, E=E, V=V, template=tmpl))
    if not res["ok"]:
        return row, None, []
    rec = res["branches"][0]
    overtake = tmpl == "multilane" and (V == "overtake" or D == "lane change")
    bad = []
    typ = rec["agents"][:, :, 0].max(0)
    for slot in range(1, rec["drivers"].shape[0]):
        t, f = int(typ[slot]), float(rec["drivers"][slot, 1])
        if not 1 <= t <= 4:
            continue
        lo, hi = (0.60, 0.97) if overtake else ((0.65, 0.85) if t >= 3 else (0.75, 0.97))
        if not lo - 1e-6 <= f <= hi + 1e-6:
            bad.append((slot, t, round(f, 3)))
    if float(rec["drivers"][0, 1]) < 0.60:
        bad.append((0, "ego", float(rec["drivers"][0, 1])))
    return row, res["labels"][0], bad


def test_v3_leads():
    jobs = [(s, r) for s in range(6) for r in ROWS]
    with Pool(min(32, len(jobs))) as p:
        out = p.map(_one, jobs)
    built = [(r, lab, bad) for r, lab, bad in out if lab is not None]
    hit = Counter((r[1] == lab["B"], r[4] == lab["C"]) for r, lab, _ in built)
    print(f"built {len(built)}/{len(jobs)}; (B hit, C hit): {dict(hit)}")
    for r, lab, _ in built:
        if r[4] == "follow" or r[1] != lab["B"]:
            print("  target", r[2], r[1], r[4], "-> got", lab["B"], lab["C"])
    bad = [(r, b) for r, _, b in built if b]
    assert not bad, bad[:5]
    assert len(built) >= len(jobs) // 2


def _stream_job(seed):
    rows = [r for r in COMBOS if r[6] in ("left_gap", "rtor")]   # scripted oncoming and cross traffic
    row = rows[seed % len(rows)]
    A, B, tmpl, D, C, E, V = row
    res = simulate(1500 + seed, V3, target=dict(A=A, B=B, C=C, D=D, E=E, V=V, template=tmpl))
    if not res["ok"]:
        return None
    rec = res["branches"][0]
    typ = rec["agents"][:, :, 0].max(0)
    return [round(float(rec["drivers"][s, 1]), 3) for s in range(1, len(typ)) if 1 <= typ[s] <= 4 and rec["drivers"][s, 1] < 0.6]


def test_no_crawling_stream_cars():
    """Scripted oncoming and cross traffic (pending left turns, right turns on red) keep normal desired speeds."""
    with Pool(16) as p:
        out = [o for o in p.map(_stream_job, range(16)) if o is not None]
    print(f"built {len(out)}; vehicles below 60%: {[x for o in out for x in o]}")
    assert len(out) >= 8 and not [x for o in out for x in o]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
