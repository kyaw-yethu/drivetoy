"""Shared helpers for the constraint tests (docs/constraints.md): simulate combos in a pool and
read the records they give."""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from drivetoy.choices import CHOICE_KINDS
from drivetoy.config import DRIVETOY_DEFAULT_CFG as CFG
from drivetoy.diversity import ego_approach
from drivetoy.scenarios import COMBOS, simulate
from drivetoy.sim import BALL_G, BALL_Y, EV

FIELDS = ("A", "B", "template", "D", "C", "E", "V")


def target(row):
    A, B, tmpl, D, C, E, V = row
    return dict(A=A, B=B, C=C, D=D, E=E, V=V, template=tmpl)


def rows(**fields):
    """COMBOS rows whose fields (A, B, template, D, C, E, V) all match."""
    return [r for r in COMBOS if all(r[FIELDS.index(k)] == v for k, v in fields.items())]


def _job(job):
    row, seed, forks = job
    res = simulate(5000 + seed, dict(CFG, DT_FORKS=forks, DT_KEEP_FORK_RECS=True), target=target(row))
    return dict(row=row, seed=seed, res=res) if res["ok"] else None


def run(rows_, seeds, forks=4, workers=64):
    """dict(row, seed, res) for every (row, seed) that builds; res["fork_recs"][0] holds the full
    records of all futures, the sample's own first."""
    jobs = [(r, s, forks) for r in rows_ for s in seeds]
    if not jobs:
        return []
    with Pool(min(workers, len(jobs))) as p:
        return [x for x in p.map(_job, jobs, chunksize=1) if x is not None]


def ego_rows(rec, kind):
    d = rec["decisions"]
    return d[(d[:, 1] == 0) & (d[:, 2] == CHOICE_KINDS.index(kind))]


def moved(rec, f0, f1):
    e = rec["ego"][f0:f1 + 1, :2]
    return float(np.sum(np.hypot(*np.diff(e, axis=0).T))) if len(e) > 1 else 0.0


def onset(rec, m):
    """First frame at which the ego's light turns from green to yellow, or None."""
    gi = ego_approach(rec, m)
    if gi is None:
        return None
    sig = rec["signals"][:, gi, 0]
    for f in range(1, len(sig)):
        if sig[f - 1] == BALL_G and sig[f] == BALL_Y:
            return f
    return None


def events(rec, name, a=None):
    """Rows (frame, type, a, b, value) of one event type, optionally of agent a."""
    ev = rec["events"]
    ev = ev[ev[:, 1] == EV[name]]
    return ev if a is None else ev[ev[:, 2] == a]


def main(ns):
    """Run every test_* function of a test module, as the other tests do."""
    for name, fn in list(ns.items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(name, "ok", flush=True)
