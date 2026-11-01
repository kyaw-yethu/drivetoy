"""H22: the ego never stands at or before its line for 1 s (1.8 s at a stop sign) while it may go,
in the sample or in any future; letting the others go at a stop sign counts as waiting for them.
python3 tests/test_didnt_go.py"""
import copy

import numpy as np

from helpers import CFG, COMBOS, main, rows, run
from drivetoy.checks import didnt_go


def test_flags_a_standing_ego():
    s = run(rows(A="steady red", V="rtor"), [0], forks=2)[0]
    rec, m = copy.deepcopy(s["res"]["branches"][0]), s["res"]["map"]
    lid = int(rec["extra"][0, 0, 0])
    ln = m.lanes[lid]
    for f in range(10, 22):   # standing 1 m before its line, free to go, with nobody around
        rec["agents"][f, 0, 4] = 0.0
        rec["extra"][f, 0, 0], rec["extra"][f, 0, 1] = lid, ln.stop["s"] - 1.0 - rec["agents"][f, 0, 6] / 2
        rec["agents"][f, 1:, 9] = 0.0
        rec["may_go"][f] = 1
    hits = didnt_go(rec, m, CFG)
    assert [h for h in hits if 10 <= h < 22] == list(range(14, 22)), hits


def test_accepted_samples_pass():
    for s in run(COMBOS, range(1)):
        for k, r in enumerate(s["res"]["fork_recs"][0]):
            assert not didnt_go(r, s["res"]["map"], CFG), (s["row"], s["seed"], k, didnt_go(r, s["res"]["map"], CFG)[:3])


if __name__ == "__main__":
    main(globals())
