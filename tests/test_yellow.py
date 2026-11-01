"""H15, H26, H36: the ego's response to a yellow is determined, the same in every future, and a
pending yellow turns at frames 27 to 31 from positions that make some clips stop and some go.
python3 tests/test_yellow.py"""
import numpy as np

from helpers import ego_rows, main, onset, rows, run

PEND = rows(A="green to yellow", E="pending", V="yellow")
DEC = [r for r in rows(A="green to yellow") if r[5] != "pending"]


def test_same_response_in_every_future():
    for s in run(PEND, range(16), forks=8):
        outs = {tuple(ego_rows(r, "yellow")[:1, 5]) for r in s["res"]["fork_recs"][0]}
        assert len(outs) == 1, (s["seed"], outs)


def test_rule():   # determined: it goes only when stopping would need more than 3 m/s^2
    for s in run(PEND + DEC, range(8)):
        for r in s["res"]["fork_recs"][0]:
            for row in ego_rows(r, "yellow"):
                assert row[4] in (0.0, 1.0) and not (row[5] == 1 and row[3] <= 3.0), (s["seed"], row)


def test_pending_onsets_spread():
    on, go = [], []
    for s in run(PEND, range(40)):
        rec = s["res"]["branches"][0]
        on.append(onset(rec, s["res"]["map"]))
        r = ego_rows(rec, "yellow")
        go.append(float(r[0, 5]) if len(r) else 0.0)
    assert all(f is not None and 27 <= f <= 31 for f in on), on
    assert 0.25 <= np.mean(go) <= 0.75, np.mean(go)


def test_decided_onsets_early():
    for s in run(DEC, range(4)):
        f = onset(s["res"]["branches"][0], s["res"]["map"])
        assert f is None or f <= 26, (s["row"], s["seed"], f)


if __name__ == "__main__":
    main(globals())
