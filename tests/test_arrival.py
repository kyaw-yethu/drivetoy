"""H35, H37, H41: a pending choice comes from the ego arriving at its decision point around frame
31, not from holding it at the front of its line; most attempts at a pending row give E pending; and
futures of different outcomes are at least 2 m apart at frame 63.
python3 tests/test_arrival.py"""
import numpy as np

from helpers import COMBOS, CHOICE_KINDS, main, run
from drivetoy.sim import RS

PENDING = [r for r in COMBOS if r[5] == "pending"]
T0 = 31


def _pending_row(rec):
    d = rec["decisions"]
    d = d[(d[:, 0] > T0) & (d[:, 0] <= 40) & (d[:, 1] == 0) & (d[:, 6] > 0)
          & (((d[:, 4] >= 0.1) & (d[:, 4] <= 0.9)) | (d[:, 2] == CHOICE_KINDS.index("yellow")))]
    return d[0] if len(d) else None


def _standing_before(rec, f1):
    """Frames before f1 the ego stands with nobody ahead holding it, less a 0.8 s full stop."""
    st = (rec["ego"][:f1, 3] < 0.3) & (rec["extra"][:f1, 0, 3] != RS["FOLLOW"])
    return max(int(st.sum()) - 4, 0)


def test_no_standing_before_choice():
    for s in run(PENDING, range(12)):
        rec = s["res"]["branches"][0]
        row = _pending_row(rec)
        if s["res"]["labels"][0]["E"] == "pending" and row is not None:
            assert _standing_before(rec, int(row[0])) <= 7, (s["row"], s["seed"], _standing_before(rec, int(row[0])))


def test_pending_rate():
    res = run(PENDING, range(16))
    for row in PENDING:
        e = [s["res"]["labels"][0]["E"] for s in res if s["row"] == row]
        print(f"  {row[0]:16s} {row[6]:10s} {row[4]:10s} pending {np.mean([x == 'pending' for x in e]):.2f} of {len(e)}", flush=True)
        assert e and np.mean([x == "pending" for x in e]) >= 0.5, row


def _outcome(r, kind):
    d = r["decisions"]
    d = d[(d[:, 0] > T0) & (d[:, 1] == 0) & (d[:, 2] == kind)]
    if not len(d):
        return None
    return (int(d[0, 5]), int(r["ego_turn"])) if CHOICE_KINDS[kind] == "stop_wave" and d[0, 5] == 1 else int(d[0, 5])


def test_distinct_outcomes():   # H41
    n = ok = 0
    for s in run(PENDING, range(12), forks=8):
        row = _pending_row(s["res"]["fork_recs"][0][0])
        if row is None:
            continue
        groups = {}
        for r in s["res"]["fork_recs"][0]:
            o = _outcome(r, int(row[2]))
            if o is not None:
                groups.setdefault(o, []).append(r["ego"][63, :2])
        if len(groups) > 1:
            n += 1
            keys = list(groups)
            ok += all(min(np.hypot(*(a - b)) for a in groups[x] for b in groups[y]) >= 2.0
                      for i, x in enumerate(keys) for y in keys[i + 1:])
    print(f"  samples whose outcomes are 2 m apart at frame 63: {ok}/{n}", flush=True)
    assert n and ok / n >= 0.9, (ok, n)


if __name__ == "__main__":
    main(globals())
