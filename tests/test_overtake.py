"""H28, H38: the ego signals once its patience behind a slow leader is over and chooses when the
move can start: pulling out moves over at once, staying behind turns the indicator off; a pending
overtake is chosen in frames 32 to 40.
python3 tests/test_overtake.py"""
import numpy as np

from helpers import ego_rows, events, main, rows, run

OVT = rows(V="overtake")


def _lc_start(r, f0, f1):
    return any(f0 <= e[0] <= f1 for e in events(r, "LC_START", 0))


def _indicator_off(r, f0, f1):
    return any(f0 <= e[0] <= f1 for e in events(r, "INDICATOR_OFF", 0))


def _lateral_shift(r, f0, f1):
    e = r["ego"]
    h = e[max(f0 - 2, 0), 2]   # the heading before the move starts
    n = np.array([-np.sin(h), np.cos(h)])
    return float(np.abs((e[f0:f1 + 1, :2] - e[f0, :2]) @ n).max())


def _signals_without_move(r):
    """The ego signalling for more than 5 s in a row before any lane change starts (the longest
    indicator lead is 4 s)."""
    on = (r["extra"][:, 0, 2] > 0) & (r["lc"][:, 0] == 0)
    run_ = 0
    for x in on:
        run_ = run_ + 1 if x else 0
        if run_ > 25:
            return True
    return False


def test_overtake_outcomes_happen():
    for s in run(OVT, range(24), forks=8):
        for k, r in enumerate(s["res"]["fork_recs"][0]):
            n = r["agents"].shape[0]
            for row in ego_rows(r, "overtake"):
                f = int(row[0])
                if row[5] == 1:
                    assert _lc_start(r, f - 1, f + 2), (s["row"][5], s["seed"], k, f, "no move")
                    if f + 20 < n:
                        assert _lateral_shift(r, f, f + 20) >= 2.5, (s["row"][5], s["seed"], k, f, "no shift")
                else:
                    assert _indicator_off(r, f - 1, f + 2), (s["row"][5], s["seed"], k, f, "indicator stays on")
            assert not _signals_without_move(r), (s["row"][5], s["seed"], k)


def test_pending_overtake_close_behind():   # at frame 31 it follows within 1.5 s of the slow leader
    gaps = []
    for s in run(rows(V="overtake", E="pending"), range(20)):
        r = s["res"]["branches"][0]
        lead = int(r["extra"][31, 0, 4])
        if lead > 0 and int(r["extra"][31, 0, 3]) == 1:   # following
            d = float(np.hypot(*(r["agents"][31, lead, 1:3] - r["ego"][31, :2]))) - (r["agents"][31, lead, 6] + 4.5) / 2
            gaps.append(d / max(r["ego"][31, 3], 1.0))
    print(f"  time gap to the leader at frame 31: median {np.median(gaps):.2f} s, max {max(gaps):.2f} s", flush=True)
    assert gaps and np.median(gaps) <= 1.5


def test_overtake_pending_rate():
    hit = []
    for s in run(rows(V="overtake", E="pending"), range(30)):
        g = ego_rows(s["res"]["branches"][0], "overtake")
        hit.append(bool(len(g)) and 32 <= g[0, 0] <= 40)
    print(f"  pending overtakes chosen in frames 32 to 40: {np.mean(hit):.2f} of {len(hit)}", flush=True)
    assert np.mean(hit) >= 0.5


if __name__ == "__main__":
    main(globals())
