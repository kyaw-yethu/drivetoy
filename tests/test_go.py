"""H24: a drawn go (left turn, right turn on red) starts at once and is carried out: the ego
crosses its line within 3 s, covers 5 m within 4 s and is not cancelled.
python3 tests/test_go.py"""
from helpers import ego_rows, events, main, moved, rows, run


def _carried_out(r, f):
    n = r["agents"].shape[0]
    enter = [int(e[0]) for e in events(r, "ENTER", 0) if e[0] >= f]
    cancel = [int(e[0]) for e in events(r, "CANCEL", 0) if e[0] >= f]
    return bool(enter) and enter[0] <= f + 15 and moved(r, f, min(f + 20, n - 1)) >= 5.0 and not cancel


def test_go_is_carried_out():
    ok = tot = 0
    bad = []
    for s in run(rows(V="left_gap") + rows(V="rtor") + rows(A="permissive left"), range(16)):
        for r in s["res"]["fork_recs"][0]:
            for kind in ("left_gap", "rtor"):
                for row in ego_rows(r, kind):
                    f = int(row[0])
                    if row[5] == 1 and f + 20 < r["agents"].shape[0]:
                        tot += 1
                        good = _carried_out(r, f)
                        ok += good
                        if not good:
                            bad.append((s["row"][0], s["row"][6], s["seed"], kind, f))
    print(f"  go carried out {ok}/{tot}", flush=True)
    assert tot and ok / tot >= 0.98, (ok, tot, bad[:6])


if __name__ == "__main__":
    main(globals())
