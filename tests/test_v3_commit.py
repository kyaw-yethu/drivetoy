"""Issue 2 and D6: a drawn "go" (left gap, right turn on red, stop-sign tie) is carried out unless a
vehicle is already inside the conflict area, and drivers move off promptly at a clear stop sign.
python3 tests/test_v3_commit.py"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from f_toy.datasets.drivetoy.choices import CHOICE_KINDS
from f_toy.datasets.drivetoy.config import DRIVETOY_DEFAULT_CFG as CFG
from f_toy.datasets.drivetoy.rules import check
from f_toy.datasets.drivetoy.scenarios import COMBOS, simulate
from f_toy.datasets.drivetoy.sim import EV, RS

V3 = dict(CFG, DT_FORKS=4, DT_KEEP_FORK_RECS=True)
GO_KINDS = {CHOICE_KINDS.index("left_gap"), CHOICE_KINDS.index("rtor")}
WAVE = CHOICE_KINDS.index("stop_wave")


def _target(row):
    A, B, tmpl, D, C, E, V = row
    return dict(A=A, B=B, C=C, D=D, E=E, V=V, template=tmpl)


def _in_junction(rec, m, f, a):
    lid = int(rec["extra"][f, a, 0])
    return lid >= 0 and m.lanes[lid].kind == "conn"


def _enter(rec, m, a, f0):
    """First frame from f0 on in which vehicle a is on a junction connector, or None."""
    for f in range(f0, rec["agents"].shape[0]):
        if rec["agents"][f, a, 9] > 0 and _in_junction(rec, m, f, a):
            return f
    return None


UNDO = {CHOICE_KINDS.index("left_gap"): RS["YIELD_ONCOMING"], CHOICE_KINDS.index("rtor"): RS["STOP_SIGNAL"]}


def _commit_job(job):
    """Per clip: [kept, total] ego go choices never undone before the ego enters or cancels,
    [crossed, total] go choices with 4 s of record left that cross within 4 s or cancel,
    [first, total] stop-sign go-in-turn choices whose chooser enters next, and collisions."""
    seed, row = job
    res = simulate(2000 + seed, V3, target=_target(row))
    if not res["ok"]:
        return None
    m, kept, go4, wave, coll = res["map"], [0, 0], [0, 0], [0, 0], []
    for rec in res["fork_recs"][0]:
        ev, st, n = rec["events"], rec["extra"][:, 0, 3], rec["agents"].shape[0]
        for fr, slot, kind, _, _, out, _ in rec["decisions"]:
            fr, slot, kind = int(fr), int(slot), int(kind)
            if out != 1 or fr < 0:
                continue
            if slot == 0 and kind in UNDO:
                enter = min([int(e[0]) for e in ev if e[1] == EV["ENTER"] and e[2] == 0 and e[0] >= fr] + [n])
                cancel = min([int(e[0]) for e in ev if e[1] == EV["CANCEL"] and e[2] == 0 and e[0] >= fr] + [n])
                kept[0] += 1
                kept[1] += not (st[fr + 1:min(enter, cancel)] == UNDO[kind]).any()
                if fr + 20 < n:
                    go4[0] += 1
                    go4[1] += enter - fr <= 20 or cancel < n
            elif kind == WAVE:
                nxt = [(int(e[0]), int(e[2])) for e in ev if e[1] == EV["ENTER"] and e[0] > fr]
                wave[0] += 1
                wave[1] += not nxt or min(nxt)[1] == slot   # nobody enters before the chooser
        coll += [v for v in check(rec, m, V3) if v[1] == "COLLISION"]
    return kept, go4, wave, coll


def test_go_commits():
    rows = [r for r in COMBOS if r[6] in ("left_gap", "rtor", "near_tie")]
    with Pool(32) as p:
        out = [o for o in p.map(_commit_job, [(s, r) for s in range(8) for r in rows]) if o is not None]
    kept, go4, wave = (np.sum([o[k] for o in out], 0) for k in range(3))
    coll = [c for o in out for c in o[3]]
    print(f"clips {len(out)}; ego go choices kept: {kept[1]}/{kept[0]}; crossing within 4 s (or cancelled): "
          f"{go4[1]}/{go4[0]}; stop-sign go-in-turn with nobody entering first: {wave[1]}/{wave[0]}; collisions: {len(coll)}")
    assert kept[0] >= 20 and kept[1] == kept[0]
    assert wave[0] == 0 or wave[1] >= 0.95 * wave[0]
    assert not coll, coll[:5]


def _stop_job(job):
    seed, cfg = job
    row = [r for r in COMBOS if r[2] == "stop4" and r[1] == "stop sign"][seed % 2]
    res = simulate(3000 + seed, dict(cfg, DT_FORKS=1), target=_target(row))
    if not res["ok"]:
        return None
    rec = res["branches"][0]
    v, st, yt = rec["ego"][:, 3], rec["extra"][:, 0, 3], rec["extra"][:, 0, 4]
    still = np.where((v < 0.3) & (st == RS["STOP_SIGN"]))[0]
    if not len(still):
        return None
    f0 = int(still[0])
    moving = np.where(v[f0:] >= 0.3)[0]
    if not len(moving) or (yt[f0:f0 + int(moving[0])] >= 0).any():
        return None   # waited for somebody, or never moved off
    return int(moving[0])


def test_prompt_stop_sign():
    slow = dict(CFG, R4_STOP_TIME=1.0, DT_START_DELAY=(0.3, 1.5))   # a 1 s stop and slower starts, for contrast
    with Pool(32) as p:
        prompt = [x for x in p.map(_stop_job, [(s, CFG) for s in range(40)]) if x is not None]
        late = [x for x in p.map(_stop_job, [(s, slow) for s in range(40)]) if x is not None]
    print(f"frames standing at a clear stop sign: max {max(prompt)} mean {np.mean(prompt):.1f} (n {len(prompt)}); "
          f"with slower timing max {max(late)} mean {np.mean(late):.1f} (n {len(late)})")
    assert len(prompt) >= 10 and max(prompt) <= 9 and max(late) > 9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
