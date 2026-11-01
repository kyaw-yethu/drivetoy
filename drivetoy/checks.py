"""Checks shared by the build and the split checks (docs/constraints.md)."""
import numpy as np

from .choices import CHOICE_KINDS


def _stand_start(v, f):
    g = f
    while g > 0 and v[g - 1] < 0.3:
        g -= 1
    return g


def didnt_go(rec, m, cfg=None):
    """H22: frames at which the ego has stood still at or before its stop line for 1 s while may_go
    was 1 and no stopped vehicle stood within 10 m ahead in its lane (task_encoder_eval.md, "didn't
    go when the way was clear"). At a stop sign the time counts after the 0.8 s full stop, frames are
    skipped while a car that stopped before it still waits at another approach, and letting the
    others go (its stop_wave choice 0 after frame 31, H39) counts as waiting for them. ``rec`` is a
    packed record (``extra``) or a sample's labels (``lane``, ``s_local``)."""
    ag, mg = rec["agents"], rec["may_go"]
    lane = (rec["extra"][..., 0] if "extra" in rec else rec["lane"]).astype(int)
    sl = rec["extra"][..., 1] if "extra" in rec else rec["s_local"]
    v = ag[:, :, 4]
    dec = rec["decisions"]
    lets_go = dec[(dec[:, 0] > 31) & (dec[:, 1] == 0) & (dec[:, 2] == CHOICE_KINDS.index("stop_wave")) & (dec[:, 5] == 0)]
    let_from = int(lets_go[0, 0]) if len(lets_go) else None
    run_, hits = 0, []
    for f in range(ag.shape[0]):
        lid = int(lane[f, 0])
        if mg[f] != 1 or v[f, 0] >= 0.3 or lid < 0:
            run_ = 0
            continue
        ahead = [s for s in range(1, ag.shape[1]) if ag[f, s, 9] > 0 and int(lane[f, s]) == lid and v[f, s] < 0.3
                 and 0 < sl[f, s] - sl[f, 0] <= 10.0 + (ag[f, s, 6] + ag[f, 0, 6]) / 2]
        if ahead:
            run_ = 0
            continue
        ln = m.lanes[lid]
        if ln.kind != "in" or ln.stop is None or sl[f, 0] + ag[f, 0, 6] / 2 > ln.stop["s"] + 0.5:
            run_ = 0   # not at or before its stop line
            continue
        stop_sign = ln.stop["control"] == "stop"
        if stop_sign:
            if let_from is not None and f >= let_from:
                run_ = 0   # it lets the others go
                continue
            mine = _stand_start(v[:, 0], f)
            first = False
            for s in range(1, ag.shape[1]):
                ls = int(lane[f, s])
                if ag[f, s, 9] <= 0 or ls < 0 or v[f, s] >= 0.3:
                    continue
                o = m.lanes[ls]
                if o.kind == "in" and o.stop is not None and o.leg != ln.leg and \
                        o.stop["s"] - (sl[f, s] + ag[f, s, 6] / 2) <= 6.0 and _stand_start(v[:, s], f) < mine:
                    first = True
                    break
            if first:
                run_ = 0
                continue
        run_ += 1
        if run_ >= 5 + (4 if stop_sign else 0):
            hits.append(f)
    return hits
