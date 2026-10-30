#!/usr/bin/env python
"""Checks of a v3 build (docs/version-3-revision.md): the base checks (scripts/base_checks.py), then one
entry per issue under "v3" in <split>/checks.json.

1. leads: vehicles whose desired speed is outside their kind's range;
2. go choices: ego go choices never undone, crossing within 4 s, stop-sign ties, standing at a clear stop sign;
3. oncoming gaps at the ego's conflict, and how many yielding left-turners finish their turn;
4. share of signal4 approach traffic in the left-only lane;
5. trees left out of the sightlines (render logs);
6. samples whose route is drawn per future: futures with several moves, ego indicator before frame 32;
7. map and may_go labels: files, round trip (with --resim), logged egos breaking "didn't go when the
   way was clear" as task_encoder_eval.md defines it.

    python scripts/v3_checks.py /root/projects/drive_v3/pilot [--resim]
"""
import json
import re
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import base_checks
from f_toy.datasets.drivetoy.choices import CHOICE_KINDS
from f_toy.datasets.drivetoy.diversity import ASPECTS, clip_labels
from f_toy.datasets.drivetoy.map_io import load_map
from f_toy.datasets.drivetoy.rules import check
from f_toy.datasets.drivetoy.scenarios import VARIANTS, simulate
from f_toy.datasets.drivetoy.sim import EV, RS

ROOT = ANN = META = None
FPS, T0 = 5, 31


def _map(n):
    return load_map(ROOT / "maps" / f"{n:06d}.json.gz")


def _enters(ev):
    return {int(e[2]): int(e[0]) for e in ev[::-1] if e[1] == EV["ENTER"]}


def _sample(n):
    """Every per-sample measurement, from the stored labels and the map."""
    a = {k: ANN[k][n] for k in ("agents", "lane", "s_local", "rule_state", "yield_to", "events", "decisions",
                                 "drivers", "lamps", "ego", "may_go", "template", "variant", "diversity")}
    m = _map(n)
    tmpl = META["templates"][int(a["template"])]
    ag, lane, ev, st = a["agents"], a["lane"], a["events"], a["rule_state"]
    out = dict(tmpl=tmpl)
    # 1. leads
    typ = ag[:, :, 0].max(0)
    overtake = int(a["variant"]) == VARIANTS.index("overtake") or (tmpl == "multilane" and ASPECTS["D"][a["diversity"][3]] == "lane change")
    f = a["drivers"][:, 1]
    out["below_60"] = int(sum(1 <= typ[s] <= 4 and f[s] < 0.6 - 1e-6 for s in range(1, len(typ))))
    out["cars_out"] = 0 if overtake else int(sum(typ[s] in (1, 2) and not 0.75 - 1e-6 <= f[s] <= 0.97 + 1e-6 for s in range(1, len(typ))))
    out["heavy_out"] = 0 if overtake else int(sum(typ[s] in (3, 4) and not 0.65 - 1e-6 <= f[s] <= 0.85 + 1e-6 for s in range(1, len(typ))))
    # 2. go choices of the ego (its own record)
    enters, n_f = _enters(ev), ag.shape[0]
    cancel = [int(e[0]) for e in ev if e[1] == EV["CANCEL"] and e[2] == 0]
    out["go"], out["wave"] = [], []
    for fr, slot, kind, _, _, o, _ in a["decisions"]:
        fr, slot, kind = int(fr), int(slot), int(kind)
        if o != 1 or fr < 0:
            continue
        if slot == 0 and CHOICE_KINDS[kind] in ("left_gap", "rtor"):
            end = min([enters.get(0, n_f)] + [c for c in cancel if c >= fr])
            undo = RS["YIELD_ONCOMING"] if CHOICE_KINDS[kind] == "left_gap" else RS["STOP_SIGNAL"]
            kept = not (st[fr + 1:end, 0] == undo).any()
            within = None if fr + 20 >= n_f else (enters.get(0, n_f) - fr <= 20 or any(c >= fr for c in cancel))
            out["go"].append((CHOICE_KINDS[kind], kept, within))
        elif CHOICE_KINDS[kind] == "stop_wave":
            nxt = sorted((f_, s) for s, f_ in enters.items() if f_ > fr)
            out["wave"].append(not nxt or nxt[0][1] == slot)
    v = a["ego"][:, 3]
    stand = []
    if tmpl == "stop4":
        still = np.where((v < 0.3) & (st[:, 0] == RS["STOP_SIGN"]))[0]
        if len(still):
            f0 = int(still[0])
            mv = np.where(v[f0:] >= 0.3)[0]
            if len(mv) and not (a["yield_to"][f0:f0 + int(mv[0]), 0] >= 0).any():
                stand.append(int(mv[0]))
    out["stand"] = stand
    # 3. oncoming gaps (signal4) and yielding left-turners
    out["gaps"], out["yielders"] = [], []
    if tmpl == "signal4":
        leg = next((m.lanes[int(l)].leg for l in lane[:, 0] if l >= 0 and m.lanes[int(l)].kind == "in"), None)
        opp = {l.id for l in m.lanes if l.kind == "in" and l.leg == (None if leg is None else (leg + 2) % 4) and l.lane_index == 1}
        frames = sorted(f_ for s, f_ in enters.items() if f_ >= 1 and int(lane[f_ - 1, s]) in opp)
        out["gaps"] = np.diff(frames).tolist()
        for s in range(ag.shape[1]):
            y = np.where(st[:, s] == RS["YIELD_ONCOMING"])[0]
            if len(y):
                after = [int(l) for l in lane[y[0]:, s] if l >= 0]
                out["yielders"].append(any(m.lanes[l].kind == "out" for l in after))
        # 4. left-only lane share
        first = {}
        for s in range(1, ag.shape[1]):
            ok = np.where((ag[:, s, 9] > 0) & (lane[:, s] >= 0))[0]
            if len(ok):
                ln = m.lanes[int(lane[ok[0], s])]
                if ln.kind == "in":
                    first[s] = ln.lane_index
        out["inner"] = [sum(x == 0 for x in first.values()), len(first)]
    yo = np.where(st[:, 0] == RS["YIELD_ONCOMING"])[0]
    out["ego_left"] = None if not len(yo) else dict(
        done=0 in enters, waits_all_futures=bool((FS["rule_state"][n][:, -1] == RS["YIELD_ONCOMING"]).all()))
    # 6. routes drawn per future
    out["route_drawn"] = bool(ANN["route_drawn"][n])
    if out["route_drawn"]:
        out["future_moves"] = ["-LSR"[int(x)] for x in FS["ego_turn"][n]]
        out["moves"] = sorted(set(out["future_moves"]))
        out["ind_before"] = int(a["lamps"][:T0 + 1, 0, 1:3].sum())
    # 7. didn't go when the way was clear (task_encoder_eval.md), logged ego
    out["didnt_go"] = _didnt_go(a, m)
    return out


def _stand_start(v, f):
    g = f
    while g > 0 and v[g - 1] < 0.3:
        g -= 1
    return g


def _didnt_go(a, m):
    """Frames at which the logged ego has stood still at or before its line for 1 s while may_go
    was 1 and no stopped vehicle stood within 10 m ahead in its lane; at a stop sign the second
    starts after the 0.8 s full stop, and frames are skipped while a car that stopped before it
    still waits at another approach."""
    ag, lane, sl, mg = a["agents"], a["lane"], a["s_local"], a["may_go"]
    v = ag[:, :, 4]
    run, hits = 0, []
    for f in range(ag.shape[0]):
        lid = int(lane[f, 0])
        if mg[f] != 1 or v[f, 0] >= 0.3 or lid < 0:
            run = 0
            continue
        ahead = [s for s in range(1, ag.shape[1]) if ag[f, s, 9] > 0 and int(lane[f, s]) == lid and v[f, s] < 0.3
                 and 0 < sl[f, s] - sl[f, 0] <= 10.0 + (ag[f, s, 6] + ag[f, 0, 6]) / 2]
        if ahead:
            run = 0
            continue
        ln = m.lanes[lid]
        if ln.kind != "in" or ln.stop is None or sl[f, 0] + ag[f, 0, 6] / 2 > ln.stop["s"] + 0.5:
            run = 0   # not at or before its stop line
            continue
        stop_sign = ln.stop["control"] == "stop"
        if stop_sign:
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
                run = 0
                continue
        run += 1
        if run >= 5 + (4 if stop_sign else 0):
            hits.append(f)
    return hits


def _roundtrip(n):
    lab = np.load(ROOT / "labels" / f"{n:06d}.npz")
    cfg = META["cfg"]
    tgt = {k: (ASPECTS[k][c] if c >= 0 else None) for k, c in zip("ABCDE", lab["target"])}
    tgt["template"] = META["templates"][int(lab["template"])]
    tgt["V"] = VARIANTS[int(lab["variant"])] if int(lab["variant"]) >= 0 else None
    r = simulate(int(lab["sim_seed"]), cfg, target=tgt)
    rec, m2 = r["branches"][int(lab["branch_id"])], _map(n)
    return check(rec, m2, cfg) == check(rec, r["map"], cfg) and clip_labels(rec, m2, cfg) == clip_labels(rec, r["map"], cfg)


def _init(root):
    global ROOT, ANN, META, FS
    ROOT = Path(root)
    META = json.loads((ROOT / "meta.json").read_text())
    ANN = dict(np.load(ROOT / "annotations.npz"))
    FS = dict(np.load(ROOT / "future_set.npz"))


def main():
    resim = "--resim" in sys.argv
    sys.argv = [sys.argv[0]] + [a for a in sys.argv[1:]]
    base_checks.main()                                  # writes checks.json with the base checks
    root = Path([a for a in sys.argv[1:] if not a.startswith("--")][0])
    _init(root)
    N = len(ANN["diversity"])
    with Pool(64, initializer=_init, initargs=(root,)) as p:
        S = p.map(_sample, range(N), chunksize=4)
    go = [g for s in S for g in s["go"]]
    gaps = [g for s in S for g in s["gaps"]]
    yielders = [y for s in S for y in s["yielders"]]
    inner = np.sum([s["inner"] for s in S if "inner" in s], 0) if any("inner" in s for s in S) else [0, 0]
    stand = [x for s in S for x in s["stand"]]
    drawn = [s for s in S if s["route_drawn"]]
    trees = [int(x) for f in (ROOT / "render").glob("log_*.txt") for x in re.findall(r"trees skipped: (\d+)", f.read_text())]
    dg = [s["didnt_go"] for s in S]
    v3 = {
        "1_leads": dict(below_60=sum(s["below_60"] for s in S), cars_out_of_range=sum(s["cars_out"] for s in S),
                        heavy_out_of_range=sum(s["heavy_out"] for s in S)),
        "2_go": dict(by_kind={k: dict(n=sum(g[0] == k for g in go), kept=sum(g[0] == k and g[1] for g in go),
                                      crossing_within_4s=f"{sum(g[0] == k and g[2] is True for g in go)}/{sum(g[0] == k and g[2] is not None for g in go)}")
                              for k in ("left_gap", "rtor")},
                     stop_sign_ties_in_order=f"{sum(s for x in S for s in x['wave'])}/{sum(len(x['wave']) for x in S)}",
                     clear_stop_sign_standing_frames=dict(n=len(stand), mean=round(float(np.mean(stand)), 2) if stand else None,
                                                          max=max(stand) if stand else None)),
        "3_gaps": dict(n=len(gaps), min_s=min(gaps) / FPS if gaps else None,
                       under_3s=round(float(np.mean(np.array(gaps) < 15)), 3) if gaps else None,
                       tail_mean_s=round(float(np.mean(np.array(gaps) / FPS - 2.0)), 2) if gaps else None,
                       yielding_left_turners_done=f"{sum(yielders)}/{len(yielders)}",
                       ego_yielding_done=f"{sum(s['ego_left']['done'] for s in S if s['ego_left'])}/{sum(bool(s['ego_left']) for s in S)}",
                       ego_waits_in_all_futures=sum(s["ego_left"]["waits_all_futures"] for s in S if s["ego_left"])),
        "4_left_lane_share": round(float(inner[0] / max(inner[1], 1)), 3),
        "5_trees_skipped": dict(clips=len(trees), total=sum(trees), mean=round(float(np.mean(trees)), 2) if trees else None),
        "6_routes": dict(drawn=len(drawn), with_2plus_moves=sum(len(s["moves"]) >= 2 for s in drawn),
                         future_move_shares={k: round(v / max(1, 16 * len(drawn)), 3)
                                             for k, v in Counter(mv for s in drawn for mv in s["future_moves"]).items()},
                         indicator_before_32=sum(s["ind_before"] > 0 for s in drawn)),
        "7_labels": dict(maps=len(list((ROOT / "maps").glob("*.json.gz"))), samples=N,
                         may_go_shape=list(ANN["may_go"].shape), future_may_go_shape=list(FS["may_go"].shape),
                         didnt_go_31_51=sum(any(31 <= f <= 51 for f in h) for h in dg),
                         didnt_go_any=sum(bool(h) for h in dg),
                         didnt_go_examples=[(n, h[:4]) for n, h in enumerate(dg) if h][:10]),
    }
    if resim:
        with Pool(20, initializer=_init, initargs=(root,)) as p:
            v3["7_labels"]["map_round_trip"] = f"{sum(p.map(_roundtrip, range(min(20, N))))}/{min(20, N)}"
    out = json.loads((root / "checks.json").read_text())
    out["v3"] = v3
    (root / "checks.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(v3, indent=1, default=str))


if __name__ == "__main__":
    main()
