"""Issue 7: the map label. A sample's road map survives JSON and gives the same rule checks and
labels as the simulator's own map. python3 tests/test_v3_map.py"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from f_toy.datasets.drivetoy.config import DRIVETOY_DEFAULT_CFG as CFG
from f_toy.datasets.drivetoy.diversity import clip_labels, cue_problems
from f_toy.datasets.drivetoy.map_io import load_map, map_from_dict, map_to_dict, save_map
from f_toy.datasets.drivetoy.rules import check
from f_toy.datasets.drivetoy.scenarios import COMBOS, simulate

V3 = dict(CFG, DT_FORKS=1)


def test_map_round_trip():
    for seed, tmpl in enumerate(("signal4", "stop4", "multilane", "residential")):
        row = next(r for r in COMBOS if r[2] == tmpl)
        A, B, _, D, C, E, V = row
        res = simulate(6000 + seed, V3, target=dict(A=A, B=B, C=C, D=D, E=E, V=V, template=tmpl))
        assert res["ok"], tmpl
        m, rec = res["map"], res["branches"][0]
        m2 = map_from_dict(json.loads(json.dumps(map_to_dict(m))))
        assert len(m2.lanes) == len(m.lanes)
        for a, b in zip(m.lanes, m2.lanes):
            for k in ("pts", "s", "head", "curv"):
                assert np.allclose(getattr(a, k), getattr(b, k), rtol=0, atol=1e-9), (tmpl, a.id, k)
            assert (a.next, a.stop, a.turn, a.zones, a.inflow_w) == (b.next, b.stop, b.turn, b.zones, b.inflow_w), (tmpl, a.id)
        assert m2.conflicts == m.conflicts
        assert check(rec, m2, V3) == check(rec, m, V3)
        assert clip_labels(rec, m2, V3) == clip_labels(rec, m, V3)
        assert cue_problems(rec, m2, V3) == cue_problems(rec, m, V3)
        print(tmpl, len(m.lanes), "lanes,", len(json.dumps(map_to_dict(m))) // 1024, "KB")


def test_round_trip_with_violations():
    """rules.check gives the same violations on the loaded map, on a record that has some."""
    cfg = dict(V3, DT_VIOLATION="red")
    for seed in range(6200, 6230):
        row = next(r for r in COMBOS if r[2] == "signal4" and r[0] == "steady red")
        A, B, tmpl, D, C, E, V = row
        res = simulate(seed, cfg, target=dict(A=A, B=B, C=C, D=D, E=E, V=V, template=tmpl))
        if not res["ok"]:
            continue
        m, rec = res["map"], res["branches"][0]
        v = check(rec, m, cfg)
        if v:
            assert check(rec, map_from_dict(json.loads(json.dumps(map_to_dict(m)))), cfg) == v
            print("violations compared:", len(v))
            return
    raise AssertionError("no record with violations found")


def test_save_load_gz(tmp="/tmp/test_v3_map.json.gz"):
    row = next(r for r in COMBOS if r[2] == "signal4")
    A, B, tmpl, D, C, E, V = row
    res = simulate(6100, V3, target=dict(A=A, B=B, C=C, D=D, E=E, V=V, template=tmpl))
    m, rec = res["map"], res["branches"][0]
    save_map(tmp, m)
    m2 = load_map(tmp)
    Path(tmp).unlink()
    assert check(rec, m2, V3) == check(rec, m, V3) and clip_labels(rec, m2, V3) == clip_labels(rec, m, V3)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
