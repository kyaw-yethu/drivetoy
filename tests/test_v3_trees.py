"""Issue 5: no tree crown between an approach and a sign or signal head, 10 to 60 m before it.
python3 tests/test_v3_trees.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "f_toy" / "datasets" / "drivetoy" / "blender"))
import geom
from f_toy.datasets.drivetoy.config import DRIVETOY_DEFAULT_CFG as CFG, V3_OVERRIDES
from f_toy.datasets.drivetoy.export import sightlines
from f_toy.datasets.drivetoy.roads import build_stop4

V3 = dict(CFG, **V3_OVERRIDES)


def test_seg_dist():
    assert abs(geom.seg_dist((5, 3, 0), (0, 0, 0), (10, 0, 0)) - 3.0) < 1e-9
    assert abs(geom.seg_dist((-4, 3, 0), (0, 0, 0), (10, 0, 0)) - 5.0) < 1e-9


def _stop_lines():
    m = build_stop4(np.random.RandomState(0), V3)
    signs = [dict(kind=s["kind"], pos=list(map(float, s["pos"])), heading=float(s["heading"])) for s in m.signs]
    return m, signs, np.asarray(sightlines(m, signs, []), float).reshape(-1, 6)


def test_stop_sign_sightlines():
    m, signs, segs = _stop_lines()
    stops = [s for s in signs if s["kind"] == "stop"]
    assert len(stops) == 4
    for s in stops:
        mine = segs[np.all(np.isclose(segs[:, 3:5], s["pos"]), axis=1)]
        assert len(mine) == 11, (s["pos"], len(mine))
        assert np.allclose(mine[:, 2], 1.4) and np.allclose(mine[:, 5], 2.3)
        d = np.hypot(mine[:, 0] - mine[:, 3], mine[:, 1] - mine[:, 4])
        assert d.min() > 9.0 and d.max() < 61.0, (d.min(), d.max())


def test_crown_blocks():
    m, signs, segs = _stop_lines()
    s = next(x for x in signs if x["kind"] == "stop")
    mine = segs[np.all(np.isclose(segs[:, 3:5], s["pos"]), axis=1)]
    seg = mine[np.argmin(np.abs(np.hypot(mine[:, 0] - mine[:, 3], mine[:, 1] - mine[:, 4]) - 30.0))]
    mid = (seg[:3] + seg[3:]) / 2
    side = np.array([seg[4] - seg[1], seg[0] - seg[3]]) / np.hypot(seg[3] - seg[0], seg[4] - seg[1])  # unit normal
    assert geom.crown_blocks((mid[0], mid[1], 3.0), 1.5, mine)
    far = mid[:2] + side * 12.0   # 12 m to the side of the sightline
    assert not geom.crown_blocks((far[0], far[1], 3.0), 1.5, mine)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
