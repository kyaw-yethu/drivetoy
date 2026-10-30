"""Issue 8 (D5): red pavement on both lanes along the whole school zone, and a filled blue
triangle (2.0 m by 4 m, apex toward the approaching driver) on each lane where its zone begins.
python3 tests/test_v3_school.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from f_toy.datasets.drivetoy.config import DRIVETOY_DEFAULT_CFG as CFG
from f_toy.datasets.drivetoy.roads import LANE_W, build_residential

V3 = dict(CFG)


def test_school_zone():
    for seed in range(5):
        m = build_residential(np.random.RandomState(seed), V3)
        zone = (m.crosswalk_y - 60.0, m.crosswalk_y + 30.0)
        school = [np.asarray(s["poly"], float) for s in m.surfaces if s["kind"] == "school"]
        assert len(school) == 1
        p = school[0]
        assert np.isclose(p[:, 0].min(), -LANE_W) and np.isclose(p[:, 0].max(), LANE_W)
        assert np.isclose(p[:, 1].min(), zone[0]) and np.isclose(p[:, 1].max(), zone[1])
        tri = [np.asarray(mk["pts"], float) for mk in m.markings if mk["style"] == "school_triangle"]
        assert len(tri) == 2 and all(t.shape == (3, 2) for t in tri)
        nb = next(t for t in tri if t[:, 0].mean() > 0)
        sb = next(t for t in tri if t[:, 0].mean() < 0)
        for t, sgn, y_in in ((nb, 1.0, zone[0]), (sb, -1.0, zone[1])):
            assert np.isclose(np.ptp(t[:, 0]), 2.0) and np.isclose(np.ptp(t[:, 1]), 4.0)
            assert np.isclose(t[:, 0].mean(), sgn * LANE_W / 2)
            assert np.isclose(t[0, 1], y_in + sgn * 2.0)          # apex, nearest the approaching driver
            assert np.allclose(t[1:, 1], y_in + sgn * 6.0)        # base, farther into the zone


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
