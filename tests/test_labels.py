"""Checks of the following thresholds: python3 tests/test_labels.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from drivetoy.config import DRIVETOY_DEFAULT_CFG as CFG
from drivetoy.diversity import CUTIN_GAP_S, LEAD_GAP, follow_binds, follow_gaps, lead_gaps

V3 = dict(CFG)


def _rec(gap, v):
    """Ego (slot 0) and a leader (slot 1), both 4.5 m long, ``gap`` metres bumper to bumper."""
    ag = np.zeros((1, 2, 10), np.float32)
    ag[0, :, 6] = 4.5
    ag[0, 1, 1] = gap + 4.5
    return dict(agents=ag, ego=np.array([[0.0, 0.0, 0.0, v]], np.float32))


def test_thresholds():
    assert (LEAD_GAP, CUTIN_GAP_S) == (10.0, 2.0)


def test_follow_binds():
    assert follow_binds(_rec(9.9, 3.0), 0, 1, LEAD_GAP)
    assert not follow_binds(_rec(10.5, 3.0), 0, 1, LEAD_GAP)
    assert follow_binds(_rec(25.0, 13.0), 0, 1, LEAD_GAP, CUTIN_GAP_S)  # a cut-in binds within 2 s
    assert not follow_binds(_rec(5.0, 13.0), 0, 0, LEAD_GAP)            # no leader


def test_follow_time_gap():
    assert follow_gaps(V3) == (0.0, 2.0) and lead_gaps(V3) == (10.0, 2.0)
    assert follow_binds(_rec(25.0, 13.0), 0, 1, *follow_gaps(V3))        # 25 m <= 2 s * 13 m/s
    assert not follow_binds(_rec(27.0, 13.0), 0, 1, *follow_gaps(V3))
    assert follow_binds(_rec(9.5, 2.0), 0, 1, *lead_gaps(V3))            # slow: the 10 m floor holds


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
