"""Checks of the following thresholds: python3 tests/test_labels.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from f_toy.datasets.drivetoy.diversity import CUTIN_GAP_S, FOLLOW_GAP, LEAD_GAP, follow_binds


def _rec(gap, v):
    """Ego (slot 0) and a leader (slot 1), both 4.5 m long, ``gap`` metres bumper to bumper."""
    ag = np.zeros((1, 2, 10), np.float32)
    ag[0, :, 6] = 4.5
    ag[0, 1, 1] = gap + 4.5
    return dict(agents=ag, ego=np.array([[0.0, 0.0, 0.0, v]], np.float32))


def test_thresholds():
    assert (FOLLOW_GAP, LEAD_GAP, CUTIN_GAP_S) == (7.0, 10.0, 2.0)


def test_follow_binds():
    assert follow_binds(_rec(6.9, 15.0), 0, 1, FOLLOW_GAP)
    assert not follow_binds(_rec(7.1, 15.0), 0, 1, FOLLOW_GAP)        # flat: speed does not stretch it
    assert follow_binds(_rec(9.9, 3.0), 0, 1, LEAD_GAP)
    assert not follow_binds(_rec(10.5, 3.0), 0, 1, LEAD_GAP)
    assert follow_binds(_rec(25.0, 13.0), 0, 1, LEAD_GAP, CUTIN_GAP_S)  # a cut-in binds within 2 s
    assert not follow_binds(_rec(5.0, 13.0), 0, 0, LEAD_GAP)            # no leader


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
