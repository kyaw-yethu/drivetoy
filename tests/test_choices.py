"""Checks of choices.py and the per-driver values: python3 tests/test_choices.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from drivetoy.choices import choice_p, driver_values, end_label, style_c50
from drivetoy.config import DRIVETOY_DEFAULT_CFG as CFG

V3 = dict(CFG)


def test_choice_p():
    assert abs(choice_p(3.0, 3.0, 0.4) - 0.5) < 1e-9
    assert choice_p(3.5, 3.0, 0.4) > choice_p(3.2, 3.0, 0.4) > 0.5 > choice_p(2.6, 3.0, 0.4)


def test_style_shift():
    for kind in CFG["DT_CHOICE"]:
        lo, _ = style_c50(CFG, kind, 1.0)
        hi, _ = style_c50(CFG, kind, 0.0)
        assert lo < hi, kind          # an assertive driver takes the assertive outcome at a smaller cue


def test_driver_values_in_range():
    rng = np.random.RandomState(0)
    for _ in range(2000):
        s = rng.rand()
        dv = driver_values(rng, CFG, s)
        for key, rk in (("v0frac", "DT_V0_FRAC"), ("T", "DT_IDM_T"), ("s0", "DT_IDM_S0"), ("a", "DT_IDM_A"),
                        ("b", "DT_IDM_B"), ("a_lat", "DT_A_LAT"), ("turn_dist", "DT_TURN_DIST"), ("stop_dist", "DT_STOP_DIST")):
            lo, hi = CFG[rk]
            assert lo <= dv[key] <= hi, (key, dv[key])
        assert abs(dv["d_pref"]) <= CFG["DT_LANE_OFFSET"] and abs(dv["d_turn"]) <= CFG["DT_TURN_OFFSET"]
        assert dv["drift"][0] <= CFG["DT_LANE_DRIFT"] and dv["drift"][3] <= CFG["DT_SPEED_DRIFT"]


def test_style_correlation():
    rng = np.random.RandomState(1)
    s = rng.rand(3000)
    T = np.array([driver_values(rng, CFG, x)["T"] for x in s])
    v = np.array([driver_values(rng, CFG, x)["v0frac"] for x in s])
    assert np.corrcoef(s, T)[0, 1] < -0.5 and np.corrcoef(s, v)[0, 1] > 0.5


def test_end_label():
    row = lambda f, p, rel=1, slot=0, kind=1: [f, slot, kind, 3.0, p, 1, rel]
    assert end_label([row(34, 0.5)]) == "pending"
    assert end_label([row(34, 0.97)]) == "none"            # a foregone choice is not pending
    assert end_label([row(20, 0.6), row(34, 0.95)]) == "decided"
    assert end_label([row(34, 0.5, rel=0)]) == "none"      # someone else's choice
    assert end_label([row(34, 0.5, slot=7)]) == "none"     # another road user's choice is never pending
    assert end_label([row(34, 1.0, kind=0)]) == "pending"  # a determined response (yellow) after frame 31
    assert end_label([row(45, 0.5)]) == "none"             # too late to count
    assert end_label(np.zeros((0, 7))) == "none"


def test_heavy_driver_values():
    rng = np.random.RandomState(2)
    heavy = [driver_values(rng, V3, rng.rand(), heavy=True)["v0frac"] for _ in range(2000)]
    light = [driver_values(rng, V3, rng.rand(), heavy=False)["v0frac"] for _ in range(2000)]
    assert 0.65 <= min(heavy) and max(heavy) <= 0.85, (min(heavy), max(heavy))
    assert 0.75 <= min(light) and max(light) <= 0.97, (min(light), max(light))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
