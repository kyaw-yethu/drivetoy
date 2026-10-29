"""DriveToy stochastic choices (README, "Stochasticity"): every road user draws a style
s in [0, 1] (cautious to assertive), its per-driver values are biased by it, and each major
choice is True with probability p = sigmoid((cue - c50') / w), where c50' = c50 - shift (2 s - 1)
and True is the assertive outcome (go, take the gap, cross, cut in, pull out)."""
import numpy as np

CHOICE_KINDS = ["yellow", "left_gap", "stop_wave", "rtor", "overtake", "nudge", "merge", "cutin", "ped_cross"]
DEC_COLS = ["frame", "slot", "kind", "cue", "p", "outcome", "ego_rel"]
DRIVER_COLS = ["style", "v0frac", "T", "s0", "a", "b", "a_lat", "turn_dist", "d_pref", "d_turn", "stop_dist"]
MAX_DECISIONS = 96


def choice_p(cue, c50, w):
    return float(1.0 / (1.0 + np.exp(-(cue - c50) / w)))


def style_c50(cfg, kind, style):
    """(c50', w) of a choice for a driver of this style."""
    c50, w, shift = cfg["DT_CHOICE"][kind]
    return c50 - shift * (2.0 * style - 1.0), w


def cautious_gap(cfg, kind="left_gap", p=0.9):
    """v3: the cue at which even the most cautious driver (style 0) takes the assertive outcome
    with probability p (5.27 s of oncoming gap for left_gap with the defaults)."""
    c50, w = style_c50(cfg, kind, 0.0)
    return float(c50 + w * np.log(p / (1.0 - p)))


def biased(rng, rng_range, style, corr, up=True):
    """A value in ``rng_range`` that leans to its upper end for assertive drivers (``up``) or to
    its lower end, with weight ``corr`` on the style and the rest drawn freely."""
    lo, hi = rng_range
    u = corr * (style if up else 1.0 - style) + (1.0 - corr) * rng.rand()
    return float(lo + (hi - lo) * u)


def driver_values(rng, cfg, style, heavy=False):
    """Per-driver values (README, "Minor variation"), drawn once when the agent appears. v3: trucks
    and buses (``heavy``) draw their desired speed from DT_V0_FRAC_HEAVY."""
    c = cfg["DT_STYLE_CORR"]
    lanes = cfg["DT_LANE_OFFSET"], cfg["DT_LANE_DRIFT"], cfg["DT_SPEED_DRIFT"], cfg["DT_TURN_OFFSET"]
    return dict(
        v0frac=biased(rng, cfg["DT_V0_FRAC_HEAVY"] if heavy and cfg.get("DT_V3") else cfg["DT_V0_FRAC"], style, c),
        T=biased(rng, cfg["DT_IDM_T"], style, c, up=False),
        s0=biased(rng, cfg["DT_IDM_S0"], style, c, up=False),
        a=biased(rng, cfg["DT_IDM_A"], style, c),
        b=biased(rng, cfg["DT_IDM_B"], style, c),
        a_lat=biased(rng, cfg["DT_A_LAT"], style, c),
        turn_dist=biased(rng, cfg["DT_TURN_DIST"], style, c, up=False),
        stop_dist=biased(rng, cfg["DT_STOP_DIST"], style, c, up=False),
        d_pref=float(rng.uniform(-1, 1) * lanes[0]),
        d_turn=float(rng.uniform(-1, 1) * lanes[3]),
        drift=(float(rng.uniform(0, 1) * lanes[1]), float(rng.uniform(*cfg["DT_DRIFT_LEN"])), float(rng.uniform(0, 2 * np.pi)),
               float(rng.uniform(0, 1) * lanes[2]), float(rng.uniform(*cfg["DT_DRIFT_LEN"])), float(rng.uniform(0, 2 * np.pi))),
    )


def end_label(dec, t0=31, t1=40, lo=0.1, hi=0.9, made=0.05):
    """Aspect E from a decision log (rows as DEC_COLS): 'pending' if a choice that concerns the
    ego falls in frames t0+1 to t1 with odds between lo and hi, 'decided' if one with odds
    between made and 1 - made fell in frames up to t0, otherwise 'none'."""
    dec = np.asarray(dec, float).reshape(-1, len(DEC_COLS))
    rel = dec[dec[:, 6] > 0]
    if ((rel[:, 0] > t0) & (rel[:, 0] <= t1) & (rel[:, 4] >= lo) & (rel[:, 4] <= hi)).any():
        return "pending"
    if ((rel[:, 0] <= t0) & (rel[:, 4] >= made) & (rel[:, 4] <= 1 - made)).any():
        return "decided"
    return "none"
