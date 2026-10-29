"""DriveToy's own defaults — clip geometry, simulator constants and the
rulebook's parameters (docs/DRIVETOY.md §2). Merged into the resolved cfg by
``f_toy.config.get_config`` when ``dataset="drivetoy"``, the same way
CubeToy's ``CUBETOY_DEFAULT_CFG`` is.

Rule parameters come in pairs: ``*_POLICY`` is what agents do, ``*_CHECK`` is
what ``rules.py`` counts as a violation. The check is always looser than the
policy so normal driving never sits on the boundary.
"""

DRIVETOY_DEFAULT_CFG = dict(
    # ---- clip geometry ----
    FRAMES=32, IMG=128,
    FRAMES_FUTURE=32,     # label-only frames simulated past the rendered clip
    DT_FPS=5.0,           # frames per second (0.2 s per frame)
    DT_SUBSTEPS=4,        # integration substeps per frame
    DT_WARMUP=25,         # frames simulated before frame 0 (queues and signal phases settle)
    DT_BRANCH_FRAME=4,    # last frame shared by every branch (= end of ctx2's observed frames)
    DT_BRANCHES=1,        # futures simulated per context (K)

    # ---- scene mix ----
    DT_TEMPLATES=dict(signal4=0.35, stop4=0.2, multilane=0.25, residential=0.2),
    DT_MAX_AGENTS=64,
    DT_MAX_SIGNALS=8,
    DT_MAX_EVENTS=96,
    DT_MIN_EVENTS=1,      # decision events required in the ego's view after DT_BRANCH_FRAME
    DT_VIEW_RANGE=70.0, DT_VIEW_HALF_ANGLE=60.0,
    DT_RESAMPLE_TRIES=30,

    # ---- the ego (camera car): placed and biased so its own trajectory is varied ----
    DT_EGO_TURN=dict(L=0.4, S=0.2, R=0.4),   # turn weights at a junction (others use sim.TURN_WEIGHTS)
    DT_EGO_INNER_P=0.4,          # signal4: start in the left-turn lane
    DT_EGO_LINE_DIST=dict(signal4=(38.0, 52.0), stop4=(6.0, 22.0)),  # to the stop line at frame 0. signal4:
        # the route decision (30 m out) falls after DT_BRANCH_FRAME; stop4: close, or the full stop and the
        # queue eat the clip (its branches differ through the cross traffic instead)
    DT_EGO_CW_DIST=(35.0, 80.0),     # residential: to the crosswalk at frame 0
    DT_EGO_PED_P=0.7,                # a pedestrian reaches the curb ahead of the ego 1.2-3.4 s into the clip
    DT_EGO_LC_P=0.6,                 # multilane: the ego wants to change lanes (drawn after the branch frame)
    DT_EGO_CLEAR_P=0.3,          # each vehicle ahead of the ego on its lane is removed with this probability
    DT_EGO_DROP_P=0.4,           # multilane: start in the closing lane (forced merge)
    DT_EGO_MOBIL_THRESH=0.05,    # multilane: the ego changes lanes more readily than traffic
    # signal4: the controller's phase when the warm-up starts, relative to the ego's approach —
    # its own green / a turn to green soon / unchanged random (which often means waiting a whole clip)
    DT_EGO_SIGNAL=dict(own_green=0.4, green_soon=0.5, random=0.1),
    DT_STOP4_DENSITY=(0.15, 0.5),

    # ---- vehicle dynamics (IDM / MOBIL) ----
    DT_IDM_T=(1.0, 2.2), DT_IDM_S0=(1.5, 3.0), DT_IDM_A=(1.0, 2.0), DT_IDM_B=(1.5, 3.0),   # per driver
    DT_B_MAX=6.0,          # hard braking limit
    DT_V0_FRAC=(0.75, 0.97),  # desired speed as a fraction of the limit (R0 holds by construction)
    DT_A_LAT=(2.0, 3.0),   # comfortable lateral acceleration in turns, per driver
    DT_LC_DURATION=3.0, DT_MOBIL_THRESH=0.2, DT_MOBIL_POLITE=(0.1, 0.6),
    DT_LC_WILL=0.35,       # a worthwhile discretionary lane change is taken at a given 1 s check with this probability
    DT_MERGE_WILL=0.4,     # a safe gap for a forced merge is taken at a given 0.6 s check with this probability

    # ---- rulebook (policy vs check) ----
    R0_SPEED_TOL=0.5,
    R1_STOP_DECEL=3.0, R1_STOP_DECEL_CHECK=2.0,
    R3_GAP_POLICY=4.0, R3_GAP_CHECK=2.5,
    R4_STOP_TIME=1.0, R4_STOP_V=0.3, R4_STOP_ZONE=5.0, R4_TIE=0.4,
    R5_LATERAL_CHECK=3.0,
    R6_GAP_MIN=1.0, R6_HEADWAY_CHECK=0.5,
    R7_SIGNAL_POLICY=1.6, R7_SIGNAL_CHECK=1.0, R7_FOLLOWER_DECEL_POLICY=2.0, R7_FOLLOWER_DECEL_CHECK=4.0,
    P2_DECEL_POLICY=3.0, P2_DECEL_CHECK=4.0,

    # ---- lamps ----
    L1_BRAKE_DECEL=0.6,
    L2_TURN_DIST=30.0,     # turn decision + indicator onset, metres before the stop line
    L2_BLINK_FRAMES=4,     # 2 on / 2 off at 5 Hz (0.8 s period, no aliasing)

    # ---- signals ----
    SIG_GREEN_WALK=(3.0, 10.0),   # WALK part of green; hidden, drawn lazily (short: a toy wants frequent changes)
    SIG_FLASH=6.0,                # flashing don't-walk (+ countdown) at the end of green
    SIG_YELLOW=3.0, SIG_ALL_RED=1.5,
    SIG_ARROW=(3.0, 5.0), SIG_ARROW_P=0.5,   # protected-left phase before the green ball
    SIG_COUNTDOWN_P=0.6,

    # ---- pedestrians ----
    PED_SPEED=(1.2, 1.6), PED_CROSS_P=0.7, PED_DELAY=(0.0, 1.5),

    # ---- camera ----
    CAM_HEIGHT=(1.3, 1.5), CAM_PITCH=(-6.0, -2.0), CAM_FOV=70.0, CAM_FORWARD=0.5,

    # ---- rendering (Blender / Cycles) ----
    DT_RENDER_GPUS="auto",   # "auto" = every GPU with < 1 GB in use; or "2,3"
    DT_RENDER_PROCS=4,       # Blender processes per GPU (128 px frames leave the GPU mostly idle)
    DT_RENDER_SAMPLES=16,
    DT_CUE_SCALE=1.5,        # size multiplier for vehicle lamps and pedestrian signal heads
    DT_SIGNAL_SCALE=4.5,     # v2: size multiplier for vehicle signal heads (real size = 1)
    DT_SIGNAL_LAYOUT="vertical",   # v2: "horizontal" puts the lamps of the mast-arm heads side by side
    DT_POLE_HEAD=False,      # v2: no head on the far-right corner pole (the pole still carries the mast arm)
    DT_NEAR_HEADS=False,     # v2: no near-side head at the near-right corner
    DT_HEADS_CORNER=True,    # v2: far-side pole on the nose of the far-right corner, so the heads sit nearer the approach
    DT_OVERTAKE_CHECK=True,  # v2: reject a lane change that does not close in on a slower leader (diversity.overtake_problems)
    DT_HIDE_NEAR_HEADS=True,   # v2: do not render vehicle heads on the ego's near side of the junction
    DT_INDICATOR_SCALE=2.25, # v2: size multiplier for indicator lamps
    DT_DUSK_SUN_AWAY=60.0,   # v2: at dusk the sun stays this many degrees away from the camera's heading
    DT_LAMP_EMIT=dict(brake=1.5, left=2.5, right=2.5, signal=1.0, ped=3.0),   # emission strengths (AgX desaturates brighter lamps; v1 used 6/10/10/12/8, v1 signal 4.0)
    DT_SIGNAL_COLORS=dict(red=(1.0, 0.02, 0.0), yellow=(1.0, 0.5, 0.0), green=(0.0, 0.85, 0.08), arrow=(0.0, 0.85, 0.08)),
        # v2: deeper signal lamps, so the v1 VAE keeps green apart from yellow (v1 green was 0.1, 1.0, 0.3 at 4.0)
    DT_VIEW=("AgX", "AgX - Medium High Contrast"),   # Blender view transform and look

    # ---- v2: cued futures and ego-first sampling (DATASETv2_plan.md) ----
    DT_V2=True,              # cued signal timing, decided ego intent, decided pedestrians, target-driven scenes
    DT_CUE_FRAME=31,         # last visible frame: hidden variables the ego reacts to are decided by then
    DT_TRIES=30,             # scene attempts per clip while looking for its target labels
    # shares of the four diversity aspects (each sums to 1); A is met, B, C, D are targets
    DT_SHARES=dict(
        A={"no light": 0.50, "steady green": 0.06, "steady red": 0.05, "red to green": 0.10,
           "green to yellow": 0.08, "yellow to red": 0.05, "arrow": 0.07, "permissive left": 0.09},
        B={"none": 0.12, "lead vehicle": 0.10, "signal": 0.19, "stop sign": 0.07, "oncoming traffic": 0.09,
           "crossing traffic": 0.07, "pedestrian": 0.12, "lane end": 0.08, "speed limit": 0.06,
           "cut-in vehicle": 0.07, "parked obstacle": 0.03},
        C={"cruise": 0.18, "follow": 0.12, "slow down": 0.14, "stop": 0.16, "wait": 0.06, "pull away": 0.16,
           "stop and go": 0.18},
        D={"straight": 0.40, "left turn": 0.18, "right turn": 0.16, "lane change": 0.10, "merge": 0.08,
           "nudge around": 0.08},
        E={"pending": 0.40, "decided": 0.45, "none": 0.15}),
    DT_BATCH=500,            # clips planned per batch; later batches steer B, C, D toward their shares

    # ---- stochastic choices and driver variation (README, "Stochasticity") ----
    DT_STYLE_CORR=0.6,       # weight of the driver's style in each per-driver value (the rest is drawn freely)
    DT_LANE_OFFSET=0.25,     # preferred offset from the lane centre, +- metres per driver
    DT_LANE_DRIFT=0.1,       # slow sideways drift around it, metres
    DT_TURN_OFFSET=0.4,      # tight or wide arc: sideways offset at the middle of a turn, +- metres
    DT_SPEED_DRIFT=0.03,     # desired speed wanders by this fraction (capped at 0.99 of the limit)
    DT_DRIFT_LEN=(60.0, 150.0),   # metres per period of both drifts
    DT_START_DELAY=(0.3, 1.5),    # s before a stopped vehicle moves off once it may
    DT_STOP_DIST=(0.5, 3.0),      # where a vehicle stops before its line, metres
    DT_TURN_DIST=(20.0, 45.0),    # turn indicator onset before the line, metres
    DT_LC_LEAD=(1.5, 4.0),        # lane change: indicator before the move, s (R7 checks 1.0 s)
    DT_LC_DUR=(2.5, 5.0),         # lane change: length of the move, s
    DT_YELLOW_REACT=(0.3, 0.7),   # s from yellow onset to the stop-or-go choice
    DT_YELLOW_BAND=(2.2, 4.5),    # braking needed at onset, m/s^2: stops below, goes above, chooses between
    DT_GAP_FLOOR=3.0,             # s: a left turn, a right turn on red never takes a gap below this (R3 checks 2.5)
    DT_PATIENCE=(1.5, 4.0),       # s an overtaker follows a slow leader before it looks for a gap
    DT_WAVE_WINDOW=0.4,           # stop sign: an arrival this many seconds later may be waved through (R4's tie)
    DT_RTOR=True,                 # right turn on red after a full stop
    DT_PED_HESITATE=0.3,          # share of undecided pedestrians (p near 0.5) that pause at the curb first
    DT_HESITATE=(1.0, 3.0),       # s of that pause
    # major choices (c50, w, shift): p = sigmoid((cue - c50') / w), c50' = c50 - shift (2 style - 1)
    DT_CHOICE=dict(yellow=(3.0, 0.4, 0.4),      # cue: deceleration needed at onset, m/s^2; True = go
                   left_gap=(4.0, 0.35, 0.5),   # cue: oncoming time to the conflict, s; True = take the gap
                   stop_wave=(0.2, 0.1, 0.1),   # cue: how much later the other car arrived, s; True = go in turn
                   rtor=(5.0, 0.6, 1.0),        # cue: cross traffic's time to the conflict, s; True = go on red
                   overtake=(4.0, 0.7, 1.0),    # cue: time gap to traffic in the target lane, s; True = pull out
                   nudge=(2.0, 0.6, 0.8),       # cue: spare time in the oncoming gap, s; True = go around
                   merge=(2.0, 0.4, 0.5),       # cue: smaller time gap in the target lane, s; True = take it
                   cutin=(0.9, 0.12, 0.15),     # cue: time gap left to the car behind, s; True = cut in
                   ped_cross=(3.0, 0.6, 0.8)),  # cue: nearest vehicle's time to the crosswalk, s; True = cross
    DT_FOLLOW_GAP=7.0,       # speed label "follow": the leader binds within this many metres
    DT_LEAD_GAP=10.0,        # causes "lead vehicle" and "parked obstacle": within this many metres
    DT_CUTIN_GAP_S=2.0,      # cause "cut-in vehicle": within this time gap
    DT_FORKS=16,             # futures per sample forked at DT_CUE_FRAME (future 0 is the sample's own)
    DT_WRITE_VIDEOS=True,    # write videos.npy (the train split keeps render/ only)
    DT_PENDING_MIN=0.2,      # E: pending gets more weight while its share is below this

    # ---- v3 (version-3-revision.md; plan docs/superpowers/plans/2026-10-29-drivetoy-v3.md) ----
    DT_V3=False,             # v3: every change below is used only when this is set (V3_OVERRIDES)
    DT_V0_FRAC_HEAVY=(0.65, 0.85),   # v3: desired speed of trucks and buses, fraction of the limit
    DT_MIN_V0_FRAC=0.6,      # v3: no vehicle targets less than this fraction without a reason in the scene
    DT_FOLLOW_GAP_S=2.0,     # v3: "follow" and lead causes bind within this bumper time gap at the ego's speed
    DT_OVERTAKE_V0_FRAC=(0.60, 0.75),   # v3: an overtaken lead drives at this fraction of the limit
    DT_OVERTAKE_EGO_FRAC=(0.90, 0.97),  # v3: ... and the overtaking ego at this one
    DT_ONCOMING_MIN=2.0,     # v3: oncoming headway = this minimum (s) ...
    DT_ONCOMING_MEAN=(4.0, 8.0),        # v3: ... plus an exponential tail with a mean drawn here per clip (s)
    DT_LEFT_SHARE=0.25,      # v3: share of a signal4 approach's traffic in its left-only lane

    SEED=0,
)

# v3 builds: the switch plus the prompter starts (D6): a full stop of 0.8 s at a stop sign and
# 0.2 to 0.6 s before moving off once the way is clear
V3_OVERRIDES = dict(DT_V3=True, R4_STOP_TIME=0.8, DT_START_DELAY=(0.2, 0.6))
