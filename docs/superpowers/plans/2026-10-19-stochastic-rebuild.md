# DriveToy Stochastic Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild DriveToy (train 10,000, val 1,500, test 1,500 x 2) into `/root/projects/drive_v2s/` with cue-dependent stochastic choices, minor per-driver and per-event variation, aspect E, and 16 future forks per sample; then zip data and code and upload them to the Drive folder.

**Architecture:** A new module `choices.py` holds driver styles, per-driver values and the logistic choice with its decision log; `sim.py` calls it at each choice point. `scenarios.py` gains pending recipes and an E target; `diversity.py` gains E and the new following rules; `rules.py` gains the right-turn-on-red and waved-through exceptions. The builder writes 16 forks per sample and a per-split `future_set.npz`.

**Tech Stack:** Python 3 (numpy 1.24), Blender 4.2 Cycles (unchanged renderer), rclone (`gdrive:` remote, root folder already the target).

**Spec:** `README.md`, section "Stochasticity", plus the user's later decisions recorded under Global Constraints.

## Global Constraints

- A to D shares come first; E aims at pending 40%, decided 45%, none 15%, and pending gets more weight when it falls under 20%.
- No attempt is kept or rejected on the sample's own frames 32 to 63; validity checks after frame 31 (`ran_out`, cue problems) run on all 16 forks together.
- Following for the speed label "follow" binds within 7 m; the cause "lead vehicle" binds within 10 m; the cause "cut-in vehicle" binds while the cut-in car leads the ego within 2 s.
- Overtakes start from following a slow leader within about 10 m, so their cause is "lead vehicle". Cut-ins start 0.6 to 1.2 s ahead of the ego.
- Every random range stays inside the rule checker: yellow random only for $2.2 < a_{need} < 4.5$ and only if the line is reached before red; left gap $\tau \ge 3.0$ s; lane-change indicator at least 1.5 s; speed below the limit plus the R0 tolerance.
- The ego's light holds after frame 31 unless yellow; after frame 31 a choice affecting the ego needs its cue on screen by frame 31.
- Output `/root/projects/drive_v2s/{pilot,train,val,test}`; render on all 4 GPUs; train keeps `render/` without `videos.npy`, as in the previous release.
- Docs: concise, LaTeX for symbols in `.md`, none of the user's avoided words.

## Review Focus

- A pending choice whose cue is not visible by frame 31 (pedestrian outside the view, cutter not yet signalling): `end_label` must call it decided or none, not pending.
- Stochastic rules creating rule violations (yellow "go" entering on red, waved-through car flagged by R4, right turn on red flagged by R1): `rules.check` on the pilot must report none.
- Lateral offset and drift mistaken for a nudge: `path_label` must not return "nudge around" for a straight clip with a 0.25 m offset.
- Forks diverging before frame 31 or not reproducing: fork 0 must equal the stored label exactly.
- Disk exhaustion during build or zip (65 GB free): the zip is streamed or written to `/dev/shm`, and train skips `videos.npy`.

---

### Task 1: Choices, styles and per-driver values

**Files:** Create `f_toy/datasets/drivetoy/choices.py`; modify `config.py`, `sim.py` (`Vehicle.__init__`, `Pedestrian.__init__`, `pose`, `_v0`); test `tests/test_choices.py`.

**Interfaces:**
- Produces: `CHOICE_KINDS` (list of names: yellow, left_gap, stop_wave, rtor, overtake, nudge, merge, cutin, ped_cross); `choice_p(cue, c50, w) -> float`; `Sim.decide(chooser, kind, cue, key=None) -> bool` (logs `[frame, slot, kind, cue, p, outcome, ego_rel]` in `sim.decisions`, caches by key); `driver_values(rng, cfg, style, kind) -> dict`.
- Config keys: `DT_STYLE_CORR`, ranges `DT_V0_FRAC=(0.75, 0.97)`, `DT_IDM_T=(1.0, 2.2)`, `DT_IDM_S0=(1.5, 3.0)`, `DT_IDM_A=(1.0, 2.0)`, `DT_IDM_B=(1.5, 3.0)`, `DT_A_LAT=(2.0, 3.0)`, `DT_LANE_OFFSET=0.25`, `DT_LANE_DRIFT=0.1`, `DT_SPEED_DRIFT=0.03`, `DT_CHOICE` (per kind: c50, w, style shift).

- [ ] Test: `choice_p` is 0.5 at `c50`, increasing in the cue; an assertive style lowers `c50` for go-type choices; per-driver values stay inside their ranges.
- [ ] Implement; drifts are sinusoids of arclength drawn once per driver (no per-step draws); `v0frac` with drift capped at 0.99.
- [ ] Run `python3 tests/test_choices.py`: all asserts pass.

### Task 2: Major choices in the simulator

**Files:** modify `sim.py` (`_may_enter`, `_oncoming_clear`, `_stop_order_first`, `_plan_nudge`, `_lane_changes`, `_ped_decide`, step bookkeeping); add event `WAVE`.

- [ ] Yellow: decision at onset plus a 0.5 to 1.0 s reaction delay; stop below 2.2, go above 4.5 or when the line cannot be reached in time is false.
- [ ] Left gap, nudge, merge, cut-in, overtake: one decision per gap, keyed by the vehicle that bounds the gap; the ego's overtake fires only after following a slow leader for its patience time.
- [ ] Stop sign: the first vehicle may wave a near-tie arrival through (log `WAVE`); right turn on red after a full stop, keyed by the cross-traffic gap.
- [ ] Pedestrians at the curb: cross, wait or hesitate with odds from the nearest vehicle's time to the crosswalk; a pedestrian not visible at frame 31 never steps out in front of the ego afterwards; other vehicles start no uncued lane change into the ego's lane after frame 31.
- [ ] Minor events: start-up delay, stop margin, creep at stop signs, turn-indicator distance, lane-change lead and duration.
- [ ] Probe: `python3 scripts/v2s_probe.py --n 40` simulates 40 targets and prints decision counts per kind, p histogram and `rules.check` violations: every kind fires and there are no violations.

### Task 3: Labels, E and rule checker

**Files:** modify `diversity.py`, `rules.py`; test `tests/test_labels.py`.

- [ ] Following thresholds as in Global Constraints; nudge detection relative to the clip's median lateral offset; new `overtake_problems` (follows within 10 m for at least 4 frames before pulling out, leader below 0.6 of the limit).
- [ ] `end_label(decisions, cfg) -> "pending" | "decided" | "none"`; `cue_problems` updated for the relaxed pedestrian rule and uncued lane changes.
- [ ] `rules.check`: R1 skips a right turn after a full stop in the zone; R4 skips an order violation covered by a `WAVE` event.
- [ ] Run `python3 tests/test_labels.py`: synthetic logs give the expected E; the two exceptions hold and real violations are still caught.

### Task 4: Pending recipes, sampler and forks

**Files:** modify `scenarios.py` (COMBOS with E, pending recipes, `plan_targets`, `_score`, `simulate` fork stage), `builder.py` (labels, `future_set.npz`, E counts), `scripts/build_drivetoy_v2.py` (default out `/root/projects/drive_v2s`).

- [ ] Pending recipes: yellow onset, permissive left, right turn on red, stop-sign near tie, overtake, merge, cut-in, parked car, pedestrian at the curb.
- [ ] `simulate` returns 16 forks per branch (fork 0 = own label) and rejects only on all-fork validity.
- [ ] Builder stores `decisions`, `drivers`, E in `diversity`, per-fork ego, agents (float16), signals, rule states and decisions; assembles `future_set.npz`.
- [ ] Smoke build: `python3 scripts/build_drivetoy_v2.py --split smoke --out /root/projects/drive_v2s --gpus 0,1,2,3` finishes, fork 0 equals the label.

### Task 5: Pilot and checks

**Files:** create `scripts/v2s_checks.py`.

- [ ] Checks: shares A to E, pending clips with two outcomes among the forks (pass at 70%), decided and none spread at frame 63, odds against drawn p by cue bin, sibling divergence, `rules.check`, re-simulation.
- [ ] Build the 500-clip pilot, run the checks, tune `DT_CHOICE` and recipe timing until they pass.

### Task 6: Full build, packaging, docs

- [ ] Build train, val, test; run the checks on val and test.
- [ ] Zip `/root/projects/drive_v2s` (render npz stored) and the code (without `.git`) to `/dev/shm`, upload with `rclone copy ... gdrive:`, compare md5.
- [ ] README: data section and shares for the new build.
