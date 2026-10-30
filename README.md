# DriveToy v2

DriveToy v2 is a procedural driving dataset: a lane-graph traffic simulator (IDM car following, MOBIL lane changes, signals, stop signs, pedestrians) rendered with Blender Cycles from the ego car's front camera. It is built by the code in this repository; [DATASETv2_revision.md](DATASETv2_revision.md) describes what changed from version 1.

**Contents**

- [Data](#data)
- [Diversity](#diversity)
- [Scripts](#scripts)
- [Stochasticity](#stochasticity)
  - [Principles](#principles)
  - [Sampling priority](#sampling-priority)
  - [Major choices](#major-choices)
  - [Minor variation](#minor-variation)
  - [After frame 31](#after-frame-31)
  - [Future sets](#future-sets)
  - [Labels and checks](#labels-and-checks)
  - [Other changes](#other-changes)
  - [Build](#build)
- [DriveToy v3](#drivetoy-v3)

## Data

The splits are in `drive_v2s_data.zip` in the shared Google Drive folder.

| Split | Clips | Samples | `signal4` | `stop4` | `multilane` | `residential` |
|---|---:|---:|---:|---:|---:|---:|
| train | 10,000 | 10,000 | 5,000 | 1,294 | 2,021 | 1,685 |
| val | 1,500 | 1,500 | 750 | 199 | 289 | 262 |
| test | 1,500 | 3,000 | 1,500 | 412 | 622 | 466 |

Each sample has 32 visible frames at $128 \times 128$ and 5 Hz (6.4 s), followed by 32 label-only frames. A test clip has two futures that share frames 0 to 4, so it gives two samples. Each split directory holds `annotations.npz` (per-sample labels, including `diversity` for aspects A to E, the decision log `decisions` and the per-driver values `drivers`), `future_set.npz` (16 futures of frames 32 to 63 per sample, forked at frame 31; future 0 is the sample's own label), `meta.json`, `checks.json`, and the per-sample `labels/`, `scenes/` and `render/` files. Val and test also hold `videos.npy` (samples, 3, 32, 128, 128); train keeps its frames in `render/<n>.npz` only. The GIFs are in `viz/`.

## Diversity

Every sample has one label per aspect, computed from frames 0 to 31 (cause from frames 5 to 31; E from the choices in frames 0 to 40), so the shares of an aspect sum to 100%. The tables give the sampler's target share and the share actually built in each split; the test shares count both futures of a clip. Under each table, a GIF shows one val clip per label, the ego view beside the top-down map, with the clip's labels above it.

**A. Traffic light** (the light that governs the ego).

| Label | Description | Target | Train | Val | Test |
|---|---|---:|---:|---:|---:|
| no light | no traffic light governs the ego (stop-sign junctions and open road) | 50% | 50.0% | 50.0% | 50.0% |
| steady green | green throughout; the ego drives through or turns | 6% | 6.0% | 6.0% | 6.0% |
| steady red | red throughout; the ego stops at the line or waits there | 5% | 5.0% | 5.0% | 5.0% |
| red to green | the light turns green on screen and the ego pulls away | 10% | 10.0% | 10.0% | 10.0% |
| green to yellow | the light turns yellow on screen; the ego stops, or goes if it is too close to stop | 8% | 8.0% | 8.0% | 8.0% |
| yellow to red | the light turns red on screen; the ego stops, or clears the junction | 5% | 5.0% | 5.0% | 5.0% |
| arrow | a green left-turn arrow lets the ego turn left without yielding | 7% | 7.0% | 7.0% | 7.0% |
| permissive left | green ball while turning left; the ego waits for a gap in oncoming traffic | 9% | 9.0% | 9.0% | 9.0% |

![One val clip per A label: ego view and top-down map](diversity_A_light_v2s.gif)

**B. Cause** (what the ego reacts to in most of frames 5 to 31).

| Label | Description | Target | Train | Val | Test |
|---|---|---:|---:|---:|---:|
| none | nothing constrains the ego; free driving at the road's normal limit | 12% | 20.1% | 20.2% | 20.9% |
| lead vehicle | keeps its distance behind a slower car ahead | 10% | 11.1% | 10.9% | 12.1% |
| signal | stops for red or yellow and goes on green | 19% | 17.4% | 17.6% | 17.3% |
| stop sign | comes to a full stop at the line and waits its turn | 7% | 5.6% | 5.8% | 6.2% |
| oncoming traffic | waits for a gap in oncoming traffic before turning left | 9% | 9.2% | 8.9% | 9.1% |
| crossing traffic | at a stop sign, waits for a car on the cross street that has the right of way | 7% | 4.8% | 4.5% | 4.7% |
| pedestrian | yields to someone on or stepping onto the crosswalk | 12% | 8.5% | 8.5% | 7.3% |
| lane end | its lane ends (cones and a lane-ends sign), so it merges into the next lane | 8% | 8.0% | 7.9% | 8.2% |
| speed limit | slows down for the 30 km/h school zone | 6% | 4.9% | 5.1% | 5.1% |
| cut-in vehicle | a car from the next lane cuts in ahead of the ego | 7% | 3.6% | 3.5% | 3.0% |
| parked obstacle | a double-parked car with hazard lamps blocks the lane; the ego waits behind it or steers around | 3% | 6.9% | 7.0% | 6.0% |

![One val clip per B label: ego view and top-down map](diversity_B_cause_v2s.gif)

**C. Longitudinal profile** (what the ego's speed does).

| Label | Description | Target | Train | Val | Test |
|---|---|---:|---:|---:|---:|
| cruise | steady speed, no stop | 18% | 20.8% | 20.5% | 20.8% |
| follow | speed set by the car ahead for at least half the clip | 12% | 3.1% | 3.7% | 4.0% |
| slow down | speed drops by more than 3 m/s without stopping | 14% | 31.3% | 31.3% | 30.1% |
| stop | moving, then comes to a stop | 16% | 16.6% | 16.7% | 16.5% |
| wait | stands still for most of the clip | 6% | 7.3% | 6.2% | 7.4% |
| pull away | starts stopped, then drives off | 16% | 14.9% | 15.5% | 15.1% |
| stop and go | stops and drives off again within the clip | 18% | 5.9% | 6.1% | 6.2% |

![One val clip per C label: ego view and top-down map](diversity_C_speed_v2s.gif)

**D. Lateral manoeuvre** (what the ego's path does).

| Label | Description | Target | Train | Val | Test |
|---|---|---:|---:|---:|---:|
| straight | heading changes by less than $15^\circ$ | 40% | 77.7% | 77.6% | 77.4% |
| left turn | turns left by more than $45^\circ$ | 18% | 8.3% | 8.1% | 8.1% |
| right turn | turns right by more than $45^\circ$ | 16% | 6.1% | 6.3% | 6.0% |
| lane change | moves into the next lane, indicator on first | 10% | 2.2% | 2.3% | 2.5% |
| merge | lane change out of a lane that ends | 8% | 2.4% | 2.1% | 2.7% |
| nudge around | steers briefly sideways around an obstacle and back into its lane | 8% | 3.4% | 3.5% | 3.3% |

![One val clip per D label: ego view and top-down map](diversity_D_path_v2s.gif)

**E. Choice at frame 31** (the state of the ego's major choices, see [Stochasticity](#stochasticity)).

| Label | Description | Target | Train | Val | Test |
|---|---|---:|---:|---:|---:|
| pending | a major choice is cued on screen by frame 31, with odds between 0.1 and 0.9, and falls in frames 32 to 40 | 40.0% | 20.8% | 20.3% | 19.6% |
| decided | the major choice is made on screen, and only minor variation is left | 45.0% | 35.4% | 35.6% | 35.2% |
| none | the clip has no major choice | 15.0% | 43.8% | 44.1% | 45.2% |

![One val clip per E label: ego view and top-down map](diversity_E_choice_v2s.gif)

The sampler meets A first and treats B, C and D as targets, so the later aspects drift where a combination is hard to build. A turn from a standstill rarely passes $45^\circ$ within 6.4 s, so "straight" is over its share and both turns are under. "Speed limit" comes only from the 30 km/h school zone of the `residential` template, which keeps it under its share.

## Scripts

- `scripts/build_drivetoy_v2.py --split train|val|test`: build a split (simulation, then rendering on all idle GPUs).
- `scripts/v2_checks.py <split>`: shares, re-simulation, cued futures, rule violations, lamp size and VAE signal colour.
- `scripts/v2_fix_checks.py <split>`: dataset-wide checks of the revision's lane-end sign and right-turn fixes.
- `scripts/v2_diversity_videos.py <split> --out <dir>`: one video per aspect with an example clip per label.
- `scripts/v2_future_cues.py <split>`: how much of the ego's future in frames 32 to 63 is cued by frames 0 to 31.
- `scripts/v2_partial_update.py <split> --old <worktree>`: the revision's fixes 7 and 8 applied to a built split (lane-change clips rebuilt, the rest relabelled).

# Stochasticity

This revision makes the ego and the road users around it behave stochastically where real drivers differ. Major choices have a few distinct outcomes whose odds follow a cue the video shows, and minor variation comes from per-driver and per-event draws. Scenes, templates, rendering and the diversity aspects stay as they are.

The task encoder learns from frames 32 to 63, either forecasting them from frames 0 to 31 or forecasting each 8-frame window from the frames before it, so these futures must vary where real driving varies. In the current build they barely do. Forking each val clip 16 ways at frame 31 (`analysis/v2_fork_spread.py`) gives these shares of clips with some fork more than 2 m from fork 0 at frame 63:

| Template | As built | Holds lifted, hidden signal timing redrawn |
|---|---:|---:|
| `signal4` | 0.0% | 6.3% |
| `stop4` | 0.0% | 1.1% |
| `multilane` | 8.6% | 8.6% |
| `residential` | 0.9% | 4.7% |

The scenarios finish the ego's decisions on screen, and the ego's rules (car following, the yellow light, the gap check) contain no randomness.

## Principles

1. Randomness enters only where real drivers differ. Stopping on red, the full stop at a stop sign and yielding to a pedestrian on the crosswalk are never random.
2. A major choice has two or three outcomes whose odds follow a visible cue $c$, $p = \sigma\big((c - c_{50})/w\big)$. The values of $c_{50}$, $w$ and the style shift are in `config.py` (`DT_CHOICE`).
3. Minor variation is drawn once per driver, which frames 0 to 31 show, or once per event. Per-frame noise stays small, since no frame can predict it.
4. Every range stays within the rule checker's thresholds, so lawful clips stay lawful.
5. The same rules apply to every road user, not only to the ego.

## Sampling priority

The sampler meets the shares of aspects A to D first, since they describe the visible frames that the generative model uses, and retries attempts on these labels as the current build does. A new aspect E, the state of the ego's major choices at frame 31, takes what A to D leave and aims at the shares below. When pending falls under 20%, the sampler raises its weight even at some cost to B to D. No attempt is kept or rejected on frames 32 to 63, so pending choices keep their drawn odds.

| E label | Description | Aim |
|---|---|---:|
| pending | a major choice is cued on screen by frame 31, with odds between 0.1 and 0.9, and falls in frames 32 to 40; cue values spread over the whole random range | 40% (at least 20%) |
| decided | the major choice is made on screen, and only minor variation is left | 45% |
| none | the clip has no major choice | 15% |

## Major choices

**The ego's own choices**

| Choice | Outcomes | Cue $c$ | Random only when | Current rule |
|---|---|---|---|---|
| Yellow onset | stop / go | braking needed to stop at the line, $a_{\text{need}} = v^2/(2d)$ | $2.2 < a_{\text{need}} < 4.5$ m/s $^2$ (stops below, goes above) | stops when $a_{\text{need}} \le 3.0$ |
| Permissive left | take this gap / wait | oncoming time to the conflict point $\tau$ | $\tau \ge 3.0$ s | takes when $\tau \ge 4.0$ s |
| Stop sign, near tie | go in turn / let the other car go | the other car's arrival relative to the ego's | the other car stopped within 0.4 s of it (R4's tie) | arrival order, 0.4 s tie |
| Right turn on red | go after the full stop / wait for green | gap in cross traffic, pedestrians | after a full stop in the stop zone, cross traffic at least 3 s away | not modelled; R1 now allows it after a full stop |
| Slow leader | overtake now / stay behind | gap to traffic in the next lane | after following a slow vehicle within 10 m for its patience (1.5 to 4 s) | planned lane change |
| Parked car | go around now / wait | gap beside it | a lawful gap | goes around when there is room |
| Lane ends | merge ahead of / behind the car alongside | relative position and speed | a lawful gap | 0.4 per 0.6 s check |

**Road users the ego reacts to**

| Choice | Outcomes | Cue $c$ | Current rule |
|---|---|---|---|
| Leader at a yellow | stops / goes through | its distance and speed, brake lamps | stops when $a_{\text{need}} \le 3.0$ |
| Oncoming left-turner while the ego has green | turns across / waits | its indicator and speed, the ego's distance | takes when $\tau \ge 4.0$ s |
| Car signalling beside the ego | cuts in / stays | its indicator, the gap ahead of the ego | forced cut-in |
| Pedestrian at an unsignalised crosswalk | crosses / waits / hesitates, then crosses | heading, walking toward the curb, the ego's distance | crosses with probability 0.7 |

**Driver style.** Each driver draws a style $s \in [0, 1]$ from cautious to assertive, which shifts every $c_{50}$ and biases the per-driver values below. The way a driver behaves in frames 0 to 31 therefore hints at its later choices.

## Minor variation

Per-driver values lean toward the driver's style (`DT_STYLE_CORR`).

| Per driver | Range | Current rule |
|---|---|---|
| Desired speed | 0.75 to 0.97 of the limit | 0.85 to 0.97 |
| Time headway / standstill gap | 1.0 to 2.2 s / 1.5 to 3 m | 1.2 to 1.8 s / 2 m |
| Acceleration / comfortable braking | 1.0 to 2.0 / 1.5 to 3.0 m/s$^2$ | 1.2 to 1.8 / 2.0 m/s$^2$ |
| Position in the lane | centre $\pm 0.25$ m | centred |
| Turn path and speed | arc up to 0.4 m tighter or wider, lateral acceleration 2 to 3 m/s$^2$ | 2.5 m/s$^2$ |

| Per event | Range | Current rule |
|---|---|---|
| Start-up delay when the light turns green or the leader moves | 0.3 to 1.5 s | none |
| Yellow: reaction before the stop-or-go choice | 0.3 to 0.7 s | none |
| Stop position | 0.5 to 3 m before the line or behind the leader | fixed |
| Stop sign: standing time, creep before going | at least 1 s, creep of up to 1.5 m behind the stop line | fixed |
| Indicator onset before a turn | 20 to 45 m before the line | 30 m |
| Lane change or merge: indicator lead / manoeuvre length | 1.5 to 4 s / 2.5 to 5 s | 3 s manoeuvre |
| Pedestrians: walking speed, pause at the curb, start delay | per person | speed and delay drawn |

Speed also drifts by about $\pm 0.3$ m/s around the desired speed (capped below the limit plus the R0 tolerance), and the position in the lane drifts slowly within $\pm 0.1$ m.

## After frame 31

The ego's light changes only out of a yellow it showed by frame 31. Any other choice after frame 31 that affects the ego, by a pedestrian, another driver or the ego itself, needs its cue on screen by frame 31; the ego's own major choices then fall in frames 32 to 40 and occur only in pending clips. `diversity.cue_problems` checks this.

## Future sets

Each sample gets 16 futures of frames 32 to 63, forked at frame 31 and stored as states in `future_set.npz`, with future 0 as its own label. A fork draws again what the video has not shown (pending choices, per-event values, other road users' choices, new arrivals) and keeps the per-driver values, the choices made on screen and the held light. The sets check the revision, offer the encoder a set target, and give val and test several valid futures to score against.

## Labels and checks

Each sample stores, per major choice, its frame, chooser, cue value, drawn probability and outcome, and each agent's style and per-driver values. The checks are:

- shares of A to D against their targets, and E against its aims;
- the share of pending clips with at least two outcomes among their 16 forks, which must reach 70% for the pilot to pass, and the ego's spread at frame 63 in decided and none clips;
- realised outcome shares against the drawn odds on the forks, binned by cue value;
- the share of test sibling pairs whose ego trajectories differ in frames 5 to 31;
- no `rules.check` violations and exact re-simulation (val).

## Other changes

Interactions happen close to the ego, so the car involved is large on screen.

| Event | Change |
|---|---|
| Following | the speed label "follow" needs the leader within 7 m; the causes "lead vehicle" and "parked obstacle" within 10 m; "cut-in vehicle" within a 2 s time gap |
| Overtake | the ego follows a slow vehicle (13 to 19 km/h) within 10 m, then pulls out from behind it, so its cause is "lead vehicle" |
| Cut-in | a car 5 to 18% faster than the ego signals and moves over 0.6 to 1.2 s ahead of it, while R7 still limits the braking it forces on the ego |

## Build

Pending clips come from nine recipe variants, one per kind of choice: yellow onset, permissive left, right turn on red, stop-sign near tie, lane-end merge, cut-in, overtake, pedestrian at the curb and parked car. Each sets the cue up so the choice falls in frames 32 to 40 (`scripts/v2s_combos.py` reports, per target combination, how often it comes out as planned).

The 500-clip pilot passed: 95.5% of its pending clips show both outcomes among their 16 futures, the realised outcome shares match the drawn odds in every bin, and every clip re-simulates exactly. Train (10,000), val (1,500) and test (1,500 clips, two futures each) are built into `/root/projects/drive_v2s/` by `run_all.sh` there, which then runs the checks, writes the GIFs and uploads the zips. The train split keeps its frames in `render/` only (no `videos.npy`).

- `scripts/v2s_checks.py <split> [--resim]`: the checks above, written to `<split>/checks.json`.
- `scripts/v2s_viz.py <split> --out <dir>`: `diversity_<aspect>_v2s.gif` (A to E, one clip per label), `pending_v2s.gif` (one pending clip per kind of choice) and `forks_v2s.gif` (four futures of a pending clip, forked at frame 31).
- `scripts/v2s_zip.py <folder> <zip>`: the data and code archives.

### Built data

In val, 94.8% of the pending samples show both outcomes of their pending choice among their 16 futures, and the realised outcome shares follow the drawn odds (0.1-0.3: 0.18 drawn, 0.19 realised; 0.3-0.5: 0.38 drawn, 0.36 realised; 0.5-0.7: 0.60 drawn, 0.54 realised; 0.7-0.9: 0.81 drawn, 0.78 realised). The ego ends more than 2 m apart across the futures at frame 63 in 66.6% of pending samples, 1.7% of decided and 2.6% of none. 1,500 of 1,500 val samples re-simulate exactly from their seed and target, and `rules.check` reports 0 violations. In test, 2.9% of sibling pairs keep the ego more than 2 m apart somewhere in frames 5 to 31.

![One pending val clip per kind of choice; the map runs on to frame 40](pending_v2s.gif)

![Four futures of pending val clips, forked at frame 31](forks_v2s.gif)

# DriveToy v3

DriveToy v3 is built by this code with `DT_V3` on (`V3_OVERRIDES` in `config.py`); [version-3-revision.md](version-3-revision.md) lists the issues it revises and [task_encoder_eval.md](task_encoder_eval.md) the evaluation its new labels serve. Without `DT_V3` the code runs exactly as before, so v2s samples still re-simulate from their seeds.

## Data (v3)

The splits are in `/root/projects/drive_v3/` on yt-workspace and in `drive_v3_data.zip` in the shared Google Drive folder, next to the code in `drivetoy_v3_code.zip`.

| Split | Clips | Samples |
|---|---:|---:|
| train | 10,000 | 10,000 |
| val | 1,500 | 1,500 |
| test | 1,500 | 3,000 |

Samples keep the v2s format: 32 visible frames at $128 \times 128$ and 5 Hz, then 32 label-only frames, and 16 futures of frames 32 to 63 per sample, forked at frame 31. Each split holds `annotations.npz`, `future_set.npz`, `meta.json`, `checks.json` and the per-sample `labels/`, `scenes/`, `maps/` and `render/` files; val and test also hold `videos.npy`, and train keeps its frames in `render/<n>.npz` only. The GIFs are in `viz/`.

The new labels are in `annotations.npz` and `labels/<n>.npz`, and the per-future ones also in `future_set.npz`.

| Label | Shape | Meaning |
|---|---|---|
| `maps/<n>.json.gz` | one file per sample | the map (lane graph, signals, signs, crosswalks, school zone), read with `map_io.load_map`; parked cars standing in a lane are vehicles in `agents`, not part of the map |
| `may_go` | (64,), futures (16, 32) | 1 if the ego may lawfully go at the stop line of its route in that frame, 0 if not, $-1$ without a junction ahead |
| `may_go_reason` | (64,), futures (16, 32) | why it may not go, as an index into `meta["may_go_reasons"]`: none, light, oncoming, pedestrian, conflict, stop order |
| `ego_turn` | (), futures (16,) | the ego's move at the junction: 0 none, 1 left, 2 straight, 3 right |
| `route_drawn` | () | whether the 16 futures drew their routes (Issue 6) |

## Changes (v3)

| Issue | What v3 does |
|---|---|
| 1. Leads | Cars draw a desired speed of at least 60% of the limit, and trucks and buses 65 to 85%. The speed label "follow" needs the leader within a 2 s time gap, and the cause "lead vehicle" within $\max(10\,\text{m},\ 2\,\text{s} \cdot v)$. The ego overtakes a lead slower than 0.85 of its own desired speed once it is that close. |
| 2. Go choices | A "go" drawn at the stop line is carried out unless a vehicle enters the conflict area or can no longer stop before it at $3\,\text{m/s}^2$; a committed left turn also gives up while an oncoming car is under 2.8 s away. A car given its turn at a four-way stop keeps it. Stops last 0.8 s and start delays 0.2 to 0.6 s. |
| 3. Oncoming gaps | Cars on the lanes a left-turner yields to keep time headways of at least 2 s (mean 4 to 8 s per scene), and arrival times allow for acceleration. |
| 4. Left-only lane | About 35% of the traffic at four-way junctions uses the left-only lane, half the signal maps have 6 to 8 s protected left arrows, and the solid line covers only the last 30 m before the junction. |
| 5. Trees | Trees whose crown would hide a stop sign or signal head from the approach are left out. |
| 6. Routes | When the ego has not signalled by frame 31 and its lane allows several moves, each future draws its move with equal odds. |
| 7. Labels | `maps`, `may_go`, `may_go_reason`, `ego_turn` and `route_drawn`, above. |
| 8. School zone | One red surface covers both lanes over the whole zone, and a blue triangle (2.0 m wide, 4 m long) marks its start. |

## Checks (v3)

`scripts/v3_checks.py <split> [--resim]` runs the v2s checks and adds a `v3` section per issue to `<split>/checks.json`.

| Check | Train | Val | Test |
|---|---:|---:|---:|
| Pending samples (aspect E) | 20.1% | 20.3% | 19.0% |
| Pending samples with both outcomes in their 16 futures | 94.6% | 94.1% | 95.8% |
| Vehicles outside their speed range | 0 | 0 | 0 |
| Committed left turns carried out | 626 / 626 | 96 / 96 | 180 / 180 |
| Committed right turns on red carried out | 477 / 479 | 81 / 81 | 143 / 143 |
| Stop-sign ties resolved in arrival order | 254 / 259 | 42 / 42 | 88 / 88 |
| Standing time at a clear stop sign, mean (max) | 1.3 s (1.6 s) | 1.3 s (1.6 s) | 1.3 s (1.6 s) |
| Shortest oncoming gap (share under 3 s) | 2.6 s (0.3%) | 2.8 s (0.4%) | 2.8 s (0.8%) |
| Ego finishes its left turn after yielding | 751 / 926 | 112 / 134 | 233 / 276 |
| Left-only lane share | 37.7% | 37.4% | 38.0% |
| Trees left out per clip | 4.7 | 4.7 | 4.7 |
| Samples whose futures draw routes | 2,297 | 365 | 728 |
| Future moves right / straight / left | 44.3 / 44.2 / 11.5% | 44.7 / 43.9 / 11.5% | 43.9 / 44.5 / 11.6% |
| Indicator before frame 32 in those samples | 0 | 0 | 0 |

All 1,500 val samples re-simulate exactly and `rules.check` reports 0 violations; the builder rejects any candidate whose record breaks a rule, so train and test hold none either. The logged ego stands at its stop line while `may_go` is 1 in 43 train, 10 val and 18 test samples; the examples in `checks.json` last 1 to 4 frames (0.2 to 0.8 s), so the "didn't go" rule should allow about 1 s before it counts.

![One pending val clip per kind of choice](pending_v3.gif)

![Four futures of pending val clips, forked at frame 31](forks_v3.gif)

One val clip per label of each aspect: [A](diversity_A_light_v3.gif), [B](diversity_B_cause_v3.gif), [C](diversity_C_speed_v3.gif), [D](diversity_D_path_v3.gif), [E](diversity_E_choice_v3.gif).

## Scripts and tests (v3)

- `scripts/build_drivetoy_v3.py --split <train|val|test> --gpus 0,1,2,3` builds a split into `/root/projects/drive_v3/<split>` and resumes where it stopped; `drive_v3/run_all.sh` builds all three, then runs the checks and writes the GIFs.
- `scripts/v2s_viz.py <split> --out <dir> --suffix v3` writes the GIFs, and `scripts/v2s_zip.py <folder> <zip>` the archives.
- The tests run as plain scripts (`python3 tests/test_v3_leads.py`); `tests/test_v3_regression.py` needs `drive_v2s/val` from `drive_v2s_data.zip` and skips without it.
