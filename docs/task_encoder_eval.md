# Task-Encoder Trajectory Evaluation

The task encoders' planned trajectories are scored on DriveToy with five terms of NAVSIM v2's EPDMS and one DriveToy term, traffic rule compliance (TRC), which replaces its traffic light compliance.

## ADE Evaluation

Two numbers summarize each encoder's path over the full 6.4 s (frames 32 to 63). Both treat all 16 futures as acceptable and compare positions in the ego frame at frame 31.

**1. Top-1 ADE to the nearest future.** The encoder's most probable path is where it says the ego should go, so it is compared with the closest of the 16 futures $f_1, \dots, f_{16}$ and averaged over clips:

$$\text{ADE}_\text{near} = \min_{j} \frac{1}{32} \sum_{t=32}^{63} \lVert \hat{p}_t - f_{j,t} \rVert$$

A path that follows any one future scores 0, while a path between futures that split matches none of them and scores poorly.

**2. Distribution match over distinct outcomes.** A vocabulary encoder's probabilities over its $K$ paths should put noticeable mass on every distinct outcome among the 16 futures and little mass anywhere else; how the mass is balanced among the outcomes does not matter. This follows precision and recall for generative models (Sajjadi et al., 2018; Kynkäänniemi et al., 2019).

- The 16 futures of a clip are clustered with complete linkage at 2 m ADE into $G$ outcomes, so speed variants of one path form one outcome, while going straight and turning right form two.
- Each vocabulary path's probability goes to the outcome of its nearest future if that future is within 2 m ADE, giving a mass $m_g$ per outcome; the remaining mass $m_\text{off}$ lies on paths that no future follows.
- Precision is $1 - m_\text{off}$, reported on all clips.
- Recall is $\frac{1}{G} \sum_{g} \min(m_g / \tau,\ 1)$ with $\tau = 0.1$, so an outcome with at least 0.1 of the mass gets full credit. It is reported on clips with two or more outcomes, 163 of the 1,500 `drive_v2s` val clips, with $\tau = 0.05$ alongside as a check.

## NAVSIM-style Evaluation

Each plan covers the 4 s after frame 31 (frames 32 to 51). As in NAVSIM, it is interpolated to 10 Hz and tracked by NAVSIM's LQR controller and kinematic bicycle model, and every term reads the tracked states.

$$\text{score} = \text{NC} \cdot \text{DAC} \cdot \text{TRC} \cdot \frac{5\,\text{TTC} + 2\,\text{HC} + 2\,\text{EC}}{9}$$

| Term | What it checks | Range |
|---|---|---|
| NC | An at-fault collision: the ego's front hits a moving agent, the ego hits a stopped one, or a side collision while the ego is across two lanes or off the road. | 0 for a vehicle or pedestrian, 1/2 for a static object, otherwise 1 |
| DAC | A corner of the ego box leaves the road surface. | $\{0, 1\}$ |
| TRC | The plan breaks any of the nine rules below. | $\{0, 1\}$ |
| TTC | The ego, projected at constant speed and heading up to 1 s ahead, would hit an agent ahead of it. | $\{0, 1\}$ |
| HC | The logged frames 24 to 31 followed by the plan exceed NAVSIM's comfort limits. | $\{0, 1\}$ |
| EC | The plan differs from the encoder's plan made at frame 29 beyond NAVSIM's limits over their shared frames 32 to 49. | $\{0, 1\}$ |

NC, DAC, TTC, HC and EC are computed with NAVSIM's own code, from inputs built out of the sample's `map` label.

## Traffic Rule Compliance

TRC is 1 only if the plan breaks none of these rules. The first eight are DriveToy's `rules.check`; the last needs the `may_go` label.

| Rule broken | Check | Source |
|---|---|---|
| Speeding | speed above the lane's limit at that point plus 0.5 m/s, including the 30 km/h school zone | R0 |
| Ran a red light | crosses the stop line on red; a green arrow allows the left turn, and a right turn on red is allowed after a full stop | R1 |
| Went on yellow when it could have stopped | crosses on yellow although it could have stopped at $\le 2$ m/s$^2$ | R1 |
| Turned left in front of oncoming traffic | an oncoming vehicle is less than 2.5 s from the conflict point when the ego crosses the line | R3 |
| No full stop at a stop sign | stands still ($v < 0.3$ m/s) for fewer than 4 consecutive frames within 6 m of the line | R4 |
| Went before a car that arrived first at a stop sign | enters while a vehicle from another approach that stopped first is still waiting | R4 |
| Didn't yield to a pedestrian on the crosswalk | enters the crosswalk while a pedestrian is crossing within 3 m of its path | R5 |
| Followed too closely | gap to the leader below $\max(1\ \text{m},\ 0.5\ \text{s} \cdot v)$ | R6 |
| Didn't go when the way was clear | stays below 0.3 m/s at or before its stop line for at least 1 s while `may_go` is true and no stopped vehicle stands within 10 m ahead in its lane; at a stop sign the 1 s starts when the full stop ends, and frames are skipped while a car that stopped before the plan still waits | new |

Collisions are left to NC, which counts only at-fault ones.

## Plans

- The encoders with a 32-frame (6.4 s) head are scored: `g16_query`, `causal_global`, `ssl` (global head) and the frozen released V-JEPA 2. The local heads plan only 8 frames (1.6 s) and are left out.
- Each encoder predicts four modes with probabilities. The plan is the most probable mode, cut to its first 20 frames; the probability-weighted mean of the four modes' scores is reported beside it.
- For EC, each encoder also plans from frames 0 to 29, 0.4 s before frame 31 and the nearest frame to NAVSIM's 0.5 s. Encoders trained only on 32-frame inputs get frame 0 repeated twice at the start.

## Futures and the Human Filter

Other road users replay their logged motion, as in NAVSIM's first stage. A plan is scored against each of the sample's 16 futures, and its score is the mean over them, since all 16 continue frames 0 to 31 equally well. Within each future, a term that the future's own ego fails after the same conversion scores 1 for the plan, which is NAVSIM's human filter.

## References

- `gt_self`: each future's own ego against its own future. It scores 1 by the filter, and its unfiltered terms measure the errors of the conversion itself.
- `gt_rec`: the sample's logged future against all 16 futures. It measures the cost of replaying road users that reacted to a different ego.
- `cv`: constant speed and heading from frame 31.
- `stop`: braking at 3 m/s$^2$ from frame 31 to a standstill.

## Report

For each encoder and reference: the mean score, the mean of each term, and the share of samples that break each TRC rule, overall and by cause (aspect B), choice state (aspect E) and template. The two trajectory headlines are reported overall and by light (aspect A) and cause (aspect B), the second only for vocabulary encoders.

## Inputs and Code

The scorer reads each sample's `map` and `may_go` labels (`version-3-revision.md`, Issue 7). On `drive_v2s` val, `scripts/task_encoder_eval/rule_check/rules_eval.py` in Task_specific_JDM_waypoint already re-simulates each clip for its map and 16 futures, matches the predictions onto the lane graph and runs `rules.check`. Every term except "didn't go when the way was clear" can therefore be computed there before drivetoy_v3 is built. NAVSIM's scorer runs in its own CPU conda env.
