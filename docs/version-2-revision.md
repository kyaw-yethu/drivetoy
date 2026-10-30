# Revision History

## Issues in Version 1

**1. The visible frames do not always determine the labels in frames 32 to 63.** Randomness in the generated frames is fine, but a hard label that frames 0 to 31 cannot predict is label noise for the task encoder.

- In 25% of `signal4` clips (about 900), the ego's own light changes in frames 32 to 63 while the ego is still before its stop line, with no yellow at frame 31. The only possible cue is a pedestrian countdown of about 28 px, which does not read at $128 \times 128$.
- In 139 clips the ego stops in frames 32 to 63 for a pedestrian who is still approaching the curb, undecided, at frame 31.
- Turn and lane-change intent is visible by frame 31 in the simulator state (the indicator is on), but only as a lamp of about 7 px.

**2. The ego rarely carries out the rules itself.** The ego's behaviour is what makes clips differ, yet most rules show only through other cars. Counting episodes that start and finish within frames 0 to 31 (definitions in `analysis/rule_episodes_v1.py`):

| Episode | Ego only | Ego or a visible car |
|---|---|---|
| left turn after yielding to oncoming traffic | 0 | 228 |
| goes on green after stopping | 0 | 435 |
| stops at a red light | 13 | 1,588 |
| lane change | 38 | 232 |
| merge | 24 | 382 |
| stop sign | 245 | 646 |
| yields to a pedestrian | 489 | 1,946 |
| follows a leader | 2,870 | 3,878 |

Only 2.4% of clips show the ego completing a turn on screen. The cause is ego placement: for example, the ego starts 38 to 52 m before a signal's stop line, so it reaches the junction around the end of the clip.

**3. Traffic lights cannot be read at $128 \times 128$.** All vehicle signal heads hang on the far side of the junction (`roads.py`), about 70 to 80 m from the ego's start and still about 30 m from the stop line. With the $70^\circ$ field of view, one lamp spans about 0.5 px at the start and about 1 px at the stop line, even at the current $1.5\times$ scale. The v1 VAE (`vae_100e18b`) loses most of the signal colour as a result.

# Issues Fixed in Version 2

The first v2 build had seven further problems, found in the per-aspect videos. Each is fixed in the released data.

### 1. The lane end is hard to see before the merge

**Issue.** The merging ego started 90 to 110 m before the end of its lane, beyond the view range of the cone taper, and the small lane-ends sign (0.8 m) was already beside or behind it at frame 0.

**Fix.** The ego starts 60 to 80 m before the end, and the sign is 1.6 m tall and stands 45 m before the end, so it is visible from frame 0.

### 2. Lane changes without a reason

**Issue.** Lane changes with cause "none" or "speed limit" were forced: the indicator was the only cue, and nothing in the scene motivated the move.

**Fix.** A lane change is built only as an overtake of a slower leader (cause "lead vehicle").

### 3. Stop-sign waits counted as crossing traffic

**Issue.** Early-arriving cars were also placed on the opposite approach, and any stop-sign wait for another vehicle, head-on ones included, was labelled "crossing traffic".

**Fix.** Early arrivals go only on the left and right approaches, and a wait counts as "crossing traffic" only while the other vehicle is roughly perpendicular to the ego, $|\cos(\theta - \theta_{\text{ego}})| < 0.5$.

### 4. "Speed limit" and "nudge around" labels that do not match the scene

**Issue.** "Speed limit" marked plain cruising near the normal limit, and steering around a parked car counted as free driving, so no nudge clip was labelled "parked obstacle".

**Fix.** "Speed limit" counts only while the 30 km/h school zone holds the ego below the normal limit, and its clips show the slowdown. Steering around a parked car counts as "parked obstacle".

### 5. Signal heads clutter the view

**Issue.** The near-side head filled the edge of the frame, and the heads of other approaches showed on the ego's near side, some lit beside the ego's own red.

**Fix.** Each approach has only its two vertical mast-arm heads on the far side, drawn at $4.5\times$. The ego's own heads sit 4.5 m nearer, and no vehicle head on the ego's near side is rendered.

### 6. Overtakes from behind a car that is not slower

**Issue.** In 46% of lane changes the ego was not faster than the car it overtook when it pulled out.

**Fix.** The leader is slower (45 to 60% of the limit) and nearer (20 to 32 m). A lane change is accepted only if the ego is at least 1 m/s faster in every frame up to the move and pulls out at most 2 s behind the leader.

### 7. "Follow" counts leaders at any distance

**Issue.** The simulator names any vehicle ahead in the lane as the leader, even 165 m away, so 23% of "follow" frames had the leader more than 3 s ahead.

**Fix.** A frame counts as following only while the gap is at most 10 m or 3 s at the ego's speed; otherwise it is free driving.