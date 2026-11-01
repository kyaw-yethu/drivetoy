# Constraints

A DriveToy sample is a solution to a constraint satisfaction problem over its scene, its 32 visible frames (0 to 31) and its 16 futures (frames 32 to 63). Hard constraints hold in every sample and every future, and soft constraints are aims that the sampler steers toward.

## Hard constraints

### Format

**H1. Clip.** Each sample has 32 visible frames at $128 \times 128$ and 5 Hz (6.4 s), followed by 32 label-only frames. A test clip has two branches that share frames 0 to 4, so it gives two samples.

**H2. Futures.** Each sample has 16 futures of frames 32 to 63, forked at frame 31 with fresh draws, and future 0 is its own label.

**H3. Camera.** The front camera sits 1.3 to 1.5 m high, with a $70^\circ$ field of view and a pitch of $-6^\circ$ to $-2^\circ$.

**H4. Reproducible.** A sample and its futures re-simulate exactly from the stored seed and target.

### Traffic rules

These hold for every road user in the sample and its futures. Each rule has a policy that agents follow and a looser check that `rules.check` counts, so normal driving never sits on a threshold.

**H5. Speed (R0).** No vehicle drives more than 0.5 m/s above the limit at its position, the 30 km/h school zone included.

**H6. Lights (R1).** No vehicle crosses its stop line on red, except to turn left on a green arrow or to turn right after a full stop. No vehicle crosses on yellow if it could have stopped at 2 m/s$^2$ (policy 3 m/s$^2$).

**H7. Permissive left (R3).** A left turn on a green ball crosses its line only while every moving oncoming vehicle is at least 2.5 s from the conflict point (agents keep at least 3 s).

**H8. Stop sign (R4).** Every vehicle stands below 0.3 m/s for at least 0.8 s within 5 m of its line. None enters before a vehicle on another approach that stopped more than 0.4 s earlier and is free to go.

**H9. Crosswalks (R5).** No vehicle enters a crosswalk while a pedestrian crosses within 3 m of its path.

**H10. Headway (R6).** No vehicle follows closer than $\max(1\,\text{m},\ 0.5\,\text{s} \cdot v)$.

**H11. Lane changes (R7).** A lane change shows its indicator for at least 1 s first (agents use 1.5 to 4 s) and never crosses a solid line. It never forces the new follower to brake harder than 4 m/s$^2$ (policy 2 m/s$^2$).

**H12. Pedestrians (P1, P2).** A pedestrian starts across a signalised crosswalk only on WALK. It never steps in front of a vehicle that would need more than 4 m/s$^2$ to stop (policy 3 m/s$^2$).

**H13. Junction priority.** Inside the junction, a vehicle that entered first keeps priority until it clears the conflict area. On equal entry times, the left-turner yields.

**H14. Stop-sign turn.** A vehicle held at its line by a pedestrian or by a vehicle in its path is not free to go. It keeps no turn over vehicles whose paths do not conflict with its own.

### Randomness

**H15. Where it enters.** Randomness enters only where real drivers differ. Stopping on red, the response to a yellow, the full stop at a stop sign, yielding to a pedestrian on the crosswalk and whether the ego goes around a parked car are never random.

**H16. Drawn choices.** A gap choice (left turn, right turn on red, overtake, merge, going round a parked car before or after an oncoming car, cut-in, a pedestrian crossing) takes its assertive outcome with $p = \sigma\big((c - c_{50}')/w\big)$, where $c$ is a cue the video shows and $c_{50}'$ shifts with the driver's style. Routes and stop-sign options are drawn with equal odds among the open ones.

**H17. Per-driver and per-event values.** Per-driver values (desired speed, headway, acceleration, lane position) are drawn once and lean toward the driver's style, and per-event values (start delay, stop position, indicator lead) are drawn once per event. All ranges, set in `config.py`, stay inside the rule checks, and per-frame noise stays small.

**H18. One model for all.** Every road user follows the same rules and choice models as the ego.

### The ego

**H19. Labelled response.** The ego carries out the response that its light label names within frames 0 to 31, except in pending samples. Its episode falls in frames 5 to 31.

**H20. Route.** The ego's route and indicator are fixed before frame 0. Where its lane allows several moves and it has not begun its move by frame 31, each future draws the move, and the indicator comes on only after frame 31.

**H21. No wait without a reason.** The ego stands at or before its stop line only for its light, its full stop and turn at a stop sign, a moving vehicle with priority, a pedestrian or a vehicle ahead. Each wait ends, after a start delay of 0.2 to 0.6 s, as soon as its reason ends.

**H22. Didn't go.** In the sample and in every future, the ego passes the "didn't go when the way was clear" rule of [task_encoder_eval.md](task_encoder_eval.md). Letting the others go at a stop sign (H39) counts as waiting for them.

**H23. Oncoming and crossing traffic.** Oncoming and crossing traffic are the vehicles actually coming across the ego's path: moving toward the conflict and not about to stop before it for a light, a stop sign or a queue. A standing vehicle is never one of them, although one standing in the ego's path still blocks it.

**H24. Go is carried out.** A go is drawn only when the vehicle can start at once, with its exit clear and after its start delay, and is then carried out. It is dropped only if a vehicle enters the conflict area or can no longer stop before it at 3 m/s$^2$, or if an oncoming car comes within 2.8 s of a left-turner still short of its line.

**H25. Declined gaps.** A declined gap lasts only until the vehicle that bounds it passes or stops. The next gap is a new choice.

**H26. Yellow.** The ego goes on yellow only if stopping would need more than 3 m/s$^2$ and it reaches its line before red, and otherwise it stops. Every future gives the same response.

**H27. Parked cars.** The ego always goes around a parked car in its lane, in front of a coming oncoming car only with its actual passing time to spare. With under 4 s to spare once it is within 10 m of the parked car, going round before that car or after it is its choice.

**H28. Overtake.** The ego pulls out only after following a slower leader within $\max(10\,\text{m},\ 2\,\text{s} \cdot v)$ for its patience, at least 1 m/s faster up to the move. It chooses when the move can start, after the indicator lead: pulling out moves over at once, and staying behind turns the indicator off.

**H29. Lane end.** The ego signals for a lane-end merge only once the lane-ends sign or the cone taper covers at least 25 px of its view. It then merges either ahead of or behind the car alongside in the target lane.

**H30. Reasons for lane changes.** A lane change is an overtake of a slower leader, a merge out of a lane that ends, or a cut-in.

### Other road users

**H31. Acts by frame 31.** A pedestrian the ego yields to after frame 31 has stepped onto the crosswalk by frame 31, and a car that cuts or merges in close ahead of the ego (where a cut-in binds) after frame 31 has started moving over by then. Other road users' choices that concern the ego are made by frame 31.

**H32. Leaders.** No vehicle targets less than 60% of the limit unless something in the scene slows it. Leaders drive at 75 to 97% of the limit, trucks and buses at 65 to 85%, and an overtaken leader at 60 to 75%.

**H33. Cut-ins.** A cut-in car is 5 to 18% faster than the ego and moves over 0.6 to 1.2 s ahead of it. It brakes the ego no harder than R7 allows.

**H34. Oncoming headways.** Oncoming cars keep time headways of at least 2 s, plus an exponential tail whose mean is 4 to 8 s per clip.

### Pending samples

**H35. Arrival.** A pending choice arises because the ego reaches its decision point (its stop line, the end of its full stop, its merge point, the end of its patience behind a slow leader or a parked car ahead) under normal driving around frame 31, so the choice falls in frames 32 to 40. No vehicle or signal phase is placed only to hold the ego there earlier.

**H36. Yellow.** In a pending yellow clip the light turns yellow late, for example at frames 27 to 31, while the ego drives toward it from positions spread over the approach and goes straight in every future. In a decided yellow clip it turns earlier, so the ego's stop or go shows on screen.

**H37. Left turn and right on red.** The ego reaches its line on a green ball, or ends its full stop first in line at a red, around frame 31. Moving oncoming or cross traffic then offers a next gap within the choice's random range.

**H38. Overtake and merge.** For an overtake, the ego's patience behind the slow leader runs out around frame 31. For a merge, the ego reaches the point where it picks its gap around frame 31, with a car alongside in the target lane.

**H39. Stop sign.** The ego and at least one other car end their full stops within 0.4 s of each other around frame 31, and the ego shows no indicator by frame 31. Its options are letting the other cars go, going straight, turning left and turning right, and each future draws one of those still open at frame 31.

**H40. Open options.** Letting the other cars go is always open while the ego stands; a move closes if its path crosses a car that visibly moves by frame 31 or that stopped more than 0.4 s before the ego (it goes first by R4), or if a pedestrian is on its crosswalk. With nobody moving all four are open, a car from the left going straight leaves only letting it go, and an oncoming car going straight leaves letting it go, going straight and turning right.

**H60. Others follow the option.** In a future where the ego goes first, the cars tied with it that have not moved wait for it, and in one where it lets them go, they go. Other road users never take an option the future did not draw for the ego.

**H61. Pending parked car.** The ego closes up behind a parked car in its lane around frame 31 while an oncoming car comes past it, on a street without a school zone. Each future draws whether it goes around before the next oncoming car or after it.

**H62. Parked car, clear road.** With nobody coming the other way, the ego reaching a parked car in its lane around frame 31 goes round it without closing up. It starts pulling out by about frame 28, so every future goes round alike.

**H41. Distinct outcomes.** Different outcomes of a pending choice give ego trajectories at least 2 m apart by frame 63.

**H42. The ego's choice.** A sample's pending choice is the ego's own, and its route and stop-sign options count as its own.

### Scene and rendering

**H43. Cues by frame 31.** Every choice the ego makes after frame 31 has its cue on screen by frame 31. The ego's light changes after frame 31 only from a yellow already shown at frame 31 to red.

**H44. Signal heads.** Vehicle signal heads are drawn at $4.5\times$ real size, the ego's heads stand 4.5 m nearer, and each approach shows only its two far-side overhead heads, none on the ego's near side. Lamp colours keep green apart from yellow after VAE encoding.

**H45. Lamps.** Indicator lamps are drawn at $2.25\times$ real size, and brake lamps and pedestrian signal heads at $1.5\times$.

**H46. Trees.** No tree crown stands between an approaching car and a sign or signal head from 10 to 60 m before it.

**H47. Lane end marking.** A lane end has a cone taper, two 4 m lane-ends signs 100 m and 45 m before the end, and merge arrows painted 60 and 30 m before it.

**H48. Left-only lane marking.** The line between a left-only lane and its neighbour is solid for the last 30 m before the junction.

**H49. School zone.** Red pavement covers both lanes along the whole 30 km/h zone, and a blue triangle 2.0 m wide and 4 m long is painted where it begins.

**H50. Sun.** At dusk, the sun stays at least $60^\circ$ away from the camera's heading.

### Labels

**H51. Windows.** Each sample has one label per aspect: A to D from frames 0 to 31 (B from frames 5 to 31), and E from the ego's choices in frames 0 to 40.

**H52. A, light.** A names the light that governs the ego, and the built shares of A match their targets exactly.

**H53. B, cause.** "Lead vehicle" binds within $\max(10\,\text{m},\ 2\,\text{s} \cdot v)$, "cut-in vehicle" within a 2 s time gap, and "crossing traffic" only while the other car is roughly perpendicular, $|\cos(\theta - \theta_\text{ego})| < 0.5$. "Speed limit" counts only while the school zone holds the ego below the normal limit, and steering around a parked car counts as "parked obstacle".

**H54. C, speed.** "Follow" needs a leader that limits the ego at a bumper gap of at most 2 s for at least half the clip. "Slow down" needs a drop of more than 3 m/s without a stop.

**H55. D, path.** "Straight" turns less than $15^\circ$ and a turn more than $45^\circ$. A lane change shows its indicator before it moves.

**H56. E, choice.** "Pending" means a major choice or response of the ego, cued on screen by frame 31, falls in frames 32 to 40, with odds between 0.1 and 0.9 when it is drawn. "Decided" means it was made on screen, and "none" means the clip has no major choice.

**H57. Stored labels.** Each sample stores its map, its decision log (frame, chooser, cue, odds, outcome) and every agent's style and per-driver values. It also stores `may_go`, `may_go_reason` and `ego_turn` for the sample and each of its futures.

**H58. `may_go`.** `may_go` is 1 when an ego standing at its stop line could lawfully enter, 0 when it could not and $-1$ with no junction ahead, and `may_go_reason` names the cause. Both come from the simulator's own yield functions.

### Build

**H59. Acceptance.** A sample is kept only if it and all its futures pass `rules.check`, keep the ego on the map and meet the visibility constraints. No sample is kept or rejected on how its frames 32 to 63 turn out beyond that, so drawn odds stay unbiased.

## Soft constraints

**S1. Label shares.** B, C and D are steered toward their target shares in `DT_SHARES`, B first. E aims at 40% pending (at least 20%), 45% decided and 15% none.

**S2. Cue spread.** The cues of drawn pending choices spread over the whole random range.

**S3. Outcomes.** A pending sample with a drawn choice shows at least two outcomes among its 16 futures, and the outcome shares follow the drawn odds.

**S4. Test siblings.** The two branches of a test clip give ego trajectories that differ in frames 5 to 31.

**S5. Close interactions.** Interactions happen close to the ego, so the vehicle involved is large on screen.

**S6. Left-only lane.** About 35% of a signal approach's traffic uses its left-only lane, and half the signal maps have a 6 to 8 s green arrow before the circular green.

**S7. Waved through.** When the ego lets another car go first, it starts its own move before frame 63.
