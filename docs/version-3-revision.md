## Issues in Version 2, Revised in Version 3

These issues were found on 2026-10-28 and 2026-10-29 by reviewing the decision-probe labels of the `drive_v2s` val split (1,500 clips) scene by scene. Version 3 (`drivetoy_v3`) is built as a new dataset, and `drive_v2s` stays as it is. The decision labels, probes and task encoders are redone on the new data.

### 1. Lead vehicles crawl so that the ego can "follow"

**Issue.** The speed label "follow" needs the leader within 7 m (`DT_FOLLOW_GAP`) for at least half the frames. At normal urban speeds that is tailgating (7 m at 40 km/h is 0.63 s), so the "lead vehicle" recipes of `signal4`, `multilane` and `residential` give the leader a fixed target speed of 9 to 14 km/h on 50 and 70 km/h roads, 13 to 29% of the limit, against 75 to 97% for every other driver. Nothing in the scene explains the crawl, yet the ego must crawl behind it. In 115 val clips a moving vehicle targets less than half the limit, and 9 of the 37 "green light, car waits" clips of the go-or-stop probe are such crawls (val 744 and 1208).

> **Fix.** "Follow" is defined by time gap: the leader limits the ego, with a bumper gap of at most 2 s at the ego's speed. Leaders drive at normal speeds, 75 to 97% of the limit for cars and 65 to 85% for trucks and buses. No vehicle targets less than 60% of the limit unless something in the scene slows it, such as a light, a turn, a pedestrian, a queue or a parked car.

### 2. A "go" choice at the stop line is drawn but not carried out

**Issue.** A choice is logged when it is drawn, but the simulator's checks run again at every step and can undo it. A left-turner at the line that draws "take the gap" yields again a few frames later, once the oncoming car's time to the conflict falls below the 3 s floor (`DT_GAP_FLOOR`). The go is latched only once the car can no longer stop comfortably, which a car pulling away from the line does not reach in time. Measured on the 16 futures of val, against whether the ego's front bumper crosses its stop line within 4 s of the choice:

| Choice | Logged "go" forks that cross | Correlation of the logged odds with the share of forks crossing |
|---|---|---|
| Left turn across traffic | 8 of 144 | $-0.11$ |
| Right turn on red | 215 of 341 | $0.61$ |
| Two cars arrive together at a stop sign | 92 of 192 | $0.01$ |
| Yellow light | 213 of 213 | $0.88$ |

The stop-sign row still needs a check of which car enters first, since waiting there also ends in crossing. The logged odds are the targets of the "genuine choices" probe, so these targets are wrong for left turns and partly wrong for right turns on red.

> **Fix.** Once a driver draws "go" for a left turn, a right turn on red or a stop-sign tie, it commits and goes. It cancels only if the gap has truly closed, that is, a vehicle is already inside the conflict area.

### 3. Oncoming streams leave no usable gap

**Issue.** The permissive left-turn recipe sends oncoming cars every 1.6 to 2.4 s until about frame 31, below the 3 s floor, and then a single gap of 3.2 to 4.8 s around the 4 s midpoint of the choice. With Issue 2 that gap is drawn but not taken, and background traffic keeps the turn waiting afterwards. Only 20% of the vehicles that yield to oncoming traffic at `signal4` complete their left turn within the 12.8 s clip. In val 959 the ego waits at a green light through all 64 frames in all 16 futures.

> **Fix.** Oncoming gaps are random, with a 2 s minimum plus an exponential tail whose mean is 4 to 8 s, so acceptable gaps appear at realistic rates.

### 4. The left-only lane at signals

**Issue.** Every `signal4` approach has two lanes for its full 200 m. The inner lane is left-only, the outer lane takes straight and right, and lane changes are forbidden along the whole approach, although the markings are dashed until 30 m before the line. Both lanes get the same inflow, so half of all approaching cars turn left. The green arrow lasts only 3 to 5 s, which clears one or two cars, so on 54% of the maps that have one it never lights up during a clip.

**Fix.**
- The line between the two lanes becomes solid near the intersection, so the markings match the ban on lane changes.
- The left-only lane gets about 35% of an approach's traffic.
- Signals keep one rule for every approach: a left is allowed on a circular green after yielding to oncoming traffic, and half of the maps add a 6 to 8 s green arrow before the circular green, shown by a fourth lamp on the signal head.
- Right turn on red stays as it is: a full stop, then go when clear, giving way to pedestrians. A right-turner queued behind straight traffic at red therefore waits for the green, as on a real shared lane.

### 5. Trees hide stop signs

**Issue.** Street trees are planted along the sidewalks every 14 to 26 m without regard to signs. In the 162 four-way-stop val clips, a tree crown leaves less than 30% of the ego's stop sign visible in 62 of the 3,295 frames with the sign in view (1.9%), spread over 9 clips (worst val 1117 and 537). Partial cover was not counted, and other signs and signal heads were not checked.

**Fix.** No tree is planted where its crown would stand between an approaching car and a sign or signal head, from 10 to 60 m before it.


### 6. The 16 futures share one route where the lane allows several

**Issue.** The ego's route and indicator are drawn before frame 0 (`scenarios.py`), and a fork keeps them, so all 16 futures take the same direction even where the ego's lane allows more than one. Only 7 of the 1,500 val clips have futures that differ in turn direction. The ego camera does not show the ego's own indicator, so frames 0 to 31 rarely reveal the route. In the straight-and-right lane of `signal4`, 317 val clips approach the junction without having started to turn: 220 go straight and 97 turn right, and the indicator is on by frame 31 in all 97. The soft-target task encoder puts 0.92 of its probability on straight paths in the right-turn clips, while their 16 futures all turn right, so the set target asks for a certainty the video cannot support.

> **Fix.** Where the ego's lane allows more than one direction and the ego has not begun to turn by frame 31, the route is drawn per future at the fork. This covers the shared straight-and-right lane at signals and the single lane of a four-way stop, which allows left, straight and right. The 16 futures split between the allowed directions in proportion to the turn weights, with at least one future for each direction, and the ego's indicator comes on only after frame 31.

### 7. Labels for scoring planned trajectories

**Issue.** The trajectory evaluation in `task_encoder_eval.md` needs the road map and a per-frame flag for when the ego may go, and neither is saved. The map exists only inside the simulator and is recovered by re-simulating each clip, about a minute per clip with its 16 futures. The logged `rule_state` cannot replace the flag, because it describes the logged ego at its own position, while a plan that stops short of the line or stays after the logged ego has left meets a different situation.

**Fix.** Two labels are added to each sample during labelling.
- `map`: the static road map. It holds the lane centerlines with their kind, turn, successors, neighbours, speed limit and school-zone intervals $(s_0, s_1, v_{\max})$; the solid-line intervals; the stop lines with their control (signal or stop sign), approach and signal group; the crosswalks; the conflict areas between lanes; the junction area; the road surfaces; and the parked cars and cones.
- `may_go` and `may_go_reason`, per frame from 31 to 51 in each of the 16 futures. `may_go` is true when an ego stopped at its stop line on its own route would be allowed to enter: the light is green for its turn (circular green or arrow), no oncoming vehicle is closer than the gap the most cautious driver accepts, no pedestrian is on or stepping onto its crosswalk, and no vehicle on a conflicting path is inside the junction. `may_go_reason` names the cause when it is false. Both come from the simulator's own yield functions.

### 8. The school zone is hard to see

**Issue.** The 30 km/h school zone of `residential` runs from 60 m before the crosswalk to 30 m after it, and is marked only by a 0.76 m pole sign with "30" on the right curb and 15 m of red pavement at its entrance. The ego starts braking for it about 34 m before the sign, when the sign is about 2 px wide in the 128 px, $70^\circ$ view and the red pavement about 1 px tall, so the "30" cannot be read. In 32 of the 262 `residential` val clips the ego is already inside the zone at frame 0, and nothing on screen shows the zone or where it ends.

**Fix.**
- The red pavement covers both lanes along the whole zone, so the zone shows in every frame inside it and its end shows where the red stops.
- A large triangle is painted on each lane where the zone begins for that direction.
