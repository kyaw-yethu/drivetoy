# DriveToy v4

## Purpose

DriveToy v4 trains and tests task encoders on decisions the ego has to take from what it sees. In v3 the ego usually started to carry out its decision before the last input frame, so an encoder could plan from the ego's own motion. In v4 every clip is built around a decision point, and the training clips place it so that windowed encoders meet it at the end of every window position.

## Decision point

A decision point is the frame $f_d$ at which the ego has to decide from the scene alone: its own motion gives nothing away, and it is close enough to the cue that the cue is clearly visible while a safe reaction is still possible. Every clip with a decision meets four conditions.

1. The ego's speed and heading hold over frames $f_d - 7$ to $f_d$, changing by less than 0.1 m/s$^2$ and 0.02 rad/s (standing still counts). An earlier stop is allowed when it ends at least eight frames before $f_d$.
2. The ego's change starts on frame $f_d + 1$, or, in a carry-on clip, the ego keeps going because the cue does not call for a change. A dry run shifts the scene's timing until the change lands on that frame. A few all-way-stop clips start one frame later.
3. At $f_d$ the cue covers at least: lit signal lamps 12 px, a pedestrian 20 px, a stop sign 20 px, a parked car 30 px, and a lane-ends sign 60 px (all in the $128 \times 128$ view). The ego approaches signals at 20 to 23 km/h, since the heads stand on the far side of the junction.
4. Braking after $f_d$ is planned at 2.5 to 3.5 m/s$^2$ and never exceeds 4 m/s$^2$ within 4 s, and swerves stay below about 2 m/s$^2$ sideways.

The ego holds its speed until the cue is clear and then reacts firmly. It takes every opening its cue offers, at drive_v3's nominal thresholds: a 4 s gap for a left turn or a pull-out, and 5 s for a right turn on red. It waits only when traffic blocks it.

Whatever happens in frames 32 to 63 follows from frames 0 to 31. An ego that has stood goes straight or keeps standing, and a right turn on red appears only in its own situation, where the ego turns right after the decision. A carry-on ego at a red or waiting to turn left stays put to the end of the clip. The ego starts again late only where the cause shows by frame 31: a crossing pedestrian, a stop, a parked car or cars under way. It turns left across traffic only with the gap clear at the decision: within 60 m, about 5.5 s of oncoming traffic, no car comes toward it but the last one passing. It never slows for something out of its way.

## Splits

| Split | Clips | Decision frame | Futures |
|---|---|---|---|
| train | 10,000 | 7 (16%), 15 (17%), 23 (17%), 31 (23%), another frame from 1 to 30 (22%), none (5%) | one ground truth, frames 32 to 63 |
| val, test (train-like part) | 1,500 each | 31 (95%), none (5%) | one ground truth, plus 16 futures aside |
| val, test (twin and single part) | 1,900 each | 31 | one ground truth, plus 16 futures aside |

The decision frames are hard quotas. Within them the situations are dealt out by weight, which steers the light, cause, response and path labels (A to D) toward drive_v3's mix: about half the decision clips are at signalised junctions. With a decision at frame 31 in val and test, an encoder evaluated on fewer input frames uses the last $n$.

Over frames 0 to 31 the labels cannot all reach drive_v3's mix, because a decision at frame 15 or later leaves its turn, lane change or pedestrian stop to the frames after 31: in train the path label (D) is straight in 89% of clips, against 40% in drive_v3. Over all 64 frames the ground-truth paths in train are 66% straight, 24% turns, 6% lane changes or merges and 4% swerves round an obstacle.

A clip whose ego covers under 1 m from frame 31 to frame 63 is labelled wait instead of a move: 14% of train, mostly decisions early enough that the ego has stopped by frame 31, and 5% of val and test. A queue that stays at its red shows little beyond that label, so train holds only 50 such clips, and val and test 8 each; the twin splits keep their 100 pairs.

Each clip keeps one ground truth. Where a situation has several correct outcomes, they are spread over clips instead of futures: the ego's move at the all-way stop is drawn per clip, and a twin situation's change and carry-on members are separate training clips.

## How a decision frame is set

Every scene is written for a decision on frame 31. For an earlier decision frame, the same scene starts $31 - f_d$ frames earlier: the warm-up is longer and the ego is placed earlier. The clip then shows the end of the approach, the decision at $f_d$, the reaction, and frames 32 to 63 as its ground truth.

## Situations

| Situation | Where | Change (member 0) | Carry-on (member 1) |
|---|---|---|---|
| Red turns green | signal | the light turns green, the ego pulls away | the red stays |
| Green turns yellow | signal | the yellow comes, the ego brakes | the green stays |
| Queue pulls away | signal | the car ahead moves off on green, the ego follows | the red stays |
| Left turn under protected-permissive heads | signal | the left arrow comes, the ego turns although oncoming cars wait | the green ball comes, the oncoming cars cross, the ego waits |
| Pedestrian steps out, on a street or in the school zone | residential | the pedestrian steps out, the ego brakes | they walk on |
| Overtaking a slow lead | multilane | the car beside moves on, the ego pulls out | it stays alongside |
| Cut-in, merger from a closing lane | multilane | a car moves into the ego's lane 13 to 17 m ahead, the ego slows | a cut-in far ahead and no slower: the ego drives on |
| Red ahead, yellow turning red ahead | signal | the ego brakes to a stop | |
| Yellow too late to stop for | signal | | the ego drives on |
| Lead stops for a yellow | signal | the car ahead brakes to its line, the ego slows | |
| Left turn on a green ball | signal | the last oncoming car passes, the ego turns | |
| Right turn on green, left turn with no oncoming car | signal | the ego slows for the turn | |
| Right turn on red | signal | a crossing car clears, the ego turns | |
| Stop sign ahead, lead stopping at a stop sign | all-way stop | the ego brakes to a stop, or slows behind the car ahead | |
| All-way stop with cars under way | all-way stop | the ego goes where its move is open | it waits where its move is blocked |
| School zone, lane ends | residential, multilane | the ego slows, or moves over | |
| Parked car | residential | the ego steers round it, or with an oncoming car coming, stops behind it and goes round after | |
| Lead brakes | multilane, residential | the car ahead brakes, the ego slows | |
| No decision | any | | cruising, or following a car at a steady gap |

At the all-way stop the open moves follow the user's rule: with no car, or the car on the ego's left turning right, all three moves are open; with the oncoming car going straight, straight and right; with the car on the right turning left, right only; with cars on both sides going straight, none. Two of these pairs pass within drive_v3's 2.8 m conflict distance, the ego's left turn beside a right turn from its left and its right turn beside a left turn from its right, so the scene marks them compatible: the other car is through before the ego gets there.

## Not covered

Two drive_v3 targets have no v4 situation. A pedestrian on the crosswalk of a right turn only grows clearly visible once the ego has already slowed for the turn, so its motion would give the decision away. A straight-lane wait during the left-arrow phase would put the green that follows inside the ground truth for early decision frames, an unannounced change drive_v3 does not allow.

Overtaking has no decision before frame 7: the scene needs a long dashed stretch before the decision, and the ego could not pull out earlier than its 8th to 10th frame. Those clips, and the queues that found no attempt free of a slowdown for something out of the ego's way, became red-light-ahead clips (61 in train, 6 each in val and test).

## Build

The code is in `/root/projects/drivetoy_v4` (scenes in `drivetoy/decision.py`, opt-in simulator rules in `drivetoy/sim.py`) and the data in `/root/projects/drive_v4`, with a per-split summary in `report.md` (`scripts/report.py`). `scripts/build.py` builds any of the five splits from its plan in `plans/`, and `scripts/build_all.sh` builds them all, rendering one split at a time, since each Blender process takes about 1.8 GB of a 24 GB GPU. Train keeps its frames in `render/` only, as drive_v3's train did; the other splits also stack them in `videos.npy`. Besides the drive_v3 labels, each sample stores its `decision_frame` (-1 without a decision; the twin splits leave it out, since it is always 31 there), `ego_turn` (none, left, straight, right or wait), `twin` (the member: 0 change, 1 carry-on), `family`, and per future the 4 s `outcome`, whether it carries out the `change`, and the `onset` frame of the ego's change.

Train was first built with the 23 situations available when it started. An extension pass then moved 2,647 clips to the nine situations added later, keeping each clip's decision frame, and rebuilt the 51 clips whose ground truth braked harder than 4 m/s$^2$; 8 samples of val_twins and 9 of test_twins were rebuilt for the same reason. A last stage, `clean`, rebuilt every clip that broke the rules on frames 32 to 63: 528 train clips, 100 in val, 96 in test, and 206 and 205 entries of the twin splits. Each plan entry therefore names the stage of the code that built it, and `set_stage` in `drivetoy/decision.py` keeps the earlier stages where they differ. The twin splits' stage aimed in at most five undamped steps, checked a pair's futures without the scene's own settings, and let a red-turns-green ego rest as it came to the line. Neither earlier stage cleared the oncoming lane beside a parked car or made an ego with a set braking stop steadily behind a parked car. `scripts/check_reproduce.py` re-simulates samples and compares them with the stored files: all 16,800 matched before the `clean` stage, and afterwards every `clean` entry and 200 others per split matched, 2,698 samples in all. Re-rendered frames differ by at most 1 of 255, and the object-index passes are identical.
