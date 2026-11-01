# Revision History

## Version 2

Download Link: https://drive.google.com/file/d/1RPPM3hEd_IxodmuwEeHVgQm4tncZyDpE/view?usp=sharing

1. **Labels after frame 31 that the video cannot predict.** In 25% of `signal4` clips the ego's light changed in frames 32 to 63 with no yellow at frame 31, in 139 clips the ego stopped for a pedestrian still undecided at frame 31, and turn intent showed only as a lamp of about 7 px. From frame 31 the ego's light now changes only out of a yellow it already shows, any other choice after frame 31 that affects the ego needs its cue on screen by frame 31 (`diversity.cue_problems` checks this), and indicator lamps are drawn at $2.25\times$ real size.
2. **The ego rarely carries out the rules itself.** The ego started 38 to 52 m before a signal's stop line and reached the junction near the end of the clip, so most rule episodes showed only through other cars, and only 2.4% of clips showed the ego completing a turn. Each clip is now built from target labels for its light, cause, speed profile and path, and the recipes place the ego so that it carries them out on screen.
3. **Traffic lights cannot be read at $128 \times 128$.** All signal heads hung on the far side of the junction at $1.5\times$ real size, so a lamp spanned 0.5 to 1 px and the VAE lost most of its colour. Vehicle signal heads are now drawn at $4.5\times$, the ego's own heads stand 4.5 m nearer, and the lamp colours are set so that the VAE keeps them.
4. **The lane end is hard to see.** The merging ego started 90 to 110 m before its lane ended and had passed the small sign by frame 0. It now starts 60 to 80 m before the end, and a 1.6 m sign stands 45 m before it.
5. **Lane changes without a reason.** A lane change is built only as an overtake of a slower leader.
6. **Stop-sign waits labelled "crossing traffic".** Early arrivals stand only on the left and right approaches, and a wait counts as "crossing traffic" only while the other vehicle is roughly perpendicular to the ego, $|\cos(\theta - \theta_{\text{ego}})| < 0.5$.
7. **"Speed limit" and "nudge around" do not match the scene.** "Speed limit" counts only while the 30 km/h school zone holds the ego below the normal limit, and steering around a parked car counts as "parked obstacle".
8. **Signal heads clutter the view.** Near-side heads filled the edge of the frame, and the heads of other approaches showed on the ego's near side, some lit beside the ego's own red. Each approach now keeps only its two far-side mast-arm heads, and no vehicle head on the ego's near side is rendered.
9. **Overtakes of cars that are not slower.** The leader drives at 45 to 60% of the limit, 20 to 32 m ahead, and a lane change is kept only if the ego is at least 1 m/s faster up to the move and pulls out at most 2 s behind the leader.
10. **"Follow" counts leaders at any distance.** A frame counts as following only while the gap is at most 10 m or 3 s at the ego's speed.

## Version 3

Download Link: https://drive.google.com/file/d/14tdae9gY0qZotpeHoUoW4sdveZDlyd71/view?usp=sharing

1. **Leaders crawl so that the ego can "follow".** Lead cars drove only at 9 to 14 km/h with nothing in the scene to explain it (too slow). Leaders now drive at 75 to 97% of the limit, and "follow" means the leader holds the ego within a 2 s gap.
2. **Turns stall at junctions.** A driver that chose to go yielded again, oncoming cars left no usable gap, and a car standing at its own line counted as oncoming. A "go" is now carried out unless a car is already in the conflict area, oncoming cars come at least 2 s apart, only moving cars count as oncoming, and a turn is chosen only into a clear exit.
3. **The left-only lane rarely shows its arrow.** The left-only lane took half of the traffic, and its short arrow rarely lit within the clip. It now takes about 35% of the traffic, and half of the maps add a 6 to 8 s green arrow before the green ball.
4. **Signs and markings are hard to see.** Trees hid stop signs, the school zone showed only as a small sign, and the lane-ends sign could not be read by frame 31. Trees no longer stand in front of signs, red pavement and a painted triangle mark the whole school zone, and two large lane-ends signs stand before each lane end, so the ego signals only once one is legible.
5. **Every future takes the same route.** The ego's route was fixed before frame 0, so its futures almost never turned differently. Where its lane allows several moves and it has not signalled by frame 31, each future now draws the move.
6. **Futures disagree at a yellow light.** Futures of the same clip both stopped and went at a yellow. The ego now goes only if it could not stop comfortably and would reach its line before red, so all futures agree; pending clips turn the light yellow around frame 31, and decided clips earlier.
7. **Choices are not timed around frame 31.** 
- **Previous**: The ego stood at its decision point waiting for frame 31. **Now**: The ego reaches its decision point around frame 31 **naturally**.
- **Previous**: Overtakes began too far behind the leader, and pedestrians. **Now**: starts an overtake close behind the leader
- **Previous**: Cut-ins acted after frame 31 with no cue. **Now** Cut-ins and other road users act on screen by frame 31
8. **Stop-sign futures ignore the ego's choice.** Futures in which the ego let the other car go and futures in which it went first looked alike. Each future now draws one of the ego's open options (letting the others go, left, straight or right), and the other cars follow it: they wait when the ego goes first and go when it lets them.
9. **The ego stays behind parked cars.** Some egos stood behind a parked car, and none had to choose how to pass one. The ego now always goes round: with an oncoming car near, each future chooses to go before it or after it, and with nobody coming every future goes round at once.
10. **The ego stands still when it could go.** The logged ego broke the "didn't go" rule of [task_encoder_eval.md](task_encoder_eval.md) in 71 samples. The builder now rejects such samples and futures.
11. **No labels for scoring planned trajectories.** Scoring a plan needs the road map and whether the ego may go. Each sample now stores its map (`maps/<n>.json.gz`), `may_go` with the reason it may not go for every frame of the sample and its futures, and the ego's move at the junction (`ego_turn`).
