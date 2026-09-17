# Checkpoint navigation: validation record

Status: **development controller; full crater acceptance not yet established**.

The acceptance gate is three repeatable full traversals on the final revision,
without intervention, policy switching or automatic resets. Component passes
below do not satisfy that gate.

## Configuration

- Task: `RexmiRl-Go2w-Crater-Bowl-RockySlope-Play-v0`
- Policy: `logs/rsl_rl/go2w_velocity_rocky_slope/2026-06-30_09-31-48/model_13994.pt`
- Terrain seed: 42. Normal commands: 0.40 m/s, yaw limited to 0.35 rad/s.
- Simulated body envelope: approximately 0.503 m radius, plus 0.10 m clearance.
- Live world-frame ray observations and simulator pose; no SLAM-map reinsertion.
- Default initial position: (13, 0, 4.5), yaw pi.

## Component trials

| Trial | Outcome | Simulation time | Maximum cross-track error |
|---|---|---:|---:|
| Outside-rim straight travel, (13,0) toward (11,0) | Completed | 6.84 s | 0.092 m |
| Floor turn and travel, (0,0), yaw pi, toward (0,2) | Completed using rocky_slope alone | 9.88 s | 0.446 m |

Both completed without manual intervention or policy switching. These were
development revisions before the latest braking/map-memory changes; they are
component evidence, not final-revision repeatability results. Native viewport
captures verified the red route, gold checkpoint and white steering marker.

### Final-revision arrival and hold check

The final controller completed the outside-rim (13,0) to (11,0) test in
25.90 simulation seconds, with maximum cross-track error 0.162 m. It recorded
zero hold failures, zero rolling checkpoints, no intervention and no policy
switching. Stop confirmation required speed below 0.08 m/s continuously for
0.5 s. This confirms the short arrival/hold behavior, not crater acceptance.

## Failures that informed the controller

- Early mapping trials falsely classified parts of a low scarp and the extended
  crater rim as boulders. The detector now accounts for within-cell slope
  variation and separates compact protrusions from extended ridges. Synthetic
  slope, scarp and boulder regressions pass; broad terrain coverage remains needed.
- A zero-command stop on the slope drifted approximately 1.8 m downhill. That
  trial stopped at 44.86 s with only the entry destination confirmed. Zero
  command through the policy is not evidence of physical braking authority.
- The first feedback-braking trial descended to the floor but repeatedly
  confirmed a frontier checkpoint near the destination. It was stopped after
  271.12 simulation seconds, with 1/4 destinations recorded and maximum
  cross-track error 0.666 m. Arrival now uses measured proximity to the mission
  destination rather than requiring the exact goal cell to be traversable.
- Slow turns exposed unnecessary replanning when older observed terrain was
  discarded. Static observations now remain in planning memory; the immediate
  motion corridor must still have fresh observations.

- The map-memory trial confirmed entry and floor destinations (2/4), then
  stalled while replanning the same locally stale uphill route. Freshness now
  constrains local A* expansion as well as the follower, with a bounded
  no-progress replan check. During descent, a hold failure was recorded at
  80.1 s: 0.34 m/s residual speed, 0.50 m drift, despite a -0.40 m/s command.
  A clear moving scan resumed the descent; the next checkpoint hold succeeded.

- The fresh-route traversal reached entry and floor (2/4), then held after
  an uphill stall at 168.52 s. Maximum cross-track error was 0.548 m; one
  hold failure/rolling checkpoint was recorded, with no intervention or policy
  switch. Near-checkpoint slowing had reduced the drive request to 0.25 m/s
  on a measured approximately 31-degree uphill grade. The approach now retains
  the normal 0.40 m/s ceiling uphill until the arrival/braking boundary.

## Braking and moving checkpoints

Position hold uses the same rocky_slope policy, with bounded forward/back
feedback opposing measured motion and displacement. It adds no artificial joint
brakes and changes no friction coefficients. A hold that exceeds three seconds,
or drifts over 0.5 m after one second, is recorded as a limitation. A confirmed
checkpoint crossing can then become a rolling checkpoint if a clear observed
route is available. Operator STOP never resumes autonomy by itself.

Telemetry records commanded motion, state/reason, checkpoint, steering target,
cross-track error, coverage and observation age. The adjacent summary records
hold failures and rolling checkpoints. An individual run always retains
`validated: false`; assess the repeatability gate separately.

## Isolated climb trial

With the 0.40 m/s uphill approach and shorter observed corridor, the floor-to-
(-13,0) trial ended at its 199.98 s step limit without reaching that destination.
Maximum cross-track error was 0.760 m. Two hold failures were logged; the last
had 0.19 m/s residual speed and 0.50 m drift. No manual intervention or policy
switch occurred. Turning, lateral drift and repeated route changes prevented
a reliable climb-out. This remains a failed component trial.

The current sensor configuration uses yaw-aligned rays and a maximum vertical
beam elevation of +7 degrees. On steep uphill terrain, fresh forward coverage
is short. The controller now checks stopping distance plus margin instead of
a fixed 1.5 m minimum; that correction alone did not solve the climb.

## Comparison with saved teleop evidence

`logs/telemetry/manual_20260910_01/samples.csv` contains a successful manually
driven exit in the scenario `climbing crater high speed 0.8m/s`, using the same
rocky_slope checkpoint. Recorded commands include ±0.80 m/s and ±0.35 rad/s.
The segment moves from approximately (1.08, 1.99, 1.09) m to
(8.75, 7.58, 4.91) m, followed by an outside-roaming segment. The new controller
uses the shared command adapter but retains the agreed 0.40 m/s normal ceiling.
This comparison does not isolate command magnitude as the only cause: the
manual approach, terrain/randomization and steering history also differ.
A controlled uphill command/stopping study is needed before enabling a higher
autonomous uphill command. The existing 0.80 m/s cruise flag does not bypass
the low-slope, freshness and braking guards.

## Automated checks

31 checkpoint/map/command tests and 4 perception geometry tests pass. They
include a regression for measured arrival at a frontier endpoint and a check
that historical route memory cannot authorize motion through stale cells.

## Remaining acceptance work

Full crater descent, floor transit/turn and climb-out must pass three times on
the same final revision. Repeat on additional terrain seeds afterward. The
0.80 m/s ceiling remains opt-in and unvalidated; normal-speed braking estimates
are a guard, not a substitute for high-speed stopping and tracking trials.

See [launch and legacy comparison commands](checkpoint_navigation.md).
