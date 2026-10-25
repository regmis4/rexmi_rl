# Teleop demonstration: command persistence, release and reverse

The recorded exterior–floor–exterior traversal demonstrates useful motion with
the unchanged rocky-slope policy and commands no greater than ±0.40 m/s. The
autonomous controller's interruption and endpoint rules are a substantial part
of the gap. Zero desired velocity remains an active policy action and sometimes
continues uphill motion; this capture does not show that it universally climbs
better or establish wheelspin as the cause.

## Recording and verification

Source: `logs/telemetry/crater-demo-20260923-222151/`, plus its sibling navigation
CSV. The retained analysis contains 7,566 complete control steps and 30,264
physics samples, covering 151.32 simulated seconds. Timestamps and sample indices
are continuous. Commands agree between both logs, and poses agree within the
navigation CSV's rounding. All retained samples use `rocky_slope`; there are no
recorded resets or episode changes.

The recorder did not write `capture_end`. Two physics substeps of the next control
step were saved; those two rows are excluded explicitly. The verified retained
prefix includes the traversal and exterior endpoint, but does not establish
normal recorder shutdown. Raw files were not changed. Hashes and exports are in
[the numerical summary](../../outputs/teleop_analysis/crater-demo-20260923-222151/summary.json).

The robot entered radius 3 m at 62.34 s, passed within 1.029 m of the centre,
and crossed outward through radius 11 m at 143.52 s. It finished near
`(1.974, 12.125)`, radius 12.284 m. This is a manual traversal through a northern
exit, rather than completion of the autonomous destination `(13, 0)`. That
distinction suggests selecting a viable exit first, then routing outside the
crater to the requested destination.

## What happened after release

All times below are simulated seconds. Displacement is net horizontal root
motion; height change is root height, not a ground-slope measurement.

| Interval | Command | Duration | Horizontal displacement | Height change | Mean absolute wheel speed | Mean loaded wheels* |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 87.94–92.36 | +0.40 m/s, zero yaw | 4.42 s | 0.989 m | +0.621 m | 7.195 rad/s | 2.22 |
| 92.36–96.78 | zero translation and yaw | 4.42 s | 0.651 m | +0.279 m | 3.382 rad/s | 3.18 |
| 137.00–141.38 | zero translation and yaw | 4.38 s | 0.203 m | −0.039 m | 0.479 rad/s | about four |

*Reconstructed from each wheel body's normal-contact-force magnitude exceeding
5 N. This is a loading proxy, not a measured traction coefficient. The logger's
wheel-count field is unavailable because it searches for body names containing
`wheel`, while the asset names these bodies `FL_foot`, etc.

After release at 92.36 s, wheel targets remained nonzero: mean absolute target
speed was 4.248 rad/s. Even after excluding the first full second, the robot
advanced another 0.475 m and gained 0.189 m over 3.42 s. This is sustained
policy-driven behavior, not something adequately explained as a brief coast.
The code continues running policy inference when both requested velocities are
zero; release does not turn off the motors.

The equal-duration forward interval immediately before release gained more
height and distance. Release was associated with slower wheel motion, more loaded
wheels and lower body tilt, but terrain, posture and contacts also changed.
This observation supports testing a policy-active zero-command settling action;
it does not support replacing climbing with zero input everywhere. The later
zero-command interval included small backward/downward movement.

## Wheelspin: plausible, not established

Between 114.82 and 116.70 s, a +0.40 command produced only 0.048 m net horizontal
movement while mean absolute wheel speed was 7.596 rad/s. Between 122 and 127 s,
mean wheel speed was 9.304 rad/s, but net horizontal movement was only 0.128 m
and height gain was 0.007 m. These are poor motion-conversion intervals.

However, high rotation often occurs during unloaded gait phases. In the
pre-release window, the rear-left/right wheels averaged about 2.95/2.81 rad/s
while loaded, compared with 20.33/16.63 rad/s while unloaded. High wheel speed
alone would falsely identify some normal gait motion as ground slip.

Current data lack individual wheel contact-point velocities and tangential
friction forces. Body velocity, wheel rotation and normal contact vectors are
not sufficient for a true slip ratio. See the conditional
[wheel-loading export](../../outputs/teleop_analysis/crater-demo-20260923-222151/wheel_loading.csv).

## How long the successful commands were held

| Observed action | Duration | Response |
| --- | ---: | --- |
| Pure forward, 62.22–92.36 s | 30.14 s | 9.395 m net horizontal motion; heading changed about 25.9°. |
| Continuous reverse, 103.66–109.44 s | 5.78 s | 0.726 m horizontal repositioning, with about 40° heading adjustment and 0.264 m height loss. |
| Pure forward climb, 122–137 s | 15.00 s | First 5 s: only 0.007 m height gain. Next 10 s, unchanged command: another 1.253 m height gain. |
| Pure yaw, 17.78–26.10 s | 8.32 s | About 86.6° heading change, with about 0.988 m of translation. It was not an in-place pivot. |

There were 15 forward-button holds, median 3.70 s and longest 34.48 s (including
steering adjustments). The one reverse hold lasted 5.78 s. The 42 yaw-button
holds had median 0.29 s and maximum 10.18 s. Button holds group each command
component independently, so a yaw adjustment does not terminate a forward hold.
Exact command combinations and timing are retained in
[command segments](../../outputs/teleop_analysis/crater-demo-20260923-222151/command_segments.csv)
and [button holds](../../outputs/teleop_analysis/crater-demo-20260923-222151/button_holds.csv).

These are examples and response ranges, not universal open-loop timers. The
same fixed command produced both low progress and later useful climbing.

## Why autonomy's reverse looked ineffective

In completed autonomous trial 04, its first recovery action spent approximately
17.82 s in BRAKE, 7.74 s aligning, and 8.12 s in DRIVE_REVERSE. Its longest
uninterrupted full reverse command was only 1.54 s. The human reverse command
remained at −0.40 m/s for 5.78 s while steering was adjusted.

These occurred at different places and poses, so they are not a matched
performance comparison. They do establish a command-scheduling difference:
autonomy repeatedly interrupted its reverse action. The policy's ability to
reverse cannot be judged from those interrupted attempts alone. The source
and calculations are preserved in the
[autonomy comparison](../../outputs/teleop_analysis/crater-demo-20260923-222151/autonomy_comparison.json).

The long pure-forward manual segment also accumulated 2.22 m lateral offset
from its initial body-heading line, and 0.545 m from its start-to-end chord,
while continuing useful traversal. A fixed narrow line is a poor substitute
for checking the available terrain corridor. This does not mean an arbitrary
2.22 m tolerance is safe everywhere.

## Changes supported by this evidence

1. Use a terrain-supported corridor and checkpoint arrival region, with the
   onward route checked from the actual pose. Avoid forcing an exact point and
   then repeatedly correcting heading near it. Define region success before
   predicting or scoring the approach.
2. Give forward/reverse actions a meaningful commitment interval. Distinguish
   temporary low progress from a completed failed attempt. This trace contains
   a five-second low-progress period followed by successful breakthrough.
3. Add a distinct zero-command, policy-active settling option within a climb.
   It must preserve the approach and evidence and monitor actual movement.
   Release, active velocity damping and stationary hold are different behaviors.
4. Base loading/slip diagnostics on contact-conditioned motion. Do not use raw
   wheel angular speed alone to trigger retreat or exclusions.
5. Characterize turn response jointly with translation and posture. The trace
   contains both small steering corrections and long turns that translate the
   robot. It does not justify blindly copying combined commands onto slopes or
   removing the user's restriction on demanding turns while climbing.

The next controlled comparison should test continued drive versus a bounded
release from matched climbing states, retaining ±0.40 m/s commands and the
terrain/posture protections. One successful demonstration guides this design
and characterization; it does not calibrate checkpoint probabilities.

No navigation-controller behavior was changed during this analysis. A separate
AI control reviewer independently reproduced the release, wheel-loading and
low-progress/breakthrough findings and agreed with the causal limitations.

## Plots and reproduction

- [Route and height](../../outputs/teleop_analysis/crater-demo-20260923-222151/route_and_height.png)
- [Climb command/response](../../outputs/teleop_analysis/crater-demo-20260923-222151/climb_response.png)

```bash
/home/susan/miniconda3/envs/env_isaacsim/bin/python scripts/analyze_teleop_navigation.py \
  logs/telemetry/crater-demo-20260923-222151 \
  --output outputs/teleop_analysis/crater-demo-20260923-222151
```
