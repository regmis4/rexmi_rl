# Spawn-area stall and cost-tile rendering

The reported run travelled from (13,0) to approximately (11.91,0.26), then
commanded zero forward velocity with approximately 0.22 rad/s yaw until HOLD.
This was a controller-imposed stop, not proof that the forward policy lacked
traction. A fixed 40 cm straight probe cancelled forward movement even when
the route curved into a clear corridor. Replaying the recorded scans reproduced
the same zero-forward command and 0.219 rad/s yaw request.

## Corrections

- Check current-heading stopping distance and the commanded turning arc.
  A slower 0.20 m/s, 0.35 rad/s turn is available only if both checks pass.
- At tight corners, turn toward the first route segment rather than steering
  only toward a distant lookahead. Report why forward motion is blocked.
- Retain travelled positions until recovery can use them. Previously the
  8-second history expired before the 30-second stall timer. A retreat still
  requires the same corridor to be freshly observed and clear.
- Reset the replan timer after the bounded retreat, while retaining the
  one-retreat limit. Do not immediately HOLD on the timeout from before retreat.
- Prevent unrelated stopping events from corrupting arrival braking estimates.
- Keep each displayed terrain cell in a stable mesh-face slot. Previously
  nearest-cell sorting reassigned faces as the robot moved.
- Use opaque active tiles, fixed mesh topology, and explicit colour/opacity
  indices from initialization. The supplied warning specifically named
  `primvars:displayOpacity:indices` on `/RexmiPerception/ObservedCosts`.

## Live validation

Task: `RexmiRl-Go2w-Crater-Bowl-RockySlope-Play-v0`, seed 42, unchanged
`model_13994.pt`, default spawn (13,0,4.5), target (11,0). This target matches the
first destination of the default traverse mission. The test used a single-goal
mission so it closed after confirming that destination.

The first arc-only correction still stalled. Corner alignment alone also failed
to complete. Restoring the travelled-corridor retreat and resetting its timer
allowed the final visible run to complete:

| Result | Value |
|---|---:|
| Arrival and confirmed stop | 36.86 simulation seconds |
| Maximum cross-track error | 0.115 m |
| Manual interventions / policy switches | 0 / 0 |
| Hold failures / rolling checkpoints | 0 / 0 |

The opacity-index warning and cost-overlay errors were absent in the final
visible run. Velocity-arrow prototype and GPU-interface performance warnings
still appeared; neither prevents the demonstrated checkpoint completion.

37 checkpoint/mapping tests and 5 perception tests pass, including stable tile
identity under reordering, stopping-room constraints, and retreat followed by
replanning. This validates the reported first-checkpoint case only. Full crater
acceptance remains unestablished. Restart an existing simulation to load these
code changes; an already-running process retains its old controller and mesh.
