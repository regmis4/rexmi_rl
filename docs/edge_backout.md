# Leaving a partial cliff overlap

The supplied run stopped because one 35.8° map cell touched the robot's circular
footprint. The saved 60 s survey shows that reversing about 0.10 m clears that
overlap, and reversing 0.40 m reduces the maximum slope under the footprint to
32.9°. The old controller rejected the starting footprint in every direction.
Its 25° ascent limit also rejected the 31.6° reverse approach. The tiny negative
commands in HOLD were position feedback, not a planned reverse maneuver.

This diagnosis comes from the terrain map and body pose. The controller does
not measure which wheel is airborne or certify contact from these cells.

## Current behavior

- Before ordinary route recovery, compare straight reverse, gentle reverse
  arcs, and forward escapes against the measured surrounding terrain.
- Choose an observed endpoint 0.4–0.8 m away with no steep-footprint overlap.
  Prefer a short straight backout, especially along a previously driven corridor.
- Allow departure from an existing overlap only when it occupies at most 25%
  of the sampled footprint. Predicted overlap must shrink without introducing
  another steep cell or re-entering a cleared cell.
- Keep the centre and travel grades within 35°. The 25° mission ascent limit is
  suspended only for this bounded escape. Obstacles, unknown ground and stale
  newly swept ground still prevent a motion. Terrain classifications are unchanged.
- Use 0.20 m/s initially, or the normal 0.40 m/s command when the escape climbs
  more than 15°. Ineffective creep can escalate only after checking the larger
  stopping corridor. Yaw is limited to 0.12 rad/s during backout.
- Keep a checked counter-command when inertia points toward the edge; replacing
  it with weaker velocity damping can worsen sliding. Recheck actual footprint
  exposure and body tilt every tick.
- If progress fails for four seconds, retire that arc and compare another. A
  small observed increase in cliff overlap also triggers a fresh comparison
  from the actual pose, rather than repeating the same command. This requires
  known partial support and less than 0.25 m displacement from the attempt start.
- Limit recovery to three attempts in the same area and ten seconds per attempt.
  Increased tilt, unknown support, excessive drift or exhausted alternatives
  cause HOLD. Operator STOP and manual control interrupt recovery immediately.
- After backing clear, stop, refresh observations and replan the original mission.
  The temporary green checkpoint shows the backout endpoint. No destination is
  skipped and no automatic reset or policy switch is introduced.

The states are `EDGE_BRAKE`, `EDGE_MOVE` and `EDGE_STOP`. The dashboard reports
the direction and steep-cell overlap. Events record candidate accept/reject
reasons, the chosen arc, retries and completion.

## Validation

The [saved-map replay](validation/edge_backout_replay.json) now selects straight
reverse at **−0.40 m/s** at the reported pose. It preserves the measured map and
reconstructs local freshness from stored observation timestamps.

Physical recovery is **not yet validated**. Nearby hillside-start trials did
issue reverse commands, but the robot continued drifting downhill. The final
trial tried straight reverse and both gentle reverse arcs, then held as cliff
overlap continued increasing. The robot began from a fresh spawn, so this does
not recreate the original settled leg configuration. A separate replay of the
recorded clicks rejected the first target as insufficiently observed and did
not reproduce the approach. These are recorded failures, not successful trials.

[Regression counts and simulator results](validation/edge_backout_summary.json)
include all trials. Geometry and command tests establish that a checked backout
is permitted; they do not establish that the policy can physically recover every
partially supported pose.

Existing launch commands are unchanged. For development, accepted clicks can be
replayed at their original simulation timestamps with:

```text
--checkpoint_replay path/to/run.events.jsonl
```

Use this with `--headless --no_dashboard` and without `--manual_checkpoints`.
This replays target requests only. It does not load a stored map, force a pose,
move the robot or guarantee identical sensor observations.
