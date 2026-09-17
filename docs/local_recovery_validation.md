# Local recovery follow-up

The supplied 90–130 second trace shows a route rejection at 91.4 seconds,
followed by permanent HOLD at 94 seconds. The robot then moves from about
(5.36, -1.51) to (4.87, -2.09), while HOLD continues sending position-hold
commands. The old controller never retries planning from that changed position.
The trace does not contain the terrain grid, so it cannot establish whether
the original obstacle classification was correct.

## Changes

- Automatic route failures enter REASSESS. Operator STOP, tipping, and exhausted
  recovery still enter HOLD and require Resume/manual assistance.
- Replanning runs once per simulation second from the current measured position.
  The obsolete stopping anchor is discarded during reassessment.
- If planning fails, compare footprint-clear forward/reverse arcs and left/right
  yaw commands. Rank progress, alignment, clearance and cross-slope travel.
- Short maneuvers are checked continuously, then followed by settling and a new
  route. Six attempts or 30 seconds without leaving the local area ends recovery.
- Previously ineffective commands at the same position/heading are not repeated.
- Reverse checks the same observed corridor as forward. Actual measured drift
  also needs stopping room; a reverse request does not imply reverse motion.
- Terrain memory, obstacle thresholds, robot footprint, policy and mission order
  are unchanged. No lateral strafe commands, automatic resets or policy switches.

## Automated checks

51 navigation/map/command tests and 6 perception tests pass. New cases include
replanning after displacement, manual STOP persistence, unknown terrain,
forward-versus-reverse selection, yaw direction, obstacles behind the robot,
failed-action suppression, newly blocked recovery corridors, measured drift,
and time/attempt bounds. Python syntax and whitespace checks also pass.

## Simulator checks

Development crater trial (before the final measured-drift guard): seed 42,
normal 0.40 m/s ceiling, 160 simulation seconds. Reached 2/4 mission destinations;
ended at the step limit while climbing out. Maximum cross-track error 0.754 m;
three hold failures and two rolling checkpoints. No manual intervention or
policy switching. This trial did not enter the new REASSESS/MANEUVER states,
so it is a route regression check, not evidence that recovery succeeds in physics.

Artifacts: `/tmp/rexmi-local-recovery/traverse.csv`, `.summary.json`, `.map.npz`
and `.log` (local temporary files, not committed).

Final-revision nearby-start trial: seed 42, initial position (5.36, -1.51,
1.95) m, yaw 2.773 rad, goal (0, 0). Completed in 27.14 simulation seconds;
maximum cross-track error 0.156 m, zero hold failures, rolling checkpoints,
manual interventions or policy switches. No recovery maneuver was needed.
Artifacts: `/tmp/rexmi-local-recovery/slope-start.csv`, `.summary.json`,
`.map.npz` and `.log`.

Reproduce the final-revision component trial from the repository root:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgcc_s.so.1 ./run.sh scripts/navigate.py \
  --task RexmiRl-Go2w-Crater-Bowl-RockySlope-Play-v0 \
  --checkpoint logs/rsl_rl/go2w_velocity_rocky_slope/2026-06-30_09-31-48/model_13994.pt \
  --mission velocity_goal --goal_x 0 --goal_y 0 \
  --spawn_x 5.36 --spawn_y -1.51 --spawn_z 1.95 --spawn_yaw 2.773 \
  --headless --no_dashboard --max_steps 2500 \
  --log_file /tmp/rexmi-slope-start.csv
```

The reported failure's accumulated sensor history was not saved, so a new
spawn near that location cannot reproduce its exact cost map. Three complete,
repeatable crater traversals and a physical-policy recovery trial remain
required before declaring the new controller validated.
