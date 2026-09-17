# Checkpoint navigation

The default navigation controller uses one **rocky_slope** policy and the same
policy loader, command ranges and velocity injection as teleop. It observes,
plans a local route, follows it, attempts to hold at a checkpoint and observes again.
This development default has not yet passed the full crater acceptance gate.

## Launch

```bash
cd /home/susan/rexmi_rl
./run.sh scripts/navigate.py \
  --task RexmiRl-Go2w-Crater-Bowl-RockySlope-Play-v0 \
  --checkpoint logs/rsl_rl/go2w_velocity_rocky_slope/2026-06-30_09-31-48/model_13994.pt \
  --perception_view --no_dashboard --max_steps 15000
```

Use `--headless` for a recorded unattended test. Use `--seed 42` (default) to
repeat the terrain. `--log_file /tmp/traverse42.csv` selects the telemetry file;
the adjacent `.summary.json` records the outcome. A HOLD ends a headless test
as a failure to complete, while a visible session stays open for inspection.

On this workstation, if Isaac Sim reports a missing `GCC_12.0.0` symbol, prefix
the command with `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgcc_s.so.1`.

Legacy comparison, using the same checkpoint and task:

```bash
./run.sh scripts/navigate.py \
  --nav_controller legacy \
  --task RexmiRl-Go2w-Crater-Bowl-RockySlope-Play-v0 \
  --checkpoint logs/rsl_rl/go2w_velocity_rocky_slope/2026-06-30_09-31-48/model_13994.pt \
  --policy_mode rocky_slope --no_turn --perception_view --no_dashboard
```

Existing multi-policy arguments remain available only in legacy mode. The new
controller accepts `--ckpt_rocky` as an alternative to `--checkpoint`; it never
loads the rough or turn checkpoints. `--policy_mode auto` does not enable
switching in checkpoint mode. Legacy speed/recovery/SLAM tuning flags do not
configure the checkpoint controller.

## Controls and display

The **REXMI | Checkpoint control** window shows state, reason, checkpoint,
command, mapped/fresh coverage within 2 m, surveyed area and observation age. STOP requests position hold through the balancing policy, using bounded
forward/back feedback when zero command alone cannot resist a slope.
Press and hold Forward/Back/Left/Right for manual control; release requests hold.
Manual control suspends autonomy. RESUME AUTO collects fresh scans and replans.
`--start_paused` opens in HOLD. Terminal equivalents are `s`/`stop`, `resume`,
and `f`, `b`, `l`, `r` (terminal movement stays active until another command).

- Red line: the active route that actually controls steering.
- Gold cross in Isaac Sim / diamond in the dashboard: local checkpoint.
- White cross: steering lookahead.
- Red tiles: detected obstacles or very steep terrain.
- Muted amber tiles: clearance buffer (the planner avoids this space).
- Cyan points: live LiDAR returns; toggle costs off to inspect them.

The existing mission destinations retain their order. Local checkpoints lie
roughly 2–4 m along the observed route, with shorter segments on slopes/bends.
The controller aims inside a 0.6 m arrival boundary and tries braking/position
hold before scanning. If holding fails, an observed checkpoint crossing can be
recorded as a **rolling checkpoint**, and scanning continues only on a clear
observed route. Each such event is recorded as a hold limitation. It does not skip destinations,
teleport the robot upright, or use policy switching as recovery.

## Movement and mapping

Normal forward/back commands are 0.40 m/s; turning is limited to 0.35 rad/s and
reduces near alignment. Uphill approaches retain the normal forward command
until the arrival/braking boundary; the lower creep command stalled in trials.
The policy stays active when asked to stop. The hold controller opposes measured
body-forward motion and position error with commands bounded to ±0.40 m/s;
heading correction is capped at ±0.20 rad/s. These are learned-policy commands,
not a mechanical brake model or an assertion of available traction.
Command injection happens before observation and inference in both teleop and
checkpoint navigation, so a stop is not delayed by the previous command's
observation. No trained weights or policy observation layout are changed.

Higher speed is experimental: `--cruise_speed 0.8` sets an upper limit, not a
constant command. It requires a straight clear route, low tracking error,
recent dense observations, low slope and braking evidence from a normal-speed
stop. The default ceiling remains 0.40 m/s until cruise trials establish when
0.80 m/s is appropriate. Slopes above 50 degrees are outside the initial
controller operating band; this is a configuration limit, not a tested limit
of the policy.

Mapping uses live world-frame LiDAR plus short-range height/forward returns.
The simulator's pose is still used for navigation. It is not a sensor-only
localization demonstration, and the sensor model is Isaac Lab RayCaster.
The drifting ICP/SLAM cloud is not reinserted into the navigation grid.

Ground inclination comes from a local fitted plane. Compact protrusions,
unexplained within-cell spread and abrupt changes between neighboring cells
detect obstacles. The detector accounts for height uncertainty within a 20 cm
cell and distinguishes extended ridges from isolated boulders. The configurable
`--step_thresh 0.20` is an obstacle threshold; not every visible small rock must
be red. Footprint inflation uses the simulated wheel/leg body envelope plus
clearance. Terrain measurements accumulate in 5 cm spatial bins inside each
20 cm planning cell. Consistent returns are averaged; untouched bins retain their
history. A higher return enters immediately, while lowering a previously measured
top requires three consistent scans at the same bin. Missing returns do not clear
obstacles. All sensors at one timestamp contribute to one fused scan.

Terrain costs no longer include an age penalty: old flat ground stays flat and
mapped boulders remain mapped. Freshness is separate. Motion still requires
observations no older than 8 simulation seconds in the immediate corridor. Small gaps between
nearby returns can be filled only with local plane support; broad unseen or
occluded regions remain unknown. Fresh coverage includes these supported small gaps;
it is not a probability of safety. Fresh coverage can fall as observations age,
without removing measured terrain or reducing the surveyed area. The viewport shows the whole observed survey, with 20 cm tiles nearby and
40 cm aggregates at a distance. The stored map retains full resolution.
Navigation checkpoints do not reset it.

The final `.map.npz` contains the accumulated elevation bins and observation
counts as well as the terrain grid. For raw-scan diagnostics, add
`--scan_log /tmp/rexmi-scans` to save each sensor batch for replay. This is opt-in
because raw recordings use more disk space. See [mapping checks](mapping_memory.md).

The immediate route check uses stopping distance plus 0.30 m, with a 0.60 m
minimum, rather than demanding 1.5 m of fresh terrain on every steep wall.
A shorter clear steering target is used when the usual lookahead is unobserved.
If the immediate stopping corridor becomes blocked or stale, the robot
brakes and replans. Staleness farther down the route does not repeatedly cancel
a safe slow turn. A stop requires speed below 0.08 m/s continuously for half a simulation second.
Holding is judged from measured motion: after three seconds,
or more than 0.5 m drift after one second, the controller records a hold failure.
It may then scan while moving through an observed clear route. An operator STOP
never triggers automatic resumption; it continues requesting hold until Resume. If a new observation places the robot in the extra clearance margin, it can
leave slowly while preserving full footprint clearance and without moving closer
to the obstacle. If the footprint itself is blocked, it holds. A stalled
robot can make one short retreat only through a recently travelled and still
observed clear corridor. Repeated replanning without meaningful displacement is bounded to 20 seconds.
If replanning or that retreat fails, automatic blockage enters REASSESS instead
of a permanent HOLD. Once per simulation second it replans from the measured
position, then compares short forward/reverse arcs and left/right yaw commands
if no route is available. Candidates must pass fresh corridor, footprint,
corner and stopping checks; scoring considers destination progress, heading,
clearance and cross-slope travel. Left/right remain yaw, not lateral strafing.

Each recovery maneuver lasts at most two seconds, 0.4 m or 0.65 radians before
settling and replanning, and is checked continuously for newly blocked terrain.
It avoids repeating the same command at the same pose. Recovery is bounded to
six attempts or 30 seconds in the same area; then it requests manual assistance.
Unknown terrain and actual footprint conflicts are never cleared to force an
escape. While reassessing, braking opposes current drift rather than pulling
the robot back toward an obsolete stopping point. A tip and operator STOP
remain HOLD and never resume automatically. No automatic resets are used.

## Validation

```bash
python -m unittest discover -s tests -p test_checkpoint_navigation.py -v
python -m unittest discover -s tests -p test_perception_view.py -v
```

Tests cover slope/obstacle separation, sensor-relative range, freshness,
contradictory evidence, route clearance, corner cutting, steering signs,
manual stop/resume and same-tick command delivery.

Acceptance requires three repeatable full crater traversals without manual
intervention, automatic resets or policy switching. A successful short run or
passing unit tests alone does not establish that acceptance. Live results are
recorded in `checkpoint_validation.md`.

The [spawn-stall follow-up](spawn_stall_validation.md) documents the motion-arc,
route-corner alignment, safe-retreat and stable-tile rendering corrections.

See [local recovery checks](local_recovery_validation.md) for the blocked-route
reassessment change and its validation limits.

The [terrain survey update](terrain_survey.md) adds whole-survey layer views,
directional climb evidence, route comparison and periodic complete exports.
