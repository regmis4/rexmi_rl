# Checkpoint navigation

For recovery when a wheel/leg envelope already overlaps a steep tile, see
[checked edge backout](edge_backout.md). The normal route restrictions remain
unchanged; a bounded escape evaluates whether moving restores footprint support.

The default navigation controller uses one **rocky_slope** policy and the same
policy loader, command ranges and velocity injection as teleop. It observes,
plans a local route, follows it, attempts to hold at a checkpoint and observes again.
This development default has not yet passed the full crater acceptance gate.

For the current ascent limits, diagonal routes and committed downhill recovery,
see [lower-grade ascent and retreat](ascent_retreat.md).

See [tight turns and return routes](navigation_turns_and_routes.md) for steep-edge-only
reverse assistance, whole-route slope costs and the full-survey return-test command.
Flat-ground turns and safe slope turns stay with ordinary route following.
See [slip tracking and recovery](slip_and_route_recovery.md) for measured-motion
failure detection, learned no-go patches and comparison of uphill approaches.

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

The controller uses the rocky-slope checkpoint throughout. `--ckpt_rocky` is an
alternative to `--checkpoint`; no automatic policy switching is used.

## Click-to-go and display performance

Start paused for dashboard checkpoint selection (do not add `--no_dashboard`):

```bash
cd /home/susan/rexmi_rl
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgcc_s.so.1 ./run.sh scripts/navigate.py \
  --task RexmiRl-Go2w-Crater-Bowl-RockySlope-Play-v0 \
  --checkpoint logs/rsl_rl/go2w_velocity_rocky_slope/2026-06-30_09-31-48/model_13994.pt \
  --perception_view --manual_checkpoints --lidar_display_points 2000 --overlay_hz 30
```

Toggle **Manual checkpoints: click dashboard map** in the perception overlay, or
use **CLICK CHECKPOINT MODE** in the control window. Left-click terrain in the separate **REXMI Nav Dashboard** window:
the full map on the **left** or the zoomed map on the **right**. This selects
a world-coordinate destination. Pan/zoom gestures and clicks outside
the map never command motion. This is a dashboard map picker, not 3D viewport
picking. The controller accepts observed, unblocked destinations with an observed
route, then refreshes scans and follows its existing cost-aware checkpoint planner.
It retains the same policy, stopping checks and 35-degree slope limit. A click can
replace the selected destination after route validation. **A rejected click keeps
the previous valid goal and its controller state**; use STOP to stop the robot.
Double-clicks and repeated clicks in the active goal's map cell do not restart
observation. Selection uses stored observed geometry; fresh scans still gate
actual motion. A new click reports its coordinates and acceptance/rejection
reason in the dashboard and `.events.jsonl`, including whether the prior goal
was kept. A clear centre tile alone is insufficient if the destination footprint
overlaps unobserved or >35° terrain.
At arrival it holds and waits for another click. STOP and manual driving suspend
it. Turning checkpoint mode off restores the original mission and progress,
paused; press RESUME AUTO to continue it.

Map retention flags (checkpoint controller):

```bash
# Default: forget terrain beyond 20 metres from the robot:
--map_retention radius --map_keep_radius 20
# Optional: retain the full survey for this run:
--map_retention full
```

The default rolling map keeps terrain within **20 m of the robot**, rather than
accumulating the whole mission survey. Radius mode discards old heights, observation support and traversal outcomes;
forgotten terrain becomes unknown and must be observed again before use. Exported
maps contain only the retained survey. This bounds retained evidence and rendered
survey area, not the reserved size of the fixed map arrays. Full mode remains
limited to the existing 64-metre map extent; neither mode loads previous runs.

Display tuning:

```bash
# Lighter overlay:
--lidar_display_points 1000 --overlay_hz 20
# Maximum current display density:
--lidar_display_points 6000 --overlay_hz 30
```

The default is 2,000 displayed points at a 30 Hz target. All sensor returns still
feed the map; normal mapping aggregates evidence rather than storing a growing
raw point cloud (`--scan_log` explicitly records raw scans). Full-map dashboard
copies now occur at map updates, local quality checks use a small neighbourhood,
and ground-plane solves skip unsupported cells. Separable neighbourhood filters
replace the previous full 2D filters with matching map-classification tests. Compressed exports run on a
worker thread using an independent snapshot; shutdown waits for the final export.
The Matplotlib dashboard still redraws at 2 Hz. Target overlay frequency is not a
guarantee of simulator FPS, especially during map rebuilds.

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
- Red tiles: detected obstacles. Orange tiles: slopes above 35° (blocked, no added slope buffer).
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
0.80 m/s is appropriate. Slopes above the user-reported 35-degree ceiling are outside the
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

Confirmed slip/stall episodes and repeated command-rejection loops now use the
separate `ESCAPE_*` sequence: brake, try a checked backoff or heading change,
then replace the local checkpoint and route. It has a 15 s budget and at most
three movement attempts for local recovery. Failed climbs instead commit to
lower ground with a separate 90 s retreat budget and retain the original
mission destination. See [ascent and downhill retreat](ascent_retreat.md) and
the [recovery guide](slip_and_route_recovery.md).

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


### September 17 performance and interaction checks

83 navigation/map tests plus 8 perception tests pass, including click gating,
blocked/unreachable target rejection, STOP, mission restoration, radius forgetting
and old/new plane-filter classification equivalence. Two 150-step visible startup
checks used the same policy, seed, 1,000 displayed points and 20 m retention. Mean
map rebuild fell from 97.1 ms to 39.9 ms after separable filtering. Both waited in
HOLD for selection and exported successfully. These short stationary checks are
not a sustained driving-FPS measurement or a live click-to-arrival validation.

## Sideways-motion assessment

The rough checkpoint has demonstrated sideways motion on nominally level terrain.
See [rough policy assessment](rough_policy_lateral_assessment.md) for measured drift,
training ranges and the reproduction command. Navigation still uses rocky-slope;
autonomous sidestepping and policy transitions are not enabled by this study.

The follow-up [flat sidestep trial](flat_sidestep_trial.md) now provides an opt-in
`--flat_sidesteps` mode with stopped, observed-flat handoffs. The earlier assessment
records isolated-policy results; it is not the validation record for switching.

## Remote controller: footprint terrain costs

Launch the experimental autonomous outside–center–outside mission with Isaac Sim,
dashboard, costmap and LiDAR overlays:

```bash
cd /home/susan/rexmi_rl
conda activate env_isaacsim
bash scripts/launch_remote_round_trip.sh
```

The Perception window and dashboard offer `terrain`, `slope`, `roughness`,
`footprint`, `steps`, `support`, and `traversal` layers. `roughness` retains the
original within-cell diagnostic. `footprint` shows plane-detrended RMS over a
square that encloses the robot at any heading (1.4 m for the present body/grid).
`steps` shows detrended adjacent-cell height jumps over that neighborhood. These
are terrain difficulty proxies, not exact wheel-contact or traction predictions.
The fit uses retained sub-bin midrange heights without suppressing high returns.

Remote navigation replaces its old 6 cm roughness deadband with two finite costs:
`min(4, 40 * RMS_metres)` and `min(4, 40 * jump_metres)`. A conditioned fit needs
at least 80% of its raw bins; the jump estimate also needs 80% of interior
adjacent pairs. Unavailable metrics remain NaN. Either unavailable component
replaces both addends with an explicit uncertainty cost of 8. `support` displays
the lower coverage, with unavailable estimates grey. The selected route's status
shows base, roughness, step and unknown contributions separately.

All eligible complete routes compete on total cost, including existing finite
slope penalties. Peak slope no longer has absolute ranking priority. The bounded
48 corridor screens and at most six connection searches remain; preliminary
connection distance costs 10 units/m and is only a search estimate. Final ranking
uses the connected route's actual cost. Existing obstacle, slope, directional
exclusion and movement protections are unchanged. Other controllers retain their
previous costs. Features are cached by map revision and copied for background
planning; map exports retain their values and coverage.

The remote heading-state search uses a reverse directed terrain-cost graph for
queue guidance, relaxing heading/turn staging only in that estimate. The
executable search still checks every edge and keeps its 60,000-state limit.
Guidance uses the greater of the existing Euclidean priority and a finite reverse
cost; unavailable estimates fall back to Euclidean priority without pruning.
Frontier scoring is unchanged. Continuous terminal shortcuts mean this is
approximate guidance, not an optimality guarantee. Logs include preprocessing
time, graph storage, nodes and edges. The reproduced initial descent query fell
from budget exhaustion to 23,079 expansions; that is an offline planning result.

Offline audit: `outputs/remote_autonomy/footprint-cost-audit/audit.json` and
`overlay.png`. It compares the same recorded candidates against original and
later retained geometry, and explicitly checks the western manual climb. Neither
map reconstructs the exact earlier planning instant. The western trace still
includes a footprint above the existing 35-degree boundary. These changes do not
establish autonomous ascent, probability calibration, or lightweight performance.

Latest frozen v15 trial: `outputs/remote_autonomy/terrain-guidance-mission-i7cdo28i`.
It autonomously settled at center at 95.32 s, gained at most 1.074 m during ascent,
then stalled. The 10.24 s ZERO segment lost 0.106 m of height; reverse recovery
displaced 0.201 m before its progress watchdog ended the maneuver. At 149.6 s,
the recovery-to-observation handoff hit the directional exclusion guard and
entered HOLD. No manual intervention occurred; the round trip did not complete.
Controller mean/p95 were 0.700/1.055 ms; mean map rebuild was 111.48 ms. These
measurements do not establish lightweight runtime or successful ascent.

### Terrain-only route objective (v17, 2026-10-22)

Manual command histories and demonstrated trajectories are evaluation inputs
only. Runtime route generation uses the observed/surveyed terrain, current pose
and requested destination. There is no demonstrated-route matching, replay or
manual waypoint prior in the planner.

The staged search now prices a complete straight connection to the destination
and queues it alongside other alternatives instead of immediately taking the
first geometrically clear connection. It uses the existing profile costs and
departure-turn charge; the optional unpriced post-search corner merge was
removed. Scalar terminal sampling and multi-cell lattice endpoint charges use
the same cost formula/units but different quadrature. Queue guidance remains
approximate; no optimality claim is made. If exploration exhausts its unchanged
budget, the cheapest already fully checked complete connection may be returned,
with exhaustion and incumbent selection explicitly reported.

The contour profile keeps its stronger continuous slope preference while using
the existing uphill planning limit (33 degrees for remote), instead of banning
every section over the preferred 20 degrees. Contour and shelf suffix searches
always target the actual destination, even when the direct search only reaches
a frontier. Complete onward connectivity, footprint/turn support, 25-degree
cross-slope and 35-degree overall terrain limits remain unchanged. Intermediate
vertices remain pass-through targets; no extra stop or steering action is added.

On the same original seed42 survey, from `(0,0)` to `(0,13)`, complete-route
cost changed from 1077.897 to 1042.844. Floor and exterior connectors improved;
the wall corridor and 3.774 m of exposure above 25 degrees did not change.
A separate, offline-only 192-screen full-circle inventory found essentially
the same corridor, displaced about 8 cm at the rim; it did not establish a
materially different gentle ascent. The original bounded runtime inventory is
retained. No simulator trial or autonomous completion claim accompanies v17.

Verification: 446 tests passed. Final controls and practicality ratings were
both 8.7/10, accepted by the coordinating reviewer. Reproducible comparison,
source snapshots and final review manifest:
`outputs/remote_autonomy/map-objective-audit/`.
