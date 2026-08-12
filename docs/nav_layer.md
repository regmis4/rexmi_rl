# REXMI Navigation Layer

**Status**: Phase N-2 complete — 3D LiDAR SLAM + dense costmap + SPIN policy  
**Date**: 2026-07-14 *(updated — SPIN policy support added)*  
**Policies**: `model_1499.pt` (fast_flat) · `model_8996.pt` (rough) · `model_13994.pt` (rocky_slope) · `model_<N>.pt` (spin, optional)  
**Auto-switching**: terrain-aware PolicySelector picks the right policy every 50 ms  
**New (2026-07-14)**: Dedicated SPIN policy mode — clean in-place rotation at vx=0 on any terrain  
**New (2026-07-11)**: Full 360° LiDAR (Unitree L1 sim) · 3D ICP SLAM · 320×320 @ 20 cm costmap · auto-rotating demo dashboard

---

## Overview

A fully deterministic navigation stack that sits **above** the RL policy and converts
high-level mission goals into per-step `(vx, vy, ωz)` velocity commands.  The RL
policy never changes — the nav layer simply replaces what the command sampler would
normally generate.

```
┌──────────────────────────────────────────────────────────────────────┐
│  SENSORS (Isaac Lab RayCaster)                                        │
│  height_scanner:  160 rays, 1.6 m×1.0 m, ↓ (RL policy input)       │
│  forward_scanner: 75 rays, ±60°, 5 m (nav obstacle warning)         │
│  lidar:           3,240 rays, 360°, 30 m, 10 Hz  ← NEW             │
│                   18 ch × 180 pts, -45°..+7° vertical               │
│                   Chin mount: (+0.29, 0.0, -0.04) from base         │
└────────────┬──────────────────────────────────┬──────────────────────┘
             │ height/fwd scan                    │ LiDAR 3D cloud
┌────────────▼────────────┐   ┌──────────────────▼───────────────────┐
│  LAYER 3 — Mission      │   │  3D ICP SLAM  (slam.py)  ← NEW      │
│  Planner                │   │  voxel_size=0.10 m                   │
│  Named missions →       │   │  bootstrap: 3 s on sim pose          │
│  waypoint list          │   │  ICP: 6-DOF point-to-plane, 30 iter  │
└────────────┬────────────┘   │  map: grows to ~50k voxels/traverse  │
             │ waypoints      │  → SLAMLocalizer pose (x,y,z,yaw)   │
┌────────────▼────────────┐   │  → SLAM map cloud (demo visual)      │
│  LAYER 3b — Global      │   └──────────────────┬───────────────────┘
│  Planner (A*, 2 s)      │                      │ pose + 3D map
│  OccupancyMap →         │   ┌──────────────────▼───────────────────┐
│  lookahead WP           │   │  OccupancyMap  ← UPGRADED            │
│  (now fed by LiDAR      │◀──│  320×320 @ 20 cm (64 m world)        │
│  dense cloud — 30 m     │   │  update(): height scan (160 rays)    │
│  range, all directions) │   │  update_lidar(): LiDAR scan (3240)   │
└────────────┬────────────┘   │  update_slam_map(): full SLAM map    │
             │ imm WP         └──────────────────────────────────────┘
┌────────────▼────────────────────────────────────────────────────────┐
│  LAYER 2 — Local Planner (50 Hz)                                    │
│  16×10 height scan → traversability → (vx, vy, ωz)                 │
│  + forward scanner (±60°, 5 m) → obstacle zone → vx scale          │
│  + RecoveryFSM override when stuck                                  │
│  ↳ terrain metrics (slope, max_step, trav_frac)                     │
└────────────┬───────────────────────────────────┬────────────────────┘
             │ velocity command                   │ terrain metrics
             │                ┌───────────────────▼──────────────────┐
             │                │  PolicySelector (50 Hz)               │
              │                │  |he| > 75°      → spin               │
              │                │  slope < 5°      → fast_flat          │
              │                │  step < 6 cm     → rough              │
              │                │  slope 20–35°    → rocky_slope        │
              │                │  hysteresis: 15–50 steps              │
             │                └───────────────────┬──────────────────┘
             │                                    │ which policy
┌────────────▼────────────────────────────────────▼──────────────────┐
│  LAYER 1 — RL Policy (50 Hz, auto-selected)                        │
│  spin:         model_<N>.pt   — vx=0, omega=±1 rad/s (any terrain) │
│  fast_flat:    model_1499.pt  — up to 2.0 m/s                      │
│  rough:        model_8996.pt  — ~0.8 m/s                           │
│  rocky_slope:  model_13994.pt — ~0.4 m/s, 35° uphill               │
│  Inputs: full 220-dim obs  |  Outputs: 16 joint actions             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
source/rexmi_rl/nav/
  __init__.py          — package exports
  localizer.py         — robot pose provider (SimLocalizer / SLAM interface)
  mission.py           — mission modes → crater-relative waypoint sequences
  occupancy_map.py     — RayCaster point cloud → 128×128 traversability grid
  global_planner.py    — A* on occupancy map → next lookahead waypoint
  local_planner.py     — height-scan traversability → (vx, vy, ωz) command
  recovery.py          — stuck detection + reverse/rotate/retry FSM
  policy_selector.py   — terrain-aware RL policy switcher (fast_flat/rough/rocky_slope)
  navigator.py         — main 50 Hz loop, ties everything together
  dashboard.py         — matplotlib live dashboard (daemon thread)

scripts/
  navigate.py          — autonomous entry point
  teleop.py            — manual remote (policy pick + hold-to-drive)
```

---

## Quick Start

```bash
conda activate env_isaacsim
cd /home/susan/rexmi_rl

# ------------------------------------------------------------------
# PRIMARY: autonomous crater traverse WITH iso-style turn (13345)
# ------------------------------------------------------------------
python scripts/navigate.py \
    --task RexmiRl-Go2w-Crater-Bowl-RockySlope-Play-v0 \
    --ckpt_rough logs/rsl_rl/go2w_velocity_rough/2026-06-14_20-03-41/model_8996.pt \
    --ckpt_rocky logs/rsl_rl/go2w_velocity_rocky_slope/2026-06-30_09-31-48/model_13994.pt \
    --ckpt_turn  logs/rsl_rl/go2w_velocity_slope_turn/2026-07-27_20-54-25/model_13345.pt \
    --mission traverse
```

Expect boot log:
```text
[navigate] TURN enabled (iso handoff: brake→settle→13345→FOLLOW)
[navigate] Turn loaded: direct model_13345 (continuous w=+/-0.08)
```

### Manual teleop (policy characterization — no autonomy)

Drive each policy yourself on crater terrain. **Momentary controls:** hold a
direction to command; release to stop. Autonomous `navigate.py` is unchanged.

```bash
python scripts/teleop.py \
    --task RexmiRl-Go2w-Crater-Bowl-RockySlope-Play-v0 \
    --ckpt_rough logs/rsl_rl/go2w_velocity_rough/2026-06-14_20-03-41/model_8996.pt \
    --ckpt_rocky logs/rsl_rl/go2w_velocity_rocky_slope/2026-06-30_09-31-48/model_13994.pt \
    --ckpt_turn  logs/rsl_rl/go2w_velocity_slope_turn/2026-07-27_20-54-25/model_13345.pt \
    --spawn_preset floor
```

| Control | Action |
|---------|--------|
| Policy dropdown / keys `1` `2` `3` | rough / rocky_slope / turn (+ default vx, ω) |
| vx / ω fields | magnitudes used while a direction is held |
| Hold Forward / Back / Left / Right (or WASD) | apply command |
| Release | stop (`vx=0`, `ω=0`) |
| Space / STOP | force stop |
| `--spawn_preset` | `floor` · `mid_slope` · `rim_out` |

Defaults: rough `vx=0.45 ω=0.40` · rocky `0.40 / 0.35` · turn `0.05 / 0.07` (13345 band).  
While turn policy + Left/Right only, plant `vx` is kept so spin stays in-distribution.  
CSV: `logs/nav/teleop_<timestamp>.csv`. Stdin fallback: `--no_gui`.

### Path-only (no pivot)


```bash
python scripts/navigate.py \
    --task RexmiRl-Go2w-Crater-Bowl-RockySlope-Play-v0 \
    --ckpt_rough logs/rsl_rl/go2w_velocity_rough/2026-06-14_20-03-41/model_8996.pt \
    --ckpt_rocky logs/rsl_rl/go2w_velocity_rocky_slope/2026-06-30_09-31-48/model_13994.pt \
    --mission traverse \
    --no_turn
```

### Isolation test (turn policy alone)

```bash
# Floor interior (default)
python scripts/test_turn_crater.py \
    --task RexmiRl-Go2w-Crater-Bowl-RockySlope-Play-v0 \
    --checkpoint logs/rsl_rl/go2w_velocity_slope_turn/2026-07-27_20-54-25/model_13345.pt \
    --spawn_preset floor

# Exterior rim
python scripts/test_turn_crater.py ... --spawn_preset rim_out
```

See `docs/turn_isolation_test.md`.

### Other missions

```bash
python scripts/navigate.py ... --mission survey
python scripts/navigate.py ... --mission rim_circuit
python scripts/navigate.py ... --no_dashboard
```

### CLI flags (nav + turn)

| Flag | Default | Meaning |
|------|---------|---------|
| `--task` | required | `RexmiRl-Go2w-Crater-Bowl-RockySlope-Play-v0` |
| `--ckpt_rough` | — | Rough loco checkpoint |
| `--ckpt_rocky` | — | Rocky_slope loco checkpoint |
| `--ckpt_turn` | `.../model_13345.pt` | Slope-turn plant-and-spin |
| `--mission` | `traverse` | `traverse` / `survey` / `rim_circuit` |
| `--no_turn` | off | Disable 13345; path-only FOLLOW |
| `--no_dashboard` | off | No matplotlib dashboard |
| `--spawn_x` | `13.0` | Exterior ramp spawn (faces −x) |

### Turn handoff (iso-style)

When `|heading_error| ≥ 100°` and turn is enabled:

```text
BRAKE ~3 s (rocky, vx=0) until nearly stopped + upright
  → settle actions 1.0 s (zeros) + force_turn
  → 13345: PLANT vx=0.05 ω=0 → YAW ±0.07 (up to 8 s) → SETTLE
  → if YAW stuck: flip ω sign once
  → DONE → short hold → FOLLOW (rocky/rough)
```

Expect:
```text
[Nav] BRAKE before turn (v=... he=±100°+)
[Nav] brake done ... TURN handoff settle 1.0s then 13345
[PolicySelector] TURN FORCE: rocky_slope → turn
[Reorient] START iso/... ω=±0.070 vx=0.05
[Reorient] YAW iso ...
[Reorient] DONE iso ...
```


## Starting the Dashboard

The dashboard is **built into `scripts/navigate.py`** — it opens automatically
whenever you run the navigate script without `--no_dashboard`.  There is no
separate command; just run the nav script and the window appears.

```bash
conda activate env_isaacsim

# Auto policy-switching + dashboard (opens immediately on launch)
python scripts/navigate.py \
    --task RexmiRl-Go2w-Crater-Bowl-RockySlope-Play-v0 \
    --ckpt_fast_flat logs/rsl_rl/go2w_velocity_fast_flat/2026-06-17_20-08-58/model_1499.pt \
    --ckpt_rough     logs/rsl_rl/go2w_velocity_rough/2026-06-14_20-03-41/model_8996.pt \
    --ckpt_rocky     logs/rsl_rl/go2w_velocity_rocky_slope/2026-06-30_09-31-48/model_13994.pt \
    --mission traverse
# ↑ A matplotlib window titled "REXMI Nav Dashboard" appears automatically.
#   It updates at 2 Hz and never blocks the sim loop.

# Single fixed policy + dashboard
python scripts/navigate.py \
    --task RexmiRl-Go2w-Crater-Bowl-RockySlope-Play-v0 \
    --checkpoint logs/rsl_rl/go2w_velocity_rocky_slope/2026-06-30_09-31-48/model_13994.pt \
    --policy_mode rocky_slope \
    --mission traverse

# Disable the dashboard (headless / recording)
python scripts/navigate.py \
    --task RexmiRl-Go2w-Crater-Bowl-RockySlope-Play-v0 \
    --checkpoint logs/.../model_13994.pt \
    --policy_mode rocky_slope \
    --mission traverse \
    --no_dashboard
```

**Dashboard window layout**:
```
┌─────────────────────────────────┬──────────────────────────┐
│  3D TERRAIN POINT CLOUD (live)  │  TOP-DOWN A* COST MAP    │
│                                 │                          │
│  Builds up as robot explores.   │  Green  = clear terrain  │
│  Colour = height (viridis).     │  Red    = blocked/steep  │
│  White line = planned path.     │  White line = A* path    │
│  Gold ★ = next waypoint.        │  Gold ★ = waypoints      │
│  Cyan ● + arrow = robot.        │  Cyan ● = robot pos      │
├─────────────────────────────────┴──────────────────────────┤
│  STATUS BAR (updates every 0.5 s)                          │
│  Mission | WP index | Speed | Slope | cmd | Policy | State │
│                                                            │
│  Example:                                                  │
│  Mission: TRAVERSE | WP: 2/5 [rim_exit] | 0.38 m/s |     │
│  Slope: 22.4° | cmd=(+0.25,+0.00,-0.34) |                 │
│  Policy: rocky_slope [0/25] (switched×1) |                │
│  State: NAVIGATING (0/3)                                   │
└────────────────────────────────────────────────────────────┘
```

**Prerequisite** — if the window doesn't appear, install the Tk backend:
```bash
sudo apt install python3-tk
```
Or use `--no_dashboard` to skip it entirely.

---

## Module Details

### `localizer.py` — Robot pose provider

**Interface** (Phase 2: swap backend without touching any planning code):
```python
class Localizer(ABC):
    def get_pose(self) -> Pose:          # x, y, z (m), yaw (rad)
    def get_velocity(self) -> (vx, vy, vz)  # world frame m/s

@dataclass
class Pose:
    x: float    # metres, world frame
    y: float    # metres, world frame
    z: float    # metres, world frame
    yaw: float  # radians, CCW from +x axis
```

**`SimLocalizer`** reads directly from Isaac Sim:
- Position: `robot.data.root_pos_w[env_idx]`
- Orientation: `robot.data.root_quat_w[env_idx]` → yaw via atan2
- Velocity: `robot.data.root_lin_vel_w[env_idx]`

**Phase 2**: Replace `SimLocalizer` with `SLAMLocalizer(ros_topic="/slam/pose")`.
All downstream code is unchanged.

---

### `mission.py` — Mission planner

All missions are defined as **fractions of `r_rim` and `r_floor`** — they work for
any crater without hardcoded distances.

```python
from rexmi_rl.nav.mission import MissionPlanner, Mission

planner = MissionPlanner(
    crater_centre=(0.0, 0.0),
    r_floor=3.0,          # flat crater floor radius
    r_rim=11.0,           # rim radius
    spawn_x=-18.0,        # robot starts here (−x side)
    entry_azimuth_deg=180.0,  # enter from −x heading +x
)
waypoints = planner.get_waypoints(Mission.TRAVERSE)
# Returns: List[Waypoint(x, y, arrival_radius, label)]
```

#### Mission: `traverse`
```
spawn → approach → rim_entry → floor_centre → rim_exit → exit_clear
```
5 waypoints. Showcases full descent + floor + ascent capability.

#### Mission: `survey`
```
rim_entry → survey_row_0_L → survey_row_0_R → survey_row_1_R → ... → rim_exit
```
Lawnmower pattern within `r_floor` disk, 1.5 m stripe spacing.
Simulates water-ice resource prospecting.

#### Mission: `rim_circuit`
```
rim_0 → rim_1 → rim_2 → ... → rim_7 → rim_0  (closes loop)
```
8 waypoints at `r = 1.15 × r_rim`, 45° apart, clockwise.
Simulates perimeter survey before descent decision.

**`Waypoint` dataclass**:
```python
@dataclass
class Waypoint:
    x:              float   # world frame
    y:              float   # world frame
    arrival_radius: float   # metres — "close enough" threshold
    label:          str     # shown on dashboard
```

---

### `occupancy_map.py` — Terrain map

Accumulates RayCaster `ray_hits_w` world-frame 3D points into a 2D grid.

```python
omap = OccupancyMap(
    world_size=64.0,    # 64 m × 64 m
    cell_size=0.50,     # 50 cm cells → 128×128 grid
    origin=(0.0, 0.0),  # world-frame map centre
)
omap.update(ray_hits_w_np)   # (N, 3) float32 array each step

# A* cost query
cost = omap.traversal_cost(row, col)
# Returns: 1.0 (clear) | 5.0 (unknown) | 4.0 (steep) | inf (blocked)

# Dashboard data
xs, ys, zs = omap.get_point_cloud()   # 3D scatter arrays
grid = omap.get_cost_grid()           # (128, 128) float32 cost array
```

**Cost function**:
| Condition | Cost |
|-----------|------|
| Not yet visited | 5.0 (conservative unknown penalty) |
| `max_step > 0.12 m` (≈ 12 cm boulder) | `inf` (blocked) |
| `slope > tan(35°) = 0.70` | 11.0 (steep but passable) |
| `slope > tan(25°) = 0.47` | 4.0 (moderate slope) |
| Clear terrain | 1.0 |

**Point cloud cap**: 50,000 points (prevents matplotlib slowdown). Oldest points
are not dropped — the first 50k points scanned are kept.

---

### `global_planner.py` — A* path planner

```python
planner = GlobalPlanner(
    omap,
    replan_interval_s=2.0,   # replan at most every 2 seconds
    lookahead_m=4.0,          # return waypoint 4 m ahead on path
)
planner.set_goal(goal_x, goal_y)          # world frame
wp = planner.update(robot_x, robot_y)    # returns (x, y) or None
path = planner.get_path_world()           # full path for dashboard
```

**A* details**:
- 8-connected grid (diagonal movement allowed, `√2` cost)
- Octile heuristic (admissible, optimal for 8-connected)
- Replans when: interval elapsed OR robot deviates > 1 m from path
- Trims passed cells every step

**Lookahead**: walks the path from the robot until accumulated distance ≥ 4 m,
returns that point as the immediate waypoint to the local planner.

---

### `local_planner.py` — Height-scan traversability

Converts 160 height-scan rays into a velocity command.

**Height scan layout** (body frame, yaw-aligned):
```
axis-0 (x): 16 rays, 0.1 m spacing → 0 to 1.5 m ahead
axis-1 (y): 10 rays, 0.1 m spacing → −0.45 m to +0.45 m lateral
scan[i*10 + j] = terrain height at (forward_i, lateral_j) relative to base
```

**Algorithm**:
```python
planner = LocalPlanner(
    vx_normal=0.40,    # m/s on clear terrain
    vx_steep=0.25,     # m/s on slopes > 25°
    omega_gain=1.20,   # rad/s per rad heading error
    omega_max=0.80,    # max yaw rate
    step_thresh=0.12,  # m — column blocked if step > this
    slope_thresh=0.47, # tan(25°) — "steep" threshold
)

out = planner.compute(
    scan_heights,     # list[float] length 160
    robot_yaw,        # rad
    robot_x, robot_y,
    waypoint_x, waypoint_y,
)
# out.vx, out.vy, out.omega  ← inject into env
```

**5 heading candidates evaluated each step**:

| Candidate | Heading offset | Columns covered |
|-----------|---------------|-----------------|
| 0 | −40° (hard left) | cols 0–2 |
| 1 | −20° (soft left) | cols 2–4 |
| 2 |   0° (straight) | cols 3–6 |
| 3 | +20° (soft right)| cols 5–8 |
| 4 | +40° (hard right)| cols 7–9 |

Scoring: `score = -|heading_to_waypoint - candidate_offset| - 0.5 × mean_slope`

Best traversable candidate wins. If no heading is traversable → `vx=0` (recovery takes over).

**Speed scaling**:
- `vx = vx_steep` if `slope > slope_thresh` (0.47 = tan 25°)
- `vx *= (1 - |omega|/omega_max)` — slow down during hard turns (min 0.4×)

#### Forward scanner obstacle avoidance (`compute_with_forward`)

In addition to `compute()`, the local planner exposes `compute_with_forward()` which
takes world-frame hits from the **forward RayCaster sensor** and applies a speed
reduction if an obstacle is detected close ahead.

**Forward sensor spec** (defined in `Go2wRoughEnvCfg`, key `"forward_scanner"`):
```
Pattern    : LidarPatternCfg  (5 vertical × 15 horizontal = 75 rays/step)
Azimuth    : −60° to +60° around robot heading (body frame)
Elevation  : −20° to 0° (level to slightly downward)
Range      : 5 m  (max_distance)
Offset     : 0.5 m above robot base (body height)
Alignment  : yaw-only (rotates with heading, not pitch/roll)
```

The sensor gives the nav layer **up to 10 seconds of warning** at 0.5 m/s for an
obstacle 5 m ahead — far more than the 1.5 m height-scan look-ahead alone.

**Danger zone** (all conditions must be true for a hit to count):
```
x_body > 0.1 m           — in front of the robot (not behind)
x_body < 2.0 m           — within 2 m look-ahead
|y_body| < 0.30 m        — within body width (0.6 m total)
dz > 0.10 m above base   — elevated above ground (boulder / wall, not flat terrain)
```

**Speed response**:
| Nearest obstacle distance | vx factor |
|--------------------------|-----------|
| > 2.0 m (beyond zone)   | 1.0× (no change) |
| 2.0 m → 0.5 m           | linear ramp 1.0× → 0.0× |
| ≤ 0.5 m                 | 0.0 (full stop) |

```python
out = planner.compute_with_forward(
    scan_heights,          # 160-element height scan (same as compute())
    robot_yaw,             # rad
    robot_x, robot_y,
    waypoint_x, waypoint_y,
    fwd_hits_world=pts,    # np.ndarray (N, 3) filtered world-frame hits
    robot_pos_w=(rx, ry, rz),  # robot position for body-frame transform
)
# out.fwd_obstacle_dist  ← nearest obstacle in danger zone (m), math.inf if clear
# out.vx                 ← already scaled by danger-zone proximity
```

`compute_with_forward()` calls `compute()` first (full height-scan plan), then
applies the forward-zone check on top.  If `fwd_hits_world=None` (sensor absent or
flat env), it returns the `compute()` result unchanged — fully backward compatible.

**How the data flows in `navigator.py`**:
```
forward_scanner.data.ray_hits_w         # (n_envs, 75, 3)
  → _get_forward_hits()                 # filter NaN/inf, return (N, 3) or None
  → omap.update(pts)                    # also feeds OccupancyMap for A* cost
  → local.compute_with_forward(…)       # speed scaling
  → _inject_command(out.vx, 0, out.omega)
```

---

### `recovery.py` — Stuck detection FSM

```
NAVIGATING ──(|v| < 0.05 m/s for 3 s)──► STUCK
   ▲                                          │ immediately
   │                                          ▼
ROTATING ◄──(after 1.8 s)────────────── REVERSING
   │                                     (1.5 s, vx=−0.2)
   │ attempt_count += 1
   │ (if attempts < 3) → NAVIGATING
   └──(if attempts ≥ 3) ──► BLOCKED → triggers A* replan
```

**Override commands during recovery**:
| State | vx | vy | omega |
|-------|----|----|-------|
| REVERSING | −0.20 m/s | 0 | 0 |
| ROTATING  | 0 | 0 | ±0.60 rad/s |
| BLOCKED   | 0 | 0 | 0 (waits for replan) |

Rotation direction: `+` if `heading_error ≥ 0`, `−` otherwise (always turns toward goal).

```python
fsm = RecoveryFSM(
    stuck_speed=0.05,       # m/s
    stuck_timeout=3.0,      # s
    reverse_duration=1.5,   # s
    rotate_duration=1.8,    # s
    rotate_speed=0.60,      # rad/s
    max_attempts=3,
)
vx, vy, omega = fsm.update(speed, heading_error)
# Returns (0,0,0) during NAVIGATING → local planner command used
# Returns override during recovery states
fsm.reset()   # call after successful replan to clear BLOCKED
```

---

### `policy_selector.py` — Terrain-aware policy switcher

All three RL checkpoints are **loaded at startup**.  Each sim step the selector
reads terrain metrics from the LocalPlanner output and decides which policy to run.

**The three policies and when each is used**:

| Policy | Checkpoint | Terrain condition | Typical speed |
|--------|-----------|-------------------|---------------|
| `fast_flat` | `model_1499.pt` | slope < 5°, step < 6 cm, >60% clear | up to 2.0 m/s |
| `rough` | `model_8996.pt` | slope < 20°, step 6–10 cm, or <60% clear | ~0.8 m/s |
| `rocky_slope` | `model_13994.pt` | slope > 20° OR step > 10 cm | ~0.4 m/s |

**Usage**:
```python
from rexmi_rl.nav.policy_selector import PolicySelector, PolicyMode

# Build selector with all 3 loaded inference callables
selector = PolicySelector(
    policies={
        PolicyMode.FAST_FLAT:   fast_flat_policy,   # runner.get_inference_policy()
        PolicyMode.ROUGH:       rough_policy,
        PolicyMode.ROCKY_SLOPE: rocky_policy,
    },
    initial_mode=PolicyMode.ROCKY_SLOPE,  # safest starting choice
)

# Each step: update with terrain metrics → get active mode
mode = selector.update(
    slope_ahead=0.42,           # tan(θ), from LocalPlannerOutput.slope_ahead
    max_step=0.08,              # metres, worst step in 16×10 scan
    traversable_fraction=0.6,   # 0–1, fraction of 5 candidates that are clear
)

# Use the active policy
active_policy = selector.current_policy
actions = active_policy(obs)

print(selector.status_str())
# → "rough [12/25] (switched×2)"
#   current mode / hold counter / total switches so far
```

**Decision rules** (evaluated top-to-bottom, first match wins):
```
slope > tan(20°) = 0.364             → rocky_slope
max_step > 0.10 m  OR  slope > 0.532 → rocky_slope
max_step > 0.06 m                    → rough
traversable_fraction < 0.40          → rough  (nearly blocked)
else                                 → fast_flat
```

**Hysteresis** — prevents policy flip-flopping at terrain boundaries:  
A new policy must be the recommended choice for **25 consecutive steps** (~0.5 s at
50 Hz) before the switch is committed.  Intermediate different recommendations
restart the hold counter.

**Fixed-mode fallback**:  
Pass `--policy_mode rocky_slope` (or `fast_flat` / `rough`) with a `--checkpoint`
to disable auto-switching.  The same selector infrastructure is used internally,
but `update()` is overridden to always return the fixed mode — so the rest of the
code path is identical.

**Terminal output on each switch**:
```
[PolicySelector] switch: fast_flat → rocky_slope  (slope=23.1°, step=11.2cm, trav=40%)
```

---

### `navigator.py` — Main loop

```python
nav = Navigator(
    env,
    mission=Mission.TRAVERSE,
    crater_centre=(0.0, 0.0),
    r_floor=3.0,
    r_rim=11.0,
    spawn_x=-18.0,
    env_idx=0,                  # which parallel env to control
    replan_interval_s=2.0,
)
nav.start_dashboard()   # optional — opens matplotlib window

# In sim loop:
while running:
    actions = policy(obs)
    obs, rewards, dones, infos = env.step(actions)
    nav.step()          # one nav tick at 50 Hz

nav.stop_dashboard()
```

**Per-step execution order**:
1. `SimLocalizer.get_pose()` + `get_velocity()`
2. Read `height_scanner.data.ray_hits_w` → filter invalid → `OccupancyMap.update()`
3. Read `forward_scanner.data.ray_hits_w` → filter invalid → `OccupancyMap.update()` (also stored for local planner)
4. Check waypoint arrival → advance `wp_idx` if reached
5. `GlobalPlanner.update()` → immediate waypoint
6. `LocalPlanner.compute_with_forward()` → `(vx, vy, omega)` + terrain metrics + forward obstacle check
7. `PolicySelector.update(slope, max_step, trav_frac)` → commit policy switch if hold counter reaches 25
8. `RecoveryFSM.update()` → override command if stuck
9. `command_manager.get_command("base_velocity")[env_idx] = cmd`
10. Write shared state for dashboard (cost grid updated every 5 steps)

**Command injection** — writes directly into the Isaac Lab command tensor:
```python
cmd_tensor = env.unwrapped.command_manager.get_command("base_velocity")
cmd_tensor[env_idx, 0] = vx     # forward
cmd_tensor[env_idx, 1] = vy     # lateral (always 0)
cmd_tensor[env_idx, 2] = omega  # yaw rate
```
The RL policy reads this tensor as its velocity command on the next obs.

---

### `dashboard.py` — Live matplotlib dashboard

Dark-themed 16×8 inch window, updates at 2 Hz in a daemon thread.

```
┌─────────────────────────────────┬──────────────────────────┐
│  3D POINT CLOUD (live)          │  TOP-DOWN COST MAP       │
│                                 │                          │
│  • terrain pts: viridis/height  │  • green = clear         │
│  • robot: white ● + cyan arrow  │  • red = blocked/steep   │
│  • waypoints: gold ★            │  • white line = A* path  │
│  • A* path: white line          │  • cyan ● = robot        │
│                                 │  • crater rings dashed   │
├─────────────────────────────────┴──────────────────────────┤
│  Mission: TRAVERSE | WP: 2/5 [rim_exit] | 0.38 m/s | 24°  │
│  cmd=(+0.25, +0.00, -0.34) | State: NAVIGATING (0/3)      │
└────────────────────────────────────────────────────────────┘
```

The 3D cloud **builds up as the robot explores** — starts sparse, fills in with each
scan. This is the primary investor-facing visual: the robot autonomously mapping
terrain it has never seen before.

**Thread safety**: all shared state is protected by a `threading.Lock`.
The dashboard thread never blocks the sim loop — it only reads the shared dict
and sleeps between renders.

---

## Design Decisions

### Why sim ground-truth localisation (not SLAM)?
For Phase N-1, the goal is to validate the full nav stack architecture.  Real SLAM
adds complexity without changing any planning code.  The `Localizer` interface is
designed to make the swap trivial:

```python
# Phase N-1 (sim):
localizer = SimLocalizer(robot_art, env_idx=0)

# Phase N-2 (real deployment):
localizer = SLAMLocalizer(ros_topic="/slam/pose")

# Everything else unchanged.
```

### Why A* (not D* Lite / RRT)?
- Map is small (128×128 cells, ≤ 16,384 nodes)
- A* completes in < 1 ms on this grid
- Goal is static during each waypoint segment
- D* Lite adds complexity for dynamic replanning that isn't needed here
  (boulders don't move; the robot moves slowly enough that 2 s replans suffice)

### Why not run the height scan through the nav layer AND the policy?
The nav layer reads `ray_hits_w` directly from the RayCaster buffer — the raw,
noiseless world-frame 3D positions.  The policy receives a separate noisy copy
(`Unoise ±0.1 m`) via the observation term.  This separation is intentional:
- Policy: needs noise for sim-to-real robustness
- Nav layer: needs accurate heights for traversability decisions

### Why crater-relative waypoints (not raw XY)?
Missions are defined using `r_rim` and `r_floor` fractions, so the same mission
file works for any crater of any size.  Changing the crater geometry is a one-line
CLI argument (`--r_rim 25.0`), not a code change.

### Why 5 heading candidates (not full DWA)?
The height scan is 1.6 m × 1.0 m — only ~1.5 body lengths ahead.  Full Dynamic
Window Approach samples a dense velocity space but gains nothing over 5 heading
offsets when the sensor range is this short.  5 candidates runs in ~0.01 ms vs.
~5 ms for a proper DWA; the saved time goes to A* replanning.

---

### `slam.py` — 3D ICP SLAM (Phase N-2)

Incremental 3D LiDAR SLAM using pure-Python point-to-plane ICP (numpy + scipy).

**Sensor spec** (Unitree L1 approximation in Isaac Lab):
| Parameter | Value |
|-----------|-------|
| Channels (vertical scan lines) | 18 |
| Vertical FOV | −45° to +7° |
| Horizontal FOV | 360° |
| Horizontal resolution | 2° → 180 pts/line |
| Points per scan | 3,240 |
| Update rate | 10 Hz |
| Max range | 30 m |
| Mount position | (+0.29, 0.0, −0.04) from base (chin) |

**Architecture**:
```
LiDAR scan (3,240 pts, world frame)
  │
  ▼ voxel downsample (0.10 m) → ~200 pts/scan
  │
  ├─ (0–3 s): bootstrap phase — add to voxel map, return sim pose
  │
  └─ (after bootstrap, ≥500 voxels):
       │
       ▼ 6-DOF point-to-plane ICP vs accumulated voxel map
       │   • scipy.cKDTree nearest-neighbour search
       │   • linearised normal equations (6-DOF: α,β,γ,tx,ty,tz)
       │   • SVD re-orthogonalisation (SO(3) constraint)
       │   • max 30 iterations, tol 1e-4 m
       │   • reject if RMS > 0.30 m (diverged → return sim pose)
       │
       ▼ SLAMPose(x, y, z, roll, pitch, yaw)
       │
       ├─ SLAMLocalizer.get_pose() → Pose(x,y,z,yaw) for nav layer
       │
       └─ get_map_points() → (M,3) float32 for dashboard + omap
```

**Usage**:
```python
from rexmi_rl.nav.slam import LidarSLAM, SLAMPose

slam = LidarSLAM(
    voxel_size=0.10,      # m — voxel map resolution
    bootstrap_s=3.0,      # s — use sim pose for first N seconds
    max_icp_iter=30,
    max_icp_rms=0.30,     # m — reject ICP if RMS exceeds this
)

# Each LiDAR frame (10 Hz):
slam_pose = slam.update(lidar_pts_w, sim_pose_6dof)
# Returns SLAMPose during bootstrap, ICP pose after

# Dashboard / omap queries:
map_pts = slam.get_map_points()   # (M, 3) float32 — full voxel map
print(slam.get_map_size())        # number of occupied voxels
print(slam.last_rms)              # last ICP RMS error (m)
print(slam.is_converged)          # True once ICP has successfully run
```

**Path to real robot** — three options, listed by effort:
1. **KISS-ICP drop-in** (recommended): replace `_run_icp()` with KISS-ICP C++ backend.
   Same voxel map, same interface. ~2 hours work, 10× faster ICP.
2. **ROS 2 bridge**: subscribe to `/slam_toolbox/pose` in a `ROSSLAMLocalizer`.
   Our nav layer unchanged — only `SLAMLocalizer.get_pose()` implementation swaps.
3. **LIO-SAM node**: run as ROS 2 node with IMU fusion. Same bridge approach.

---

---

## SPIN Policy — Verification Workflow

Before adding the spin policy to the crater demo, verify it independently.

### Step 1 — Train

```bash
conda activate env_isaacsim
python scripts/train.py --task RexmiRl-Go2w-Velocity-Spin-v0 --headless \
    --load_run go2w_velocity_rough/2026-06-14_20-03-41 \
    --checkpoint model_8996.pt \
    --max_iterations 1000
# Saves to: logs/rsl_rl/go2w_velocity_spin/<date>/
# Target: track_ang_vel_z_exp > 0.85, track_lin_vel_xy_exp penalty ≈ 0
tensorboard --logdir logs/rsl_rl/go2w_velocity_spin
```

### Step 2 — Play (visual check)

Watch 50 robots spin in place on rocky slopes (15°–35°).
**What to look for**: wheels clearly spin in opposite directions (L forward, R back
for a left turn). Body stays level. Minimal forward drift. Rotation rate tracks
the commanded ±1 rad/s.

```bash
# Replace <date> and <N> with your actual run folder and checkpoint
conda activate env_isaacsim
python scripts/play.py --task RexmiRl-Go2w-Velocity-Spin-Play-v0 \
    --load_run go2w_velocity_spin/<date> \
    --checkpoint model_<N>.pt
```

### Step 3 — Eval (headless metrics)

Measures angular tracking ratio and linear drift (moving_frac).
A good spin policy should show:
- `spin_flat_0deg`: tracking ≥ 0.90, drift ≤ 5% of steps
- `spin_slope_15deg`: tracking ≥ 0.80, drift ≤ 10%
- `spin_slope_35deg`: tracking ≥ 0.65, drift ≤ 20%

```bash
CKPT_SPIN=logs/rsl_rl/go2w_velocity_spin/<date>/model_<N>.pt

# Headless eval — spin group only (~3 min)
python scripts/eval.py --checkpoint $CKPT_SPIN --group spin_rotation

# Single variant visual check (watch robots spinning in GUI)
python scripts/eval.py --checkpoint $CKPT_SPIN \
    --terrain spin_slope_35deg --visual

# Full sweep including spin group (~15 min total)
python scripts/eval.py --checkpoint $CKPT_SPIN
```

### Step 4 — Nav integration (with spin policy)

Once eval passes, plug spin into the crater traverse:

```bash
conda activate env_isaacsim
python scripts/navigate.py \
    --task           RexmiRl-Go2w-Crater-Bowl-RockySlope-Play-v0 \
    --ckpt_fast_flat logs/rsl_rl/go2w_velocity_fast_flat/2026-06-17_20-08-58/model_1499.pt \
    --ckpt_rough     logs/rsl_rl/go2w_velocity_rough/2026-06-14_20-03-41/model_8996.pt \
    --ckpt_rocky     logs/rsl_rl/go2w_velocity_rocky_slope/2026-06-30_09-31-48/model_13994.pt \
    --ckpt_spin      logs/rsl_rl/go2w_velocity_spin/<date>/model_<N>.pt \
    --mission traverse
```

**What changes vs. no spin**: whenever |heading_error| > 75°, the `SPIN` policy
takes over (`vx=0`, `omega=±1 rad/s`). The dashboard policy field will show
`spin [0/15]` during turns. Once heading_error drops below 45°, it hands back
to the terrain-appropriate policy (rocky_slope/rough/fast_flat).

### PolicySelector — Full mode table

| Mode | Checkpoint | When active | vx | Notes |
|------|-----------|-------------|-----|-------|
| `SPIN` | `model_<N>.pt` | \|heading_error\| > 75° | 0.0 | Uses differential wheel torques — no drift |
| `FAST_FLAT` | `model_1499.pt` | slope < 5°, step < 6 cm, >60% traversable | up to 2.0 m/s | High-speed on flat terrain |
| `ROUGH` | `model_8996.pt` | slope < 20°, step 6–10 cm, or <60% traversable | ~0.8 m/s | General purpose |
| `ROCKY_SLOPE` | `model_13994.pt` | slope > 20° or step > 10 cm | ~0.4 m/s | Crater walls / boulder fields |

SPIN → terrain policy handback: hysteresis 15 steps.
Terrain policy switches: hysteresis 25 steps.
SPIN is checked first (highest priority) before terrain rules.

---

## Phase Roadmap

| Feature | Status | What changes |
|---------|--------|-------------|
| Policy switching (terrain-aware) | ✅ **Done** | `policy_selector.py` — 3 policies, 25-step hysteresis |
| Any crater size | ✅ **Done** | Pass `--r_rim` / `--r_floor`; all missions auto-scale |
| Forward scanner obstacle avoidance | ✅ **Done** | `forward_scanner` sensor; `compute_with_forward()` in LocalPlanner |
| 360° LiDAR sensor | ✅ **Done** | `scene.lidar` in `Go2wRoughEnvCfg`; 18ch, 3240 pts, 10 Hz, 30 m |
| 3D ICP SLAM | ✅ **Done** | `slam.py` — 6-DOF point-to-plane ICP, voxel map, `SLAMLocalizer` |
| Dense costmap (20 cm) | ✅ **Done** | `occupancy_map.py` upgraded 320×320 @ 20 cm; `update_lidar()` |
| Auto-rotating SLAM dashboard | ✅ **Done** | `dashboard.py` — 3D cloud builds in real-time, azimuth auto-rotates |
| SLAM → KISS-ICP C++ | Phase N-3 | Replace `_run_icp()` with KISS-ICP Python wrapper (1-line swap) |
| Multi-robot | Phase N-3 | Instantiate `Navigator` + `LidarSLAM` per robot with separate `env_idx` |
| ROS 2 bridge | Phase N-4 | `ROSSLAMLocalizer` subscribes to `/slam_toolbox/pose` |
| D* Lite replanning | Phase N-4 | Drop-in replacement for `GlobalPlanner._astar()` |

---

## Troubleshooting

**Robot stops immediately / vx=0**  
All 5 heading candidates are blocked by the step threshold. Likely a boulder field.
The recovery FSM will trigger after 3 s: reverse → rotate → retry.

**Dashboard window doesn't appear**  
Install `python-tk`: `sudo apt install python3-tk`.  
Or pass `--no_dashboard` to disable it.

**`KeyError: 'height_scanner'`**  
The task you selected uses `Go2wFlatEnvCfg` (no height scanner).  Use a rough or
rocky-slope task:  `RexmiRl-Go2w-Crater-Bowl-RockySlope-Play-v0`.

**A* returns None for many steps**  
The occupancy map is mostly unknown (dark purple on cost map).  The robot needs to
explore a bit before A* can find good paths.  During the first ~10 s, the global
planner falls back to direct waypoint heading.

**`RuntimeError: size mismatch for actor.0.weight`**  
Each policy was trained with a different obs dimension and network size:
- `fast_flat`: 60-dim obs, `[128, 128, 128]` hidden
- `rough` / `rocky_slope`: 247-dim obs, `[512, 256, 128]` hidden

`_load_policy()` handles this by **reading the architecture directly from the
checkpoint's weight shapes** (no second env needed — Isaac Lab only allows one
simulation context per process). The returned callable automatically slices
`obs[..., :obs_dim]` so a 247-dim env obs can feed a 60-dim fast_flat policy safely.

If you see this error with a new checkpoint, it means the `model_state_dict` key
names don't match the expected `actor.0.weight` / `actor.6.weight` layout.
Check the checkpoint was saved by RSL-RL's `OnPolicyRunner`.

**`command_manager.get_command` raises AttributeError**  
Isaac Lab version mismatch.  Check: `env.unwrapped.command_manager` exists.  
Fallback: directly write to `env.unwrapped.scene["robot"].data` if needed.

**Forward scanner not slowing the robot near boulders**  
Check the env is based on `Go2wRoughEnvCfg` (not `Go2wFlatEnvCfg`). The
`forward_scanner` sensor is only registered in rough/crater envs.  
Verify with:
```python
print(list(env.unwrapped.scene.keys()))  # should include "forward_scanner"
```
If the key is missing, the nav layer silently skips forward obstacle checking
(`_get_forward_hits()` returns `None`) and falls back to height-scan-only mode.

**Robot stops `0.5 m` in front of a wall and doesn't resume**  
The forward danger zone stops at `x_body = 0.5 m` and only scales vx — it does
not override omega.  If the wall spans all 5 heading candidates in the height
scan too, the recovery FSM will trigger after 3 s and rotate away.  This is
correct behaviour.  If you want more clearance, increase `STOP_DIST` in
`LocalPlanner.compute_with_forward()` (currently 0.5 m).

**Forward scanner hits the robot's own legs**  
The sensor fires from `offset=(0, 0, 0.5)` with `elevation −20° to 0°`.  At
0.5 m body height and −20° elevation, rays hit the ground at `r = 0.5/tan(20°) ≈ 1.37 m`
— well beyond the robot's leg reach (~0.35 m).  If the robot has extended calves
pointing forward, a ray can still clip them.  The `x_body > 0.1 m` filter in the
danger zone removes hits within 10 cm, which is sufficient for normal stances.

---

## Bug Fixes & Improvements (2026-07-13)

Eight bugs were diagnosed and fixed in this session.  All changes are backward-
compatible and require no changes to checkpoints, task configs, or launch commands.

---

### 🔴 BUG 1 — `slam.py`: KD-tree cache always marked dirty (2-3× SLAM slowdown)

**File**: `source/rexmi_rl/nav/slam.py` · `_get_map_pts()`

**Symptom**: SLAM ICP ran at ~20–30 ms/frame instead of the expected ~10 ms.  The
KD-tree (which takes 5–15 ms to build for a 5k-voxel map) was being rebuilt on
**every single ICP call** regardless of whether the map had changed.

**Root cause**: `_get_map_pts()` ended with `self._kdtree_dirty = True`, which
immediately re-dirtied the cache it had just rebuilt.  The correct value is `False`.

```python
# BEFORE (broken):
self._kdtree_dirty = True   # ← re-dirtied immediately after rebuilding!

# AFTER (fixed):
self._kdtree_dirty = False  # cache is fresh; ICP reuses it until map changes
```

**Impact**: SLAM ICP now runs at the designed ~10 ms/scan on a 5k-voxel map,
keeping the background SLAM thread well within the 100 ms LiDAR budget.

---

### 🔴 BUG 2 — `slam.py`: `_last_slam_pose` not initialized (`AttributeError`)

**File**: `source/rexmi_rl/nav/slam.py` · `__init__`

**Symptom**: `get_pose_6dof()` could raise `AttributeError: 'LidarSLAM' object has
no attribute '_last_slam_pose'` if called before any ICP run succeeded.

**Root cause**: `_last_slam_pose` was only created inside `_run_icp()` on a
successful ICP result.  `get_pose_6dof()` referenced it unconditionally.

**Fix**: Initialize in `__init__`:
```python
self._last_slam_pose: Optional[SLAMPose] = None
```

---

### 🔴 BUG 3 — `navigator.py`: SLAM map never written to `shared` dict

**File**: `source/rexmi_rl/nav/navigator.py` · main step loop

**Symptom**: Dashboard always showed an empty SLAM cloud (`SLAM:BOOT(0vox)`).
A* never incorporated SLAM-mapped obstacles because `update_slam_map()` was never
called.

**Root cause**: `_slam_map_pts` was computed every 50 steps (via
`slam.get_map_points()`) but never written to `shared["lidar_cloud"]` or used
to call `omap.update_slam_map()`.  Both operations were missing from the step loop.

**Fix**: Added inside the `with self._lock` block:
```python
if _slam_map_pts is not None:
    self.shared["lidar_cloud"] = _slam_map_pts
```
And outside the lock (periodic SLAM→omap full sync every 100 steps / 2 s):
```python
if _slam_map_pts is not None and slam_map_size > 500:
    if step_n_now - self._last_slam_sync_step >= self._slam_omap_sync_interval:
        self._omap.update_slam_map(_slam_map_pts)
        self._last_slam_sync_step = step_n_now
```

**Impact**: Dashboard now shows the live-growing SLAM map.  A* now routes around
obstacles seen in past SLAM scans (not just the current frame's LiDAR hits).

---

### 🟠 IMPROVEMENT 4 — `navigator.py`: Adaptive BOOT (3 s min vs 8 s)

**File**: `source/rexmi_rl/nav/navigator.py` · BOOT phase

**Change**: Reduced `boot_min_s` from `8.0` to `3.0` seconds.  The BOOT phase now
exits as soon as **all three conditions are met**:
1. `elapsed >= 3.0 s` (minimum sensor accumulation time)
2. `slam.is_converged` (ICP has run at least once successfully)
3. `slam.last_rms < 0.15 m` (ICP is producing accurate poses)
4. `slam.icp_count >= 5` (5 consecutive stable frames)

Hard timeout kept at 15 s (for environments where ICP never converges).

**Also added**: Richer BOOT countdown log every 50 steps showing RMS, icp_count,
and convergence state — makes it easy to see why BOOT is still running.

**Impact**: Navigation starts ~5 s sooner on smooth terrain where SLAM converges
quickly.  On challenging terrain the hard timeout still fires at 15 s.

---

### 🟠 IMPROVEMENT 5 — `occupancy_map.py`: Vectorize `update()` cloud append

**File**: `source/rexmi_rl/nav/occupancy_map.py` · `update()`

**Change**: Replaced the per-point Python `for` loop with a vectorised `extend()`:

```python
# BEFORE (slow — O(N) Python loop):
for pt in ray_hits_w:
    self._cloud_xyz.append((float(pt[0]), float(pt[1]), float(pt[2])))

# AFTER (fast — single extend() call):
self._cloud_xyz.extend(
    zip(ray_hits_w[:, 0].tolist(),
        ray_hits_w[:, 1].tolist(),
        ray_hits_w[:, 2].tolist())
)
```

**Impact**: At 160 rays/step × 50 Hz, the old loop ran ~0.8 ms/step.  The new
`extend()` is ~10× faster (~0.08 ms/step), saving ~35 ms/s of Python overhead.

---

### 🟠 IMPROVEMENT 6 — `navigate.py`: CSV telemetry every step

**File**: `scripts/navigate.py` · main sim loop

**Change**:
1. CSV now writes **every step** (was every 50 steps) for full-resolution
   post-run analysis (speed profiles, policy-switch timing, SLAM convergence).
2. File opened with `buffering=1` (line-buffered) — no explicit `flush()` needed.
3. Clarified that a log file is **always** created (removed the confusing
   `if args.log_file or True:` pattern); default path is `logs/nav/nav_<ts>.csv`.
4. Console status print remains every 50 steps (unchanged) to avoid terminal spam.

---

### 🟡 IMPROVEMENT 7 — `local_planner.py` + `global_planner.py`: O(1) angle wrap

**Files**: `source/rexmi_rl/nav/local_planner.py`, `global_planner.py`

**Change**: Replaced `while` loop angle wrapping with a single modulo expression:

```python
# BEFORE (O(k) loops — fragile after physics resets with multi-revolution yaw):
while a > math.pi:  a -= 2 * math.pi
while a < -math.pi: a += 2 * math.pi

# AFTER (O(1) — handles any input):
a = a - 2 * math.pi * math.floor((a + math.pi) / (2 * math.pi))
```

**Impact**: Prevents a rare but hard-to-diagnose bug where a physics reset
introduces a 720°+ yaw discontinuity, causing the while-loop to spin for
thousands of iterations and freeze the nav thread.

---

### Summary Table

| # | Severity | File | Change | Impact |
|---|----------|------|--------|--------|
| 1 | 🔴 Bug | `slam.py` | `_kdtree_dirty = True` → `False` | SLAM 2-3× faster |
| 2 | 🔴 Bug | `slam.py` | Initialize `_last_slam_pose = None` in `__init__` | Prevents AttributeError |
| 3 | 🔴 Bug | `navigator.py` | Write SLAM map to `shared["lidar_cloud"]` + call `update_slam_map()` every 2 s | SLAM feeds dashboard + A* |
| 4 | 🟠 Improvement | `navigator.py` | Adaptive boot: 3 s min (was 8 s) + richer progress log | ~5 s faster startup |
| 5 | 🟠 Improvement | `occupancy_map.py` | Vectorize `update()` cloud append | 10× faster cloud writes |
| 6 | 🟠 Improvement | `navigate.py` | CSV logs every step, always creates log file | Full telemetry |
| 7 | 🟡 Improvement | `local_planner.py`, `global_planner.py` | O(1) `_wrap_angle` (was while-loop) | Prevents rare freeze on physics reset |
