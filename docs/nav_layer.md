# REXMI Navigation Layer

**Status**: Phase N-1 complete (deterministic nav on sim ground-truth localisation)  
**Date**: 2026-07-06  
**Policies**: `model_1499.pt` (fast_flat) · `model_8996.pt` (rough) · `model_13994.pt` (rocky_slope)  
**Auto-switching**: terrain-aware PolicySelector picks the right policy every 50 ms

---

## Overview

A fully deterministic navigation stack that sits **above** the RL policy and converts
high-level mission goals into per-step `(vx, vy, ωz)` velocity commands.  The RL
policy never changes — the nav layer simply replaces what the command sampler would
normally generate.

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 3 — Mission Planner                                  │
│  Named missions → ordered waypoint list (crater-relative)   │
└────────────────────┬────────────────────────────────────────┘
                     │ waypoints
┌────────────────────▼────────────────────────────────────────┐
│  LAYER 3b — Global Planner (A*, replans every 2 s)          │
│  OccupancyMap (accumulated height-scan) → lookahead WP      │
└────────────────────┬────────────────────────────────────────┘
                     │ immediate waypoint 4 m ahead
┌────────────────────▼────────────────────────────────────────┐
│  LAYER 2 — Local Planner (50 Hz)                            │
│  16×10 height scan → traversability → (vx, vy, ωz)         │
│  + RecoveryFSM override when stuck                          │
│  ↳ terrain metrics (slope, max_step, trav_frac)             │
└────────────┬───────────────────────────────┬────────────────┘
             │ velocity command               │ terrain metrics
             │                ┌──────────────▼────────────────┐
             │                │  PolicySelector (50 Hz)        │
             │                │  slope < 5°      → fast_flat   │
             │                │  step < 6 cm     → rough       │
             │                │  slope 20–35°    → rocky_slope │
             │                │  hysteresis: 25 steps          │
             │                └──────────────┬────────────────┘
             │                               │ which policy to use
┌────────────▼───────────────────────────────▼────────────────┐
│  LAYER 1 — RL Policy (50 Hz, auto-selected)                 │
│  fast_flat:    model_1499.pt  — up to 2.0 m/s               │
│  rough:        model_8996.pt  — ~0.8 m/s                    │
│  rocky_slope:  model_13994.pt — ~0.4 m/s, 35° uphill        │
│  Inputs: full 220-dim obs  |  Outputs: 16 joint actions      │
└─────────────────────────────────────────────────────────────┘
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
  navigate.py          — entry point (run this)
```

---

## Quick Start

```bash
conda activate env_isaacsim

# ── Auto policy-switching (recommended) ──────────────────────────────────
# Each policy is loaded from its own task (different obs dims / net sizes):
#   fast_flat:   60-dim obs, 128-wide net
#   rough:       ~120-dim obs, varies
#   rocky_slope: 247-dim obs, 512-wide net  (this is also the nav env)
python scripts/navigate.py \
    --task           RexmiRl-Go2w-Crater-Bowl-RockySlope-Play-v0 \
    --ckpt_fast_flat logs/rsl_rl/go2w_velocity_fast_flat/2026-06-17_20-08-58/model_1499.pt \
    --ckpt_rough     logs/rsl_rl/go2w_velocity_rough/2026-06-14_20-03-41/model_8996.pt \
    --ckpt_rocky     logs/rsl_rl/go2w_velocity_rocky_slope/2026-06-30_09-31-48/model_13994.pt \
    --mission traverse
# task_fast_flat and task_rough default to the right crater-bowl variants automatically

# Floor survey (lawnmower scan of crater interior)
python scripts/navigate.py \
    --task RexmiRl-Go2w-Crater-Bowl-RockySlope-Play-v0 \
    --ckpt_fast_flat logs/.../model_1499.pt \
    --ckpt_rough     logs/.../model_8996.pt \
    --ckpt_rocky     logs/.../model_13994.pt \
    --mission survey

# Rim perimeter circuit
python scripts/navigate.py ... --mission rim_circuit

# ── Fixed single policy (no auto-switching) ───────────────────────────────
python scripts/navigate.py \
    --task RexmiRl-Go2w-Crater-Bowl-RockySlope-Play-v0 \
    --checkpoint logs/.../model_13994.pt \
    --policy_mode rocky_slope \
    --mission traverse

# ── Headless (no matplotlib window, e.g. for recording) ──────────────────
python scripts/navigate.py ... --no_dashboard
```

### All CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | *(required)* | Isaac Lab gym task name |
| `--ckpt_fast_flat` | — | Checkpoint for fast_flat policy (auto mode) |
| `--ckpt_rough` | — | Checkpoint for rough policy (auto mode) |
| `--ckpt_rocky` | — | Checkpoint for rocky_slope policy (auto mode) |
| `--task_fast_flat` | `RexmiRl-Go2w-Crater-Bowl-FastFlat-Play-v0` | Task to build fast_flat net architecture |
| `--task_rough` | `RexmiRl-Go2w-Crater-Bowl-Play-v0` | Task to build rough net architecture |
| `--checkpoint` | — | Single checkpoint path (use with `--policy_mode`) |
| `--policy_mode` | `auto` | `auto` / `fast_flat` / `rough` / `rocky_slope` |
| `--mission` | `traverse` | `traverse` / `survey` / `rim_circuit` |
| `--num_envs` | `1` | Parallel environments |
| `--device` | `cuda:0` | Torch device |
| `--crater_x` | `0.0` | Crater centre X (world frame, m) |
| `--crater_y` | `0.0` | Crater centre Y (world frame, m) |
| `--r_floor` | `3.0` | Crater floor radius (m) |
| `--r_rim` | `11.0` | Crater rim radius (m) |
| `--spawn_x` | `13.0` | Robot spawn X (m). Matches `LunarCraterDemoBowlEnvCfg` (exterior ramp, facing −x) |
| `--entry_azimuth_deg` | `0.0` | Crater entry azimuth (°, CCW from +x). `0°` = enters from +x side heading −x |
| `--replan_interval` | `2.0` | Seconds between A* replans |
| `--vx_normal` | `0.40` | Forward speed on clear terrain (m/s) |
| `--vx_steep` | `0.25` | Forward speed on steep terrain (m/s) |
| `--no_dashboard` | off | Disable matplotlib window |
| `--max_steps` | `15000` | Max sim steps (~300 s at 50 Hz) |

---

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
3. Check waypoint arrival → advance `wp_idx` if reached
4. `GlobalPlanner.update()` → immediate waypoint
5. `LocalPlanner.compute()` → `(vx, vy, omega)` + terrain metrics
6. `PolicySelector.update(slope, max_step, trav_frac)` → commit policy switch if hold counter reaches 25
7. `RecoveryFSM.update()` → override command if stuck
8. `command_manager.get_command("base_velocity")[env_idx] = cmd`
9. Write shared state for dashboard (cost grid updated every 10 steps)

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

## Phase N-2 Roadmap

| Feature | Status | What changes |
|---------|--------|-------------|
| Policy switching (terrain-aware) | ✅ **Done** | `policy_selector.py` — 3 policies, 25-step hysteresis |
| Any crater size | ✅ **Done** | Pass `--r_rim` / `--r_floor`; all missions auto-scale |
| Real SLAM | Phase N-2 | Add `SLAMLocalizer(ros_topic=…)`, no other changes needed |
| Multi-robot | Phase N-2 | Instantiate `Navigator` per robot with separate `env_idx` |
| D* Lite replanning | Phase N-3 | Drop-in replacement for `GlobalPlanner._astar()` |
| ROS 2 bridge | Phase N-3 | Replace `_inject_command()` with ROS topic publish |

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
