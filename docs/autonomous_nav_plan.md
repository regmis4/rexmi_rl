# REXMI Autonomous Navigation — Implementation Plan

**Date:** 2026-07-19  
**Status:** In progress  
**Author:** AI-assisted development session

---

## Project Goal

Enable the Go2W robot quadruped to navigate autonomously through lunar crater terrain
on Earth gravity. The robot must:
1. Move forward handling varied roughness (already working via 3 policies)
2. Detect terrain obstacles (boulders, steep no-friction slopes, crater rims)
3. Stop forward motion, turn to face a clear heading, and re-route
4. Link terrain handling ↔ obstacle detection ↔ SLAM-based re-routing into a seamless loop

---

## Existing Working Policies (Do Not Retrain)

| Policy | Checkpoint | Capability | Obs Space |
|--------|-----------|------------|-----------|
| `fast_flat` | `model_1499.pt` | Flat terrain, up to 2 m/s | ~60-dim (NO height scan) |
| `rough` | `model_8996.pt` | Stairs/boxes/rough, slopes <20° | ~208-dim (WITH height scan) |
| `rocky_slope` | `model_13994.pt` | Boulder slopes 15–35° | ~247-dim (WITH height scan) |

**PolicySelector** switches between these at runtime based on terrain metrics from the height scanner.

---

## Phase 1: Dead Code Removal ✅

**Completed 2026-07-19**

Deleted the following failed experiments that contaminated the codebase:

### Deleted files
- `source/rexmi_rl/tasks/locomotion/velocity/config/go2w/spin_env_cfg.py`
  - Reason: All spin training attempts failed (measured 1°/s actual vs 57°/s commanded).
    v4 terrain had apex singularity (platform_width=0 → robot spawned at pyramid tip → death).
    v5 terrain was correctly designed but never converged. Starting blank.
- `source/rexmi_rl/tasks/locomotion/velocity/config/go2w/spin_static_env_cfg.py`
  - Reason: Explicitly failed (postmortem §7: "ABANDONED — concept failed"). Two-stage
    curriculum approach is fundamentally flawed — can't teach slope-hold in isolation
    from locomotion when the pre-trained network is conditioned on velocity gradients.

### Removed registrations (from `__init__.py`)
- `RexmiRl-Go2w-Velocity-Spin-v0`
- `RexmiRl-Go2w-Velocity-Spin-Play-v0`
- `RexmiRl-Go2w-Velocity-SpinStatic-v0`
- `RexmiRl-Go2w-Velocity-SpinStatic-Play-v0`

### Removed PPO configs (from `rsl_rl_ppo_cfg.py`)
- `Go2wSpinPPORunnerCfg`
- `Go2wSpinStaticPPORunnerCfg`

### Removed duplicate utility
- `_wrap_angle` removed from `global_planner.py` — was identical to the one in
  `local_planner.py`. Now imported from there.

---

## Phase 2: SLAM Localization Fix ✅

**Completed 2026-07-19**

### Problem
`Navigator.__init__` was cheating: it instantiated `SLAMLocalizer` but then hardwired
`self._localizer = _odom_localizer` (sim ground-truth pose). SLAM was used only for map
building, never for localization. This is not autonomous navigation — it's scripted
navigation with a SLAM sidecar.

### Fix
When a LiDAR sensor is present, `self._localizer = _slam_localizer` (the `SLAMLocalizer`
wrapping the ICP engine). `SLAMLocalizer.get_pose()` already has a proper stability gate:
- Requires 5 consecutive low-RMS ICP frames before switching to SLAM pose
- Falls back to odometry on divergence (single bad frame doesn't cause a jump)
- During BOOT phase: odometry used (SLAM not yet warmed up) — correct

The BOOT phase exists precisely to give SLAM time to converge before navigation starts,
so the localization switch is seamless. In deployment (real robot), odometry = wheel
encoders + IMU dead-reckoning, which the `SimLocalizer` simulates perfectly.

### Dead variable removed
The `_slam_localizer` variable was assigned in a branch but never referenced again.
Cleaned up by actually using it as the nav localizer.

---

## Phase 3: Recovery FSM → Policy Selector Integration ✅

**Completed 2026-07-19**

### Problem
`RecoveryFSM` in ROTATING state sends `(vx=0, vy=0, omega=±1.0)` to the command tensor.
But `PolicySelector` was chosen *before* RecoveryFSM ran — so the `rough` or `rocky_slope`
policy was active during rotation. These policies were trained at vx∈(0.2–0.5) and produce
~1°/s actual rotation at vx=0 (out-of-distribution).

### Fix
`Navigator.step()` now checks `recovery.is_rotating` BEFORE running PolicySelector.
When recovering:
- If a dedicated turn policy checkpoint is loaded (`PolicyMode.TURN`): use it
- Otherwise: use `rough` policy (it at least has vx=0 in its training distribution)
- Log a warning each time recovery rotation uses rough-as-fallback

Added `RecoveryFSM.is_rotating` property (True in REVERSING or ROTATING states).

---

## Phase 4: New Turn Policy — Blank Slate Design ✅

**Completed 2026-07-19**

### Why blank slate
Previous attempts (spin v1–v5, spin_static) all inherited from `Go2wRockySlopeEnvCfg`.
This was the root cause of failure — rocky slope joint penalties (hip_crossing, thigh_salute,
calf_symmetry, hip_symmetry) were designed for 35° forward locomotion postures. At vx=0
with a completely different reward structure, they drove the robot into a death spiral
(net reward floor negative → robot falls immediately → can never learn spinning).

### New design: `Go2wTurnEnvCfg` in `turn_env_cfg.py`

**Inherits from:** `Go2wRoughEnvCfg` (NOT rocky slope)
- Reason: rough policy has vx∈(-0.5, 0.5) — it has seen vx=0 before. Rocky slope
  has vx∈(0.2, 0.5) always forward — vx=0 is completely OOD.
- Rough also has height scan in obs (same architecture) → checkpoint transfer works.

**Terrain:** Mixed flat+gentle slope curriculum
- Zone 1 (50%): `HfRandomUniformTerrainCfg` ±2cm noise — flat, grip, no apex death
- Zone 2 (50%): `HfPyramidSlopedTerrainCfg` 0–20° slope, platform_width=2.0m
- Max slope: 20° (within Kd=2.0 wheel-brake capacity on μ=0.8 terrain)
- num_rows=8, num_cols=20 — curriculum from flat → gentle slope

**Commands:**
- `lin_vel_x = (0.0, 0.0)` — locked zero (must learn to spin without translating)
- `lin_vel_y = (0.0, 0.0)` — locked zero
- `ang_vel_z = (-1.0, 1.0)` — full spin range (same as rough training range)

**Rewards (3 primary signals, everything else inherited or reduced):**
1. `track_ang_vel_z_exp` weight = +4.0 (dominant — must spin at commanded rate)
2. `track_lin_vel_xy_exp` weight = -0.8 (penalty — drift = failure, but not catastrophic)
3. `is_alive` weight = +0.3 (survival signal — stay upright)

**Removed rewards:** stagnation (vx_cmd=0 always, never fires), climb_progress (same)

**Joint penalties:** Keep only `leg_deviation` (mild -0.05) and `dof_pos_limits` (-0.2).
Do NOT include hip_crossing/thigh_salute/calf_symmetry/hip_symmetry — these are the
penalties that killed all previous spin attempts.

**Starting checkpoint:** `model_8996.pt` (rough run)
- Already has vx=0 in its training distribution
- Weaker wheel-drive priors than model_13994 → less catastrophic initial wheel_brake cost
- Height scan obs matches architecture

**Training command:**
```bash
conda activate env_isaacsim
python scripts/train.py --task RexmiRl-Go2w-Velocity-Turn-v0 --headless \
    --load_run go2w_velocity_rough/2026-06-14_20-03-41 \
    --checkpoint model_8996.pt \
    --max_iterations 1500
```

**Health signals to watch:**
- `track_ang_vel_z_exp` > 0.5 by iter 300 → spinning established
- `base_contact` < 5% → robot not falling
- `track_lin_vel_xy_exp` → small negative (< -0.1) → minimal drift
- `terrain_levels` advancing past 4.0 by iter 600 → slope generalisation

**Convergence target:** 800–1200 iterations

**Nav integration:**
```bash
python scripts/navigate.py \
    --ckpt_fast_flat logs/.../model_1499.pt \
    --ckpt_rough     logs/.../model_8996.pt \
    --ckpt_rocky     logs/.../model_13994.pt \
    --ckpt_turn      logs/rsl_rl/go2w_velocity_turn/<date>/model_<N>.pt \
    --mission traverse
```

`PolicySelector` activates TURN mode when `|heading_error| > 75°` (already wired).

---

## Phase 5: Fast-Flat Sister Policy with Height Scan (Future Work)

**Not yet implemented — training pending**

### Design intent
`fast_flat` (`model_1499.pt`) has NO height scan in its observation space. This means
it cannot see crater terrain and must only be used on flat exterior terrain. For the
crater traversal demo, the robot should use `fast_flat` only on the exterior approach
and immediately switch to `rough` when the height scanner starts seeing slope.

For a full crater demo where the robot can sprint on flat crater floor at high speed
AND see terrain ahead, we need a sister policy: **`fast_flat_scan`**.

### Plan
- Create `Go2wFastFlatWithScanEnvCfg` inheriting `Go2wFastFlatEnvCfg`
- Re-enable `height_scanner` + add `height_scan` observation term
- Train from scratch (different obs dim → not compatible with model_1499.pt)
- Register as `RexmiRl-Go2w-Velocity-FastFlatScan-v0`
- Target: vx up to 2 m/s on flat/crater-floor terrain with full terrain perception

### Geometry note
Train on flat terrain + slight inclines only (crater floor is essentially flat).
Do NOT train on steep slopes — let `rough` and `rocky_slope` handle those.
The policy switching hysteresis (50 steps to commit to fast_flat) already prevents
premature fast-flat activation on slopes.

---

## Phase 6: Obstacle Avoidance Behavioral Loop (Architecture)

The complete autonomous navigation loop:

```
SLAM (10 Hz)
  → OccupancyMap (boulder/wall cells marked inf cost)
  → GlobalPlanner A* (routes around obstacles)
  → LocalPlanner (heading candidates, fwd_obstacle_dist)
  → PolicySelector (terrain type → which policy)
  → RecoveryFSM (stuck detection → reverse → turn)
  → Navigator.step() (ties everything together)
```

**Current gaps and fixes:**

### Gap 1: Turn policy not ready
**Status:** Fixed by Phase 4 (turn policy design). Awaiting training.

### Gap 2: Obstacle stop is incomplete
`compute_with_forward()` scales vx→0 at 0.5m obstacle. But this only slows forward
motion — it doesn't actively steer toward an open heading. The robot slows to a stop
facing the boulder, then RecoveryFSM triggers (stuck), then recovery does reverse+rotate.
This is correct but slow (5s stuck timeout + 2s reverse + 3s rotate = 10s per attempt).

**Future improvement:** When fwd_obstacle_dist < 1.5m AND center columns are blocked,
LocalPlanner should actively steer toward the best open lateral column (already computed)
rather than going straight at full stop speed. This converts the 10s recovery cycle into
a proactive detour. Implementation: add `obstacle_detour_omega` to `LocalPlannerOutput`
and use it when vx is near zero due to forward obstacle.

### Gap 3: Slipping detection
Progress-based stuck check in RecoveryFSM (`progress_timeout=60s`) catches the case
where the robot is moving (speed > stuck_speed) but not making waypoint progress
(slipping in place on low-friction slope). Already implemented. Threshold: 0.5m progress
in 60s window.

---

## File Change Summary

| File | Change | Phase |
|------|--------|-------|
| `spin_env_cfg.py` | **DELETED** | 1 |
| `spin_static_env_cfg.py` | **DELETED** | 1 |
| `config/go2w/__init__.py` | Removed Spin + SpinStatic registrations; added Turn registration | 1, 4 |
| `agents/rsl_rl_ppo_cfg.py` | Removed Spin + SpinStatic PPO configs; added Turn PPO config | 1, 4 |
| `nav/navigator.py` | Fixed SLAM localizer wiring; fixed recovery→policy integration | 2, 3 |
| `nav/global_planner.py` | Removed duplicate `_wrap_angle` | 1 |
| `nav/recovery.py` | Added `is_rotating` property | 3 |
| `nav/policy_selector.py` | Renamed SPIN→TURN for clarity; clean turn-override logic | 4 |
| `nav/local_planner.py` | Updated TURN mode reference | 4 |
| `turn_env_cfg.py` | **NEW** — clean turn policy training environment | 4 |
| `scripts/navigate.py` | Added `--ckpt_turn` argument; updated SPIN→TURN references | 4 |

---

## Key Principles Going Forward

1. **No ground-truth pose cheating** — SLAM localizer must be used when LiDAR is available.
   Odometry is the fallback, not the primary.

2. **No policy inheritance from rocky_slope for turn/spin** — rocky_slope joint penalties
   are tuned for 35° forward locomotion. They kill any vx=0 reward structure.

3. **Inherit from rough for turn policy** — rough has vx=0 in its training distribution,
   compatible obs space, and transferable weights.

4. **Blank slate for turn policy** — no carry-over of spin/spin_static configs, weights,
   or reward structures that proved wrong. Start clean.

5. **The terrain handling system already works** — rough + rocky_slope handle the physical
   challenge. The missing piece is purely the turn capability. Add that and everything
   connects.
