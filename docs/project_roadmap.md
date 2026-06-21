# REXMI RL — Project Roadmap & Strategic Plan

> **Last updated: 2026-06-20**
>
> **End goal**: A Go2W wheeled-hybrid quadruped that demonstrates autonomous,
> insane-terrain navigation capability on a simulated lunar surface — suitable
> for investor and mission-partner demos.  Full-stack: RL locomotion policies
> + hierarchical navigation (SLAM + global coordinates + policy selector).

---

## Table of Contents

1. [Vision & End State](#1-vision--end-state)
2. [System Architecture](#2-system-architecture)
3. [Current Policy Library](#3-current-policy-library)
4. [Phase Roadmap](#4-phase-roadmap)
5. [Capability Prioritisation & ROI Analysis](#5-capability-prioritisation--roi-analysis)
6. [Rejected / Deferred Ideas](#6-rejected--deferred-ideas)
7. [Open Research Questions](#7-open-research-questions)

---

## 1. Vision & End State

The robot must autonomously traverse a simulated lunar crater (targeting Shackleton
crater geometry — 35° average inner wall slope) from a flat mare approach, through
a rough ejecta field, down a steep crater wall, and back out.  No human intervention.

```
                  ┌────────────────────────────────────────────────────────┐
                  │                LUNAR SURFACE MISSION                   │
                  │                                                          │
                  │  [Lander/GPS fix]──▶ Global Nav ──▶ Waypoint commands   │
                  │                          │                               │
                  │                     SLAM map                            │
                  │                    /    |    \                           │
                  │          terrain type classifier                         │
                  │          /         |          \                          │
                  │    [Fast flat]  [Rough]  [Steep-slope]                   │
                  │          \         |          /                          │
                  │           velocity commands (vx, vy, ωz)                │
                  │                    │                                     │
                  │               Go2W Robot                                 │
                  └────────────────────────────────────────────────────────┘
```

### What "insane terrain capability" means concretely

| Terrain type | Target capability | Demo shot |
|---|---|---|
| Flat mare | 2.0 m/s forward sprint | Side-by-side speed comparison with Boston Dynamics Spot |
| Rough ejecta / rocks | Obstacles ≤23 cm, slopes ≤23° | Continuous traverse of cluttered field |
| Steep crater wall | ≥40° slope (target: 45°) | Robot descending/ascending Shackleton-scale crater |
| Recovery | Self-righting from fallen state | Robot falls, gets back up, continues mission |

---

## 2. System Architecture

### Hierarchical control layers

```
Layer 3 — Mission Planning (future)
  Global path planning, waypoint sequencing, mission objectives
  Input: GPS/lander tracking, SLAM map, battery/time budget
  Output: waypoints + terrain context

Layer 2 — Navigation (Phase 9)
  Terrain-type classifier + policy selector
  Input: height scan (160-dim), IMU, current policy performance
  Output: which policy to invoke + (vx, vy, ωz) commands to that policy

Layer 1 — Locomotion Policies (current focus)
  RL-trained policies for each terrain class
  Input: (vx, vy, ωz) command + onboard sensors (IMU, height scan)
  Output: 16-DOF joint commands (4 wheel velocities + 12 leg positions)
```

### Navigation layer design (Phase 9)

The Phase 9 policy selector starts as a **rule-based classifier** (not learned),
based on the height scanner data already used by the RL policies:

```
height_scan → terrain feature extraction:
  max_relief     = max(ray_hits_z) - min(ray_hits_z)          # overall roughness
  max_gradient   = max local height slope from 4-neighbour diff  # steepness
  flat_ratio     = fraction of rays within ±2cm of median        # flat fraction

Rules:
  if max_gradient > tan(20°) × resolution   → SteepSlope policy
  elif max_relief > 0.05 m                  → Rough policy
  else                                       → FastFlat policy

Hysteresis: require 20 consecutive matching classifications before switching
(prevents rapid oscillation at terrain boundaries)
```

This is intentionally simple for Phase 9.  A learned meta-policy (Phase 13+)
would replace this when sufficient training data exists.

---

## 3. Current Policy Library

| Task ID | Policy | Checkpoint | Speed | Terrain capability | Status |
|---|---|---|---|---|---|
| `RexmiRl-Go2w-Velocity-FastFlat-v0` | Fast flat | TBD | 2.0 m/s fwd | Flat only | ✅ Production |
| `RexmiRl-Go2w-Velocity-Rough-v0` | Rough terrain | `model_8996.pt` (2026-06-14_20-03-41) | ±0.5 m/s | Stairs/boxes ≤23cm, slopes ≤23° | ✅ Production (frozen) |
| `RexmiRl-Go2w-Velocity-SteepSlope-v0` | Steep slope | `model_2999.pt` (2026-06-20_11-32-10) | ±0.5 m/s | Slopes 23°–~36°, **targeting 40°+** | 🔄 Phase 8 (active) |

### Checkpoint freeze policy

- **`model_8996.pt` (rough terrain)**: FROZEN.  Do not retrain.  Any rough-terrain
  improvements must be done in a new experiment with a new run name.
- **Steep-slope best**: `model_2999.pt`, terrain_level=5.38 (~36.1°).
  Resume command:
  ```bash
  python scripts/train.py --task RexmiRl-Go2w-Velocity-SteepSlope-v0 --headless \
      --load_run go2w_velocity_steep_slope/2026-06-20_11-32-10 --checkpoint model_2999.pt
  ```

---

## 4. Phase Roadmap

### Phase 8 — Steep slopes ≥40° 🔄 IN PROGRESS

**Goal**: Push the steep-slope policy from ~36° to ≥40° (hard target: 45°).

**Status**: terrain_level=5.38 at model_2999 (row 5.38/9.0 ≈ 36.1°).

**Remaining work**:
- [ ] Continue training from model_2999 — assess TensorBoard trend for plateau vs. convergence
- [ ] If plateauing before 40°: investigate reward adjustments (see options below)
- [ ] Achieve ≥40° (terrain_level ≥ 7.4) on TensorBoard `terrain_levels` metric
- [ ] Run `eval.py --group steep_slope` to confirm 40° survival_rate ≥ 80%
- [ ] Freeze best checkpoint and update `docs/rl_setup.md`

**Levers if plateau before 40°**:

| Issue | Symptom | Fix |
|---|---|---|
| Policy still drifting laterally | yaw tracking < 0.7 | Increase `track_ang_vel_z_exp` weight further (1.5 → 2.0) |
| Policy sliding back on steep face | terrain_level oscillates at 5-6 | Increase wheel action scale (10 → 15) to allow more grip force |
| New exploit emerging | Any unexpected negative reward term growing | Add targeted threshold penalty (same pattern as hip_crossing, thigh_salute) |
| True curriculum plateau | terrain_level flat for 500+ iters | Consider two-policy approach: one for 23-35°, one for 35-45° |

---

### Phase 9 — Hierarchical Nav Demo ⏳ PLANNED

**Goal**: Single simulation demo showing a Go2W autonomously traversing flat terrain
→ rough ejecta field → steep crater wall, with automatic policy switching.

**Deliverables**:

1. **Crater traverse environment** (`crater_env_cfg.py`)
   - Contiguous terrain map: flat buffer zone → rough mixed terrain → steep pyramid slopes
   - Not a training environment — a fixed demo map (no curriculum, no randomisation)
   - Size: ~60m × 40m, traversable in one continuous episode

2. **Policy selector** (`scripts/nav_demo.py`)
   - Loads all three policy checkpoints simultaneously
   - Runs terrain classifier on current height scan every 0.5s
   - Switches active policy with hysteresis (20-step debounce)
   - Logs active policy, terrain classification, velocity tracking to TensorBoard

3. **Demo script** (`scripts/run_crater_demo.py`)
   - Isaac Sim GUI mode (not headless)
   - Follows one selected robot with cinematic camera
   - Saves video

**Key technical challenges**:
- Policy switching must be **seamless** (no re-initialisation of joint targets mid-step)
- At policy boundary, carry over last action from old policy as first action for new policy
- The flat policy uses `scale=40` vs rough/steep `scale=10` — this mismatch requires
  a smooth ramp-down of wheel commands at the transition point

**Open question**: Is a rule-based classifier good enough for a clean demo, or do we
need a small learned classifier to handle noisy height scan transitions?  Try rule-based
first — if it produces policy-switch oscillations, fall back to a 2-second forced
minimum dwell time per policy.

---

### Phase 10 — Lunar Physics ⏳ PLANNED

**Goal**: Scientifically valid simulation — all policies retrained under lunar conditions.

**Changes**:

1. **Lunar gravity** — one-line Isaac Lab change:
   ```python
   # in scene config or sim config
   self.sim.gravity = (0.0, 0.0, -1.62)   # m/s² (Earth = -9.81)
   ```
   All three policies must be retrained from scratch (or from Phase 8/9 checkpoints
   with fine-tuning).  Under 1/6th gravity, energy budgets change fundamentally:
   wheel torque needed for same speed is ~6× less, but terrain interaction forces
   also change (softer impacts, different friction balance).

2. **Updated URDF — lunar wheel** 
   - Current: round rubber wheel (5cm radius, smooth tread)
   - Target: wider, paddle/lug wheel for regolith traction
   - Requires URDF mesh update + collision geometry + inertia recalculation
   - Contact with JPL/ESA wheel research: paddle wheels provide 2-3× traction
     improvement in loose regolith vs. smooth wheels

3. **Regolith friction domain randomisation**
   - Lunar regolith friction coefficient: ~0.5–0.6 (compacted), ~0.3 (loose surface)
   - Add to `events` in training config:
     ```python
     events.randomise_terrain_friction = EventTermCfg(
         func=mdp.randomise_terrain_material,
         mode="reset",
         params={"friction_range": (0.30, 0.65)},
     )
     ```
   - **Project Chrono integration** (longer term): PhysX handles rigid-body contact well
     but does not model granular/particulate media.  Project Chrono's SPH/DEM solver
     can simulate loose regolith accurately.  This is a simulation backend swap
     (months of integration work) — treat as Phase 10 stretch goal, not baseline.

4. **Training order under lunar gravity**:
   - Flat → fast-flat → rough → steep-slope  (same curriculum as original)
   - Starting from Phase 8/9 Earth checkpoints as init (partial transfer expected;
     policy may need 1000+ "relearning" iterations before improving beyond init)

---

### Phase 11 — Recovery Policies ⏳ PLANNED

**Goal**: Robot can autonomously recover from a fallen state and resume mission.

**Decision**: Deferred to after lunar physics phase because:
- Under lunar gravity (1/6th g), self-righting torques needed are ~6× smaller
- The Go2W leg actuators (stiffness=5 soft PD) are borderline for Earth-gravity righting
- Lunar gravity makes the policy physically achievable and sim-to-real transfer easier
- The "robot falls on crater wall, recovers, continues" demo is far more compelling than
  Earth-gravity recovery

**Sub-problems**:

1. **Stagnation escape** — handled in the nav layer (not a separate RL policy):
   - If velocity tracking < 0.1 m/s for > 5 seconds while commanded > 0.2 m/s:
     issue reverse + yaw command for 2s, then retry forward
   - If still stuck after 3 attempts: issue "stuck" signal to mission planner

2. **Fall-righting policy** — new RL policy:
   - New environment: `recovery_env_cfg.py`
   - Reset state: random fallen pose (roll/pitch from full distribution, base on ground)
   - Reward:
     ```
     projected_gravity_z:  +2.0 × (gz - gz_fallen) / (1.0 - gz_fallen)  # upward progress
     base_height:          +1.0 × clamp(h - h_ground, 0, 0.4)           # rising off ground
     base_contact:         -2.0 per step (base link on ground)           # stay off floor
     is_upright:           +5.0 bonus when |gravity_projected_xy| < 0.1  # reached upright
     ```
   - Terminations: upright achieved (success) OR 10 seconds elapsed (failure)
   - Training: 3000 iterations, 4096 envs, separate experiment name
   - Integration: recovery policy called by nav layer when `base_contact` detected

---

### Phase 12 — Jumping (Stretch Goal) ⏳ DEFERRED

**Rationale**: Under lunar gravity (1/6th g), the same leg impulse produces ~6× jump height.
A 0.3m Earth jump → ~1.8m lunar jump.  Visual impact is extraordinary.  However:

- Go2W is not architecturally optimised for jumping (5cm wheels, soft PD, no spring)
- Requires completely new reward structure and terminal state handling
- Impact landing is the hardest part (shock absorption policy)
- Development risk: 2-4 months minimum

**Condition to start**: All Phases 9-11 complete, Phase 12 begins only if lunar physics
retraining shows the robot has sufficient leg actuation headroom under 1.62 m/s².

---

### Phase 13 — Learned Policy Selector (Stretch Goal) ⏳ DEFERRED

Replace the rule-based terrain classifier from Phase 9 with a small learned network
that takes the height scan + velocity history → policy selection probability.

Train via imitation from the rule-based selector, then fine-tune via RL with the
nav-level reward (mission completion rate).

---

## 5. Capability Prioritisation & ROI Analysis

### For investor/demo purposes

| Capability | Investor visual impact | Dev effort | ROI |
|---|---|---|---|
| 40°+ steep slope | ⭐⭐⭐⭐⭐ | Low (weeks, nearly there) | **🏆 Highest** |
| Crater traverse demo (nav switching) | ⭐⭐⭐⭐⭐ | Medium (1-2 months) | **🏆 Highest** |
| Lunar gravity physics | ⭐⭐⭐⭐ | Medium (retraining + URDF) | Very high |
| Recovery / fall-righting | ⭐⭐⭐⭐ | High (new policy design) | High |
| Regolith friction domain rand. | ⭐⭐ (technical credibility) | Low | Medium |
| Jumping (lunar gravity) | ⭐⭐⭐⭐⭐ | Very high (months) | Medium (high risk) |
| Energy efficiency metrics | ⭐ (mission planning) | Very low (add to eval) | Low for demo |
| Bipedal two-leg climbing | ⭐⭐⭐⭐ | Extremely high (URDF change) | Very low (too risky) |

### The single most important demo moment

> **A robot autonomously descending a 40°+ crater slope, with the policy selector
> shown switching from "rough terrain mode" to "steep slope mode" as the crater
> wall begins, under lunar gravity, with scientific caption: "Shackleton crater
> inner wall — 35° average slope."**

Everything we build should serve this single demo shot.

---

## 6. Rejected / Deferred Ideas

### Bipedal two-leg obstacle climbing ❌ Rejected

The Go2W hip/thigh URDF range of motion does not support stable bipedal stance.
Hip joint limits ≈ ±0.5 rad — not enough lateral range for weight-bearing bipedal.
Thigh joint max forward pitch ≈ 1.5 rad — legs would not clear body for bipedal step.
Would require a redesigned URDF (different robot) — out of scope for this platform.

### Energy efficiency as primary training objective ❌ Deferred to eval metrics

`dof_torques_l2` already penalises energy use during training.  Adding energy as
a primary objective would conflict with peak-torque requirements for slope climbing.
Better approach: add energy consumption (total joint torque integral per meter) as
a **metric** to `eval.py`, not a reward term.  Can then say "policy uses X Wh/km"
for mission planning purposes.

### Project Chrono regolith simulation (near-term) ❌ Deferred to Phase 10+

Full granular media simulation (SPH/DEM) requires either:
- A custom Isaac Lab → Project Chrono bridge (significant engineering)
- Or switching to Project Chrono as primary simulator (loses GPU vectorisation)

Domain randomisation of friction coefficients (0.3–0.65) in PhysX is the near-term
proxy.  True Chrono integration is a Phase 10+ stretch goal requiring dedicated
simulation infrastructure work.

---

## 7. Open Research Questions

1. **Can the Go2W actuators physically self-right on Earth?**  
   Need to calculate max torque from soft PD (stiffness=5, damping=0.5) at the hip
   joint in a fallen configuration.  If τ_max < mg×l_CoM, a pure RL approach won't
   work and we'd need to add a "launching" impulse or change actuator params.

2. **Will Earth-gravity steep-slope policies transfer to lunar gravity?**  
   Hypothesis: yes for leg shape/gait (gravity direction is the same), no for absolute
   force magnitudes.  Expect 500–1000 fine-tuning iterations needed.  Test with 0 fine-
   tuning iters first to measure baseline transfer quality.

3. **What is the maximum achievable slope angle for this robot?**  
   Physical upper limit: when the centre of mass is directly above the contact polygon
   and gravity component along slope > maximum static friction force.  For Go2W:
   μ_static ≈ 0.8 (rubber on stone), CoM height ≈ 0.3m, contact polygon ≈ 0.4×0.3m.
   Theoretical max ≈ arctan(0.8) ≈ 39°.  With active leg repositioning (shifting CoM
   forward), the RL policy may push beyond 45° briefly.  Track empirically.

4. **Rule-based terrain classifier vs. learned — when does the rule-based version fail?**  
   The rule-based classifier will struggle at:
   - Gradual slope transitions (smooth gradient → mis-classified as rough)
   - Stairs on a slope (should use rough policy but looks like steep slope)
   - Novel terrain types not in training distribution
   Monitor false-positive policy-switch rate in Phase 9 nav demo and use it to
   justify (or not) a learned classifier in Phase 13.

---

*This document is the authoritative project roadmap reference.  Update it at each
phase completion.  Companion documents:*
- `docs/rl_setup.md` — complete technical reference for the RL system
- `docs/lunar_crater_terrain_research.md` — crater geometry and terrain data
- `docs/lunar_crater_demo_run.md` — demo run instructions
