# Spin Policy Campaign — Post-Mortem

**Date range:** 2026-07-13 to 2026-07-15  
**Status:** Failed to converge. Scrapping and starting fresh.  
**Author:** AI-assisted development session

---

## 1. Problem Statement

The navigation stack (`navigate.py --mission traverse`) requires the robot to turn to face
a waypoint before driving toward it. When the heading error exceeds ~75°, `PolicySelector`
enters **committed turn mode**: `vx=0, vy=0, omega=±1.0`.

**Bug:** All existing policies (rough `model_8996`, rocky slope `model_13994`, spin
`model_9995`) were trained with `vx ∈ (0.2–0.5)`. At `vx=0` they produce near-zero joint
torques. Measured yaw rate: ~1°/s actual vs 57°/s commanded. The robot oscillates
indefinitely on crater slopes instead of turning.

---

## 2. Files Modified During This Campaign

| File | What Changed | State Now |
|------|-------------|-----------|
| `source/rexmi_rl/tasks/locomotion/velocity/mdp/rewards.py` | Added `position_drift_penalty()` | **Kept** — useful function |
| `source/rexmi_rl/tasks/locomotion/velocity/mdp/__init__.py` | Exports `position_drift_penalty` | **Kept** |
| `source/rexmi_rl/tasks/locomotion/velocity/config/go2w/spin_env_cfg.py` | Rewrote terrain to v5 (flat rows 0–4, gentle slope rows 5–9) | **Modified** — v5 terrain is correct |
| `source/rexmi_rl/tasks/locomotion/velocity/config/go2w/spin_static_env_cfg.py` | **New file** — Stage 1 intermediate env | **Dead end** — see below |
| `source/rexmi_rl/tasks/locomotion/velocity/config/go2w/agents/rsl_rl_ppo_cfg.py` | Added `Go2wSpinStaticPPORunnerCfg` | **Kept** — may still be useful |
| `source/rexmi_rl/tasks/locomotion/velocity/config/go2w/__init__.py` | Registered SpinStatic tasks | **Kept** |

---

## 3. Attempts and Results

### Attempt 1 — SpinStatic on Flat Terrain

**Hypothesis:** Train the robot to "hold still" on flat terrain before introducing spinning.
All commands zero. Reward: `is_alive` + `position_drift` + `wheel_brake`.

**Terrain:** Flat (no slope). `num_rows=5`.

**Result (training log — iter 10188–10194):**
```
base_contact:      75%     ← robot falling constantly
position_drift:   -22 to -30  ← drifting 11–15m from spawn
wheel_brake:       0.000   ← wheels ARE locked (good)
time_out:          24%     ← only 24% of episodes survive
```

**Root causes:**
1. **Wrong terrain.** Flat terrain teaches nothing — no gravity means no braking challenge.
   The robot can't learn slope holding on flat ground.
2. **Inherited joint penalties overwhelming is_alive.** `hip_crossing + thigh_salute +
   calf_symmetry + hip_symmetry` from the rocky slope parent env combined to ~-1.5/step.
   With `is_alive = +0.5/step`, the net reward floor was **-1.0/step** — robot fell
   immediately as "falling fast" was better than staying up.
3. **`position_drift` weight=-2.0 catastrophic after fall.** Robot slides across floor at
   -22 to -30/step. Policy is in a death spiral with no recovery path.

---

### Attempt 2 — SpinStatic on Slope, Curriculum Disabled

**Fix:** Switch terrain to `RockyPyramidSlopeCfg` 10°–15°, `platform_width=1.5m`,
`num_rows=1` (curriculum disabled). Remove 4 joint penalties.

**Result (training log — iter 10186–10194):**
```
base_contact:      0%      ← robot not falling (huge improvement)
bad_orientation:   63%     ← robot TIPPING OVER sideways
position_drift:   -56 to -70  ← drifting 28–35m (sliding off, then tumbling)
wheel_brake:       0.000   ← wheels locked
terrain_levels:    2.4     ← CURRICULUM ADVANCED despite num_rows=1 ???
time_out:          36%
```

**Root causes:**
1. **`position_drift` weight=-2.0 still catastrophic.** At 30m slide = -60/step.
   Robot slides off slope, tumbles, tip → `bad_orientation` termination. The gradient
   is so overwhelmingly negative that the policy can never learn braking from this signal.
2. **`bad_orientation: 63%`** was NOT the robot tilting at >80°. It was tipping over
   after sliding off the slope edge. The terrain issue was: robot spawning on platform
   correctly but then sliding to the slope edge and going over.

**Fix applied:** Reduced `position_drift` weight from -2.0 to -0.5, threshold from 0.05m
to 0.10m. Increased `is_alive` to +2.0 (dominant signal).

---

### Attempt 3 — Rebalanced Rewards

**Final spin_static_env_cfg.py state:**
```python
# Terrain
RockyPyramidSlopeCfg: slope 10°–15°, platform_width=1.5m, num_rows=1, no boulders

# Rewards
is_alive.weight        = +2.0    # dominant positive signal
position_drift.weight  = -0.5    # threshold=0.10m, spawn-XY reference
wheel_brake.weight     = -0.02   # ungated, omega_threshold=0.0

# Disabled
hip_crossing           = None
thigh_salute           = None
calf_symmetry          = None
hip_symmetry           = None
track_lin_vel_xy_exp.weight = 0.0
track_ang_vel_z_exp.weight  = 0.0
stagnation             = None
climb_progress         = None
uphill_lean            = None
action_rate_l2.weight  = 0.0
dof_acc_l2.weight      = 0.0
```

**Result (training log — iter 10186–10194):**
```
base_contact:      0%      ← robot not falling
bad_orientation:   63%     ← SAME problem — still tipping
position_drift:   -56 to -70  ← SAME scale — still sliding 28–35m
wheel_brake:       0.000   ← wheels locked
time_out:          36%
```

**Visual inspection (play mode):** Robots spawn on inverted pyramid platform, immediately
fall to the ground. Policy has not learned to stand up at all after 200 iterations.

**Root cause (final diagnosis):**
- 200 iterations is insufficient for a policy loaded from a spin checkpoint to relearn
  stable standing on slope terrain with a completely different reward structure.
- The `position_drift_penalty` is still useless when the robot falls — the fallen body
  slides across the floor accumulating -56 to -70/ep that the policy can never avoid.
- **SpinStatic as a concept has a fundamental flaw:** You cannot teach "hold position on
  slope" as a separate stage from "spin" when the base policy (model_9995) was trained
  on a completely different reward structure. The fine-tuning objective is too far from
  the pre-training objective.

---

## 4. Why the Two-Stage Approach Failed

The two-stage curriculum (SpinStatic → Spin) was based on a reasonable hypothesis:
1. Teach wheel braking first (Stage 1)
2. Then add spinning (Stage 2)

**The flaw:** The policy's ability to STAND UP on slope terrain depends on:
- Correct joint posture (tuned by hip/thigh/calf penalties in rocky slope training)
- The specific reward balance that was used during rocky slope training

When we strip all the joint penalties and change the terrain, we're essentially asking
`model_9995` to transfer to a fundamentally different environment. It cannot do this
in 200 iterations — the network weights are too strongly biased toward the rocky slope
reward structure.

**What actually happened:** The robot falls immediately because:
1. `model_9995` was trained on smooth pyramid slopes with forward velocity commands
2. SpinStatic environment has ZERO velocity commands + different reward structure  
3. The network's learned standing behavior was conditioned on having velocity gradients
4. Without those gradients, the policy outputs random joint torques → falls

---

## 5. What Actually Works (Prior to This Campaign)

| Checkpoint | Task | Status |
|-----------|------|--------|
| `model_1499.pt` (go2w_velocity_fast_flat) | Flat terrain, fast forward | ✅ Working |
| `model_8996.pt` (go2w_velocity_rough) | Rough terrain, moderate slopes | ✅ Working |
| `model_13994.pt` (go2w_velocity_rocky_slope) | Rocky 15°–35° slopes, boulders | ✅ Working |
| `model_9995.pt` (go2w_velocity_spin) | Spin attempt v4 — FAILED | ❌ Premature convergence |

**What `model_9995` does well:** Survive on slope terrain. Angular velocity tracking is
poor (~1°/s) but the robot stays upright.

**What `model_9995` does NOT do:** Actually spin. `track_ang_vel_z_exp ≈ 0` in the
training logs.

---

## 6. Root Cause of v4 Spin Failure (model_9995)

The v4 spin training (`go2w_velocity_spin/2026-07-15_08-55-35`) ran for ~10,000 iterations
but never learned spinning because:

**Terrain:** `RockyPyramidSlopeCfg` with `platform_width=0.0` (default).  
→ Robot spawns AT the pyramid apex (highest point) → immediately slides off in all
directions → episode terminates in ~18 steps → policy never gets enough steps to
learn spinning.

This is a **1-line bug**: `platform_width` was not set, so robots spawn at the apex
singularity. Adding `platform_width=1.5` would have fixed this from the start.

The training log showed `base_contact > 60%` from the very first iteration, meaning
60%+ of episodes were immediate spawn deaths. The policy learned to survive slightly
better but never had enough steps in a stable position to discover spinning.

---

## 7. Current State of Codebase

### spin_env_cfg.py (v5) — CORRECT, DO NOT CHANGE
The v5 terrain is properly designed:
- Rows 0–4: `HfRandomUniformTerrainCfg` flat rough (±2cm) — learn spin with grip
- Rows 5–9: `RockyPyramidSlopeCfg` 5°–25°, `platform_width=1.5` — no apex death

The reward structure is correct:
- `track_ang_vel_z_exp` weight = +3.0 (dominant positive)
- `track_lin_vel_xy_exp` weight = -0.5 (drift penalty)
- `wheel_brake` weight = -0.05 (gated at |omega_cmd| > 0.3)
- `stagnation = None`, `climb_progress = None`

### spin_static_env_cfg.py — ABANDONED
Three iterations of this file all failed. The concept is flawed.
The file remains in the codebase but should not be used.

### rewards.py — position_drift_penalty() — KEPT
This function is correctly implemented and may be useful in future waypoint-holding
contexts. It uses `default_root_state[:, :2]` as the spawn reference (correct).

---

## 8. The Minimal Fix That Would Have Worked

**ONE change to the v4 spin training environment:**

```python
# In spin_env_cfg.py (v4), the sub_terrain should have had:
"rocky_slope": RockyPyramidSlopeCfg(
    proportion=1.0,
    slope_min_deg=15.0,
    slope_max_deg=35.0,
    platform_width=1.5,  # ← THIS IS THE ONLY MISSING LINE
    ...
)
```

With `platform_width=1.5`, robots spawn on a flat 1.5m pad at the tile centre.
The slope begins 0.75m away. The robot has time to establish balance before sliding
off. The 10,000 iterations already accumulated in `model_9995` would then be building
on a stable base.

**Estimated additional training needed:** 500–1000 iterations on fixed terrain.

---

## 9. Recommendations for Fresh Start

See `docs/spin_policy_next_steps.md` for the clean plan.

**Summary:**
1. Do NOT use SpinStatic. Delete or ignore `spin_static_env_cfg.py`.
2. The `spin_env_cfg.py` v5 terrain is already correct. Use it directly.
3. Start from `model_8996.pt` (rougher weights, less wheel-drive prior than model_13994).
4. Train for 1000 iterations on the v5 spin environment.
5. The only reliable metric is `track_ang_vel_z_exp` in TensorBoard — it must rise.

---

## 10. Files to Keep vs. Delete

| File | Action |
|------|--------|
| `spin_env_cfg.py` | **KEEP** — v5 terrain is correct |
| `spin_static_env_cfg.py` | **IGNORE** — concept failed |
| `rewards.py` (position_drift_penalty) | **KEEP** — useful for future waypoint holding |
| `rsl_rl_ppo_cfg.py` (Go2wSpinStaticPPORunnerCfg) | **IGNORE** — won't be used |
| All `go2w_velocity_spin_static/` logs | **IGNORE** — failed experiments |
