# Turn Policy Development Log

**Robot:** Unitree Go2W (wheeled quadruped)  
**Goal:** Train a pivot-turn policy so the navigator can stop, reorient, and reroute around terrain obstacles (boulders, steep slopes, slip zones)  
**Status:** ✅ COMPLETE — flat terrain turn policy ready

---

## Best Policy

**Task:** `RexmiRl-Go2w-Velocity-Turn-B-v0` (Phase B tight-arc/near-pivot)  
**Checkpoint:** `model_10992.pt`  
**Run dir:** `go2w_velocity_turn_b/2026-07-25_13-51-28`  
**Saved to:** `logs/rsl_rl/best_policies/go2w_turn_flat_v6_model_10992.pt`  
**Terrain:** Flat  
**Capability:** Pivot/spin turn using coordinated micro-tap foot gait. Spins cleanly both CW and CCW. Command omega range: `(-0.2, 0.2) rad/s`.

**Play command:**
```bash
conda activate env_isaacsim && cd /home/susan/rexmi_rl && \
python scripts/play.py \
    --task RexmiRl-Go2w-Velocity-Turn-B-Play-v0 \
    --load_run go2w_velocity_turn_b/2026-07-25_13-51-28 \
    --checkpoint model_10992.pt
```

**Play config omega** (in `Go2wTurnBEnvCfg_PLAY`): `(0.15, 0.2) rad/s` — comfortable demo speed.  
To adjust spin speed in demo, change only the play config omega range — no retraining needed.

---

## Overview

The robot already has three working locomotion policies:
- `fast_forward` — flat surface, high speed
- `plain_rough` — rough terrain, general obstacle traversal
- `steep_rough` — high-slope handling

The missing capability: **reorientation**. The robot cannot yet stop and change heading. Without this, autonomous navigation cannot reroute around obstacles. The turn policy is the 4th policy needed to close this gap.

The navigator's use: detect obstacle → stop → invoke turn policy → spin to new heading → resume locomotion policy.

---

## Architecture: Curved-Path Curriculum

Direct pivot-turn training (vx=0, omega=0.3) was attempted 17+ times and failed every time. The root cause: a policy that only knows how to go straight has **no prior** for the motor pattern needed to spin in place. Every exploit (body tilt, joint wind-up, leg oscillation) earns more reward than the correct solution because the correct solution requires discovering a completely new motor pattern.

**Solution: Curved-path curriculum**

| Phase | Task | vx | omega | Turn radius |
|-------|------|----|-------|-------------|
| A | Wide arc | 0.3-0.5 m/s | ±0.15-0.3 rad/s | 1.0-3.3 m |
| B | Tight arc/near-pivot | 0.0-0.2 m/s | ±0.15-0.2 rad/s | 0-1.3 m |
| C | Pure pivot | 0.0 m/s | ±0.15-0.3 rad/s | 0 m |

Phase A starts from the robot's existing forward-walking prior. A wide leftward curve is 95% the same gait with slightly differential leg/wheel effort. Phase B narrows the radius. Phase C is pure pivot warm-started from Phase B.

---

## Exploits Discovered and Countermeasures Added

### Exploit 1: Body twist / trunk winding
**What:** Robot winds hip joints to rotate the trunk (base link) into a severely tilted pose (40-70° tilt). The trunk rotation registers as yaw on the IMU. Feet do not move.  
**Countermeasure:** `trunk_stability_penalty` — threshold-based penalty (free within ±15°, costly beyond).

### Exploit 2: Body rocking
**What:** Robot oscillates body side-to-side, generating ±ω_z IMU readings. Instantaneous `track_ang_vel_z_exp` is satisfied; net heading change is zero.  
**Countermeasure:** `heading_progress` reward — measures actual quaternion yaw change Δyaw per step. Only monotonic real rotation earns reward.

### Exploit 3: Yaw stagnation (subtle rocking)
**What:** Robot rocks asymmetrically — slightly more in commanded direction than back. Per-step Δyaw is tiny but positive. Net yaw over 2 seconds < 5°.  
**Countermeasure:** `yaw_stagnation_penalty` — rolling 100-step window. If accumulated yaw < 20° over 100 steps, penalty fires.

### Exploit 4: Crawl-spin
**What:** Robot lowers body nose-down to ~0.10 m (from normal 0.35 m), reducing rotational inertia. Pivots around nose with rear knees dragging.  
**Countermeasure:** `base_height_penalty` — fires when body drops more than 7 cm below spawn height.

### Exploit 5: Wheel-spin pseudo-rotation
**What:** Wheels spin differentially to produce an angular velocity sensor reading without actually pivoting the body. `track_ang_vel_z_exp` fires; heading doesn't change.  
**Countermeasure:** `wheel_lock` penalty — penalises wheel angular velocity during turn commands.

### Exploit 6: Joint-limit arch (body arch)
**What:** Robot pushes legs to joint limits to generate body arch/torque. Actual heading changes but body arches severely.  
**Countermeasure:** `dof_pos_limits=-2.0` (not -10), `pivot_step_coordination` reward.

### Exploit 7: All-4-wheels-glued binding
**What:** Robot keeps all 4 wheels on ground and twists body without stepping. Satisfies yaw command via body torsion.  
**Countermeasure:** `foot_alternation_reward` — rewards having 1-2 feet off ground during turn command.

### Exploit 8: Foot lift without direction
**What:** Robot lifts a foot but swings it randomly. `foot_alternation_reward` fires without any actual coordinated stepping.  
**Countermeasure:** `pivot_step_coordination` — rewards feet swinging in the correct lateral direction (front feet +y, rear feet -y for left turn).

---

## Reward Terms in Turn Policy (Full List)

| Term | Weight | Purpose |
|------|--------|---------|
| `track_ang_vel_z_exp` | +2.0 | Primary: match commanded yaw rate |
| `track_lin_vel_xy_exp` | +0.5 (Phase B) | Match small forward velocity |
| `heading_progress_turn` | +100.0 (Phase B) | Actual quaternion yaw change |
| `foot_alternation` | +1.5 | Reward foot lift during turn (anti-binding) |
| `pivot_step_coord` | +2.0 | Reward correct lateral foot direction |
| `trunk_stability` | -5.0 @ 15° | Prevent body arch/tilt exploit |
| `flat_orientation_l2` | -3.0 | General tilt penalty |
| `base_height` | -20.0 | Prevent crawl-spin posture |
| `wheel_lock` | -0.25 | Penalise wheel spin during pivot |
| `yaw_stagnation` | -2.0 | Prevent asymmetric rocking exploit |
| `position_drift` | -0.5 @ 0.25m | Prevent lateral drift during arc |
| `ang_vel_xy_l2` | **-0.05** | Unitree official — LOCKED, do not increase. Lateral body sway IS the stepping motion during pivot. Increasing this destroys the gait. |
| `dof_pos_limits` | -2.0 | Deter joint limit hitting (not -10: that blocks hip range needed for turns) |
| `foot_air_time` | -1.0 @ 6 steps | Enforce short micro-tap shuffles, penalise long air time |
| `is_alive` | +0.2 | Survival incentive |
| `undesired_contacts` | -4.0 | Penalise body/knee contact |
| `action_rate_l2` | -0.035 | Smooth joint commands, reduce jerk |

---

## Key Code Files

| File | Role |
|------|------|
| `source/rexmi_rl/tasks/locomotion/velocity/config/go2w/turn_env_cfg.py` | Phase A/B/C environment configs |
| `source/rexmi_rl/tasks/locomotion/velocity/mdp/rewards.py` | All custom reward functions |
| `source/rexmi_rl/tasks/locomotion/velocity/mdp/__init__.py` | Reward export list |

---

## Registered Tasks

| Task name | Class | Use |
|-----------|-------|-----|
| `RexmiRl-Go2w-Velocity-Turn-A-v0` | `Go2wTurnAEnvCfg` | Phase A training |
| `RexmiRl-Go2w-Velocity-Turn-A-Play-v0` | `Go2wTurnAEnvCfg_PLAY` | Phase A visual eval |
| `RexmiRl-Go2w-Velocity-Turn-B-v0` | `Go2wTurnBEnvCfg` | Phase B training |
| `RexmiRl-Go2w-Velocity-Turn-B-Play-v0` | `Go2wTurnBEnvCfg_PLAY` | Phase B visual eval |
| `RexmiRl-Go2w-Velocity-Turn-v0` | `Go2wTurnEnvCfg` | Phase C training (pure pivot) |
| `RexmiRl-Go2w-Velocity-Turn-Play-v0` | `Go2wTurnEnvCfg_PLAY` | Phase C visual eval |

---

## Standard Commands

### Visual eval of best policy
```bash
conda activate env_isaacsim && cd /home/susan/rexmi_rl
python scripts/play.py \
    --task RexmiRl-Go2w-Velocity-Turn-B-Play-v0 \
    --load_run go2w_velocity_turn_b/2026-07-25_13-51-28 \
    --checkpoint model_10992.pt
```

### Continue training from best policy (if needed for Phase C)
```bash
conda activate env_isaacsim && cd /home/susan/rexmi_rl
python scripts/train.py \
    --task RexmiRl-Go2w-Velocity-Turn-v0 \
    --headless \
    --load_run go2w_velocity_turn_b/2026-07-25_13-51-28 \
    --checkpoint model_10992.pt
```

---

## Key Lessons Learned

1. **Never train pivot from scratch from a walking prior.** Use curriculum: walk → wide arc → tight arc → pivot.

2. **`dof_pos_limits=-10` blocks coordination.** At -10, the policy stays well inside joint limits, preventing hip abductors from generating lateral foot placement for turns. Use -2.0.

3. **`ang_vel_xy` above -0.05 destroys pivot gait.** The lateral angular velocity during a pivot IS the stepping motion. Increasing this penalty above Unitree's -0.05 suppresses the micro-tap mechanics and causes catastrophic failure. LOCKED at -0.05.

4. **Foot lift without direction is not enough.** `foot_alternation_reward` teaches the robot to lift feet but not where to put them. `pivot_step_coordination` adds the directional signal.

5. **`trunk_stability` at 10° causes binding.** Normal walking gait pitches ±5-8°. Use 15°.

6. **The arch is not a reward exploit — it's a coordination failure.** Fix coordination (`pivot_step_coordination`), not limits.

7. **Always check the official Unitree repo for baseline weights** before setting your own.

8. **`pivot_step_coord` with `min_swing_vel=0.03` biases toward large fast swings.** Lower to 0.01 to reward any intentional lateral movement equally.

9. **Air time reward shaping requires both a ceiling AND a floor.** `foot_alternation_reward` is the floor; `foot_air_time_penalty` is the ceiling. The combination forces the micro-tap gait.

10. **Stats can lie — always do a visual eval before concluding an exploit fired.** The micro-tap gait looked bad in stats but was a breakthrough visually.

11. **Command range upper bound determines gait speed.** To slow the pivot down, reduce the command range ceiling — not damping penalties.

12. **To change demo spin speed, change only the play config omega range — no retraining needed.** The policy executes at whatever omega is commanded.
