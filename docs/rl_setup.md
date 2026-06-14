# REXMI RL — Complete Developer Reference

> **Current status: Phase 4 complete — Terrain Capability Evaluation available**
>
> Phases 1–4 complete. Use `scripts/eval.py` to characterise performance across 36 terrain variants.

This document explains **every design decision** made across all four phases of
the Go2W RL setup.  The goal is that after reading this you understand the full
pipeline from the physical robot model to a trained policy, with zero assumed
prior knowledge.

---

## Table of Contents

1. [Big Picture — What Are We Actually Doing?](#1-big-picture)
2. [The Robot: Unitree Go2W](#2-the-robot-unitree-go2w)
3. [What Is an MDP?](#3-what-is-an-mdp)
4. [Isaac Lab Architecture](#4-isaac-lab-architecture)
5. [File-by-File Walkthrough](#5-file-by-file-walkthrough)
6. [The MDP in Detail (Phase 3+)](#6-the-mdp-in-detail)
7. [PPO Algorithm Explained](#7-ppo-algorithm-explained)
8. [Training Workflow — All Tasks](#8-training-workflow)
9. [Terrain Capability Evaluation](#9-terrain-capability-evaluation)
10. [Reward Engineering Guide](#10-reward-engineering-guide)
11. [Phase History & What We Learned](#11-phase-history)
12. [Phase 5 Roadmap](#12-phase-5-roadmap)

---

## 1. Big Picture — What Are We Actually Doing?

We are teaching a simulated robot to **drive in any commanded direction across
varied terrain** using **Reinforcement Learning (RL)**.

At the highest level:
```
        ┌─────────────────────────────────────────────────────┐
        │                   Training Loop                      │
        │                                                       │
        │   Policy (neural net)  ──actions──▶  Environment     │
        │           ▲                              │            │
        │           │           ◀──obs, reward─── │            │
        └─────────────────────────────────────────────────────┘
```

- **Policy**: A neural network that maps observations → actions
- **Observations**: Velocity, joint angles, IMU, commanded direction, height scan
- **Actions**: Wheel velocity targets + all 12 leg joint position targets (16 DOF)
- **Reward**: Velocity tracking + stability penalties + contact/deviation penalties
- **Environment**: Isaac Sim running 4096 robot copies on GPU simultaneously

The project has progressed through 4 phases, each unlocking more capability:

| Phase | Terrain | Action DOF | Key addition |
|-------|---------|-----------|--------------|
| 1 | Flat | 4 (wheels only) | Robot rolls, legs locked |
| 2 | Flat | 8 (wheels + thighs) | CG shifting — 99% success rate |
| 3 | Flat | 16 (all joints) | Full leg control, spider-walk fix |
| 4 | **Rough** | 16 | Height scanner, terrain curriculum |

---

## 2. The Robot: Unitree Go2W

The Go2W is a **wheeled hybrid quadruped** — it has 4 legs each ending in a wheel.

### Kinematic structure

```
            base (body)
           /     |    \    \
         FL     FR     RL   RR
        hip    hip    hip  hip       ← revolute (abducts leg L/R)
         |      |      |    |
       thigh  thigh  thigh thigh     ← revolute (pitches leg F/B)
         |      |      |    |
       calf   calf   calf  calf      ← revolute (bends lower leg)
         |      |      |    |
       wheel  wheel  wheel wheel     ← continuous (spins freely)
```

### Joint inventory (16 controllable DOF)

| Group | Joints | Type | Actuator (Phase 3+) | Count |
|-------|--------|------|---------------------|-------|
| Hips | `FL/FR/RL/RR_hip_joint` | revolute | soft PD stiffness=5 | 4 |
| Thighs | `FL/FR/RL/RR_thigh_joint` | revolute | soft PD stiffness=5 | 4 |
| Calves | `FL/FR/RL/RR_calf_joint` | revolute | soft PD stiffness=5, damping=0.8 | 4 |
| Wheels | `FL/FR/RL/RR_foot_joint` | continuous | velocity mode stiffness=0, damping=2 | 4 |

### Phase 3+ control strategy

All 16 DOFs are RL-controllable.  The actuator model uses **ImplicitActuatorCfg**
(PhysX internal PD):

```
τ = stiffness × (θ_target + Δθ_action − θ_actual) + damping × (0 − ω_actual)
```

- `stiffness=5` → soft restoring force; RL can easily override default pose
- `stiffness=0` → pure velocity mode for wheels; no position restoring force

---

## 3. What Is an MDP?

An MDP (Markov Decision Process) is the formal mathematical framework for RL.
It has 5 components: **(S, A, T, R, γ)**

| Symbol | Name | In our context |
|--------|------|----------------|
| **S** | State space | Robot velocity, joint angles, IMU, height scan |
| **A** | Action space | 16 floats: 4 wheel velocities + 12 leg position offsets |
| **T** | Transition | PhysX physics simulation |
| **R** | Reward function | Velocity tracking + stability + contact + deviation penalties |
| **γ** | Discount factor | 0.99 (future rewards count almost as much as immediate) |

### Episode structure

```
t=0: Robot spawned at random XY position and orientation on terrain tile
     Velocity command sampled: vx ∈ [-0.5,0.5] m/s, vy ∈ [-0.5,0.5], ωz ∈ [-1,1] rad/s
t=1..N: Policy observes state, outputs 16 actions, physics steps forward
        Reward computed each step
t=T: Episode ends when:
     (a) base touches ground (fell over) → terminated
     (b) tilt > 1.0 rad (57°) from vertical → terminated (bad_orientation)
     (c) 20 seconds elapsed (1000 steps at 50Hz) → timeout
     Then robot is reset to terrain tile matching current curriculum level
```

---

## 4. Isaac Lab Architecture

Isaac Lab is NVIDIA's RL framework built on top of Isaac Sim (PhysX).

### Key concepts

**Vectorised simulation** (the big advantage of Isaac Lab):
- 4096 independent robot copies run **simultaneously** on one GPU
- All PhysX computations are batched into GPU tensor operations
- This is 100× faster than running 4096 separate simulations

**Manager-based environments** (`ManagerBasedRLEnv`):
- The environment behaviour is entirely driven by config classes
- No subclassing needed — you just configure managers:
  - `SceneCfg` → what goes in the world (terrain, robot, sensors)
  - `ObservationsCfg` → what the policy sees
  - `ActionsCfg` → what the policy controls
  - `RewardsCfg` → reward terms and weights
  - `TerminationsCfg` → episode end conditions
  - `EventsCfg` → randomisation at reset/interval
  - `CurriculumCfg` → automatic difficulty scaling

**TerrainGeneratorCfg** (Phase 4):
- Builds a `num_rows × num_cols` grid of terrain tiles
- Each row = one difficulty level (row 0 = flat, row 9 = hardest)
- Sub-terrain types are procedurally generated: stairs, slopes, boxes, rough

### Data flow each physics step

```
1. EventManager  → apply randomisation if interval triggered (e.g. push_robot)
2. ObservationManager → collect sensor readings → obs tensor [N_envs, ~208]
3. ActionManager → decode policy output → 16 joint commands [N_envs, 16]
4. PhysX step    → simulate physics for dt=0.005s × decimation=4 = 0.02s
5. RewardManager → compute reward [N_envs, 1]
6. TerminationManager → check done flags [N_envs, 1]
7. CurriculumManager → update terrain level if applicable
8. If done: reset that env to terrain tile of appropriate difficulty
```

---

## 5. File-by-File Walkthrough

### Package install: `setup.cfg`

Tells pip how to install `rexmi_rl` as a Python package.
Critical for Isaac Lab to discover our environments via `import rexmi_rl`.

```bash
pip install -e .   # from repo root; -e = editable (no re-install after edits)
```

---

### Robot asset: `source/rexmi_rl/assets/go2w.py`

```python
GO2W_CFG = ArticulationCfg(
    spawn = UsdFileCfg("assets/robots/go2w/urdf/go2w/go2w.usd",
                       activate_contact_sensors=True, ...)
    init_state = InitialStateCfg(
        pos=(0, 0, 0.43),
        joint_pos={
            ".*L_hip_joint": 0.1, ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,  "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,       ".*_foot_joint": 0.0,
        }
    )
    actuators = {
        "hip_joints":   ImplicitActuatorCfg(stiffness=5,   damping=0.5)  # soft PD
        "thigh_joints": ImplicitActuatorCfg(stiffness=5,   damping=0.5)  # soft PD
        "calf_joints":  ImplicitActuatorCfg(stiffness=5,   damping=0.8)  # soft PD
        "wheel_joints": ImplicitActuatorCfg(stiffness=0,   damping=2.0)  # vel mode
    }
)
```

**Phase evolution of actuators:**
- Phase 1: `leg_joints` (stiffness=25, locked) + `wheel_joints`
- Phase 2: `hip_calf_joints` (stiffness=25) + `thigh_joints` (stiffness=5) + `wheel_joints`
- Phase 3+: all 3 leg groups stiffness=5 (soft PD, fully RL-controllable)

---

### Base environment: `rough_env_cfg.py`

This file hosts ALL environment config classes:

```
Go2wFlatEnvCfg(LocomotionVelocityRoughEnvCfg)   — Phase 1-3 foundation
    └── Go2wFlatEnvCfg_PLAY                      — visualization variant
    └── Go2wRoughEnvCfg(Go2wFlatEnvCfg)          — Phase 4: adds terrain+scanner
            └── Go2wRoughEnvCfg_PLAY             — visualization variant
```

`Go2wFlatEnvCfg.__post_init__()` overrides:
1. Robot → `GO2W_CFG`
2. Terrain → flat plane, no generator
3. Height scanner → `None` (saves GPU ray-cast time on flat terrain)
4. Actions → 4 wheel velocity + 12 leg position DOFs
5. Rewards → velocity tracking + stability + contact penalties (see §6)
6. Commands → ±0.5 m/s forward/lateral, ±1.0 rad/s yaw
7. Terminations → `base_contact` + `bad_orientation` (1.0 rad limit)
8. Curriculum → `None` (flat terrain has no difficulty levels)

`Go2wRoughEnvCfg.__post_init__()` additionally:
1. Re-enables terrain generator (stairs/slopes/boxes/rough, 10 difficulty rows)
2. Re-creates `RayCasterCfg` height scanner (1.6 m × 1.0 m, 10 cm resolution)
3. Adds `height_scan` observation term (160 dims)
4. Re-enables `terrain_levels_vel` curriculum

---

### Entry-point shim: `flat_env_cfg.py`

A thin re-export of all 4 config classes from `rough_env_cfg.py`.
`__init__.py` uses this module path for all `gym.register()` entry points.

---

### PPO config: `agents/rsl_rl_ppo_cfg.py`

| Config class | Task | Network | Iterations | Obs dims |
|---|---|---|---|---|
| `Go2wFlatPPORunnerCfg` | Flat | [128,128,128] | 1000 | ~48 |
| `Go2wRoughPPORunnerCfg` | Rough | [512,256,128] | 3000 | ~208 |

Key difference: `empirical_normalization=True` for rough terrain — height scan
values span ±0.5 m, which would dominate the network input without normalisation.

---

### Gym registration: `config/go2w/__init__.py`

```python
# Flat terrain
gym.register("RexmiRl-Go2w-Velocity-Flat-v0",      env=Go2wFlatEnvCfg,      ppo=Go2wFlatPPORunnerCfg)
gym.register("RexmiRl-Go2w-Velocity-Flat-Play-v0", env=Go2wFlatEnvCfg_PLAY, ppo=Go2wFlatPPORunnerCfg)
# Rough terrain
gym.register("RexmiRl-Go2w-Velocity-Rough-v0",      env=Go2wRoughEnvCfg,      ppo=Go2wRoughPPORunnerCfg)
gym.register("RexmiRl-Go2w-Velocity-Rough-Play-v0", env=Go2wRoughEnvCfg_PLAY, ppo=Go2wRoughPPORunnerCfg)
```

---

## 6. The MDP in Detail

### Observations (Phase 3+ — flat; Phase 4 adds height scan)

| Term | Dim | Description |
|------|-----|-------------|
| `base_lin_vel` | 3 | Linear velocity of base (x,y,z) in body frame [m/s] |
| `base_ang_vel` | 3 | Angular velocity (roll,pitch,yaw) [rad/s] |
| `projected_gravity` | 3 | Gravity vector in body frame — encodes tilt |
| `velocity_commands` | 3 | Commanded (vx, vy, ωz) |
| `joint_pos` | 16 | All joint positions relative to default [rad] |
| `joint_vel` | 16 | All joint velocities [rad/s] |
| `actions` | 16 | Previous 16 actions (history) |
| `height_scan` *(Phase 4)* | 160 | Terrain heights under 1.6m×1.0m grid [m] |
| **Total flat** | **~60** | (exact depends on base class fields) |
| **Total rough** | **~220** | |

**Noise** is added to obs during training for sim-to-real robustness:
- `base_lin_vel`: ±0.1 m/s
- `base_ang_vel`: ±0.2 rad/s
- `projected_gravity`: ±0.05
- `height_scan`: ±0.1 m (Phase 4)

### Actions (Phase 3+)

```
output ∈ [-1, 1]¹⁶  split into:
  [0:4]   → wheel velocity targets  × scale=10 rad/s
  [4:8]   → thigh position offsets  × scale=0.5 rad  (from default)
  [8:12]  → hip   position offsets  × scale=0.3 rad  (from default)
  [12:16] → calf  position offsets  × scale=0.5 rad  (from default)
```

`use_default_offset=True` means action=0 → hold the default stance.

### Reward function (Phase 3+)

| Term | Weight | Formula | Purpose |
|------|--------|---------|---------|
| `track_lin_vel_xy_exp` | +2.0 | exp(-‖vxy - cmd‖² / 0.25) | Match forward/lateral speed |
| `track_ang_vel_z_exp` | +0.75 | exp(-(ωz - ωz_cmd)² / 0.25) | Match turning rate |
| `is_alive` | +0.2 | 1 while running, 0 at termination | Stay upright |
| `flat_orientation_l2` | -2.5 | ‖gravity_projected_xy‖² | Keep base level |
| `lin_vel_z_l2` | -2.0 | vz² | No bouncing |
| `ang_vel_xy_l2` | -0.5 | ωx² + ωy² | No pitch/roll wobble |
| `dof_torques_l2` | -1e-5 | ‖τ‖² | Minimise energy |
| `dof_acc_l2` | -2.5e-7 | ‖q̈‖² | Smooth accelerations |
| `action_rate_l2` | -0.01 | ‖a_t - a_{t-1}‖² | Smooth commands |
| `undesired_contacts` | -1.0 | contact force on hip/thigh/calf > 1N | No spider-walking |
| `dof_pos_limits` | -0.1 | joints beyond soft limit | Stay within URDF range |
| `leg_deviation` | -0.2 | Σ\|joint - default\| (leg joints) | No excessive leg spread |

**`undesired_contacts`** and **`leg_deviation`** were added after Phase 3 revealed
two reward-gaming behaviours: spider-walking (legs touching ground) and extreme
hip spreading.

### Terminations

| Condition | Type | Note |
|-----------|------|------|
| `base_contact` (base link touches ground) | failure | Legs/wheels contact is normal |
| `bad_orientation` (tilt > 1.0 rad = 57°) | failure | Catches falls before base hits |
| 20 seconds elapsed (1000 steps) | timeout | Success — episode ran to completion |

### Velocity commands

```
vx  ∈ [-0.5, 0.5] m/s    (forward/backward)
vy  ∈ [-0.5, 0.5] m/s    (lateral — all 4 wheels + legs for sideways)
ωz  ∈ [-1.0, 1.0] rad/s  (turning)
```

Commands are re-sampled every ~10 seconds (UniformVelocityCommandCfg default).

### Domain randomisation (Events)

| Event | When | Effect |
|-------|------|--------|
| `add_base_mass` | startup | ±1 to +3 kg added to base link |
| `base_external_force_torque` | reset | Reference body for force application |
| `reset_robot_joints` | reset | All joints reset to default (scale=1.0) |
| `reset_base` | reset | Random XY ±0.5 m, yaw ±π |
| `push_robot` | interval (~10 s) | Random impulse — tests recovery |

---

## 7. PPO Algorithm Explained

PPO (Proximal Policy Optimisation, Schulman et al. 2017) is the standard
algorithm for locomotion RL in Isaac Lab.

### The training loop

```
for iteration in range(max_iterations):
    # ── Rollout phase ────────────────────────────────────────────
    for step in range(24):  # 24 steps × 4096 envs = 98,304 transitions
        obs = env.get_observations()          # [4096, ~60 or ~220]
        actions, log_probs, values = policy(obs)
        obs_next, rewards, dones = env.step(actions)
        store (obs, actions, log_probs, values, rewards, dones)

    # ── Compute returns and advantages ───────────────────────────
    advantages = GAE(rewards, values, dones, γ=0.99, λ=0.95)
    returns = advantages + values

    # ── Update phase (5 epochs, 4 mini-batches) ──────────────────
    for epoch in range(5):
        for mini_batch in split(data, 4):
            ratio = exp(new_log_probs - old_log_probs)
            policy_loss = -mean(min(ratio × A, clip(ratio, 0.8, 1.2) × A))
            value_loss  = MSE(new_values, returns)
            loss = policy_loss + 1.0 × value_loss - 0.01 × entropy
            loss.backward(); clip_grad_norm(1.0); optimizer.step()
```

### What the training metrics mean

| Metric | Good sign | Bad sign |
|--------|-----------|----------|
| `mean_reward` | Increasing | Flat after 50 iters |
| `Episode_length` | → 1000 (full timeout) | Stuck < 100 |
| `track_lin_vel_xy_exp` | → 2.0 (max weight) | < 0.5 |
| `undesired_contacts` | → 0 | Increasingly negative |
| `leg_deviation` | Small negative, stable | Growing negative (spreading) |
| `value_loss` | Decreasing | Exploding |
| `entropy` | Slowly decreasing | Collapses to near 0 |

### Final Phase 3 metrics (reference baseline)

```
Episode_length:               ~1000 (99.8% timeout)
track_lin_vel_xy_exp:         1.88 / 2.0  (94% max)
track_ang_vel_z_exp:          0.69 / 0.75 (92% max)
undesired_contacts:           ≈ 0.000  ✓ no spider-walking
flat_orientation_l2:          ≈ -0.006  ✓ nearly flat
leg_deviation:                ≈ -0.22   (stable, not worsening)
bad_orientation terminations: 0.2%
```

---

## 8. Training Workflow — All Tasks

### Quick reference

Run all commands with `conda activate env_isaacsim` active first.
`scripts/train.py` and `scripts/play.py` self-register `rexmi_rl` environments
so no extra PYTHONPATH setup is needed.

```bash
# Flat terrain (Phase 1-3 baseline, ~8 min, 1000 iters)
python scripts/train.py --task RexmiRl-Go2w-Velocity-Flat-v0 --headless

# Rough terrain (Phase 4, ~25-30 min, 3000 iters, from scratch)
python scripts/train.py --task RexmiRl-Go2w-Velocity-Rough-v0 --headless

# Resume a run from the latest checkpoint
python scripts/train.py --task RexmiRl-Go2w-Velocity-Rough-v0 --headless --resume

# Quick smoke test (128 envs — verifies config loads without errors)
python scripts/train.py --task RexmiRl-Go2w-Velocity-Rough-v0 --num_envs 128

# Visualise flat policy
python scripts/play.py --task RexmiRl-Go2w-Velocity-Flat-Play-v0

# Visualise rough policy
python scripts/play.py --task RexmiRl-Go2w-Velocity-Rough-Play-v0

# Watch TensorBoard (flat)
tensorboard --logdir logs/rsl_rl/go2w_velocity_flat

# Watch TensorBoard (rough)
tensorboard --logdir logs/rsl_rl/go2w_velocity_rough
```

> **Note on run.sh**: `run.sh` is a convenience wrapper but has historically had a
> PYTHONPATH bug that shadowed the installed `rsl_rl` package.  Always prefer the
> direct `python scripts/...` invocation above.

### Setup (one-time)

```bash
# 1. Create .env with Isaac Lab path
echo "ISAACLAB_DIR=/home/susan/IsaacLab" > .env

# 2. Activate conda env
conda activate env_isaacsim

# 3. Install package in editable mode
pip install -e .

# 4. Verify
python -c "import rexmi_rl; print('OK')"
```

### GPU memory requirements

| Num envs | Approx VRAM (flat) | Approx VRAM (rough + scanner) |
|----------|--------------------|-------------------------------|
| 128 | ~4 GB | ~5 GB |
| 1024 | ~6 GB | ~8 GB |
| 4096 | ~10 GB | ~12 GB |

### Checkpoint locations

```
logs/rsl_rl/
  go2w_velocity_flat/
    <date>_<time>/
      model_<iter>.pt     ← policy checkpoints
      params/             ← hydra config snapshot
  go2w_velocity_rough/
    <date>_<time>/
      model_<iter>.pt
```

---

## 9. Terrain Capability Evaluation

`scripts/eval.py` loads a trained rough-terrain checkpoint and runs it on **36 individually
parameterised terrain variants**, each at a single fixed difficulty.  The result is a table
that tells you exactly where the policy succeeds and where it gives up.

### Why not characterise the training terrain?

The training terrain uses *ranges* (e.g. `step_height_range=(0.05, 0.23)` m), so every tile
has a different difficulty.  You cannot say "it failed at 15 cm steps" because each tile
mixes difficulties.  The eval environment fixes each parameter to a single value so the
measurement is unambiguous.

### How the eval config differs from training

`eval_env_cfg.py` defines `Go2wEvalEnvCfg(Go2wRoughEnvCfg)` with these overrides:

| Setting | Training | Eval |
|---------|----------|------|
| `lin_vel_x` command | random ∈ [-0.5, 0.5] | fixed = 0.5 m/s |
| `lin_vel_y` command | random ∈ [-0.5, 0.5] | fixed = 0.0 |
| `ang_vel_z` command | random ∈ [-1.0, 1.0] | fixed = 0.0 |
| Terrain generator | mixed types, ranges | single type, fixed value |
| Curriculum | `terrain_levels_vel` | disabled |
| Sensor noise | enabled | disabled |
| `push_robot` | enabled | disabled |
| `num_envs` | 4096 | 50 |

### Terrain variants (36 total)

| Group | Variants | Parameter |
|-------|---------|-----------|
| `stairs_up` | 9 | step height: 3, 5, 8, 10, 12, 15, 18, 20, 23 cm |
| `stairs_down` | 9 | step height: 3, 5, 8, 10, 12, 15, 18, 20, 23 cm |
| `boxes` | 6 | box height: 3, 5, 8, 10, 15, 20 cm |
| `slope` | 7 | slope angle: 2, 5, 8, 10, 15, 20, 23° |
| `rough` | 5 | noise amplitude: 2, 4, 6, 8, 10 cm |

### Metrics collected per variant

| Metric | Meaning | Target |
|--------|---------|--------|
| `tracking_ratio` | mean(actual forward vel) / 0.5 m/s | 1.0 = perfect |
| `survival_rate` | % episodes reaching timeout (not falling) | 100% |
| `mean_ep_len` | mean steps per episode | 1000 (max) |
| `mean_dist_m` | estimated distance per episode (m) | ~10 m |

A `←` flag marks any variant where **tracking < 0.50 or survival < 50%**.

### Commands

> **Finding your checkpoint**
>
> Isaac Lab saves checkpoints under `logs/rsl_rl/go2w_velocity_rough/<run_folder>/`.
> The run folder is named after the date+time training started (e.g. `2026-06-13_23-19-06`).
> The final checkpoint for a 3000-iteration run is `model_2999.pt` (RSL-RL saves at the
> end of the last completed iteration, so 3000 iters → `model_2999.pt`).
>
> To find your latest checkpoint:
> ```bash
> ls logs/rsl_rl/go2w_velocity_rough/
> # shows: 2026-06-13_23-19-06/
> ls logs/rsl_rl/go2w_velocity_rough/2026-06-13_23-19-06/ | grep "\.pt" | sort -V | tail -3
> # shows: model_2800.pt  model_2900.pt  model_2999.pt  ← use the last one
> ```

```bash
# Set your checkpoint path once (copy/paste this for your run)
CKPT=logs/rsl_rl/go2w_velocity_rough/2026-06-13_23-19-06/model_2999.pt

# Visual sanity check first — watch 50 robots on 10 cm stairs in GUI
python scripts/eval.py --checkpoint $CKPT --visual --terrain stairs_up_10cm

# Single variant headless (fast, ~1 min)
python scripts/eval.py --checkpoint $CKPT --terrain stairs_up_15cm

# One terrain group only (9 variants, ~3-5 min)
python scripts/eval.py --checkpoint $CKPT --group stairs_up

# Full sweep — all 36 variants (headless, ~10-20 min)
python scripts/eval.py --checkpoint $CKPT

# Visual mode without --terrain: defaults to stairs_up_10cm
python scripts/eval.py --checkpoint $CKPT --visual

# Override number of robots per variant (default 50)
python scripts/eval.py --checkpoint $CKPT --group slope --num_envs 100

# Override steps per variant (default 1000 ≈ 20 s at 50 Hz)
# Use higher values for more reliable survival_rate statistics
python scripts/eval.py --checkpoint $CKPT --terrain rough_10cm --steps 3000

# Save CSV to a custom path
python scripts/eval.py --checkpoint $CKPT --out results/my_eval.csv
```

### Example output

```
  Go2W Terrain Capability Evaluation Results
──────────────────────────────────────────────────────────────────
  STAIRS UP
──────────────────────────────────────────────────────────────────
  Variant                    Tracking  Survival  Ep.len  Dist(m)
──────────────────────────────────────────────────────────────────
  stairs_up_3cm                  0.93     100%     998      9.3
  stairs_up_5cm                  0.91      99%     985      8.9
  stairs_up_8cm                  0.84      97%     941      7.9
  stairs_up_10cm                 0.76      88%     847      6.4
  stairs_up_12cm                 0.61      74%     731      4.5
  stairs_up_15cm                 0.45      63%     612      2.8  ←
  stairs_up_18cm                 0.22      31%     290      1.0  ←
  stairs_up_20cm                 0.11      18%     201      0.4  ←
  stairs_up_23cm                 0.03       6%      87      0.0  ←
  ← tracking < 0.50 or survival < 50%
```

Results are auto-saved to `logs/eval_results/eval_TIMESTAMP.csv`.

### Files added

| File | Purpose |
|------|---------|
| `source/rexmi_rl/tasks/locomotion/velocity/config/go2w/eval_env_cfg.py` | `Go2wEvalEnvCfg` base class + 5 factory functions + `EVAL_VARIANTS` list |
| `scripts/eval.py` | Evaluation runner — headless sweep or visual single-variant |

### Interpreting results for Phase 5

The cliff in `tracking_ratio` and `survival_rate` as difficulty increases identifies the
**training distribution boundary** — the terrain the policy was trained on vs. what it
was never exposed to.  Use these numbers to:

1. **Diagnose**: e.g., if `stairs_up` fails at 15 cm but training used 5–23 cm, the
   curriculum never reached the hardest rows consistently.
2. **Target retraining**: increase curriculum convergence by extending `max_iterations`,
   or add harder sub-terrains to push the policy to explore high-difficulty rows.
3. **Set safety limits**: for deployment, the `tracking > 0.7 AND survival > 80%` boundary
   defines the terrain the robot can reliably navigate.

---

## 10. Reward Engineering Guide

### Diagnosing misbehaviour

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Robot tips immediately | `bad_orientation` too tight | Relax limit (e.g. 0.8→1.0 rad) |
| Robot stands still | `is_alive` weight too high (reward gaming) | Reduce to 0.2 |
| Robot moves wrong way | Velocity command range too large | Reduce to ±0.3 m/s first |
| Robot spider-walks on legs | `undesired_contacts` missing | Add with weight=-1.0 on hip/thigh/calf bodies |
| Legs splay extremely wide | `leg_deviation` missing | Add `joint_deviation_l1` weight=-0.2 |
| Wheels spin but not driving | Wheel damping too high | Reduce damping 5.0→2.0 |
| Jerky motion | `action_rate_l2` weight too small | Increase (more negative) |
| Pitches forward driving | `ang_vel_xy_l2` weight too small | Increase -0.05→-0.5 |
| Training plateaus early | Network too small or too few iters | Scale up [128]→[512,256,128] |
| Height scan ignored | No empirical normalisation | Set `empirical_normalization=True` |

### The reward gaming problem

The policy will exploit ANY loophole in the reward:
1. **Phase 1 standing still** → `is_alive` weight 1.0 made standing more profitable
   than moving. Fixed by reducing weight to 0.2.
2. **Phase 3 spider-walking** → Policy walked on leg links while wheels spun in air.
   Fixed by `undesired_contacts` penalty on hip/thigh/calf bodies.
3. **Phase 3 leg spreading** → Policy splayed legs into extreme wide stance.
   Fixed by `joint_deviation_l1` penalty on all leg joints.

### reward weight scaling

Total per-step reward should be approximately O(1) — typically 1–5 for good behaviour,
-1 to -3 for bad behaviour.  Check `mean_reward`; if it's O(0.001) or O(1000),
rescale weights proportionally.

---

## 11. Phase History

### Phase 1 — Wheels only (4 DOF, flat terrain)
**Actions**: 4 wheel velocity targets  
**Result**: Robot rolls stably. ~300 iterations.  
**Key fix**: `bad_orientation` termination — without it, fallen robots wasted 20s
  of episode time instead of resetting.  

### Phase 2 — Thighs unlocked (8 DOF, flat terrain)
**Actions**: + 4 thigh position offsets  
**Result**: 99% timeout rate (robot runs full 1000 steps without falling).  
**Key insight**: Thigh pitch shifts the centre of gravity forward, counteracting
  the nose-down pitch torque from wheel driving — like a person leaning into a scooter.  
**Key fix**: Wheel damping reduced 5.0→2.0 to prevent pitch torque overload.

### Phase 3 — All legs unlocked (16 DOF, flat terrain)
**Actions**: + 4 hip + 4 calf position offsets  
**Result**: ~94% max velocity tracking, near-zero contact violations.  
**Reward gaming observed**:
  - Spider-walking: robot walked on calf/thigh links, wheels in air
  - Leg spreading: wide-stance stability exploit
**Fixes**: `undesired_contacts` on leg bodies (weight=-1.0) +
  `joint_deviation_l1` on all leg joints (weight=-0.2).

### Phase 4 — Rough terrain (16 DOF, procedural terrain + height scan)
**New additions**:
  - TerrainGeneratorCfg: pyramid stairs (up+down), boxes, random rough, slopes
  - RayCasterCfg height scanner: 1.6m×1.0m grid, 10cm resolution, 160 dims
  - Terrain difficulty curriculum: 10 levels, auto-advances with velocity tracking
  - Larger network: [512, 256, 128] for ~220-dim observation
  - Empirical observation normalisation for height scan values
  - 3000 training iterations  
**Training from scratch** (obs space changed — old flat policy weights incompatible).

---

## 12. Phase 5 Roadmap

### Phase 5a: Energy efficiency (elegance pass)
- Add `dof_torques_l2` on leg joints with higher weight to discourage unnecessary
  leg movement when wheels alone are sufficient (energy-aware gait)
- Optionally: reward that increases when legs are near default pose at high speed

### Phase 5b: Custom REXMI robot
- Swap `GO2W_CFG` for `REXMI_CFG` (custom wheel geometry, sensor suite)
- All env/reward configs should transfer directly (same 16-DOF structure)

### Phase 5c: Lunar terrain (Project Chrono integration)
- Integrate lunar regolith deformation model from Project Chrono
- Train on simulated crater terrain with soft soil contact
- Height scanner may need to be replaced with depth camera for deformable surfaces

### Phase 5d: Sim-to-real transfer
- Domain randomisation: friction (0.3–1.2), mass (±20%), motor damping (±30%)
- Deploy to real Go2W hardware using Isaac Lab's `--real_time` mode

---

*Last updated: 2026-06-14 · REXMI Project · Phase 4 complete, eval script added*
