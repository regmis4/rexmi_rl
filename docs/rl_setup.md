# REXMI RL — Complete Developer Reference

> **Current status: Phase 8 in progress — Steep slopes (23°–45°) via two-policy split**
>
> Phases 1–7 complete. `model_8996.pt` (Phase 7 best checkpoint) is the production
> rough-terrain policy — it handles stairs (up/down to 23 cm), boxes (20 cm), rough, and
> moderate slopes (0°–23°) and is **frozen** (no further training).
> Phase 8 trains a SEPARATE dedicated steep-slope policy (`RexmiRl-Go2w-Velocity-SteepSlope-v0`)
> starting from model_8996 weights. The steep-slope env trains exclusively on 23°–45° slopes
> with a very light orientation penalty (`flat_orientation_l2=-0.1`) and a 80° termination
> limit — settings that would cause exploits in the unified rough policy but are safe in
> an isolated slope-only training distribution.

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
12. [Phase 6 Roadmap](#12-phase-6-roadmap)

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

The project has progressed through 5 complete phases:

| Phase | Terrain | Action DOF | Key addition |
|-------|---------|-----------|--------------|
| 1 | Flat | 4 (wheels only) | Robot rolls, legs locked |
| 2 | Flat | 8 (wheels + thighs) | CG shifting — 99% success rate |
| 3 | Flat | 16 (all joints) | Full leg control, spider-walk fix |
| 4 | **Rough** | 16 | Height scanner, terrain curriculum |
| 5 | **Rough** | 16 | `stagnation_penalty` — solved 12 cm cliff |

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
     (b) tilt > limit_angle from vertical → terminated (bad_orientation)
         flat env:              1.0 rad (57°)
         rough env (Phase 7):   1.0 rad (57°) — **frozen at model_8996 state**
         steep-slope env (Ph8): 1.4 rad (80°) — 35° headroom above 45° max slope
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

This file hosts the core config classes:

```
Go2wFlatEnvCfg(LocomotionVelocityRoughEnvCfg)   — Phase 1-3 foundation (scale=10, ±0.5 m/s)
    └── Go2wFlatEnvCfg_PLAY                      — visualization variant
    └── Go2wRoughEnvCfg(Go2wFlatEnvCfg)          — Phase 4: adds terrain+scanner
            └── Go2wRoughEnvCfg_PLAY             — visualization variant
```

### Fast flat config: `fast_flat_env_cfg.py`

Sibling to the rough env — inherits from `Go2wFlatEnvCfg` without touching the
`Go2wRoughEnvCfg` inheritance chain:

```
Go2wFlatEnvCfg                     — base (scale=10, ±0.5 m/s forward)
    └── Go2wFastFlatEnvCfg         — scale=40, forward up to 2.0 m/s
            └── Go2wFastFlatEnvCfg_PLAY
```

Overrides vs. `Go2wFlatEnvCfg`:

| Setting | Standard flat | Fast flat |
|---------|--------------|-----------|
| Wheel action `scale` | 10 rad/s | **40 rad/s** (2.0 m/s max) |
| `lin_vel_x` command | (-0.5, 0.5) m/s | **(-0.5, 2.0) m/s** |
| `action_rate_l2` | -0.01 | **-0.02** |
| `ang_vel_xy_l2` | -0.5 | **-0.8** |

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
| `Go2wFlatPPORunnerCfg` | Flat (standard) | [128,128,128] | 1000 | ~60 |
| `Go2wFastFlatPPORunnerCfg` | Fast flat (2 m/s) | [128,128,128] | **1500** | ~60 |
| `Go2wRoughPPORunnerCfg` | Rough (Phase 7 frozen) | [512,256,128] | 3000 | ~220 |
| `Go2wSteepSlopePPORunnerCfg` | **Steep-slope (Phase 8)** | [512,256,128] | **2000** | **~220** |

Key differences: `empirical_normalization=True` for rough/steep-slope terrain — height scan
values span ±0.5 m, which would dominate the network input without normalisation.
`Go2wSteepSlopePPORunnerCfg` inherits the rough network identically (same obs/action space,
same architecture) and only differs in `experiment_name="go2w_velocity_steep_slope"`,
`max_iterations=2000`, and `save_interval=50`.  model_8996 weights load directly.

---

### Gym registration: `config/go2w/__init__.py`

```python
# Flat terrain — standard (±0.5 m/s)
gym.register("RexmiRl-Go2w-Velocity-Flat-v0",              env=Go2wFlatEnvCfg,              ppo=Go2wFlatPPORunnerCfg)
gym.register("RexmiRl-Go2w-Velocity-Flat-Play-v0",         env=Go2wFlatEnvCfg_PLAY,         ppo=Go2wFlatPPORunnerCfg)
# Fast flat terrain — high-speed forward up to 2.0 m/s (wheel scale=40, train from scratch)
gym.register("RexmiRl-Go2w-Velocity-FastFlat-v0",          env=Go2wFastFlatEnvCfg,          ppo=Go2wFastFlatPPORunnerCfg)
gym.register("RexmiRl-Go2w-Velocity-FastFlat-Play-v0",     env=Go2wFastFlatEnvCfg_PLAY,     ppo=Go2wFastFlatPPORunnerCfg)
# Rough terrain — height scanner + curriculum (PRODUCTION: model_8996, Phase 7 frozen)
gym.register("RexmiRl-Go2w-Velocity-Rough-v0",             env=Go2wRoughEnvCfg,             ppo=Go2wRoughPPORunnerCfg)
gym.register("RexmiRl-Go2w-Velocity-Rough-Play-v0",        env=Go2wRoughEnvCfg_PLAY,        ppo=Go2wRoughPPORunnerCfg)
# Steep-slope terrain — Phase 8 dedicated policy (23°–45°, start from model_8996)
gym.register("RexmiRl-Go2w-Velocity-SteepSlope-v0",        env=Go2wSteepSlopeEnvCfg,        ppo=Go2wSteepSlopePPORunnerCfg)
gym.register("RexmiRl-Go2w-Velocity-SteepSlope-Play-v0",   env=Go2wSteepSlopeEnvCfg_PLAY,   ppo=Go2wSteepSlopePPORunnerCfg)
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

### Reward function (Phase 7, current)

Terms marked *(rough only)* are added only in `Go2wRoughEnvCfg.__post_init__()`.

| Term | Weight | Formula | Purpose |
|------|--------|---------|---------|
| `track_lin_vel_xy_exp` | +2.0 | exp(-‖vxy - cmd‖² / 0.25) | Match forward/lateral speed |
| `track_ang_vel_z_exp` | +0.75 | exp(-(ωz - ωz_cmd)² / 0.25) | Match turning rate |
| `is_alive` | +0.2 | 1 while running, 0 at termination | Stay upright |
| `climb_progress` *(rough only)* | **1.0** *(weight baked in)* | **0.4×** max(0,vz) flat terrain; **1.5×** max(0,vz) near obstacle (height scan gated) | Reward climbing; reduced on flat to prevent bouncing exploit |
| `flat_orientation_l2` | **-0.8 (both flat and rough env, Phase 7 frozen)** | ‖gravity_projected_xy‖² | Keep base level; steep-slope env overrides to -0.1 to allow sustained body tilt on 23°–45° slopes |
| `lin_vel_z_l2` | **-1.5** | vz² | Damp bouncing (restored from -0.3 to kill bouncing exploit) |
| `ang_vel_xy_l2` | -0.5 | ωx² + ωy² | No pitch/roll wobble |
| `dof_torques_l2` | -1e-5 | ‖τ‖² | Minimise energy |
| `dof_acc_l2` | -2.5e-7 | ‖q̈‖² | Smooth accelerations |
| `action_rate_l2` | -0.01 | ‖a_t - a_{t-1}‖² | Smooth commands |
| `undesired_contacts` | -1.0 | contact force on hip/thigh/calf > 1N | No spider-walking |
| `dof_pos_limits` | -0.1 | joints beyond soft limit | Stay within URDF range |
| `leg_deviation` | -0.05 | Σ\|joint - default\| (leg joints) | Soft bias to default stance (relaxed for climbing) |
| `stagnation` *(rough only)* | -1.5 | 1 when fwd commanded but \|vx\| < 0.05 m/s | Escape stuck states |

**Phase 6 weight changes vs. Phase 5:**
- `flat_orientation_l2`: -2.5 → **-0.8** — pitching 30° to climb was more costly than stagnation
- `lin_vel_z_l2`: -2.0 → **-0.3** — upward velocity (climbing) was being penalised
- `leg_deviation`: -0.2 → **-0.05** — calf extension needed to lift wheels was suppressed
- `stagnation` weight: -0.5 → **-1.5** — tripled pressure to escape stuck states
- `climb_progress`: new term, **+1.5** terrain-blind — directly rewards rising above obstacle level

**Phase 7 weight changes vs. Phase 6** (fix for flat-terrain bouncing regression):
- `lin_vel_z_l2`: -0.3 → **-1.5** — restored; makes flat-terrain bouncing unprofitable
- `climb_progress`: terrain-blind weight +1.5 → **hybrid** (0.4 on flat, 1.5 near obstacle)
  - Obstacle gated by height scanner: `max(ray_hits_z − median(ray_hits_z)) > 0.10 m`
  - RewardTermCfg `weight=1.0`; effective weight returned from function itself

**`undesired_contacts`** and **`leg_deviation`** were added after Phase 3 revealed
two reward-gaming behaviours: spider-walking (legs touching ground) and extreme
hip spreading.

### Terminations

| Condition | Type | Note |
|-----------|------|------|
| `base_contact` (base link touches ground) | failure | Legs/wheels contact is normal |
| `bad_orientation` — flat env: tilt > 1.0 rad (57°); rough env (Phase 7 frozen): 1.0 rad (57°); steep-slope env (Phase 8): 1.4 rad (80°) | failure | Steep-slope env only relaxed — 35° headroom above 45° max slope; rough env unchanged at model_8996 state |
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
# ── Flat terrain (Phase 1-3 baseline, ~8 min, 1000 iters) ─────────────────
python scripts/train.py --task RexmiRl-Go2w-Velocity-Flat-v0 --headless

# Visualise flat policy (standard speed, ±0.5 m/s)
python scripts/play.py --task RexmiRl-Go2w-Velocity-Flat-Play-v0

# Watch TensorBoard (flat)
tensorboard --logdir logs/rsl_rl/go2w_velocity_flat

# ── Fast flat terrain (high-speed, ~12 min, 1500 iters, from scratch) ──────
# Forward up to 2.0 m/s; backward/lateral/yaw unchanged.
# IMPORTANT: wheel scale changed (10→40 rad/s) — old flat checkpoints incompatible.
python scripts/train.py --task RexmiRl-Go2w-Velocity-FastFlat-v0 --headless

# Visualise fast flat policy (up to 2 m/s forward)
python scripts/play.py --task RexmiRl-Go2w-Velocity-FastFlat-Play-v0

# Watch TensorBoard (fast flat)
tensorboard --logdir logs/rsl_rl/go2w_velocity_fast_flat

# ── Rough terrain (Phase 4+, ~25-30 min, 1500 iters resume) ───────────────
python scripts/train.py --task RexmiRl-Go2w-Velocity-Rough-v0 --headless

# Resume a run from the latest checkpoint
python scripts/train.py --task RexmiRl-Go2w-Velocity-Rough-v0 --headless --resume

# Phase 8 resume from best Phase 7 checkpoint
python scripts/train.py --task RexmiRl-Go2w-Velocity-Rough-v0 --headless \
    --load_run go2w_velocity_rough/2026-06-14_20-03-41 --checkpoint model_8996.pt

# Quick smoke test (128 envs — verifies config loads without errors)
python scripts/train.py --task RexmiRl-Go2w-Velocity-Rough-v0 --num_envs 128

# Visualise rough policy (model_8996 — PRODUCTION, Phase 7 frozen)
python scripts/play.py --task RexmiRl-Go2w-Velocity-Rough-Play-v0

# Watch TensorBoard (rough)
tensorboard --logdir logs/rsl_rl/go2w_velocity_rough

# ── Steep-slope terrain (Phase 8 — dedicated 23°–45° policy) ──────────────
# Trained from model_8996 weights. Separate log dir: go2w_velocity_steep_slope/
# IMPORTANT: do NOT use --resume in between new runs (resets curriculum to 0)
# Use --load_run + --checkpoint to resume from the best existing checkpoint.

# First run — load model_8996 as starting point
python scripts/train.py --task RexmiRl-Go2w-Velocity-SteepSlope-v0 --headless \
    --load_run go2w_velocity_rough/2026-06-14_20-03-41 --checkpoint model_8996.pt

# Resume a subsequent run from the latest steep-slope checkpoint
python scripts/train.py --task RexmiRl-Go2w-Velocity-SteepSlope-v0 --headless --resume

# Visualise steep-slope policy (50 robots on 23°–45° slope tiles)
python scripts/play.py --task RexmiRl-Go2w-Velocity-SteepSlope-Play-v0

# Visualise steep-slope policy with a specific checkpoint
python scripts/play.py --task RexmiRl-Go2w-Velocity-SteepSlope-Play-v0 \
    --load_run go2w_velocity_steep_slope/<date>_<time> --checkpoint model_<iter>.pt

# Watch TensorBoard (steep-slope)
tensorboard --logdir logs/rsl_rl/go2w_velocity_steep_slope

# Evaluate steep-slope policy on the 5 steep-slope variants
CKPT_STEEP=logs/rsl_rl/go2w_velocity_steep_slope/<date>_<time>/model_<iter>.pt
python scripts/eval.py --checkpoint $CKPT_STEEP --group steep_slope
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
  go2w_velocity_flat/          ← standard flat policy (±0.5 m/s)
    <date>_<time>/
      model_<iter>.pt     ← policy checkpoints (save every 50 iters)
      params/             ← hydra config snapshot
  go2w_velocity_fast_flat/     ← high-speed flat policy (up to 2.0 m/s fwd)
    <date>_<time>/
      model_<iter>.pt     ← policy checkpoints (save every 100 iters)
  go2w_velocity_rough/         ← rough terrain policy (height scan + curriculum)
    2026-06-14_20-03-41/       ← PRODUCTION run — model_8996.pt is the best checkpoint
      model_8996.pt       ← Phase 7 production checkpoint (FROZEN — do not retrain)
      model_<iter>.pt     ← other checkpoints (save every 100 iters)
  go2w_velocity_steep_slope/   ← Phase 8 dedicated steep-slope policy (23°–45°)
    <date>_<time>/
      model_<iter>.pt     ← checkpoints (save every 50 iters)
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

### Terrain variants (41 total — Phase 8)

| Group | Variants | Parameter |
|-------|---------|-----------|
| `stairs_up` | 9 | step height: 3, 5, 8, 10, 12, 15, 18, 20, 23 cm |
| `stairs_down` | 9 | step height: 3, 5, 8, 10, 12, 15, 18, 20, 23 cm |
| `boxes` | 6 | box height: 3, 5, 8, 10, 15, 20 cm |
| `slope` | 7 | slope angle: 2, 5, 8, 10, 15, 20, 23° |
| `rough` | 5 | noise amplitude: 2, 4, 6, 8, 10 cm |
| **`steep_slope`** *(Phase 8 new)* | **5** | **slope angle: 25, 30, 35, 40, 45° (35° = Shackleton crater target)** |

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
CKPT=logs/rsl_rl/go2w_velocity_rough/2026-06-14_20-03-41/model_8996.pt

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
**Eval results** (36-variant sweep, `model_2999.pt`, 50 envs, 1000 steps each):

| Terrain | Best | Cliff | Fails at |
|---------|------|-------|----------|
| stairs_up | 0.91 tracking | 12 cm | 15+ cm (tracking drops <0.75) |
| stairs_down | 0.87 tracking | **10 cm** | 12+ cm (frozen, ~14% moving) |
| boxes | 0.90 tracking | 10 cm | 15+ cm (tracking drops <0.45) |
| slope | 0.89–0.93 | none | Excellent all the way to 23° |
| rough | 0.85–0.88 | none | Excellent all the way to 10 cm |

**Root cause of failures**: wheels catch on discrete step faces; robot freezes with
wheels spinning but no forward progress.  No gradient signal to try a different
approach.  Fixed in Phase 5 with `stagnation_penalty`.

### Phase 5 — Stagnation-escape training ✅ COMPLETE

**Diagnosis**: Phase 4 eval sweep revealed the robot freezes on `stairs_down ≥12 cm`
and `boxes ≥15 cm`, spinning wheels in place with near-zero forward progress.

**Change made** (conservative — one term added to rough env only):

```python
# source/rexmi_rl/tasks/locomotion/velocity/mdp/rewards.py
def stagnation_penalty(env, threshold=0.05):
    """Returns 1.0 when commanded forward but barely moving; else 0.0."""
    fwd_vel = env.scene["robot"].data.root_lin_vel_b[:, 0]
    cmd_vel = env.command_manager.get_command("base_velocity")[:, 0]
    return ((cmd_vel > 0.1) & (fwd_vel.abs() < threshold)).float()
```

Wired into `Go2wRoughEnvCfg` with `weight=-0.5, threshold=0.05`.  
Trained 3000 iterations from Phase 4 checkpoint (`--resume`, same obs space).

**Phase 5 eval results** (`eval_2026-06-14_16-00-15.csv`):

| Terrain | Phase 4 tracking | Phase 5 tracking | Δ | Phase 4 moving | Phase 5 moving | Δ |
|---------|------------------|------------------|---|----------------|----------------|---|
| stairs_down_10cm | 0.743 | 0.866 | **+0.123** | 95.4% | 98.6% | +3.2pp |
| **stairs_down_12cm** | **0.450** | **0.820** | **+0.370 🏆** | 63.6% | 97.2% | **+33.6pp** |
| stairs_down_15cm | 0.138 | 0.337 | +0.199 | 18.5% | 49.8% | +31.3pp |
| stairs_down_18cm | 0.126 | 0.205 | +0.079 | 16.1% | 32.3% | +16.2pp |
| stairs_down_23cm | 0.118 | 0.134 | +0.016 | 14.7% | 18.2% | +3.5pp |
| **boxes_15cm** | **0.452** | **0.759** | **+0.307 🏆** | 57.4% | 93.1% | **+35.7pp** |
| boxes_20cm | 0.348 | 0.548 | +0.200 | 45.9% | 71.3% | +25.4pp |
| stairs_up (all) | 0.675–0.909 | 0.732–0.923 | +0.02–0.06 | improved | improved | |
| slope / rough (all) | 0.85–0.93 | 0.90–0.94 | ~+0.03 | ✅ stable | ✅ stable | |

**Key insight from Phase 5**: The 12 cm cliff (= 120% of wheel radius) was solved
completely.  The stagnation penalty gave enough gradient for the policy to discover
"try harder" strategies near the wheel-diameter boundary.  However, at 15–23 cm
(>150% wheel radius), the wheel physically contacts the step's vertical wall face —
no amount of wheel torque can roll over it.  The ONLY path is leg-assisted climbing,
which requires:
1. Pitching the body forward (~30°) — blocked by `flat_orientation_l2 = -2.5`
2. Extending front calves to lift wheels above step edge — suppressed by `leg_deviation = -0.2`
3. Generating positive vz while climbing — penalised by `lin_vel_z_l2 = -2.0`

These three reward contradictions are the root cause of the residual cliff.  Phase 6
resolves them.

**Physical limit discovered**: The Go2W wheel radius is **5 cm** (confirmed in URDF:
`cylinder radius="0.05"`).  Steps > 5 cm always present a flat wall to the wheel.
Steps > ~10 cm (2× radius) are not surmountable by rolling physics alone.
The "gives up" behaviour is not a policy failure — it is the reward function
preventing the policy from using the climbing motion it is physically capable of.

### Phase 6 — Obstacle climbing ✅ COMPLETE

**Phase 6 eval results** (`eval_2026-06-14_18-56-15.csv`, 1500 iters from Phase 5 checkpoint):

Stair climbing — **massive improvement** across the board:

| Terrain | Phase 5 tracking | Phase 6 tracking | Δ |
|---------|------------------|------------------|---|
| **stairs_down_15cm** | 0.337 | **0.795** | **+0.458 🏆** |
| **stairs_down_18cm** | 0.205 | **0.761** | **+0.556 🏆** |
| **stairs_down_20cm** | 0.180 | **0.720** | **+0.540 🏆** |
| **stairs_down_23cm** | 0.134 | **0.585** | **+0.451 🏆** |
| boxes_20cm | 0.548 | **0.738** | **+0.190 ✅** |

Flat/slope/rough — **regression due to bouncing exploit:**

| Terrain | Phase 5 tracking | Phase 6 tracking | Δ |
|---------|------------------|------------------|---|
| slope_2deg | 0.934 | 0.776 | **-0.158 ⬇️** |
| slope_20deg | 0.941 | 0.824 | **-0.117 ⬇️** |
| rough_all (avg) | 0.90–0.92 | 0.80–0.82 | **~-0.10 ⬇️** |

**Root cause of regression (bouncing exploit)**:
- `lin_vel_z_l2` was weakened to -0.3 to allow climbing vz
- `climb_progress` weight was +1.5 with no terrain awareness
- On flat ground, bouncing at 0.15 m/s earned: +1.5×0.15 − 0.3×0.0225 = **+0.218/step** profit
- Policy learned to bounce on flat terrain to farm the climb reward
- Fixed in Phase 7 with sensor-gated dynamic weighting

**Goal**: Teach the robot to pitch its body and use leg extension to climb discrete
steps taller than its wheel radius (5 cm), targeting recovery at `stairs_down ≥15 cm`
and `boxes ≥20 cm`.

**Changes made** (1500 iters resumed from Phase 5 checkpoint — same obs space, no architecture change):

**1. Relax three penalty contradictions:**

| Term | Before | After | Reason |
|------|--------|-------|--------|
| `flat_orientation_l2` | -2.5 | **-0.8** | Body MUST pitch 20-30° to mount a step; -2.5 made that costlier than stagnation |
| `lin_vel_z_l2` | -2.0 | **-0.3** | Climbing produces positive vz; penalising it suppressed the needed motion |
| `leg_deviation` | -0.2 | **-0.05** | Calf extension lifts wheel above step edge; -0.2 suppressed this 4× |
| `stagnation` weight | -0.5 | **-1.5** | Triple pressure: 10 stuck steps = -15 reward, making any escape attempt better than waiting |

**2. Add `climb_progress` reward (new MDP function, rough env only):**

```python
# source/rexmi_rl/tasks/locomotion/velocity/mdp/rewards.py
def climb_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward upward base velocity (vz > 0) when a forward command is active."""
    vz = env.scene["robot"].data.root_lin_vel_w[:, 2]
    cmd_fwd = env.command_manager.get_command("base_velocity")[:, 0]
    has_cmd = cmd_fwd > 0.1
    climb = torch.clamp(vz, min=0.0, max=0.5)   # cap at 0.5 m/s
    return has_cmd.float() * climb
```

Wired into `Go2wRoughEnvCfg` with `weight=+1.5`.

**Reward balance analysis at a 30° climbing pitch:**
```
climb_progress:        +1.5 × 0.3 m/s vz    = +0.45/step
flat_orientation_l2:   -0.8 × (0.52 rad)²   = -0.22/step
lin_vel_z_l2:          -0.3 × (0.09)        = -0.027/step
net:                                          ≈ +0.20/step  ← climbing is PROFITABLE
```

vs. Phase 5 where the same pitch cost:
```
flat_orientation_l2:   -2.5 × (0.52 rad)²   = -0.68/step  ← worse than stagnation (-0.5)
lin_vel_z_l2:          -2.0 × (0.09)        = -0.18/step
```

**Why backtracking is structurally hard**: The MDP uses fixed velocity commands
(forward = 0.5 m/s).  Going backward gives velocity tracking reward ≈ 0 (same as stuck),
with no positive signal.  PPO's credit assignment window (~24 steps × γ=0.99) is too
short to bridge the "back up 2s → successful crossing" causal chain.  Climbing is
the reliable engineering path; backtracking would require a hierarchical policy
or waypoint-based reward shaping.

**TensorBoard signals to watch for Phase 6:**
```
stagnation:          should trend toward 0 (robot escapes faster)
climb_progress:      should become non-zero on hard terrain rows
flat_orientation_l2: will go more negative (robot pitching more) — this is expected
track_lin_vel_xy_exp: should stay ≥ Phase 5 baseline (~1.90)
```

```bash
# Train Phase 6 (resume from Phase 5 checkpoint)
python scripts/train.py --task RexmiRl-Go2w-Velocity-Rough-v0 --headless --resume
```

---

## 12. Roadmap

### Phase 6 ✅ COMPLETE: Obstacle climbing
- **Done**: `climb_progress` (+1.5, terrain-blind), relaxed orientation/deviation penalties,
  tripled stagnation penalty weight; 1500 iters from Phase 5 checkpoint
- **Achieved**: stairs_down_15-23cm: 0.13–0.34 → **0.59–0.80** 🏆
- **Side-effect**: flat-terrain bouncing exploit caused -0.10 to -0.16 regression on
  slopes/rough → fixed in Phase 7

### Phase 7 ✅ COMPLETE: Bouncing fix — hybrid sensor-gated climb_progress
- **Problem**: `lin_vel_z_l2=-0.3` + terrain-blind `climb_progress=+1.5` made bouncing
  on flat terrain worth **+0.218/step**, causing uniform slope/rough regression
- **Fix**: `lin_vel_z_l2` → -1.5 (restored); `climb_progress` → hybrid median-floor gated
  (`base_weight=0.4` flat, `obstacle_weight=1.5` near obstacle); checkpoint `model_8996.pt`
- **Achieved**: slope/rough tracking recovered to Phase 5 levels; stair-climbing gains preserved

### Phase 8 (current): Steep slopes (23°–45°) — two-policy split

**Why a unified policy failed** (4600 iters, model crashed into these walls):

| Problem | Explanation |
|---------|-------------|
| Curriculum reset | Every `--load_run` restart reset curriculum to level 0; all 4600 iters stayed at `terrain_level ≈ 0.45` — never reached steep-slope rows |
| Reward conflict | Relaxing `flat_orientation_l2 = -0.3` opened a wheel-lift exploit on flat terrain (thigh-salute gait) |
| Anti-exploit overfit | Adding `hip_deviation=-0.5` + `thigh_deviation=-0.15` to block exploits also blocked legitimate lateral stepping (no vy tracking) |

**Solution: dedicated steep-slope policy** — new file `steep_slope_env_cfg.py`

Inherits the complete Phase 7 rough env (same robot, scanner, stagnation, climb_progress,
leg_deviation, curriculum).  Only three things change:

| Setting | Phase 7 rough env (frozen) | Steep-slope env (Phase 8) |
|---------|----------------------------|---------------------------|
| Terrain | 5 types: stairs/boxes/rough/slope (0°–23°) | **ONLY `HfPyramidSlopedTerrainCfg` (23°–45°)** |
| `flat_orientation_l2` | -0.8 | **-0.1** — body must tilt with 35°–45° slope |
| `bad_orientation` limit | 1.0 rad (57°) | **1.4 rad (80°)** — 35° headroom above 45° |

Curriculum rows 0→9 interpolate slope: **row 0 = 23°** (handoff from model_8996), **row 9 = 45°**.
The obs/action space is identical to model_8996 → weights load cleanly, no architecture change.

**Training commands:**

```bash
# First run — load model_8996 as starting point
conda activate env_isaacsim
python scripts/train.py --task RexmiRl-Go2w-Velocity-SteepSlope-v0 --headless \
    --load_run go2w_velocity_rough/2026-06-14_20-03-41 --checkpoint model_8996.pt

# Resume from the latest steep-slope checkpoint
python scripts/train.py --task RexmiRl-Go2w-Velocity-SteepSlope-v0 --headless --resume
```

**Visualise the steep-slope policy (50 robots on 23°–45° tiles):**

```bash
# Latest checkpoint (auto-picks most recent steep-slope run)
python scripts/play.py --task RexmiRl-Go2w-Velocity-SteepSlope-Play-v0

# Specific checkpoint
python scripts/play.py --task RexmiRl-Go2w-Velocity-SteepSlope-Play-v0 \
    --load_run go2w_velocity_steep_slope/<date>_<time> --checkpoint model_<iter>.pt
```

**TensorBoard signals to watch:**

```
terrain_levels:       should climb 0 → 4–5 within 1000 iters (23° → 32°)
flat_orientation_l2:  grows more negative (body tilting onto slopes) — expected
bad_orientation:      termination rate should drop vs rough env (1.4 rad limit)
climb_progress:       always in obstacle_weight=1.5 mode (any slope > 3° triggers it)
track_lin_vel_xy_exp: target ≥ Phase 7 baseline (~1.90+) at final curriculum rows
```

**Eval steep-slope policy:**

```bash
CKPT_STEEP=logs/rsl_rl/go2w_velocity_steep_slope/<date>_<time>/model_<iter>.pt
python scripts/eval.py --checkpoint $CKPT_STEEP --group steep_slope
```

### Phase 9 (planned): Energy efficiency and gait elegance
- Increase `dof_torques_l2` weight for leg joints to discourage unnecessary leg movement
  when wheels alone are sufficient (energy-aware gait)
- Add reward for "legs near default when speed is high" — sprint with locked legs,
  only activate leg repositioning for obstacle crossings
- Potential: asymmetric leg deviation penalty (cheap to extend calves, expensive to splay hips)

### Phase 10 (planned): Custom REXMI robot
- Swap `GO2W_CFG` for `REXMI_CFG` (custom wheel geometry, sensor suite)
- All env/reward configs should transfer directly (same 16-DOF structure assumed)
- Re-run full 41-variant eval sweep on the new geometry to identify new cliffs

### Phase 11 (planned): Lunar terrain
- Integrate lunar regolith deformation model from Project Chrono
- Train on simulated crater terrain with soft soil contact
- Height scanner may need to be replaced with depth camera for deformable surfaces

### Phase 12 (planned): Sim-to-real transfer
- Domain randomisation: friction (0.3–1.2), mass (±20%), motor damping (±30%)
- Deploy to real Go2W hardware using Isaac Lab's `--real_time` mode

---

*Last updated: 2026-06-19 · REXMI Project · Phase 8 in progress — two-policy split (rough frozen @ model_8996, steep-slope 23°–45° in training)*
