# REXMI RL — Complete Developer Reference

> **Phase 1: Unitree Go2W · Flat Terrain · Wheel-Only Locomotion**

This document explains **every design decision** made in the Phase 1 RL setup.
The goal is that after reading this you understand the full pipeline from
the physical robot model to a trained policy, with zero assumed prior knowledge.

---

## Table of Contents

1. [Big Picture — What Are We Actually Doing?](#1-big-picture)
2. [The Robot: Unitree Go2W](#2-the-robot-unitree-go2w)
3. [What Is an MDP?](#3-what-is-an-mdp)
4. [Isaac Lab Architecture](#4-isaac-lab-architecture)
5. [File-by-File Walkthrough](#5-file-by-file-walkthrough)
6. [The MDP in Detail](#6-the-mdp-in-detail)
7. [PPO Algorithm Explained](#7-ppo-algorithm-explained)
8. [Training Workflow](#8-training-workflow)
9. [Reward Engineering Guide](#9-reward-engineering-guide)
10. [Phase 2 Roadmap](#10-phase-2-roadmap)

---

## 1. Big Picture — What Are We Actually Doing?

We are teaching a simulated robot to **drive in any commanded direction**
using **Reinforcement Learning (RL)**.

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
- **Observations**: What the robot "senses" (velocity, joint angles, commanded direction)
- **Actions**: Wheel velocity targets (how fast each wheel should spin)
- **Reward**: A scalar signal that tells the policy how well it did
- **Environment**: Isaac Sim running the physics of 4096 robot copies simultaneously

After ~300 updates the policy has seen ~29 million robot-steps of experience
and has learned a controller good enough to track velocity commands on flat terrain.

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

| Group | Joints | Type | Count |
|-------|--------|------|-------|
| Hips | `FL/FR/RL/RR_hip_joint` | revolute | 4 |
| Thighs | `FL/FR/RL/RR_thigh_joint` | revolute | 4 |
| Calfs | `FL/FR/RL/RR_calf_joint` | revolute | 4 |
| Wheels | `FL/FR/RL/RR_foot_joint` | continuous | 4 |

### Phase 1 control strategy

In Phase 1 the **legs are locked** in their default stance by a stiff PD controller.
The RL policy controls only the **4 wheel joints** via velocity targets.

This simplifies the problem enormously:
- 4 action dimensions instead of 16
- No gait patterns to learn
- Robot behaves like a wheeled vehicle with adjustable wheel heights

---

## 3. What Is an MDP?

An MDP (Markov Decision Process) is the formal mathematical framework for RL.
It has 5 components: **(S, A, T, R, γ)**

| Symbol | Name | In our context |
|--------|------|----------------|
| **S** | State space | Robot position, velocity, joint angles, IMU readings |
| **A** | Action space | Wheel velocity targets (4 floats ∈ [-1, 1] × scale) |
| **T** | Transition | PhysX physics simulation |
| **R** | Reward function | Velocity tracking + stability penalties |
| **γ** | Discount factor | 0.99 (future rewards count almost as much as immediate) |

### Markov property

"The next state depends only on the current state and action, not on history."

In our case this is approximately true — the robot's next position depends on
where it is now and what the wheels do, not on what happened 10 steps ago.
Small violations (e.g., joint velocity effects) are handled by including
joint velocities in the observation.

### Episode structure

```
t=0: Robot spawned at random XY position and orientation
     Velocity command sampled uniformly: vx ∈ [-1,1], vy ∈ [-1,1], ωz ∈ [-1,1]
t=1..N: Policy observes state, outputs wheel velocities, physics steps forward
        Reward computed each step
t=T: Episode ends when:
     (a) base touches ground (fell over) → terminated
     (b) 20 seconds elapsed (4000 steps at 50Hz) → timeout
     Then robot is reset to a new random pose
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
- No subclassing needed — you just configure:
  - `SceneCfg` → what goes in the world
  - `ObservationsCfg` → what the policy sees
  - `ActionsCfg` → what the policy controls
  - `RewardsCfg` → reward terms and weights
  - `TerminationsCfg` → episode end conditions
  - `EventsCfg` → randomisation

**ArticulationCfg** (how robots are described):
- Points to a USD file containing the robot geometry + physics
- Specifies actuator models (motor physics)
- Specifies initial state (spawn height, joint angles)

### Data flow each physics step

```
1. EventManager  → apply randomisation if interval triggered
2. ObservationManager → collect sensor readings into obs tensor [N_envs, obs_dim]
3. ActionManager → decode policy output into joint commands [N_envs, action_dim]
4. PhysX step    → simulate physics for dt=0.005s × decimation=4 = 0.02s
5. RewardManager → compute reward [N_envs, 1]
6. TerminationManager → check done flags [N_envs, 1]
7. If done: reset that env's robot to new random state
```

---

## 5. File-by-File Walkthrough

### Package install: `setup.cfg`

```
setup.cfg
```

Tells pip how to install `rexmi_rl` as a Python package.
Critical for Isaac Lab to discover our environments via `import rexmi_rl`.

**Command:** `pip install -e .` (from repo root)
The `-e` flag = "editable" — edits to source files take effect without re-install.

---

### Robot asset: `source/rexmi_rl/assets/go2w.py`

```
GO2W_CFG = ArticulationCfg(
    spawn  → UsdFileCfg(usd_path=".../go2w.usd", activate_contact_sensors=True, ...)
    init_state → InitialStateCfg(pos=(0,0,0.43), joint_pos={...})
    actuators → {
        "leg_joints":   ImplicitActuatorCfg(stiffness=25, damping=0.5)   # PD hold
        "wheel_joints": ImplicitActuatorCfg(stiffness=0,  damping=5.0)   # vel mode
    }
)
```

**USD path resolution**: The path is built relative to `__file__` so it works
regardless of where the repo is cloned.

**Actuator models**:
- `ImplicitActuatorCfg` lets PhysX handle the motor physics internally.
- `stiffness = Kp` (position gain), `damping = Kd` (velocity gain)
- For wheels: `stiffness=0` disables position control entirely.
  The motor only applies torque proportional to velocity error:
  ```
  τ = Kd × (ω_desired - ω_actual)
  ```

---

### Base environment: `source/rexmi_rl/tasks/locomotion/velocity/config/go2w/rough_env_cfg.py`

The core file — defines everything about the MDP.

```
Go2wFlatEnvCfg(LocomotionVelocityRoughEnvCfg)
    __post_init__:
        scene.robot = GO2W_CFG          # set our robot
        terrain_type = "plane"          # flat ground
        height_scanner = None           # not needed
        actions.joint_pos = JointVelocityActionCfg([".*_foot_joint"], scale=10)
        rewards.track_lin_vel_xy_exp.weight = 1.5
        rewards.flat_orientation_l2.weight = -2.5
        ... (all reward/event/termination tuning)
```

**Why inherit from `LocomotionVelocityRoughEnvCfg`?**
That base class (from Isaac Lab) already wires up:
- The scene template with terrain, robot placeholder, sensors, lighting
- All MDP manager instances
- Simulation timing (dt, decimation)
- Default reward terms (tracking, energy, air time)

We only override what's different for the Go2W.

---

### Flat env wrapper: `flat_env_cfg.py`

A thin re-export. In Phase 1 it just re-exports `Go2wFlatEnvCfg` from
`rough_env_cfg.py`. In Phase 2, this file will become the "flat baseline"
and `rough_env_cfg.py` will add terrain complexity.

---

### PPO config: `agents/rsl_rl_ppo_cfg.py`

```
Go2wFlatPPORunnerCfg:
    num_steps_per_env = 24        # steps per rollout per env
    max_iterations = 300          # total updates
    experiment_name = "go2w_velocity_flat"
    policy:
        actor_hidden_dims = [128, 128, 128]
        critic_hidden_dims = [128, 128, 128]
        activation = "elu"
    algorithm:
        clip_param = 0.2
        learning_rate = 1e-3
        gamma = 0.99
        lam = 0.95
```

---

### Gym registration: `config/go2w/__init__.py`

```python
gym.register(
    id="RexmiRl-Go2w-Velocity-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "...flat_env_cfg:Go2wFlatEnvCfg",
        "rsl_rl_cfg_entry_point": "...rsl_rl_ppo_cfg:Go2wFlatPPORunnerCfg",
    }
)
```

This is called once when `import rexmi_rl` runs (triggered by `train.py`).

---

### Scripts: `scripts/train.py` and `scripts/play.py`

Both scripts follow the same pattern:
1. Set up `sys.path` so `import rexmi_rl` works
2. `import rexmi_rl` → triggers `gym.register()`
3. `runpy.run_path(IsaacLab_script)` → delegates to Isaac Lab's trainer/player

This pattern avoids duplicating Isaac Lab's complex argument parsing logic.

---

## 6. The MDP in Detail

### Observations (what the policy "sees")

The observation vector is concatenated in this order:

| Term | Dim | Description |
|------|-----|-------------|
| `base_lin_vel` | 3 | Linear velocity of base (x,y,z) in body frame [m/s] |
| `base_ang_vel` | 3 | Angular velocity of base (roll,pitch,yaw) [rad/s] |
| `projected_gravity` | 3 | Gravity vector projected into body frame — encodes tilt |
| `velocity_commands` | 3 | Commanded (vx, vy, ωz) — what we're being asked to do |
| `joint_pos` | 16 | All joint positions relative to default [rad] |
| `joint_vel` | 16 | All joint velocities [rad/s] |
| `actions` | 4 | Previous wheel velocity actions (history) |
| **Total** | **48** | |

**Gaussian noise** is added to obs during training (not during play) to
make the policy robust to sensor imperfections:
- `base_lin_vel`: ±0.1 m/s
- `base_ang_vel`: ±0.2 rad/s
- `projected_gravity`: ±0.05

### Actions (what the policy controls)

```
output ∈ [-1, 1]⁴  (one per wheel: FL, FR, RL, RR)
target_velocity = output × scale = output × 10 rad/s
```

The four wheels can be set independently, allowing:
- **Forward**: all 4 wheels at +v
- **Backward**: all 4 wheels at -v
- **Left turn**: left wheels at -v, right at +v (differential steering)
- **Right turn**: left wheels at +v, right at -v
- **Any combination** for diagonal motion

### Reward function

The total reward at each step is the **sum of all active terms**:

| Term | Weight | Formula | Purpose |
|------|--------|---------|---------|
| `track_lin_vel_xy_exp` | +1.5 | exp(-‖vxy - cmd‖² / 0.25) | Match forward/lateral speed |
| `track_ang_vel_z_exp` | +0.75 | exp(-(ωz - ωz_cmd)² / 0.25) | Match turning rate |
| `flat_orientation_l2` | -2.5 | ‖gravity_projected_xy‖² | Keep base level |
| `lin_vel_z_l2` | -2.0 | vz² | No bouncing |
| `ang_vel_xy_l2` | -0.05 | (ωx² + ωy²) | No roll/pitch wobble |
| `dof_torques_l2` | -1e-5 | ‖τ‖² | Minimise energy |
| `dof_acc_l2` | -2.5e-7 | ‖q̈‖² | Smooth accelerations |
| `action_rate_l2` | -0.01 | ‖a_t - a_{t-1}‖² | Smooth wheel commands |

**Why exponential tracking rewards?**
`exp(-error²/σ²)` gives:
- Reward = 1.0 when perfectly on target
- Reward ≈ 0 when far from target
- Smooth gradient everywhere (unlike piecewise rewards)

The `std` parameter σ=√0.25=0.5 means the reward drops to e⁻¹≈0.37 when
the velocity error is 0.5 m/s.

### Terminations

| Condition | Type |
|-----------|------|
| Base link contacts ground (`threshold > 1 N`) | Episode end (failure) |
| 20 seconds elapsed (4000 steps) | Timeout (not counted as failure) |

The distinction matters for GAE advantage estimation:
- Terminated episodes have value = 0 at the last step (the robot failed)
- Timed-out episodes use the value function's estimate of future returns

### Commands

Velocity commands are sampled uniformly each episode:
```
vx  ∈ [-1.0, 1.0] m/s    (forward/backward)
vy  ∈ [-1.0, 1.0] m/s    (lateral)
ωz  ∈ [-1.0, 1.0] rad/s  (turning)
```

`rel_standing_envs=0.02` means 2% of environments get a zero command —
this teaches the robot to actively hold still rather than drift.

`heading_command=True` means the yaw command is specified as a **heading direction**
(where the robot should face) and converted to ωz internally. This produces
more natural turning behaviour than raw ωz targets.

### Domain randomisation (Events)

These perturbations are applied at episode reset to prevent overfitting to
a single simulated scenario:

| Event | When | Effect |
|-------|------|--------|
| `randomize_rigid_body_material` | startup | Random friction on all links |
| `randomize_rigid_body_mass` | startup | ±1 to +3 kg added to base |
| `reset_root_state_uniform` | reset | Random XY position + yaw orientation |
| `reset_joints_by_scale` | reset | Joints reset to exactly default (scale=1.0) |

Friction randomisation is especially important — it forces the policy to
work across a range of wheel-ground friction coefficients, which will help
transfer to the real robot.

---

## 7. PPO Algorithm Explained

PPO (Proximal Policy Optimisation, Schulman et al. 2017) is the standard
algorithm for locomotion RL in Isaac Lab.

### The training loop

```
for iteration in range(300):
    # ── Rollout phase ────────────────────────────────────────────
    for step in range(24):  # 24 steps × 4096 envs = 98,304 transitions
        obs = env.get_observations()          # [4096, 48]
        actions, log_probs, values = policy(obs)
        obs_next, rewards, dones = env.step(actions)
        store (obs, actions, log_probs, values, rewards, dones)

    # ── Compute returns and advantages ───────────────────────────
    advantages = GAE(rewards, values, dones, γ=0.99, λ=0.95)
    returns = advantages + values

    # ── Update phase (5 epochs, 4 mini-batches) ──────────────────
    for epoch in range(5):
        for mini_batch in split(data, 4):
            new_log_probs, new_values, entropy = policy(mini_batch.obs)

            # Policy loss (clipped ratio)
            ratio = exp(new_log_probs - old_log_probs)
            surr1 = ratio × advantages
            surr2 = clip(ratio, 1-0.2, 1+0.2) × advantages
            policy_loss = -mean(min(surr1, surr2))

            # Value loss
            value_loss = MSE(new_values, returns)

            # Total loss
            loss = policy_loss + 1.0 × value_loss - 0.01 × entropy
            loss.backward()
            clip_grad_norm(params, 1.0)
            optimizer.step()

    log_metrics()
    if iteration % 50 == 0:
        save_checkpoint()
```

### Key PPO concepts

**Ratio clipping** — the "proximal" part:
If the new policy's probability differs too much from the old policy's
(ratio > 1+ε or < 1-ε), the gradient is clipped. This prevents
catastrophically large updates that could ruin a good policy.

**GAE (Generalised Advantage Estimation)**:
```
A_t = δ_t + γλ·δ_{t+1} + (γλ)²·δ_{t+2} + ...
where δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
```
λ=0.95 gives a good bias-variance tradeoff for locomotion.

**Adaptive learning rate**:
If `KL(old_policy, new_policy) > 2 × desired_kl=0.01`, halve the LR.
If `KL < 0.5 × desired_kl`, double the LR.
This keeps the policy updates in a safe regime automatically.

### What the metrics mean

| Metric | Good sign | Bad sign |
|--------|-----------|----------|
| `mean_reward` | Increasing | Flat or decreasing after 50 iters |
| `value_loss` | Decreasing and stable | Exploding |
| `policy_loss` | Oscillating slightly | Monotone (entropy collapsed) |
| `mean_episode_length` | Increasing toward 4000 | Stuck at < 100 (falling constantly) |
| `entropy` | Slowly decreasing | Collapses to near 0 too fast |

---

## 8. Training Workflow

### Step 1: Install the package

```bash
# Activate Isaac Lab's virtual environment
source ~/IsaacLab/.venv/bin/activate

# Install rexmi_rl in editable mode
cd ~/rexmi_rl
pip install -e .

# Verify it's installed
python -c "import rexmi_rl; print('OK')"
```

### Step 2: Set the Isaac Lab path

```bash
# Option A: environment variable
export ISAACLAB_DIR=~/IsaacLab

# Option B: create .env file (used by run.sh)
echo "ISAACLAB_DIR=/home/susan/IsaacLab" > .env
```

### Step 3: Train

```bash
# Full training (4096 envs, ~20-30 min on RTX 3080+)
python scripts/train.py --task RexmiRl-Go2w-Velocity-Flat-v0 --headless

# Quick test (128 envs, check it runs without errors)
python scripts/train.py --task RexmiRl-Go2w-Velocity-Flat-v0 --num_envs 128

# Watch TensorBoard
tensorboard --logdir logs/rsl_rl/go2w_velocity_flat
```

### Step 4: Play

```bash
# Visualise the trained policy
python scripts/play.py --task RexmiRl-Go2w-Velocity-Flat-Play-v0
```

### GPU memory requirements

| Num envs | Approx VRAM |
|----------|-------------|
| 128 | ~4 GB |
| 1024 | ~6 GB |
| 4096 | ~10 GB |

---

## 9. Reward Engineering Guide

Tuning rewards is the most important (and most art-like) aspect of locomotion RL.
Here are the key levers and what to change if training misbehaves:

### Robot spins in circles instead of going forward
→ Reduce `track_ang_vel_z_exp.weight` or increase `track_lin_vel_xy_exp.weight`

### Robot tips over immediately
→ Increase `flat_orientation_l2.weight` (make more negative)
→ Check that `base_contact` termination body_names="base" is correct

### Robot learns but moves jerkily
→ Increase `action_rate_l2.weight` (make more negative)
→ Try `num_steps_per_env=48` (longer rollouts = smoother advantage estimates)

### Training plateaus early
→ Increase `entropy_coef` (more exploration)
→ Increase `max_iterations`
→ Check reward scale — total per-step reward should be ~O(1), not O(0.001)

### Wheels spin full speed regardless of command
→ Decrease `scale` in `JointVelocityActionCfg` (reduces max speed)
→ Increase `dof_torques_l2.weight` penalty

---

## 10. Phase 2 Roadmap

Once Phase 1 training converges (mean reward stable, good velocity tracking),
the next steps are:

### Phase 2a: Full 16-DOF control
- Add a second action head for leg joints in `rough_env_cfg.py`
- Reduce leg joint stiffness from 25 → 5 (allow leg movement)
- Add `feet_air_time` reward back (encourage lifting legs over obstacles)
- Increase network to `[512, 256, 128]`

### Phase 2b: Rough terrain curriculum
- Change terrain back to `ROUGH_TERRAINS_CFG`
- Re-enable height scanner observation (adds ~160 dims)
- Enable `terrain_levels` curriculum (starts on flat, progresses to rough)
- Train for 1500 iterations

### Phase 2c: Stair/slope climbing
- Add custom terrain with stairs and slopes
- Add reward for height gain (encourage climbing)
- Use Phase 2b checkpoint as starting point (transfer learning)

### Phase 3: Custom REXMI robot
- Swap `GO2W_CFG` for `REXMI_CFG` (custom robot with modified wheel geometry)
- All env/reward configs should transfer directly

### Phase 4: Lunar terrain (Project Chrono)
- Integrate lunar regolith deformation from Project Chrono
- Train on simulated crater terrain

---

*Generated: 2026 · REXMI Project*
