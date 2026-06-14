# REXMI RL

Reinforcement learning project for **REXMI** — a wheeled hybrid quadruped robot designed for lunar terrain traversal — built on [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab).

**📖 Full developer reference: [`docs/rl_setup.md`](docs/rl_setup.md)**

---

## Project Structure

```
rexmi_rl/
├── assets/
│   └── robots/
│       └── go2w/
│           ├── urdf/           # Unitree Go2W URDF + converted USD
│           │   └── go2w/       # go2w.usd + configuration/ sub-USDs
│           ├── meshes/         # DAE/STL mesh files
│           └── config/         # Joint name config from Unitree
├── source/
│   └── rexmi_rl/               # Isaac Lab extension package
│       ├── __init__.py         # Package root — triggers gym.register()
│       ├── assets/
│       │   └── go2w.py         # GO2W_CFG: ArticulationCfg (USD path, actuators)
│       └── tasks/
│           └── locomotion/
│               └── velocity/
│                   └── config/
│                       └── go2w/
│                           ├── __init__.py         # gym.register() calls
│                           ├── rough_env_cfg.py    # Full MDP definition
│                           ├── flat_env_cfg.py     # Flat terrain wrapper
│                           └── agents/
│                               └── rsl_rl_ppo_cfg.py  # PPO hyperparameters
├── scripts/
│   ├── train.py                # RL training entry point
│   └── play.py                 # Policy playback / visualisation
├── docs/
│   └── rl_setup.md             # Complete developer reference (read this!)
├── logs/                       # Training logs (gitignored)
├── setup.cfg                   # Python package metadata
├── run.sh                      # Convenience launcher
└── .env                        # Local config: set ISAACLAB_DIR (gitignored)
```

---

## Prerequisites

- [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab) installed at `$ISAACLAB_DIR` (default: `~/IsaacLab`)
- Isaac Sim 4.x (bundled with Isaac Lab)
- Python 3.10+, CUDA 11.8+, ~8 GB VRAM for full training

---

## Quick Start

### 1. Configure Isaac Lab path

```bash
echo "ISAACLAB_DIR=/home/susan/IsaacLab" > .env
chmod +x run.sh
```

### 2. Install the package

```bash
source /home/susan/IsaacLab/.venv/bin/activate
pip install -e .

# Verify
python -c "import rexmi_rl; print('OK')"
```

### 3. Train (Phase 1 — wheel locomotion, flat terrain)

```bash
# Full training — ~20-30 min on RTX 3080+
./run.sh scripts/train.py --task RexmiRl-Go2w-Velocity-Flat-v0 --headless

# Quick smoke test (128 envs)
./run.sh scripts/train.py --task RexmiRl-Go2w-Velocity-Flat-v0 --num_envs 128

# Monitor training
tensorboard --logdir logs/rsl_rl/go2w_velocity_flat
```

### 4. Play back the trained policy

```bash
./run.sh scripts/play.py --task RexmiRl-Go2w-Velocity-Flat-Play-v0
```

---

## Robot: Unitree Go2W

The Go2W is a wheeled hybrid quadruped with:
- **12 revolute joints**: 3 per leg (hip/thigh/calf)
- **4 continuous wheel joints**: one at each foot (FL/FR/RL/RR)
- Total controllable DOF: **16** (12 leg positions + 4 wheel velocities)

**Phase 1 strategy**: Legs are locked in default stance by a PD controller.
Only the 4 wheels are controlled by the RL policy (4-DOF velocity commands).
This is the simplest possible locomotion task and trains in ~300 iterations.

---

## Registered Environments

| ID | Description | Config |
|----|-------------|--------|
| `RexmiRl-Go2w-Velocity-Flat-v0` | Training: 4096 envs, flat terrain | `flat_env_cfg:Go2wFlatEnvCfg` |
| `RexmiRl-Go2w-Velocity-Flat-Play-v0` | Playback: 50 envs, no noise | `flat_env_cfg:Go2wFlatEnvCfg_PLAY` |

---

## Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| 1a | Go2W USD import + verification in Isaac Sim | ✅ Done |
| 1b | Wheel-only locomotion on flat terrain (this PR) | ✅ Done |
| 2a | Full 16-DOF control (legs + wheels) | ⏳ Planned |
| 2b | Rough terrain + height scan curriculum | ⏳ Planned |
| 3 | Custom REXMI robot (modified wheel geometry) | ⏳ Planned |
| 4 | Project Chrono lunar soil terrain integration | ⏳ Planned |
| 5 | Full REXMI lunar crater traversal showcase | ⏳ Planned |

---

## Documentation

See **[`docs/rl_setup.md`](docs/rl_setup.md)** for a complete, beginner-friendly
reference covering:

- What RL is doing and why we made each design choice
- Detailed MDP breakdown (observations, actions, rewards, terminations)
- PPO algorithm walkthrough with pseudocode
- Reward engineering troubleshooting guide
- Phase 2+ expansion instructions

---

## References

- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [Isaac Lab Locomotion Tutorial](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/run_locomotion_task.html)
- [Unitree Go2W URDF](https://github.com/unitreerobotics/unitree_ros/tree/master/robots/go2w_description)
- [RSL-RL (PPO library)](https://github.com/leggedrobotics/rsl_rl)
- [PPO Paper (Schulman et al. 2017)](https://arxiv.org/abs/1707.06347)
