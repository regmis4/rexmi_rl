# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Gymnasium environment registration for Go2W velocity-tracking tasks.

What does gym.register() do?
------------------------------
It tells the Gymnasium (OpenAI Gym) registry about an environment ID string and
how to instantiate it.  Once registered, any code can create the env with:
    gym.make("RexmiRl-Go2w-Velocity-Flat-v0")

Isaac Lab's training scripts use the kwargs pattern:
  env_cfg_entry_point    — Python dotted path to the env config class
  rsl_rl_cfg_entry_point — Python dotted path to the PPO runner config class

The entry_point for the env itself is always "isaaclab.envs:ManagerBasedRLEnv"
because Isaac Lab's ManagerBasedRLEnv is the universal environment class that
reads whichever EnvCfg we give it and builds the scene/MDP accordingly.

Registered environments
-----------------------
  Flat terrain (no height scanner, ~48-dim obs, [128,128,128] network)
  -----------------------------------------------------------------------
  RexmiRl-Go2w-Velocity-Flat-v0
      → Training env: 4096 parallel environments, flat terrain, noise enabled.
      → Train with: python scripts/train.py --task RexmiRl-Go2w-Velocity-Flat-v0

  RexmiRl-Go2w-Velocity-Flat-Play-v0
      → Visualisation: 50 environments, no sensor noise, no random forces.
      → Play with: python scripts/play.py --task RexmiRl-Go2w-Velocity-Flat-Play-v0

  Rough terrain (height scanner, ~208-dim obs, [512,256,128] network)
  -----------------------------------------------------------------------
  RexmiRl-Go2w-Velocity-Rough-v0
      → Training env: 4096 envs, procedural terrain (stairs/slopes/boxes/rough).
      → Train from scratch — obs space differs from flat policy.
      → Train with: python scripts/train.py --task RexmiRl-Go2w-Velocity-Rough-v0

  RexmiRl-Go2w-Velocity-Rough-Play-v0
      → Visualisation: 50 environments, no noise, no pushes.
      → Play with: python scripts/play.py --task RexmiRl-Go2w-Velocity-Rough-Play-v0
"""

import gymnasium as gym

# Import agents sub-package so it appears in the local namespace for entry point strings.
from rexmi_rl.tasks.locomotion.velocity.config.go2w import agents

##
# Register Gym environments
##

# ---------------------------------------------------------------------------
# Flat terrain
# ---------------------------------------------------------------------------

gym.register(
    id="RexmiRl-Go2w-Velocity-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "rexmi_rl.tasks.locomotion.velocity.config.go2w.flat_env_cfg:Go2wFlatEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Go2wFlatPPORunnerCfg"
        ),
    },
)

gym.register(
    id="RexmiRl-Go2w-Velocity-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "rexmi_rl.tasks.locomotion.velocity.config.go2w.flat_env_cfg:Go2wFlatEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Go2wFlatPPORunnerCfg"
        ),
    },
)

# ---------------------------------------------------------------------------
# Rough terrain (Phase 4 — full height scanner + terrain curriculum)
# ---------------------------------------------------------------------------

gym.register(
    id="RexmiRl-Go2w-Velocity-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "rexmi_rl.tasks.locomotion.velocity.config.go2w.flat_env_cfg:Go2wRoughEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Go2wRoughPPORunnerCfg"
        ),
    },
)

gym.register(
    id="RexmiRl-Go2w-Velocity-Rough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "rexmi_rl.tasks.locomotion.velocity.config.go2w.flat_env_cfg:Go2wRoughEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Go2wRoughPPORunnerCfg"
        ),
    },
)
