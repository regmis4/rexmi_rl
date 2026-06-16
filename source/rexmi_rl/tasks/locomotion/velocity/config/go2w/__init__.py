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

  Lunar crater traversal demo (uses rough-terrain policy, 32 m × 32 m tiles)
  -----------------------------------------------------------------------
  These environments load the Phase 6-Optimized rough-terrain checkpoint onto
  LOLA-calibrated procedural crater wall terrains.  No re-training required.
  See source/rexmi_rl/tasks/locomotion/velocity/config/go2w/crater_env_cfg.py
  and docs/lunar_crater_terrain_research.md for full morphometry details.

  Upslope (floor → rim)  — default direction, yaw=0
  ---------------------------------------------------------
  RexmiRl-Go2w-Crater-Type1-Play-v0
      → 10 robots on Haworth-archetype wall (7.5–14°). ✅ All zones traversable.
  RexmiRl-Go2w-Crater-Type1-Record-v0
      → Single-robot upslope recording variant.
  RexmiRl-Go2w-Crater-Type2-Play-v0
      → 10 robots on Faustini-archetype wall (11.5–17.5°). ✅ PRIMARY DEMO.
  RexmiRl-Go2w-Crater-Type2-Record-v0
      → Single-robot upslope recording variant (use for investor video).
  RexmiRl-Go2w-Crater-Type3-Play-v0
      → 10 robots on Shackleton-archetype wall (22.5–35°). ⚠️ Phase 8 required.
  RexmiRl-Go2w-Crater-Type3-Record-v0
      → Single-robot upslope recording variant.

  Downslope (rim → floor)  — yaw=π, spawn near rim
  ---------------------------------------------------------
  RexmiRl-Go2w-Crater-Type1-Down-Play-v0
      → 10 robots descending Haworth wall from rim to floor.
  RexmiRl-Go2w-Crater-Type1-Down-Record-v0
      → Single-robot downslope recording.
  RexmiRl-Go2w-Crater-Type2-Down-Play-v0
      → 10 robots descending Faustini wall. ✅ PRIMARY DOWNSLOPE DEMO.
  RexmiRl-Go2w-Crater-Type2-Down-Record-v0
      → Single-robot downslope recording (investor video).
  RexmiRl-Go2w-Crater-Type3-Down-Play-v0
      → 10 robots descending Shackleton — hits 35° wall immediately.
  RexmiRl-Go2w-Crater-Type3-Down-Record-v0
      → Single-robot downslope recording.
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

# ---------------------------------------------------------------------------
# Lunar crater traversal demo environments (investor showcase)
# ---------------------------------------------------------------------------
# All crater envs use the Go2wRoughPPORunnerCfg because the obs/action space
# is identical to the rough-terrain training config (height scanner + 16 DOF).
# The policy checkpoint from go2w_velocity_rough is loaded unchanged.

# -- Type 1: Ancient/degraded crater (Haworth-archetype, 7.5–14°) -----------

gym.register(
    id="RexmiRl-Go2w-Crater-Type1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "rexmi_rl.tasks.locomotion.velocity.config.go2w.crater_env_cfg"
            ":LunarCraterType1EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Go2wRoughPPORunnerCfg"
        ),
    },
)

gym.register(
    id="RexmiRl-Go2w-Crater-Type1-Record-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "rexmi_rl.tasks.locomotion.velocity.config.go2w.crater_env_cfg"
            ":LunarCraterType1EnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Go2wRoughPPORunnerCfg"
        ),
    },
)

# -- Type 2: Faustini-archetype (11.5–17.5°) — PRIMARY DEMO -----------------

gym.register(
    id="RexmiRl-Go2w-Crater-Type2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "rexmi_rl.tasks.locomotion.velocity.config.go2w.crater_env_cfg"
            ":LunarCraterType2EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Go2wRoughPPORunnerCfg"
        ),
    },
)

gym.register(
    id="RexmiRl-Go2w-Crater-Type2-Record-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "rexmi_rl.tasks.locomotion.velocity.config.go2w.crater_env_cfg"
            ":LunarCraterType2EnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Go2wRoughPPORunnerCfg"
        ),
    },
)

# -- Type 3: Shackleton-archetype (22.5–35°) — Phase 8 target ---------------

gym.register(
    id="RexmiRl-Go2w-Crater-Type3-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "rexmi_rl.tasks.locomotion.velocity.config.go2w.crater_env_cfg"
            ":LunarCraterType3EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Go2wRoughPPORunnerCfg"
        ),
    },
)

gym.register(
    id="RexmiRl-Go2w-Crater-Type3-Record-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "rexmi_rl.tasks.locomotion.velocity.config.go2w.crater_env_cfg"
            ":LunarCraterType3EnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Go2wRoughPPORunnerCfg"
        ),
    },
)

# ---------------------------------------------------------------------------
# Downslope variants (rim → floor)
# ---------------------------------------------------------------------------

gym.register(
    id="RexmiRl-Go2w-Crater-Type1-Down-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "rexmi_rl.tasks.locomotion.velocity.config.go2w.crater_env_cfg"
            ":LunarCraterType1DownEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Go2wRoughPPORunnerCfg"
        ),
    },
)

gym.register(
    id="RexmiRl-Go2w-Crater-Type1-Down-Record-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "rexmi_rl.tasks.locomotion.velocity.config.go2w.crater_env_cfg"
            ":LunarCraterType1DownEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Go2wRoughPPORunnerCfg"
        ),
    },
)

gym.register(
    id="RexmiRl-Go2w-Crater-Type2-Down-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "rexmi_rl.tasks.locomotion.velocity.config.go2w.crater_env_cfg"
            ":LunarCraterType2DownEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Go2wRoughPPORunnerCfg"
        ),
    },
)

gym.register(
    id="RexmiRl-Go2w-Crater-Type2-Down-Record-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "rexmi_rl.tasks.locomotion.velocity.config.go2w.crater_env_cfg"
            ":LunarCraterType2DownEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Go2wRoughPPORunnerCfg"
        ),
    },
)

gym.register(
    id="RexmiRl-Go2w-Crater-Type3-Down-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "rexmi_rl.tasks.locomotion.velocity.config.go2w.crater_env_cfg"
            ":LunarCraterType3DownEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Go2wRoughPPORunnerCfg"
        ),
    },
)

gym.register(
    id="RexmiRl-Go2w-Crater-Type3-Down-Record-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "rexmi_rl.tasks.locomotion.velocity.config.go2w.crater_env_cfg"
            ":LunarCraterType3DownEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Go2wRoughPPORunnerCfg"
        ),
    },
)
