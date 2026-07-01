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

  Lunar crater traversal demo (crater_env_cfg.py, 32–64 m tiles)
  -----------------------------------------------------------------------
  Type3: Shackleton-archetype cross-section (22.5–35°) — Phase 8 policy ✅
  Bowl:  Full radial crater (22 m dia) with geology — HEADLINE DEMO
         Robot traverses exterior → rim → floor → rim → exterior in one pass.
         64 m × 64 m tile; mountain backdrop (27°, 20 boulders) NE of crater.

  RexmiRl-Go2w-Crater-Type3-Play-v0    → 10 robots, Shackleton wall, steep policy
  RexmiRl-Go2w-Crater-Type3-Record-v0  → single-robot recording
  RexmiRl-Go2w-Crater-Type3-Down-Play-v0   → 10 robots descending, steep policy
  RexmiRl-Go2w-Crater-Type3-Down-Record-v0 → single-robot downslope recording

  Bowl (rough/steep obs — height scanner):
  RexmiRl-Go2w-Crater-Bowl-Play-v0         → 10 robots, Go2wRoughPPORunnerCfg
  RexmiRl-Go2w-Crater-Bowl-Record-v0       → single-robot, Go2wRoughPPORunnerCfg
  RexmiRl-Go2w-Crater-Bowl-SteepSlope-Play-v0   → same env, steep-slope runner
  RexmiRl-Go2w-Crater-Bowl-SteepSlope-Record-v0

  Bowl (flat obs — no height scanner):
  RexmiRl-Go2w-Crater-Bowl-Flat-Play-v0         → flat runner
  RexmiRl-Go2w-Crater-Bowl-Flat-Record-v0
  RexmiRl-Go2w-Crater-Bowl-FastFlat-Play-v0     → fast-flat runner
  RexmiRl-Go2w-Crater-Bowl-FastFlat-Record-v0
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
# Fast flat terrain — high-speed forward locomotion (up to 2 m/s)
# Wheel action scale=40 rad/s; forward command range (-0.5, 2.0) m/s.
# Train from scratch — incompatible with standard flat checkpoints.
# ---------------------------------------------------------------------------

gym.register(
    id="RexmiRl-Go2w-Velocity-FastFlat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "rexmi_rl.tasks.locomotion.velocity.config.go2w.fast_flat_env_cfg:Go2wFastFlatEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Go2wFastFlatPPORunnerCfg"
        ),
    },
)

gym.register(
    id="RexmiRl-Go2w-Velocity-FastFlat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "rexmi_rl.tasks.locomotion.velocity.config.go2w.fast_flat_env_cfg:Go2wFastFlatEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Go2wFastFlatPPORunnerCfg"
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
# Steep-slope terrain (Phase 8 — dedicated 23°–45° slope policy)
# ---------------------------------------------------------------------------
# Separate from the rough-terrain policy (model_8996) which handles stairs,
# boxes, rough, and moderate slopes (0°–23°).  The steep-slope policy trains
# exclusively on 23°–45° slopes and is loaded from model_8996 weights.
#
# Train:
#   python scripts/train.py --task RexmiRl-Go2w-Velocity-SteepSlope-v0 --headless \
#       --load_run go2w_velocity_rough/2026-06-14_20-03-41 --checkpoint model_8996.pt
# Resume:
#   python scripts/train.py --task RexmiRl-Go2w-Velocity-SteepSlope-v0 --headless --resume
# Play:
#   python scripts/play.py --task RexmiRl-Go2w-Velocity-SteepSlope-Play-v0
# Logs: logs/rsl_rl/go2w_velocity_steep_slope/
# FROZEN: model_5998.pt (2026-06-20_15-37-32) — do not retrain this experiment

gym.register(
    id="RexmiRl-Go2w-Velocity-SteepSlope-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "rexmi_rl.tasks.locomotion.velocity.config.go2w"
            ".steep_slope_env_cfg:Go2wSteepSlopeEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Go2wSteepSlopePPORunnerCfg"
        ),
    },
)

gym.register(
    id="RexmiRl-Go2w-Velocity-SteepSlope-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "rexmi_rl.tasks.locomotion.velocity.config.go2w"
            ".steep_slope_env_cfg:Go2wSteepSlopeEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Go2wSteepSlopePPORunnerCfg"
        ),
    },
)

# ---------------------------------------------------------------------------
# Lunar crater traversal demo environments (investor showcase)
# ---------------------------------------------------------------------------
# All crater envs use the Go2wRoughPPORunnerCfg because the obs/action space
# is identical to the rough-terrain training config (height scanner + 16 DOF).
# The policy checkpoint from go2w_velocity_rough is loaded unchanged.

# -- Type 3: Shackleton-archetype cross-section (22.5–35°) — Phase 8 ✅ ----

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
# Downslope variants (rim → floor) — Type 3 only
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Bowl: full radial crater (64 m tile, 22 m diameter, mountain NE backdrop)
# ---------------------------------------------------------------------------
# Robot spawns OUTSIDE, drives THROUGH the bowl (exterior→rim→floor→rim→exterior).
# 64 m × 64 m tile; crater 22 m dia + exterior boulders + 27° mountain (NE corner).
#
# Policy selection guide:
#   rough obs (height scanner, ~208-dim):   use Rough or SteepSlope runner
#   flat obs  (no height scanner, ~48-dim): use Flat or FastFlat runner
#
#   --task Crater-Bowl-Play-v0             --load_run <rough_run>
#   --task Crater-Bowl-SteepSlope-Play-v0  --load_run <steep_slope_run>
#   --task Crater-Bowl-Flat-Play-v0        --load_run <flat_run>
#   --task Crater-Bowl-FastFlat-Play-v0    --load_run <fastflat_run>

# Rough obs (height scanner) — rough & steep-slope policies
gym.register(
    id="RexmiRl-Go2w-Crater-Bowl-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "rexmi_rl.tasks.locomotion.velocity.config.go2w.crater_env_cfg"
            ":LunarCraterDemoBowlEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Go2wRoughPPORunnerCfg"
        ),
    },
)

gym.register(
    id="RexmiRl-Go2w-Crater-Bowl-Record-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "rexmi_rl.tasks.locomotion.velocity.config.go2w.crater_env_cfg"
            ":LunarCraterDemoBowlEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Go2wRoughPPORunnerCfg"
        ),
    },
)

gym.register(
    id="RexmiRl-Go2w-Crater-Bowl-SteepSlope-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "rexmi_rl.tasks.locomotion.velocity.config.go2w.crater_env_cfg"
            ":LunarCraterDemoBowlEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Go2wSteepSlopePPORunnerCfg"
        ),
    },
)

gym.register(
    id="RexmiRl-Go2w-Crater-Bowl-SteepSlope-Record-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "rexmi_rl.tasks.locomotion.velocity.config.go2w.crater_env_cfg"
            ":LunarCraterDemoBowlEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Go2wSteepSlopePPORunnerCfg"
        ),
    },
)

# Flat obs (no height scanner) — flat & fast-flat policies
gym.register(
    id="RexmiRl-Go2w-Crater-Bowl-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "rexmi_rl.tasks.locomotion.velocity.config.go2w.crater_env_cfg"
            ":LunarCraterDemoBowlFlatEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Go2wFlatPPORunnerCfg"
        ),
    },
)

gym.register(
    id="RexmiRl-Go2w-Crater-Bowl-Flat-Record-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "rexmi_rl.tasks.locomotion.velocity.config.go2w.crater_env_cfg"
            ":LunarCraterDemoBowlFlatEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Go2wFlatPPORunnerCfg"
        ),
    },
)

gym.register(
    id="RexmiRl-Go2w-Crater-Bowl-FastFlat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "rexmi_rl.tasks.locomotion.velocity.config.go2w.crater_env_cfg"
            ":LunarCraterDemoBowlFastFlatEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Go2wFastFlatPPORunnerCfg"
        ),
    },
)

gym.register(
    id="RexmiRl-Go2w-Crater-Bowl-FastFlat-Record-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "rexmi_rl.tasks.locomotion.velocity.config.go2w.crater_env_cfg"
            ":LunarCraterDemoBowlFastFlatEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Go2wFastFlatPPORunnerCfg"
        ),
    },
)

# ---------------------------------------------------------------------------
# Rocky slope terrain (Phase 8b — steep slopes WITH boulders, 15°–35°)
# ---------------------------------------------------------------------------
# Addresses the crater bowl failure mode: robot falls when a boulder snags
# its leg on a steep slope.  This policy is trained on pyramid slopes that
# have difficulty-scaled Gaussian boulder bumps and surface roughness added.
#
# Train (load from frozen steep-slope checkpoint):
#   python scripts/train.py --task RexmiRl-Go2w-Velocity-RockySlope-v0 --headless \
#       --load_run go2w_velocity_steep_slope/2026-06-20_15-37-32 --checkpoint model_5998.pt
# Resume:
#   python scripts/train.py --task RexmiRl-Go2w-Velocity-RockySlope-v0 --headless --resume
# Play training terrain:
#   python scripts/play.py --task RexmiRl-Go2w-Velocity-RockySlope-Play-v0 \
#       --load_run go2w_velocity_rocky_slope/<run_date>
# Demo on crater bowl:
#   python scripts/play.py --task RexmiRl-Go2w-Crater-Bowl-RockySlope-Play-v0 \
#       --load_run go2w_velocity_rocky_slope/<run_date>
# Logs: logs/rsl_rl/go2w_velocity_rocky_slope/

gym.register(
    id="RexmiRl-Go2w-Velocity-RockySlope-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "rexmi_rl.tasks.locomotion.velocity.config.go2w"
            ".rocky_slope_env_cfg:Go2wRockySlopeEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Go2wRockySlopePPORunnerCfg"
        ),
    },
)

gym.register(
    id="RexmiRl-Go2w-Velocity-RockySlope-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "rexmi_rl.tasks.locomotion.velocity.config.go2w"
            ".rocky_slope_env_cfg:Go2wRockySlopeEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Go2wRockySlopePPORunnerCfg"
        ),
    },
)

# Crater bowl with rocky-slope policy (PRIMARY DEMO after Phase 8b training)
gym.register(
    id="RexmiRl-Go2w-Crater-Bowl-RockySlope-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "rexmi_rl.tasks.locomotion.velocity.config.go2w.crater_env_cfg"
            ":LunarCraterDemoBowlEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Go2wRockySlopePPORunnerCfg"
        ),
    },
)

gym.register(
    id="RexmiRl-Go2w-Crater-Bowl-RockySlope-Record-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "rexmi_rl.tasks.locomotion.velocity.config.go2w.crater_env_cfg"
            ":LunarCraterDemoBowlEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Go2wRockySlopePPORunnerCfg"
        ),
    },
)
