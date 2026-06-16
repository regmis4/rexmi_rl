# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Environment configuration entry-point for the Go2W velocity tasks.

Why a separate file?
---------------------
Isaac Lab's convention separates "rough" (base) and "flat" configs so that:
  1. rough_env_cfg.py  defines all robot-specific overrides (joints, rewards, etc.)
                       and hosts ALL env config classes (Flat, Rough and their PLAY
                       variants).
  2. flat_env_cfg.py   is a thin re-export shim so __init__.py can reference a
                       single stable module path for gym.register entry points.

Registered environments (from __init__.py)
------------------------------------------
  RexmiRl-Go2w-Velocity-Flat-v0        — training, flat terrain (4096 envs)
  RexmiRl-Go2w-Velocity-Flat-Play-v0   — visualisation, flat (50 envs, no noise)
  RexmiRl-Go2w-Velocity-Rough-v0       — training, rough terrain + height scan
  RexmiRl-Go2w-Velocity-Rough-Play-v0  — visualisation, rough (50 envs, no noise)

  Lunar crater demo (LOLA-calibrated procedural crater wall terrains)
  ------------------------------------------------------------------
  RexmiRl-Go2w-Crater-Type1-Play-v0    — Haworth-archetype (7.5–14°), 10 envs
  RexmiRl-Go2w-Crater-Type1-Record-v0  — Haworth single-robot recording
  RexmiRl-Go2w-Crater-Type2-Play-v0         — Faustini upslope (11.5–17.5°), PRIMARY
  RexmiRl-Go2w-Crater-Type2-Record-v0       — Faustini upslope single-robot
  RexmiRl-Go2w-Crater-Type2-Down-Play-v0    — Faustini downslope (rim→floor)
  RexmiRl-Go2w-Crater-Type2-Down-Record-v0  — Faustini downslope single-robot
  RexmiRl-Go2w-Crater-Type3-Play-v0         — Shackleton upslope (22.5–35°), Phase 8
  RexmiRl-Go2w-Crater-Type3-Record-v0       — Shackleton upslope single-robot
  RexmiRl-Go2w-Crater-Type3-Down-Play-v0    — Shackleton downslope
  RexmiRl-Go2w-Crater-Type3-Down-Record-v0  — Shackleton downslope single-robot
"""

from rexmi_rl.tasks.locomotion.velocity.config.go2w.rough_env_cfg import (
    Go2wFlatEnvCfg,
    Go2wFlatEnvCfg_PLAY,
    Go2wRoughEnvCfg,
    Go2wRoughEnvCfg_PLAY,
)

# Crater demo configs — re-exported here so __init__.py entry_point strings resolve
from rexmi_rl.tasks.locomotion.velocity.config.go2w.crater_env_cfg import (
    # Upslope (floor → rim)
    LunarCraterType1EnvCfg,
    LunarCraterType1EnvCfg_PLAY,
    LunarCraterType2EnvCfg,
    LunarCraterType2EnvCfg_PLAY,
    LunarCraterType3EnvCfg,
    LunarCraterType3EnvCfg_PLAY,
    # Downslope (rim → floor)
    LunarCraterType1DownEnvCfg,
    LunarCraterType1DownEnvCfg_PLAY,
    LunarCraterType2DownEnvCfg,
    LunarCraterType2DownEnvCfg_PLAY,
    LunarCraterType3DownEnvCfg,
    LunarCraterType3DownEnvCfg_PLAY,
)

# Re-export for gym entry_point discovery
__all__ = [
    # Velocity-tracking training / play
    "Go2wFlatEnvCfg",
    "Go2wFlatEnvCfg_PLAY",
    "Go2wRoughEnvCfg",
    "Go2wRoughEnvCfg_PLAY",
    # Lunar crater demo — upslope
    "LunarCraterType1EnvCfg",
    "LunarCraterType1EnvCfg_PLAY",
    "LunarCraterType2EnvCfg",
    "LunarCraterType2EnvCfg_PLAY",
    "LunarCraterType3EnvCfg",
    "LunarCraterType3EnvCfg_PLAY",
    # Lunar crater demo — downslope
    "LunarCraterType1DownEnvCfg",
    "LunarCraterType1DownEnvCfg_PLAY",
    "LunarCraterType2DownEnvCfg",
    "LunarCraterType2DownEnvCfg_PLAY",
    "LunarCraterType3DownEnvCfg",
    "LunarCraterType3DownEnvCfg_PLAY",
]
