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
"""

from rexmi_rl.tasks.locomotion.velocity.config.go2w.rough_env_cfg import (
    Go2wFlatEnvCfg,
    Go2wFlatEnvCfg_PLAY,
    Go2wRoughEnvCfg,
    Go2wRoughEnvCfg_PLAY,
)

# Re-export for gym entry_point discovery
__all__ = [
    "Go2wFlatEnvCfg",
    "Go2wFlatEnvCfg_PLAY",
    "Go2wRoughEnvCfg",
    "Go2wRoughEnvCfg_PLAY",
]
