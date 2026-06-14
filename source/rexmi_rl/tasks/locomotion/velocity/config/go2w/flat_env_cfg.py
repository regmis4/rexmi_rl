# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Flat-terrain environment configuration for the Go2W — thin wrapper.

Why a separate file?
---------------------
Isaac Lab's convention separates "rough" (base) and "flat" configs so that:
  1. rough_env_cfg.py  defines the robot-specific overrides (joints, rewards, etc.)
  2. flat_env_cfg.py   is a thin override that switches terrain to flat and
                       re-tunes any reward that doesn't make sense on flat ground.

In Phase 1 our rough_env_cfg.py already forces flat terrain, so this file is
mostly a pass-through.  It exists to keep the file structure consistent with
Isaac Lab conventions — when we add rough/staircase terrain in Phase 2, we'll
use rough_env_cfg.py for that and keep flat here as the clean baseline.

Registered environments (from __init__.py)
------------------------------------------
  RexmiRl-Go2w-Velocity-Flat-v0       — training (4096 envs)
  RexmiRl-Go2w-Velocity-Flat-Play-v0  — visualisation (50 envs, no noise)
"""

from isaaclab.utils import configclass

# Import the base Go2W config which already sets flat terrain and
# wheel-only actions in its __post_init__.
from rexmi_rl.tasks.locomotion.velocity.config.go2w.rough_env_cfg import (
    Go2wFlatEnvCfg,
    Go2wFlatEnvCfg_PLAY,
)


# ---------------------------------------------------------------------------
# We simply re-export the classes so __init__.py can reference this file
# using the standard "flat_env_cfg:Go2wFlatEnvCfg" entry point string.
# No additional overrides needed for Phase 1.
# ---------------------------------------------------------------------------

# Re-export for gym entry_point discovery
__all__ = ["Go2wFlatEnvCfg", "Go2wFlatEnvCfg_PLAY"]
