# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Go2W terrain capability evaluation configurations.

Each factory function returns a fully-configured environment that places
ALL 50 (or N) robots on the SAME terrain type at a SINGLE FIXED difficulty.
This gives a precise measurement of where the policy succeeds and fails.

Differences from training config
---------------------------------
• Fixed forward command (0.5 m/s) — no random sampling
• Single terrain type, no curriculum — all robots face the same obstacle
• No sensor noise — clean observations for fair measurement
• No random pushes or external forces — isolate terrain difficulty
• 50 robots, env_spacing = 8 m (one per terrain tile)

Usage (from scripts/eval.py)
-------------------------------
    from rexmi_rl.tasks.locomotion.velocity.config.go2w.eval_env_cfg import (
        EVAL_VARIANTS,
    )
    for name, cfg_fn in EVAL_VARIANTS:
        cfg = cfg_fn()           # returns a Go2wEvalEnvCfg instance
        env = ManagerBasedRLEnv(cfg=cfg)
        ...

EVAL_VARIANTS is a list of (variant_name: str, cfg_fn: Callable[[], Go2wEvalEnvCfg]).
"""

import math

from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.trimesh.mesh_terrains_cfg import (
    MeshPyramidStairsTerrainCfg,
    MeshInvertedPyramidStairsTerrainCfg,
    MeshRandomGridTerrainCfg,
)
from isaaclab.terrains.height_field.hf_terrains_cfg import (
    HfRandomUniformTerrainCfg,
    HfPyramidSlopedTerrainCfg,
)
from isaaclab.utils import configclass

from rexmi_rl.tasks.locomotion.velocity.config.go2w.rough_env_cfg import Go2wRoughEnvCfg


# ============================================================================
# Base evaluation config — common overrides for all terrain characterisation runs
# ============================================================================

@configclass
class Go2wEvalEnvCfg(Go2wRoughEnvCfg):
    """
    Base evaluation config shared by all terrain variants.

    After calling cfg = Go2wEvalEnvCfg(), override cfg.scene.terrain.terrain_generator
    with a single-variant terrain (see factory functions below).

    What this base class changes vs training:
      1. Fixed forward command    — always 0.5 m/s, no random sampling
      2. No curriculum            — terrain_levels disabled
      3. No sensor noise          — enable_corruption = False
      4. No perturbations         — push_robot and external forces removed
      5. 50 robots, 8 m spacing   — one robot per terrain tile
    """

    def __post_init__(self):
        # ---------------------------------------------------------------
        # Apply all Go2W rough env overrides (robot, actions, rewards,
        # height scanner, terrain generator, …) from the parent class.
        # We will REPLACE terrain_generator in the factory functions below.
        # ---------------------------------------------------------------
        super().__post_init__()

        # ==================================================================
        # 1. FIXED FORWARD-ONLY VELOCITY COMMAND
        # ==================================================================
        # Always command 0.5 m/s forward.  Fixed value (not a range) ensures
        # every episode has the same target — we are testing terrain difficulty,
        # not command-following generalisation.
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # ==================================================================
        # 2. ENVIRONMENT SIZE — 50 robots, one per terrain tile
        # ==================================================================
        # env_spacing = 8 m matches the 8 m × 8 m terrain tile size so each
        # robot starts centred on its own tile without overlap.
        self.scene.num_envs = 50
        self.scene.env_spacing = 8.0

        # ==================================================================
        # 3. NO TERRAIN CURRICULUM
        # ==================================================================
        # Robots must stay at the difficulty we assign — no auto-promotion.
        self.curriculum.terrain_levels = None

        # ==================================================================
        # 4. CLEAN OBSERVATIONS — no sensor noise
        # ==================================================================
        # During training, noise helps generalisation.  For evaluation we want
        # to measure the policy's true performance without noise artefacts.
        self.observations.policy.enable_corruption = False

        # ==================================================================
        # 5. NO PERTURBATIONS
        # ==================================================================
        # We want to isolate terrain difficulty from external disturbances.
        self.events.push_robot = None
        self.events.base_external_force_torque = None


# ============================================================================
# Internal helper — wrap one terrain type in a minimal generator
# ============================================================================

def _single_terrain_gen(sub_terrain_cfg, num_cols: int = 50) -> TerrainGeneratorCfg:
    """
    Build a TerrainGeneratorCfg with:
      • 1 difficulty row  — no curriculum levels
      • num_cols tiles    — one tile per environment
      • 8 m × 8 m tiles  — matches env_spacing

    The sub_terrain_cfg must have proportion=1.0 (only one terrain type) and
    step_height_range / slope_range / noise_range set to a FIXED value (a, a)
    so every tile has the exact same difficulty.
    """
    return TerrainGeneratorCfg(
        seed=42,
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=1,          # single difficulty level — no curriculum
        num_cols=num_cols,   # one tile per env
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        sub_terrains={"terrain": sub_terrain_cfg},
    )


# ============================================================================
# Factory functions — one per terrain family
# ============================================================================
# Each returns a ready-to-use Go2wEvalEnvCfg instance.
# Call cfg_fn() to get the config, then optionally adjust cfg.scene.num_envs.

def stairs_up_cfg(step_cm: int) -> Go2wEvalEnvCfg:
    """
    Pyramid stairs (ascending) — robot must climb up step_cm cm steps.

    step_cm : step height in centimetres (e.g. 8 → 8 cm steps)
    """
    step_m = step_cm / 100.0
    cfg = Go2wEvalEnvCfg()
    cfg.scene.terrain.terrain_generator = _single_terrain_gen(
        MeshPyramidStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(step_m, step_m),   # fixed — not a range
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        )
    )
    return cfg


def stairs_down_cfg(step_cm: int) -> Go2wEvalEnvCfg:
    """
    Inverted pyramid stairs (descending) — robot must descend step_cm cm steps.

    step_cm : step height in centimetres
    """
    step_m = step_cm / 100.0
    cfg = Go2wEvalEnvCfg()
    cfg.scene.terrain.terrain_generator = _single_terrain_gen(
        MeshInvertedPyramidStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(step_m, step_m),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        )
    )
    return cfg


def boxes_cfg(height_cm: int) -> Go2wEvalEnvCfg:
    """
    Random grid of boxes — cobblestone-like uneven surface.

    height_cm : fixed box height in centimetres (e.g. 10 → 10 cm boxes)
    """
    height_m = height_cm / 100.0
    cfg = Go2wEvalEnvCfg()
    cfg.scene.terrain.terrain_generator = _single_terrain_gen(
        MeshRandomGridTerrainCfg(
            proportion=1.0,
            grid_width=0.45,
            grid_height_range=(height_m, height_m),   # fixed height
            platform_width=2.0,
        )
    )
    return cfg


def slope_cfg(slope_deg: int) -> Go2wEvalEnvCfg:
    """
    Pyramid slope — robot must climb/descend a uniform ramp.

    slope_deg : slope angle in degrees (e.g. 15 → 15° ramp)
    """
    slope_rad = math.radians(slope_deg)
    cfg = Go2wEvalEnvCfg()
    cfg.scene.terrain.terrain_generator = _single_terrain_gen(
        HfPyramidSlopedTerrainCfg(
            proportion=1.0,
            slope_range=(slope_rad, slope_rad),   # fixed slope
            platform_width=2.0,
            border_width=0.25,
        )
    )
    return cfg


def rough_cfg(noise_cm: int) -> Go2wEvalEnvCfg:
    """
    Random rough heightfield — gravel/bumpy surface with fixed amplitude.

    noise_cm : noise amplitude in centimetres (e.g. 6 → ±6 cm bumps)
    """
    noise_m = noise_cm / 100.0
    cfg = Go2wEvalEnvCfg()
    cfg.scene.terrain.terrain_generator = _single_terrain_gen(
        HfRandomUniformTerrainCfg(
            proportion=1.0,
            noise_range=(noise_m, noise_m),   # fixed amplitude
            noise_step=0.02,
            border_width=0.25,
        )
    )
    return cfg


# ============================================================================
# Master evaluation variant list
# ============================================================================
# Format: list of (variant_name: str, factory_fn: Callable[[], Go2wEvalEnvCfg])
#
# variant_name conventions:
#   "stairs_up_Xcm"   — ascending pyramid stairs, X cm step height
#   "stairs_down_Xcm" — descending inverted stairs, X cm step height
#   "boxes_Xcm"       — random grid boxes, X cm height
#   "slope_Xdeg"      — pyramid slope, X degrees
#   "rough_Xcm"       — random rough heightfield, X cm amplitude
#
# Total: 9 + 9 + 6 + 7 + 5 = 36 variants

EVAL_VARIANTS: list[tuple[str, callable]] = []

# -- Ascending pyramid stairs (9 variants) --
# Training range was (0.05, 0.23) m.  We sweep the full range and a little beyond.
for _s in [3, 5, 8, 10, 12, 15, 18, 20, 23]:
    EVAL_VARIANTS.append((f"stairs_up_{_s}cm", lambda s=_s: stairs_up_cfg(s)))

# -- Descending pyramid stairs (9 variants) --
for _s in [3, 5, 8, 10, 12, 15, 18, 20, 23]:
    EVAL_VARIANTS.append((f"stairs_down_{_s}cm", lambda s=_s: stairs_down_cfg(s)))

# -- Random box terrain (6 variants) --
# Training range was (0.05, 0.20) m.
for _h in [3, 5, 8, 10, 15, 20]:
    EVAL_VARIANTS.append((f"boxes_{_h}cm", lambda h=_h: boxes_cfg(h)))

# -- Pyramid slopes (7 variants) --
# Training range was (0.0, 0.4) rad ≈ 0–23°.
for _d in [2, 5, 8, 10, 15, 20, 23]:
    EVAL_VARIANTS.append((f"slope_{_d}deg", lambda d=_d: slope_cfg(d)))

# -- Random rough heightfield (5 variants) --
# Training range was (0.02, 0.10) m.
for _n in [2, 4, 6, 8, 10]:
    EVAL_VARIANTS.append((f"rough_{_n}cm", lambda n=_n: rough_cfg(n)))
