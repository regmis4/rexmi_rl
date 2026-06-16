# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Lunar crater traversal environment configurations for the Go2W investor demo.

Overview
--------
These configs deploy the trained Phase 6-Optimized rough-terrain policy onto
procedurally-generated lunar south pole crater wall terrains.  No re-training
is required — the existing policy checkpoint is loaded by play.py unchanged.

Each config wraps Go2wRoughEnvCfg (which already has the height scanner,
full 16-DOF actions, and all rewards) and replaces the terrain generator with
one of the three LOLA-calibrated crater wall profiles from crater_terrain.py.

Environment hierarchy
---------------------
Go2wFlatEnvCfg                     ← robot, actions, flat rewards
  └── Go2wRoughEnvCfg              ← height scanner, rough terrain generator
        └── LunarCraterBaseEnvCfg  ← demo overrides (no curriculum / noise / push)
              ├── LunarCraterType1EnvCfg       (Haworth-archetype, gentle)
              ├── LunarCraterType1EnvCfg_PLAY  (1 env, for recording)
              ├── LunarCraterType2EnvCfg       (Faustini-archetype, primary demo)
              ├── LunarCraterType2EnvCfg_PLAY  (1 env, for recording)
              ├── LunarCraterType3EnvCfg       (Shackleton-archetype, steep)
              └── LunarCraterType3EnvCfg_PLAY  (1 env, for recording)

Terrain tile geometry (all types)
----------------------------------
  Tile size     : 32 m × 32 m
  Resolution    : 10 cm/pixel (320 × 320 pixels)
  Direction     : col 0 = crater floor (low) → col 319 = crater rim (high)
  env_spacing   : 34 m (tile + 2 m buffer)
  num_envs      : 10 for demo, 1 for recording variant

Usage
-----
  # Watch the primary Type 2 demo (10 parallel robots on crater wall):
  ./run.sh scripts/play.py --task RexmiRl-Go2w-Crater-Type2-Play-v0 \\
      --load_run go2w_velocity_rough/2026-06-14_16-34-44

  # Single-robot recording variant:
  ./run.sh scripts/play.py --task RexmiRl-Go2w-Crater-Type2-Record-v0 \\
      --load_run go2w_velocity_rough/2026-06-14_16-34-44

  # Type 1 (gentle ancient crater — easiest demo):
  ./run.sh scripts/play.py --task RexmiRl-Go2w-Crater-Type1-Play-v0 \\
      --load_run go2w_velocity_rough/2026-06-14_16-34-44

  # Type 3 (Shackleton — future Phase 8 demo):
  ./run.sh scripts/play.py --task RexmiRl-Go2w-Crater-Type3-Play-v0 \\
      --load_run go2w_velocity_rough/2026-06-14_16-34-44
"""

import math

from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass

from rexmi_rl.tasks.locomotion.velocity.config.go2w.rough_env_cfg import Go2wRoughEnvCfg
from rexmi_rl.tasks.locomotion.velocity.config.go2w.crater_terrain import (
    CraterType1WallCfg,
    CraterType2WallCfg,
    CraterType3WallCfg,
)


# ===========================================================================
# Terrain generator factory
# ===========================================================================

def _crater_terrain_gen(sub_terrain_cfg, num_cols: int = 10) -> TerrainGeneratorCfg:
    """
    Create a TerrainGeneratorCfg for a crater wall demo.

    Parameters
    ----------
    sub_terrain_cfg : CraterType*WallCfg
        The crater type configuration (Type1, Type2, or Type3).
    num_cols : int
        Number of parallel terrain tiles to generate (= number of demo robots).

    Returns
    -------
    TerrainGeneratorCfg
        32 m × 32 m tiles at 10 cm resolution, 1 difficulty row (no curriculum).
    """
    return TerrainGeneratorCfg(
        seed=0,
        size=(32.0, 32.0),           # 32 m × 32 m per tile
        border_width=20.0,           # flat border around the terrain grid
        num_rows=1,                   # 1 difficulty level — no curriculum
        num_cols=num_cols,            # one tile per robot
        horizontal_scale=0.1,        # 10 cm/pixel
        vertical_scale=0.005,        # 5 mm per raw height unit
        slope_threshold=0.75,        # trim steep triangles (cosmetic)
        use_cache=False,             # always generate fresh (no stale cache)
        sub_terrains={"crater_wall": sub_terrain_cfg},
    )


# ===========================================================================
# Base demo environment — shared overrides for all crater types
# ===========================================================================

@configclass
class LunarCraterBaseEnvCfg(Go2wRoughEnvCfg):
    """
    Base configuration for lunar crater traversal demos.

    Inherits the full Go2W rough-terrain MDP (height scanner, 16-DOF actions,
    Phase 6-Opt rewards) and overrides the following for demo use:

      1. No terrain curriculum     — robot stays on the crater wall it's placed on
      2. No sensor noise           — clean observations for fair policy evaluation
      3. No random pushes          — isolate terrain difficulty from perturbations
      4. Fixed forward command     — 0.5 m/s in the traverse direction
      5. 10 parallel robots        — side-by-side traversal for visual comparison
      6. 34 m env spacing          — matches 32 m tile + 2 m clearance

    Subclasses must override `self.scene.terrain.terrain_generator` with the
    appropriate crater type terrain generator.
    """

    def __post_init__(self):
        # ---------------------------------------------------------------
        # Apply all Go2W rough-terrain overrides first:
        #   • Robot (Go2W USD) + full 16-DOF action space
        #   • Height scanner (1.6 m × 1.0 m grid, 160 rays)
        #   • Stagnation penalty + climb_progress reward
        #   • Terminations, events, commands
        # ---------------------------------------------------------------
        super().__post_init__()

        # ==============================================================
        # 1. EPISODE LENGTH — long enough to see a full traversal
        # ==============================================================
        # Training default is ~20 s (optimised for fast curriculum cycling).
        # At 0.5 m/s over a 32 m tile the robot needs ≥64 s for one crossing.
        # 300 s = 5 minutes gives time for 4–5 traversals per demo episode.
        self.episode_length_s = 300.0

        # ==============================================================
        # 2. ENVIRONMENT SIZE — 10 robots, 34 m spacing
        # ==============================================================
        # 34 m > 32 m tile size: 1 m clearance each side prevents robots
        # from the adjacent tile appearing in each other's camera view.
        self.scene.num_envs = 10
        self.scene.env_spacing = 34.0

        # ==============================================================
        # 2. NO TERRAIN CURRICULUM
        # ==============================================================
        # The crater wall is a fixed terrain — there is no difficulty
        # progression.  Robots must stay on the same tile throughout the
        # demo episode.
        self.curriculum.terrain_levels = None

        # ==============================================================
        # 3. CLEAN OBSERVATIONS — no sensor noise
        # ==============================================================
        # During training, noise builds robustness.  For the investor demo
        # we want deterministic, repeatable behaviour with no noise artefacts.
        self.observations.policy.enable_corruption = False

        # ==============================================================
        # 4. NO PERTURBATIONS — clean traversal showcase
        # ==============================================================
        self.events.push_robot = None
        self.events.base_external_force_torque = None

        # ==============================================================
        # 5. FIXED FORWARD VELOCITY COMMAND
        # ==============================================================
        # Always command 0.5 m/s in the traverse direction (upslope toward rim).
        # The robot will demonstrate how it adapts to increasing slope angle
        # as it crosses from the floor zone to the rim zone.
        #
        # 0.5 m/s = maximum achievable speed with the Go2W wheel radius (5 cm)
        # at full motor speed (10 rad/s × 0.05 m).  Commanding max speed keeps
        # the policy continuously active — no coasting episodes.
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # ==============================================================
        # 6. SPAWN POSITION — crater floor end, facing upslope
        # ==============================================================
        # The parent Go2wFlatEnvCfg uses yaw=(-π, +π) — fully random.
        # For the crater demo we fix yaw=0 so all robots face upslope
        # (+x = floor→rim direction) and spawn near the crater floor.
        #
        # Tile geometry (32 m × 32 m):
        #   origin  = tile centre = (16 m, 16 m, terrain_z_at_centre)
        #   floor   ≈ tile x = 0  → x_offset = 0 - 16 = -16 m from origin
        #   rim     ≈ tile x = 32 → x_offset = 32 - 16 = +16 m from origin
        #
        # We spawn 2–4 m from the floor edge to give the robot a flat
        # warm-up zone before the slope starts.
        #   tile x ≈ 2–4 m  →  x_offset = -14 to -12 m
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-14.0, -12.0),  # tile x ≈ 2–4 m (flat PSR floor zone)
                "y": (-3.0, 3.0),     # lateral spread across tile width
                "yaw": (0.0, 0.0),    # always face upslope (+x = floor→rim)
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # ==============================================================
        # NOTE: subclasses MUST override terrain_generator
        # ==============================================================
        # Go2wRoughEnvCfg already sets terrain_type="generator" with the
        # procedural rough terrain.  Each LunarCrater*EnvCfg subclass
        # replaces terrain_generator with the appropriate crater profile.


# ===========================================================================
# Type 1 — Ancient complex crater (Haworth / Shoemaker / Nobile archetype)
# ===========================================================================

@configclass
class LunarCraterType1EnvCfg(LunarCraterBaseEnvCfg):
    """
    Type 1 lunar crater traversal demo — ancient, heavily degraded crater wall.

    Archetype: Haworth / Shoemaker / Nobile (52–79 km, Pre-Nectarian, ~4.2 Ga)
    Wall slope: 7.5–14°  |  Max: ~20°  |  Policy capability: ✅ all zones

    This is the gentlest crater demo — an excellent starting point for investors
    unfamiliar with the robot.  Slopes match the training distribution exactly,
    so the policy demonstrates its best-case performance.

    Narrative: "This is Haworth crater.  At 52 km diameter it's one of the largest
    PSR craters near the south pole.  The robot traverses the ancient inner wall —
    a 4 km long slope at 10–14° average grade."
    """

    def __post_init__(self):
        super().__post_init__()

        # Replace the rough-terrain generator with the Type 1 crater profile.
        # 10 tiles of 32 m × 32 m, each representing a section of Haworth's wall.
        self.scene.terrain.terrain_generator = _crater_terrain_gen(
            CraterType1WallCfg(proportion=1.0), num_cols=10
        )

        # Spawn z correction (upslope).
        # Floor at x=3 m has terrain height ≈ 0.08 m.
        # Tile centre (x=16 m) has terrain height ≈ 1.43 m (= env_origin_z).
        # Without correction the robot spawns 1.35 m above the floor terrain
        # and falls — fine for Type 1 but still messy.  Shift z down so it
        # lands cleanly ≈ 0.3–0.5 m above the actual floor surface.
        self.events.reset_base.params["pose_range"]["z"] = (-1.6, -1.0)


@configclass
class LunarCraterType1EnvCfg_PLAY(LunarCraterType1EnvCfg):
    """
    Type 1 single-robot recording variant.

    Use this for investor presentations and video captures.
    One robot on one terrain tile — maximally cinematic.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        # Single tile for the single robot
        self.scene.terrain.terrain_generator = _crater_terrain_gen(
            CraterType1WallCfg(proportion=1.0), num_cols=1
        )


@configclass
class LunarCraterType1DownEnvCfg(LunarCraterType1EnvCfg):
    """
    Type 1 crater — downslope variant.

    Spawns near the rim, robot faces downslope (-x direction, yaw=π).
    Useful for showing the robot descending from the rim into the PSR floor —
    the reverse of the default upslope demo.
    """

    def __post_init__(self):
        super().__post_init__()
        # Spawn near rim end (tile x ≈ 28–30 m), face downslope (yaw=π).
        # Rim terrain (x=29 m) ≈ 4.12 m above tile floor.
        # env_origin_z ≈ 1.43 m (centre height) → z correction = +2.69 m.
        self.events.reset_base.params = {
            "pose_range": {
                "x": (12.0, 14.0),           # tile x ≈ 28–30 m (near rim)
                "y": (-3.0, 3.0),
                "yaw": (math.pi, math.pi),   # face downslope (-x direction)
                "z": (2.4, 3.2),             # +rim_height - centre_height
            },
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        }


@configclass
class LunarCraterType1DownEnvCfg_PLAY(LunarCraterType1DownEnvCfg):
    """Type 1 downslope — single-robot recording variant."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.terrain.terrain_generator = _crater_terrain_gen(
            CraterType1WallCfg(proportion=1.0), num_cols=1
        )


# ===========================================================================
# Type 2 — Intermediate crater (Faustini archetype) — PRIMARY DEMO TARGET
# ===========================================================================

@configclass
class LunarCraterType2EnvCfg(LunarCraterBaseEnvCfg):
    """
    Type 2 lunar crater traversal demo — Faustini archetype (PRIMARY DEMO).

    Archetype: Faustini (42 km, Nectarian, ~3.9 Ga)
    Wall slope: 11.5–17.5° (peak)  |  Max: ~25°  |  Policy capability: ✅ all zones

    Faustini is the most scientifically compelling crater for investors:
      • Named after a real, known south pole crater visible in LRO images
      • Water ice confirmed in the PSR floor (LCROSS / LRO data)
      • PRIME robotics target for future ISRU (in-situ resource utilisation)
      • Accessible slope angles that the current policy handles cleanly

    The robot traverses from the flat PSR floor (1.5°) through progressively
    steeper wall sections (11.5° → 15° → 17.5°) to the rim crest (6.5°).
    Height gain of ~6.7 m over 32 m = compelling visual demonstration of
    the wheeled-legged architecture's slope-handling capability.

    Narrative: "This is Faustini crater — one of the primary targets for
    lunar water ice extraction.  The robot descends 6.7 m from the rim to the
    permanently shadowed floor, demonstrating the slope negotiation required
    for any mission to access those ice deposits."
    """

    def __post_init__(self):
        super().__post_init__()

        # Replace the rough-terrain generator with the Type 2 (Faustini) profile.
        self.scene.terrain.terrain_generator = _crater_terrain_gen(
            CraterType2WallCfg(proportion=1.0), num_cols=10
        )

        # Spawn z correction (upslope).
        # Floor at x=3 m ≈ 0.08 m.  Tile centre (x=16 m) ≈ 2.40 m (env_origin_z).
        # Without correction the robot falls 2.3 m → tips over on landing.
        self.events.reset_base.params["pose_range"]["z"] = (-2.6, -2.0)


@configclass
class LunarCraterType2EnvCfg_PLAY(LunarCraterType2EnvCfg):
    """
    Type 2 (Faustini) single-robot recording variant.

    PRIMARY RECORDING CONFIG for investor demo videos.
    One robot on one 32 m × 32 m Faustini wall tile.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.terrain.terrain_generator = _crater_terrain_gen(
            CraterType2WallCfg(proportion=1.0), num_cols=1
        )


@configclass
class LunarCraterType2DownEnvCfg(LunarCraterType2EnvCfg):
    """
    Type 2 Faustini crater — downslope variant.  (PRIMARY DOWNSLOPE DEMO)

    Spawns near the rim, robot faces downslope.
    Narrative: "The robot descends from the rim into the permanently shadowed
    floor — 6.7 m of height change through progressively gentler slopes as it
    approaches the ice-bearing PSR."
    """

    def __post_init__(self):
        super().__post_init__()
        # Rim terrain (x=29 m) ≈ 6.05 m above floor.
        # env_origin_z ≈ 2.40 m → z correction = +3.65 m.
        self.events.reset_base.params = {
            "pose_range": {
                "x": (12.0, 14.0),           # tile x ≈ 28–30 m (near rim)
                "y": (-3.0, 3.0),
                "yaw": (math.pi, math.pi),   # face downslope
                "z": (3.4, 4.2),             # +rim_height - centre_height
            },
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        }


@configclass
class LunarCraterType2DownEnvCfg_PLAY(LunarCraterType2DownEnvCfg):
    """Type 2 downslope — single-robot recording variant."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.terrain.terrain_generator = _crater_terrain_gen(
            CraterType2WallCfg(proportion=1.0), num_cols=1
        )


# ===========================================================================
# Type 3 — Young steep crater (Shackleton archetype)
# ===========================================================================

@configclass
class LunarCraterType3EnvCfg(LunarCraterBaseEnvCfg):
    """
    Type 3 lunar crater traversal demo — Shackleton archetype (steep, young).

    Archetype: Shackleton (21 km, Imbrian, ~3.6 Ga)
    Wall slope: 22.5–35°  |  Max: 35°  |  Policy capability: ⚠️ Phase 8 required

    Shackleton is the most famous lunar south pole crater:
      • Sits directly at the South Pole (89.9°S)
      • Permanently shadowed interior (PSR) for ~4 billion years
      • Selected as candidate landing/base site for Artemis and commercial missions
      • Most sharply preserved walls of any major PSR crater

    ⚠️  CURRENT STATUS: The mid-wall (31.5°) and rim (35°) exceed the Phase 6-Opt
    policy training range (~20°).  The robot will traverse the floor and lower wall
    (3–22.5°) successfully, then struggle on the upper wall.  This is intentional —
    it motivates the Phase 8 training run and shows a clear future capability roadmap.

    Narrative: "This is Shackleton — the summit of the lunar south pole.  The current
    policy handles the floor and lower wall.  Phase 8 training will extend capability
    to the 35° upper walls, enabling full crater rim descent."
    """

    def __post_init__(self):
        super().__post_init__()

        # Replace the rough-terrain generator with the Type 3 (Shackleton) profile.
        self.scene.terrain.terrain_generator = _crater_terrain_gen(
            CraterType3WallCfg(proportion=1.0), num_cols=10
        )

        # Spawn z correction (upslope).
        # Floor at x=3 m ≈ 0.16 m (3° slope).  Centre (x=16 m) ≈ 6.74 m.
        # Without correction the robot falls 6.6 m → always tips on impact.
        self.events.reset_base.params["pose_range"]["z"] = (-6.9, -6.3)


@configclass
class LunarCraterType3EnvCfg_PLAY(LunarCraterType3EnvCfg):
    """
    Type 3 (Shackleton) single-robot recording variant.

    Useful for Phase 8 capability comparison: record the same scene before and
    after the steeper-slope training run to show clear policy improvement.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.terrain.terrain_generator = _crater_terrain_gen(
            CraterType3WallCfg(proportion=1.0), num_cols=1
        )


@configclass
class LunarCraterType3DownEnvCfg(LunarCraterType3EnvCfg):
    """
    Type 3 Shackleton crater — downslope variant.

    Spawns near the rim (35° zone), robot faces downslope.
    The robot immediately encounters the steepest part first — useful for
    demonstrating the Phase 8 capability limit from above.
    """

    def __post_init__(self):
        super().__post_init__()
        # Rim terrain (x=29 m) ≈ 14.17 m above floor.
        # env_origin_z ≈ 6.74 m → z correction = +7.43 m.
        self.events.reset_base.params = {
            "pose_range": {
                "x": (12.0, 14.0),           # tile x ≈ 28–30 m (near 35° rim)
                "y": (-3.0, 3.0),
                "yaw": (math.pi, math.pi),   # face downslope
                "z": (7.2, 8.2),             # +rim_height - centre_height
            },
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        }


@configclass
class LunarCraterType3DownEnvCfg_PLAY(LunarCraterType3DownEnvCfg):
    """Type 3 downslope — single-robot recording variant."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.terrain.terrain_generator = _crater_terrain_gen(
            CraterType3WallCfg(proportion=1.0), num_cols=1
        )
