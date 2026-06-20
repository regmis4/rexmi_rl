# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Go2W dedicated steep-slope training environment — Phase 8.

Motivation
----------
The unified rough-terrain policy (model_8996.pt, Phase 7) achieved excellent
performance on stairs (up/down to 23 cm), boxes (to 20 cm), rough, and moderate
slopes (0°–23°).  Attempts to extend it to 23°–45° slopes within the same
training run failed because:

  1. Curriculum reset: every ``--load_run`` restart resets the curriculum to
     level 0, so the policy spends iterations re-learning easy terrain instead
     of advancing to hard slopes.

  2. Reward conflict: orientation penalties needed for steep slopes (loose) are
     in direct tension with those needed for stable flat/stair driving (tight).
     A single set of weights cannot serve both — the policy consistently found
     exploits (wheel-lift, thigh-salute) when the penalty was loosened.

The solution: keep model_8996 as the production rough policy and train a
SEPARATE specialised policy exclusively on steep slopes.

Inheritance chain
-----------------
Go2wRoughEnvCfg (Phase 7 — stairs/boxes/rough, flat_orientation=-0.8, bad_orient=1.0 rad)
    └── Go2wSteepSlopeEnvCfg  ← this file
            • terrain: ONLY HfPyramidSlopedTerrainCfg, 23°–45°
            • flat_orientation_l2: -0.1  (slopes require sustained body tilt)
            • bad_orientation limit: 1.4 rad (80°)  — headroom for 45° + oscillations
            • All other rewards/scanner/stagnation/climb_progress inherited

Design decisions
----------------
Terrain:
  Single sub-terrain (HfPyramidSlopedTerrainCfg) with slope_range=(23°, 45°).
  With num_rows=10, Isaac Lab's curriculum interpolates difficulty across rows:
    row 0  → slope = 23° (easiest — handoff from model_8996 moderate slopes)
    row 9  → slope = 45° (hardest — physical upper limit)
  The robot starts at 23° and advances as its tracking improves, naturally
  avoiding the cold-start problem of immediately facing 45° slopes.

Orientation:
  flat_orientation_l2 = -0.1 (vs -0.8 in flat env, -0.8 inherited by rough env)
    On a 35° slope: -0.1 × (0.611 rad)² ≈ -0.037/step — nearly free to tilt.
    On a 45° slope: -0.1 × (0.785 rad)² ≈ -0.062/step — still negligible vs
    tracking reward (+1.88/step at 0.5 m/s).
  bad_orientation = 1.4 rad (80°) — gives 35° headroom above 45° max slope.

Everything inherited from Go2wRoughEnvCfg:
  • Height scanner (160-dim, same obs space → model_8996 weights load cleanly)
  • stagnation penalty (-1.5) — escape if stuck on a slope face
  • climb_progress (hybrid 0.4/1.5) — always in obstacle_weight mode on slopes
    (max elevation >> 0.10 m threshold on any slope > 3°)
  • leg_deviation (-0.05 all joints) — weak bias, preserves climbing freedom
  • undesired_contacts (-1.0) — no spider-walking
  • terrain curriculum (terrain_levels_vel) — advances rows 0→9

Training command
----------------
    conda activate env_isaacsim
    python scripts/train.py --task RexmiRl-Go2w-Velocity-SteepSlope-v0 --headless \\
        --load_run go2w_velocity_rough/2026-06-14_20-03-41 --checkpoint model_8996.pt

This loads model_8996 weights only.  The curriculum resets to level 0 (= 23°
slopes) which is intentional — the policy builds slope intuition from the
easiest angle and advances to 45° as performance improves.
Logs → logs/rsl_rl/go2w_velocity_steep_slope/
"""

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp_utils

from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.height_field.hf_terrains_cfg import HfPyramidSlopedTerrainCfg

from rexmi_rl.tasks.locomotion.velocity.config.go2w.rough_env_cfg import Go2wRoughEnvCfg


@configclass
class Go2wSteepSlopeEnvCfg(Go2wRoughEnvCfg):
    """
    Dedicated steep-slope training environment for the Go2W.

    Inherits the full Phase 7 rough-terrain MDP (robot, actions, height scanner,
    stagnation, climb_progress, leg_deviation, curriculum) from Go2wRoughEnvCfg
    and replaces the mixed terrain generator with pure steep-slope terrain.

    Key differences vs Go2wRoughEnvCfg
    ------------------------------------
    1. Terrain: ONLY HfPyramidSlopedTerrainCfg, slope 23°–45°, 10 curriculum rows
       (row 0 = 23°, row 9 = 45° — smooth handoff from model_8996 which mastered 23°)
    2. flat_orientation_l2: -0.8 → -0.1  (slopes require sustained body tilt)
    3. bad_orientation limit: 1.0 → 1.4 rad (80°, gives 35° headroom above 45° max)
    4. Stagnation penalty: inherited (-1.5) — still useful when stuck on steep face
    5. Everything else: identical to Go2wRoughEnvCfg (Phase 7 baseline)
    """

    def __post_init__(self):
        # ---------------------------------------------------------------
        # Apply all rough-env overrides (robot, actions, scanner, rewards,
        # stagnation, climb_progress, curriculum) from parent class.
        # We will REPLACE the terrain generator below.
        # ---------------------------------------------------------------
        super().__post_init__()

        # ==================================================================
        # 1. TERRAIN — replace mixed rough terrain with pure steep slopes
        # ==================================================================
        # Curriculum rows 0–9 interpolate slope from 23° (row 0) to 45° (row 9).
        # Isaac Lab's HfPyramidSlopedTerrainCfg accepts a difficulty parameter
        # in [0, 1] and linearly interpolates within slope_range:
        #   slope = slope_range[0] + difficulty × (slope_range[1] - slope_range[0])
        # With difficulty = row / (num_rows - 1):
        #   row 0 → difficulty=0.0 → slope = 0.401 rad ≈ 23°  (matches model_8996 top)
        #   row 4 → difficulty=0.4 → slope = 0.554 rad ≈ 32°  (Shackleton lower wall)
        #   row 7 → difficulty=0.8 → slope = 0.708 rad ≈ 41°  (Shackleton upper wall)
        #   row 9 → difficulty=1.0 → slope = 0.785 rad ≈ 45°  (physical maximum)
        #
        # Single terrain type — all 20 tiles per row are the same slope angle,
        # so the curriculum measures exactly "can the robot track at X degrees?"
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            seed=0,
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=10,     # 10 difficulty levels: 23° (row 0) → 45° (row 9)
            num_cols=20,     # 20 parallel tiles per level
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={
                # Pure steep slopes — all tiles are the same pyramid slope type.
                # Proportion=1.0 because this is the only sub-terrain.
                "steep_slope": HfPyramidSlopedTerrainCfg(
                    proportion=1.0,
                    slope_range=(0.401, 0.785),  # 23° (0.401 rad) → 45° (0.785 rad)
                    platform_width=2.0,           # 2 m flat centre — safe starting point
                    border_width=0.25,
                ),
            },
        )

        # ==================================================================
        # 2. HIP CROSSING PENALTY — targeted "weirdo territory" boundary
        # ==================================================================
        # Problem observed at iter ~2000: rear leg crossing exploit.
        # Left wheel on ground, right wheel in air, right wheel spinning fast,
        # left leg migrating to where right leg should be.
        #
        # Root cause: leg_deviation=-0.05 costs only 0.025/step for a 0.5 rad
        # hip crossing — not enough to make the exploit unprofitable.
        # A linear L1 penalty cannot discriminate "acceptable ±0.15 rad slope
        # lean" from "±0.5 rad weirdo crossing"; both are penalised the same
        # proportionally, so any crossing that confers stability wins.
        #
        # Fix: threshold-based penalty with a ±0.25 rad DEAD ZONE.
        #   • 0.15 rad lean  → excess = 0      → zero cost  ✓ (vy tracking free)
        #   • 0.50 rad cross → excess = 0.25   → cost = -2.0×0.25 = -0.50/step ✗
        #
        # Why -2.0 weight and NOT -0.5 (which killed vy tracking in attempt 1)?
        #   Attempt 1 used hip_deviation=-0.5 LINEAR on all hip deviation —
        #   even 0.05 rad cost 0.025/step, blocking lateral stepping entirely.
        #   This threshold version fires at ZERO cost inside ±0.25 rad, so
        #   lateral stepping (±0.15 rad) remains completely free.
        from rexmi_rl.tasks.locomotion.velocity.mdp import (
            hip_crossing_penalty as _hip_crossing_penalty,
        )
        from isaaclab.managers import SceneEntityCfg as _SECfg2

        self.rewards.hip_crossing = RewTerm(
            func=_hip_crossing_penalty,
            weight=-2.0,
            params={
                "threshold_rad": 0.25,
                "asset_cfg": _SECfg2("robot", joint_names=[".*_hip_joint"]),
            },
        )

        # Thigh salute penalty — same threshold pattern, generous dead zone.
        # On 35° slope: thigh uses ~0.20 rad for contact → 0.20 rad budget left.
        # 0.40 rad threshold leaves 0.20 rad free for vy/CG shifts.
        # Salute at 0.65 rad: excess=0.25, cost=-1.0×0.25=-0.25/step per group.
        # Weight -1.0 (softer than hip -2.0): all wheels stay on ground in salute,
        # it's less destabilising than lateral roll — reflects severity difference.
        # CALF penalty intentionally NOT added: calves need the most freedom for
        # terrain adaptation on varying slopes. Only add if a calf exploit appears.
        from rexmi_rl.tasks.locomotion.velocity.mdp import (
            joint_deviation_threshold as _jdt,
        )
        from isaaclab.managers import SceneEntityCfg as _SECfg3

        self.rewards.thigh_salute = RewTerm(
            func=_jdt,
            weight=-1.0,
            params={
                "threshold_rad": 0.40,
                "asset_cfg": _SECfg3("robot", joint_names=[".*_thigh_joint"]),
            },
        )

        # ==================================================================
        # 3. ORIENTATION — relax for steep slopes
        # ==================================================================
        # flat_orientation_l2: -0.8 (inherited) → -0.1
        #
        # On steep slopes the robot body MUST tilt to align with the surface:
        #   35° slope: body pitch ≈ 0.611 rad → cost = -0.1 × 0.611² ≈ -0.037/step
        #   45° slope: body pitch ≈ 0.785 rad → cost = -0.1 × 0.785² ≈ -0.062/step
        # These are negligible compared to velocity tracking (+1.88/step at 0.5 m/s).
        # -0.1 is still positive (rather than 0) so arbitrary sway is still slightly
        # penalised — the policy should tilt WITH the slope, not randomly.
        self.rewards.flat_orientation_l2.weight = -0.1

        # bad_orientation: 1.0 rad (57°, inherited) → 1.4 rad (80°)
        #
        # Without this relaxation, the episode terminates at 57° tilt, which can
        # happen the moment the robot pitches into a 45° slope (body tilt + terrain
        # tilt ≈ 90° worst case).  1.4 rad gives 35° headroom above the maximum
        # training slope (45°) for transient oscillations and lateral roll.
        self.terminations.bad_orientation = DoneTerm(
            func=mdp_utils.bad_orientation,
            params={"limit_angle": 1.4},  # 1.4 rad ≈ 80° from vertical
        )


@configclass
class Go2wSteepSlopeEnvCfg_PLAY(Go2wSteepSlopeEnvCfg):
    """
    Play-mode variant for steep-slope policy: fewer envs, no noise, no pushes.

    Use with:
        python scripts/play.py --task RexmiRl-Go2w-Velocity-SteepSlope-Play-v0

    Spawns 50 robots on steep-slope tiles (23°–45°, curriculum disabled by
    the play wrapper) so you can visually inspect the slope-traversal behaviour.
    """

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 8.0   # match 8 m × 8 m slope tile size
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
