# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Slope-turn environment configuration for the Go2W wheeled quadruped.

PURPOSE
-------
This is Phase B of the two-phase turn training:

  Phase A — Go2wTurnEnvCfg (flat):
    Trains the basic pivot-turn mechanics on flat terrain.
    Robot learns: stop wheels, step legs, rotate body.

  Phase B — Go2wSlopeTurnEnvCfg (THIS FILE):
    Fine-tunes the pivot-turn on slopes up to 35°.
    Robot learns: same pivot mechanics but while fighting gravity,
    maintaining footing on tilted terrain, and counteracting
    slope-induced roll/pitch during the turn.

INHERITANCE
-----------
Inherits from Go2wTurnEnvCfg (NOT Go2wRoughEnvCfg directly).
This preserves all turn-specific reward shaping from Phase A:
  - heading_progress reward (prevents body-rocking exploit)
  - wheel_lock penalty
  - track_ang_vel_z_exp weight=3.0
  - is_alive weight=0.5

The ONLY changes from flat-turn are:
  1. Terrain: pyramid slopes 15°–35° (replaces flat num_rows=1)
  2. ang_vel_xy_l2: -0.3 (very forgiving — gravity forces this on slope)
  3. flat_orientation_l2: removed (body MUST tilt on slope)
  4. lin_vel_z_l2: -0.5 (relaxed from -1.5 — body naturally rises on slope)
  5. Curriculum: re-enabled (difficulty progression from 15° to 35°)
  6. Push events: re-enabled (robustness training)

WHY ang_vel_xy_l2 MUST BE VERY LOW ON SLOPE
--------------------------------------------
On a 35° slope:
  - gravity component perpendicular to body = g × sin(35°) ≈ 5.6 m/s²
  - any asymmetric leg movement shifts weight → body roll
  - even standing still generates small roll oscillations (wheel compliance)
  At weight=-0.5: roll rate of 0.3 rad/s costs -0.045/step → acceptable
  At weight=-0.5: on flat this was already too tight (torso-twist exploit)
  At weight=-0.3: roll rate of 0.3 rad/s costs -0.027/step → very forgiving

WHY flat_orientation_l2 MUST BE REMOVED
-----------------------------------------
On a 35° slope the body IS tilted ~35° from the gravity vector.
flat_orientation_l2 penalises any deviation from upright orientation.
Keeping it at -0.2 on a 35° slope costs:
  -0.2 × (35° in rad)² = -0.2 × 0.374 = -0.075/step EVERY STEP
This is not recoverable — the robot is always being penalised just for
existing on the slope. Remove it entirely for slope training.

WARM-START
----------
Warm-start from the converged flat-turn checkpoint. The flat-turn policy
already knows: stop wheels, step legs, rotate body. The slope-turn training
just needs to adapt this gait to the tilted terrain and gravity asymmetry.

TRAINING COMMAND
----------------
  # First find the latest flat-turn checkpoint:
  ls logs/rsl_rl/go2w_velocity_turn/ | sort | tail -1

  conda activate env_isaacsim
  python scripts/train.py --task RexmiRl-Go2w-Velocity-SlopeTurn-v0 --headless \\
      --load_run go2w_velocity_turn/<latest_flat_turn_run> \\
      --checkpoint model_<N>.pt \\
      --max_iterations 1500

TENSORBOARD HEALTH SIGNALS
---------------------------
  heading_progress     → should START >0.10 (inherited from flat-turn)
  base_contact         → should stay <5% even on 35° (robot knows balance)
  bad_orientation      → may be 5-15% initially (slope destabilises turns)
  terrain_levels       → should increase as policy adapts to steeper slopes
  wheel_lock           → should stay low (-1 to -3, inherited)
  track_ang_vel_z_exp  → may temporarily drop then recover (slope destabilises)
"""

from __future__ import annotations

from isaaclab.utils import configclass

from rexmi_rl.tasks.locomotion.velocity.config.go2w.turn_env_cfg import (
    Go2wTurnEnvCfg,
    Go2wTurnEnvCfg_PLAY,
)


@configclass
class Go2wSlopeTurnEnvCfg(Go2wTurnEnvCfg):
    """
    Slope-turn training environment — pivot turning on slopes up to 35°.

    Inherits from Go2wTurnEnvCfg (flat-turn) and adapts for slope operation:
      - Terrain: pyramid slopes 15°–35°, 6 difficulty rows
      - Rewards: relaxed orientation penalties (body tilts on slope)
      - Curriculum: re-enabled (advances from 15° to 35° as yaw improves)
    """

    def __post_init__(self):
        super().__post_init__()

        # ==================================================================
        # 1. TERRAIN: pyramid slopes 15°–35°
        # ==================================================================
        # Override the flat num_rows=1 from Go2wTurnEnvCfg.
        # Use pyramid slope terrain only — clean slope geometry for learning
        # cross-slope turning without stair/box complexity.
        #
        # num_rows=6: 6 difficulty levels from ~15° to ~35°
        #   row 0: ~15° (gentle hill)
        #   row 5: ~35° (steep lunar crater wall gradient)
        #
        # The terrain generator was already configured by Go2wRoughEnvCfg
        # (inherited via Go2wTurnEnvCfg → Go2wRoughEnvCfg). We only need
        # to change num_rows back to allow the slope curriculum.
        if (
            self.scene.terrain.terrain_generator is not None
            and hasattr(self.scene.terrain.terrain_generator, "num_rows")
        ):
            self.scene.terrain.terrain_generator.num_rows = 6

        # Override sub-terrains to use only pyramid slopes (no stairs/boxes).
        # Slope range: 0.26–0.61 rad (15°–35°).
        # This keeps all training tiles as slopes — pure slope adaptation.
        from isaaclab.terrains import TerrainGeneratorCfg
        from isaaclab.terrains.height_field.hf_terrains_cfg import (
            HfPyramidSlopedTerrainCfg,
        )

        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            seed=1,
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=6,     # 6 difficulty levels (15° to 35°)
            num_cols=20,    # 20 parallel tiles per level
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={
                # Upward pyramid slope — robot turns on an ascending slope
                "slope_up": HfPyramidSlopedTerrainCfg(
                    proportion=0.5,
                    slope_range=(0.26, 0.61),   # 15° to 35°
                    platform_width=2.5,         # flat spawn platform
                    border_width=0.25,
                ),
                # Inverted pyramid slope — robot turns on descending slope
                # (same angles, different geometry — tests both up and down)
                "slope_down": HfPyramidSlopedTerrainCfg(
                    proportion=0.5,
                    slope_range=(0.26, 0.61),   # 15° to 35°
                    platform_width=2.5,
                    border_width=0.25,
                ),
            },
        )

        # Re-enable terrain curriculum (advance through slope difficulties)
        # The curriculum advances when mean yaw tracking improves.
        # Re-import from rough env which defined the curriculum manager.
        from isaaclab.envs.mdp.curriculum import TerrainLevelCurriculum

        # Restore curriculum that Go2wTurnEnvCfg disabled
        # (Go2wRoughEnvCfg.post_init set it up originally)
        from isaaclab.managers import CurriculumTermCfg
        from isaaclab.envs.mdp import terrain_levels_vel

        self.curriculum.terrain_levels = CurriculumTermCfg(
            func=terrain_levels_vel,
            params={"asset_cfg": None, "std": 1},  # uses default
        )

        # ==================================================================
        # 2. REWARD ADJUSTMENTS FOR SLOPE
        # ==================================================================

        # flat_orientation_l2: REMOVE for slope
        # On a 35° slope the body tilts ~35°. Keeping any orientation penalty
        # causes -0.075/step irreducible cost just for standing on the slope.
        # Set to zero — the bad_orientation termination (at ~57°) is the
        # safety constraint; we don't need a per-step tilt penalty as well.
        if hasattr(self.rewards, "flat_orientation_l2"):
            self.rewards.flat_orientation_l2.weight = 0.0

        # ang_vel_xy_l2: VERY FORGIVING at -0.3
        # On a 35° slope, gravity causes roll/pitch angular velocity even
        # during normal weight-shifting. Any leg step on a slope creates
        # asymmetric ground reaction forces → body roll.
        # At -0.3: roll rate of 0.3 rad/s costs -0.027/step → tolerable.
        # (Flat-turn has -0.5; slope needs even more forgiveness.)
        if hasattr(self.rewards, "ang_vel_xy_l2"):
            self.rewards.ang_vel_xy_l2.weight = -0.3

        # lin_vel_z_l2: RELAXED to -0.5 (was -1.5)
        # On a slope the body naturally moves vertically as the robot steps
        # across the slope face. Vertical velocity is unavoidable during
        # cross-slope turning. At -1.5 this was harshly penalised.
        # At -0.5: vertical motion of 0.1 m/s costs only -0.005/step.
        if hasattr(self.rewards, "lin_vel_z_l2"):
            self.rewards.lin_vel_z_l2.weight = -0.5

        # is_alive: INCREASED to 0.8 on slope
        # Slopes are harder to survive on. Stronger survival signal ensures
        # the policy prioritises staying alive over optimising turn speed.
        # 0.8/step × 1000 steps = +800 for surviving — dominates fall cost.
        if hasattr(self.rewards, "is_alive"):
            self.rewards.is_alive.weight = 0.8

        # ==================================================================
        # 3. PUSH EVENTS: RE-ENABLE for slope robustness
        # ==================================================================
        # Flat-turn disabled push events. On slopes, disturbance robustness
        # is critical — lunar terrain has loose regolith and uneven surfaces.
        # Restore push events from Go2wRoughEnvCfg (already configured there).
        # This is a no-op if push_robot was None — just removes the override.
        # The rough env push config: force 0.5–1.0 N, random direction, 0.5 Hz.
        # We leave self.events.push_robot as inherited from rough env
        # (Go2wTurnEnvCfg set it to None — we need to RESTORE it).
        # Since we can't un-None it here, import and re-configure:
        from isaaclab.managers import EventTermCfg
        from isaaclab.envs.mdp.events import push_by_setting_velocity

        self.events.push_robot = EventTermCfg(
            func=push_by_setting_velocity,
            mode="interval",
            interval_range_s=(5.0, 15.0),   # push every 5-15 seconds
            params={
                "velocity_range": {
                    "x": (-0.3, 0.3),
                    "y": (-0.3, 0.3),
                    "z": (0.0, 0.0),
                },
            },
        )


@configclass
class Go2wSlopeTurnEnvCfg_PLAY(Go2wSlopeTurnEnvCfg):
    """Play/eval version of the slope-turn environment."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0

        self.observations.policy.enable_corruption = False

        self.events.base_external_force_torque = None
        self.events.push_robot = None
