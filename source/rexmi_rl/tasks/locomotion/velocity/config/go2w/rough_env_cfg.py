# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Go2W base environment configuration — velocity tracking on flat/rough terrain.

This file defines the FULL MDP (Markov Decision Process) for the Go2W:
  - Scene      : terrain, robot, sensors, lighting
  - Commands   : what velocity the robot is asked to achieve
  - Observations: what the policy "sees" every timestep
  - Actions    : what the policy outputs (wheel velocities + all leg joints)
  - Rewards    : how good/bad each action was
  - Terminations: when to end an episode early
  - Events     : randomisation at startup/reset/interval

Inheritance pattern (mirrors Isaac Lab's design)
------------------------------------------------
LocomotionVelocityRoughEnvCfg   ← imported from Isaac Lab (the general template)
    └── Go2wFlatEnvCfg          ← this file adds Go2W-specific overrides
            └── Go2wFlatEnvCfg_PLAY  ← flat_env_cfg.py reduces env count for viz

You NEVER edit the base class — only the overrides in this file and flat_env_cfg.py.

Phase 3 key choices
-------------------
* Actions    → 4 wheel velocity DOFs + 12 leg position DOFs (hip/thigh/calf)
* All leg joints → soft PD (stiffness=5) in GO2W_CFG, fully RL-controllable
* Height scan → still disabled (flat terrain)
* feet_air_time → still disabled (wheels ARE the feet; we don't reward bouncing)
* base_contact termination → triggers if the base body hits the ground
* bad_orientation → relaxed to 1.0 rad (56°) — full legs can recover more
"""

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp_utils  # for bad_orientation, is_alive

# Isaac Lab provides a ready-made base class for velocity-tracking environments.
# It sets up the scene, all MDP managers, and sim settings.  We only need to
# override the parts that differ for Go2W.
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)

# Our custom robot asset config (points to the local USD file)
from rexmi_rl.assets.go2w import GO2W_CFG


@configclass
class Go2wFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    """
    Go2W velocity-tracking environment — base configuration.

    Inherits all scene/MDP structure from LocomotionVelocityRoughEnvCfg and
    applies Go2W-specific overrides below.
    """

    def __post_init__(self):
        # Always call the parent's __post_init__ first so all default values
        # are populated before we start overriding them.
        super().__post_init__()

        # ==================================================================
        # 1. ROBOT
        # ==================================================================
        # Replace the placeholder robot with our Go2W.
        # {ENV_REGEX_NS} is a template string that Isaac Lab expands to a
        # per-environment prim path like /World/envs/env_0/Robot,
        # /World/envs/env_1/Robot, etc. for vectorised simulation.
        self.scene.robot = GO2W_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ==================================================================
        # 2. TERRAIN — flat plane
        # ==================================================================
        # Override the default rough terrain generator with a simple plane.
        # This is the fastest setup for Phase 1 and eliminates terrain-related
        # complexity so the policy can focus purely on wheel velocity control.
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None  # no procedural generation

        # ==================================================================
        # 3. HEIGHT SCANNER — not needed on flat terrain
        # ==================================================================
        # The base class includes a RayCaster height scanner that samples the
        # terrain elevation around the robot.  On a flat plane it provides no
        # useful information and wastes GPU ray-cast time.
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None  # remove from observation vector

        # ==================================================================
        # 4. ACTIONS — full 16-DOF control (Phase 3)
        # ==================================================================
        # The policy now controls ALL joints:
        #   4 × wheel velocity   (continuous, stiffness=0)
        #   4 × thigh position   (revolute, stiffness=5)
        #   4 × hip position     (revolute, stiffness=5)
        #   4 × calf position    (revolute, stiffness=5)
        #
        # JointVelocityActionCfg: wheel target velocities (rad/s)
        # JointPositionActionCfg: joint position OFFSETS from default (rad),
        #   use_default_offset=True means action=0 → hold default stance.
        from isaaclab.envs.mdp.actions import JointVelocityActionCfg  # lazy import
        from isaaclab.envs.mdp.actions import JointPositionActionCfg   # leg joints

        # Wheel velocity actions (4 DOF) — unchanged
        self.actions.joint_pos = JointVelocityActionCfg(
            asset_name="robot",
            joint_names=[".*_foot_joint"],  # 4 continuous wheel joints
            scale=10.0,                      # rad/s per unit action
            use_default_offset=False,
        )

        # Thigh position actions (4 DOF) — scale bumped 0.3 → 0.5 rad
        # With hips and calves now active, the policy has more ways to
        # stabilise, so thighs can pitch further for greater CG range.
        self.actions.thigh_pos = JointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*_thigh_joint"],  # 4 thigh joints (FL/FR/RL/RR)
            scale=0.5,                        # ±0.5 rad offset from default pose
            use_default_offset=True,          # relative to default stance
        )

        # Hip position actions (4 DOF) — Phase 3 addition
        #
        # Hips control lateral leg spread (abduction/adduction).
        # scale=0.3 rad lets the policy widen/narrow the stance for balance,
        # lean into turns, and assist lateral velocity commands.
        # Default hip pose is ±0.1 rad (slight outward splay), so a 0.3 rad
        # offset keeps the legs well within their abduction limits.
        self.actions.hip_pos = JointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*_hip_joint"],    # 4 hip joints (FL/FR/RL/RR)
            scale=0.3,                        # ±0.3 rad lateral spread offset
            use_default_offset=True,          # relative to default stance
        )

        # Calf position actions (4 DOF) — Phase 3 addition
        #
        # Calves control knee extension and foot height.  Unlocking them
        # lets the policy:
        #   • extend legs to brace against impacts
        #   • flex knees to lower the body (lower CG = more stable)
        #   • lift feet off the ground for stepping gaits
        # scale=0.5 rad: larger than hips because the calf default (-1.5 rad)
        # is far from zero and has more useful range to exploit.
        self.actions.calf_pos = JointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*_calf_joint"],   # 4 calf joints (FL/FR/RL/RR)
            scale=0.5,                        # ±0.5 rad offset from default pose
            use_default_offset=True,          # relative to default stance
        )

        # ==================================================================
        # 5. OBSERVATIONS — update prim path for height scanner reference
        # ==================================================================
        # The height scanner prim_path in the base class references Robot/base.
        # We already set scene.height_scanner = None, so this is a no-op here,
        # but kept as a reminder for when we add a height scanner later.
        # (No change needed for Phase 1.)

        # ==================================================================
        # 6. REWARDS — tune for wheeled locomotion
        # ==================================================================
        # Velocity tracking weights
        # -------------------------
        # track_lin_vel_xy_exp: reward for matching commanded forward/lateral vel.
        # Exponential form: exp(-(v_actual - v_cmd)² / std²)
        # Weight 1.5 is slightly higher than the default to emphasise forward motion.
        self.rewards.track_lin_vel_xy_exp.weight = 2.0

        # track_ang_vel_z_exp: reward for matching commanded yaw rate.
        self.rewards.track_ang_vel_z_exp.weight = 0.75

        # Stability penalties
        # -------------------
        # lin_vel_z_l2: penalise vertical bouncing (robot should glide, not hop).
        # Phase 6: reduced -2.0 → -0.3.  The climb_progress term in Go2wRoughEnvCfg
        # provides a positive counter-incentive for upward velocity during obstacle
        # crossing; keeping this too strong here would fight that signal even on
        # flat terrain.  -0.3 is still enough to damp erratic bouncing.
        self.rewards.lin_vel_z_l2.weight = -0.3

        # ang_vel_xy_l2: penalise rolling/pitching of the base.
        # Increased -0.05 → -0.5: the robot's only degree of freedom is wheel
        # speed; it CANNOT recover from a pitch by leaning (legs are locked).
        # A strong pitch penalty teaches the policy to modulate wheel speed
        # smoothly to avoid pitching momentum rather than accelerating into it.
        self.rewards.ang_vel_xy_l2.weight = -0.5

        # flat_orientation_l2: penalise deviation from level base orientation.
        # Phase 6: reduced -2.5 → -0.8.  Surmounting a step taller than the wheel
        # radius (~5 cm) REQUIRES the body to pitch 20-30° — the old weight of -2.5
        # made that pitch more costly per step than the stagnation penalty, so the
        # policy always chose to stay stuck rather than attempt to climb.  -0.8 still
        # penalises sloppy flat-ground driving while permitting climbing pitches.
        self.rewards.flat_orientation_l2.weight = -0.8

        # Energy / smoothness penalties
        # ------------------------------
        # dof_torques_l2: penalise large joint torques (reduces wheel spin energy)
        self.rewards.dof_torques_l2.weight = -1.0e-5

        # dof_acc_l2: penalise rapid joint acceleration (smooth wheel commands)
        self.rewards.dof_acc_l2.weight = -2.5e-7

        # action_rate_l2: penalise large changes in action between steps.
        # This is the single most important smoothness term — stops the policy
        # from jittering the wheel speed every step.
        self.rewards.action_rate_l2.weight = -0.01

        # Feet air-time: set to None (not just weight=0) to REMOVE the term entirely.
        #
        # IMPORTANT: Isaac Lab resolves the sensor's body_names regex even for
        # zero-weight terms during environment construction.  The base class uses
        # the pattern ".*FOOT" (uppercase) but our robot has "FL_foot" etc.
        # (lowercase).  Setting weight=0.0 would still trigger a body-name
        # resolution error at startup.  Setting = None skips resolution completely.
        self.rewards.feet_air_time = None

        # Undesired contacts: re-enabled for Phase 3 with correct body names.
        #
        # The robot was discovered to "spider-walk" — land on its calf/thigh/hip
        # links and walk with legs while wheels spin freely in the air.
        # This penalty fires whenever a leg body exerts force > threshold N on the
        # ground, making leg-based locomotion cost -1.0/step per contact.
        # The policy will learn to keep leg links off the ground and use wheels.
        #
        # Note: base class uses ".*THIGH" (uppercase) — our robot has lowercase
        # names (FL_thigh, FR_calf, etc.) so we must override with the correct regex.
        # SceneEntityCfg("contact_forces", ...) references the ContactSensorCfg
        # that the base class already sets up for all rigid bodies.
        from isaaclab.managers import SceneEntityCfg
        self.rewards.undesired_contacts = RewTerm(
            func=mdp_utils.undesired_contacts,
            weight=-1.0,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=[".*_hip", ".*_thigh", ".*_calf"],
                ),
                "threshold": 1.0,  # N — ignore micro-forces from brush contacts
            },
        )

        # Joint position limits penalty: re-enabled for Phase 3.
        # All 12 leg joints are now RL-controllable, so we need a soft barrier
        # to prevent the policy from driving joints into hard URDF limits.
        # Weight -0.1 is mild — enough to signal "don't slam into the limit"
        # without dominating the velocity tracking reward.
        # (wheels are continuous with no limits, so this only fires on leg joints)
        self.rewards.dof_pos_limits = RewTerm(
            func=mdp_utils.joint_pos_limits,
            weight=-0.1,
        )

        # Leg joint deviation penalty: prevent excessive leg spreading / loophole gaits.
        #
        # The policy was observed to splay hips and extend calves into extreme
        # positions to find wide-stance stable configurations that satisfy the
        # velocity reward without clean locomotion.
        # joint_deviation_l1 computes the L1 norm of (joint_pos - joint_default)
        # for the specified joints, penalising any deviation from the default stance.
        # Phase 6: reduced -0.2 → -0.05.  Climbing a step requires the front calves
        # to extend significantly to lift the wheel above the step face.  The old
        # weight (-0.2) made that extension 4× more costly than stagnation, suppressing
        # the very leg motion needed for climbing.  -0.05 retains a soft bias toward
        # the default stance while permitting large calf deflections during climbing.
        from isaaclab.managers import SceneEntityCfg as _SECfg  # already imported above
        self.rewards.leg_deviation = RewTerm(
            func=mdp_utils.joint_deviation_l1,
            weight=-0.05,
            params={
                "asset_cfg": _SECfg(
                    "robot",
                    joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                )
            },
        )

        # is_alive: positive reward every step the robot is NOT terminated.
        #
        # WHY THIS IS CRITICAL
        # Without a positive upright reward, all reward signals are penalties.
        # When the robot lies flat it gets constant -flat_orientation, but so
        # does an upright robot that simply isn't moving — there's no gradient
        # to differentiate "trying and failing" from "doing nothing".
        # is_alive() returns 1.0 when the episode is running, 0.0 at termination.
        # Combined with the bad_orientation termination below, this creates a
        # clear incentive: stay upright = keep getting +1.0/step.
        # is_alive weight set LOW (0.2) to avoid reward-gaming.
        # At weight=1.0 the robot found it more profitable to stand still
        # (collecting +~930 alive/episode) than to move and risk tipping.
        # At weight=0.2, alive contributes ~186/episode while velocity
        # tracking (weight=2.0) can contribute up to ~1860 — clear incentive
        # to actually drive toward the command.
        self.rewards.is_alive = RewTerm(func=mdp_utils.is_alive, weight=0.2)

        # ==================================================================
        # 7. EVENTS — domain randomisation at startup / reset
        # ==================================================================
        # add_base_mass: randomly add/remove mass from the base link.
        # Range (-1, 3) kg covers payload variation.
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"

        # base_external_force_torque: apply random forces at reset (set to zero
        # for Phase 1 — we don't push the robot yet).
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"

        # reset_robot_joints: reset all joints to exactly the default position
        # (scale=1.0 means no noise around the default — clean starts).
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)

        # reset_base: reset body pose with some XY and yaw randomisation so
        # the robot doesn't always start facing the same direction.
        # Velocity range is all zeros — robot starts stationary.
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # Disable centre-of-mass randomisation for now.
        self.events.base_com = None

        # push_robot: re-enabled for Phase 3.
        # With all leg DOFs unlocked the robot has enough actuation authority
        # to resist external perturbations, so we add random shoves during
        # training to build robustness.  The base class default applies a
        # random impulse every ~10 s.
        # (self.events.push_robot is already configured in the base class;
        # we just un-None it by not overriding it here)

        # ==================================================================
        # 8. VELOCITY COMMANDS — restrict to what our wheels can achieve
        # ==================================================================
        # The base class commands velocities of ±1 m/s, but our wheels have:
        #   max speed = scale (10 rad/s) × wheel_radius (~0.05 m) ≈ 0.5 m/s
        # Asking for 2× the physical maximum means the velocity tracking reward
        # is always near zero regardless of what the policy does → no gradient.
        #
        # Restrict commands to ±0.5 m/s so perfect tracking is achievable and
        # the reward signal is informative from the first iteration.
        # Phase 3: increased to ±0.5 m/s — with all leg DOFs unlocked the
        # policy can dynamically shift the CG to stay balanced at higher speed.
        # Lateral velocity (lin_vel_y) also increased: hip abduction + calf
        # extension let the robot lean and step sideways.
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        # Yaw expanded to ±1.0 rad/s: full leg DOFs mean the robot can use
        # differential leg loading (not just differential wheel speed) for turning.
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # ==================================================================
        # 9. TERMINATIONS
        # ==================================================================
        # Terminate if the base link touches the ground (robot fell over).
        # body_names="base" limits the contact check to just the main body,
        # not the legs/wheels which are expected to touch the ground.
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"

        # bad_orientation: terminate if the robot tips past limit_angle radians.
        #
        # Relaxed 0.8 → 1.0 rad (57°) for Phase 3:
        # With all leg DOFs active the robot can actively recover from larger
        # tilts (e.g., by widening its stance or pushing off with a calf).
        # 1.0 rad gives enough headroom for dynamic recovery attempts before
        # we declare the episode lost, while still catching genuine falls
        # (>57° from vertical = robot is going down regardless of leg action).
        self.terminations.bad_orientation = DoneTerm(
            func=mdp_utils.bad_orientation,
            params={"limit_angle": 1.0},  # 1.0 rad ≈ 57° from vertical
        )

        # ==================================================================
        # 10. CURRICULUM — disabled for flat terrain
        # ==================================================================
        # Terrain-level curriculum increases difficulty over training.
        # Not applicable for a single flat plane.
        self.curriculum.terrain_levels = None

        # ==================================================================
        # 11. SIMULATION TIMING
        # ==================================================================
        # These are inherited from the base class but restated here for clarity:
        #   sim.dt = 0.005 s  → 200 Hz physics
        #   decimation = 4    → policy runs at 200/4 = 50 Hz
        # (These values come from LocomotionVelocityRoughEnvCfg.__post_init__)


# ==============================================================================
# Phase 4 — Rough terrain with full height-scan perception
# ==============================================================================

@configclass
class Go2wRoughEnvCfg(Go2wFlatEnvCfg):
    """
    Go2W velocity-tracking environment — rough terrain with height scanner.

    Inherits all Go2W-specific overrides from Go2wFlatEnvCfg (robot, actions,
    rewards, terminations) and then RE-ENABLES the components that were disabled
    for flat terrain:
      • terrain generator  — stairs, slopes, boxes, rough heightfields
      • RayCaster height scanner — 1.6 m × 1.0 m grid at 10 cm resolution
      • height_scan observation  — 160 dims added to the policy input
      • terrain curriculum       — robot advances to harder tiles as speed improves

    Train with:
        ./run.sh scripts/train.py --task RexmiRl-Go2w-Velocity-Rough-v0 --headless

    NOTE: The obs space is different from the flat policy (~208 vs ~48 dims) so
    this must be trained from scratch with a larger network (see
    Go2wRoughPPORunnerCfg in rsl_rl_ppo_cfg.py).
    """

    def __post_init__(self):
        # Apply all Go2W-specific flat env overrides first (robot, actions, rewards…)
        super().__post_init__()

        # ==================================================================
        # A. TERRAIN — switch from flat plane to procedural generator
        # ==================================================================
        # Re-enable the terrain generator that was disabled in Go2wFlatEnvCfg.
        # We use the same 5 sub-terrain types as the Isaac Lab default rough env:
        #   pyramid_stairs, pyramid_stairs_inv, boxes, random_rough, hf_pyramid_slope
        # All have equal proportion (0.2 each = 20% of tiles each).
        #
        # TerrainGeneratorCfg builds a num_rows × num_cols grid of tiles:
        #   num_rows = 10 curriculum difficulty levels (row 0 = easiest/flat)
        #   num_cols = 20 tiles per difficulty level (parallelism)
        #   size     = 8 m × 8 m per tile
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

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            seed=0,
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=10,    # 10 difficulty levels for curriculum
            num_cols=20,    # 20 parallel tiles per level
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={
                # Upward pyramid stairs — requires stepping up
                "pyramid_stairs": MeshPyramidStairsTerrainCfg(
                    proportion=0.2,
                    step_height_range=(0.05, 0.23),
                    step_width=0.3,
                    platform_width=3.0,
                    border_width=1.0,
                    holes=False,
                ),
                # Inverted pyramid stairs — requires stepping down
                "pyramid_stairs_inv": MeshInvertedPyramidStairsTerrainCfg(
                    proportion=0.2,
                    step_height_range=(0.05, 0.23),
                    step_width=0.3,
                    platform_width=3.0,
                    border_width=1.0,
                    holes=False,
                ),
                # Random height grid — uneven stepping stones / cobblestones
                "boxes": MeshRandomGridTerrainCfg(
                    proportion=0.2,
                    grid_width=0.45,
                    grid_height_range=(0.05, 0.2),
                    platform_width=2.0,
                ),
                # Random rough heightfield — bumpy / gravel-like surface
                "random_rough": HfRandomUniformTerrainCfg(
                    proportion=0.2,
                    noise_range=(0.02, 0.10),
                    noise_step=0.02,
                    border_width=0.25,
                ),
                # Pyramid slope — ramps requiring active balance
                "hf_pyramid_slope": HfPyramidSlopedTerrainCfg(
                    proportion=0.2,
                    slope_range=(0.0, 0.4),
                    platform_width=2.0,
                    border_width=0.25,
                ),
            },
        )

        # ==================================================================
        # B. HEIGHT SCANNER — RayCaster reading terrain elevation
        # ==================================================================
        # Restore the RayCasterCfg that Go2wFlatEnvCfg set to None.
        # This casts 160 rays (1.6 m × 1.0 m grid at 0.1 m resolution) down
        # from 20 m above the robot base and records the height of the terrain
        # below each ray.  The result is a 160-element observation vector that
        # lets the policy "see" upcoming obstacles before they hit the wheels.
        #
        # attach_yaw_only=True: the scan rotates with the robot's yaw but not
        # pitch/roll — the grid stays level as the robot tilts.
        # mesh_prim_paths=["/World/ground"]: only cast against the terrain mesh,
        # not against the robot's own geometry.
        from isaaclab.sensors import RayCasterCfg, patterns

        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",   # attach_yaw_only=True is deprecated in this version
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )

        # ==================================================================
        # C. HEIGHT SCAN OBSERVATION — add 160 height values to policy input
        # ==================================================================
        # The base class already has height_scan defined; Go2wFlatEnvCfg set it
        # to None.  We restore it here with the same spec as the base class:
        #   func       — height_scan reads the RayCaster sensor buffer
        #   noise      — ±0.1 m uniform noise for sim-to-real robustness
        #   clip       — clamp to ±1 m (prevents outliers from skewing the net)
        from isaaclab.managers import ObservationTermCfg as ObsTerm
        from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
        from isaaclab.managers import SceneEntityCfg as _SECfg

        self.observations.policy.height_scan = ObsTerm(
            func=mdp_utils.height_scan,
            params={"sensor_cfg": _SECfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        # ==================================================================
        # E. STAGNATION PENALTY — teach the robot to escape when stuck
        # ==================================================================
        # stagnation_penalty fires on every step where:
        #   • the commanded forward velocity is > 0.1 m/s  (asked to move)
        #   • the actual forward velocity is < 0.05 m/s    (barely moving)
        #
        # Why this matters for rough terrain:
        #   On flat ground the robot never gets stuck so this never fires.
        #   On discrete steps / box edges a wheel can catch on a face and
        #   spin uselessly with zero forward progress.  Without any gradient
        #   signal the policy has no incentive to try something different.
        #   With this penalty, staying stuck for 30 steps costs -45 reward —
        #   comparable to ~22 steps of failed velocity tracking.
        #
        # Phase 6: weight increased -0.5 → -1.5.  Phase 5 eval showed that
        # -0.5 was sufficient to solve the 12 cm cliff (from 0.45 → 0.82
        # tracking) but not enough to incentivise recovery at 15-23 cm.
        # With -1.5, 10 stuck steps = -15 reward, making "try anything else"
        # (yaw, reverse, leg reposition) strictly better than staying stuck.
        from rexmi_rl.tasks.locomotion.velocity.mdp import (
            stagnation_penalty as _stagnation_penalty,
            climb_progress as _climb_progress,
        )

        self.rewards.stagnation = RewTerm(
            func=_stagnation_penalty,
            weight=-1.5,
            params={"threshold": 0.05},
        )

        # ==================================================================
        # F. CLIMB PROGRESS — reward upward velocity during forward command
        # ==================================================================
        # climb_progress fires whenever:
        #   • the commanded forward velocity is > 0.1 m/s  (asked to move)
        #   • the world-frame vertical velocity vz > 0      (body is rising)
        #
        # Why this is needed:
        #   Any step taller than the wheel radius (5 cm) cannot be surmounted
        #   by rolling alone — the wheel face contacts the step's vertical wall.
        #   The ONLY physical paths over such a wall are:
        #     1. Pitch the body forward + extend front calves to place the wheel
        #        ABOVE the step edge, then drive forward onto the step surface.
        #     2. Build enough momentum that contact geometry carries the wheel up.
        #   Both paths produce positive vz during the crossing.
        #
        #   The existing lin_vel_z_l2 (-0.3) still penalises ALL vertical velocity,
        #   but with weight 1.5 here the net reward for climbing at 0.3 m/s is:
        #     +1.5 × 0.3  (climb_progress)  = +0.45/step
        #     -0.3 × 0.09 (lin_vel_z_l2)    = -0.027/step
        #     net: +0.42/step — climbing is now profitable
        #
        # Weight = +1.5: strong enough to counteract flat_orientation_l2
        # (-0.8 × pitch²) during a 30° climbing pitch, giving a near-zero
        # net penalty for the orientation cost of climbing.  Zero on flat
        # ground (vz ≈ 0) so it doesn't distort flat driving behaviour.
        self.rewards.climb_progress = RewTerm(
            func=_climb_progress,
            weight=1.5,
        )

        # ==================================================================
        # D. TERRAIN CURRICULUM — advance robot to harder tiles as it improves
        # ==================================================================
        # terrain_levels_vel is defined in isaaclab_tasks (the velocity
        # locomotion task MDP module), NOT in isaaclab.envs.mdp.
        # It measures the robot's actual forward speed vs commanded speed and
        # moves the robot to a harder terrain tile (higher row index) when it
        # consistently achieves the commanded velocity, or to an easier tile
        # when it fails.  This automatic curriculum is why rough terrain
        # training converges — the robot starts on flat/easy tiles and
        # gradually advances to stairs and steep slopes.
        from isaaclab.managers import CurriculumTermCfg as CurrTerm
        from isaaclab_tasks.manager_based.locomotion.velocity.mdp.curriculums import (
            terrain_levels_vel as _terrain_levels_vel,
        )

        self.curriculum.terrain_levels = CurrTerm(
            func=_terrain_levels_vel,
        )


@configclass
class Go2wRoughEnvCfg_PLAY(Go2wRoughEnvCfg):
    """
    Play-mode variant for rough terrain: fewer envs, no noise, no pushes.
    """

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 200
        self.scene.env_spacing = 8.0   # match the 8 m × 8 m rough terrain tile size
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class Go2wFlatEnvCfg_PLAY(Go2wFlatEnvCfg):
    """
    Play-mode variant: fewer environments, no randomisation, no external forces.

    Use this config when running play.py to visualise a trained policy.
    It spawns 50 robots in a small grid so you can watch them all at once without
    the GPU memory cost of 4096 environments.
    """

    def __post_init__(self):
        super().__post_init__()

        # Fewer environments for visualisation (50 vs 4096 for training)
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5  # metres between each robot in the grid

        # Disable sensor noise so observations reflect the true state.
        # During training, noise helps the policy generalise; during play we
        # want to see the policy's true performance without noise artefacts.
        self.observations.policy.enable_corruption = False

        # Remove all force/push events — clean playback only.
        self.events.base_external_force_torque = None
        self.events.push_robot = None
