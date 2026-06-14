# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Go2W base environment configuration — velocity tracking on flat/rough terrain.

This file defines the FULL MDP (Markov Decision Process) for the Go2W:
  - Scene      : terrain, robot, sensors, lighting
  - Commands   : what velocity the robot is asked to achieve
  - Observations: what the policy "sees" every timestep
  - Actions    : what the policy outputs (wheel velocities)
  - Rewards    : how good/bad each action was
  - Terminations: when to end an episode early
  - Events     : randomisation at startup/reset/interval

Inheritance pattern (mirrors Isaac Lab's design)
------------------------------------------------
LocomotionVelocityRoughEnvCfg   ← imported from Isaac Lab (the general template)
    └── Go2wFlatEnvCfg          ← this file adds Go2W-specific overrides
            └── Go2wFlatEnvCfg_PLAY  ← flat_env_cfg.py reduces env count for viz

You NEVER edit the base class — only the overrides in this file and flat_env_cfg.py.

Phase 1 key choices
-------------------
* Actions    → JointVelocityActionCfg on ONLY the 4 wheel joints
* Leg joints → held by the PD controller defined in GO2W_CFG (no policy output)
* Height scan → disabled (flat terrain, saves computation)
* feet_air_time reward → disabled (not relevant for wheels)
* base_contact termination → triggers if the body hits the ground
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
        # 4. ACTIONS — wheels only (Phase 1)
        # ==================================================================
        # Replace the base class's JointPositionActionCfg (which targets ALL
        # joints) with a JointVelocityActionCfg that ONLY targets the 4 wheel
        # joints.  The leg joints are not in this expression, so they remain
        # under PD control at their default stance positions.
        #
        # JointVelocityActionCfg parameters:
        #   asset_name   — must match the key in the scene ("robot")
        #   joint_names  — regex matched against joint names in the USD
        #   scale        — output multiplier: action ∈ [-1,1] × scale = target vel
        #                  scale=10 → max wheel speed ±10 rad/s ≈ ±0.5 m/s for
        #                  a wheel of radius ~0.05 m (typical Go2W wheel)
        #   use_default_offset — if True, action is RELATIVE to the default vel
        #                        (which is 0 for wheels, so this has no effect)
        from isaaclab.envs.mdp.actions import JointVelocityActionCfg  # lazy import
        from isaaclab.envs.mdp.actions import JointPositionActionCfg   # for thighs

        # Wheel velocity actions (4 DOF) — unchanged from Phase 1
        self.actions.joint_pos = JointVelocityActionCfg(
            asset_name="robot",
            joint_names=[".*_foot_joint"],  # only the 4 continuous wheel joints
            scale=10.0,                      # rad/s per unit action
            use_default_offset=False,
        )

        # Thigh position actions (4 DOF) — Phase 2 addition
        #
        # The policy can now pitch the thighs ±0.3 rad (≈±17°) from default.
        # use_default_offset=True means action=0 → no change from default stance.
        # action=+0.3 → thighs tilt 0.3 rad from default (shifts CG forward).
        #
        # Scale=0.3 rad is intentionally small:
        #   - Large thigh movements destabilise the robot (feet leave the ground)
        #   - 0.3 rad is enough to noticeably shift the CG without collapse
        #   - In Phase 3 we can expand this range once the policy is stable
        self.actions.thigh_pos = JointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*_thigh_joint"],  # 4 thigh joints (FL/FR/RL/RR)
            scale=0.3,                        # ±0.3 rad offset from default pose
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
        # lin_vel_z_l2: penalise vertical bouncing (robot should glide, not hop)
        self.rewards.lin_vel_z_l2.weight = -2.0

        # ang_vel_xy_l2: penalise rolling/pitching of the base.
        # Increased -0.05 → -0.5: the robot's only degree of freedom is wheel
        # speed; it CANNOT recover from a pitch by leaning (legs are locked).
        # A strong pitch penalty teaches the policy to modulate wheel speed
        # smoothly to avoid pitching momentum rather than accelerating into it.
        self.rewards.ang_vel_xy_l2.weight = -0.5

        # flat_orientation_l2: penalise deviation from level base orientation.
        # Weight 2.5 ensures the robot keeps its belly parallel to the ground.
        self.rewards.flat_orientation_l2.weight = -2.5

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

        # Undesired contacts: remove entirely.
        # Base class penalises ".*THIGH" contacts; also uses uppercase pattern.
        self.rewards.undesired_contacts = None

        # Joint position limits penalty: remove entirely for Phase 1.
        # (Legs are held by PD; wheels are continuous with no hard limits.)
        self.rewards.dof_pos_limits = None

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

        # push_robot: randomly shove the robot during training.
        # Disabled in Phase 1 for simplicity; re-enable in Phase 2.
        self.events.push_robot = None

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
        # Command speed reduced ±0.5 → ±0.3 m/s:
        # At 0.5 m/s (max wheel speed), steady-state wheel torque is high enough
        # to pitch the robot forward even with lower damping.  At 0.3 m/s the
        # wheel only needs 60% of the torque, keeping pitch well within limits.
        self.commands.base_velocity.ranges.lin_vel_x = (-0.3, 0.3)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        # Yaw also reduced: differential wheel speed for turning is tricky and
        # competes with forward motion.  Restrict to ±0.5 rad/s so the policy
        # can focus on straight-line tracking first.
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)

        # ==================================================================
        # 9. TERMINATIONS
        # ==================================================================
        # Terminate if the base link touches the ground (robot fell over).
        # body_names="base" limits the contact check to just the main body,
        # not the legs/wheels which are expected to touch the ground.
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"

        # bad_orientation: terminate if the robot tips past limit_angle radians.
        #
        # WHY THIS IS THE MOST CRITICAL FIX
        # The base_contact termination only fires if the base mesh physically
        # hits the ground.  For the Go2W, the legs/hips make contact first,
        # so base_contact NEVER fires (confirmed: 0.0 in training metrics).
        # The robot lies tilted for the full 1000-step episode = wasted time.
        #
        # bad_orientation terminates as soon as the gravity-projection angle
        # exceeds limit_angle.  0.5 rad ≈ 28° is tight enough to catch early
        # tipping but loose enough not to terminate on normal stance wobble.
        #
        # With this fix:
        #   - Fallen episodes reset in <0.5 s instead of running 20 s to timeout
        #   - The policy collects ~40× more "upright" transitions per wall-clock time
        #   - is_alive reward gives clear gradient: stay up = +1/step
        # Relaxed 0.5 → 0.8 rad (46°):
        # A wheeled robot with locked legs physically pitches forward slightly
        # when driving (like a person leaning on a scooter).  0.5 rad was too
        # tight — even at low speed the robot's steady-state forward pitch
        # (~33°) hit the limit.  0.8 rad gives enough headroom for the robot
        # to maintain forward motion without immediately terminating, while
        # still catching catastrophic falls (>46° = clearly tipped over).
        self.terminations.bad_orientation = DoneTerm(
            func=mdp_utils.bad_orientation,
            params={"limit_angle": 0.8},  # 0.8 rad ≈ 46° from vertical
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
