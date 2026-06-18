# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Go2W fast flat-terrain environment configuration.

WHY A SEPARATE CONFIG?
-----------------------
Go2wFlatEnvCfg (in rough_env_cfg.py) is the BASE CLASS for Go2wRoughEnvCfg.
Changing it directly would silently propagate to the rough env via inheritance.

Go2wFastFlatEnvCfg is a SIBLING — it inherits from Go2wFlatEnvCfg and overrides
only the speed-related settings, leaving the rough env completely untouched.

Inheritance
-----------
Go2wFlatEnvCfg                 ← robot, actions (scale=10), rewards, terminations
    └── Go2wFastFlatEnvCfg     ← this file — wheel scale=40, forward up to 2 m/s
            └── Go2wFastFlatEnvCfg_PLAY

WHY scale=40?
--------------
target max speed      = 2.0 m/s
wheel radius          = 0.05 m  (URDF confirmed)
required wheel speed  = 2.0 / 0.05 = 40 rad/s
policy output range   ≈ ±1 → scale=40 gives ±40 rad/s ≡ ±2.0 m/s

IMPORTANT: This changes the action-to-speed mapping vs. the base flat env
(which uses scale=10).  Old flat checkpoints are NOT compatible — must train
from scratch.  Use:
    python scripts/train.py --task RexmiRl-Go2w-Velocity-FastFlat-v0 --headless

ASYMMETRIC COMMAND RANGE:
    Forward:   0 – 2.0 m/s  (the high-speed target)
    Backward:  0 – 0.5 m/s  (unchanged — no high-speed reverse needed)
    Lateral:  ±0.5 m/s      (unchanged)
    Yaw:      ±1.0 rad/s    (unchanged)

Achieved by setting lin_vel_x = (-0.5, 2.0) — uniform sampling means ~80% of
commands are forward at various speeds, 20% are backward at ≤0.5 m/s.

HIGH-SPEED REWARD TUNING:
    action_rate_l2 : -0.01 → -0.02  (4× wheel speed → smoother commands needed)
    ang_vel_xy_l2  : -0.5  → -0.8   (tighter pitch/roll at 2 m/s prevents tip-over)
"""

from isaaclab.utils import configclass

from rexmi_rl.tasks.locomotion.velocity.config.go2w.rough_env_cfg import Go2wFlatEnvCfg


@configclass
class Go2wFastFlatEnvCfg(Go2wFlatEnvCfg):
    """
    Go2W high-speed flat-terrain environment — forward speed up to 2 m/s.

    Inherits everything from Go2wFlatEnvCfg (robot, full 16-DOF actions,
    height-scanner-free observations, flat terrain, all rewards/terminations)
    and overrides only the speed-relevant settings.

    Train with:
        python scripts/train.py --task RexmiRl-Go2w-Velocity-FastFlat-v0 --headless
    """

    def __post_init__(self):
        # Apply all base flat env overrides first (robot, actions, rewards, …)
        super().__post_init__()

        # ==================================================================
        # 1. WHEEL SPEED SCALE — 10 → 40 rad/s for 2 m/s max ground speed
        # ==================================================================
        # The parent creates self.actions.joint_pos as a JointVelocityActionCfg
        # with scale=10.0.  We override scale here after super().__post_init__()
        # so the rest of the action spec (joint_names, use_default_offset) is
        # inherited unchanged.
        #
        # Physical chain:
        #   policy output ≈ ±1.0  × scale=40  =  ±40 rad/s wheel speed
        #   ground speed   = ω × r = 40 × 0.05  =  ±2.0 m/s
        self.actions.joint_pos.scale = 40.0

        # ==================================================================
        # 2. VELOCITY COMMANDS — asymmetric forward range
        # ==================================================================
        # Forward up to 2 m/s; backward, lateral, yaw unchanged.
        # Uniform sampling over (-0.5, 2.0) means ~80% of episodes have a
        # forward command — the policy gets sufficient training at all forward
        # speeds from slow crawl to 2 m/s sprint.
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 2.0)
        # lin_vel_y = (-0.5, 0.5) and ang_vel_z = (-1.0, 1.0) unchanged

        # ==================================================================
        # 3. REWARD TUNING FOR HIGH-SPEED STABILITY
        # ==================================================================
        # action_rate_l2: -0.01 → -0.02
        # At 4× wheel scale, a given action-delta produces 4× more jerk.
        # Doubling this penalty keeps wheel commands smooth at sprint speeds.
        self.rewards.action_rate_l2.weight = -0.02

        # ang_vel_xy_l2: -0.5 → -0.8
        # At 2 m/s, aerodynamic pitch-up and bump impacts are more destabilising.
        # A tighter penalty teaches the policy to pre-emptively modulate wheel speed
        # to damp pitch before it builds up, rather than reacting after the fact.
        self.rewards.ang_vel_xy_l2.weight = -0.8


@configclass
class Go2wFastFlatEnvCfg_PLAY(Go2wFastFlatEnvCfg):
    """
    Play-mode variant: 50 robots, no noise, no random forces.

    Use this config when running play.py to visualise the fast flat policy.
    """

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # Disable sensor noise — clean playback to see true policy behaviour.
        self.observations.policy.enable_corruption = False

        # Remove perturbations — isolate terrain tracking from external forces.
        self.events.base_external_force_torque = None
        self.events.push_robot = None
