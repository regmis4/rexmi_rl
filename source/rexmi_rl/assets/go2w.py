# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
ArticulationCfg for the Unitree Go2W wheeled-legged quadruped.

What is an ArticulationCfg?
-----------------------------
Isaac Lab uses a data-class called ArticulationCfg to fully describe a robot:
  1. spawn       — USD file path + physics properties (gravity, damping, collision)
  2. init_state  — where the robot spawns and what joint values it starts with
  3. actuators   — which joints are controlled and HOW (motor model, gains)

The Go2W has 16 controllable joints:
  - 12 revolute "leg" joints: hip / thigh / calf  (3 per leg × 4 legs)
  - 4  continuous "wheel" joints: FL/FR/RL/RR_foot_joint

Phase 1 strategy (wheels-only)
--------------------------------
We split the actuators into TWO groups:

  "leg_joints"   → ImplicitActuatorCfg with high stiffness + damping
                   This acts like a PD position controller holding the legs
                   in their default standing stance.  The RL policy does NOT
                   output actions for these joints in Phase 1.

  "wheel_joints" → ImplicitActuatorCfg with stiffness=0 (velocity mode).
                   stiffness=0 means the motor applies ZERO restoring torque
                   toward a position target — it only uses the damping term as
                   a velocity-proportional torque.  Isaac Lab's
                   JointVelocityActionCfg drives these joints by setting the
                   desired velocity, and the damping converts that into torque.

This lets the robot roll around while keeping its legs rigid, which is the
simplest possible locomotion task and trains very quickly (~300 iterations).

USD path
---------
We reference the locally converted USD that lives in the repo at:
  assets/robots/go2w/urdf/go2w/go2w.usd
The path is computed relative to this Python file's location so it works
regardless of where the repo is cloned.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# ---------------------------------------------------------------------------
# Resolve the absolute path to the USD file.
# ---------------------------------------------------------------------------
# __file__  → .../source/rexmi_rl/assets/go2w.py
# Go up 3 levels to reach the repo root, then navigate to the USD.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_GO2W_USD = os.path.join(_REPO_ROOT, "assets", "robots", "go2w", "urdf", "go2w", "go2w.usd")

# ---------------------------------------------------------------------------
# GO2W_CFG — the main robot configuration object
# ---------------------------------------------------------------------------
GO2W_CFG = ArticulationCfg(
    # ------------------------------------------------------------------
    # spawn — tells Isaac Lab how to load the robot into the scene
    # ------------------------------------------------------------------
    spawn=sim_utils.UsdFileCfg(
        usd_path=_GO2W_USD,

        # activate_contact_sensors=True adds ContactReporter API to every
        # rigid body so ContactSensorCfg can read foot/body contacts.
        # Required for the "base contact → termination" and "undesired contact
        # → penalty" reward terms.
        activate_contact_sensors=True,

        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,         # gravity acts on the robot (duh)
            retain_accelerations=False,    # don't cache accelerations between steps
            linear_damping=0.0,            # no artificial linear drag on bodies
            angular_damping=0.0,           # no artificial angular drag on bodies
            max_linear_velocity=1000.0,    # safety cap (m/s) — effectively unlimited
            max_angular_velocity=1000.0,   # safety cap (rad/s)
            max_depenetration_velocity=1.0,# how fast overlapping bodies are pushed apart
        ),

        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # Self-collisions between robot links cause numerical instability
            # with minimal physical benefit for a quadruped; disable them.
            enabled_self_collisions=False,

            # Position iteration count: how many solver passes are used to
            # resolve joint position constraints per physics step.
            # 4 is standard for quadrupeds; higher = more accurate but slower.
            solver_position_iteration_count=4,

            # Velocity iterations: additional passes for velocity constraints.
            # 0 is fine for most quadruped joints.
            solver_velocity_iteration_count=0,
        ),
    ),

    # ------------------------------------------------------------------
    # init_state — robot pose and joint values at episode reset
    # ------------------------------------------------------------------
    # The Go2W standing stance has:
    #   hip joints  ≈ 0 (legs straight out to the side)
    #   thigh joints ≈ 0.8 rad (tilted forward/back so the body is low)
    #   calf joints  ≈ -1.5 rad (bent backward to keep feet near ground)
    #   wheel joints = 0 velocity (stationary at reset)
    #
    # pos z=0.43 — the approximate height of the base above the ground plane
    # when the robot is in its default stance.  This must match what you
    # verified in Isaac Sim after importing the USD.
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.43),
        joint_pos={
            # Hip joints: slight outward splay on left legs, inward on right
            # (mirrors the real robot's natural stance)
            ".*L_hip_joint": 0.1,   # FL_hip, RL_hip  (left side, positive = abduction)
            ".*R_hip_joint": -0.1,  # FR_hip, RR_hip  (right side, negative = abduction)

            # Thigh joints: front legs pitched slightly forward
            "F[L,R]_thigh_joint": 0.8,   # front legs
            "R[L,R]_thigh_joint": 1.0,   # rear legs (slightly more bent for balance)

            # Calf joints: bent back to ~85° so feet are near the ground
            ".*_calf_joint": -1.5,

            # Wheel joints: start at rest (zero velocity)
            ".*_foot_joint": 0.0,
        },
        # All joint velocities start at zero
        joint_vel={".*": 0.0},
    ),

    # soft_joint_pos_limit_factor: multiply joint URDF limits by this factor
    # to create a "soft" inner limit.  The reward term `joint_pos_limits`
    # penalises the policy when joints exceed the soft limit, providing a
    # smooth gradient before hard clamping kicks in.
    soft_joint_pos_limit_factor=0.9,

    # ------------------------------------------------------------------
    # actuators — motor models for each joint group
    # ------------------------------------------------------------------
    actuators={
        # ---------------------------------------------------------------
        # Phase 2: leg joints split into two groups
        # ---------------------------------------------------------------
        #
        # hip_calf_joints: LOCKED — held rigidly at default stance.
        # The hip joints determine lateral leg spread; the calf joints
        # control foot height.  We keep both rigid for now so the only
        # active leg DOF is the thigh (pitch direction).
        "hip_calf_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_calf_joint"],
            effort_limit=23.5,
            velocity_limit=30.0,
            stiffness=25.0,   # Kp: stiff PD — locks at default position
            damping=0.5,
            friction=0.0,
        ),

        # thigh_joints: SOFT PD — RL policy can move these.
        #
        # WHY stiffness=5 (not 25)?
        # With stiffness=5, the PD still provides a gentle pull back toward
        # the default stance (restoring force), but the RL policy can easily
        # override it.  The JointPositionActionCfg will output a position
        # offset Δθ; PhysX computes:
        #   τ = stiffness × (θ_default + Δθ − θ_actual) + damping × (0 − ω_actual)
        # At stiffness=5, a 0.3 rad offset costs only 1.5 N·m — well within
        # the 23.5 N·m effort limit.  At stiffness=25, the same offset would
        # cost 7.5 N·m and the legs wouldn't move much.
        #
        # WHY thighs specifically?
        # The thigh pitch is the CG-shifting joint.  Tilting the thighs back
        # moves the robot's CG forward, counteracting the nose-down pitch torque
        # from wheel driving — exactly what a person does when leaning into a scooter.
        "thigh_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*_thigh_joint"],
            effort_limit=23.5,
            velocity_limit=30.0,
            stiffness=5.0,    # Kp: soft PD — RL can override default pose
            damping=0.5,      # Kd: velocity damping to avoid oscillations
            friction=0.0,
        ),

        # ---------------------------------------------------------------
        # Wheel joints (4 continuous: FL/FR/RL/RR_foot_joint)
        # ---------------------------------------------------------------
        # stiffness=0 → pure velocity control (no position restoring force)
        # damping=5.0 → torque = damping × (desired_vel − actual_vel)
        #
        # When JointVelocityActionCfg writes a target velocity ω_des, PhysX
        # computes: τ = damping × (ω_des − ω_actual) and applies it to the
        # wheel.  This is a simple first-order velocity controller — good
        # enough for flat terrain.  Higher damping = stiffer velocity tracking
        # but more torque and heat.
        #
        # effort_limit matches the URDF continuous joint effort (23.7 N·m).
        "wheel_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*_foot_joint"],
            effort_limit=23.7,
            velocity_limit=30.1,
            stiffness=0.0,    # ← KEY: no position stiffness → velocity mode
            # damping reduced 5.0 → 2.0:
            # At damping=5.0, holding 0.3 m/s (≈6 rad/s) requires
            # τ = 5.0 × 6 = 30 N·m per wheel → 120 N·m total pitch moment,
            # which is enough to tip the robot past the bad_orientation limit.
            # At damping=2.0, the same speed only needs τ = 12 N·m/wheel,
            # allowing the robot to maintain speed without pitching over.
            damping=2.0,
            friction=0.0,
        ),
    },
)
"""
ArticulationCfg for the Unitree Go2W.

Actuator summary:
  leg_joints   — 12 revolute joints held in default stance by PD (stiffness=25)
  wheel_joints —  4 continuous wheel joints driven by velocity commands (stiffness=0)
"""
