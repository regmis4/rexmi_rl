# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Pivot-turn environment configuration for the Go2W wheeled quadruped.

CURVED-PATH CURRICULUM (runs 18-20)
=====================================
After 17 failed attempts using direct omega_z commands (vx=0, omega=0.3),
the fundamental problem was identified: commanding vx=0, omega=0.3 requires
the robot to discover a completely new gait it has never seen. Every exploit
(body tilt, joint wind, leg oscillation) earns more reward than the correct
solution because the correct solution requires discovering a new motor pattern
from scratch.

SOLUTION: Curved-path curriculum — train wide-arc turning first, gradually
reduce turn radius to zero.

  Phase A (Turn-A): vx=(0.3, 0.5), omega=(0.15, 0.3) → R = 1.0-3.3 m
    Robot already knows forward walking. A wide leftward curve is 95% the same
    gait — just slightly differential leg/wheel effort. Policy learns "what
    omega_z while moving feels like" from a prior it already has.

  Phase B (Turn-B): vx=(0.0, 0.2), omega=(0.15, 0.3) → R = 0-1.3 m
    Tight curve/near-pivot. Robot is mostly turning with a little forward creep.
    Warm-start from Phase A — policy already understands curved motion, now
    pushes the radius inward.

  Phase C (Turn-C / final): vx=(0.0, 0.0), omega=(0.15, 0.3) → R = 0 (pure pivot)
    Pure in-place pivot. Warm-start from Phase B — the transition to R=0 is
    small (from R=0-0.3m in Phase B to R=0 in Phase C).

REWARD STRUCTURE (all phases)
==============================
All posture guards from run 17 are kept:
  - trunk_stability=-5.0 (15° threshold) — prevents trunk-winding exploit
  - flat_orientation_l2=-3.0 — heavy tilt penalty
  - yaw_stagnation min_yaw=20° — catches subtle oscillation exploits

Phase A uses track_lin_vel_xy_exp ACTIVE (not zeroed) — robot must walk
forward AND turn. This is the rough policy's native skill.

Phase B reduces track_lin_vel_xy_exp weight — turning becomes dominant.

Phase C zeros track_lin_vel_xy_exp — pure pivot, same as before but now
the robot has the motor pattern from A→B.

TRAINING COMMANDS
==================
  # Phase A: wide curve (500 iters from rough prior)
  python scripts/train.py --task RexmiRl-Go2w-Velocity-Turn-A-v0 --headless \\
      --load_run go2w_velocity_rough/2026-06-14_20-03-41 \\
      --checkpoint model_8996.pt --max_iterations 9500

  # Phase B: tight curve (500 iters from Phase A)
  python scripts/train.py --task RexmiRl-Go2w-Velocity-Turn-B-v0 --headless \\
      --load_run go2w_velocity_turn_a/<date> \\
      --checkpoint model_9500.pt --max_iterations 10000

  # Phase C: pure pivot (500 iters from Phase B)
  python scripts/train.py --task RexmiRl-Go2w-Velocity-Turn-v0 --headless \\
      --load_run go2w_velocity_turn_b/<date> \\
      --checkpoint model_10000.pt --max_iterations 10500
"""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from rexmi_rl.tasks.locomotion.velocity.config.go2w.rough_env_cfg import (
    Go2wRoughEnvCfg,
)
from rexmi_rl.tasks.locomotion.velocity.mdp import (
    base_height_penalty,
    foot_air_time_penalty,
    foot_alternation_reward,
    heading_progress,
    pivot_step_coordination,
    position_drift_penalty,
    trunk_stability_penalty,
    wheel_velocity_penalty,
    yaw_stagnation_penalty,
)


# ===========================================================================
# SHARED BASE: all turn phases inherit from this
# ===========================================================================

@configclass
class _Go2wTurnBase(Go2wRoughEnvCfg):
    """
    Shared base for all turn-curriculum phases.

    Sets up: flat terrain (num_rows=1), posture guards, foot alternation,
    wheel lock, yaw stagnation, base height, position drift.

    Each phase subclass overrides ONLY the command range and
    track_lin_vel_xy_exp weight.
    """

    def __post_init__(self):
        super().__post_init__()

        # ==================================================================
        # TERRAIN: flat rows only (inherited by all phases)
        # ==================================================================
        if (
            self.scene.terrain.terrain_generator is not None
            and hasattr(self.scene.terrain.terrain_generator, "num_rows")
        ):
            self.scene.terrain.terrain_generator.num_rows = 1
        self.curriculum.terrain_levels = None

        # ==================================================================
        # CORE YAW REWARD: track_ang_vel_z_exp
        # ==================================================================
        if hasattr(self.rewards, "track_ang_vel_z_exp"):
            self.rewards.track_ang_vel_z_exp.weight = 2.0

        if hasattr(self.rewards, "heading_progress"):
            self.rewards.heading_progress.weight = 0.0

        # ==================================================================
        # POSTURE GUARD 1: trunk stability (run 17 fix — never remove)
        # ==================================================================
        # Penalises trunk tilt >15° during active turn command.
        # 15° covers normal stepping tilt (±10°) with 5° margin.
        # 12° was too tight — caused body binding.
        # 10° was worst — completely froze gait dynamics.
        self.rewards.trunk_stability = RewTerm(
            func=trunk_stability_penalty,
            weight=-5.0,
            params={"max_tilt_deg": 15.0},
        )

        # ==================================================================
        # POSTURE GUARD 2: flat orientation (10× stronger than before)
        # ==================================================================
        # Combined with trunk_stability, 60° tilt costs -7.22/step.
        # 10° step-swing: -0.091/step → affordable.
        if hasattr(self.rewards, "flat_orientation_l2"):
            self.rewards.flat_orientation_l2.weight = -3.0

        # ==================================================================
        # FOOT ALTERNATION: reward stepping during turn command
        # ==================================================================
        self.rewards.foot_alternation = RewTerm(
            func=foot_alternation_reward,
            weight=1.5,
            params={
                "asset_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                ),
                "omega_threshold": 0.1,
                "contact_threshold": 5.0,
            },
        )

        # ==================================================================
        # WHEEL LOCK: penalise wheel spin during pivot command
        # ==================================================================
        self.rewards.wheel_lock = RewTerm(
            func=wheel_velocity_penalty,
            weight=-0.25,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_foot_joint"]),
                "omega_threshold": 0.1,
            },
        )

        # ==================================================================
        # BASE HEIGHT: prevent crawl-spin posture
        # ==================================================================
        self.rewards.base_height = RewTerm(
            func=base_height_penalty,
            weight=-20.0,
            params={"min_height": 0.28},
        )

        # ==================================================================
        # POSITION DRIFT: prevent sustained lateral drift
        # ==================================================================
        # Threshold 0.5m for Phase A (robot legitimately translates on arc)
        # Tightened in Phase C (pure pivot — should stay in place)
        self.rewards.position_drift = RewTerm(
            func=position_drift_penalty,
            weight=-0.5,
            params={"drift_threshold": 0.5},
        )

        # ==================================================================
        # YAW STAGNATION: min_yaw=20° — catches joint-winding exploit
        # ==================================================================
        self.rewards.yaw_stagnation = RewTerm(
            func=yaw_stagnation_penalty,
            weight=-2.0,
            params={
                "window_steps": 100,
                "min_yaw_deg": 20.0,  # joint-winding <15°, real spin >34°
                "min_cmd": 0.1,
            },
        )

        # ==================================================================
        # HEADING PROGRESS: accumulates actual heading change
        # ==================================================================
        # Weight is low here — the main signal for Phase A is track_lin+ang.
        # Phase C will rely on this more heavily.
        self.rewards.heading_progress_turn = RewTerm(
            func=heading_progress,
            weight=50.0,
            params={"min_cmd": 0.05},
        )

        # ==================================================================
        # STANDARD RE-WEIGHTING
        # ==================================================================
        if hasattr(self.rewards, "is_alive"):
            self.rewards.is_alive.weight = 0.2
        if hasattr(self.rewards, "leg_deviation"):
            self.rewards.leg_deviation.weight = -0.005
        if hasattr(self.rewards, "action_rate_l2"):
            # Phase B v6: tightened from -0.020 → -0.035 to reduce high-frequency
            # jerky joint commands observed in visual eval (action_rate=-0.68).
            self.rewards.action_rate_l2.weight = -0.035
        if hasattr(self.rewards, "ang_vel_xy_l2"):
            # Phase B v8: reverted to -0.05 (Unitree official walking value).
            # v7 attempted -0.15 to damp centripetal sway during fast pivot.
            # RESULT: destroyed the gait — bad_orientation jumped 23%→85.6%.
            # Root cause: lateral angular velocity IS the stepping motion during
            # a pivot. Penalising it suppresses the micro-tap gait mechanics.
            # The overspin issue should be addressed ONLY via command range,
            # not via ang_vel_xy damping.
            self.rewards.ang_vel_xy_l2.weight = -0.05
        if hasattr(self.rewards, "undesired_contacts"):
            self.rewards.undesired_contacts.weight = -4.0
        if hasattr(self.rewards, "dof_pos_limits"):
            # -2.0 (Phase B v1 level): sufficient to deter limit-hitting
            # without blocking the hip/abductor range needed for coordinated turns.
            # -10.0 collapsed heading_progress from +0.48 to +0.28 (v2/v3/v4).
            self.rewards.dof_pos_limits.weight = -2.0

        # ==================================================================
        # PIVOT COORDINATION: reward feet stepping in the correct arc direction
        # ==================================================================
        # The missing signal: foot_alternation rewards COUNT of lifted feet
        # but not DIRECTION. This reward fires when swinging feet move
        # laterally in the correct direction for the commanded yaw:
        #   Left turn: front feet +y, rear feet -y (tank-tread/scissor pattern)
        #   Right turn: signs reversed.
        # This teaches natural coordinated sidestepping without joint limits.
        # Two cfgs required: contact_cfg for swing detection, robot_body_cfg for velocity.
        self.rewards.pivot_step_coord = RewTerm(
            func=pivot_step_coordination,
            weight=2.0,
            params={
                "contact_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                ),
                "robot_body_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                ),
                # Phase B v6: lowered from 0.03 → 0.01 to reward gentle shuffles
                # equally with fast lunges. 0.03 inadvertently biased toward high
                # foot velocity (easier to satisfy → more reward per step).
                "min_swing_vel": 0.01,
            },
        )

        # ==================================================================
        # FOOT AIR TIME: penalise feet staying off ground > 6 steps (0.12s)
        # ==================================================================
        # Phase B v6: added after visual eval showed large lunging strides.
        # The robot takes big arcs with long air time instead of quick shuffles.
        # Root cause: foot_alternation rewards COUNT of feet up, not air duration.
        # This penalty forces quick plant-and-go: free zone = 6 steps (0.12s),
        # every step beyond that incurs cost proportional to excess.
        # At weight=-1.0 a 20-step hang (-14 penalty) is still net positive
        # vs foot_alternation (+20) but no longer free, pushing toward 6-step shuffles.
        self.rewards.foot_air_time = RewTerm(
            func=foot_air_time_penalty,
            weight=-1.0,
            params={
                "asset_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                ),
                "max_air_steps": 6,
                "omega_threshold": 0.05,
            },
        )

        # NOTE: dof_acc_l2 is intentionally NOT overridden here.
        # Raw joint acceleration values during turning are very large (~12k/step).
        # Any weight override causes exponential reward explosion.
        # The rough_env_cfg default weight is sufficient.

        # No push events during turn training
        self.events.push_robot = None


# ===========================================================================
# PHASE A: Wide-arc turning (vx=0.3-0.5, omega=0.15-0.3, R=1-3m)
# ===========================================================================

@configclass
class Go2wTurnAEnvCfg(_Go2wTurnBase):
    """
    Phase A: Wide-arc turning — robot walks forward while curving.

    vx=(0.3, 0.5), omega=(0.15, 0.3) → turn radius R = 1.0-3.3m

    The rough policy already knows forward walking at 0.3-0.5 m/s.
    Adding a simultaneous omega=0.15-0.3 rad/s command requires only
    a slight differential effort — this is immediately learnable from
    the rough prior. The policy learns "what rotating feels like" while
    still using its comfortable forward-walking gait.

    track_lin_vel_xy_exp: ACTIVE at +1.0 (robot must also walk forward).
    track_ang_vel_z_exp: ACTIVE at +2.0 (robot must also turn).
    Both must be satisfied simultaneously — no shortcut for either alone.

    Warm-start: model_8996.pt (rough walking policy)
    Expected convergence: 500 iterations (robot already knows ~90% of this)
    """

    def __post_init__(self):
        super().__post_init__()

        # Wide-arc command: vx active, omega active (both CW and CCW)
        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.3, 0.3)

        # track_lin_vel_xy_exp: ACTIVE — robot must walk forward AND turn
        # The rough prior makes this immediately learnable.
        # This is the KEY difference from previous turn attempts:
        # the robot has a "foot in the door" via its walking prior.
        if hasattr(self.rewards, "track_lin_vel_xy_exp"):
            self.rewards.track_lin_vel_xy_exp.weight = 1.0

        # Position drift: large threshold (0.5m) — robot legitimately
        # translates along the arc. Don't penalise normal arc motion.
        # Override the base class default of 0.5m — same, but explicit.
        self.rewards.position_drift.params["drift_threshold"] = 0.5

        # heading_progress: moderate weight — arc motion naturally
        # produces heading change so this fires easily.
        self.rewards.heading_progress_turn.weight = 30.0


@configclass
class Go2wTurnAEnvCfg_PLAY(Go2wTurnAEnvCfg):
    """Play config for Phase A visual eval."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


# ===========================================================================
# PHASE B: Tight-arc turning (vx=0.0-0.2, omega=0.15-0.3, R=0-1.3m)
# ===========================================================================

@configclass
class Go2wTurnBEnvCfg(_Go2wTurnBase):
    """
    Phase B: Tight-arc to near-pivot — mostly turning, little forward motion.

    vx=(0.0, 0.2), omega=(0.15, 0.3) → turn radius R = 0-1.3m

    Warm-start from Phase A checkpoint. Robot already understands
    "walking + turning simultaneously." Now vx is reduced to near zero
    while omega stays the same — the gait must adapt from wide arc to
    tight spiral to near-pivot. The gradient is smooth because Phase A
    already established the turning pattern.

    track_lin_vel_xy_exp: REDUCED weight (0.5) — turning is more important.
    heading_progress: INCREASED weight (100) — net heading change is primary.

    Phase B v2 (posture refinement):
    Warm-start from Phase B model_9994.pt with stricter posture guards:
      - dof_pos_limits: -2 → -10 (eliminates calf-splay/leg-arch exploit)
      - ang_vel_xy_l2: -1.5 → -3.0 (prevents body pitch/roll during turn)
      - dof_acc_l2: → -0.05 (smooth motions, no jerky leg flinging)
      - trunk_stability threshold: 15° → 10° (tighter body upright requirement)

    Warm-start: Phase A checkpoint OR Phase B model_9994.pt for refinement
    Expected convergence: 300-500 iterations
    """

    def __post_init__(self):
        super().__post_init__()

        # Tight-arc to near-pivot (both CW and CCW)
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.2)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        # Phase B v7: reduced from (-0.3, 0.3) → (-0.2, 0.2) rad/s to slow spin.
        # v6 visual eval confirmed breakthrough gait but robot spins too fast,
        # causing centripetal instability. 0.2 rad/s = ~11.5°/s, comfortable
        # pivot rate for the Go2W mass distribution.
        self.commands.base_velocity.ranges.ang_vel_z = (-0.2, 0.2)

        # track_lin_vel_xy_exp: REDUCED — turning is more dominant now
        if hasattr(self.rewards, "track_lin_vel_xy_exp"):
            self.rewards.track_lin_vel_xy_exp.weight = 0.5

        # Position drift: medium threshold (0.25m) — robot still moves
        # forward a little, but pivot-like motion should stay near spawn.
        self.rewards.position_drift.params["drift_threshold"] = 0.25

        # heading_progress: higher weight — net rotation is now primary
        self.rewards.heading_progress_turn.weight = 100.0


@configclass
class Go2wTurnBEnvCfg_PLAY(Go2wTurnBEnvCfg):
    """Play config for Phase B visual eval.

    Uses fixed strong turn commands so ALL robots get a meaningful command —
    avoids the near-zero omega dead zone that causes weird poses.
    vx=(0.1, 0.1): tiny fixed creep (robot needs it to stay upright)
    omega=(0.25, 0.3): all robots turn left at strong angular velocity
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        # Phase B v7: reduced from (0.25, 0.3) → (0.15, 0.2) to match slower
        # training range. Visual eval at 0.25-0.3 rad/s was causing falls due
        # to centripetal instability at that speed. Match training command range.
        self.commands.base_velocity.ranges.lin_vel_x = (0.1, 0.1)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.15, 0.2)


# ===========================================================================
# PHASE C: Pure pivot (vx=0, omega=0.15-0.3, R=0)
# ===========================================================================

@configclass
class Go2wTurnEnvCfg(_Go2wTurnBase):
    """
    Phase C (final): Pure in-place pivot turn.

    vx=0, omega=(0.15, 0.3) → R=0, pure pivot

    Warm-start from Phase B checkpoint. The robot has now experienced
    curved arcs at all radii from ~3m down to ~0.3m. The final step to
    R=0 is a small extrapolation, not a large discovery.

    track_lin_vel_xy_exp: ZERO — no forward walking allowed.
    heading_progress: weight=150 — net rotation is the ONLY objective.
    position_drift: tight threshold (0.15m) — stay in place while spinning.

    Warm-start: Phase B checkpoint
    Expected convergence: 300-500 iterations
    """

    def __post_init__(self):
        super().__post_init__()

        # Pure pivot: no translation, both CW and CCW
        # (-0.3, 0.3) gives symmetric commands — robot learns both directions.
        # yaw_stagnation uses |net_yaw| so it fires for near-zero omega correctly.
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.3, 0.3)

        # track_lin_vel_xy_exp: ZERO — pure pivot only
        if hasattr(self.rewards, "track_lin_vel_xy_exp"):
            self.rewards.track_lin_vel_xy_exp.weight = 0.0

        # Position drift: tight (0.15m) — robot must stay in place
        self.rewards.position_drift.params["drift_threshold"] = 0.15

        # heading_progress: maximum weight — net rotation is everything
        self.rewards.heading_progress_turn.weight = 150.0


@configclass
class Go2wTurnEnvCfg_PLAY(Go2wTurnEnvCfg):
    """
    Play/eval version of Phase C.

    Fixed strong turn commands for honest visual eval — no near-zero dead zone.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        # Fixed commands: all robots turn left with small forward creep
        self.commands.base_velocity.ranges.lin_vel_x = (0.05, 0.05)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.25, 0.3)
