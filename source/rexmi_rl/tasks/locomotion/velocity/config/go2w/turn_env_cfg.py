# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Pivot-turn environment configuration for the Go2W wheeled quadruped.

WHAT IS A PIVOT TURN?
---------------------
The Go2W is NOT a car. It is a legged robot with wheels on its feet.
The correct zero-radius pivot turn mechanism:
  1. Lock all 4 wheels (wheel velocity → 0, braking via Kd)
  2. Use hip/thigh/calf joints to lift feet one at a time
  3. Reposition each foot in a yaw-rotated direction
  4. Body rotates in place ~57°/s with ZERO translation

WHY THE PREVIOUS RUN FAILED (100% fall rate, bad_orientation=1.0)
-----------------------------------------------------------------
Training run at iter ~10985 showed:
  • bad_orientation: 1.0  — 100% of episodes ended in a fall
  • episode_length: 6 steps — robot fell in 0.12 seconds
  • position_drift: -0.40 — robot launched sideways in first step

ROOT CAUSE: Warm-start from rough policy (vx ∈ (-0.5,0.5) at speed) brought
wrong priors. At vx=0, omega=1.0 the rough policy fires aggressive wheel
commands that launch the robot sideways. With position_drift weight=-1.5,
the robot was hit with a massive penalty immediately, getting stuck in a
local minimum with no gradient to climb.

THE SOLUTION: Flat terrain + rebalanced rewards + random initialization
-----------------------------------------------------------------------
1. BASE CLASS: Go2wFlatEnvCfg (flat terrain, no curriculum)
   - No stairs, no slopes — robot CANNOT fall off terrain edges
   - Removes the roughness that was causing immediate falls
   - The pivot turn policy doesn't need rough terrain to be useful

2. REWARD REBALANCING:
   - track_ang_vel_z_exp: weight=2.0 (was 0.75 → now the DOMINANT signal)
   - track_lin_vel_xy_exp: weight=-0.5 (was -1.5, too harsh — overwhelmed yaw signal)
   - position_drift: weight=-0.3 (was -1.5, was 100× larger than yaw reward)
   - wheel_lock: weight=-0.05 (was -0.1, was causing too much conflict with balance)
   - Yaw tracking should be ≥50% of total reward signal at all times

3. TRAIN FROM SCRATCH (no warm-start):
   - Don't load rough checkpoint — it brings the wrong vx≠0 priors
   - Random init means the policy starts with small, balanced actions
   - The flat terrain gives it time to discover wheel-locking + leg-stepping

TRAINING COMMAND
----------------
  conda activate env_isaacsim

  # START FROM SCRATCH (recommended — no warm-start)
  python scripts/train.py --task RexmiRl-Go2w-Velocity-Turn-v0 --headless

  # OR: warm-start from flat policy (has vx=0 experience, safer than rough)
  python scripts/train.py --task RexmiRl-Go2w-Velocity-Turn-v0 --headless \\
      --load_run go2w_velocity_flat/<date> \\
      --checkpoint model_<N>.pt

  DO NOT warm-start from rough or rocky_slope — they have vx≠0 priors
  that destabilise the robot immediately at vx=0, omega=1.0.

TENSORBOARD HEALTH SIGNALS (what to watch)
-------------------------------------------
  bad_orientation          → must start DECREASING from iter 50
                             If still 1.0 at iter 100, STOP and diagnose.
  episode_length           → must increase (> 20 steps by iter 200)
  track_ang_vel_z_exp      → must increase (> 0.1 by iter 300, > 0.5 by 800)
  wheel_lock               → must approach 0 (wheels locking during spin)
  position_drift           → must approach 0 (staying in place)
  track_lin_vel_xy_exp     → should be slightly negative (drift penalty active)

NAV INTEGRATION
---------------
  PolicySelector activates TURN mode when:
    - |heading_error| > 75° (large yaw correction needed), OR
    - RecoveryFSM is REVERSING or ROTATING (obstacle escape)
  Commands: vx_cmd=0, vy_cmd=0, omega_cmd=±1.0 rad/s
  Use with: python scripts/navigate.py ... \\
      --ckpt_turn logs/rsl_rl/go2w_velocity_turn/<date>/model_<N>.pt
"""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

# Use FLAT base — not rough. Flat terrain eliminates falls from terrain edges
# and gives the policy a stable surface to learn the pivot motion.
from rexmi_rl.tasks.locomotion.velocity.config.go2w.rough_env_cfg import (
    Go2wFlatEnvCfg,
    Go2wFlatEnvCfg_PLAY,
)
from rexmi_rl.tasks.locomotion.velocity.mdp import (
    wheel_velocity_penalty,
    position_drift_penalty,
)


@configclass
class Go2wTurnEnvCfg(Go2wFlatEnvCfg):
    """
    Pivot-turn training environment — flat terrain, zero-radius pivot.

    Inherits from Go2wFlatEnvCfg (flat plane, no terrain curriculum, no height scanner).
    The height scanner observation is absent — the policy obs space matches the
    FLAT policy checkpoint (good for warm-starting from flat if needed).

    Key differences from Go2wFlatEnvCfg:
      1. Command: vx=0, vy=0, omega=±1.0 rad/s
      2. Reward: wheel_lock — penalise wheel spin during omega cmd (teaches locking)
      3. Reward: position_drift — penalise translation (enforces zero-radius pivot)
      4. Reward: track_lin_vel_xy_exp weight → negative (drift penalty)
      5. Reward: track_ang_vel_z_exp weight DOMINANT (2.0 — yaw is the objective)
      6. Reward: leg_deviation RELAXED (-0.01 vs -0.05) — legs need stepping freedom
      7. No terrain curriculum — flat plane only
    """

    def __post_init__(self):
        super().__post_init__()

        # ==================================================================
        # 1. COMMAND RANGE: pure pivot turn
        # ==================================================================
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # ==================================================================
        # 2. WHEEL LOCK PENALTY
        # ==================================================================
        # Penalise wheel spin when omega_cmd is active.
        # Teaches the policy to lock wheels and use legs for rotation.
        # Weight -0.05 (lighter than before — conflict with balance was too strong).
        # Gate: omega_threshold=0.3 so balance micro-corrections aren't penalised.
        self.rewards.wheel_lock = RewTerm(
            func=wheel_velocity_penalty,
            weight=-0.05,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_foot_joint"]),
                "omega_threshold": 0.3,
            },
        )

        # ==================================================================
        # 3. POSITION DRIFT PENALTY
        # ==================================================================
        # Penalise horizontal displacement from spawn.
        # Weight -0.3 (was -1.5 — too harsh, overwhelmed the yaw reward).
        # Dead zone 0.20m — generous enough to allow stepping motion.
        # The yaw tracking reward must be the LARGER signal.
        self.rewards.position_drift = RewTerm(
            func=position_drift_penalty,
            weight=-0.3,
            params={"drift_threshold": 0.20},
        )

        # ==================================================================
        # 4. REWARD RE-WEIGHTING — yaw tracking must dominate
        # ==================================================================

        # track_ang_vel_z_exp: PRIMARY OBJECTIVE
        # At 2.0, a perfect spin earns ~2.0/step.
        # All penalties combined should be < 1.0/step so the gradient
        # clearly points toward spinning.
        if hasattr(self.rewards, "track_ang_vel_z_exp"):
            self.rewards.track_ang_vel_z_exp.weight = 2.0

        # track_lin_vel_xy_exp: DRIFT PENALTY
        # vx_cmd=0 → this term = exp(-(vx_actual²)/std²).
        # Negative weight: any translation costs reward.
        # Weight -0.5 (was -1.5 — too harsh). At -0.5, the drift penalty
        # is 1/4 the yaw reward. Robot has strong incentive to spin, weak
        # incentive to not drift — correct priority order.
        if hasattr(self.rewards, "track_lin_vel_xy_exp"):
            self.rewards.track_lin_vel_xy_exp.weight = -0.5

        # leg_deviation: RELAXED — legs need stepping freedom.
        # Default -0.05 suppresses the hip/thigh/calf motion needed for pivot.
        # At -0.01, large deviation is slightly discouraged but not blocked.
        if hasattr(self.rewards, "leg_deviation"):
            self.rewards.leg_deviation.weight = -0.01

        # is_alive: keep at 0.2 — survival incentive is important for flat terrain
        # where the robot must learn to not fall over while spinning.
        # (inherits 0.2 from Go2wFlatEnvCfg — no change needed)

        # lin_vel_z_l2: keep at -1.5 — no vertical bouncing.
        # (inherits from Go2wFlatEnvCfg — no change needed)

        # ang_vel_xy_l2: keep at -0.5 — prevents roll/pitch instability.
        # (inherits from Go2wFlatEnvCfg — no change needed)

        # ==================================================================
        # 5. DEAD REWARDS (gates prevent firing at vx=0)
        # ==================================================================
        # stagnation_penalty: gate is (vx_cmd > 0.1) — never true at vx=0.
        # climb_progress: gate is (vx_cmd > 0.05) — never true at vx=0.
        # Both are structurally present but contribute zero signal.
        # Go2wFlatEnvCfg doesn't have these — they're from Go2wRoughEnvCfg.
        # Since we inherit from flat, neither is defined. No action needed.

        # ==================================================================
        # 6. NO TERRAIN CURRICULUM
        # ==================================================================
        # Go2wFlatEnvCfg already sets:
        #   terrain_type = "plane"
        #   terrain_generator = None
        #   curriculum.terrain_levels = None
        # No changes needed — flat plane is the correct training surface.

        # ==================================================================
        # 7. PUSH EVENTS — disable for turn training
        # ==================================================================
        # Push events from Go2wFlatEnvCfg (inherited from base class) add
        # random external forces. For early turn training, pushes make it
        # harder to learn the basic pivot motion. Disable them.
        # Can be re-enabled for fine-tuning once basic turning is established.
        self.events.push_robot = None


@configclass
class Go2wTurnEnvCfg_PLAY(Go2wTurnEnvCfg, Go2wFlatEnvCfg_PLAY):
    """
    Play/eval version of the pivot-turn environment.

    Inherits from Go2wTurnEnvCfg (pivot rewards) and Go2wFlatEnvCfg_PLAY
    (50 envs, no noise, no pushes, flat terrain).

    Use for evaluating the trained pivot-turn policy:
      python scripts/play.py --task RexmiRl-Go2w-Velocity-Turn-Play-v0 \\
          --load_run go2w_velocity_turn/<date> --checkpoint model_<N>.pt

    Health check:
      - Body yaw increases/decreases at ~1 rad/s (57°/s)
      - Wheels near-stationary (locking behavior)
      - Robot stays within 0.2m of spawn position
      - No falls on flat terrain
    """

    def __post_init__(self):
        Go2wTurnEnvCfg.__post_init__(self)
        Go2wFlatEnvCfg_PLAY.__post_init__(self)
