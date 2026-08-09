# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Custom reward functions for REXMI velocity-tracking environments.

Each function follows the Isaac Lab MDP convention:
  signature : (env: ManagerBasedRLEnv, **kwargs) -> torch.Tensor  shape (num_envs,)
  positive  : bonus
  negative  : penalty (apply a negative weight in the RewardTermCfg)
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply_inverse


def climb_progress(
    env: ManagerBasedRLEnv,
    base_weight: float = 0.4,
    obstacle_weight: float = 1.5,
    height_threshold: float = 0.10,
) -> torch.Tensor:
    """
    Reward upward base movement while a forward command is active.

    HYBRID DESIGN — Phase 7 (Option A + B combined):
    --------------------------------------------------
    The effective reward weight is dynamically scaled based on what the height
    scanner detects ahead of the robot:

      • Flat / gentle terrain  (no obstacle detected):
          weight = base_weight  (default 0.4)
          On flat ground vz ≈ 0 so the contribution is near zero per step.
          Even if the robot attempts to bounce, the penalty math works out:
            With lin_vel_z_l2 = -1.5, bouncing at vz = 0.15 m/s costs:
              +0.4 × 0.15  − 1.5 × 0.15²  =  +0.060 − 0.034  =  +0.026/step
            This is marginal and far below the velocity-tracking reward —
            bouncing on flat ground is no longer profitable.

      • Near an obstacle  (step / stair / box detected in height scan):
          weight = obstacle_weight  (default 1.5)
          Strong signal: climbing at 0.3 m/s earns +0.45/step, which
          outweighs the flat_orientation_l2 penalty during a 30° climbing pitch
          (−0.8 × (30° in rad)² ≈ −0.34/step), making climbing profitable.

    Obstacle detection — median floor method:
    ------------------------------------------
    The height scanner covers a 1.6 × 1.0 m yaw-aligned grid (default ~187 rays
    at 0.1 m resolution).  Each ray hits the terrain and records its world-frame
    Z coordinate.

    Algorithm:
      1. current_floor  = median of all ray hit heights
                          (robust: most rays still see the current floor even
                          when some forward rays are already over a stair edge)
      2. max_elevation  = max(ray_hits_z − current_floor)
                          (= how high the tallest terrain feature in the scan is)
      3. near_obstacle  = max_elevation > height_threshold

    Why median (not mean):
      When 20% of the 187 rays see a 20 cm stair ahead and 80% are on the
      current floor, the median is unaffected by the elevated minority and
      correctly estimates the current floor.  The mean would be pulled up by
      ~4 cm, diluting the elevation signal.

    Why height_threshold = 0.10 m:
      • Gentle slope 2° over 1.6 m: max_elevation ≈ 1.6 × tan(2°) ≈ 0.056 m  < threshold → flat mode  ✓
      • Random rough max 0.10 m noise:  max_elevation ≈ 0.10 m  ≤ threshold → flat mode  ✓ (borderline)
      • 12 cm step:  max_elevation ≈ 0.12 m  > threshold → obstacle mode  ✓
      • 20 cm step:  max_elevation ≈ 0.20 m  > threshold → obstacle mode  ✓
      • Steep slope 10°: max_elevation ≈ 0.28 m  > threshold → obstacle mode  ✓
        (acceptable: robot IS climbing, full weight is appropriate)

    IMPORTANT: Set the RewardTermCfg weight to 1.0 — the effective weight is
    returned directly from this function (baked into the return value).

    Parameters
    ----------
    base_weight       : Reward weight used when no obstacle is detected.
                        Default 0.4 — still slightly rewards genuine upward
                        motion but too small to make bouncing worthwhile.
    obstacle_weight   : Reward weight used when an obstacle is detected.
                        Default 1.5 — strong enough to counteract the
                        flat_orientation_l2 penalty during a climbing pitch.
    height_threshold  : Terrain elevation above the estimated current floor (m)
                        that triggers obstacle mode.  Default 0.10 m is above
                        the wheel radius (0.05 m) and safe random-rough noise.

    Returns
    -------
    Tensor shape (num_envs,), value ∈ [0, obstacle_weight × 0.5].
    Multiply by a RewardTermCfg weight of 1.0.
    """
    robot = env.scene["robot"]
    height_scanner = env.scene["height_scanner"]

    # World-frame vertical velocity of the base link
    vz: torch.Tensor = robot.data.root_lin_vel_w[:, 2]  # (num_envs,)

    # Only active when there is a meaningful forward command
    cmd_fwd: torch.Tensor = env.command_manager.get_command("base_velocity")[:, 0]
    has_cmd: torch.Tensor = cmd_fwd > 0.1  # bool (num_envs,)

    # Reward only upward motion, capped at 0.5 m/s (wheel tangential speed limit)
    climb: torch.Tensor = torch.clamp(vz, min=0.0, max=0.5)

    # ------------------------------------------------------------------
    # Obstacle detection via height scanner
    # ------------------------------------------------------------------
    # ray_hits_w: (num_envs, num_rays, 3) — world-frame terrain hit positions.
    # All rays shoot straight down so the Z coordinate is the terrain height.
    ray_hits_z: torch.Tensor = height_scanner.data.ray_hits_w[..., 2]  # (N, num_rays)

    # Step 1: Estimate current floor as the median ray hit height.
    # Median is robust to a minority of elevated readings (e.g., forward rays
    # already above a stair) that would bias a mean estimate.
    current_floor: torch.Tensor = ray_hits_z.median(dim=1).values  # (N,)

    # Step 2: Compute maximum terrain elevation above current floor in the scan.
    max_elevation: torch.Tensor = (
        ray_hits_z - current_floor.unsqueeze(1)
    ).amax(dim=1)  # (N,)

    # Step 3: Flag obstacle when any scan point exceeds the height threshold.
    near_obstacle: torch.Tensor = (max_elevation > height_threshold).float()  # (N,)

    # Dynamic weight: base_weight on flat terrain, obstacle_weight near obstacles.
    # Linear interpolation: weight ∈ {base_weight, obstacle_weight}
    weight: torch.Tensor = base_weight + (obstacle_weight - base_weight) * near_obstacle

    return has_cmd.float() * climb * weight


def stagnation_penalty(
    env: ManagerBasedRLEnv,
    threshold: float = 0.05,
) -> torch.Tensor:
    """
    Penalise each step where the robot is nearly stationary despite a forward command.

    The signal fires whenever ALL of these are true:
      • The commanded forward velocity is > 0.1 m/s  (robot is being asked to move)
      • The actual forward velocity is < ``threshold`` m/s  (robot is stuck)

    With weight = -0.5, a robot frozen for 60 consecutive steps accumulates -30
    reward — equivalent to failing to track a 0.5 m/s command for the same period.
    This gives the policy a gradient to try *something different* (back off, kick
    the legs, reorient) rather than spinning wheels in place indefinitely.

    Why no window / rolling counter?
    ----------------------------------
    Isaac Lab accumulates rewards over the full episode, so a per-step penalty that
    fires every stuck step IS a window effect — the longer the robot stays stuck, the
    larger the total penalty.  Avoiding persistent state also makes the function
    reset-safe: no counter desynchronisation across vectorised episodes.

    Parameters
    ----------
    env       : the running ManagerBasedRLEnv
    threshold : forward velocity (m/s) below which the robot is considered stuck.
                Default 0.05 m/s = 10% of the 0.5 m/s training command.

    Returns
    -------
    Tensor shape (num_envs,), value 1.0 when stuck-while-commanded, else 0.0.
    Multiply by a negative weight in RewardTermCfg for a penalty.
    """
    robot = env.scene["robot"]

    # Actual forward velocity in the body frame (x-axis), unclamped
    fwd_vel: torch.Tensor = robot.data.root_lin_vel_b[:, 0]

    # Commanded forward velocity from the velocity command manager
    # shape (num_envs, 3) — index 0 is lin_vel_x
    cmd_vel: torch.Tensor = env.command_manager.get_command("base_velocity")[:, 0]

    # Only penalise when there is a meaningful forward command
    has_fwd_cmd: torch.Tensor = cmd_vel > 0.1  # bool (num_envs,)

    # Robot is "stuck" when its actual speed is below threshold in either direction
    is_stuck: torch.Tensor = fwd_vel.abs() < threshold  # bool (num_envs,)

    return (has_fwd_cmd & is_stuck).float()


def hip_crossing_penalty(
    env: ManagerBasedRLEnv,
    threshold_rad: float = 0.25,
    asset_cfg=None,
) -> torch.Tensor:
    """
    Penalise hip joints that exceed a threshold deviation from their default position.

    This creates a DEAD ZONE ± threshold_rad around the default hip stance where the
    policy is completely free — no cost for normal slope-balance adjustments.
    Only "weirdo territory" (hip deviation beyond ± threshold_rad) is penalised.

    WHY A THRESHOLD PENALTY INSTEAD OF PLAIN L1?
    --------------------------------------------
    The existing ``leg_deviation`` term uses L1 (linear penalty on all joint deviation).
    At weight -0.05, a hip crossing 0.5 rad from default costs only 0.025/step —
    far too weak to deter the crossing exploit when the crossing provides even
    slight stability benefit.

    A linear penalty also CANNOT discriminate between:
      • Normal slope-balance lean (±0.15 rad): acceptable — robot needs this to
        traverse slopes and respond to lateral (vy) velocity commands.
      • Rear leg crossing (±0.5 rad): weirdo territory — robot tips sideways onto
        one wheel, left leg migrates to where right leg should be.

    With threshold_rad = 0.25:
      • 0.15 rad lean → excess = 0   → zero cost (slope balance preserved ✓)
      • 0.40 rad crossing → excess = 0.15 → at weight -2.0: cost = -0.30/step
      • 0.50 rad crossing → excess = 0.25 → at weight -2.0: cost = -0.50/step

    At 0.50 rad crossing: -0.50/step makes crossing unprofitable vs. the +1.6/step
    velocity tracking reward.  The policy will switch from the crossing gait to
    proper slope-aligned traversal.

    WHY THIS WON'T BLOCK vy TRACKING:
    ----------------------------------
    In Phase 8 attempt 1, ``hip_deviation=-0.5`` (linear) blocked vy tracking because
    it penalised ANY hip deviation, even the 0.05–0.15 rad needed for lateral stepping.
    This threshold version leaves ±0.25 rad completely free, which covers all normal
    lateral stepping.  The free zone is 1.7× wider than the maximum hip use needed for
    vy tracking (±0.15 rad), providing a comfortable safety margin.

    WHY ONLY IN THE STEEP-SLOPE ENV:
    ---------------------------------
    This penalty is specific to ``steep_slope_env_cfg.py`` (not the rough env).
    The rough env (model_8996) is frozen — we do NOT change its reward function.
    On flat/rough terrain the hip crossing exploit never developed because the
    curriculum never reached 33°+ slopes consistently.

    Parameters
    ----------
    env           : the running ManagerBasedRLEnv
    threshold_rad : dead zone radius around each hip's default position (rad).
                    Default 0.25 rad: free zone covers normal slope-balance use
                    (±0.15 rad) with 0.10 rad margin.
    asset_cfg     : SceneEntityCfg with joint_ids resolved to hip joint indices.
                    Use joint_names=[".*_hip_joint"] in the RewardTermCfg params.

    Returns
    -------
    Tensor shape (num_envs,).
    Sum of excess hip deviations beyond threshold across all 4 hip joints.
    Multiply by a negative weight (-2.0 recommended) in RewardTermCfg.

    Example RewardTermCfg (in steep_slope_env_cfg.py)::

        from isaaclab.managers import SceneEntityCfg
        self.rewards.hip_crossing = RewTerm(
            func=hip_crossing_penalty,
            weight=-2.0,
            params={
                "threshold_rad": 0.25,
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"]),
            },
        )
    """
    if asset_cfg is None:
        raise ValueError("hip_crossing_penalty requires asset_cfg with joint_names=['.*_hip_joint']")

    robot = env.scene[asset_cfg.name]

    # Current hip joint positions vs. their default (spawn) positions
    # joint_pos shape: (num_envs, total_joints)
    # joint_ids: list of hip joint indices resolved at env startup
    hip_pos: torch.Tensor = robot.data.joint_pos[:, asset_cfg.joint_ids]
    hip_default: torch.Tensor = robot.data.default_joint_pos[:, asset_cfg.joint_ids]

    # Absolute deviation from default stance for each hip joint
    hip_dev: torch.Tensor = (hip_pos - hip_default).abs()  # (num_envs, 4)

    # Dead zone: no cost within ±threshold_rad of default.
    # Beyond the threshold, return the excess deviation (not the full deviation).
    # This creates a soft boundary: cheap inside the zone, costly outside.
    excess: torch.Tensor = (hip_dev - threshold_rad).clamp(min=0.0)  # (num_envs, 4)

    # Sum excess across all 4 hip joints (FL, FR, RL, RR)
    return excess.sum(dim=-1)  # (num_envs,)


def joint_deviation_threshold(
    env: ManagerBasedRLEnv,
    threshold_rad: float = 0.40,
    asset_cfg=None,
) -> torch.Tensor:
    """
    Generalised threshold-based joint deviation penalty.

    Identical logic to ``hip_crossing_penalty`` but applicable to any joint group
    (thighs, calves, or any other set of joints).  Returns ZERO cost within
    ± threshold_rad of each joint's default position, and the excess deviation
    beyond that threshold otherwise.

    This is the correct tool for defining "normal behaviour boundaries" for each
    joint group separately, without over-constraining the action space globally.

    WHY SEPARATE THRESHOLDS PER JOINT GROUP MATTER:
    ------------------------------------------------
    On steep slopes the legs are ALREADY displaced from their defaults just to
    maintain wheel contact with the tilted surface.  On a 35° slope:
      • Thighs: ~0.20 rad used for slope adaptation
      • Calves: ~0.30 rad used for wheel reach on tilted ground

    This "slope budget" is consumed before any lateral (vy) or forward motion
    begins.  If the threshold is too tight, the remaining free range inside the
    dead zone shrinks, and the policy may find that lateral stepping costs
    something and give up vy tracking in favour of wheel-only steering.

    Each joint group has a different threshold sized to its natural range:
      - Hips  (implemented via hip_crossing_penalty): threshold=0.25 rad
            Lateral lean needs ~0.15 rad; crossing exploit starts at ~0.50 rad.
      - Thighs (this function, recommended threshold=0.40 rad):
            CG shift + slope adapt needs ~0.25-0.30 rad; salute starts ~0.55 rad.
            0.40 rad leaves ~0.15-0.20 rad free above the slope budget.
      - Calves (NOT recommended unless specific exploit observed):
            Full wheel reach on 45° slope needs ~0.40-0.50 rad.
            Any threshold risks blocking terrain adaptation.

    WEIGHT CALIBRATION:
    -------------------
    Use lower weights here than for hip_crossing_penalty because thigh/calf
    exploits are less severe than lateral rolling (hip crossing puts one wheel
    fully in the air — the most destabilising configuration):
      - Hips:   weight=-2.0  (one wheel off ground — severe)
      - Thighs: weight=-1.0  (body pitches oddly but all wheels still on ground)
      - Calves: weight=-0.5  (if ever needed — very mild)

    Parameters
    ----------
    env           : the running ManagerBasedRLEnv
    threshold_rad : dead zone radius around each joint's default position (rad).
    asset_cfg     : SceneEntityCfg with joint_ids resolved to target joint indices.

    Returns
    -------
    Tensor shape (num_envs,).
    Sum of excess deviations beyond threshold across all specified joints.
    Multiply by a negative weight in RewardTermCfg.

    Example (thigh salute penalty in steep_slope_env_cfg.py)::

        self.rewards.thigh_salute = RewTerm(
            func=joint_deviation_threshold,
            weight=-1.0,
            params={
                "threshold_rad": 0.40,
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_thigh_joint"]),
            },
        )
    """
    if asset_cfg is None:
        raise ValueError("joint_deviation_threshold requires asset_cfg with joint_names specified")

    robot = env.scene[asset_cfg.name]

    joint_pos: torch.Tensor = robot.data.joint_pos[:, asset_cfg.joint_ids]
    default_pos: torch.Tensor = robot.data.default_joint_pos[:, asset_cfg.joint_ids]

    dev: torch.Tensor = (joint_pos - default_pos).abs()
    excess: torch.Tensor = (dev - threshold_rad).clamp(min=0.0)

    return excess.sum(dim=-1)


def wheel_velocity_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg=None,
    omega_threshold: float = 0.3,
) -> torch.Tensor:
    """
    Penalise wheel angular velocity — but ONLY when a spin command is active.

    Used exclusively in the spin-in-place training environment to enforce
    the legged-pivot turning strategy:

      Goal: lock wheels (Kd × ω braking = active hold) while the policy
      uses hip/thigh/calf leg dynamics to rotate the body around its own
      vertical axis.  The wheels are ANCHORS, not actuators, during a spin.

    Why the omega_cmd gate is critical (v4 fix)
    -------------------------------------------
    v3 used weight=-2.0 with NO gate.  model_13994 had strong wheel-drive priors
    from rocky-slope training (normal driving speed ω ≈ 6 rad/s per wheel).
    At episode start the penalty was:
        -2.0 × (6² × 4) = -288/step
    This completely overwhelmed every positive signal (+3.0 ang_vel max,
    +0.2 is_alive) from the first step.  The policy discovered:
        "fall over fast = shorter episode = less total wheel_brake penalty"
    Result: 96.7% base_contact termination, ~18-step episodes, exploding value
    function loss (50–91).

    The gate ``|omega_cmd| > omega_threshold`` ensures:
    • The penalty only fires when the spin command is active (|ω_cmd| > 0.3 rad/s).
    • During normal stance/balance (omega_cmd ≈ 0) wheels can spin freely for
      posture adjustment — no penalty.
    • The policy first learns to survive on the slope (positive reward landscape),
      THEN gradually learns to lock wheels while spinning as the gate activates.

    Weight -0.05 (reduced from v3's -2.0)
    --------------------------------------
    At ω=6 rad/s with gate active: -0.05 × (6² × 4) = -7.2/step
    Angular tracking reward (perfect):                 +3.0/step
    is_alive reward (per step):                        +0.2/step

    Net at episode start (mostly falling, ω≈6): ≈ -7.2 + 0.016 + 0.003 ≈ -7.2
    But the policy can earn positive reward by surviving (is_alive × steps).
    At 1000 steps: +0.2 × 1000 = +200 per episode — a reachable positive target.
    The policy is not trapped in a death spiral.

    As training progresses (policy locks wheels → ω → 0):
        wheel_brake contribution → 0
        angular tracking rises   → +3.0/step at 1000 steps = +3000/episode
    The policy naturally converges to: lock wheels, use legs.

    Gradient curriculum effect:
    ----------------------------
    Early training: gate rarely fires (policy exploring, omega_cmd drawn ∈ ±1.0
    but robot falls before executing many spin steps).
    Mid training: policy stays alive longer, gate fires more, wheel_brake grows,
    policy learns to reduce wheel velocity under spin command.
    Late training: wheels locked under spin → penalty → 0 → full angular reward.

    Sim → Real mapping
    ------------------
    On the real Unitree Go2W (QDD motors):
      • ``damping`` maps to motor Kd — higher Kd = stronger active braking
      • ``stiffness=0`` = velocity mode (no position restoring force)
    Commanding target_ω = 0 on a real QDD produces Kd × ω resistive torque.
    This penalty enforces what the real motor should do: brake wheels when
    spinning is commanded, use legs for rotation.

    No ``friction`` added — joint friction is always-on passive drag that would
    hurt normal forward driving.  Active Kd braking (commanded ω=0) is the
    correct, hardware-equivalent approach.

    Parameters
    ----------
    asset_cfg       : SceneEntityCfg with joint_ids resolved to wheel joint indices.
                      Use joint_names=[".*_foot_joint"] in the RewardTermCfg params.
    omega_threshold : Only penalise when |omega_cmd| exceeds this value (rad/s).
                      Default 0.3 rad/s — below this the spin command is negligible
                      and the robot should be free to use wheels for balance.

    Returns
    -------
    Tensor shape (num_envs,).
    Sum of squared wheel velocities × spin-command gate.
    Multiply by a small negative weight (-0.05 recommended) in RewardTermCfg.

    Example RewardTermCfg (in spin_env_cfg.py)::

        from isaaclab.managers import SceneEntityCfg
        self.rewards.wheel_brake = RewTerm(
            func=wheel_velocity_penalty,
            weight=-0.05,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_foot_joint"]),
                "omega_threshold": 0.3,
            },
        )
    """
    if asset_cfg is None:
        raise ValueError(
            "wheel_velocity_penalty requires asset_cfg with joint_names=['.*_foot_joint']"
        )

    robot = env.scene[asset_cfg.name]

    # Gate: only penalise when a meaningful spin command is active.
    # omega_cmd is index 2 of the (vx, vy, omega) base_velocity command vector.
    omega_cmd: torch.Tensor = env.command_manager.get_command("base_velocity")[:, 2]
    spin_gate: torch.Tensor = (omega_cmd.abs() > omega_threshold).float()  # (num_envs,)

    # Wheel joint angular velocities: (num_envs, 4)
    wheel_vel: torch.Tensor = robot.data.joint_vel[:, asset_cfg.joint_ids]

    # L2 norm squared across all 4 wheels — penalises wheel spin under spin command.
    # Gated: zero cost when omega_cmd is small (robot using wheels for balance/stance).
    return spin_gate * wheel_vel.pow(2).sum(dim=-1)  # (num_envs,)


def uphill_lean_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Reward nose-down body pitch while the robot is actively climbing ascending terrain.

    On uphill slopes the robot needs to shift weight forward (lean into the slope) to
    maintain wheel contact pressure and resist backward slipping under gravity.  This
    term provides a direct gradient for that posture change, complementing the weak
    flat_orientation_l2 penalty (−0.1 in the steep/rocky slope envs) which only
    discourages tilting but never *rewards* the correct slope-aligned lean.

    Implementation
    --------------
    Body lean is read from ``projected_gravity_b[:, 0]``.  When the body tilts
    nose-down by angle θ, gravity has a forward component sin(θ) in the body frame,
    so ``projected_gravity_b[0] ≈ sin(θ)`` (positive = nose-down = leaning into slope).

    The reward is **gated by actual upward world-frame velocity** (vz > 0.01 m/s)
    so it only fires when the robot is genuinely climbing.  This prevents rewarding
    an arbitrary forward lean on flat terrain where it is not needed.

    Reward accounting (RewardTermCfg weight = +0.5)
    ------------------------------------------------
      θ = 10° lean : 0.5 × sin(10°) ≈ 0.09/step
      θ = 15° lean : 0.5 × sin(15°) ≈ 0.13/step
      θ = 24° lean : 0.5 × sin(24°) ≈ 0.20/step   ← matches 25° slope lean target

    At 0.20/step this is comparable to the stagnation penalty (-2.5/step when stuck)
    — large enough to shape posture but not large enough to compete with velocity
    tracking (+1.71–1.74/step observed in Phase 8d).

    Phase 8e motivation
    --------------------
    Phase 8d (100% uphill, 1500 iters, resume model_12495.pt) stalled at
    terrain_levels ≈ 0.19 (≈18° slope).  The curriculum could not advance because
    the combined load at 25°+ exceeded the policy's uphill push capacity.  The
    missing behaviour: the robot was NOT leaning into the slope — it was trying
    to stay flat (penalised by flat_orientation_l2=-0.1 if it did lean) while
    fighting the full component of gravity.  This term teaches the correct posture.

    Returns
    -------
    Tensor shape (num_envs,), value ≥ 0.0.
    Multiply by a positive weight (+0.5 recommended) in RewardTermCfg.
    """
    robot = env.scene["robot"]

    # World-frame vertical velocity — positive = climbing upward
    vz: torch.Tensor = robot.data.root_lin_vel_w[:, 2]  # (num_envs,)

    # Forward command gate: only reward when commanded to move forward
    cmd_fwd: torch.Tensor = env.command_manager.get_command("base_velocity")[:, 0]
    climbing: torch.Tensor = ((cmd_fwd > 0.05) & (vz > 0.01)).float()  # (num_envs,)

    # Body nose-down lean: projected_gravity_b[0] > 0 means nose-down (forward tilt)
    # sin(θ) where θ is the forward pitch angle — ranges 0 to 1 for 0° to 90°
    grav_fwd: torch.Tensor = robot.data.projected_gravity_b[:, 0]
    fwd_lean: torch.Tensor = grav_fwd.clamp(min=0.0)  # ignore nose-up (negative) lean

    return climbing * fwd_lean


def position_drift_penalty(
    env: ManagerBasedRLEnv,
    drift_threshold: float = 0.10,
) -> torch.Tensor:
    """
    Penalise horizontal displacement of the robot base from its spawn position.

    Designed for the SpinStatic training environment (Stage 1 of the two-stage
    spin curriculum) where the robot should stand completely still while learning
    to lock its wheels via Kd braking.

    MOTIVATION — why we need this for Stage 1:
    -------------------------------------------
    model_9995.pt was trained on pyramid slope terrain.  When placed on flat
    terrain with vx=0, vy=0, omega=0 commands, the robot's learned priors will
    still produce residual wheel torques that cause small drifts.  Without a
    position-hold signal, the policy has no gradient to eliminate this drift —
    is_alive alone only penalises falling, not wandering.

    The drift threshold (default 0.10 m) creates a dead zone: the robot is free
    to make micro-adjustments (stance width, weight shifting) within 10 cm of
    spawn without penalty.  Drift beyond 10 cm earns a penalty proportional to
    the excess distance — forcing the policy to actively resist its wheel-drive
    priors.

    DEAD ZONE CALIBRATION:
    ----------------------
    0.10 m covers:
      • Normal stance sway from balance corrections: ±0.03–0.05 m  < threshold ✓
      • Random push event displacement: ±0.05–0.08 m               < threshold ✓  
      • Gravity-induced micro-drift on flat: ≈0 (no slope)         < threshold ✓
      • Wheel-drive prior residual drift: 0.10–0.30 m/s over time  > threshold ✗ → penalised ✓

    SPAWN POSITION TRACKING:
    -------------------------
    Isaac Lab resets each env independently and initialises `root_state_w` at
    the start of each episode.  We read the CURRENT root position and subtract
    the INITIAL position stored in `default_root_state`.

    Note: `default_root_state` is the per-env spawn position set by the terrain
    curriculum, not a fixed world origin.  This correctly handles the case where
    different envs spawn at different terrain tiles.

    COMBINATION WITH TRACK_LIN_VEL_XY_EXP (weight = -1.0):
    --------------------------------------------------------
    `track_lin_vel_xy_exp` penalises nonzero velocity (good for instantaneous
    velocity control).  `position_drift_penalty` penalises accumulated displacement
    (good for eliminating slow persistent drift that has near-zero instantaneous
    velocity but grows over time).  Both signals together provide:
      • Immediate feedback: don't start moving      (velocity penalty)
      • Long-term feedback: return if you did move  (position penalty)

    Parameters
    ----------
    env             : the running ManagerBasedRLEnv
    drift_threshold : horizontal displacement (m) below which no penalty is applied.
                      Default 0.10 m — covers normal balance corrections.

    Returns
    -------
    Tensor shape (num_envs,), value ≥ 0.0.
    Excess horizontal drift beyond threshold (metres).
    Multiply by a negative weight (−1.0 recommended) in RewardTermCfg.

    Example RewardTermCfg (in spin_static_env_cfg.py)::

        self.rewards.position_drift = RewTerm(
            func=position_drift_penalty,
            weight=-1.0,
            params={"drift_threshold": 0.10},
        )
    """
    robot = env.scene["robot"]

    # Current XY position in world frame
    pos_w: torch.Tensor = robot.data.root_pos_w[:, :2]  # (num_envs, 2)

    # ------------------------------------------------------------------
    # SPAWN POSITION — tracked via env.extras on the first step of each episode.
    #
    # WHY NOT default_root_state[:, :2]:
    #   default_root_state is the asset's canonical pose from the URDF definition
    #   (often the world origin [0, 0]). The terrain curriculum spawns the robot
    #   at different tile positions across the terrain grid — potentially 5-20 m
    #   from the URDF origin. Using default_root_state causes the penalty to fire
    #   at full strength from step 1 even on a perfectly stationary robot.
    #
    # CORRECT APPROACH — record actual spawn position on first step:
    #   episode_length_buf == 1 → first step of new episode → record root_pos_w.
    #   Subsequent steps → measure drift from the recorded spawn position.
    #   This correctly handles per-env per-episode spawn at arbitrary terrain tiles.
    # ------------------------------------------------------------------
    is_first_step: torch.Tensor = (env.episode_length_buf == 1)  # (num_envs,) bool

    if "drift_spawn_xy" not in env.extras:
        # Lazy initialisation on very first call (before any episode runs)
        env.extras["drift_spawn_xy"] = pos_w.clone()

    spawn_xy: torch.Tensor = env.extras["drift_spawn_xy"]  # (num_envs, 2)

    # On first step of each episode: update spawn position for those envs.
    # Other envs keep their existing spawn position from this episode's first step.
    spawn_xy = torch.where(
        is_first_step.unsqueeze(1).expand_as(spawn_xy),
        pos_w,
        spawn_xy,
    )
    env.extras["drift_spawn_xy"] = spawn_xy

    # Horizontal displacement from this episode's actual spawn position
    displacement: torch.Tensor = (pos_w - spawn_xy).norm(dim=-1)  # (num_envs,)

    # Dead zone: no penalty within drift_threshold metres of spawn.
    excess: torch.Tensor = (displacement - drift_threshold).clamp(min=0.0)

    return excess


def base_height_penalty(
    env: ManagerBasedRLEnv,
    min_height: float = 0.28,
) -> torch.Tensor:
    """
    Penalise the robot for lowering its body below a minimum standing height.

    PURPOSE — closing the crawl-spin exploit
    -----------------------------------------
    In turn training run 7, the robot discovered that lowering its body
    nose-down to ~0.10 m (from normal standing ~0.35 m) reduces rotational
    inertia around the vertical axis.  It then pivots around its nose with
    rear knees dragging on the ground.  This IS genuine rotation (heading_progress
    is real) but the posture is completely wrong for real-world slope operation.

    The body height is the most direct measurement of this posture:
      Normal standing:    base_z ≈ 0.35 m above terrain
      Crawl-spin posture: base_z ≈ 0.08–0.15 m above terrain

    This penalty fires when base height drops below min_height, with cost
    proportional to the excess drop — similar to the dead-zone threshold
    pattern used in hip_crossing_penalty.

    HEIGHT MEASUREMENT:
    -------------------
    base_z = robot.data.root_pos_w[:, 2] — world-frame Z of the base link
    terrain_z = estimated from root_pos_w at spawn (default_root_state[:, 2])

    For flat terrain: root_pos_w[:, 2] directly gives height above ground
    (terrain is at Z=0 or terrain_z = default_root_state[:, 2]).

    For slope terrain: the base height above LOCAL terrain is more complex.
    We use root_pos_w[:, 2] directly — this is the absolute world-frame Z,
    which is sufficient to detect the crawl posture because:
      - On flat terrain: normal standing Z ≈ 0.35 m, crawl Z ≈ 0.10 m
      - On 35° slope: spawn Z is higher than flat, but relative to terrain
        the body height is still measurable via the gravity-projected height
    For simplicity on flat terrain (Phase A training): raw world Z works.
    For slope-turn (Phase B): min_height can be reduced to 0.20 m.

    WEIGHT CALIBRATION:
    -------------------
    At normal standing (Z=0.35 m, threshold=0.28 m): excess = 0 → zero cost
    At mild stoop (Z=0.22 m): excess = 0.06 m → at weight=-20: cost=-1.2/step
    At crawl posture (Z=0.10 m): excess = 0.18 m → at weight=-20: cost=-3.6/step

    heading_progress at perfect 10°/s: ~0.175/step × weight=80 = 14/step
    crawl saves inertia but costs -3.6/step → still profitable at large weight

    COMBINE WITH flat_orientation_l2=-3.0:
    The body pitched 60° nose-down also triggers orientation penalty heavily:
      -3.0 × (60° in rad)² = -3.0 × 1.097 = -3.29/step
    Combined with base_height at -3.6/step: total posture penalty = -6.9/step
    heading_progress at perfect = 14/step → crawl gives 14 - 6.9 = +7.1/episode
    upright gives 14 - 0 = 14/episode → upright is 2× more profitable ✓

    Parameters
    ----------
    env        : the running ManagerBasedRLEnv
    min_height : minimum acceptable base height above terrain (metres).
                 Default 0.28 m — below normal standing (0.35 m) but well
                 above crawl posture (0.10 m).

    Returns
    -------
    Tensor shape (num_envs,), value ≥ 0.0.
    Excess height drop below threshold (metres).
    Multiply by a negative weight (-20.0 recommended) in RewardTermCfg.
    """
    robot = env.scene["robot"]

    # World-frame base Z height
    base_z: torch.Tensor = robot.data.root_pos_w[:, 2]  # (num_envs,)

    # ------------------------------------------------------------------
    # TERRAIN Z REFERENCE — tracked via env.extras on the first step.
    #
    # WHY NOT default_root_state[:, 2]:
    #   Same bug as position_drift_penalty: default_root_state is the URDF
    #   canonical pose (Z=0 or asset origin), not the per-episode spawn Z.
    #   On sloped terrain the robot spawns at elevated Z (e.g., 2-5 m above
    #   the URDF origin). Using default_root_state would give
    #   height_above_terrain = base_z - 0.0 = base_z (world Z), which for
    #   a robot spawned at tile Z=3.0 m never triggers min_height=0.28 m.
    #
    # CORRECT APPROACH — record actual spawn Z on first step:
    #   The terrain surface under the robot at spawn is approximately
    #   root_pos_w[:, 2] - standing_height (≈ 0.35 m). But since we want
    #   the height ABOVE the terrain at any point (not just spawn), and the
    #   terrain is roughly flat within a tile, we use spawn_z as the reference:
    #     height_above_terrain ≈ base_z - spawn_z + 0.35
    #   where 0.35 is the normal standing height at spawn.
    #
    #   Simpler: just measure height_above_terrain = base_z - spawn_z + spawn_height
    #   where spawn_height = root_pos_w[:, 2] at first step (which already includes
    #   the 0.35 m standing height from the ground).
    #
    #   Actually simplest: record base_z at spawn (first step), treat that as the
    #   reference height for normal standing. Then penalise when current base_z
    #   drops more than (spawn_z - min_height) below spawn.
    #
    #   Concretely: excess = (spawn_base_z - min_height) - base_z, clamped ≥ 0.
    #   This fires when base_z drops more than (0.35 - 0.28) = 0.07 m below spawn.
    # ------------------------------------------------------------------
    is_first_step: torch.Tensor = (env.episode_length_buf == 1)  # (num_envs,) bool

    if "height_spawn_z" not in env.extras:
        env.extras["height_spawn_z"] = base_z.clone()

    spawn_base_z: torch.Tensor = env.extras["height_spawn_z"]  # (num_envs,)

    # Update on first step of each episode
    spawn_base_z = torch.where(is_first_step, base_z, spawn_base_z)
    env.extras["height_spawn_z"] = spawn_base_z

    # Height below the spawn standing height (positive = robot has dropped)
    height_drop: torch.Tensor = spawn_base_z - base_z  # (num_envs,) ≥ 0 when dropped

    # Dead zone: no cost for drops ≤ (standing_height - min_height) = ~0.07 m
    # At normal stepping, body may dip 2-3 cm → no cost.
    # At crawl posture, body drops ~0.25 m → large cost.
    allowed_drop = 0.35 - min_height  # ~0.07 m dead zone
    excess_drop: torch.Tensor = (height_drop - allowed_drop).clamp(min=0.0)

    return excess_drop


def foot_alternation_reward(
    env: ManagerBasedRLEnv,
    asset_cfg=None,
    omega_threshold: float = 0.05,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """
    Reward lifting 1-2 feet off the ground during an active turn command.

    PURPOSE — breaking the "all-4-wheels-glued" binding exploit
    -----------------------------------------------------------
    After 10 training runs, the robot consistently "arcs and binds" during
    pivot turn commands — it twists the body at the hip/torso while keeping
    all 4 wheels on the ground. This is because:

    1. The rough policy prior says "4 wheels on ground = stable = good"
    2. Every reward term in the current structure is either neutral or
       negative about foot lift. Nothing DIRECTLY rewards it.
    3. Foot lift during a step causes transient body tilt, which fires
       flat_orientation_l2 even at weight=-0.3.

    For a genuine pivot turn, the robot MUST lift 1-2 feet at a time and
    swing them in arcs. This is the physically correct gait — same as a
    dog turning in place. The smooth stepping gait requires:
      - Lift one foot (or a diagonal pair)
      - Swing it in the turn direction
      - Plant it, then lift the next

    This function directly rewards the correct intermediate states:

      feet_off_ground = 0  (binding — all wheels glued during turn):
          Returns -1.0 → at weight=+1.5: -1.5/step  [punish binding]

      feet_off_ground = 1  (single foot lifted — correct single-step):
          Returns +1.0 → at weight=+1.5: +1.5/step  [reward stepping]

      feet_off_ground = 2  (diagonal pair lifted — trot stepping):
          Returns +0.5 → at weight=+1.5: +0.75/step [reward trot gait]

      feet_off_ground >= 3 (too many feet up — unstable):
          Returns -2.0 → at weight=+1.5: -3.0/step  [punish instability]

      omega_cmd < threshold (no turn command active):
          Returns  0.0 → [don't interfere with standing/walking gaits]

    CONTACT SENSOR USAGE:
    ---------------------
    Uses the existing `contact_forces` sensor (already in the scene for
    `undesired_contacts`). For each foot link, checks if the contact force
    magnitude exceeds `contact_threshold` (default 1.0 N). Below threshold
    = foot is off the ground (or just brushing).

    The asset_cfg should specify the FOOT links (wheel hubs), not the
    calf/thigh/hip links used by undesired_contacts.

    WEIGHT CALIBRATION:
    -------------------
    At weight=+1.5:
      - Binding (all 4 wheels down) during turn: -1.5/step
      - Single foot lifted: +1.5/step
      - Perfect pivot (alternating single lifts): +1.5/step average
      - heading_progress at perfect: ~0.22/step × 80 = ~17.6/step (dominant)
      - foot_alternation adds +1.5/step = +8.5% improvement to dominant signal

    The weight is intentionally moderate — foot_alternation is a SHAPING
    signal, not the primary objective. heading_progress remains dominant.

    WHY THIS WASN'T NEEDED FOR LATERAL WALKING:
    --------------------------------------------
    For vy commands (lateral stepping), foot lift is implicitly required —
    the robot physically cannot step sideways without lifting a foot first.
    The reward signal for "match vy_cmd" already creates the gradient.
    For pivot turns (vx=0, vy=0, omega_cmd), there IS no translation
    command — the robot can satisfy "no drift" by staying still with all
    4 wheels planted and just twisting. This function adds the missing
    gradient that says "during a turn, MOVE YOUR FEET."

    Parameters
    ----------
    asset_cfg        : SceneEntityCfg with body_names resolved to foot links.
                       Use body_names=[".*_foot"] for the wheel hub links.
    omega_threshold  : minimum |omega_cmd| to activate the reward (rad/s).
                       Default 0.05 — ignores near-zero yaw commands.
    contact_threshold: contact force (N) below which foot is "off ground".
                       Default 1.0 N — filters out brush contacts.

    Returns
    -------
    Tensor shape (num_envs,).
    Values: -2.0 (too many feet up), +1.0 (one foot up), +0.5 (two up),
            -1.0 (zero feet up / binding), 0.0 (no turn command).
    Multiply by a POSITIVE weight (+1.5 recommended) in RewardTermCfg.
    """
    if asset_cfg is None:
        raise ValueError(
            "foot_alternation_reward requires asset_cfg with body_names=['.*_foot']"
        )

    # --- Gate: only active during turn commands ---
    omega_cmd: torch.Tensor = env.command_manager.get_command("base_velocity")[:, 2]
    turn_active: torch.Tensor = (omega_cmd.abs() > omega_threshold)  # (num_envs,) bool

    # --- Count feet on the ground via contact_forces sensor ---
    # Isaac Lab ContactSensor: env.scene["contact_forces"]
    # asset_cfg is SceneEntityCfg("contact_forces", body_names=[...])
    # After env startup, asset_cfg.body_ids contains the resolved body indices.
    # net_forces_w shape: (num_envs, num_bodies, 3) — ALL bodies in the sensor.
    # We slice to just the foot body indices.
    contact_sensor = env.scene[asset_cfg.name]  # ContactSensor

    # net_forces_w: (num_envs, num_bodies, 3) — world-frame contact force per body
    foot_forces: torch.Tensor = contact_sensor.data.net_forces_w[
        :, asset_cfg.body_ids, :
    ]  # (num_envs, 4, 3)

    # Force magnitude per foot: (num_envs, 4)
    foot_force_mag: torch.Tensor = foot_forces.norm(dim=-1)

    # Binary: foot on ground = force > threshold
    foot_on_ground: torch.Tensor = (foot_force_mag > contact_threshold).float()  # (num_envs, 4)

    # Number of feet on the ground (0-4)
    feet_on_ground: torch.Tensor = foot_on_ground.sum(dim=-1)  # (num_envs,)
    feet_off: torch.Tensor = 4.0 - feet_on_ground  # (num_envs,)

    # --- Reward signal based on feet off ground ---
    # Binding (0 off): -1.0
    binding = (feet_off == 0).float()
    # Single step (1 off): +1.0
    single_lift = (feet_off == 1).float()
    # Trot pair (2 off): +0.5
    trot_lift = (feet_off == 2).float()
    # Unstable (3+ off): -2.0
    unstable = (feet_off >= 3).float()

    reward: torch.Tensor = (
        -1.0 * binding
        + 1.0 * single_lift
        + 0.5 * trot_lift
        - 2.0 * unstable
    )  # (num_envs,)

    # Zero out when no turn command is active
    reward = reward * turn_active.float()

    return reward


def yaw_stagnation_penalty(
    env: ManagerBasedRLEnv,
    window_steps: int = 100,
    min_yaw_deg: float = 5.0,
    min_cmd: float = 0.1,
) -> torch.Tensor:
    """
    Penalise the robot when it fails to accumulate meaningful yaw over a rolling window.

    PURPOSE — closing the asymmetric rocking exploit (run 11)
    ----------------------------------------------------------
    Run 11 discovered that the robot satisfies heading_progress by rocking
    its body asymmetrically: it oscillates slightly more in the commanded
    turn direction than back. Per-step Δyaw is tiny (~0.002 rad/step) but
    positive, so heading_progress collects reward. Over 1000 steps this
    produces only ~2 rad (114°) of apparent yaw progress — but because
    oscillations partially cancel, the ACTUAL net rotation may be much less.

    This function measures total yaw accumulated over the last `window_steps`
    steps. If the robot has genuinely turned, total_yaw_accumulated will be
    near window_steps × cmd_rate × dt = 100 × 0.5 × 0.02 = 1.0 rad in 2 seconds.
    If it's rocking, most of the yaw cancels and total_yaw_accumulated ≈ 0.1 rad.

    The penalty fires when total yaw accumulated falls below min_yaw_deg
    (in degrees) after window_steps steps of a turn command. This directly
    makes the rocking exploit unprofitable: the robot must accumulate real
    monotonic yaw, not just instantaneous Δyaw that partially cancels.

    IMPLEMENTATION — rolling yaw accumulator:
    -----------------------------------------
    We store a running sum of per-step Δyaw in `env.extras["yaw_window_sum"]`
    and a step counter in `env.extras["yaw_window_count"]`. Every step:
      1. Compute Δyaw (same as heading_progress — actual quaternion yaw change)
      2. Add to running sum (separate per env)
      3. Every window_steps steps: check if sum > min_yaw_threshold
         - If NO: return penalty (-1.0) for EACH step in that window
         - If YES: return 0 (no penalty)
         - Reset sum to 0 and counter to 0

    WEIGHT CALIBRATION:
    -------------------
    At weight=-2.0 and firing every step of a 100-step window:
      Total penalty per window if stagnating: -2.0 × 100 = -200
      heading_progress at rocking: +0.30 × 100 = +30 (rough upper bound)
      Net at rocking: +30 - 200 = -170/window → rocking is VERY unprofitable

      heading_progress at genuine turn: +0.30 × 100 = +30
      yaw_stagnation at genuine turn: 0 (threshold met)
      Net at genuine turn: +30/window → strongly positive

    The asymmetry (−170 vs +30) makes rocking completely non-viable.

    GATING:
    -------
    Only fires when |omega_cmd| > min_cmd (default 0.1 rad/s).
    At omega=0 (no turn command) the robot should stand still —
    no penalty for zero yaw accumulation when not commanded to turn.

    Parameters
    ----------
    window_steps : number of steps over which to measure yaw accumulation.
                   Default 100 = 2 seconds at 50 Hz.
    min_yaw_deg  : minimum yaw accumulated in window_steps to avoid penalty.
                   Default 5.0° = 0.0873 rad.
                   At perfect 0.5 rad/s × 2s = 1.0 rad accumulated.
                   5° threshold accepts 5% efficiency (rocking achieves ~0-2%).
    min_cmd      : minimum |omega_cmd| to activate the penalty (rad/s).

    Returns
    -------
    Tensor shape (num_envs,), value 0.0 or 1.0.
    Returns 1.0 when stagnating (yaw < threshold after window).
    Multiply by a NEGATIVE weight (-2.0 recommended).
    """
    robot = env.scene["robot"]

    # --- Current yaw from quaternion ---
    quat = robot.data.root_quat_w  # (N, 4) [w, x, y, z]
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw_now: torch.Tensor = torch.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )  # (N,) in [-π, π]

    # --- Omega command gate ---
    omega_cmd: torch.Tensor = env.command_manager.get_command("base_velocity")[:, 2]
    turn_active: torch.Tensor = (omega_cmd.abs() > min_cmd).float()  # (N,)

    # --- Threshold in radians ---
    min_yaw_rad = min_yaw_deg * (torch.pi / 180.0)

    # --- Initialize extras on first call ---
    is_first_step: torch.Tensor = (env.episode_length_buf == 1)  # (N,) bool

    if "yaw_stag_prev_yaw" not in env.extras:
        env.extras["yaw_stag_prev_yaw"] = yaw_now.clone()
        env.extras["yaw_stag_sum"] = torch.zeros(
            env.num_envs, device=yaw_now.device, dtype=yaw_now.dtype
        )
        env.extras["yaw_stag_count"] = torch.zeros(
            env.num_envs, device=yaw_now.device, dtype=torch.long
        )

    prev_yaw: torch.Tensor = env.extras["yaw_stag_prev_yaw"]
    yaw_sum: torch.Tensor = env.extras["yaw_stag_sum"]
    step_count: torch.Tensor = env.extras["yaw_stag_count"]

    # Reset on first step of each episode
    yaw_sum = torch.where(is_first_step, torch.zeros_like(yaw_sum), yaw_sum)
    step_count = torch.where(is_first_step, torch.zeros_like(step_count), step_count)
    prev_yaw = torch.where(is_first_step, yaw_now, prev_yaw)

    # --- Per-step Δyaw in correct-direction ---
    delta_yaw = yaw_now - prev_yaw
    # wrap to [-π, π]
    delta_yaw = (delta_yaw + torch.pi) % (2 * torch.pi) - torch.pi
    # signed progress in commanded direction (positive = correct direction)
    signed_progress = delta_yaw * omega_cmd.sign()
    # only accumulate when turn is active
    signed_progress = signed_progress * turn_active

    # --- Update rolling window ---
    yaw_sum = yaw_sum + signed_progress
    step_count = step_count + turn_active.long()

    # --- Check window every window_steps ---
    window_done: torch.Tensor = (step_count >= window_steps)  # (N,) bool

    # Penalty fires when window complete AND yaw sum below threshold
    stagnating: torch.Tensor = (
        window_done & (yaw_sum < min_yaw_rad)
    ).float()  # (N,)

    # Reset window when done
    yaw_sum = torch.where(window_done, torch.zeros_like(yaw_sum), yaw_sum)
    step_count = torch.where(window_done, torch.zeros_like(step_count), step_count)

    # --- Save state ---
    env.extras["yaw_stag_prev_yaw"] = yaw_now.clone()
    env.extras["yaw_stag_sum"] = yaw_sum
    env.extras["yaw_stag_count"] = step_count

    # --- Diagnostic: track cumulative net yaw per episode ---
    # Logged to env.extras["net_yaw_deg"] for tensorboard visibility.
    # This is the ONLY metric that cannot be gamed by joint-winding:
    # - Genuine spin 0.3 rad/s × 20s × (180/π) = 344° per 1000-step episode
    # - Joint-wind / static: <20° per episode (joint limits prevent more)
    # Isaac Lab logs env.extras["net_yaw_deg"] automatically as
    # Episode_Extras/net_yaw_deg in tensorboard.
    if "net_yaw_episode" not in env.extras:
        env.extras["net_yaw_episode"] = torch.zeros(
            env.num_envs, device=yaw_now.device, dtype=yaw_now.dtype
        )
    net_yaw: torch.Tensor = env.extras["net_yaw_episode"]
    # Accumulate per-step yaw (use the same signed_progress already computed above)
    net_yaw = torch.where(is_first_step, torch.zeros_like(net_yaw), net_yaw + signed_progress.abs())
    env.extras["net_yaw_episode"] = net_yaw
    # Expose as degrees for human readability
    env.extras["net_yaw_deg"] = (net_yaw * (180.0 / torch.pi)).mean().item()

    return stagnating


def trunk_stability_penalty(
    env: ManagerBasedRLEnv,
    max_tilt_deg: float = 15.0,
) -> torch.Tensor:
    """
    Penalise trunk pitch and roll beyond a tight dead-zone during a turn command.

    PURPOSE — eliminate the trunk-wind / nose-dive exploit
    -------------------------------------------------------
    Runs 12-16 all show the same exploit: the robot winds its hip joints to
    rotate the trunk (base link) into a severely tilted pose — nose-down, or
    rolled 40-70° sideways. The trunk rotation registers as yaw on the IMU
    (track_ang_vel_z_exp fires) and as quaternion yaw change (heading_progress
    fires), but the FEET DO NOT MOVE. The robot is not spinning its heading —
    it's deforming its body shape.

    The existing `flat_orientation_l2` term uses L2 over the full gravity
    projection, which is quadratic and costs very little at moderate tilt.
    At weight=-0.3, a 60° tilt only costs -0.33/step — far less than the
    +1.44/step gained from `track_ang_vel_z_exp`. The twisted pose is profitable.

    This function adds a THRESHOLD-based penalty that:
      - Is completely FREE within ±max_tilt_deg (15°) of upright
      - Costs steeply beyond the threshold
      - Only fires during active turn commands (not during walking/standing)

    At weight=-5.0:
      Normal step swing (10° tilt): excess = 0 → zero cost ✓
      Moderate lean (20° tilt):     excess = 5° = 0.087 rad → -0.44/step
      Nose-down twist (60° tilt):   excess = 45° = 0.785 rad → -3.93/step

    Combined with flat_orientation_l2=-3.0 at 60° tilt:
      Total posture penalty = -3.29 + -3.93 = -7.22/step
      track_ang_vel_z_exp gain at twisted pose ≈ +1.44/step
      Net = -5.78/step → COMPLETELY UNPROFITABLE ✓

    Upright genuine spin:
      flat_orientation_l2 ≈ 0, trunk_stability ≈ 0
      track_ang_vel_z_exp = +2.0/step at perfect 0.3 rad/s
      Net = +2.0/step → clearly the winning strategy ✓

    IMPLEMENTATION:
    ---------------
    `projected_gravity_b` is a unit vector of (0,0,-1) projected into the
    body frame. When the body is perfectly upright:
        projected_gravity_b = [0, 0, -1]  → no tilt
    When the body tilts nose-down by θ:
        projected_gravity_b[0] = sin(θ)   → forward component
    When the body rolls right by θ:
        projected_gravity_b[1] = sin(θ)   → lateral component

    The tilt angle magnitude is:
        sin(total_tilt) ≈ sqrt(grav_xy_sq) for small angles
        total_tilt_rad = arcsin(sqrt(grav_b[:,0]² + grav_b[:,1]²))

    We use the sin directly (grav_b[:,0]² + grav_b[:,1]²) as the tilt
    magnitude squared, which is simpler and differentiable everywhere.

    Dead zone in radians:
        max_tilt_rad = max_tilt_deg × π/180
        sin(15°) = 0.259
    Beyond the threshold, penalty = sqrt(grav_xy_sq) - sin(max_tilt_rad), clamped ≥ 0.

    Parameters
    ----------
    env          : the running ManagerBasedRLEnv
    max_tilt_deg : trunk tilt (degrees) below which no penalty is applied.
                   Default 15° — covers normal stepping tilt (±10°) with 5° margin.
                   Nose-down/side-over exploits are 40-70° — clearly above threshold.

    Returns
    -------
    Tensor shape (num_envs,), value ≥ 0.0.
    Excess trunk tilt beyond threshold (in sin units, ≈ radians for small angles).
    Multiply by a negative weight (-5.0 recommended) in RewardTermCfg.

    Example RewardTermCfg (in turn_env_cfg.py)::

        self.rewards.trunk_stability = RewTerm(
            func=trunk_stability_penalty,
            weight=-5.0,
            params={"max_tilt_deg": 15.0},
        )
    """
    robot = env.scene["robot"]

    # projected_gravity_b: (num_envs, 3) — gravity in body frame
    # [0] = forward/pitch component (nose-down = positive)
    # [1] = lateral/roll component (roll right = positive)
    # [2] = vertical component (≈ -1 when upright)
    grav_b: torch.Tensor = robot.data.projected_gravity_b  # (N, 3)

    # XY magnitude = sin(tilt_angle) ≈ tilt_angle in radians for small angles
    grav_xy_sq: torch.Tensor = grav_b[:, 0].pow(2) + grav_b[:, 1].pow(2)  # (N,)
    tilt_sin: torch.Tensor = grav_xy_sq.sqrt()  # (N,) = sin(tilt_angle)

    # Threshold in sin units
    max_tilt_rad = max_tilt_deg * (torch.pi / 180.0)
    max_tilt_sin = torch.sin(torch.tensor(max_tilt_rad, device=tilt_sin.device))

    # Dead-zone: no cost within ±max_tilt_deg
    excess: torch.Tensor = (tilt_sin - max_tilt_sin).clamp(min=0.0)  # (N,)

    # Gate: only penalise during active turn commands
    omega_cmd: torch.Tensor = env.command_manager.get_command("base_velocity")[:, 2]
    turn_active: torch.Tensor = (omega_cmd.abs() > 0.05).float()

    return excess * turn_active


def heading_progress(
    env: ManagerBasedRLEnv,
    min_cmd: float = 0.05,
    max_rate_scale: float | None = None,
    step_dt: float = 0.02,
) -> torch.Tensor:
    """
    Reward actual heading displacement in the commanded yaw direction.

    PURPOSE — closing the body-rocking exploit
    -------------------------------------------
    ``track_ang_vel_z_exp`` rewards INSTANTANEOUS yaw rate from the IMU.
    A robot can satisfy this by rocking its body side-to-side, generating
    oscillating ±ω_z IMU readings without ever moving its feet.  Net heading
    change over any window = zero.  The exploit looks perfect in the stats
    (high track_ang_vel_z_exp, low bad_orientation) but visually the robot
    never actually spins.

    This function rewards the ACTUAL change in heading angle:
        Δyaw = sign(omega_cmd) × (current_yaw − prev_yaw)
    Rocking: current_yaw oscillates ±, so Δyaw alternates ±0.0035 → clamped to 0
             half the time → ~0.5× the maximum reward.
    Genuine turn: current_yaw monotonically increases → Δyaw always positive
                  → full reward every step.

    The function uses ``env.episode_length_buf`` to detect episode reset
    (length == 1 = first step of new episode) and skips the Δyaw computation
    on that step to avoid stale prev_yaw values.

    STATELESS IMPLEMENTATION (no persistent tensors):
    --------------------------------------------------
    We cannot store state in the function itself (called fresh each step).
    Instead we store ``prev_yaw`` in ``env.extras`` — a dict that persists
    within an episode and is accessible across reward function calls.
    Isaac Lab's ManagerBasedRLEnv initialises env.extras = {} at startup;
    we lazily initialise our key on first call.

    MATH:
        quaternion → yaw via atan2(2(wz+xy), 1−2(y²+z²))
        Δyaw = sign(omega_cmd) × wrap_to_pi(yaw_now − yaw_prev)
        reward = clamp(Δyaw, min=0.0)   [only reward correct-direction turning]

    wrap_to_pi handles the ±π discontinuity: if the robot crosses the ±180°
    boundary the naive difference is ±2π; wrapping gives the correct small Δ.

    Parameters
    ----------
    env       : the running ManagerBasedRLEnv
    min_cmd   : minimum |omega_cmd| to activate the reward (rad/s).
                Default 0.05 — ignores tiny residual commands from the
                command sampler near zero.
    max_rate_scale : if set > 0, cap per-step reward at
                |omega_cmd| * step_dt * max_rate_scale so one-shot twists
                cannot bank unlimited Δyaw in a single step. None = uncapped.
    step_dt   : control dt (s) used with max_rate_scale. Default 0.02.

    Returns
    -------
    Tensor shape (num_envs,), value ≥ 0.0.
    Expected magnitude: ~0.0035 rad per step at 10°/s (0.175 rad/s ÷ 50 Hz).
    Multiply by a large positive weight (+50.0 recommended) in RewardTermCfg
    so the per-step contribution (~0.175) competes with track_ang_vel_z_exp
    (~2.7/step at convergence).
    """
    robot = env.scene["robot"]

    # --- Extract current yaw from base quaternion (world frame) ---
    # root_quat_w: (num_envs, 4) as [w, x, y, z] in Isaac Lab convention
    quat = robot.data.root_quat_w  # (N, 4)
    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]
    yaw_now: torch.Tensor = torch.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )  # (N,) in [-π, π]

    # --- Omega command gate ---
    omega_cmd: torch.Tensor = env.command_manager.get_command("base_velocity")[:, 2]
    active: torch.Tensor = (omega_cmd.abs() > min_cmd).float()  # (N,)

    # --- First step of episode: initialise prev_yaw, return zero reward ---
    is_first_step: torch.Tensor = (env.episode_length_buf == 1)  # (N,) bool

    if "turn_prev_yaw" not in env.extras:
        # Lazy initialisation on very first call (before any episode runs)
        env.extras["turn_prev_yaw"] = yaw_now.clone()

    prev_yaw: torch.Tensor = env.extras["turn_prev_yaw"]  # (N,)

    # --- Compute heading displacement ---
    # wrap_to_pi: keep difference in [-π, π] to handle the ±180° crossing
    delta_yaw = yaw_now - prev_yaw
    delta_yaw = (delta_yaw + torch.pi) % (2 * torch.pi) - torch.pi  # wrap to [-π, π]

    # Signed heading progress: positive when turning in commanded direction
    signed_progress: torch.Tensor = delta_yaw * omega_cmd.sign()

    # Only reward correct-direction turning (clamp negatives to 0)
    reward: torch.Tensor = signed_progress.clamp(min=0.0)

    # Cap per-step pay near commanded micro-rate (anti one-shot twist).
    # When max_rate_scale is None/0: uncapped (Pulse20 default behavior).
    if max_rate_scale is not None and float(max_rate_scale) > 0.0:
        max_dyaw = omega_cmd.abs() * float(step_dt) * float(max_rate_scale)
        reward = torch.minimum(reward, max_dyaw)

    # Zero out on first step (prev_yaw not valid yet)
    reward = reward * (~is_first_step).float()

    # Apply gate: only reward when command is active
    reward = reward * active

    # --- Update prev_yaw: use yaw_now for all envs; first-step envs also update ---
    env.extras["turn_prev_yaw"] = yaw_now.clone()

    return reward



def heading_error_l1(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    deadzone_rad: float = 0.05,
) -> torch.Tensor:
    """
    Penalise |heading_target - heading_now| (error-driven turn pressure).

    PURPOSE (S25-v3 / steep slope)
    ------------------------------
    Continuous open-loop ω bursts are the wrong interface on slopes. The
    viable skill is: hold plant → step to reduce heading error → rebalance
    when error is small / body is unstable.

    This term makes **waiting with residual heading error expensive**. Error
    accumulates in the return every step until the policy reduces it. When
    |error| is inside deadzone, penalty is zero so the robot can rebalance
    without artificial settle timers.

    Requires heading_command=True command term with ``heading_target``
    (UniformVelocityCommand / RelativeHeadingVelocityCommand).

    Standing envs (is_standing_env) are zeroed — no goal pressure while
    explicitly commanded to stand.

    Returns
    -------
    (num_envs,) >= 0. Multiply by NEGATIVE weight (e.g. -2.0).
    """
    n = env.num_envs
    device = env.device
    zero = torch.zeros(n, device=device)

    try:
        term = env.command_manager.get_term(command_name)
    except Exception:
        return zero

    if not getattr(term.cfg, "heading_command", False):
        return zero
    if not hasattr(term, "heading_target"):
        return zero

    robot = env.scene[term.cfg.asset_name]
    err = term.heading_target - robot.data.heading_w
    err = (err + torch.pi) % (2.0 * torch.pi) - torch.pi
    abs_err = err.abs()

    # Deadzone: no pressure when close enough — natural rebalance window
    excess = (abs_err - float(deadzone_rad)).clamp(min=0.0)

    active = torch.ones(n, device=device)
    if hasattr(term, "is_standing_env"):
        active = (~term.is_standing_env).float()

    return excess * active


def heading_error_reduction(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    deadzone_rad: float = 0.05,
) -> torch.Tensor:
    """
    Reward *reduction* in |heading error| since last step.

    Complements heading_error_l1: paying only when error actually shrinks
    (real yaw toward target), not when rocking in place (track_ang exploit).

    Returns
    -------
    (num_envs,) >= 0. Multiply by POSITIVE weight (e.g. +5.0).
    """
    n = env.num_envs
    device = env.device
    zero = torch.zeros(n, device=device)

    try:
        term = env.command_manager.get_term(command_name)
    except Exception:
        return zero

    if not getattr(term.cfg, "heading_command", False):
        return zero
    if not hasattr(term, "heading_target"):
        return zero

    robot = env.scene[term.cfg.asset_name]
    err = term.heading_target - robot.data.heading_w
    err = (err + torch.pi) % (2.0 * torch.pi) - torch.pi
    abs_err = err.abs()

    key = "prev_heading_abs_err"
    if key not in env.extras:
        env.extras[key] = abs_err.clone()

    prev = env.extras[key]
    is_first = env.episode_length_buf <= 1
    reduction = (prev - abs_err).clamp(min=0.0)
    reduction = reduction * (~is_first).float()

    # No credit inside deadzone churn
    reduction = reduction * (abs_err > float(deadzone_rad)).float()

    if hasattr(term, "is_standing_env"):
        reduction = reduction * (~term.is_standing_env).float()

    env.extras[key] = abs_err.clone()
    return reduction


def pivot_step_coordination(
    env: ManagerBasedRLEnv,
    contact_cfg=None,
    robot_body_cfg=None,
    min_swing_vel: float = 0.03,
    omega_threshold: float = 0.05,
) -> torch.Tensor:
    """
    Reward feet that step in the correct lateral direction for a pivot turn.

    PURPOSE — the missing coordination signal
    ------------------------------------------
    foot_alternation_reward rewards HOW MANY feet are lifted, but has zero
    knowledge of WHICH direction each foot moves. The robot can earn the full
    foot_alternation reward by lifting a foot and wiggling it randomly.

    A natural pivot turn requires:
      Left turn (omega > 0):
        Front feet (FL, FR) → step RIGHTWARD (+y in body frame)
        Rear feet  (RL, RR) → step LEFTWARD  (-y in body frame)
      Right turn (omega < 0): signs reversed.

    This is the scissor/tank-tread pattern. It generates yaw rotation entirely
    within normal joint range — no joint limits needed. The robot is currently
    using a "torque wind-up" exploit (arch body to max limits, release) because
    there is no reward signal that says "move this specific foot in this direction."

    IMPLEMENTATION:
    ---------------
    Two separate SceneEntityCfg are required:
      contact_cfg    → contact_forces sensor, body_names=foot links (for swing detection)
      robot_body_cfg → robot articulation, body_names=foot links (for body velocity)
    Both must list feet in the SAME order: [FL_foot, FR_foot, RL_foot, RR_foot].

    For each foot in SWING phase (off ground, detected via contact_forces):
      Measure its lateral velocity in the BODY FRAME.
      Front feet: reward if vy_b × sign(omega_cmd) > 0  (stepping outward)
      Rear feet:  reward if vy_b × sign(omega_cmd) < 0  (stepping outward from rear)

    The sign convention for "correct lateral step":
      foot_sign = [+1, +1, -1, -1] for [FL, FR, RL, RR] during left turn
    So correct_swing = foot_vy_b × foot_sign × sign(omega_cmd) > min_swing_vel

    GATING:
    -------
    - Only fires when |omega_cmd| > omega_threshold (default 0.05; slope-turn
      overrides use 0.04 so gates still fire at slow ω clips ±0.06 / ±0.08)
    - Only rewards feet actually in swing (contact force < 1N)
    - Only rewards velocity exceeding min_swing_vel (filters micro-vibration)

    WEIGHT CALIBRATION:
    -------------------
    At weight=+2.0, perfect coordination (2 feet in swing at correct vel):
      +2.0 × (2 × 1.0) = +4.0/step — comparable to heading_progress at +0.48

    Parameters
    ----------
    contact_cfg   : SceneEntityCfg("contact_forces", body_names=[...]) for swing detection.
    robot_body_cfg: SceneEntityCfg("robot", body_names=[...]) for foot body velocities.
    min_swing_vel : minimum lateral velocity (m/s) to count as an intentional step.
    omega_threshold: minimum |omega_cmd| to activate (rad/s). Default 0.05.

    Returns
    -------
    Tensor shape (num_envs,), value ≥ 0.0.
    """
    if contact_cfg is None or robot_body_cfg is None:
        raise ValueError(
            "pivot_step_coordination requires both contact_cfg and robot_body_cfg"
        )

    robot = env.scene["robot"]
    contact_sensor = env.scene[contact_cfg.name]

    # --- Gate: only active during turn commands ---
    omega_cmd: torch.Tensor = env.command_manager.get_command("base_velocity")[:, 2]
    turn_active: torch.Tensor = (omega_cmd.abs() > omega_threshold)  # (N,) bool
    omega_sign: torch.Tensor = omega_cmd.sign()  # +1 left, -1 right, 0 neutral

    # --- Identify swing feet via contact sensor ---
    foot_forces: torch.Tensor = contact_sensor.data.net_forces_w[
        :, contact_cfg.body_ids, :
    ]  # (N, 4, 3)
    foot_force_mag: torch.Tensor = foot_forces.norm(dim=-1)  # (N, 4)
    in_swing: torch.Tensor = (foot_force_mag < 1.0).float()  # (N, 4)

    # --- Lateral velocity of each foot in body frame ---
    # body_lin_vel_w: world-frame linear velocity of each rigid body in the articulation.
    # Indexed by robot body indices (different from contact sensor body indices).
    foot_vel_w: torch.Tensor = robot.data.body_lin_vel_w[
        :, robot_body_cfg.body_ids, :
    ]  # (N, 4, 3)

    # Rotate world-frame foot velocities into robot body frame.
    base_quat: torch.Tensor = robot.data.root_quat_w  # (N, 4)
    N = foot_vel_w.shape[0]
    foot_vel_w_flat = foot_vel_w.reshape(N * 4, 3)
    base_quat_rep = base_quat.unsqueeze(1).expand(-1, 4, -1).reshape(N * 4, 4)
    foot_vel_b_flat = quat_apply_inverse(base_quat_rep, foot_vel_w_flat)
    foot_vel_b = foot_vel_b_flat.reshape(N, 4, 3)

    foot_vy_b: torch.Tensor = foot_vel_b[:, :, 1]  # (N, 4) — lateral body-frame velocity

    # --- Correct direction sign per foot ---
    # For left turn (omega > 0):
    #   FL (+1): should step right (+y) → correct if vy_b > 0
    #   FR (+1): should step right (+y) → correct if vy_b > 0
    #   RL (-1): should step left  (-y) → correct if vy_b < 0
    #   RR (-1): should step left  (-y) → correct if vy_b < 0
    # foot_dir: [FL, FR, RL, RR] = [+1, +1, -1, -1]
    foot_dir = torch.tensor([1.0, 1.0, -1.0, -1.0],
                             device=foot_vy_b.device, dtype=foot_vy_b.dtype)  # (4,)

    # Signed correctness: positive = stepping in correct direction
    # omega_sign: (N,) → expand to (N, 4)
    correct_vel: torch.Tensor = foot_vy_b * foot_dir.unsqueeze(0) * omega_sign.unsqueeze(1)
    # (N, 4) — positive when foot moves correctly for commanded turn direction

    # --- Reward: swing feet with correct lateral velocity above threshold ---
    # Only reward feet that are actually swinging AND moving correctly
    swing_correct: torch.Tensor = (correct_vel > min_swing_vel).float()  # (N, 4)
    swing_correct = swing_correct * in_swing  # only count feet that are off ground

    # Sum across all 4 feet — max 4.0 per step (all feet swinging correctly)
    coord_score: torch.Tensor = swing_correct.sum(dim=-1)  # (N,)

    # Zero out when no turn command is active
    coord_score = coord_score * turn_active.float()

    return coord_score


def joint_group_symmetry_penalty(
    env: ManagerBasedRLEnv,
    threshold_from_mean: float = 0.15,
    asset_cfg=None,
) -> torch.Tensor:
    """
    Penalise any single joint that deviates significantly from the GROUP MEAN
    of all joints in the same group.

    This is the correct tool for the "tripod / tail-leg" exploit where the policy
    lifts ONE rear wheel off the ground for a triangular stance, gaining slope
    stability while spinning the raised wheel fast to fake velocity tracking.

    WHY GROUP-MEAN (NOT ABSOLUTE THRESHOLD) FOR THE RAISED-LEG EXPLOIT:
    --------------------------------------------------------------------
    An absolute calf threshold (e.g. ±0.55 rad from default) also fires on ALL
    four calves during steep-slope traversal, where calves already use 0.40-0.50 rad
    just to reach tilted ground — effectively limiting terrain adaptation range.

    A group-mean deviation threshold fires ONLY when ONE joint differs significantly
    from its peers.  On symmetric slopes all four calves move together (near-zero
    spread → zero cost).  A hard left turn creates ~0.05-0.10 rad spread
    (still below threshold → zero cost).  Only the raised leg (~0.30-0.40 rad
    from the mean) triggers the penalty.

    DEAD-ZONE CALIBRATION  (threshold_from_mean):
    ----------------------------------------------
    Recommended 0.15 rad for calves:
      • Symmetric slope traversal:   spread ≈ 0.00 rad  <  0.15  → 0 cost  ✓
      • Hard left/right turn:        spread ≈ 0.05 rad  <  0.15  → 0 cost  ✓
      • Raised rear leg at 0.80 rad  (others at 0.35):
            mean   = (0.35×3 + 0.80)/4 = 0.4625 rad
            raised  spread: |0.80 − 0.4625| = 0.3375 rad  >> threshold
            normal  spread: |0.35 − 0.4625| = 0.1125 rad  < threshold → 0 cost
            excess (raised leg only): 0.3375 − 0.15 = 0.1875 rad
            cost at w=-2.0: −2.0 × 0.1875 = −0.375/step  → exploit unprofitable  ✓

    Recommended 0.10 rad for hips:
      • Normal slope balance: hip spread ≤ 0.05 rad  <  0.10  → 0 cost  ✓
      • One leg raised: hip shifts out by ~0.15-0.20 rad  >  0.10  → cost fires  ✓
      • Complements hip_crossing_penalty (absolute threshold) for per-leg case.

    WEIGHT CALIBRATION:
    -------------------
      Calves:  weight = -2.0  (raised leg is ~0.34 from mean → -0.37/step)
      Hips:    weight = -1.0  (secondary signal; hip_crossing_penalty is primary)

    Parameters
    ----------
    env                 : the running ManagerBasedRLEnv
    threshold_from_mean : dead-zone radius around the per-step group mean (rad).
    asset_cfg           : SceneEntityCfg with joint_ids resolved to the joint group.

    Returns
    -------
    Tensor shape (num_envs,).
    Sum of excess deviations beyond threshold_from_mean, across all joints.
    Multiply by a negative weight in RewardTermCfg.

    Example (calf symmetry — catches tripod gait)::

        self.rewards.calf_symmetry = RewTerm(
            func=joint_group_symmetry_penalty,
            weight=-2.0,
            params={
                "threshold_from_mean": 0.15,
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_calf_joint"]),
            },
        )
    """
    if asset_cfg is None:
        raise ValueError(
            "joint_group_symmetry_penalty requires asset_cfg with joint_names specified"
        )

    robot = env.scene[asset_cfg.name]

    joint_pos: torch.Tensor = robot.data.joint_pos[:, asset_cfg.joint_ids]  # (N, num_joints)

    # Per-step group mean — moves with terrain adaptation, does not penalise
    # symmetric slope changes (all joints shift together).
    group_mean: torch.Tensor = joint_pos.mean(dim=-1, keepdim=True)  # (N, 1)

    # Absolute deviation of each joint from the group mean.
    dev_from_mean: torch.Tensor = (joint_pos - group_mean).abs()  # (N, num_joints)

    # Dead zone: no cost when all joints are within threshold_from_mean of the mean.
    excess: torch.Tensor = (dev_from_mean - threshold_from_mean).clamp(min=0.0)

    return excess.sum(dim=-1)


def foot_air_time_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg=None,
    max_air_steps: int = 6,
    omega_threshold: float = 0.05,
) -> torch.Tensor:
    """
    Penalise feet that stay off the ground longer than max_air_steps during a turn.

    PURPOSE — enforce short quick shuffles instead of long lunging strides
    -----------------------------------------------------------------------
    Phase B v5 (2026-07-25) confirmed that ``pivot_step_coordination`` successfully
    teaches the robot to lift feet and move them in the correct lateral direction.
    However the visual eval showed a new issue: the robot takes LARGE, LONG strides
    — big arcs with significant air time — rather than quick short shuffles.

    WHY LONG STRIDES HAPPEN:
    - ``foot_alternation_reward`` rewards COUNT of feet lifted (1=+1.0, 2=+0.5).
      It does not penalise HOW LONG each foot stays in the air.
    - ``pivot_step_coord`` rewards foot lateral velocity ≥ min_swing_vel.
      At higher velocity (fast big swing), the foot satisfies the threshold MORE
      easily and for more steps → inadvertently biases toward high-velocity lunges.
    - Without any air-time penalty, the policy has no incentive to plant the foot
      quickly — it stays up as long as needed for stability recovery.

    WHY SHORT SHUFFLES ARE BETTER:
    - Short swing time → less body instability during mid-air phase
    - Less need for aggressive stability compensation → less ``bad_orientation``
    - More frequent foot contacts → smoother angular velocity → less jerk
    - More natural quadruped gait (dogs take short quick steps when turning)

    IMPLEMENTATION — per-foot step counter in env.extras:
    ------------------------------------------------------
    Each foot has a counter tracking consecutive steps in swing (off ground).
    - In contact (force ≥ contact_threshold): counter resets to 0
    - In swing (force < contact_threshold): counter increments by 1

    Penalty fires for each foot where counter > max_air_steps.
    Penalty is proportional to excess air steps (dead-zone threshold style):
        excess_air = max(0, air_step_count - max_air_steps)

    DEAD-ZONE CALIBRATION (max_air_steps):
    ----------------------------------------
    At 50 Hz simulation:
      max_air_steps=6 = 0.12s air time free
      Normal quadruped shuffle at 2 Hz gait: swing phase ≈ 0.25s = 12.5 steps
      → 6 steps is tight but realistic for REDUCED swing time target
      Normal quadruped gait at 3 Hz: swing ≈ 0.16s = 8 steps → 6 still reasonable

    A 6-step free zone allows the foot to clear the ground and plant.
    Each step beyond 6 incurs penalty proportional to excess.

    WEIGHT CALIBRATION:
    -------------------
    At weight=-1.0, a foot staying up for 20 steps (0.4s):
      excess = 20 - 6 = 14 steps
      penalty = -1.0 × 14 = -14 (over 20 steps = -0.7/step average)

    foot_alternation reward for same period (1 foot up for 20 steps):
      = +1.0 × 20 = +20 total
    Net: +20 - 14 = +6 — still positive, but the free-lunge is now costly.

    A 6-step shuffle (barely above threshold, foot comes right back down):
      excess = 0 → zero penalty
      foot_alternation: +1.0 × 6 = +6 — full benefit, no cost ✓

    Gate: only fires during turn commands (|omega_cmd| > omega_threshold).

    Parameters
    ----------
    asset_cfg      : SceneEntityCfg("contact_forces", body_names=[foot links])
    max_air_steps  : steps of swing allowed before penalty fires. Default 6 = 0.12s.
    omega_threshold: minimum |omega_cmd| to activate (rad/s). Default 0.05.

    Returns
    -------
    Tensor shape (num_envs,), value ≥ 0.0.
    Sum of excess air steps across all 4 feet.
    Multiply by a NEGATIVE weight (-1.0 recommended) in RewardTermCfg.

    Example::

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
    """
    if asset_cfg is None:
        raise ValueError(
            "foot_air_time_penalty requires asset_cfg with body_names=[foot links]"
        )

    contact_sensor = env.scene[asset_cfg.name]

    # --- Gate: only active during turn commands ---
    omega_cmd: torch.Tensor = env.command_manager.get_command("base_velocity")[:, 2]
    turn_active: torch.Tensor = (omega_cmd.abs() > omega_threshold).float()  # (N,)

    # --- Contact force magnitude per foot ---
    foot_forces: torch.Tensor = contact_sensor.data.net_forces_w[
        :, asset_cfg.body_ids, :
    ]  # (N, 4, 3)
    foot_force_mag: torch.Tensor = foot_forces.norm(dim=-1)  # (N, 4)
    in_swing: torch.Tensor = (foot_force_mag < 1.0)  # (N, 4) bool — off ground

    # --- Per-foot air-step counter (persisted in env.extras) ---
    is_first_step: torch.Tensor = (env.episode_length_buf == 1)  # (N,) bool

    if "foot_air_count" not in env.extras:
        env.extras["foot_air_count"] = torch.zeros(
            env.num_envs, 4, device=foot_force_mag.device, dtype=torch.long
        )

    air_count: torch.Tensor = env.extras["foot_air_count"]  # (N, 4) long

    # Reset on first step of episode
    air_count = torch.where(
        is_first_step.unsqueeze(1).expand_as(air_count),
        torch.zeros_like(air_count),
        air_count,
    )

    # Increment counter for feet in swing, reset for feet in contact
    air_count = torch.where(in_swing, air_count + 1, torch.zeros_like(air_count))
    env.extras["foot_air_count"] = air_count

    # --- Penalty: excess air steps beyond threshold ---
    excess_air: torch.Tensor = (air_count - max_air_steps).clamp(min=0).float()  # (N, 4)
    penalty: torch.Tensor = excess_air.sum(dim=-1)  # (N,)

    # Only penalise during active turn commands
    return penalty * turn_active
