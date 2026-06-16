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
