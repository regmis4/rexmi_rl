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
