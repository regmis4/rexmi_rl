# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Custom MDP terms for REXMI velocity-tracking environments."""

from .rewards import (  # noqa: F401
    climb_progress,
    stagnation_penalty,
    hip_crossing_penalty,
    joint_deviation_threshold,
    joint_group_symmetry_penalty,
    wheel_velocity_penalty,
    uphill_lean_reward,
    position_drift_penalty,
    base_height_penalty,
    foot_alternation_reward,
    yaw_stagnation_penalty,
    heading_progress,
    trunk_stability_penalty,
    pivot_step_coordination,
    foot_air_time_penalty,
)
