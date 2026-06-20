# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Custom MDP terms for REXMI velocity-tracking environments."""

from .rewards import (  # noqa: F401
    climb_progress,
    stagnation_penalty,
    hip_crossing_penalty,
    joint_deviation_threshold,
)
