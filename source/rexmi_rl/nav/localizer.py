# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Localizer interface and simulation backend.

The entire nav stack calls ``localizer.get_pose()`` — never the sim API directly.
Phase 2 (real deployment): swap ``SimLocalizer`` for ``SLAMLocalizer`` without
touching any planning code.

Pose convention
---------------
  x, y   — world-frame horizontal position (metres)
  z      — world-frame height (metres)
  yaw    — heading angle (radians, 0 = +x axis, positive = counter-clockwise)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class Pose:
    """Robot pose in world frame."""
    x:   float   # metres
    y:   float   # metres
    z:   float   # metres
    yaw: float   # radians, CCW from +x


class Localizer(ABC):
    """Abstract localizer interface. Swap implementations for sim vs. real."""

    @abstractmethod
    def get_pose(self) -> Pose:
        """Return the current robot pose."""
        ...

    @abstractmethod
    def get_velocity(self) -> tuple[float, float, float]:
        """Return (vx, vy, vz) in world frame (m/s)."""
        ...


class SimLocalizer(Localizer):
    """
    Ground-truth localizer for Isaac Sim.

    Reads pose directly from ``robot.data.root_pos_w`` and orientation from
    ``robot.data.root_quat_w``.  Zero latency, perfect accuracy.

    Phase 2: replace with a ``SLAMLocalizer`` that reads from a SLAM topic.
    The interface (``get_pose()``, ``get_velocity()``) stays identical.

    Parameters
    ----------
    robot_art : ArticulationView or similar
        The robot articulation object from ``env.unwrapped.scene["robot"]``.
    env_idx : int
        Which parallel environment index to track (0 for single-robot nav).
    """

    def __init__(self, robot_art, env_idx: int = 0):
        self._robot = robot_art
        self._idx = env_idx

    def get_pose(self) -> Pose:
        pos  = self._robot.data.root_pos_w[self._idx]   # (3,) tensor
        quat = self._robot.data.root_quat_w[self._idx]  # (4,) tensor wxyz

        x = float(pos[0])
        y = float(pos[1])
        z = float(pos[2])

        # Extract yaw from quaternion (w, x, y, z) → rotation about z-axis
        w, qx, qy, qz = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
        yaw = math.atan2(2.0 * (w * qz + qx * qy),
                         1.0 - 2.0 * (qy * qy + qz * qz))
        return Pose(x=x, y=y, z=z, yaw=yaw)

    def get_velocity(self) -> tuple[float, float, float]:
        vel = self._robot.data.root_lin_vel_w[self._idx]  # world-frame (3,)
        return float(vel[0]), float(vel[1]), float(vel[2])
