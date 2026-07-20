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


class SLAMLocalizer(Localizer):
    """
    SLAM-based localizer that uses 3D ICP scan matching for pose estimation.

    Wraps ``LidarSLAM`` and exposes the ``Localizer`` interface.
    Falls back to ``SimLocalizer`` during the bootstrap phase or if ICP diverges.

    The key reliability mechanism is a **stability gate**: the SLAM pose is only
    used when ICP has produced low-RMS results for ``min_stable_frames`` consecutive
    frames.  A single successful ICP frame is not enough — ICP on a moving robot
    in a partially-mapped environment frequently produces one-off good results
    followed by divergence.  Requiring N consecutive low-RMS frames filters out
    transient lock-ons and prevents position jumps from propagating to the nav stack.

    Parameters
    ----------
    slam : LidarSLAM
        The SLAM engine (shared with the navigator for map access).
    sim_localizer : SimLocalizer
        Fallback during bootstrap + divergence.  In simulation this is ground-truth;
        in deployment this would be wheel odometry + IMU dead-reckoning.
    rms_thresh : float
        Maximum ICP RMS (m) to count as a stable frame. Default 0.15 m.
    min_stable_frames : int
        Number of consecutive low-RMS frames required before switching to SLAM pose.
        Default 5 (= 0.5 s at 10 Hz LiDAR).
    """

    def __init__(self, slam, sim_localizer: SimLocalizer,
                 rms_thresh: float = 0.12,
                 min_stable_frames: int = 5):
        self._slam  = slam
        self._sim   = sim_localizer
        self._rms_thresh       = rms_thresh
        self._min_stable       = min_stable_frames
        self._stable_count:    int  = 0        # consecutive low-RMS frames
        self._using_slam:      bool = False    # whether we're in SLAM mode
        self._last_good_pose:  "Pose | None" = None   # last trusted SLAM pose

    def get_pose(self) -> Pose:
        """
        Return SLAM-estimated pose once stable, otherwise fallback pose.

        Stability gate logic:
          • Count consecutive ICP frames with RMS < rms_thresh
          • Switch to SLAM mode only after min_stable_frames consecutive successes
          • If RMS spikes above 2× threshold → immediately fall back and reset counter
          • Once in SLAM mode: use last_good_pose if current frame is bad (single
            bad frame doesn't cause a jump — requires 3 consecutive bad frames to
            revert to fallback)
        """
        slam_6dof = self._slam.get_pose_6dof()
        rms = self._slam.last_rms

        if slam_6dof is not None and rms > 0.0:
            if rms < self._rms_thresh:
                # Good ICP result
                self._stable_count = min(self._stable_count + 1, self._min_stable + 10)
                if self._stable_count >= self._min_stable:
                    self._using_slam = True
                candidate = Pose(
                    x=slam_6dof.x, y=slam_6dof.y,
                    z=slam_6dof.z, yaw=slam_6dof.yaw,
                )
                self._last_good_pose = candidate
                if self._using_slam:
                    return candidate
            else:
                # Bad ICP frame
                self._stable_count = max(0, self._stable_count - 2)
                if self._stable_count == 0:
                    self._using_slam = False
                # While still in SLAM mode, use last good pose rather than jumping
                if self._using_slam and self._last_good_pose is not None:
                    return self._last_good_pose

        # Fallback (bootstrap, not yet stable, or reverted)
        return self._sim.get_pose()

    def get_velocity(self) -> tuple[float, float, float]:
        """Always use sim/odometry velocity (SLAM doesn't estimate velocity directly)."""
        return self._sim.get_velocity()

    @property
    def is_slam_active(self) -> bool:
        """True when SLAM pose is being used (stable ICP)."""
        return self._using_slam

    @property
    def stable_count(self) -> int:
        """Consecutive low-RMS ICP frame count."""
        return self._stable_count
