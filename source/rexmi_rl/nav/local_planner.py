# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Local planner — converts the 16×10 height-scan into a (vx, vy, ωz) command.

Algorithm
---------
1. Parse the 160-ray height scan into a 16×10 grid (x=forward, y=lateral).
2. Compute per-column traversability: max vertical step and mean slope.
3. Score each of 5 candidate heading offsets (−40°, −20°, 0°, +20°, +40°).
4. Pick the heading closest to the global waypoint direction that is traversable.
5. Set vx proportional to traversability confidence, ωz proportional to heading error.

Height scan layout (body frame, yaw-aligned)
--------------------------------------------
  axis-0 (x): 16 rays, 0.1 m apart → covers 0 to 1.5 m ahead
  axis-1 (y): 10 rays, 0.1 m apart → covers −0.45 m to +0.45 m laterally
  Values: terrain height RELATIVE to the robot's base frame z
          (negative = terrain below robot, positive = terrain above)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class LocalPlannerOutput:
    vx: float         # forward speed command (m/s), negative = reverse
    vy: float         # lateral speed command (m/s), always 0 for legged+wheel
    omega: float      # yaw rate command (rad/s)
    slope_ahead: float        # estimated slope of best column (rad)
    heading_error: float      # angle to waypoint (rad)
    best_col_idx: int         # which of 5 candidate headings was selected
    traversable_mask: list    # per-column traversability bool (10 columns)
    fwd_obstacle_dist: float = math.inf   # nearest forward obstacle (m); inf = clear


class LocalPlanner:
    """
    Height-scan traversability planner.

    Parameters
    ----------
    vx_normal : float
        Forward speed on clear terrain (m/s). Default 0.4.
    vx_steep : float
        Forward speed on steep terrain (m/s). Default 0.25.
    omega_gain : float
        Proportional gain: omega = gain × heading_error. Default 1.2 rad/s per rad.
    omega_max : float
        Maximum yaw rate command (rad/s). Default 0.8.
    step_thresh : float
        Vertical step threshold for "blocked" (m). Default 0.12.
    slope_thresh : float
        Slope (tan θ) threshold for "steep" mode. Default 0.47 (tan 25°).
    """

    # Candidate heading offsets to evaluate (radians)
    HEADING_CANDIDATES = [
        math.radians(d) for d in [-40, -20, 0, +20, +40]
    ]
    # Map each candidate to lateral column offset index (0=far left, 9=far right)
    # 0° offset = centre column (4-5), ±20° = ±2 cols, ±40° = ±4 cols
    COL_OFFSETS = [-4, -2, 0, +2, +4]   # column shift from centre (col 4)

    def __init__(
        self,
        vx_normal:   float = 0.40,
        vx_steep:    float = 0.25,
        omega_gain:  float = 1.20,
        omega_max:   float = 0.80,
        step_thresh: float = 0.12,
        slope_thresh: float = 0.47,
    ):
        self.vx_normal    = vx_normal
        self.vx_steep     = vx_steep
        self.omega_gain   = omega_gain
        self.omega_max    = omega_max
        self.step_thresh  = step_thresh
        self.slope_thresh = slope_thresh

    def compute(
        self,
        scan_heights: list[float],    # 160-element flat list (policy obs format)
        robot_yaw:    float,           # radians
        robot_x:      float,
        robot_y:      float,
        waypoint_x:   float,
        waypoint_y:   float,
    ) -> LocalPlannerOutput:
        """
        Compute velocity command from height scan and desired waypoint.

        Parameters
        ----------
        scan_heights : list[float]
            160 height values from the RayCaster, in row-major order:
            scan[i*10 + j] = height at forward_index i, lateral_index j.
            Values are relative to robot base (negative = terrain below base).
        robot_yaw : float
            Current robot heading (radians, CCW from +x).
        robot_x, robot_y : float
            Current robot world-frame position.
        waypoint_x, waypoint_y : float
            Next waypoint world-frame position.

        Returns
        -------
        LocalPlannerOutput
            Contains vx, vy, omega and diagnostic fields for the dashboard.
        """
        # Reshape scan to (16, 10): rows=forward, cols=lateral
        scan = [[scan_heights[i * 10 + j] for j in range(10)] for i in range(16)]

        # ------------------------------------------------------------------
        # 1. Per-column traversability (10 lateral columns)
        # ------------------------------------------------------------------
        traversable = []
        col_slopes  = []
        for j in range(10):
            col_z = [scan[i][j] for i in range(16)]
            # Max vertical step between adjacent forward rows
            max_step = max(abs(col_z[i+1] - col_z[i]) for i in range(15))
            # Overall slope across 1.5 m forward reach
            slope = abs(col_z[-1] - col_z[0]) / 1.5
            traversable.append(max_step < self.step_thresh)
            col_slopes.append(slope)

        # ------------------------------------------------------------------
        # 2. Direction to waypoint in body frame
        # ------------------------------------------------------------------
        dx = waypoint_x - robot_x
        dy = waypoint_y - robot_y
        wp_angle_world = math.atan2(dy, dx)
        heading_error  = _wrap_angle(wp_angle_world - robot_yaw)

        # ------------------------------------------------------------------
        # 3. Score each heading candidate
        # ------------------------------------------------------------------
        best_heading_offset = 0.0
        best_col = 2   # default = straight ahead
        best_score = -1e9

        for k, (h_offset, col_shift) in enumerate(
            zip(self.HEADING_CANDIDATES, self.COL_OFFSETS)
        ):
            # Lateral columns covered by this heading (centre ± 1)
            centre_col = 4 + col_shift   # 0-indexed
            cols = [max(0, min(9, centre_col + d)) for d in [-1, 0, 1]]

            # Reject if any covered column is blocked
            if not all(traversable[c] for c in cols):
                continue

            mean_slope = sum(col_slopes[c] for c in cols) / len(cols)

            # Score: prefer headings closest to waypoint direction,
            #        penalise steep slopes slightly
            heading_alignment = -abs(_wrap_angle(h_offset - heading_error))
            score = heading_alignment - 0.5 * mean_slope

            if score > best_score:
                best_score      = score
                best_heading_offset = h_offset
                best_col        = k

        # ------------------------------------------------------------------
        # 4. Velocity command
        # ------------------------------------------------------------------
        # If no traversable heading found: the RL policy (rocky_slope) was
        # trained to negotiate steep walls and boulders.  Rather than stopping
        # and triggering a futile recovery loop, fall back to pointing straight
        # at the waypoint and commanding vx_steep.  The policy handles it.
        if best_score == -1e9:
            # Fallback: all columns blocked (steep wall / crater rim).
            # The RL policy handles the terrain — just push it straight toward
            # the waypoint.  Use reduced omega so the policy isn't fighting a
            # large rotation command while also trying to descend.
            # Only steer if heading error is large (> 30°), otherwise go straight.
            mean_slope_all = sum(col_slopes) / len(col_slopes)
            if abs(heading_error) > math.radians(30):
                omega_fallback = float(
                    max(-self.omega_max * 0.5,
                        min(self.omega_max * 0.5, self.omega_gain * heading_error))
                )
            else:
                omega_fallback = 0.0   # already roughly aimed — just go straight
            return LocalPlannerOutput(
                vx=self.vx_steep,        # always move; policy handles terrain
                vy=0.0,
                omega=omega_fallback,
                slope_ahead=float(mean_slope_all),
                heading_error=float(heading_error),
                best_col_idx=-1,         # signals "fallback mode" to dashboard
                traversable_mask=traversable,
            )

        # Steering: error between robot heading and desired heading
        steer_error = _wrap_angle(best_heading_offset - heading_error)
        omega = float(max(-self.omega_max,
                          min(self.omega_max, -self.omega_gain * steer_error)))

        # Slope-dependent speed
        centre_col = 4 + self.COL_OFFSETS[best_col]
        slope_ahead = col_slopes[max(0, min(9, centre_col))]
        vx = self.vx_steep if slope_ahead > self.slope_thresh else self.vx_normal

        # Slow down when turning hard
        turn_factor = max(0.4, 1.0 - abs(omega) / self.omega_max)
        vx *= turn_factor

        return LocalPlannerOutput(
            vx=float(vx),
            vy=0.0,
            omega=omega,
            slope_ahead=float(slope_ahead),
            heading_error=float(heading_error),
            best_col_idx=int(best_col),
            traversable_mask=traversable,
        )


    def compute_with_forward(
        self,
        scan_heights:   list[float],
        robot_yaw:      float,
        robot_x:        float,
        robot_y:        float,
        waypoint_x:     float,
        waypoint_y:     float,
        fwd_hits_world: "np.ndarray | None" = None,   # (N, 3) world-frame forward hits
        robot_pos_w:    "tuple[float,float,float] | None" = None,
    ) -> LocalPlannerOutput:
        """
        Compute velocity command with optional forward-scanner obstacle awareness.

        Calls ``compute()`` for the full height-scan-based plan, then applies
        a forward obstacle zone check using the LiDAR-like forward scanner hits:

        Forward danger zone:
          • Width  : 0.6 m (robot body width + 10 cm clearance each side)
          • Depth  : 0 – 2.0 m ahead in body frame
          • Height : any hit > 0.10 m above ground level (i.e. a boulder, wall, or step)

        If an obstacle is detected within the danger zone:
          • vx is scaled down linearly to zero at 0.5 m, full speed at 2 m
          • heading candidates that point toward the obstacle sector are flagged

        Parameters
        ----------
        fwd_hits_world : np.ndarray shape (N, 3), optional
            World-frame (x, y, z) forward scanner ray hits for this env.
            Filtered hits only (no inf/NaN). Pass None to skip forward check.
        robot_pos_w : (rx, ry, rz), optional
            Robot world-frame position. Required when fwd_hits_world is provided.

        Returns
        -------
        LocalPlannerOutput
            Same as compute() but fwd_obstacle_dist is set to the nearest
            obstacle distance (metres) when one is detected.
        """
        import numpy as np

        # Base plan from height scan
        out = self.compute(scan_heights, robot_yaw, robot_x, robot_y,
                           waypoint_x, waypoint_y)

        if fwd_hits_world is None or robot_pos_w is None or len(fwd_hits_world) == 0:
            return out

        rx, ry, rz = robot_pos_w
        cos_yaw = math.cos(robot_yaw)
        sin_yaw = math.sin(robot_yaw)

        # Transform forward hits to body frame (2D: forward=x_b, lateral=y_b)
        dx_w = fwd_hits_world[:, 0] - rx
        dy_w = fwd_hits_world[:, 1] - ry
        dz_w = fwd_hits_world[:, 2] - rz    # height above robot base

        x_b = dx_w * cos_yaw + dy_w * sin_yaw   # forward in body frame
        y_b = -dx_w * sin_yaw + dy_w * cos_yaw  # lateral in body frame

        # Forward danger zone: in front of robot, within body width, above ground
        # dz_w > 0.10 m: filters out ground hits — only raises above terrain count
        DANGER_DEPTH  = 2.0   # m — look-ahead distance
        DANGER_WIDTH  = 0.6   # m half-width (±0.3 m each side)
        OBS_HEIGHT    = 0.10  # m above robot base z to count as obstacle

        in_zone = (
            (x_b > 0.1)              # ahead of robot
            & (x_b < DANGER_DEPTH)  # within look-ahead
            & (np.abs(y_b) < DANGER_WIDTH / 2.0)  # within body width
            & (dz_w > OBS_HEIGHT)   # above ground (boulder/wall, not flat terrain)
        )

        if not in_zone.any():
            return out   # no obstacle in danger zone — return plan unchanged

        # Find nearest obstacle distance
        nearest_dist = float(np.min(x_b[in_zone]))

        # Scale vx: full speed at DANGER_DEPTH, zero at 0.5 m
        SLOW_START = DANGER_DEPTH   # m — start slowing here
        STOP_DIST  = 0.5            # m — stop here
        if nearest_dist <= STOP_DIST:
            vx_scaled = 0.0
        elif nearest_dist < SLOW_START:
            frac = (nearest_dist - STOP_DIST) / (SLOW_START - STOP_DIST)
            vx_scaled = float(out.vx * frac)
        else:
            vx_scaled = out.vx  # beyond danger zone — no change

        return LocalPlannerOutput(
            vx=vx_scaled,
            vy=out.vy,
            omega=out.omega,
            slope_ahead=out.slope_ahead,
            heading_error=out.heading_error,
            best_col_idx=out.best_col_idx,
            traversable_mask=out.traversable_mask,
            fwd_obstacle_dist=nearest_dist,
        )


def _wrap_angle(a: float) -> float:
    """Wrap angle to [-π, π]."""
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a
