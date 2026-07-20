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
        vx_normal:    float = 0.40,   # rocky_slope trained: lin_vel_x in (0.2, 0.5)
        vx_steep:     float = 0.30,
        vx_min:       float = 0.20,   # minimum forward speed during turns
                                      # rough: lin_vel_x ∈ (-0.5, 0.5) so 0.20 is in-distribution
        omega_gain:   float = 0.8,    # proportional gain for heading correction
        omega_max:    float = 1.00,   # env patched to ang_vel_z ∈ (−1.0, 1.0) in navigate.py
        turn_slow_thresh: float = 0.35,  # rad (~20°): start slowing vx when |he| > this
        step_thresh:  float = 0.20,   # match OccupancyMap.STEP_THRESH
        slope_thresh: float = 0.47,
    ):
        self.turn_slow_thresh = turn_slow_thresh
        self.vx_normal    = vx_normal
        self.vx_steep     = vx_steep
        self.vx_min       = vx_min
        self.omega_gain   = omega_gain
        self.omega_max    = omega_max
        self.step_thresh  = step_thresh
        self.slope_thresh = slope_thresh

        # Committed turn state — prevents ±180° oscillation.
        # When |he| > 90°, we latch the turn direction and don't flip it
        # until |he| drops below 45°.  Without this, any slope drift that
        # pushes he across ±180° causes omega to flip sign every few steps,
        # creating an infinite oscillation at the singularity.
        self._turn_committed:     bool  = False
        self._committed_omega_sign: float = 0.0  # +1.0 or -1.0

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

            # Score: prefer headings closest to waypoint direction ONLY.
            # Do NOT penalise slope — slope only affects speed (vx), not direction.
            # Penalising slope caused the planner to prefer flat exterior terrain
            # over the crater slope, routing the robot away from the goal.
            heading_alignment = -abs(_wrap_angle(h_offset - heading_error))
            score = heading_alignment

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
            if abs(heading_error) > math.radians(10):
                # Same sign as main path: omega = +gain × he (verified correct)
                # Dead-zone reduced from 30° → 10°: any error > 10° gets corrected.
                # At 30°, heading drift on the crater slope grew unchecked from 12°
                # to 90° before TURN OVERRIDE fired (42 s of unnecessary spinning).
                omega_fallback = float(
                    max(-self.omega_max,
                        min(self.omega_max, self.omega_gain * heading_error))
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

        # Steering: omega proportional to heading_error (angle to waypoint in body frame).
        #
        # SIGN CONVENTION (verified from nav logs step 600):
        #   Robot yaw=-84°, goal at wp_angle=+176°.
        #   heading_error = wrap(+176° - (-84°)) = -100° (negative → goal is CW = right).
        #   Observed: ang_vel_z=+1.0 → yaw INCREASES (CCW), NOT CW.
        #   Conclusion: ang_vel_z > 0 → CCW (yaw increasing), same as standard math.
        #
        # To turn CW (decrease yaw) toward goal:
        #   heading_error = -100° (negative) → need negative omega → omega = gain × he
        #   heading_error = +100° (positive) → need positive omega → omega = gain × he
        #
        # Therefore: omega = +gain × heading_error  (same sign as standard math)
        omega = float(max(-self.omega_max,
                          min(self.omega_max, self.omega_gain * heading_error)))

        # Slope-dependent speed: use best traversable column's slope
        centre_col = 4 + self.COL_OFFSETS[best_col]
        slope_ahead = col_slopes[max(0, min(9, centre_col))]
        vx = self.vx_steep if slope_ahead > self.slope_thresh else self.vx_normal

        # ------------------------------------------------------------------
        # Committed turn direction — prevents ±180° oscillation.
        #
        # Problem: when |he| is near ±180° (goal almost directly behind), any
        # small drift on a slope pushes he across the ±π boundary, flipping
        # omega sign every few steps → robot oscillates in place indefinitely.
        #
        # Solution: once we commit to a large turn (|he| > 90°), latch the
        # turn direction and hold it until the heading error drops below 45°.
        # This gives the turn a hysteresis dead-band: enter at 90°, exit at 45°.
        # ------------------------------------------------------------------
        he_abs = abs(heading_error)

        if he_abs > math.radians(90) and not self._turn_committed:
            # Enter committed turn mode — latch ONCE, never re-latch while turning.
            #
            # CRITICAL: we only set _committed_omega_sign on the FIRST entry
            # (not self._turn_committed).  Once we're in committed mode, we NEVER
            # change the sign, even if heading_error crosses ±π (the wraparound
            # singularity).  Changing sign at ±180° is what caused the orbit:
            #   • robot turns CW, he crosses −180° → +164°
            #   • he > 0 → re-latch to +1 → robot flips to CCW
            #   • robot turns CCW, crosses +180° → flips back
            #   → infinite orbit at the ±π boundary
            #
            # By never re-latching, the robot commits to one rotation direction
            # and holds it until |he| < 45° regardless of the sign of he.
            self._turn_committed       = True
            self._committed_omega_sign = math.copysign(1.0, heading_error)
        elif he_abs < math.radians(45):
            # Exit committed turn mode — heading is within acceptable range
            self._turn_committed = False

        if self._turn_committed:
            # Hold full omega in the committed direction regardless of current he sign.
            #
            # vx=0.0: The dedicated TURN policy (PolicyMode.TURN) is now the active
            # policy during committed turns — it was trained with vx=0, vy=0,
            # omega∈(-1,+1) on rocky slopes.  It produces ~57°/s clean rotation
            # without any forward drift.
            #
            # The old vx=0.15 workaround (to compensate for the rough policy's
            # OOD stall at vx=0) is NO LONGER NEEDED — the spin policy handles it.
            # Keeping vx=0.15 here would push the spin policy out-of-distribution
            # (it was trained at vx=0 and drift-penalised), causing oscillation.
            omega = self.omega_max * self._committed_omega_sign
            vx    = 0.0
        elif slope_ahead > self.slope_thresh and he_abs > self.turn_slow_thresh:
            # STEEP SLOPE + significant heading error → go straight, don't steer.
            vx    = self.vx_steep
            omega = 0.0
        elif he_abs >= self.turn_slow_thresh:
            # Moderate heading error — creep forward at vx_min while turning.
            vx = self.vx_min
        else:
            # Small heading error — blend smoothly from vx_min to vx_normal
            t  = he_abs / self.turn_slow_thresh
            vx = self.vx_min + (vx - self.vx_min) * (1.0 - t)

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

        # Forward danger zone: in front of robot, within body width, above ground.
        # OBS_HEIGHT raised to 0.25 m: the crater slope appears 0.8-1.0 m ahead
        # in the forward scanner at ~0.10-0.20 m above robot base z because the
        # slope rises into the scanner's field of view.  With 0.10 m threshold,
        # the slope wall constantly triggers the obstacle slowdown, leaving the
        # robot with vx≈0 and preventing it from making forward progress.
        # At 0.25 m we only slow for genuine boulders/walls (vertical obstacles),
        # not the crater slope surface the robot needs to descend.
        DANGER_DEPTH  = 2.0   # m — look-ahead distance
        DANGER_WIDTH  = 0.6   # m half-width (±0.3 m each side)
        OBS_HEIGHT    = 0.25  # m above robot base z — was 0.10 (slope false positive)

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
    # Single-expression modulo wrap: always O(1) regardless of how far out-of-range
    # the input is.  The while-loop version was O(k) where k = number of extra
    # full rotations, which is fine for small angles but fragile for edge cases
    # (e.g. after a physics reset that introduces a 720° yaw discontinuity).
    a = a - 2 * math.pi * math.floor((a + math.pi) / (2 * math.pi))
    return a
