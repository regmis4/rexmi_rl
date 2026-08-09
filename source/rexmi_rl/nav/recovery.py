# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Recovery FSM — detects stuck conditions and executes escape manoeuvres.

States
------
  NAVIGATING  — normal operation
  STUCK       — robot velocity < threshold for > stuck_timeout seconds
  REVERSING   — backing up for reverse_duration seconds
  ROTATING    — rotating in place for rotate_duration seconds
  BLOCKED     — recovery failed 3 times; signal global replan

Transitions
-----------
  NAVIGATING → STUCK      : |v| < stuck_speed for stuck_timeout s
  STUCK      → REVERSING  : immediately
  REVERSING  → ROTATING   : after reverse_duration s
  ROTATING   → NAVIGATING : after rotate_duration s (attempt counter += 1)
  NAVIGATING → BLOCKED    : attempt counter >= max_attempts
  BLOCKED    → NAVIGATING : global planner found new path (reset externally)
"""

from __future__ import annotations

import math
import time
from enum import Enum, auto


class RecoveryState(Enum):
    NAVIGATING = auto()
    STUCK      = auto()
    REVERSING  = auto()
    ROTATING   = auto()
    BLOCKED    = auto()


class RecoveryFSM:
    """
    Stuck detection and escape behaviour.

    Parameters
    ----------
    stuck_speed : float
        Speed below which the robot is considered possibly stuck (m/s). Default 0.05.
    stuck_timeout : float
        Seconds below stuck_speed before triggering recovery. Default 3.0.
    reverse_duration : float
        Seconds to reverse during escape. Default 1.5.
    rotate_duration : float
        Seconds to rotate during escape. Default 1.8.
    rotate_speed : float
        Yaw rate during rotation (rad/s). Default 0.6.
    max_attempts : int
        Recovery attempts before declaring BLOCKED. Default 3.
    """

    def __init__(
        self,
        stuck_speed:       float = 0.05,  # m/s — robot at 0.04 m/s against a wall IS stuck
        stuck_timeout:     float = 5.0,   # s — 5s wall time: enough for slow slope steps,
                                          # fast enough to escape a boulder press quickly.
                                          # Was 8s (too long — 8s of vx=0.40 against boulder),
                                          # was 4s (too short — triggered on slow crater descent).
        reverse_duration:  float = 2.0,
        # ROTATING: Navigator starts ReorientController on entry.  Stay here
        # long enough for at least one steep HOLD+YAW+SETTLE cycle (~6 s) so
        # is_rotating remains true while the pulse turn runs.  Reorient may
        # finish earlier; Navigator still owns cmd via reorient.active.
        rotate_duration:   float = 8.0,
        rotate_speed:      float = 0.0,   # no continuous ω — reorient owns yaw
        max_attempts:      int   = 3,


        progress_thresh:   float = 0.5,   # m — 0.5 m progress counts (not 1.0 m)
        progress_timeout:  float = 60.0,  # s — raised: slow descent on crater wall takes ~60 s/metre
    ):
        self.stuck_speed      = stuck_speed
        self.stuck_timeout    = stuck_timeout
        self.reverse_duration = reverse_duration
        self.rotate_duration  = rotate_duration
        self.rotate_speed     = rotate_speed
        self.max_attempts     = max_attempts
        self.progress_thresh  = progress_thresh
        self.progress_timeout = progress_timeout

        self.state             = RecoveryState.NAVIGATING
        self._stuck_since:  float | None = None
        self._action_start: float | None = None
        self._attempts:     int          = 0
        self._rotate_dir:   float        = 1.0   # +1 or -1

        # Progress tracking: best (closest) distance to waypoint seen so far
        # and the last time we beat it.  Reset when waypoint changes.
        self._best_dist:          float       = math.inf
        self._last_progress_time: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_blocked(self) -> bool:
        return self.state == RecoveryState.BLOCKED

    @property
    def is_recovering(self) -> bool:
        return self.state in (
            RecoveryState.STUCK,
            RecoveryState.REVERSING,
            RecoveryState.ROTATING,
        )

    @property
    def is_rotating(self) -> bool:
        """
        True when the FSM is actively executing a rotation manoeuvre.

        Used by Navigator.step() to force PolicyMode.TURN and to start
        ReorientController (Language A pulses) during recovery rotation.

        Also True during REVERSING — the robot is backing away from an obstacle
        and will transition to ROTATING immediately after.
        """
        return self.state in (RecoveryState.REVERSING, RecoveryState.ROTATING)

    @property
    def wants_reorient(self) -> bool:
        """True in ROTATING — Navigator should drive ReorientController."""
        return self.state == RecoveryState.ROTATING

    @property
    def rotate_dir(self) -> float:
        """Latched +1 / -1 rotation direction toward the goal."""
        return self._rotate_dir


    def reset(self) -> None:
        """Call after global planner finds a new path to clear BLOCKED state."""
        self.state                = RecoveryState.NAVIGATING
        self._attempts            = 0
        self._stuck_since         = None
        self._action_start        = None
        self._best_dist           = math.inf
        self._last_progress_time  = None

    def update(
        self,
        speed:         float,            # |v| in m/s
        heading_error: float,            # used to pick rotation direction
        waypoint_dist: float = math.inf, # current distance to active waypoint (m)
    ) -> tuple[float, float, float]:
        """
        Advance the FSM and return the override (vx, vy, omega) command.

        During NAVIGATING the returned command is (0,0,0) — the local planner
        command is used instead.  During recovery the returned command overrides
        the local planner.

        Parameters
        ----------
        speed : float
            Current robot speed magnitude (m/s).
        heading_error : float
            Current heading error to waypoint (rad). Used to pick rotation dir.
        waypoint_dist : float
            Current Euclidean distance to the active mission waypoint (m).
            Used for progress-based stuck detection: if the robot hasn't closed
            the gap by ``progress_thresh`` metres in ``progress_timeout`` seconds
            it is declared stuck — even if speed is non-zero (e.g. slipping in
            place on a frictionless slope).

        Returns
        -------
        (vx, vy, omega) override.  All zeros when NAVIGATING normally.
        """
        now = time.monotonic()

        if self.state == RecoveryState.NAVIGATING:
            # ------------------------------------------------------------------
            # TURNING EXEMPTION: if the robot is executing a large heading
            # correction (|he| > 60°), it is intentionally turning — not stuck.
            # Speed will be low (vx≈0 during point turns) and waypoint distance
            # will grow (robot pivots without advancing).  Both stuck detectors
            # would fire spuriously.  Suppress them until the robot has roughly
            # aligned with its goal (|he| < 60°).
            # ------------------------------------------------------------------
            turning = abs(heading_error) > math.radians(60)

            # --- speed-based stuck check (catches hard stops) ---
            speed_stuck = False
            if not turning:
                if speed < self.stuck_speed:
                    if self._stuck_since is None:
                        self._stuck_since = now
                    elif now - self._stuck_since > self.stuck_timeout:
                        speed_stuck = True
                else:
                    self._stuck_since = None
            else:
                # Reset stuck timer while turning so it doesn't accumulate
                self._stuck_since = None

            # --- progress-based stuck check (catches slipping/oscillating) ---
            # Suppressed during large turns: the robot is expected to pivot in
            # place, temporarily increasing waypoint distance, which would
            # falsely trigger progress_stuck within seconds.
            progress_stuck = False
            if not turning and waypoint_dist < math.inf:
                if self._last_progress_time is None:
                    self._last_progress_time = now
                    self._best_dist = waypoint_dist
                elif waypoint_dist < self._best_dist - self.progress_thresh:
                    # Made meaningful progress — reset progress timer
                    self._best_dist          = waypoint_dist
                    self._last_progress_time = now
                elif now - self._last_progress_time > self.progress_timeout:
                    progress_stuck = True
            elif turning:
                # Reset progress timer during turns so it doesn't expire
                self._last_progress_time = now
                self._best_dist = min(self._best_dist, waypoint_dist)

            if speed_stuck or progress_stuck:
                self._transition_to(RecoveryState.STUCK, now)

            return 0.0, 0.0, 0.0

        elif self.state == RecoveryState.STUCK:
            # Immediately start reversing
            # Negate: policy convention is ang_vel_z positive = CW (right turn).
            # SIGN CONVENTION (same as local_planner.py — verified from logs):
            #   ang_vel_z > 0 → yaw increases (CCW = left turn)
            #   ang_vel_z < 0 → yaw decreases (CW  = right turn)
            # heading_error > 0 → goal is LEFT  → need CCW → positive omega
            # heading_error < 0 → goal is RIGHT → need CW  → negative omega
            self._rotate_dir = 1.0 if heading_error >= 0 else -1.0
            self._transition_to(RecoveryState.REVERSING, now)
            return -0.2, 0.0, 0.0

        elif self.state == RecoveryState.REVERSING:
            elapsed = now - self._action_start
            if elapsed < self.reverse_duration:
                return -0.2, 0.0, 0.0
            self._transition_to(RecoveryState.ROTATING, now)
            return 0.0, 0.0, self.rotate_speed * self._rotate_dir

        elif self.state == RecoveryState.ROTATING:
            elapsed = now - self._action_start
            if elapsed < self.rotate_duration:
                return 0.0, 0.0, self.rotate_speed * self._rotate_dir
            # Recovery attempt complete
            self._attempts += 1
            self._stuck_since = None
            # Reset progress tracking so the robot gets a fresh 60 s window
            # to make progress from its new post-recovery position.
            # Without this, progress_stuck fires immediately on return to
            # NAVIGATING because the timer was never reset.
            self._last_progress_time = None
            self._best_dist          = math.inf
            # Alternate rotation direction for the next attempt so the robot
            # tries both sides of the obstacle.
            self._rotate_dir *= -1.0
            if self._attempts >= self.max_attempts:
                self.state = RecoveryState.BLOCKED
                return 0.0, 0.0, 0.0
            self.state = RecoveryState.NAVIGATING
            return 0.0, 0.0, 0.0

        elif self.state == RecoveryState.BLOCKED:
            return 0.0, 0.0, 0.0   # wait for external reset()

        return 0.0, 0.0, 0.0

    def status_str(self) -> str:
        return f"{self.state.name} (attempt {self._attempts}/{self.max_attempts})"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _transition_to(self, new_state: RecoveryState, now: float) -> None:
        self.state         = new_state
        self._action_start = now
