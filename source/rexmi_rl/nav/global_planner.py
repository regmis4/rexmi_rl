# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Global planner — A* on the occupancy map.

Replans every 2 s or when the robot deviates > 1 m from the current path.
Returns the next waypoint 3–5 m ahead of the robot on the planned path.
"""

from __future__ import annotations

import heapq
import math
import time
from typing import List, Optional

import numpy as np

from rexmi_rl.nav.occupancy_map import OccupancyMap


class GlobalPlanner:
    """
    A* path planner operating on the OccupancyMap grid.

    Parameters
    ----------
    omap : OccupancyMap
        Shared occupancy map updated by the navigator each step.
    replan_interval_s : float
        Minimum seconds between full replans. Default 2.0.
    lookahead_m : float
        Distance ahead of robot to pick the next immediate waypoint. Default 4.0 m.
    """

    def __init__(
        self,
        omap: OccupancyMap,
        replan_interval_s: float = 3.0,
        lookahead_m: float = 4.0,   # 4 m lookahead: snappier path tracking.
                                    # 8 m was too long — on a 14 m approach the 8 m
                                    # reference barely moves as the robot drifts,
                                    # giving a sluggish heading response.  4 m gives
                                    # a tighter, more responsive steering signal while
                                    # still smoothing out minor path oscillations.
                                    # that causes the robot to spiral away from the
                                    # goal.  8 m keeps the reference stable while
                                    # still allowing A* to route around boulders.
        direct_bearing_thresh_deg: float = 30.0,
    ):
        self._map = omap
        self._replan_interval = replan_interval_s
        self._lookahead_m = lookahead_m
        self._direct_bearing_thresh = math.radians(direct_bearing_thresh_deg)

        self._last_replan_t: float = 0.0
        self._path_cells: List[tuple[int, int]] = []   # cell indices on current path
        self._goal_cell: Optional[tuple[int, int]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_goal(self, goal_x: float, goal_y: float) -> None:
        """Set a new goal in world frame and trigger immediate replan."""
        self._goal_cell = self._map.world_to_cell(goal_x, goal_y)
        self._last_replan_t = 0.0   # force replan next update()

    def update(
        self,
        robot_x: float,
        robot_y: float,
        goal_x: Optional[float] = None,
        goal_y: Optional[float] = None,
    ) -> Optional[tuple[float, float]]:
        """
        Update the planner and return the next immediate waypoint in world frame.

        Strategy (direct-bearing first):
        ──────────────────────────────────
        The RL policy (rocky_slope) is trained to handle steep, boulder-covered
        terrain directly.  A* on an unknown map routes through cost=5 "unknown"
        cells in arbitrary directions, causing the robot to orbit the crater
        instead of descending.

        New strategy:
          1. Always start with a direct lookahead toward the goal.
          2. Only deviate from direct bearing if A* finds a significantly
             lower-cost route AND that route has NO cells with cost > 6
             (i.e., genuine traversable terrain, not unknown default cost).
          3. A* is still run in the background and its path is shown on the
             dashboard — it just doesn't steer the robot unless it is genuinely
             better than straight-line.

        This means the robot steers directly toward each mission waypoint and
        lets the RL policy handle all terrain details.  A* overrides only when
        there is a real obstacle (boulder, cliff) that requires a detour.
        """
        if self._goal_cell is None:
            return None

        start_cell = self._map.world_to_cell(robot_x, robot_y)
        if start_cell is None:
            return None

        # Always compute the direct-bearing lookahead first.
        #
        # Key fix: clamp the lookahead point to the actual goal when the robot
        # is within lookahead_m of the goal.  Without this, on a 14 m approach
        # with an 8 m lookahead, the reference point is 8 m ahead in the goal
        # direction — which is fine.  But as the robot drifts laterally the
        # reference drifts too, creating an unstable heading error.  Clamping
        # to the goal when close (dist < lookahead_m) gives a fixed target.
        if goal_x is not None and goal_y is not None:
            dist_to_goal = math.hypot(goal_x - robot_x, goal_y - robot_y)
            direct_angle = math.atan2(goal_y - robot_y, goal_x - robot_x)
            if dist_to_goal <= self._lookahead_m:
                # Within lookahead range — aim directly at the goal itself
                direct_lh = (goal_x, goal_y)
            else:
                # Far from goal — place lookahead point on direct ray
                direct_lh = (
                    robot_x + self._lookahead_m * math.cos(direct_angle),
                    robot_y + self._lookahead_m * math.sin(direct_angle),
                )
        else:
            dist_to_goal = math.inf
            direct_lh = None

        # Run A* periodically for obstacle-aware routing
        now = time.monotonic()
        needs_replan = (
            now - self._last_replan_t > self._replan_interval
            or not self._path_cells
            or self._deviation_exceeds(robot_x, robot_y, 2.0)
        )

        if needs_replan:
            self._path_cells = self._astar(start_cell, self._goal_cell)
            self._last_replan_t = now

        # If no A* path, use direct bearing
        if not self._path_cells:
            return direct_lh

        # Trim already-passed cells
        self._trim_passed(robot_x, robot_y)

        # Get A* lookahead
        astar_lh = self._lookahead_point(robot_x, robot_y)

        # Use A* only if it meaningfully deviates from direct bearing AND
        # the deviation is because of real obstacles (high-cost cells on
        # the direct path).  Otherwise, always use direct bearing.
        #
        # Check whether the direct-bearing path has any blocked cells:
        # sample 5 points along the direct ray and check their cost.
        direct_blocked = False
        if goal_x is not None and goal_y is not None:
            for frac in [0.25, 0.4, 0.55, 0.7, 0.85]:
                sample_x = robot_x + frac * self._lookahead_m * math.cos(direct_angle)
                sample_y = robot_y + frac * self._lookahead_m * math.sin(direct_angle)
                cell = self._map.world_to_cell(sample_x, sample_y)
                if cell is not None:
                    cost = self._map.traversal_cost(cell[0], cell[1])
                    if cost == math.inf:   # hard obstacle on direct path
                        direct_blocked = True
                        break

        if direct_blocked and astar_lh is not None:
            # There's a real obstacle — use A* to route around it
            if not getattr(self, "_using_astar", False):
                print(f"[GlobalPlanner] Direct path blocked — following A* route")
                self._using_astar = True
            return astar_lh
        else:
            # Direct path is clear (or unknown) — go straight toward goal
            if getattr(self, "_using_astar", False):
                print(f"[GlobalPlanner] Direct path clear — resuming direct bearing")
                self._using_astar = False
            return direct_lh if direct_lh is not None else astar_lh

    def get_path_world(self) -> List[tuple[float, float]]:
        """Return current planned path as world-frame (x, y) list (for dashboard)."""
        return [self._map.cell_to_world(r, c) for r, c in self._path_cells]

    # ------------------------------------------------------------------
    # A* implementation
    # ------------------------------------------------------------------

    def _astar(
        self,
        start: tuple[int, int],
        goal:  tuple[int, int],
    ) -> List[tuple[int, int]]:
        """Return cell list from start to goal, or empty list if unreachable."""
        n = self._map.n_cells

        def h(a, b):  # octile heuristic (allows diagonal movement)
            dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
            return max(dr, dc) + (math.sqrt(2) - 1) * min(dr, dc)

        open_heap = []
        g_score = {start: 0.0}
        came_from: dict[tuple[int, int], tuple[int, int]] = {}

        heapq.heappush(open_heap, (h(start, goal), start))

        # 8-connected neighbours
        DIRS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        DIAG_COST = math.sqrt(2)

        while open_heap:
            _, current = heapq.heappop(open_heap)

            if current == goal:
                return self._reconstruct(came_from, current)

            cr, cc = current
            for dr, dc in DIRS:
                nr, nc = cr + dr, cc + dc
                if not (0 <= nr < n and 0 <= nc < n):
                    continue
                move_cost = self._map.traversal_cost(nr, nc)
                if move_cost == math.inf:
                    continue
                step = DIAG_COST if (dr != 0 and dc != 0) else 1.0
                tentative_g = g_score[current] + step * move_cost
                neighbour = (nr, nc)
                if tentative_g < g_score.get(neighbour, math.inf):
                    g_score[neighbour] = tentative_g
                    came_from[neighbour] = current
                    f = tentative_g + h(neighbour, goal)
                    heapq.heappush(open_heap, (f, neighbour))

        return []   # unreachable

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _reconstruct(
        came_from: dict[tuple[int, int], tuple[int, int]],
        current: tuple[int, int],
    ) -> List[tuple[int, int]]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    # ------------------------------------------------------------------
    # Path following helpers
    # ------------------------------------------------------------------

    def _deviation_exceeds(self, rx: float, ry: float, threshold_m: float) -> bool:
        """True if robot is more than threshold_m from every cell in current path.

        Checks all cells within 8 m of the robot (not just the first 10) so
        that normal path tracking at 0.3–0.5 m/s doesn't trigger spurious
        replans when the robot is moving along a valid but curved path.
        """
        if not self._path_cells:
            return True
        lookahead_cells = int(8.0 / self._map.cell_size)  # 8 m ÷ 0.2 m = 40 cells
        for r, c in self._path_cells[:lookahead_cells]:
            wx, wy = self._map.cell_to_world(r, c)
            if math.hypot(rx - wx, ry - wy) < threshold_m:
                return False
        return True

    def _trim_passed(self, rx: float, ry: float) -> None:
        """Remove cells behind the robot (already passed)."""
        cell_size = self._map.cell_size
        while len(self._path_cells) > 1:
            wx, wy = self._map.cell_to_world(*self._path_cells[0])
            if math.hypot(rx - wx, ry - wy) < cell_size:
                self._path_cells.pop(0)
            else:
                break

    def _lookahead_point(
        self, rx: float, ry: float
    ) -> Optional[tuple[float, float]]:
        """Return world-frame point ~lookahead_m ahead of robot on path."""
        if not self._path_cells:
            return None
        # Walk along path until accumulated distance >= lookahead
        dist = 0.0
        prev_x, prev_y = rx, ry
        for r, c in self._path_cells:
            wx, wy = self._map.cell_to_world(r, c)
            dist += math.hypot(wx - prev_x, wy - prev_y)
            prev_x, prev_y = wx, wy
            if dist >= self._lookahead_m:
                return wx, wy
        # Return last cell if path shorter than lookahead
        return self._map.cell_to_world(*self._path_cells[-1])


# _wrap_angle lives in local_planner.py — import from there if ever needed here.
# Removed duplicate 2026-07-19.
