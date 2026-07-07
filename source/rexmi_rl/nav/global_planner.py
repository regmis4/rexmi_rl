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
        replan_interval_s: float = 2.0,
        lookahead_m: float = 4.0,
    ):
        self._map = omap
        self._replan_interval = replan_interval_s
        self._lookahead_m = lookahead_m

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

    def update(self, robot_x: float, robot_y: float) -> Optional[tuple[float, float]]:
        """
        Update the planner and return the next immediate waypoint in world frame.

        Returns None if no path exists or goal not set.
        """
        if self._goal_cell is None:
            return None

        start_cell = self._map.world_to_cell(robot_x, robot_y)
        if start_cell is None:
            return None

        now = time.monotonic()
        needs_replan = (
            now - self._last_replan_t > self._replan_interval
            or not self._path_cells
            or self._deviation_exceeds(robot_x, robot_y, 1.0)
        )

        if needs_replan:
            self._path_cells = self._astar(start_cell, self._goal_cell)
            self._last_replan_t = now

        if not self._path_cells:
            return None   # no path found

        # Trim already-passed cells
        self._trim_passed(robot_x, robot_y)

        # Pick the lookahead waypoint
        return self._lookahead_point(robot_x, robot_y)

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
        """True if robot is more than threshold_m from every cell in current path."""
        if not self._path_cells:
            return True
        for r, c in self._path_cells[:10]:   # check first 10 cells
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
