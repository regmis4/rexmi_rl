# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Occupancy map — accumulates RayCaster point cloud into a traversability grid.

Each cell (50 cm × 50 cm) stores the worst-case terrain metrics seen so far.
The global planner reads this map for A* cost computation.
The dashboard reads it for the 2D cost overlay and 3D point cloud.
"""

from __future__ import annotations

import math
from collections import deque
import numpy as np
from dataclasses import dataclass, field


@dataclass
class MapCell:
    max_step:    float = 0.0   # largest vertical step seen (m) — obstacle indicator
    mean_slope:  float = 0.0   # terrain slope (tan θ) — traversability
    min_height:  float = 1e6   # lowest z seen in cell
    max_height:  float = -1e6  # highest z seen in cell
    visit_count: int   = 0     # number of times scanned


class OccupancyMap:
    """
    2D grid map built from accumulated RayCaster height-scan readings.

    Parameters
    ----------
    world_size : float
        Side length of the square world region to map (metres). Default 64 m.
    cell_size : float
        Cell side length (metres). Default 0.5 m → 128×128 grid for 64 m world.
    origin : (float, float)
        World-frame (x, y) of the map centre. Default (0, 0).
    """

    STEP_THRESH  = 0.12   # m — max vertical step before cell is "blocked"
    SLOPE_THRESH = 0.70   # tan(35°) — max slope before cell is "expensive"

    def __init__(
        self,
        world_size: float = 64.0,
        cell_size:  float = 0.50,
        origin:     tuple[float, float] = (0.0, 0.0),
    ):
        self.world_size = world_size
        self.cell_size  = cell_size
        self.origin     = origin
        self.n_cells    = int(world_size / cell_size)

        # Grid of cells
        self._cells: list[list[MapCell]] = [
            [MapCell() for _ in range(self.n_cells)]
            for _ in range(self.n_cells)
        ]

        # Full cumulative terrain point cloud for 3D dashboard rendering.
        # This grows forever (capped at 200k for matplotlib performance) so
        # the 3D plot shows ALL terrain the robot has ever scanned, not just
        # a sliding window.  Old points are never dropped — downsampling
        # happens at render time in Dashboard._redraw().
        self._cloud_xyz: list = []
        self._cloud_max: int  = 200_000

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def world_to_cell(self, x: float, y: float) -> tuple[int, int] | None:
        """Return (row, col) cell index for world-frame (x, y). None if outside map."""
        half = self.world_size / 2.0
        cx   = x - self.origin[0]
        cy   = y - self.origin[1]
        if abs(cx) > half or abs(cy) > half:
            return None
        row = int((cx + half) / self.cell_size)
        col = int((cy + half) / self.cell_size)
        row = max(0, min(self.n_cells - 1, row))
        col = max(0, min(self.n_cells - 1, col))
        return row, col

    def cell_to_world(self, row: int, col: int) -> tuple[float, float]:
        """Return world-frame (x, y) centre of a cell."""
        half = self.world_size / 2.0
        x = self.origin[0] + (row + 0.5) * self.cell_size - half
        y = self.origin[1] + (col + 0.5) * self.cell_size - half
        return x, y

    # ------------------------------------------------------------------
    # Map update
    # ------------------------------------------------------------------

    def update(self, ray_hits_w: np.ndarray) -> None:
        """
        Ingest a (N, 3) array of world-frame ray hit positions and update cells.

        Parameters
        ----------
        ray_hits_w : np.ndarray  shape (N, 3)
            World-frame (x, y, z) positions of all RayCaster hits this step.
            Obtained from ``sensor.data.ray_hits_w[env_idx].cpu().numpy()``.
        """
        if ray_hits_w.shape[0] == 0:
            return

        # Accumulate into cumulative terrain point cloud for 3D dashboard.
        # Cap at _cloud_max (200k) to avoid matplotlib OOM.  Points are NEVER
        # dropped — if cap is hit, new points are simply not added.
        # Downsampling at render time ensures performance stays acceptable.
        if len(self._cloud_xyz) < self._cloud_max:
            remaining = self._cloud_max - len(self._cloud_xyz)
            for pt in ray_hits_w[:remaining]:
                self._cloud_xyz.append((float(pt[0]), float(pt[1]), float(pt[2])))

        # Update grid cells
        # Group hits by cell, then compute step (z range) and slope
        cell_hits: dict[tuple[int, int], list[float]] = {}
        for pt in ray_hits_w:
            idx = self.world_to_cell(float(pt[0]), float(pt[1]))
            if idx is None:
                continue
            if idx not in cell_hits:
                cell_hits[idx] = []
            cell_hits[idx].append(float(pt[2]))

        for (row, col), zs in cell_hits.items():
            cell = self._cells[row][col]
            z_min = min(zs)
            z_max = max(zs)
            step  = z_max - z_min

            # Slope estimate: z_range / cell_size (conservative)
            slope = step / self.cell_size

            cell.max_step   = max(cell.max_step, step)
            cell.mean_slope = max(cell.mean_slope, slope)  # worst-case
            cell.min_height = min(cell.min_height, z_min)
            cell.max_height = max(cell.max_height, z_max)
            cell.visit_count += 1

    # ------------------------------------------------------------------
    # Cost query (used by A*)
    # ------------------------------------------------------------------

    def traversal_cost(self, row: int, col: int) -> float:
        """
        A* traversal cost for a cell.

        Returns
        -------
        float
            1.0 = open ground.  Large values = expensive or impassable.
            math.inf = physically blocked (step too large).
        """
        cell = self._cells[row][col]

        if cell.visit_count == 0:
            return 5.0   # unknown → penalty (conservative)

        if cell.max_step > self.STEP_THRESH:
            return math.inf   # blocked

        cost = 1.0
        if cell.mean_slope > self.SLOPE_THRESH:
            cost += 10.0   # steep — expensive but not blocked
        elif cell.mean_slope > 0.47:   # tan(25°)
            cost += 3.0
        return cost

    # ------------------------------------------------------------------
    # Dashboard data accessors
    # ------------------------------------------------------------------

    def get_cost_grid(self) -> np.ndarray:
        """Return (n_cells, n_cells) float32 array of traversal costs (capped at 20)."""
        grid = np.zeros((self.n_cells, self.n_cells), dtype=np.float32)
        for r in range(self.n_cells):
            for c in range(self.n_cells):
                cost = self.traversal_cost(r, c)
                grid[r, c] = min(cost, 20.0) if cost != math.inf else 20.0
        return grid

    def get_point_cloud(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (xs, ys, zs) arrays for 3D matplotlib scatter."""
        if not self._cloud_xyz:
            empty = np.array([], dtype=np.float32)
            return empty, empty, empty
        arr = np.array(self._cloud_xyz, dtype=np.float32)
        return arr[:, 0], arr[:, 1], arr[:, 2]
