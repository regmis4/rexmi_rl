# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Occupancy map — accumulates RayCaster + LiDAR point clouds into a traversability grid.

Performance-optimised rewrite (Phase N-2 perf fix):
  • Cell data stored as flat numpy arrays (not Python dataclass objects)
    → get_cost_grid(): 320×320 in <1 ms (was ~80 ms Python loop)
    → _update_cells(): pure vectorised numpy, no Python per-point loop
  • STEP_THRESH raised 0.12 → 0.20 m (grazing-angle LiDAR false-blocking fix)
  • update_lidar() range-gates to LIDAR_COST_MAX_RANGE=15 m for step/slope
    (far hits still go in cloud for visualisation, not cell metrics)

Grid: 320×320 cells, 20 cm/cell → 64 m × 64 m world coverage.
"""

from __future__ import annotations

import math
from collections import deque
import numpy as np


class OccupancyMap:
    """
    2D grid map built from accumulated RayCaster and LiDAR readings.

    All cell data is stored as (n_cells, n_cells) numpy arrays for O(1)
    vectorised access.  No Python dataclass objects — avoids per-cell overhead.

    Parameters
    ----------
    world_size : float
        Side length of the square world region to map (metres). Default 64 m.
    cell_size : float
        Cell side length (metres). Default 0.20 m → 320×320 grid for 64 m world.
    origin : (float, float)
        World-frame (x, y) of the map centre. Default (0, 0).
    """

    STEP_THRESH          = 0.20   # m — max vertical step before cell is "blocked"
                                  # Raised from 0.12 m: at 20 cm cell size,
                                  # grazing-angle LiDAR hits (>15 m range) can
                                  # produce false 0.3 m+ steps in a single cell.
    SLOPE_THRESH         = 0.70   # tan(35°) — max slope before cell is "expensive"
    LIDAR_COST_MAX_RANGE = 15.0   # m — only use LiDAR hits within this range for
                                  # step/slope cost; beyond 15 m = grazing artifacts

    def __init__(
        self,
        world_size: float = 64.0,
        cell_size:  float = 0.20,
        origin:     tuple[float, float] = (0.0, 0.0),
    ):
        self.world_size = world_size
        self.cell_size  = cell_size
        self.origin     = origin
        self.n_cells    = int(world_size / cell_size)

        n = self.n_cells
        INF = 1e6

        # Cell arrays — all (n, n) float32/int32, direct numpy access
        self._max_step:    np.ndarray = np.zeros((n, n), dtype=np.float32)
        self._mean_slope:  np.ndarray = np.zeros((n, n), dtype=np.float32)
        self._min_height:  np.ndarray = np.full((n, n),  INF, dtype=np.float32)
        self._max_height:  np.ndarray = np.full((n, n), -INF, dtype=np.float32)
        self._visit_count: np.ndarray = np.zeros((n, n), dtype=np.int32)

        # Inflation cost layer — added by inflate_blocked().
        # Separate from the step/slope metrics so it can be set manually
        # (e.g. by the BLOCKED handler) without corrupting sensor data.
        # 0.0 = no inflation penalty; > 0 = added cost from nearby obstacle.
        self._inflation_cost: np.ndarray = np.zeros((n, n), dtype=np.float32)

        # Downward height-scan cloud (for dashboard — small sensor, short range)
        # Ring buffer: 160 rays × 50 Hz = 8k pts/s → maxlen=100k ≈ 12 s
        self._cloud_xyz: deque = deque(maxlen=100_000)
        self._cloud_arr: np.ndarray | None = None   # cached array view
        self._cloud_dirty: bool = False

        # Dense LiDAR cloud (for dashboard — SLAM map, 3240 pts/scan @ 10 Hz)
        self._lidar_cloud: np.ndarray = np.zeros((0, 3), dtype=np.float32)

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
    # Map update — height scan (downward + forward scanner)
    # ------------------------------------------------------------------

    def update(self, ray_hits_w: np.ndarray) -> None:
        """
        Ingest (N, 3) world-frame ray hits from the **downward / forward** scanner.

        ⚠️  These scanners have very short range (1.6 m downward, 5 m forward).
        At short range on a slope, adjacent rays naturally differ by large z
        values (the terrain IS sloping) — using this for step/slope metrics
        would falsely mark all crater walls as blocked.

        Therefore: update() only accumulates the visual cloud for the dashboard.
        It does NOT write step/slope metrics.  Only update_lidar() (LiDAR,
        range-gated to 15 m) writes traversability metrics to the costmap.
        """
        if ray_hits_w.shape[0] == 0:
            return
        # Vectorised bulk-extend instead of per-point Python loop.
        # At 160 rays × 50 Hz = 8 000 pts/s, the old for-loop added ~0.8 ms/step;
        # extend() with a list of tuples is ~10× faster for the same payload.
        self._cloud_xyz.extend(
            zip(ray_hits_w[:, 0].tolist(),
                ray_hits_w[:, 1].tolist(),
                ray_hits_w[:, 2].tolist())
        )
        self._cloud_dirty = True
        # ── Intentionally NOT calling _update_cells_numpy() here ──
        # Short-range scanners produce false step readings on slopes.
        # Traversability metrics come from LiDAR only (update_lidar / update_slam_map).

    # ------------------------------------------------------------------
    # Map update — dense LiDAR (360° SLAM cloud)
    # ------------------------------------------------------------------

    def update_lidar(self, lidar_pts_w: np.ndarray) -> None:
        """
        Ingest dense LiDAR point cloud and update cells.

        Only points within LIDAR_COST_MAX_RANGE (15 m) are used for step/slope
        metrics to avoid grazing-angle artifacts from far returns.

        Parameters
        ----------
        lidar_pts_w : np.ndarray  shape (N, 3)
            World-frame LiDAR hits, already filtered (no NaN/inf, range < 30 m).
        """
        if lidar_pts_w.shape[0] == 0:
            return
        self._update_cells_numpy(lidar_pts_w,
                                  max_range=self.LIDAR_COST_MAX_RANGE)

    def update_slam_map(self, slam_pts: np.ndarray) -> None:
        """
        Replace the LiDAR cloud with the full accumulated SLAM map and
        re-ingest all map points into the occupancy grid.

        Parameters
        ----------
        slam_pts : np.ndarray  shape (M, 3)
            All accumulated SLAM voxel map points (0.10 m voxel resolution).
        """
        if slam_pts.shape[0] == 0:
            return
        self._lidar_cloud = slam_pts.astype(np.float32)
        # SLAM map points are already 0.10 m voxels — all within reasonable range
        self._update_cells_numpy(slam_pts)

    # ------------------------------------------------------------------
    # Internal cell update — fully vectorised
    # ------------------------------------------------------------------

    def _update_cells_numpy(
        self,
        pts: np.ndarray,
        max_range: float | None = None,
    ) -> None:
        """
        Update grid cells from (N, 3) world-frame point array.

        Vectorised — no Python loops over points.  For N=3240 (full LiDAR scan):
          ~0.5 ms total on a typical laptop CPU.

        Parameters
        ----------
        pts : (N, 3) float array
        max_range : optional float
            If given, restrict step/slope updates to points within this range
            of the map origin (used for LiDAR to suppress far grazing artifacts).
        """
        if len(pts) == 0:
            return

        xs = pts[:, 0].astype(np.float32)
        ys = pts[:, 1].astype(np.float32)
        zs = pts[:, 2].astype(np.float32)

        half = np.float32(self.world_size / 2.0)
        cs   = np.float32(self.cell_size)
        ox   = np.float32(self.origin[0])
        oy   = np.float32(self.origin[1])

        cx = xs - ox
        cy = ys - oy

        # Within-bounds mask
        in_map = (np.abs(cx) <= half) & (np.abs(cy) <= half)

        # Optional range gate (for LiDAR far-return suppression)
        if max_range is not None:
            range2 = cx * cx + cy * cy
            in_map &= range2 <= (max_range * max_range)

        if not in_map.any():
            return

        cx_in = cx[in_map]
        cy_in = cy[in_map]
        zs_in = zs[in_map]

        rows = np.clip(((cx_in + half) / cs).astype(np.int32),
                       0, self.n_cells - 1)
        cols = np.clip(((cy_in + half) / cs).astype(np.int32),
                       0, self.n_cells - 1)

        # Vectorised per-cell min/max z using np.minimum/maximum.at
        # These are scatter operations — O(N) with numpy C loops (fast)
        cell_zmin = np.full(self.n_cells * self.n_cells,
                            np.float32(1e6), dtype=np.float32)
        cell_zmax = np.full(self.n_cells * self.n_cells,
                            np.float32(-1e6), dtype=np.float32)
        cell_hits = np.zeros(self.n_cells * self.n_cells, dtype=np.int32)

        flat_idx = rows * self.n_cells + cols

        np.minimum.at(cell_zmin, flat_idx, zs_in)
        np.maximum.at(cell_zmax, flat_idx, zs_in)
        np.add.at(cell_hits, flat_idx, 1)

        # Only update cells that actually received hits this frame
        hit_mask_flat = cell_hits > 0
        hit_idx = np.where(hit_mask_flat)[0]
        if len(hit_idx) == 0:
            return

        hit_rows = hit_idx // self.n_cells
        hit_cols = hit_idx %  self.n_cells

        z_min_new = cell_zmin[hit_idx]
        z_max_new = cell_zmax[hit_idx]
        step_new  = (z_max_new - z_min_new).clip(min=0)
        slope_new = step_new / cs

        # Worst-case monotonic updates
        np.maximum.at(self._max_step,   (hit_rows, hit_cols), step_new)
        np.maximum.at(self._mean_slope, (hit_rows, hit_cols), slope_new)
        np.minimum.at(self._min_height, (hit_rows, hit_cols), z_min_new)
        np.maximum.at(self._max_height, (hit_rows, hit_cols), z_max_new)
        self._visit_count[hit_rows, hit_cols] += 1

    # ------------------------------------------------------------------
    # Obstacle inflation (ROS-style costmap inflation)
    # ------------------------------------------------------------------

    def inflate_blocked(self, world_x: float, world_y: float,
                        radius_m: float = 1.0) -> None:
        """
        Inflate a blocked area around (world_x, world_y) with a cost gradient.

        Mimics ROS costmap inflation: nearby cells get high cost so A* routes
        around obstacles with clearance, not right through their edge.

        Cost gradient (from centre):
          0    cells  : math.inf written to _max_step (hard block)
          1-2  cells  : _inflation_cost = 12.0  (strongly avoid)
          3-4  cells  : _inflation_cost = 6.0   (prefer to avoid)
          5+   cells  : _inflation_cost = 2.0   (slight preference)

        Parameters
        ----------
        world_x, world_y : float
            Centre of the blocked area in world frame.
        radius_m : float
            Inflation radius in metres. Default 1.0 m (5 cells at 0.20 m/cell).
        """
        centre = self.world_to_cell(world_x, world_y)
        if centre is None:
            return

        cr, cc = centre
        radius_cells = int(math.ceil(radius_m / self.cell_size))
        n = self.n_cells

        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                nr, nc = cr + dr, cc + dc
                if not (0 <= nr < n and 0 <= nc < n):
                    continue
                dist_cells = math.hypot(dr, dc)
                if dist_cells <= 1.0:
                    # Hard block — force step metric above threshold
                    self._max_step[nr, nc] = float(self.STEP_THRESH + 0.50)
                    self._inflation_cost[nr, nc] = max(
                        self._inflation_cost[nr, nc], 15.0
                    )
                elif dist_cells <= 2.0:
                    self._inflation_cost[nr, nc] = max(
                        self._inflation_cost[nr, nc], 12.0
                    )
                elif dist_cells <= radius_cells:
                    # Gradient: fades from 10 → 2 over the inflation radius
                    t = (dist_cells - 2.0) / max(1.0, radius_cells - 2.0)
                    cost = 10.0 * (1.0 - t) + 2.0 * t
                    self._inflation_cost[nr, nc] = max(
                        self._inflation_cost[nr, nc], float(cost)
                    )

    # ------------------------------------------------------------------
    # Cost query (used by A*)
    # ------------------------------------------------------------------

    def traversal_cost(self, row: int, col: int) -> float:
        """
        A* traversal cost for a cell.  Returns math.inf if physically blocked.

        Step-blocked rule: only apply STEP_THRESH on relatively FLAT cells.
        On a slope, multiple LiDAR beams at different elevation angles naturally
        hit the same 20 cm cell at different heights — the intra-scan z-spread
        equals cell_size × tan(slope_angle).  At 40° that's 0.168 m; at 45° it's
        0.20 m — right at our threshold.  So the entire crater wall would be
        falsely marked blocked.

        Fix: only call a cell "blocked by step" if its measured slope is low
        (< tan(20°) = 0.36).  On steep terrain the height spread is expected
        and only the slope metric matters.  Our rocky_slope policy handles up
        to ~45° slopes fine.
        """
        if self._visit_count[row, col] == 0:
            return 5.0   # unknown

        s    = float(self._mean_slope[row, col])
        step = float(self._max_step[row, col])

        # True vertical obstacle (boulder/wall) on near-flat ground — blocked
        # Slope cells: height spread is expected, not a real step
        is_slope_cell = s > 0.36   # tan(20°) — some incline present
        if not is_slope_cell and step > self.STEP_THRESH:
            return math.inf   # real vertical obstacle on flat terrain

        # Cost by slope magnitude — rocky_slope policy handles all of these
        if s > self.SLOPE_THRESH:   # tan(35°) = 0.70
            base = 6.0   # steep but traversable
        elif s > 0.36:              # tan(20°)
            base = 2.0   # moderate slope
        else:
            base = 1.0   # flat/clear

        # Add inflation penalty from inflate_blocked() — this creates the
        # ROS-style cost gradient around obstacles so A* routes with clearance.
        infl = float(self._inflation_cost[row, col])
        return max(base, infl)

    # ------------------------------------------------------------------
    # Dashboard data accessors
    # ------------------------------------------------------------------

    def get_cost_grid(self) -> np.ndarray:
        """
        Return (n_cells, n_cells) float32 traversal cost array.

        Matches traversal_cost() logic exactly:
          - unknown          : 5.0
          - flat/clear       : 1.0
          - moderate slope   : 2.0  (>tan20°)
          - steep slope      : 6.0  (>tan35°)
          - blocked obstacle : 20.0 (step > STEP_THRESH on flat ground only)

        Fully vectorised — <1 ms for 320×320.
        """
        grid = np.full((self.n_cells, self.n_cells), 5.0, dtype=np.float32)

        known      = self._visit_count > 0
        is_slope   = self._mean_slope > 0.36   # tan(20°)
        is_steep   = self._mean_slope > self.SLOPE_THRESH  # tan(35°)
        is_blocked = (~is_slope) & (self._max_step > self.STEP_THRESH)

        grid[known] = 1.0
        grid[known & is_slope & ~is_steep] = 2.0   # moderate slope
        grid[known & is_steep]             = 6.0   # steep but traversable
        grid[known & is_blocked]           = 20.0  # true vertical obstacle on flat

        # Apply inflation layer on top — visible on dashboard as a cost gradient
        # around boulders/blocked areas.  np.maximum preserves blocked=20 cells.
        has_inflation = self._inflation_cost > 0
        grid[has_inflation] = np.maximum(
            grid[has_inflation], self._inflation_cost[has_inflation]
        )

        return grid

    def get_visual_grid(self):
        """Read-only display snapshot: planner costs, measured heights, observed mask.

        Call from the simulation thread. Unknown heights remain masked; this
        never consults the simulator's hidden terrain mesh.
        """
        return (self.get_cost_grid(), self._max_height.copy(),
                (self._visit_count > 0).copy())

    def get_point_cloud(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (xs, ys, zs) arrays for the downward height-scan cloud."""
        if not self._cloud_xyz:
            empty = np.array([], dtype=np.float32)
            return empty, empty, empty
        if self._cloud_dirty or self._cloud_arr is None:
            self._cloud_arr   = np.array(self._cloud_xyz, dtype=np.float32)
            self._cloud_dirty = False
        arr = self._cloud_arr
        return arr[:, 0], arr[:, 1], arr[:, 2]

    def get_lidar_cloud(self) -> np.ndarray:
        """Return the accumulated SLAM/LiDAR point cloud as (M, 3) float32."""
        return self._lidar_cloud
