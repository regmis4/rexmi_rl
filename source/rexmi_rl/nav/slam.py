# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
3D LiDAR SLAM — incremental point-to-plane ICP scan matching.

This module provides ``LidarSLAM``, a pure-Python (numpy + scipy) 3D SLAM
implementation that:

  1. Accepts raw 3D point clouds from the 360° LiDAR sensor each step.
  2. Voxel-downsamples each scan (0.1 m voxels → ~200 pts/scan from 3240 raw).
  3. Runs 6-DOF point-to-plane ICP against the accumulated voxel map to estimate
     the robot's pose correction relative to the ground-truth initialisation.
  4. Builds a growing 3D voxel map (0.05 m cells) as the robot explores.
  5. Exposes the estimated pose and accumulated 3D map for the nav layer.

Architecture
------------
  • Cold-start (first ``bootstrap_s`` seconds): uses ground-truth sim pose to
    seed the voxel map.  ICP is not run until enough map points exist.
  • After bootstrap: runs ICP every LiDAR update (10 Hz) to refine the pose.
  • Fallback: if ICP diverges (RMS > ``max_icp_rms``), discards the result and
    reports ``is_converged = False`` for that frame.  The nav layer falls back to
    sim pose for that step.

Coordinate system
-----------------
  All coordinates are in world frame (metres), same as Isaac Lab's root_pos_w.
  The SLAM pose is a 6-DOF transform: (x, y, z, roll, pitch, yaw).
  For the nav layer, only (x, y, z, yaw) are exposed via ``get_pose()``.

Design notes
------------
  • scipy.spatial.cKDTree: nearest-neighbour search — available in every
    conda env that has scipy (including env_isaacsim).
  • Point-to-plane ICP: more accurate on smooth surfaces than point-to-point
    because it allows sliding along the surface tangent plane.
  • Voxel map: stored as a dict keyed by (ix, iy, iz) integer voxel indices.
    Lookup is O(1); building the KD-tree for ICP is O(M log M) where M is
    the number of occupied voxels (typically 2k-20k for a crater traverse).
  • Thread safety: all SLAM state is owned by the nav thread.  The dashboard
    reads ``get_map_points()`` which returns a copy — no lock needed.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from scipy.spatial import cKDTree
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False
    print("[SLAM] WARNING: scipy not available — ICP disabled, using sim pose only")


# ---------------------------------------------------------------------------
# Pose dataclass (6-DOF)
# ---------------------------------------------------------------------------

@dataclass
class SLAMPose:
    """6-DOF robot pose estimated by SLAM."""
    x:     float   # metres, world frame
    y:     float   # metres, world frame
    z:     float   # metres, world frame
    roll:  float   # radians
    pitch: float   # radians
    yaw:   float   # radians, CCW from +x


# ---------------------------------------------------------------------------
# Main SLAM class
# ---------------------------------------------------------------------------

class LidarSLAM:
    """
    Incremental 3D LiDAR SLAM using point-to-plane ICP.

    Parameters
    ----------
    voxel_size : float
        Voxel grid cell size for map storage and scan downsampling (metres).
        Default 0.10 m — fine enough to resolve boulders, coarse enough for speed.
    bootstrap_s : float
        Duration (seconds) to use ground-truth pose for map seeding before
        switching to ICP.  Default 3.0 s gives ~30 LiDAR scans worth of map.
    max_icp_iter : int
        Maximum ICP iterations per update.  Default 30.
    icp_tol : float
        ICP convergence tolerance (RMS change between iterations).  Default 1e-4 m.
    max_icp_rms : float
        Maximum acceptable ICP RMS error before result is rejected as diverged.
        Default 0.30 m (tuned for 30 m range, 2° resolution LiDAR).
    max_corr_dist : float
        Maximum nearest-neighbour distance for ICP correspondences (metres).
        Points further than this are treated as outliers.  Default 1.0 m.
    min_map_pts : int
        Minimum voxel map size before ICP is attempted.  Default 500.
    """

    def __init__(
        self,
        voxel_size:    float = 0.10,
        bootstrap_s:   float = 3.0,
        max_icp_iter:  int   = 10,    # reduced from 30 — 10 iter sufficient for
                                       # incremental scan matching (delta pose is small);
                                       # saves ~20 ms/scan vs 30 iterations
        icp_tol:       float = 1e-4,
        max_icp_rms:   float = 0.30,
        max_corr_dist: float = 1.0,
        min_map_pts:   int   = 500,
    ):
        if not _SCIPY_OK:
            print("[SLAM] scipy missing — SLAM will use sim pose passthrough only")

        self.voxel_size    = voxel_size
        self.bootstrap_s   = bootstrap_s
        self.max_icp_iter  = max_icp_iter
        self.icp_tol       = icp_tol
        self.max_icp_rms   = max_icp_rms
        self.max_corr_dist = max_corr_dist
        self.min_map_pts   = min_map_pts

        # 3D voxel map: {(ix, iy, iz): np.array([x, y, z], mean position)}
        # Using dict for O(1) lookup; rebuilt into KD-tree for ICP.
        self._voxels:  dict[tuple[int,int,int], np.ndarray] = {}
        # Voxel accumulator: maps voxel key → (sum_xyz, count) for mean
        self._vox_sum: dict[tuple[int,int,int], list] = {}

        # ICP pose correction accumulated over all frames.
        # Stored as a 4×4 homogeneous transform.  Starts as identity.
        self._T_correction: np.ndarray = np.eye(4)

        # State
        self._start_time:   Optional[float] = None   # set on first update call
        self.is_converged:  bool = False              # True once ICP has run once OK
        self._last_rms:     float = 0.0               # last ICP RMS for dashboard
        self._frame_count:  int   = 0
        self._icp_count:    int   = 0                 # number of successful ICP runs

        # Cached KD-tree over voxel map (rebuilt every N updates)
        self._kdtree:        Optional[cKDTree] = None if not _SCIPY_OK else None
        self._map_pts_cache: Optional[np.ndarray] = None
        self._kdtree_dirty:  bool = True   # rebuild tree when map changes significantly

        # Last successfully estimated SLAM pose (set on first successful ICP run)
        self._last_slam_pose: Optional[SLAMPose] = None

        # Normals for each map point (computed from local neighbourhood)
        self._map_normals: Optional[np.ndarray] = None

        # How often to rebuild KD-tree (every N new voxels added)
        # Raised from 100 → 200: KD-tree build over ~5k voxels takes ~5 ms;
        # rebuilding every 200 new voxels vs 100 cuts this overhead in half.
        self._rebuild_interval: int = 200
        self._voxels_since_rebuild: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        lidar_pts_w: np.ndarray,
        sim_pose: "SLAMPose",
    ) -> SLAMPose:
        """
        Process one LiDAR scan and return the SLAM-estimated pose.

        Parameters
        ----------
        lidar_pts_w : np.ndarray  shape (N, 3)
            World-frame 3D points from the LiDAR sensor this frame.
            Obtain via: sensor.data.ray_hits_w[env_idx].cpu().numpy()
            Caller should filter NaN/inf before calling.
        sim_pose : SLAMPose
            Ground-truth pose from Isaac Sim (used for bootstrap and fallback).

        Returns
        -------
        SLAMPose
            Estimated pose.  During bootstrap or on ICP failure, returns sim_pose.
            After bootstrap with successful ICP, returns ICP-corrected pose.
        """
        now = time.monotonic()
        if self._start_time is None:
            self._start_time = now

        elapsed = now - self._start_time
        self._frame_count += 1

        # Filter: remove points too close to sensor (self-reflection) or too far
        if len(lidar_pts_w) == 0:
            return sim_pose

        dist_to_origin = np.linalg.norm(lidar_pts_w - np.array([sim_pose.x, sim_pose.y, sim_pose.z]), axis=1)
        valid = (dist_to_origin > 0.30) & (dist_to_origin < 30.0)
        pts = lidar_pts_w[valid]
        if len(pts) < 10:
            return sim_pose

        # Voxel-downsample the incoming scan
        scan_down = self._voxel_downsample(pts, self.voxel_size)

        # Always add to the map (using sim pose during bootstrap, ICP pose after)
        n_added = self._add_to_map(scan_down)
        if n_added > 0:
            self._voxels_since_rebuild += n_added
            if self._voxels_since_rebuild >= self._rebuild_interval:
                self._kdtree_dirty = True
                self._voxels_since_rebuild = 0

        # Bootstrap phase: just accumulate map, return sim pose
        if elapsed < self.bootstrap_s:
            return sim_pose

        # Need minimum map density before ICP makes sense
        map_pts = self._get_map_pts()
        if len(map_pts) < self.min_map_pts or not _SCIPY_OK:
            return sim_pose

        # Run ICP
        icp_pose = self._run_icp(scan_down, sim_pose, map_pts)
        if icp_pose is not None:
            return icp_pose
        else:
            # ICP diverged — fall back to sim pose
            return sim_pose

    def get_pose_6dof(self) -> Optional[SLAMPose]:
        """Return the last SLAM-estimated 6-DOF pose, or None before convergence."""
        return self._last_slam_pose if self.is_converged else None

    def get_map_points(self) -> np.ndarray:
        """
        Return accumulated SLAM map as (M, 3) float32 array.

        Returns a copy — safe to pass to dashboard without locking.
        The map grows as the robot explores.
        """
        pts = self._get_map_pts()
        if len(pts) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        return pts.astype(np.float32)

    def get_map_size(self) -> int:
        """Return number of occupied voxels in the SLAM map."""
        return len(self._voxels)

    @property
    def last_rms(self) -> float:
        """Last ICP RMS error (metres). 0.0 if ICP has not run yet."""
        return self._last_rms

    @property
    def frame_count(self) -> int:
        """Total number of LiDAR frames processed."""
        return self._frame_count

    @property
    def icp_count(self) -> int:
        """Number of successful ICP runs."""
        return self._icp_count

    # ------------------------------------------------------------------
    # Voxel map operations
    # ------------------------------------------------------------------

    def _voxel_downsample(self, pts: np.ndarray, vsize: float) -> np.ndarray:
        """
        Voxel grid downsampling: keep one representative point per voxel.

        Parameters
        ----------
        pts : (N, 3) float array
        vsize : voxel cell size (metres)

        Returns
        -------
        (M, 3) float array, M ≤ N.  Typically N/10 to N/20 for LiDAR.
        """
        if len(pts) == 0:
            return pts

        # Compute voxel indices
        keys = np.floor(pts / vsize).astype(np.int32)

        # One representative per unique voxel (use centroid)
        unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
        centroids = np.zeros((len(unique_keys), 3), dtype=np.float64)
        counts = np.zeros(len(unique_keys), dtype=np.int32)

        np.add.at(centroids, inverse, pts)
        np.add.at(counts, inverse, 1)

        centroids /= counts[:, np.newaxis]
        return centroids

    def _add_to_map(self, pts: np.ndarray) -> int:
        """
        Add downsampled points to the voxel map.
        Returns number of NEW voxels created (not updated).
        """
        if len(pts) == 0:
            return 0

        keys = np.floor(pts / self.voxel_size).astype(np.int32)
        new_count = 0

        for i, key in enumerate(keys):
            k = (int(key[0]), int(key[1]), int(key[2]))
            if k not in self._vox_sum:
                self._vox_sum[k] = [pts[i].copy(), 1]
                new_count += 1
            else:
                self._vox_sum[k][0] += pts[i]
                self._vox_sum[k][1] += 1
            # Update voxel mean
            self._voxels[k] = self._vox_sum[k][0] / self._vox_sum[k][1]

        return new_count

    def _get_map_pts(self) -> np.ndarray:
        """Return map points as (M, 3) array. Uses cached version when not dirty."""
        if not self._kdtree_dirty and self._map_pts_cache is not None:
            return self._map_pts_cache

        if len(self._voxels) == 0:
            self._map_pts_cache = np.zeros((0, 3), dtype=np.float64)
            return self._map_pts_cache

        pts = np.array(list(self._voxels.values()), dtype=np.float64)
        self._map_pts_cache = pts
        self._kdtree_dirty = False   # cache is now fresh; ICP will use it directly
        return pts

    def _get_or_build_kdtree(self, map_pts: np.ndarray) -> cKDTree:
        """Build or return cached KD-tree over map points."""
        if self._kdtree is None or self._kdtree_dirty:
            self._kdtree = cKDTree(map_pts)
            self._map_normals = self._estimate_normals(map_pts, self._kdtree)
            self._kdtree_dirty = False
        return self._kdtree

    # ------------------------------------------------------------------
    # Normal estimation
    # ------------------------------------------------------------------

    def _estimate_normals(
        self,
        pts: np.ndarray,
        tree: "cKDTree",
        k: int = 10,
    ) -> np.ndarray:
        """
        Estimate surface normals at each map point using PCA on k-NN.

        Returns (M, 3) unit normal vectors.
        For points with < k neighbours, uses [0, 0, 1] (up-pointing default).
        """
        normals = np.zeros((len(pts), 3), dtype=np.float64)
        normals[:, 2] = 1.0   # default: up-pointing

        if len(pts) < k:
            return normals

        # Query k nearest neighbours for each point
        k_query = min(k, len(pts))
        dists, idx = tree.query(pts, k=k_query)

        for i in range(len(pts)):
            neighbours = pts[idx[i]]
            cov = np.cov(neighbours.T)
            if cov.shape == (3, 3):
                try:
                    eigenvalues, eigenvectors = np.linalg.eigh(cov)
                    # Smallest eigenvalue → normal direction
                    normal = eigenvectors[:, 0]
                    # Orient normal to point upward (z > 0)
                    if normal[2] < 0:
                        normal = -normal
                    normals[i] = normal
                except np.linalg.LinAlgError:
                    pass   # keep default [0,0,1]

        return normals

    # ------------------------------------------------------------------
    # ICP
    # ------------------------------------------------------------------

    def _run_icp(
        self,
        scan_pts: np.ndarray,
        sim_pose: SLAMPose,
        map_pts: np.ndarray,
    ) -> Optional[SLAMPose]:
        """
        Run point-to-plane ICP between the current scan and the voxel map.

        Parameters
        ----------
        scan_pts : (N, 3) — downsampled current LiDAR scan (world frame)
        sim_pose : SLAMPose — ground truth for initialisation
        map_pts  : (M, 3) — all voxel map points

        Returns
        -------
        SLAMPose if ICP converged, None if it diverged.
        """
        if len(scan_pts) < 10 or len(map_pts) < 20:
            return None

        tree = self._get_or_build_kdtree(map_pts)
        normals = self._map_normals

        # Start from identity correction (we're in world frame already)
        T = np.eye(4)
        src = scan_pts.copy()

        prev_rms = float("inf")

        for iteration in range(self.max_icp_iter):
            # 1. Find nearest neighbours in map
            dists, idx = tree.query(src, k=1, distance_upper_bound=self.max_corr_dist)
            valid = dists < self.max_corr_dist

            if valid.sum() < 10:
                # Too few correspondences — diverged
                return None

            src_valid  = src[valid]
            tgt_valid  = map_pts[idx[valid]]
            nrm_valid  = normals[idx[valid]]

            # 2. Point-to-plane error: (p_src - p_tgt) · n_tgt
            diff = src_valid - tgt_valid   # (K, 3)
            pt_plane_err = np.einsum("ij,ij->i", diff, nrm_valid)  # (K,)
            rms = float(np.sqrt(np.mean(pt_plane_err**2)))

            # 3. Check convergence
            if abs(prev_rms - rms) < self.icp_tol and iteration > 2:
                break
            prev_rms = rms

            # 4. Build linear system for 6-DOF: [ω, t] (small-angle approx)
            #    Point-to-plane metric:
            #    minimize Σ ((R·p + t - q) · n)²
            #    Linearised: n · (ω × p) + n · t = n · (q - p)
            #    where ω = (α, β, γ) small rotation vector
            A = np.zeros((len(src_valid), 6), dtype=np.float64)
            b = np.zeros(len(src_valid), dtype=np.float64)

            # Cross products for rotation part: n × p (linearised rotation effect)
            # a_i = [n_y·p_z - n_z·p_y,  n_z·p_x - n_x·p_z,  n_x·p_y - n_y·p_x,  n_x, n_y, n_z]
            nx, ny, nz = nrm_valid[:, 0], nrm_valid[:, 1], nrm_valid[:, 2]
            px, py, pz = src_valid[:, 0], src_valid[:, 1], src_valid[:, 2]

            A[:, 0] = ny * pz - nz * py   # α (rotation around x)
            A[:, 1] = nz * px - nx * pz   # β (rotation around y)
            A[:, 2] = nx * py - ny * px   # γ (rotation around z)
            A[:, 3] = nx                   # tx
            A[:, 4] = ny                   # ty
            A[:, 5] = nz                   # tz

            b[:] = np.einsum("ij,ij->i", (tgt_valid - src_valid), nrm_valid)

            # 5. Solve: [α, β, γ, tx, ty, tz] = (AᵀA)⁻¹ Aᵀb
            try:
                x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                return None

            alpha, beta, gamma = x[0], x[1], x[2]
            tx, ty, tz         = x[3], x[4], x[5]

            # 6. Build incremental rotation from small-angle approximation
            #    Re-orthogonalise with SVD to ensure SO(3)
            Ri = np.array([
                [1,      -gamma,  beta ],
                [gamma,   1,     -alpha],
                [-beta,   alpha,  1    ],
            ])
            U, _, Vt = np.linalg.svd(Ri)
            Ri = U @ Vt   # nearest orthogonal matrix

            # 7. Build incremental transform
            dT = np.eye(4)
            dT[:3, :3] = Ri
            dT[:3,  3] = [tx, ty, tz]

            # 8. Apply to source points and accumulate transform
            src_h = np.hstack([src, np.ones((len(src), 1))])
            src   = (dT @ src_h.T).T[:, :3]
            T     = dT @ T

        # Final RMS check
        dists_final, idx_f = tree.query(src, k=1, distance_upper_bound=self.max_corr_dist)
        valid_f = dists_final < self.max_corr_dist
        if valid_f.sum() < 10:
            return None

        diff_f = src[valid_f] - map_pts[idx_f[valid_f]]
        nrm_f  = normals[idx_f[valid_f]]
        pt_plane_final = np.einsum("ij,ij->i", diff_f, nrm_f)
        rms_final = float(np.sqrt(np.mean(pt_plane_final**2)))
        self._last_rms = rms_final

        if rms_final > self.max_icp_rms:
            # Diverged — reject result
            return None

        # Extract corrected pose from T @ sim_pose_homogeneous
        # Apply total correction T to the sim pose
        R_total = T[:3, :3]
        t_total = T[:3,  3]

        # The ICP correction is in world frame: corrected_pt = R·pt + t
        # Apply to sim position:
        sim_xyz = np.array([sim_pose.x, sim_pose.y, sim_pose.z])
        # T maps scan (world-frame) → corrected (world-frame)
        # Since scan was already in world frame, T is a small correction
        corrected_xyz = R_total @ sim_xyz + t_total

        # Extract roll, pitch, yaw from R_total composed with sim rotation
        R_sim = _euler_to_rot(sim_pose.roll, sim_pose.pitch, sim_pose.yaw)
        R_out = R_total @ R_sim
        roll_out, pitch_out, yaw_out = _rot_to_euler(R_out)

        icp_pose = SLAMPose(
            x=float(corrected_xyz[0]),
            y=float(corrected_xyz[1]),
            z=float(corrected_xyz[2]),
            roll=roll_out,
            pitch=pitch_out,
            yaw=yaw_out,
        )

        self.is_converged = True
        self._icp_count  += 1
        self._last_slam_pose = icp_pose

        # Add the ICP-corrected scan to the map to keep it up to date
        self._add_to_map(src)
        if self._voxels_since_rebuild > self._rebuild_interval // 2:
            self._kdtree_dirty = True

        return icp_pose


# ---------------------------------------------------------------------------
# Rotation utilities
# ---------------------------------------------------------------------------

def _euler_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert ZYX Euler angles (yaw, pitch, roll) to 3×3 rotation matrix."""
    cr, sr = math.cos(roll),  math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw),   math.sin(yaw)

    return np.array([
        [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [-sp,    cp*sr,              cp*cr            ],
    ])


def _rot_to_euler(R: np.ndarray) -> tuple[float, float, float]:
    """Extract (roll, pitch, yaw) from 3×3 rotation matrix (ZYX convention)."""
    sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        roll  = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw   = math.atan2(R[1, 0], R[0, 0])
    else:
        roll  = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw   = 0.0

    return roll, pitch, yaw
