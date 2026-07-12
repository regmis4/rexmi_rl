# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Navigator — top-level loop that ties all nav modules together.

The Navigator runs one step per sim step (50 Hz).  It:
  1. Reads robot pose from the Localizer
  2. Reads raw height-scan hits from the RayCaster sensor
  3. Updates the OccupancyMap
  4. Asks the GlobalPlanner for the next immediate waypoint
  5. Asks the LocalPlanner for (vx, vy, ωz)
  6. Checks RecoveryFSM — overrides command if stuck
  7. Injects the command into the Isaac Lab env's command tensor
  8. Advances the mission waypoint index when waypoint is reached
  9. Writes shared state for the Dashboard thread

Usage (inside the sim loop)
----------------------------
    nav = Navigator(env, mission=Mission.TRAVERSE, ...)
    nav.start_dashboard()   # launches matplotlib daemon thread
    while sim_running:
        sim_step(env)
        nav.step()          # one nav tick
    nav.stop_dashboard()
"""

from __future__ import annotations

import math
import threading
import time
from typing import Optional

import numpy as np
import torch

from rexmi_rl.nav.localizer import SimLocalizer, Pose
from rexmi_rl.nav.mission import Mission, MissionPlanner, Waypoint
from rexmi_rl.nav.occupancy_map import OccupancyMap
from rexmi_rl.nav.global_planner import GlobalPlanner
from rexmi_rl.nav.local_planner import LocalPlanner, LocalPlannerOutput
from rexmi_rl.nav.recovery import RecoveryFSM, RecoveryState
from rexmi_rl.nav.policy_selector import PolicySelector, PolicyMode


class Navigator:
    """
    Top-level navigation controller.

    Parameters
    ----------
    env : IsaacLab VecEnv
        The unwrapped Isaac Lab environment.
    mission : Mission
        Which mission to execute.
    crater_centre : (float, float)
        World-frame (x, y) of the crater centre.
    r_floor, r_rim : float
        Crater floor and rim radii (metres).
    spawn_x : float
        Robot's initial x position (for mission waypoint calculation).
    env_idx : int
        Which parallel env index to control (0 for single-robot nav).
    replan_interval_s : float
        Seconds between A* replans.
    """

    # Index into the command tensor: [vx, vy, omega]
    CMD_VX_IDX    = 0
    CMD_VY_IDX    = 1
    CMD_OMEGA_IDX = 2

    def __init__(
        self,
        env,
        mission:        Mission = Mission.TRAVERSE,
        crater_centre:  tuple[float, float] = (0.0, 0.0),
        r_floor:        float = 3.0,
        r_rim:          float = 11.0,
        spawn_x:        float = -18.0,
        entry_azimuth_deg: float = 180.0,
        env_idx:        int   = 0,
        replan_interval_s: float = 2.0,
    ):
        self._env     = env
        self._env_idx = env_idx

        # Resolve robot articulation and sensors from env
        unwrapped = env.unwrapped
        self._robot   = unwrapped.scene["robot"]
        try:
            self._scanner = unwrapped.scene["height_scanner"]
        except KeyError:
            self._scanner = None
        # Forward scanner: fires ±60° ahead at body height, 5 m range.
        # Read alongside the downward height scanner to give the nav layer
        # early warning of rocks, walls, and crater rims ahead.
        try:
            self._fwd_scanner = unwrapped.scene["forward_scanner"]
        except KeyError:
            self._fwd_scanner = None

        # Nav modules
        self._localizer = SimLocalizer(self._robot, env_idx)
        self._omap      = OccupancyMap(world_size=64.0, cell_size=0.5,
                                       origin=crater_centre)
        self._global    = GlobalPlanner(self._omap,
                                        replan_interval_s=replan_interval_s)
        self._local     = LocalPlanner()
        self._recovery  = RecoveryFSM()

        # Mission waypoints
        self._mission_planner = MissionPlanner(
            crater_centre=crater_centre,
            r_floor=r_floor,
            r_rim=r_rim,
            spawn_x=spawn_x,
            entry_azimuth_deg=entry_azimuth_deg,
        )
        self._waypoints = self._mission_planner.get_waypoints(mission)
        self._wp_idx    = 0

        # Set first waypoint in global planner
        if self._waypoints:
            wp = self._waypoints[0]
            self._global.set_goal(wp.x, wp.y)

        # Shared state for dashboard (written every step, read by dashboard thread)
        empty = np.array([], dtype=np.float32)
        self.shared: dict = {
            "pose":             Pose(0, 0, 0, 0),
            "speed":            0.0,
            "waypoints":        self._waypoints,
            "wp_idx":           0,
            "planned_path":     [],
            "local_out":        None,
            "recovery_status":  "NAVIGATING",
            "mission":          mission.value,
            "cmd":              (0.0, 0.0, 0.0),
            "cloud_xyz":        (empty, empty, empty),       # downward scan (viridis)
            "fwd_cloud_xyz":    (empty, empty, empty),       # forward scan (orange)
            "cost_grid":        None,
            "step_count":       0,
        }
        self._lock = threading.Lock()

        # Policy selector (set externally by navigate.py after loading checkpoints)
        self.policy_selector: Optional[PolicySelector] = None

        # Dashboard handle (set by start_dashboard)
        self._dashboard = None
        self._dashboard_thread: Optional[threading.Thread] = None
        self._stop_dashboard = threading.Event()

    # ------------------------------------------------------------------
    # Main step (call once per sim step)
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Execute one navigation tick. Call after each sim step."""
        # 1. Localise
        pose = self._localizer.get_pose()
        vx_w, vy_w, _ = self._localizer.get_velocity()
        speed = math.hypot(vx_w, vy_w)

        # 2. Update occupancy map from raw scanner hits
        #    2a. Downward height scanner (1.6 m × 1.0 m, 160 rays, fine detail)
        if self._scanner is not None:
            try:
                hits_w = self._scanner.data.ray_hits_w   # (n_envs, N_rays, 3)
                pts = hits_w[self._env_idx].cpu().numpy()  # (N_rays, 3)
                # Filter invalid hits (Isaac Lab marks misses with very large z)
                valid = np.isfinite(pts[:, 2]) & (pts[:, 2] < 1e5)
                if valid.any():
                    self._omap.update(pts[valid])
            except Exception:
                pass

        #    2b. Forward scanner (±60° ahead, 5 m range, 75 rays)
        #    Feeds occupancy map for A* cost.
        #    Also stored directly in shared["fwd_cloud_xyz"] as the CURRENT frame
        #    only (no history) — dashboard renders it orange immediately ahead of
        #    the robot.  Using current-frame only avoids the trailing-cloud artefact
        #    that appeared when old forward hits (now behind the robot) were kept.
        _fwd_cloud_this_step: "tuple[np.ndarray, np.ndarray, np.ndarray] | None" = None
        if self._fwd_scanner is not None:
            try:
                fwd_hits_w = self._fwd_scanner.data.ray_hits_w  # (n_envs, N_rays, 3)
                fwd_pts = fwd_hits_w[self._env_idx].cpu().numpy()
                valid_fwd = (
                    np.isfinite(fwd_pts[:, 2])
                    & (fwd_pts[:, 2] < 1e5)
                    & (np.isfinite(fwd_pts[:, 0]))
                )
                if valid_fwd.any():
                    raw_fwd = fwd_pts[valid_fwd]

                    # Body-frame forward filter: keep only hits that are
                    # genuinely AHEAD of the robot (x_body > 0.1 m).
                    # This eliminates any rear-facing rays or sensor-mount
                    # artifacts that make the orange cloud appear behind.
                    cos_yaw = math.cos(pose.yaw)
                    sin_yaw = math.sin(pose.yaw)
                    dx_w = raw_fwd[:, 0] - pose.x
                    dy_w = raw_fwd[:, 1] - pose.y
                    # x_body = forward component in robot frame
                    x_body = dx_w * cos_yaw + dy_w * sin_yaw
                    front_mask = x_body > 0.1   # must be at least 10 cm ahead
                    clean_fwd = raw_fwd[front_mask]

                    if len(clean_fwd) > 0:
                        self._omap.update(clean_fwd)   # grid update (A* cost)
                        # Store current frame as (xs, ys, zs) for dashboard
                        _fwd_cloud_this_step = (
                            clean_fwd[:, 0].astype(np.float32),
                            clean_fwd[:, 1].astype(np.float32),
                            clean_fwd[:, 2].astype(np.float32),
                        )
            except Exception:
                pass

        # 3. Advance mission waypoint if reached
        self._advance_waypoint(pose)

        if self._wp_idx >= len(self._waypoints):
            # Mission complete — stop
            self._inject_command(0.0, 0.0, 0.0)
            return

        # 4. Global planner → immediate waypoint
        # Pass the mission waypoint as goal_x/goal_y so the sanity check in
        # GlobalPlanner.update() can detect when A* is pointing the wrong way
        # and fall back to a direct bearing on an unknown map.
        current_wp = self._waypoints[self._wp_idx]
        imm_wp = self._global.update(
            pose.x, pose.y,
            goal_x=current_wp.x, goal_y=current_wp.y,
        )
        if imm_wp is None:
            imm_wp = (current_wp.x, current_wp.y)

        # 5. Local planner → (vx, vy, omega)
        #    Use compute_with_forward so the forward scanner provides early
        #    obstacle warning and speed reduction before contact.
        scan_heights = self._get_scan_heights()
        fwd_pts = self._get_forward_hits()
        robot_pos = (pose.x, pose.y, pose.z)
        local_out = self._local.compute_with_forward(
            scan_heights, pose.yaw,
            pose.x, pose.y,
            imm_wp[0], imm_wp[1],
            fwd_hits_world=fwd_pts,
            robot_pos_w=robot_pos,
        )

        # 6. Policy selector — choose which RL policy runs this step
        policy_status = "fixed"
        if self.policy_selector is not None:
            # Extract terrain metrics from local_out for selector decision
            trav_count = sum(1 for t in local_out.traversable_mask if t)
            trav_frac  = trav_count / max(1, len(local_out.traversable_mask))
            # max_step: worst step across all 10 columns (recompute from scan)
            scan = [[scan_heights[i * 10 + j] for j in range(10)] for i in range(16)]
            max_step = max(
                max(abs(scan[i+1][j] - scan[i][j]) for i in range(15))
                for j in range(10)
            )
            self.policy_selector.update(
                slope_ahead=local_out.slope_ahead if not (
                    local_out.slope_ahead != local_out.slope_ahead  # nan check
                ) else 0.0,
                max_step=max_step,
                traversable_fraction=trav_frac,
            )
            policy_status = self.policy_selector.status_str()

        # 7. Recovery FSM — pass current distance to waypoint for progress tracking
        dist_to_wp = math.hypot(pose.x - current_wp.x, pose.y - current_wp.y)
        rec_vx, rec_vy, rec_omega = self._recovery.update(
            speed, local_out.heading_error, waypoint_dist=dist_to_wp
        )

        if self._recovery.is_blocked:
            # Force global replan around the blocked area
            self._global.set_goal(current_wp.x, current_wp.y)
            self._recovery.reset()

        if self._recovery.is_recovering:
            cmd = (rec_vx, rec_vy, rec_omega)
        else:
            cmd = (local_out.vx, local_out.vy, local_out.omega)

        # 8. Inject command into env
        self._inject_command(*cmd)

        # 9. Update shared state for dashboard
        with self._lock:
            step_n = self.shared["step_count"] + 1
            self.shared.update({
                "pose":            pose,
                "speed":           speed,
                "wp_idx":          self._wp_idx,
                "planned_path":    self._global.get_path_world(),
                "local_out":       local_out,
                "recovery_status": self._recovery.status_str(),
                "policy_status":   policy_status,
                "cmd":             cmd,
                "step_count":      step_n,
            })
            # Update point clouds and cost grid every 5 steps for responsive dashboard
            # Forward cloud: always write the current frame (no deque — no trail)
            if _fwd_cloud_this_step is not None:
                self.shared["fwd_cloud_xyz"] = _fwd_cloud_this_step
            # Downward cloud and cost grid update every 5 steps (heavier ops)
            if step_n % 5 == 0:
                self.shared["cloud_xyz"]  = self._omap.get_point_cloud()
                self.shared["cost_grid"]  = self._omap.get_cost_grid()

    # ------------------------------------------------------------------
    # Dashboard control
    # ------------------------------------------------------------------

    def start_dashboard(self) -> None:
        """Launch the matplotlib dashboard in a daemon thread."""
        from rexmi_rl.nav.dashboard import Dashboard
        self._dashboard = Dashboard(self.shared, self._lock,
                                    self._omap, self._waypoints)
        self._stop_dashboard.clear()
        self._dashboard_thread = threading.Thread(
            target=self._dashboard.run,
            args=(self._stop_dashboard,),
            daemon=True,
            name="nav-dashboard",
        )
        self._dashboard_thread.start()

    def stop_dashboard(self) -> None:
        """Signal the dashboard thread to stop."""
        self._stop_dashboard.set()
        if self._dashboard_thread is not None:
            self._dashboard_thread.join(timeout=3.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _advance_waypoint(self, pose: Pose) -> None:
        """Check if current waypoint is reached; if so, advance to next."""
        if self._wp_idx >= len(self._waypoints):
            return
        wp = self._waypoints[self._wp_idx]
        dist = math.hypot(pose.x - wp.x, pose.y - wp.y)
        if dist < wp.arrival_radius:
            self._wp_idx += 1
            if self._wp_idx < len(self._waypoints):
                nwp = self._waypoints[self._wp_idx]
                self._global.set_goal(nwp.x, nwp.y)
                self._recovery.reset()   # fresh slate for each waypoint

    def _get_scan_heights(self) -> list[float]:
        """
        Extract 160 height values from the RayCaster sensor.

        Returns heights relative to robot base (metres).
        Falls back to a flat zero scan if scanner unavailable.
        """
        if self._scanner is None:
            return [0.0] * 160

        try:
            hits_w = self._scanner.data.ray_hits_w   # (n_envs, 160, 3)
            robot_z = float(self._robot.data.root_pos_w[self._env_idx, 2])
            z_hits  = hits_w[self._env_idx, :, 2].cpu().tolist()  # (160,)
            # Convert absolute z to relative (terrain height below/above base)
            return [z - robot_z for z in z_hits]
        except Exception:
            return [0.0] * 160

    def _get_forward_hits(self) -> "np.ndarray | None":
        """
        Return filtered world-frame hits from the forward scanner as (N, 3) array.

        Returns None if the forward scanner is unavailable or has no valid hits.
        The LocalPlanner uses these to detect obstacles in the forward danger zone
        and scale vx accordingly before contact.
        """
        if self._fwd_scanner is None:
            return None
        try:
            hits_w = self._fwd_scanner.data.ray_hits_w  # (n_envs, N_rays, 3)
            pts = hits_w[self._env_idx].cpu().numpy()   # (N_rays, 3)
            valid = (
                np.isfinite(pts[:, 2])
                & (pts[:, 2] < 1e5)
                & np.isfinite(pts[:, 0])
            )
            if valid.any():
                return pts[valid]
        except Exception:
            pass
        return None

    def _inject_command(self, vx: float, vy: float, omega: float) -> None:
        """
        Write (vx, vy, omega) into the Isaac Lab command manager buffer.

        The command_manager stores velocity commands as a (n_envs, 3) tensor.
        We overwrite only the row for our env_idx.
        """
        try:
            cmd_tensor = self._env.unwrapped.command_manager.get_command(
                "base_velocity"
            )   # shape (n_envs, 3)
            cmd_tensor[self._env_idx, 0] = vx
            cmd_tensor[self._env_idx, 1] = vy
            cmd_tensor[self._env_idx, 2] = omega
        except Exception:
            pass
