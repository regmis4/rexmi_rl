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

from rexmi_rl.nav.localizer import SimLocalizer, SLAMLocalizer, Pose
from rexmi_rl.nav.slam import LidarSLAM, SLAMPose
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
        # 360° LiDAR: 18-channel Unitree L1 simulation, 3240 pts/scan @ 10 Hz.
        # Used for 3D SLAM and dense occupancy map updates (nav layer only).
        try:
            self._lidar = unwrapped.scene["lidar"]
        except KeyError:
            self._lidar = None

        # SLAM engine — 3D ICP scan matching + voxel map
        # Bootstraps on sim pose for 3 s, then refines with ICP.
        self._slam = LidarSLAM(
            voxel_size=0.10,
            bootstrap_s=3.0,
            max_icp_iter=10,   # 10 iterations sufficient for incremental matching
            max_icp_rms=0.20,  # tightened: reject diverged ICP at 20 cm (was 30 cm)
                               # icp_count only increments on accepted runs → BOOT exit
                               # condition (icp_count >= 5) is a reliable convergence gate
        )

        # SLAM background thread — ICP runs at LiDAR rate (10 Hz) without
        # blocking the 50 Hz sim thread.  The sim thread deposits LiDAR scans
        # into a queue; the SLAM thread drains it and updates shared pose.
        self._slam_queue:     "queue.Queue" = __import__("queue").Queue(maxsize=2)
        self._slam_pose_lock: threading.Lock = threading.Lock()
        self._slam_last_pose: "SLAMPose | None" = None
        self._slam_thread_stop = threading.Event()
        self._slam_bg_thread   = threading.Thread(
            target=self._slam_worker,
            daemon=True,
            name="nav-slam",
        )
        self._slam_bg_thread.start()

        # Nav modules
        # -------------------------------------------------------------------
        # Pose architecture: DECOUPLE slam pose from nav pose
        # -------------------------------------------------------------------
        # SLAM is used ONLY for building the occupancy map (LiDAR hits → omap).
        # Navigation pose (heading error, waypoint distance) uses odometry.
        #
        # Why: ICP on crater-slope terrain produces jumpy poses (single divergent
        # frames cause 3–5 m position spikes) that corrupt heading error computation
        # and cause the robot to spin away from the goal.  The occupancy map is
        # unaffected because it is updated from raw LiDAR hit positions (using
        # odometry-bootstrapped sim pose) — it does not go through SLAMLocalizer.
        #
        # In simulation: odometry = SimLocalizer (perfect velocity integration).
        # In deployment: replace with OdometryLocalizer (wheel encoders + IMU).
        #   SLAM corrections are applied to odometry only when ICP has been
        #   stable for ≥10 consecutive frames (RMS < 0.10 m).
        # -------------------------------------------------------------------
        _odom_localizer = SimLocalizer(self._robot, env_idx)   # = odometry in deployment
        if self._lidar is not None:
            _slam_localizer = SLAMLocalizer(self._slam, _odom_localizer)
            print("[Navigator] LiDAR detected — SLAM localizer active for nav pose; "
                  "falls back to odometry when ICP is not yet converged.")
        else:
            _slam_localizer = None
            print("[Navigator] No LiDAR sensor — odometry only")

        # Nav pose: use SLAM localizer when LiDAR is available.
        # SLAMLocalizer blends ICP-corrected pose with odometry — it returns the
        # SLAM pose only when ICP has converged (icp_count ≥ 5, RMS < 0.20 m),
        # and transparently falls back to odometry on diverged frames.
        # This gives the full benefit of SLAM loop closure while staying robust
        # to the rare ICP divergence that previously caused 3–5 m position spikes.
        # In deployment: replace SimLocalizer with OdometryLocalizer (encoders+IMU).
        if _slam_localizer is not None:
            self._localizer = _slam_localizer   # ICP-corrected when stable
        else:
            self._localizer = _odom_localizer   # odometry only (no LiDAR)
        self._sim_localizer = _odom_localizer   # always pure odometry — used in BOOT loop

        self._omap = OccupancyMap(world_size=64.0, cell_size=0.20,
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

        # Per-waypoint closest approach tracking — prevents immediate skip at spawn.
        # A waypoint is only considered "reached" after the robot has actually
        # approached it (driven to within 1.5 × arrival_radius at some point).
        # Without this, a robot spawning near WP[0] triggers arrival immediately
        # before the first step, skipping straight to WP[2].
        self._wp_closest_dist: list[float] = [math.inf] * len(self._waypoints)

        # Set first waypoint in global planner
        if self._waypoints:
            wp = self._waypoints[0]
            self._global.set_goal(wp.x, wp.y)

        # ------------------------------------------------------------------
        # BOOT phase — hold still while SLAM warms up
        # ------------------------------------------------------------------
        # The LiDAR SLAM needs several seconds of stationary scanning to build
        # a dense enough local map for ICP to converge reliably.  If the robot
        # moves immediately, ICP runs on a sparse, rapidly-changing map and
        # produces noisy/jumping pose estimates that corrupt all nav decisions.
        #
        # During BOOT:
        #   • Command = (vx=0, vy=0, omega=0) — hold still
        #   • SLAM, occupancy map, and sensor processing all run normally
        #   • Exit BOOT when: elapsed ≥ boot_min_s AND slam is stable
        #     (stable = slam.is_converged AND last_rms < 0.15 AND icp_count ≥ 5
        #      AND map has ≥ 1000 voxels for dense enough ICP coverage)
        #   • If LiDAR is not present: skip BOOT immediately (no SLAM to warm up)
        #
        # Adaptive boot: minimum is 3 s (not 8 s) — if SLAM is fully stable by
        # 3 s we start navigating immediately.  The 15 s hard timeout is kept
        # as a safety net for challenging terrain where ICP never converges.
        # ------------------------------------------------------------------
        import time as _time_module
        self._boot_start_t: float = _time_module.monotonic()
        self._boot_min_s:   float = 3.0    # minimum stationary time (s) — adaptive exit
                                           # allows early start when SLAM converges fast
        self._boot_max_s:   float = 15.0   # hard timeout — start navigating regardless
                                           # of SLAM state if this elapses.  On smooth
                                           # crater-slope terrain ICP RMS stays ~0.4 m
                                           # indefinitely; waiting for RMS < 0.15 would
                                           # block the robot forever.
        self._in_boot:      bool  = (self._lidar is not None)  # skip if no LiDAR
        if self._in_boot:
            print(f"[Navigator] BOOT phase: adaptive exit after ≥{self._boot_min_s:.0f}s "
                  f"once SLAM stable (hard limit {self._boot_max_s:.0f}s) ...")
        else:
            print("[Navigator] No LiDAR — skipping BOOT phase, navigating immediately")

        # SLAM → omap sync interval: every N steps push the full SLAM map into the
        # occupancy map so A* can route around SLAM-mapped obstacles.
        self._slam_omap_sync_interval: int = 100   # every 2 s at 50 Hz
        self._last_slam_sync_step:     int = 0

        # Shared state for dashboard (written every step, read by dashboard thread)
        empty = np.array([], dtype=np.float32)
        # Trajectory ring buffer — last 500 (x, y) positions for dashboard trace
        import collections
        self._trajectory: "collections.deque[tuple[float,float]]" = collections.deque(maxlen=500)
        self.shared: dict = {
            "pose":             Pose(0, 0, 0, 0),
            "speed":            0.0,
            "waypoints":        self._waypoints,
            "wp_idx":           0,
            "planned_path":     [],
            "local_out":        None,
            "recovery_status":  "BOOT",
            "mission":          mission.value,
            "cmd":              (0.0, 0.0, 0.0),
            "cloud_xyz":        (empty, empty, empty),       # downward scan (viridis)
            "fwd_cloud_xyz":    (empty, empty, empty),       # forward scan (orange)
            "lidar_cloud":      np.zeros((0, 3), dtype=np.float32),  # SLAM map cloud
            "slam_converged":   False,                       # SLAM ICP has locked on
            "slam_map_size":    0,                           # number of voxels in map
            "slam_rms":         0.0,                         # last ICP RMS error
            "slam_icp_count":   0,                           # total ICP runs
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

        # ──────────────────────────────────────────────────────────────────
        # BOOT PHASE — hold still until SLAM has warmed up
        # ──────────────────────────────────────────────────────────────────
        # All sensor processing (LiDAR, SLAM, omap) still runs so the map
        # is being built while stationary.  Only the drive command is zeroed.
        if self._in_boot:
            import time as _t
            elapsed = _t.monotonic() - self._boot_start_t
            slam_ok = (
                self._slam.is_converged
                and self._slam.last_rms < 0.15
                and self._slam.icp_count >= 5
            )
            timed_out = elapsed >= self._boot_max_s
            if (elapsed >= self._boot_min_s and slam_ok) or timed_out:
                self._in_boot = False
                reason = "SLAM converged" if slam_ok else f"timeout ({elapsed:.1f}s)"
                print(
                    f"[Navigator] BOOT complete — {reason} "
                    f"(RMS={self._slam.last_rms:.3f}m, icp_n={self._slam.icp_count}) "
                    f"— starting navigation"
                )
            else:
                # Still in BOOT — process sensors but hold still
                self._process_sensors_only()

                # Read actual pose from sim localizer so dashboard shows real position
                # (shared["pose"] was initialized to 0,0,0 and must be updated each step)
                _boot_pose = self._sim_localizer.get_pose()

                # Print countdown every 50 steps (~1 s)
                step_n_now = self.shared.get("step_count", 0) + 1
                if step_n_now % 50 == 0:
                    remain = max(0.0, self._boot_min_s - elapsed)
                    remain_max = max(0.0, self._boot_max_s - elapsed)
                    rms_str = f"{self._slam.last_rms:.3f}m" if self._slam.last_rms > 0 else "?"
                    print(
                        f"[Nav][BOOT] {remain:.1f}s remain (hard limit {remain_max:.1f}s) | "
                        f"RMS={rms_str} icp_n={self._slam.icp_count} "
                        f"converged={self._slam.is_converged}"
                    )
                self._inject_command(0.0, 0.0, 0.0)
                with self._lock:
                    self.shared["step_count"] = self.shared.get("step_count", 0) + 1
                    self.shared["recovery_status"] = "BOOT"
                    self.shared["pose"]            = _boot_pose   # show real position
                    self.shared["slam_converged"]  = self._slam.is_converged
                    self.shared["slam_rms"]        = self._slam.last_rms
                    self.shared["slam_icp_count"]  = self._slam.icp_count
                    # Update cost grid during BOOT so map is visible on dashboard
                    step_n = self.shared["step_count"]
                    if step_n % 10 == 0:
                        self.shared["cost_grid"] = self._omap.get_cost_grid()
                return
        # ──────────────────────────────────────────────────────────────────

        # 0. Process 360° LiDAR scan → SLAM update → dense omap update
        #    The LiDAR runs at 10 Hz (update_period=0.1 s); Isaac Lab only
        #    provides new data when the sensor has actually fired.  We detect
        #    a new scan by checking whether the sensor's current sim-time
        #    differs from the last time we processed it.
        _slam_converged = self._slam.is_converged
        _slam_map_pts: "np.ndarray | None" = None
        if self._lidar is not None:
            try:
                lidar_hits = self._lidar.data.ray_hits_w   # (n_envs, N_rays, 3)
                raw_pts = lidar_hits[self._env_idx].cpu().numpy()   # (3240, 3)

                # Filter: remove NaN, inf, and sensor-miss sentinel values
                valid_mask = (
                    np.isfinite(raw_pts[:, 0])
                    & np.isfinite(raw_pts[:, 1])
                    & np.isfinite(raw_pts[:, 2])
                    & (raw_pts[:, 2] < 1e5)   # Isaac Lab miss sentinel
                )
                clean_pts = raw_pts[valid_mask]

                if len(clean_pts) > 10:
                    # Build sim pose for SLAM bootstrap/fallback
                    sim_p = self._sim_localizer.get_pose()

                    # Compute roll/pitch from quaternion for full 6-DOF
                    quat = self._robot.data.root_quat_w[self._env_idx]
                    w, qx, qy, qz = (float(quat[0]), float(quat[1]),
                                     float(quat[2]), float(quat[3]))
                    roll_s  = math.atan2(2*(w*qx + qy*qz),
                                         1 - 2*(qx*qx + qy*qy))
                    sinp = 2*(w*qy - qz*qx)
                    sinp = max(-1.0, min(1.0, sinp))
                    pitch_s = math.asin(sinp)

                    slam_input = SLAMPose(
                        x=sim_p.x, y=sim_p.y, z=sim_p.z,
                        roll=roll_s, pitch=pitch_s, yaw=sim_p.yaw,
                    )

                    # ── SLAM ICP runs in background thread — non-blocking ──
                    # Drop scan if background thread is still busy (queue full).
                    # This keeps the sim thread at full 50 Hz speed.
                    try:
                        self._slam_queue.put_nowait((clean_pts.copy(), slam_input))
                    except Exception:
                        pass  # queue full — skip this scan (ICP catching up)

                    # Feed dense LiDAR hits to occupancy map (fast vectorised op)
                    self._omap.update_lidar(clean_pts)

                    # Every 50 steps, snapshot the SLAM map for dashboard + omap
                    step_now = self.shared.get("step_count", 0)
                    if step_now % 50 == 0 and self._slam.get_map_size() > 100:
                        _slam_map_pts = self._slam.get_map_points()

            except Exception as _e:
                pass   # never crash the nav loop on sensor error

        # 1. Localise (odometry — SimLocalizer reads root_pos_w directly)
        pose = self._localizer.get_pose()
        vx_w, vy_w, _ = self._localizer.get_velocity()
        speed = math.hypot(vx_w, vy_w)

        # Physics-reset detector: Isaac Lab may reset the robot (fall/termination)
        # without the nav stack knowing.  Detect via sudden large position jump.
        _prev_pose = self.shared.get("pose")
        if _prev_pose is not None and hasattr(_prev_pose, "x"):
            _jump = math.hypot(pose.x - _prev_pose.x, pose.y - _prev_pose.y)
            if _jump > 3.0:
                print(
                    f"[Nav][RESET] Physics reset detected at step "
                    f"{self.shared.get('step_count', 0)+1}: "
                    f"pos jumped {_jump:.1f}m "
                    f"({_prev_pose.x:+.1f},{_prev_pose.y:+.1f}) → "
                    f"({pose.x:+.1f},{pose.y:+.1f}). "
                    f"Forcing A* replan from new position."
                )
                # Force A* to replan from the new position
                if self._wp_idx < len(self._waypoints):
                    _wp = self._waypoints[self._wp_idx]
                    self._global.set_goal(_wp.x, _wp.y)
                self._recovery.reset()   # clear stuck state from pre-reset run
                # Clear committed-turn latch — the robot teleported to a new
                # position and heading; the old committed omega sign is wrong.
                # Without this, the robot would continue spinning in the pre-reset
                # direction even though it may now be correctly aimed at the goal.
                self._local._turn_committed      = False
                self._local._committed_omega_sign = 0.0
                # Reset progress tracking in recovery so the new position gets
                # a fresh 60-s window to make progress toward the waypoint.
                self._recovery._last_progress_time = None
                self._recovery._best_dist          = math.inf

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

            # Force TURN mode when RecoveryFSM is executing a rotation manoeuvre.
            # is_rotating is True during REVERSING and ROTATING states — the robot
            # needs vx=0 capability that rough/rocky_slope don't reliably provide.
            # We override heading_error to 180° (worst case) so the turn threshold
            # is guaranteed to fire regardless of the actual heading at that moment.
            if self._recovery.is_rotating:
                _he_for_selector = math.pi   # force turn-override in PolicySelector
            else:
                _he_for_selector = local_out.heading_error

            self.policy_selector.update(
                slope_ahead=local_out.slope_ahead if not (
                    local_out.slope_ahead != local_out.slope_ahead  # nan check
                ) else 0.0,
                heading_error_rad=_he_for_selector,
                max_step=max_step,
                traversable_fraction=trav_frac,
            )
            policy_status = self.policy_selector.status_str()

            # Update local planner's vx_normal to match the active policy's
            # in-distribution speed.  This ensures the robot drives at the speed
            # the policy was trained at (fast_flat=1.5 m/s, rough=0.45, rocky=0.40)
            # rather than the CLI default, which may be out-of-distribution.
            self._local.vx_normal = self.policy_selector.current_vx()

        # 7. Recovery FSM — pass current distance to waypoint for progress tracking
        dist_to_wp = math.hypot(pose.x - current_wp.x, pose.y - current_wp.y)
        rec_vx, rec_vy, rec_omega = self._recovery.update(
            speed, local_out.heading_error, waypoint_dist=dist_to_wp
        )

        if self._recovery.is_blocked:
            # ----------------------------------------------------------------
            # BLOCKED handler — inflate obstacle radius + optionally skip WP
            # ----------------------------------------------------------------
            # 1. Inflate a 1.0 m radius around the robot's current position.
            #    inflate_blocked() marks a hard-block core (0-1 cells) and a
            #    cost-gradient halo (2-5 cells) that A* will actively avoid.
            #    This prevents the robot from being sent straight back to the
            #    same boulder after every replan.
            #
            # 2. If the robot has been BLOCKED at this waypoint more than once,
            #    skip to the next waypoint.  Repeated BLOCKs at the same WP
            #    mean the waypoint itself is unreachable from the current
            #    position — keep trying forever is futile.
            # ----------------------------------------------------------------
            _fwd_x = pose.x + 0.5 * math.cos(pose.yaw)
            _fwd_y = pose.y + 0.5 * math.sin(pose.yaw)

            # Inflate the obstacle area in the costmap (1.0 m radius)
            self._omap.inflate_blocked(_fwd_x, _fwd_y, radius_m=1.0)
            # Also inflate at the robot's current position to push future paths away
            self._omap.inflate_blocked(pose.x, pose.y, radius_m=0.6)

            # Count how many times we've been BLOCKED at this waypoint
            _block_key = self._wp_idx
            if not hasattr(self, "_block_count"):
                self._block_count: dict = {}
            self._block_count[_block_key] = self._block_count.get(_block_key, 0) + 1

            if self._block_count[_block_key] >= 3:
                # 3 strikes — skip to next waypoint rather than hammering
                # the same impassable boulder indefinitely
                self._block_count[_block_key] = 0
                print(f"[Nav] BLOCKED at ({pose.x:.1f},{pose.y:.1f}) × 3 — "
                      f"skipping WP[{self._wp_idx}] → WP[{self._wp_idx+1}]")
                self._wp_idx += 1
                if self._wp_idx < len(self._waypoints):
                    nwp = self._waypoints[self._wp_idx]
                    self._global.set_goal(nwp.x, nwp.y)
            else:
                # Replan — A* will route around the inflated area
                self._global.set_goal(current_wp.x, current_wp.y)
                print(f"[Nav] BLOCKED at ({pose.x:.1f},{pose.y:.1f}) "
                      f"(×{self._block_count[_block_key]}) — "
                      f"inflated 1.0m radius, forcing replan")

            # Reset recovery FSM so it can detect the next stuck event
            self._recovery.reset()

        if self._recovery.is_recovering:
            cmd = (rec_vx, rec_vy, rec_omega)
        else:
            cmd = (local_out.vx, local_out.vy, local_out.omega)

        # 8. Inject command into env
        self._inject_command(*cmd)

        # 8b. Per-50-step diagnostic log
        step_n_now = self.shared.get("step_count", 0) + 1
        if step_n_now % 50 == 0:
            _cell = self._omap.world_to_cell(pose.x, pose.y)
            _cell_cost = (
                self._omap.traversal_cost(_cell[0], _cell[1])
                if _cell else -1.0
            )
            _he_deg  = math.degrees(local_out.heading_error)
            _omega   = cmd[2]
            _vx      = cmd[0]
            _fwd_obs = getattr(local_out, "fwd_obstacle_dist", math.inf)
            _fwd_str = f" fwd_obs={_fwd_obs:.1f}m" if _fwd_obs < 5.0 else ""
            _state   = self._recovery.status_str()
            # Also show the normalized omega value the policy actually sees in its obs
            # cmd tensor index 2 = omega; command manager normalizes it via ang_vel_z range
            try:
                _cmd_tensor = self._env.unwrapped.command_manager.get_command("base_velocity")
                _omega_obs = float(_cmd_tensor[self._env_idx, 2])
            except Exception:
                _omega_obs = float("nan")
            print(
                f"[Nav][{step_n_now:5d}] "
                f"he={_he_deg:+.1f}° "
                f"ω_cmd={_omega:+.2f} ω_obs={_omega_obs:+.2f} "
                f"vx={_vx:+.2f} "
                f"cell_cost={_cell_cost:.1f} "
                f"dist_wp={dist_to_wp:.1f}m "
                f"imm_wp=({imm_wp[0]:+.1f},{imm_wp[1]:+.1f}) "
                f"state={_state}"
                f"{_fwd_str}"
            )

        # 9. Update shared state for dashboard
        self._trajectory.append((pose.x, pose.y))
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
                "trajectory":      list(self._trajectory),
            })
            # Update point clouds and cost grid every 5 steps for responsive dashboard
            # Forward cloud: always write the current frame (no deque — no trail)
            if _fwd_cloud_this_step is not None:
                self.shared["fwd_cloud_xyz"] = _fwd_cloud_this_step
            # SLAM status — always update so dashboard reflects current ICP state
            self.shared["slam_converged"] = self._slam.is_converged
            self.shared["slam_map_size"]  = self._slam.get_map_size()
            self.shared["slam_rms"]       = self._slam.last_rms
            self.shared["slam_icp_count"] = self._slam.icp_count

            # SLAM map → shared dict: publish every 50 steps so dashboard shows
            # the live-growing 3D cloud.  Also used by update_slam_map() below.
            if _slam_map_pts is not None:
                self.shared["lidar_cloud"] = _slam_map_pts

            # Cost grid update every 10 steps (~5 Hz) — feeds the dashboard.
            # get_cost_grid() is now vectorised (<1 ms) so this is very cheap.
            # Reduced from every 5 steps to further trim lock-hold time.
            if step_n % 10 == 0:
                self.shared["cost_grid"] = self._omap.get_cost_grid()

        # ------------------------------------------------------------------
        # SLAM → OccupancyMap full-map sync (every 2 s, outside the lock)
        # ------------------------------------------------------------------
        # Flush the full accumulated SLAM voxel map into the costmap so A*
        # can route around SLAM-discovered obstacles.  Without this sync,
        # SLAM builds a beautiful map but the A* grid only knows about
        # LiDAR hits from the current frame — it misses walls and boulders
        # seen earlier once the robot has moved past them.
        #
        # Called outside the shared-dict lock to avoid holding it during the
        # O(M) _update_cells_numpy() call.  _slam_map_pts is a local snapshot
        # (copied in get_map_points()) so it's safe to use without locks.
        # ------------------------------------------------------------------
        if _slam_map_pts is not None:
            slam_map_size = len(_slam_map_pts)
            if (slam_map_size > 500
                    and step_n_now - self._last_slam_sync_step
                    >= self._slam_omap_sync_interval):
                self._omap.update_slam_map(_slam_map_pts)
                self._last_slam_sync_step = step_n_now

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
        # Also stop SLAM background thread
        self._slam_thread_stop.set()
        self._slam_bg_thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # SLAM background worker
    # ------------------------------------------------------------------

    def _slam_worker(self) -> None:
        """
        Background thread: drains the LiDAR scan queue and runs ICP.

        Runs at ~10 Hz (LiDAR rate).  The sim thread (50 Hz) only does fast ops:
          • sensor read + numpy filter (< 0.5 ms)
          • omap.update_lidar() — vectorised (< 1 ms)
          • queue.put_nowait() — O(1)

        This thread does the heavy ICP:
          • voxel downsample (~0.1 ms)
          • _add_to_map (~1 ms)
          • KD-tree query + 10 ICP iterations (~10-15 ms)

        Drops scan if queue is full (maxsize=2) — ICP falling behind is better
        than blocking the sim thread.
        """
        import queue as _queue

        while not self._slam_thread_stop.is_set():
            try:
                item = self._slam_queue.get(timeout=0.1)
            except _queue.Empty:
                continue

            clean_pts, slam_input = item
            try:
                pose = self._slam.update(clean_pts, slam_input)
                with self._slam_pose_lock:
                    self._slam_last_pose = pose
            except Exception as _e:
                pass   # never crash SLAM thread

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _advance_waypoint(self, pose: Pose) -> None:
        """
        Check if current waypoint is reached; if so, advance to next.

        Closest-approach gate prevents immediate waypoint skip at spawn.
        The robot must have actually approached to within 1.5 × arrival_radius
        before the arrival check is armed.  This handles the case where the
        robot spawns within arrival_radius of the first waypoint (e.g. spawn_x=+13
        and approach WP at x=+12 with arrival_radius=2 m) — without this gate
        the waypoint would be skipped in the very first step before the robot moves.
        """
        if self._wp_idx >= len(self._waypoints):
            return
        wp = self._waypoints[self._wp_idx]
        dist = math.hypot(pose.x - wp.x, pose.y - wp.y)

        # Update closest-distance tracking for this waypoint
        if dist < self._wp_closest_dist[self._wp_idx]:
            self._wp_closest_dist[self._wp_idx] = dist

        # Only arm arrival check once the robot has genuinely approached.
        # Gate: closest-ever distance < 1.5 × arrival_radius (robot was close).
        # This prevents triggering at spawn if the robot happens to start near
        # an early waypoint.
        armed = self._wp_closest_dist[self._wp_idx] < (1.5 * wp.arrival_radius)

        if armed and dist < wp.arrival_radius:
            print(f"[Nav] Waypoint {self._wp_idx} '{wp.label}' reached "
                  f"(dist={dist:.1f} m < radius={wp.arrival_radius:.1f} m)")
            self._wp_idx += 1
            if self._wp_idx < len(self._waypoints):
                nwp = self._waypoints[self._wp_idx]
                self._global.set_goal(nwp.x, nwp.y)
                self._recovery.reset()   # fresh slate for each waypoint
                # Reset closest-dist tracking for newly active waypoint
                # (already initialized to inf in __init__, no action needed)

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

    def _process_sensors_only(self) -> None:
        """
        Run all sensor processing (LiDAR→SLAM, omap updates) without navigating.

        Called every step during the BOOT phase so the map is fully built by the
        time the robot starts moving.  Does NOT call the local planner, global
        planner, or inject any drive command.
        """
        # LiDAR → SLAM queue + omap update
        if self._lidar is not None:
            try:
                lidar_hits = self._lidar.data.ray_hits_w
                raw_pts = lidar_hits[self._env_idx].cpu().numpy()
                valid_mask = (
                    np.isfinite(raw_pts[:, 0])
                    & np.isfinite(raw_pts[:, 1])
                    & np.isfinite(raw_pts[:, 2])
                    & (raw_pts[:, 2] < 1e5)
                )
                clean_pts = raw_pts[valid_mask]
                if len(clean_pts) > 10:
                    sim_p = self._sim_localizer.get_pose()
                    quat = self._robot.data.root_quat_w[self._env_idx]
                    w, qx, qy, qz = (float(quat[0]), float(quat[1]),
                                     float(quat[2]), float(quat[3]))
                    roll_s  = math.atan2(2*(w*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
                    sinp = max(-1.0, min(1.0, 2*(w*qy - qz*qx)))
                    pitch_s = math.asin(sinp)
                    slam_input = SLAMPose(
                        x=sim_p.x, y=sim_p.y, z=sim_p.z,
                        roll=roll_s, pitch=pitch_s, yaw=sim_p.yaw,
                    )
                    try:
                        self._slam_queue.put_nowait((clean_pts.copy(), slam_input))
                    except Exception:
                        pass
                    self._omap.update_lidar(clean_pts)
            except Exception:
                pass

        # Height scanner → omap
        if self._scanner is not None:
            try:
                hits_w = self._scanner.data.ray_hits_w
                pts = hits_w[self._env_idx].cpu().numpy()
                valid = np.isfinite(pts[:, 2]) & (pts[:, 2] < 1e5)
                if valid.any():
                    self._omap.update(pts[valid])
            except Exception:
                pass

        # Forward scanner → omap
        if self._fwd_scanner is not None:
            try:
                fwd_hits_w = self._fwd_scanner.data.ray_hits_w
                fwd_pts = fwd_hits_w[self._env_idx].cpu().numpy()
                valid_fwd = np.isfinite(fwd_pts[:, 2]) & (fwd_pts[:, 2] < 1e5)
                if valid_fwd.any():
                    self._omap.update(fwd_pts[valid_fwd])
            except Exception:
                pass

        # Update dashboard cost grid every 10 steps during BOOT
        step_n = self.shared.get("step_count", 0)
        if step_n % 10 == 0:
            with self._lock:
                self.shared["cost_grid"] = self._omap.get_cost_grid()

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
