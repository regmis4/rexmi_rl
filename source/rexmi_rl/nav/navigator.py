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
from rexmi_rl.nav.reorient import ReorientController, ReorientPhase



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
        # Language A HOLD→YAW→SETTLE pulse scheduler for brake + turn on slope.
        # Continuous ω is OOD for Pulse turn policies — this owns reorient cmds.
        self._reorient  = ReorientController()

        # Soft-reset (nav-owned tip recovery — terminations are off for demo)
        self._last_good_pose: Pose | None = None
        self._soft_reset_count = 0
        self._soft_reset_cooldown_until = 0.0
        self._post_fail_reverse_until = 0.0
        self._post_soft_hold_until = 0.0   # zero cmd after soft-reset plant
        self._brake_until = 0.0            # decelerate before reorient
        self._pending_reorient = False     # start reorient after brake
        self._pending_reorient_sign = None
        self._pending_reorient_slope = 0.0
        # Primary tip signal = body +Z · world +Z (1=upright, -1=on back).
        # Euler roll/pitch alone miss some back-flat attitudes.
        self._tip_up_soft = 0.25    # ~75° from upright → start tip timer
        self._tip_up_hard = -0.20   # clearly inverted / on back → immediate
        self._tip_limit_rad = math.radians(95.0)
        self._tip_hard_rad = math.radians(120.0)
        self._tip_since: float | None = None
        self._tip_sustain_s = 0.45
        self._stuck_flat_since: float | None = None
        self._stuck_flat_s = 1.5    # motionless + not upright → force reset
        self._prev_dist_wp = math.inf
        self._last_body_up_z = 1.0

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

        # Per-waypoint tracking — prevents immediate skip at spawn.
        # A waypoint is only "reached" after the robot has made real progress
        # toward it (closed the gap by ≥ progress_arm_m from first sighting)
        # OR was never already inside the arrival disk at first sighting.
        self._wp_closest_dist: list[float] = [math.inf] * len(self._waypoints)
        self._wp_first_dist: list[float] = [math.inf] * len(self._waypoints)
        self._wp_progress_arm_m: float = 1.0  # must close gap by this much to arm


        # Set first waypoint in global planner + log mission path
        if self._waypoints:
            wp = self._waypoints[0]
            self._global.set_goal(wp.x, wp.y)
            print(f"[Navigator] Mission '{mission.value}' — {len(self._waypoints)} waypoints:")
            for i, w in enumerate(self._waypoints):
                print(f"  WP[{i}] {w.label:16s} ({w.x:+6.1f}, {w.y:+6.1f}) r={w.arrival_radius:.1f}m")


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
            "reorient_status":  "IDLE",
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

        # Orientation from robot quat
        try:
            quat = self._robot.data.root_quat_w[self._env_idx]
            w, qx, qy, qz = (float(quat[0]), float(quat[1]),
                             float(quat[2]), float(quat[3]))
            roll = math.atan2(2 * (w * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
            sinp = max(-1.0, min(1.0, 2 * (w * qy - qz * qx)))
            pitch = math.asin(sinp)
            # Body +Z axis in world (1 = upright, -1 = flat on back)
            body_up_z = 1.0 - 2.0 * (qx * qx + qy * qy)
        except Exception:
            roll, pitch, body_up_z = 0.0, 0.0, 1.0
        self._last_body_up_z = body_up_z

        # Soft-reset detection (priority: body_up_z, then euler, then stuck-flat)
        now_m = time.monotonic()
        inverted = body_up_z < self._tip_up_soft
        hard_inverted = body_up_z < self._tip_up_hard
        euler_soft = abs(roll) > self._tip_limit_rad or abs(pitch) > self._tip_limit_rad
        euler_hard = abs(roll) > self._tip_hard_rad or abs(pitch) > self._tip_hard_rad
        tipped_soft = inverted or euler_soft
        tipped_hard = hard_inverted or euler_hard
        reorient_busy = self._reorient.active

        # Sustained tip timer (also counts during reorient if inverted)
        if tipped_soft:
            if self._tip_since is None:
                self._tip_since = now_m
            tip_held = (now_m - self._tip_since) >= self._tip_sustain_s
        else:
            self._tip_since = None
            tip_held = False

        # Stuck lying down: nearly zero speed + not upright for a while
        # Catches "on back but euler looks mild" and frozen physics poses.
        not_upright = body_up_z < 0.50  # >~60° from upright
        if not_upright and speed < 0.08:
            if self._stuck_flat_since is None:
                self._stuck_flat_since = now_m
            stuck_flat = (now_m - self._stuck_flat_since) >= self._stuck_flat_s
        else:
            self._stuck_flat_since = None
            stuck_flat = False

        do_soft = False
        reason = ""
        if now_m >= self._soft_reset_cooldown_until:
            if tipped_hard:
                do_soft = True
                reason = f"hard invert up_z={body_up_z:+.2f}"
            elif tip_held and (not reorient_busy or hard_inverted or body_up_z < 0.0):
                # During reorient only if actually inverted (not mild roll while turning)
                do_soft = True
                reason = f"sustained tip up_z={body_up_z:+.2f}"
            elif stuck_flat:
                do_soft = True
                reason = f"stuck-flat up_z={body_up_z:+.2f} v={speed:.2f}"

        if do_soft:
            if reorient_busy:
                print(f"[Nav] tip during reorient — soft-reset ({reason})")
            else:
                print(f"[Nav] tip — soft-reset ({reason})")
            self._soft_reset_upright(pose, roll, pitch)
            pose = self._localizer.get_pose()
            speed = 0.0

        # Track last good upright pose for soft-reset target
        if body_up_z > 0.70 and abs(roll) < math.radians(40) and abs(pitch) < math.radians(40) and speed < 1.5:
            self._last_good_pose = Pose(pose.x, pose.y, pose.z, pose.yaw)

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
                self._reorient.cancel()
                # Reset progress tracking in recovery so the new position gets
                # a fresh 60-s window to make progress toward the waypoint.
                self._recovery._last_progress_time = None
                self._recovery._best_dist          = math.inf
                # Do NOT advance waypoints on reset; re-arm progress gate
                if self._wp_idx < len(self._wp_first_dist):
                    self._wp_first_dist[self._wp_idx] = math.inf
                    self._wp_closest_dist[self._wp_idx] = math.inf
                if hasattr(self, "policy_selector") and self.policy_selector is not None:
                    if hasattr(self.policy_selector, "release_turn"):
                        self.policy_selector.release_turn()


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

        # 6. Terrain metrics for policy selector
        slope_for_sel = (
            0.0 if (local_out.slope_ahead != local_out.slope_ahead)
            else float(local_out.slope_ahead)
        )
        trav_count = sum(1 for t in local_out.traversable_mask if t)
        trav_frac  = trav_count / max(1, len(local_out.traversable_mask))
        scan = [[scan_heights[i * 10 + j] for j in range(10)] for i in range(16)]
        max_step = max(
            max(abs(scan[i + 1][j] - scan[i][j]) for i in range(15))
            for j in range(10)
        )

        # 7. Recovery FSM — pass current distance to waypoint for progress tracking
        #    While reorient is active, treat as "turning" so stuck timers stay quiet.
        dist_to_wp = math.hypot(pose.x - current_wp.x, pose.y - current_wp.y)
        _he_for_recovery = local_out.heading_error
        if self._reorient.active:
            # Keep |he| large enough that RecoveryFSM turning exemption holds
            _he_for_recovery = math.copysign(
                max(abs(local_out.heading_error), math.radians(70)),
                local_out.heading_error if abs(local_out.heading_error) > 1e-6 else 1.0,
            )
        rec_vx, rec_vy, rec_omega = self._recovery.update(
            speed, _he_for_recovery, waypoint_dist=dist_to_wp
        )

        if self._recovery.is_blocked:
            # ----------------------------------------------------------------
            # BLOCKED handler — inflate obstacle radius + optionally skip WP
            # ----------------------------------------------------------------
            _fwd_x = pose.x + 0.5 * math.cos(pose.yaw)
            _fwd_y = pose.y + 0.5 * math.sin(pose.yaw)

            self._omap.inflate_blocked(_fwd_x, _fwd_y, radius_m=1.0)
            self._omap.inflate_blocked(pose.x, pose.y, radius_m=0.6)

            _block_key = self._wp_idx
            if not hasattr(self, "_block_count"):
                self._block_count: dict = {}
            self._block_count[_block_key] = self._block_count.get(_block_key, 0) + 1

            if self._block_count[_block_key] >= 3:
                self._block_count[_block_key] = 0
                print(f"[Nav] BLOCKED at ({pose.x:.1f},{pose.y:.1f}) × 3 — "
                      f"skipping WP[{self._wp_idx}] → WP[{self._wp_idx+1}]")
                self._wp_idx += 1
                if self._wp_idx < len(self._waypoints):
                    nwp = self._waypoints[self._wp_idx]
                    self._global.set_goal(nwp.x, nwp.y)
            else:
                self._global.set_goal(current_wp.x, current_wp.y)
                print(f"[Nav] BLOCKED at ({pose.x:.1f},{pose.y:.1f}) "
                      f"(×{self._block_count[_block_key]}) — "
                      f"inflated 1.0m radius, forcing replan")

            self._recovery.reset()
            self._reorient.cancel()

        # ------------------------------------------------------------------
        # 7b. ReorientController — Language A HOLD→YAW→SETTLE
        # ------------------------------------------------------------------
        # Start when:
        #   • local planner wants a committed large turn / obstacle stop-turn
        #   • RecoveryFSM enters ROTATING (after reverse)
        #   • |heading_error| exceeds enter threshold (backup path)
        # While active: command = pulse schedule; force TURN policy.
        # ------------------------------------------------------------------
        # Skip new reorient while post-fail reverse / post soft-reset hold / braking
        now_m2 = time.monotonic()
        in_post_fail_reverse = now_m2 < self._post_fail_reverse_until
        in_post_soft_hold = now_m2 < self._post_soft_hold_until
        in_brake = now_m2 < self._brake_until

        # Drive-and-correct: if closing on WP, do not open a full reorient
        making_progress = (
            dist_to_wp < self._prev_dist_wp - 0.08
            and speed > 0.10
            and abs(local_out.heading_error) < math.radians(55)
        )
        self._prev_dist_wp = dist_to_wp

        # Finish pending reorient after brake window (must be nearly stopped)
        if (
            self._pending_reorient
            and not self._reorient.active
            and not in_brake
            and speed < 0.10
            and now_m2 >= getattr(self._reorient, "_cooldown_until", 0.0)
        ):
            self._pending_reorient = False
            self._reorient.start(
                heading_error=local_out.heading_error,
                slope_ahead=self._pending_reorient_slope,
                yaw_sign=self._pending_reorient_sign,
                body_yaw=pose.yaw,
            )
            print(f"[Nav] brake done (v={speed:.2f}) — starting slow reorient")

        if (
            not self._reorient.active
            and not self._pending_reorient
            and not self._recovery.is_blocked
            and self._recovery.state != RecoveryState.REVERSING
            and not in_post_fail_reverse
            and not in_post_soft_hold
            and not in_brake
            and not making_progress
        ):
            start_reorient = False
            yaw_sign = None
            cooled = now_m2 >= getattr(self._reorient, "_cooldown_until", 0.0)
            he_abs = abs(local_out.heading_error)
            he_need = he_abs >= self._reorient.enter_rad
            if cooled and he_need and getattr(local_out, "want_reorient", False):
                start_reorient = True
                if abs(getattr(local_out, "reorient_yaw_sign", 0.0)) > 1e-6:
                    yaw_sign = local_out.reorient_yaw_sign
            elif cooled and he_need and self._recovery.wants_reorient:
                start_reorient = True
                yaw_sign = self._recovery.rotate_dir
            elif cooled and he_need and self._reorient.should_start(local_out.heading_error):
                start_reorient = True

            if start_reorient:
                # CRITICAL: never pivot while still sliding — brake first
                if speed > 0.12:
                    self._pending_reorient = True
                    self._pending_reorient_sign = yaw_sign
                    self._pending_reorient_slope = slope_for_sel
                    self._brake_until = now_m2 + 1.5
                    print(
                        f"[Nav] BRAKE before turn "
                        f"(v={speed:.2f} he={math.degrees(local_out.heading_error):+.0f}°)"
                    )
                else:
                    self._reorient.start(
                        heading_error=local_out.heading_error,
                        slope_ahead=slope_for_sel,
                        yaw_sign=yaw_sign,
                        body_yaw=pose.yaw,
                    )

        reorient_out = self._reorient.update(
            local_out.heading_error,
            slope_ahead=slope_for_sel,
            body_up_z=self._last_body_up_z,
            body_yaw=pose.yaw,
        )
        if reorient_out.done:
            if self._recovery.is_recovering:
                self._recovery.reset()
            self._local._turn_committed = False
            self._local._committed_omega_sign = 0.0
            # Extra stillness after micro-turn success (policy already settled 2s)
            self._post_soft_hold_until = max(
                self._post_soft_hold_until, time.monotonic() + 0.8
            )
        if reorient_out.failed:
            # Reverse away from stuck cell; cooldown blocks reorient restart
            print("[Nav] Reorient failed — reverse 2s + inflate + replan (cooldown active)")
            self._omap.inflate_blocked(pose.x, pose.y, radius_m=0.8)
            self._global.set_goal(current_wp.x, current_wp.y)
            self._recovery.reset()
            self._local._turn_committed = False
            self._local._committed_omega_sign = 0.0
            self._post_fail_reverse_until = time.monotonic() + 2.0
            if self.policy_selector is not None and hasattr(self.policy_selector, "release_turn"):
                self.policy_selector.release_turn()


        # 8. Policy selector
        policy_status = "fixed"
        if self.policy_selector is not None:
            # Use turn policy for reorient AND while braking into a turn
            if (
                self._reorient.active
                or self._pending_reorient
                or (time.monotonic() < self._brake_until)
                or self._recovery.is_rotating
            ):
                # Turn policy for brake + pivot (leg plant)
                self.policy_selector.force_turn(slope_ahead=slope_for_sel)
            else:
                he_for_sel = local_out.heading_error
                if abs(he_for_sel) > math.radians(50):
                    he_for_sel = math.copysign(math.radians(50), he_for_sel)
                self.policy_selector.update(
                    slope_ahead=slope_for_sel,
                    heading_error_rad=he_for_sel,
                    max_step=max_step,
                    traversable_fraction=trav_frac,
                )
            policy_status = self.policy_selector.status_str()
            self._local.vx_normal = self.policy_selector.current_vx()

        # 9. Command arbitration
        # Priority: post-soft hold > brake > reorient > post-fail reverse > recovery > local
        if time.monotonic() < self._post_soft_hold_until:
            cmd = (0.0, 0.0, 0.0)  # plant after soft-reset
        elif time.monotonic() < self._brake_until or self._pending_reorient:
            # In-distribution plant for 13345 (NOT vx=0,ω=0 — that flings legs)
            cmd = (0.05, 0.0, 0.0)
        elif self._reorient.active or reorient_out.done:
            cmd = (reorient_out.vx, reorient_out.vy, reorient_out.omega)
        elif time.monotonic() < self._post_fail_reverse_until:
            cmd = (-0.20, 0.0, 0.0)  # light reverse after failed reorient
        elif self._recovery.state == RecoveryState.REVERSING:
            cmd = (rec_vx, rec_vy, rec_omega)
        elif self._recovery.is_recovering and not self._recovery.wants_reorient:
            cmd = (rec_vx, rec_vy, rec_omega)
        else:
            cmd = (local_out.vx, local_out.vy, local_out.omega)


        # 10. Inject command into env
        self._inject_command(*cmd)

        # 10b. Per-50-step diagnostic log
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
            _reo     = self._reorient.status_str()
            try:
                _cmd_tensor = self._env.unwrapped.command_manager.get_command("base_velocity")
                _omega_obs = float(_cmd_tensor[self._env_idx, 2])
            except Exception:
                _omega_obs = float("nan")
            print(
                f"[Nav][{step_n_now:5d}] "
                f"he={_he_deg:+.1f}° "
                f"ω_cmd={_omega:+.3f} ω_obs={_omega_obs:+.3f} "
                f"vx={_vx:+.2f} "
                f"cell_cost={_cell_cost:.1f} "
                f"dist_wp={dist_to_wp:.1f}m "
                f"imm_wp=({imm_wp[0]:+.1f},{imm_wp[1]:+.1f}) "
                f"state={_state} reorient={_reo}"
                f"{_fwd_str}"
            )
            # Extra turn diagnostics while reorient active
            if self._reorient.active:
                try:
                    _ang = self._robot.data.root_ang_vel_b[self._env_idx]
                    _wz = float(_ang[2])
                except Exception:
                    _wz = float("nan")
                print(
                    "[Turn] "
                    + self._reorient.diag_str(
                        local_out.heading_error,
                        pose.yaw,
                        self._last_body_up_z,
                        speed,
                        omega_body=_wz,
                    )
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
                "reorient_status": self._reorient.status_str(),
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

        Progress gate: if the robot spawns already inside arrival_radius, do
        NOT count that as arrival.  Require either:
          • first sighting was outside the disk, then enter it, OR
          • robot closed the gap by ≥ progress_arm_m from first sighting
            (genuine approach) and is now inside arrival_radius.
        """
        if self._wp_idx >= len(self._waypoints):
            return
        wp = self._waypoints[self._wp_idx]
        dist = math.hypot(pose.x - wp.x, pose.y - wp.y)

        # Record first-sighting distance
        if self._wp_first_dist[self._wp_idx] == math.inf:
            self._wp_first_dist[self._wp_idx] = dist


        if dist < self._wp_closest_dist[self._wp_idx]:
            self._wp_closest_dist[self._wp_idx] = dist

        first = self._wp_first_dist[self._wp_idx]
        closed = first - self._wp_closest_dist[self._wp_idx]
        # Armed if we started outside and entered, or made real progress inward
        started_outside = first >= wp.arrival_radius
        made_progress = closed >= self._wp_progress_arm_m
        armed = (started_outside and dist < wp.arrival_radius) or (
            made_progress and dist < wp.arrival_radius
        )

        if armed:
            print(f"[Nav] Waypoint {self._wp_idx} '{wp.label}' reached "
                  f"(dist={dist:.1f} m < radius={wp.arrival_radius:.1f} m, "
                  f"closed={closed:.1f} m)")
            self._wp_idx += 1
            self._recovery.reset()
            self._reorient.reset()
            self._local._turn_committed = False
            self._local._committed_omega_sign = 0.0
            if self._wp_idx < len(self._waypoints):
                nwp = self._waypoints[self._wp_idx]
                self._global.set_goal(nwp.x, nwp.y)




    def _soft_reset_upright(self, pose: Pose, roll: float, pitch: float) -> None:
        """
        Nav-owned tip recovery when Isaac terminations are disabled.

        Uses Isaac Lab Articulation APIs correctly:
          write_root_pose_to_sim(pose (1,7), env_ids (1,))
          write_root_velocity_to_sim(vel (1,6), env_ids)
          write_joint_state_to_sim(q, qd, env_ids)  # default stand
        """
        import torch

        self._soft_reset_count += 1
        self._soft_reset_cooldown_until = time.monotonic() + 4.0

        # Recovery pose: current xy, lift z, yaw toward mission WP (upright only)
        tx, ty = float(pose.x), float(pose.y)
        tz = float(pose.z) + 0.40  # clear terrain while untangling
        if self._last_good_pose is not None:
            # Prefer last good xy if we haven't drifted far
            if math.hypot(self._last_good_pose.x - tx, self._last_good_pose.y - ty) < 2.5:
                tx, ty = self._last_good_pose.x, self._last_good_pose.y
                tz = max(tz, self._last_good_pose.z + 0.35)

        tyaw = float(pose.yaw)
        if self._wp_idx < len(self._waypoints):
            wp = self._waypoints[self._wp_idx]
            tyaw = math.atan2(wp.y - ty, wp.x - tx)
        elif self._last_good_pose is not None:
            tyaw = self._last_good_pose.yaw

        half = 0.5 * tyaw
        qw, qx, qy, qz = math.cos(half), 0.0, 0.0, math.sin(half)

        print(
            f"[Nav][SOFT_RESET] #{self._soft_reset_count} "
            f"tip roll={math.degrees(roll):+.0f}° pitch={math.degrees(pitch):+.0f}° "
            f"up_z={getattr(self, '_last_body_up_z', float('nan')):+.2f} → "
            f"({tx:+.1f},{ty:+.1f},{tz:+.1f}) yaw={math.degrees(tyaw):+.0f}°"
        )

        robot = self._robot
        idx = int(self._env_idx)
        wrote_ok = False
        try:
            device = robot.data.root_pos_w.device
            dtype = robot.data.root_pos_w.dtype
            env_ids = torch.tensor([idx], device=device, dtype=torch.long)

            # (1, 7) pose: pos + quat wxyz
            root_pose = torch.tensor(
                [[tx, ty, tz, qw, qx, qy, qz]], device=device, dtype=dtype
            )
            # (1, 6) zero twist
            root_vel = torch.zeros((1, 6), device=device, dtype=dtype)

            # Prefer explicit pose + velocity with env_ids (Isaac Lab contract)
            if hasattr(robot, "write_root_pose_to_sim"):
                robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
            elif hasattr(robot, "write_root_link_pose_to_sim"):
                robot.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)
            else:
                raise RuntimeError("no write_root_pose_to_sim on robot")

            if hasattr(robot, "write_root_velocity_to_sim"):
                robot.write_root_velocity_to_sim(root_vel, env_ids=env_ids)
            elif hasattr(robot, "write_root_com_velocity_to_sim"):
                robot.write_root_com_velocity_to_sim(root_vel, env_ids=env_ids)

            # Joints → default stand (critical — legs stay collapsed otherwise)
            if hasattr(robot, "write_joint_state_to_sim") and hasattr(robot.data, "default_joint_pos"):
                q = robot.data.default_joint_pos[idx].unsqueeze(0).clone()
                qd = torch.zeros_like(q)
                # joint_ids=None → all joints
                try:
                    robot.write_joint_state_to_sim(q, qd, env_ids=env_ids)
                except TypeError:
                    # older signature: (pos, vel, joint_ids, env_ids)
                    robot.write_joint_state_to_sim(q, qd, None, env_ids)

            wrote_ok = True
        except Exception as e:
            print(f"[Nav][SOFT_RESET] write failed: {e}")
            # Fallback: full-state (1, 13) + env_ids
            try:
                device = robot.data.root_state_w.device
                dtype = robot.data.root_state_w.dtype
                env_ids = torch.tensor([idx], device=device, dtype=torch.long)
                state = torch.zeros((1, 13), device=device, dtype=dtype)
                state[0, 0] = tx
                state[0, 1] = ty
                state[0, 2] = tz
                state[0, 3] = qw
                state[0, 4] = qx
                state[0, 5] = qy
                state[0, 6] = qz
                robot.write_root_state_to_sim(state, env_ids=env_ids)
                wrote_ok = True
                print("[Nav][SOFT_RESET] fallback write_root_state_to_sim(1,13) OK")
            except Exception as e2:
                print(f"[Nav][SOFT_RESET] fallback also failed: {e2}")

        # Verify orientation from buffers (PhysX applies on next step; buffers should update)
        try:
            quat = robot.data.root_quat_w[idx]
            w, qx2, qy2, qz2 = (float(quat[0]), float(quat[1]),
                                float(quat[2]), float(quat[3]))
            r2 = math.atan2(2 * (w * qx2 + qy2 * qz2), 1 - 2 * (qx2 * qx2 + qy2 * qy2))
            sp = max(-1.0, min(1.0, 2 * (w * qy2 - qz2 * qx2)))
            p2 = math.asin(sp)
            if abs(r2) < math.radians(45) and abs(p2) < math.radians(45):
                print(
                    f"[Nav][SOFT_RESET] verify OK "
                    f"roll={math.degrees(r2):+.1f}° pitch={math.degrees(p2):+.1f}°"
                )
            else:
                print(
                    f"[Nav][SOFT_RESET] verify STILL TIPPED "
                    f"roll={math.degrees(r2):+.1f}° pitch={math.degrees(p2):+.1f}° "
                    f"(wrote_ok={wrote_ok}) — will retry next tip window"
                )
                # Allow quicker retry if write didn't stick
                self._soft_reset_cooldown_until = time.monotonic() + 0.5
        except Exception as e:
            print(f"[Nav][SOFT_RESET] verify read failed: {e}")

        # Clear nav state — long plant so policy doesn't immediately thrash
        self._reorient.cancel()
        if hasattr(self._reorient, "_cooldown_until"):
            self._reorient._cooldown_until = time.monotonic() + 8.0
        self._recovery.reset()
        self._local._turn_committed = False
        self._local._committed_omega_sign = 0.0
        self._post_fail_reverse_until = 0.0
        self._post_soft_hold_until = time.monotonic() + 3.0  # 3 s plant
        self._brake_until = 0.0
        self._pending_reorient = False
        self._tip_since = None
        self._stuck_flat_since = None
        self._last_good_pose = Pose(tx, ty, tz - 0.15, tyaw)
        if self.policy_selector is not None:
            if hasattr(self.policy_selector, "release_turn"):
                self.policy_selector.release_turn()
            # Prefer rocky after recovery
            try:
                from rexmi_rl.nav.policy_selector import PolicyMode
                if hasattr(self.policy_selector, "_current_mode"):
                    self.policy_selector._current_mode = PolicyMode.ROCKY_SLOPE
                    self.policy_selector._candidate = PolicyMode.ROCKY_SLOPE
                    self.policy_selector._in_turn_override = False
            except Exception:
                pass
        if self._wp_idx < len(self._waypoints):
            wp = self._waypoints[self._wp_idx]
            self._global.set_goal(wp.x, wp.y)
            if self._wp_idx < len(self._wp_first_dist):
                self._wp_first_dist[self._wp_idx] = math.inf
                self._wp_closest_dist[self._wp_idx] = math.inf

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
