# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Live navigation dashboard — 3D point cloud + 2D cost map + status bar.

Runs in a daemon thread at ~2 Hz so it never blocks the sim loop.

Layout
------
  Left panel  : 3D scatter of accumulated terrain point cloud (viridis by height)
                + forward scanner hits (orange — most recent ~1.3 s)
                + robot position (white sphere) + planned path (white line)
                + waypoints (gold stars) + heading arrow (cyan)
  Right panel : 2D top-down cost map (green=clear, red=blocked)
                + robot position + path + crater boundary circle
  Bottom bar  : Mission / waypoint / speed / slope / state text

Fixes (2026-07-09)
------------------
  • Forward scanner hits shown in orange — separate from downward viridis cloud
  • Downward cloud subsampled to 8k at render time (ring buffer holds 100k so
    the cloud stays fresh; old points replaced automatically as robot moves)
  • 3D rotation works: view is saved BEFORE ax.cla(), and the inter-frame sleep
    is replaced by a flush_events loop so Tk mouse events are processed continuously
"""

from __future__ import annotations

import math
import threading
import time
from typing import Optional

import numpy as np


class Dashboard:
    """
    Matplotlib live dashboard for the navigation layer.

    Parameters
    ----------
    shared : dict
        Shared state dict written by Navigator.step().
    lock : threading.Lock
        Lock protecting the shared dict.
    omap : OccupancyMap
        Reference to the occupancy map (for world_size, origin, crater geometry).
    waypoints : list[Waypoint]
        Mission waypoint list (for labels and star markers).
    update_interval_s : float
        Seconds between full redraws. Default 0.5 s.
        Between redraws, flush_events() is called every 50 ms to keep
        the Tk window responsive to mouse rotation.
    """

    # Max points to render in the 3D scatter (downward cloud)
    RENDER_MAX_DOWN = 8_000
    # Max points to render for forward cloud (render all — only 5k max)
    RENDER_MAX_FWD  = 5_000

    def __init__(self, shared, lock, omap, waypoints,
                 update_interval_s: float = 0.5):
        self._shared   = shared
        self._lock     = lock
        self._omap     = omap
        self._waypoints = waypoints
        self._interval = update_interval_s

        # These are set up lazily in run() so matplotlib imports happen in the
        # dashboard thread (avoids GUI backend conflicts with Isaac Sim)
        self._fig  = None
        self._ax3d = None
        self._ax2d = None
        self._ax_status = None

        # Preserved view state — survives redraws so user interactions persist
        self._view3d_elev: float | None = None   # 3D elevation angle
        self._view3d_azim: float | None = None   # 3D azimuth angle
        self._view2d_xlim: tuple | None = None   # 2D pan/zoom x range
        self._view2d_ylim: tuple | None = None   # 2D pan/zoom y range

    # ------------------------------------------------------------------
    # Main loop (runs in daemon thread)
    # ------------------------------------------------------------------

    def run(self, stop_event: threading.Event) -> None:
        """Dashboard main loop. Called by Navigator.start_dashboard()."""
        import matplotlib
        matplotlib.use("TkAgg")   # use Tk backend (works headless + GUI)
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        plt.ion()
        self._fig = plt.figure(figsize=(16, 8), facecolor="#0a0a1a")
        self._fig.canvas.manager.set_window_title("REXMI Nav Dashboard")

        # Layout: left=3D, right=2D, bottom status
        gs = self._fig.add_gridspec(
            2, 2,
            height_ratios=[10, 1],
            hspace=0.05, wspace=0.15,
        )
        self._ax3d     = self._fig.add_subplot(gs[0, 0], projection="3d")
        self._ax2d     = self._fig.add_subplot(gs[0, 1])
        self._ax_status = self._fig.add_subplot(gs[1, :])

        self._style_axes()

        # Flush interval: process Tk events every 50 ms between full redraws
        # so mouse rotation/pan stays responsive during the 0.5 s sleep.
        _flush_interval = 0.05
        _next_redraw    = time.monotonic()

        while not stop_event.is_set():
            now = time.monotonic()
            if now >= _next_redraw:
                try:
                    self._redraw(plt)
                except Exception as e:
                    # Never crash the dashboard thread
                    print(f"[Dashboard] render error: {e}")
                _next_redraw = now + self._interval

            # Process Tk mouse/keyboard events without blocking
            try:
                self._fig.canvas.flush_events()
            except Exception:
                pass

            time.sleep(_flush_interval)

        plt.close(self._fig)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _redraw(self, plt) -> None:
        """Redraw all panels with current shared state."""
        with self._lock:
            pose           = self._shared["pose"]
            wp_idx         = self._shared["wp_idx"]
            path           = list(self._shared["planned_path"])
            local_out      = self._shared["local_out"]
            rec_status     = self._shared["recovery_status"]
            mission        = self._shared["mission"]
            cmd            = self._shared["cmd"]
            xs, ys, zs     = self._shared["cloud_xyz"]
            fxs, fys, fzs  = self._shared["fwd_cloud_xyz"]
            cost_grid      = self._shared["cost_grid"]
            speed          = self._shared["speed"]

        # ---- 3D panel ------------------------------------------------
        ax3 = self._ax3d

        # Save current view angle BEFORE clearing so user rotation persists.
        # On first draw, elev/azim may not exist yet — use defaults.
        try:
            saved_elev = ax3.elev
            saved_azim = ax3.azim
        except AttributeError:
            saved_elev = self._view3d_elev if self._view3d_elev is not None else 25.0
            saved_azim = self._view3d_azim if self._view3d_azim is not None else -60.0

        if self._view3d_elev is None:
            # First draw — use defaults
            self._view3d_elev = 25.0
            self._view3d_azim = -60.0
        else:
            # Preserve whatever the user has rotated to
            self._view3d_elev = saved_elev
            self._view3d_azim = saved_azim

        ax3.cla()
        ax3.set_facecolor("#0a0a1a")
        ax3.view_init(elev=self._view3d_elev, azim=self._view3d_azim)

        # Downward scan — viridis coloured by height, subsampled to RENDER_MAX_DOWN
        if len(xs) > 0:
            n = len(xs)
            if n > self.RENDER_MAX_DOWN:
                idx = np.random.choice(n, self.RENDER_MAX_DOWN, replace=False)
                xs_d, ys_d, zs_d = xs[idx], ys[idx], zs[idx]
            else:
                xs_d, ys_d, zs_d = xs, ys, zs

            ax3.scatter(xs_d, ys_d, zs_d,
                        c=zs_d, cmap="viridis",
                        s=1.5, alpha=0.65, linewidths=0,
                        label="height scan")

        # Forward scanner hits — orange, rendered on top of downward cloud.
        # Shows the current forward LiDAR fan (~1.3 s rolling window).
        if len(fxs) > 0:
            nf = len(fxs)
            if nf > self.RENDER_MAX_FWD:
                idx_f = np.random.choice(nf, self.RENDER_MAX_FWD, replace=False)
                fxs_d, fys_d, fzs_d = fxs[idx_f], fys[idx_f], fzs[idx_f]
            else:
                fxs_d, fys_d, fzs_d = fxs, fys, fzs

            ax3.scatter(fxs_d, fys_d, fzs_d,
                        c="#FF8800",       # orange — forward scanner
                        s=6, alpha=0.85, linewidths=0,
                        label="fwd scanner")

        # Planned path (at robot's current z height)
        if len(path) >= 2:
            px = [p[0] for p in path]
            py = [p[1] for p in path]
            ax3.plot(px, py,
                     [float(pose.z)] * len(px),
                     color="white", linewidth=1.5, alpha=0.8, zorder=10)

        # Waypoints
        for i, wp in enumerate(self._waypoints):
            color = "#FFD700" if i == wp_idx else "#888888"
            size  = 80 if i == wp_idx else 40
            ax3.scatter([wp.x], [wp.y], [float(pose.z) + 0.5],
                        color=color, marker="*", s=size, zorder=20)

        # Robot position
        ax3.scatter([pose.x], [pose.y], [pose.z + 0.3],
                    color="white", marker="o", s=120, zorder=30,
                    edgecolors="#00FFFF", linewidths=2)

        # Robot heading arrow
        arrow_len = 1.5
        ax3.quiver(pose.x, pose.y, pose.z + 0.3,
                   arrow_len * math.cos(pose.yaw),
                   arrow_len * math.sin(pose.yaw),
                   0.0, color="#00FFFF", linewidth=2)

        # Equal-aspect 3D scaling: compute bounding box from all rendered pts
        # so the crater's real height variation (e.g. 0–5 m over 30 m width)
        # is not squished into a flat pancake by Matplotlib's auto-scaling.
        _all_x = []
        _all_y = []
        _all_z = []
        if len(xs) > 0:
            _all_x.extend([float(xs.min()), float(xs.max())])
            _all_y.extend([float(ys.min()), float(ys.max())])
            _all_z.extend([float(zs.min()), float(zs.max())])
        if len(fxs) > 0:
            _all_x.extend([float(fxs.min()), float(fxs.max())])
            _all_y.extend([float(fys.min()), float(fys.max())])
            _all_z.extend([float(fzs.min()), float(fzs.max())])
        # Always include the robot position so the plot doesn't start empty
        _all_x.extend([pose.x - 1.0, pose.x + 1.0])
        _all_y.extend([pose.y - 1.0, pose.y + 1.0])
        _all_z.extend([pose.z - 0.5, pose.z + 0.5])

        x_min, x_max = min(_all_x), max(_all_x)
        y_min, y_max = min(_all_y), max(_all_y)
        z_min, z_max = min(_all_z), max(_all_z)

        # Use a single range for all axes so z height is true to scale
        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min, 2.0)
        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2
        z_mid = (z_min + z_max) / 2
        ax3.set_xlim(x_mid - max_range / 2, x_mid + max_range / 2)
        ax3.set_ylim(y_mid - max_range / 2, y_mid + max_range / 2)
        ax3.set_zlim(z_mid - max_range / 2, z_mid + max_range / 2)

        ax3.set_xlabel("X (m)", color="gray", fontsize=7)
        ax3.set_ylabel("Y (m)", color="gray", fontsize=7)
        ax3.set_zlabel("Z (m)", color="gray", fontsize=7)
        ax3.tick_params(colors="gray", labelsize=6)
        ax3.set_title(
            "Terrain (viridis = downward scan · orange = forward scanner)  "
            "[drag to rotate]",
            color="white", fontsize=8, pad=4,
        )

        # ---- 2D cost map panel ---------------------------------------
        ax2 = self._ax2d
        # Save user pan/zoom before clearing so interactive view persists
        if self._view2d_xlim is not None:
            self._view2d_xlim = ax2.get_xlim()
            self._view2d_ylim = ax2.get_ylim()
        ax2.cla()
        ax2.set_facecolor("#0a0a1a")

        ws   = self._omap.world_size
        orig = self._omap.origin
        ext  = [orig[0] - ws/2, orig[0] + ws/2,
                orig[1] - ws/2, orig[1] + ws/2]

        if cost_grid is not None:
            ax2.imshow(
                cost_grid.T,   # transpose: rows=x, cols=y → imshow rows=y
                origin="lower",
                extent=ext,
                cmap="RdYlGn_r",
                vmin=1.0, vmax=20.0,
                alpha=0.85,
                interpolation="nearest",
            )

        # Crater boundary circles
        theta = np.linspace(0, 2*math.pi, 200)
        for r, ls, lw, color in [
            (self._omap.world_size/2 * 0.17, "--", 1, "#888888"),  # r_rim approx
            (self._omap.world_size/2 * 0.05, ":",  1, "#666666"),  # r_floor approx
        ]:
            ax2.plot(orig[0] + r * np.cos(theta),
                     orig[1] + r * np.sin(theta),
                     ls, linewidth=lw, color=color, alpha=0.6)

        # Planned path
        if len(path) >= 2:
            px = [p[0] for p in path]
            py = [p[1] for p in path]
            ax2.plot(px, py, color="white", linewidth=1.5, alpha=0.9)

        # Waypoints
        for i, wp in enumerate(self._waypoints):
            color = "#FFD700" if i == wp_idx else "#555555"
            size  = 120 if i == wp_idx else 50
            ax2.scatter([wp.x], [wp.y], color=color, marker="*",
                        s=size, zorder=10)
            if i == wp_idx:
                ax2.annotate(wp.label, (wp.x, wp.y),
                             color="#FFD700", fontsize=6,
                             xytext=(3, 3), textcoords="offset points")

        # Robot
        ax2.scatter([pose.x], [pose.y], color="#00FFFF",
                    marker="o", s=80, zorder=20, edgecolors="white", linewidths=1.5)
        ax2.annotate("robot", (pose.x, pose.y), color="#00FFFF",
                     fontsize=6, xytext=(3, 3), textcoords="offset points")

        # Heading arrow
        arrow_len = 2.0
        ax2.annotate("",
            xy=(pose.x + arrow_len * math.cos(pose.yaw),
                pose.y + arrow_len * math.sin(pose.yaw)),
            xytext=(pose.x, pose.y),
            arrowprops=dict(arrowstyle="->", color="#00FFFF", lw=1.5))

        # Restore user pan/zoom if set; use full-map extent on first draw
        if self._view2d_xlim is not None:
            ax2.set_xlim(self._view2d_xlim)
            ax2.set_ylim(self._view2d_ylim)
        else:
            ax2.set_xlim(ext[0], ext[1])
            ax2.set_ylim(ext[2], ext[3])
            # Record limits after first draw so next frame can save them
            self._view2d_xlim = (ext[0], ext[1])
            self._view2d_ylim = (ext[2], ext[3])
        # NOTE: do NOT call set_aspect("equal") here — it overrides user zoom
        ax2.set_xlabel("X (m)", color="gray", fontsize=7)
        ax2.set_ylabel("Y (m)", color="gray", fontsize=7)
        ax2.tick_params(colors="gray", labelsize=6)
        ax2.set_title("Cost Map + Path (A*)  [pan/zoom with toolbar]",
                      color="white", fontsize=9, pad=4)

        # ---- Status bar ----------------------------------------------
        ax_s = self._ax_status
        ax_s.cla()
        ax_s.set_facecolor("#111122")
        ax_s.axis("off")

        wp_label = (self._waypoints[wp_idx].label
                    if wp_idx < len(self._waypoints) else "DONE")
        slope_deg = (math.degrees(math.atan(local_out.slope_ahead))
                     if local_out and not math.isnan(local_out.slope_ahead) else 0.0)
        cmd_str = f"cmd=({cmd[0]:+.2f}, {cmd[1]:+.2f}, {cmd[2]:+.2f})"
        pol_str = self._shared.get("policy_status", "fixed")

        # Show forward obstacle distance if within 5 m
        fwd_str = ""
        if local_out and hasattr(local_out, "fwd_obstacle_dist"):
            d = local_out.fwd_obstacle_dist
            if d < 5.0:
                fwd_str = f"  |  fwd_obs={d:.1f}m"

        status = (
            f"Mission: {mission.upper()}  |  "
            f"WP: {wp_idx}/{len(self._waypoints)}  [{wp_label}]  |  "
            f"Speed: {speed:.2f} m/s  |  "
            f"Slope: {slope_deg:.1f}°  |  "
            f"{cmd_str}  |  "
            f"Policy: {pol_str}  |  "
            f"State: {rec_status}"
            f"{fwd_str}"
        )
        ax_s.text(0.02, 0.5, status,
                  transform=ax_s.transAxes,
                  color="#00FFFF", fontsize=8.5,
                  verticalalignment="center",
                  fontfamily="monospace")

        # draw_idle() redraws only if dirty — don't call flush_events() here;
        # it's called every 50 ms in the run() loop so rotation stays smooth.
        self._fig.canvas.draw_idle()

    def _style_axes(self) -> None:
        """Apply dark theme styling to all axes."""
        for ax in [self._ax2d, self._ax_status]:
            ax.set_facecolor("#0a0a1a")
            for spine in ax.spines.values():
                spine.set_edgecolor("#333355")
        self._ax3d.set_facecolor("#0a0a1a")
        self._fig.patch.set_facecolor("#0a0a1a")
