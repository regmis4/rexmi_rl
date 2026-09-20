# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Live navigation dashboard — 2D costmap overview + 2D local zoom + status bar.

Runs in a daemon thread at ~2 Hz so it never blocks the sim loop.

Layout (Phase N-2 perf rewrite)
---------------------------------
  Left panel  : 2D cost map overview  (full 64 m × 64 m world)
                • green=clear, red=blocked, yellow=steep slope
                • A* path overlay + robot + waypoints

  Right panel : 2D cost map LOCAL zoom (16 m × 16 m centred on robot)
                • Same cost map, zoomed in so fine detail is visible
                • Robot position (cyan circle + heading arrow)
                • Forward scanner hit strip (orange dots)
                • A* sub-path to next waypoint

  Bottom bar  : Mission | WP | Speed | Slope | SLAM status | Policy | State

Performance note
----------------
  Previous design had a 3D Axes3D SLAM scatter plot (up to 30 k points) with
  auto-rotation — this consumed ~50–100 ms/redraw in mpl's C backend and
  monopolised the Tk event loop.

  Replaced with two 2D imshow() panels:
    • imshow() renders a 320×320 cost grid in <2 ms (GPU texture upload)
    • No scatter over 30k points → redraw time drops from ~80 ms to ~5 ms
    • The 2D local zoom gives the same "am I about to hit a wall?" clarity
      that the 3D cloud was meant to convey

  SLAM status is still shown in the status bar (voxel count, ICP RMS).
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
        Reference to the occupancy map (for world_size, origin).
    waypoints : list[Waypoint]
        Mission waypoint list (for labels and star markers).
    update_interval_s : float
        Seconds between full redraws. Default 0.5 s.
    local_zoom_m : float
        Half-side of the local zoom panel in metres. Default 10 m → 20 m×20 m view.
    """

    def __init__(self, shared, lock, omap, waypoints,
                 update_interval_s: float = 0.5,
                 local_zoom_m: float = 10.0, request_queue=None):
        self._requests = request_queue
        self._shared    = shared
        self._lock      = lock
        self._omap      = omap
        self._waypoints = waypoints
        self._interval  = update_interval_s
        self._zoom_m    = local_zoom_m
        self._layer = "terrain"

        # Matplotlib handles — set up lazily in run()
        self._fig        = None
        self._ax_global  = None   # left: full world cost map
        self._ax_local   = None   # right: local 20 m zoom
        self._ax_status  = None   # bottom: text status bar

    # ------------------------------------------------------------------
    # Main loop (runs in daemon thread)
    # ------------------------------------------------------------------

    def _on_map_click(self, event):
        if self._requests is None or event.button != 1 or event.inaxes not in (self._ax_global,self._ax_local):
            return
        if event.xdata is None or event.ydata is None:
            return
        if self._fig.canvas.toolbar and self._fig.canvas.toolbar.mode:
            return  # pan/zoom gestures never command the robot
        with self._lock:
            enabled = self._shared.get('manual_checkpoint_mode', False)
        if enabled:
            self._requests.put(('target',float(event.xdata),float(event.ydata)))

    def run(self, stop_event: threading.Event) -> None:
        """Dashboard main loop. Called by Navigator.start_dashboard()."""
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt

        plt.ion()
        self._fig = plt.figure(figsize=(16, 7), facecolor="#0a0a1a")
        self._fig.canvas.manager.set_window_title("REXMI Nav Dashboard")

        gs = self._fig.add_gridspec(
            2, 2,
            height_ratios=[11, 1],
            hspace=0.05, wspace=0.18,
        )
        self._ax_global = self._fig.add_subplot(gs[0, 0])
        self._ax_local  = self._fig.add_subplot(gs[0, 1])
        self._ax_status = self._fig.add_subplot(gs[1, :])

        self._fig.canvas.mpl_connect("button_press_event", self._on_map_click)
        self._style_axes()
        if hasattr(self._omap,'roughness'):
            from matplotlib.widgets import RadioButtons
            selector=self._fig.add_axes([.38,.925,.25,.07],facecolor='#dddddd')
            self._layer_buttons=RadioButtons(selector,('terrain','slope','roughness','traversal'))
            self._layer_buttons.on_clicked(lambda value:setattr(self,'_layer',value))

        _flush_interval = 0.05
        _next_redraw    = time.monotonic()

        while not stop_event.is_set():
            now = time.monotonic()
            if now >= _next_redraw:
                try:
                    self._redraw(plt)
                except Exception as e:
                    print(f"[Dashboard] render error: {e}")
                _next_redraw = now + self._interval

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
            pose       = self._shared["pose"]
            wp_idx     = self._shared["wp_idx"]
            path       = list(self._shared["planned_path"])
            local_out  = self._shared["local_out"]
            rec_status = self._shared["recovery_status"]
            mission    = self._shared["mission"]
            cmd        = self._shared["cmd"]
            cost_grid  = self._shared["cost_grid"]
            observed_mask = self._shared.get("observed_mask")
            clearance_mask = self._shared.get("clearance_mask")
            speed      = self._shared["speed"]
            slam_conv  = self._shared["slam_converged"]
            slam_size  = self._shared["slam_map_size"]
            slam_rms   = self._shared["slam_rms"]
            slam_icpn  = self._shared["slam_icp_count"]
            fxs, fys, _ = self._shared["fwd_cloud_xyz"]   # forward scan (orange)
            pol_str    = self._shared.get("policy_status", "fixed")
            trajectory = list(self._shared.get("trajectory", []))
            checkpoint = self._shared.get("checkpoint")
            steering = self._shared.get("steering_target")
            nav_reason = self._shared.get("nav_reason", "")
            coverage = self._shared.get("coverage", 0.)
            age = self._shared.get("observation_age", 0.)
            mapped = self._shared.get("mapped_coverage")
            mapped_area = self._shared.get("mapped_area", 0.)
            layers=self._shared.get('survey_layers')

        ws   = self._omap.world_size
        orig = self._omap.origin
        ext  = [orig[0] - ws/2, orig[0] + ws/2,
                orig[1] - ws/2, orig[1] + ws/2]

        # Same discrete planner-cost bands as the Isaac Sim overlay.
        import matplotlib.colors as mcolors
        from matplotlib.lines import Line2D
        _cmap = mcolors.ListedColormap(["#14A6A6", "#F2A31F", "#FF00FF"]
                                     + (["#8C7542"] if clearance_mask is not None else []) + ["#FF0000"])
        _cmap.set_bad("#555577")
        _norm = mcolors.BoundaryNorm([0,2,6]+([12] if clearance_mask is not None else [])+[20,1000], _cmap.N)
        if cost_grid is not None:
            unknown = ~observed_mask if observed_mask is not None else cost_grid == 5.
            if clearance_mask is not None:
                cost_grid=cost_grid.copy()
                cost_grid[clearance_mask]=12.  # display category, never a planner cost
            cost_grid = np.ma.masked_where(unknown,cost_grid)
        cmap_kwargs = dict(cmap=_cmap,norm=_norm,alpha=.88,interpolation="nearest")

        image_grid=cost_grid.T if cost_grid is not None else None
        layer_legend=None
        if layers is not None and cost_grid is not None:
            from .perception_view import survey_colors
            # The buffer category above is display-only; restore its hazard cost
            # for the common palette, which receives a separate clearance mask.
            values=np.asarray(cost_grid.filled(5.)).copy()
            if clearance_mask is not None: values[clearance_mask]=20.
            rgb,layer_legend=survey_colors(values,observed_mask,clearance_mask,self._layer,
                layers['slope'],layers['roughness'],layers['failures'],layers['successes'])
            image_grid=rgb.transpose(1,0,2)
            cmap_kwargs=dict(interpolation='nearest')

        # ── Left panel: full-world cost map ──────────────────────────────
        ag = self._ax_global
        ag.cla()
        ag.set_facecolor("#0a0a1a")

        if cost_grid is not None:
            ag.imshow(image_grid, origin="lower", extent=ext, **cmap_kwargs)

        # Crater boundary circles (approximate — proportional to world_size)
        theta = np.linspace(0, 2 * math.pi, 200)
        for r_frac, ls in [(0.34, "--"), (0.09, ":")]:
            r = ws / 2 * r_frac
            ag.plot(orig[0] + r * np.cos(theta),
                    orig[1] + r * np.sin(theta),
                    ls, linewidth=1, color="#888888", alpha=0.6)

        # Robot trajectory trace (magenta, faded — last 500 positions)
        if len(trajectory) >= 2:
            txs = [p[0] for p in trajectory]
            tys = [p[1] for p in trajectory]
            ag.plot(txs, tys,
                    color="#FF44AA", linewidth=1.2, alpha=0.65,
                    zorder=7, label="robot trail")
            # Mark start of trace with a small circle
            ag.scatter([txs[0]], [tys[0]], color="#FF44AA", s=20,
                       alpha=0.5, zorder=7)

        # A* planned path (bright white, thick — clearly distinct from trajectory)
        if len(path) >= 2:
            ag.plot([p[0] for p in path], [p[1] for p in path],
                    color="#FF1828", linewidth=2.5, alpha=0.95,
                    zorder=8, linestyle="--", label="Active route")

        # Waypoints
        for i, wp in enumerate(self._waypoints):
            col  = "#FFD700" if i == wp_idx else "#555555"
            size = 120 if i == wp_idx else 50
            ag.scatter([wp.x], [wp.y], color=col, marker="*", s=size, zorder=10)
            if i == wp_idx:
                ag.annotate(wp.label, (wp.x, wp.y), color="#FFD700",
                             fontsize=6, xytext=(3, 3),
                             textcoords="offset points")

        # Robot
        ag.scatter([pose.x], [pose.y], color="#00FFFF",
                   marker="o", s=80, zorder=20,
                   edgecolors="white", linewidths=1.5)
        _al = 2.0
        ag.annotate("",
            xy=(pose.x + _al * math.cos(pose.yaw),
                pose.y + _al * math.sin(pose.yaw)),
            xytext=(pose.x, pose.y),
            arrowprops=dict(arrowstyle="->", color="#00FFFF", lw=1.5),
            zorder=21)

        ag.set_xlim(ext[0], ext[1])
        ag.set_ylim(ext[2], ext[3])
        ag.set_xlabel("X (m)", color="gray", fontsize=7)
        ag.set_ylabel("Y (m)", color="gray", fontsize=7)
        ag.tick_params(colors="gray", labelsize=6)
        ag.set_title(
            f"{self._layer.title()} — full mission survey  "
            f"({self._omap.n_cells}×{self._omap.n_cells} @ {self._omap.cell_size*100:.0f} cm/cell)",
            color="white", fontsize=9, pad=4,
        )

        # ── Costmap legend (colour patches) ──────────────────────────────
        import matplotlib.patches as mpatches
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        _legend_items = [
            mpatches.Patch(color=_cmap(_norm(1.0)),  label="Low planner cost"),
            mpatches.Patch(color=_cmap(_norm(2.0)),  label="Elevated cost"),
            mpatches.Patch(color=_cmap(_norm(6.0)),  label="High cost"),
            mpatches.Patch(color=_cmap(_norm(20.0)),  label="Obstacle / very steep terrain"),
            *([mpatches.Patch(color=_cmap(_norm(12.0)), label="Clearance buffer (planner avoids)")]
              if clearance_mask is not None else []),
            mpatches.Patch(color="#555577", label="Unknown"),
            Line2D([0],[0],color="#FF1828",ls="--",label="Active route"),
            Line2D([0],[0],color="#FFD700",marker="D",ls="",label="Local checkpoint"),
        ]
        if layer_legend is not None:
            _legend_items=[Line2D([],[],color='none',label=item.strip()) for item in layer_legend.split('|')] + _legend_items[-2:]
        ag.legend(handles=_legend_items, loc="lower left",
                  fontsize=6, framealpha=0.6,
                  facecolor="#111133", edgecolor="#334466",
                  labelcolor="white")

        # ── Right panel: local zoom (robot-centred) ───────────────────────
        al = self._ax_local
        al.cla()
        al.set_facecolor("#0a0a1a")

        z = self._zoom_m
        lx0, lx1 = pose.x - z, pose.x + z
        ly0, ly1 = pose.y - z, pose.y + z

        if cost_grid is not None:
            al.imshow(image_grid, origin="lower", extent=ext, **cmap_kwargs)

        # Robot trajectory trace in local zoom (magenta — recent path history)
        if len(trajectory) >= 2:
            txs = [p[0] for p in trajectory]
            tys = [p[1] for p in trajectory]
            # Only draw points that fall within the local view (± 2× zoom for smooth clipping)
            al.plot(txs, tys,
                    color="#FF44AA", linewidth=1.8, alpha=0.80,
                    zorder=7, label="robot trail")

        # A* planned path (bright white dashed — clearly distinct from magenta trail)
        if len(path) >= 2:
            al.plot([p[0] for p in path], [p[1] for p in path],
                    color="#FF1828", linewidth=2.5, alpha=0.95,
                    zorder=8, linestyle="--", label="Active route")

        # Waypoints in local view
        for i, wp in enumerate(self._waypoints):
            if lx0 <= wp.x <= lx1 and ly0 <= wp.y <= ly1:
                col = "#FFD700" if i == wp_idx else "#888888"
                al.scatter([wp.x], [wp.y], color=col,
                            marker="*", s=100, zorder=10)
                al.annotate(wp.label, (wp.x, wp.y), color=col,
                             fontsize=7, xytext=(3, 3),
                             textcoords="offset points")

        # Forward scanner hits (orange)
        if len(fxs) > 0:
            al.scatter(fxs, fys, color="#FF8800", s=12,
                       alpha=0.9, zorder=15, label="fwd obs")

        # Robot
        al.scatter([pose.x], [pose.y], color="#00FFFF",
                   marker="o", s=100, zorder=20,
                   edgecolors="white", linewidths=2)
        _al2 = 1.5
        al.annotate("",
            xy=(pose.x + _al2 * math.cos(pose.yaw),
                pose.y + _al2 * math.sin(pose.yaw)),
            xytext=(pose.x, pose.y),
            arrowprops=dict(arrowstyle="->", color="#00FFFF", lw=2),
            zorder=21)
        al.annotate("robot", (pose.x, pose.y), color="#00FFFF",
                    fontsize=7, xytext=(3, 3), textcoords="offset points")

        al.set_xlim(lx0, lx1)
        al.set_ylim(ly0, ly1)
        al.set_xlabel("X (m)", color="gray", fontsize=7)
        al.set_ylabel("Y (m)", color="gray", fontsize=7)
        al.tick_params(colors="gray", labelsize=6)
        al.set_title(
            f"Local zoom  {2*z:.0f} m × {2*z:.0f} m  |  pan/zoom: toolbar",
            color="white", fontsize=9, pad=4,
        )

        # ── Status bar ────────────────────────────────────────────────────
        ax_s = self._ax_status
        ax_s.cla()
        for ax in (ag,al):
            if checkpoint is not None:
                ax.scatter(*checkpoint,color="#FFD700",marker="D",s=55,zorder=15,label="Local checkpoint")
            if steering is not None:
                ax.scatter(*steering,color="white",marker="+",s=60,zorder=15,label="Steering target")
        ax_s.set_facecolor("#111122")
        ax_s.axis("off")

        wp_label  = (self._waypoints[wp_idx].label
                     if wp_idx < len(self._waypoints) else "DONE")
        slope_deg = (math.degrees(math.atan(local_out.slope_ahead))
                     if local_out and not math.isnan(local_out.slope_ahead) else 0.0)
        if layers is not None:
            cell=self._omap.world_to_cell(pose.x,pose.y)
            if cell is not None: slope_deg=float(np.degrees(np.arctan(layers['slope'][cell])))
        cmd_str   = f"cmd=({cmd[0]:+.2f}, {cmd[1]:+.2f}, {cmd[2]:+.2f})"

        if slam_conv:
            slam_str = f"SLAM:ON(rms={slam_rms:.3f}m,{slam_size}vox)"
        elif slam_size > 0:
            slam_str = f"SLAM:BOOT({slam_size}vox)"
        else:
            slam_str = "SLAM:SIM"

        fwd_str = ""
        if local_out and hasattr(local_out, "fwd_obstacle_dist"):
            d = local_out.fwd_obstacle_dist
            if d < 5.0:
                fwd_str = f"  |  fwd_obs={d:.1f}m"

        status = (
            f"Mission: {mission.upper()}  |  "
            f"WP: {wp_idx}/{len(self._waypoints)} [{wp_label}]  |  "
            f"Speed: {speed:.2f} m/s  |  "
            f"Slope: {slope_deg:.1f}°  |  "
            f"{slam_str}  |  "
            f"{cmd_str}  |  "
            f"Policy: {pol_str}  |  "
            f"State: {rec_status}"
            f"{fwd_str}"
        )
        if nav_reason:
            status += f"\n{nav_reason} | fresh={coverage:.0%} | oldest={age:.1f}s"
            if mapped is not None:
                status += f" | mapped={mapped:.0%} | surveyed={mapped_area:.1f} m²"
        ax_s.text(0.01, 0.5, status,
                  transform=ax_s.transAxes,
                  color="#00FFFF", fontsize=8.0,
                  verticalalignment="center",
                  fontfamily="monospace")

        self._fig.canvas.draw_idle()

    def _style_axes(self) -> None:
        """Apply dark theme styling to all axes."""
        for ax in [self._ax_global, self._ax_local, self._ax_status]:
            ax.set_facecolor("#0a0a1a")
            for spine in ax.spines.values():
                spine.set_edgecolor("#333355")
        self._fig.patch.set_facecolor("#0a0a1a")
