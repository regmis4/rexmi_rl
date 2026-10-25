# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Live navigation dashboard — survey and legend left, local zoom and logs right.

Runs in a daemon thread at ~2 Hz so it never blocks the sim loop.

Layout
---------------------------------
  Left panel  : 2D cost map overview  (full 64 m × 64 m world)
                • terrain, slope, roughness or traversal colours
                • A* path overlay + robot + waypoints

  Right panel : 2D cost map LOCAL zoom (16 m × 16 m centred on robot)
                • Same cost map, zoomed in so fine detail is visible
                • Robot position (cyan circle + heading arrow)
                • Forward scanner hit strip (orange dots)
                • A* sub-path to next waypoint

  Lower left : Colour/symbol legend and survey-layer selector
  Lower right: Navigation, policy and telemetry below the local zoom

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

  SLAM status is still shown below the local map (voxel count, ICP RMS).
"""

from __future__ import annotations

import math
import threading
import time
import textwrap
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
                 local_zoom_m: float = 10.0, request_queue=None,
                 selected_path_only: bool = False):
        self._requests = request_queue
        self._shared    = shared
        self._lock      = lock
        self._omap      = omap
        self._waypoints = waypoints
        self._interval  = update_interval_s
        self._zoom_m    = local_zoom_m
        self._layer = "terrain"
        self._selected_path_only = selected_path_only

        # Matplotlib handles — set up lazily in run()
        self._fig        = None
        self._ax_global  = None   # left: full world cost map
        self._ax_local   = None   # right: local 20 m zoom
        self._ax_legend  = None   # below survey: all colours and map symbols
        self._ax_status  = None   # below local zoom: navigation and telemetry
        self._ax_layers  = None

    # ------------------------------------------------------------------
    # Main loop (runs in daemon thread)
    # ------------------------------------------------------------------

    def _on_map_click(self, event):
        if getattr(event,'dblclick',False): return
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
        self._create_figure(plt)

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

    def _create_figure(self, plt) -> None:
        """Keep maps clickable and reserve separate left/right information panels."""
        self._fig = plt.figure(figsize=(16, 9.5), facecolor="#0a0a1a")
        self._fig.canvas.manager.set_window_title("REXMI Nav Dashboard")

        gs = self._fig.add_gridspec(
            2, 2,
            height_ratios=[2.25, 1],
            left=.055, right=.97, bottom=.045, top=.95,
            hspace=.16, wspace=.16,
        )
        self._ax_global = self._fig.add_subplot(gs[0, 0])
        self._ax_local  = self._fig.add_subplot(gs[0, 1])
        legend_grid = gs[1, 0].subgridspec(1, 2, width_ratios=[4, 1], wspace=.06)
        self._ax_legend = self._fig.add_subplot(legend_grid[0, 0])
        self._ax_layers = self._fig.add_subplot(legend_grid[0, 1])
        self._ax_status = self._fig.add_subplot(gs[1, 1])

        self._fig.canvas.mpl_connect("button_press_event", self._on_map_click)
        self._style_axes()
        if hasattr(self._omap,'roughness'):
            from matplotlib.widgets import RadioButtons
            self._ax_layers.set_title('VIEW', color='#aab4cc', fontsize=8, loc='left')
            from .perception_view import LAYER_NAMES
            self._layer_buttons=RadioButtons(self._ax_layers,LAYER_NAMES,activecolor='#00dddd')
            for label in self._layer_buttons.labels:
                label.set_color('white');label.set_fontsize(8)
            self._layer_buttons.on_clicked(lambda value:setattr(self,'_layer',value))
        else:
            self._ax_layers.axis('off')

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _redraw(self, plt) -> None:
        """Redraw all panels with current shared state."""
        with self._lock:
            pose       = self._shared["pose"]
            wp_idx     = self._shared["wp_idx"]
            path       = list(self._shared["planned_path"])
            mission_path = list(self._shared.get('mission_path', []))
            route_profile = self._shared.get('route_profile', {})
            route_complete = self._shared.get('mission_route_complete', True)
            local_out  = self._shared["local_out"]
            rec_status = self._shared["recovery_status"]
            mission    = self._shared["mission"]
            mission_stage = self._shared.get('mission_stage')
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
            selection_status = self._shared.get('selection_status','')
            slip_metrics = self._shared.get('slip_metrics',{})
            route_decision = self._shared.get('route_decision','')
            route_purpose = self._shared.get('route_purpose','mission')
            retreat_goal = self._shared.get('retreat_goal')
            grade_limits = self._shared.get('grade_limits')
            switch_reason = self._shared.get("switch_reason", "")
            coverage = self._shared.get("coverage", 0.)
            age = self._shared.get("observation_age", 0.)
            mapped = self._shared.get("mapped_coverage")
            mapped_area = self._shared.get("mapped_area", 0.)
            layers=self._shared.get('survey_layers')
            navigation_status=self._shared.get('navigation_status')
            approach_options=list(self._shared.get('route_options', []))
        if self._selected_path_only:
            mission_path=[]
            approach_options=[]

        ws   = self._omap.world_size
        orig = self._omap.origin
        ext  = [orig[0] - ws/2, orig[0] + ws/2,
                orig[1] - ws/2, orig[1] + ws/2]

        # Same discrete planner-cost bands as the Isaac Sim overlay.
        import matplotlib.colors as mcolors
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
        if layers is not None and cost_grid is not None:
            from .perception_view import survey_colors
            # The buffer category above is display-only; restore its hazard cost
            # for the common palette, which receives a separate clearance mask.
            values=np.asarray(cost_grid.filled(5.)).copy()
            if clearance_mask is not None: values[clearance_mask]=20.
            rgb,_=survey_colors(values,observed_mask,clearance_mask,self._layer,
                layers['slope'],layers['roughness'],layers['failures'],layers['successes'],
                layers.get('no_go'),layers.get('approach_blocked'),layers.get('shelf_mask'),layers)
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

        # Show the larger route from which local checkpoints are selected.
        if len(mission_path)>=2:
            ag.plot([p[0] for p in mission_path],[p[1] for p in mission_path],
                    color='#FF1828',linewidth=1.3,alpha=.5,ls=':',zorder=7)
        # Active checkpoint route remains a stronger red dashed line.
        if len(path) >= 2:
            ag.plot([p[0] for p in path], [p[1] for p in path],
                    color="#FF1828", linewidth=2.5, alpha=0.95,
                    zorder=8, linestyle="--", label="Active route")

        # Waypoints
        for i, wp in enumerate(self._waypoints):
            col  = "#39E75F" if i == wp_idx else "#555555"
            size = 120 if i == wp_idx else 50
            ag.scatter([wp.x], [wp.y], color=col, marker="*", s=size, zorder=10)
            if i == wp_idx:
                ag.annotate(wp.label, (wp.x, wp.y), color="#39E75F",
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

        # All legends live below the survey, never over either clickable map.
        self._draw_legend(layers is not None, clearance_mask is not None,
                          continuous=layers is not None and 'shelf_mask' in layers)

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
                col = "#39E75F" if i == wp_idx else "#888888"
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

        # ── Logs beneath the right-hand zoom ─────────────────────────────
        ax_s = self._ax_status
        ax_s.cla()
        for ax in (ag,al):
            if navigation_status:
                for number,option in enumerate(approach_options,1):
                    endpoint=option.get('endpoint')
                    if endpoint is None:continue
                    probability=option.get('reachability',{}).get('probability')
                    label=f'A{number}' if probability is None else f'A{number}: {probability:.0%}'
                    color='#79FFD4' if option.get('selected') else '#a9a9bd'
                    approach=option.get('path',[])
                    if len(approach)>1:
                        line=np.asarray(approach)
                        ax.plot(line[:,0],line[:,1],color=color,lw=1.,ls='--',alpha=.7,zorder=12)
                    ax.scatter(*endpoint,marker='o',s=45,facecolors='none',edgecolors=color,zorder=14)
                    ax.annotate(label,endpoint,xytext=(7,5+12*(number-1)),textcoords='offset points',color=color,fontsize=7,zorder=15)
            if checkpoint is not None:
                ax.scatter(*checkpoint,color="#39E75F",marker="D",s=55,zorder=15,label="Local checkpoint")
            if steering is not None:
                ax.scatter(*steering,color="white",marker="+",s=60,zorder=15,label="Steering target")
        ax_s.set_facecolor("#111122")
        ax_s.axis("off")

        wp_label  = (self._waypoints[wp_idx].label
                     if wp_idx < len(self._waypoints) else "Selecting safe exit"
                     if mission_stage and mission_stage.get('goal_mode')=='any_exit' and not self._waypoints else "DONE")
        slope_deg = (math.degrees(math.atan(local_out.slope_ahead))
                     if local_out and not math.isnan(local_out.slope_ahead) else 0.0)
        if layers is not None:
            cell=self._omap.world_to_cell(pose.x,pose.y)
            if cell is not None: slope_deg=float(np.degrees(np.arctan(layers['slope'][cell])))
        cmd_str   = f"Command: x {cmd[0]:+.2f}, y {cmd[1]:+.2f} m/s | yaw {cmd[2]:+.2f} rad/s"

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

        lines = [
            f"Mission: {mission.upper()} | WP: {wp_idx}/{len(self._waypoints)} [{wp_label}]",
            f"State: {rec_status} | Policy: {pol_str}",
            f"Speed: {speed:.2f} m/s | Ground slope: {slope_deg:.1f}°{fwd_str}",
            cmd_str,
        ]
        if nav_reason:
            lines.append('Navigation: '+nav_reason)
        if mission_stage:
            lines.append(f'Goal: {mission_stage["stage"]} {mission_stage["goal"] or mission_stage.get("goal_description", "")}')
        if navigation_status:
            prior=getattr(self._omap,'terrain_prior',None)
            if prior:
                lines.append('Terrain prior: '+prior['kind']+f' | initial coverage {prior.get("coverage",0.):.0%}')
            estimate=navigation_status.get('reachability') or {}
            p=estimate.get('probability')
            text='Unvalidated' if p is None else f'{p:.0%}'
            lines.append(f'Reach and settle: {text} | {navigation_status["mode"]} | limit {navigation_status["command_limit"]:.2f} m/s')
            if navigation_status.get('architecture') == 'behavior_tree':
                lines.append(f'Behavior: {navigation_status["branch"]} | movement: {navigation_status["movement_state"]}')
                lines.append(f'Turn staging: {"ready" if navigation_status["turn_enabled"] else "unavailable"} | uphill limit {navigation_status["uphill_limit_deg"]:.0f}°')
            interval=estimate.get('interval')
            if interval:
                lines.append(f'Conditional 90% interval: {interval[0]:.0%}–{interval[1]:.0%}; local outcomes {estimate.get("successes",0)} success / {estimate.get("failures",0)} failure')
            else:
                lines.append(f'Local evidence: {estimate.get("successes",0)} successes, {estimate.get("failures",0)} failures; probability unvalidated')
            if estimate.get('reason'):lines.append('Evidence: '+estimate['reason'])
            for number,option in enumerate(approach_options,1):
                value=option.get('reachability',{}).get('probability')
                detail=option.get('rejection') or ('unvalidated' if value is None else f'{value:.0%}')
                lines.append(f'A{number} {option["profile"]}{" (selected)" if option.get("selected") else ""}: '+detail)
            from .perception_view import route_cost_text
            explanation=route_cost_text(approach_options)
            if explanation:lines.extend(explanation.splitlines())
            if layers is not None and 'shelf_mask' in layers:
                lines.append('Terrain cost: teal 1 → gold 6.5 → rose 12+; mint shelf outline')
        if selection_status:
            lines.append('Checkpoint click: '+selection_status)
        if slip_metrics:
            lines.append(f"Motion: {slip_metrics['status']} | progress {slip_metrics['progress_mps']:+.2f} m/s | side {slip_metrics['lateral_mps']:+.2f} m/s")
        if route_decision:lines.append('Route choice: '+route_decision)
        if retreat_goal is not None:lines.append(f'Regroup point: ({retreat_goal[0]:.1f}, {retreat_goal[1]:.1f}) | mission target kept')
        if rec_status.startswith('EDGE_'):
            lines.append('Edge escape: restore footprint support | centre/travel limit 35°')
        elif grade_limits:lines.append(f'Ascent: prefer {grade_limits[0]:.0f}° | max {grade_limits[1]:.0f}° | side max {grade_limits[2]:.0f}°')
        if switch_reason:
            lines.append('Switch: '+switch_reason)
        if nav_reason:
            lines.append(f"Fresh: {coverage:.0%} | Oldest observation: {age:.1f}s")
            if mapped is not None:
                lines.append(f"Mapped: {mapped:.0%} | Surveyed: {mapped_area:.1f} m²")
        lines.append(slam_str)
        if route_profile:
            route_label='Edge backout' if rec_status.startswith('EDGE_') else 'Downhill retreat' if route_purpose=='retreat' else 'Mission route' if route_complete else 'Observed frontier route'
            lines.append(f"{route_label}: {route_profile['length_m']:.1f} m | max uphill {route_profile['max_uphill_deg']:.1f}°")
            lines.append(f"Side slope max: {route_profile['max_side_slope_deg']:.1f}° | uphill >25°: {route_profile['uphill_over_25deg_m']:.1f} m")
        ax_s.set_title('NAVIGATION & TELEMETRY', color='#aab4cc', fontsize=8, loc='left')
        # Wrap to this panel rather than letting long rejection reasons cross
        # the window. Recompute on redraw so window resizing remains usable.
        bounds=ax_s.get_window_extent()
        for fontsize in (9.,8.5,8.,7.5):
            width=max(24,int(bounds.width*.97/(fontsize*self._fig.dpi/72*.61)))
            status='\n'.join(textwrap.fill(line,width=width,subsequent_indent='  ')
                             for line in lines)
            if len(status.splitlines())*fontsize*self._fig.dpi/72*1.35<bounds.height*.92:
                break
        ax_s.text(0.0, 0.98, status,
                  transform=ax_s.transAxes,
                  color="#bde8eb", fontsize=fontsize,
                  verticalalignment="top", linespacing=1.35,
                  fontfamily="monospace", clip_on=True)

        self._fig.canvas.draw_idle()

    def _draw_legend(self, survey_layers, has_clearance, continuous=False) -> None:
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        palette = {
            'terrain': [((.08,.65,.65),'Low cost'), ((.95,.64,.12),'Elevated cost'),
                        ((1.,0.,1.),'High cost (traversable)')],
            'slope': [((.08,.65,.65),'Slope 0°'), ((.54,.65,.365),'Slope 17.5°'),
                      ((1.,.65,.08),'Slope 35°')],
            'roughness': [((.08,.65,.65),'Roughness 0 cm RMS'), ((.54,.65,.365),'Roughness 5 cm RMS'),
                          ((1.,.65,.08),'Roughness ≥10 cm RMS'), ((.32,.32,.32),'Insufficient samples')],
            'footprint': [((.08,.65,.65),'Support-neighborhood RMS 0 cm'),((1.,.65,.08),'RMS ≥10 cm: +4 cost'),
                          ((.32,.32,.32),'Unavailable: +8 unknown cost')],
            'steps': [((.08,.65,.65),'Detrended cell jump 0 cm'),((1.,.65,.08),'Jump ≥10 cm: +4 cost'),
                      ((.32,.32,.32),'Unavailable: +8 unknown cost')],
            'support': [((.08,.65,.65),'Minimum bin/pair coverage 0%'),((1.,.65,.08),'Coverage 100%'),
                        ((.32,.32,.32),'Either coverage <80% or invalid fit: +8 cost')],
            'traversal': [((.32,.32,.32),'Untested'), ((.12,.75,.3),'Successful progress'),
                          ((1.,.55,.12),'Failed approach'), ((.65,.25,.85),'Direction excluded')],
        }
        entries=list(palette[self._layer if survey_layers else 'terrain'])
        if continuous and self._layer=='terrain':
            entries=[((.08,.65,.65),'Cost 1'),((.95,.72,.18),'Cost 6.5'),
                     ((.85,.15,.45),'Cost 12+'),((.65,1.,.85),'Shelf boundary')]
        entries.append(((1.,0.,0.),'Obstacle' if survey_layers else 'Obstacle / steep terrain'))
        if survey_layers:entries.append(((.60,.05,.08),'Learned no-go zone'))
        if survey_layers: entries.append(((1.,.29,.12),'Slope >35° (blocked)'))
        if has_clearance: entries.append(((.55,.46,.26),'Cream: clearance buffer'))
        entries.append(((.33,.33,.47),'Unknown terrain'))
        handles=[Patch(color=color,label=label) for color,label in entries]
        handles += [
            Line2D([],[],color='#FF1828',ls='--',lw=2,label='Active route'),
            Line2D([],[],color='#FF1828',ls=':',alpha=.5,label='Larger planned route'),
            Line2D([],[],color='#FF44AA',label='Robot trail'),
            Line2D([],[],color='#00FFFF',marker='o',ls='',label='Robot / heading'),
            Line2D([],[],color='#39E75F',marker='*',ls='',label='Mission waypoint'),
            Line2D([],[],color='#39E75F',marker='D',ls='',label='Local checkpoint'),
            Line2D([],[],color='white',marker='+',ls='',label='Steering target'),
            Line2D([],[],color='#FF8800',marker='.',ls='',label='Forward scanner hits'),
            Line2D([],[],color='#888888',ls='--',label='Crater outline (approx.)'),
        ]
        ax=self._ax_legend
        ax.cla();ax.axis('off')
        ax.set_title('MAP COLORS & SYMBOLS',color='#aab4cc',fontsize=8,loc='left')
        ax.legend(handles=handles,loc='upper left',ncol=2,fontsize=8,
                  frameon=False,labelcolor='white',borderaxespad=0.,
                  handlelength=1.8,columnspacing=1.4,labelspacing=.7)

    def _style_axes(self) -> None:
        """Apply dark theme styling to all axes."""
        for ax in [self._ax_global, self._ax_local, self._ax_legend, self._ax_status, self._ax_layers]:
            ax.set_facecolor("#0a0a1a")
            for spine in ax.spines.values():
                spine.set_edgecolor("#333355")
        self._fig.patch.set_facecolor("#0a0a1a")
