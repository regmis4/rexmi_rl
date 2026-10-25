# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause
"""Live local map, distance profile and command/response dashboard (~2 Hz)."""
from __future__ import annotations

import math
import threading
import time
import textwrap
import numpy as np

BACKGROUND = '#0b0d18'
PANEL = '#111522'
MUTED = '#929db5'
INK = '#edf2ff'
CYAN = '#00d9ff'
GOLD = '#ffba76'


class Dashboard:
    def __init__(self, shared, lock, omap, waypoints, update_interval_s=.5,
                 local_zoom_m=10., request_queue=None, selected_path_only=False):
        self._shared, self._lock, self._omap = shared, lock, omap
        self._waypoints, self._requests = waypoints, request_queue
        self._interval, self._zoom_m = update_interval_s, local_zoom_m
        self._selected_path_only = selected_path_only
        self._layer = 'terrain'
        self._fig = self._ax_global = self._ax_local = None
        self._colorbar = None

    def _on_map_click(self, event):
        if getattr(event, 'dblclick', False): return
        if self._requests is None or event.button != 1 or event.inaxes is None:
            return
        if event.inaxes not in (self._ax_global, self._ax_local): return
        if event.xdata is None or event.ydata is None: return
        if self._fig.canvas.toolbar and self._fig.canvas.toolbar.mode: return
        with self._lock:
            enabled = self._shared.get('manual_checkpoint_mode', False)
        if enabled:
            self._requests.put(('target', float(event.xdata), float(event.ydata)))

    def run(self, stop_event: threading.Event):
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        plt.ion()
        self._create_figure(plt)
        next_redraw = time.monotonic()
        while not stop_event.is_set():
            if time.monotonic() >= next_redraw:
                try: self._redraw(plt)
                except Exception as e: print(f'[Dashboard] render error: {e}')
                next_redraw = time.monotonic()+self._interval
            try: self._fig.canvas.flush_events()
            except Exception: pass
            time.sleep(.05)
        plt.close(self._fig)

    def _create_figure(self, plt):
        self._fig = plt.figure(figsize=(16, 10), facecolor=BACKGROUND)
        self._fig.canvas.manager.set_window_title('REXMI | Navigation')
        self._mode_title = self._fig.text(.045, .958, '', color=CYAN, fontsize=19, weight='bold')
        with self._lock:
            self._update_mode_title(dict(self._shared))
        self._headline = self._fig.text(.955, .96, '', color=MUTED, fontsize=10, ha='right')
        self._ax_local = self._fig.add_axes([.055, .28, .49, .615])
        self._ax_profile = self._fig.add_axes([.635, .655, .30, .235])
        self._ax_slope = self._ax_profile.twinx()
        self._ax_drive = self._fig.add_axes([.625, .225, .32, .345])
        self._ax_status = self._fig.add_axes([.615, .045, .34, .13])
        self._ax_colorbar = self._fig.add_axes([.065, .205, .47, .014])
        self._ax_legend = self._fig.add_axes([.055, .05, .35, .115])
        self._ax_layers = self._fig.add_axes([.44, .035, .10, .13])
        self._fig.canvas.mpl_connect('button_press_event', self._on_map_click)
        from matplotlib.widgets import RadioButtons
        from .perception_view import LAYER_NAMES
        self._ax_layers.set_facecolor(BACKGROUND)
        self._layer_buttons = RadioButtons(self._ax_layers, LAYER_NAMES, activecolor=CYAN)
        for label in self._layer_buttons.labels:
            label.set_color(MUTED); label.set_fontsize(8)
        for spine in self._ax_layers.spines.values(): spine.set_visible(False)
        self._layer_buttons.on_clicked(lambda value: setattr(self, '_layer', value))

    @staticmethod
    def _style(ax):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED, labelsize=8)
        for spine in ax.spines.values(): spine.set_color('#30394b')
        ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED)

    def _update_mode_title(self, state):
        mode = state.get('operator_mode')
        status = state.get('recovery_status', '')
        if mode == 'manual' or status == 'MANUAL':
            label, color = 'MANUAL CONTROL', GOLD
        elif mode == 'stop' or status in ('HOLD', 'STOP'):
            label, color = 'NAVIGATION PAUSED', GOLD
        elif state.get('assisted_control') or status == 'REVIEWED_BUTTONS':
            label, color = 'ASSISTED NAVIGATION', GOLD
        else:
            label, color = 'AUTONOMOUS NAVIGATION', CYAN
        self._mode_title.set_text('REXMI / '+label)
        self._mode_title.set_color(color)

    def _redraw(self, plt):
        with self._lock:
            state = dict(self._shared)
        self._update_mode_title(state)
        pose = state['pose']
        path = list(state.get('planned_path', []))
        trail = list(state.get('trajectory', []))
        view = state.get('terrain_display')
        cost = state.get('cost_grid')
        layers = state.get('survey_layers')
        origin, ws = self._omap.origin, self._omap.world_size
        ext = [origin[0]-ws/2, origin[0]+ws/2, origin[1]-ws/2, origin[1]+ws/2]
        if view is None and cost is not None:
            from .terrain_display import cost_view
            known = state.get('observed_mask')
            if known is None: known = cost != 5.
            view = cost_view(cost, known, cost >= 20.)
        image = view['rgb'] if view is not None else None
        layer_legend = None
        if self._layer != 'terrain' and layers is not None and cost is not None:
            from .perception_view import survey_colors
            image, layer_legend = survey_colors(cost, state['observed_mask'], state.get('clearance_mask'),
                self._layer, layers['slope'], layers['roughness'], layers['failures'], layers['successes'],
                layers.get('no_go'), layers.get('approach_blocked'), layers.get('shelf_mask'), layers)
        ax = self._ax_local
        ax.clear(); self._style(ax)
        if image is not None:
            ax.imshow(image.transpose(1, 0, 2), origin='lower', extent=ext, interpolation='nearest')
        if self._layer == 'terrain' and view is not None and 'surface' in view:
            h = np.where(view['known'], view['surface'], np.nan)
            if np.isfinite(h).any() and np.nanmax(h)-np.nanmin(h) > .5:
                x = ext[0]+(np.arange(h.shape[0])+.5)*view['cell_size']
                y = ext[2]+(np.arange(h.shape[1])+.5)*view['cell_size']
                ax.contour(x, y, h.T, levels=np.arange(0, 6, .5), colors='white', alpha=.23, linewidths=.5)
        import matplotlib.patheffects as pe
        if len(trail) > 1:
            trace = np.asarray(trail)
            line, = ax.plot(trace[:, 0], trace[:, 1], color='white', lw=1.8, zorder=7)
            line.set_path_effects([pe.Stroke(linewidth=3.3, foreground='#10121a'), pe.Normal()])
        if not self._selected_path_only:
            for option in state.get('route_options', []):
                points = np.asarray(option.get('path', []))
                if len(points) > 1 and not option.get('selected'):
                    ax.plot(points[:, 0], points[:, 1], color='#a5aec1', alpha=.5, ls=':', lw=1)
        if len(path) > 1:
            route = np.asarray(path)
            ax.plot(route[:, 0], route[:, 1], color=CYAN, lw=2.1, zorder=8)
        for i, wp in enumerate(self._waypoints):
            if i != state.get('wp_idx', 0): continue
            ax.scatter(wp.x, wp.y, marker='*', c='#72f23b', s=140, zorder=10)
            ax.annotate(wp.label, (wp.x, wp.y), xytext=(7, 7), textcoords='offset points',
                        color='#72f23b', fontsize=8, clip_on=True)
        for key, marker, color in [('checkpoint', 'D', '#72f23b'), ('steering_target', '+', 'white')]:
            point = state.get(key)
            if point is not None: ax.scatter(*point, marker=marker, c=color, s=48, zorder=12)
        hits = state.get('fwd_cloud_xyz', ([], [], []))
        if len(hits[0]): ax.scatter(hits[0], hits[1], color='#ff9c24', s=5, alpha=.65, zorder=6)
        ax.scatter(pose.x, pose.y, c=CYAN, s=80, edgecolors='white', lw=1.6, zorder=20)
        ax.annotate('', (pose.x+1.1*math.cos(pose.yaw), pose.y+1.1*math.sin(pose.yaw)),
                    (pose.x, pose.y), arrowprops=dict(arrowstyle='->', color=CYAN, lw=2), zorder=21)
        ax.set(xlim=(pose.x-self._zoom_m, pose.x+self._zoom_m),
               ylim=(pose.y-self._zoom_m, pose.y+self._zoom_m), xlabel='World X (m)', ylabel='World Y (m)')
        ax.set_title(f'LOCAL TERRAIN  /  {2*self._zoom_m:.0f} m view', loc='left', color=INK, fontsize=11, pad=12)
        self._draw_legend(view, layer_legend)
        self._draw_profile(state)
        self._draw_drive(state)
        self._draw_status(state)
        seconds = state.get('sim_time')
        self._headline.set_text(f"{state.get('mission', 'navigation').upper()}   /   "
                                +(f'{seconds:.1f} s' if seconds is not None else 'LIVE'))
        self._fig.canvas.draw_idle()

    def _draw_legend(self, view, layer_legend=None):
        from matplotlib.lines import Line2D
        from matplotlib.colors import LogNorm
        from matplotlib.cm import ScalarMappable
        from matplotlib.patches import Patch
        ax = self._ax_legend
        ax.clear(); ax.axis('off')
        handles = [Line2D([], [], color=CYAN, lw=2, label='Selected path'),
                   Line2D([], [], color='white', lw=2, label='Actual trail'),
                   Patch(color=(.30, .31, .33), label='Planning mask'),
                   Patch(color=(.12, .13, .15), label='Unknown'),
                   Line2D([], [], color='#72f23b', marker='*', ls='', label='Goal / checkpoint'),
                   Line2D([], [], color=CYAN, marker='o', ls='', label='Robot / heading')]
        if layer_legend:
            handles = handles[:2]+handles[-2:]
            ax.text(0, .38, '\n'.join(textwrap.wrap(layer_legend, width=66)),
                    transform=ax.transAxes, color=MUTED, fontsize=7, va='top')
        ax.legend(handles=handles, loc='upper left', ncol=2, frameon=False,
                  labelcolor=MUTED, fontsize=8, labelspacing=.8, borderaxespad=0)
        key = None if view is None or self._layer != 'terrain' else (view['vmin'], view['vmax'], view['label'])
        if key != getattr(self, '_colorbar_key', None):
            self._ax_colorbar.clear()
            if key is not None:
                self._colorbar = self._fig.colorbar(ScalarMappable(norm=LogNorm(*key[:2]), cmap='magma'),
                                                  cax=self._ax_colorbar, orientation='horizontal')
                self._colorbar.set_label(key[2]+'  /  logarithmic scale', color=MUTED, fontsize=8)
                self._colorbar.ax.tick_params(colors=MUTED, labelsize=7)
            self._colorbar_key = key
        self._ax_colorbar.set_visible(key is not None)

    def _draw_profile(self, state):
        ax, slope = self._ax_profile, self._ax_slope
        ax.clear(); slope.clear(); self._style(ax); self._style(slope)
        ax.set_title('CLIMB PROFILE', loc='left', color=INK, fontsize=11, pad=24)
        rows = np.asarray(state.get('travel_profile', []), dtype=float).reshape(-1, 3)
        if len(rows):
            d, z, grade = rows.T
            ax.plot(d, z, color=CYAN, lw=2.2)
            ax.fill_between(d, 0, z, color=CYAN, alpha=.08)
            slope.plot(d, grade, color=GOLD, lw=1.4, alpha=.9)
            ax.scatter(d[-1], z[-1], c=CYAN, s=28, zorder=5)
            ax.set_xlim(0, max(2., d[-1]*1.03))
            low, high = min(0., float(z.min())), max(.3, float(z.max()))
            pad = max(.15, (high-low)*.15)
            ax.set_ylim(low-pad, high+pad)
            finite = grade[np.isfinite(grade)]
            bound = max(20., math.ceil(float(np.abs(finite).max())/10)*10) if len(finite) else 20.
            slope.set_ylim(-bound, bound)
            ax.text(0, 1.035, f'{d[-1]:.2f} m traveled   ·   {z[-1]:+.2f} m gain',
                    transform=ax.transAxes, color=MUTED, fontsize=9)
        else:
            ax.text(.5, .5, 'Waiting for motion telemetry', transform=ax.transAxes,
                    ha='center', color=MUTED)
        ax.set_xlabel('Traveled horizontal distance (m)', fontsize=9)
        ax.set_ylabel('Body elevation gain (m)', color=CYAN, fontsize=9)
        slope.set_ylabel('Terrain grade along heading (°)', color=GOLD, fontsize=9)
        slope.yaxis.set_label_position('right')
        slope.yaxis.tick_right()
        ax.tick_params(axis='y', colors=CYAN); slope.tick_params(axis='y', colors=GOLD)
        ax.grid(color='#384155', alpha=.35, lw=.6)
        slope.axhline(0, color=GOLD, alpha=.2, lw=.6)

    def _draw_drive(self, state):
        from matplotlib.patches import Wedge, Circle, Polygon
        from .dashboard_telemetry import drive_activity
        ax = self._ax_drive
        ax.clear(); ax.set_aspect('equal'); ax.axis('off')
        ax.set(xlim=(-1.36, 1.36), ylim=(-1.24, 1.22))
        ax.set_title('DRIVE / RESPONSE', color=INK, fontsize=11, pad=12)
        cmd = state.get('cmd', (0., 0., 0.)); speed = state.get('speed', float('nan'))
        yaw_rate = state.get('actual_yaw_rate', float('nan'))
        active = drive_activity(cmd, speed, yaw_rate, state.get('recovery_status', ''))
        sectors = [('FORWARD', 90, CYAN, f'{max(0., cmd[0]):+.2f} m/s'),
                   ('LEFT', 150, '#b7a0ff', f'{max(0., cmd[2]):+.2f} rad/s'),
                   ('BRAKING', 210, GOLD, 'zero input'),
                   ('REVERSE', 270, '#ff8397', f'{min(0., cmd[0]):+.2f} m/s'),
                   ('STOP', 330, '#72f2b6', 'settled'),
                   ('RIGHT', 30, '#b7a0ff', f'{min(0., cmd[2]):+.2f} rad/s')]
        ax.add_patch(Circle((0, 0), .68, facecolor=PANEL, edgecolor='#283248', lw=1))
        for label, angle, color, value in sectors:
            lit = label in active
            ax.add_patch(Wedge((0, 0), 1.13, angle-28, angle+28, width=.36,
                              facecolor=color if lit else '#202739', alpha=.95 if lit else .7))
            x, y = .95*np.cos(np.radians(angle)), .95*np.sin(np.radians(angle))
            ax.text(x, y+.04, label, ha='center', va='center', fontsize=7.2, weight='bold',
                    color=BACKGROUND if lit else MUTED)
            ax.text(x, y-.065, value, ha='center', va='center', fontsize=6.5,
                    color=BACKGROUND if lit else '#748099')
        title = '\n'.join(sorted(active))
        ax.add_patch(Polygon([(-.07, .42), (0, .53), (.07, .42)], color=CYAN))
        ax.text(0, .21, title, ha='center', va='center', color=INK, fontsize=10, weight='bold')
        ax.text(0, -.10, f'{speed:.2f}', ha='center', color=CYAN, fontsize=25, weight='bold')
        ax.text(0, -.25, 'm/s measured', ha='center', color=MUTED, fontsize=8)
        yaw_text = f'{yaw_rate:+.2f} rad/s' if math.isfinite(yaw_rate) else 'unavailable'
        ax.text(0, -.43, 'Yaw '+yaw_text, ha='center', color=MUTED, fontsize=8)
        ax.text(.5, -.045, f'COMMAND  x {cmd[0]:+.2f}   y {cmd[1]:+.2f} m/s   yaw {cmd[2]:+.2f} rad/s',
                transform=ax.transAxes, ha='center', color=MUTED, fontsize=8)

    def _draw_status(self, state):
        ax = self._ax_status
        ax.clear(); ax.axis('off')
        label = state.get('recovery_status', 'WAITING')
        ax.text(0, 1., label, transform=ax.transAxes, color=INK, fontsize=11, weight='bold', va='top')
        reason = state.get('nav_reason', '')
        width = max(28, int(ax.get_window_extent().width/6.3))
        wrapped = textwrap.wrap(reason, width=width)
        if len(wrapped) > 3: wrapped = wrapped[:2]+[wrapped[2][:max(1, width-1)]+'…']
        detail = '\n'.join(wrapped)
        detail += f"\nPolicy: {state.get('policy_status', 'fixed')}  |  mapped {state.get('mapped_coverage', 0.):.0%}"
        ax.text(0, .74, detail.strip(), transform=ax.transAxes, color=MUTED,
                fontsize=9, va='top', linespacing=1.45, clip_on=True)
