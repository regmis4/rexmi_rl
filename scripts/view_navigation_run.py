#!/usr/bin/env python3
"""View saved navigation telemetry without starting Isaac Sim."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='Directory containing run.csv and run.map.npz')
    parser.add_argument('--save', type=Path, help='Save a static overview instead of opening a window')
    args = parser.parse_args()
    matplotlib.use('Agg' if args.save else 'TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider

    with (args.run / 'run.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    keys = ('sim_time', 'x', 'y', 'z', 'yaw', 'speed', 'vx_cmd', 'omega_cmd')
    data = {key: np.array([float(row[key]) for row in rows]) for key in keys}
    t = data['sim_time']
    summary = json.loads((args.run / 'run.summary.json').read_text())
    with np.load(args.run / 'run.map.npz') as grid:
        cost = np.ma.masked_where(~grid['known'], grid['cost']).T
        origin = grid['origin']
        size = float(grid['cell_size'])
        half_x, half_y = cost.shape[1] * size / 2, cost.shape[0] * size / 2
        extent = (origin[0] - half_x, origin[0] + half_x,
                  origin[1] - half_y, origin[1] + half_y)
        fig = plt.figure(figsize=(14, 8))
        layout = fig.add_gridspec(3, 2, left=.07, right=.96, top=.87, bottom=.17)
        map_ax = fig.add_subplot(layout[:, 0])
        chart = map_ax.imshow(cost, origin='lower', extent=extent, cmap='viridis', vmin=0,
                              vmax=max(1., float(np.percentile(cost.compressed(), 95))))
        fig.colorbar(chart, ax=map_ax, label='Terrain cost (final retained map)', shrink=.65)
        map_ax.plot(data['x'], data['y'], color='white', alpha=.65, linewidth=1)
        trail, = map_ax.plot([], [], color='cyan', linewidth=2)
        robot, = map_ax.plot([], [], 'ro', markersize=7)
        heading, = map_ax.plot([], [], 'r-', linewidth=2)
        for milestone in summary.get('reviewed_buttons', {}).get('milestones', []):
            x, y = milestone['position'][:2]
            map_ax.plot(x, y, 'wx')
            map_ax.annotate(milestone['name'], (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
        map_ax.set(xlabel='World x (m)', ylabel='World y (m)', title='Recorded route · final map, not evolving LiDAR')
        map_ax.set_xlim(min(data['x']) - 5, max(data['x']) + 3)
        map_ax.set_ylim(min(data['y']) - 3, max(data['y']) + 5)
        cursors = []
        for row, series, label in ((0, ('z',), 'Height (m)'),
                                   (1, ('speed', 'vx_cmd'), 'Speed / command (m/s)'),
                                   (2, ('omega_cmd',), 'Yaw command (rad/s)')):
            axis = fig.add_subplot(layout[row, 1])
            for key in series:
                axis.plot(t, data[key], label=key, linewidth=1)
            axis.set(xlim=(t[0], t[-1]), ylabel=label)
            axis.grid(alpha=.25)
            axis.legend(loc='upper left', fontsize=8)
            cursors.append(axis.axvline(t[0], color='red', alpha=.7))
        axis.set_xlabel('Simulated time (s)')
        fig.suptitle(f'{args.run.name} — saved telemetry; not a simulation replay', fontsize=14)
        status = fig.text(.07, .91, '', fontsize=10)
        slider = Slider(fig.add_axes((.24, .065, .61, .03)), 'Time (s)', t[0], t[-1], valinit=t[0])
        button = Button(fig.add_axes((.07, .055, .075, .05)), 'Play')
        playing = [False]

        def update(value):
            index = min(int(np.searchsorted(t, value)), len(t) - 1)
            x, y, yaw = (data[key][index] for key in ('x', 'y', 'yaw'))
            trail.set_data(data['x'][:index + 1], data['y'][:index + 1])
            robot.set_data([x], [y])
            heading.set_data([x, x + .7 * np.cos(yaw)], [y, y + .7 * np.sin(yaw)])
            for cursor in cursors:
                cursor.set_xdata([t[index], t[index]])
            status.set_text(f't={t[index]:.2f}s  |  {rows[index]["reason"]}  |  '
                            f'vx={data["vx_cmd"][index]:+.2f}, yaw={data["omega_cmd"][index]:+.2f}')
            fig.canvas.draw_idle()

        def toggle(_event):
            playing[0] = not playing[0]
            button.label.set_text('Pause' if playing[0] else 'Play')

        def tick():
            if playing[0]:
                slider.set_val(min(slider.val + .1, t[-1]))
                if slider.val >= t[-1]:
                    toggle(None)

        slider.on_changed(update)
        button.on_clicked(toggle)
        timer = fig.canvas.new_timer(interval=100)
        timer.add_callback(tick)
        timer.start()
        update(t[0])
        if args.save:
            slider.set_val(t[-1])
            fig.savefig(args.save, dpi=140)
            print(args.save)
        else:
            plt.show()


if __name__ == '__main__':
    main()
