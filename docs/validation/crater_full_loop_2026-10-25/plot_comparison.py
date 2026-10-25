"""Offline route comparison using one frozen map and one cost model."""
import importlib
import json
import sys
import types
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[3]
directory = Path(__file__).parent
data = json.loads((directory/'route_comparison.json').read_text())
pkg = types.ModuleType('navplot')
pkg.__path__ = [str(root/'source/rexmi_rl')]
sys.modules['navplot'] = pkg
snapshot = importlib.import_module('navplot.nav.planning_snapshot')
slope = importlib.import_module('navplot.nav.slope_cost')
g, _ = snapshot.load(data['map_source'])
model = slope.Model(g, slope.Parameters())
sys.path.insert(0, str(root/'scripts'))
from slope_cost_plot import draw

plt.style.use('dark_background')
fig = plt.figure(figsize=(15, 8), facecolor='#0c0f19')
gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1])
ax = fig.add_subplot(gs[:, 0])
bar = draw(ax, model)
elevation = fig.add_subplot(gs[0, 1])
grade = fig.add_subplot(gs[1, 1])
for key, color, label in [('selected', '#00d9ff', 'New live-map best route'),
                           ('historical_blue', '#72f23b', 'Earlier successful route')]:
    route = np.asarray(data[key]['route'])
    ax.plot(*route.T, color=color, lw=2, label=label)
    ax.scatter(*route[-1], marker='*', color=color, s=85)
    distances, heights, grades = [], [], []
    traveled = 0.
    for a, b in zip(route, route[1:]):
        delta = b-a
        length = np.linalg.norm(delta)
        axis = delta/length
        for fraction in np.linspace(0., 1., max(2, int(np.ceil(length/.05))), endpoint=False):
            cell = g.world_to_cell(*(a+fraction*delta))
            distances.append(traveled+fraction*length)
            heights.append(g.surface[cell])
            grades.append(np.degrees(np.arctan(model.gx[cell]*axis[0]+model.gy[cell]*axis[1])))
        traveled += length
    elevation.plot(distances, heights, color=color, lw=1.4, label=f'{label}: cost {data[key]["score"]:.1f}')
    grade.plot(distances, grades, color=color, lw=1.1)
ax.set(xlim=(-3, 14), ylim=(-5, 12), xlabel='World X (m)', ylabel='World Y (m)',
       title='Same retained map · no prescribed exit direction')
ax.legend(fontsize=8, loc='upper left')
cb = fig.colorbar(bar, ax=ax, orientation='horizontal', fraction=.04, pad=.09)
cb.set_label('Exponential cost · log scale · grey: blocked', fontsize=8)
elevation.set(xlabel='Planned distance (m)', ylabel='Mapped terrain elevation (m)', title='Elevation along each planned route')
elevation.legend(fontsize=8)
grade.set(xlabel='Planned distance (m)', ylabel='Signed wheelbase grade (°)', title='Uphill grade along each planned route')
grade.axhline(15., color='#ffbc76', ls=':', lw=1., label='15° exponential knee')
grade.legend(fontsize=8)
for panel in (elevation, grade):
    panel.grid(alpha=.15)
fig.suptitle('ASCENT PLAN COMPARISON · live observations after northern descent')
fig.tight_layout()
fig.savefig(directory/'route_comparison.png', dpi=150)
plt.close(fig)
