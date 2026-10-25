"""Archive and plot a completed physical loop; never supplies navigation input."""
import csv
from collections import Counter
import hashlib
import importlib
import json
import shutil
import sys
import types
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[3]
run = Path(sys.argv[1]).resolve()
out = Path(__file__).parent / sys.argv[2]
out.mkdir(exist_ok=True)
rows = list(csv.DictReader((run / 'run.csv').open()))
summary_path = run / 'run.summary.json'
summary = json.loads(summary_path.read_text()) if summary_path.exists() else dict(
    result='interrupted; no final mission summary', counts_as_mission=False)
loop = json.loads((run / 'run.loop.json').read_text())
provenance = json.loads((run / 'run.provenance.json').read_text())
events = [json.loads(s) for s in (run / 'run.events.jsonl').read_text().splitlines()]
audit = dict(run=str(run.relative_to(root)), result=summary['result'],
    counts_as_mission=summary.get('counts_as_mission'), stage=loop['stage'],
    center_arrival_time=loop['center_arrival_time'],
    manual_intervention=loop['manual_intervention'],
    last_sim_time=float(rows[-1]['sim_time']),
    continuous_20ms_timestamps=bool(np.allclose(np.diff([float(r['sim_time']) for r in rows]), .02)),
    source_mismatches=[name for name, digest in provenance['sources'].items()
        if not (root/name).exists() or hashlib.sha256((root/name).read_bytes()).hexdigest() != digest],
    raw_csv_sha256=hashlib.sha256((run/'run.csv').read_bytes()).hexdigest(), phases={})
for phase in ('DESCEND_TO_CENTER', 'ASCEND_TO_EXTERIOR'):
    part = [r for r in rows if r['mission_stage'] == phase and float(r['sim_time']) >= 1.]
    if not part:
        continue
    xy = np.array([[float(r[k]) for k in ('x', 'y')] for r in part])
    audit['phases'][phase] = dict(first_time=float(part[0]['sim_time']), last_time=float(part[-1]['sim_time']),
        travel_m=float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()),
        peak_speed_mps=max(float(r['speed']) for r in part),
        minimum_up_z=min(float(r['up_z']) for r in part), final_position=xy[-1].tolist())
transitions = [e for e in events if e['event']=='remote_transition']
audit['immediate_turn_restarts'] = sum(a.get('previous')=='TURN' and a.get('state')=='DRIVE'
    and b.get('state')=='TURN' and a['time']==b['time'] for a,b in zip(transitions,transitions[1:]))
audit['ascent_motion_seconds'] = {k:v*.02 for k,v in Counter(r['motion_status'] for r in rows
    if r['mission_stage']=='ASCEND_TO_EXTERIOR').items()}
audit['ascent_recovery_starts'] = dict(Counter(e['action'] for e in events
    if e['event']=='feedback_recovery_start'))
for name in ('summary', 'loop', 'provenance'):
    if (run/f'run.{name}.json').exists():
        shutil.copy2(run/f'run.{name}.json', out/f'{name}.json')
fields = ('sim_time', 'x', 'y', 'z', 'yaw', 'speed', 'up_z', 'state', 'reason', 'vx_cmd', 'omega_cmd', 'mission_stage')
with (out/'trace_10hz.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows({k:r[k] for k in fields} for r in rows[::5])
(out/'events.json').write_text(json.dumps(events, indent=2)+'\n')
(out/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
print(json.dumps(audit, indent=2))
if not loop.get('plan'):
    raise SystemExit(0)
plan = loop['plan']
shutil.copy2(plan['map_path'], out/'ascent_input.npz')
shutil.copy2(plan['result_path'], out/'exit_plan.json')
shutil.copy2(Path(plan['result_path']).with_name('exit_route.csv'), out/'exit_route.csv')
pkg = types.ModuleType('navplot')
pkg.__path__ = [str(root/'source/rexmi_rl')]
sys.modules['navplot'] = pkg
snapshot = importlib.import_module('navplot.nav.planning_snapshot')
slope = importlib.import_module('navplot.nav.slope_cost')
display = importlib.import_module('navplot.nav.terrain_display')
sys.path.insert(0, str(root/'scripts'))
from slope_cost_plot import draw
g, _ = snapshot.load(out/'ascent_input.npz')
model = slope.Model(g, slope.Parameters(**plan['parameters']))
route = np.asarray(plan['routes']['any_exit'])
plt.style.use('dark_background')
fig = plt.figure(figsize=(14, 8), facecolor='#0c0f19')
gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1])
ax = fig.add_subplot(gs[:, 0])
bar = draw(ax, model)
ax.plot(*route.T, color='#00d9ff', lw=2.5, label='Validated blue ascent' if plan.get('original_route_preserved') else 'Best planned ascent')
for phase, color, label in [('DESCEND_TO_CENTER', '#72f23b', 'Measured descent'),
                            ('ASCEND_TO_EXTERIOR', 'white', 'Measured ascent')]:
    part = [r for r in rows[::5] if r['mission_stage']==phase and float(r['sim_time'])>=1.]
    if part:
        ax.plot(*np.array([[float(r[k]) for k in ('x','y')] for r in part]).T,
                color=color, lw=1.1, label=label)
ax.scatter(*route[-1], marker='*', s=100, color='#00d9ff', zorder=5)
ax.set(xlabel='World X (m)', ylabel='World Y (m)', title='Retained descent map · selected ascent')
ax.set_xlim(-5, 16)
ax.set_ylim(-7, 14)
ax.legend(fontsize=8, loc='lower left')
cb = fig.colorbar(bar, ax=ax, orientation='horizontal', fraction=.04, pad=.09)
cb.set_label('Exponential slope cost · logarithmic scale · grey: blocked', fontsize=8)
for index, phase in enumerate(('DESCEND_TO_CENTER', 'ASCEND_TO_EXTERIOR')):
    part = [r for r in rows[::5] if r['mission_stage']==phase and float(r['sim_time'])>=1.]
    if not part:
        continue
    x,y,z = np.array([[float(r[k]) for k in ('x','y','z')] for r in part]).T
    distance = np.r_[0,np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    grade = [display.heading_grade(g,(float(r['x']),float(r['y'])),float(r['yaw']),model) for r in part]
    panel = fig.add_subplot(gs[index,1])
    panel.plot(distance,z,color='#00d9ff',label='Measured robot elevation')
    panel.set(xlabel='Measured distance travelled (m)',ylabel='Robot elevation (m)',
              title='Descent' if index==0 else 'Ascent')
    twin = panel.twinx()
    twin.plot(distance,grade,color='#ffbc76',lw=.7,alpha=.65)
    twin.set_ylabel('Heading grade from retained map (°)',color='#ffbc76')
    panel.grid(alpha=.15)
fig.suptitle(f"AUTONOMOUS CRATER LOOP · {summary['result']} · {run.name}")
fig.tight_layout()
fig.savefig(out/'mission.png', dpi=150)
plt.close(fig)
