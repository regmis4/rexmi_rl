"""Recorded decision replay: saved 67.8 map, 67.92 observation; no physics.

Restore saved derived fields verbatim. Do not rebuild or retrospectively repair
the archived low/high intervals. The map is 0.12 s older than the observation.
"""
import csv
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tests'))
import test_remote_control as t


def fixture():
    run = ROOT / 'outputs/remote_autonomy/height-retaining-ascent-JWGRFI3h'
    prior_path = ROOT / 'outputs/ascent_full_survey/trial01/run.prior.npz'
    with np.load(run / 'run.map.npz') as d:
        g = t.bt.a.AscentTerrainMap(
            world_size=d['height'].shape[0] * float(d['cell_size']),
            cell_size=float(d['cell_size']), origin=tuple(d['origin']),
            footprint_radius=float(d['footprint_radius']),
            half_length=float(d['half_length']), half_width=float(d['half_width']))
        prior = importlib.import_module('rexmi_bt_tests.nav.terrain_prior')
        prior.load_prior(g, prior_path,
            task='RexmiRl-Go2w-Crater-Bowl-RockySlope-Play-v0', seed=42)
        c = t.r.RemoteAscentController(g, [t.mission.Waypoint(0., 13., .6, 'exterior')],
            mode='characterize', uphill_limit=33., side_limit=32.)
        for name in d.files:
            field = getattr(g, name, None)
            if isinstance(field, np.ndarray) and field.shape == d[name].shape:
                setattr(g, name, d[name].copy())
        for src, dst in [('bin_low', '_bin_low'), ('bin_high', '_bin_high'),
                         ('bin_count', '_bin_count')]:
            setattr(g, dst, d[src].copy())
        g.now = float(d['sim_time'])
    g.fresh = g.known & (g.now - g.last_seen <= g.FRESH_SECONDS)
    g.no_go_depth = ndimage.distance_transform_edt(g.no_go, sampling=g.cell_size)
    g.failure_records = json.loads((run / 'run.summary.json').read_text())['failed_climbs']
    failure = g.failure_records[-1]
    c.recovery_permission = t.r.RecoveryPermission.capture(
        g, failure['position'], failure['time'])
    rows = list(csv.DictReader((run / 'run.csv').open()))
    for q in rows:
        if float(q['sim_time']) >= failure['time']:
            c.recovery_permission = c.recovery_permission.observe(
                g, (float(q['x']), float(q['y'])))
    q = rows[-1]
    c.obs = t.bt.observation(float(q['sim_time']),
        xy=tuple(float(q[k]) for k in ('x', 'y')), yaw=float(q['yaw']),
        velocity=tuple(float(q[k]) for k in ('velocity_world_x', 'velocity_world_y')),
        yaw_rate=float(q['actual_yaw_rate']))
    c.recovery_episode = failure['time']
    c.direction = -1
    c.button_state = 'OBSERVE'
    return c, rows, failure


if __name__ == '__main__':
    c, rows, failure = fixture()
    reverse, zero = t.r.Motion(-.4, 0.), t.r.Motion()
    result = dict(scope=__doc__, map_time=c.grid.now, observation_time=c.obs.time,
        speed=c.obs.speed, patches=len(c.recovery_permission.patches),
        reverse_command=c._command_geometry(reverse, escape=True),
        reverse_full=c._geometry(reverse, escape=True),
        reverse_forecast=c._anticipation(reverse, escape=True),
        zero_without_context=c._geometry(zero),
        actual_with_checked_signed_context=c._actual_geometry(
            reverse, escape=True, checked_intent=True))
    inputs = [ROOT / 'outputs/ascent_full_survey/trial01/run.prior.npz']
    inputs += [ROOT / 'outputs/remote_autonomy/height-retaining-ascent-JWGRFI3h' / name
               for name in ('run.map.npz', 'run.csv', 'run.summary.json')]
    result['input_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in inputs}
    out = Path(__file__).with_suffix('.json')
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
