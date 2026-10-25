"""Read-only vector posture analysis of a recorded navigation run."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def analyze(path):
    rows = pd.read_csv(path)
    w, x, y, z = (rows['body_quaternion_'+c].to_numpy() for c in 'wxyz')
    body_up = np.column_stack((2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)))
    normals = rows[['observed_normal_'+c for c in 'xyz']].to_numpy()
    angular = rows[['angular_velocity_world_'+c for c in 'xyz']].to_numpy()
    up_rate = np.cross(angular, body_up)
    relative_cosine = np.clip((body_up*normals).sum(axis=1), -1., 1.)
    relative = np.arccos(relative_cosine)
    absolute = np.arccos(np.clip(body_up[:, 2], -1., 1.))
    # Angular velocity rotates body up: du/dt = omega x u. The fitted normal
    # is held fixed in this instantaneous rate, excluding map-fit variation.
    relative_rate = np.divide(-(up_rate*normals).sum(axis=1), np.sin(relative),
        out=np.full(len(rows), np.nan), where=np.sin(relative)>1e-5)
    absolute_rate = np.divide(-up_rate[:, 2], np.sin(absolute),
        out=np.full(len(rows), np.nan), where=np.sin(absolute)>1e-5)
    rows['body_tilt_from_normal_deg'] = np.degrees(relative)
    rows['outward_relative_rate_deg_s'] = np.degrees(relative_rate)
    rows['outward_absolute_rate_deg_s'] = np.degrees(absolute_rate)
    events_path = path.with_suffix('.events.jsonl')
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    selected = []
    for event in events:
        if (event.get('event')=='posture_entry_signal'
                or (event.get('event')=='movement_transition' and event.get('state')=='POSTURE_PAUSE')):
            index = (rows.sim_time-float(event['time'])).abs().idxmin()
            row = rows.loc[index]
            if abs(row.sim_time-event['time'])>1e-7:
                raise ValueError('Posture event has no same-time observation')
            selected.append(dict(event=event, **{key:float(row[key]) for key in
                ('body_tilt_from_normal_deg','outward_relative_rate_deg_s',
                 'outward_absolute_rate_deg_s','speed','body_tilt_deg')}))
    actions = []
    for action, group in rows[rows.action_id.notna()].groupby('action_id',sort=False):
        actions.append(dict(action_id=action,first_s=float(group.sim_time.iloc[0]),
            last_s=float(group.sim_time.iloc[-1]),peak_absolute_tilt_deg=float(group.body_tilt_deg.max()),
            peak_relative_tilt_deg=float(group.body_tilt_from_normal_deg.max()),
            peak_outward_absolute_rate_deg_s=float(group.outward_absolute_rate_deg_s.max()),
            peak_outward_relative_rate_deg_s=float(group.outward_relative_rate_deg_s.max()),
            peak_planar_speed_mps=float(group.speed.max())))
    result = dict(source=str(path.resolve()),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        events_sha256=hashlib.sha256(events_path.read_bytes()).hexdigest(),
        limitation='Observed fitted normal is not a contact support plane. Instantaneous relative angular rate freezes that normal. Observation associations do not establish causation or a safe stability envelope.',
        posture_events=selected,actions=actions)
    def finite(value):
        if isinstance(value,dict):return {key:finite(v) for key,v in value.items()}
        if isinstance(value,list):return [finite(v) for v in value]
        if isinstance(value,float) and not np.isfinite(value):return None
        return value
    path.with_suffix('.posture.json').write_text(json.dumps(finite(result),indent=2,allow_nan=False)+'\n')
    rows.to_csv(path.with_suffix('.posture.csv'),index=False)
    print(json.dumps(dict(events=len(selected),actions=len(actions),output=str(path.with_suffix('.posture.json')))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv',type=Path)
    analyze(parser.parse_args().csv)
