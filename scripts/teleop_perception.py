"""Read-only reference-map display and comparison logging for manual teleop.

No controller, planning, command filtering, map rebuilding or sensor fusion.
Live LiDAR is visualized separately from the archived terrain-cost reference.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import threading
from types import SimpleNamespace

import numpy as np


def validate_reference(meta, task, seed):
    prior=meta['config'].get('terrain_prior') or {}
    if (prior.get('task')!=task or prior.get('seed')!=seed
            or prior.get('coordinate_frame')!='world'):
        raise ValueError('Display snapshot must match teleop task, seed and world frame')


def terrain_reading(grid, xy, yaw):
    cell=grid.world_to_cell(*xy)
    if cell is None or not grid.known[cell]:return dict(known=False)
    def number(name,scale=1.):
        value=float(getattr(grid,name)[cell])*scale
        return value if math.isfinite(value) else None
    gx,gy=number('gx'),number('gy')
    slope=number('slope')
    return dict(known=True,blocked=bool(grid.blocked[cell]),cost=number('cost'),
        slope_deg=math.degrees(math.atan(slope)) if slope is not None else None,
        along_deg=math.degrees(math.atan(gx*math.cos(yaw)+gy*math.sin(yaw))) if gx is not None and gy is not None else None,
        cross_deg=math.degrees(math.atan(abs(-gx*math.sin(yaw)+gy*math.cos(yaw)))) if gx is not None and gy is not None else None,
        roughness_cm=number('roughness',100),footprint_rms_cm=number('footprint_rms',100),
        step_cm=number('footprint_step',100))


class TeleopPerception:
    def __init__(self, env, args, state):
        from rexmi_rl.nav.planning_snapshot import load, ARRAYS
        from rexmi_rl.nav.localizer import SimLocalizer
        from rexmi_rl.nav.perception_view import PerceptionView
        from matplotlib import colormaps
        self.grid,meta=load(args.display_map)
        validate_reference(meta,args.task,args.seed)
        for name in ARRAYS:getattr(self.grid,name).setflags(write=False)
        self.localizer=SimLocalizer(env.scene['robot'],0)
        self.state=state;self.dt=float(env.step_dt);self.episode=0;self.trail=[];self.last=None
        self.directory=Path(args.joint_log_dir).parent
        self.palette=colormaps['plasma']
        source=Path(args.display_map).resolve()
        with source.open('rb') as src, (self.directory/'display_map.npz').open('xb') as dst:
            shutil.copyfileobj(src,dst)
        self.metadata=dict(task=args.task,seed=args.seed,source=str(source),
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            source_goal=meta['goal'],source_start=meta['position'],source_yaw=meta['yaw'],
            source_terrain=meta['config'].get('terrain_prior'),
            mode='manual; frozen reference costs, live LiDAR; no autonomous command filtering',
            palette='plasma 1..20, gray reference blocked cells',
            cost_semantics='Archived displayed cost at robot center; not full route objective or probability',
            comparison_limit='Task/seed match checked; identical geometry/startup/pose distribution not certified. Teleop startup force-spawns repeatedly; telemetry begins afterward.',
            capture_start_pose=vars(self.localizer.get_pose()),
            files=dict(physics='run/samples.csv',events='run/events.jsonl',commands='run.csv',map_samples='map_samples.csv'))
        repo=Path(__file__).resolve().parents[1]
        self.metadata['display_sources']={name:hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in
            ('scripts/teleop_perception.py','source/rexmi_rl/nav/perception_view.py',
             'source/rexmi_rl/nav/planning_snapshot.py','scripts/launch_manual_ascent_baseline.sh')}
        (self.directory/'manual_capture.json').write_text(json.dumps(self.metadata,indent=2)+'\n')
        self.stream=(self.directory/'map_samples.csv').open('x',newline='',buffering=1)
        self.fields=['step','sim_time_s','episode','x','y','z','yaw_rad','policy','vx','omega','scenario',
                     'known','blocked','cost','slope_deg','along_deg','cross_deg','roughness_cm','footprint_rms_cm','step_cm','reset']
        self.writer=csv.DictWriter(self.stream,fieldnames=self.fields);self.writer.writeheader()
        self.nav=SimpleNamespace(_sim_localizer=self.localizer,_lidar=env.scene['lidar'],_env_idx=0,
            _omap=self.grid,_lock=threading.Lock(),shared=dict(planned_path=[],policy_status=args.policy,
            switch_reason='MANUAL | frozen reference map'),manual_trace=True,terrain_colorizer=self.colors)
        try:self.view=PerceptionView(self.nav,hz=10.,point_limit=2000)
        except BaseException:
            self.stream.close();raise
        print('[teleop overlay] Frozen comparison costmap: purple 1 → yellow 20; gray exclusions. Green manual trail; live cyan LiDAR.')

    def colors(self,cost,observed):
        palette=self.palette(np.clip((cost-1.)/19.,0,1))[...,:3].astype(np.float32)
        palette[self.grid.blocked]=[.22,.24,.27]
        palette[~observed]=[.85,.85,.85]
        return palette,'FROZEN reference cost: purple 1 → yellow 20 | gray: planning mask | no command restrictions'

    def update(self,step,policy,vx,omega,done=False):
        pose=self.localizer.get_pose();xy=(pose.x,pose.y)
        reading=terrain_reading(self.grid,xy,pose.yaw)
        self.writer.writerow(dict(step=step,sim_time_s=(step+1)*self.dt,episode=self.episode,
            x=pose.x,y=pose.y,z=pose.z,yaw_rad=pose.yaw,policy=policy,vx=vx,omega=omega,
            scenario=self.state.scenario_label(),reset=done,**reading))
        if done:
            self.trail=[];self.episode+=1
        elif not self.trail or math.dist(xy,self.trail[-1])>=.08:
            self.trail.append(xy)
        self.last=dict(step=step,sim_time_s=(step+1)*self.dt,pose=vars(pose),policy=policy,reset=done)
        def value(key):
            v=reading.get(key);return 'unknown' if v is None else f'{v:.2f}'
        self.nav.shared.update(planned_path=self.trail,policy_status=policy,
            inspection_text=f"At robot center: cost {value('cost')} | slope {value('slope_deg')}°\n"
                f"Along heading {value('along_deg')}° | across {value('cross_deg')}°\n"
                f"Footprint RMS {value('footprint_rms_cm')} cm | step {value('step_cm')} cm\n"
                'Archived map held fixed; live LiDAR does not change these weights.')
        self.view.update()

    def close(self,reason):
        self.stream.close()
        self.metadata.update(end_reason=reason,last_sample=self.last,episodes=self.episode+1,
            overlay_enabled_at_end=self.view.enabled,
            candidate_for_comparison=bool(self.last and self.episode==0 and reason!='exception'),
            success='not inferred; inspect actual route and selected exterior endpoint')
        (self.directory/'manual_capture.json').write_text(json.dumps(self.metadata,indent=2,allow_nan=False)+'\n')
        self.view.close()
