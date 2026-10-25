#!/usr/bin/env python3
"""Closed-loop flat-handoff checks using actual sensor maps and frozen policies."""
import argparse
import importlib
import json
import math
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
sys.path.insert(0,os.environ.get('ISAACLAB_DIR',os.path.expanduser('~/IsaacLab')))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--rough',required=True);p.add_argument('--rocky',required=True)
p.add_argument('--no_switches',action='store_true');p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=42)
a=p.parse_args();app=AppLauncher(headless=True).app
try:
    import numpy as np
    import torch
    import gymnasium as gym
    import rexmi_rl
    from isaaclab.terrains import TerrainGeneratorCfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rexmi_rl.tasks.locomotion.velocity.config.go2w.crater_terrain import RockyPyramidSlopeCfg
    from rexmi_rl.drive_adapter import load_policy,inject_command
    from rexmi_rl.nav.observed_map import ObservedTerrainMap
    from rexmi_rl.nav.checkpoint_control import CheckpointController
    from rexmi_rl.nav.flat_sidestep import FlatSidestep
    from types import SimpleNamespace
    task='RexmiRl-Go2w-Crater-Bowl-RockySlope-Play-v0'
    spec=gym.spec(task)
    def config(key):
        mod,cls=spec.kwargs[key].rsplit(':',1)
        return getattr(importlib.import_module(mod),cls)()
    cfg,agent=config('env_cfg_entry_point'),config('rsl_rl_cfg_entry_point')
    cfg.seed=a.seed;cfg.scene.num_envs=4
    def terrain(slope,rough):
        return RockyPyramidSlopeCfg(proportion=.25,slope_min_deg=slope,slope_max_deg=slope,
            roughness_min_m=rough,roughness_max_m=rough,boulder_count_min=0,boulder_count_max=0,
            platform_width=1.,seed=a.seed)
    cfg.scene.terrain.terrain_generator=TerrainGeneratorCfg(seed=a.seed,curriculum=True,
        size=(24.,24.),num_rows=1,num_cols=4,sub_terrains={
            'flat_left':terrain(0.,0.),'flat_right':terrain(0.,0.),
            'textured':terrain(0.,.04),'slope_guard':terrain(10.,.04)},
        horizontal_scale=.1,vertical_scale=.005,slope_threshold=None,use_cache=False)
    cfg.scene.terrain.max_init_terrain_level=0
    for key in vars(cfg.terminations):
        if not key.startswith('_'):setattr(cfg.terminations,key,None)
    for key in ('add_base_mass','push_robot','base_external_force_torque'):
        if hasattr(cfg.events,key):setattr(cfg.events,key,None)
    cfg.curriculum.terrain_levels=None
    cfg.commands.base_velocity.heading_command=False
    cfg.commands.base_velocity.rel_heading_envs=0.
    cfg.commands.base_velocity.resampling_time_range=(1e9,1e9)
    cfg.commands.base_velocity.debug_vis=False
    cfg.observations.policy.enable_corruption=False
    cfg.events.reset_base.params={'pose_range':{'x':(0.,0.),'y':(0.,0.),'z':(.45,.45),'yaw':(0.,0.)},
        'velocity_range':{k:(0.,0.) for k in ('x','y','z','roll','pitch','yaw')}}
    env=RslRlVecEnvWrapper(gym.make(task,cfg=cfg),clip_actions=agent.clip_actions)
    policies={'rough':load_policy(a.rough,cfg.sim.device),'rocky_slope':load_policy(a.rocky,cfg.sim.device)}
    env.reset();robot=env.unwrapped.scene['robot']
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(),torch.zeros_like(robot.data.joint_vel))
    materials=robot.root_physx_view.get_material_properties()
    materials[:,:,0]=.8;materials[:,:,1]=.6;materials[:,:,2]=0.
    robot.root_physx_view.set_material_properties(materials,torch.arange(4,dtype=torch.int32,device='cpu'))
    types=env.unwrapped.scene.terrain.terrain_types.cpu().numpy()
    origins=env.unwrapped.scene.env_origins.detach().cpu().numpy()
    grids=[];controls=[];sides=[];records=[[] for _ in range(4)];chosen=[False]*4
    names=['flat_left','flat_right','textured','slope_guard']
    radius=max(.45,float(torch.linalg.vector_norm(robot.data.body_pos_w[0,:,:2]-robot.data.root_pos_w[0,:2],dim=-1).max())+.1)
    for i in range(4):
        g=ObservedTerrainMap(world_size=32.,origin=tuple(origins[i,:2]),footprint_radius=radius)
        sign=-1. if types[i]==1 else 1.
        c=CheckpointController(g,[SimpleNamespace(x=origins[i,0],y=origins[i,1]+sign,label='test')])
        c.checkpoint_mode(True,0.)
        grids.append(g);controls.append(c);sides.append(FlatSidestep(g))
    dt=env.unwrapped.step_dt;rows=[];failed=np.zeros(4,bool)
    for step in range(round(25/dt)):
        t=step*dt
        pos=robot.data.root_pos_w.detach().cpu().numpy()
        vel=robot.data.root_lin_vel_w.detach().cpu().numpy()
        angular=robot.data.root_ang_vel_b.detach().cpu().numpy()
        q=robot.data.root_quat_w.detach().cpu().numpy()
        up=1-2*(q[:,1]**2+q[:,2]**2)
        yaw=np.arctan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2))
        failed|=up<.5
        commands=[]
        for i,(g,c,side) in enumerate(zip(grids,controls,sides)):
            xy=pos[i,:2];v=vel[i,:2];speed=float(np.linalg.norm(v))
            if step%10==0:
                for name,limit in [('lidar',15.),('height_scanner',25.),('forward_scanner',5.)]:
                    sensor=env.unwrapped.scene[name].data
                    g.ingest(sensor.ray_hits_w[i].detach().cpu().numpy(),sensor.pos_w[i].detach().cpu().numpy(),t,max_range=limit)
                g.rebuild(t)
            if t>=2. and not chosen[i]:
                wp=c.waypoints[0];chosen[i]=c.choose_checkpoint(xy,(wp.x,wp.y),t)
            if side.active:
                c.velocity_world=v
                command=side.tick(c,xy,float(yaw[i]),v,float(angular[i,2]),float(np.linalg.norm(angular[i])),t,float(up[i]),vertical_speed=float(vel[i,2]))
                c.progress_at=t;c.last_tick=t;c.drive_samples=[]
            else:
                command=c.tick(xy,float(yaw[i]),speed,t,upright=float(up[i]),velocity=vel[i])
                if not a.no_switches and side.maybe_start(c,xy,float(yaw[i]),t,float(up[i])):
                    command=side.tick(c,xy,float(yaw[i]),v,float(angular[i,2]),float(np.linalg.norm(angular[i])),t,float(up[i]),vertical_speed=float(vel[i,2]))
            if failed[i]:side.cancel('test tip');c.request('stop',t)
            vx,vy,w=(0.,0.,0.) if failed[i] else (command.vx,getattr(command,'vy',0.),command.omega)
            inject_command(env,vx,w,i,vy=vy)
            records[i].append(dict(time=t,position=(pos[i]-origins[i]).tolist(),speed=speed,
                policy=side.policy,phase=side.phase,reason=side.reason,state=c.state,command=[vx,vy,w],
                tilt_deg=math.degrees(math.acos(float(np.clip(up[i],-1,1))))))
        with torch.inference_mode():
            obs=env.unwrapped.observation_manager.compute(update_history=False)['policy']
            actions=torch.zeros((4,16),device=cfg.sim.device)
            for key,policy in policies.items():
                ids=[i for i,side in enumerate(sides) if side.policy==key]
                if ids:actions[ids]=policy(obs[ids])
            _,_,done,_=env.step(actions)
        if bool(done.any()):raise RuntimeError('Unexpected automatic reset')
    summary=[]
    for i,(c,side) in enumerate(zip(controls,sides)):
        summary.append(dict(case=names[types[i]],goal_accepted=chosen[i],tip=bool(failed[i]),
            switches=side.switches,events=side.events,final_policy=side.policy,phase=side.phase,
            reason=side.reason,controller_state=c.state,final_position=records[i][-1]['position'],
            max_tilt_deg=max(r['tilt_deg'] for r in records[i]),max_speed=max(r['speed'] for r in records[i])))
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(cases=summary,samples=records),indent=2))
    out.with_suffix('.summary.json').write_text(json.dumps(dict(cases=summary,seed=a.seed,automatic_resets=0),indent=2))
    print('[flat-switch-study] '+json.dumps(summary),flush=True)
except Exception:
    import traceback
    traceback.print_exc()
    raise
finally:
    if 'env' in globals():env.close()
    app.close()
