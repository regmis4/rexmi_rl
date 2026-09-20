#!/usr/bin/env python3
"""Matched rough/rocky-slope sidestep assessment. Does not enable nav switching."""
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
p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=42)
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
    task='RexmiRl-Go2w-Crater-Bowl-RockySlope-Play-v0'
    spec=gym.spec(task)
    def config(key):
        mod,cls=spec.kwargs[key].rsplit(':',1)
        return getattr(importlib.import_module(mod),cls)()
    cfg,agent=config('env_cfg_entry_point'),config('rsl_rl_cfg_entry_point')
    motions=[('left',0.,.2,0.),('right',0.,-.2,0.),('diagonal_left',.3,.2,0.),
             ('diagonal_right',.3,-.2,0.),('forward',.3,0.,0.),('yaw',0.,0.,.35)]
    cfg.seed=a.seed;cfg.scene.num_envs=24
    def terrain(rough):
        return RockyPyramidSlopeCfg(proportion=.5,slope_min_deg=0.,slope_max_deg=0.,
            roughness_min_m=rough,roughness_max_m=rough,boulder_count_min=0,boulder_count_max=0,
            platform_width=2.,seed=a.seed)
    cfg.scene.terrain.terrain_generator=TerrainGeneratorCfg(seed=a.seed,curriculum=True,
        size=(24.,24.),num_rows=1,num_cols=2,sub_terrains={'flat':terrain(0.),'textured':terrain(.04)},
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
    robot.root_physx_view.set_material_properties(materials,torch.arange(24,dtype=torch.int32,device='cpu'))
    types=env.unwrapped.scene.terrain.terrain_types.cpu().numpy()
    cases=[]
    for i in range(24):
        name,vx,vy,w=motions[i%6]
        cases.append(dict(env=i,policy='rough' if i%12<6 else 'rocky_slope',motion=name,
            command=[vx,vy,w],roughness_m=.04 if types[i] else 0.,seed=a.seed))
    indices={key:torch.tensor([i for i,c in enumerate(cases) if c['policy']==key],device=cfg.sim.device) for key in policies}
    dt=env.unwrapped.step_dt;rows=[];failed=np.zeros(24,bool)
    for step in range(round(12/dt)):
        t=step*dt
        pos=(robot.data.root_pos_w-env.unwrapped.scene.env_origins).detach().cpu().numpy()
        vel=robot.data.root_lin_vel_b.detach().cpu().numpy()
        q=robot.data.root_quat_w.detach().cpu().numpy()
        up=1-2*(q[:,1]**2+q[:,2]**2)
        yaw=np.arctan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2))
        failed|=up<.5
        rows.append(dict(time=t,position=pos.tolist(),velocity=vel.tolist(),yaw=yaw.tolist(),up=up.tolist()))
        for i,c in enumerate(cases):
            vx,vy,w=c['command'] if 2<=t<8 and not failed[i] else (0.,0.,0.)
            inject_command(env,vx,w,i,vy=vy)
        with torch.inference_mode():
            obs=env.unwrapped.observation_manager.compute(update_history=False)['policy']
            actions=torch.zeros((24,16),device=cfg.sim.device)
            for key,policy in policies.items():actions[indices[key]]=policy(obs[indices[key]])
            _,_,done,_=env.step(actions)
        if bool(done.any()):raise RuntimeError('Unexpected automatic reset')
    for i,c in enumerate(cases):
        drive=[r for r in rows if 2<=r['time']<8];stop=[r for r in rows if r['time']>=8]
        delta=np.array(drive[-1]['position'][i][:2])-drive[0]['position'][i][:2]
        yaw0=drive[0]['yaw'][i]
        body_delta=[math.cos(yaw0)*delta[0]+math.sin(yaw0)*delta[1],-math.sin(yaw0)*delta[0]+math.cos(yaw0)*delta[1]]
        c.update(tipped=bool(failed[i]),displacement_body_m=body_delta,
            mean_body_velocity=np.mean([r['velocity'][i][:2] for r in drive[50:]],axis=0).tolist(),
            max_yaw_drift_deg=math.degrees(max(abs(math.atan2(math.sin(r['yaw'][i]-yaw0),math.cos(r['yaw'][i]-yaw0))) for r in drive)),
            stop_drift_m=math.dist(stop[0]['position'][i][:2],stop[-1]['position'][i][:2]),
            final_speed_mps=float(np.linalg.norm(stop[-1]['velocity'][i][:2])))
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(cases=cases,samples=rows,checkpoints={'rough':a.rough,'rocky_slope':a.rocky}),indent=2))
    out.with_suffix('.summary.json').write_text(json.dumps(dict(cases=cases,seconds=12.,automatic_resets=0),indent=2))
    print('[lateral-study] '+json.dumps(cases),flush=True)
finally:
    if 'env' in globals():env.close()
    app.close()
