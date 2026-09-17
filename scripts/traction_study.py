#!/usr/bin/env python3
"""Controlled 2x2x2 uphill comparison; never changes demo configuration files.

Eight simultaneous trials: geometric roughness 0/5 cm, dynamic friction
0.6/0.8, command 0.4/0.8 m/s. Static friction is fixed at 0.8. One initial
reset only; a tipped robot is recorded as failed and receives a zero command.
"""
import argparse
import importlib
import json
import math
from pathlib import Path
import sys
import os
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
sys.path.insert(0,os.environ.get('ISAACLAB_DIR',os.path.expanduser('~/IsaacLab')))
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser()
parser.add_argument('--checkpoint',required=True)
parser.add_argument('--slope_deg',type=float,default=30.)
parser.add_argument('--output',required=True)
parser.add_argument('--seed',type=int,default=42)
args=parser.parse_args()
app=AppLauncher(headless=True).app
try:
    import numpy as np
    import torch
    import gymnasium as gym
    import rexmi_rl
    from isaaclab.terrains import TerrainGeneratorCfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rexmi_rl.tasks.locomotion.velocity.config.go2w.crater_terrain import RockyPyramidSlopeCfg
    from rexmi_rl.drive_adapter import load_policy,patch_command_ranges,inject_command,policy_actions
    task='RexmiRl-Go2w-Crater-Bowl-RockySlope-Play-v0'
    spec=gym.spec(task)
    def config(key):
        mod,cls=spec.kwargs[key].rsplit(':',1)
        return getattr(importlib.import_module(mod),cls)()
    cfg=config('env_cfg_entry_point');agent=config('rsl_rl_cfg_entry_point')
    cfg.seed=args.seed;cfg.scene.num_envs=8
    terrain=lambda rough:RockyPyramidSlopeCfg(proportion=.5,slope_min_deg=args.slope_deg,
        slope_max_deg=args.slope_deg,roughness_min_m=rough,roughness_max_m=rough,
        boulder_count_min=0,boulder_count_max=0,platform_width=2.,seed=args.seed)
    cfg.scene.terrain.terrain_generator=TerrainGeneratorCfg(seed=args.seed,curriculum=True,
        size=(24.,24.),num_rows=1,num_cols=2,sub_terrains={'smooth':terrain(0.),'textured':terrain(.05)},
        horizontal_scale=.1,vertical_scale=.005,slope_threshold=None,use_cache=False)
    cfg.scene.terrain.max_init_terrain_level=0
    cfg.scene.terrain.physics_material.static_friction=1.
    cfg.scene.terrain.physics_material.dynamic_friction=1.
    cfg.scene.terrain.physics_material.friction_combine_mode='multiply'
    for key in vars(cfg.terminations):
        if not key.startswith('_'): setattr(cfg.terminations,key,None)
    for key in ('add_base_mass','push_robot','base_external_force_torque'):
        if hasattr(cfg.events,key): setattr(cfg.events,key,None)
    cfg.curriculum.terrain_levels=None
    cfg.commands.base_velocity.heading_command=False
    cfg.commands.base_velocity.rel_heading_envs=0.
    cfg.commands.base_velocity.resampling_time_range=(1e9,1e9)
    cfg.commands.base_velocity.debug_vis=False
    cfg.observations.policy.enable_corruption=False
    z=.45
    cfg.events.reset_base.params={'pose_range':{'x':(0.,0.),'y':(0.,0.),'z':(z,z),'yaw':(0.,0.)},
        'velocity_range':{k:(0.,0.) for k in ('x','y','z','roll','pitch','yaw')}}
    env=RslRlVecEnvWrapper(gym.make(task,cfg=cfg),clip_actions=agent.clip_actions)
    patch_command_ranges(env,'rocky_slope')
    policy=load_policy(args.checkpoint,cfg.sim.device)
    env.reset();robot=env.unwrapped.scene['robot']
    # Identical joint starting state and material assignment for matched cases.
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(),torch.zeros_like(robot.data.joint_vel))
    material=robot.root_physx_view.get_material_properties()
    cases=[]
    types=env.unwrapped.scene.terrain.terrain_types.cpu().numpy()
    for i in range(8):
        dynamic=.6 if i%4<2 else .8
        speed=.4 if i%2==0 else .8
        material[i,:,0]=.8;material[i,:,1]=dynamic;material[i,:,2]=0.
        cases.append(dict(env=i,roughness_m=0. if types[i]==0 else .05,
                          dynamic_friction=dynamic,static_friction=.8,speed_command=speed,
                          slope_deg=args.slope_deg,seed=args.seed))
    robot.root_physx_view.set_material_properties(material,torch.arange(8,dtype=torch.int32,device='cpu'))
    actual=robot.root_physx_view.get_material_properties().cpu().numpy()
    for i,case in enumerate(cases):
        case['effective_robot_dynamic_friction']=float(actual[i,:,1].mean())
    dt=env.unwrapped.step_dt;failed=np.zeros(8,bool);rows=[];first_tip=[None]*8
    for step in range(round(15/dt)):
        now=step*dt
        pos=robot.data.root_pos_w.detach().cpu().numpy()-env.unwrapped.scene.env_origins.cpu().numpy()
        vel=robot.data.root_lin_vel_b.detach().cpu().numpy()
        q=robot.data.root_quat_w.detach().cpu().numpy()
        up=1-2*(q[:,1]**2+q[:,2]**2)
        for i in range(8):
            if up[i]<.3 and first_tip[i] is None: first_tip[i]=now
        failed|=up<.3
        commands=[case['speed_command'] if 2<=now<12 and not failed[i] else 0. for i,case in enumerate(cases)]
        rows.append(dict(time=now,position=pos.tolist(),velocity=vel.tolist(),up=up.tolist(),command=commands))
        for i in range(8): inject_command(env,commands[i],0.,i)
        with torch.inference_mode():
            actions=policy_actions(env,policy,commands[0],0.)
            _,_,done,_=env.step(actions)
        if bool(done.any()): raise RuntimeError('Unexpected reset in controlled trial')
    for i,case in enumerate(cases):
        drive=[r for r in rows if 2<=r['time']<12]
        stopped=[r for r in rows if r['time']>=12]
        case.update(tipped=bool(failed[i]),first_tip_s=first_tip[i],initialization_valid=first_tip[i] is None or first_tip[i]>=2.,distance_uphill_m=drive[-1]['position'][i][0]-drive[0]['position'][i][0],
            height_gain_m=drive[-1]['position'][i][2]-drive[0]['position'][i][2],
            backslide_m=sum(max(0.,-r['velocity'][i][0])*dt for r in drive),
            max_lateral_error_m=max(abs(r['position'][i][1]) for r in drive),
            stop_drift_m=math.dist(stopped[0]['position'][i][:2],stopped[-1]['position'][i][:2]),
            final_speed_mps=float(np.linalg.norm(stopped[-1]['velocity'][i][:2])))
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'cases':cases,'samples':rows},indent=2))
    print('[traction-study] '+json.dumps(cases),flush=True)
finally:
    if 'env' in globals(): env.close()
    app.close()
