"""Isaac Sim adapter for checkpoint control. Legacy Navigator is never executed."""
import csv
import importlib
import json
import math
from pathlib import Path
import queue
import threading
import time
from types import SimpleNamespace
import numpy as np


def run(args, app):
    checkpoint = args.ckpt_rocky or args.checkpoint
    if not checkpoint or not Path(checkpoint).is_file():
        raise SystemExit('Checkpoint control needs --checkpoint or --ckpt_rocky with the rocky_slope model.')
    if args.policy_mode == 'rough':
        raise SystemExit('Checkpoint control uses rocky_slope; use --nav_controller legacy for rough.')
    env = view = remote = None
    grid = path = None
    dashboard = None
    from concurrent.futures import ThreadPoolExecutor
    exporter = ThreadPoolExecutor(max_workers=1)
    export_job = None
    dashboard_stop = threading.Event()
    try:
        import gymnasium as gym
        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rexmi_rl.drive_adapter import load_policy, patch_command_ranges, policy_actions
        from rexmi_rl.nav.mission import Mission, MissionPlanner
        from rexmi_rl.nav.localizer import SimLocalizer
        from rexmi_rl.nav.observed_map import ObservedTerrainMap
        from rexmi_rl.nav.checkpoint_control import CheckpointController
        spec = gym.spec(args.task)
        def config(key):
            mod,cls=spec.kwargs[key].rsplit(':',1)
            return getattr(importlib.import_module(mod),cls)()
        cfg,agent=config('env_cfg_entry_point'),config('rsl_rl_cfg_entry_point')
        cfg.seed=args.seed
        cfg.scene.num_envs=1
        cfg.sim.device=args.device
        tg=getattr(cfg.scene.terrain,'terrain_generator',None)
        if tg is not None:
            tg.num_cols=tg.num_rows=1
            tg.seed=args.seed
        cfg.events.reset_base.params={
            'pose_range': {'x':(args.spawn_x,args.spawn_x),'y':(args.spawn_y,args.spawn_y),
                           'z':(args.spawn_z,args.spawn_z),'yaw':(args.spawn_yaw,args.spawn_yaw)},
            'velocity_range': {k:(0.,0.) for k in ('x','y','z','roll','pitch','yaw')}}
        # No automatic reset can rescue a validation run. Controller detects tip
        # and holds. Disable every termination, including the episode timeout.
        for key,value in vars(cfg.terminations).items():
            if not key.startswith('_') and value is not None:
                setattr(cfg.terminations,key,None)
        cfg.episode_length_s=max(cfg.episode_length_s,args.max_steps*cfg.sim.dt*cfg.decimation+10.)
        cfg.commands.base_velocity.heading_command=False
        cfg.commands.base_velocity.rel_heading_envs=0.
        cfg.commands.base_velocity.resampling_time_range=(1e9,1e9)
        cfg.commands.base_velocity.debug_vis=True
        cfg.observations.policy.enable_corruption=False
        env=RslRlVecEnvWrapper(gym.make(args.task,cfg=cfg,render_mode=None if args.headless else 'rgb_array'),
                              clip_actions=agent.clip_actions)
        patch_command_ranges(env,'rocky_slope')
        policy=load_policy(checkpoint,args.device)
        env.reset()
        robot=env.unwrapped.scene['robot']
        localizer=SimLocalizer(robot,0)
        sensors={}
        for name in ('lidar','height_scanner','forward_scanner'):
            try: sensors[name]=env.unwrapped.scene[name]
            except KeyError: pass
        if 'lidar' not in sensors:
            raise RuntimeError('Checkpoint controller requires LiDAR; no blind fallback.')
        # Body origins give a conservative wheel/leg envelope, expanded by 10 cm.
        positions=robot.data.body_pos_w[0].detach().cpu().numpy()
        base=robot.data.root_pos_w[0,:2].detach().cpu().numpy()
        radius=max(.45,float(np.max(np.linalg.norm(positions[:,:2]-base,axis=1)))+.10)
        grid=ObservedTerrainMap(origin=(args.crater_x,args.crater_y),footprint_radius=radius)
        grid.STEP_THRESH=args.step_thresh
        mission=Mission(args.mission)
        planner=MissionPlanner(crater_centre=grid.origin,r_floor=args.r_floor,r_rim=args.r_rim,
                               spawn_x=args.spawn_x,entry_azimuth_deg=args.entry_azimuth_deg)
        waypoints=planner.get_waypoints(mission,goal_x=args.goal_x,goal_y=args.goal_y,goal_radius=args.goal_radius)
        control=CheckpointController(grid,waypoints,cruise_speed=args.cruise_speed)
        requests=queue.SimpleQueue()
        if args.start_paused: control.request('stop',0.)
        if args.manual_checkpoints: control.checkpoint_mode(True,0.)
        lock=threading.Lock()
        empty=np.array([])
        shared=dict(manual_checkpoint_mode=args.manual_checkpoints,pose=localizer.get_pose(),wp_idx=0,planned_path=[],local_out=None,
                    recovery_status='OBSERVE',mission=args.mission,cmd=(0.,0.,0.),cost_grid=grid.get_cost_grid(),
                    speed=0.,slam_converged=False,slam_map_size=0,slam_rms=0.,slam_icp_count=0,
                    fwd_cloud_xyz=(empty,empty,empty),policy_status='rocky_slope',trajectory=[])
        nav=SimpleNamespace(_sim_localizer=localizer,_lidar=sensors['lidar'],_env_idx=0,
                            _omap=grid,_lock=lock,shared=shared,requests=requests if not args.no_dashboard else None)
        if args.perception_view:
            from rexmi_rl.nav.perception_view import PerceptionView
            view=PerceptionView(nav,hz=args.overlay_hz,point_limit=args.lidar_display_points)
            view.layer=getattr(args,"survey_layer","terrain")
        if not args.headless:
            import omni.ui as ui
            remote=ui.Window('REXMI | Checkpoint control',width=420,height=300)
            with remote.frame:
                with ui.VStack(spacing=5):
                    ui.Label('ROCKY SLOPE | autonomous checkpoints')
                    status=ui.Label('Observing',word_wrap=True)
                    with ui.HStack():
                        ui.Button('STOP',clicked_fn=lambda:requests.put(('stop',0.,0.)))
                        ui.Button('RESUME AUTO',clicked_fn=lambda:requests.put(('resume',0.,0.)))
                    ui.Button('CLICK CHECKPOINT MODE',clicked_fn=lambda:requests.put(('checkpoint_mode',1.,0.)))
                    ui.Button('RESTORE MISSION (paused)',clicked_fn=lambda:requests.put(('checkpoint_mode',0.,0.)))
                    ui.Label('Manual controls suspend autonomy; release to stop.')
                    for label,vx,omega in [('Forward',.4,0.),('Back',-.4,0.),('Left',0.,.35),('Right',0.,-.35)]:
                        ui.Button(label,mouse_pressed_fn=lambda x,y,b,m,v=vx,w=omega:requests.put(('manual',v,w)),
                                  mouse_released_fn=lambda x,y,b,m:requests.put(('manual',0.,0.)))
        if not args.no_dashboard and not args.headless:
            from rexmi_rl.nav.dashboard import Dashboard
            dashboard=Dashboard(shared,lock,grid,waypoints,request_queue=requests)
            threading.Thread(target=dashboard.run,args=(dashboard_stop,),daemon=True).start()
        # Optional terminal commands. Never require stdin to run unattended.
        def terminal():
            import sys
            for line in sys.stdin:
                command=line.strip().lower()
                if command in ('stop','s'): requests.put(('stop',0.,0.))
                elif command in ('resume','auto'): requests.put(('resume',0.,0.))
                elif command in ('f','b','l','r'):
                    vx,omega={'f':(.4,0.),'b':(-.4,0.),'l':(0.,.35),'r':(0.,-.35)}[command]
                    requests.put(('manual',vx,omega))
        threading.Thread(target=terminal,daemon=True).start()
        path=Path(args.log_file or time.strftime('logs/nav/checkpoint_%Y-%m-%d_%H-%M-%S.csv'))
        path.parent.mkdir(parents=True,exist_ok=True)
        path.with_suffix('.events.jsonl').write_text('')
        dt=env.unwrapped.step_dt
        def write_survey(snapshot):
            temporary=path.with_suffix('.map.tmp')
            with temporary.open('wb') as out:
                np.savez_compressed(out,**snapshot)
            temporary.replace(path.with_suffix('.map.npz'))
        def export_survey(final=False):
            nonlocal export_job
            if export_job is not None:
                if not final and not export_job.done(): return
                export_job.result()
            export_job=exporter.submit(write_survey,grid.survey_snapshot())
            if final: export_job.result()
        try:
            material=robot.root_physx_view.get_material_properties().detach().cpu().numpy()
            properties={'robot_static_friction_range':[float(material[...,0].min()),float(material[...,0].max())],
                        'robot_dynamic_friction_range':[float(material[...,1].min()),float(material[...,1].max())],
                        'terrain_static_friction':cfg.scene.terrain.physics_material.static_friction,
                        'terrain_dynamic_friction':cfg.scene.terrain.physics_material.dynamic_friction,
                        'terrain_combine_mode':cfg.scene.terrain.physics_material.friction_combine_mode}
        except Exception as exc:
            properties={'inspection_error':str(exc)}
        path.with_suffix('.materials.json').write_text(json.dumps(properties,indent=2))
        trajectory=[]
        last_export=-30.
        exported_failures=0
        map_update_ms=[]
        previous=None
        result='step_limit'
        max_error=0.
        now=0.
        capture=None
        print(f'[checkpoint] rocky_slope only; envelope={radius:.2f}m + 0.10m clearance; log={path}',flush=True)
        with path.open('w',newline='',buffering=1) as f:
            writer=csv.writer(f)
            writer.writerow(['step','sim_time','x','y','z','yaw','speed','up_z','wp_idx','state','reason',
                             'vx_cmd','omega_cmd','checkpoint_x','checkpoint_y','target_x','target_y',
                             'heading_error','cross_track','coverage','oldest_observation_age','policy',
                             'slope_deg','roughness_m','climb_failures','backslide_m','route_decision','render_ms'])
            for step in range(args.max_steps):
                if not app.is_running():
                    result='window_closed';break
                now=step*dt
                pose=localizer.get_pose()
                xy=(pose.x,pose.y)
                velocity=localizer.get_velocity()
                speed=math.hypot(velocity[0],velocity[1])
                q=robot.data.root_quat_w[0]
                up=1.-2.*float(q[1]**2+q[2]**2)
                if step%max(1,round(.2/dt))==0:
                    scan_record={'time':now}
                    for name,sensor in sensors.items():
                        data=sensor.data
                        hits=data.ray_hits_w[0].detach().cpu().numpy()
                        sensor_origin=data.pos_w[0].detach().cpu().numpy()
                        # The policy's local height grid casts down from a virtual
                        # 20 m offset. Its footprint is local, but ray length is
                        # not 5 m. Keep that existing simulated sensor intact.
                        max_range={'lidar':15.,'forward_scanner':5.,'height_scanner':25.}[name]
                        grid.ingest(hits,sensor_origin,now,max_range=max_range)
                        if args.scan_log:
                            scan_record[name+'_hits']=hits
                            scan_record[name+'_origin']=sensor_origin
                            scan_record[name+'_range']=max_range
                    started=time.perf_counter()
                    if args.map_retention == 'radius': grid.retain_radius(xy,args.map_keep_radius)
                    grid.rebuild(now)
                    map_update_ms.append(1000*(time.perf_counter()-started))
                    with lock:
                        shared.update(cost_grid=grid.get_cost_grid(),observed_mask=grid.known.copy(),clearance_mask=grid.clearance_mask)
                        shared['survey_layers']={k:getattr(grid,k).copy() for k in
                                                 ('slope','roughness','failures','successes')}
                    if args.scan_log:
                        scan_dir=Path(args.scan_log)
                        scan_dir.mkdir(parents=True,exist_ok=True)
                        np.savez_compressed(scan_dir/f'{step:06d}.npz',**scan_record)
                while not requests.empty():
                    action,vx,omega=requests.get()
                    if action=='checkpoint_mode':
                        if vx and args.no_dashboard:
                            control.request('stop',now);control.reason='checkpoint picking requires dashboard; relaunch without --no_dashboard'
                        else: control.checkpoint_mode(bool(vx),now)
                    elif action=='target': control.choose_checkpoint(xy,(vx,omega),now)
                    elif action=='resume' and getattr(control,'manual_checkpoint_mode',False) and control.checkpoint is None:
                        pass
                    else: control.request(action,now,vx,omega)
                    with lock: shared['manual_checkpoint_mode']=getattr(control,'manual_checkpoint_mode',False)
                    if dashboard: dashboard._waypoints=control.waypoints
                motion=control.tick(xy,pose.yaw,speed,now,upright=up,velocity=velocity)
                if control.state=='COMPLETE' and getattr(control,'manual_checkpoint_mode',False):
                    control.request('stop',now)
                    control.reason='selected checkpoint reached; click another point'
                if now-last_export>=30.:
                    export_survey();last_export=now
                if control.climb_failures>exported_failures:
                    with path.with_suffix('.events.jsonl').open('a') as events:
                        events.write(json.dumps(dict(time=now,position=xy,yaw=pose.yaw,
                            event=control.last_climb_event,count=control.climb_failures))+'\n')
                    exported_failures=control.climb_failures
                quality=grid.quality(xy)
                max_error=max(max_error,control.tracking_error)
                if not trajectory or math.dist(xy,trajectory[-1])>.1: trajectory.append(xy)
                cp=control.checkpoint or (math.nan,math.nan)
                target=control.target or (math.nan,math.nan)
                with lock:
                    shared.update(pose=pose,wp_idx=control.wp_idx,planned_path=list(control.path),
                                  recovery_status=control.state,cmd=(motion.vx,0.,motion.omega),
                                  speed=speed,trajectory=list(trajectory),
                                  checkpoint=control.checkpoint,steering_target=control.target,
                                  nav_reason=control.reason,coverage=quality['coverage'],observation_age=quality['age'],
                                  mapped_coverage=quality['mapped_coverage'],mapped_area=quality['mapped_area'])
                if view: view.update()
                if args.capture_file and not args.headless:
                    if step==2:
                        from isaacsim.core.utils.viewports import set_camera_view
                        set_camera_view(eye=(pose.x+6,pose.y+6,pose.z+6),
                                        target=(pose.x-2,pose.y,pose.z-1))
                    if step==args.capture_step:
                        from omni.kit.viewport.utility import get_active_viewport,capture_viewport_to_file
                        Path(args.capture_file).parent.mkdir(parents=True,exist_ok=True)
                        capture=capture_viewport_to_file(get_active_viewport(),file_path=args.capture_file)
                description=(f'{control.state}: {control.reason}\nWP {control.wp_idx}/{len(control.waypoints)} | '
                             f'checkpoint ({cp[0]:.1f}, {cp[1]:.1f})\n'
                             f'command {motion.vx:+.2f} m/s, {motion.omega:+.2f} rad/s\n'
                             f'mapped {quality["mapped_coverage"]:.0%} | fresh {quality["coverage"]:.0%}\n'
                             f'surveyed {quality["mapped_area"]:.1f} m² | oldest local observation {quality["age"]:.1f}s')
                if not args.headless: status.text=description
                if control.state!=previous or step%250==0:
                    print(f'[checkpoint {now:.1f}s] {description.replace(chr(10)," | ")} pos={xy}',flush=True)
                    previous=control.state
                writer.writerow([step,now,pose.x,pose.y,pose.z,pose.yaw,speed,up,control.wp_idx,control.state,
                                 control.reason,motion.vx,motion.omega,*cp,*target,control.heading_error,
                                 control.tracking_error,quality['coverage'],quality['age'],'rocky_slope',
                                 float(np.degrees(np.arctan(grid.slope[grid.world_to_cell(*xy)]))),
                                 float(grid.roughness[grid.world_to_cell(*xy)]),control.climb_failures,
                                 control.backslide_distance,control.route_decision,view.render_ms if view else 0.])
                if control.state=='COMPLETE' and not getattr(control,'manual_checkpoint_mode',False): result='complete';break
                if args.headless and control.state=='HOLD': result='hold';break
                actions=policy_actions(env,policy,motion.vx,motion.omega)
                _,_,done,_=env.step(actions)
                if bool(done[0]):
                    result='unexpected_environment_reset';break
        summary=dict(result=result,seed=args.seed,sim_seconds=now,waypoints_reached=control.wp_idx,
                     total_waypoints=len(control.waypoints),manual_intervention=control.intervened,
                     max_cross_track_m=max_error,policy='rocky_slope',policy_switches=0,
                     footprint_radius_m=radius,cruise_ceiling=args.cruise_speed,
                     hold_failures=control.hold_failures,rolling_checkpoints=control.rolling_checkpoints,
                     final_state=control.state,final_reason=control.reason,
                     local_recovery_attempts=len(control.recovery_attempts),
                     last_hold_failure=control.last_hold_failure,
                     climb_failures=control.climb_failures,backslide_m=control.backslide_distance,
                     map_update_ms_mean=float(np.mean(map_update_ms)) if map_update_ms else 0.,
                     validated=False)
        export_survey()
        path.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2))
        print('[checkpoint] RESULT '+json.dumps(summary),flush=True)
    finally:
        if grid is not None and path is not None:
            try: export_survey(final=True)
            except Exception as exc: print(f'[survey] final export failed: {exc}')
        exporter.shutdown(wait=True)
        if view: view.close()
        if remote: remote.destroy()
        if dashboard: dashboard_stop.set()
        if env:
            # Disconnect arrow callbacks while their tensors still live on GPU.
            env.unwrapped.command_manager.get_term('base_velocity').set_debug_vis(False)
            env.close()
        app.close()
