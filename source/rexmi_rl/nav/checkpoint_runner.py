"""Isaac Sim adapter for checkpoint control. Legacy Navigator is never executed."""
import csv
from dataclasses import asdict
import importlib
import hashlib
import json
import math
from pathlib import Path
import queue
import tempfile
import threading
import time
import traceback
from types import SimpleNamespace
import numpy as np


POSTURE_TELEMETRY_COLUMNS = ['body_quaternion_w','body_quaternion_x','body_quaternion_y','body_quaternion_z',
    'angular_velocity_world_x','angular_velocity_world_y','angular_velocity_world_z',
    'observed_gradient_x','observed_gradient_y','observed_normal_x','observed_normal_y','observed_normal_z',
    'observed_normal_valid']


def route_comparison_key(control):
    """Shared runtime consumer for planner decision records."""
    return (tuple((p['profile'],round(p['length_m'],1),round(p['max_uphill_deg'],1),
                   tuple(map(tuple,p.get('route_path',p.get('path',[]))))) for p in control.route_options),
            tuple((p['profile'],p['connected'],p['has_route'],p['uphill_limit'],
                   (p.get('planning_snapshot') or {}).get('path')) for p in control.route_searches))


def validate_output_paths(args):
    """Fail before simulator startup without creating/truncating run artifacts."""
    log_file=getattr(args,'log_file',None)
    joint_dir=getattr(args,'joint_log_dir',None)
    targets=[]
    if log_file:
        path=Path(log_file)
        targets.extend((path,path.with_suffix('.events.jsonl')))
    if joint_dir:
        directory=Path(joint_dir).expanduser()
        if directory.exists():
            raise ValueError(f'Joint log directory already exists: {directory}. Choose a fresh output directory.')
        targets.append(directory)
    for target in targets:
        try:
            target.parent.mkdir(parents=True,exist_ok=True)
            # Exercise real directory write permission, not just mode bits.
            with tempfile.TemporaryFile(dir=target.parent) as probe:
                probe.write(b'1');probe.flush()
        except OSError as exc:
            raise ValueError(f'Cannot write navigation output at {target}: {exc}. '
                'RUN may be unset; use scripts/launch_ascent_gui.sh or initialize a writable output directory.') from exc


def completed_physics_diagnostics(snapshots, exporter, logger, done, env):
    """Record known step completion before fallible post-step diagnostics."""
    if logger is not None:logger.physics_completed()
    if snapshots is not None:snapshots.flush(exporter,completed=True,reason='env.step returned')
    if logger is not None:
        logger.end_navigation_step(done=bool(done[0]),
            terminated=bool(env.reset_terminated[0]),truncated=bool(env.reset_time_outs[0]))


def posture_telemetry(quaternion, angular_velocity, observation):
    """Passive, same-tick orientation and observed fitted normal; no control input."""
    gradient = getattr(observation,'terrain_gradient',None)
    valid = bool(getattr(observation,'fresh',False) and gradient is not None
                 and len(gradient)==2 and all(math.isfinite(v) for v in gradient))
    scale = math.hypot(*gradient,1.) if valid else math.nan
    valid = valid and math.isfinite(scale)
    if valid:
        gx,gy = gradient
        normal = (-gx/scale,-gy/scale,1./scale)
    else:
        gradient = (math.nan,math.nan)
        normal = (math.nan,math.nan,math.nan)
    return [*map(float,quaternion),*map(float,angular_velocity),*gradient,*normal,valid]


def record_run_failure(log_path, error, sim_seconds=None, source_id=None):
    """Preserve software errors before simulator shutdown can mask traceback."""
    detail=''.join(traceback.format_exception(type(error),error,error.__traceback__))
    print('[checkpoint] SOFTWARE FAILURE\n'+detail,flush=True)
    if log_path is not None:
        failure=dict(completed=False,result='software_error',sim_seconds=sim_seconds,
            reason=str(error),exception_type=type(error).__name__,traceback=detail,
            source_id=source_id,counts_as_physical_failure=False,counts_as_qualification=False)
        try:
            Path(log_path).with_suffix('.failure.json').write_text(json.dumps(failure,indent=2)+'\n')
        except OSError as exc:
            print(f'[checkpoint] Could not save failure report: {exc}',flush=True)


def load_operator_replay(path):
    if not path:return []
    events=json.loads(Path(path).read_text())
    if not isinstance(events,list):raise ValueError('operator replay must be a list')
    result=[]
    for event in events:
        now=float(event['time']);action=event['action']
        if not math.isfinite(now) or now<0 or action not in ('stop','resume'):
            raise ValueError('operator replay accepts finite nonnegative stop/resume events only')
        if result and now<=result[-1][0]:raise ValueError('operator replay times must increase')
        result.append((now,action))
    return result


def load_checkpoint_replay(path):
    """Replay only accepted target selections, never recorded poses or physics."""
    if not path:return []
    selections=[]
    for line in Path(path).read_text().splitlines():
        event=json.loads(line)
        if event.get('event')!='checkpoint_selection' or not event.get('accepted') or event.get('duplicate'):continue
        t=float(event['time']);target=tuple(map(float,event['requested']))
        if t<0 or len(target)!=2 or not all(math.isfinite(v) for v in (t,*target)):
            raise ValueError('Replay times and target coordinates must be finite; time must be nonnegative.')
        selections.append((t,target))
    if not selections:raise ValueError('No accepted checkpoint selections in replay file.')
    return sorted(selections)


def apply_operator_request(control, side, action, vx, omega, xy, yaw, now, no_dashboard=False, locked_mission=False):
    """Validate a map click before disturbing an active controller/policy handoff."""
    if locked_mission and action in ('target', 'checkpoint_mode'):
        return dict(event='mission_override_rejected', time=now, action=action,
                    reason='autonomous mission goal remains locked; STOP/manual takeover remain available')
    if action=='target':
        if not getattr(control,'manual_checkpoint_mode',False): return None
        accepted=control.choose_checkpoint(xy,(vx,omega),now,yaw)
        event=control.last_selection
        if not accepted or event.get('duplicate'): return event
        if side:
            if side.active: side.cancel('operator target')
            side.suppressed_target=None
        return event
    if side and side.active: side.cancel('operator '+action)
    if side and action=='resume': side.suppressed_target=None
    if action=='checkpoint_mode':
        if vx and no_dashboard:
            control.request('stop',now)
            control.reason='checkpoint picking requires dashboard; relaunch without --no_dashboard'
        else: control.checkpoint_mode(bool(vx),now)
    elif action=='resume' and getattr(control,'manual_checkpoint_mode',False) and control.checkpoint is None:
        pass
    else: control.request(action,now,vx,omega)
    return None


def run(args, app):
    round_trip = args.mission == 'round_trip'
    loop_mode = args.mission == 'crater_loop'
    loop_mission = None
    if round_trip or loop_mode:
        from rexmi_rl.nav.round_trip import DEADLINE_SECONDS, deadline_reached, stage_status
    replay=load_checkpoint_replay(getattr(args,'checkpoint_replay',None))
    replay_index=0
    operator_replay=load_operator_replay(getattr(args,'operator_replay',None))
    operator_replay_index=0
    checkpoint = args.ckpt_rocky or args.checkpoint
    if not checkpoint or not Path(checkpoint).is_file():
        raise SystemExit('Checkpoint control needs --checkpoint or --ckpt_rocky with the rocky_slope model.')
    if args.policy_mode == 'rough':
        raise SystemExit('Checkpoint control uses rocky_slope; use --nav_controller legacy for rough.')
    env = view = remote = joint_logger = None
    grid = path = None
    dashboard = None
    from concurrent.futures import ThreadPoolExecutor
    exporter = ThreadPoolExecutor(max_workers=1)
    export_job = None
    export_survey = None
    dashboard_stop = threading.Event()
    try:
        import gymnasium as gym
        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rexmi_rl.drive_adapter import load_policy, patch_command_ranges, policy_actions
        from rexmi_rl.nav.mission import Mission, MissionPlanner, Waypoint
        from rexmi_rl.nav.localizer import SimLocalizer
        from rexmi_rl.nav.observed_map import ObservedTerrainMap
        from rexmi_rl.nav.checkpoint_control import CheckpointController
        ascent = args.nav_controller == 'ascent'
        if ascent or loop_mode:
            from rexmi_rl.nav.ascent_map import AscentTerrainMap
            from rexmi_rl.nav.ascent_control import AscentController
            if getattr(args, 'ascent_architecture', 'behavior_tree') == 'behavior_tree':
                from rexmi_rl.nav.ascent_behavior import BehaviorTreeAscentController as AscentController
            elif args.ascent_architecture == 'remote':
                from rexmi_rl.nav.remote_control import RemoteAscentController as AscentController
                if getattr(args, 'ascent_path_result', None):
                    from rexmi_rl.nav.saved_path_control import SavedPathController as AscentController
                    if getattr(args, 'ascent_unguarded', False):
                        from rexmi_rl.nav.experimental_path_control import ExperimentalPathController as AscentController
            from rexmi_rl.nav.reachability import ReachabilityModel
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
        policies={'rocky_slope':load_policy(checkpoint,args.device)}
        if args.flat_sidesteps:
            policies['rough']=load_policy(args.ckpt_rough,args.device)
            if ((policies['rough'].observation_size,policies['rough'].action_size) !=
                (policies['rocky_slope'].observation_size,policies['rocky_slope'].action_size)):
                raise ValueError('Rough and rocky-slope observation/action sizes must match')
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
        shape={}
        if ascent or loop_mode:
            heading=localizer.get_pose().yaw
            relative=positions[:,:2]-base
            shape=dict(half_length=float(np.max(np.abs(relative[:,0]*math.cos(heading)+relative[:,1]*math.sin(heading))))+.10,
                       half_width=float(np.max(np.abs(-relative[:,0]*math.sin(heading)+relative[:,1]*math.cos(heading))))+.10)
        grid=(AscentTerrainMap if ascent or loop_mode else ObservedTerrainMap)(origin=(args.crater_x,args.crater_y),footprint_radius=radius,**shape)
        grid.STEP_THRESH=args.step_thresh
        terrain_prior=None
        initial_survey=None
        if ascent and (getattr(args,'simulator_survey',False) or getattr(args,'terrain_prior',None)):
            from rexmi_rl.nav.terrain_prior import simulator_survey,load_prior,prior_snapshot
            survey_radius=max(args.r_rim+4.,math.hypot(args.goal_x-args.crater_x,args.goal_y-args.crater_y)+3.,
                              math.hypot(args.spawn_x-args.crater_x,args.spawn_y-args.crater_y)+3.)
            if args.simulator_survey:
                terrain_prior=simulator_survey(grid,sensors['lidar'],task=args.task,seed=args.seed,
                                               radius=survey_radius,device=args.device)
            else:
                terrain_prior=load_prior(grid,args.terrain_prior,task=args.task,seed=args.seed,required_radius=survey_radius)
            initial_survey=prior_snapshot(grid)
            print('[survey] '+json.dumps(terrain_prior),flush=True)
        mission=Mission.VELOCITY_GOAL if args.mission=='ascent' or loop_mode else Mission(args.mission)
        planner=MissionPlanner(crater_centre=grid.origin,r_floor=args.r_floor,r_rim=args.r_rim,
                               spawn_x=args.spawn_x,entry_azimuth_deg=args.entry_azimuth_deg)
        waypoints=planner.get_waypoints(mission,goal_x=args.goal_x,goal_y=args.goal_y,goal_radius=args.goal_radius)
        extra = dict(model=ReachabilityModel.load(args.reachability_model) if args.reachability_model else None,
                     mode=args.ascent_mode, async_planning=True,
                     run_id=f'{Path(args.log_file).parent.name}-{Path(args.log_file).stem}' if args.log_file else f'ascent-{time.time_ns()}') if ascent else {}
        if ascent and getattr(args,'ascent_architecture','')=='remote':
            extra['route_objective']=getattr(args,'ascent_route_objective','baseline')
            extra['goal_mode']=getattr(args,'ascent_goal','fixed')
            extra['exit_rim_radius']=planner.r_rim
            if getattr(args, 'ascent_path_result', None):
                extra['saved_path_result']=args.ascent_path_result
        if loop_mode:
            from .crater_loop import DescentController
        control=(DescentController if loop_mode else AscentController if ascent else CheckpointController)(grid,waypoints,cruise_speed=args.cruise_speed,
                                     preferred_uphill=args.preferred_uphill_deg,
                                     uphill_limit=args.max_uphill_deg,side_limit=args.max_side_slope_deg, **extra)
        if getattr(args, 'ascent_path_result', None):
            args.goal_x,args.goal_y=control.selected_exit
            waypoints[:]=[Waypoint(*control.selected_exit,.4,'Inspected blue exit')]
        if ascent and getattr(args,'ascent_architecture','')=='remote' and args.log_file:
            control.planning_snapshot_dir=Path(args.log_file).with_suffix('.planning')
        if getattr(args,'ascent_uphill_speed_response',False):
            control.configure_uphill_speed_response()
        if getattr(args,'ascent_recovery_continuation',False):
            control.configure_recovery_continuation()
        if getattr(args,'ascent_release_memory',False):
            control.configure_release_memory()
        if getattr(args,'ascent_correction_release',False):
            control.configure_correction_release()
        if getattr(args,'ascent_gentle_turn_entry',False):
            control.configure_gentle_turn_entry()
        if getattr(args,'ascent_active_zero_envelope',False):
            control.configure_active_zero_envelope()
        if args.mission == 'round_trip':
            control.configure_outbound_survey_routes()
        from rexmi_rl.nav.flat_sidestep import FlatSidestep
        side=FlatSidestep(grid) if args.flat_sidesteps else None
        requests=queue.SimpleQueue()
        from .dashboard_telemetry import TravelHistory
        from .terrain_display import from_model, cost_view, heading_grade
        travel_history=TravelHistory()
        display_last_heading=None
        display_model=None
        display_reference=None
        if not args.headless and hasattr(control,'saved_path_metadata'):
            from . import planning_snapshot, slope_cost
            display_model=getattr(control,'preference',None)
            if display_model is None:
                frozen,_=planning_snapshot.load(control.saved_path_metadata['map_path'])
                display_model=slope_cost.Model(frozen,slope_cost.Parameters(**control.saved_path_metadata['parameters']))
            display_reference=from_model(display_model)
        if args.start_paused: control.request('stop',0.)
        if args.manual_checkpoints or replay: control.checkpoint_mode(True,0.)
        lock=threading.Lock()
        empty=np.array([])
        shared=dict(manual_checkpoint_mode=args.manual_checkpoints,pose=localizer.get_pose(),wp_idx=0,planned_path=[],local_out=None,
                    operator_mode=getattr(control,'operator_mode',None),
                    assisted_control=bool(getattr(args,'reviewed_buttons',None)),
                    recovery_status='OBSERVE',mission=args.mission,cmd=(0.,0.,0.),cost_grid=grid.get_cost_grid(),
                    speed=0.,slam_converged=False,slam_map_size=0,slam_rms=0.,slam_icp_count=0,
                    fwd_cloud_xyz=(empty,empty,empty),policy_status='rocky_slope',trajectory=[])
        if display_reference is not None:
            shared['terrain_display']=display_reference
        nav=SimpleNamespace(_sim_localizer=localizer,_lidar=sensors['lidar'],_env_idx=0,
                            _omap=grid,_lock=lock,shared=shared,requests=requests if not args.no_dashboard else None)
        if args.perception_view and not args.headless:
            from rexmi_rl.nav.perception_view import PerceptionView
            view=PerceptionView(nav,hz=args.overlay_hz,point_limit=args.lidar_display_points,
                                selected_path_only=getattr(args,'selected_path_only',False))
            view.layer=getattr(args,"survey_layer","terrain")
        if not args.headless:
            import omni.ui as ui
            remote=ui.Window('REXMI | Checkpoint control',width=420,height=300)
            with remote.frame:
                with ui.VStack(spacing=5):
                    ui.Label('FLAT SIDESTEP TRIAL | stopped policy handoffs' if side else 'ROCKY SLOPE | autonomous checkpoints')
                    status=ui.Label('Observing',word_wrap=True)
                    with ui.HStack():
                        ui.Button('STOP',clicked_fn=lambda:requests.put(('stop',0.,0.)))
                        ui.Button('RESUME AUTO',clicked_fn=lambda:requests.put(('resume',0.,0.)))
                    if not round_trip and getattr(control,'goal_mode','fixed')!='any_exit':
                        ui.Button('CLICK CHECKPOINT MODE',clicked_fn=lambda:requests.put(('checkpoint_mode',1.,0.)))
                        ui.Button('RESTORE MISSION (paused)',clicked_fn=lambda:requests.put(('checkpoint_mode',0.,0.)))
                    ui.Label('Manual controls suspend autonomy; release to stop.')
                    for label,vx,omega in [('Forward',.4,0.),('Back',-.4,0.),('Left',0.,.35),('Right',0.,-.35)]:
                        ui.Button(label,mouse_pressed_fn=lambda x,y,b,m,v=vx,w=omega:requests.put(('manual',v,w)),
                                  mouse_released_fn=lambda x,y,b,m:requests.put(('manual',0.,0.)))
        if not args.no_dashboard and not args.headless:
            from rexmi_rl.nav.dashboard import Dashboard
            dashboard=Dashboard(shared,lock,grid,[] if getattr(control,'goal_mode','fixed')=='any_exit' else waypoints,request_queue=requests,
                                selected_path_only=getattr(args,'selected_path_only',False))
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
        if loop_mode:
            from .crater_loop import CraterLoop, load_validated_route
            loop_mission=CraterLoop(grid,path.parent,args.r_rim,validated_route=
                load_validated_route(args.crater_ascent_route) if getattr(args,'crater_ascent_route',None) else None)
            if args.start_paused: loop_mission.operator_mode='stop'
        source_root=Path(__file__).resolve().parents[3]
        sources=[Path(__file__),Path(__file__).with_name('checkpoint_control.py'),Path(__file__).with_name('observed_map.py')]
        if loop_mode:
            sources += list(Path(__file__).parent.glob('*.py'))
            sources += [source_root/'scripts/navigate.py',source_root/'crater_auto_nav.sh',Path(__file__).parents[1]/'drive_adapter.py']
        if not args.headless:
            sources += [Path(__file__).with_name(name+'.py') for name in
                        ('dashboard','dashboard_telemetry','terrain_display','perception_view')]
        if ascent:
            sources += [Path(__file__).with_name(name+'.py') for name in ('ascent_control','ascent_map','ascent_routes','reachability')]
            if getattr(args, 'ascent_architecture', 'behavior_tree') == 'behavior_tree':
                sources += [Path(__file__).with_name(name+'.py') for name in ('ascent_behavior','behavior_tree','movement','staged_routes','swept_footprint','departure_diagnostics','route_exploration','command_selection','recovery_continuation','release_memory','recovery_intent','recovery_turn')]
            elif args.ascent_architecture == 'remote':
                sources += [Path(__file__).with_name(name+'.py') for name in ('remote_control','remote_routes','success_selection','footprint_costs','movement','release_memory','swept_footprint','staged_routes','turn_staging','traversal_objective','stage_graph','planning_snapshot','ascent_first','purple_corridor','climb_burden','wheelbase_demand','sustained_climb','exit_region','checked_turn')]
            sources += [Path(__file__).with_name('terrain_prior.py')]
            sources += [source_root/'scripts/navigate.py',Path(__file__).parents[1]/'drive_adapter.py']
        if round_trip:
            sources += [Path(__file__).with_name('mission.py'),Path(__file__).with_name('round_trip.py')]
        if getattr(args,'joint_log_dir',None):
            sources += [source_root/'scripts/joint_telemetry.py',Path(__file__).with_name('joint_diagnostics.py')]
        if getattr(args,'reviewed_buttons',None):
            sources += [Path(__file__).with_name('reviewed_buttons.py')]
        if getattr(args,'ascent_path_result',None):
            sources += [Path(__file__).with_name('saved_path_control.py')]
        if getattr(args,'ascent_unguarded',False):
            sources += [Path(__file__).with_name(name+'.py') for name in ('experimental_path_control','slope_cost','reverse_recovery')]
        provenance=dict(operator_replay=operator_replay,controller=args.nav_controller,seed=args.seed,policy_path=str(Path(checkpoint).resolve()),
                        policy_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                        sources={str(p.relative_to(source_root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                        arguments=vars(args))
        if getattr(args,'ascent_path_result',None):
            provenance['saved_path']=control.saved_path_metadata
            saved_path_bytes=Path(args.ascent_path_result).read_bytes()
            if hashlib.sha256(saved_path_bytes).hexdigest()!=control.saved_path_metadata['result_sha256']:
                raise ValueError('Inspected path result changed during launch')
            path.with_suffix('.path-result.json').write_bytes(saved_path_bytes)
        if ascent:
            provenance['capability_profile']=dict(asdict(control.capability),task=args.task,
                footprint_radius=round(radius,4),**{k:round(v,4) for k,v in shape.items()},
                terrain_information=terrain_prior['kind'] if terrain_prior else 'local_only',
                terrain_geometry_id=terrain_prior['geometry_id'] if terrain_prior else None)
            if getattr(args, 'ascent_architecture', '') == 'remote':
                provenance['capability_profile'].update(execution_protocol=control.protocol_version,
                    execution='persistent teleop buttons; experimental characterization',
                    routine_braking=False, turn_pulse_cutoff=None, hard_tip_up_z=.5,
                    emergency_speed_mps=.8, emergency_clear_mps=.3, emergency_clear_dwell_s=.5,
                    terminal_actual_speed_mps=1., stopping_forecast_deceleration_mps2=.4,
                    static_survey_support=control.static_prior_enabled,
                    static_survey_support_mode=('retained_static_coverage' if control.static_retention_enabled
                                                else 'survey_agreement'),
                    static_survey_agreement_m=None if control.static_retention_enabled else grid.MATCH_TOLERANCE)
            elif hasattr(control, 'protocol_version'):
                from .command_selection import FORWARD_SPEED_PROFILE
                from .ascent_map import DIRECTIONAL_EXIT_PROFILE
                provenance['capability_profile'].update(FORWARD_SPEED_PROFILE)
                provenance['capability_profile'].update(DIRECTIONAL_EXIT_PROFILE)
                if control.executor.uphill_speed_response_enabled:
                    from .movement import UPHILL_SPEED_RESPONSE_PROFILE
                    provenance['capability_profile'].update(UPHILL_SPEED_RESPONSE_PROFILE)
                if control.executor.release_memory.enabled:
                    from .release_memory import RELEASE_MEMORY_PROFILE
                    provenance['capability_profile'].update(RELEASE_MEMORY_PROFILE)
                if control.executor.release_memory.correction_release_enabled:
                    from .release_memory import CORRECTION_RELEASE_PROFILE
                    provenance['capability_profile'].update(CORRECTION_RELEASE_PROFILE)
                if control.recovery_continuation_enabled:
                    from .recovery_continuation import RECOVERY_CONTINUATION_PROFILE
                    provenance['capability_profile'].update(RECOVERY_CONTINUATION_PROFILE)
                provenance['capability_profile'].update(execution_protocol=control.protocol_version,
                    turn_staging_deg=12., steep_yaw_lock_deg=15., body_turn_pause_deg=25., body_turn_resume_deg=20.,
                    diagonal_ascent_preference_deg=[10.,30.], release_drive_seconds=control.executor.RELEASE_DRIVE_SECONDS,
                    release_drive_clock='cumulative deliberate forward output within primitive; braking excluded',
                    drive_release_response_seconds=control.executor.DRIVE_RESPONSE_SECONDS,
                    drive_release_progress_mps=control.executor.DRIVE_RELEASE_PROGRESS_MPS,
                    drive_release_dwell_seconds=control.executor.DRIVE_RELEASE_DWELL_SECONDS,
                    drive_release_max_sample_gap_seconds=control.executor.DRIVE_RESPONSE_MAX_GAP_SECONDS,
                    drive_release_contract='minimum 5 s cumulative forward output AND 1 s continuous deficiency of either axis or gradient progress below 0.10 m/s from complete 2 s current DRIVE window; low dwell may accrue before 5 s; brake/posture/alignment/release/new-request/invalid/gap resets short history; outer budgets unchanged; provisional response heuristic, not wheel-slip or probability measurement',
                    release_response_seconds=control.executor.RELEASE_RESPONSE_SECONDS,
                    release_min_progress_mps=control.executor.RELEASE_PROGRESS_MPS,
                    release_plateau_seconds=control.executor.RELEASE_PLATEAU_SECONDS,
                    release_plateau_contract='complete 2 s measurement and release response before 5 continuous seconds with either axis or terrain-gradient progress below 0.01 m/s; outer budgets unchanged',
                    release_drift_contract='physical swept support, freshness and exclusions; commanded grade checked separately',
                    release_brake_resume='checked suspended release; persistent traversal progress across braking',
                    posture_pause_seconds=control.executor.POSTURE_PAUSE_SECONDS,
                    posture_recovery_seconds=control.executor.POSTURE_RECOVERY_SECONDS,
                    posture_relative_envelope_deg=control.executor.POSTURE_RELATIVE_ENVELOPE_DEG,
                    posture_absolute_pause_deg=control.executor.POSTURE_ABSOLUTE_PAUSE_DEG,
                    posture_exemption_contract='provisional; already executed forward uphill mission DRIVE/RELEASE only; fresh consistent world attitude and observed fit; no active-pause cancellation; all other protections retained',
                    posture_timeout='censored unless measured deterioration continues; geometry protections and measured tip/backslide failures retained',
                    straight_admission_window_s=.3, straight_admission_heading_excursion_rad=.10,
                    straight_admission_max_sample_gap_s=.04,
                    arrival_heading_inner_radius_fraction=control.executor.ARRIVAL_HEADING_RESERVE,
                    arrival_heading_contract='gentle straight body-axis ray reaches unchanged arrival region; all motion and settling guards retained',
                    straight_obstacle_contract='continuous configured body sweep; circular terrain support and global planning retained',
                    stopping_heading_contract='0.30 s measured heading envelope and net-trend forecast; all fan poses; unavailable history rejects oriented fallback',
                    straight_admission_forecast='observed heading envelope plus net-trend stopping fan; rejection censored',
                    shelf_shortcut_max_checks=16, shelf_shortcut_min_saved_m=.5,
                    straight_admission_speed_mps=.25, straight_admission_opposing_speed_mps=.10,
                    exclusion_direction_source='locally executed traversal; no planned-direction fallback',
                    uphill_patience_seconds=control.executor.UPHILL_PATIENCE_SECONDS,
                    uphill_exposure_limit_seconds=control.executor.UPHILL_PATIENCE_SECONDS,
                    uphill_wall_limit_seconds=control.executor.UPHILL_WALL_SECONDS,
                    uphill_exposure_clock='held emitted forward DRIVE or RELEASE intervals counted once at next observation; BRAKE and POSTURE excluded; clipped to fixed low-progress epoch',
                    uphill_expiry_contract='first of exposure or wall limit; zero deliberate exposure is censored; independent physical failures retained',
                    uphill_renewal_m=control.executor.UPHILL_RENEWAL_METERS,
                    uphill_renewal_contract='net advance from fixed epoch origin; transitions cannot renew either budget',
                    max_checked_lateral_tolerance_m=1., max_arrival_radius_m=.3)
                provenance['capability_profile'].update(
                    recovery_stage_seconds=control.executor.RECOVERY_STAGE_SECONDS,
                    recovery_stage_dwell_seconds=control.executor.RECOVERY_STAGE_DWELL_SECONDS,
                    recovery_stage_min_displacement_m=control.executor.RECOVERY_STAGE_MIN_DISPLACEMENT,
                    recovery_stage_max_drift_m=control.executor.RECOVERY_STAGE_MAX_DRIFT,
                    recovery_stage_contract='recovery only after actual drive and >=0.4 m displacement; fresh unobstructed and exclusion-free circular footprint <=12 degrees, body <=20 degrees, checked stopping; brake only; contiguous 0.6 s stable dwell; censored reposition for mission replanning, no goal or success credit; original request and recovery deadlines retained')
            if getattr(args,'ascent_unguarded',False):
                provenance['capability_profile'].update(motion_vetoes=False,
                    reverse_recovery_ceiling_mps=1.,
                    hard_tip_up_z=None,emergency_speed_mps=None,terminal_actual_speed_mps=None,
                    stopping_forecast_deceleration_mps2=None,
                    execution='User-authorized unguarded simulation; measured-progress reverse/turn/push recovery')
                provenance['capability_profile']['physical_climb_failure_classifier']=False
            provenance['terrain_prior']=terrain_prior
        if ascent and control.model.artifact is not None and control.model.artifact.get('policy_sha256') != provenance['policy_sha256']:
            raise ValueError('Reachability model was not characterized with this policy checkpoint')
        source_id=hashlib.sha256(json.dumps(provenance['sources'],sort_keys=True).encode()).hexdigest()
        if ascent and args.ascent_mode=='qualified' and control.model.artifact is not None and control.model.artifact.get('source_id')!=source_id:
            raise ValueError('Reachability model was not validated with this controller source version')
        if ascent and args.ascent_mode=='qualified' and control.model.artifact is not None and control.model.artifact.get('capability_profile')!=provenance['capability_profile']:
            raise ValueError('Reachability model was not validated with this environment and capability profile')
        if hasattr(control, 'probability_domain_validated'):
            from .success_selection import validated_domain
            control.probability_domain_validated = validated_domain(control.model,
                provenance['policy_sha256'], source_id, provenance['capability_profile'])
            provenance['probability_ranking_domain_validated'] = control.probability_domain_validated
        path.with_suffix('.provenance.json').write_text(json.dumps(provenance,indent=2))
        if initial_survey is not None:
            np.savez_compressed(path.with_suffix('.prior.npz'),**initial_survey)
            initial_survey=None
        if ascent or loop_mode:
            for source in sources:
                destination=path.with_suffix('.sources')/source.relative_to(source_root)
                destination.parent.mkdir(parents=True,exist_ok=True)
                destination.write_bytes(source.read_bytes())
        dt=env.unwrapped.step_dt
        if round_trip or loop_mode:
            args.max_steps=min(args.max_steps,math.ceil(DEADLINE_SECONDS/dt)+1)
        mission_events=[]
        previous_stage=None
        reviewed = None
        if getattr(args,'reviewed_buttons',None):
            from .reviewed_buttons import ReviewedButtons
            reviewed = ReviewedButtons(args.reviewed_buttons, grid, dt)
            control.intervened = True
        departure=None
        event_snapshots=None
        if getattr(args,'ascent_event_snapshots',False):
            from rexmi_rl.nav.departure_diagnostics import EventSnapshots
            event_snapshots=EventSnapshots(control,path,dict(source_id=source_id,
                policy_sha256=provenance['policy_sha256'],seed=args.seed,
                protocol=control.protocol_version,sources=provenance['sources']))
            control.executor.event_observer=event_snapshots.observe
            control.executor.brake_evaluator=control._brake_evaluation
        if getattr(args,'capture_ascent_departure',False):
            from rexmi_rl.nav.departure_diagnostics import DepartureCapture
            departure=DepartureCapture(control,path,dict(source_id=source_id,policy_sha256=provenance['policy_sha256'],
                seed=args.seed,protocol=control.protocol_version,sources=provenance['sources']),
                purpose=getattr(args,'capture_ascent_departure_purpose','any'))
            control.executor.departure_observer=departure.observe
            args.max_steps=min(args.max_steps,math.ceil(75./dt))
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
            snapshot=prior_snapshot(grid) if terrain_prior else grid.survey_snapshot()
            snapshot.update(mission_route=np.asarray(control.mission_route,dtype=float).reshape(-1,2),
                            checkpoint_route=np.asarray(control.path,dtype=float).reshape(-1,2))
            export_job=exporter.submit(write_survey,snapshot)
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
        if getattr(args,'ascent_exploration_history',None):
            from rexmi_rl.nav.route_exploration import load_coverage, merge_coverage
            coverage=merge_coverage([load_coverage(run,grid,provenance,properties) for run in args.ascent_exploration_history])
            control.configure_exploration(coverage)
            if getattr(args,'ascent_launch_variant',False):
                control.configure_launch_variant()
                provenance['launch_comparison'] = dict(experiment='paired_launch_variant',comparison='same_climb_class',
                    selection='initial only; fewer compacted pre-climb bends and flatter full-footprint launch; original cost ranking',
                    no_variant='diagnostic HOLD; no baseline fallback',qualified=False)
            provenance['route_exploration']=coverage
            path.with_suffix('.provenance.json').write_text(json.dumps(provenance,indent=2))
        trajectory=[]
        last_export=-30.
        exported_failures=0
        exported_motion_events=0
        exported_escape_events=0
        last_route_choice=None
        map_update_ms=[]
        active_policy='rocky_slope'
        previous=None
        switch_event_count=0
        result='step_limit'
        max_error=0.
        now=0.
        capture=None
        exported_attempts=0
        control_ms=[]
        recovery_control_ms={name:[] for name in ('admission','continued_reverse','core_observation','termination')}
        recovery_admission_timing=[]
        min_up=1.
        min_active_up=1.
        peak_active_speed=0.
        slope_overspeed_seconds=0.
        peak_slope_speed=0.
        peak_speed=0.
        peak_command=0.
        print(f"[checkpoint] {'flat sidestep trial' if side else 'rocky_slope only'}; envelope={radius:.2f}m + 0.10m clearance; log={path}",flush=True)
        if getattr(args,'joint_log_dir',None):
            from .joint_diagnostics import NavigationJointTelemetry
            from joint_telemetry import fingerprint
            joint_logger = NavigationJointTelemetry(args.joint_log_dir,env.unwrapped,
                metadata=dict(navigation_provenance=provenance,
                    robot_asset=fingerprint(robot.cfg.spawn.usd_path),
                    sampling_alignment='Navigation startup settling included; navigation CSV is pre-step. '
                        'Join physics rows and raw policy_action events by control_step/nav_time_s. '
                        'Raw policy output is before wrapper clipping; actual joint targets are in physics rows.'))
        with path.open('w',newline='',buffering=1) as f:
            writer=csv.writer(f)
            writer.writerow(['step','sim_time','x','y','z','yaw','speed','up_z','wp_idx','state','reason',
                             'vx_cmd','omega_cmd','checkpoint_x','checkpoint_y','target_x','target_y',
                             'heading_error','cross_track','coverage','oldest_observation_age','policy',
                             'slope_deg','roughness_m','climb_failures','backslide_m','route_decision','render_ms','vy_cmd','sidestep_phase','switch_reason',
                             'mission_route_m','route_max_uphill_deg','route_max_side_slope_deg','route_uphill_over_25deg_m','turn_repositions','mission_route_complete',
                             'motion_status','measured_forward_mps','net_progress_mps','lateral_mps','learned_no_go_cells','route_candidates',
                             'route_purpose','retreat_goal_x','retreat_goal_y','max_uphill_deg','max_side_slope_deg',
                             'behavior_branch','movement_state','action_id','support_inclination_deg','actual_yaw_rate','body_tilt_deg',
                             'directional_grade_deg','uphill_progress_mps','release_active','forward_exposure_s',
                             'velocity_world_x','velocity_world_y','velocity_world_z',
                             'uphill_traversal_progress_mps','uphill_traversal_window_s','uphill_budget_remaining_s',
                             'uphill_exposure_s','uphill_wall_remaining_s',*POSTURE_TELEMETRY_COLUMNS,
                             'terrain_relative_tilt_deg','posture_raw_rapid','posture_entry_exempt','posture_absolute_pause',
                             'release_window_complete','release_plateau_s','drive_response_window_s',
                             'drive_response_axis_mps','drive_response_uphill_mps','drive_slow_dwell_s',
                             'mission_stage','mission_goal_x','mission_goal_y'])
            for step in range(args.max_steps):
                if not app.is_running():
                    result='window_closed';break
                now=step*dt
                pose=localizer.get_pose()
                xy=(pose.x,pose.y)
                velocity=localizer.get_velocity()
                speed=math.hypot(velocity[0],velocity[1])
                q=robot.data.root_quat_w[0]
                body_quaternion=tuple(q.detach().cpu().tolist())
                up=1.-2.*float(q[1]**2+q[2]**2)
                if (round_trip or loop_mode) and deadline_reached(now,dt):
                    result='mission_deadline';break
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
                    live_display=None
                    if not args.headless and display_reference is None:
                        live_display=cost_view(grid.cost,grid.known,grid.blocked)
                        if loop_mode:
                            from .descent_planning import refresh as refresh_descent_costs, measured_coverage
                            refresh_descent_costs(control)
                            display_model=control.descent_model
                            live_display=from_model(display_model,observed_mask=measured_coverage(grid))
                    with lock:
                        if live_display is not None: shared['terrain_display']=live_display
                        shared.update(cost_grid=grid.get_cost_grid(),observed_mask=grid.known.copy(),clearance_mask=grid.clearance_mask)
                        shared['survey_layers']={k:getattr(grid,k).copy() for k in
                                                 ('slope','roughness','failures','successes','no_go','approach_blocked')}
                        if ascent or loop_mode:
                            shared['survey_layers']['shelf_mask']=grid.shelf_mask.copy()
                            from .footprint_costs import FIELDS
                            shared['survey_layers'].update({k:getattr(grid,k).copy() for k in FIELDS})
                    if args.scan_log:
                        scan_dir=Path(args.scan_log)
                        scan_dir.mkdir(parents=True,exist_ok=True)
                        np.savez_compressed(scan_dir/f'{step:06d}.npz',**scan_record)
                while replay_index<len(replay) and replay[replay_index][0]<=now+1e-8:
                    _,(tx,ty)=replay[replay_index]
                    requests.put(('target',tx,ty));replay_index+=1
                while operator_replay_index<len(operator_replay) and operator_replay[operator_replay_index][0]<=now+1e-8:
                    scheduled,action=operator_replay[operator_replay_index]
                    requests.put((action,0.,0.));operator_replay_index+=1
                    with path.with_suffix('.events.jsonl').open('a') as events:
                        events.write(json.dumps(dict(event='operator_replay',time=now,scheduled_time=scheduled,action=action))+'\n')
                while not requests.empty() and reviewed is None:
                    action,vx,omega=requests.get()
                    event=apply_operator_request(control,side,action,vx,omega,xy,pose.yaw,now,args.no_dashboard,
                                                 locked_mission=round_trip or loop_mode or getattr(control,'goal_mode','fixed')=='any_exit')
                    if loop_mission is not None: loop_mission.request(action)
                    if event is not None:
                        with path.with_suffix('.events.jsonl').open('a') as events:
                            events.write(json.dumps(event)+'\n')
                        print(f'[checkpoint click {now:.2f}s] {control.selection_status}',flush=True)
                    with lock: shared['manual_checkpoint_mode']=getattr(control,'manual_checkpoint_mode',False)
                    if dashboard and getattr(control,'goal_mode','fixed')!='any_exit':dashboard._waypoints=control.waypoints
                if loop_mission is not None:
                    previous_control=control
                    control_start=time.perf_counter()
                    control,motion=loop_mission.tick(control,xy,pose.yaw,speed,now,up,velocity)
                    control_ms.append((time.perf_counter()-control_start)*1000)
                    if control is not previous_control:
                        ascent=True
                        exported_attempts=exported_failures=exported_motion_events=exported_escape_events=0
                        args.goal_x,args.goal_y=control.selected_exit
                        waypoints.append(control.waypoints[0])
                        display_model=loop_mission.plan.model
                        if not args.headless:
                            display_reference=from_model(display_model)
                            with lock: shared['terrain_display']=display_reference
                        if dashboard: dashboard._waypoints=control.waypoints
                        path.with_suffix('.loop.json').write_text(json.dumps(loop_mission.report(),indent=2)+'\n')
                elif reviewed is not None:
                    motion = reviewed.next_motion(now, (pose.x,pose.y,pose.z), pose.yaw,
                                                   velocity, up, app.is_running, requests)
                    control.state = 'REVIEWED_BUTTONS'
                    control.reason = reviewed.pause_reason
                    if motion is None:
                        result = reviewed.terminal
                        min_up=min(min_up,up);peak_speed=max(peak_speed,speed)
                        if now>=3.:
                            min_active_up=min(min_active_up,up);peak_active_speed=max(peak_active_speed,speed)
                        break
                    control.reason = 'agent-assisted '+reviewed.active['button']
                elif side and side.active:
                    control.velocity_world=np.asarray(velocity[:2])
                    angular=robot.data.root_ang_vel_b[0].detach().cpu().numpy()
                    motion=side.tick(control,xy,pose.yaw,velocity[:2],float(angular[2]),float(np.linalg.norm(angular)),now,up,vertical_speed=float(velocity[2]))
                    control.previous_command=motion
                    control.last_tick=now
                    control.progress_at=now
                    control.drive_samples=[]
                    control.slip_tracker.reset()
                else:
                    executor=getattr(control,'executor',None)
                    previous_region=(getattr(executor,'recovery_continuation',None)
                                     if executor is not None and executor.request is not None else None)
                    control_start=time.perf_counter()
                    attitude={'body_quaternion':body_quaternion} if hasattr(control,'executor') else {}
                    motion=control.tick(xy,pose.yaw,speed,now,upright=up,velocity=velocity,**attitude)
                    elapsed_control_ms=(time.perf_counter()-control_start)*1000
                    control_ms.append(elapsed_control_ms)
                    region=getattr(executor,'recovery_continuation',None)
                    if region is not None and (previous_region is not None or executor.request is not None):
                        phase=('admission' if previous_region is None else 'termination' if executor.request is None
                               else 'core_observation' if executor.recovery_core_started is not None else 'continued_reverse')
                        recovery_control_ms[phase].append(elapsed_control_ms)
                        if phase=='admission':recovery_admission_timing.append(dict(time=now,ms=elapsed_control_ms))
                    if departure is not None and departure.attempted:
                        departure.complete_tick(motion,pose,velocity)
                        departure.submit(exporter)
                    if side and side.maybe_start(control,xy,pose.yaw,now,up):
                        angular=robot.data.root_ang_vel_b[0].detach().cpu().numpy()
                        motion=side.tick(control,xy,pose.yaw,velocity[:2],float(angular[2]),float(np.linalg.norm(angular)),now,up,vertical_speed=float(velocity[2]))
                if event_snapshots is not None:
                    event_snapshots.complete_tick(motion,pose,velocity)
                mission_status=loop_mission.status() if loop_mission is not None else stage_status(waypoints,control.wp_idx) if round_trip else None
                if getattr(control,'goal_mode','fixed')=='any_exit':
                    from rexmi_rl.nav.exit_region import mission_status as exterior_status
                    if not loop_mode: mission_status=exterior_status(control)
                    if dashboard:
                        dashboard._waypoints=([Waypoint(*control.selected_exit,.4,'Selected safe exit')]
                                             if control.selected_exit else [])
                if mission_status and mission_status['stage'] != previous_stage:
                    event=dict(event='mission_stage',time=now,position=list(xy),**mission_status)
                    mission_events.append(event)
                    with path.with_suffix('.events.jsonl').open('a') as events:
                        events.write(json.dumps(event)+'\n')
                    previous_stage=mission_status['stage']
                active_policy=side.policy if side else 'rocky_slope'
                min_up=min(min_up,up);peak_speed=max(peak_speed,speed);peak_command=max(peak_command,abs(motion.vx))
                if now>=3.:
                    min_active_up=min(min_active_up,up);peak_active_speed=max(peak_active_speed,speed)
                    centre=grid.world_to_cell(*xy)
                    if centre is not None and grid.known[centre] and grid.slope[centre]>math.tan(math.radians(5)):
                        peak_slope_speed=max(peak_slope_speed,speed)
                        if speed>.45:slope_overspeed_seconds+=dt
                if ascent:
                    with lock: shared['navigation_status']=control.navigation_status()
                    if len(control.attempt_events)>exported_attempts:
                        with path.with_suffix('.events.jsonl').open('a') as events:
                            for event in control.attempt_events[exported_attempts:]:
                                events.write(json.dumps(event)+'\n')
                        exported_attempts=len(control.attempt_events)
                vy=getattr(motion,'vy',0.)
                if side and len(side.events)>switch_event_count:
                    with path.with_suffix('.events.jsonl').open('a') as events:
                        for event in side.events[switch_event_count:]: events.write(json.dumps(event)+'\n')
                    switch_event_count=len(side.events)
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
                if len(control.slip_events)>exported_motion_events:
                    with path.with_suffix('.events.jsonl').open('a') as events:
                        for event in control.slip_events[exported_motion_events:]:
                            events.write(json.dumps(event)+'\n')
                    exported_motion_events=len(control.slip_events)
                if len(control.escape_events)>exported_escape_events:
                    with path.with_suffix('.events.jsonl').open('a') as events:
                        for event in control.escape_events[exported_escape_events:]:
                            events.write(json.dumps(event)+'\n')
                    exported_escape_events=len(control.escape_events)
                route_key=route_comparison_key(control)
                if control.route_searches and route_key!=last_route_choice:
                    with path.with_suffix('.events.jsonl').open('a') as events:
                        events.write(json.dumps(dict(event='route_comparison',time=now,position=xy,
                            decision=control.route_decision,candidates=control.route_options,searches=control.route_searches))+'\n')
                last_route_choice=route_key
                motion_metrics=control.motion_metrics()
                posture_current=motion_metrics.get('posture_signal_time')==now
                quality=grid.quality(xy)
                max_error=max(max_error,control.tracking_error)
                if not trajectory or math.dist(xy,trajectory[-1])>.1: trajectory.append(xy)
                cp=control.checkpoint or (math.nan,math.nan)
                target=control.target or (math.nan,math.nan)
                if not args.headless:
                    grade_grid=display_model.grid if display_model is not None else grid
                    grade=heading_grade(grade_grid,xy,pose.yaw,display_model)
                    display_yaw_rate=float('nan')
                    if display_last_heading is not None and now>display_last_heading[0]:
                        delta=pose.yaw-display_last_heading[1]
                        display_yaw_rate=math.atan2(math.sin(delta),math.cos(delta))/(now-display_last_heading[0])
                    display_last_heading=(now,pose.yaw)
                    # Start the climb profile at first commanded motion, after spawn settling.
                    if travel_history.time is not None or abs(motion.vx)+abs(vy)+abs(motion.omega)>1e-6:
                        travel_history.update(now,(pose.x,pose.y,pose.z),grade)
                with lock:
                    if not args.headless:
                        shared.update(sim_time=now,actual_yaw_rate=display_yaw_rate,
                                      travel_distance=travel_history.distance)
                        if step%max(1,round(.1/dt))==0:
                            shared['travel_profile']=travel_history.snapshot()
                    shared.update(pose=pose,wp_idx=control.wp_idx,planned_path=list(control.path),
                                  operator_mode=getattr(control,'operator_mode',None),
                                  mission_stage=mission_status,
                                  mission_path=list(control.mission_route),route_profile=dict(control.route_profile),
                                  mission_route_complete=control.mission_route_complete,
                                  route_decision=control.route_decision,
                                  route_purpose=control.route_purpose,retreat_goal=control.retreat_goal,
                                  grade_limits=(control.preferred_uphill,grid.uphill_limit_deg,grid.side_limit_deg),
                                  slip_metrics=motion_metrics,route_options=list(control.route_options),
                                  recovery_status=control.state,cmd=(motion.vx,vy,motion.omega),
                                  policy_status=active_policy,switch_reason=side.reason if side else 'single policy',
                                  sidestep_phase=side.phase if side else 'disabled',
                                  speed=speed,trajectory=list(trajectory),
                                  checkpoint=control.checkpoint,steering_target=control.target,
                                  nav_reason=control.reason,coverage=quality['coverage'],observation_age=quality['age'],
                                  selection_status=getattr(control,'selection_status',''),
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
                             f'policy {active_policy} | command x {motion.vx:+.2f}, y {vy:+.2f} m/s, yaw {motion.omega:+.2f} rad/s\n'
                             f'{side.phase + ": " + side.reason if side else "single policy"}\n'
                             f'mapped {quality["mapped_coverage"]:.0%} | fresh {quality["coverage"]:.0%}\n'
                             f'surveyed {quality["mapped_area"]:.1f} m² | oldest local observation {quality["age"]:.1f}s')
                if ascent:
                    estimate=control.reachability or {}
                    probability=estimate.get('probability')
                    description += (f'\nreachability {probability:.0%}' if probability is not None else '\nreachability unvalidated')
                    description += f' | {control.mode} | command limit {control.command_limit:.2f} m/s'
                    if hasattr(control, 'button_state'):
                        description += f' | REMOTE {control.button_state}'
                        from .success_selection import selection_text
                        explanation = selection_text(control.route_options)
                        if explanation:
                            description += '\n'+explanation
                    if hasattr(control, 'executor'):
                        description += f'\nBT {control.tree_branch} | FSM {control.executor.state.value}'
                if mission_status:
                    description=f'MISSION {mission_status["stage"]} | goal {mission_status["goal"] or mission_status.get("goal_description", "")}\n'+description
                if not args.headless: status.text=description
                if (control.state,active_policy,side.phase if side else '')!=previous or step%250==0:
                    print(f'[checkpoint {now:.1f}s] {description.replace(chr(10)," | ")} pos={xy}',flush=True)
                    previous=(control.state,active_policy,side.phase if side else '')
                writer.writerow([step,now,pose.x,pose.y,pose.z,pose.yaw,speed,up,control.wp_idx,control.state,
                                 control.reason,motion.vx,motion.omega,*cp,*target,control.heading_error,
                                 control.tracking_error,quality['coverage'],quality['age'],active_policy,
                                 float(np.degrees(np.arctan(grid.slope[grid.world_to_cell(*xy)]))),
                                 float(grid.roughness[grid.world_to_cell(*xy)]),control.climb_failures,
                                 control.backslide_distance,control.route_decision,view.render_ms if view else 0.,vy,side.phase if side else '',side.reason if side else '',
                                 control.route_profile.get('length_m',0.),control.route_profile.get('max_uphill_deg',0.),
                                 control.route_profile.get('max_side_slope_deg',0.),control.route_profile.get('uphill_over_25deg_m',0.),
                                 control.turn_repositions,control.mission_route_complete,
                                 motion_metrics['status'],motion_metrics['forward_mps'],
                                 motion_metrics['progress_mps'],motion_metrics['lateral_mps'],
                                 int(grid.no_go.sum()),len(control.route_options),control.route_purpose,
                                 *(control.retreat_goal or (math.nan,math.nan)),grid.uphill_limit_deg,grid.side_limit_deg,
                                 getattr(control,'tree_branch',''),getattr(getattr(control,'executor',None),'state',getattr(control,'button_state','')),
                                 getattr(getattr(getattr(control,'executor',None),'request',None),'action_id',''),
                                 getattr(getattr(control,'obs',None),'inclination',math.nan),
                                 getattr(getattr(control,'obs',None),'yaw_rate',math.nan),
                                 getattr(getattr(control,'obs',None),'body_tilt',math.nan),
                                 motion_metrics.get('directional_grade_deg',math.nan),
                                 motion_metrics.get('uphill_progress_mps',math.nan),motion_metrics.get('release_active',False),
                                 motion_metrics.get('forward_exposure_s',0.), *velocity[:3],
                                 motion_metrics.get('uphill_traversal_progress_mps',math.nan),
                                 motion_metrics.get('uphill_traversal_window_s',0.),
                                 motion_metrics.get('uphill_budget_remaining_s',math.nan),
                                 motion_metrics.get('uphill_exposure_s',0.),
                                 motion_metrics.get('uphill_wall_remaining_s',math.nan),
                                 *posture_telemetry(body_quaternion,
                                    robot.data.root_ang_vel_w[0].detach().cpu().tolist(),getattr(control,'obs',None)),
                                 getattr(getattr(control,'obs',None),'terrain_relative_tilt',None),
                                 motion_metrics.get('posture_raw_rapid') if posture_current else None,
                                 motion_metrics.get('posture_entry_exempt') if posture_current else None,
                                 motion_metrics.get('posture_absolute_pause') if posture_current else None,
                                 motion_metrics.get('release_window_complete') if posture_current else None,
                                 motion_metrics.get('release_plateau_s') if posture_current else None,
                                 motion_metrics.get('drive_response_window_s') if posture_current else None,
                                 motion_metrics.get('drive_response_axis_mps') if posture_current else None,
                                 motion_metrics.get('drive_response_uphill_mps') if posture_current else None,
                                 motion_metrics.get('drive_slow_dwell_s') if posture_current else None,
                                 mission_status['stage'] if mission_status else '',
                                 *(mission_status['goal'] if mission_status and mission_status['goal'] else (None,None))])
                if departure is not None and departure.attempted:
                    result='departure_diagnostic';break
                if (control.state=='COMPLETE' and not getattr(control,'manual_checkpoint_mode',False)
                        and (loop_mission is None or loop_mission.phase=='COMPLETE')): result='complete';break
                if args.headless and control.state=='HOLD' and not (side and side.active) and replay_index>=len(replay) and operator_replay_index>=len(operator_replay):
                    result='selected_checkpoint_reached' if control.reason.startswith('selected checkpoint reached') else 'hold'
                    break
                actions=policy_actions(env,policies[active_policy],motion.vx,motion.omega,vy=vy)
                if joint_logger is not None:
                    joint_logger.begin_navigation_step(step,now,active_policy,
                        getattr(getattr(getattr(control,'executor',None),'state',None),'value',control.state),
                        getattr(getattr(getattr(control,'executor',None),'request',None),'action_id',None),
                        motion.vx,motion.omega,actions)
                if event_snapshots is not None:event_snapshots.submitted()
                if joint_logger is not None:joint_logger.submitted()
                _,_,done,_=env.step(actions)
                completed_physics_diagnostics(event_snapshots,exporter,joint_logger,done,env.unwrapped)
                if bool(done[0]):
                    result='unexpected_environment_reset'
                    if reviewed is not None:
                        reviewed.terminal=result
                        reviewed.finish(now+dt, (pose.x,pose.y,pose.z), result)
                    break
        if ascent:
            if event_snapshots is not None:event_snapshots.flush(exporter,reason=result)
            control.finish_attempt('censored',result,xy,pose.yaw,now)
            with path.with_suffix('.events.jsonl').open('a') as events:
                for event in control.attempt_events[exported_attempts:]:events.write(json.dumps(event)+'\n')
        summary=dict(result=result,seed=args.seed,sim_seconds=now,waypoints_reached=control.wp_idx,
                     diagnostic=bool(departure),
                     controller=args.nav_controller,mission=args.mission,terrain_prior=terrain_prior,
                     ascent_architecture=getattr(args,'ascent_architecture','legacy') if ascent else None,
                     perception_overlay_requested=bool(args.perception_view and not args.headless),
                     perception_overlay_enabled=bool(view and view.enabled),
                     perception_overlay_tiles=getattr(view,'_tile_count',0),
                     spawn=[args.spawn_x,args.spawn_y,args.spawn_z,args.spawn_yaw],goal=[args.goal_x,args.goal_y],
                     ascent_mode=args.ascent_mode if ascent else None,
                     model_id=control.model.model_id if ascent else None,
                     probability_model_qualified=(bool(control.model.artifact and control.model.artifact.get('evaluation',{}).get('qualified')) if ascent else False),
                     source_id=source_id,
                     policy_sha256=provenance['policy_sha256'],
                     min_up_z=min_up,peak_actual_speed_mps=peak_speed,peak_command_mps=peak_command,
                     min_active_up_z=min_active_up,peak_active_speed_mps=peak_active_speed,
                     peak_slope_speed_mps=peak_slope_speed,slope_overspeed_seconds=slope_overspeed_seconds,
                     controller_ms_mean=float(np.mean(control_ms)) if control_ms else 0.,
                     controller_ms_p95=float(np.percentile(control_ms,95)) if control_ms else 0.,
                     recovery_controller_timing_ms={name:dict(count=len(values),
                         mean=float(np.mean(values)) if values else None,
                         p95=float(np.percentile(values,95)) if values else None,
                         maximum=max(values) if values else None) for name,values in recovery_control_ms.items()},
                     recovery_admission_timing=recovery_admission_timing,
                     checkpoint_replay=getattr(args,'checkpoint_replay',None),replayed_selections=replay_index,
                     operator_replay=operator_replay,replayed_operator_events=operator_replay_index,
                     total_waypoints=len(control.waypoints),manual_intervention=control.intervened,
                     max_cross_track_m=max_error,policy=active_policy,policy_switches=side.switches if side else 0,
                     flat_sidesteps=bool(side),switch_events=side.events if side else [],
                     footprint_radius_m=radius,cruise_ceiling=args.cruise_speed,
                     body_envelope=shape,
                     hold_failures=control.hold_failures,rolling_checkpoints=control.rolling_checkpoints,
                     final_state=control.state,final_reason=control.reason,
                     local_recovery_attempts=(getattr(control,'recovery_count',0) if getattr(args,'ascent_unguarded',False) else sum(e['event']=='maneuver_start' for e in control.escape_events)
                                              if ascent else len(control.recovery_attempts)),
                     feedback_recovery_history=getattr(control,'recovery_history',None),
                     feedback_recovery_active=getattr(control,'recovery',None),
                     furthest_saved_route_progress_m=getattr(control,'saved_progress',None),
                     final_saved_route_progress_m=getattr(control,'actual_route_distance',None),
                     last_hold_failure=control.last_hold_failure,
                     climb_failures=None if getattr(args,'ascent_unguarded',False) else control.climb_failures,
                     climb_failure_classification_available=not getattr(args,'ascent_unguarded',False),
                     backslide_m=control.backslide_distance,
                     turn_repositions=control.turn_repositions,route_profile=control.route_profile,
                     motion_events=control.slip_events,escape_events=control.escape_events,
                     failed_climbs=grid.failure_records if ascent else grid.failed_climbs,route_purpose=control.route_purpose,
                     retreat_goal=control.retreat_goal,retreat_candidates=control.retreat_candidates,
                     preferred_uphill_deg=control.preferred_uphill,max_uphill_deg=grid.uphill_limit_deg,
                     max_side_slope_deg=grid.side_limit_deg,
                     learned_no_go_cells=int(grid.no_go.sum()),route_options=control.route_options,route_searches=control.route_searches,
                     mission_route_complete=control.mission_route_complete,
                     map_update_ms_mean=float(np.mean(map_update_ms)) if map_update_ms else 0.,
                     validated=False)
        export_survey()
        if getattr(control,'goal_mode','fixed') == 'any_exit':
            summary.update(goal=None,goal_mode='any_exit',selected_exit=control.selected_exit,
                           exit_rim_radius_m=control.exit_rim_radius)
        if round_trip:
            summary.update(round_trip=stage_status(waypoints,control.wp_idx),
                           mission_stages=mission_events,mission_deadline_seconds=DEADLINE_SECONDS,
                           destinations=[dict(x=wp.x,y=wp.y,radius=wp.arrival_radius,label=wp.label) for wp in waypoints],
                           counts_as_qualification=False,
                           counts_as_mission=result=='complete' and not control.intervened)
        if loop_mission is not None:
            report=loop_mission.report()
            descent_recoveries=(report['descent'] or {}).get('steering_recoveries',
                sum(e['event']=='descent_turn_stall' for e in control.escape_events))
            descent_reverse=(report['descent'] or {}).get('reverse_recoveries',
                sum(e['event']=='descent_reverse_start' for e in control.escape_events))
            summary.update(crater_loop=report,mission_stages=mission_events,
                waypoints_reached=report['completed_destinations'],total_waypoints=2,
                descent_steering_recoveries=descent_recoveries,
                descent_reverse_recoveries=descent_reverse,reverse_recovery_ceiling_mps=1.,
                ascent_feedback_recoveries=getattr(control,'recovery_count',0),
                local_recovery_attempts=descent_reverse+getattr(control,'recovery_count',0),
                mission_deadline_seconds=DEADLINE_SECONDS,
                counts_as_mission=result=='complete' and not (control.intervened or loop_mission.intervened),
                counts_as_qualification=False)
            path.with_suffix('.loop.json').write_text(json.dumps(report,indent=2)+'\n')
        if reviewed is not None:
            summary.update(reviewed_buttons=reviewed.report(), counts_as_mission=False,
                           counts_as_qualification=False, manual_intervention=True)
        if departure is not None:
            summary.update(counts_as_mission=False,counts_as_qualification=False)
            summary['departure_capture']=(departure.job.result() if departure.job is not None else
                dict(ok=False,error=departure.error or 'No lateral rejection before diagnostic ceiling'))
        if event_snapshots is not None:
            summary['event_snapshots']=event_snapshots.report()
        if joint_logger is not None:summary['joint_diagnostics']=joint_logger.report()
        path.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2))
        print('[checkpoint] RESULT '+json.dumps(summary),flush=True)
    except Exception as exc:
        record_run_failure(path,exc,locals().get('now'),locals().get('source_id'))
        raise
    finally:
        if joint_logger is not None:
            try:
                joint_logger.close()
            except Exception as exc:
                joint_logger.final_error=f'{type(exc).__name__}: {exc}'
                print(f'[joint diagnostics] close failed: {exc}')
            try:
                path.with_suffix('.joints.json').write_text(json.dumps(joint_logger.report(),indent=2))
            except Exception as exc:print(f'[joint diagnostics] final export failed: {exc}')
        if 'event_snapshots' in locals() and event_snapshots is not None:
            try:
                event_snapshots.flush(exporter,reason='run ended or raised before completed physics step')
                path.with_suffix('.snapshots.json').write_text(json.dumps(event_snapshots.report(),indent=2))
            except Exception as exc:print(f'[event snapshots] final export failed: {exc}')
        if loop_mission is not None:loop_mission.close()
        if 'control' in locals() and hasattr(control,'close'):control.close()
        if grid is not None and path is not None and export_survey is not None:
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
