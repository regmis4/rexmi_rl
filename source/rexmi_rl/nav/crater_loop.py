"""Outside → observed checkpoints → center → selected exterior route.

One environment, policy and retained live map. The launcher selects the validated
western blue route; its short center connector starts at the measured arrival pose.
"""
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
import copy
import hashlib
import json
import math
import time

import numpy as np

from . import costmap_path, exit_region, footprint_costs, planning_snapshot, slope_cost
from .checkpoint_control import CheckpointController, Motion
from .mission import Waypoint
from .staged_routes import compact_path
from . import descent_planning
from .descent_response import DescentReverseRecovery

DEADLINE_SECONDS = 900.
DESCENT_GRADE_LIMIT_DEG = 40.


class DescentController(CheckpointController):
    """Checkpoint descent with the requested simulation motion-veto ablation.

    Keep observed-route planning, tracking and measured-progress recovery.
    Do not replace steering with predictive turn/edge clearance refusals.
    Other missions retain their existing checkpoint controller behavior.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ascent_grade_limits=(self.grid.uphill_limit_deg,self.grid.side_limit_deg)
        self.grid.uphill_limit_deg=self.grid.side_limit_deg=DESCENT_GRADE_LIMIT_DEG
        self._descent_profile_active=True
        self.alignment_response=None
        self.reverse_recovery=DescentReverseRecovery()
        self.waiting_for_map=False
        self.needs_backout=False
        self.grid.footprint_costs_enabled=True

    def restore_ascent_grade_limits(self, now):
        if not self._descent_profile_active:return
        self.grid.uphill_limit_deg,self.grid.side_limit_deg=self._ascent_grade_limits
        self._descent_profile_active=False

    def choose_route(self, xy, goal, yaw):
        return descent_planning.choose_route(self,xy,goal,yaw)

    def route_score(self, route, yaw):
        return descent_planning.route_score(self,route)

    def _plan(self, xy, yaw, now):
        self.waiting_for_map=False
        self.needs_backout=False
        self.wait_pose=(tuple(xy),yaw)
        descent_planning.refresh(self)
        cell=self.grid.world_to_cell(*xy)
        if cell is None or self.observed_distance[cell]<self.grid.footprint_radius+self.grid.clearance:
            self.waiting_for_map=True
            self.planning_failure='insufficient measured data under the body; rescan before choosing a checkpoint'
            return False
        if self.grid.blocked[cell]:
            self.needs_backout=True
            self.backout_pose=(tuple(xy),yaw)
            self.planning_failure='newly observed obstacle overlaps current clearance; reverse and replan'
            return False
        self.mission_route=[]  # Re-evaluate with the current observed-space mask.
        okay=super()._plan(xy,yaw,now)
        if not okay:return False
        descent_planning.refresh(self)
        eligible=[i for i,p in enumerate(self.path) if i>0 and
            self.checkpoint_supported[self.grid.world_to_cell(*p)] and
            (math.dist(xy,p)>=.3 or self.is_mission_checkpoint)]
        if not eligible:
            self.waiting_for_map=True
            self.planning_failure='checkpoint too close to missing data; collect more observations'
            return False
        end=eligible[-1]
        self.is_mission_checkpoint &= end==len(self.path)-1
        self.path=self.path[:end+1];self.checkpoint=self.path[-1]
        self.checkpoint_arrival_radius=.4 if self.is_mission_checkpoint else .2
        self.checkpoint_plan_revision=self.grid.revision
        cell=self.grid.world_to_cell(*self.checkpoint)
        self.escape_events.append(dict(event='descent_checkpoint_selected',time=now,
            position=list(xy),checkpoint=list(self.checkpoint),
            observation_target=getattr(self,'observation_target',None),
            measured_data_clearance_m=float(self.observed_distance[cell]),
            extra_data_reserve_m=float(self.checkpoint_reserve[cell]),
            cost_model='wheelbase exponential absolute-grade; 15 degree knee / 6 degree scale'))
        return True

    def hold(self, reason, now, terminal=False):
        if self.needs_backout and not terminal:
            self.needs_backout=False
            xy,yaw=self.backout_pose
            return self._start_backout(xy,yaw,now,reason)
        if self.waiting_for_map and not terminal:
            self.data_retry_at=now
            self.set_state('DATA_WAIT',reason,now)
            xy,yaw=self.wait_pose
            self.hold_anchor=np.asarray(xy).copy();self.hold_heading=yaw
            self.path=[];self.target=None;self.checkpoint=None
            return self.position_hold(xy,yaw)
        return super().hold(reason,now,terminal)

    def _start_backout(self, xy, yaw, now, reason):
        if not descent_planning.backout_observed(self.grid,xy,yaw):
            return self.hold('reverse needs an observed corridor; '+reason,now,terminal=True)
        self.reverse_recovery.start(now,xy,yaw)
        self.escape_events.append(dict(event='descent_reverse_start',time=now,position=list(xy),reason=reason))
        self.set_state('BACKOUT',reason,now)
        return Motion(-self.reverse_recovery.SPEEDS[self.reverse_recovery.level],0.)

    def begin_escape(self, xy, yaw, now, evidence, mark=True):
        self.slip_tracker.reset()
        self._start_backout(xy,yaw,now,'measured motion difficulty: '+evidence['kind'])

    def retreat(self, xy, yaw, now):
        return self._start_backout(xy,yaw,now,'checkpoint progress stalled')

    def request(self, action, now, vx=0., omega=0.):
        self.reverse_recovery.active=False
        self.alignment_response=None
        return super().request(action,now,vx,omega)

    def _tick(self, xy, yaw, speed, now, upright=1., velocity=None):
        self.velocity_world=np.asarray(velocity[:2] if velocity is not None else
                                       (speed*math.cos(yaw),speed*math.sin(yaw)))
        if self.state=='DATA_WAIT':
            started=self.since
            if self.hold_anchor is not None and math.dist(xy,self.hold_anchor)>.20:
                return self._start_backout(xy,yaw,now,'drifting during data wait; back off')
            if now-started>=10.:
                return self.hold('insufficient observed checkpoint clearance after rescanning',now,terminal=True)
            if now-self.data_retry_at>=2. and speed<.10:
                self.data_retry_at=now
                if self._plan(xy,yaw,now):return Motion()
                if self.needs_backout:return self.hold(self.planning_failure,now)
                self.state='DATA_WAIT';self.since=started
                self.reason=self.planning_failure
            return self.position_hold(xy,yaw)
        if self.state=='BACKOUT':
            command,result=self.reverse_recovery.tick(now,xy)
            if result is not None:
                self.escape_events.append(dict(event='descent_reverse_end',time=now,position=list(xy),**result))
                self.pending_arrival=False;self.replan_anchor=None;self.mission_route=[]
                self.alignment_response=None;self.slip_tracker.reset()
                self.set_state('SETTLE' if result['success'] else 'HOLD',
                    'reverse complete; rescan and replan' if result['success'] else
                    'reverse made insufficient progress; recovery unresolved',now)
                return Motion()
            self.reason=f'reverse recovery: {abs(command):.1f} m/s requested; escalate only without progress'
            return Motion(command,0.)
        if (self.state=='FOLLOW' and not self.mission_route_complete and self.checkpoint is not None
                and math.dist(xy,self.checkpoint)<=.4 and speed<.5
                and self.grid.revision!=getattr(self,'checkpoint_plan_revision',None)):
            # Roll the short horizon forward as new data arrives. Do not stop
            # at every old frontier or turn away merely to collect more points.
            if not self._plan(xy,yaw,now):return self.hold(self.planning_failure,now)
        return super()._tick(xy,yaw,speed,now,upright,velocity)

    def motion_clear(self, xy, yaw, speed, motion):
        self.motion_rejection=''
        return True

    def prepare_edge_exit(self, xy, yaw, now):
        return False

    def tick(self, xy, yaw, speed, now, upright=1., velocity=None):
        motion=super().tick(xy,yaw,speed,now,upright,velocity)
        if self.state=='FOLLOW' and motion.vx>0 and abs(self.heading_error)>.35:
            motion=Motion(0.,math.copysign(.35,self.heading_error))
            self.previous_command=motion
            self.reason='align to the observed checkpoint corridor before driving'
        if self.state!='FOLLOW' or abs(motion.vx)>1e-6 or abs(motion.omega)<.1:
            self.alignment_response=None
            return motion
        error=abs(self.heading_error)
        if self.alignment_response is None or error<self.alignment_response[1]-.05:
            self.alignment_response=(now,error)
        if now-self.alignment_response[0]<5.:return motion
        self.alignment_response=(now,error)
        self.escape_events.append(dict(event='descent_turn_stall',time=now,position=list(xy),
            heading_error=error))
        command=self._start_backout(xy,yaw,now,'steering stalled: reverse before replanning')
        self.alignment_response=None
        self.previous_command=command
        return command


def configure(args, argv):
    if args.terrain_prior or args.simulator_survey:
        raise ValueError('crater_loop builds its map from live descent observations; omit terrain priors/survey')
    if any((args.manual_checkpoints, args.checkpoint_replay, args.reviewed_buttons,
            args.ascent_path_result, args.flat_sidesteps, args.capture_ascent_departure,
            args.ascent_event_snapshots, args.ascent_exploration_history, args.ascent_launch_variant)):
        raise ValueError('crater_loop owns its destinations, map and ascent plan; incompatible override')
    args.nav_controller='checkpoint'
    args.ascent_architecture='remote'
    args.ascent_mode='characterize'
    args.ascent_goal='any_exit'
    args.ascent_unguarded=True
    defaults=dict(spawn_x=args.crater_x+13.,spawn_y=args.crater_y,spawn_z=4.5,spawn_yaw=math.pi,
                  goal_x=args.crater_x,goal_y=args.crater_y,goal_radius=.8,max_steps=45000)
    for name,value in defaults.items():
        if not any(arg=='--'+name or arg.startswith('--'+name+'=') for arg in argv):setattr(args,name,value)
    if not np.isfinite([args.spawn_x,args.spawn_y,args.spawn_z,args.spawn_yaw,args.crater_x,args.crater_y,args.r_rim]).all():
        raise ValueError('crater_loop requires finite spawn and crater geometry')
    if args.r_rim<=1 or math.dist((args.spawn_x,args.spawn_y),(args.crater_x,args.crater_y))<=args.r_rim:
        raise ValueError('crater_loop must spawn outside the crater')
    if (args.goal_x,args.goal_y)!=(args.crater_x,args.crater_y) or not 0<args.goal_radius<=1.:
        raise ValueError('crater_loop must descend to the crater center within 1 m')
    args.map_retention='full'
    if getattr(args,'crater_ascent_route',None):
        validated=load_validated_route(args.crater_ascent_route)
        identity=dict(task=args.task,seed=args.seed,crater_origin=[args.crater_x,args.crater_y],rim_radius=args.r_rim)
        for key,value in identity.items():
            if validated[key]!=value:
                raise ValueError('Validated ascent route scenario mismatch: '+key)



@dataclass
class ExitPlan:
    route: list
    model: object
    metadata: dict


def load_validated_route(filename):
    """A versioned route, not a historical map or a replacement sensor stream."""
    path=Path(filename).resolve()
    raw=path.read_bytes();record=json.loads(raw)
    route=np.asarray(record['route'],dtype=float)
    if (route.ndim!=2 or route.shape[1]!=2 or len(route)<2 or not np.isfinite(route).all()
            or (np.linalg.norm(np.diff(route,axis=0),axis=1)<1e-8).any()):
        raise ValueError('Validated route must contain finite distinct XY segments')
    record.update(artifact_path=str(path),artifact_sha256=hashlib.sha256(raw).hexdigest())
    return record


def plan_exit(grid, xy, yaw, now, rim_radius, directory, *, validated_route=None):
    """Freeze live costs; connect the validated route or explicitly search a new exit."""
    started=time.monotonic()
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    parameters=slope_cost.Parameters()  # accepted 15° knee / 6° exponential scale
    grid.footprint_costs_enabled=True
    footprint_costs.rebuild(grid)
    model=slope_cost.Model(grid,parameters)
    holder=SimpleNamespace(grid=grid,planning_snapshot_dir=directory,waypoints=[],wp_idx=0,
        plan_generation=0,protocol_version='live-descent-map-exit-v1',goal_mode='any_exit',
        exit_rim_radius=rim_radius,selected_exit=None)
    snapshot=planning_snapshot.save(holder,xy,yaw,now)
    if validated_route is not None:
        original=[tuple(p) for p in validated_route['route']]
        # Center arrival is within 0.8 m. Preserve every original waypoint;
        # prepend only the measured connection, with no new exit/slope veto.
        route=([tuple(xy)] if math.dist(xy,original[0])>1e-8 else [])+original
        score=sum(model.route_components(route).values())
        solution=dict(nodes=0,directed_edges=0)
    else:
        allowed=grid.known & ~grid.blocked & np.isfinite(grid.gx) & np.isfinite(grid.gy)
        lower=np.asarray(grid.origin)-grid.world_size/2
        solution=costmap_path.solve(grid.cost,allowed,xy,lower,grid.cell_size,density=model)
        rr,cc=np.indices(grid.cost.shape)
        distance=np.hypot(lower[0]+(rr+.5)*grid.cell_size-grid.origin[0],
                          lower[1]+(cc+.5)*grid.cell_size-grid.origin[1])
        terminals=np.argwhere(allowed & np.isfinite(solution['distance']) & (distance>=rim_radius+.8))
        # Cheapest admissible terminal, identical objective to the accepted fit.
        order=np.argsort(solution['distance'][tuple(terminals.T)]) if len(terminals) else []
        selected=None
        for index in order:
            cell=tuple(terminals[index]);point=lower+(np.asarray(cell)+.5)*grid.cell_size
            if not exit_region.reason(grid,point,rim_radius):
                selected=cell;break
        if selected is None:
            raise ValueError('Retained descent map has no connected observed gentle exterior exit')
        route=compact_path(costmap_path.path_to_cell(solution,selected))
        score=sum(model.route_components(route).values())
        if not math.isclose(score,float(solution['distance'][selected]),rel_tol=1e-8):
            raise RuntimeError('Compacted route changed the solved cost')
    metadata=dict(map_path=snapshot['path'],map_sha256=hashlib.sha256(Path(snapshot['path']).read_bytes()).hexdigest(),
        route_name='validated_west_blue' if validated_route is not None else 'live_descent_any_exit',parameters=asdict(parameters),start=list(xy),yaw=yaw,
        planned_at_sim_time=now,planning_seconds=time.monotonic()-started,
        map_source='live sensors retained during this descent',known_cells=int(grid.known.sum()),
        observed_area_m2=float(grid.known.sum()*grid.cell_size**2),
        predicted_metrics=dict(length_m=sum(math.dist(a,b) for a,b in zip(route,route[1:])),fitted_score=score),
        selected_exit=list(route[-1]),routes=dict(any_exit=route),rim_radius=rim_radius,
        nodes=solution['nodes'],directed_edges=solution['directed_edges'])
    if validated_route is not None:
        metadata.update(route_source=validated_route,selection='user-selected previously validated corridor',
            connector_length_m=math.dist(xy,validated_route['route'][0]),
            original_route_preserved=True)
        (directory/'validated_route.json').write_text(json.dumps(validated_route,indent=2)+'\n')
    result_path=directory/'exit_plan.json'
    result_path.write_text(json.dumps(metadata,indent=2)+'\n')
    metadata.update(result_path=str(result_path),result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest())
    np.savetxt(directory/'exit_route.csv',np.asarray(route),delimiter=',',header='x,y',comments='')
    return ExitPlan(route,model,metadata)


class CraterLoop:
    def __init__(self, grid, directory, rim_radius=11., *, planner=plan_exit, validated_route=None):
        self.grid=grid;self.directory=Path(directory);self.rim_radius=rim_radius
        self.phase='DESCEND_TO_CENTER';self.operator_mode='auto'
        self.events=[];self.plan=None;self.future=None;self.generation=0
        self.center_at=None;self.stable_since=None;self.failure=None
        self.intervened=False;self.descent=None;self.last_yaw=None;self.last_time=None
        self.worker=ThreadPoolExecutor(max_workers=1,thread_name_prefix='crater-exit')
        self.planner=partial(planner,validated_route=validated_route) if validated_route is not None else planner

    def request(self, action):
        if action in ('stop','manual','resume'):
            self.operator_mode={'stop':'stop','manual':'manual','resume':'auto'}[action]
            self.intervened=True

    def _event(self, phase, now, xy, **extra):
        self.phase=phase
        self.events.append(dict(event='crater_loop_stage',stage=phase,time=now,position=list(xy),**extra))

    def status(self):
        goal=list(self.plan.route[-1]) if self.plan is not None else list(self.grid.origin)
        return dict(stage=self.phase,goal=goal,goal_mode='crater_loop',
                    completed_destinations=2 if self.phase=='COMPLETE' else int(self.center_at is not None))

    def _start_plan(self, control, xy, yaw, now):
        if isinstance(control,DescentController):control.restore_ascent_grade_limits(now)
        frozen=copy.deepcopy(self.grid)
        self.generation+=1
        target=self.directory/f'exit-plan-{self.generation:02d}'
        self.future=self.worker.submit(self.planner,frozen,tuple(xy),yaw,now,self.rim_radius,target)
        self._event('PLAN_ASCENT',now,xy,output=str(target))
        control.state='PLAN_ASCENT';control.reason='planning an exit from retained descent observations'

    def tick(self, control, xy, yaw, speed, now, upright, velocity):
        dt=now-self.last_time if self.last_time is not None else 0.
        delta=yaw-self.last_yaw if self.last_yaw is not None else 0.
        rate=math.atan2(math.sin(delta),math.cos(delta))/dt if dt>0 else math.inf
        self.last_yaw,self.last_time=yaw,now
        if now>=DEADLINE_SECONDS:
            self.failure='mission deadline';control.state='HOLD';control.reason=self.failure
            self._event('FAILED',now,xy,reason=self.failure)
            return control,Motion()
        if self.operator_mode!='auto':
            return control,control.tick(xy,yaw,speed,now,upright=upright,velocity=velocity)
        if self.phase=='DESCEND_TO_CENTER':
            motion=control.tick(xy,yaw,speed,now,upright=upright,velocity=velocity)
            if control.state=='COMPLETE':
                self.descent=dict(time=now,position=list(xy),waypoints_reached=control.wp_idx,
                    hold_failures=control.hold_failures,rolling_checkpoints=control.rolling_checkpoints,
                    steering_recoveries=sum(e['event']=='descent_turn_stall' for e in control.escape_events),
                    reverse_recoveries=sum(e['event']=='descent_reverse_start' for e in control.escape_events),
                    recovery_events=copy.deepcopy(control.escape_events))
                self._event('SETTLE_AT_CENTER',now,xy)
            return control,motion
        if self.phase=='SETTLE_AT_CENTER':
            if math.dist(xy,self.grid.origin)>control.waypoints[0].arrival_radius:
                self.stable_since=None
                control.wp_idx=0
                control.request('resume',now)
                self._event('DESCEND_TO_CENTER',now,xy,reason='return to center after settling drift')
                return control,Motion()
            stable=(speed<.12 and abs(rate)<.12
                    and upright>.8 and 0<dt<=.041)
            self.stable_since=(self.stable_since if self.stable_since is not None else now) if stable else None
            if self.stable_since is not None and now-self.stable_since>=.6:
                self.center_at=now;self._start_plan(control,xy,yaw,now)
            return control,Motion()
        if self.phase=='PLAN_ASCENT':
            if not self.future.done():return control,Motion()
            try: plan=self.future.result()
            except Exception as exc:
                self.failure=str(exc);control.state='HOLD';control.reason=self.failure
                self._event('FAILED',now,xy,reason=self.failure)
                return control,Motion()
            if math.dist(xy,plan.route[0])>.3:
                self._start_plan(control,xy,yaw,now)
                return control,Motion()
            from .experimental_path_control import ExperimentalPathController
            previous=control
            control=ExperimentalPathController(self.grid,[Waypoint(*plan.route[-1],.4,'Selected exit')],
                mode='characterize',goal_mode='any_exit',exit_rim_radius=self.rim_radius,
                cruise_speed=.4,planned_route=plan.route,planning_metadata=plan.metadata,
                preference_model=plan.model,run_id=self.directory.name,async_planning=False)
            control.intervened=previous.intervened or self.intervened
            self.plan=plan
            self._event('ASCEND_TO_EXTERIOR',now,xy,selected_exit=list(plan.route[-1]),
                        plan_path=plan.metadata['result_path'])
        if self.phase=='ASCEND_TO_EXTERIOR':
            motion=control.tick(xy,yaw,speed,now,upright=upright,velocity=velocity)
            if control.state=='COMPLETE':self._event('COMPLETE',now,xy)
            return control,motion
        return control,Motion()

    def report(self):
        return dict(**self.status(),center_arrival_time=self.center_at,descent=self.descent,
            plan=self.plan.metadata if self.plan else None,failure=self.failure,
            manual_intervention=self.intervened,events=self.events,
            map_source='live sensor observations from this mission; no loaded prior')

    def close(self):
        self.worker.shutdown(wait=True,cancel_futures=True)
