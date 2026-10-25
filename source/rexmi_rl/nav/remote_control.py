"""Persistent teleop buttons driven by observed waypoints. Characterization only.

The inherited controller supplies planning, maps and attempt serialization.
Its tick, tracking, braking and recovery implementations are never called.
"""
from collections import deque
import math
import time

import numpy as np

from ..drive_adapter import directional_command
from .ascent_control import AscentController
from .checkpoint_control import Motion, wrap_angle
from .movement import MotionObservation
from .swept_footprint import oriented_sweep_reason
from .staged_routes import observed_edge_reason
from .recovery_intent import RecoveryPermission
from . import remote_routes
from . import footprint_costs, stage_graph, ascent_first, climb_burden, wheelbase_demand, sustained_climb
from .turn_staging import stage_reason, turn_reason
from . import exit_region, checked_turn


class RemoteAscentController(AscentController):
    protocol_version = 'persistent-remote-v22-stable-survey-candidates'

    def __init__(self, *args, route_objective='baseline', goal_mode='fixed', exit_rim_radius=11., **kwargs):
        if goal_mode not in ('fixed', 'any_exit'):
            raise ValueError('Unknown ascent goal mode')
        if goal_mode == 'any_exit' and route_objective != 'baseline':
            raise ValueError('Any-exit uses the shared additive objective')
        self.goal_mode = goal_mode
        self.exit_rim_radius = float(exit_rim_radius)
        self.selected_exit = None
        self.checked_turn = None
        self.correction_anchor = None
        self.rejoin_refusals = 0
        if route_objective not in ('baseline', 'ascent_first', 'climb_burden', 'wheelbase_demand', 'sustained_climb'):
            raise ValueError('Unknown remote route objective')
        self.route_objective = route_objective
        self.ascent_first_active = False
        super().__init__(*args, **kwargs)
        if route_objective == 'ascent_first':
            self.protocol_version = 'persistent-remote-v23-ascent-first-experimental'
        if route_objective == 'climb_burden':
            self.protocol_version = 'persistent-remote-v24-climb-burden-experimental'
        if route_objective == 'wheelbase_demand':
            self.protocol_version = 'persistent-remote-v26-wheelbase-fourth-experimental'
        if route_objective == 'sustained_climb':
            self.protocol_version = 'persistent-remote-v27-sustained-climb-experimental'
        if self.mode != 'characterize':
            raise ValueError('Remote controller requires characterization; no calibrated model exists')
        self.grid.motion_turn_limit = 15.
        self.grid.local_minimum_turns_enabled = True
        self.grid.route_conditioned_turns = True
        self.grid.planning_grade_reserve = 0.
        self.grid.diagonal_ascent_preference = False
        self.grid.traversal_objective = 'gentle-additive-v1'
        self.grid.footprint_costs_enabled = True
        footprint_costs.rebuild(self.grid)
        self.minimum_leg_distance = 0.
        # Runner binds a qualified model to this exact policy/controller/profile.
        # Characterization does not bypass the domain check for probability rank.
        self.probability_domain_validated = False
        self.button_state = 'OBSERVE'
        self.button_since = 0.
        self.operator_mode = 'auto'
        self.obs = None
        self.samples = deque()
        self.metrics = dict(status='observe', forward_mps=0., progress_mps=0., lateral_mps=0.)
        self.vertex = 1
        self.direction = 1
        self.turn_sign = 0
        self.turn_target = 0.
        self.turn_best = math.inf
        self.turn_progress_at = 0.
        self.turn_deadline = 0.
        self.drive_exposure = 0.
        self.slow_since = None
        self.progress_anchor = None
        self.progress_time = 0.
        self.low_progress_active = 0.
        self.recovery_episode = None
        self.recovery_permission = None
        self.recovery_pause = None
        self.recovery_actions = 0
        self.recovery_endpoint = None
        self.recovery_start = None
        self.recovery_onward_anchor = None
        self.recovery_route = ()
        self.recovery_target = None
        self.recovery_waypoint = None
        self.recovery_kind = None
        self.recovery_best_remaining = math.inf
        self.recovery_credited_remaining = math.inf
        self.recovery_progress_at = 0.
        self.emergency = False
        self.emergency_clear_since = None
        self.goal_settle_since = None
        self.command = Motion()
        self.last_output_time = None
        self.physical_execution = None
        self.output_owner = 'OBSERVE'
        self.pending_goal_success = False
        self.observation_dt = 0.
        prior = getattr(self.grid, 'terrain_prior', {})
        self.static_prior_enabled = bool(prior.get('kind') == 'simulator_development'
            and not prior.get('parent_geometry_id') and prior.get('geometry_id') and prior.get('file_sha256'))
        # This task casts only against its immutable generated ground mesh.
        # load_prior validates task/seed/frame/checksum before construction.
        self.static_retention_enabled = bool(self.static_prior_enabled
            and prior.get('task') == 'RexmiRl-Go2w-Crater-Bowl-RockySlope-Play-v0'
            and prior.get('version') == 1 and prior.get('coordinate_frame') == 'world'
            and type(prior.get('seed')) is int)
        if route_objective != 'baseline' and not self.static_retention_enabled:
            raise ValueError('Experimental route objectives require a validated full-crater development survey')
        if goal_mode == 'any_exit':
            if not self.static_retention_enabled or not math.isfinite(self.exit_rim_radius) or self.exit_rim_radius <= 0:
                raise ValueError('Any-exit requires a validated full-crater development survey and rim radius')
            if len(self.waypoints) != 1:
                raise ValueError('Any-exit is currently an ascent-only single-stage mission')
            self.protocol_version = 'persistent-remote-v29-checked-small-turns'
            self.grid.checked_small_turns = True
        self.prior_support = (self.grid.known & np.isfinite(self.grid.surface)).copy() if self.static_prior_enabled else None
        self.prior_surface = self.grid.surface.copy() if self.static_prior_enabled else None
        if self.static_prior_enabled:
            self.prior_support.setflags(write=False)
            self.prior_surface.setflags(write=False)
        self.prior_admissions = 0
        self.last_prior_event = -math.inf
        self.minimum_turn = None
        self.graph_objective_active = False
        self.graph_plan_stats = None
        self.candidate_source = (stage_graph.freeze_candidate_source(self.grid)
            if self.static_prior_enabled else None)

    def _leg(self, route):
        # Intermediate A* vertices guide travel; only destinations require rest.
        return route

    def _adoption_edge_reason(self, a, b, xy, retreat):
        reason = observed_edge_reason(self.grid, a, b, xy, no_go_exit=retreat)
        if not reason and remote_routes.outbound_scope(self, xy):
            reason, _ = remote_routes.physical_profile(self, [a, b])
        return reason

    def _augment_routes(self, routes, xy, yaw):
        return remote_routes.augment(self, routes, xy, yaw)

    def _compute_plan(self,xy,yaw,now):
        from .planning_snapshot import save,finish
        self.planning_snapshot=save(self,xy,yaw,now)
        self.graph_objective_active=False
        self.ascent_first_active=False
        self.graph_plan_stats=None
        result=super()._compute_plan(xy,yaw,now)
        if self.goal_mode == 'any_exit':
            if result and self.route_purpose == 'mission':
                self.selected_exit = tuple(self.path[-1])
                self.is_mission_checkpoint = True
            elif self.graph_plan_stats:
                self.planning_failure = self.graph_plan_stats.get('reason') or self.planning_failure
        if not result and self.ascent_first_active and self.graph_plan_stats:
            self.planning_failure=self.graph_plan_stats.get('rejection') or self.planning_failure
        finish(self.planning_snapshot,self,result)
        return result

    def _propose_routes(self,xy,yaw,support,route_limit):
        self.graph_objective_active=False
        self.ascent_first_active=False
        self.graph_plan_stats=None
        if self.goal_mode == 'any_exit':
            path, stats = exit_region.plan(self.grid, xy, yaw, self.exit_rim_radius,
                lambda a,b: remote_routes.physical_profile(self,[a,b])[0],
                candidate_source=self.candidate_source)
            stats.update(profile=exit_region.NAME,connected=bool(path),has_route=bool(path),
                         uphill_limit=self.grid.uphill_limit_deg,rejection=stats.get('reason',''))
            self.graph_plan_stats = stats
            self.route_searches.append(stats)
            return [(exit_region.NAME,path,True)] if path else []
        # The new option applies throughout the outbound mission, including
        # replans above the floor. Descent and recovery keep their own owners.
        if (self.route_objective in ('ascent_first','climb_burden','wheelbase_demand','sustained_climb') and self.static_retention_enabled
                and self.wp_idx < len(self.waypoints)
                and math.dist((self.waypoints[self.wp_idx].x,self.waypoints[self.wp_idx].y),self.grid.origin)>=10.):
            goal=(self.waypoints[self.wp_idx].x,self.waypoints[self.wp_idx].y)
            planner={'climb_burden':climb_burden,'wheelbase_demand':wheelbase_demand,
                     'ascent_first':ascent_first,'sustained_climb':sustained_climb}[self.route_objective]
            label=self.route_objective.replace('_','-')
            graph=planner.build(self.grid,xy,goal,
                lambda a,b:remote_routes.physical_profile(self,[a,b])[0],
                candidate_source=self.candidate_source)
            path,stats=planner.search(self.grid,graph,yaw)
            reason=(planner.route_reason(self.grid,path,yaw,xy) if planner is not ascent_first else
                ascent_first.route_reason(self.grid,path,yaw,stats['exit_index'],xy)) if path else stats['reason']
            self.graph_plan_stats=dict(path=list(path),profile=label,connected=bool(path and not reason),
                has_route=bool(path),uphill_limit=self.grid.uphill_limit_deg,
                candidate_inventory=graph['inventory'],nodes=len(graph['points']),
                directed_checks=graph['directed_checks'],accepted_links=len(graph['edges']),
                build_seconds=graph['build_seconds'],edge_rejections=graph['rejections'],
                objective_unavailable_links=graph.get('objective_unavailable_links',0),
                objective_unavailable_reasons=graph.get('objective_unavailable_reasons',[]),
                rejection=reason,**stats)
            self.route_searches.append(self.graph_plan_stats)
            self.ascent_first_active=True  # Explicit no-witness result also owns planning.
            return [(label,path,True)] if path and not reason else []
        if remote_routes.outbound_scope(self,xy):
            goal=(self.waypoints[self.wp_idx].x,self.waypoints[self.wp_idx].y)
            started=time.perf_counter()
            graph=stage_graph.build(self.grid,xy,goal,
                lambda a,b:remote_routes.physical_profile(self,[a,b])[0],
                candidate_source=self.candidate_source)
            stage_graph.height_profiles(self.grid,graph)
            path,stats=stage_graph.search(self.grid,graph,yaw,uphill_first=True,
                preserve_height=True,height_budget=.20)
            reason=stage_graph.route_height_reason(self.grid,path) if path else 'no complete height-retaining checkpoint route'
            self.graph_plan_stats=dict(profile='terrain-checkpoints',connected=bool(path and not reason),
                planning_snapshot=getattr(self,'planning_snapshot',None),
                has_route=bool(path),uphill_limit=self.grid.uphill_limit_deg,
                nodes=len(graph['points']),candidate_inventory=graph['inventory'],
                directed_checks=graph['directed_checks'],
                accepted_links=len(graph['edges']),turn_checks=len(graph['turns']),
                build_budget_exhausted=graph['build_budget_exhausted'],
                profile_budget_exhausted=graph.get('profile_budget_exhausted',False),
                wall_seconds=time.perf_counter()-started,rejection=reason,**stats)
            self.route_searches.append(self.graph_plan_stats)
            if path and not reason:
                self.graph_objective_active=True
                return [('terrain-checkpoints',path,True)]
            self.graph_plan_stats['fallback']='shared additive planner; sampled height retention is unverified'
        return super()._propose_routes(xy,yaw,support,route_limit)

    def _adoption_route_reason(self,path,xy,yaw,shadow):
        if self.goal_mode == 'any_exit' and shadow.route_purpose == 'mission':
            return exit_region.route_reason(self.grid,path,yaw,xy,self.exit_rim_radius,
                lambda a,b: remote_routes.physical_profile(self,[a,b])[0])
        if getattr(shadow,'ascent_first_active',False) and shadow.route_purpose=='mission':
            if shadow.graph_plan_stats.get('objective')==sustained_climb.NAME:
                return sustained_climb.route_reason(self.grid,path,yaw,xy)
            if shadow.graph_plan_stats.get('objective')==wheelbase_demand.NAME:
                return wheelbase_demand.route_reason(self.grid,path,yaw,xy)
            if shadow.graph_plan_stats.get('objective')==climb_burden.NAME:
                return climb_burden.route_reason(self.grid,path,yaw,xy)
            return ascent_first.route_reason(self.grid,path,yaw,shadow.graph_plan_stats['exit_index'],xy)
        if shadow.graph_objective_active and shadow.route_purpose=='mission':
            return stage_graph.route_height_reason(self.grid,path)
        return ''

    def _adopt_route_metadata(self,shadow):
        self.selected_exit = getattr(shadow,'selected_exit',None)
        self.ascent_first_active=getattr(shadow,'ascent_first_active',False) and shadow.route_purpose=='mission'
        self.graph_objective_active=shadow.graph_objective_active and shadow.route_purpose=='mission'
        self.graph_plan_stats=shadow.graph_plan_stats if shadow.route_purpose=='mission' else None

    def route_score(self, route, yaw):
        if self.goal_mode == 'any_exit' and self.graph_plan_stats and list(route)==self.graph_plan_stats.get('path'):
            return self.graph_plan_stats['rank']
        if (self.ascent_first_active and self.graph_plan_stats and self.graph_plan_stats.get('complete')
                and list(route)==self.graph_plan_stats.get('path')):
            return self.graph_plan_stats['rank'][0]
        return self.grid.route_cost(route, yaw)

    def _recent_approach_cost(self,route,now):
        # Predictions are not physical exclusions. The new graph already honors
        # actual directional failures; legacy soft costs cannot change its rank.
        if self.ascent_first_active or self.goal_mode == 'any_exit':return 0.
        return super()._recent_approach_cost(route,now)

    def _select_route_candidate(self, candidates, xy, yaw):
        return remote_routes.select(self, candidates, xy, yaw)

    def configure_outbound_survey_routes(self):
        # Both mission stages use the same map and independent A* searches.
        pass

    def position_hold(self, xy, yaw):
        return Motion()  # Teleop release keeps policy inference running.

    def request(self, action, now, vx=0., omega=0.):
        super().request(action, now, vx, omega)
        self.operator_mode = 'manual' if action == 'manual' else 'stop' if action == 'stop' else 'auto'
        self._button('OBSERVE', now, 'operator '+action)
        self.samples.clear()
        self.progress_anchor = None
        self.recovery_episode = None
        self.recovery_permission = None
        self.recovery_pause = None
        self.recovery_onward_anchor = None
        self.goal_settle_since = None
        self.minimum_turn = None
        self.checked_turn = None
        self.correction_anchor = None
        self.command = Motion()

    def _button(self, state, now, reason):
        if state != self.button_state:
            self.attempt_events.append(dict(event='remote_transition', time=now,
                previous=self.button_state, state=state, reason=reason))
            self.button_since = now
        self.button_state = state
        self.reason = reason

    def _hold(self, reason, now):
        self.operator_mode = 'stop'
        self.minimum_turn = None
        self.checked_turn = None
        self.recovery_permission = None
        self.recovery_pause = None
        self.finish_attempt('censored', reason, self.obs.xy, self.obs.yaw, now)
        self.set_state('HOLD', reason, now)
        self._button('HOLD', now, reason)
        return Motion()

    def _geometry(self, command, *, escape=False):
        return (self._command_geometry(command, escape=escape)
                or self._actual_geometry(command, escape=escape, checked_intent=True))

    def _command_geometry(self, command, *, escape=False, check_grade=True):
        """Validate intended travel independently of measured motion."""
        o = self.obs
        longitudinal = o.velocity[0]*math.cos(o.yaw)+o.velocity[1]*math.sin(o.yaw)
        opposing_damping = (self.emergency and command.omega == 0.
                            and command.vx*longitudinal < 0.)
        # Check intended movement without departure permission first. Only
        # incidental gait motion may borrow the frozen occupied-patch context.
        # Opposing damping predicts continued measured motion, not travel in
        # the command's sign. Recompute opposition on every observation.
        if (command.vx or command.omega) and not opposing_damping:
            if command.omega and self.checked_turn is not None:
                reason=checked_turn.runtime_reason(self,self.checked_turn['target'])
                if reason:return reason
            end = (o.xy[0]+command.vx*.5*math.cos(o.yaw),
                   o.xy[1]+command.vx*.5*math.sin(o.yaw))
            yaw_delta = command.omega*.5
            if command.omega and self.minimum_turn is not None:
                reason=turn_reason(self.grid,o.xy,o.yaw,o.yaw+yaw_delta,
                    stage_headings=(self.minimum_turn['incoming'],self.minimum_turn['target']))
                if reason:return reason
            reason = self._physical_sweep(o.xy, end, o.yaw,
                yaw_offsets=(min(0., yaw_delta), max(0., yaw_delta)),
                commanded=bool(command.vx) and check_grade,
                circular_support=False, no_go_exit=escape)
            if reason:
                return reason
        return ''

    def _unfinished_recovery_pause(self):
        """Pure lifecycle query: permission belongs to one unfinished maneuver."""
        p, o = self.recovery_pause, self.obs
        if (p is None or self.recovery_episode != p['episode']
                or not 0. <= o.time-p['episode'] < 120.
                or self.operator_mode != 'auto' or self.button_state != 'OBSERVE' or self.path
                or self.recovery_permission is None or not self.recovery_permission.patches):
            return False
        axis = np.subtract(p['endpoint'], p['start'])
        return (math.dist(o.xy, p['endpoint']) > .25
            and np.dot(np.subtract(o.xy, p['start']), axis) < np.dot(axis, axis)
            and p['direction']*np.dot(np.subtract(p['endpoint'], o.xy),
                                    (math.cos(o.yaw), math.sin(o.yaw))) > 0.)

    def _checked_recovery_pause(self, command):
        if (command.vx or command.omega or self.obs.speed > .12
                or self.emergency or not self._unfinished_recovery_pause()):
            return False
        probe = Motion(self.recovery_pause['direction']*self.cruise_speed, 0.)
        # No directional permission for intentional departure or its reserve.
        return not (self._command_geometry(probe, escape=True)
                    or self._anticipation(probe, escape=True))

    def _actual_geometry(self, command, *, escape=False, checked_intent=False):
        """Provisional actual stopping fan, never evidence of calibrated braking."""
        o = self.obs
        longitudinal = o.velocity[0]*math.cos(o.yaw)+o.velocity[1]*math.sin(o.yaw)
        opposing_damping = (self.emergency and command.omega == 0.
                            and command.vx*longitudinal < 0.)
        checked_recovery = (checked_intent and escape and self.recovery_episode is not None
                            and command.omega == 0. and command.vx*self.direction > 0.)
        checked_pause = self._checked_recovery_pause(command)
        permission = (self.recovery_permission
                      if checked_recovery or opposing_damping or checked_pause else None)
        stopping_time = o.speed/.4
        drift = tuple(o.xy[i]+o.velocity[i]*(.25+.5*stopping_time) for i in (0, 1))
        rotation = o.yaw_rate*(.25+stopping_time)
        return self._physical_sweep(o.xy, drift, o.yaw,
            yaw_offsets=(min(0., rotation), max(0., rotation)), commanded=False,
            circular_support=False, no_go_exit=escape or checked_pause, directional_exit=permission or False)

    def _anticipation(self, command, *, escape=False):
        """Provisional stopping room at requested speed, before it is attained.

        This supplements the intended segment and actual drift checks; it is
        not a claim of measured braking performance or failure evidence.
        """
        o = self.obs
        longitudinal = o.velocity[0]*math.cos(o.yaw)+o.velocity[1]*math.sin(o.yaw)
        if not command.vx or (self.emergency and command.omega == 0.
                              and command.vx*longitudinal < 0.):
            return ''
        speed = max(abs(command.vx), o.speed)
        distance = .25*speed+speed*speed/(2.*.4)
        sign = 1 if command.vx > 0. else -1
        end = (o.xy[0]+sign*distance*math.cos(o.yaw),
               o.xy[1]+sign*distance*math.sin(o.yaw))
        rotation = o.yaw_rate*(.25+speed/.4)
        intended_rotation = command.omega*.5
        return self._physical_sweep(o.xy, end, o.yaw,
            yaw_offsets=(min(0., rotation, intended_rotation),
                         max(0., rotation, intended_rotation)),
            # The original command segment already checks powered grades.
            # Stopping reserve checks physical terrain, not a longer drive.
            commanded=False, circular_support=False, no_go_exit=escape)

    def _physical_sweep(self, start, end, yaw, **kwargs):
        g = self.grid
        reason = oriented_sweep_reason(g, start, end, yaw, **kwargs)
        if reason not in ('stale oriented swept footprint', 'unobserved or stale centre corridor') or not self.static_prior_enabled:
            return reason
        support = self.prior_support & g.known & np.isfinite(g.surface)
        if not self.static_retention_enabled:
            support &= np.abs(g.surface-self.prior_surface) <= g.MATCH_TOLERANCE
        reason = oriented_sweep_reason(g, start, end, yaw,
            observation_support=g.fresh | support, **kwargs)
        if not reason:
            self.prior_admissions += 1
            if self.obs.time-self.last_prior_event >= 1.:
                self.last_prior_event = self.obs.time
                self.attempt_events.append(dict(event='remote_static_survey_admission', time=self.obs.time,
                    position=list(start), endpoint=list(end), source='simulator_development',
                    support_mode='retained_static_coverage' if self.static_retention_enabled else 'survey_agreement',
                    reason=('original static coverage; current retained geometry checked; LiDAR freshness unchanged'
                            if self.static_retention_enabled else
                            'original survey agrees with current geometry; LiDAR freshness unchanged')))
        return reason

    def _emit(self, command):
        o = self.obs
        owner = ('EMERGENCY' if self.emergency else self.operator_mode.upper()
                 if self.operator_mode != 'auto' else self.button_state)
        previous_owner = self.output_owner
        self.output_owner = owner
        self.command = command
        self.previous_command = command
        self.last_output_time = o.time
        if command.vx and owner in ('DRIVE', 'RECOVER'):
            sign = 1 if command.vx > 0 else -1
            old = self.physical_execution
            if (old is None or previous_owner != owner or old['drive_sign'] != sign
                    or old['interval_end_time'] != o.time):
                self.physical_execution = dict(maneuver='DRIVE_FORWARD' if sign > 0 else 'DRIVE_REVERSE',
                    drive_sign=sign, heading=wrap_angle(o.yaw+(math.pi if sign < 0 else 0.)), body_yaw=o.yaw,
                    start_position=list(o.xy), last_position=list(o.xy), first_emitted_time=o.time,
                    interval_end_time=o.time, duration_s=0.)
        return command

    def _start_turn(self, desired):
        o = self.obs
        self.turn_target = desired
        error = wrap_angle(desired-o.yaw)
        self.turn_sign = 1 if error > 0 else -1
        self.turn_best, self.turn_progress_at = abs(error), o.time
        self.turn_deadline = o.time+30.
        self._button('TURN', o.time, 'hold yaw button until measured heading')

    def _begin_minimum_turn(self, desired, advance_vertex=False, deadline=None):
        if self.goal_mode=='any_exit' and abs(wrap_angle(desired-self.obs.yaw))<=checked_turn.MAX_CORRECTION:
            return self._begin_checked_turn(desired,advance_vertex,deadline)
        self.minimum_turn=dict(target=desired,incoming=self.obs.yaw,deadline=self.obs.time+30. if deadline is None else deadline,stable_since=None,
                               phase='settle',advance_vertex=advance_vertex)
        self._button('MINIMUM_SETTLE',self.obs.time,'climb ended at local minimum; release before turn')
        return Motion()

    def _try_checked_rejoin(self):
        o=self.obs
        rejected=[]
        for path in checked_turn.rejoins(self.path,self.vertex,o.xy,o.yaw):
            reason=exit_region.route_reason(self.grid,path,o.yaw,o.xy,self.exit_rim_radius,
                lambda a,b:remote_routes.physical_profile(self,[a,b])[0])
            heading=math.atan2(path[1][1]-o.xy[1],path[1][0]-o.xy[0])
            runtime_reason=checked_turn.runtime_reason(self,heading) if not reason else ''
            if reason or runtime_reason:
                if self.rejoin_refusals<4:
                    if reason:
                        runtime_reason=checked_turn.runtime_reason(checked_turn.diagnostic_copy(self),heading)
                    rejected.append(dict(path=path,route_reason=reason,runtime_reason=runtime_reason,
                        checks=checked_turn.continuation_audit(self,path)))
                continue
            self.path=path;self.vertex=1;self.path_index=0;self.target=tuple(path[1])
            self.mission_route=list(path);self.route_profile=self.grid.route_profile(path)
            # Changing the target cannot manufacture progress or renew patience.
            self.progress_anchor=o.xy
            self.attempt_events.append(dict(event='checked_route_rejoin',time=o.time,
                position=list(o.xy),path=path,patience_renewed=False))
            return heading
        checked_turn.record_refusal(self,rejected)
        return None

    def _begin_checked_turn(self, desired, advance_vertex=False, deadline=None):
        o=self.obs
        change=wrap_angle(desired-o.yaw)
        reason=checked_turn.static_reason(self.grid,o.xy,o.yaw,desired) or checked_turn.runtime_reason(self,desired)
        if not reason:
            suffix=self.path[self.vertex+int(advance_vertex):]
            reason=exit_region.route_reason(self.grid,[o.xy]+list(suffix),desired,o.xy,self.exit_rim_radius,
                lambda a,b:remote_routes.physical_profile(self,[a,b])[0])
        if reason:return self._recover('checked correction unavailable: '+reason)
        if abs(change)<=checked_turn.HEADING_TOLERANCE:
            if advance_vertex:self.vertex+=1
            self._button('DRIVE',o.time,'heading already within correction tolerance')
            return Motion()
        if self.correction_anchor is not None:
            point,axis=self.correction_anchor
            if np.dot(np.subtract(o.xy,point),axis)<.5:
                return self._recover('correction requires measured onward response')
        if self.recovery_episode is None:self.recovery_episode=o.time
        if self.recovery_actions>=8 or o.time-self.recovery_episode>=120.:
            return self._hold('bounded correction recovery exhausted',o.time)
        self.recovery_actions+=1
        deadline=min(o.time+6.,deadline) if deadline is not None else o.time+6.
        self.checked_turn=dict(target=desired,sign=1 if change>0 else -1,deadline=deadline,
            best=abs(change),progress_at=o.time,advance_vertex=advance_vertex)
        self.correction_anchor=(o.xy,(math.cos(desired),math.sin(desired)))
        self.recovery_onward_anchor=None
        self.minimum_turn=None
        self._button('CHECKED_TURN',o.time,'committed physical heading correction')
        self.attempt_events.append(dict(event='checked_turn_start',time=o.time,position=list(o.xy),
            incoming=o.yaw,target=desired,angle_rad=change,deadline=deadline,
            recovery_actions=self.recovery_actions,physical_failure=False))
        return self._checked_turn_tick()

    def _checked_turn_tick(self):
        o=self.obs;turn=self.checked_turn
        if o.time>=turn['deadline'] or o.time-turn['progress_at']>=3.:
            return self._recover('checked correction stopped making angular progress')
        reason=checked_turn.runtime_reason(self,turn['target'])
        if reason:return self._recover('checked correction envelope changed: '+reason)
        error=wrap_angle(turn['target']-o.yaw)
        if abs(error)<turn['best']-.03:turn['best']=abs(error);turn['progress_at']=o.time
        if abs(error)<=checked_turn.HEADING_TOLERANCE or turn['sign']*error<=0.:
            if turn['advance_vertex']:self.vertex+=1
            self.correction_anchor=(o.xy,(math.cos(turn['target']),math.sin(turn['target'])))
            self.recovery_onward_anchor=o.xy
            self.checked_turn=None
            self._button('DRIVE',o.time,'checked correction complete; observe onward response')
            self.attempt_events.append(dict(event='checked_turn_end',time=o.time,position=list(o.xy),
                reason='measured heading reached',patience_renewed=False))
            return Motion()
        return Motion(0.,turn['sign']*checked_turn.YAW_COMMAND)

    def _steep_correction(self, desired, deadline=None):
        rejoin=self._try_checked_rejoin()
        if rejoin is not None:
            desired=rejoin
            if abs(wrap_angle(desired-self.obs.yaw))<=checked_turn.HEADING_TOLERANCE:
                self._button('DRIVE',self.obs.time,'checked forward route rejoin')
                return Motion(self.cruise_speed,0.)
        if abs(wrap_angle(desired-self.obs.yaw))<=checked_turn.MAX_CORRECTION:
            return self._begin_checked_turn(desired,deadline=deadline)
        return self._recover('heading correction exceeds bounded angle')

    def _minimum_turn_tick(self):
        o=self.obs; maneuver=self.minimum_turn
        if o.time>=maneuver['deadline']:
            return self._recover('local-minimum settle/turn deadline')
        reason=turn_reason(self.grid,o.xy,o.yaw,maneuver['target'],
            stage_headings=(maneuver['incoming'],maneuver['target']))
        if reason:return self._recover('local-minimum turn unavailable: '+reason)
        contiguous=0.<self.observation_dt<=.041
        stationary=o.speed<.12 and contiguous
        if maneuver['phase']=='turn' and not stationary:
            maneuver['phase']='settle';maneuver['stable_since']=None
            self._button('MINIMUM_SETTLE',o.time,'turn drift; release and stabilize within original deadline')
        if maneuver['phase']=='settle':
            stable=stationary and abs(o.yaw_rate)<.12
            maneuver['stable_since']=(maneuver['stable_since'] if maneuver['stable_since'] is not None else o.time) if stable else None
            if maneuver['stable_since'] is None or o.time-maneuver['stable_since']<.6:
                return Motion()
            maneuver['phase']='turn'
            error=wrap_angle(maneuver['target']-o.yaw)
            maneuver['sign']=1 if error>0 else -1
            self._button('MINIMUM_TURN',o.time,'measured stability; yaw only at local slope minimum')
        error=wrap_angle(maneuver['target']-o.yaw)
        if abs(error)<.10 or maneuver['sign']*error<=0.:
            if maneuver['advance_vertex']:self.vertex+=1
            self.minimum_turn=None
            self._button('DRIVE',o.time,'minimum turn complete; next straight leg')
            return Motion()
        return Motion(0.,maneuver['sign']*.35)

    def _minimum_bend(self,index):
        if not 0<index<len(self.path)-1:return False
        a,b,c=self.path[index-1:index+2]
        angle=wrap_angle(math.atan2(c[1]-b[1],c[0]-b[0])-math.atan2(b[1]-a[1],b[0]-a[0]))
        if self.goal_mode=='any_exit' and abs(angle)<=checked_turn.MAX_CORRECTION:return False
        return abs(angle)>.10 and self.grid.max_support_slope(b)>self.grid.motion_turn_limit

    def _new_progress(self):
        o = self.obs
        self.progress_anchor = o.xy
        self.progress_time = o.time
        self.low_progress_active = 0.

    def _recover(self, reason, physical=False):
        self.recovery_pause = None
        o = self.obs
        self.minimum_turn = None
        self.checked_turn = None
        self.recovery_onward_anchor = None
        new_episode = self.recovery_episode is None
        if new_episode:
            self.recovery_episode, self.recovery_actions = o.time, 0
        if self.recovery_actions >= 8 or o.time-self.recovery_episode >= 120.:
            return self._hold('bounded remote recovery exhausted: '+reason, o.time)
        self.finish_attempt('failure' if physical else 'censored', reason, o.xy, o.yaw, o.time,
            physical=physical, execution=self.physical_execution, terminal_maneuver=self.button_state)
        if new_episode or self.recovery_permission is None:
            self.recovery_permission = RecoveryPermission.capture(self.grid, o.xy, o.time)
            self.attempt_events.append(dict(event='remote_departure_permission', time=o.time,
                occupied_heading_patches=len(self.recovery_permission.patches)))
        candidates = []
        for sign in (1, -1):
            for distance in (1., 2., 3.):
                end = (o.xy[0]+sign*distance*math.cos(o.yaw),
                       o.xy[1]+sign*distance*math.sin(o.yaw))
                if self._physical_sweep(o.xy, end, o.yaw,
                        commanded=True, circular_support=False, no_go_exit=True):
                    continue
                gentle = self.grid.max_support_slope(end) <= 15. and not self.grid.turn_support_reason(end)
                connected = False
                if self.path and self.target is not None and math.dist(end, self.target) > .05:
                    error = abs(wrap_angle(math.atan2(self.target[1]-end[1], self.target[0]-end[0])-o.yaw))
                    advance = math.dist(o.xy, self.target)-math.dist(end, self.target)
                    connected = (advance >= .3 and error <= (.22 if gentle else .35)
                        and not self._physical_sweep(end, self.target, o.yaw,
                            commanded=True, circular_support=False))
                tier = 0 if connected else 1 if gentle else 2
                if tier == 2 and distance != 1.:
                    continue
                goal = self.waypoints[self.wp_idx]
                fallback = (math.dist(end,self.selected_exit) if self.selected_exit is not None else 0.
                            ) if self.goal_mode == 'any_exit' else math.dist(end,(goal.x,goal.y))
                score = min((math.dist(end, p) for p in self.path[self.vertex:]),
                            default=fallback)
                candidates.append((tier, distance, score, sign, end))
        if not candidates:
            return self._hold('no physical forward/reverse escape: '+reason, o.time)
        tier, distance, _, self.direction, self.recovery_endpoint = min(candidates)
        self.recovery_kind = ('mission connection', 'gentle staging', 'checked egress')[tier]
        self.recovery_best_remaining = self.recovery_credited_remaining = distance
        self.recovery_progress_at = o.time
        self.recovery_start = o.xy
        self.recovery_route = tuple(map(tuple, self.path))
        self.recovery_target = self.target
        self.recovery_waypoint = self.wp_idx
        self.recovery_actions += 1
        self.escape_events.append(dict(event='maneuver_start', time=o.time, direction=self.direction,
            endpoint=self.recovery_endpoint, deadline=self.recovery_episode+120.,
            no_progress_allowance_s=8., purpose=self.recovery_kind, reason=reason))
        self.set_state('RECOVER', reason, o.time)
        self._button('RECOVER', o.time, reason)
        return Motion(self.direction*self.cruise_speed, 0.)

    def _navigate(self):
        o = self.obs
        now = o.time
        if now < 3.:
            return Motion()
        if self.wp_idx >= len(self.waypoints):
            self.recovery_permission = None
            self.recovery_pause = None
            self.set_state('COMPLETE', 'all destinations reached and settled', now)
            self._button('COMPLETE', now, self.reason)
            return Motion()
        goal = self.waypoints[self.wp_idx]
        radius = goal.arrival_radius
        goal_distance = math.dist(o.xy, (goal.x, goal.y))
        arrived = (not exit_region.reason(self.grid,o.xy,self.exit_rim_radius)
                   if self.goal_mode == 'any_exit' else goal_distance <= radius)
        if arrived:
            self._button('SETTLE', now, 'release at mission destination')
            stable = o.speed < .12 and abs(o.yaw_rate) < .12 and 0. < self.observation_dt <= .041
            self.goal_settle_since = (self.goal_settle_since if self.goal_settle_since is not None else now) if stable else None
            if self.goal_settle_since is not None and now-self.goal_settle_since >= .6:
                # Commit only after this tick's physical output gate succeeds.
                self.pending_goal_success = True
            return Motion()
        self.goal_settle_since = None
        if self.checked_turn is not None:return self._checked_turn_tick()
        if self.button_state == 'RECOVER':
            remaining = math.dist(o.xy, self.recovery_endpoint)
            self.recovery_best_remaining = min(self.recovery_best_remaining, remaining)
            if self.recovery_credited_remaining-self.recovery_best_remaining >= .1:
                self.recovery_credited_remaining = self.recovery_best_remaining
                self.recovery_progress_at = now
            axis = np.subtract(self.recovery_endpoint, self.recovery_start)
            passed_endpoint = np.dot(np.subtract(o.xy, self.recovery_start), axis) >= np.dot(axis, axis)
            reached = remaining <= .25
            stalled = now-self.recovery_progress_at >= 8.
            if reached or passed_endpoint or stalled:
                self.recovery_pause = (dict(episode=self.recovery_episode,
                    start=tuple(self.recovery_start), endpoint=tuple(self.recovery_endpoint),
                    direction=self.direction) if stalled and not reached and not passed_endpoint else None)
                result = 'endpoint reached' if reached else 'endpoint passed outside arrival disk' if passed_endpoint else 'recovery made no useful progress'
                self.attempt_events.append(dict(event='remote_maneuver_end', time=now,
                    result=result, remaining_m=remaining, purpose=self.recovery_kind,
                    unfinished_zero_handoff=bool(self.recovery_pause),
                    position=list(o.xy), onward_proven=False,
                    observed_gentle_staging=bool(self.grid.max_support_slope(o.xy) <= 15.
                        and not self.grid.turn_support_reason(o.xy))))
                same_route = (bool(self.path) and tuple(map(tuple, self.path)) == self.recovery_route
                    and self.target == self.recovery_target and self.wp_idx == self.recovery_waypoint
                    and self.recovery_target is not None)
                if reached and same_route:
                    desired = math.atan2(self.target[1]-o.xy[1], self.target[0]-o.xy[0])
                    error = abs(wrap_angle(desired-o.yaw))
                    advance = math.dist(self.recovery_start, self.target)-math.dist(o.xy, self.target)
                    if advance >= .3 and error <= (.22 if o.inclination <= 15. else .35):
                        self.direction = 1
                        self.recovery_onward_anchor = o.xy
                        self.set_state('FOLLOW', 'productive recovery; retain original route', now)
                        self._button('DRIVE', now, self.reason)
                        self.attempt_events.append(dict(event='remote_route_resumed', time=now,
                            position=list(o.xy), target=list(self.target), advance_m=advance,
                            heading_error_rad=error, onward_proven=False))
                        return Motion(self.cruise_speed, 0.)
                self.path = []
                self.next_plan_at = now
                self.set_state('OBSERVE', result+'; replan from measured pose', now)
                self._button('OBSERVE', now, self.reason)
                return Motion()
            return Motion(self.direction*self.cruise_speed, 0.)
        if not self.path:
            self._button('OBSERVE', now, 'observe and plan')
            if now >= self.next_plan_at and self._plan(o.xy, o.yaw, now):
                self.recovery_pause = None
                self.vertex = 1
                self.direction = 1
                self.drive_exposure = 0.
                self.samples.clear()
                self.slow_since = None
                if self.goal_mode!='any_exit' or self.progress_anchor is None:
                    self._new_progress()
                else:
                    self.progress_anchor=o.xy
                if self.recovery_episode is not None:
                    self.recovery_onward_anchor = o.xy
                self._button('DRIVE', now, 'follow new A* route')
            elif not self.plan_pending and now >= self.next_plan_at:
                self.next_plan_at = now+2.
                if now-self.button_since > 15.:
                    return self._recover('planner has no route')
            return Motion()
        if self.minimum_turn is not None:
            return self._minimum_turn_tick()
        # Above-gentle bends finish the incoming climb before any yaw command.
        while self.vertex < len(self.path)-1 and math.dist(o.xy, self.path[self.vertex]) < .5:
            if self._minimum_bend(self.vertex):
                if math.dist(o.xy,self.path[self.vertex])<.25 and not stage_reason(self.grid,o.xy):
                    p,q=self.path[self.vertex:self.vertex+2]
                    return self._begin_minimum_turn(math.atan2(q[1]-p[1],q[0]-p[0]),True)
                break
            self.vertex += 1
        self.path_index = max(0, self.vertex-1)
        self.target = tuple(self.path[self.vertex])
        if self.vertex == len(self.path)-1 and math.dist(o.xy, self.target) < .4 and not self.is_mission_checkpoint:
            self.finish_attempt('success', 'observed route frontier reached', o.xy, o.yaw, now)
            self.path = []
            return Motion()
        desired = math.atan2(self.target[1]-o.xy[1], self.target[0]-o.xy[0])
        error = wrap_angle(desired-o.yaw)
        self.heading_error = error
        if self.button_state == 'TURN':
            if self.goal_mode=='any_exit' and self.grid.max_support_slope(o.xy)>self.grid.motion_turn_limit:
                return self._steep_correction(desired,deadline=self.turn_deadline)
            if stage_reason(self.grid,o.xy):
                return self._recover('turn translated onto steep terrain')
            if self.grid.max_support_slope(o.xy)>self.grid.motion_turn_limit:
                return self._begin_minimum_turn(self.turn_target,deadline=self.turn_deadline)
            residual = wrap_angle(self.turn_target-o.yaw)
            if abs(residual) < self.turn_best-.03:
                self.turn_best, self.turn_progress_at = abs(residual), now
            if self.turn_sign*residual <= .10:
                self._button('DRIVE', now, 'measured turn complete; resume travel')
            elif now >= self.turn_deadline or now-self.turn_progress_at >= 12.:
                return self._recover('turn stopped making useful angular progress')
            else:
                return Motion(0., self.turn_sign*.35)
        # Progress patience belongs to travel, not each individual button hold.
        if self.progress_anchor is None:
            self._new_progress()
        target_advance = math.dist(self.progress_anchor, self.target)-math.dist(o.xy, self.target)
        if target_advance >= .10:
            self._new_progress()
        elif now-self.progress_time >= 24. or self.low_progress_active >= 12.:
            return self._recover('travel made no useful progress within shared budget', physical=True)
        if self.button_state == 'RELEASE':
            self.reason = 'policy-active release; retain while progress is useful'
            return Motion()
        grade = self.metrics.get('directional_grade_deg', 0.)
        slow = self.metrics.get('window_seconds', 0.) >= 2.-1e-6 and (
            self.metrics['progress_mps'] < .10 or self.metrics.get('uphill_progress_mps', 0.) < .10)
        self.slow_since = (self.slow_since if self.slow_since is not None else now) if slow and grade >= 10. else None
        if self.drive_exposure >= 5. and self.slow_since is not None and now-self.slow_since >= 1.:
            self._button('RELEASE', now, 'powered uphill slowdown; release buttons')
            return Motion()
        can_turn=not stage_reason(self.grid,o.xy)
        special=self.grid.max_support_slope(o.xy)>self.grid.motion_turn_limit
        if self.goal_mode=='any_exit' and special and abs(error)>.22:
            return self._steep_correction(desired)
        if abs(error) > .65 and not can_turn:
            return self._recover('steep heading needs repositioning')
        if abs(error) > .22 and can_turn:
            # Keep driving the incoming leg until its planned minimum is
            # reached; do not turn early merely because a nearby tile qualifies.
            if special and self._minimum_bend(self.vertex):
                return Motion(*directional_command(forward=True,vx=self.cruise_speed))
            if special:return self._begin_minimum_turn(desired)
            self._start_turn(desired)
            return Motion(0., self.turn_sign*.35)
        self._button('DRIVE', now, 'hold forward while productive')
        return Motion(*directional_command(forward=True, vx=self.cruise_speed))

    def tick(self, xy, yaw, speed, now, upright=1., velocity=None):
        started = time.perf_counter()
        velocity = tuple(velocity[:2]) if velocity is not None else (speed*math.cos(yaw), speed*math.sin(yaw))
        if not np.isfinite([*xy, yaw, speed, now, upright, *velocity]).all():
            self.operator_mode = 'stop'
            self.recovery_permission = None
            self.recovery_pause = None
            self.command, self.output_owner = Motion(), 'STOP'
            self.last_output_time = None
            self.set_state('HOLD', 'nonfinite observation', now)
            return Motion()
        if self.last_tick is not None and now < self.last_tick:
            self.recovery_permission = None
            self.recovery_pause = None
        dt = max(0., now-self.last_tick) if self.last_tick is not None else 0.
        self.observation_dt = dt
        yaw_rate = wrap_angle(yaw-self.last_yaw)/dt if dt else 0.
        cell = self.grid.world_to_cell(*xy)
        inclination = math.degrees(math.atan(self.grid.slope[cell])) if cell is not None else math.nan
        gradient = (float(self.grid.gx[cell]), float(self.grid.gy[cell])) if cell is not None else (0., 0.)
        self.obs = MotionObservation(tuple(xy), yaw, velocity, upright, yaw_rate, now, inclination,
            inclination <= 15., cell is not None and bool(self.grid.fresh[cell]), gradient)
        if self.recovery_permission is not None:
            retained = self.recovery_permission.observe(self.grid, xy)
            if len(retained.patches) != len(self.recovery_permission.patches):
                self.attempt_events.append(dict(event='remote_departure_retired', time=now,
                    occupied_heading_patches=len(retained.patches), reason='observed exit'))
            self.recovery_permission = retained
        if self.recovery_pause is not None and not self._unfinished_recovery_pause():
            self.recovery_pause = None
        if (self.physical_execution is not None and self.last_output_time is not None and dt > 0
                and self.command.vx and self.output_owner in ('DRIVE', 'RECOVER')):
            self.physical_execution.update(last_position=list(xy), interval_end_time=now,
                duration_s=self.physical_execution['duration_s']+dt)
        if self.command.vx > 0 and self.output_owner == 'DRIVE':
            self.drive_exposure += dt
        if self.output_owner in ('DRIVE', 'RELEASE'):
            self.low_progress_active += dt
        self.last_xy, self.last_yaw, self.last_tick = tuple(xy), yaw, now
        self.actual_speed, self.upright = speed, upright
        self.samples.append((now, tuple(xy)))
        while len(self.samples) > 1 and self.samples[1][0] <= now-2.:
            self.samples.popleft()
        elapsed = now-self.samples[0][0]
        delta = np.subtract(xy, self.samples[0][1])/max(elapsed, 1e-9)
        axis = (math.cos(yaw), math.sin(yaw))
        gn = math.hypot(*gradient)
        progress = float(np.dot(delta, axis))
        forward = float(np.dot(velocity, axis))
        self.backslide_distance += max(0., -forward)*dt if self.button_state in ('DRIVE', 'RELEASE') else 0.
        self.metrics = dict(status=self.button_state, forward_mps=forward, progress_mps=progress,
            lateral_mps=-velocity[0]*axis[1]+velocity[1]*axis[0], window_seconds=elapsed,
            directional_grade_deg=math.degrees(math.atan(np.dot(gradient, axis))),
            uphill_progress_mps=float(np.dot(delta, gradient)/gn) if gn else 0.,
            release_active=self.button_state == 'RELEASE', forward_exposure_s=self.drive_exposure)
        if self.operator_mode == 'stop':
            return self._emit(Motion())
        if upright < .5:
            return self._emit(self._hold('excessive body tilt', now))
        if self.operator_mode == 'manual':
            return self._emit(self.manual)
        if now >= 900.:
            return self._emit(self._hold('mission deadline', now))
        if self.recovery_episode is not None and now-self.recovery_episode >= 120.:
            return self._emit(self._hold('recovery episode deadline', now))
        if now >= 3. and speed >= 1.:
            return self._emit(self._hold('terminal actual overspeed', now))
        if now >= 3. and speed >= .8 and not self.emergency:
            self.emergency = True
            self.emergency_clear_since = None
            self.attempt_events.append(dict(event='remote_emergency', time=now, reason='actual speed >=0.8'))
        if self.emergency:
            self.goal_settle_since = None
            if self.minimum_turn is not None:
                self.minimum_turn['phase']='settle'
                self.minimum_turn['stable_since']=None
            self.pending_goal_success = False
            self.emergency_clear_since = (self.emergency_clear_since if self.emergency_clear_since is not None else now) if speed < .3 else None
            if self.emergency_clear_since is not None and now-self.emergency_clear_since >= .5:
                self.emergency = False
                command = Motion()
            else:
                command = Motion(max(-.4, min(.4, -2.*forward)), 0.) if speed >= .3 else Motion()
        else:
            self.pending_goal_success = False
            command = self._navigate()
        if now >= 3. and self.operator_mode == 'auto':
            escape = self.button_state == 'RECOVER'
            reason = self._geometry(command, escape=escape)
            conflict = ''
            source = 'command'
            if reason and not self.emergency:
                # A failed intention cannot unlock occupied-patch permission.
                actual = self._actual_geometry(command, escape=escape)
                if not actual:
                    conflict, reason = reason, ''
                else:
                    reason = actual
            elif not reason:
                conflict = self._anticipation(command, escape=escape)
                source = 'forecast'
            if conflict and not self.emergency:
                end = (xy[0]+command.vx*.5*math.cos(yaw), xy[1]+command.vx*.5*math.sin(yaw))
                grade_only = (source == 'command' and bool(self.grid.grade_reason(xy, end))
                    and not self._command_geometry(command, escape=escape, check_grade=False))
                release = (grade_only and self.button_state == 'DRIVE' and self.output_owner == 'DRIVE'
                    and command.vx > 0. and command.omega == 0. and self.drive_exposure >= 5.
                    and self.metrics['directional_grade_deg'] >= 10.)
                self.attempt_events.append(dict(event='remote_command_fallback', time=now,
                    source=source, reason=conflict, position=list(xy), proposed_vx=command.vx,
                    proposed_omega=command.omega, selected='RELEASE' if release else 'RECOVER',
                    physical_failure=False))
                if release:
                    self._button('RELEASE', now, 'powered grade permission ended; policy-active release')
                    command = Motion()
                else:
                    command = self._recover('unavailable '+source+': '+conflict)
                # One replacement only, including independent ZERO validation.
                if self.operator_mode == 'auto':
                    reason = (self._geometry(command, escape=self.button_state == 'RECOVER')
                              or self._anticipation(command, escape=self.button_state == 'RECOVER'))
            elif conflict:
                reason = conflict  # Emergency never yields to ordinary recovery.
            if reason:
                self.attempt_events.append(dict(event='remote_geometry_stop', time=now, reason=reason,
                    position=list(xy), proposed_vx=command.vx, proposed_omega=command.omega))
                command = self._hold('physical envelope: '+reason, now)
        if (self.operator_mode == 'auto' and not self.emergency
                and self.recovery_onward_anchor is not None and self.target is not None
                and self.button_state in ('DRIVE', 'RELEASE')
                and math.dist(self.recovery_onward_anchor, self.target)-math.dist(xy, self.target) >= .5):
            self.attempt_events.append(dict(event='remote_recovery_complete', time=now,
                position=list(xy), reason='checked onward travel advanced at least 0.5 m'))
            self.recovery_episode = None
            self.recovery_permission = None
            self.recovery_pause = None
            self.recovery_onward_anchor = None
            self.recovery_actions = 0
        if self.pending_goal_success and self.operator_mode == 'auto' and not self.emergency:
            self.finish_attempt('success', 'destination reached and settled', xy, yaw, now)
            self.wp_idx += 1
            self.path, self.checkpoint, self.target = [], None, None
            self.goal_settle_since = None
            self.recovery_episode = None
            self.recovery_permission = None
            self.recovery_pause = None
            self.samples.clear()
            self._new_progress()
            self.set_state('OBSERVE', 'observe next mission destination', now)
            self._button('OBSERVE', now, self.reason)
            self.pending_goal_success = False
        self.tick_ms = (time.perf_counter()-started)*1000
        return self._emit(command)

    def motion_metrics(self):
        return dict(self.metrics)

    def navigation_status(self):
        status = super().navigation_status()
        status.update(button=self.button_state, emergency=self.emergency, protocol=self.protocol_version)
        return status
