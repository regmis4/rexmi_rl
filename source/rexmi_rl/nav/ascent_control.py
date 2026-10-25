"""Mission-first ascent supervisor, sharing baseline geometric checks only.

One state machine owns attempts, recovery, and termination. CheckpointController
remains unchanged and selectable as the baseline implementation.
"""
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
import copy
import itertools
import math
import time

import numpy as np

from .checkpoint_control import CheckpointController, Motion, path_progress, wrap_angle
from .ascent_routes import observed_route
from .reachability import (ApproachEvidence, AttemptOutcome, CapabilityProfile,
                           NavigationStatus, ReachabilityModel, eligibility, route_features)


class AscentController(CheckpointController):
    def __init__(self, grid, waypoints, *, model=None, mode='qualified', run_id='ascent', async_planning=False, **kwargs):
        super().__init__(grid, waypoints, **kwargs)
        if mode not in ('qualified', 'characterize'):
            raise ValueError('Use qualified or characterize mode')
        if self.cruise_speed > .4:
            raise ValueError('Ascent qualification is capped at 0.40 m/s on all terrain')
        self.capability = CapabilityProfile(slope_speed=self.cruise_speed, flat_speed=self.cruise_speed,
                                            preferred_uphill_deg=self.preferred_uphill,
                                            uphill_limit_deg=grid.uphill_limit_deg,side_limit_deg=grid.side_limit_deg)
        self.model = model or ReachabilityModel()
        self.mode, self.run_id = mode, run_id
        self.evidence = ApproachEvidence()
        self.attempt = None
        self.attempt_number = 0
        self.attempt_events = []
        self.reachability = None
        self.last_xy, self.last_yaw = (0., 0.), 0.
        self.actual_speed = 0.
        self.settle_samples = []
        self.recovery_anchor_v2 = None
        self.recovery_start_v2 = None
        self.recovery_tried = set()
        self.recovery_count = 0
        self.maneuver = None
        self.recovery_phase = 'adjust'
        self.stage = 'ascent'
        self.plan_ms = 0.
        self.tick_ms = 0.
        self.next_plan_at = 0.
        self.command_limit = self.capability.slope_speed
        self.visited_checkpoints = []
        self.alignment_best = math.inf
        self.alignment_progress_at = 0.
        self.planner = ThreadPoolExecutor(max_workers=1, thread_name_prefix='ascent-plan') if async_planning else None
        self.plan_job = None
        self.plan_pending = False
        self.plan_generation = 0
        self.deferred_approaches = []
        self.recovery_settling = False
        self.regroup_required = False
        self.regroup_goal = None
        self.regroup_started = None
        self.regroup_origin = None
        self.unproductive_frontiers = []
        self.stall_since = None
        self.tracking_lookahead=.8
        self.brake_anchor_v2=None
        self.brake_time_v2=None
        self.recovery_diagnostics={}
        self.last_recovery_report=None
        self.recovery_plan_started=None

    def close(self):
        if self.planner is not None:
            self.planner.shutdown(wait=True, cancel_futures=True)

    def choose_checkpoint(self, xy, target, now, yaw=0.):
        accepted=super().choose_checkpoint(xy,target,now,yaw)
        if accepted and not self.last_selection.get('duplicate'):
            self.regroup_required=False
            self.regroup_goal=self.regroup_origin=self.regroup_started=None
            self.retreat_goal=None
            self.route_purpose='mission'
        return accepted

    def request(self, action, now, vx=0., omega=0.):
        if hasattr(self, 'plan_generation'):
            self.plan_generation += 1
            self.plan_pending = False
        if hasattr(self, 'attempt'):
            self.finish_attempt('censored', 'operator '+action, self.last_xy, self.last_yaw, now)
        super().request(action, now, vx, omega)
        if hasattr(self, 'maneuver'):
            self.maneuver = None
        if action == 'resume':
            self.next_plan_at = now
            self.recovery_anchor_v2 = None
            self.settle_samples = []

    def finish_attempt(self, outcome, reason, xy, yaw, now, physical=False, *, execution=None, terminal_maneuver=None):
        attempt = self.attempt
        if attempt is None:
            return
        self.attempt = None
        if attempt['id'] in self.evidence.completed:
            return
        self.evidence.finish(attempt['id'], attempt['key'], outcome)
        prediction = attempt['estimate']
        updated = self.evidence.estimate(attempt['key'], prediction.prior, prediction.reason)
        event = AttemptOutcome(attempt['id'], self.run_id, attempt['features'], attempt['time'], now,
                               list(attempt['start']), list(attempt['endpoint']), list(xy), yaw,
                               outcome, reason, asdict(prediction), asdict(updated), attempt['key'])
        event.physical = physical
        event.executed_traversal = execution
        event.terminal_maneuver = terminal_maneuver
        radius = max(.35, self.grid.footprint_radius+.1)
        directional = terminal_maneuver is None  # Preserve the selectable legacy supervisor's contract.
        if terminal_maneuver is not None:
            directional = bool(execution and execution['duration_s'] > 0.
                               and math.dist(execution['last_position'], xy) <= radius)
            event.exclusion_reason = ('locally executed traversal' if directional
                                      else 'no locally executed directional traversal')
        if outcome == 'failure' and physical:
            if directional:
                if execution is not None:
                    heading = execution['heading']
                else:
                    # Legacy supervisor retains its planned local tangent.
                    # BT never substitutes a plan for missing execution evidence.
                    segments=[]
                    for a,b in zip(attempt['path'],attempt['path'][1:]):
                        delta=np.subtract(b,a);squared=float(np.dot(delta,delta))
                        if squared<1e-9:continue
                        fraction=float(np.clip(np.dot(np.subtract(xy,a),delta)/squared,0,1))
                        distance=float(np.linalg.norm(np.asarray(a)+fraction*delta-np.asarray(xy)))
                        segments.append((distance,math.atan2(delta[1],delta[0])))
                    heading=min(segments)[1] if segments else attempt['heading']
                event.exclusion_applied = self.grid.mark_failure(attempt['id'], xy, heading, now, reason)
                cell=self.grid.world_to_cell(*xy)
                self.escape_events.append(dict(event='exclusion_update',time=now,attempt_id=attempt['id'],
                                           position=list(xy),heading=heading,
                                           radius_m=max(.35,self.grid.footprint_radius+.1),
                                           heading_bins=np.flatnonzero(self.grid.approach_blocked[cell]).tolist() if cell is not None else [],
                                           all_directions=bool(self.grid.no_go[cell]) if cell is not None else False,reason=reason))
            self.climb_failures += 1
            self.last_climb_event = reason
            self.slip_events.append(dict(event='physical_approach_failure', time=now,
                                         position=list(xy), reason=reason, attempt_id=attempt['id']))
        elif outcome == 'success':
            self.grid.record_traversal(attempt['path'], attempt['heading'])
        self.attempt_events.append(asdict(event))

    def _leg(self, route):
        """Stop at a supported shelf or the end of the observed corridor.

        An arbitrary distance along a steep climb is not a settling checkpoint.
        Keeping an observed climb in one leg avoids repeatedly removing its
        drive command merely because another 1.5 metres have been covered.
        """
        length = 0.; options = []
        for i in range(1, len(route)):
            length += math.dist(route[i-1], route[i])
            cell = self.grid.world_to_cell(*route[i])
            if cell is None:
                break
            if length >= 1.5 and self.grid.shelf_mask[cell]:
                score = float(self.grid.local_slope[cell]) if np.isfinite(self.grid.local_slope[cell]) else float(self.grid.slope[cell])
                visited = any(math.dist(route[i], point) < .8 for point in self.visited_checkpoints)
                if not visited:options.append((score+.06*abs(length-3.),i))
            if length >= 6.:
                break
        if length < 1.5 or len(route) <= 2:
            return route
        end = min(options)[1] if options else len(route)-1
        return route[:end+1]

    def _retreat_exists(self, path):
        # A checked geometric reverse corridor is a prerequisite, not a
        # prediction that descent succeeds. Check from the candidate endpoint.
        backwards = list(reversed(path))
        return all(self.grid.segment_clear(a, b, require_fresh=False)
                   and not self.grid.support_reason(a, b) for a, b in zip(backwards, backwards[1:]))

    def _adoption_edge_reason(self, a, b, xy, retreat):
        if (not self.grid.segment_clear(a, b, require_fresh=math.dist(xy, a)<2., no_go_exit=retreat)
                or self.grid.support_reason(a, b)):
            return 'approach geometry or required observations changed'
        return ''

    def _adoption_route_reason(self, path, xy, yaw, shadow):
        return ''

    def _adopt_route_metadata(self, shadow):
        pass

    def _plan(self, xy, yaw, now):
        if self.planner is None:
            return self._compute_plan(xy, yaw, now)
        if self.plan_job is None:
            shadow = copy.copy(self)
            shadow.grid = copy.copy(self.grid)
            # Planning consumes geometry, not raw scan bins. Copy only inputs
            # it reads so map rebuilds cannot race with route construction.
            from .planning_snapshot import ARRAYS
            fields = ARRAYS
            for field in fields:
                setattr(shadow.grid, field, getattr(self.grid, field).copy())
            shadow.grid.failure_records=copy.deepcopy(self.grid.failure_records)
            shadow.evidence = copy.deepcopy(self.evidence)
            shadow.attempt_events = []
            shadow.slip_tracker = copy.deepcopy(self.slip_tracker)
            shadow.visited_checkpoints = list(self.visited_checkpoints)
            shadow.unproductive_frontiers = list(self.unproductive_frontiers)
            shadow.planner = None
            generation = self.plan_generation
            def compute():
                success = shadow._compute_plan(xy, yaw, now)
                return generation, success, shadow
            self.plan_job = self.planner.submit(compute)
            self.plan_pending = True
            return False
        if not self.plan_job.done():
            self.plan_pending = True
            return False
        generation, success, shadow = self.plan_job.result()
        self.plan_job = None
        self.plan_pending = False
        self.plan_ms = shadow.plan_ms
        if generation != self.plan_generation:
            self.planning_failure = 'discarded plan after operator action'
            return False
        self.route_options, self.route_searches = shadow.route_options, shadow.route_searches
        self.planning_failure = shadow.planning_failure
        if not success:
            return False
        path = [tuple(xy)]+shadow.path[1:]
        rejection=self._adoption_route_reason(path,xy,yaw,shadow)
        if rejection:
            self.planning_failure=rejection
            return False
        if getattr(self.grid, 'motion_turn_limit', None) is not None:
            from .staged_routes import motion_route_reason
            rejection = motion_route_reason(self.grid, path, yaw)
            if rejection:
                self.planning_failure = rejection
                return False
        retreat = shadow.route_purpose == 'retreat'
        moved = math.dist(xy, shadow.path[0]) > .3 or abs(wrap_angle(yaw-shadow.last_yaw)) > .35
        rejected = [] if moved else [(index, a, b, reason) for index, (a, b) in enumerate(zip(path, path[1:]))
                                    if (reason := self._adoption_edge_reason(a, b, xy, retreat))]
        if moved or rejected:
            self.planning_failure = 'plan changed while collecting observations; replan from measured pose'
            def age(point):
                cell = self.grid.world_to_cell(*point)
                seen = self.grid.last_seen[cell] if cell is not None else -math.inf
                return max(0., now-float(seen)) if math.isfinite(seen) else None
            self.attempt_events.append(dict(event='plan_adoption_rejection', time=now,
                position=list(xy), reason='start pose changed' if moved else rejected[0][3],
                segments=[dict(index=i, start=list(a), end=list(b), reason=reason,
                               endpoint_observation_age_s=[age(a), age(b)]) for i, a, b, reason in rejected]))
            self.plan_pending = True
            return False
        features = route_features(self.grid,path,yaw,self.actual_speed,self.upright,shadow.attempt['features'][-1])
        p0, reason = self.model.predict(features)
        estimate = self.evidence.estimate(shadow.attempt['key'],p0,reason)
        if (self.mode == 'qualified'
                and eligibility(estimate,self._retreat_exists(path),self.capability) not in ('normal','fallback')):
            self.planning_failure = 'updated approach no longer meets probability gate'
            return False
        for name in ('path','checkpoint','path_index','mission_route','mission_route_complete','route_profile',
                     'is_mission_checkpoint','route_purpose','route_decision','attempt_number','attempt',
                     'regroup_goal','regroup_origin','regroup_started','retreat_goal','retreat_candidates'):
            setattr(self,name,getattr(shadow,name))
        self.path = path
        self._adopt_route_metadata(shadow)
        self.attempt.update(time=now,start=tuple(xy),features=features,estimate=estimate,path=list(path))
        self.reachability = asdict(estimate)
        event = shadow.attempt_events[-1]
        event.update(time=now,features=features,prediction=self.reachability)
        self.attempt_events.append(event)
        self.best_remaining,self.progress_at = math.inf,now
        self.alignment_best,self.alignment_progress_at = math.inf,now
        self.settle_samples=[]
        self.tracking_lookahead=.8
        self.slip_tracker.reset()
        self.set_state('FOLLOW',self.route_decision,now)
        return True

    def _augment_routes(self, routes, xy, yaw):
        """Architecture-specific alternatives before any checkpoint is admitted."""
        return routes

    def _route_rejection_reason(self, route, xy, yaw):
        return 'route fails execution geometry'

    def _route_signature(self, route, leg):
        return tuple(tuple(p) for p in leg)

    def _select_route_candidate(self, candidates, xy, yaw):
        return min(candidates, key=lambda candidate: candidate[1])

    def _propose_routes(self, xy, yaw, support, route_limit):
        goal = (self.waypoints[self.wp_idx].x, self.waypoints[self.wp_idx].y)
        routes = []
        search_goal=goal
        profiles = [('direct', 'direct', route_limit), ('contour', 'gentle', route_limit)]
        if getattr(self.grid, 'traversal_objective', None):
            profiles = profiles[:1]  # Both labels use the same shared objective.
        for label, profile, grade in profiles:
            path, complete = observed_route(self.grid, xy, search_goal, yaw, profile=profile, support=support,
                                            uphill_limit=grade, strict_frontier=True,allow_frontier=label=='direct',
                                            frontier_penalties=self.unproductive_frontiers)
            self.route_searches.append(dict(profile=label, connected=complete, has_route=len(path)>1,
                                            uphill_limit=grade,**self.grid.last_route_search))
            if len(path) > 1:
                routes.append((label, path, complete and math.dist(path[-1],goal)<.1))
        # Two bounded shelf alternatives. Require onward connectivity to an
        # observed route/frontier; an isolated flat patch isn't a destination.
        for i, shelf in enumerate(self.grid.shelf_candidates(xy, goal, limit=2)):
            prefix, reached = observed_route(self.grid, xy, shelf, yaw, support=support, allow_frontier=False,
                                              uphill_limit=route_limit)
            if not reached or len(prefix) < 2:
                self.route_options.append(dict(profile=f'shelf-{i}',selected=False,complete=False,
                                                endpoint=list(shelf),path=prefix,score=None,
                                                rejection='shelf has no observed footprint-supported approach',
                                                **self.grid.route_profile(prefix)))
                continue
            heading = math.atan2(prefix[-1][1]-prefix[-2][1], prefix[-1][0]-prefix[-2][0])
            onward, done = observed_route(self.grid, shelf, search_goal, heading, support=support,
                                           allow_frontier=False, require_fresh_start=False,uphill_limit=route_limit)
            if done and len(onward) > 1 and math.dist(onward[-1], goal) < math.dist(shelf, goal)-.5:
                routes.append((f'shelf-{i}', prefix+onward[1:], math.dist(onward[-1],goal)<.1))
            else:
                self.route_options.append(dict(profile=f'shelf-{i}',selected=False,complete=False,
                                                endpoint=list(shelf),path=prefix,score=None,
                                                rejection='shelf has no checked onward route or useful frontier',
                                                **self.grid.route_profile(prefix)))
        return self._augment_routes(routes, xy, yaw)

    def _compute_plan(self, xy, yaw, now):
        started = time.perf_counter()
        self.route_options, self.route_searches = [], []
        if self.regroup_required:
            result=self._compute_regroup(xy,yaw,now)
            self.plan_ms=(time.perf_counter()-started)*1000
            return result
        if self.wp_idx >= len(self.waypoints):
            return False
        goal=(self.waypoints[self.wp_idx].x,self.waypoints[self.wp_idx].y)
        support = self.grid.planning_support()
        route_limit=max(.1,self.grid.uphill_limit_deg-getattr(self.grid,'planning_grade_reserve',1.))
        routes = self._propose_routes(xy,yaw,support,route_limit)
        candidates, signatures = [], set()
        for label, route, complete in routes:
            leg = self._leg(route)
            signature = self._route_signature(route,leg)
            if signature in signatures or len(leg) < 2 or math.dist(xy, leg[-1]) < getattr(self,'minimum_leg_distance',.65):
                continue
            signatures.add(signature)
            features = route_features(self.grid, leg, yaw, self.actual_speed, self.upright,self.command_limit)
            p0, reason = self.model.predict(features)
            heading = math.atan2(leg[-1][1]-leg[-2][1], leg[-1][0]-leg[-2][0])
            key = self.evidence.key(xy, leg[-1], heading, self.capability.slope_speed,path=leg)
            estimate = self.evidence.estimate(key, p0, reason)
            retreat = self._retreat_exists(leg)
            tier = eligibility(estimate, retreat, self.capability)
            allowed = tier in ('normal', 'fallback') or self.mode == 'characterize'
            adjustment_cost=self._recent_approach_cost(route,now)
            score = self.route_score(route, yaw)+adjustment_cost
            info = dict(profile=label, complete=complete, selected=False,
                        score=score if math.isfinite(score) else None,
                        endpoint=list(leg[-1]), path=leg, route_path=route, reachability=asdict(estimate),
                        tier=tier, retreat=retreat, recent_approach_cost=adjustment_cost,
                        rejection=(self._route_rejection_reason(route,xy,yaw) if not math.isfinite(score) else '' if allowed
                                   else reason if tier=='unvalidated' else 'below attempt threshold'),
                        **self.grid.route_profile(route))
            self.route_options.append(info)
            if allowed and math.isfinite(score):
                candidates.append((tier, score, route, complete, leg, features, estimate, key, heading, info))
        self.plan_ms = (time.perf_counter()-started)*1000
        if not candidates:
            self.planning_failure = ('reachability unvalidated; collect characterization outcomes and fit a model'
                                     if self.route_options and self.mode == 'qualified' else 'no eligible observed ascent approach')
            return False
        if any(c[0] == 'normal' for c in candidates):
            candidates = [c for c in candidates if c[0] == 'normal']
        if any(c[3] for c in candidates):
            candidates = [c for c in candidates if c[3]]
        selected = self._select_route_candidate(candidates, xy, yaw)
        if selected is None:
            return False
        _, _, route, complete, leg, features, estimate, key, heading, info = selected
        info['selected'] = True
        self.path, self.checkpoint, self.path_index = leg, tuple(leg[-1]), 0
        self.mission_route, self.mission_route_complete = route, complete
        self.route_profile = self.grid.route_profile(route)
        self.is_mission_checkpoint = complete and math.dist(leg[-1], goal) < .1
        self.route_purpose = 'mission'
        self.reachability = asdict(estimate)
        self.route_decision = f"{info['profile']}: {info['tier']}" + (' (characterization)' if self.mode=='characterize' else '')
        if info.get('paired_launch_variant'):
            self.route_decision += ' [paired_launch_variant: same_climb_class]'
        elif info.get('exploration'):
            self.route_decision += ' [untested corridor coverage experiment]'
        self.attempt_number += 1
        self.attempt = dict(id=f'{self.run_id}:{self.attempt_number}', time=now, start=tuple(xy),
                            endpoint=self.checkpoint, heading=heading, features=features, estimate=estimate,
                            key=key, path=list(leg), budget=max(45., features[0]/self.capability.slope_speed*5+30.))
        if info.get('exploration'):
            self.attempt['exploration'] = dict(info['exploration'])
        if info.get('paired_launch_variant'):
            self.attempt['paired_launch_variant'] = dict(info['paired_launch_variant'])
        self.attempt_events.append(dict(event='approach_start', attempt_id=self.attempt['id'], time=now,
                                       run_id=self.run_id, features=features, prediction=self.reachability,
                                       endpoint=list(self.checkpoint), approach_key=key, mode=self.mode))
        self.best_remaining = math.inf
        self.progress_at = now
        self.alignment_best = math.inf
        self.alignment_progress_at = now
        self.settle_samples = []
        self.tracking_lookahead=.8
        self.slip_tracker.reset()
        self.set_state('FOLLOW', self.route_decision, now)
        return True

    def _recent_approach_cost(self, route, now):
        """Temporarily price the rejected local section, never its destination.

        A predicted corridor rejection is not a physical no-go. Another path
        to the same exterior goal must remain eligible, and even the same
        approach may become executable after a bounded repositioning maneuver.
        """
        penalty=0.
        for old in self.deferred_approaches:
            if now>=old['until']:continue
            for a,b in zip(route,route[1:]):
                delta=np.subtract(b,a);length=float(np.linalg.norm(delta))
                if length<1e-9:continue
                fraction=float(np.clip(np.dot(np.subtract(old['position'],a),delta)/(length*length),0,1))
                distance=math.dist(np.asarray(a)+fraction*delta,old['position'])
                angle=abs(wrap_angle(math.atan2(delta[1],delta[0])-old['heading']))
                if distance<.5 and angle<.45:
                    penalty+=20.*(1.-distance/.5)
                    break
        return penalty

    def _compute_regroup(self, xy, yaw, now):
        """A latched lower destination, never a sequence of unrelated nudges."""
        g=self.grid;cell=g.world_to_cell(*xy)
        if cell is None or not g.known[cell]:return False
        support=g.planning_support()
        if self.regroup_origin is None:
            self.regroup_origin=tuple(xy)
            self.regroup_started=now
        if self.regroup_goal is None:
            rr,cc=np.indices(g.known.shape)
            xx=g.origin[0]+(rr+.5)*g.cell_size-g.world_size/2
            yy=g.origin[1]+(cc+.5)*g.cell_size-g.world_size/2
            distance=np.hypot(xx-xy[0],yy-xy[1])
            possible=(g.known & support & ~g.blocked & (g.surface<=g.surface[cell]-.4)
                      & (g.slope<=math.tan(math.radians(20))) & (distance>=2.) & (distance<=7.))
            candidates=[]
            for rc in sorted(np.argwhere(possible),key=lambda p:distance[tuple(p)]+3*g.slope[tuple(p)]):
                goal=g.cell_to_world(*rc)
                if all(math.dist(goal,p)>1. for p in candidates):candidates.append(goal)
                if len(candidates)>=6:break
        else:candidates=[self.regroup_goal]
        choices=[];self.retreat_candidates=[]
        for goal in candidates:
            path,done=observed_route(g,xy,goal,yaw,support=support,uphill_limit=8.,
                                     no_go_exit=True,allow_frontier=False)
            profile=g.route_profile(path)
            self.retreat_candidates.append(dict(goal=list(goal),reachable=done,**profile))
            if done and len(path)>1:choices.append((profile['length_m'],goal,path))
        if not choices:
            if self.regroup_goal is not None:
                # A committed destination can become unreachable after a new
                # physical exclusion. Replace that invalid destination with a
                # checked alternative; retain the original retreat budget.
                self.regroup_goal=None
                return self._compute_regroup(xy,yaw,now)
            self.planning_failure='no checked route to lower regroup point; compare local escapes'
            return False
        _,goal,route=min(choices,key=lambda v:v[0])
        leg=self._leg(route)
        heading=math.atan2(leg[1][1]-xy[1],leg[1][0]-xy[0])
        reverse=abs(wrap_angle(heading-yaw))>math.pi/2
        command=-self.command_limit if reverse else self.command_limit
        features=route_features(g,leg,yaw,self.actual_speed,self.upright,command)
        p0,reason=self.model.predict(features)
        key=self.evidence.key(xy,leg[-1],heading,command,path=leg)
        estimate=self.evidence.estimate(key,p0,reason)
        if self.mode == 'qualified' and eligibility(estimate,self._retreat_exists(leg),self.capability) not in ('normal','fallback'):
            self.planning_failure='regroup reachability does not meet probability gate: '+reason
            return False
        self.regroup_goal=goal
        self.retreat_goal=goal
        self.route_options=[dict(profile='regroup',selected=True,complete=False,endpoint=list(leg[-1]),
                                 path=leg,score=features[0],reachability=asdict(estimate),
                                 rejection='',**g.route_profile(route))]
        self.reachability=asdict(estimate)
        self.path,self.checkpoint,self.path_index=leg,tuple(leg[-1]),0
        self.mission_route,self.mission_route_complete=route,False
        self.route_profile=g.route_profile(route)
        self.route_purpose='retreat';self.is_mission_checkpoint=False
        self.route_decision='committed lower regroup; original mission retained'
        self.attempt_number+=1
        self.attempt=dict(id=f'{self.run_id}:{self.attempt_number}',time=now,start=tuple(xy),
                          endpoint=self.checkpoint,heading=heading,features=features,estimate=estimate,
                          key=key,path=list(leg),budget=90.)
        self.attempt_events.append(dict(event='approach_start',attempt_id=self.attempt['id'],time=now,
                                        run_id=self.run_id,features=features,prediction=self.reachability,
                                        endpoint=list(self.checkpoint),approach_key=key,mode='checked downhill regroup'))
        self.best_remaining,self.progress_at=math.inf,now
        self.alignment_best,self.alignment_progress_at=math.inf,now
        self.settle_samples=[];self.slip_tracker.reset()
        self.tracking_lookahead=.8
        self.set_state('FOLLOW',self.route_decision,now)
        return True

    def _brake(self, xy, yaw):
        # Hold the onset of this braking episode, not an obsolete checkpoint
        # or heading. Velocity damping alone permits steady hillside creep.
        axis = np.array([math.cos(yaw), math.sin(yaw)])
        # A small reverse request did not arrest observed hillside creep.
        # Reach the existing command ceiling sooner when damping that drift;
        # stronger braking never raises the autonomous speed ceiling.
        cell=self.grid.world_to_cell(*xy)
        gain=4. if cell is not None and self.grid.known[cell] and self.grid.slope[cell]>math.tan(math.radians(15)) else 1.2
        now=self.last_tick if self.last_tick is not None else 0.
        if self.brake_time_v2 is None or now-self.brake_time_v2>.3:
            self.brake_anchor_v2=np.asarray(xy).copy()
        anchor_cell=self.grid.world_to_cell(*self.brake_anchor_v2)
        if cell is not None and anchor_cell is not None and self.grid.no_go[anchor_cell] and not self.grid.no_go[cell]:
            self.brake_anchor_v2=np.asarray(xy).copy()
        self.brake_time_v2=now
        correction=.8*np.dot(self.brake_anchor_v2-np.asarray(xy),axis)
        return Motion(float(np.clip(correction-gain*np.dot(self.velocity_world, axis), -.4, .4)), 0.)

    def motion_clear(self, xy, yaw, speed, motion):
        if super().motion_clear(xy,yaw,speed,motion):return True
        # Reconsider conservative obstacle margins with the measured body.
        # The broader slope-envelope exemption failed characterization, so
        # ground-slope/edge rejections retain the established circular guard.
        if not any(word in self.motion_rejection for word in ('obstacle','clearance')):
            return False
        if abs(motion.vx)<1e-6:return False
        distance=max(.8,max(speed,abs(motion.vx))**2/(2*max(.1,self.braking_accel or .2))+.3)
        duration=distance/abs(motion.vx)
        for omega in (0.,.35*motion.omega,motion.omega):
            points=[xy]
            for t in np.linspace(0,duration,max(12,math.ceil(distance/.05)))[1:]:
                dx=motion.vx*t if abs(omega)<1e-9 else motion.vx*math.sin(omega*t)/omega
                dy=0. if abs(omega)<1e-9 else motion.vx*(1-math.cos(omega*t))/omega
                points.append((xy[0]+dx*math.cos(yaw)-dy*math.sin(yaw),xy[1]+dx*math.sin(yaw)+dy*math.cos(yaw)))
            if not self._oriented_escape(points,yaw,omega,duration):return False
        self.motion_rejection=''
        return True

    def recover(self, xy, yaw, now, reason, physical=False):
        if self.attempt is not None and not physical:
            self.deferred_approaches = [item for item in self.deferred_approaches if item['until'] > now]
            _,_,ahead,_,_=path_progress(self.attempt['path'],xy,lookahead=.6)
            self.deferred_approaches.append(dict(position=tuple(ahead),
                heading=math.atan2(ahead[1]-xy[1],ahead[0]-xy[0]),until=now+20.))
        # A route invalidated after real travel failed to reach its endpoint,
        # but does not prove poor traction. Pre-motion rejection is censored.
        departed = self.attempt is not None and math.dist(xy,self.attempt['start']) >= .5
        if self.route_purpose=='mission' and self.mission_route and not self.mission_route_complete:
            # Failed exploration is useful search information without being
            # another physical failure vote. Keep it separate from exclusions
            # and probability evidence, so a retreat explores another exit.
            frontier=tuple(self.mission_route[-1])
            if all(math.dist(frontier,p)>1. for p in self.unproductive_frontiers):
                self.unproductive_frontiers.append(frontier)
                self.escape_events.append(dict(event='frontier_deferred',time=now,position=list(xy),
                                               endpoint=list(frontier),reason=reason))
        self.finish_attempt('failure' if physical or departed else 'censored', reason, xy, yaw, now,physical=physical)
        cell=self.grid.world_to_cell(*xy)
        if cell is not None and self.grid.known[cell] and self.grid.slope[cell]>math.tan(math.radians(15)):
            # A geometric rejection can need only a short backout and a new
            # alignment. Escalate to the latched lower destination after a
            # physical failure or two distinct local adjustments.
            local_count=self.recovery_count if self.recovery_anchor_v2 is not None and math.dist(xy,self.recovery_anchor_v2)<=1.5 else 0
            self.regroup_required |= physical or local_count>=2
        self.path, self.target, self.mission_route = [], None, []
        self.maneuver = None
        self.recovery_settling = False
        self.slip_tracker.reset()
        self.stall_since=None
        if self.recovery_anchor_v2 is None or math.dist(xy, self.recovery_anchor_v2) > 1.5:
            self.recovery_anchor_v2, self.recovery_start_v2 = tuple(xy), now
            self.recovery_tried, self.recovery_count = set(), 0
        self.recovery_phase = 'brake'
        self.recovery_diagnostics={}
        self.recovery_plan_started=None
        self.set_state('RECOVER', reason, now)
        self.next_plan_at = now+1.
        return self._brake(xy, yaw)

    def _recovery_candidates(self, xy, yaw, speed):
        candidates = []
        original_purpose=self.route_purpose
        rejected={}
        def reject(reason):
            label=reason or 'swept footprint or direction rejected'
            rejected[label]=rejected.get(label,0)+1
        goal = (self.waypoints[self.wp_idx].x, self.waypoints[self.wp_idx].y)
        exposure = self.grid.support_exposure(xy)
        if exposure is None:
            self.recovery_diagnostics={'unobserved recovery footprint':1}
            return []
        inherited, support_count = exposure
        for vx,distance in itertools.product((-.4, .4, -.2, .2),(.4,.8)):
            for omega in (0., -.12, .12):
                key = (round(xy[0]*2), round(xy[1]*2), round(yaw/.35), vx, omega, distance)
                if key in self.recovery_tried:
                    continue
                duration = distance/abs(vx)
                points = [tuple(xy)]
                for t in np.linspace(0, duration, 12)[1:]:
                    dx = vx*t if omega == 0 else vx*math.sin(omega*t)/omega
                    dy = 0. if omega == 0 else vx*(1-math.cos(omega*t))/omega
                    points.append((xy[0]+dx*math.cos(yaw)-dy*math.sin(yaw), xy[1]+dx*math.sin(yaw)+dy*math.cos(yaw)))
                # Reverse is a direction, not automatically a downhill leg.
                # A short reposition may need a small uphill section; apply
                # the normal grade limits, not the regroup route's 8° cap.
                self.route_purpose = 'adjust'
                motion = Motion(vx, omega)
                partial = bool(inherited) and len(inherited)/support_count <= .25
                if partial:
                    reason, _ = self.grid.support_exit_check(points, inherited)
                    # This escape only restores support; no new steep centre,
                    # obstacle, unknown ground, or speed exemption is allowed.
                    okay = not reason
                else:
                    okay = self.motion_clear(xy, yaw, speed, motion) and all(
                        self.grid.segment_clear(a, b, no_go_exit=True)
                        and not self.grid.support_reason(a, b) for a, b in zip(points, points[1:]))
                    reason = self.motion_rejection
                oriented=False
                if not okay:
                    oriented=self._oriented_escape(points,yaw,omega,duration)
                    okay=oriented
                if not okay:
                    reject(reason)
                    continue
                end = points[-1]
                cell = self.grid.world_to_cell(*end)
                lower = float(self.grid.surface[cell]-self.grid.surface[self.grid.world_to_cell(*xy)])
                # Prefer a useful displacement when the whole longer corridor
                # is checked. Goal distance alone selected tiny reverse nudges.
                # Local adjustment first creates room to execute the next leg;
                # being closer to the exterior cannot outweigh escaping a rock.
                room=min(1.5,float(self.grid.obstacle_distance[cell]))
                score = (.25*math.dist(end, goal)-1.5*math.dist(xy,end)
                         +2*max(0., lower)+3*abs(omega)-3*room)
                if vx < 0:
                    score -= .5
                if self.mode == 'qualified':
                    features=route_features(self.grid,points,yaw,speed,self.upright,vx)
                    prior,detail=self.model.predict(features)
                    heading=wrap_angle(yaw+(math.pi if vx<0 else 0.))
                    estimate=self.evidence.estimate(self.evidence.key(xy,end,heading,vx,path=points),prior,detail)
                    tier=eligibility(estimate,self._retreat_exists(points),self.capability)
                    if tier not in ('normal','fallback'):
                        reject('reachability '+tier);continue
                    # Normal attempts always rank ahead of fallback attempts.
                    score += 100000. if tier=='fallback' else 0.
                candidates.append((score, dict(key=key, motion=motion, path=points, endpoint=end,
                                               start=tuple(xy), yaw=yaw, partial=partial and not oriented,
                                               inherited=inherited, duration=duration,oriented=oriented)))
        self.recovery_diagnostics=rejected
        self.route_purpose=original_purpose
        return sorted(candidates, key=lambda c: c[0])

    def _oriented_escape(self, points, yaw, omega, duration):
        # Used only to leave a conservative circular overlap. The measured
        # rectangular body must remain entirely clear throughout the maneuver.
        g=self.grid
        if g.oriented_support_reason(points[0],yaw):return False
        original=set(zip(*g.oriented_cells(points[0],yaw)))
        drift=float(np.linalg.norm(self.velocity_world))
        if drift>.1:
            stopping=drift*drift/(2*max(.1,self.braking_accel or .2))+.1
            for distance in np.linspace(0,stopping,8):
                position=tuple(np.asarray(points[0])+distance*self.velocity_world/drift)
                if g.oriented_support_reason(position,yaw):return False
        depth=g.no_go_depth[g.world_to_cell(*points[0])]
        previous=points[0]
        for i,point in enumerate(points):
            heading=yaw+omega*duration*i/max(1,len(points)-1)
            if g.oriented_support_reason(point,heading):return False
            cell=g.world_to_cell(*point)
            if not g.fresh[cell] or g.no_go_depth[cell]>depth+1e-6:return False
            cells=set(zip(*g.oriented_cells(point,heading)))
            if any(not g.fresh[p] for p in cells-original):return False
            depth=g.no_go_depth[cell]
            if i:
                travel=g.heading_bin(math.atan2(point[1]-previous[1],point[0]-previous[0]))
                if g.grade_reason(previous,point) or (not g.no_go[cell] and not g.direction_allowed(cell,travel)):
                    return False
            previous=point
        # End where ordinary planning can resume; do not trade one overlap for
        # a new one. This is not permission to plow through a mapped boulder.
        return g.traversable(g.world_to_cell(*points[-1])) and not g.support_reason(points[-1])

    def _recover_tick(self, xy, yaw, speed, now):
        if now-self.recovery_start_v2 > 90.:
            self.finish_attempt('censored','recovery budget exhausted',xy,yaw,now)
            detail=('recovery timed out without an executable maneuver' if self.recovery_count==0
                    else 'recovery time budget exhausted after committed maneuvers')
            self.set_state('HOLD', detail+'; assistance required', now)
            return self._brake(xy, yaw)
        if self.recovery_settling:
            self.reason='settling after committed local adjustment'
            if now >= self.next_plan_at and speed < .15:
                self.finish_attempt('success','recovery endpoint reached and settled',xy,yaw,now)
                self.recovery_settling = False
            elif now > self.next_plan_at+5.:
                self.finish_attempt('failure','recovery endpoint could not settle',xy,yaw,now,physical=True)
                self.recovery_settling = False
            return self._brake(xy,yaw)
        if self.maneuver is not None:
            m = self.maneuver
            posture=m.setdefault('posture',[])
            posture.append((now,self.upright))
            while len(posture)>1 and posture[1][0]<=now-.6:posture.pop(0)
            tilt=math.acos(float(np.clip(self.upright,-1.,1.)))
            earlier=math.acos(float(np.clip(posture[0][1],-1.,1.)))
            if now-posture[0][0]>=.4 and self.upright<.8 and tilt-earlier>math.radians(8.):
                return self._failed_recovery_motion('rapidly worsening posture during adjustment',xy,yaw,now)
            if speed>.45:
                self.reason='braking measured overspeed during committed recovery'
                self.slip_tracker.reset();self.stall_since=None
                return self._brake(xy,yaw)
            travel_yaw=wrap_angle(yaw+(math.pi if m['motion'].vx<0 else 0.))
            heading_error=wrap_angle(math.atan2(m['endpoint'][1]-xy[1],m['endpoint'][0]-xy[0])-travel_yaw)
            _,_,_,_,cross=path_progress(m['path'],xy)
            monitoring=self.previous_command.vx*m['motion'].vx>0
            evidence=self.slip_tracker.update(now,xy,travel_yaw,self.velocity_world,
                Motion(abs(m['motion'].vx),m['motion'].omega),'FOLLOW' if monitoring else 'braking',heading_error,cross)
            if evidence and evidence['kind']=='poor drive progress':
                # The slope policy can make useful, stable progress far below
                # the requested velocity. Do not label that response a stall.
                if evidence['progress_mps']>=.02 and evidence['forward_mps']>=0:
                    self.stall_since=None;evidence=None
                    self.slip_tracker.metrics['status']='slow progress'
                else:
                    if self.stall_since is None:self.stall_since=now
                    if now-self.stall_since<.5:evidence=None
            else:self.stall_since=None
            if evidence:
                return self._failed_recovery_motion('adjustment '+evidence['kind'],xy,yaw,now)
            if (math.dist(xy, m['endpoint']) < .12
                    and self.grid.traversable(self.grid.world_to_cell(*xy))
                    and not self.grid.support_reason(xy)):
                self.escape_events.append(dict(event='maneuver_complete', time=now, position=list(xy)))
                self.maneuver = None
                self.next_plan_at = now+1.
                self.recovery_settling = True
                self.recovery_plan_started=None
                self.reason='braking at committed adjustment endpoint'
                return self._brake(xy, yaw)
            if (math.dist(xy,m['start'])>=math.dist(m['start'],m['endpoint'])+.15
                    and self.grid.traversable(self.grid.world_to_cell(*xy))
                    and not self.grid.support_reason(xy)):
                # Accept useful checked displacement without driving past a
                # bounded maneuver while chasing an idealized endpoint. This
                # did not reach the predicted endpoint, so censor its label.
                self.finish_attempt('censored','bounded displacement reached at a different measured endpoint',xy,yaw,now)
                self.escape_events.append(dict(event='maneuver_displacement_complete',time=now,position=list(xy)))
                self.maneuver=None;self.recovery_settling=True
                self.next_plan_at=now+1.;self.recovery_plan_started=None
                self.reason='settling at measured adjustment endpoint before replanning'
                return self._brake(xy,yaw)
            motion=m['motion']
            # Hold the intended approach against the policy's measured yaw
            # drift. Feedback receives the same complete stopping/arc checks.
            correction=Motion(motion.vx,float(np.clip(.8*heading_error,-.35,.35)))
            if abs(correction.omega-motion.omega)>.015 and self.motion_clear(xy,yaw,speed,correction):
                motion=correction
            allowed = self.motion_clear(xy, yaw, speed, motion)
            if m.get('oriented'):
                # Recheck a fresh arc from the actual pose, including observed
                # inertial drift, rather than trusting the initial prediction.
                remaining=min(m['duration'],max(.25,math.dist(xy,m['endpoint'])/abs(m['motion'].vx)))
                pts=[xy]
                vx,w=motion.vx,motion.omega
                for t in np.linspace(0,remaining,12)[1:]:
                    dx=vx*t if abs(w)<1e-9 else vx*math.sin(w*t)/w
                    dy=0. if abs(w)<1e-9 else vx*(1-math.cos(w*t))/w
                    pts.append((xy[0]+dx*math.cos(yaw)-dy*math.sin(yaw),xy[1]+dx*math.sin(yaw)+dy*math.cos(yaw)))
                allowed=self._oriented_escape(pts,yaw,w,remaining)
            if m['partial']:
                remaining = [xy]+[p for p in m['path'] if math.dist(p, m['endpoint']) < math.dist(xy, m['endpoint'])]
                detail, _ = self.grid.support_exit_check(remaining, m['inherited'])
                allowed = not detail
            if not allowed or now-m['time'] > m['budget']:
                self.escape_events.append(dict(event='maneuver_retired', time=now, position=list(xy),
                                               reason=self.motion_rejection or 'insufficient progress'))
                progress = math.dist(m['start'],m['endpoint'])-math.dist(xy,m['endpoint'])
                if now-m['time']<1. and progress<.1:
                    # A newly rejected prediction after a few control ticks
                    # is not a completed recovery effort. Keep its exact
                    # command deferred, but do not burn the maneuver budget.
                    self.recovery_count=max(0,self.recovery_count-1)
                physical = allowed and progress < .1 and now-m['time'] >= 3.
                self.finish_attempt('failure' if physical else 'censored',
                                    'recovery stall' if physical else self.motion_rejection or 'recovery time budget',xy,yaw,now,physical=physical)
                self.maneuver = None
                self.next_plan_at = now+1.
                self.reason='adjustment stopped: '+(self.motion_rejection or 'time budget')
                return self._brake(xy, yaw)
            progress=max(0.,math.dist(m['start'],m['endpoint'])-math.dist(xy,m['endpoint']))
            direction='reverse' if m['motion'].vx<0 else 'forward'
            self.reason=f'committed {direction} adjustment: {progress:.2f}/{math.dist(m["start"],m["endpoint"]):.2f} m'
            return motion
        if now < self.next_plan_at:
            return self._brake(xy, yaw)
        # Replan once per second, not on every map rebuild or physics tick.
        self.next_plan_at = now+1.
        cell=self.grid.world_to_cell(*xy)
        if self.recovery_count>=2 and cell is not None and self.grid.slope[cell]>math.tan(math.radians(15)):
            self.regroup_required=True
        if speed > .3:
            return self._brake(xy, yaw)
        # First clear the local problem with a checked, bounded maneuver.
        # Starting another full route search here used to preempt every
        # reverse opportunity and leave the robot damping drift indefinitely.
        choices=None
        if self.recovery_count==0:
            choices=self._recovery_candidates(xy,yaw,speed)
            if choices:return self._start_recovery_maneuver(choices[0][1],xy,yaw,speed,now)
        if self.recovery_plan_started is None:self.recovery_plan_started=now
        if speed <= .2 and self._plan(xy, yaw, now):
            return Motion()
        if self.plan_pending and now-self.recovery_plan_started<2.:
            self.reason='planning after local adjustment; bounded recovery remains available'
            return self._brake(xy,yaw)
        if self.recovery_count>=6:
            if self.plan_pending:return self._brake(xy,yaw)
            self.set_state('HOLD','distinct recovery approaches exhausted after observed replan; assistance required',now)
            return self._brake(xy,yaw)
        if choices is None:choices = self._recovery_candidates(xy, yaw, speed)
        if choices:
            return self._start_recovery_maneuver(choices[0][1],xy,yaw,speed,now)
        blockers=sorted(self.recovery_diagnostics.items(),key=lambda x:-x[1])
        self.reason = 'no executable local maneuver: '+(blockers[0][0] if blockers else self.planning_failure)
        report=(tuple(blockers),self.planning_failure)
        if report!=self.last_recovery_report:
            self.escape_events.append(dict(event='recovery_candidates_rejected',time=now,position=list(xy),
                                           reasons=dict(blockers),planning_reason=self.planning_failure))
            self.last_recovery_report=report
        return self._brake(xy, yaw)

    def _start_recovery_maneuver(self, maneuver, xy, yaw, speed, now):
        self.plan_generation+=1  # A queued plan cannot interrupt the commitment.
        self.plan_pending=False
        self.recovery_plan_started=None
        self.maneuver=maneuver
        maneuver['time']=now
        # A slow but progressing policy gets time to cover the same bounded
        # distance; command magnitude and the maneuver endpoint stay fixed.
        maneuver['budget']=min(20.,max(8.,10*maneuver['duration']))
        maneuver['posture']=[(now,self.upright)]
        self.slip_tracker.reset();self.stall_since=None
        self.recovery_tried.add(maneuver['key'])
        self.recovery_count+=1
        self.route_purpose='adjust'
        self.checkpoint=maneuver['endpoint'];self.path=maneuver['path']
        self.reason='committed reverse adjustment' if maneuver['motion'].vx<0 else 'committed forward adjustment'
        features=route_features(self.grid,self.path,yaw,speed,self.upright,maneuver['motion'].vx)
        p0,detail=self.model.predict(features)
        heading=wrap_angle(yaw+(math.pi if maneuver['motion'].vx<0 else 0.))
        key=self.evidence.key(xy,self.checkpoint,heading,maneuver['motion'].vx,path=self.path)
        estimate=self.evidence.estimate(key,p0,detail)
        self.reachability=asdict(estimate)
        self.attempt_number+=1
        self.attempt=dict(id=f'{self.run_id}:{self.attempt_number}',time=now,start=tuple(xy),
                          endpoint=self.checkpoint,heading=heading,features=features,estimate=estimate,
                          key=key,path=list(self.path),budget=maneuver['budget'])
        self.attempt_events.append(dict(event='approach_start',attempt_id=self.attempt['id'],time=now,
                                        run_id=self.run_id,features=features,prediction=self.reachability,
                                        endpoint=list(self.checkpoint),approach_key=key,mode='geometrically checked recovery'))
        self.escape_events.append(dict(event='maneuver_start',time=now,position=list(xy),
                                       endpoint=list(self.checkpoint),vx=maneuver['motion'].vx,
                                       omega=maneuver['motion'].omega,reason=self.reason))
        return maneuver['motion']

    def _failed_recovery_motion(self, reason, xy, yaw, now):
        """An observed motion failure retires one independent attempt."""
        self.escape_events.append(dict(event='maneuver_retired',time=now,position=list(xy),reason=reason))
        self.finish_attempt('failure',reason,xy,yaw,now,physical=True)
        self.maneuver=None
        self.next_plan_at=now+.5
        self.recovery_plan_started=None
        self.slip_tracker.reset();self.stall_since=None
        cell=self.grid.world_to_cell(*xy)
        if cell is not None and self.grid.slope[cell]>math.tan(math.radians(15)):
            self.regroup_required=True
        self.reason='adjustment stopped: '+reason
        return self._brake(xy,yaw)

    def tick(self, xy, yaw, speed, now, upright=1., velocity=None):
        started = time.perf_counter()
        self.last_xy, self.last_yaw, self.actual_speed, self.upright = tuple(xy), yaw, speed, upright
        self.velocity_world = np.asarray(velocity[:2] if velocity is not None else
                                        (speed*math.cos(yaw), speed*math.sin(yaw)))
        if self.last_tick is not None and self.previous_command.vx > 0:
            forward = float(np.dot(self.velocity_world, [math.cos(yaw), math.sin(yaw)]))
            self.backslide_distance += max(0., -forward)*max(0., now-self.last_tick)
        self.last_tick = now
        command = self._step(tuple(xy), yaw, speed, now, upright)
        # Final command authority applies to every state, including recovery,
        # braking, and operator hold. Stage 1 never raises speed on any terrain.
        command = Motion(float(np.clip(command.vx, -self.command_limit, self.command_limit)),
                         float(np.clip(command.omega, -.35, .35)))
        self.previous_command = command
        self.tick_ms = (time.perf_counter()-started)*1000
        return command

    def _step(self, xy, yaw, speed, now, upright):
        if upright < .3:
            self.finish_attempt('failure', 'tip', xy, yaw, now,physical=True)
            self.set_state('HOLD', 'robot tipped; assistance required', now)
            return Motion()
        if self.state == 'MANUAL':
            return self.manual if self.manual != Motion() else self.position_hold(xy, yaw)
        if self.state in ('HOLD', 'COMPLETE'):
            return self.position_hold(xy, yaw) if self.state == 'HOLD' else Motion()
        if self.regroup_required and self.regroup_started is not None and now-self.regroup_started>90.:
            self.finish_attempt('failure','committed regroup timed out',xy,yaw,now)
            self.set_state('HOLD','committed regroup timed out; assistance required',now)
            return self._brake(xy,yaw)
        if upright < .65:
            self.reason = 'braking excessive body tilt'
            if upright < .5:
                self.finish_attempt('failure','excessive body tilt',xy,yaw,now,physical=True)
                self.set_state('HOLD','excessive body tilt; assistance required',now)
            return self._brake(xy,yaw)
        if self.state == 'OBSERVE':
            if now-self.since < 3. or self.grid.revision-self.observe_revision < 3 or speed > .12:
                return self.position_hold(xy, yaw)
            if self._plan(xy, yaw, now):
                return Motion()
            if self.plan_pending:
                return self.position_hold(xy,yaw)
            if self.mode == 'qualified' and self.route_options:
                self.set_state('HOLD', self.planning_failure, now)
                return Motion()
            return self.recover(xy, yaw, now, self.planning_failure)
        if self.state == 'RECOVER':
            return self._recover_tick(xy, yaw, speed, now)
        if self.state == 'SETTLE':
            self.settle_samples.append((now, xy))
            while len(self.settle_samples) > 1 and self.settle_samples[1][0] <= now-.5:
                self.settle_samples.pop(0)
            elapsed = now-self.settle_samples[0][0]
            net = math.dist(xy, self.settle_samples[0][1])/max(elapsed, 1e-6)
            inside = self.checkpoint is not None and math.dist(xy, self.checkpoint) <= .6
            if elapsed >= .5-1e-6 and net < .08 and speed < .15 and inside:
                self.finish_attempt('success', 'checkpoint reached and settled', xy, yaw, now)
                self.visited_checkpoints.append(tuple(self.checkpoint))
                if self.regroup_required and self.regroup_goal is not None and math.dist(xy,self.regroup_goal)<.6:
                    self.regroup_required=False;self.regroup_goal=None;self.regroup_origin=None
                    self.regroup_started=None;self.retreat_goal=None;self.route_purpose='mission'
                    self.recovery_anchor_v2=None
                if self.is_mission_checkpoint:
                    self.wp_idx += 1
                    if self.wp_idx >= len(self.waypoints):
                        self.set_state('COMPLETE', 'exterior destination reached and settled', now)
                        return Motion()
                self.observe_revision = self.grid.revision
                self.set_state('OBSERVE', 'checkpoint confirmed; refresh observations', now)
                self.path = []
                return self.position_hold(xy, yaw)
            if now-self.since > 8. or not inside:
                return self.recover(xy, yaw, now, 'could not settle on checkpoint', physical=True)
            return self.position_hold(xy, yaw)
        if self.state != 'FOLLOW' or len(self.path) < 2:
            return self.recover(xy, yaw, now, 'route unavailable')
        self.path_index, foot, target, remaining, self.tracking_error = path_progress(self.path, xy, self.path_index, lookahead=self.tracking_lookahead)
        if remaining < self.best_remaining-.12:
            self.best_remaining, self.progress_at = remaining, now
        self.target = tuple(target)
        if self.tracking_error > .75:
            return self.recover(xy, yaw, now, 'physical route tracking failure', physical=True)
        if self.attempt and now-self.attempt['time'] > self.attempt['budget']:
            return self.recover(xy, yaw, now, 'approach timed out', physical=True)
        if remaining < .4 and math.dist(xy, self.checkpoint) < .6:
            self.settle_samples = []
            self.set_state('SETTLE', 'braking and settling at checkpoint', now)
            return self.position_hold(xy, yaw)
        self.heading_error = wrap_angle(math.atan2(target[1]-xy[1], target[0]-xy[0])-yaw)
        reverse=False
        if self.route_purpose=='retreat' and abs(self.heading_error)>math.pi/2:
            cell=self.grid.world_to_cell(*xy)
            # Back out while steep. At lower ground, align for forward descent.
            if self.grid.slope[cell]>math.tan(math.radians(20)) or math.dist(xy,self.regroup_origin)<.6:
                reverse=True
                self.heading_error=wrap_angle(self.heading_error+math.pi)
        # Evaluate progress along commanded travel, including reverse. The
        # baseline tracker intentionally ignores retreat, so transform its
        # frame here instead of silently losing reverse stall evidence.
        if speed > .45:
            self.reason = 'braking measured overspeed'
            self.slip_tracker.reset();self.stall_since=None
            return self._brake(xy, yaw)
        backwards=reverse and self.previous_command.vx<-.1
        travel_yaw=wrap_angle(yaw+math.pi) if backwards else yaw
        demand=Motion(abs(self.previous_command.vx),self.previous_command.omega)
        monitoring=self.previous_command.vx*(-1 if reverse else 1)>.1
        evidence = self.slip_tracker.update(now, xy, travel_yaw, self.velocity_world, demand,
                                             'FOLLOW' if monitoring else 'braking', self.heading_error, self.tracking_error)
        if evidence and evidence['kind']=='poor drive progress':
            if self.stall_since is None:self.stall_since=now
            if now-self.stall_since<3.:evidence=None
        else:self.stall_since=None
        if evidence:
            return self.recover(xy, yaw, now, ('reverse ' if backwards else '')+evidence['kind'], physical=True)
        error = abs(self.heading_error)
        if error < self.alignment_best-.08:
            self.alignment_best, self.alignment_progress_at = error, now
            # Rotation is measured maneuver progress. It must not be mislabeled
            # a stalled climb just because route arc length is unchanged.
            if abs(self.previous_command.vx) < .1:
                self.progress_at = now
        omega = float(np.clip(.8*self.heading_error, -.35, .35)) if error > .035 else 0.
        # A turn must not cut across the grade/obstacle checked route.
        cell=self.grid.world_to_cell(*xy)
        alignment_limit=.30 if self.grid.slope[cell]>math.tan(math.radians(15)) else .85
        vx = self.command_limit*max(.25, math.cos(error)) if error < alignment_limit else 0.
        if remaining < .8:
            # The normal command is needed to maintain uphill traction until
            # the actual arrival boundary; do not impose a creeping climb.
            cell=self.grid.world_to_cell(*xy)
            uphill=self.grid.gx[cell]*math.cos(yaw)+self.grid.gy[cell]*math.sin(yaw)>.15
            if not uphill and not reverse:vx = min(vx, .25)
        if reverse:vx=-vx
        motion = Motion(vx, omega)
        clear=self.motion_clear(xy,yaw,speed,motion)
        chord_grade=self.grid.grade_reason(xy,target,uphill_limit=8. if self.route_purpose=='retreat' else None)
        if not clear or chord_grade:
            # A long lookahead can cut across an otherwise valid contour or
            # corner. Try the local route tangent before abandoning the leg.
            for lookahead in (.35,.15):
                if lookahead>=self.tracking_lookahead:continue
                _,_,near,_,_=path_progress(self.path,xy,self.path_index,lookahead=lookahead)
                if self.grid.grade_reason(xy,near,uphill_limit=8. if self.route_purpose=='retreat' else None):continue
                angle=wrap_angle(math.atan2(near[1]-xy[1],near[0]-xy[0])-yaw+(math.pi if reverse else 0.))
                w=float(np.clip(.8*angle,-.35,.35))
                v=self.command_limit*math.cos(angle) if abs(angle)<alignment_limit else 0.
                if remaining<.8 and not uphill and not reverse:v=min(v,.25)
                alternative=Motion(-v if reverse else v,w)
                if self.motion_clear(xy,yaw,speed,alternative):
                    if now-self.progress_at>8. and now-self.alignment_progress_at>8.:
                        return self.recover(xy,yaw,now,'local tangent made no measured progress')
                    self.tracking_lookahead=lookahead
                    self.target=tuple(near);self.heading_error=angle
                    self.reason='tracking the checked local route tangent'
                    return alternative
            # Short alignment before abandoning a route; rejection is censored
            # and cannot manufacture a terrain failure or no-go zone.
            turn = Motion(0., omega)
            if error > .08 and self.motion_clear(xy, yaw, speed, turn) and now-self.alignment_progress_at < 8.:
                self.reason = 'aligning with checked route'
                return turn
            slow = Motion(float(np.clip(vx,-.2,.2)), omega*.75)
            if vx != 0 and self.motion_clear(xy, yaw, speed, slow):
                self.reason = 'using body clearance at reduced speed'
                return slow
            return self.recover(xy, yaw, now, 'command rejected: '+self.motion_rejection)
        if now-self.progress_at > 8.:
            driven=(monitoring and error<.3 and abs(self.previous_command.vx)>=.2
                    and self.slip_tracker.metrics.get('window_s',0.)>=2.)
            return self.recover(xy, yaw, now,
                                'sustained drive without route progress' if driven else 'no measured route progress',
                                physical=driven)
        self.reason = 'following '+self.route_decision
        return motion

    def navigation_status(self):
        return asdict(NavigationStatus(self.state, self.reason, self.command_limit, self.reachability, self.mode))
