"""Ascent decisions through a behavior tree; actuation through one executor.

AscentController supplies planning, map geometry and evidence serialization.
Its former _step/recover/_recover_tick/position_hold execution is never used.
"""
from dataclasses import asdict, replace
import math
import time

import numpy as np

from .ascent_control import AscentController
from .behavior_tree import Action, ReactiveSelector, Status
from .checkpoint_control import Motion, wrap_angle
from .movement import MovementExecutor, MovementRequest, MotionObservation, MovementState, ResultDisposition
from .reachability import eligibility, route_features
from .staged_routes import compact_path, motion_route_reason, observed_edge_reason, detour_shortcuts
from .swept_footprint import PassiveHeadingWindow, oriented_sweep_reason, oriented_stopping_reason
from .ascent_routes import observed_route
from .route_exploration import survey_portals, first_climb, same_corridor, same_entrance, launch_description
from .command_selection import select_forward_command
from .recovery_continuation import select_region, region_reason, approach_reason
from .swept_footprint import arrival_ray_interval
from .recovery_intent import RecoveryPermission
from .recovery_turn import RecoveryTurn, rotation_reason, exit_reason


class BehaviorTreeAscentController(AscentController):
    protocol_version = 'staged-teleop-v29-paired-launch+purposeful-recovery-v1+bounded-recovery-turn-v1'

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('uphill_limit', 33.)
        kwargs.setdefault('side_limit', 25.)
        super().__init__(*args, **kwargs)
        if self.mode == 'qualified' and self.model.artifact is not None:
            profile = self.model.artifact.get('capability_profile') or {}
            if profile.get('execution_protocol') != self.protocol_version:
                raise ValueError('Reachability calibration must match the staged teleop execution protocol')
        self.grid.motion_turn_limit = 12.
        self.grid.planning_grade_reserve = 0.
        self.grid.diagonal_ascent_preference = True
        self.minimum_leg_distance = 0.
        self.exploration_coverage = None
        self.paired_launch_variant = False
        self.launch_variant_sources = {}
        self.executor = MovementExecutor()
        self.active_zero_envelope_enabled = False
        self.active_zero_admissions = set()
        self.active_zero_departures = {}
        self.survey_ascent_waypoint = None
        self.executor.brake_guard = self._brake_envelope
        self.executor.brake_evaluator = self._brake_evaluation
        self.executor.corridor_guard = self._remaining_corridor
        self.executor.straight_entry_guard = self._straight_entry_envelope
        self.executor.release_guard = self._release_envelope
        self.executor.posture_guard = self._release_envelope
        self.executor.recovery_stage_guard = self._recovery_staging_envelope
        self.executor.recovery_turn_guard = self._recovery_turn_guard
        self.executor.drive_selector = self._select_forward_command
        self.operator_mode = 'auto'
        self.fault = ''
        self.tree_branch = 'observe'
        self.command = Motion()
        self.obs = None
        self.turn_enabled = False
        self.gentle_turn_entry_enabled = False
        self.gentle_turn_since = None
        self.segments = []
        self.segment_index = 0
        self.action_number = 0
        self.recovering = False
        self.recovery_since = None
        self.recovery_trials = set()
        self.recovery_direction = None
        self.recovery_actions = 0
        self.recovery_turn_used = False
        self.recovery_turn_failure_id = None
        self.recovery_request_id = None
        self.recovery_progress = 0.
        self.recovery_permission = None
        self.recovery_route = []
        self.recovery_origin = None
        self.supported_trail = []
        self.recovery_continuation_enabled = False
        self.recovery_continuation_used = False
        self.invalid_observation_hold = False
        self.prediction_committed = False
        self.guard_evidence_seen = set()
        self.oriented_admission_seen = set()
        self.passive_heading = PassiveHeadingWindow()
        self.exported_transitions = 0
        self.last_plan_at = -math.inf
        self.tree = ReactiveSelector(
            Action('operator_stop', self._operator_stop),
            Action('hard_fault', self._hard_fault),
            Action('manual_control', self._manual_control),
            Action('terminal', self._terminal),
            Action('active_primitive', self._active),
            Action('consume_primitive_result', self._result),
            Action('straight_recovery', self._recovery),
            Action('execute_checkpoint_leg', self._leg_action),
            Action('observe_and_plan', self._mission))

    def request(self, action, now, vx=0., omega=0.):
        if getattr(self, 'gentle_turn_entry_enabled', False):
            self.gentle_turn_since = None
            self.turn_enabled = False
        if action=='resume':self.invalid_observation_hold=False
        if hasattr(self, 'executor'):
            if self.obs is not None:
                self.executor.stop(self.obs, 'operator '+action)
                self.executor.consume()
            self.segments = []
            self.recovering = False
            self.recovery_permission = None
            self.prediction_committed = False
            self.operator_mode = 'manual' if action == 'manual' else 'stop' if action == 'stop' else 'auto'
        super().request(action, now, vx, omega)

    def choose_checkpoint(self, xy, target, now, yaw=0.):
        accepted = super().choose_checkpoint(xy, target, now, yaw)
        if accepted and not self.last_selection.get('duplicate'):
            self.request('resume', now)
        return accepted

    def position_hold(self, xy, yaw):
        return self.executor.brake(self.obs) if self.obs is not None else Motion()

    def route_score(self, route, yaw):
        p = self.grid.route_profile(route)
        return (self.grid.route_cost(route, yaw)+.5*max(0., p['max_uphill_deg']-28.)**2
                +3.*max(0., p['max_side_slope_deg']-15.)**2)

    def configure_exploration(self, coverage):
        if self.mode != 'characterize' or self.attempt_number:
            raise ValueError('Exploration requires a fresh characterization mission')
        if not (coverage.get('records') or coverage.get('entrance_attempts')) or not self.grid.terrain_prior:
            raise ValueError('Exploration needs comparable executed history and a labeled survey')
        self.exploration_coverage = coverage

    def _exploring_initial_route(self):
        return self.exploration_coverage is not None and self.attempt_number == 0

    def configure_launch_variant(self):
        if self.mode != 'characterize' or not self._exploring_initial_route():
            raise ValueError('Launch comparison requires initial full-survey characterization with reviewed scheduling history')
        self.paired_launch_variant = True

    def configure_gentle_turn_entry(self):
        if self.mode != 'characterize' or self.attempt_number:
            raise ValueError('Gentle-turn entry requires fresh characterization')
        self.gentle_turn_entry_enabled = True
        self.grid.motion_turn_limit = 15.
        self.protocol_version += '+gentle-turn-entry-v1'

    def configure_active_zero_envelope(self):
        if self.mode != 'characterize' or self.attempt_number:
            raise ValueError('Active-zero envelope requires fresh characterization')
        self.active_zero_envelope_enabled = True
        self.executor.release_departure_guard = self._active_zero_departure
        self.protocol_version += '+active-zero-envelope-v3-selected-actuation'

    def configure_outbound_survey_routes(self, waypoint_index=1):
        if (self.mode != 'characterize' or self.attempt_number
                or not 0 <= waypoint_index < len(self.waypoints)):
            raise ValueError('Outbound survey routes require a fresh characterization mission stage')
        self.survey_ascent_waypoint = waypoint_index
        self.protocol_version += '+outbound-survey-routes-v1'

    def _update_turn_permission(self, xy, fresh, inclination, tilt, speed, yaw_rate, now, dt):
        if self.gentle_turn_entry_enabled and (dt <= 0. or dt > .041):
            self.turn_enabled = False
            self.gentle_turn_since = None
        if not fresh or not math.isfinite(inclination) or inclination > 15. or tilt > 25.:
            self.turn_enabled = False
            self.gentle_turn_since = None
            return
        stable = tilt <= 20. and speed < .12 and abs(yaw_rate) < .12
        if self.gentle_turn_entry_enabled and self.operator_mode != 'auto':
            self.turn_enabled = False
            self.gentle_turn_since = None
            return
        if inclination <= 12. and stable:
            self.turn_enabled = True
            self.gentle_turn_since = None
        elif self.gentle_turn_entry_enabled:
            if not stable or self.grid.turn_support_reason(xy) or not self.grid.segment_clear(xy, xy):
                self.gentle_turn_since = None
            else:
                if self.gentle_turn_since is None:
                    self.gentle_turn_since = now
                if now-self.gentle_turn_since >= .6-1e-9:
                    self.turn_enabled = True

    def configure_uphill_speed_response(self):
        if self.mode != 'characterize' or self.attempt_number:
            raise ValueError('Uphill speed response requires a fresh characterization mission')
        self.executor.uphill_speed_response_enabled = True
        self.protocol_version = ('staged-teleop-v33-correction-release+purposeful-recovery-v1+bounded-recovery-turn-v1' if self.executor.release_memory.correction_release_enabled else
                                 'staged-teleop-v32-release-response-memory+purposeful-recovery-v1+bounded-recovery-turn-v1' if self.executor.release_memory.enabled else
                                 'staged-teleop-v31-functional-recovery+purposeful-recovery-v1+bounded-recovery-turn-v1' if self.recovery_continuation_enabled
                                 else 'staged-teleop-v30-uphill-speed-response+purposeful-recovery-v1+bounded-recovery-turn-v1') + '+protective-drive-resume-v1+owned-ascent-v1'

    def configure_release_memory(self):
        if (self.mode!='characterize' or self.attempt_number or not self.executor.uphill_speed_response_enabled
                or self.recovery_continuation_enabled):
            raise ValueError('Release memory requires fresh uphill-speed characterization without recovery continuation')
        self.executor.release_memory.enabled=True
        self.protocol_version=('staged-teleop-v33-correction-release+purposeful-recovery-v1+bounded-recovery-turn-v1' if self.executor.release_memory.correction_release_enabled
                               else 'staged-teleop-v32-release-response-memory+purposeful-recovery-v1+bounded-recovery-turn-v1') + '+protective-drive-resume-v1+owned-ascent-v1'

    def configure_correction_release(self):
        if self.mode!='characterize' or self.attempt_number or not self.executor.release_memory.enabled:
            raise ValueError('Correction release requires fresh release-memory characterization')
        self.executor.release_memory.correction_release_enabled=True
        self.protocol_version='staged-teleop-v33-correction-release+purposeful-recovery-v1+bounded-recovery-turn-v1+protective-drive-resume-v1+owned-ascent-v1'

    def configure_recovery_continuation(self):
        if (self.mode!='characterize' or self.attempt_number or not self.executor.uphill_speed_response_enabled
                or self.executor.release_memory.enabled):
            raise ValueError('Recovery continuation requires fresh characterization with uphill speed response')
        self.recovery_continuation_enabled=True
        self.executor.recovery_continuation_provider=self._recovery_region_candidate
        self.executor.recovery_continuation_guard=self._recovery_region_guard
        self.executor.recovery_continuation_admitted=self._adopt_recovery_region
        self.protocol_version='staged-teleop-v31-functional-recovery+purposeful-recovery-v1+bounded-recovery-turn-v1+protective-drive-resume-v1+owned-ascent-v1'

    def _recovery_region_candidate(self, obs, request, samples):
        if (not self.recovering or self.recovery_continuation_used or self.recovery_actions>=8
                or self.recovery_since is None or obs.time-self.recovery_since>120.):return None
        forecast=self.passive_heading.offsets(obs.time,obs.yaw,
            max(obs.speed,request.speed)/max(.1,self.braking_accel or .2))
        candidate,reason=select_region(self.grid,obs,request,samples,forecast)
        self.escape_events.append(dict(event='recovery_region_candidate',time=obs.time,
            action_id=request.action_id,accepted=candidate is not None,reason=reason,
            certificate=asdict(candidate) if candidate else None,probability_validated=False))
        return candidate

    def _recovery_region_guard(self, obs, certificate, staging):
        if not certificate.contains_approach(obs.xy):return 'left frozen recovery approach corridor'
        if obs.time>certificate.deadline:return 'original recovery deadline elapsed'
        reason=approach_reason(self.grid,certificate)
        if reason:return 'recovery certificate geometry changed: '+reason
        distance=math.dist(obs.xy,certificate.center)
        if staging or distance<=certificate.inner_radius:
            reason=region_reason(self.grid,certificate,fresh=True)
            if reason:return reason
            if distance>certificate.outer_radius:return 'left frozen recovery staging region'
        elif arrival_ray_interval(obs.xy,certificate.center,obs.yaw,-1,certificate.inner_radius) is None:
            return 'current reverse ray misses frozen recovery entry core'
        return self._guard(Motion(-self.executor.request.speed,0.),record=False)

    def _adopt_recovery_region(self, obs, certificate, execution):
        # Compute the new unvalidated estimate before changing evidence ownership.
        path=[obs.xy,certificate.center]
        features=route_features(self.grid,path,obs.yaw,obs.speed,obs.up_z,-self.command_limit)
        p0,reason=self.model.predict(features)
        heading=math.atan2(certificate.axis[1],certificate.axis[0])
        key=(f'recovery-region-v31:outer={certificate.outer_radius}:inner={certificate.inner_radius}:'
             +self.evidence.key(obs.xy,certificate.center,heading,-self.command_limit,path=path))
        # A point-leg calibration cannot validate reaching/settling in a region.
        # Keep its raw model output only as labeled diagnostic information.
        estimate=self.evidence.estimate(key,None,'functional recovery region is unvalidated: '+reason)
        previous=self.attempt['id'] if self.attempt else None
        if self.attempt is not None:
            self.attempt['recovery_phase']=dict(action_id=certificate.action_id,
                phase_id=certificate.action_id+':nominal',certificate_id=certificate.certificate_id,
                target_kind='nominal_recovery_endpoint')
        self.finish_attempt('censored','nominal recovery target replaced by checked fixed staging region',
            obs.xy,obs.yaw,obs.time,execution=asdict(execution),terminal_maneuver=execution.maneuver)
        self.attempt_number+=1
        phase=dict(action_id=certificate.action_id,phase_id=certificate.certificate_id,
            certificate_id=certificate.certificate_id,previous_attempt_id=previous,
            target_kind='recovery_staging_region',outer_radius=certificate.outer_radius,
            inner_radius=certificate.inner_radius,probability_validated=False,
            point_model_diagnostic=p0)
        self.attempt=dict(id=f'{self.run_id}:{self.attempt_number}',time=obs.time,start=obs.xy,
            endpoint=certificate.center,heading=heading,features=features,estimate=estimate,key=key,
            path=path,recovery_phase=phase)
        self.reachability=dict(asdict(estimate),validated=False,target_kind='recovery_staging_region')
        self.path=path;self.checkpoint=certificate.center
        self.attempt_events.append(dict(event='approach_start',attempt_id=self.attempt['id'],time=obs.time,
            run_id=self.run_id,features=features,prediction=self.reachability,endpoint=list(certificate.center),
            approach_key=key,mode=self.mode,protocol=self.protocol_version,**phase))
        self.prediction_committed=True
        self.recovery_continuation_used=True

    def finish_attempt(self, *args, **kwargs):
        phase=self.attempt.get('recovery_phase') if self.attempt else None
        before=len(self.attempt_events)
        super().finish_attempt(*args,**kwargs)
        if phase and len(self.attempt_events)>before:self.attempt_events[-1].update(phase)

    def _launch_suffix_variants(self, routes, xy, yaw):
        """Keep map-derived climb prefixes; remove unnecessary portal suffixes."""
        goal = (self.waypoints[self.wp_idx].x,self.waypoints[self.wp_idx].y)
        support = self.grid.planning_support()
        variants, seen = [], set()
        for label, route, complete in routes:
            if not complete:
                continue
            launch = launch_description(self.grid,route)
            if launch is None or not launch['climb_prefix']:
                continue
            prefix = launch['climb_prefix']
            signature = tuple(prefix)
            if signature in seen or math.dist(prefix[-1],goal) < .1:
                continue
            seen.add(signature)
            heading = math.atan2(prefix[-1][1]-prefix[-2][1],prefix[-1][0]-prefix[-2][0])
            onward, done = observed_route(self.grid,prefix[-1],goal,heading,support=support,
                allow_frontier=False,require_fresh_start=False,uphill_limit=self.grid.uphill_limit_deg)
            name = label+'-launch-variant'
            path = prefix+onward[1:] if len(onward)>1 else prefix
            self.route_searches.append(dict(profile=name,source_profile=label,connected=done,
                has_route=len(onward)>1,uphill_limit=self.grid.uphill_limit_deg,route_path=path,
                retained_prefix=prefix,**self.grid.last_route_search))
            if done and len(onward)>1 and math.dist(onward[-1],goal)<.1:
                self.launch_variant_sources[name] = label
                variants.append((name,path,True))
        return variants

    def _survey_alternatives(self, xy, yaw):
        goal = (self.waypoints[self.wp_idx].x, self.waypoints[self.wp_idx].y)
        support = self.grid.planning_support()
        alternatives = []
        for candidate in survey_portals(self.grid, xy, goal, support):
            name = f'survey-sector-{candidate["sector"]}'
            portal = candidate['portal']
            if portal is None:
                self.route_searches.append(dict(profile=name, connected=False, has_route=False,
                    uphill_limit=self.grid.uphill_limit_deg, rejection='no sampled supported gentle portal', **candidate))
                continue
            path = []
            for label, origin, destination in [('approach',xy,portal), ('onward',portal,goal)]:
                heading = yaw if not path else math.atan2(path[-1][1]-path[-2][1],path[-1][0]-path[-2][0])
                found, complete = observed_route(self.grid, origin, destination, heading, support=support,
                    allow_frontier=False, require_fresh_start=label=='approach', uphill_limit=self.grid.uphill_limit_deg)
                self.route_searches.append(dict(profile=name+'-'+label, connected=complete, has_route=len(found)>1,
                    uphill_limit=self.grid.uphill_limit_deg, route_path=found,
                    **candidate, **self.grid.last_route_search))
                if not complete or len(found)<2:
                    exhausted = self.grid.last_route_search.get('budget_exhausted',False)
                    self.route_options.append(dict(profile=name, selected=False, complete=False,
                        endpoint=list(portal), path=found, route_path=found, score=None,
                        rejection=label+(': search budget exhausted' if exhausted else ': no route found by heading search'),
                        **self.grid.route_profile(found)))
                    break
                path = found if not path else compact_path(path+found[1:])
            else:
                alternatives.append((name,path,True))
        return alternatives

    def _select_route_candidate(self, candidates, xy, yaw):
        if not self._exploring_initial_route():
            return super()._select_route_candidate(candidates,xy,yaw)
        eligible = []
        for candidate in candidates:
            _, score, route, complete, *unused, info = candidate
            corridor = first_climb(self.grid, route)
            reason = '' if complete else 'complete exact-goal route required'
            if not reason:
                reason = motion_route_reason(self.grid,route,yaw)
            if not reason:
                reason = next((r for a,b in zip(route,route[1:])
                               if (r := observed_edge_reason(self.grid,a,b,xy))), '')
            if not reason and not self._retreat_exists(route):
                reason = 'full route lacks checked geometric retreat'
            if not reason and not same_corridor(corridor, first_climb(self.grid,candidate[4])):
                reason = 'selected checkpoint leg does not include sustained climb'
            covered = corridor is not None and any(same_corridor(corridor,r['corridor']) for r in self.exploration_coverage['records'])
            entrance_tested = corridor is not None and any(same_entrance(corridor,r['corridor'])
                for r in self.exploration_coverage.get('entrance_attempts',[]))
            if not reason and corridor is None:
                reason = 'no sustained climbing corridor identified'
            if not reason and covered:
                reason = 'climb corridor already executed in characterization history'
            if not reason and entrance_tested:
                reason = 'entrance already attempted and geometry-censored; sustained climb remains uncharacterized'
            info['exploration'] = dict(corridor=corridor, covered=covered, entrance_attempted=entrance_tested, eligible=not reason,
                                      reason=reason or 'untested climbing corridor; original cost ranking',
                                      stage='candidate', coverage_only=True,
                                      history_scopes=sorted({r.get('execution_scope','matching_executor_scheduling_witness')
                                          for r in self.exploration_coverage['records']+self.exploration_coverage.get('entrance_attempts',[])}),
                                      coverage_claim='Experiment scheduling only; legacy records do not establish current-executor capability.')
            if not reason:
                eligible.append(candidate)
        if not eligible:
            self.planning_failure = ('paired launch diagnostic: no eligible complete reference or variant'
                if self.paired_launch_variant else 'no eligible untested corridor for characterization experiment')
            return None
        if self.paired_launch_variant:
            return self._select_launch_variant(eligible)
        selected = min(eligible, key=lambda c:c[1])
        selected[-1]['exploration']['stage'] = 'selected; adoption and actual climbing remain unverified'
        return selected

    def _select_launch_variant(self, eligible):
        originals = [c for c in eligible if c[-1]['profile'] not in self.launch_variant_sources]
        if not originals:
            self.planning_failure = 'paired launch diagnostic: no original eligible reference route'
            return None
        reference = min(originals,key=lambda c:c[1])
        baseline = launch_description(self.grid,reference[2])
        alternatives = []
        for candidate in eligible:
            info = candidate[-1]
            launch = launch_description(self.grid,candidate[2])
            reason = ''
            if not baseline or not launch or not baseline['climb_prefix'] or not launch['climb_prefix']:
                reason = 'launch or post-climb staging support unavailable'
            elif not same_corridor(baseline['corridor'],launch['corridor']):
                reason = 'different climb class from reference'
            elif (not np.isfinite([baseline['launch_inclination_deg'],launch['launch_inclination_deg']]).all()
                  or baseline['launch_support_reason'] or launch['launch_support_reason']
                  or not baseline['launch_geometry_clear'] or not launch['launch_geometry_clear']):
                reason = 'launch footprint support unavailable'
            elif launch['pre_climb_bends'] >= baseline['pre_climb_bends']:
                reason = 'launch does not reduce pre-climb bends'
            elif launch['launch_inclination_deg'] >= baseline['launch_inclination_deg']:
                reason = 'launch footprint is not flatter than reference'
            info['paired_launch_variant'] = dict(experiment='paired_launch_variant',comparison='same_climb_class',
                reference_profile=reference[-1]['profile'],reference_score=reference[1],
                reference_route=reference[2],reference=baseline,variant=launch,
                source_profile=self.launch_variant_sources.get(info['profile'],info['profile']),
                eligible=not reason,reason=reason or 'fewer bends and flatter launch; unchanged route cost ranking',
                evidence_claim='launch comparison only; no new climb class, confidence or mission evidence')
            if not reason:
                alternatives.append(candidate)
        if not alternatives:
            self.planning_failure = 'paired launch diagnostic: no same-class variant improves both bends and launch inclination'
            return None
        selected = min(alternatives,key=lambda c:c[1])
        selected[-1]['exploration'].update(stage='paired launch selected; adoption and traversal unverified',
            reason='same climb class as original reference; paired launch characterization',
            coverage_claim='Same-class launch comparison only; history schedules experiments, not current-executor capability.')
        return selected

    def _augment_routes(self, routes, xy, yaw):
        self.launch_variant_sources = {}
        augmented = list(routes)
        for label, route, complete in routes:
            if not label.startswith('shelf-'):
                continue
            for proposal in detour_shortcuts(self.grid, route, yaw):
                name = f'{label}-shortcut-{proposal["join_index"]}'
                path, reason = proposal['path'], proposal['rejection']
                self.route_searches.append(dict(profile=name, source_profile=label,
                    shortcut_saved_m=proposal['saved_m'], route_path=path,
                    connected=complete, has_route=not bool(reason), rejection=reason,
                    uphill_limit=self.grid.uphill_limit_deg))
                if reason:
                    self.route_options.append(dict(profile=name, selected=False, complete=complete,
                        endpoint=list(path[-1]), path=path, route_path=path, score=None, rejection=reason,
                        **self.grid.route_profile(path)))
                else:
                    augmented.append((name,path,complete))
        if self._exploring_initial_route():
            augmented.extend(self._survey_alternatives(xy,yaw))
            if self.paired_launch_variant:
                augmented.extend(self._launch_suffix_variants(augmented,xy,yaw))
        elif self.wp_idx == self.survey_ascent_waypoint and self.grid.terrain_prior:
            # This hook belongs to ordinary mission planning; regroup returns
            # before it. route_purpose still describes the previous maneuver.
            # Existing bounded searches, on the existing asynchronous planning
            # snapshot. Portal destinations alone can retain needless rim
            # detours; reconnect after the climb to the actual mission goal.
            started = time.perf_counter()
            alternatives = self._survey_alternatives(xy,yaw)
            variants = self._launch_suffix_variants(alternatives,xy,yaw)
            seen = {self._route_signature(path,path) for _,path,_ in augmented}
            added = 0
            for proposal in alternatives+variants:
                signature = self._route_signature(proposal[1],proposal[1])
                if signature not in seen:
                    seen.add(signature)
                    augmented.append(proposal)
                    added += 1
            self.route_searches.append(dict(profile='outbound-survey-batch',
                connected=bool(added),has_route=bool(added),uphill_limit=self.grid.uphill_limit_deg,
                portal_routes=len(alternatives),suffix_variants=len(variants),added_routes=added,
                elapsed_ms=1000*(time.perf_counter()-started)))
        return augmented

    def _route_rejection_reason(self, route, xy, yaw):
        reason = motion_route_reason(self.grid, route, yaw)
        if reason:
            return reason
        for index, (a,b) in enumerate(zip(route,route[1:])):
            reason = observed_edge_reason(self.grid,a,b,xy)
            if reason:
                return f'route segment {index}: {reason}'
        return 'route cost rejects execution geometry'

    def _route_signature(self, route, leg):
        # Two candidates may share the checkpoint prefix but have different
        # onward geometry. A rejected suffix must not hide a usable alternative.
        return tuple(tuple(p) for p in route)

    def _adoption_edge_reason(self, a, b, xy, retreat):
        return observed_edge_reason(self.grid, a, b, xy, no_go_exit=retreat)

    def _leg(self, route):
        leg = super()._leg(route)
        if math.dist(leg[-1], route[-1]) < .65:
            return route
        return leg if self.grid.max_support_slope(leg[-1]) <= 12. else route

    def tick(self, xy, yaw, speed, now, upright=1., velocity=None, body_quaternion=None):
        started = time.perf_counter()
        if self.recovery_continuation_enabled or self.gentle_turn_entry_enabled:
            values=(*xy,yaw,speed,now,upright,
                *(velocity if velocity is not None else ()),
                *(body_quaternion if body_quaternion is not None else ()))
            invalid=not all(math.isfinite(v) for v in values)
            if invalid or self.invalid_observation_hold:
                self.gentle_turn_since = None
                if self.gentle_turn_entry_enabled: self.turn_enabled = False
                self.recovery_permission = None
                # This boundary precedes map lookups, finite differences and
                # damping. A censored executor result must not leak a NaN into
                # next-tick recovery/braking. Operator STOP/manual retain priority.
                when=now if math.isfinite(now) else self.obs.time if self.obs else 0.
                if math.isfinite(upright) and upright<.5 and not self.fault:
                    self.fault='excessive body tilt'
                    self.escape_events.append(dict(event='tip_during_invalid_observation',time=when,
                        up_z=upright,reason='finite tip observation; unavailable pose cannot localize a terrain failure'))
                if self.operator_mode=='manual' and not self.fault:
                    self.command=Motion(float(np.clip(self.manual.vx,-self.command_limit,self.command_limit)),
                                        float(np.clip(self.manual.omega,-.35,.35)))
                    self.set_state('MANUAL','operator teleop; navigation observation unavailable',when)
                else:
                    self.command=Motion()
                    if self.operator_mode!='stop' and not self.invalid_observation_hold:
                        reason=(self.fault+'; navigation observation unavailable' if self.fault
                                else 'invalid navigation observation; explicit resume required')
                        self.executor.pending_traversal=None
                        if self.obs is not None:
                            if self.executor.result is not None:
                                # A completed result belongs to the last valid
                                # tick: keep its established physical evidence.
                                self._result(None)
                            else:
                                self.executor.finish('censored',reason,self.obs)
                                result=self.executor.consume()
                                metadata=dict(last_valid_time=self.obs.time,last_valid_position=self.obs.xy,
                                    current_observation_valid=False,detection_time=when)
                                if result is not None:
                                    self.attempt_events.append(dict(event='movement_outcome',**asdict(result),
                                        protocol=self.protocol_version,**metadata))
                                count=len(self.attempt_events)
                                self.finish_attempt('censored',reason,self.obs.xy,self.obs.yaw,self.obs.time)
                                if len(self.attempt_events)>count:self.attempt_events[-1].update(metadata)
                        self._invalidate();self.segments=[];self.prediction_committed=False
                        self.invalid_observation_hold=True
                        self.escape_events.append(dict(event='navigation_observation_invalid',time=when,
                            time_valid=math.isfinite(now),last_valid_time=self.obs.time if self.obs else None,
                            reason=reason))
                    self.set_state('HOLD','operator STOP; zero teleop command' if self.operator_mode=='stop'
                                   else self.fault+'; navigation observation unavailable' if self.fault
                                   else 'invalid navigation observation; explicit resume required',when)
                    self.command=Motion()
                self.previous_command=self.command
                self.tick_ms=(time.perf_counter()-started)*1000
                return self.command
        dt = now-self.last_tick if self.last_tick is not None else 0.
        measured_yaw_rate = wrap_angle(yaw-self.last_yaw)/dt if dt > 0 else 0.
        self.velocity_world = np.asarray(velocity[:2] if velocity is not None else
                                        (speed*math.cos(yaw), speed*math.sin(yaw)))
        support = self.grid.max_support_slope(xy)
        tilt = math.degrees(math.acos(float(np.clip(upright, -1., 1.))))
        inclination = support
        cells = self.grid.oriented_cells(xy, yaw)
        fresh = cells is not None and self.grid.known[cells].all() and self.grid.fresh[cells].all()
        self._update_turn_permission(xy,fresh,inclination,tilt,speed,measured_yaw_rate,now,dt)
        gradient = None
        if fresh and np.isfinite(self.grid.gx[cells]).all() and np.isfinite(self.grid.gy[cells]).all():
            gradient = (float(np.median(self.grid.gx[cells])), float(np.median(self.grid.gy[cells])))
        self.obs = MotionObservation(tuple(xy), yaw, tuple(self.velocity_world), upright,
            measured_yaw_rate, now, inclination, self.turn_enabled, bool(fresh), gradient, body_quaternion)
        self.passive_heading.observe(now, yaw)
        if self.recovery_permission is not None:
            updated = self.recovery_permission.observe(self.grid, self.obs.xy)
            if updated != self.recovery_permission:
                self.escape_events.append(dict(event='recovery_patch_departure',time=now,
                    position=list(xy),remaining_patches=len(updated.patches)))
                self.recovery_permission = updated
        if (not self.recovering and self.operator_mode == 'auto' and fresh and self._supported(xy)
                and (not self.supported_trail or math.dist(xy,self.supported_trail[-1]) >= .4)):
            self.supported_trail = (self.supported_trail+[tuple(xy)])[-32:]
        if dt > 0 and self.executor.state in (MovementState.RELEASE, MovementState.POSTURE_PAUSE) and self.executor.travel_axis is not None:
            self.backslide_distance += max(0., -float(np.dot(self.velocity_world, self.executor.travel_axis)))*dt
        elif dt > 0 and abs(self.previous_command.vx) > .1:
            longitudinal = np.dot(self.velocity_world, (math.cos(yaw), math.sin(yaw)))
            self.backslide_distance += max(0., -math.copysign(1., self.previous_command.vx)*longitudinal)*dt
        self.last_xy, self.last_yaw, self.last_tick = tuple(xy), yaw, now
        self.actual_speed, self.upright = speed, upright
        self.command = Motion()
        self.tree.tick(self)
        if self.operator_mode == 'auto':
            # Executed commands are independently constructed and checked as
            # straight drive or stationary turn; no curved command is trimmed.
            if self.command.omega:
                if self.executor.recovery_turn is not None:
                    rejected = (not fresh or tilt >= 50. or speed >= .12
                                or self._guard(self.command))
                else:
                    rejected = inclination > 15. or tilt > 25. or not fresh
                if rejected:
                    self.executor.finish('censored', 'turn envelope became unavailable', self.obs)
                    self.command = self.executor.brake(self.obs)
            if self.command.vx and self.command.omega:
                raise RuntimeError('Autonomy may own only one teleop primitive at a time')
        self.command = Motion(float(np.clip(self.command.vx, -self.command_limit, self.command_limit)),
                              float(np.clip(self.command.omega, -.35, .35)))
        if self.state in ('HOLD','COMPLETE') or self.fault or self.operator_mode != 'auto':
            self.recovery_permission = None
        self.previous_command = self.command
        self.escape_events.extend(self.executor.history[self.exported_transitions:])
        self.exported_transitions = len(self.executor.history)
        self.tick_ms = (time.perf_counter()-started)*1000
        return self.command

    def _operator_stop(self, _):
        if self.operator_mode != 'stop': return Status.FAILURE
        self.recovery_permission = None
        self.command = Motion()
        self.set_state('HOLD', 'operator STOP; zero teleop command', self.obs.time)
        return Status.RUNNING

    def _hard_fault(self, _):
        if self.obs.up_z < .5: self.fault = 'excessive body tilt'
        if not self.fault: return Status.FAILURE
        self.recovery_permission = None
        self.executor.finish('failure', self.fault, self.obs, physical=True)
        result = self.executor.consume()
        if result is not None:
            self.attempt_events.append(dict(event='movement_outcome', **asdict(result), protocol=self.protocol_version))
        self.finish_attempt('failure', self.fault, self.obs.xy, self.obs.yaw, self.obs.time, physical=True,
            execution=asdict(result.executed_traversal) if result and result.executed_traversal else None,
            terminal_maneuver=result.phase if result else 'FAULT')
        self.segments = []
        self.set_state('HOLD', self.fault+'; assistance required', self.obs.time)
        self.command = Motion()
        return Status.RUNNING

    def _manual_control(self, _):
        if self.operator_mode != 'manual': return Status.FAILURE
        self.recovery_permission = None
        self.command = self.manual
        self.set_state('MANUAL', 'operator teleop; autonomy suspended', self.obs.time)
        return Status.RUNNING

    def _terminal(self, _):
        if self.state not in ('HOLD', 'COMPLETE'): return Status.FAILURE
        self.recovery_permission = None
        self.command = self.executor.brake(self.obs) if self.state == 'HOLD' else Motion()
        return Status.RUNNING

    def _supported(self, point):
        cell = self.grid.world_to_cell(*point)
        return cell is not None and self.grid.traversable(cell) and not self.grid.support_reason(point)

    def _select_forward_command(self, command):
        return select_forward_command(command, lambda candidate: self._guard(candidate, record=False))

    def _directional_exit_context(self, command=None, *, damping=False):
        if not self.recovering or self.recovery_permission is None:return False
        if command is not None and command.omega and self.executor.recovery_turn is not None:
            if self._recovery_turn_guard(self.obs, self.executor.request, 'rotation'):return False
            return self.recovery_permission
        request = self.executor.request
        if request is not None and request.purpose != 'recovery':
            return False
        if command is None or command.omega or not command.vx:return False
        if damping:
            longitudinal = sum(v*a for v,a in zip(self.obs.velocity,(math.cos(self.obs.yaw),math.sin(self.obs.yaw))))
            if command.vx*longitudinal >= 0.:return False
        elif request is not None and command.vx*request.direction <= 0.:
            return False
        # Independently check the intended command before forgiving incidental
        # drift. This path deliberately does not call _guard recursively.
        if self._commanded_stopping_reason(self.obs,command,record=False,damping=damping):return False
        return self.recovery_permission

    def _commanded_stopping_reason(self, obs, command, *, record=True, damping=False):
        distance=max(.4,max(obs.speed,abs(command.vx))**2/(2*max(.1,self.braking_accel or .2))+.1)
        end=(obs.xy[0]+math.copysign(distance,command.vx)*math.cos(obs.yaw),
             obs.xy[1]+math.copysign(distance,command.vx)*math.sin(obs.yaw))
        # Opposing damping retains the physical 35-degree stopping envelope;
        # planned traversal grades apply to intentional drive, not inertial
        # cancellation. Neither path forgives directional bans on this ray.
        return self._stopping_reason(obs,end,max(obs.speed,abs(command.vx)),commanded=not damping,record=record)

    def _guard(self, command, *, record=True, recovery_escape=True):
        o, g = self.obs, self.grid
        if command.omega:
            if self.executor.recovery_turn is not None:
                reason = self._recovery_turn_guard(o, self.executor.request, 'rotation')
                if reason: return reason
                if command.vx or command.omega*self.executor.recovery_turn.sign <= 0.:
                    return 'recovery turn cannot reverse yaw or add translation'
            else:
                if not o.turn_allowed or o.inclination > 15.: return 'turn requires gentle staging'
                if g.turn_support_reason(o.xy): return 'rotation footprint is unsupported'
        accel = max(.1, self.braking_accel or .2)
        if o.speed > .1:
            stopping = o.speed*o.speed/(2*accel)+.1
            end = tuple(np.asarray(o.xy)+stopping*np.asarray(o.velocity)/o.speed)
            if self._stopping_reason(o, end, o.speed, commanded=False, record=record,
                                     directional_exit=self._directional_exit_context(command) if recovery_escape else False):
                if record: self._record_guard_rejection('actual_drift',end,command)
                return 'actual drift stopping corridor is blocked'
        if command.vx:
            distance = max(.4, max(o.speed, abs(command.vx))**2/(2*accel)+.1)
            end = (o.xy[0]+math.copysign(distance, command.vx)*math.cos(o.yaw),
                   o.xy[1]+math.copysign(distance, command.vx)*math.sin(o.yaw))
            if self._stopping_reason(o, end, max(o.speed, abs(command.vx)), commanded=True, record=record):
                if record: self._record_guard_rejection('commanded_straight',end,command)
                return 'straight drive or stopping footprint is blocked'
        return ''

    def _stopping_reason(self, obs, end, speed, *, commanded, record=True,directional_exit=False):
        """Use the body sweep only when the circular obstacle margin rejects.

        Static route/turn clearance stays circular. All stopping intents share
        the same fresh rectangle, measured heading envelope and stopping fan.
        Terrain, observation and exclusion rejections cannot become exemptions.
        """
        g=self.grid
        if commanded:
            reason=g.support_reason(obs.xy,end) or g.grade_reason(obs.xy,end)
            if reason:
                return reason
            if g.segment_clear(obs.xy,end,no_go_exit=True):
                return ''
        else:
            reason=g.drift_segment_reason(obs.xy,end,no_go_exit=True,directional_exit=directional_exit)
            if not reason or not reason.startswith('obstacle:'):
                return reason
        offsets=self.passive_heading.offsets(obs.time,obs.yaw,speed/max(.1,self.braking_accel or .2))
        reason=oriented_stopping_reason(g,obs.xy,end,obs.yaw,offsets,commanded=commanded,directional_exit=directional_exit)
        if record and not reason and self.executor.request is not None:
            key=(self.executor.request.action_id,commanded)
            if key not in self.oriented_admission_seen:
                self.oriented_admission_seen.add(key)
                self.escape_events.append(dict(event='oriented_stopping_admitted',time=obs.time,
                    action_id=key[0],commanded=commanded,position=obs.xy,corridor_endpoint=end,
                    body_yaw=obs.yaw,passive_heading_offsets_rad=offsets,
                    reason='fresh continuous body sweep and stopping fan clear despite circular obstacle margin'))
        return reason

    def _record_guard_rejection(self, category, end, command):
        request=self.executor.request
        if request is None:
            return
        key=(request.action_id,category)
        if key in self.guard_evidence_seen:
            return
        self.guard_evidence_seen.add(key)
        o=self.obs
        event=dict(event='geometry_guard_rejection',time=o.time,action_id=request.action_id,
            category=category,position=list(o.xy),body_yaw=o.yaw,velocity_world=list(o.velocity),
            prospective_command=[command.vx,command.omega],corridor_endpoint=list(end))
        event['passive_heading_offsets_rad']=self.passive_heading.offsets(
            o.time,o.yaw,max(o.speed,abs(command.vx))/max(.1,self.braking_accel or .2))
        try:
            event['terrain_evidence']=self.grid.corridor_evidence(o.xy,end,o.yaw,o.time)
        except Exception as exc:
            # Evidence collection must not change a motion decision or hide the
            # guard's original rejection behind a diagnostic failure.
            event['diagnostic_error']=f'{type(exc).__name__}: {exc}'
        self.escape_events.append(event)

    def _brake_envelope(self, obs):
        """Authorize inertial damping, not a new reverse approach."""
        return self._brake_evaluation(obs)['reason']

    def _brake_evaluation(self, obs, command=None):
        """Return the original decision and its evidence from one evaluation.

        Only the executor's actual brake output can trigger a snapshot. Calls
        from staging or other admission probes never capture an event.
        """
        detail = dict(reason='', underlying_reason='', position=list(obs.xy),
                      velocity_world=list(obs.velocity), actual_speed_mps=obs.speed)
        if obs.speed < .03:
            return detail
        if not obs.fresh:
            detail.update(reason='brake corridor unobserved; zero desired velocity',
                          underlying_reason='local observation stale')
            return detail
        if command is None:
            longitudinal=sum(v*a for v,a in zip(obs.velocity,(math.cos(obs.yaw),math.sin(obs.yaw))))
            command=Motion(max(-.4,min(.4,-4.*longitudinal)),0.)
        distance = obs.speed**2/(2*max(.1, self.braking_accel or .2))+.1
        end = tuple(np.asarray(obs.xy)+distance*np.asarray(obs.velocity)/obs.speed)
        reason = self._stopping_reason(obs,end,obs.speed,commanded=False,
                                      directional_exit=self._directional_exit_context(command,damping=True))
        detail.update(corridor_endpoint=list(end),stopping_distance_m=distance,underlying_reason=reason)
        if reason:
            detail['reason'] = 'brake stopping envelope blocked; zero desired velocity'
        return detail

    def _recovery_staging_envelope(self, obs):
        """Fresh circular rotation room and live stopping room at actual pose."""
        if not all(math.isfinite(v) for v in (*obs.xy,obs.yaw,obs.up_z,*obs.velocity,
                                             obs.speed,obs.yaw_rate,obs.inclination,obs.time)):
            return 'recovery staging observation is not finite'
        g = self.grid
        certificate=self.executor.recovery_continuation
        if certificate is not None:
            reason=region_reason(g,certificate,fresh=True)
            if reason:return reason
        cells = g._support_cells(obs.xy)
        centre = g.world_to_cell(*obs.xy)
        if (cells is None or centre is None or not obs.fresh
                or not g.known[cells].all() or not g.fresh[cells].all()):
            return 'recovery staging rotation footprint is not fully observed and fresh'
        if not np.isfinite(g.slope[cells]).all() or (g.slope[cells] > math.tan(math.radians(12.))).any():
            return 'recovery staging rotation footprint is not gentle'
        if (g.hazard[cells].any() or g.no_go[cells].any()
                or g.approach_blocked[cells].any()
                # +inf is the map's explicit no-obstacle sentinel. NaN has
                # no clearance meaning; negative/insufficient values reject.
                or math.isnan(g.obstacle_distance[centre])
                or g.obstacle_distance[centre] < g.footprint_radius+g.clearance):
            return 'recovery staging rotation footprint contains obstacle or mission exclusion'
        return self._brake_envelope(obs)

    def _owns_ascent_motion(self, obs, request):
        """Observed execution owns this fixed mission leg, across F/zero."""
        e = self.executor
        executed = e.executed_traversal
        # This observer is updated once at tick entry, before path departure
        # handling as well as before the later RELEASE command selection.
        samples = e.uphill_samples
        if (not self.active_zero_envelope_enabled or self.operator_mode != 'auto' or not obs.fresh
                or not all(math.isfinite(v) for v in (*obs.xy,*obs.velocity,obs.yaw,obs.time,obs.up_z))
                or request is not e.request or request.purpose != 'mission' or request.direction != 1
                or e.state not in (MovementState.DRIVE_FORWARD, MovementState.RELEASE) or e.posture_pause is not None
                or not e._uphill_eligible(obs) or executed is None
                or executed.maneuver not in ('DRIVE_FORWARD', 'RELEASE') or executed.drive_sign != 1
                or executed.duration_s <= 0. or executed.interval_end_time != obs.time
                or e.committed_heading is None or len(samples) < 2):
            return False
        heading = math.atan2(e.travel_axis[1], e.travel_axis[0])
        if (abs(wrap_angle(executed.heading-heading)) > 1e-9
                or abs(wrap_angle(e.committed_heading-heading)) > 1e-9
                or samples[-1][0] != obs.time
                or not 2.-1e-9 <= obs.time-samples[0][0] <= 2.04+1e-9):
            return False
        horizon = max(obs.speed, request.speed)/max(.1, self.braking_accel or .2)
        if self.passive_heading.offsets(obs.time, obs.yaw, horizon) is None:
            return False
        return (all(math.isfinite(v) for t, xy in samples for v in (t, *xy))
                and all(0. < newer[0]-older[0] <= .04+1e-9
                        for older, newer in zip(samples, samples[1:])))

    def _owns_active_zero(self, obs, request):
        e = self.executor
        return (e.state == MovementState.RELEASE
                and obs.time-e.entered >= e.RELEASE_RESPONSE_SECONDS-1e-9
                and e.executed_traversal is not None
                and e.executed_traversal.maneuver == 'RELEASE'
                and self._owns_ascent_motion(obs, request))

    def _active_zero_departure(self, obs, request):
        """Shared F/zero intent bound; terrain permission comes from live guards."""
        e = self.executor
        if not self._owns_ascent_motion(obs, request):
            return 'path departure lacks established ascent execution evidence'
        start_time, start = e.uphill_samples[0]
        elapsed = obs.time-start_time
        delta = np.subtract(obs.xy, start)
        axis_rate = float(np.dot(delta, e.travel_axis))/elapsed
        gradient = np.asarray(obs.terrain_gradient)
        uphill_rate = float(np.dot(delta, gradient))/(float(np.linalg.norm(gradient))*elapsed)
        goal_rate = (math.dist(start,request.endpoint)-math.dist(obs.xy,request.endpoint))/elapsed
        admitted = request.action_id in self.active_zero_departures
        bound = math.dist(request.start,request.endpoint)+e.RELEASE_BACKSLIDE_METERS
        if not admitted and min(axis_rate,uphill_rate,goal_rate) < e.RELEASE_PROGRESS_MPS:
            return 'path departure has no useful measured ascent progress'
        if math.dist(obs.xy,request.endpoint) > bound:
            return 'ascent departure exceeds fixed goal-distance intent bound'
        reason = (self._release_envelope(obs,request) if admitted or e.state == MovementState.RELEASE
                  else self._guard(Motion(request.speed,0.)))
        if reason:return reason
        if not admitted:
            self.active_zero_departures[request.action_id] = bound
            e.ascent_intent_bound = bound
            reserve = max(.4, max(obs.speed,request.speed)**2/(2*max(.1,self.braking_accel or .2))+.1)
            end = (obs.xy[0]+reserve*math.cos(obs.yaw),obs.xy[1]+reserve*math.sin(obs.yaw))
            event = dict(event='ascent_continuation_admitted',time=obs.time,
                action_id=request.action_id,position=list(obs.xy),original_start=list(request.start),
                endpoint=list(request.endpoint),original_bound_m=request.lateral_tolerance,
                goal_distance_intent_bound_m=bound,axis_progress_mps=axis_rate,
                uphill_progress_mps=uphill_rate,goal_progress_mps=goal_rate,
                state=e.state.value,local_stopping_checked=True,onward_route_proven=False)
            try:
                event['terrain_evidence'] = self.grid.corridor_evidence(obs.xy,end,obs.yaw,obs.time)
            except Exception as exc:
                event['diagnostic_error'] = f'{type(exc).__name__}: {exc}'
            self.escape_events.append(event)
            e._observe_event('ascent_continuation_admitted',obs,event)
        return ''

    def _release_envelope(self, obs, request):
        """Reserve forward room for active zero travel and check actual drift."""
        certificate=self.executor.recovery_continuation
        if certificate is not None:
            reason=self._recovery_region_guard(obs,certificate,self.executor.recovery_core_started is not None)
            if reason:return reason
        # Once admitted, intent remains physically checked across actual F/zero
        # execution. The selector separately validates each emitted powered
        # command; zero must not inherit a hypothetical nominal-speed veto.
        admitted_owned = (self.executor.ascent_intent_bound is not None
                          and self._owns_ascent_motion(obs,request))
        if admitted_owned or self._owns_active_zero(obs, request):
            # Reserve the same physical room as nominal travel. Directional
            # planning grades belong to the separately selected powered command.
            speed = max(obs.speed, request.speed)
            distance = max(.4, speed**2/(2*max(.1, self.braking_accel or .2))+.1)
            end = (obs.xy[0]+distance*math.cos(obs.yaw), obs.xy[1]+distance*math.sin(obs.yaw))
            rejection = self._stopping_reason(obs, end, speed, commanded=False)
            if rejection:
                self._record_guard_rejection('active_zero_prospective', end, Motion())
            elif request.action_id not in self.active_zero_admissions:
                self.active_zero_admissions.add(request.action_id)
                self.escape_events.append(dict(event='active_zero_prospective_admitted',time=obs.time,
                    action_id=request.action_id,position=list(obs.xy),corridor_endpoint=list(end),
                    stopping_distance_m=distance,command_vx=0.,command_omega=0.))
        else:
            rejection = self._guard(Motion(request.direction*request.speed, 0.))
        if rejection:
            return 'release prospective corridor: '+rejection
        if obs.speed > .02:
            distance = max(.2, obs.speed**2/(2*max(.1, self.braking_accel or .2))+.1)
            end = tuple(np.asarray(obs.xy)+distance*np.asarray(obs.velocity)/obs.speed)
            reason = self._stopping_reason(obs,end,obs.speed,commanded=False,
                directional_exit=self._directional_exit_context(Motion(request.direction*request.speed,0.)))
            if reason:
                self.escape_events.append(dict(event='release_guard_rejection', time=obs.time,
                    category=reason.split(':', 1)[0], reason=reason, position=list(obs.xy),
                    velocity_world=list(obs.velocity), corridor_endpoint=list(end),
                    stopping_distance_m=distance, request_id=request.action_id,
                    prospective_command=[request.direction*request.speed, 0.]))
                return 'release actual drift corridor: '+reason
        return ''

    def _remaining_corridor(self, obs, request, heading):
        """Check remaining geometry before an alignment or a straight commit.

        Saved survey geometry supports the distant corridor; fresh local scans
        and stopping room are independently required by the per-command guard.
        """
        distance = math.dist(obs.xy, request.endpoint)
        emitted_end = (obs.xy[0]+request.direction*distance*math.cos(heading),
                       obs.xy[1]+request.direction*distance*math.sin(heading))
        if (not self._straight_clear(obs.xy, request.endpoint, heading, fresh=False)
                or not self._straight_clear(obs.xy, emitted_end, heading, fresh=False)):
            return 'remaining straight corridor unavailable from measured pose'
        return ''

    def _straight_entry_envelope(self, obs, request, yaw_trend, heading_offsets=(0., 0.)):
        """Check the stopping-footprint fan for measured passive yaw drift.

        Zero yaw remains the only emitted angular command. The observation
        window's net trend forecasts motion. The entire observed heading
        envelope is included even when oscillation has zero net trend.
        """
        speed = max(obs.speed, request.speed)
        accel = max(.1, self.braking_accel or .2)
        distance = max(.4, speed*speed/(2*accel)+.1)
        change = yaw_trend*speed/accel
        low, high = min(0., change, *heading_offsets), max(0., change, *heading_offsets)
        for offset in np.linspace(low, high, max(2, math.ceil((high-low)/.05)+1)):
            yaw = obs.yaw+offset
            end = (obs.xy[0]+request.direction*distance*math.cos(yaw),
                   obs.xy[1]+request.direction*distance*math.sin(yaw))
            if not self._straight_clear(obs.xy, end, yaw):
                return 'passive yaw forecast leaves checked straight stopping envelope'
        return ''

    def _straight_clear(self, start, end, yaw, fresh=True):
        g = self.grid
        if (g.segment_clear(start, end, require_fresh=fresh, no_go_exit=True)
                and not g.support_reason(start, end)):
            return True
        # Candidate geometry is exact-heading only. The emitted command must
        # separately pass the measured passive-yaw stopping fan on every tick.
        return not oriented_sweep_reason(g,start,end,yaw,require_fresh=fresh,no_go_exit=True)

    def _commit_prediction(self):
        if self.attempt is None or self.prediction_committed: return True
        o = self.obs
        path = [o.xy]+list(self.path[1:])
        sign = self.executor.request.direction
        features = route_features(self.grid, path, o.yaw, o.speed, o.up_z, sign*self.command_limit)
        p0, reason = self.model.predict(features)
        heading = math.atan2(path[1][1]-o.xy[1], path[1][0]-o.xy[0])
        key = self.evidence.key(o.xy, path[-1], heading, sign*self.command_limit, path=path)
        estimate = self.evidence.estimate(key, p0, reason)
        if self.mode == 'qualified' and eligibility(estimate, self._retreat_exists(path), self.capability) not in ('normal', 'fallback'):
            return False
        self.attempt.update(time=o.time, start=o.xy, features=features, key=key, estimate=estimate, path=path)
        self.reachability = asdict(estimate)
        self.attempt_events.append(dict(event='approach_start', attempt_id=self.attempt['id'], time=o.time,
            run_id=self.run_id, features=features, prediction=self.reachability, endpoint=list(path[-1]),
            approach_key=key, mode=self.mode, protocol=self.protocol_version))
        if self.attempt.get('exploration'):
            self.attempt_events[-1]['exploration'] = dict(self.attempt['exploration'],
                stage='adopted; actual sustained climb must be checked in movement outcomes')
        if self.attempt.get('paired_launch_variant'):
            self.attempt_events[-1]['paired_launch_variant'] = dict(self.attempt['paired_launch_variant'],
                stage='adopted; same-class launch comparison, physical reachability unvalidated')
        self.prediction_committed = True
        return True

    def _active(self, _):
        if self.executor.request is None: return Status.FAILURE
        if self.recovering and self.obs.time-self.recovery_since > 120.:
            self.recovery_permission = None
            self.executor.finish('censored', 'recovery episode deadline elapsed', self.obs)
            self.command = self.executor.brake(self.obs)
        elif not self._commit_prediction():
            self.executor.finish('censored', 'primitive entry no longer meets probability gate', self.obs)
            self.command = self.executor.brake(self.obs)
        else:
            self.command = self.executor.tick(self.obs, self._guard, self._supported)
        self.state = 'RECOVER' if self.recovering else 'FOLLOW'
        self.reason = self.executor.reason
        certificate=self.executor.recovery_continuation
        self.target = (certificate.center if certificate else
            self.executor.request.endpoint if self.executor.request else self.checkpoint)
        return Status.RUNNING

    def _invalidate(self):
        self.plan_generation += 1
        self.plan_pending = False

    def _result(self, _):
        result = self.executor.consume()
        if result is None: return Status.FAILURE
        o = self.obs
        self.attempt_events.append(dict(event='movement_outcome', **asdict(result), protocol=self.protocol_version))
        self._record_recovery_turn_failure(result)
        staging_rejection = ''
        if result.disposition == ResultDisposition.REPOSITIONED and result.phase == MovementState.RECOVERY_STAGE.value:
            staging_rejection = self._recovery_staging_envelope(o)
            if staging_rejection:
                self.escape_events.append(dict(event='recovery_stage_handoff_rejected',time=o.time,
                    action_id=result.action_id,position=o.xy,reason=staging_rejection))
        onward = self._recovery_connection(o.xy,o.yaw,live=True) if self.recovering else None
        usable_staging = o.fresh and o.inclination <= 12. and o.turn_allowed and self._supported(o.xy)
        if (result.disposition == ResultDisposition.REPOSITIONED and not staging_rejection
                and (usable_staging or onward is not None)):
            self.finish_attempt('censored', result.reason, o.xy, o.yaw, o.time)
            self._invalidate()
            self.segments = []
            self.recovering = False
            self.recovery_permission = None
            self.recovery_since = None
            self.observe_revision = self.grid.revision
            self.set_state('OBSERVE', 'useful supported displacement; plan from measured pose', o.time)
        elif result.outcome != 'success':
            physical = result.physical and self.prediction_committed
            reason = staging_rejection or result.reason
            self.finish_attempt('failure' if physical else 'censored', reason, result.position, o.yaw,
                o.time if staging_rejection else result.time,
                physical=physical, execution=asdict(result.executed_traversal) if result.executed_traversal else None,
                terminal_maneuver=result.phase)
            self._begin_recovery(reason)
        elif self.recovering:
            self.finish_attempt('success', result.reason, o.xy, o.yaw, o.time)
            self.recovery_progress += 1.
            self.segments = []
            self._invalidate()
            if usable_staging or onward is not None:
                self.recovering = False
                self.recovery_permission = None
                self.recovery_since = None
                self.observe_revision = self.grid.revision
                self.escape_events.append(dict(event='recovery_mission_handoff',time=o.time,
                    position=list(o.xy),onward_checked=onward is not None,
                    reason='executable onward connection' if onward is not None else 'usable turn staging'))
                self.set_state('OBSERVE', 'recovery reached mission continuation or staging', o.time)
        else:
            self.segment_index += 1
            if self.segment_index >= len(self.segments):
                self.finish_attempt('success', 'checkpoint reached and settled', o.xy, o.yaw, o.time)
                self.visited_checkpoints.append(tuple(self.checkpoint))
                self.segments = []
                if self.is_mission_checkpoint:
                    self.wp_idx += 1
                self.observe_revision = self.grid.revision
                self.set_state('COMPLETE' if self.wp_idx >= len(self.waypoints) else 'OBSERVE',
                               'destination reached and settled' if self.wp_idx >= len(self.waypoints) else 'checkpoint reached; observe', o.time)
        self.command = self.executor.brake(o)
        return Status.RUNNING

    def _begin_recovery(self, reason):
        self._invalidate()
        self.segments = []
        if not self.recovering:
            self.recovery_since = self.obs.time
            self.recovery_trials = set()
            self.recovery_direction = None
            self.recovery_actions = 0
            self.recovery_turn_used = False
            self.recovery_turn_failure_id = None
            self.recovery_request_id = None
            self.recovery_continuation_used = False
            self.recovery_permission = RecoveryPermission.capture(self.grid,self.obs.xy,self.obs.time)
            self.recovery_route = [tuple(p) for p in self.mission_route]
            self.recovery_origin = self.obs.xy
            self.escape_events.append(dict(event='recovery_episode_started',time=self.obs.time,
                origin=list(self.obs.xy),failure_ids=sorted({p.failure_id for p in self.recovery_permission.patches})))
        self.recovering = True
        self.prediction_committed = False
        self.set_state('RECOVER', reason, self.obs.time)

    def _start(self, endpoint, direction, purpose):
        o = self.obs
        self.action_number += 1
        length = math.dist(o.xy, endpoint)
        if length < .12:
            purpose = 'settle'
        width = self._corridor_tolerance(o.xy, endpoint) if purpose == 'mission' else .5
        request = MovementRequest(f'{self.run_id}:move:{self.action_number}', o.xy, tuple(endpoint),
            direction, self.command_limit, o.time+max(30., length/.02+15.), purpose,
            lateral_tolerance=width, arrival_radius=min(.3, max(.12, length*.25)))
        self.executor.start(request, o.time)
        self.recovery_request_id = request.action_id if purpose == 'recovery' else None
        self.escape_events.append(dict(event='movement_envelope', time=o.time, action_id=request.action_id,
            lateral_tolerance_m=width, arrival_radius_m=request.arrival_radius,
            reason='wider tracking room requires the entire observed swept band to be supported'))
        self.target = tuple(endpoint)
        # Alignment and braking can physically translate the learned policy.
        # Predict once for the complete leg before its first primitive output,
        # including yaw; later segments never create independent evidence votes.
        if not self._commit_prediction():
            self.executor.finish('censored', 'primitive entry no longer meets probability gate', o)
        self.command = self.executor.brake(o)

    def _corridor_tolerance(self, start, end):
        """Allow natural drift only in an independently checked wider corridor.

        Each band includes the full footprint. Sampling is no wider than a map
        cell; hazards between the two outside boundaries are not skipped.
        Live command/arrival guards still validate the actual pose.
        """
        length = math.dist(start, end)
        if length < .5:
            return .5
        normal = np.array((-(end[1]-start[1])/length, (end[0]-start[0])/length))
        width = .5
        checked = 0.
        for candidate in (.6, .8, 1.):
            for offset in np.arange(checked+self.grid.cell_size/2, candidate+self.grid.cell_size/4, self.grid.cell_size/2):
                for sign in (-1, 1):
                    a = tuple(np.asarray(start)+sign*offset*normal)
                    b = tuple(np.asarray(end)+sign*offset*normal)
                    if (not self.grid.segment_clear(a, b, require_fresh=False)
                            or self.grid.support_reason(a, b)):
                        return width
            checked = candidate
            width = candidate
        return width

    def _recovery_connection(self, start, yaw, *, live=False):
        """Bounded checks against the saved mission suffix, never new A* per tick.

        Candidate connections are geometry hypotheses. Actual handoff adds the
        current signed-command and passive-heading stopping checks.
        """
        o,g=self.obs,self.grid
        for index in range(1,len(self.recovery_route)):
            path=compact_path([start]+self.recovery_route[index:])
            if len(path)<2 or motion_route_reason(g,path,yaw):continue
            if any(observed_edge_reason(g,a,b,o.xy) for a,b in zip(path,path[1:])):continue
            if live:
                heading=math.atan2(path[1][1]-start[1],path[1][0]-start[0])
                needs_turn=abs(wrap_angle(heading-yaw))>.10
                if needs_turn:
                    if not o.turn_allowed or self._guard(Motion(0.,.35),record=False,recovery_escape=False):continue
                else:
                    if self._guard(Motion(self.command_limit,0.),record=False,recovery_escape=False):continue
                    offsets=self.passive_heading.offsets(o.time,o.yaw,
                        max(o.speed,self.command_limit)/max(.1,self.braking_accel or .2))
                    if offsets is None:continue
                    probe=MovementRequest('onward-probe',o.xy,path[1],1,self.command_limit,o.time+1.)
                    if self._straight_entry_envelope(o,probe,0.,offsets):continue
                    if self._remaining_corridor(o,probe,o.yaw):continue
            return path
        return None

    def _recovery_purpose(self, end, yaw=None):
        g,o=self.grid,self.obs
        yaw = o.yaw if yaw is None else yaw
        connection=self._recovery_connection(end,yaw)
        if connection is not None:
            return 0,'checked mission connection',g.route_cost(connection,o.yaw)
        if g.max_support_slope(end)<=12. and not g.turn_support_reason(end):
            return 1,'usable turn staging; onward route unproven',0.
        cell=g.world_to_cell(*end)
        permission=self.recovery_permission
        if permission is not None and permission.patches:
            occupied=frozenset(c for p in permission.patches for c in p.cells)
            if cell not in occupied:
                return 2,'departure from occupied failure patch; onward route unproven',0.
        # A partial hop can approach known traversed terrain without demanding
        # a complete onward route. Repeated same-cell trials remain excluded.
        for point in reversed(self.supported_trail):
            before,after=math.dist(o.xy,point),math.dist(end,point)
            if after+.1<before and self._straight_clear(end,point,yaw,fresh=False):
                return 2,'toward previously supported corridor; onward route unproven',after
        return None

    def _record_recovery_turn_failure(self, result):
        executed = result.executed_traversal
        if (self.recovering and self.recovery_since is not None
                and result.action_id == self.recovery_request_id
                and result.outcome == 'failure' and result.physical
                and executed is not None and executed.maneuver in ('DRIVE_FORWARD','DRIVE_REVERSE')
                and executed.duration_s > 0.
                and self.recovery_since <= executed.first_emitted_time < executed.interval_end_time <= result.time):
            self.recovery_turn_failure_id = result.action_id

    def _recovery_turn_guard(self, obs, request, phase):
        intent = self.executor.recovery_turn
        if (intent is None or request is None or intent.action_id != request.action_id
                or request.purpose != 'recovery' or not self.recovering
                or self.operator_mode != 'auto' or self.mode != 'characterize'):
            return 'no owned recovery turn permission'
        reason = rotation_reason(self.grid, obs, intent)
        if reason or phase == 'rotation':
            return reason
        # Forecast only from actual post-turn measurements. No synthetic yaw
        # samples or inherited pre-turn stability may authorize translation.
        offsets = self.passive_heading.offsets(obs.time, obs.yaw, 2.)
        if offsets is None:
            return 'post-turn measured heading history unavailable'
        if arrival_ray_interval(obs.xy, request.endpoint, obs.yaw, request.direction,
                                request.arrival_radius) is None:
            return 'post-turn measured ray misses the frozen endpoint'
        return (exit_reason(self.grid, obs, replace(intent, target_yaw=obs.yaw), offsets)
                or self._straight_entry_envelope(obs, request, 0., offsets))

    def _recovery_turn_candidates(self):
        o, g = self.obs, self.grid
        if (not o.fresh or o.speed >= .12 or abs(o.yaw_rate) >= .12
                or o.body_tilt >= 50. or not 15. < o.inclination <= 35.):
            return []
        offsets = self.passive_heading.offsets(o.time, o.yaw, 2.)
        if offsets is None:
            return []
        candidates = []
        for degrees in (-10, 10, -20, 20, -30, 30):
            yaw = wrap_angle(o.yaw+math.radians(degrees))
            for direction in (-1, 1):
                end = (o.xy[0]+direction*.4*math.cos(yaw), o.xy[1]+direction*.4*math.sin(yaw))
                intent = RecoveryTurn(o.xy, o.yaw, yaw, end, direction, self.recovery_since+120.)
                rejection = rotation_reason(g, o, intent) or exit_reason(g, o, intent, offsets)
                purpose = None if rejection else self._recovery_purpose(end, yaw)
                self.escape_events.append(dict(event='recovery_turn_candidate', time=o.time,
                    delta_deg=degrees, direction=direction, endpoint=end,
                    rejection=rejection or ('' if purpose else 'no useful checked reposition'),
                    onward_checked=bool(purpose and purpose[0] == 0)))
                if purpose is None:
                    continue
                features = route_features(g, [o.xy, end], yaw, o.speed, o.up_z, direction*self.command_limit)
                p0, detail = self.model.predict(features)
                heading = wrap_angle(yaw+(math.pi if direction < 0 else 0.))
                key = self.evidence.key(o.xy, end, heading, direction*self.command_limit, path=[o.xy, end])
                estimate = self.evidence.estimate(key, p0, detail)
                rank, reason, cost = purpose
                score = (False, rank, abs(degrees), cost+g.route_cost([o.xy,end],yaw), direction < 0)
                trial = ('turn', round(o.xy[0]*2), round(o.xy[1]*2), degrees, direction)
                candidates.append((score, direction, end, trial, features, estimate, key, heading, reason, intent))
        return candidates

    def _recovery(self, _):
        if not self.recovering: return Status.FAILURE
        o, g = self.obs, self.grid
        if o.time-self.recovery_since > 120. or self.recovery_actions >= 8:
            self.recovery_permission = None
            self.command = self.executor.brake(o)
            self.set_state('HOLD', 'straight recovery exhausted; no usable staging approach', o.time)
            return Status.RUNNING
        self.command = self.executor.brake(o)
        if not o.fresh or o.time-self.last_plan_at < 1.:
            return Status.RUNNING
        self.last_plan_at = o.time
        candidates = []
        for direction in (-1, 1):
            if self._guard(Motion(direction*self.command_limit, 0.)):
                continue
            for distance in (.4, .8, 1.6, 2.4):
                key = (round(o.xy[0]*2), round(o.xy[1]*2), round(o.yaw/.2), direction, distance)
                if key in self.recovery_trials: continue
                end = (o.xy[0]+direction*distance*math.cos(o.yaw), o.xy[1]+direction*distance*math.sin(o.yaw))
                if not self._straight_clear(o.xy, end, o.yaw): continue
                purpose=self._recovery_purpose(end)
                if purpose is None:continue
                features = route_features(g, [o.xy, end], o.yaw, o.speed, o.up_z, direction*self.command_limit)
                p0, detail = self.model.predict(features)
                heading = wrap_angle(o.yaw+(math.pi if direction < 0 else 0.))
                evidence_key = self.evidence.key(o.xy, end, heading, direction*self.command_limit, path=[o.xy, end])
                estimate = self.evidence.estimate(evidence_key, p0, detail)
                tier = eligibility(estimate, self._retreat_exists([o.xy, end]), self.capability)
                if self.mode == 'qualified' and tier not in ('normal', 'fallback'): continue
                rank,reason,onward_cost=purpose
                score=(tier=='fallback',rank,g.route_cost([o.xy,end],o.yaw)+onward_cost,
                       distance,0 if direction==1 else 1)
                candidates.append((score, direction, end, key, features, estimate, evidence_key, heading,reason, None))
        # A physically failed executed straight recovery justifies trying a
        # new heading once. Preserve better checked connections/staging first;
        # a planning censor or a braking-only failure never triggers this.
        unproven_hops = candidates and all(v[0][1] >= 2 for v in candidates)
        if ((not candidates or self.recovery_turn_failure_id is not None and unproven_hops)
                and not self.recovery_turn_used and self.mode == 'characterize'
                and o.speed < .12 and abs(o.yaw_rate) < .12
                and self.passive_heading.offsets(o.time,o.yaw,2.) is not None):
            self.recovery_turn_used = True
            turns = self._recovery_turn_candidates()
            if turns:
                candidates = turns
        if not candidates:
            self.reason = 'no checked straight escape or bounded recovery turn'
            return Status.RUNNING
        _, direction, end, key, features, estimate, evidence_key, heading,reason, turn = min(candidates, key=lambda v: v[0])
        self.recovery_direction = direction
        self.recovery_trials.add(key)
        self.recovery_actions += 1
        self.recovery_count = self.recovery_actions
        self.route_purpose = 'adjust'
        self.path = [o.xy, end]
        self.checkpoint = end
        self.attempt_number += 1
        self.attempt = dict(id=f'{self.run_id}:{self.attempt_number}', time=o.time, start=o.xy,
            endpoint=end, heading=heading, features=features, estimate=estimate, key=evidence_key, path=self.path)
        self.prediction_committed = False
        self._start(end, direction, 'recovery')
        if turn is not None and self.executor.request is not None:
            self.executor.recovery_turn = replace(turn, action_id=self.executor.request.action_id)
            self.escape_events.append(dict(event='recovery_turn_committed', time=o.time,
                **asdict(self.executor.recovery_turn), onward_checked=False,
                purpose=reason, trigger_failure_id=self.recovery_turn_failure_id,
                evidence='unvalidated recovery characterization'))
        self.escape_events.append(dict(event='maneuver_start', time=o.time, position=list(o.xy),
            endpoint=list(end), vx=direction*self.command_limit, omega=0., reason=reason,
            candidate_count=len(candidates),onward_checked=reason=='checked mission connection'))
        return Status.RUNNING

    def _leg_action(self, _):
        if not self.segments: return Status.FAILURE
        remaining = [self.obs.xy]+self.segments[self.segment_index:]
        rejection = motion_route_reason(self.grid, remaining, self.obs.yaw)
        if rejection:
            self.finish_attempt('censored', rejection, self.obs.xy, self.obs.yaw, self.obs.time)
            self._begin_recovery(rejection)
            self.command = self.executor.brake(self.obs)
            return Status.RUNNING
        self._start(self.segments[self.segment_index], 1, 'mission')
        return Status.RUNNING

    def _mission(self, _):
        o = self.obs
        self.command = self.executor.brake(o)
        if not o.fresh or o.speed > .12 or o.time-self.since < 3. or self.grid.revision-self.observe_revision < 3:
            self.reason = 'observe and settle before planning'
            return Status.RUNNING
        if o.time-self.last_plan_at < .5: return Status.RUNNING
        self.last_plan_at = o.time
        if self._plan(o.xy, o.yaw, o.time):
            # The planner's selection is provisional. The actual prediction is
            # recorded before the first primitive, since even yaw can translate.
            self.attempt_events[-1]['event'] = 'approach_selection'
            self.prediction_committed = False
            self.path = compact_path(self.path)
            self.segments = self.path[1:]
            self.segment_index = 0
        elif not self.plan_pending:
            if self.mode == 'qualified' or self._exploring_initial_route():
                self.set_state('HOLD', self.planning_failure, o.time)
            else:
                self._begin_recovery(self.planning_failure)
        return Status.RUNNING

    def motion_metrics(self):
        return dict(self.executor.metrics)

    def navigation_status(self):
        status = super().navigation_status()
        status.update(architecture='behavior_tree', protocol=self.protocol_version, branch=self.tree_branch,
            movement_state=self.executor.state.value, action_id=self.executor.request.action_id if self.executor.request else None,
            support_inclination_deg=self.obs.inclination if self.obs else None,
            body_tilt_deg=self.obs.body_tilt if self.obs else None,
            turn_enabled=self.turn_enabled, uphill_limit_deg=self.grid.uphill_limit_deg,
            side_limit_deg=self.grid.side_limit_deg)
        status['brake_envelope'] = self.executor.brake_rejection or 'checked inertial deceleration'
        status['release'] = dict(active=self.executor.state == MovementState.RELEASE,
            directional_grade_deg=self.executor.metrics.get('directional_grade_deg'),
            uphill_progress_mps=self.executor.metrics.get('uphill_progress_mps'),
            policy_active=True, calibrated=False)
        return status
