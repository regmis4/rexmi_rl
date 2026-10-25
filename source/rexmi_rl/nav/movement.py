"""Exclusive teleop-style motion executor. No mapping, planning or policy calls."""
from dataclasses import dataclass
from enum import Enum
import math

from ..drive_adapter import directional_command
from .checkpoint_control import Motion, wrap_angle
from .swept_footprint import arrival_ray_interval
from .release_memory import ReleaseResponseMemory


UPHILL_SPEED_RESPONSE_PROFILE = dict(
    uphill_speed_response_enabled=True,
    uphill_speed_response_trigger_mps=.45,
    uphill_speed_response_intervention_mps=.50,
    uphill_speed_response_seconds=.20,
    uphill_speed_response_window_seconds=2.,
    uphill_speed_response_min_progress_mps=.02,
    uphill_speed_response_contract='characterization opt-in; each bounded protective zero episode requires fresh continuous actual-forward 2 s history with axis and gradient progress >=0.02, current positive projections, valid normal/attitude and up_z>=0.8; >=0.50 or failed conditions use damping; <=0.45 decay independently checks and restores interrupted DRIVE; deliberate slowdown RELEASE unchanged; accumulated powered exposure retained without counting zero time; no clock renewal or success vote; provisional intervention boundary, not physical speed guarantee')


class MovementState(str, Enum):
    IDLE = 'IDLE'
    BRAKE = 'BRAKE'
    ALIGN_LEFT = 'ALIGN_LEFT'
    ALIGN_RIGHT = 'ALIGN_RIGHT'
    DRIVE_FORWARD = 'DRIVE_FORWARD'
    DRIVE_REVERSE = 'DRIVE_REVERSE'
    RELEASE = 'RELEASE'
    POSTURE_PAUSE = 'POSTURE_PAUSE'
    SETTLE = 'SETTLE'
    RECOVERY_STAGE = 'RECOVERY_STAGE'
    STOP = 'STOP'


class ResultDisposition(str, Enum):
    TERMINAL = 'terminal'
    REPOSITIONED = 'repositioned'


@dataclass(frozen=True)
class MovementRequest:
    action_id: str
    start: tuple
    endpoint: tuple
    direction: int
    speed: float
    deadline: float
    purpose: str = 'mission'
    lateral_tolerance: float = .5
    arrival_radius: float = .12

    def __post_init__(self):
        if self.direction not in (-1, 1) or not 0 < self.speed <= .4:
            raise ValueError('Movement requires a latched direction and <=0.40 m/s')
        if not all(math.isfinite(x) for x in (*self.start, *self.endpoint, self.deadline)):
            raise ValueError('Movement requires finite geometry and a simulation deadline')
        if math.dist(self.start, self.endpoint) < .05 and self.purpose != 'settle':
            raise ValueError('Movement endpoint is too close')
        if not .1 <= self.lateral_tolerance <= 1. or not .05 <= self.arrival_radius <= .3:
            raise ValueError('Movement corridor and arrival bounds must be finite and bounded')


@dataclass(frozen=True)
class ExecutedTraversal:
    maneuver: str
    drive_sign: int
    heading: float
    body_yaw: float
    start_position: tuple
    last_position: tuple
    first_emitted_time: float
    interval_end_time: float
    duration_s: float


@dataclass
class PosturePause:
    started: float
    intent: MovementState
    stable_since: float | None = None
    braking_required: bool = False
    motion_stable_since: float | None = None


@dataclass(frozen=True)
class MovementResult:
    action_id: str
    outcome: str
    reason: str
    physical: bool
    time: float
    position: tuple
    phase: str = ''
    disposition: ResultDisposition = ResultDisposition.TERMINAL
    executed_traversal: ExecutedTraversal | None = None


@dataclass(frozen=True)
class MotionObservation:
    xy: tuple
    yaw: float
    velocity: tuple
    up_z: float
    yaw_rate: float
    time: float
    inclination: float
    turn_allowed: bool
    fresh: bool
    terrain_gradient: tuple | None = None
    body_quaternion: tuple | None = None

    @property
    def speed(self):
        return math.hypot(*self.velocity)

    @property
    def body_tilt(self):
        return math.degrees(math.acos(max(-1., min(1., self.up_z))))

    @property
    def terrain_relative_tilt(self):
        """Angle of measured body up to a fresh fitted world normal, if valid.

        This is not the contact support plane or a validated stability margin.
        Invalid/inconsistent attitude must never grant a posture exemption.
        """
        q,g = self.body_quaternion,self.terrain_gradient
        if (not self.fresh or q is None or g is None or len(q)!=4 or len(g)!=2
                or not all(math.isfinite(v) for v in (*q,*g,self.up_z,self.yaw,self.inclination))):
            return None
        norm2=sum(v*v for v in q)
        if abs(norm2-1.)>1e-3:return None
        w,x,y,z=(v/math.sqrt(norm2) for v in q)
        body_up=(2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y))
        yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
        if abs(body_up[2]-self.up_z)>1e-3 or abs(wrap_angle(yaw-self.yaw))>1e-3:
            return None
        scale=math.hypot(g[0],g[1],1.)
        if not math.isfinite(scale):return None
        normal=(-g[0]/scale,-g[1]/scale,1./scale)
        dot=sum(a*b for a,b in zip(body_up,normal))
        if not math.isfinite(dot):return None
        return math.degrees(math.acos(max(-1.,min(1.,dot))))


class MovementExecutor:
    """One owner, one immutable request, one terminal result per request.

    Guards validate the command actually sent. BRAKE has longitudinal damping
    only; STOP means the same zero command as teleop's space/release action.
    """
    # Characterization settings, not timings inferred to be universally optimal.
    RELEASE_DRIVE_SECONDS = 5.
    DRIVE_RESPONSE_SECONDS = 2.
    DRIVE_RELEASE_PROGRESS_MPS = .10
    DRIVE_RELEASE_DWELL_SECONDS = 1.
    DRIVE_RESPONSE_MAX_GAP_SECONDS = .04
    RELEASE_RESPONSE_SECONDS = 2.
    RELEASE_BACKSLIDE_METERS = .08
    RELEASE_PROGRESS_MPS = .01
    RELEASE_PLATEAU_SECONDS = 5.
    UPHILL_PATIENCE_SECONDS = 12.
    UPHILL_WALL_SECONDS = 24.
    UPHILL_RENEWAL_METERS = .10
    POSTURE_RELATIVE_ENVELOPE_DEG = 20.
    POSTURE_ABSOLUTE_PAUSE_DEG = 50.
    RECOVERY_STAGE_SECONDS = 6.
    RECOVERY_STAGE_DWELL_SECONDS = .6
    RECOVERY_STAGE_MIN_DISPLACEMENT = .4
    RECOVERY_STAGE_MAX_DRIFT = .5
    POSTURE_PAUSE_SECONDS = 1.5
    POSTURE_RECOVERY_SECONDS = .2
    ARRIVAL_HEADING_RESERVE = .75

    def __init__(self):
        self.state = MovementState.IDLE
        self.request = None
        self.result = None
        self.reason = 'idle'
        self.entered = 0.
        self.started = 0.
        self.stable_since = None
        self.samples = []
        self.stall_since = None
        self.stall_origin = None
        self.history = []
        self.seen = set()
        self.brake_guard = None
        self.brake_evaluator = None
        self.last_brake_evaluation = None
        self.event_observer = None
        self.corridor_guard = None
        self.straight_entry_guard = None
        self.entry_samples = []
        self.release_guard = None
        self.release_departure_guard = None
        self.ascent_intent_bound = None
        self.posture_guard = None
        self.recovery_stage_guard = None
        self.recovery_turn_guard = None
        self.recovery_turn = None
        self.recovery_turn_finished = False
        self.recovery_turn_samples = []
        self.drive_selector = None
        self.last_selected_speed = None
        self.recovery_stage_started = None
        self.recovery_stage_anchor = None
        self.recovery_stage_samples = []
        self.departure_observer = None
        self.recovery_continuation_provider = None
        self.recovery_continuation_guard = None
        self.recovery_continuation_admitted = None
        self.recovery_continuation = None
        self.recovery_execution_heading = None
        self.recovery_core_started = None
        self.recovery_core_anchor = None
        self.recovery_motion_samples = []
        self.recovery_region_samples = []
        self.recovery_region_stall_since = None
        self.posture_pause = None
        self.committed_heading = None
        self.travel_axis = None
        self.heading_deviation_reported = False
        self.arrival_heading_reported = False
        self.brake_rejection = ''
        self.brake_episode_started = None
        self.pending_drive_continuation = None
        self.posture = []
        self.pulse_direction = 0
        self.alignment_best = math.inf
        self.alignment_progress_at = 0.
        self.drive_exposure = 0.
        self.drive_response_samples = []
        self.drive_slow_since = None
        self.uphill_speed_response_enabled = False
        self.speed_response_used = False
        self.speed_response_started = None
        self.last_output_time = None
        self.forward_output = False
        self.release_origin = None
        self.release_bad_since = None
        self.uphill_patience = False
        self.executed_traversal = None
        self.pending_traversal = None
        self.suspended_release = False
        self.release_memory = ReleaseResponseMemory()
        self.uphill_samples = []
        self.uphill_goal_only_budget = False
        self.uphill_stall_since = None
        self.uphill_stall_origin = None
        self.uphill_exposure = 0.
        self.posture_signal = None
        self.metrics = dict(status='idle', forward_mps=0., progress_mps=0., lateral_mps=0., window_s=0.)

    def _state(self, state, now, reason):
        if state in (MovementState.IDLE,MovementState.STOP,MovementState.ALIGN_LEFT,
                     MovementState.ALIGN_RIGHT,MovementState.SETTLE):
            self.release_memory.clear()
        if state != MovementState.RELEASE and self.speed_response_started is not None:
            self.history.append(dict(event='uphill_speed_response_ended',time=now,
                action_id=self.request.action_id if self.request else None,
                elapsed_s=now-self.speed_response_started,reason=reason,decayed=False))
            self.speed_response_started = None
        if state != MovementState.DRIVE_FORWARD:
            self.drive_response_samples = []
            self.drive_slow_since = None
        if state not in (MovementState.DRIVE_FORWARD,MovementState.DRIVE_REVERSE):
            self.pending_drive_continuation = None
        # Posture observation can temporarily own a braking action. Its entry
        # time must not renew the original bounded braking episode.
        if state == MovementState.BRAKE and self.brake_episode_started is None:
            self.brake_episode_started = now
        elif state not in (MovementState.BRAKE, MovementState.POSTURE_PAUSE, MovementState.RECOVERY_STAGE):
            self.brake_episode_started = None
        if self.state != state:
            travel = (MovementState.DRIVE_FORWARD, MovementState.DRIVE_REVERSE, MovementState.RELEASE)
            if state == MovementState.BRAKE and self.state in travel:
                # A new correction episode gets its own angular-progress clock.
                # Coast phases between pulses retain the same clock.
                self.alignment_best, self.alignment_progress_at = math.inf, now
            if state == MovementState.BRAKE:
                self.entry_samples = []
            self.history.append(dict(event='movement_transition', time=now,
                previous=self.state.value, state=state.value, reason=reason,
                action_id=self.request.action_id if self.request else None))
            preserve_progress = (self.recovery_continuation is not None
                or self.state in travel and state in (*travel, MovementState.RECOVERY_STAGE))
            self.state, self.entered = state, now
            self.metrics['release_active'] = state == MovementState.RELEASE
            self.stable_since = None
            if not preserve_progress:
                self.samples = []
                self.stall_since = None
                self.stall_origin = None
            if state in (MovementState.RELEASE, MovementState.ALIGN_LEFT, MovementState.ALIGN_RIGHT):
                self.drive_exposure = 0.
                self.metrics['forward_exposure_s'] = 0.
            self.release_bad_since = None
            if state in (MovementState.ALIGN_LEFT, MovementState.ALIGN_RIGHT):
                self.executed_traversal = None
                self.pending_traversal = None
                self.uphill_samples = []
                self.pulse_direction = 1 if state == MovementState.ALIGN_LEFT else -1
            if state not in (MovementState.BRAKE, MovementState.RELEASE):
                self.suspended_release = False
            if state in (MovementState.IDLE, MovementState.STOP,
                         MovementState.ALIGN_LEFT, MovementState.ALIGN_RIGHT, MovementState.SETTLE):
                self.posture_pause = None
        self.reason = reason

    def _step_recovery_turn(self, obs, guard):
        """One scoped use of ALIGN; normal climbing never receives yaw permission."""
        intent, req = self.recovery_turn, self.request
        reason = self.recovery_turn_guard(obs, req, 'rotation') if self.recovery_turn_guard else 'missing recovery turn guard'
        if reason or obs.time-self.alignment_progress_at > 12.:
            self.finish('censored', reason or 'recovery turn alignment deadline elapsed', obs)
            return self.brake(obs)
        error = wrap_angle(intent.target_yaw-obs.yaw)
        if not self.recovery_turn_finished:
            if intent.sign*error <= .02:
                self.recovery_turn_finished = True
                self.recovery_turn_samples = []
                self._state(MovementState.BRAKE, obs.time, 'observe measured heading after recovery turn')
                return self.brake(obs)
            if obs.speed >= .12 or abs(obs.yaw_rate) >= .12 and self.state == MovementState.BRAKE:
                self.stable_since = None
                return self.brake(obs)
            if self.state == MovementState.BRAKE:
                self.stable_since = self.stable_since if self.stable_since is not None else obs.time
                if obs.time-self.stable_since < .3:
                    return self.brake(obs)
                self._state(MovementState.ALIGN_LEFT if intent.sign > 0 else MovementState.ALIGN_RIGHT,
                            obs.time, 'committed bounded recovery turn; onward route unproven')
            if obs.time-self.entered >= 2.:
                self.recovery_turn_finished = True
                self.recovery_turn_samples = []
                self._state(MovementState.BRAKE, obs.time, 'recovery yaw pulse ended; observe before drive')
                return self.brake(obs)
            command = Motion(*directional_command(left=intent.sign > 0, right=intent.sign < 0, omega=.35))
            reason = guard(command)
            if reason:
                self.finish('censored', reason, obs)
                return self.brake(obs)
            return command
        # No opposite pulse and no renewed yaw attempt. Collect actual,
        # contiguous settled observations after the final angular command.
        samples = self.recovery_turn_samples
        if (obs.speed >= .12 or abs(obs.yaw_rate) >= .12
                or samples and not 0. < obs.time-samples[-1][0] <= .041):
            samples.clear()
        else:
            samples.append((obs.time, obs.yaw))
        if not samples or obs.time-samples[0][0] < 2.-1e-9:
            return self.brake(obs)
        if max(wrap_angle(v-intent.target_yaw) for _,v in samples)-min(wrap_angle(v-intent.target_yaw) for _,v in samples) > .10:
            self.finish('censored', 'recovery turn heading did not settle', obs)
            return self.brake(obs)
        reason = self.recovery_turn_guard(obs, req, 'drive')
        if not reason and self.corridor_guard:
            reason = self.corridor_guard(obs, req, obs.yaw)
        command, drive_reason = self._checked_drive(Motion(req.direction*req.speed, 0.), guard)
        if reason or drive_reason:
            self.finish('censored', reason or drive_reason, obs)
            return self.brake(obs)
        self.history.append(dict(event='recovery_turn_drive_admitted', time=obs.time,
            action_id=req.action_id, position=obs.xy, target_yaw=intent.target_yaw,
            measured_yaw=obs.yaw, endpoint=req.endpoint, onward_checked=False))
        self.committed_heading = obs.yaw
        self.travel_axis = (req.direction*math.cos(obs.yaw), req.direction*math.sin(obs.yaw))
        self.recovery_turn = None
        self._state(MovementState.DRIVE_FORWARD if req.direction > 0 else MovementState.DRIVE_REVERSE,
                    obs.time, 'checked translation after recovery turn; onward route unproven')
        return self._emit_traversal(command, obs)

    def start(self, request, now):
        if self.request is not None or self.result is not None:
            raise RuntimeError('Consume the previous action result before starting another')
        if request.action_id in self.seen:
            raise ValueError('Movement action IDs are single use')
        self.seen.add(request.action_id)
        self.request, self.started = request, now
        self.recovery_turn = None
        self.recovery_turn_finished = False
        self.recovery_turn_samples = []
        self.release_memory.clear(new_request=True)
        self.recovery_continuation = None
        self.recovery_execution_heading = None
        self.recovery_core_started = None
        self.recovery_core_anchor = None
        self.recovery_motion_samples = []
        self.recovery_region_samples = []
        self.recovery_region_stall_since = None
        self.last_selected_speed = None
        self.brake_episode_started = None
        self.posture = []
        self.posture_pause = None
        self.posture_signal = None
        self.entry_samples = []
        self.recovery_stage_started = None
        self.recovery_stage_anchor = None
        self.recovery_stage_samples = []
        self.alignment_best, self.alignment_progress_at = math.inf, now
        self.committed_heading, self.travel_axis = None, None
        self.heading_deviation_reported = False
        self.arrival_heading_reported = False
        self.release_origin = None
        self.uphill_patience = False
        self.drive_exposure = 0.
        self.drive_response_samples = []
        self.drive_slow_since = None
        self.last_output_time = now
        self.speed_response_used = False
        self.speed_response_started = None
        self.forward_output = False
        self.executed_traversal = None
        self.pending_traversal = None
        self.suspended_release = False
        self.uphill_samples = []
        self.ascent_intent_bound = None
        self.uphill_goal_only_budget = False
        self.uphill_stall_since = self.uphill_stall_origin = None
        self.uphill_exposure = 0.
        self.metrics.update(uphill_traversal_progress_mps=math.nan, uphill_traversal_window_s=0.,
                            uphill_budget_remaining_s=self.UPHILL_PATIENCE_SECONDS,
                            uphill_exposure_s=0., uphill_wall_remaining_s=self.UPHILL_WALL_SECONDS)
        self._state(MovementState.SETTLE if request.purpose == 'settle' else MovementState.BRAKE,
                    now, 'settle nearby endpoint' if request.purpose == 'settle' else 'brake before primitive')

    def finish(self, outcome, reason, obs, physical=False, disposition=ResultDisposition.TERMINAL):
        if self.request is None:
            return
        self._observe_emitted_traversal(obs)
        if outcome != 'success':
            self._observe_event('termination', obs, dict(outcome=outcome, reason=reason,
                                physical=physical, disposition=disposition))
        self.result = MovementResult(self.request.action_id, outcome, reason, physical, obs.time, obs.xy,
                                     self.state.value, disposition, self.executed_traversal)
        self.request = None
        self.recovery_turn = None
        self.recovery_turn_samples = []
        self.ascent_intent_bound = None
        self._state(MovementState.IDLE, obs.time, reason)

    def consume(self):
        result, self.result = self.result, None
        return result

    def stop(self, obs, reason='operator stop'):
        self.finish('censored', reason, obs)
        self._state(MovementState.STOP, obs.time, reason)
        return Motion()

    def brake(self, obs):
        longitudinal = obs.velocity[0]*math.cos(obs.yaw)+obs.velocity[1]*math.sin(obs.yaw)
        command = Motion(max(-.4, min(.4, -4.*longitudinal)), 0.)
        detail = self.brake_evaluator(obs, command) if self.brake_evaluator else None
        if detail is not None:
            self.last_brake_evaluation = dict(detail, time=obs.time)
        self.brake_rejection = (detail['reason'] if detail is not None else
                                self.brake_guard(obs) if self.brake_guard else '')
        if self.brake_rejection:
            self._observe_event('brake_rejection', obs, detail or dict(reason=self.brake_rejection))
        # This is measured inertial deceleration, never position restoration.
        # When its stopping envelope is unavailable, use teleop's zero desired
        # velocity, not an unvalidated reverse drive into a new corridor.
        return Motion() if self.brake_rejection else command

    def _observe_event(self, kind, obs, details):
        """Diagnostics cannot authorize, suppress or replace a control output."""
        if self.event_observer is None or self.request is None:
            return
        try:
            self.event_observer(kind, obs, self.request, details)
        except Exception as exc:
            self.history.append(dict(event='event_snapshot_observer_error',time=obs.time,
                action_id=self.request.action_id,error=f'{type(exc).__name__}: {exc}'))

    def _emit_traversal(self, command, obs):
        """Tag actual drive/release output; state transitions are not exposure."""
        req = self.request
        if (req is not None and self.release_departure_guard is not None
                and req.purpose == 'mission' and req.direction == 1
                and self.ascent_intent_bound is None):
            dx,dy=req.endpoint[0]-req.start[0],req.endpoint[1]-req.start[1]
            cross=abs((obs.xy[0]-req.start[0])*dy-(obs.xy[1]-req.start[1])*dx)/max(math.hypot(dx,dy),1e-9)
            if cross > req.lateral_tolerance:
                reason=self.release_departure_guard(obs,req)
                if reason:
                    self.finish('censored',reason,obs)
                    return self.brake(obs)
        if (req is not None and self.ascent_intent_bound is not None
                and math.dist(obs.xy,req.endpoint) > self.ascent_intent_bound):
            self.finish('censored','ascent departure exceeds fixed goal-distance intent bound',obs)
            return self.brake(obs)
        if self.drive_selector is not None and self.state == MovementState.DRIVE_FORWARD:
            if command.vx != self.last_selected_speed and (
                    command.vx != self.request.speed or self.last_selected_speed is not None):
                self.history.append(dict(event='forward_speed_selected',time=obs.time,
                    action_id=self.request.action_id,position=obs.xy,
                    requested_command_vx=self.request.speed,command_vx=command.vx,
                    command_omega=command.omega,actual_speed_mps=obs.speed,
                    reason='bounded freshly checked stopping demand' if command.vx < self.request.speed
                           else 'nominal demand fits observed stopping envelope again'))
            self.last_selected_speed = command.vx
        if (self.pending_drive_continuation is not None
                and self.state in (MovementState.DRIVE_FORWARD,MovementState.DRIVE_REVERSE)
                and self.request.direction*command.vx>0.):
            self.history.append(dict(event='drive_continuation_emitted',time=obs.time,
                action_id=self.request.action_id,admitted_at=self.pending_drive_continuation,
                command_vx=command.vx,command_omega=command.omega,position=obs.xy))
            self.pending_drive_continuation = None
        # Latch observed body travel heading at the first emitted phase output;
        # a planned region axis is never evidence of actual approach direction.
        axis = self.travel_axis
        if axis is not None:
            heading=math.atan2(axis[1],axis[0])
            if self.recovery_continuation is not None:
                if self.recovery_execution_heading is None:
                    self.recovery_execution_heading=wrap_angle(obs.yaw+math.pi)
                heading=self.recovery_execution_heading
            self.pending_traversal = (obs.time, obs.xy, obs.yaw, self.state.value,
                                      self.request.direction, heading)
        self.release_memory.emitted(obs,self.request,axis,self.state.value,command)
        return command

    def _checked_drive(self, command, guard):
        """Select before accounting output; admission alone credits no actuation."""
        if command.vx > 0. and self.drive_selector is not None:
            # Motion is mutable: a selector must not move its own validation
            # ceiling by editing the object it receives.
            ceiling = min(command.vx, self.request.speed if self.request else .4, .4)
            try:
                selected = self.drive_selector(Motion(command.vx, command.omega))
                vx, omega = float(selected.vx), float(selected.omega)
                valid = (math.isfinite(vx) and math.isfinite(omega)
                         and omega == 0. and 0. < vx <= ceiling)
            except Exception:
                valid = False
            if not valid:
                return command, 'invalid forward speed selection'
            command = Motion(vx, omega)
        return command, guard(command)

    def _recovery_stage(self, obs):
        """End an escape at a measured staging area, not its nominal endpoint.

        This is a bounded observation with braking only. It cannot credit a
        checkpoint, discard failed approaches, or renew the recovery episode.
        """
        req = self.request
        certificate = self.recovery_continuation
        active = self.recovery_stage_started is not None
        if certificate is not None:
            radius = certificate.outer_radius if active else certificate.inner_radius
            if math.dist(obs.xy, certificate.center) > radius:
                if active:
                    self.finish('censored','left frozen recovery staging region',obs)
                    return self.brake(obs)
                return None
        finite = all(math.isfinite(v) for v in (*obs.xy,obs.yaw,obs.up_z,*obs.velocity,
                                               obs.speed,obs.yaw_rate,obs.inclination,obs.time))
        if not finite:
            if not active:
                return None
            self.finish('censored','recovery staging observation is not finite',obs)
            return Motion()
        if not active:
            executed = self.executed_traversal
            if (req.purpose != 'recovery' or self.recovery_stage_guard is None
                    or self.state not in (MovementState.DRIVE_FORWARD, MovementState.DRIVE_REVERSE, MovementState.BRAKE)
                    or executed is None or executed.maneuver not in ('DRIVE_FORWARD', 'DRIVE_REVERSE')
                    or executed.duration_s <= 0. or executed.interval_end_time > obs.time
                    or math.dist(req.start, obs.xy) < self.RECOVERY_STAGE_MIN_DISPLACEMENT
                    or obs.body_tilt > 20. or obs.inclination > 12.
                    or self.recovery_stage_guard(obs)):
                return None
            self.recovery_stage_started = (self.recovery_core_started
                if certificate is not None and self.recovery_core_started is not None else obs.time)
            self.recovery_stage_anchor = (self.recovery_core_anchor
                if certificate is not None and self.recovery_core_anchor is not None else obs.xy)
            self.recovery_stage_samples = []
            self.history.append(dict(event='recovery_stage_entered', time=obs.time,
                action_id=req.action_id, position=obs.xy, nominal_endpoint=req.endpoint,
                displacement_m=math.dist(req.start,obs.xy), deadline=self.recovery_stage_started+self.RECOVERY_STAGE_SECONDS,
                reason='fresh gentle rotation and stopping footprint; brake for functional staging'))
        reason = self.recovery_stage_guard(obs) if self.recovery_stage_guard else 'recovery staging guard unavailable'
        if not reason and (obs.body_tilt > 20. or obs.inclination > 12.):
            reason = 'recovery staging posture or inclination changed'
        if not reason and math.dist(obs.xy,self.recovery_stage_anchor) > self.RECOVERY_STAGE_MAX_DRIFT:
            reason = 'recovery staging left bounded observation area'
        if not reason and obs.time-self.recovery_stage_started >= self.RECOVERY_STAGE_SECONDS-1e-9:
            reason = 'recovery staging did not settle within observation deadline'
        if reason:
            self.finish('censored',reason,obs)
            return self.brake(obs)
        self._state(MovementState.RECOVERY_STAGE,obs.time,'brake on checked recovery staging footprint')
        samples = self.recovery_stage_samples
        # Dwell is contiguous; repeated timestamps cannot manufacture evidence.
        stable = obs.speed < .12 and abs(obs.yaw_rate) < .12 and obs.turn_allowed
        if not stable or (samples and obs.time-samples[-1][0] > .04+1e-9):
            samples.clear()
        if stable and (not samples or obs.time > samples[-1][0]):
            samples.append((obs.time,obs.xy))
        while len(samples)>1 and samples[1][0] <= obs.time-self.RECOVERY_STAGE_DWELL_SECONDS:
            samples.pop(0)
        if samples:
            elapsed=obs.time-samples[0][0]
            drift=math.dist(obs.xy,samples[0][1])/max(elapsed,1e-9)
            if elapsed >= self.RECOVERY_STAGE_DWELL_SECONDS-1e-9 and drift < .05:
                self.history.append(dict(event='recovery_stage_settled',time=obs.time,
                    action_id=req.action_id,position=obs.xy,observation_s=obs.time-self.recovery_stage_started,
                    dwell_s=elapsed,drift_mps=drift,reason='functional recovery goal reached; mission destination still outstanding'))
                self.finish('censored','checked recovery staging reached and settled',obs,
                            disposition=ResultDisposition.REPOSITIONED)
        return self.brake(obs)

    def _observe_drive_response(self, obs):
        """A continuous measured window belonging to actual powered output.

        Positive damping, state-entry labels, and gaps cannot supply evidence
        for selecting a new maneuver. This window never changes outer budgets.
        """
        self.metrics.update(drive_response_window_s=0.,drive_response_axis_mps=None,
                            drive_response_uphill_mps=None,drive_slow_dwell_s=0.)
        pending=self.pending_traversal
        valid=(self.state == MovementState.DRIVE_FORWARD and pending is not None
               and pending[3] == MovementState.DRIVE_FORWARD.value and pending[4] == 1
               and 0. < obs.time-pending[0] <= self.DRIVE_RESPONSE_MAX_GAP_SECONDS+1e-9
               and obs.fresh and obs.terrain_gradient is not None
               and all(math.isfinite(v) for v in (*obs.xy,*pending[1],*obs.terrain_gradient,obs.time)))
        if not valid:
            self.drive_response_samples=[]
            self.drive_slow_since=None
            return
        samples=self.drive_response_samples
        if not samples or abs(samples[-1][0]-pending[0])>1e-9:
            samples[:]=[(pending[0],pending[1])]
            self.drive_slow_since=None
        samples.append((obs.time,obs.xy))
        while len(samples)>1 and samples[1][0] <= obs.time-self.DRIVE_RESPONSE_SECONDS:
            samples.pop(0)

    def _drive_release_ready(self, obs, eligible):
        samples=self.drive_response_samples
        elapsed=obs.time-samples[0][0] if samples else 0.
        gradient=obs.terrain_gradient
        norm=math.hypot(*gradient) if gradient is not None else 0.
        measured=(eligible and self.state == MovementState.DRIVE_FORWARD and self.travel_axis is not None
                  and elapsed>=self.DRIVE_RESPONSE_SECONDS-1e-9 and math.isfinite(norm) and norm>0.)
        self.metrics['drive_response_window_s']=elapsed
        if not measured:
            self.drive_slow_since=None
            return False
        displacement=tuple(obs.xy[i]-samples[0][1][i] for i in (0,1))
        axis_rate=sum(d*a for d,a in zip(displacement,self.travel_axis))/elapsed
        uphill_rate=sum(d*(g/norm) for d,g in zip(displacement,gradient))/elapsed
        deficient=axis_rate<self.DRIVE_RELEASE_PROGRESS_MPS or uphill_rate<self.DRIVE_RELEASE_PROGRESS_MPS
        self.drive_slow_since=(self.drive_slow_since if self.drive_slow_since is not None else obs.time) if deficient else None
        dwell=0. if self.drive_slow_since is None else obs.time-self.drive_slow_since
        self.metrics.update(drive_response_axis_mps=axis_rate,drive_response_uphill_mps=uphill_rate,
                            drive_slow_dwell_s=dwell)
        minimum_exposure=(self.DRIVE_RESPONSE_SECONDS+self.DRIVE_RELEASE_DWELL_SECONDS
            if self.release_memory.correction_active else self.RELEASE_DRIVE_SECONDS)
        return (self.drive_exposure>=minimum_exposure-1e-9
                and dwell>=self.DRIVE_RELEASE_DWELL_SECONDS-1e-9)

    def _speed_response_decision(self, obs, progress_failure):
        """Decide only; all common motion checks still precede zero emission."""
        if not self.uphill_speed_response_enabled:
            return None
        active = self.speed_response_started is not None
        if not active and obs.speed <= .45:
            return None
        req, axis, gradient = self.request,self.travel_axis,obs.terrain_gradient
        reason = ''
        if (req.purpose != 'mission' or req.direction != 1
                or self.state != (MovementState.RELEASE if active else MovementState.DRIVE_FORWARD)
                or axis is None or gradient is None):
            reason = 'not an executing mission-forward uphill primitive'
        elif (not obs.fresh or obs.terrain_relative_tilt is None or obs.up_z < .8
                or self.posture_pause is not None or progress_failure
                or not all(math.isfinite(v) for v in (*obs.xy,*obs.velocity,*gradient,*axis,obs.time,obs.speed,obs.yaw_rate))):
            reason = 'invalid observations, posture or progress condition'
        elif obs.speed >= .50:
            reason = 'provisional 0.50 m/s intervention boundary'
        metrics = {}
        if not reason:
            norm = math.hypot(*gradient)
            grade = math.degrees(math.atan(sum(g*a for g,a in zip(gradient,axis))))
            if norm <= 0. or not math.isfinite(norm) or grade < 10.:
                reason = 'no valid uphill terrain projection'
            else:
                axis_velocity = sum(v*a for v,a in zip(obs.velocity,axis))
                uphill_velocity = sum(v*g/norm for v,g in zip(obs.velocity,gradient))
                metrics.update(axis_velocity_mps=axis_velocity,uphill_velocity_mps=uphill_velocity,
                               actual_speed_mps=obs.speed,directional_grade_deg=grade)
                if min(axis_velocity,uphill_velocity) <= 0.:
                    reason = 'motion is not positive along both ascent projections'
        if not reason and not active:
            samples,execution = self.drive_response_samples,self.executed_traversal
            elapsed = obs.time-samples[0][0] if samples else 0.
            if (elapsed < 2.-1e-9 or not samples or abs(samples[-1][0]-obs.time)>1e-9
                    or execution is None or execution.maneuver != MovementState.DRIVE_FORWARD.value
                    or execution.drive_sign != 1 or execution.duration_s <= 0.
                    or execution.interval_end_time != obs.time):
                reason = 'complete continuous actual-forward response unavailable'
            else:
                displacement = [p-old for p,old in zip(obs.xy,samples[0][1])]
                axis_progress = sum(d*a for d,a in zip(displacement,axis))/elapsed
                uphill_progress = sum(d*g/norm for d,g in zip(displacement,gradient))/elapsed
                metrics.update(response_window_s=elapsed,axis_progress_mps=axis_progress,uphill_progress_mps=uphill_progress)
                if min(axis_progress,uphill_progress) < .02:
                    reason = 'recent forward response is not productive on both ascent projections'
        if not reason:
            reason = self.release_guard(obs,req) if self.release_guard else 'release envelope unavailable'
        if not reason and active and obs.time-self.speed_response_started >= .20-1e-9 and obs.speed > .45:
            reason = 'speed did not decay within fixed 0.20 s response'
        if reason:
            return dict(action='cancel',reason=reason,**metrics) if active else None
        return dict(action=('decayed' if obs.speed <= .45 else 'observe') if active else 'start',**metrics)

    def _emit_speed_response(self, decision, obs, guard):
        """Called only after common arrival, tracking and progress handling."""
        if decision is None:
            return None
        action = decision['action']
        if action == 'cancel':
            self._state(MovementState.BRAKE,obs.time,'uphill zero-response cancelled: '+decision['reason'])
            return self.brake(obs)
        if action == 'decayed':
            self.history.append(dict(event='uphill_speed_response_ended',time=obs.time,
                action_id=self.request.action_id,elapsed_s=obs.time-self.speed_response_started,
                reason='speed below original cutoff; independently check interrupted drive',
                decayed=True,**decision))
            self.speed_response_started = None
            req = self.request
            desired = (self.committed_heading if self.committed_heading is not None else
                       math.atan2(req.endpoint[1]-req.start[1], req.endpoint[0]-req.start[0]))
            if obs.turn_allowed:
                desired = math.atan2(req.endpoint[1]-obs.xy[1], req.endpoint[0]-obs.xy[0])
            error = wrap_angle(desired-obs.yaw)
            if obs.turn_allowed and not self._heading_acceptable(obs, req, error, .16):
                self._state(MovementState.BRAKE, obs.time, 'speed response ended; settle before gentle heading correction')
                return self.brake(obs)
            dx, dy = req.endpoint[0]-req.start[0], req.endpoint[1]-req.start[1]
            cross = abs((obs.xy[0]-req.start[0])*dy-(obs.xy[1]-req.start[1])*dx)/max(math.hypot(dx, dy), 1e-9)
            reason = ('speed response ended outside original powered corridor'
                      if cross > req.lateral_tolerance and self.release_departure_guard is None else
                      'command heading no longer advances the committed corridor' if math.cos(error) <= 0. else '')
            if not reason and self.corridor_guard:
                reason = self.corridor_guard(obs, req, obs.yaw)
            command = Motion(req.speed, 0.)
            if not reason:
                command, reason = self._checked_drive(command, guard)
            if reason:
                self.finish('censored', reason, obs)
                return self.brake(obs)
            self._state(MovementState.DRIVE_FORWARD, obs.time, 'speed settled; restore checked powered climb')
            self.history.append(dict(event='uphill_speed_response_drive_restored', time=obs.time,
                action_id=req.action_id, position=obs.xy, command_vx=command.vx,
                command_omega=command.omega, powered_exposure_s=self.drive_exposure))
            self.forward_output = True
            return self._emit_traversal(command, obs)
        if action == 'start':
            exposure = self.drive_exposure
            self.release_origin = obs.xy
            self._state(MovementState.RELEASE,obs.time,'checked uphill speed response: policy-active zero command')
            # This interrupts DRIVE; it is not a deliberate slowdown release.
            # The actual-powered response samples still reset on state change.
            self.drive_exposure = exposure
            self.metrics['forward_exposure_s'] = exposure
            self.speed_response_used = True
            self.speed_response_started = obs.time
            self.history.append(dict(event='uphill_speed_response_started',time=obs.time,
                action_id=self.request.action_id,position=obs.xy,response_deadline=obs.time+.20,
                intervention_mps=.50,command_vx=0.,command_omega=0.,**decision))
        self.reason = 'checked uphill speed response: policy-active zero command'
        return self._emit_traversal(Motion(),obs)

    def _observe_emitted_traversal(self, obs):
        pending, self.pending_traversal = self.pending_traversal, None
        if pending is None or obs.time <= pending[0]:
            return
        now, xy, yaw, maneuver, sign, heading = pending
        # Attribute the held output, never the newly entered state or the sign
        # of a damping command. A consumed pending interval cannot count twice.
        if (self.uphill_stall_since is not None and sign == 1
                and maneuver in (MovementState.DRIVE_FORWARD.value, MovementState.RELEASE.value)):
            self.uphill_exposure += max(0., obs.time-max(now, self.uphill_stall_since))
        previous = self.executed_traversal
        self.executed_traversal = ExecutedTraversal(maneuver, sign, heading, yaw,
            previous.start_position if previous else xy, obs.xy,
            previous.first_emitted_time if previous else now, obs.time,
            (previous.duration_s if previous else 0.)+obs.time-now)

    def _uphill_eligible(self, obs):
        gradient = obs.terrain_gradient
        return (self.request.direction == 1 and self.request.purpose == 'mission'
                and self.release_guard is not None and self.travel_axis is not None
                and gradient is not None and all(math.isfinite(v) for v in gradient)
                and sum(a*g for a, g in zip(self.travel_axis, gradient)) >= math.tan(math.radians(10.)))

    def _slope_posture_exempt(self, obs, relative_tilt):
        """Provisional envelope for an already executed uphill forward intent."""
        execution=self.executed_traversal
        axis=self.travel_axis
        return bool(self.posture_pause is None
            and self.state in (MovementState.DRIVE_FORWARD,MovementState.RELEASE)
            and self._uphill_eligible(obs) and obs.fresh
            and relative_tilt is not None and relative_tilt<=self.POSTURE_RELATIVE_ENVELOPE_DEG+1e-9
            and obs.body_tilt<self.POSTURE_ABSOLUTE_PAUSE_DEG-1e-9
            and self.committed_heading is not None and execution is not None
            and execution.drive_sign==1 and execution.maneuver in ('DRIVE_FORWARD','RELEASE')
            and execution.duration_s>0. and execution.first_emitted_time<execution.interval_end_time<=obs.time
            and abs(wrap_angle(execution.heading-math.atan2(axis[1],axis[0])))<1e-9
            and abs(wrap_angle(self.committed_heading-math.atan2(axis[1],axis[0])))<1e-9)

    def _observe_uphill_progress(self, obs):
        """Bound one low-progress episode by deliberate exposure and wall time.

        The short RELEASE response window may restart after measured braking;
        this observer cannot. Useful physical progress, not command transitions,
        renews both allowances. Braking/posture consumes wall time only.
        """
        self.uphill_patience |= self._uphill_eligible(obs)
        if (not self.uphill_patience or self.travel_axis is None or (self.uphill_stall_since is None and self.state not in
                (MovementState.DRIVE_FORWARD, MovementState.DRIVE_REVERSE,
                 MovementState.RELEASE, MovementState.BRAKE, MovementState.POSTURE_PAUSE))):
            return ''
        self.uphill_samples.append((obs.time, obs.xy))
        while len(self.uphill_samples) > 1 and self.uphill_samples[1][0] <= obs.time-2.:
            self.uphill_samples.pop(0)
        elapsed = obs.time-self.uphill_samples[0][0]
        # Reproject world-space displacements in one basis. Gentle brake
        # admission can refine the travel axis without actual realignment;
        # mixing old/new absolute projections fabricates position jumps.
        advance = lambda xy: sum((p-old)*d for p, old, d in zip(obs.xy, xy, self.travel_axis))
        progress = advance(self.uphill_samples[0][1])/max(elapsed, 1e-6)
        shared_goal = (self.release_departure_guard is not None
                       and self.request.purpose == 'mission' and self.request.direction == 1)
        goal_advance = lambda xy: math.dist(xy,self.request.endpoint)-math.dist(obs.xy,self.request.endpoint)
        goal_progress = goal_advance(self.uphill_samples[0][1])/max(elapsed,1e-6)
        deficient = progress < .02 or (shared_goal and goal_progress < .02)
        if elapsed >= 2.-1e-6 and deficient and self.uphill_stall_since is None:
            self.uphill_stall_since, self.uphill_stall_origin = obs.time, obs.xy
            self.uphill_exposure = 0.
            self.uphill_goal_only_budget = shared_goal and progress >= .02 and goal_progress < .02
            self.history.append(dict(event='uphill_budget_started',time=obs.time,
                action_id=self.request.action_id,origin=obs.xy,axis=self.travel_axis,
                goal_only=self.uphill_goal_only_budget,goal_progress_mps=goal_progress,
                exposure_limit_s=self.UPHILL_PATIENCE_SECONDS,wall_limit_s=self.UPHILL_WALL_SECONDS))
        if (shared_goal and self.uphill_stall_origin is not None
                and advance(self.uphill_stall_origin) >= self.UPHILL_RENEWAL_METERS
                and goal_advance(self.uphill_stall_origin) < self.UPHILL_RENEWAL_METERS):
            # The original axis-only budget would renew here. Remaining delay
            # belongs to goal geometry and must not manufacture terrain failure.
            self.uphill_goal_only_budget = True
        if (self.uphill_stall_origin is not None and advance(self.uphill_stall_origin) >= self.UPHILL_RENEWAL_METERS
                and (not shared_goal or goal_advance(self.uphill_stall_origin) >= self.UPHILL_RENEWAL_METERS)):
            self.history.append(dict(event='uphill_budget_renewed',time=obs.time,
                action_id=self.request.action_id,origin=self.uphill_stall_origin,
                net_advance_m=advance(self.uphill_stall_origin),exposure_s=self.uphill_exposure,
                elapsed_s=obs.time-self.uphill_stall_since))
            self.uphill_stall_since = self.uphill_stall_origin = None
            self.uphill_goal_only_budget = False
            self.uphill_exposure = 0.
        self.metrics.update(uphill_goal_progress_mps=goal_progress,
            uphill_goal_only_stall=shared_goal and progress >= .02 and goal_progress < .02,
            uphill_traversal_progress_mps=progress, uphill_traversal_window_s=elapsed,
            uphill_exposure_s=self.uphill_exposure,
            uphill_budget_remaining_s=max(0.,self.UPHILL_PATIENCE_SECONDS-self.uphill_exposure),
            uphill_wall_remaining_s=(self.UPHILL_WALL_SECONDS if self.uphill_stall_since is None else
                                    max(0.,self.UPHILL_WALL_SECONDS-(obs.time-self.uphill_stall_since))))
        if elapsed >= 1.-1e-6 and progress < -.10:
            return 'measured backslide against committed uphill direction'
        if self.uphill_stall_since is not None:
            if self.uphill_exposure >= self.UPHILL_PATIENCE_SECONDS-1e-9:
                return 'uphill attempt exhausted shared low-progress exposure budget'
            if obs.time-self.uphill_stall_since >= self.UPHILL_WALL_SECONDS-1e-9:
                return 'uphill attempt exhausted shared low-progress wall budget'
        return ''

    def _finish_progress_failure(self, reason, obs):
        physical = not ('shared low-progress' in reason and
                        (self.uphill_exposure <= 0. or self.uphill_goal_only_budget))
        if 'shared low-progress' in reason:
            self.history.append(dict(event='uphill_budget_exhausted',time=obs.time,
                action_id=self.request.action_id,started=self.uphill_stall_since,
                exposure_s=self.uphill_exposure,wall_s=obs.time-self.uphill_stall_since,
                reason=reason,physical=physical))
        self.finish('failure' if physical else 'censored',reason,obs,physical=physical)

    def _posture_response(self, obs, req, supported, error, rapid, progress_failure):
        pause = self.posture_pause
        reason = self.posture_guard(obs, req) if self.posture_guard else 'posture observation envelope unavailable'
        if reason:
            self.finish('censored', reason, obs)
            return Motion()
        if obs.time-pause.started >= self.POSTURE_PAUSE_SECONDS:
            self.finish('failure' if rapid else 'censored',
                        'continuing posture deterioration during pause' if rapid else 'posture observation unresolved',
                        obs, physical=rapid)
            return Motion()
        brake_settled = (not pause.braking_required or (pause.motion_stable_since is not None
                         and obs.time-pause.motion_stable_since >= .3-1e-9))
        settled = (pause.stable_since is not None and
                   obs.time-pause.stable_since >= self.POSTURE_RECOVERY_SECONDS-1e-9 and brake_settled)
        if settled:
            if self.recovery_turn is None and self.recovery_continuation is None and math.dist(obs.xy, req.endpoint) <= req.arrival_radius and supported(obs.xy):
                self._state(MovementState.SETTLE, obs.time, 'posture recovered at supported endpoint')
                return self.brake(obs)
        if progress_failure:
            self._finish_progress_failure(progress_failure, obs)
            return self.brake(obs)
        if settled:
            if pause.intent == MovementState.RELEASE:
                reason = ''
                if not self._uphill_eligible(obs):
                    reason = 'posture recovery lacks fresh uphill release support'
                elif math.cos(error) <= 0. or (obs.turn_allowed and abs(error) > .16):
                    reason = 'posture recovery needs a new checked heading'
                elif self.corridor_guard:
                    reason = self.corridor_guard(obs, req, obs.yaw)
                if not reason:
                    reason = self.release_guard(obs, req)
                if reason:
                    self.finish('censored', reason, obs)
                    return Motion()
                self.posture_pause = None
                self.release_origin = obs.xy
                self._state(MovementState.RELEASE, obs.time, 'posture recovered; restore checked release')
                return self._emit_traversal(Motion(), obs)
            # Re-admit drive or alignment through measured braking and all
            # existing command guards. No stored yaw output is ever resumed.
            self.posture_pause = None
            self._state(MovementState.BRAKE, obs.time, 'posture recovered; re-admit interrupted primitive')
            return self.brake(obs)
        if not brake_settled:
            self._state(MovementState.BRAKE, obs.time, 'brake before posture resumption')
            return self.brake(obs)
        self._state(MovementState.POSTURE_PAUSE, obs.time, 'observe policy-active zero command for posture recovery')
        return Motion()

    def _transfer_recovered_posture_to_brake(self, obs):
        """Resolved posture must not masquerade as an unresolved brake timer.

        This transfers ownership only. It cannot release or drive, and does
        not renew the request, braking, or shared uphill progress deadlines.
        Recovery first observed at the posture deadline is still too late.
        """
        pause = self.posture_pause
        if (pause is None or not pause.braking_required or pause.stable_since is None
                or obs.time-pause.started >= self.POSTURE_PAUSE_SECONDS-1e-9
                or obs.time-pause.stable_since < self.POSTURE_RECOVERY_SECONDS-1e-9):
            return False
        self.suspended_release = pause.intent == MovementState.RELEASE
        self.posture_pause = None
        self._state(MovementState.BRAKE, obs.time,
                    'posture recovered; continue original bounded braking episode')
        return True

    def _observe_entry(self, obs, req):
        if self.entry_samples and self.entry_samples[-1][0] == obs.time:
            return
        longitudinal = obs.velocity[0]*math.cos(obs.yaw)+obs.velocity[1]*math.sin(obs.yaw)
        self.entry_samples.append((obs.time, obs.yaw, obs.speed, req.direction*longitudinal))
        while len(self.entry_samples) > 1 and self.entry_samples[1][0] <= obs.time-.3:
            self.entry_samples.pop(0)

    def _heading_acceptable(self, obs, req, error, angular_tolerance):
        """Keep useful gentle-ground headings that reach the unchanged target.

        Reserve the outer quarter of the arrival disk for tracking variation.
        This only avoids an unnecessary turn; all motion/settling guards and
        the immutable target, direction and deadline still apply. Steep motion
        retains its committed-axis contract rather than adopting a new bearing.
        """
        if abs(error) <= angular_tolerance:
            return True
        acceptable = (obs.turn_allowed and obs.inclination <= 15.
                and arrival_ray_interval(obs.xy, req.endpoint, obs.yaw, req.direction,
                                         req.arrival_radius*self.ARRIVAL_HEADING_RESERVE) is not None)
        if acceptable and not self.arrival_heading_reported:
            self.history.append(dict(event='arrival_heading_accepted', time=obs.time,
                action_id=req.action_id, position=obs.xy, endpoint=req.endpoint,
                body_yaw=obs.yaw, direction=req.direction, heading_error_rad=error,
                inner_radius_m=req.arrival_radius*self.ARRIVAL_HEADING_RESERVE,
                reason='straight ray reaches existing arrival region; motion guards still required'))
            self.arrival_heading_reported = True
        return acceptable

    def _straight_motion_window(self, obs):
        """Shared measured motion envelope; grants no heading or command right."""
        if not self.entry_samples:
            return None
        elapsed = obs.time-self.entry_samples[0][0]
        if (not all(math.isfinite(v) for sample in self.entry_samples for v in sample)
                or abs(self.entry_samples[-1][0]-obs.time)>1e-9
                or elapsed < .3-1e-9 or elapsed > .34+1e-9
                or any(speed > .25 or directed < -.10 for _, _, speed, directed in self.entry_samples)
                or any(not -1e-9 <= b[0]-a[0] <= .04+1e-9 for a,b in zip(self.entry_samples,self.entry_samples[1:]))):
            return None
        angles = [wrap_angle(yaw-obs.yaw) for _, yaw, _, _ in self.entry_samples]
        if max(angles)-min(angles) > .10:
            return None
        trend = wrap_angle(obs.yaw-self.entry_samples[0][1])/max(elapsed, 1e-9)
        return trend,(min(angles),max(angles))

    def _straight_admission(self, obs, req, error):
        """Separate fresh heading alignment from an executed steep continuation."""
        self.metrics['straight_entry_continuation']=False
        aligned=self._heading_acceptable(obs,req,error,.10)
        continuation=self._committed_drive_continuation(obs,req,error)
        if self.straight_entry_guard is None or not (aligned or continuation):
            return False,''
        window=self._straight_motion_window(obs)
        if window is None:return False,''
        trend,offsets=window
        reason = self.straight_entry_guard(obs, req, trend, offsets)
        self.metrics.update(straight_entry_heading_excursion_rad=offsets[1]-offsets[0],
                            straight_entry_yaw_trend_rps=trend, straight_entry_rejection=reason,
                            straight_entry_continuation=continuation and not aligned and not reason)
        return not reason, reason

    def _committed_drive_continuation(self, obs, req, error):
        """Only observed DRIVE in this request can establish steep continuation.

        Start and actual alignment clear execution evidence. Damping and
        RELEASE-only evidence cannot bypass a fresh drive's heading contract.
        """
        executed=self.executed_traversal
        if (req is not self.request or self.suspended_release or self.posture_pause is not None
                or not obs.fresh or not obs.inclination>15. or not obs.up_z>=.8
                or not math.cos(error)>0. or self.committed_heading is None
                or self.travel_axis is None or executed is None):
            return False
        expected=MovementState.DRIVE_FORWARD.value if req.direction>0 else MovementState.DRIVE_REVERSE.value
        axis=math.atan2(self.travel_axis[1],self.travel_axis[0])
        return (executed.maneuver==expected and executed.drive_sign==req.direction
                and executed.duration_s>0. and executed.interval_end_time>executed.first_emitted_time
                and executed.interval_end_time<=obs.time
                and abs(wrap_angle(executed.heading-axis))<1e-6
                and abs(wrap_angle(axis-self.committed_heading-(math.pi if req.direction<0 else 0.)))<1e-6)

    def _moving_release_admission(self, obs, req, error):
        """Restore an interrupted uphill RELEASE from bounded useful motion.

        This does not admit a turn or a new drive, destination, or approach.
        Full geometry is checked here and by the existing restoration branch.
        Missing motion evidence keeps the original bounded brake active.
        """
        if (not self.suspended_release or self.posture_pause is not None
                or not all(math.isfinite(v) for v in (obs.time,obs.up_z,obs.yaw,obs.yaw_rate,*obs.xy,*obs.velocity))
                or not obs.fresh or obs.up_z < .8 or not self._uphill_eligible(obs)
                or math.cos(error)<=0. or (obs.turn_allowed and abs(error)>.16)
                or self.straight_entry_guard is None):
            return False,''
        window=self._straight_motion_window(obs)
        samples=self.uphill_samples
        if window is None or not samples:return False,''
        elapsed=obs.time-samples[0][0]
        if (elapsed<2.-1e-9 or elapsed>2.04+1e-9 or abs(samples[-1][0]-obs.time)>1e-9
                or not all(math.isfinite(v) for t,xy in samples for v in (t,*xy))
                or any(not -1e-9 <= b[0]-a[0] <= .04+1e-9 for a,b in zip(samples,samples[1:]))):
            return False,''
        displacement=tuple(obs.xy[i]-samples[0][1][i] for i in (0,1))
        axis_progress=sum(d*a for d,a in zip(displacement,self.travel_axis))/elapsed
        gradient=obs.terrain_gradient
        uphill_progress=sum(d*g for d,g in zip(displacement,gradient))/(math.hypot(*gradient)*elapsed)
        self.metrics.update(release_entry_axis_progress_mps=axis_progress,
                            release_entry_uphill_progress_mps=uphill_progress)
        if axis_progress<.02 or uphill_progress<.02:return False,''
        trend,offsets=window
        reason=self.straight_entry_guard(obs,req,trend,offsets)
        self.metrics.update(release_entry_heading_excursion_rad=offsets[1]-offsets[0],
                            release_entry_yaw_trend_rps=trend,release_entry_rejection=reason)
        return not reason,reason

    def _observe_recovery_motion(self, obs):
        if self.recovery_continuation_provider is None:return
        pending=self.pending_traversal
        valid=(self.request.purpose=='recovery' and self.request.direction==-1
               and self.state==MovementState.DRIVE_REVERSE and pending is not None
               and pending[3]==MovementState.DRIVE_REVERSE.value and pending[4]==-1
               and 0.<obs.time-pending[0]<=.04+1e-9)
        if not valid:
            if not self.recovery_motion_samples or self.recovery_motion_samples[-1][0]!=obs.time:
                self.recovery_motion_samples=[]
            return
        if not self.recovery_motion_samples or self.recovery_motion_samples[-1][0]<obs.time:
            self.recovery_motion_samples.append((obs.time,obs.xy,obs.yaw))
        while len(self.recovery_motion_samples)>1 and self.recovery_motion_samples[1][0]<=obs.time-2.:
            self.recovery_motion_samples.pop(0)

    def _recovery_physical_progress(self, obs, monitor_stall=True):
        """Original-axis stall/backslide precedes a recovery geometry handoff."""
        if self.travel_axis is None:return 'recovery lost original travel axis'
        position=sum(a*b for a,b in zip(obs.xy,self.travel_axis))
        if not self.samples or self.samples[-1][0]<obs.time:
            self.samples.append((obs.time,position,obs.xy))
        while len(self.samples)>1 and self.samples[1][0]<=obs.time-2.:
            self.samples.pop(0)
        elapsed=obs.time-self.samples[0][0]
        progress=(position-self.samples[0][1])/max(elapsed,1e-9)
        self.metrics.update(progress_mps=progress,window_s=elapsed)
        if elapsed>=1.-1e-6 and progress<-.10:return 'measured backslide against committed direction'
        # Staging intentionally stops translation. Keep its original six-second
        # settling deadline and backslide checks, without creating a drive stall.
        if not monitor_stall:return ''
        if elapsed>=2.-1e-6 and progress<.02:
            if self.stall_since is None:self.stall_since,self.stall_origin=obs.time,position
            if obs.time-self.stall_since>=3.:return 'sustained measured stall or backslide'
        elif self.stall_origin is None or position-self.stall_origin>=.10:
            self.stall_since=self.stall_origin=None
        return ''

    def _admit_recovery_continuation(self, obs, guard):
        req=self.request
        if (self.recovery_continuation_provider is None or self.recovery_continuation is not None
                or req.purpose!='recovery' or req.direction!=-1
                or self.state!=MovementState.DRIVE_REVERSE or self.posture_pause is not None
                or self.executed_traversal is None or self.executed_traversal.duration_s<=0.
                or obs.speed>.45 or obs.up_z<.8 or not obs.fresh
                or self.metrics.get('window_s',0.)<2.-1e-9
                or self.metrics.get('progress_mps',0.)<.02
                or guard(Motion(-req.speed,0.))):return False
        certificate=self.recovery_continuation_provider(obs,req,tuple(self.recovery_motion_samples))
        if certificate is None:return False
        if (certificate.action_id!=req.action_id or certificate.deadline!=req.deadline
                or certificate.issued!=obs.time or certificate.origin!=tuple(obs.xy)
                or self.recovery_continuation_guard is None
                or self.recovery_continuation_guard(obs,certificate,False)):
            return False
        if self.recovery_continuation_admitted is None:return False
        # The old held interval was already observed. Evidence is split before
        # the new phase can emit; no physical progress clock/sample is cleared.
        self.recovery_continuation_admitted(obs,certificate,self.executed_traversal)
        self.recovery_continuation=certificate
        self.recovery_execution_heading=None
        self.executed_traversal=self.pending_traversal=None
        self.recovery_region_samples=[(obs.time,math.dist(obs.xy,certificate.center))]
        self.history.append(dict(event='recovery_continuation_admitted',time=obs.time,
            action_id=req.action_id,certificate_id=certificate.certificate_id,
            origin=certificate.origin,center=certificate.center,outer_radius=certificate.outer_radius,
            inner_radius=certificate.inner_radius,deadline=req.deadline,original_axis=self.travel_axis,
            reason='one fixed surveyed staging region; original progress and recovery budget retained'))
        return True

    def _continue_recovery(self, obs, guard):
        c=self.recovery_continuation
        reason=self.recovery_continuation_guard(obs,c,self.recovery_core_started is not None)
        if reason:
            self.finish('censored',reason,obs)
            return self.brake(obs)
        distance=math.dist(obs.xy,c.center)
        if self.recovery_core_started is None and distance<=c.inner_radius:
            self.recovery_core_started=obs.time
            self.recovery_core_anchor=obs.xy
            self.history.append(dict(event='recovery_core_entered',time=obs.time,
                action_id=self.request.action_id,certificate_id=c.certificate_id,
                position=obs.xy,deadline=obs.time+self.RECOVERY_STAGE_SECONDS,
                reason='fixed core observation owns braking; no further locomotion demand'))
        if (self.recovery_core_anchor is not None
                and math.dist(obs.xy,self.recovery_core_anchor)>self.RECOVERY_STAGE_MAX_DRIFT):
            self.finish('censored','recovery core left bounded observation area',obs)
            return self.brake(obs)
        samples=self.recovery_region_samples
        if not samples or obs.time>samples[-1][0]:samples.append((obs.time,distance))
        while len(samples)>1 and samples[1][0]<=obs.time-2.:samples.pop(0)
        elapsed=obs.time-samples[0][0]
        progress=(samples[0][1]-distance)/max(elapsed,1e-9)
        self.metrics.update(recovery_region_distance_m=distance,recovery_region_progress_mps=progress)
        if self.recovery_core_started is None and elapsed>=2.-1e-9:
            if progress<.02:
                if self.recovery_region_stall_since is None:self.recovery_region_stall_since=obs.time
                if progress<-.10 or obs.time-self.recovery_region_stall_since>=3.:
                    self.finish('censored','no sustained progress toward frozen recovery region',obs)
                    return self.brake(obs)
            else:self.recovery_region_stall_since=None
        stage=self._recovery_stage(obs)
        if stage is not None:return stage
        if self.recovery_core_started is not None:
            self._state(MovementState.BRAKE,obs.time,'observe and settle inside fixed recovery entry core')
            return self.brake(obs)
        if self.state==MovementState.BRAKE:
            self._observe_entry(obs,self.request)
            window=self._straight_motion_window(obs)
            if window is None or obs.speed>=.25 or abs(obs.yaw_rate)>=.10:
                return self.brake(obs)
            if self.straight_entry_guard is None:
                self.finish('censored','recovery resumption requires checked straight entry',obs)
                return self.brake(obs)
            reason=self.straight_entry_guard(obs,self.request,*window)
            if reason:
                self.finish('censored',reason,obs)
                return self.brake(obs)
            self._state(MovementState.DRIVE_REVERSE,obs.time,'resume same checked recovery certificate')
        if self.state!=MovementState.DRIVE_REVERSE:
            self.finish('censored','recovery certificate lost straight reverse ownership',obs)
            return self.brake(obs)
        command=Motion(-self.request.speed,0.)
        if self.committed_heading is None or math.cos(wrap_angle(self.committed_heading-obs.yaw))<=0.:
            self.finish('censored','command heading no longer advances the committed corridor',obs)
            return self.brake(obs)
        reason=guard(command)
        if reason:
            self.finish('censored',reason,obs)
            return self.brake(obs)
        self.reason='committed reverse toward frozen recovery staging region'
        return self._emit_traversal(command,obs)

    def tick(self, obs, guard, supported):
        req = self.request
        if req is None:
            return Motion()
        if ((self.recovery_continuation_provider is not None or self.recovery_continuation is not None)
                and req.purpose=='recovery'
                and not all(math.isfinite(v) for v in (*obs.xy,*obs.velocity,obs.yaw,
                    obs.time,obs.yaw_rate,obs.up_z,obs.inclination,
                    *(obs.body_quaternion if obs.body_quaternion is not None else ())))):
            # Missing observations cannot establish a terrain failure or an
            # executed displacement, and cannot be passed to velocity damping.
            self.pending_traversal=None
            self.finish('censored','invalid recovery observation; zero desired velocity',obs)
            return Motion()
        self._observe_recovery_motion(obs)
        self.release_memory.observe(obs,req,self.travel_axis)
        self._observe_drive_response(obs)
        self._observe_emitted_traversal(obs)
        # Attribute elapsed simulation time to the previous *emitted* intent.
        # BRAKE may itself emit positive velocity, and BRAKE->DRIVE changes state
        # before returning a damping output. Neither is deliberate forward drive.
        if self.last_output_time is not None and self.forward_output:
            self.drive_exposure += max(0., obs.time-self.last_output_time)
        self.last_output_time = obs.time
        self.forward_output = False
        self.metrics['forward_exposure_s'] = self.drive_exposure
        self.metrics.update(release_window_complete=None,release_plateau_s=None)
        if obs.time > req.deadline:
            self.finish('failure', 'primitive deadline elapsed', obs, physical=self.state in
                        (MovementState.DRIVE_FORWARD, MovementState.DRIVE_REVERSE, MovementState.RELEASE, MovementState.SETTLE))
            return self.brake(obs)
        dx, dy = req.endpoint[0]-req.start[0], req.endpoint[1]-req.start[1]
        length = math.hypot(dx, dy)
        ux, uy = dx/max(length, 1e-9), dy/max(length, 1e-9)
        along = (obs.xy[0]-req.start[0])*ux+(obs.xy[1]-req.start[1])*uy
        cross = abs((obs.xy[0]-req.start[0])*uy-(obs.xy[1]-req.start[1])*ux)
        desired = math.atan2(dy, dx)+(math.pi if req.direction < 0 else 0.) if length > 1e-8 else obs.yaw
        if self.committed_heading is not None:
            desired = self.committed_heading
        if obs.turn_allowed and math.dist(obs.xy, req.endpoint) > .05:
            # Gentle-ground alignment uses the actual pose and fixed endpoint.
            # Restoring the original heading cannot remove lateral drift.
            desired = math.atan2(req.endpoint[1]-obs.xy[1], req.endpoint[0]-obs.xy[0])+(math.pi if req.direction < 0 else 0.)
        error = wrap_angle(desired-obs.yaw)
        tilt = math.acos(max(-1., min(1., obs.up_z)))
        self.posture.append((obs.time, tilt))
        while len(self.posture) > 1 and self.posture[1][0] <= obs.time-.6:
            self.posture.pop(0)
        rapid = (obs.time-self.posture[0][0] >= .4 and obs.up_z < .8
                 and tilt-self.posture[0][1] > math.radians(8.))
        relative_tilt=obs.terrain_relative_tilt
        absolute_pause=obs.body_tilt>=self.POSTURE_ABSOLUTE_PAUSE_DEG-1e-9
        exempt=rapid and self._slope_posture_exempt(obs,relative_tilt)
        signal=(rapid,exempt,absolute_pause)
        self.metrics.update(posture_raw_rapid=rapid,posture_entry_exempt=exempt,
                            posture_absolute_pause=absolute_pause,
                            posture_signal_time=obs.time,
                            terrain_relative_tilt_deg=relative_tilt if relative_tilt is not None else math.nan)
        if signal!=self.posture_signal and (any(signal) or self.posture_signal is not None):
            self.history.append(dict(event='posture_entry_signal',time=obs.time,
                action_id=req.action_id,state=self.state.value,raw_rapid=rapid,exempt=exempt,
                absolute_pause=absolute_pause,body_tilt_deg=obs.body_tilt,
                terrain_relative_tilt_deg=relative_tilt,position=obs.xy,
                reason='provisional observed slope posture envelope' if exempt else
                       'absolute pause boundary' if absolute_pause else 'original rapid-change trigger' if rapid else 'signal cleared'))
        self.posture_signal=signal
        if obs.up_z < .5:
            self.finish('failure', 'excessive body tilt', obs, physical=True)
            return self.brake(obs)
        if not obs.fresh:
            self.finish('censored', 'local observations stale; refresh before drive', obs)
            return self.brake(obs)
        progress_failure = self._observe_uphill_progress(obs)
        if self.recovery_continuation is not None:
            progress_failure = progress_failure or self._recovery_physical_progress(obs,
                monitor_stall=self.recovery_core_started is None)
            if (self.recovery_core_started is not None
                    and obs.time-self.recovery_core_started>=self.RECOVERY_STAGE_SECONDS-1e-9):
                if progress_failure:self._finish_progress_failure(progress_failure,obs)
                else:self.finish('censored','recovery core observation deadline elapsed before settling',obs)
                return self.brake(obs)
        if self.recovery_stage_started is not None:
            if progress_failure:
                self._finish_progress_failure(progress_failure,obs)
                return self.brake(obs)
            if obs.time-self.recovery_stage_started >= self.RECOVERY_STAGE_SECONDS-1e-9:
                self.finish('censored','recovery staging did not settle within observation deadline',obs)
                return self.brake(obs)
            if self.recovery_continuation is None and cross > req.lateral_tolerance:
                self.finish('censored','recovery staging left original lateral corridor',obs)
                return self.brake(obs)
            # Stage eligibility is handled only after posture/overspeed have
            # owned their response. A >20 degree stage limit must not censor
            # a simultaneous rapid/50 degree protective pause.
            if (obs.speed >= .12 or abs(obs.yaw_rate) >= .12 or not obs.turn_allowed
                    or rapid or absolute_pause or self.posture_pause is not None):
                self.recovery_stage_samples.clear()
        if self.state == MovementState.BRAKE:
            self._observe_entry(obs, req)
        if (absolute_pause or (rapid and not exempt)) and self.posture_pause is None:
            intent = MovementState.RELEASE if self.suspended_release else self.state
            self.posture_pause = PosturePause(obs.time, intent, braking_required=self.state == MovementState.BRAKE)
            self._state(MovementState.POSTURE_PAUSE, obs.time,
                'absolute posture pause boundary; observe before judging approach' if absolute_pause else
                'rapid posture change; observe before judging approach')
        if self.posture_pause is not None:
            pause = self.posture_pause
            pause.stable_since = (pause.stable_since if pause.stable_since is not None else obs.time) if obs.up_z >= .8 else None
            pause.braking_required |= obs.speed > .45
            threshold = .25 if obs.inclination > 15. and abs(error) <= .10 else .10
            stopped = obs.speed < threshold and abs(obs.yaw_rate) < .10
            pause.motion_stable_since = (pause.motion_stable_since if pause.motion_stable_since is not None else obs.time) if stopped else None
        transferred = self._transfer_recovered_posture_to_brake(obs)
        if (self.brake_episode_started is not None
                and obs.time-self.brake_episode_started > 8.):
            if progress_failure:
                self._finish_progress_failure(progress_failure, obs)
            else:
                self.finish('failure','unable to brake and settle before primitive',obs,physical=True)
            return self.brake(obs)
        if transferred:
            if progress_failure:
                self._finish_progress_failure(progress_failure, obs)
            return self.brake(obs)
        speed_response = self._speed_response_decision(obs,progress_failure)
        speed_response_allows_zero = speed_response is not None and speed_response['action'] in ('start','observe')
        if obs.speed > .45 and not speed_response_allows_zero:
            if progress_failure:
                self._finish_progress_failure(progress_failure, obs)
                return self.brake(obs)
            if self.posture_pause is not None and obs.time-self.posture_pause.started >= self.POSTURE_PAUSE_SECONDS:
                self.finish('failure' if rapid else 'censored',
                            'continuing posture deterioration during pause' if rapid else 'posture observation unresolved',
                            obs, physical=rapid)
                return self.brake(obs)
            if self.state == MovementState.RELEASE:
                token=self.release_memory.interrupt(obs,req,self.travel_axis)
                if token is not None:
                    self.history.append(dict(event='release_response_retained',time=obs.time,**token))
            release_owner = (self.state == MovementState.RELEASE or self.suspended_release
                or (self.posture_pause is not None and self.posture_pause.intent == MovementState.RELEASE))
            if self.state != MovementState.BRAKE and release_owner:
                gradient = obs.terrain_gradient
                norm = math.hypot(*gradient) if gradient is not None else 0.
                self._observe_event('release_overspeed_brake',obs,dict(
                    prior_maneuver=self.state.value,ownership='RELEASE',reason='braking actual overspeed',
                    velocity_world=obs.velocity,actual_speed_mps=obs.speed,
                    axis_velocity_mps=(sum(v*a for v,a in zip(obs.velocity,self.travel_axis))
                                       if self.travel_axis is not None else None),
                    gradient_velocity_mps=(sum(v*g/norm for v,g in zip(obs.velocity,gradient)) if norm>0. else None),
                    speed_response_decision=speed_response,metrics=dict(self.metrics)))
            if self.state == MovementState.RELEASE:
                self.suspended_release = True
            self._state(MovementState.BRAKE, obs.time,
                'uphill zero-response cancelled: '+speed_response['reason'] if speed_response is not None
                and speed_response['action']=='cancel' else 'braking actual overspeed')
            self.stable_since = None
            self._observe_entry(obs, req)
            return self.brake(obs)
        # The learned policy can translate during yaw or braking. Arrival is a
        # measured response, not a privilege of the DRIVE command state.
        if (self.recovery_turn is None and self.recovery_continuation is None and self.state != MovementState.SETTLE
                and self.recovery_stage_started is None):
            if self.posture_pause is None and math.dist(obs.xy, req.endpoint) <= req.arrival_radius and supported(obs.xy):
                self._state(MovementState.SETTLE, obs.time, 'measured arrival during '+self.state.value.lower())
                return self.brake(obs)
            if along > length+max(.15, req.arrival_radius):
                disposition = ResultDisposition.REPOSITIONED if supported(obs.xy) else ResultDisposition.TERMINAL
                self.finish('censored', 'bounded displacement ended away from requested endpoint', obs, disposition=disposition)
                return self.brake(obs)
            if (cross > req.lateral_tolerance and not
                    (self.ascent_intent_bound is not None
                     and (self.state == MovementState.BRAKE or self.posture_pause is not None))):
                if (self.recovery_continuation_provider is not None and req.purpose=='recovery'
                        and req.direction==-1 and self.state==MovementState.DRIVE_REVERSE
                        and self.posture_pause is None):
                    failure=progress_failure or self._recovery_physical_progress(obs)
                    if failure:
                        self._finish_progress_failure(failure,obs)
                        return self.brake(obs)
                    if self._admit_recovery_continuation(obs,guard):
                        return self._continue_recovery(obs,guard)
                if self.departure_observer is not None:
                    try:
                        self.departure_observer(obs,req,dict(along_m=along,
                            signed_lateral_m=(obs.xy[0]-req.start[0])*uy-(obs.xy[1]-req.start[1])*ux,
                            lateral_m=cross,bound_m=req.lateral_tolerance,
                            distance_to_endpoint_m=math.dist(obs.xy,req.endpoint),length_m=length))
                    except Exception as exc:
                        self.history.append(dict(event='departure_observer_error',time=obs.time,
                            action_id=req.action_id,error=f'{type(exc).__name__}: {exc}'))
                if self.release_departure_guard is not None:
                    # Independent physical failure takes precedence over path
                    # reassessment. A tracking error alone is not terrain data.
                    if progress_failure:
                        self._finish_progress_failure(progress_failure,obs)
                        return self.brake(obs)
                    reason = (self.release_departure_guard(obs,req)
                              if self.state in (MovementState.DRIVE_FORWARD,MovementState.RELEASE)
                              else 'measured lateral departure from straight corridor')
                    if reason:
                        disposition = ResultDisposition.REPOSITIONED if supported(obs.xy) else ResultDisposition.TERMINAL
                        self.finish('censored',reason,obs,disposition=disposition)
                        return self.brake(obs)
                else:
                    physical = self.state in (MovementState.DRIVE_FORWARD, MovementState.DRIVE_REVERSE, MovementState.RELEASE)
                    disposition = ResultDisposition.REPOSITIONED if not physical and supported(obs.xy) else ResultDisposition.TERMINAL
                    self.finish('failure' if physical else 'censored', 'measured lateral departure from straight corridor',
                                obs, physical=physical, disposition=disposition)
                    return self.brake(obs)
        if progress_failure and self.posture_pause is None:
            self._finish_progress_failure(progress_failure, obs)
            return self.brake(obs)
        if self.posture_pause is not None:
            return self._posture_response(obs, req, supported, error, rapid, progress_failure)
        if self.recovery_turn is not None:
            return self._step_recovery_turn(obs, guard)
        if self.recovery_continuation is not None:
            return self._continue_recovery(obs,guard)
        if self.recovery_stage_started is not None or self.state == MovementState.BRAKE:
            stage_command = self._recovery_stage(obs)
            if stage_command is not None:
                return stage_command
        if self.state == MovementState.BRAKE:
            resume_steep = (obs.inclination > 15. and self.committed_heading is not None
                            and math.cos(error) > 0.)
            # A checked, aligned straight exit can take over from slow creep.
            # Turning retains the stricter stationary admission.
            threshold = .25 if obs.inclination > 15. and abs(error) <= .10 else .10
            stopped = obs.speed < threshold and abs(obs.yaw_rate) < .10
            self.stable_since = (self.stable_since if self.stable_since is not None else obs.time) if stopped else None
            straight_ready, entry_rejection = self._straight_admission(obs, req, error) if not self.suspended_release else (False, '')
            release_ready, release_rejection = self._moving_release_admission(obs,req,error)
            entry_rejection = entry_rejection or release_rejection
            if entry_rejection:
                self.finish('censored', entry_rejection, obs)
                return self.brake(obs)
            stationary_ready = self.stable_since is not None and obs.time-self.stable_since >= .3
            # A straight sign reversal must finish opposing translation first.
            # Turning and suspended RELEASE retain their stricter admission.
            longitudinal = obs.velocity[0]*math.cos(obs.yaw)+obs.velocity[1]*math.sin(obs.yaw)
            heading_ready = self._heading_acceptable(obs, req, error, .10)
            if not self.suspended_release and heading_ready and req.direction*longitudinal < -.10:
                stationary_ready = False
            if stationary_ready or straight_ready or release_ready:
                if self.suspended_release:
                    rejection = ''
                    if not self._uphill_eligible(obs):
                        rejection = 'suspended release lacks fresh uphill support'
                    elif math.cos(error) <= 0. or (obs.turn_allowed and abs(error) > .16):
                        rejection = 'suspended release heading needs a new checked approach'
                    elif self.corridor_guard:
                        rejection = self.corridor_guard(obs, req, obs.yaw)
                    if not rejection:
                        rejection = self.release_guard(obs, req)
                    if rejection:
                        self.finish('censored', rejection, obs)
                        return self.brake(obs)
                    if self.release_memory.token is not None:
                        # A retained RELEASE response is not powered admission.
                        # Require the full moving-release condition as well as
                        # the independently checked forward command below.
                        if (not release_ready or not self.release_memory.matches(req,self.travel_axis)
                                or obs.up_z<.8 or obs.terrain_relative_tilt is None
                                or obs.terrain_relative_tilt>self.POSTURE_RELATIVE_ENVELOPE_DEG
                                or self._straight_motion_window(obs) is None):
                            self.reason='bounded braking; retained release response awaits powered admission'
                            return self.brake(obs)
                        command,rejection=self._checked_drive(Motion(req.speed,0.),guard)
                        if rejection:
                            self.finish('censored',rejection,obs)
                            return self.brake(obs)
                        token=self.release_memory.token
                        self.history.append(dict(event='release_forward_correction_admitted',time=obs.time,
                            action_id=req.action_id,interrupted_at=token['interrupted_at'],
                            body_yaw=obs.yaw,axis=self.travel_axis,
                            axis_progress_mps=self.metrics['release_entry_axis_progress_mps'],
                            uphill_progress_mps=self.metrics['release_entry_uphill_progress_mps'],
                            reason='independent bounded powered admission after nonpositive pre-brake release response'))
                        self._state(MovementState.DRIVE_FORWARD,obs.time,
                            'resume same checked forward climb after nonpositive pre-brake release response')
                        self.history.append(dict(event='release_forward_correction_emitted',time=obs.time,
                            action_id=req.action_id,interrupted_at=token['interrupted_at'],
                            command_vx=command.vx,command_omega=command.omega,position=obs.xy,
                            uphill_budget_remaining_s=self.metrics.get('uphill_budget_remaining_s'),
                            uphill_wall_remaining_s=self.metrics.get('uphill_wall_remaining_s')))
                        self.release_memory.consume()
                        self.forward_output=True
                        return self._emit_traversal(command,obs)
                    self.suspended_release = False
                    self.release_origin = obs.xy
                    if release_ready and not stationary_ready:
                        self.history.append(dict(event='moving_release_admitted',time=obs.time,
                            action_id=req.action_id,position=obs.xy,body_yaw=obs.yaw,
                            brake_episode_started=self.brake_episode_started,
                            axis_progress_mps=self.metrics['release_entry_axis_progress_mps'],
                            uphill_progress_mps=self.metrics['release_entry_uphill_progress_mps'],
                            heading_excursion_rad=self.metrics['release_entry_heading_excursion_rad'],
                            reason='bounded uphill motion and fresh stopping geometry; unchanged request'))
                    self._state(MovementState.RELEASE, obs.time,
                        'restore checked release from bounded uphill motion' if release_ready and not stationary_ready
                        else 'restore checked release after measured braking')
                    return self._emit_traversal(Motion(), obs)
                candidate_heading = desired if not heading_ready and not resume_steep else obs.yaw
                rejection = self.corridor_guard(obs, req, candidate_heading) if self.corridor_guard else ''
                if rejection:
                    self.finish('censored', rejection, obs)
                    return self.brake(obs)
                if not heading_ready and not resume_steep:
                    if not obs.turn_allowed:
                        if obs.inclination > 15.:
                            self.finish('censored', 'alignment requires a gentle staging footprint', obs)
                        else:
                            self.reason = 'waiting for posture to settle before alignment'
                        return self.brake(obs)
                    self._state(MovementState.ALIGN_LEFT if error > 0 else MovementState.ALIGN_RIGHT,
                                obs.time, 'align before straight drive')
                else:
                    _, reason = self._checked_drive(Motion(req.direction*req.speed, 0.), guard)
                    if reason:
                        self.finish('censored', reason, obs)
                        return self.brake(obs)
                    # Commit the checked body heading, including after gentle
                    # realignment. Steep entry must not restore the old bearing.
                    if not resume_steep:
                        self.committed_heading = obs.yaw
                        self.travel_axis = (req.direction*math.cos(obs.yaw), req.direction*math.sin(obs.yaw))
                    self._state(MovementState.DRIVE_FORWARD if req.direction > 0 else MovementState.DRIVE_REVERSE,
                                obs.time, 'committed aligned straight takeover' if straight_ready and not stationary_ready else 'committed straight primitive')
                    if straight_ready and not stationary_ready and self.metrics.get('straight_entry_continuation'):
                        self.pending_drive_continuation=obs.time
                        self.history.append(dict(event='committed_drive_continuation_admitted',time=obs.time,
                            action_id=req.action_id,direction=req.direction,position=obs.xy,body_yaw=obs.yaw,
                            committed_axis=self.travel_axis,heading_error_rad=error,
                            uphill_budget_remaining_s=self.metrics.get('uphill_budget_remaining_s'),
                            reason='same executed steep drive, bounded motion and checked geometry; next output remains braking'))
            elif self.brake_episode_started is not None and obs.time-self.brake_episode_started > 8.:
                self.finish('failure', 'unable to brake and settle before primitive', obs, physical=True)
            return self.brake(obs)
        if self.state in (MovementState.ALIGN_LEFT, MovementState.ALIGN_RIGHT):
            if obs.inclination > 15.:
                self.finish('censored', 'turn staging lost; no hillside yaw', obs)
                return self.brake(obs)
            if not obs.turn_allowed or obs.body_tilt > 25. or obs.speed > .15:
                self._state(MovementState.BRAKE, obs.time, 'pause alignment for measured posture or drift')
                return self.brake(obs)
            if abs(error) < self.alignment_best-.03:
                self.alignment_best, self.alignment_progress_at = abs(error), obs.time
            if obs.time-self.alignment_progress_at > 12.:
                self.finish('censored', 'alignment lacks measured angular progress', obs)
                return self.brake(obs)
            # The learned policy can unwind yaw after release. Terminate on
            # measured residual/crossing, not an assumed positive coast angle.
            if self.pulse_direction*error <= .02 or obs.time-self.entered >= 2.:
                self._state(MovementState.BRAKE, obs.time, 'settle measured yaw before drive')
                return self.brake(obs)
            # Match the teleop button's demonstrated demand. Small proportional
            # commands were below the rocky policy's measured turning response.
            # Stop on measured heading, then settle yaw before deciding again.
            vx, omega = directional_command(left=self.pulse_direction > 0, right=self.pulse_direction < 0, omega=.35)
            command = Motion(vx, omega)
            reason = guard(command)
            if reason:
                self.finish('censored', reason, obs)
                return self.brake(obs)
            return command
        if self.state == MovementState.SETTLE:
            if not supported(obs.xy):
                self.finish('censored', 'endpoint support unavailable while settling', obs)
                return self.brake(obs)
            if math.dist(obs.xy, req.endpoint) > max(.25, req.arrival_radius):
                # Position tolerance is not terrain capability evidence. Keep
                # the destination outstanding and let the supervisor replan
                # from this measured, supported pose without a no-go vote.
                self.finish('censored', 'supported endpoint departure while settling', obs,
                            disposition=ResultDisposition.REPOSITIONED)
                return self.brake(obs)
            self.samples.append((obs.time, obs.xy))
            while len(self.samples) > 1 and self.samples[1][0] <= obs.time-.6:
                self.samples.pop(0)
            elapsed = obs.time-self.samples[0][0]
            drift = math.dist(obs.xy, self.samples[0][1])/max(elapsed, 1e-6)
            if elapsed >= .6-1e-6 and drift < .05 and obs.speed < .12 and abs(obs.yaw_rate) < .12:
                self.finish('success', 'primitive endpoint reached and settled', obs)
            elif obs.time-self.entered > 6.:
                self.finish('failure', 'endpoint did not settle', obs, physical=True)
            return self.brake(obs)
        if not self._heading_acceptable(obs, req, error, .16) and self.state != MovementState.RELEASE:
            if obs.turn_allowed:
                self._state(MovementState.BRAKE, obs.time, 'stop before correcting gentle-terrain heading')
                return self.brake(obs)
            if math.cos(error) <= 0.:
                self.finish('censored', 'command heading no longer advances the committed corridor', obs)
                return self.brake(obs)
            if not self.heading_deviation_reported:
                self.history.append(dict(event='movement_heading_variation', time=obs.time,
                    action_id=req.action_id, heading_error_rad=error, position=obs.xy,
                    reason='continue zero-yaw drive subject to measured geometry and progress'))
                self.heading_deviation_reported = True
            # Learned-policy body yaw is not traversal failure. The unchanged
            # axis, corridor, progress, posture and emitted-command guards below
            # decide whether this same bounded straight command can continue.
        self.metrics['heading_variation_rad'] = error
        # Progress uses the latest checked travel axis. Request-relative along
        # and cross remain independent total-displacement bounds.
        tx, ty = self.travel_axis or (ux, uy)
        progress_position = obs.xy[0]*tx+obs.xy[1]*ty
        self.samples.append((obs.time, progress_position, obs.xy))
        while len(self.samples) > 1 and self.samples[1][0] <= obs.time-2.:
            self.samples.pop(0)
        elapsed = obs.time-self.samples[0][0]
        progress = (progress_position-self.samples[0][1])/max(elapsed, 1e-6)
        forward = obs.velocity[0]*tx+obs.velocity[1]*ty
        lateral = abs(obs.velocity[0]*ty-obs.velocity[1]*tx)
        gradient = obs.terrain_gradient
        valid_gradient = gradient is not None and all(math.isfinite(v) for v in gradient)
        grade = math.degrees(math.atan(gradient[0]*tx+gradient[1]*ty)) if valid_gradient else 0.
        uphill_rate = 0.
        if valid_gradient:
            distance_uphill = sum((obs.xy[i]-self.samples[0][2][i])*gradient[i] for i in (0, 1))
            uphill_rate = distance_uphill/max(math.hypot(*gradient)*elapsed, 1e-9)
        release_eligible = (req.direction == 1 and req.purpose == 'mission' and grade >= 10.
                            and self.release_guard is not None)
        self.uphill_patience |= release_eligible
        self.metrics.update(status='progress' if progress >= .02 else 'measuring',
                            forward_mps=forward, progress_mps=progress, lateral_mps=lateral, window_s=elapsed,
                            uphill_progress_mps=uphill_rate, directional_grade_deg=grade,
                            release_active=self.state == MovementState.RELEASE)
        if not self.uphill_patience and elapsed >= 2.-1e-6 and progress < .02:
            if self.stall_since is None:
                self.stall_since = obs.time
                self.stall_origin = progress_position
            if obs.time-self.stall_since >= 3.:
                self.finish('failure', 'sustained measured stall or backslide', obs, physical=True)
                return self.brake(obs)
        elif self.stall_origin is None or progress_position-self.stall_origin >= .10:
            # A threshold blip is not a new independent recovery opportunity.
            # Require useful net advancement before renewing uphill patience.
            self.stall_since = None
            self.stall_origin = None
        if elapsed >= 1.-1e-6 and progress < -.10:
            self.finish('failure', 'measured backslide against committed direction', obs, physical=True)
            return self.brake(obs)
        # A completed DRIVE failure on this observation belongs to the approach,
        # even when the robot has just reached otherwise usable staging ground.
        stage_command = self._recovery_stage(obs)
        if stage_command is not None:
            return stage_command
        speed_command = self._emit_speed_response(speed_response,obs,guard)
        if speed_command is not None:
            return speed_command
        if self.state == MovementState.RELEASE:
            # A zero teleop command still drives the learned policy. Check both
            # prospective creep and actual drift; it is not a stationary brake.
            reason = self.release_guard(obs, req) if self.release_guard else 'release envelope unavailable'
            if reason:
                self.finish('censored', reason, obs)
                return self.brake(obs)
            release_elapsed = obs.time-self.entered
            net = sum((obs.xy[i]-self.release_origin[i])*v for i, v in enumerate((tx, ty)))
            measured = (elapsed >= self.RELEASE_RESPONSE_SECONDS-1e-9
                        and release_elapsed >= self.RELEASE_RESPONSE_SECONDS-1e-9)
            useful = progress >= self.RELEASE_PROGRESS_MPS and uphill_rate >= self.RELEASE_PROGRESS_MPS
            # A slow but still climbing policy gets a finite observation window.
            # Either deficient projection starts the plateau clock, only after
            # a complete measured response. Outer attempt clocks never restart.
            self.release_bad_since = (None if not measured or useful else
                self.release_bad_since if self.release_bad_since is not None else obs.time)
            losing_progress = (self.release_bad_since is not None
                               and obs.time-self.release_bad_since >= self.RELEASE_PLATEAU_SECONDS-1e-9)
            self.metrics.update(release_window_complete=measured,
                release_plateau_s=0. if self.release_bad_since is None else obs.time-self.release_bad_since)
            if (not release_eligible or net < -self.RELEASE_BACKSLIDE_METERS
                    or (release_elapsed >= self.RELEASE_RESPONSE_SECONDS and losing_progress)):
                if obs.turn_allowed and abs(error) > .16:
                    self._state(MovementState.BRAKE, obs.time, 'release reached gentle terrain; settle before heading correction')
                    return self.brake(obs)
                reason = ('release heading no longer advances the committed corridor' if math.cos(error) <= 0. else '')
                if not reason and self.corridor_guard:
                    reason = self.corridor_guard(obs, req, obs.yaw)
                if not reason:
                    command, reason = self._checked_drive(Motion(req.direction*req.speed, 0.), guard)
                if reason:
                    self.finish('censored', reason, obs)
                    return self.brake(obs)
                self.history.append(dict(event='release_correction_emitted',time=obs.time,
                    action_id=req.action_id,command_vx=command.vx,command_omega=command.omega,
                    reason='eligibility lost' if not release_eligible else
                           'net backward displacement' if net < -self.RELEASE_BACKSLIDE_METERS else 'sustained measured plateau',
                    axis_progress_mps=progress,uphill_progress_mps=uphill_rate,
                    net_axis_m=net,release_elapsed_s=release_elapsed,
                    plateau_s=self.metrics['release_plateau_s'],
                    exposure_remaining_s=self.metrics.get('uphill_budget_remaining_s'),
                    wall_remaining_s=self.metrics.get('uphill_wall_remaining_s')))
                self._state(MovementState.DRIVE_FORWARD, obs.time, 'release lost uphill progress; resume same checked climb')
                self.metrics['release_active'] = False
                self.forward_output = True
                return self._emit_traversal(command, obs)
            self.reason = 'policy-active release: retain zero command while measured uphill progress continues'
            self.metrics['status'] = 'release_progress' if useful else 'release_observe'
            return self._emit_traversal(Motion(), obs)
        if self._drive_release_ready(obs,release_eligible):
            reason = self.release_guard(obs, req)
            if not reason:
                self.release_origin = obs.xy
                self.history.append(dict(event='drive_release_selected',time=obs.time,
                    action_id=req.action_id,position=obs.xy,forward_exposure_s=self.drive_exposure,
                    response_window_s=self.metrics['drive_response_window_s'],
                    axis_progress_mps=self.metrics['drive_response_axis_mps'],
                    uphill_progress_mps=self.metrics['drive_response_uphill_mps'],
                    low_progress_dwell_s=self.metrics['drive_slow_dwell_s'],
                    **({'correction_episode':True} if self.release_memory.correction_active else {}),
                    reason='sustained measured powered slowdown; try checked policy-active zero command'))
                self._state(MovementState.RELEASE, obs.time, 'powered uphill progress slowed; observe policy-active zero command')
                self.metrics['release_active'] = True
                self.metrics['forward_exposure_s'] = 0.
                return self._emit_traversal(Motion(), obs)
        vx, omega = directional_command(forward=req.direction > 0, back=req.direction < 0, vx=req.speed)
        command = Motion(vx, omega)
        command, reason = self._checked_drive(command, guard)
        if reason:
            self.finish('censored', reason, obs)
            return self.brake(obs)
        self.reason = f'committed {"forward" if req.direction > 0 else "reverse"}: {max(0., along):.2f}/{length:.2f} m'
        self.forward_output = req.direction > 0
        return self._emit_traversal(command, obs)
