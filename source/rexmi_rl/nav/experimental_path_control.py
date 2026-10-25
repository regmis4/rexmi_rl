"""User-authorized simulator exploration: no terrain or motion vetoes.

The fixed fitted path supplies direction. Measured lack of progress selects
release, push, reverse or a new heading. STOP/manual, invalid-data handling and
the finite simulation budget are execution controls, not terrain admission.
"""
import math
import time

import numpy as np

from .checkpoint_control import Motion, wrap_angle
from .movement import MotionObservation
from .saved_path_control import SavedPathController
from . import planning_snapshot, slope_cost
from .reverse_recovery import ReverseRecovery


class ExperimentalPathController(SavedPathController):
    def __init__(self, *args, preference_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.protocol_version = 'saved-slope-path-v2-unguarded-feedback-recovery'
        if preference_model is None:
            frozen, _ = planning_snapshot.load(self.saved_path_metadata['map_path'])
            preference_model = slope_cost.Model(frozen, slope_cost.Parameters(**self.saved_path_metadata['parameters']))
        self.preference = preference_model
        self.response_at = None
        self.response_score = 0.
        self.recovery = None
        self.recovery_history = []
        self.recovery_count = 0
        self.turn_after = 'DRIVE'
        self.actual_route_distance = 0.
        self.reverse_recovery = ReverseRecovery()
        self.translation_at=None
        self.translation_credit=0.
        self.translation_turns=0
        self.translation_exposure=0.

    def request(self, action, now, vx=0., omega=0.):
        super().request(action, now, vx, omega)
        self.recovery = None
        self.reverse_recovery.active = False
        self.response_at = now
        self.response_score = self.actual_route_distance-self.tracking_error
        self.translation_at=now
        self.translation_credit=self.response_score
        self.translation_turns=0
        self.translation_exposure=self.drive_exposure

    def finish_attempt(self, outcome, reason, xy, yaw, now, physical=False, **kwargs):
        if getattr(self,'recovery',None) is not None:
            self._finish_action('interrupted: '+reason,censored=True,end_time=now)
        return super().finish_attempt(outcome,reason,xy,yaw,now,physical,**kwargs)

    def _evidence(self):
        o = self.obs
        return dict(position=list(o.xy), yaw=o.yaw, velocity=list(o.velocity),
            yaw_rate=o.yaw_rate, upright=o.up_z, route_progress_m=self.actual_route_distance,
            cross_track_m=self.tracking_error, heading_error_rad=self.heading_error,
            forward_mps=self.metrics['forward_mps'], window_progress_mps=self.metrics['progress_mps'],
            directional_grade_deg=self.metrics['directional_grade_deg'])

    def _start_action(self, action, reason, heading=None):
        o = self.obs
        self.recovery_count += 1
        self.recovery = dict(action=action, started=o.time, start_xy=o.xy,
            start_score=self.actual_route_distance-self.tracking_error,
            heading=o.yaw if heading is None else heading, reason=reason)
        if action == 'REVERSE':
            self.reverse_recovery.start(o.time,o.xy,o.yaw)
            self.translation_at=o.time
        self.attempt_events.append(dict(event='feedback_recovery_start', time=o.time,
            action=action, reason=reason, **self._evidence()))
        if action == 'TURN':
            self._start_turn(heading)
            self.turn_after = 'PUSH'
        else:
            self._button('RECOVER', o.time, action.lower()+': '+reason)

    def _finish_action(self, reason, censored=False, end_time=None):
        a = self.recovery
        score = self.actual_route_distance-self.tracking_error
        gain = score-a['start_score']
        now=self.obs.time if end_time is None else end_time
        record = dict(event='feedback_recovery_end', time=now, action=a['action'], censored=censored,
            duration_s=now-a['started'], displacement_m=math.dist(self.obs.xy,a['start_xy']),
            route_score_gain_m=gain, reason=reason, start_xy=list(a['start_xy']),
            heading=a['heading'], **self._evidence())
        self.attempt_events.append(record)
        self.recovery_history.append(record)
        self.recovery_history = self.recovery_history[-64:]
        self.recovery = None
        return gain

    def _recovery_heading(self):
        """Soft terrain/route preference with memory of unsuccessful pushes.

        No candidate is vetoed by slope, footprint, freshness or clearance.
        These short projections rank attempts; they do not predict success.
        """
        o = self.obs
        target_s = min(self.actual_route_distance+1., self.route_distance[-1])
        target = np.array([np.interp(target_s,self.route_distance,self.saved_route[:,i]) for i in (0,1)])
        direct = math.atan2(target[1]-o.xy[1],target[0]-o.xy[0])
        options = []
        for offset in (0.,-.25,.25,-.5,.5):
            heading = wrap_angle(direct+offset)
            axis = np.array([math.cos(heading),math.sin(heading)])
            endpoint = np.asarray(o.xy)+.8*axis
            cell = self.preference.grid.world_to_cell(*endpoint)
            terrain = 100. if cell is None else float(self.preference(cell, *axis))
            if not math.isfinite(terrain):terrain=100.
            failed = sum(1 for e in self.recovery_history
                if e['action']=='PUSH' and not e['censored'] and e['route_score_gain_m']<.1
                and math.dist(e['start_xy'],o.xy)<1.5 and abs(wrap_angle(e['heading']-heading))<.3)
            score = 4.*math.dist(endpoint,target)+.05*terrain+2.*failed
            options.append(dict(heading=heading, score=score, terrain_cost=terrain, failed_pushes=failed))
        chosen = min(options,key=lambda item:item['score'])
        self.attempt_events.append(dict(event='feedback_heading_selection',time=o.time,
            candidates=options,selected=chosen,**self._evidence()))
        return chosen['heading']

    def _action_tick(self):
        o, a = self.obs, self.recovery
        elapsed = o.time-a['started']
        if a['action'] == 'REVERSE':
            command,result=self.reverse_recovery.tick(o.time,o.xy)
            if result is not None:
                self.attempt_events.append(dict(event='reverse_response',time=o.time,**result))
                self._finish_action('reverse displacement reached' if result['success'] else 'reverse response window elapsed')
                heading = self._recovery_heading()
                self._start_action('TURN','repositioned; try a terrain-ranked approach',heading)
                return self._action_tick()
            return Motion(command,0.)
        if a['action'] == 'TURN':
            residual = wrap_angle(a['heading']-o.yaw)
            if abs(residual)<self.turn_best-.03:
                self.turn_best,self.turn_progress_at=abs(residual),o.time
            if abs(residual)<=.1:
                heading=a['heading'];self._finish_action('measured heading reached')
                self._start_action('PUSH','test the new approach',heading)
                return Motion(self.cruise_speed,0.)
            if elapsed>=12. or o.time-self.turn_progress_at>=5.:
                self._finish_action('yaw response stalled')
                self._start_action('REVERSE','turn did not respond; make room')
                return self._action_tick()
            return Motion(0.,math.copysign(.35,residual))
        if a['action'] == 'PUSH':
            gain=self.actual_route_distance-self.tracking_error-a['start_score']
            if gain >= .3 or elapsed >= 5.:
                self._finish_action('useful route progress' if gain>=.3 else 'push response window elapsed')
                if gain>=.1:
                    self.response_at=o.time
                    self.response_score=self.actual_route_distance-self.tracking_error
                    self.saved_drive_since=o.time
                    self.translation_at=o.time
                    self.translation_credit=self.actual_route_distance-self.tracking_error
                    self.translation_turns=0;self.translation_exposure=self.drive_exposure
                    self._button('DRIVE',o.time,'recovery advanced; resume path')
                    return Motion(self.cruise_speed,0.)
                self._start_action('REVERSE','forward effort did not advance; change approach')
                return self._action_tick()
            return Motion(self.cruise_speed,0.)
        raise RuntimeError('Unknown recovery action')

    def _navigate(self):
        o=self.obs
        if o.time<3.:return Motion()
        if not self.saved_started:
            self.saved_started=True
            self.saved_progress_at=self.response_at=self.saved_drive_since=o.time
            self.translation_at=o.time
            self.translation_exposure=self.drive_exposure
            self.set_state('FOLLOW','unguarded simulation path execution',o.time)
            record=planning_snapshot.save(self,o.xy,o.yaw,o.time)
            planning_snapshot.finish(record,self,True)
            self.attempt_events.append(dict(event='saved_path_start',time=o.time,route=self.path,
                metadata=self.saved_path_metadata,snapshot=record,unguarded=True))
        distance,cross,index,target=self._projection(o.xy)
        self.actual_route_distance=distance
        self.path_index,self.tracking_error,self.target=index,cross,target
        self.saved_progress=max(self.saved_progress,distance)
        score=distance-cross
        if score-self.translation_credit>=.1:
            self.translation_credit,self.translation_at=score,o.time
            self.translation_turns=0;self.translation_exposure=self.drive_exposure
        if score-self.response_score>=.1:
            self.response_score,self.response_at=score,o.time
        self.metrics.update(route_progress_m=distance,cross_track_m=cross)
        if math.dist(o.xy,self.selected_exit)<=.4:
            if self.recovery:self._finish_action('endpoint settling takes ownership',censored=True)
            self._button('SETTLE',o.time,'selected endpoint reached; observe actual settling')
            stable=o.speed<.12 and abs(o.yaw_rate)<.12 and o.up_z>.5 and 0.<self.observation_dt<=.041
            self.goal_settle_since=(self.goal_settle_since if self.goal_settle_since is not None else o.time) if stable else None
            if self.goal_settle_since is not None and o.time-self.goal_settle_since>=.6:
                self.wp_idx=len(self.waypoints)
                self.set_state('COMPLETE','selected exterior endpoint reached upright and settled',o.time)
                self._button('COMPLETE',o.time,self.reason)
            return Motion()
        self.goal_settle_since=None
        error=wrap_angle(math.atan2(target[1]-o.xy[1],target[0]-o.xy[0])-o.yaw)
        self.heading_error=error
        if self.recovery:return self._action_tick()
        if (o.time-self.translation_at>=20. and
                (self.translation_turns>=2 or self.drive_exposure-self.translation_exposure>=5.)):
            self._start_action('REVERSE','no route advance despite repeated steering or forward effort')
            return self._action_tick()
        if self.button_state=='TURN':
            residual=wrap_angle(self.turn_target-o.yaw)
            if abs(residual)<self.turn_best-.03:
                self.turn_best,self.turn_progress_at=abs(residual),o.time
                self.response_at=o.time
            if abs(residual)<=.1:
                self._button('DRIVE',o.time,'measured turn complete')
                self.saved_drive_since=o.time
            elif o.time-self.turn_progress_at>=5. or o.time>=self.turn_deadline:
                self._start_action('REVERSE','normal yaw stopped responding')
                return self._action_tick()
            else:return Motion(0.,math.copysign(.35,residual))
        if o.time-self.response_at>=8.:
            if self.button_state=='RELEASE':
                self._start_action('PUSH','zero-input progress plateau; initiate forward motion')
            elif abs(error)>.35:
                self._start_action('TURN','stalled while misaligned; correct approach',self._recovery_heading())
            else:
                self._start_action('TURN','forward motion stalled; test a different approach',self._recovery_heading())
            return self._action_tick()
        if abs(error)>.22:
            self.translation_turns+=1
            self._start_turn(wrap_angle(o.yaw+error))
            return Motion(0.,math.copysign(.35,error))
        if self.button_state=='RELEASE':return Motion()
        slow=self.metrics['window_seconds']>=2.-1e-6 and self.metrics['progress_mps']<.1
        if o.time-self.saved_drive_since>=5. and slow and self.metrics['directional_grade_deg']>=10.:
            self._button('RELEASE',o.time,'slow powered climb; observe zero-input response')
            return Motion()
        self._button('DRIVE',o.time,'follow inspected path without motion vetoes')
        return Motion(self.cruise_speed,0.)

    def tick(self,xy,yaw,speed,now,upright=1.,velocity=None):
        started=time.perf_counter()
        velocity=tuple(velocity[:2]) if velocity is not None else (speed*math.cos(yaw),speed*math.sin(yaw))
        if not np.isfinite([*xy,yaw,speed,now,upright,*velocity]).all():
            if self.recovery:self._finish_action('invalid observation; last valid pose retained',censored=True)
            self.operator_mode='stop';self.set_state('HOLD','invalid simulator observation',now)
            self.command=self.previous_command=Motion()
            return Motion()
        dt=max(0.,now-self.last_tick) if self.last_tick is not None else 0.
        self.observation_dt=dt
        yaw_rate=wrap_angle(yaw-self.last_yaw)/dt if dt else 0.
        cell=self.grid.world_to_cell(*xy)
        gradient=np.array([self.grid.gx[cell],self.grid.gy[cell]]) if cell is not None else np.zeros(2)
        gradient=np.nan_to_num(gradient,nan=0.)
        inclination=math.degrees(math.atan(float(np.linalg.norm(gradient))))
        self.obs=MotionObservation(tuple(xy),yaw,velocity,upright,yaw_rate,now,inclination,
            True,cell is not None and bool(self.grid.fresh[cell]),tuple(gradient))
        self.last_xy,self.last_yaw,self.last_tick=tuple(xy),yaw,now
        self.actual_speed,self.upright=speed,upright
        self.samples.append((now,tuple(xy)))
        while len(self.samples)>1 and self.samples[1][0]<=now-2.:self.samples.popleft()
        elapsed=now-self.samples[0][0]
        delta=np.subtract(xy,self.samples[0][1])/max(elapsed,1e-9)
        axis=(math.cos(yaw),math.sin(yaw))
        forward=float(np.dot(velocity,axis))
        self.backslide_distance+=max(0.,-forward)*dt if self.button_state in ('DRIVE','RELEASE') else 0.
        if self.command.vx>0:self.drive_exposure+=dt
        self.metrics=dict(status=self.button_state,forward_mps=forward,progress_mps=float(delta@axis),
            lateral_mps=-velocity[0]*axis[1]+velocity[1]*axis[0],window_seconds=elapsed,
            directional_grade_deg=math.degrees(math.atan(float(gradient@axis))),
            uphill_progress_mps=float(delta@gradient/max(np.linalg.norm(gradient),1e-9)),
            release_active=self.button_state=='RELEASE',forward_exposure_s=self.drive_exposure)
        if self.operator_mode=='stop':command=Motion()
        elif self.operator_mode=='manual':command=self.manual
        elif now>=900.:command=self._hold('simulation experiment budget exhausted',now)
        else:command=self._navigate()
        self.tick_ms=(time.perf_counter()-started)*1000
        return self._emit(command)
