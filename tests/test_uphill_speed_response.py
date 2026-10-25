"""Bound zero-command speed response without changing common motion guards."""
from dataclasses import replace
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from test_behavior_navigation import b,m,ground


class UphillSpeedResponseTests(unittest.TestCase):
    def obs(self,t,speed=.2,xy=None,tilt=30.):
        angle=math.radians(tilt)
        return m.MotionObservation(xy or (.2*t,0.),0.,(speed,0.),math.cos(angle),0.,t,30.,False,True,
            (math.tan(math.radians(30.)),0.),(math.cos(angle/2),0.,-math.sin(angle/2),0.))

    def executor(self,enabled=True):
        e=m.MovementExecutor();e.uphill_speed_response_enabled=enabled
        e.release_guard=e.posture_guard=e.corridor_guard=e.straight_entry_guard=lambda *args:''
        e.start(m.MovementRequest('first',(0.,0.),(20.,0.),1,.4,100.),0.)
        e.travel_axis=(1.,0.);e.committed_heading=0.
        e._state(m.MovementState.DRIVE_FORWARD,0.,'fixture')
        for t in np.arange(0.,2.001,.02):self.tick(e,self.obs(float(t)))
        return e

    def tick(self,e,obs):
        return e.tick(obs,lambda cmd:'',lambda xy:True)

    def enter(self,e=None):
        e=e or self.executor()
        command=self.tick(e,self.obs(2.02,.46))
        self.assertEqual(command,m.Motion())
        self.assertEqual(e.state,m.MovementState.RELEASE)
        self.assertTrue(e.speed_response_used)
        return e

    def test_first_surge_emits_zero_then_restores_checked_power_after_decay(self):
        e=self.executor();req=e.request;axis=e.travel_axis
        self.enter(e)
        self.assertEqual(e.speed_response_started,2.02)
        self.assertEqual(self.tick(e,self.obs(2.04,.44)),m.Motion(.4,0.))
        self.assertEqual(e.state,m.MovementState.DRIVE_FORWARD)
        self.assertIsNone(e.speed_response_started)
        self.assertIs(e.request,req);self.assertIs(e.travel_axis,axis)
        self.assertFalse(e.result)
        event=next(h for h in e.history if h['event']=='uphill_speed_response_ended')
        self.assertTrue(event['decayed']);self.assertAlmostEqual(event['elapsed_s'],.02)

    def test_powered_exposure_survives_pause_but_zero_time_is_not_power(self):
        e = self.executor()
        e.drive_exposure = 6.
        self.enter(e)
        self.assertAlmostEqual(e.drive_exposure, 6.02)
        self.assertEqual(e.drive_response_samples, [])
        self.tick(e, self.obs(2.04, .46))
        self.tick(e, self.obs(2.06, .44))
        self.assertAlmostEqual(e.drive_exposure, 6.02)
        self.assertEqual(e.drive_response_samples, [])
        self.assertEqual(e.executed_traversal.maneuver, m.MovementState.RELEASE.value)
        self.tick(e, self.obs(2.08, .2))
        self.assertAlmostEqual(e.drive_exposure, 6.04)
        self.assertEqual(e.executed_traversal.maneuver, m.MovementState.DRIVE_FORWARD.value)

    def test_new_episode_requires_new_continuous_powered_history(self):
        e = self.enter()
        self.tick(e, self.obs(2.04, .44))
        for t in np.arange(2.06, 4.041, .02):
            self.assertEqual(self.tick(e, self.obs(float(t))), m.Motion(.4, 0.))
        self.assertEqual(self.tick(e, self.obs(4.06, .46)), m.Motion())
        self.assertEqual(e.state, m.MovementState.RELEASE)
        self.assertEqual(sum(h['event']=='uphill_speed_response_started' for h in e.history), 2)
        self.assertEqual(e.request.deadline, 100.)

    def test_decay_requires_independent_powered_readmission(self):
        for kind in ('corridor', 'power', 'heading'):
            e = self.enter()
            obs = self.obs(2.04, .44)
            if kind == 'corridor':
                e.corridor_guard = lambda *args: 'new obstacle'
            elif kind == 'heading':
                obs = replace(obs, yaw=math.pi)
            command = e.tick(obs, lambda cmd: 'power rejected' if kind=='power' and cmd.vx>0 else '', lambda xy: True)
            if kind != 'heading':
                self.assertNotEqual(command, m.Motion(.4, 0.))
            # With a reversed body, positive body-x can be opposing damping.
            self.assertNotEqual(e.state, m.MovementState.DRIVE_FORWARD)
            if kind == 'heading':
                self.assertFalse(any(h['event']=='uphill_speed_response_drive_restored' for h in e.history))
            else:
                self.assertIsNotNone(e.result, kind)
                self.assertFalse(e.result.physical, kind)

    def test_deliberate_release_does_not_restore_power_just_because_speed_is_low(self):
        e = self.executor()
        e._state(m.MovementState.RELEASE, 2., 'intentional slowdown release')
        e.release_origin = (.4, 0.)
        for t in np.arange(2.02, 4.021, .02):
            self.assertEqual(self.tick(e, self.obs(float(t))), m.Motion())
        self.assertEqual(e.state, m.MovementState.RELEASE)
        self.assertFalse(any(h['event']=='uphill_speed_response_drive_restored' for h in e.history))

    def test_protective_pause_does_not_starve_later_measured_slowdown_release(self):
        e = self.executor()
        e.drive_exposure = 6.
        self.enter(e)
        self.tick(e, self.obs(2.04, .44))
        for t in np.arange(2.06, 5.401, .02):
            t = float(t)
            self.tick(e, self.obs(t, .005, xy=(.408+.005*(t-2.04), 0.)))
        selected = [h for h in e.history if h['event']=='drive_release_selected']
        self.assertEqual(len(selected), 1)
        self.assertGreaterEqual(selected[0]['forward_exposure_s'], 6.)
        self.assertGreaterEqual(selected[0]['low_progress_dwell_s'], 1.-1e-9)
        self.assertEqual(e.state, m.MovementState.RELEASE)
        self.assertIsNone(e.speed_response_started)

    def test_recorded_20ms_speed_decay_restores_drive_without_slowdown_release(self):
        data = json.loads((Path(__file__).parent/'fixtures/protective_zero_intent.json').read_text())
        e = m.MovementExecutor(); e.uphill_speed_response_enabled = True
        e.release_guard=e.posture_guard=e.corridor_guard=e.straight_entry_guard=lambda *args:''
        e.start(m.MovementRequest('recorded', tuple(data['request_start']), tuple(data['endpoint']), 1, .4, 500.), 336.72)
        e.travel_axis=tuple(data['travel_axis']); e.committed_heading=math.atan2(e.travel_axis[1],e.travel_axis[0])
        e._state(m.MovementState.DRIVE_FORWARD, 336.72, 'recorded resumed drive')
        commands=[]
        for row in data['rows']:
            t,x,y,yaw,vx,vy,up,yaw_rate,slope,gx,gy,*quaternion=row
            obs=m.MotionObservation((x,y),yaw,(vx,vy),up,yaw_rate,t,slope,False,True,(gx,gy),tuple(quaternion))
            commands.append(self.tick(e,obs))
        self.assertEqual(commands[-2],m.Motion())
        self.assertEqual(commands[-1],m.Motion(.4,0.))
        self.assertFalse(any(h['event']=='drive_release_selected' for h in e.history))
        self.assertEqual(sum(h['event']=='uphill_speed_response_started' for h in e.history),1)
        self.assertEqual(e.state,m.MovementState.DRIVE_FORWARD)

    def test_failed_decay_brakes_at_fixed_deadline_without_renewal(self):
        e=self.enter()
        for t in np.arange(2.04,2.201,.02):
            self.assertEqual(self.tick(e,self.obs(float(t),.46)),m.Motion())
            self.assertEqual(e.speed_response_started,2.02)
        self.assertEqual(self.tick(e,self.obs(2.22,.46)),m.Motion(-.4,0.))
        self.assertEqual(e.state,m.MovementState.BRAKE)
        self.assertTrue(e.speed_response_used);self.assertIsNone(e.speed_response_started)

    def test_intervention_boundary_and_downhill_keep_damping(self):
        for speed in (.5,.7,-.46):
            e=self.executor();command=self.tick(e,self.obs(2.02,speed))
            self.assertEqual(command,m.Motion(-math.copysign(.4,speed),0.))
            self.assertEqual(e.state,m.MovementState.BRAKE);self.assertFalse(e.speed_response_used)
        e=self.enter();self.assertEqual(self.tick(e,self.obs(2.04,.5)),m.Motion(-.4,0.))
        self.assertEqual(e.state,m.MovementState.BRAKE)

    def test_active_downhill_even_below_trigger_cancels_to_brake(self):
        e=self.enter()
        self.assertEqual(self.tick(e,self.obs(2.04,-.2)),m.Motion(.4,0.))
        self.assertEqual(e.state,m.MovementState.BRAKE);self.assertTrue(e.speed_response_used)

    def test_missing_history_and_brake_time_cannot_qualify(self):
        for kind in ('samples','execution','brake','gap'):
            e=self.executor()
            if kind=='samples':e.drive_response_samples=[]
            elif kind=='execution':e.executed_traversal=None;e.pending_traversal=None
            elif kind=='brake':e._state(m.MovementState.BRAKE,2.,'fixture')
            obs=self.obs(2.08 if kind=='gap' else 2.02,.46)
            self.tick(e,obs)
            self.assertFalse(e.speed_response_used,kind)
            self.assertEqual(e.state,m.MovementState.BRAKE,kind)

    def test_missing_normal_attitude_unrecovered_posture_and_nonuphill_reject(self):
        for kind in ('normal','attitude','posture','terrain','projection'):
            e=self.executor();obs=self.obs(2.02,.46)
            if kind=='normal':obs=replace(obs,terrain_gradient=None)
            elif kind=='attitude':obs=replace(obs,body_quaternion=None)
            elif kind=='posture':obs=self.obs(2.02,.46,tilt=38.)
            elif kind=='terrain':obs=replace(obs,terrain_gradient=(0.,0.))
            else:obs=replace(obs,velocity=(0.,.46))
            self.tick(e,obs);self.assertFalse(e.speed_response_used,kind)

    def test_geometry_guards_bind_on_entry_and_during_response(self):
        e=self.executor();e.release_guard=lambda *args:'blocked'
        self.assertEqual(self.tick(e,self.obs(2.02,.46)),m.Motion(-.4,0.))
        self.assertFalse(e.speed_response_used)
        e=self.enter();e.release_guard=lambda *args:'blocked'
        self.assertEqual(self.tick(e,self.obs(2.04,.44)),m.Motion(-.4,0.))
        self.assertEqual(e.state,m.MovementState.BRAKE)

    def test_latch_survives_brake_posture_and_alignment_until_new_primitive(self):
        e=self.enter()
        for state in (m.MovementState.BRAKE,m.MovementState.POSTURE_PAUSE,m.MovementState.ALIGN_LEFT):
            e._state(state,2.1,'fixture interruption');self.assertTrue(e.speed_response_used)
        e.finish('censored','fixture',self.obs(2.2));e.consume()
        e.start(m.MovementRequest('second',(0.,0.),(20.,0.),1,.4,100.),2.3)
        self.assertFalse(e.speed_response_used)

    def test_second_surge_cannot_reenter_after_successful_decay(self):
        e=self.enter();self.tick(e,self.obs(2.04,.44))
        self.assertEqual(self.tick(e,self.obs(2.06,.46)),m.Motion(-.4,0.))
        self.assertEqual(e.state,m.MovementState.BRAKE)
        self.assertEqual(sum(h['event']=='uphill_speed_response_started' for h in e.history),1)

    def test_arrival_overshoot_lateral_and_progress_take_precedence(self):
        for kind in ('arrival','overshoot','lateral','progress','deadline','tip','stop'):
            e=self.executor();obs=self.obs(2.02,.46)
            if kind=='arrival':obs=replace(obs,xy=(20.,0.))
            elif kind=='overshoot':obs=replace(obs,xy=(21.,0.))
            elif kind=='lateral':obs=replace(obs,xy=(.404,.6))
            elif kind=='progress':
                e.uphill_patience=True;e.uphill_stall_since=1.;e.uphill_stall_origin=(100.,0.);e.uphill_exposure=12.
            elif kind=='deadline':obs=replace(obs,time=101.)
            elif kind=='tip':obs=self.obs(2.02,.46,tilt=65.)
            else:e.stop(obs)
            self.tick(e,obs)
            self.assertFalse(e.speed_response_used,kind)
            self.assertNotEqual(e.state,m.MovementState.RELEASE,kind)
            if kind=='arrival':self.assertEqual(e.state,m.MovementState.SETTLE)
            elif kind in ('overshoot','lateral','progress'):self.assertIsNotNone(e.result,kind)

    def test_active_common_checks_are_not_bypassed(self):
        for kind in ('lateral','overshoot','arrival','stale','tip'):
            e=self.enter();obs=self.obs(2.04,.46)
            if kind=='lateral':obs=replace(obs,xy=(.408,.6))
            elif kind=='overshoot':obs=replace(obs,xy=(21.,0.))
            elif kind=='arrival':obs=replace(obs,xy=(20.,0.))
            elif kind=='stale':obs=replace(obs,fresh=False)
            else:obs=self.obs(2.04,.46,tilt=65.)
            self.tick(e,obs);self.assertNotEqual(e.state,m.MovementState.RELEASE,kind)
            self.assertTrue(e.speed_response_used,kind)

    def test_existing_low_progress_budget_is_not_renewed_by_admission(self):
        e=self.executor();e.uphill_stall_since=1.;e.uphill_stall_origin=(.4,0.);e.uphill_exposure=4.
        self.enter(e)
        self.assertEqual(e.uphill_stall_since,1.);self.assertEqual(e.uphill_stall_origin,(.4,0.))
        self.assertAlmostEqual(e.uphill_exposure,4.02)
        self.tick(e,self.obs(2.04,.44))
        self.assertAlmostEqual(e.uphill_exposure,4.04)
        self.assertEqual(e.request.deadline,100.)

    def test_default_mode_retains_original_uphill_braking(self):
        e=self.executor(enabled=False)
        self.assertEqual(self.tick(e,self.obs(2.02,.46)),m.Motion(-.4,0.))
        self.assertEqual(e.state,m.MovementState.BRAKE)
        self.assertFalse(any(h['event'].startswith('uphill_speed_response') for h in e.history))

    def test_configuration_is_characterization_only_and_before_attempts(self):
        c=b.BehaviorTreeAscentController(ground(),[SimpleNamespace(x=4.,y=0.,label='goal')],mode='characterize')
        c.configure_uphill_speed_response();self.assertTrue(c.executor.uphill_speed_response_enabled)
        self.assertIn('v30',c.protocol_version)
        c.attempt_number=1
        with self.assertRaises(ValueError):c.configure_uphill_speed_response()
        c.attempt_number=0;c.mode='qualified'
        with self.assertRaises(ValueError):c.configure_uphill_speed_response()

    def test_resume_protocol_survives_supported_combined_configurations(self):
        marker = 'protective-drive-resume-v1'
        for options in (('configure_release_memory', 'configure_gentle_turn_entry', 'configure_active_zero_envelope'),
                        ('configure_release_memory', 'configure_correction_release', 'configure_uphill_speed_response'),
                        ('configure_recovery_continuation', 'configure_uphill_speed_response')):
            c=b.BehaviorTreeAscentController(ground(),[SimpleNamespace(x=4.,y=0.,label='goal')],mode='characterize')
            c.configure_uphill_speed_response()
            for option in options:
                getattr(c, option)()
                self.assertEqual(c.protocol_version.count(marker), 1, option)


if __name__ == '__main__':unittest.main()
