"""Mode-selection evidence must never become a motion or failure permission."""
import copy
from dataclasses import replace
import math
import unittest

from test_behavior_navigation import m,b,ground


def obs(t, x=0., *, vx=0., yaw=0., yaw_rate=0.):
    pitch=-math.pi/6
    # ZYX quaternion with observed body up following the 30 degree slope.
    cy,sy=math.cos(yaw/2),math.sin(yaw/2)
    cp,sp=math.cos(pitch/2),math.sin(pitch/2)
    q=(cy*cp,-sy*sp,cy*sp,sy*cp)
    return m.MotionObservation((x,0.),yaw,(vx,0.),math.cos(pitch),yaw_rate,t,30.,False,True,
        (math.tan(math.pi/6),0.),q)


class ReleaseMemoryTests(unittest.TestCase):
    def executor(self):
        e=m.MovementExecutor();e.release_memory.enabled=True
        e.release_guard=e.corridor_guard=e.straight_entry_guard=e.posture_guard=lambda *_:''
        e.start(m.MovementRequest('one',(0.,0.),(20.,0.),1,.4,100.),0.)
        e.travel_axis=(1.,0.);e.committed_heading=0.
        return e

    def history(self,e,position=lambda t:-.005*t,*,witness=True,gap=None,invalid=None):
        memory=e.release_memory;req=e.request;axis=e.travel_axis
        if witness:
            memory.emitted(obs(0.),req,axis,'DRIVE_FORWARD',m.Motion(.4,0.))
            memory.observe(obs(.02),req,axis)
        for i in range(1,202):
            t=round(i*.02,8);o=obs(t,position(t))
            if invalid and i==150:o=replace(o,**invalid)
            if gap is None or i not in (gap,gap+1):
                memory.observe(o,req,axis)
                memory.emitted(o,req,axis,'RELEASE',m.Motion())
        return obs(4.02,position(4.02),vx=.51)

    def token(self,e=None):
        e=e or self.executor();o=self.history(e)
        self.assertIsNotNone(e.release_memory.interrupt(o,e.request,e.travel_axis))
        return e,o

    def test_both_complete_windows_and_detached_witness_are_required(self):
        e,o=self.token();token=e.release_memory.token
        self.assertEqual(len(token['samples']),201)
        self.assertEqual(len(token['windows']),2)
        self.assertLess(token['drive_witness']['observed_time'],token['samples'][-1][0])
        self.assertFalse(token['probability_validated'])
        e.release_memory.samples.clear();self.assertEqual(len(token['samples']),201)
        e=self.executor();o=self.history(e,lambda t:.1*min(t,2.)-.01*max(0.,t-2.))
        self.assertIsNone(e.release_memory.interrupt(o,e.request,e.travel_axis))

    def test_productive_release_no_witness_and_changed_axis_cannot_latch(self):
        for kind in ('productive','witness','axis','gradient'):
            e=self.executor();o=self.history(e,lambda t:.04*t if kind=='productive' else -.005*t,
                                            witness=kind!='witness')
            if kind=='axis':e.travel_axis=(math.cos(.1),math.sin(.1))
            if kind=='gradient':o=replace(o,terrain_gradient=None)
            self.assertIsNone(e.release_memory.interrupt(o,e.request,e.travel_axis),kind)

    def test_missing_fresh_contiguous_finite_history_cannot_latch(self):
        for changes in (dict(gap=150),dict(invalid={'fresh':False}),
                        dict(invalid={'xy':(math.nan,0.)}),dict(invalid={'body_quaternion':None})):
            e=self.executor();o=self.history(e,**changes)
            self.assertIsNone(e.release_memory.interrupt(o,e.request,e.travel_axis))

    def test_state_label_without_observed_held_zero_is_not_evidence(self):
        e=self.executor();e.state=m.MovementState.RELEASE
        for i in range(210):e.release_memory.observe(obs(i*.02),e.request,e.travel_axis)
        self.assertIsNone(e.release_memory.interrupt(obs(4.18),e.request,e.travel_axis))
        e=self.executor()
        for i in range(210):
            o=obs(i*.02);e.release_memory.observe(o,e.request,e.travel_axis)
            e.release_memory.emitted(o,e.request,e.travel_axis,'RELEASE',m.Motion(.2,0.))
        self.assertFalse(e.release_memory.samples)

    def braking(self,e):
        e.suspended_release=True;e.state=m.MovementState.BRAKE;e.brake_episode_started=4.02
        e.uphill_stall_since=4.;e.uphill_stall_origin=(0.,0.)
        e.uphill_exposure=.1;e.release_origin=(0.,0.)

    def drive_admission(self,e,guard=lambda command:'',end=6.10):
        command=None
        for i in range(202,round(end/.02)+1):
            t=i*.02
            command=e.tick(obs(t,.04*(t-4.02),vx=.04,yaw=.015*math.sin(10*t),
                yaw_rate=.15*math.cos(10*t)),guard,lambda xy:True)
            if e.result or e.state==m.MovementState.DRIVE_FORWARD:break
        return command

    def test_independent_admission_emits_once_without_clock_axis_or_target_reset(self):
        e,o=self.token();self.braking(e);request=e.request;axis=e.travel_axis
        command=self.drive_admission(e)
        self.assertEqual(command,m.Motion(.4,0.));self.assertEqual(e.state,m.MovementState.DRIVE_FORWARD)
        self.assertTrue(e.release_memory.used);self.assertIsNone(e.release_memory.token)
        self.assertIs(e.request,request);self.assertIs(e.travel_axis,axis)
        self.assertEqual(e.uphill_stall_since,4.);self.assertEqual(e.uphill_exposure,.1)
        self.assertEqual(request.deadline,100.);self.assertIsNone(e.result)
        events=[x['event'] for x in e.history]
        self.assertEqual(events.count('release_forward_correction_admitted'),1)
        self.assertEqual(events.count('release_forward_correction_emitted'),1)
        self.assertIsNone(e.release_memory.interrupt(o,e.request,e.travel_axis))

    def test_forward_guard_rejection_censors_without_consumption_or_output(self):
        e,o=self.token();self.braking(e)
        self.drive_admission(e,lambda command:'powered footprint blocked' if command.vx>.1 else '')
        self.assertEqual(e.result.outcome,'censored');self.assertFalse(e.result.physical)
        self.assertFalse(e.release_memory.used)
        self.assertFalse(any(h['event']=='release_forward_correction_emitted' for h in e.history))

    def test_missing_powered_evidence_keeps_bounded_brake_instead_of_zero_restoration(self):
        for kind in ('axis','witness','attitude','fan'):
            e,o=self.token();self.braking(e)
            if kind=='axis':e.release_memory.token['axis']=(0.,1.)
            if kind=='witness':e.release_memory.drive_witness=None
            if kind=='attitude':e.POSTURE_RELATIVE_ENVELOPE_DEG=-1.
            if kind=='fan':e.straight_entry_guard=None
            self.drive_admission(e)
            self.assertEqual(e.state,m.MovementState.BRAKE,kind)
            self.assertFalse(e.release_memory.used,kind)

    def test_token_survives_braking_posture_and_clears_on_stop_alignment_result(self):
        for kind in ('stop','alignment','result'):
            e,o=self.token();token=copy.deepcopy(e.release_memory.token)
            self.braking(e);e._state(m.MovementState.POSTURE_PAUSE,4.1,'fixture')
            e._state(m.MovementState.BRAKE,4.2,'fixture')
            self.assertEqual(e.release_memory.token,token);self.assertEqual(e.brake_episode_started,4.02)
            if kind=='stop':e.stop(obs(4.22))
            elif kind=='alignment':e._state(m.MovementState.ALIGN_LEFT,4.22,'actual alignment')
            else:e.finish('censored','fixture',obs(4.22))
            self.assertIsNone(e.release_memory.token);self.assertIsNone(e.release_memory.drive_witness)

    def test_stop_tip_deadline_stale_backslide_and_budget_precede_correction(self):
        for kind in ('stop','tip','deadline','stale','backslide','budget'):
            e,o=self.token();self.braking(e)
            if kind=='stop':e.stop(obs(4.04));continue
            o=obs(4.04)
            if kind=='tip':o=replace(o,up_z=.4)
            if kind=='deadline':o=replace(o,time=101.)
            if kind=='stale':o=replace(o,fresh=False)
            if kind=='backslide':
                e.uphill_samples=[(2.04,(.5,0.))];o=replace(o,xy=(0.,0.))
            if kind=='budget':e.uphill_exposure=12.
            e.tick(o,lambda cmd:'',lambda xy:True)
            self.assertIsNotNone(e.result,kind)
            self.assertFalse(e.release_memory.used,kind)
            self.assertFalse(any(h['event']=='release_forward_correction_emitted' for h in e.history),kind)

    def test_no_token_without_actual_overspeed_interruption(self):
        e=self.executor();o=self.history(e)
        self.assertIsNone(e.release_memory.token)
        e.state=m.MovementState.RELEASE;e.release_origin=(0.,0.)
        e.tick(replace(o,velocity=(.1,0.)),lambda cmd:'',lambda xy:True)
        self.assertIsNone(e.release_memory.token)

    def test_actual_interrupt_keeps_brake_command_then_freezes_response(self):
        e=self.executor();self.history(e)
        e.state=m.MovementState.RELEASE;e.entered=.02;e.release_origin=(0.,0.)
        command=e.tick(obs(4.04,-.005*4.04,vx=.51),lambda cmd:'',lambda xy:True)
        self.assertEqual(command,m.Motion(-.4,0.));self.assertEqual(e.state,m.MovementState.BRAKE)
        self.assertIsNotNone(e.release_memory.token);self.assertFalse(e.release_memory.used)
        self.assertEqual(sum(h['event']=='release_response_retained' for h in e.history),1)

    def test_characterization_configuration_and_protocol_are_explicit(self):
        from types import SimpleNamespace
        c=b.BehaviorTreeAscentController(ground(10),[SimpleNamespace(x=5.,y=0.,label='exit')],mode='characterize')
        with self.assertRaises(ValueError):c.configure_release_memory()
        c.configure_uphill_speed_response();c.configure_release_memory()
        self.assertTrue(c.executor.release_memory.enabled)
        self.assertEqual(c.protocol_version,'staged-teleop-v32-release-response-memory+purposeful-recovery-v1+bounded-recovery-turn-v1+protective-drive-resume-v1+owned-ascent-v1')
        with self.assertRaises(ValueError):c.configure_recovery_continuation()
        c.attempt_number=1
        with self.assertRaises(ValueError):c.configure_release_memory()
        c.close()


if __name__=='__main__':unittest.main()
