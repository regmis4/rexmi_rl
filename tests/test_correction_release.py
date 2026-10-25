"""The response-memory correction gets one measured return-to-zero opportunity."""
from dataclasses import replace
import unittest

import test_measured_release as measured
import test_release_memory as memory
from test_behavior_navigation import m,b,ground


class CorrectionReleaseTests(unittest.TestCase):
    obs=measured.MeasuredReleaseTests.obs
    tick=measured.MeasuredReleaseTests.tick

    def executor(self,enabled=True):
        e=measured.MeasuredReleaseTests.executor(self)
        e.release_memory.enabled=True
        e.release_memory.correction_release_enabled=enabled
        # The production correction branch consumes only after command guard.
        e.release_memory.consume()
        return e

    def test_responds_at_three_seconds_only_when_the_full_window_and_dwell_qualify(self):
        e=self.executor();req=e.request;axis=e.travel_axis
        for i in range(150):self.assertEqual(self.tick(e,self.obs(i*.02)),m.Motion(.4,0.))
        self.assertEqual(self.tick(e,self.obs(3.)),m.Motion())
        self.assertIs(e.request,req);self.assertIs(e.travel_axis,axis)
        self.assertTrue(e.release_memory.used);self.assertFalse(e.release_memory.correction_active)
        event=next(h for h in e.history if h['event']=='drive_release_selected')
        self.assertTrue(event['correction_episode']);self.assertAlmostEqual(event['forward_exposure_s'],3.)
        self.assertAlmostEqual(event['response_window_s'],2.);self.assertAlmostEqual(event['low_progress_dwell_s'],1.)

    def test_disabled_correction_and_ordinary_drive_keep_five_seconds(self):
        for kind in ('disabled','ordinary'):
            e=self.executor(enabled=kind!='disabled')
            if kind=='ordinary':e.release_memory.clear(new_request=True)
            for i in range(250):self.assertEqual(self.tick(e,self.obs(i*.02)),m.Motion(.4,0.))
            self.assertEqual(self.tick(e,self.obs(5.)),m.Motion())

    def test_productive_correction_continues_without_a_three_second_timer(self):
        e=self.executor()
        for i in range(351):self.assertEqual(self.tick(e,self.obs(i*.02,vx=.3)),m.Motion(.4,0.))
        self.assertTrue(e.release_memory.correction_active)

    def test_protective_time_does_not_supply_powered_response_or_exposure(self):
        e=self.executor();req=e.request
        for i in range(100):self.tick(e,self.obs(i*.02))
        before=e.drive_exposure
        self.tick(e,replace(self.obs(2.),velocity=(.6,0.)))
        self.assertTrue(e.release_memory.correction_active)
        e._state(m.MovementState.POSTURE_PAUSE,2.1,'fixture')
        e._state(m.MovementState.BRAKE,2.2,'fixture')
        self.assertTrue(e.release_memory.correction_active);self.assertIs(e.request,req)
        self.assertAlmostEqual(e.drive_exposure,before+.02)
        self.assertFalse(e.drive_response_samples);self.assertIsNone(e.drive_slow_since)

    def test_guarded_release_failure_does_not_retire_opportunity(self):
        e=self.executor();e.release_guard=lambda *_:'release footprint blocked'
        for i in range(151):command=self.tick(e,self.obs(i*.02))
        self.assertEqual(command,m.Motion(.4,0.))
        self.assertTrue(e.release_memory.correction_active);self.assertEqual(e.state,m.MovementState.DRIVE_FORWARD)

    def test_protections_and_missing_history_precede_three_second_return(self):
        for kind in ('backslide','stale','tip','budget','deadline','gap','missing_gradient','overspeed'):
            e=self.executor()
            for i in range(150):self.tick(e,self.obs(i*.02))
            o=self.obs(3.)
            if kind=='backslide':o=replace(o,xy=(-1.,0.))
            if kind=='stale':o=replace(o,fresh=False)
            if kind=='tip':o=replace(o,up_z=.4)
            if kind=='budget':
                e.uphill_stall_since=0.;e.uphill_stall_origin=(1.,0.);e.uphill_exposure=12.
            if kind=='deadline':e.request=replace(e.request,deadline=2.99)
            if kind=='gap':o=replace(o,time=3.10)
            if kind=='missing_gradient':o=replace(o,terrain_gradient=None)
            if kind=='overspeed':o=replace(o,velocity=(.6,0.))
            self.tick(e,o)
            self.assertNotEqual(e.state,m.MovementState.RELEASE,kind)
            self.assertFalse(any(h['event']=='drive_release_selected' for h in e.history),kind)

    def test_stop_result_alignment_new_request_retire_correction(self):
        for state in (m.MovementState.STOP,m.MovementState.IDLE,m.MovementState.ALIGN_LEFT,m.MovementState.SETTLE):
            e=self.executor();e._state(state,0.,'fixture')
            self.assertFalse(e.release_memory.correction_active)
            self.assertTrue(e.release_memory.used)
        e=self.executor();e.finish('censored','fixture',self.obs(0.));e.consume()
        e.start(m.MovementRequest('new',(0.,0.),(20.,0.),1,.4,100.),0.)
        self.assertFalse(e.release_memory.correction_active);self.assertFalse(e.release_memory.used)

    def test_real_correction_emission_enables_and_first_return_retires_privilege(self):
        fixture=memory.ReleaseMemoryTests()
        e,o=fixture.token();e.release_memory.correction_release_enabled=True
        fixture.braking(e);fixture.drive_admission(e)
        self.assertTrue(e.release_memory.correction_active)
        e._state(m.MovementState.RELEASE,6.2,'fixture')
        e._emit_traversal(m.Motion(),memory.obs(6.2))
        self.assertFalse(e.release_memory.correction_active)
        e._state(m.MovementState.DRIVE_FORWARD,6.22,'later ordinary correction')
        self.assertFalse(e.release_memory.correction_active)

    def test_configuration_requires_memory_and_keeps_protocol_on_repeated_configuration(self):
        from types import SimpleNamespace
        c=b.BehaviorTreeAscentController(ground(10),[SimpleNamespace(x=5.,y=0.,label='exit')],mode='characterize')
        with self.assertRaises(ValueError):c.configure_correction_release()
        c.configure_uphill_speed_response();c.configure_release_memory();c.configure_correction_release()
        c.configure_uphill_speed_response();c.configure_release_memory()
        self.assertEqual(c.protocol_version,'staged-teleop-v33-correction-release+purposeful-recovery-v1+bounded-recovery-turn-v1+protective-drive-resume-v1+owned-ascent-v1')
        c.attempt_number=1
        with self.assertRaises(ValueError):c.configure_correction_release()
        c.close()


if __name__=='__main__':unittest.main()
