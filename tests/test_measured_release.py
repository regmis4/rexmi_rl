"""Select zero input from measured powered response, never a timer alone."""
from dataclasses import replace
import math
import unittest

from test_behavior_navigation import m,observation


class MeasuredReleaseTests(unittest.TestCase):
    def executor(self):
        e=m.MovementExecutor();e.release_guard=e.posture_guard=e.corridor_guard=lambda *args:''
        e.start(m.MovementRequest('one',(0.,0.),(20.,0.),1,.4,100.,lateral_tolerance=1.),0.)
        e.committed_heading=0.;e.travel_axis=(1.,0.)
        e._state(m.MovementState.DRIVE_FORWARD,0.,'fixture')
        return e

    def obs(self,t,vx=.05,vy=0.,gradient=(.5,0.),**kw):
        return replace(observation(t,xy=(vx*t,vy*t),velocity=(vx,vy),grade=30.,turn=False,**kw),
                       terrain_gradient=gradient)

    def tick(self,e,obs,guard=lambda command:''):
        return e.tick(obs,guard,lambda xy:True)

    def test_productive_climb_continues_beyond_five_seconds(self):
        e=self.executor();req=e.request
        for i in range(751):
            self.assertEqual(self.tick(e,self.obs(i*.02,vx=.3)),m.Motion(.4,0.))
        self.assertIs(e.request,req);self.assertEqual(e.state,m.MovementState.DRIVE_FORWARD)
        self.assertFalse(any(h['event']=='drive_release_selected' for h in e.history))
        self.assertAlmostEqual(e.drive_exposure,15.)

    def test_slow_drive_waits_for_exact_minimum_exposure_and_records_evidence(self):
        e=self.executor();req=e.request
        for i in range(250):self.assertEqual(self.tick(e,self.obs(i*.02)),m.Motion(.4,0.))
        self.assertEqual(self.tick(e,self.obs(5.)),m.Motion())
        self.assertEqual(e.state,m.MovementState.RELEASE);self.assertIs(e.request,req)
        event=next(h for h in e.history if h['event']=='drive_release_selected')
        self.assertAlmostEqual(event['forward_exposure_s'],5.)
        self.assertGreaterEqual(event['response_window_s'],2.-1e-9)
        self.assertGreaterEqual(event['low_progress_dwell_s'],1.)
        self.assertAlmostEqual(event['axis_progress_mps'],.05)

    def test_either_deficient_projection_selects_release(self):
        for vx,vy in ((.09,.09),(.12,0.)):
            e=self.executor()
            for i in range(251):self.tick(e,self.obs(i*.02,vx=vx,vy=vy,gradient=(.3,.5)))
            self.assertEqual(e.state,m.MovementState.RELEASE)

    def test_exact_threshold_and_dwell_boundaries(self):
        e=self.executor();e.drive_exposure=5.
        e.drive_response_samples=[(0.,(0.,0.)),(2.,(.2,0.))]
        self.assertFalse(e._drive_release_ready(self.obs(2.,vx=.1),True))
        self.assertIsNone(e.drive_slow_since)
        for t in (2.,2.5,2.98,3.):
            e.drive_response_samples=[(t-2.,(.05*(t-2.),0.)),(t,(.05*t,0.))]
            self.assertEqual(e._drive_release_ready(self.obs(t),True),t==3.)
        e.drive_response_samples=[(3.,(.3,0.)),(5.,(.5,0.))]
        self.assertFalse(e._drive_release_ready(self.obs(5.,vx=.1),True))
        self.assertIsNone(e.drive_slow_since)

    def test_gap_invalid_gradient_and_braking_clear_short_history(self):
        for case in ('gap','gradient','brake'):
            e=self.executor()
            for i in range(246):self.tick(e,self.obs(i*.02))
            self.assertIsNotNone(e.drive_slow_since)
            if case=='gap':self.tick(e,self.obs(5.1))
            elif case=='gradient':self.tick(e,self.obs(4.92,gradient=None))
            else:self.tick(e,replace(self.obs(4.92),velocity=(.6,0.)))
            self.assertIsNone(e.drive_slow_since,case)
            self.assertFalse(e.drive_response_samples,case)
            self.assertFalse(any(h['event']=='drive_release_selected' for h in e.history),case)

    def test_labels_and_positive_damping_cannot_supply_powered_window(self):
        e=self.executor();e._state(m.MovementState.BRAKE,0.,'fixture')
        for i in range(151):
            self.tick(e,self.obs(i*.02))
            if e.state==m.MovementState.DRIVE_FORWARD:break
        self.assertFalse(e.drive_response_samples)
        self.assertIsNone(e.drive_slow_since)
        self.assertEqual(e.drive_exposure,0.)

    def test_release_correction_cannot_inherit_slow_trigger(self):
        e=self.executor()
        for i in range(251):self.tick(e,self.obs(i*.02))
        self.assertEqual(e.state,m.MovementState.RELEASE)
        self.tick(e,self.obs(5.02,vx=.01))
        self.assertEqual(e.state,m.MovementState.DRIVE_FORWARD)
        self.assertIsNone(e.drive_slow_since);self.assertFalse(e.drive_response_samples)
        self.assertEqual(e.drive_exposure,0.)

    def test_protections_precede_new_release_entry(self):
        for case in ('backslide','posture','overspeed','geometry','budget','stale','stop','deadline'):
            e=self.executor()
            for i in range(250):self.tick(e,self.obs(i*.02))
            obs=self.obs(5.)
            if case=='backslide':obs=replace(obs,xy=(-1.,0.))
            if case=='posture':obs=replace(obs,up_z=math.cos(math.radians(50.)))
            if case=='overspeed':obs=replace(obs,velocity=(.6,0.))
            if case=='geometry':e.release_guard=lambda *args:'blocked release'
            if case=='budget':
                e.uphill_stall_since=0.;e.uphill_stall_origin=(1.,0.);e.uphill_exposure=12.
            if case=='stale':obs=replace(obs,fresh=False)
            if case=='stop':e.stop(obs)
            if case=='deadline':e.request=replace(e.request,deadline=4.99)
            self.tick(e,obs,lambda command:'blocked drive' if case=='geometry' else '')
            self.assertNotEqual(e.state,m.MovementState.RELEASE,case)
            self.assertFalse(any(h['event']=='drive_release_selected' for h in e.history),case)
            if case in ('posture','overspeed','stop','stale','deadline'):
                self.assertIsNone(e.drive_slow_since,case)
                self.assertFalse(e.drive_response_samples,case)


if __name__=='__main__':unittest.main()
