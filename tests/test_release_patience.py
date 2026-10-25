"""Observe slow useful release; keep correction and failure decisions bounded."""
from dataclasses import replace
import math
import unittest

import numpy as np

from test_behavior_navigation import m,observation


class ReleasePatienceTests(unittest.TestCase):
    def executor(self):
        e=m.MovementExecutor();e.release_guard=e.posture_guard=e.corridor_guard=lambda *args:''
        e.start(m.MovementRequest('one',(0.,0.),(10.,0.),1,.4,100.),0.)
        e.committed_heading=0.;e.travel_axis=(1.,0.);e.release_origin=(0.,0.)
        e._state(m.MovementState.RELEASE,0.,'fixture')
        return e

    def obs(self,t,xy=(0.,0.),gradient=(.5,0.),**kw):
        return replace(observation(t,xy=xy,grade=30.,turn=False,**kw),terrain_gradient=gradient)

    def tick(self,e,obs,guard=None):
        return e.tick(obs,guard or (lambda command:''),lambda xy:True)

    def test_full_two_seconds_then_exact_five_second_plateau(self):
        e=self.executor();request=e.request;axis=e.travel_axis
        for i in range(350):
            self.assertEqual(self.tick(e,self.obs(i*.02)),m.Motion())
            if i<100:self.assertIsNone(e.release_bad_since)
        self.assertAlmostEqual(e.release_bad_since,2.)
        self.assertEqual(self.tick(e,self.obs(7.)),m.Motion(.4,0.))
        self.assertIs(e.request,request);self.assertEqual(e.travel_axis,axis)
        event=next(x for x in e.history if x.get('event')=='release_correction_emitted')
        self.assertEqual(event['reason'],'sustained measured plateau');self.assertAlmostEqual(event['plateau_s'],5.)
        self.assertIsNone(e.result)

    def test_either_deficient_projection_accumulates_plateau(self):
        for vx,vy in ((.005,.03),(.015,-.005)):
            e=self.executor()
            for i in range(351):
                t=i*.02;cmd=self.tick(e,self.obs(t,xy=(vx*t,vy*t),gradient=(.3,.5)))
            self.assertEqual(cmd,m.Motion(.4,0.))
            self.assertEqual(e.state,m.MovementState.DRIVE_FORWARD)

    def test_both_slow_positive_rates_preserve_release(self):
        e=self.executor()
        for t in np.arange(0.,20.,.02):
            self.assertEqual(self.tick(e,self.obs(float(t),xy=(.015*t,0.),velocity=(.015,0.))),m.Motion())
            self.assertIsNone(e.release_bad_since);self.assertIsNone(e.result)
        self.assertEqual(e.state,m.MovementState.RELEASE)

    def test_brake_restoration_needs_new_full_window_without_outer_renewal(self):
        e=self.executor()
        for i in range(201):self.tick(e,self.obs(i*.02))
        origin=e.uphill_stall_origin;started=e.uphill_stall_since;exposure=e.uphill_exposure
        self.tick(e,self.obs(4.02,velocity=(.6,0.)))
        for i in range(202,220):self.tick(e,self.obs(i*.02))
        self.assertEqual(e.state,m.MovementState.RELEASE)
        self.assertEqual(e.uphill_stall_origin,origin);self.assertEqual(e.uphill_stall_since,started)
        self.assertGreaterEqual(e.uphill_exposure,exposure)
        for i in range(220,310):
            self.tick(e,self.obs(i*.02));self.assertIsNone(e.release_bad_since)
        self.assertIsNone(e.result)

    def test_outer_budget_can_end_release_before_plateau_correction(self):
        e=self.executor();e.uphill_patience=True;e.uphill_stall_since=-22.;e.uphill_stall_origin=(10.,0.)
        e.uphill_exposure=1.
        for i in range(101):
            self.tick(e,self.obs(i*.02))
            if e.result:break
        self.assertTrue(e.result.physical);self.assertIn('wall budget',e.result.reason)
        self.assertFalse(any(x.get('event')=='release_correction_emitted' for x in e.history))

    def test_immediate_net_loss_and_eligibility_loss_still_correct(self):
        for case in ('net_loss','gradient_loss'):
            e=self.executor()
            obs=self.obs(.02,xy=(-.081,0.)) if case=='net_loss' else self.obs(.02,gradient=None)
            self.assertEqual(self.tick(e,obs),m.Motion(.4,0.))
            self.assertEqual(e.state,m.MovementState.DRIVE_FORWARD)

    def test_independent_protections_precede_additional_observation(self):
        for case in ('stop','tip','posture','stale','overspeed','backslide','geometry','deadline'):
            e=self.executor();obs=self.obs(.6)
            if case=='stop':e.stop(obs);self.assertEqual(e.state,m.MovementState.STOP);continue
            if case=='tip':obs=replace(obs,up_z=.4)
            if case=='posture':obs=replace(obs,up_z=math.cos(math.radians(50.)))
            if case=='stale':obs=replace(obs,fresh=False)
            if case=='overspeed':obs=replace(obs,velocity=(.6,0.))
            if case=='backslide':e.uphill_samples=[(-1.,(.5,0.))]
            if case=='geometry':e.release_guard=lambda *args:'geometry changed'
            if case=='deadline':e.request=replace(e.request,deadline=.5)
            self.tick(e,obs)
            self.assertNotEqual(e.state,m.MovementState.RELEASE,case)
            self.assertFalse(any(x.get('event')=='release_correction_emitted' for x in e.history),case)


if __name__=='__main__':unittest.main()
