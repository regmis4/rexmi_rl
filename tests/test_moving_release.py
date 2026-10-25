"""Moving RELEASE is a checked continuation, never a new direction or turn."""
from dataclasses import replace
import math
import unittest

import numpy as np

from test_behavior_navigation import m,observation


class MovingReleaseTests(unittest.TestCase):
    def executor(self):
        e=m.MovementExecutor()
        e.release_guard=e.posture_guard=e.corridor_guard=e.straight_entry_guard=lambda *args:''
        e.start(m.MovementRequest('uphill',(0.,0.),(10.,0.),1,.4,100.),0.)
        e.committed_heading=0.;e.travel_axis=(1.,0.);e.release_origin=(0.,0.)
        e.suspended_release=True
        return e

    def obs(self,t,dy=0.):
        return replace(observation(t,xy=(.08*t,dy*t),yaw=.018*math.sin(10*t),
            velocity=(.08,dy),grade=30.,turn=False,yaw_rate=.18*math.cos(10*t)),
            terrain_gradient=(.25,.5))

    def tick(self,e,obs):
        return e.tick(obs,lambda command:'',lambda xy:True)

    def advance(self,e,end=2.04,dy=0.,step=.02):
        for t in np.arange(0.,end+1e-9,step):
            cmd=self.tick(e,self.obs(float(t),dy))
            if e.result:break
        return cmd

    def test_bounded_useful_motion_restores_exact_zero_without_changing_request(self):
        e=self.executor();request=e.request;self.advance(e,1.98)
        self.assertEqual(e.state,m.MovementState.BRAKE)
        self.assertEqual(self.tick(e,self.obs(2.)),m.Motion())
        self.assertEqual(e.state,m.MovementState.RELEASE)
        self.assertIs(e.request,request);self.assertEqual(e.started,0.)
        self.assertEqual(e.committed_heading,0.);self.assertEqual(e.travel_axis,(1.,0.))
        event=next(x for x in e.history if x.get('event')=='moving_release_admitted')
        self.assertEqual(event['time'],2.)
        self.assertGreater(event['uphill_progress_mps'],.02)
        self.assertIsNone(e.result)

    def test_positive_axis_displacement_cannot_hide_actual_downhill_motion(self):
        e=self.executor();self.advance(e,dy=-.1)
        self.assertGreater(e.metrics['release_entry_axis_progress_mps'],.02)
        self.assertLess(e.metrics['release_entry_uphill_progress_mps'],0.)
        self.assertEqual(e.state,m.MovementState.BRAKE)
        self.assertTrue(e.suspended_release)

    def test_sparse_and_nonfinite_evidence_cannot_admit(self):
        e=self.executor();self.advance(e,step=.1)
        self.assertEqual(e.state,m.MovementState.BRAKE)
        e=self.executor();self.advance(e,1.98)
        obs=self.obs(1.98)
        for field in ('up_z','yaw','time'):
            self.assertFalse(e._moving_release_admission(replace(obs,**{field:math.nan}),e.request,0.)[0])
        for field in ('xy','terrain_gradient'):
            self.assertFalse(e._moving_release_admission(replace(obs,**{field:(math.nan,0.)}),e.request,0.)[0])
        e.entry_samples[-1]=(math.nan,0.,0.,0.)
        self.assertIsNone(e._straight_motion_window(obs))

    def test_every_geometry_layer_remains_binding_and_censored(self):
        for name in ('straight_entry_guard','corridor_guard','release_guard'):
            e=self.executor();setattr(e,name,lambda *args:'blocked '+name)
            self.advance(e)
            self.assertEqual(e.result.outcome,'censored',name)
            self.assertFalse(e.result.physical,name)
            self.assertFalse(any(x.get('event')=='moving_release_admitted' for x in e.history))

    def test_ordinary_drive_heading_is_not_relaxed(self):
        e=self.executor();e.suspended_release=False;e.committed_heading=.4
        self.advance(e,.5)
        self.assertEqual(e.state,m.MovementState.BRAKE)
        self.assertFalse(any(x.get('event')=='moving_release_admitted' for x in e.history))

    def test_overspeed_cycles_preserve_shared_uphill_anchor_and_deadline(self):
        e=self.executor();self.advance(e);request=e.request
        e.uphill_stall_since=1.;e.uphill_stall_origin=(100.,0.)
        for index in range(103,156):
            obs=self.obs(index*.02)
            self.tick(e,replace(obs,velocity=(.6,0.)) if index in (103,129) else obs)
            self.assertEqual(e.uphill_stall_since,1.)
            self.assertEqual(e.uphill_stall_origin,(100.,0.))
            self.assertIs(e.request,request);self.assertEqual(request.deadline,100.)
        self.assertEqual(e.state,m.MovementState.RELEASE)
        self.assertEqual(sum(x.get('event')=='moving_release_admitted' for x in e.history),3)

    def test_stop_posture_deadline_tip_stale_and_backslide_precede_admission(self):
        for case in ('stop','posture','deadline','tip','stale','backslide'):
            e=self.executor();self.advance(e,1.98);obs=self.obs(2.)
            if case=='stop':
                self.assertEqual(e.stop(obs),m.Motion())
                self.assertEqual(self.tick(e,obs),m.Motion())
            elif case=='posture':
                self.tick(e,replace(obs,up_z=math.cos(math.radians(45.))))
                self.assertIsNotNone(e.posture_pause)
            elif case=='deadline':self.tick(e,self.obs(101.))
            elif case=='tip':self.tick(e,replace(obs,up_z=.4))
            elif case=='stale':self.tick(e,replace(obs,fresh=False))
            else:self.tick(e,replace(obs,xy=(-.5,0.)))
            self.assertNotEqual(e.state,m.MovementState.RELEASE,case)
            self.assertFalse(any(x.get('event')=='moving_release_admitted' for x in e.history),case)

    def test_full_two_second_history_is_required_even_with_recent_stable_motion(self):
        e=self.executor();self.advance(e,1.98)
        e.uphill_samples=[s for i,s in enumerate(e.uphill_samples) if i%5==0]
        self.tick(e,self.obs(2.))
        self.assertEqual(e.state,m.MovementState.BRAKE)


if __name__=='__main__':unittest.main()
