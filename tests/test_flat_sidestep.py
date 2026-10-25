"""Guarded handoff and lateral-command checks, without Isaac Sim."""
import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
from test_checkpoint_navigation import surface,module,c as checkpoint
f=module('flat_sidestep','nav/flat_sidestep.py')


class FlatSwitchTests(unittest.TestCase):
    def setup_case(self):
        g=surface();s=f.FlatSidestep(g)
        c=SimpleNamespace(state='FOLLOW',target=(0.,2.),escape_clearance=None,request=Mock())
        return g,s,c

    def switch_in(self,g,s,c):
        self.assertTrue(s.maybe_start(c,(0,0),0,0,1.))
        for t in (0.,.3,.6):s.tick(c,(0,0),0,(0,0),0,0,t)
        self.assertEqual(s.policy,'rough')
        self.assertEqual(s.switches,1)

    def test_stable_stop_then_sidestep_then_stable_return(self):
        g,s,c=self.setup_case();self.switch_in(g,s,c)
        motion=s.tick(c,(0,0),0,(0,0),0,0,.8)
        self.assertGreater(motion.vy,0);self.assertLessEqual(motion.vy,.15)
        self.assertAlmostEqual(motion.omega,0)
        s.tick(c,(0,.75),0,(0,.1),0,0,1.)
        self.assertEqual(s.phase,'STOP_ROUGH');self.assertEqual(s.policy,'rough')
        for t in (1.2,1.5,1.8):s.tick(c,(0,.75),0,(0,0),0,0,t)
        self.assertEqual(s.policy,'rocky_slope');self.assertEqual(s.switches,2)
        c.request.assert_called_once_with('resume',1.8)
        self.assertFalse(s.maybe_start(c,(0,.75),0,2.,1.))

    def test_slope_anywhere_in_footprint_or_reserve_blocks_entry(self):
        for point in ((0,0),(.8,.5),(0,1.4)):
            g,s,c=self.setup_case();g.slope[g.world_to_cell(*point)]=math.tan(math.radians(5.1))
            self.assertFalse(s.maybe_start(c,(0,0),0,0,1.))
            self.assertEqual(s.policy,'rocky_slope');self.assertEqual(s.switches,0)

    def test_unknown_stale_obstacle_and_body_tilt_block_entry(self):
        for attr,val in [('known',False),('hazard',True),('surface_seen',-6.)]:
            g,s,c=self.setup_case();getattr(g,attr)[g.world_to_cell(.5,.5)]=val
            self.assertFalse(s.maybe_start(c,(0,0),0,0,1.))
        g,s,c=self.setup_case()
        self.assertFalse(s.maybe_start(c,(0,0),0,0,math.cos(math.radians(9))))

    def test_motion_and_rotation_reset_stability_timer(self):
        g,s,c=self.setup_case();self.assertTrue(s.maybe_start(c,(0,0),0,0,1.))
        for t,v,w in [(0.,(0,0),0),(.4,(.06,0),0),(.8,(0,0),.2),(1.,(0,0),0),(1.4,(0,0),0)]:
            s.tick(c,(0,0),0,v,w,abs(w),t)
            self.assertEqual(s.policy,'rocky_slope')
        s.tick(c,(0,0),0,(0,0),0,0,1.6)
        self.assertEqual(s.policy,'rough')

    def test_slope_discovery_brakes_without_switching_on_slope(self):
        g,s,c=self.setup_case();self.switch_in(g,s,c)
        g.slope[g.world_to_cell(0,0)]=math.tan(math.radians(10))
        for t in (.8,1.5,2.5):s.tick(c,(0,0),0,(0,0),0,0,t)
        self.assertEqual(s.phase,'STOP_ROUGH');self.assertEqual(s.policy,'rough')
        self.assertEqual(s.switches,1);self.assertIn('handoff blocked',s.reason)
        g.slope[:]=0
        for t in (3.,3.6):s.tick(c,(0,0),0,(0,0),0,0,t)
        self.assertEqual(s.policy,'rocky_slope')

    def test_operator_stop_never_resumes_autonomy(self):
        g,s,c=self.setup_case();self.switch_in(g,s,c)
        c.state='HOLD';s.cancel('operator stop')
        for t in (.8,1.4):s.tick(c,(0,0),0,(0,0),0,0,t)
        self.assertEqual(s.policy,'rocky_slope');c.request.assert_not_called()

    def test_switch_cancelled_when_ground_changes_before_entry(self):
        g,s,c=self.setup_case();s.maybe_start(c,(0,0),0,0,1.)
        s.tick(c,(0,0),0,(0,0),0,0,0)
        g.slope[g.world_to_cell(0,.8)]=1.
        s.tick(c,(0,0),0,(0,0),0,0,.6)
        self.assertEqual(s.policy,'rocky_slope');self.assertFalse(s.active)

    def test_right_command_and_heading_correction(self):
        g,s,c=self.setup_case();c.target=(0,-2.);self.switch_in(g,s,c)
        command=s.tick(c,(0,0),.1,(.05,0),0,0,.8)
        self.assertLess(command.vy,0);self.assertLess(command.omega,0)

    def test_excessive_speed_brakes_current_policy(self):
        g,s,c=self.setup_case();self.switch_in(g,s,c)
        command=s.tick(c,(0,0),0,(0,.4),0,0,.8)
        self.assertEqual(s.phase,'STOP_ROUGH');self.assertLess(command.vy,0)

    def test_vertical_motion_blocks_switch_and_events_are_serializable(self):
        import json
        g,s,c=self.setup_case();s.maybe_start(c,(0,0),0,0,1.)
        for t in (0.,.6):s.tick(c,(0,0),0,(0,0),0,0,t,vertical_speed=.1)
        self.assertEqual(s.switches,0)
        for t in (1.,1.6):s.tick(c,np.array([0,0],np.float32),0,(0,0),0,0,t)
        self.assertEqual(s.switches,1)
        json.dumps(s.events)
        self.assertLessEqual(s.events[0]['measured_max_slope_deg'],5.)

    def test_exact_slope_boundary_and_stale_return_handoff(self):
        g,s,c=self.setup_case();g.slope[:]=math.tan(math.radians(5.))
        self.switch_in(g,s,c)
        s.cancel('operator stop');c.state='HOLD';g.surface_seen[:]=-6.
        for t in (1.,2.):s.tick(c,(0,0),0,(0,0),0,0,t)
        self.assertEqual(s.policy,'rough');self.assertEqual(s.switches,1)
        self.assertIn('fresh',s.reason)
        g.surface_seen[:]=0
        for t in (2.2,2.8):s.tick(c,(0,0),0,(0,0),0,0,t)
        self.assertEqual(s.policy,'rocky_slope')

    def test_new_obstacle_ends_rough_motion(self):
        g,s,c=self.setup_case();self.switch_in(g,s,c)
        g.hazard[g.world_to_cell(.6,.5)]=True
        command=s.tick(c,(0,0),0,(0,.15),0,0,.8)
        self.assertEqual(s.phase,'STOP_ROUGH');self.assertLessEqual(command.vy,0.)
        self.assertEqual(s.policy,'rough')

    def test_handoff_near_mission_goal_verifies_arrival_without_turning_away(self):
        g=surface();s=f.FlatSidestep(g)
        c=checkpoint.CheckpointController(g,[SimpleNamespace(x=0.,y=1.)])
        c.state='FOLLOW';c.target=(0.,1.)
        self.switch_in(g,s,c)
        s.phase='STOP_ROUGH';s.anchor=np.array([0.,.7])
        for t in (1.,1.6):s.tick(c,(0,.7),0,(0,0),0,0,t)
        self.assertEqual(s.policy,'rocky_slope')
        self.assertEqual(c.state,'SETTLE');self.assertTrue(c.pending_arrival)
        self.assertEqual(c.wp_idx,0)
        c.tick((0,.7),0,0,1.7);c.tick((0,.7),0,0,2.3)
        self.assertEqual(c.wp_idx,1)

    def test_cream_and_operator_states_do_not_start(self):
        g,s,c=self.setup_case();c.escape_clearance=.3
        self.assertFalse(s.maybe_start(c,(0,0),0,0,1.))
        c.escape_clearance=None
        for state in ('HOLD','MANUAL','OBSERVE','COMPLETE'):
            c.state=state;self.assertFalse(s.maybe_start(c,(0,0),0,0,1.))

if __name__=='__main__':unittest.main()
