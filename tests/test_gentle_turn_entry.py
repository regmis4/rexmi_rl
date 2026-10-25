"""Guarded entry in the previously unreachable 12–15 degree turn band."""
import copy
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import test_behavior_navigation as bt


class GentleTurnTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.grid=bt.ground(13.44)

    def control(self, enabled=True):
        c=bt.b.BehaviorTreeAscentController(copy.deepcopy(self.grid),
            [SimpleNamespace(x=-3.,y=0.,label='goal')],mode='characterize')
        if enabled:c.configure_gentle_turn_entry()
        return c

    def sample(self,c,t,**kwargs):
        values=dict(xy=(0,0),fresh=True,inclination=13.44,tilt=10.58,speed=.03,
                    yaw_rate=0.,now=t,dt=.02)
        values.update(kwargs);c._update_turn_permission(**values)

    def test_initial_route_deadlock_resolved_without_changing_default(self):
        c=self.control(False);yaw=math.radians(169.47)
        path,done=bt.r.observed_route(c.grid,(0,0),(-3,0),yaw,allow_frontier=False)
        self.assertFalse(done);self.assertEqual(c.grid.last_route_search['expansions'],1)
        c.configure_gentle_turn_entry()
        path,done=bt.r.observed_route(c.grid,(0,0),(-3,0),yaw,allow_frontier=False)
        self.assertTrue(done);self.assertGreater(len(path),1)
        self.assertIn('+gentle-turn-entry-v1',c.protocol_version)

    def test_stable_dwell_and_existing_hard_exit(self):
        c=self.control()
        for i in range(30):self.sample(c,i*.02)
        self.assertFalse(c.turn_enabled)
        self.sample(c,.6);self.assertTrue(c.turn_enabled)
        self.sample(c,.62,yaw_rate=.35) # Turning does not revoke an already armed permission.
        self.assertTrue(c.turn_enabled)
        self.sample(c,.64,inclination=15.01);self.assertFalse(c.turn_enabled)

    def test_instability_and_gaps_restart_unearned_dwell(self):
        for change in ({'fresh':False},{'tilt':21.},{'speed':.2},{'yaw_rate':.2},{'dt':.2}):
            c=self.control()
            for i in range(25):self.sample(c,i*.02)
            self.sample(c,.5,**change)
            for i in range(26,40):self.sample(c,i*.02)
            self.assertFalse(c.turn_enabled,change)

    def test_blocked_rotation_cannot_arm(self):
        c=self.control()
        with patch.object(c.grid,'segment_clear',return_value=False):
            for i in range(40):self.sample(c,i*.02)
        self.assertFalse(c.turn_enabled)

    def test_gap_revokes_already_armed_permission_even_if_unstable(self):
        for dt in (0., -.02, .2):
            c=self.control()
            for i in range(31):self.sample(c,i*.02)
            self.assertTrue(c.turn_enabled)
            self.sample(c,.8,dt=dt,speed=.2)
            self.assertFalse(c.turn_enabled);self.assertIsNone(c.gentle_turn_since)
            for i in range(41,60):self.sample(c,i*.02)
            self.assertFalse(c.turn_enabled)

    def test_stop_manual_and_invalid_observation_reset(self):
        for action in ('stop','manual','resume'):
            c=self.control()
            for i in range(31):self.sample(c,i*.02)
            c.request(action,.62)
            self.assertFalse(c.turn_enabled);self.assertIsNone(c.gentle_turn_since)
        c=self.control();c.turn_enabled=True;c.gentle_turn_since=0.
        c.tick((math.nan,0),0,0,1.)
        self.assertFalse(c.turn_enabled);self.assertIsNone(c.gentle_turn_since)

    def test_default_band_and_low_grade_behavior_preserved(self):
        c=self.control(False)
        for i in range(40):self.sample(c,i*.02)
        self.assertFalse(c.turn_enabled)
        self.sample(c,1.,inclination=12.)
        self.assertTrue(c.turn_enabled)
        c.mode='qualified'
        with self.assertRaises(ValueError):c.configure_gentle_turn_entry()


if __name__=='__main__':unittest.main()
