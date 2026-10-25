"""Stopping-envelope forecasts must retain passive yaw and terrain constraints."""
import math
import unittest

import numpy as np

from test_behavior_navigation import ground
from rexmi_bt_tests.nav.swept_footprint import PassiveHeadingWindow,oriented_stopping_reason,oriented_sweep_reason


class PassiveHeadingTests(unittest.TestCase):
    def window(self,fn):
        h=PassiveHeadingWindow()
        for t in np.arange(0.,.301,.02):h.observe(float(t),fn(t))
        return h

    def test_forecast_includes_history_and_future_in_both_directions(self):
        for sign in (-1,1):
            h=self.window(lambda t:sign*.2*t)
            low,high=h.offsets(.3,sign*.06,2.)
            self.assertAlmostEqual(low,-.06 if sign==1 else -.4)
            self.assertAlmostEqual(high,.4 if sign==1 else .06)

    def test_zero_net_oscillation_cannot_collapse_forecast(self):
        h=self.window(lambda t:.05*math.sin(2*math.pi*t/.3))
        low,high=h.offsets(.3,0.,2.)
        self.assertLess(low,-.049)
        self.assertGreater(high,.049)

    def test_heading_wrap_is_continuous(self):
        fn=lambda t:math.atan2(math.sin(math.pi+.2*t),math.cos(math.pi+.2*t))
        h=self.window(fn)
        np.testing.assert_allclose(h.offsets(.3,fn(.3),2.),(-.06,.4),atol=1e-12)

    def test_missing_sparse_stale_and_mismatched_history_are_unavailable(self):
        h=PassiveHeadingWindow()
        self.assertIsNone(h.offsets(.3,0.,2.))
        h.observe(0.,0.);h.observe(.3,0.)
        self.assertIsNone(h.offsets(.3,0.,2.))
        h=self.window(lambda t:0.)
        self.assertIsNone(h.offsets(.32,0.,2.))
        self.assertIsNone(h.offsets(.3,.1,2.))
        self.assertIsNone(h.offsets(.3,0.,math.inf))

    def test_large_forecast_is_not_clipped_and_time_restart_clears_history(self):
        h=self.window(lambda t:2.*t)
        self.assertIsNone(h.offsets(.3,.6,2.))
        h.observe(.1,0.)
        self.assertIsNone(h.offsets(.1,0.,2.))
        h.observe(.12,math.nan)
        self.assertEqual(h.samples,[])


class StoppingFootprintTests(unittest.TestCase):
    def grid(self):
        g=ground();g.half_length=.3;g.half_width=.08
        return g

    def test_predicted_body_rotation_and_centre_path_both_need_clearance(self):
        g=self.grid();g.hazard[g.world_to_cell(1.1,.5)]=True
        self.assertFalse(oriented_sweep_reason(g,(0.,0.),(1.,0.),0.))
        self.assertIn('obstacle',oriented_stopping_reason(g,(0.,0.),(1.,0.),0.,(0.,.5)))

    def test_reverse_keeps_body_pose_but_reverses_uncertain_travel(self):
        g=self.grid();g.hazard[g.world_to_cell(-1.1,-.5)]=True
        self.assertFalse(oriented_stopping_reason(g,(0.,0.),(1.,0.),0.,(0.,.5)))
        self.assertIn('obstacle',oriented_stopping_reason(g,(0.,0.),(-1.,0.),0.,(0.,.5)))

    def test_no_heading_history_does_not_become_exact_heading(self):
        g=self.grid()
        for offsets in (None,(.1,-.1),(0.,math.inf),(0.,4.)):
            self.assertTrue(oriented_stopping_reason(g,(0.,0.),(.5,0.),0.,offsets))

    def test_freshness_boundary_and_directional_exclusions_survive_fan(self):
        for kind in ('stale','slope','exclusion'):
            g=self.grid();cell=g.world_to_cell(.5,0.)
            if kind=='stale':g.fresh[cell]=False
            elif kind=='slope':g.slope[cell]=math.tan(math.radians(36.))
            else:g.mark_failure('failed',(.5,0.),0.,0.,'fixture')
            self.assertTrue(oriented_stopping_reason(g,(0.,0.),(.5,0.),0.,(-.02,.02)),kind)

    def test_incidental_sway_does_not_acquire_commanded_cross_grade(self):
        g=ground(30.);g.uphill_limit_deg=33.;g.side_limit_deg=25.
        self.assertTrue(oriented_stopping_reason(g,(0.,0.),(0.,.5),0.,(-.02,.02)))
        self.assertFalse(oriented_stopping_reason(g,(0.,0.),(0.,.5),0.,(-.02,.02),commanded=False))


if __name__=='__main__':unittest.main()
