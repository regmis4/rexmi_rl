"""Exact destinations on slopes must not require an artificial final pivot."""
import copy
import math
import unittest

import numpy as np

from test_behavior_navigation import ground
from rexmi_bt_tests.nav.ascent_routes import observed_route
from rexmi_bt_tests.nav.staged_routes import motion_route_reason, observed_edge_reason


class TerminalConnectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture=ground()
        g=cls.fixture;g.motion_turn_limit=12.;g.uphill_limit_deg=33.;g.side_limit_deg=25.
        # Gentle approach to a 13-degree ramp. Start/goal have distinct
        # sub-cell offsets and the final ray is outside lattice headings.
        x=g.origin[0]+(np.arange(g.n_cells)+.5)*g.cell_size-g.world_size/2
        ramp=x>1.
        g.slope[ramp,:]=math.tan(math.radians(13.))
        g.gx[ramp,:]=g.slope[ramp,:]
    start=(-.13,.07)
    goal=(2.34,.31)

    def route(self,g,**kw):
        return observed_route(g,self.start,self.goal,0.,allow_frontier=False,max_expansions=1,**kw)

    def test_exact_sloping_goal_connected_without_terminal_turn(self):
        g=copy.deepcopy(self.fixture)
        self.assertGreater(g.max_support_slope(self.goal),12.)
        path,done=self.route(g)
        self.assertTrue(done);self.assertEqual(path,[self.start,self.goal])
        self.assertFalse(motion_route_reason(g,path,0.))
        self.assertFalse(observed_edge_reason(g,*path,self.start))

    def test_sloping_start_does_not_gain_arbitrary_heading_connector(self):
        g=copy.deepcopy(self.fixture);g.slope[:]=math.tan(math.radians(20.));g.gx[:]=g.slope
        path,done=self.route(g)
        self.assertFalse(done);self.assertFalse(path)

    def test_terminal_connector_retains_obstacles_unknown_and_direction_exclusion(self):
        for kind in ('obstacle','unknown','exclusion'):
            with self.subTest(kind=kind):
                g=copy.deepcopy(self.fixture);cell=g.world_to_cell(.7,.15)
                if kind=='obstacle':g.hazard[cell]=True;g.blocked[cell]=True
                elif kind=='unknown':g.known[cell]=False
                else:g.mark_failure('terminal-fixture',(.7,.15),.1,1.,'fixture')
                self.assertFalse(self.route(g)[1])

    def test_footprint_overall_uphill_and_cross_grade_are_independent(self):
        for kind in ('overall','uphill','cross'):
            with self.subTest(kind=kind):
                g=copy.deepcopy(self.fixture);cell=g.world_to_cell(.7,.15)
                if kind=='overall':g.slope[cell]=math.tan(math.radians(36.))
                elif kind=='uphill':g.gx[cell]=math.tan(math.radians(34.))
                else:g.gy[cell]=math.tan(math.radians(28.))
                self.assertFalse(self.route(g)[1])

    def test_freshness_remains_relative_to_original_observer(self):
        g=copy.deepcopy(self.fixture);g.fresh[g.world_to_cell(.7,.15)]=False
        self.assertFalse(self.route(g)[1])
        self.assertTrue(self.route(g,require_fresh_start=False)[1])
        g=copy.deepcopy(self.fixture);g.fresh[g.world_to_cell(*self.goal)]=False
        self.assertTrue(self.route(g)[1])

    def test_requested_uphill_profile_is_not_relaxed(self):
        self.assertFalse(self.route(copy.deepcopy(self.fixture),uphill_limit=10.)[1])


if __name__=='__main__':unittest.main()
