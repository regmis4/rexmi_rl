"""Lower-grade ascent, diagonal clearance and committed downhill regrouping."""
import math
import unittest
from types import SimpleNamespace
import numpy as np
from test_checkpoint_navigation import c,m,surface


def hillside():
    g=m.ObservedTerrainMap(world_size=16.,footprint_radius=.3)
    x,y=np.meshgrid(np.arange(-6.9,7.,.1),np.arange(-6.9,7.,.1),indexing='ij')
    g.ingest(np.c_[x.ravel(),y.ravel(),(.5*np.maximum(x,0)).ravel()],(0,0,3),0.)
    g.rebuild(0.)
    return g


class AscentTests(unittest.TestCase):
    def test_diagonal_climb_reduces_along_grade_without_excess_side_slope(self):
        g=surface(.5);ctrl=c.CheckpointController(g,[])
        path,done=ctrl.choose_route((-2.1,.1),(2.1,.1),0.)
        self.assertTrue(done)
        profile=g.route_profile(path)
        self.assertLessEqual(profile['max_uphill_deg'],20.+1e-6)
        self.assertLessEqual(profile['max_side_slope_deg'],25.+1e-6)
        self.assertGreater(max(abs(y-.1) for x,y in path),.5)
        self.assertTrue(all(g.segment_clear(a,b) for a,b in zip(path,path[1:])))

    def test_follower_does_not_cut_steep_shortcut_between_diagonals(self):
        g=surface(.5);ctrl=c.CheckpointController(g,[SimpleNamespace(x=2.1,y=.1)])
        self.assertTrue(ctrl._plan((-2.1,.1),0.,1.))
        motion=ctrl.tick((-2.1,.1),0.,0.,1.1,velocity=(0,0,0))
        self.assertEqual(motion.vx,0.)
        self.assertGreater(abs(motion.omega),.1)
        self.assertIn('corner',ctrl.reason)

    def test_dangerous_side_slope_is_not_a_diagonal_escape(self):
        g=surface(math.tan(math.radians(35.)));ctrl=c.CheckpointController(g,[])
        self.assertIn('side slope',g.grade_reason((0,0),(0,1)))
        self.assertFalse(g.segment_clear((0,0),(1,1)))
        _,done=c.observed_route(g,(-2.1,.1),(2.1,.1),0.,allow_frontier=False)
        self.assertFalse(done)

    def test_long_diagonal_cannot_jump_an_unknown_cell(self):
        g=m.ObservedTerrainMap(world_size=4.,cell_size=.2,footprint_radius=.01,clearance=0.)
        start=g.world_to_cell(0,0);end=(start[0]+2,start[1]+1)
        for cell in (start,end):g.known[cell]=g.fresh[cell]=True;g.cost[cell]=1.
        path,done=c.observed_route(g,g.cell_to_world(*start),g.cell_to_world(*end),0,allow_frontier=False)
        self.assertFalse(done);self.assertFalse(path)

    def test_grade_limits_match_scalar_and_vectorized_edge_costs(self):
        g=surface(.6);ctrl=c.CheckpointController(g,[]);cell=g.world_to_cell(0,0)
        costs,_=g.directional_costs()
        self.assertTrue(math.isinf(costs[*cell,0]))
        self.assertTrue(math.isinf(g.edge_cost(cell,1,0,0)))
        self.assertTrue(math.isfinite(costs[*cell,1]))
        self.assertTrue(math.isfinite(g.edge_cost(cell,1,1,0)))

    def test_low_grade_preference_cannot_retain_steep_cached_route(self):
        g=surface(.4);ctrl=c.CheckpointController(g,[SimpleNamespace(x=2.1,y=.1)])
        direct=[(-2.1,.1),(2.1,.1)]
        ctrl.mission_route=direct;ctrl.route_mission=0;ctrl.mission_route_complete=True
        self.assertTrue(ctrl._plan((-2.1,.1),0.,1.))
        self.assertLessEqual(ctrl.route_profile['max_uphill_deg'],20.+1e-6)


class RetreatTests(unittest.TestCase):
    def setup_controller(self):
        g=hillside();ctrl=c.CheckpointController(g,[SimpleNamespace(x=4.1,y=.1)])
        xy=(2.1,.1)
        ctrl.begin_escape(xy,0.,0.,dict(kind='poor drive progress'))
        return g,ctrl,xy

    def test_failed_climb_excludes_patch_and_plans_lower_point(self):
        g,ctrl,xy=self.setup_controller()
        self.assertTrue(g.no_go[g.world_to_cell(*xy)])
        self.assertTrue(ctrl.retreat_required)
        self.assertTrue(ctrl.plan_downhill_retreat(xy,0.,.4))
        self.assertEqual(ctrl.route_purpose,'retreat')
        self.assertGreaterEqual(math.dist(xy,ctrl.retreat_goal),2.5)
        self.assertLess(g.surface[g.world_to_cell(*ctrl.retreat_goal)],ctrl.retreat_height-.4)
        self.assertTrue(all(g.segment_clear(a,b,no_go_exit=True) for a,b in zip(ctrl.path,ctrl.path[1:])))
        command=ctrl.tick(xy,0.,0.,.5,velocity=(0,0,0))
        self.assertLess(command.vx,-.25)

    def test_retreat_latch_survives_replans_until_regroup_point(self):
        g,ctrl,xy=self.setup_controller();ctrl.plan_downhill_retreat(xy,0.,.4)
        target=ctrl.retreat_goal
        for t in (1.,3.,5.):
            self.assertTrue(ctrl._plan((2.0,.1),0.,t))
            self.assertEqual(ctrl.retreat_goal,target)
            self.assertEqual(ctrl.route_purpose,'retreat')
            self.assertEqual(ctrl.wp_idx,0)
        self.assertFalse(ctrl.is_mission_checkpoint)
        self.assertTrue(ctrl._plan(target,0.,7.))
        self.assertFalse(ctrl.retreat_required)
        self.assertEqual(ctrl.route_purpose,'mission')
        self.assertTrue(all(not g.no_go[g.world_to_cell(*p)] for p in ctrl.mission_route))
        self.assertEqual(ctrl.waypoints[0].x,4.1)

    def test_stop_during_retreat_does_not_resume_it(self):
        g,ctrl,xy=self.setup_controller();ctrl.plan_downhill_retreat(xy,0.,.4)
        ctrl.request('stop',.5)
        for t in (1.,5.,10.):ctrl.tick(xy,0,0,t)
        self.assertEqual(ctrl.state,'HOLD')
        self.assertTrue(ctrl.retreat_required)

    def test_downhill_overspeed_brakes_without_marking_it_as_failed_ascent(self):
        g,ctrl,xy=self.setup_controller();ctrl.plan_downhill_retreat(xy,0.,.4)
        command=ctrl.tick(xy,0.,.6,.5,velocity=(-.6,0,0))
        self.assertGreater(command.vx,0.)
        self.assertIn('braking downhill',ctrl.reason)
        ctrl.tick(xy,0.,.6,.7,velocity=(-.6,0,0))
        self.assertEqual(ctrl.climb_failures,1)

    def test_downhill_braking_does_not_undo_heading_progress(self):
        g,ctrl,xy=self.setup_controller();ctrl.plan_downhill_retreat(xy,0.,.4)
        ctrl.hold_anchor=np.array(xy);ctrl.hold_heading=-1.5
        command=ctrl.tick(xy,.2,.6,.5,velocity=(-.6,0,0))
        self.assertIn('braking downhill',ctrl.reason)
        self.assertGreater(command.vx,0.)
        self.assertEqual(command.omega,0.)

    def test_clearance_exit_during_retreat_uses_lower_target(self):
        g,ctrl,xy=self.setup_controller();ctrl.plan_downhill_retreat(xy,0.,.4)
        called=[]
        def exit_path(position,goal=None):
            called.append(goal)
            return [],None
        g.clearance_exit=exit_path
        ctrl.tick(xy,0.,0.,.5,velocity=(0,0,0))
        self.assertEqual(called,[ctrl.retreat_goal])

    def test_backoff_transitions_to_forward_orientation_on_lower_grade(self):
        g,ctrl,xy=self.setup_controller();ctrl.plan_downhill_retreat(xy,0.,.4)
        lower=(.2,.1)
        ctrl.path=[lower,ctrl.retreat_goal];ctrl.checkpoint=ctrl.retreat_goal
        ctrl.path_index=0;ctrl.last_revision=g.revision
        command=ctrl.tick(lower,0.,.3,.5,velocity=(-.3,0,0))
        self.assertTrue(ctrl.retreat_forward_only)
        self.assertGreater(command.vx,0.)
        command=ctrl.tick(lower,0.,0.,1.,velocity=(0,0,0))
        self.assertEqual(command.vx,0.)
        self.assertGreater(abs(command.omega),.1)

    def test_inertial_drift_into_learned_zone_does_not_forbid_checked_exit(self):
        g,ctrl,xy=self.setup_controller();ctrl.plan_downhill_retreat(xy,0.,.4)
        ctrl.velocity_world=np.array([.15,0.])
        self.assertTrue(ctrl.motion_clear((1.8,.1),0.,.15,c.Motion(-.4,0)))
        self.assertFalse(ctrl.motion_clear((1.8,.1),0.,.15,c.Motion(.4,0)))

    def test_committed_retreat_has_its_own_bounded_budget(self):
        g,ctrl,xy=self.setup_controller();ctrl.plan_downhill_retreat(xy,0.,.4)
        ctrl.begin_escape(xy,0,20.,dict(kind='repeated command rejection'),mark=False)
        for t in (20.,20.3):ctrl.tick(xy,0,0,t)
        self.assertEqual(ctrl.state,'FOLLOW')
        self.assertTrue(ctrl.retreat_required)
        ctrl.tick(xy,0,0,91.)
        self.assertEqual(ctrl.state,'HOLD')
        self.assertIn('timeout',ctrl.reason)

    def test_unknown_lower_ground_is_not_invented(self):
        g,ctrl,xy=self.setup_controller()
        for r in range(g.n_cells):
            if g.cell_to_world(r,0)[0]<1.:g.known[r,:]=g.fresh[r,:]=False
        self.assertFalse(ctrl.plan_downhill_retreat(xy,0.,.4))
        self.assertTrue(ctrl.retreat_required)
        self.assertIsNone(ctrl.retreat_goal)

    def test_rejection_loop_on_failed_ascent_excludes_site_without_claiming_slip(self):
        g=hillside();ctrl=c.CheckpointController(g,[SimpleNamespace(x=4.,y=0.)])
        ctrl.begin_escape((2.,0.),0,0,dict(kind='repeated command rejection'),mark=False)
        self.assertEqual(ctrl.climb_failures,0)
        self.assertFalse(g.failures.any())
        self.assertTrue(g.no_go.any());self.assertTrue(ctrl.retreat_required)
        self.assertEqual(g.failed_climbs[0]['reason'],'repeated command rejection')

    def test_region_and_grade_limits_are_exported_and_survive_retention(self):
        g,ctrl,xy=self.setup_controller();g.retain_radius((-5.,0),1.);g.rebuild(1.)
        snap=g.survey_snapshot()
        self.assertTrue(snap['no_go'][g.world_to_cell(*xy)])
        self.assertEqual(snap['uphill_limit_deg'],25.)
        np.testing.assert_allclose(snap['failed_climb_positions'],[xy])


if __name__=='__main__':unittest.main()
