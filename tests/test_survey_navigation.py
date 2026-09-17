"""Survey memory, directional climb evidence and whole-map rendering regressions."""
import math
import unittest
from types import SimpleNamespace
import numpy as np
from test_checkpoint_navigation import m,c,surface,module
v=module('survey_perception','nav/perception_view.py')

class SurveyTests(unittest.TestCase):
    def test_roughness_removes_plane_slope_and_retains_texture(self):
        def make(textured):
            g=m.ObservedTerrainMap(world_size=12.)
            x,y=np.meshgrid(np.arange(-3.975,4.,.05),np.arange(-3.975,4.,.05),indexing='ij')
            z=.5*x+.2*y
            if textured: z+=.04*np.where((np.indices(x.shape).sum(axis=0)%2)==0,1.,-1.)
            g.ingest(np.c_[x.ravel(),y.ravel(),z.ravel()],(0,0,1),0)
            g.rebuild(0);return g
        flat,rough=make(False),make(True)
        cell=flat.world_to_cell(0,0)
        self.assertLess(flat.roughness[cell],.01)
        self.assertGreater(rough.roughness[cell],.03)
        self.assertEqual(rough.roughness_support[cell],16)
        self.assertFalse(rough.hazard[cell])

    def test_sparse_roughness_is_unknown(self):
        g=m.ObservedTerrainMap(world_size=12.)
        g.ingest(np.array([[.01,.01,0.]]),(0,0,1),0);g.rebuild(0)
        self.assertTrue(np.isnan(g.roughness[g.world_to_cell(.01,.01)]))

    def test_failure_is_directional_persistent_and_exported(self):
        g=surface();cell=g.world_to_cell(0,0)
        g.record_traversal([(0,0)]*50,0,failed=True)
        self.assertEqual(g.failures[*cell,0],1)
        self.assertTrue(g.direction_allowed(cell,0))
        before=g.edge_cost(cell,1,0,0)
        g.record_traversal([(0,0)],0,failed=True)
        g.rebuild(10.)
        self.assertFalse(g.direction_allowed(cell,0))
        self.assertTrue(g.direction_allowed(cell,4))
        self.assertTrue(math.isinf(g.edge_cost(cell,1,0,0)))
        self.assertGreater(before,g.edge_cost(cell,-1,0,4))
        self.assertEqual(g.survey_snapshot()['failures'][*cell,0],2)
        self.assertTrue(g.known[cell])

    def test_second_failed_approach_changes_route(self):
        g=surface()
        for _ in range(2): g.record_traversal([(.5,0)],0,failed=True)
        route,complete=c.observed_route(g,(-1,0),(2,0),0)
        self.assertTrue(complete)
        self.assertTrue(math.isfinite(g.route_cost(route,0)))
        self.assertTrue(any(abs(y)>.3 for x,y in route))

    def test_gentler_route_beats_short_steep_corridor(self):
        g=surface()
        for r in range(g.n_cells):
            for col in range(g.n_cells):
                x,y=g.cell_to_world(r,col)
                if -.5<x<1.5 and abs(y)<.5:
                    g.gx[r,col]=1.;g.cost[r,col]=5.
        route,complete=c.observed_route(g,(-1,0),(2,0),0)
        self.assertTrue(complete)
        self.assertTrue(all(g.cost[g.world_to_cell(*p)]<2. for p in route))

    def test_survey_frontier_can_move_away_from_blocked_goal(self):
        g=surface()
        # Entire known corridor extends away from a goal behind unknown space.
        g.known[:]=False;g.fresh[:]=False
        for r in range(g.n_cells):
            for col in range(g.n_cells):
                x,y=g.cell_to_world(r,col)
                if -.1<=x<3. and abs(y)<.5: g.known[r,col]=g.fresh[r,col]=True
        route,complete=c.observed_route(g,(.1,.1),(-3.,0.),0)
        self.assertFalse(complete)
        self.assertGreater(len(route),1)
        self.assertTrue(all(g.known[g.world_to_cell(*p)] for p in route))
        self.assertGreater(route[-1][0],.8)

    def test_failed_approach_does_not_block_braking_out_of_drift(self):
        g=surface();cell=g.world_to_cell(.2,0)
        g.failures[*cell,0]=2
        self.assertFalse(g.segment_clear((0,0),(.4,0)))
        self.assertTrue(g.segment_clear((0,0),(.4,0),check_direction=False))
        control=c.CheckpointController(g,[SimpleNamespace(x=-3.,y=0.)])
        control.velocity_world=np.array([.2,0.])
        self.assertTrue(control.motion_clear((0,0),0,.2,c.Motion(-.2,0.)))
        self.assertIn('approach direction',g.segment_reason((0,0),(.4,0)))

    def test_yellow_tiles_do_not_force_stop(self):
        g=surface();g.cost[g.known]=6.
        control=c.CheckpointController(g,[SimpleNamespace(x=3.,y=0.)]);g.revision=5
        command=control.tick((0.,0.),0,0,2.)
        self.assertEqual(control.state,'FOLLOW');self.assertGreater(command.vx,0.)

    def test_cream_exit_can_continue_without_settle(self):
        g=surface(rock=True)
        for r,col in zip(*np.nonzero(g.clearance_mask)):
            xy=g.cell_to_world(r,col)
            route,clearance=g.clearance_exit(xy)
            if not route: continue
            yaw=math.atan2(route[-1][1]-xy[1],route[-1][0]-xy[0])
            control=c.CheckpointController(g,[SimpleNamespace(x=route[-1][0],y=route[-1][1])])
            control.state='FOLLOW';control.path=route;control.checkpoint=route[-1]
            motion=control.tick(xy,yaw,0,1.)
            if control.state=='FOLLOW' and motion.vx>0: break
        else: self.fail('No verified cream exit permitted forward motion')
        self.assertLessEqual(motion.vx,.2)
        self.assertGreaterEqual(control.escape_clearance,g.footprint_radius)

    def test_commanded_stop_and_turn_never_count_as_failed_climb(self):
        g=surface(.6);control=c.CheckpointController(g,[SimpleNamespace(x=3.,y=0.)])
        control.request('stop',0)
        for t in np.arange(0,4,.02): control.tick((0,0),0,0,float(t))
        self.assertEqual(control.climb_failures,0)
        self.assertFalse(g.failures.any())

    def test_failed_climb_records_once_then_replans(self):
        g=surface(.6);control=c.CheckpointController(g,[SimpleNamespace(x=3.,y=0.)])
        control.state='FOLLOW';control.path=[(0,0),(3,0)];control.checkpoint=(3,0)
        control.previous_command=c.Motion(.4,0)
        for t in np.arange(0,2.1,.02): control.tick((0,0),0,0,float(t),velocity=(0,0,0))
        self.assertEqual(control.climb_failures,1)
        self.assertIn(control.state,('SETTLE','OBSERVE'))
        self.assertEqual(int(g.failures.max()),1)
        self.assertFalse(control.mission_route)

    def test_side_slip_records_failure_despite_forward_motion(self):
        g=surface(.6);control=c.CheckpointController(g,[SimpleNamespace(x=3.,y=0.)])
        control.state='FOLLOW';control.path=[(0,0),(3,0)];control.checkpoint=(3,0)
        control.previous_command=c.Motion(.4,0.)
        control.tick((1.,.8),0,.3,2.,velocity=(.2,.2,0.))
        self.assertEqual(control.state,'SETTLE')
        self.assertEqual(control.climb_failures,1)
        self.assertIn('lateral route error',control.reason)

    def test_cached_route_retained_until_meaningful_improvement(self):
        g=surface();control=c.CheckpointController(g,[SimpleNamespace(x=3.,y=0.)])
        self.assertTrue(control._plan((0.,0.),0,1.))
        original=list(control.mission_route)
        self.assertTrue(control._plan((0.,0.),0,2.))
        self.assertIn('not 15%',control.route_decision)
        self.assertEqual(control.mission_route,original)
        for r,col in zip(*np.nonzero(g.known)):
            x,y=g.cell_to_world(r,col)
            if .5<x<2.5 and abs(y)<.3: g.cost[r,col]=10.
        self.assertTrue(control._plan((0.,0.),0,3.))
        self.assertNotEqual(control.mission_route,original)
        self.assertIn('best observed',control.route_decision)

    def test_successful_corridor_receives_lower_cost(self):
        g=surface();cell=g.world_to_cell(0,0)
        previous=g.edge_cost(cell,1,0,0)
        g.record_traversal([(0,0)],0)
        self.assertLess(g.edge_cost(cell,1,0,0),previous)
        self.assertGreaterEqual(g.edge_cost(cell,1,0,0),1.)

    def test_historical_surface_can_plan_but_not_authorize_motion(self):
        g=surface();route=[(0,0),(3,0)]
        g.rebuild(20)
        self.assertTrue(math.isfinite(g.route_cost(route)))
        self.assertFalse(g.segment_clear(*route))

class FullSurveyDisplayTests(unittest.TestCase):
    def test_distant_red_hazard_preserved(self):
        height=np.zeros((40,40));known=np.ones_like(height,bool);cost=np.ones_like(height)
        cost[2,2]=20
        palette,_=v.survey_colors(cost,known)
        vertices,colors=v.survey_tiles(height,known,palette,(0,0),.2,(30,30))
        self.assertEqual(len(colors),400)
        self.assertTrue(np.any(np.all(colors==[1,0,0],axis=1)))
        self.assertLess(vertices[:,0].min(),-3.)

    def test_distant_outcomes_keep_most_severe_evidence(self):
        h=np.zeros((2,2));known=np.ones((2,2),bool)
        failures=np.zeros((2,2,8),np.uint8);successes=failures.copy()
        failures[0,0,0]=1;failures[1,0,0]=2;successes[0,1,0]=1
        palette,_=v.survey_colors(np.ones_like(h),known,layer='traversal',failures=failures,successes=successes)
        _,colors=v.survey_tiles(h,known,palette,(0,0),.2,(30,30))
        np.testing.assert_allclose(colors[0],[.65,.25,.85])

    def test_lod_does_not_fill_unknown_holes(self):
        h=np.zeros((4,4));known=np.ones((4,4),bool);known[0,0]=False
        colors,_=v.survey_colors(np.ones_like(h),known)
        vertices,_=v.survey_tiles(h,known,colors,(0,0),.2,(30,30))
        quads=vertices.reshape(-1,4,3)
        area=np.sum((quads[:,1,0]-quads[:,0,0])*(quads[:,3,1]-quads[:,0,1]))/(.86**2)
        self.assertAlmostEqual(area,15*.04,places=5)

    def test_layer_diagnostics_never_mutate_planner_costs(self):
        g=surface(rock=True);before=g.cost.copy()
        for layer in v.LAYER_NAMES:
            colors,legend=v.survey_colors(g.cost,g.known,g.clearance_mask,layer,g.slope,
                                         g.roughness,g.failures,g.successes)
            self.assertEqual(colors.shape,(*g.cost.shape,3));self.assertTrue(legend)
        np.testing.assert_equal(g.cost,before)

if __name__=='__main__': unittest.main()
