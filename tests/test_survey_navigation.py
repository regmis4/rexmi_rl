"""Survey memory, directional climb evidence and whole-map rendering regressions."""
import math
import unittest
from types import SimpleNamespace
import numpy as np
from test_checkpoint_navigation import m,c,surface,module
v=module('survey_perception','nav/perception_view.py')

class SurveyTests(unittest.TestCase):
    def test_fast_plane_filters_match_original_2d_filters(self):
        from pathlib import Path
        source=Path(m.__file__).read_text()
        source=source.replace('sums = moments(weights,powers)',
            "sums = [ndi.correlate(weights,k,mode='constant') for k in [np.ones_like(xx),xx,yy,xx*xx,xx*yy,yy*yy]]")
        source=source.replace('np.stack(moments(weights*z,powers[:3]),axis=-1)',
            "np.stack([ndi.correlate(weights*z,k,mode='constant') for k in [np.ones_like(xx),xx,yy]],axis=-1)")
        scope={};exec(compile(source,'reference_map','exec'),scope)
        fast=m.ObservedTerrainMap(world_size=12.)
        reference=scope['ObservedTerrainMap'](world_size=12.)
        x,y=np.meshgrid(np.arange(-4.9,5.,.1),np.arange(-4.9,5.,.1),indexing='ij')
        z=.45*x+.2*y+np.where((abs(x-1)<.35)&(abs(y)<.35),.5,0.)
        points=np.c_[x.ravel(),y.ravel(),z.ravel()]
        for g in (fast,reference):g.ingest(points,(0,0,1),0);g.rebuild(0)
        for name in ('known','hazard','blocked'):np.testing.assert_array_equal(getattr(fast,name),getattr(reference,name))
        for name in ('gx','gy','cost'):np.testing.assert_allclose(getattr(fast,name),getattr(reference,name),atol=1e-5)

    def test_radius_retention_forgets_evidence_without_resurrection(self):
        g=surface();near=g.world_to_cell(0,0);far=g.world_to_cell(4,0)
        g.record_traversal([(4,0)],0,failed=True)
        g.retain_radius((0,0),2.);g.rebuild(1.)
        self.assertTrue(g.known[near]);self.assertFalse(g.known[far])
        self.assertFalse(np.isfinite(g._bin_low[far]).any())
        self.assertFalse(g.failures[far].any())
        g.rebuild(2.);self.assertFalse(g.known[far])
        self.assertFalse(g.segment_clear((0,0),(4,0)))

    def test_click_mode_waits_then_plans_and_restores_mission_progress(self):
        g=surface();waypoints=[SimpleNamespace(x=1.,y=0.),SimpleNamespace(x=3.,y=0.)]
        control=c.CheckpointController(g,waypoints);control.wp_idx=1
        self.assertFalse(control.choose_checkpoint((0,0),(2,0),0))
        control.checkpoint_mode(True,0)
        self.assertEqual(control.state,'HOLD')
        self.assertTrue(control.choose_checkpoint((0,0),(2,0),1))
        self.assertEqual(control.state,'OBSERVE');self.assertEqual(control.waypoints[0].x,2)
        for t in (1.2,1.4,1.6): g.rebuild(t)
        command=control.tick((0,0),0,0,3.)
        self.assertEqual(control.state,'FOLLOW');self.assertGreater(command.vx,0.)
        control.request('stop',4.);self.assertEqual(control.state,'HOLD')
        control.checkpoint_mode(False,5.)
        self.assertIs(control.waypoints,waypoints);self.assertEqual(control.wp_idx,1)
        self.assertEqual(control.state,'HOLD')

    def test_click_rejects_unknown_obstacles_and_unreachable_targets(self):
        g=surface();control=c.CheckpointController(g,[SimpleNamespace(x=3.,y=0.)])
        control.checkpoint_mode(True,0)
        for target in [(100,100),(float('nan'),0)]:
            self.assertFalse(control.choose_checkpoint((0,0),target,1))
        cell=g.world_to_cell(2,0);g.blocked[cell]=True
        self.assertFalse(control.choose_checkpoint((0,0),(2,0),1))
        g.blocked[cell]=False
        g.known[g.world_to_cell(1,0)[0],:]=False
        self.assertFalse(control.choose_checkpoint((0,0),(2,0),1))
        self.assertEqual(control.state,'HOLD')

    def test_dashboard_click_only_enqueues_in_enabled_map(self):
        from queue import SimpleQueue
        from threading import Lock
        dashboard=module('click_dashboard','nav/dashboard.py')
        requests=SimpleQueue();shared={'manual_checkpoint_mode':False}
        view=dashboard.Dashboard(shared,Lock(),surface(),[],request_queue=requests)
        view._ax_global=object();view._ax_local=object()
        view._fig=SimpleNamespace(canvas=SimpleNamespace(toolbar=SimpleNamespace(mode='')))
        event=SimpleNamespace(button=1,inaxes=view._ax_global,xdata=1.2,ydata=-.8)
        view._on_map_click(event);self.assertTrue(requests.empty())
        shared['manual_checkpoint_mode']=True;view._on_map_click(event)
        self.assertEqual(requests.get(),('target',1.2,-.8))
        view._fig.canvas.toolbar.mode='pan';view._on_map_click(event)
        self.assertTrue(requests.empty())

    def test_35_degree_limit_applies_to_routes_and_cream_exits(self):
        for degrees in (34.9,35.1):
            g=surface(math.tan(math.radians(degrees)));cell=g.world_to_cell(0,0)
            self.assertFalse(g.hazard[cell])
            self.assertEqual(bool(g.blocked[cell]),degrees>35)
            self.assertEqual(g.segment_clear((0,0),(.2,0),g.footprint_radius),degrees<35)
            colors,legend=v.survey_colors(g.cost,g.known,g.clearance_mask,slope=g.slope)
            if degrees>35:
                np.testing.assert_allclose(colors[cell],[1.,.29,.12])
                self.assertFalse(g.clearance_mask[cell])

    def test_steep_boundary_does_not_inflate_onto_gentler_ground(self):
        g=m.ObservedTerrainMap(world_size=12.)
        x,y=np.meshgrid(np.arange(-4.9,5.,.1),np.arange(-4.9,5.,.1),indexing='ij')
        z=np.where(x<0,.5*x,.9*x)
        g.ingest(np.c_[x.ravel(),y.ravel(),z.ravel()],(0,0,1),0);g.rebuild(0)
        column=g.world_to_cell(0,0)[1]
        for row in range(15,45):
            cell=(row,column)
            self.assertFalse(g.hazard[cell])
            self.assertEqual(bool(g.blocked[cell]),g.slope[cell]>math.tan(math.radians(35)))
        self.assertFalse(g.clearance_mask[15:45,column].any())

    def test_distant_steep_tile_remains_orange(self):
        cost=np.ones((4,4));slope=np.zeros((4,4));slope[0,0]=1.;cost[0,0]=20.
        colors,_=v.survey_colors(cost,np.ones((4,4),bool),slope=slope)
        _,coarse=v.survey_tiles(np.zeros((4,4)),np.ones((4,4),bool),colors,(0,0),.2,(30,30))
        self.assertTrue(np.any(np.all(np.isclose(coarse,[1.,.29,.12]),axis=1)))

    def test_demo_boulder_is_offset_sensed_and_avoidable(self):
        import ast
        from pathlib import Path
        tree=ast.parse((Path(__file__).resolve().parents[1]/'source/rexmi_rl/tasks/locomotion/velocity/config/go2w/crater_terrain.py').read_text())
        helper=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_add_floor_demo_boulder')
        cfgclass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LunarCraterDemoBowlCfg')
        fields={n.target.id:ast.literal_eval(n.value) for n in cfgclass.body if isinstance(n,ast.AnnAssign) and n.target.id.startswith('floor_boulder_')}
        cfg=SimpleNamespace(**fields);scope={'np':np}
        exec(compile(ast.Module(body=[helper],type_ignores=[]),'terrain_helper','exec'),scope)
        x,y=np.meshgrid(np.arange(-4.9,5.,.1),np.arange(-4.9,5.,.1),indexing='ij');h=np.zeros_like(x)
        scope['_add_floor_demo_boulder'](h,x,y,cfg)
        self.assertAlmostEqual(h.max(),.45)
        self.assertEqual(h[np.unravel_index(np.argmin(x*x+y*y),h.shape)],0.)
        g=m.ObservedTerrainMap(world_size=12.,footprint_radius=.5)
        g.ingest(np.c_[x.ravel(),y.ravel(),h.ravel()],(0,0,1),0);g.rebuild(0)
        self.assertTrue(g.hazard[g.world_to_cell(*cfg.floor_boulder_xy)])
        route,complete=c.observed_route(g,(-1.,.8),(3.,.8),0.)
        self.assertTrue(complete)
        self.assertTrue(all(g.segment_clear(a,b) for a,b in zip(route,route[1:])))
        self.assertTrue(any(abs(y-.8)>.6 for x,y in route))

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

    def cream_corridor(self):
        g=surface()
        cell=g.world_to_cell(.3,.1)
        g.blocked[cell]=True;g.cost[cell]=20.
        g.obstacle_distance[cell]=g.footprint_radius+.05
        return g,cell

    def test_slow_motion_uses_extra_margin_but_route_does_not(self):
        g,cell=self.cream_corridor()
        control=c.CheckpointController(g,[SimpleNamespace(x=3.,y=.1)])
        self.assertFalse(g.segment_clear((.1,.1),(.5,.1)))
        self.assertTrue(control.motion_clear((.1,.1),0,0,c.Motion(.2,0)))
        self.assertFalse(control.motion_clear((.1,.1),0,0,c.Motion(.4,0)))
        g.obstacle_distance[cell]=g.footprint_radius-.01
        self.assertFalse(control.motion_clear((.1,.1),0,0,c.Motion(.2,0)))
        self.assertIn('footprint',control.motion_rejection)

    def test_drift_into_extra_margin_does_not_cancel_safe_slow_motion(self):
        g,cell=self.cream_corridor()
        control=c.CheckpointController(g,[SimpleNamespace(x=3.,y=.1)])
        control.velocity_world=np.array([.3,0.])
        self.assertTrue(control.motion_clear((.1,.1),0,.3,c.Motion(.2,0)))
        g.obstacle_distance[cell]=g.footprint_radius-.01
        self.assertFalse(control.motion_clear((.1,.1),0,.3,c.Motion(.2,0)))
        self.assertIn('drift stopping corridor',control.motion_rejection)

    def test_cream_corner_checks_preserve_body_clearance(self):
        g=surface();cell=g.world_to_cell(.3,.1)
        g.blocked[cell]=True;g.obstacle_distance[cell]=g.footprint_radius+.05
        a,b=(.1,.1),(.3,.3)
        self.assertFalse(g.segment_clear(a,b))
        self.assertTrue(g.segment_clear(a,b,g.footprint_radius))
        g.obstacle_distance[cell]=g.footprint_radius-.01
        self.assertFalse(g.segment_clear(a,b,g.footprint_radius))
        g.obstacle_distance[cell]=g.footprint_radius+.05
        g.fresh[cell]=False
        self.assertFalse(g.segment_clear(a,b,g.footprint_radius))

    def test_cream_permission_never_overrides_unknown_or_red(self):
        for attr,value in [('known',False),('fresh',False),('hazard',True),('slope',2.)]:
            g,cell=self.cream_corridor();getattr(g,attr)[cell]=value
            self.assertFalse(g.segment_clear((.1,.1),(.5,.1),g.footprint_radius),attr)

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
