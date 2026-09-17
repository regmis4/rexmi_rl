"""Deterministic map, path and command checks, without launching Isaac Sim."""
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]/'source/rexmi_rl'
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    obj=importlib.util.module_from_spec(spec)
    sys.modules[name]=obj
    spec.loader.exec_module(obj)
    return obj
m=module('observed_map','nav/observed_map.py')
c=module('checkpoint_control','nav/checkpoint_control.py')
d=module('drive_adapter','drive_adapter.py')

def surface(slope=0.,rock=False):
    grid=m.ObservedTerrainMap(world_size=12.,footprint_radius=.3)
    x,y=np.meshgrid(np.arange(-4.9,5.,.1),np.arange(-4.9,5.,.1),indexing='ij')
    z=slope*x
    if rock: z=z+np.where((abs(x)<.35)&(abs(y)<.35),.5,0.)
    grid.ingest(np.c_[x.ravel(),y.ravel(),z.ravel()],(0,0,1),0)
    grid.rebuild(0)
    return grid

class MapTests(unittest.TestCase):
    def test_clearance_display_mask_excludes_real_hazard(self):
        grid=surface(0.,True)
        self.assertTrue(grid.clearance_mask.any())
        self.assertFalse(np.any(grid.clearance_mask & grid.hazard))
        self.assertTrue(np.all(grid.blocked[grid.clearance_mask]))
        self.assertTrue(np.all(grid.cost[grid.clearance_mask]==20.))
    def test_old_route_memory_does_not_authorize_stale_motion(self):
        grid=surface()
        grid.rebuild(10.)
        route,complete=c.observed_route(grid,(0.,0.),(3.,0.))
        self.assertFalse(complete)
        self.assertFalse(route)
        self.assertTrue(grid.known[grid.world_to_cell(3.,0.)])
        self.assertFalse(grid.segment_clear((0.,0.),(1.,0.)))
        # New contradictory evidence must invalidate the old free corridor.
        grid.ingest([[1.,0.,1.]],(1.,0.,2.),10.)
        grid.rebuild(10.)
        self.assertTrue(grid.blocked[grid.world_to_cell(1.,0.)])

    def test_smooth_slope_not_boulder(self):
        for slope in (0.,.4,.75):
            grid=surface(slope)
            cell=grid.world_to_cell(0,0)
            self.assertFalse(grid.hazard[cell])
            self.assertFalse(grid.blocked[cell])
            self.assertAlmostEqual(grid.gx[cell],slope,delta=.08)

    def test_flat_top_boulder_and_boulder_on_slope(self):
        for slope in (0.,.4):
            grid=surface(slope,True)
            cell=grid.world_to_cell(0,0)
            self.assertTrue(grid.hazard[cell])
            self.assertTrue(grid.blocked[cell])
            self.assertEqual(grid.get_cost_grid()[cell],20.)
            self.assertFalse(grid.traversable(cell))

    def test_range_is_sensor_relative(self):
        grid=m.ObservedTerrainMap()
        grid.ingest([[21.,0,0],[0,0,0]],(20,0,1),1,max_range=5.)
        grid.rebuild(1.)
        self.assertTrue(np.isfinite(grid.height[grid.world_to_cell(21,0)]))
        self.assertFalse(np.isfinite(grid.height[grid.world_to_cell(0,0)]))

    def test_rounded_boulder_on_slope(self):
        grid=m.ObservedTerrainMap(world_size=12.)
        x,y=np.meshgrid(np.arange(-4.9,5.,.1),np.arange(-4.9,5.,.1),indexing='ij')
        z=.4*x+.4*np.exp(-(x*x+y*y)/(2*.5**2))
        grid.ingest(np.c_[x.ravel(),y.ravel(),z.ravel()],(0,0,1),0)
        grid.rebuild(0)
        self.assertTrue(grid.hazard[grid.world_to_cell(0,0)])

    def test_low_scarp_on_slope_is_not_a_tall_obstacle(self):
        grid=m.ObservedTerrainMap(world_size=12.)
        x,y=np.meshgrid(np.arange(-4.9,5.,.1),np.arange(-4.9,5.,.1),indexing='ij')
        z=.6*x+.18*np.exp(-x*x/(2*.24**2))
        grid.ingest(np.c_[x.ravel(),y.ravel(),z.ravel()],(0,0,1),0)
        grid.rebuild(0)
        self.assertFalse(grid.hazard[grid.world_to_cell(0,0)])

    def test_fuse_current_sensor_batch_without_old_outlier(self):
        grid=m.ObservedTerrainMap(world_size=12.)
        grid.ingest([[.1,.1,0]],(0,0,1),1.)
        grid.ingest([[.1,.1,.5]],(0,0,1),1.)
        grid.rebuild(1.)
        cell=grid.world_to_cell(.1,.1)
        self.assertEqual(grid.low[cell],0.)
        self.assertEqual(grid.height[cell],.5)
        grid.ingest([[.1,.1,0]],(0,0,1),2.)
        grid.rebuild(2.)
        self.assertEqual(grid.height[cell],.5)
        for t in (3.,4.):
            grid.ingest([[.1,.1,0]],(0,0,1),t)
            grid.rebuild(t)
        self.assertEqual(grid.height[cell],0.)

    def test_sparse_unknown_and_stale(self):
        grid=m.ObservedTerrainMap(world_size=12.)
        grid.ingest([[0,0,0]],(0,0,1),0)
        grid.rebuild(0)
        self.assertFalse(grid.known[grid.world_to_cell(1,0)])
        grid.rebuild(9)
        self.assertTrue(grid.known.any())
        self.assertFalse(grid.fresh.any())
        self.assertFalse(grid.segment_clear((0,0),(.1,0)))

    def test_repeated_contradictory_observations_can_clear_old_height(self):
        grid=surface(0.,True)
        plain=surface()
        x,y=np.meshgrid(np.arange(-4.9,5.,.1),np.arange(-4.9,5.,.1),indexing='ij')
        grid.ingest(np.c_[x.ravel(),y.ravel(),np.zeros(x.size)],(0,0,1),1)
        grid.rebuild(1)
        self.assertTrue(grid.hazard[grid.world_to_cell(0,0)])
        for t in (2.,3.):
            grid.ingest(np.c_[x.ravel(),y.ravel(),np.zeros(x.size)],(0,0,1),t)
            grid.rebuild(t)
        self.assertFalse(grid.hazard[grid.world_to_cell(0,0)])

    def test_age_changes_freshness_not_terrain_or_map_extent(self):
        grid=surface(.4,True)
        before=grid.cost.copy();known=grid.known.copy();height=grid.height.copy()
        grid.rebuild(100.)
        np.testing.assert_array_equal(grid.cost,before)
        np.testing.assert_array_equal(grid.known,known)
        np.testing.assert_array_equal(grid.height,height)
        self.assertFalse(grid.fresh.any())
        self.assertGreater(grid.quality((0,0))['mapped_coverage'],.99)

    def test_viewpoint_change_retains_other_parts_of_cell(self):
        grid=m.ObservedTerrainMap(world_size=12.)
        grid.ingest([[.01,.01,0],[.16,.16,.5]],(0,0,1),0.)
        grid.rebuild(0.)
        cell=grid.world_to_cell(.01,.01)
        for t in range(1,10):
            grid.ingest([[.01,.01,0]],(t*.1,0,1),float(t))
            grid.rebuild(float(t))
        self.assertEqual(grid.low[cell],0.)
        self.assertEqual(grid.height[cell],.5)

    def test_moving_scan_retains_traversed_terrain(self):
        grid=surface(.4,True)
        original=grid.height.copy()
        old=grid.world_to_cell(-3.,0.)
        cell=grid.world_to_cell(0.,0.)
        area=grid.quality((0,0))['mapped_area']
        for t in range(1,12):
            x,y=np.meshgrid(np.arange(1.+t*.1,4.9,.1),np.arange(-2.,2.,.1),indexing='ij')
            grid.ingest(np.c_[x.ravel(),y.ravel(),(.4*x).ravel()],(t*.2,0,1),float(t))
            grid.rebuild(float(t))
            self.assertGreaterEqual(grid.quality((0,0))['mapped_area'],area)
            self.assertEqual(grid.height[old],original[old])
            self.assertTrue(grid.hazard[cell])

    def test_rebuild_without_measurements_does_not_clear_obstacle(self):
        grid=surface(0.,True)
        for t in range(20): grid.rebuild(float(t))
        self.assertTrue(grid.hazard[grid.world_to_cell(0,0)])

    def test_clearance_exit_preserves_physical_footprint(self):
        grid=surface(0.,True)
        candidates=np.argwhere(grid.known & grid.blocked & ~grid.hazard &
                               (grid.obstacle_distance>=grid.footprint_radius))
        exits=[]
        for cell in candidates:
            xy=grid.cell_to_world(*cell)
            path,clearance=grid.clearance_exit(xy)
            if path:
                self.assertGreaterEqual(clearance,grid.footprint_radius)
                self.assertTrue(grid.traversable(grid.world_to_cell(*path[-1])))
                self.assertTrue(grid.segment_clear(*path,escape_clearance=clearance))
                exits.append(path)
        self.assertTrue(exits)
        self.assertEqual(grid.clearance_exit((0.,0.)),([],None))

class RouteTests(unittest.TestCase):
    def test_stall_retreat_retains_travelled_corridor_until_needed(self):
        grid=surface()
        grid.revision=5
        control=c.CheckpointController(grid,[SimpleNamespace(x=4.,y=0.)])
        control.tick((0,0),0,0,2.)
        control.tick((.5,0),0,0,4.)
        control.tick((.5,0),0,0,13.)
        control.tick((.5,0),0,0,40.)
        self.assertEqual(control.state,'REVERSE')
        control.tick((0,0),0,0,42.)
        control.tick((0,0),0,0,42.1)
        control.tick((0,0),0,0,42.7)
        grid.revision+=5
        control.tick((0,0),0,0,43.8)
        self.assertEqual(control.state,'FOLLOW')
        self.assertEqual(control.retry_count,1)
    def test_clear_turn_is_not_cancelled_by_long_straight_probe(self):
        grid=m.ObservedTerrainMap(world_size=32.)
        grid.known[:]=True;grid.fresh[:]=True;grid.cost[:]=1.;grid.revision=5
        cell=grid.world_to_cell(11.5,.26)
        grid.blocked[cell]=True;grid.cost[cell]=20.
        control=c.CheckpointController(grid,[SimpleNamespace(x=11.,y=0.)])
        motion=control.tick((11.907487869,.260900706),-3.135724231,.01,2.)
        self.assertEqual(control.state,'FOLLOW')
        self.assertAlmostEqual(motion.vx,.2)
        self.assertAlmostEqual(motion.omega,.35)
        self.assertTrue(control.motion_clear((11.907487869,.260900706),-3.135724231,.01,motion))
        # The same turn is not allowed when measured speed would carry the
        # robot into the blocked straight corridor before it could stop.
        self.assertFalse(control.motion_clear((11.907487869,.260900706),-3.135724231,.5,motion))

    def test_polyline_progress_and_lookahead(self):
        i,foot,target,remaining,cross=c.path_progress([(0,0),(4,0),(4,2)],(2,.1))
        self.assertEqual(i,0)
        np.testing.assert_allclose(foot,[2,0])
        np.testing.assert_allclose(target,[3.2,0])
        self.assertAlmostEqual(remaining,4.)
        self.assertAlmostEqual(cross,.1)

    def test_detour_matches_observed_clear_segments(self):
        grid=surface(0.,True)
        route,done=c.observed_route(grid,(-3.,0.),(3.,0.))
        self.assertTrue(done)
        self.assertGreater(max(abs(p[1]) for p in route),.5)
        self.assertTrue(all(grid.segment_clear(a,b) for a,b in zip(route,route[1:])))

    def test_no_diagonal_corner_cut(self):
        grid=m.ObservedTerrainMap(world_size=2.)
        grid.known[:]=True
        grid.fresh[:]=True
        grid.cost[:]=1.
        grid.blocked[:]=True
        grid.blocked[4,4]=grid.blocked[5,5]=False
        route,done=c.observed_route(grid,grid.cell_to_world(4,4),grid.cell_to_world(5,5))
        self.assertFalse(done)
        self.assertEqual(route,[])
        self.assertFalse(grid.segment_clear(grid.cell_to_world(4,4),grid.cell_to_world(5,5)))

    def test_no_route_through_unknown(self):
        grid=surface()
        grid.known[:]=False
        route,done=c.observed_route(grid,(0,0),(2,0))
        self.assertEqual(route,[])

    def test_heading_sign_and_stop_resume(self):
        grid=surface()
        wp=SimpleNamespace(x=3.,y=0.)
        control=c.CheckpointController(grid,[wp])
        grid.revision=5
        motion=control.tick((0,0),-.4,0.,2.)
        self.assertGreater(motion.omega,0.)
        self.assertGreater(motion.vx,0.)
        control.request('stop',2.1)
        self.assertEqual(control.tick((0,0),0,0,2.2),c.Motion())
        control.request('resume',2.3)
        self.assertEqual(control.state,'OBSERVE')
        self.assertEqual(control.path,[])

    def test_new_obstacle_stops_before_replan(self):
        grid=surface()
        control=c.CheckpointController(grid,[SimpleNamespace(x=3.,y=0.)])
        grid.revision=5
        control.tick((0,0),0.,0.,2.)
        grid.blocked[grid.world_to_cell(.5,0)]=True
        grid.revision+=1
        self.assertEqual(control.tick((0,0),0.,.4,2.2),c.Motion())
        self.assertEqual(control.state,'SETTLE')

    def test_tip_never_resets_or_advances_mission(self):
        grid=surface()
        control=c.CheckpointController(grid,[SimpleNamespace(x=3.,y=0.)])
        control.tick((0,0),0,0,2,upright=-1.)
        self.assertEqual(control.state,'HOLD')
        self.assertEqual(control.wp_idx,0)

    def test_cruise_requires_braking_evidence(self):
        grid=surface()
        control=c.CheckpointController(grid,[SimpleNamespace(x=4.,y=0.)],.8)
        grid.revision=5
        motion=control.tick((0,0),0,0,2.)
        self.assertLessEqual(motion.vx,.4)

    def test_arrival_requires_stop_and_then_new_scans(self):
        grid=surface()
        controller=c.CheckpointController(grid,[SimpleNamespace(x=1.,y=0.)])
        grid.revision=5
        controller.tick((0,0),0,0,2.)
        controller.tick((.7,0),0,.2,3.)
        self.assertEqual(controller.state,'SETTLE')
        self.assertEqual(controller.wp_idx,0)
        controller.tick((.72,0),0,.03,3.05)
        controller.tick((.72,0),0,.03,3.6)
        self.assertEqual(controller.wp_idx,1)
        self.assertEqual(controller.tick((.72,0),0,0,3.7),c.Motion())
        self.assertEqual(controller.state,'COMPLETE')

    def test_unknown_map_holds_without_motion(self):
        grid=m.ObservedTerrainMap(world_size=12.)
        controller=c.CheckpointController(grid,[SimpleNamespace(x=3.,y=0.)])
        grid.revision=5
        self.assertEqual(controller.tick((0,0),0,0,2.),c.Motion())
        self.assertEqual(controller.state,'REASSESS')

    def test_braking_opposes_measured_motion(self):
        grid=surface()
        control=c.CheckpointController(grid,[SimpleNamespace(x=3.,y=0.)])
        control.velocity_world=np.array([.5,0.])
        brake=control.position_hold((0,0),0.)
        self.assertEqual(brake.vx,-.4)
        control.velocity_world=np.array([-.2,0.])
        self.assertGreater(control.position_hold((0,0),0.).vx,0.)

    def test_failed_checkpoint_hold_uses_clear_moving_scan(self):
        grid=surface()
        control=c.CheckpointController(grid,[SimpleNamespace(x=4.,y=0.)])
        grid.revision=5
        control.tick((0,0),0,0,2.)
        control.checkpoint=(1.,0.)
        control.pending_arrival=True
        control.is_mission_checkpoint=False
        control.set_state('SETTLE','braking',3.)
        control.tick((.8,0),0,.5,3.,velocity=(.5,0,0))
        control.tick((1.4,0),0,.5,4.2,velocity=(.5,0,0))
        self.assertEqual(control.rolling_checkpoints,1)
        self.assertEqual(control.hold_failures,1)
        self.assertEqual(control.state,'FOLLOW')
        self.assertEqual(control.wp_idx,0)

    def test_operator_stop_never_auto_resumes_on_slope(self):
        grid=surface()
        control=c.CheckpointController(grid,[SimpleNamespace(x=4.,y=0.)])
        control.request('stop',0.)
        brake=control.tick((0,0),0,.5,5.,velocity=(.5,0,0))
        self.assertLess(brake.vx,0.)
        self.assertEqual(control.state,'HOLD')

    def test_scan_hold_failure_can_continue_only_on_observed_route(self):
        for observed in (True,False):
            grid=surface() if observed else m.ObservedTerrainMap(world_size=12.)
            grid.revision=5
            control=c.CheckpointController(grid,[SimpleNamespace(x=3.,y=0.)])
            control.tick((0,0),0,.3,4.,velocity=(.3,0,0))
            self.assertEqual(control.hold_failures,1)
            self.assertEqual(control.state,'FOLLOW' if observed else 'REASSESS')

    def test_replanning_without_displacement_is_bounded(self):
        grid=surface()
        control=c.CheckpointController(grid,[SimpleNamespace(x=3.,y=0.)])
        self.assertTrue(control._plan((0,0),0,1.))
        self.assertFalse(control._plan((0,0),0,22.))
        self.assertIn('without progress',control.planning_failure)

    def test_uphill_approach_keeps_normal_teleop_command(self):
        grid=surface(slope=.6)
        grid.revision=5
        control=c.CheckpointController(grid,[SimpleNamespace(x=.8,y=0.)])
        motion=control.tick((0,0),0,0,2.)
        self.assertGreater(motion.vx,.35)

    def test_short_fresh_corridor_allows_low_speed_approach(self):
        grid=surface(slope=.6)
        control=c.CheckpointController(grid,[SimpleNamespace(x=3.,y=0.)])
        control.path=[(i*.2,0.) for i in range(16)]
        control.checkpoint=(3.,0.)
        control.state='FOLLOW'
        for r in range(grid.n_cells):
            if grid.cell_to_world(r,0)[0]>.85:
                grid.fresh[r,:]=False
        command=control.tick((0,0),0,.2,2.)
        self.assertEqual(control.state,'FOLLOW')
        self.assertGreater(command.vx,0.)

    def test_frontier_inside_mission_boundary_advances_after_hold(self):
        grid=surface()
        control=c.CheckpointController(grid,[SimpleNamespace(x=0.,y=0.)])
        control.path=[(-.15,.46),(-.3,.1)]
        control.checkpoint=(-.3,.1)
        control.is_mission_checkpoint=False
        control.state='FOLLOW'
        control.tick((-.15,.46),0,.02,2.)
        self.assertEqual(control.state,'SETTLE')
        control.tick((-.15,.46),0,0,2.05)
        control.tick((-.15,.46),0,0,2.6)
        self.assertEqual(control.wp_idx,1)

    def test_one_low_speed_sample_does_not_confirm_stop(self):
        grid=surface()
        control=c.CheckpointController(grid,[SimpleNamespace(x=0.,y=0.)])
        control.checkpoint=(0.,0.)
        control.pending_arrival=True
        control.is_mission_checkpoint=True
        control.set_state('SETTLE','arrival',1.)
        control.tick((0,0),0,.01,1.6)
        control.tick((0,0),0,.2,1.8)
        control.tick((0,0),0,.01,2.0)
        self.assertEqual(control.wp_idx,0)
        control.tick((0,0),0,.01,2.6)
        self.assertEqual(control.wp_idx,1)

class RecoveryTests(unittest.TestCase):
    def control(self):
        return c.CheckpointController(surface(),[SimpleNamespace(x=3.,y=0.)])

    def test_blocked_route_rechecks_from_new_position(self):
        control=self.control()
        control.hold('route blocked',1.)
        control.hold_anchor=np.array([0.,0.])
        control.tick((1.,0.),0,0,2.1)
        self.assertEqual(control.state,'FOLLOW')
        self.assertEqual(control.path[0],(1.,0.))
        self.assertEqual(control.wp_idx,0)

    def test_stop_remains_stopped_after_map_changes_and_displacement(self):
        control=self.control()
        control.request('stop',0.)
        control.grid.revision+=5
        control.tick((1.,0.),0,0,20.)
        self.assertEqual(control.state,'HOLD')

    def test_unknown_reassesses_then_stops_without_exploration(self):
        control=self.control()
        control.grid.fresh[:]=False
        control.hold('blocked',0.)
        self.assertEqual(control.tick((0.,0.),0,0,1.1),c.Motion())
        self.assertEqual(control.state,'REASSESS')
        control.tick((0.,0.),0,0,32.)
        self.assertEqual(control.state,'HOLD')

    def test_reverse_checks_obstacles_and_unknown_behind(self):
        control=self.control()
        self.assertTrue(control.motion_clear((0.,0.),0,0,c.Motion(-.2,0)))
        control.grid.fresh[control.grid.world_to_cell(-.2,0.)]=False
        self.assertFalse(control.motion_clear((0.,0.),0,0,c.Motion(-.2,0)))

    def test_local_options_choose_reverse_when_forward_blocked(self):
        control=self.control()
        control.waypoints=[SimpleNamespace(x=-3.,y=0.)]
        control.grid.blocked[control.grid.world_to_cell(.2,0.)]=True
        motion,_=control.local_maneuver((0.,0.),0,0)
        self.assertLess(motion.vx,0.)

    def test_failed_option_not_repeated_at_same_pose(self):
        control=self.control()
        motion,_=control.local_maneuver((0.,0.),0,0)
        control.recovery_attempts.append((motion,(0.,0.),0.))
        other,_=control.local_maneuver((0.,0.),0,0)
        self.assertNotEqual(motion,other)

    def test_obstacle_behind_does_not_veto_forward(self):
        control=self.control()
        control.grid.blocked[control.grid.world_to_cell(-.4,0.)]=True
        motion,_=control.local_maneuver((0.,0.),0,0)
        self.assertGreater(motion.vx,0.)

    def test_maneuver_stops_on_new_obstacle(self):
        control=self.control()
        control.state='MANEUVER'
        control.since=1.
        control.recovery_origin=(0.,0.)
        control.recovery_motion=c.Motion(.2,0.)
        control.grid.blocked[control.grid.world_to_cell(.2,0.)]=True
        control.tick((0.,0.),0,0,1.2)
        self.assertEqual(control.state,'SETTLE')

    def test_turn_choice_matches_left_yaw(self):
        control=self.control()
        control.waypoints=[SimpleNamespace(x=0.,y=3.)]
        control.grid.blocked[control.grid.world_to_cell(.2,0.)]=True
        motion,_=control.local_maneuver((0.,0.),0,0)
        self.assertGreater(motion.omega,0.)

    def test_actual_drift_checked_when_command_is_reverse(self):
        control=self.control()
        control.velocity_world=np.array([.3,0.])
        control.grid.blocked[control.grid.world_to_cell(.2,0.)]=True
        self.assertFalse(control.motion_clear((0.,0.),0,.3,c.Motion(-.2,0.)))

    def test_recovery_execution_is_bounded(self):
        control=self.control()
        control.state='MANEUVER'
        control.since=0.
        control.recovery_origin=(0.,0.)
        control.recovery_motion=c.Motion(.2,.35)
        control.tick((.1,0.),.1,.05,2.1)
        self.assertEqual(control.state,'SETTLE')

    def test_recovery_attempt_limit_requires_operator(self):
        control=self.control()
        control.recovery_anchor=(0.,0.)
        control.recovery_started=0.
        control.recovery_attempts=[(c.Motion(.2,0),(0.,0.),0.)]*6
        control.hold('blocked',0.)
        control.tick((0.,0.),0,0,1.1)
        self.assertEqual(control.state,'HOLD')
        self.assertEqual(control.wp_idx,0)

    def test_reassessment_does_not_pull_toward_old_stop(self):
        control=self.control()
        control.grid.fresh[:]=False
        control.hold('blocked',0.)
        control.hold_anchor=np.array([0.,0.])
        command=control.tick((1.,0.),0,0,1.1)
        self.assertEqual(command,c.Motion())

class AdapterTests(unittest.TestCase):
    def test_directions(self):
        self.assertEqual(d.directional_command(forward=True,left=True),(.4,.35))
        self.assertEqual(d.directional_command(back=True,right=True),(-.4,-.35))
        self.assertEqual(d.directional_command(),(0.,0.))
        self.assertAlmostEqual(c.wrap_angle(-math.pi-.1),math.pi-.1)

    def test_command_reaches_policy_before_inference(self):
        command=np.zeros((1,3))
        cm=SimpleNamespace(get_command=lambda name:command)
        def compute(update_history):
            self.assertFalse(update_history)
            return {'policy':command.copy()}
        env=SimpleNamespace(unwrapped=SimpleNamespace(command_manager=cm,
                            observation_manager=SimpleNamespace(compute=compute)))
        np.testing.assert_allclose(d.policy_actions(env,lambda obs:obs,.4,-.35),[[.4,0,-.35]])
        np.testing.assert_allclose(d.policy_actions(env,lambda obs:obs,0,0),[[0,0,0]])

if __name__=='__main__': unittest.main()
