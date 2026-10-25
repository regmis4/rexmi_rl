"""Cliff-turn recovery and whole-route grade tradeoffs, independent of Isaac."""
import math
import unittest
from types import SimpleNamespace
import numpy as np
from test_checkpoint_navigation import surface, m, c


def cliff_case():
    grid=surface()
    for row in range(grid.n_cells):
        if grid.cell_to_world(row,0)[0]>.4:
            grid.slope[row,:]=math.tan(math.radians(45))
            grid.gx[row,:]=1.
            grid.blocked[row,:]=True
            grid.cost[row,:]=20.
    control=c.CheckpointController(grid,[SimpleNamespace(x=0.,y=3.)])
    control.state='FOLLOW';control.path=[(0.,0.),(0.,1.),(0.,3.)]
    control.checkpoint=(0.,3.)
    return grid,control


class CliffTurnTests(unittest.TestCase):
    def test_turn_checks_body_footprint_not_only_centre(self):
        grid,control=cliff_case()
        self.assertTrue(grid.traversable(grid.world_to_cell(0,0)))
        self.assertFalse(control.motion_clear((0,0),0,0,c.Motion(0,.2)))
        self.assertIn('turn envelope',control.motion_rejection)

    def test_yaw_only_command_does_not_skip_drift_check(self):
        grid,control=cliff_case();control.velocity_world=np.array([.5,0.])
        self.assertFalse(control.motion_clear((0,0),0,.5,c.Motion(0,.2)))
        self.assertIn('drift stopping corridor',control.motion_rejection)

    def test_old_perimeter_geometry_does_not_erase_fresh_motion_corridor(self):
        grid=surface();control=c.CheckpointController(grid,[SimpleNamespace(x=3.,y=0.)])
        cell=grid.world_to_cell(.3,.3);grid.fresh[cell]=False
        self.assertTrue(control.motion_clear((0,0),0,0,c.Motion(.2,0)))
        grid.slope[cell]=1.
        self.assertFalse(control.motion_clear((0,0),0,0,c.Motion(.2,0)))
        self.assertIn('slope >35',control.motion_rejection)

    def test_reverse_starts_only_after_stable_stop(self):
        grid,control=cliff_case()
        control.tick((0,0),0,.2,0.,velocity=(.2,0.,0.))
        self.assertEqual(control.state,'TURN_BRAKE')
        self.assertLessEqual(control.tick((0,0),0,.2,.2,velocity=(.2,0.,0.)).vx,0.)
        self.assertEqual(control.state,'TURN_BRAKE')
        for t in (.4,.8): control.tick((0,0),0,0,t)
        self.assertEqual(control.state,'TURN_REVERSE')
        motion=control.tick((0,0),0,0,.9)
        self.assertEqual(motion.vx,-.15)
        self.assertTrue(grid.segment_clear((0,0),control.reverse_target,grid.footprint_radius))
        self.assertEqual(control.wp_idx,0)

    def test_stop_after_reverse_then_resume_normal_turn_when_clear(self):
        grid,control=cliff_case()
        self.assertTrue(control.prepare_turn((0,0),0,0,math.pi/2))
        rear=control.reverse_target
        control.set_state('TURN_REVERSE','test',1.)
        control.tick(rear,0,.12,2.,velocity=(-.12,0.,0.))
        self.assertEqual(control.state,'TURN_STOP')
        for t in (2.2,2.6): control.tick(rear,0,0,t)
        self.assertEqual(control.state,'TURN_ALIGN')
        command=control.tick(rear,0,0,2.7)
        self.assertEqual(control.state,'FOLLOW')
        command=control.tick(rear,0,0,2.72)
        self.assertGreater(command.omega,0.)
        self.assertLessEqual(command.omega,.35)
        self.assertGreaterEqual(command.vx,0.)

    def test_no_reverse_into_unknown_or_stale_rear(self):
        for field in ('known','fresh'):
            grid,control=cliff_case()
            for row in range(grid.n_cells):
                if grid.cell_to_world(row,0)[0]<-.4:getattr(grid,field)[row,:]=False
            self.assertFalse(control.prepare_turn((0,0),0,0,math.pi/2))
            command=control.tick((0,0),0,0,1.)
            self.assertNotEqual(control.state,'TURN_REVERSE')
            self.assertEqual(command,c.Motion())

    def test_reverse_ends_as_soon_as_turn_has_room(self):
        grid,control=cliff_case()
        self.assertTrue(control.prepare_turn((0,0),0,0,math.pi/2))
        control.set_state('TURN_REVERSE','test',1.)
        self.assertGreater(math.dist((-.2,0),control.reverse_target),.12)
        control.tick((-.2,0),0,.1,2.,velocity=(-.1,0.,0.))
        self.assertEqual(control.state,'TURN_STOP')

    def test_new_rear_obstacle_interrupts_reverse(self):
        grid,control=cliff_case();control.prepare_turn((0,0),0,0,math.pi/2)
        control.set_state('TURN_REVERSE','test',1.)
        cell=grid.world_to_cell(*control.reverse_target)
        grid.hazard[cell]=grid.blocked[cell]=True
        control.tick((0,0),0,0,1.1)
        self.assertEqual(control.state,'SETTLE')
        self.assertEqual(control.wp_idx,0)

    def test_manual_stop_cancels_turn_sequence(self):
        grid,control=cliff_case();control.prepare_turn((0,0),0,0,math.pi/2)
        control.request('stop',.1)
        for t in (1.,5.,15.): control.tick((0,0),0,0,t)
        self.assertEqual(control.state,'HOLD')
        self.assertEqual(control.turn_repositions,1)

    def test_reverse_attempts_are_bounded_at_same_location(self):
        grid,control=cliff_case()
        for t in range(3): self.assertTrue(control.prepare_turn((0,0),0,t,math.pi/2))
        self.assertFalse(control.prepare_turn((0,0),0,4,math.pi/2))

    def test_ineffective_creep_uses_checked_teleop_reverse(self):
        grid,control=cliff_case();control.prepare_turn((0,0),0,0,math.pi/2)
        control.set_state('TURN_REVERSE','test',1.)
        command=control.tick((0,0),0,0,3.2)
        self.assertEqual(command.vx,-.4)
        control.tick((0,0),0,0,7.2)
        self.assertEqual(control.state,'HOLD')
        self.assertIn('no travel',control.reason)

    def test_reverse_boost_cannot_extend_into_unknown_stopping_room(self):
        grid,control=cliff_case();control.prepare_turn((0,0),0,0,math.pi/2)
        for row in range(grid.n_cells):
            if grid.cell_to_world(row,0)[0]<-1.:grid.known[row,:]=False
        control.set_state('TURN_REVERSE','test',1.)
        command=control.tick((0,0),0,0,3.2)
        self.assertEqual(command.vx,-.15)
        self.assertEqual(control.state,'TURN_REVERSE')

    def test_clear_flat_ground_exits_slow_alignment_immediately(self):
        grid=surface();control=c.CheckpointController(grid,[SimpleNamespace(x=0.,y=3.)])
        grid.failures[:,:,0]=2
        control.state='TURN_ALIGN';control.turn_heading=math.pi/2
        control.turn_origin=(0.,0.);control.turn_progress_at=0.
        command=control.tick((0,0),0,0,.1)
        self.assertEqual(control.state,'FOLLOW')
        self.assertEqual(command.vx,0.)
        command=control.tick((0,0),0,0,.12)
        self.assertGreater(command.omega,.2)
        self.assertGreaterEqual(command.vx,0.)

    def test_loss_of_slope_evidence_does_not_continue_reverse_alignment(self):
        grid=surface();control=c.CheckpointController(grid,[SimpleNamespace(x=3.,y=0.)])
        control.state='TURN_ALIGN';control.turn_heading=.6;control.turn_origin=(0.,0.)
        grid.known[:]=False
        control.tick((0,0),0,0,.1)
        self.assertEqual(control.state,'SETTLE')
        self.assertIn('unavailable route',control.reason)

    def test_cleared_danger_brakes_existing_reverse_before_normal_follow(self):
        grid=surface();control=c.CheckpointController(grid,[SimpleNamespace(x=3.,y=0.)])
        control.state='TURN_ALIGN';control.turn_heading=.6;control.turn_origin=(0.,0.)
        control.tick((0,0),0,.15,.1,velocity=(-.15,0,0))
        self.assertEqual(control.state,'TURN_STOP')

    def test_successful_turn_does_not_consume_failed_attempt_budget(self):
        grid,control=cliff_case();control.prepare_turn((0,0),0,0,math.pi/2)
        rear=control.reverse_target
        control.state='TURN_ALIGN';control.turn_origin=rear
        control.tick(rear,math.pi/2,0,1.)
        self.assertFalse(control.turn_attempts)
        self.assertEqual(control.state,'FOLLOW')

    def test_flat_and_permitted_slopes_cannot_start_reverse_turn_mode(self):
        for degrees in (0.,5.,15.,30.):
            grid=surface(math.tan(math.radians(degrees)))
            control=c.CheckpointController(grid,[SimpleNamespace(x=0.,y=3.)])
            self.assertFalse(control.prepare_turn((0,0),0,0,math.pi/2))
            self.assertEqual(control.turn_repositions,0)

    def test_flat_unknown_extra_reserve_does_not_delay_normal_turn(self):
        grid=surface();grid.known[grid.world_to_cell(.5,.1)]=False
        control=c.CheckpointController(grid,[SimpleNamespace(x=0.,y=3.)])
        control.state='FOLLOW';control.path=[(0.,0.),(0.,3.)];control.checkpoint=(0.,3.)
        for t in (0.,.1,.2):
            command=control.tick((0,0),0,0,t)
            self.assertEqual(control.state,'FOLLOW')
            self.assertAlmostEqual(command.omega,.35)
            self.assertEqual(control.turn_repositions,0)

    def test_clear_forward_turn_continues_on_safe_slope(self):
        for degrees in (0.,15.,24.):
            grid=surface(math.tan(math.radians(degrees)))
            control=c.CheckpointController(grid,[SimpleNamespace(x=3.,y=0.)])
            control.state='FOLLOW';control.path=[(0.,0.),(3.,0.)];control.checkpoint=(3.,0.)
            command=control.tick((0,0),-.6,0,0)
            self.assertEqual(control.state,'FOLLOW')
            self.assertGreater(command.vx,.3)
            self.assertEqual(command.omega,.35)
            self.assertEqual(control.turn_repositions,0)

    def test_steep_terrain_behind_does_not_trigger_reverse(self):
        grid=surface()
        for row in range(grid.n_cells):
            if grid.cell_to_world(row,0)[0]<-1.:
                grid.slope[row,:]=1.;grid.blocked[row,:]=True
        control=c.CheckpointController(grid,[SimpleNamespace(x=3.,y=2.)])
        self.assertFalse(control.prepare_turn((0,0),0,0,.6))

    def test_no_new_reverse_when_unknown_footprint_requires_stop(self):
        grid=surface();grid.known[grid.world_to_cell(.3,.1)]=False
        control=c.CheckpointController(grid,[SimpleNamespace(x=0.,y=3.)])
        control.state='FOLLOW';control.path=[(0.,0.),(0.,3.)];control.checkpoint=(0.,3.)
        command=control.tick((0,0),0,0,0)
        self.assertEqual(control.state,'SETTLE')
        self.assertEqual(command,c.Motion())
        self.assertEqual(control.turn_repositions,0)

    def test_slow_forward_arc_is_tried_before_reversing_near_steep_edge(self):
        grid=surface()
        for row in range(grid.n_cells):
            for col in range(grid.n_cells):
                x,y=grid.cell_to_world(row,col)
                if x>.85 and y<.2:grid.slope[row,col]=1.;grid.blocked[row,col]=True
        control=c.CheckpointController(grid,[SimpleNamespace(x=3.,y=2.)])
        control.state='FOLLOW';control.path=[(0.,0.),(3.,2.)];control.checkpoint=(3.,2.)
        command=control.tick((0,0),0,0,0)
        self.assertEqual(control.state,'FOLLOW')
        self.assertGreater(command.vx,0.)
        self.assertLessEqual(command.vx,.2)
        self.assertEqual(control.turn_repositions,0)


class WholeRouteTests(unittest.TestCase):
    def test_larger_route_keeps_body_off_steep_cells_not_only_centre(self):
        grid=surface()
        for row in range(grid.n_cells):
            for col in range(grid.n_cells):
                x,y=grid.cell_to_world(row,col)
                if -1.<x<1. and y>.5:
                    grid.slope[row,col]=1.;grid.blocked[row,col]=True;grid.cost[row,col]=20.
        route,complete=c.observed_route(grid,(-2.,.4),(2.,.4))
        self.assertTrue(complete)
        self.assertTrue(all(not grid.support_reason(a,b) for a,b in zip(route,route[1:])))
        self.assertTrue(grid.traversable(grid.world_to_cell(0,.4)))  # No extra coloured buffer.
        self.assertFalse(grid.planning_support()[grid.world_to_cell(0,.4)])

    def test_centre_observation_alone_cannot_authorize_an_unsupported_start(self):
        grid=surface();grid.known[:]=False
        cell=grid.world_to_cell(0,0);grid.known[cell]=True
        self.assertTrue(grid.traversable(cell))
        self.assertEqual(c.observed_route(grid,(0,0),(2,0)),([],False))

    def test_cached_route_is_replaced_when_new_cliff_overlaps_body(self):
        grid=surface()
        control=c.CheckpointController(grid,[SimpleNamespace(x=2.,y=.4)])
        start=(-2.,.4)
        self.assertTrue(control._plan(start,0.,0.))
        old_route=list(control.mission_route)
        for row in range(grid.n_cells):
            for col in range(grid.n_cells):
                x,y=grid.cell_to_world(row,col)
                if -1.<x<1. and y>.5:
                    grid.slope[row,col]=1.;grid.blocked[row,col]=True
        self.assertTrue(all(grid.segment_clear(a,b) for a,b in zip(old_route,old_route[1:])))
        self.assertTrue(math.isinf(grid.route_cost(old_route)))
        self.assertTrue(control._plan(start,0.,1.))
        self.assertNotEqual(control.mission_route,old_route)
        self.assertTrue(math.isfinite(grid.route_cost(control.mission_route)))

    def test_cliff_proximity_changes_preference_without_enlarging_blocked_tiles(self):
        grid=surface();cell=grid.world_to_cell(0,0)
        before=grid.edge_cost(cell,1,0,0);tiles=grid.cost.copy();blocked=grid.blocked.copy()
        grid.steep_distance[cell]=.2
        self.assertGreater(grid.edge_cost(cell,1,0,0),before+5.)
        np.testing.assert_array_equal(grid.cost,tiles)
        np.testing.assert_array_equal(grid.blocked,blocked)

    def test_failed_drive_is_learned_even_when_fitted_slope_looks_easy(self):
        grid=surface(.12);control=c.CheckpointController(grid,[SimpleNamespace(x=3.,y=0.)])
        control.state='FOLLOW';control.path=[(0.,0.),(3.,0.)];control.checkpoint=(3.,0.)
        control.previous_command=c.Motion(.4,0.)
        for t in np.arange(0,2.1,.02):control.tick((0,0),0,0,float(t),velocity=(0,0,0))
        self.assertEqual(control.climb_failures,1)
        self.assertIn('poor drive progress',control.last_climb_event)
        self.assertFalse(control.mission_route)

    def test_velocity_without_net_travel_does_not_count_as_success(self):
        grid=surface(.3);control=c.CheckpointController(grid,[SimpleNamespace(x=3.,y=0.)])
        control.state='FOLLOW';control.path=[(0.,0.),(3.,0.)];control.checkpoint=(3.,0.)
        control.previous_command=c.Motion(.4,0.)
        for t in np.arange(0,2.1,.02):control.tick((0,0),0,.15,float(t),velocity=(.15,0,0))
        self.assertEqual(control.climb_failures,1)
        self.assertFalse(grid.successes.any())

    def routes(self):
        # Two observed corridors. The gentler option first heads away from the
        # goal, outside the local 2–4 m checkpoint horizon, before climbing.
        grid=m.ObservedTerrainMap(world_size=20.,cell_size=.2,footprint_radius=.3)
        grid.known[:]=True;grid.fresh[:]=True;grid.blocked[:]=True;grid.cost[:]=20.
        for r in range(grid.n_cells):
            for col in range(grid.n_cells):
                x,y=grid.cell_to_world(r,col)
                direct=-4.2<x<4.2 and abs(y)<.3
                detour=((abs(x+4)<.3 or abs(x-4)<.3) and -3.2<y<.3
                        or -4.2<x<4.2 and abs(y+3)<.3)
                if direct or detour:
                    grade=32. if direct and abs(x)<2. else (20. if abs(y+3)<.3 and abs(x)<3.2 else 0.)
                    slope=math.tan(math.radians(grade))
                    grid.blocked[r,col]=False;grid.slope[r,col]=grid.gx[r,col]=slope
                    grid.cost[r,col]=1.+4.*slope
        return grid,(-4.,0.),(4.,0.)

    def test_long_gentle_exit_beats_short_steep_exit(self):
        grid,start,goal=self.routes()
        route,complete=c.observed_route(grid,start,goal,0.)
        self.assertTrue(complete)
        self.assertLess(min(p[1] for p in route),-2.5)
        profile=grid.route_profile(route)
        self.assertLessEqual(profile['max_uphill_deg'],21.)
        self.assertEqual(profile['uphill_over_25deg_m'],0.)
        self.assertGreater(profile['length_m'],12.)
        self.assertTrue(all(grid.segment_clear(a,b) for a,b in zip(route,route[1:])))

    def test_checkpoint_is_prefix_of_larger_exit_plan(self):
        grid,start,goal=self.routes()
        control=c.CheckpointController(grid,[SimpleNamespace(x=goal[0],y=goal[1])])
        self.assertTrue(control._plan(start,0.,1.))
        self.assertTrue(control.mission_route_complete)
        self.assertEqual(control.path,control.mission_route[:len(control.path)])
        self.assertGreater(len(control.mission_route),len(control.path))
        self.assertLess(control.checkpoint[1],-2.)
        self.assertEqual(control.wp_idx,0)

    def test_uphill_cost_grows_non_linearly_and_sidehill_is_penalized(self):
        grid=surface();cell=grid.world_to_cell(0,0)
        costs=[]
        for deg in (15.,25.,35.):
            grid.gx[cell]=grid.slope[cell]=math.tan(math.radians(deg))
            grid.cost[cell]=1.+4.*grid.slope[cell]
            costs.append(grid.edge_cost(cell,1,0,0))
        self.assertGreater(costs[2]-costs[1],costs[1]-costs[0])
        self.assertGreater(grid.edge_cost(cell,1,0,0),grid.edge_cost(cell,-1,0,4))
        self.assertGreater(grid.edge_cost(cell,0,1,2),grid.edge_cost(cell,-1,0,4))
        self.assertTrue(grid.traversable(cell))  # Costs are preferences, not new slope exclusions.

    def test_old_surface_is_available_to_full_route_but_not_immediate_motion(self):
        grid,start,goal=self.routes()
        for r in range(grid.n_cells):
            if grid.cell_to_world(r,0)[0]>-1.:grid.fresh[r,:]=False
        route,complete=c.observed_route(grid,start,goal,0.)
        self.assertTrue(complete)
        self.assertTrue(math.isfinite(grid.route_cost(route)))
        self.assertFalse(grid.segment_clear(route[-2],route[-1]))


if __name__=='__main__':unittest.main()
