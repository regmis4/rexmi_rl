"""Measured motion evidence, persistent exclusions and bounded escape behavior."""
import math
import unittest
from types import SimpleNamespace
import numpy as np
from test_checkpoint_navigation import c,m,surface,module
v=module('slip_colors','nav/perception_view.py')


class TrackerTests(unittest.TestCase):
    def trace(self, velocity, command=.4, seconds=3., heading=.0, state='FOLLOW', position=None, cross=0.):
        tracker=c.SlipTracker();event=None
        for t in np.arange(0,seconds+.001,.02):
            xy=np.asarray(velocity)*t if position is None else position(t)
            event=tracker.update(float(t),xy,0,velocity,c.Motion(command,0),state,heading,cross*t)
            if event:break
        return tracker,event

    def test_stall_detected_during_slow_forward_turn(self):
        tracker,event=self.trace((0,0),command=.15,heading=.8)
        self.assertEqual(event['kind'],'poor drive progress')
        self.assertAlmostEqual(event['window_s'],3.)

    def test_backward_slide_detected_before_normal_stall_timeout(self):
        tracker,event=self.trace((-.15,0))
        self.assertEqual(event['kind'],'backsliding')
        self.assertLess(event['window_s'],2.)

    def test_sideways_motion_with_growing_route_error(self):
        _,event=self.trace((.15,.2),cross=.2)
        self.assertEqual(event['kind'],'sideways slip')

    def test_velocity_jitter_without_displacement_is_not_progress(self):
        _,event=self.trace((.2,0),position=lambda t:(0,0))
        self.assertEqual(event['kind'],'poor drive progress')

    def test_slow_success_is_not_a_stall(self):
        _,event=self.trace((.06,0),command=.15)
        self.assertIsNone(event)

    def test_stops_manual_alignment_and_reverse_do_not_mark_terrain(self):
        for state,command,heading in [('HOLD',.4,0),('MANUAL',.4,0),('OBSERVE',.4,0),
                                      ('FOLLOW',0.,0),('FOLLOW',-.4,0),('FOLLOW',.15,1.5)]:
            _,event=self.trace((0,0),state=state,command=command,heading=heading)
            self.assertIsNone(event)


class FailureMapTests(unittest.TestCase):
    def test_vectorized_search_costs_match_route_costs(self):
        g=surface(.4,True)
        g.record_difficulty((0,0),0)
        offsets=((1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1))
        for profile in ('direct','balanced','gentle'):
            edges,turn=g.directional_costs(profile)
            for xy in ((0,0),(.5,.5),(2,2),(-1,1)):
                cell=g.world_to_cell(*xy)
                for heading,(dr,dc) in enumerate(offsets):
                    for previous in range(8):
                        expected=g.edge_cost(cell,dr,dc,previous,profile)
                        actual=edges[*cell,heading]+turn[cell]*min((heading-previous)%8,(previous-heading)%8)
                        if math.isinf(expected):self.assertTrue(math.isinf(actual))
                        else:self.assertAlmostEqual(actual,expected,places=5)

    def test_failed_source_direction_matches_planner_and_follower(self):
        g=surface();g.record_traversal([(0,0)],0,failed=True);g.record_traversal([(0,0)],0,failed=True)
        path,complete=c.observed_route(g,(.03,.04),(3.,0.),0)
        self.assertTrue(complete)
        self.assertTrue(all(g.segment_clear(a,b) for a,b in zip(path,path[1:])))

    def test_first_episode_blocks_approach_second_marks_zone(self):
        g=surface();cell=g.world_to_cell(0,0)
        g.record_difficulty((0,0),0)
        self.assertFalse(g.direction_allowed(cell,0))
        self.assertFalse(g.direction_allowed(cell,1))
        self.assertTrue(g.direction_allowed(cell,4))
        self.assertFalse(g.no_go[cell])
        g.record_difficulty((0,0),math.pi/2)
        self.assertTrue(g.no_go[cell]);self.assertFalse(g.traversable(cell))
        self.assertEqual(g.cost[cell],20.)

    def test_learned_no_go_allows_only_checked_outward_escape(self):
        g=surface()
        for _ in range(2):g.record_difficulty((0,0),0)
        self.assertFalse(g.segment_clear((0,0),(-1,0)))
        self.assertTrue(g.segment_clear((0,0),(-1,0),no_go_exit=True))
        self.assertFalse(g.segment_clear((-1,0),(0,0),no_go_exit=True))
        g.hazard[g.world_to_cell(-.5,0)]=True;g.rebuild(0)
        # Use explicit map geometry, since rebuild intentionally recomputes hazards.
        cell=g.world_to_cell(-.5,0);g.hazard[cell]=g.blocked[cell]=True
        self.assertFalse(g.segment_clear((0,0),(-1,0),no_go_exit=True))

    def test_escape_does_not_bypass_unknown_or_steep_ground(self):
        for unknown in (True,False):
            g=surface()
            for _ in range(2):g.record_difficulty((0,0),0)
            cell=g.world_to_cell(-.5,0)
            if unknown:g.known[cell]=False
            else:g.slope[cell]=1.;g.blocked[cell]=True
            self.assertFalse(g.segment_clear((0,0),(-1,0),no_go_exit=True))

    def test_failure_memory_survives_scan_retention_and_exports(self):
        g=surface();cell=g.world_to_cell(3,0)
        for _ in range(2):g.record_difficulty((3,0),0)
        g.retain_radius((0,0),1.);g.rebuild(10.)
        self.assertTrue(g.no_go[cell]);self.assertFalse(g.known[cell])
        self.assertTrue(g.survey_snapshot()['no_go'][cell])
        self.assertTrue(g.survey_snapshot()['approach_blocked'][*cell,0])

    def test_rendered_learned_no_go_matches_blocked_mask(self):
        g=surface()
        for _ in range(2):g.record_difficulty((0,0),0)
        rgb,legend=v.survey_colors(g.cost,g.known,g.clearance_mask,no_go=g.no_go)
        np.testing.assert_allclose(rgb[g.no_go],np.tile([.60,.05,.08],(int(g.no_go.sum()),1)),atol=1e-7)
        self.assertTrue(g.blocked[g.no_go].all());self.assertIn('learned no-go',legend)


class RecoveryTests(unittest.TestCase):
    def controller(self):
        g=surface();ctrl=c.CheckpointController(g,[SimpleNamespace(x=3.,y=0.)])
        return g,ctrl

    def test_confirmed_failure_brakes_then_uses_different_motion(self):
        g,ctrl=self.controller();ctrl.begin_escape((0,0),0,0,{'kind':'poor drive progress'})
        ctrl.tick((0,0),0,.3,.1,velocity=(.3,0,0))
        self.assertEqual(ctrl.state,'ESCAPE_BRAKE')
        for t in (.2,.42):command=ctrl.tick((0,0),0,0,t)
        self.assertEqual(ctrl.state,'ESCAPE_MOVE')
        self.assertTrue(command.vx<0 or command.vx==0 and command.omega!=0)
        self.assertEqual(ctrl.wp_idx,0)
        trace=ctrl.escape_events[-1]
        self.assertEqual(trace['chosen'],dict(vx=command.vx,omega=command.omega))
        self.assertTrue(any(p.get('reason') for p in trace['candidates'] if not p['accepted']))

    def test_operator_stop_cancels_escape(self):
        g,ctrl=self.controller();ctrl.begin_escape((0,0),0,0,{'kind':'backsliding'})
        ctrl.request('stop',.1)
        for t in (.2,2.,5.):ctrl.tick((0,0),0,0,t)
        self.assertEqual(ctrl.state,'HOLD');self.assertEqual(ctrl.climb_failures,1)

    def test_small_gait_oscillation_allows_checked_recovery(self):
        g,ctrl=self.controller();ctrl.begin_escape((0,0),0,0,{'kind':'poor drive progress'})
        for t in (0.,.1,.2,.3):
            xy=(.008*math.sin(t*2*math.pi/.3),0.)
            ctrl.velocity_world=np.array([.16,0.])
            command=ctrl.escape_step(xy,0,.16,t)
        self.assertEqual(ctrl.state,'ESCAPE_MOVE')
        self.assertTrue(ctrl.motion_clear(xy,0,.16,command))

    def test_sustained_drift_does_not_pass_gait_settling(self):
        g,ctrl=self.controller();ctrl.begin_escape((0,0),0,0,{'kind':'backsliding'})
        for t in np.arange(0.,3.11,.1):
            ctrl.velocity_world=np.array([-.16,0.])
            ctrl.escape_step((-.16*t,0.),0,.16,t)
        self.assertEqual(ctrl.state,'HOLD')
        self.assertIn('cannot brake',ctrl.reason)

    def test_transient_drift_rejection_rechecks_before_giving_up(self):
        g,ctrl=self.controller();xy=(0,.1)
        ctrl.begin_escape(xy,0,0,{'kind':'backsliding'})
        # A side obstacle intersects the current drift stopping corridor, but
        # the robot has clear rear terrain once braking removes that drift.
        g.obstacle_distance[g.world_to_cell(0,.3)]=.29
        g.blocked[g.world_to_cell(0,.3)]=True
        for t in (0.,.1,.2,.3):
            ctrl.velocity_world=np.array([0.,.16])
            ctrl.escape_step(xy,0,.16,t)
        self.assertEqual(ctrl.state,'ESCAPE_BRAKE')
        self.assertIsNone(ctrl.escape_events[-1]['chosen'])
        self.assertIn('rechecking',ctrl.reason)
        self.assertTrue(all('drift stopping corridor' in p['reason']
                            for p in ctrl.escape_events[-1]['candidates']))
        ctrl.velocity_world=np.zeros(2)
        for t in (.5,.72):ctrl.escape_step(xy,0,0,t)
        self.assertEqual(ctrl.state,'ESCAPE_MOVE')

    def test_rejection_cycle_is_not_claimed_as_tire_slip(self):
        g,ctrl=self.controller()
        ctrl.begin_escape((0,0),0,0,{'kind':'repeated command rejection'},mark=False)
        self.assertFalse(g.approach_blocked.any());self.assertFalse(g.no_go.any())
        self.assertEqual(ctrl.climb_failures,0)

    def test_escape_attempts_are_bounded(self):
        g,ctrl=self.controller();ctrl.begin_escape((0,0),0,0,{'kind':'backsliding'})
        ctrl.tick((0,0),0,0,16.)
        self.assertEqual(ctrl.state,'HOLD')
        self.assertIn('exhausted',ctrl.reason)

    def test_reentering_recovery_keeps_attempts_and_time_budget(self):
        g,ctrl=self.controller();ctrl.begin_escape((0,0),0,0,{'kind':'backsliding'})
        ctrl.escape_attempts.append((c.Motion(-.4,0),(0,0),0))
        ctrl.set_state('FOLLOW','retry',5.)
        ctrl.begin_escape((.1,0),0,7.,{'kind':'backsliding'})
        self.assertEqual(len(ctrl.escape_attempts),1)
        self.assertEqual(ctrl.escape_total_started,0.)
        ctrl.tick((.1,0),0,0,16.)
        self.assertEqual(ctrl.state,'HOLD')
        ctrl.begin_escape((2,0),0,20.,{'kind':'backsliding'})
        self.assertFalse(ctrl.escape_attempts)
        self.assertEqual(ctrl.escape_total_started,20.)

    def test_uphill_comparison_returns_valid_bounded_route(self):
        g=surface(.45);ctrl=c.CheckpointController(g,[SimpleNamespace(x=3.,y=0.)])
        path,done=ctrl.choose_route((-3.,0.),(3.,0.),0)
        self.assertTrue(done)
        self.assertTrue(all(g.segment_clear(a,b) for a,b in zip(path,path[1:])))
        lengths=[p['length_m'] for p in ctrl.route_options if p['score'] is not None]
        self.assertLessEqual(g.route_profile(path)['length_m'],max(min(lengths)*2.,min(lengths)+6.))
        self.assertTrue(any(p.get('selected') for p in ctrl.route_options))

    def test_uphill_frontier_also_compares_approaches(self):
        g=surface(.45);ctrl=c.CheckpointController(g,[SimpleNamespace(x=5.8,y=0.)])
        path,done=ctrl.choose_route((-3.,0.),(5.8,0.),0)
        self.assertFalse(done)
        self.assertGreater(len(ctrl.route_options),1)
        self.assertFalse(any(p['complete'] for p in ctrl.route_options))
        self.assertTrue(all(g.segment_clear(a,b) for a,b in zip(path,path[1:])))
        self.assertTrue(g.known[g.world_to_cell(*path[-1])])

    def test_already_reached_cell_centre_remains_a_valid_plan(self):
        g,ctrl=self.controller()
        xy=g.cell_to_world(*g.world_to_cell(0,0))
        path,done=ctrl.choose_route(xy,xy,0)
        self.assertTrue(done)
        self.assertTrue(math.isfinite(ctrl.route_score(path,0)))


if __name__=='__main__':unittest.main()
