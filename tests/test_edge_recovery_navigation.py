"""Escaping an existing partial cliff overlap must differ from entering one."""
import math
import unittest
from types import SimpleNamespace
import numpy as np
from test_checkpoint_navigation import c,m,surface


def edge(behind=False):
    g=surface(.60)
    xy=(.1,.1)
    # One perimeter cell at 35.8 degrees, while the centre is 31 degrees.
    cliff=g.world_to_cell(-.3 if behind else .5,.1)
    g.slope[cliff]=math.tan(math.radians(35.8));g.blocked[cliff]=True
    ctrl=c.CheckpointController(g,[SimpleNamespace(x=3.,y=0.)])
    ctrl.state='REASSESS'
    return g,ctrl,xy,cliff


class EdgeRecoveryTests(unittest.TestCase):
    def test_reverse_leaves_existing_overlap_above_normal_ascent_limit(self):
        # Point downhill toward a cliff behind the lower X side instead.
        g,ctrl,xy,cliff=edge(behind=True)
        yaw=math.pi
        self.assertIn('slope >35',g.support_reason(xy))
        self.assertFalse(ctrl.motion_clear(xy,yaw,0,c.Motion(-.2,0)))
        command=ctrl.tick(xy,yaw,0,1,upright=.75,velocity=(0,0,0))
        self.assertEqual(ctrl.state,'EDGE_MOVE')
        self.assertEqual(command.vx,-.4)
        self.assertEqual(command.omega,0)
        self.assertEqual(g.uphill_limit_deg,25.)
        self.assertTrue(g.blocked[cliff])

    def test_edge_behind_selects_forward_escape(self):
        g,ctrl,xy,cliff=edge(behind=True)
        command=ctrl.tick(xy,0,0,1,upright=.75,velocity=(0,0,0))
        self.assertGreater(command.vx,0)
        self.assertEqual(ctrl.state,'EDGE_MOVE')

    def test_footprint_may_not_add_another_steep_cell(self):
        g,ctrl,xy,cliff=edge()
        extra=g.world_to_cell(-.7,.1);g.slope[extra]=1.;g.blocked[extra]=True
        inherited=g.support_exposure(xy)[0]
        reason,_=g.support_exit_check(ctrl.edge_arc(xy,0,.8,-1),inherited)
        self.assertIn('exposure',reason)

    def test_unknown_newly_swept_ground_blocks_backout(self):
        g,ctrl,xy,cliff=edge()
        g.known[g.world_to_cell(-.7,.1)]=False
        reason,_=g.support_exit_check(ctrl.edge_arc(xy,0,.6,-1),g.support_exposure(xy)[0])
        self.assertIn('unobserved',reason)

    def test_stale_newly_swept_ground_blocks_backout(self):
        g,ctrl,xy,cliff=edge()
        g.fresh[g.world_to_cell(-.7,.1)]=False
        reason,_=g.support_exit_check(ctrl.edge_arc(xy,0,.6,-1),g.support_exposure(xy)[0])
        self.assertIn('stale',reason)

    def test_rear_boulder_is_not_relaxed_by_edge_escape(self):
        g,ctrl,xy,cliff=edge()
        for x in np.arange(-1.,.01,.1):g.obstacle_distance[g.world_to_cell(x,.1)]=.1
        self.assertFalse(ctrl.prepare_edge_exit(xy,0,1))
        self.assertTrue(any('obstacle' in e['reason'] for e in ctrl.edge_evaluations))

    def test_body_centre_on_excessive_slope_cannot_use_exception(self):
        g,ctrl,xy,cliff=edge()
        cell=g.world_to_cell(*xy);g.slope[cell]=1.;g.blocked[cell]=True
        self.assertFalse(ctrl.prepare_edge_exit(xy,0,1))

    def test_large_unsupported_overlap_is_not_a_backout_permission(self):
        g,ctrl,xy,cliff=edge()
        cells=g._support_cells(xy)
        for p in list(zip(*cells))[:len(cells[0])//2]:g.slope[p]=1.
        self.assertFalse(ctrl.prepare_edge_exit(xy,0,1))

    def test_cannot_clear_then_reenter_same_cliff_cell(self):
        g,ctrl,xy,cliff=edge()
        points=ctrl.edge_arc(xy,0,.4,-1)
        reason,_=g.support_exit_check(points+points[::-1],g.support_exposure(xy)[0])
        self.assertIn('re-enters',reason)

    def test_no_support_exception_on_normal_flat_ground(self):
        g=surface();ctrl=c.CheckpointController(g,[SimpleNamespace(x=3.,y=0.)])
        self.assertFalse(ctrl.prepare_edge_exit((0,0),0,1))

    def test_measured_drift_keeps_strong_checked_counter_command(self):
        g,ctrl,xy,cliff=edge()
        ctrl.upright=.75;self.assertTrue(ctrl.prepare_edge_exit(xy,0,1))
        ctrl.state='EDGE_MOVE';ctrl.edge_plan['move_started']=1.
        cmd=ctrl.tick(xy,0,.5,1.1,upright=.75,velocity=(.5,0,0))
        self.assertLess(cmd.vx,0)
        self.assertIn('checked escape',ctrl.reason)

    def test_zero_progress_uses_teleop_reverse_then_a_different_checked_arc(self):
        g,ctrl,xy,cliff=edge()
        ctrl.tick(xy,0,0,1,upright=.75,velocity=(0,0,0))
        command=ctrl.tick(xy,0,0,2.6,upright=.75,velocity=(0,0,0))
        self.assertEqual(command.vx,-.4)
        ctrl.tick(xy,0,0,5.1,upright=.75,velocity=(0,0,0))
        self.assertEqual(ctrl.state,'EDGE_BRAKE');self.assertIn('no travel',ctrl.reason)
        self.assertNotEqual(ctrl.edge_plan['curvature'],0.)
        self.assertEqual(len(ctrl.edge_attempts),2)

    def test_support_improvement_then_stop_and_reobserve(self):
        g,ctrl,xy,cliff=edge()
        ctrl.tick(xy,0,0,1,upright=.75,velocity=(0,0,0))
        target=ctrl.edge_plan['target']
        self.assertEqual(ctrl.checkpoint,target)
        self.assertFalse(ctrl.is_mission_checkpoint)
        ctrl.tick(target,0,0,2.,upright=.8,velocity=(0,0,0))
        self.assertEqual(ctrl.state,'EDGE_STOP')
        for t in (2.1,2.5):ctrl.tick(target,0,0,t,upright=.8,velocity=(0,0,0))
        self.assertEqual(ctrl.state,'OBSERVE');self.assertEqual(ctrl.wp_idx,0)
        self.assertEqual(ctrl.waypoints[0].x,3.)
        self.assertTrue(any(e['event']=='edge_exit_complete' for e in ctrl.escape_events))

    def test_new_hazard_during_backout_brakes(self):
        g,ctrl,xy,cliff=edge()
        ctrl.tick(xy,0,0,1,upright=.75,velocity=(0,0,0))
        g.hazard[g.world_to_cell(-.1,.1)]=True
        ctrl.tick(xy,0,0,1.1,upright=.75,velocity=(0,0,0))
        self.assertIn('obstacle',ctrl.reason)

    def test_manual_stop_and_takeover_cancel_escape(self):
        for action in ('stop','manual'):
            g,ctrl,xy,cliff=edge();ctrl.tick(xy,0,0,1,upright=.75)
            ctrl.request(action,1.1,vx=.1)
            ctrl.tick(xy,0,0,2,upright=.75)
            self.assertEqual(ctrl.state,'HOLD' if action=='stop' else 'MANUAL')
            self.assertIsNone(ctrl.edge_plan)

    def test_tilt_deterioration_stops_escape(self):
        g,ctrl,xy,cliff=edge();ctrl.tick(xy,0,0,1,upright=.8)
        ctrl.tick(xy,0,0,1.1,upright=.65)
        self.assertEqual(ctrl.state,'HOLD');self.assertIn('tilt',ctrl.reason)

    def test_ineffective_arcs_cannot_restart_forever(self):
        g,ctrl,xy,cliff=edge();ctrl.tick(xy,0,0,1,upright=.75)
        for t in (5.1,5.2,9.3,9.4,13.5):ctrl.tick(xy,0,0,t,upright=.75)
        self.assertEqual(ctrl.state,'HOLD')
        self.assertEqual(len(ctrl.edge_attempts),3)

    def test_small_observed_drift_retires_failed_arc_and_rechecks(self):
        g,ctrl,xy,cliff=edge();ctrl.tick(xy,0,0,1,upright=.75)
        extra=g.world_to_cell(.3,.3);g.slope[extra]=math.tan(math.radians(36))
        ctrl.tick(xy,0,0,1.1,upright=.75)
        self.assertEqual(ctrl.state,'EDGE_BRAKE')
        self.assertNotEqual(ctrl.edge_plan['curvature'],0.)
        self.assertEqual(len(ctrl.edge_attempts),2)


if __name__=='__main__':unittest.main()
