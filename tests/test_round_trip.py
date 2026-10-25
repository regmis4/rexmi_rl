"""Two-goal mission wiring; does not establish physical traversal success."""
import copy
import importlib
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import test_behavior_navigation as bt

mission = importlib.import_module('rexmi_bt_tests.nav.mission')
rt = importlib.import_module('rexmi_bt_tests.nav.round_trip')
spec = importlib.util.spec_from_file_location('round_trip_runner',
    Path(__file__).resolve().parents[1]/'source/rexmi_rl/nav/checkpoint_runner.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def args():
    return SimpleNamespace(nav_controller='ascent', ascent_architecture='behavior_tree',
        ascent_mode='characterize', terrain_prior='survey.npz', simulator_survey=False,
        reviewed_buttons=None, checkpoint_replay=None, manual_checkpoints=False,
        capture_ascent_departure=False, flat_sidesteps=False, ascent_exploration_history=None,
        ascent_launch_variant=False, crater_x=0., crater_y=0., r_rim=11., spawn_x=13.,
        spawn_y=0., spawn_z=4.5, spawn_yaw=math.pi, goal_x=0., goal_y=0.,
        goal_radius=1.5, max_steps=100, map_retention='radius')


class PresetTests(unittest.TestCase):
    def test_only_center_and_northern_exterior_goals(self):
        a = args(); rt.configure(a, [])
        waypoints = mission.MissionPlanner(crater_centre=(0,0), r_rim=11).get_waypoints(
            mission.Mission.ROUND_TRIP, a.goal_x, a.goal_y, a.goal_radius)
        self.assertEqual([(w.x,w.y,w.arrival_radius) for w in waypoints], [(0,0,1),(0,13,.6)])
        self.assertEqual((a.spawn_x,a.spawn_y,a.spawn_z,a.spawn_yaw),(13,0,4.5,math.pi))
        self.assertEqual(a.map_retention, 'full')
        self.assertEqual(rt.stage_status(waypoints,1)['stage'], 'ASCEND_TO_EXTERIOR')
        self.assertEqual(rt.stage_status(waypoints,2)['stage'], 'COMPLETE')

    def test_explicit_short_budget_and_exterior_override(self):
        a=args(); a.goal_x=13; a.goal_y=0
        rt.configure(a,['--max_steps','100','--goal_x=13','--goal_y','0'])
        self.assertEqual(a.max_steps,100)
        self.assertEqual((a.goal_x,a.goal_y),(13,0))

    def test_invalid_start_and_exterior_disk_rejected(self):
        for updates, flags in [({'spawn_x':0},['--spawn_x=0']),
                               ({'goal_x':11.3,'goal_y':0},['--goal_x=11.3','--goal_y=0']),
                               ({'goal_x':math.nan},['--goal_x=nan'])]:
            a=args()
            for key,value in updates.items():setattr(a,key,value)
            with self.assertRaises(ValueError):rt.configure(a,flags)

    def test_competing_command_owners_rejected(self):
        for key,value in [('reviewed_buttons','tickets'),('checkpoint_replay','events'),
                          ('manual_checkpoints',True),('ascent_mode','qualified'),
                          ('ascent_architecture','legacy')]:
            a=args();setattr(a,key,value)
            with self.assertRaises(ValueError):rt.configure(a,[])

    def test_one_deadline_uses_actual_timestep_without_overrun(self):
        self.assertFalse(rt.deadline_reached(899.98,.02))
        self.assertTrue(rt.deadline_reached(900,.02))
        self.assertTrue(rt.deadline_reached(899.99,.07))
        self.assertFalse(rt.deadline_reached(500,.02))


class DestinationTransitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.grid=bt.ground()

    def control(self):
        waypoints=[mission.Waypoint(0,0,1,'center'),mission.Waypoint(3,0,.6,'exterior')]
        ctrl=bt.b.BehaviorTreeAscentController(copy.deepcopy(self.grid),waypoints,mode='characterize')
        ctrl.obs=bt.observation(0)
        ctrl.path=[(0,0),(0,0)];ctrl.checkpoint=(0,0)
        ctrl.segments=[(0,0)];ctrl.segment_index=0;ctrl.is_mission_checkpoint=True
        ctrl._start((0,0),1,'mission')
        return ctrl

    def test_center_requires_settling_and_preserves_map_evidence(self):
        ctrl=self.control();grid=ctrl.grid;evidence=ctrl.evidence
        grid.no_go[0,0]=True
        for t in (0,.2,.4):
            ctrl.executor.tick(bt.observation(t),lambda command:'',lambda xy:True)
        self.assertIsNone(ctrl.executor.result)
        self.assertEqual(ctrl.wp_idx,0)
        ctrl.obs=bt.observation(.6)
        ctrl.executor.tick(ctrl.obs,lambda command:'',lambda xy:True)
        ctrl._result(None)
        self.assertEqual(ctrl.wp_idx,1)
        self.assertNotEqual(ctrl.state,'COMPLETE')
        self.assertIs(ctrl.grid,grid);self.assertIs(ctrl.evidence,evidence)
        self.assertTrue(grid.no_go[0,0])

    def test_rejected_leg_cannot_skip_center(self):
        ctrl=self.control();ctrl._begin_recovery=Mock()
        ctrl.executor.finish('censored','planning rejection',ctrl.obs)
        ctrl._result(None)
        self.assertEqual(ctrl.wp_idx,0)
        ctrl._begin_recovery.assert_called_once()

    def test_stop_and_manual_preserved_but_goal_replacement_blocked(self):
        ctrl=self.control();original=ctrl.waypoints
        for action in ('target','checkpoint_mode'):
            event=runner.apply_operator_request(ctrl,None,action,1,0,(0,0),0,0,locked_mission=True)
            self.assertEqual(event['event'],'mission_override_rejected')
        self.assertIs(ctrl.waypoints,original)
        runner.apply_operator_request(ctrl,None,'stop',0,0,(0,0),0,1,locked_mission=True)
        self.assertEqual(ctrl.tick((0,0),0,0,1),bt.m.Motion())
        self.assertEqual(ctrl.wp_idx,0)
        runner.apply_operator_request(ctrl,None,'manual',.4,0,(0,0),0,2,locked_mission=True)
        self.assertTrue(ctrl.intervened)
        self.assertIs(ctrl.waypoints,original)


if __name__=='__main__':unittest.main()
