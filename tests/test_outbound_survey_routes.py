"""Outbound diversity wiring and asynchronous control isolation (no physics claims)."""
import copy
from concurrent.futures import Future
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import test_behavior_navigation as bt


class OutboundSurveyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grid = bt.ground()

    def control(self, enabled=True):
        c = bt.b.BehaviorTreeAscentController(copy.deepcopy(self.grid),
            [SimpleNamespace(x=0., y=0.), SimpleNamespace(x=3., y=0.)], mode='characterize')
        if enabled:
            c.configure_outbound_survey_routes()
        c.grid.terrain_prior = True
        c.wp_idx = 1
        return c

    def test_only_surveyed_outbound_mission_expands(self):
        for enabled, index, purpose, prior in [
                (False, 1, 'mission', True), (True, 0, 'mission', True),
                (True, 1, 'mission', False)]:
            c = self.control(enabled)
            c.wp_idx, c.route_purpose, c.grid.terrain_prior = index, purpose, prior
            c._survey_alternatives = Mock(side_effect=AssertionError('wrong stage'))
            original = [('direct', [(0., 0.), (3., 0.)], True)]
            self.assertEqual(c._augment_routes(original, (0., 0.), 0.), original)

    def test_mission_compute_after_recovery_still_generates_alternatives(self):
        for previous in ('adjust', 'retreat'):
            c = self.control()
            c.route_purpose = previous
            c.attempt_number = 16
            c._survey_alternatives = Mock(return_value=[])
            c._launch_suffix_variants = Mock(return_value=[])
            self.assertTrue(c._compute_plan((0., 0.), 0., 10.))
            c._survey_alternatives.assert_called_once_with((0., 0.), 0.)
            self.assertEqual(c.route_purpose, 'mission')
            self.assertEqual(c.mission_route[-1], (3., 0.))

    def test_regroup_compute_bypasses_mission_augmentation(self):
        c = self.control()
        c.regroup_required = True
        c._augment_routes = Mock(side_effect=AssertionError('regroup is not mission planning'))
        # Flat fixture contains no lower regroup destination; real regroup
        # search must return without ever expanding the mission candidates.
        self.assertFalse(c._compute_plan((0., 0.), 0., 10.))
        c._augment_routes.assert_not_called()

    def test_recovery_maneuver_bypasses_mission_augmentation(self):
        c = self.control()
        c._augment_routes = Mock(side_effect=AssertionError('recovery is not mission planning'))
        for t in (0., .2, .4):
            c.tick((0., 0.), 0., 0., t)
        c._begin_recovery('test mission rejection')
        c._recovery(None)
        c._augment_routes.assert_not_called()
        self.assertEqual(c.route_purpose, 'adjust')
        self.assertEqual(c.recovery_actions, 1)

    def test_distinct_candidates_keep_original_goal_without_duplicate_paths(self):
        c = self.control()
        original = ('direct', [(0., 0.), (3., 0.)], True)
        alternate = ('portal', [(0., 0.), (1., 1.), (3., 0.)], True)
        c._survey_alternatives = Mock(return_value=[original, alternate])
        c._launch_suffix_variants = Mock(return_value=[alternate])
        self.assertEqual(c._augment_routes([original], (0., 0.), 0.), [original, alternate])
        self.assertEqual(c.route_searches[-1]['added_routes'], 1)
        self.assertEqual(c.route_searches[-1]['portal_routes'], 2)
        self.assertEqual(c.route_searches[-1]['suffix_variants'], 1)
        self.assertGreaterEqual(c.route_searches[-1]['elapsed_ms'], 0.)
        self.assertFalse(c._exploring_initial_route())
        self.assertIsNone(c.exploration_coverage)

    def test_replans_remain_available_after_first_attempt(self):
        c = self.control()
        c.attempt_number = 4
        c._survey_alternatives = Mock(return_value=[])
        c._launch_suffix_variants = Mock(return_value=[])
        c._augment_routes([], (0., 0.), 0.)
        c._survey_alternatives.assert_called_once()

    def test_configuration_requires_fresh_characterization_and_valid_stage(self):
        for mode, attempt, index in [('qualified', 0, 1), ('characterize', 1, 1),
                                     ('characterize', 0, 2), ('characterize', 0, -1)]:
            c = self.control(False)
            c.mode, c.attempt_number = mode, attempt
            with self.assertRaises(ValueError):
                c.configure_outbound_survey_routes(index)

    def test_pending_plan_does_not_block_commands_or_start_more_jobs(self):
        c = self.control()
        pending = Future()
        c.planner = SimpleNamespace(submit=Mock(return_value=pending))
        try:
            self.assertFalse(c._plan((0., 0.), 0., 0.))
            c.since = 0.
            c.observe_revision = c.grid.revision - 3
            for t in (4., 4.5, 5., 10., 50.):
                c.obs = bt.observation(t)
                c._mission(None)
                self.assertEqual(c.command, bt.m.Motion())
                self.assertTrue(c.plan_pending)
            c.planner.submit.assert_called_once()
            c.request('stop', 51.)
            self.assertEqual(c.tick((0., 0.), 0., .2, 51., velocity=(.2, 0.)), bt.m.Motion())
        finally:
            c.planner = None

    def prepared_result(self):
        c = self.control()
        shadow = copy.copy(c)
        shadow.grid = copy.deepcopy(c.grid)
        shadow.attempt_events = []
        shadow._survey_alternatives = Mock(return_value=[])
        shadow._launch_suffix_variants = Mock(return_value=[])
        self.assertTrue(shadow._compute_plan((0., 0.), 0., 0.))
        c.planner = SimpleNamespace()
        c.plan_job = Future()
        c.plan_job.set_result((c.plan_generation, True, shadow))
        return c

    def test_completed_plan_rejected_after_operator_generation_change(self):
        c = self.prepared_result()
        c.plan_generation += 1
        try:
            self.assertFalse(c._plan((0., 0.), 0., 50.))
            self.assertIn('operator', c.planning_failure)
        finally:
            c.planner = None

    def test_completed_plan_rejected_after_pose_drift(self):
        c = self.prepared_result()
        try:
            self.assertFalse(c._plan((.31, 0.), 0., 50.))
            self.assertEqual(c.attempt_events[-1]['reason'], 'start pose changed')
        finally:
            c.planner = None

    def test_completed_plan_rechecks_live_terrain(self):
        c = self.prepared_result()
        c._adoption_edge_reason = Mock(return_value='newly observed obstacle')
        try:
            self.assertFalse(c._plan((0., 0.), 0., 50.))
            self.assertEqual(c.attempt_events[-1]['reason'], 'newly observed obstacle')
        finally:
            c.planner = None
