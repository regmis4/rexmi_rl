"""Physical response, planning grades and directional learning are distinct."""
from dataclasses import asdict, replace
import math
from types import SimpleNamespace
import unittest

import numpy as np

from test_behavior_navigation import b, m, ground, observation


class DriftContractTests(unittest.TestCase):
    def test_sideways_sway_is_not_a_commanded_cross_slope_approach(self):
        g = ground(28.)
        g.side_limit_deg = 25.
        self.assertFalse(g.segment_clear((0., 0.), (0., .2)))
        self.assertEqual(g.drift_segment_reason((0., 0.), (0., .2)), '')

    def test_drift_keeps_physical_and_observation_checks(self):
        for hazard in ('freshness', 'obstacle', 'unknown', 'slope'):
            with self.subTest(hazard=hazard):
                g = ground(28.)
                cell = g.world_to_cell(0., .2)
                if hazard == 'freshness': g.fresh[cell] = False
                if hazard == 'obstacle': g.hazard[cell] = True
                if hazard == 'unknown': g.known[cell] = False
                if hazard == 'slope': g.slope[cell] = math.tan(math.radians(36.))
                self.assertTrue(g.drift_segment_reason((0., 0.), (0., .2)))

    def test_drift_keeps_directional_exclusions(self):
        g = ground(); g.mark_failure('one', (0., .2), math.pi/2, 1., 'fixture')
        self.assertIn('exclusion:', g.drift_segment_reason((0., 0.), (0., .2), no_go_exit=True))

    def close_patch(self, g, xy, name):
        g.mark_failure(name+'a', xy, 0., 1., 'fixture')
        g.mark_failure(name+'b', xy, math.pi, 2., 'fixture')

    def test_occupied_exclusion_exit_never_authorizes_entry_or_reentry(self):
        g = ground(); self.close_patch(g, (0., 0.), 'first')
        self.assertEqual(g.drift_segment_reason((0., 0.), (.8, 0.), no_go_exit=True), '')
        self.assertIn('exclusion:', g.drift_segment_reason((.8, 0.), (0., 0.), no_go_exit=True))
        self.close_patch(g, (1.4, 0.), 'second')
        self.assertIn('exclusion:', g.drift_segment_reason((0., 0.), (1.4, 0.), no_go_exit=True))
        self.assertTrue(g.drift_segment_reason((.2, 0.), (0., 0.), no_go_exit=True))


class ResponseEvidenceTests(unittest.TestCase):
    def executor(self, sign=1):
        e = m.MovementExecutor()
        e.start(m.MovementRequest('one', (0., 0.), (sign*2., 0.), sign, .4, 100., arrival_radius=.3), 0.)
        return e

    def tick(self, e, obs, supported=True):
        return e.tick(obs, lambda command: '', lambda xy: supported)

    def drive(self, e):
        for t in (0., .2, .4, .5, .6):
            self.tick(e, observation(t))

    def test_supported_submillimetre_departure_is_not_terrain_failure(self):
        e = self.executor(); self.drive(e)
        self.tick(e, observation(1., xy=(1.7001, 0.)))
        self.assertEqual(e.state, m.MovementState.SETTLE)
        self.tick(e, observation(1.02, xy=(1.6992, 0.)))
        self.assertEqual(e.result.outcome, 'censored')
        self.assertFalse(e.result.physical)
        self.assertEqual(e.result.disposition, m.ResultDisposition.REPOSITIONED)

    def test_unavailable_support_does_not_invent_physical_failure(self):
        e = self.executor(); self.drive(e)
        self.tick(e, observation(1., xy=(1.71, 0.)))
        self.tick(e, observation(1.02, xy=(1.71, 0.)), supported=False)
        self.assertFalse(e.result.physical)
        self.assertEqual(e.result.outcome, 'censored')
        self.assertIsNotNone(e.result.executed_traversal)

    def test_brake_only_positive_damping_has_no_traversal_evidence(self):
        e = self.executor(-1)
        for t in (0., 1., 8.1):
            self.tick(e, observation(t, velocity=(-.2, 0.)))
        self.assertTrue(e.result.physical)
        self.assertEqual(e.result.phase, 'BRAKE')
        self.assertIsNone(e.result.executed_traversal)

    def test_drive_handoff_is_not_exposure_and_zero_motion_can_be_attempt(self):
        e = self.executor()
        for t in (0., .2, .4): self.tick(e, observation(t))
        self.assertIsNone(e.executed_traversal)
        self.tick(e, observation(.5))
        self.assertIsNone(e.executed_traversal)
        self.tick(e, observation(.6))
        self.assertAlmostEqual(e.executed_traversal.duration_s, .1)
        self.assertEqual(e.executed_traversal.last_position, (0., 0.))

    def test_reverse_direction_is_travel_heading_not_body_or_sway(self):
        e = self.executor(-1); self.drive(e)
        self.assertAlmostEqual(abs(e.executed_traversal.heading), math.pi)
        self.assertEqual(e.executed_traversal.drive_sign, -1)
        self.assertEqual(e.executed_traversal.body_yaw, 0.)

    def test_release_keeps_committed_direction_despite_sway(self):
        e = self.executor(); self.drive(e)
        e.release_guard = lambda obs, req: ''
        e.release_origin = (0., 0.)
        e._state(m.MovementState.RELEASE, .6, 'fixture')
        for t in (.7, .8):
            self.tick(e, replace(observation(t, xy=(.1, .01), yaw=.2, velocity=(0., .03), grade=28., turn=False),
                                terrain_gradient=(.5, 0.)))
        self.assertEqual(e.executed_traversal.maneuver, 'RELEASE')
        self.assertEqual(e.executed_traversal.heading, 0.)

    def test_new_request_and_actual_alignment_clear_evidence(self):
        e = self.executor(); self.drive(e)
        e._state(m.MovementState.ALIGN_LEFT, .7, 'realignment')
        self.assertIsNone(e.executed_traversal)
        self.assertIsNone(e.pending_traversal)
        e.finish('censored', 'fixture', observation(.8)); e.consume()
        e.start(m.MovementRequest('two', (0., 0.), (-2., 0.), -1, .4, 100.), 1.)
        self.assertIsNone(e.executed_traversal)

    def control(self):
        c = b.BehaviorTreeAscentController(ground(), [SimpleNamespace(x=3., y=0., label='goal')], mode='characterize')
        c.tick((0., 0.), 0., 0., 0.)
        self.assertTrue(c._compute_plan((0., 0.), 0., 0.))
        return c

    def test_brake_failure_updates_attempt_but_not_directional_exclusions(self):
        c = self.control()
        c.finish_attempt('failure', 'brake-only', (0., 0.), 0., 9., physical=True,
                         execution=None, terminal_maneuver='BRAKE')
        event = c.attempt_events[-1]
        self.assertEqual(event['outcome'], 'failure')
        self.assertTrue(event['physical'])
        self.assertFalse(event['exclusion_applied'])
        self.assertFalse(c.grid.approach_blocked.any())

    def test_post_drive_failure_localizes_once_using_executed_reverse(self):
        c = self.control(); e = self.executor(-1); self.drive(e)
        e._state(m.MovementState.BRAKE, .7, 'settle')
        e.finish('failure', 'braking failed after traversal', observation(8.), physical=True)
        c.finish_attempt('failure', e.result.reason, e.result.position, 0., e.result.time, physical=True,
                         execution=asdict(e.result.executed_traversal), terminal_maneuver=e.result.phase)
        record = c.grid.failure_records[-1]
        self.assertAlmostEqual(abs(record['heading']), math.pi)
        self.assertEqual(record['time'], 8.)
        self.assertTrue(c.attempt_events[-1]['exclusion_applied'])
        self.assertFalse(c.grid.no_go.any())
        c.finish_attempt('failure', 'duplicate', (0., 0.), 0., 8.1, physical=True,
                         terminal_maneuver='BRAKE')
        self.assertEqual(len(c.grid.failure_records), 1)

    def test_distant_traversal_does_not_mark_failure_patch(self):
        c = self.control(); e = self.executor(); self.drive(e)
        c.finish_attempt('failure', 'distant settling failure', (1., 0.), 0., 10., physical=True,
                         execution=asdict(e.executed_traversal), terminal_maneuver='SETTLE')
        self.assertFalse(c.grid.failure_records)


if __name__ == '__main__':
    unittest.main()
