"""Release is powered travel; verify its complete movement contract."""
from dataclasses import replace
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch
import copy
from concurrent.futures import Future

import numpy as np

from test_behavior_navigation import a, b, m, ground, observation
from rexmi_bt_tests.nav.staged_routes import observed_edge_reason, staged_route
from rexmi_bt_tests.nav.ascent_routes import OFFSETS


class UphillReleaseTests(unittest.TestCase):
    def obs(self, t, x=None, **kwargs):
        return replace(observation(t, xy=(.1*t if x is None else x, 0.),
            grade=30., turn=False, **kwargs), terrain_gradient=(math.tan(math.radians(30.)), 0.))

    def setup_drive(self, sign=1, purpose='mission'):
        e = m.MovementExecutor()
        e.release_guard = lambda obs, req: ''
        e.start(m.MovementRequest('release', (0., 0.), (sign*10., 0.), sign, .4, 100., purpose), 0.)
        for t in (0., .2, .4): self.tick(e, self.obs(t, 0.))
        return e

    def tick(self, e, obs):
        return e.tick(obs, lambda cmd: '', lambda xy: True)

    def start_release(self):
        e = self.setup_drive()
        for t in np.arange(.5, 5.6, .1): self.tick(e, self.obs(t))
        # These tests exercise the RELEASE response contract. Its measured
        # DRIVE admission is independently covered at50Hz in
        # test_measured_release; this coarse .1s fixture cannot supply it.
        self.assertIsNone(e.result)
        e.release_origin=(.55,0.)
        e._state(m.MovementState.RELEASE,5.5,'release-response fixture')
        self.assertEqual(e.state, m.MovementState.RELEASE)
        return e

    def test_productive_release_is_zero_without_brake_and_keeps_identity(self):
        e = self.start_release()
        req, axis, heading = e.request, e.travel_axis, e.committed_heading
        with patch.object(e, 'brake', side_effect=AssertionError('release is not brake')):
            for t in np.arange(5.6, 25., .1):
                self.assertEqual(self.tick(e, self.obs(t, velocity=(.1, 0.))), m.Motion())
        self.assertIs(e.request, req)
        self.assertEqual(e.request.deadline, 100.)
        self.assertEqual((e.travel_axis, e.committed_heading), (axis, heading))
        self.assertIsNone(e.result)

    def interrupted_drive(self):
        e = self.setup_drive()
        for t in np.arange(.5, 2.6, .1): self.tick(e, self.obs(t))
        self.tick(e, self.obs(2.6, velocity=(.6,0.)))
        self.assertEqual(e.state, m.MovementState.BRAKE)
        return e

    def test_braking_interruptions_do_not_starve_release_exposure(self):
        e = self.interrupted_drive(); request = e.request
        exposure = e.drive_exposure
        for t in (2.7, 2.9, 3.1): self.tick(e, self.obs(t))
        self.assertAlmostEqual(e.drive_exposure, exposure)
        # Re-measure a full powered window after the interruption. The old
        # exposure is retained, while the new low-response dwell is fresh.
        for t in np.arange(3.12, 6.4, .02):
            command = self.tick(e, self.obs(t,.31+.05*(t-3.1)))
        self.assertEqual(e.state, m.MovementState.RELEASE)
        self.assertEqual(command, m.Motion())
        self.assertIs(e.request, request)
        self.assertAlmostEqual(e.drive_exposure, 0.)

    def test_positive_damping_and_brake_to_drive_handoff_are_not_drive_exposure(self):
        e = self.interrupted_drive(); exposure = e.drive_exposure
        for t in (2.7, 2.9, 3.1):
            self.assertEqual(self.tick(e, self.obs(t, velocity=(-.1,0.))), m.Motion(.4,0.))
        self.assertEqual(e.state, m.MovementState.DRIVE_FORWARD)
        self.assertFalse(e.forward_output)
        self.tick(e, self.obs(3.2))
        self.assertAlmostEqual(e.drive_exposure, exposure)
        self.assertTrue(e.forward_output)

    def test_actual_alignment_and_new_request_reset_forward_exposure(self):
        e = self.setup_drive()
        for t in np.arange(.5, 2.6, .1): self.tick(e, self.obs(t))
        self.assertGreater(e.drive_exposure, 1.9)
        for t in (2.6, 2.7, 2.9, 3.1):
            self.tick(e, replace(self.obs(t, yaw=-.3), inclination=0., turn_allowed=True, terrain_gradient=(0.,0.)))
        self.assertEqual(e.state, m.MovementState.ALIGN_LEFT)
        self.assertEqual(e.drive_exposure, 0.)
        e.finish('censored', 'new request fixture', self.obs(3.2)); e.consume()
        e.start(m.MovementRequest('next', (0.,0.), (10.,0.), 1, .4, 100.), 4.)
        self.assertEqual(e.drive_exposure, 0.)
        self.assertFalse(e.forward_output)

    def test_low_progress_resumes_drive_without_alignment_or_new_attempt(self):
        e = self.start_release(); req = e.request
        for t in np.arange(5.6, 13., .1): command = self.tick(e, self.obs(t, .55))
        self.assertEqual(command, m.Motion(.4, 0.))
        self.assertEqual(e.state, m.MovementState.DRIVE_FORWARD)
        self.assertIs(e.request, req)
        self.assertFalse(any('ALIGN' in h.get('state', '') for h in e.history))

    def test_negative_release_progress_resumes_without_waiting_for_stall(self):
        e = self.start_release()
        command = self.tick(e, self.obs(5.7, .44))
        self.assertEqual(command, m.Motion(.4, 0.))
        self.assertIsNone(e.result)

    def test_release_resume_rechecks_heading_and_remaining_corridor(self):
        for reason in ('opposing_heading', 'remaining_geometry'):
            e = self.start_release()
            obs = self.obs(5.7, .44, yaw=math.pi if reason == 'opposing_heading' else 0.)
            if reason == 'remaining_geometry': e.corridor_guard = lambda obs, req, yaw: 'remaining corridor changed'
            command = self.tick(e, obs)
            self.assertEqual(command, m.Motion())
            self.assertEqual(e.result.outcome, 'censored')
            self.assertFalse(e.result.physical)

    def test_release_correction_at_gentle_staging_brakes_before_drive(self):
        e = self.start_release()
        obs = replace(self.obs(5.7, .44, yaw=.3), inclination=5., turn_allowed=True, terrain_gradient=(0.,0.))
        command = self.tick(e, obs)
        self.assertEqual(command, m.Motion())
        self.assertEqual(e.state, m.MovementState.BRAKE)
        self.assertFalse(e.metrics['release_active'])

    def test_arrival_region_and_settle_region_are_consistent(self):
        e = self.setup_drive()
        e.request = replace(e.request, arrival_radius=.3)
        self.tick(e, replace(self.obs(1.), xy=(10.29, .29)))
        self.assertNotEqual(e.state, m.MovementState.SETTLE)
        self.assertIsNone(e.result)

    def test_arrival_during_alignment_or_braking_requires_settle_without_more_yaw(self):
        for phase in (m.MovementState.BRAKE, m.MovementState.ALIGN_LEFT, m.MovementState.ALIGN_RIGHT):
            e = self.setup_drive()
            e.request = replace(e.request, arrival_radius=.3)
            e._state(phase, 1., 'measured policy translation fixture')
            command = self.tick(e, self.obs(1.1, 9.8, yaw=.7))
            self.assertEqual(command, m.Motion())
            self.assertEqual(e.state, m.MovementState.SETTLE)
            self.assertIsNone(e.result)
            for t in (1.2, 1.4, 1.6, 1.8): self.tick(e, self.obs(t, 9.8, yaw=.7))
            self.assertEqual(e.consume().outcome, 'success')
            self.assertIsNone(e.consume())

    def test_alignment_overshoot_is_censored_without_physical_failure(self):
        e = self.setup_drive(); e._state(m.MovementState.ALIGN_LEFT, 1., 'fixture')
        self.tick(e, self.obs(1.1, 10.5, yaw=.7))
        self.assertEqual(e.result.outcome, 'censored')
        self.assertEqual(e.result.disposition, m.ResultDisposition.REPOSITIONED)
        self.assertFalse(e.result.physical)

    def test_alignment_translation_has_lateral_bounds_without_a_terrain_failure_vote(self):
        e = self.setup_drive(); e._state(m.MovementState.ALIGN_LEFT, 1., 'fixture')
        self.tick(e, replace(self.obs(1.1), xy=(.2,.6)))
        self.assertEqual(e.result.outcome, 'censored')
        self.assertEqual(e.result.disposition, m.ResultDisposition.REPOSITIONED)
        self.assertFalse(e.result.physical)

    def test_alignment_arrival_still_requires_fresh_supported_ground_and_controlled_speed(self):
        for constraint in ('support', 'freshness', 'overspeed'):
            e = self.setup_drive(); e._state(m.MovementState.ALIGN_LEFT, 1., 'fixture')
            obs = self.obs(1.1, 9.95, yaw=.7)
            if constraint == 'freshness': obs = replace(obs, fresh=False)
            if constraint == 'overspeed': obs = replace(obs, velocity=(.6,0.))
            e.tick(obs, lambda cmd: '', lambda xy: constraint != 'support')
            self.assertNotEqual(e.state, m.MovementState.SETTLE)
            self.assertFalse(e.result and e.result.outcome == 'success')

    def test_release_cannot_reset_stall_budget(self):
        e = self.setup_drive()
        for t in np.arange(.5, 20., .02): self.tick(e, self.obs(t, 0.))
        self.assertTrue(any(h.get('state') == 'RELEASE' for h in e.history))
        self.assertIsNotNone(e.result)
        self.assertTrue(e.result.physical)
        self.assertLess(e.result.time, 15.)
        self.assertIsNotNone(e.consume())
        self.assertIsNone(e.consume())

    def test_gradient_loss_does_not_retroactively_shorten_admitted_patience(self):
        for gradient in (None, (.169, 0.)):
            e = self.setup_drive()
            for t in np.arange(.5, 5.6, .02): self.tick(e, self.obs(t, 0.))
            self.assertEqual(e.state, m.MovementState.RELEASE)
            command = self.tick(e, replace(self.obs(5.6, 0.), terrain_gradient=gradient))
            self.assertEqual(command, m.Motion(.4, 0.))
            self.assertIsNone(e.result)
            for t in np.arange(5.7, 16., .1): self.tick(e, replace(self.obs(t, 0.), terrain_gradient=gradient))
            self.assertTrue(e.result.physical)
            self.assertGreater(e.result.time, 14.)

    def test_threshold_jitter_does_not_renew_low_progress_patience(self):
        e = self.setup_drive()
        for t in np.arange(.5, 20., .1):
            self.tick(e, self.obs(t, .035*math.sin(t)))
        self.assertIsNotNone(e.result)
        self.assertTrue(e.result.physical)
        self.assertLess(e.result.time, 18.)

    def test_geometry_is_checked_even_at_slow_zero_command_creep(self):
        e = self.start_release()
        e.release_guard = lambda obs, req: 'prospective footprint blocked'
        self.tick(e, self.obs(5.6, velocity=(.03, 0.)))
        self.assertEqual(e.result.outcome, 'censored')
        self.assertFalse(e.result.physical)

    def test_release_arrival_settles_and_overshoot_is_not_success(self):
        for x, expected in ((9.95, m.MovementState.SETTLE), (10.5, m.MovementState.IDLE)):
            e = self.start_release()
            self.tick(e, self.obs(5.6, x))
            self.assertEqual(e.state, expected)
            if e.result is not None: self.assertEqual(e.result.outcome, 'censored')

    def test_stop_stale_overspeed_posture_and_deadline_still_interrupt(self):
        for kind in ('stop', 'stale', 'overspeed', 'posture', 'deadline'):
            with self.subTest(kind=kind):
                e = self.start_release(); obs = self.obs(5.6)
                if kind == 'stop':
                    self.assertEqual(e.stop(obs), m.Motion())
                    self.assertEqual(e.state, m.MovementState.STOP)
                    self.assertFalse(e.result.physical)
                elif kind == 'stale':
                    self.tick(e, replace(obs, fresh=False))
                    self.assertFalse(e.result.physical)
                elif kind == 'overspeed':
                    self.assertEqual(self.tick(e, replace(obs, velocity=(.6, 0.))), m.Motion(-.4, 0.))
                    self.assertEqual(e.state, m.MovementState.BRAKE)
                elif kind == 'posture':
                    self.tick(e, replace(obs, up_z=math.cos(math.radians(65.))))
                    self.assertTrue(e.result.physical)
                else:
                    self.tick(e, replace(obs, time=101.))
                    self.assertTrue(e.result.physical)

    def test_reverse_recovery_flat_downhill_and_unknown_gradient_do_not_release(self):
        for case in ('reverse', 'recovery', 'flat', 'downhill', 'unknown'):
            e = self.setup_drive(sign=-1 if case == 'reverse' else 1,
                                 purpose='recovery' if case == 'recovery' else 'mission')
            for t in np.arange(.5, 10., .1):
                obs = self.obs(t, -.1*t if case == 'reverse' else .1*t)
                if case == 'flat': obs = replace(obs, terrain_gradient=(0., 0.))
                if case == 'downhill': obs = replace(obs, terrain_gradient=(-.5, 0.))
                if case == 'unknown': obs = replace(obs, terrain_gradient=None)
                self.tick(e, obs)
            self.assertFalse(any(h.get('state') == 'RELEASE' for h in e.history), case)

    def test_checked_brake_resume_restores_release_with_exact_zero(self):
        e = self.start_release(); request = e.request; axis = e.travel_axis
        self.tick(e, self.obs(5.6, velocity=(.6, 0.)))
        self.assertTrue(e.suspended_release)
        for t in (5.7, 5.9, 6.1): command = self.tick(e, self.obs(t))
        self.assertEqual(e.state, m.MovementState.RELEASE)
        self.assertEqual(command, m.Motion())
        self.assertFalse(e.suspended_release)
        self.assertIs(e.request, request)
        self.assertEqual(e.travel_axis, axis)
        self.assertEqual(request.deadline, 100.)
        self.assertEqual(e.samples, [])
        self.assertGreater(len(e.uphill_samples), 2)

    def test_suspended_restore_checks_gradient_heading_corridor_and_release(self):
        for case in ('gradient', 'heading', 'corridor', 'release'):
            with self.subTest(case=case):
                e = self.start_release()
                self.tick(e, self.obs(5.6, velocity=(.6, 0.)))
                if case == 'corridor': e.corridor_guard = lambda *args: 'corridor changed'
                if case == 'release': e.release_guard = lambda *args: 'release changed'
                for t in (5.7, 5.9, 6.1):
                    obs = self.obs(t, yaw=math.pi if case == 'heading' else 0.)
                    if case == 'gradient': obs = replace(obs, terrain_gradient=None)
                    self.tick(e, obs)
                self.assertEqual(e.result.outcome, 'censored')
                self.assertFalse(e.result.physical)
                self.assertFalse(e.suspended_release)

    def test_repeated_interruptions_cannot_renew_unstarted_progress_budget(self):
        e = self.start_release()
        for i, t in enumerate(np.arange(5.6, 35., .1)):
            velocity = (.6, 0.) if i % 10 == 0 else (0., 0.)
            self.tick(e, self.obs(t, .55, velocity=velocity))
            if e.result is not None: break
        self.assertIsNotNone(e.result)
        self.assertTrue(e.result.physical)
        self.assertIn('shared low-progress', e.result.reason)
        self.assertLess(e.result.time, 33.)
        restores = [h for h in e.history if h.get('reason') == 'restore checked release after measured braking']
        self.assertGreater(len(restores), 5)

    def test_low_progress_can_expire_during_braking(self):
        e = self.start_release()
        for t in np.arange(5.6, 17., .1): self.tick(e, self.obs(t, .55))
        self.assertIsNotNone(e.uphill_stall_since)
        # Observe the final portion of an otherwise unchanged wall-time epoch.
        e.uphill_stall_since=17.-23.8
        for t in np.arange(17., 22., .1):
            self.tick(e, self.obs(t, .55, velocity=(.6, 0.)))
            if e.result is not None: break
        self.assertEqual(e.result.phase, 'BRAKE')
        self.assertIn('shared low-progress', e.result.reason)

    def test_actual_braking_advance_can_renew_progress_budget(self):
        e = self.start_release()
        for t in np.arange(5.6, 10., .1): self.tick(e, self.obs(t, .55))
        self.assertIsNotNone(e.uphill_stall_since)
        self.tick(e, self.obs(10., .55, velocity=(.6, 0.)))
        self.tick(e, self.obs(10.1, .70))
        self.assertIsNone(e.uphill_stall_since)
        self.assertIsNone(e.result)

    def test_suspended_release_stop_and_arrival_win(self):
        for case in ('stop', 'arrival', 'stale'):
            e = self.start_release()
            self.tick(e, self.obs(5.6, velocity=(.6, 0.)))
            if case == 'stop':
                self.assertEqual(e.stop(self.obs(5.7)), m.Motion())
                self.assertEqual(e.state, m.MovementState.STOP)
            elif case == 'arrival':
                self.tick(e, self.obs(5.7, 9.95))
                self.assertEqual(e.state, m.MovementState.SETTLE)
            else:
                self.tick(e, replace(self.obs(5.7), fresh=False))
                self.assertEqual(e.result.outcome, 'censored')
            self.assertFalse(e.suspended_release)

    def test_suspended_release_retains_backslide_history(self):
        e = self.start_release()
        self.tick(e, self.obs(5.6, .55, velocity=(.6, 0.)))
        for t in np.arange(5.7, 7.8, .1):
            self.tick(e, self.obs(t, .55-.2*(t-5.6), velocity=(-.2, 0.)))
            if e.result is not None: break
        self.assertTrue(e.result.physical)
        self.assertIn('backslide', e.result.reason)

    def test_progress_basis_is_translation_invariant_after_gentle_resume(self):
        for offset in (0., 10., -100.):
            e = m.MovementExecutor(); e.release_guard = lambda *args: ''
            e.start(m.MovementRequest('axis', (0., offset), (10., offset), 1, .4, 100.), 0.)
            def obs(t, x=0., yaw=0., velocity=(0.,0.)):
                return replace(observation(t, xy=(x, offset), yaw=yaw, velocity=velocity, grade=12.),
                               terrain_gradient=(math.tan(math.radians(12.)), 0.))
            for t in np.arange(0., 3.1, .1): self.tick(e, obs(t))
            stall_since, stall_origin = e.uphill_stall_since, e.uphill_stall_origin
            self.assertIsNotNone(stall_since)
            self.tick(e, obs(3.1, .01, velocity=(.6, 0.)))
            for t in (3.2, 3.4, 3.6, 3.7): self.tick(e, obs(t, .02, yaw=-.09))
            self.assertIsNone(e.result)
            self.assertEqual(e.state, m.MovementState.DRIVE_FORWARD)
            self.assertAlmostEqual(e.travel_axis[1], math.sin(-.09))
            self.assertEqual(e.uphill_stall_since, stall_since)
            self.assertEqual(e.uphill_stall_origin, stall_origin)

    def test_supported_arrival_gets_settling_before_low_progress_expiry(self):
        e = m.MovementExecutor(); e.release_guard = lambda *args: ''
        e.start(m.MovementRequest('near', (9.8, 0.), (10., 0.), 1, .4, 100.), 0.)
        for t in np.arange(0., 14., .1): self.tick(e, self.obs(t, 9.8))
        self.assertIsNotNone(e.uphill_stall_since)
        self.tick(e, self.obs(e.uphill_stall_since+12., 9.89))
        self.assertIsNone(e.result)
        self.assertEqual(e.state, m.MovementState.SETTLE)


class ApproachContractTests(unittest.TestCase):
    def control(self, grid):
        from types import SimpleNamespace
        return b.BehaviorTreeAscentController(grid, [SimpleNamespace(x=3., y=0., label='goal')], mode='characterize')

    def test_diagonal_is_a_soft_preference_not_a_slope_exemption(self):
        g = ground(30.); self.control(g)
        centre = g.world_to_cell(0., 0.)
        h = math.tan(math.radians(30.))
        self.assertGreater(g.diagonal_preference_cost(h, 0.), g.diagonal_preference_cost(h*math.cos(.35), h*math.sin(.35)))
        self.assertEqual(g.diagonal_preference_cost(-h, 0.), 0.)
        g.uphill_limit_deg = 20.
        costs, _ = g.directional_costs(offsets=((1, 0), (2, 1)))
        self.assertTrue(np.isinf(costs[centre]).all())

    def test_adaptive_width_requires_support_throughout_band(self):
        g = ground(); c = self.control(g)
        self.assertEqual(c._corridor_tolerance((0., 0.), (2., 0.)), 1.)
        original = g.segment_clear
        with patch.object(g, 'segment_clear', side_effect=lambda a, b, **kw: False if .15 < abs(a[1]) < .25 else original(a, b, **kw)):
            self.assertEqual(c._corridor_tolerance((0., 0.), (2., 0.)), .5)

    def test_slow_release_drift_uses_separate_physical_envelope(self):
        c = self.control(ground()); c.tick((0., 0.), 0., .03, 0., velocity=(0., .03))
        req = m.MovementRequest('r', (0., 0.), (2., 0.), 1, .4, 100.)
        with patch.object(c, '_guard', return_value=''), patch.object(c.grid, 'drift_segment_reason', return_value='exclusion: fixture') as check:
            self.assertIn('actual drift', c._release_envelope(c.obs, req))
            self.assertTrue(check.call_args.kwargs['no_go_exit'])
            event = c.escape_events[-1]
            self.assertEqual(event['category'], 'exclusion')
            self.assertEqual(event['velocity_world'], [0., .03])

    def test_release_backslide_is_included_in_controller_metrics(self):
        c = self.control(ground(30.)); c.tick((0., 0.), 0., 0., 0.)
        c.executor.state = m.MovementState.RELEASE
        c.executor.travel_axis = (1., 0.)
        c.previous_command = m.Motion()
        with patch.object(c.tree, 'tick'):
            c.tick((-.05, 0.), 0., .05, 1., velocity=(-.05, 0.))
        self.assertAlmostEqual(c.backslide_distance, .05)

    def test_complete_leg_prediction_precedes_alignment_and_is_not_repeated(self):
        c = self.control(ground()); c.tick((0.,0.), 1., 0., 0.)
        self.assertTrue(c._compute_plan((0.,0.), 1., 0.))
        c.attempt_events[-1]['event'] = 'approach_selection'
        c._start(c.path[1], 1, 'mission')
        self.assertTrue(c.prediction_committed)
        predictions = [e for e in c.attempt_events if e['event'] == 'approach_start']
        self.assertEqual(len(predictions), 1)
        self.assertEqual(c.executor.state, m.MovementState.BRAKE)
        attempt_id = c.attempt['id']
        c.executor.finish('success', 'first primitive reached', c.obs)
        c.executor.consume()
        c._start((2.,0.), 1, 'mission')
        self.assertEqual(c.attempt['id'], attempt_id)
        self.assertEqual(sum(e['event'] == 'approach_start' for e in c.attempt_events), 1)
        self.assertFalse(any(e['event'] == 'approach_outcome' for e in c.attempt_events))

    def near_field_grid(self):
        g = ground(); self.control(g)
        rr, cc = np.indices(g.fresh.shape)
        x = (rr+.5)*g.cell_size-g.world_size/2
        y = (cc+.5)*g.cell_size-g.world_size/2
        g.fresh[:] = np.hypot(x, y) <= 2.35
        return g

    def test_compacting_route_does_not_expand_freshness_requirement(self):
        g = self.near_field_grid()
        route, done = staged_route(g, (0.,0.), (4.,0.), 0., offsets=OFFSETS, allow_frontier=False)
        self.assertTrue(done)
        self.assertEqual(route, [(0.,0.), (4.,0.)])
        self.assertEqual(observed_edge_reason(g, *route, (0.,0.)), '')
        for x in np.arange(0., 4., .2):
            self.assertEqual(observed_edge_reason(g, (x,0.), (x+.2,0.), (0.,0.)), '')
        self.assertFalse(g.segment_clear(*route, require_fresh=True))

    def test_stale_near_field_and_distant_hazards_still_reject(self):
        for case in ('near_stale', 'far_hazard'):
            g = self.near_field_grid()
            if case == 'near_stale': g.fresh[g.world_to_cell(1.,0.)] = False
            else: g.blocked[g.world_to_cell(3.,0.)] = True
            self.assertTrue(observed_edge_reason(g, (0.,0.), (4.,0.), (0.,0.)))

    def test_segment_crossing_near_field_checks_interior_not_only_endpoints(self):
        g = self.near_field_grid()
        g.fresh[g.world_to_cell(0.,0.)] = False
        self.assertTrue(observed_edge_reason(g, (-4.,0.), (4.,0.), (0.,0.)))

    def test_async_handoff_adopts_compacted_route_into_retained_survey(self):
        from types import SimpleNamespace
        ctrl = self.control(self.near_field_grid())
        shadow = copy.copy(ctrl); shadow.grid = copy.deepcopy(ctrl.grid)
        shadow.attempt_events = []; shadow.planner = None
        self.assertTrue(shadow._compute_plan((0.,0.), 0., 0.))
        ctrl.planner = SimpleNamespace()
        ctrl.plan_job = Future(); ctrl.plan_job.set_result((ctrl.plan_generation, True, shadow))
        self.assertTrue(ctrl._plan((0.,0.), 0., 1.))
        self.assertEqual(ctrl.state, 'FOLLOW')
        self.assertEqual(ctrl.checkpoint, (3.,0.))
        self.assertEqual(ctrl.grid.failure_records, [])
        ctrl.planner = None


class RecordedReleaseTests(unittest.TestCase):
    """Replay measured response, not a claim about counterfactual dynamics.

    Prospective terrain is supplied as clear in this fixture. Map guards have
    separate tests; these traces cannot establish a safe autonomous corridor.
    """
    def replay(self, name):
        data = json.loads((Path(__file__).parent/'fixtures/teleop_release_response.json').read_text())['cases'][name]
        e = m.MovementExecutor()
        e.release_guard = lambda obs, req: ''
        e.posture_guard = lambda obs, req: ''
        yaw = data['entry_yaw']
        axis = math.cos(yaw), math.sin(yaw)
        e.start(m.MovementRequest(name, (0., 0.), tuple(5.*v for v in axis), 1, .4, 100.,
                                 lateral_tolerance=1.), -.4)
        gradient = tuple(data['gradient'])
        for t in (-.4, -.2, 0.):
            obs = m.MotionObservation((0., 0.), yaw, (0., 0.), data['rows'][0][-1], 0., t, 30., False, True, gradient)
            command=e.tick(obs, lambda cmd: '', lambda xy: True)
            # The recorded bad entry already exceeds the new absolute pause
            # boundary. Never force RELEASE over that intervention (or revive
            # a completed request) just to manufacture fixture admission.
            if e.result is not None or e.state == m.MovementState.POSTURE_PAUSE:
                return e,max(0.,t),command
        e.release_origin = (0., 0.)
        e._state(m.MovementState.RELEASE, 0., 'recorded release entry')
        previous_yaw, previous_time = yaw, 0.
        for t, x, y, yaw, vx, vy, up in data['rows']:
            rate = math.atan2(math.sin(yaw-previous_yaw), math.cos(yaw-previous_yaw))/(t-previous_time)
            obs = m.MotionObservation((x, y), yaw, (vx, vy), up, rate, t, 30., False, True, gradient)
            command = e.tick(obs, lambda cmd: '', lambda xy: True)
            if e.state != m.MovementState.RELEASE: return e, t, command
            self.assertEqual(command, m.Motion())
            previous_yaw, previous_time = yaw, t
        return e, t, command

    def test_first_measured_useful_release_remains_active(self):
        e, duration, _ = self.replay('first_useful')
        self.assertEqual(e.state, m.MovementState.RELEASE)
        self.assertGreater(duration, 4.)

    def test_long_measured_release_does_not_correct_before_sustained_plateau(self):
        e, duration, command = self.replay('repeat_useful')
        self.assertGreater(duration, 16.)
        self.assertLess(duration, 16.64)
        self.assertEqual(e.state, m.MovementState.RELEASE)
        self.assertEqual(command, m.Motion())
        self.assertIsNone(e.result)

    def test_measured_bad_release_is_not_allowed_to_idle_indefinitely(self):
        e, duration, _ = self.replay('repeat_bad')
        self.assertLess(duration, 3.)
        self.assertNotEqual(e.state, m.MovementState.RELEASE)
        self.assertFalse(e.result and e.result.outcome == 'success')


if __name__ == '__main__': unittest.main()
