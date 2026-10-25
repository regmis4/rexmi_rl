"""Motion protocol tests; simulator mission acceptance is a separate gate."""
import copy
from concurrent.futures import Future
from dataclasses import replace
import importlib
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]/'source/rexmi_rl'
for name, directory in [('rexmi_bt_tests', ROOT), ('rexmi_bt_tests.nav', ROOT/'nav')]:
    package = ModuleType(name)
    package.__path__ = [str(directory)]
    sys.modules[name] = package
m = importlib.import_module('rexmi_bt_tests.nav.movement')
b = importlib.import_module('rexmi_bt_tests.nav.ascent_behavior')
a = importlib.import_module('rexmi_bt_tests.nav.ascent_map')
r = importlib.import_module('rexmi_bt_tests.nav.ascent_routes')


def observation(t, xy=(0., 0.), yaw=0., velocity=(0., 0.), grade=0., turn=True, fresh=True, yaw_rate=0.):
    return m.MotionObservation(xy, yaw, velocity, 1., yaw_rate, t, grade, turn, fresh)


class MovementTests(unittest.TestCase):
    def executor(self, end=(2., 0.), sign=1):
        e = m.MovementExecutor()
        e.start(m.MovementRequest('one', (0., 0.), end, sign, .4, 100.), 0.)
        return e

    def tick(self, e, obs):
        return e.tick(obs, lambda cmd: '', lambda xy: True)

    def drive(self, e, grade=0., turn=True):
        for t in (0., .2, .4): self.tick(e, observation(t, grade=grade, turn=turn))

    def test_repeated_ticks_do_not_restart_request_or_deadline(self):
        e = self.executor(); request = e.request
        self.drive(e)
        for t in np.arange(.5, 3., .1): self.tick(e, observation(t, xy=(.03*t, 0.), velocity=(.03, 0.)))
        self.assertIs(e.request, request)
        self.assertEqual(e.started, 0.)
        self.assertEqual(e.request.deadline, 100.)

    def test_reverse_keeps_body_heading_and_never_turns_on_slope(self):
        e = self.executor((-2., 0.), -1); self.drive(e, grade=30., turn=False)
        command = self.tick(e, observation(.5, grade=30., turn=False))
        self.assertEqual(command, m.Motion(-.4, 0.))

    def test_left_is_yaw_only_before_forward(self):
        e = self.executor((0., 2.)); self.drive(e)
        command = self.tick(e, observation(.5))
        self.assertEqual(command.vx, 0.)
        self.assertGreater(command.omega, 0.)

    def test_small_alignment_uses_full_teleop_pulse(self):
        e = self.executor((2.*math.cos(.15), 2.*math.sin(.15))); self.drive(e)
        self.assertEqual(self.tick(e, observation(.5)), m.Motion(0., .35))

    def test_policy_lean_is_separate_from_terrain_grade(self):
        e = self.executor((0., 2.)); self.drive(e)
        obs = replace(observation(.5, grade=3.), up_z=math.cos(math.radians(18.)))
        self.assertEqual(self.tick(e, obs), m.Motion(0., .35))

    def test_body_posture_pause_preserves_action_and_deadline(self):
        e = self.executor((0., 2.)); self.drive(e)
        req = e.request
        obs = replace(observation(.5, turn=False), up_z=math.cos(math.radians(26.)))
        self.assertEqual(self.tick(e, obs).omega, 0.)
        self.assertEqual(e.state, m.MovementState.BRAKE)
        self.assertIs(e.request, req)
        self.assertIsNone(e.result)

    def test_yaw_overshoot_coasts_before_opposite_pulse(self):
        e = self.executor((0., 2.)); self.drive(e)
        self.assertGreater(self.tick(e, observation(.5)).omega, 0.)
        command = self.tick(e, observation(.6, yaw=1.9, yaw_rate=.3))
        self.assertEqual(command.omega, 0.)
        self.assertEqual(e.state, m.MovementState.BRAKE)
        self.assertEqual(self.tick(e, observation(.8, yaw=1.95, yaw_rate=.2)).omega, 0.)
        for t in (1., 1.2, 1.4): self.tick(e, observation(t, yaw=1.95))
        self.assertLess(self.tick(e, observation(1.5, yaw=1.95)).omega, 0.)

    def test_yaw_pulse_duration_is_bounded(self):
        e = self.executor((0., 2.)); self.drive(e)
        self.tick(e, observation(2.5, yaw=.2))
        self.assertEqual(e.state, m.MovementState.BRAKE)

    def test_posture_pause_cannot_extend_absolute_deadline(self):
        e = self.executor((0., 2.)); self.drive(e)
        obs = replace(observation(.5, turn=False), up_z=math.cos(math.radians(26.)))
        self.tick(e, obs)
        self.tick(e, replace(obs, time=101.))
        self.assertEqual(e.result.reason, 'primitive deadline elapsed')

    def test_high_slope_alignment_is_rejected(self):
        e = self.executor((0., 2.)); self.drive(e, grade=30., turn=False)
        self.assertEqual(e.result.outcome, 'censored')
        self.assertIn('staging', e.result.reason)

    def test_lost_staging_cancels_turn(self):
        e = self.executor((0., 2.)); self.drive(e)
        cmd = self.tick(e, observation(.5, grade=25., turn=False))
        self.assertEqual(cmd.omega, 0.)
        self.assertEqual(e.result.outcome, 'censored')

    def test_actual_yaw_must_settle_before_driving(self):
        e = self.executor()
        for t in np.arange(0., 2., .1): self.tick(e, observation(t, yaw_rate=.2))
        self.assertEqual(e.state, m.MovementState.BRAKE)

    def test_stop_cancels_once_and_cannot_resume_on_tick(self):
        e = self.executor(); self.drive(e)
        self.assertEqual(e.stop(observation(1.)), m.Motion())
        self.assertEqual(e.consume().outcome, 'censored')
        self.assertIsNone(e.consume())
        self.assertEqual(self.tick(e, observation(2.)), m.Motion())
        with self.assertRaises(ValueError): e.start(m.MovementRequest('one', (0., 0.), (1., 0.), 1, .4, 20.), 2.)

    def test_steady_slow_response_remains_committed(self):
        e = self.executor(); self.drive(e, grade=30., turn=False)
        for t in np.arange(.5, 12., .1):
            cmd = self.tick(e, observation(t, xy=(.03*t, 0.), velocity=(.03, 0.), grade=30., turn=False))
            self.assertEqual(cmd, m.Motion(.4, 0.))
        self.assertIsNone(e.result)

    def test_stall_produces_one_physical_result(self):
        e = self.executor(); self.drive(e)
        for t in np.arange(.5, 7., .1): self.tick(e, observation(t))
        result = e.consume()
        self.assertTrue(result.physical)
        self.assertEqual(result.outcome, 'failure')
        self.assertIsNone(e.consume())

    def test_overspeed_brakes_without_yaw_or_reversing_request(self):
        e = self.executor(); self.drive(e)
        cmd = self.tick(e, observation(1., velocity=(.6, 0.), grade=30., turn=False))
        self.assertEqual(cmd, m.Motion(-.4, 0.))
        self.assertEqual(e.request.direction, 1)

    def test_steep_body_yaw_variation_retains_checked_progress_both_directions(self):
        for sign in (-1, 1):
            e = self.executor((sign*2., 0.), sign); self.drive(e, grade=30., turn=False)
            for t in np.arange(.5, 8., .1):
                cmd = self.tick(e, observation(t, xy=(sign*.03*t, 0.), yaw=.3,
                    velocity=(sign*.03, 0.), grade=30., turn=False))
                self.assertEqual(cmd, m.Motion(sign*.4, 0.))
            self.assertIsNone(e.result)
            self.assertEqual(sum(v['event']=='movement_heading_variation' for v in e.history), 1)
            self.assertAlmostEqual(e.metrics['progress_mps'], .03)

    def test_steep_body_yaw_cannot_authorize_travel_away_from_corridor(self):
        e = self.executor(); self.drive(e, grade=30., turn=False)
        cmd = self.tick(e, observation(1., yaw=2., grade=30., turn=False))
        self.assertEqual(cmd.omega, 0.)
        self.assertEqual(e.result.outcome, 'censored')
        self.assertFalse(e.result.physical)

    def test_steep_brake_resumes_same_checked_action_and_axis_both_directions(self):
        for sign in (-1, 1):
            e = self.executor((sign*2., 0.), sign); self.drive(e, grade=30., turn=False)
            req, axis, heading = e.request, e.travel_axis, e.committed_heading
            self.tick(e, observation(1., yaw=.3, grade=30., turn=False, velocity=(sign*.5, 0.)))
            self.assertEqual(e.state, m.MovementState.BRAKE)
            checked = []
            e.corridor_guard = lambda obs, request, yaw: checked.append(yaw) or ''
            for t in (1.2, 1.4, 1.6):
                self.tick(e, observation(t, yaw=.3, grade=30., turn=False, velocity=(sign*.03, 0.)))
            cmd = self.tick(e, observation(1.7, yaw=.3, grade=30., turn=False, velocity=(sign*.03, 0.)))
            self.assertEqual(cmd, m.Motion(sign*.4, 0.))
            self.assertEqual(checked, [.3])
            self.assertIs(e.request, req)
            self.assertEqual(e.travel_axis, axis)
            self.assertEqual(e.committed_heading, heading)

    def test_steep_resume_rechecks_full_remaining_corridor(self):
        e = self.executor(); self.drive(e, grade=30., turn=False)
        self.tick(e, observation(1., yaw=.3, grade=30., turn=False, velocity=(.5, 0.)))
        e.corridor_guard = lambda obs, req, yaw: 'remaining corridor changed'
        for t in (1.2, 1.4, 1.6): self.tick(e, observation(t, yaw=.3, grade=30., turn=False))
        self.assertEqual(e.result.reason, 'remaining corridor changed')
        self.assertFalse(e.result.physical)

    def test_steep_yaw_variation_still_checks_actual_stopping_corridor(self):
        e = self.executor(); self.drive(e, grade=30., turn=False)
        cmd = e.tick(observation(1., yaw=.3, grade=30., turn=False),
                     lambda v: 'actual stopping corridor blocked', lambda xy: True)
        self.assertEqual(cmd.omega, 0.)
        self.assertEqual(e.result.outcome, 'censored')
        self.assertFalse(e.result.physical)

    def test_steep_yaw_variation_does_not_hide_actual_lateral_departure(self):
        e = self.executor(); self.drive(e, grade=30., turn=False)
        self.tick(e, observation(1., xy=(.1, .6), yaw=.3, grade=30., turn=False))
        self.assertIn('lateral departure', e.result.reason)
        self.assertTrue(e.result.physical)

    def test_settling_drift_cannot_manufacture_success(self):
        e = self.executor((.4, 0.)); self.drive(e)
        self.tick(e, observation(1., xy=(.35, 0.)))
        self.assertEqual(e.state, m.MovementState.SETTLE)
        self.tick(e, observation(2., xy=(.1, 0.)))
        self.assertEqual(e.result.outcome, 'censored')
        self.assertFalse(e.result.physical)
        self.assertEqual(e.result.disposition, m.ResultDisposition.REPOSITIONED)

    def test_braking_creep_has_a_deadline(self):
        e = self.executor()
        for t in np.arange(0., 9., .1): self.tick(e, observation(t, velocity=(.2, 0.)))
        self.assertIn('brake', e.result.reason)

    def test_guard_checks_the_emitted_straight_command(self):
        e = self.executor(); self.drive(e)
        checked = []
        cmd = e.tick(observation(1.), lambda v: checked.append(v) or '', lambda xy: True)
        self.assertEqual(checked[-1], cmd)
        self.assertEqual(cmd.omega, 0.)

    def test_gentle_approach_converges_with_uncommanded_heading_drift(self):
        e = self.executor((3., 1.))
        xy = np.zeros(2); yaw = 0.; speed = 0.; rate = 0.; dt = .05
        for step in range(1800):
            velocity = (speed*math.cos(yaw), speed*math.sin(yaw))
            obs = observation(step*dt, tuple(xy), yaw, velocity, yaw_rate=rate)
            cmd = self.tick(e, obs)
            if e.result is not None: break
            speed += (cmd.vx-speed)*.2
            rate += (cmd.omega+.025*(abs(cmd.vx)>.1)-rate)*.2
            yaw += rate*dt
            xy += speed*dt*np.array((math.cos(yaw), math.sin(yaw)))
        self.assertIsNotNone(e.result)
        self.assertEqual(e.result.outcome, 'success', e.result)
        self.assertLess(math.dist(xy, (3., 1.)), .25)

    def yaw_response_fixture(self, rebound, lag):
        target = .25
        e = self.executor((2.*math.cos(target), 2.*math.sin(target)))
        permanent = elastic = yaw = rate = held = 0.
        previous_sign = 0
        dt = .02
        for step in range(1500):
            command = self.tick(e, observation(step*dt, yaw=yaw, yaw_rate=rate))
            if e.result is not None or e.state == m.MovementState.DRIVE_FORWARD: break
            sign = 1 if command.omega > 0 else -1 if command.omega < 0 else 0
            held = held+dt if sign and sign == previous_sign else 0.
            previous_sign = sign
            # Temporary policy yaw deflection decays after release. This is a
            # response fixture, not a fitted or validated robot dynamics model.
            if held > .15: permanent += command.omega*.7*dt
            elastic += (sign*rebound-elastic)*dt/lag
            new_yaw = permanent+elastic
            rate = (new_yaw-yaw)/dt
            yaw = new_yaw
        return e, target-yaw

    def test_alignment_handles_observed_range_of_policy_yaw_unwind(self):
        for rebound in (.03, .05, .07):
            for lag in (.04, .06, .1):
                with self.subTest(rebound=rebound, lag=lag):
                    e, error = self.yaw_response_fixture(rebound, lag)
                    self.assertIsNone(e.result, e.result)
                    self.assertEqual(e.state, m.MovementState.DRIVE_FORWARD)
                    self.assertLess(abs(error), .10)

    def test_larger_uncharacterized_yaw_unwind_stops_without_false_success(self):
        for lag in (.04, .06, .1):
            e, _ = self.yaw_response_fixture(.10, lag)
            self.assertEqual(e.result.outcome, 'censored')
            self.assertFalse(e.result.physical)
            self.assertIn('angular progress', e.result.reason)

    def test_invalid_brake_envelope_uses_zero_not_new_reverse_drive(self):
        e = self.executor(); e.brake_guard = lambda obs: 'excluded stopping corridor'
        self.assertEqual(e.brake(observation(0., velocity=(.3, 0.))), m.Motion())
        self.assertIn('excluded', e.brake_rejection)

    def test_remaining_corridor_checked_before_alignment(self):
        e = self.executor((0., 2.))
        e.corridor_guard = lambda obs, req, heading: 'new approach blocked'
        self.drive(e)
        self.assertEqual(e.result.reason, 'new approach blocked')
        self.assertFalse(e.result.physical)
        self.assertNotIn(m.MovementState.ALIGN_LEFT.value, [v['state'] for v in e.history])

    def test_new_correction_does_not_inherit_old_alignment_timeout(self):
        e = self.executor((10., 0.)); self.drive(e)
        e.alignment_best, e.alignment_progress_at = .05, 0.
        self.tick(e, observation(20., xy=(1., .3), yaw=.2))
        for t in (20.1, 20.3, 20.5): self.tick(e, observation(t, xy=(1., .3), yaw=.2))
        self.assertLess(self.tick(e, observation(20.6, xy=(1., .3), yaw=.2)).omega, 0.)
        self.assertIsNone(e.result)

    def test_corrected_heading_is_latched_across_steep_entry(self):
        e = self.executor()
        xy = (.5, .35)
        heading = math.atan2(-.35, 1.5)
        for t in (0., .2, .4): self.tick(e, observation(t, xy=xy, yaw=heading))
        self.assertAlmostEqual(e.committed_heading, heading)
        command = self.tick(e, observation(.5, xy=xy, yaw=heading, grade=30., turn=False))
        self.assertEqual(command, m.Motion(.4, 0.))
        self.assertIsNone(e.result)

    def test_progress_tracks_realigned_axis_instead_of_original_bearing(self):
        e = self.executor()
        origin = np.array((1.9, .4))
        heading = math.atan2(-.4, .1)
        axis = np.array((math.cos(heading), math.sin(heading)))
        for t in (0., .2, .4): self.tick(e, observation(t, tuple(origin), heading))
        for t in np.arange(.5, 6.5, .1):
            self.tick(e, observation(t, tuple(origin+.04*(t-.5)*axis), heading, tuple(.04*axis)))
        self.assertIsNone(e.result)
        self.assertAlmostEqual(e.metrics['progress_mps'], .04)

    def test_overshoot_is_repositioned_not_fabricated_settle_failure(self):
        e = self.executor(); self.drive(e)
        self.tick(e, observation(1., xy=(2.4, 0.)))
        self.assertEqual(e.result.disposition, m.ResultDisposition.REPOSITIONED)
        self.assertEqual(e.result.outcome, 'censored')
        self.assertFalse(e.result.physical)

    def test_rapid_posture_change_observes_before_judging_approach(self):
        e = self.executor(); self.drive(e, grade=30., turn=False)
        e.posture_guard = lambda *args: ''
        self.tick(e, replace(observation(.5, yaw=.3, grade=30., turn=False), up_z=math.cos(math.radians(25.))))
        self.tick(e, replace(observation(1., yaw=.3, grade=30., turn=False), up_z=math.cos(math.radians(45.))))
        self.assertEqual(e.state, m.MovementState.POSTURE_PAUSE)
        self.assertIsNone(e.result)

    def test_backslide_is_faster_than_ordinary_stall(self):
        e = self.executor(); self.drive(e, grade=30., turn=False)
        for t in np.arange(.5, 1.8, .1):
            self.tick(e, observation(t, xy=(-.15*t, 0.), yaw=.3, velocity=(-.15, 0.), grade=30., turn=False))
        self.assertIn('backslide', e.result.reason)

    def test_near_endpoint_is_a_settle_action_not_a_tiny_drive(self):
        e = m.MovementExecutor()
        e.start(m.MovementRequest('near', (0., 0.), (.01, 0.), 1, .4, 10., 'settle'), 0.)
        for t in (0., .2, .4, .6): self.tick(e, observation(t))
        self.assertEqual(e.result.outcome, 'success')

    def test_aligned_slope_exit_can_take_over_from_slow_creep(self):
        e = self.executor((-2., 0.), -1)
        for t in (0., .2, .4): self.tick(e, observation(t, velocity=(-.12, 0.), grade=30., turn=False))
        self.assertEqual(self.tick(e, observation(.5, velocity=(-.12, 0.), grade=30., turn=False)), m.Motion(-.4, 0.))


def ground(degrees=0.):
    grid = a.AscentTerrainMap(world_size=12., footprint_radius=.3)
    x, y = np.meshgrid(np.arange(-4.9, 5., .1), np.arange(-4.9, 5., .1), indexing='ij')
    grid.ingest(np.c_[x.ravel(), y.ravel(), (math.tan(math.radians(degrees))*x).ravel()], (0, 0, 3), 0.)
    grid.rebuild(0.)
    return grid


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.flat = ground()

    def control(self, grid=None):
        return b.BehaviorTreeAscentController(copy.deepcopy(grid or self.flat),
            [SimpleNamespace(x=3., y=0., label='goal')], mode='characterize')

    def test_profile_raises_uphill_without_changing_baseline(self):
        ctrl = self.control()
        self.assertEqual(ctrl.capability.uphill_limit_deg, 33.)
        self.assertEqual(ctrl.capability.side_limit_deg, 25.)
        self.assertEqual(ctrl.grid.planning_grade_reserve, 0.)
        self.assertEqual(ctrl.grid.MAX_SLOPE_DEG, 35.)

    def test_stop_wins_and_manual_does_not_auto_resume(self):
        ctrl = self.control()
        ctrl.tick((0, 0), 0., 0., 0.)
        ctrl._start((2, 0), 1, 'mission')
        generation = ctrl.plan_generation
        ctrl.request('stop', 1.)
        self.assertGreater(ctrl.plan_generation, generation)
        self.assertEqual(ctrl.tick((0, 0), 0., .2, 1., velocity=(.2, 0.)), m.Motion())
        self.assertIsNone(ctrl.executor.request)
        ctrl.request('manual', 2., vx=-.4)
        self.assertEqual(ctrl.tick((0, 0), 0., 0., 2.), m.Motion(-.4, 0.))
        ctrl.request('manual', 3.)
        self.assertEqual(ctrl.tick((0, 0), 0., 0., 3.), m.Motion())
        self.assertEqual(ctrl.state, 'MANUAL')

    def test_steep_footprint_disables_turn_even_with_flat_centre(self):
        ctrl = self.control(); g = ctrl.grid
        centre = g.world_to_cell(0, 0)
        g.slope[centre[0]+1, centre[1]] = math.tan(math.radians(25.))
        ctrl.tick((0, 0), 0., 0., 0.)
        self.assertFalse(ctrl.turn_enabled)
        self.assertGreater(ctrl.obs.inclination, 15.)

    def test_gentle_ground_policy_lean_does_not_become_steep_terrain(self):
        ctrl = self.control()
        ctrl.tick((0, 0), 0., 0., 0., upright=math.cos(math.radians(18.)))
        self.assertTrue(ctrl.turn_enabled)
        self.assertLess(ctrl.obs.inclination, 12.)
        self.assertAlmostEqual(ctrl.obs.body_tilt, 18.)

    def test_recovery_does_not_start_a_known_blocked_command(self):
        ctrl = self.control(); ctrl.tick((0, 0), 0., 0., 0.)
        ctrl._begin_recovery('fixture')
        with patch.object(ctrl, '_guard', return_value='blocked stopping corridor'):
            ctrl.tick((0, 0), 0., 0., 2.)
        self.assertIsNone(ctrl.executor.request)
        self.assertEqual(ctrl.recovery_actions, 0)

    def test_hold_and_observe_have_no_yaw_on_slopes(self):
        ctrl = self.control(ground(30.))
        ctrl.state = 'HOLD'
        for t in (0., .1, .2):
            command = ctrl.tick((0, 0), t, .1, t, velocity=(.1, 0.))
            self.assertEqual(command.omega, 0.)

    def test_new_plan_cannot_interrupt_running_primitive(self):
        ctrl = self.control(); ctrl.tick((0, 0), 0., 0., 0.)
        ctrl._start((2., 0.), 1, 'mission')
        with patch.object(ctrl, '_plan', side_effect=AssertionError('preempted')):
            for t in (.1, .2, .5, .6): ctrl.tick((0, 0), 0., 0., t)

    def test_useful_supported_displacement_replans_without_reverse_recovery(self):
        ctrl = self.control(); ctrl.tick((0., 0.), 0., 0., 0.)
        ctrl.executor.result = m.MovementResult('fixture', 'censored', 'different supported endpoint', False,
            1., (0., 0.), 'DRIVE_FORWARD', m.ResultDisposition.REPOSITIONED)
        ctrl.tick((0., 0.), 0., 0., 1.)
        self.assertEqual(ctrl.state, 'OBSERVE')
        self.assertFalse(ctrl.recovering)
        self.assertIsNone(ctrl.executor.request)
        self.assertIn('measured pose', ctrl.reason)

    def test_straight_33_degree_corridor_and_no_steep_bends(self):
        ctrl = self.control(ground(33.)); g = ctrl.grid
        start, goal = g.cell_to_world(25, 30), g.cell_to_world(40, 30)
        path, done = r.observed_route(g, start, goal, 0., allow_frontier=False)
        self.assertTrue(done)
        self.assertEqual(len(path), 2)
        _, bent = r.observed_route(g, start, g.cell_to_world(40, 33), 0., allow_frontier=False)
        self.assertFalse(bent)

    def test_staging_route_can_turn_on_flat_terrain(self):
        ctrl = self.control(); g = ctrl.grid
        path, done = r.observed_route(g, (0., 0.), (3., 1.4), 0., allow_frontier=False)
        self.assertTrue(done)
        self.assertGreaterEqual(len(path), 2)

    def test_manual_checkpoint_selection_preserves_target(self):
        ctrl = self.control(); ctrl.checkpoint_mode(True, 0.)
        self.assertTrue(ctrl.choose_checkpoint((0., 0.), (2., 0.), 1.))
        self.assertEqual((ctrl.waypoints[0].x, ctrl.waypoints[0].y), (2., 0.))
        self.assertEqual(ctrl.operator_mode, 'auto')

    def test_continuous_steep_starts_do_not_need_cell_centre_alignment(self):
        ctrl = self.control(ground(33.)); g = ctrl.grid
        for start in ((0., 0.), (-.899, -.299), (.021, .079)):
            goal = (start[0]+2., start[1])
            path, done = r.observed_route(g, start, goal, 0., allow_frontier=False)
            self.assertTrue(done, start)
            self.assertEqual(path, [start, goal])

    def test_async_adoption_rechecks_changed_turn_staging(self):
        ctrl = self.control()
        shadow = copy.copy(ctrl); shadow.grid = copy.deepcopy(ctrl.grid)
        shadow.attempt_events = []; shadow.planner = None
        self.assertTrue(shadow._compute_plan((0., 0.), 0., 0.))
        shadow.path = [(0., 0.), (1., 0.), (1., 1.), (3., 0.)]
        corner = ctrl.grid.world_to_cell(1., 0.)
        ctrl.grid.slope[corner] = math.tan(math.radians(20.))
        ctrl.planner = SimpleNamespace()
        ctrl.plan_job = Future(); ctrl.plan_job.set_result((ctrl.plan_generation, True, shadow))
        self.assertFalse(ctrl._plan((0., 0.), 0., 1.))
        self.assertIn('staging', ctrl.planning_failure)
        ctrl.planner = None

    def test_near_goal_is_terminal_and_short_final_leg_is_allowed(self):
        ctrl = self.control()
        self.assertTrue(ctrl._compute_plan((0., 0.), 0., 0.))
        self.assertTrue(ctrl.is_mission_checkpoint)
        self.assertEqual(ctrl.checkpoint, (3., 0.))
        self.assertTrue(ctrl._compute_plan((2.9, 0.), 0., 0.))
        self.assertEqual(ctrl.checkpoint, (3., 0.))

    def test_arbitrary_manual_goal_does_not_leave_tiny_drive(self):
        ctrl = self.control(); g = ctrl.grid
        path, done = r.observed_route(g, (0., 0.), (3.09, .1), 0., allow_frontier=False)
        self.assertTrue(done)
        ctrl.tick((3.1, .1), 0., 0., 0.)
        ctrl._start(path[-1], 1, 'mission')
        self.assertEqual(ctrl.executor.state, m.MovementState.SETTLE)

    def test_active_recovery_cannot_extend_episode_deadline(self):
        ctrl = self.control(); ctrl.tick((0., 0.), 0., 0., 0.)
        ctrl.recovering = True; ctrl.recovery_since = 0.
        ctrl._start((2., 0.), -1, 'recovery')
        ctrl.tick((0., 0.), 0., 0., 121.)
        self.assertIsNone(ctrl.executor.request)
        self.assertEqual(ctrl.executor.result.reason, 'recovery episode deadline elapsed')

    def test_legacy_model_cannot_qualify_new_protocol(self):
        model = SimpleNamespace(artifact={'capability_profile': {}}, model_id='legacy')
        with self.assertRaisesRegex(ValueError, 'protocol'):
            b.BehaviorTreeAscentController(copy.deepcopy(self.flat),
                [SimpleNamespace(x=3., y=0.)], mode='qualified', model=model)


if __name__ == '__main__': unittest.main()
