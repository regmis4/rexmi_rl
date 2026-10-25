"""Exact local348.78 map replay; no claim about subsequent physical response."""
import copy
import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from test_behavior_navigation import a, b, m
from rexmi_bt_tests.nav.departure_diagnostics import MAP_FIELDS


FIXTURE = Path(__file__).parent / 'fixtures/grade_release_34878.npz'


def recorded_grade_release():
    """Restore captured post-observation state and exact local map fields."""
    with np.load(FIXTURE, allow_pickle=False) as data:
        meta = json.loads(data['metadata_json'].item())
        cfg = meta['grid_config']
        g = a.AscentTerrainMap(**{k: cfg[k] for k in (
            'world_size', 'cell_size', 'origin', 'footprint_radius',
            'clearance', 'half_length', 'half_width')})
        selection = tuple(slice(i, i+25) for i in meta['first_cell'])
        for name in MAP_FIELDS:
            getattr(g, name)[selection] = data[name]
    for name, value in cfg.items():
        setattr(g, name, value)
    g.now = meta['map_time']
    g.revision = meta['map_revision']
    g.terrain_prior = meta['terrain_prior']
    g.failure_records = copy.deepcopy(meta['failure_records'])
    c = b.BehaviorTreeAscentController(
        g, [SimpleNamespace(x=0., y=13., label='exterior')], mode='characterize')
    c.configure_active_zero_envelope()
    c.obs = m.MotionObservation(**meta['observation'])
    c.passive_heading.samples = copy.deepcopy(meta['passive_heading_samples'])
    e = c.executor
    e.start(m.MovementRequest(**meta['request']), meta['executor']['started'])
    for name, value in meta['executor'].items():
        if name in ('state', 'executed_traversal', 'release_memory'):
            continue
        setattr(e, name, copy.deepcopy(value))
    e.state = m.MovementState(meta['executor']['state'])
    e.executed_traversal = m.ExecutedTraversal(**meta['executor']['executed_traversal'])
    for name, value in meta['executor']['release_memory'].items():
        setattr(e.release_memory, name, copy.deepcopy(value))
    c.active_zero_departures[e.request.action_id] = e.ascent_intent_bound
    return c, meta


def prepare_recorded_tick(c):
    """Undo only the captured current interval; tick must observe it once."""
    e, o = c.executor, c.obs
    previous_t, previous_xy = e.uphill_samples[-2]
    dt = o.time-previous_t
    e.uphill_samples = e.uphill_samples[:-1]
    e.drive_response_samples = e.drive_response_samples[:-1]
    e.posture = [p for p in e.posture if p[0] < o.time]
    e.pending_traversal = (previous_t, previous_xy, e.executed_traversal.body_yaw,
                           'DRIVE_FORWARD', 1, e.committed_heading)
    e.executed_traversal = replace(e.executed_traversal,
        interval_end_time=previous_t, last_position=previous_xy,
        duration_s=e.executed_traversal.duration_s-dt)
    e.drive_exposure -= dt
    e.last_output_time = previous_t
    e.forward_output = True


class GradeReleaseTests(unittest.TestCase):
    def test_exact_map_distinguishes_powered_grade_from_physical_support(self):
        c, meta = recorded_grade_release()
        o = c.obs
        end = tuple(o.xy[i]+.5*v for i, v in enumerate((math.cos(o.yaw), math.sin(o.yaw))))
        self.assertEqual(c.grid.now, 348.6)
        self.assertIn('Exact25x25', meta['fixture_provenance']['map'])
        cell = c.grid.world_to_cell(.7, 6.3)
        self.assertEqual(c.grid.last_seen[cell], 348.6)
        self.assertTrue(c.grid.fresh[cell])
        self.assertFalse(c.grid.known[c.grid.world_to_cell(20., 20.)])
        self.assertIn('uphill grade', c.grid.grade_reason(o.xy, end))
        self.assertEqual(c._stopping_reason(o, end, .4, commanded=False, record=False), '')
        self.assertTrue(c._guard(m.Motion(.4, 0.), record=False))
        self.assertEqual(c._active_zero_departure(o, c.executor.request), '')
        self.assertEqual(c.executor.state, m.MovementState.DRIVE_FORWARD)

    def test_recorded_tick_exposes_existing_lower_power_selection(self):
        c, _ = recorded_grade_release()
        e, o = c.executor, c.obs
        before = (e.request, e.started, e.ascent_intent_bound, e.uphill_stall_since)
        prepare_recorded_tick(c)
        command = e.tick(o, c._guard, lambda xy: not c.grid.support_reason(xy))
        self.assertAlmostEqual(command.vx, .35)
        self.assertEqual(command.omega, 0.)
        self.assertEqual(c._guard(command, record=False), '')
        self.assertEqual(e.state, m.MovementState.DRIVE_FORWARD)
        self.assertIsNone(e.result)
        self.assertEqual(before, (e.request, e.started, e.ascent_intent_bound, e.uphill_stall_since))
        self.assertEqual(sum(x['event']=='powered_grade_release_selected' for x in e.history), 0)
        self.assertTrue(any(x['event']=='forward_speed_selected' for x in e.history))
        # Captured posture is degraded, so this is permission to observe, not
        # proof of useful ascent or a fabricated positive height response.
        self.assertGreater(o.body_tilt, 47.)

    def test_cold_wrong_owner_history_and_bound_do_not_gain_permission(self):
        for kind in ('cold', 'reverse', 'recovery', 'missing_history', 'heading_gap',
                     'outside_bound'):
            with self.subTest(kind=kind):
                c, _ = recorded_grade_release(); e = c.executor
                if kind == 'cold':
                    e.ascent_intent_bound = None
                    c.active_zero_departures.clear()
                elif kind == 'reverse': e.request = replace(e.request, direction=-1)
                elif kind == 'recovery': e.request = replace(e.request, purpose='recovery')
                elif kind == 'missing_history': e.executed_traversal = None
                elif kind == 'heading_gap': del c.passive_heading.samples[3:7]
                else:
                    c.obs = replace(c.obs, xy=(-20., -20.))
                self.assertTrue(c._active_zero_departure(c.obs, e.request))

    def test_physical_rejection_cannot_hide_behind_first_powered_grade(self):
        for kind in ('steep', 'unknown', 'stale', 'hazard', 'no_go'):
            with self.subTest(kind=kind):
                c, _ = recorded_grade_release(); g = c.grid
                cell = g.world_to_cell(.7, 6.3)
                if kind == 'steep': g.slope[cell] = math.tan(math.radians(36.))
                elif kind == 'unknown': g.known[cell] = False
                elif kind == 'stale':
                    g.fresh[cell] = False; g.last_seen[cell] = 0.; g.surface_seen[cell] = 0.
                elif kind == 'hazard': g.hazard[cell] = True; g.blocked[cell] = True
                else: g.no_go[cell] = True; g.blocked[cell] = True
                self.assertTrue(c._active_zero_departure(c.obs, c.executor.request))

    def test_actual_selected_command_remains_guarded_without_automatic_zero(self):
        c, _ = recorded_grade_release(); e = c.executor
        # Counterexample: selecting nominal power cannot borrow the independent
        # physical intent verdict. This is not the recorded selector setting.
        e.drive_selector = lambda command: command
        prepare_recorded_tick(c)
        e.tick(c.obs, c._guard, lambda xy: not c.grid.support_reason(xy))
        self.assertEqual(e.result.outcome, 'censored')
        self.assertIn('straight drive', e.result.reason)
        self.assertFalse(e.result.physical)
        self.assertFalse(any(x['event']=='drive_release_selected' for x in e.history))

    def test_real_slowdown_predicate_selects_zero_and_retains_early_history(self):
        # Synthetic response on33.5-degree terrain: this exercises the existing
        # measured slowdown predicate; it is not348.78 telemetry or a new trigger.
        from test_active_zero_envelope import ActiveZeroTests
        c = ActiveZeroTests().synthetic(); e, o = c.executor, c.obs
        e.state = m.MovementState.DRIVE_FORWARD
        e.entered = 1.; e.drive_exposure = 6.; e.drive_slow_since = o.time-1.1
        e.travel_axis = (1., 0.); e.committed_heading = 0.
        e.uphill_samples = [(float(t), (o.xy[0]+.01*(t-o.time), o.xy[1]))
                            for t in np.linspace(o.time-2., o.time, 101)]
        e.drive_response_samples = copy.deepcopy(e.uphill_samples)
        e.samples = [(t, xy[0], xy) for t, xy in e.uphill_samples[:-1]]
        e.executed_traversal = replace(e.executed_traversal, maneuver='DRIVE_FORWARD')
        e.ascent_intent_bound = math.dist(e.request.start, e.request.endpoint)+.08
        c.active_zero_departures[e.request.action_id] = e.ascent_intent_bound
        before = (e.started, e.request.deadline, e.request.start, e.request.endpoint)
        prepare_recorded_tick(c)
        self.assertEqual(e.tick(o, c._guard, lambda xy: True), m.Motion())
        self.assertEqual(e.state, m.MovementState.RELEASE)
        self.assertTrue(any(x['event']=='drive_release_selected' for x in e.history))
        entered = e.entered
        next_obs = replace(o, time=o.time+.02, xy=(o.xy[0]+.0002, o.xy[1]))
        c.obs = next_obs
        c.passive_heading.observe(next_obs.time, next_obs.yaw)
        self.assertEqual(e.tick(next_obs, c._guard, lambda xy: True), m.Motion())
        self.assertIsNone(e.result)
        self.assertEqual(e.state, m.MovementState.RELEASE)
        self.assertEqual(e.entered, entered)
        self.assertEqual(before, (e.started, e.request.deadline, e.request.start, e.request.endpoint))


if __name__ == '__main__':
    unittest.main()
