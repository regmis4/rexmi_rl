"""Unfinished recovery pause ownership; recorded replay is not physics."""
import copy
import importlib.util
import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import test_remote_control as t

r = t.r


@pytest.fixture(scope='module')
def ground():
    return t.bt.ground()


def pause(ground, sign=-1):
    c = t.departure_fixture(ground, sign)
    c.obs = t.bt.observation(11.02, velocity=(-sign*.05, 0.))
    assert c._navigate() == r.Motion()
    assert c.button_state == 'OBSERVE' and not c.path
    assert c.recovery_pause is not None
    return c


def budgets(c):
    return (c.recovery_episode, c.recovery_actions, c.recovery_progress_at,
            c.recovery_best_remaining, c.recovery_credited_remaining,
            c.progress_time, c.low_progress_active, c.drive_exposure)


@pytest.mark.parametrize('sign', [-1, 1])
def test_pause_checks_both_signs_without_emitting_or_renewing(ground, sign):
    c = pause(ground, sign)
    original = (budgets(c), copy.deepcopy(c.recovery_pause), c.recovery_permission,
                c.grid.failures.copy(), c.grid.approach_blocked.copy(), c.command)
    for _ in range(3):
        assert c._unfinished_recovery_pause()
        assert c._checked_recovery_pause(r.Motion())
        assert c._geometry(r.Motion()) == ''
        assert c._geometry(r.Motion(-sign*.4, 0.))
    assert budgets(c) == original[0]
    assert c.recovery_pause == original[1]
    assert c.recovery_permission is original[2]
    np.testing.assert_array_equal(c.grid.failures, original[3])
    np.testing.assert_array_equal(c.grid.approach_blocked, original[4])
    assert c.command == original[5]


@pytest.mark.parametrize('which', ['intent', 'forecast'])
def test_pause_cannot_borrow_after_departure_rejection(ground, which):
    c = pause(ground)
    name = '_command_geometry' if which == 'intent' else '_anticipation'
    with patch.object(c, name, return_value='blocked departure'):
        assert not c._checked_recovery_pause(r.Motion())
        assert 'excluded approach' in c._actual_geometry(r.Motion())


@pytest.mark.parametrize('defect', ['speed', 'expired', 'new_episode', 'reached',
                                  'passed', 'behind', 'path', 'emergency', 'yaw', 'powered'])
def test_invalid_or_stale_pause_context_does_not_grant_permission(ground, defect):
    c = pause(ground)
    command = r.Motion()
    if defect == 'speed': c.obs = t.bt.observation(11.02, velocity=(.12001, 0.))
    if defect == 'expired': c.obs = t.bt.observation(c.recovery_episode+120., velocity=(.05, 0.))
    if defect == 'new_episode': c.recovery_episode += .02
    if defect == 'reached': c.obs = t.bt.observation(11.02, xy=c.recovery_endpoint)
    if defect == 'passed': c.obs = t.bt.observation(11.02, xy=(c.recovery_endpoint[0]-.3, .5))
    if defect == 'behind': c.obs = t.bt.observation(11.02, yaw=math.pi)
    if defect == 'path': c.path = [(0., 0.), (2., 0.)]
    if defect == 'emergency': c.emergency = True
    if defect == 'yaw': command = r.Motion(0., .35)
    if defect == 'powered': command = r.Motion(-.4, 0.)
    assert not c._checked_recovery_pause(command)


@pytest.mark.parametrize('action', ['stop', 'manual'])
def test_operator_request_discards_pause(ground, action):
    c = pause(ground)
    c.request(action, 11.04)
    assert c.recovery_pause is None and c.recovery_permission is None


def test_observed_exit_retires_pause_and_return_does_not_revive(ground):
    c = pause(ground)
    with patch.object(c, '_navigate', return_value=r.Motion()):
        c.tick((-1.01, 0.), 0., 0., 11.04, velocity=(0., 0.))
    assert c.recovery_pause is None
    assert not c.recovery_permission.patches
    c.obs = t.bt.observation(11.06, velocity=(.05, 0.))
    assert 'excluded approach' in c._geometry(r.Motion())


def test_new_failure_patch_is_not_inherited(ground):
    c = pause(ground)
    old = c.recovery_permission
    c.grid.mark_failure('new', (0., 0.), 0., 11.03, 'new evidence')
    assert c._geometry(r.Motion())
    assert c.recovery_permission is old


def test_actual_hazard_still_rejects_pause(ground):
    c = pause(ground)
    c.grid.hazard[c.grid.world_to_cell(0., 0.)] = True
    assert c._geometry(r.Motion())


def test_episode_deadline_holds_and_discards_context(ground):
    c = pause(ground)
    out = c.tick((0., 0.), 0., 0., c.recovery_episode+120., velocity=(0., 0.))
    assert out == r.Motion() and c.button_state == 'HOLD'
    assert c.recovery_pause is None and c.recovery_permission is None


def test_adopted_plan_discards_pause_without_resetting_episode(ground):
    c = pause(ground)
    before = c.recovery_episode, c.recovery_actions
    def adopted(*args):
        c.path = [(0., 0.), (2., 0.)]
        c.target = (2., 0.)
        return True
    with patch.object(c, '_plan', side_effect=adopted): c._navigate()
    assert c.recovery_pause is None and c.button_state == 'DRIVE'
    assert (c.recovery_episode, c.recovery_actions) == before


def test_new_maneuver_does_not_inherit_pause_context(ground):
    c = pause(ground)
    episode, permission, actions = c.recovery_episode, c.recovery_permission, c.recovery_actions
    with patch.object(c, 'finish_attempt'):
        c._recover('another checked maneuver')
    assert c.recovery_pause is None
    assert c.recovery_episode == episode and c.recovery_actions == actions+1
    assert c.recovery_permission is permission


def test_time_rewind_discards_pause_and_permission(ground):
    c = pause(ground)
    c.last_tick = 11.02; c.last_yaw = 0.
    with patch.object(c, '_navigate', return_value=r.Motion()):
        c.tick((0., 0.), 0., 0., 10., velocity=(0., 0.))
    assert c.recovery_pause is None and c.recovery_permission is None


def test_mission_completion_discards_pause(ground):
    c = pause(ground)
    c.wp_idx = len(c.waypoints)
    c._navigate()
    assert c.recovery_pause is None and c.recovery_permission is None


@pytest.mark.parametrize('endpoint_result', ['reached', 'passed'])
def test_finished_maneuver_never_creates_pause(ground, endpoint_result):
    c = t.departure_fixture(ground)
    xy = c.recovery_endpoint if endpoint_result == 'reached' else (c.recovery_endpoint[0]-.3, .5)
    c.obs = t.bt.observation(11.02, xy=xy)
    c._navigate()
    assert c.recovery_pause is None


def test_recorded_6792_handoff_uses_original_unfinished_maneuver():
    root = Path(__file__).resolve().parents[1]
    helper = root/'outputs/remote_autonomy/reachable-height-objective/handoff_replay.py'
    spec = importlib.util.spec_from_file_location('recorded_handoff', helper)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    c, rows, failure = module.fixture()
    run = root/'outputs/remote_autonomy/height-retaining-ascent-JWGRFI3h'
    events = [json.loads(line) for line in (run/'run.events.jsonl').read_text().splitlines()]
    event = next(e for e in events if e.get('event') == 'maneuver_start' and e['time'] == failure['time'])
    first = next(q for q in rows if float(q['sim_time']) == event['time'])
    c.recovery_start = tuple(float(first[k]) for k in ('x', 'y'))
    c.recovery_endpoint = tuple(event['endpoint'])
    c.direction = event['direction']; c.recovery_kind = event['purpose']
    c.recovery_actions = 1
    c.recovery_best_remaining = c.recovery_credited_remaining = math.dist(c.obs.xy, c.recovery_endpoint)
    c.recovery_progress_at = c.obs.time-8.
    c.button_state = 'RECOVER'; c.path = []
    assert c._geometry(r.Motion())
    before = budgets(c)
    assert c._navigate() == r.Motion()
    assert c._checked_recovery_pause(r.Motion())
    assert c._geometry(r.Motion()) == ''
    assert budgets(c) == before
    assert c.recovery_pause['endpoint'] == tuple(event['endpoint'])
    assert len(c.recovery_permission.patches) == 3
    previous, last = rows[-2:]
    c.last_tick = float(previous['sim_time']); c.last_yaw = float(previous['yaw'])
    c.next_plan_at = c.obs.time+1.  # Planning is deliberately absent from this gate replay.
    result = c.tick(c.obs.xy, c.obs.yaw, c.obs.speed, c.obs.time,
                    upright=float(last['up_z']), velocity=c.obs.velocity)
    assert result == r.Motion() and c.button_state == 'OBSERVE'
    assert budgets(c) == before
