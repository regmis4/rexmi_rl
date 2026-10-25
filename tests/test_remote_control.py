"""Final remote command tests. Recorded observation replay is NOT a physics replay."""
import copy
import csv
import importlib
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import test_behavior_navigation as bt

r = importlib.import_module('rexmi_bt_tests.nav.remote_control')
adapter = importlib.import_module('rexmi_bt_tests.drive_adapter')
mission = importlib.import_module('rexmi_bt_tests.nav.mission')


@pytest.fixture(scope='module')
def ground():
    return bt.ground()


def controller(ground, target=(4., 0.)):
    c = r.RemoteAscentController(copy.deepcopy(ground),
        [mission.Waypoint(*target, .4, 'destination')], mode='characterize')
    c.path = [(0., 0.), target]
    c.checkpoint = target
    c.is_mission_checkpoint = True
    c.button_state = 'DRIVE'
    c.progress_time = 3.
    c.progress_anchor = (0., 0.)
    return c


def test_turn_holds_beyond_two_seconds_through_final_policy_input(ground):
    c = controller(ground, (0., 4.))
    cmd = np.zeros((1, 3))
    env = SimpleNamespace(unwrapped=SimpleNamespace(
        command_manager=SimpleNamespace(get_command=lambda _: cmd),
        observation_manager=SimpleNamespace(compute=lambda **_: {'policy': cmd.copy()})))
    for t in np.arange(3., 8., .02):
        m = c.tick((0., 0.), (t-3.)*.12, 0., t, velocity=(0., 0.))
        delivered = adapter.policy_actions(env, lambda x: x, m.vx, m.omega)
        assert delivered[0, 2] == .35
        assert delivered[0, 0] == 0.
    assert c.button_state == 'TURN'
    assert not any(e.get('state') == 'BRAKE' for e in c.attempt_events)


def test_measured_turn_crossing_resumes_drive_without_settling(ground):
    c = controller(ground, (0., 4.))
    c.tick((0., 0.), 0., 0., 3.)
    m = c.tick((0., 0.), math.pi/2-.05, 0., 8.)
    assert (m.vx, m.omega) == (.4, 0.)


def test_ordinary_speed_variation_does_not_brake(ground):
    c = controller(ground)
    for i, v in enumerate((.4, .451, .46, .52, .48, .4)):
        m = c.tick((i*.1, 0.), 0., v, 3.+i*.25, velocity=(v, 0.))
        assert (m.vx, m.omega) == (.4, 0.)


def test_emergency_latches_until_stable_and_hard_limit_stops(ground):
    c = controller(ground)
    c.tick((0., 0.), 0., .81, 3., velocity=(.81, 0.))
    assert c.emergency and c.command.vx == -.4
    c.tick((0., 0.), 0., .4, 3.2, velocity=(.4, 0.))
    assert c.emergency and c.command.vx < 0.
    for t in (3.3, 3.5, 3.79):
        c.tick((0., 0.), 0., .1, t, velocity=(.1, 0.))
        assert c.emergency and c.command.vx == 0.
    c.tick((0., 0.), 0., .1, 3.81, velocity=(.1, 0.))
    assert not c.emergency
    assert c.tick((0., 0.), 0., 1., 4., velocity=(1., 0.)).vx == 0.
    assert c.state == 'HOLD'


def test_productive_zero_not_replaced_by_timer(ground):
    c = controller(ground)
    c.button_state = 'RELEASE'
    for t in np.arange(3., 23., .02):
        m = c.tick(((t-3.)*.06, 0.), 0., .06, t, velocity=(.06, 0.))
        assert m == r.Motion()
        assert c.button_state == 'RELEASE'


def test_progress_budget_survives_button_changes(ground):
    c = controller(ground)
    c.tick((0., 0.), 0., 0., 3.)
    with patch.object(c, '_recover', return_value=r.Motion()) as recover:
        for t in np.arange(3.02, 16., .02):
            c.button_state = 'DRIVE' if int(t)%2 else 'RELEASE'
            c.tick((0., 0.), 0., 0., t)
        assert recover.called
        assert c.progress_time == 3.


def test_stop_manual_deadline_and_nonfinite_precedence(ground):
    c = controller(ground)
    c.tick((0., 0.), 0., 0., 3.)
    c.request('stop', 3.1)
    assert c.tick((0., 0.), 0., .85, 3.2) == r.Motion()
    c.request('manual', 4., vx=-.4, omega=.35)
    assert c.tick((0., 0.), 0., 0., 4.1) == r.Motion(-.4, .35)
    c.request('resume', 5.)
    assert c.tick((math.nan, 0.), 0., 0., 5.) == r.Motion()
    assert c.operator_mode == 'stop'
    c = controller(ground)
    assert c.tick((0., 0.), 0., 0., 900.) == r.Motion()
    assert c.state == 'HOLD'


def test_waypoint_passage_does_not_stop_and_goal_requires_settling(ground):
    c = controller(ground)
    c.path = [(0., 0.), (.3, 0.), (4., 0.)]
    assert c.tick((0., 0.), 0., .2, 3., velocity=(.2, 0.)).vx == .4
    assert c.vertex == 2
    c.tick((3.7, 0.), 0., .3, 20., velocity=(.3, 0.))
    assert c.wp_idx == 0
    for t in np.arange(20.02, 20.7, .02):
        c.tick((3.7, 0.), 0., 0., t)
    assert c.wp_idx == 1


def test_geometry_rejection_not_physical_failure_and_tip_retained(ground):
    c = controller(ground)
    cell = c.grid.world_to_cell(.3, 0.)
    c.grid.hazard[cell] = True
    assert c.tick((0., 0.), 0., 0., 3.) == r.Motion()
    assert c.state == 'HOLD' and c.climb_failures == 0
    c = controller(ground)
    assert c.tick((0., 0.), 0., 0., 3., upright=.49) == r.Motion()
    assert c.state == 'HOLD'


def test_recovery_evaluates_both_directions_and_commits(ground):
    c = controller(ground)
    c.tick((0., 0.), 0., 0., 3.)
    directions = []
    def check(g, start, end, *args, **kwargs):
        directions.append(end[0]-start[0])
        return 'forward blocked' if end[0]>start[0] else ''
    with patch.object(r, 'oriented_sweep_reason', side_effect=check):
        m = c._recover('fixture stall')
    assert any(d>0 for d in directions) and any(d<0 for d in directions)
    assert m.vx == -.4
    with patch.object(c, '_geometry', return_value=''):
        for t in (3.1, 4., 5., 6.):
            assert c.tick((-.1*(t-3.), 0.), 0., .1, t, velocity=(-.1, 0.)).vx == -.4


def test_recorded_gui_turn_no_periodic_interruptions(ground):
    p = Path(__file__).resolve().parents[1]/'outputs/round_trip_user_check/round-trip-3f3TWTdP/run.csv'
    rows = [x for x in csv.DictReader(p.open()) if 38.06 <= float(x['sim_time']) <= 49.34]
    c = controller(ground, (-3., 3.))
    c.progress_time = 38.06
    origin = (float(rows[0]['x']), float(rows[0]['y']))
    with patch.object(c, '_geometry', return_value=''):
        for row in rows:
            xy = tuple(float(row[k])-origin[i] for i, k in enumerate(('x', 'y')))
            m = c.tick(xy, float(row['yaw']), float(row['speed']), float(row['sim_time']),
                upright=float(row['up_z']), velocity=(float(row['velocity_world_x']), float(row['velocity_world_y'])))
            assert m.omega == .35 and m.vx == 0.
    # The trajectory is recorded under old commands; this proves decisions only.


def test_recorded_gui_ordinary_speed_trace_never_sends_opposing_brake(ground):
    p = Path(__file__).resolve().parents[1]/'outputs/round_trip_user_check/round-trip-3f3TWTdP/run.csv'
    rows = [x for x in csv.DictReader(p.open()) if 58.82 <= float(x['sim_time']) <= 74.94]
    c = controller(ground, (-3., 3.))
    c.progress_time = 58.82
    origin = (float(rows[0]['x']), float(rows[0]['y']))
    with patch.object(c, '_geometry', return_value=''):
        for row in rows:
            xy = tuple(float(row[k])-origin[i] for i, k in enumerate(('x', 'y')))
            m = c.tick(xy, float(row['yaw']), float(row['speed']), float(row['sim_time']),
                upright=float(row['up_z']), velocity=(float(row['velocity_world_x']), float(row['velocity_world_y'])))
            assert m.vx >= 0. and not c.emergency


def test_remote_round_trip_configuration(ground):
    import test_round_trip as rt
    a = rt.args()
    a.ascent_architecture = 'remote'
    rt.rt.configure(a, [])
    assert a.map_retention == 'full'


def test_goal_not_credited_when_final_gate_rejects(ground):
    c = controller(ground)
    for t in np.arange(3., 3.6, .02):
        c.tick((3.8, 0.), 0., 0., t)
    with patch.object(c, '_geometry', return_value='stale goal footprint'):
        c.tick((3.8, 0.), 0., 0., 3.62)
    assert c.wp_idx == 0 and c.state == 'HOLD'


def test_emergency_and_zero_do_not_count_as_powered_execution(ground):
    c = controller(ground)
    c.tick((0., 0.), 0., .1, 3., velocity=(.1, 0.))
    c.tick((.02, 0.), 0., .81, 3.2, velocity=(.81, 0.))
    exposure = c.drive_exposure
    execution = dict(c.physical_execution)
    c.tick((.04, 0.), 0., .4, 3.4, velocity=(.4, 0.))
    assert c.drive_exposure == exposure
    assert c.physical_execution == execution
    c = controller(ground)
    c.tick((0., 0.), 0., .1, 3.)
    c.button_state = 'RELEASE'
    c.tick((.02, 0.), 0., .1, 3.2)
    execution = dict(c.physical_execution)
    c.tick((.04, 0.), 0., .1, 3.4)
    assert c.physical_execution == execution


def test_steep_turn_has_no_ordinary_yaw_output(ground):
    c = controller(ground, (0., 4.))
    c.grid.slope[:] = math.tan(math.radians(25.))
    with patch.object(c, '_geometry', return_value=''), patch.object(c, '_recover', return_value=r.Motion(-.4, 0.)) as recover:
        m = c.tick((0., 0.), 0., 0., 3.)
    assert recover.called and m.omega == 0.


def test_recovery_deadline_retires_only_after_checked_onward_progress(ground):
    c = controller(ground)
    c.recovery_episode = 0.
    c.recovery_actions = 2
    c.recovery_onward_anchor = (0., 0.)
    c.tick((.1, 0.), 0., .1, 3., velocity=(.1, 0.))
    assert c.recovery_episode == 0.
    c.tick((.6, 0.), 0., .1, 6., velocity=(.1, 0.))
    assert c.recovery_episode is None and c.recovery_actions == 0
    c.tick((.8, 0.), 0., .1, 125., velocity=(.1, 0.))
    c._recover('later independent stall')
    assert c.button_state == 'RECOVER' and c.recovery_episode == 125.


def test_goal_observation_gap_does_not_create_settle_credit(ground):
    c = controller(ground)
    for t in (3., 3.02, 5., 5.02):
        c.tick((3.8, 0.), 0., 0., t)
    assert c.wp_idx == 0


def test_emergency_discards_destination_settling_credit(ground):
    c = controller(ground)
    for t in np.arange(3., 3.5, .02):
        c.tick((3.8, 0.), 0., 0., t)
    with patch.object(c, '_geometry', return_value=''):
        c.tick((3.8, 0.), 0., .81, 3.5, velocity=(.81, 0.))
        for t in np.arange(3.52, 4.1, .02):
            c.tick((3.8, 0.), 0., 0., t)
    assert c.wp_idx == 0 and not c.emergency
    assert c.goal_settle_since > 4.


def survey_controller(ground, **metadata):
    g = copy.deepcopy(ground)
    g.terrain_prior = dict(kind='simulator_development', geometry_id='validated-fixture',
        file_sha256='validated-fixture', **metadata)
    return controller(g)


def test_static_survey_admits_geometry_without_mutating_freshness(ground):
    c = survey_controller(ground)
    c.grid.fresh[:] = False
    before = c.grid.fresh.copy()
    assert c.tick((0., 0.), 0., 0., 3.).vx == .4
    assert np.array_equal(c.grid.fresh, before)
    assert c.prior_admissions and not c.prior_support.flags.writeable
    assert any(e['event'] == 'remote_static_survey_admission' for e in c.attempt_events)


@pytest.mark.parametrize('defect', ['surface_conflict','unknown','hazard','steep','no_go'])
def test_static_survey_never_overrides_physical_or_conflicting_evidence(ground, defect):
    c = survey_controller(ground)
    c.grid.fresh[:] = False
    cell = c.grid.world_to_cell(0., 0.)
    if defect == 'surface_conflict': c.grid.surface[cell] += .081
    if defect == 'unknown': c.grid.known[cell] = False
    if defect == 'hazard': c.grid.hazard[cell] = True
    if defect == 'steep': c.grid.slope[cell] = math.tan(math.radians(36.))
    if defect == 'no_go': c.grid.no_go[cell] = True
    assert c.tick((0., 0.), 0., 0., 3.) == r.Motion()
    assert c.state == 'HOLD'


@pytest.mark.parametrize('metadata', [{}, {'kind':'retained_sensor_survey'},
    {'kind':'simulator_development','geometry_id':'x'},
    {'kind':'simulator_development','geometry_id':'x','file_sha256':'x','parent_geometry_id':'x'}])
def test_nonoriginal_or_unlabeled_prior_cannot_supply_stale_support(ground, metadata):
    g = copy.deepcopy(ground)
    g.terrain_prior = metadata
    c = controller(g)
    c.grid.fresh[:] = False
    assert c.tick((0., 0.), 0., 0., 3.) == r.Motion()
    assert c.state == 'HOLD' and not c.static_prior_enabled


def test_recovery_candidates_use_same_static_support_and_goal_ranking(ground):
    c = survey_controller(ground)
    c.grid.fresh[:] = False
    c.tick((0., 0.), 0., 0., 3.)
    c.path = []
    assert c._recover('no route').vx == .4
    assert c.prior_admissions > 0


def recovery_fixture(ground):
    c = controller(ground)
    c.tick((0., 0.), 0., 0., 3.)
    c._recover('heading reposition')
    return c


def test_productive_recovery_retains_route_and_budget_until_more_progress(ground):
    c = recovery_fixture(ground)
    path = list(c.path); clock = c.progress_time
    m = c.tick((1.01, 0.), 0., .3, 5., velocity=(.3, 0.))
    assert m.vx == .4 and c.path == path and c.button_state == 'DRIVE'
    assert c.progress_time == clock and c.recovery_episode == 3.
    assert c.recovery_actions == 1
    c.tick((1.4, 0.), 0., .3, 6., velocity=(.3, 0.))
    assert c.recovery_episode == 3.
    c.tick((1.6, 0.), 0., .3, 7., velocity=(.3, 0.))
    assert c.recovery_episode is None


@pytest.mark.parametrize('change', ['target','path','misaligned','unproductive'])
def test_recovery_requires_same_executable_intent(ground, change):
    c = recovery_fixture(ground)
    xy, yaw = (1.01, 0.), 0.
    if change == 'target': c.target = (4., 1.)
    if change == 'path': c.path = [(0., 0.), (3., 1.)]
    if change == 'misaligned': yaw = .5
    if change == 'unproductive': xy = (0., 1.01)
    with patch.object(c, '_geometry', return_value=''):
        c.tick(xy, yaw, 0., 11.01 if change == 'unproductive' else 5.)
    assert c.button_state == 'OBSERVE' and not c.path and c.recovery_episode == 3.


def test_route_resume_still_passes_final_physical_gate(ground):
    c = recovery_fixture(ground)
    with patch.object(c, '_geometry', return_value='new obstacle'):
        m = c.tick((1.01, 0.), 0., .3, 5., velocity=(.3, 0.))
    assert m == r.Motion() and c.state == 'HOLD' and c.wp_idx == 0


def test_recorded_recovery_handoff_keeps_original_descent_target(ground):
    # Recorded poses translated into the known test plane; physical map replay
    # is audited separately. This checks final controller intent handoff.
    target = (-1.9715797901153564, -10.19771671295166)
    c = controller(ground, target)
    start = (0., 0.)
    end = (-.2835718393325806, -.9704790115356445)
    c.grid.slope[:] = math.tan(math.radians(25.))
    with patch.object(c, '_geometry', return_value=''), patch.object(c, '_physical_sweep', return_value=''):
        c.progress_time = 77.84
        c.tick(start, -1.8863347981743168, .15686876315018405, 77.84,
            velocity=(-.04597580432891846, -.1499801129102707))
        c.target = target
        c._recover('turn translated onto steep terrain')
        clock = c.progress_time
        m = c.tick(end, -1.9355399782547704, .664796167798372, 79.62,
            velocity=(-.31114065647125244, -.5874907970428467))
    assert m.vx == .4 and m.omega == 0.
    assert c.path and c.target == target and c.wp_idx == 0
    assert c.progress_time == clock


def brake_controller(ground):
    c = controller(ground)
    c.grid.gx[:] = -math.tan(math.radians(34.))
    c.grid.gy[:] = 0.
    c.grid.slope[:] = abs(c.grid.gx)
    c.grid.uphill_limit_deg = 33.
    c.emergency = True
    return c


def test_opposing_damping_passes_full_tick_below_physical_boundary(ground):
    c = brake_controller(ground)
    assert c.tick((0., 0.), 0., .62, 3., velocity=(.62, 0.)) == r.Motion(-.4, 0.)
    assert c.state != 'HOLD' and c.emergency


@pytest.mark.parametrize('emergency,velocity,omega', [
    (False, (.62, 0.), 0.), (True, (-.62, 0.), 0.),
    (True, (0., 0.), 0.), (True, (0., .62), 0.), (True, (.62, 0.), .1)])
def test_intentional_reverse_sign_change_and_turn_cannot_borrow_damping(ground, emergency, velocity, omega):
    c = brake_controller(ground)
    c.obs = bt.observation(3., velocity=velocity)
    c.emergency = emergency
    assert 'uphill grade' in c._geometry(r.Motion(-.4, omega))


@pytest.mark.parametrize('defect', ['unknown','hazard','steep','stale','no_go','direction'])
def test_opposing_damping_preserves_physical_rejections(ground, defect):
    c = brake_controller(ground)
    cell = c.grid.world_to_cell(0., 0.)
    if defect == 'unknown': c.grid.known[cell] = False
    if defect == 'hazard': c.grid.hazard[cell] = True
    if defect == 'steep': c.grid.slope[cell] = math.tan(math.radians(36.))
    if defect == 'stale': c.grid.fresh[cell] = False
    if defect == 'no_go': c.grid.no_go[cell] = True
    if defect == 'direction': c.grid.approach_blocked[:] = True
    assert c.tick((0., 0.), 0., .62, 3., velocity=(.62, 0.)) == r.Motion()
    assert c.state == 'HOLD'


@pytest.mark.parametrize('defect', ['far_stale','near_stale','crossing_stale','far_hazard','far_no_go','far_steep'])
def test_adoption_shares_planner_near_field_and_full_geometry(ground, defect):
    c = controller(ground)
    a, b, observer = (1., 0.), (3., 0.), (0., 0.)
    far = c.grid.world_to_cell(2.8, 0.)
    near = c.grid.world_to_cell(1.5, 0.)
    if defect == 'far_stale': c.grid.fresh[far] = False
    if defect == 'near_stale': c.grid.fresh[near] = False
    if defect == 'crossing_stale':
        a, b = (-3., 0.), (3., 0.)
        c.grid.fresh[near] = False
    if defect == 'far_hazard':
        c.grid.hazard[far] = True
        c.grid.blocked[far] = True
    if defect == 'far_no_go':
        c.grid.no_go[far] = True
        c.grid.blocked[far] = True  # Map rebuild includes no-go in blocked.
    if defect == 'far_steep': c.grid.slope[far] = math.tan(math.radians(36.))
    reason = c._adoption_edge_reason(a, b, observer, False)
    assert bool(reason) == (defect != 'far_stale')


def departure_fixture(ground, sign=-1):
    c = controller(ground, (-sign*4., 0.))
    c.tick((0., 0.), 0., 0., 3.)
    def ingest(*args, **kwargs):
        c.grid.mark_failure('original', (0., 0.), 0. if sign == -1 else math.pi, 3., 'stall')
    with patch.object(c, 'finish_attempt', side_effect=ingest):
        assert c._recover('stall', physical=True).vx == sign*.4
    assert c.recovery_permission.patches  # Captured after ingestion.
    return c


@pytest.mark.parametrize('sign', [-1, 1])
def test_checked_departure_tolerates_residual_drift_in_both_directions(ground, sign):
    c = departure_fixture(ground, sign)
    cmd = c.tick((0., 0.), 0., .1, 3.02, velocity=(-sign*.1, 0.))
    assert cmd == r.Motion(sign*.4, 0.) and c.button_state == 'RECOVER'
    assert c._geometry(r.Motion(-sign*.4, 0.), escape=True) # Intentional banned retry.
    assert c._geometry(r.Motion(sign*.4, 0.), escape=False) # No recovery ownership.
    permission = c.recovery_permission
    mask = c.grid.approach_blocked.copy()
    for _ in range(3): assert c._geometry(cmd, escape=True) == ''
    assert c.recovery_permission is permission
    np.testing.assert_array_equal(mask, c.grid.approach_blocked)


def test_departure_never_inherits_later_failure_or_grown_patch(ground):
    c = departure_fixture(ground)
    frozen = c.recovery_permission
    c.grid.mark_failure('new', (0., 0.), 0., 3.01, 'later failure')
    assert c._recover('different recovery').vx == -.4
    assert c.recovery_permission is frozen
    assert c.tick((0., 0.), 0., .1, 3.02, velocity=(.1, 0.)) == r.Motion()
    assert c.state == 'HOLD'


def test_observed_exit_retires_permission_before_return(ground):
    c = departure_fixture(ground)
    c.tick((-1.01, 0.), 0., .1, 3.02, velocity=(-.1, 0.))
    assert not c.recovery_permission.patches
    c.obs = bt.observation(3.04, xy=(0., 0.), velocity=(.1, 0.))
    assert c._geometry(r.Motion(-.4, 0.), escape=True)


@pytest.mark.parametrize('action', ['stop','manual'])
def test_operator_request_clears_departure_permission(ground, action):
    c = departure_fixture(ground)
    c.request(action, 3.01)
    assert c.recovery_permission is None


@pytest.mark.parametrize('terminal', ['tilt','deadline','nonfinite','complete'])
def test_terminal_clears_departure_permission(ground, terminal):
    c = departure_fixture(ground)
    if terminal == 'complete': c.wp_idx = len(c.waypoints)
    c.tick((0., 0.), 0., 0., 900. if terminal=='deadline' else 3.02,
        upright=.49 if terminal=='tilt' else 1.,
        velocity=(math.nan, 0.) if terminal=='nonfinite' else (0., 0.))
    assert c.recovery_permission is None


@pytest.mark.parametrize('defect', ['unknown','hazard','steep','stale','no_go'])
def test_departure_does_not_relax_physical_checks(ground, defect):
    c = departure_fixture(ground)
    cell = c.grid.world_to_cell(0., 0.)
    if defect == 'unknown': c.grid.known[cell] = False
    if defect == 'hazard': c.grid.hazard[cell] = True
    if defect == 'steep': c.grid.slope[cell] = math.tan(math.radians(36.))
    if defect == 'stale': c.grid.fresh[cell] = False
    if defect == 'no_go':
        cell = c.grid.world_to_cell(-.15, 0.)
        c.grid.no_go[cell] = True
        c.grid.no_go_depth[cell] = 1.
    assert c.tick((0., 0.), 0., .1, 3.02, velocity=(.1, 0.)) == r.Motion()
    assert c.state == 'HOLD'


def test_productive_recovery_continues_beyond_eight_seconds(ground):
    c = departure_fixture(ground)
    frozen = c.recovery_endpoint
    for t in range(4, 16):
        assert c.tick((-.06*(t-3), 0.), 0., .06, float(t), velocity=(-.06, 0.)) == r.Motion(-.4, 0.)
        assert c.recovery_endpoint == frozen and c.recovery_episode == 3.
    c.tick((-.78, 0.), 0., .06, 16., velocity=(-.06, 0.))
    assert c.button_state == 'OBSERVE'
    assert next(e for e in c.attempt_events if e['event']=='remote_maneuver_end')['result']=='endpoint reached'


def test_recovery_watchdog_uses_new_best_progress_not_oscillation(ground):
    c = departure_fixture(ground)
    for t, x in ((5., -.12), (6., 0.), (9., -.12), (12., 0.)):
        c.tick((x, 0.), 0., 0., t)
        assert c.button_state == 'RECOVER'
    c.tick((-.12, 0.), 0., 0., 13.)
    assert c.button_state == 'OBSERVE' and c.recovery_progress_at == 5.
    assert c.recovery_episode == 3.


def test_recovery_overshoot_outside_disk_is_not_endpoint_success(ground):
    c = departure_fixture(ground)
    c.tick((-1.01, .4), 0., 0., 4.)
    event = next(e for e in c.attempt_events if e['event']=='remote_maneuver_end')
    assert event['result']=='endpoint passed outside arrival disk'
    assert not event['onward_proven'] and c.button_state=='OBSERVE'


@pytest.mark.parametrize('state,emergency', [('RECOVER',False),('OBSERVE',False),('DRIVE',False),('RECOVER',True)])
def test_recovery_episode_deadline_binds_every_state(ground, state, emergency):
    c = departure_fixture(ground)
    c.button_state, c.emergency = state, emergency
    c.recovery_progress_at = 122.99
    assert c.tick((0., 0.), 0., .1, 123., velocity=(.1, 0.)) == r.Motion()
    assert c.state=='HOLD' and c.reason=='recovery episode deadline'
    assert c.recovery_permission is None


def test_recovery_prefers_supported_staging_over_short_steep_escape(ground):
    c = controller(ground)
    c.tick((0., 0.), 0., 0., 3.)
    c.path=[];c.target=None
    def physical(start,end,yaw,**kw):
        return 'forward unavailable' if end[0]>0 else ''
    with patch.object(c,'_physical_sweep',side_effect=physical), \
         patch.object(c.grid,'max_support_slope',side_effect=lambda xy, **kwargs: 10. if xy[0] <= -2.9 else 25.):
        m=c._recover('need staging')
    assert m.vx==-.4 and c.recovery_endpoint==(-3.,0.)
    assert c.recovery_kind=='gentle staging'


def test_recovery_retains_checked_one_meter_fallback(ground):
    c = controller(ground);c.tick((0.,0.),0.,0.,3.)
    c.path=[];c.target=None
    with patch.object(c.grid,'max_support_slope',return_value=25.):
        c._recover('no staging')
    assert math.dist((0.,0.),c.recovery_endpoint)==1.
    assert c.recovery_kind=='checked egress'


def test_unobserved_longer_staging_cannot_be_selected(ground):
    c = controller(ground);c.tick((0.,0.),0.,0.,3.)
    c.path=[];c.target=None
    def physical(start,end,yaw,**kw):
        return 'unknown long corridor' if abs(end[0])>1.1 else ''
    with patch.object(c,'_physical_sweep',side_effect=physical), \
         patch.object(c.grid,'max_support_slope',side_effect=lambda xy, **kwargs: 10. if abs(xy[0])>1.1 else 25.):
        c._recover('long corridor unavailable')
    assert math.dist((0.,0.),c.recovery_endpoint)==1.


def test_recorded_productive_retreat_does_not_release_at_eight_seconds(ground):
    # Decision replay only: these poses were generated before the new target.
    p=Path(__file__).resolve().parents[1]/'outputs/remote_autonomy/departure-replacement-9wz8gwrt/run.csv'
    rows=[q for q in csv.DictReader(p.open()) if 141.66<=float(q['sim_time'])<=149.66]
    c=controller(ground, (0.,13.))
    c.button_state='RECOVER';c.button_since=c.recovery_episode=141.66;c.direction=-1
    c.recovery_start=(float(rows[0]['x']),float(rows[0]['y']))
    c.recovery_endpoint=(-.315922919022085,2.81633385507275)
    c.recovery_best_remaining=c.recovery_credited_remaining=3.;c.recovery_progress_at=141.66
    with patch.object(c,'_geometry',return_value=''), patch.object(c,'_anticipation',return_value=''):
        for q in rows:
            m=c.tick((float(q['x']),float(q['y'])),float(q['yaw']),float(q['speed']),float(q['sim_time']),
                     upright=float(q['up_z']),velocity=(float(q['velocity_world_x']),float(q['velocity_world_y'])))
            assert m==r.Motion(-.4,0.)
    assert c.recovery_endpoint==(-.315922919022085,2.81633385507275)
    assert c.recovery_progress_at>148.


@pytest.mark.parametrize('sign', [-1, 1])
def test_anticipation_checks_requested_speed_and_preserves_yaw_coverage(ground, sign):
    c=controller(ground);c.obs=bt.observation(3.,velocity=(.1,0.),yaw_rate=-.1)
    with patch.object(c,'_physical_sweep',return_value='') as sweep:
        c._anticipation(r.Motion(sign*.4,.4))
    args,kw=sweep.call_args
    assert args[1]==pytest.approx((sign*.3,0.))
    assert kw['yaw_offsets']==pytest.approx((-.125,.2))
    assert not kw['commanded'] and not kw.get('directional_exit',False)


def test_anticipation_does_not_reinterpret_damping_or_zero(ground):
    c=controller(ground);c.obs=bt.observation(3.,velocity=(.6,0.));c.emergency=True
    with patch.object(c,'_physical_sweep') as sweep:
        assert c._anticipation(r.Motion(-.4,0.))==''
        assert c._anticipation(r.Motion())==''
        sweep.assert_not_called()


@pytest.mark.parametrize('replacement_blocked', [False,True])
def test_anticipated_conflict_replaces_once_without_new_failure_or_budget(ground,replacement_blocked):
    c=controller(ground);c.recovery_episode=2.;c.recovery_actions=2
    initial=c.grid.approach_blocked.copy()
    def forecast(cmd,**kwargs):return 'future obstacle' if cmd.vx>0 or replacement_blocked else ''
    def physical(start,end,yaw,**kwargs):return 'forward obstacle' if end[0]>start[0] else ''
    with patch.object(c,'_geometry',return_value=''), \
         patch.object(c,'_anticipation',side_effect=forecast), \
         patch.object(c,'_physical_sweep',side_effect=physical), \
         patch.object(c,'_recover',wraps=c._recover) as recovery:
        out=c.tick((0.,0.),0.,.1,3.,velocity=(.1,0.))
    assert recovery.call_count==1
    assert out==r.Motion() if replacement_blocked else out==r.Motion(-.4,0.)
    assert c.recovery_episode==2. and c.recovery_actions==3
    assert c.progress_time==3. and c.climb_failures==0
    np.testing.assert_array_equal(c.grid.approach_blocked,initial)


def test_actual_drift_conflict_never_tries_anticipatory_recovery(ground):
    c=controller(ground)
    with patch.object(c,'_actual_geometry',return_value='actual hazard'), \
         patch.object(c,'_recover') as recovery:
        assert c.tick((0.,0.),0.,.2,3.,velocity=(.2,0.))==r.Motion()
    recovery.assert_not_called();assert c.state=='HOLD'


def test_anticipation_without_checked_escape_holds(ground):
    c=controller(ground)
    with patch.object(c,'_geometry',return_value=''), \
         patch.object(c,'_anticipation',return_value='future obstacle'), \
         patch.object(c,'_physical_sweep',return_value='no corridor'):
        assert c.tick((0.,0.),0.,.1,3.,velocity=(.1,0.))==r.Motion()
    assert c.state=='HOLD' and c.climb_failures==0


@pytest.mark.parametrize('run,time,expected', [('anticipation-mission-m81v71r4',129.48,.4),('staging-mission-dfw2u2tg',187.30,-.4),('staging-mission-dfw2u2tg',187.60,0.)])
def test_recorded_forecast_on_original_survey(run,time,expected):
    # Decision replay on pre-mission survey: no future live geometry or physics replay.
    root=Path(__file__).resolve().parents[1]
    prior=importlib.import_module('rexmi_bt_tests.nav.terrain_prior')
    g=bt.a.AscentTerrainMap(world_size=64.,cell_size=.2)
    prior.load_prior(g,root/'outputs/ascent_full_survey/trial01/run.prior.npz',
        task='RexmiRl-Go2w-Crater-Bowl-RockySlope-Play-v0',seed=42)
    c=r.RemoteAscentController(g,[mission.Waypoint(0.,13.,1.,'exterior')],mode='characterize',uphill_limit=33.)
    rows=list(csv.DictReader((root/'outputs/remote_autonomy'/run/'run.csv').open()))
    index=next(i for i,q in enumerate(rows) if float(q['sim_time'])==time)
    previous,q=rows[index-1:index+1]
    xy=tuple(float(q[k]) for k in ('x','y'))
    c.path=[xy,(0.,13.)];c.target=(0.,13.);c.button_state='DRIVE'
    c.last_yaw=float(previous['yaw']);c.last_tick=float(previous['sim_time'])
    with patch.object(c,'_navigate',return_value=r.Motion(.4,0.)):
        out=c.tick(xy,float(q['yaw']),float(q['speed']),time,upright=float(q['up_z']),
            velocity=(float(q['velocity_world_x']),float(q['velocity_world_y'])))
    assert out==r.Motion(expected,0.)
    assert (c.state=='HOLD')==(expected==0.)
    assert c.climb_failures==0
    assert not c.grid.approach_blocked.any()


def test_physical_forecast_retains_directional_ban_without_incidental_permission(ground):
    c=controller(ground);c.obs=bt.observation(3.)
    c.grid.approach_blocked[:]=True
    assert 'excluded approach' in c._anticipation(r.Motion(.4,0.),escape=True)


def test_actual_powered_grade_still_rejected_even_if_stopping_reserve_is_clear(ground):
    c=brake_controller(ground);c.emergency=False
    with patch.object(c,'_navigate',return_value=r.Motion(-.4,0.)):
        out=c.tick((0.,0.),0.,0.,3.,velocity=(0.,0.))
    assert out.vx==.4 and c.button_state=='RECOVER'
    assert 'uphill grade' in c.reason


def retained_survey_controller(ground, **metadata):
    profile=dict(task='RexmiRl-Go2w-Crater-Bowl-RockySlope-Play-v0',version=1,
                 coordinate_frame='world',seed=42)
    profile.update(metadata)
    return survey_controller(ground,**profile)


def test_static_coverage_retains_current_changed_geometry_and_observation_age(ground):
    c=retained_survey_controller(ground);assert c.static_retention_enabled
    c.grid.fresh[:]=False;c.grid.surface[:]+=.09
    height=c.grid.surface.copy();fresh=c.grid.fresh.copy();age=c.grid.last_seen.copy()
    assert c.tick((0.,0.),0.,0.,3.).vx==.4
    np.testing.assert_array_equal(c.grid.surface,height)
    np.testing.assert_array_equal(c.grid.fresh,fresh)
    np.testing.assert_array_equal(c.grid.last_seen,age)
    assert any(e.get('support_mode')=='retained_static_coverage' for e in c.attempt_events)


@pytest.mark.parametrize('defect',['hazard','steep','unknown','nonfinite','no_go','direction','outside_prior'])
def test_retained_static_coverage_keeps_current_physical_rejections(ground,defect):
    c=retained_survey_controller(ground)
    cell=c.grid.world_to_cell(0.,0.)
    if defect=='outside_prior':
        c.prior_support=c.prior_support.copy();c.prior_support[cell]=False;c.prior_support.setflags(write=False)
    c.grid.fresh[:]=False;c.grid.surface[:]+=.09
    if defect=='hazard':c.grid.hazard[cell]=True
    if defect=='steep':c.grid.slope[cell]=math.tan(math.radians(36.))
    if defect=='unknown':c.grid.known[cell]=False
    if defect=='nonfinite':c.grid.surface[cell]=math.nan
    if defect=='no_go':c.grid.no_go[cell]=True
    if defect=='direction':c.grid.approach_blocked[:]=True
    assert c.tick((0.,0.),0.,0.,3.)==r.Motion()
    assert c.state=='HOLD'


@pytest.mark.parametrize('metadata',[{'task':'other'},{'version':2},{'coordinate_frame':'local'},
    {'seed':None},{'parent_geometry_id':'new'}])
def test_static_retention_restricted_to_original_matching_static_task(ground,metadata):
    c=retained_survey_controller(ground,**metadata)
    assert not c.static_retention_enabled
    c.grid.fresh[:]=False;c.grid.surface[:]+=.09
    assert c.tick((0.,0.),0.,0.,3.)==r.Motion()


def test_recorded_static_coverage_stop_checks_current_map_without_refreshing_it():
    root=Path(__file__).resolve().parents[1]
    p=root/'outputs/remote_autonomy/physical-forecast-mission-ey6hdud9'
    prior=importlib.import_module('rexmi_bt_tests.nav.terrain_prior')
    g=bt.a.AscentTerrainMap(world_size=64.,cell_size=.2)
    prior.load_prior(g,root/'outputs/ascent_full_survey/trial01/run.prior.npz',
        task='RexmiRl-Go2w-Crater-Bowl-RockySlope-Play-v0',seed=42)
    c=r.RemoteAscentController(g,[mission.Waypoint(0.,0.,1.,'center')],mode='characterize',uphill_limit=33.)
    with np.load(p/'run.map.npz') as saved:
        for src,dst in [('bin_low','_bin_low'),('bin_high','_bin_high'),('bin_count','_bin_count'),
                        ('last_seen','last_seen'),('samples','samples'),('height','height'),('low','low')]:
            getattr(g,dst)[:]=saved[src]
        g.rebuild(float(saved['sim_time']))
        for name in ('approach_blocked','no_go','failures','successes'):getattr(g,name)[:]=saved[name]
    rows=list(csv.DictReader((p/'run.csv').open()));previous,q=rows[-2:]
    xy=tuple(float(q[k]) for k in ('x','y'));c.last_yaw=float(previous['yaw']);c.last_tick=float(previous['sim_time'])
    c.path=[xy,(0.,0.)];c.target=(0.,0.);c.button_state='DRIVE'
    before=g.surface.copy();ages=g.last_seen.copy();fresh=g.fresh.copy()
    with patch.object(c,'_navigate',return_value=r.Motion(.4,0.)):
        out=c.tick(xy,float(q['yaw']),float(q['speed']),float(q['sim_time']),upright=float(q['up_z']),
            velocity=(float(q['velocity_world_x']),float(q['velocity_world_y'])))
    assert out==r.Motion(.4,0.) and c.state!='HOLD'
    np.testing.assert_array_equal(g.surface,before)
    np.testing.assert_array_equal(g.last_seen,ages)
    np.testing.assert_array_equal(g.fresh,fresh)


def test_retained_static_coverage_rejects_new_drop_in_rebuilt_geometry(ground):
    c=retained_survey_controller(ground)
    row=c.grid.world_to_cell(.2,0.)[0]
    for name in ('height','low','_bin_low','_bin_high'):
        getattr(c.grid,name)[row:]-=1.
    c.grid.rebuild(20.)
    assert c.grid.hazard[row-1:row+1].any()
    assert c.tick((0.,0.),0.,0.,20.)==r.Motion()
    assert c.state=='HOLD'


@pytest.mark.parametrize('sign',[-1,1])
def test_opposing_brake_skips_fictitious_direction_but_checks_actual_motion(ground,sign):
    c=controller(ground);c.emergency=True
    c.grid.mark_failure('failed',(0.,0.),0. if sign>0 else math.pi,2.,'stall')
    # Only the forbidden commanded direction is blocked; actual travel is opposite.
    c.obs=bt.observation(3.,velocity=(-sign*.6,0.))
    assert c._geometry(r.Motion(sign*.4,0.))==''
    for velocity,omega,emergency in [((sign*.1,0.),0.,True),((0.,0.),0.,True),
                                   ((0.,.1),0.,True),((-sign*.6,0.),.1,True),
                                   ((-sign*.6,0.),0.,False)]:
        c.obs=bt.observation(3.,velocity=velocity);c.emergency=emergency
        assert 'excluded approach' in c._geometry(r.Motion(sign*.4,omega))


def test_opposing_damping_cannot_cross_actual_excluded_motion(ground):
    c=controller(ground);c.emergency=True
    c.grid.mark_failure('failed',(0.,0.),math.pi,2.,'stall')
    c.obs=bt.observation(3.,velocity=(-.6,0.))
    assert 'excluded approach' in c._geometry(r.Motion(.4,0.))


def test_recorded_reverse_braking_preserves_full_world_motion_checks():
    import json
    root=Path(__file__).resolve().parents[1]
    p=root/'outputs/remote_autonomy/static-coverage-mission-185chj1a'
    saved=np.load(p/'run.map.npz')
    g=bt.a.AscentTerrainMap(world_size=saved['height'].shape[0]*float(saved['cell_size']),
        cell_size=float(saved['cell_size']),origin=tuple(saved['origin']),
        footprint_radius=float(saved['footprint_radius']),half_length=float(saved['half_length']),half_width=float(saved['half_width']))
    prior=importlib.import_module('rexmi_bt_tests.nav.terrain_prior')
    prior.load_prior(g,root/'outputs/ascent_full_survey/trial01/run.prior.npz',
        task='RexmiRl-Go2w-Crater-Bowl-RockySlope-Play-v0',seed=42)
    c=r.RemoteAscentController(g,[mission.Waypoint(0.,13.,.6,'exterior')],mode='characterize',uphill_limit=33.)
    for src,dst in [('bin_low','_bin_low'),('bin_high','_bin_high'),('bin_count','_bin_count'),
                    ('last_seen','last_seen'),('samples','samples'),('height','height'),('low','low')]:getattr(g,dst)[:]=saved[src]
    g.rebuild(float(saved['sim_time']))
    for name in ('approach_blocked','no_go','failures','successes'):getattr(g,name)[:]=saved[name]
    g.failure_records=json.loads((p/'run.summary.json').read_text())['failed_climbs']
    failure=g.failure_records[-1]
    c.recovery_permission=r.RecoveryPermission.capture(g,failure['position'],failure['time'])
    rows=list(csv.DictReader((p/'run.csv').open()))
    for q in rows:
        if float(q['sim_time'])>=failure['time']:
            c.recovery_permission=c.recovery_permission.observe(g,(float(q['x']),float(q['y'])))
    assert len(c.recovery_permission.patches)==3
    previous,q=rows[-2:];xy=tuple(float(q[k]) for k in ('x','y'))
    c.last_yaw=float(previous['yaw']);c.last_tick=float(previous['sim_time'])
    c.button_state='RECOVER';c.recovery_episode=failure['time'];c.direction=-1
    out=c.tick(xy,float(q['yaw']),float(q['speed']),float(q['sim_time']),upright=float(q['up_z']),
        velocity=(float(q['velocity_world_x']),float(q['velocity_world_y'])))
    assert out==r.Motion(.4,0.) and c.emergency and c.state!='HOLD'
    # Lateral motion is substantial; its physical fan cannot be omitted.
    c.recovery_permission=None
    assert c._geometry(out,escape=True)==''
    sf=importlib.import_module('rexmi_bt_tests.nav.swept_footprint')
    drift=tuple(xy[i]+c.obs.velocity[i]*(.25+.5*c.obs.speed/.4) for i in (0,1))
    cell=g.world_to_cell(*drift);g.hazard[cell]=True
    assert 'obstacle' in c._geometry(out,escape=True)


def constrained_climb(ground):
    c=controller(ground);c.grid.gx[:]=math.tan(math.radians(34.));c.grid.gy[:]=0.
    c.grid.slope[:]=math.tan(math.radians(34.));c.grid.uphill_limit_deg=33.
    c.drive_exposure=6.;c.command=r.Motion(.4,0.);c.output_owner='DRIVE'
    return c


def test_powered_grade_release_preserves_progress_budget_and_does_not_repower(ground):
    c=constrained_climb(ground);c.low_progress_active=4.;anchor=c.progress_anchor
    assert c.tick((0.,0.),0.,.1,3.,velocity=(.1,0.))==r.Motion()
    assert c.button_state=='RELEASE' and c.reason=='powered grade permission ended; policy-active release'
    assert c.progress_time==3. and c.progress_anchor==anchor and c.low_progress_active==4.
    for t in np.arange(3.02,5.,.02):
        assert c.tick((.06*(t-3.),0.),0.,.06,t,velocity=(.06,0.))==r.Motion()
    assert c.button_state=='RELEASE' and c.climb_failures==0
    assert not c.grid.approach_blocked.any()


@pytest.mark.parametrize('case',['cold','short_power','reverse','recovery'])
def test_grade_release_requires_owned_established_forward_climb(ground,case):
    c=constrained_climb(ground);proposal=r.Motion(.4,0.)
    if case=='cold':c.output_owner='OBSERVE'
    if case=='short_power':c.drive_exposure=4.99
    if case=='reverse':proposal=r.Motion(-.4,0.);c.grid.gx[:]*=-1
    if case=='recovery':c.button_state='RECOVER';c.output_owner='RECOVER'
    with patch.object(c,'_navigate',return_value=proposal),patch.object(c,'_recover',wraps=c._recover) as recovery:
        c.tick((0.,0.),0.,0.,3.)
    recovery.assert_called_once();assert c.button_state!='RELEASE'


def test_grade_conflict_with_proposed_physical_obstacle_uses_recovery_not_release(ground):
    c=constrained_climb(ground)
    original=c._command_geometry
    def physical(cmd,**kwargs):
        if not kwargs.get('check_grade',True) and cmd.vx>0:return 'future obstacle'
        return original(cmd,**kwargs)
    with patch.object(c,'_command_geometry',side_effect=physical),patch.object(c,'_recover',wraps=c._recover) as recovery:
        c.tick((0.,0.),0.,.1,3.,velocity=(.1,0.))
    recovery.assert_called_once();assert c.button_state!='RELEASE'


def test_grade_release_replacement_is_independently_checked(ground):
    c=constrained_climb(ground);original=c._actual_geometry
    def actual(cmd,**kwargs):return 'zero drift forbidden' if cmd==r.Motion() else original(cmd,**kwargs)
    with patch.object(c,'_actual_geometry',side_effect=actual):
        assert c.tick((0.,0.),0.,.1,3.,velocity=(.1,0.))==r.Motion()
    assert c.state=='HOLD' and c.reason=='physical envelope: zero drift forbidden'


def test_grade_conflict_never_overrides_actual_hazard_or_emergency(ground):
    for emergency in (False,True):
        c=constrained_climb(ground);c.emergency=emergency
        with patch.object(c,'_actual_geometry',return_value='actual obstacle'),patch.object(c,'_recover') as recovery:
            assert c.tick((0.,0.),0.,.1,3.,velocity=(.1,0.))==r.Motion()
        recovery.assert_not_called();assert c.state=='HOLD'


def test_recorded_grade_stop_selects_independently_clear_policy_release():
    root=Path(__file__).resolve().parents[1];p=root/'outputs/remote_autonomy/opposing-damping-mission-wzodlne7'
    saved=np.load(p/'run.map.npz')
    g=bt.a.AscentTerrainMap(world_size=saved['height'].shape[0]*float(saved['cell_size']),
        cell_size=float(saved['cell_size']),origin=tuple(saved['origin']),
        footprint_radius=float(saved['footprint_radius']),half_length=float(saved['half_length']),half_width=float(saved['half_width']))
    prior=importlib.import_module('rexmi_bt_tests.nav.terrain_prior')
    prior.load_prior(g,root/'outputs/ascent_full_survey/trial01/run.prior.npz',
        task='RexmiRl-Go2w-Crater-Bowl-RockySlope-Play-v0',seed=42)
    c=r.RemoteAscentController(g,[mission.Waypoint(0.,13.,.6,'exterior')],mode='characterize',uphill_limit=33.)
    for src,dst in [('bin_low','_bin_low'),('bin_high','_bin_high'),('bin_count','_bin_count'),
                    ('last_seen','last_seen'),('samples','samples'),('height','height'),('low','low')]:getattr(g,dst)[:]=saved[src]
    g.rebuild(float(saved['sim_time']))
    for name in ('approach_blocked','no_go','failures','successes'):getattr(g,name)[:]=saved[name]
    previous,q=list(csv.DictReader((p/'run.csv').open()))[-2:]
    xy=tuple(float(q[k]) for k in ('x','y'));c.last_yaw=float(previous['yaw']);c.last_tick=float(previous['sim_time'])
    c.path=[xy,(0.,13.)];c.target=(0.,13.);c.button_state='DRIVE';c.output_owner='DRIVE'
    c.drive_exposure=13.04;c.command=r.Motion(.4,0.);c.progress_anchor=xy;c.progress_time=145.
    with patch.object(c,'_navigate',return_value=r.Motion(.4,0.)):
        out=c.tick(xy,float(q['yaw']),float(q['speed']),float(q['sim_time']),upright=float(q['up_z']),
            velocity=(float(q['velocity_world_x']),float(q['velocity_world_y'])))
    assert out==r.Motion() and c.button_state=='RELEASE' and c.state!='HOLD'
    assert c.progress_time==145. and c.climb_failures==0 and not c.grid.approach_blocked.any()
