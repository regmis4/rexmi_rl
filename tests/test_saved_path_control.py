"""Saved-route feedback and final command gates; no physics success claims."""
import copy
import hashlib
import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import test_remote_control as fixture
from rexmi_bt_tests.nav import saved_path_control as s


@pytest.fixture(scope='module')
def ground():
    g = fixture.bt.ground()
    g.terrain_prior = dict(kind='simulator_development', geometry_id='fixture', file_sha256='fixture',
        task='RexmiRl-Go2w-Crater-Bowl-RockySlope-Play-v0', version=1, seed=42, coordinate_frame='world')
    return g


def controller(ground, route=((0.,0.),(3.,0.))):
    with patch.object(s, 'load_route_result', return_value=(np.asarray(route), {})):
        return s.SavedPathController(copy.deepcopy(ground),
            [fixture.mission.Waypoint(*route[-1],.4,'exit')], mode='characterize',
            goal_mode='any_exit', exit_rim_radius=1., saved_path_result='fixture')


def test_turn_on_slope_has_no_minimum_stage_veto(ground):
    c = controller(ground, ((0.,0.),(0.,3.)))
    c.grid.slope[:] = math.tan(math.radians(25.))
    c.grid.max_support_slope = lambda *a, **kw: 25.
    assert c.tick((0.,0.),0.,0.,3.) == s.Motion(0.,.35)
    assert c.checked_turn is None and c.minimum_turn is None
    assert c.tick((0.,0.),math.pi/2-.05,0.,4.) == s.Motion(.4,0.)


def test_projection_does_not_jump_to_adjacent_future_branch(ground):
    c = controller(ground, ((0.,0.),(3.,0.),(3.,.2),(0.,.2)))
    distance,cross,index,_ = c._projection((0.,.2))
    assert distance == 0. and index == 0 and cross == pytest.approx(.2)
    c.saved_progress = 2.
    assert c._projection((1.4,0.))[0] == pytest.approx(1.4)


def test_stop_manual_resume_and_terminal_bounds(ground):
    c = controller(ground)
    c.tick((0.,0.),0.,0.,3.)
    c.request('stop',3.1)
    assert c.tick((0.,0.),0.,.2,3.2) == s.Motion()
    c.request('manual',3.3,vx=-.4,omega=.35)
    assert c.tick((0.,0.),0.,0.,3.4) == s.Motion(-.4,.35)
    c.request('resume',3.5)
    assert c.path == c.saved_route.tolist()
    assert c.mission_route == c.path and c.mission_route_complete
    assert c.route_profile['length_m'] == pytest.approx(3.)
    assert c.tick((0.,0.),0.,0.,3.6) == s.Motion(.4,0.)
    for kwargs in (dict(upright=.49), dict(speed=1.), dict(now=900.), dict(xy=(math.nan,0.))):
        c = controller(ground)
        args = dict(xy=(0.,0.),yaw=0.,speed=0.,now=3.)
        args.update(kwargs)
        assert c.tick(**args) == s.Motion()
        assert c.state == 'HOLD'


def test_mismatched_start_and_cross_track_stop_without_replan(ground):
    c = controller(ground)
    assert c.tick((1.,0.),0.,0.,3.) == s.Motion()
    assert 'spawn' in c.reason
    c = controller(ground)
    c.tick((0.,0.),0.,0.,3.)
    c.tick((0.,1.1),0.,0.,4.)
    assert c.state == 'HOLD' and '1 m' in c.reason


def test_shared_progress_deadline_survives_button_changes(ground):
    c = controller(ground)
    c.tick((0.,0.),0.,0.,3.)
    for now in np.arange(3.02,38.1,.02):
        c.button_state = 'DRIVE' if int(now)%2 else 'RELEASE'
        c.tick((0.,0.),0.,0.,float(now))
        if c.state == 'HOLD': break
    assert c.state == 'HOLD' and '35 s' in c.reason
    assert c.saved_progress_at == 3.


def test_productive_release_and_one_reinitiation_per_new_progress(ground):
    c = controller(ground)
    c.tick((0.,0.),0.,0.,3.)
    c.button_state = 'RELEASE'
    for now in np.arange(3.02,10.,.02):
        assert c.tick(((now-3.)*.06,0.),0.,.06,float(now)) == s.Motion()
    xy = c.last_xy
    for now in np.arange(10.,18.,.02):
        c.tick(xy,0.,0.,float(now))
    assert c.command.vx == .4
    assert c.saved_relaunch_credit is not None
    c.button_state = 'RELEASE'
    assert c.tick(xy,0.,0.,18.1) == s.Motion()


def test_combined_translated_yaw_is_checked(ground):
    c = controller(ground)
    c.tick((0.,0.),0.,0.,3.)
    c.obs = fixture.bt.observation(4., velocity=(.2,0.))
    calls = []
    def sweep(start,end,yaw,**kwargs):
        calls.append((start,end,kwargs))
        return 'combined collision' if end[0] > start[0] and kwargs['yaw_offsets'][1] > 0 else ''
    with patch.object(c,'_physical_sweep',side_effect=sweep):
        assert c._command_geometry(s.Motion(0.,.35)) == 'combined collision'
    assert calls[-1][1][0] == pytest.approx(.2)


def test_turn_preserves_full_sweep_cross_slope_limit(ground):
    c = controller(ground)
    c.tick((0.,0.),0.,0.,3.)
    c.grid.gy[:]=math.tan(math.radians(33.))
    c.grid.gx[:]=0.
    c.grid.side_limit_deg=32.
    assert 'side-slope' in c._command_geometry(s.Motion(0.,.35))


def test_final_gate_cannot_report_completion_and_capture_failure_still_stops(ground):
    c = controller(ground)
    c.tick((0.,0.),0.,0.,3.)
    c.saved_progress = 3.
    with patch.object(s.planning_snapshot,'save',side_effect=OSError('full disk')):
        with patch.object(c,'_geometry',return_value='obstacle'):
            c.tick((3.,0.),0.,0.,4.)
    assert c.state == 'HOLD' and c.wp_idx == 0
    assert next(e for e in c.attempt_events if e['event']=='saved_path_stop')['snapshot']['status']=='capture_failed'


def test_completion_needs_selected_endpoint_supported_exterior_and_settling(ground):
    c = controller(ground)
    c.tick((0.,0.),0.,0.,3.)
    c.saved_progress = 3.
    for now in np.arange(4.,4.7,.02):
        c.tick((3.,0.),0.,0.,float(now))
    assert c.wp_idx == 1 and c.state == 'COMPLETE'
    c = controller(ground)
    c.tick((0.,0.),0.,0.,3.)
    c.saved_progress = 3.
    with patch.object(s.exit_region,'reason',return_value='not gentle'):
        c.tick((3.,0.),0.,0.,4.)
    assert c.wp_idx == 0


def test_loader_validates_identity_map_hash_and_route_shape(ground,tmp_path):
    map_file = tmp_path/'map.npz'
    np.savez(map_file,metadata_json=json.dumps(dict(config=dict(terrain_prior=ground.terrain_prior))))
    result = dict(routes=dict(any_exit=[[0.,0.],[3.,0.]]),parameters={},metrics=dict(any_exit={}),
        inputs={str(map_file):hashlib.sha256(map_file.read_bytes()).hexdigest()})
    result_file = tmp_path/'result.json'
    result_file.write_text(json.dumps(result))
    route,meta = s.load_route_result(result_file,ground)
    assert route.shape == (2,2) and len(meta['result_sha256']) == 64
    other = copy.deepcopy(ground);other.terrain_prior['seed']=43
    with pytest.raises(ValueError,match='identity'):s.load_route_result(result_file,other)
    result['routes']['any_exit']=[[0.,0.],[float('nan'),0.]]
    result_file.write_text(json.dumps(result))
    with pytest.raises(ValueError,match='finite'):s.load_route_result(result_file,ground)
    result['routes']['any_exit']=[[0.,0.],[3.,0.]]
    result['inputs'][str(map_file)]='wrong'
    result_file.write_text(json.dumps(result))
    with pytest.raises(ValueError,match='hash'):s.load_route_result(result_file,ground)


def test_archived_input_relocates_to_clone_without_rewriting_provenance(tmp_path,monkeypatch):
    root=tmp_path/'clone'
    target=root/'logs/telemetry/manual/display_map.npz'
    target.parent.mkdir(parents=True);target.write_bytes(b'fixture')
    monkeypatch.setattr(s.planning_snapshot,'__file__',str(root/'source/rexmi_rl/nav/planning_snapshot.py'))
    original=Path('/original/machine/rexmi_rl/logs/telemetry/manual/display_map.npz')
    assert s.planning_snapshot.repository_input(original)==target
    assert s.planning_snapshot.repository_input(target)==target
    missing=Path('/original/machine/elsewhere/missing.npz')
    assert s.planning_snapshot.repository_input(missing)==missing
