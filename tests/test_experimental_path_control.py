"""Explicit unguarded simulation contract and response-based recovery."""
import copy
import math
from unittest.mock import patch

import numpy as np
import pytest
import test_remote_control as fixture
from test_saved_path_control import ground
from rexmi_bt_tests.nav import experimental_path_control as e, saved_path_control as s


def control(ground,route=((0.,0.),(3.,0.))):
    g=copy.deepcopy(ground)
    with patch.object(s,'load_route_result',return_value=(np.asarray(route),dict(map_path='fixture',parameters={}))), \
         patch.object(e.planning_snapshot,'load',return_value=(copy.deepcopy(g),{})):
        return e.ExperimentalPathController(g,[fixture.mission.Waypoint(*route[-1],.4,'exit')],
            mode='characterize',goal_mode='any_exit',exit_rim_radius=1.,saved_path_result='fixture')


def test_no_terrain_motion_posture_speed_or_cross_track_veto(ground):
    c=control(ground)
    c.grid.hazard[:]=True;c.grid.no_go[:]=True;c.grid.fresh[:]=False
    c.grid.slope[:]=math.tan(math.radians(60.))
    with patch.object(c,'_geometry',side_effect=AssertionError('must not call geometry')):
        assert c.tick((0.,0.),0.,1.5,3.,upright=.2)==e.Motion(.4,0.)
        c.tick((0.,1.5),0.,1.5,3.02,upright=.2)
    assert c.state!='HOLD' and not c.emergency and c.tracking_error>1.


def test_turn_continues_with_productive_angular_response(ground):
    c=control(ground,((0.,0.),(0.,3.)))
    for now in np.arange(3.,12.,.02):
        m=c.tick((0.,0.),(now-3.)*.1,0.,float(now))
        assert m==e.Motion(0.,.35)
        assert c.recovery is None


def test_stall_tries_heading_then_push_then_short_reverse(ground):
    c=control(ground)
    c.tick((0.,0.),0.,0.,3.)
    c.tick((0.,0.),0.,0.,11.1)
    # Heading aligned on flat test map: adjustment completes before push.
    assert c.recovery['action']=='PUSH' and c.command.vx==.4
    c.tick((0.,0.),0.,0.,16.2)
    assert c.recovery['action']=='REVERSE' and c.command.vx==-.4
    c.tick((-.51,0.),0.,.2,17.)
    assert c.recovery['action'] in ('TURN','PUSH')
    ended=[r for r in c.recovery_history if r['action']=='PUSH']
    assert ended and ended[0]['route_score_gain_m']==0.


def test_release_plateau_pushes_but_productive_release_continues(ground):
    c=control(ground)
    c.tick((0.,0.),0.,0.,3.);c.button_state='RELEASE'
    for now in np.arange(3.02,20.,.02):
        assert c.tick(((now-3.)*.06,0.),0.,.06,float(now))==e.Motion()
    xy=c.last_xy
    for now in np.arange(20.,29.,.02):c.tick(xy,0.,0.,float(now))
    assert c.recovery['action']=='PUSH' and c.command.vx==.4


def test_yaw_stall_selects_reverse_not_hold(ground):
    c=control(ground,((0.,0.),(0.,3.)))
    c.tick((0.,0.),0.,0.,3.)
    c.tick((0.,0.),0.,0.,8.1)
    assert c.recovery['action']=='REVERSE' and c.command.vx==-.4 and c.state!='HOLD'


def test_push_progress_returns_to_path_and_failed_heading_memory_is_finite(ground):
    c=control(ground);c.tick((0.,0.),0.,0.,3.)
    c._start_action('PUSH','fixture')
    assert c.tick((.35,0.),0.,.2,4.)==e.Motion(.4,0.)
    assert c.recovery is None and c.recovery_history[-1]['route_score_gain_m']>=.3
    c._start_action('PUSH','second fixture');c.tick((.35,0.),0.,0.,9.1)
    heading=c._recovery_heading()
    assert math.isfinite(heading)
    assert any(o['failed_pushes'] for o in c.attempt_events[-1]['candidates'])


def test_stop_manual_invalid_data_budget_and_observed_completion(ground):
    c=control(ground);c.tick((0.,0.),0.,0.,3.)
    c.request('stop',3.1);assert c.tick((0.,0.),0.,2.,3.2)==e.Motion()
    c.request('manual',3.3,vx=-.4,omega=.35)
    assert c.tick((0.,0.),0.,2.,3.4,upright=.2)==e.Motion(-.4,.35)
    c.request('resume',3.5);assert c.tick((0.,0.),0.,0.,3.6)==e.Motion(.4,0.)
    assert c.tick((0.,0.),0.,0.,900.)==e.Motion() and c.state=='HOLD'
    c=control(ground);assert c.tick((math.nan,0.),0.,0.,3.)==e.Motion()
    c=control(ground);c.tick((0.,0.),0.,0.,3.);c.saved_progress=3.
    c.grid.hazard[:]=True;c.grid.slope[:]=1.
    for now in np.arange(4.,4.7,.02):c.tick((3.,0.),0.,0.,float(now))
    assert c.state=='COMPLETE' and c.wp_idx==1


def test_progress_recovery_not_credited_by_reverse_or_button_changes(ground):
    c=control(ground);c.tick((0.,0.),0.,0.,3.)
    c.tick((1.,0.),0.,.2,5.)
    high=c.saved_progress
    c._start_action('REVERSE','fixture')
    c.tick((.5,0.),0.,.2,6.)
    assert c.saved_progress==high
    reverse=next(r for r in c.recovery_history if r['action']=='REVERSE')
    assert reverse['route_score_gain_m']<0


@pytest.mark.parametrize('ending',['stop','manual','deadline','finish','endpoint'])
def test_recovery_action_is_closed_when_ownership_changes(ground,ending):
    c=control(ground);c.tick((0.,0.),0.,0.,3.);c._start_action('PUSH','fixture')
    if ending in ('stop','manual'):c.request(ending,3.1)
    elif ending=='deadline':c.tick((0.,0.),0.,0.,900.)
    elif ending=='finish':c.finish_attempt('censored','simulation end',(0.,0.),0.,3.1)
    else:
        c.saved_progress=3.;c.tick((3.,0.),0.,0.,3.1)
    assert c.recovery is None
    assert c.recovery_history[-1]['censored']
    if ending in ('stop','manual'):
        c._recovery_heading()
        assert not any(o['failed_pushes'] for o in c.attempt_events[-1]['candidates'])
