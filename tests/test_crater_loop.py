"""Mission handoff, retained-map planning and unchanged ascent execution."""
import copy
from concurrent.futures import Future
from dataclasses import asdict
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import test_remote_control as fixture
from rexmi_bt_tests.nav import crater_loop as loop, slope_cost
from rexmi_bt_tests.nav.checkpoint_control import CheckpointController, Motion
from rexmi_bt_tests.nav.experimental_path_control import ExperimentalPathController
from rexmi_bt_tests.nav.mission import Waypoint


@pytest.fixture
def ground():return fixture.bt.ground()


def test_plan_uses_actual_arrival_and_only_observed_connected_exterior(ground,tmp_path):
    original=ground.surface.copy()
    plan=loop.plan_exit(ground,(.2,.1),1.2,12.,2.,tmp_path/'exit')
    assert plan.route[0]==(.2,.1)
    assert math.dist(plan.route[-1],ground.origin)>2.
    assert plan.metadata['map_source']=='live sensors retained during this descent'
    assert plan.metadata['parameters']==asdict(slope_cost.Parameters())
    assert Path(plan.metadata['result_path']).is_file()
    assert not loop.exit_region.reason(ground,plan.route[-1],2.)
    np.testing.assert_array_equal(ground.surface,original)


def test_missing_exterior_is_not_replaced_with_historical_map(ground,tmp_path):
    with pytest.raises(ValueError,match='no connected observed'):
        loop.plan_exit(ground,(0,0),0.,12.,100.,tmp_path/'missing')


def completed_descent(ground):
    c=CheckpointController(ground,[Waypoint(0,0,.8,'center')])
    c.state='COMPLETE';c.wp_idx=1
    return c


def test_center_settle_and_handoff_preserve_same_live_map_and_policy_interface(ground,tmp_path):
    c=completed_descent(ground)
    c.escape_events.append(dict(event='descent_turn_stall',time=4.))
    session=loop.CraterLoop(ground,tmp_path,2.)
    try:
        c,cmd=session.tick(c,(0,0),0.,0.,10.,1.,(0,0))
        assert session.phase=='SETTLE_AT_CENTER'
        assert session.descent['steering_recoveries']==1
        assert session.descent['recovery_events'] is not c.escape_events
        for t in np.arange(10.02,10.64,.02):
            c,cmd=session.tick(c,(0,0),0.,.3,float(t),1.,(.3,0))
        assert session.future is None
        for t in np.arange(10.64,11.30,.02):
            c,cmd=session.tick(c,(0,0),0.,0.,float(t),1.,(0,0))
        assert session.future is not None
        plan=session.future.result(timeout=30)
        c,cmd=session.tick(c,(0,0),0.,0.,11.32,1.,(0,0))
        assert isinstance(c,ExperimentalPathController)
        assert c.grid is ground and session.phase=='ASCEND_TO_EXTERIOR'
        assert c.preference is plan.model
        assert c.selected_exit==tuple(plan.route[-1])
        assert session.center_at is not None
        assert cmd.vx in (0,.4) and abs(cmd.omega)<=.35
    finally:session.close()


def test_operator_stop_blocks_ready_plan_and_manual_is_not_autonomous_completion(ground,tmp_path):
    c=completed_descent(ground)
    session=loop.CraterLoop(ground,tmp_path,2.)
    try:
        plan=loop.plan_exit(copy.deepcopy(ground),(0,0),0.,10.,2.,tmp_path/'plan')
        session.phase='PLAN_ASCENT';session.center_at=10.
        session.future=Future();session.future.set_result(plan)
        session.request('stop');c.request('stop',11.)
        unchanged,cmd=session.tick(c,(0,0),0.,0.,11.,1.,(0,0))
        assert unchanged is c and session.plan is None and cmd==Motion()
        session.request('resume');c.request('resume',11.02)
        active,_=session.tick(c,(0,0),0.,0.,11.02,1.,(0,0))
        assert active is not c and active.intervened
    finally:session.close()


def test_drift_during_planning_replans_from_measured_pose(ground,tmp_path):
    c=completed_descent(ground)
    session=loop.CraterLoop(ground,tmp_path,2.)
    try:
        plan=loop.plan_exit(copy.deepcopy(ground),(0,0),0.,10.,2.,tmp_path/'old')
        session.phase='PLAN_ASCENT';session.center_at=10.
        session.future=Future();session.future.set_result(plan)
        session._start_plan=Mock()
        active,cmd=session.tick(c,(.7,0),0.,0.,11.,1.,(0,0))
        assert active is c and cmd==Motion()
        session._start_plan.assert_called_once_with(c,(.7,0),0.,11.)
    finally:session.close()


def test_deadline_cannot_restart_at_ascent_handoff(ground,tmp_path):
    c=completed_descent(ground);session=loop.CraterLoop(ground,tmp_path,2.)
    try:
        _,cmd=session.tick(c,(0,0),0.,0.,900.,1.,(0,0))
        assert cmd==Motion() and session.phase=='FAILED'
    finally:session.close()


def test_center_drift_reopens_descent_destination(ground,tmp_path):
    c=completed_descent(ground);session=loop.CraterLoop(ground,tmp_path,2.)
    try:
        session.phase='SETTLE_AT_CENTER'
        c,cmd=session.tick(c,(1.,0.),0.,.2,10.,1.,(.2,0))
        assert session.phase=='DESCEND_TO_CENTER' and c.wp_idx==0
        assert c.state=='OBSERVE' and not c.intervened
    finally:session.close()


def test_descent_turn_is_executed_despite_predictive_clearance_veto(ground):
    c=loop.DescentController(ground,[Waypoint(0,0,.8,'center')])
    ground.support_reason=Mock(return_value='slope > 35 degrees')
    assert c.motion_clear((0,0),0.,0.,Motion(0.,.35))
    assert not c.prepare_edge_exit((0,0),0.,2.)
    c.request('stop',2.)
    assert c.tick((0,0),0.,0.,2.02)==Motion()


def loop_args():
    from test_round_trip import args
    result=args()
    result.terrain_prior=None
    result.ascent_path_result=None
    result.ascent_event_snapshots=None
    return result


def test_preset_owns_map_and_handoff_but_preserves_explicit_trial_budget():
    args=loop_args()
    loop.configure(args,['--max_steps=100'])
    assert args.map_retention=='full' and args.max_steps==100
    assert args.nav_controller=='checkpoint' and args.ascent_unguarded
    assert (args.goal_x,args.goal_y,args.goal_radius)==(0,0,.8)


@pytest.mark.parametrize('field,value',[
    ('terrain_prior','previous-run.npz'),('simulator_survey',True),
    ('checkpoint_replay','clicks.json'),('ascent_path_result','blue-path.json'),
])
def test_preset_rejects_historical_inputs_and_competing_destinations(field,value):
    args=loop_args();setattr(args,field,value)
    with pytest.raises(ValueError):loop.configure(args,[])


def test_no_selected_route_reports_failure_instead_of_crashing(ground):
    c=loop.DescentController(ground,[Waypoint(0,0,.8,'center')])
    c.route_options=[dict(profile='balanced',selected=False),dict(profile='low-grade',selected=False)]
    c.choose_route=Mock(return_value=([],False))
    assert not c._plan((0,0),0.,2.)


def test_stalled_alignment_reverses_but_heading_progress_renews_patience(ground,monkeypatch):
    c=loop.DescentController(ground,[Waypoint(0,0,.8,'center')])
    c.state='FOLLOW';c.heading_error=2.
    monkeypatch.setattr(CheckpointController,'tick',lambda *args:Motion(0.,.35))
    c.local_maneuver=Mock(return_value=(Motion(.2,.35),(.4,.1)))
    for t in (1.,3.,5.):assert c.tick((0,0),0.,0.,t)==Motion(0.,.35)
    c.heading_error=1.9
    c.tick((0,0),.1,0.,5.9)
    c.tick((0,0),.1,0.,8.)
    assert not c.local_maneuver.called
    result=c.tick((0,0),.1,0.,11.)
    assert result==Motion(-.4,0.) and c.state=='BACKOUT'
    assert c.escape_events[-1]['event']=='descent_reverse_start'


def test_checkpoint_has_observed_body_and_extra_data_reserve(ground):
    c=loop.DescentController(ground,[Waypoint(4.8,0,.8,'unknown edge')])
    assert c._plan((0,0),0.,2.)
    cell=ground.world_to_cell(*c.checkpoint)
    required=ground.footprint_radius+ground.clearance+ground.cell_size*math.sqrt(.5)+c.checkpoint_reserve[cell]
    assert c.observed_distance[cell]>=required
    assert c.checkpoint[0]<4.8 and not c.is_mission_checkpoint
    for p in c.path:assert c.descent_allowed[ground.world_to_cell(*p)]
    from test_round_trip import runner
    assert runner.route_comparison_key(c)


def test_missing_checkpoint_data_waits_instead_of_pushing_through(ground):
    c=loop.DescentController(ground,[Waypoint(4.8,0,.8,'goal')])
    ground.known[:]=False;ground.known[25:35,25:35]=True;ground.revision+=1
    assert not c._plan((0,0),0.,2.)
    assert c.waiting_for_map
    c.hold(c.planning_failure,2.)
    assert c.state=='DATA_WAIT'
    assert c.tick((0,0),0.,0.,2.1)==Motion()
    assert c.tick((0,0),0.,0.,12.1)==Motion() and c.state=='HOLD'


def test_reverse_does_not_enter_unobserved_space_and_stop_cancels_recovery(ground):
    c=loop.DescentController(ground,[Waypoint(3,0,.8,'goal')])
    assert c._start_backout((0,0),0.,2.,'stalled')==Motion(-.4,0)
    assert c.tick((0,0),0.,0.,4.1)==Motion(-.7,0)
    assert c.tick((0,0),0.,0.,6.2)==Motion(-1.,0)
    c.request('stop',6.21)
    assert c.tick((0,0),0.,0.,6.22)==Motion()
    ground.known[:30]=False
    assert not loop.descent_planning.backout_observed(ground,(0,0),0.)


def test_descent_route_spreads_steep_face_over_longer_distance(ground):
    rr,cc=np.indices(ground.surface.shape)
    xx=(rr+.5)*ground.cell_size-ground.world_size/2
    yy=(cc+.5)*ground.cell_size-ground.world_size/2
    # A short face hidden by the broad display-plane slope, as in the incident.
    ground.surface[:]=.8*np.exp(-(xx/.5)**2-(yy/.7)**2)
    ground.revision+=1
    c=loop.DescentController(ground,[Waypoint(2,0,.8,'goal')])
    route,done=c.choose_route((-2,0),(2,0),0.)
    assert done and max(abs(p[1]) for p in route)>.6
    assert c.route_score(route,0.)<c.route_score([(-2,0),(2,0)],0.)


def test_plane_extrapolation_does_not_extend_measured_data_frontier(ground):
    ground.height[40:]=np.nan
    ground.revision+=1
    c=loop.DescentController(ground,[Waypoint(4,0,.8,'goal')])
    loop.descent_planning.refresh(c)
    assert ground.known[41,30]  # The old visual/interpolated map says known.
    assert c.observed_distance[41,30]==0
    assert not c.checkpoint_supported[39,30]


def test_frontier_step_approaches_edge_toward_goal_without_rim_detour(ground):
    c=loop.DescentController(ground,[Waypoint(4.8,0,.8,'goal')])
    route,done=c.choose_route((3.5,0),(4.8,0),0.)
    assert route and not done
    assert .55<=math.dist(route[-1],(3.5,0))<=2.
    assert abs(route[-1][1])<=.2 and route[-1][0]>4.
    assert math.dist(route[-1],(4.8,0))<math.dist((3.5,0),(4.8,0))
    assert loop.descent_planning.recoverable_frontier(c,route)
    # When no further inward step is observed, don't start a lateral tour.
    route,done=c.choose_route((4.2,0),(4.8,0),0.)
    assert not route and c.waiting_for_map


@pytest.mark.parametrize('degrees,expected',[(0,True),(15,True),(25,False)])
def test_frontier_requires_gentle_two_way_approach(degrees,expected):
    g=fixture.bt.ground(degrees)
    c=loop.DescentController(g,[Waypoint(4.8,0,.8,'goal')])
    route,done=c.choose_route((3.5,0),(4.8,0),0.)
    assert bool(route)==expected and not done
    if route:assert loop.descent_planning.recoverable_frontier(c,route)


def test_live_map_growth_rolls_checkpoint_forward_without_stop_or_angle_hunt(ground):
    measured=ground.height.copy()
    ground.height[38:]=np.nan;ground.revision+=1
    c=loop.DescentController(ground,[Waypoint(4.8,0,.8,'goal')])
    assert c._plan((0,0),0.,2.)
    old=c.checkpoint
    assert .55<=old[0]<=1.1
    ground.height[:44]=measured[:44];ground.revision+=1
    xy=(old[0]-.3,old[1])
    command=c.tick(xy,0.,.2,2.02,velocity=(.2,0.))
    assert c.checkpoint[0]>old[0]+.4
    assert c.state=='FOLLOW' and command.vx>0
    assert not c.pending_arrival


def test_stale_sideways_viewpoint_does_not_override_goalward_step(ground):
    c=loop.DescentController(ground,[Waypoint(4.8,0,.8,'goal')])
    c.observation_target=(3.5,2.)
    route,done=c.choose_route((3.5,0),(4.8,0),0.)
    assert route[-1][0]>4. and abs(route[-1][1])<=.2
