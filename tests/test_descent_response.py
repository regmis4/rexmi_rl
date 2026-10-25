"""Focused regressions: data-edge checkpoints and descent reverse ownership."""
import math
import numpy as np
import test_remote_control as fixture
from rexmi_bt_tests.nav import crater_loop as loop, descent_planning
from rexmi_bt_tests.nav.descent_response import DescentReverseRecovery
from rexmi_bt_tests.nav.reverse_recovery import ReverseRecovery
from rexmi_bt_tests.nav.mission import Waypoint


def control():return loop.DescentController(fixture.bt.ground(),[Waypoint(4.8,0,.8,'goal')])


def test_checkpoint_requires_extra_space_beyond_measured_body_envelope():
    c=control();g=c.grid
    # Align the two reserves to distinct cells on this 20 cm fixture grid.
    g.footprint_radius=.4
    descent_planning.refresh(c)
    margin=g.footprint_radius+g.clearance+g.cell_size*math.sqrt(.5)
    assert descent_planning.CHECKPOINT_RESERVE_M==.25
    assert c.checkpoint_supported.any()
    assert np.all(c.observed_distance[c.checkpoint_supported]>=margin+.25)
    assert np.all(c.descent_allowed[c.checkpoint_supported])


def test_reverse_escalates_to_one_mps_and_does_not_quit_after_six_seconds():
    r=DescentReverseRecovery();r.start(0.,(0,0),0.)
    assert r.tick(2.01,(0,0))[0]==-.7
    assert r.tick(4.02,(0,0))[0]==-1.
    command,result=r.tick(6.1,(-.34,0))
    assert command==-1. and result is None
    command,result=r.tick(8.,(-.51,0))
    assert command==0 and result['success']


def test_ascent_reverse_keeps_its_original_six_second_window():
    r=ReverseRecovery();r.start(0.,(0,0),0.)
    _,result=r.tick(6.,(-.34,0))
    assert result is not None and not result['success']


def test_failed_reverse_cannot_resume_forward_navigation():
    c=control();c._start_backout((0,0),0.,1.,'fixture')
    for t in (3.1,5.2,7.3,19.1):c.tick((0,0),0.,0.,t,velocity=(0,0))
    assert c.state=='HOLD'
    assert not c.escape_events[-1]['success']


def test_data_wait_resists_drift_and_backs_out_before_losing_more_ground():
    c=control();c.waiting_for_map=True;c.wait_pose=((0,0),0.)
    c.hold('need observations',1.)
    command=c.tick((.1,0),0.,.2,1.1,velocity=(.2,0))
    assert c.state=='DATA_WAIT' and command.vx<0
    command=c.tick((.21,0),0.,.2,1.2,velocity=(.2,0))
    assert c.state=='BACKOUT' and command.vx<0


def test_operator_stop_cancels_descent_recovery():
    c=control();c._start_backout((0,0),0.,1.,'fixture')
    c.request('stop',2.)
    assert not c.reverse_recovery.active and c.state=='HOLD'


def test_operator_replay_only_accepts_ordered_stop_resume(tmp_path):
    import json,pytest
    from test_round_trip import runner
    path=tmp_path/'replay.json'
    path.write_text(json.dumps([dict(time=1.2,action='stop'),dict(time=10.66,action='resume')]))
    assert runner.load_operator_replay(path)==[(1.2,'stop'),(10.66,'resume')]
    path.write_text(json.dumps([dict(time=1.,action='manual')]))
    with pytest.raises(ValueError):runner.load_operator_replay(path)



def test_supported_current_cell_can_connect_to_reserved_route_without_opening_other_cells():
    from pathlib import Path
    from rexmi_bt_tests.nav import planning_snapshot
    g,meta=planning_snapshot.load(Path(__file__).resolve().parents[1]/
        'docs/validation/crater_descent_completion_2026-10-25/supported_start.npz')
    c=loop.DescentController(g,[Waypoint(0,0,.8,'center')])
    xy=meta['position'];cell=g.world_to_cell(*xy)
    descent_planning.refresh(c)
    assert c.descent_body_supported[cell] and not c.descent_route_support[cell]
    reserved=c.descent_route_support.copy()
    route,complete=c.choose_route(xy,(0,0),meta['yaw'])
    assert complete and route[-1]==(0,0)
    changed=np.argwhere(c.descent_allowed & ~reserved)
    assert changed.tolist()==[list(cell)]
    assert not c.checkpoint_supported[cell]
    # The exception follows the robot; it cannot accumulate permissive cells.
    c.choose_route((0,0),(0,0),0.)
    assert not c.descent_allowed[cell]
    # A genuinely unsupported body never receives this exception.
    c.descent_body_supported[cell]=False
    route,complete=c.choose_route(xy,(0,0),meta['yaw'])
    assert not route and not complete
