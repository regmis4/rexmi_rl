"""Physical geometry and lifecycle regressions, not a simulation replay."""
import math
from unittest.mock import patch

import numpy as np
import pytest
import test_remote_control as fixture
from rexmi_bt_tests.nav import checked_turn as ct, turn_staging as ts
from rexmi_bt_tests.nav.swept_footprint import oriented_swept_cells


def control():
    c=fixture.controller(fixture.bt.ground())
    c.goal_mode='any_exit';c.exit_rim_radius=1.
    c.grid.checked_small_turns=True;c.grid.side_limit_deg=32.
    c.obs=fixture.bt.observation(3.02,velocity=(.2,0.))
    c.observation_dt=.02
    c.target=(4.,0.)
    return c


def slope(g, degrees=20.):
    g.slope[:]=math.tan(math.radians(degrees))
    g.gx[:]=g.slope;g.gy[:]=0.
    # Preserve a supported gentle synthetic exterior around the 4 m target.
    edge=g.world_to_cell(3.4,0.)[0]
    g.slope[edge:]=0.;g.gx[edge:]=0.


def test_small_and_straight_turns_do_not_require_slope_minimum():
    c=control();g=c.grid;slope(g)
    assert ts.stage_reason(g,(0.,0.))
    assert not ts.turn_reason(g,(0.,0.),0.,0.)
    assert not ts.turn_reason(g,(0.,0.),0.,.3)
    assert ts.turn_reason(g,(0.,0.),0.,.36)
    g.checked_small_turns=False
    assert ts.turn_reason(g,(0.,0.),0.,.3)


@pytest.mark.parametrize('field,value', [('known',False),('hazard',True),
    ('no_go',True),('slope',math.tan(math.radians(36.))),('gx',math.nan)])
def test_complete_footprint_guards_remain(field,value):
    c=control();g=c.grid
    cells=oriented_swept_cells(g,(0.,0.),(1.5,0.),0.,(0.,.35))
    item=(int(cells[0][-1]),int(cells[1][-1]))
    getattr(g,field)[item]=value
    assert ct.physical_reason(g,(0.,0.),(1.5,0.),0.,(0.,.35))


def test_combined_drift_and_yaw_includes_cells_missed_by_separate_checks():
    c=control();g=c.grid
    def cells(end, offsets):
        return set(zip(*oriented_swept_cells(g,(0.,0.),end,0.,offsets)))
    joint=cells((1.3,.6),(0.,.35))
    separate=cells((0.,0.),(0.,.35)) | cells((1.3,.6),(0.,0.))
    extra=joint-separate
    assert extra
    g.hazard[next(iter(extra))]=True
    assert ct.physical_reason(g,(0.,0.),(1.3,.6),0.,(0.,.35))
    assert not ct.physical_reason(g,(0.,0.),(0.,0.),0.,(0.,.35))
    assert not ct.physical_reason(g,(0.,0.),(1.3,.6),0.,(0.,0.))


def test_cross_slope_checks_off_center_cells_and_interior_heading_extrema():
    c=control();g=c.grid;g.side_limit_deg=32.8
    cells=oriented_swept_cells(g,(0.,0.),(0.,0.),0.,(0.,.35))
    item=(int(cells[0][-1]),int(cells[1][-1]))
    assert item!=g.world_to_cell(0.,0.)
    magnitude=math.tan(math.radians(33.));angle=.175-math.pi/2
    g.gx[item]=magnitude*math.cos(angle);g.gy[item]=magnitude*math.sin(angle)
    assert not ct.physical_reason(g,(0.,0.),(0.,0.),0.,(0.,0.))
    assert not ct.physical_reason(g,(0.,0.),(0.,0.),.35,(0.,0.))
    assert 'side-slope' in ct.static_reason(g,(0.,0.),0.,.35)


def test_runtime_requires_continuity_freshness_and_unclipped_uncertainty():
    c=control()
    assert not ct.runtime_reason(c,.3)
    c.observation_dt=.2
    assert ct.runtime_reason(c,.3)
    c.observation_dt=.02;c.grid.fresh[:]=False
    assert 'stale' in ct.runtime_reason(c,.3)
    c.grid.fresh[:]=True
    c.obs=fixture.bt.observation(3.02,velocity=(.2,0.),yaw_rate=4.)
    assert 'uncertainty' in ct.runtime_reason(c,.3)


def test_yaw_only_commitment_survives_drift_without_renewing_patience():
    c=control();slope(c.grid)
    c.progress_time=2.;c.low_progress_active=4.
    command=c._begin_checked_turn(.3)
    assert command==fixture.r.Motion(0.,.35)
    deadline=c.checked_turn['deadline'];episode=c.recovery_episode
    c.obs=fixture.bt.observation(3.22,xy=(.04,0.),yaw=.12,velocity=(.2,0.),yaw_rate=.1)
    assert c._checked_turn_tick()==fixture.r.Motion(0.,.35)
    assert c.checked_turn['deadline']==deadline
    c.obs=fixture.bt.observation(3.42,xy=(.08,0.),yaw=.24,velocity=(.2,0.),yaw_rate=.1)
    assert c._checked_turn_tick()==fixture.r.Motion()
    assert c.checked_turn is None and c.recovery_episode==episode and c.recovery_actions==1
    assert (c.progress_time,c.low_progress_active)==(2.,4.)
    with patch.object(c,'_recover',return_value=fixture.r.Motion()) as recover:
        c._begin_checked_turn(.35)
    assert 'onward response' in recover.call_args.args[0]


@pytest.mark.parametrize('failure', ['deadline','no_progress','hazard'])
def test_committed_correction_has_bounded_duration_and_rechecks_geometry(failure):
    c=control();c._begin_checked_turn(.3)
    now=c.checked_turn['deadline'] if failure=='deadline' else 6.03 if failure=='no_progress' else 3.04
    c.obs=fixture.bt.observation(now,velocity=(.2,0.))
    if failure=='hazard':c.grid.hazard[c.grid.world_to_cell(.1,0.)]=True
    with patch.object(c,'_recover',return_value=fixture.r.Motion()) as recover:
        assert c._checked_turn_tick()==fixture.r.Motion()
    assert recover.called and c.climb_failures==0


@pytest.mark.parametrize('bound', ['actions','episode'])
def test_corrections_share_recovery_limits(bound):
    c=control()
    if bound=='actions':c.recovery_actions=8
    else:c.recovery_episode=-117.
    assert c._begin_checked_turn(.3)==fixture.r.Motion()
    assert c.operator_mode=='stop'


def test_no_op_heading_does_not_spend_action_and_operator_takeover_clears_turn():
    c=control()
    c.path=[(0.,0.),(.1,0.),(4.,0.)]
    c._begin_checked_turn(.05,True)
    assert c.vertex==2 and c.checked_turn is None and c.recovery_actions==0
    c._begin_checked_turn(.3)
    c.request('stop',3.04)
    assert c.checked_turn is None and c.correction_anchor is None
    assert c.tick((0.,0.),0.,.3,3.06)==fixture.r.Motion()
    c.request('manual',3.08,vx=-.4,omega=.35)
    assert c.tick((0.,0.),0.,0.,3.1)==fixture.r.Motion(-.4,.35)


def test_final_command_gate_independently_checks_committed_turn():
    c=control();c._begin_checked_turn(.3)
    with patch.object(ct,'runtime_reason',return_value='changed envelope'):
        assert c._command_geometry(fixture.r.Motion(0.,.35))=='changed envelope'


def test_forward_rejoin_checks_full_suffix_and_preserves_budgets():
    c=control();c.obs=fixture.bt.observation(3.02,xy=(0.,.2),velocity=(.2,0.))
    original=list(c.path)
    c.progress_time=1.;c.low_progress_active=5.;c.recovery_episode=2.;c.recovery_actions=2
    cell=c.grid.world_to_cell(3.,0.);c.grid.hazard[cell]=True
    assert c._try_checked_rejoin() is None and c.path==original
    c.grid.hazard[cell]=False
    assert c._try_checked_rejoin() is not None
    assert c.path[0]==c.obs.xy and c.path[-1]==original[-1]
    assert 0.<c.path[1][0]<original[-1][0]
    assert (c.progress_time,c.low_progress_active,c.recovery_episode,c.recovery_actions)==(1.,5.,2.,2)
    assert not ct.rejoins(original,1,(0.,2.),0.)
    assert not ct.rejoins(original,1,(4.1,0.),0.)


def test_any_exit_replanning_does_not_refresh_stalled_progress():
    c=control();c.path=[];c.progress_time=1.;c.low_progress_active=5.
    with patch.object(c,'_plan',return_value=True):c._navigate()
    assert (c.progress_time,c.low_progress_active)==(1.,5.)


def test_small_bend_no_longer_waits_for_exact_minimum_and_large_bend_still_does():
    c=control();slope(c.grid)
    c.path=[(-1.,0.),(0.,0.),(1.,.2)]
    assert not c._minimum_bend(1)
    c.path[-1]=(1.,1.)
    assert c._minimum_bend(1)


def test_translation_during_yaw_cannot_buy_another_correction():
    c=control();c._begin_checked_turn(.3)
    c.obs=fixture.bt.observation(3.6,xy=(.6,0.),yaw=.25,velocity=(.2,0.))
    c._checked_turn_tick()
    assert c.checked_turn is None
    assert c.correction_anchor[0]==c.recovery_onward_anchor==(.6,0.)
    with patch.object(c,'_recover',return_value=fixture.r.Motion()) as recover:
        c._begin_checked_turn(.5)
    assert 'onward response' in recover.call_args.args[0]


def test_failed_rejoin_cannot_fall_through_without_checked_continuation():
    c=control();c.grid.hazard[c.grid.world_to_cell(3.,0.)]=True
    with patch.object(c,'_recover',return_value=fixture.r.Motion()) as recover:
        c._steep_correction(.3)
    assert c.checked_turn is None and recover.called


def test_small_turn_handoff_keeps_original_deadline():
    c=control();c._begin_minimum_turn(.3,deadline=4.)
    assert c.checked_turn['deadline']==4.


def test_physical_recovery_after_correction_captures_permission_once():
    c=control();c._begin_checked_turn(.3)
    episode=c.recovery_episode
    with patch.object(fixture.r.RecoveryPermission,'capture',wraps=fixture.r.RecoveryPermission.capture) as capture:
        c._recover('correction did not progress')
        c._recover('still within same recovery episode')
    assert capture.call_count==1 and c.recovery_episode==episode


def test_refusal_archives_exact_map_and_separate_candidate_checks(tmp_path):
    from rexmi_bt_tests.nav import planning_snapshot
    c=control();c.planning_snapshot_dir=tmp_path/'plans'
    c.grid.hazard[c.grid.world_to_cell(3.,0.)]=True
    assert c._try_checked_rejoin() is None
    event=c.attempt_events[-1]
    assert event['event']=='checked_rejoin_refusal' and len(event['candidates'])==3
    assert any(check.get('physical_reason') or check.get('observed_reason')
        for candidate in event['candidates'] for check in candidate['checks'] if check['stage']=='suffix')
    g,meta=planning_snapshot.load(event['snapshot']['path'])
    assert meta['time']==c.obs.time and meta['yaw']==c.obs.yaw
    assert g.checked_small_turns and np.array_equal(g.hazard,c.grid.hazard)
    for _ in range(6):c._try_checked_rejoin()
    assert len(list((tmp_path/'plans-rejoins').glob('*.npz')))==4


def test_refusal_capture_failure_cannot_admit_motion(tmp_path):
    c=control();c.planning_snapshot_dir=tmp_path/'plans'
    c.grid.hazard[c.grid.world_to_cell(3.,0.)]=True
    from rexmi_bt_tests.nav import planning_snapshot
    with patch.object(planning_snapshot,'save',side_effect=OSError('no storage')):
        assert c._try_checked_rejoin() is None
    assert c.attempt_events[-1]['snapshot']['status']=='capture failed'
    assert c.checked_turn is None


def test_diagnostic_geometry_cannot_change_execution_admission_counters():
    c=control();c.static_prior_enabled=c.static_retention_enabled=True
    c.prior_support=c.grid.known.copy();c.grid.fresh[:]=False
    c.prior_admissions=0;c.last_prior_event=-math.inf
    before=list(c.attempt_events)
    ct.continuation_audit(c,[(0.,0.),(1.,0.),(4.,0.)])
    assert not ct.runtime_reason(ct.diagnostic_copy(c),.3)
    assert c.prior_admissions==0 and c.last_prior_event==-math.inf
    assert c.attempt_events==before
