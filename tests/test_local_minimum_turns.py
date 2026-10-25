"""Map admission and ordered commands, not physical turn qualification."""
import copy
import math
from unittest.mock import patch
import numpy as np
import pytest
import test_behavior_navigation as bt
import test_remote_control as remote
from rexmi_bt_tests.nav.turn_staging import minimum_fields, stage_reason, turn_reason
from rexmi_bt_tests.nav.staged_routes import motion_route_reason


@pytest.fixture(scope='module')
def flat():return bt.ground()


def minimum_grid(flat,small=False):
    g=copy.deepcopy(flat);g.motion_turn_limit=15.;g.local_minimum_turns_enabled=True
    g.uphill_limit_deg=33.;g.side_limit_deg=25.
    axis=(np.arange(g.n_cells)+.5)*g.cell_size-g.world_size/2
    xx,yy=np.meshgrid(axis,axis,indexing='ij')
    g.slope[:]=math.tan(math.radians(28.))
    g.slope[np.hypot(xx-.1,yy-.1)<(.12 if small else .55)]=math.tan(math.radians(20.))
    g.gx[:]=g.slope;g.gy[:]=0.;g.revision+=1
    return g


def test_true_footprint_minimum_admits_bend_but_uniform_plane_and_tiny_notch_do_not(flat):
    g=minimum_grid(flat)
    assert not stage_reason(g,(.1,.1))
    assert not turn_reason(g,(.1,.1),0.,math.pi/2)
    assert not motion_route_reason(g,[(-1.,.1),(.1,.1),(.1,1.)],0.)
    g.slope[:]=math.tan(math.radians(20.));g.revision+=1
    assert stage_reason(g,(.1,.1))
    assert stage_reason(minimum_grid(flat,True),(.1,.1))


def test_unknown_comparison_footprint_cannot_establish_minimum(flat):
    g=minimum_grid(flat)
    g.known[g.world_to_cell(.7,.1)]=False;g.revision+=1
    assert stage_reason(g,(.1,.1))


def test_requested_rotation_checks_intermediate_cross_grade_and_hazard(flat):
    g=minimum_grid(flat)
    g.gx[g.world_to_cell(.1,.1)]=math.tan(math.radians(30.))
    assert 'side-slope' in turn_reason(g,(.1,.1),0.,math.pi)
    g.gx[g.world_to_cell(.1,.1)]=math.tan(math.radians(20.))
    g.hazard[g.world_to_cell(.1,.1)]=True
    assert 'obstacle' in turn_reason(g,(.1,.1),0.,math.pi/2)


def test_map_revision_invalidates_minimum_cache(flat):
    g=minimum_grid(flat);assert minimum_fields(g)[0][g.world_to_cell(.1,.1)]
    g.slope[:]=math.tan(math.radians(28.));g.revision+=1
    assert not minimum_fields(g)[0][g.world_to_cell(.1,.1)]


def controller(flat):
    c=remote.controller(minimum_grid(flat),(.1,3.))
    c.path=[(-1.,.1),(.1,.1),(.1,3.)]
    c.progress_anchor=(-.2,.1)
    return c


def test_minimum_checkpoint_not_skipped_and_turn_waits_for_stability(flat):
    c=controller(flat)
    with patch.object(c,'_geometry',return_value=''),patch.object(c,'_anticipation',return_value=''):
        assert c.tick((.1,.1),0.,.2,3.,velocity=(.2,0.))==remote.r.Motion()
        assert c.vertex==1 and c.button_state=='MINIMUM_SETTLE'
        for t in (3.+.02*k for k in range(1,30)):
            assert c.tick((.1,.1),0.,0.,t)==remote.r.Motion()
        m=c.tick((.1,.1),0.,0.,3.6)
        assert m.omega==0.
        m=c.tick((.1,.1),0.,0.,3.64)
        assert m.vx==0. and m.omega==.35 and c.vertex==1
        deadline=c.minimum_turn['deadline']
        m=c.tick((.1,.1),.007,0.,3.66)
        assert m.omega==.35 # Intentional yaw does not cancel its own turn.
        m=c.tick((.1,.1),.014,.2,3.68,velocity=(.2,0.))
        assert m==remote.r.Motion() and c.minimum_turn['deadline']==deadline
        assert c.button_state=='MINIMUM_SETTLE'
        c.request('stop',3.69)
        assert c.minimum_turn is None and c.tick((.1,.1),.014,0.,3.70)==remote.r.Motion()


def test_settle_timeout_and_observation_gap_do_not_renew_deadline(flat):
    c=controller(flat)
    with patch.object(c,'_geometry',return_value=''),patch.object(c,'_anticipation',return_value=''):
        c.tick((.1,.1),0.,0.,3.)
        deadline=c.minimum_turn['deadline']
        c.tick((.1,.1),0.,0.,3.5)
        assert c.minimum_turn['stable_since'] is None
        with patch.object(c,'_recover',return_value=remote.r.Motion()) as recover:
            c.tick((.1,.1),0.,0.,deadline)
            recover.assert_called_once_with('local-minimum settle/turn deadline')


def test_emergency_resets_stability_without_changing_target_or_deadline(flat):
    c=controller(flat)
    with patch.object(c,'_geometry',return_value=''),patch.object(c,'_anticipation',return_value=''):
        c.tick((.1,.1),0.,0.,3.)
        c.minimum_turn.update(phase='turn',stable_since=3.,sign=1)
        before=(c.minimum_turn['target'],c.minimum_turn['deadline'])
        c.tick((.1,.1),0.,.81,3.02,velocity=(.81,0.))
        assert c.emergency and c.minimum_turn['phase']=='settle'
        assert c.minimum_turn['stable_since'] is None
        assert before==(c.minimum_turn['target'],c.minimum_turn['deadline'])


def test_legacy_turn_conversion_preserves_original_deadline(flat):
    c=controller(flat);c.path=[(.1,.1),(.1,3.)]
    c.button_state='TURN';c.turn_target=math.pi/2;c.turn_deadline=9.
    with patch.object(c,'_geometry',return_value=''),patch.object(c,'_anticipation',return_value=''):
        c.tick((.1,.1),0.,0.,8.5)
        assert c.minimum_turn['deadline']==9.


def test_nonfinite_comparison_surface_and_gradient_fail_closed(flat):
    g=minimum_grid(flat)
    g.gx[g.world_to_cell(.1,.1)]=math.nan
    assert 'unmeasured' in turn_reason(g,(.1,.1),0.,math.pi/2)
    g.surface[g.world_to_cell(.7,.1)]=math.nan;g.revision+=1
    assert stage_reason(g,(.1,.1))


def test_emitted_yaw_forecast_cannot_exceed_checked_target_cross_grade(flat):
    c=controller(flat)
    c.grid.gx[c.grid.world_to_cell(.1,.1)]=math.tan(math.radians(30.))
    c.obs=bt.observation(4.,xy=(.1,.1),yaw=.8)
    c.minimum_turn={'target':.91,'incoming':.8}
    assert not turn_reason(c.grid,c.obs.xy,.8,.91)
    assert 'side-slope' in c._command_geometry(remote.r.Motion(0.,.35))
