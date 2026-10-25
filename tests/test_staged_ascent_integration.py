"""Remote planner integration with synthetic local-observation support."""
import copy
import math
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
import test_remote_routes as f
from rexmi_bt_tests.nav import stage_graph as sg
from rexmi_bt_tests.nav.ascent_control import AscentController


@pytest.fixture(scope='module')
def survey():return f.survey.__wrapped__()


def control(survey):
    c=f.r.RemoteAscentController(copy.deepcopy(survey),[f.m.Waypoint(0.,13.,.6,'goal')],
        mode='characterize',uphill_limit=33.,side_limit=32.)
    c.obs=f.bt.observation(3.,xy=(0.,0.),yaw=math.pi/2);c.last_yaw=math.pi/2
    return c


def test_current_map_with_local_fresh_observations_uses_graph_without_legacy_search(survey):
    c=control(survey);g=c.grid
    rr,cc=np.indices(g.known.shape)
    x=g.origin[0]+(rr+.5)*g.cell_size-g.world_size/2
    y=g.origin[1]+(cc+.5)*g.cell_size-g.world_size/2
    g.fresh[:]=g.known & (np.hypot(x,y)<2.5)
    fresh=g.fresh.copy()
    with patch.object(AscentController,'_propose_routes',side_effect=AssertionError('parallel legacy search')):
        assert c._compute_plan((0.,0.),math.pi/2,3.)
    assert c.graph_objective_active and c.mission_route_complete
    assert c.path[0]==(0.,0.) and c.path[-1]==(0.,13.)
    assert np.array_equal(g.fresh,fresh)
    assert not f.rr.motion_route_reason(g,c.path,math.pi/2)
    assert not f.rr.physical_profile(c,c.path)[0]
    assert not sg.route_height_reason(g,c.path)
    assert all(math.dist(a,b)>=2*g.footprint_radius-1e-8 for a,b in zip(c.path,c.path[1:]))
    selected=next(r for r in c.route_options if r['selected'])
    assert selected['profile']=='terrain-checkpoints'
    assert selected['max_uphill_deg']<26. and selected['uphill_over_25deg_m']<1.
    assert selected['height_retention_status'].startswith('sampled')
    assert 'uphill burden first' in selected['success_selection']['reason']


def test_missing_local_observations_trigger_sequential_explicit_fallback(survey):
    c=control(survey)
    assert not c.grid.fresh.any()
    with patch.object(AscentController,'_propose_routes',return_value=[]) as fallback:
        assert c._propose_routes((0.,0.),math.pi/2,c.grid.planning_support(),33.)==[]
    fallback.assert_called_once()
    assert not c.graph_objective_active
    assert 'unverified' in c.graph_plan_stats['fallback']


@pytest.mark.parametrize('kind',['no_prior','outside_floor','descent'])
def test_nonmatching_scope_preserves_existing_generator(survey,kind):
    c=control(survey);xy=(0.,0.)
    if kind=='no_prior':c.static_retention_enabled=False
    if kind=='outside_floor':xy=(5.,0.)
    if kind=='descent':c.waypoints=[f.m.Waypoint(0.,0.,.6,'floor')]
    with patch.object(sg,'build',side_effect=AssertionError('wrong graph scope')),patch.object(
        AscentController,'_propose_routes',return_value=['legacy']):
        assert c._propose_routes(xy,0.,c.grid.planning_support(),33.)==['legacy']


def test_adoption_rechecks_height_contract_and_preserves_non_graph_behavior(survey):
    c=control(survey);shadow=SimpleNamespace(graph_objective_active=True,graph_plan_stats={'profile':'terrain-checkpoints'},route_purpose='mission')
    with patch.object(sg,'route_height_reason',return_value='sampled ascent-height retention changed') as check:
        assert c._adoption_route_reason([(0.,0.),(0.,13.)],(0.,0.),0.,shadow)
        check.assert_called_once()
        shadow.graph_objective_active=False
        assert not c._adoption_route_reason([(0.,0.),(0.,13.)],(0.,0.),0.,shadow)
        check.assert_called_once()
    c._adopt_route_metadata(shadow)
    assert c.graph_plan_stats==shadow.graph_plan_stats
    shadow.graph_objective_active=True;shadow.route_purpose='retreat'
    assert not c._adoption_route_reason([(0.,0.),(0.,13.)],(0.,0.),0.,shadow)
    c._adopt_route_metadata(shadow)
    assert not c.graph_objective_active and c.graph_plan_stats is None


def test_regroup_early_return_clears_previous_ascent_objective(survey):
    c=control(survey)
    c.graph_objective_active=True
    c.graph_plan_stats={'profile':'terrain-checkpoints'}
    c.regroup_required=True
    with patch.object(c,'_compute_regroup',return_value=False) as regroup:
        assert not c._compute_plan((0.,0.),math.pi/2,3.)
    regroup.assert_called_once()
    assert not c.graph_objective_active and c.graph_plan_stats is None
    c.route_purpose='retreat'
    with patch.object(sg,'route_height_reason',side_effect=AssertionError('ascent rule on regroup')):
        assert not c._adoption_route_reason([(0.,0.),(0.,-1.)],(0.,0.),0.,c)
