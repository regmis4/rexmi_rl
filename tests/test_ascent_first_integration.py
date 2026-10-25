"""New planner mode owns mission ranking without changing baseline/recovery."""
import copy
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import test_staged_ascent_integration as old
from rexmi_bt_tests.nav import ascent_first as af, stage_graph as sg, climb_burden as cb, wheelbase_demand as wd, sustained_climb as sc
from rexmi_bt_tests.nav.ascent_control import AscentController
from rexmi_bt_tests.nav.remote_control import RemoteAscentController


@pytest.fixture(scope='module')
def survey():return old.survey.__wrapped__()


def test_new_option_requires_full_survey_and_baseline_stays_default():
    import test_remote_control as fixture
    c=fixture.controller(fixture.bt.ground())
    assert c.route_objective=='baseline' and not c.ascent_first_active
    with pytest.raises(ValueError,match='survey'):
        RemoteAscentController(c.grid,c.waypoints,mode='characterize',route_objective='ascent_first')


@pytest.mark.parametrize('objective',['ascent_first','climb_burden','wheelbase_demand','sustained_climb'])
def test_explicit_new_mode_never_falls_back_on_no_witness_even_above_floor(survey,objective):
    c=old.control(survey);c.route_objective=objective
    planner={'climb_burden':cb,'ascent_first':af,'wheelbase_demand':wd,'sustained_climb':sc}[objective]
    fake=dict(inventory={},points=[(6.,0.),(0.,13.)],directed_checks=2,
        edges={},build_seconds=0.,rejections={})
    stats=dict(complete=False,reason='heading graph budget exhausted',budget_exhausted=True)
    with patch.object(planner,'build',return_value=fake),patch.object(planner,'search',return_value=([],stats)),patch.object(
        AscentController,'_propose_routes',side_effect=AssertionError('silent old fallback')):
        assert c._propose_routes((6.,0.),0.,None,33.)==[]
    assert c.ascent_first_active and not c.graph_objective_active
    assert c.graph_plan_stats['rejection']==stats['reason']


def test_baseline_and_descent_keep_old_generator(survey):
    c=old.control(survey);c.route_objective='ascent_first'
    c.waypoints=[SimpleNamespace(x=0.,y=0.)]
    with patch.object(af,'build',side_effect=AssertionError('ascent planner during descent')),patch.object(
        AscentController,'_propose_routes',return_value=['baseline']):
        assert c._propose_routes((13.,0.),0.,None,33.)==['baseline']
    assert not c.ascent_first_active


def test_objective_score_and_adoption_keep_mode_and_exit(survey):
    c=old.control(survey);path=[(0.,0.),(1.,2.),(0.,13.)]
    c.ascent_first_active=True
    c.graph_plan_stats=dict(complete=True,rank=[10.,20.,15.],exit_index=1,path=path)
    with patch.object(c.grid,'route_cost',side_effect=AssertionError('legacy rank')):
        assert c.route_score(path,0.)==10.
        assert c._recent_approach_cost(path,3.)==0.
    shadow=SimpleNamespace(ascent_first_active=True,graph_objective_active=False,
        graph_plan_stats=c.graph_plan_stats,route_purpose='mission')
    with patch.object(af,'route_reason',return_value='changed suffix') as check:
        assert c._adoption_route_reason(path,(0.,0.),0.,shadow)=='changed suffix'
        assert check.call_args.args[3]==1
    c._adopt_route_metadata(shadow)
    assert c.ascent_first_active
    shadow.route_purpose='retreat';c._adopt_route_metadata(shadow)
    assert not c.ascent_first_active and c.graph_plan_stats is None


def test_new_graph_never_evaluates_old_cost_terms():
    import test_behavior_navigation as bt
    g=bt.ground();g.motion_turn_limit=15.
    with patch.object(sg,'stable_nodes',return_value=([(0.,0.),(2.,0.)],{})),patch.object(
        sg,'segment_cost',side_effect=AssertionError('old weights')):
        data=af.build(g,(0.,0.),(2.,0.),lambda a,b:'')
    assert data['edges'] and all('ascent' in row for row in data['edges'].values())


def test_climb_burden_adoption_has_no_old_suffix_classification(survey):
    c=old.control(survey);path=[(0.,0.),(0.,13.)]
    shadow=SimpleNamespace(ascent_first_active=True,graph_objective_active=False,
        graph_plan_stats={'objective':cb.NAME},route_purpose='mission')
    with patch.object(cb,'route_reason',return_value='changed physical support') as check,patch.object(
        af,'route_reason',side_effect=AssertionError('legacy suffix boundary')):
        assert c._adoption_route_reason(path,(0.,0.),0.,shadow)=='changed physical support'
        check.assert_called_once()


def test_wheelbase_adoption_preserves_physical_rejection(survey):
    c=old.control(survey);path=[(0.,0.),(0.,13.)]
    shadow=SimpleNamespace(ascent_first_active=True,graph_objective_active=False,
        graph_plan_stats={'objective':wd.NAME},route_purpose='mission')
    with patch.object(wd,'route_reason',return_value='changed physical support') as check:
        assert c._adoption_route_reason(path,(0.,0.),0.,shadow)=='changed physical support'
        check.assert_called_once()


def test_wheelbase_constructor_and_descent_ownership(survey):
    c=RemoteAscentController(copy.deepcopy(survey),[],mode='characterize',route_objective='wheelbase_demand')
    assert 'v26' in c.protocol_version
    c.waypoints=[SimpleNamespace(x=0.,y=0.)]
    with patch.object(wd,'build',side_effect=AssertionError('ascent planner during descent')),patch.object(
        AscentController,'_propose_routes',return_value=['baseline']):
        assert c._propose_routes((13.,0.),0.,None,33.)==['baseline']
    assert not c.ascent_first_active


def test_sustained_constructor_adoption_and_descent_ownership(survey):
    c=RemoteAscentController(copy.deepcopy(survey),[],mode='characterize',route_objective='sustained_climb')
    assert 'v27' in c.protocol_version
    shadow=SimpleNamespace(ascent_first_active=True,graph_objective_active=False,
        graph_plan_stats={'objective':sc.NAME},route_purpose='mission')
    with patch.object(sc,'route_reason',return_value='new physical rejection'):
        assert c._adoption_route_reason([(0.,0.),(0.,13.)],(0.,0.),0.,shadow)=='new physical rejection'
    c.waypoints=[SimpleNamespace(x=0.,y=0.)]
    with patch.object(sc,'build',side_effect=AssertionError('ascent during descent')),patch.object(
        AscentController,'_propose_routes',return_value=['baseline']):
        assert c._propose_routes((13.,0.),0.,None,33.)==['baseline']
