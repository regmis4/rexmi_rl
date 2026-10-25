"""Geometry-only integration; no trajectory or traction success claims."""
import copy
import importlib
import math
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
import test_behavior_navigation as bt

r=importlib.import_module('rexmi_bt_tests.nav.remote_control')
rr=importlib.import_module('rexmi_bt_tests.nav.remote_routes')
tp=importlib.import_module('rexmi_bt_tests.nav.terrain_prior')
m=importlib.import_module('rexmi_bt_tests.nav.mission')
ROOT=Path(__file__).resolve().parents[1]
XY=(-.3122323155403137,.9353443384170532)

@pytest.fixture(scope='module')
def survey():
    d=np.load(ROOT/'outputs/remote_autonomy/static-coverage-mission-185chj1a/run.map.npz')
    g=bt.a.AscentTerrainMap(world_size=64.,cell_size=.2,origin=tuple(d['origin']),
        footprint_radius=float(d['footprint_radius']),half_length=float(d['half_length']),half_width=float(d['half_width']))
    tp.load_prior(g,ROOT/'outputs/ascent_full_survey/trial01/run.prior.npz',
        task='RexmiRl-Go2w-Crater-Bowl-RockySlope-Play-v0',seed=42)
    return g

def control(survey,goal=(0.,13.)):
    c=r.RemoteAscentController(copy.deepcopy(survey),[m.Waypoint(*goal,.6,'goal')],mode='characterize',uphill_limit=33.)
    c.obs=bt.observation(104.4,xy=XY,yaw=math.pi/2);c.last_yaw=math.pi/2
    return c

def test_integrated_planner_selects_complete_connected_margin_route_with_bounded_work(survey):
    c=control(survey)
    with patch.object(rr,'observed_route',wraps=rr.observed_route) as search:
        assert c._compute_plan(XY,math.pi/2,104.4)
    assert search.call_count<=6
    assert all(call.kwargs['max_expansions']==10000 for call in search.call_args_list)
    assert sum(x['profile']=='wall-screen' for x in c.route_searches)==48
    assert sum(x['profile'].startswith('wall-corridor-') for x in c.route_searches)<=3
    selected=next(x for x in c.route_options if x['selected'])
    assert selected['profile'].startswith('wall-corridor-') and selected['complete']
    assert selected['max_body_slope_deg']<33.
    assert c.path[-1]==(0.,13.) and c.wp_idx==0
    assert not c.grid.fresh.any() # Static survey remains honestly aged.
    assert not rr.physical_profile(c,c.path)[0]
    assert not rr.motion_route_reason(c.grid,c.path,math.pi/2)

def test_integrated_search_diagnostics_satisfy_runner_logging_contract(survey):
    c=control(survey)
    assert c._compute_plan(XY,math.pi/2,104.4)
    # This is the runner's route-choice deduplication expression. Screens are
    # not complete routes, even when their individual climb segment is clear.
    key=tuple((p['profile'],p['connected'],p['has_route'],p['uphill_limit'])
              for p in c.route_searches)
    screens=[p for p in c.route_searches if p['profile']=='wall-screen']
    assert any(p['rejection'] for p in screens)
    assert any(not p['rejection'] for p in screens)
    assert all(not p['connected'] and not p['has_route'] for p in screens)
    assert any(label.startswith('wall-corridor-') and connected and has_route
               for label,connected,has_route,limit in key)
    assert all(p['uphill_limit']==c.grid.uphill_limit_deg for p in c.route_searches
               if p['profile'].startswith('wall-'))

@pytest.mark.parametrize('mode',['descent','nonstatic','outside_floor'])
def test_scope_retains_existing_planner_candidates(survey,mode):
    c=control(survey,(0.,0.) if mode=='descent' else (0.,13.))
    if mode=='nonstatic':c.static_retention_enabled=False
    xy=(13.,0.) if mode=='outside_floor' else XY
    routes=[('ordinary',[xy,(0.,0.)],True)]
    with patch.object(rr,'observed_route') as search:
        assert c._augment_routes(routes,xy,0.) is routes
    search.assert_not_called()

@pytest.mark.parametrize('rejection',['failed_approach','exit'])
def test_climb_failure_or_unsafe_exit_cannot_be_admitted(survey,rejection):
    c=control(survey)
    if rejection=='failed_approach':
        c.grid.mark_failure('real_failure',(1.47,6.),math.radians(69.),100.,'stall')
        assert not c._augment_routes([],XY,math.pi/2)
    else:
        original=c.grid.turn_support_reason
        with patch.object(c.grid,'turn_support_reason',side_effect=lambda p:'blocked exit' if math.hypot(*p)>10. else original(p)):
            assert not c._augment_routes([],XY,math.pi/2)


def test_ranking_uses_total_cost_with_physical_eligibility_for_all_candidates(survey):
    c=control(survey)
    def candidate(cost,label,marker,complete=True):
        return ('normal',cost,[(marker,0.),(marker,1.)],complete,None,None,None,None,None,{'profile':label})
    ordinary=candidate(615.7025,'ordinary',1.)
    corridor=candidate(601.8394,'wall-corridor-0',2.)
    with patch.object(rr,'physical_profile',side_effect=lambda control,path:('',31.6552 if path[0][0]==1. else 32.5328)):
        assert c._select_route_candidate([ordinary,corridor],XY,0.) is corridor
    with patch.object(rr,'physical_profile',side_effect=lambda control,path:('',31.6552) if path[0][0]==1. else ('obstacle',math.inf)):
        assert c._select_route_candidate([ordinary,corridor],XY,0.) is ordinary
        assert corridor[9]['rejection']=='obstacle'
    fronts=[candidate(20.,'frontier',1.,False),candidate(10.,'frontier2',2.,False)]
    assert c._select_route_candidate(fronts,XY,0.) is fronts[1]


def test_adoption_rechecks_current_climb_obstacle(survey):
    c=control(survey)
    a=(.2239934024,2.7501705795);b=(3.9255809322,12.3931357753)
    assert c._adoption_edge_reason(a,b,(0.,-1.),False)==''
    cell=c.grid.world_to_cell(1.47,6.);c.grid.hazard[cell]=True
    assert c._adoption_edge_reason(a,b,(0.,-1.),False)


def test_recorded_descent_search_finishes_inside_unchanged_budget(survey):
    c=control(survey,(0.,0.))
    start=(13.032443046569824,.02697044238448143);yaw=2.9865735323076525
    c.obs=bt.observation(3.4,xy=start,yaw=yaw);c.last_yaw=yaw
    path,complete=rr.observed_route(c.grid,start,(0.,0.),yaw,profile='direct',
        require_fresh_start=False,uphill_limit=33.,max_expansions=60000)
    assert complete and path[-1]==(0.,0.)
    assert c.grid.last_route_search['expansions']<60000
    assert not c.grid.last_route_search['budget_exhausted']
    assert not rr.physical_profile(c,path)[0]
    assert not rr.motion_route_reason(c.grid,path,yaw)
    assert all(not rr.observed_edge_reason(c.grid,a,b,start,fresh_radius=None,uphill_limit=33.)
               for a,b in zip(path,path[1:]))


def test_captured_query_remains_complete_and_cost_driven_without_calibrated_model(survey):
    c=control(survey)
    xy=(-.07983610033988953,.9571846723556519)
    c.obs=bt.observation(95.68,xy=xy,yaw=math.pi/2)
    assert c._compute_plan(xy,math.pi/2,95.68)
    assert c.path[0] == xy and c.path[-1] == (0.,13.)
    assert c.mission_route_complete
    assert not rr.physical_profile(c,c.path)[0]
    assert not rr.motion_route_reason(c.grid,c.path,math.pi/2)
    selected=next(row for row in c.route_options if row['selected'])
    assert selected['success_selection']['objective']=='characterization_cost_fallback'
    assert selected['success_selection']['probability_used'] is None
    # Correctly pricing terminal connectors can improve floor/exterior edges;
    # an unavailable probability still cannot change the cost-based ordering.
    assert selected['score'] <= 1048.9803124769658
