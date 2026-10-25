import math
import json
import time
from collections import defaultdict
from unittest.mock import patch
import numpy as np
import pytest
from test_climb_burden import terrain
from rexmi_bt_tests.nav import sustained_climb as sc, climb_burden as cb


def item(length,effort,respite,start=0.):
    return sc.Interval(length,effort/length,(1.,0.,0.),respite,start)


def test_uninterrupted_interrupted_initial_and_final_bout():
    state,events=sc.advance(sc.State(),[item(1,2,False),item(.4,0,True),item(1,3,False)],1.)
    assert not events and state.score==25 and state.respite_distance==0
    state,events=sc.advance(sc.State(),[item(.9,2,True)],1.)
    assert not events and state.score==4 and not state.latched


def test_exact_threshold_crossing_one_reset_and_no_waypoint_credit():
    first=item(.4,1,True)
    state,_=sc.advance(sc.State(),[first],1.)
    second=item(1.6,4,True)
    state,events=sc.advance(state,[second],1.)
    assert len(events)==1 and events[0]['distance_m']==pytest.approx(.6)
    assert state.score==pytest.approx(2*2.5**2)
    assert state.latched and state.respite_distance==1.
    state,events=sc.advance(state,[item(5,0,True)],1.)
    assert not events and state.score==pytest.approx(12.5)
    state,_=sc.advance(state,[item(.2,0,False)],1.)
    state,events=sc.advance(state,[item(1,0,True)],1.)
    assert len(events)==1 and state.burden==0


def test_polynomial_integration_and_grouping_preserve_reset_accounting():
    # demand=t, integral t² from0 to1 is1/3; reset at midpoint.
    part=sc.Interval(1.,1.,(0.,1.,0.),True)
    result,events=sc.advance(sc.State(),[part],.5)
    assert events[0]['bout_effort']==pytest.approx(1/24)
    assert result.score==pytest.approx((1/24)**2+(7/24)**2)
    raw=[item(.4,1,True,0),item(.6,2,True,.4),item(1,3,False,1)]
    a,ea=sc.advance(sc.State(),raw,.7)
    b,eb=sc.advance(sc.State(),sc.runs(raw),.7)
    assert a.score==pytest.approx(b.score) and a.burden==pytest.approx(b.burden)
    assert ea[0]['distance_m']==pytest.approx(eb[0]['distance_m'])


def test_real_terrain_collinear_splits_and_signed_respite_boundaries():
    g,x,y=terrain(lambda x,y:.12*x*x,cost=2.)
    field=.12*x*x
    a,b=(-2.,0.),(2.,0.)
    whole=sc.intervals(g,field,a,b)
    assert any(v.respite for v in whole) and any(not v.respite for v in whole)
    # Negative high grade must not count as respite.
    assert not whole[0].respite and whole[0].effort()==0
    original,_=sc.advance(sc.State(),whole,2*g.footprint_radius)
    for cut in (-1.13,-.21,.73):
        state,_=sc.advance(sc.State(),sc.intervals(g,field,a,(cut,0.)),2*g.footprint_radius)
        state,_=sc.advance(state,sc.intervals(g,field,(cut,0.),b),2*g.footprint_radius)
        assert state.score==pytest.approx(original.score,rel=1e-9,abs=1e-10)
        assert state.burden==pytest.approx(original.burden,abs=1e-10)
    # Every threshold-split interval satisfies its label at interior probes.
    for v in whole:
        grades=np.polynomial.polynomial.polyval(np.linspace(.001,.999,9),v.coeff)
        assert np.all(np.abs(grades)<=sc.LIMIT+1e-10) if v.respite else np.all(np.abs(grades)>=sc.LIMIT-1e-10)


def graph(links):
    n=max(max(a,b) for a,b,_,_ in links)+1
    edges={(a,b):dict(length=1.,runs=(item(1.,effort,relief),)) for a,b,effort,relief in links}
    forward=defaultdict(list);reverse=defaultdict(list)
    for s in [(-1,0)]+list(edges):
        if s[1]==1:continue
        for target in edges:
            if target[0]==s[1]:forward[s].append(target);reverse[target].append(s)
    data=dict(points=[(float(i),0.) for i in range(n)],edges=edges,deadline=time.perf_counter()+10.,relief_distance=1.)
    return data,(forward,reverse,dict(transition_budget_exhausted=False))


def solve(links):
    data,transitions=graph(links)
    with patch.object(sc.ascent_first,'transitions',return_value=transitions):
        return sc.search(None,data,0.)


def test_higher_prefix_cost_lower_open_burden_must_survive():
    links=[(0,2,2,False),(0,3,math.sqrt(5),False),(3,2,0,True),
           (2,4,0,False),(4,1,3,False)]
    path,stats=solve(links)
    assert [int(p[0]) for p in path]==[0,3,2,4,1]
    assert stats['rank']==pytest.approx([14.,4.]) and stats['optimal']


def exhaustive(links):
    outgoing=defaultdict(list)
    for a,b,e,r in links:outgoing[a].append((b,e,r))
    found=[]
    def visit(node,closed,active,length):
        if node==1:found.append((closed+active**2,length));return
        for target,effort,relief in outgoing[node]:
            # Each graph edge starts with a non-respite marker to re-arm;
            # this independent recurrence matches the fixture below.
            burden=active+effort
            visit(target,closed+burden**2 if relief else closed,0. if relief else burden,length+1.01)
    visit(0,0.,0.,0)
    return min(found) if found else None


def test_random_dags_match_independent_path_enumeration():
    rng=np.random.default_rng(42);order=[0,2,3,4,5,1]
    for _ in range(25):
        links=[(a,b,float(rng.integers(0,5)),bool(rng.integers(0,2)))
            for i,a in enumerate(order) for b in order[i+1:] if rng.random()<.5]
        if not links:continue
        data,transitions=graph(links)
        for edge in data['edges'].values():
            edge['runs']=(item(0.01,0,False),)+edge['runs']
            edge['length']=1.01
        with patch.object(sc.ascent_first,'transitions',return_value=transitions):
            path,stats=sc.search(None,data,0.)
        expected=exhaustive(links)
        if expected is None:assert not path
        else:assert stats['rank']==pytest.approx(expected)


def test_resource_caps_and_unreachable_states_fail_closed():
    links=[(0,2,1,False),(2,1,1,False),(0,3,0,True)]
    path,stats=solve(links)
    assert path and stats['reverse_pruned']>0
    for name in ('MAX_LABELS','MAX_EXPANSIONS','MAX_RELAXATIONS'):
        with patch.object(sc,name,0):
            path,stats=solve(links)
        assert not path and stats['budget_exhausted'] and not stats['optimal']


def test_domination_does_not_merge_different_relief_phases():
    a=dict(state=sc.State(),length=0.)
    b=dict(state=sc.State(1.,1.,.5,False),length=2.)
    assert not sc.dominates(a,b)
    b['state']=sc.State(1.,1.,0.,False)
    assert sc.dominates(a,b)


def test_real_graph_build_search_and_rescore_contract():
    g,_,_=terrain(lambda x,y:0*x)
    g.motion_turn_limit=15.
    with patch.object(sc.stage_graph,'stable_nodes',return_value=([(0.,0.),(2.,0.)],{})):
        data=sc.build(g,(0.,0.),(2.,0.),lambda a,b:'')
    path,stats=sc.search(g,data,0.)
    assert stats['complete'] and stats['rank']==pytest.approx([0.,2.])
    assert sc.describe_route(g,path)['rank']==pytest.approx(stats['rank'])
    json.dumps(sc.describe_route(g,path),allow_nan=False)
    assert all(edge['length']==2. for edge in data['edges'].values())
