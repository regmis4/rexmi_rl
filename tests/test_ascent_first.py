"""Ascent-first ownership, phase separation and directed heading certificates."""
import math
import time
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
import test_behavior_navigation as bt
from rexmi_bt_tests.nav import ascent_first as af


def graph(points, links):
    edges={}
    for i,j,ascent,exterior,gentle in links:
        a,b=points[i],points[j]
        edges[i,j]=dict(heading=math.atan2(b[1]-a[1],b[0]-a[0]),ascent=ascent,
            exterior=exterior,gentle=gentle,length=math.dist(a,b))
    return dict(points=points,edges=edges,turns={},reference=[0.,4.],deadline=time.perf_counter()+30.)


def solve(data,**kwargs):
    with patch.object(af,'turn_reason',**kwargs or {'return_value':''}):
        return af.search(bt.ground(),data,0.)


def test_ascent_wins_even_when_exterior_is_longer_and_brighter():
    p=[(0.,0.),(10.,0.),(3.,1.),(4.,-1.)]
    d=graph(p,[(0,2,10.,5.,False),(2,1,100.,100.,True),
               (0,3,20.,10.,False),(3,1,1.,1.,True)])
    path,s=solve(d)
    assert path==[p[0],p[2],p[1]]
    assert s['rank'][:2]==[10.,100.] and s['exit_index']==1 and s['optimal']


def test_invalid_join_heading_is_not_a_node_only_exit_certificate():
    p=[(0.,-2.),(2.,4.),(2.,0.),(2.,-2.)]
    d=graph(p,[(0,2,1.,1.,False),(0,3,2.,2.,False),(3,2,2.,2.,False),(2,1,1.,1.,True)])
    def turn(g,point,incoming,outgoing):
        return 'bad joining turn' if point==p[2] and abs(incoming-math.pi/2)>.01 else ''
    path,s=solve(d,side_effect=turn)
    assert path==[p[0],p[3],p[2],p[1]]
    assert s['turn_rejections']['bad joining turn']>0
    assert s['exit_state']==[3,2]


def test_goal_is_fallback_exit_and_dead_end_does_not_win():
    p=[(0.,0.),(5.,0.),(1.,0.),(1.,1.)]
    d=graph(p,[(0,2,1.,1.,False),(0,3,3.,3.,False),(3,1,4.,4.,False)])
    path,s=solve(d)
    assert path==[p[0],p[3],p[1]] and s['exit_index']==2 and s['rank'][0]==7.


def test_phase_entry_irreversible_and_all_gentle_start_has_zero_ascent():
    p=[(0.,0.),(4.,0.),(2.,0.)]
    d=graph(p,[(0,2,100.,2.,True),(2,1,100.,2.,True),(2,0,.01,.01,False)])
    path,s=solve(d)
    assert path==[p[0],p[2],p[1]] and s['exit_index']==0
    assert s['rank'][:2]==[0.,4.]


def test_reverse_graph_does_not_invent_forward_connection():
    p=[(0.,0.),(4.,0.),(2.,0.)]
    d=graph(p,[(0,2,1.,1.,False),(1,2,1.,1.,True)])
    path,s=solve(d)
    assert not path and not s['budget_exhausted']


def test_contour_and_loss_are_not_rejected_by_old_height_veto():
    p=[(0.,0.),(5.,0.),(1.,1.),(3.,1.)]
    d=graph(p,[(0,2,1.,1.,False),(2,3,1.,1.,False),(3,1,1.,1.,True)])
    d['heights']=[0.,4.,2.,1.]
    path,s=solve(d)
    assert path==[p[0],p[2],p[3],p[1]] and s['complete']


@pytest.mark.parametrize('kind',['deadline','transitions'])
def test_partial_search_is_censored_not_executable_or_impossible(kind):
    p=[(0.,0.),(2.,0.)];d=graph(p,[(0,1,1.,1.,True)])
    if kind=='deadline':d['deadline']=time.perf_counter()-1.
    with patch.object(af,'MAX_TRANSITIONS',0 if kind=='transitions' else 100000):
        path,s=solve(d)
    assert not path and s['budget_exhausted'] and not s['optimal']


def test_segment_metric_is_split_invariant_and_nonascending_has_unit_factor():
    g=bt.ground();g.motion_turn_limit=15.
    a,b,c=(0.,0.),(.83,0.),(2.,0.)
    for reference in ((0.,4.),(4.,0.)):
        whole=af.leg_metrics(g,a,c,reference)
        parts=[af.leg_metrics(g,x,y,reference) for x,y in ((a,b),(b,c))]
        for key in ('ascent','exterior','length'):
            assert whole[key]==pytest.approx(sum(p[key] for p in parts))
        if reference[0]>reference[1]:assert whole['ascent']==whole['exterior']


def test_gentle_suffix_checks_whole_footprint_and_live_turn():
    g=bt.ground();g.motion_turn_limit=15.
    path=[(0.,0.),(2.,0.)]
    assert not af.route_reason(g,path,0.,0,path[0])
    g.slope[g.world_to_cell(1.,0.)]=math.tan(math.radians(20.))
    assert not af.leg_metrics(g,*path,(0.,4.))['gentle']
    assert af.route_reason(g,path,0.,0,path[0])
    with patch.object(af,'turn_reason',return_value='changed turn'):
        assert af.route_reason(g,path,math.pi/2,1,path[0])=='changed turn'
