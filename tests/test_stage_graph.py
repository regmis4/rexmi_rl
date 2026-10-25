"""Meaningful checkpoint search: mission progress never becomes failure evidence."""
import copy
import math
import time
from unittest.mock import patch
import pytest
import test_behavior_navigation as bt
from rexmi_bt_tests.nav import stage_graph as sg


def graph():
    points=[(0.,0.),(3.,0.),(1.,0.),(2.,0.)]
    edges={}
    for i,j,cost in ((0,2,1.),(2,3,1.),(3,1,1.),(2,1,5.)):
        edges[i,j]=dict(heading=0.,total=cost,uphill=cost)
    return dict(points=points,heights=[0.,2.,1.,.5],edges=edges,turns={},deadline=time.perf_counter()+30.)


@pytest.mark.parametrize('uphill_first',[False,True])
def test_progress_filter_removes_giveback_without_marking_terrain_failed(uphill_first):
    g=bt.ground();data=graph();failures=g.failures.copy()
    with patch.object(sg,'turn_reason',return_value=''),patch.object(sg,'turn_cost',return_value=0.):
        old,_=sg.search(g,data,0.,uphill_first=uphill_first)
        new,stats=sg.search(g,data,0.,uphill_first=uphill_first,preserve_height=True)
    assert old==[(0.,0.),(1.,0.),(2.,0.),(3.,0.)]
    assert new==[(0.,0.),(1.,0.),(3.,0.)]
    assert stats['progress_rejected_links']==1
    assert (g.failures==failures).all()


def test_rim_above_goal_can_descend_toward_exact_destination():
    g=bt.ground();data=graph();data['heights']=[0.,2.,3.,2.5]
    with patch.object(sg,'turn_reason',return_value=''),patch.object(sg,'turn_cost',return_value=0.):
        path,stats=sg.search(g,data,0.,preserve_height=True)
    assert path==[(0.,0.),(1.,0.),(2.,0.),(3.,0.)]
    assert stats['progress_rejected_links']==0


def test_missing_height_and_actual_turn_rejection_cannot_be_overridden_by_cost():
    g=bt.ground();data=graph();data['heights'][2]=math.nan
    with patch.object(sg,'turn_reason',return_value=''):
        assert not sg.search(g,data,0.,preserve_height=True)[0]
    data=graph()
    with patch.object(sg,'turn_reason',return_value='swept body crosses cliff'):
        assert not sg.search(g,data,0.)[0]


def test_budget_censoring_is_not_infeasibility_or_fabricated_route():
    g=bt.ground();data=graph();data['deadline']=time.perf_counter()-1.
    path,stats=sg.search(g,data,0.)
    assert not path and stats['budget_exhausted'] and not stats['complete']


def test_footprint_mean_requires_complete_observed_support():
    g=bt.ground();p=(0.,0.)
    assert math.isfinite(sg.mean_height(g,p))
    g.known[g.world_to_cell(*p)]=False
    assert math.isnan(sg.mean_height(g,p))


@pytest.mark.parametrize('profile,allowed',[
    ([1.,1.5,1.6],True),([1.,.95,1.1],True),([1.,.5,1.6],False),
    ([1.,1.5,1.25,1.6],False),([1.,1.5,1.3],False),([1.,math.nan,1.6],False)])
def test_full_leg_retention_catches_endpoint_only_loophole(profile,allowed):
    assert sg.retained_height(profile,4.,.20) is allowed


def test_retention_clips_at_goal_and_bounds_losses_across_joins():
    assert sg.retained_height([4.,5.,4.2],4.,.20)
    first=[1.,1.2,1.1];second=[1.1,1.,1.3]
    assert sg.retained_height(first,4.,.20) and sg.retained_height(second,4.,.20)
    assert max(first)-min(second)==pytest.approx(.20)
