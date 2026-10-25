import heapq
import math
import time
from unittest.mock import patch
import numpy as np
import pytest
import test_behavior_navigation as bt
from rexmi_bt_tests.nav import climb_burden as cb, ascent_first as af


def terrain(fn=lambda x,y:x,cost=2.):
    g=bt.ground();axis=g.origin[0]+(np.arange(g.n_cells)+.5)*g.cell_size-g.world_size/2
    x,y=np.meshgrid(axis,axis,indexing='ij');g.surface[:]=fn(x,y);g.cost[:]=cost
    return g,x,y


def test_plane_uphill_downhill_flat_and_reclimb():
    g,_,_=terrain(lambda x,y:.5*x);field=cb.height_field(g)
    assert cb.metric(g,field,(0.,0.),(2.,0.))['burden']==pytest.approx(2.)
    assert cb.metric(g,field,(2.,0.),(0.,0.))['burden']==pytest.approx(0.,abs=1e-12)
    assert cb.metric(g,field,(0.,0.),(0.,2.))['burden']==pytest.approx(0.,abs=1e-12)
    r=cb.describe_route(g,[(0.,0.),(2.,0.),(0.,0.),(2.,0.)])
    assert r['rank'][0]==pytest.approx(4.) and r['positive_gain_m']==pytest.approx(2.)
    assert r['height_loss_m']==pytest.approx(1.)


def test_quadratic_interior_hump_and_collinear_partition_invariance():
    g,x,y=terrain(lambda x,y:x*y,cost=1.);field=x*y
    a,b,c=(0.,1.),(.371,.629),(1.,0.)
    assert cb.metric(g,field,a,c)['burden']==pytest.approx(.25)
    assert cb.metric(g,field,a,b)['burden']+cb.metric(g,field,b,c)['burden']==pytest.approx(.25)
    assert cb.metric(g,field,c,a)['burden']==pytest.approx(.25)


def test_cost_crossing_and_interpolation_crossing_are_both_split():
    g,x,y=terrain();g.cost[:]=np.where(x<1.,2.,4.);field=cb.height_field(g)
    assert cb.metric(g,field,(0.,0.),(2.,0.))['burden']==pytest.approx(6.)
    for cut in (.137,.5,1.,1.319):
        assert sum(cb.metric(g,field,a,b)['burden'] for a,b in [((0.,0.),(cut,0.)),((cut,0.),(2.,0.))])==pytest.approx(6.)


def test_all_four_interpolation_supports_required_and_no_unknown_as_zero():
    g,_,_=terrain();g.known[g.world_to_cell(.2,.2)]=False;field=cb.height_field(g)
    with pytest.raises(ValueError,match='height support'):cb.metric(g,field,(0.,0.),(1.,0.))
    field=np.zeros_like(g.surface,dtype=float);i,j=g.world_to_cell(0.,0.);field[i,j]=np.nan
    assert math.isnan(cb.height_at(field,np.array([i+.5,j+.5])))


def test_bands_charge_repeated_gain_and_partial_window_is_labelled():
    g,_,_=terrain(lambda x,y:.25*x,cost=3.)
    r=cb.describe_route(g,[(0.,0.),(1.,0.)])
    assert r['worst_gain_window']['partial']
    assert r['worst_gain_window']['positive_gain_m']==pytest.approx(.25)
    r=cb.describe_route(g,[(0.,0.),(4.,0.),(0.,0.),(4.,0.)])
    assert sum(b['positive_gain_m'] for b in r['bands'])==pytest.approx(2.)
    assert r['worst_gain_window']['burden']==pytest.approx(3.)


def graph(n,links):
    points=[(float(i),float(i%2)) for i in range(n)]
    edges={(i,j):dict(burden=float(c),length=math.dist(points[i],points[j]),
        heading=math.atan2(points[j][1]-points[i][1],points[j][0]-points[i][0])) for i,j,c in links}
    return dict(points=points,edges=edges,turns={},deadline=time.perf_counter()+30.)


def reference(data):
    best={0:(0.,0.)};q=[((0.,0.),0)]
    while q:
        value,node=heapq.heappop(q)
        if value!=best[node]:continue
        if node==1:return value
        for (i,j),edge in data['edges'].items():
            if i!=node:continue
            trial=(value[0]+edge['burden'],value[1]+edge['length'])
            if trial<best.get(j,(math.inf,math.inf)):
                best[j]=trial;heapq.heappush(q,(trial,j))
    return None


def solve(data):
    with patch.object(af,'turn_reason',return_value=''):
        return cb.search(bt.ground(),data,0.)


def test_bidirectional_matches_independent_single_source_lex_search():
    rng=np.random.default_rng(42)
    for _ in range(30):
        links=[(i,j,int(rng.integers(0,9))) for i in range(7) for j in range(7) if i!=j and rng.random()<.3]
        data=graph(7,links);path,stats=solve(data);expected=reference(data)
        if expected is None:assert not path
        else:
            assert stats['optimal'] and stats['rank']==pytest.approx(expected)
            states=stats['path_states'];total=[0.,0.]
            for state in states[1:]:
                edge=data['edges'][tuple(state)];total[0]+=edge['burden'];total[1]+=edge['length']
            assert total==pytest.approx(stats['rank'])


def test_first_meeting_reverse_asymmetry_and_flat_loops():
    data=graph(4,[(0,2,1),(2,1,10),(0,3,5),(3,1,5),(1,2,100),(2,0,0),(0,1,20)])
    path,stats=solve(data)
    assert path==[data['points'][0],data['points'][3],data['points'][1]] and stats['rank'][0]==10.
    data=graph(4,[(0,2,0),(2,0,0),(2,1,0),(0,3,0),(3,1,0)])
    path,stats=solve(data);assert len(path)==3 and stats['rank']==pytest.approx(reference(data))


def test_join_heading_cannot_be_ignored_and_budget_fails_closed():
    data=graph(3,[(0,2,1),(2,1,1)])
    with patch.object(af,'turn_reason',return_value='invalid joining turn'):
        path,stats=cb.search(bt.ground(),data,0.)
    assert not path
    with patch.object(cb,'MAX_RELAXATIONS',0):
        path,stats=solve(data)
    assert not path and stats['budget_exhausted'] and not stats['optimal']
