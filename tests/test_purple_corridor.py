"""Independent shortest-path checks for map-only bidirectional corridors."""
import math
from unittest.mock import patch
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import test_behavior_navigation as bt
from rexmi_bt_tests.nav import purple_corridor as pc


def matrix(edges,n=4):
    return csr_matrix(([w for a,b,w in edges],([a for a,b,w in edges],[b for a,b,w in edges])),shape=(n,n))


def test_transpose_preserves_outbound_direction_and_weights():
    g=matrix([(0,1,2),(1,3,2),(0,2,1),(2,3,8),(3,1,100),(1,0,99)])
    path,stats=pc.bidirectional(g,0,3)
    assert path==[0,1,3] and stats['cost']==4 and stats['optimal']
    assert pc.bidirectional(g,3,0)[1]['cost']==199


def test_first_meeting_is_not_a_valid_stopping_rule():
    g=matrix([(0,1,1),(1,3,10),(0,2,5),(2,3,5)])
    path,stats=pc.bidirectional(g,0,3)
    assert path==[0,2,3] and stats['cost']==10


def test_against_independent_scipy_directed_reference():
    rng=np.random.default_rng(42)
    for _ in range(12):
        weights=rng.uniform(.01,30,(20,20));mask=rng.random((20,20))<.15
        np.fill_diagonal(mask,False);g=csr_matrix(np.where(mask,weights,0.))
        path,stats=pc.bidirectional(g,0,19);expected=dijkstra(g,indices=0)[19]
        if math.isfinite(expected):
            assert stats['cost']==pytest.approx(expected)
            assert path[0]==0 and path[-1]==19
            assert sum(float(g[a,b]) for a,b in zip(path,path[1:]))==pytest.approx(expected)
        else:assert not path


def test_budget_and_invalid_weights_are_explicit():
    g=matrix([(0,1,1),(1,3,1)])
    path,s=pc.bidirectional(g,0,3,max_relaxations=0)
    assert not path and s['budget_exhausted'] and not s['optimal']
    with pytest.raises(ValueError):pc.bidirectional(matrix([(0,3,-1)]),0,3)


def test_fixed_height_factor_rewards_early_gain_without_negative_loops():
    assert pc.height_factor([-1,0,1,2,3],(0,2)).tolist()==[2,2,1.5,1,1]
    assert pc.height_factor([0,1],(2,0)).tolist()==[1,1]
    g=matrix([(0,1,1),(1,3,1),(0,2,1),(2,3,1)])
    heights=np.array([0.,2.,0.,2.]);factor=pc.height_factor(heights,(0.,2.))
    rows,cols=g.nonzero();weighted=csr_matrix((g.data*.5*(factor[rows]+factor[cols]),(rows,cols)),shape=g.shape)
    path,s=pc.bidirectional(weighted,0,3)
    assert path==[0,1,3] and s['cost']==2.5


def configured():
    g=bt.ground();g.uphill_limit_deg=33.;g.side_limit_deg=32.
    return g


def test_collinear_cost_does_not_change_when_split():
    g=configured();g.cost[:]=2.
    for reference in (None,(0.,2.)):
        full=pc.integral(g,(-1.,0.),(2.,0.),reference)
        split=pc.integral(g,(-1.,0.),(.13,0.),reference)+pc.integral(g,(.13,0.),(2.,0.),reference)
        assert full==pytest.approx(split)


def test_dense_graph_blocks_diagonal_corners_and_unknown_cells():
    g=configured();a=g.world_to_cell(0.,0.);b=(a[0]+1,a[1]+1)
    g.blocked[a[0]+1,a[1]]=True
    graph=pc.build(g,g.cell_to_world(*a),g.cell_to_world(a[0]+3,a[1]))
    n=g.n_cells
    assert graph['base'][a[0]*n+a[1],b[0]*n+b[1]]==0
    g.known[b]=False
    graph=pc.build(g,g.cell_to_world(*a),g.cell_to_world(a[0]+3,a[1]))
    assert not graph['allowed'][b]
    assert graph['stats']['directed_edges']<=pc.MAX_EDGES


def test_uphill_and_downhill_edges_are_not_confused():
    g=configured();g.gx[:]=math.tan(math.radians(34));g.gy[:]=0.
    # Preserve physically supported test geometry; only directional grade changes.
    a=g.world_to_cell(0.,0.);b=(a[0]+1,a[1]);n=g.n_cells
    graph=pc.build(g,g.cell_to_world(*a),g.cell_to_world(*b))
    i,j=a[0]*n+a[1],b[0]*n+b[1]
    assert graph['base'][i,j]==0 and graph['base'][j,i]>0
    assert np.array_equal(graph['base'].indptr,graph['early'].indptr)
    assert np.array_equal(graph['base'].indices,graph['early'].indices)


def test_two_pass_minimax_does_not_lose_best_secondary_route():
    # The cheaper prefix by maximum is NOT always better after a common ridge.
    peaks=matrix([(0,1,5),(1,3,10),(0,2,6),(2,3,10)])
    weights=matrix([(0,1,100),(1,3,1),(0,2,1),(2,3,1)])
    ceiling,stats=pc.minimum_ceiling(peaks,0,3)
    assert ceiling==10 and stats['certified']
    path,stats=pc.bidirectional(pc.restrict_ceiling(weights,peaks,ceiling),0,3)
    assert path==[0,2,3] and stats['cost']==2


def test_uncertified_ceiling_never_becomes_an_optimal_threshold():
    limit,stats=pc.minimum_ceiling(matrix([(0,3,2)]),0,3,max_relaxations=0)
    assert limit is None and not stats['certified'] and stats['budget_exhausted']


def test_connector_peak_includes_endpoint_even_when_length_is_zero():
    g=configured();a=g.cell_to_world(*g.world_to_cell(0.,0.));g.cost[g.world_to_cell(*a)]=17.
    assert pc.integral(g,a,a,peak=True)==17.
    b=(a[0]+g.cell_size,a[1]);assert pc.integral(g,b,a,peak=True)==17.
    graph=pc.build(g,a,b)
    assert np.array_equal(graph['base'].indices,graph['ceiling'].indices)
    assert graph['ceiling'].getrow(graph['start']).data.min()==17.
