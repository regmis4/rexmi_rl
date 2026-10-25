"""Independent shortest-path and exact-cost checks for turn-free map planning."""
import heapq
import math
import numpy as np
import pytest
import test_remote_control
from rexmi_bt_tests.nav import costmap_path as cp


def test_exact_integral_and_subdivision_invariance():
    cost=np.array([[2.],[10.],[4.]])
    allowed=np.ones_like(cost,bool)
    a,b,c=(.25,.5),(1.2,.5),(2.75,.5)
    assert cp.edge_cost(cost,allowed,a,c,1.)==14.5
    assert cp.edge_cost(cost,allowed,a,b,1.)+cp.edge_cost(cost,allowed,b,c,1.)==pytest.approx(14.5)


def test_corner_cutting_rejected_even_at_zero_length_contact():
    cost=np.ones((3,3));allowed=np.ones((3,3),bool);allowed[1,0]=False
    assert math.isinf(cp.edge_cost(cost,allowed,(.5,.5),(1.5,1.5),1.))
    allowed[1,0]=True
    assert cp.edge_cost(cost,allowed,(.5,.5),(1.5,1.5),1.)==pytest.approx(math.sqrt(2))


def test_dijkstra_matches_independent_eight_neighbor_search():
    cost=np.array([[1,2,3,4],[1,7,2,1],[1,5,2,1],[1,1,1,1]],float)
    allowed=np.ones_like(cost,bool);allowed[1,1]=False
    actual=cp.solve(cost,allowed,(.5,.5),np.array([0.,0.]),1.,radius=1)
    distances={(0,0):0.};queue=[(0.,(0,0))]
    while queue:
        value,a=heapq.heappop(queue)
        if value!=distances[a]:continue
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(-1,1),(1,-1)):
            b=(a[0]+dx,a[1]+dy)
            if not (0<=b[0]<4 and 0<=b[1]<4) or not allowed[b]:continue
            if dx and dy and not (allowed[a[0]+dx,a[1]] and allowed[a[0],a[1]+dy]):continue
            trial=value+math.hypot(dx,dy)*.5*(cost[a]+cost[b])
            if trial<distances.get(b,math.inf):
                distances[b]=trial;heapq.heappush(queue,(trial,b))
    for cell,value in distances.items():assert actual['distance'][cell]==pytest.approx(value)


def test_exact_endpoint_cost_and_complete_path_agree_without_turn_penalty():
    cost=np.full((6,6),2.);allowed=np.ones_like(cost,bool)
    start=np.array([.6,.7]);end=np.array([4.8,4.1])
    solution=cp.solve(cost,allowed,start,np.zeros(2),1.)
    path,value=cp.path_to_point(solution,cost,allowed,end)
    assert path[0]==tuple(start) and path[-1]==tuple(end)
    assert value==pytest.approx(sum(cp.edge_cost(cost,allowed,a,b,1.) for a,b in zip(path,path[1:])))
    assert value>=2.*np.linalg.norm(end-start)-1e-10


def test_unreachable_target_does_not_return_partial_route():
    cost=np.ones((7,7));allowed=np.ones_like(cost,bool);allowed[3,:]=False
    solution=cp.solve(cost,allowed,(1.5,1.5),np.zeros(2),1.)
    assert cp.path_to_cell(solution,(5,5))==[]
    assert cp.path_to_point(solution,cost,allowed,(5.5,5.5))==([],None)
