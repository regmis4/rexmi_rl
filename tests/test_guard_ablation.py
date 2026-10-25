import math
import time
from unittest.mock import patch

import pytest
import numpy as np
from test_sustained_climb import graph as fixture_graph, item
from test_climb_burden import terrain
from rexmi_bt_tests.nav import guard_ablation as ga, sustained_climb as sc


def data(links):
    graph,(forward,reverse,_) = fixture_graph(links)
    graph['inventory']={'legacy_count':len(graph['points'])}
    for edge in graph['edges'].values():
        edge.update(reasons=['','','',''],effort=sum(r.total for r in edge['runs']),heading=0.)
    successor={};todo=[s for s in graph['edges'] if s[1]==1];seen=set(todo)
    for s in todo:
        for p in reverse[s]:
            if p not in seen:successor[p]=s;todo.append(p);seen.add(p)
    return graph,forward,successor


def test_cumulative_masks_cannot_reintroduce_rejected_edge():
    for index in range(4):
        edge={'reasons':['']*4};edge['reasons'][index]='failed guard'
        assert all(ga.admitted(edge,level) for level in range(index+1))
        assert not any(ga.admitted(edge,level) for level in range(index+1,6))


def test_complete_incumbent_survives_budget_and_is_not_called_optimal():
    graph,forward,successor=data([(0,2,2,False),(2,1,3,False)])
    witness=min(ga.cheap_witnesses(graph,forward,successor),key=lambda w:w['rank'])
    result,stats=ga.improve(graph,forward,witness,max_relaxations=0)
    assert result['states']==witness['states']
    assert result['rank']==pytest.approx([25.,2.])
    assert stats['censored'] and not stats['optimal']


def test_reverse_witness_persisted_before_optional_search_timeout():
    graph,forward,successor=data([(0,2,2,False),(2,1,3,False)])
    graph['deadline']=time.perf_counter()-1
    saved=[]
    result=ga.cheap_witnesses(graph,forward,successor,on_witness=saved.append)
    assert len(saved)==len(result)==1
    assert saved[0]['states'][-1][1]==1 and saved[0]['rank']==pytest.approx([25.,2.])


def test_history_dependent_objective_improves_seed_without_losing_alternative():
    graph,forward,_=data([(0,2,2,False),(0,3,math.sqrt(5),False),(3,2,0,True),
                          (2,4,0,False),(4,1,3,False)])
    states=[(-1,0),(0,2),(2,4),(4,1)]
    witness=dict(states=states,rank=ga.evaluate(graph,states),provenance='test')
    result,stats=ga.improve(graph,forward,witness)
    assert result['rank']==pytest.approx([14.,4.]) and stats['optimal']
    assert [s[1] for s in result['states']]==[0,3,2,4,1]


def test_heading_guard_includes_initial_turn_and_dead_ends_pruned_once():
    graph,_,_=data([(0,2,1,False),(2,1,1,False),(0,3,0,False)])
    graph['turn_cache']={}
    with patch.object(ga,'turn_reason',return_value='turn blocked'):
        forward,_,_,stats=ga.transitions(None,graph,1.,5)
    assert not stats['initial_reachable'] and (-1,0) not in forward
    forward,_,successor,stats=ga.transitions(None,graph,1.,4)
    assert stats['initial_reachable'] and stats['pruned_dead_end_transitions']>0
    assert (0,3) not in forward[(-1,0)]
    assert ga.cheap_witnesses(graph,forward,successor)


def test_inventory_preserves_legacy_edges_and_neutral_blocked_candidates():
    g,_,_=terrain(lambda x,y:0*x)
    g.planning_candidate_source=None
    g.blocked[:]=True;g.hazard[:]=True
    legacy=[(0.,0.),(2.,0.),(.7,1.3)]
    with patch.object(ga.stage_graph,'stable_nodes',return_value=(legacy,{})):
        points,pairs,stats=ga.inventory(g,legacy[0],legacy[1],sc.cb.height_field(g))
    assert points[:3]==legacy
    assert all((i,j) in pairs for i in range(3) for j in range(3) if i!=j)
    assert stats['neutral_count']>0 and len(points)<=512 and len(pairs)<=16770


def test_random_dags_match_existing_exact_solver():
    rng=np.random.default_rng(99);order=[0,2,3,4,5,1]
    for _ in range(25):
        links=[(a,b,float(rng.integers(0,5)),bool(rng.integers(0,2)))
               for i,a in enumerate(order) for b in order[i+1:] if rng.random()<.55]
        if not links:continue
        graph,forward,successor=data(links)
        witnesses=ga.cheap_witnesses(graph,forward,successor)
        if not witnesses:continue
        result,stats=ga.improve(graph,forward,min(witnesses,key=lambda w:w['rank']))
        _,transitions=fixture_graph(links)
        with patch.object(sc.ascent_first,'transitions',return_value=transitions):
            _,reference=sc.search(None,graph,0.)
        assert stats['optimal'] and result['rank']==pytest.approx(reference['rank'])
