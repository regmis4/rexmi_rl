"""Offline guard attribution. Never imported by the navigation controller.

One immutable candidate/edge universe; cumulative independent guard predicates.
Cheap witnesses establish connectivity. History-aware optimization keeps its
complete incumbent if censored; it does not authorize physical execution.
"""
import heapq
import math
import time
from collections import Counter, defaultdict, deque

import numpy as np
from scipy.spatial import cKDTree

from . import sustained_climb as sc, stage_graph
from .staged_routes import observed_edge_reason
from .swept_footprint import oriented_swept_cells
from .turn_staging import turn_reason

LEVELS = ('Finite terrain only', '+ Swept obstacle clearance', '+ 35 degree body slope',
          '+ 33 uphill / 32 cross slope', '+ Existing edge contracts', '+ Checked turns')
MAX_NODES = 512
MAX_EDGES = 16770


def inventory(g, start, goal, field):
    legacy, provenance = stage_graph.stable_nodes(g, start, goal, g.planning_candidate_source)
    points = list(legacy)
    stride = max(1, round(1.6 / g.cell_size))
    neutral = []
    for i in range(0, g.n_cells, stride):
        for j in range(0, g.n_cells, stride):
            p = tuple(g.origin + (np.array([i, j]) + .5)*g.cell_size - g.world_size/2)
            if g.known[i,j] and np.isfinite(g.cost[i,j]) and g.cost[i,j] > 0 and math.isfinite(
                    sc.cb.height_at(field, np.array([i+.5, j+.5]))):
                if all(math.dist(p, old) > 1e-8 for old in legacy):
                    neutral.append(p)
    # Deterministic spatial sampling, fixed before scoring or route outcomes.
    available = MAX_NODES - len(legacy)
    if len(neutral) > available:
        neutral = [neutral[i] for i in np.linspace(0, len(neutral)-1, available, dtype=int)]
    points.extend(neutral)
    nlegacy = len(legacy)
    pairs = {(i,j) for i in range(nlegacy) for j in range(nlegacy) if i != j}
    # Sparse local links also connect new anchors to the retained candidates.
    for i,j in cKDTree(points).query_pairs(1.6*math.sqrt(2)+1e-8):
        pairs.add((i,j)); pairs.add((j,i))
    if len(pairs) > MAX_EDGES:
        raise ValueError('frozen edge inventory exceeds declared bound')
    return points, sorted(pairs), dict(legacy_count=nlegacy, neutral_count=len(neutral),
        total_nodes=len(points), directed_pairs=len(pairs), grid_spacing_m=stride*g.cell_size,
        legacy_provenance=provenance,
        limitation='Finite sampled anchors and sparse straight links; not continuous-space optimality')


def edge_reasons(g, a, b, start, physical_check):
    heading = math.atan2(b[1]-a[1], b[0]-a[0])
    cells = oriented_swept_cells(g, a, b, heading)
    known = cells is not None and g.known[cells].all()
    obstacle = ('unobserved oriented footprint' if not known else
                'obstacle in oriented footprint' if g.hazard[cells].any() else '')
    slope = ('unobserved oriented slope' if not known else
             'oriented footprint exceeds 35 degrees' if not np.isfinite(g.slope[cells]).all()
             or (g.slope[cells] > math.tan(math.radians(35))).any() else '')
    grade = g.grade_reason(a, b, uphill_limit=33., side_limit=32.)
    existing = observed_edge_reason(g, a, b, start) or physical_check(a, b)
    return [str(obstacle), str(slope), str(grade), str(existing)]


def build(g, start, goal, physical_check, deadline, progress=lambda _: None):
    field = sc.cb.height_field(g)
    points, pairs, info = inventory(g, start, goal, field)
    edges = {}; unavailable = Counter()
    for k, (i,j) in enumerate(pairs):
        if time.perf_counter() >= deadline:
            raise TimeoutError('shared feature build exceeded overall wall budget')
        a,b = points[i], points[j]
        try:
            runs = sc.runs(sc.intervals(g, field, a, b))
        except ValueError as exc:
            unavailable[str(exc)] += 1
            continue
        edges[i,j] = dict(runs=runs, length=math.dist(a,b),
            effort=math.fsum(r.total for r in runs),
            heading=math.atan2(b[1]-a[1],b[0]-a[0]),
            reasons=edge_reasons(g,a,b,start,physical_check))
        if k % 500 == 0: progress(dict(phase='shared features', processed=k, total=len(pairs)))
    info.update(objective_supported_edges=len(edges), objective_unavailable=dict(unavailable))
    return dict(points=points, edges=edges, inventory=info, deadline=deadline,
                relief_distance=2*g.footprint_radius, turn_cache={})


def admitted(edge, level):
    return not any(edge['reasons'][:min(level,4)])


def transitions(g, graph, yaw, level, legacy_only=False):
    nlegacy=graph['inventory']['legacy_count']
    edges={s:e for s,e in graph['edges'].items() if admitted(e,level)
           and (not legacy_only or max(s)<nlegacy)}
    outgoing=defaultdict(list)
    for i,j in edges: outgoing[i].append((i,j))
    forward=defaultdict(list); reverse=defaultdict(list); rejected=Counter(); checks=0
    for state in [(-1,0)]+list(edges):
        if state[1]==1: continue
        incoming=yaw if state[0]<0 else edges[state]['heading']
        for target in outgoing[state[1]]:
            if time.perf_counter() >= graph['deadline']:
                raise TimeoutError('heading transitions exceeded overall wall budget')
            checks += 1
            reason=''
            if level >= 5:
                key=(state,target)
                if key not in graph['turn_cache']:
                    heading=edges[target]['heading']
                    delta=abs(math.atan2(math.sin(heading-incoming),math.cos(heading-incoming)))
                    graph['turn_cache'][key]=(turn_reason(g,graph['points'][state[1]],incoming,heading)
                                              if delta>1e-6 else '')
                reason=graph['turn_cache'][key]
            if reason: rejected[reason]+=1; continue
            forward[state].append(target); reverse[target].append(state)
    # Remove dead ends ONCE, before repeated label expansions.
    reachable={s for s in edges if s[1]==1}; todo=deque(sorted(reachable)); successor={}
    while todo:
        state=todo.popleft()
        for parent in reverse[state]:
            if parent not in reachable:
                reachable.add(parent); successor[parent]=state; todo.append(parent)
    kept={s:[t for t in targets if t in reachable] for s,targets in forward.items() if s in reachable}
    return kept, edges, successor, dict(edges=len(edges), transition_checks=checks,
        turn_rejections=dict(rejected), goal_reachable_states=len(reachable),
        pruned_dead_end_transitions=sum(len(v) for v in forward.values())-sum(len(v) for v in kept.values()),
        initial_reachable=(-1,0) in reachable)


def evaluate(graph, states):
    state=sc.State(); length=0.
    for s in states[1:]:
        edge=graph['edges'][s]
        state,_=sc.advance(state,edge['runs'],graph['relief_distance'])
        length+=edge['length']
    return (state.score,length)


def cheap_witnesses(graph, forward, successor, on_witness=lambda _: None):
    origin=(-1,0)
    if origin not in successor: return []
    chain=[origin]
    while chain[-1][1]!=1: chain.append(successor[chain[-1]])
    result=[dict(states=chain,rank=evaluate(graph,chain),provenance='reverse reachability witness')]
    on_witness(result[0])
    for mode in ('effort','length'):
        q=[((0.,0.),origin)]; best={origin:(0.,0.)}; parent={}
        while q:
            if time.perf_counter()>=graph['deadline']:
                return result  # Feasibility survives a censored optional refinement.
            value,s=heapq.heappop(q)
            if value!=best[s]:continue
            if s[1]==1:
                chain=[s]
                while chain[-1]!=origin:chain.append(parent[chain[-1]])
                chain.reverse()
                result.append(dict(states=chain,rank=evaluate(graph,chain),provenance='minimum '+mode+' witness'))
                on_witness(result[-1])
                break
            for t in forward.get(s,()):
                edge=graph['edges'][t]
                trial=(value[0]+edge[mode], value[1]+edge['length'])
                if trial<best.get(t,(math.inf,math.inf)):
                    best[t]=trial;parent[t]=s;heapq.heappush(q,(trial,t))
    return result


def improve(graph, forward, incumbent, *, max_relaxations=200000, max_labels=60000):
    """Exact conservative dominance; complete incumbent survives censorship."""
    initial=(-1,0); labels=[dict(heading=initial,state=sc.State(),length=0.,parent=None,live=True)]
    groups=defaultdict(list);groups[initial]=[0];q=[((0.,0.),0)]
    stats=dict(optimal=False,censored=False,relaxations=0,labels=1,expansions=0,improvements=0)
    rank=tuple(incumbent['rank']); witness=dict(incumbent)
    def finish(reason,censored=False):
        stats.update(reason=reason,censored=censored,rank=list(rank))
        return witness,stats
    while q:
        if time.perf_counter()>=graph['deadline']:
            return finish('overall wall budget; complete incumbent retained',True)
        value,index=heapq.heappop(q); label=labels[index]
        if not label['live']:continue
        if value>=rank:
            stats['optimal']=True
            return finish('incumbent certified minimum in this sampled heading graph')
        stats['expansions']+=1
        for target in forward.get(label['heading'],()):
            if stats['relaxations']>=max_relaxations or len(labels)>=max_labels:
                return finish('work budget; complete incumbent retained',True)
            stats['relaxations']+=1
            edge=graph['edges'][target]
            state,_=sc.advance(label['state'],edge['runs'],graph['relief_distance'])
            candidate=dict(heading=target,state=state,length=label['length']+edge['length'],parent=index,live=True)
            candidate_rank=(state.score,candidate['length'])
            if candidate_rank>=rank:continue
            if target[1]==1:
                states=[target];cursor=index
                while cursor is not None:
                    row=labels[cursor];states.append(row['heading']);cursor=row['parent']
                states.reverse();rank=candidate_rank
                witness=dict(states=states,rank=rank,provenance='sustained-objective label improvement')
                stats['improvements']+=1
                continue
            if any(sc.dominates(labels[i],candidate) for i in groups[target]):continue
            kept=[]
            for i in groups[target]:
                if sc.dominates(candidate,labels[i]):labels[i]['live']=False
                else:kept.append(i)
            idx=len(labels);labels.append(candidate);groups[target]=kept+[idx]
            stats['labels']=len(labels);heapq.heappush(q,(candidate_rank,idx))
    stats['optimal']=True
    return finish('all potentially better sampled labels exhausted')


def audit(g, graph, states, yaw):
    """Report every omitted check at every segment/turn, not first rejection."""
    violations=[]; incoming=yaw
    for index,s in enumerate(states[1:]):
        edge=graph['edges'][s];a,b=(graph['points'][i] for i in s)
        for group,reason in enumerate(edge['reasons'],1):
            if reason:violations.append(dict(group=group,segment=index,start=a,end=b,reason=reason))
        heading=edge['heading'];delta=abs(math.atan2(math.sin(heading-incoming),math.cos(heading-incoming)))
        reason=turn_reason(g,a,incoming,heading) if delta>1e-6 else ''
        if reason:violations.append(dict(group=5,segment=index,point=a,reason=reason))
        incoming=heading
    return violations
