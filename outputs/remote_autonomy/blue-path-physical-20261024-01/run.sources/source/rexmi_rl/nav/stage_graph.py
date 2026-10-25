"""Bounded terrain-derived checkpoints and complete straight-leg connections.

No route demonstrations or coordinate templates. Elevation/azimuth bins select
coverage representatives, not forbidden regions or prescribed travel order.
"""
import copy
import heapq
import math
import time
from collections import Counter
import numpy as np
from .turn_staging import minimum_fields, stage_reason, turn_reason
from .staged_routes import observed_edge_reason
from .traversal_objective import segment_cost, turn_cost

MAX_NODES=130
MAX_LINKS=MAX_NODES*(MAX_NODES-1)
MAX_TRANSITIONS=100000


def mean_height(g,point):
    cells=g._support_cells(point)
    if cells is None or not g.known[cells].all() or not np.isfinite(g.surface[cells]).all():
        return math.nan
    return float(np.mean(g.surface[cells]))


def height_profiles(g,graph):
    """Sample complete supported footprints; geometry links stay unchanged."""
    for (i,j),edge in graph['edges'].items():
        if time.perf_counter()>=graph['deadline']:
            graph['profile_budget_exhausted']=True
            return False
        a,b=graph['points'][i],graph['points'][j]
        steps=max(1,math.ceil(math.dist(a,b)/min(.2,g.cell_size)))
        edge['height_profile']=[mean_height(g,np.asarray(a)+t*np.subtract(b,a))
            for t in np.linspace(0.,1.,steps+1)]
    return True


def retained_height(profile,goal_height,budget):
    """Explicit sampled mission preference, never a physical no-go label.

    Half-budget margins at leg boundaries also bound drawdown across joins.
    Terrain above the destination elevation does not owe further climbing.
    """
    heights=np.minimum(np.asarray(profile),goal_height)
    if not len(heights) or not np.isfinite(heights).all():return False
    return bool(heights.min()>=heights[0]-budget/2-1e-8
        and heights[-1]>=heights.max()-budget/2-1e-8
        and np.max(np.maximum.accumulate(heights)-heights)<=budget+1e-8)


def route_height_reason(g,path,budget=.20):
    if len(path)<2:return 'height-retaining route is empty'
    goal_height=mean_height(g,path[-1])
    if not math.isfinite(goal_height):return 'destination footprint height is unobserved'
    for a,b in zip(path,path[1:]):
        steps=max(1,math.ceil(math.dist(a,b)/min(.2,g.cell_size)))
        profile=[mean_height(g,np.asarray(a)+t*np.subtract(b,a)) for t in np.linspace(0,1,steps+1)]
        if not retained_height(profile,goal_height,budget):
            return 'sampled ascent-height retention changed'
        if min(profile[-1],goal_height)<min(profile[0],goal_height)-1e-8:
            return 'ascent checkpoint gives back elevation'
    return ''


def nodes(g,start,goal,*,unrestricted_height=False):
    _,score,_=minimum_fields(g)
    z0,z1=(float(g.surface[g.world_to_cell(*p)]) for p in (start,goal))
    low,high=sorted((z0,z1))
    available=g.known & ~g.blocked & g.planning_support() & np.isfinite(score)
    if unrestricted_height:
        measured=g.surface[available & np.isfinite(g.surface)]
        if len(measured):low,high=float(measured.min()),float(measured.max())
    else:
        available &= (g.surface>=low-.1)&(g.surface<=high+.1)
    cells=np.argwhere(available)
    positions=g.origin+(cells+.5)*g.cell_size-g.world_size/2
    angles=np.arctan2(positions[:,1]-g.origin[1],positions[:,0]-g.origin[0])
    sectors=np.clip(((angles+math.pi)/(2*math.pi)*16).astype(int),0,15)
    heights=np.clip(((g.surface[tuple(cells.T)]-low)/max(high-low,.1)*4).astype(int),0,3)
    result=[tuple(start),tuple(goal)]
    separation=2*g.footprint_radius
    for height in range(4):
        for sector in range(16):
            ids=np.flatnonzero((heights==height)&(sectors==sector))
            if not len(ids):continue
            # Lowest footprint inclination first; cost and cell index break ties.
            candidates=sorted(ids,key=lambda i:(float(score[tuple(cells[i])]),float(g.cost[tuple(cells[i])]),tuple(cells[i])))
            count=0
            for i in candidates:
                point=tuple(positions[i])
                if any(math.dist(point,p)<separation for p in result):continue
                if stage_reason(g,point):continue
                result.append(point);count+=1
                if count==2:break
    assert len(result)<=MAX_NODES
    return result


SOURCE_FIELDS=('surface','slope','known','blocked','cost','hazard')
SOURCE_CACHES=('_turn_minimum_key','_turn_minimum_fields','_route_stage_key','_route_stage_candidates')


def freeze_candidate_source(g):
    """Immutable survey geometry for locations only; never authorizes a link."""
    source=copy.copy(g)
    source.origin=np.array(g.origin,copy=True);source.origin.setflags(write=False)
    for name in SOURCE_FIELDS:
        value=getattr(g,name).copy();value.setflags(write=False)
        setattr(source,name,value)
    for name in SOURCE_CACHES:source.__dict__.pop(name,None)
    source.terrain_prior=copy.deepcopy(getattr(g,'terrain_prior',None))
    source.failure_records=copy.deepcopy(getattr(g,'failure_records',[]))
    return source


def stable_nodes(g,start,goal,source,*,unrestricted_height=False):
    """Retain valid survey representatives before admitting live replacements."""
    options={'unrestricted_height':True} if unrestricted_height else {}
    if source is None:return nodes(g,start,goal,**options),{'source':'live only'}
    prior=nodes(source,start,goal,**options)[2:]
    live=nodes(g,start,goal,**options)[2:]
    result=[tuple(start),tuple(goal)];counts=Counter()
    support=g.known & ~g.blocked & g.planning_support()
    lo,hi=sorted(float(g.surface[g.world_to_cell(*p)]) for p in (start,goal))
    for kind,candidates in (('survey',prior),('live',live)):
        pending=list(candidates)
        while pending:
            # Farthest supported live additions spread remaining capacity;
            # original survey order stays stable and has retention priority.
            index=(max(range(len(pending)),key=lambda i:min(math.dist(pending[i],p) for p in result))
                if kind=='live' else 0)
            point=pending.pop(index)
            cell=g.world_to_cell(*point)
            if (cell is None or not support[cell]
                    or (not unrestricted_height and not lo-.1<=float(g.surface[cell])<=hi+.1) or stage_reason(g,point)):
                counts[kind+'_rejected_current']+=1;continue
            if any(math.dist(point,p)<2*g.footprint_radius for p in result):
                counts[kind+'_covered']+=1;continue
            if len(result)>=MAX_NODES:
                counts[kind+'_omitted_capacity']+=1;continue
            result.append(point);counts[kind+'_retained']+=1
    return result,dict(source='validated survey candidates with live validation',
        survey_candidates=len(prior),live_candidates=len(live),**counts)


def build(g,start,goal,physical_check,*,fresh_radius=2.,wall_budget=180.,candidate_source=None,cost_model=None,terminals=None):
    if terminals is not None and not 1<=len(terminals)<=64:
        raise ValueError('Multi-exit graph requires 1 to 64 terminal candidates')
    began=time.perf_counter();points,inventory=stable_nodes(g,start,goal,candidate_source,
        unrestricted_height=terminals is not None)
    goal_indices=[1]
    if terminals is not None:
        goal_indices=[]
        for point in terminals:
            point=tuple(point)
            index=next((i for i,p in enumerate(points) if math.dist(p,point)<1e-8),None)
            if index is None:index=len(points);points.append(point)
            goal_indices.append(index)
        inventory.update(terminal_candidates=len(terminals),unrestricted_candidate_height=True)
    deadline=began+wall_budget
    edges={};rejections=Counter();checks=0;exhausted=False
    for i,a in enumerate(points):
        for j,b in enumerate(points):
            if i==j:continue
            if time.perf_counter()>=deadline:
                exhausted=True;break
            checks+=1
            reason=observed_edge_reason(g,a,b,start,fresh_radius=fresh_radius)
            if not reason:reason=physical_check(a,b)
            if reason:
                rejections[reason]+=1;continue
            parts=(segment_cost(g,a,b,breakdown=True) if cost_model is None else cost_model(g,a,b))
            if not all(math.isfinite(v) for v in parts.values()):
                rejections['nonfinite traversal cost']+=1;continue
            edges[i,j]=dict(heading=math.atan2(b[1]-a[1],b[0]-a[0]),
                total=sum(parts.values()),uphill=parts.get('uphill',0.),components=parts)
        if exhausted:break
    assert len(points)<=MAX_NODES+(64 if terminals is not None else 0)
    assert checks<=len(points)*(len(points)-1)
    return dict(points=points,inventory=inventory,heights=[mean_height(g,p) for p in points],edges=edges,turns={},build_seconds=time.perf_counter()-began,
        directed_checks=checks,rejections=dict(rejections),transition_budget=MAX_TRANSITIONS,
        deadline=deadline,build_budget_exhausted=exhausted,goal_indices=goal_indices)


def search(g,graph,yaw,*,uphill_first=False,preserve_height=False,height_budget=None):
    began=time.perf_counter();points,edges=graph['points'],graph['edges']
    goals=set(graph.get('goal_indices',(1,)))
    if goals!={1} and (preserve_height or height_budget is not None):
        raise ValueError('Goal-region search cannot use single-destination height retention')
    zero=(0.,0.) if uphill_first else 0.
    infinity=(math.inf,math.inf) if uphill_first else math.inf
    best={(-1,0):zero};parent={};queue=[(zero,(-1,0))]
    adjacency={i:[] for i in range(len(points))}
    heights=np.asarray(graph['heights'])
    potential=np.maximum(0.,heights[1]-heights)
    rejected_progress=0
    for i,j in edges:
        if preserve_height and (not np.isfinite(potential[[i,j]]).all() or potential[j]>potential[i]+1e-8):
            rejected_progress+=1;continue
        if height_budget is not None and not retained_height(edges[i,j].get('height_profile',[]),heights[1],height_budget):
            rejected_progress+=1;continue
        adjacency[i].append(j)
    transitions=0;expanded=0;end=None;incumbent=None;exhausted=False
    while queue:
        if time.perf_counter()>=graph['deadline'] or expanded>=60000:
            exhausted=True;break
        value,state=heapq.heappop(queue)
        if value!=best[state]:continue
        previous,current=state
        if current in goals:end=state;break
        expanded+=1
        incoming=yaw if previous<0 else edges[previous,current]['heading']
        for nxt in adjacency[current]:
            if transitions>=MAX_TRANSITIONS:
                exhausted=True;break
            transitions+=1
            edge=edges[current,nxt];key=(previous,current,nxt,yaw if previous<0 else None)
            if key not in graph['turns']:
                graph['turns'][key]=turn_reason(g,points[current],incoming,edge['heading'])
            if graph['turns'][key]:continue
            total=edge['total']+turn_cost(g,points[current],incoming,edge['heading'])
            trial=(value[0]+edge['uphill'],value[1]+total) if uphill_first else value+total
            target=current,nxt
            if trial>=best.get(target,infinity):continue
            best[target]=trial;parent[target]=state;heapq.heappush(queue,(trial,target))
            if nxt in goals and (incumbent is None or trial<best[incumbent]):incumbent=target
        if exhausted:break
    if end is None:end=incumbent
    path=[]
    if end is not None:
        state=end
        while True:
            path.append(points[state[1]])
            if state==(-1,0):break
            state=parent[state]
        path.reverse()
    stats=dict(complete=end is not None,rank=best[end] if end else None,
        expansions=expanded,transitions=transitions,transition_budget=MAX_TRANSITIONS,
        budget_exhausted=exhausted,search_seconds=time.perf_counter()-began,
        preserve_checkpoint_height=preserve_height,progress_rejected_links=rejected_progress,
        sampled_height_budget_m=height_budget,
        selected_goal_index=end[1] if end is not None else None,
        optimal_in_sampled_graph=bool(end is not None and not exhausted and not graph.get('build_budget_exhausted',False)),
        objective='uphill burden then terrain cost' if uphill_first else 'shared additive terrain cost')
    return path,stats
