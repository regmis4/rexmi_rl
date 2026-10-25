"""Offline conversion of a fixed color corridor into checked straight legs."""
import heapq
import math
import time
from collections import Counter
import numpy as np
from .purple_corridor import integral
from .staged_routes import observed_edge_reason
from .swept_footprint import oriented_sweep_reason
from .turn_staging import stage_reason,turn_reason

MAX_NODES=130
MAX_LINKS=16770
MAX_STATES=60000
MAX_TRANSITIONS=100000


def crossed_cells(g,a,b):
    a=(np.asarray(a)-g.origin+g.world_size/2)/g.cell_size
    b=(np.asarray(b)-g.origin+g.world_size/2)/g.cell_size
    cuts=[0.,1.]
    for axis in (0,1):
        if abs(b[axis]-a[axis])<1e-12:continue
        low,high=sorted((a[axis],b[axis]))
        cuts.extend(((np.arange(math.floor(low)+1,math.ceil(high))-a[axis])/(b[axis]-a[axis])).tolist())
    cuts=np.unique(np.clip(cuts,0.,1.));ts=np.r_[0.,(cuts[:-1]+cuts[1:])/2.,1.]
    cells=np.floor(a+ts[:,None]*(b-a)).astype(int)
    if np.any(cells<0) or np.any(cells>=g.n_cells):return None
    return tuple(cells.T)


def inside(g,roi,a,b):
    cells=crossed_cells(g,a,b)
    return cells is not None and bool(roi[cells].all())


def candidates(g,path,ceiling):
    cells=np.indices(g.known.shape).reshape(2,-1).T
    points=g.origin+(cells+.5)*g.cell_size-g.world_size/2
    distance=np.full(len(points),np.inf)
    for a,b in zip(path,path[1:]):
        delta=np.subtract(b,a);length2=float(delta@delta)
        t=np.clip((points-a)@delta/max(length2,1e-20),0.,1.)
        distance=np.minimum(distance,np.linalg.norm(points-(a+t[:,None]*delta),axis=1))
    roi=(distance<=g.footprint_radius+1e-9).reshape(g.known.shape)
    mask=roi & g.known & ~g.blocked & g.planning_support() & np.isfinite(g.cost) & (g.cost<=ceiling+1e-9)
    available=[tuple(g.cell_to_world(*cell)) for cell in np.argwhere(mask)]
    available=[p for p in available if not stage_reason(g,p)]
    selected=[tuple(path[0]),tuple(path[-1])]
    occupied={g.world_to_cell(*p) for p in selected}
    # Prefer broadphase-eligible raw stages; actual turns are checked in solve.
    preferred=[tuple(p) for p in path[1:-1] if g.world_to_cell(*p) is not None
        and mask[g.world_to_cell(*p)] and not stage_reason(g,p)]
    for p in preferred:
        cell=g.world_to_cell(*p)
        if cell not in occupied and len(selected)<MAX_NODES:
            selected.append(p);occupied.add(cell)
    pending=[p for p in available if g.world_to_cell(*p) not in occupied]
    if pending:
        remaining=np.asarray(pending);nearest=np.min(np.linalg.norm(remaining[:,None,:]-np.asarray(selected)[None,:,:],axis=2),axis=1)
        while len(selected)<MAX_NODES and np.any(nearest>=0.):
            idx=int(np.argmax(nearest));chosen=remaining[idx];selected.append(tuple(chosen))
            nearest=np.minimum(nearest,np.linalg.norm(remaining-chosen,axis=1));nearest[idx]=-1.
    return selected,roi,dict(eligible_cells=len(available),selected_nodes=len(selected),
        omitted_cells=len({g.world_to_cell(*p) for p in available}-{g.world_to_cell(*p) for p in selected}),
        buffer_m=g.footprint_radius,sampling='eligible raw stages then deterministic farthest coverage; unique intermediate cells, no hard spacing')


def solve(g,path,yaw,ceiling,reference,*,wall_seconds=180.):
    began=time.perf_counter();deadline=began+wall_seconds
    points,roi,inventory=candidates(g,np.asarray(path),ceiling)
    edges={};rejections=Counter();checks=0;exhausted=False
    for i,a in enumerate(points):
        for j,b in enumerate(points):
            if i==j:continue
            if time.perf_counter()>=deadline:exhausted=True;break
            checks+=1
            if not inside(g,roi,a,b):rejections['leaves fixed corridor']+=1;continue
            if integral(g,a,b,peak=True)>ceiling+1e-9:rejections['exceeds fixed color ceiling']+=1;continue
            heading=math.atan2(b[1]-a[1],b[0]-a[0])
            reason=observed_edge_reason(g,a,b,path[0]) or oriented_sweep_reason(g,a,b,heading,require_fresh=False,commanded=True,circular_support=False)
            if reason:rejections[reason]+=1;continue
            cost=integral(g,a,b,reference)
            if not math.isfinite(cost) or cost<=0.:rejections['nonpositive or invalid leg']+=1;continue
            edges[i,j]=(cost,heading)
        if exhausted:break
    assert checks<=MAX_LINKS
    adjacency={i:[] for i in range(len(points))}
    for i,j in edges:adjacency[i].append(j)
    initial=(-1,0);best={initial:0.};queue=[(0.,initial)];parent={};end=None;transitions=0;expanded=0;turns=Counter()
    while queue and not exhausted:
        value,state=heapq.heappop(queue)
        if value!=best[state]:continue
        previous,current=state
        if current==1:end=state;break
        if time.perf_counter()>=deadline or expanded>=MAX_STATES:exhausted=True;break
        expanded+=1;incoming=yaw if previous<0 else edges[previous,current][1]
        for target in adjacency[current]:
            if transitions>=MAX_TRANSITIONS:exhausted=True;break
            transitions+=1;cost,heading=edges[current,target]
            # Straight continuation is not a new requested turn.
            delta=abs(math.atan2(math.sin(heading-incoming),math.cos(heading-incoming)))
            reason=turn_reason(g,points[current],incoming,heading) if delta>1e-6 else ''
            if reason:turns[reason]+=1;continue
            nxt=(current,target);trial=value+cost
            if trial>=best.get(nxt,math.inf):continue
            best[nxt]=trial;parent[nxt]=state;heapq.heappush(queue,(trial,nxt))
    route=[]
    if end is not None:
        state=end
        while True:
            route.append(points[state[1]])
            if state==initial:break
            state=parent[state]
        route.reverse()
    return route,dict(complete=bool(route),cost=best[end] if end else None,
        inventory=inventory,directed_checks=checks,accepted_links=len(edges),edge_rejections=dict(rejections),
        turn_rejections=dict(turns),expanded_states=expanded,transitions=transitions,
        budget_exhausted=exhausted,seconds=time.perf_counter()-began,
        scope='Heading-state route in a sampled fixed corridor; geometry only, not physical success')
