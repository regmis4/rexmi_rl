"""Climbing burden: integral displayed difficulty against positive height gain.

Footprint-mean height is continuous bilinear terrain, not robot body motion.
Distance breaks ties. No probability, exterior classification or route sketch.
"""
import heapq
import math
import time
from collections import defaultdict
import numpy as np
from scipy import ndimage as ndi
from . import ascent_first, stage_graph

NAME = 'climb-burden-v1'
MAX_RELAXATIONS = 200000


def height_field(g):
    radius = g.footprint_radius/g.cell_size + .71
    axis = np.arange(-math.ceil(radius), math.ceil(radius)+1)
    kernel = (axis[:,None]**2+axis[None,:]**2 <= radius**2).astype(float)
    known = g.known & np.isfinite(g.surface)
    support = ndi.convolve(known.astype(float), kernel, mode='constant', cval=0.)
    total = ndi.convolve(np.where(known,g.surface,0.).astype(float), kernel, mode='constant', cval=0.)
    return np.where(support >= kernel.sum()-.5, total/kernel.sum(), np.nan)


def height_at(field, p):
    """p is in grid corner coordinates; samples lie at half-cell centres."""
    x,y = np.asarray(p)-.5
    i,j = math.floor(x),math.floor(y)
    if i<0 or j<0 or i+1>=field.shape[0] or j+1>=field.shape[1]:return math.nan
    values=field[i:i+2,j:j+2]
    if not np.isfinite(values).all():return math.nan
    u,v=x-i,y-j;z=values[0,0]
    return float(z+u*(values[1,0]-z)+v*(values[0,1]-z)+u*v*(values[1,1]-values[1,0]-values[0,1]+z))


def pieces(g, field, a, b):
    """Exact C·positive dz on piecewise bilinear height and constant cell cost.

    Between crossings z(t) is quadratic. Its stationary point splits ascent
    from descent, so endpoint-only gain cannot hide an intervening climb.
    """
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    length=float(np.linalg.norm(b-a))
    if length<=0:return []
    first=(a-g.origin+g.world_size/2)/g.cell_size
    last=(b-g.origin+g.world_size/2)/g.cell_size
    cuts=[0.,1.]
    for dim in (0,1):
        delta=last[dim]-first[dim]
        if abs(delta)<1e-14:continue
        lo,hi=sorted((first[dim],last[dim]))
        for offset in (0.,.5):
            boundaries=np.arange(math.floor(lo-offset)+1,math.ceil(hi-offset))+offset
            cuts.extend(((boundaries-first[dim])/delta).tolist())
    cuts=np.unique(np.clip(cuts,0.,1.));rows=[]
    for lo,hi in zip(cuts,cuts[1:]):
        mid=(lo+hi)/2
        cell=tuple(np.floor(first+mid*(last-first)).astype(int))
        if any(i<0 or i>=g.n_cells for i in cell):raise ValueError('cost outside map')
        cost=float(g.cost[cell])
        if not g.known[cell] or not math.isfinite(cost) or cost<=0:raise ValueError('unobserved terrain cost')
        heights=[height_at(field,first+t*(last-first)) for t in (lo,mid,hi)]
        if not np.isfinite(heights).all():raise ValueError('unobserved footprint height support')
        z0,zm,z1=heights
        q2=2*(z1+z0-2*zm);q1=z1-z0-q2
        splits=[0.,1.]
        if q2!=0:
            stationary=-q1/(2*q2)
            if 0<stationary<1:splits.insert(1,stationary)
        for u,v in zip(splits,splits[1:]):
            za=z0+q1*u+q2*u*u;zb=z0+q1*v+q2*v*v
            rows.append(dict(z0=za,z1=zb,cost=cost,
                d0=length*(lo+(hi-lo)*u),d1=length*(lo+(hi-lo)*v),
                gain=max(0.,zb-za),loss=max(0.,za-zb),burden=cost*max(0.,zb-za)))
    return rows


def metric(g,field,a,b):
    rows=pieces(g,field,a,b)
    return dict(burden=math.fsum(r['burden'] for r in rows),length=math.dist(a,b))


def build(g,start,goal,physical_check,*,candidate_source=None,wall_budget=180.):
    began=time.perf_counter();field=height_field(g);cache={};unavailable=[]
    def cost(grid,a,b):
        try:cache[a,b]=metric(grid,field,a,b)
        except ValueError as exc:
            unavailable.append(str(exc));return {'unavailable_height':math.inf}
        return {'climb_burden':cache[a,b]['burden']}
    graph=stage_graph.build(g,start,goal,physical_check,candidate_source=candidate_source,
        wall_budget=max(0.,wall_budget-(time.perf_counter()-began)),cost_model=cost)
    for (i,j),edge in graph['edges'].items():
        edge.update(cache[graph['points'][i],graph['points'][j]])
    graph.update(height_field=field,objective_unavailable_links=len(unavailable),
                 objective_unavailable_reasons=sorted(set(unavailable)))
    return graph


def _add(a,b):return a[0]+b[0],a[1]+b[1]


def search(g,graph,yaw):
    began=time.perf_counter();stats=dict(objective=NAME,complete=False,optimal=False,
        budget_exhausted=False,expansions=[0,0],relaxations=0)
    def finish(path=()):
        stats['search_seconds']=time.perf_counter()-began
        return list(path),stats
    if graph.get('build_budget_exhausted') or time.perf_counter()>=graph['deadline']:
        stats.update(budget_exhausted=True,reason='graph budget exhausted');return finish()
    edges=graph['edges']
    if any(not math.isfinite(e['burden']) or e['burden']<0 or not math.isfinite(e['length']) or e['length']<=0 for e in edges.values()):
        stats['reason']='invalid climbing metric';return finish()
    forward,reverse,checks=ascent_first.transitions(g,graph,yaw);stats.update(checks)
    if checks['transition_budget_exhausted']:
        stats.update(budget_exhausted=True,reason='heading graph budget exhausted');return finish()
    initial=(-1,0);zero=(0.,0.);infinity=(math.inf,math.inf)
    distances=[{initial:zero},{state:zero for state in edges if state[1]==1}]
    queues=[[(zero,initial)],[(zero,state) for state in distances[1]]]
    heapq.heapify(queues[1]);parents=[{},{}];settled=[set(),set()]
    mu=infinity;meeting=None
    while queues[0] and queues[1]:
        for side in (0,1):
            q=queues[side]
            while q and (q[0][1] in settled[side] or q[0][0]!=distances[side][q[0][1]]):heapq.heappop(q)
        if not queues[0] or not queues[1]:break
        if _add(queues[0][0][0],queues[1][0][0])>=mu:break
        if (time.perf_counter()>=graph['deadline'] or stats['relaxations']>=MAX_RELAXATIONS
                or sum(stats['expansions'])>=ascent_first.MAX_STATES):
            stats.update(budget_exhausted=True,reason='bidirectional search budget exhausted');return finish()
        side=0 if queues[0][0][0]<=queues[1][0][0] else 1
        value,state=heapq.heappop(queues[side]);settled[side].add(state);stats['expansions'][side]+=1
        for target in (forward if side==0 else reverse)[state]:
            if stats['relaxations']>=MAX_RELAXATIONS or time.perf_counter()>=graph['deadline']:
                stats.update(budget_exhausted=True,reason='bidirectional search budget exhausted');return finish()
            stats['relaxations']+=1
            # Reverse expansion transposes the ORIGINAL outgoing transition.
            edge=edges[target if side==0 else state]
            trial=_add(value,(edge['burden'],edge['length']))
            if trial<distances[side].get(target,infinity):
                distances[side][target]=trial;parents[side][target]=state
                heapq.heappush(queues[side],(trial,target))
            joined=_add(distances[side].get(target,infinity),distances[1-side].get(target,infinity))
            if joined<mu:mu=joined;meeting=target
    if meeting is None:
        stats['reason']='no complete heading-compatible route in sampled graph';return finish()
    states=[meeting];state=meeting
    while state in parents[0]:state=parents[0][state];states.append(state)
    states.reverse();state=meeting
    while state in parents[1]:state=parents[1][state];states.append(state)
    path=[graph['points'][s[1]] for s in states]
    stats.update(complete=True,optimal=True,rank=list(mu),meeting_state=list(meeting),
        meeting_point=graph['points'][meeting[1]],path_states=[list(s) for s in states],
        reason='minimum climbing burden; distance breaks ties')
    return finish(path)


def route_reason(g,path,yaw,start):
    # No exterior/gentle-suffix test: every segment uses ordinary physical checks.
    return ascent_first.route_reason(g,path,yaw,len(path)-1,start)


def describe_route(g,path):
    field=height_field(g);rows=[];distance=0.
    for a,b in zip(path,path[1:]):
        for row in pieces(g,field,a,b):
            row['d0']+=distance;row['d1']+=distance;rows.append(row)
        distance+=math.dist(a,b)
    if not rows:return {}
    start=rows[0]['z0'];bands=defaultdict(lambda:dict(positive_gain_m=0.,burden=0.))
    for row in rows:
        if row['gain']<=0:continue
        z0,z1=row['z0'],row['z1']
        cuts=[z0]+[start+n for n in range(math.floor(z0-start)+1,math.ceil(z1-start))]+[z1]
        for a,b in zip(cuts,cuts[1:]):
            n=math.floor((a+b)/2-start);bands[n]['positive_gain_m']+=b-a;bands[n]['burden']+=row['cost']*(b-a)
    positive=[row for row in rows if row['gain']>0]
    gains=np.r_[0.,np.cumsum([r['gain'] for r in positive])]
    burden=np.r_[0.,np.cumsum([r['burden'] for r in positive])]
    width=min(1.,float(gains[-1]));window=None
    if width>0:
        starts=np.unique(np.clip(np.r_[gains,gains-width],0,gains[-1]-width))
        values=np.interp(starts+width,gains,burden)-np.interp(starts,gains,burden)
        index=int(np.argmax(values));lo=float(starts[index]);hi=lo+width
        covered=[r for i,r in enumerate(positive) if gains[i+1]>lo and gains[i]<hi]
        window=dict(positive_gain_m=width,partial=width<1.,burden=float(values[index]),
            cumulative_gain_start_m=lo,cumulative_gain_end_m=hi,
            containing_interval_start_m=covered[0]['d0'],containing_interval_end_m=covered[-1]['d1'],
            definition='contiguous window of cumulative positive height gain; spatial interval contains intervening flat/descent and may include partial end pieces')
    heights=np.array([rows[0]['z0']]+[r['z1'] for r in rows])
    return dict(rank=[math.fsum(r['burden'] for r in rows),distance],length_m=distance,
        positive_gain_m=math.fsum(r['gain'] for r in rows),height_loss_m=math.fsum(r['loss'] for r in rows),
        drawdown_m=float(np.max(np.maximum.accumulate(heights)-heights)),
        bands=[dict(lower_height_m=start+n,upper_height_m=start+n+1,
            cost_per_positive_m=v['burden']/v['positive_gain_m'],**v) for n,v in sorted(bands.items())],
        worst_gain_window=window,height_profile=[dict(distance_m=0.,height_m=start,burden=0.)]+[
            dict(distance_m=r['d1'],height_m=r['z1'],burden=float(b)) for r,b in zip(rows,np.cumsum([r['burden'] for r in rows]))])
