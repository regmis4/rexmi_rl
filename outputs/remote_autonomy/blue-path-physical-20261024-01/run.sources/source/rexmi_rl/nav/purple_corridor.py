"""Map-only corridor study: displayed terrain cost, with optional early height.

No trajectory, control command or calibrated probability is produced. Geometry
broadphase is shared by both objectives; requested turns need separate checking.
"""
import heapq
import math
import time
import numpy as np
from scipy.sparse import csr_matrix
from .staged_routes import observed_edge_reason

MAX_CELLS=102400
MAX_EDGES=8*MAX_CELLS+18
MAX_RELAXATIONS=2*MAX_EDGES


def height_factor(height, reference):
    lo,hi=reference
    values=np.asarray(height,dtype=float)
    if not np.isfinite([lo,hi]).all() or hi-lo<=1e-6:return np.ones_like(values)
    return 1.+np.clip((hi-values)/(hi-lo),0.,1.)


def integral(g,a,b,reference=None,*,peak=False):
    """Exact piecewise-cell integral of the displayed field, in cost-metres."""
    delta=np.subtract(b,a);length=float(np.linalg.norm(delta))
    if length<1e-12:
        cell=g.world_to_cell(*a)
        return float(g.cost[cell]) if peak and cell is not None else math.inf if peak else 0.
    first=(np.asarray(a)-g.origin+g.world_size/2)/g.cell_size
    last=(np.asarray(b)-g.origin+g.world_size/2)/g.cell_size
    cuts=[0.,1.]
    for dim in (0,1):
        if abs(last[dim]-first[dim])<1e-12:continue
        lo,hi=sorted((first[dim],last[dim]))
        cuts.extend(((np.arange(math.floor(lo)+1,math.ceil(hi))-first[dim])/(last[dim]-first[dim])).tolist())
    cuts=np.unique(np.clip(cuts,0.,1.));mid=(cuts[1:]+cuts[:-1])/2
    cells=np.floor(first+mid[:,None]*(last-first)).astype(int)
    if np.any(cells<0) or np.any(cells>=g.n_cells):return math.inf
    cells=tuple(cells.T);values=g.cost[cells].astype(float)
    if not np.all(g.known[cells]&np.isfinite(values)&(values>0)&np.isfinite(g.surface[cells])):return math.inf
    if peak:
        endpoints=[g.world_to_cell(*p) for p in (a,b)]
        if any(cell is None for cell in endpoints):return math.inf
        return max(float(values.max()),*(float(g.cost[cell]) for cell in endpoints))
    if reference is not None:values=values*height_factor(g.surface[cells],reference)
    return float(np.sum(values*np.diff(cuts))*length)


def build(g,start,goal):
    """One dense directed graph. No candidate bins or demonstration points."""
    began=time.perf_counter();n=g.n_cells;count=n*n
    if count>MAX_CELLS:raise ValueError('corridor cell budget exceeded')
    start_cell,goal_cell=(g.world_to_cell(*p) for p in (start,goal))
    if start_cell is None or goal_cell is None:raise ValueError('endpoint outside map')
    reference=tuple(float(g.surface[c]) for c in (start_cell,goal_cell))
    allowed=g.known & ~g.blocked & g.planning_support() & np.isfinite(g.cost) & (g.cost>0) & np.isfinite(g.surface)
    allowed &= np.isfinite(g.gx)&np.isfinite(g.gy)
    ids=np.arange(count).reshape(n,n)
    rows=[];cols=[];base=[];early=[];peaks=[]
    factor=height_factor(g.surface,reference)
    for dx,dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
        aa=(slice(max(0,-dx),min(n,n-dx)),slice(max(0,-dy),min(n,n-dy)))
        bb=(slice(max(0,dx),min(n,n+dx)),slice(max(0,dy),min(n,n+dy)))
        norm=math.hypot(dx,dy);heading=g.heading_bin(math.atan2(dy,dx))
        along=(g.gx*dx+g.gy*dy)/norm;side=np.abs(-g.gx*dy+g.gy*dx)/norm
        direction=(along<=math.tan(math.radians(g.uphill_limit_deg))+1e-8)&(side<=math.tan(math.radians(g.side_limit_deg))+1e-8)
        direction &= ~g.approach_blocked[...,heading] & (g.failures[...,heading]<2)
        valid=allowed[aa]&allowed[bb]&direction[aa]&direction[bb]
        if dx and dy:
            ab=(aa[0],bb[1]);ba=(bb[0],aa[1])
            valid &= allowed[ab]&allowed[ba]&direction[ab]&direction[ba]
        rows.append(ids[aa][valid]);cols.append(ids[bb][valid])
        distance=g.cell_size*norm
        peaks.append(np.maximum(g.cost[aa][valid],g.cost[bb][valid]))
        base.append(.5*distance*(g.cost[aa][valid]+g.cost[bb][valid]))
        early.append(.5*distance*(g.cost[aa][valid]*factor[aa][valid]+g.cost[bb][valid]*factor[bb][valid]))
    connectors=[]
    for endpoint,cell,virtual,is_start in ((start,start_cell,count,True),(goal,goal_cell,count+1,False)):
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                target=(cell[0]+dx,cell[1]+dy)
                if not (0<=target[0]<n and 0<=target[1]<n and allowed[target]):continue
                point=tuple(g.cell_to_world(*target));a,b=(tuple(endpoint),point) if is_start else (point,tuple(endpoint))
                reason=observed_edge_reason(g,a,b,start)
                if reason:connectors.append(dict(start=list(a),end=list(b),reason=reason));continue
                rows.append(np.array([virtual if is_start else ids[target]]));cols.append(np.array([ids[target] if is_start else virtual]))
                peaks.append(np.array([integral(g,a,b,peak=True)]))
                base.append(np.array([integral(g,a,b)]));early.append(np.array([integral(g,a,b,reference)]))
    rows,cols=np.concatenate(rows),np.concatenate(cols)
    base,early,peaks=np.concatenate(base),np.concatenate(early),np.concatenate(peaks)
    if len(rows)>MAX_EDGES:raise ValueError('corridor edge budget exceeded')
    if not np.all(np.isfinite(base)&np.isfinite(early)&np.isfinite(peaks)&(base>=0)&(early>=0)&(peaks>=0)):raise ValueError('invalid corridor weight')
    points=np.stack(np.meshgrid(np.arange(n),np.arange(n),indexing='ij'),axis=-1).reshape(-1,2)
    points=g.origin+(points+.5)*g.cell_size-g.world_size/2
    points=np.vstack([points,start,goal])
    return dict(ceiling=csr_matrix((peaks,(rows,cols)),shape=(count+2,count+2)),base=csr_matrix((base,(rows,cols)),shape=(count+2,count+2)),
        early=csr_matrix((early,(rows,cols)),shape=(count+2,count+2)),points=points,
        start=count,goal=count+1,reference=reference,allowed=allowed,
        stats=dict(cells=count,eligible_cells=int(allowed.sum()),directed_edges=len(rows),
            build_seconds=time.perf_counter()-began,connector_rejections=connectors))


def bidirectional(graph,start,goal,*,max_relaxations=MAX_RELAXATIONS,wall_seconds=30.):
    """Bidirectional Dijkstra on original directed weights; never first-meet."""
    began=time.perf_counter();graphs=(graph.tocsr(),graph.transpose().tocsr());n=graph.shape[0]
    if not np.isfinite(graph.data).all() or np.any(graph.data<0):raise ValueError('nonnegative finite weights required')
    dist=[np.full(n,np.inf),np.full(n,np.inf)];parent=[np.full(n,-1,dtype=int),np.full(n,-1,dtype=int)]
    settled=[np.zeros(n,dtype=bool),np.zeros(n,dtype=bool)];queues=[[(0.,start)],[(0.,goal)]]
    dist[0][start]=dist[1][goal]=0.;mu=math.inf;meeting=None;expanded=[0,0];relaxations=0;exhausted=False
    if start==goal:return [start],dict(cost=0.,meeting=start,optimal=True,expansions=[0,0],relaxations=0,budget_exhausted=False,seconds=0.)
    while queues[0] and queues[1]:
        for side in (0,1):
            q=queues[side]
            while q and (q[0][0]!=dist[side][q[0][1]] or settled[side][q[0][1]]):heapq.heappop(q)
        if not queues[0] or not queues[1]:break
        if queues[0][0][0]+queues[1][0][0]>=mu:break
        if relaxations>=max_relaxations or time.perf_counter()-began>=wall_seconds:
            exhausted=True;break
        side=0 if queues[0][0][0]<=queues[1][0][0] else 1
        value,node=heapq.heappop(queues[side]);settled[side][node]=True;expanded[side]+=1
        matrix=graphs[side]
        for k in range(matrix.indptr[node],matrix.indptr[node+1]):
            if relaxations>=max_relaxations:exhausted=True;break
            relaxations+=1;target=matrix.indices[k];trial=value+matrix.data[k]
            if trial<dist[side][target]:
                dist[side][target]=trial;parent[side][target]=node;heapq.heappush(queues[side],(trial,target))
            joined=dist[side][target]+dist[1-side][target]
            if joined<mu:mu=float(joined);meeting=int(target)
        if exhausted:break
    route=[]
    if meeting is not None:
        node=meeting
        while node!=-1:route.append(int(node));node=parent[0][node]
        route.reverse();node=parent[1][meeting]
        while node!=-1:route.append(int(node));node=parent[1][node]
    return route,dict(cost=mu if route else None,meeting=meeting,optimal=not exhausted,
        expansions=expanded,relaxations=relaxations,budget_exhausted=exhausted,seconds=time.perf_counter()-began)


def minimum_ceiling(graph,start,goal,*,max_relaxations=MAX_RELAXATIONS,wall_seconds=30.):
    """Scalar minimax; certify B only after settling the goal."""
    began=time.perf_counter();graph=graph.tocsr();n=graph.shape[0]
    if not np.isfinite(graph.data).all() or np.any(graph.data<0):raise ValueError('finite nonnegative ceiling required')
    best=np.full(n,np.inf);best[start]=0.;queue=[(0.,start)]
    expanded=0;relaxations=0;settled=False;exhausted=False
    while queue:
        value,node=heapq.heappop(queue)
        if value!=best[node]:continue
        if node==goal:settled=True;break
        if relaxations>=max_relaxations or time.perf_counter()-began>=wall_seconds:
            exhausted=True;break
        expanded+=1
        for k in range(graph.indptr[node],graph.indptr[node+1]):
            if relaxations>=max_relaxations:exhausted=True;break
            relaxations+=1;target=graph.indices[k];trial=max(value,graph.data[k])
            if trial<best[target]:best[target]=trial;heapq.heappush(queue,(trial,int(target)))
        if exhausted:break
    return (float(best[goal]) if settled else None),dict(certified=settled,
        incumbent=float(best[goal]) if math.isfinite(best[goal]) else None,
        expansions=expanded,relaxations=relaxations,budget_exhausted=exhausted,seconds=time.perf_counter()-began)


def restrict_ceiling(weighted,ceiling,limit):
    if not (np.array_equal(weighted.indptr,ceiling.indptr) and np.array_equal(weighted.indices,ceiling.indices)):
        raise ValueError('ceiling and weights must share exact directed edges')
    coo=weighted.tocoo();keep=ceiling.data<=limit+1e-9
    return csr_matrix((coo.data[keep],(coo.row[keep],coo.col[keep])),shape=weighted.shape)
