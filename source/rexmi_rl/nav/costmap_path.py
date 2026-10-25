"""Offline, turn-free shortest paths through a frozen displayed cost map.

No controller state, staging locations, heading limits, turn costs or recovery
logic. Optimality is in the explicitly sampled graph, not continuous space.
"""
import math
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra


def crossings(a,b):
    """Cell intervals and all touched cells, in grid coordinates.

    Zero-length corner touches contribute no cost but prevent corner cutting.
    """
    a,b=np.asarray(a,float),np.asarray(b,float);delta=b-a
    cuts=[0.,1.]
    for axis in (0,1):
        if abs(delta[axis])<1e-14:continue
        lo,hi=sorted((a[axis],b[axis]))
        cuts.extend(((np.arange(math.floor(lo)+1,math.ceil(hi))-a[axis])/delta[axis]).tolist())
    cuts=np.unique(np.clip(cuts,0.,1.))
    fractions=np.diff(cuts)
    cells=np.floor(a+(.5*(cuts[:-1]+cuts[1:]))[:,None]*delta).astype(int)
    touched=set(map(tuple,cells))
    for point in a+cuts[:,None]*delta:
        choices=[]
        for value in point:
            choices.append([round(value)-1,round(value)] if abs(value-round(value))<1e-10 else [math.floor(value)])
        touched.update((x,y) for x in choices[0] for y in choices[1])
    return cells,fractions,np.array(sorted(touched),dtype=int)


def edge_cost(cost,allowed,a,b,cell_size,*,density=None):
    cells,fractions,touched=crossings(a,b)
    if (touched<0).any() or (touched>=cost.shape).any() or not allowed[tuple(touched.T)].all():
        return math.inf
    delta=np.subtract(b,a)
    values=cost[tuple(cells.T)] if density is None else density(tuple(cells.T),*delta)
    return float(np.sum(values*fractions)*np.linalg.norm(delta)*cell_size)


def neighbors(radius=3):
    return [(x,y) for x in range(-radius,radius+1) for y in range(-radius,radius+1)
            if math.gcd(abs(x),abs(y))==1]


def connectors(cost,allowed,point,cell_size,radius,*,density=None,reverse=False):
    cell=np.floor(point).astype(int)
    for dx in range(-radius,radius+1):
        for dy in range(-radius,radius+1):
            target=cell+[dx,dy]
            if (target<0).any() or (target>=cost.shape).any():continue
            a,b=(target+.5,point) if reverse else (point,target+.5)
            value=edge_cost(cost,allowed,a,b,cell_size,density=density)
            if math.isfinite(value):yield tuple(target),value


def solve(cost,allowed,start,lower,cell_size,*,radius=3,density=None):
    """One Dijkstra search; every reachable target shares the same cost field."""
    cost=np.asarray(cost,float);allowed=np.asarray(allowed,bool)
    if cost.ndim!=2 or allowed.shape!=cost.shape or cell_size<=0 or radius<1:
        raise ValueError('Invalid cost-map graph geometry')
    if np.any(allowed & (~np.isfinite(cost) | (cost<=0))):
        raise ValueError('Usable cells require finite positive displayed cost')
    coords=np.argwhere(allowed);shape=np.array(cost.shape)
    rows=[];cols=[];weights=[]
    offsets=neighbors(radius)
    for dx,dy in offsets:
        if dx<0 or dx==0 and dy<0:continue
        cells,fractions,touched=crossings((.5,.5),(.5+dx,.5+dy))
        valid=np.ones(len(coords),bool)
        for offset in touched:
            p=coords+offset
            inside=((p>=0)&(p<shape)).all(axis=1)
            good=np.zeros(len(coords),bool)
            good[inside]=allowed[tuple(p[inside].T)]
            valid &= good
        source=coords[valid];target=source+[dx,dy]
        if not len(source):continue
        forward=sum((cost[tuple((source+offset).T)] if density is None else
                     density(tuple((source+offset).T),dx,dy))*fraction for offset,fraction in zip(cells,fractions))
        backward=forward if density is None else sum(density(tuple((source+offset).T),-dx,-dy)*fraction
            for offset,fraction in zip(cells,fractions))
        value=forward*math.hypot(dx,dy)*cell_size
        reverse_value=backward*math.hypot(dx,dy)*cell_size
        a=np.ravel_multi_index(source.T,cost.shape);b=np.ravel_multi_index(target.T,cost.shape)
        rows.extend((a,b));cols.extend((b,a));weights.extend((value,reverse_value))
    start_grid=(np.asarray(start)-lower)/cell_size
    links=list(connectors(cost,allowed,start_grid,cell_size,radius,density=density))
    if not links:raise ValueError('Start has no connection to the cost-map graph')
    source_id=cost.size
    rows.append(np.full(len(links),source_id));cols.append(np.array([np.ravel_multi_index(c,cost.shape) for c,_ in links]))
    weights.append(np.array([w for _,w in links]))
    graph=coo_matrix((np.concatenate(weights),(np.concatenate(rows),np.concatenate(cols))),
                     shape=(cost.size+1,cost.size+1)).tocsr()
    distance,parent=dijkstra(graph,directed=True,indices=source_id,return_predecessors=True)
    return dict(distance=distance[:-1].reshape(cost.shape),parent=parent,start=tuple(start),density=density,
        lower=np.asarray(lower),cell_size=cell_size,shape=cost.shape,source_id=source_id,
        nodes=int(allowed.sum())+1,directed_edges=int(graph.nnz),headings=len(offsets),radius=radius)


def path_to_cell(solution,cell):
    index=int(np.ravel_multi_index(cell,solution['shape']))
    if not math.isfinite(solution['distance'][tuple(cell)]):return []
    path=[]
    while index!=solution['source_id']:
        point=solution['lower']+(np.array(np.unravel_index(index,solution['shape']))+.5)*solution['cell_size']
        path.append(tuple(point));index=int(solution['parent'][index])
        if index<0:raise RuntimeError('Broken shortest-path predecessor chain')
    return [solution['start']]+path[::-1]


def path_to_point(solution,cost,allowed,point):
    end_grid=(np.asarray(point)-solution['lower'])/solution['cell_size']
    choices=[(solution['distance'][cell]+value,cell) for cell,value in
        connectors(cost,allowed,end_grid,solution['cell_size'],solution['radius'],density=solution.get('density'),reverse=True)]
    value,cell=min(choices,default=(math.inf,None))
    return (path_to_cell(solution,cell)+[tuple(point)],float(value)) if math.isfinite(value) else ([],None)
