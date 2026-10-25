"""Sustained longitudinal climb demand with supported low-grade respite."""
import math
import heapq
import time
from bisect import bisect_left
from collections import defaultdict, deque
from dataclasses import dataclass
import numpy as np
from . import wheelbase_demand as wd, stage_graph, ascent_first
cb=wd.cb
NAME='sustained-climb-v1'
LIMIT=math.tan(math.radians(15.))
MAX_LABELS=60000
MAX_EXPANSIONS=60000
MAX_RELAXATIONS=200000

@dataclass(frozen=True)
class Interval:
    length: float
    cost: float
    coeff: tuple
    respite: bool
    start_m: float=0.

    def __post_init__(self):
        positive=np.polynomial.polynomial.polyval(.5,self.coeff)>0
        poly=np.polynomial.polynomial.polyint(np.polynomial.polynomial.polymul(self.coeff,self.coeff)) if positive else np.zeros(6)
        object.__setattr__(self,'integral',tuple(float(v)*self.cost*self.length for v in poly))
        object.__setattr__(self,'total',self.effort())

    def effort(self,lo=0.,hi=1.):
        def at(x):
            value=0.
            for c in reversed(self.integral):value=value*x+c
            return value
        return max(0.,at(hi)-at(lo))

def intervals(g,field,a,b):
    a,b=np.asarray(a),np.asarray(b);length=math.dist(a,b)
    if not length:return []
    direction=(b-a)/length;shift=direction*wd.WHEELBASE_M/2
    def grade(at):
        p=a+direction*at
        values=[cb.height_at(field,(p+sign*shift-g.origin+g.world_size/2)/g.cell_size) for sign in (1,-1)]
        if not np.isfinite(values).all():raise ValueError('unsupported respite window')
        return (values[0]-values[1])/wd.WHEELBASE_M
    out=[]
    for row in wd.sections(g,field,a,b):
        lo,hi=row['d0'],row['d1'];span=hi-lo
        if span<=1e-12:continue
        z0,zm,z1=[grade(t) for t in (lo,(lo+hi)/2,hi)]
        q2=2*(z1+z0-2*zm);q1=z1-z0-q2
        cuts=[0.,1.]
        for level in (-LIMIT,LIMIT):
            roots=np.roots([q2,q1,z0-level]) if q2!=0 else ([-(z0-level)/q1] if q1!=0 else [])
            cuts.extend(float(np.real(v)) for v in roots if abs(np.imag(v))<1e-10 and 0<np.real(v)<1)
        cuts=sorted(set(cuts))
        for u,v in zip(cuts,cuts[1:]):
            coeff=(z0+q1*u+q2*u*u,(q1+2*q2*u)*(v-u),q2*(v-u)**2)
            out.append(Interval(span*(v-u),row['cost'],coeff,
                bool(abs(np.polynomial.polynomial.polyval(.5,coeff))<=LIMIT+1e-12),lo+span*u))
    return out

@dataclass(frozen=True)
class State:
    completed: float=0.
    burden: float=0.
    respite_distance: float=0.
    latched: bool=False

    @property
    def score(self):return self.completed+self.burden**2

def advance(state,items,required):
    completed,burden,progress,latched=state.completed,state.burden,state.respite_distance,state.latched
    events=[]
    for item in items:
        if not item.respite:
            progress=0.;latched=False;burden+=item.effort();continue
        if not latched and progress+item.length>=required:
            cut=max(0.,required-progress)/item.length
            burden+=item.effort(0.,cut);completed+=burden**2
            events.append(dict(distance_m=item.start_m+cut*item.length,bout_effort=burden,bout_squared=burden**2))
            burden=item.effort(cut,1.);progress=required;latched=True
        else:
            burden+=item.effort();progress=min(required,progress+item.length)
    return State(completed,burden,progress,latched),events


@dataclass(frozen=True)
class Run:
    items: tuple
    def __post_init__(self):
        ends=np.cumsum([p.length for p in self.items])
        values=np.r_[0.,np.cumsum([p.total for p in self.items])]
        object.__setattr__(self,'ends',ends)
        object.__setattr__(self,'values',values)
        object.__setattr__(self,'length',float(ends[-1]))
        object.__setattr__(self,'start_m',self.items[0].start_m)
        object.__setattr__(self,'respite',self.items[0].respite)
        object.__setattr__(self,'total',float(values[-1]))

    def effort(self,lo=0.,hi=1.):
        def at(t):
            if t<=0:return 0.
            if t>=1:return self.total
            d=t*self.length;i=min(bisect_left(self.ends,d),len(self.items)-1)
            start=self.ends[i-1] if i else 0.
            return self.values[i]+self.items[i].effort(0.,(d-start)/self.items[i].length)
        return max(0.,float(at(hi)-at(lo)))


def runs(items):
    groups=[]
    for item in items:
        if not groups or groups[-1][-1].respite!=item.respite:groups.append([])
        groups[-1].append(item)
    return tuple(Run(tuple(group)) for group in groups)


def build(g,start,goal,physical_check,*,candidate_source=None,wall_budget=180.):
    began=time.perf_counter();field=cb.height_field(g);cache={};unavailable=[]
    def cost(grid,a,b):
        try:cache[a,b]=runs(intervals(grid,field,a,b))
        except ValueError as exc:
            unavailable.append(str(exc));return {'unsupported_respite':math.inf}
        return {'squared_grade_effort':math.fsum(r.total for r in cache[a,b])}
    graph=stage_graph.build(g,start,goal,physical_check,candidate_source=candidate_source,
        wall_budget=max(0.,wall_budget-(time.perf_counter()-began)),cost_model=cost)
    for (i,j),edge in graph['edges'].items():
        edge['runs']=cache[graph['points'][i],graph['points'][j]]
        edge['length']=math.dist(graph['points'][i],graph['points'][j])
    graph.update(relief_distance=2*g.footprint_radius,height_field=field,
        objective_unavailable_links=len(unavailable),objective_unavailable_reasons=sorted(set(unavailable)))
    return graph


def dominates(a,b):
    # Compare only equal heading and exact relief phase; no rounding/buckets.
    sa,sb=a['state'],b['state']
    return (sa.respite_distance==sb.respite_distance and sa.latched==sb.latched
        and sa.completed<=sb.completed and sa.burden<=sb.burden and a['length']<=b['length'])


def search(g,graph,yaw):
    began=time.perf_counter();stats=dict(objective=NAME,complete=False,optimal=False,
        budget_exhausted=False,labels_admitted=0,expansions=0,relaxations=0,
        dominance_rejections=0,labels_dominated=0,reverse_pruned=0)
    def finish(path=()):
        stats['search_seconds']=time.perf_counter()-began
        return list(path),stats
    def exhausted(reason):
        stats.update(budget_exhausted=True,reason=reason)
        return finish()
    if graph.get('build_budget_exhausted') or time.perf_counter()>=graph['deadline']:
        return exhausted('graph or feature budget exhausted')
    forward,reverse,checks=ascent_first.transitions(g,graph,yaw);stats.update(checks)
    if checks['transition_budget_exhausted']:return exhausted('heading transition budget exhausted')
    reachable={s for s in graph['edges'] if s[1]==1};todo=deque(reachable)
    while todo:
        if time.perf_counter()>=graph['deadline']:return exhausted('reverse reachability budget exhausted')
        state=todo.popleft()
        for parent in reverse[state]:
            if parent not in reachable:reachable.add(parent);todo.append(parent)
    stats['goal_reachable_heading_states']=len(reachable)
    initial=(-1,0)
    if initial not in reachable:
        stats['reason']='no complete heading-compatible route in sampled graph';return finish()
    if MAX_LABELS<1:return exhausted('label budget exhausted')
    labels=[dict(heading=initial,state=State(),length=0.,parent=None,live=True)]
    groups=defaultdict(list);groups[initial]=[0];queue=[((0.,0.),0)];stats['labels_admitted']=1
    while queue:
        if time.perf_counter()>=graph['deadline'] or stats['expansions']>=MAX_EXPANSIONS:
            return exhausted('label expansion/time budget exhausted')
        _,index=heapq.heappop(queue);label=labels[index]
        if not label['live']:continue
        stats['expansions']+=1
        heading=label['heading']
        if heading[1]==1:
            states=[];cursor=index
            while cursor is not None:
                row=labels[cursor];states.append(row['heading']);cursor=row['parent']
            states.reverse();path=[graph['points'][s[1]] for s in states]
            stats.update(complete=True,optimal=True,rank=[label['state'].score,label['length']],
                path_states=[list(s) for s in states],repeated_vertices=len(states)-len({s[1] for s in states}),
                reason='minimum sustained longitudinal demand in sampled heading graph; distance breaks ties')
            return finish(path)
        for target in forward[heading]:
            if time.perf_counter()>=graph['deadline'] or stats['relaxations']>=MAX_RELAXATIONS:
                return exhausted('label relaxation/time budget exhausted')
            stats['relaxations']+=1
            if target not in reachable:stats['reverse_pruned']+=1;continue
            edge=graph['edges'][target]
            state,_=advance(label['state'],edge['runs'],graph['relief_distance'])
            candidate=dict(heading=target,state=state,length=label['length']+edge['length'],parent=index,live=True)
            existing=groups[target]
            if any(dominates(labels[i],candidate) for i in existing):
                stats['dominance_rejections']+=1;continue
            if len(labels)>=MAX_LABELS:return exhausted('total admitted label budget exhausted')
            kept=[]
            for i in existing:
                if dominates(candidate,labels[i]):labels[i]['live']=False;stats['labels_dominated']+=1
                else:kept.append(i)
            new=len(labels);labels.append(candidate);groups[target]=kept+[new]
            stats['labels_admitted']+=1
            heapq.heappush(queue,((state.score,candidate['length']),new))
    stats['reason']='no complete sustained-climb label reaches goal';return finish()


route_reason=wd.route_reason


def describe_route(g,path):
    result=wd.describe_route(g,path)
    if not result:return result
    result['quartic_diagnostic_rank']=result['rank']
    # Preserve names: inherited sections are quartic diagnostics, not bout cost.
    result['quartic_diagnostic_sections']=result.pop('sections')
    result['quartic_diagnostic_exponent']=result.pop('demand_exponent')
    field=cb.height_field(g);state=State();travel=0.;events=[];all_runs=[]
    for a,b in zip(path,path[1:]):
        leg=runs(intervals(g,field,a,b));state,added=advance(state,leg,2*g.footprint_radius)
        for event in added:
            p=np.asarray(a)+(np.asarray(b)-a)*event['distance_m']/math.dist(a,b)
            event.update(point=p.tolist(),height_m=cb.height_at(field,(p-g.origin+g.world_size/2)/g.cell_size))
            event['distance_m']+=travel
        events.extend(added)
        all_runs.extend(dict(start_m=travel+v.start_m,end_m=travel+v.start_m+v.length,respite=v.respite,effort=v.total) for v in leg)
        travel+=math.dist(a,b)
    result.update(rank=[state.score,travel],reset_events=events,longitudinal_runs=all_runs,
        completed_bout_cost=state.completed,final_bout_effort=state.burden,
        final_bout_cost=state.burden**2,relief_distance_m=2*g.footprint_radius,
        relief_grade_deg=15.,definition='Sum squared grade²-effort bouts; supported longitudinal respite only, not physical recovery or turn permission')
    return result
