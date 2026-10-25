"""Straight-motion geometry with configured body and passive-yaw stopping sweeps.

Global planning, turning and ground support retain their circular contracts.
These geometric forecasts are under simulator characterization, not calibration.
"""
import math

import numpy as np


def arrival_ray_interval(position, endpoint, body_yaw, direction, radius):
    """Distances along a body-axis ray that enter an unchanged arrival disk.

    Pure geometry for evaluating unnecessary alignment. It grants no motion
    permission: terrain, stopping, posture, progress and settling remain separate.
    Reverse changes travel direction, not body orientation or the destination.
    """
    if direction not in (-1,1) or radius<=0 or not np.isfinite([*position,*endpoint,body_yaw,radius]).all():
        raise ValueError('Require finite planar geometry, positive radius and direction +1 or -1')
    delta=np.subtract(endpoint,position)
    axis=direction*np.array([math.cos(body_yaw),math.sin(body_yaw)])
    along=float(delta@axis)
    cross_squared=max(0.,float(delta@delta)-along*along)
    if cross_squared>radius*radius+1e-12:
        return None
    half=math.sqrt(max(0.,radius*radius-cross_squared))
    if along+half<0.:
        return None
    return max(0.,along-half),max(0.,along+half)


def oriented_swept_cells(grid, start, end, yaw, yaw_offsets=(0.,0.), position_reserve=0.):
    """Conservative cell intersection for the full translated rectangle.

    Separating-axis tests cover the continuous translation, including cells
    touched only between endpoints. The union over heading uncertainty includes
    an explicit rotation reserve between sampled angles, not point sampling
    alone. Half dimensions already include the configured body reserve.
    """
    start,end=np.asarray(start,dtype=float),np.asarray(end,dtype=float)
    lo,hi=map(float,yaw_offsets)
    if (not np.isfinite([*start,*end,yaw,lo,hi,position_reserve]).all()
            or position_reserve<0. or lo>hi or hi-lo>2*math.pi):
        raise ValueError('Require finite poses and ordered heading uncertainty spanning at most one turn')
    count=max(1,math.ceil((hi-lo)/.04))
    angles=np.linspace(yaw+lo,yaw+hi,count+1) if hi>lo else np.array([yaw+lo])
    # Every intermediate vertex is within r*dtheta/2 of one sampled angle.
    reserve=math.hypot(grid.half_length,grid.half_width)*(hi-lo)/count/2+position_reserve
    length,width=grid.half_length+reserve,grid.half_width+reserve
    u=np.stack([np.cos(angles),np.sin(angles)],axis=1)
    v=np.stack([-np.sin(angles),np.cos(angles)],axis=1)
    extent=np.max(length*np.abs(u)+width*np.abs(v),axis=0)
    lower=np.minimum(start,end)-extent;upper=np.maximum(start,end)+extent
    world_lower=np.asarray(grid.origin)-grid.world_size/2
    world_upper=world_lower+grid.world_size
    if (lower<world_lower-1e-9).any() or (upper>world_upper+1e-9).any():
        return None
    first=np.maximum(0,np.floor((lower-world_lower)/grid.cell_size).astype(int))
    last=np.minimum(grid.n_cells-1,np.floor((upper-world_lower)/grid.cell_size).astype(int))
    rr,cc=np.meshgrid(np.arange(first[0],last[0]+1),np.arange(first[1],last[1]+1),indexing='ij')
    centres=world_lower+(np.stack([rr,cc],axis=-1)+.5)*grid.cell_size
    covered=np.zeros(rr.shape,dtype=bool)
    delta=end-start;distance=np.linalg.norm(delta)
    normal=np.array([-delta[1],delta[0]])/distance if distance>1e-12 else np.array([1.,0.])
    for longitudinal,lateral in zip(u,v):
        intersects=np.ones(rr.shape,dtype=bool)
        for axis in (longitudinal,lateral,normal,np.array([1.,0.]),np.array([0.,1.])):
            body_extent=length*abs(np.dot(longitudinal,axis))+width*abs(np.dot(lateral,axis))
            tile_extent=grid.cell_size/2*np.abs(axis).sum()
            a,b=float(start@axis),float(end@axis)
            projections=centres@axis
            intersects&=(projections>=min(a,b)-body_extent-tile_extent-1e-9)
            intersects&=(projections<=max(a,b)+body_extent+tile_extent+1e-9)
        covered|=intersects
    return rr[covered],cc[covered]


def oriented_sweep_reason(grid, start, end, yaw, *, yaw_offsets=(0.,0.),
                          require_fresh=True, commanded=True, no_go_exit=False,
                          position_reserve=0., directional_exit=False, observation_support=None, circular_support=True):
    """Candidate check retaining terrain, freshness and exclusions.

    `circular_support=False` explicitly selects the oriented physical ground
    envelope; its known/fresh/hazard/35-degree checks below remain mandatory.
    Existing callers retain the circular ground precheck by default.

    `commanded=False` is only for incidental stopping drift: physical terrain
    and directional exclusions remain, but intended uphill/cross-grade limits
    are not assigned to instantaneous gait sway.
    """
    observed=grid.fresh if observation_support is None else observation_support
    if circular_support:
        reason=grid.support_reason(start,end)
        if reason:
            return reason
    if commanded:
        reason=grid.grade_reason(start,end)
        if reason:
            return reason
    cells=oriented_swept_cells(grid,start,end,yaw,yaw_offsets,position_reserve)
    if cells is None or not grid.known[cells].all():
        return 'unobserved oriented swept footprint'
    if grid.hazard[cells].any():
        return 'obstacle in oriented swept footprint'
    if require_fresh and not observed[cells].all():
        return 'stale oriented swept footprint'
    if not np.isfinite(grid.slope[cells]).all() or (grid.slope[cells]>math.tan(math.radians(35))).any():
        return 'oriented footprint exceeds 35 degree terrain boundary'
    first=grid.world_to_cell(*start)
    departing=bool(no_go_exit and first is not None and grid.no_go[first])
    depth=grid.no_go_depth[first] if first is not None else 0.
    distance=math.dist(start,end)
    heading=grid.heading_bin(math.atan2(end[1]-start[1],end[0]-start[0])) if distance>1e-9 else None
    permission=None if isinstance(directional_exit,bool) else directional_exit
    leaving_direction=(not commanded and directional_exit and not grid.directional_exit_reason(start,end,permission))
    previous=None
    for fraction in np.linspace(0.,1.,max(2,math.ceil(distance/(grid.cell_size*.25))+1)):
        cell=grid.world_to_cell(*(np.asarray(start)+fraction*np.subtract(end,start)))
        if cell is None:
            return 'oriented trajectory leaves observed map'
        touched=[cell]
        if previous is not None and cell[0]!=previous[0] and cell[1]!=previous[1]:
            touched.extend(((cell[0],previous[1]),(previous[0],cell[1])))
        for item in touched:
            if not grid.known[item] or (require_fresh and not observed[item]):
                return 'unobserved or stale centre corridor'
            if grid.no_go[item]:
                if not departing or grid.no_go_depth[item]>depth+1e-6:
                    return 'oriented trajectory enters or deepens a no-go patch'
            elif heading is not None and not grid.direction_allowed(item,heading) and not leaving_direction:
                return 'oriented trajectory uses an excluded approach direction'
        depth=grid.no_go_depth[cell];previous=cell
    return ''


class PassiveHeadingWindow:
    """Observed heading history for a provisional stopping-envelope forecast.

    Net trend alone cannot hide an oscillation: extrema are always included.
    No clipping of a large prediction is allowed. Missing/discontinuous history
    or an envelope wider than a half-turn cannot authorize an oriented fallback.
    This is a geometric forecast to characterize, not a calibrated motion model.
    """
    def __init__(self):
        self.samples=[]

    def observe(self, time, yaw):
        if not math.isfinite(time) or not math.isfinite(yaw):
            self.samples=[]
            return
        if self.samples and time<=self.samples[-1][0]:
            if time==self.samples[-1][0] and yaw==self.samples[-1][1]:
                return
            self.samples=[]
        self.samples.append((time,yaw))
        while len(self.samples)>1 and self.samples[1][0]<=time-.3:
            self.samples.pop(0)

    def offsets(self, time, yaw, stopping_seconds):
        if (not self.samples or not np.isfinite([time,yaw,stopping_seconds]).all() or stopping_seconds<0.
                or abs(self.samples[-1][0]-time)>1e-9
                or abs(math.atan2(math.sin(yaw-self.samples[-1][1]),math.cos(yaw-self.samples[-1][1])))>1e-9
                or time-self.samples[0][0]<.3-1e-9
                or any(b[0]-a[0]>.04+1e-9 for a,b in zip(self.samples,self.samples[1:]))):
            return None
        # Accumulate wrapped increments, preserving true oscillation and wraps.
        angles=[0.]
        for (_,a),(_,b) in zip(self.samples,self.samples[1:]):
            angles.append(angles[-1]+math.atan2(math.sin(b-a),math.cos(b-a)))
        relative=[x-angles[-1] for x in angles]
        trend=angles[-1]/(time-self.samples[0][0])
        projected=trend*stopping_seconds
        low,high=min(0.,projected,*relative),max(0.,projected,*relative)
        return (low,high) if high-low<=math.pi else None


def oriented_stopping_reason(grid,start,end,body_yaw,offsets,*,commanded=True,directional_exit=False,
                             observation_support=None, circular_support=True):
    """Stopping fan with independent body and centre-path uncertainty.

    Cover all translated rectangles for each centre-path ray and all observed /
    forecast body headings. Between-ray centre error is included as a geometric
    reserve; unknown heading uncertainty never silently becomes zero uncertainty.
    """
    if offsets is None:
        return 'passive heading history unavailable for oriented stopping envelope'
    low,high=offsets
    if (not np.isfinite([low,high]).all() or low>high or high-low>math.pi):
        return 'passive heading forecast exceeds bounded stopping envelope'
    distance=math.dist(start,end)
    travel=math.atan2(end[1]-start[1],end[0]-start[0])
    count=max(1,math.ceil((high-low)/.04))
    reserve=distance*(high-low)/count/2
    angles=np.linspace(low,high,count+1) if high>low else (low,)
    for offset in angles:
        endpoint=(start[0]+distance*math.cos(travel+offset),start[1]+distance*math.sin(travel+offset))
        reason=oriented_sweep_reason(grid,start,endpoint,body_yaw,yaw_offsets=offsets,
            commanded=commanded,no_go_exit=True,position_reserve=reserve,directional_exit=directional_exit,
            observation_support=observation_support,circular_support=circular_support)
        if reason:
            return reason
    return ''
