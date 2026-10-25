"""Shared geometric turn admission; relative minima remain experimental."""
import math
import numpy as np
from scipy import ndimage as ndi
from .swept_footprint import oriented_sweep_reason


def minimum_fields(g):
    key=(g.revision,g.cell_size,g.footprint_radius)
    if getattr(g,'_turn_minimum_key',None)==key:
        return g._turn_minimum_fields
    radius=g.footprint_radius/g.cell_size+.71
    a=np.arange(-math.ceil(radius),math.ceil(radius)+1)
    disk=a[:,None]**2+a[None,:]**2<=radius**2
    known=g.known & np.isfinite(g.slope) & np.isfinite(g.surface)
    supported=ndi.minimum_filter(known,footprint=disk,mode='constant',cval=0)
    angles=np.degrees(np.arctan(g.slope))
    score=ndi.maximum_filter(np.where(known,angles,np.inf),footprint=disk,mode='constant',cval=np.inf)
    distance=2*g.footprint_radius/g.cell_size
    probes=[];available=supported.copy()
    for theta in np.arange(8)*math.pi/4:
        dx,dy=(int(round(distance*v)) for v in (math.cos(theta),math.sin(theta)))
        probe=ndi.shift(score,(-dx,-dy),order=0,mode='constant',cval=np.inf,prefilter=False)
        valid=ndi.shift(supported,(-dx,-dy),order=0,mode='constant',cval=0,prefilter=False)
        probes.append(probe);available &= valid
    probes=np.stack(probes)
    prominence=np.zeros_like(score)
    np.subtract(np.median(probes,axis=0),score,out=prominence,where=available)
    minimum=available & (score<=np.min(probes,axis=0)+1e-6) & (prominence>=2.)
    g._turn_minimum_key=key
    g._turn_minimum_fields=(minimum,score,prominence)
    return g._turn_minimum_fields


def stage_reason(g,point):
    reason=g.turn_support_reason(point)
    if reason:return reason
    if g.max_support_slope(point)<=g.motion_turn_limit:return ''
    if not getattr(g,'local_minimum_turns_enabled',False):
        return 'turn requires gentle staging'
    cell=g.world_to_cell(*point)
    if cell is None or not g.traversable(cell):return 'turn stage blocked or unknown'
    if getattr(g,'route_conditioned_turns',False):
        _,score,_=minimum_fields(g)
        # Permissive candidate test only. The actual arrival/departure probes
        # are authoritative in turn_reason; a 2-D basin is not required.
        key=(g.revision,g.cell_size,g.footprint_radius)
        if getattr(g,'_route_stage_key',None)!=key:
            radius=math.ceil(2*g.footprint_radius/g.cell_size)+1
            neighbor=ndi.maximum_filter(np.where(np.isfinite(score),score,-np.inf),
                size=2*radius+1,mode='constant',cval=-np.inf)
            g._route_stage_candidates=np.isfinite(score)&(neighbor-score>=2.)
            g._route_stage_key=key
        return '' if g._route_stage_candidates[cell] else 'no supported route-relative slope minimum'
    minimum,_,_=minimum_fields(g)
    if not minimum[cell]:return 'not a footprint-scale slope minimum'
    return ''


def route_minimum_reason(g,point,incoming,outgoing):
    _,score,_=minimum_fields(g)
    distance=2*g.footprint_radius
    upstream=(point[0]-distance*math.cos(incoming),point[1]-distance*math.sin(incoming))
    downstream=(point[0]+distance*math.cos(outgoing),point[1]+distance*math.sin(outgoing))
    cells=[g.world_to_cell(*p) for p in (point,upstream,downstream)]
    if any(c is None or not np.isfinite(score[c]) for c in cells):
        return 'route-relative minimum needs observed comparison footprints'
    center,previous,following=(float(score[c]) for c in cells)
    if center>min(previous,following)+1e-6 or max(previous,following)-center<2.:
        return 'not a minimum along the requested approach and departure'
    return ''


def turn_reason(g,point,incoming,outgoing,*,stage_headings=None):
    if getattr(g,'checked_small_turns',False):
        from .checked_turn import delta, MAX_CORRECTION, static_reason
        if abs(delta(incoming,outgoing))<=MAX_CORRECTION+1e-8:
            return static_reason(g,point,incoming,outgoing)
    reason=stage_reason(g,point)
    if reason:return reason
    # Preserve established gentle-turn contract; new admissions check the
    # entire requested stationary rotation, not just its final heading.
    if g.max_support_slope(point)<=g.motion_turn_limit:return ''
    if getattr(g,'route_conditioned_turns',False):
        reason=route_minimum_reason(g,point,*(stage_headings or (incoming,outgoing)))
        if reason:return reason
    delta=math.atan2(math.sin(outgoing-incoming),math.cos(outgoing-incoming))
    reason=oriented_sweep_reason(g,point,point,incoming,
        yaw_offsets=(min(0.,delta),max(0.,delta)),require_fresh=False,
        commanded=False,circular_support=False)
    if reason:return reason
    cell=g.world_to_cell(*point)
    gx,gy=float(g.gx[cell]),float(g.gy[cell])
    if not math.isfinite(gx) or not math.isfinite(gy):return 'unmeasured turn gradient'
    lo,hi=sorted((incoming,incoming+delta))
    angles=[lo,hi]
    # Extrema of |gradient dot lateral-axis| occur perpendicular to gradient.
    peak=math.atan2(gy,gx)+math.pi/2
    for k in range(math.floor((lo-peak)/math.pi)-1,math.ceil((hi-peak)/math.pi)+2):
        if lo<=peak+k*math.pi<=hi:angles.append(peak+k*math.pi)
    across=max(abs(-gx*math.sin(t)+gy*math.cos(t)) for t in angles)
    if math.degrees(math.atan(across))>g.side_limit_deg+1e-6:
        return 'requested turn crosses side-slope limit'
    return ''
