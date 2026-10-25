"""Small yaw corrections with translated footprints and checked route rejoins.

The measured-velocity forecast is provisional characterization geometry, not a
calibrated stopping model. It never grants occupied-exclusion escape permission.
"""
import math
import numpy as np

from .swept_footprint import oriented_sweep_reason, oriented_swept_cells

MAX_CORRECTION = .35
YAW_COMMAND = .35
HEADING_TOLERANCE = .10


def delta(a, b):
    return math.atan2(math.sin(b-a), math.cos(b-a))


def physical_reason(g, start, end, yaw, offsets, *, sweep=None):
    """Check every heading against gradients across the entire swept footprint."""
    lo,hi=offsets
    if not np.isfinite([*start,*end,yaw,lo,hi]).all() or lo>hi or hi-lo>math.pi:
        return 'invalid or excessive correction uncertainty'
    if sweep is None:
        reason=oriented_sweep_reason(g,start,end,yaw,yaw_offsets=offsets,
            require_fresh=False,commanded=False,circular_support=False)
    else:
        reason=sweep(start,end,yaw,yaw_offsets=offsets,commanded=False,circular_support=False)
    if reason:return reason
    cells=oriented_swept_cells(g,start,end,yaw,offsets)
    if cells is None:return 'correction leaves observed map'
    if g.no_go[cells].any():return 'correction intersects a no-go footprint'
    gx,gy=g.gx[cells],g.gy[cells]
    if not np.isfinite(gx).all() or not np.isfinite(gy).all():
        return 'unmeasured gradient in correction envelope'
    low,high=yaw+lo,yaw+hi
    across=np.maximum(np.abs(-gx*math.sin(low)+gy*math.cos(low)),
                      np.abs(-gx*math.sin(high)+gy*math.cos(high)))
    peak=np.arctan2(gy,gx)+math.pi/2
    interior=np.ceil((low-peak)/math.pi)<=np.floor((high-peak)/math.pi)
    across=np.where(interior,np.hypot(gx,gy),across)
    if np.any(across>math.tan(math.radians(g.side_limit_deg))+1e-8):
        return 'correction crosses side-slope limit'
    return ''


def static_reason(g, point, incoming, outgoing):
    change=delta(incoming,outgoing)
    if abs(change)>MAX_CORRECTION+1e-8:
        return 'correction exceeds bounded angle'
    return physical_reason(g,point,point,incoming,(min(0.,change),max(0.,change)))


def forecast(observation, target):
    change=delta(observation.yaw,target)
    # Include nominal correction duration and the existing stopping allowance.
    horizon=abs(change)/YAW_COMMAND+.25+.5*observation.speed/.4
    end=tuple(np.asarray(observation.xy)+horizon*np.asarray(observation.velocity))
    momentum=observation.yaw_rate*(abs(change)/YAW_COMMAND+.25+observation.speed/.4)
    offsets=(min(0.,change,momentum,change+momentum),max(0.,change,momentum,change+momentum))
    return end,offsets


def runtime_reason(control, target):
    o=control.obs
    if not 0.<control.observation_dt<=.041:
        return 'correction requires contiguous observations'
    end,offsets=forecast(o,target)
    return physical_reason(control.grid,o.xy,end,o.yaw,offsets,sweep=control._physical_sweep)


def rejoins(path, vertex, xy, yaw):
    """A fixed small inventory of forward points inside the active route leg.

    Returned complete paths are proposals only. Caller checks connector, join
    turn, retained suffix and actual exterior endpoint on the current map.
    """
    if not 1<=vertex<len(path):return []
    a,b=np.asarray(path[vertex-1]),np.asarray(path[vertex])
    length=float(np.linalg.norm(b-a))
    if length<1e-8:return []
    axis=(b-a)/length
    progress=float(np.clip(np.dot(np.asarray(xy)-a,axis),0.,length))
    if np.linalg.norm(np.asarray(xy)-(a+progress*axis))>1.:
        return []
    choices=[];seen=set()
    for ahead in (.5,1.,2.):
        along=min(length,progress+ahead)
        if along-progress<.25:continue
        point=tuple(a+along*axis)
        if point in seen:continue
        seen.add(point)
        heading=math.atan2(point[1]-xy[1],point[0]-xy[0])
        if abs(delta(yaw,heading))>MAX_CORRECTION:continue
        suffix=list(path[vertex:]) if along<length-1e-8 else list(path[vertex+1:])
        candidate=[tuple(xy),point]+suffix
        choices.append((abs(delta(yaw,heading)),math.dist(xy,point),candidate))
    return [item[2] for item in sorted(choices)]


def diagnostic_copy(control):
    """Geometry is shared read-only; admission counters/events are isolated."""
    import copy
    result=copy.copy(control)
    result.attempt_events=[]
    return result


def continuation_audit(control, path):
    """Explain existing route checks without changing their admission decision."""
    from . import exit_region, remote_routes
    from .staged_routes import motion_route_reason, observed_edge_reason
    control=diagnostic_copy(control)
    g,o=control.grid,control.obs
    checks=[dict(stage='exterior',reason=exit_region.reason(g,path[-1],control.exit_rim_radius)),
            dict(stage='initial_heading',reason=motion_route_reason(g,path[:2],o.yaw))]
    for index,(a,b) in enumerate(zip(path,path[1:])):
        checks.append(dict(stage='connector' if index==0 else 'suffix',index=index,
            observed_reason=observed_edge_reason(g,a,b,o.xy),
            physical_reason=remote_routes.physical_profile(control,[a,b])[0]))
    for index in range(1,len(path)-1):
        checks.append(dict(stage='joining_turn' if index==1 else 'suffix_turn',index=index,
            reason=motion_route_reason(g,path[index-1:index+2])))
    return checks


def record_refusal(control, candidates):
    """Bounded exact-map evidence, captured before a recovery changes the state."""
    import copy
    import json
    from pathlib import Path
    import time
    from . import planning_snapshot
    if control.rejoin_refusals>=4:return
    control.rejoin_refusals+=1
    o=control.obs;started=time.perf_counter()
    event=dict(event='checked_rejoin_refusal',time=o.time,position=list(o.xy),
        yaw=o.yaw,velocity=list(o.velocity),yaw_rate=o.yaw_rate,
        observation_dt=control.observation_dt,route=control.path,vertex=control.vertex,
        map_revision=control.grid.revision,candidates=candidates,
        scope='exact refusal geometry and measured motion; not a simulator restart')
    directory=getattr(control,'planning_snapshot_dir',None)
    if directory is None:
        event['snapshot']=dict(status='snapshot output not configured')
    else:
        shadow=copy.copy(control)
        directory=Path(directory)
        shadow.planning_snapshot_dir=directory.with_name(directory.name+'-rejoins')
        try:
            event['snapshot']=planning_snapshot.save(shadow,o.xy,o.yaw,o.time)
            if event['snapshot'] and event['snapshot'].get('status')=='saved':
                Path(event['snapshot']['path']).with_suffix('.json').write_text(
                    json.dumps(event,indent=2,allow_nan=False)+'\n')
        except (OSError,ValueError) as error:
            event['snapshot']=dict(status='capture failed',reason=str(error))
    event['capture_ms']=1000*(time.perf_counter()-started)
    control.attempt_events.append(event)
