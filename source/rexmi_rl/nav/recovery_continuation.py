"""One fixed, geometry-checked recovery region; no policy or simulator access.

Survey geometry plans the distant approach. Live command stopping guards and
fresh staging observations remain mandatory. This is a characterization
contract, not a calibrated probability or a prediction of future passive yaw.
"""
from dataclasses import dataclass
import math
import numpy as np

from .swept_footprint import oriented_sweep_reason, arrival_ray_interval

RECOVERY_CONTINUATION_PROFILE = dict(
    recovery_continuation_enabled=True,
    recovery_continuation_outer_radius_m=1.,
    recovery_continuation_inner_radius_m=.5,
    recovery_continuation_corridor_width_m=.5,
    recovery_continuation_max_distance_m=4.,
    recovery_continuation_contract='characterization only; one frozen reverse staging certificate per recovery episode, same action/deadline/axis/progress/budget; surveyed full approach and region, live fresh stopping, fully fresh region before inner-core staging; nominal prediction censored and region phase unvalidated; settled region is censored reposition, never endpoint success')


@dataclass(frozen=True)
class RecoveryContinuation:
    certificate_id: str
    action_id: str
    origin: tuple
    center: tuple
    issued: float
    deadline: float
    body_yaw: float
    heading_offsets: tuple
    response_heading: float
    outer_radius: float = 1.
    inner_radius: float = .5
    corridor_width: float = .5

    def __post_init__(self):
        values=(*self.origin,*self.center,self.issued,self.deadline,self.body_yaw,
                *self.heading_offsets,self.response_heading,self.outer_radius,self.inner_radius,self.corridor_width)
        if (not all(math.isfinite(v) for v in values) or self.deadline<=self.issued
                or not .1<=math.dist(self.origin,self.center)<=4.+1e-9
                or self.outer_radius!=1. or self.inner_radius!=.5 or self.corridor_width!=.5
                or len(self.heading_offsets)!=2 or self.heading_offsets[0]>self.heading_offsets[1]):
            raise ValueError('Invalid bounded recovery certificate')

    @property
    def axis(self):
        length=math.dist(self.origin,self.center)
        return tuple((b-a)/length for a,b in zip(self.origin,self.center))

    def contains_approach(self, xy):
        delta=np.subtract(xy,self.origin);axis=np.asarray(self.axis)
        along=float(delta@axis);cross=abs(float(delta@np.array([-axis[1],axis[0]])))
        return (-.1<=along<=math.dist(self.origin,self.center)+self.outer_radius
                and cross<=self.corridor_width) or math.dist(xy,self.center)<=self.outer_radius


def region_reason(grid, certificate, *, fresh=False):
    cells=grid._support_cells(certificate.center,reserve=certificate.outer_radius+grid.clearance)
    if cells is None or not grid.known[cells].all():return 'unobserved recovery staging region'
    if (grid.hazard[cells].any() or grid.no_go[cells].any() or grid.approach_blocked[cells].any()):
        return 'recovery staging region contains obstacle or exclusion'
    if not np.isfinite(grid.slope[cells]).all() or (grid.slope[cells]>math.tan(math.radians(12.))).any():
        return 'recovery staging region is not wholly gentle'
    if fresh and not grid.fresh[cells].all():return 'recovery staging region is not fully fresh'
    return ''


def approach_reason(grid, certificate):
    """Continuous conservative body support plus checked directional corridors."""
    c=certificate
    cells=grid._support_cells(c.origin,c.center,reserve=c.corridor_width+grid.clearance)
    if cells is None or not grid.known[cells].all():return 'unobserved recovery approach band'
    if grid.hazard[cells].any():return 'obstacle in recovery approach band'
    if not np.isfinite(grid.slope[cells]).all() or (grid.slope[cells]>math.tan(math.radians(35.))).any():
        return 'recovery approach exceeds terrain boundary'
    normal=np.array([-c.axis[1],c.axis[0]])
    for offset in np.linspace(-c.corridor_width,c.corridor_width,
                              math.ceil(2*c.corridor_width/(grid.cell_size*.5))+1):
        a=np.asarray(c.origin)+offset*normal;b=np.asarray(c.center)+offset*normal
        reason=oriented_sweep_reason(grid,a,b,c.body_yaw,require_fresh=False,no_go_exit=True)
        if reason:return reason
    # Include the full observed/forecast heading fan, not just two outside rays.
    lo,hi=c.heading_offsets
    for offset in np.linspace(lo,hi,max(2,math.ceil((hi-lo)/.025)+1)):
        interval=arrival_ray_interval(c.origin,c.center,c.body_yaw+offset,-1,c.inner_radius-.1)
        if interval is None:return 'reverse response fan misses recovery entry core'
        travel=c.body_yaw+offset+math.pi
        end=np.asarray(c.origin)+(interval[1]+c.outer_radius-c.inner_radius+.5)*np.array([math.cos(travel),math.sin(travel)])
        reason=oriented_sweep_reason(grid,c.origin,end,c.body_yaw,yaw_offsets=c.heading_offsets,
                                     position_reserve=.1,require_fresh=False,no_go_exit=True)
        if reason:return reason
    return region_reason(grid,c)


def select_region(grid, obs, request, samples, forecast):
    """Bounded deterministic geometry search; all candidates come from the map."""
    if forecast is None or len(samples)<2:return None,'missing recovery heading evidence'
    elapsed=obs.time-samples[0][0]
    if (not 2.-1e-9<=elapsed<=2.04+1e-9 or abs(samples[-1][0]-obs.time)>1e-9
            or any(not 0.<b[0]-a[0]<=.04+1e-9 for a,b in zip(samples,samples[1:]))
            or not all(math.isfinite(v) for t,xy,yaw in samples for v in (t,*xy,yaw))):
        return None,'incomplete continuous recovery response window'
    delta=np.subtract(obs.xy,samples[0][1]);gradient=obs.terrain_gradient
    if (gradient is None or not np.isfinite(gradient).all() or np.linalg.norm(gradient)<1e-6
            or float(delta@np.asarray(gradient))/np.linalg.norm(gradient)/elapsed>=-.02
            or np.linalg.norm(delta)/elapsed<.02):
        return None,'recovery response lacks measured downhill progress'
    wrap=lambda angle:math.atan2(math.sin(angle),math.cos(angle))
    response=math.atan2(delta[1],delta[0]);travel=obs.yaw+math.pi
    angles=[wrap(yaw-obs.yaw) for _,_,yaw in samples]+[wrap(response-travel),*forecast,0.]
    lo,hi=min(angles),max(angles)
    if hi-lo>math.pi/2:return None,'recovery response heading spread is too wide'
    heading=travel+(lo+hi)/2
    last='no bounded surveyed staging region'
    for length in np.arange(1.2,4.01,.2):
        center=tuple(np.asarray(obs.xy)+length*np.array([math.cos(heading),math.sin(heading)]))
        c=RecoveryContinuation(request.action_id+':region',request.action_id,tuple(obs.xy),center,
                              obs.time,request.deadline,obs.yaw,(lo,hi),response)
        last=region_reason(grid,c) or approach_reason(grid,c)
        if not last:return c,''
    return None,last
