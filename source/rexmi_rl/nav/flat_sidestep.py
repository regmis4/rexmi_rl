"""Opt-in, stopped-on-flat policy handoff for bounded route sidesteps.

No simulator writes or default policy changes. A loss of flat-ground evidence
brakes with the current policy; it never triggers a policy swap on a slope.
"""
from dataclasses import dataclass
import math
import numpy as np


@dataclass
class SideMotion:
    vx: float = 0.
    omega: float = 0.
    vy: float = 0.


def wrap(a):
    return math.atan2(math.sin(a),math.cos(a))


class FlatSidestep:
    MAX_SLOPE = 5.0
    MAX_TILT = 8.0
    MAX_AGE = 5.0
    STOP_SPEED = .05
    STOP_ANGULAR = .15
    STABLE_SECONDS = .5
    RESERVE = .65  # above footprint; open-loop study saw up to .41 m stop drift

    def __init__(self, grid):
        self.grid=grid
        self.policy='rocky_slope'
        self.phase='IDLE'
        self.reason='ready; flat-ground sidesteps enabled'
        self.switches=0
        self.events=[]
        self.stable_since=None
        self.cooldown_until=0.
        self.anchor=None
        self.target=None
        self.started=0.
        self.best=math.inf
        self.progress_at=0.
        self.heading=0.
        self.interrupted=False
        self.suppressed_target=None
        self.vertical_speed=0.
        self.switch_metrics={}

    @property
    def active(self): return self.phase!='IDLE'

    def flat_corridor(self, start, end, reserve=0.):
        """Inspect full footprint capsule, including freshness and obstacle edges."""
        g=self.grid;a=np.asarray(start);b=np.asarray(end)
        radius=g.footprint_radius+reserve
        lo=np.minimum(a,b)-radius;hi=np.maximum(a,b)+radius
        lower=g.world_to_cell(*lo);upper=g.world_to_cell(*hi)
        if lower is None or upper is None: return False,'flat corridor extends outside observed map'
        r0,c0=lower;r1,c1=upper
        x=g.origin[0]+(np.arange(r0,r1+1)+.5)*g.cell_size-g.world_size/2
        y=g.origin[1]+(np.arange(c0,c1+1)+.5)*g.cell_size-g.world_size/2
        xx,yy=np.meshgrid(x,y,indexing='ij');points=np.stack((xx,yy),axis=-1)
        delta=b-a
        t=np.clip(np.sum((points-a)*delta,axis=-1)/max(1e-12,float(delta@delta)),0,1)
        # Include grid-cell extent, not just centres, at the capsule boundary.
        mask=np.linalg.norm(points-(a+t[...,None]*delta),axis=-1)<=radius+g.cell_size*.71
        sl=np.s_[r0:r1+1,c0:c1+1]
        if not mask.any() or not g.known[sl][mask].all(): return False,'unknown terrain in flat corridor'
        ages=g.now-g.surface_seen[sl][mask]
        if not np.isfinite(ages).all() or (ages>self.MAX_AGE).any(): return False,'flat corridor needs fresh observations'
        slopes=g.slope[sl][mask]
        if not np.isfinite(slopes).all() or (slopes>math.tan(math.radians(self.MAX_SLOPE))).any():
            return False,'ground exceeds 5 degree switching limit'
        self.max_ground_slope_deg=float(np.degrees(np.arctan(slopes)).max())
        if g.hazard[sl][mask].any(): return False,'obstacle in sidestep footprint/reserve'
        if not g.segment_clear(start,end): return False,'route clearance or direction is blocked'
        return True,'flat and clear'

    def _body_level(self, upright):
        return math.isfinite(upright) and upright>=math.cos(math.radians(self.MAX_TILT))

    def maybe_start(self, control, xy, yaw, now, upright):
        if self.active or now<self.cooldown_until or control.state!='FOLLOW' or control.target is None:
            return False
        if control.escape_clearance is not None: return False
        delta=np.asarray(control.target)-xy;distance=float(np.linalg.norm(delta))
        if self.suppressed_target is not None and math.dist(control.target,self.suppressed_target)<.4:
            return False
        angle=abs(wrap(math.atan2(delta[1],delta[0])-yaw))
        if distance<.35 or not math.radians(30)<=angle<=math.radians(105):
            self.reason='rocky-slope: route does not need a short sideways correction'
            return False
        target=np.asarray(xy)+delta/max(distance,1e-9)*min(.8,distance)
        good,reason=self.flat_corridor(xy,target,self.RESERVE)
        if not good or not self._body_level(upright):
            self.reason='sidestep unavailable: '+(reason if not good else 'body tilt exceeds 8 degrees')
            return False
        self.phase='STOP_ROCKY';self.reason='stop on flat ground before rough policy'
        self.anchor=np.asarray(xy).copy();self.target=target;self.heading=yaw
        self.started=now;self.stable_since=None;self.interrupted=False
        return True

    def cancel(self, reason='operator request'):
        if not self.active: return
        self.interrupted=True
        if self.policy=='rough':
            self.phase='STOP_ROUGH';self.anchor=None;self.stable_since=None
        else:
            self.phase='IDLE';self.stable_since=None
        self.reason=reason+'; stopping before any policy handoff'

    def _stationary(self, speed, angular_speed, now):
        if speed<self.STOP_SPEED and angular_speed<self.STOP_ANGULAR and abs(self.vertical_speed)<.08:
            if self.stable_since is None: self.stable_since=now
        else: self.stable_since=None
        return self.stable_since is not None and now-self.stable_since>=self.STABLE_SECONDS

    def _switch(self, policy, xy, now):
        self.events.append(dict(time=now,event='flat_policy_switch',source=self.policy,
                                destination=policy,position=[float(v) for v in xy],max_slope_deg=self.MAX_SLOPE,measured_max_slope_deg=self.max_ground_slope_deg,
                                **self.switch_metrics))
        self.policy=policy;self.switches+=1;self.stable_since=None

    def _brake(self, xy, yaw, velocity, angular_z):
        if self.anchor is None:self.anchor=np.asarray(xy).copy()
        world=.7*(self.anchor-xy)-1.2*np.asarray(velocity)
        ca,sa=math.cos(yaw),math.sin(yaw)
        vx=float(np.clip(ca*world[0]+sa*world[1],-.15,.15))
        vy=float(np.clip(-sa*world[0]+ca*world[1],-.15,.15)) if self.policy=='rough' else 0.
        omega=float(np.clip(wrap(self.heading-yaw)-.3*angular_z,-.2,.2))
        end=np.asarray(xy)+.3*np.array([ca*vx-sa*vy,sa*vx+ca*vy])
        if not self.grid.segment_clear(xy,end,escape_clearance=self.grid.footprint_radius,check_direction=False):
            return SideMotion()  # maintain balancing policy, no blind corrective travel
        return SideMotion(vx,omega,vy)

    def tick(self, control, xy, yaw, velocity, angular_z, angular_speed, now, upright=1., vertical_speed=0.):
        xy=np.asarray(xy);velocity=np.asarray(velocity);speed=float(np.linalg.norm(velocity))
        if not self.active:return None
        self.vertical_speed=vertical_speed
        self.switch_metrics=dict(speed_mps=speed,angular_speed_radps=angular_speed,vertical_speed_mps=vertical_speed,
                                 body_tilt_deg=math.degrees(math.acos(float(np.clip(upright,-1,1)))))
        if not np.isfinite([*xy,yaw,*velocity,angular_z,angular_speed,upright,vertical_speed]).all():
            self.cancel('invalid robot state');return SideMotion()
        if self.phase=='STOP_ROCKY':
            good,why=self.flat_corridor(xy,self.target,self.RESERVE)
            if not good or not self._body_level(upright) or now-self.started>5.:
                self.phase='IDLE';self.cooldown_until=now+5.
                self.reason='rough switch cancelled: '+(why if not good else 'could not settle level')
                return SideMotion()
            if self._stationary(speed,angular_speed,now):
                self._switch('rough',xy,now);self.phase='SIDESTEP'
                self.heading=yaw;self.started=now;self.best=math.dist(xy,self.target);self.progress_at=now
                self.reason='rough: short flat-ground sidestep'
                return SideMotion()  # zero command on the exact handoff frame
            return self._brake(xy,yaw,velocity,angular_z)
        if self.phase=='SIDESTEP':
            good,why=self.flat_corridor(xy,self.target,self.RESERVE)
            distance=math.dist(xy,self.target)
            if distance<self.best-.05:self.best=distance;self.progress_at=now
            reasons=[]
            if not good:reasons.append(why)
            if not self._body_level(upright):reasons.append('body tilt exceeds 8 degrees')
            if abs(wrap(self.heading-yaw))>math.radians(20):reasons.append('heading drift exceeds 20 degrees')
            if speed>.35:reasons.append('sidestep speed exceeds 0.35 m/s')
            if distance<.12:reasons.append('sidestep complete')
            if now-self.started>8. or now-self.progress_at>3.:reasons.append('sidestep progress timeout')
            if reasons:
                self.phase='STOP_ROUGH';self.anchor=xy.copy();self.stable_since=None
                self.reason='; '.join(reasons)+'; braking with rough policy'
                if distance>=.12:self.suppressed_target=tuple(control.target if control.target is not None else self.target)
            else:
                world=.7*(self.target-xy)-.35*velocity
                ca,sa=math.cos(yaw),math.sin(yaw)
                vx=float(np.clip(ca*world[0]+sa*world[1],-.12,.15))
                vy=float(np.clip(-sa*world[0]+ca*world[1],-.15,.15))
                omega=float(np.clip(wrap(self.heading-yaw)-.3*angular_z,-.25,.25))
                # Check stopping drift as well as commanded lateral travel.
                drift_end=xy+velocity*max(.5,speed/.2)
                good,why=self.flat_corridor(xy,drift_end,self.RESERVE)
                if good:return SideMotion(vx,omega,vy)
                self.phase='STOP_ROUGH';self.anchor=xy.copy();self.stable_since=None
                self.reason='sidestep drift rejected: '+why
        good,why=self.flat_corridor(xy,xy)
        if good and self._body_level(upright) and self._stationary(speed,angular_speed,now):
            self._switch('rocky_slope',xy,now)
            self.phase='IDLE';self.cooldown_until=now+5.
            self.reason='rocky-slope restored after stable flat-ground stop'
            # Replan from actual position; do not resume an operator STOP/manual hold.
            if control.state in ('FOLLOW','OBSERVE','SETTLE'):
                # A stable handoff may already be within the existing 0.6 m
                # mission arrival radius. Verify it instead of steering away
                # from a nearby goal to chase the next grid-cell centre.
                wp=(control.waypoints[control.wp_idx] if hasattr(control,'waypoints')
                    and control.wp_idx<len(control.waypoints) else None)
                if wp is not None and math.dist(xy,(wp.x,wp.y))<=.6:
                    control.checkpoint=(wp.x,wp.y)
                    control.pending_arrival=control.is_mission_checkpoint=True
                    control.stop_started=None;control.stop_speed=0.
                    control.set_state('SETTLE','verify arrival after flat sidestep',now)
                else:
                    control.request('resume',now)
            return SideMotion()
        if not good or not self._body_level(upright):
            self.stable_since=None
            self.reason='holding rough; handoff blocked: '+(why if not good else 'body tilt exceeds 8 degrees')
        return self._brake(xy,yaw,velocity,angular_z)
