"""Measured RELEASE response retained across a protective interruption.

This optional characterization observer selects no command and labels no
terrain failure. Its one-use token changes only an independently guarded
post-brake mode choice. No clock, destination or travel axis belongs here.
"""
from dataclasses import dataclass, field
import copy
import math


RELEASE_MEMORY_PROFILE=dict(release_memory_enabled=True,
    release_memory_windows_s=[2.,4.],release_memory_max_gap_s=.04,
    release_memory_contract='characterization only; one corrective forward emission per primitive; '
        'nonpositive axis AND uphill displacement over both complete held-RELEASE windows at actual overspeed brake entry; '
        'same-request actually observed earlier forward witness; independent powered re-entry and existing useful-motion admission; '
        'no failure/probability vote, no changed speed boundary or renewed budget')

CORRECTION_RELEASE_PROFILE=dict(correction_release_enabled=True,
    correction_release_contract='only the one actually emitted response-memory correction episode; '
        'existing complete 2 s powered response plus 1 s deficient dwell, minimum 3 s actual forward exposure; '
        'ordinary DRIVE remains 5 s; productive correction continues; protective interruption does not accrue exposure; '
        'retire on first intentional RELEASE/alignment/STOP/result/new request; all guards and outer budgets retained')


@dataclass
class ReleaseResponseMemory:
    enabled: bool = False
    correction_release_enabled: bool = False
    correction_active: bool = False
    used: bool = False
    pending: dict | None = None
    samples: list = field(default_factory=list)
    drive_witness: dict | None = None
    token: dict | None = None

    def clear(self, *, new_request=False):
        self.pending=None;self.samples=[];self.drive_witness=None;self.token=None
        self.correction_active=False
        if new_request:self.used=False

    @staticmethod
    def valid(obs, req, axis):
        g=obs.terrain_gradient
        return bool(req is not None and req.purpose=='mission' and req.direction==1
            and axis is not None and obs.fresh and g is not None
            and all(math.isfinite(v) for v in (*obs.xy,*obs.velocity,obs.time,obs.yaw,
                obs.yaw_rate,obs.up_z,obs.inclination,*axis,*g))
            and abs(math.hypot(*axis)-1.)<1e-6
            and sum(a*b for a,b in zip(axis,g))>=math.tan(math.radians(10.))
            and obs.terrain_relative_tilt is not None)

    def emitted(self, obs, req, axis, state, command):
        self.pending=None
        if state=='RELEASE' and command.vx==0. and command.omega==0.:
            self.correction_active=False
        if not self.enabled or self.used or not self.valid(obs,req,axis):return
        if command.omega!=0.:return
        if not (state=='DRIVE_FORWARD' and command.vx>0. or state=='RELEASE' and command.vx==0.):return
        self.pending=dict(time=obs.time,xy=tuple(obs.xy),axis=tuple(axis),
            action_id=req.action_id,state=state)

    def observe(self, obs, req, axis):
        pending,self.pending=self.pending,None
        if not self.enabled or self.used:return
        if (pending is None or not self.valid(obs,req,axis)
                or pending['action_id']!=req.action_id or pending['axis']!=tuple(axis)
                or not 0.<obs.time-pending['time']<=.04+1e-9):
            self.samples=[]
            return
        if pending['state']=='DRIVE_FORWARD':
            self.drive_witness=dict(action_id=req.action_id,axis=tuple(axis),
                emitted_time=pending['time'],observed_time=obs.time,
                start=pending['xy'],end=tuple(obs.xy))
            self.samples=[]
        else:
            if not self.samples or abs(self.samples[-1][0]-pending['time'])>1e-9:
                self.samples=[(pending['time'],pending['xy'])]
            self.samples.append((obs.time,tuple(obs.xy)))
            while len(self.samples)>1 and self.samples[1][0]<=obs.time-4.:
                self.samples.pop(0)
            if len(self.samples)>512:self.samples=[]

    def interrupt(self, obs, req, axis):
        if (not self.enabled or self.used or self.token is not None
                or not self.valid(obs,req,axis) or not self.samples or self.drive_witness is None):return None
        witness=self.drive_witness
        if (witness['action_id']!=req.action_id or witness['axis']!=tuple(axis)
                or not witness['emitted_time']<witness['observed_time']<=self.samples[0][0]
                or abs(self.samples[-1][0]-obs.time)>1e-9):return None
        if any(not 0.<b[0]-a[0]<=.04+1e-9 for a,b in zip(self.samples,self.samples[1:])):return None
        g=obs.terrain_gradient;norm=math.hypot(*g);windows=[]
        for seconds in (2.,4.):
            eligible=[s for s in self.samples if s[0]<=obs.time-seconds+1e-9]
            if not eligible:return None
            start,xy=eligible[-1];dt=obs.time-start
            if not seconds-1e-9<=dt<=seconds+.04+1e-9:return None
            delta=[a-b for a,b in zip(obs.xy,xy)]
            along=sum(d*a for d,a in zip(delta,axis))/dt
            uphill=sum(d*a/norm for d,a in zip(delta,g))/dt
            if along>0. or uphill>0.:return None
            windows.append(dict(seconds=dt,axis_progress_mps=along,uphill_progress_mps=uphill))
        self.token=dict(action_id=req.action_id,axis=tuple(axis),interrupted_at=obs.time,
            samples=copy.deepcopy(self.samples),windows=windows,drive_witness=copy.deepcopy(witness),
            reason='nonpositive pre-brake release response',probability_validated=False)
        return copy.deepcopy(self.token)

    def matches(self, req, axis):
        return bool(self.enabled and not self.used and self.token is not None
            and req is not None and self.token['action_id']==req.action_id
            and axis is not None and self.token['axis']==tuple(axis)
            and self.drive_witness is not None
            and self.drive_witness['action_id']==req.action_id
            and self.drive_witness['axis']==tuple(axis))

    def consume(self):
        self.used=True
        self.clear()
        self.correction_active=self.correction_release_enabled
