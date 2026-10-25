"""Opt-in agent-assisted diagnostic; pauses are frozen physics, not zero input.

Approval IDs are an audit trail, not an authentication mechanism. The executor
must obtain the independent reviews before placing a ticket in the inbox.
"""
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from .checkpoint_control import Motion
from .swept_footprint import PassiveHeadingWindow, oriented_stopping_reason, oriented_sweep_reason
from ..drive_adapter import directional_command


BUTTONS = {'forward': (.4, 0., 10.), 'back': (-.4, 0., 10.),
           'left': (0., .35, 2.), 'right': (0., -.35, 2.),
           'release': (0., 0., 20.),
           'forward_left': (.4, .35, 3.), 'forward_right': (.4, -.35, 3.)}


class ReviewedButtons:
    MAX_TICKETS = 80
    MAX_SECONDS = 300.

    def __init__(self, directory, grid, dt):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.grid, self.dt = grid, dt
        self.sequence = 0
        self.active = None
        self.end = 0.
        self.terminal = ''
        self.pause_reason = 'initial observation; settling requires a RELEASE ticket'
        self.heading = PassiveHeadingWindow()
        self.milestones = []
        self.state = None
        self.publications = 0
        self.settle_since = None
        self.peak_speed = self.peak_tilt = 0.
        self.initializing = True
        self.initialization_stable_since = None
        self.spawn_xy = None
        prior = getattr(grid, 'terrain_prior', {})
        self.static_prior = (prior.get('kind') == 'simulator_development'
                             and not prior.get('parent_geometry_id')
                             and bool(prior.get('file_sha256')) and bool(prior.get('geometry_id')))
        self.prior_support = (grid.known & np.isfinite(grid.surface)).copy() if self.static_prior else None
        self.prior_surface = grid.surface.copy() if self.static_prior else None
        if self.static_prior:
            self.prior_support.setflags(write=False)
            self.prior_surface.setflags(write=False)
        self.prior_admissions = 0
        self.admission_keys = set()
        self.progress_samples = []
        self.descent_overspeed = False
        self.ground_admission_tickets = set()

    def descent_phase(self):
        return not self.initializing and len(self.milestones) == 1

    def current_support_reason(self):
        """Only a circular slope veto may fall back to the measured body shape."""
        s, g = self.current, self.grid
        xy = s['position'][:2]
        reason = g.support_reason(xy)
        if reason.startswith('slope >35 degrees'):
            return oriented_sweep_reason(g,xy,xy,s['yaw'],commanded=False,
                                         circular_support=False)
        if reason: return reason
        cells = g._support_cells(xy)
        if cells is None or not g.fresh[cells].all():
            return 'actual body footprint lacks fresh observations'
        return ''

    def sample_progress(self):
        """Terrain-relative progress, not suspension/body height bounce."""
        s, g = self.current, self.grid
        xy, now = s['position'][:2], s['time']
        cell = g.world_to_cell(*xy)
        if (cell is None or self.current_support_reason()
                or not np.isfinite([g.gx[cell], g.gy[cell]]).all()):
            self.progress_samples.clear()
            return
        rate = float(np.dot([g.gx[cell], g.gy[cell]], s['velocity'][:2]))
        if self.progress_samples:
            last = self.progress_samples[-1][0]
            if now == last: return  # Frozen observations are not new evidence.
            if now < last or now-last > .04+1e-9: self.progress_samples.clear()
        self.progress_samples.append((now, rate))
        while len(self.progress_samples)>2 and self.progress_samples[1][0] <= now-.5:
            self.progress_samples.pop(0)

    def projected_progress(self):
        now = self.current['time']
        samples = self.progress_samples
        if (len(samples)<2 or abs(samples[-1][0]-now)>1e-9
                or samples[0][0]>now-.5+1e-9): return None
        times, rates = np.asarray(samples).T
        start = now-.5
        mask = times>start
        times = np.r_[start, times[mask]]
        rates = np.r_[np.interp(start, *np.asarray(samples).T), rates[mask]]
        return float(np.sum(np.diff(times)*(rates[:-1]+rates[1:])*.5))

    def descent_permission(self):
        if not self.descent_phase(): return False
        s, g = self.current, self.grid
        xy, yaw = s['position'][:2], s['yaw']
        cell = g.world_to_cell(*xy)
        if cell is None or self.current_support_reason(): return False
        grade = g.gx[cell]*math.cos(yaw)+g.gy[cell]*math.sin(yaw)
        progress = self.projected_progress()
        return bool(np.isfinite(grade) and grade<0. and progress is not None and progress<=.01)

    def duration_limit(self, button, profile='normal'):
        if profile=='descent' and button in ('left', 'right') and self.descent_permission(): return 8.
        return BUTTONS[button][2]

    def descent_sweep_reason(self, start, end, *, observed, reserve=0., commanded=False):
        """Purposeful descent geometry: full rotation envelope, no escape waiver.

        The configured body rectangle already contains the physical reserve.
        Circumscribing it covers every heading during yaw, without adding the
        unrelated normal staging/turn reserve. Centre direction bans still apply.
        """
        g = self.grid
        radius = max(g.footprint_radius, math.hypot(g.half_length,g.half_width))
        cells = g._support_cells(start,end,radius-g.footprint_radius+reserve)
        if cells is None or not g.known[cells].all(): return 'unobserved descent rotation envelope'
        if not observed[cells].all(): return 'stale descent rotation envelope'
        if g.hazard[cells].any(): return 'obstacle in descent rotation envelope'
        if not np.isfinite(g.slope[cells]).all() or (g.slope[cells]>math.tan(math.radians(35))).any():
            return 'descent rotation envelope exceeds 35 degrees'
        if commanded:
            reason = g.grade_reason(start,end,uphill_limit=0.,side_limit=28.)
            if reason: return reason
        distance = math.dist(start,end)
        heading = g.heading_bin(math.atan2(end[1]-start[1],end[0]-start[0])) if distance>1e-9 else None
        previous = None
        for t in np.linspace(0.,1.,max(2,math.ceil(distance/(g.cell_size*.25))+1)):
            cell = g.world_to_cell(*(np.asarray(start)+t*np.subtract(end,start)))
            if cell is None: return 'descent corridor leaves map'
            touched = [cell]
            if previous is not None and cell[0]!=previous[0] and cell[1]!=previous[1]:
                touched += [(cell[0],previous[1]),(previous[0],cell[1])]
            for item in touched:
                if g.no_go[item]: return 'descent corridor enters or occupies no-go patch'
                if heading is not None and not g.direction_allowed(item,heading):
                    return 'descent corridor uses excluded approach direction'
            previous = cell
        return ''

    def descent_command_reason(self, vx, omega, observed):
        """Recheck the whole stopping-length curved command on each physics tick."""
        s, g = self.current, self.grid
        xy, yaw = np.array(s['position'][:2]), s['yaw']
        speed = max(s['speed'],abs(vx))
        distance = max(.4,speed*speed/.4+.1)
        if not vx:
            return self.descent_sweep_reason(xy,xy,observed=observed)
        horizon = distance/abs(vx)
        # Chord error reserve encloses the continuous curve between samples.
        count = max(1,math.ceil(distance/(g.cell_size*.25)), math.ceil(abs(omega*horizon)/.04))
        step = horizon/count
        reserve = abs(vx/omega)*(1-math.cos(abs(omega*step)/2)) if omega else 0.
        previous = xy
        for t in np.linspace(step,horizon,count):
            end = (xy + np.array([vx/omega*(math.sin(yaw+omega*t)-math.sin(yaw)),
                                  vx/omega*(math.cos(yaw)-math.cos(yaw+omega*t))]) if omega else
                   xy + vx*t*np.array([math.cos(yaw),math.sin(yaw)]))
            reason = self.descent_sweep_reason(previous,end,observed=observed,reserve=reserve,commanded=True)
            if reason: return reason
            previous = end
        return ''


    def forecast_support(self):
        """Static simulator geometry support; never advertised as fresh LiDAR."""
        if not self.static_prior: return None
        g = self.grid
        agreeing = (self.prior_support & g.known & np.isfinite(g.surface)
                    & (np.abs(g.surface-self.prior_surface) <= g.MATCH_TOLERANCE))
        return g.fresh | agreeing

    def initialization_reason(self):
        s, g = self.current, self.grid
        xy = s['position'][:2]
        if math.dist(xy, self.spawn_xy) > .5: return 'initialization left 0.5 m spawn region'
        reason = g.support_reason(xy)
        if reason: return reason
        cells = g._support_cells(xy)
        if cells is None or not g.fresh[cells].all(): return 'initial footprint is not freshly observed'
        if g.hazard[cells].any() or g.no_go[cells].any(): return 'initial footprint hazard or exclusion'
        return ''

    def log(self, event, **fields):
        with (self.directory/'ledger.jsonl').open('a') as out:
            out.write(json.dumps(dict(event=event, **fields), allow_nan=False)+'\n')

    def observe(self, now, xyz, yaw, velocity, up):
        values = [now, *xyz, yaw, *velocity, up]
        if not np.isfinite(values).all():
            self.terminal = 'nonfinite observation'
            return
        self.heading.observe(now, yaw)
        speed = math.hypot(*velocity[:2])
        tilt = math.degrees(math.acos(max(-1., min(1., up))))
        self.peak_speed = max(self.peak_speed, speed)
        self.peak_tilt = max(self.peak_tilt, tilt)
        if tilt > 50.: self.terminal = 'body tilt exceeds 50 degrees'
        # A descent overspeed latch survives ticket expiry and loss of grade
        # eligibility. Only measured deceleration clears RELEASE-only ownership.
        if not self.initializing and ((self.active and self.active.get('profile')=='descent') or self.descent_overspeed):
            self.descent_overspeed = speed>.8
            speed_limit = 1.
        else:
            speed_limit = 1.1 if self.initializing else .8
        if speed > speed_limit: self.terminal = f'actual horizontal speed exceeds {speed_limit} m/s'
        if now >= self.MAX_SECONDS-1e-9: self.terminal = '300 simulated second budget exhausted'
        self.current = dict(time=now, position=list(xyz), yaw=yaw,
                            velocity=list(velocity), speed=speed, tilt=tilt)
        self.sample_progress()
        if self.spawn_xy is None: self.spawn_xy = tuple(xyz[:2])
        if self.terminal: return
        if self.initializing:
            reason = self.initialization_reason()
            if reason:
                self.terminal = 'initialization rejected: '+reason
                return
            stable = (math.sqrt(sum(float(v)**2 for v in velocity)) <= .12 and tilt <= 25.
                      and self.active and self.active['button'] == 'release')
            if not stable: self.initialization_stable_since = None
            elif self.initialization_stable_since is None: self.initialization_stable_since = now
            elif now-self.initialization_stable_since >= .6-1e-9:
                self.initializing = False
                self.log('initialization_complete', **self.current)
                self.finish(now, xyz, 'initialization complete; ordinary movement guards permanently armed')
            if self.initializing:
                if now >= 3.-1e-9: self.terminal = 'initialization failed to settle within 3 seconds'
                return
        xy = xyz[:2]
        milestone = None
        if not self.milestones and math.dist(xy, (10.0913, 1.5252)) <= .8:
            milestone = 'boulder checkpoint'
        elif len(self.milestones) == 1 and math.hypot(*xy) <= 1.:
            milestone = 'crater floor'
        elif len(self.milestones) == 2 and xy[1] > 0. and math.hypot(*xy) >= 12.:
            milestone = 'northern exterior'
        if milestone and not self.grid.support_reason(xy):
            self.milestones.append(dict(name=milestone, time=now, position=list(xyz)))
            self.log('milestone', **self.milestones[-1])
            self.pause_reason = 'milestone reached: '+milestone
            if self.active: self.finish(now, xyz, self.pause_reason)
        settling = (len(self.milestones) == 3 and xy[1] > 0. and math.hypot(*xy) >= 12.
                    and self.active and self.active['button'] == 'release'
                    and math.sqrt(sum(float(v)**2 for v in velocity)) <= .12
                    and not self.guard('release'))
        if not settling: self.settle_since = None
        elif self.settle_since is None: self.settle_since = now
        elif now-self.settle_since >= .6-1e-9:
            self.terminal = 'agent-assisted traversal complete; northern exterior settled'

    def corridor_reason(self, xy, end, yaw, now, speed, commanded, *, static_support=True):
        g = self.grid
        observed = self.forecast_support() if static_support else None
        circular_support = True
        if commanded:
            reason = g.support_reason(xy,end)
            if reason.startswith('slope >35 degrees'):
                circular_support = False
            else:
                reason = reason or g.grade_reason(xy,end)
                if reason: return reason
                if g.segment_clear(xy,end,observation_support=observed): return ''
        else:
            reason = g.drift_segment_reason(xy,end,observation_support=observed)
            if reason.startswith('support: slope >35 degrees'):
                circular_support = False
            elif not reason or not reason.startswith('obstacle:'):
                return reason
        offsets = self.heading.offsets(now, yaw, speed/.2)
        # This diagnostic starts with fresh evidence and grants no exclusion
        # escape permission. The oriented helper's legacy no_go_exit cannot
        # authorize departure from an occupied exclusion here.
        cell = g.world_to_cell(*xy)
        if cell is None or g.no_go[cell]: return 'occupied no-go patch'
        return oriented_stopping_reason(g, xy, end, yaw, offsets, commanded=commanded,
                                        observation_support=observed,circular_support=circular_support)

    def guard(self, button, *, static_support=True, profile='normal'):
        s, g = self.current, self.grid
        xy, yaw, speed, now = s['position'][:2], s['yaw'], s['speed'], s['time']
        vx, omega, _ = BUTTONS[button]
        if profile=='descent' and self.initializing: return 'descent profile unavailable during initialization'
        if profile=='descent' and button=='back': return 'reverse requires normal profile'
        if self.initializing:
            if button != 'release': return 'initialization permits reviewed RELEASE only'
            return self.initialization_reason()
        cell = g.world_to_cell(*xy)
        if cell is None or not g.fresh[cell]: return 'current footprint centre is stale or unknown'
        if g.hazard[cell] or g.no_go[cell]: return 'current footprint centre is hazardous or excluded'
        reason = self.current_support_reason()
        if reason: return reason
        if self.descent_overspeed and button != 'release': return 'descent overspeed: RELEASE only until speed <=0.8 m/s'
        descent = profile=='descent' and self.descent_permission()
        if profile=='descent' and not descent and not (self.descent_overspeed and button=='release'):
            return 'descent permission unavailable or ended'
        if button.startswith('forward_') and not descent: return 'combined buttons require descent permission'
        observed = self.forecast_support() if static_support else None
        if observed is None: observed = g.fresh
        if omega and descent:
            reason = self.descent_command_reason(vx,omega,observed)
            if reason: return 'descent command: '+reason
        if omega and not descent:
            slope = g.max_support_slope(xy)
            if not math.isfinite(slope) or slope > 15.: return 'yaw requires footprint slope <=15 degrees'
            if s['tilt'] > 25.: return 'yaw requires body tilt <=25 degrees'
            reason = g.turn_support_reason(xy)
            if reason: return reason
            if not g.segment_clear(xy, xy): return 'rotation clearance blocked'
        if speed > .1:
            distance = speed*speed/.4+.1
            end = tuple(np.asarray(xy)+distance*np.asarray(s['velocity'][:2])/speed)
            reason = self.corridor_reason(xy, end, yaw, now, speed, False, static_support=static_support)
            if reason: return 'actual drift: '+reason
        if vx and descent and vx>0:
            if not omega:
                reason = self.descent_command_reason(vx,omega,observed)
                if reason: return 'descent command: '+reason
        elif vx:
            distance = max(.4, max(speed, abs(vx))**2/.4+.1)
            end = (xy[0]+math.copysign(distance, vx)*math.cos(yaw),
                   xy[1]+math.copysign(distance, vx)*math.sin(yaw))
            reason = self.corridor_reason(xy, end, yaw, now, max(speed, abs(vx)), True, static_support=static_support)
            if reason: return 'commanded motion: '+reason
        return ''

    def ground_fallback_contexts(self, button, profile):
        """Pure evidence query; only accepted emission may log an admission."""
        s,g = self.current,self.grid
        xy,yaw,speed = s['position'][:2],s['yaw'],s['speed']
        paths = [('current body',xy,0.)]
        if speed>.1:
            end=np.asarray(xy)+(speed*speed/.4+.1)*np.asarray(s['velocity'][:2])/speed
            paths.append(('actual stopping',end,speed))
        vx=BUTTONS[button][0]
        if vx and profile=='normal':
            forecast_speed=max(speed,abs(vx));distance=max(.4,forecast_speed**2/.4+.1)
            end=np.asarray(xy)+math.copysign(distance,vx)*np.array([math.cos(yaw),math.sin(yaw)])
            paths.append(('commanded stopping',end,forecast_speed))
        contexts=[]
        for context,end,forecast_speed in paths:
            reason=g.support_reason(xy,end)
            if reason.startswith('slope >35 degrees'):
                contexts.append(dict(context=context,circular_reason=reason,endpoint=list(end),
                    heading_offsets=(0.,0.) if context=='current body' else
                        self.heading.offsets(s['time'],yaw,forecast_speed/.2)))
        return contexts

    def publish(self):
        g = self.grid
        self.publications += 1
        slope = g.max_support_slope(self.current['position'][:2])
        state = dict(self.current, sequence=self.sequence+1, milestones=self.milestones,
                     reason=self.terminal or self.pause_reason, terminal=bool(self.terminal),
                     footprint_slope_deg=slope if math.isfinite(slope) else None,
                     command_checks={button:self.guard(button) for button in BUTTONS},
                     descent_command_checks={button:self.guard(button,profile='descent') for button in BUTTONS},
                     classification='agent-assisted diagnostic; physics frozen between tickets')
        state.update(phase='INITIALIZE' if self.initializing else 'TRAVERSE',
                     initialization_limits=dict(deadline_s=3.,actual_horizontal_speed_mps=1.1,
                         spawn_displacement_m=.5,settled_3d_speed_mps=.12,settled_dwell_s=.6),
                     active_speed_limit_mps=1.1 if self.initializing else (1. if self.descent_overspeed else .8),
                     descent_permission=self.descent_permission(),
                     projected_progress_0_5s_m=self.projected_progress(),
                     descent_release_only=self.descent_overspeed,
                     button_duration_limits={name:self.duration_limit(name) for name in BUTTONS},
                     descent_button_duration_limits={name:self.duration_limit(name,'descent') for name in BUTTONS})
        state['forecast_observation_contract'] = ('fresh LiDAR or unchanged original simulator survey; '
            'fresh actual body footprint required' if self.static_prior else 'fresh LiDAR only')
        # Persist the exact observed map at every decision, not just final map.
        state['map_file'] = f'state-{self.publications:03d}.npz'
        np.savez_compressed(self.directory/state['map_file'], **g.survey_snapshot())
        state['map_sha256'] = hashlib.sha256((self.directory/state['map_file']).read_bytes()).hexdigest()
        state['digest'] = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
        self.state = state
        target = self.directory/'state.json'
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2, allow_nan=False))
        temporary.replace(target)
        self.log('paused', **state)
        print('[reviewed buttons] '+json.dumps(state), flush=True)

    def accept(self, ticket):
        if not isinstance(ticket, dict): raise ValueError('ticket must be an object')
        if self.sequence >= self.MAX_TICKETS: raise ValueError('80 ticket budget exhausted')
        if ticket.get('sequence') != self.sequence+1 or ticket.get('state_digest') != self.state['digest']:
            raise ValueError('ticket does not match current decision state')
        if not all(isinstance(ticket.get(k), str) and ticket[k].strip()
                   for k in ('review_id','authorization_id','reason')):
            raise ValueError('review, authorization and rationale required')
        button = ticket.get('button')
        if button not in BUTTONS: raise ValueError('one supported button required')
        profile = ticket.get('profile','normal')
        if profile not in ('normal','descent'): raise ValueError('unknown ticket profile')
        if profile=='descent' and self.guard(button,profile=profile):
            raise ValueError('descent ticket is not currently executable: '+self.guard(button,profile=profile))
        seconds = ticket.get('seconds')
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds):
            raise ValueError('finite duration required')
        if seconds < self.dt or seconds > self.duration_limit(button,profile): raise ValueError('button duration out of bounds')
        if self.current['time']+seconds > self.MAX_SECONDS+1e-9: raise ValueError('ticket exceeds simulation budget')
        self.sequence += 1
        self.active = dict(ticket, profile=profile, start=self.current.copy())
        self.end = self.current['time']+math.floor(seconds/self.dt+1e-8)*self.dt
        self.log('approved_ticket', **self.active)

    def finish(self, now, xyz, reason):
        if self.active:
            self.log('ticket_outcome', sequence=self.sequence, time=now, position=list(xyz),
                     elapsed=now-self.active['start']['time'],
                     displacement=math.dist(xyz[:2],self.active['start']['position'][:2]),
                     height_change=xyz[2]-self.active['start']['position'][2],
                     button=self.active['button'], reason=reason,
                     physical_failure=bool(self.terminal and self.terminal.startswith(('body tilt','actual horizontal'))))
        self.active = None
        self.pause_reason = reason

    def next_motion(self, now, xyz, yaw, velocity, up, running, requests):
        self.observe(now, xyz, yaw, velocity, up)
        if self.terminal:
            self.finish(now, xyz, self.terminal)
            if hasattr(self, 'current'): self.publish()
            return None
        if self.active and now >= self.end-1e-9: self.finish(now, xyz, 'ticket duration completed')
        while True:
            # STOP/manual takeover cancels the whole diagnostic, even while
            # frozen. Autonomy cannot silently resume after external control.
            if not requests.empty():
                requests.get()
                self.terminal = 'operator STOP or takeover'
            if not running(): self.terminal = 'application closed'
            if self.terminal:
                self.finish(now, xyz, self.terminal)
                self.publish()
                return None
            if self.active:
                reason = self.guard(self.active['button'],profile=self.active['profile'])
                if not reason:
                    button = self.active['button']
                    if not self.initializing and self.sequence not in self.ground_admission_tickets:
                        contexts=self.ground_fallback_contexts(button,self.active['profile'])
                        if contexts:
                            self.ground_admission_tickets.add(self.sequence)
                            self.log('oriented_ground_admission',sequence=self.sequence,
                                     button=button,profile=self.active['profile'],contexts=contexts,**self.current)
                    if self.static_prior and not self.initializing:
                        live_only = self.guard(button, static_support=False,profile=self.active['profile'])
                        key = (self.sequence, live_only)
                        if live_only and key not in self.admission_keys:
                            self.admission_keys.add(key)
                            self.prior_admissions += 1
                            self.log('static_survey_admission', sequence=self.sequence,
                                     time=now, button=button, live_only_reason=live_only,
                                     survey_sha256=self.grid.terrain_prior['file_sha256'])
                    return Motion(*directional_command(forward=button in ('forward','forward_left','forward_right'), back=button=='back',
                                  left=button in ('left','forward_left'), right=button in ('right','forward_right'), vx=.4, omega=.35))
                self.finish(now, xyz, 'geometry pause: '+reason)
            if self.sequence >= self.MAX_TICKETS:
                self.terminal = '80 ticket budget exhausted'
                self.publish()
                return None
            self.publish()
            inbox = self.directory/f'ticket-{self.sequence+1:03d}.json'
            while not inbox.exists() and requests.empty() and running():
                time.sleep(.1)  # Deliberately no env.step: wall time costs no physics.
            if not requests.empty() or not running(): continue
            try:
                self.accept(json.loads(inbox.read_text()))
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self.log('invalid_ticket', sequence=self.sequence+1, reason=str(exc))
                self.terminal = 'invalid ticket: '+str(exc)

    def report(self):
        return dict(classification='agent-assisted diagnostic', tickets=self.sequence,
                    milestones=self.milestones, terminal=self.terminal,
                    final_state=getattr(self,'current',None), peak_speed_mps=self.peak_speed,
                    peak_tilt_deg=self.peak_tilt,
                    initialization_complete=not self.initializing,
                    static_survey_admissions=self.prior_admissions,
                    oriented_ground_admissions=len(self.ground_admission_tickets),
                    counts_as_qualification=False, autonomous=False)
