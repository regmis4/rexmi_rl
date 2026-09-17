"""Pure checkpoint planning/control. No physics writes or policy selection."""
from dataclasses import dataclass
import heapq
import math
import numpy as np


def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def path_progress(path, xy, first=0, lookahead=1.2):
    """Project onto a polyline, then advance by arc length, never by goal bearing."""
    best=None
    for i in range(first,min(len(path)-1,first+16)):
        a,b=np.asarray(path[i]),np.asarray(path[i+1])
        delta=b-a
        t=float(np.clip(np.dot(np.asarray(xy)-a,delta)/max(1e-12,np.dot(delta,delta)),0.,1.))
        foot=a+t*delta
        distance=float(np.linalg.norm(foot-xy))
        candidate=(distance,i,foot)
        if best is None or distance<best[0]: best=candidate
    if best is None:
        return first,np.asarray(path[-1]),tuple(path[-1]),math.dist(xy,path[-1]),math.dist(xy,path[-1])
    cross,index,foot=best
    remaining=math.dist(foot,path[index+1])+sum(math.dist(a,b) for a,b in zip(path[index+1:],path[index+2:]))
    cursor=foot
    left=lookahead
    target=path[-1]
    for p in path[index+1:]:
        distance=math.dist(cursor,p)
        if distance>=left:
            target=tuple(cursor+left/max(distance,1e-9)*(np.asarray(p)-cursor));break
        left-=distance;cursor=np.asarray(p)
    return index,foot,target,remaining,cross


def observed_route(grid, start, goal, yaw=0.):
    """A* through observed surface, with heading and directional-slope costs.

    Returns a route to the goal or a reachable observation frontier. Unknown
    cells are excluded. Eight heading states discourage gratuitous zigzags.
    """
    first, target = grid.world_to_cell(*start), grid.world_to_cell(*goal)
    if not grid.traversable(first) or target is None:
        return [], False
    offsets = [(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]
    direction = int(round((yaw % (2*math.pi))/(math.pi/4))) % 8
    source = (*first, direction)
    def allowed(cell):
        # Global memory preserves route continuity, but cannot supply the first
        # motion corridor. Otherwise replan repeatedly chooses the stale route
        # that the follower has just rejected.
        local=math.hypot(cell[0]-first[0],cell[1]-first[1])*grid.cell_size<=2.
        return grid.traversable(cell,require_fresh=local)
    def heuristic(r, c):
        return math.hypot(r-target[0], c-target[1])
    queue = [(heuristic(*first), 0., source)]
    costs, parent = {source: 0.}, {}
    best, best_h, found = source, heuristic(*first), False
    while queue:
        _, travelled, node = heapq.heappop(queue)
        if travelled != costs.get(node):
            continue
        r,c,heading = node
        h = heuristic(r,c)
        if h < best_h:
            best, best_h = node, h
        if (r,c) == target:
            best, found = node, True
            break
        for k,(dr,dc) in enumerate(offsets):
            nr,nc = r+dr,c+dc
            if not (0<=nr<grid.n_cells and 0<=nc<grid.n_cells):
                continue
            if not allowed((nr,nc)):
                continue
            if dr and dc and (not allowed((r+dr,c)) or not allowed((r,c+dc))):
                continue
            norm = math.hypot(dr,dc)
            if not grid.direction_allowed((nr,nc),k): continue
            g = travelled + grid.edge_cost((nr,nc),dr,dc,heading)
            nxt = (nr,nc,k)
            if g < costs.get(nxt, math.inf):
                costs[nxt], parent[nxt] = g, node
                heapq.heappush(queue, (g+heuristic(nr,nc),g,nxt))
    # If the goal approach is unavailable, survey a reachable observed boundary.
    # Targets remain known cells; unknown neighbours only measure survey value.
    if not found and heuristic(*first)-best_h < .6/grid.cell_size:
        candidates=[]
        for node,travelled in costs.items():
            r,c,_=node
            if math.hypot(r-first[0],c-first[1])*grid.cell_size<.8: continue
            patch=grid.known[max(0,r-3):r+4,max(0,c-3):c+4]
            unseen=int((~patch).sum())
            if unseen>=3: candidates.append((unseen/(1.+.05*travelled)-.05*heuristic(r,c),node))
        if not candidates: return [],False
        best=max(candidates,key=lambda item:item[0])[1]
    cells = [best]
    while cells[-1] != source:
        cells.append(parent[cells[-1]])
    points = [grid.cell_to_world(r,c) for r,c,_ in reversed(cells)]
    points[0] = tuple(start)
    if found and grid.segment_clear(points[-1], goal):
        points.append(tuple(goal))
    return points, found


@dataclass
class Motion:
    vx: float = 0.
    omega: float = 0.


class CheckpointController:
    def __init__(self, grid, waypoints, cruise_speed=.4):
        if not .1 <= cruise_speed <= .8:
            raise ValueError('cruise_speed must be between 0.1 and 0.8 m/s')
        self.grid, self.waypoints = grid, waypoints
        self.cruise_speed = cruise_speed
        self.wp_idx = 0
        self.state, self.reason = 'OBSERVE', 'collecting fresh scans'
        self.since = 0.
        self.path, self.checkpoint, self.target = [], None, None
        self.path_index = 0
        self.heading_error = self.tracking_error = 0.
        self.best_remaining, self.progress_at = math.inf, 0.
        self.retry_count = 0
        self.manual = Motion()
        self.intervened = False
        self.is_mission_checkpoint = False
        self.last_revision = -1
        self.observe_revision = 0
        self.pending_arrival = False
        self.history = []
        self.reverse_target = None
        self.stop_started = None
        self.stop_speed = self.stop_distance = 0.
        self.braking_accel = None  # cruise requires measured evidence at normal speed
        self.escape_clearance = None
        self.mission_route = []
        self.mission_route_complete = False
        self.route_mission = -1
        self.hold_anchor = None
        self.hold_heading = 0.
        self.velocity_world = np.zeros(2)
        self.hold_failures = 0
        self.rolling_checkpoints = 0
        self.last_hold_failure = ''
        self.replan_anchor = None
        self.replan_since = 0.
        self.planning_failure = 'no observed clear route; inspect or reposition manually'
        self.stable_since = None
        self.recovery_started = None
        self.recovery_anchor = None
        self.recovery_attempts = []
        self.reassess_at = -math.inf
        self.recovery_motion = Motion()
        self.recovery_origin = None
        self.recovery_yaw = 0.
        self.drive_samples=[]
        self.previous_command=Motion()
        self.climb_failures=0
        self.backslide_distance=0.
        self.last_tick=None
        self.last_climb_event=''
        self.route_decision='initial route'
        self.motion_rejection=''
        self.last_success_position=None

    def set_state(self, state, reason, now):
        if state=='SETTLE' and self.state!='SETTLE':
            self.stable_since=None
            if not self.pending_arrival:
                self.stop_started=None
                self.stop_speed=0.
        if state in ('SETTLE','HOLD','REASSESS','MANUAL','FOLLOW') and state != self.state:
            self.hold_anchor = None
        self.state, self.reason, self.since = state, reason, now

    def position_hold(self, xy, yaw):
        """Feedback through the same forward/back/yaw commands, no joint brakes.

        A zero *desired* speed can require a reverse command to resist a slope.
        This cannot create traction: the caller measures whether holding works.
        """
        if self.hold_anchor is None:
            self.hold_anchor=np.asarray(xy).copy()
            self.hold_heading=yaw
        forward=np.array([math.cos(yaw),math.sin(yaw)])
        error=self.hold_anchor-np.asarray(xy)
        vx=float(np.clip(.8*np.dot(error,forward)-np.dot(self.velocity_world,forward),-.4,.4))
        omega=float(np.clip(.8*wrap_angle(self.hold_heading-yaw),-.2,.2))
        return Motion(vx,omega)

    def request(self, action, now, vx=0., omega=0.):
        if action == 'resume':
            self.drive_samples=[]
            self.previous_command=Motion()
            self.manual = Motion()
            self.recovery_started = None
            self.recovery_attempts = []
            self.path, self.checkpoint = [], None
            self.mission_route = []
            self.retry_count = 0
            self.replan_anchor = None
            self.observe_revision = self.grid.revision
            self.set_state('OBSERVE', 'resume: refresh and replan', now)
        elif action in ('stop', 'manual'):
            self.intervened = True
            self.hold_anchor = None
            self.manual = Motion(float(np.clip(vx,-.4,.4)),float(np.clip(omega,-.35,.35)))
            self.set_state('MANUAL' if action=='manual' else 'HOLD', 'operator control', now)

    def hold(self, reason, now, terminal=False):
        self.path = []
        self.target = None
        self.set_state('HOLD' if terminal else 'REASSESS', reason, now)
        self.reassess_at = now
        return Motion()

    def retreat(self, xy, yaw, now):
        behind=(xy[0]-.5*math.cos(yaw),xy[1]-.5*math.sin(yaw))
        visited=any(math.dist(behind,p)<.25 for _,p in self.history)
        if self.retry_count==0 and visited and self.grid.segment_clear(xy,behind):
            self.retry_count=1
            self.reverse_target=behind
            self.set_state('REVERSE','stalled; retreat along observed travelled corridor',now)
            return Motion()
        return self.hold('stalled; no safe automatic recovery',now)

    def _plan(self, xy, yaw, now):
        self.planning_failure='no observed clear route; inspect or reposition manually'
        if self.replan_anchor is None or math.dist(xy,self.replan_anchor)>.5:
            self.replan_anchor,self.replan_since=tuple(xy),now
        elif now-self.replan_since>20.:
            self.planning_failure='repeated replans without progress; manual inspection required'
            return False
        wp = self.waypoints[self.wp_idx]
        self.escape_clearance = None
        if not self.grid.traversable(self.grid.world_to_cell(*xy)):
            route,clearance=self.grid.clearance_exit(xy,goal=(wp.x,wp.y))
            if route:
                self.escape_clearance=clearance
                self.path,self.checkpoint,self.path_index=route,route[-1],0
                self.is_mission_checkpoint=False
                self.best_remaining,self.progress_at=math.inf,now
                self.set_state('FOLLOW','leave newly observed clearance margin',now)
                return True
            cell=self.grid.world_to_cell(*xy)
            if cell is None or not self.grid.known[cell]:
                self.planning_failure='current position unobserved'
            elif not self.grid.fresh[cell]:
                self.planning_failure='current position needs a fresh scan'
            elif self.grid.hazard[cell]:
                self.planning_failure='hazard mapped at current position'
            elif self.grid.slope[cell]>math.tan(math.radians(50)):
                self.planning_failure='current terrain exceeds slope limit'
            else:
                self.planning_failure='robot footprint or clearance overlaps a mapped hazard'
            return False
        candidate=[]
        if self.route_mission == self.wp_idx and len(self.mission_route)>2:
            nearest=min(range(len(self.mission_route)),key=lambda i:math.dist(xy,self.mission_route[i]))
            candidate=[tuple(xy)]+self.mission_route[nearest+1:]
            if (len(candidate)<2 or not all(self.grid.segment_clear(a,b)
                    for a,b in zip(candidate[:12],candidate[1:12]))): candidate=[]
        route,complete=observed_route(self.grid,xy,(wp.x,wp.y),yaw)
        fresh_cost=self.grid.route_cost(route,yaw) if len(route)>1 else math.inf
        old_cost=self.grid.route_cost(candidate,yaw) if len(candidate)>1 else math.inf
        # Only compare equivalent destinations. A short frontier is not a cheap
        # substitute for an available complete mission route.
        comparable=(len(route)>1 and len(candidate)>1 and math.dist(route[-1],candidate[-1])<.4)
        if math.isfinite(old_cost) and (not math.isfinite(fresh_cost)
                or (self.mission_route_complete and not complete)
                or (comparable and fresh_cost>=.85*old_cost)):
            route,complete=candidate,self.mission_route_complete
            self.route_decision='retained route: alternative is not 15% cheaper'
        else:
            self.route_decision='selected best observed route' if complete else 'surveying reachable observed frontier'
        self.mission_route,self.mission_route_complete,self.route_mission=route,complete,self.wp_idx
        if len(route)<2:
            return False
        # Shorten the local checkpoint on inclined terrain or a bend.
        cell = self.grid.world_to_cell(*xy)
        length = 2. if self.grid.slope[cell]>.25 else 4.
        distance, end = 0., 1
        for i in range(1,len(route)):
            distance += math.dist(route[i-1],route[i])
            end = i
            if distance>=length:
                break
            if i>2:
                v1=np.subtract(route[i-1],route[i-2]);v2=np.subtract(route[i],route[i-1])
                if distance>=2. and abs(wrap_angle(math.atan2(v2[1],v2[0])-math.atan2(v1[1],v1[0])))>.6:
                    break
        self.path=route[:end+1]
        self.checkpoint=self.path[-1]
        self.is_mission_checkpoint=complete and end==len(route)-1
        self.path_index=0
        self.best_remaining=math.inf
        self.progress_at=now
        self.set_state('FOLLOW','following observed route',now)
        return True

    def _remaining(self, xy):
        return math.dist(xy,self.path[self.path_index]) + sum(
            math.dist(a,b) for a,b in zip(self.path[self.path_index:],self.path[self.path_index+1:]))

    def motion_clear(self, xy, yaw, speed, motion):
        """Check both current-heading stopping room and the commanded arc.

        The learned policy need not turn as quickly as requested, so a curved
        prediction alone is insufficient to authorize forward motion.
        """
        self.motion_rejection=''
        if abs(motion.vx)<1e-6:
            return self.grid.segment_clear(xy,xy,self.escape_clearance)
        # Reverse uses the same observed footprint and stopping checks.
        direction=1. if motion.vx>0 else -1.
        accel=max(.1,self.braking_accel or .2)
        stop=max(.1,max(speed,abs(motion.vx))**2/(2*accel)+.1)
        straight=(xy[0]+direction*stop*math.cos(yaw),xy[1]+direction*stop*math.sin(yaw))
        # On a slope, actual drift can oppose the requested motion. Check its
        # stopping corridor too instead of assuming command equals velocity.
        drift=float(np.linalg.norm(self.velocity_world))
        if drift>.10:
            drift_distance=drift**2/(2*accel)+.1
            drift_end=tuple(np.asarray(xy)+drift_distance*self.velocity_world/drift)
            if not self.grid.segment_clear(xy,drift_end,self.escape_clearance,check_direction=False):
                self.motion_rejection='drift stopping corridor: '+self.grid.segment_reason(xy,drift_end,self.escape_clearance,check_direction=False)
                return False
        if not self.grid.segment_clear(xy,straight,self.escape_clearance):
            self.motion_rejection=self.grid.segment_reason(xy,straight,self.escape_clearance)
            return False
        horizon=max(.4,motion.vx**2/(2*accel)+.3)
        previous=xy
        for t in np.linspace(0,horizon/abs(motion.vx),21)[1:]:
            if abs(motion.omega)<1e-6:
                dx,dy=motion.vx*t,0.
            else:
                dx=motion.vx*math.sin(motion.omega*t)/motion.omega
                dy=motion.vx*(1-math.cos(motion.omega*t))/motion.omega
            point=(xy[0]+dx*math.cos(yaw)-dy*math.sin(yaw),
                   xy[1]+dx*math.sin(yaw)+dy*math.cos(yaw))
            if not self.grid.segment_clear(previous,point,self.escape_clearance):
                self.motion_rejection=self.grid.segment_reason(previous,point,self.escape_clearance)
                return False
            previous=point
        return True

    def local_maneuver(self, xy, yaw, speed):
        """Rank bounded teleop arcs, without clearing hazards or inventing strafing."""
        if speed>.12:
            return None  # Brake before reversing or changing the recovery heading.
        cell=self.grid.world_to_cell(*xy)
        if cell is None or not self.grid.fresh[cell]:
            return None
        self.escape_clearance=None
        if self.grid.blocked[cell]:
            distance=float(self.grid.obstacle_distance[cell])
            if distance<self.grid.footprint_radius or self.grid.hazard[cell]:
                return None
            self.escape_clearance=distance
        wp=self.waypoints[self.wp_idx]
        goal=(wp.x,wp.y)
        bearing=math.atan2(goal[1]-xy[1],goal[0]-xy[0])
        candidates=[]
        for vx in (.2,-.2,0.):
            for omega in (0.,.35,-.35):
                if vx==omega==0: continue
                motion=Motion(vx,omega)
                # Do not repeat an ineffective action at the same place/heading.
                if any(motion==old and math.dist(xy,pos)<.35
                       and abs(wrap_angle(yaw-heading))<.25
                       for old,pos,heading in self.recovery_attempts):
                    continue
                if not self.motion_clear(xy,yaw,speed,motion): continue
                duration=2.
                if omega:
                    dx=vx*math.sin(omega*duration)/omega
                    dy=vx*(1-math.cos(omega*duration))/omega
                else:
                    dx,dy=vx*duration,0.
                end=(xy[0]+dx*math.cos(yaw)-dy*math.sin(yaw),
                     xy[1]+dx*math.sin(yaw)+dy*math.cos(yaw))
                endcell=self.grid.world_to_cell(*end)
                if endcell is None: continue
                progress=math.dist(xy,goal)-math.dist(end,goal)
                alignment=(abs(wrap_angle(bearing-yaw))-
                           abs(wrap_angle(bearing-yaw-omega*duration)))
                clearance=min(2.,float(self.grid.obstacle_distance[endcell]))
                across=abs(-self.grid.gx[endcell]*math.sin(yaw)+self.grid.gy[endcell]*math.cos(yaw))
                score=3.*progress+.5*alignment+.3*clearance-.03*self.grid.cost[endcell]-.3*across
                if vx==0: score-=.25
                candidates.append((score,motion,end))
        return max(candidates,key=lambda item:item[0])[1:] if candidates else None

    def reassess(self, xy, yaw, speed, now):
        if self.recovery_anchor is None or math.dist(xy,self.recovery_anchor)>1.2:
            self.recovery_anchor=xy
            self.recovery_started=now
            self.recovery_attempts=[]
        if self.recovery_started is None: self.recovery_started=now
        if now-self.recovery_started>30. or len(self.recovery_attempts)>=6:
            return self.hold('local recovery exhausted; manual assistance required',now,terminal=True)
        # A slipped robot must not chase an obsolete stop anchor while deciding.
        # Resist current motion, and evaluate from its actual new position.
        self.hold_anchor=np.asarray(xy).copy()
        braking=self.position_hold(xy,yaw)
        if now-self.reassess_at<1.: return braking
        self.reassess_at=now
        self.replan_anchor=None
        self.mission_route=[]
        if self._plan(xy,yaw,now):
            self.reason='route available from current position; resume following'
            return braking
        option=self.local_maneuver(xy,yaw,speed)
        if option is None:
            self.reason='reassessing: '+self.planning_failure+'; no clear maneuver yet'
            return braking
        motion,end=option
        self.recovery_motion=motion
        self.recovery_origin=xy
        self.recovery_yaw=yaw
        self.recovery_attempts.append((motion,xy,yaw))
        self.target=end
        self.path=[xy,end] if motion.vx else []
        direction='forward' if motion.vx>0 else 'reverse' if motion.vx<0 else 'turn'
        turn=' left' if motion.omega>0 else ' right' if motion.omega<0 else ''
        self.set_state('MANEUVER',f'recovery: {direction}{turn} through observed clearance',now)
        return motion

    def tick(self, xy, yaw, speed, now, upright=1., velocity=None):
        actual=np.asarray(velocity[:2] if velocity is not None else
                          (speed*math.cos(yaw),speed*math.sin(yaw)))
        forward=float(np.dot(actual,(math.cos(yaw),math.sin(yaw))))
        if self.last_tick is not None and self.previous_command.vx>0 and self.state=='FOLLOW':
            self.backslide_distance+=max(0.,-forward)*max(0.,now-self.last_tick)
        self.last_tick=now
        cell=self.grid.world_to_cell(*xy)
        uphill=(cell is not None and self.grid.gx[cell]*math.cos(yaw)+self.grid.gy[cell]*math.sin(yaw)>.15)
        driving=(self.state=='FOLLOW' and self.previous_command.vx>.25
                 and abs(self.heading_error)<.35 and upright>=.3)
        if driving:
            self.drive_samples.append((now,tuple(xy),forward,yaw))
            while len(self.drive_samples)>1 and self.drive_samples[1][0]<=now-2.:
                self.drive_samples.pop(0)
            if now-self.drive_samples[0][0]>=2.-1e-6:
                mean=float(np.mean([sample[2] for sample in self.drive_samples]))
                if uphill and mean<.05:
                    self.grid.record_traversal([sample[1] for sample in self.drive_samples],yaw,failed=True)
                    self.climb_failures+=1
                    kind='backsliding' if mean<-.10 else 'poor uphill progress'
                    self.last_climb_event=f'{kind}: {mean:.2f} m/s over 2s; approach recorded'
                    self.drive_samples=[]
                    self.mission_route=[]
                    self.replan_anchor=None
                    self.pending_arrival=False
                    self.set_state('SETTLE',self.last_climb_event,now)
                elif mean>=.10 and (self.last_success_position is None or math.dist(xy,self.last_success_position)>.4):
                    self.grid.record_traversal([sample[1] for sample in self.drive_samples],yaw)
                    self.last_success_position=tuple(xy)
        else:
            self.drive_samples=[]
        was_following=self.state=='FOLLOW'
        command=self._tick(xy,yaw,speed,now,upright,velocity)
        if (was_following and self.state=='SETTLE' and self.reason.startswith('cross-track deviation')
                and cell is not None and self.grid.slope[cell]>.25 and self.previous_command.vx>.25):
            # Side slip may retain forward wheel motion. Crossing the existing
            # tracking boundary is an observed failed approach, not a friction estimate.
            points=[sample[1] for sample in self.drive_samples] or [tuple(xy)]
            self.grid.record_traversal(points,yaw,failed=True)
            self.climb_failures+=1
            self.last_climb_event='slope tracking failure: lateral route error exceeded 0.75m; approach recorded'
            self.reason=self.last_climb_event
            self.drive_samples=[]
            self.mission_route=[]
            self.replan_anchor=None
        self.previous_command=command
        return command

    def _tick(self, xy, yaw, speed, now, upright=1., velocity=None):
        xy = tuple(xy)
        self.velocity_world=np.asarray(velocity[:2] if velocity is not None else
                                       (speed*math.cos(yaw),speed*math.sin(yaw)))
        if upright < .30:
            return self.hold('robot tipped; manual assistance required',now,terminal=True)
        if self.state=='MANUAL':
            return self.position_hold(xy,yaw) if self.manual==Motion() else self.manual
        if self.state=='HOLD':
            return self.position_hold(xy,yaw)
        if self.state=='COMPLETE':
            return Motion()
        if self.wp_idx>=len(self.waypoints):
            self.set_state('COMPLETE','all mission destinations confirmed',now)
            return Motion()
        if self.state=='REASSESS':
            return self.reassess(xy,yaw,speed,now)
        if self.state=='MANEUVER':
            elapsed=now-self.since
            distance=math.dist(xy,self.recovery_origin)
            turned=abs(wrap_angle(yaw-self.recovery_yaw))
            if (elapsed>=2. or distance>=.4 or turned>=.65
                    or not self.motion_clear(xy,yaw,speed,self.recovery_motion)):
                self.pending_arrival=False
                self.replan_anchor=None
                self.mission_route=[]
                self.set_state('SETTLE','local maneuver complete; refresh route',now)
                return self.position_hold(xy,yaw)
            return self.recovery_motion
        if self.state=='OBSERVE':
            # Full 360-degree scan is already available: no compulsory spin.
            if now-self.since<1. or self.grid.revision-self.observe_revision<3 or speed>.10:
                if (now-self.since>3. and speed>.10
                        and self.grid.revision-self.observe_revision>=3):
                    self.hold_failures+=1
                    self.last_hold_failure=f'observation hold unavailable: speed={speed:.2f}m/s'
                    if self._plan(xy,yaw,now):
                        self.reason=self.last_hold_failure+'; moving scan on clear route'
                        return self.position_hold(xy,yaw)
                    return self.hold(self.last_hold_failure+'; '+self.planning_failure,now)
                if now-self.since>10.:
                    return self.hold('cannot settle or obtain fresh scans',now)
                return self.position_hold(xy,yaw)
            if not self._plan(xy,yaw,now):
                if self.planning_failure.startswith('repeated replans'):
                    return self.retreat(xy,yaw,now)
                return self.hold(self.planning_failure,now)
        if self.state=='SETTLE':
            braking=self.position_hold(xy,yaw)
            if speed<.08:
                if self.stable_since is None: self.stable_since=now
            else:
                self.stable_since=None
            if self.stop_started is not None:
                self.stop_distance += speed * max(0., now-self.stop_started)
                self.stop_started=now
            if self.stable_since is not None and now-self.stable_since>=.5:
                if self.stop_speed>.25 and self.stop_distance>.02:
                    estimate=self.stop_speed**2/(2*self.stop_distance)
                    self.braking_accel=min(estimate,self.braking_accel or estimate)
                if self.pending_arrival:
                    if math.dist(xy,self.checkpoint)>.6:
                        self.hold_failures+=1
                        self.rolling_checkpoints+=1
                        self.last_hold_failure='checkpoint crossed, but requested hold position was not maintained'
                    if self.is_mission_checkpoint:
                        self.wp_idx+=1
                    self.retry_count=0
                self.path=[]
                self.observe_revision=self.grid.revision
                self.set_state('OBSERVE','checkpoint scan' if self.pending_arrival else 'replan after stop',now)
                self.pending_arrival=False
            elif (now-self.since>3. or (now-self.since>1. and math.dist(xy,self.hold_anchor)>.5)):
                self.hold_failures+=1
                self.last_hold_failure=(f'hold unavailable: speed={speed:.2f}m/s, '
                                        f'drift={math.dist(xy,self.hold_anchor):.2f}m')
                if self.pending_arrival:
                    # Arrival was observed inside the checkpoint boundary before
                    # braking. The operator-approved fallback is a moving scan.
                    self.rolling_checkpoints+=1
                    if self.is_mission_checkpoint: self.wp_idx+=1
                    self.pending_arrival=False
                if self.wp_idx>=len(self.waypoints):
                    self.set_state('COMPLETE','mission reached; final hold unavailable',now)
                    return braking
                if self._plan(xy,yaw,now):
                    self.reason=self.last_hold_failure+'; moving scan on clear route'
                    return braking
                return self.hold(self.last_hold_failure+'; no clear route for moving scan',now)
            return braking
        if self.state=='REVERSE':
            if not self.grid.segment_clear(xy,self.reverse_target):
                return self.hold('reverse corridor no longer clear',now)
            if math.dist(xy,self.reverse_target)<.15 or now-self.since>3.:
                self.pending_arrival=False
                self.replan_anchor=None
                self.mission_route=[]
                self.set_state('SETTLE','reverse complete; stop and reassess',now)
                return Motion()
            return Motion(-.2,0.)
        cell=self.grid.world_to_cell(*xy)
        if cell is not None and self.grid.blocked[cell] and self.escape_clearance is None:
            wp=self.waypoints[self.wp_idx]
            exit_path,clearance=self.grid.clearance_exit(xy,goal=(wp.x,wp.y))
            if exit_path:
                self.path,self.checkpoint,self.path_index=exit_path,exit_path[-1],0
                self.escape_clearance=clearance
                self.is_mission_checkpoint=False
                self.best_remaining,self.progress_at=math.inf,now
                self.reason='exit planning margin while preserving body clearance'
        # Follow continuous path progress, not distance to individual grid points.
        self.path_index,foot,lookahead,remaining,self.tracking_error=path_progress(
            self.path,xy,self.path_index)
        # A path recheck uses the SAME blocked/observed mask as the planner.
        if self.grid.revision != self.last_revision:
            self.last_revision=self.grid.revision
            corridor=[xy]+self.path[self.path_index+1:]
            if len(corridor)==1:
                corridor.append(self.path[-1])
            # Recheck the immediate stopping/lookahead corridor. A distant cell
            # expiring must not interrupt an otherwise clear slow alignment turn;
            # it will be checked before the robot can advance into it.
            horizon=max(.6,speed*speed/(2*max(.1,self.braking_accel or .2))+.3)
            checked=0.
            clear=True
            for a,b in zip(corridor,corridor[1:]):
                if not self.grid.segment_clear(a,b,self.escape_clearance):
                    clear=False;break
                checked+=math.dist(a,b)
                if checked>=horizon: break
            if not clear:
                self.pending_arrival=False
                self.set_state('SETTLE','route rejected: '+self.grid.segment_reason(a,b,self.escape_clearance),now)
                return Motion()
        if self.tracking_error>.75:
            self.pending_arrival=False
            self.set_state('SETTLE','cross-track deviation; stop and replan',now)
            return Motion()
        if remaining<self.best_remaining-.15:
            self.best_remaining,self.progress_at=remaining,now
        arrival = .12 if self.escape_clearance is not None else .40
        if math.dist(xy,self.checkpoint)<=arrival and remaining<.8:
            # A frontier endpoint may already lie within the mission arrival
            # boundary even if the exact goal cell is blocked/unobserved. Count
            # actual measured arrival, not A*'s exact-cell completion flag.
            wp=self.waypoints[self.wp_idx]
            if math.dist(xy,(wp.x,wp.y))<=.6:
                self.is_mission_checkpoint=True
            self.pending_arrival=True
            self.stop_speed,self.stop_distance,self.stop_started=speed,0.,now
            self.set_state('SETTLE','verify checkpoint arrival and stop',now)
            return Motion()
        # Short lookahead with line-of-sight validation; no direct goal shortcut.
        target=lookahead
        if not self.grid.segment_clear(xy,target,self.escape_clearance):
            target=self.path[min(self.path_index+1,len(self.path)-1)]
            if not self.grid.segment_clear(xy,target,self.escape_clearance):
                self.pending_arrival=False
                self.set_state('SETTLE','lookahead rejected: '+self.grid.segment_reason(xy,target,self.escape_clearance),now)
                return Motion()
        self.target=target
        self.heading_error=wrap_angle(math.atan2(target[1]-xy[1],target[0]-xy[0])-yaw)
        if now-self.progress_at>30. and abs(self.heading_error)<.5:
            return self.retreat(xy,yaw,now)
        if now-self.progress_at>90.:
            return self.hold('alignment timeout; inspect steering response',now)
        # Retain where we drove. The map's fresh-clear check, not the age of
        # that visit, determines whether backing into the corridor is safe now.
        # An 8 s history expired before the 30 s stall timer could use it.
        self.history=self.history[-2000:]
        if not self.history or math.dist(xy,self.history[-1][1])>.1:
            self.history.append((now,xy))
        error=abs(self.heading_error)
        omega=float(np.clip(.8*self.heading_error,-.35,.35)) if error>.035 else 0.
        vx=.4*max(.25,math.cos(error)) if error<math.radians(80) else 0.
        quality=self.grid.quality(xy)
        cell=self.grid.world_to_cell(*xy)
        # Higher cruise is opt-in and gated by measured normal-speed braking.
        braking=self.cruise_speed**2/(2*max(.05,self.braking_accel or .05))+.5
        if (self.cruise_speed>.4 and self.braking_accel is not None and error<.08
                and self.tracking_error<.2 and self.grid.slope[cell]<.12
                and quality['coverage']>.90 and quality['age']<1.
                and remaining>max(3.,braking)):
            corridor=[xy]+self.path[self.path_index:]
            headings=[math.atan2(b[1]-a[1],b[0]-a[0]) for a,b in zip(corridor,corridor[1:]) if math.dist(a,b)>.01]
            if headings and max(abs(wrap_angle(h-yaw)) for h in headings)<.15:
                vx=self.cruise_speed
        uphill=self.grid.gx[cell]*math.cos(yaw)+self.grid.gy[cell]*math.sin(yaw)>.15
        # Creeping at 0.25 m/s stalled the learned policy on the crater wall.
        # Keep its normal teleop drive request uphill until the arrival/braking
        # boundary, rather than assuming a lower command preserves traction.
        approach_limit=.4 if uphill else max(.25,(remaining-.3)*.4)
        vx=min(vx,self.cruise_speed,approach_limit)
        if self.escape_clearance is not None:
            vx=min(vx,.20)
        motion=Motion(vx,omega)
        if vx>0 and not self.motion_clear(xy,yaw,speed,motion):
            # The old fixed 40 cm straight probe silently zeroed forward drive
            # even when the planned turn was clear, leaving a weak yaw request.
            # Try a slower, tighter turn only with verified stopping room.
            turning=Motion(min(.2,vx),math.copysign(.35,self.heading_error))
            if error>.035 and self.motion_clear(xy,yaw,speed,turning):
                self.reason='following clear slow turning arc'
                return turning
            corner=self.path[min(self.path_index+1,len(self.path)-1)]
            corner_error=wrap_angle(math.atan2(corner[1]-xy[1],corner[0]-xy[0])-yaw)
            if speed<.1 and abs(corner_error)>.08:
                # At a tight corner, align with the first planned segment.
                # Holding the old heading while replanning cannot make that
                # segment drivable; a distant lookahead may also under-turn it.
                self.target=corner
                self.heading_error=corner_error
                self.reason='turning toward route corner; forward corridor not yet clear'
                return Motion(0.,math.copysign(.35,corner_error))
            self.pending_arrival=False
            self.set_state('SETTLE','command rejected: '+self.motion_rejection,now)
            return self.position_hold(xy,yaw)
        self.reason='aligning to route' if vx==0 else 'following observed route'
        return motion
