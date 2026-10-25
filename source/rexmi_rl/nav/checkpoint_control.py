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


def observed_route(grid, start, goal, yaw=0., require_fresh_start=True, profile='balanced', support=None, avoid=None,
                   uphill_limit=None, no_go_exit=False, strict_frontier=False, allow_frontier=True):
    """A* through observed surface, with heading and directional-slope costs.

    Returns a route to the goal or a reachable observation frontier. Unknown
    cells are excluded. Thirty-two heading states support diagonal approaches.
    """
    first, target = grid.world_to_cell(*start), grid.world_to_cell(*goal)
    if not (grid.exit_traversable(first) if no_go_exit else grid.traversable(first)) or target is None:
        return [], False
    support=grid.planning_support() if support is None else support
    if not support[first]: return [],False
    # Extra diagonals permit contouring a hillside instead of restricting every
    # climb to cardinal/45-degree steps. Validate every crossed cell, including
    # corner neighbours; the longer edge cannot jump over a rock or unknown gap.
    offsets=sorted(((dr,dc) for dr in range(-3,4) for dc in range(-3,4)
                    if (dr or dc) and math.gcd(abs(dr),abs(dc))==1),
                   key=lambda p:math.atan2(p[1],p[0])%(2*math.pi))
    angles=np.array([math.atan2(dc,dr) for dr,dc in offsets])
    edge_costs,turn_cost=grid.directional_costs(profile,uphill_limit,offsets)
    direction=int(np.argmin(np.abs(np.arctan2(np.sin(angles-yaw),np.cos(angles-yaw)))))
    source = (*first, direction)
    rr,cc=np.indices(grid.known.shape)
    base=(grid.known & ~grid.hazard & (grid.obstacle_distance>=grid.footprint_radius+grid.clearance)
          & (grid.slope<=math.tan(math.radians(grid.MAX_SLOPE_DEG)))) if no_go_exit else (grid.known & ~grid.blocked)
    allowed=base & support
    if require_fresh_start:allowed &= (np.hypot(rr-first[0],cc-first[1])*grid.cell_size>2.) | grid.fresh
    def shifted(array,dr,dc,fill):
        result=np.full(array.shape,fill,dtype=array.dtype)
        a=slice(max(0,-dr),min(grid.n_cells,grid.n_cells-dr))
        b=slice(max(0,-dc),min(grid.n_cells,grid.n_cells-dc))
        result[a,b]=array[slice(max(0,dr),min(grid.n_cells,grid.n_cells+dr)),
                          slice(max(0,dc),min(grid.n_cells,grid.n_cells+dc))]
        return result
    valid_edges=np.empty_like(edge_costs,dtype=bool)
    for k,(dr,dc) in enumerate(offsets):
        touched=[]
        for t in np.linspace(0,1,math.ceil(math.hypot(dr,dc)*4)+1):
            p=tuple(np.floor(.5+t*np.array([dr,dc])).astype(int))
            if not touched or p!=touched[-1]:touched.append(p)
        corners=set(touched)
        for a,b in zip(touched,touched[1:]):
            if a[0]!=b[0] and a[1]!=b[1]:corners.update(((a[0],b[1]),(b[0],a[1])))
        valid=np.ones_like(allowed)
        for r,c in corners:valid &= shifted(allowed,r,c,False)
        for r,c in corners:valid &= shifted(np.isfinite(edge_costs[:,:,k]),r,c,False)
        if no_go_exit:
            for a,b in zip(touched,touched[1:]):
                valid &= shifted(grid.no_go_depth,*b,np.inf)<=shifted(grid.no_go_depth,*a,-np.inf)+1e-6
        valid_edges[:,:,k]=valid
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
            if not valid_edges[r,c,k]:continue
            norm = math.hypot(dr,dc)
            # Execution tests both ends of an edge. Ignoring a blocked source
            # direction produced routes the follower rejected forever.
            if not math.isfinite(edge_costs[r,c,k]) or not math.isfinite(edge_costs[nr,nc,k]): continue
            if node==source and not grid.segment_clear(start,grid.cell_to_world(nr,nc),
                                                       require_fresh=require_fresh_start,no_go_exit=no_go_exit):continue
            turns=abs(wrap_angle(angles[k]-angles[heading]))/(math.pi/4)
            g = travelled + edge_costs[nr,nc,k]+turns*turn_cost[nr,nc]
            if avoid is not None and avoid[nr,nc]:g+=6.*norm
            nxt = (nr,nc,k)
            if g < costs.get(nxt, math.inf):
                costs[nxt], parent[nxt] = g, node
                heapq.heappush(queue, (g+heuristic(nr,nc),g,nxt))
    # If the goal approach is unavailable, survey a reachable observed boundary.
    # Targets remain known cells; unknown neighbours only measure survey value.
    if not found and not allow_frontier:return [],False
    if not found and (strict_frontier or heuristic(*first)-best_h < .6/grid.cell_size):
        candidates=[]
        frontier_span=max(4,math.ceil((grid.footprint_radius+.8)/grid.cell_size))
        for node,travelled in costs.items():
            r,c,_=node
            if math.hypot(r-first[0],c-first[1])*grid.cell_size<.8: continue
            patch=grid.known[max(0,r-frontier_span):r+frontier_span+1,
                             max(0,c-frontier_span):c+frontier_span+1]
            unseen=int((~patch).sum())
            if unseen>=3:
                progress=(heuristic(*first)-heuristic(r,c))*grid.cell_size
                # The nearest unknown boundary can be behind the robot. Give
                # useful goal progress more weight than cheap nearby scan area.
                utility=3.*progress+.03*min(unseen,40)-.025*travelled*grid.cell_size
                candidates.append((utility,node))
        if not candidates: return [],False
        best=max(candidates,key=lambda item:item[0])[1]
    cells = [best]
    while cells[-1] != source:
        cells.append(parent[cells[-1]])
    points = [grid.cell_to_world(r,c) for r,c,_ in reversed(cells)]
    points[0] = tuple(start)
    if found and grid.segment_clear(points[-1], goal,no_go_exit=no_go_exit):
        points.append(tuple(goal))
    return points, found


@dataclass
class Motion:
    vx: float = 0.
    omega: float = 0.


class SlipTracker:
    """Command-versus-motion evidence, not an inferred friction coefficient."""
    def __init__(self):
        self.samples=[]
        self.metrics={'status':'idle','forward_mps':0.,'progress_mps':0.,'lateral_mps':0.,'window_s':0.}

    def reset(self):
        self.samples=[]
        self.metrics={'status':'idle','forward_mps':0.,'progress_mps':0.,'lateral_mps':0.,'window_s':0.}

    def update(self, now, xy, yaw, velocity, command, state, heading_error, cross_track):
        if state!='FOLLOW' or command.vx<.10-1e-6 or abs(heading_error)>1.1:
            self.reset();return None
        if self.samples and (now<=self.samples[-1][0] or now-self.samples[-1][0]>.3):self.reset()
        axis=np.array([math.cos(yaw),math.sin(yaw)])
        forward=float(np.dot(velocity,axis));lateral=float(np.dot(velocity,[-axis[1],axis[0]]))
        self.samples.append((now,np.asarray(xy).copy(),axis,forward,lateral,command.vx,cross_track))
        while len(self.samples)>1 and self.samples[1][0]<=now-3.:self.samples.pop(0)
        elapsed=now-self.samples[0][0]
        if elapsed<=0:return None
        travel=sum(float(np.dot(b[1]-a[1],a[2])) for a,b in zip(self.samples,self.samples[1:]))
        progress=travel/elapsed
        mean=float(np.mean([p[3] for p in self.samples]))
        lateral_mean=float(np.mean([p[4] for p in self.samples]))
        demand=float(np.mean([p[5] for p in self.samples]))
        threshold=min(.05,max(.02,.15*demand))
        duration=2. if demand>.25 else 3.
        self.metrics=dict(status='tracking',forward_mps=mean,progress_mps=progress,
                          lateral_mps=lateral_mean,window_s=elapsed,command_mps=demand)
        kind=None
        if elapsed>=1. and min(mean,progress)<-.10:kind='backsliding'
        elif elapsed>=1.5 and abs(lateral_mean)>.12 and cross_track-self.samples[0][6]>.20:
            kind='sideways slip'
        elif min(mean,progress)<threshold:
            self.metrics['status']='suspected stall'
            if elapsed>=duration-1e-6:kind='poor drive progress'
        if kind:
            self.metrics['status']=kind
            return dict(kind=kind,**self.metrics)
        return None


class CheckpointController:
    def __init__(self, grid, waypoints, cruise_speed=.4, preferred_uphill=20., uphill_limit=25., side_limit=25.):
        if not .1 <= cruise_speed <= .8:
            raise ValueError('cruise_speed must be between 0.1 and 0.8 m/s')
        self.grid, self.waypoints = grid, waypoints
        if not 0<preferred_uphill<=uphill_limit<=35 or not 0<side_limit<=35:
            raise ValueError('Require 0 < preferred uphill <= uphill limit <= 35 and 0 < side limit <= 35')
        self.preferred_uphill=preferred_uphill
        grid.uphill_limit_deg=uphill_limit;grid.side_limit_deg=side_limit
        self.retreat_required=False
        self.retreat_goal=None
        self.retreat_origin=None
        self.retreat_height=None
        self.retreat_started=None
        self.retreat_candidates=[]
        self.retreat_reverse_origin=None
        self.retreat_forward_only=False
        self.retreat_motion_samples=[]
        self.route_purpose='mission'
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
        self.slip_tracker=SlipTracker()
        self.slip_events=[]
        self.escape_events=[]
        self.maneuver_evaluations=[]
        self.escape_attempts=[]
        self.escape_origin=None
        self.escape_started=0.
        self.escape_total_started=0.
        self.escape_area_anchor=None
        self.escape_motion=Motion()
        self.escape_target=None
        self.escape_brake_samples=[]
        self.escape_retry_at=0.
        self.rejection_history=[]
        self.route_options=[]
        self.route_searches=[]
        self.previous_command=Motion()
        self.climb_failures=0
        self.backslide_distance=0.
        self.last_tick=None
        self.last_climb_event=''
        self.route_decision='initial route'
        self.motion_rejection=''
        self.last_success_position=None
        self.route_profile={}
        self.turn_repositions=0
        self.turn_attempts=[]
        self.turn_heading=0.
        self.turn_origin=None
        self.turn_last_yaw=0.
        self.turn_progress_at=0.
        self.reverse_boost=False
        self.turn_align_vx=None
        self.edge_plan=None
        self.edge_attempts=[]
        self.edge_failed_motions=[]
        self.edge_evaluations=[]
        self.edge_checked_at=-math.inf
        self.upright=1.

    def set_state(self, state, reason, now):
        if state in ('ESCAPE_BRAKE','ESCAPE_STOP') and state!=self.state:
            self.escape_brake_samples=[]
            self.escape_retry_at=now
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

    def checkpoint_mode(self, enabled, now):
        if enabled and not getattr(self, 'manual_checkpoint_mode', False):
            self.saved_mission = (self.waypoints, self.wp_idx)
        elif not enabled and getattr(self, 'manual_checkpoint_mode', False):
            self.waypoints, self.wp_idx = self.saved_mission
        self.manual_checkpoint_mode = enabled
        self.selection_status=''
        self.last_selection=None
        self.selected_goal=None
        self.request('stop', now)
        self.path, self.checkpoint, self.target, self.mission_route = [], None, None, []
        self.reason = 'click observed terrain in dashboard to choose checkpoint' if enabled else 'mission restored; press RESUME AUTO'

    def choose_checkpoint(self, xy, target, now, yaw=0.):
        if not getattr(self, 'manual_checkpoint_mode', False):
            return False
        target=tuple(float(v) for v in target)
        def report(accepted,detail,duplicate=False):
            kept=not accepted and getattr(self,'selected_goal',None) is not None
            self.selection_status=(f'({target[0]:.2f}, {target[1]:.2f}): '
                                   +('' if accepted else 'rejected: ')+detail
                                   +('; prior goal kept' if kept else ''))
            self.last_selection=dict(event='checkpoint_selection',time=now,position=list(xy),
                                     requested=list(target),accepted=accepted,duplicate=duplicate,
                                     prior_goal_kept=kept,reason=detail)
            if not accepted and self.state=='HOLD':
                self.reason='checkpoint rejected: '+detail
            return accepted
        if not np.isfinite(target).all(): return report(False,'invalid map coordinates')
        # A second button event for the same selected map cell must not restart
        # observation or revoke an accepted goal as the live map updates.
        if (self.state in ('OBSERVE','FOLLOW','SETTLE')
                and getattr(self,'selected_goal',None) is not None
                and self.grid.world_to_cell(*target)==self.grid.world_to_cell(*self.selected_goal)):
            return report(True,'already selected; continuing',duplicate=True)
        target_cell=self.grid.world_to_cell(*target)
        if not self.grid.traversable(target_cell):
            if target_cell is None or not self.grid.known[target_cell]: detail='unobserved terrain'
            elif self.grid.no_go[target_cell]:detail='learned no-go zone after repeated slip/stall episodes'
            elif self.grid.hazard[target_cell]: detail='mapped obstacle'
            elif self.grid.slope[target_cell]>math.tan(math.radians(self.grid.MAX_SLOPE_DEG)):
                detail='terrain exceeds 35-degree slope limit'
            else: detail='obstacle clearance margin'
            return report(False,'selected cell: '+detail)
        start=xy
        if not self.grid.traversable(self.grid.world_to_cell(*xy)):
            exit_path,_ = self.grid.clearance_exit(xy,goal=target)
            if exit_path: start=exit_path[-1]
        # Picking a destination uses accumulated geometry. OBSERVE and the
        # follower retain their fresh-scan gates before any commanded motion.
        route, complete = observed_route(self.grid, start, target,yaw,require_fresh_start=False)
        if not complete:
            detail=self.grid.support_reason(start)
            if detail: return report(False,'current footprint: '+detail)
            start_cell=self.grid.world_to_cell(*start)
            centre=self.grid.cell_to_world(*start_cell) if start_cell is not None else start
            detail=self.grid.support_reason(centre)
            if detail: return report(False,'current map-cell footprint: '+detail)
            detail=self.grid.support_reason(target)
            if not detail:
                detail=self.grid.support_reason(self.grid.cell_to_world(*self.grid.world_to_cell(*target)))
            if detail: return report(False,'destination footprint: '+detail)
            return report(False,'no connected observed route within footprint and uphill/side-slope limits')
        from types import SimpleNamespace
        # Commit the new selection only after validation. Rejected clicks leave
        # the active route, command state and mission untouched.
        self.waypoints = [SimpleNamespace(x=float(target[0]), y=float(target[1]), label='Selected checkpoint')]
        self.selected_goal=target
        self.retreat_required=False;self.retreat_goal=None;self.route_purpose='mission'
        self.wp_idx = 0
        self.route_mission = -1
        self.escape_clearance = None
        self.request('resume', now)
        return report(True,'accepted; refresh scans before following')

    def request(self, action, now, vx=0., omega=0.):
        self.edge_plan=None
        if action == 'resume':
            self.escape_area_anchor=None
            if self.retreat_required and self.retreat_started is not None:self.retreat_started=now
            self.drive_samples=[]
            self.slip_tracker.reset()
            self.rejection_history=[]
            self.escape_attempts=[]
            self.previous_command=Motion()
            self.manual = Motion()
            self.recovery_started = None
            self.recovery_attempts = []
            self.path, self.checkpoint, self.target = [], None, None
            self.mission_route = []
            self.mission_route_complete=False
            self.route_profile={}
            self.retry_count = 0
            self.replan_anchor = None
            self.turn_attempts=[]
            self.edge_attempts=[]
            self.edge_failed_motions=[]
            self.observe_revision = self.grid.revision
            self.set_state('OBSERVE', 'resume: refresh and replan', now)
        elif action in ('stop', 'manual'):
            self.slip_tracker.reset()
            self.rejection_history=[]
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

    def route_score(self, route, yaw):
        profile=self.grid.route_profile(route)
        # A short steep pinch can be hidden by integrated route cost. Penalize
        # that peak as well as accumulated effort, without maximizing distance.
        return (self.grid.route_cost(route,yaw)
                +4.*max(0.,profile['max_uphill_deg']-25.)**2
                +2.*max(0.,profile['max_side_slope_deg']-20.)**2
                +2.*max(0.,profile['max_footprint_slope_deg']-28.)**2)

    def choose_route(self, xy, goal, yaw):
        def route_key(path):
            # A reached cell centre may be repeated as the exact goal. Do not
            # report that harmless duplicate as a different route candidate.
            points=[]
            for point in path:
                if not points or math.dist(point,points[-1])>1e-9:points.append(tuple(point))
            return points
        support=self.grid.planning_support()
        route,complete=observed_route(self.grid,xy,goal,yaw,support=support,strict_frontier=True)
        candidates=[('balanced',route,complete)]
        self.route_searches=[dict(profile='balanced',uphill_limit=self.grid.uphill_limit_deg,
                                  connected=complete,has_route=len(route)>1)]
        if len(route)>1 and self.grid.route_profile(route)['max_uphill_deg']>15.:
            # Compare like destinations even before the final goal is connected:
            # alternative approaches to this same observed survey endpoint.
            search_goal=goal if complete else route[-1]
            avoid=np.zeros(self.grid.known.shape,bool)
            for point in route:
                cell=self.grid.world_to_cell(*point)
                if math.dist(point,xy)>1. and math.dist(point,search_goal)>1. and self.grid.slope[cell]>.25:
                    avoid[cell]=True
            for profile in ('low-grade','gentle'):
                option,done=observed_route(self.grid,xy,search_goal,yaw,profile='gentle',support=support,
                                           avoid=avoid if profile=='gentle' else None,
                                           uphill_limit=self.preferred_uphill if profile=='low-grade' else None,
                                           allow_frontier=False)
                self.route_searches.append(dict(profile=profile,connected=done,has_route=len(option)>1,
                    uphill_limit=self.preferred_uphill if profile=='low-grade' else self.grid.uphill_limit_deg))
                if done and not any(route_key(option)==route_key(old[1]) for old in candidates):
                    candidates.append((profile,option,complete))
        self.route_options=[]
        valid=[]
        for label,path,done in candidates:
            score=self.route_score(path,yaw) if len(path)>1 else math.inf
            info=dict(profile=label,complete=done,score=score if math.isfinite(score) else None,**self.grid.route_profile(path))
            self.route_options.append(info)
            if math.isfinite(score):valid.append((score,label,path,done,info['length_m']))
        if not valid:return [],False
        # Never replace a connected destination with a cheap short frontier.
        if any(item[3] for item in valid):valid=[item for item in valid if item[3]]
        shortest=min(item[4] for item in valid)
        practical=[item for item in valid if item[4]<=max(shortest*2.,shortest+6.)]
        modest=[item for item in practical if self.grid.route_profile(item[2])['max_uphill_deg']<=self.preferred_uphill+1e-6]
        if modest:practical=modest
        selected=min(practical,key=lambda item:item[0])
        for item in self.route_options:item['selected']=item['profile']==selected[1]
        return selected[2],selected[3]

    def plan_downhill_retreat(self, xy, yaw, now):
        """Commit to lower ground before considering the original ascent again."""
        grid=self.grid;cell=grid.world_to_cell(*xy)
        if cell is None or not grid.known[cell]:return False
        if self.retreat_origin is None:
            self.retreat_origin=tuple(xy);self.retreat_height=float(grid.surface[cell])
            self.retreat_started=now
            self.retreat_reverse_origin=tuple(xy);self.retreat_forward_only=False
            self.retreat_motion_samples=[]
        if now-self.retreat_started>90.:
            self.planning_failure='downhill retreat timed out; inspect or reposition manually'
            return False
        if (self.retreat_goal is not None and math.dist(xy,self.retreat_goal)<.6
                and not grid.no_go[cell] and grid.surface[cell]<=self.retreat_height-.4):
            self.escape_events.append(dict(event='retreat_complete',time=now,position=list(xy)))
            self.retreat_required=False;self.retreat_goal=None;self.route_purpose='mission'
            self.retreat_origin=None;self.retreat_height=None;self.retreat_started=None
            self.mission_route=[];self.replan_anchor=None;self.escape_area_anchor=None
            return self._plan(xy,yaw,now)
        support=grid.planning_support()
        if self.retreat_goal is None:
            rr,cc=np.indices(grid.known.shape)
            xx=grid.origin[0]+(rr+.5)*grid.cell_size-grid.world_size/2
            yy=grid.origin[1]+(cc+.5)*grid.cell_size-grid.world_size/2
            distance=np.hypot(xx-self.retreat_origin[0],yy-self.retreat_origin[1])
            possible=(grid.known & support & ~grid.blocked & np.isfinite(grid.surface)
                      & (grid.surface<=self.retreat_height-.5)
                      & (grid.slope<=math.tan(math.radians(self.preferred_uphill)))
                      & (distance>=2.5) & (distance<=8.))
            cells=np.argwhere(possible)
            score=distance+3.*grid.slope
            ordered=sorted(cells,key=lambda p:score[tuple(p)])
            goals=[]
            for p in ordered:
                goal=grid.cell_to_world(*p)
                if all(math.dist(goal,old)>1.2 for old in goals):goals.append(goal)
                if len(goals)>=4:break
        else:goals=[self.retreat_goal]
        choices=[];self.retreat_candidates=[]
        for goal in goals:
            route,done=observed_route(grid,xy,goal,yaw,support=support,uphill_limit=8.,
                                       no_go_exit=True,allow_frontier=False)
            profile=grid.route_profile(route)
            valid=done and len(route)>1 and profile['length_m']<=12.
            self.retreat_candidates.append(dict(goal=list(goal),reachable=valid,**profile))
            if valid:choices.append((profile['length_m']+profile['max_side_slope_deg']*.08,goal,route))
        if not choices:
            self.planning_failure='no observed downhill route to lower ground outside the failed climb'
            return False
        _,self.retreat_goal,route=min(choices,key=lambda option:option[0])
        self.route_purpose='retreat';self.mission_route=route;self.mission_route_complete=False
        self.route_mission=-1;self.route_profile=grid.route_profile(route)
        self.route_options=[];self.route_searches=[]
        end=1;distance=0.
        for i in range(1,len(route)):
            distance+=math.dist(route[i-1],route[i]);end=i
            if distance>=3.:break
        self.path=route[:end+1];self.checkpoint=self.path[-1];self.path_index=0
        self.is_mission_checkpoint=False;self.escape_clearance=None
        self.best_remaining=math.inf;self.progress_at=now
        self.route_decision='committed downhill retreat; original destination preserved'
        self.escape_events.append(dict(event='retreat_plan',time=now,position=list(xy),
                                      goal=list(self.retreat_goal),candidates=self.retreat_candidates))
        self.set_state('FOLLOW',self.route_decision,now)
        return True

    def _plan(self, xy, yaw, now):
        self.planning_failure='no observed clear route; inspect or reposition manually'
        if self.retreat_required:return self.plan_downhill_retreat(xy,yaw,now)
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
            elif self.grid.no_go[cell]:
                self.planning_failure='inside learned no-go zone; checked escape required'
            elif self.grid.hazard[cell]:
                self.planning_failure='hazard mapped at current position'
            elif self.grid.slope[cell]>math.tan(math.radians(self.grid.MAX_SLOPE_DEG)):
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
        route,complete=self.choose_route(xy,(wp.x,wp.y),yaw)
        fresh_cost=self.route_score(route,yaw) if len(route)>1 else math.inf
        old_cost=self.route_score(candidate,yaw) if len(candidate)>1 else math.inf
        # Only compare equivalent destinations. A short frontier is not a cheap
        # substitute for an available complete mission route.
        comparable=(len(route)>1 and len(candidate)>1 and math.dist(route[-1],candidate[-1])<.4)
        old_grade=self.grid.route_profile(candidate)['max_uphill_deg'] if candidate else math.inf
        new_grade=self.grid.route_profile(route)['max_uphill_deg'] if route else math.inf
        preferable_grade=new_grade<=self.preferred_uphill and old_grade>self.preferred_uphill
        if math.isfinite(old_cost) and not preferable_grade and (not math.isfinite(fresh_cost)
                or (self.mission_route_complete and not complete)
                or (comparable and fresh_cost>=.85*old_cost)):
            route,complete=candidate,self.mission_route_complete
            self.route_decision='retained route: alternative is not 15% cheaper'
        else:
            self.route_decision='selected best observed route' if complete else 'surveying reachable observed frontier'
            if len(self.route_options)>1:
                choice=next(item for item in self.route_options if item.get('selected'))
                self.route_decision=f"selected {choice['profile']} from {len(self.route_options)} uphill routes"
                if not complete:self.route_decision+=' to observed frontier'
        self.mission_route,self.mission_route_complete,self.route_mission=route,complete,self.wp_idx
        self.route_profile=self.grid.route_profile(route)
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
        def corridor(a,b,margin,check_direction=True):
            if self.route_purpose=='retreat' and check_direction:
                reason=self.grid.grade_reason(a,b,uphill_limit=8.)
                if reason:
                    self.motion_rejection=reason
                    return False
            return self.grid.segment_clear(a,b,margin,check_direction=check_direction,
                                           no_go_exit=self.state.startswith('ESCAPE_') or self.route_purpose=='retreat')
        # Route selection keeps the full planning margin. Slow execution may
        # use its extra buffer, but never the actual body envelope.
        margin = self.grid.footprint_radius if abs(motion.vx) <= .20 + 1e-6 else None
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
            if not corridor(xy,drift_end,self.grid.footprint_radius,check_direction=False):
                self.motion_rejection='drift stopping corridor: '+self.grid.segment_reason(xy,drift_end,self.grid.footprint_radius,check_direction=False)
                return False
            reason=self.grid.support_reason(xy,drift_end)
            if reason:
                self.motion_rejection='drift stopping corridor: '+reason
                return False
        if abs(motion.vx)<1e-6:
            # Extra turning room is for a measured steep edge, not a universal
            # flat-ground buffer around unknown cells or nearby obstacles.
            cliff=(abs(motion.omega)>.03 and
                   self.grid.max_support_slope(xy,reserve=.25)>self.grid.MAX_SLOPE_DEG)
            reason=self.grid.support_reason(xy,reserve=.25 if cliff else 0.)
            if reason:
                self.motion_rejection='turn clearance: '+reason
                return False
            if not corridor(xy,xy,margin):
                self.motion_rejection=self.grid.segment_reason(xy,xy,margin)
                return False
            return True
        if not corridor(xy,straight,margin):
            self.motion_rejection=self.motion_rejection or self.grid.segment_reason(xy,straight,margin)
            return False
        reason=self.grid.support_reason(xy,straight)
        if reason:
            self.motion_rejection='stopping footprint: '+reason
            return False
        horizon=max(.4,motion.vx**2/(2*accel)+.3)
        # Check under-turning as well as the requested arc. Rocky-slope can
        # respond slowly; trusting the commanded yaw alone cuts cliff corners.
        rates=(motion.omega,) if abs(motion.omega)<1e-6 else (0.,.35*motion.omega,motion.omega)
        for rate in rates:
            previous=xy
            for t in np.linspace(0,horizon/abs(motion.vx),9)[1:]:
                if abs(rate)<1e-6:
                    dx,dy=motion.vx*t,0.
                else:
                    dx=motion.vx*math.sin(rate*t)/rate
                    dy=motion.vx*(1-math.cos(rate*t))/rate
                point=(xy[0]+dx*math.cos(yaw)-dy*math.sin(yaw),
                       xy[1]+dx*math.sin(yaw)+dy*math.cos(yaw))
                if not corridor(previous,point,margin):
                    self.motion_rejection=self.motion_rejection or self.grid.segment_reason(previous,point,margin)
                    return False
                reason=self.grid.support_reason(previous,point)
                if reason:
                    self.motion_rejection='turn sweep: '+reason
                    return False
                previous=point
        return True

    def turn_slope_danger(self, xy, yaw, speed, motion):
        """Reverse assistance is only for measured excessive slopes in our sweep.

        Flat obstacles, unknown cells and planning margins cannot activate it.
        Look along actual drift and the possible under-turning paths, so a
        cliff beside a flat rim still counts, but a distant/behind cliff does not.
        """
        g=self.grid
        def steep(a,b=None,reserve=0.):
            return g.max_support_slope(a,b,reserve)>g.MAX_SLOPE_DEG
        accel=max(.1,self.braking_accel or .2)
        drift=float(np.linalg.norm(self.velocity_world))
        if drift>.10:
            end=tuple(np.asarray(xy)+(drift**2/(2*accel)+.1)*self.velocity_world/drift)
            if steep(xy,end): return True
        if abs(motion.vx)<1e-6:
            return steep(xy,reserve=.25)
        distance=max(.1,max(speed,abs(motion.vx))**2/(2*accel)+.1)
        direction=math.copysign(1.,motion.vx)
        end=(xy[0]+direction*distance*math.cos(yaw),xy[1]+direction*distance*math.sin(yaw))
        if steep(xy,end): return True
        horizon=max(.4,motion.vx**2/(2*accel)+.3)
        for rate in (0.,.35*motion.omega,motion.omega):
            previous=xy
            for t in np.linspace(0,horizon/abs(motion.vx),9)[1:]:
                dx=motion.vx*t if abs(rate)<1e-6 else motion.vx*math.sin(rate*t)/rate
                dy=0. if abs(rate)<1e-6 else motion.vx*(1-math.cos(rate*t))/rate
                point=(xy[0]+dx*math.cos(yaw)-dy*math.sin(yaw),
                       xy[1]+dx*math.sin(yaw)+dy*math.cos(yaw))
                if steep(previous,point): return True
                previous=point
        return False

    def prepare_turn(self, xy, yaw, now, heading, cause='steep terrain in turn sweep', motion=None, speed=0.):
        """Make room behind us before asking a slow-yaw policy to turn at an edge."""
        if motion is None:
            motion=Motion(.2,float(np.clip(.8*wrap_angle(heading-yaw),-.35,.35)))
        if not self.turn_slope_danger(xy,yaw,speed,motion): return False
        if sum(math.dist(xy,p)<1.5 for p in self.turn_attempts)>=3:
            return False
        for distance in (.4,.6,.8,1.):
            rear=(xy[0]-distance*math.cos(yaw),xy[1]-distance*math.sin(yaw))
            if not self.grid.segment_clear(xy,rear,self.grid.footprint_radius): continue
            if self.grid.support_reason(xy,rear): continue
            if self.grid.support_reason(rear,reserve=.25): continue
            # Stay out of stale/unknown rear terrain even if an old trajectory
            # passed through it. This is a measured corridor, not a blind reverse.
            self.reverse_target=rear
            self.turn_heading=heading
            self.turn_origin=tuple(xy)
            self.turn_last_yaw=yaw
            self.turn_progress_at=now
            self.turn_attempts.append(tuple(xy))
            self.turn_repositions+=1
            self.reverse_boost=False
            self.hold_anchor=None;self.stable_since=None
            self.pending_arrival=False
            self.set_state('TURN_BRAKE','tight turn: '+cause+'; brake before reversing',now)
            return True
        return False

    def turn_step(self, xy, yaw, speed, now):
        """Stop, back up, stop again, then turn slowly under the same policy."""
        error=wrap_angle(self.turn_heading-yaw)
        vx=.4*max(.25,math.cos(error)) if abs(error)<math.radians(80) else 0.
        normal=Motion(vx,float(np.clip(.8*error,-.35,.35)))
        # Once repositioning has bought enough space, immediately restore the
        # normal follower. Do not spend another 4–8 seconds in slow alignment.
        # Never change drive direction while still reversing.
        if self.state in ('TURN_BRAKE','TURN_ALIGN'):
            danger=self.turn_slope_danger(xy,yaw,speed,normal)
            if not danger and self.state=='TURN_ALIGN' and speed>=.08:
                self.hold_anchor=None;self.stable_since=None
                self.set_state('TURN_STOP','steep-edge risk cleared; stop before resuming route',now)
                return self.position_hold(xy,yaw)
            if ((self.state=='TURN_BRAKE' or speed<.08)
                    and (not danger or self.motion_clear(xy,yaw,speed,normal))):
                if self._plan(xy,yaw,now):
                    if self.turn_attempts:self.turn_attempts.pop()
                    self.hold_anchor=None
                    self.reason='turn corridor clear; normal route following'
                    return Motion()
                self.pending_arrival=False;self.mission_route=[]
                self.set_state('SETTLE','turn corridor clear; refresh unavailable route',now)
                return self.position_hold(xy,yaw)
        if self.state in ('TURN_BRAKE','TURN_STOP'):
            braking=self.position_hold(xy,yaw)
            if speed<.08:
                if self.stable_since is None:self.stable_since=now
            else:self.stable_since=None
            if self.stable_since is not None and now-self.stable_since>=.3:
                self.hold_anchor=None;self.stable_since=None
                if self.state=='TURN_BRAKE':
                    self.set_state('TURN_REVERSE','backing into room for the route turn',now)
                else:
                    self.turn_progress_at=now;self.turn_last_yaw=yaw
                    self.turn_origin=tuple(xy)
                    self.turn_align_vx=None
                    self.set_state('TURN_ALIGN','slow turn after backing clear of the edge',now)
            elif now-self.since>5.:
                return self.hold('cannot stop for tight turn; manual assistance required',now,terminal=True)
            return braking
        if self.state=='TURN_REVERSE':
            room=(math.dist(xy,self.turn_origin)>=.15 and
                  (not self.turn_slope_danger(xy,yaw,speed,normal)
                   or self.motion_clear(xy,yaw,speed,normal)))
            if math.dist(xy,self.reverse_target)<.12 or room:
                self.hold_anchor=None;self.stable_since=None
                self.set_state('TURN_STOP','turn room available; stop before changing direction',now)
                return self.position_hold(xy,yaw)
            bearing=math.atan2(xy[1]-self.reverse_target[1],xy[0]-self.reverse_target[0])
            error=wrap_angle(bearing-yaw)
            motion=Motion(-.15,float(np.clip(.6*error,-.12,.12)))
            if now-self.since>2. and math.dist(xy,self.turn_origin)<.08:
                self.reverse_boost=True
            if self.reverse_boost and math.dist(xy,self.reverse_target)>.20:
                # A creep request can sit in the policy's deadband. Use the
                # proven teleop reverse magnitude only if its larger stopping
                # corridor is clear; do not confuse command with actual travel.
                stronger=Motion(-.4,motion.omega)
                if self.motion_clear(xy,yaw,speed,stronger):
                    motion=stronger
                    self.reason='reverse creep ineffective; using checked teleop reverse command'
            if now-self.since>6. and math.dist(xy,self.turn_origin)<.08:
                return self.hold('reverse command produced no travel; manual assistance required',now,terminal=True)
            if (now-self.since>10. or abs(error)>.5
                    or not self.grid.segment_clear(xy,self.reverse_target,self.grid.footprint_radius)
                    or not self.motion_clear(xy,yaw,speed,motion)):
                self.pending_arrival=False;self.mission_route=[]
                self.set_state('SETTLE','reverse for turn stopped: '+(self.motion_rejection or 'corridor/heading/time limit'),now)
                return self.position_hold(xy,yaw)
            return motion
        error=wrap_angle(self.turn_heading-yaw)
        if abs(wrap_angle(yaw-self.turn_last_yaw))>.05:
            self.turn_last_yaw=yaw;self.turn_progress_at=now
        stalled=now-self.turn_progress_at>4.
        if abs(error)<.18 or math.dist(xy,self.turn_origin)>.6 or now-self.since>8. or stalled:
            if abs(error)<.18 and self.turn_attempts:
                self.turn_attempts.pop()  # Successful turns are not failed recovery attempts.
            self.pending_arrival=False;self.mission_route=[];self.replan_anchor=None
            self.hold_anchor=None
            reason='slow turn stalled' if stalled else 'slow turn complete'
            self.set_state('SETTLE',reason+'; refresh the mission route',now)
            return self.position_hold(xy,yaw)
        omega=float(np.clip(.6*error,-.20,.20))
        # Turn while backing if forward travel is excluded. Choose once after
        # the stop so changing costs cannot alternate forward/reverse each tick.
        choices=(.12,-.15,0.) if self.turn_align_vx is None else (self.turn_align_vx,)
        for vx in choices:
            motion=Motion(vx,omega)
            if self.motion_clear(xy,yaw,speed,motion):
                self.turn_align_vx=vx
                self.reason='slow reverse turning arc' if vx<0 else 'slow turn after backing clear of the edge'
                return motion
        self.pending_arrival=False;self.mission_route=[];self.hold_anchor=None
        self.set_state('SETTLE','slow turn stopped: '+self.motion_rejection,now)
        return self.position_hold(xy,yaw)

    @staticmethod
    def edge_arc(xy, yaw, distance, direction, curvature=0.):
        points=[]
        for s in np.linspace(0,distance,max(2,math.ceil(distance/.025)+1)):
            angle=curvature*s
            dx=direction*s if abs(curvature)<1e-9 else direction*math.sin(angle)/curvature
            dy=0. if abs(curvature)<1e-9 else direction*(1-math.cos(angle))/curvature
            points.append((xy[0]+dx*math.cos(yaw)-dy*math.sin(yaw),
                           xy[1]+dx*math.sin(yaw)+dy*math.cos(yaw)))
        return points

    def prepare_edge_exit(self, xy, yaw, now):
        """Choose a support-improving escape before applying ordinary route rules."""
        exposure=self.grid.support_exposure(xy)
        if exposure is None or not exposure[0] or self.upright<.65:return False
        self.edge_evaluations=[]
        if sum(math.dist(xy,p)<1.2 for p in self.edge_attempts)>=3:return False
        choices=[]
        for direction in (-1.,1.):
            for curvature in (0.,-.25,.25):
                if any(direction==d and curvature==k and math.dist(xy,p)<.5
                       and abs(wrap_angle(yaw-h))<.4 for d,k,p,h in self.edge_failed_motions):continue
                for distance in (.4,.6,.8):
                    paths=[self.edge_arc(xy,yaw,distance,direction,curvature*scale)
                           for scale in ((1.,) if not curvature else (0.,.35,1.))]
                    detail='';profile={}
                    for points in paths:
                        detail,profile=self.grid.support_exit_check(points,exposure[0])
                        if detail:break
                    entry=dict(profile,direction='reverse' if direction<0 else 'forward',curvature=curvature,
                               distance_m=distance,accepted=not bool(detail),reason=detail)
                    self.edge_evaluations.append(entry)
                    if detail:continue
                    end=paths[-1][-1]
                    visited=any(math.dist(end,p)<.3 for _,p in self.history)
                    score=distance+.5*abs(curvature)-(.12 if visited else 0.)
                    # When both improve support equally, prefer straight backout
                    # along the approach over pushing farther into the turn.
                    if direction<0:score-=.05
                    choices.append((score,direction,curvature,distance,end))
                    break
        self.escape_events.append(dict(event='edge_exit_decision',time=now,position=list(xy),yaw=yaw,
            initial_steep_cells=len(exposure[0]),candidates=self.edge_evaluations))
        if not choices:return False
        _,direction,curvature,distance,end=min(choices,key=lambda item:item[0])
        cell=self.grid.world_to_cell(*xy)
        uphill=direction*(self.grid.gx[cell]*math.cos(yaw)+self.grid.gy[cell]*math.sin(yaw))
        self.edge_plan=dict(origin=tuple(xy),yaw=yaw,direction=direction,curvature=curvature,
            distance=distance,target=end,remaining=exposure[0],initial_count=len(exposure[0]),
            initial_upright=self.upright,started=now,move_started=None,boost=uphill>math.tan(math.radians(15.)))
        self.edge_attempts.append(tuple(xy));self.pending_arrival=False
        self.mission_route=[];self.path=[tuple(xy),end];self.target=end
        self.checkpoint=end;self.is_mission_checkpoint=False
        self.route_profile=self.grid.route_profile(self.path)
        self.hold_anchor=None;self.stable_since=None;self.slip_tracker.reset()
        self.set_state('EDGE_BRAKE',f'partial cliff overlap: brake, then {"reverse" if direction<0 else "forward"} {distance:.1f}m to restore support',now)
        return True

    def retry_edge_exit(self, xy, yaw, now, reason):
        old=self.edge_plan
        self.edge_failed_motions.append((old['direction'],old['curvature'],tuple(xy),yaw))
        self.escape_events.append(dict(event='edge_exit_retry',time=now,position=list(xy),reason=reason))
        self.edge_plan=None
        self.edge_checked_at=now
        if self.prepare_edge_exit(xy,yaw,now):
            self.reason='edge recovery changed approach: '+reason
            self.hold_anchor=np.asarray(xy).copy();self.hold_heading=yaw
            return self.position_hold(xy,yaw)
        return self.hold('edge backout exhausted: '+reason,now,terminal=True)

    def edge_exit_step(self, xy, yaw, speed, now):
        plan=self.edge_plan
        if plan is None:return self.hold('edge recovery state unavailable',now,terminal=True)
        self.hold_anchor=np.asarray(xy).copy();self.hold_heading=yaw
        braking=self.position_hold(xy,yaw)
        if self.upright<.6 or self.upright<plan['initial_upright']-.12:
            self.edge_plan=None
            return self.hold('edge backout stopped: body tilt deteriorated',now,terminal=True)
        if now-plan['started']>10.:
            self.edge_plan=None
            return self.hold('edge backout timed out; manual assistance required',now,terminal=True)
        current=self.grid.support_exposure(xy)
        if current is None or not current[0].issubset(plan['remaining']):
            if (current is not None and len(current[0])/current[1]<=.25
                    and math.dist(xy,plan['origin'])<.25):
                # Measured drift is not the same as a commanded cliff entry.
                # Re-evaluate from the actual pose, retire the ineffective arc,
                # and permit at most three checked approaches in this area.
                return self.retry_edge_exit(xy,yaw,now,'observed cliff overlap increased')
            self.edge_plan=None
            return self.hold('edge backout stopped: footprint exposure increased or became unknown',now,terminal=True)
        plan['remaining']=current[0]
        if self.state=='EDGE_STOP':
            if speed<.10:
                if self.stable_since is None:self.stable_since=now
            else:self.stable_since=None
            if self.stable_since is not None and now-self.stable_since>=.3:
                self.escape_events.append(dict(event='edge_exit_complete',time=now,position=list(xy),
                    distance_m=math.dist(xy,plan['origin']),final_steep_cells=len(current[0])))
                self.edge_plan=None;self.replan_anchor=None;self.path=[];self.mission_route=[]
                self.observe_revision=self.grid.revision
                self.set_state('OBSERVE','backout restored support; refresh and replan',now)
            return braking
        if self.state=='EDGE_BRAKE':
            if speed>.12:
                if now-self.since>3.:
                    self.edge_plan=None
                    return self.hold('cannot slow for edge backout',now,terminal=True)
                return braking
            plan['move_started']=now
            self.escape_events.append(dict(event='edge_exit_started',time=now,position=list(xy),
                target=list(plan['target']),direction=plan['direction'],curvature=plan['curvature'],
                initial_steep_cells=plan['initial_count']))
            self.set_state('EDGE_MOVE','backing out of existing cliff overlap' if plan['direction']<0 else 'moving forward out of existing cliff overlap',now)
        distance=math.dist(xy,plan['origin'])
        progress=plan['direction']*float(np.dot(np.subtract(xy,plan['origin']),
                                              (math.cos(plan['yaw']),math.sin(plan['yaw']))))
        if not current[0] and (distance>=plan['distance']-.08 or math.dist(xy,plan['target'])<.1):
            self.set_state('EDGE_STOP','footprint clear of cliff; stop before replanning',now)
            self.stable_since=None
            return braking
        if now-plan['move_started']>=1.5 and progress<.06:plan['boost']=True
        if now-plan['move_started']>=4. and progress<.06:
            return self.retry_edge_exit(xy,yaw,now,'backout produced no travel in the requested direction')
        magnitude=.4 if plan['boost'] else .2
        expected=wrap_angle(plan['yaw']+plan['curvature']*min(distance,plan['distance']))
        error=wrap_angle(expected-yaw)
        if abs(error)>.4:
            self.edge_plan=None
            return self.hold('edge backout heading diverged; stop and inspect',now,terminal=True)
        omega=float(np.clip(plan['curvature']*magnitude+.6*error,-.12,.12))
        motion=Motion(plan['direction']*magnitude,omega)
        # Recheck actual drift and both ideal/under-turning footprints each tick.
        # The escape cannot grow a cliff overlap or sweep into unknown ground.
        accel=max(.1,self.braking_accel or .2)
        drift=float(np.linalg.norm(self.velocity_world))
        checks=[]
        if drift>.10:
            end=np.asarray(xy)+(drift*drift/(2*accel)+.05)*self.velocity_world/drift
            drift_points=[tuple(np.asarray(xy)+t*(end-xy)) for t in np.linspace(0,1,21)]
            counteracting=np.dot(self.velocity_world,plan['direction']*np.array([math.cos(yaw),math.sin(yaw)]))<-.03
            # Do not replace a checked -0.4 counter-command with weak -0.1
            # damping merely because inertia points toward the existing edge.
            # The requested sweep is still checked below; actual drift/tilt
            # is bounded separately and causes a different escape or HOLD.
            if not counteracting:checks.append(drift_points)
        horizon=max(.25,max(speed,magnitude)**2/(2*accel)+.1)
        for rate in (0.,.35*omega,omega):
            checks.append(self.edge_arc(xy,yaw,horizon,plan['direction'],rate/magnitude))
        for points in checks:
            if current[0]:
                detail,_=self.grid.support_exit_check(points,current[0],require_clear_end=False)
            else:
                # Once clear, preserve all ordinary physical constraints while
                # finishing the short backout, with the same 35-degree escape cap.
                detail=self.grid.grade_reason(points[0],points[-1],uphill_limit=35.,side_limit=35.)
                for a,b in zip(points,points[1:]):
                    if detail:break
                    if not self.grid.segment_clear(a,b,self.grid.footprint_radius,check_direction=False):
                        detail=self.grid.segment_reason(a,b,self.grid.footprint_radius,check_direction=False)
                    detail=detail or self.grid.support_reason(a,b)
            if detail:
                self.reason='edge backout braking: '+detail
                return braking
        self.reason=(f'{"reverse" if motion.vx<0 else "forward"} backout: steep footprint cells '
                     f'{plan["initial_count"]} → {len(current[0])}; '+('teleop boost' if plan['boost'] else 'checked escape'))
        return motion

    def local_maneuver(self, xy, yaw, speed, allow_gait_motion=False):
        """Rank bounded teleop arcs, without clearing hazards or inventing strafing."""
        self.maneuver_evaluations=[]
        if speed>(.20 if allow_gait_motion else .12):
            self.maneuver_evaluations.append(dict(reason='body still drifting too fast'))
            return None  # Brake before reversing or changing the recovery heading.
        cell=self.grid.world_to_cell(*xy)
        if cell is None or not self.grid.fresh[cell]:
            self.maneuver_evaluations.append(dict(reason='current position needs a fresh scan'))
            return None
        self.escape_clearance=None
        if self.grid.blocked[cell]:
            distance=float(self.grid.obstacle_distance[cell])
            if distance<self.grid.footprint_radius or self.grid.hazard[cell]:
                self.maneuver_evaluations.append(dict(reason='mapped obstacle inside current body footprint'))
                return None
            self.escape_clearance=self.grid.footprint_radius
        wp=self.waypoints[self.wp_idx]
        goal=(wp.x,wp.y)
        if self.retreat_required:
            gradient=np.array([self.grid.gx[cell],self.grid.gy[cell]])
            goal=self.retreat_goal or tuple(np.asarray(xy)-3.*gradient/max(np.linalg.norm(gradient),1e-6))
        bearing=math.atan2(goal[1]-xy[1],goal[0]-xy[0])
        candidates=[]
        motions=(.2,-.4,-.2,0.) if self.state.startswith('ESCAPE_') else (.2,-.2,0.)
        for vx in motions:
            for omega in (0.,.35,-.35):
                if vx==omega==0: continue
                motion=Motion(vx,omega)
                evaluation=dict(vx=vx,omega=omega,accepted=False)
                self.maneuver_evaluations.append(evaluation)
                # Do not repeat an ineffective action at the same place/heading.
                if any(motion==old and math.dist(xy,pos)<.35
                       and abs(wrap_angle(yaw-heading))<.25
                       for old,pos,heading in self.recovery_attempts):
                    evaluation['reason']='same motion already ineffective at this pose'
                    continue
                if not self.motion_clear(xy,yaw,speed,motion):
                    evaluation['reason']=self.motion_rejection
                    continue
                duration=2.
                if omega:
                    dx=vx*math.sin(omega*duration)/omega
                    dy=vx*(1-math.cos(omega*duration))/omega
                else:
                    dx,dy=vx*duration,0.
                end=(xy[0]+dx*math.cos(yaw)-dy*math.sin(yaw),
                     xy[1]+dx*math.sin(yaw)+dy*math.cos(yaw))
                endcell=self.grid.world_to_cell(*end)
                if endcell is None:
                    evaluation['reason']='endpoint outside map'
                    continue
                progress=math.dist(xy,goal)-math.dist(end,goal)
                alignment=(abs(wrap_angle(bearing-yaw))-
                           abs(wrap_angle(bearing-yaw-omega*duration)))
                clearance=min(2.,float(self.grid.obstacle_distance[endcell]))
                across=abs(-self.grid.gx[endcell]*math.sin(yaw)+self.grid.gy[endcell]*math.cos(yaw))
                score=3.*progress+.5*alignment+.3*clearance-.03*self.grid.cost[endcell]-.3*across
                if vx==0: score-=.25
                if self.state.startswith('ESCAPE_') and vx==-.4:score+=1.5
                evaluation.update(accepted=True,score=float(score),endpoint=list(end))
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

    def begin_escape(self, xy, yaw, now, evidence, mark=True):
        """One recorded episode, then a bounded change of approach."""
        if mark:
            self.grid.record_traversal([xy],yaw,failed=True)
            self.grid.record_difficulty(xy,yaw)
            self.climb_failures+=1
        cell=self.grid.world_to_cell(*xy)
        on_hillside=(cell is not None and self.grid.slope[cell]>math.tan(math.radians(15.)))
        wp=self.waypoints[self.wp_idx]
        goal_cell=self.grid.world_to_cell(wp.x,wp.y)
        climbing=(on_hillside and (self.grid.gx[cell]*math.cos(yaw)+self.grid.gy[cell]*math.sin(yaw)>math.tan(math.radians(8.))
                   or (goal_cell is not None and self.grid.known[goal_cell]
                       and self.grid.surface[goal_cell]>self.grid.surface[cell]+.4)))
        if on_hillside and climbing and self.route_purpose!='retreat':
            # Repeated route rejections also prove this approach is unusable to
            # this controller, without claiming that every rejection was slip.
            self.grid.exclude_failed_climb(xy,evidence['kind'],now)
            self.retreat_required=True;self.retreat_goal=None
            self.retreat_origin=None;self.retreat_height=None;self.retreat_started=None
        self.last_climb_event=(evidence['kind']+': '+evidence.get('detail','change approach')+'; '
                               +('failed zone recorded' if mark else 'no traction claim'))
        self.slip_events.append(dict(event='motion_difficulty',time=now,position=list(xy),yaw=yaw,
                                     marked=mark,**evidence))
        self.mission_route=[];self.path=[];self.target=None
        self.replan_anchor=None;self.pending_arrival=False
        # Briefly resuming FOLLOW must not erase failed recovery commands or
        # grant another full budget at the same patch.
        if self.escape_area_anchor is None or math.dist(xy,self.escape_area_anchor)>1.2:
            self.escape_attempts=[];self.escape_total_started=now
            self.escape_area_anchor=tuple(xy)
        self.hold_anchor=None;self.stable_since=None
        self.slip_tracker.reset();self.drive_samples=[];self.rejection_history=[]
        self.set_state('ESCAPE_BRAKE',self.last_climb_event,now)

    def motion_metrics(self):
        metrics=dict(self.slip_tracker.metrics)
        if self.state.startswith('ESCAPE_') and self.slip_events:
            event=self.slip_events[-1]
            metrics.update({k:event[k] for k in metrics if k in event})
            metrics['status']=event['kind']+'; recovering'
        return metrics

    def escape_step(self, xy, yaw, speed, now):
        """Brake, back off or turn on observed clearance, then choose a new route."""
        committed=self.retreat_required and self.retreat_started is not None
        expired=now-(self.retreat_started if committed else self.escape_total_started)>(90. if committed else 15.)
        if expired or len(self.escape_attempts)>3:
            return self.hold('slip recovery exhausted; no safe productive maneuver',now,terminal=True)
        if self.state in ('ESCAPE_BRAKE','ESCAPE_STOP'):
            # Resist drift at the actual position, not an obsolete uphill anchor.
            self.hold_anchor=np.asarray(xy).copy()
            braking=self.position_hold(xy,yaw)
            if speed<.10:
                if self.stable_since is None:self.stable_since=now
            else:self.stable_since=None
            self.escape_brake_samples.append((now,np.asarray(xy).copy()))
            while len(self.escape_brake_samples)>1 and self.escape_brake_samples[1][0]<=now-.3:
                self.escape_brake_samples.pop(0)
            elapsed=now-self.escape_brake_samples[0][0]
            net_speed=math.dist(xy,self.escape_brake_samples[0][1])/max(elapsed,1e-6)
            # A rocky-slope gait can oscillate faster than the net body drift.
            # This permits a checked motion, not a claim that the body stopped.
            # motion_clear still uses actual instantaneous velocity/stopping room.
            gait_settled=elapsed>=.3-1e-6 and net_speed<.08 and speed<.20
            steady=self.stable_since is not None and now-self.stable_since>=.2
            if not (steady or gait_settled):
                if now-self.since>3.:
                    return self.hold('cannot brake for slip recovery; manual assistance required',now,terminal=True)
                return braking
            self.stable_since=None;self.hold_anchor=None
            if self.state=='ESCAPE_STOP' or self.retreat_required:
                self.replan_anchor=None;self.mission_route=[]
                if self._plan(xy,yaw,now):
                    if not self.retreat_required:self.reason='new approach after slip/stall recovery'
                    return Motion()
            if len(self.escape_attempts)>=3:
                return self.hold('slip recovery exhausted; three checked motions attempted',now,terminal=True)
            if now<self.escape_retry_at:return braking
            # Try real, checked teleop motions. Do not repeat a failed command
            # at the same position/heading, or use the rejected goal bearing alone.
            old=self.recovery_attempts
            self.recovery_attempts=list(self.escape_attempts)
            option=self.local_maneuver(xy,yaw,speed,allow_gait_motion=gait_settled)
            self.recovery_attempts=old
            self.escape_events.append(dict(event='escape_decision',time=now,position=list(xy),
                net_drift_mps=net_speed,instantaneous_speed_mps=speed,
                chosen=None if option is None else dict(vx=option[0].vx,omega=option[0].omega),
                candidates=self.maneuver_evaluations))
            if option is None:
                reasons=[p.get('reason','') for p in self.maneuver_evaluations]
                reason=max(set(reasons),key=reasons.count) if reasons else 'no observed corridor'
                if now-self.since<=3.:
                    self.escape_retry_at=now+.2
                    self.reason='braking and rechecking escape: '+reason
                    return braking
                return self.hold('escape rejected: '+reason+'; manual assistance required',now,terminal=True)
            motion,end=option
            self.escape_motion=motion;self.escape_origin=tuple(xy)
            self.escape_yaw=yaw;self.escape_started=now;self.escape_target=end
            self.escape_attempts.append((motion,xy,yaw))
            self.target=end;self.path=[tuple(xy),end] if motion.vx else []
            self.set_state('ESCAPE_MOVE','slip recovery: '+('back off' if motion.vx<0 else 'change heading' if not motion.vx else 'leave failed approach'),now)
            return motion
        elapsed=now-self.escape_started
        travelled=math.dist(xy,self.escape_origin)
        turned=abs(wrap_angle(yaw-self.escape_yaw))
        ineffective=elapsed>=2.5 and (travelled<.08 if self.escape_motion.vx else turned<.08)
        if (elapsed>=3. or travelled>=.85 or turned>=.65 or ineffective
                or not self.motion_clear(xy,yaw,speed,self.escape_motion)):
            self.stable_since=None;self.hold_anchor=None
            self.set_state('ESCAPE_BRAKE' if ineffective else 'ESCAPE_STOP',
                           'escape made no progress; try another checked motion' if ineffective else 'escape complete; reassess route',now)
            return self.position_hold(xy,yaw)
        return self.escape_motion

    def tick(self, xy, yaw, speed, now, upright=1., velocity=None):
        actual=np.asarray(velocity[:2] if velocity is not None else
                          (speed*math.cos(yaw),speed*math.sin(yaw)))
        forward=float(np.dot(actual,(math.cos(yaw),math.sin(yaw))))
        if self.last_tick is not None and self.previous_command.vx>0 and self.state=='FOLLOW':
            self.backslide_distance+=max(0.,-forward)*max(0.,now-self.last_tick)
        self.last_tick=now
        cell=self.grid.world_to_cell(*xy)
        uphill=(cell is not None and self.grid.gx[cell]*math.cos(yaw)+self.grid.gy[cell]*math.sin(yaw)>.15)
        evidence=self.slip_tracker.update(now,xy,yaw,actual,self.previous_command,
                                         self.state if upright>=.3 and self.route_purpose!='retreat' else 'retreat or tipped',
                                         self.heading_error,self.tracking_error)
        if evidence:self.begin_escape(xy,yaw,now,evidence)
        elif (self.state=='FOLLOW' and self.slip_tracker.metrics['progress_mps']>=.1
              and (self.last_success_position is None or math.dist(xy,self.last_success_position)>.4)):
            self.grid.record_traversal([xy],yaw)
            self.last_success_position=tuple(xy)
        was_following=self.state=='FOLLOW'
        command=self._tick(xy,yaw,speed,now,upright,velocity)
        if (was_following and self.state=='SETTLE' and self.reason.startswith('cross-track deviation')
                and cell is not None and self.grid.slope[cell]>.25 and self.previous_command.vx>.25):
            # Side slip may retain forward wheel motion. Crossing the existing
            # tracking boundary is an observed failed approach, not a friction estimate.
            self.begin_escape(xy,yaw,now,dict(kind='slope tracking failure',cross_track_m=self.tracking_error,
                                            detail='lateral route error exceeded 0.75m'))
            command=self.position_hold(xy,yaw)
        # A command-rejection loop is not proof of tire slip. Detect it
        # separately, escape/replan, and do not invent a traction hazard.
        if was_following and self.state=='SETTLE' and self.reason.startswith(('command rejected:','route rejected:')):
            self.rejection_history=[(t,p) for t,p in self.rejection_history if now-t<15. and math.dist(xy,p)<.75]
            self.rejection_history.append((now,tuple(xy)))
            if len(self.rejection_history)>=3:
                self.begin_escape(xy,yaw,now,dict(kind='repeated command rejection',rejections=len(self.rejection_history)),mark=False)
                command=self.position_hold(xy,yaw)
        self.previous_command=command
        return command

    def _tick(self, xy, yaw, speed, now, upright=1., velocity=None):
        xy = tuple(xy)
        self.upright=upright
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
        if self.retreat_required and self.retreat_started is not None and now-self.retreat_started>90.:
            return self.hold('downhill regroup timeout; manual assistance required',now,terminal=True)
        if self.wp_idx>=len(self.waypoints):
            self.set_state('COMPLETE','all mission destinations confirmed',now)
            return Motion()
        if self.state.startswith('EDGE_'):
            return self.edge_exit_step(xy,yaw,speed,now)
        if (now-self.edge_checked_at>=.2 and not (self.state=='OBSERVE' and now-self.since<1.)
                and abs(float(velocity[2]) if velocity is not None and len(velocity)>2 else 0.)<.2):
            self.edge_checked_at=now
            if self.prepare_edge_exit(xy,yaw,now):
                return self.edge_exit_step(xy,yaw,speed,now)
        if self.state.startswith('TURN_'):
            return self.turn_step(xy,yaw,speed,now)
        if self.state.startswith('ESCAPE_'):
            return self.escape_step(xy,yaw,speed,now)
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
            goal=self.retreat_goal if self.route_purpose=='retreat' else (wp.x,wp.y)
            exit_path,clearance=self.grid.clearance_exit(xy,goal=goal)
            if exit_path and not (self.route_purpose=='retreat' and
                    self.grid.grade_reason(xy,exit_path[-1],uphill_limit=8.)):
                self.path,self.checkpoint,self.path_index=exit_path,exit_path[-1],0
                self.escape_clearance=clearance
                self.is_mission_checkpoint=False
                self.best_remaining,self.progress_at=math.inf,now
                self.reason='exit planning margin while preserving body clearance'
        # Follow continuous path progress, not distance to individual grid points.
        def route_clear(a,b):
            return (not (self.route_purpose=='retreat' and self.grid.grade_reason(a,b,uphill_limit=8.))
                    and self.grid.segment_clear(a,b,self.escape_clearance,no_go_exit=self.route_purpose=='retreat'))
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
                if not route_clear(a,b):
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
            if self.route_purpose!='retreat' and math.dist(xy,(wp.x,wp.y))<=.6:
                self.is_mission_checkpoint=True
            self.pending_arrival=True
            self.stop_speed,self.stop_distance,self.stop_started=speed,0.,now
            self.set_state('SETTLE','verify checkpoint arrival and stop',now)
            return Motion()
        # Short lookahead with line-of-sight validation; no direct goal shortcut.
        target=lookahead
        if not route_clear(xy,target):
            target=self.path[min(self.path_index+1,len(self.path)-1)]
            if not route_clear(xy,target):
                self.pending_arrival=False
                self.set_state('SETTLE','lookahead rejected: '+self.grid.segment_reason(xy,target,self.escape_clearance),now)
                return Motion()
        self.target=target
        self.heading_error=wrap_angle(math.atan2(target[1]-xy[1],target[0]-xy[0])-yaw)
        if self.route_purpose=='retreat':
            grade=math.degrees(math.atan(self.grid.slope[cell]))
            self.retreat_motion_samples.append((now,xy))
            while len(self.retreat_motion_samples)>1 and self.retreat_motion_samples[1][0]<=now-.3:
                self.retreat_motion_samples.pop(0)
            dt=now-self.retreat_motion_samples[0][0]
            net_speed=math.dist(xy,self.retreat_motion_samples[0][1])/dt if dt>=.25 else speed
            # Back off to obtain room, then face downhill on gentler terrain.
            # Do not keep a long reverse descent merely because the body began
            # pointed uphill. Measured overspeed invokes the same hold feedback.
            if (self.retreat_reverse_origin is not None and math.dist(xy,self.retreat_reverse_origin)>.6
                    and grade<=20.):self.retreat_forward_only=True
            turning_forward=self.retreat_forward_only and abs(self.heading_error)>math.pi/2
            if speed>.45 or (turning_forward and (speed>.20 or net_speed>.10)):
                self.hold_anchor=np.asarray(xy).copy()
                # This is velocity damping during a turn, not a request to
                # return to the heading of the last checkpoint stop.
                self.hold_heading=yaw
                self.reason='braking downhill speed' if speed>.45 else 'braking before facing downhill'
                return self.position_hold(xy,yaw)
        reverse_drive=(self.route_purpose=='retreat' and not self.retreat_forward_only
                       and abs(self.heading_error)>math.pi/2)
        if reverse_drive:self.heading_error=wrap_angle(self.heading_error-math.pi)
        if now-self.progress_at>30. and abs(self.heading_error)<.5:
            if self.route_purpose=='retreat':return self.hold('downhill retreat made no progress; inspect clearance',now,terminal=True)
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
        if reverse_drive:vx=-vx
        if self.escape_clearance is not None:
            vx=math.copysign(min(abs(vx),.20),vx)
        motion=Motion(vx,omega)
        if not self.motion_clear(xy,yaw,speed,motion):
            rejection=self.motion_rejection
            slow=Motion(math.copysign(min(.2,abs(vx)),vx),omega)
            if vx!=0 and self.motion_clear(xy,yaw,speed,slow):
                self.reason='following observed corridor at reduced speed'
                return slow
            # The old fixed 40 cm straight probe silently zeroed forward drive
            # even when the planned turn was clear, leaving a weak yaw request.
            # Try a slower, tighter turn only with verified stopping room.
            turning=Motion(math.copysign(min(.2,abs(vx)),vx),math.copysign(.35,self.heading_error))
            if error>.035 and self.motion_clear(xy,yaw,speed,turning):
                self.reason='following clear slow turning arc'
                return turning
            corner=self.path[min(self.path_index+1,len(self.path)-1)]
            corner_error=wrap_angle(math.atan2(corner[1]-xy[1],corner[0]-xy[0])-yaw)
            if reverse_drive:corner_error=wrap_angle(corner_error-math.pi)
            alignment=Motion(0.,float(np.clip(.8*corner_error,-.35,.35)))
            if speed<.1 and abs(corner_error)>.08 and self.motion_clear(xy,yaw,speed,alignment):
                # At a tight corner, align with the first planned segment.
                # Holding the old heading while replanning cannot make that
                # segment drivable; a distant lookahead may also under-turn it.
                self.target=corner
                self.heading_error=corner_error
                self.reason='turning toward route corner; forward corridor not yet clear'
                return alignment
            # Reposition only after normal/slow forward arcs and an ordinary
            # alignment turn are unsafe, and only for a measured steep edge.
            # Nearby obstacles, unknown cells or a large heading error alone
            # must never start this sequence on flat ground or a safe slope.
            if self.route_purpose!='retreat' and error>.20 and self.turn_slope_danger(xy,yaw,speed,motion):
                if self.prepare_turn(xy,yaw,now,wrap_angle(yaw+self.heading_error),
                                     rejection,motion=motion,speed=speed):
                    return self.position_hold(xy,yaw)
                if sum(math.dist(xy,p)<1.5 for p in self.turn_attempts)>=3:
                    return self.hold('steep-edge turn recovery exhausted; manual assistance required',now,terminal=True)
            self.pending_arrival=False
            self.set_state('SETTLE','command rejected: '+(self.motion_rejection or rejection),now)
            return self.position_hold(xy,yaw)
        self.reason=('aligning to downhill retreat' if vx==0 else 'backing downhill to regroup' if reverse_drive
                     else 'following downhill retreat') if self.route_purpose=='retreat' else ('aligning to route' if vx==0 else 'following observed route')
        return motion
