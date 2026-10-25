"""World-frame elevation evidence; local plane slope and protrusion are separate.

No simulator mesh queries or SLAM-map reinsertion. All distances are metres and
all timestamps are simulation seconds. Row=X, column=Y.
"""
import math
import numpy as np
from scipy import ndimage as ndi


class ObservedTerrainMap:
    MAX_SLOPE_DEG = 35.0
    STEP_THRESH = 0.20
    FRESH_SECONDS = 8.0
    SUBDIVISIONS = 4  # 5 cm evidence bins within each 20 cm planning cell
    CHANGE_CONFIRMATIONS = 3
    MATCH_TOLERANCE = .08

    def __init__(self, world_size=64.0, cell_size=0.20, origin=(0., 0.),
                 footprint_radius=0.55, clearance=0.10):
        self.world_size, self.cell_size, self.origin = world_size, cell_size, origin
        self.n_cells = int(round(world_size / cell_size))
        shape = (self.n_cells, self.n_cells)
        self.height = np.full(shape, np.nan, np.float32)
        self.low = np.full(shape, np.nan, np.float32)
        self.last_seen = np.full(shape, -np.inf)
        self.samples = np.zeros(shape, np.int32)
        evidence_shape = (*shape, self.SUBDIVISIONS**2)
        self._bin_low = np.full(evidence_shape, np.nan, np.float32)
        self._bin_high = np.full(evidence_shape, np.nan, np.float32)
        self._bin_count = np.zeros(evidence_shape, np.uint8)
        self._clear_count = np.zeros(evidence_shape, np.uint8)
        self._raise_count = np.zeros(evidence_shape, np.uint8)
        self._clear_candidate = np.zeros(evidence_shape, np.float32)
        self._raise_candidate = np.zeros(evidence_shape, np.float32)
        self._scan_low = np.full(evidence_shape, np.inf, np.float32)
        self._scan_high = np.full(evidence_shape, -np.inf, np.float32)
        self._scan_time = None
        self.known = np.zeros(shape, bool)
        self.fresh = self.known.copy()
        self.blocked = self.known.copy()
        self.hazard = self.known.copy()
        self.obstacle_distance = np.full(shape, np.inf)
        self.steep_distance = np.full(shape, np.inf)
        self.gx = np.zeros(shape)
        self.gy = np.zeros(shape)
        self.slope = np.zeros(shape)
        self.roughness = np.full(shape,np.nan,np.float32)
        self.roughness_support = np.zeros(shape,np.uint8)
        self.failures = np.zeros((*shape,8),np.uint8)
        self.successes = np.zeros((*shape,8),np.uint16)
        self.approach_blocked = np.zeros((*shape,8),bool)
        self.difficulty_episodes = np.zeros(shape,np.uint8)
        self.no_go = np.zeros(shape,bool)
        self.no_go_depth = np.zeros(shape)
        self.failed_climbs = []
        self.uphill_limit_deg = self.MAX_SLOPE_DEG
        self.side_limit_deg = self.MAX_SLOPE_DEG
        self.traversal_revision = 0
        self.residual = np.zeros(shape)
        self.cost = np.full(shape, 5., np.float32)
        self.surface = self.height.copy()
        self.surface_seen = self.last_seen.copy()
        self.footprint_radius = footprint_radius
        self.clearance = clearance
        self.now = 0.0
        self.revision = 0

    def world_to_cell(self, x, y):
        r = math.floor((x - self.origin[0] + self.world_size / 2) / self.cell_size)
        c = math.floor((y - self.origin[1] + self.world_size / 2) / self.cell_size)
        return (r, c) if 0 <= r < self.n_cells and 0 <= c < self.n_cells else None

    def cell_to_world(self, r, c):
        return (self.origin[0] + (r + .5) * self.cell_size - self.world_size / 2,
                self.origin[1] + (c + .5) * self.cell_size - self.world_size / 2)

    def ingest(self, points, sensor_origin, now, max_range=15.0):
        pts = np.asarray(points).reshape(-1, 3)
        pts = pts[np.isfinite(pts).all(axis=1)]
        ranges = np.linalg.norm(pts - np.asarray(sensor_origin), axis=1)
        pts = pts[(ranges > .10) & (ranges <= max_range)]
        ij = np.floor((pts[:, :2] - self.origin + self.world_size / 2) / self.cell_size).astype(int)
        valid = ((ij >= 0) & (ij < self.n_cells)).all(axis=1)
        ij, pts = ij[valid], pts[valid]
        if not len(pts):
            return
        if self._scan_time is not None and self._scan_time != now:
            self._fuse_scan()
        self._scan_time = now
        # Preserve different parts of a cell. A ray landing beside a boulder
        # must not overwrite a previous ray on its top, or the reverse.
        fraction = (pts[:, :2]-self.origin+self.world_size/2)/self.cell_size-ij
        local = np.clip(np.floor(fraction*self.SUBDIVISIONS).astype(int),
                        0,self.SUBDIVISIONS-1)
        ids = ((ij[:,0]*self.n_cells+ij[:,1])*self.SUBDIVISIONS**2
               + local[:,0]*self.SUBDIVISIONS+local[:,1])
        np.minimum.at(self._scan_low.ravel(), ids, pts[:,2])
        np.maximum.at(self._scan_high.ravel(), ids, pts[:,2])

    def _fuse_scan(self):
        """Fuse all sensors once per timestamp, retaining spatial evidence.

        Newly higher returns enter immediately for obstacle avoidance. Clearing
        a previously measured top requires three consistent scans at that same
        fine spatial bin; missing returns never vote that an obstacle is gone.
        """
        hit = np.isfinite(self._scan_low)
        if not hit.any():
            return
        low,high = self._bin_low[hit],self._bin_high[hit]
        lo,hi = self._scan_low[hit],self._scan_high[hit]
        count = self._bin_count[hit]
        new = count == 0
        alpha = 1./np.minimum(count.astype(float)+1.,32.)
        tolerance = self.MATCH_TOLERANCE
        for old,value,streak,candidate,inward in (
                (low,lo,self._raise_count,self._raise_candidate,'up'),
                (high,hi,self._clear_count,self._clear_candidate,'down')):
            change = value-old
            narrowing = (~new) & ((change>tolerance) if inward=='up' else (change < -tolerance))
            matches = np.abs(value-candidate[hit]) <= tolerance
            votes = np.where(narrowing,np.where(matches,streak[hit].astype(int)+1,1),0)
            streak[hit] = np.minimum(votes,self.CHANGE_CONFIRMATIONS)
            candidate[hit] = value
            confirmed = votes >= self.CHANGE_CONFIRMATIONS
            ordinary = (~new) & (np.abs(change)<=tolerance)
            old[ordinary] += alpha[ordinary]*(value[ordinary]-old[ordinary])
            widening = (~new) & (~narrowing) & (~ordinary)
            old[new | confirmed | widening] = value[new | confirmed | widening]
            streak[hit] = np.where(confirmed,0,streak[hit])
        # Independently confirmed/EWMA endpoints can cross when a formerly
        # broad interval collapses. Preserve both proposals as an enclosing
        # interval; do not let the lower bound exceed the upper bound.
        self._bin_low[hit],self._bin_high[hit] = np.minimum(low,high),np.maximum(low,high)
        self._bin_count[hit] = np.minimum(count.astype(int)+1,32)
        touched = hit.any(axis=-1)
        lows = self._bin_low[touched]
        highs = self._bin_high[touched]
        self.low[touched] = np.min(np.where(np.isfinite(lows),lows,np.inf),axis=-1)
        self.height[touched] = np.max(np.where(np.isfinite(highs),highs,-np.inf),axis=-1)
        self.last_seen[touched] = self._scan_time
        self.samples[touched] += 1
        self._scan_low[hit],self._scan_high[hit] = np.inf,-np.inf
        self._scan_time = None

    def retain_radius(self, xy, radius):
        """Forget evidence outside a rolling radius; caller rebuilds before use."""
        self._fuse_scan()
        axis=(np.arange(self.n_cells)+.5)*self.cell_size-self.world_size/2
        outside=(axis[:,None]+self.origin[0]-xy[0])**2+(axis[None,:]+self.origin[1]-xy[1])**2 > radius**2
        self._retention_mask=~outside
        for name in ('height','low','_bin_low','_bin_high'):
            getattr(self,name)[outside]=np.nan
        self.last_seen[outside]=-np.inf
        for name in ('samples','_bin_count','_clear_count','_raise_count','_clear_candidate','_raise_candidate','failures','successes'):
            getattr(self,name)[outside]=0
        self._scan_low[outside]=np.inf;self._scan_high[outside]=-np.inf
        self.roughness[outside]=np.nan;self.roughness_support[outside]=0
        self.traversal_revision+=1

    def rebuild(self, now):
        self._fuse_scan()
        self.now = now
        # Preserve observed static terrain for global route continuity. Freshness
        # gates local motion separately; old observations are not unseen space.
        measured = np.isfinite(self.height)
        if not measured.any():
            self.known[:] = False
            self.fresh[:] = False
            self.cost[:] = 5
            self.blocked[:] = False
            self.fresh[:] = False
            self.hazard[:] = False
            self.surface[:] = np.nan
            self._mapped_area = 0.
            self.revision += 1
            return
        z = np.where(measured, .5 * (self.low + self.height), 0.)
        # Local least-squares ground plane from a 2.2 m neighbourhood. Two
        # passes suppress elevated returns before fitting the supporting ground.
        d = np.arange(-5, 6) * self.cell_size
        xx, yy = np.meshgrid(d, d, indexing='ij')
        # Plane moments are separable outer products. Two 11-tap passes
        # replace each 121-tap 2D filter without changing neighbourhoods.
        def moments(values, powers):
            first={p:ndi.correlate1d(values,d**p,axis=0,mode='constant') for p in {a for a,b in powers}}
            return [ndi.correlate1d(first[p],d**q,axis=1,mode='constant') for p,q in powers]
        powers=[(0,0),(1,0),(0,1),(2,0),(1,1),(0,2)]
        weights = measured.astype(float)
        for _ in range(2):
            sums = moments(weights,powers)
            rhs = np.stack(moments(weights*z,powers[:3]),axis=-1)
            matrix = np.empty((*z.shape, 3, 3))
            matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2] = sums[:3]
            matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2] = sums[1], sums[3], sums[4]
            matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2] = sums[2], sums[4], sums[5]
            matrix += np.eye(3) * 1e-5
            active = sums[0] > 0
            fit = np.zeros_like(rhs)
            fit[active] = np.linalg.solve(matrix[active], rhs[active, ..., None])[..., 0]
            ground, gx, gy = fit[..., 0], fit[..., 1], fit[..., 2]
            residual = z - ground
            weights = measured * np.where(residual > self.STEP_THRESH * .5, .10, 1.)
        support = np.rint(ndi.uniform_filter(measured.astype(float),size=11,mode='constant')*121)
        distance, nearest = ndi.distance_transform_edt(~measured, sampling=self.cell_size,
                                                     return_indices=True)
        self.gx, self.gy = gx, gy
        # Within-cell relief after removing the fitted ground inclination.
        axis=(np.arange(self.SUBDIVISIONS)+.5)*self.cell_size/self.SUBDIVISIONS-self.cell_size/2
        bx,by=np.meshgrid(axis,axis,indexing='ij')
        values=(.5*(self._bin_low[measured]+self._bin_high[measured])
                -gx[measured,None]*bx.ravel()-gy[measured,None]*by.ravel())
        good=np.isfinite(values)
        count=good.sum(axis=-1)
        mean=np.where(good,values,0.).sum(axis=-1)/np.maximum(count,1)
        variance=np.where(good,(values-mean[...,None])**2,0.).sum(axis=-1)/np.maximum(count,1)
        self.roughness_support[measured]=count.astype(np.uint8)
        self.roughness[measured]=np.where(count>=6,np.sqrt(variance),np.nan).astype(np.float32)
        self.slope = np.hypot(gx, gy)
        spread = np.where(measured, self.height - self.low, 0.)
        # Remove the height variation explained by a sloping 20 cm cell.
        excess = spread - self.cell_size * (np.abs(gx) + np.abs(gy))
        # Returns are aggregated into 20 cm cells, not sampled at their centres.
        # Reserve one cell's ground-height variation as spatial uncertainty.
        # Without this, a small scarp on a slope can appear taller than 20 cm.
        self.residual = np.where(measured, self.height - ground
                                 - self.cell_size*(np.abs(gx)+np.abs(gy)), 0.)
        # A crater rim is an extended ridge, not an isolated boulder. Require
        # protrusion in multiple directions; a direction that follows the ridge
        # has little height residual. Unknown opposite samples cannot clear it.
        compact = np.full(z.shape, np.inf)
        jump = np.zeros(z.shape)
        def shifted(array, dr, dc, fill=0.):
            out=np.full_like(array,fill)
            sr=slice(max(0,dr),min(array.shape[0],array.shape[0]+dr))
            sc=slice(max(0,dc),min(array.shape[1],array.shape[1]+dc))
            tr=slice(max(0,-dr),min(array.shape[0],array.shape[0]-dr))
            tc=slice(max(0,-dc),min(array.shape[1],array.shape[1]-dc))
            out[tr,tc]=array[sr,sc]
            return out
        for dr,dc in [(5,0),(0,5),(3,4),(4,-3)]:
            both=measured & shifted(measured,dr,dc) & shifted(measured,-dr,-dc)
            bump=z-.5*(shifted(z,dr,dc)+shifted(z,-dr,-dc))
            compact=np.where(both,np.minimum(compact,bump),compact)
        for dr,dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            pair=measured & shifted(measured,dr,dc)
            unexplained=(np.abs(shifted(z,dr,dc)-z-self.cell_size*(gx*dr+gy*dc))
                         - self.cell_size*(np.abs(gx)+np.abs(gy)))
            jump=np.maximum(jump,np.where(pair,unexplained,0.))
        self.hazard = measured & (support >= 6) & (
            ((self.residual > self.STEP_THRESH) & (compact > self.STEP_THRESH*.5))
            | (excess > self.STEP_THRESH) | (jump > self.STEP_THRESH))
        # Fill only small sampling gaps supported by a plane and nearby returns.
        # Unknown expanses and occluded holes never become free by interpolation.
        supported = (support >= 6) & (distance <= self.cell_size * 1.42)
        self.known = measured | (supported & ~ndi.maximum_filter(self.hazard, size=3))
        if hasattr(self,'_retention_mask'): self.known &= self._retention_mask
        self.surface = np.where(measured, self.height, np.where(self.known, ground, np.nan))
        self.surface_seen = self.last_seen[tuple(nearest)]
        self.fresh = self.known & (now-self.surface_seen <= self.FRESH_SECONDS)
        obstacle_distance = (ndi.distance_transform_edt(~self.hazard, sampling=self.cell_size)
                             if self.hazard.any() else np.full(self.hazard.shape, np.inf))
        self.obstacle_distance = obstacle_distance
        inflated = obstacle_distance <= self.footprint_radius + self.clearance
        # User-reported operating ceiling; steep cells are not inflated.
        # Roughness does not override this limit or guarantee traction below it.
        steep=self.known & (self.slope > math.tan(math.radians(self.MAX_SLOPE_DEG)))
        self.steep_distance=(ndi.distance_transform_edt(~steep,sampling=self.cell_size)
                             if steep.any() else np.full(shape=self.known.shape,fill_value=np.inf))
        self.blocked = inflated | steep | self.no_go
        self.cost = np.where(self.known, 1. + 4.*self.slope, 5.).astype(np.float32)
        near = np.maximum(0., 1. - obstacle_distance / (self.footprint_radius + self.clearance + .6))
        self.cost[self.known] += 4 * near[self.known]
        # Excess relief adds difficulty, never a presumed traction reward.
        self.cost[self.known] += np.minimum(4.,10.*np.maximum(0.,np.nan_to_num(self.roughness[self.known])-.06))
        self.cost[self.blocked] = 20.
        self._mapped_area=float(measured.sum()*self.cell_size**2)
        self.revision += 1

    def traversable(self, cell, require_fresh=False):
        return (cell is not None and self.known[cell] and not self.blocked[cell]
                and (not require_fresh or self.fresh[cell]))

    def exit_traversable(self, cell, require_fresh=False):
        """Permit leaving a learned exclusion, never a geometric obstacle."""
        return (cell is not None and self.known[cell] and (not require_fresh or self.fresh[cell])
                and not self.hazard[cell] and self.obstacle_distance[cell]>=self.footprint_radius+self.clearance
                and self.slope[cell]<=math.tan(math.radians(self.MAX_SLOPE_DEG)))

    def grade_reason(self, a, b, uphill_limit=None, side_limit=None):
        delta=np.subtract(b,a);length=float(np.linalg.norm(delta))
        if length<1e-9:return ''
        up=self.uphill_limit_deg if uphill_limit is None else uphill_limit
        side=self.side_limit_deg if side_limit is None else side_limit
        for t in np.linspace(0,1,max(2,math.ceil(length/(self.cell_size*.5))+1)):
            cell=self.world_to_cell(*(np.asarray(a)+t*delta))
            if cell is None or not self.known[cell]:return 'unobserved grade'
            along=math.degrees(math.atan((self.gx[cell]*delta[0]+self.gy[cell]*delta[1])/length))
            across=math.degrees(math.atan(abs(-self.gx[cell]*delta[1]+self.gy[cell]*delta[0])/length))
            if along>up+1e-6:return f'uphill grade {along:.1f} exceeds autonomous limit {up:.0f} degrees'
            if across>side+1e-6:return f'side slope {across:.1f} exceeds autonomous limit {side:.0f} degrees'
        return ''

    def segment_clear(self, a, b, escape_clearance=None, require_fresh=True, check_direction=True,
                      no_go_exit=False, *, observation_support=None):
        observed = self.fresh if observation_support is None else observation_support
        length = np.linalg.norm(np.subtract(b, a))
        if check_direction and self.grade_reason(a,b):return False
        def allowed_cell(cell):
            allowed = (self.traversable(cell,require_fresh=False)
                       and (not require_fresh or observed[cell]))
            if not allowed and escape_clearance is not None and cell is not None:
                allowed = (self.known[cell] and observed[cell] and not self.hazard[cell]
                           and not self.no_go[cell]
                           and self.obstacle_distance[cell] <= self.footprint_radius + self.clearance
                           and self.obstacle_distance[cell] >= max(self.footprint_radius, escape_clearance) - 1e-6
                           and self.slope[cell] <= math.tan(math.radians(self.MAX_SLOPE_DEG)))
            if not allowed and no_go_exit and cell is not None and self.no_go[cell]:
                margin=self.footprint_radius if escape_clearance is not None else self.footprint_radius+self.clearance
                allowed=(self.known[cell] and (observed[cell] or not require_fresh)
                         and not self.hazard[cell] and self.obstacle_distance[cell]>=margin
                         and self.slope[cell]<=math.tan(math.radians(self.MAX_SLOPE_DEG)))
            return allowed
        heading=self.heading_bin(math.atan2(b[1]-a[1],b[0]-a[0])) if length>1e-9 else None
        previous = None
        first=self.world_to_cell(*a)
        depth=self.no_go_depth[first] if first is not None else 0.
        for t in np.linspace(0, 1, max(2, math.ceil(length / (self.cell_size * .4)) + 1)):
            p = np.asarray(a) + t * np.subtract(b, a)
            cell = self.world_to_cell(*p)
            if not allowed_cell(cell) or (check_direction and heading is not None and not self.direction_allowed(cell,heading)):
                return False
            if no_go_exit and check_direction:
                # Only a commanded escape must move monotonically outwards.
                # Predicted inertial drift is checked against physical hazards,
                # not treated as an intentional retry of the learned exclusion.
                if self.no_go_depth[cell]>depth+1e-6: return False
                depth=self.no_go_depth[cell]
            if previous is not None and cell[0]!=previous[0] and cell[1]!=previous[1]:
                if not allowed_cell((cell[0],previous[1])) or not allowed_cell((previous[0],cell[1])):
                    return False
            previous=cell
        return True

    def clearance_exit(self, xy, goal=None):
        """Exit a newly inflated safety margin without entering the body envelope.

        Never relax actual footprint clearance. The extra planning buffer may
        be crossed while reconnecting to the normal route.
        This handles a newly observed side boulder placing the robot itself in
        the extra 10 cm planning margin. A blocked footprint still requires HOLD.
        """
        start = self.world_to_cell(*xy)
        if start is None or not self.fresh[start]:
            return [], None
        clearance = self.footprint_radius
        if self.obstacle_distance[start] < self.footprint_radius or self.hazard[start]:
            return [], None
        candidates=[]
        for dr in range(-5,6):
            for dc in range(-5,6):
                r,c=start[0]+dr,start[1]+dc
                if 0<=r<self.n_cells and 0<=c<self.n_cells and self.traversable((r,c)):
                    target=self.cell_to_world(r,c)
                    if math.dist(xy,target)>.3 and self.segment_clear(xy,target,clearance):
                        candidates.append((math.dist(xy,target)+(.5*math.dist(target,goal) if goal is not None else 0.),target))
        if not candidates:
            return [], None
        return [tuple(xy),min(candidates)[1]],clearance

    @staticmethod
    def heading_bin(yaw):
        return int(round((yaw % (2*math.pi))/(math.pi/4))) % 8

    def record_traversal(self, points, yaw, failed=False):
        """One episode votes once per traversed cell and direction."""
        direction=self.heading_bin(yaw)
        cells=set()
        radius=max(1,math.ceil(.3/self.cell_size)) if failed else 0
        for point in points:
            cell=self.world_to_cell(*point)
            if cell is None: continue
            for dr in range(-radius,radius+1):
                for dc in range(-radius,radius+1):
                    r,c=cell[0]+dr,cell[1]+dc
                    if (dr*dr+dc*dc<=radius*radius and 0<=r<self.n_cells
                            and 0<=c<self.n_cells and self.known[r,c]): cells.add((r,c))
        array=self.failures if failed else self.successes
        ceiling=np.iinfo(array.dtype).max
        for r,c in cells: array[r,c,direction]=min(ceiling,int(array[r,c,direction])+1)
        if cells: self.traversal_revision+=1

    def direction_allowed(self, cell, heading):
        return (cell is not None and self.failures[cell[0],cell[1],heading]<2
                and not self.approach_blocked[cell[0],cell[1],heading])

    def record_difficulty(self, xy, yaw):
        """Confirmed episodes block an approach immediately; recurrence closes a zone.

        Unlike raw scan retention, learned exclusions survive for this mission.
        Escape may leave a learned zone, never bypass a geometric obstacle.
        """
        centre=self.world_to_cell(*xy)
        if centre is None: return
        radius=max(.35,self.footprint_radius+.1)
        extent=math.ceil(radius/self.cell_size)
        direction=self.heading_bin(yaw)
        for dr in range(-extent,extent+1):
            for dc in range(-extent,extent+1):
                r,k=centre[0]+dr,centre[1]+dc
                if (math.hypot(dr,dc)*self.cell_size>radius or not
                        (0<=r<self.n_cells and 0<=k<self.n_cells) or not self.known[r,k]):continue
                self.difficulty_episodes[r,k]=min(255,int(self.difficulty_episodes[r,k])+1)
                for heading in ((direction-1)%8,direction,(direction+1)%8):
                    self.approach_blocked[r,k,heading]=True
                if self.difficulty_episodes[r,k]>=2:self.no_go[r,k]=True
        self.no_go_depth=ndi.distance_transform_edt(self.no_go,sampling=self.cell_size)
        self.blocked|=self.no_go
        self.cost[self.no_go]=20.
        self.traversal_revision+=1
        self.revision+=1

    def exclude_failed_climb(self, xy, reason, now):
        """Stop re-entering a failed hillside from a marginally different yaw.

        A route exclusion is evidence of an unsuccessful approach, not a new
        claim about friction or geometric slope. A checked downhill exit remains
        possible from inside it.
        """
        radius=max(1.2,2.5*self.footprint_radius)
        if any(math.dist(xy,z['position'])<radius*.7 for z in self.failed_climbs):return
        axis=(np.arange(self.n_cells)+.5)*self.cell_size-self.world_size/2
        area=((axis[:,None]+self.origin[0]-xy[0])**2+(axis[None,:]+self.origin[1]-xy[1])**2<=radius**2)
        self.no_go|=area & self.known
        self.no_go_depth=ndi.distance_transform_edt(self.no_go,sampling=self.cell_size)
        self.blocked|=self.no_go;self.cost[self.no_go]=20.
        self.failed_climbs.append(dict(position=list(xy),radius_m=radius,reason=reason,time=now))
        self.revision+=1;self.traversal_revision+=1

    def _support_cells(self, a, b=None, reserve=0.):
        """Grid cells intersecting the swept circular body envelope."""
        a=np.asarray(a);b=a if b is None else np.asarray(b)
        radius=self.footprint_radius+reserve
        lo=self.world_to_cell(*(np.minimum(a,b)-radius-self.cell_size))
        hi=self.world_to_cell(*(np.maximum(a,b)+radius+self.cell_size))
        if lo is None or hi is None: return None
        rr,cc=np.meshgrid(np.arange(lo[0],hi[0]+1),np.arange(lo[1],hi[1]+1),indexing='ij')
        points=np.stack((self.origin[0]+(rr+.5)*self.cell_size-self.world_size/2,
                         self.origin[1]+(cc+.5)*self.cell_size-self.world_size/2),axis=-1)
        delta=b-a
        t=np.clip(np.sum((points-a)*delta,axis=-1)/max(1e-12,float(delta@delta)),0,1)
        mask=np.linalg.norm(points-a-t[...,None]*delta,axis=-1)<=radius+self.cell_size*.71
        return rr[mask],cc[mask]

    def max_support_slope(self, a, b=None, reserve=0.):
        """Maximum measured slope in degrees; missing evidence is not a cliff.

        This classifies slope danger only. support_reason/segment_clear still
        prevent authorizing motion across unknown or stale ground.
        """
        cells=self._support_cells(a,b,reserve)
        if cells is None: return math.nan
        slopes=self.slope[cells]
        measured=self.known[cells] & np.isfinite(slopes)
        return math.degrees(math.atan(float(slopes[measured].max()))) if measured.any() else math.nan

    def support_reason(self, a, b=None, reserve=0.):
        """Known ground under a swept body footprint, without changing tile colours.

        Steep cells have no inflated display buffer. Checking only the centre
        of a turning robot can therefore miss a wheel hanging over that edge.
        The optional reserve allows for drift during a slow alignment turn.
        """
        cells=self._support_cells(a,b,reserve)
        if cells is None: return 'body footprint extends outside observed map'
        if not self.known[cells].all(): return 'unobserved ground under body/turn envelope'
        # segment_clear checks freshness along the immediate motion corridor.
        # Perimeter cells may be occluded by the robot itself. Preserve their
        # measured static geometry instead of requiring every wheel-adjacent
        # sample to be reobserved continuously; unknown ground stays excluded.
        # Ordinary obstacle checks already inflate by the body radius. Do not
        # add another grid-cell margin to cream exits. Only an extra turn-drift
        # reserve needs a separate obstacle-envelope check here.
        if reserve>0 and self.hazard[cells].any(): return 'obstacle in body/turn envelope'
        slopes=self.slope[cells]
        if not np.isfinite(slopes).all(): return 'unmeasured slope under body/turn envelope'
        if (slopes>math.tan(math.radians(self.MAX_SLOPE_DEG))).any():
            return 'slope >35 degrees in body/turn envelope'
        return ''

    def planning_support(self):
        """Apply the execution footprint to route centres, with no extra reserve.

        Tile colours describe terrain at a cell; a centre-only route beside a
        cliff could still place wheels on that cliff. Do not repeatedly offer
        such a route to a follower that correctly rejects its body envelope.
        """
        radius=self.footprint_radius/self.cell_size+.71
        axis=np.arange(-math.ceil(radius),math.ceil(radius)+1)
        disk=axis[:,None]**2+axis[None,:]**2<=radius**2
        ground=self.known & np.isfinite(self.slope) & (self.slope<=math.tan(math.radians(self.MAX_SLOPE_DEG)))
        return ndi.minimum_filter(ground,footprint=disk,mode='constant',cval=0)

    def support_exposure(self, xy):
        """Observed steep cells overlapping the body, not measured foot contacts."""
        cells=self._support_cells(xy)
        if cells is None or not self.known[cells].all() or not np.isfinite(self.slope[cells]).all():
            return None
        steep=self.slope[cells]>math.tan(math.radians(self.MAX_SLOPE_DEG))
        return set(zip(cells[0][steep].tolist(),cells[1][steep].tolist())),len(cells[0])

    def support_exit_check(self, points, inherited, require_clear_end=True):
        """Allow a bounded departure from existing steep overlap, not entry.

        Normal planning remains unchanged. Every footprint must be observed;
        newly swept cells must be fresh. Existing steep overlap may only shrink.
        The 25-degree ascent preference is not applied to this short escape;
        actual centre terrain and travel grades still stay within 35 degrees.
        """
        if not points or not inherited:return 'no partial steep overlap to leave',{}
        first=self._support_cells(points[0])
        if first is None:return 'current footprint outside map',{}
        original=set(zip(first[0].tolist(),first[1].tolist()))
        if len(inherited)/len(original)>.25:return 'too much of the footprint overlaps steep terrain',{}
        previous=points[0];remaining=set(inherited);counts=[]
        start=self.world_to_cell(*previous)
        depth=self.no_go_depth[start] if start is not None else 0.
        for point in points:
            cell=self.world_to_cell(*point)
            if cell is None or not self.known[cell] or not self.fresh[cell]:
                return 'backout centre needs fresh observed ground',{}
            if self.hazard[cell] or self.obstacle_distance[cell]<self.footprint_radius:
                return 'obstacle inside backout footprint',{}
            if self.slope[cell]>math.tan(math.radians(self.MAX_SLOPE_DEG)):
                return 'backout centre would enter terrain above 35 degrees',{}
            if self.no_go_depth[cell]>depth+1e-6:return 'backout would enter a learned exclusion',{}
            depth=self.no_go_depth[cell]
            detail=self.grade_reason(previous,point,uphill_limit=self.MAX_SLOPE_DEG,side_limit=self.MAX_SLOPE_DEG)
            if detail:return detail,{}
            cells=self._support_cells(point)
            if cells is None or not self.known[cells].all():return 'unobserved ground in backout footprint',{}
            if not np.isfinite(self.slope[cells]).all():return 'unmeasured backout slope',{}
            footprint=set(zip(cells[0].tolist(),cells[1].tolist()))
            entered=footprint-original
            if any(not self.fresh[p] for p in entered):return 'newly swept backout ground is stale',{}
            steep=self.slope[cells]>math.tan(math.radians(self.MAX_SLOPE_DEG))
            exposed=set(zip(cells[0][steep].tolist(),cells[1][steep].tolist()))
            if not exposed.issubset(remaining):return 'backout increases or re-enters cliff exposure',{}
            remaining=exposed;counts.append(len(exposed));previous=point
        if require_clear_end and remaining:return 'backout endpoint still overlaps steep terrain',{}
        return '',dict(initial_steep_cells=len(inherited),final_steep_cells=len(remaining),
                       peak_steep_cells=max(counts),distance_m=sum(math.dist(a,b) for a,b in zip(points,points[1:])))

    def edge_cost(self, cell, dr, dc, previous_heading, profile='balanced'):
        norm=math.hypot(dr,dc)
        if norm<1e-9: return 0.
        heading=self.heading_bin(math.atan2(dc,dr))
        if not self.direction_allowed(cell,heading): return math.inf
        along=(self.gx[cell]*dr+self.gy[cell]*dc)/norm
        across=abs(-self.gx[cell]*dc+self.gy[cell]*dr)/norm
        failures=float(self.failures[cell[0],cell[1],heading])
        turn=min((heading-previous_heading)%8,(previous_heading-heading)%8)
        untested=.3 if self.successes[cell[0],cell[1],heading]==0 else 0.
        # A 34-degree climb is not merely a slightly more costly 20-degree
        # climb. Price sustained steep exposure across the entire mission route.
        # These are planning preferences below the unchanged 35-degree ceiling,
        # not new claims about traction or measured policy limits.
        grade=math.degrees(math.atan(along))
        side=math.degrees(math.atan(across))
        if grade>self.uphill_limit_deg+1e-6 or side>self.side_limit_deg+1e-6:return math.inf
        uphill=12.*(max(0.,grade-15.)/20.)**2
        downhill=4.*(max(0.,-grade-20.)/15.)**2
        sidehill=8.*(side/35.)**2
        multiplier={'direct':.35,'balanced':1.,'gentle':2.5}[profile]
        uphill*=multiplier
        sidehill*=math.sqrt(multiplier)
        # Prefer room for the whole robot beside a cliff, without painting an
        # extra forbidden band around every steep tile. Local motion still
        # checks the actual swept footprint before authorizing travel.
        edge_room=24.*max(0.,1.-self.steep_distance[cell]/(self.footprint_radius+.3))**2
        slope_ratio=min(1.,float(self.slope[cell])/math.tan(math.radians(self.MAX_SLOPE_DEG)))
        turning=.8*turn*(1.+2.*slope_ratio**2)
        return norm*(float(self.cost[cell])+2.*abs(along)+4.*across+uphill+downhill+sidehill+edge_room+6.*failures+untested)+turning

    def directional_costs(self, profile='balanced', uphill_limit=None, offsets=None):
        """Vectorized A* costs; avoid recomputing trigonometry for every edge."""
        offsets=offsets or ((1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1))
        values=np.empty((*self.known.shape,len(offsets)),float)
        multiplier={'direct':.35,'balanced':1.,'gentle':2.5}[profile]
        uphill_limit=self.uphill_limit_deg if uphill_limit is None else uphill_limit
        room=24.*np.maximum(0.,1.-self.steep_distance/(self.footprint_radius+.3))**2
        for index,(dr,dc) in enumerate(offsets):
            heading=self.heading_bin(math.atan2(dc,dr))
            norm=math.hypot(dr,dc)
            along=(self.gx*dr+self.gy*dc)/norm
            across=np.abs(-self.gx*dc+self.gy*dr)/norm
            grade=np.degrees(np.arctan(along));side=np.degrees(np.arctan(across))
            cost=(self.cost.astype(float)+2.*np.abs(along)+4.*across
                  +multiplier*12.*(np.maximum(0.,grade-15.)/20.)**2
                  +4.*(np.maximum(0.,-grade-20.)/15.)**2
                  +math.sqrt(multiplier)*8.*(side/35.)**2+room
                  +6.*self.failures[:,:,heading]+.3*(self.successes[:,:,heading]==0))
            permitted=((self.failures[:,:,heading]<2)&~self.approach_blocked[:,:,heading]
                       &(grade<=uphill_limit+1e-6)&(side<=self.side_limit_deg+1e-6))
            values[:,:,index]=np.where(permitted,norm*cost,np.inf)
        turn=.8*(1.+2.*np.minimum(1.,self.slope/math.tan(math.radians(self.MAX_SLOPE_DEG)))**2)
        return values,turn

    def route_profile(self, path):
        """Describe the whole selected route, not just the next checkpoint."""
        length=ascent=steep_length=0.;max_up=max_side=max_slope=max_body=0.
        for a,b in zip(path,path[1:]):
            delta=np.subtract(b,a);distance=float(np.linalg.norm(delta))
            if distance<1e-9: continue
            body=self.max_support_slope(a,b)
            if math.isfinite(body):max_body=max(max_body,body)
            steps=max(1,math.ceil(distance/self.cell_size))
            for t in np.linspace(0,1,steps+1)[1:]:
                cell=self.world_to_cell(*(np.asarray(a)+t*delta))
                if cell is None or not self.known[cell]: continue
                along=(self.gx[cell]*delta[0]+self.gy[cell]*delta[1])/distance
                across=abs(-self.gx[cell]*delta[1]+self.gy[cell]*delta[0])/distance
                grade=math.degrees(math.atan(along))
                slope=math.degrees(math.atan(self.slope[cell]))
                length+=distance/steps;ascent+=max(0.,along)*distance/steps
                if grade>25.: steep_length+=distance/steps
                max_up=max(max_up,grade);max_side=max(max_side,math.degrees(math.atan(across)))
                max_slope=max(max_slope,slope)
        return dict(length_m=length,ascent_m=ascent,uphill_over_25deg_m=steep_length,
                    max_uphill_deg=max_up,max_side_slope_deg=max_side,max_ground_slope_deg=max_slope,
                    max_footprint_slope_deg=max_body)

    def route_cost(self, path, yaw=0., require_fresh=False):
        """Sample every edge with the same directional costs used by A*."""
        total=0.;heading=self.heading_bin(yaw)
        for a,b in zip(path,path[1:]):
            if not self.segment_clear(a,b,require_fresh=require_fresh): return math.inf
            # Cached routes must meet the same body-support requirement as a
            # newly planned route. A clear centreline cannot preserve an old
            # route after a steep edge is discovered beside it.
            if self.support_reason(a,b): return math.inf
            delta=np.subtract(b,a);length=float(np.linalg.norm(delta))
            if length<1e-9: continue
            steps=max(1,math.ceil(float(np.max(np.abs(delta)))/self.cell_size-1e-8))
            dr,dc=delta/self.cell_size/steps
            for t in np.linspace(0,1,steps+1)[1:]:
                cell=self.world_to_cell(*(np.asarray(a)+t*delta))
                total+=self.edge_cost(cell,dr,dc,heading)
                heading=self.heading_bin(math.atan2(dc,dr))
        return total

    def segment_reason(self, a, b, escape_clearance=None, check_direction=True):
        if self.segment_clear(a,b,escape_clearance,check_direction=check_direction): return 'clear'
        length=math.dist(a,b)
        for t in np.linspace(0,1,max(2,math.ceil(length/(self.cell_size*.4))+1)):
            cell=self.world_to_cell(*(np.asarray(a)+t*np.subtract(b,a)))
            if cell is None or not self.known[cell]: return 'unobserved terrain'
            if not self.fresh[cell]: return 'stale immediate observations'
            if self.hazard[cell]: return 'detected obstacle'
            if self.no_go[cell]: return 'excluded failed approach; use committed downhill exit'
            if self.slope[cell]>math.tan(math.radians(self.MAX_SLOPE_DEG)): return 'terrain exceeds slope limit'
            if self.obstacle_distance[cell]<self.footprint_radius: return 'obstacle inside robot footprint clearance'
            if self.blocked[cell] and escape_clearance is None: return 'additional obstacle planning margin'
            if check_direction and length>1e-9 and not self.direction_allowed(
                    cell,self.heading_bin(math.atan2(b[1]-a[1],b[0]-a[0]))):
                return 'approach direction excluded after traversal failure'
        return (self.grade_reason(a,b) if check_direction else '') or 'diagonal obstacle-corner clearance'

    def survey_snapshot(self):
        return dict(height=self.height.copy(),low=self.low.copy(),surface=self.surface.copy(),
                    known=self.known.copy(),blocked=self.blocked.copy(),hazard=self.hazard.copy(),
                    clearance=self.clearance_mask.copy(),slope=self.slope.copy(),gx=self.gx.copy(),gy=self.gy.copy(),
                    roughness=self.roughness.copy(),roughness_support=self.roughness_support.copy(),
                    failures=self.failures.copy(),successes=self.successes.copy(),cost=self.cost.copy(),
                    approach_blocked=self.approach_blocked.copy(),no_go=self.no_go.copy(),
                    difficulty_episodes=self.difficulty_episodes.copy(),
                    last_seen=self.last_seen.copy(),samples=self.samples.copy(),
                    bin_low=self._bin_low.copy(),bin_high=self._bin_high.copy(),bin_count=self._bin_count.copy(),
                    origin=np.asarray(self.origin),cell_size=self.cell_size,sim_time=self.now,
                    uphill_limit_deg=self.uphill_limit_deg,side_limit_deg=self.side_limit_deg,
                    failed_climb_positions=np.asarray([z['position'] for z in self.failed_climbs]).reshape(-1,2),
                    failed_climb_radii=np.asarray([z['radius_m'] for z in self.failed_climbs]),
                    footprint_radius=self.footprint_radius,clearance_m=self.clearance)

    def get_cost_grid(self):
        return self.cost.copy()

    @property
    def clearance_mask(self):
        """Planning margin around hazards, distinct from hazardous terrain."""
        steep=self.known & (self.slope>math.tan(math.radians(self.MAX_SLOPE_DEG)))
        return self.blocked & ~self.hazard & ~steep & ~self.no_go

    def get_visual_grid(self):
        return self.get_cost_grid(), self.surface.copy(), self.known.copy()

    def quality(self, position, radius=2.):
        cell = self.world_to_cell(*position)
        if cell is None:
            return dict(coverage=0., age=math.inf, mapped_coverage=0., mapped_area=0.)
        span=math.ceil(radius/self.cell_size)
        r0,r1=max(0,cell[0]-span),min(self.n_cells,cell[0]+span+1)
        c0,c1=max(0,cell[1]-span),min(self.n_cells,cell[1]+span+1)
        area=(np.arange(r0,r1)[:,None]-cell[0])**2+(np.arange(c0,c1)[None,:]-cell[1])**2 <= (radius/self.cell_size)**2
        seen=area & self.known[r0:r1,c0:c1]
        return dict(coverage=float((area & self.fresh[r0:r1,c0:c1]).sum()/area.sum()),
                    mapped_coverage=float(seen.sum()/area.sum()),
                    mapped_area=getattr(self,'_mapped_area',0.),
                    age=float(np.max(self.now-self.surface_seen[r0:r1,c0:c1][seen])) if seen.any() else math.inf)
