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
        self.gx = np.zeros(shape)
        self.gy = np.zeros(shape)
        self.slope = np.zeros(shape)
        self.roughness = np.full(shape,np.nan,np.float32)
        self.roughness_support = np.zeros(shape,np.uint8)
        self.failures = np.zeros((*shape,8),np.uint8)
        self.successes = np.zeros((*shape,8),np.uint16)
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
        self._bin_low[hit],self._bin_high[hit] = low,high
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
        self.blocked = inflated | (self.known & (self.slope > math.tan(math.radians(self.MAX_SLOPE_DEG))))
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

    def segment_clear(self, a, b, escape_clearance=None, require_fresh=True, check_direction=True):
        length = np.linalg.norm(np.subtract(b, a))
        def allowed_cell(cell):
            allowed = self.traversable(cell,require_fresh=require_fresh)
            if not allowed and escape_clearance is not None and cell is not None:
                allowed = (self.known[cell] and self.fresh[cell] and not self.hazard[cell]
                           and self.obstacle_distance[cell] <= self.footprint_radius + self.clearance
                           and self.obstacle_distance[cell] >= max(self.footprint_radius, escape_clearance) - 1e-6
                           and self.slope[cell] <= math.tan(math.radians(self.MAX_SLOPE_DEG)))
            return allowed
        heading=self.heading_bin(math.atan2(b[1]-a[1],b[0]-a[0])) if length>1e-9 else None
        previous = None
        for t in np.linspace(0, 1, max(2, math.ceil(length / (self.cell_size * .4)) + 1)):
            p = np.asarray(a) + t * np.subtract(b, a)
            cell = self.world_to_cell(*p)
            if not allowed_cell(cell) or (check_direction and heading is not None and not self.direction_allowed(cell,heading)):
                return False
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
        return cell is not None and self.failures[cell[0],cell[1],heading]<2

    def edge_cost(self, cell, dr, dc, previous_heading):
        norm=math.hypot(dr,dc)
        if norm<1e-9: return 0.
        heading=self.heading_bin(math.atan2(dc,dr))
        if not self.direction_allowed(cell,heading): return math.inf
        along=(self.gx[cell]*dr+self.gy[cell]*dc)/norm
        across=abs(-self.gx[cell]*dc+self.gy[cell]*dr)/norm
        failures=float(self.failures[cell[0],cell[1],heading])
        turn=min((heading-previous_heading)%8,(previous_heading-heading)%8)
        untested=.3 if self.successes[cell[0],cell[1],heading]==0 else 0.
        return norm*(float(self.cost[cell])+2.*abs(along)+4.*across+6.*failures+untested)+.8*turn

    def route_cost(self, path, yaw=0., require_fresh=False):
        """Sample every edge with the same directional costs used by A*."""
        total=0.;heading=self.heading_bin(yaw)
        for a,b in zip(path,path[1:]):
            if not self.segment_clear(a,b,require_fresh=require_fresh): return math.inf
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
            if self.slope[cell]>math.tan(math.radians(self.MAX_SLOPE_DEG)): return 'terrain exceeds slope limit'
            if self.obstacle_distance[cell]<self.footprint_radius: return 'obstacle inside robot footprint clearance'
            if self.blocked[cell] and escape_clearance is None: return 'additional obstacle planning margin'
            if check_direction and length>1e-9 and not self.direction_allowed(
                    cell,self.heading_bin(math.atan2(b[1]-a[1],b[0]-a[0]))):
                return 'approach direction excluded after repeated traversal failures'
        return 'diagonal obstacle-corner clearance'

    def survey_snapshot(self):
        return dict(height=self.height.copy(),low=self.low.copy(),surface=self.surface.copy(),
                    known=self.known.copy(),blocked=self.blocked.copy(),hazard=self.hazard.copy(),
                    clearance=self.clearance_mask.copy(),slope=self.slope.copy(),gx=self.gx.copy(),gy=self.gy.copy(),
                    roughness=self.roughness.copy(),roughness_support=self.roughness_support.copy(),
                    failures=self.failures.copy(),successes=self.successes.copy(),cost=self.cost.copy(),
                    last_seen=self.last_seen.copy(),samples=self.samples.copy(),
                    bin_low=self._bin_low.copy(),bin_high=self._bin_high.copy(),bin_count=self._bin_count.copy(),
                    origin=np.asarray(self.origin),cell_size=self.cell_size,sim_time=self.now,
                    footprint_radius=self.footprint_radius,clearance_m=self.clearance)

    def get_cost_grid(self):
        return self.cost.copy()

    @property
    def clearance_mask(self):
        """Planning margin around hazards, distinct from hazardous terrain."""
        steep=self.known & (self.slope>math.tan(math.radians(self.MAX_SLOPE_DEG)))
        return self.blocked & ~self.hazard & ~steep

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
