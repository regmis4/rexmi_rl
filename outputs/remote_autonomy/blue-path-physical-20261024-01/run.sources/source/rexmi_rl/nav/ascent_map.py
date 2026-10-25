"""Observed terrain with local shelf geometry and directional mission memory."""
import math

import numpy as np
from scipy import ndimage as ndi

from .observed_map import ObservedTerrainMap
from . import footprint_costs


DIRECTIONAL_EXIT_PROFILE = dict(directional_recovery_exit_contract=
    'incidental stopping during independently checked signed recovery or opposing damping; '
    'freeze original occupied failure identities, cells and bins per episode; actual exit retires permission; '
    'no new patch, growth inheritance, entry or re-entry; intentional direction bans and all physical checks retained',
    recovery_choice_contract='both signs at action boundaries; mission connection before staging before purposeful egress; '
    'terrain and path costs within tier; no descent bias; same episode deadline and tried-action memory')


class AscentTerrainMap(ObservedTerrainMap):
    def __init__(self, *args, **kwargs):
        half_length=kwargs.pop('half_length',None)
        half_width=kwargs.pop('half_width',None)
        super().__init__(*args, **kwargs)
        self.half_length=self.footprint_radius if half_length is None else float(half_length)
        self.half_width=self.footprint_radius if half_width is None else float(half_width)
        self.local_slope = np.full_like(self.slope, np.nan)
        self.local_fit_support = np.zeros_like(self.known)
        self.shelf_mask = np.zeros_like(self.known)
        self.failure_directions = np.zeros((*self.known.shape, 8), bool)
        self.failure_ids = set()
        self.failure_records = []
        footprint_costs.initialize(self)

    @staticmethod
    def diagonal_preference_cost(along, across):
        """Soft 10–30 degree offset from steepest ascent; no capability claim.

        Ground/cross-slope limits and complete swept support remain hard gates.
        This preference never adds a turn to an already committed hillside leg.
        """
        angle = np.degrees(np.arctan2(np.abs(across), along))
        departure = np.maximum(0., np.maximum(10.-angle, angle-30.))
        active = (along >= math.tan(math.radians(10.))) & (np.hypot(along, across) >= math.tan(math.radians(15.)))
        return np.where(active, 4.*(departure/20.)**2, 0.)

    def directional_costs(self, profile='balanced', uphill_limit=None, offsets=None):
        offsets = offsets or ((1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1))
        if getattr(self, 'traversal_objective', None):
            from . import traversal_objective as objective
            values = np.stack([math.hypot(dx, dy)*objective.density(self, np.s_[:, :], dx, dy)
                               for dx, dy in offsets], axis=-1)
            if uphill_limit is not None:
                for k, (dx, dy) in enumerate(offsets):
                    grade = np.degrees(np.arctan((self.gx*dx+self.gy*dy)/math.hypot(dx, dy)))
                    values[..., k] = np.where(grade <= uphill_limit+1e-6, values[..., k], np.inf)
            turns = .8*(1.+2.*np.minimum(1., self.slope/math.tan(math.radians(self.MAX_SLOPE_DEG)))**2)/(math.pi/4.)
            return values, turns
        costs, turns = super().directional_costs(profile, uphill_limit, offsets)
        if getattr(self, 'diagonal_ascent_preference', False):
            for i, (dx, dy) in enumerate(offsets):
                norm = math.hypot(dx, dy)
                along = (self.gx*dx+self.gy*dy)/norm
                across = (-self.gx*dy+self.gy*dx)/norm
                costs[..., i] += norm*self.diagonal_preference_cost(along, across)
        return costs, turns

    def edge_cost(self, cell, dr, dc, heading, profile='balanced'):
        if getattr(self, 'traversal_objective', None):
            from . import traversal_objective as objective
            if math.hypot(dr, dc) < 1e-9:
                return 0.
            # Travel only. The shared route/search contract charges the actual
            # continuous departure angle, once, at the departure position.
            return math.hypot(dr, dc)*float(objective.density(self, cell, dr, dc))
        value = super().edge_cost(cell, dr, dc, heading, profile)
        if getattr(self, 'diagonal_ascent_preference', False) and math.isfinite(value):
            norm = math.hypot(dr, dc)
            along = (self.gx[cell]*dr+self.gy[cell]*dc)/norm
            across = (-self.gx[cell]*dc+self.gy[cell]*dr)/norm
            value += norm*float(self.diagonal_preference_cost(along, across))
        return value

    def route_cost(self, path, yaw=0., require_fresh=False):
        if getattr(self, 'traversal_objective', None):
            from . import traversal_objective as objective
            return objective.route_cost(self, path, yaw, require_fresh=require_fresh)
        return super().route_cost(path, yaw, require_fresh)

    def oriented_cells(self, xy, yaw):
        """Measured body/leg bounding box plus the same 10 cm body reserve.

        Circle-based global planning is retained. This resolves a blocked circle
        during a checked escape, without declaring an actual obstacle traversable.
        """
        reach=math.hypot(self.half_length,self.half_width)+self.cell_size
        lo=self.world_to_cell(xy[0]-reach,xy[1]-reach)
        hi=self.world_to_cell(xy[0]+reach,xy[1]+reach)
        if lo is None or hi is None:return None
        rr,cc=np.meshgrid(np.arange(lo[0],hi[0]+1),np.arange(lo[1],hi[1]+1),indexing='ij')
        x=self.origin[0]+(rr+.5)*self.cell_size-self.world_size/2-xy[0]
        y=self.origin[1]+(cc+.5)*self.cell_size-self.world_size/2-xy[1]
        longitudinal=x*math.cos(yaw)+y*math.sin(yaw)
        lateral=-x*math.sin(yaw)+y*math.cos(yaw)
        # A cell's half-diagonal covers every cell square, not just its centre.
        mask=(np.abs(longitudinal)<=self.half_length+self.cell_size*.71)&(np.abs(lateral)<=self.half_width+self.cell_size*.71)
        return rr[mask],cc[mask]

    def oriented_support_reason(self, xy, yaw, fresh=False):
        cells=self.oriented_cells(xy,yaw)
        if cells is None or not self.known[cells].all():return 'unobserved oriented body support'
        if self.hazard[cells].any():return 'obstacle in oriented body envelope'
        if not np.isfinite(self.slope[cells]).all() or (self.slope[cells]>math.tan(math.radians(35))).any():
            return 'oriented support exceeds ground slope limit'
        if fresh and not self.fresh[cells].all():return 'oriented body needs fresh observations'
        return ''

    def corridor_evidence(self, start, end, yaw, now):
        """Copy observed evidence at rejection time; never changes admissibility.

        A later map may clear or move a hazard. Preserve the local cells and
        their observation times instead of explaining an old decision with a
        newer reconstruction. No simulator terrain queries are used here.
        """
        margin=self.footprint_radius+self.clearance
        radius=margin+2*self.cell_size
        limits=np.array([np.minimum(start,end)-radius,np.maximum(start,end)+radius])
        indices=np.floor((limits-self.origin+self.world_size/2)/self.cell_size).astype(int)
        lower=np.maximum(0,indices[0]);upper=np.minimum(self.n_cells,indices[1]+1)
        selection=tuple(slice(int(a),int(b)) for a,b in zip(lower,upper))
        def values(array):
            value=np.asarray(array)
            return np.where(np.isfinite(value),value,None).tolist()
        fields=('height','low','surface','last_seen','surface_seen','samples','known','fresh',
                'blocked','hazard','slope','gx','gy','roughness','residual','obstacle_distance',
                'approach_blocked','no_go','no_go_depth')
        patch={name:values(getattr(self,name)[selection]) for name in fields}
        hazards=np.argwhere(self.hazard[selection])+lower
        samples=[]
        for fraction in np.linspace(0.,1.,max(2,math.ceil(math.dist(start,end)/(self.cell_size/4))+1)):
            position=tuple(np.asarray(start)+fraction*np.subtract(end,start))
            cell=self.world_to_cell(*position)
            sample=dict(position=list(position),cell=list(cell) if cell else None,
                oriented_support_reason=self.oriented_support_reason(position,yaw,fresh=True))
            if cell is not None:
                sample.update(known=bool(self.known[cell]),fresh=bool(self.fresh[cell]),
                    blocked=bool(self.blocked[cell]),hazard=bool(self.hazard[cell]),
                    obstacle_distance_m=values(self.obstacle_distance[cell]),
                    observation_time=values(self.last_seen[cell]),
                    footprint_inclination_deg=values(self.max_support_slope(position)))
                if len(hazards):
                    nearest=hazards[np.argmin(np.sum((hazards-np.asarray(cell))**2,axis=1))]
                    sample['nearest_hazard_cell']=nearest.tolist()
                    sample['nearest_hazard_world']=list(self.cell_to_world(*nearest))
            samples.append(sample)
        return dict(kind='observed_map_patch',time=float(now),map_time=float(self.now),revision=int(self.revision),
            start=list(start),end=list(end),body_yaw=float(yaw),required_circle_clearance_m=margin,
            half_length_m=self.half_length,half_width_m=self.half_width,
            cell_size_m=self.cell_size,grid_origin=list(self.origin),world_size_m=self.world_size,
            patch_first_cell=lower.tolist(),patch_shape=(upper-lower).tolist(),fields=patch,samples=samples,
            support_reason=self.support_reason(start,end),grade_reason=self.grade_reason(start,end),
            limitation='Observed geometry at rejection; not a physical collision or terrain ground-truth label.')

    def rebuild(self, now):
        super().rebuild(now)
        measured = np.isfinite(self.height)
        # A 1 m fit complements the baseline's 2.2 m ground plane. Only well
        # supported fits can assert a shelf; obstacles remain independently hard.
        z = np.where(measured, .5*(self.height+self.low), 0.)
        d = np.arange(-2, 3)*self.cell_size
        powers = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
        def moments(values, pairs):
            return [ndi.correlate1d(ndi.correlate1d(values, d**a, axis=0, mode='constant'),
                                    d**b, axis=1, mode='constant') for a, b in pairs]
        sums = moments(measured.astype(float), powers)
        rhs = np.stack(moments(z, powers[:3]), axis=-1)
        matrix = np.empty((*z.shape, 3, 3))
        matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2] = sums[:3]
        matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2] = sums[1], sums[3], sums[4]
        matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2] = sums[2], sums[4], sums[5]
        active = (sums[0] >= 12) & self.known
        fit = np.zeros_like(rhs)
        fit[active] = np.linalg.solve(matrix[active]+np.eye(3)*1e-6, rhs[active, ..., None])[..., 0]
        residual = np.abs(z-fit[..., 0])
        self.local_fit_support = active & (residual < .08)
        self.local_slope = np.where(self.local_fit_support, np.hypot(fit[..., 1], fit[..., 2]), np.nan)
        squared = moments(z*z, [(0, 0)])[0]
        local_rms = np.sqrt(np.maximum(0., (squared-np.sum(fit*rhs, axis=-1))/np.maximum(sums[0], 1)))
        self.local_fit_support &= (local_rms <= .04) & (sums[0] >= 20)
        self.local_slope[~self.local_fit_support] = np.nan
        # The local fit resolves shelves, not a second hard obstacle classifier.
        # Overall ground inclination and discontinuities retain the established
        # map's 35-degree/support/hazard checks. Short rock facets must not turn
        # into a new, inflated cliff classification.
        flat = (self.known & self.local_fit_support & ~self.blocked
                & (self.local_slope <= math.tan(math.radians(5)))
                & np.where(np.isfinite(self.roughness), self.roughness <= .06, local_rms <= .04))
        space = ndi.distance_transform_edt(np.pad(flat, 1), sampling=self.cell_size)[1:-1, 1:-1]
        self.shelf_mask = flat & (space >= self.footprint_radius+self.clearance)
        footprint_costs.rebuild(self)

    def segment_clear(self, a, b, escape_clearance=None, require_fresh=True, check_direction=True, no_go_exit=False,
                      *, observation_support=None):
        first = self.world_to_cell(*a)
        departing = no_go_exit and first is not None and self.no_go[first]
        if not departing:
            return super().segment_clear(a, b, escape_clearance, require_fresh, check_direction, no_go_exit,
                                         observation_support=observation_support)
        if check_direction and self.grade_reason(a, b):
            return False
        if not super().segment_clear(a, b, escape_clearance, require_fresh, False, True,
                                     observation_support=observation_support):
            return False
        depth = self.no_go_depth[first]
        for t in np.linspace(0, 1, max(2, math.ceil(math.dist(a, b)/(self.cell_size*.4))+1)):
            point = np.asarray(a)+t*np.subtract(b, a)
            cell = self.world_to_cell(*point)
            if self.no_go_depth[cell] > depth+1e-6:
                return False
            depth = self.no_go_depth[cell]
            if not self.no_go[cell] and check_direction:
                heading = self.heading_bin(math.atan2(b[1]-a[1], b[0]-a[0]))
                if not self.direction_allowed(cell, heading):
                    return False
        return True

    def directional_exit_reason(self, a, b, permission=None):
        """Check outward departure from an occupied directional exclusion.

        This is only an incidental recovery-stopping exception. It does not
        clear evidence, authorize a forward retry, or relax physical geometry.
        Every crossed centre cell and diagonal bridge retains a non-increasing
        depth in the failed-heading mask; an outside start cannot enter it.
        """
        if permission is not None:
            return permission.reason(self, a, b)
        first = self.world_to_cell(*a)
        length = math.dist(a,b)
        if first is None:return 'directional escape starts outside observed map'
        if length < 1e-9:return ''
        heading = self.heading_bin(math.atan2(b[1]-a[1],b[0]-a[0]))
        excluded = self.approach_blocked[...,heading] | (self.failures[...,heading] >= 2)
        departing = bool(excluded[first])
        # Depth is derived from the full mask, never a cropped diagnostic patch.
        depth_map = ndi.distance_transform_edt(excluded,sampling=self.cell_size) if departing else None
        depth = float(depth_map[first]) if departing else 0.
        exited = not departing
        previous = None
        for fraction in np.linspace(0.,1.,max(2,math.ceil(length/(self.cell_size*.25))+1)):
            cell = self.world_to_cell(*(np.asarray(a)+fraction*np.subtract(b,a)))
            if cell is None:return 'directional escape leaves observed map'
            touched = [cell]
            if previous is not None and cell[0] != previous[0] and cell[1] != previous[1]:
                touched.extend(((cell[0],previous[1]),(previous[0],cell[1])))
            for item in touched:
                if excluded[item] and (exited or depth_map[item] > depth+1e-6):
                    return 'directional escape enters or deepens a failed patch'
            if not excluded[cell]:exited = True
            depth = float(depth_map[cell]) if depth_map is not None else 0.
            previous = cell
        return ''

    def drift_segment_reason(self, a, b, require_fresh=True, no_go_exit=False, directional_exit=False,
                             *, observation_support=None):
        """Physical stopping envelope for incidental motion, not a new approach.

        Keep the footprint's overall slope boundary and approach exclusions.
        Uphill/cross-slope planning grades belong to the commanded approach,
        not the instantaneous direction of gait sway. An occupied exclusion
        permits only checked outward departure, never entry or re-entry.
        """
        observed = self.fresh if observation_support is None else observation_support
        support = self.support_reason(a, b)
        if support:
            return 'support: '+support
        first = self.world_to_cell(*a)
        departing = bool(no_go_exit and first is not None and self.no_go[first])
        depth = self.no_go_depth[first] if first is not None else 0.
        length = math.dist(a, b)
        heading = self.heading_bin(math.atan2(b[1]-a[1], b[0]-a[0])) if length > 1e-9 else None
        permission = None if isinstance(directional_exit, bool) else directional_exit
        leaving_direction = directional_exit and not self.directional_exit_reason(a,b,permission)
        previous = None
        for t in np.linspace(0, 1, max(2, math.ceil(length/(self.cell_size*.4))+1)):
            cell = self.world_to_cell(*(np.asarray(a)+t*np.subtract(b, a)))
            if cell is None:
                return 'geometry: drift leaves map'
            touched = [cell]
            if previous is not None and cell[0] != previous[0] and cell[1] != previous[1]:
                touched += [(cell[0], previous[1]), (previous[0], cell[1])]
            for item in touched:
                if not self.known[item]:
                    return 'observation: unknown drift corridor'
                if require_fresh and not observed[item]:
                    return 'freshness: stale drift corridor'
                if self.hazard[item] or self.obstacle_distance[item] < self.footprint_radius+self.clearance:
                    return 'obstacle: insufficient drift clearance'
                if self.no_go[item]:
                    if not departing or self.no_go_depth[item] > depth+1e-6:
                        return 'exclusion: drift would enter or deepen a no-go patch'
                elif heading is not None and not self.direction_allowed(item, heading) and not leaving_direction:
                    return 'exclusion: drift would reuse a failed approach direction'
            depth = self.no_go_depth[cell]
            previous = cell
        if not self.segment_clear(a, b, require_fresh=require_fresh,
                                  check_direction=False, no_go_exit=departing,
                                  observation_support=observation_support):
            return 'geometry: drift corridor is blocked'
        return ''

    def mark_failure(self, attempt_id, xy, heading, now, reason):
        if attempt_id in self.failure_ids:
            return False
        self.failure_ids.add(attempt_id)
        centre = self.world_to_cell(*xy)
        if centre is None:
            return False
        radius = max(.35, self.footprint_radius+.1)
        count = math.ceil(radius/self.cell_size)
        direction = self.heading_bin(heading)
        for dr in range(-count, count+1):
            for dc in range(-count, count+1):
                r, c = centre[0]+dr, centre[1]+dc
                if not (0 <= r < self.n_cells and 0 <= c < self.n_cells):
                    continue
                if math.hypot(dr, dc)*self.cell_size > radius:
                    continue
                # Neighbouring bins prevent nominal yaw tweaks from bypassing
                # a failed approach. Only *distinct* unblocked directions add
                # evidence for an all-direction exclusion.
                distinct = not self.approach_blocked[r, c, direction]
                if distinct:
                    self.failure_directions[r, c, direction] = True
                for h in ((direction-1) % 8, direction, (direction+1) % 8):
                    self.approach_blocked[r, c, h] = True
                self.difficulty_episodes[r, c] = min(255, int(self.failure_directions[r, c].sum()))
                if self.difficulty_episodes[r, c] >= 2:
                    self.no_go[r, c] = True
        self.record_traversal([xy], heading, failed=True)
        self.no_go_depth = ndi.distance_transform_edt(self.no_go, sampling=self.cell_size)
        self.blocked |= self.no_go
        self.cost[self.no_go] = 20.
        self.failure_records.append(dict(attempt_id=attempt_id, position=list(xy), radius_m=radius,
                                         heading=heading, time=now, reason=reason))
        self.revision += 1
        self.traversal_revision += 1
        return True

    def support_exit_check(self, points, inherited, require_clear_end=True):
        reason,metrics=super().support_exit_check(points,inherited,require_clear_end)
        if reason:return reason,metrics
        # The baseline partial-cliff backout predates directional mission bans.
        # Its geometry exception must not bypass a physically failed approach.
        # Departure from an occupied all-direction patch has already passed
        # the monotonically decreasing depth check in the shared geometry.
        for a,b in zip(points,points[1:]):
            if math.dist(a,b)<1e-9:continue
            heading=self.heading_bin(math.atan2(b[1]-a[1],b[0]-a[0]))
            for t in np.linspace(0,1,max(2,math.ceil(math.dist(a,b)/(self.cell_size*.4))+1)):
                cell=self.world_to_cell(*(np.asarray(a)+t*np.subtract(b,a)))
                if not self.no_go[cell] and not self.direction_allowed(cell,heading):
                    return 'backout would reuse a failed approach direction',{}
        return '',metrics

    def turn_support_reason(self, xy):
        """Use the follower's drift reserve before planning a hillside turn."""
        reserve=.25 if self.max_support_slope(xy,reserve=.25)>self.MAX_SLOPE_DEG else 0.
        return self.support_reason(xy,reserve=reserve)

    def shelf_candidates(self, xy, goal, limit=4):
        cells = np.argwhere(self.shelf_mask)
        if not len(cells):
            return []
        points = np.asarray([self.cell_to_world(*c) for c in cells])
        distance = np.linalg.norm(points-np.asarray(xy), axis=1)
        score = distance+np.linalg.norm(points-np.asarray(goal), axis=1)
        chosen = []
        for i in np.argsort(score):
            if not 1.5 <= distance[i] <= 7.:
                continue
            p = tuple(points[i])
            if all(math.dist(p, old) >= 1.2 for old in chosen):
                chosen.append(p)
            if len(chosen) >= limit:
                break
        return chosen

    def survey_snapshot(self):
        result = super().survey_snapshot()
        result.update(local_slope=self.local_slope.copy(), shelf_mask=self.shelf_mask.copy(),
                      failure_directions=self.failure_directions.copy(),
                      half_length=self.half_length,half_width=self.half_width)
        result.update({name: getattr(self, name).copy() for name in footprint_costs.FIELDS})
        return result

    def quality(self, position, radius=2.):
        result=super().quality(position,radius)
        if not getattr(self,'terrain_prior',None):return result
        # An imported survey has no timestamp on this mission's clock. Report
        # the age of actual mission scans separately from retained geometry.
        cell=self.world_to_cell(*position)
        if cell is None:return result
        span=math.ceil(radius/self.cell_size)
        r0,r1=max(0,cell[0]-span),min(self.n_cells,cell[0]+span+1)
        c0,c1=max(0,cell[1]-span),min(self.n_cells,cell[1]+span+1)
        area=(np.arange(r0,r1)[:,None]-cell[0])**2+(np.arange(c0,c1)[None,:]-cell[1])**2<=(radius/self.cell_size)**2
        times=self.surface_seen[r0:r1,c0:c1]
        observed=area & np.isfinite(times) & self.known[r0:r1,c0:c1]
        result['age']=float(np.max(self.now-times[observed])) if observed.any() else math.inf
        result['prior_kind']=self.terrain_prior['kind']
        return result
