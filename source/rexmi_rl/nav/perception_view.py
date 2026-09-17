"""Optional Isaac Sim perception overlay. Call only from the simulation thread.

Pure geometry helpers are deliberately independent of Isaac/torch for CPU tests.
All overlays are non-colliding USD geometry in the session layer.
"""
from __future__ import annotations

import time
import numpy as np


def filter_hits(points, origin, radius=30.0, limit=6000):
    """Finite, in-range world-frame returns; misses and self-hits are excluded."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    origin = np.asarray(origin, dtype=np.float32)
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    distance = np.linalg.norm(points - origin, axis=1)
    points = points[(distance > 0.20) & (distance <= radius)]
    return points[::max(1, int(np.ceil(len(points) / limit)))].copy()


def cost_tiles(cost, height, observed, origin, cell_size, robot_xy,
               radius=12.0, limit=4000, clearance_mask=None):
    """Small quads at measured cell heights, never on unobserved cells.

    Row is world X, column is world Y (the occupancy map's convention).
    Costs are planner costs, not probabilities or proof of safe traversal.
    """
    cost, height, observed = map(np.asarray, (cost, height, observed))
    valid = observed & np.isfinite(height) & (np.abs(height) < 1e5)
    rows, cols = np.nonzero(valid)
    xy = np.column_stack((rows, cols)).astype(np.float32)
    xy = np.asarray(origin) + (xy + 0.5 - np.asarray(cost.shape) / 2) * cell_size
    within = np.linalg.norm(xy - robot_xy, axis=1) <= radius
    rows, cols, xy = rows[within], cols[within], xy[within]
    # Keep nearer cells rather than allocating unbounded stage geometry.
    order = np.argsort(np.linalg.norm(xy - robot_xy, axis=1))[:limit]
    rows, cols, xy = rows[order], cols[order], xy[order]
    centres = np.column_stack((xy, height[rows, cols] + 0.045)).astype(np.float32)
    half = cell_size * 0.43
    offsets = np.array([[-half,-half,0],[half,-half,0],
                        [half,half,0],[-half,half,0]], dtype=np.float32)
    vertices = (centres[:, None, :] + offsets).reshape(-1, 3)
    values = cost[rows, cols]
    colors = np.tile([0.08, 0.65, 0.65], (len(rows), 1))
    colors[values >= 2] = [0.95, 0.64, 0.12]
    colors[values >= 6] = [1.0, 0.29, 0.12]
    colors[(values >= 20) | ~np.isfinite(values)] = [1.0, 0.0, 0.0]
    if clearance_mask is not None:
        colors[np.asarray(clearance_mask)[rows,cols]] = [.55,.46,.26]
    return vertices, colors.astype(np.float32)


LAYER_NAMES=('terrain','slope','roughness','traversal')


def survey_colors(cost, observed, clearance=None, layer='terrain', slope=None,
                  roughness=None, failures=None, successes=None):
    """Shared viewport/dashboard palette. Diagnostic layers do not alter costs."""
    colors=np.full((*cost.shape,3),.32,np.float32)
    if layer=='terrain':
        colors[:]=[.08,.65,.65]
        colors[cost>=2]=[.95,.64,.12]
        colors[cost>=6]=[1.,.29,.12]
        legend='Teal: low cost | yellow/orange: higher cost'
    elif layer in ('slope','roughness'):
        values=np.degrees(np.arctan(slope)) if layer=='slope' else roughness*100
        maximum=50. if layer=='slope' else 10.
        t=np.clip(np.nan_to_num(values)/maximum,0,1)[...,None]
        colors[:]=(1-t)*np.array([.08,.65,.65])+t*np.array([1.,.65,.08])
        colors[~np.isfinite(values)]=.32
        legend=('Slope: teal 0° → gold 50°' if layer=='slope' else
                'Roughness RMS: teal 0 → gold 10 cm | grey: insufficient samples')
    else:
        good=np.max(successes,axis=-1)>0
        bad=np.max(failures,axis=-1)
        colors[good]=[.12,.75,.3]
        colors[bad==1]=[1.,.55,.12]
        colors[bad>=2]=[.65,.25,.85]
        legend='Grey: untested | green: successful progress | orange: failed | purple: direction excluded'
    buffer=np.zeros(cost.shape,bool) if clearance is None else clearance
    colors[(cost>=20)&~buffer]=[1.,0.,0.]
    colors[buffer]=[.55,.46,.26]
    colors[~observed]=[.33,.33,.47]
    return colors,legend+' | red: hazard/slope limit | cream: planning margin'


def survey_tiles(height, observed, colors, origin, cell_size, robot_xy, radius=12.):
    """Whole survey: 20 cm nearby, 40 cm distant, no unknown-cell infill.

    Aggregate only completely measured 2x2 groups. Use the most hazardous
    colour and highest surface in a group, retaining isolated red obstacles.
    """
    valid=observed & np.isfinite(height)
    n,m=height.shape
    groups=[]
    # Vectorized full-grid fine cells; fully observed distant groups replace 4.
    rr,cc=np.indices(height.shape)
    xy=np.stack((origin[0]+(rr+.5-n/2)*cell_size,
                 origin[1]+(cc+.5-m/2)*cell_size),axis=-1)
    far=np.linalg.norm(xy-np.asarray(robot_xy),axis=-1)>radius
    coarse=np.zeros_like(valid)
    if n%2==0 and m%2==0:
        blocks=lambda a:a.reshape(n//2,2,m//2,2).transpose(0,2,1,3).reshape(n//2,m//2,4)
        eligible=blocks(valid & far).all(axis=-1)
        coarse=np.repeat(np.repeat(eligible,2,axis=0),2,axis=1)
        r,c=np.nonzero(eligible)
        if len(r):
            x=origin[0]+(2*r+1-n/2)*cell_size
            y=origin[1]+(2*c+1-m/2)*cell_size
            z=blocks(height)[r,c].max(axis=-1)
            block_colors=colors.reshape(n//2,2,m//2,2,3).transpose(0,2,1,3,4).reshape(n//2,m//2,4,3)[r,c]
            # Red hazards outrank cream; retain other high diagnostic values.
            red=(block_colors[:,:,0]>.99)&(block_colors[:,:,1]<.05)
            cream=np.all(np.isclose(block_colors,[.55,.46,.26]),axis=-1)
            excluded=np.all(np.isclose(block_colors,[.65,.25,.85]),axis=-1)
            failed=np.all(np.isclose(block_colors,[1.,.55,.12]),axis=-1)
            success=np.all(np.isclose(block_colors,[.12,.75,.3]),axis=-1)
            rank=block_colors[:,:,0]+red*10+cream*5+excluded*3+failed*2+success
            chosen=block_colors[np.arange(len(r)),rank.argmax(axis=-1)]
            groups.append((np.c_[x,y,z+.045],chosen,cell_size*.86))
    r,c=np.nonzero(valid & ~coarse)
    groups.append((np.c_[xy[r,c],height[r,c]+.045],colors[r,c],cell_size*.43))
    vertices=[];palette=[]
    for centres,col,half in groups:
        offsets=np.array([[-half,-half,0],[half,-half,0],[half,half,0],[-half,half,0]])
        vertices.append((centres[:,None,:]+offsets).reshape(-1,3));palette.append(col)
    return np.concatenate(vertices).astype(np.float32),np.concatenate(palette).astype(np.float32)


def drape_path(path, height, observed, origin, cell_size):
    """Separate line segments only where BOTH endpoints have observed heights."""
    xy = np.asarray(path, dtype=np.float32).reshape(-1, 2)
    if len(xy) < 2:
        return np.empty((0, 3), np.float32)
    finite = np.isfinite(xy).all(axis=1)
    cells = np.floor((np.where(np.isfinite(xy), xy, 0) - origin) / cell_size
                     + np.asarray(height.shape) / 2).astype(int)
    inside = finite & (cells >= 0).all(axis=1) & (cells < height.shape).all(axis=1)
    valid = np.zeros(len(xy), bool)
    z = np.zeros(len(xy), np.float32)
    ids = np.flatnonzero(inside)
    r, c = cells[ids].T
    valid[ids] = observed[r, c] & np.isfinite(height[r, c]) & (np.abs(height[r, c]) < 1e5)
    z[ids] = height[r, c] + 0.12
    points = np.column_stack((xy, z))
    pairs = np.flatnonzero(valid[:-1] & valid[1:])
    return np.stack((points[pairs], points[pairs + 1]), axis=1).reshape(-1, 3)


class StableTilePool:
    """Keep a world cell in the same mesh face while it remains displayed."""
    def __init__(self, capacity=4000):
        self.capacity=capacity
        self.slots={}

    def update(self, vertices, colors):
        quads=np.asarray(vertices,dtype=np.float32).reshape(-1,4,3)
        keys=[tuple(x) for x in np.round(quads.mean(axis=1)[:,:2],4)]
        wanted=set(keys)
        self.slots={key:slot for key,slot in self.slots.items() if key in wanted}
        free=iter(sorted(set(range(self.capacity))-set(self.slots.values())))
        points=np.zeros((self.capacity,4,3),np.float32)
        rgba=np.zeros((self.capacity,3),np.float32)
        opacity=np.zeros(self.capacity,np.float32)
        for key,quad,color in zip(keys,quads,colors):
            if key not in self.slots:
                self.slots[key]=next(free)
            slot=self.slots[key]
            points[slot],rgba[slot],opacity[slot]=quad,color,1.
        return points.reshape(-1,3),rgba,opacity


class PerceptionView:
    """Live returns + measured terrain cost tiles + observed portions of A* path."""
    ROOT = '/RexmiPerception'

    def __init__(self, nav, hz=5.0):
        import omni.usd
        from pxr import Sdf, Usd, UsdGeom
        self.nav = nav
        self.stage = omni.usd.get_context().get_stage()
        if self.stage.GetPrimAtPath(self.ROOT):
            raise RuntimeError(f'{self.ROOT} already exists; refusing to overwrite it')
        self._Usd, self._Geom, self._Sdf = Usd, UsdGeom, Sdf
        self.interval = 1.0 / hz
        self.next_update = 0.0
        self.enabled = True
        self.window = None
        self.show = dict(lidar=True, costs=True, path=True, rays=False)
        self.tile_chunks={}
        self.layer='terrain'
        self.render_ms=0.
        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            UsdGeom.Xform.Define(self.stage, self.ROOT)
            self.points = UsdGeom.Points.Define(self.stage, self.ROOT + '/LiveLidar')
            self.points.CreatePointsAttr([(0.0, 0.0, 0.0)] * 6000)
            self.points.CreateWidthsAttr([0.0] * 6000)
            self.points.SetWidthsInterpolation('vertex')
            self.points.CreateDisplayColorAttr([(0.15, 0.90, 1.0)])
            self.tiles = UsdGeom.Xform.Define(self.stage,self.ROOT+'/ObservedCosts')
            self.route = self._curve('/ObservedRoute', (1.0, 0.04, 0.06), 0.065)
            self.checkpoint_marker = self._curve('/Checkpoint', (1.0, 0.82, 0.1), 0.055)
            self.target_marker = self._curve('/SteeringTarget', (1.0, 1.0, 1.0), 0.035)
            self.rays = self._curve('/SampledRays', (0.25, 0.65, 1.0), 0.008)
        try:
            import omni.ui as ui
            self.window = ui.Window('REXMI | Perception', width=430, height=410)
            with self.window.frame:
                with ui.VStack(spacing=5):
                    ui.Label('SIMULATED SENSOR VIEW | world frame')
                    for key, label in [('lidar','Live LiDAR returns'),('costs','Observed terrain costs'),
                                       ('path','Route on observed terrain'),('rays','Sampled LiDAR rays')]:
                        with ui.HStack():
                            check = ui.CheckBox(width=22)
                            check.model.set_value(self.show[key])
                            check.model.add_value_changed_fn(
                                lambda model, k=key: self.show.__setitem__(k, model.as_bool))
                            ui.Label(label)
                    if hasattr(nav._omap,'roughness'):
                        with ui.HStack():
                            for name in LAYER_NAMES:
                                ui.Button(name.title(),clicked_fn=lambda key=name:setattr(self,'layer',key))
                    self.legend=ui.Label('',word_wrap=True)
                    ui.Label('Red line: route | red tiles: obstacle / steep terrain | gold: checkpoint', word_wrap=True)
                    ui.Label('Muted amber: clearance buffer (planner avoids)', word_wrap=True)
                    ui.Label('Unseen cells omitted. Colors are planner costs, not safety guarantees.', word_wrap=True)
                    self.status = ui.Label('Waiting for sensor data', word_wrap=True)
        except ImportError:
            self.status = None
        print('[Perception] Live LiDAR + observed costs + route; session-only USD overlay')

    def _curve(self, suffix, color, width):
        curve = self._Geom.BasisCurves.Define(self.stage, self.ROOT + suffix)
        curve.CreateTypeAttr('linear')
        curve.CreateWrapAttr('nonperiodic')
        curve.CreateWidthsAttr([width])
        curve.SetWidthsInterpolation('constant')
        curve.CreateDisplayColorAttr([color])
        return curve

    def _set_curve(self, curve, points):
        curve.GetPointsAttr().Set(points.tolist())
        curve.GetCurveVertexCountsAttr().Set([2] * (len(points) // 2))

    def update(self):
        if not self.enabled or time.monotonic() < self.next_update:
            return
        self.next_update = time.monotonic() + self.interval
        try:
            self._update()
        except Exception as exc:
            print(f'[Perception] Overlay disabled after render error: {exc}')
            self.close()  # A display failure must not stop navigation.

    def _update(self):
        nav = self.nav
        pose = nav._sim_localizer.get_pose()
        origin = np.array([pose.x, pose.y, pose.z])
        hits = np.empty((0, 3), np.float32)
        if nav._lidar is not None:
            data = nav._lidar.data
            origin = data.pos_w[nav._env_idx].detach().cpu().numpy()
            hits = filter_hits(data.ray_hits_w[nav._env_idx].detach().cpu().numpy(), origin)
        cost, height, observed = nav._omap.get_visual_grid()
        with nav._lock:
            path = list(nav.shared['planned_path'])
            checkpoint = nav.shared.get('checkpoint')
            steering = nav.shared.get('steering_target')
            coverage = nav.shared.get('coverage')
            age = nav.shared.get('observation_age')
            mapped = nav.shared.get('mapped_coverage')
            mapped_area = nav.shared.get('mapped_area')
        started=time.perf_counter()
        if hasattr(nav._omap,'roughness'):
            palette,legend=survey_colors(cost,observed,nav._omap.clearance_mask,self.layer,
                                        nav._omap.slope,nav._omap.roughness,
                                        nav._omap.failures,nav._omap.successes)
            vertices,colors=survey_tiles(height,observed,palette,nav._omap.origin,
                                        nav._omap.cell_size,(pose.x,pose.y))
            if hasattr(self,'legend'): self.legend.text=legend
        else:
            vertices,colors=cost_tiles(cost,height,observed,nav._omap.origin,
                                       nav._omap.cell_size,(pose.x,pose.y))
        tile_count=len(colors)
        route = drape_path(path, height, observed, nav._omap.origin, nav._omap.cell_size)
        ray_ends = hits[::max(1, int(np.ceil(len(hits) / 32)))]
        rays = np.stack((np.broadcast_to(origin, ray_ends.shape), ray_ends), axis=1).reshape(-1, 3)
        # Defining new USD prims requires composition updates. Do it outside
        # Sdf.ChangeBlock; that batch is only safe for existing attributes.
        with self._Usd.EditContext(self.stage,self.stage.GetSessionLayer()):
            self._update_tiles(vertices,colors)
        with self._Usd.EditContext(self.stage, self.stage.GetSessionLayer()), self._Sdf.ChangeBlock():
            for key, prim in [('lidar',self.points),('costs',self.tiles),('path',self.route),('rays',self.rays)]:
                prim.GetVisibilityAttr().Set('inherited' if self.show[key] else 'invisible')
            # Keep point topology stable for Hydra as scan return counts change.
            point_buffer = np.broadcast_to(origin, (6000, 3)).copy()
            point_buffer[:len(hits)] = hits
            self.points.GetPointsAttr().Set(point_buffer.tolist())
            self.points.GetWidthsAttr().Set([0.035] * len(hits) + [0.0] * (6000 - len(hits)))
            self._set_curve(self.route, route)
            self._set_curve(self.rays, rays)
            for marker, xy, size in [(self.checkpoint_marker,checkpoint,.3), (self.target_marker,steering,.15)]:
                segments = np.empty((0,3))
                if xy is not None:
                    cell = nav._omap.world_to_cell(*xy)
                    if cell is not None and observed[cell] and np.isfinite(height[cell]):
                        x,y = xy
                        z = float(height[cell]) + .20
                        segments = np.array([[x-size,y,z],[x+size,y,z],[x,y-size,z],[x,y+size,z]])
                self._set_curve(marker,segments)
                marker.GetVisibilityAttr().Set('inherited' if self.show['path'] else 'invisible')
        self.render_ms=1000*(time.perf_counter()-started)
        if self.status is not None:
            self.status.text = (f'{len(hits):,} live returns | {tile_count:,} observed tiles\n'
                                f'Full survey | {self.render_ms:.0f} ms update | display capped at 5 Hz\n'
                                'Navigation pose uses the existing simulator-assisted localizer.')
            if coverage is not None:
                self.status.text += f'\nFresh local coverage {coverage:.0%} | oldest {age:.1f}s'
            if mapped is not None:
                self.status.text += f'\nMapped locally {mapped:.0%} | surveyed {mapped_area:.1f} m²'

    def _update_tiles(self, vertices, colors):
        quads=vertices.reshape(-1,4,3)
        keys=np.floor(quads.mean(axis=1)[:,:2]/6.4).astype(int)
        wanted=set(map(tuple,keys))
        for key in wanted:
            selected=np.all(keys==key,axis=1)
            v,c=quads[selected].reshape(-1,3),colors[selected]
            if key not in self.tile_chunks:
                mesh=self._Geom.Mesh.Define(self.stage,self.ROOT+f'/ObservedCosts/C_{key[0]+100}_{key[1]+100}')
                mesh.CreateSubdivisionSchemeAttr('none');mesh.CreateDoubleSidedAttr(True)
                capacity=1156  # boundary rounding can add a row/column to 32x32
                mesh.CreatePointsAttr([(0.,0.,0.)]*(capacity*4))
                mesh.CreateFaceVertexCountsAttr([4]*capacity)
                mesh.CreateFaceVertexIndicesAttr(list(range(capacity*4)))
                mesh.CreateDisplayColorAttr([(0.,0.,0.)]*capacity)
                mesh.CreateDisplayOpacityAttr([0.]*capacity)
                for primvar in (mesh.GetDisplayColorPrimvar(),mesh.GetDisplayOpacityPrimvar()):
                    primvar.SetInterpolation('uniform');primvar.SetIndices(list(range(capacity)))
                self.tile_chunks[key]=(mesh,StableTilePool(capacity),None)
            mesh,pool,previous=self.tile_chunks[key]
            data=pool.update(v,c)
            if previous is None or any(not np.array_equal(a,b) for a,b in zip(data,previous)):
                mesh.GetPointsAttr().Set(data[0].tolist())
                mesh.GetDisplayColorAttr().Set(data[1].tolist())
                mesh.GetDisplayOpacityAttr().Set(data[2].tolist())
            mesh.GetVisibilityAttr().Set('inherited')
            self.tile_chunks[key]=(mesh,pool,data)
        for key,(mesh,_,_) in self.tile_chunks.items():
            if key not in wanted: mesh.GetVisibilityAttr().Set('invisible')

    def close(self):
        self.enabled = False
        with self._Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            self.stage.RemovePrim(self.ROOT)
        if self.window is not None:
            self.window.destroy()
            self.window = None
