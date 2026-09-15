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
               radius=12.0, limit=4000):
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
    colors[(values >= 20) | ~np.isfinite(values)] = [1.0, 0.08, 0.26]
    return vertices, colors.astype(np.float32)


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
        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            UsdGeom.Xform.Define(self.stage, self.ROOT)
            self.points = UsdGeom.Points.Define(self.stage, self.ROOT + '/LiveLidar')
            self.points.CreatePointsAttr([(0.0, 0.0, 0.0)] * 6000)
            self.points.CreateWidthsAttr([0.0] * 6000)
            self.points.SetWidthsInterpolation('vertex')
            self.points.CreateDisplayColorAttr([(0.15, 0.90, 1.0)])
            self.tiles = UsdGeom.Mesh.Define(self.stage, self.ROOT + '/ObservedCosts')
            self.tiles.CreateSubdivisionSchemeAttr('none')
            self.tiles.CreateDoubleSidedAttr(True)
            self.tiles.CreateDisplayOpacityAttr([0.55])
            self.route = self._curve('/ObservedRoute', (0.1, 0.95, 1.0), 0.045)
            self.rays = self._curve('/SampledRays', (0.25, 0.65, 1.0), 0.008)
        try:
            import omni.ui as ui
            self.window = ui.Window('REXMI | Perception', width=365, height=260)
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
                    ui.Label('Teal: low cost | amber: elevated | red: blocked', word_wrap=True)
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
        vertices, colors = cost_tiles(cost, height, observed, nav._omap.origin,
                                      nav._omap.cell_size, (pose.x, pose.y))
        route = drape_path(path, height, observed, nav._omap.origin, nav._omap.cell_size)
        ray_ends = hits[::max(1, int(np.ceil(len(hits) / 32)))]
        rays = np.stack((np.broadcast_to(origin, ray_ends.shape), ray_ends), axis=1).reshape(-1, 3)
        with self._Usd.EditContext(self.stage, self.stage.GetSessionLayer()), self._Sdf.ChangeBlock():
            for key, prim in [('lidar',self.points),('costs',self.tiles),('path',self.route),('rays',self.rays)]:
                prim.GetVisibilityAttr().Set('inherited' if self.show[key] else 'invisible')
            # Keep point topology stable for Hydra as scan return counts change.
            point_buffer = np.broadcast_to(origin, (6000, 3)).copy()
            point_buffer[:len(hits)] = hits
            self.points.GetPointsAttr().Set(point_buffer.tolist())
            self.points.GetWidthsAttr().Set([0.035] * len(hits) + [0.0] * (6000 - len(hits)))
            self.tiles.GetPointsAttr().Set(vertices.tolist())
            self.tiles.GetFaceVertexCountsAttr().Set([4] * len(colors))
            self.tiles.GetFaceVertexIndicesAttr().Set(list(range(len(vertices))))
            self.tiles.GetDisplayColorPrimvar().SetInterpolation('uniform')
            self.tiles.GetDisplayColorAttr().Set(colors.tolist())
            self._set_curve(self.route, route)
            self._set_curve(self.rays, rays)
        if self.status is not None:
            self.status.text = (f'{len(hits):,} live returns | {len(colors):,} observed tiles\n'
                                f'{len(route)//2} route segments | display capped at 5 Hz\n'
                                'Navigation pose uses the existing simulator-assisted localizer.')

    def close(self):
        self.enabled = False
        with self._Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            self.stage.RemovePrim(self.ROOT)
        if self.window is not None:
            self.window.destroy()
            self.window = None
