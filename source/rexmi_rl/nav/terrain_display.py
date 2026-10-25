"""Display-only cost snapshots shared by reports, dashboard and Isaac overlay."""
import numpy as np

ROUTE = '#00d9ff'
TRAIL = '#ffffff'
BLOCKED = (.30, .31, .33)
UNKNOWN = (.12, .13, .15)


def cost_view(density, known, blocked, *, vmax=None, label='Terrain cost'):
    from matplotlib import colormaps
    from matplotlib.colors import LogNorm
    density = np.asarray(density)
    valid = known & ~blocked & np.isfinite(density) & (density > 0)
    if vmax is None:
        vmax = max(20., float(np.percentile(density[valid], 99))) if valid.any() else 20.
    norm = LogNorm(vmin=2., vmax=vmax)
    rgb = colormaps['magma'](norm(density))[..., :3].copy()
    rgb[blocked] = BLOCKED
    rgb[~known | ~np.isfinite(density)] = UNKNOWN
    return dict(rgb=rgb, vmin=2., vmax=vmax, label=label,
                legend=f'{label}: magma, log scale 2–{vmax:.0f} | grey: planning mask | dark grey: unknown')


def from_model(model):
    g, p = model.grid, model.parameters
    penalty = p.uphill_weight * np.expm1(
        np.maximum(model.terrain_angle-p.easy_angle_deg, 0.) / p.exponential_scale_deg)
    density = g.cost + (1+p.terrain_weight)*penalty*(1+p.roughness_interaction*model.roughness)
    view = cost_view(density, g.known, g.blocked, label='Frozen fitted uphill cost')
    view.update(surface=g.surface.copy(), known=g.known.copy(), origin=g.origin,
                world_size=g.world_size, cell_size=g.cell_size)
    return view


def heading_grade(grid, xy, yaw, model=None):
    """Signed terrain grade along the robot heading, not body pitch."""
    cell = grid.world_to_cell(*xy)
    if cell is None or not grid.known[cell]:
        return float('nan')
    geometry = model if model is not None else grid
    return float(np.degrees(np.arctan(
        geometry.gx[cell]*np.cos(yaw) + geometry.gy[cell]*np.sin(yaw))))
