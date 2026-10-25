"""Soft terrain difficulty from retained samples; never a physical permission.

A square circumscribing the body at any yaw is a support-neighborhood proxy,
not an exact contact or traction model. Weights are characterization settings.
"""
import math
import numpy as np
from scipy import ndimage as ndi

FIELDS = ('footprint_rms', 'footprint_step', 'footprint_spread', 'footprint_coverage', 'footprint_step_coverage',
          'footprint_rough_cost', 'footprint_step_cost', 'footprint_unknown_cost')


def initialize(grid):
    for name in FIELDS:
        setattr(grid, name, np.full(grid.known.shape, np.nan, np.float32))
    grid.footprint_costs_enabled = False
    grid._footprint_cost_key = None


def rebuild(grid):
    """Apply once per freshly rebuilt base map; keep geometry and masks intact."""
    if not grid.footprint_costs_enabled:
        return
    radius = math.ceil(math.hypot(grid.half_length, grid.half_width)/grid.cell_size)
    key = (grid.revision, radius, grid.cell_size, grid.SUBDIVISIONS)
    if grid._footprint_cost_key == key:
        return
    size = 2*radius+1
    d = np.arange(-radius, radius+1)*grid.cell_size
    axis = ((np.arange(grid.SUBDIVISIONS)+.5)/grid.SUBDIVISIONS-.5)*grid.cell_size
    bx, by = (a.ravel() for a in np.meshgrid(axis, axis, indexing='ij'))
    valid = (grid._bin_count > 0) & np.isfinite(grid._bin_low) & np.isfinite(grid._bin_high)
    valid &= grid.known[..., None]
    z = np.where(valid, .5*(grid._bin_low.astype(float)+grid._bin_high), 0.)
    n = valid.sum(axis=-1).astype(float)
    sx, sy = (np.sum(valid*b, axis=-1) for b in (bx, by))
    sz = z.sum(axis=-1)
    def corr(a, px=0, py=0):
        return ndi.correlate1d(ndi.correlate1d(a, d**px, axis=0, mode='constant'),
                               d**py, axis=1, mode='constant')
    count = corr(n)
    matrix = np.empty((*n.shape, 3, 3))
    matrix[..., 0, 0] = count
    matrix[..., 0, 1] = matrix[..., 1, 0] = corr(n, 1)+corr(sx)
    matrix[..., 0, 2] = matrix[..., 2, 0] = corr(n, 0, 1)+corr(sy)
    matrix[..., 1, 1] = corr(n, 2)+2*corr(sx, 1)+corr(np.sum(valid*bx*bx, axis=-1))
    matrix[..., 2, 2] = corr(n, 0, 2)+2*corr(sy, 0, 1)+corr(np.sum(valid*by*by, axis=-1))
    matrix[..., 1, 2] = matrix[..., 2, 1] = (corr(n, 1, 1)+corr(sx, 0, 1)
                                          +corr(sy, 1)+corr(np.sum(valid*bx*by, axis=-1)))
    rhs = np.stack((corr(sz), corr(sz, 1)+corr(np.sum(z*bx, axis=-1)),
                    corr(sz, 0, 1)+corr(np.sum(z*by, axis=-1))), axis=-1)
    coverage = count/(size*size*grid.SUBDIVISIONS**2)
    supported = grid.known & (coverage >= .8)
    # Scale coordinates before checking conditioning (metres versus counts).
    scale = np.array([1., max(radius*grid.cell_size, grid.cell_size),
                      max(radius*grid.cell_size, grid.cell_size)])
    normalized = matrix/(scale[:, None]*scale[None, :])
    eigen = np.linalg.eigvalsh(normalized[supported])
    supported[supported] &= (eigen[:, 0] > 1e-8) & (eigen[:, -1]/np.maximum(eigen[:, 0], 1e-12) < 1e4)
    fit = np.full_like(rhs, np.nan)
    fit[supported] = np.linalg.solve(matrix[supported], rhs[supported, ..., None])[..., 0]
    rms = np.sqrt(np.maximum(0., (corr(np.sum(z*z, axis=-1))-np.sum(fit*rhs, axis=-1))
                             /np.maximum(count, 1)))
    # Detrended differences of adjacent cell means: a discontinuity proxy.
    # Centroids account for partially sampled subcells; no raw-slope penalty.
    means = sz/np.maximum(n, 1)
    cx, cy = sx/np.maximum(n, 1), sy/np.maximum(n, 1)
    step = np.zeros(n.shape)
    pair_count = np.zeros(n.shape)
    for dim in (0, 1):
        a, b = [slice(None), slice(None)], [slice(None), slice(None)]
        a[dim], b[dim] = slice(None, -1), slice(1, None)
        a, b = tuple(a), tuple(b)
        okay = (n[a] > 0) & (n[b] > 0) & supported[a] & supported[b]
        dx = cx[b]-cx[a]+(grid.cell_size if dim == 0 else 0.)
        dy = cy[b]-cy[a]+(grid.cell_size if dim == 1 else 0.)
        gradient = .5*(fit[a][..., 1:]+fit[b][..., 1:])
        jump = np.where(okay, np.abs(means[b]-means[a]-gradient[..., 0]*dx-gradient[..., 1]*dy), 0.)
        edges = np.zeros(n.shape)
        available = np.zeros(n.shape)
        edges[a], available[a] = jump, okay
        domain = np.ones((size, size), bool)
        boundary = [slice(None), slice(None)]; boundary[dim] = -1
        domain[tuple(boundary)] = False
        step = np.maximum(step, ndi.maximum_filter(edges, footprint=domain, mode='constant'))
        weights = [np.ones(size), np.ones(size)]; weights[dim][-1] = 0.
        pair_count += ndi.correlate1d(ndi.correlate1d(available, weights[0], axis=0, mode='constant'),
                                      weights[1], axis=1, mode='constant')
    pair_coverage = pair_count/(2*size*(size-1))
    step_supported = supported & (pair_coverage >= .8)
    spread = np.max(np.where(valid, grid._bin_high-grid._bin_low, 0.), axis=-1)
    spread = ndi.maximum_filter(spread, size=size, mode='constant')
    grid.footprint_rms = np.where(supported, rms, np.nan).astype(np.float32)
    grid.footprint_step = np.where(step_supported, step, np.nan).astype(np.float32)
    grid.footprint_spread = np.where(supported, spread, np.nan).astype(np.float32)
    grid.footprint_coverage = coverage.astype(np.float32)
    grid.footprint_step_coverage = pair_coverage.astype(np.float32)
    grid.footprint_rough_cost = np.where(step_supported, np.minimum(4., 40.*rms), 0.).astype(np.float32)
    grid.footprint_step_cost = np.where(step_supported, np.minimum(4., 40.*step), 0.).astype(np.float32)
    grid.footprint_unknown_cost = np.where(step_supported, 0., 8.).astype(np.float32)
    # Replace the old subcell deadband term, avoiding double-counting relief.
    usable = grid.known & ~grid.blocked
    old = np.minimum(4., 10.*np.maximum(0., np.nan_to_num(grid.roughness)-.06))
    grid.cost[usable] += (grid.footprint_rough_cost+grid.footprint_step_cost
                         +grid.footprint_unknown_cost-old)[usable]
    grid._footprint_cost_key = key


def route_components(grid, path, score):
    """Use the same distance/cell quadrature as route_cost; no extra penalty."""
    totals = np.zeros(3)
    length = supported_length = peak_rms = peak_step = 0.
    for a, b in zip(path, path[1:]):
        delta = np.subtract(b, a)
        distance = float(np.linalg.norm(delta))
        if distance < 1e-9:
            continue
        steps = max(1, math.ceil(float(np.max(np.abs(delta)))/grid.cell_size-1e-8))
        ds = distance/steps
        for t in np.linspace(0, 1, steps+1)[1:]:
            cell = grid.world_to_cell(*(np.asarray(a)+t*delta))
            if cell is None:
                return dict(total=None, reason='route leaves map')
            totals += np.array([grid.footprint_rough_cost[cell], grid.footprint_step_cost[cell],
                                grid.footprint_unknown_cost[cell]])*ds/grid.cell_size
            length += ds
            if np.isfinite(grid.footprint_rms[cell]) and np.isfinite(grid.footprint_step[cell]):
                supported_length += ds
                peak_rms = max(peak_rms, float(grid.footprint_rms[cell]))
                peak_step = max(peak_step, float(grid.footprint_step[cell]))
    return dict(base=float(score-totals.sum()) if math.isfinite(score) else None,
                roughness=float(totals[0]), discontinuity=float(totals[1]), unknown=float(totals[2]),
                total=float(score) if math.isfinite(score) else None,
                supported_fraction=supported_length/max(length, 1e-9),
                peak_rms_m=peak_rms if supported_length else None,
                peak_step_m=peak_step if supported_length else None,
                model='square support-neighborhood; experimental soft costs')
