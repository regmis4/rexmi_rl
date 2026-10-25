"""Nominal wheelbase uphill demand fourth power, integrated against displayed cost.

This is a geometric characterization score, not traction/success probability.
Use the existing footprint-mean height field, with no additional smoothing.
Each straight leg evaluates its tangent-oriented front/rear window on actual
terrain, including beyond endpoints. Collinear subdivision cannot alter score.
"""
import math
import time
import numpy as np
from . import climb_burden as cb, stage_graph

NAME = 'wheelbase-demand-v2-fourth'
# URDF hip x=±.1934, thigh/calf lengths=.213/.2264 m. assets/go2w.py
# reference stance: front thigh .8, rear thigh 1.0, calf -1.5 rad.
# Longitudinal wheel-centre separation; articulated gait changes live spacing.
WHEELBASE_M = (.3868 - .213*math.sin(.8) - .2264*math.sin(-.7)
               + .213*math.sin(1.) + .2264*math.sin(-.5))
GAUSS_X, GAUSS_W = np.polynomial.legendre.leggauss(5)


def sections(g, field, a, b):
    """Exact integral of C * positive(front-minus-rear height / L)^4 ds.

    Cuts include cost and BOTH shifted interpolation boundaries. Delta height
    is quadratic within each interval; split its zeroes before integrating the
    positive-part fourth power (degree eight) using five-point Gaussian quadrature.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    length = float(np.linalg.norm(b-a))
    if length == 0:
        return []
    first = (a-g.origin+g.world_size/2)/g.cell_size
    delta = (b-a)/g.cell_size
    shift = (b-a)/length*(WHEELBASE_M/2/g.cell_size)
    cuts = [0., 1.]
    for axis in (0, 1):
        if abs(delta[axis]) < 1e-14:
            continue
        for origin, offset in ((first[axis], 0.),
                               (first[axis]+shift[axis], .5),
                               (first[axis]-shift[axis], .5)):
            lo, hi = sorted((origin, origin+delta[axis]))
            boundaries = np.arange(math.floor(lo-offset)+1, math.ceil(hi-offset))+offset
            cuts.extend(((boundaries-origin)/delta[axis]).tolist())
    cuts = np.unique(np.clip(cuts, 0., 1.))
    rows = []
    for lo, hi in zip(cuts, cuts[1:]):
        mid = (lo+hi)/2
        cell = tuple(np.floor(first+mid*delta).astype(int))
        if any(i < 0 or i >= g.n_cells for i in cell):
            raise ValueError('cost outside map')
        cost = float(g.cost[cell])
        if not g.known[cell] or not math.isfinite(cost) or cost <= 0:
            raise ValueError('unobserved terrain cost')
        heights = [cb.height_at(field, first+t*delta+sign*shift)
                   for t in (lo, mid, hi) for sign in (1, -1)]
        if not np.isfinite(heights).all():
            raise ValueError('unobserved extended wheelbase height support')
        z0, zm, z1 = np.asarray(heights).reshape(3, 2) @ np.array([1., -1.])
        q2 = 2*(z1+z0-2*zm)
        q1 = z1-z0-q2
        roots = np.roots([q2, q1, z0]) if q2 != 0 else (
            [-z0/q1] if q1 != 0 else [])
        split = sorted([0., 1.] + [float(np.real(r)) for r in roots
            if abs(np.imag(r)) < 1e-12 and 0 < np.real(r) < 1])
        for u, v in zip(split, split[1:]):
            nodes = (u+v)/2 + (v-u)/2*GAUSS_X
            demand = np.maximum(0., z0+q1*nodes+q2*nodes**2)/WHEELBASE_M
            ds = length*(hi-lo)*(v-u)
            burden = cost*ds/2*float(GAUSS_W @ demand**4)
            probes = [u, v]
            if q2 != 0 and u < -q1/(2*q2) < v:
                probes.append(-q1/(2*q2))
            peak = max(0., *(z0+q1*t+q2*t*t for t in probes))/WHEELBASE_M
            rows.append(dict(d0=length*(lo+(hi-lo)*u), d1=length*(lo+(hi-lo)*v),
                cost=cost, burden=burden, peak_grade=peak,
                peak_window_dz_m=peak*WHEELBASE_M))
    return rows


def metric(g, field, a, b):
    return dict(burden=math.fsum(row['burden'] for row in sections(g, field, a, b)),
                length=math.dist(a, b))


def build(g, start, goal, physical_check, *, candidate_source=None, wall_budget=180.):
    began = time.perf_counter()
    field = cb.height_field(g)
    cache, unavailable = {}, []
    def cost(grid, a, b):
        try:
            cache[a, b] = metric(grid, field, a, b)
        except ValueError as exc:
            unavailable.append(str(exc))
            return {'unavailable_height': math.inf}
        return {'wheelbase_demand': cache[a, b]['burden']}
    graph = stage_graph.build(g, start, goal, physical_check,
        candidate_source=candidate_source,
        wall_budget=max(0., wall_budget-(time.perf_counter()-began)), cost_model=cost)
    for (i, j), edge in graph['edges'].items():
        edge.update(cache[graph['points'][i], graph['points'][j]])
    graph.update(height_field=field, objective_unavailable_links=len(unavailable),
                 objective_unavailable_reasons=sorted(set(unavailable)))
    return graph


def search(g, graph, yaw):
    path, stats = cb.search(g, graph, yaw)
    stats['objective'] = NAME
    stats['wheelbase_m'] = WHEELBASE_M
    stats['demand_exponent'] = 4
    if stats['complete']:
        stats['reason'] = 'minimum wheelbase uphill demand fourth power; distance breaks ties'
    return path, stats


route_reason = cb.route_reason


def describe_route(g, path):
    result = cb.describe_route(g, path)
    if not result:
        return result
    result['old_climb_burden'] = result['rank'][0]
    result['old_height_bands'] = result.pop('bands')
    result['old_worst_gain_window'] = result.pop('worst_gain_window')
    for point in result['height_profile']:
        point['old_climb_burden'] = point.pop('burden')
    field = cb.height_field(g)
    rows, distance = [], 0.
    for a, b in zip(path, path[1:]):
        for row in sections(g, field, a, b):
            row['d0'] += distance
            row['d1'] += distance
            rows.append(row)
        distance += math.dist(a, b)
    # Integration intervals are numerical, not wheelbase-sized bins. Every
    # location is scored with the same overlapping wheelbase-length window.
    result.update(rank=[math.fsum(r['burden'] for r in rows), distance],
        wheelbase_m=WHEELBASE_M, demand_exponent=4, sections=rows,
        peak_wheelbase_grade=max(r['peak_grade'] for r in rows),
        peak_wheelbase_dz_m=max(r['peak_window_dz_m'] for r in rows))
    return result
