"""One additive remote-navigation objective, in cell-distance cost units.

Reuses the existing gentle travel coefficients. These characterize terrain
difficulty; they are NOT a calibrated probability of traversal success.
No diagonal-angle preference or post-search peak penalties are applied.
"""
import math
import numpy as np

NAME = 'gentle-additive-v1'


def components(g, cells, dx, dy):
    norm = math.hypot(dx, dy)
    along = (g.gx[cells]*dx+g.gy[cells]*dy)/norm
    across = np.abs(-g.gx[cells]*dy+g.gy[cells]*dx)/norm
    grade, side = np.degrees(np.arctan(along)), np.degrees(np.arctan(across))
    heading = g.heading_bin(math.atan2(dy, dx))
    failures, successes = g.failures[..., heading][cells], g.successes[..., heading][cells]
    terms = dict(base=g.cost[cells].astype(float), longitudinal=2.*np.abs(along),
        cross_linear=4.*across, uphill=30.*(np.maximum(0., grade-15.)/20.)**2,
        downhill=4.*(np.maximum(0., -grade-20.)/15.)**2,
        cross_quadratic=math.sqrt(2.5)*8.*(side/35.)**2,
        cliff_room=24.*np.maximum(0., 1.-g.steep_distance[cells]/(g.footprint_radius+.3))**2,
        failure_memory=6.*failures, untested=.3*(successes == 0))
    allowed = ((failures < 2) & ~g.approach_blocked[..., heading][cells]
               & (grade <= g.uphill_limit_deg+1e-6) & (side <= g.side_limit_deg+1e-6))
    return terms, allowed


def density(g, cells, dx, dy):
    terms, allowed = components(g, cells, dx, dy)
    return np.where(allowed, sum(terms.values()), np.inf)


def turn_cost(g, point, incoming, outgoing):
    cell = g.world_to_cell(*point)
    if cell is None:
        return math.inf
    angle = abs(math.atan2(math.sin(outgoing-incoming), math.cos(outgoing-incoming)))
    ratio = min(1., float(g.slope[cell])/math.tan(math.radians(g.MAX_SLOPE_DEG)))
    return .8*(1.+2.*ratio**2)*angle/(math.pi/4.)


def segment_cost(g, a, b, *, breakdown=False):
    delta = np.subtract(b, a)
    length = float(np.linalg.norm(delta))
    if length < 1e-9:
        return {} if breakdown else 0.
    # Integrate the piecewise-constant cell field at its actual crossings.
    # Splitting/compacting a collinear segment must not change its price.
    start=(np.asarray(a)-g.origin+g.world_size/2)/g.cell_size
    end=(np.asarray(b)-g.origin+g.world_size/2)/g.cell_size
    cuts=[0.,1.]
    for dim in (0,1):
        if abs(end[dim]-start[dim])<1e-12:continue
        lo,hi=sorted((start[dim],end[dim]))
        boundaries=np.arange(math.floor(lo)+1,math.ceil(hi))
        cuts.extend(((boundaries-start[dim])/(end[dim]-start[dim])).tolist())
    cuts=np.unique(np.clip(cuts,0.,1.))
    intervals=np.diff(cuts)
    points=np.asarray(a)+(.5*(cuts[:-1]+cuts[1:]))[:,None]*delta
    cells = np.floor((points-g.origin+g.world_size/2)/g.cell_size).astype(int)
    if np.any(cells < 0) or np.any(cells >= g.n_cells):
        return {'invalid': math.inf} if breakdown else math.inf
    cells = cells[:, 0], cells[:, 1]
    terms, allowed = components(g, cells, *delta)
    if not np.all(g.known[cells] & allowed):
        return {'invalid': math.inf} if breakdown else math.inf
    terms = {k: float(np.sum(v*intervals))*length/g.cell_size for k, v in terms.items()}
    return terms if breakdown else sum(terms.values())


def route_cost(g, path, yaw=0., *, require_fresh=False, breakdown=False):
    totals = {'turn': 0.}
    incoming = yaw
    for a, b in zip(path, path[1:]):
        if math.dist(a, b) < 1e-9:
            continue
        if (not g.segment_clear(a, b, require_fresh=require_fresh) or g.support_reason(a, b)):
            return {'invalid': math.inf} if breakdown else math.inf
        outgoing = math.atan2(b[1]-a[1], b[0]-a[0])
        for name, value in segment_cost(g, a, b, breakdown=True).items():
            totals[name] = totals.get(name, 0.)+value
        totals['turn'] += turn_cost(g, a, incoming, outgoing)
        incoming = outgoing
    return totals if breakdown else sum(totals.values())
