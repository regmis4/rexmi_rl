"""Supported exterior goal region, independent of any demonstrated trajectory.

One bounded heading-state graph searches all sampled terminals together. Static
connectivity is a prediction; arrival still needs measured settling and the
controller's current-map physical output gate.
"""
import math

import numpy as np

from . import stage_graph
from .staged_routes import motion_route_reason, observed_edge_reason


NAME = 'any-safe-exit-additive-v1'


def mission_status(control):
    """Numeric target coordinates stay separate from user-facing description."""
    return dict(stage='complete' if control.wp_idx else 'climb to safe exterior',
                goal=control.selected_exit,goal_mode='any_exit',
                goal_description='any supported gentle exit')


def reason(g, point, rim_radius):
    if not math.isfinite(rim_radius) or rim_radius <= 0:
        return 'invalid crater rim radius'
    if not np.isfinite(point).all():
        return 'nonfinite exterior position'
    radius = max(g.footprint_radius, math.hypot(g.half_length, g.half_width))
    if math.dist(point, g.origin) - radius < rim_radius + .25:
        return 'full body has not cleared the crater rim with reserve'
    cells = g._support_cells(point, reserve=radius-g.footprint_radius)
    if cells is None or not g.known[cells].all():
        return 'exterior footprint lacks observed support'
    if not np.isfinite(g.surface[cells]).all() or not np.isfinite(g.slope[cells]).all():
        return 'exterior terrain is unmeasured'
    if g.hazard[cells].any() or g.no_go[cells].any():
        return 'hazard or recorded exclusion beneath exterior footprint'
    if (g.slope[cells] > math.tan(math.radians(15.))).any():
        return 'exterior footprint is not gentle settling terrain'
    return ''


def candidates(g, rim_radius):
    # Uniform geography, not a preferred azimuth or demonstration endpoint.
    points = []
    for radius in (rim_radius+1.5, rim_radius+3.):
        for theta in np.arange(32)*2.*math.pi/32:
            point = tuple(np.asarray(g.origin)+radius*np.array([math.cos(theta), math.sin(theta)]))
            if not reason(g, point, rim_radius):
                points.append(point)
    return points


def route_reason(g, path, yaw, observer, rim_radius, physical_check):
    if len(path) < 2:
        return 'no complete exterior route'
    rejection = reason(g, path[-1], rim_radius) or motion_route_reason(g, path, yaw)
    if rejection:
        return rejection
    for a, b in zip(path, path[1:]):
        rejection = observed_edge_reason(g, a, b, observer) or physical_check(a, b)
        if rejection:
            return rejection
    return ''


def plan(g, start, yaw, rim_radius, physical_check, *, candidate_source=None, wall_budget=180.):
    terminals = candidates(g, rim_radius)
    if not terminals:
        return [], dict(objective=NAME, complete=False, budget_exhausted=False,
                        reason='no supported exterior terminals in bounded inventory', terminal_candidates=0)
    graph = stage_graph.build(g, start, terminals[0], physical_check,
        candidate_source=candidate_source, terminals=terminals, wall_budget=wall_budget)
    path, stats = stage_graph.search(g, graph, yaw)
    rejection = route_reason(g, path, yaw, start, rim_radius, physical_check) if path else (
        'search budget exhausted without a complete incumbent' if stats['budget_exhausted']
        or graph['build_budget_exhausted'] else 'no complete heading-compatible route in sampled graph')
    stats.update(objective=NAME, reason=rejection, complete=bool(path and not rejection),
        path=path, terminal_candidates=len(terminals), candidate_inventory=graph['inventory'],
        nodes=len(graph['points']), directed_checks=graph['directed_checks'],
        accepted_links=len(graph['edges']), edge_rejections=graph['rejections'],
        build_seconds=graph['build_seconds'], build_budget_exhausted=graph['build_budget_exhausted'],
        selected_exit=list(path[-1]) if path else None,
        length_m=sum(math.dist(a,b) for a,b in zip(path,path[1:])),
        scope='static map prediction; no physical ascent qualification')
    return (path if not rejection else []), stats
