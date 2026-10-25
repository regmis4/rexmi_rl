"""Bounded static-crater ascent alternatives; no command or mission changes."""
import math

import numpy as np

from .ascent_routes import observed_route
from .staged_routes import motion_route_reason, observed_edge_reason
from .swept_footprint import oriented_swept_cells
from .footprint_costs import route_components
from .success_selection import select as select_by_success
from .turn_staging import stage_reason


def outbound_scope(control, xy):
    if not control.static_retention_enabled or control.wp_idx >= len(control.waypoints):
        return False
    goal = control.waypoints[control.wp_idx]
    centre = control.grid.origin
    return math.dist(xy, centre) <= 4. and math.dist((goal.x, goal.y), centre) >= 10.


def physical_profile(control, path):
    maximum = 0.
    for a, b in zip(path, path[1:]):
        heading = math.atan2(b[1]-a[1], b[0]-a[0])
        reason = control._physical_sweep(a, b, heading, commanded=True, circular_support=False)
        if reason:
            return reason, math.inf
        cells = oriented_swept_cells(control.grid, a, b, heading)
        maximum = max(maximum, float(np.degrees(np.arctan(control.grid.slope[cells].max()))))
    return '', maximum


def augment(control, routes, xy, yaw):
    if not outbound_scope(control, xy):
        return routes
    g = control.grid
    waypoint = control.waypoints[control.wp_idx]
    goal, centre = (waypoint.x, waypoint.y), np.asarray(g.origin)
    radius = math.dist(goal, centre)
    bearing = math.atan2(goal[1]-centre[1], goal[0]-centre[0])
    survivors = []
    def gentle(point):
        # This straight-wall screen does not yet know its connection headings.
        # A permissive route-relative broad phase cannot stand in for a checked
        # turn. Steep turn stages are searched with actual headings by A*.
        return g.max_support_slope(point)<=g.motion_turn_limit and not stage_reason(g,point)
    # Fixed inventory relative to the destination, not a recorded route.
    for radial in np.linspace(bearing-math.pi/3, bearing+5*math.pi/12, 16):
        start = centre+3.*np.array([math.cos(radial), math.sin(radial)])
        for offset in (-math.pi/12, 0., math.pi/12):
            heading = radial+offset
            axis = np.array([math.cos(heading), math.sin(heading)])
            dot = float((start-centre)@axis)
            length = -dot+math.sqrt(dot*dot+radius*radius-9.)
            end = start+length*axis
            reason, peak = physical_profile(control, [start, end])
            if not reason:
                reason = observed_edge_reason(g, start, end, xy, fresh_radius=None)
            if not reason and not gentle(end):
                reason = 'exit lacks gentle turning support'
            launch = None
            if not reason:
                launch = next((start-step*.25*axis for step in range(9)
                               if gentle(start-step*.25*axis)), None)
                if launch is None:
                    reason = 'no gentle launch within 2 m on the same line'
                else:
                    reason, peak = physical_profile(control, [launch, end])
                    reason = reason or observed_edge_reason(g, launch, end, xy, fresh_radius=None)
            control.route_searches.append(dict(profile='wall-screen', heading_rad=heading,
                connected=False, has_route=False, uphill_limit=g.uphill_limit_deg,
                start=start.tolist(), endpoint=end.tolist(), rejection=reason, max_body_slope_deg=peak if math.isfinite(peak) else None))
            if not reason:
                # Bounded preselection uses the same finite terrain preferences.
                score = control.route_score([tuple(launch), tuple(end)], heading)
                connection_estimate = 10.*(math.dist(xy, launch)+math.dist(end, goal))
                control.route_searches[-1].update(screen_score=score if math.isfinite(score) else None,
                    connection_estimate=connection_estimate,
                    cost_components=route_components(g, [launch, end], score))
                if math.isfinite(score):
                    survivors.append((score+connection_estimate, math.dist(end, goal), tuple(launch), tuple(end), heading))
                else:
                    control.route_searches[-1]['rejection'] = 'existing route cost rejects climb support or grade'
    for index, (_, _, launch, end, heading) in enumerate(sorted(survivors)[:3]):
        prefix, reached = observed_route(g, xy, launch, yaw, allow_frontier=False,
            require_fresh_start=False, max_expansions=10000)
        suffix, finished = observed_route(g, end, goal, heading, allow_frontier=False,
            require_fresh_start=False, max_expansions=10000)
        path = prefix+[end]+suffix[1:] if reached and finished else []
        reason = 'launch or exterior connection unavailable'
        peak = math.inf
        if len(path) >= 3:
            reason = motion_route_reason(g, path, yaw)
            physical, peak = physical_profile(control, path)
            reason = reason or physical
            for a, b in zip(path, path[1:]):
                reason = reason or observed_edge_reason(g, a, b, xy, fresh_radius=None)
        label = f'wall-corridor-{index}'
        control.route_searches.append(dict(profile=label, launch=list(launch), exit=list(end),
            path=path, connected=not bool(reason), has_route=len(path)>1,
            uphill_limit=g.uphill_limit_deg, rejection=reason,
            max_body_slope_deg=peak if math.isfinite(peak) else None))
        if not reason:
            routes.append((label, path, True))
    return routes


def select(control, candidates, xy, yaw):
    for candidate in candidates:
        candidate[9]['cost_components'] = route_components(control.grid, candidate[2], candidate[1])
        if getattr(control,'ascent_first_active',False):
            candidate[9]['traversal_objective']=control.graph_plan_stats.get('objective','ascent-first-v1')
            candidate[9]['objective_rank']=control.graph_plan_stats.get('rank')
        elif getattr(control.grid, 'traversal_objective', None):
            from .traversal_objective import route_cost
            candidate[9]['traversal_objective'] = control.grid.traversal_objective
            candidate[9]['objective_components'] = route_cost(control.grid, candidate[2], yaw, breakdown=True)
        if getattr(control,'graph_plan_stats',None):
            candidate[9]['checkpoint_planner']=control.graph_plan_stats
            candidate[9]['height_retention_status']=('any-exit additive planning; no height-retention veto'
                if getattr(control,'goal_mode','fixed')=='any_exit' else 'sampled 0.20 m ascent preference checked'
                if control.graph_objective_active else 'losses measured, no hard height veto'
                if getattr(control,'ascent_first_active',False) else 'unverified: additive fallback')
    if not outbound_scope(control, xy) or not any(candidate[3] for candidate in candidates):
        return select_by_success(control, candidates, xy)
    ranked = []
    for candidate in candidates:
        reason, peak = physical_profile(control, candidate[2])
        info = candidate[9]
        info.update(max_body_slope_deg=peak if math.isfinite(peak) else None,
                    selection_rule='finite total route cost; unchanged physical eligibility')
        if reason:
            info['rejection'] = reason
        else:
            ranked.append((candidate[1], candidate))
    if ranked:
        control.planning_failure = ''
        return select_by_success(control, [item[1] for item in ranked], xy)
    control.planning_failure = 'no candidate passes oriented ascent geometry'
    return None
