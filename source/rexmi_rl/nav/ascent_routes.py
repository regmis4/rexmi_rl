"""Bounded observed-route search for the ascent supervisor.

The baseline planner remains unchanged. This search prices direction on each
edge rather than allocating 32 copies of every cell to heading states. Every
returned edge is checked against the execution footprint.
"""
import heapq
import math

import numpy as np


OFFSETS = tuple((a, b) for a in range(-2, 3) for b in range(-2, 3)
                if (a or b) and math.gcd(abs(a), abs(b)) == 1)


def observed_route(grid, start, goal, yaw=0., require_fresh_start=True, profile='balanced',
                   support=None, uphill_limit=None, strict_frontier=True, allow_frontier=True,
                   no_go_exit=False, max_expansions=None, frontier_penalties=()):
    if getattr(grid, 'motion_turn_limit', None) is not None:
        from .staged_routes import staged_route
        return staged_route(grid, start, goal, yaw, offsets=OFFSETS, support=support,
                            profile=profile, uphill_limit=uphill_limit,
                            require_fresh_start=require_fresh_start, allow_frontier=allow_frontier,
                            no_go_exit=no_go_exit, max_expansions=max_expansions,
                            frontier_penalties=frontier_penalties)
    if max_expansions is None:
        max_expansions=25000 if getattr(grid,'terrain_prior',None) else 5000
    grid.last_route_search=dict(expansions=0,budget=max_expansions,budget_exhausted=False)
    first, target = grid.world_to_cell(*start), grid.world_to_cell(*goal)
    if first is None or target is None:
        return [], False
    support = grid.planning_support() if support is None else support
    if not support[first] or not (grid.exit_traversable(first) if no_go_exit else grid.traversable(first)):
        return [], False
    cost, turn_cost = grid.directional_costs(profile, uphill_limit, OFFSETS)
    allowed = grid.known & ~grid.blocked & support
    if no_go_exit:
        allowed = (grid.known & support & ~grid.hazard
                   & (grid.obstacle_distance >= grid.footprint_radius+grid.clearance))
    if require_fresh_start:
        rr, cc = np.indices(grid.known.shape)
        allowed &= grid.fresh | (np.hypot(rr-first[0], cc-first[1])*grid.cell_size > 2.)
    if not allowed[first]:
        return [], False
    def h(cell):
        return math.hypot(cell[0]-target[0], cell[1]-target[1])
    queue = [(h(first), 0., first)]
    costs, parents, headings = {first: 0.}, {}, {first: yaw}
    expanded = 0
    found = False
    end = first
    edge_cache = {}
    turn_cache = {}
    def turn_supported(node, point):
        if node not in turn_cache:
            turn_cache[node] = not grid.turn_support_reason(point)
        return turn_cache[node]
    while queue and expanded < max_expansions:
        _, value, node = heapq.heappop(queue)
        if value != costs.get(node):
            continue
        expanded += 1
        if node == target:
            end, found = node, True
            break
        a = start if node == first else grid.cell_to_world(*node)
        for k, (dr, dc) in enumerate(OFFSETS):
            nxt = node[0]+dr, node[1]+dc
            if not (0 <= nxt[0] < grid.n_cells and 0 <= nxt[1] < grid.n_cells and allowed[nxt]):
                continue
            weight = float(cost[nxt[0], nxt[1], k])
            current_weight = float(cost[node[0], node[1], k])
            # Directional bans inside an occupied all-direction exclusion may
            # not imprison the robot. segment_clear below still requires
            # monotonically outward travel and forbids re-entry; grade and
            # physical support are independently checked on the entire edge.
            if no_go_exit:
                if grid.no_go[nxt]:weight=math.hypot(dr,dc)*float(grid.cost[nxt])
                if grid.no_go[node]:current_weight=math.hypot(dr,dc)*float(grid.cost[node])
            if not math.isfinite(weight) or not math.isfinite(current_weight):
                continue
            arrival_heading=math.atan2(dc,dr)
            angle=abs(math.atan2(math.sin(arrival_heading-headings[node]),math.cos(arrival_heading-headings[node])))
            # Straight segments alone do not establish an executable bend.
            # The follower aligns in place above this error on a hillside;
            # reserve the same room here before offering that approach.
            if angle>.30 and grid.slope[node]>math.tan(math.radians(15)) and not turn_supported(node,a):
                continue
            # Pricing only cells produces alternating diagonal tacks on a
            # slope: each edge is legal but the tracking chord is too steep.
            # Retain the best arrival heading and price turns so contours have
            # useful continuous runs. This bounded search is approximate, not
            # a claim of optimality over every heading-state combination.
            trial = value+weight+float(turn_cost[node])*angle/(math.pi/4)
            if trial >= costs.get(nxt, math.inf):
                continue
            b = grid.cell_to_world(*nxt)
            edge = node, nxt
            if edge not in edge_cache:
                # A conservative fresh near-field check and exact sweep prevent
                # offering an A* edge the follower can never execute.
                fresh = require_fresh_start and math.dist(start, a) < 2.
                edge_cache[edge] = (grid.segment_clear(a, b, require_fresh=fresh, no_go_exit=no_go_exit)
                                    and not grid.support_reason(a, b)
                                    and not grid.grade_reason(a, b, uphill_limit=uphill_limit))
            if not edge_cache[edge]:
                continue
            parents[nxt], costs[nxt] = node, trial
            headings[nxt]=arrival_heading
            heapq.heappush(queue, (trial+h(nxt), trial, nxt))
    grid.last_route_search=dict(expansions=expanded,budget=max_expansions,
                                budget_exhausted=not found and bool(queue) and expanded>=max_expansions)
    if not found:
        if not allow_frontier:
            return [], False
        candidates = []
        span = max(4, math.ceil((grid.footprint_radius+.8)/grid.cell_size))
        for node, value in costs.items():
            if math.dist(grid.cell_to_world(*node), start) < 1.:
                continue
            r, c = node
            patch = grid.known[max(0, r-span):r+span+1, max(0, c-span):c+span+1]
            unknown = int((~patch).sum())
            if unknown < 3:
                continue
            progress = (h(first)-h(node))*grid.cell_size
            point=grid.cell_to_world(*node)
            # A survey stop must leave room to orient toward newly observed
            # terrain; otherwise it is an attractive but trapped frontier.
            if not turn_supported(node,point):
                continue
            deferred=sum(25.*max(0.,1.-math.dist(point,old)/4.) for old in frontier_penalties)
            candidates.append((3*progress+.03*min(unknown, 40)-.025*value*grid.cell_size-deferred, node))
        if not candidates:
            return [], False
        end = max(candidates)[1]
    chain = [end]
    while chain[-1] != first:
        chain.append(parents[chain[-1]])
    path = [grid.cell_to_world(*cell) for cell in reversed(chain)]
    path[0] = tuple(start)
    if found:
        if (not grid.segment_clear(path[-1], goal, require_fresh=False, no_go_exit=no_go_exit)
                or grid.support_reason(path[-1], goal)):
            return [], False
        path.append(tuple(goal))
    return path, found
