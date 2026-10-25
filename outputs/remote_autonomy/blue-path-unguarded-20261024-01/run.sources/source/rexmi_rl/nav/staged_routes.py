"""Heading-state A*: straight steep corridors, turns at gentle staging only."""
import heapq
import math
import time
import numpy as np
from .turn_staging import stage_reason, turn_reason


def terrain_guidance(allowed, costs, offsets, target):
    """Frozen reverse-graph cost guidance, with at most one state per map cell.

    Relax heading and turn staging only. This is not an admissibility/optimality
    claim: the executable search also has continuous terminal shortcuts.
    Unreachable cells must fall back to Euclidean priority, never be pruned.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    started = time.perf_counter()
    nr, nc = allowed.shape
    ids = np.arange(nr*nc).reshape(nr, nc)
    rows, cols, weights = [], [], []
    for k, (dr, dc) in enumerate(offsets):
        src = (slice(max(0, -dr), min(nr, nr-dr)), slice(max(0, -dc), min(nc, nc-dc)))
        dst = (slice(max(0, dr), min(nr, nr+dr)), slice(max(0, dc), min(nc, nc+dc)))
        valid = allowed[src] & allowed[dst] & np.isfinite(costs[src][..., k]) & np.isfinite(costs[dst][..., k])
        # Actual search charges the destination's directional cost. Reverse
        # that exact edge, not its opposite-heading or predecessor cost.
        rows.append(ids[dst][valid]); cols.append(ids[src][valid])
        weights.append(costs[dst][..., k][valid])
    graph = csr_matrix((np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(nr*nc, nr*nc))
    distance = dijkstra(graph, directed=True, indices=int(ids[target]))
    stats = dict(kind='approximate reverse directed terrain cost; no optimality claim',
                 nodes=int(np.isfinite(distance).sum()), node_budget=nr*nc, edges=int(graph.nnz),
                 storage_bytes=int(graph.data.nbytes+graph.indices.nbytes+graph.indptr.nbytes+distance.nbytes),
                 prepass_ms=1000*(time.perf_counter()-started))
    return distance.reshape(nr, nc), stats


def observed_edge_reason(grid, a, b, observer, *, fresh_radius=2., no_go_exit=False, uphill_limit=None):
    """One spatial contract for search edges and compacted route admission.

    Full geometry is checked throughout. Freshness applies only to the part
    inside the observer's near field, even if compacting joins many cells into
    one long segment. A segment crossing the circle is clipped at both ends.
    """
    if not grid.segment_clear(a, b, require_fresh=False, no_go_exit=no_go_exit):
        return 'route geometry or directional exclusion changed'
    reason = grid.support_reason(a, b) or grid.grade_reason(a, b, uphill_limit=uphill_limit)
    if reason:
        return reason
    if fresh_radius is None:
        return ''
    ax, ay = a[0]-observer[0], a[1]-observer[1]
    dx, dy = b[0]-a[0], b[1]-a[1]
    squared = dx*dx+dy*dy
    if squared < 1e-12:
        interval = (0., 1.) if ax*ax+ay*ay <= fresh_radius**2 else None
    else:
        dot = ax*dx+ay*dy
        discriminant = dot*dot-squared*(ax*ax+ay*ay-fresh_radius**2)
        interval = None
        if discriminant >= 0.:
            root = math.sqrt(discriminant)
            lo, hi = max(0., (-dot-root)/squared), min(1., (-dot+root)/squared)
            if lo <= hi:
                interval = lo, hi
    if interval is not None:
        near_a = (a[0]+interval[0]*dx, a[1]+interval[0]*dy)
        near_b = (a[0]+interval[1]*dx, a[1]+interval[1]*dy)
        if not grid.segment_clear(near_a, near_b, require_fresh=True, no_go_exit=no_go_exit):
            return 'fresh observations missing inside 2 m planning near field'
    return ''


def compact_path(path):
    result = []
    for point in path:
        point = tuple(point)
        if result and math.dist(point, result[-1]) < 1e-8:
            continue
        if len(result) >= 2:
            a, b = np.subtract(result[-1], result[-2]), np.subtract(point, result[-1])
            if abs(a[0]*b[1]-a[1]*b[0]) < 1e-8 and np.dot(a, b) > 0:
                result[-1] = point
                continue
        result.append(point)
    return result


def terminal_travel_cost(grid, a, b, profile, no_go_exit=False):
    """Price the complete checked connector, in the lattice's cell-cost units.

    Scalar edge_cost uses the same profile formula as directional_costs. Its
    sampled quadrature can differ from a multi-cell lattice endpoint charge.
    Hold
    the segment heading fixed here; the search separately charges the departure
    turn with the same rule as a lattice edge. Geometry is never waived by cost.
    """
    if getattr(grid, 'traversal_objective', None) and not no_go_exit:
        from .traversal_objective import segment_cost
        return segment_cost(grid, a, b)
    delta = np.subtract(b, a)
    if np.linalg.norm(delta) < 1e-8:
        return 0.
    steps = max(1, math.ceil(float(np.max(np.abs(delta)))/grid.cell_size-1e-8))
    dr, dc = delta/grid.cell_size/steps
    heading = grid.heading_bin(math.atan2(dc, dr))
    total = 0.
    for t in np.linspace(0., 1., steps+1)[1:]:
        cell = grid.world_to_cell(*(np.asarray(a)+t*delta))
        if cell is None or not grid.known[cell]:
            return math.inf
        value = (math.hypot(dr, dc)*float(grid.cost[cell])
                 if no_go_exit and grid.no_go[cell]
                 else grid.edge_cost(cell, dr, dc, heading, profile))
        if not math.isfinite(value):
            return math.inf
        total += value
    return total


def motion_route_reason(grid, path, yaw=None):
    """Recheck turn staging after planning and again at action handoff."""
    path = compact_path(path)
    if len(path) < 2:
        return ''
    first = math.atan2(path[1][1]-path[0][1], path[1][0]-path[0][0])
    if yaw is not None and abs(math.atan2(math.sin(first-yaw), math.cos(first-yaw))) > .10:
        reason=turn_reason(grid,path[0],yaw,first)
        if reason:return 'first heading: '+reason
    for a, b, c in zip(path, path[1:], path[2:]):
        v, w = np.subtract(b, a), np.subtract(c, b)
        if abs(v[0]*w[1]-v[1]*w[0]) > 1e-8 or np.dot(v,w)<0:
            reason=turn_reason(grid,b,math.atan2(v[1],v[0]),math.atan2(w[1],w[0]))
            if reason:return 'route bend: '+reason
    return ''


def detour_shortcuts(grid, path, yaw, *, max_checks=16, min_saved=.5):
    """Propose bounded connections to later staging vertices of one route.

    These are new complete candidate geometries, never edits to an admitted
    checkpoint. Their costs, retreat and probability must be recomputed by the
    caller. The original terminal point is retained, including a frontier.
    """
    path = compact_path(path)
    if len(path) < 3:
        return []
    prefix, choices = 0., []
    for index in range(1, len(path)):
        prefix += math.dist(path[index-1], path[index])
        distance = math.dist(path[0], path[index])
        saved = prefix-distance
        if index > 1 and distance > .05 and saved >= min_saved:
            # Prefer nearby reconnections that remove substantial backtracking.
            choices.append((saved/(distance+.5), index, saved))
    proposals = []
    for _, index, saved in sorted(choices, reverse=True)[:max_checks]:
        join = path[index]
        candidate = compact_path([path[0]]+path[index:])
        reason = ''
        if stage_reason(grid,join):
            reason = 'shortcut connection lacks gentle staging'
        if not reason:
            reason = motion_route_reason(grid, candidate, yaw)
        if not reason:
            for edge, (a,b) in enumerate(zip(candidate,candidate[1:])):
                rejection = observed_edge_reason(grid,a,b,path[0])
                if rejection:
                    reason = f'shortcut segment {edge}: {rejection}'
                    break
        proposals.append(dict(path=candidate, join_index=index, saved_m=saved, rejection=reason))
    return proposals


def staged_route(grid, start, goal, yaw, *, offsets, support=None, profile='balanced',
                 uphill_limit=None, require_fresh_start=True, allow_frontier=True,
                 no_go_exit=False, max_expansions=None, frontier_penalties=()):
    budget = max_expansions or (60000 if getattr(grid, 'terrain_prior', None) else 15000)
    first, target = grid.world_to_cell(*start), grid.world_to_cell(*goal)
    grid.last_route_search = dict(expansions=0, budget=budget, budget_exhausted=False,
                                  contract='straight steep corridors; gentle turn staging')
    if first is None or target is None:
        return [], False
    # Preserve the measured sub-cell offset throughout the search. The robot
    # never has to pivot or snap onto a cell centre to enter a straight climb.
    def point(cell):
        return (start[0]+(cell[0]-first[0])*grid.cell_size,
                start[1]+(cell[1]-first[1])*grid.cell_size)
    support = grid.planning_support() if support is None else support
    allowed = grid.known & support & ~grid.blocked
    if no_go_exit:
        allowed = grid.known & support & ~grid.hazard & (grid.obstacle_distance >= grid.footprint_radius+grid.clearance)
    if not allowed[first]:
        return [], False
    costs, turns = grid.directional_costs(profile, uphill_limit, offsets)
    turn_cache, gentle_cache, edges, terminal_edges = {}, {}, {}, {}
    travel_weights = {}
    shared_objective = bool(getattr(grid, 'traversal_objective', None))
    # Recovery departure retains the established scalar escape cost. Its
    # directional exception is spatially scoped by observed_edge_reason; an
    # infinite primary label must never become a complete-route incumbent.
    uphill_first = shared_objective and bool(getattr(grid, 'uphill_first', False)) and not no_go_exit
    grid.last_route_search['objective'] = ('uphill burden then traversal cost' if uphill_first
        else 'scalar checked-exclusion escape' if no_go_exit and shared_objective else 'traversal cost')
    def weight(a,b,total):
        if not uphill_first:return total
        if not math.isfinite(total):return (math.inf,math.inf)
        from .traversal_objective import segment_cost
        parts=segment_cost(grid,a,b,breakdown=True)
        primary=parts.get('uphill',math.inf if 'invalid' in parts else 0.)
        return (primary,total) if math.isfinite(primary) else (math.inf,math.inf)
    def extend(value,travel,turn):
        return (value[0]+travel[0],value[1]+travel[1]+turn) if uphill_first else value+travel+turn
    infinity=(math.inf,math.inf) if uphill_first else math.inf
    zero=(0.,0.) if uphill_first else 0.
    def departure_cost(cell, angle):
        return (float(turns[cell])+(0. if shared_objective else 3.))*angle
    def staging(cell):
        if cell not in turn_cache:
            p = point(cell)
            turn_cache[cell] = not stage_reason(grid,p)
            gentle_cache[cell] = grid.max_support_slope(p)<=grid.motion_turn_limit
        return turn_cache[cell]
    def checked_turn(cell,a,incoming,outgoing):
        return gentle_cache.get(cell,False) or not turn_reason(grid,a,incoming,outgoing)
    def euclidean(cell):
        # Weighted A*: bounded executable-route search, not an optimality claim.
        return 3.*math.hypot(cell[0]-target[0], cell[1]-target[1])
    guidance = None
    climb_guidance = None
    if getattr(grid, 'footprint_costs_enabled', False):
        guidance, stats = terrain_guidance(allowed, costs, offsets, target)
        grid.last_route_search['queue_guidance'] = stats
        if uphill_first:
            from .traversal_objective import components
            climb_costs=np.stack([math.hypot(dx,dy)*components(grid,np.s_[:,:],dx,dy)[0]['uphill']
                for dx,dy in offsets],axis=-1)
            climb_costs=np.where(np.isfinite(costs),climb_costs,np.inf)
            climb_guidance, climb_stats=terrain_guidance(allowed,climb_costs,offsets,target)
            grid.last_route_search['climb_guidance']=climb_stats
    def heuristic(cell):
        value = euclidean(cell)
        if guidance is not None and math.isfinite(float(guidance[cell])):
            value = max(value, float(guidance[cell]))
        if uphill_first:
            climb=float(climb_guidance[cell]) if climb_guidance is not None else 0.
            return (climb if math.isfinite(climb) else 0.,value)
        return value
    def priority(value,cell):
        h=heuristic(cell)
        return (value[0]+h[0],value[1]+h[1]) if uphill_first else value+h
    origin = (*first, -1)
    terminal = (*target, len(offsets))
    best, parent = {origin: zero}, {}
    queue = [(heuristic(first), zero, origin)]
    end, expanded = None, 0
    found = False
    while queue and expanded < budget:
        _, value, node = heapq.heappop(queue)
        if best.get(node) != value:
            continue
        if node == terminal:
            end, found = parent[node], True
            break
        expanded += 1
        cell, incoming = node[:2], node[2]
        a = point(cell)
        can_turn = staging(cell)
        delta = np.subtract(goal, a)
        last_heading = yaw if incoming < 0 else math.atan2(offsets[incoming][1], offsets[incoming][0])
        difference = (0. if math.dist(a, goal) < 1e-8 else abs(math.atan2(
            math.sin(math.atan2(delta[1], delta[0])-last_heading),
            math.cos(math.atan2(delta[1], delta[0])-last_heading))))
        # A continuous final straight leg may start at gentle staging and end
        # on a slope without turning at the destination. Requiring a lattice
        # node in the goal's cell to pivot through a tiny sub-cell connector
        # can otherwise reject an executable exact destination indefinitely.
        # Cache per position: arrival headings do not change gentle staging or
        # the observer-relative freshness contract for this terminal edge.
        if can_turn or (cell == target and (math.dist(a, goal) < 1e-8 or difference < 1e-6)):
            if cell not in terminal_edges:
                clear = not observed_edge_reason(grid, a, goal, start,
                    fresh_radius=2. if require_fresh_start else None,
                    no_go_exit=no_go_exit, uphill_limit=uphill_limit)
                terminal_edges[cell] = weight(a,goal,terminal_travel_cost(grid, a, goal, profile, no_go_exit)) if clear else infinity
            turn_ok=(difference<1e-6 or checked_turn(cell,a,last_heading,math.atan2(delta[1],delta[0])))
            trial = extend(value,terminal_edges[cell],departure_cost(cell,difference)) if turn_ok else infinity
            if trial < best.get(terminal, infinity):
                best[terminal], parent[terminal] = trial, node
                heapq.heappush(queue, (trial, trial, terminal))
        for k, (dr, dc) in enumerate(offsets):
            if incoming >= 0 and k != incoming and not can_turn:
                continue
            nxt = (cell[0]+dr, cell[1]+dc)
            if not (0 <= nxt[0] < grid.n_cells and 0 <= nxt[1] < grid.n_cells and allowed[nxt]):
                continue
            b = point(nxt)
            heading = math.atan2(b[1]-a[1], b[0]-a[0])
            old_heading = yaw if incoming < 0 else math.atan2(offsets[incoming][1], offsets[incoming][0])
            angle = abs(math.atan2(math.sin(heading-old_heading), math.cos(heading-old_heading)))
            if angle>1e-6 and can_turn and not checked_turn(cell,a,old_heading,heading):
                continue
            if incoming < 0 and not can_turn:
                nominal = math.atan2(dc, dr)
                if angle > .10 or abs(math.atan2(math.sin(heading-nominal), math.cos(heading-nominal))) > 1e-6:
                    continue
            edge_weight, current = float(costs[nxt[0], nxt[1], k]), float(costs[cell[0], cell[1], k])
            if no_go_exit:
                if grid.no_go[nxt]: edge_weight = math.hypot(dr, dc)*float(grid.cost[nxt])
                if grid.no_go[cell]: current = math.hypot(dr, dc)*float(grid.cost[cell])
            if not math.isfinite(edge_weight) or not math.isfinite(current):
                continue
            state = (*nxt, k)
            if shared_objective:
                edge = cell, nxt
                if edge not in travel_weights:
                    travel_weights[edge] = weight(a,b,terminal_travel_cost(grid,a,b,profile,no_go_exit))
                edge_weight = travel_weights[edge]
            trial = extend(value,edge_weight,departure_cost(cell,angle))
            if trial >= best.get(state, infinity):
                continue
            edge = cell, nxt
            if edge not in edges:
                edges[edge] = not observed_edge_reason(grid, a, b, start,
                    fresh_radius=2. if require_fresh_start else None,
                    no_go_exit=no_go_exit, uphill_limit=uphill_limit)
            if not edges[edge]:
                continue
            best[state], parent[state] = trial, node
            heapq.heappush(queue, (priority(trial,nxt), trial, state))
    exhausted = not found and bool(queue) and expanded >= budget
    # The budget limits exploration, not use of a previously fully checked exit.
    # This is a bounded-search incumbent, never an optimality claim.
    if not found and terminal in best:
        end, found = parent[terminal], True
    grid.last_route_search.update(expansions=expanded, budget_exhausted=exhausted,
        terminal_connections=len(terminal_edges),
        terminal_total_cost=(best[terminal][1] if uphill_first else best[terminal]) if terminal in best else None,
        terminal_uphill_burden=best[terminal][0] if uphill_first and terminal in best else None,
        goal_selection=('best checked complete route at search budget' if exhausted and found
                        else 'priced goal popped from queue' if found else 'no checked complete route'))
    if not found:
        if not allow_frontier:
            return [], False
        candidates = []
        span = math.ceil((grid.footprint_radius+.8)/grid.cell_size)
        for node, value in best.items():
            if node == terminal:
                continue
            cell = node[:2]
            p = point(cell)
            if math.dist(start, p) < 1. or not staging(cell):
                continue
            r, c = cell
            patch = grid.known[max(0, r-span):r+span+1, max(0, c-span):c+span+1]
            unknown = int((~patch).sum())
            if unknown < 3:
                continue
            penalty = sum(25.*max(0., 1.-math.dist(p, old)/4.) for old in frontier_penalties)
            score = 3.*(euclidean(first)-euclidean(cell))+.03*min(unknown, 40)-.025*(value[1] if uphill_first else value)-penalty
            candidates.append((score, node))
        if not candidates:
            return [], False
        end = max(candidates)[1]
    chain = [end]
    while chain[-1] != origin:
        chain.append(parent[chain[-1]])
    path = [point(n[:2]) for n in reversed(chain)]
    path[0] = tuple(start)
    if found:
        path.append(tuple(goal))
    path = compact_path(path)
    # Retain the priced terminal geometry. A geometry-only corner merge here
    # could replace the chosen connector with a more expensive one after search.
    if motion_route_reason(grid, path, yaw):
        return [], False
    return path, found
