"""Ascent-first checkpoint search; terrain effort is not success probability.

Exit means a sampled arrival-heading state with a checked gentle continuation
all the way to the exact goal. It is not an independently detected crater rim.
The graph is shared by both searches; the user's sketch is never an input.
"""
import heapq
import math
import time
from collections import Counter, defaultdict
import numpy as np
from . import stage_graph
from .purple_corridor import integral
from .swept_footprint import oriented_swept_cells
from .turn_staging import turn_reason

NAME = 'ascent-first-v1'
MAX_TRANSITIONS = 100000
MAX_STATES = 60000


def leg_metrics(g, a, b, reference):
    heading = math.atan2(b[1]-a[1], b[0]-a[0])
    cells = oriented_swept_cells(g, a, b, heading)
    gentle = cells is not None and bool(g.known[cells].all())
    if gentle:
        slopes = g.slope[cells]
        gentle = bool(np.isfinite(slopes).all() and
            np.degrees(np.arctan(slopes)).max() <= g.motion_turn_limit + 1e-6)
    return dict(ascent=integral(g, a, b, reference), exterior=integral(g, a, b),
                length=math.dist(a, b), gentle=gentle)


def build(g, start, goal, physical_check, *, candidate_source=None, wall_budget=180.):
    reference = tuple(float(g.surface[g.world_to_cell(*p)]) for p in (start, goal))
    # Geometry and candidate inventory are shared; legacy directional weights
    # must neither rank nor discard links belonging to this objective.
    graph = stage_graph.build(g, start, goal, physical_check,
        candidate_source=candidate_source, wall_budget=wall_budget,
        cost_model=lambda grid, a, b: {'displayed': integral(grid, a, b)})
    graph['reference'] = reference
    graph['metric_budget_exhausted'] = False
    for (i, j), edge in graph['edges'].items():
        if time.perf_counter() >= graph['deadline']:
            graph['metric_budget_exhausted'] = True
            break
        edge.update(leg_metrics(g, graph['points'][i], graph['points'][j], reference))
    return graph


def transitions(g, graph, yaw):
    """Build actual heading transitions once; no independent-route splice."""
    edges = graph['edges']; outgoing = defaultdict(list)
    for i, j in edges:
        outgoing[i].append(j)
    initial = (-1, 0)
    states = [initial] + list(edges)
    forward = defaultdict(list); reverse = defaultdict(list)
    rejected = Counter(); checks = 0; exhausted = False
    for state in states:
        if time.perf_counter() >= graph['deadline']:
            exhausted = True; break
        previous, current = state
        if current == 1:  # Exact goal is terminal; no final orientation required.
            continue
        incoming = yaw if previous < 0 else edges[state]['heading']
        for target in outgoing[current]:
            if checks >= MAX_TRANSITIONS or time.perf_counter() >= graph['deadline']:
                exhausted = True; break
            checks += 1
            following = (current, target); heading = edges[following]['heading']
            delta = abs(math.atan2(math.sin(heading-incoming), math.cos(heading-incoming)))
            reason = turn_reason(g, graph['points'][current], incoming, heading) if delta > 1e-6 else ''
            graph['turns'][previous, current, target, yaw if previous < 0 else None] = reason
            if reason:
                rejected[reason] += 1
                continue
            forward[state].append(following)
            reverse[following].append(state)
        if exhausted:
            break
    return forward, reverse, dict(transition_checks=checks, turn_rejections=dict(rejected),
        transition_budget_exhausted=exhausted)


def search(g, graph, yaw):
    """Rank (ascent effort, gentle-suffix effort, total length), lexicographically.

    Ascent stops irreversibly at the first state with a certified suffix. Both
    searches retain the arrival heading, including the climb/suffix joining turn.
    No route is returned from a partially built transition graph.
    """
    began = time.perf_counter()
    stats = dict(objective=NAME, complete=False, optimal=False, budget_exhausted=False,
        reference=list(graph['reference']), exit_definition='first sampled heading state with checked gentle suffix',
        backward_expansions=0, forward_expansions=0)
    def finish(path=()):
        stats['search_seconds'] = time.perf_counter()-began
        return list(path), stats
    if (graph.get('build_budget_exhausted') or graph.get('metric_budget_exhausted')
            or time.perf_counter() >= graph['deadline']):
        stats.update(budget_exhausted=True, reason='graph or metric budget exhausted')
        return finish()
    edges = graph['edges']
    for edge in edges.values():
        if not all(math.isfinite(edge[k]) and edge[k] > 0 for k in ('ascent', 'exterior', 'length')):
            stats['reason'] = 'invalid nonpositive edge metric'
            return finish()
    forward, reverse, build_stats = transitions(g, graph, yaw)
    stats.update(build_stats)
    if build_stats['transition_budget_exhausted']:
        stats.update(budget_exhausted=True, reason='heading graph budget exhausted')
        return finish()
    # Reverse directed search. A predecessor can enter this certificate only
    # through a gentle outgoing edge and its already checked heading transition.
    suffix = {state: (0., 0.) for state in edges if state[1] == 1}
    queue = [(value, state) for state, value in suffix.items()]
    heapq.heapify(queue); next_state = {}
    while queue:
        if time.perf_counter() >= graph['deadline'] or stats['backward_expansions'] >= MAX_STATES:
            stats.update(budget_exhausted=True, reason='suffix search budget exhausted')
            return finish()
        value, state = heapq.heappop(queue)
        if value != suffix[state]:
            continue
        stats['backward_expansions'] += 1
        if state == (-1, 0) or not edges[state]['gentle']:
            continue
        edge = edges[state]
        candidate = (value[0]+edge['exterior'], value[1]+edge['length'])
        for predecessor in reverse[state]:
            if candidate < suffix.get(predecessor, (math.inf, math.inf)):
                suffix[predecessor] = candidate; next_state[predecessor] = state
                heapq.heappush(queue, (candidate, predecessor))
    stats['certified_exit_states'] = len(suffix)
    # Positive ascent costs allow one best (ascent,length) label per heading.
    initial = (-1, 0); best = {initial: (0., 0.)}; parent = {}
    queue = [((0., 0.), initial)]; winner = None; rank = None
    while queue:
        if time.perf_counter() >= graph['deadline'] or stats['forward_expansions'] >= MAX_STATES:
            stats.update(budget_exhausted=True, reason='ascent search budget exhausted')
            return finish()
        value, state = heapq.heappop(queue)
        if value != best[state]:
            continue
        if rank is not None and value[0] > rank[0]:
            break
        stats['forward_expansions'] += 1
        if state in suffix:
            score = (value[0], suffix[state][0], value[1]+suffix[state][1])
            if rank is None or score < rank:
                winner, rank = state, score
            continue  # First entry, never leave and re-enter the ascent phase.
        for target in forward[state]:
            edge = edges[target]
            trial = (value[0]+edge['ascent'], value[1]+edge['length'])
            if trial < best.get(target, (math.inf, math.inf)):
                best[target] = trial; parent[target] = state
                heapq.heappush(queue, (trial, target))
    if winner is None:
        stats['reason'] = 'no complete heading-compatible route in sampled graph'
        return finish()
    prefix = [winner]; state = winner
    while state != initial:
        state = parent[state]; prefix.append(state)
    prefix.reverse(); states = prefix.copy(); state = winner
    while state[1] != 1:
        state = next_state[state]; states.append(state)
    path = [graph['points'][state[1]] for state in states]
    stats.update(complete=True, optimal=True, rank=list(rank),
        exit_index=len(prefix)-1, exit_state=list(winner), exit_point=path[len(prefix)-1],
        path_states=[list(state) for state in states], reason='checked ascent plus gentle goal continuation')
    return finish(path)


def route_reason(g, path, yaw, exit_index, start):
    """Recheck the entire route and suffix on current map, including join turn."""
    from .staged_routes import observed_edge_reason
    from .swept_footprint import oriented_sweep_reason
    if len(path) < 2 or not 0 <= exit_index < len(path):
        return 'missing ascent-first route or exit'
    incoming = yaw
    for i, (a, b) in enumerate(zip(path, path[1:])):
        heading = math.atan2(b[1]-a[1], b[0]-a[0])
        delta = abs(math.atan2(math.sin(heading-incoming), math.cos(heading-incoming)))
        reason = observed_edge_reason(g, a, b, start) or oriented_sweep_reason(
            g, a, b, heading, require_fresh=False, commanded=True, circular_support=False)
        if not reason and delta > 1e-6:
            reason = turn_reason(g, a, incoming, heading)
        if reason:
            return reason
        if i >= exit_index and not leg_metrics(g, a, b, (0., 0.))['gentle']:
            return 'gentle goal continuation changed'
        incoming = heading
    return ''


def describe_route(g, path, exit_index, reference):
    """Audit climbing separately; sampled heights use full footprint means."""
    ascent = exterior = total = base = distance = 0.
    profile = []; lengths = []; peak = 0.
    for i, (a, b) in enumerate(zip(path, path[1:])):
        metrics = leg_metrics(g, a, b, reference)
        ascent += metrics['ascent'] if i < exit_index else 0.
        exterior += metrics['exterior'] if i >= exit_index else 0.
        base += metrics['exterior']; total += metrics['length']; lengths.append(metrics['length'])
        peak = max(peak, integral(g, a, b, peak=True))
        points = np.linspace(a, b, max(2, math.ceil(metrics['length']/.1)+1))
        if not profile:
            profile.append(dict(distance_m=0.,height_m=stage_graph.mean_height(g,a),phase='ascent' if exit_index else 'gentle_suffix'))
        for u, v in zip(points, points[1:]):
            distance += math.dist(u, v)
            profile.append(dict(distance_m=distance,height_m=stage_graph.mean_height(g,v),
                phase='ascent' if i < exit_index else 'gentle_suffix'))
    heights = np.asarray([row['height_m'] for row in profile])
    delta = np.diff(heights)
    return dict(rank=[ascent,exterior,total],length_m=total,base_cost=base,
        mean_cost=base/total if total else None,peak_cost=peak,height_profile=profile,
        footprint_height_drawdown_m=float(np.max(np.maximum.accumulate(heights)-heights)) if len(heights) else 0.,
        cumulative_height_loss_m=float(np.maximum(-delta,0).sum()),
        cumulative_height_gain_m=float(np.maximum(delta,0).sum()),
        height_attainment={str(z):next((row for row in profile if row['height_m']>=z),None) for z in (1.,2.,3.,4.)},
        shortest_leg_m=min(lengths) if lengths else None,
        short_legs=sum(v<2*g.footprint_radius for v in lengths))
