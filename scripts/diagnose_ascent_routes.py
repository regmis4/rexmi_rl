#!/usr/bin/env python3
"""Read-only, frozen-survey route diversity study; no physical success claim."""
import argparse
import csv
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import time
import types

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
# Load the navigation modules without importing Isaac or registering tasks.
for name, path in [('route_diagnostic', ROOT/'source/rexmi_rl'),
                   ('route_diagnostic.nav', ROOT/'source/rexmi_rl/nav')]:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module
Map = importlib.import_module('route_diagnostic.nav.ascent_map').AscentTerrainMap
load_prior = importlib.import_module('route_diagnostic.nav.terrain_prior').load_prior
routes = importlib.import_module('route_diagnostic.nav.staged_routes')
OFFSETS = importlib.import_module('route_diagnostic.nav.ascent_routes').OFFSETS


def load_grid(prior):
    with np.load(prior, allow_pickle=False) as data:
        grid = Map(world_size=data['known'].shape[0]*float(data['cell_size']),
                   cell_size=float(data['cell_size']), origin=data['origin'],
                   footprint_radius=float(data['footprint_radius']),
                   clearance=float(data['clearance_m']),
                   half_length=float(data['half_length']), half_width=float(data['half_width']))
        metadata = json.loads(str(data['terrain_prior_json'].item()))
    grid.uphill_limit_deg = 33.
    grid.side_limit_deg = 25.
    grid.motion_turn_limit = 12.
    grid.planning_grade_reserve = 0.
    grid.diagonal_ascent_preference = True
    load_prior(grid, prior, task=metadata['task'], seed=42)
    return grid


def initial_state(run):
    events = [json.loads(line) for line in (run/'run.events.jsonl').read_text().splitlines()]
    event = next(e for e in events if e.get('event') == 'route_comparison')
    selected = next(c for c in event['candidates'] if c.get('selected'))
    start = selected['route_path'][0]
    with (run/'run.csv').open() as stream:
        rows = [r for r in csv.DictReader(stream) if float(r['sim_time']) <= event['time']]
    row = min(rows, key=lambda r: math.dist(start, (float(r['x']), float(r['y']))))
    error = math.dist(start, (float(row['x']), float(row['y'])))
    if error > 1e-6:
        raise ValueError('Cannot match planner origin to recorded pose/yaw')
    return start, float(row['yaw']), float(row['sim_time']), selected


def portals(grid, start, goal, support):
    """One gentle elevated staging point per bearing sector, from survey only.

    Exterior elevation is defined relative to the exact destination's observed
    height (within 25 cm). This is a portal sampling rule, not a terrain gate.
    All existing footprint, staging and route checks still apply.
    """
    target_height = float(grid.height[grid.world_to_cell(*goal)])
    mask = (grid.known & support & ~grid.blocked & np.isfinite(grid.slope)
            & (grid.slope <= math.tan(math.radians(12.)))
            & (grid.height >= target_height-.25))
    choices = [[] for _ in range(8)]
    for cell in np.argwhere(mask):
        point = grid.cell_to_world(*cell)
        delta = np.subtract(point, grid.origin)
        if np.linalg.norm(delta) > grid.terrain_prior['radius_m']-grid.footprint_radius-grid.clearance:
            continue
        angle = math.atan2(delta[1], delta[0])
        sector = int(math.floor((angle+math.pi/8)/(math.pi/4))) % 8
        offset = abs(math.atan2(math.sin(angle-sector*math.pi/4), math.cos(angle-sector*math.pi/4)))
        # Prefer nearby supported rim entries while covering the sector centre.
        rank = math.dist(start, point)+2.*offset
        choices[sector].append((rank, tuple(map(float, point))))
    result = []
    for sector, options in enumerate(choices):
        tested = 0
        point = None
        for _, candidate in sorted(options):
            tested += 1
            if (grid.max_support_slope(candidate) <= 12.
                    and not grid.turn_support_reason(candidate)
                    and grid.segment_clear(candidate, candidate, require_fresh=False)):
                point = candidate
                break
        result.append(dict(sector=sector, bearing_deg=45*sector, portal=point,
                           raw_candidates=len(options), staging_candidates_checked=tested))
    return result


def cost_components(grid, path, yaw):
    totals = dict(terrain=0., along_grade=0., cross_grade=0., uphill=0., downhill=0.,
                  sidehill=0., edge_room=0., failure=0., untested=0., turn=0., diagonal=0.)
    previous = grid.heading_bin(yaw)
    for a, b in zip(path, path[1:]):
        delta = np.subtract(b, a)
        count = max(1, math.ceil(float(np.max(np.abs(delta)))/grid.cell_size-1e-8))
        dr, dc = delta/grid.cell_size/count
        norm = math.hypot(dr, dc)
        if norm < 1e-9:
            continue
        heading = grid.heading_bin(math.atan2(dc, dr))
        for t in np.linspace(0, 1, count+1)[1:]:
            cell = grid.world_to_cell(*(np.asarray(a)+t*delta))
            along = (grid.gx[cell]*dr+grid.gy[cell]*dc)/norm
            across = abs(-grid.gx[cell]*dc+grid.gy[cell]*dr)/norm
            grade, side = math.degrees(math.atan(along)), math.degrees(math.atan(across))
            values = dict(terrain=float(grid.cost[cell]), along_grade=2*abs(along), cross_grade=4*across,
                uphill=12*(max(0., grade-15)/20)**2, downhill=4*(max(0., -grade-20)/15)**2,
                sidehill=8*(side/35)**2,
                edge_room=24*max(0., 1-grid.steep_distance[cell]/(grid.footprint_radius+.3))**2,
                failure=6*float(grid.failures[*cell, heading]),
                untested=.3*float(grid.successes[*cell, heading] == 0),
                diagonal=float(grid.diagonal_preference_cost(along, across)))
            for key, value in values.items():
                totals[key] += norm*value
            turn = min((heading-previous) % 8, (previous-heading) % 8)
            totals['turn'] += .8*turn*(1+2*min(1., grid.slope[cell]/math.tan(math.radians(35)))**2)
            previous = heading
    return totals


def describe(grid, path, yaw):
    reason = routes.motion_route_reason(grid, path, yaw)
    edge_rejections, retreat_rejections, legs = [], [], []
    clearance = math.inf
    for index, (a, b) in enumerate(zip(path, path[1:])):
        rejection = routes.observed_edge_reason(grid, a, b, path[0], fresh_radius=None)
        reverse = routes.observed_edge_reason(grid, b, a, path[-1], fresh_radius=None)
        if rejection:
            edge_rejections.append(dict(edge=index, reason=rejection))
        if reverse:
            retreat_rejections.append(dict(edge=index, reason=reverse))
        for t in np.linspace(0, 1, max(2, math.ceil(math.dist(a, b)/.08)+1)):
            cell = grid.world_to_cell(*(np.asarray(a)+t*np.subtract(b, a)))
            clearance = min(clearance, float(grid.obstacle_distance[cell]))
        profile = grid.route_profile([a, b])
        if profile['max_uphill_deg'] > 15. and profile['length_m'] > 1.:
            legs.append(dict(start=a, end=b, heading_deg=math.degrees(math.atan2(b[1]-a[1], b[0]-a[0])), **profile))
    profile = grid.route_profile(path)
    components = cost_components(grid, path, yaw)
    route_cost = grid.route_cost(path, yaw)
    if math.isfinite(route_cost) and abs(route_cost-sum(components.values())) > 1e-5:
        raise ValueError('Cost decomposition disagrees with production route cost')
    penalties = dict(uphill_peak=.5*max(0., profile['max_uphill_deg']-28)**2,
                     cross_peak=3*max(0., profile['max_side_slope_deg']-15)**2)
    return dict(path=path, geometry_feasible=not (reason or edge_rejections),
                turn_rejection=reason, edge_rejections=edge_rejections,
                retreat_geometry_feasible=not retreat_rejections, retreat_rejections=retreat_rejections,
                live_admission='not evaluated: survey observations are deliberately stale',
                physical_success='unknown; no probability or stability claim',
                route_cost=route_cost if math.isfinite(route_cost) else None,
                cost_components=components, peak_penalties=penalties,
                ranking_score=route_cost+sum(penalties.values()) if math.isfinite(route_cost) else None,
                minimum_centre_obstacle_distance_m=clearance,
                minimum_circle_margin_m=clearance-grid.footprint_radius-grid.clearance,
                turns=[dict(position=p, footprint_inclination_deg=grid.max_support_slope(p)) for p in path[1:-1]],
                climb_legs=legs, **profile)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', type=Path, default=ROOT/'outputs/ascent_full_survey/trial01/run.prior.npz')
    parser.add_argument('--run', type=Path, default=ROOT/'outputs/ascent_behavior_tree/trial26')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--grade-ladder', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing to overwrite diagnostic')
    grid = load_grid(args.prior)
    start, yaw, pose_time, baseline = initial_state(args.run)
    goal = (13., 0.)
    support = grid.planning_support()
    evidence_before = [a.copy() for a in (grid.no_go, grid.approach_blocked, grid.failures, grid.successes, grid.fresh)]
    result = dict(source='labeled simulator development survey; read-only geometry diagnostic',
        prior_sha256=hashlib.sha256(args.prior.read_bytes()).hexdigest(),
        start=start, yaw=yaw, recorded_pose_time=pose_time, goal=goal,
        limits=dict(uphill=33., cross=25., overall=35., staging=12.),
        search_interpretation='Weighted A* (3x distance), first-found. Early exact-goal connector stops search immediately; its edge cost is not compared with other queued solutions. No optimality claim.',
        limitations=['One portal per sector does not prove a sector unreachable.',
                     'Portal bearing does not identify actual climbing corridor.',
                     'No fresh live admission or physical qualification.'],
        baseline=describe(grid, baseline['route_path'], yaw), candidates=[])
    result['files_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__).resolve()]+sorted((ROOT/'source/rexmi_rl/nav').glob('*.py'))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    choices = ([dict(uphill_cap=cap) for cap in (28., 30., 32., 33.)] if args.grade_ladder
               else portals(grid, start, goal, support))
    for candidate in choices:
        result['candidates'].append(candidate)
        if args.grade_ladder:
            began = time.monotonic()
            found, complete = routes.staged_route(grid, start, goal, yaw, offsets=OFFSETS,
                support=support, profile='balanced', uphill_limit=candidate['uphill_cap'], require_fresh_start=False,
                allow_frontier=False, max_expansions=60000)
            candidate.update(complete=complete, elapsed_s=time.monotonic()-began, **grid.last_route_search)
            if complete:
                candidate.update(describe(grid, found, yaw))
            else:
                candidate['rejection'] = 'search budget exhausted' if candidate['budget_exhausted'] else 'open set exhausted; no route found by heading search'
            save()
            print(json.dumps(candidate), flush=True)
            continue
        if candidate['portal'] is None:
            candidate['rejection'] = 'no sampled gentle exterior portal'
            save()
            continue
        path = []
        candidate['searches'] = []
        for label, origin, destination in [('approach', start, candidate['portal']), ('onward', candidate['portal'], goal)]:
            heading = yaw if not path else math.atan2(path[-1][1]-path[-2][1], path[-1][0]-path[-2][0])
            began = time.monotonic()
            found, complete = routes.staged_route(grid, origin, destination, heading, offsets=OFFSETS,
                support=support, profile='balanced', uphill_limit=33., require_fresh_start=False,
                allow_frontier=False, max_expansions=60000)
            record = dict(leg=label, complete=complete, path=found, elapsed_s=time.monotonic()-began,
                          **grid.last_route_search)
            candidate['searches'].append(record)
            print(json.dumps(dict(sector=candidate['sector'], **record)), flush=True)
            if not complete:
                candidate['rejection'] = label+': '+('search budget exhausted' if record['budget_exhausted'] else 'no route found by heading search')
                break
            path = found if not path else routes.compact_path(path+found[1:])
        else:
            candidate.update(describe(grid, path, yaw))
        save()
    for before, after in zip(evidence_before, (grid.no_go, grid.approach_blocked, grid.failures, grid.successes, grid.fresh)):
        assert np.array_equal(before, after), 'Diagnostic mutated mission evidence'
    result['evidence_unchanged'] = True
    save()


if __name__ == '__main__':
    main()
