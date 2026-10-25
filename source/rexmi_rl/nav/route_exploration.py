"""Explicit characterization coverage; never a probability or mission no-go."""
import ast
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .staged_routes import compact_path
from .command_selection import FORWARD_SPEED_PROFILE
from .ascent_map import DIRECTIONAL_EXIT_PROFILE
from .movement import UPHILL_SPEED_RESPONSE_PROFILE
from .recovery_continuation import RECOVERY_CONTINUATION_PROFILE


# Explicitly reviewed scheduling witnesses for the forward-speed experiment.
# These hashes never establish capability under the new executor.
LEGACY_SPEED_HISTORY = {
    ('9eea607493e9325e2cfed8ae1ee8b8eb5f1a23d9ab98fca6bd02f24a2fc08fad',
     '4146ce3c9429353e4194f42b4ce83f1c5b3a1baf0f3a4823390cfab747925e8c'),
    ('9eea607493e9325e2cfed8ae1ee8b8eb5f1a23d9ab98fca6bd02f24a2fc08fad',
     '1910278202632e25aed03f1ed5c51e487b98bf6e8441b140b50461d5a61d48c2'),
}
LEGACY_EXIT_GEOMETRY = {
    'ascent_map':'a8bed2eaddbfbc3b3e197c05f98468f463106660bbeca2907c347637dbd401f1',
    'swept_footprint':'b3ca6484b909478c239b1ffe48dcccc061872930d60034dd726684d268afac71',
}
LEGACY_PROFILES = {
    'staged-teleop-v27-checked-forward-speed':FORWARD_SPEED_PROFILE,
    'staged-teleop-v28-directional-recovery-exit':dict(FORWARD_SPEED_PROFILE,**DIRECTIONAL_EXIT_PROFILE),
    'staged-teleop-v29-paired-launch':dict(FORWARD_SPEED_PROFILE,**DIRECTIONAL_EXIT_PROFILE),
    'staged-teleop-v30-uphill-speed-response':dict(FORWARD_SPEED_PROFILE,**DIRECTIONAL_EXIT_PROFILE,**UPHILL_SPEED_RESPONSE_PROFILE),
    'staged-teleop-v31-functional-recovery':dict(FORWARD_SPEED_PROFILE,**DIRECTIONAL_EXIT_PROFILE,**UPHILL_SPEED_RESPONSE_PROFILE,**RECOVERY_CONTINUATION_PROFILE),
}


def legacy_speed_history(run, previous, current):
    key = 'source/rexmi_rl/nav/movement.py'
    if previous['sources'].get(key) == current['sources'].get(key):
        return False
    names = ('movement','ascent_behavior')
    keys = ['source/rexmi_rl/nav/'+name+'.py' for name in names]
    pair = tuple(previous['sources'].get(k) for k in keys)
    protocol = current['capability_profile'].get('execution_protocol')
    additions = LEGACY_PROFILES.get(protocol)
    if (pair not in LEGACY_SPEED_HISTORY or additions is None
            or any(current['capability_profile'].get(k) != v for k,v in additions.items())
            or any(k in previous['capability_profile'] for k in additions)):
        raise ValueError('Exploration history mismatch: movement')
    if protocol in ('staged-teleop-v28-directional-recovery-exit','staged-teleop-v29-paired-launch','staged-teleop-v30-uphill-speed-response','staged-teleop-v31-functional-recovery'):
        for name,expected in LEGACY_EXIT_GEOMETRY.items():
            key='source/rexmi_rl/nav/'+name+'.py'
            if previous['sources'].get(key) != expected:
                raise ValueError('Legacy directional-exit geometry source mismatch')
        names += tuple(LEGACY_EXIT_GEOMETRY)
        keys = ['source/rexmi_rl/nav/'+name+'.py' for name in names]
    for name,key in zip(names,keys):
        archived = run/'run.sources'/key
        live = Path(__file__).with_name(name+'.py')
        if (not archived.is_file() or hashlib.sha256(archived.read_bytes()).hexdigest() != previous['sources'][key]
                or hashlib.sha256(live.read_bytes()).hexdigest() != current['sources'].get(key)):
            raise ValueError('Legacy scheduling history has mismatched archived or current source')
    return True


def survey_portals(grid, start, goal, support):
    """At most one map-derived gentle exterior staging point in each sector."""
    target = grid.world_to_cell(*goal)
    if target is None or not grid.known[target] or not grid.terrain_prior:
        return []
    mask = (grid.known & support & ~grid.blocked & np.isfinite(grid.slope)
            & (grid.slope <= math.tan(math.radians(12.)))
            & (grid.height >= float(grid.height[target])-.25))
    choices = [[] for _ in range(8)]
    for cell in np.argwhere(mask):
        point = grid.cell_to_world(*cell)
        delta = np.subtract(point, grid.origin)
        if np.linalg.norm(delta) > grid.terrain_prior['radius_m']-grid.footprint_radius-grid.clearance:
            continue
        angle = math.atan2(delta[1], delta[0])
        sector = int(math.floor((angle+math.pi/8)/(math.pi/4))) % 8
        offset = abs(math.atan2(math.sin(angle-sector*math.pi/4), math.cos(angle-sector*math.pi/4)))
        choices[sector].append((math.dist(start, point)+2.*offset, tuple(map(float, point))))
    result = []
    for sector, options in enumerate(choices):
        point, tested = None, 0
        for _, candidate in sorted(options):
            tested += 1
            if (grid.max_support_slope(candidate) <= 12.
                    and not grid.turn_support_reason(candidate)
                    and grid.segment_clear(candidate, candidate, require_fresh=False)):
                point = candidate
                break
        result.append(dict(sector=sector, portal=point, raw_candidates=len(options), staging_checks=tested))
    return result


def first_climb(grid, path):
    """First continuous metre above 15° uphill on a straight route section.

    The descriptor covers the entire sustained section, excluding access and
    rim detours. It is experiment coverage, not a hard navigation exclusion.
    """
    path = compact_path(path)
    for a, b in zip(path, path[1:]):
        delta = np.subtract(b, a)
        length = float(np.linalg.norm(delta))
        if length < 1.:
            continue
        axis = delta/length
        count = math.ceil(length/(grid.cell_size*.4))
        run = None
        for index in range(count+1):
            t = index/count
            point = np.asarray(a)+t*delta
            cell = grid.world_to_cell(*point)
            steep = (cell is not None and grid.known[cell]
                     and math.degrees(math.atan(grid.gx[cell]*axis[0]+grid.gy[cell]*axis[1])) > 15.)
            if steep and run is None:
                run = t
            if run is not None and (not steep or index == count):
                end = t if steep else max(run, (index-1)/count)
                if (end-run)*length >= 1.:
                    return dict(start=(np.asarray(a)+run*delta).tolist(), end=(np.asarray(a)+end*delta).tolist(),
                                heading=math.atan2(axis[1], axis[0]), approach_start=list(a), approach_end=list(b))
                run = None
    return None


def launch_description(grid, path):
    """Compare launch preparation on one snapshot, not traversal probability.

    Compaction uses the executor's turn semantics. Count route bends through
    the turn into the first sustained climb, excluding initial body alignment.
    """
    path = compact_path(path)
    climb = first_climb(grid, path)
    if climb is None:
        return None
    index = next(i for i,(a,b) in enumerate(zip(path,path[1:]))
                 if a == tuple(climb['approach_start']) and b == tuple(climb['approach_end']))
    launch = path[index]
    staging = next((i for i in range(index+1,len(path))
                    if grid.max_support_slope(path[i]) <= grid.motion_turn_limit
                    and not grid.turn_support_reason(path[i])
                    and grid.segment_clear(path[i],path[i],require_fresh=False)), None)
    return dict(corridor=climb, launch=list(launch), pre_climb_bends=index,
                bend_definition='compacted route bends through climb entry; initial body alignment excluded',
                launch_inclination_deg=grid.max_support_slope(launch),
                launch_support_reason=grid.turn_support_reason(launch),
                launch_geometry_clear=bool(grid.segment_clear(launch,launch,require_fresh=False)),
                approach_prefix=path[:index+1],
                climb_prefix=path[:staging+1] if staging is not None else [],
                staging_index=staging,
                predicted_approach_heading_deg=math.degrees(climb['heading']),
                physical_reachability='unvalidated; geometric launch comparison only')


def corridor_overlap(a, b):
    if not a or not b:
        return None
    if not all(np.isfinite([*c['start'], *c['end'], c['heading']]).all() for c in (a, b)):
        return None
    difference = abs(math.atan2(math.sin(a['heading']-b['heading']), math.cos(a['heading']-b['heading'])))
    if difference > math.radians(10.):
        return None
    # Symmetric overlapping centreline bands; sub-cell shifts cannot create a
    # new class. Distinct parallel corridors farther than 1.2 m stay distinct.
    details = []
    for reference, candidate in ((a, b), (b, a)):
        origin = np.asarray(reference['start'])
        axis = np.array([math.cos(reference['heading']), math.sin(reference['heading'])])
        side = np.array([-axis[1], axis[0]])
        offsets = np.asarray([candidate['start'], candidate['end']])-origin
        longitudinal = offsets@axis
        lo = max(0., min(longitudinal))
        hi = min(math.dist(reference['start'], reference['end']), max(longitudinal))
        overlap = hi-lo
        if overlap < 1. or abs(longitudinal[1]-longitudinal[0]) < 1e-9:
            return None
        lateral = offsets@side
        separation = max(abs(lateral[0]+(v-longitudinal[0])/(longitudinal[1]-longitudinal[0])*(lateral[1]-lateral[0])) for v in (lo,hi))
        if separation > 1.2:
            return None
        details.append(dict(overlap_m=overlap, reference_fraction=overlap/math.dist(reference['start'],reference['end']),
                            maximum_lateral_separation_m=separation))
    return details


def same_corridor(a, b):
    return corridor_overlap(a, b) is not None


def same_entrance(a, b):
    return (same_corridor(a,b) and 'approach_start' in a and 'approach_start' in b
            and math.dist(a['approach_start'],b['approach_start']) <= 1.2)


def verify_guard_source(run, provenance, *, legacy=False):
    key = 'source/rexmi_rl/nav/ascent_behavior.py'
    path = run/'run.sources'/key
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != provenance['sources'].get(key):
        raise ValueError('Entrance history missing or mismatched archived guard source')
    def methods(source):
        tree = ast.parse(source)
        # Legacy full-source hashes were separately verified against the
        # reviewed pair. Only these two methods determine entrance censorship;
        # the newer stopping guard adds pure probe controls and speed selection.
        names = ('_remaining_corridor','_straight_clear') if legacy else (
            '_remaining_corridor','_straight_clear','_guard','_stopping_reason')
        return {node.name:ast.dump(node,include_attributes=False) for node in ast.walk(tree)
                if isinstance(node,ast.FunctionDef) and node.name in names}
    previous = methods(path.read_text())
    current = methods(Path(__file__).with_name('ascent_behavior.py').read_text())
    if len(previous) != (2 if legacy else 4) or previous != current:
        raise ValueError('Entrance history execution-geometry guard mismatch')


def entrance_record(corridor, outcome, movements, rows):
    """A censored entrance is scheduled once; its steep climb stays unknown."""
    reason = 'remaining straight corridor unavailable from measured pose'
    if not corridor or outcome.get('outcome') != 'censored' or outcome.get('reason') != reason:
        return None
    terminals = [m for m in movements if m.get('outcome') == 'censored' and m.get('reason') == reason
                 and abs(m.get('time',-math.inf)-outcome['end_time']) < 1e-6]
    if len(terminals) != 1:
        raise ValueError('Entrance history needs one associated terminal movement')
    movement = terminals[0]
    executed = movement.get('executed_traversal')
    if not executed or executed.get('maneuver') != 'DRIVE_FORWARD':
        return None
    if executed != outcome.get('executed_traversal') or movement.get('physical') or outcome.get('physical'):
        raise ValueError('Entrance history mismatched terminal traversal')
    fields = [*executed['start_position'],*executed['last_position'],executed['heading'],executed['duration_s'],
              executed['first_emitted_time'],executed['interval_end_time'],*outcome['position'],*movement['position'],movement['time']]
    if not np.isfinite(fields).all() or math.dist(movement['position'],outcome['position']) > 1e-6:
        raise ValueError('Entrance history invalid measured values')
    if not (outcome['start_time'] <= executed['first_emitted_time'] < executed['interval_end_time'] <= outcome['end_time']
            and executed['duration_s'] > 0.):
        raise ValueError('Entrance history invalid execution interval')
    angle = abs(math.atan2(math.sin(executed['heading']-corridor['heading']),math.cos(executed['heading']-corridor['heading'])))
    axis = np.array([math.cos(corridor['heading']), math.sin(corridor['heading'])])
    start,end = np.asarray(executed['start_position']),np.asarray(executed['last_position'])
    advance = float((end-start)@axis)
    offsets = np.asarray([start,end,np.asarray(outcome['position'])])-np.asarray(corridor['approach_start'])
    separation = float(np.max(np.abs(offsets@np.array([-axis[1],axis[0]]))))
    to_steep = float((np.asarray(corridor['start'])-np.asarray(outcome['position']))@axis)
    to_end = float((np.asarray(corridor['end'])-np.asarray(outcome['position']))@axis)
    if angle > math.radians(10.) or advance < .4 or separation > 1.2 or to_steep > 1. or to_end < 0.:
        return None
    emitted = [r for r in rows if abs(float(r['sim_time'])-executed['first_emitted_time']) < 1e-6]
    if len(emitted) != 1:
        raise ValueError('Entrance history needs one first-emission CSV row')
    row = emitted[0]
    values = [float(row[k]) for k in ('x','y','target_x','target_y','vx_cmd','omega_cmd')]
    if (not np.isfinite(values).all() or row['action_id'] != movement['action_id']
            or row['route_purpose'] != 'mission' or row['movement_state'] != 'DRIVE_FORWARD'
            or values[4] <= 0. or values[5] != 0.
            or math.dist(values[:2],start) > .02
            or math.dist(values[2:4],corridor['approach_end']) > 1e-6):
        raise ValueError('Entrance history CSV does not establish the climb-containing primitive')
    return dict(corridor=corridor,attempt_id=outcome['attempt_id'],action_id=movement['action_id'],
                stage='entrance_attempted_climb_uncharacterized',outcome='censored',reason=reason,
                advance_along_approach_m=advance,maximum_lateral_separation_m=separation,
                distance_before_sustained_climb_m=to_steep,
                membership_rule='Within 1 m before the sustained section; coarse experiment membership only',
                probability_vote=False,exclusion_vote=False)


def load_coverage(run, grid, provenance, materials):
    """Verify comparable run metadata and retain completed executed corridors.

    Historical material logs do not contain body masses/configuration hashes;
    the result says so explicitly. Task/seed/spawn, material values, source and
    capability checks are mandatory. No counters or exclusions are imported.
    """
    run = Path(run)
    paths = {name: run/('run.'+name) for name in ('provenance.json', 'materials.json', 'events.jsonl')}
    old = json.loads(paths['provenance.json'].read_text())
    if provenance['arguments']['ascent_mode'] != 'characterize':
        raise ValueError('Route exploration is characterization only')
    for key in ('policy_sha256', 'seed'):
        if old.get(key) != provenance.get(key):
            raise ValueError('Exploration history mismatch: '+key)
    for key in ('task','spawn_x','spawn_y','spawn_z','spawn_yaw','cruise_speed','max_uphill_deg','max_side_slope_deg'):
        if old['arguments'].get(key) != provenance['arguments'].get(key):
            raise ValueError('Exploration history mismatch: '+key)
    if old.get('terrain_prior', {}).get('geometry_id') != grid.terrain_prior['geometry_id']:
        raise ValueError('Exploration history mismatch: terrain geometry')
    legacy = legacy_speed_history(run,old,provenance)
    legacy_additions = LEGACY_PROFILES[provenance['capability_profile']['execution_protocol']] if legacy else {}
    for name in ('movement','swept_footprint','ascent_map','observed_map','drive_adapter'):
        if legacy and (name == 'movement' or (name in LEGACY_EXIT_GEOMETRY
                and provenance['capability_profile']['execution_protocol'] in (
                    'staged-teleop-v28-directional-recovery-exit','staged-teleop-v29-paired-launch','staged-teleop-v30-uphill-speed-response','staged-teleop-v31-functional-recovery'))):
            continue
        key = 'source/rexmi_rl/'+('' if name == 'drive_adapter' else 'nav/')+name+'.py'
        if old['sources'].get(key) != provenance['sources'].get(key) or not old['sources'].get(key):
            raise ValueError('Exploration history mismatch: '+name)
    # Protocol identifier includes planning changes, so compare the remaining
    # capability fields exactly rather than pretending the identifiers match.
    old_cap = {k:v for k,v in old['capability_profile'].items() if k != 'execution_protocol'}
    excluded = {'execution_protocol', *legacy_additions}
    new_cap = {k:v for k,v in provenance['capability_profile'].items() if k not in excluded}
    if old_cap != new_cap:
        raise ValueError('Exploration history mismatch: capability')
    if json.loads(paths['materials.json'].read_text()) != materials or 'inspection_error' in materials:
        raise ValueError('Exploration history mismatch: measured materials')
    events = [json.loads(line) for line in paths['events.jsonl'].read_text().splitlines()]
    plans, attempts, records, completed = None, {}, [], set()
    active, traversals, movement_ids = None, {}, set()
    movement_results, entrance_attempts = {}, []
    csv_rows = None
    for event in events:
        kind = event.get('event')
        if kind == 'route_comparison':
            plans = next((c for c in event['candidates'] if c.get('selected') and c.get('complete')), None)
        elif kind == 'approach_start':
            active = event['attempt_id']
            attempts[event['attempt_id']] = first_climb(grid, plans['route_path']) if plans else None
            traversals.setdefault(active, [])
            movement_results.setdefault(active, [])
        elif kind == 'movement_outcome' and active is not None:
            if event['action_id'] in movement_ids:
                raise ValueError('Duplicate movement outcome in exploration history')
            movement_ids.add(event['action_id'])
            movement_results[active].append(event)
            if event.get('executed_traversal'):
                traversals[active].append(event['executed_traversal'])
        elif kind == 'approach_outcome':
            if event['attempt_id'] in completed:
                raise ValueError('Duplicate completed attempt in exploration history')
            completed.add(event['attempt_id'])
            if active == event['attempt_id']:
                active = None
            corridor = attempts.get(event['attempt_id'])
            if event.get('outcome') == 'censored' and event.get('reason') == 'remaining straight corridor unavailable from measured pose':
                if csv_rows is None:
                    paths['csv'] = run/'run.csv'
                    with paths['csv'].open() as stream:
                        csv_rows = list(csv.DictReader(stream))
                entrance = entrance_record(corridor,event,movement_results.get(event['attempt_id'],[]),csv_rows)
                if entrance:
                    verify_guard_source(run,old,legacy=legacy)
                    entrance_attempts.append(entrance)
            if event.get('outcome') != 'success' and not (event.get('outcome') == 'failure' and event.get('physical')):
                continue
            chunks = list(traversals.get(event['attempt_id'], []))
            if event.get('executed_traversal'):
                chunks.append(event['executed_traversal'])
            for executed in chunks:
                if executed.get('maneuver') != 'DRIVE_FORWARD' or executed.get('duration_s', 0.) <= 0.:
                    continue
                actual = dict(start=executed['start_position'], end=executed['last_position'], heading=executed['heading'])
                if corridor and same_corridor(corridor, actual):
                    records.append(dict(corridor=corridor, attempt_id=event['attempt_id'], outcome=event['outcome'],
                                        executed_overlap=corridor_overlap(corridor, actual)))
                    break
    if not records and not entrance_attempts:
        raise ValueError('History has no independently completed, executed sustained climb')
    scope = 'legacy_executor_scheduling_witness' if legacy else 'matching_executor_scheduling_witness'
    for record in records+entrance_attempts:
        record.update(execution_scope=scope,execution_protocol=old['capability_profile'].get('execution_protocol'),
                      probability_vote=False,exclusion_vote=False)
    return dict(kind='characterization_coverage_only', run=str(run.resolve()), records=records, entrance_attempts=entrance_attempts,
                execution_scope=scope,legacy_source_pair=(
                    {k:old['sources'][k] for k in ('source/rexmi_rl/nav/movement.py','source/rexmi_rl/nav/ascent_behavior.py',
                        'source/rexmi_rl/nav/ascent_map.py','source/rexmi_rl/nav/swept_footprint.py')} if legacy else None),
                files_sha256={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in paths.items()},
                limitation='Historical body masses and physics config hashes were not recorded; task/seed/spawn/materials and applicable source/capability checks passed. '
                    + ('Reviewed legacy executor scheduling only; not current-executor characterization. ' if legacy else 'Matching executor. ')
                    + 'Not imported mission evidence, probability or exclusions.')


def merge_coverage(histories):
    if len({h['run'] for h in histories}) != len(histories):
        raise ValueError('Duplicate exploration history')
    return dict(kind='characterization_coverage_only', histories=histories,
                records=[r for h in histories for r in h['records']],
                entrance_attempts=[r for h in histories for r in h.get('entrance_attempts',[])])
