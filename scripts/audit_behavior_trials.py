#!/usr/bin/env python3
"""Audit completed behavior-tree trials using emitted commands and event IDs."""
import argparse
import csv
import json
import math
from pathlib import Path


def audit(folder):
    summary = json.loads((folder/'run.summary.json').read_text())
    rows = list(csv.DictReader((folder/'run.csv').open()))
    events = [json.loads(line) for line in (folder/'run.events.jsonl').read_text().splitlines()]
    active = [r for r in rows if float(r['sim_time']) >= 3. and r['state'] != 'MANUAL']
    v2 = 'body_tilt_deg' in rows[0]
    violations = dict(steep_terrain_yaw=[], simultaneous_drive_yaw=[], command_ceiling=[], body_pause_yaw=[],
                      release_nonzero_command=[], release_state_telemetry=[], unexecuted_direction_exclusion=[],
                      posture_pause_nonzero_command=[])
    for row in active:
        t, vx, omega = (float(row[k]) for k in ('sim_time', 'vx_cmd', 'omega_cmd'))
        support = float(row['support_inclination_deg'])
        if support > 15.+1e-6 and abs(omega) > 1e-8: violations['steep_terrain_yaw'].append(t)
        if abs(vx) > 1e-8 and abs(omega) > 1e-8: violations['simultaneous_drive_yaw'].append(t)
        if abs(vx) > .4+1e-8 or abs(omega) > .35+1e-8: violations['command_ceiling'].append(t)
        if v2 and float(row['body_tilt_deg']) > 25.+1e-6 and abs(omega) > 1e-8:
            violations['body_pause_yaw'].append(t)
        released = row['movement_state'] == 'RELEASE'
        if released and (abs(vx) > 1e-8 or abs(omega) > 1e-8): violations['release_nonzero_command'].append(t)
        if row['movement_state'] == 'POSTURE_PAUSE' and (abs(vx) > 1e-8 or abs(omega) > 1e-8):
            violations['posture_pause_nonzero_command'].append(t)
        if 'release_active' in row and (row['release_active'] == 'True') != released:
            violations['release_state_telemetry'].append(t)
    for event in events:
        if event.get('event') != 'approach_outcome' or not event.get('exclusion_applied'):
            continue
        # Only the explicit executed-evidence contract is audited here. Older
        # protocols retain their historical logs without retroactive relabeling.
        if event.get('terminal_maneuver') is not None:
            execution = event.get('executed_traversal')
            if (event['outcome'] != 'failure' or not event.get('physical') or not execution
                    or execution.get('duration_s', 0.) <= 0.):
                violations['unexecuted_direction_exclusion'].append(event['end_time'])
    counts = {k: len(v) for k, v in violations.items()}
    duplicates = {}
    for kind, key in [('movement_outcome', 'action_id'), ('approach_outcome', 'attempt_id')]:
        ids = [e[key] for e in events if e.get('event') == kind]
        duplicates[kind] = len(ids)-len(set(ids))
    last = rows[-1]
    distance = math.dist((float(last['x']), float(last['y'])), summary['goal'])
    accounting = summary['result'] != 'complete' or (summary['waypoints_reached'] == summary['total_waypoints'] and distance <= .6)
    exposure = {}
    drive_backslide = 0.
    release_backslide = 0.
    for before, after in zip(rows, rows[1:]):
        duration = float(after['sim_time'])-float(before['sim_time'])
        phase = before['movement_state']
        exposure[phase] = exposure.get(phase, 0.)+duration
        if phase in ('DRIVE_FORWARD', 'DRIVE_REVERSE') and before['action_id']==after['action_id']:
            # Measured displacement against requested body-axis travel. Exclude
            # damping commands, whose sign intentionally opposes inertial motion.
            dx, dy = float(after['x'])-float(before['x']), float(after['y'])-float(before['y'])
            yaw = float(before['yaw'])
            sign = 1. if phase == 'DRIVE_FORWARD' else -1.
            drive_backslide += max(0., -sign*(dx*math.cos(yaw)+dy*math.sin(yaw)))
        if phase == 'RELEASE' and before['action_id'] == after['action_id']:
            # This field projects actual velocity onto the immutable committed
            # axis. Zero input must not conceal physically backward movement.
            release_backslide += max(0., -float(before['measured_forward_mps']))*duration
    return dict(name=folder.name, result=summary['result'], sim_seconds=summary['sim_seconds'],
        final_reason=summary['final_reason'], source_id=summary['source_id'],
        support_field='footprint terrain' if v2 else 'maximum of terrain and body tilt',
        invariants_pass=not any(counts.values()) and not any(duplicates.values()) and accounting,
        violation_counts=counts, first_violations={k: v[:5] for k, v in violations.items()},
        duplicate_terminal_events=duplicates, destination_accounting=accounting, final_goal_distance_m=distance,
        phase_sim_seconds=exposure, movement_outcomes=[e for e in events if e.get('event') == 'movement_outcome'],
        peak_actual_speed_mps=summary['peak_active_speed_mps'], peak_active_tilt_deg=math.degrees(math.acos(summary['min_active_up_z'])),
        command_opposed_motion_m=summary['backslide_m'], committed_drive_backslide_m=drive_backslide,
        release_backslide_m=release_backslide,
        controller_ms_p95=summary['controller_ms_p95'],
        limitation='Command invariants are not physical mission completion or probability calibration.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='outputs/ascent_behavior_tree')
    parser.add_argument('--output', default='docs/validation/ascent_behavior_audit.json')
    args = parser.parse_args()
    results = [audit(path.parent) for path in sorted(Path(args.root).glob('*/run.summary.json'))]
    report = dict(trials=results, ascent_qualified=False)
    Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps([dict(name=r['name'], result=r['result'], invariants_pass=r['invariants_pass'],
                          violations=r['violation_counts']) for r in results]))


if __name__ == '__main__': main()
