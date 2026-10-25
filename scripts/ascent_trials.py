#!/usr/bin/env python3
"""Reproducible ascent runs and a fail-closed, three-run qualification gate.

This launcher runs only ascent. It never advances into descent on partial
success or relabels characterization as qualification.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def qualifies(summary):
    prior=summary.get('terrain_prior') or {}
    return (not summary.get('diagnostic',False) and summary.get('counts_as_qualification',True)
            and summary.get('result') == 'complete' and summary.get('mission') == 'ascent'
            and summary.get('ascent_mode') == 'qualified' and bool(summary.get('model_id'))
            and summary.get('probability_model_qualified',False)
            and bool(prior.get('geometry_id')) and prior.get('mission_coverage',prior.get('coverage',0.))>=.95
            and prior.get('kind') in ('simulator_development','retained_sensor_survey')
            and summary.get('spawn') == [0., 0., 1.15, 0.] and summary.get('goal') == [13., 0.]
            and summary.get('seed') == 42 and summary.get('sim_seconds', float('inf')) <= 900.
            and not summary.get('manual_intervention', True) and summary.get('policy_switches', 1) == 0
            and summary.get('peak_command_mps', float('inf')) <= .400001
            and summary.get('min_active_up_z', 0.) >= .3
            and summary.get('waypoints_reached') == summary.get('total_waypoints') == 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--model')
    p.add_argument('--mode', choices=['characterize', 'qualified'], default='characterize')
    p.add_argument('--runs', type=int, default=3)
    p.add_argument('--output', required=True)
    p.add_argument('--max-uphill-deg',type=float,default=33.)
    p.add_argument('--architecture',choices=['behavior_tree','legacy'],default='behavior_tree')
    p.add_argument('--max-side-slope-deg',type=float,default=25.)
    p.add_argument('--preferred-uphill-deg',type=float,default=20.)
    information=p.add_mutually_exclusive_group()
    information.add_argument('--simulator-survey',action='store_true',help='Labeled full-crater simulator prior for development')
    information.add_argument('--terrain-prior',help='Retained labeled full-crater survey NPZ')
    p.add_argument('--vary-floor-starts', action='store_true', help='Characterization only: small floor start/yaw variations')
    args = p.parse_args()
    if args.runs < 1 or (args.mode == 'qualified' and (not args.model or args.vary_floor_starts)):
        p.error('Qualification needs a frozen model, standard start, and positive run count')
    if not (0<args.preferred_uphill_deg<=args.max_uphill_deg<=35. and 0<args.max_side_slope_deg<=35.):
        p.error('Grade preferences and limits must be positive and stay within 35 degrees')
    if args.mode=='qualified' and not (args.simulator_survey or args.terrain_prior):
        p.error('Ascent qualification now requires the full-crater information condition: --simulator-survey or --terrain-prior')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output/'gate.json').exists():
        p.error('Use a new output directory; prior validation evidence is never overwritten')
    manifest = dict(stage='ascent', mode=args.mode, passed=False, descent_allowed=False, trials=[], consecutive=0,
                    information_condition='simulator_survey' if args.simulator_survey else 'retained_survey' if args.terrain_prior else 'local_only_baseline')
    expected_hash = hashlib.sha256(Path(args.model).read_bytes()).hexdigest() if args.model else None
    expected_id=(hashlib.sha256(json.dumps(json.loads(Path(args.model).read_text()),sort_keys=True).encode()).hexdigest()
                 if args.model else None)
    starts = [(0.,0.,0.), (.6,0.,.2), (-.6,0.,-.2), (0.,.6,.3), (0.,-.6,-.3)]
    source_id = None
    prior_id = None
    for index in range(args.runs):
        destination = output/f'run{index+1:02d}.csv'
        command = [str(ROOT/'run.sh'), 'scripts/navigate.py', '--task', 'RexmiRl-Go2w-Crater-Bowl-RockySlope-Play-v0',
                   '--checkpoint', str(Path(args.checkpoint).resolve()), '--mission', 'ascent',
                   '--ascent_mode', args.mode, '--seed', '42', '--headless', '--no_dashboard',
                   '--ascent_architecture', args.architecture,
                   '--max_uphill_deg',str(args.max_uphill_deg),
                   '--max_side_slope_deg',str(args.max_side_slope_deg),
                   '--preferred_uphill_deg',str(args.preferred_uphill_deg),
                   '--log_file', str(destination)]
        if args.simulator_survey:command+=['--simulator_survey']
        if args.terrain_prior:command+=['--terrain_prior',str(Path(args.terrain_prior).resolve())]
        if args.model:
            if hashlib.sha256(Path(args.model).read_bytes()).hexdigest() != expected_hash:
                raise RuntimeError('Model changed during the qualification series')
            command += ['--reachability_model', str(Path(args.model).resolve())]
        if args.vary_floor_starts:
            x,y,yaw=starts[index % len(starts)]
            command += ['--spawn_x',str(x),'--spawn_y',str(y),'--spawn_yaw',str(yaw)]
        with destination.with_suffix('.console.log').open('w') as out:
            completed = subprocess.run(command, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, env=os.environ.copy())
        summary_path = destination.with_suffix('.summary.json')
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else dict(result='startup_failure')
        okay = completed.returncode == 0 and qualifies(summary)
        if index == 0:
            source_id = summary.get('source_id')
            prior_id=(summary.get('terrain_prior') or {}).get('geometry_id')
        okay = okay and source_id is not None and summary.get('source_id') == source_id
        okay = okay and prior_id is not None and (summary.get('terrain_prior') or {}).get('geometry_id')==prior_id
        okay = okay and summary.get('model_id') == expected_id
        manifest['consecutive'] = manifest['consecutive']+1 if okay else 0
        manifest['trials'].append(dict(path=str(summary_path), returncode=completed.returncode,
                                       result=summary['result'], qualifies=okay, model_id=summary.get('model_id')))
        manifest['passed'] = args.mode == 'qualified' and manifest['consecutive'] >= 3
        manifest['descent_allowed'] = manifest['passed']
        (output/'gate.json').write_text(json.dumps(manifest, indent=2)+'\n')
        print(json.dumps(manifest['trials'][-1]), flush=True)
        if manifest['passed'] or (args.mode == 'qualified' and not okay):
            break
    return 0 if args.mode == 'characterize' or manifest['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
