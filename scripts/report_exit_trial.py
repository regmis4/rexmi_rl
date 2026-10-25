#!/usr/bin/env python3
"""Report one any-exit diagnostic without changing its raw evidence."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
import test_remote_control  # Pure navigation import setup, no simulator.
from rexmi_bt_tests.nav import planning_snapshot


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--stop-reason',help='Required for a deliberately interrupted run lacking its normal summary')
    args=parser.parse_args();run=args.run
    summary_path=run/'run.summary.json'
    summary=json.loads(summary_path.read_text()) if summary_path.exists() else None
    if summary is None and not args.stop_reason:
        parser.error('Missing normal summary: supply the actual diagnostic --stop-reason')
    result=summary['result'] if summary is not None else 'diagnostic_operator_truncation'
    with (run/'run.csv').open() as f:
        raw=list(csv.reader(f))
    header=raw[0];valid=[r for r in raw[1:] if len(r)==len(header)]
    rows=[dict(zip(header,r)) for r in valid]
    column=lambda name:np.array([float(r[name]) for r in rows])
    t,x,y,z=(column(k) for k in ('sim_time','x','y','z'))
    events=[json.loads(s) for s in (run/'run.events.jsonl').read_text().splitlines()]
    turns=[e for e in events if e.get('event')=='remote_transition'
           and e.get('state')=='RECOVER' and 'local-minimum turn unavailable' in e.get('reason','')]
    plans=[]
    for p in (run/'run.planning').glob('*.json'):
        outcome=json.loads(p.read_text());g,meta=planning_snapshot.load(p.with_suffix('.npz'))
        plans.append((meta['time'],outcome,meta,g))
    plans.sort(key=lambda item:item[0])
    first=next(item for item in plans if item[1]['success'])
    anchor=int(np.searchsorted(t,3.))
    report=dict(result=result,counts_as_mission_success=result=='complete',
        counts_as_qualification=False,stop_reason=args.stop_reason or rows[-1]['reason'],
        sim_seconds=float(t[-1]),complete_csv_rows=len(rows),excluded_partial_rows=len(raw)-1-len(rows),
        continuous_control_samples=bool(np.allclose(np.diff(t),.02,atol=1e-8)),
        final_position=[float(x[-1]),float(y[-1]),float(z[-1])],
        max_radius_m=float(np.hypot(x,y).max()),peak_body_z_m=float(z[anchor:].max()),
        peak_body_gain_from_3s_m=float(z[anchor:].max()-z[anchor]),
        completed_waypoints=int(rows[-1]['wp_idx']),final_state=rows[-1]['state'],
        final_reason=rows[-1]['reason'],policies=sorted(set(r['policy'] for r in rows)),
        peak_command_mps=float(np.abs(column('vx_cmd')).max()),
        physical_climb_failures=(None if summary and summary.get('climb_failure_classification_available') is False
                                else int(column('climb_failures').max())),
        feedback_recoveries=[e for e in events if e.get('event')=='feedback_recovery_end'],
        furthest_saved_route_progress_m=summary.get('furthest_saved_route_progress_m') if summary else None,
        final_saved_route_progress_m=summary.get('final_saved_route_progress_m') if summary else None,
        first_turn_refusal=turns[0] if turns else None,turn_refusals=turns,
        checked_corrections=[e for e in events if e.get('event') in ('checked_turn_start','checked_turn_end')],
        saved_path_stops=[e for e in events if e.get('event')=='saved_path_stop'],
        saved_path_input=(json.loads((run/'run.provenance.json').read_text()).get('saved_path')),
        route_rejoins=[e for e in events if e.get('event')=='checked_route_rejoin'],
        archived_plan_results=[dict(time=item[0],success=item[1]['success'],
            failure=item[1].get('planning_failure'),
            expansions=(item[1].get('searches') or [{}])[-1].get('expansions')) for item in plans],
        initial_runtime_route=first[1]['route'],initial_runtime_yaw=first[2]['yaw'],
        limitation=('Signal interruption lacks normal runner summary. ' if summary is None else '')+
            'One characterization run does not establish reliable ascent or hard-climb infeasibility.',
        files_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (run/'run.csv',run/'run.events.jsonl',run/'run.provenance.json',run/'review.json')})
    (run/'diagnostic_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(12,8),layout='constrained')
    gs=fig.add_gridspec(2,2,width_ratios=(1.1,1.))
    ax=fig.add_subplot(gs[:,0]);height=fig.add_subplot(gs[0,1]);commands=fig.add_subplot(gs[1,1])
    from matplotlib import patheffects
    saved=report['saved_path_input']
    path=np.asarray(first[1]['route'])
    if saved:
        from rexmi_bt_tests.nav import slope_cost
        from slope_cost_plot import draw, COLORS
        frozen,_=planning_snapshot.load(saved['map_path'])
        model=slope_cost.Model(frozen,slope_cost.Parameters(**saved['parameters']))
        scale=draw(ax,model)
        cb=fig.colorbar(scale,ax=ax,orientation='horizontal',fraction=.04,pad=.02)
        cb.set_label('Steepest-uphill cost density, log scale; frozen fitted map')
        fitted=json.loads((run/'run.path-result.json').read_text())
        for name,label in [('rejected','Rejected corridor'),('manual','Successful manual'),('matched_manual_exit','Fitted manual endpoint')]:
            other=np.asarray(fitted['routes'][name])
            ax.plot(*other.T,color=COLORS[name],lw=1.,ls='--' if name!='manual' else '-',alpha=1.,label=label)
        ax.plot(*path.T,color=COLORS['any_exit'],lw=2.,label='Blue planned path')
        comparison=np.concatenate([np.asarray(route) for route in fitted['routes'].values()])
    else:
        g=first[3];half=g.world_size/2.
        ax.imshow(np.where(g.known,g.cost,np.nan).T,origin='lower',extent=[g.origin[0]-half,g.origin[0]+half,g.origin[1]-half,g.origin[1]+half],cmap='magma')
        ax.plot(*path.T,color='#00d9ff',lw=2,label='Planned route')
        comparison=path
    ax.plot(x,y,c='white',lw=1.5,label='Actual autonomous trajectory',
        path_effects=[patheffects.Stroke(linewidth=2.6,foreground='black'),patheffects.Normal()])
    if turns:
        i=int(np.searchsorted(t,turns[0]['time']));ax.scatter(x[i],y[i],c='red',marker='x',s=90,label='First turn refusal')
    ax.scatter(x[-1],y[-1],c='white',s=35,label='Final measured pose')
    points=np.vstack([comparison,np.column_stack([x,y])]);low=points.min(axis=0)-1.;high=points.max(axis=0)+1.
    ax.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]),aspect='equal',xlabel='World X (m)',ylabel='World Y (m)',
           title=('Exit reached' if result=='complete' else 'Exit not reached')+' · unchanged policy')
    ax.legend(loc='upper left',fontsize=7,facecolor='#171924',labelcolor='white',framealpha=.95)
    height.plot(t[anchor:],z[anchor:],label='Measured body height after settling',c='#3573ac')
    height.set(xlabel='Simulated seconds',ylabel='Body Z (m)',title='Body height is not terrain elevation')
    commands.step(t,column('vx_cmd'),where='post',label='Forward/reverse (m/s)')
    commands.step(t,column('omega_cmd'),where='post',label='Yaw (rad/s)',alpha=.8)
    commands.set(xlabel='Simulated seconds',ylabel='Requested command',title='Zero input keeps policy active')
    commands.legend(fontsize=8)
    for e in turns:
        for panel in (height,commands):panel.axvline(e['time'],c='red',ls=':',alpha=.6)
    fig.suptitle('Any-safe-exit diagnostic — '+result.replace('_',' '))
    fig.savefig(run/'trial.png',dpi=150);plt.close(fig)
    print(json.dumps({k:report[k] for k in ('sim_seconds','complete_csv_rows','continuous_control_samples',
        'final_position','max_radius_m','peak_body_gain_from_3s_m','physical_climb_failures','completed_waypoints')}))


if __name__=='__main__':main()
