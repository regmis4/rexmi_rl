"""Archive small, reproducible evidence; raw simulation logs stay local."""
import csv, hashlib, json, math, shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path.cwd()
folder=root/'docs/validation/crater_descent_response_2026-10-24'
folder.mkdir(parents=True,exist_ok=True)
trials={'original_user_run':'mission-pdI7CPgR','initial_response':'mission-P4z1d3yh',
        'stronger_reverse':'mission-sCi3vpRM','whole_approach':'mission-jNAbxxGX'}
for label in ('paused_final','uninterrupted_final'):
    marker=root/'outputs/descent_response_validation'/f'{label}.run'
    if marker.exists():trials[label]=Path(marker.read_text().strip()).name
review={}
fields=('sim_time','x','y','z','yaw','speed','up_z','state','reason','vx_cmd','omega_cmd','mission_stage')
traces={}
for label,name in trials.items():
    run=root/'outputs/crater_loop'/name
    rows=list(csv.DictReader((run/'run.csv').open()))
    events=[json.loads(line) for line in (run/'run.events.jsonl').read_text().splitlines()]
    summary_path=run/'run.summary.json'
    summary=json.loads(summary_path.read_text()) if summary_path.exists() else {}
    active=[r for r in rows if r['state']!='HOLD' and float(r['sim_time'])>1.]
    reverse=[e for e in events if e['event'] in ('descent_reverse_start','descent_reverse_end')]
    descent=[r for r in rows if r.get('mission_stage')=='DESCEND_TO_CENTER'] or rows
    xy=np.array([[float(r[k]) for k in ('x','y')] for r in descent])
    mismatch=[]
    provenance=json.loads((run/'run.provenance.json').read_text())
    for path,digest in provenance['sources'].items():
        p=root/path
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:mismatch.append(path)
    review[label]=dict(run=str(run.relative_to(root)),result=summary.get('result','user_closed'),
        sim_seconds=float(rows[-1]['sim_time']),final={k:rows[-1][k] for k in fields},
        peak_active_speed_mps=max(float(r['speed']) for r in active),
        minimum_active_up_z=min(float(r['up_z']) for r in active),
        descent_min_radius_m=float(np.linalg.norm(xy,axis=1).min()),
        descent_travel_m=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()),
        reverse_events=reverse,source_mismatches=mismatch,
        counts_as_mission=summary.get('counts_as_mission',False),
        raw_csv_sha256=hashlib.sha256((run/'run.csv').read_bytes()).hexdigest())
    if label.endswith('_final'):
        destination=folder/label;destination.mkdir(exist_ok=True)
        for filename in ('run.summary.json','run.provenance.json','run.loop.json'):
            if (run/filename).exists():shutil.copy2(run/filename,destination/filename.removeprefix('run.'))
        with (destination/'trace_10hz.csv').open('w') as out:
            writer=csv.DictWriter(out,fieldnames=fields);writer.writeheader()
            writer.writerows({k:r[k] for k in fields} for r in rows[::5])
        (destination/'events.json').write_text(json.dumps(events,indent=2)+'\n')
    traces[label]=rows
(folder/'review.json').write_text(json.dumps(review,indent=2)+'\n')
if Path(__file__).resolve()!=folder/'collect_review.py':shutil.copy2(__file__,folder/'collect_review.py')
if 'paused_final' in traces:
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True,layout='constrained')
    for label,color,title in [('original_user_run','#bb5a50','Original user run'),('paused_final','#008fc2','Current stop/resume replay')]:
        rows=traces[label];t=np.array([float(r['sim_time']) for r in rows])
        x=np.array([float(r['x']) for r in rows]);y=np.array([float(r['y']) for r in rows])
        axes[0].plot(t,np.hypot(x,y),color=color,label=title,lw=1.6)
        axes[1].plot(t,[float(r['z']) for r in rows],color=color,lw=1.6)
        axes[2].plot(t,[float(r['speed']) for r in rows],color=color,lw=1.2)
        for event in review[label]['reverse_events']:
            if event['event']=='descent_reverse_start':
                axes[0].axvline(event['time'],color=color,alpha=.3,ls='--')
    for axis in axes:axis.grid(alpha=.18)
    axes[0].set_ylabel('Distance to center (m)');axes[0].legend()
    axes[1].set_ylabel('Body elevation (m)');axes[2].set_ylabel('Actual XY speed (m/s)')
    axes[2].set_xlabel('Simulated time (s)')
    axes[0].set_title('Recorded trajectories; dashed lines mark backout starts\nSame stop/resume timing, independently simulated routes')
    fig.savefig(folder/'response_comparison.png',dpi=150);plt.close(fig)
print(json.dumps({k:{x:v[x] for x in ('result','sim_seconds','peak_active_speed_mps','descent_min_radius_m')} for k,v in review.items()},indent=2))
