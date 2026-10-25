"""Archive compact physical evidence; retain bulk telemetry only in outputs/."""
import csv,hashlib,json,shutil,sys
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parents[3]
run=Path(sys.argv[1]).resolve();destination=Path(__file__).parent/sys.argv[2]
destination.mkdir(exist_ok=True)
rows=list(csv.DictReader((run/'run.csv').open()))
events=[json.loads(line) for line in (run/'run.events.jsonl').read_text().splitlines()]
provenance=json.loads((run/'run.provenance.json').read_text())
summary_path=run/'run.summary.json'
summary=json.loads(summary_path.read_text()) if summary_path.exists() else {'result':'interrupted; no final mission summary','counts_as_mission':False}
loop=json.loads((run/'run.loop.json').read_text())
descent=[r for r in rows if r['mission_stage']=='DESCEND_TO_CENTER']
xy=np.array([[float(r[k]) for k in ('x','y')] for r in descent])
active=[r for r in descent if float(r['sim_time'])>1 and r['state']!='HOLD']
reverse=[e for e in events if e['event']=='descent_reverse_end']
source_mismatches=[name for name,digest in provenance['sources'].items()
    if not (root/name).exists() or hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest]
audit=dict(run=str(run.relative_to(root)),result=summary['result'],
    counts_as_mission=summary.get('counts_as_mission'),
    center_arrival_time=loop.get('center_arrival_time'),
    final_recorded_stage=loop.get('stage'),
    center_confirmation_time=(loop.get('descent') or {}).get('time'),
    manual_intervention=loop.get('manual_intervention'),
    last_logged_sim_time=float(rows[-1]['sim_time']),
    minimum_descent_radius_m=float(np.linalg.norm(xy,axis=1).min()),
    descent_travel_m=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()),
    descent_travel_after_1s_m=float(np.linalg.norm(np.diff(np.array([[float(r[k]) for k in ('x','y')] for r in descent if float(r['sim_time'])>1]),axis=0),axis=1).sum()),
    peak_active_descent_speed_mps=max(float(r['speed']) for r in active),
    minimum_active_descent_up_z=min(float(r['up_z']) for r in active),
    reverse_attempts=len(reverse),successful_reverses=sum(e['success'] for e in reverse),
    peak_emitted_reverse_request_mps=max([-float(r['vx_cmd']) for r in descent if r['state']=='BACKOUT'] or [0.]),
    continuous_timestamps=bool(np.allclose(np.diff([float(r['sim_time']) for r in rows]),.02)),
    source_mismatches=source_mismatches,
    raw_csv_sha256=hashlib.sha256((run/'run.csv').read_bytes()).hexdigest())
for name in ('summary','loop','provenance'):
    source=run/f'run.{name}.json'
    if source.exists():shutil.copy2(source,destination/f'{name}.json')
fields=('sim_time','x','y','z','yaw','speed','up_z','state','reason','vx_cmd','omega_cmd','mission_stage')
with (destination/'trace_10hz.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
    writer.writerows({k:r[k] for k in fields} for r in rows[::5])
(destination/'events.json').write_text(json.dumps(events,indent=2)+'\n')
(destination/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit,indent=2))
