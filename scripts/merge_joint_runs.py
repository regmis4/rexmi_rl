"""Combine independent captures without bridging windows or averaging away extremes."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from analyze_joint_telemetry import metrics, sha256

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs' / 'motor_selection_evidence'

def main():
    OUT.mkdir(exist_ok=True)
    frames, sources, records = [], [], []
    for run_id in ['manual_20260909_01', 'manual_20260910_01']:
        run = ROOT / 'logs' / 'telemetry' / run_id
        meta = json.loads((run / 'metadata.json').read_text())
        raw = pd.read_csv(run / 'samples.csv')
        assert np.allclose(raw.dt_s, .005)
        assert np.allclose(np.diff(raw.sim_time_s), .005)
        f = raw.loc[raw.sim_time_s >= 2].copy()
        f['episode'] = run_id + ':' + f.episode.astype(str)
        f['run_id'] = run_id
        names = [c.split('.')[0] for c in f if c.endswith('.torque_applied_est_Nm')]
        assert len(names) == 16
        for j in names:
            assert np.isfinite(f[[j+'.torque_applied_est_Nm', j+'.velocity_rad_s', j+'.torque_demand_est_Nm']]).all().all()
            records.append(dict(run_id=run_id, **metrics(f, j)))
        forces = np.stack([f[[f'contact.{leg}_foot.{a}_N' for a in ['fx_w','fy_w','fz_w']]].to_numpy() for leg in ['FL','FR','RL','RR']], axis=1)
        f['contact_count'] = (np.linalg.norm(forces, axis=2) > 5).sum(axis=1)
        sources.append(dict(run_id=run_id, hashes={n:sha256(run/n) for n in ['samples.csv','metadata.json','events.jsonl']}, full_rows=len(raw), retained_rows=len(f), mass_kg=sum(meta['live_body_masses_kg'])))
        frames.append(f)
    merged = pd.concat(frames, ignore_index=True)
    per_run = pd.DataFrame(records)
    envelope = []
    for j in names:
        m = metrics(merged, j)
        m['worst_run_rms_Nm'] = float(per_run.loc[per_run.joint==j, 'rms_est_Nm'].max())
        m['peak_first_run_id'] = str(merged.loc[merged[j+'.torque_applied_est_Nm'].abs().idxmax(), 'run_id'])
        t, v = merged[j+'.torque_applied_est_Nm'], merged[j+'.velocity_rad_s']
        k = v.abs().idxmax()
        m['torque_at_max_speed_Nm'] = float(t[k])
        m['signed_max_speed_rad_s'] = float(v[k])
        m['max_speed_run_id'] = str(merged.run_id[k])
        m['max_speed_time_s'] = float(merged.sim_time_s[k])
        m['max_motoring_speed_rad_s'] = float(v[t*v>0].abs().max())
        envelope.append(m)
    result = pd.DataFrame(envelope)
    result.to_csv(OUT/'joint_envelope.csv', index=False)
    per_run.to_csv(OUT/'per_run_joint_metrics.csv', index=False)
    summary = dict(sources=sources, retained_duration_s=float(merged.dt_s.sum()), retained_rows=len(merged),
        method='Virtual concatenation of raw captures; exclude sim_time_s < 2 in each; unique run/episode keys prevent cross-run windows. Pooled RMS is duration weighted. Worst-run RMS preserved separately. Raw files unchanged.',
        contact_pct={str(i):float(100*np.average(merged.contact_count==i, weights=merged.dt_s)) for i in range(5)})
    (OUT/'manifest.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))
    print(result[['joint','rms_est_Nm','worst_run_rms_Nm','max_10s_rms_est_Nm','max_abs_speed_rad_s','torque_at_max_speed_Nm','max_motoring_speed_rad_s']].to_string(index=False))

if __name__ == '__main__':
    main()
