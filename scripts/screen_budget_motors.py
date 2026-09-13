"""Nominal-voltage motoring screen only; preserves raw captures."""
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
curves={'RS01_36V':([135,184,228,265,305],[17,14,10,6,.5]),'RS02_48V':([219,273,326,365,407],[17,14,10,7,.5])}
f=pd.concat([pd.read_csv(ROOT/'logs/telemetry'/r/'samples.csv').query('sim_time_s >= 2') for r in ['manual_20260909_01','manual_20260910_01']],ignore_index=True)
rows=[]
for kind in ['hip','thigh','calf']:
    names=[c.split('.')[0] for c in f if c.endswith('.torque_applied_est_Nm') and '_'+kind+'_' in c]
    for model,(speeds,torques) in curves.items():
        for N in [1,1.5,1.6,1.75,2,2.25,2.5]:
            eta=1 if N==1 else .95
            bad=total=0; worst=0
            for j in names:
                t=f[j+'.torque_applied_est_Nm'].to_numpy();v=f[j+'.velocity_rad_s'].to_numpy();mask=t*v>0
                need=abs(t[mask])/(eta*N); rpm=abs(v[mask])*N*60/(2*np.pi)
                lim=np.interp(rpm,speeds,torques,left=17,right=0)
                bad+=int((need>lim).sum());total+=len(need);worst=max(worst,float(np.max(need-lim)))
            rows.append(dict(family=kind,motor=model,ratio=N,outside_pct=100*bad/total,worst_excess_module_Nm=worst))
result=pd.DataFrame(rows)
result.to_csv(ROOT/'docs/motor_selection_evidence/budget_motor_screen.csv',index=False)
print(result.to_string(index=False))
