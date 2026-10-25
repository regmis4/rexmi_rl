"""Plot measured descent against its retained map; no input to navigation."""
import csv, importlib, json, sys, types
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[3]
pkg=types.ModuleType('navplot');pkg.__path__=[str(root/'source/rexmi_rl')];sys.modules['navplot']=pkg
snapshot=importlib.import_module('navplot.nav.planning_snapshot')
slope=importlib.import_module('navplot.nav.slope_cost')
display=importlib.import_module('navplot.nav.terrain_display')
run=Path(sys.argv[1]);output=Path(sys.argv[2])
g,_=snapshot.load(next((run/'exit-plan-01').glob('request-*.npz')))
model=slope.Model(g,slope.Parameters(),bidirectional=True)
view=display.from_model(model)
rows=[r for r in csv.DictReader((run/'run.csv').open())
      if r['mission_stage']=='DESCEND_TO_CENTER' and float(r['sim_time'])>=1.]
x,y,z,t,speed,up=np.array([[float(r[k]) for k in
    ('x','y','z','sim_time','speed','up_z')] for r in rows]).T
tilt=np.degrees(np.arccos(np.clip(up,-1,1)))
grade=[display.heading_grade(g,(float(r['x']),float(r['y'])),float(r['yaw']),model) for r in rows]
distance=np.r_[0,np.cumsum(np.hypot(np.diff(x),np.diff(y)))]
plt.style.use('dark_background')
fig=plt.figure(figsize=(13,7),facecolor='#0c0f19');gs=fig.add_gridspec(2,2,width_ratios=[1.15,1])
a=fig.add_subplot(gs[:,0]);lo=np.asarray(g.origin)-g.world_size/2
rgb=np.transpose(view['rgb'],(1,0,2));a.imshow(rgb,origin='lower',extent=[lo[0],lo[0]+g.world_size,lo[1],lo[1]+g.world_size])
a.plot(x,y,color=display.ROUTE,lw=2,label='Measured autonomous descent')
a.scatter([x[0],x[-1]],[y[0],y[-1]],c=['white','#8ded52'],s=35,zorder=4)
a.set(xlim=(-2,14),ylim=(-3,10),xlabel='World X (m)',ylabel='World Y (m)',title='Retained measured terrain · magma / log cost')
a.legend(loc='lower left',fontsize=8)
from matplotlib.colors import LogNorm
bar=fig.colorbar(plt.cm.ScalarMappable(norm=LogNorm(view['vmin'],view['vmax']),cmap='magma'),ax=a,orientation='horizontal',fraction=.05,pad=.1)
bar.set_label('Slope cost · log scale; grey: blocked, dark grey: unknown',fontsize=8)
b=fig.add_subplot(gs[0,1]);b.plot(distance,z,color=display.ROUTE,label='Robot elevation')
b.set(xlabel='Measured travel (m)',ylabel='Elevation (m)',title='Distance vs elevation')
b2=b.twinx();b2.plot(distance,grade,color='#ffbc76',alpha=.65,lw=.8)
b2.set_ylabel('Retained-map heading grade (degrees)',color='#ffbc76')
b.grid(alpha=.15)
c=fig.add_subplot(gs[1,1]);c.plot(t,speed,color=display.ROUTE,label='Measured speed');c.set(xlabel='Simulation time (s)',ylabel='Speed (m/s)',title='Physical response')
c2=c.twinx();c2.plot(t,tilt,color='#ffbc76',alpha=.7,label='Body tilt');c2.set_ylabel('Body tilt (degrees)',color='#ffbc76')
c.grid(alpha=.15);c.legend(loc='upper left',fontsize=8);c2.legend(loc='upper right',fontsize=8)
fig.suptitle('AUTONOMOUS DESCENT · '+run.name)
fig.tight_layout();fig.savefig(output,dpi=150);plt.close(fig)
