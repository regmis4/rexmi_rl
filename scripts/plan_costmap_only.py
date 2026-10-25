#!/usr/bin/env python3
"""Turn-free cost-map route study; never launches or commands the robot."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
import test_remote_control  # Pure navigation imports without simulator startup.
from rexmi_bt_tests.nav import costmap_path, planning_snapshot, exit_region
from rexmi_bt_tests.nav.staged_routes import compact_path
from analyze_manual_ascent import path_cost_profile, sample_with_end


def measures(g,points):
    lower=np.asarray(g.origin)-g.world_size/2
    profile=path_cost_profile(points,g.cost,g.known,lower,g.cell_size)
    rows=[]
    for a,b in zip(points,points[1:]):
        delta=np.subtract(b,a);length=np.linalg.norm(delta)
        if length<1e-10:continue
        cells,fractions,_=costmap_path.crossings((np.asarray(a)-lower)/g.cell_size,(np.asarray(b)-lower)/g.cell_size)
        for cell,f in zip(cells,fractions):
            cell=tuple(cell);gx,gy=g.gx[cell],g.gy[cell]
            along=(gx*delta[0]+gy*delta[1])/length
            across=abs(-gx*delta[1]+gy*delta[0])/length
            rows.append((length*f,math.degrees(math.atan(along)),math.degrees(math.atan(across)),bool(g.blocked[cell])))
    data=np.asarray(rows)
    metrics=dict(length_m=float(profile.end_m.iloc[-1]),displayed_cost_m=float(profile.cumulative_cost.iloc[-1]),
        peak_uphill_deg=float(data[:,1].max()),metres_above_25deg=float(data[data[:,1]>25,0].sum()),
        peak_cross_slope_deg=float(data[:,2].max()),metres_in_display_exclusions=float(data[data[:,3]>0,0].sum()),
        start=list(points[0]),end=list(points[-1]))
    sampled=[];distance=[];travelled=0.
    for a,b in zip(points,points[1:]):
        length=math.dist(a,b)
        for t in np.linspace(0,1,max(2,math.ceil(length/.05)+1))[:-1]:
            sampled.append(np.asarray(a)+t*np.subtract(b,a));distance.append(travelled+t*length)
        travelled+=length
    sampled.append(np.asarray(points[-1]));distance.append(travelled)
    coords=((np.array(sampled)-lower)/g.cell_size-.5).T
    height=map_coordinates(g.surface,coords,order=1,mode='nearest')
    return metrics,profile,(np.array(distance),height)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--map',type=Path,default=ROOT/'logs/telemetry/manual-ascent-fr4H8xPa/display_map.npz')
    p.add_argument('--manual',type=Path,default=ROOT/'docs/validation/manual_ascent_2026-10-24/baseline_route.csv')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--radius',type=int,default=3,choices=(1,2,3,4))
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'result.json').exists():raise SystemExit('Use a new output directory')
    g,meta=planning_snapshot.load(args.map)
    before={key:hashlib.sha256(getattr(g,key).tobytes()).hexdigest() for key in planning_snapshot.ARRAYS}
    manual=sample_with_end(pd.read_csv(args.manual),5)[['x','y']].to_numpy()
    start,end=manual[0],manual[-1]
    allowed=g.known & ~g.blocked & np.isfinite(g.cost) & np.isfinite(g.surface)
    lower=np.asarray(g.origin)-g.world_size/2
    began=time.perf_counter()
    solution=costmap_path.solve(g.cost,allowed,start,lower,g.cell_size,radius=args.radius)
    reachable=np.argwhere(allowed & np.isfinite(solution['distance']))
    exterior=[]
    minimum_radius=11.+.25+max(g.footprint_radius,math.hypot(g.half_length,g.half_width))
    for cell in reachable:
        xy=lower+(cell+.5)*g.cell_size
        if np.linalg.norm(xy-g.origin)>=minimum_radius and not exit_region.reason(g,xy,11.):
            exterior.append(tuple(cell))
    if not exterior:raise RuntimeError('No supported gentle exterior is connected on this cost map')
    chosen=min(exterior,key=lambda cell:solution['distance'][cell])
    best=compact_path(costmap_path.path_to_cell(solution,chosen))
    matched,matched_cost=costmap_path.path_to_point(solution,g.cost,allowed,end)
    if not matched:raise RuntimeError('Manual endpoint has no cost-map route')
    matched=compact_path(matched)
    routes={'any_exit':best,'matched_manual_exit':matched,'manual':manual.tolist()}
    colors={'any_exit':'#00d9ff','matched_manual_exit':'#ff9c24','manual':'#73f240'}
    labels={'any_exit':'Cheapest safe exit','matched_manual_exit':'Cheapest to manual endpoint','manual':'Successful manual run'}
    metrics={};profiles={};heights={}
    for name,points in routes.items():
        metrics[name],profiles[name],heights[name]=measures(g,points)
        pd.DataFrame(points,columns=['x','y']).to_csv(args.output/(name+'_route.csv'),index=False)
        profiles[name].to_csv(args.output/(name+'_cost_profile.csv'),index=False)
    assert math.isclose(metrics['any_exit']['displayed_cost_m'],solution['distance'][chosen],rel_tol=1e-10)
    assert math.isclose(metrics['matched_manual_exit']['displayed_cost_m'],matched_cost,rel_tol=1e-10)
    unchanged=before=={key:hashlib.sha256(getattr(g,key).tobytes()).hexdigest() for key in planning_snapshot.ARRAYS}
    result=dict(scope='Offline path selection only; no execution or recovery experiment',
        objective='Exact integral of displayed cost C(x,y) ds; no extra penalties',
        turn_restrictions='NONE: no heading state, staging/minimum requirement, turn angle/slope/rate bound or turn penalty',
        remaining_mask='Original known/unblocked displayed cells; no corner cutting. No additional body-turn footprint or directional-grade veto.',
        exit_criterion='Full footprint outside radius11m plus0.25m reserve; supported hazard-free <=15degree exterior',
        optimality='Global minimum in the specified finite graph; not a continuous-space optimum or physical execution guarantee',
        graph=dict(nodes=solution['nodes'],directed_edges=solution['directed_edges'],headings=solution['headings'],
                   cell_size_m=g.cell_size,max_link_cell_radius=args.radius,safe_exterior_candidates=len(exterior)),
        elapsed_seconds=time.perf_counter()-began,map_arrays_unchanged=unchanged,
        map=str(args.map.resolve()),map_sha256=hashlib.sha256(args.map.read_bytes()).hexdigest(),
        manual_source=str(args.manual.resolve()),manual_sha256=hashlib.sha256(args.manual.read_bytes()).hexdigest(),
        manual_usage='Start pose and separate matched-endpoint comparison; measured intermediate route not supplied to the solver',
        metrics=metrics,routes=routes,grade_scope='Static reference gradients along geometric travel; manual body yaw can differ')
    (args.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    render(g,routes,metrics,profiles,heights,colors,labels,args.output)
    print(json.dumps({k:result[k] for k in ('graph','elapsed_seconds','metrics','map_arrays_unchanged')},indent=2))


def render(g,routes,metrics,profiles,heights,colors,labels,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    fig=plt.figure(figsize=(16,10),layout='constrained')
    gs=fig.add_gridspec(2,2,width_ratios=(1.35,1.))
    ax=fig.add_subplot(gs[:,0]);height=fig.add_subplot(gs[0,1]);cost=fig.add_subplot(gs[1,1])
    lower=np.asarray(g.origin)-g.world_size/2;upper=lower+g.world_size
    extent=[lower[0],upper[0],lower[1],upper[1]]
    rgba=plt.cm.plasma(Normalize(1,20)(g.cost.T));rgba[g.blocked.T]=(.28,.29,.32,1.);rgba[~g.known.T]=(0.12,.13,.15,1.)
    ax.imshow(rgba,origin='lower',extent=extent)
    x=lower[0]+(np.arange(g.n_cells)+.5)*g.cell_size;y=lower[1]+(np.arange(g.n_cells)+.5)*g.cell_size
    contours=ax.contour(x,y,np.where(g.known,g.surface,np.nan).T,levels=np.arange(0.,6.,.5),colors='white',alpha=.4,linewidths=.6)
    ax.clabel(contours,levels=np.arange(1.,5.),inline=True,fontsize=8,fmt='%.0f m')
    ax.add_patch(plt.Circle(g.origin,11.,fill=False,edgecolor='white',ls=':',lw=1.,alpha=.65))
    for name in ('manual','matched_manual_exit','any_exit'):
        xy=np.asarray(routes[name]);m=metrics[name]
        ax.plot(*xy.T,color=colors[name],lw=2.6 if name=='any_exit' else 1.8,
            ls='--' if name=='matched_manual_exit' else '-',label=f'{labels[name]}: {m["length_m"]:.1f} m / {m["displayed_cost_m"]:.1f} cost·m')
        if name!='manual':ax.scatter(*xy[-1],c=colors[name],marker='*',s=190,edgecolor='black',lw=.7,zorder=10)
        distance,z=heights[name]
        height.plot(distance,z,color=colors[name] if name!='any_exit' else '#007f9b',lw=2,label=labels[name])
        p=profiles[name]
        cost.plot(np.r_[0,p.end_m],np.r_[0,p.cumulative_cost],color=colors[name] if name!='any_exit' else '#007f9b',lw=2,label=labels[name])
    ax.scatter(*routes['any_exit'][0],c='white',s=65,edgecolor='black',zorder=10)
    ax.set(xlim=(-15,15),ylim=(-15,15),aspect='equal',xlabel='World X (m)',ylabel='World Y (m)',title='All turns allowed · original displayed map')
    ax.legend(loc='lower left',fontsize=9,facecolor='#171924',labelcolor='white',framealpha=.95)
    height.set(xlabel='Distance along route (m)',ylabel='Reference terrain height (m)',title='Where each route gains height')
    cost.set(xlabel='Distance along route (m)',ylabel='Accumulated displayed cost (cost·m)',title='Same cost definition for all three paths')
    for panel in (height,cost):panel.grid(alpha=.2);panel.legend(fontsize=8)
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(1,20),cmap='plasma'),ax=ax,orientation='horizontal',fraction=.04,pad=.025)
    cb.set_label('Displayed cost: dark = low, yellow = high; gray = original map exclusions')
    fig.suptitle('Path selection first: minimum displayed terrain cost, zero turning restrictions',fontsize=17)
    fig.savefig(out/'routes.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
