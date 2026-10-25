#!/usr/bin/env python3
"""Fit terrain preferences to a successful route and a user-rejected corridor.

This is same-crater preference fitting, not held-out validation or a learned
probability of success. Search receives no human intermediate waypoints.
"""
import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
import test_remote_control
from rexmi_bt_tests.nav import costmap_path as cp,planning_snapshot,exit_region,slope_cost
from rexmi_bt_tests.nav.staged_routes import compact_path
from plan_costmap_only import measures
from analyze_manual_ascent import sample_with_end


def resample(path,spacing=.2):
    points=np.asarray(path)
    distance=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    keep=np.r_[True,np.diff(distance)>1e-10];points=points[keep];distance=distance[keep]
    samples=np.r_[np.arange(0,distance[-1],spacing),distance[-1]]
    return np.column_stack([np.interp(samples,distance,points[:,axis]) for axis in (0,1)])


def similarity(route,manual):
    a,b=resample(route),resample(manual)
    return float(.5*(cKDTree(a).query(b)[0].mean()+cKDTree(b).query(a)[0].mean()))


def local_metrics(model,path):
    g=model.grid;lower=np.asarray(g.origin)-g.world_size/2;rows=[]
    for a,b in zip(path,path[1:]):
        delta=np.subtract(b,a);length=np.linalg.norm(delta)
        if length<1e-12:continue
        cells,fractions,_=cp.crossings((np.asarray(a)-lower)/g.cell_size,(np.asarray(b)-lower)/g.cell_size)
        cells=tuple(cells.T)
        grade=np.degrees(np.arctan((model.gx[cells]*delta[0]+model.gy[cells]*delta[1])/length))
        cross=np.degrees(np.arctan(np.abs(-model.gx[cells]*delta[1]+model.gy[cells]*delta[0])/length))
        rows.extend(zip(length*fractions,grade,cross))
    values=np.asarray(rows)
    return dict(local_peak_uphill_deg=float(values[:,1].max()),
        local_metres_above_25deg=float(values[values[:,1]>25,0].sum()),
        local_metres_above_35deg=float(values[values[:,1]>35,0].sum()),
        local_peak_cross_deg=float(values[:,2].max()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reuse-candidates',type=Path,help='Reuse exact wheelbase-model candidates; selected cost is independently rechecked')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    map_path=ROOT/'logs/telemetry/manual-ascent-fr4H8xPa/display_map.npz'
    manual_path=ROOT/'docs/validation/manual_ascent_2026-10-24/baseline_route.csv'
    rejected_path=ROOT/'outputs/remote_autonomy/turn-free-costmap-20261024-01/matched_manual_exit_route.csv'
    g,_=planning_snapshot.load(map_path)
    before={key:hashlib.sha256(getattr(g,key).tobytes()).hexdigest() for key in planning_snapshot.ARRAYS}
    manual=sample_with_end(pd.read_csv(manual_path),5)[['x','y']].to_numpy()
    rejected=pd.read_csv(rejected_path)[['x','y']].to_numpy()
    allowed=g.known & ~g.blocked & np.isfinite(g.gx) & np.isfinite(g.gy)
    lower=np.asarray(g.origin)-g.world_size/2
    candidates=[];began=time.perf_counter()
    if args.reuse_candidates:
        candidates=[json.loads(s) for s in args.reuse_candidates.read_text().splitlines()]
    # Explicit small preference-fit inventory; no route coordinates in the cost.
    for scale,weight,rough,terrain in itertools.product((2.5,4.,6.),(1.,4.,16.),(.5,1.5),(.25,1.,4.)):
        if args.reuse_candidates:break
        p=slope_cost.Parameters(exponential_scale_deg=scale,uphill_weight=weight,
                               roughness_interaction=rough,cross_weight=1.,terrain_weight=terrain)
        model=slope_cost.Model(g,p)
        solution=cp.solve(g.cost,allowed,manual[0],lower,g.cell_size,density=model)
        route,value=cp.path_to_point(solution,g.cost,allowed,manual[-1])
        route=compact_path(route)
        human_parts=model.route_components(manual);rejected_parts=model.route_components(rejected)
        record=dict(parameters=asdict(p),manual_score=sum(human_parts.values()),rejected_score=sum(rejected_parts.values()),
                    manual_preferred=sum(human_parts.values())<sum(rejected_parts.values()),
                    route_score=value,route=route,mean_bidirectional_route_distance_m=similarity(route,manual),
                    geometry=measures(g,route)[0])
        candidates.append(record)
        with (out/'candidates.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        print(json.dumps({k:record[k] for k in ('parameters','manual_preferred','mean_bidirectional_route_distance_m','geometry')}),flush=True)
    # Recorded human motion contains failed attempts and corrective movement.
    # Fit its corridor geometry, not the claim that every executed detour was
    # optimal. Retain the raw manual-vs-rejected score as an honest diagnostic.
    eligible=[r for r in candidates if r['route_score']<r['rejected_score']]
    if not eligible:raise RuntimeError('No generated route is preferred to the rejected corridor')
    chosen=min(eligible,key=lambda r:(r['mean_bidirectional_route_distance_m'],r['parameters']['uphill_weight']))
    parameters=slope_cost.Parameters(**chosen['parameters']);model=slope_cost.Model(g,parameters)
    solution=cp.solve(g.cost,allowed,manual[0],lower,g.cell_size,density=model)
    terminals=[]
    for cell in np.argwhere(allowed & np.isfinite(solution['distance'])):
        point=lower+(cell+.5)*g.cell_size
        if np.linalg.norm(point-g.origin)>=11.8 and not exit_region.reason(g,point,11.):terminals.append(tuple(cell))
    terminal=min(terminals,key=lambda cell:solution['distance'][cell])
    any_exit=compact_path(cp.path_to_cell(solution,terminal))
    routes=dict(any_exit=any_exit,matched_manual_exit=chosen['route'],manual=manual.tolist(),rejected=rejected.tolist())
    metrics={};heights={}
    for name,route in routes.items():
        metrics[name],_,heights[name]=measures(g,route)
        metrics[name]['cost_components']=model.route_components(route)
        metrics[name]['fitted_score']=sum(metrics[name]['cost_components'].values())
        metrics[name]['distance_to_manual_m']=similarity(route,manual)
        metrics[name].update(local_metrics(model,route))
        pd.DataFrame(route,columns=['x','y']).to_csv(out/(name+'_route.csv'),index=False)
    assert math.isclose(metrics['any_exit']['fitted_score'],solution['distance'][terminal],rel_tol=1e-9)
    assert math.isclose(metrics['matched_manual_exit']['fitted_score'],chosen['route_score'],rel_tol=1e-9)
    result=dict(scope='Offline preference fitting on one demonstrated ascent and one user-rejected corridor; no physics run',
        parameters=asdict(parameters),formula='C + A*(expm1(max(uphill_deg-15,0)/scale) + terrain_weight*expm1(max(terrain_deg-15,0)/scale))*(1 + beta*R) + (cross_deg/15)^4',
        roughness='R = 0.5*(detrended footprint RMS/0.02m + footprint step/0.05m); explicit unknown-coverage fallback',
        slope_measurement='Centered terrain-height difference across nominal 0.451m wheelbase on each grid axis; replaces 2.2m displayed-plane gradient for scoring; falls back to displayed gradients at missing interpolation support',
        selection='Among 54 models whose generated route is cheaper than the rejected route, minimize mean bidirectional geometric distance to demonstration. Same start/end for fitting. Any-exit search afterward.',
        raw_trajectory_caveat='The recorded human trajectory includes corrective/failed motion; it is not assumed globally cost-optimal. Its raw score remains reported even when higher than the rejected nominal polyline.',
        turning='Unrestricted: no turn veto, staging requirement or turn penalty',
        masks='Original displayed-map unknown/blocked cells and no corner cutting; no new slope exclusions',
        no_height_reward='Positive edge costs and an exterior destination require useful elevation gain without rewarding loops',
        limitation='Preference fit, not measured traction/energy/probability or evidence of generalization. Human corrective motion is part of training evidence.',
        candidate_count=len(candidates),eligible_count=len(eligible),metrics=metrics,routes=routes,
        reused_candidates=str(args.reuse_candidates) if args.reuse_candidates else None,
        reused_candidates_sha256=hashlib.sha256(args.reuse_candidates.read_bytes()).hexdigest() if args.reuse_candidates else None,
        source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
            (Path(__file__),ROOT/'source/rexmi_rl/nav/slope_cost.py',ROOT/'source/rexmi_rl/nav/costmap_path.py')},
        fit_elapsed_seconds=time.perf_counter()-began,
        inputs={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (map_path,manual_path,rejected_path)},
        map_arrays_unchanged=before=={key:hashlib.sha256(getattr(g,key).tobytes()).hexdigest() for key in planning_snapshot.ARRAYS})
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (out/'parameters.json').write_text(json.dumps(asdict(parameters),indent=2)+'\n')
    render(g,routes,metrics,heights,model,out)
    print(json.dumps(dict(selected=asdict(parameters),metrics=metrics),indent=2))


def render(g,routes,metrics,heights,model,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(16,13),layout='constrained');gs=fig.add_gridspec(3,2,width_ratios=(1.2,1.))
    ax=fig.add_subplot(gs[:,0]);curve=fig.add_subplot(gs[0,1]);height=fig.add_subplot(gs[1,1])
    slope=fig.add_subplot(gs[2,1],sharex=height)
    lower=np.asarray(g.origin)-g.world_size/2
    from slope_cost_plot import draw, COLORS
    color_scale=draw(ax,model)
    colors=COLORS
    labels=dict(any_exit='Fitted: any safe exit',matched_manual_exit='Fitted: manual endpoint',manual='Successful manual',rejected='Rejected direct corridor')
    for name in ('rejected','manual','matched_manual_exit','any_exit'):
        route=np.asarray(routes[name]);m=metrics[name]
        ax.plot(*route.T,c=colors[name],lw=2.4 if name=='any_exit' else 1.8,
            ls='--' if name in ('rejected','matched_manual_exit') else '-',label=f'{labels[name]} ({m["length_m"]:.1f} m)')
        height.plot(*heights[name],c=colors[name] if name!='any_exit' else '#007f9b',lw=2,label=labels[name])
        # Exact cell intervals, using the same wheelbase gradient as scoring.
        # Preserve short steep sections and the sign of downhill corrections.
        lengths=[];grades=[]
        for a,b in zip(route,route[1:]):
            delta=b-a;length=np.linalg.norm(delta)
            if length<1e-12:continue
            cells,fractions,_=cp.crossings((a-lower)/g.cell_size,(b-lower)/g.cell_size)
            cells=tuple(cells.T)
            lengths.extend(length*fractions)
            grades.extend(np.degrees(np.arctan((model.gx[cells]*delta[0]+model.gy[cells]*delta[1])/length)))
        slope.stairs(grades,np.r_[0.,np.cumsum(lengths)],
            baseline=None,color=colors[name] if name!='any_exit' else '#007f9b',lw=1.3,label=labels[name])
        if name in ('any_exit','matched_manual_exit'):ax.scatter(*route[-1],marker='*',s=160,c=colors[name],edgecolors='black',zorder=10)
    ax.scatter(*routes['any_exit'][0],c='white',s=50,edgecolors='black',zorder=10)
    all_points=np.concatenate([np.asarray(route) for route in routes.values()])
    minimum=all_points.min(axis=0)-1.2;maximum=all_points.max(axis=0)+1.2
    ax.set(xlim=(minimum[0],maximum[0]),ylim=(minimum[1],maximum[1]),aspect='equal',xlabel='World X (m)',ylabel='World Y (m)',title='New routes vs demonstrated climb and rejected corridor')
    ax.legend(loc='lower left',fontsize=9,facecolor='#171924',labelcolor='white',framealpha=.95)
    cb=fig.colorbar(color_scale,ax=ax,orientation='horizontal',fraction=.04,pad=.02)
    cb.set_label('Steepest-uphill cost density, log scale; actual route cost also depends on heading')
    angles=np.linspace(0,35,351)
    for rough,label in ((0.,'Smooth reference, R=0'),(1.,'R=1: RMS 2 cm, step 5 cm'),(2.,'R=2: RMS 4 cm, step 10 cm')):
        curve.plot(angles,1+(1+model.parameters.terrain_weight)*slope_cost.uphill_penalty(angles,model.parameters)*(1+model.parameters.roughness_interaction*rough),label=label,lw=2)
    curve.axvspan(0,15,color='#69ba81',alpha=.15);curve.axvline(15,color='gray',ls=':')
    curve.set(xlabel='Uphill grade (degrees)',ylabel='1 + added climb penalty per metre (log scale)',yscale='log',title='Cheap through 15°; exponential climb penalty thereafter')
    curve.legend(fontsize=8);curve.grid(alpha=.2)
    height.set(xlabel='Distance along route (m)',ylabel='Reference terrain height (m)',title='Spreading elevation gain over distance')
    height.legend(fontsize=8);height.grid(alpha=.2)
    slope.axhspan(0,15,color='#69ba81',alpha=.15)
    slope.axhline(0,color='gray',lw=.7)
    for threshold in (15,25,35):slope.axhline(threshold,color='gray',ls=':',lw=.8)
    slope.set(xlabel='Distance along route (m)',ylabel='Along-route slope (degrees)',
        title='Wheelbase-scale slope (0.451 m): uphill + / downhill −')
    slope.legend(fontsize=8,loc='lower right');slope.grid(alpha=.2)
    fig.suptitle('Slope-sensitive preference fit — all turns allowed; no robot trial',fontsize=17)
    fig.savefig(out/'routes_and_costs.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
