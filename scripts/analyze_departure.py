#!/usr/bin/env python3
"""Offline geometry and measured-response analysis of an exact departure capture.

Rays are hypotheses, not predictions of policy motion or calibrated success.
No simulator terrain is accessed and copied map arrays are never rebuilt.
"""
import argparse
import csv
import importlib
import json
import math
from pathlib import Path
import sys
from types import ModuleType

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / 'source/rexmi_rl'
for name, directory in [('departure_analysis', ROOT), ('departure_analysis.nav', ROOT/'nav')]:
    package = ModuleType(name)
    package.__path__ = [str(directory)]
    sys.modules[name] = package
D = importlib.import_module('departure_analysis.nav.departure_diagnostics')
S = importlib.import_module('departure_analysis.nav.swept_footprint')
R = importlib.import_module('departure_analysis.nav.ascent_routes')


def analyze(path):
    g, m = D.load_departure(path)
    o = m['observation']; p = np.asarray(o['xy']); yaw = o['yaw']
    req = m['request']
    rows = list(csv.DictReader(path.with_suffix('.csv').with_name('run.csv').open()))
    recent = [r for r in rows if float(r['sim_time']) >= o['time']-2.-1e-9]
    delta = p-np.array([float(recent[0][k]) for k in ('x','y')])
    elapsed = o['time']-float(recent[0]['sim_time'])
    response = math.atan2(delta[1], delta[0])
    angles = dict(body_forward=yaw, body_reverse=yaw+math.pi,
                  original_axis=m['executor']['committed_heading'], response_2s=response)
    result = dict(time=o['time'], position=p.tolist(), lateral=m['geometry'],
        original_endpoint=req['endpoint'], original_endpoint_ray=S.arrival_ray_interval(
            p, req['endpoint'], yaw, req['direction'], req['arrival_radius']),
        measured_response=dict(window_s=elapsed, displacement=delta.tolist(),
            heading_deg=math.degrees(response), height_gain_m=float(recent[-1]['z'])-float(recent[0]['z'])),
        failure_records_before_decision=m['failure_records'], rays=[],
        source='Exact pre-rejection observed arrays; no rebuild or simulator terrain access.',
        limitation='Geometric hypotheses, not controllable trajectories or probabilities.')
    for name, angle in angles.items():
        ray=dict(name=name,heading_deg=math.degrees(angle),samples=[])
        for length in (.4,.8,1.6,2.4,4.,6.,8.,10.,12.):
            end=p+length*np.array([math.cos(angle),math.sin(angle)])
            near=p+min(2.,length)*np.array([math.cos(angle),math.sin(angle)])
            reason=S.oriented_sweep_reason(g,p,end,yaw,require_fresh=False)
            fresh=S.oriented_sweep_reason(g,p,near,yaw)
            profile=g.route_profile([p,end])
            slope=g.max_support_slope(end)
            sample=dict(length_m=length,endpoint=end.tolist(),static_reason=reason,
                fresh_2m_reason=fresh,endpoint_support_deg=slope,
                gentle_staging=slope<=12. and not g.turn_support_reason(end),
                max_uphill_deg=profile['max_uphill_deg'],max_side_slope_deg=profile['max_side_slope_deg'])
            # At most one onward search per ray; future observations must
            # refresh distant surveyed terrain before execution.
            if not reason and sample['gentle_staging'] and 'onward' not in ray:
                route, complete=R.observed_route(g,tuple(end),tuple(m['mission_goals'][m['waypoint_index']]),
                    yaw,require_fresh_start=False,allow_frontier=False,max_expansions=15000)
                ray['onward']=dict(from_endpoint=end.tolist(),route=route,complete=complete,
                    search=g.last_route_search.copy(),survey_only=True)
            ray['samples'].append(sample)
        result['rays'].append(ray)
    return g,m,D.plain(result),rows


def render(g,m,result,rows,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(11,8),layout='constrained')
    low=np.asarray(g.origin)-g.world_size/2;high=low+g.world_size
    extent=[low[0],high[0],low[1],high[1]]
    image=ax.imshow(np.where(g.known,np.degrees(np.arctan(g.slope)),np.nan).T,
        origin='lower',extent=extent,cmap='viridis',vmin=0,vmax=35)
    fig.colorbar(image,ax=ax,label='Observed terrain inclination (degrees)')
    rgba=np.zeros((*g.hazard.shape,4));rgba[g.hazard]=(1,.15,.12,.8)
    ax.imshow(rgba.transpose(1,0,2),origin='lower',extent=extent)
    p=np.asarray(result['position'])
    for ray,color in zip(result['rays'],['#fff','#ffae00','#25d7ff','#ff79ef']):
        a=math.radians(ray['heading_deg']);e=p+8*np.array([math.cos(a),math.sin(a)])
        ax.plot([p[0],e[0]],[p[1],e[1]],'--',color=color,label=ray['name'].replace('_',' '))
        if ray.get('onward',{}).get('complete'):
            route=np.array(ray['onward']['route']);ax.plot(*route.T,color=color,lw=2)
    xy=np.array([[float(r['x']),float(r['y'])] for r in rows])
    ax.plot(*xy.T,color='#ffeb55',lw=2,label='Measured motion')
    start=np.asarray(m['request']['start']);end=np.asarray(m['request']['endpoint'])
    ax.plot([start[0],end[0]],[start[1],end[1]],':',color='#25d7ff',lw=2)
    ax.scatter(*end,c='#25d7ff',marker='x',s=100,label='Original primitive endpoint')
    ax.scatter(*p,c='white',s=55,zorder=5)
    ax.set(xlim=(-4,16),ylim=(-3,14),xlabel='World X (m)',ylabel='World Y (m)',
        title=f'Exact observed terrain at {result["time"]:.2f}s\nDashed rays are geometric hypotheses, not predicted policy trajectories')
    ax.set_aspect('equal');ax.legend(fontsize=9,loc='lower left')
    fig.savefig(path,dpi=150);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture',type=Path)
    args=parser.parse_args()
    grid,metadata,analysis,trace=analyze(args.capture)
    output=args.capture.with_suffix('.analysis.json')
    output.write_text(json.dumps(analysis,indent=2,allow_nan=False)+'\n')
    render(grid,metadata,analysis,trace,args.capture.with_suffix('.png'))
    print(output)
