#!/usr/bin/env python3
"""Explain a geometry rejection using its contemporaneous observed-map patch."""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path,help='Run directory containing run.events.jsonl')
    parser.add_argument('--event-index',type=int,default=0)
    args=parser.parse_args()
    events=[json.loads(line) for line in (args.run/'run.events.jsonl').read_text().splitlines()]
    events=[e for e in events if e.get('event')=='geometry_guard_rejection' and 'terrain_evidence' in e]
    if not events:
        raise SystemExit('No contemporaneous guard evidence recorded.')
    event=events[args.event_index];data=event['terrain_evidence'];samples=data['samples']
    distances=[s['obstacle_distance_m'] for s in samples if s.get('obstacle_distance_m') is not None]
    analysis=dict(time=event['time'],action_id=event['action_id'],category=event['category'],
        observed_map_revision=data['revision'],required_circle_clearance_m=data['required_circle_clearance_m'],
        minimum_sample_clearance_m=min(distances) if distances else None,
        sampled_oriented_envelopes_clear=all(not s['oriented_support_reason'] for s in samples),
        all_sample_centres_known=all(s.get('known',False) for s in samples),
        all_sample_centres_fresh=all(s.get('fresh',False) for s in samples),
        support_reason=data['support_reason'],grade_reason=data['grade_reason'],
        source='copied observed cells at guard rejection, not the final retained map',
        limitation='Configured footprint geometry; neither a measured physical collision nor traversal success.')
    output=args.run/f'guard-{args.event_index:02d}'
    output.with_suffix('.json').write_text(json.dumps(analysis,indent=2)+'\n')
    render(event,analysis,output.with_suffix('.png'))
    print(json.dumps(analysis))


def render(event,analysis,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle,Polygon,Patch
    from matplotlib.lines import Line2D
    d=event['terrain_evidence'];fields=d['fields'];cell=d['cell_size_m']
    origin=np.asarray(d['grid_origin'])-d['world_size_m']/2
    low=origin+np.asarray(d['patch_first_cell'])*cell
    size=np.asarray(d['patch_shape']);high=low+size*cell
    inclination=np.degrees(np.arctan(np.asarray(fields['slope'],dtype=float)))
    known=np.asarray(fields['known'],dtype=bool)
    hazard=np.asarray(fields['hazard'],dtype=bool)
    fig,ax=plt.subplots(figsize=(10,8),layout='constrained')
    colors=ax.imshow(np.where(known,inclination,np.nan).T,origin='lower',
        extent=[low[0],high[0],low[1],high[1]],vmin=0.,vmax=35.,cmap='viridis',interpolation='nearest')
    fig.colorbar(colors,ax=ax,label='Observed terrain inclination (degrees)',shrink=.8)
    mask=np.zeros((*hazard.shape,4));mask[hazard]=(1.,.15,.12,.88)
    ax.imshow(mask.transpose(1,0,2),origin='lower',extent=[low[0],high[0],low[1],high[1]],interpolation='nearest')
    start,end=np.asarray(d['start']),np.asarray(d['end']);yaw=d['body_yaw']
    rot=np.array([[math.cos(yaw),-math.sin(yaw)],[math.sin(yaw),math.cos(yaw)]])
    corners=np.array([[-1,-1],[-1,1],[1,1],[1,-1]])*np.array([d['half_length_m'],d['half_width_m']])
    for position,alpha in ((start,.45),(end,1.)):
        ax.add_patch(Circle(position,d['required_circle_clearance_m'],fill=False,
                           ec='#ff941f',lw=2.,ls='--',alpha=alpha))
        ax.add_patch(Polygon(corners@rot.T+position,closed=True,fill=False,
                            ec='#49e9ff',lw=2.,alpha=alpha))
    ax.plot([start[0],end[0]],[start[1],end[1]],color='white',lw=2.)
    ax.scatter(*start,c='white',s=55,zorder=5)
    ax.scatter(*end,c='#ff941f',s=55,zorder=5)
    ax.annotate('Command / stopping check',end,xytext=(8,12),textcoords='offset points',color='white',weight='bold')
    for s in d['samples']:
        if s.get('obstacle_distance_m')==analysis['minimum_sample_clearance_m'] and 'nearest_hazard_world' in s:
            near=np.asarray(s['nearest_hazard_world']);point=np.asarray(s['position'])
            ax.plot([point[0],near[0]],[point[1],near[1]],color='#ff5ea0',lw=2.)
            ax.scatter(*near,c='#ff5ea0',marker='x',s=75,zorder=6)
            break
    oriented='clear at every sampled pose' if analysis['sampled_oriented_envelopes_clear'] else 'has a rejected pose'
    clearance=analysis['minimum_sample_clearance_m']
    details=(f'Circle clearance: {clearance:.3f} m; required {d["required_circle_clearance_m"]:.3f} m\n'
             f'Oriented body envelope: {oriented}') if clearance is not None else f'Oriented body envelope: {oriented}'
    ax.set_title(f'Observed evidence at {event["time"]:.2f} s · {event["category"]}\n'+details)
    ax.set_xlabel('World X (m)');ax.set_ylabel('World Y (m)');ax.set_aspect('equal')
    ax.legend(handles=[Patch(color='#ff3026',label='Observed hazard cell'),
        Line2D([],[],color='#ff941f',ls='--',label='Circular planning clearance'),
        Line2D([],[],color='#49e9ff',label='Configured oriented body envelope')],loc='upper left',fontsize=9)
    fig.savefig(output,dpi=160);plt.close(fig)


if __name__=='__main__':main()
