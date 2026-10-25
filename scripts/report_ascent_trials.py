#!/usr/bin/env python3
"""Summarize preserved ascent trials and render their observed map/trajectory."""
import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='outputs/ascent_v2')
    parser.add_argument('--output',default='docs/validation/ascent_v2_summary.json')
    parser.add_argument('--plot-run',default=None)
    parser.add_argument('--visual-check',action='append',default=[],
                        help='Folder name of a short overlay check; report separately from mission trials')
    args=parser.parse_args()
    root=Path(args.root);trials=[]
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        summaries=list(folder.glob('*.summary.json'))
        if not summaries:
            failures=list(folder.glob('*.failure.json'))
            if failures:
                trials.extend(dict(name=folder.name,**json.loads(path.read_text())) for path in failures)
                continue
            fault=folder/'interruption.json'
            if not fault.exists(): fault=folder/'infrastructure_failure.json'
            if fault.exists():trials.append(dict(name=folder.name,**json.loads(fault.read_text())))
            continue
        for path in summaries:
            summary=json.loads(path.read_text())
            fields=('result','sim_seconds','controller','ascent_mode','model_id','source_id','seed',
                    'min_up_z','min_active_up_z','peak_command_mps','peak_actual_speed_mps',
                    'peak_active_speed_mps','peak_slope_speed_mps','slope_overspeed_seconds',
                    'manual_intervention','policy_switches','climb_failures','backslide_m',
                    'map_update_ms_mean','controller_ms_mean','controller_ms_p95',
                    'final_state','final_reason','spawn','goal','max_uphill_deg','max_side_slope_deg','terrain_prior',
                    'perception_overlay_requested','perception_overlay_enabled','perception_overlay_tiles',
                    'diagnostic','counts_as_mission','counts_as_qualification')
            record={k:summary[k] for k in fields if k in summary}
            events=path.with_name(path.name.replace('.summary.json','.events.jsonl'))
            outcomes=[json.loads(line) for line in events.read_text().splitlines()] if events.exists() else []
            outcomes=[e for e in outcomes if e.get('event')=='approach_outcome']
            record.update(name=folder.name,summary=str(path),
                          purpose='diagnostic' if summary.get('diagnostic') else 'overlay_check' if folder.name in args.visual_check else 'mission_trial',
                          checkpoint_attempt_successes=sum(e['outcome']=='success' for e in outcomes),
                          checkpoint_attempt_failures=sum(e['outcome']=='failure' for e in outcomes),
                          censored_attempts=sum(e['outcome']=='censored' for e in outcomes))
            record['physical_failures']=sum(e['outcome']=='failure' and e.get('physical',False) for e in outcomes)
            record['recovery_maneuvers']=sum(e.get('event')=='maneuver_start' for e in summary.get('escape_events',[]))
            record['peak_active_tilt_deg']=math.degrees(math.acos(np.clip(summary.get('min_active_up_z',1.),-1,1)))
            csv_path=path.with_name(path.name.replace('.summary.json','.csv'))
            if csv_path.exists():
                with csv_path.open() as stream:rows=list(csv.DictReader(stream))
                active=[row for row in rows if float(row['sim_time'])>=3.]
                if active:
                    record['final_xy']=[float(active[-1][key]) for key in ('x','y')]
                    record['speed_by_terrain']={}
                    slope_key='support_inclination_deg' if 'body_tilt_deg' in active[0] else 'slope_deg'
                    record['speed_terrain_basis']='footprint inclination' if slope_key=='support_inclination_deg' else 'centre terrain inclination'
                    for label,lo,hi in (('flat',0.,5.),('slope_5_to_25',5.,25.),('slope_above_25',25.,91.)):
                        samples=[row for row in active if lo<=float(row.get(slope_key,0.))<hi]
                        if samples:
                            record['speed_by_terrain'][label]=dict(samples=len(samples),
                                peak_command_mps=max(abs(float(row['vx_cmd'])) for row in samples),
                                peak_actual_mps=max(float(row['speed']) for row in samples),
                                actual_p95_mps=float(np.percentile([float(row['speed']) for row in samples],95)))
            trials.append(record)
    manifests=[json.loads(p.read_text()) for p in root.glob('**/gate.json')]
    qualified=any(g.get('stage')=='ascent' and g.get('mode')=='qualified' and g.get('passed')
                  and g.get('consecutive',0)>=3 for g in manifests)
    model_path=root/'diagnostic_model.json'
    model=json.loads(model_path.read_text()).get('evaluation') if model_path.exists() else None
    ascent=[trial for trial in trials if trial.get('controller')=='ascent'
            and trial.get('purpose') not in ('overlay_check','diagnostic')
            and trial.get('counts_as_mission',True) and trial.get('completed',True)]
    result=dict(scope='Ascent development only; descent and performance stages remain gated.',
                ascent_qualified=qualified,descent_started=False,optimization_started=False,
                completed_ascent_trials=len(ascent),
                interrupted_trials=sum(not trial.get('completed',True) for trial in trials),
                completed_ascent_missions=sum(trial.get('result')=='complete' for trial in ascent),
                diagnostic_model_evaluation=model,qualification_manifests=manifests,trials=trials,
                limitation='Characterization outcomes are not calibrated probabilities or qualification successes. Source versions differ; preserved failures are not overwritten.')
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n')
    if args.plot_run:
        render(Path(args.plot_run))
    print(json.dumps(dict(trials=len(trials),report=str(out),ascent_qualified=qualified)))


def render(folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    source=Path(__file__).resolve().parents[1]/'source/rexmi_rl/nav/perception_view.py'
    spec=importlib.util.spec_from_file_location('ascent_report_colors',source)
    view=importlib.util.module_from_spec(spec);spec.loader.exec_module(view)
    snapshot=np.load(folder/'run.map.npz')
    summary=json.loads((folder/'run.summary.json').read_text())
    events=[json.loads(line) for line in (folder/'run.events.jsonl').read_text().splitlines()]
    rows=list(csv.DictReader((folder/'run.csv').open()))
    xy=np.array([[float(row['x']),float(row['y'])] for row in rows])
    cost,known=snapshot['cost'],snapshot['known']
    rgb,legend=view.survey_colors(cost,known,snapshot['clearance'],slope=snapshot['slope'],
                                 no_go=snapshot['no_go'],shelf_mask=snapshot['shelf_mask'])
    n=cost.shape[0];size=n*float(snapshot['cell_size']);origin=snapshot['origin']
    extent=[origin[0]-size/2,origin[0]+size/2,origin[1]-size/2,origin[1]+size/2]
    fig,axes=plt.subplots(1,2,figsize=(14,7),facecolor='#101c28',layout='constrained')
    for ax in axes:
        ax.set_facecolor('#172635');ax.tick_params(colors='white')
        ax.set_xlabel('World X (m)',color='white');ax.set_ylabel('World Y (m)',color='white')
        ax.imshow(rgb.transpose(1,0,2),origin='lower',extent=extent,interpolation='nearest')
        ax.plot(xy[:,0],xy[:,1],color='#ff5464',lw=1.5,label='Measured trajectory')
        ax.scatter(*xy[0],c='white',marker='o',s=40,label='Floor start')
        ax.scatter(*xy[-1],c='#24eaff',marker='x',s=70,label='Final pose')
        ax.scatter(13,0,c='#aaffbd',marker='*',s=130,label='Exterior goal')
        ax.grid(alpha=.12);ax.set_aspect('equal')
    settled=[event['position'] for event in events if event.get('event')=='approach_outcome'
             and event.get('outcome')=='success']
    if settled:
        settled=np.asarray(settled)
        axes[0].scatter(settled[:,0],settled[:,1],s=22,marker='s',facecolors='none',edgecolors='white',label='Reached and settled')
    for index,option in enumerate(summary.get('route_options',[]),1):
        path=np.asarray(option.get('path',[]))
        if len(path)<2:continue
        color='#24eaff' if option.get('selected') else '#e0c0ff'
        axes[1].plot(path[:,0],path[:,1],color=color,ls='--',lw=1.2,alpha=.8)
        axes[1].annotate(f'A{index}',path[-1],xytext=(5,6+10*(index%2)),textcoords='offset points',color=color,fontsize=8)
    # Preserve the initial global planning context even when the final state
    # has cleared its candidate list while recovering or holding.
    initial=next((option for event in events if event.get('event')=='route_comparison'
                  for option in event.get('candidates',[]) if option.get('complete')
                  and option.get('selected') and not option.get('rejection')),None)
    if initial:
        route=np.asarray(initial.get('route_path',initial['path']))
        axes[0].plot(route[:,0],route[:,1],color='#ffc34d',ls='--',lw=1.2,label='Initial selected complete route')
    for failure in summary.get('failed_climbs',[]):
        if 'heading' not in failure:continue
        point=np.asarray(failure['position']);heading=failure['heading']
        delta=.55*np.array([math.cos(heading),math.sin(heading)])
        for ax in axes:
            ax.annotate('',point+delta,point,arrowprops=dict(arrowstyle='->',color='#d18dff',lw=1.6))
            ax.scatter(*point,color='#d18dff',marker='x',s=35,zorder=14)
    prior=summary.get('terrain_prior')
    if prior:
        span=prior.get('radius_m',16.)+1.
        axes[0].set_xlim(origin[0]-span,origin[0]+span)
        axes[0].set_ylim(origin[1]-span,origin[1]+span)
    else:
        axes[0].set_xlim(-6,15);axes[0].set_ylim(-10,10)
    axes[1].set_xlim(xy[-1,0]-3,xy[-1,0]+3);axes[1].set_ylim(xy[-1,1]-3,xy[-1,1]+3)
    axes[0].set_title('Simulator development survey + live updates' if prior and prior['kind']=='simulator_development'
                      else 'Observed survey and actual route',color='white')
    axes[1].set_title('Final approach and local exclusions',color='white')
    axes[0].legend(loc='upper left',fontsize=8)
    hazards=[Patch(facecolor=color,label=label) for color,label in
             [([1.,0.,0.],'Obstacle'),([1.,.29,.12],'>35° terrain'),([.60,.05,.08],'Learned no-go'),
              ([.55,.46,.26],'Obstacle clearance'),([.33,.33,.47],'Unobserved'),([.65,1.,.85],'Shelf boundary')]]
    hazards.append(Line2D([],[],color='#d18dff',marker='x',label='Failed approach direction'))
    axes[1].legend(handles=hazards,loc='upper right',fontsize=7)
    cmap=LinearSegmentedColormap.from_list('terrain',[(.08,.65,.65),(.95,.72,.18),(.85,.15,.45)])
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(1,12),cmap=cmap),ax=axes,shrink=.7,pad=.02)
    cb.set_label('Terrain cost (not a probability)',color='white');cb.ax.tick_params(colors='white')
    fig.suptitle(f'Rexmi ascent characterization · {folder.name} · reachability unvalidated',color='white',fontsize=14)
    fig.supxlabel(f"{summary['result'].upper()} · {summary['sim_seconds']:.1f} simulated seconds · {summary['final_reason']}",
                  color='white',fontsize=10)
    fig.savefig(folder/'observed_route.png',dpi=160,facecolor=fig.get_facecolor())
    plt.close(fig)
    render_release(folder, rows, summary)


def render_release(folder, rows, summary):
    """Measured release intervals; body height is not wheel slip or ground truth traction."""
    import matplotlib.pyplot as plt
    time = np.array([float(r['sim_time']) for r in rows])
    released = np.array([r.get('movement_state') == 'RELEASE' for r in rows])
    if not released.any():
        return
    values = lambda key: np.array([float(r[key]) for r in rows])
    intervals = []
    start = None
    for i, active in enumerate(released):
        if active and start is None: start = i
        if start is not None and (not active or i == len(rows)-1):
            intervals.append((start, i)); start = None
    duration = sum(time[end]-time[start] for start, end in intervals)
    xy = np.column_stack((values('x'), values('y')))
    z = values('z')
    gain = sum(z[end]-z[start] for start, end in intervals)
    distance = sum(np.linalg.norm(xy[end]-xy[start]) for start, end in intervals)
    bounds = (max(0., time[intervals[0][0]]-6.), min(time[-1], time[intervals[-1][1]]+5.))
    visible = (time >= bounds[0]) & (time <= bounds[1])
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 8), layout='constrained')
    axes[0].plot(time[visible], values('vx_cmd')[visible], label='Forward command (m/s)', color='#166d68')
    axes[0].plot(time[visible], values('speed')[visible], label='Actual planar speed (m/s)', color='#da8866')
    axes[0].axhline(.45, linestyle=':', color='#9c4c60', label='Braking threshold')
    axes[0].set_ylabel('Speed (m/s)')
    axes[1].plot(time[visible], z[visible], color='#223f63', label='Measured body height')
    axes[1].set_ylabel('Body height (m)')
    axes[2].plot(time[visible], values('support_inclination_deg')[visible], label='Footprint terrain inclination')
    axes[2].plot(time[visible], values('body_tilt_deg')[visible], label='Body tilt')
    axes[2].set_ylabel('Degrees'); axes[2].set_xlabel('Simulated seconds')
    for ax in axes:
        for start, end in intervals: ax.axvspan(time[start], time[end], color='#739aca', alpha=.18)
        ax.grid(alpha=.2); ax.legend(fontsize=8, loc='upper left')
        ax.set_xlim(*bounds)
    fig.suptitle(f'{folder.name}: policy-active zero-command intervals · {summary["result"]}\n'
                 f'{duration:.2f} s total; {distance:.3f} m summed net XY; {gain:+.3f} m body-height change')
    fig.savefig(folder/'release_response.png', dpi=160)
    plt.close(fig)


if __name__=='__main__':
    main()
