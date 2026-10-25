#!/usr/bin/env python3
"""One archived-map any-exit plan; no simulator or human route input."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
import test_remote_control as fixture
from rexmi_bt_tests.nav import planning_snapshot, exit_region, remote_routes, traversal_objective
from rexmi_bt_tests.nav.mission import Waypoint
from rexmi_bt_tests.nav.remote_control import RemoteAscentController


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,default=ROOT/'outputs/remote_autonomy/reachable-height-objective/stable_capture/request-d55yep2q.npz')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    target=args.output/'result.json'
    if target.exists():raise SystemExit('Use a new output directory')
    g,meta=planning_snapshot.load(args.snapshot)
    before={k:hashlib.sha256(getattr(g,k).tobytes()).hexdigest() for k in planning_snapshot.ARRAYS}
    c=RemoteAscentController(copy.deepcopy(g),[Waypoint(13.,0.,.6,'Any safe exterior exit')],
        mode='characterize',goal_mode='any_exit',uphill_limit=33.,side_limit=32.)
    c.grid=g;c.candidate_source=g.planning_candidate_source
    # Current implementation on archived geometry; preserve archived arrays.
    g.checked_small_turns=True
    c.obs=fixture.bt.observation(meta['time'],xy=tuple(meta['position']),yaw=meta['yaw'])
    c.last_yaw=meta['yaw']
    started=time.perf_counter()
    result=dict(scope='MAP ONLY; no robot commands or physical mission success',
        source=str(args.snapshot.resolve()),source_sha256=hashlib.sha256(args.snapshot.read_bytes()).hexdigest(),
        position=meta['position'],yaw=meta['yaw'],goal_mode='any_exit',rim_radius_m=c.exit_rim_radius,
        protocol=c.protocol_version,configuration_override=dict(checked_small_turns=True),
        status='planning',physical_trials=0)
    target.write_text(json.dumps(result,indent=2)+'\n')
    success=c._compute_plan(tuple(meta['position']),meta['yaw'],meta['time'])
    result.update(status='complete route' if success else 'no route',elapsed_seconds=time.perf_counter()-started,
        selected_exit=c.selected_exit,path=c.path if success else [],planner=c.graph_plan_stats,
        mission_checkpoint=c.is_mission_checkpoint,planning_failure=c.planning_failure,
        map_arrays_unchanged=before=={k:hashlib.sha256(getattr(g,k).tobytes()).hexdigest() for k in planning_snapshot.ARRAYS})
    if success:
        result['terrain_profile']=g.route_profile(c.path)
        result['cost_components']=traversal_objective.route_cost(g,c.path,meta['yaw'],breakdown=True)
        result['legs']=[dict(start=a,end=b,terrain_profile=g.route_profile([a,b]),
            cost_components=traversal_objective.segment_cost(g,a,b,breakdown=True)) for a,b in zip(c.path,c.path[1:])]
        result['independent_route_rejection']=exit_region.route_reason(g,c.path,meta['yaw'],meta['position'],
            c.exit_rim_radius,lambda a,b:remote_routes.physical_profile(c,[a,b])[0])
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,8),layout='constrained')
    half=g.world_size/2.;extent=[g.origin[0]-half,g.origin[0]+half,g.origin[1]-half,g.origin[1]+half]
    mesh=ax.imshow(np.where(g.known,g.cost,np.nan).T,origin='lower',extent=extent,vmin=1,vmax=20,cmap='plasma')
    mask=np.zeros((*g.blocked.shape,4));mask[g.blocked]=(.25,.25,.25,1.)
    ax.imshow(mask.transpose(1,0,2),origin='lower',extent=extent)
    coords=g.origin[0]-half+(np.arange(g.n_cells)+.5)*g.cell_size
    ycoords=g.origin[1]-half+(np.arange(g.n_cells)+.5)*g.cell_size
    ax.contour(coords,ycoords,np.where(g.known,g.surface,np.nan).T,levels=np.arange(0.,7.),colors='white',alpha=.35,linewidths=.6)
    terminals=np.asarray(exit_region.candidates(g,c.exit_rim_radius))
    if len(terminals):ax.scatter(*terminals.T,s=12,c='white',label='Supported exterior candidates')
    if success:
        points=np.asarray(c.path);ax.plot(*points.T,c='cyan',lw=2,label='Independent any-exit plan')
        ax.scatter(*points[-1],c='lime',s=70,label='Selected exit')
    ax.set(xlim=(-16,16),ylim=(-16,16),xlabel='World X (m)',ylabel='World Y (m)',
        title='Any safe exit — static prediction, not physical validation',aspect='equal')
    ax.legend();fig.colorbar(mesh,ax=ax,label='Displayed terrain cost (1–20)')
    fig.savefig(args.output/'route.png',dpi=150);plt.close(fig)
    print(json.dumps({k:result[k] for k in ('status','elapsed_seconds','selected_exit','map_arrays_unchanged')}))


if __name__=='__main__':main()
