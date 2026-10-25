#!/usr/bin/env python3
"""Frozen-map guard ladder; no simulator, robot commands, or manual route input."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
import test_remote_control as fixture
from rexmi_bt_tests.nav import guard_ablation as ga, planning_snapshot, remote_routes, sustained_climb as sc
from rexmi_bt_tests.nav.remote_control import RemoteAscentController
from rexmi_bt_tests.nav.mission import Waypoint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,default=ROOT/'outputs/remote_autonomy/reachable-height-objective/stable_capture/request-d55yep2q.npz')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/remote_autonomy/guard-ablation')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise SystemExit('Use a new output directory; previous trials are preserved.')
    started=time.perf_counter();deadline=started+900.
    g,meta=planning_snapshot.load(args.snapshot)
    before={n:hashlib.sha256(getattr(g,n).tobytes()).hexdigest() for n in planning_snapshot.ARRAYS}
    c=RemoteAscentController(copy.deepcopy(g),[Waypoint(*meta['goal'],.6,'goal')],mode='characterize',
        uphill_limit=33.,side_limit=32.,route_objective='sustained_climb')
    c.grid=g;c.candidate_source=g.planning_candidate_source
    start=tuple(meta['position']);goal=tuple(meta['goal']);yaw=meta['yaw']
    c.obs=fixture.bt.observation(meta['time'],xy=start,yaw=yaw);c.last_yaw=yaw
    def physical(a,b):return remote_routes.physical_profile(c,[a,b])[0]
    result=dict(scope='MAP ONLY. No runtime changes or physical commands.',source=str(args.snapshot.resolve()),
        source_sha256=hashlib.sha256(args.snapshot.read_bytes()).hexdigest(),start=start,goal=goal,yaw=yaw,
        objective=sc.NAME,objective_unchanged=True,overall_wall_budget_s=900,stages=[],
        selection='Strongest tested cumulative guard stage with a complete witness; not global optimality',
        implementation={n:hashlib.sha256((ROOT/'source/rexmi_rl/nav'/n).read_bytes()).hexdigest()
            for n in ('guard_ablation.py','sustained_climb.py','stage_graph.py','turn_staging.py','swept_footprint.py',
                      'wheelbase_demand.py','climb_burden.py','observed_map.py','remote_routes.py',
                      'staged_routes.py','remote_control.py','planning_snapshot.py','ascent_map.py')})
    def save():
        result['elapsed_seconds']=time.perf_counter()-started
        temporary=out/'result.tmp';temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        temporary.replace(out/'result.json')
    def progress(value):print(json.dumps(value),flush=True)
    save()  # Even an incomplete graph build has a preserved outcome/provenance.
    try:
        graph=ga.build(g,start,goal,physical,deadline,progress)
        result['inventory']=graph['inventory'];save()
        selected=None;previous=None
        def report(witness,level):
            states=witness['states'];path=[graph['points'][s[1]] for s in states]
            violations=ga.audit(g,graph,states,yaw)
            assert not any(v['group']<=level for v in violations)
            metrics=sc.describe_route(g,path)
            assert np.allclose(metrics['rank'],witness['rank'],rtol=1e-9,atol=1e-9)
            assert path[0]==start and path[-1]==goal
            return dict(route=path,states=states,rank=list(witness['rank']),provenance=witness['provenance'],
                metrics=metrics,violations=violations,passes_all_static_guards=not violations,
                checked_level=level,complete=True,optimal=False)
        for level,name in enumerate(ga.LEVELS):
            forward,edges,successor,stats=ga.transitions(g,graph,yaw,level)
            def persist_seed(witness):
                result['latest_complete_witness']=dict(witness,level=level,
                    route=[graph['points'][s[1]] for s in witness['states']],
                    status='complete stage witness; independent report pending',optimal=False)
                save()
            witnesses=ga.cheap_witnesses(graph,forward,successor,on_witness=persist_seed)
            if previous and all(t in forward.get(s,()) for s,t in zip(previous['states'],previous['states'][1:])):
                witnesses.append(dict(previous,provenance='retained previous-stage witness'))
            best=min(witnesses,key=lambda w:w['rank']) if witnesses else None
            row=dict(level=level,name=name,connectivity=stats,complete=best is not None,
                optimization='Best of recorded complete witnesses; sustained optimum not certified',
                witnesses=[report(w,level) for w in witnesses])
            if best:
                row['best']=report(best,level);selected=(level,best,forward);previous=best
            result['stages'].append(row);save()
            progress(dict(phase='guard stage',level=level,complete=bool(best),rank=best['rank'] if best else None,
                          edges=stats['edges']))
        forward,edges,successor,stats=ga.transitions(g,graph,yaw,5,legacy_only=True)
        witnesses=ga.cheap_witnesses(graph,forward,successor)
        result['legacy_inventory_control']=dict(connectivity=stats,
            explanation='Same full guards; original candidate inventory only. This is sampling, not an additional safety guard.',
            best=report(min(witnesses,key=lambda w:w['rank']),5) if witnesses else None)
        save()
        if selected:
            level,incumbent,forward=selected
            progress(dict(phase='improve strictest complete stage',level=level,incumbent=incumbent['rank']))
            improved,stats=ga.improve(graph,forward,incumbent)
            final=report(improved,level);final.update(optimization=stats,optimal=stats['optimal'])
            result['selected']=final
            result['selected_level']=level
            if level==5:
                final['independent_runtime_geometry_reason']=sc.route_reason(g,final['route'],yaw,start)
                final['independent_physical_reasons']=[physical(a,b) for a,b in zip(final['route'],final['route'][1:])]
                assert not final['independent_runtime_geometry_reason'] and not any(final['independent_physical_reasons'])
    except TimeoutError as exc:
        result['overall_censored']=True
        result['censor_reason']=str(exc)
        completed=[row for row in result['stages'] if row.get('best')]
        if completed:
            strongest=max(completed,key=lambda row:row['level'])
            result['selected']=dict(strongest['best'],optimization=dict(optimal=False,censored=True,reason=str(exc)))
            result['selected_level']=strongest['level']
        save()
    after={n:hashlib.sha256(getattr(g,n).tobytes()).hexdigest() for n in planning_snapshot.ARRAYS}
    assert before==after
    result.update(map_arrays_unchanged=True,physical_trials=0,
        not_established=['Physical traction and mission completion','Calibrated success probability',
                         'Optimality outside the frozen candidate graph','Dynamic stopping/posture checks from a static map'])
    save();progress(dict(phase='done',level=result.get('selected_level'),
        rank=result.get('selected',{}).get('rank'),optimization=result.get('selected',{}).get('optimization')))


if __name__=='__main__':main()
