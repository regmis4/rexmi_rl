"""Bounded worker-side archives of the exact inputs consumed by planning."""
import copy
import json
import math
from pathlib import Path
import tempfile
import numpy as np
from .footprint_costs import FIELDS

ARRAYS=('height','surface','known','fresh','blocked','hazard','cost','gx','gy','slope',
    'roughness','obstacle_distance','steep_distance','local_slope','local_fit_support',
    'shelf_mask','failures','successes','approach_blocked','no_go','no_go_depth','last_seen')+FIELDS
LIMIT=16


def repository_input(path):
    """Locate archived inputs after cloning, preserving recorded provenance paths."""
    path=Path(path)
    if path.exists() or not path.is_absolute():return path
    root=Path(__file__).resolve().parents[3]
    for index,part in enumerate(path.parts):
        if part in ('outputs','logs','docs'):
            candidate=root.joinpath(*path.parts[index:])
            if candidate.is_file():return candidate
    return path


def configuration(g):
    config={}
    for name,value in vars(g).items():
        if name.startswith('_'):continue
        if isinstance(value,(str,bool,int)) or isinstance(value,float) and math.isfinite(value):
            config[name]=value
    config.update(origin=list(g.origin),terrain_prior=getattr(g,'terrain_prior',None),
        failure_records=getattr(g,'failure_records',[]))
    return config


def save(control,xy,yaw,now):
    directory=getattr(control,'planning_snapshot_dir',None)
    if directory is None:return None
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if len(list(directory.glob('request-*.npz')))>=LIMIT:
        return {'status':'snapshot limit reached','limit':LIMIT}
    g=control.grid
    config=configuration(g)
    source=getattr(control,'candidate_source',None)
    waypoint=control.waypoints[control.wp_idx] if control.wp_idx<len(control.waypoints) else None
    meta=dict(version=2,candidate_source_config=configuration(source) if source is not None else None,position=list(xy),yaw=yaw,time=now,config=config,
        goal=[waypoint.x,waypoint.y] if waypoint else None,
        protocol=getattr(control,'protocol_version',None),generation=control.plan_generation,
        scope='Frozen planning geometry and route decision; not raw sensor data, controller state, or a simulator state',
        height_reference=getattr(control,'ascent_height_reference',None))
    if getattr(control,'goal_mode','fixed') == 'any_exit':
        meta.update(goal=None,goal_mode='any_exit',exit_rim_radius=control.exit_rim_radius,
                    selected_exit=control.selected_exit)
    # Called in the planning worker, after the control thread has copied inputs.
    from .stage_graph import SOURCE_FIELDS
    source_arrays={} if source is None else {'candidate_source_'+name:getattr(source,name) for name in SOURCE_FIELDS}
    with tempfile.NamedTemporaryFile(dir=directory,prefix='request-',suffix='.npz',delete=False) as out:
        np.savez_compressed(out,metadata_json=json.dumps(meta,allow_nan=False),
            **{name:getattr(g,name) for name in ARRAYS},**source_arrays)
        filename=out.name
    return dict(status='saved',path=filename,time=now,position=list(xy),yaw=yaw)


def load(path):
    """Restore derived fields directly; rebuilding would change the evidence.

    For replay, construct the controller BEFORE assigning this restored grid.
    RemoteAscentController initialization rebuilds footprint costs. This archive
    supports route/geometry diagnosis, not a complete controller restart.
    Assign g.planning_candidate_source to the replay controller too (version2).
    """
    from .ascent_map import AscentTerrainMap
    with np.load(repository_input(path),allow_pickle=False) as saved:
        meta=json.loads(str(saved['metadata_json'].item()));cfg=meta['config']
        g=AscentTerrainMap(**{k:cfg[k] for k in ('world_size','cell_size','origin',
            'footprint_radius','clearance','half_length','half_width')})
        for name,value in cfg.items():setattr(g,name,value)
        for name in ARRAYS:setattr(g,name,saved[name].copy())
        g.planning_candidate_source=None
        source_cfg=meta.get('candidate_source_config')
        if source_cfg is not None:
            from .stage_graph import SOURCE_FIELDS,freeze_candidate_source
            source=copy.copy(g)
            for name,value in source_cfg.items():setattr(source,name,value)
            for name in SOURCE_FIELDS:setattr(source,name,saved['candidate_source_'+name].copy())
            g.planning_candidate_source=freeze_candidate_source(source)
    return g,meta


def finish(record,control,success):
    if not record or record.get('status')!='saved':return
    def clean(value):
        if isinstance(value,dict):return {key:clean(item) for key,item in value.items()}
        if isinstance(value,(list,tuple)):return [clean(item) for item in value]
        if isinstance(value,np.ndarray):return clean(value.tolist())
        if isinstance(value,np.generic):return clean(value.item())
        if isinstance(value,float) and not math.isfinite(value):return None
        return value
    result=dict(success=bool(success),planning_failure=control.planning_failure,
        route=control.path if success else [],route_purpose=control.route_purpose,
        candidates=control.route_options,searches=control.route_searches)
    Path(record['path']).with_suffix('.json').write_text(json.dumps(clean(result),indent=2,allow_nan=False)+'\n')
