"""Explicit pre-mission terrain information; never imported traversal outcomes.

The simulator survey is development ground truth, not an onboard sensor claim.
Only its construction touches a simulator mesh. Planning consumes the resulting
elevation evidence through the ordinary costmap and refreshes it with live scans.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np


FORMAT_VERSION = 1


def geometry_id(grid):
    digest=hashlib.sha256()
    for value in (np.asarray(grid.origin,dtype=np.float64),np.asarray(grid.cell_size,dtype=np.float64),
                  grid._bin_low,grid._bin_high,grid._bin_count):
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def prior_coverage(grid, radius):
    axis=(np.arange(grid.n_cells)+.5)*grid.cell_size-grid.world_size/2
    area=axis[:,None]**2+axis[None,:]**2<=radius**2
    return float(grid.known[area].mean()) if area.any() else 0.


def simulator_survey(grid, sensor, *, task, seed, radius, device):
    """Sample the whole crater before the mission, without moving the robot."""
    import torch
    from isaaclab.utils.warp import raycast_mesh

    if not math.isfinite(radius) or not 1.<radius<grid.world_size/2-1.:
        raise ValueError('Simulator survey radius must fit within the costmap')
    spacing=grid.cell_size/grid.SUBDIVISIONS
    extent=math.ceil(radius/grid.cell_size)*grid.cell_size
    axis=np.arange(-extent+spacing/2,extent,spacing,dtype=np.float64)
    xx,yy=np.meshgrid(axis+grid.origin[0],axis+grid.origin[1],indexing='ij')
    starts=np.column_stack((xx.ravel(),yy.ravel(),np.full(xx.size,50.))).astype(np.float32)
    mesh=sensor.meshes[sensor.cfg.mesh_prim_paths[0]]
    hit_count=0
    for offset in range(0,len(starts),65536):
        rays=torch.as_tensor(starts[offset:offset+65536],device=device)
        directions=torch.zeros_like(rays);directions[:,2]=-1.
        hits=raycast_mesh(rays,directions,mesh,max_dist=100.)[0].cpu().numpy()
        hit_count+=int(np.isfinite(hits).all(axis=1).sum())
        grid.ingest(hits,(*grid.origin,50.),0.,max_range=100.)
    grid.rebuild(0.)
    # Prior geometry is available globally; it is not a fresh robot scan.
    grid.last_seen[:]=-np.inf
    grid.rebuild(0.)
    metadata=dict(version=FORMAT_VERSION,kind='simulator_development',task=task,seed=int(seed),
                  coordinate_frame='world',radius_m=radius,sample_spacing_m=spacing,
                  ray_count=len(starts),hit_count=hit_count,
                  description='Top-down simulator terrain mesh survey; development ground truth, not onboard LiDAR',
                  geometry_id=geometry_id(grid),coverage=prior_coverage(grid,radius))
    if metadata['coverage']<.95:
        raise ValueError(f"Simulator survey only covers {metadata['coverage']:.1%} of the requested area")
    grid.terrain_prior=metadata
    return metadata


def load_prior(grid, path, *, task, seed, required_radius=None):
    """Restore geometry into a fresh map, recomputing costs for this robot."""
    if grid.samples.any() or grid.approach_blocked.any() or grid.no_go.any():
        raise ValueError('Load a terrain prior only into a fresh mission map')
    with np.load(path,allow_pickle=False) as saved:
        if 'terrain_prior_json' not in saved:
            raise ValueError('Terrain prior needs labeled source/task/seed metadata')
        metadata=json.loads(str(saved['terrain_prior_json'].item()))
        if (metadata.get('version')!=FORMAT_VERSION or metadata.get('coordinate_frame')!='world'
                or metadata.get('task')!=task or metadata.get('seed')!=seed
                or metadata.get('kind') not in ('simulator_development','retained_sensor_survey')):
            raise ValueError('Terrain prior source, task, seed or coordinate frame does not match this mission')
        if not np.allclose(saved['origin'],grid.origin) or not np.isclose(float(saved['cell_size']),grid.cell_size):
            raise ValueError('Terrain prior grid resolution/origin mismatch')
        arrays={}
        for name,destination in (('bin_low','_bin_low'),('bin_high','_bin_high'),('bin_count','_bin_count')):
            value=np.asarray(saved[name])
            if value.shape!=getattr(grid,destination).shape:
                raise ValueError('Terrain prior grid dimensions mismatch')
            arrays[destination]=value.copy()
        counts=arrays['_bin_count'];low=arrays['_bin_low'];high=arrays['_bin_high']
        measured=counts>0
        if (not measured.any() or not np.isfinite(low[measured]).all() or not np.isfinite(high[measured]).all()
                or (low[measured]>high[measured]).any() or (counts<0).any() or (counts>32).any()):
            raise ValueError('Terrain prior contains invalid elevation evidence')
        for name,value in arrays.items():getattr(grid,name)[:]=value
    touched=measured.any(axis=-1)
    grid.low[touched]=np.min(np.where(measured,low,np.inf),axis=-1)[touched]
    grid.height[touched]=np.max(np.where(measured,high,-np.inf),axis=-1)[touched]
    grid.samples[touched]=1
    grid.last_seen[:]=-np.inf
    grid.rebuild(0.)
    if geometry_id(grid)!=metadata.get('geometry_id'):
        raise ValueError('Terrain prior geometry checksum mismatch')
    if required_radius is not None:
        coverage=prior_coverage(grid,required_radius)
        if not 1.<required_radius<grid.world_size/2-1. or coverage<.95:
            raise ValueError(f'Terrain prior covers only {coverage:.1%} of the required mission area')
        metadata.update(mission_radius_m=required_radius,mission_coverage=coverage)
    grid.terrain_prior=dict(metadata,loaded_from=str(Path(path).resolve()),
                            file_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
    return grid.terrain_prior


def prior_snapshot(grid):
    snapshot=grid.survey_snapshot()
    metadata=dict(grid.terrain_prior)
    current=geometry_id(grid)
    if current!=metadata['geometry_id']:
        metadata['parent_geometry_id']=metadata['geometry_id']
        metadata['geometry_id']=current
        metadata['description']+='; refreshed with mission sensor observations'
    snapshot['terrain_prior_json']=np.asarray(json.dumps(metadata,sort_keys=True))
    return snapshot
