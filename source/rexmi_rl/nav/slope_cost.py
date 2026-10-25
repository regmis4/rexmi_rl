"""Preference-fit ascent difficulty, not a traction or success model.

Positive per-metre costs permit contour travel and spread elevation gain over
distance without a negative elevation reward or a prescribed human corridor.
"""
from dataclasses import dataclass,asdict
import math
import numpy as np


@dataclass(frozen=True)
class Parameters:
    easy_angle_deg: float=15.
    exponential_scale_deg: float=6.
    uphill_weight: float=1.
    roughness_interaction: float=.5
    cross_weight: float=1.
    terrain_weight: float=1.

    def __post_init__(self):
        values=list(asdict(self).values())
        if not np.isfinite(values).all() or not 0<=self.easy_angle_deg<35 or self.exponential_scale_deg<=0:
            raise ValueError('Invalid slope cost parameters')
        if min(self.uphill_weight,self.roughness_interaction,self.cross_weight,self.terrain_weight)<0:
            raise ValueError('Difficulty weights must be nonnegative')


def uphill_penalty(angle,parameters):
    excess=np.maximum(0.,np.asarray(angle)-parameters.easy_angle_deg)
    return parameters.uphill_weight*np.expm1(excess/parameters.exponential_scale_deg)


class Model:
    def __init__(self,grid,parameters,*,bidirectional=False):
        self.grid=grid;self.parameters=parameters
        self.bidirectional=bidirectional
        # The displayed plane uses a 2.2 m neighborhood, which hides short
        # climbing faces. Price slope at the nominal 0.451 m wheelbase instead.
        from scipy.ndimage import map_coordinates
        from .wheelbase_demand import WHEELBASE_M
        key=(grid.revision if hasattr(grid,'revision') else None,WHEELBASE_M)
        if getattr(grid,'_slope_cost_geometry_key',None)!=key:
            rr,cc=np.indices(grid.cost.shape,dtype=float);shift=WHEELBASE_M/(2*grid.cell_size)
            x=(map_coordinates(grid.surface,[rr+shift,cc],order=1,mode='constant',cval=np.nan)
               -map_coordinates(grid.surface,[rr-shift,cc],order=1,mode='constant',cval=np.nan))/WHEELBASE_M
            y=(map_coordinates(grid.surface,[rr,cc+shift],order=1,mode='constant',cval=np.nan)
               -map_coordinates(grid.surface,[rr,cc-shift],order=1,mode='constant',cval=np.nan))/WHEELBASE_M
            grid._slope_cost_geometry=(np.where(np.isfinite(x),x,grid.gx),np.where(np.isfinite(y),y,grid.gy))
            grid._slope_cost_geometry_key=key
        self.gx,self.gy=grid._slope_cost_geometry
        self.terrain_angle=np.degrees(np.arctan(np.hypot(self.gx,self.gy)))
        # Detrended footprint-scale relief; missing coverage is not smoothness.
        rms=np.where(np.isfinite(grid.footprint_rms),grid.footprint_rms,.08)
        step=np.where(np.isfinite(grid.footprint_step),grid.footprint_step,.15)
        self.roughness=.5*(rms/.02+step/.05)

    def components(self,cells,dx,dy):
        g,p=self.grid,self.parameters
        length=math.hypot(dx,dy)
        if length<1e-14:
            return dict(base=np.asarray(g.cost[cells],float),uphill=np.zeros_like(g.cost[cells]),
                        terrain=np.zeros_like(g.cost[cells]),rough_slope=np.zeros_like(g.cost[cells]),cross=np.zeros_like(g.cost[cells]))
        grade=np.degrees(np.arctan((self.gx[cells]*dx+self.gy[cells]*dy)/length))
        if self.bidirectional:grade=np.abs(grade)
        cross=np.degrees(np.arctan(np.abs(-self.gx[cells]*dy+self.gy[cells]*dx)/length))
        uphill=uphill_penalty(grade,p)
        terrain=p.terrain_weight*uphill_penalty(self.terrain_angle[cells],p)
        return dict(base=np.asarray(g.cost[cells],float),uphill=uphill,terrain=terrain,
                    rough_slope=(uphill+terrain)*p.roughness_interaction*self.roughness[cells],
                    cross=p.cross_weight*(cross/15.)**4)

    def __call__(self,cells,dx,dy):
        return sum(self.components(cells,dx,dy).values())

    def route_components(self,path):
        from .costmap_path import crossings
        g=self.grid;lower=np.asarray(g.origin)-g.world_size/2
        totals={name:0. for name in ('base','uphill','terrain','rough_slope','cross')}
        for a,b in zip(path,path[1:]):
            delta=np.subtract(b,a);length=np.linalg.norm(delta)
            if length<1e-12:continue
            cells,fractions,_=crossings((np.asarray(a)-lower)/g.cell_size,(np.asarray(b)-lower)/g.cell_size)
            for key,values in self.components(tuple(cells.T),*delta).items():
                totals[key]+=float(np.sum(values*fractions)*length)
        return totals
