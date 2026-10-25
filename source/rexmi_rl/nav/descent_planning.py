"""Exponential slope routing with explicit observed-space checkpoint reserves."""
import math
import numpy as np
from scipy.ndimage import binary_closing, distance_transform_edt
from . import costmap_path, slope_cost, footprint_costs

PATH_RESERVE_M=.10
CHECKPOINT_RESERVE_M=.15
FRONTIER_MAX_SLOPE_DEG=20.
FRONTIER_REVERSE_M=.6
FRONTIER_STEP_M=2.


def measured_coverage(g):
    measured=np.isfinite(g.height)
    return g.known & (measured | binary_closing(measured,structure=np.ones((3,3),bool)))


def refresh(control):
    g=control.grid
    if getattr(control,'descent_map_revision',None)==g.revision:return
    g.footprint_costs_enabled=True;footprint_costs.rebuild(g)
    control.descent_model=slope_cost.Model(g,slope_cost.Parameters(),bidirectional=True)
    # The display map extrapolates fitted planes around sparse returns. That
    # interpolation must not move the evidence frontier toward an unseen cliff.
    # Close only small sampling gaps; never use the extrapolated outer band.
    observed=measured_coverage(g) & np.isfinite(g.surface) & np.isfinite(g.gx) & np.isfinite(g.gy)
    distance=distance_transform_edt(np.pad(observed,1))[1:-1,1:-1]*g.cell_size
    # Account for the cell square as well as the robot's circular envelope.
    margin=g.footprint_radius+g.clearance+g.cell_size*math.sqrt(.5)
    control.observed_distance=distance
    control.descent_allowed=(observed & ~g.blocked & g.planning_support()
                             & (distance>=margin+PATH_RESERVE_M))
    # On easy ground approach the sampled edge; increase stand-off smoothly
    # above 15 degrees. Extrapolation still contributes no observed clearance.
    control.checkpoint_reserve=CHECKPOINT_RESERVE_M+.02*np.maximum(0.,control.descent_model.terrain_angle-15.)
    control.checkpoint_supported=control.descent_allowed & (distance>=margin+control.checkpoint_reserve)
    control.descent_map_revision=g.revision


def route_score(control, route):
    refresh(control);g=control.grid
    if len(route)<2:return math.inf
    lower=np.asarray(g.origin)-g.world_size/2
    values=[costmap_path.edge_cost(g.cost,control.descent_allowed,
        (np.asarray(a)-lower)/g.cell_size,(np.asarray(b)-lower)/g.cell_size,g.cell_size,
        density=control.descent_model) for a,b in zip(route,route[1:])]
    return sum(values)


def recoverable_frontier(control, route):
    """The last approach has a measured, gentle 0.6 m straight backout."""
    if len(route)<2:return False
    g=control.grid;end=np.asarray(route[-1])
    delta=end-np.asarray(route[-2]);length=np.linalg.norm(delta)
    if length<1e-8:return False
    axis=delta/length;behind=end-FRONTIER_REVERSE_M*axis
    lower=np.asarray(g.origin)-g.world_size/2
    # Check all touched cells, not only a sampled endpoint.
    _,_,cells=costmap_path.crossings((behind-lower)/g.cell_size,(end-lower)/g.cell_size)
    if (cells<0).any() or (cells>=g.cost.shape).any():return False
    return bool(control.descent_allowed[tuple(cells.T)].all() and
        (control.descent_model.terrain_angle[tuple(cells.T)]<=FRONTIER_MAX_SLOPE_DEG).all())


def local_step(control, solution, xy, goal, yaw):
    """Rank short goalward steps; no reward for unknown area or rim touring."""
    g=control.grid;lower=solution['lower']
    cells=np.argwhere(control.checkpoint_supported & np.isfinite(solution['distance']) &
                     (control.descent_model.terrain_angle<=FRONTIER_MAX_SLOPE_DEG))
    points=lower+(cells+.5)*g.cell_size
    delta=points-xy;travel=np.linalg.norm(delta,axis=1)
    advance=math.dist(xy,goal)-np.linalg.norm(points-goal,axis=1)
    # Only genuinely goalward local motion; no backwards/lateral-only fallback.
    valid=(travel>=.55) & (travel<=FRONTIER_STEP_M) & (advance>=.25) & (advance>=.5*travel)
    headings=np.arctan2(delta[:,1],delta[:,0])
    turns=np.abs(np.arctan2(np.sin(headings-yaw),np.cos(headings-yaw)))
    cost=solution['distance'][tuple(cells.T)]
    scores=advance-.25*(travel-advance)-.025*cost-.15*turns
    order=np.argsort(np.where(valid,scores,-np.inf))[::-1]
    choices=[]
    for index in order[:48]:
        if not valid[index]:break
        route=costmap_path.path_to_cell(solution,tuple(cells[index]))
        direct=[tuple(xy),route[-1]]
        # Avoid needless grid bends when the same observed corridor is clear
        # and its exponential cost is within 10% of the sampled optimum.
        if route_score(control,direct)<=1.1*cost[index]:route=direct
        length=sum(math.dist(a,b) for a,b in zip(route,route[1:]))
        if length>1.5*travel[index] or length>2.5 or not recoverable_frontier(control,route):continue
        radii=[math.dist(p,goal) for p in route]
        if max(radii)>radii[0]+.1:continue
        choices.append((scores[index]-.25*(length-travel[index]),route))
    return max(choices,key=lambda item:item[0])[1] if choices else []


def choose_route(control, xy, goal, yaw=0.):
    refresh(control);g=control.grid;control.waiting_for_map=False
    control.route_options=[];control.route_searches=[]
    control.observation_target=None
    lower=np.asarray(g.origin)-g.world_size/2
    try:
        solution=costmap_path.solve(g.cost,control.descent_allowed,xy,lower,g.cell_size,
                                   density=control.descent_model)
    except ValueError:
        control.waiting_for_map=True
        control.planning_failure='insufficient observed space around the current body and route'
        return [],False
    goalcell=g.world_to_cell(*goal);route=[];complete=False
    if goalcell is not None and control.checkpoint_supported[goalcell]:
        route,_=costmap_path.path_to_point(solution,g.cost,control.descent_allowed,goal)
        complete=bool(route)
    if not route:
        route=local_step(control,solution,xy,goal,yaw)
        if route:control.observation_target=route[-1]
    if not route:
        control.waiting_for_map=True
        control.planning_failure='no short goalward step with measured body clearance and gentle reverse corridor'
        return [],False
    score=route_score(control,route)
    reserve=float(control.checkpoint_reserve[g.world_to_cell(*route[-1])])
    control.route_options=[dict(profile='exponential slope / incremental observed approach',selected=True,
        complete=complete,score=score,checkpoint_data_reserve_m=reserve,
        path_data_reserve_m=PATH_RESERVE_M,**g.route_profile(route))]
    control.route_searches=[dict(profile='exponential slope',connected=complete,has_route=True,uphill_limit=None)]
    return route,complete


def backout_observed(grid, xy, yaw):
    """Leave existing overlap; do not reverse into newly unknown/obstacle cells."""
    bad=~measured_coverage(grid) | grid.hazard
    initial=grid._support_cells(xy)
    if initial is None:return False
    inherited=set(zip(initial[0][bad[initial]],initial[1][bad[initial]]))
    for distance in np.linspace(0,.6,13):
        point=np.asarray(xy)-distance*np.array([math.cos(yaw),math.sin(yaw)])
        cells=grid._support_cells(point)
        if cells is None:return False
        occupied=set(zip(cells[0][bad[cells]],cells[1][bad[cells]]))
        if not occupied.issubset(inherited):return False
    return not occupied
