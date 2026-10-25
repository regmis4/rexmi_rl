"""Preference-model invariants and directed route-cost integration."""
import math
from types import SimpleNamespace
import numpy as np
import pytest
import test_remote_control
from rexmi_bt_tests.nav import slope_cost as sc,costmap_path as cp


def grid(angle=20.):
    shape=(6,6);ones=np.ones(shape)
    return SimpleNamespace(cost=ones*3,gx=ones*math.tan(math.radians(angle)),gy=ones*0,
        surface=np.indices(shape)[0]*math.tan(math.radians(angle)),
        slope=ones*math.tan(math.radians(angle)),footprint_rms=ones*.02,footprint_step=ones*.05,
        origin=np.array([0.,0.]),world_size=6.,cell_size=1.)


def test_easy_grades_uncharged_and_difficulty_grows_exponentially():
    p=sc.Parameters()
    assert np.all(sc.uphill_penalty(np.array([-20,0,5,10,15]),p)==0)
    values=sc.uphill_penalty(np.arange(16.,34.),p)
    assert np.all(np.diff(values)>0) and np.all(np.diff(values,n=2)>0)
    assert sc.uphill_penalty(30.,p)>5*sc.uphill_penalty(20.,p)


def test_steep_ground_remains_expensive_when_moving_sideways():
    p=sc.Parameters(terrain_weight=1.)
    m=sc.Model(grid(30.),p);parts=m.components((2,2),0.,1.)
    assert parts['uphill']==0 and parts['terrain']>0 and parts['rough_slope']>0
    assert m((2,2),0.,1.)>sc.Model(grid(10.),p)((2,2),0.,1.)


def test_descent_uses_same_exponential_penalty_for_both_grade_signs():
    p=sc.Parameters();g=grid(30.)
    ascent=sc.Model(g,p);descent=sc.Model(g,p,bidirectional=True)
    assert descent((2,2),-1.,0.)==pytest.approx(ascent((2,2),1.,0.))
    assert descent((2,2),-1.,0.)>ascent((2,2),-1.,0.)
    assert descent((2,2),1.,0.)==pytest.approx(descent((2,2),-1.,0.))


def test_wheelbase_gradient_reveals_face_hidden_by_display_plane():
    g=grid(10.);g.cell_size=.2;g.world_size=1.2
    g.surface[:]=0.;g.surface[3:]=.4
    m=sc.Model(g,sc.Parameters())
    assert m.terrain_angle[2,2]>30.
    assert m.components((2,2),1.,0.)['uphill']>0.


def test_roughness_multiplies_steep_cost_and_unknown_is_not_smooth():
    p=sc.Parameters();g=grid();m=sc.Model(g,p);before=m((2,2),1.,0.)
    g.footprint_rms[:]*=2;g.footprint_step[:]*=2
    assert sc.Model(g,p)((2,2),1.,0.)>before
    g.footprint_rms[:]=np.nan;g.footprint_step[:]=np.nan
    assert np.isfinite(sc.Model(g,p)((2,2),1.,0.))
    assert sc.Model(g,p)((2,2),1.,0.)>before


def test_directional_end_connector_uses_travel_toward_goal():
    cost=np.ones((6,1));allowed=np.ones_like(cost,bool)
    def density(cells,dx,dy):return cost[cells]*(2. if dx>0 else 1.)
    solution=cp.solve(cost,allowed,(.5,.5),np.zeros(2),1.,density=density)
    path,value=cp.path_to_point(solution,cost,allowed,(4.8,.5))
    assert value==pytest.approx(8.6)
    reverse=cp.solve(cost,allowed,(4.8,.5),np.zeros(2),1.,density=density)
    _,value=cp.path_to_point(reverse,cost,allowed,(.5,.5))
    assert value==pytest.approx(4.3)


def test_model_exact_integral_matches_planner_and_is_subdivision_invariant():
    g=grid();m=sc.Model(g,sc.Parameters());allowed=np.ones((6,6),bool)
    lower=np.array([-3.,-3.]);a=(-2.4,-2.3);b=(1.8,1.1)
    solution=cp.solve(g.cost,allowed,a,lower,1.,density=m)
    path,value=cp.path_to_point(solution,g.cost,allowed,b)
    assert value==pytest.approx(sum(m.route_components(path).values()))
    middle=tuple(.4*np.asarray(a)+.6*np.asarray(b))
    assert m.route_components([a,b])==pytest.approx(m.route_components([a,middle,b]))


@pytest.mark.parametrize('values',[dict(exponential_scale_deg=0),dict(terrain_weight=-1),dict(uphill_weight=math.nan)])
def test_invalid_parameters_rejected(values):
    with pytest.raises(ValueError):sc.Parameters(**values)
