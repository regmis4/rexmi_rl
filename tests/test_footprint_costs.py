"""Terrain soft-cost invariants; no simulator or traction assumptions."""
import importlib
import math
import numpy as np
import pytest
import test_behavior_navigation as bt

fc=importlib.import_module('rexmi_bt_tests.nav.footprint_costs')


def terrain(surface):
    g=bt.a.AscentTerrainMap(world_size=8.,cell_size=.2,footprint_radius=.5,
                          half_length=.47,half_width=.27)
    centres=-4.+(np.arange(40)+.5)*.2
    axis=(np.arange(4)+.5)*.05-.1
    bx,by=(v.ravel() for v in np.meshgrid(axis,axis,indexing='ij'))
    x=centres[:,None,None]+bx
    y=centres[None,:,None]+by
    z=np.broadcast_to(surface(x,y),(40,40,16)).astype(np.float32).copy()
    g._bin_low[:]=z;g._bin_high[:]=z;g._bin_count[:]=1
    g.height[:]=z.max(axis=-1);g.low[:]=z.min(axis=-1);g.samples[:]=16
    g.rebuild(0.)
    return g


@pytest.mark.parametrize('gradient',[(0.,0.),(math.tan(math.radians(33)),0.),(.4,.4)])
def test_plane_grade_is_not_roughness_or_step(gradient):
    a,b=gradient
    g=terrain(lambda x,y:3.+a*x+b*y)
    g.footprint_costs_enabled=True;fc.rebuild(g)
    core=np.s_[8:-8,8:-8]
    assert np.nanmax(g.footprint_rms[core])<1e-5
    assert np.nanmax(g.footprint_step[core])<1e-5
    assert np.all(g.footprint_unknown_cost[core]==0)


def test_small_bumps_and_steps_cost_more_than_smooth_plane_without_deadband():
    smooth=terrain(lambda x,y:.3*x+.1*y)
    bumps=terrain(lambda x,y:.3*x+.1*y+.025*np.sin(16*x)*np.cos(13*y))
    step=terrain(lambda x,y:.3*x+.1*y+.12*(x>0))
    for g in (smooth,bumps,step):
        g.footprint_costs_enabled=True;fc.rebuild(g)
    cell=(20,20)
    assert 0<bumps.footprint_rms[cell]<.06
    assert bumps.footprint_rough_cost[cell]>.1
    assert step.footprint_step[cell]>.04
    assert step.footprint_step_cost[cell]>smooth.footprint_step_cost[cell]+1.


def test_missing_samples_are_explicit_unknown_and_do_not_look_flat():
    g=terrain(lambda x,y:.2*x+.1*y)
    g._bin_count[15:25,15:25]=0
    g.footprint_costs_enabled=True;fc.rebuild(g)
    assert np.isnan(g.footprint_rms[20,20]) and np.isnan(g.footprint_step[20,20])
    assert g.footprint_unknown_cost[20,20]==8
    assert g.footprint_coverage[20,20]==0


def test_raw_spread_preserved_and_no_physical_fields_or_masks_changed():
    g=terrain(lambda x,y:.2*x+.1*y)
    g._bin_high[20,20,0]+=.16
    names=('height','low','surface','slope','gx','gy','known','fresh','hazard','blocked','no_go')
    before={name:getattr(g,name).copy() for name in names}
    g.footprint_costs_enabled=True;fc.rebuild(g)
    for name in names:np.testing.assert_array_equal(getattr(g,name),before[name])
    assert g.footprint_spread[20,20]>=.159


def test_supported_plane_without_supported_jump_pairs_is_not_zero_step():
    g=terrain(lambda x,y:.12*(x>0))
    g._bin_count[:]=0
    g._bin_count[17:24,17:24]=1
    for row,col in ((19,19),(19,20),(20,19),(21,20),(21,21)):
        g._bin_count[row,col]=0
    g.footprint_costs_enabled=True;fc.rebuild(g)
    assert g.footprint_coverage[20,20]==pytest.approx(44/49)
    assert np.isfinite(g.footprint_rms[20,20])
    assert g.footprint_step_coverage[20,20]<.8
    assert np.isnan(g.footprint_step[20,20])
    assert g.footprint_rough_cost[20,20]==0
    assert g.footprint_step_cost[20,20]==0
    assert g.footprint_unknown_cost[20,20]==8


def test_application_is_idempotent_and_rebuild_does_not_accumulate_cost():
    g=terrain(lambda x,y:.02*np.sin(16*x)*np.cos(13*y))
    g.footprint_costs_enabled=True;fc.rebuild(g)
    expected=g.cost.copy();features=g.footprint_rms
    fc.rebuild(g)
    assert g.footprint_rms is features
    np.testing.assert_array_equal(g.cost,expected)
    g.rebuild(1.)
    np.testing.assert_allclose(g.cost,expected,atol=1e-6)


def test_scalar_vector_cost_and_route_breakdown_are_consistent():
    g=terrain(lambda x,y:.02*np.sin(16*x)*np.cos(13*y))
    g.footprint_costs_enabled=True;fc.rebuild(g)
    g.uphill_limit_deg=33.;g.side_limit_deg=25.
    offsets=((1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1))
    costs,_=g.directional_costs(offsets=offsets)
    for i,(dx,dy) in enumerate(offsets):
        assert g.edge_cost((20,20),dx,dy,i)==pytest.approx(costs[20,20,i])
    path=[(-1.,0.),(1.,0.)]
    score=g.route_cost(path)
    components=fc.route_components(g,path,score)
    assert components['total']==pytest.approx(sum(components[k] for k in ('base','roughness','discontinuity','unknown')))
    subdivided=fc.route_components(g,[(-1.,0.),(0.,0.),(1.,0.)],score)
    for k in ('roughness','discontinuity','unknown'):
        assert components[k]==pytest.approx(subdivided[k])
    assert components['supported_fraction']==1.
