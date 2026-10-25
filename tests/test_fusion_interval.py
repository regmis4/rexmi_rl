import json
import numpy as np
import pytest
import test_behavior_navigation as bt
from rexmi_bt_tests.nav import terrain_prior as tp


def interval(low,high):
    g=bt.a.AscentTerrainMap(world_size=12.)
    g.ingest([[.11,.11,low],[.11,.11,high]],(0,0,3),0.)
    g.rebuild(0.)
    hit=g._bin_count>0;g._bin_count[hit]=20
    return g,hit


@pytest.mark.parametrize('bounds,measurement', [((0.,1.),1.04),((1.,2.),.96)])
def test_independent_endpoint_confirmation_keeps_enclosing_interval(bounds,measurement):
    g,hit=interval(*bounds)
    for t in (1.,2.,3.):
        g.ingest([[.11,.11,measurement]],(0,0,3),t);g.rebuild(t)
        assert np.all(g._bin_low[hit]<=g._bin_high[hit])
    assert g._bin_low[hit][0]<=measurement+1e-6<=g._bin_high[hit][0]+2e-6
    assert 0.<float((g._bin_high-g._bin_low)[hit][0])<.04
    assert (g._raise_count[hit]==0).all() and (g._clear_count[hit]==0).all()


def test_obstacle_expands_immediately_clears_only_after_three_returns():
    g,hit=interval(0.,0.)
    g.ingest([[.11,.11,.5]],(0,0,3),1.);g.rebuild(1.)
    assert g._bin_high[hit][0]==.5
    before=g._bin_high.copy();g.rebuild(2.)
    np.testing.assert_array_equal(g._bin_high,before)
    for t in (3.,4.):
        g.ingest([[.11,.11,0.]],(0,0,3),t);g.rebuild(t)
        assert g._bin_high[hit][0]==.5
    g.ingest([[.11,.11,0.]],(0,0,3),5.);g.rebuild(5.)
    assert g._bin_low[hit][0]==g._bin_high[hit][0]==0.


def test_ordinary_non_crossing_update_preserves_ewma():
    g,hit=interval(0.,1.)
    g.ingest([[.11,.11,.04],[.11,.11,1.04]],(0,0,3),1.);g.rebuild(1.)
    assert g._bin_low[hit][0]==pytest.approx(.04/21.)
    assert g._bin_high[hit][0]==pytest.approx(1.+.04/21.)


def test_post_crossing_map_serializes_and_loads_without_repair(tmp_path):
    g,hit=interval(0.,1.)
    for t in (1.,2.,3.):
        g.ingest([[.11,.11,1.04]],(0,0,3),t);g.rebuild(t)
    # Roughness is intentionally unknown for one subcell. Add sufficient
    # neighboring samples without rewriting the crossed interval under test.
    x,y=np.meshgrid(np.arange(-.975,1.,.05),np.arange(-.975,1.,.05))
    keep=~((x>=.1)&(x<.15)&(y>=.1)&(y<.15))
    g.ingest(np.c_[x[keep],y[keep],np.full(keep.sum(),1.02)],(0,0,3),4.);g.rebuild(4.)
    g.terrain_prior=dict(version=1,coordinate_frame='world',task='test',seed=42,
        kind='retained_sensor_survey',geometry_id=tp.geometry_id(g))
    path=tmp_path/'prior.npz';np.savez_compressed(path,**tp.prior_snapshot(g))
    loaded=bt.a.AscentTerrainMap(world_size=12.)
    tp.load_prior(loaded,path,task='test',seed=42)
    for name in ('_bin_low','_bin_high','_bin_count'):
        np.testing.assert_array_equal(getattr(g,name),getattr(loaded,name))
    supported=loaded.known & (loaded.roughness_support>=6)
    assert supported.any() and np.isfinite(loaded.roughness[supported]).all()
    assert (loaded.roughness[supported]>=0.).all()
