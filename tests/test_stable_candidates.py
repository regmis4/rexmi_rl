"""Survey inventory retention never grants live terrain permission."""
import copy
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
import test_behavior_navigation as bt
from rexmi_bt_tests.nav import stage_graph as sg, planning_snapshot as snap


def ground():
    g=bt.ground();g.motion_turn_limit=15.
    return g


def test_source_is_independent_readonly_and_caches_are_private():
    g=ground();sg.minimum_fields(g)
    source=sg.freeze_candidate_source(g)
    for name in sg.SOURCE_FIELDS:
        assert not np.shares_memory(getattr(source,name),getattr(g,name))
        assert not getattr(source,name).flags.writeable
    for name in sg.SOURCE_CACHES:assert name not in source.__dict__
    before=source.slope.copy();g.slope[:]=1.
    np.testing.assert_array_equal(source.slope,before)


def inventory_fixture():
    g=ground();source=sg.freeze_candidate_source(g)
    start,goal=(0.,0.),(5.,0.)
    prior=[start,goal,(2.,0.)];live=[start,goal,(2.1,0.),(3.5,0.)]
    def proposals(grid,*args):return prior if grid is source else live
    return g,source,start,goal,proposals


def test_valid_old_location_survives_nearby_ranking_change_and_live_fill():
    g,source,start,goal,proposals=inventory_fixture()
    with patch.object(sg,'nodes',side_effect=proposals):
        points,counts=sg.stable_nodes(g,start,goal,source)
        assert points==[start,goal,(2.,0.),(3.5,0.)]
        assert counts['survey_retained']==1 and counts['live_covered']==1
        fallback,_=sg.stable_nodes(g,start,goal,None)
        assert fallback==proposals(g)


@pytest.mark.parametrize('changed',['blocked','unknown','slope','height','turn'])
def test_current_invalid_node_is_not_retained(changed):
    g,source,start,goal,proposals=inventory_fixture();cell=g.world_to_cell(2.,0.)
    if changed=='blocked':g.blocked[cell]=True;g.hazard[cell]=True
    if changed=='unknown':g.known[cell]=False
    if changed=='slope':g.slope[cell]=1.
    if changed=='height':g.surface[cell]=3.
    with patch.object(sg,'nodes',side_effect=proposals),patch.object(sg,'stage_reason',side_effect=lambda grid,p:'unsafe turn' if changed=='turn' and p==(2.,0.) else ''):
        points,counts=sg.stable_nodes(g,start,goal,source)
    assert (2.,0.) not in points and counts['survey_rejected_current']==1


def test_capacity_is_fixed_and_omissions_are_reported():
    g,source,start,goal,proposals=inventory_fixture()
    with patch.object(sg,'nodes',side_effect=proposals),patch.object(sg,'MAX_NODES',3):
        points,counts=sg.stable_nodes(g,start,goal,source)
    assert len(points)==3 and counts['live_omitted_capacity']==1


def test_directional_failure_still_removes_live_edge():
    g,source,start,goal,proposals=inventory_fixture()
    g.mark_failure('failed',(2.,0.),0.,3.,'physical failure')
    with patch.object(sg,'nodes',side_effect=proposals):
        graph=sg.build(g,start,goal,lambda a,b:'',candidate_source=source)
    assert (0,1) not in graph['edges']
    assert not any(i==0 and j!=0 and graph['points'][j][0]>=2. for i,j in graph['edges'])


def test_snapshot_restores_source_inventory_and_provenance(tmp_path):
    g=ground();g.terrain_prior={'kind':'simulator_development','file_sha256':'fixture'}
    source=sg.freeze_candidate_source(g);g.slope[g.world_to_cell(1.,0.)]=.02
    control=SimpleNamespace(grid=g,candidate_source=source,planning_snapshot_dir=tmp_path,
        wp_idx=0,waypoints=[SimpleNamespace(x=4.,y=0.)],plan_generation=1)
    record=snap.save(control,(0.,0.),0.,3.);loaded,meta=snap.load(record['path'])
    restored=loaded.planning_candidate_source
    assert meta['candidate_source_config']['terrain_prior']==g.terrain_prior
    for name in sg.SOURCE_FIELDS:
        np.testing.assert_array_equal(getattr(source,name),getattr(restored,name))
        assert not getattr(restored,name).flags.writeable
    assert sg.stable_nodes(loaded,(0.,0.),(4.,0.),restored)==sg.stable_nodes(g,(0.,0.),(4.,0.),source)


def test_capped_live_fill_prefers_spatial_coverage_not_enumeration():
    g=ground();source=sg.freeze_candidate_source(g)
    start,goal=(0.,0.),(5.,0.)
    with patch.object(sg,'nodes',side_effect=lambda grid,*args:[start,goal]+([] if grid is source else [(1.2,0.),(2.5,0.)])),patch.object(sg,'MAX_NODES',3):
        points,counts=sg.stable_nodes(g,start,goal,source)
    assert points==[start,goal,(2.5,0.)]
    assert counts['live_omitted_capacity']==1
