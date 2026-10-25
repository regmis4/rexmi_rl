import numpy as np
from unittest.mock import patch
import test_behavior_navigation as bt
from rexmi_bt_tests.nav import corridor_waypoints as cw


def ground():
    g=bt.ground();g.motion_turn_limit=15.;g.uphill_limit_deg=33.;g.side_limit_deg=32.
    return g


def test_whole_segment_must_stay_in_corridor_not_only_endpoints():
    g=ground();roi=np.ones_like(g.known);roi[g.world_to_cell(1.,0.)]=False
    assert not cw.inside(g,roi,(0.,0.),(2.,0.))
    assert cw.inside(g,roi,(0.,1.),(2.,1.))


def test_small_candidate_sets_keep_nearby_stages_without_body_spacing_filter():
    g=ground();path=np.array([[0.,0.],[.2,0.],[.4,0.]])
    points,roi,stats=cw.candidates(g,path,20.)
    assert (.2,0.) in points
    assert len(points)<=cw.MAX_NODES and stats['omitted_cells']==0


def test_complete_straight_flat_route_and_actual_turn_rejection():
    g=ground();path=[(0.,0.),(2.,0.)]
    with patch.object(cw,'candidates',return_value=(path,np.ones_like(g.known),{})):
        result,stats=cw.solve(g,path,0.,20.,(0.,2.))
        assert result==path and stats['complete'] and not stats['budget_exhausted']
        with patch.object(cw,'turn_reason',return_value='forbidden requested turn'):
            result,stats=cw.solve(g,path,1.,20.,(0.,2.))
        assert not result and stats['turn_rejections']['forbidden requested turn']


def test_ceiling_and_geometry_cannot_be_bypassed_by_shortcut():
    g=ground();path=[(0.,0.),(2.,0.)];g.cost[g.world_to_cell(1.,0.)]=19.
    with patch.object(cw,'candidates',return_value=(path,np.ones_like(g.known),{})):
        result,stats=cw.solve(g,path,0.,10.,(0.,2.))
    assert not result and stats['edge_rejections']['exceeds fixed color ceiling']==2


def test_search_budget_is_reported_not_called_impossible():
    g=ground();path=[(0.,0.),(2.,0.)]
    with patch.object(cw,'candidates',return_value=(path,np.ones_like(g.known),{})):
        result,stats=cw.solve(g,path,0.,20.,(0.,2.),wall_seconds=0.)
    assert not result and stats['budget_exhausted']


def test_exact_subcell_endpoints_do_not_duplicate_intermediate_cells():
    g=ground();path=np.array([[.03,.01],[.12,.01],[2.03,.01]])
    points,_,stats=cw.candidates(g,path,20.)
    assert points[:2]==[tuple(path[0]),tuple(path[-1])]
    cells=[g.world_to_cell(*p) for p in points]
    assert len(cells)==len(set(cells))
    assert stats['omitted_cells']==0
