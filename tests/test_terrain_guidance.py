"""Queue guidance is approximate; executable checks remain authoritative."""
import importlib
from unittest.mock import patch
import numpy as np
import pytest
import test_behavior_navigation as bt
from test_footprint_costs import terrain

sr=importlib.import_module('rexmi_bt_tests.nav.staged_routes')


def test_reverse_graph_charges_actual_destination_direction():
    allowed=np.zeros((3,3),bool);allowed[0,:]=True
    costs=np.ones((3,3,2));costs[0,:,0]=[100.,3.,4.];costs[0,:,1]=[7.,8.,900.]
    d,stats=sr.terrain_guidance(allowed,costs,((0,1),(0,-1)),(0,2))
    np.testing.assert_allclose(d[0],[7.,4.,0.])
    assert np.isinf(d[1:]).all()
    assert stats['nodes']<=stats['node_budget']==9
    back,_=sr.terrain_guidance(allowed,costs,((0,1),(0,-1)),(0,0))
    np.testing.assert_allclose(back[0],[0.,7.,15.])


def test_nonfinite_edge_direction_and_disallowed_cells_remain_unreachable():
    allowed=np.ones((3,3),bool)
    costs=np.ones((3,3,1));costs[0,1,0]=np.inf
    d,_=sr.terrain_guidance(allowed,costs,((0,1),),(0,2))
    assert np.isinf(d[0,0]) and np.isinf(d[0,1])
    assert d[0,2]==0


@pytest.mark.parametrize('unavailable',[True,False])
def test_unavailable_guidance_cannot_prune_continuous_terminal_shortcut(unavailable):
    g=terrain(lambda x,y:0.*x+0.*y)
    g.motion_turn_limit=15.;g.footprint_costs_enabled=True
    offsets=((1,0),(0,1),(-1,0),(0,-1))
    values=np.full(g.known.shape,np.inf if unavailable else 1000.)
    with patch.object(sr,'terrain_guidance',return_value=(values,{'kind':'test'})):
        path,done=sr.staged_route(g,(-1.,0.),(.13,.07),0.,offsets=offsets,require_fresh_start=False)
    assert done and path[-1]==(.13,.07)
    assert g.last_route_search['expansions']==1


def test_other_controllers_do_not_use_new_queue_guidance():
    g=terrain(lambda x,y:0.*x+0.*y);g.motion_turn_limit=15.
    with patch.object(sr,'terrain_guidance',side_effect=AssertionError('not enabled')):
        path,done=sr.staged_route(g,(-1.,0.),(1.,0.),0.,offsets=((1,0),),require_fresh_start=False)
    assert done
