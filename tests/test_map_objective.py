"""Terrain-only route objective: full costs, legal steep pinches, actual exit."""
import copy
import importlib
import math
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import test_behavior_navigation as bt

sr = importlib.import_module('rexmi_bt_tests.nav.staged_routes')
routes = importlib.import_module('rexmi_bt_tests.nav.ascent_routes')
ac = importlib.import_module('rexmi_bt_tests.nav.ascent_control')


@pytest.fixture
def grid():
    g = bt.ground()
    g.motion_turn_limit = 15.
    g.uphill_limit_deg = 33.
    g.side_limit_deg = 25.
    return g


def steep_patch(g, width, across):
    x = (np.arange(g.n_cells)+.5)*g.cell_size-g.world_size/2
    mask = (np.abs(x[:, None]) < width) & (np.abs(x[None, :]) < across)
    g.gx[mask] = math.tan(math.radians(28.))
    g.slope[mask] = g.gx[mask]
    return mask


def test_clear_steep_shortcut_competes_with_longer_gentle_route(grid):
    steep_patch(grid, .8, .6)
    start, goal = (-3., 0.), (3., 0.)
    assert not sr.observed_edge_reason(grid, start, goal, start)
    direct = sr.terminal_travel_cost(grid, start, goal, 'gentle')
    path, complete = routes.observed_route(grid, start, goal, 0., profile='gentle',
        uphill_limit=33., allow_frontier=False, max_expansions=10000)
    assert complete and path[-1] == goal
    assert len(path) > 2
    assert grid.last_route_search['terminal_total_cost'] < direct
    assert grid.route_profile(path)['uphill_over_25deg_m'] < grid.route_profile([start, goal])['uphill_over_25deg_m']
    assert not sr.motion_route_reason(grid, path, 0.)


def test_gentle_preference_allows_necessary_legal_steep_pinch(grid):
    steep_patch(grid, .2, 6.)
    path, complete = sr.staged_route(grid, (-2., 0.), (2., 0.), 0., offsets=((1,0),),
        profile='gentle', uphill_limit=33., allow_frontier=False, max_expansions=3000)
    assert complete and grid.route_profile(path)['max_uphill_deg'] > 20.
    assert not sr.staged_route(grid, (-2., 0.), (2., 0.), 0., offsets=((1,0),),
        profile='gentle', uphill_limit=20., allow_frontier=False, max_expansions=3000)[1]


@pytest.mark.parametrize('profile', ['direct', 'balanced', 'gentle'])
@pytest.mark.parametrize('offset', [(1, 0), (2, 1), (0, 1)])
def test_terminal_one_edge_matches_vectorized_profile_cost(grid, profile, offset):
    grid.gx[:] = .2
    grid.gy[:] = .1
    grid.diagonal_ascent_preference = True
    first = grid.world_to_cell(0., 0.)
    last = (first[0]+offset[0], first[1]+offset[1])
    costs, _ = grid.directional_costs(profile, 33., [offset])
    price = sr.terminal_travel_cost(grid, grid.cell_to_world(*first), grid.cell_to_world(*last), profile)
    assert price == pytest.approx(costs[last][0])


def test_budget_retains_only_fully_checked_complete_exit_and_reports_exhaustion(grid):
    path, complete = routes.observed_route(grid, (-2., 0.), (2., 0.), 0.,
        allow_frontier=False, max_expansions=1)
    assert complete and path == [(-2., 0.), (2., 0.)]
    assert grid.last_route_search['expansions'] == 1
    assert grid.last_route_search['budget_exhausted']
    assert 'at search budget' in grid.last_route_search['goal_selection']
    grid.known[grid.world_to_cell(0., 0.)] = False
    assert not routes.observed_route(grid, (-2., 0.), (2., 0.), 0.,
        allow_frontier=False, max_expansions=1)[1]


def test_incomplete_direct_search_cannot_redirect_contour_and_shelf_to_frontier(grid):
    ctrl = ac.AscentController(grid, [SimpleNamespace(x=3., y=0.)], mode='characterize', uphill_limit=33.)
    calls = []
    def search(g, start, goal, *args, **kwargs):
        calls.append((start, goal, kwargs))
        g.last_route_search = {}
        if kwargs.get('profile') == 'direct':
            return [start, (1., 0.)], False
        return [start, goal], True
    with patch.object(ac, 'observed_route', side_effect=search), patch.object(
        grid, 'shelf_candidates', return_value=[(1.5, 1.)]):
        assert ctrl._compute_plan((0., 0.), 0., 0.)
    assert calls[1][1] == (3., 0.)
    assert calls[1][2]['profile'] == 'gentle'
    assert calls[1][2]['uphill_limit'] == 32. # Existing 1-degree reserve outside remote.
    assert calls[3][0] == (1.5, 1.) and calls[3][1] == (3., 0.)
