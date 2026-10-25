"""The remote route has one cost contract, independent of generator labels."""
import math
import numpy as np
import pytest
import test_behavior_navigation as bt
from rexmi_bt_tests.nav import traversal_objective as obj
from rexmi_bt_tests.nav.staged_routes import terminal_travel_cost, staged_route


@pytest.fixture
def grid():
    g = bt.ground()
    g.traversal_objective = obj.NAME
    g.motion_turn_limit = 15.
    g.uphill_limit_deg, g.side_limit_deg = 33., 32.
    g.gx[:] = .18
    g.gy[:] = .07
    g.slope[:] = np.hypot(g.gx, g.gy)
    return g


@pytest.mark.parametrize('offset', [(1, 0), (2, 1), (-1, 2)])
def test_scalar_vector_terminal_and_route_use_identical_density(grid, offset):
    a = grid.cell_to_world(*grid.world_to_cell(0., 0.))
    b = tuple(np.asarray(a)+np.asarray(offset)*grid.cell_size)
    cell = grid.world_to_cell(*b)
    yaw = math.atan2(offset[1], offset[0])
    expected = math.hypot(*offset)*float(obj.density(grid, cell, *offset))
    for profile in ('direct', 'balanced', 'gentle'):
        costs, _ = grid.directional_costs(profile, offsets=[offset])
        assert costs[cell][0] == pytest.approx(expected)
        assert grid.edge_cost(cell, *offset, 0, profile) == pytest.approx(expected)
        assert terminal_travel_cost(grid, a, b, profile) == pytest.approx(expected)
    assert grid.route_cost([a, b], yaw) == pytest.approx(expected)


def test_continuous_turn_charged_once_and_wraps(grid):
    a, b, c = (.1, .1), (1.1, .1), (1.1, 1.1)
    expected = obj.segment_cost(grid, a, b)+obj.segment_cost(grid, b, c)
    expected += obj.turn_cost(grid, b, 0., math.pi/2)
    assert grid.route_cost([a, b, c], 2*math.pi) == pytest.approx(expected)
    split = [a, (.5, .1), b, (1.1, .5), c]
    assert grid.route_cost(split, 0.) == pytest.approx(expected)
    assert sum(obj.route_cost(grid, split, breakdown=True).values()) == pytest.approx(expected)


def test_heterogeneous_collinear_compaction_preserves_exact_cell_integral(grid):
    rows, cols=np.indices(grid.known.shape)
    grid.cost[:]=1.+(rows%5)+.3*(cols%3)
    a,b=(-1.93,-.84),(2.37,1.31)
    middle=tuple(np.asarray(a)+.371*np.subtract(b,a))
    whole=obj.segment_cost(grid,a,b)
    assert obj.segment_cost(grid,a,middle)+obj.segment_cost(grid,middle,b)==pytest.approx(whole,abs=1e-10)


def test_uphill_priority_escape_retains_finite_checked_scalar_incumbent(grid):
    grid.uphill_first=True
    grid.mark_failure('escape-fixture',(0.,0.),0.,0.,'physical stall')
    # Original occupied-patch departure is checked by the spatial escape gate.
    path,done=staged_route(grid,(0.,0.),(-2.,0.),math.pi,offsets=((-1,0),),
        no_go_exit=True,require_fresh_start=False,allow_frontier=False)
    assert done and path[-1]==(-2.,0.)
    assert grid.last_route_search['objective']=='scalar checked-exclusion escape'
    assert math.isfinite(grid.last_route_search['terminal_total_cost'])
    assert grid.last_route_search['terminal_uphill_burden'] is None


def test_search_price_equals_final_route_price(grid):
    a, b = (-2., 0.), (2., 1.)
    path, done = staged_route(grid, a, b, 0., offsets=((1,0),(2,1),(1,1),(0,1)),
                             require_fresh_start=False, allow_frontier=False)
    assert done
    assert grid.last_route_search['terminal_total_cost'] == pytest.approx(grid.route_cost(path, 0.))


def test_uphill_first_search_reports_same_pair_as_selected_route(grid):
    grid.uphill_first = True
    grid.gx[:] = math.tan(math.radians(20.))
    grid.slope[:] = grid.gx
    # Straight travel requires no steep turn permission.
    path, done = staged_route(grid, (-2., 0.), (2., 0.), 0., offsets=((1,0),),
        require_fresh_start=False, allow_frontier=False)
    assert done
    parts=obj.route_cost(grid,path,0.,breakdown=True)
    assert grid.last_route_search['terminal_total_cost'] == pytest.approx(sum(parts.values()))
    assert grid.last_route_search['terminal_uphill_burden'] == pytest.approx(parts['uphill'])


def test_uphill_first_preserves_budget_incumbent_and_hard_limits(grid):
    grid.uphill_first = True
    path, done = staged_route(grid, (-2., 0.), (2., 1.), 0., offsets=((1,0),),
        require_fresh_start=False, allow_frontier=False, max_expansions=1)
    assert done and path[-1] == (2.,1.)
    assert grid.last_route_search['budget_exhausted']
    grid.approach_blocked[:] = True
    assert not staged_route(grid,(-2.,0.),(2.,1.),0.,offsets=((1,0),),
        require_fresh_start=False,allow_frontier=False,max_expansions=1)[1]


@pytest.mark.parametrize('kind', ['uphill', 'cross', 'failure', 'obstacle', 'unknown'])
def test_cost_cleanup_does_not_override_rejections(grid, kind):
    if kind == 'uphill': grid.gx[:] = math.tan(math.radians(34.))
    if kind == 'cross': grid.gy[:] = math.tan(math.radians(33.))
    if kind == 'failure': grid.approach_blocked[:] = True
    if kind == 'obstacle': grid.hazard[:] = grid.blocked[:] = True
    if kind == 'unknown': grid.known[:] = False
    assert math.isinf(grid.route_cost([(-1., 0.), (1., 0.)]))
