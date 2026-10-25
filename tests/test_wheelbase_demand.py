import math
from unittest.mock import patch
import numpy as np
import pytest
from test_climb_burden import terrain, graph, reference
from rexmi_bt_tests.nav import wheelbase_demand as wd, climb_burden as cb, ascent_first as af


def test_nominal_geometry_and_flat_ramp_fourth_power_scaling_reversal():
    assert wd.WHEELBASE_M == pytest.approx(.4505454148531053)
    for slope in (0., .1, .2, .4):
        g, _, _ = terrain(lambda x, y: slope*x, cost=3.)
        field = cb.height_field(g)
        assert wd.metric(g, field, (0., 0.), (2., 0.))['burden'] == pytest.approx(6*slope**4)
        assert wd.metric(g, field, (2., 0.), (0., 0.))['burden'] == pytest.approx(0., abs=1e-20)


def test_same_gain_over_shorter_distance_costs_more():
    values = []
    for length in (1., 2., 4.):
        g, _, _ = terrain(lambda x, y: x/length)
        values.append(wd.metric(g, cb.height_field(g), (0., 0.), (length, 0.))['burden'])
    assert values == pytest.approx([2., .25, .03125])


def test_doubling_window_rise_has_sixteenfold_penalty():
    values = []
    for slope in (.15, .3):
        g, _, _ = terrain(lambda x, y: slope*x)
        values.append(wd.metric(g, cb.height_field(g), (0., 0.), (2., 0.))['burden'])
    assert values[1]/values[0] == pytest.approx(16.)


def test_collinear_partition_invariance_with_cost_and_shifted_grid_crossings():
    g, x, y = terrain(lambda x, y: x*y)
    g.cost[:] = np.where(x < .4, 2., 9.)
    field = x*y
    a, b = np.array([-.7, 1.2]), np.array([1.4, -.4])
    whole = wd.metric(g, field, a, b)['burden']
    for t in (.137, .5, .831):
        m = a+t*(b-a)
        parts = wd.metric(g, field, a, m)['burden']+wd.metric(g, field, m, b)['burden']
        assert parts == pytest.approx(whole, rel=1e-11, abs=1e-12)


def test_short_hump_has_window_cancellation_but_nonzero_approach_burden():
    g, x, y = terrain()
    # Narrow triangular height peak: front/rear equal at the centre.
    field = np.maximum(0., .3-np.abs(x))
    first = (np.array([0., 0.])-g.origin+g.world_size/2)/g.cell_size
    offset = np.array([wd.WHEELBASE_M/2/g.cell_size, 0.])
    assert cb.height_at(field, first+offset) == pytest.approx(cb.height_at(field, first-offset))
    forward = wd.metric(g, field, (-1., 0.), (1., 0.))['burden']
    reverse = wd.metric(g, field, (1., 0.), (-1., 0.))['burden']
    assert forward > 0 and reverse == pytest.approx(forward)


def test_extended_support_is_required_even_beyond_endpoints():
    g, x, y = terrain()
    field = x.copy()
    field[x > .2] = np.nan
    a, b = (-.2, 0.), (0., 0.)
    # Old metric can score the short leg; front window reaches unsupported cells.
    cb.metric(g, field, a, b)
    with pytest.raises(ValueError, match='extended wheelbase'):
        wd.metric(g, field, a, b)


def test_exact_integration_matches_independent_dense_quadrature():
    g, x, y = terrain()
    g.cost[:] = np.where(x < .4, 2., 7.)
    field = np.sin(2*x)+.3*x*y
    a, b = np.array([-.7, .6]), np.array([1.2, -.4])
    n = 20000
    points = a+(np.arange(n)[:, None]+.5)/n*(b-a)
    shift = (b-a)/np.linalg.norm(b-a)*wd.WHEELBASE_M/2
    vals = []
    for p in points:
        front = cb.height_at(field, (p+shift-g.origin+g.world_size/2)/g.cell_size)
        rear = cb.height_at(field, (p-shift-g.origin+g.world_size/2)/g.cell_size)
        vals.append(g.cost[g.world_to_cell(*p)]*max(0., (front-rear)/wd.WHEELBASE_M)**4)
    reference_value = np.mean(vals)*np.linalg.norm(b-a)
    assert wd.metric(g, field, a, b)['burden'] == pytest.approx(reference_value, rel=2e-4)


def test_shared_search_matches_reference_and_reports_new_objective():
    g, _, _ = terrain()
    data = graph(4, [(0, 2, 1), (2, 1, 10), (0, 3, 5), (3, 1, 5)])
    with patch.object(af, 'turn_reason', return_value=''):
        path, stats = wd.search(g, data, 0.)
    assert stats['rank'] == pytest.approx(reference(data))
    assert stats['objective'] == wd.NAME and stats['wheelbase_m'] == wd.WHEELBASE_M


def test_route_diagnostics_sum_exact_metric_and_keep_old_burden():
    g, _, _ = terrain(lambda x, y: .3*x)
    path = [(0., 0.), (.37, 0.), (2., 0.)]
    result = wd.describe_route(g, path)
    assert result['rank'] == pytest.approx([.0324, 2.])
    assert result['old_climb_burden'] == pytest.approx(1.2)
    assert 'bands' not in result and 'old_height_bands' in result
    assert 'burden' not in result['height_profile'][0]
    assert result['peak_wheelbase_dz_m'] == pytest.approx(.3*wd.WHEELBASE_M)
    assert sum(r['burden'] for r in result['sections']) == pytest.approx(.0324)
