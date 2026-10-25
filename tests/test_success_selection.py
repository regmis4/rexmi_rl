"""Probability ranking is independent of terrain preferences and physical admission."""
import copy
import importlib
import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import test_behavior_navigation

s = importlib.import_module('rexmi_bt_tests.nav.success_selection')
rr = importlib.import_module('rexmi_bt_tests.nav.remote_routes')


def controller():
    return SimpleNamespace(probability_domain_validated=True,
        model=SimpleNamespace(model_id='frozen-model'),
        waypoints=[SimpleNamespace(x=0., y=13.)], wp_idx=0,
        static_retention_enabled=True, grid=SimpleNamespace(origin=(0., 0.)))


def candidate(probability, cost, *, validated=True, complete=True, endpoint=(0., 13.)):
    path = [(0., 0.), (1., 3.), endpoint]
    estimate = SimpleNamespace(probability=probability, validated=validated)
    return ('normal', cost, path, complete, list(path), None, estimate, None, None, {})


def test_higher_whole_route_probability_beats_cheaper_terrain():
    c = controller()
    cheap, successful = candidate(.81, 10.), candidate(.94, 1000.)
    assert s.select(c, [cheap, successful], (0., 0.)) is successful
    assert successful[9]['success_selection']['objective'] == 'success_probability'
    assert successful[9]['success_selection']['comparable_validated_count'] == 2


def test_equal_probability_uses_cost_only_as_tie_breaker():
    expensive, cheap = candidate(.9, 100.), candidate(.9, 10.)
    assert s.select(controller(), [expensive, cheap], (0., 0.)) is cheap


@pytest.mark.parametrize('probability', [None, math.nan, math.inf, -.1, 1.1, True])
def test_unknown_invalid_probability_never_becomes_numeric_zero(probability):
    unknown, known = candidate(probability, 1.), candidate(.81, 50.)
    assert s.select(controller(), [unknown, known], (0., 0.)) is known
    assert unknown[9]['success_selection']['probability_used'] is None
    assert 'unranked' in unknown[9]['success_selection']['display']


@pytest.mark.parametrize('mismatch', ['unvalidated', 'different_goal', 'different_start', 'partial_leg', 'incomplete', 'ineligible'])
def test_only_same_complete_validated_leg_is_comparable(mismatch):
    bad, good = candidate(.99, 1.), candidate(.81, 100.)
    if mismatch == 'unvalidated':
        bad[6].validated = False
    elif mismatch == 'different_goal':
        bad = candidate(.99, 1., endpoint=(13., 0.))
    elif mismatch == 'different_start':
        bad[2][0] = (1., 0.)
        bad[4][0] = (1., 0.)
    elif mismatch == 'partial_leg':
        bad = bad[:4]+([bad[2][0], bad[2][1]],)+bad[5:]
    elif mismatch == 'incomplete':
        bad = candidate(.99, 1., complete=False)
    else:
        bad = ('reject',)+bad[1:]
    assert s.select(controller(), [bad, good], (0., 0.)) is good
    assert bad[9]['success_selection']['probability_used'] is None


def test_absent_current_domain_preserves_exact_existing_fallback_and_reports_it():
    c = controller()
    c.probability_domain_validated = False
    cheap, expensive = candidate(None, 5.), candidate(.99, 100.)
    result = s.select(c, [expensive, cheap], (0., 0.))
    assert result is cheap
    result[9]['selected'] = True
    assert 'fallback' in s.selection_text([result[9]])
    assert result[9]['success_selection']['comparable_validated_count'] == 0


def test_hard_physical_rejection_precedes_probability_ordering():
    c = controller()
    safe, unsafe = candidate(.81, 50.), candidate(.99, 1.)
    unsafe[2][1] = (5., 3.)
    with patch.object(rr, 'route_components', return_value={}), patch.object(
        rr, 'physical_profile', side_effect=lambda _, path: ('obstacle', math.inf) if path[1][0] == 5 else ('', 30.)):
        assert rr.select(c, [unsafe, safe], (0., 0.), 0.) is safe
    assert unsafe[9]['rejection'] == 'obstacle'


@pytest.mark.parametrize('changed', [None, 'policy_sha256', 'source_id', 'capability_profile', 'qualified', 'model_id'])
def test_model_domain_requires_exact_frozen_runtime_and_held_out_qualification(changed):
    model = SimpleNamespace(model_id='frozen', artifact=dict(evaluation=dict(qualified=True),
        policy_sha256='policy', source_id='source', capability_profile={'terrain': 35}))
    if changed in ('policy_sha256', 'source_id', 'capability_profile'):
        model.artifact[changed] = 'other'
    elif changed == 'qualified':
        model.artifact['evaluation']['qualified'] = False
    elif changed == 'model_id':
        model.model_id = None
    assert s.validated_domain(model, 'policy', 'source', {'terrain': 35}) is (changed is None)


def test_shared_overlay_shows_unknown_probability_separately_from_cost():
    view = importlib.import_module('rexmi_bt_tests.nav.perception_view')
    c = controller()
    row = candidate(None, 10.)
    s.select(c, [row], (0., 0.))
    row[9].update(selected=True, cost_components=dict(total=10., base=7., roughness=1.,
        discontinuity=2., unknown=0., supported_fraction=1.))
    text = view.route_cost_text([row[9]])
    assert 'probability unavailable' in text and 'fallback' in text
    assert 'Route cost 10.0' in text
