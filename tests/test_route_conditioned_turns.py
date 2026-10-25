"""A route valley may be a 2-D saddle; qualification must not rotate with yaw."""
import copy
import math
from unittest.mock import patch
import numpy as np
import pytest
import test_behavior_navigation as bt
import test_remote_control as remote
from rexmi_bt_tests.nav import turn_staging as ts


@pytest.fixture
def saddle():
    g = bt.ground()
    g.motion_turn_limit = 15.
    g.local_minimum_turns_enabled = g.route_conditioned_turns = True
    g.side_limit_deg = 32.
    g.slope[:] = math.tan(math.radians(20.))
    g.gx[:] = g.slope
    g.gy[:] = 0.
    axis = (np.arange(g.n_cells)+.5)*g.cell_size-g.world_size/2-.1
    x, y = np.meshgrid(axis, axis, indexing='ij')
    score = np.clip(20.+4.*y*y-8.*x, 0., 34.)
    return g, score


def test_saddle_admission_depends_on_route_not_all_neighbor_directions(saddle):
    g, score = saddle
    with patch.object(ts, 'minimum_fields', return_value=(np.zeros_like(g.known), score, np.zeros_like(score))):
        assert not ts.stage_reason(g, (.1, .1))
        assert not ts.turn_reason(g, (.1, .1), 0., math.pi/2)
        assert ts.turn_reason(g, (.1, .1), math.pi, math.pi/2)
        # Original eastward approach stays authoritative during rotation.
        assert not ts.turn_reason(g, (.1, .1), math.pi/3, math.pi/2,
                                 stage_headings=(0., math.pi/2))
        # Rotating from a rejected approach cannot create qualification.
        assert ts.turn_reason(g, (.1, .1), 0., math.pi/2,
                              stage_headings=(math.pi, math.pi/2))


@pytest.mark.parametrize('bad', ['uniform', 'unknown', 'insufficient_prominence'])
def test_bad_comparison_footprints_reject(saddle, bad):
    g, score = saddle
    if bad == 'uniform': score[:] = 20.
    if bad == 'insufficient_prominence': score[:] = 20.+.1*(score-20.)
    if bad == 'unknown': score[g.world_to_cell(.1-2*g.footprint_radius, .1)] = np.inf
    with patch.object(ts, 'minimum_fields', return_value=(np.zeros_like(g.known), score, np.zeros_like(score))):
        assert ts.turn_reason(g, (.1, .1), 0., math.pi/2)


def test_runtime_freezes_approach_through_settle_rotation_and_stop(saddle):
    g, score = saddle
    c = remote.controller(copy.deepcopy(g), (.1, 3.))
    c.obs = bt.observation(3., xy=(.1,.1), yaw=0.)
    c._begin_minimum_turn(math.pi/2)
    frozen = (c.minimum_turn['incoming'], c.minimum_turn['target'])
    deadline = c.minimum_turn['deadline']
    with patch.object(remote.r, 'turn_reason', return_value='') as check:
        c.obs = bt.observation(3.02, xy=(.1,.1), yaw=.6)
        c.observation_dt = .02
        c._minimum_turn_tick()
        assert check.call_args.kwargs['stage_headings'] == frozen
        assert check.call_args.args[2] == .6
    assert c.minimum_turn['deadline'] == deadline
    c.request('stop', 3.04)
    assert c.minimum_turn is None
