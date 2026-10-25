"""Display contracts using CPU telemetry and the accepted planning artifact."""
import json
import math
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from queue import SimpleQueue

import numpy as np
import pytest
import test_remote_control
from rexmi_bt_tests.nav.dashboard_telemetry import TravelHistory, drive_activity
from rexmi_bt_tests.nav import terrain_display, planning_snapshot, slope_cost
from rexmi_bt_tests.nav.dashboard import Dashboard

ROOT = Path(__file__).resolve().parents[1]


def accepted_model():
    result = json.loads((ROOT/'outputs/remote_autonomy/slope-fit-20261024-04/result.json').read_text())
    path = next(path for path in result['inputs'] if path.endswith('.npz'))
    grid, _ = planning_snapshot.load(path)
    return slope_cost.Model(grid, slope_cost.Parameters(**result['parameters'])), result


def test_live_colors_equal_accepted_figure_and_do_not_mutate_grid():
    from matplotlib import colormaps
    from matplotlib.colors import LogNorm
    model, _ = accepted_model()
    g, p = model.grid, model.parameters
    original = g.cost.copy()
    penalty = p.uphill_weight*np.expm1(np.maximum(model.terrain_angle-p.easy_angle_deg, 0.)/p.exponential_scale_deg)
    density = g.cost+(1+p.terrain_weight)*penalty*(1+p.roughness_interaction*model.roughness)
    vmax = max(20., float(np.nanpercentile(density[g.known & ~g.blocked], 99)))
    expected = colormaps['magma'](LogNorm(2., vmax)(density))[..., :3]
    expected[g.blocked] = (.3, .31, .33)
    expected[~g.known] = (.12, .13, .15)
    view = terrain_display.from_model(model)
    np.testing.assert_allclose(view['rgb'], expected, atol=1e-12)
    np.testing.assert_array_equal(g.cost, original)
    assert view['vmax'] == vmax


def test_distance_includes_reverse_without_negative_progress_and_resets_on_clock_reset():
    h = TravelHistory()
    h.update(0, (0, 0, 1), 0)
    h.update(1, (3, 4, 2), 10)
    h.update(1, (30, 40, 20), 0)  # duplicate frame is not extra motion
    h.update(2, (0, 0, 1.5), -10)
    np.testing.assert_allclose(h.snapshot(), [[0, 0, 0], [5, 1, 10], [10, .5, -10]])
    h.update(0, (20, 20, 5), 3)
    np.testing.assert_allclose(h.snapshot(), [[0, 0, 3]])


def test_invalid_telemetry_does_not_bridge_missing_position_or_invent_slope():
    h = TravelHistory()
    h.update(0, (0, 0, 1), 0)
    h.update(1, (np.nan, 0, 1), 0)
    h.update(2, (100, 0, 1), np.nan)
    assert h.distance == 0
    assert np.isnan(h.snapshot()[-1, 2])


def test_profile_is_bounded_without_truncating_start_or_latest():
    h = TravelHistory(limit=10)
    for n in range(100): h.update(n, (n, 0, n/10), n/10)
    rows = h.snapshot()
    assert len(rows) <= 11
    assert rows[0, 0] == 0 and rows[-1, 0] == 99
    assert (np.diff(rows[:, 0]) > 0).all()


@pytest.mark.parametrize('cmd,speed,yaw,state,expected', [
    ((.4, 0, 0), .3, 0, 'DRIVE', {'FORWARD'}),
    ((.4, 0, .35), .3, .2, 'DRIVE', {'FORWARD', 'LEFT'}),
    ((0, 0, -.35), .02, -.3, 'TURN', {'RIGHT'}),
    ((-.4, 0, 0), .3, 0, 'RECOVER', {'REVERSE'}),
    ((0, 0, 0), .2, 0, 'STOP', {'BRAKING'}),
    ((0, 0, 0), 0, .1, 'STOP', {'BRAKING'}),
    ((0, 0, 0), 0, 0, 'STOP', {'STOP'}),
    ((0, 0, 0), 0, np.nan, 'STOP', {'ZERO COMMAND'}),
])
def test_drive_intent_is_separate_from_measured_response(cmd, speed, yaw, state, expected):
    assert drive_activity(cmd, speed, yaw, state) == expected


def test_heading_grade_uses_terrain_not_body_pitch():
    grid = SimpleNamespace(world_to_cell=lambda *xy: (0, 0), known=np.array([[True]]),
                           gx=np.array([[math.tan(math.radians(20))]]), gy=np.zeros((1, 1)))
    assert terrain_display.heading_grade(grid, (0, 0), 0) == pytest.approx(20)
    assert terrain_display.heading_grade(grid, (0, 0), math.pi) == pytest.approx(-20)
    assert terrain_display.heading_grade(grid, (0, 0), math.pi/2) == pytest.approx(0)


def test_descent_display_does_not_color_extrapolated_coverage_as_observed():
    model,_=accepted_model()
    observed=model.grid.known.copy()
    cell=tuple(np.argwhere(observed & ~model.grid.blocked)[0])
    observed[cell]=False
    view=terrain_display.from_model(model,observed_mask=observed)
    assert not view['known'][cell]
    np.testing.assert_array_equal(view['rgb'][cell],terrain_display.UNKNOWN)
    assert model.grid.known[cell]


def test_dashboard_redraw_and_only_map_can_issue_target(tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    model, result = accepted_model()
    g = model.grid
    shared = dict(pose=SimpleNamespace(x=0., y=0., z=1., yaw=0.), cmd=(.4, 0, 0),
                  planned_path=result['routes']['any_exit'], terrain_display=terrain_display.from_model(model),
                  speed=.3, actual_yaw_rate=0., manual_checkpoint_mode=True,
                  travel_profile=np.array([[0, 0, 0], [1, .1, 10]]))
    requests = SimpleQueue()
    view = Dashboard(shared, Lock(), g, [], request_queue=requests)
    view._create_figure(plt)
    for _ in range(2): view._redraw(plt)
    assert view._ax_global is None
    assert view._ax_local.get_position().x0 < view._ax_profile.get_position().x0
    for ax in (view._ax_profile, view._ax_slope, view._ax_drive):
        view._on_map_click(SimpleNamespace(button=1, inaxes=ax, xdata=1, ydata=2))
    assert requests.empty()
    view._on_map_click(SimpleNamespace(button=1, inaxes=view._ax_local, xdata=1, ydata=2))
    assert requests.get() == ('target', 1., 2.)
    view._fig.savefig(tmp_path/'dashboard.png')
    plt.close(view._fig)


def test_overlay_receives_frozen_colors_and_heights_instead_of_live_old_palette():
    from unittest.mock import MagicMock, patch
    import test_perception_view as pv
    model, _ = accepted_model()
    display = terrain_display.from_model(model)
    g = model.grid
    view = pv.v.PerceptionView.__new__(pv.v.PerceptionView)
    view.nav = SimpleNamespace(_omap=g, _lidar=None, _lock=Lock(),
        shared={'planned_path': [], 'terrain_display': display},
        _sim_localizer=SimpleNamespace(get_pose=lambda: SimpleNamespace(x=0., y=0., z=1.)))
    view.point_limit=2000; view._tile_key=None; view._tile_count=0
    view.layer='terrain'; view.status=None; view.selected_path_only=False
    view._Usd=MagicMock(); view._Sdf=MagicMock(); view.stage=MagicMock()
    view.show=dict(lidar=True, costs=True, path=True, rays=False)
    for name in ('points','tiles','route','mission_curve','approaches','rays','checkpoint_marker',
                 'target_marker','_set_curve','_update_tiles'):
        setattr(view,name,MagicMock())
    with patch.object(pv.v, 'survey_tiles', wraps=pv.v.survey_tiles) as tiles:
        view._update()
    assert tiles.call_args.args[0] is display['surface']
    assert tiles.call_args.args[2] is display['rgb']
    assert tiles.call_args.kwargs['continuous'] is True


def test_live_colorbar_reuses_locator_through_long_run_and_layer_switches():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    model,_=accepted_model()
    dashboard=Dashboard({},Lock(),model.grid,[])
    dashboard._create_figure(plt)
    view=dict(vmin=2.,vmax=20.,label='Live fitted slope cost')
    dashboard._draw_legend(view)
    bar=dashboard._colorbar
    locator=dashboard._ax_colorbar.get_axes_locator()
    axes_count=len(dashboard._fig.axes)
    try:
        for frame in range(1100):
            view['vmax']=20.+frame
            dashboard._layer='slope' if frame%11==0 else 'terrain'
            dashboard._draw_legend(view)
            assert dashboard._colorbar is bar
            assert dashboard._ax_colorbar.get_axes_locator() is locator
            if frame%250==0:dashboard._fig.canvas.draw()
        dashboard._layer='terrain';dashboard._draw_legend(view)
        dashboard._fig.canvas.draw()
        assert len(dashboard._fig.axes)==axes_count
        assert dashboard._colorbar.norm.vmax==1119.
    finally:plt.close(dashboard._fig)
