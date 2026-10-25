"""Analytic checks for the report's displayed-cost integral."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('manual_analysis', Path(__file__).resolve().parents[1]/'scripts/analyze_manual_ascent.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_exact_crossing_stationary_points_and_subdivision():
    cost = np.array([[2.], [10.], [4.]])
    def measure(points):
        return analysis.path_cost_profile(points, cost, np.ones_like(cost, dtype=bool), [0, 0], 1)
    direct = measure([[.25, .5], [2.75, .5]])
    divided = measure([[.25, .5], [.25, .5], [.9, .5], [1.7, .5], [2.75, .5]])
    assert direct.cumulative_cost.iloc[-1] == pytest.approx(.75*2+10+.75*4)
    assert divided.cumulative_cost.iloc[-1] == pytest.approx(direct.cumulative_cost.iloc[-1])
    assert direct.end_m.iloc[-1] == pytest.approx(2.5)


def test_diagonal_and_reverse_count_actual_travel():
    cost = np.full((3, 3), 5.)
    result = analysis.path_cost_profile([[.5, .5], [2.5, 2.5], [.5, .5]], cost, np.ones_like(cost, dtype=bool), [0, 0], 1)
    assert result.cumulative_cost.iloc[-1] == pytest.approx(5*4*np.sqrt(2))


def test_unknown_is_not_silently_free():
    with pytest.raises(ValueError, match='unsupported'):
        analysis.path_cost_profile([[.5, .5], [1.5, .5]], np.ones((2, 1)), np.array([[True], [False]]), [0, 0], 1)


def test_sampling_retains_final_endpoint():
    import pandas as pd
    frame = pd.DataFrame({'x': range(7)})
    assert analysis.sample_with_end(frame, 5).x.tolist() == [0, 5, 6]
