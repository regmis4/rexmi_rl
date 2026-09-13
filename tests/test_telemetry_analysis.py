"""Numerical screening invariants; no GPU/Isaac imports."""
import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'analysis'))
from analyze_joint_telemetry import longest_duration, ratio_screen, metrics


class AnalysisTests(unittest.TestCase):
    def test_duration_breaks_at_reset(self):
        self.assertAlmostEqual(longest_duration([1,1,1,1], [.1]*4, [0,0,1,1]), .2)
        self.assertAlmostEqual(longest_duration([1,0,1,1], [.1]*4, [0]*4), .2)

    def test_ratio_tradeoff_and_no_double_internal_gearing(self):
        x = ratio_screen(15.8, 20.05, 2.5)
        self.assertAlmostEqual(x['module_rms_screen_Nm'], 15.8/2.25)
        self.assertAlmostEqual(x['joint_nominal_screen_Nm'], 16.875)
        self.assertTrue(x['rms_endpoint_screen_pass'])
        self.assertFalse(x['speed_endpoint_screen_pass'])
        self.assertEqual(ratio_screen(15.8,20.05,1)['assumed_efficiency'], 1)

    def test_metrics_preserve_peak_pairing_and_sign(self):
        f = pd.DataFrame({'j.torque_applied_est_Nm':[3.,-4.], 'j.velocity_rad_s':[10.,-2.],
                          'j.torque_demand_est_Nm':[3.,-5.], 'j.torque_clipped_est':[0,1],
                          'dt_s':[.005,.005], 'episode':[0,0], 'sim_time_s':[1.,1.005], 'scenario':['a','b']})
        s = metrics(f,'j')
        self.assertAlmostEqual(s['rms_est_Nm'], np.sqrt(12.5))
        self.assertEqual(s['speed_at_first_torque_peak_rad_s'], -2)
        self.assertEqual(s['peak_first_scenario'],'b')
        self.assertEqual(s['clipped_time_pct'],50)
        self.assertIsNone(s['max_10s_rms_est_Nm'])


if __name__ == '__main__':
    unittest.main()
