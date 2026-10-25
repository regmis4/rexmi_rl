"""An observer must preserve the original terminal decision and its command."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import copy
import json
from pathlib import Path
import tempfile
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from test_behavior_navigation import b,m,ground,observation
from rexmi_bt_tests.nav.departure_diagnostics import DepartureCapture,MAP_FIELDS,load_departure,write_departure
from rexmi_bt_tests.nav import departure_diagnostics


class DepartureDiagnosticTests(unittest.TestCase):
    def controller(self):
        c=b.BehaviorTreeAscentController(ground(),[SimpleNamespace(x=13.,y=0.,label='exterior')],mode='characterize')
        c.obs=observation(5.,xy=(.6,.6),velocity=(.08,0.))
        e=c.executor;e.start(m.MovementRequest('departure',(0.,0.),(3.,0.),1,.4,30.),0.)
        e.committed_heading=0.;e.travel_axis=(1.,0.)
        e._state(m.MovementState.DRIVE_FORWARD,1.,'fixture committed drive')
        e.uphill_samples=[(3.,(.3,.2)),(4.,(.5,.4))]
        c.passive_heading.samples=[(4.7,0.),(5.,0.)]
        return c

    def tick(self,c):
        return c.executor.tick(c.obs,lambda command:'',lambda xy:True)

    def test_observer_preserves_original_result_state_and_returned_command(self):
        baseline=self.controller();expected=self.tick(baseline)
        c=self.controller();d=DepartureCapture(c,'/unused/run.csv',{'seed':42})
        c.executor.departure_observer=d.observe
        actual=self.tick(c);d.complete_tick(actual)
        self.assertEqual(actual,expected)
        self.assertEqual(c.executor.result,baseline.executor.result)
        self.assertEqual(c.executor.state,baseline.executor.state)
        self.assertEqual(d.metadata['executor']['state'],'DRIVE_FORWARD')
        self.assertEqual(d.metadata['returned_state'],'IDLE')
        self.assertEqual(d.metadata['returned_result']['reason'],'measured lateral departure from straight corridor')
        self.assertFalse(d.metadata['output_applied_to_physics'])
        self.assertEqual(d.metadata['request']['endpoint'],[3.,0.])
        self.assertEqual(d.metadata['geometry']['lateral_m'],.6)

    def test_copied_map_freshness_and_exclusions_cannot_change_after_capture(self):
        c=self.controller();cell=c.grid.world_to_cell(0.,0.)
        c.grid.fresh[cell]=False;c.grid.last_seen[cell]=5.
        d=DepartureCapture(c,'/unused/run.csv',{});c.executor.departure_observer=d.observe
        self.tick(c)
        for name in MAP_FIELDS:
            self.assertFalse(np.shares_memory(d.arrays[name],getattr(c.grid,name)),name)
        self.assertFalse(d.arrays['fresh'][cell])
        self.assertFalse(d.arrays['approach_blocked'].any())
        c.grid.mark_failure('later',(0.,0.),0.,6.,'post-outcome mutation')
        c.grid.fresh[:]=True;c.executor.uphill_samples.clear()
        self.assertFalse(d.arrays['fresh'][cell])
        self.assertFalse(d.arrays['approach_blocked'].any())
        self.assertEqual(len(d.metadata['executor']['uphill_samples']),2)

    def test_once_only_and_no_capture_before_lateral_rejection(self):
        c=self.controller();c.obs=observation(5.,xy=(.6,.2))
        d=DepartureCapture(c,'/unused/run.csv',{});c.executor.departure_observer=d.observe
        self.tick(c);self.assertFalse(d.attempted)
        c.obs=observation(5.02,xy=(.6,.6));self.tick(c)
        original=copy.deepcopy(d.metadata)
        d.observe(observation(8.),None,{})
        self.assertEqual(d.metadata,original)

    def test_skipped_mission_then_matched_recovery_retains_prior_failure_and_command(self):
        baseline=self.controller();c=self.controller()
        d=DepartureCapture(c,'/unused/run.csv',{},purpose='recovery')
        c.executor.departure_observer=d.observe
        self.assertEqual(self.tick(c),self.tick(baseline))
        self.assertEqual(c.executor.result,baseline.executor.result)
        self.assertFalse(d.attempted);self.assertIsNone(d.arrays);self.assertIsNone(d.error)
        for controller in (c,baseline):
            controller.grid.mark_failure('mission-failure',(.6,.6),0.,5.,'physical mission departure')
            controller.executor.consume()
            controller.executor.start(m.MovementRequest('recovery',(0.,0.),(3.,0.),
                1,.4,30.,purpose='recovery'),5.01)
            controller.executor.committed_heading=0.;controller.executor.travel_axis=(1.,0.)
            controller.executor._state(m.MovementState.DRIVE_FORWARD,5.02,'recovery fixture')
            controller.obs=observation(5.1,xy=(.6,.6))
            controller.recovering=True;controller.recovery_since=5.
            controller.recovery_actions=1;controller.recovery_direction=1
            controller.recovery_trials={(1,2,3)}
        actual=self.tick(c);self.assertEqual(actual,self.tick(baseline))
        d.complete_tick(actual)
        self.assertEqual(c.executor.result,baseline.executor.result)
        self.assertTrue(d.attempted);self.assertEqual(d.metadata['capture_purpose'],'recovery')
        self.assertEqual(d.metadata['request']['purpose'],'recovery')
        self.assertTrue(d.arrays['approach_blocked'].any())
        self.assertEqual(d.metadata['recovery']['recovery_since'],5.)
        self.assertEqual(d.metadata['recovery']['recovery_actions'],1)
        c.recovery_trials.add((4,5,6))
        self.assertEqual(d.metadata['recovery']['recovery_trials'],[[1,2,3]])
        saved=copy.deepcopy(d.metadata);d.observe(observation(9.),None,{})
        self.assertEqual(d.metadata,saved)

    def test_nonmatching_purpose_never_copies_or_submits(self):
        c=self.controller();d=DepartureCapture(c,'/unused/run.csv',{},purpose='recovery')
        c.executor.departure_observer=d.observe
        with patch.object(departure_diagnostics,'plain',side_effect=AssertionError('no capture')):
            self.tick(c)
        worker=SimpleNamespace(submit=lambda *args:self.fail('no job without a match'))
        d.submit(worker);self.assertFalse(d.attempted)
        self.assertIsNone(d.metadata);self.assertIsNone(d.arrays);self.assertIsNone(d.error)

    def test_default_any_and_mission_filter_capture_and_copy_executor_state(self):
        for purpose in ('any','mission'):
            c=self.controller();d=DepartureCapture(c,'/unused/run.csv',{},purpose=purpose)
            c.executor.departure_observer=d.observe
            c.executor.recovery_stage_samples=[(4.,[.1,.2])]
            self.tick(c)
            self.assertTrue(d.attempted)
            self.assertEqual(d.metadata['capture_purpose'],purpose)
            c.executor.recovery_stage_samples[0][1].append(.3)
            self.assertEqual(d.metadata['executor']['recovery_stage_samples'],[[4.,[.1,.2]]])
            self.assertIn('speed_response_used',d.metadata['executor'])
        self.assertEqual(DepartureCapture(self.controller(),'/unused/run.csv',{}).purpose,'any')
        with self.assertRaises(ValueError):
            DepartureCapture(self.controller(),'/unused/run.csv',{},purpose='unknown')

    def test_capture_and_callback_failures_cannot_change_command_or_result(self):
        baseline=self.controller();expected=self.tick(baseline)
        for stage in ('capture','callback'):
            c=self.controller();d=DepartureCapture(c,'/unused/run.csv',{})
            if stage=='capture':
                c.executor.departure_observer=d.observe
            else:
                def broken(*args):raise RuntimeError('observer failed')
                c.executor.departure_observer=broken
            with patch.object(departure_diagnostics,'MAP_FIELDS',('missing_observer_field',)):
                self.assertEqual(self.tick(c),expected)
            self.assertEqual(c.executor.result,baseline.executor.result)
            if stage=='capture':self.assertIn('AttributeError',d.error)

    def test_background_roundtrip_preserves_exact_arrays_without_rebuild(self):
        c=self.controller()
        with tempfile.TemporaryDirectory() as folder:
            d=DepartureCapture(c,Path(folder)/'run.csv',{'seed':42})
            c.executor.departure_observer=d.observe;cmd=self.tick(c);d.complete_tick(cmd)
            with ThreadPoolExecutor(max_workers=1) as worker:
                d.submit(worker);self.assertTrue(d.job.result()['ok'])
            with patch.object(type(c.grid),'rebuild',side_effect=AssertionError('must not recompute')):
                loaded,meta=load_departure(d.path)
            for name in MAP_FIELDS:np.testing.assert_equal(getattr(loaded,name),d.arrays[name])
            self.assertEqual(meta,d.metadata)
            self.assertFalse(meta['counts_as_mission']);self.assertFalse(meta['counts_as_qualification'])

    def test_writer_and_submission_errors_are_diagnostic_results(self):
        c=self.controller();d=DepartureCapture(c,'/unused/run.csv',{})
        c.executor.departure_observer=d.observe;cmd=self.tick(c);d.complete_tick(cmd)
        with patch.object(Path,'mkdir',side_effect=OSError('no space')):
            result=write_departure(d.path,d.arrays,d.metadata)
        self.assertFalse(result['ok']);self.assertIn('no space',result['error'])
        worker=SimpleNamespace(submit=lambda *args:(_ for _ in ()).throw(RuntimeError('worker unavailable')))
        d.submit(worker);self.assertIn('worker unavailable',d.error)
        self.assertEqual(c.executor.result.reason,'measured lateral departure from straight corridor')

    def test_report_never_counts_a_diagnostic_as_a_mission(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run=root/'diagnostic';run.mkdir()
            (run/'run.summary.json').write_text(json.dumps(dict(result='complete',controller='ascent',
                diagnostic=True,counts_as_mission=False,counts_as_qualification=False)))
            script=Path(__file__).resolve().parents[1]/'scripts/report_ascent_trials.py'
            subprocess.run([sys.executable,str(script),'--root',str(root),'--output',str(root/'report.json')],
                           check=True,capture_output=True,text=True)
            report=json.loads((root/'report.json').read_text())
            self.assertEqual(report['completed_ascent_trials'],0)
            self.assertEqual(report['completed_ascent_missions'],0)
            self.assertEqual(report['trials'][0]['purpose'],'diagnostic')


if __name__=='__main__':unittest.main()
