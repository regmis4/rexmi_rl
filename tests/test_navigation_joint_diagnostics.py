"""Physics/control alignment and exception cleanup for optional nav telemetry."""
import importlib.util
import json
from pathlib import Path
import unittest
from types import SimpleNamespace

import test_joint_telemetry as telemetry

spec=importlib.util.spec_from_file_location('nav_joint_diagnostics',
    Path(__file__).resolve().parents[1]/'source/rexmi_rl/nav/joint_diagnostics.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class NavigationJointDiagnosticsTests(unittest.TestCase):
    setUp=telemetry.TelemetryTests.setUp
    rows=telemetry.TelemetryTests.rows
    def logger(self):
        logger=module.NavigationJointTelemetry(self.path,self.env,metadata={})
        self.addCleanup(logger.close)
        return logger

    def test_raw_actions_are_detached_and_joinable_to_physics(self):
        logger=self.logger();actions=[[.5,-2.]]
        logger.begin_navigation_step(31,.62,'rocky_slope','RELEASE','move:2',0.,0.,actions)
        actions[0][0]=99.
        logger.submitted()
        for _ in range(4):self.env.scene.update(dt=.005)
        logger.end_navigation_step(done=False)
        logger.close()
        event=next(json.loads(line) for line in (self.path/'events.jsonl').read_text().splitlines()
                   if json.loads(line)['event']=='policy_action')
        self.assertEqual(event['raw_policy_output_before_wrapper_clipping'],[[.5,-2.]])
        self.assertEqual(event['control_step'],31);self.assertEqual(event['nav_time_s'],.62)
        self.assertEqual({r['control_step'] for r in self.rows()},{'31'})
        self.assertEqual(logger.report()['completed_control_steps'],1)
        self.assertEqual(logger.report()['logging_timing_ms']['physics_sample_logging']['count'],4)

    def test_partial_application_is_never_called_unapplied(self):
        logger=self.logger()
        logger.begin_navigation_step(0,0.,'rocky_slope','DRIVE_FORWARD','move:2',.4,0.,[[0.]])
        logger.submitted();self.env.scene.update(dt=.005);logger.close()
        event=next(json.loads(line) for line in (self.path/'events.jsonl').read_text().splitlines()
                   if json.loads(line)['event']=='partial_navigation_step')
        self.assertEqual(event['physics_application'],'unknown_partial')
        self.assertEqual(event['captured_substeps'],1)
        self.assertFalse(logger.active);self.assertEqual(self.env.scene.update,logger._original_update)

    def test_never_submitted_is_explicit_and_close_is_idempotent(self):
        logger=self.logger()
        logger.begin_navigation_step(0,0.,'rocky_slope','BRAKE','move:1',0.,0.,[[0.]])
        logger.close();logger.close()
        events=[json.loads(line) for line in (self.path/'events.jsonl').read_text().splitlines()]
        partial=[e for e in events if e['event']=='partial_navigation_step']
        self.assertEqual(len(partial),1);self.assertEqual(partial[0]['physics_application'],'not_submitted')

    def test_scene_hook_restored_even_when_partial_event_write_fails(self):
        logger=self.logger()
        logger.begin_navigation_step(0,0.,'rocky_slope','BRAKE','move:1',0.,0.,[[0.]])
        event=logger.event
        def fail(kind,**details):
            if kind=='partial_navigation_step':raise OSError('disk error')
            event(kind,**details)
        logger.event=fail
        with self.assertRaises(OSError):logger.close()
        self.assertEqual(self.env.scene.update,logger._original_update)

    def test_all_event_writes_failing_still_close_both_streams(self):
        logger=self.logger()
        def fail(*args,**kwargs):raise OSError('all writes fail')
        logger.event=fail
        with self.assertRaisesRegex(OSError,'all writes fail'):logger.close()
        self.assertTrue(logger.stream.closed);self.assertTrue(logger.events.closed)
        self.assertTrue(logger.closed);logger.close()
        self.assertEqual(self.env.scene.update,logger._original_update)
        self.assertIn('all writes fail',logger.report()['final_error'])

    def test_completed_physics_survives_post_step_logger_exception(self):
        runner_spec=importlib.util.spec_from_file_location('diagnostic_runner',
            Path(__file__).resolve().parents[1]/'source/rexmi_rl/nav/checkpoint_runner.py')
        runner=importlib.util.module_from_spec(runner_spec);runner_spec.loader.exec_module(runner)
        logger=self.logger()
        logger.begin_navigation_step(0,0.,'rocky_slope','BRAKE','move:1',0.,0.,[[0.]])
        logger.submitted()
        calls=[]
        snapshots=SimpleNamespace(flush=lambda *a,**kw:calls.append(kw))
        # A completed env.step with missing hook samples makes end_step raise.
        env=SimpleNamespace(reset_terminated=[False],reset_time_outs=[False])
        with self.assertRaisesRegex(RuntimeError,'Telemetry saw'):
            runner.completed_physics_diagnostics(snapshots,None,logger,[False],env)
        self.assertTrue(calls[0]['completed'])
        self.assertTrue(logger.partial_step['completed'])


if __name__=='__main__':unittest.main()
