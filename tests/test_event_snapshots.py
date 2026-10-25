"""Passive event records must preserve the original motion and outcome."""
import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace,asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import test_departure_diagnostics as fixtures
from test_behavior_navigation import m,observation
from rexmi_bt_tests.nav.departure_diagnostics import EventSnapshots,MAP_FIELDS,load_departure


class EventSnapshotTests(unittest.TestCase):
    def controller(self):return fixtures.DepartureDiagnosticTests().controller()
    def round_trip_controller(self):
        c=self.controller()
        c.waypoints=[SimpleNamespace(x=0.,y=0.),SimpleNamespace(x=0.,y=13.)]
        c.configure_outbound_survey_routes()
        return c
    def attach(self,c,path='/unused/run.csv',**options):
        d=EventSnapshots(c,path,{},**options);c.executor.event_observer=d.observe
        c.executor.brake_evaluator=c._brake_evaluation
        return d
    def tick(self,c):return c.executor.tick(c.obs,lambda _: '',lambda _:True)

    def test_termination_keeps_command_result_and_pre_exclusion_observation(self):
        a=self.controller();expected=self.tick(a)
        c=self.controller();d=self.attach(c)
        c.executor._emit_traversal(m.Motion(.4,0.),replace(c.obs,time=4.98,xy=(.5,.6)))
        actual=self.tick(c);d.complete_tick(actual)
        self.assertEqual(actual,expected);self.assertEqual(c.executor.result.reason,a.executor.result.reason)
        cap=d.captures[0]
        self.assertEqual(cap.metadata['event_kind'],'termination')
        self.assertEqual(cap.metadata['executor']['executed_traversal']['last_position'],[.6,.6])
        self.assertIsNone(cap.metadata['executor']['pending_traversal'])
        self.assertEqual(cap.metadata['executor']['state'],'DRIVE_FORWARD')
        self.assertEqual(cap.metadata['returned_state'],'IDLE')
        self.assertFalse(cap.arrays['approach_blocked'].any())
        c.grid.mark_failure('later',c.obs.xy,0.,5.,'physical')
        self.assertFalse(cap.arrays['approach_blocked'].any())
        for name in MAP_FIELDS:self.assertFalse(np.shares_memory(cap.arrays[name],getattr(c.grid,name)))

    def test_actual_brake_rejection_captures_underlying_reason_once(self):
        c=self.controller();c.obs=replace(c.obs,xy=(0.,0.),velocity=(.3,0.));d=self.attach(c)
        with patch.object(c,'_stopping_reason',return_value='obstacle: fresh body sweep') as check:
            expected=c._brake_envelope(c.obs)
            self.assertFalse(d.captures) # admission probe is not emitted braking
            command=c.executor.brake(c.obs)
            self.assertEqual(check.call_count,2) # exactly one per decision
        self.assertEqual(command,m.Motion());self.assertEqual(expected,c.executor.brake_rejection)
        d.complete_tick(command)
        cap=d.captures[0]
        self.assertEqual(cap.metadata['event_details']['underlying_reason'],'obstacle: fresh body sweep')
        self.assertEqual(cap.metadata['returned_tick_brake_evaluation']['underlying_reason'],'obstacle: fresh body sweep')
        c.executor.brake_guard=lambda *_:'blocked'
        with patch.object(c,'_stopping_reason',return_value='another cell'):
            c.executor.brake(c.obs)
        self.assertEqual(len(d.captures),1);self.assertEqual(d.counts['duplicates'],1)

    def test_pure_command_and_staging_probes_never_capture(self):
        c=self.controller();c.obs=replace(c.obs,xy=(0.,0.),velocity=(.3,0.));d=self.attach(c)
        for _ in range(3):
            c._guard(m.Motion(.4,0.),record=False)
            c._brake_envelope(c.obs)
            c._recovery_staging_envelope(c.obs)
        self.assertFalse(d.captures);self.assertFalse(d.pending)

    def test_four_capture_cap_and_owned_byte_limit(self):
        c=self.controller();d=self.attach(c)
        for i in range(7):
            req=replace(c.executor.request,action_id=str(i))
            d.observe('termination',c.obs,req,dict(reason='fixture'))
        self.assertEqual(len(d.captures),4);self.assertEqual(d.counts['cap_dropped'],3)
        self.assertLessEqual(d.admitted_bytes,d.max_bytes)
        small=EventSnapshots(c,'/unused/small.csv',{},max_bytes=1)
        small.observe('termination',c.obs,c.executor.request,{})
        self.assertFalse(small.captures);self.assertEqual(small.counts['bytes_dropped'],1)

    def test_round_trip_reserves_pair_for_first_outbound_continuation(self):
        c=self.round_trip_controller();d=self.attach(c)
        # Descent can contain many outcomes without spending outbound slots.
        for i in range(8):
            d.observe('termination',c.obs,replace(c.executor.request,action_id=f'descent:{i}'),{})
        self.assertFalse(d.captures);self.assertEqual(d.copy_count,0)
        self.assertEqual(d.counts['stage_skipped'],8)
        c.wp_idx=1
        for i in range(5):
            d.observe('termination',c.obs,replace(c.executor.request,action_id=f'early:{i}'),{})
        self.assertEqual(len(d.captures),2)
        self.assertEqual(d.counts['reserved_skipped'],3)
        request=replace(c.executor.request,action_id='continued')
        c.executor.request=request
        c.executor._observe_event('ascent_continuation_admitted',c.obs,{'fixed_bound_m':3.08})
        # Later admissions and other outcomes cannot consume the paired slot.
        other=replace(request,action_id='other')
        d.observe('ascent_continuation_admitted',c.obs,other,{})
        d.observe('brake_rejection',c.obs,request,{'reason':'fixture'})
        d.observe('termination',c.obs,other,{})
        d.observe('termination',c.obs,request,{'outcome':'censored'})
        self.assertEqual(len(d.captures),4)
        self.assertEqual([x.metadata['capture_priority'] for x in d.captures],
            ['other_outbound_event','other_outbound_event',
             'first_outbound_continuation','first_outbound_continuation_termination'])
        self.assertEqual(d.captures[-1].metadata['request']['action_id'],'continued')
        self.assertEqual(d.max_captures,4);self.assertEqual(d.max_bytes,64*1024*1024)
        self.assertLessEqual(d.admitted_bytes,d.max_bytes)

    def test_outbound_capture_keeps_exact_map_and_does_not_spend_priority_on_error(self):
        c=self.round_trip_controller();c.wp_idx=1;d=self.attach(c)
        request=c.executor.request;cell=c.grid.world_to_cell(0.,0.)
        c.grid.fresh[cell]=False;c.grid.last_seen[cell]=1.23
        with patch('rexmi_bt_tests.nav.departure_diagnostics.DepartureCapture.observe',
                   side_effect=RuntimeError('copy')):
            d.observe('ascent_continuation_admitted',c.obs,request,{})
        self.assertIsNone(d.continuation_action);self.assertEqual(d.counts['errors'],1)
        request=replace(request,action_id='next')
        c.executor.uphill_goal_only_budget=True
        d.observe('ascent_continuation_admitted',c.obs,request,{})
        cap=d.captures[0]
        c.grid.fresh[cell]=True;c.grid.last_seen[cell]=99.
        self.assertFalse(cap.arrays['fresh'][cell])
        self.assertEqual(cap.arrays['last_seen'][cell],1.23)
        self.assertEqual(cap.metadata['waypoint_index'],1)
        self.assertTrue(cap.metadata['executor']['uphill_goal_only_budget'])
        self.assertEqual(cap.metadata['event_details'],{})

    def test_non_round_trip_keeps_old_event_selection(self):
        c=self.controller();d=self.attach(c)
        d.observe('ascent_continuation_admitted',c.obs,c.executor.request,{})
        self.assertFalse(d.captures);self.assertEqual(d.counts['ignored'],1)
        for i in range(5):
            d.observe('termination',c.obs,replace(c.executor.request,action_id=str(i)),{})
        self.assertEqual(len(d.captures),4)
        self.assertEqual(d.counts['cap_dropped'],1)
        self.assertFalse(any('capture_priority' in x.metadata for x in d.captures))

    def test_writer_receives_detached_completed_step_and_runner_stop_flags(self):
        for submitted,completed in [(False,False),(True,False),(True,True)]:
            with self.subTest(submitted=submitted,completed=completed),tempfile.TemporaryDirectory() as tmp,ThreadPoolExecutor(1) as worker:
                c=self.controller();d=self.attach(c,Path(tmp)/'run.csv');cmd=self.tick(c);d.complete_tick(cmd)
                if submitted:d.submitted()
                d.flush(worker,completed=completed,reason='fixture dispatch')
                self.assertIsNone(d.captures[0].arrays)
                report=d.report();self.assertTrue(report['captures'][0]['result']['ok'])
                _,meta=load_departure(d.captures[0].path)
                self.assertEqual(meta['output_returned'],True)
                self.assertEqual(meta['output_submitted_to_physics'],submitted)
                self.assertEqual(meta['output_step_completed'],completed)
                self.assertEqual(meta['output_applied_to_physics'],True if completed else None if submitted else False)
                self.assertFalse(meta['counts_as_mission'])

    def test_capture_and_submission_errors_do_not_change_command(self):
        baseline=self.controller();expected=self.tick(baseline)
        c=self.controller();d=self.attach(c)
        with patch('rexmi_bt_tests.nav.departure_diagnostics.DepartureCapture.observe',side_effect=RuntimeError('copy')):
            self.assertEqual(self.tick(c),expected)
        self.assertEqual(d.counts['errors'],1)
        self.assertEqual(c.executor.result,baseline.executor.result)
        c=self.controller();d=self.attach(c);cmd=self.tick(c)
        with patch.object(d.captures[0],'complete_tick',side_effect=RuntimeError('metadata')):
            d.complete_tick(cmd)
        worker=SimpleNamespace(submit=lambda *_:(_ for _ in ()).throw(RuntimeError('worker')))
        d.flush(worker);report=d.report()
        self.assertFalse(report['captures'][0]['result']['ok']);self.assertEqual(c.executor.result,baseline.executor.result)

    def test_executor_contains_arbitrary_observer_exception(self):
        baseline=self.controller();expected=self.tick(baseline)
        c=self.controller();c.executor.event_observer=lambda *_:(_ for _ in ()).throw(RuntimeError('observer'))
        self.assertEqual(self.tick(c),expected);self.assertEqual(c.executor.result,baseline.executor.result)

    def test_hard_fault_flushes_original_movement_once_without_recovery(self):
        c=self.controller();c.obs=replace(c.obs,up_z=.4);d=self.attach(c)
        with patch.object(c,'_begin_recovery',side_effect=AssertionError('no recovery')):
            c._hard_fault(None);c._hard_fault(None)
        events=[e for e in c.attempt_events if e['event']=='movement_outcome']
        self.assertEqual(len(events),1);self.assertTrue(events[0]['physical'])
        self.assertEqual(events[0]['reason'],'excessive body tilt')
        self.assertIsNone(c.executor.result);self.assertEqual(c.command,m.Motion())
        self.assertEqual(len(d.captures),1)

    def test_hard_fault_keeps_pending_result_identity(self):
        c=self.controller();self.tick(c);original=asdict(c.executor.result)
        c.obs=replace(c.obs,time=5.02,up_z=.4);c._hard_fault(None)
        event=next(e for e in c.attempt_events if e['event']=='movement_outcome')
        self.assertEqual({k:event[k] for k in original},original)

    def test_release_overspeed_entry_captures_before_transition_not_repeated_brake(self):
        c=self.controller();c.obs=replace(c.obs,xy=(.4,0.),velocity=(.51,0.),inclination=20.,terrain_gradient=(.4,0.))
        e=c.executor;e._state(m.MovementState.RELEASE,4.,'release fixture');e.release_origin=(.3,0.);d=self.attach(c)
        cmd=self.tick(c);d.complete_tick(cmd)
        cap=next(x for x in d.captures if x.metadata['event_kind']=='release_overspeed_brake')
        self.assertEqual(cap.metadata['executor']['state'],'RELEASE')
        self.assertEqual(cap.metadata['returned_state'],'BRAKE')
        self.assertGreater(cap.metadata['event_details']['gradient_velocity_mps'],0.)
        self.assertEqual(cap.metadata['returned_command']['vx'],-.4)
        c.obs=replace(c.obs,time=5.02);self.tick(c)
        self.assertEqual(sum(x.metadata['event_kind']=='release_overspeed_brake' for x in d.captures),1)

    def test_trial34_sequence_reserves_release_brake_and_terminal_under_cap(self):
        c=self.controller();d=self.attach(c)
        # Worst case: an unlogged first brake rejection on EACH of the two
        # mission actions. DRIVE speed transitions don't consume a capture.
        for action in ('move:1','move:2'):
            req=replace(c.executor.request,action_id=action)
            d.observe('brake_rejection',c.obs,req,{'reason':'possible earlier guard'})
        req=replace(c.executor.request,action_id='move:2')
        d.observe('release_overspeed_brake',replace(c.obs,time=68.22),req,{'ownership':'RELEASE'})
        d.observe('termination',replace(c.obs,time=77.02),req,{'outcome':'failure'})
        self.assertEqual([x.metadata['event_kind'] for x in d.captures],
            ['brake_rejection','brake_rejection','release_overspeed_brake','termination'])
        self.assertEqual(d.counts['cap_dropped'],0)

if __name__=='__main__':unittest.main()
