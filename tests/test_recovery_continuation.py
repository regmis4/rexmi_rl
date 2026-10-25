"""Fixed functional recovery, with physical/evidence ownership kept distinct."""
import copy
from dataclasses import replace
import importlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from test_behavior_navigation import b,m,ground,observation
C=importlib.import_module('rexmi_bt_tests.nav.recovery_continuation')


class RecoveryContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.grid=ground()

    def cert(self,origin=(-.4,.6),center=(-2.4,.6),issued=2.02,deadline=100.):
        return C.RecoveryContinuation('escape:region','escape',origin,center,issued,deadline,0.,(0.,0.),math.pi)

    def executor(self,enabled=True):
        e=m.MovementExecutor();e.brake_guard=e.posture_guard=lambda *args:''
        e.straight_entry_guard=lambda *args:''
        e.recovery_stage_guard=lambda obs:''
        e.start(m.MovementRequest('escape',(0.,0.),(-2.,0.),-1,.4,100.,purpose='recovery'),0.)
        e.committed_heading=0.;e.travel_axis=(-1.,0.)
        e._state(m.MovementState.DRIVE_REVERSE,0.,'fixture')
        if enabled:
            e.recovery_continuation_provider=lambda obs,req,samples:self.cert(origin=tuple(obs.xy),issued=obs.time)
            e.recovery_continuation_guard=lambda *args:''
            e.recovery_continuation_admitted=lambda *args:None
        for t in np.arange(0.,2.001,.02):
            self.tick(e,self.obs(float(t),(-.2*t,0.)))
        return e

    def obs(self,t,xy,**kw):
        return replace(observation(t,xy=xy,velocity=(-.1,0.),grade=25.,turn=False,**kw),terrain_gradient=(.4,0.))

    def tick(self,e,obs,guard=None):return e.tick(obs,guard or (lambda cmd:''),lambda xy:True)

    def enter(self):
        e=self.executor();self.assertEqual(self.tick(e,self.obs(2.02,(-.404,.6))),m.Motion(-.4,0.))
        self.assertIsNotNone(e.recovery_continuation);return e

    def test_admission_preserves_request_axis_clocks_progress_and_splits_attribution(self):
        e=self.executor();req=e.request;axis=e.travel_axis;started=e.started
        e.stall_since=1.;e.stall_origin=.39
        old=list(e.samples);executed=e.executed_traversal;seen=[]
        e.recovery_continuation_admitted=lambda obs,c,trace:seen.append(trace)
        self.tick(e,self.obs(2.02,(-.404,.6)))
        self.assertIs(e.request,req);self.assertIs(e.travel_axis,axis)
        self.assertEqual(e.started,started);self.assertEqual(e.stall_since,1.)
        self.assertEqual(e.stall_origin,.39);self.assertEqual(e.samples[0],old[1])
        self.assertEqual(seen[0].interval_end_time,2.02)
        self.assertIsNone(e.executed_traversal)
        self.assertEqual(e.pending_traversal[0],2.02)
        self.assertGreater(seen[0].duration_s,executed.duration_s)

    def test_old_endpoint_arrival_crossing_and_overshoot_cannot_credit_success(self):
        e=self.enter();c=e.recovery_continuation
        for t,xy in [(2.04,(-1.8,0.)),(2.06,(-2.4,.1))]:
            self.tick(e,self.obs(t,xy));self.assertIs(e.recovery_continuation,c)
            self.assertIsNone(e.result);self.assertNotEqual(e.state,m.MovementState.SETTLE)

    def test_missing_rejected_and_default_disabled_keep_original_lateral_failure(self):
        for kind in ('disabled','none','guard'):
            e=self.executor(enabled=kind!='disabled')
            if kind=='none':e.recovery_continuation_provider=lambda *args:None
            if kind=='guard':e.recovery_continuation_guard=lambda *args:'bad certificate'
            self.tick(e,self.obs(2.02,(-.404,.6)))
            self.assertTrue(e.result.physical);self.assertIn('lateral departure',e.result.reason)
            self.assertIsNone(e.recovery_continuation)

    def test_boundary_with_simultaneous_backslide_or_stall_uses_physical_failure(self):
        for kind in ('backslide','stall'):
            e=self.executor();calls=[];e.recovery_continuation_provider=lambda *args:calls.append(args)
            e.samples=[(0.,1. if kind=='backslide' else .403,(-1.,0.))]
            if kind=='stall':e.stall_since=-2.;e.stall_origin=.403
            self.tick(e,self.obs(2.02,(-.404,.6)))
            self.assertFalse(calls);self.assertTrue(e.result.physical)
            self.assertIn('backslide',e.result.reason)

    def test_tip_and_stale_stop_before_admission(self):
        for kind in ('tip','stale'):
            e=self.executor();calls=[];e.recovery_continuation_provider=lambda *args:calls.append(args)
            obs=self.obs(2.02,(-.404,.6));obs=replace(obs,up_z=.4) if kind=='tip' else replace(obs,fresh=False)
            self.tick(e,obs);self.assertFalse(calls);self.assertIsNotNone(e.result)

    def test_nonfinite_observations_never_admit_or_continue_motion(self):
        changes=({'up_z':math.nan},{'xy':(math.nan,.6)},{'velocity':(math.inf,0.)},
                 {'yaw':math.nan},{'time':math.nan},{'yaw_rate':math.nan},
                 {'inclination':math.nan},{'body_quaternion':(1.,math.nan,0.,0.)})
        for active in (False,True):
            for change in changes:
                e=self.enter() if active else self.executor()
                command=self.tick(e,replace(self.obs(2.04,(-.41,.6)),**change))
                self.assertEqual(command,m.Motion());self.assertEqual(e.result.outcome,'censored')
                self.assertFalse(e.result.physical);self.assertIsNone(e.pending_traversal)
                if not active:self.assertIsNone(e.recovery_continuation)

    def test_repeated_tick_does_not_admit_twice_or_extend_deadline(self):
        e=self.enter();cert=e.recovery_continuation;samples=list(e.recovery_region_samples)
        for _ in range(20):self.tick(e,self.obs(2.02,(-.404,.6)))
        self.assertIs(e.recovery_continuation,cert);self.assertEqual(e.recovery_region_samples,samples)
        self.assertEqual(sum(x['event']=='recovery_continuation_admitted' for x in e.history),1)
        self.tick(e,self.obs(100.01,(-.5,.6)))
        self.assertIn('deadline',e.result.reason)

    def test_guard_loss_censors_without_new_target_or_failure_vote(self):
        e=self.enter();cert=e.recovery_continuation
        e.recovery_continuation_guard=lambda *args:'stale stopping room'
        self.tick(e,self.obs(2.04,(-.41,.6)))
        self.assertEqual(e.result.outcome,'censored');self.assertFalse(e.result.physical)
        self.assertIs(e.recovery_continuation,cert)

    def test_region_progress_cannot_be_replaced_by_sideways_original_axis_progress(self):
        e=self.enter();cert=e.recovery_continuation
        e.recovery_region_samples=[(.02,1.)]
        e.recovery_region_stall_since=-1.
        self.tick(e,self.obs(2.04,(-.41,.6)))
        self.assertEqual(e.result.outcome,'censored');self.assertIn('frozen recovery region',e.result.reason)
        self.assertIs(e.recovery_continuation,cert)

    def test_brake_and_posture_preserve_certificate_original_axis_and_samples(self):
        e=self.enter();cert=e.recovery_continuation;axis=e.travel_axis
        samples=list(e.samples);e.stall_since=1.;e.stall_origin=.4
        e._state(m.MovementState.BRAKE,2.03,'overspeed fixture')
        self.assertEqual(e.samples,samples);self.assertEqual(e.stall_since,1.)
        e._state(m.MovementState.POSTURE_PAUSE,2.04,'posture fixture')
        self.assertEqual(e.samples,samples);self.assertIs(e.recovery_continuation,cert);self.assertIs(e.travel_axis,axis)
        e._state(m.MovementState.BRAKE,2.05,'resume')
        for i in range(20):self.tick(e,self.obs(2.06+i*.02,(-.41-.002*i,.6)))
        self.assertEqual(e.state,m.MovementState.DRIVE_REVERSE)
        self.assertIs(e.recovery_continuation,cert)

    def test_only_fresh_inner_core_can_enter_measured_staging_and_reposition(self):
        e=self.enter();cert=e.recovery_continuation
        xy=cert.center
        for i in range(32):
            obs=replace(self.obs(2.04+i*.02,xy),velocity=(0.,0.),inclination=0.,turn_allowed=True)
            self.tick(e,obs)
        self.assertEqual(e.result.outcome,'censored')
        self.assertEqual(e.result.disposition,m.ResultDisposition.REPOSITIONED)
        self.assertEqual(e.result.action_id,'escape');self.assertFalse(e.result.physical)
        self.assertIsNotNone(e.consume());self.assertIsNone(e.consume())

    def test_staging_cannot_drift_outside_outer_region(self):
        e=self.enter();c=e.recovery_continuation
        self.tick(e,replace(self.obs(2.04,c.center),velocity=(0.,0.),inclination=0.,turn_allowed=True))
        self.tick(e,replace(self.obs(2.06,(c.center[0]-1.01,c.center[1])),velocity=(0.,0.),inclination=0.,turn_allowed=True))
        self.assertEqual(e.result.outcome,'censored');self.assertFalse(e.result.physical)
        self.assertIn('bounded observation',e.result.reason)

    def test_stationary_stage_retains_six_second_censored_timeout(self):
        e=self.enter();c=e.recovery_continuation
        for i in range(301):
            obs=replace(self.obs(2.04+i*.02,c.center),velocity=(0.,0.),inclination=0.,
                        turn_allowed=False,yaw_rate=.2)
            self.tick(e,obs)
            if i<300:self.assertIsNone(e.result)
        self.assertEqual(e.result.outcome,'censored');self.assertFalse(e.result.physical)
        self.assertIn('observation deadline',e.result.reason)

    def test_core_posture_wait_has_fixed_censored_deadline_without_physical_stall(self):
        e=self.enter();c=e.recovery_continuation
        for i in range(301):
            obs=replace(self.obs(2.04+i*.02,c.center),velocity=(0.,0.),inclination=0.,
                        up_z=math.cos(math.radians(21.)),turn_allowed=False)
            self.tick(e,obs)
            if i<300:self.assertIsNone(e.result)
        self.assertEqual(e.recovery_core_started,2.04);self.assertIsNone(e.recovery_stage_started)
        self.assertEqual(e.result.outcome,'censored');self.assertFalse(e.result.physical)
        self.assertIn('core observation deadline',e.result.reason)

    def test_core_clock_survives_jitter_and_later_stage_entry(self):
        e=self.enter();c=e.recovery_continuation
        for i in range(260):
            xy=(c.center[0]+(.51 if i%2 else .49),c.center[1])
            self.tick(e,replace(self.obs(2.04+i*.02,xy),velocity=(0.,0.),inclination=0.,
                up_z=math.cos(math.radians(21.)),turn_allowed=False))
            self.assertIsNone(e.result)
        self.assertEqual(e.recovery_core_started,2.04)
        for i in range(260,301):
            self.tick(e,replace(self.obs(2.04+i*.02,c.center),velocity=(0.,0.),inclination=0.,
                yaw_rate=.2,turn_allowed=False))
            if i<300:self.assertIsNone(e.result)
        self.assertEqual(e.recovery_stage_started,2.04)
        self.assertFalse(e.result.physical);self.assertEqual(e.result.outcome,'censored')

    def test_core_observation_still_detects_actual_backslide(self):
        e=self.enter();c=e.recovery_continuation
        self.tick(e,replace(self.obs(2.04,c.center),velocity=(0.,0.),inclination=0.,
                up_z=math.cos(math.radians(21.)),turn_allowed=False))
        for i in range(1,103):
            self.tick(e,replace(self.obs(2.04+i*.02,(c.center[0]+.2*i*.02,c.center[1])),
                velocity=(.2,0.),inclination=0.,up_z=math.cos(math.radians(21.)),turn_allowed=False))
            if e.result:break
        self.assertTrue(e.result.physical);self.assertIn('backslide',e.result.reason)

    def test_brake_resumption_requires_existing_straight_entry_guard(self):
        for callback in (None,lambda *args:'unobserved straight stopping fan'):
            e=self.enter();e.straight_entry_guard=callback
            e._state(m.MovementState.BRAKE,2.03,'fixture brake')
            for i in range(30):
                self.tick(e,self.obs(2.04+i*.02,(-.41-.002*i,.6)))
                if e.result:break
            self.assertIsNotNone(e.result);self.assertEqual(e.result.outcome,'censored')

    def test_opposite_body_heading_cannot_resume_or_emit_reverse(self):
        e=self.enter();obs=replace(self.obs(2.04,(-.41,.6)),yaw=math.pi)
        command=self.tick(e,obs)
        self.assertEqual(e.result.outcome,'censored');self.assertEqual(command.omega,0.)
        self.assertIn('no longer advances',e.result.reason)

    def test_occupied_outer_disk_is_not_permission_to_drive_away_from_inner_core(self):
        c=self.controller();certificate=self.cert();obs=replace(c.obs,xy=(-1.6,.6),yaw=math.pi)
        self.assertLess(math.dist(obs.xy,certificate.center),certificate.outer_radius)
        self.assertIn('entry core',c._recovery_region_guard(obs,certificate,False))

    def test_stale_full_region_cannot_enter_core_even_when_current_footprint_fresh(self):
        c=self.controller();certificate=self.cert();obs=replace(c.obs,xy=certificate.center,fresh=True)
        cell=c.grid.world_to_cell(certificate.center[0]+.9,certificate.center[1])
        c.grid.fresh[cell]=False
        self.assertIn('fully fresh',c._recovery_region_guard(obs,certificate,False))

    def test_exclusion_inside_static_approach_cannot_be_crossed_or_erased(self):
        g=copy.deepcopy(self.grid);certificate=self.cert()
        g.mark_failure('older',(-1.4,.6),math.pi,1.,'existing reverse failure')
        before=g.approach_blocked.copy()
        self.assertTrue(C.approach_reason(g,certificate))
        np.testing.assert_equal(g.approach_blocked,before)

    def test_whole_region_rejects_interior_hole_slope_and_exclusions(self):
        for kind in ('unknown','hazard','slope','exclusion'):
            g=copy.deepcopy(self.grid);c=self.cert();cell=g.world_to_cell(*c.center)
            if kind=='unknown':g.known[cell]=False
            elif kind=='hazard':g.hazard[cell]=True
            elif kind=='slope':g.slope[cell]=math.tan(math.radians(13.))
            else:g.approach_blocked[cell][0]=True
            self.assertTrue(C.region_reason(g,c),kind)

    def test_surveyed_region_stays_ineligible_for_staging_until_wholly_fresh(self):
        g=copy.deepcopy(self.grid);c=self.cert();cell=g.world_to_cell(*c.center);g.fresh[cell]=False
        self.assertEqual(C.region_reason(g,c),'')
        self.assertIn('fresh',C.region_reason(g,c,fresh=True))

    def test_approach_and_stopping_extension_cannot_skip_boulder(self):
        g=copy.deepcopy(self.grid);c=self.cert()
        self.assertEqual(C.approach_reason(g,c),'')
        g.hazard[g.world_to_cell(-3.8,.6)]=True
        self.assertTrue(C.approach_reason(g,c))

    def test_select_region_rejects_missing_heading_wrong_way_and_wide_spread(self):
        g=copy.deepcopy(self.grid);obs=self.obs(2.,(-.4,0.));req=m.MovementRequest('escape',(0.,0.),(-2.,0.),-1,.4,100.,purpose='recovery')
        samples=[(float(t),(-.2*t,0.),0.) for t in np.arange(0.,2.001,.02)]
        for case in ('missing','uphill','wide'):
            current=replace(obs,terrain_gradient=(-.4,0.)) if case=='uphill' else obs
            forecast=None if case=='missing' else (-2.,2.) if case=='wide' else (0.,0.)
            cert,reason=C.select_region(g,current,req,samples,forecast)
            self.assertIsNone(cert);self.assertTrue(reason)

    def controller(self):
        c=b.BehaviorTreeAscentController(copy.deepcopy(self.grid),[SimpleNamespace(x=3.,y=0.,label='goal')],mode='characterize')
        c.configure_uphill_speed_response();c.configure_recovery_continuation()
        c.recovering=True;c.recovery_since=0.;c.recovery_actions=1
        c.obs=self.obs(2.02,(-.404,.6));c.executor=self.executor()
        return c

    def test_phase_split_censors_old_prediction_once_and_never_validates_region(self):
        c=self.controller();cert=self.cert(origin=c.obs.xy);e=c.executor
        features=[0.]*14;estimate=c.evidence.estimate('old',.9)
        c.attempt=dict(id='old',time=0.,start=(0.,0.),endpoint=(-2.,0.),heading=math.pi,
            features=features,estimate=estimate,key='old',path=[(0.,0.),(-2.,0.)]);c.prediction_committed=True
        with patch.object(c.model,'predict',return_value=(.95,'point model valid')):
            c._adopt_recovery_region(c.obs,cert,e.executed_traversal)
        self.assertTrue(c.recovery_continuation_used);self.assertEqual(c.recovery_actions,1)
        self.assertEqual(c.recovery_since,0.);self.assertFalse(c.attempt['estimate'].validated)
        self.assertIsNone(c.attempt['estimate'].probability)
        ordinary=c.evidence.key(c.obs.xy,cert.center,math.atan2(cert.axis[1],cert.axis[0]),
            -c.command_limit,path=[c.obs.xy,cert.center])
        self.assertNotEqual(c.attempt['key'],ordinary)
        self.assertTrue(c.attempt['key'].startswith('recovery-region-v31:outer=1.0:inner=0.5:'))
        self.assertNotIn('old',c.evidence.counts)
        outcomes=[x for x in c.attempt_events if x.get('event')=='approach_outcome']
        self.assertEqual(len(outcomes),1);self.assertEqual(outcomes[0]['outcome'],'censored')
        self.assertFalse(c.grid.approach_blocked.any())
        self.assertIsNone(c._recovery_region_candidate(c.obs,e.request,[]))
        c.finish_attempt('censored','stage settled',c.obs.xy,0.,3.)
        c.finish_attempt('censored','duplicate',c.obs.xy,0.,3.)
        self.assertEqual(len([x for x in c.attempt_events if x.get('event')=='approach_outcome']),2)
        self.assertFalse(c.evidence.counts)

    def test_phase_execution_heading_uses_first_measured_reverse_output_not_plan(self):
        e=self.executor()
        e.recovery_continuation_provider=lambda obs,req,samples:self.cert(
            origin=tuple(obs.xy),center=(-2.4,1.0),issued=obs.time)
        obs=replace(self.obs(2.02,(-.404,.6)),yaw=.15)
        self.tick(e,obs)
        expected=m.wrap_angle(.15+math.pi)
        planned=math.atan2(e.recovery_continuation.axis[1],e.recovery_continuation.axis[0])
        self.assertGreater(abs(m.wrap_angle(expected-planned)),.1)
        self.tick(e,replace(self.obs(2.04,(-.408,.6)),yaw=.17))
        self.assertAlmostEqual(e.executed_traversal.heading,expected)
        self.assertAlmostEqual(e.recovery_execution_heading,expected)
        e._state(m.MovementState.BRAKE,2.05,'interruption')
        self.assertAlmostEqual(e.recovery_execution_heading,expected)
        self.assertEqual(e.travel_axis,(-1.,0.))

    def test_stop_invalidates_certificate_output_without_success_or_new_prediction(self):
        c=self.controller();c.executor=self.enter();c.request('stop',2.04)
        command=c.tick((-.41,.6),0.,.1,2.06,velocity=(-.1,0.))
        self.assertEqual(command,m.Motion());self.assertEqual(c.state,'HOLD')
        self.assertIsNone(c.executor.request)

    def test_invalid_input_remains_zero_through_controller_result_and_recovery_ticks(self):
        for kind in ('velocity','position','upright'):
            c=self.controller();c.executor=self.enter()
            for t in (2.04,2.06,2.08):
                command=c.tick((math.nan,.6) if kind=='position' else (-.41,.6),0.,.1,t,
                    upright=math.nan if kind=='upright' else 1.,
                    velocity=(math.nan,0.) if kind=='velocity' else (-.1,0.))
                self.assertEqual(command,m.Motion());self.assertEqual(c.state,'HOLD')
            self.assertIsNone(c.executor.request);self.assertIsNone(c.executor.result)
            self.assertEqual(sum(x['event']=='navigation_observation_invalid' for x in c.escape_events),1)
            self.assertFalse(c.grid.approach_blocked.any())
            self.assertEqual(c.tick((-.41,.6),0.,0.,2.10,velocity=(0.,0.)),m.Motion())
            self.assertTrue(c.invalid_observation_hold)
            c.request('resume',2.12);self.assertFalse(c.invalid_observation_hold)
            self.assertEqual(c.tick((-.41,.6),0.,.1,2.14,velocity=(math.nan,0.)),m.Motion())
            self.assertTrue(c.invalid_observation_hold)

    def test_invalid_input_hold_keeps_stop_and_manual_precedence(self):
        c=self.controller();c.executor=self.enter()
        c.tick((-.41,.6),0.,.1,2.04,velocity=(math.nan,0.))
        c.request('manual',2.06,vx=-.4)
        self.assertEqual(c.tick((math.nan,.6),0.,.1,2.08,velocity=(math.nan,0.)),m.Motion(-.4,0.))
        c.request('stop',2.10)
        self.assertEqual(c.tick((math.nan,.6),0.,.1,2.12,velocity=(math.nan,0.)),m.Motion())

    def test_invalid_input_manual_cannot_override_current_tip_or_latched_fault(self):
        for previous_fault in (False,True):
            c=self.controller();c.executor=self.enter();c.request('manual',2.03,vx=-.4)
            if previous_fault:c.fault='previously established fault'
            for t in (2.04,2.06,2.08):
                command=c.tick((math.nan,.6),0.,.1,t,upright=1. if previous_fault else .4,
                               velocity=(math.nan,0.))
                self.assertEqual(command,m.Motion());self.assertEqual(c.state,'HOLD')
            self.assertTrue(c.fault);self.assertFalse(c.grid.approach_blocked.any())
            c.request('stop',2.10)
            self.assertEqual(c.tick((math.nan,.6),0.,.1,2.12,velocity=(math.nan,0.)),m.Motion())

    def test_invalid_first_observation_does_not_invent_execution_or_pose(self):
        c=self.controller();c.obs=None;c.executor=m.MovementExecutor()
        for t in (0.,.02,.04):
            self.assertEqual(c.tick((math.nan,0.),0.,0.,t),m.Motion())
        self.assertFalse(c.attempt_events);self.assertIsNone(c.obs)
        events=[x for x in c.escape_events if x['event']=='navigation_observation_invalid']
        self.assertEqual(len(events),1);self.assertIsNone(events[0]['last_valid_time'])

    def test_invalid_next_observation_preserves_completed_physical_result_once(self):
        c=self.controller();e=c.executor
        c.attempt=dict(id='physical',time=0.,start=(0.,0.),endpoint=(-2.,0.),heading=math.pi,
            features=[0.]*14,estimate=c.evidence.estimate('physical',None),key='physical',path=[(0.,0.),(-2.,0.)])
        c.prediction_committed=True
        e.finish('failure','completed physical failure',c.obs,physical=True)
        original=e.result
        for t in (2.04,2.06,2.08):
            self.assertEqual(c.tick((-.41,.6),0.,.1,t,velocity=(math.nan,0.)),m.Motion())
        self.assertEqual(c.evidence.counts['physical'],(0,1))
        self.assertEqual(c.climb_failures,1)
        self.assertEqual(len(c.grid.failure_records),1)
        results=[x for x in c.attempt_events if x.get('event')=='movement_outcome']
        self.assertEqual(len(results),1);self.assertEqual(results[0]['time'],original.time)
        self.assertTrue(results[0]['physical']);self.assertEqual(results[0]['position'],original.position)

    def test_episode_budget_and_one_use_survive_new_request(self):
        c=self.controller();c.recovery_continuation_used=True
        c._begin_recovery('another primitive failed')
        self.assertTrue(c.recovery_continuation_used);self.assertEqual(c.recovery_since,0.)
        for used,actions in ((True,1),(False,8)):
            c.recovery_continuation_used=used;c.recovery_actions=actions
            self.assertIsNone(c._recovery_region_candidate(c.obs,c.executor.request,[]))
        c.recovering=False;c._begin_recovery('new independent episode')
        self.assertFalse(c.recovery_continuation_used)

    def test_region_failure_is_preserved_but_cannot_label_point_model(self):
        path=Path(__file__).resolve().parents[1]/'scripts/fit_reachability.py'
        spec=importlib.util.spec_from_file_location('recovery_point_fitter',path)
        fit=importlib.util.module_from_spec(spec);spec.loader.exec_module(fit)
        point=dict(event='approach_outcome',outcome='failure',physical=True,feature_version=2,
            run_id='run',attempt_id='point',end_time=1.,approach_key='point')
        region=dict(point,attempt_id='region',end_time=2.,target_kind='recovery_staging_region',
                    approach_key='recovery-region-v31:outer=1.0:inner=0.5:point')
        with tempfile.TemporaryDirectory() as folder:
            log=Path(folder)/'run.events.jsonl';text='\n'.join(json.dumps(e) for e in (point,region))+'\n'
            log.write_text(text)
            (Path(folder)/'run.summary.json').write_text(json.dumps(dict(result='hold',manual_intervention=False)))
            rows=fit.read_runs([log]);self.assertEqual([r['attempt_id'] for r in rows],['point'])
            self.assertEqual(log.read_text(),text)
        with self.assertRaisesRegex(ValueError,'functional region'):
            fit.fit([region],[point],[point])


if __name__=='__main__':unittest.main()
