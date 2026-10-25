"""Tracking is not terrain failure; checked zero curvature stays bounded."""
import math
import json
from pathlib import Path
from dataclasses import replace
import unittest
import numpy as np
import test_active_zero_envelope as zero_fixture
from test_behavior_navigation import m, ground


class ActiveZeroDepartureTests(unittest.TestCase):
    def setup_controller(self, recorded=False):
        helper=zero_fixture.ActiveZeroTests()
        if recorded:
            c,f=helper.recorded('active_zero_departure_approximate.json')
            self.assertIn('APPROXIMATE',f['provenance']['terrain'])
            # Fixture contains a local patch only: explicitly mock distant geometry.
            c._remaining_corridor=lambda *args:''
        else:
            c=helper.synthetic();c.grid=ground(30.);c.grid.uphill_limit_deg=33.;e=c.executor
            c.obs=replace(c.obs,xy=(.5,.501),velocity=(.1,0.))
            e.request=replace(e.request,start=(0.,0.),endpoint=(4.,0.))
            e.uphill_samples=[(float(t),(.5+.1*(t-c.obs.time),.501)) for t in np.linspace(c.obs.time-2,c.obs.time,101)]
            e.samples=[(t,xy[0],xy) for t,xy in e.uphill_samples]
        c.executor.release_origin=c.executor.request.start
        return c

    def tick_recorded_state(self,c):
        e=c.executor;o=c.obs
        e.uphill_samples=e.uphill_samples[:-1];e.samples=e.samples[:-1]
        t,xy=(e.uphill_samples[-1] if e.uphill_samples else (o.time-.02,o.xy))
        e.pending_traversal=(t,xy,o.yaw,e.state.value,1,e.committed_heading)
        e.last_output_time=t
        return e.tick(o,c._guard,lambda xy:True)

    def test_recorded_crossing_with_explicitly_approximate_terrain(self):
        c=self.setup_controller(True);e=c.executor;req=e.request
        before=(e.started,e.entered,req.deadline,e.committed_heading,e.travel_axis)
        self.assertEqual(self.tick_recorded_state(c),m.Motion())
        self.assertEqual(e.state,m.MovementState.RELEASE)
        self.assertIs(e.request,req);self.assertIsNone(e.result)
        self.assertEqual(before,(e.started,e.entered,req.deadline,e.committed_heading,e.travel_axis))
        event=next(x for x in c.escape_events if x['event']=='ascent_continuation_admitted')
        self.assertAlmostEqual(event['goal_distance_intent_bound_m'],math.dist(req.start,req.endpoint)+.08)
        self.assertIn('terrain_evidence',event)

    def test_allowance_is_frozen_in_original_frame(self):
        c=self.setup_controller();req=c.executor.request
        self.assertEqual(c._active_zero_departure(c.obs,req),'')
        self.assertEqual(c.active_zero_departures[req.action_id],4.08)
        c.obs=replace(c.obs,xy=(.5,4.1),velocity=(.4,0.))
        self.assertIn('fixed goal-distance intent bound',c._active_zero_departure(c.obs,req))
        self.assertEqual(c.active_zero_departures[req.action_id],4.08)
        self.assertEqual(req.start,(0.,0.))

    def test_initial_admission_requires_actual_goal_and_uphill_progress(self):
        for kind in ('stationary','away_from_goal','backslide'):
            c=self.setup_controller();e=c.executor;o=c.obs
            if kind=='stationary':e.uphill_samples=[(t,o.xy) for t,xy in e.uphill_samples]
            elif kind=='away_from_goal':e.request=replace(e.request,endpoint=(-4.,0.))
            else:e.uphill_samples=[(t,(o.xy[0]-.1*(t-o.time),o.xy[1])) for t,xy in e.uphill_samples]
            self.assertTrue(c._active_zero_departure(o,e.request),kind)
            self.assertFalse(c.active_zero_departures)

    def test_new_hazard_and_history_loss_revoke_continuation(self):
        for kind in ('hazard','history'):
            c=self.setup_controller();e=c.executor
            self.assertEqual(c._active_zero_departure(c.obs,e.request),'')
            if kind=='hazard':
                cell=c.grid.world_to_cell(*c.obs.xy);c.grid.hazard[cell]=True;c.grid.blocked[cell]=True
            else:c.passive_heading.samples=[]
            self.assertTrue(c._active_zero_departure(c.obs,e.request),kind)

    def test_productive_drive_and_zero_share_same_contract(self):
        for state in (m.MovementState.DRIVE_FORWARD,m.MovementState.RELEASE):
            c=self.setup_controller();e=c.executor;e.state=state
            command=self.tick_recorded_state(c)
            self.assertIsNone(e.result)
            self.assertEqual(command.vx,.4 if state==m.MovementState.DRIVE_FORWARD else 0.)
            self.assertIn(e.request.action_id,c.active_zero_departures)

    def test_cold_drive_is_censored_without_a_physical_vote(self):
        c=self.setup_controller();e=c.executor;e.state=m.MovementState.DRIVE_FORWARD
        e.uphill_samples=[]
        self.tick_recorded_state(c)
        self.assertEqual(e.result.outcome,'censored');self.assertFalse(e.result.physical)

    def test_real_backslide_failure_precedes_tracking_continuation(self):
        c=self.setup_controller();e=c.executor;o=c.obs
        e.uphill_samples=[(t,(o.xy[0]-.2*(t-o.time),o.xy[1])) for t,xy in e.uphill_samples]
        self.tick_recorded_state(c)
        self.assertTrue(e.result.physical);self.assertIn('backslide',e.result.reason)
        self.assertFalse(c.active_zero_departures)

    def test_plateau_may_repower_only_after_independent_powered_checks(self):
        c=self.setup_controller();e=c.executor;o=c.obs
        self.assertEqual(c._active_zero_departure(o,e.request),'')
        e.release_bad_since=o.time-6.
        e.uphill_samples=[(t,o.xy) for t,xy in e.uphill_samples]
        e.samples=[(t,o.xy[0],o.xy) for t,xy in e.uphill_samples]
        result=self.tick_recorded_state(c)
        self.assertEqual(result.vx,.4)
        self.assertIsNone(e.result)
        self.assertEqual(e.state,m.MovementState.DRIVE_FORWARD)

    def test_powered_guard_cannot_be_bypassed(self):
        c=self.setup_controller();e=c.executor;e.state=m.MovementState.DRIVE_FORWARD
        e.executed_traversal=replace(e.executed_traversal,maneuver='DRIVE_FORWARD')
        c._guard=lambda *args:'powered blocked'
        self.assertIn('blocked',c._active_zero_departure(c.obs,e.request))
        self.assertFalse(c.active_zero_departures)

    def test_all_task_outputs_check_intent_bound_even_inside_centerline(self):
        for state in (m.MovementState.DRIVE_FORWARD,m.MovementState.RELEASE):
            c=self.setup_controller();e=c.executor;e.state=state
            self.assertEqual(c._active_zero_departure(c.obs,e.request),'')
            obs=replace(c.obs,xy=(-.09,0.),velocity=(0.,0.))
            command=e._emit_traversal(m.Motion(.4 if state==m.MovementState.DRIVE_FORWARD else 0.,0.),obs)
            self.assertEqual(command,m.Motion())
            self.assertFalse(e.result.physical)
            self.assertIn('intent bound',e.result.reason)

    def test_unadmitted_forward_and_descent_do_not_gain_new_bound(self):
        c=self.setup_controller();e=c.executor;e.state=m.MovementState.DRIVE_FORWARD
        for grade in (30.,0.):
            obs=replace(c.obs,xy=(-.2,0.),inclination=grade)
            self.assertEqual(e._emit_traversal(m.Motion(.4,0.),obs),m.Motion(.4,0.))
            self.assertIsNone(e.result)

    def test_unadmitted_brake_cannot_resume_outside_line_without_evidence(self):
        c=self.setup_controller();e=c.executor;e.state=m.MovementState.BRAKE
        e.executed_traversal=None;e.uphill_samples=[]
        e._state(m.MovementState.DRIVE_FORWARD,c.obs.time,'resume')
        e._emit_traversal(m.Motion(.4,0.),c.obs)
        self.assertEqual(e.result.outcome,'censored')
        self.assertIn('execution evidence',e.result.reason)
        self.assertIsNone(e.ascent_intent_bound)

    def test_admitted_bound_survives_braking_and_clears_on_stop(self):
        c=self.setup_controller();e=c.executor
        self.assertEqual(c._active_zero_departure(c.obs,e.request),'')
        bound=e.ascent_intent_bound
        e._state(m.MovementState.BRAKE,c.obs.time,'safety')
        self.assertEqual(e.ascent_intent_bound,bound)
        obs=replace(c.obs,xy=(-.2,0.),velocity=(0.,0.))
        e._state(m.MovementState.DRIVE_FORWARD,obs.time,'resume')
        self.assertEqual(e._emit_traversal(m.Motion(.4,0.),obs),m.Motion())
        self.assertFalse(e.result.physical)
        self.assertIsNone(e.ascent_intent_bound)

    def test_brake_and_posture_own_response_outside_centerline(self):
        for state in (m.MovementState.BRAKE,m.MovementState.POSTURE_PAUSE):
            c=self.setup_controller();e=c.executor
            self.assertEqual(c._active_zero_departure(c.obs,e.request),'')
            e.state=state
            if state==m.MovementState.POSTURE_PAUSE:
                e.posture_pause=m.PosturePause(c.obs.time,m.MovementState.DRIVE_FORWARD)
                e.posture_guard=lambda *args:''
            e.entry_guard=lambda *args:'forbidden new entry'
            self.tick_recorded_state(c)
            if e.result:self.assertNotIn('lateral departure',e.result.reason)
            self.assertIn(e.request.action_id if e.request else 'zero',c.active_zero_departures)

    def test_recorded_cold_brake_restores_original_censor_time(self):
        f=json.loads((Path(__file__).parent/'fixtures/cold_brake_departure.json').read_text())
        e=m.MovementExecutor();e.start(m.MovementRequest(**f['request']),f['executor']['started'])
        for name,value in f['executor'].items():
            if name in ('state','release_memory'):continue
            if hasattr(e,name):setattr(e,name,value)
        e.state=m.MovementState.BRAKE
        e.release_departure_guard=lambda *args:'must not borrow ascent ownership'
        obs=m.MotionObservation(**f['observation'])
        e.tick(obs,lambda *args:'',lambda *args:True)
        self.assertEqual(e.result.time,32.24)
        self.assertEqual(e.result.reason,f['returned_result']['reason'])
        self.assertEqual(e.result.outcome,'censored');self.assertFalse(e.result.physical)
        self.assertEqual(e.result.phase,'BRAKE')
        self.assertIsNone(e.ascent_intent_bound)

    def test_stop_and_deadline_still_terminate(self):
        for kind in ('stop','deadline'):
            c=self.setup_controller();e=c.executor
            self.assertEqual(c._active_zero_departure(c.obs,e.request),'')
            if kind=='stop':
                self.assertEqual(e.stop(c.obs),m.Motion())
                self.assertEqual(e.state,m.MovementState.STOP)
            else:
                e.request=replace(e.request,deadline=c.obs.time-.01)
                self.tick_recorded_state(c)
                self.assertIn('deadline',e.result.reason)
            self.assertIsNone(e.request)


if __name__=='__main__':unittest.main()
