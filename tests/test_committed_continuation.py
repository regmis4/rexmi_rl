"""Established steep drive may resume without becoming a new heading attempt."""
from dataclasses import replace
import math
import unittest

import numpy as np

from test_behavior_navigation import m,observation


class CommittedContinuationTests(unittest.TestCase):
    def executor(self,sign=1):
        e=m.MovementExecutor()
        e.straight_entry_guard=e.corridor_guard=lambda *args:''
        e.release_guard=e.posture_guard=lambda *args:''
        e.start(m.MovementRequest('executed',(0.,0.),(10.*sign,0.),sign,.4,100.),0.)
        e.committed_heading=0.;e.travel_axis=(float(sign),0.)
        e._state(m.MovementState.DRIVE_FORWARD if sign>0 else m.MovementState.DRIVE_REVERSE,0.,'fixture')
        e._emit_traversal(m.Motion(sign*.4,0.),self.obs(0.,sign))
        e._observe_emitted_traversal(self.obs(.02,sign))
        e._state(m.MovementState.BRAKE,.04,'interrupted executed drive')
        return e

    def obs(self,t,sign=1):
        return replace(observation(t,xy=(-sign*.04*t,0.),yaw=.35+.015*math.sin(15*t),
            velocity=(-sign*.04,0.),grade=30.,turn=False,yaw_rate=.225*math.cos(15*t)),
            terrain_gradient=(.4,0.))

    def tick(self,e,obs,guard=None):
        return e.tick(obs,guard or (lambda cmd:''),lambda xy:True)

    def advance(self,e,sign=1,end=.5,guard=None):
        for t in np.arange(.04,end+1e-9,.02):
            cmd=self.tick(e,self.obs(float(t),sign),guard)
            if e.result:break
        return cmd

    def test_forward_and_reverse_resume_same_axis_request_and_deadline(self):
        for sign in (1,-1):
            e=self.executor(sign);request=e.request;axis=e.travel_axis
            command=self.advance(e,sign)
            self.assertEqual(command,m.Motion(sign*.4,0.))
            self.assertIs(e.request,request);self.assertEqual(e.travel_axis,axis)
            self.assertEqual(e.committed_heading,0.);self.assertEqual(request.deadline,100.)
            admitted=next(x for x in e.history if x.get('event')=='committed_drive_continuation_admitted')
            emitted=next(x for x in e.history if x.get('event')=='drive_continuation_emitted')
            self.assertGreater(emitted['time'],admitted['time'])
            self.assertEqual(emitted['admitted_at'],admitted['time'])
            self.assertEqual(emitted['command_vx'],sign*.4)
            self.assertEqual(emitted['command_omega'],0.)

    def test_missing_brake_release_mismatched_or_unobserved_execution_cannot_admit(self):
        variants=[None,dict(maneuver='BRAKE'),dict(maneuver='RELEASE'),dict(drive_sign=-1),
                  dict(heading=.2),dict(duration_s=0.),dict(interval_end_time=0.)]
        for changes in variants:
            e=self.executor()
            e.executed_traversal=None if changes is None else replace(e.executed_traversal,**changes)
            self.advance(e)
            self.assertEqual(e.state,m.MovementState.BRAKE,str(changes))

    def test_actual_alignment_and_new_requests_clear_eligibility(self):
        for kind in ('alignment','request'):
            e=self.executor()
            if kind=='alignment':
                e._state(m.MovementState.ALIGN_LEFT,.025,'actual alignment')
                e._state(m.MovementState.BRAKE,.04,'settle turn')
            else:
                e.finish('censored','new request',self.obs(.025));e.consume()
                e.start(m.MovementRequest('fresh',(0.,0.),(10.,0.),1,.4,100.),.03)
                e.committed_heading=0.;e.travel_axis=(1.,0.)
            self.assertIsNone(e.executed_traversal)
            self.advance(e);self.assertEqual(e.state,m.MovementState.BRAKE)

    def test_gentle_ground_still_requires_current_endpoint_heading(self):
        e=self.executor()
        for t in np.arange(.04,.51,.02):
            self.tick(e,replace(self.obs(float(t)),inclination=5.,turn_allowed=True))
        self.assertEqual(e.state,m.MovementState.BRAKE)
        self.assertFalse(any(x.get('event')=='committed_drive_continuation_admitted' for x in e.history))

    def test_forecast_remaining_and_emitted_geometry_remain_binding(self):
        for layer in ('forecast','remaining','command'):
            e=self.executor()
            if layer=='forecast':e.straight_entry_guard=lambda *args:'forecast blocked'
            if layer=='remaining':e.corridor_guard=lambda *args:'remaining blocked'
            self.advance(e,guard=(lambda cmd:'command blocked') if layer=='command' else None)
            self.assertEqual(e.result.outcome,'censored');self.assertFalse(e.result.physical)
            self.assertFalse(any(x.get('event')=='drive_continuation_emitted' for x in e.history))

    def test_release_ownership_cannot_be_stolen_by_drive_evidence(self):
        e=self.executor();e.suspended_release=True
        self.advance(e)
        self.assertEqual(e.state,m.MovementState.BRAKE)
        self.assertTrue(e.suspended_release)

    def test_small_negative_progress_is_allowed_but_budget_and_backslide_still_bind(self):
        e=self.executor();self.advance(e)
        self.assertEqual(e.state,m.MovementState.DRIVE_FORWARD)
        for case in ('budget','backslide','deadline','stop','tip','posture'):
            e=self.executor();obs=self.obs(.04)
            if case=='budget':e.uphill_patience=True;e.uphill_stall_since=-24.;e.uphill_stall_origin=(100.,0.);e.uphill_exposure=.2
            elif case=='backslide':e.uphill_samples=[(-1.,(1.,0.))]
            elif case=='deadline':obs=self.obs(101.)
            elif case=='stop':e.stop(obs)
            elif case=='tip':obs=replace(obs,up_z=.4)
            else:e.posture_pause=m.PosturePause(.03,m.MovementState.DRIVE_FORWARD)
            self.tick(e,obs)
            self.assertFalse(any(x.get('event')=='committed_drive_continuation_admitted' for x in e.history),case)
            self.assertNotEqual(e.state,m.MovementState.DRIVE_FORWARD,case)

    def test_interruption_after_admission_is_not_logged_as_an_emitted_drive(self):
        e=self.executor()
        for t in np.arange(.04,.51,.02):
            self.tick(e,self.obs(float(t)))
            if e.pending_drive_continuation is not None:break
        self.assertIsNotNone(e.pending_drive_continuation)
        e.stop(self.obs(float(t)+.02))
        self.assertFalse(any(x.get('event')=='drive_continuation_emitted' for x in e.history))


if __name__=='__main__':unittest.main()
