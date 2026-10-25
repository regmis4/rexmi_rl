"""Straight takeover uses a bounded heading envelope; turning still settles."""
from dataclasses import asdict, replace
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from test_behavior_navigation import b, m, ground, observation


class StraightAdmissionTests(unittest.TestCase):
    def executor(self, sign=-1, endpoint=None):
        e=m.MovementExecutor()
        e.straight_entry_guard=lambda *args: ''
        e.start(m.MovementRequest('straight',(0.,0.),endpoint or (2.*sign,0.),sign,.4,100.,'adjust'),0.)
        return e

    def obs(self,t,velocity=(.08,0.),yaw=None,**kwargs):
        return observation(t,velocity=velocity,yaw=.018*math.sin(10*t) if yaw is None else yaw,
                           yaw_rate=.18*math.cos(10*t),grade=30.,turn=False,**kwargs)

    def tick(self,e,obs,guard=None):
        return e.tick(obs,guard or (lambda cmd: ''),lambda xy: True)

    def test_aligned_reverse_accepts_small_heading_oscillation(self):
        e=self.executor()
        for t in np.arange(0.,.51,.02): command=self.tick(e,self.obs(t))
        self.assertEqual(e.state,m.MovementState.DRIVE_REVERSE)
        self.assertEqual(command,m.Motion(-.4,0.))
        self.assertTrue(any(h['reason']=='committed aligned straight takeover' for h in e.history))

    def test_opposing_motion_must_decelerate_before_straight_reversal(self):
        e=self.executor()
        for t in np.arange(0.,.61,.02): self.tick(e,self.obs(t,velocity=(.2,0.)))
        self.assertEqual(e.state,m.MovementState.BRAKE)
        for t in np.arange(.62,1.01,.02): command=self.tick(e,self.obs(t))
        self.assertEqual(e.state,m.MovementState.DRIVE_REVERSE)
        self.assertEqual(command,m.Motion(-.4,0.))

    def test_overspeed_sample_cannot_be_skipped_by_observation_window(self):
        e=self.executor()
        for t in np.arange(0.,.25,.02): self.tick(e,self.obs(t))
        self.tick(e,self.obs(.26,velocity=(.6,0.)))
        for t in np.arange(.28,.56,.02): self.tick(e,self.obs(t))
        self.assertEqual(e.state,m.MovementState.BRAKE)
        for t in np.arange(.56,.72,.02): self.tick(e,self.obs(t))
        self.assertEqual(e.state,m.MovementState.DRIVE_REVERSE)

    def test_high_speed_sustained_rotation_and_large_oscillation_do_not_admit(self):
        for case in ('speed','rotation','oscillation'):
            e=self.executor()
            for t in np.arange(0.,.61,.02):
                obs=self.obs(t,velocity=(-.26,0.) if case=='speed' else (.08,0.),
                             yaw=.4*t if case=='rotation' else .08*math.sin(30*t) if case=='oscillation' else None)
                self.tick(e,obs)
            self.assertNotEqual(e.state,m.MovementState.DRIVE_REVERSE)

    def test_heading_wrap_does_not_fabricate_excursion(self):
        e=self.executor(sign=1,endpoint=(-2.,0.))
        for t in np.arange(0.,.51,.02):
            yaw=math.atan2(math.sin(math.pi+.018*math.sin(10*t)),math.cos(math.pi+.018*math.sin(10*t)))
            self.tick(e,self.obs(t,velocity=(0.,0.),yaw=yaw))
        self.assertEqual(e.state,m.MovementState.DRIVE_FORWARD)

    def test_stationary_turn_gate_is_unchanged(self):
        e=self.executor(sign=1,endpoint=(0.,2.))
        for t in np.arange(0.,.61,.02):
            self.tick(e,replace(self.obs(t),inclination=0.,turn_allowed=True))
        self.assertEqual(e.state,m.MovementState.BRAKE)

    def test_forecast_remaining_and_emitted_corridors_all_remain_binding(self):
        for case in ('forecast','remaining','emitted'):
            e=self.executor()
            if case=='forecast': e.straight_entry_guard=lambda *args: 'forecast blocked'
            if case=='remaining': e.corridor_guard=lambda *args: 'remaining blocked'
            for t in np.arange(0.,.61,.02):
                self.tick(e,self.obs(t),guard=(lambda cmd: 'emitted blocked') if case=='emitted' else None)
            self.assertNotEqual(e.state,m.MovementState.DRIVE_REVERSE)
            self.assertEqual(e.result.outcome,'censored')
            self.assertFalse(e.result.physical)

    def test_new_request_cannot_inherit_previous_admission_window(self):
        e=self.executor()
        for t in (0.,.1,.2): self.tick(e,self.obs(t))
        e.stop(self.obs(.21));e.consume()
        e.start(m.MovementRequest('new',(0.,0.),(-2.,0.),-1,.4,100.,'adjust'),.22)
        self.tick(e,self.obs(.23))
        self.assertEqual(e.state,m.MovementState.BRAKE)

    def test_forecast_fan_checks_intermediate_headings(self):
        c=b.BehaviorTreeAscentController(ground(),[SimpleNamespace(x=3.,y=0.,label='g')],mode='characterize')
        obs=observation(0.,velocity=(.1,0.))
        req=m.MovementRequest('fan',(0.,0.),(-2.,0.),-1,.4,100.,'adjust')
        with patch.object(c,'_straight_clear',side_effect=lambda start,end,yaw: not .08<yaw<.13) as check:
            self.assertIn('forecast',c._straight_entry_envelope(obs,req,.1))
            self.assertGreater(check.call_count,2)

    def test_zero_net_oscillation_checks_entire_measured_fan_both_directions(self):
        c=b.BehaviorTreeAscentController(ground(),[SimpleNamespace(x=3.,y=0.,label='g')],mode='characterize')
        for sign in (-1,1):
            e=self.executor(sign)
            e.straight_entry_guard=c._straight_entry_envelope
            with patch.object(c,'_straight_clear',side_effect=lambda start,end,yaw: abs(yaw)<.025):
                for t in np.arange(0.,.32,.02):
                    obs=replace(self.obs(t,velocity=(0.,0.),yaw=.04*math.sin(2*math.pi*t/.3)),yaw_rate=.8)
                    self.tick(e,obs)
            self.assertEqual(e.result.outcome,'censored')
            self.assertFalse(e.result.physical)
            self.assertIsNone(e.result.executed_traversal)

    def test_denied_same_request_resume_censors_even_with_executed_evidence(self):
        e=self.executor(sign=1)
        for t in np.arange(0.,.61,.02): self.tick(e,self.obs(t,velocity=(0.,0.)))
        self.assertIsNotNone(e.executed_traversal)
        e._state(m.MovementState.BRAKE,.62,'fixture interruption')
        e.straight_entry_guard=lambda *args:'forecast blocked'
        for t in np.arange(.62,1.03,.02): self.tick(e,self.obs(t,velocity=(0.,0.)))
        result=e.result
        self.assertEqual(result.outcome,'censored')
        self.assertFalse(result.physical)
        self.assertIsNotNone(result.executed_traversal)
        c=b.BehaviorTreeAscentController(ground(),[SimpleNamespace(x=3.,y=0.,label='g')],mode='characterize')
        c.tick((0.,0.),0.,0.,0.)
        self.assertTrue(c._compute_plan((0.,0.),0.,0.))
        c.finish_attempt(result.outcome,result.reason,result.position,0.,result.time,
                         physical=result.physical,execution=asdict(result.executed_traversal),terminal_maneuver=result.phase)
        event=c.attempt_events[-1]
        self.assertEqual(event['outcome'],'censored')
        self.assertFalse(event['exclusion_applied'])
        self.assertFalse(c.grid.no_go.any())

    def test_sparse_fresh_observations_do_not_establish_continuous_admission(self):
        e=self.executor()
        for t in (0.,.3): self.tick(e,self.obs(t,velocity=(0.,0.),yaw=0.))
        self.assertEqual(e.state,m.MovementState.BRAKE)


if __name__=='__main__':unittest.main()
