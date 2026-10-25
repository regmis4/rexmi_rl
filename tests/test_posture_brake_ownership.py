"""Posture recovery ends posture observation, not unresolved braking."""
import json
import math
from pathlib import Path
import unittest

import numpy as np

import test_posture_observation as posture_tests
from test_behavior_navigation import m


class PostureBrakeOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.helper=posture_tests.PostureObservationTests()

    def tick(self,e,t,tilt=35.,speed=.2,**kw):
        return self.helper.tick(e,self.helper.obs(t,tilt,velocity=(speed,0.),**kw))

    def test_recorded_posture_recovers_while_original_brake_episode_remains(self):
        fixture=json.loads((Path(__file__).parent/'fixtures/recovered_posture_braking.json').read_text())
        e=m.MovementExecutor();e.release_guard=e.posture_guard=e.corridor_guard=lambda *args:''
        req=fixture['request'];samples=fixture['samples'];heading=fixture['committed_heading']
        e.start(m.MovementRequest('recorded',tuple(req['start']),tuple(req['endpoint']),1,.4,
            req['deadline'],lateral_tolerance=req['lateral_tolerance'],arrival_radius=.3),samples[0]['sim_time'])
        e.committed_heading=heading;e.travel_axis=(math.cos(heading),math.sin(heading))
        e.release_origin=tuple(fixture['release_origin'])
        e._state(m.MovementState.RELEASE,48.46,'recorded release')
        request=e.request;transfers=[]
        for s in samples:
            obs=m.MotionObservation((s['x'],s['y']),s['yaw'],
                (s['velocity_world_x'],s['velocity_world_y']),s['up_z'],s['actual_yaw_rate'],
                s['sim_time'],s['support_inclination_deg'],False,True,(.3,.4))
            had_pause=e.posture_pause is not None
            cmd=e.tick(obs,lambda command:'',lambda xy:True)
            if had_pause and e.posture_pause is None:
                transfers.append(s['sim_time']);self.assertEqual(e.state,m.MovementState.BRAKE)
                self.assertEqual(cmd.omega,0.)
            self.assertIsNone(e.result)
        self.assertEqual(transfers,[66.78])
        self.assertAlmostEqual(e.brake_episode_started,66.66)
        self.assertTrue(e.suspended_release);self.assertIs(e.request,request)
        self.assertEqual(e.committed_heading,heading)
        self.assertEqual(e.state,m.MovementState.BRAKE)

    def test_prolonged_overspeed_cannot_hide_recovered_posture_or_renew_braking(self):
        e=self.helper.pause()
        self.tick(e,.8,40.,.6)
        for t in np.arange(.9,9.01,.1):
            self.tick(e,float(t),35.,.6)
            if e.result:break
        self.assertIsNone(e.posture_pause)
        self.assertIn('unable to brake',e.result.reason)
        self.assertGreater(e.result.time,8.8);self.assertLess(e.result.time,9.01)

    def test_repeated_posture_entries_retain_brake_and_shared_uphill_anchors(self):
        e=self.helper.pause();self.tick(e,.8,40.,.6)
        e.uphill_stall_since=.5;e.uphill_stall_origin=(100.,0.)
        for t in np.arange(.9,8.91,.1):
            t=float(t)
            # Alternating observed tilt episodes interrupt the same brake.
            tilt=46. if int(round(t*10))%10<3 else 35.
            self.tick(e,t,tilt,.2,yaw_rate=.3)
            if e.result:break
            self.assertEqual(e.brake_episode_started,.8)
            self.assertEqual(e.uphill_stall_since,.5)
            self.assertEqual(e.uphill_stall_origin,(100.,0.))
        self.assertIsNotNone(e.result);self.assertIn('unable to brake',e.result.reason)
        self.assertGreaterEqual(sum(x['state']=='POSTURE_PAUSE' for x in e.history),3)

    def test_recovery_at_deadline_is_too_late(self):
        e=self.helper.pause();self.tick(e,.8,40.,.6)
        for t in (1.,1.2,1.4,1.6,1.8):self.tick(e,t,40.)
        self.tick(e,2.,35.)
        self.tick(e,2.2,35.)
        self.assertEqual(e.result.outcome,'censored')
        self.assertEqual(e.result.reason,'posture observation unresolved')

    def test_transfer_does_not_resume_nonrelease_drive_or_yaw(self):
        for intent in (m.MovementState.DRIVE_FORWARD,m.MovementState.ALIGN_LEFT):
            e=self.helper.pause();e.posture_pause.intent=intent
            self.tick(e,.8,40.,.6)
            self.tick(e,.9,35.,.2)
            self.tick(e,1.,35.,.2)
            cmd=self.tick(e,1.1,35.,.2)
            self.assertIsNone(e.posture_pause);self.assertFalse(e.suspended_release)
            self.assertEqual(e.state,m.MovementState.BRAKE)
            self.assertEqual(cmd.omega,0.);self.assertLessEqual(cmd.vx,0.)

    def test_shared_uphill_budget_still_expires_before_motion_resumption(self):
        e=self.helper.pause();self.tick(e,.8,40.,.6)
        e.uphill_stall_since=-23.;e.uphill_stall_origin=(100.,0.);e.uphill_exposure=.2
        self.tick(e,.9,35.,.2);self.tick(e,1.,35.,.2)
        self.assertTrue(e.result.physical)
        self.assertIn('shared low-progress',e.result.reason)


if __name__=='__main__':unittest.main()
