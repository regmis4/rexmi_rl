"""A brief policy posture response is not a completed terrain failure."""
from dataclasses import replace
import json
import math
from pathlib import Path
import unittest

import numpy as np

from types import SimpleNamespace

from test_behavior_navigation import b, m, ground, observation


class PostureObservationTests(unittest.TestCase):
    def setup_release(self):
        e = m.MovementExecutor()
        e.release_guard = e.posture_guard = lambda *args: ''
        e.start(m.MovementRequest('one', (0.,0.), (10.,0.), 1, .4, 100.), 0.)
        e.committed_heading = 0.; e.travel_axis = (1.,0.)
        e.release_origin = (0.,0.)
        e._state(m.MovementState.RELEASE, 0., 'fixture')
        for t in np.arange(0., .61, .1): self.tick(e, self.obs(t))
        return e

    def obs(self, t, tilt=25., **kwargs):
        return replace(observation(t, xy=(.1*t,0.), grade=30., turn=False, **kwargs),
                       up_z=math.cos(math.radians(tilt)), terrain_gradient=(.5,0.))

    def tick(self, e, obs, supported=True):
        return e.tick(obs, lambda cmd: '', lambda xy: supported)

    def pause(self):
        e = self.setup_release()
        self.assertEqual(self.tick(e, self.obs(.7, 40.)), m.Motion())
        self.assertEqual(e.state, m.MovementState.POSTURE_PAUSE)
        return e

    def test_recovery_requires_absolute_tilt_and_continuous_dwell(self):
        e = self.pause(); request = e.request
        for t in (.8, 1., 1.2): self.tick(e, self.obs(t, 40.))
        self.assertEqual(e.state, m.MovementState.POSTURE_PAUSE)
        for t in (1.3, 1.4): self.tick(e, self.obs(t, 35.))
        self.assertEqual(e.state, m.MovementState.POSTURE_PAUSE)
        self.assertEqual(self.tick(e, self.obs(1.5, 35.)), m.Motion())
        self.assertEqual(e.state, m.MovementState.RELEASE)
        self.assertIs(e.request, request)
        self.assertIsNone(e.result)

    def test_stable_high_timeout_is_censored_without_a_physical_failure_vote(self):
        e = self.pause()
        for t in np.arange(.8, 2.31, .1): self.tick(e, self.obs(t, 40.))
        self.assertEqual(e.result.outcome, 'censored')
        self.assertFalse(e.result.physical)
        self.assertEqual(e.result.reason, 'posture observation unresolved')

    def test_continuing_measured_deterioration_is_physical(self):
        e = self.pause()
        for t in np.arange(.8, 2.31, .1):
            self.tick(e, self.obs(t, 40.+max(0., t-1.6)*20.))
        self.assertTrue(e.result.physical)

    def test_overspeed_brakes_without_renewing_posture_deadline(self):
        e = self.pause(); started = e.posture_pause.started
        self.assertEqual(self.tick(e, self.obs(.8, 45., velocity=(.6,0.))), m.Motion(-.4,0.))
        self.assertEqual(e.state, m.MovementState.BRAKE)
        self.tick(e, self.obs(.9, 40.))
        self.assertEqual(e.state, m.MovementState.BRAKE)
        self.assertEqual(e.posture_pause.started, started)
        for t in np.arange(1., 2.31, .1): self.tick(e, self.obs(t, 40., velocity=(.6,0.)))
        self.assertEqual(e.result.outcome, 'censored')
        self.assertLess(e.result.time, 2.4)

    def test_overspeed_observation_resets_absolute_posture_recovery_dwell(self):
        e = self.pause()
        self.tick(e, self.obs(.8, 35.))
        self.tick(e, self.obs(.9, 45., velocity=(.6,0.)))
        self.tick(e, self.obs(1., 35.))
        self.assertNotEqual(e.state, m.MovementState.RELEASE)
        self.assertEqual(e.posture_pause.stable_since, 1.)

    def test_posture_recovery_cannot_bypass_measured_brake_settling(self):
        for entered_while_braking in (False, True):
            e = self.setup_release()
            if entered_while_braking: self.tick(e, self.obs(.65, 25., velocity=(.6,0.)))
            self.tick(e, self.obs(.7, 40., velocity=(.6,0.)))
            for t in (.8,.9,1.,1.1): self.tick(e, self.obs(t,35.,velocity=(.4,0.),yaw_rate=.5))
            self.assertEqual(e.state, m.MovementState.BRAKE)
            self.assertIsNone(e.posture_pause)
            self.assertTrue(e.suspended_release)
            for t in (1.2,1.3,1.4,1.5): command = self.tick(e, self.obs(t,35.))
            self.assertEqual(e.state, m.MovementState.RELEASE)
            self.assertEqual(command, m.Motion())

    def test_late_recovery_cannot_extend_absolute_episode_deadline(self):
        e = self.pause()
        for t in (.8,1.,1.2,1.4,1.6,1.8,2.): self.tick(e,self.obs(t,40.))
        self.tick(e,self.obs(2.1,35.))
        self.tick(e,self.obs(2.31,35.))
        self.assertEqual(e.result.outcome, 'censored')
        self.assertEqual(e.result.reason, 'posture observation unresolved')

    def test_route_exclusion_does_not_establish_physical_support_failure(self):
        e = self.pause()
        self.tick(e, self.obs(.8,40.), supported=False)
        self.assertIsNone(e.result)
        e.posture_guard = lambda *args: 'support observation unavailable'
        self.tick(e, self.obs(.9,40.), supported=False)
        self.assertFalse(e.result.physical)
        self.assertEqual(e.result.outcome, 'censored')

    def test_occupied_map_exclusion_allows_checked_posture_observation(self):
        g = ground()
        g.mark_failure('first',(0.,0.),0.,0.,'fixture')
        g.mark_failure('second',(0.,0.),math.pi,0.,'fixture')
        c = b.BehaviorTreeAscentController(g,[SimpleNamespace(x=3.,y=0.,label='goal')],mode='characterize')
        e = c.executor
        e.start(m.MovementRequest('escape',(0.,0.),(2.,0.),1,.4,100.),0.)
        e.committed_heading=0.;e.travel_axis=(1.,0.);e.release_origin=(0.,0.)
        e._state(m.MovementState.RELEASE,0.,'fixture')
        self.assertFalse(c._supported((0.,0.)))
        self.assertEqual(g.support_reason((0.,0.)), '')
        for t,tilt in ((0.,25.),(.2,25.),(.4,25.),(.6,40.)):
            c.obs=replace(self.obs(t,tilt),xy=(0.,0.))
            self.assertEqual(e.tick(c.obs,c._guard,c._supported),m.Motion())
        self.assertIsNone(e.result)
        self.assertEqual(e.state,m.MovementState.POSTURE_PAUSE)
        self.assertEqual(len(g.failure_records),2)

    def test_stop_tip_stale_geometry_and_deadline_are_preserved(self):
        for case in ('stop', 'tip', 'stale', 'geometry', 'deadline'):
            with self.subTest(case=case):
                e = self.pause()
                if case == 'stop':
                    self.assertEqual(e.stop(self.obs(.8)), m.Motion())
                    self.assertEqual(e.state, m.MovementState.STOP)
                elif case == 'tip':
                    self.tick(e, self.obs(.8, 65.)); self.assertTrue(e.result.physical)
                elif case == 'stale':
                    self.tick(e, self.obs(.8, fresh=False)); self.assertFalse(e.result.physical)
                elif case == 'geometry':
                    e.posture_guard = lambda *args: 'footprint changed'
                    self.tick(e, self.obs(.8)); self.assertEqual(e.result.outcome, 'censored')
                else:
                    self.tick(e, self.obs(101.)); self.assertEqual(e.result.reason, 'primitive deadline elapsed')
                self.assertIsNone(e.posture_pause)

    def test_supported_arrival_waits_for_posture_then_settles(self):
        e = self.pause()
        self.tick(e, replace(self.obs(.8, 40.), xy=(9.95,0.)))
        self.assertEqual(e.state, m.MovementState.POSTURE_PAUSE)
        for t in (.9, 1., 1.1): self.tick(e, replace(self.obs(t, 35.), xy=(9.95,0.)))
        self.assertEqual(e.state, m.MovementState.SETTLE)
        self.assertIsNone(e.result)

    def test_interrupted_yaw_cannot_resume_directly(self):
        e = self.setup_release()
        e._state(m.MovementState.ALIGN_LEFT, .6, 'fixture')
        self.tick(e, self.obs(.7, 40.))
        for t in (.8, .9, 1.): command = self.tick(e, self.obs(t, 35.))
        self.assertEqual(e.state, m.MovementState.BRAKE)
        self.assertEqual(command.omega, 0.)

    def test_successful_manual_spike_keeps_zero_and_recovers(self):
        fixture = json.loads((Path(__file__).parent/'fixtures/posture_response.json').read_text())
        samples = fixture['windows']['successful_zero']; first = samples[0]
        start = (first['x'],first['y']); yaw = first['yaw_rad']
        axis = (math.cos(yaw),math.sin(yaw))
        e = m.MovementExecutor(); e.release_guard = e.posture_guard = lambda *args: ''
        e.start(m.MovementRequest('replay', start, tuple(start[i]+10.*axis[i] for i in (0,1)),
                                 1,.4,first['sim_time_s']+100.),first['sim_time_s'])
        e.committed_heading = yaw; e.travel_axis = axis; e.release_origin = start
        e._state(m.MovementState.RELEASE, first['sim_time_s'], 'open-loop fixture')
        states = []
        for sample in samples:
            obs = m.MotionObservation((sample['x'],sample['y']),sample['yaw_rad'],
                (sample['root_linear_velocity_w_m_s.x'],sample['root_linear_velocity_w_m_s.y']),
                math.cos(math.radians(sample['body_tilt_deg'])),0.,sample['sim_time_s'],30.,False,True,
                tuple(.5*v for v in axis))
            self.assertEqual(self.tick(e,obs),m.Motion())
            self.assertIsNone(e.result)
            states.append(e.state)
        self.assertIn(m.MovementState.POSTURE_PAUSE,states)
        self.assertEqual(e.state,m.MovementState.RELEASE)

    def test_severe_manual_observation_does_not_resume_when_rise_signal_clears(self):
        fixture = json.loads((Path(__file__).parent/'fixtures/posture_response.json').read_text())
        e = self.setup_release()
        # Replay only measured tilt here, holding other geometry constant.
        # The human changed commands in this window; it is not a prediction
        # of what an earlier autonomous zero command would physically do.
        samples = fixture['windows']['severe_event']
        origin = samples[0]['sim_time_s']
        for sample in samples:
            if sample['sim_time_s'] > 130.29: break
            last_observation = self.obs(1.+sample['sim_time_s']-origin, sample['body_tilt_deg'])
            self.tick(e, last_observation)
        self.assertGreater(last_observation.body_tilt, 50.)
        self.assertEqual(e.state, m.MovementState.POSTURE_PAUSE)
        self.assertIsNone(e.result)


if __name__ == '__main__': unittest.main()
