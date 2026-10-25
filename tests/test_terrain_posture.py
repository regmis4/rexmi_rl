"""Provisional posture entry envelope; hard protections remain independent."""
from dataclasses import replace
import json
import math
from pathlib import Path
import unittest

from test_behavior_navigation import m, observation


def slope_observation(t,body=40.,terrain=30.,yaw=0.):
    cy,sy=math.cos(yaw/2),math.sin(yaw/2)
    cp,sp=math.cos(math.radians(-body)/2),math.sin(math.radians(-body)/2)
    q=(cy*cp,-sy*sp,cy*sp,sy*cp)
    return replace(observation(t,yaw=yaw,grade=terrain,turn=False),
        up_z=math.cos(math.radians(body)),body_quaternion=q,
        terrain_gradient=(math.tan(math.radians(terrain))*math.cos(yaw),
                          math.tan(math.radians(terrain))*math.sin(yaw)))


class TerrainPostureTests(unittest.TestCase):
    def executor(self,phase=m.MovementState.DRIVE_FORWARD):
        e=m.MovementExecutor();e.release_guard=e.posture_guard=e.corridor_guard=lambda *args:''
        e.start(m.MovementRequest('one',(0.,0.),(10.,0.),1,.4,100.),0.)
        e.committed_heading=0.;e.travel_axis=(1.,0.);e.release_origin=(0.,0.)
        e._state(phase,0.,'fixture')
        e._emit_traversal(m.Motion(.4 if phase==m.MovementState.DRIVE_FORWARD else 0.,0.),slope_observation(0.))
        e._observe_emitted_traversal(slope_observation(.02))
        e.posture=[(0.,math.radians(20.))]
        return e

    def tick(self,e,obs,guard=None):
        return e.tick(obs,guard or (lambda command:''),lambda xy:True)

    def test_world_frame_and_quaternion_sign(self):
        for yaw in (0.,math.pi/2,-math.pi/2,math.pi):
            obs=slope_observation(.6,40.,30.,yaw)
            self.assertAlmostEqual(obs.terrain_relative_tilt,10.)
            self.assertAlmostEqual(replace(obs,body_quaternion=tuple(-v for v in obs.body_quaternion)).terrain_relative_tilt,10.)
        # Same absolute tilt in the opposite terrain direction is 70°, not 10°.
        obs=slope_observation(.6)
        self.assertAlmostEqual(replace(obs,terrain_gradient=(-math.tan(math.radians(30)),0.)).terrain_relative_tilt,70.)

    def test_invalid_stale_or_inconsistent_inputs_never_exempt(self):
        obs=slope_observation(.6)
        for changed in (dict(body_quaternion=None),dict(body_quaternion=(0.,0.,0.,0.)),
                dict(body_quaternion=(math.nan,0.,0.,1.)),dict(body_quaternion=(1.,0.,0.)),
                dict(body_quaternion=tuple(v*1.1 for v in obs.body_quaternion)),
                dict(terrain_gradient=None),dict(terrain_gradient=(math.nan,0.)),
                dict(terrain_gradient=(1.79e308,1.79e308)),
                dict(fresh=False),dict(up_z=.9),dict(yaw=.5)):
            invalid=replace(obs,**changed)
            self.assertIsNone(invalid.terrain_relative_tilt,changed)
            self.assertFalse(self.executor()._slope_posture_exempt(invalid,invalid.terrain_relative_tilt),changed)

    def test_exact_envelope_boundaries(self):
        e=self.executor()
        for body,terrain,expected in ((40.,20.,True),(40.0001,20.,False),(49.999,30.,True),(50.,30.,False),(50.001,31.,False)):
            obs=slope_observation(.6,body,terrain)
            self.assertEqual(e._slope_posture_exempt(obs,obs.terrain_relative_tilt),expected,(body,terrain))

    def test_established_forward_and_release_only(self):
        obs=slope_observation(.6)
        for phase in m.MovementState:
            e=self.executor();e._state(phase,.1,'fixture')
            expected=phase in (m.MovementState.DRIVE_FORWARD,m.MovementState.RELEASE)
            # Real alignment clears old execution evidence.
            self.assertEqual(e._slope_posture_exempt(obs,obs.terrain_relative_tilt),expected,phase)
        for case in ('reverse','recovery','fresh','other_axis','unobserved','previous_request'):
            e=self.executor()
            if case=='reverse':e.request=replace(e.request,direction=-1)
            if case=='recovery':e.request=replace(e.request,purpose='recovery')
            if case=='fresh':e.executed_traversal=None
            if case=='other_axis':e.executed_traversal=replace(e.executed_traversal,heading=.2)
            if case=='unobserved':e.executed_traversal=replace(e.executed_traversal,duration_s=0.)
            if case=='previous_request':
                e.finish('censored','fixture',obs);e.consume()
                e.start(m.MovementRequest('two',(0.,0.),(10.,0.),1,.4,100.),.7)
            self.assertFalse(e._slope_posture_exempt(obs,obs.terrain_relative_tilt),case)

    def test_fast_outward_change_keeps_raw_signal_and_checked_drive(self):
        e=self.executor();request=e.request;axis=e.travel_axis
        # 15° additional body lean in 0.10 s, within the provisional envelope.
        e.posture=[(0.,math.radians(20.)),(.5,math.radians(25.))]
        cmd=self.tick(e,slope_observation(.6,40.))
        self.assertEqual(cmd,m.Motion(.4,0.));self.assertTrue(e.metrics['posture_raw_rapid'])
        self.assertTrue(e.metrics['posture_entry_exempt']);self.assertIsNone(e.posture_pause)
        self.assertIs(e.request,request);self.assertEqual(e.travel_axis,axis)
        event=next(x for x in e.history if x.get('event')=='posture_entry_signal')
        self.assertTrue(event['raw_rapid']);self.assertTrue(event['exempt'])

    def test_fifty_degree_pause_is_independent_of_rate_or_normal(self):
        for qvalid in (True,False):
            e=self.executor();e.posture=[]
            obs=slope_observation(.6,50.,30.)
            if not qvalid:obs=replace(obs,body_quaternion=None)
            self.assertEqual(self.tick(e,obs),m.Motion())
            self.assertEqual(e.state,m.MovementState.POSTURE_PAUSE)
            self.assertFalse(e.metrics['posture_raw_rapid']);self.assertTrue(e.metrics['posture_absolute_pause'])
            self.assertTrue(any('absolute posture pause' in x.get('reason','') for x in e.history))

    def test_active_pause_is_not_cancelled_by_compatible_normal(self):
        e=self.executor();e.posture_pause=m.PosturePause(.1,m.MovementState.DRIVE_FORWARD)
        e._state(m.MovementState.POSTURE_PAUSE,.1,'fixture');pause=e.posture_pause
        self.assertEqual(self.tick(e,slope_observation(.6,40.)),m.Motion())
        self.assertIs(e.posture_pause,pause);self.assertEqual(pause.started,.1)
        self.assertFalse(e.metrics['posture_entry_exempt'])
        # Recovery still needs absolute <=36.87° and the entire 0.20 s dwell.
        for t in (.7,.8):self.tick(e,slope_observation(t,35.));self.assertIs(e.posture_pause,pause)
        self.tick(e,slope_observation(.9,35.));self.assertIsNone(e.posture_pause)
        self.assertEqual(e.state,m.MovementState.BRAKE)

    def test_raw_deterioration_still_classifies_active_pause_timeout(self):
        e=self.executor();e.posture_pause=m.PosturePause(0.,m.MovementState.DRIVE_FORWARD)
        e._state(m.MovementState.POSTURE_PAUSE,0.,'fixture');e.posture=[(1.,math.radians(20.))]
        self.tick(e,slope_observation(1.5,40.))
        self.assertTrue(e.result.physical);self.assertIn('deterioration',e.result.reason)

    def test_tip_overspeed_backslide_stale_geometry_deadline_and_stop(self):
        for case in ('tip','overspeed','backslide','stale','geometry','deadline','budget','stop'):
            e=self.executor();obs=slope_observation(.6,40.)
            if case=='stop':e.stop(obs);self.assertEqual(e.state,m.MovementState.STOP);continue
            if case=='tip':obs=slope_observation(.6,60.1)
            if case=='overspeed':obs=replace(obs,velocity=(.6,0.))
            if case=='backslide':e.uphill_samples=[(-1.,(.5,0.))]
            if case=='stale':obs=replace(obs,fresh=False)
            if case=='deadline':e.request=replace(e.request,deadline=.5)
            if case=='budget':e.uphill_stall_since=-24.;e.uphill_stall_origin=(10.,0.);e.uphill_exposure=1.
            cmd=self.tick(e,obs,(lambda cmd:'geometry changed') if case=='geometry' else None)
            if case=='overspeed':self.assertEqual(e.state,m.MovementState.BRAKE);self.assertLess(cmd.vx,0.)
            else:self.assertIsNotNone(e.result,case);self.assertNotEqual(e.result.outcome,'success')

    def test_measured_productive_and_adverse_release_observation_replays(self):
        fixture=json.loads((Path(__file__).parent/'fixtures/terrain_posture_response.json').read_text())
        for case in fixture['cases']:
            obs=[m.MotionObservation(**row) for row in case['observations']]
            e=m.MovementExecutor();e.release_guard=e.posture_guard=e.corridor_guard=lambda *args:''
            heading=case['committed_heading'];axis=(math.cos(heading),math.sin(heading))
            start=tuple(obs[0].xy);end=tuple(p+10*d for p,d in zip(start,axis))
            e.start(m.MovementRequest(case['name'],start,end,1,.4,100.,lateral_tolerance=1.),0.)
            e.committed_heading=heading;e.travel_axis=axis;e.release_origin=start
            e._state(m.MovementState.RELEASE,0.,'partial measured response fixture')
            e.executed_traversal=m.ExecutedTraversal('RELEASE',1,heading,obs[0].yaw,start,start,-.02,0.,.02)
            for o in obs:
                self.tick(e,o)
                if e.result:break
            signals=[x for x in e.history if x.get('event')=='posture_entry_signal']
            if case['name']=='productive_release':
                self.assertTrue(any(x['raw_rapid'] and x['exempt'] for x in signals))
                self.assertFalse(any(x.get('state')=='POSTURE_PAUSE' for x in e.history))
            else:
                self.assertTrue(any(x['absolute_pause'] and not x['exempt'] for x in signals))
                self.assertTrue(any(x.get('state')=='POSTURE_PAUSE' for x in e.history))


if __name__=='__main__':unittest.main()
