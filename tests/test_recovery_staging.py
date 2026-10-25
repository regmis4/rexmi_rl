"""Recovery ends at checked staging, without inventing checkpoint success."""
from dataclasses import replace
import math
import unittest

import numpy as np

from test_behavior_navigation import m, observation
import test_behavior_navigation as fixtures


class RecoveryStagingTests(unittest.TestCase):
    def executor(self, purpose='recovery'):
        e=m.MovementExecutor()
        e.recovery_stage_guard=e.brake_guard=lambda obs:''
        e.posture_guard=lambda *args:''
        e.start(m.MovementRequest('escape',(0.,0.),(-2.,0.),-1,.4,50.,purpose),0.)
        e.committed_heading=0.;e.travel_axis=(-1.,0.)
        e._state(m.MovementState.DRIVE_REVERSE,0.,'fixture')
        self.tick(e,observation(0.))
        return e

    def tick(self,e,obs):
        return e.tick(obs,lambda command:'',lambda xy:True)

    def entered(self):
        e=self.executor()
        self.tick(e,observation(1.,xy=(-.5,0.)))
        self.assertEqual(e.state,m.MovementState.RECOVERY_STAGE)
        return e

    def test_checked_staging_replans_far_from_endpoint_without_success_vote(self):
        e=self.entered();req=e.request
        for i in range(1,31):
            self.assertEqual(self.tick(e,observation(1.+i*.02,xy=(-.5,0.))),m.Motion())
        result=e.consume()
        self.assertEqual(result.outcome,'censored');self.assertFalse(result.physical)
        self.assertEqual(result.disposition,m.ResultDisposition.REPOSITIONED)
        self.assertGreater(math.dist(result.position,req.endpoint),1.)
        self.assertIsNone(e.consume())
        self.assertEqual(len([v for v in e.history if v['event']=='recovery_stage_entered']),1)
        self.assertEqual(len([v for v in e.history if v['event']=='recovery_stage_settled']),1)

    def test_mission_no_drive_evidence_and_small_displacement_do_not_handoff(self):
        for case in ('mission','no_drive','small','blocked','steep','tilted'):
            e=self.executor('mission' if case=='mission' else 'recovery')
            if case=='no_drive':e.pending_traversal=e.executed_traversal=None
            if case=='blocked':e.recovery_stage_guard=lambda obs:'blocked rotation footprint'
            obs=observation(1.,xy=(-.1 if case=='small' else -.5,0.),grade=13. if case=='steep' else 0.)
            if case=='tilted':obs=replace(obs,up_z=math.cos(math.radians(21.)))
            self.tick(e,obs)
            self.assertIsNone(e.recovery_stage_started,case)
            self.assertIsNone(e.result,case)

    def test_dwell_resets_for_motion_and_missing_samples_not_repeated_ticks(self):
        e=self.entered()
        for _ in range(100):self.tick(e,observation(1.,xy=(-.5,0.)))
        self.assertIsNone(e.result)
        for i in range(1,21):self.tick(e,observation(1.+i*.02,xy=(-.5,0.)))
        self.tick(e,observation(1.42,xy=(-.5,0.),velocity=(.13,0.)))
        self.assertFalse(e.recovery_stage_samples)
        for i in range(23,45):self.tick(e,observation(1.+i*.02,xy=(-.5,0.)))
        self.assertIsNone(e.result)
        self.tick(e,observation(2.5,xy=(-.5,0.)))
        self.assertEqual(len(e.recovery_stage_samples),1)
        self.assertIsNone(e.result)

    def test_six_second_clock_survives_overspeed_and_brake_state(self):
        e=self.entered();req=e.request
        for i in range(1,301):
            command=self.tick(e,observation(1.+i*.02,xy=(-.5,0.),velocity=(-.5,0.)))
            self.assertEqual(command.omega,0.)
            if i<300:self.assertIs(e.request,req)
        self.assertEqual(e.result.outcome,'censored')
        self.assertIn('observation deadline',e.result.reason)
        self.assertEqual(e.recovery_stage_started,1.)

    def test_existing_braking_deadline_is_not_renewed(self):
        e=self.executor();e._state(m.MovementState.BRAKE,0.,'fixture')
        e.brake_episode_started=-6.
        self.tick(e,observation(1.,xy=(-.5,0.)))
        self.assertEqual(e.state,m.MovementState.RECOVERY_STAGE)
        self.assertEqual(e.brake_episode_started,-6.)
        self.tick(e,observation(2.02,xy=(-.5,0.)))
        self.assertTrue(e.result.physical)
        self.assertIn('brake and settle',e.result.reason)

    def test_posture_owns_output_before_stage_eligibility_loss(self):
        e=self.entered();req=e.request
        obs=replace(observation(1.02,xy=(-.5,0.)),up_z=math.cos(math.radians(50.)))
        self.assertEqual(self.tick(e,obs),m.Motion())
        self.assertEqual(e.state,m.MovementState.POSTURE_PAUSE)
        self.assertIs(e.request,req);self.assertIsNotNone(e.posture_pause)
        self.assertEqual(e.recovery_stage_started,1.)
        self.assertFalse(e.recovery_stage_samples)
        # Recovery from the protective pause does not imply admission to a
        # <=20 degree staging envelope, nor reset its six-second deadline.
        for i in range(3,25):
            self.tick(e,replace(observation(1.+i*.02,xy=(-.5,0.)),up_z=math.cos(math.radians(21.))))
            if e.result:break
        self.assertEqual(e.result.outcome,'censored')
        self.assertIn('staging posture',e.result.reason)
        self.assertEqual(e.recovery_stage_started,1.)

    def test_rapid_posture_and_active_pause_keep_original_anchor(self):
        e=self.entered();anchor=e.recovery_stage_anchor
        e.posture=[(.42,0.)]
        obs=replace(observation(1.02,xy=(-.5,0.)),up_z=math.cos(math.radians(40.)))
        self.tick(e,obs)
        self.assertIsNotNone(e.posture_pause);pause=e.posture_pause
        self.tick(e,replace(obs,time=1.04))
        self.assertIs(e.posture_pause,pause);self.assertIsNone(e.result)
        self.assertEqual(e.recovery_stage_anchor,anchor)
        self.assertEqual(e.recovery_stage_started,1.)
        # A fixed stage deadline still bounds a late protective interruption.
        self.tick(e,replace(obs,time=7.))
        self.assertIn('observation deadline',e.result.reason)
        self.assertNotEqual(e.result.disposition,m.ResultDisposition.REPOSITIONED)

    def test_stage_loss_and_protections_cannot_become_success(self):
        for case in ('geometry','stale','tilt','tip','slope','anchor','cross','deadline','stop'):
            e=self.entered();obs=observation(1.02,xy=(-.5,0.))
            if case=='geometry':e.recovery_stage_guard=lambda obs:'new obstacle'
            if case=='stale':obs=replace(obs,fresh=False)
            if case=='tilt':obs=replace(obs,up_z=math.cos(math.radians(21.)))
            if case=='tip':obs=replace(obs,up_z=.4)
            if case=='slope':obs=replace(obs,inclination=13.)
            if case=='anchor':obs=replace(obs,xy=(-1.01,0.))
            if case=='cross':obs=replace(obs,xy=(-.5,.51))
            if case=='deadline':e.request=replace(e.request,deadline=1.01)
            if case=='stop':e.stop(obs)
            command=self.tick(e,obs)
            self.assertEqual(command.omega,0.,case)
            self.assertIsNotNone(e.result,case)
            self.assertNotEqual(e.result.disposition,m.ResultDisposition.REPOSITIONED,case)
            self.assertNotEqual(e.result.outcome,'success',case)

    def test_physical_backslide_precedes_stage_entry_for_both_drive_signs(self):
        for sign in (-1,1):
            for enabled in (False,True):
                e=m.MovementExecutor()
                e.recovery_stage_guard=(lambda obs:'') if enabled else None
                e.start(m.MovementRequest('escape',(0.,0.),(sign*2.,0.),sign,.4,50.,'recovery'),0.)
                e.committed_heading=0.;e.travel_axis=(sign,0.)
                e._state(m.MovementState.DRIVE_FORWARD if sign>0 else m.MovementState.DRIVE_REVERSE,0.,'fixture')
                for t,d,grade in ((0.,0.,13.),(1.,.8,13.),(2.,.7,13.),(3.02,.5,0.)):
                    self.tick(e,observation(t,xy=(sign*d,0.),grade=grade,turn=grade<=12.))
                self.assertTrue(e.result.physical)
                self.assertEqual(e.result.reason,'measured backslide against committed direction')
                self.assertIsNone(e.recovery_stage_started)

    def test_nan_posture_during_stage_cannot_fabricate_settle(self):
        e=self.entered()
        self.tick(e,replace(observation(1.02,xy=(-.5,0.)),up_z=math.nan))
        self.assertEqual(e.result.outcome,'censored')
        self.assertIn('not finite',e.result.reason)
        self.assertNotEqual(e.result.disposition,m.ResultDisposition.REPOSITIONED)


class StagingIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.flat=fixtures.ground()

    control=fixtures.IntegrationTests.control

    def test_rotation_perimeter_must_be_fresh_gentle_clear_and_exclusion_free(self):
        for case in ('fresh','known','slope','hazard','no_go','direction','clearance','nan_clearance','edge','stopping'):
            ctrl=self.control();ctrl.tick((0,0),0.,0.,0.)
            g=ctrl.grid;cells=g._support_cells((0,0));item=(cells[0][0],cells[1][0]);centre=g.world_to_cell(0,0)
            self.assertEqual(ctrl._recovery_staging_envelope(ctrl.obs),'')
            if case=='fresh':g.fresh[item]=False
            if case=='known':g.known[item]=False
            if case=='slope':g.slope[item]=math.tan(math.radians(12.01))
            if case=='hazard':g.hazard[item]=True
            if case=='no_go':g.no_go[item]=True
            if case=='direction':g.approach_blocked[item+(3,)]=True
            if case=='clearance':g.obstacle_distance[centre]=0.
            if case=='nan_clearance':g.obstacle_distance[centre]=math.nan
            if case=='edge':ctrl.obs=replace(ctrl.obs,xy=(5.99,0.))
            if case=='stopping':ctrl._brake_envelope=lambda obs:'stopping room blocked'
            self.assertTrue(ctrl._recovery_staging_envelope(ctrl.obs),case)

    def test_nonfinite_observations_never_admit_stage(self):
        ctrl=self.control();ctrl.tick((0,0),0.,0.,0.)
        for field,value in (('xy',(math.nan,0.)),('yaw',math.inf),('up_z',math.nan),
                            ('velocity',(math.nan,0.)),('yaw_rate',math.nan),('inclination',math.nan)):
            self.assertIn('not finite',ctrl._recovery_staging_envelope(replace(ctrl.obs,**{field:value})))

    def test_handoff_rechecks_changed_perimeter_and_keeps_episode(self):
        for case in ('fresh','direction'):
            ctrl=self.control();ctrl.tick((0,0),0.,0.,0.)
            ctrl.recovering=True;ctrl.recovery_since=.1;ctrl.recovery_actions=2
            ctrl.executor.result=m.MovementResult('escape','censored','checked recovery staging reached and settled',False,
                1.,(0.,0.),'RECOVERY_STAGE',m.ResultDisposition.REPOSITIONED)
            cells=ctrl.grid._support_cells((0.,0.));item=(cells[0][0],cells[1][0])
            if case=='fresh':ctrl.grid.fresh[item]=False
            else:ctrl.grid.approach_blocked[item+(3,)]=True
            ctrl.tick((0,0),0.,0.,1.02)
            self.assertTrue(ctrl.recovering);self.assertEqual(ctrl.recovery_since,.1)
            self.assertEqual(ctrl.recovery_actions,2);self.assertEqual(ctrl.wp_idx,0)
            self.assertEqual(ctrl.state,'RECOVER')
            self.assertEqual(sum(e.get('event')=='recovery_stage_handoff_rejected' for e in ctrl.escape_events),1)

    def test_reposition_retains_mission_goal_and_failure_evidence(self):
        ctrl=self.control();ctrl.tick((0,0),0.,0.,0.)
        ctrl.grid.mark_failure('old',(2.,0.),0.,0.,'physical fixture')
        failure=ctrl.grid.approach_blocked.copy();wp=ctrl.wp_idx;visited=list(ctrl.visited_checkpoints)
        ctrl.recovering=True;ctrl.recovery_since=0.;ctrl.recovery_actions=2
        ctrl.executor.result=m.MovementResult('escape','censored','checked recovery staging reached and settled',False,
            1.,(0.,0.),'RECOVERY_STAGE',m.ResultDisposition.REPOSITIONED)
        ctrl.tick((0,0),0.,0.,1.)
        self.assertFalse(ctrl.recovering);self.assertEqual(ctrl.state,'OBSERVE')
        self.assertEqual(ctrl.wp_idx,wp);self.assertEqual(ctrl.visited_checkpoints,visited)
        np.testing.assert_array_equal(ctrl.grid.approach_blocked,failure)
        self.assertIsNone(ctrl.executor.result)


if __name__=='__main__':unittest.main()
