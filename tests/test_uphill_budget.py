"""Finite uphill actuation opportunity, distinct from its absolute wall bound."""
from dataclasses import replace
import math
import unittest

from test_behavior_navigation import m, observation
from test_checkpoint_navigation import module

runner = module('budget_telemetry_runner','nav/checkpoint_runner.py')


class UphillBudgetTests(unittest.TestCase):
    def executor(self):
        e=m.MovementExecutor()
        e.release_guard=e.posture_guard=e.corridor_guard=lambda *args:''
        e.start(m.MovementRequest('one',(0.,0.),(10.,0.),1,.4,100.),0.)
        e.travel_axis=(1.,0.);e.committed_heading=0.;e.uphill_patience=True
        e._state(m.MovementState.DRIVE_FORWARD,0.,'fixture')
        e.uphill_stall_since=0.;e.uphill_stall_origin=(0.,0.)
        return e

    def obs(self,t,x=0.,**kw):
        return replace(observation(t,xy=(x,0.),grade=30.,turn=False,**kw),terrain_gradient=(.5,0.))

    def tick(self,e,obs):
        return e.tick(obs,lambda command:'',lambda xy:True)

    def test_only_emitted_intervals_count_once_and_clip_to_epoch(self):
        e=self.executor();e.uphill_stall_since=1.
        e._emit_traversal(m.Motion(.4,0.),self.obs(.9))
        self.assertEqual(e.uphill_exposure,0.)
        e._state(m.MovementState.BRAKE,1.,'interruption')
        e._observe_emitted_traversal(self.obs(1.2));e._observe_emitted_traversal(self.obs(1.2))
        self.assertAlmostEqual(e.uphill_exposure,.2)
        # Same-sign damping is still not deliberate drive.
        e.brake(self.obs(1.2,velocity=(-.1,0.)))
        e._observe_emitted_traversal(self.obs(1.4))
        self.assertAlmostEqual(e.uphill_exposure,.2)
        e._state(m.MovementState.RELEASE,1.4,'release');e._emit_traversal(m.Motion(),self.obs(1.4))
        e._state(m.MovementState.POSTURE_PAUSE,1.5,'pause')
        e._observe_emitted_traversal(self.obs(1.6))
        self.assertAlmostEqual(e.uphill_exposure,.4)
        e._observe_emitted_traversal(self.obs(1.8))
        self.assertAlmostEqual(e.uphill_exposure,.4)

    def test_admission_tick_is_not_drive_exposure(self):
        e=self.executor();e._state(m.MovementState.BRAKE,0.,'fixture')
        for t in (0.,.1,.2,.3):self.tick(e,self.obs(t))
        self.assertEqual(e.state,m.MovementState.DRIVE_FORWARD)
        self.assertEqual(e.uphill_exposure,0.)
        self.tick(e,self.obs(.4));self.assertEqual(e.uphill_exposure,0.)
        self.tick(e,self.obs(.5));self.assertAlmostEqual(e.uphill_exposure,.1)

    def test_twelve_exposed_seconds_expire_without_waiting_for_wall(self):
        e=self.executor()
        for i in range(601):
            self.tick(e,self.obs(i*.02))
            if e.result:break
        self.assertIsNotNone(e.result)
        self.assertAlmostEqual(e.result.time,12.)
        self.assertIn('exposure budget',e.result.reason);self.assertTrue(e.result.physical)
        self.assertAlmostEqual(e.uphill_exposure,12.)
        self.assertIsNotNone(e.consume());self.assertIsNone(e.consume())

    def test_interruptions_preserve_episode_until_absolute_wall_bound(self):
        e=self.executor();request=e.request;origin=e.uphill_stall_origin
        # Each second: 0.2 s emitted DRIVE followed by 0.8 s brake/posture.
        for i in range(24):
            e._state(m.MovementState.DRIVE_FORWARD,float(i),'resume')
            e._emit_traversal(m.Motion(.4,0.),self.obs(float(i)))
            e._observe_emitted_traversal(self.obs(i+.2))
            e._state(m.MovementState.BRAKE,i+.2,'interrupt')
            e._state(m.MovementState.POSTURE_PAUSE,i+.4,'observe')
            self.assertEqual(e._observe_uphill_progress(self.obs(i+.8)),'')
            self.assertEqual(e.uphill_stall_since,0.);self.assertEqual(e.uphill_stall_origin,origin)
            self.assertIs(e.request,request)
        reason=e._observe_uphill_progress(self.obs(24.))
        self.assertIn('wall budget',reason);self.assertAlmostEqual(e.uphill_exposure,4.8)
        e._finish_progress_failure(reason,self.obs(24.));self.assertTrue(e.result.physical)

    def test_wall_expiry_without_deliberate_exposure_is_censored(self):
        e=self.executor();e._state(m.MovementState.BRAKE,23.,'bounded brake')
        self.tick(e,self.obs(24.))
        self.assertEqual(e.result.outcome,'censored');self.assertFalse(e.result.physical)
        self.assertIn('wall budget',e.result.reason)

    def test_alignment_transition_cannot_renew_low_progress_episode(self):
        e=self.executor();e.uphill_exposure=3.
        for state in (m.MovementState.RELEASE,m.MovementState.BRAKE,m.MovementState.POSTURE_PAUSE,
                      m.MovementState.ALIGN_LEFT,m.MovementState.SETTLE):
            e._state(state,1.,'transition')
            self.assertEqual(e.uphill_stall_since,0.);self.assertEqual(e.uphill_exposure,3.)

    def test_only_net_ten_centimetres_renews_both_budgets(self):
        e=self.executor();e.uphill_exposure=6.
        e._observe_uphill_progress(self.obs(15.,.099))
        self.assertEqual(e.uphill_stall_since,0.);self.assertEqual(e.uphill_exposure,6.)
        e._observe_uphill_progress(self.obs(15.02,.101))
        self.assertIsNone(e.uphill_stall_since);self.assertEqual(e.uphill_exposure,0.)
        self.assertEqual(e.metrics['uphill_wall_remaining_s'],24.)
        self.assertEqual(e.metrics['uphill_budget_remaining_s'],12.)
        self.assertEqual(len([x for x in e.history if x['event']=='uphill_budget_renewed']),1)

    def test_stop_tip_stale_backslide_and_deadline_keep_priority(self):
        for case in ('stop','tip','stale','backslide','deadline'):
            e=self.executor();obs=self.obs(24.)
            if case=='stop':e.stop(obs);self.assertEqual(e.result.reason,'operator stop');continue
            if case=='tip':obs=replace(obs,up_z=.4)
            if case=='stale':obs=replace(obs,fresh=False)
            if case=='backslide':e.uphill_samples=[(22.,(.4,0.))]
            if case=='deadline':e.request=replace(e.request,deadline=23.)
            self.tick(e,obs)
            self.assertNotIn('shared low-progress',e.result.reason,case)
            self.assertEqual(e.result.physical,case!='stale')

    def test_overspeed_and_posture_still_interrupt_before_more_exposure(self):
        e=self.executor()
        self.tick(e,self.obs(.02,velocity=(.6,0.)))
        self.assertEqual(e.state,m.MovementState.BRAKE);self.assertEqual(e.uphill_exposure,0.)
        e=self.executor();e.posture=[(0.,math.radians(20.))]
        self.tick(e,replace(self.obs(.6),up_z=math.cos(math.radians(40.))))
        self.assertEqual(e.state,m.MovementState.POSTURE_PAUSE);self.assertEqual(e.uphill_exposure,0.)


class SharedGoalBudgetTests(UphillBudgetTests):
    def executor(self):
        e=super().executor();e.release_departure_guard=lambda *args:''
        return e

    def test_axis_only_gain_cannot_renew_goal_stagnation(self):
        e=self.executor();e.uphill_exposure=6.
        e._observe_uphill_progress(replace(self.obs(15.,.2),xy=(.2,3.)))
        self.assertEqual(e.uphill_stall_since,0.);self.assertEqual(e.uphill_exposure,6.)
        e._observe_uphill_progress(self.obs(15.02,.3))
        self.assertIsNone(e.uphill_stall_since)

    def test_goal_only_exhaustion_is_censored(self):
        e=self.executor();e.uphill_exposure=12.;e.uphill_goal_only_budget=True
        e.uphill_samples=[(22.,(0.,0.))]
        obs=replace(self.obs(24.,.2),xy=(.2,3.))
        reason=e._observe_uphill_progress(obs)
        self.assertIn('budget',reason)
        e._finish_progress_failure(reason,obs)
        self.assertFalse(e.result.physical);self.assertEqual(e.result.outcome,'censored')

    def test_goal_budget_cause_survives_late_axis_slowdown(self):
        e=self.executor();e.uphill_goal_only_budget=True;e.uphill_exposure=12.
        e.uphill_samples=[(22.,(0.,0.))]
        obs=self.obs(24.)
        reason=e._observe_uphill_progress(obs)
        e._finish_progress_failure(reason,obs)
        self.assertFalse(e.result.physical)

    def test_axis_recovered_goal_stalled_budget_is_censored(self):
        e=self.executor();e.uphill_exposure=12.
        e.uphill_samples=[(22.,(0.,0.))]
        obs=replace(self.obs(24.,.2),xy=(.2,3.))
        reason=e._observe_uphill_progress(obs)
        self.assertTrue(e.uphill_goal_only_budget)
        e._finish_progress_failure(reason,obs)
        self.assertFalse(e.result.physical)

    def test_goal_deficit_starts_existing_budget(self):
        e=self.executor();e.uphill_stall_since=e.uphill_stall_origin=None
        e.uphill_samples=[(0.,(0.,0.))]
        e._observe_uphill_progress(replace(self.obs(2.,.2),xy=(.2,3.)))
        self.assertEqual(e.uphill_stall_since,2.)


class PassivePostureTelemetryTests(unittest.TestCase):
    def test_normal_is_the_contemporaneous_observed_fit(self):
        obs=replace(observation(0.),terrain_gradient=(.5,.2))
        values=runner.posture_telemetry((1.,0.,0.,0.),(.1,.2,.3),obs)
        self.assertEqual(len(values),len(runner.POSTURE_TELEMETRY_COLUMNS))
        self.assertEqual(values[:9],[1.,0.,0.,0.,.1,.2,.3,.5,.2])
        self.assertTrue(values[-1]);self.assertAlmostEqual(sum(v*v for v in values[9:12]),1.)
        self.assertAlmostEqual(values[9]/values[11],-.5)

    def test_stale_missing_and_nonfinite_fits_are_explicitly_invalid(self):
        for obs in (None,observation(0.),replace(observation(0.),fresh=False,terrain_gradient=(.5,0.)),
                    replace(observation(0.),terrain_gradient=(math.nan,0.)),
                    replace(observation(0.),terrain_gradient=(1.79e308,1.79e308))):
            values=runner.posture_telemetry((1.,0.,0.,0.),(0.,0.,0.),obs)
            self.assertFalse(values[-1]);self.assertTrue(all(math.isnan(v) for v in values[7:12]))


if __name__=='__main__':unittest.main()
