"""Purposeful signed recovery, immutable escape ownership, and real captures."""
from dataclasses import replace
import importlib
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from test_behavior_navigation import a,b,m,ground,observation
from rexmi_bt_tests.nav.recovery_intent import RecoveryPermission
from rexmi_bt_tests.nav.departure_diagnostics import load_departure


class PurposefulRecoveryTests(unittest.TestCase):
    def controller(self, grade=0., yaw=0., velocity=(.11,0.)):
        g=ground(grade)
        c=b.BehaviorTreeAscentController(g,[SimpleNamespace(x=3.,y=0.,label='goal')],mode='characterize')
        self.addCleanup(c.close)
        c.obs=observation(1.,yaw=yaw,velocity=velocity,grade=grade,turn=grade<=12.)
        for time in np.linspace(.7,1.,16):c.passive_heading.observe(time,yaw)
        g.mark_failure('first',(0.,0.),0.,.5,'physical fixture')
        c._begin_recovery('fixture')
        return c

    def test_forward_and_reverse_egress_preserve_intentional_ban(self):
        for yaw,sign in [(0.,-1),(math.pi,1)]:
            c=self.controller(yaw=yaw)
            c.executor.start(m.MovementRequest('escape',(0.,0.),(-1.,0.),sign,.4,100.,'recovery'),1.)
            self.assertEqual(c._guard(m.Motion(sign*.4,0.),record=False),'')
            self.assertTrue(c._guard(m.Motion(-sign*.4,0.),record=False))

    def test_braking_checks_actual_opposition_and_command_geometry(self):
        c=self.controller()
        self.assertEqual(c._brake_evaluation(c.obs,m.Motion(-.4,0.))['reason'],'')
        self.assertTrue(c._brake_evaluation(c.obs,m.Motion(.4,0.))['reason'])
        c.grid.hazard[c.grid.world_to_cell(-.4,0.)]=True
        c.grid.blocked[c.grid.world_to_cell(-.4,0.)]=True
        c.grid.obstacle_distance[:]=0.
        self.assertTrue(c._brake_evaluation(c.obs,m.Motion(-.4,0.))['reason'])

    def test_queries_do_not_retire_permission_or_change_evidence(self):
        c=self.controller();permission=c.recovery_permission
        arrays={name:getattr(c.grid,name).copy() for name in ('approach_blocked','failures','no_go')}
        for _ in range(4):
            self.assertEqual(permission.reason(c.grid,(0.,0.),(1.,0.)),'')
            self.assertEqual(c._guard(m.Motion(-.4,0.),record=False),'')
        self.assertIs(c.recovery_permission,permission)
        for name,values in arrays.items():np.testing.assert_array_equal(getattr(c.grid,name),values)
        self.assertFalse(c.evidence.completed)

    def test_observed_exit_is_irrevocable_and_heading_does_not_renew(self):
        c=self.controller();g=c.grid;p=c.recovery_permission
        self.assertTrue(p.patches)
        p=p.observe(g,(2.,0.));self.assertFalse(p.patches)
        p=p.observe(g,(0.,0.));self.assertTrue(p.reason(g,(0.,0.),(.2,0.)))
        # A newly excluded heading cannot inherit any original permission.
        heading=g.heading_bin(math.pi/2);g.approach_blocked[...,heading]=True
        self.assertTrue(c.recovery_permission.reason(g,(0.,0.),(0.,.2)))

    def test_new_overlapping_patch_and_mask_expansion_are_not_inherited(self):
        c=self.controller();g=c.grid;p=c.recovery_permission
        g.mark_failure('new',(.2,0.),0.,1.1,'new physical fixture')
        self.assertTrue(p.reason(g,(0.,0.),(.2,0.)))
        c=self.controller();g=c.grid;p=c.recovery_permission
        cell=g.world_to_cell(1.,0.);g.approach_blocked[*cell,g.heading_bin(0.)]=True
        self.assertTrue(p.reason(g,(0.,0.),(1.2,0.)))
        self.assertTrue(p.reason(g,(-1.,0.),(0.,0.)))

    def test_whole_patch_hazards_freshness_and_slope_remain_binding(self):
        for kind in ('whole','hazard','stale','steep'):
            c=self.controller(velocity=(.3,0.));g=c.grid;cell=g.world_to_cell(.2,0.)
            if kind=='whole':g.no_go[cell]=True;g.no_go_depth[cell]=1.
            elif kind=='hazard':g.hazard[cell]=True
            elif kind=='stale':g.fresh[cell]=False
            else:g.slope[cell]=math.tan(math.radians(36.))
            self.assertTrue(c._brake_evaluation(c.obs,m.Motion(-.4,0.))['reason'],kind)

    def test_stop_manual_fault_and_hold_retire_permission(self):
        for mode in ('stop','manual','fault','hold'):
            c=self.controller()
            if mode in ('stop','manual'):c.request(mode,1.)
            elif mode=='fault':c.fault='fixture';c._hard_fault(None)
            else:c.state='HOLD';c._terminal(None)
            self.assertIsNone(c.recovery_permission,mode)

    def test_same_episode_never_recaptures_or_restarts_deadline(self):
        c=self.controller();p=c.recovery_permission;c.recovery_trials.add(('tried',))
        c.obs=replace(c.obs,time=2.);c.grid.mark_failure('later',(.1,0.),0.,2.,'fixture')
        c._begin_recovery('again')
        self.assertIs(c.recovery_permission,p);self.assertEqual(c.recovery_since,1.)
        self.assertIn(('tried',),c.recovery_trials)

    def test_episode_expiry_retires_before_evaluating_brake(self):
        for active,cap in ((False,False),(True,False),(False,True)):
            c=self.controller()
            if active:c.executor.start(m.MovementRequest('escape',(0.,0.),(-1.,0.),-1,.4,200.,'recovery'),1.)
            if cap:c.recovery_actions=8
            else:c.obs=replace(c.obs,time=121.02)
            (c._active if active else c._recovery)(None)
            self.assertIsNone(c.recovery_permission)
            self.assertEqual(c.command,m.Motion())
            # Ordinary checked damping remains possible after retirement.
            c.grid.approach_blocked[:]=False;c.grid.failures[:]=0
            self.assertLess(c.executor.brake(c.obs).vx,0.)

    def test_both_signs_are_considered_despite_previous_direction(self):
        c=self.controller(velocity=(0.,0.));c.grid.approach_blocked[:]=False;c.grid.failures[:]=0
        c.recovery_direction=-1;c.recovery_route=[(0.,0.),(3.,0.)]
        with patch.object(c,'_recovery_purpose',side_effect=lambda end:(0,'checked mission connection',0.) if end[0]>0 else (1,'staging',0.)):
            c._recovery(None)
        self.assertEqual(c.executor.request.direction,1)
        self.assertEqual(c.recovery_since,1.)
        self.assertEqual(c.recovery_actions,1)

    def test_active_action_commits_without_reranking(self):
        c=self.controller(velocity=(0.,0.));c.grid.approach_blocked[:]=False;c.grid.failures[:]=0
        c._recovery(None);req=c.executor.request
        with patch.object(c,'_recovery',side_effect=AssertionError('must not rerank active action')):
            c.tree.tick(c)
        self.assertIs(c.executor.request,req)

    def test_recovery_prefers_connection_then_staging_then_egress(self):
        c=self.controller()
        with patch.object(c,'_recovery_connection',return_value=[(1.,0.),(3.,0.)]):
            self.assertEqual(c._recovery_purpose((1.,0.))[0],0)
        with patch.object(c,'_recovery_connection',return_value=None):
            self.assertEqual(c._recovery_purpose((1.,0.))[0],1)
            with patch.object(c.grid,'max_support_slope',return_value=20.):
                purpose=c._recovery_purpose((-1.,0.))
                self.assertEqual(purpose[0],2);self.assertIn('unproven',purpose[1])

    def test_steep_straight_continuation_is_checked_without_turn_permission(self):
        c=self.controller(grade=13.,velocity=(0.,0.));c.grid.approach_blocked[:]=False;c.grid.failures[:]=0
        c.recovery_route=[(0.,0.),(3.,0.)]
        self.assertIsNotNone(c._recovery_connection(c.obs.xy,0.,live=True))
        self.assertIsNone(c._recovery_connection(c.obs.xy,.3,live=True))
        c.passive_heading.samples=[]
        self.assertIsNone(c._recovery_connection(c.obs.xy,0.,live=True))

    def test_planning_rejection_has_no_failure_vote(self):
        c=self.controller();count=len(c.grid.failure_records)
        c._begin_recovery('planner rejected costly terrain')
        self.assertEqual(len(c.grid.failure_records),count)
        self.assertFalse(c.evidence.completed)


class CapturedRecoveryTests(unittest.TestCase):
    root=Path(__file__).resolve().parents[1]

    def captured(self,path):
        if not path.exists():self.skipTest('retained simulator capture unavailable')
        g,d=load_departure(path)
        c=b.BehaviorTreeAscentController(g,[SimpleNamespace(x=13.,y=0.,label='goal')],mode='characterize')
        self.addCleanup(c.close);c.obs=m.MotionObservation(**d['observation'])
        for t,yaw in d['passive_heading_samples']:c.passive_heading.observe(t,yaw)
        c._begin_recovery('recorded failure')
        c.executor.start(m.MovementRequest(**d['request']),c.obs.time)
        return c,d

    def test_trial35_exact_blocked_reverse_and_brake_now_pass(self):
        for name in ('run.event-03.npz','run.event-04.npz'):
            c,d=self.captured(self.root/'outputs/ascent_behavior_tree/trial35'/name)
            before=c.grid.approach_blocked.copy()
            self.assertEqual(c._guard(m.Motion(-.4,0.),record=False),'')
            self.assertEqual(c._brake_evaluation(c.obs)['reason'],'')
            np.testing.assert_array_equal(c.grid.approach_blocked,before)
            self.assertEqual(len(c.grid.failure_records),1)

    def test_gui_later_grade_and_passive_yaw_rejections_still_reject(self):
        c,d=self.captured(self.root/'outputs/ascent_user_check/20260926-142351/run.event-02.npz')
        self.assertTrue(c._guard(m.Motion(-.4,0.),record=False))
        offsets=c.passive_heading.offsets(c.obs.time,c.obs.yaw,2.)
        self.assertTrue(c._straight_entry_envelope(c.obs,c.executor.request,0.,offsets))
        self.assertEqual(c._brake_evaluation(c.obs)['reason'],'')


if __name__=='__main__':unittest.main()
