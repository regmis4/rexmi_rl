"""Recorded geometry and synthetic state transitions; neither proves traction."""
import copy
import importlib
import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
from test_behavior_navigation import a, b, m, observation
from rexmi_bt_tests.nav.departure_diagnostics import MAP_FIELDS
from rexmi_bt_tests.nav.recovery_intent import RecoveryPermission
from rexmi_bt_tests.nav.recovery_turn import RecoveryTurn, rotation_reason


def recorded_turn():
    with np.load(Path(__file__).parent/'fixtures/recovery_turn_35814.npz', allow_pickle=False) as data:
        meta = json.loads(data['metadata_json'].item())
        cfg = meta['grid_config']
        g = a.AscentTerrainMap(**{k:cfg[k] for k in ('world_size','cell_size','origin',
            'footprint_radius','clearance','half_length','half_width')})
        region = tuple(slice(i, i+meta['fixture_width']) for i in meta['first_cell'])
        for name in MAP_FIELDS:
            getattr(g,name)[region] = data[name]
    for k,v in cfg.items():setattr(g,k,v)
    g.now = meta['map_time']; g.failure_records = copy.deepcopy(meta['failure_records'])
    ev = meta['postfailure_event']
    g.mark_failure(ev['attempt_id'],ev['position'],ev['heading'],ev['time'],ev['reason'])
    c = b.BehaviorTreeAscentController(g,[SimpleNamespace(x=0.,y=13.)],mode='characterize')
    c.obs = m.MotionObservation(**meta['observation'])
    c.passive_heading.samples = copy.deepcopy(meta['passive_heading_samples'])
    c.recovering = True; c.recovery_since = meta['recovery']['recovery_since']
    c.recovery_permission = RecoveryPermission(c.recovery_since, ())
    c.supported_trail = meta['recovery']['supported_trail']
    c.recovery_route = meta['recovery']['recovery_route']
    c.recovery_request_id = meta['request']['action_id']
    result = copy.deepcopy(meta['returned_result'])
    result['executed_traversal'] = m.ExecutedTraversal(**result['executed_traversal'])
    c.recorded_failure = m.MovementResult(**result)
    return c


class RecordedTurnTests(unittest.TestCase):
    def test_exact_case_selects_smallest_useful_right_turn(self):
        c = recorded_turn()
        candidates = c._recovery_turn_candidates()
        self.assertTrue(candidates)
        selected = min(candidates,key=lambda v:v[0]); intent = selected[-1]
        self.assertAlmostEqual(math.degrees(intent.target_yaw-intent.initial_yaw),-20.)
        self.assertEqual(intent.direction,1)
        self.assertIn('previously supported',selected[-2])
        self.assertGreater(c.grid.max_support_slope(intent.endpoint),15.)
        self.assertEqual(c.recovery_permission.patches,())
        self.assertTrue(all(v[1] == 1 for v in candidates))

    def test_hidden_hazard_stale_or_unobserved_rotation_blocks(self):
        for field,value in [('hazard',True),('fresh',False),('known',False),('no_go',True)]:
            c = recorded_turn(); cell = c.grid.world_to_cell(*c.obs.xy)
            getattr(c.grid,field)[cell] = value
            self.assertFalse(c._recovery_turn_candidates(),field)

    def test_rotation_bounds_and_deadline_are_frozen(self):
        c = recorded_turn(); intent = min(c._recovery_turn_candidates(),key=lambda v:v[0])[-1]
        for obs in (replace(c.obs,xy=(c.obs.xy[0]+.251,c.obs.xy[1])),
                    replace(c.obs,yaw=c.obs.yaw-math.radians(23.)),
                    replace(c.obs,yaw=c.obs.yaw-math.radians(31.)),
                    replace(c.obs,time=intent.deadline+.01)):
            self.assertTrue(rotation_reason(c.grid,obs,intent))

    def test_new_reverse_failure_is_not_inherited_for_drift(self):
        c = recorded_turn(); intent = min(c._recovery_turn_candidates(),key=lambda v:v[0])[-1]
        c.executor.start(m.MovementRequest('turn',c.obs.xy,intent.endpoint,1,.4,c.obs.time+30.,'recovery'),c.obs.time)
        c.executor.recovery_turn = replace(intent,action_id='turn')
        self.assertIs(c._directional_exit_context(m.Motion(0.,-.35)),c.recovery_permission)
        end = tuple(c.obs.xy[i]-.1*v for i,v in enumerate((math.cos(c.obs.yaw),math.sin(c.obs.yaw))))
        self.assertTrue(c.recovery_permission.reason(c.grid,c.obs.xy,end))
        self.assertEqual(c._directional_exit_context(),False)

    def test_ordinary_steep_yaw_stays_rejected(self):
        c = recorded_turn()
        self.assertEqual(c._guard(m.Motion(0.,-.35),record=False),'turn requires gentle staging')

    def test_arrival_drift_cannot_bypass_real_rotation_region_guard(self):
        c=recorded_turn(); intent=min(c._recovery_turn_candidates(),key=lambda v:v[0])[-1]
        o=c.obs; end=intent.endpoint
        c.executor.start(m.MovementRequest('drift',o.xy,end,1,.4,o.time+30.,'recovery'),o.time)
        c.executor.recovery_turn=replace(intent,action_id='drift')
        c.obs=replace(o,xy=tuple(o.xy[i]+.9*(end[i]-o.xy[i]) for i in range(2)))
        cmd=c.executor.tick(c.obs,c._guard,c._supported)
        self.assertEqual(cmd.omega,0.)
        self.assertIsNotNone(c.executor.result)
        self.assertIn('position reserve',c.executor.result.reason)
        self.assertNotEqual(c.executor.result.outcome,'success')

    def test_batch_once_and_existing_action_cap(self):
        c = recorded_turn(); c.last_plan_at = 0.
        c._guard = Mock(return_value='straight blocked')
        c._recovery_turn_candidates = Mock(return_value=[])
        c._recovery(None)
        c.obs = replace(c.obs,time=c.obs.time+2.)
        c._recovery(None)
        c._recovery_turn_candidates.assert_called_once()
        c = recorded_turn(); c.last_plan_at = 0.; c.recovery_actions = 8
        c._recovery_turn_candidates = Mock(side_effect=AssertionError('ninth action'))
        c._recovery(None)
        self.assertEqual(c.state,'HOLD')

    def test_real_supervisor_prefers_turn_after_recorded_executed_stall(self):
        baseline=recorded_turn();baseline.last_plan_at=0.
        baseline._recovery(None)
        self.assertIsNone(baseline.executor.recovery_turn)
        self.assertEqual(baseline.executor.request.direction,1)
        self.assertIn('previously supported',baseline.escape_events[-1]['reason'])
        c=recorded_turn();c.last_plan_at=0.
        c._record_recovery_turn_failure(c.recorded_failure)
        original=(c.recovery_since,c.recovery_permission)
        c._recovery(None)
        self.assertIsNotNone(c.executor.recovery_turn)
        self.assertAlmostEqual(math.degrees(c.executor.recovery_turn.target_yaw-c.obs.yaw),-20.)
        self.assertEqual(c.recovery_actions,1)
        self.assertEqual(original,(c.recovery_since,c.recovery_permission))
        self.assertIsNotNone(c.recovery_turn_failure_id)

    def test_censored_empty_or_old_episode_cannot_trigger(self):
        c=recorded_turn(); r=c.recorded_failure
        for invalid in (replace(r,outcome='censored'),replace(r,physical=False),
                        replace(r,executed_traversal=None),replace(r,action_id='older'),
                        replace(r,executed_traversal=replace(r.executed_traversal,duration_s=0.))):
            c._record_recovery_turn_failure(invalid)
            self.assertIsNone(c.recovery_turn_failure_id)
        c.recovery_since=r.time+1.
        c._record_recovery_turn_failure(r)
        self.assertIsNone(c.recovery_turn_failure_id)

    def test_checked_straight_connection_keeps_priority(self):
        c=recorded_turn();c.last_plan_at=0.
        c._record_recovery_turn_failure(c.recorded_failure)
        c._recovery_purpose=Mock(return_value=(0,'checked mission connection',0.))
        c._recovery_turn_candidates=Mock(side_effect=AssertionError('unnecessary turn'))
        c._recovery(None)
        self.assertIsNone(c.executor.recovery_turn)


class TurnExecutorTests(unittest.TestCase):
    def executor(self):
        e=m.MovementExecutor(); yaw=-math.radians(20.)
        end=(.4*math.cos(yaw),.4*math.sin(yaw))
        e.start(m.MovementRequest('turn',(0.,0.),end,1,.4,30.,'recovery'),0.)
        e.recovery_turn=RecoveryTurn((0.,0.),0.,yaw,end,1,30.,'turn')
        e.recovery_turn_guard=Mock(return_value='')
        return e

    def tick(self,e,t,yaw=0.,**kw):
        return e.tick(observation(t,yaw=yaw,grade=30.,turn=False,**kw),lambda cmd:'',lambda xy:True)

    def test_yaw_only_then_fresh_settled_history_before_forward(self):
        e=self.executor()
        commands=[self.tick(e,float(t)) for t in np.arange(0.,.36,.02)]
        self.assertTrue(any(c.omega<0 for c in commands));self.assertTrue(all(c.vx==0 for c in commands))
        yaw=-math.radians(20.)
        self.assertEqual(self.tick(e,.36,yaw).omega,0.)
        for t in np.arange(.38,2.38,.02):self.assertEqual(self.tick(e,float(t),yaw).vx,0.)
        command=self.tick(e,2.4,yaw)
        self.assertGreater(command.vx,0.); self.assertEqual(command.omega,0.)
        self.assertIsNone(e.recovery_turn)
        self.assertIsNone(e.result)  # A yaw action alone cannot be successful.

    def test_stop_and_failed_readmission_clear_permission(self):
        e=self.executor();self.tick(e,0.)
        self.assertEqual(e.stop(observation(.1)),m.Motion())
        self.assertIsNone(e.recovery_turn)
        e=self.executor(); e.recovery_turn_guard.return_value='new obstacle'
        self.tick(e,0.)
        self.assertEqual(e.result.outcome,'censored');self.assertFalse(e.result.physical)
        self.assertIsNone(e.recovery_turn)

    def test_overshoot_never_reverses_yaw(self):
        e=self.executor()
        for t in np.arange(0.,.4,.02):self.tick(e,float(t))
        for t in np.arange(.4,.8,.02):
            self.assertEqual(self.tick(e,float(t),-math.radians(23.)).omega,0.)

    def test_absolute_tilt_and_overspeed_take_precedence(self):
        for velocity,up in [((.6,0.),1.),((0.,0.),math.cos(math.radians(51.)))]:
            e=self.executor()
            o=replace(observation(0.,grade=30.,turn=False,velocity=velocity),up_z=up)
            command=e.tick(o,lambda cmd:'',lambda xy:True)
            self.assertEqual(command.omega,0.)

    def test_post_turn_rejection_has_no_drive_or_success_vote(self):
        e=self.executor()
        e.recovery_turn_guard.side_effect=lambda obs,req,phase: 'fresh corridor lost' if phase=='drive' else ''
        for t in np.arange(0.,.36,.02):self.tick(e,float(t))
        for t in np.arange(.36,2.42,.02):
            cmd=self.tick(e,float(t),-math.radians(20.))
            self.assertEqual(cmd.vx,0.)
        self.assertEqual(e.result.reason,'fresh corridor lost')
        self.assertFalse(e.result.physical)

    def test_missing_post_turn_history_and_request_deadline(self):
        e=self.executor();e.recovery_turn_finished=True
        for t in np.arange(0.,3.,.2):
            self.assertEqual(self.tick(e,float(t),-math.radians(20.)).vx,0.)
        self.assertIsNotNone(e.recovery_turn)
        self.tick(e,31.)
        self.assertEqual(e.result.reason,'primitive deadline elapsed')


class ControllerDeliveryTests(unittest.TestCase):
    """Exercise the real tree, executor, and final returned-command boundary."""
    def controller(self):
        c=recorded_turn();c.last_plan_at=0.
        c._record_recovery_turn_failure(c.recorded_failure)
        c._recovery(None)
        return c,c.obs

    def drive_ticks(self,c,o,**changes):
        commands=[]
        for t in np.arange(o.time,o.time+.7,.02):
            commands.append(c.tick(o.xy,o.yaw,changes.get('speed',0.),float(t),
                upright=changes.get('up',o.up_z),velocity=changes.get('velocity',(0.,0.)),
                body_quaternion=o.body_quaternion))
        return commands

    def test_actual_controller_returns_checked_yaw(self):
        c,o=self.controller()
        commands=self.drive_ticks(c,o)
        self.assertTrue(any(cmd.omega==-.35 for cmd in commands))
        self.assertTrue(all(cmd.vx==0. for cmd in commands))
        self.assertFalse(c.turn_enabled)  # Ordinary slope permission stays off.
        self.assertIsNone(c.executor.result)

    def test_real_output_keeps_physical_scope_and_stop_protections(self):
        for case in ('ordinary','wrong_owner','stale','hazard','tilt','speed','stop'):
            c,o=self.controller();changes={}
            if case=='ordinary':c.executor.recovery_turn=None
            if case=='wrong_owner':c.executor.recovery_turn=replace(c.executor.recovery_turn,action_id='wrong')
            if case=='stale':c.grid.fresh[:]=False
            if case=='hazard':c.grid.hazard[c.grid.world_to_cell(*o.xy)]=True
            if case=='tilt':changes['up']=math.cos(math.radians(51.))
            if case=='speed':changes.update(speed=.16,velocity=(.16,0.))
            if case=='stop':c.request('stop',o.time)
            commands=self.drive_ticks(c,o,**changes)
            self.assertTrue(all(cmd.omega==0. for cmd in commands),case)


if __name__=='__main__':unittest.main()
