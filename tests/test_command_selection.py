"""Select a feasible forward demand without weakening the motion envelope."""
import copy
from dataclasses import replace
import importlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from test_behavior_navigation import a,b,m,observation

s = importlib.import_module('rexmi_bt_tests.nav.command_selection')


class CommandSelectionTests(unittest.TestCase):
    def recorded(self):
        data=json.loads((Path(__file__).parent/'fixtures/guard_stale_fan.json').read_text())
        event=data['event'];d=event['terrain_evidence']
        g=a.AscentTerrainMap(world_size=d['world_size_m'],cell_size=d['cell_size_m'],
            origin=d['grid_origin'],footprint_radius=data['footprint_radius_m'],clearance=.1,
            half_length=d['half_length_m'],half_width=d['half_width_m'])
        region=tuple(slice(x,x+n) for x,n in zip(d['patch_first_cell'],d['patch_shape']))
        for name,values in d['fields'].items():
            target=getattr(g,name);target[region]=np.asarray(values,dtype=target.dtype)
        c=b.BehaviorTreeAscentController(g,[SimpleNamespace(x=13.,y=0.,label='exterior')],mode='characterize')
        c.obs=m.MotionObservation(tuple(event['position']),event['body_yaw'],tuple(event['velocity_world']),
            data['up_z'],0.,event['time'],data['support_inclination_deg'],False,True)
        for time,yaw in data['heading_samples']:c.passive_heading.observe(time,yaw)
        end=tuple(np.asarray(c.obs.xy)+5*np.array([math.cos(c.obs.yaw),math.sin(c.obs.yaw)]))
        c.executor.start(m.MovementRequest('recorded',c.obs.xy,end,1,.4,100.),c.obs.time)
        return c

    def executor(self, direction=1):
        e=m.MovementExecutor()
        e.start(m.MovementRequest('one',(0.,0.),(20.*direction,0.),direction,.4,100.),0.)
        e.release_guard=e.posture_guard=e.corridor_guard=lambda *args:''
        e.travel_axis=(float(direction),0.);e.committed_heading=0.
        e._state(m.MovementState.DRIVE_FORWARD if direction==1 else m.MovementState.DRIVE_REVERSE,0.,'fixture')
        e.drive_selector=lambda command:m.Motion(.375,0.)
        return e

    def test_recorded_fan_stale_at_nominal_and_clear_at_selected_speed(self):
        c=self.recorded();nominal=m.Motion(.4,0.)
        self.assertEqual(c._guard(nominal,record=False),'straight drive or stopping footprint is blocked')
        selected=c._select_forward_command(nominal)
        self.assertEqual(selected,m.Motion(.375,0.))
        self.assertEqual(c._guard(selected),'')
        self.assertEqual(c.executor.request.speed,.4)

    def test_candidate_probes_do_not_modify_evidence_or_deduplication(self):
        c=self.recorded();g=c.grid
        arrays={k:v.copy() for k,v in vars(g).items() if isinstance(v,np.ndarray)}
        history=copy.deepcopy(c.executor.history)
        for _ in range(3):self.assertEqual(c._select_forward_command(m.Motion(.4,0.)),m.Motion(.375,0.))
        self.assertFalse(c.guard_evidence_seen);self.assertFalse(c.oriented_admission_seen)
        self.assertFalse(c.escape_events);self.assertEqual(c.executor.history,history)
        self.assertIsNone(c.executor.pending_traversal)
        for k,v in arrays.items():np.testing.assert_array_equal(getattr(g,k),v,err_msg=k)

    def test_measured_velocity_floor_cannot_be_shrunk_by_command_selection(self):
        for speed in (.4,.45):
            c=self.recorded()
            c.obs=replace(c.obs,velocity=tuple(speed*np.array([math.cos(c.obs.yaw),math.sin(c.obs.yaw)])))
            self.assertEqual(c._select_forward_command(m.Motion(.4,0.)),m.Motion(.4,0.))
            self.assertTrue(c._guard(m.Motion(.1,0.),record=False))

    def test_real_unknown_obstacle_and_exclusion_cannot_be_bypassed(self):
        for kind in ('unknown','obstacle','exclusion','steep','stale'):
            c=self.recorded();g=c.grid
            point=np.asarray(c.obs.xy)+.25*np.array([math.cos(c.obs.yaw),math.sin(c.obs.yaw)])
            cell=g.world_to_cell(*point)
            if kind=='unknown':g.known[cell]=False
            elif kind=='obstacle':g.hazard[cell]=True;g.obstacle_distance[cell]=0.
            elif kind=='exclusion':g.approach_blocked[cell]=True
            elif kind=='steep':g.slope[cell]=math.tan(math.radians(36.))
            else:g.fresh[cell]=False
            self.assertEqual(c._select_forward_command(m.Motion(.4,0.)),m.Motion(.4,0.),kind)
            self.assertTrue(c._guard(m.Motion(.1,0.),record=False),kind)

    def test_bounded_descending_search_without_monotonicity_assumption(self):
        called=[]
        def check(cmd):
            called.append(cmd.vx)
            return '' if abs(cmd.vx-.325)<1e-9 else 'blocked'
        selected=s.select_forward_command(m.Motion(.38,0.),check)
        self.assertAlmostEqual(selected.vx,.325)
        np.testing.assert_allclose(called,[.38,.375,.35,.325])
        called.clear()
        result=s.select_forward_command(m.Motion(.4,0.),lambda cmd:called.append(cmd.vx) or 'blocked')
        self.assertEqual(result,m.Motion(.4,0.));self.assertEqual(len(called),13)
        self.assertAlmostEqual(min(called),.1)

    def test_low_nominal_reverse_zero_and_yaw_are_unchanged(self):
        calls=[]
        for cmd in (m.Motion(-.4,0.),m.Motion(),m.Motion(0.,.35)):
            self.assertEqual(s.select_forward_command(cmd,lambda c:calls.append(c)),cmd)
        self.assertFalse(calls)
        self.assertEqual(s.select_forward_command(m.Motion(.075,0.),lambda c:'blocked'),m.Motion(.075,0.))

    def test_drive_output_and_exposure_match_selected_command(self):
        e=self.executor();req=e.request;checked=[]
        for t in (0.,.02):
            cmd=e.tick(observation(t,xy=(.1*t,0.),velocity=(.1,0.)),lambda cmd:checked.append(cmd) or '',lambda p:True)
            self.assertEqual(cmd,m.Motion(.375,0.))
        self.assertIs(e.request,req);self.assertEqual(req.speed,.4)
        self.assertAlmostEqual(e.drive_exposure,.02)
        self.assertEqual(e.executed_traversal.maneuver,'DRIVE_FORWARD')
        records=[h for h in e.history if h['event']=='forward_speed_selected']
        self.assertEqual(len(records),1);self.assertEqual(records[0]['command_vx'],.375)
        self.assertTrue(all(c.vx==.375 for c in checked))

    def test_brake_admission_selects_but_does_not_credit_drive(self):
        e=self.executor();e._state(m.MovementState.BRAKE,0.,'fixture');e.stable_since=0.
        checked=[]
        command=e.tick(observation(.4),lambda c:checked.append(c) or '',lambda p:True)
        self.assertEqual(e.state,m.MovementState.DRIVE_FORWARD)
        self.assertEqual(command,m.Motion());self.assertFalse(e.forward_output)
        self.assertIsNone(e.pending_traversal);self.assertEqual(e.drive_exposure,0.)
        self.assertEqual(checked,[m.Motion(.375,0.)])
        self.assertFalse(any(h['event']=='forward_speed_selected' for h in e.history))

    def test_release_correction_records_actual_selected_output(self):
        e=self.executor();e._state(m.MovementState.RELEASE,0.,'fixture');e.release_origin=(0.,0.)
        command=e.tick(observation(.2,velocity=(.01,0.)),lambda c:'',lambda p:True)
        self.assertEqual(command,m.Motion(.375,0.));self.assertTrue(e.forward_output)
        self.assertEqual(e.state,m.MovementState.DRIVE_FORWARD)
        event=next(h for h in e.history if h['event']=='release_correction_emitted')
        self.assertEqual(event['command_vx'],command.vx)

    def test_selector_is_not_used_for_reverse_or_policy_active_release(self):
        e=self.executor(-1)
        def forbidden(cmd):raise AssertionError('unexpected selector')
        e.drive_selector=forbidden
        self.assertEqual(e.tick(observation(0.),lambda c:'',lambda p:True),m.Motion(-.4,0.))
        e=self.executor();e.drive_selector=forbidden
        e._state(m.MovementState.RELEASE,0.,'fixture');e.release_origin=(0.,0.)
        obs=replace(observation(.2,xy=(.01,0.),grade=30.,turn=False),terrain_gradient=(.5,0.))
        self.assertEqual(e.tick(obs,lambda c:'',lambda p:True),m.Motion())
        self.assertEqual(e.state,m.MovementState.RELEASE)

    def test_invalid_selector_fails_closed_and_final_guard_is_authoritative(self):
        for value in (None,m.Motion(0.,0.),m.Motion(-.2,0.),m.Motion(.41,0.),
                      m.Motion(math.nan,0.),m.Motion(.2,.1),m.Motion(.2,math.nan)):
            e=self.executor();e.drive_selector=lambda c:value
            e.tick(observation(0.),lambda c:'',lambda p:True)
            self.assertEqual(e.result.outcome,'censored');self.assertEqual(e.result.reason,'invalid forward speed selection')
            self.assertIsNone(e.pending_traversal);self.assertFalse(e.forward_output)
        e=self.executor()
        e.tick(observation(0.),lambda c:'updated map rejects chosen command',lambda p:True)
        self.assertEqual(e.result.reason,'updated map rejects chosen command')
        self.assertFalse(any(h['event']=='forward_speed_selected' for h in e.history))

    def test_priorities_remain_ahead_of_selection(self):
        for kind in ('stop','tip','overspeed','deadline','stale'):
            e=self.executor();calls=[];e.drive_selector=lambda c:calls.append(c) or c
            obs=observation(.2)
            if kind=='stop':e.stop(obs)
            elif kind=='tip':obs=replace(obs,up_z=.49)
            elif kind=='overspeed':obs=replace(obs,velocity=(.5,0.))
            elif kind=='deadline':obs=replace(obs,time=101.)
            else:obs=replace(obs,fresh=False)
            e.tick(obs,lambda c:'',lambda p:True)
            self.assertFalse(calls,kind)
            self.assertFalse(any(h['event']=='forward_speed_selected' for h in e.history),kind)

    def test_mutating_selector_cannot_raise_its_own_ceiling(self):
        e=self.executor();original=m.Motion(.4,0.)
        def mutate(command):
            command.vx=.8
            return command
        e.drive_selector=mutate
        selected,reason=e._checked_drive(original,lambda command:'')
        self.assertEqual(reason,'invalid forward speed selection')
        self.assertEqual(original,m.Motion(.4,0.))
        self.assertEqual(e.request.speed,.4)

    def test_selector_exception_is_censored_without_actuation(self):
        e=self.executor()
        def broken(command):raise RuntimeError('fixture')
        e.drive_selector=broken
        e.tick(observation(0.),lambda c:'',lambda p:True)
        self.assertEqual(e.result.reason,'invalid forward speed selection')
        self.assertIsNone(e.pending_traversal)


if __name__=='__main__':unittest.main()
