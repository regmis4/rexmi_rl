"""Occupied directional exclusions permit checked outward recovery only."""
import json
import importlib
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from test_behavior_navigation import a,b,m,ground
from rexmi_bt_tests.nav.recovery_intent import RecoveryPermission

sf=importlib.import_module('rexmi_bt_tests.nav.swept_footprint')


class DirectionalRecoveryExitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture=json.loads((Path(__file__).parent/'fixtures/directional_recovery_exits.json').read_text())

    def recorded(self,index):
        e=self.fixture['guards'][index];d=e['terrain_evidence']
        g=a.AscentTerrainMap(world_size=d['world_size_m'],cell_size=d['cell_size_m'],origin=d['grid_origin'],
            footprint_radius=d['required_circle_clearance_m']-.1,clearance=.1,
            half_length=d['half_length_m'],half_width=d['half_width_m'])
        region=tuple(slice(x,x+n) for x,n in zip(d['patch_first_cell'],d['patch_shape']))
        for name,values in d['fields'].items():
            target=getattr(g,name);target[region]=np.asarray(values,dtype=target.dtype)
        # The complete retained mask is unchanged after the run's one failure.
        # Never estimate exclusion depths on a cropped local patch.
        for indices in self.fixture['blocked_indices']:g.approach_blocked[tuple(indices)]=True
        for indices in self.fixture['repeated_failure_indices']:g.failures[tuple(indices)]=2
        # Original trial29 event identity, retained separately from raster masks.
        g.failure_records=[dict(attempt_id='trial29-run:1',position=[.9581948518753052,5.6317291259765625],
            radius_m=.6028254151344299,heading=1.2340031340427888,time=65.78,
            reason='measured lateral departure from straight corridor')]
        c=b.BehaviorTreeAscentController(g,[SimpleNamespace(x=13.,y=0.,label='exterior')],mode='characterize')
        c.obs=m.MotionObservation(tuple(e['position']),e['body_yaw'],tuple(e['velocity_world']),e['up_z'],0.,
            e['time'],e['support_inclination_deg'],False,True)
        for time,yaw in e['heading_samples']:c.passive_heading.observe(time,yaw)
        c.recovering=True
        c.recovery_permission=RecoveryPermission.capture(g,g.failure_records[0]['position'],65.78).observe(g,c.obs.xy)
        end=np.asarray(c.obs.xy)-1.6*np.array([math.cos(c.obs.yaw),math.sin(c.obs.yaw)])
        c.executor.start(m.MovementRequest('reverse',c.obs.xy,tuple(end),-1,.4,100.,'recovery'),c.obs.time)
        return c,e

    def test_exact_captured_outward_exits_and_two_required_rejections(self):
        for i in range(8):
            c,e=self.recorded(i)
            old=c.grid.drift_segment_reason(e['position'],e['corridor_endpoint'],no_go_exit=True)
            self.assertIn('excluded',old.replace('exclusion','excluded'),i)
            reason=c._stopping_reason(c.obs,e['corridor_endpoint'],c.obs.speed,
                commanded=False,record=False,directional_exit=True)
            if i in (3,7):self.assertTrue(reason,i)
            else:self.assertEqual(reason,'',i)
            full=c._guard(m.Motion(-.4,0.),record=False)
            # Index4's prospective reverse footprint extends beyond its captured
            # drift patch: absent replay geometry must still reject it.
            # Index3 only failed the old per-cell depth monotonicity rule;
            # its independently checked reverse and frozen occupied patch pass.
            if i in (4,7):self.assertTrue(full,i)
            else:self.assertEqual(full,'',i)

    def test_mission_forward_and_forward_recovery_do_not_gain_permission(self):
        for mode in ('mission','forward_recovery','forward_candidate','not_recovering'):
            c,e=self.recorded(0)
            command=m.Motion(-.4,0.)
            if mode=='not_recovering':c.recovering=False
            elif mode=='forward_candidate':command=m.Motion(.4,0.)
            else:
                c.executor.request=m.MovementRequest('other',c.obs.xy,(2.,8.),1,.4,100.,
                    'mission' if mode=='mission' else 'recovery')
            self.assertTrue(c._guard(command,record=False),mode)

    def test_outward_braking_and_reverse_candidate_before_request_are_checked(self):
        c,e=self.recorded(0)
        self.assertEqual(c._brake_envelope(c.obs),'')
        c.executor.request=None
        self.assertEqual(c._guard(m.Motion(-.4,0.),record=False),'')
        self.assertTrue(c._guard(m.Motion(.4,0.),record=False))
        self.assertEqual(c._brake_envelope(c.obs),'')
        c.recovering=False
        self.assertTrue(c._brake_envelope(c.obs))

    def test_positive_damping_requires_opposing_velocity_not_reverse_ownership(self):
        c,e=self.recorded(0)
        axis=np.array([math.cos(c.obs.yaw),math.sin(c.obs.yaw)])
        # Retain the captured outward direction but face the opposite body
        # heading. Its damping sign becomes positive; ownership stays reverse.
        from dataclasses import replace
        c.obs=replace(c.obs,yaw=c.obs.yaw+math.pi)
        self.assertLess(float(np.dot(c.obs.velocity,-axis)),0.)
        c.executor.brake_guard=c._brake_envelope
        self.assertGreater(c.executor.brake(c.obs).vx,0.)
        self.assertEqual(c.executor.brake_rejection,'')
        c.executor.request=replace(c.executor.request,direction=1)
        self.assertGreater(c.executor.brake(c.obs).vx,0.)
        self.assertEqual(c.executor.brake_rejection,'')

    def test_unknown_stale_obstacle_and_ground_boundary_still_block(self):
        for kind in ('unknown','stale','obstacle','steep'):
            c,e=self.recorded(0);cell=c.grid.world_to_cell(*c.obs.xy)
            if kind=='unknown':c.grid.known[cell]=False
            elif kind=='stale':c.grid.fresh[cell]=False
            elif kind=='obstacle':c.grid.hazard[cell]=True;c.grid.obstacle_distance[cell]=0.
            else:c.grid.slope[cell]=math.tan(math.radians(36.))
            self.assertTrue(c._guard(m.Motion(-.4,0.),record=False),kind)

    def test_all_direction_no_go_entry_and_deepening_still_reject(self):
        c,e=self.recorded(0);g=c.grid
        first=g.world_to_cell(*c.obs.xy);end=g.world_to_cell(*e['corridor_endpoint'])
        g.no_go[end]=True;g.no_go_depth[end]=1.
        self.assertTrue(c._guard(m.Motion(-.4,0.),record=False))
        g.no_go[first]=True;g.no_go_depth[first]=.2
        self.assertTrue(c._guard(m.Motion(-.4,0.),record=False))

    def test_mask_evidence_and_attempt_state_are_unchanged_by_checks(self):
        c,e=self.recorded(0);g=c.grid
        arrays={k:v.copy() for k,v in vars(g).items() if isinstance(v,np.ndarray)}
        for _ in range(3):self.assertEqual(c._guard(m.Motion(-.4,0.),record=False),'')
        for k,v in arrays.items():np.testing.assert_array_equal(getattr(g,k),v,err_msg=k)
        self.assertFalse(c.evidence.counts);self.assertFalse(c.evidence.completed)
        self.assertFalse(c.escape_events);self.assertFalse(c.guard_evidence_seen)

    def test_helper_cannot_enter_or_reenter_a_directional_patch(self):
        g=ground();heading=g.heading_bin(0.)
        for x in np.arange(-.6,.7,.2):
            for y in np.arange(-.6,.7,.2):g.approach_blocked[*g.world_to_cell(x,y),heading]=True
        self.assertEqual(g.directional_exit_reason((0.,0.),(1.,0.)),'')
        self.assertTrue(g.directional_exit_reason((-1.,0.),(0.,0.)))
        g.approach_blocked[*g.world_to_cell(1.2,0.),heading]=True
        self.assertTrue(g.directional_exit_reason((0.,0.),(1.5,0.)))

    def test_repeated_failure_counter_also_defines_directional_mask(self):
        g=ground();heading=g.heading_bin(0.)
        cell=g.world_to_cell(0.,0.);g.failures[*cell,heading]=2
        self.assertEqual(g.directional_exit_reason((.01,.01),(.5,.01)),'')
        self.assertTrue(g.directional_exit_reason((-.5,.01),(.01,.01)))

    def test_oriented_fallback_retains_departure_and_commanded_direction_rules(self):
        g=ground();g.mark_failure('one',(0.,0.),0.,0.,'fixture')
        start,end=(.01,.01),(1.,.01)
        self.assertTrue(sf.oriented_sweep_reason(g,start,end,0.,commanded=False))
        self.assertEqual(sf.oriented_sweep_reason(g,start,end,0.,commanded=False,directional_exit=True),'')
        self.assertEqual(sf.oriented_stopping_reason(g,start,end,0.,(0.,0.),commanded=False,directional_exit=True),'')
        self.assertTrue(sf.oriented_sweep_reason(g,start,end,0.,commanded=True,directional_exit=True))
        self.assertTrue(sf.oriented_stopping_reason(g,start,end,0.,(0.,0.),commanded=True,directional_exit=True))
        self.assertTrue(sf.oriented_sweep_reason(g,(-1.,.01),start,0.,commanded=False,directional_exit=True))
        g.approach_blocked[...,4]=g.approach_blocked[...,0]
        self.assertTrue(sf.oriented_sweep_reason(g,(.4,.01),start,math.pi,commanded=False,directional_exit=True))
        g.approach_blocked[*g.world_to_cell(1.2,.01),0]=True
        self.assertTrue(sf.oriented_sweep_reason(g,start,(1.5,.01),0.,commanded=False,directional_exit=True))


if __name__=='__main__':unittest.main()
