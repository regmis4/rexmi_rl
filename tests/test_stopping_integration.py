"""Actuation integration for the continuously swept body stopping envelope."""
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from test_behavior_navigation import a,b,m,ground,observation


class StoppingIntegrationTests(unittest.TestCase):
    def recorded(self):
        fixture=json.loads((Path(__file__).parent/'fixtures/guard_rejection.json').read_text())
        event=fixture['event'];d=event['terrain_evidence'];radius=fixture['footprint_radius_m']
        g=a.AscentTerrainMap(world_size=d['world_size_m'],cell_size=d['cell_size_m'],
            origin=tuple(d['grid_origin']),footprint_radius=radius,
            clearance=d['required_circle_clearance_m']-radius,
            half_length=d['half_length_m'],half_width=d['half_width_m'])
        selection=tuple(slice(x,x+n) for x,n in zip(d['patch_first_cell'],d['patch_shape']))
        for name,values in d['fields'].items():
            target=getattr(g,name);target[selection]=np.asarray(values,dtype=target.dtype)
        c=b.BehaviorTreeAscentController(g,[SimpleNamespace(x=13.,y=0.,label='exterior')],mode='characterize')
        c.obs=observation(event['time'],tuple(d['start']),d['body_yaw'],tuple(event['velocity_world']),grade=27.,turn=False)
        for time,yaw in fixture['heading_samples']:c.passive_heading.observe(time,yaw)
        end=tuple(np.asarray(c.obs.xy)+2*np.array([math.cos(c.obs.yaw),math.sin(c.obs.yaw)]))
        c.executor.start(m.MovementRequest('recorded',c.obs.xy,end,1,.4,100.),c.obs.time)
        return c,d

    def test_recorded_clear_body_sweep_can_continue_with_measured_passive_fan(self):
        c,d=self.recorded()
        self.assertFalse(c.grid.segment_clear(d['start'],d['end']))
        self.assertEqual(c._guard(m.Motion(.4,0.)),'')
        admission=[e for e in c.escape_events if e['event']=='oriented_stopping_admitted']
        self.assertEqual(len(admission),1)
        self.assertTrue(admission[0]['commanded'])
        self.assertGreater(admission[0]['passive_heading_offsets_rad'][1],.1)
        for _ in range(4):self.assertEqual(c._guard(m.Motion(.4,0.)),'')
        self.assertEqual(len(c.escape_events),1)

    def test_missing_history_or_larger_real_yaw_excursion_does_not_pass(self):
        for missing in (True,False):
            c,d=self.recorded();c.passive_heading.samples=[]
            if not missing:
                for t in np.arange(c.obs.time-.3,c.obs.time+.001,.02):
                    c.passive_heading.observe(t,c.obs.yaw+.2*math.sin(2*math.pi*(t-c.obs.time)/.3))
            self.assertTrue(c._guard(m.Motion(.4,0.)))
            self.assertFalse(any(e['event']=='oriented_stopping_admitted' for e in c.escape_events))

    def test_collision_steep_unknown_and_directional_exclusion_remain_binding(self):
        for kind in ('hazard','steep','unknown','exclusion'):
            c,d=self.recorded();cell=c.grid.world_to_cell(*d['end'])
            if kind=='hazard':c.grid.hazard[cell]=True;c.grid.blocked[cell]=True
            elif kind=='steep':c.grid.slope[cell]=math.tan(math.radians(36.))
            elif kind=='unknown':c.grid.known[cell]=False
            else:c.grid.mark_failure('failed',d['end'],d['body_yaw'],52.,'fixture')
            self.assertTrue(c._guard(m.Motion(.4,0.)),kind)

    def test_static_geometry_fallback_does_not_require_circle_clear_endpoint(self):
        c,d=self.recorded()
        self.assertFalse(c.grid.traversable(c.grid.world_to_cell(*d['end'])))
        self.assertTrue(c._straight_clear(d['start'],d['end'],d['body_yaw']))
        c.grid.hazard[c.grid.world_to_cell(*d['end'])]=True
        self.assertFalse(c._straight_clear(d['start'],d['end'],d['body_yaw']))

    def test_drift_and_brake_retain_physical_grade_and_exclusion_contract(self):
        g=ground(30.);c=b.BehaviorTreeAscentController(g,[SimpleNamespace(x=3.,y=0.,label='g')],mode='characterize')
        c.obs=observation(1.,velocity=(0.,.1),grade=30.,turn=False)
        self.assertEqual(c._brake_envelope(c.obs),'')
        g.mark_failure('failed',(0.,0.),math.pi/2,0.,'fixture')
        self.assertTrue(c._brake_envelope(c.obs))

    def test_turn_staging_is_not_granted_by_clear_straight_sweep(self):
        c,d=self.recorded()
        self.assertEqual(c._guard(m.Motion(0.,.35)),'turn requires gentle staging')


if __name__=='__main__':unittest.main()
