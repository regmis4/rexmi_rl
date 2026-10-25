"""Continuous swept geometry; physical mission acceptance remains separate."""
import json
import math
from pathlib import Path
import unittest

import numpy as np

from test_behavior_navigation import a,ground
from rexmi_bt_tests.nav.swept_footprint import oriented_swept_cells,oriented_sweep_reason,arrival_ray_interval


class SweptFootprintTests(unittest.TestCase):
    def grid(self):
        g=ground();g.half_length=.6;g.half_width=.1
        g.uphill_limit_deg=33.;g.side_limit_deg=25.
        return g

    def hazard(self,g,xy):
        cell=g.world_to_cell(*xy);g.hazard[cell]=True;g.blocked[cell]=True
        return cell

    def test_continuous_sweep_catches_obstacle_between_clear_endpoints(self):
        g=self.grid();g.half_length=.1
        self.hazard(g,(0.,0.))
        self.assertFalse(oriented_sweep_reason(g,(-1.,0.),(-1.,0.),0.))
        self.assertFalse(oriented_sweep_reason(g,(1.,0.),(1.,0.),0.))
        self.assertIn('obstacle',oriented_sweep_reason(g,(-1.,0.),(1.,0.),0.))

    def test_cell_corner_intersection_cannot_disappear_between_centres(self):
        g=self.grid();g.half_length=.02;g.half_width=.02
        self.hazard(g,(.1,.1))
        self.assertIn('obstacle',oriented_sweep_reason(g,(0.,0.),(0.,0.),0.))

    def test_passive_heading_envelope_catches_intermediate_rotation(self):
        g=self.grid();self.hazard(g,(.5,.5))
        for yaw in (0.,math.pi/2):
            self.assertFalse(oriented_sweep_reason(g,(0.,0.),(0.,0.),yaw))
        self.assertIn('obstacle',oriented_sweep_reason(g,(0.,0.),(0.,0.),0.,yaw_offsets=(0.,math.pi/2)))

    def test_uncertainty_union_contains_all_intermediate_exact_sweeps(self):
        g=self.grid();start=(-.31,.02);end=(.37,.64);low,high=.13,.61
        union=set(zip(*oriented_swept_cells(g,start,end,0.,(low,high))))
        for yaw in np.linspace(low,high,37):
            self.assertLessEqual(set(zip(*oriented_swept_cells(g,start,end,yaw))),union)

    def test_body_yaw_is_independent_of_reverse_travel(self):
        g=self.grid();self.hazard(g,(0.,.5))
        for start,end in (((-1.,0.),(1.,0.)),((1.,0.),(-1.,0.))):
            self.assertFalse(oriented_sweep_reason(g,start,end,0.))
            self.assertIn('obstacle',oriented_sweep_reason(g,start,end,math.pi/2))

    def test_unknown_stale_and_over_35_degree_support_remain_rejected(self):
        for kind in ('unknown','stale','slope'):
            g=self.grid();cell=g.world_to_cell(.5,0.)
            if kind=='unknown':g.known[cell]=False
            if kind=='stale':g.fresh[cell]=False
            if kind=='slope':g.slope[cell]=math.tan(math.radians(35.1))
            self.assertTrue(oriented_sweep_reason(g,(0.,0.),(1.,0.),0.),kind)

    def test_commanded_grade_and_incidental_drift_are_distinct(self):
        g=ground(30.);g.uphill_limit_deg=33.;g.side_limit_deg=25.
        self.assertIn('side slope',oriented_sweep_reason(g,(0.,0.),(0.,.4),0.))
        self.assertFalse(oriented_sweep_reason(g,(0.,0.),(0.,.4),0.,commanded=False))
        self.assertFalse(oriented_sweep_reason(g,(0.,0.),(.4,0.),0.))
        g.uphill_limit_deg=29.
        self.assertIn('uphill',oriented_sweep_reason(g,(0.,0.),(.4,0.),0.))

    def close_patch(self,g,xy,prefix):
        g.mark_failure(prefix+'a',xy,0.,1.,'fixture')
        g.mark_failure(prefix+'b',xy,math.pi,2.,'fixture')

    def test_directional_failure_is_binding_for_command_and_drift(self):
        g=self.grid();g.mark_failure('one',(0.,0.),0.,1.,'fixture')
        for commanded in (True,False):
            self.assertIn('excluded approach',oriented_sweep_reason(g,(0.,0.),(.8,0.),0.,commanded=commanded))
            self.assertFalse(oriented_sweep_reason(g,(0.,0.),(-.8,0.),0.,commanded=commanded))

    def test_occupied_exclusion_escape_cannot_deepen_or_reenter(self):
        g=self.grid();self.close_patch(g,(0.,0.),'one')
        self.assertFalse(oriented_sweep_reason(g,(0.,0.),(.8,0.),0.,no_go_exit=True))
        self.assertTrue(oriented_sweep_reason(g,(.8,0.),(0.,0.),0.,no_go_exit=True))
        self.assertTrue(oriented_sweep_reason(g,(.2,0.),(0.,0.),0.,no_go_exit=True))
        self.close_patch(g,(1.6,0.),'two')
        self.assertTrue(oriented_sweep_reason(g,(0.,0.),(2.4,0.),0.,no_go_exit=True))

    def test_outside_map_and_invalid_uncertainty_reject(self):
        g=self.grid()
        self.assertIsNone(oriented_swept_cells(g,(5.9,0.),(6.,0.),0.))
        for offsets in ((.1,-.1),(0.,math.inf),(0.,7.)):
            with self.assertRaises(ValueError):oriented_swept_cells(g,(0.,0.),(.1,0.),0.,offsets)

    def test_recorded_rejection_has_clear_continuous_oriented_geometry(self):
        fixture=json.loads((Path(__file__).parent/'fixtures/guard_rejection.json').read_text())
        d=fixture['event']['terrain_evidence'];radius=fixture['footprint_radius_m']
        g=a.AscentTerrainMap(world_size=d['world_size_m'],cell_size=d['cell_size_m'],
            origin=tuple(d['grid_origin']),footprint_radius=radius,
            clearance=d['required_circle_clearance_m']-radius,
            half_length=d['half_length_m'],half_width=d['half_width_m'])
        lo=d['patch_first_cell'];shape=d['patch_shape'];selection=tuple(slice(x,x+n) for x,n in zip(lo,shape))
        for name,values in d['fields'].items():
            target=getattr(g,name);target[selection]=np.asarray(values,dtype=target.dtype)
        g.uphill_limit_deg=33.;g.side_limit_deg=25.
        start,end,yaw=d['start'],d['end'],d['body_yaw']
        self.assertFalse(g.segment_clear(start,end))
        self.assertFalse(oriented_sweep_reason(g,start,end,yaw))
        self.assertEqual(fixture['event']['time'],52.6)


class ArrivalRayTests(unittest.TestCase):
    def test_forward_reverse_and_behind_target(self):
        self.assertEqual(arrival_ray_interval((0.,0.),(1.,0.),0.,1,.2),(.8,1.2))
        self.assertEqual(arrival_ray_interval((0.,0.),(-1.,0.),0.,-1,.2),(.8,1.2))
        self.assertIsNone(arrival_ray_interval((0.,0.),(-1.,0.),0.,1,.2))

    def test_intersection_does_not_enlarge_radius_or_move_destination(self):
        self.assertIsNone(arrival_ray_interval((0.,0.),(1.,.21),0.,1,.2))
        near,far=arrival_ray_interval((0.,0.),(1.,.12),0.,1,.2)
        for x in (near,far):self.assertAlmostEqual(math.hypot(x-1.,.12),.2)

    def test_inside_tangent_and_wrapped_heading(self):
        self.assertEqual(arrival_ray_interval((0.,0.),(0.,0.),0.,1,.2),(0.,.2))
        self.assertEqual(arrival_ray_interval((0.,0.),(1.,.2),0.,1,.2),(1.,1.))
        near,far=arrival_ray_interval((0.,0.),(-1.,0.),3*math.pi,1,.2)
        self.assertAlmostEqual(near,.8);self.assertAlmostEqual(far,1.2)

    def test_invalid_inputs_are_not_silently_accepted(self):
        for sign,radius,yaw in ((0,.2,0.),(1,0.,0.),(1,.2,math.nan)):
            with self.assertRaises(ValueError):arrival_ray_interval((0.,0.),(1.,0.),yaw,sign,radius)

    def test_recorded_corrections_already_intersect_inner_arrival_region(self):
        fixture=json.loads((Path(__file__).parent/'fixtures/arrival_ray.json').read_text())
        self.assertEqual(len(fixture['samples']),20)
        for row in fixture['samples']:
            interval=arrival_ray_interval(row['position'],row['endpoint'],row['body_yaw'],
                                          row['direction'],.75*row['arrival_radius'])
            self.assertIsNotNone(interval,row['time'])
            self.assertGreater(interval[1],interval[0],row['time'])


if __name__=='__main__':unittest.main()
