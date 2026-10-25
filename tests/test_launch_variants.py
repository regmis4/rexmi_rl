"""Paired launch selection is an experiment, never new traversal evidence."""
import copy
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from test_behavior_navigation import b, ground, observation
from rexmi_bt_tests.nav.route_exploration import launch_description


class LaunchVariantTests(unittest.TestCase):
    reference = [(0.,0.),(0.,1.),(-1.,1.),(-1.,2.),(4.,2.)]
    variant = [(0.,0.),(0.,2.),(4.,2.)]

    def control(self):
        c = b.BehaviorTreeAscentController(ground(20),[SimpleNamespace(x=5.,y=2.,label='goal')],mode='characterize')
        c.grid.terrain_prior = dict(geometry_id='fixture',radius_m=10.)
        c.configure_exploration(dict(records=[dict(corridor=None)]))
        c.configure_launch_variant()
        c.grid.max_support_slope = lambda xy,**kwargs: 10. if tuple(xy)==(-1.,2.) else 5.
        return c

    def candidate(self, path, cost, label):
        return ('unvalidated',cost,path,True,path,[],None,'key',0.,dict(profile=label,score=cost))

    def select(self,c,candidates):
        with patch.object(b,'motion_route_reason',return_value=''),patch.object(b,'observed_edge_reason',return_value=''), \
                patch.object(c,'_retreat_exists',return_value=True):
            return c._select_route_candidate(candidates,(0.,0.),math.pi/2)

    def test_bends_use_compacted_geometry_and_include_climb_entry(self):
        c=self.control()
        base=launch_description(c.grid,self.reference)
        split=launch_description(c.grid,[(0.,0.),(0.,.5)]+self.reference[1:-1]+[(1.,2.),(4.,2.)])
        self.assertEqual(base,split)
        self.assertEqual(base['pre_climb_bends'],3)
        self.assertEqual(launch_description(c.grid,self.variant)['pre_climb_bends'],1)

    def test_revisited_launch_coordinate_uses_actual_climb_edge(self):
        c=self.control();path=[(0.,0.),(0.,1.),(0.,0.),(4.,0.)]
        description=launch_description(c.grid,path)
        self.assertEqual(description['pre_climb_bends'],2)
        self.assertEqual(description['approach_prefix'],path[:3])
        self.assertEqual(description['climb_prefix'],path)

    def test_actual_prediction_event_retains_paired_reference(self):
        c=self.control();ref=self.candidate(self.reference,10.,'reference');new=self.candidate(self.variant,12.,'variant')
        selected=self.select(c,[ref,new]);metadata=selected[-1]['paired_launch_variant']
        c.attempt=dict(id='fixture:one',paired_launch_variant=metadata)
        c.path=list(self.variant);c.obs=observation(1.);c.executor.request=SimpleNamespace(direction=1)
        self.assertTrue(c._commit_prediction())
        event=c.attempt_events[-1]
        self.assertEqual(event['event'],'approach_start')
        self.assertEqual(event['paired_launch_variant']['reference'],metadata['reference'])
        self.assertEqual(event['paired_launch_variant']['variant'],metadata['variant'])
        self.assertTrue(c._commit_prediction())
        self.assertEqual(len(c.attempt_events),1)

    def test_variant_has_same_class_fewer_bends_and_flatter_launch(self):
        c=self.control();ref=self.candidate(self.reference,10.,'reference');new=self.candidate(self.variant,12.,'variant')
        selected=self.select(c,[new,ref])
        self.assertIs(selected,new)
        info=new[-1]['paired_launch_variant']
        self.assertEqual(info['comparison'],'same_climb_class')
        self.assertEqual(info['reference_route'],self.reference)
        self.assertEqual(info['reference']['climb_prefix'],self.reference)
        self.assertEqual(info['variant']['climb_prefix'],self.variant)
        self.assertEqual([r[-1]['score'] for r in (ref,new)],[10.,12.])
        self.assertIn('Same-class',new[-1]['exploration']['coverage_claim'])

    def test_both_improvements_are_required(self):
        for condition in ('flat_only','bends_only','nan'):
            c=self.control();ref=self.candidate(self.reference,10.,'reference')
            if condition=='flat_only':
                path=[(0.,0.),(0.,1.),(-.5,1.),(-.5,2.),(4.,2.)]
            else:
                path=self.variant
                c.grid.max_support_slope=lambda xy,**kwargs: float('nan') if condition=='nan' else 5.
            new=self.candidate(path,12.,'variant')
            self.assertIsNone(self.select(c,[ref,new]),condition)
            self.assertIn('diagnostic',c.planning_failure)

    def test_different_class_cannot_win_with_better_launch(self):
        c=self.control();ref=self.candidate(self.reference,10.,'reference')
        other=self.candidate([(0.,0.),(0.,4.),(4.,4.)],12.,'other')
        self.assertIsNone(self.select(c,[ref,other]))
        self.assertIn('different climb class',other[-1]['paired_launch_variant']['reason'])

    def test_original_reference_excludes_new_suffix_variants(self):
        c=self.control();c.launch_variant_sources={'trimmed':'source'}
        ref=self.candidate(self.reference,10.,'reference');new=self.candidate(self.variant,8.,'trimmed')
        self.assertIs(self.select(c,[ref,new]),new)
        self.assertEqual(new[-1]['paired_launch_variant']['reference_score'],10.)
        self.assertIsNone(self.select(c,[new]))
        self.assertIn('no original eligible reference',c.planning_failure)

    def test_lowest_original_cost_among_eligible_variants(self):
        c=self.control();ref=self.candidate(self.reference,10.,'reference')
        one=self.candidate(self.variant,12.,'one');two=self.candidate([(0.,0.),(0.,2.2),(5.,2.2)],14.,'two')
        self.assertIs(self.select(c,[two,ref,one]),one)

    def test_geometry_and_retreat_rejections_cannot_become_reference_or_variant(self):
        for failed in ('reference','variant'):
            for gate in ('geometry','retreat','incomplete'):
                c=self.control();ref=self.candidate(self.reference,10.,'reference');new=self.candidate(self.variant,12.,'variant')
                reject=self.reference if failed=='reference' else self.variant
                if gate=='incomplete':
                    selected=ref if failed=='reference' else new
                    updated=(*selected[:3],False,*selected[4:])
                    if failed=='reference':ref=updated
                    else:new=updated
                with patch.object(b,'motion_route_reason',side_effect=lambda g,r,y:'' if gate!='geometry' or r!=reject else 'unsafe'), \
                        patch.object(b,'observed_edge_reason',return_value=''), \
                        patch.object(c,'_retreat_exists',side_effect=lambda r:gate!='retreat' or r!=reject):
                    self.assertIsNone(c._select_route_candidate([ref,new],(0.,0.),math.pi/2),(failed,gate))

    def test_no_evidence_mutation_and_initial_only(self):
        c=self.control();before=[a.copy() for a in (c.grid.no_go,c.grid.failures,c.grid.successes,c.grid.approach_blocked)]
        ref=self.candidate(self.reference,10.,'reference');new=self.candidate(self.variant,12.,'variant')
        self.select(c,[ref,new])
        for old,current in zip(before,(c.grid.no_go,c.grid.failures,c.grid.successes,c.grid.approach_blocked)):
            np.testing.assert_array_equal(old,current)
        self.assertFalse(c.evidence.counts)
        c.attempt_number=1
        self.assertIs(self.select(c,[ref,new]),ref)
        with self.assertRaises(ValueError):c.configure_launch_variant()

    def test_opt_in_and_characterization_required(self):
        c=self.control();c.paired_launch_variant=False
        ref=self.candidate(self.reference,10.,'reference');new=self.candidate(self.variant,12.,'variant')
        self.assertIs(self.select(c,[ref,new]),ref)
        c.mode='qualified'
        with self.assertRaises(ValueError):c.configure_launch_variant()
        c.mode='characterize';c.exploration_coverage=None
        with self.assertRaises(ValueError):c.configure_launch_variant()

    def test_suffix_trim_preserves_prefix_and_exact_goal_and_deduplicates(self):
        c=self.control();route=self.variant+[(4.,4.),(7.,4.),(4.,2.)]
        routes=[('first',route,True),('duplicate',route,True)]
        original=copy.deepcopy(routes)
        def search(grid,origin,goal,heading,**kwargs):
            grid.last_route_search=dict(expansions=1,budget=60000,budget_exhausted=False)
            # staging is already at goal in this fixture; move destination to test suffix.
            self.assertEqual(origin,(4.,2.));self.assertEqual(goal,(6.,0.))
            self.assertFalse(kwargs['allow_frontier']);self.assertFalse(kwargs['require_fresh_start'])
            return [origin,(6.,2.),goal],True
        c.waypoints=[SimpleNamespace(x=6.,y=0.,label='goal')]
        with patch.object(b,'observed_route',side_effect=search) as searched:
            result=c._launch_suffix_variants(routes,(0.,0.),math.pi/2)
        self.assertEqual(searched.call_count,1)
        self.assertEqual(routes,original)
        self.assertEqual(result[0][1][:3],self.variant)
        self.assertEqual(result[0][1][-1],(6.,0.))

    def test_incomplete_suffix_is_never_admitted(self):
        c=self.control();c.waypoints=[SimpleNamespace(x=6.,y=0.,label='goal')]
        c.grid.last_route_search=dict(expansions=60000,budget=60000,budget_exhausted=True)
        with patch.object(b,'observed_route',return_value=([(4.,2.),(6.,2.)],False)):
            self.assertFalse(c._launch_suffix_variants([('route',self.variant,True)],(0.,0.),math.pi/2))
        self.assertFalse(c.launch_variant_sources)


if __name__ == '__main__':
    unittest.main()
