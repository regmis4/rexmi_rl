"""Shelf alternatives may supply useful onward geometry without forcing loops."""
import math
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_behavior_navigation import b, ground
from rexmi_bt_tests.nav import ascent_control
from rexmi_bt_tests.nav.checkpoint_runner import record_run_failure
from rexmi_bt_tests.nav.staged_routes import detour_shortcuts


class RouteShortcutTests(unittest.TestCase):
    route = [(0.,0.),(0.,-2.),(0.,1.),(3.,1.)]

    def controller(self):
        return b.BehaviorTreeAscentController(ground(),[SimpleNamespace(x=3.,y=1.,label='goal')],mode='characterize')

    def join(self,grid,yaw=math.pi/2):
        return next(p for p in detour_shortcuts(grid,self.route,yaw) if p['join_index']==2)

    def test_checked_loop_shortcut_retains_terminal_and_removes_four_metres(self):
        c=self.controller(); proposal=self.join(c.grid)
        self.assertFalse(proposal['rejection'])
        self.assertEqual(proposal['path'],[(0.,0.),(0.,1.),(3.,1.)])
        self.assertAlmostEqual(proposal['saved_m'],4.)

    def test_blocked_unknown_stale_and_directionally_excluded_connectors_reject(self):
        for kind in ('obstacle','unknown','stale','direction'):
            c=self.controller();g=c.grid;cell=g.world_to_cell(0.,.5)
            if kind=='obstacle': g.hazard[cell]=True;g.blocked[cell]=True
            if kind=='unknown': g.known[cell]=False
            if kind=='stale': g.fresh[cell]=False
            if kind=='direction': g.mark_failure('fixture',(0.,.5),math.pi/2,1.,'fixture')
            proposal=self.join(g)
            self.assertIn('segment 0:',proposal['rejection'],kind)

    def test_join_requires_gentle_full_staging_footprint(self):
        c=self.controller()
        with patch.object(c.grid,'max_support_slope',return_value=13.):
            self.assertIn('gentle staging',self.join(c.grid)['rejection'])
        with patch.object(c.grid,'turn_support_reason',return_value='blocked turn envelope'):
            self.assertIn('gentle staging',self.join(c.grid)['rejection'])

    def test_steep_start_cannot_use_shortcut_to_invent_turn(self):
        c=self.controller();slope=c.grid.max_support_slope
        with patch.object(c.grid,'max_support_slope',side_effect=lambda xy,**kw:30. if tuple(xy)==(0.,0.) else slope(xy,**kw)):
            self.assertIn('first heading',self.join(c.grid,yaw=0.)['rejection'])

    def test_remaining_suffix_is_checked_and_rejection_names_segment(self):
        c=self.controller();c.grid.known[c.grid.world_to_cell(2.,1.)]=False
        self.assertIn('segment 1:',self.join(c.grid)['rejection'])

    def test_distant_prior_remains_usable_but_local_observation_is_required(self):
        c=self.controller();c.grid.fresh[c.grid.world_to_cell(3.,1.)]=False
        self.assertFalse(self.join(c.grid)['rejection'])
        c.grid.fresh[c.grid.world_to_cell(1.,1.)]=False
        self.assertIn('fresh observations',self.join(c.grid)['rejection'])

    def test_originals_and_incomplete_frontier_status_are_preserved(self):
        c=self.controller();routes=[('shelf-0',self.route,False),('direct',[(0.,0.),(3.,1.)],True)]
        augmented=c._augment_routes(routes,(0.,0.),math.pi/2)
        self.assertEqual(augmented[:2],routes)
        self.assertGreater(len(augmented),2)
        self.assertTrue(all(not complete and path[-1]==self.route[-1] for _,path,complete in augmented[2:]))
        with patch.object(c.grid,'turn_support_reason',return_value='no staging'):
            self.assertEqual(c._augment_routes(routes,(0.,0.),math.pi/2),routes)

    def test_candidate_work_is_bounded(self):
        c=self.controller()
        self.assertEqual(len(detour_shortcuts(c.grid,self.route,0.,max_checks=1)),1)

    def test_rejected_shortcuts_keep_runtime_logger_and_json_contract(self):
        c=self.controller()
        with patch.object(c.grid,'turn_support_reason',return_value='blocked staging'):
            c._augment_routes([('shelf-0',self.route,False)],(0.,0.),0.)
        # This is the real runner's route-choice signature, including rejected
        # proposals. Missing fields formerly exited Isaac before its first row.
        signature=(tuple((p['profile'],round(p['length_m'],1),round(p['max_uphill_deg'],1)) for p in c.route_options),
                   tuple((p['profile'],p['connected'],p['has_route'],p['uphill_limit']) for p in c.route_searches))
        self.assertTrue(signature[0]);self.assertTrue(signature[1])
        json.dumps(dict(candidates=c.route_options,searches=c.route_searches))

    def test_software_failure_is_saved_before_simulator_cleanup(self):
        with tempfile.TemporaryDirectory() as directory,patch('builtins.print') as output:
            log=Path(directory)/'run.csv'
            try:
                raise KeyError('length_m')
            except KeyError as exc:
                record_run_failure(log,exc,4.,'source-fixture')
            saved=json.loads(log.with_suffix('.failure.json').read_text())
            self.assertFalse(saved['completed'])
            self.assertFalse(saved['counts_as_physical_failure'])
            self.assertFalse(saved['counts_as_qualification'])
            self.assertIn('KeyError',saved['traceback'])
            self.assertEqual(saved['source_id'],'source-fixture')
            self.assertTrue(output.call_args.kwargs['flush'])

    def test_cost_prediction_and_retreat_are_recomputed_before_selection(self):
        c=self.controller();c.actual_speed=0.;c.upright=1.;c.grid.last_route_search={}
        replies=[([],False),([],False),(self.route[:2],True),(self.route[1:],True)]
        with patch.object(c.grid,'shelf_candidates',return_value=[(0.,-2.)]), \
             patch.object(ascent_control,'observed_route',side_effect=replies), \
             patch.object(c,'_leg',side_effect=lambda path:path), \
             patch.object(c.model,'predict',wraps=c.model.predict) as predict, \
             patch.object(c,'_retreat_exists',wraps=c._retreat_exists) as retreat:
            self.assertTrue(c._compute_plan((0.,0.),math.pi/2,0.))
            self.assertEqual(predict.call_count,retreat.call_count)
            self.assertGreater(predict.call_count,1)
            self.assertIn(c.path,[call.args[0] for call in retreat.call_args_list])
        selected=next(option for option in c.route_options if option['selected'])
        self.assertIn('shortcut',selected['profile'])
        self.assertEqual(selected['route_path'],c.mission_route)
        self.assertEqual(c.mission_route[-1],(3.,1.))
        self.assertEqual(c.wp_idx,0)
        self.assertTrue(c.is_mission_checkpoint)
        self.assertTrue(any(o['profile']=='shelf-0' and not o['selected'] for o in c.route_options))

    def test_different_onward_geometry_cannot_be_hidden_by_equal_checkpoint_prefix(self):
        c=self.controller();leg=[(0.,0.),(0.,1.)]
        a=leg+[(3.,1.)];d=leg+[(0.,2.),(3.,1.)]
        self.assertNotEqual(c._route_signature(a,leg),c._route_signature(d,leg))


if __name__=='__main__':unittest.main()
