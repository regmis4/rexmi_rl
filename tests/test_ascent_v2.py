"""Ascent contracts without Isaac Sim; physical acceptance is a separate gate."""
import importlib
import json
import math
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

PKG = 'rexmi_ascent_tests'
if PKG not in sys.modules:
    package = ModuleType(PKG)
    package.__path__ = [str(Path(__file__).resolve().parents[1]/'source/rexmi_rl/nav')]
    sys.modules[PKG] = package
r = importlib.import_module(PKG+'.reachability')
m = importlib.import_module(PKG+'.ascent_map')
c = importlib.import_module(PKG+'.ascent_control')
p = importlib.import_module(PKG+'.terrain_prior')


def ground(grade=0.):
    grid = m.AscentTerrainMap(world_size=12., footprint_radius=.3)
    x, y = np.meshgrid(np.arange(-4.9, 5., .1), np.arange(-4.9, 5., .1), indexing='ij')
    grid.ingest(np.c_[x.ravel(), y.ravel(), (grade*x).ravel()], (0, 0, 3), 0.)
    grid.rebuild(0.)
    return grid


class ProbabilityTests(unittest.TestCase):
    def test_no_model_is_not_a_probability(self):
        p, reason = r.ReachabilityModel().predict([0.]*len(r.FEATURES))
        self.assertIsNone(p)
        self.assertEqual(r.eligibility(r.ApproachEvidence().estimate('a', p, reason), True), 'unvalidated')

    def test_independent_attempts_update_once(self):
        e = r.ApproachEvidence()
        self.assertAlmostEqual(e.estimate('a', .8).probability, .8)
        e.finish('run:1', 'a', 'failure')
        self.assertAlmostEqual(e.estimate('a', .8).probability, .64)
        self.assertFalse(e.finish('run:1', 'a', 'failure'))
        e.finish('run:2', 'a', 'censored')
        self.assertEqual(e.counts['a'], (0, 1))
        e.finish('run:3', 'a', 'success')
        self.assertAlmostEqual(e.estimate('a', .8).probability, .7)

    def test_attempt_thresholds_and_retreat_requirement(self):
        e = r.ApproachEvidence()
        self.assertEqual(r.eligibility(e.estimate('a', .8), False), 'normal')
        self.assertEqual(r.eligibility(e.estimate('b', .7), False), 'rejected')
        self.assertEqual(r.eligibility(e.estimate('b', .7), True), 'fallback')
        self.assertEqual(r.eligibility(e.estimate('c', .59), True), 'rejected')

    def test_directions_and_speed_bands_do_not_share_counts(self):
        self.assertNotEqual(r.ApproachEvidence.key((0,0),(2,0),0), r.ApproachEvidence.key((0,0),(2,0),math.pi))
        self.assertNotEqual(r.ApproachEvidence.key((0,0),(2,0),0,.4), r.ApproachEvidence.key((0,0),(2,0),0,-.4))

    def test_different_approaches_to_same_endpoint_keep_separate_evidence(self):
        a=[(0,0),(1,1),(2,1),(3,0),(4,0)]
        b=[(0,0),(1,-1),(2,-1),(3,0),(4,0)]
        key_a=r.ApproachEvidence.key(a[0],a[-1],0,path=a)
        key_b=r.ApproachEvidence.key(b[0],b[-1],0,path=b)
        self.assertNotEqual(key_a,key_b)
        e=r.ApproachEvidence();e.finish('one',key_a,'failure')
        self.assertAlmostEqual(e.estimate(key_a,.8).probability,.64)
        self.assertAlmostEqual(e.estimate(key_b,.8).probability,.8)

    def test_model_requires_disjoint_run_groups(self):
        artifact=dict(version=r.VERSION,features=list(r.FEATURES),mean=[0.]*len(r.FEATURES),
                      scale=[1.]*len(r.FEATURES),coefficients=[0.]*len(r.FEATURES),
                      minimum=[0.]*len(r.FEATURES),maximum=[1.]*len(r.FEATURES),
                      intercept=0.,calibration=[1.,0.],run_groups=dict(train=['a'],calibrate=['a'],evaluate=['b']))
        with self.assertRaises(ValueError):r.ReachabilityModel(artifact)

    def test_unqualified_and_out_of_range_predictions_cannot_authorize(self):
        artifact=dict(version=r.VERSION,features=list(r.FEATURES),mean=[0.]*len(r.FEATURES),
                      scale=[1.]*len(r.FEATURES),coefficients=[0.]*len(r.FEATURES),
                      minimum=[0.]*len(r.FEATURES),maximum=[1.]*len(r.FEATURES),
                      intercept=0.,calibration=[1.,0.],run_groups=dict(train=['a'],calibrate=['b'],evaluate=['c']),
                      evaluation=dict(qualified=False))
        model=r.ReachabilityModel(artifact)
        self.assertIsNone(model.predict([.5]*len(r.FEATURES))[0])
        artifact['evaluation']['qualified']=True
        self.assertIsNone(model.predict([9.]*len(r.FEATURES))[0])


class QualificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(__file__).resolve().parents[1]/'scripts/ascent_trials.py'
        spec=importlib.util.spec_from_file_location('ascent_trial_tests',path)
        cls.launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.launcher)

    def test_characterization_cannot_pass_the_gate_even_if_it_finishes(self):
        s=dict(result='complete',mission='ascent',ascent_mode='characterize',model_id='test',
               probability_model_qualified=True,
               spawn=[0.,0.,1.15,0.],goal=[13.,0.],seed=42,sim_seconds=300.,manual_intervention=False,
               policy_switches=0,peak_command_mps=.4,min_active_up_z=.9,waypoints_reached=1,total_waypoints=1)
        self.assertFalse(self.launcher.qualifies(s))
        s['ascent_mode']='qualified'
        self.assertFalse(self.launcher.qualifies(s))
        s['terrain_prior']=dict(geometry_id='test',kind='simulator_development',coverage=1.)
        self.assertTrue(self.launcher.qualifies(s))
        for field,value in (('diagnostic',True),('counts_as_qualification',False)):
            s[field]=value
            self.assertFalse(self.launcher.qualifies(s))
            del s[field]
        s['peak_command_mps']=.41
        self.assertFalse(self.launcher.qualifies(s))


class FittingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(__file__).resolve().parents[1]/'scripts/fit_reachability.py'
        spec=importlib.util.spec_from_file_location('ascent_fit_tests',path)
        cls.fitter=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.fitter)

    def data(self,run):
        rng=np.random.default_rng(1)
        data=[]
        for i in range(30):
            x=rng.normal(size=len(r.FEATURES))
            data.append(dict(run_id=run,attempt_id=f'{run}:{i}',features=x.tolist(),
                             approach_key=str(i),outcome='success' if x[0]>0 else 'failure',
                             policy_sha256='synthetic-unit-test-only'))
        return data

    def test_fit_reports_insufficient_evidence_instead_of_qualifying(self):
        artifact=self.fitter.fit(self.data('train'),self.data('calibrate'),self.data('evaluate'))
        self.assertFalse(artifact['evaluation']['qualified'])
        self.assertTrue(artifact['evaluation']['reasons'])
        self.assertTrue(math.isfinite(artifact['evaluation']['brier']))
        self.assertEqual(len(artifact['coefficients']),len(r.FEATURES))
        json.dumps(artifact,allow_nan=False)

    def test_unsupported_repeated_attempts_do_not_validate_online_updates(self):
        train=self.data('train');cal=self.data('cal');evaluation=self.data('eval')
        for row in evaluation:
            row['features']=[100.]*len(r.FEATURES)
            row['approach_key']='same-unsupported-route'
        report=self.fitter.fit(train,cal,evaluation)['evaluation']
        self.assertEqual(report['repeated_evidence_attempts'],0)
        self.assertEqual(report['supported_evaluation_attempts'],0)
        self.assertIsNone(report['brier'])
        self.assertFalse(report['qualified'])

    def test_fit_rejects_run_leakage(self):
        with self.assertRaises(ValueError):
            self.fitter.fit(self.data('same'),self.data('same'),self.data('test'))


class TerrainExitTests(unittest.TestCase):
    def test_partial_cliff_escape_cannot_bypass_directional_failure(self):
        g=ground(.4);xy=(.1,.1)
        cliff=g.world_to_cell(.5,.1)
        g.slope[cliff]=math.tan(math.radians(35.8));g.blocked[cliff]=True
        points=[(x,.1) for x in np.linspace(.1,-.7,12)]
        inherited=g.support_exposure(xy)[0]
        self.assertTrue(inherited)
        self.assertFalse(g.support_exit_check(points,inherited)[0])
        g.mark_failure('failed-backout',(-.3,.1),math.pi,1.,'physical reverse stall')
        self.assertIn('failed approach direction',g.support_exit_check(points,inherited)[0])


class TerrainPriorTests(unittest.TestCase):
    def test_global_route_uses_prior_beyond_fresh_near_field(self):
        g=ground();g.terrain_prior=dict(kind='simulator_development')
        start=(-3.,0.);goal=(3.,0.)
        g.fresh[:]=False
        for cell in np.argwhere(g.known):
            g.fresh[tuple(cell)]=math.dist(g.cell_to_world(*cell),start)<2.8
        path,done=c.observed_route(g,start,goal)
        self.assertTrue(done)
        self.assertEqual(path[-1],goal)
        self.assertFalse(g.fresh[g.world_to_cell(*goal)])
        self.assertFalse(g.last_route_search['budget_exhausted'])
        g.fresh[:]=False
        self.assertFalse(c.observed_route(g,start,goal)[0])

    def survey(self):
        g=ground(.2)
        g.terrain_prior=dict(version=1,kind='simulator_development',task='test-crater',seed=42,
                             coordinate_frame='world',geometry_id=p.geometry_id(g),radius_m=4.,coverage=1.,
                             description='Synthetic test geometry')
        return g

    def test_geometry_import_rebuilds_costs_without_importing_outcomes_or_freshness(self):
        source=self.survey();source.mark_failure('old',(0,0),0,1,'stall')
        source.mark_failure('other',(0,0),math.pi,2,'stall')
        source.cost[:]=999.
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'prior.npz';np.savez_compressed(path,**p.prior_snapshot(source))
            target=m.AscentTerrainMap(world_size=12.,footprint_radius=.3)
            p.load_prior(target,path,task='test-crater',seed=42)
            self.assertTrue(target.known[target.world_to_cell(0,0)])
            self.assertFalse(target.fresh.any())
            self.assertFalse(target.no_go.any())
            self.assertFalse(target.approach_blocked.any())
            self.assertEqual(int(target.successes.sum()+target.failures.sum()),0)
            self.assertLess(target.cost[target.world_to_cell(0,0)],999.)
            target.ingest([[.1,.1,.02]],(.1,.1,2.),3.)
            target.rebuild(3.)
            self.assertTrue(target.fresh[target.world_to_cell(.1,.1)])
            self.assertFalse(target.fresh[target.world_to_cell(3.,3.)])
            self.assertTrue(math.isfinite(target.quality((0,0))['age']))

    def test_prior_task_seed_grid_and_checksum_are_checked(self):
        source=self.survey()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'prior.npz';np.savez_compressed(path,**p.prior_snapshot(source))
            for task,seed,size in [('another',42,12.),('test-crater',41,12.),('test-crater',42,14.)]:
                target=m.AscentTerrainMap(world_size=size,footprint_radius=.3)
                with self.assertRaises(ValueError):p.load_prior(target,path,task=task,seed=seed)
            snapshot=p.prior_snapshot(source)
            measured=tuple(np.argwhere(snapshot['bin_count']>0)[0])
            snapshot['bin_high'][measured]+=1.
            np.savez_compressed(path,**snapshot)
            with self.assertRaisesRegex(ValueError,'checksum'):
                p.load_prior(m.AscentTerrainMap(world_size=12.),path,task='test-crater',seed=42)

    def test_prior_cannot_reset_evidence_during_a_mission(self):
        with self.assertRaisesRegex(ValueError,'fresh mission'):
            p.load_prior(self.survey(),'unused',task='test-crater',seed=42)

    def test_small_prior_cannot_claim_full_mission_coverage(self):
        source=self.survey()
        source._bin_count[source.n_cells//2:]=0
        source.terrain_prior['geometry_id']=p.geometry_id(source)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'prior.npz';np.savez_compressed(path,**p.prior_snapshot(source))
            target=m.AscentTerrainMap(world_size=12.)
            with self.assertRaisesRegex(ValueError,'required mission area'):
                p.load_prior(target,path,task='test-crater',seed=42,required_radius=4.)

    def test_updated_survey_can_be_reloaded_with_its_own_geometry_identity(self):
        source=self.survey();before=source.terrain_prior['geometry_id']
        source.ingest([[0.1,0.1,.2]],(0,0,2),4.);source.rebuild(4.)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'updated.npz';np.savez_compressed(path,**p.prior_snapshot(source))
            target=m.AscentTerrainMap(world_size=12.)
            metadata=p.load_prior(target,path,task='test-crater',seed=42)
            self.assertNotEqual(metadata['geometry_id'],before)
            self.assertEqual(metadata['parent_geometry_id'],before)

class TerrainTests(unittest.TestCase):
    def test_supported_flat_terrain_has_shelves_but_slope_does_not(self):
        self.assertTrue(ground().shelf_mask.any())
        g = ground(.3)
        self.assertFalse(g.shelf_mask[g.world_to_cell(0,0)])

    def test_first_failure_blocks_approach_not_whole_patch(self):
        g = ground(); cell = g.world_to_cell(0,0)
        g.mark_failure('a', (0,0), 0., 1., 'stall')
        self.assertFalse(g.direction_allowed(cell, 0))
        self.assertTrue(g.direction_allowed(cell, 4))
        self.assertFalse(g.no_go[cell])
        self.assertFalse(g.mark_failure('a', (0,0), math.pi, 2., 'duplicate'))
        self.assertFalse(g.no_go[cell])

    def test_distinct_approaches_close_zone_but_yaw_tweak_does_not(self):
        g = ground(); cell = g.world_to_cell(0,0)
        g.mark_failure('a', (0,0), 0., 1., 'stall')
        g.mark_failure('b', (0,0), math.pi/4, 2., 'same direction')
        self.assertFalse(g.no_go[cell])
        g.mark_failure('c', (0,0), math.pi, 3., 'other approach')
        self.assertTrue(g.no_go[cell])

    def test_exclusion_survives_map_retention(self):
        g = ground(); cell = g.world_to_cell(0,0)
        g.mark_failure('a',(0,0),0,1,'stall')
        g.retain_radius((4,4),1.)
        g.rebuild(2.)
        self.assertTrue(g.approach_blocked[cell[0],cell[1],0])

    def test_unknown_gap_and_steep_slope_do_not_create_shelves(self):
        g = ground(.8)
        self.assertFalse(g.shelf_mask[g.world_to_cell(0,0)])
        self.assertFalse(g.planning_support()[g.world_to_cell(0,0)])

    def test_no_go_exit_can_leave_but_cannot_reenter(self):
        g = ground()
        g.mark_failure('a',(0,0),0,1,'stall')
        g.mark_failure('b',(0,0),math.pi,2,'stall')
        self.assertTrue(g.segment_clear((0,0),(1.,0),no_go_exit=True))
        self.assertFalse(g.segment_clear((1.,0),(0,0),no_go_exit=True))

    def test_planner_can_escape_occupied_exclusion_but_not_reenter(self):
        g=ground()
        g.mark_failure('a',(0,0),0,1,'stall')
        g.mark_failure('b',(0,0),math.pi,2,'stall')
        path,done=c.observed_route(g,(0,0),(2.,0),no_go_exit=True,allow_frontier=False)
        self.assertTrue(done)
        self.assertTrue(all(g.segment_clear(a,b,no_go_exit=True) for a,b in zip(path,path[1:])))
        self.assertFalse(c.observed_route(g,(2.,0),(0,0),no_go_exit=True,allow_frontier=False)[1])

    def test_returned_route_edges_match_execution_support(self):
        g = ground(.4)
        g.uphill_limit_deg=25.;g.side_limit_deg=25.
        route, done = c.observed_route(g,(-2.,0),(2.,0),0)
        self.assertTrue(done)
        self.assertTrue(all(not g.support_reason(a,b) and g.segment_clear(a,b,require_fresh=False)
                            for a,b in zip(route,route[1:])))

    def test_frontier_learning_prefers_a_different_observation_approach(self):
        g=ground()
        first,done=c.observed_route(g,(0,0),(5.5,0))
        self.assertFalse(done)
        alternative,done=c.observed_route(g,(0,0),(5.5,0),frontier_penalties=[first[-1]])
        self.assertFalse(done)
        self.assertGreater(math.dist(first[-1],alternative[-1]),1.5)
        self.assertFalse(g.approach_blocked.any())

    def test_contour_search_avoids_alternating_diagonal_staircases(self):
        g=ground(.55);g.uphill_limit_deg=25.;g.side_limit_deg=25.
        path,done=c.observed_route(g,(-3.,0),(3.,0),0.)
        self.assertTrue(done)
        self.assertLess(r.route_features(g,path)[7],6.)
        self.assertTrue(all(not g.grade_reason(a,b) for a,b in zip(path,path[1:])))

    def test_oriented_body_distinguishes_lateral_clearance_from_contact(self):
        g=ground();g.half_length=.3;g.half_width=.12
        cell=g.world_to_cell(.1,.5);g.hazard[cell]=True
        self.assertEqual(g.oriented_support_reason((.1,.1),0.),'')
        self.assertIn('obstacle',g.oriented_support_reason((.1,.1),math.pi/2))
        g.hazard[cell]=False;g.known[cell]=False
        self.assertIn('unobserved',g.oriented_support_reason((.1,.1),math.pi/2))


class ControllerTests(unittest.TestCase):
    def control(self, mode='characterize'):
        return c.AscentController(ground(), [SimpleNamespace(x=3.,y=0.,label='goal')], mode=mode, run_id='test')

    def test_unvalidated_model_does_not_authorize_qualified_route(self):
        ctrl = self.control('qualified')
        self.assertFalse(ctrl._plan((0,0),0,0))
        self.assertIsNone(ctrl.attempt)
        self.assertIn('unvalidated', ctrl.planning_failure)

    def test_characterization_records_censored_rejection_without_no_go(self):
        ctrl = self.control()
        self.assertTrue(ctrl._plan((0,0),0,0))
        ctrl.recover((0,0),0,1.,'planning rejection')
        self.assertFalse(ctrl.grid.approach_blocked.any())
        self.assertEqual(ctrl.attempt_events[-1]['outcome'], 'censored')

    def test_rejected_local_approach_does_not_ban_the_shared_destination(self):
        ctrl=self.control();ctrl._plan((0,0),0,0.)
        ctrl.recover((.8,0),0,1.,'predicted stopping corridor')
        self.assertGreater(ctrl._recent_approach_cost([(0,0),(3,0)],2.),0.)
        self.assertEqual(ctrl._recent_approach_cost([(0,0),(0,1),(3,1),(3,0)],2.),0.)
        self.assertEqual(ctrl._recent_approach_cost([(0,0),(3,0)],22.),0.)
        ctrl.grid.shelf_candidates=lambda *args,**kwargs:[]
        self.assertTrue(ctrl._compute_plan((.8,0),0,2.))
        self.assertEqual(ctrl.checkpoint,(3.,0.))
        self.assertTrue(ctrl.mission_route_complete)
        self.assertFalse(ctrl.grid.no_go.any())

    def test_shelf_onward_connection_does_not_require_a_direct_connection(self):
        ctrl=self.control();ctrl.grid.shelf_candidates=lambda *args,**kwargs:[(1.5,1.)]
        search=c.observed_route
        def through_shelf(grid,start,goal,*args,**kwargs):
            if tuple(start)==(0.,0.) and tuple(goal)==(3.,0.):
                grid.last_route_search=dict(expansions=1,budget=5000,budget_exhausted=False)
                return [],False
            return search(grid,start,goal,*args,**kwargs)
        with patch.object(c,'observed_route',side_effect=through_shelf):
            self.assertTrue(ctrl._compute_plan((0.,0.),0.,0.))
        self.assertTrue(ctrl.mission_route_complete)
        ctrl.recover((.8,0.),0.,1.,'local geometry rejection')
        self.assertFalse(ctrl.unproductive_frontiers)

    def reverse_candidate(self,ctrl):
        choices=ctrl._recovery_candidates((0,0),0,0)
        return next(option for option in choices if option[1]['motion']==c.Motion(-.4,0.)
                    and math.dist(option[1]['start'],option[1]['endpoint'])>.7)

    def test_checked_reverse_commits_before_replanning_and_stop_still_wins(self):
        ctrl=self.control();candidate=self.reverse_candidate(ctrl)
        ctrl.recover((0,0),0,0.,'blocked forward approach')
        ctrl._recovery_candidates=lambda *args:[candidate]
        ctrl._plan=lambda *args:self.fail('global replan preempted local recovery')
        for index in range(5):
            now=1.1+index*.2;xy=(-.04*index,0.)
            motion=ctrl.tick(xy,0.,.1,now,velocity=(-.1,0.,0.))
            self.assertEqual(motion.vx,-.4)
        self.assertEqual(ctrl.recovery_count,1)
        self.assertEqual(sum(e['event']=='maneuver_start' for e in ctrl.escape_events),1)
        ctrl.request('stop',2.2)
        ctrl.tick((-.16,0),0,0,2.3,velocity=(0,0,0))
        self.assertEqual(ctrl.state,'HOLD');self.assertIsNone(ctrl.maneuver)
        self.assertFalse(ctrl.grid.no_go.any())

    def test_slow_planner_cannot_starve_a_checked_recovery(self):
        ctrl=self.control();candidate=self.reverse_candidate(ctrl)
        ctrl.recover((0,0),0,0.,'blocked forward approach')
        ctrl.recovery_count=1;ctrl.recovery_plan_started=0.
        ctrl._recovery_candidates=lambda *args:[candidate]
        def pending(*args):ctrl.plan_pending=True;return False
        ctrl._plan=pending
        self.assertEqual(ctrl.tick((0,0),0,0,3.,velocity=(0,0,0)).vx,-.4)
        self.assertFalse(ctrl.plan_pending)
        self.assertEqual(ctrl.recovery_count,2)

    def test_recovery_overspeed_brakes_without_abandoning_commitment(self):
        ctrl=self.control();candidate=self.reverse_candidate(ctrl)
        ctrl.recover((0,0),0,0.,'blocked forward approach')
        ctrl._start_recovery_maneuver(candidate[1],(0,0),0,0,1.)
        motion=ctrl.tick((-.1,0),0,.6,1.2,velocity=(-.6,0,0))
        self.assertGreater(motion.vx,0.)
        self.assertIsNotNone(ctrl.maneuver)
        self.assertIn('overspeed',ctrl.reason)

    def test_unstarted_recovery_does_not_claim_exhausted_attempts(self):
        ctrl=self.control();ctrl.recover((0,0),0,0.,'blocked approach')
        ctrl.tick((0,0),0,0,91.,velocity=(0,0,0))
        self.assertEqual(ctrl.state,'HOLD')
        self.assertIn('without an executable maneuver',ctrl.reason)
        self.assertEqual(ctrl.climb_failures,0)

    def test_candidate_enumeration_preserves_the_active_travel_purpose(self):
        ctrl=self.control();ctrl.route_purpose='retreat'
        ctrl._recovery_candidates((0,0),0,0)
        self.assertEqual(ctrl.route_purpose,'retreat')

    def test_bounded_reverse_is_not_mistaken_for_a_downhill_only_route(self):
        ctrl=self.control();ctrl.grid=ground(-.25)
        # Reversing here climbs 14°, within the normal 25° capability. The 8°
        # regroup restriction must not reject a short positioning maneuver.
        reverse=[item for _,item in ctrl._recovery_candidates((0,0),0,0) if item['motion'].vx<0]
        self.assertTrue(reverse)
        ctrl.recover((0,0),0,0.,'needs reposition')
        ctrl._start_recovery_maneuver(reverse[0],(0,0),0,0,1.)
        self.assertEqual(ctrl.route_purpose,'adjust')
        self.assertLess(ctrl.tick((0,0),0,0,1.1,velocity=(0,0,0)).vx,0.)

    def test_reverse_runs_to_the_checked_endpoint_instead_of_stopping_short(self):
        ctrl=self.control();candidate=self.reverse_candidate(ctrl)
        ctrl.recover((0,0),0,0.,'needs backout')
        ctrl._start_recovery_maneuver(candidate[1],(0,0),0,0,1.)
        self.assertEqual(ctrl.tick((-.56,0),0,.1,2.,velocity=(-.1,0,0)).vx,-.4)
        self.assertIsNotNone(ctrl.maneuver)
        ctrl.tick((-.71,0),0,.1,2.5,velocity=(-.1,0,0))
        self.assertIsNone(ctrl.maneuver)
        self.assertTrue(ctrl.recovery_settling)

    def test_recovery_prefers_useful_displacement_when_long_corridor_is_checked(self):
        ctrl=self.control();ctrl.waypoints=[SimpleNamespace(x=-3.,y=0.,label='goal')]
        candidate=ctrl._recovery_candidates((0,0),0,0)[0][1]
        self.assertEqual(candidate['motion'].vx,-.4)
        self.assertGreater(math.dist(candidate['start'],candidate['endpoint']),.7)

    def test_local_recovery_values_obstacle_room_over_goal_distance(self):
        ctrl=self.control();g=ctrl.grid
        g.hazard[g.world_to_cell(.8,.8)]=True
        g.obstacle_distance=m.ndi.distance_transform_edt(~g.hazard,sampling=g.cell_size)
        g.blocked=g.hazard | (g.obstacle_distance<g.footprint_radius+g.clearance)
        choices=ctrl._recovery_candidates((0,0),0,0)
        self.assertTrue(any(item['motion'].vx>0 for _,item in choices))
        selected=choices[0][1]
        self.assertEqual(selected['motion'].vx,-.4)
        self.assertGreater(g.obstacle_distance[g.world_to_cell(*selected['endpoint'])],
                           g.obstacle_distance[g.world_to_cell(.8,0)])

    def test_immediate_geometry_cancellation_does_not_exhaust_maneuver_budget(self):
        ctrl=self.control();candidate=self.reverse_candidate(ctrl)
        ctrl.recover((0,0),0,0.,'needs backout')
        ctrl._start_recovery_maneuver(candidate[1],(0,0),0,0,1.)
        ctrl.motion_clear=lambda *args:False
        ctrl.tick((-.01,0),0,0,1.02,velocity=(0,0,0))
        self.assertEqual(ctrl.recovery_count,0)
        self.assertIsNone(ctrl.maneuver)
        self.assertEqual(ctrl.climb_failures,0)
        self.assertEqual(ctrl.attempt_events[-1]['outcome'],'censored')
        self.assertIn(candidate[1]['key'],ctrl.recovery_tried)

    def test_committed_reverse_stall_records_one_directional_failure(self):
        ctrl=self.control();candidate=self.reverse_candidate(ctrl)
        ctrl.recover((0,0),0,0.,'needs backout')
        ctrl._start_recovery_maneuver(candidate[1],(0,0),0,0,1.)
        for now in np.arange(1.1,5.,.1):
            ctrl.tick((0,0),0,0,float(now),velocity=(0,0,0))
            if ctrl.maneuver is None:break
        self.assertEqual(ctrl.climb_failures,1)
        self.assertIn('poor drive progress',ctrl.reason)
        cell=ctrl.grid.world_to_cell(0,0)
        self.assertFalse(ctrl.grid.direction_allowed(cell,4))
        self.assertTrue(ctrl.grid.direction_allowed(cell,0))

    def test_slow_but_steady_reverse_progress_keeps_its_command(self):
        ctrl=self.control();candidate=self.reverse_candidate(ctrl)
        ctrl.recover((0,0),0,0.,'needs backout')
        ctrl._start_recovery_maneuver(candidate[1],(0,0),0,0,1.)
        for now in np.arange(1.1,12.,.1):
            self.assertEqual(ctrl.tick((-.03*(now-1.),0),0,.03,float(now),velocity=(-.03,0,0)).vx,-.4)
        self.assertEqual(ctrl.climb_failures,0)

    def test_recovery_reacts_to_worsening_posture_before_hard_tilt_stop(self):
        ctrl=self.control();candidate=self.reverse_candidate(ctrl)
        ctrl.recover((0,0),0,0.,'needs backout')
        ctrl._start_recovery_maneuver(candidate[1],(0,0),0,0,1.)
        for i in range(1,7):
            ctrl.tick((-.01*i,0),0,.1,1.+i*.1,upright=1.-.04*i,velocity=(-.1,0,0))
            if ctrl.maneuver is None:break
        self.assertIsNone(ctrl.maneuver)
        self.assertGreater(ctrl.upright,.65)
        self.assertEqual(ctrl.climb_failures,1)
        self.assertIn('worsening posture',ctrl.reason)
        self.assertEqual(ctrl.state,'RECOVER')

    def test_reverse_corrects_measured_yaw_drift_with_checked_steering(self):
        ctrl=self.control();candidate=self.reverse_candidate(ctrl)
        ctrl.recover((0,0),0,0.,'needs backout')
        ctrl._start_recovery_maneuver(candidate[1],(0,0),0,0,1.)
        motion=ctrl.tick((-.1,-.06),.2,.1,1.2,velocity=(-.1,0,0))
        self.assertEqual(motion.vx,-.4)
        self.assertLess(motion.omega,0.)
        self.assertTrue(ctrl.motion_clear((-.1,-.06),.2,.1,motion))

    def test_actual_displacement_stays_bounded_without_fabricating_endpoint_success(self):
        ctrl=self.control();candidate=self.reverse_candidate(ctrl)
        ctrl.recover((0,0),0,0.,'needs backout')
        ctrl._start_recovery_maneuver(candidate[1],(0,0),0,0,1.)
        ctrl.tick((-.95,-.2),0,.1,3.,velocity=(-.1,0,0))
        self.assertIsNone(ctrl.maneuver)
        self.assertTrue(ctrl.recovery_settling)
        self.assertEqual(ctrl.attempt_events[-1]['outcome'],'censored')
        self.assertEqual(ctrl.climb_failures,0)

    def test_unmoved_frontier_rejection_is_search_memory_not_physical_evidence(self):
        ctrl=self.control();ctrl._plan((0,0),0,0.)
        ctrl.mission_route_complete=False
        endpoint=ctrl.mission_route[-1]
        ctrl.recover((0,0),0,1.,'newly observed approach obstruction')
        self.assertEqual(ctrl.unproductive_frontiers,[endpoint])
        self.assertFalse(ctrl.evidence.counts)
        self.assertFalse(ctrl.grid.approach_blocked.any())

    def test_physical_failure_updates_memory_and_operator_stop_wins(self):
        ctrl = self.control()
        ctrl._plan((0,0),0,0)
        ctrl.recover((0,0),0,2.,'stall',physical=True)
        self.assertTrue(ctrl.grid.approach_blocked.any())
        ctrl.request('stop',3.)
        ctrl.tick((0,0),0,0,4.,velocity=(0,0,0))
        self.assertEqual(ctrl.state,'HOLD')
        self.assertEqual(ctrl.attempt_events[-1]['outcome'],'failure')

    def test_route_lost_after_departure_updates_probability_not_traction(self):
        ctrl=self.control();ctrl._plan((0,0),0,0.)
        ctrl.recover((.8,0),0,3.,'newly observed obstacle')
        outcome=ctrl.attempt_events[-1]
        self.assertEqual(outcome['outcome'],'failure')
        self.assertFalse(outcome['physical'])
        self.assertFalse(ctrl.grid.approach_blocked.any())

    def test_final_limit_applies_to_every_state(self):
        ctrl = self.control()
        ctrl._step = lambda *args: c.Motion(-1.2,1.)
        for state in ('FOLLOW','RECOVER','OBSERVE','SETTLE','HOLD','MANUAL'):
            ctrl.state = state
            motion = ctrl.tick((0,0),0,0,1.,velocity=(0,0,0))
            self.assertEqual(motion.vx,-.4)
            self.assertEqual(motion.omega,.35)

    def test_high_speed_is_not_an_ascent_configuration(self):
        with self.assertRaises(ValueError):
            c.AscentController(ground(),[],cruise_speed=.8)

    def test_arrival_requires_settling_not_merely_crossing(self):
        ctrl=self.control(); ctrl._plan((0,0),0,0.)
        ctrl.checkpoint=(3.,0.);ctrl.is_mission_checkpoint=True
        ctrl.set_state('SETTLE','test',1.)
        ctrl.tick((3.,0.),0,.2,1.1,velocity=(.2,0,0))
        self.assertNotEqual(ctrl.state,'COMPLETE')
        for t in np.arange(1.2,2.2,.1):
            ctrl.tick((3.,0.),0,0,float(t),velocity=(0,0,0))
        self.assertEqual(ctrl.state,'COMPLETE')
        self.assertEqual(ctrl.attempt_events[-1]['outcome'],'success')

    def test_measured_turn_progress_does_not_become_ground_failure(self):
        ctrl = self.control();ctrl._plan((0,0),math.pi,0.)
        ctrl.motion_clear = lambda *args: True
        for t in range(1,12):
            ctrl.tick((0,0),math.pi-.16*t,0,float(t),velocity=(0,0,0))
        self.assertEqual(ctrl.state,'FOLLOW')
        self.assertFalse(ctrl.grid.approach_blocked.any())

    def test_operator_stop_invalidates_pending_plan(self):
        ctrl = c.AscentController(ground(),[SimpleNamespace(x=3.,y=0.,label='goal')],
                                  mode='characterize',async_planning=True)
        try:
            self.assertFalse(ctrl._plan((0,0),0,0))
            ctrl.request('stop',.1)
            ctrl.plan_job.result(timeout=20)
            ctrl.tick((0,0),0,0,1.,velocity=(0,0,0))
            self.assertEqual(ctrl.state,'HOLD')
            self.assertIsNone(ctrl.attempt)
            self.assertFalse(ctrl.attempt_events)
        finally:
            ctrl.close()

    def test_regroup_destination_is_latched_and_preserves_mission(self):
        grid=ground(.3)
        ctrl=c.AscentController(grid,[SimpleNamespace(x=3.,y=0.,label='goal')],mode='characterize')
        ctrl.regroup_required=True
        self.assertTrue(ctrl._compute_regroup((2.,0),0,1.))
        destination=ctrl.regroup_goal
        self.assertLess(destination[0],.7)
        self.assertEqual(ctrl.attempt['features'][-1],-.4)
        ctrl.finish_attempt('censored','test replan',(2.,0),0,2.)
        self.assertTrue(ctrl._compute_regroup((1.8,0),0,3.))
        self.assertEqual(ctrl.regroup_goal,destination)
        self.assertEqual(ctrl.waypoints[0].x,3.)
        self.assertFalse(ctrl.is_mission_checkpoint)
        ctrl.request('stop',4.)
        self.assertEqual(ctrl.tick((1.8,0),0,0,5.,velocity=(0,0,0)).vx,0.)
        self.assertEqual(ctrl.state,'HOLD')

    def test_unvalidated_regroup_cannot_bypass_qualified_mode(self):
        ctrl=self.control('qualified');ctrl.grid=ground(.3)
        ctrl.regroup_required=True
        self.assertFalse(ctrl._compute_regroup((2.,0),0,1.))
        self.assertIsNone(ctrl.attempt)

    def test_unvalidated_recovery_cannot_bypass_qualified_mode(self):
        ctrl=self.control('qualified')
        self.assertFalse(ctrl._recovery_candidates((0,0),0,0))

    def test_exhausted_local_maneuvers_still_check_a_new_observed_route(self):
        ctrl=self.control();ctrl.recover((0,0),0,0.,'test recovery')
        ctrl.recovery_count=6
        ctrl._recover_tick((0,0),0,0.,2.)
        self.assertEqual(ctrl.state,'FOLLOW')
        self.assertIsNotNone(ctrl.attempt)

    def test_selected_checkpoint_replaces_regroup_without_clearing_exclusions(self):
        ctrl=self.control()
        ctrl.grid.mark_failure('old',(0,2),0,0,'stall')
        ctrl.checkpoint_mode(True,0.)
        ctrl.regroup_required=True;ctrl.regroup_goal=(-2,0)
        runner=importlib.import_module(PKG+'.checkpoint_runner')
        event=runner.apply_operator_request(ctrl,None,'target',2.,0.,(0,0),0.,1.)
        self.assertTrue(event['accepted'])
        self.assertFalse(ctrl.regroup_required)
        self.assertEqual(ctrl.waypoints[0].x,2.)
        self.assertTrue(ctrl.grid.approach_blocked.any())

    def test_unreachable_shelf_has_an_explicit_rejection(self):
        ctrl=self.control()
        g=ctrl.grid
        row=g.world_to_cell(1.,0)[0]
        g.known[row:row+4,:]=False
        g.shelf_candidates=lambda *args,**kw:[(3.,0)]
        ctrl._plan((0,0),0,0.)
        shelf=next(option for option in ctrl.route_options if option['profile']=='shelf-0')
        self.assertIn('no observed',shelf['rejection'])
        self.assertFalse(shelf['selected'])

    def test_steep_corridor_does_not_invent_a_mid_climb_settling_shelf(self):
        ctrl=self.control();ctrl.grid=ground(.4)
        route=[(float(x),0.) for x in np.arange(-2.,3.,.2)]
        self.assertEqual(ctrl._leg(route),route)

    def test_hillside_turn_planning_uses_the_execution_drift_reserve(self):
        ctrl=self.control();g=ground(.4);ctrl.grid=g
        # A steep cell is outside the ordinary circular footprint but inside
        # the extra drift reserve required by a stationary hillside turn.
        g.slope[g.world_to_cell(.7,.1)]=math.tan(math.radians(40))
        point=(.1,.1)
        self.assertFalse(g.support_reason(point))
        self.assertIn('35 degrees',g.turn_support_reason(point))
        self.assertFalse(ctrl.motion_clear(point,0.,0.,c.Motion(0.,.35)))
        path,done=c.observed_route(g,point,(.1,2.1),0.,allow_frontier=False)
        # Any allowed departure must continue straight until there is enough
        # support to change heading; the unsafe immediate turn is excluded.
        if path:
            angle=math.atan2(path[1][1]-point[1],path[1][0]-point[0])
            self.assertLessEqual(abs(angle),.30)

    def test_geometric_rejection_adjusts_locally_before_downhill_regroup(self):
        ctrl=self.control();ctrl.grid=ground(.4)
        ctrl.recover((0,0),0,1.,'turn clearance')
        self.assertFalse(ctrl.regroup_required)
        ctrl.recovery_count=2
        ctrl.recover((0,0),0,3.,'second adjustment')
        self.assertTrue(ctrl.regroup_required)
        other=self.control();other.grid=ground(.4)
        other.recover((0,0),0,1.,'physical stall',physical=True)
        self.assertTrue(other.regroup_required)

    def test_lookahead_does_not_cut_across_a_valid_contour_corner(self):
        ctrl=self.control();ctrl.grid=ground(.55)
        ctrl.grid.uphill_limit_deg=ctrl.grid.side_limit_deg=25.
        ctrl._plan((-.2,-.2),math.pi/4,0.)
        ctrl.path=[(-.5,-.5),(0.,0.),(.5,-.5),(1.,0.)]
        ctrl.checkpoint=(1.,0.);ctrl.path_index=0
        motion=ctrl.tick((-.2,-.2),math.pi/4,0.,.1,velocity=(0.,0.,0.))
        self.assertGreater(motion.vx,.3)
        self.assertLess(ctrl.tracking_lookahead,.8)
        self.assertFalse(ctrl.grid.grade_reason((-.2,-.2),ctrl.target))

    def test_measured_overspeed_brakes_before_entering_next_terrain(self):
        ctrl=self.control();ctrl._plan((0,0),0,0.)
        motion=ctrl.tick((0,0),0,.8,.1,velocity=(.8,0,0))
        self.assertLess(motion.vx,0.)
        self.assertIn('overspeed',ctrl.reason)

    def test_hillside_braking_opposes_creep_without_raising_command_limit(self):
        ctrl=self.control();ctrl.grid=ground(.4)
        ctrl.velocity_world=np.array([.12,0.])
        self.assertEqual(ctrl._brake((0,0),0).vx,-.4)
        ctrl.velocity_world=np.array([-.12,0.])
        self.assertEqual(ctrl._brake((0,0),0).vx,.4)

    def test_braking_anchor_is_local_to_the_current_episode(self):
        ctrl=self.control();ctrl.grid=ground(.4);ctrl.velocity_world=np.array([.04,0.])
        ctrl.last_tick=0.;initial=ctrl._brake((0,0),0).vx
        ctrl.last_tick=.2
        self.assertLess(ctrl._brake((.08,0),0).vx,initial)
        ctrl.last_tick=2.
        self.assertAlmostEqual(ctrl._brake((2.,0),0).vx,initial)

    def test_reverse_stall_is_one_physical_directional_failure(self):
        ctrl=self.control();ctrl.grid=ground(.5)
        # Supply a simple checked reverse leg to isolate outcome detection.
        ctrl.regroup_required=True;ctrl.regroup_goal=(0.,0);ctrl.regroup_origin=(2.,0)
        ctrl.regroup_started=0.
        self.assertTrue(ctrl._compute_regroup((2.,0),0,0.))
        ctrl.motion_clear=lambda *args:True
        for now in np.arange(.1,7.,.1):
            ctrl.tick((2.,0),0,0,float(now),velocity=(0,0,0))
            if ctrl.climb_failures:break
        self.assertEqual(ctrl.climb_failures,1)
        self.assertIn('reverse poor drive',ctrl.attempt_events[-1]['reason'])
        cell=ctrl.grid.world_to_cell(2.,0)
        self.assertFalse(ctrl.grid.direction_allowed(cell,4))
        self.assertTrue(ctrl.grid.direction_allowed(cell,0))

    def test_braking_command_does_not_reverse_the_failed_approach_label(self):
        ctrl=self.control();ctrl._plan((0,0),0,0.)
        ctrl.previous_command=c.Motion(-.4,0.)
        ctrl.finish_attempt('failure','could not settle',(.8,0),0,3.,physical=True)
        cell=ctrl.grid.world_to_cell(.8,0)
        self.assertFalse(ctrl.grid.direction_allowed(cell,0))
        self.assertTrue(ctrl.grid.direction_allowed(cell,4))

    def test_rectangle_recheck_keeps_slope_and_drift_footprint_protections(self):
        ctrl=self.control();g=ctrl.grid
        g.half_length=.3;g.half_width=.12
        lateral=g.world_to_cell(.1,.5)
        g.slope[lateral]=math.tan(math.radians(40))
        self.assertFalse(ctrl.motion_clear((.1,.1),0,0,c.Motion(.4,0)))
        ctrl.velocity_world=np.array([0.,.4])
        self.assertFalse(ctrl.motion_clear((.1,.1),0,.4,c.Motion(.4,0)))
        ctrl.velocity_world=np.zeros(2)
        g.slope[g.world_to_cell(.5,.1)]=math.tan(math.radians(40))
        self.assertFalse(ctrl.motion_clear((.1,.1),0,0,c.Motion(.4,0)))


if __name__ == '__main__':
    unittest.main()
