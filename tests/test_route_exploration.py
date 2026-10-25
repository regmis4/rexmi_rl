"""Characterization scheduling cannot manufacture traversal evidence."""
import copy
import importlib
import hashlib
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from test_behavior_navigation import b, ground

e = importlib.import_module('rexmi_bt_tests.nav.route_exploration')


def corridor(start=(0.,0.), end=(4.,0.)):
    return dict(start=list(start),end=list(end),heading=math.atan2(end[1]-start[1],end[0]-start[0]))


class RouteExplorationTests(unittest.TestCase):
    def control(self):
        c = b.BehaviorTreeAscentController(ground(20),[SimpleNamespace(x=4.,y=0.,label='goal')],mode='characterize')
        c.grid.terrain_prior = dict(geometry_id='geometry',radius_m=5.)
        c.configure_exploration(dict(records=[dict(corridor=corridor())]))
        return c

    def candidate(self, cost=10., path=None, complete=True):
        path = path or [(0.,0.),(4.,0.)]
        return ('unvalidated',cost,path,complete,path,[],None,'key',0.,dict(score=cost))

    def test_subcell_shift_and_detour_do_not_make_novel_climb(self):
        g = ground(20)
        first = e.first_climb(g,[(0.,0.),(4.,0.)])
        shifted = e.first_climb(g,[(.05,.04),(3.9,.04)])
        detour = e.first_climb(g,[(0.,-1.),(0.,0.),(4.,0.),(4.,1.)])
        self.assertTrue(e.same_corridor(first,shifted))
        self.assertTrue(e.same_corridor(first,detour))
        split=e.first_climb(g,[(x,0.) for x in np.arange(0.,4.1,.5)])
        self.assertTrue(e.same_corridor(first,split))

    def test_distinct_heading_or_parallel_corridor_remains_new(self):
        self.assertFalse(e.same_corridor(corridor(),corridor((0.,0.),(4.,2.))))
        self.assertFalse(e.same_corridor(corridor(),corridor((0.,1.3),(4.,1.3))))

    def test_touching_and_crossing_endpoints_are_not_enough(self):
        self.assertFalse(e.same_corridor(corridor(),corridor((3.1,0.),(5.,0.))))
        # The lines intersect but diverge beyond the 1.2 m band over overlap.
        self.assertFalse(e.same_corridor(corridor((0.,0.),(30.,0.)),corridor((0.,-2.),(30.,2.))))
        detail=e.corridor_overlap(corridor(),corridor((1.,.1),(3.,.1)))
        self.assertAlmostEqual(detail[0]['overlap_m'],2.)
        self.assertAlmostEqual(detail[0]['reference_fraction'],.5)

    def test_default_selection_and_after_initial_attempt_use_original_cost(self):
        c=self.control();covered=self.candidate(1.);new=self.candidate(2.,[(0.,0.),(4.,2.)])
        c.exploration_coverage=None
        self.assertIs(c._select_route_candidate([new,covered],(0.,0.),0.),covered)
        c.configure_exploration(dict(records=[dict(corridor=corridor())]));c.attempt_number=1
        self.assertIs(c._select_route_candidate([new,covered],(0.,0.),0.),covered)

    def test_selection_uses_lowest_unchanged_cost_among_untested(self):
        c=self.control();old=self.candidate(1.);new=self.candidate(5.,[(0.,0.),(4.,2.)]);other=self.candidate(7.,[(0.,0.),(4.,-2.)])
        with patch.object(b,'motion_route_reason',return_value=''),patch.object(b,'observed_edge_reason',return_value=''):
            selected=c._select_route_candidate([old,other,new],(0.,0.),0.)
        self.assertIs(selected,new)
        self.assertEqual([old[-1]['score'],new[-1]['score'],other[-1]['score']],[1.,5.,7.])
        self.assertTrue(old[-1]['exploration']['covered'])

    def test_no_eligible_untested_does_not_silently_select_old(self):
        c=self.control()
        self.assertIsNone(c._select_route_candidate([self.candidate()],(0.,0.),0.))
        self.assertIn('no eligible untested',c.planning_failure)

    def test_complete_route_retreat_and_freshness_remain_required(self):
        for kind in ('incomplete','retreat','fresh','direction'):
            c=self.control();new=self.candidate(path=[(0.,0.),(4.,2.)],complete=kind!='incomplete')
            with patch.object(b,'motion_route_reason',return_value=''), \
                 patch.object(b,'observed_edge_reason',return_value='fresh observations missing' if kind=='fresh' else 'direction excluded' if kind=='direction' else ''), \
                 patch.object(c,'_retreat_exists',return_value=kind!='retreat'):
                self.assertIsNone(c._select_route_candidate([new],(0.,0.),0.),kind)

    def test_selection_never_changes_mission_evidence(self):
        c=self.control();arrays=[a.copy() for a in (c.grid.no_go,c.grid.failures,c.grid.successes,c.grid.approach_blocked)]
        c._select_route_candidate([self.candidate()],(0.,0.),0.)
        for old,current in zip(arrays,(c.grid.no_go,c.grid.failures,c.grid.successes,c.grid.approach_blocked)):
            np.testing.assert_array_equal(old,current)
        self.assertFalse(c.evidence.counts)
        self.assertFalse(c.evidence.completed)

    def test_qualified_or_started_mission_cannot_configure_exploration(self):
        c=self.control();c.mode='qualified'
        with self.assertRaises(ValueError):c.configure_exploration(dict(records=[{}]))
        c.mode='characterize';c.attempt_number=1
        with self.assertRaises(ValueError):c.configure_exploration(dict(records=[{}]))

    def history(self, directory, c):
        source={f'source/rexmi_rl/nav/{name}.py':'hash' for name in ('movement','swept_footprint','ascent_map','observed_map')}
        source['source/rexmi_rl/drive_adapter.py']='hash'
        provenance=dict(policy_sha256='policy',seed=42,sources=source,
            arguments=dict(ascent_mode='characterize',task='task',spawn_x=0.,spawn_y=0.,spawn_z=1.,spawn_yaw=0.,
                           cruise_speed=.4,max_uphill_deg=33.,max_side_slope_deg=25.),
            terrain_prior=dict(geometry_id='geometry'),capability_profile=dict(execution_protocol='old',speed=.4))
        events=[dict(event='route_comparison',candidates=[dict(selected=True,complete=True,route_path=[(0.,0.),(4.,0.)])]),
                dict(event='approach_start',attempt_id='one'),
                dict(event='approach_outcome',attempt_id='one',physical=True,outcome='failure',
                     executed_traversal=dict(maneuver='DRIVE_FORWARD',duration_s=3.,start_position=[.1,0.],last_position=[2.5,0.],heading=0.))]
        materials=dict(friction=.8)
        for suffix,data in [('provenance.json',provenance),('materials.json',materials)]:
            (directory/('run.'+suffix)).write_text(json.dumps(data))
        (directory/'run.events.jsonl').write_text('\n'.join(map(json.dumps,events))+'\n')
        return provenance,materials,events

    def test_history_requires_matching_recorded_sources_configuration_and_geometry(self):
        c=self.control()
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);p,materials,_=self.history(directory,c)
            coverage=e.load_coverage(directory,c.grid,p,materials)
            self.assertEqual(len(coverage['records']),1)
            self.assertIn('not recorded',coverage['limitation'])
            for field in ('movement','policy','seed','capability','materials','geometry','mode'):
                changed=copy.deepcopy(p);props=copy.deepcopy(materials)
                if field=='movement':changed['sources']['source/rexmi_rl/nav/movement.py']='new'
                if field=='policy':changed['policy_sha256']='new'
                if field=='seed':changed['seed']=43
                if field=='capability':changed['capability_profile']['speed']=.5
                if field=='materials':props['friction']=.1
                if field=='geometry':c.grid.terrain_prior['geometry_id']='new'
                if field=='mode':changed['arguments']['ascent_mode']='qualified'
                with self.assertRaises(ValueError,msg=field):e.load_coverage(directory,c.grid,changed,props)
                c.grid.terrain_prior['geometry_id']='geometry'

    def test_reviewed_legacy_history_is_labeled_and_all_other_checks_remain(self):
        c=self.control()
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);old,materials,_=self.history(directory,c)
            current=copy.deepcopy(old)
            for name in ('movement','ascent_behavior'):
                key='source/rexmi_rl/nav/'+name+'.py'
                live=Path(e.__file__).with_name(name+'.py').read_bytes()
                archived=directory/'run.sources'/key;archived.parent.mkdir(parents=True,exist_ok=True)
                archived.write_bytes(live+b'\n# archived fixture revision\n')
                old['sources'][key]=hashlib.sha256(archived.read_bytes()).hexdigest()
                current['sources'][key]=hashlib.sha256(live).hexdigest()
            pair=tuple(old['sources']['source/rexmi_rl/nav/'+name+'.py'] for name in ('movement','ascent_behavior'))
            current['capability_profile'].update(e.FORWARD_SPEED_PROFILE)
            current['capability_profile']['execution_protocol']='staged-teleop-v27-checked-forward-speed'
            (directory/'run.provenance.json').write_text(json.dumps(old))
            with self.assertRaises(ValueError):e.load_coverage(directory,c.grid,current,materials)
            with patch.object(e,'LEGACY_SPEED_HISTORY',{pair}):
                result=e.load_coverage(directory,c.grid,current,materials)
                self.assertEqual(result['execution_scope'],'legacy_executor_scheduling_witness')
                self.assertEqual(result['records'][0]['execution_scope'],result['execution_scope'])
                self.assertFalse(result['records'][0]['probability_vote'])
                self.assertFalse(result['records'][0]['exclusion_vote'])
                e.verify_guard_source(directory,old,legacy=True)
                for kind in ('protocol','selector','speed','map','current_source','archive'):
                    changed=copy.deepcopy(current)
                    if kind=='protocol':changed['capability_profile']['execution_protocol']='future'
                    elif kind=='selector':changed['capability_profile']['forward_speed_floor_mps']=.05
                    elif kind=='speed':changed['capability_profile']['speed']=.5
                    elif kind=='map':changed['sources']['source/rexmi_rl/nav/ascent_map.py']='other'
                    elif kind=='current_source':changed['sources']['source/rexmi_rl/nav/movement.py']='other'
                    else:(directory/'run.sources/source/rexmi_rl/nav/movement.py').write_text('tampered')
                    with self.assertRaises(ValueError,msg=kind):e.load_coverage(directory,c.grid,changed,materials)

    def test_censored_and_unexecuted_selection_do_not_establish_coverage(self):
        c=self.control()
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);p,materials,events=self.history(directory,c)
            for change in ('censored','reverse','too_short'):
                data=copy.deepcopy(events)
                if change=='censored':data[-1]['outcome']='censored';data[-1]['physical']=False
                if change=='reverse':data[-1]['executed_traversal']['maneuver']='DRIVE_REVERSE'
                if change=='too_short':data[-1]['executed_traversal']['last_position']=[.4,0.]
                (directory/'run.events.jsonl').write_text('\n'.join(map(json.dumps,data))+'\n')
                with self.assertRaises(ValueError,msg=change):e.load_coverage(directory,c.grid,p,materials)

    def test_v28_legacy_geometry_requires_exact_archives_and_profile(self):
        c=self.control()
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);old,materials,_=self.history(directory,c)
            current=copy.deepcopy(old);archives={};hashes={}
            for name in ('movement','ascent_behavior','ascent_map','swept_footprint'):
                key='source/rexmi_rl/nav/'+name+'.py'
                live=Path(e.__file__).with_name(name+'.py').read_bytes()
                archived=directory/'run.sources'/key;archived.parent.mkdir(parents=True,exist_ok=True)
                archived.write_bytes(live+b'\n# reviewed archive fixture\n')
                hashes[name]=hashlib.sha256(archived.read_bytes()).hexdigest()
                old['sources'][key]=hashes[name];current['sources'][key]=hashlib.sha256(live).hexdigest()
                archives[name]=archived
            current['capability_profile'].update(e.FORWARD_SPEED_PROFILE,**e.DIRECTIONAL_EXIT_PROFILE)
            current['capability_profile']['execution_protocol']='staged-teleop-v28-directional-recovery-exit'
            (directory/'run.provenance.json').write_text(json.dumps(old))
            with patch.object(e,'LEGACY_SPEED_HISTORY',{(hashes['movement'],hashes['ascent_behavior'])}), \
                    patch.object(e,'LEGACY_EXIT_GEOMETRY',{k:hashes[k] for k in ('ascent_map','swept_footprint')}):
                result=e.load_coverage(directory,c.grid,current,materials)
                self.assertEqual(result['execution_scope'],'legacy_executor_scheduling_witness')
                for name in ('ascent_map','swept_footprint'):
                    data=archives[name].read_bytes();archives[name].write_bytes(data+b'tamper')
                    with self.assertRaises(ValueError):e.load_coverage(directory,c.grid,current,materials)
                    archives[name].write_bytes(data)
                    altered=copy.deepcopy(old);altered['sources']['source/rexmi_rl/nav/'+name+'.py']='unreviewed'
                    (directory/'run.provenance.json').write_text(json.dumps(altered))
                    with self.assertRaises(ValueError):e.load_coverage(directory,c.grid,current,materials)
                    (directory/'run.provenance.json').write_text(json.dumps(old))
                for value in (None,'different contract'):
                    changed=copy.deepcopy(current)
                    changed['capability_profile']['directional_recovery_exit_contract']=value
                    with self.assertRaises(ValueError):e.load_coverage(directory,c.grid,changed,materials)

    def test_completed_success_uses_actual_climb_chunk_before_exterior(self):
        c=self.control()
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);p,materials,events=self.history(directory,c)
            climb=events[-1]['executed_traversal']
            floor=dict(climb,start_position=[-3.,0.],last_position=[-1.,0.])
            exterior=dict(climb,start_position=[5.,0.],last_position=[7.,0.])
            movements=[dict(event='movement_outcome',action_id=str(i),outcome='success',physical=False,
                            executed_traversal=chunk) for i,chunk in enumerate((floor,climb,exterior))]
            outcome=dict(events[-1],outcome='success',physical=False,executed_traversal=None)
            sequence=events[:2]+movements+[outcome]
            path=directory/'run.events.jsonl'
            path.write_text('\n'.join(map(json.dumps,sequence))+'\n')
            result=e.load_coverage(directory,c.grid,p,materials)
            self.assertEqual(len(result['records']),1)
            self.assertEqual(result['records'][0]['outcome'],'success')
            for duplicate in (movements[-1],outcome):
                path.write_text('\n'.join(map(json.dumps,sequence+[duplicate]))+'\n')
                # An orphan movement after a closed attempt cannot add coverage.
                if duplicate.get('event')=='movement_outcome':
                    self.assertEqual(len(e.load_coverage(directory,c.grid,p,materials)['records']),1)
                else:
                    with self.assertRaises(ValueError):e.load_coverage(directory,c.grid,p,materials)

    def entrance_fixture(self):
        route=dict(start=[1.,0.],end=[4.,0.],heading=0.,approach_start=[0.,0.],approach_end=[5.,0.])
        executed=dict(maneuver='DRIVE_FORWARD',duration_s=2.,heading=0.,start_position=[0.,0.],last_position=[.8,0.],
                      first_emitted_time=2.,interval_end_time=4.)
        reason='remaining straight corridor unavailable from measured pose'
        outcome=dict(attempt_id='one',outcome='censored',physical=False,reason=reason,start_time=1.,end_time=4.,
                     executed_traversal=executed,position=[.8,0.])
        movement=dict(action_id='one:3',outcome='censored',physical=False,reason=reason,time=4.,
                      executed_traversal=executed,position=[.8,0.])
        row=dict(sim_time='2.',x='0.',y='0.',target_x='5.',target_y='0.',vx_cmd='.4',omega_cmd='0.',
                 action_id='one:3',route_purpose='mission',movement_state='DRIVE_FORWARD')
        return route,outcome,[movement],[row]

    def test_entrance_is_separate_from_characterized_climb(self):
        route,outcome,movements,rows=self.entrance_fixture()
        record=e.entrance_record(route,outcome,movements,rows)
        self.assertEqual(record['stage'],'entrance_attempted_climb_uncharacterized')
        self.assertFalse(record['probability_vote']);self.assertFalse(record['exclusion_vote'])
        self.assertAlmostEqual(record['advance_along_approach_m'],.8)

    def test_floor_interruption_reverse_and_lateral_motion_do_not_count(self):
        for kind in ('floor','operator','stale','reverse','lateral','wrong_direction'):
            route,outcome,movements,rows=self.entrance_fixture()
            if kind=='floor':route['start']=[3.,0.]
            if kind=='operator':outcome['reason']='operator stop'
            if kind=='stale':outcome['reason']='fresh observations missing'
            if kind=='reverse':outcome['executed_traversal']['maneuver']='DRIVE_REVERSE'
            if kind=='lateral':outcome['executed_traversal']['last_position']=[.2,.8]
            if kind=='wrong_direction':outcome['executed_traversal']['heading']=.4
            self.assertIsNone(e.entrance_record(route,outcome,movements,rows),kind)

    def test_invalid_terminal_csv_and_identity_evidence_rejects(self):
        for kind in ('missing_movement','duplicate_movement','missing_row','duplicate_row','action','target','pose','nan','velocity','purpose','interval','terminal_position'):
            route,outcome,movements,rows=self.entrance_fixture()
            if kind=='missing_movement':movements=[]
            if kind=='duplicate_movement':movements*=2
            if kind=='missing_row':rows=[]
            if kind=='duplicate_row':rows*=2
            if kind=='action':rows[0]['action_id']='another'
            if kind=='target':rows[0]['target_x']='3.'
            if kind=='pose':rows[0]['x']='.3'
            if kind=='nan':rows[0]['y']='nan'
            if kind=='velocity':rows[0]['vx_cmd']='0.'
            if kind=='purpose':rows[0]['route_purpose']='adjust'
            if kind=='interval':outcome['executed_traversal']['interval_end_time']=10.
            if kind=='terminal_position':movements[0]['position']=[float('nan'),0.]
            with self.assertRaises(ValueError,msg=kind):e.entrance_record(route,outcome,movements,rows)

    def test_entrance_guard_archive_is_required_and_must_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);key='source/rexmi_rl/nav/ascent_behavior.py'
            source=Path(b.__file__).read_bytes();p=dict(sources={key:hashlib.sha256(source).hexdigest()})
            with self.assertRaises(ValueError):e.verify_guard_source(directory,p)
            destination=directory/'run.sources'/key;destination.parent.mkdir(parents=True);destination.write_bytes(source)
            e.verify_guard_source(directory,p)
            destination.write_bytes(source+b'\n# changed\n')
            with self.assertRaises(ValueError):e.verify_guard_source(directory,p)
            changed=source.replace(b"distance = math.dist(obs.xy, request.endpoint)",b"distance = 0.")
            destination.write_bytes(changed);p['sources'][key]=hashlib.sha256(changed).hexdigest()
            with self.assertRaises(ValueError):e.verify_guard_source(directory,p)

    def test_selector_skips_attempted_entrance_without_climb_or_failure_vote(self):
        c=self.control();c.grid.gy=c.grid.gx.copy()
        old=self.candidate(1.);entrance=self.candidate(5.,[(0.,0.),(4.,2.)]);next_route=self.candidate(7.,[(0.,0.),(2.,4.)])
        c.exploration_coverage['entrance_attempts']=[dict(corridor=e.first_climb(c.grid,entrance[2]))]
        with patch.object(b,'motion_route_reason',return_value=''),patch.object(b,'observed_edge_reason',return_value=''),patch.object(c,'_retreat_exists',return_value=True):
            self.assertIs(c._select_route_candidate([old,entrance,next_route],(0.,0.),0.),next_route)
        self.assertTrue(entrance[-1]['exploration']['entrance_attempted'])
        self.assertFalse(entrance[-1]['exploration']['covered'])
        self.assertFalse(c.evidence.counts)

    def test_merge_keeps_entrance_and_completed_climb_separate(self):
        a=dict(run='a',records=[dict(attempt_id='a')],entrance_attempts=[])
        b=dict(run='b',records=[],entrance_attempts=[dict(attempt_id='b')])
        merged=e.merge_coverage([a,b])
        self.assertEqual(merged['records'],a['records']);self.assertEqual(merged['entrance_attempts'],b['entrance_attempts'])
        with self.assertRaises(ValueError):e.merge_coverage([a,a])


if __name__=='__main__':unittest.main()
