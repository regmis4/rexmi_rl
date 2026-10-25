"""Diagnostic ownership and geometry tests; no simulator qualification claim."""
import copy
import importlib
import json
import math
from pathlib import Path
import queue
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from test_behavior_navigation import ground, a

rb = importlib.import_module('rexmi_bt_tests.nav.reviewed_buttons')


class ReviewedButtonsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.flat = ground()

    def session(self, grid=None, initializing=False):
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        s = rb.ReviewedButtons(Path(root.name)/'tickets', copy.deepcopy(grid or self.flat), .02)
        s.initializing = initializing
        s.observe(0., (0.,0.,.5), 0., (0.,0.,0.), 1.)
        s.publish()
        return s

    def ticket(self, s, button='forward', seconds=.1):
        return dict(sequence=s.sequence+1, state_digest=s.state['digest'], button=button,
                    seconds=seconds, review_id='review-8.4', authorization_id='agent2-001', reason='test')

    def test_forward_reverse_and_release_match_shared_teleop(self):
        for name, expected in [('forward',(.4,0.)),('back',(-.4,0.)),('left',(0.,.35)),
                               ('right',(0.,-.35)),('release',(0.,0.))]:
            s=self.session();s.accept(self.ticket(s,name))
            m=s.next_motion(0.,(0.,0.,.5),0.,(0.,0.,0.),1.,lambda:True,queue.SimpleQueue())
            self.assertEqual((m.vx,m.omega),expected)

    def test_stale_duplicate_unapproved_and_duration_tickets_rejected(self):
        s=self.session()
        for fields in [dict(sequence=0),dict(state_digest='stale'),dict(review_id=''),
                       dict(authorization_id=''),dict(seconds=float('nan')),dict(seconds=True),
                       dict(button='forward+left'),dict(seconds=11),dict(button='left',seconds=3),
                       dict(button='release',seconds=21)]:
            t=self.ticket(s);t.update(fields)
            with self.assertRaises(ValueError):s.accept(t)
        ticket=self.ticket(s);s.accept(ticket)
        with self.assertRaises(ValueError):s.accept(ticket)

    def test_pause_is_not_release_or_extra_physics(self):
        s=self.session();q=queue.SimpleQueue()
        # Waiting invokes no policy or physics; interrupt after one poll.
        with patch.object(rb.time,'sleep',side_effect=lambda _:q.put(('stop',0,0))) as sleep:
            self.assertIsNone(s.next_motion(0.,(0.,0.,.5),0.,(0.,0.,0.),1.,lambda:True,q))
            sleep.assert_called_once()
        self.assertEqual(s.sequence,0)
        self.assertEqual(s.current['time'],0.)

    def test_stop_precedes_active_motion(self):
        s=self.session();s.accept(self.ticket(s));q=queue.SimpleQueue();q.put(('stop',0,0))
        self.assertIsNone(s.next_motion(.02,(0.,0.,.5),0.,(0.,0.,0.),1.,lambda:True,q))
        self.assertIn('STOP',s.terminal)

    def test_expiry_preserves_elapsed_and_pauses(self):
        s=self.session();s.accept(self.ticket(s,'release',.1));q=queue.SimpleQueue()
        with patch.object(rb.time,'sleep',side_effect=lambda _:q.put(('stop',0,0))):
            self.assertIsNone(s.next_motion(.1,(.02,0.,.51),0.,(.1,0.,0.),1.,lambda:True,q))
        events=[json.loads(x) for x in (s.directory/'ledger.jsonl').read_text().splitlines()]
        result=next(x for x in events if x['event']=='ticket_outcome')
        self.assertAlmostEqual(result['elapsed'],.1)
        self.assertAlmostEqual(result['height_change'],.01)

    def test_geometry_pause_does_not_vote_failure_or_clear_evidence(self):
        s=self.session();cell=s.grid.world_to_cell(.4,0)
        s.grid.mark_failure('fixture',(.4,0),0.,.3,'physical fixture')
        before=s.grid.failures.copy();s.accept(self.ticket(s));q=queue.SimpleQueue()
        with patch.object(rb.time,'sleep',side_effect=lambda _:q.put(('stop',0,0))):
            s.next_motion(0.,(0.,0.,.5),0.,(0.,0.,0.),1.,lambda:True,q)
        self.assertTrue((before==s.grid.failures).all())
        events=[json.loads(x) for x in (s.directory/'ledger.jsonl').read_text().splitlines()]
        outcome=next(x for x in events if x['event']=='ticket_outcome')
        self.assertFalse(outcome['physical_failure'])

    def test_turn_15_boundary_and_slope_speed(self):
        s=self.session(ground(20.))
        self.assertIn('15',s.guard('left'))
        self.assertEqual(rb.BUTTONS['forward'][0],.4)
        s=self.session(ground(13.))
        self.assertEqual(s.guard('left'),'')

    def test_release_and_yaw_require_fresh_nonhazardous_centre(self):
        s=self.session();cell=s.grid.world_to_cell(0,0);s.grid.fresh[cell]=False
        for name in ('release','left'):self.assertTrue(s.guard(name))
        s.grid.fresh[cell]=True;s.grid.hazard[cell]=True
        for name in ('release','left'):self.assertTrue(s.guard(name))

    def test_physical_fault_wins_over_milestone(self):
        s=self.session();s.milestones=[{},{}]
        s.observe(10.,(0.,12.1,.5),0.,(.81,0.,0.),1.)
        self.assertEqual(len(s.milestones),2)
        self.assertIn('speed',s.terminal)
        s=self.session();s.observe(1.,(0.,0.,.5),0.,(0.,0.,0.),math.cos(math.radians(51)))
        self.assertIn('tilt',s.terminal)

    def test_budgets_and_nonfinite_terminate(self):
        s=self.session();s.observe(300.,(0.,0.,.5),0.,(0.,0.,0.),1.)
        self.assertIn('budget',s.terminal)
        s=self.session();s.sequence=80
        with self.assertRaises(ValueError):s.accept(self.ticket(s))
        s=self.session();s.observe(1.,(math.nan,0.,.5),0.,(0.,0.,0.),1.)
        self.assertIn('nonfinite',s.terminal)

    def downhill(self, degrees=-23., yaw=0., tilt=0.):
        s=self.session(ground(degrees));s.milestones=[{}]
        for t in np.arange(.02,.521,.02):
            s.observe(float(t),(2.,0.,.5),yaw,(.2,0.,0.),math.cos(math.radians(tilt)))
        return s

    def test_descent_yaw_and_combined_buttons_match_manual_contract(self):
        for name, expected in [('left',(0.,.35)),('right',(0.,-.35)),
                               ('forward_left',(.4,.35)),('forward_right',(.4,-.35))]:
            s=self.downhill(tilt=40.)
            self.assertTrue(s.descent_permission())
            self.assertEqual(s.guard(name,profile='descent'),'')
            s.publish();s.accept(dict(self.ticket(s,name,3.),profile='descent'))
            m=s.next_motion(.52,(2.,0.,.5),0.,(.2,0.,0.),math.cos(math.radians(40)),lambda:True,queue.SimpleQueue())
            self.assertEqual((m.vx,m.omega),expected)
        self.assertEqual(s.duration_limit('left','descent'),8.)
        with self.assertRaises(ValueError):s.accept(self.ticket(s,'forward_left',3.02))

    def test_descent_profile_must_be_explicit_even_after_boulder_milestone(self):
        s=self.downhill()
        self.assertTrue(s.descent_permission())
        self.assertIn('normal profile',s.guard('back',profile='descent'))
        self.assertEqual(s.guard('back'),'')
        init=self.session(initializing=True)
        self.assertIn('initialization',init.guard('release',profile='descent'))
        with self.assertRaises(ValueError):init.accept(dict(self.ticket(init,'release'),profile='descent'))
        self.assertIn('15',s.guard('left'))
        self.assertEqual(s.guard('left',profile='descent'),'')
        with self.assertRaises(ValueError):s.accept(self.ticket(s,'left',8.))
        with self.assertRaises(ValueError):s.accept(dict(self.ticket(s),profile='invalid'))
        s.observe(.54,(2.,0.,.5),0.,(.81,0.,0.),1.)
        self.assertIn('0.8',s.terminal)

    def test_release_latch_remains_executable_after_descent_permission_loss(self):
        s=self.downhill();s.publish();s.accept(dict(self.ticket(s,'forward',1.),profile='descent'))
        s.observe(.54,(2.,0.,.5),0.,(.85,0.,0.),1.)
        s.finish(.54,(2.,0.,.5),'speed pause');s.milestones.append({})
        self.assertFalse(s.descent_permission())
        self.assertEqual(s.guard('release',profile='descent'),'')
        self.assertEqual(s.guard('release'),'')
        self.assertIn('RELEASE',s.guard('forward'))

    def test_descent_history_gap_freshness_progress_and_phase_remove_permission(self):
        s=self.downhill();self.assertTrue(s.descent_permission())
        self.assertLess(s.projected_progress(),0.)
        s.observe(.60,(2.,0.,.5),0.,(.2,0.,0.),1.)
        self.assertFalse(s.descent_permission())
        self.assertIsNone(s.projected_progress())
        s=self.downhill();s.grid.fresh[s.grid.world_to_cell(2,0)]=False
        self.assertFalse(s.descent_permission())
        s=self.downhill();s.milestones.append({});self.assertFalse(s.descent_permission())
        self.assertIn('15',s.guard('left'));self.assertEqual(s.duration_limit('left'),2.)
        s=self.downhill()
        for t in np.arange(.54,1.081,.02):s.observe(float(t),(2.,0.,.5),0.,(-.3,0.,0.),1.)
        self.assertGreater(s.projected_progress(),.01);self.assertFalse(s.descent_permission())

    def test_descent_long_ticket_loses_permission_without_normal_fallback(self):
        s=self.downhill();s.publish();s.accept(dict(self.ticket(s,'left',8.),profile='descent'))
        s.current['yaw']=math.pi
        self.assertIn('permission unavailable or ended',s.guard('left',profile='descent'))
        s.finish(.52,(2.,0.,.5),'fixture')
        s.grid=copy.deepcopy(self.flat)
        self.assertEqual(s.guard('left'),'')  # Ordinary gentle-terrain fallback.
        self.assertTrue(s.guard('forward_left',profile='descent'))

    def test_descent_overspeed_release_ownership_survives_ticket_and_phase(self):
        s=self.downhill();s.publish();s.accept(dict(self.ticket(s,'forward',1.),profile='descent'))
        s.observe(.54,(2.,0.,.5),0.,(.85,0.,0.),1.)
        self.assertEqual(s.terminal,'');self.assertTrue(s.descent_overspeed)
        self.assertIn('RELEASE',s.guard('forward',profile='descent'))
        s.finish(.54,(2.,0.,.5),'expiry');s.milestones.append({})
        s.observe(.56,(2.,0.,.5),0.,(.85,0.,0.),1.)
        self.assertEqual(s.terminal,'');self.assertTrue(s.descent_overspeed)
        self.assertIn('RELEASE',s.guard('left',profile='descent'))
        s.observe(.58,(2.,0.,.5),0.,(.79,0.,0.),1.)
        self.assertFalse(s.descent_overspeed)
        s=self.downhill();s.publish();s.accept(dict(self.ticket(s,'forward',1.),profile='descent'));s.observe(.54,(2.,0.,.5),0.,(1.01,0.,0.),1.)
        self.assertIn('1.0',s.terminal)

    def test_descent_rotation_sweep_retains_terrain_hazard_and_direction_bans(self):
        for kind, text in [('slope','35'),('hazard','obstacle'),('known','unobserved')]:
            s=self.downhill();cell=s.grid.world_to_cell(2.1,.1)
            getattr(s.grid,kind)[cell]=math.tan(math.radians(36)) if kind=='slope' else (kind=='hazard')
            self.assertTrue(s.guard('forward_left',profile='descent'),text)
        s=self.downhill();s.grid.mark_failure('fixture',(2.3,0.),0.,.6,'physical failure')
        self.assertIn('exclu',s.guard('forward',profile='descent'))
        s=self.downhill();s.grid.no_go[s.grid.world_to_cell(2.3,0.)]=True
        self.assertIn('no-go',s.guard('forward',profile='descent'))

    def test_descent_entire_command_cannot_cross_into_uphill(self):
        s=self.downhill()
        # The current axis is downhill; the second half of the intended leg is not.
        g=s.grid;c=g.world_to_cell(2.3,0.);g.gx[c]=.1
        self.assertTrue(s.descent_permission())
        self.assertIn('uphill grade',s.guard('forward',profile='descent'))

    def test_descent_rotation_checks_body_corner_without_extra_turn_reserve(self):
        s=self.downhill();g=s.grid
        g.half_length=.47;g.half_width=.27
        # Place hazard outside original support circle but inside rotated body.
        cell=g.world_to_cell(2.,.5);g.hazard[cell]=True
        self.assertIn('rotation envelope',s.guard('left',profile='descent'))

    def test_descent_curved_translation_checks_off_axis_obstacle(self):
        s=self.downhill();g=s.grid
        # Chord ending at ~(.48,.11) relative to start; obstacle on outside arc.
        cell=g.world_to_cell(2.45,.4);g.hazard[cell]=True
        self.assertIn('obstacle',s.guard('forward_left',profile='descent'))

    def test_map_snapshots_never_overwritten(self):
        s=self.session();first=(s.directory/s.state['map_file']).read_bytes()
        s.publish()
        self.assertEqual((s.directory/'state-001.npz').read_bytes(),first)
        self.assertEqual(len(list(s.directory.glob('state-*.npz'))),2)

    def test_settling_requires_active_release_and_continuous_dwell(self):
        s=self.session();s.milestones=[{}, {}, {}]
        with patch.object(s.grid,'support_reason',return_value=''), patch.object(s,'guard',return_value=''):
            s.observe(1.,(0.,12.1,.5),0.,(0.,0.,0.),1.)
            self.assertIsNone(s.settle_since)
            s.active={'button':'release'}
            s.observe(2.,(0.,12.1,.5),0.,(0.,0.,0.),1.)
            s.observe(2.4,(0.,12.1,.5),0.,(.13,0.,0.),1.)
            self.assertIsNone(s.settle_since)
            s.observe(3.,(0.,12.1,.5),0.,(0.,0.,0.),1.)
            s.observe(3.6,(0.,12.1,.5),0.,(0.,0.,0.),1.)
            self.assertIn('settled',s.terminal)

    def test_vertical_motion_or_guard_rejection_prevents_settling(self):
        s=self.session();s.milestones=[{}, {}, {}];s.active={'button':'release'}
        with patch.object(s,'guard',return_value=''):
            for t in (1.,1.6):s.observe(t,(0.,12.1,.5),0.,(0.,0.,-.13),1.)
        self.assertEqual(s.terminal,'');self.assertIsNone(s.settle_since)
        with patch.object(s,'guard',return_value='blocked drift'):
            for t in (2.,2.6):s.observe(t,(0.,12.1,.5),0.,(0.,0.,0.),1.)
        self.assertEqual(s.terminal,'');self.assertIsNone(s.settle_since)

    def test_initialization_only_release_without_future_stopping_fan(self):
        s=self.session(initializing=True)
        self.assertEqual(s.guard('release'),'')
        for name in ('forward','back','left','right'):self.assertTrue(s.guard(name))
        s.accept(self.ticket(s,'release',3.))
        s.observe(.48,(.1,0.,.5),0.,(.94,0.,-.2),1.)
        self.assertEqual(s.terminal,'')
        with patch.object(s,'corridor_reason',side_effect=AssertionError('initialization must not use future fan')):
            self.assertEqual(s.guard('release'),'')

    def test_initialization_completion_arms_normal_guards_once(self):
        s=self.session(initializing=True);s.accept(self.ticket(s,'release',3.))
        for t in (1.,1.6):s.observe(t,(.1,0.,.5),0.,(.05,0.,0.),1.)
        self.assertFalse(s.initializing);self.assertIsNone(s.active)
        self.assertIn('permanently',s.pause_reason)
        s.observe(1.62,(.1,0.,.5),0.,(.81,0.,0.),1.)
        self.assertIn('0.8',s.terminal)

    def test_initialization_deadline_and_region_and_speed_caps(self):
        for xyz, velocity, now, message in [((0.,0.,.5),(0.,0.,-.13),3.,'3 seconds'),
                ((.51,0.,.5),(0.,0.,0.),1.,'spawn region'),
                ((0.,0.,.5),(1.11,0.,0.),1.,'1.1')]:
            s=self.session(initializing=True);s.accept(self.ticket(s,'release',3.))
            s.observe(now,xyz,0.,velocity,1.)
            self.assertIn(message,s.terminal)

    def test_initialization_requires_fresh_whole_footprint(self):
        s=self.session(initializing=True)
        cells=s.grid._support_cells((0.,0.));s.grid.fresh[cells[0][0],cells[1][0]]=False
        self.assertIn('fresh',s.guard('release'))

    def recorded_state8(self, sequence=8):
        data=np.load(Path(__file__).parent/f'fixtures/reviewed_buttons_state{sequence}.npz')
        meta=json.loads(str(data['metadata'].item()))
        g=a.AscentTerrainMap(**{k:meta[k] for k in ('origin','world_size','cell_size',
            'footprint_radius','clearance','half_length','half_width')})
        g.uphill_limit_deg=meta['uphill_limit_deg'];g.side_limit_deg=meta['side_limit_deg']
        g.known=data['prior_known'].copy();g.surface=data['prior_surface'].copy()
        g.terrain_prior=dict(meta['prior'],file_sha256=meta['prior_file_sha256'])
        s=self.session(g)
        for key in data.files:
            if key not in ('metadata','prior_known','prior_surface'):setattr(s.grid,key,data[key].copy())
        s.grid.now=meta['state']['time']
        v=meta['state'];s.heading.samples=[]
        for now,yaw in meta['history']:s.heading.observe(now,yaw)
        s.observe(v['time'],v['position'],v['yaw'],v['velocity'],math.cos(math.radians(v['tilt'])))
        return s

    def stopping_fan_cells(self,s):
        sf=importlib.import_module('rexmi_bt_tests.nav.swept_footprint')
        v=s.current;xy=np.array(v['position'][:2]);distance=v['speed']**2/.4+.1
        travel=math.atan2(v['velocity'][1],v['velocity'][0])
        low,high=s.heading.offsets(v['time'],v['yaw'],v['speed']/.2)
        count=max(1,math.ceil((high-low)/.04));reserve=distance*(high-low)/count/2
        cells=set()
        for angle in np.linspace(low,high,count+1):
            end=xy+distance*np.array([math.cos(travel+angle),math.sin(travel+angle)])
            cells.update(zip(*sf.oriented_swept_cells(s.grid,xy,end,v['yaw'],(low,high),reserve)))
        return cells

    def test_recorded_circular_slope_corner_does_not_block_physical_stopping_fan(self):
        s=self.recorded_state8(31);v=s.current;g=s.grid
        xy=np.array(v['position'][:2]);end=xy+(v['speed']**2/.4+.1)*np.array(v['velocity'][:2])/v['speed']
        self.assertIn('35',g.drift_segment_reason(xy,end,observation_support=s.forecast_support()))
        flagged=g.world_to_cell(8.1,-1.5)
        self.assertNotIn(flagged,self.stopping_fan_cells(s))
        self.assertLess(max(math.degrees(math.atan(g.slope[c])) for c in self.stopping_fan_cells(s)),35.)
        self.assertEqual(s.guard('release'),'')
        offsets=s.heading.offsets(v['time'],v['yaw'],v['speed']/.2)
        # Other callers still keep their existing circular ground precheck.
        self.assertIn('35',rb.oriented_stopping_reason(g,xy,end,v['yaw'],offsets,commanded=False,
                      observation_support=s.forecast_support()))

    def test_ground_admission_logged_once_on_emission_never_on_guard_probe(self):
        s=self.recorded_state8(31);v=s.current
        for _ in range(3):self.assertEqual(s.guard('release'),'')
        self.assertFalse(s.ground_admission_tickets)
        s.publish();s.accept(self.ticket(s,'release',.2))
        for _ in range(2):
            motion=s.next_motion(v['time'],v['position'],v['yaw'],v['velocity'],
                math.cos(math.radians(v['tilt'])),lambda:True,queue.SimpleQueue())
            self.assertEqual((motion.vx,motion.omega),(0.,0.))
        events=[json.loads(x) for x in (s.directory/'ledger.jsonl').read_text().splitlines()]
        admissions=[e for e in events if e['event']=='oriented_ground_admission']
        self.assertEqual(len(admissions),1)
        self.assertEqual(admissions[0]['contexts'][0]['context'],'actual stopping')
        self.assertIsNotNone(admissions[0]['contexts'][0]['heading_offsets'])

    def test_oriented_slope_fallback_retains_physical_and_observation_rejections(self):
        for kind in ('slope','unknown','hazard','unsupported','conflict'):
            s=self.recorded_state8(31);g=s.grid;xy=s.current['position'][:2]
            future=self.stopping_fan_cells(s)-set(zip(*g._support_cells(xy)))
            self.assertTrue(future)
            c=max(future,key=lambda c:math.dist(g.cell_to_world(*c),xy))
            if kind=='slope':g.slope[c]=math.tan(math.radians(35.01))
            elif kind=='unknown':g.known[c]=False
            elif kind=='hazard':g.hazard[c]=True
            else:
                g.fresh[c]=False
                if kind=='conflict':g.surface[c]=s.prior_surface[c]+.081
                else:
                    mask=s.prior_support.copy();mask[c]=False;mask.setflags(write=False);s.prior_support=mask
            self.assertTrue(s.guard('release'),kind)

    def test_oriented_slope_fallback_requires_history_and_preserves_exclusions(self):
        s=self.recorded_state8(31);s.heading.samples=[]
        self.assertIn('history',s.guard('release'))
        for kind in ('no_go','direction','occupied'):
            s=self.recorded_state8(31);v=s.current;g=s.grid;xy=np.array(v['position'][:2])
            end=xy+(v['speed']**2/.4+.1)*np.array(v['velocity'][:2])/v['speed']
            c=g.world_to_cell(*(xy if kind=='occupied' else end))
            if kind in ('no_go','occupied'):g.no_go[c]=True
            else:g.mark_failure('fixture',end,math.atan2(end[1]-xy[1],end[0]-xy[0]),v['time'],'physical fixture')
            self.assertTrue(s.guard('release'),kind)

    def test_current_body_fallback_is_consistent_but_does_not_authorize_rotation(self):
        sf=importlib.import_module('rexmi_bt_tests.nav.swept_footprint')
        s=self.recorded_state8(31);v=s.current;g=s.grid;xy=v['position'][:2]
        rectangle=set(zip(*sf.oriented_swept_cells(g,xy,xy,v['yaw'])))
        circular=set(zip(*g._support_cells(xy)))
        c=next(iter(circular-rectangle));g.slope[c]=math.tan(math.radians(36))
        self.assertIn('35',g.support_reason(xy))
        self.assertEqual(s.current_support_reason(),'')
        s.milestones=[{}]
        for dt in np.arange(.02,.541,.02):
            s.observe(v['time']+float(dt),v['position'],v['yaw'],v['velocity'],math.cos(math.radians(v['tilt'])))
        self.assertTrue(s.descent_permission())
        self.assertIn('35',s.guard('left',profile='descent'))
        # The actual oriented body must remain freshly observed and <=35.
        c=g.world_to_cell(*xy);g.fresh[c]=False
        self.assertTrue(s.current_support_reason())
        g.fresh[c]=True;g.slope[c]=math.tan(math.radians(36))
        self.assertIn('35',s.current_support_reason())

    def test_exact_recorded_stale_veto_is_admitted_without_faking_freshness(self):
        s=self.recorded_state8();before=s.grid.fresh.copy()
        self.assertIn('stale',s.guard('release',static_support=False))
        self.assertEqual(s.guard('release'),'')
        self.assertEqual(s.guard('forward'),'')
        self.assertTrue((before==s.grid.fresh).all())
        self.assertFalse(s.prior_support.flags.writeable)
        self.assertFalse(s.prior_surface.flags.writeable)

    def test_recorded_real_forward_boulder_still_rejected(self):
        s=self.recorded_state8(4)
        self.assertIn('obstacle',s.guard('forward',static_support=False))
        self.assertIn('obstacle',s.guard('forward'))
        self.assertEqual(s.guard('release'),'')

    def test_recorded_unknown_outside_prior_conflict_and_hazard_remain_rejected(self):
        for field in ('unknown','outside_prior','conflict','hazard','no_go','too_steep'):
            s=self.recorded_state8();cell=s.grid.world_to_cell(10.9,.5)
            if field=='unknown':s.grid.known[cell]=False
            elif field=='outside_prior':
                # Build another immutable original-support fixture with a hole.
                mask=s.prior_support.copy();mask[cell]=False;mask.setflags(write=False);s.prior_support=mask
            elif field=='conflict':s.grid.surface[cell]=s.prior_surface[cell]+.081
            elif field=='hazard':s.grid.hazard[cell]=True
            elif field=='no_go':
                # Exclusion on the actual centre corridor, not only body edge.
                centre=np.asarray(s.current['position'][:2])+.3*np.asarray(s.current['velocity'][:2])/s.current['speed']
                s.grid.no_go[s.grid.world_to_cell(*centre)]=True
            else:s.grid.slope[cell]=math.tan(math.radians(36))
            self.assertTrue(s.guard('release'),field)

    def test_static_survey_does_not_replace_actual_footprint_freshness(self):
        s=self.recorded_state8();s.grid.fresh[s.grid.world_to_cell(*s.current['position'][:2])]=False
        self.assertIn('stale',s.guard('release'))

    def test_retained_or_refreshed_surveys_are_not_static_ground_truth(self):
        for prior in ({'kind':'retained_sensor_survey','file_sha256':'a','geometry_id':'b'},
                      {'kind':'simulator_development','file_sha256':'a','geometry_id':'b','parent_geometry_id':'c'},
                      {'kind':'simulator_development','geometry_id':'b'}):
            g=copy.deepcopy(self.flat);g.terrain_prior=prior
            s=self.session(g);self.assertFalse(s.static_prior);self.assertIsNone(s.forecast_support())

    def test_circular_queries_accept_mask_without_bypassing_hazard_or_exclusion(self):
        g=copy.deepcopy(self.flat);g.fresh[:]=False;mask=np.ones_like(g.fresh)
        self.assertFalse(g.segment_clear((0.,0.),(.5,0.)))
        self.assertTrue(g.segment_clear((0.,0.),(.5,0.),observation_support=mask))
        self.assertTrue(g.drift_segment_reason((0.,0.),(.5,0.)))
        self.assertEqual(g.drift_segment_reason((0.,0.),(.5,0.),observation_support=mask),'')
        g.mark_failure('fixture',(.4,0.),0.,.2,'physical fixture')
        self.assertFalse(g.segment_clear((0.,0.),(.5,0.),observation_support=mask))
        self.assertTrue(g.drift_segment_reason((0.,0.),(.5,0.),observation_support=mask))


if __name__ == '__main__':unittest.main()
