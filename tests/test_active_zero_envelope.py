"""Active-zero continuation keeps physical guards without requesting repower."""
import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from test_behavior_navigation import a,b,m,ground,observation


class ActiveZeroTests(unittest.TestCase):
    def active(self, c, obs, rows=None, executed=None, entered=1.):
        e=c.executor
        end=(obs.xy[0]+10*math.cos(obs.yaw),obs.xy[1]+10*math.sin(obs.yaw))
        e.start(m.MovementRequest('zero',obs.xy,end,1,.4,obs.time+100),obs.time-5)
        e.state=m.MovementState.RELEASE;e.entered=entered
        e.travel_axis=(math.cos(obs.yaw),math.sin(obs.yaw));e.committed_heading=obs.yaw
        e.executed_traversal=executed or m.ExecutedTraversal('RELEASE',1,obs.yaw,obs.yaw,obs.xy,obs.xy,1.,obs.time,4.)
        e.samples=rows or [(float(t),.1*t,(.1*t,0.)) for t in np.linspace(obs.time-2,obs.time,101)]
        e.uphill_samples=[(t,xy) for t,along,xy in e.samples]
        c.obs=obs;c.configure_active_zero_envelope()
        for t in np.linspace(obs.time-.3,obs.time,16):c.passive_heading.observe(float(t),obs.yaw)
        return c

    def synthetic(self):
        g=ground(33.5)
        c=b.BehaviorTreeAscentController(g,[SimpleNamespace(x=10.,y=0.,label='goal')],mode='characterize')
        obs=replace(observation(5.,velocity=(.05,0.),grade=33.5,turn=False),terrain_gradient=(math.tan(math.radians(33.5)),0.))
        return self.active(c,obs)

    def recorded(self, filename='active_zero_release.json'):
        f=json.loads((Path(__file__).parent/'fixtures'/filename).read_text())
        d=f['guard_event']['terrain_evidence'];r=f['observation']
        g=a.AscentTerrainMap(world_size=d['world_size_m'],cell_size=d['cell_size_m'],origin=tuple(d['grid_origin']),
            footprint_radius=.5028250873088836,clearance=.1,half_length=d['half_length_m'],half_width=d['half_width_m'])
        selection=tuple(slice(i,i+n) for i,n in zip(d['patch_first_cell'],d['patch_shape']))
        for name,values in d['fields'].items():
            target=getattr(g,name);target[selection]=np.asarray(values,dtype=target.dtype)
        c=b.BehaviorTreeAscentController(g,[SimpleNamespace(x=0.,y=13.,label='exterior')],mode='characterize')
        obs=m.MotionObservation((r['x'],r['y']),r['yaw'],(r['velocity_world_x'],r['velocity_world_y']),r['up_z'],
            r['actual_yaw_rate'],r['sim_time'],r['support_inclination_deg'],False,True,
            (r['observed_gradient_x'],r['observed_gradient_y']),tuple(r['body_quaternion_'+k] for k in 'wxyz'))
        axis=f['request_reconstruction']['travel_axis']
        rows=[(q['sim_time'],q['x']*axis[0]+q['y']*axis[1],(q['x'],q['y'])) for q in f['history_2s'] if q['sim_time']>=obs.time-2.-1e-9]
        self.active(c,obs,rows,m.ExecutedTraversal(**f['executed_traversal']),f['release_entered'])
        c.executor.request=replace(c.executor.request,start=tuple(f['request_reconstruction']['start']),
                                   endpoint=tuple(f['request_reconstruction']['endpoint']))
        c.executor.travel_axis=tuple(axis);c.executor.committed_heading=f['request_reconstruction']['committed_heading']
        c.passive_heading.samples=[]
        for q in f['history_2s']: c.passive_heading.observe(q['sim_time'],q['yaw'])
        return c,f

    def test_recorded_guard_replay_preserves_reserve_and_live_map(self):
        c,f=self.recorded();o=c.obs;e=c.executor
        c.active_zero_envelope_enabled=False
        self.assertIn('straight drive',c._release_envelope(o,e.request))
        c.active_zero_envelope_enabled=True
        self.assertTrue(c._owns_active_zero(o,e.request))
        self.assertEqual(c._release_envelope(o,e.request),'')
        event=next(x for x in c.escape_events if x['event']=='active_zero_prospective_admitted')
        self.assertAlmostEqual(event['stopping_distance_m'],f['unchanged_reserve_m'])
        np.testing.assert_allclose(event['corridor_endpoint'],f['guard_event']['corridor_endpoint'])
        self.assertTrue(c._guard(m.Motion(.4,0.))) # Repower still fails.

    def test_powered_grade_does_not_stop_existing_zero(self):
        c=self.synthetic()
        self.assertTrue(c._guard(m.Motion(.4,0.)))
        self.assertEqual(c._release_envelope(c.obs,c.executor.request),'')

    def test_cold_gap_stale_wrong_owner_and_nonfinite_do_not_gain_permission(self):
        for kind in ('cold','gap','stale','missing','drive','reverse','recovery','stop','nonfinite','heading_missing','heading_gap'):
            c=self.synthetic();e=c.executor;o=c.obs
            if kind=='cold':e.entered=o.time
            elif kind=='gap':e.uphill_samples.pop(20);e.uphill_samples.pop(20)
            elif kind=='stale':o=replace(o,time=o.time+.02)
            elif kind=='missing':e.executed_traversal=None
            elif kind=='drive':e.state=m.MovementState.DRIVE_FORWARD
            elif kind=='reverse':e.request=replace(e.request,direction=-1)
            elif kind=='recovery':e.request=replace(e.request,purpose='recovery')
            elif kind=='stop':c.operator_mode='stop'
            elif kind=='heading_missing':c.passive_heading.samples=[]
            elif kind=='heading_gap':del c.passive_heading.samples[3:6]
            else:e.uphill_samples[-1]=(o.time,(float('nan'),0.))
            self.assertFalse(c._owns_active_zero(o,e.request),kind)

    def test_physical_rejections_still_bind(self):
        for kind in ('steep','unknown','stale','hazard','exclusion'):
            c=self.synthetic();g=c.grid;o=c.obs;cell=g.world_to_cell(.4,0.)
            if kind=='steep':g.slope[cell]=math.tan(math.radians(36.))
            elif kind=='unknown':g.known[cell]=False
            elif kind=='stale':g.fresh[cell]=False
            elif kind=='hazard':g.hazard[cell]=True;g.blocked[cell]=True
            else:g.mark_failure('real-failure',(.4,0.),0.,4.,'physical failure')
            self.assertTrue(c._release_envelope(o,c.executor.request),kind)

    def test_guard_queries_do_not_renew_progress_or_deadline(self):
        c=self.synthetic();e=c.executor
        before=(e.started,e.entered,e.request.deadline,e.uphill_exposure,e.uphill_stall_since)
        for _ in range(3):c._release_envelope(c.obs,e.request)
        self.assertEqual(before,(e.started,e.entered,e.request.deadline,e.uphill_exposure,e.uphill_stall_since))
        self.assertEqual(sum(x['event']=='active_zero_prospective_admitted' for x in c.escape_events),1)


if __name__=='__main__':unittest.main()
