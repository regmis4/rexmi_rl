"""Rejection evidence is contemporaneous, detached, bounded and observational."""
import json
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_behavior_navigation import b,m,ground,observation


class GuardEvidenceTests(unittest.TestCase):
    def controller(self):
        c=b.BehaviorTreeAscentController(ground(),[SimpleNamespace(x=3.,y=0.,label='goal')],mode='characterize')
        c.obs=observation(4.)
        c.executor.start(m.MovementRequest('one',(0.,0.),(2.,0.),1,.4,30.,'mission'),4.)
        cell=c.grid.world_to_cell(.5,.1)
        c.grid.hazard[cell]=True;c.grid.blocked[cell]=True
        return c,cell

    def test_snapshot_keeps_hazard_and_observation_state_after_map_changes(self):
        c,cell=self.controller()
        reason=c._guard(m.Motion(.4,0.))
        self.assertEqual(reason,'straight drive or stopping footprint is blocked')
        event=c.escape_events[-1];data=event['terrain_evidence']
        offset=[cell[i]-data['patch_first_cell'][i] for i in (0,1)]
        self.assertTrue(data['fields']['hazard'][offset[0]][offset[1]])
        c.grid.hazard[cell]=False;c.grid.last_seen[cell]=100.
        self.assertTrue(data['fields']['hazard'][offset[0]][offset[1]])
        self.assertNotEqual(data['fields']['last_seen'][offset[0]][offset[1]],100.)
        self.assertEqual(event['action_id'],'one')
        self.assertEqual(event['time'],4.)
        self.assertEqual(event['category'],'commanded_straight')
        self.assertTrue(any(s.get('nearest_hazard_cell')==list(cell) for s in data['samples']))

    def test_repeated_ticks_and_candidate_checks_do_not_duplicate_evidence(self):
        c,_=self.controller()
        for _ in range(5): c._guard(m.Motion(.4,0.))
        self.assertEqual(len(c.escape_events),1)
        c.executor.stop(c.obs);c.executor.consume()
        c._guard(m.Motion(.4,0.))
        self.assertEqual(len(c.escape_events),1)
        c.executor.start(m.MovementRequest('two',(0.,0.),(2.,0.),1,.4,30.,'mission'),4.)
        c._guard(m.Motion(.4,0.))
        self.assertEqual(len(c.escape_events),2)

    def test_evidence_failure_cannot_change_guard_decision(self):
        c,_=self.controller()
        with patch.object(c.grid,'corridor_evidence',side_effect=RuntimeError('fixture diagnostic failure')):
            reason=c._guard(m.Motion(.4,0.))
        self.assertEqual(reason,'straight drive or stopping footprint is blocked')
        self.assertIn('fixture diagnostic failure',c.escape_events[-1]['diagnostic_error'])
        self.assertIsNotNone(c.executor.request)
        self.assertIsNone(c.executor.result)

    def test_capture_is_local_json_safe_and_does_not_mutate_grid(self):
        g=ground();revision=g.revision;known=g.known.copy();hazard=g.hazard.copy()
        snapshot=g.corridor_evidence((4.5,0.),(6.1,0.),0.,3.)
        json.dumps(snapshot,allow_nan=False)
        self.assertLess(math.prod(snapshot['patch_shape']),g.known.size//4)
        self.assertEqual(g.revision,revision)
        self.assertTrue((g.known==known).all());self.assertTrue((g.hazard==hazard).all())
        self.assertIsNone(snapshot['samples'][-1]['cell'])


if __name__=='__main__':unittest.main()
