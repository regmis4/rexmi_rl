"""Map-click validation must not revoke a previously accepted destination."""
import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from test_checkpoint_navigation import surface,c,module

runner=module('click_request_runner','nav/checkpoint_runner.py')


class SelectionTests(unittest.TestCase):
    def test_replay_only_accepts_recorded_valid_targets_in_time_order(self):
        events=[dict(event='checkpoint_selection',time=2,requested=[1,2],accepted=True),
                dict(event='checkpoint_selection',time=1,requested=[0,1],accepted=True),
                dict(event='checkpoint_selection',time=3,requested=[8,9],accepted=False),
                dict(event='checkpoint_selection',time=4,requested=[1,2],accepted=True,duplicate=True)]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'events.jsonl';path.write_text('\n'.join(map(json.dumps,events)))
            self.assertEqual(runner.load_checkpoint_replay(path),[(1.,(0.,1.)),(2.,(1.,2.))])

    def test_replay_rejects_nonfinite_targets_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'events.jsonl'
            path.write_text(json.dumps(dict(event='checkpoint_selection',time=1,requested=[float('nan'),2],accepted=True)))
            with self.assertRaises(ValueError):runner.load_checkpoint_replay(path)

    def controller(self):
        grid=surface()
        control=c.CheckpointController(grid,[SimpleNamespace(x=3.,y=0.)])
        control.checkpoint_mode(True,0.)
        return grid,control

    def test_rejected_second_click_preserves_valid_observation_and_route(self):
        grid,control=self.controller()
        self.assertTrue(control.choose_checkpoint((0,0),(2,0),1.))
        waypoints=control.waypoints;since=control.since
        self.assertFalse(control.choose_checkpoint((0,0),(100,100),1.2))
        self.assertIs(control.waypoints,waypoints)
        self.assertEqual(control.state,'OBSERVE')
        self.assertEqual(control.since,since)
        for t in (1.2,1.4,1.6):grid.rebuild(t)
        motion=control.tick((0,0),0,0,2.2)
        self.assertEqual(control.state,'FOLLOW')
        self.assertGreater(motion.vx,0.)
        path=list(control.path);checkpoint=control.checkpoint
        self.assertFalse(control.choose_checkpoint((0,0),(100,100),2.3))
        self.assertEqual(control.path,path)
        self.assertEqual(control.checkpoint,checkpoint)
        self.assertEqual(control.state,'FOLLOW')
        self.assertGreater(control.tick((0,0),0,0,2.4).vx,0.)

    def test_duplicate_click_does_not_restart_observation(self):
        grid,control=self.controller()
        self.assertTrue(control.choose_checkpoint((0,0),(2.01,.01),1.))
        since=control.since
        self.assertTrue(control.choose_checkpoint((0,0),(2.02,.02),1.2))
        self.assertEqual(control.since,since)
        self.assertEqual(control.selected_goal,(2.01,.01))
        self.assertTrue(control.last_selection['duplicate'])

    def test_destination_footprint_rejection_explains_unknown_ground(self):
        grid,control=self.controller()
        grid.known[grid.world_to_cell(2.3,.1)]=False
        self.assertTrue(grid.traversable(grid.world_to_cell(2,0)))
        self.assertFalse(control.choose_checkpoint((0,0),(2,0),1.))
        self.assertIn('destination footprint',control.selection_status)
        self.assertIn('unobserved',control.selection_status)
        self.assertEqual(control.last_selection['requested'],[2.,0.])
        self.assertEqual(control.state,'HOLD')

    def test_stored_map_can_select_but_not_authorize_stale_motion(self):
        grid,control=self.controller();grid.rebuild(20.)
        self.assertFalse(grid.fresh.any())
        self.assertTrue(control.choose_checkpoint((0,0),(2,0),20.))
        for t in (20.2,20.4,20.6):grid.rebuild(t)
        command=control.tick((0,0),0,0,21.2)
        self.assertEqual(command,c.Motion())
        self.assertNotEqual(control.state,'FOLLOW')

    def test_invalid_click_does_not_interrupt_active_sidestep(self):
        grid,control=self.controller()
        control.choose_checkpoint((0,0),(2,0),1.)
        cancelled=[]
        side=SimpleNamespace(active=True,cancel=cancelled.append,suppressed_target=(1,0))
        event=runner.apply_operator_request(control,side,'target',100,100,(0,0),0,1.2)
        self.assertFalse(event['accepted']);self.assertEqual(cancelled,[])
        self.assertEqual(side.suppressed_target,(1,0))
        runner.apply_operator_request(control,side,'stop',0,0,(0,0),0,1.3)
        self.assertEqual(control.state,'HOLD')
        self.assertEqual(cancelled,['operator stop'])

    def test_valid_replacement_cancels_sidestep_and_uses_new_goal(self):
        grid,control=self.controller();control.choose_checkpoint((0,0),(2,0),1.)
        cancelled=[]
        side=SimpleNamespace(active=True,cancel=cancelled.append,suppressed_target=(1,0))
        event=runner.apply_operator_request(control,side,'target',2,1,(0,0),0,1.2)
        self.assertTrue(event['accepted'])
        self.assertEqual(cancelled,['operator target'])
        self.assertIsNone(side.suppressed_target)
        self.assertEqual(control.selected_goal,(2.,1.))

    def test_same_goal_click_after_stop_explicitly_resumes(self):
        grid,control=self.controller();control.choose_checkpoint((0,0),(2,0),1.)
        control.request('stop',1.1)
        self.assertTrue(control.choose_checkpoint((0,0),(2,0),1.2))
        self.assertEqual(control.state,'OBSERVE')
        self.assertFalse(control.last_selection['duplicate'])

    def test_double_click_queues_only_one_target(self):
        from queue import SimpleQueue
        from threading import Lock
        dashboard=module('double_click_dashboard','nav/dashboard.py')
        queue=SimpleQueue()
        view=dashboard.Dashboard({'manual_checkpoint_mode':True},Lock(),surface(),[],request_queue=queue)
        view._ax_global=object();view._ax_local=object()
        view._fig=SimpleNamespace(canvas=SimpleNamespace(toolbar=SimpleNamespace(mode='')))
        event=SimpleNamespace(button=1,inaxes=view._ax_global,xdata=1.,ydata=0.,dblclick=False)
        view._on_map_click(event)
        event.dblclick=True;view._on_map_click(event)
        self.assertEqual(queue.get(),('target',1.,0.))
        self.assertTrue(queue.empty())


if __name__=='__main__':unittest.main()
