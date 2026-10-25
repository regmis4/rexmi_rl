"""Keep a useful straight direction without changing where arrival is required."""
from dataclasses import replace
import json
import math
from pathlib import Path
import unittest

import numpy as np

from test_behavior_navigation import m, observation


class ArrivalHeadingTests(unittest.TestCase):
    def executor(self, sign=1, endpoint=None, radius=.2):
        e=m.MovementExecutor()
        e.straight_entry_guard=lambda *args: ''
        e.start(m.MovementRequest('ray',(0.,0.),endpoint or (sign*.75,sign*.12),
                                sign,.4,100.,'adjust',arrival_radius=radius),0.)
        return e

    def tick(self,e,obs,guard=None,supported=None):
        return e.tick(obs,guard or (lambda cmd:''),supported or (lambda xy:True))

    def admit(self,e,**kwargs):
        for t in np.arange(0.,.51,.02): command=self.tick(e,observation(t,**kwargs))
        return command

    def test_both_signs_admit_ray_that_reaches_inner_arrival_disk(self):
        for sign in (-1,1):
            e=self.executor(sign);req=e.request
            self.assertEqual(self.admit(e),m.Motion(sign*.4,0.))
            self.assertIs(e.request,req)
            self.assertEqual(e.committed_heading,0.)
            self.assertEqual(e.travel_axis,(float(sign),0.))
            self.assertFalse(any(h.get('state','').startswith('ALIGN') for h in e.history))

    def test_all_recorded_corrections_keep_straight_command(self):
        fixture=json.loads((Path(__file__).parent/'fixtures/arrival_ray.json').read_text())
        for row in fixture['samples']:
            e=m.MovementExecutor();sign=row['direction'];xy=tuple(row['position'])
            req=m.MovementRequest(row['action_id'],xy,tuple(row['endpoint']),sign,.4,
                                  row['time']+30.,'adjust',arrival_radius=row['arrival_radius'])
            e.start(req,row['time']-.1)
            e.committed_heading=row['body_yaw']
            e.travel_axis=(sign*math.cos(e.committed_heading),sign*math.sin(e.committed_heading))
            e._state(m.MovementState.DRIVE_FORWARD if sign==1 else m.MovementState.DRIVE_REVERSE,
                     row['time']-.1,'recorded correction fixture')
            cmd=self.tick(e,observation(row['time'],xy=xy,yaw=row['body_yaw']))
            self.assertEqual(cmd,m.Motion(sign*.4,0.),row['time'])
            self.assertIs(e.request,req)
            self.assertIsNone(e.result)

    def test_outer_disk_and_target_behind_do_not_authorize_new_heading(self):
        for endpoint in ((.75,.18),(-.75,0.)):
            e=self.executor(endpoint=endpoint)
            self.admit(e)
            self.assertIn(e.state,(m.MovementState.ALIGN_LEFT,m.MovementState.ALIGN_RIGHT))

    def test_steep_entry_does_not_gain_alignment_exemption(self):
        e=self.executor()
        self.admit(e,grade=25.,turn=False)
        self.assertEqual(e.result.outcome,'censored')
        self.assertIn('staging',e.result.reason)

    def test_opposing_motion_still_must_decelerate_before_reverse(self):
        e=self.executor(-1)
        self.admit(e,velocity=(.2,0.))
        self.assertEqual(e.state,m.MovementState.BRAKE)
        for t in np.arange(.52,1.01,.02):cmd=self.tick(e,observation(t,velocity=(.08,0.)))
        self.assertEqual(cmd,m.Motion(-.4,0.))

    def test_remaining_and_emitted_guards_still_reject_admission(self):
        for which in ('remaining','emitted'):
            e=self.executor(-1)
            if which=='remaining':e.corridor_guard=lambda *args:'remaining blocked'
            for t in np.arange(0.,.51,.02):
                self.tick(e,observation(t),guard=(lambda cmd:'emitted blocked') if which=='emitted' else None)
            self.assertEqual(e.result.outcome,'censored')
            self.assertFalse(e.result.physical)

    def test_continuation_geometry_and_overspeed_are_not_bypassed(self):
        e=self.executor(-1);self.admit(e)
        self.tick(e,observation(.52),guard=lambda cmd:'blocked')
        self.assertEqual(e.result.reason,'blocked')
        e=self.executor(-1);self.admit(e)
        self.tick(e,observation(.52,velocity=(-.5,0.)))
        self.assertEqual(e.state,m.MovementState.BRAKE)
        self.assertIsNone(e.result)

    def test_existing_turn_pulse_does_not_stop_early_on_transient_ray(self):
        e=self.executor(endpoint=(.75,.2));self.admit(e)
        self.assertEqual(e.state,m.MovementState.ALIGN_LEFT)
        # Admission/continuation tolerance must not introduce early pulse stops:
        # measured policy unwind can otherwise force an endless turn/coast loop.
        cmd=self.tick(e,observation(.52,yaw=.12,yaw_rate=.3))
        self.assertEqual(cmd.omega,.35)
        self.assertEqual(e.state,m.MovementState.ALIGN_LEFT)
        self.tick(e,observation(.54,yaw=math.atan2(.2,.75),yaw_rate=.3))
        self.assertEqual(e.state,m.MovementState.BRAKE)

    def test_useful_ray_does_not_skip_settling_or_rename_target(self):
        e=self.executor();self.admit(e);req=e.request
        self.tick(e,observation(.52,xy=(.55,0.),velocity=(.1,0.)))
        self.assertNotEqual(e.state,m.MovementState.SETTLE)
        self.tick(e,observation(.54,xy=(.65,0.),velocity=(.1,0.)))
        self.assertEqual(e.state,m.MovementState.SETTLE)
        self.assertIsNone(e.result)
        self.assertEqual(req.endpoint,(.75,.12))
        for t in np.arange(.56,1.3,.02):self.tick(e,observation(t,xy=(.65,0.)))
        self.assertEqual(e.result.outcome,'success')

    def test_stop_and_stale_observations_still_preempt(self):
        for cause in ('stop','stale'):
            e=self.executor(-1);self.admit(e)
            obs=observation(.52)
            cmd=e.stop(obs) if cause=='stop' else self.tick(e,replace(obs,fresh=False))
            self.assertEqual(cmd,m.Motion())
            self.assertEqual(e.result.outcome,'censored')
            self.assertIsNone(e.request)


if __name__=='__main__':unittest.main()
