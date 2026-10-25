"""Execute one externally inspected planner result; never replan its corridor.

Characterization only. Inherits teleop delivery, physical output checks, STOP,
manual takeover, tilt/overspeed handling and the 900 s deadline. Terrain turn
staging is deliberately absent; actual turning still checks the body sweep.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .remote_control import RemoteAscentController
from .checkpoint_control import Motion, wrap_angle
from . import checked_turn, exit_region, planning_snapshot


def load_route_result(filename, grid):
    filename = Path(filename)
    raw = filename.read_bytes()
    result = json.loads(raw)
    route = np.asarray(result['routes']['any_exit'], dtype=float)
    if route.ndim != 2 or route.shape[1] != 2 or len(route) < 2 or not np.isfinite(route).all():
        raise ValueError('Saved path must contain at least two finite XY points')
    if (np.linalg.norm(np.diff(route, axis=0), axis=1) < 1e-8).any():
        raise ValueError('Saved path contains a zero-length segment')
    maps = [(Path(p), digest) for p, digest in result['inputs'].items() if p.endswith('.npz')]
    if len(maps) != 1:
        raise ValueError('Saved path requires one identified planning map')
    source, digest = maps[0]
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise ValueError('Saved planning map hash mismatch')
    with np.load(source, allow_pickle=False) as archive:
        meta = json.loads(str(archive['metadata_json'].item()))
    expected = meta['config']['terrain_prior']
    actual = grid.terrain_prior
    for key in ('geometry_id', 'seed', 'task', 'coordinate_frame', 'file_sha256'):
        if not expected.get(key) or expected[key] != actual.get(key):
            raise ValueError('Saved path terrain identity mismatch: '+key)
    return route, dict(result_path=str(filename.resolve()), result_sha256=hashlib.sha256(raw).hexdigest(),
        route_name='any_exit', map_path=str(source), map_sha256=digest,
        parameters=result['parameters'], predicted_metrics=result['metrics']['any_exit'])


class SavedPathController(RemoteAscentController):
    def __init__(self, *args, saved_path_result, **kwargs):
        super().__init__(*args, **kwargs)
        if self.goal_mode != 'any_exit':
            raise ValueError('Saved path execution requires any-exit characterization')
        self.saved_route, self.saved_path_metadata = load_route_result(saved_path_result, self.grid)
        self.route_distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(self.saved_route, axis=0), axis=1))]
        self.protocol_version = 'saved-slope-path-v1-unrestricted-turn-staging'
        self.path = self.saved_route.tolist()
        self.mission_route = self.saved_route.tolist()
        self.route_profile = self.grid.route_profile(self.path)
        self.selected_exit = tuple(self.path[-1])
        self.checkpoint = self.selected_exit
        self.is_mission_checkpoint = True
        self.mission_route_complete = True
        self.route_purpose = 'mission'
        self.route_decision = 'Execute inspected any-exit slope-cost path'
        self.saved_started = False
        self.saved_progress = 0.
        self.saved_credit = 0.
        self.saved_progress_at = None
        self.saved_drive_since = None
        self.saved_relaunch_credit = None
        self.saved_stop_archived = False

    def request(self, action, now, vx=0., omega=0.):
        super().request(action, now, vx, omega)
        if action == 'resume':
            self.path = self.saved_route.tolist()
            self.mission_route = self.saved_route.tolist()
            self.mission_route_complete = True
            self.route_profile = self.grid.route_profile(self.path)
            self.checkpoint = self.selected_exit
            self.saved_drive_since = now

    def _command_geometry(self, command, *, escape=False, check_grade=True):
        reason = super()._command_geometry(command, escape=escape, check_grade=check_grade)
        if reason or not command.omega:
            return reason
        # Yaw-only commands still translate under this policy. Check their
        # combined observed drift and requested/measured rotation, not just
        # independent stationary rotation and uncommanded drift envelopes.
        o = self.obs
        horizon = .5+o.speed/.4
        end = tuple(o.xy[i]+o.velocity[i]*horizon for i in (0,1))
        rotations = (0., command.omega*horizon, o.yaw_rate*horizon)
        return checked_turn.physical_reason(self.grid, o.xy, end, o.yaw,
            (min(rotations), max(rotations)), sweep=self._physical_sweep)

    def _projection(self, xy):
        # Restrict matching to the current corridor neighborhood; a nearby
        # later branch cannot teleport the progress cursor across a bend.
        lo = max(0., self.saved_progress-1.5)
        hi = min(self.route_distance[-1], self.saved_progress+1.)
        candidates = []
        for i, (a, b) in enumerate(zip(self.saved_route, self.saved_route[1:])):
            start, end = self.route_distance[i:i+2]
            if end < lo or start > hi:
                continue
            fraction = float(np.clip(np.dot(np.subtract(xy, a), b-a)/np.dot(b-a, b-a),
                max(0., (lo-start)/(end-start)), min(1., (hi-start)/(end-start))))
            foot = a+fraction*(b-a)
            candidates.append((math.dist(xy, foot), start+fraction*(end-start), i))
        cross, distance, index = min(candidates)
        target_s = min(distance+.35, self.route_distance[-1])
        target = tuple(np.interp(target_s, self.route_distance, self.saved_route[:,axis]) for axis in (0,1))
        return distance, cross, index, target

    def _hold(self, reason, now):
        if not self.saved_stop_archived:
            try:
                record = planning_snapshot.save(self, self.obs.xy, self.obs.yaw, now)
            except Exception as error:
                record = dict(status='capture_failed', error=str(error))
            self.attempt_events.append(dict(event='saved_path_stop', time=now, reason=reason,
                position=list(self.obs.xy), progress_m=self.saved_progress,
                target=self.target, snapshot=record, physical_failure=False))
            self.saved_stop_archived = True
        return super()._hold(reason, now)

    def _recover(self, reason, physical=False):
        # This experiment measures this corridor; changing it would confound
        # route selection with a second planner. Keep the first refusal exact.
        return self._hold('saved-path execution stopped: '+reason, self.obs.time)

    def _navigate(self):
        o = self.obs
        if o.time < 3.:
            return Motion()
        if self.wp_idx >= len(self.waypoints):
            self.set_state('COMPLETE', 'saved path exterior reached and settled', o.time)
            self._button('COMPLETE', o.time, self.reason)
            return Motion()
        if not self.saved_started:
            if math.dist(o.xy, self.saved_route[0]) > .5:
                return self._hold('spawn differs from saved path start by more than 0.5 m', o.time)
            self.saved_started = True
            self.saved_progress_at = o.time
            self.saved_drive_since = o.time
            self.set_state('FOLLOW', 'execute inspected slope-cost path', o.time)
            record = planning_snapshot.save(self, o.xy, o.yaw, o.time)
            planning_snapshot.finish(record, self, True)
            self.attempt_events.append(dict(event='saved_path_start', time=o.time,
                route=self.path, metadata=self.saved_path_metadata, snapshot=record))
        distance, cross, index, target = self._projection(o.xy)
        self.path_index, self.tracking_error, self.target = index, cross, target
        self.saved_progress = max(self.saved_progress, distance)
        self.metrics.update(route_progress_m=distance, cross_track_m=cross)
        if distance-self.saved_credit >= .1:
            self.saved_credit, self.saved_progress_at = distance, o.time
        if cross > 1.:
            return self._hold('more than 1 m from inspected path', o.time)
        arrived = math.dist(o.xy, self.selected_exit) <= .4 and not exit_region.reason(self.grid,o.xy,self.exit_rim_radius)
        if arrived:
            self._button('SETTLE', o.time, 'release at selected exterior endpoint')
            stable = o.speed < .12 and abs(o.yaw_rate) < .12 and 0. < self.observation_dt <= .041
            self.goal_settle_since = (self.goal_settle_since if self.goal_settle_since is not None else o.time) if stable else None
            if self.goal_settle_since is not None and o.time-self.goal_settle_since >= .6:
                self.pending_goal_success = True
            return Motion()
        self.goal_settle_since = None
        if o.time-self.saved_progress_at >= 35.:
            return self._hold('35 s without 0.1 m new route progress', o.time)
        desired = math.atan2(target[1]-o.xy[1], target[0]-o.xy[0])
        error = wrap_angle(desired-o.yaw)
        self.heading_error = error
        if self.button_state == 'TURN':
            residual = wrap_angle(self.turn_target-o.yaw)
            if abs(residual) < self.turn_best-.03:
                self.turn_best, self.turn_progress_at = abs(residual), o.time
            if self.turn_sign*residual <= .10:
                self._button('DRIVE', o.time, 'heading correction completed')
                self.saved_drive_since = o.time
            elif o.time >= self.turn_deadline or o.time-self.turn_progress_at >= 12.:
                return self._hold('turn stopped making angular progress', o.time)
            else:
                return Motion(0., self.turn_sign*.35)
        if abs(error) > .22:
            self._start_turn(desired)
            return Motion(0., self.turn_sign*.35)
        if self.button_state == 'RELEASE':
            if o.time-self.saved_progress_at < 8.:
                return Motion()
            if self.saved_relaunch_credit is not None and self.saved_credit-self.saved_relaunch_credit < .1:
                return Motion()  # shared 35 s progress deadline remains active
            self.saved_relaunch_credit = self.saved_credit
            self.saved_drive_since = o.time
            self._button('DRIVE', o.time, 'release plateau: one forward re-initiation')
        slow = self.metrics.get('window_seconds', 0.) >= 2.-1e-6 and self.metrics['progress_mps'] < .10
        if (o.time-self.saved_drive_since >= 5. and slow and self.metrics.get('directional_grade_deg',0.) >= 10.):
            self._button('RELEASE', o.time, 'powered climb slow: observe policy-active release')
            return Motion()
        self._button('DRIVE', o.time, 'follow inspected path')
        return Motion(self.cruise_speed, 0.)
