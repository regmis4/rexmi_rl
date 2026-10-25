"""Bounded display telemetry; never reads or changes controller decisions."""
import math
import numpy as np


class TravelHistory:
    """Accumulate every control tick; publish spatial samples for cheap redraws."""
    def __init__(self, spacing=.05, limit=6000):
        self.spacing, self.limit = spacing, limit
        self.reset()

    def reset(self):
        self.last = None
        self.time = None
        self.start_z = None
        self.distance = 0.
        self.rows = []
        self.current = None

    def update(self, now, xyz, grade):
        xyz = np.asarray(xyz, dtype=float)
        if not math.isfinite(now) or not np.isfinite(xyz).all():
            self.last = None
            return
        if self.time is not None and now < self.time:
            self.reset()
        if self.time == now:
            return
        if self.start_z is None:
            self.start_z = xyz[2]
        if self.last is not None:
            self.distance += float(np.linalg.norm(xyz[:2]-self.last[:2]))
        self.last, self.time = xyz.copy(), now
        self.current = (self.distance, xyz[2]-self.start_z, grade)
        if not self.rows or self.distance-self.rows[-1][0] >= self.spacing:
            self.rows.append(self.current)
        if len(self.rows) >= self.limit:
            self.rows = self.rows[::2]
            self.spacing *= 2

    def snapshot(self):
        rows = list(self.rows)
        if self.current is not None:
            if rows and rows[-1][0] == self.current[0]:
                rows[-1] = self.current
            else:
                rows.append(self.current)
        return np.asarray(rows, dtype=float).reshape(-1, 3)


def drive_activity(cmd, speed, yaw_rate, state=''):
    """Command intent plus measured motion; zero command is not proof of stop."""
    vx, vy, omega = cmd
    if not np.isfinite([vx, vy, omega, speed]).all():
        return {'NO DATA'}
    active = set()
    if vx > .01: active.add('FORWARD')
    if vx < -.01: active.add('REVERSE')
    if vy > .01: active.add('STRAFE LEFT')
    if vy < -.01: active.add('STRAFE RIGHT')
    if omega > .01: active.add('LEFT')
    if omega < -.01: active.add('RIGHT')
    if not active:
        moving = speed > .05 or (math.isfinite(yaw_rate) and abs(yaw_rate) > .05)
        if moving or any(word in state.upper() for word in ('BRAKE', 'DAMP', 'RELEASE', 'SETTLE')):
            active.add('BRAKING')
        elif not math.isfinite(yaw_rate):
            active.add('ZERO COMMAND')
        else:
            active.add('STOP')
    return active
