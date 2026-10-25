"""Finite, recovery-only reorientation; geometry is not a capability claim."""
from dataclasses import dataclass
import math

from .checkpoint_control import wrap_angle
from .swept_footprint import oriented_sweep_reason, oriented_stopping_reason


@dataclass(frozen=True)
class RecoveryTurn:
    origin: tuple
    initial_yaw: float
    target_yaw: float
    endpoint: tuple
    direction: int
    deadline: float
    action_id: str = ''

    @property
    def sign(self):
        return 1 if wrap_angle(self.target_yaw-self.initial_yaw) > 0 else -1


def rotation_reason(grid, obs, intent):
    if obs.time > intent.deadline:
        return 'recovery turn original deadline elapsed'
    if math.dist(obs.xy, intent.origin) > .25:
        return 'recovery turn left frozen position reserve'
    delta = wrap_angle(intent.target_yaw-intent.initial_yaw)
    if not 0. < abs(delta) <= math.radians(30.)+1e-9:
        return 'invalid recovery turn angle'
    low = max(-math.radians(30.), min(0., delta)-.03)
    high = min(math.radians(30.), max(0., delta)+.03)
    swept = wrap_angle(obs.yaw-intent.initial_yaw)
    if not low-1e-9 <= swept <= high+1e-9:
        return 'recovery turn left frozen angular bounds'
    # Keep checking the original reserved volume, not a recentered permission.
    return oriented_sweep_reason(grid, intent.origin, intent.origin, intent.initial_yaw,
        yaw_offsets=(low, high),
        position_reserve=.25, commanded=False)


def exit_reason(grid, obs, intent, offsets):
    if offsets is None:
        return 'recovery turn needs measured heading history'
    end = (intent.endpoint[0]+intent.direction*.5*math.cos(intent.target_yaw),
           intent.endpoint[1]+intent.direction*.5*math.sin(intent.target_yaw))
    return (grid.support_reason(intent.endpoint) or
        oriented_stopping_reason(grid, obs.xy, end, intent.target_yaw, offsets, commanded=True))
