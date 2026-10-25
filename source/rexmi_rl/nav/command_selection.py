"""Bounded forward demand selection inside unchanged observed stopping checks."""
import math

from .checkpoint_control import Motion


FORWARD_SPEED_PROFILE = dict(
    forward_speed_resolution_mps=.025,
    forward_speed_floor_mps=.10,
    forward_speed_contract='highest passing forward candidate on descending 0.025 m/s grid, '
        'nominal request first, floor 0.10; pure probes; unchanged actual-drift-first guard '
        'and measured-speed floor; final selected command rechecked before emission; '
        'no feasible candidate retains original censor; reverse/brake/turn/release unchanged')


def select_forward_command(requested, check):
    """Finite characterization search, not an optimized or calibrated speed.

    ``check`` must be observational: rejected candidates are not attempted
    movement or evidence. The executor independently checks the returned output.
    """
    if requested.vx <= 0. or requested.omega or not math.isfinite(requested.vx):
        return requested
    if not check(requested):
        return requested
    step = FORWARD_SPEED_PROFILE['forward_speed_resolution_mps']
    minimum = math.ceil(FORWARD_SPEED_PROFILE['forward_speed_floor_mps']/step)
    maximum = min(16, math.ceil(requested.vx/step)-1)
    for index in range(maximum, minimum-1, -1):
        candidate = Motion(index*step, 0.)
        if not check(candidate):
            return candidate
    return requested
