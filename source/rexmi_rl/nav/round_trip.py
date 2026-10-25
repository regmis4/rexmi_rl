"""High-level round-trip configuration; movement remains owned by the existing BT."""
import math

DEADLINE_SECONDS = 900.


def configure(args, argv):
    if args.nav_controller != 'ascent' or args.ascent_architecture not in ('behavior_tree', 'remote') or args.ascent_mode != 'characterize':
        raise ValueError('round_trip requires a behavior-tree or remote ascent controller in characterize mode')
    if not (args.terrain_prior or args.simulator_survey):
        raise ValueError('round_trip requires a full-crater terrain prior or simulator survey')
    if any((args.reviewed_buttons, args.checkpoint_replay, args.manual_checkpoints,
            args.capture_ascent_departure, args.flat_sidesteps, args.ascent_exploration_history,
            args.ascent_launch_variant)):
        raise ValueError('round_trip owns its destinations; replay, checkpoint overrides and route variants are incompatible')
    defaults = dict(spawn_x=args.crater_x + 13., spawn_y=args.crater_y, spawn_z=4.5,
                    spawn_yaw=math.pi, goal_x=args.crater_x, goal_y=args.crater_y + 13.,
                    goal_radius=.6, max_steps=45000)
    for key, value in defaults.items():
        if not any(item == '--'+key or item.startswith('--'+key+'=') for item in argv):
            setattr(args, key, value)
    values = (args.crater_x, args.crater_y, args.r_rim, args.spawn_x, args.spawn_y,
              args.spawn_z, args.spawn_yaw, args.goal_x, args.goal_y, args.goal_radius)
    if not all(math.isfinite(value) for value in values) or args.r_rim <= 1 or args.goal_radius <= 0:
        raise ValueError('round_trip requires finite geometry, rim radius >1 m and a positive destination radius')
    center = (args.crater_x, args.crater_y)
    if math.dist((args.spawn_x, args.spawn_y), center) <= args.r_rim:
        raise ValueError('round_trip must start outside the crater rim')
    if math.dist((args.goal_x, args.goal_y), center) - args.goal_radius <= args.r_rim:
        raise ValueError('The entire exterior destination disk must lie outside the crater rim')
    args.map_retention = 'full'


def stage_status(waypoints, index):
    if index >= len(waypoints):
        return dict(stage='COMPLETE', goal=None, completed_destinations=len(waypoints))
    waypoint = waypoints[index]
    return dict(stage=('DESCEND_TO_CENTER', 'ASCEND_TO_EXTERIOR')[index],
                goal=[waypoint.x, waypoint.y], completed_destinations=index)


def deadline_reached(now, dt):
    """Never start a physics step that would overrun the mission-wide budget."""
    return now + dt > DEADLINE_SECONDS + 1e-9
