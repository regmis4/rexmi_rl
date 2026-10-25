"""Offline command/response analysis of a teleop demonstration; no control writes."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(run, output):
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((run/'metadata.json').read_text())
    events = [json.loads(s) for s in (run/'events.jsonl').read_text().splitlines()]
    wheels = [j for j in metadata['joint_names'] if '_foot_' in j]
    bodies = [j.replace('_joint', '') for j in wheels]
    fields = ['sample', 'sim_time_s', 'dt_s', 'episode', 'control_step', 'substep',
              'policy', 'scenario', 'vx_command_m_s', 'omega_command_rad_s']
    fields += [c for c in metadata['columns'] if c.startswith(('root_', 'contact.'))]
    fields += [j+'.'+s for j in wheels for s in
               ('velocity_rad_s', 'velocity_target_rad_s', 'torque_applied_est_Nm')]
    raw = pd.read_csv(run/'samples.csv', usecols=fields)
    if not np.all(np.diff(raw['sample']) == 1):
        raise ValueError('Noncontiguous raw samples')
    if raw[fields].isna().any().any():
        raise ValueError('Missing required telemetry')
    dt, decimation = metadata['physics_dt_s'], metadata['decimation']
    control_dt = metadata['control_dt_s']
    if not np.allclose(np.diff(raw.sim_time_s), dt) or not np.allclose(raw.dt_s, dt):
        raise ValueError('Sample timing mismatch')
    counts = raw.groupby('control_step').size()
    partial = counts[counts != decimation]
    if len(partial) and (len(partial) != 1 or partial.index[0] != counts.index[-1]):
        raise ValueError('Only an incomplete final control step can be excluded')
    retained = raw[raw.control_step.isin(counts[counts == decimation].index)].copy()
    for _, group in raw.groupby('control_step', sort=False):
        if list(group.substep) != list(range(len(group))):
            raise ValueError('Unexpected substep ordering')
    if retained.episode.nunique() != 1 or any(e['event'] == 'auto_reset' for e in events):
        raise ValueError('Analyze episodes separately; never bridge resets')
    for _, group in retained.groupby('control_step', sort=False):
        if group[['vx_command_m_s', 'omega_command_rad_s', 'policy']].nunique().max() != 1:
            raise ValueError('Command changed within a control step')
    retained['wheel_abs_rad_s'] = retained[[j+'.velocity_rad_s' for j in wheels]].abs().mean(axis=1)
    retained['wheel_target_abs_rad_s'] = retained[[j+'.velocity_target_rad_s' for j in wheels]].abs().mean(axis=1)
    forces = np.stack([retained[[f'contact.{body}.{axis}_N' for axis in ('fx_w', 'fy_w', 'fz_w')]].to_numpy()
                       for body in bodies], axis=1)
    retained['contact_count'] = (np.linalg.norm(forces, axis=2) > 5.).sum(axis=1)
    velocity = retained[[f'root_linear_velocity_w_m_s.{a}' for a in 'xyz']].to_numpy()
    retained['speed_xy_m_s'] = np.linalg.norm(velocity[:, :2], axis=1)
    means = retained.groupby('control_step')[['wheel_abs_rad_s', 'wheel_target_abs_rad_s', 'contact_count', 'speed_xy_m_s']].mean()
    c = retained.groupby('control_step', sort=False).last().join(means, rsuffix='_mean')
    c['start_s'] = c.index.to_numpy()*control_dt
    c['end_s'] = c.start_s+control_dt
    for axis in 'xyz': c[axis] = c[f'root_position_w_m.{axis}']
    qw, qx, qy, qz = (c[f'root_quaternion_w.{a}'].to_numpy() for a in 'wxyz')
    c['yaw_rad'] = np.unwrap(np.arctan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz)))
    c['body_tilt_deg'] = np.degrees(np.arccos(np.clip(1-2*(qx*qx+qy*qy), -1, 1)))
    c['radius_m'] = np.hypot(c.x, c.y)
    c['vx'] = c.vx_command_m_s
    c['omega'] = c.omega_command_rad_s
    c['mode'] = np.select([(c.vx == 0)&(c.omega == 0), (c.vx < 0), (c.vx > 0)],
                          ['zero command', 'reverse', 'forward'], default='yaw only')
    nav = pd.read_csv(run.with_suffix('.csv'))
    if list(nav.step) != list(c.index):
        raise ValueError('Navigation and retained physics control steps do not match')
    if not np.allclose(nav[['vx_cmd', 'omega_cmd']], c[['vx', 'omega']]):
        raise ValueError('Command logs disagree')
    if not np.allclose(nav[['x', 'y', 'z']], c[['x', 'y', 'z']], atol=.00051):
        raise ValueError('Pose logs disagree beyond CSV rounding')

    def window(start, end):
        a, b = int(round(start/control_dt)), int(round(end/control_dt))
        g = c.iloc[a:b]
        before = c.iloc[max(0, a-1)]
        after = g.iloc[-1]
        xy = np.vstack([before[['x', 'y']].to_numpy(float), g[['x', 'y']].to_numpy(float)])
        delta = xy[-1]-xy[0]
        chord = delta/max(np.linalg.norm(delta), 1e-9)
        axis = np.array((math.cos(before.yaw_rad), math.sin(before.yaw_rad)))
        rel = xy-xy[0]
        duration = len(g)*control_dt
        return dict(start_s=float(start), end_s=float(end), duration_s=duration,
            start_xy=xy[0].tolist(), end_xy=xy[-1].tolist(),
            displacement_xy_m=float(np.linalg.norm(delta)), path_xy_m=float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()),
            body_height_gain_m=float(after.z-before.z), radial_gain_m=float(after.radius_m-before.radius_m),
            net_planar_rate_m_s=float(np.linalg.norm(delta)/duration),
            height_rate_m_s=float((after.z-before.z)/duration),
            yaw_change_deg=float(np.degrees(after.yaw_rad-before.yaw_rad)),
            max_initial_heading_offset_m=float(np.abs(rel[:, 0]*axis[1]-rel[:, 1]*axis[0]).max()),
            max_chord_offset_m=float(np.abs(rel[:, 0]*chord[1]-rel[:, 1]*chord[0]).max()),
            wheel_abs_mean_rad_s=float(g.wheel_abs_rad_s_mean.mean()),
            wheel_target_abs_mean_rad_s=float(g.wheel_target_abs_rad_s_mean.mean()),
            contact_count_mean=float(g.contact_count_mean.mean()),
            body_tilt_peak_deg=float(g.body_tilt_deg.max()),
            vx_values=sorted(g.vx.unique().tolist()), omega_values=sorted(g.omega.unique().tolist()))

    segments = []
    for _, group in c.groupby((c[['vx', 'omega']].ne(c[['vx', 'omega']].shift()).any(axis=1)).cumsum()):
        segments.append(dict(vx=float(group.vx.iloc[0]), omega=float(group.omega.iloc[0]),
                             **window(group.start_s.iloc[0], group.end_s.iloc[-1])))
    bouts = []
    for channel in ('vx', 'omega'):
        for _, group in c.groupby(c[channel].ne(c[channel].shift()).cumsum()):
            value = float(group[channel].iloc[0])
            if value:
                bouts.append(dict(channel=channel, command=value,
                    **window(group.start_s.iloc[0], group.end_s.iloc[-1])))
    key_windows = {
        'long_forward': (62.22, 92.36),
        'before_uphill_release_equal_window': (87.94, 92.36),
        'uphill_zero_command': (92.36, 96.78),
        'uphill_zero_after_first_second': (93.36, 96.78),
        'uphill_zero_late': (94.36, 96.78),
        'reverse_reposition': (103.66, 109.44),
        'forward_after_reverse': (109.64, 114.8),
        'forward_low_progress': (114.82, 116.7),
        'sustained_final_climb': (122., 137.),
        'climb_low_progress_first_five': (122., 127.),
        'climb_breakthrough_next_five': (127., 132.),
        'climb_continued_last_five': (132., 137.),
        'zero_after_final_climb': (137., 141.38),
        'long_yaw_only': (17.78, 26.10),
    }
    # This named case study is bound to the measured command boundaries. A new
    # capture needs its own reviewed windows rather than inherited conclusions.
    if run.name == 'crater-demo-repeat-20260923-225750':
        key_windows = {
            'initial_climb': (92.84, 104.10),
            'reverse_reposition': (117.34, 123.62),
            'ineffective_release': (130.08, 135.92),
            'approach_adjustment': (135.92, 139.16),
            'sustained_uphill_release': (139.16, 155.80),
            'release_after_first_second': (140.16, 155.80),
            'release_first_five': (139.16, 144.16),
            'release_next_five': (144.16, 149.16),
            'release_last_seconds': (151.16, 155.80),
            'release_near_rim': (157.84, 162.80),
            'exit_drive': (162.80, 172.34),
        }
    elif run.name != 'crater-demo-20260923-222151':
        key_windows = {}
    windows = {name: window(*bounds) for name, bounds in key_windows.items()}
    loading = []
    for label, bounds in key_windows.items() if windows else []:
        a, b = (int(round(t/control_dt)) for t in bounds)
        g = retained[(retained.control_step >= a)&(retained.control_step < b)]
        for joint, body in zip(wheels, bodies):
            force = g[[f'contact.{body}.{axis}_N' for axis in ('fx_w', 'fy_w', 'fz_w')]].to_numpy()
            loaded = np.linalg.norm(force, axis=1) > 5.
            speed = g[joint+'.velocity_rad_s'].abs().to_numpy()
            loading.append(dict(window=label, wheel=joint, loaded_fraction=float(loaded.mean()),
                loaded_mean_abs_rad_s=float(speed[loaded].mean()) if loaded.any() else None,
                unloaded_mean_abs_rad_s=float(speed[~loaded].mean()) if (~loaded).any() else None))
    duration_stats = {}
    for label, channel, sign in [('forward', 'vx', 1), ('reverse', 'vx', -1), ('yaw', 'omega', None)]:
        selected = [b for b in bouts if b['channel'] == channel and (sign is None or b['command']*sign > 0)]
        durations = [b['duration_s'] for b in selected]
        duration_stats[label] = dict(bouts=len(durations), median_s=float(np.median(durations)) if durations else None,
                                    max_s=max(durations) if durations else None)
    floor = c.radius_m <= 3.
    min_i = c.radius_m.idxmin()
    after_floor = c.loc[min_i:]
    exited = after_floor[after_floor.radius_m >= 11.]
    summary = dict(run=str(run.resolve()), source_hashes={name: fingerprint(run/name) for name in ('samples.csv', 'metadata.json', 'events.jsonl')},
        navigation_csv_sha256=fingerprint(run.with_suffix('.csv')),
        integrity=dict(raw_physics_samples=len(raw), retained_physics_samples=len(retained),
            excluded_tail_samples=len(raw)-len(retained), complete_control_steps=len(c), duration_s=float(c.end_s.iloc[-1]),
            gap_free=True, cross_log_commands_and_pose_agree=True, episodes=1,
            reset_events=0, capture_end_present=any(e['event'] == 'capture_end' for e in events),
            note='Complete control steps verified; any incomplete tail is excluded. Clean-close marker reported separately.'),
        policy=list(c.policy.unique()), commands=dict(vx=sorted(c.vx.unique().tolist()), omega=sorted(c.omega.unique().tolist())),
        traversal=dict(start_xy=c[['x', 'y']].iloc[0].tolist(), final_xy=c[['x', 'y']].iloc[-1].tolist(),
            minimum_radius_m=float(c.radius_m.min()), floor_first_s=float(c.loc[floor, 'end_s'].iloc[0]) if floor.any() else None,
            closest_centre_s=float(c.loc[min_i, 'end_s']), exterior_after_floor_s=float(exited.end_s.iloc[0]) if len(exited) else None,
            final_radius_m=float(c.radius_m.iloc[-1]), final_distance_to_autonomous_goal_m=float(math.hypot(c.x.iloc[-1]-13, c.y.iloc[-1])),
            interpretation='Recorded exterior-floor-exterior traversal; exit differs from autonomous destination (13,0). Not an autonomous qualification run.'),
        duration_stats=duration_stats, selected_windows=windows,
        interpretation_limits=['Observational single run; command and terrain are confounded.',
            'Wheel speeds and body progression are not per-wheel contact-point slip measurements.',
            'Wheel contacts use world normal-force magnitude >5N; no friction-force measurement.',
            'Body height/tilt are not ground elevation/slope.',
            'Zero velocity command keeps the trained policy and its joint targets active.'])
    pd.DataFrame(segments).to_csv(output/'command_segments.csv', index=False)
    pd.DataFrame(bouts).to_csv(output/'button_holds.csv', index=False)
    pd.DataFrame(windows).T.to_csv(output/'selected_windows.csv', index_label='window')
    pd.DataFrame(loading).to_csv(output/'wheel_loading.csv', index=False)
    c.to_csv(output/'control_response.csv', index_label='control_step')
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    render(c, summary, output)
    print(json.dumps({k: summary[k] for k in ('integrity', 'traversal', 'duration_stats')}, indent=2))
    return summary


def render(c, summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    colors = {'forward': '#168878', 'reverse': '#bd3f64', 'zero command': '#526fbc', 'yaw only': '#ba801c'}
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout='constrained', gridspec_kw={'width_ratios': [1, 1.3]})
    ax = axes[0]
    xy = c[['x', 'y']].to_numpy(); segments = np.stack([xy[:-1], xy[1:]], axis=1)
    ax.add_collection(LineCollection(segments, colors=[colors[m] for m in c['mode'].iloc[1:]], linewidths=2))
    for r, label in [(3, 'floor radius'), (11, 'rim radius')]:
        ax.add_patch(plt.Circle((0, 0), r, fill=False, ls='--', color='#b6bac2', label=label))
    for label, color in colors.items(): ax.plot([], [], color=color, lw=3, label=label)
    ax.scatter(*xy[0], c='black', marker='s', label='start'); ax.scatter(*xy[-1], c='black', marker='*', s=100, label='finish')
    ax.set(xlim=(-5, 15), ylim=(-5, 15), xlabel='World X (m)', ylabel='World Y (m)', title='Recorded manual route — commands by color', aspect='equal')
    ax.legend(fontsize=8, loc='upper left'); ax.grid(alpha=.15)
    ax = axes[1]
    ax.plot(c.end_s, c.z, color='#213b54', lw=1.6)
    for label, color in colors.items():
        mask = c['mode'] == label
        ax.scatter(c.end_s[mask], c.z[mask], s=2, color=color, label=label)
    ax.set(xlabel='Simulated time (s)', ylabel='Body height (m)', title='Height versus command (height is not terrain slope)')
    ax.grid(alpha=.15)
    fig.suptitle('Teleop crater traversal: one policy, ±0.40 m/s commands')
    fig.savefig(output/'route_and_height.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True, layout='constrained')
    climb_start = summary['traversal']['closest_centre_s']
    g = c[c.end_s >= climb_start]
    axes[0].step(g.start_s, g.vx, where='post', color='#168878', label='vx command (m/s)')
    axes[0].step(g.start_s, g.omega, where='post', color='#ba801c', alpha=.8, label='yaw command (rad/s)')
    axes[0].legend(loc='lower left', ncol=2); axes[0].set_ylabel('Requested motion')
    axes[1].plot(g.end_s, g.z, color='#213b54'); axes[1].set_ylabel('Body height (m)')
    axes[2].plot(g.end_s, g.speed_xy_m_s_mean.rolling(10, center=True, min_periods=1).mean(), color='#168878')
    axes[2].set_ylabel('Actual planar speed\n(m/s; 0.2 s mean)')
    axes[3].plot(g.end_s, g.wheel_abs_rad_s_mean.rolling(10, center=True, min_periods=1).mean(), color='#bd3f64')
    axes[3].set_ylabel('Mean |wheel speed|\n(rad/s; 0.2 s mean)'); axes[3].set_xlabel('Simulated time (s)')
    for ax in axes:
        for _, group in g.groupby(g['mode'].ne(g['mode'].shift()).cumsum()):
            mode = group['mode'].iloc[0]
            if mode in ('zero command', 'reverse'):
                ax.axvspan(group.start_s.iloc[0], group.end_s.iloc[-1], color=colors[mode], alpha=.12)
        ax.grid(alpha=.15)
    fig.suptitle('Release and reverse response — blue: zero command; pink: reverse')
    fig.savefig(output/'climb_response.png', dpi=160); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.run, args.output)
