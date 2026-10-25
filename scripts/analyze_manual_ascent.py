#!/usr/bin/env python3
"""Audit and plot one manual ascent capture, without running or changing navigation."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd


def path_cost_profile(points, costs, known, lower, cell_size):
    """Integrate the displayed piecewise-constant cell cost exactly along a polyline."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError('Expected finite XY points')
    distance = total = 0.
    rows = []
    for a, b in zip(points, points[1:]):
        length = float(np.linalg.norm(b-a))
        if length == 0:
            continue
        first, delta = (a-np.asarray(lower))/cell_size, (b-a)/cell_size
        cuts = [0., 1.]
        for axis in (0, 1):
            if abs(delta[axis]) < 1e-14:
                continue
            lo, hi = sorted((first[axis], first[axis]+delta[axis]))
            cuts.extend(((np.arange(np.floor(lo)+1, np.ceil(hi))-first[axis])/delta[axis]).tolist())
        cuts = np.unique(np.clip(cuts, 0, 1))
        for lo, hi in zip(cuts, cuts[1:]):
            idx = np.floor(first+(lo+hi)/2*delta).astype(int)
            if (idx < 0).any() or (idx >= costs.shape).any() or not known[tuple(idx)]:
                raise ValueError('Path crosses unsupported reference cost')
            cost = float(costs[tuple(idx)])
            if not np.isfinite(cost) or cost < 0:
                raise ValueError('Invalid reference cost')
            d = length*(hi-lo)
            rows.append(dict(start_m=distance, end_m=distance+d, cost=cost,
                section_cost=cost*d, cumulative_cost=total+cost*d))
            distance += d
            total += cost*d
    return pd.DataFrame(rows, columns=['start_m', 'end_m', 'cost', 'section_cost', 'cumulative_cost'])


def sample_with_end(frame, stride):
    positions = np.unique(np.r_[np.arange(0, len(frame), stride), len(frame)-1])
    return frame.iloc[positions]


def cost_analysis(out, active, amap, cfg, exit_step):
    lower = np.asarray(cfg['origin'])-cfg['world_size']/2
    sample = sample_with_end(active, 5)
    def measure(frame):
        return path_cost_profile(frame[['x', 'y']].to_numpy(), amap['cost'], amap['known'], lower, cfg['cell_size'])
    profile = measure(sample)
    full = measure(active)
    exterior = measure(sample_with_end(active.loc[:exit_step], 5)) if exit_step is not None else None
    total = float(profile.cumulative_cost.iloc[-1])
    length = float(profile.end_m.iloc[-1])
    # Nominal wheel-centre spacing from the existing wheelbase-demand model.
    wheelbase = .3868-.213*np.sin(.8)-.2264*np.sin(-.7)+.213*np.sin(1.)+.2264*np.sin(-.5)
    boundaries = np.r_[np.arange(0, length, wheelbase), length]
    cumulative = np.interp(boundaries, np.r_[0, profile.end_m], np.r_[0, profile.cumulative_cost])
    sections = pd.DataFrame(dict(start_m=boundaries[:-1], end_m=boundaries[1:],
        integrated_cost=np.diff(cumulative), mean_cost=np.diff(cumulative)/np.diff(boundaries)))
    sections.to_csv(out/'wheelbase_cost_sections.csv', index=False)
    profile.to_csv(out/'path_cost_profile.csv', index=False)
    result = dict(definition='Integral C(x,y) ds along measured XY; exact cell crossings on a 10 Hz polyline.',
        units='displayed terrain cost × metre', total_displayed_cost_m=total,
        distance_m=length, distance_weighted_mean_cost=total/length,
        peak_cell_cost=float(profile.cost.max()), nominal_wheelbase_m=float(wheelbase),
        wheelbase_sections=len(sections),
        full_50hz_total_cost_m=float(full.cumulative_cost.iloc[-1]),
        full_50hz_distance_m=float(full.end_m.iloc[-1]),
        sampling_difference_percent=float((full.cumulative_cost.iloc[-1]/total-1)*100),
        to_first_exterior_cost_m=float(exterior.cumulative_cost.iloc[-1]) if exterior is not None else None,
        to_first_exterior_distance_m=float(exterior.end_m.iloc[-1]) if exterior is not None else None,
        scope='Settled pre-command start to last complete recorded position, including actual drift/repositioning; startup excluded. Endpoints retained.',
        limitations='Displayed-cost integral, not the history-dependent sustained-climb objective, energy, or success probability. Stationary time has no cost; actual motion during zero input does. No planner mask penalty added.')
    (out/'path_cost_summary.json').write_text(json.dumps(result, indent=2)+'\n')
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, layout='constrained')
    axes[0].plot(np.r_[0, profile.end_m], np.r_[0, profile.cumulative_cost], color='#219d25', lw=2.5)
    axes[0].scatter([length], [total], color='#219d25')
    axes[0].annotate(f'TOTAL {total:.2f} cost·m\n{length:.2f} m travelled', (length, total),
                     xytext=(-12, -43), textcoords='offset points', ha='right', fontsize=11, weight='bold')
    axes[0].set_ylabel('Cumulative displayed cost\n(cost·m)')
    colors = plt.cm.plasma(Normalize(1, 20)(sections.mean_cost))
    axes[1].bar(sections.start_m, sections.integrated_cost, width=sections.end_m-sections.start_m,
                align='edge', color=colors, edgecolor='white', linewidth=.3)
    axes[1].set_ylabel('Cost per wheelbase section\n(cost·m)')
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(sample[['x', 'y']], axis=0), axis=1))]
    axes[2].plot(distance, sample.surface_z, label='Reference ground at robot center', color='#219d25')
    axes[2].plot(distance, sample.z, label='Measured body height', color='#334d80')
    axes[2].set(xlabel='Distance along measured path (m)', ylabel='World height (m)')
    axes[2].legend(loc='upper left', fontsize=9)
    if exterior is not None:
        for ax in axes:
            ax.axvline(exterior.end_m.iloc[-1], color='#777777', ls='--', alpha=.7)
    for ax in axes:
        ax.grid(alpha=.2); ax.set_xlim(0, length)
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(1, 20), cmap='plasma'), ax=axes[1], pad=.02)
    colorbar.set_label('Section mean terrain cost')
    fig.suptitle(f'Manual ascent: total displayed terrain cost = {total:.2f} cost·m\n'
                 f'∫ C ds | mean {total/length:.2f} | wheelbase {wheelbase:.3f} m | dashed: first radius 11 m', fontsize=14)
    fig.savefig(out/'total_path_cost.png', dpi=160); plt.close(fig)
    return result


def analyze(directory):
    out = directory / 'analysis'
    out.mkdir(exist_ok=True)
    meta = json.loads((directory / 'run/metadata.json').read_text())
    capture = json.loads((directory / 'manual_capture.json').read_text())
    raw = pd.read_csv(directory / 'run/samples.csv')
    nav = pd.read_csv(directory / 'run.csv')
    mapped = pd.read_csv(directory / 'map_samples.csv')
    events = [json.loads(v) for v in (directory / 'run/events.jsonl').read_text().splitlines()]
    dt, cd, decimation = meta['physics_dt_s'], meta['control_dt_s'], meta['decimation']
    required = ['sample', 'sim_time_s', 'dt_s', 'episode', 'control_step', 'substep',
                'vx_command_m_s', 'omega_command_rad_s']
    required += [v for v in raw.columns if v.startswith(('root_', 'contact.'))]
    assert np.isfinite(raw[required].to_numpy()).all(), 'Missing required motion/contact data'
    assert np.all(np.diff(raw['sample']) == 1), 'Physics sample gaps'
    assert np.allclose(np.diff(raw.sim_time_s), dt) and np.allclose(raw.dt_s, dt)
    assert raw.episode.nunique() == 1 and not any(v['event'] == 'auto_reset' for v in events)
    sizes = raw.groupby('control_step').size()
    partial = sizes[sizes != decimation]
    assert len(partial) == 0 or (len(partial) == 1 and partial.index[0] == sizes.index[-1]
                                and partial.iloc[0] < decimation)
    assert np.array_equal(raw.substep, raw.groupby('control_step').cumcount())
    data = raw[raw.control_step.isin(sizes[sizes == decimation].index)].copy()
    assert data.groupby('control_step')[['vx_command_m_s', 'omega_command_rad_s', 'policy']].nunique().max().max() == 1
    c = data.groupby('control_step', sort=False).last()
    assert np.array_equal(nav.step, c.index) and np.array_equal(mapped.step, c.index)
    assert np.allclose(nav[['vx_cmd', 'omega_cmd']], c[['vx_command_m_s', 'omega_command_rad_s']])
    assert np.allclose(mapped[['vx', 'omega']], c[['vx_command_m_s', 'omega_command_rad_s']])
    xyz = c[[f'root_position_w_m.{a}' for a in 'xyz']].to_numpy()
    assert np.allclose(nav[['x', 'y', 'z']], xyz, atol=.00051)
    assert np.allclose(mapped[['x', 'y', 'z']], xyz, atol=1e-7)
    assert not mapped.reset.any()
    c['t'] = (c.index.to_numpy() + 1) * cd
    for i, a in enumerate('xyz'):
        c[a] = xyz[:, i]
    c['vx'] = c.vx_command_m_s
    c['omega'] = c.omega_command_rad_s
    qw, qx, qy, qz = (c[f'root_quaternion_w.{a}'].to_numpy() for a in 'wxyz')
    c['yaw'] = np.unwrap(np.arctan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz)))
    c['tilt'] = np.degrees(np.arccos(np.clip(1-2*(qx*qx+qy*qy), -1, 1)))
    c['forward_speed'] = (c['root_linear_velocity_w_m_s.x'] * np.cos(c.yaw)
                          + c['root_linear_velocity_w_m_s.y'] * np.sin(c.yaw))
    c['speed_xy'] = np.hypot(c['root_linear_velocity_w_m_s.x'], c['root_linear_velocity_w_m_s.y'])
    c['radius'] = np.hypot(c.x, c.y)
    c['mode'] = np.select([(c.vx == 0) & (c.omega == 0), c.vx > 0, c.vx < 0],
                          ['zero input', 'forward', 'reverse'], default='turn only')
    for k in ['cost', 'slope_deg', 'along_deg', 'cross_deg', 'footprint_rms_cm', 'step_cm', 'blocked']:
        c[k] = mapped[k].to_numpy()
    wheels = [j for j in meta['joint_names'] if '_foot_' in j]
    contacts = []
    for j in wheels:
        body = j.replace('_joint', '')
        force = data[[f'contact.{body}.{a}_N' for a in ('fx_w', 'fy_w', 'fz_w')]].to_numpy()
        contacts.append(np.linalg.norm(force, axis=1) > 5)
    data['loaded_wheels'] = np.sum(contacts, axis=0)
    data['wheel_speed'] = data[[j+'.velocity_rad_s' for j in wheels]].abs().mean(axis=1)
    c['wheel_speed'] = data.groupby('control_step').wheel_speed.mean()
    c['loaded_wheels'] = data.groupby('control_step').loaded_wheels.mean()
    first = int(c.index[(c.vx != 0) | (c.omega != 0)][0])
    start = max(0, first-1)
    active = c.loc[start:]
    exterior = active[active.radius >= 11]
    exit_step = int(exterior.index[0]) if len(exterior) else None
    with np.load(directory / 'display_map.npz', allow_pickle=False) as archive:
        amap = {k: archive[k].copy() for k in ('cost', 'surface', 'blocked', 'known')}
        cfg = json.loads(archive['metadata_json'].item())['config']
    lower = np.asarray(cfg['origin']) - cfg['world_size']/2
    cell = np.floor((xyz[:, :2]-lower) / cfg['cell_size']).astype(int)
    valid = (cell >= 0).all(axis=1) & (cell < amap['cost'].shape).all(axis=1)
    c['surface_z'] = np.nan
    c.loc[valid, 'surface_z'] = amap['surface'][cell[valid, 0], cell[valid, 1]]
    # Exact per-step attribution: command at this step, displacement from preceding endpoint.
    c['dz'] = c.z.diff().fillna(0)
    c['dxy'] = np.hypot(c.x.diff(), c.y.diff()).fillna(0)
    segment_id = c[['vx', 'omega']].ne(c[['vx', 'omega']].shift()).any(axis=1).cumsum()
    segments = []
    for _, g in c.groupby(segment_id):
        before = c.loc[max(0, int(g.index[0])-1)]
        after = g.iloc[-1]
        segments.append(dict(start_s=float(g.t.iloc[0]-cd), end_s=float(after.t),
            duration_s=len(g)*cd, vx=float(after.vx), omega=float(after.omega),
            mode=after['mode'], dz_m=float(after.z-before.z),
            displacement_xy_m=float(np.hypot(after.x-before.x, after.y-before.y)),
            peak_tilt_deg=float(g.tilt.max()), mean_cost=float(g.cost.mean()),
            mean_wheel_abs_rad_s=float(g.wheel_speed.mean()),
            start_xy=[float(before.x), float(before.y)], end_xy=[float(after.x), float(after.y)]))
    pd.DataFrame(segments).to_csv(out/'command_segments.csv', index=False)
    zero = sorted([s for s in segments if s['mode'] == 'zero input' and s['duration_s'] >= 1
                   and s['start_s'] >= c.loc[start, 't']], key=lambda s: s['dz_m'], reverse=True)
    # 10 Hz path distance reduces body vibration contributions; raw data remains at 200 Hz.
    slim = active.iloc[::5]
    path_length = float(np.linalg.norm(np.diff(slim[['x', 'y']], axis=0), axis=1).sum())
    active = c.loc[start:]
    cost_summary = cost_analysis(out, active, amap, cfg, exit_step)
    mode_stats = {name: dict(seconds=len(g)*cd, net_body_dz_m=float(g.dz.sum()))
                  for name, g in c.loc[first:exit_step].groupby('mode')}
    summary = dict(run=str(directory), physics_hz=1/dt, control_hz=1/cd,
        raw_samples=len(raw), retained_samples=len(data), excluded_partial_tail=len(raw)-len(data),
        complete_control_steps=len(c), recorded_duration_s=float(c.t.iloc[-1]),
        no_sample_gaps=True, pose_and_command_logs_agree=True, reset_events=0,
        policies=c.policy.unique().tolist(),
        close_marker_present=any(v['event'] == 'capture_end' for v in events),
        shutdown='Requested SIGINT stopped the process; final close marker was not written. Last incomplete control step excluded.',
        first_command_s=float(c.loc[first, 't']-cd),
        exterior_criterion='First radius >= 11 m, geometric reference only; final position and terrain shown separately.',
        exterior_time_s=float(c.loc[exit_step, 't']) if exit_step is not None else None,
        elapsed_to_exterior_s=float(c.loc[exit_step, 't']-c.loc[first, 't']+cd) if exit_step is not None else None,
        settled_start_xyz=c.loc[start, ['x', 'y', 'z']].tolist(),
        final_xyz=c.iloc[-1][['x', 'y', 'z']].tolist(), final_radius_m=float(c.radius.iloc[-1]),
        peak_body_z_m=float(active.z.max()), body_height_gain_to_peak_m=float(active.z.max()-c.loc[start, 'z']),
        peak_tilt_after_first_command_deg=float(active.tilt.max()),
        path_length_10hz_m=path_length, command_modes_to_exterior=mode_stats,
        best_zero_input_segments=zero[:5], missing_columns=raw.columns[raw.isna().any()].tolist(),
        max_command_m_s=float(c.vx.abs().max()), max_command_rad_s=float(c.omega.abs().max()),
        displayed_path_cost=cost_summary,
        files_sha256={str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [directory/'run/samples.csv', directory/'run/metadata.json',
                                directory/'run/events.jsonl', directory/'run.csv', directory/'map_samples.csv',
                                directory/'display_map.npz', directory/'manual_capture.json']},
        limits=['Single human-driven ascent, not autonomous success or calibrated success probability.',
                'Initial spawn drop/settling excluded from ascent gain; all raw data preserved.',
                'Cost and terrain values are frozen reference-map estimates, not traction measurements.',
                'Contact loading is force magnitude >5 N; exact slip requires contact kinematics and friction data.',
                'Wheel torque/power are actuator-model estimates. Raw LiDAR frames were not recorded.',
                'Zero-input versus powered intervals have different terrain/posture; not a causal comparison.'])
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    cols = ['t', 'x', 'y', 'z', 'yaw', 'vx', 'omega', 'mode', 'forward_speed', 'speed_xy', 'tilt',
            'surface_z', 'cost', 'slope_deg', 'along_deg', 'cross_deg', 'wheel_speed', 'loaded_wheels']
    c[cols].to_csv(out/'trajectory.csv', index_label='control_step')
    c.loc[start:, cols].to_csv(out/'baseline_route.csv', index_label='control_step')
    (out/'baseline.json').write_text(json.dumps(dict(
        purpose='Measured human route for offline comparison only; never a supplied autonomous route.',
        trajectory='baseline_route.csv', source_run=str(directory), seed=capture['seed'],
        task=capture['task'], policy=c.policy.unique().tolist(),
        start_xyz=summary['settled_start_xyz'], start_yaw_rad=float(c.loc[start, 'yaw']),
        actual_exit_xyz=summary['final_xyz'], first_exterior_time_s=summary['exterior_time_s'],
        reference_map_sha256=summary['files_sha256']['display_map.npz'],
        close_marker_present=summary['close_marker_present'],
        startup_excluded=True, raw_files_unchanged=True), indent=2)+'\n')
    render(out, c, data, amap, cfg, start, exit_step, zero)
    text = [f'# Recorded manual ascent: {directory.name}', '',
        f'- Exterior reached at {summary["exterior_time_s"]:.2f} simulated seconds; '
        f'{summary["elapsed_to_exterior_s"]:.2f} seconds after first command.',
        f'- Final XY: ({c.x.iloc[-1]:.3f}, {c.y.iloc[-1]:.3f}) m; radius {c.radius.iloc[-1]:.2f} m.',
        f'- Settled body height {c.loc[start,"z"]:.3f} m → peak {active.z.max():.3f} m '
        f'({summary["body_height_gain_to_peak_m"]:.3f} m gain).',
        f'- {len(data):,} verified physics samples at {1/dt:g} Hz; {len(c):,} complete control steps at {1/cd:g} Hz.',
        '- No gaps, resets or policy switches; all completed pose/command logs agree.',
        f'- Shutdown has no final close marker; {len(raw)-len(data)} partial-step samples excluded. Raw files unchanged.',
        '- Automatic wheel-contact count column unavailable; individual foot forces are recorded and used instead.', '',
        '## Total displayed path cost', '',
        f'**{cost_summary["total_displayed_cost_m"]:.2f} cost·m** over '
        f'**{cost_summary["distance_m"]:.2f} m**; distance-weighted mean **{cost_summary["distance_weighted_mean_cost"]:.2f}**.', '',
        'Definition: ∫ C(x,y) ds on the frozen displayed map, integrating every cell crossing along '
        'the 10 Hz measured XY polyline. Starts immediately before the first command and ends at the '
        'last complete sample. Initial settling is excluded; actual drift and repositioning count, '
        'stationary time does not. Bar sections use the nominal 0.451 m wheelbase; the final section is shorter.', '',
        f'To the first 11 m exterior-radius crossing: **{cost_summary["to_first_exterior_cost_m"]:.2f} cost·m** '
        f'over **{cost_summary["to_first_exterior_distance_m"]:.2f} m**. At full 50 Hz, total is '
        f'{cost_summary["full_50hz_total_cost_m"]:.2f} cost·m '
        f'({cost_summary["sampling_difference_percent"]:+.2f}% sampling difference).', '',
        'This is the integral of the visible terrain costs, not the experimental sustained-climb '
        'planner score, energy, or success probability. No extra directional, turn, history or mask penalty '
        'is added. Compare machine and human paths using the same map, endpoints and metric.', '',
        '![Total displayed path cost and wheelbase sections](total_path_cost.png)', '',
        '[Cost summary](path_cost_summary.json) · [Cell-crossing profile](path_cost_profile.csv) · '
        '[Wheelbase sections](wheelbase_cost_sections.csv)', '',
        '## Plots', '', '![Recorded route](route_cost_map.png)', '', '![Motion](motion_and_commands.png)', '',
        '![Release detail](zero_input_detail.png)', '', '## Measured zero-input intervals', '']
    for s in zero[:5]:
        text.append(f'- {s["start_s"]:.2f}–{s["end_s"]:.2f}s: '
                    f'{s["dz_m"]:+.3f} m body-height change, {s["displacement_xy_m"]:.3f} m net XY displacement.')
    text += ['', '## Interpretation limits', ''] + ['- '+v for v in summary['limits']]
    text += ['', 'For a matched autonomous comparison, use this measured settled start and actual exit. '
             'The recorded trace remains evaluation evidence, not an autonomous route input.']
    experience = Path(__file__).resolve().parents[1]/'docs/validation/manual_ascent_2026-10-24_experience.md'
    if experience.exists():
        text += ['', experience.read_text().strip()]
    (out/'report.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('files_sha256', 'limits')}, indent=2))


def render(out, c, data, amap, cfg, start, exit_step, zero):
    green = '#42ee20'
    active = c.loc[start:]
    lower = np.asarray(cfg['origin']) - cfg['world_size']/2
    n = amap['cost'].shape[0]
    xs = lower[0]+(np.arange(n)+.5)*cfg['cell_size']
    ys = lower[1]+(np.arange(n)+.5)*cfg['cell_size']
    norm = Normalize(1, 20)
    rgba = plt.cm.plasma(norm(amap['cost'].T))
    rgba[amap['blocked'].T] = [.22, .24, .27, 1]
    rgba[~amap['known'].T] = [.85, .85, .85, 1]
    bounds = (min(-6, active.x.min()-1), max(8, active.x.max()+1), min(-2, active.y.min()-1), max(15, active.y.max()+1))
    fig = plt.figure(figsize=(15, 10), layout='constrained')
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(rgba, origin='lower', extent=[lower[0], lower[0]+cfg['world_size'], lower[1], lower[1]+cfg['world_size']])
    contours = ax.contour(xs, ys, amap['surface'].T, levels=np.arange(0, 7), colors='white', linewidths=.6, alpha=.6)
    ax.clabel(contours, fmt='%g m', fontsize=8)
    ax.plot(active.x, active.y, color=green, lw=2.5, label='Your measured route')
    for idx, label in [(start, 'START'), (exit_step, 'r = 11 m'), (int(c.index[-1]), 'END')]:
        if idx is None: continue
        p = c.loc[idx]
        ax.scatter(p.x, p.y, color='white', edgecolor='black', s=35, zorder=8)
        ax.annotate(label, (p.x, p.y), xytext=(5, 7), textcoords='offset points', color='white', weight='bold', fontsize=9)
    for t in range(30, int(active.t.iloc[-1]), 30):
        p = c.iloc[np.argmin(abs(c.t.to_numpy()-t))]
        ax.annotate(f'{t}s', (p.x, p.y), xytext=(5, -10), textcoords='offset points', color='white', fontsize=8)
    ax.set(xlim=bounds[:2], ylim=bounds[2:], xlabel='World X (m)', ylabel='World Y (m)', title='Frozen reference costs + height contours')
    ax.legend(loc='lower left', fontsize=9)
    bx = fig.add_subplot(gs[0, 1], projection='3d', computed_zorder=False)
    ix = np.flatnonzero((xs >= bounds[0]) & (xs <= bounds[1]))
    iy = np.flatnonzero((ys >= bounds[2]) & (ys <= bounds[3]))
    X, Y = np.meshgrid(xs[ix], ys[iy])
    bx.plot_surface(X, Y, amap['surface'][np.ix_(ix, iy)].T, facecolors=rgba[np.ix_(iy, ix)],
                    shade=False, linewidth=0, antialiased=False, rcount=len(iy), ccount=len(ix), zorder=1)
    bx.plot(active.x.iloc[::5], active.y.iloc[::5], active.surface_z.iloc[::5]+.12, color=green, lw=3, zorder=5)
    bx.view_init(elev=35, azim=-65)
    bx.set(xlabel='World X (m)', ylabel='World Y (m)', zlabel='Terrain height (m)', title='Same costs in 3D; measured XY projected onto terrain')
    cx = fig.add_subplot(gs[1, :])
    sample = active.iloc[::5]
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(sample[['x', 'y']], axis=0), axis=1))]
    cx.plot(distance, sample.surface_z, color='#28a418', label='Reference ground at robot center')
    cx.plot(distance, sample.z, color='#334d80', label='Measured robot body height', alpha=.9)
    cx.set(xlabel='Distance along recorded path (m; 10 Hz positions)', ylabel='World height (m)')
    cx.grid(alpha=.2); cx.legend(fontsize=9)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='plasma'), ax=[ax, bx], orientation='horizontal', fraction=.045, pad=.06)
    cb.set_label('Unchanged terrain cost: purple low → yellow high (1–20). Gray: reference planning exclusions.')
    fig.suptitle('Successful manual ascent — measured route and terrain costs', fontsize=16, weight='bold')
    fig.savefig(out/'route_cost_map.png', dpi=160); plt.close(fig)

    fig, axes = plt.subplots(6, 1, figsize=(15, 15), sharex=True, layout='constrained')
    axes[0].step(c.t, c.vx, where='post', label='Forward command (m/s)', color='#15756d')
    axes[0].step(c.t, c.omega, where='post', label='Yaw command (rad/s)', color='#af6725')
    axes[0].set_ylabel('Command'); axes[0].legend(loc='upper left', ncol=2)
    axes[1].plot(c.t, c.z, label='Body height', color='#334d80')
    axes[1].plot(c.t, c.surface_z, label='Reference ground height', color='#28a418')
    axes[1].set_ylabel('World Z (m)'); axes[1].legend(loc='upper left', ncol=2)
    axes[2].plot(c.t, c.forward_speed, color='#355e85', label='Measured forward speed')
    axes[2].step(c.t, c.vx, color='#dd732c', alpha=.8, label='Command')
    axes[2].set_ylabel('Speed (m/s)'); axes[2].legend(loc='upper left', ncol=2)
    axes[3].plot(c.t, c['root_angular_velocity_w_rad_s.z'], label='Measured world yaw rate', color='#355e85')
    axes[3].step(c.t, c.omega, label='Command', color='#dd732c', alpha=.8)
    axes[3].set_ylabel('Yaw rate (rad/s)'); axes[3].legend(loc='upper left', ncol=2)
    axes[4].plot(c.t, c.tilt, label='Body tilt', color='#334d80')
    axes[4].plot(c.t, c.along_deg, label='Reference along-heading grade', color='#b77d19')
    axes[4].plot(c.t, c.cross_deg, label='Reference cross slope', color='#7d4197', alpha=.6)
    axes[4].set_ylabel('Angle (degrees)'); axes[4].legend(loc='upper left', ncol=3)
    axes[5].scatter(c.t, c.cost, c=c.cost, cmap='plasma', norm=norm, s=3)
    axes[5].set(ylabel='Reference cost', xlabel='Simulated time (s)', ylim=(0, 21))
    for ax in axes:
        ax.axvspan(0, c.loc[start, 't'], color='gray', alpha=.15)
        if exit_step is not None: ax.axvline(c.loc[exit_step, 't'], color='#269e25', ls='--', alpha=.8)
        ax.grid(alpha=.2); ax.set_xlim(0, c.t.iloc[-1])
    fig.suptitle('Manual commands and measured response | gray: startup settling; dashed green: radius 11 m', fontsize=14)
    fig.savefig(out/'motion_and_commands.png', dpi=140); plt.close(fig)

    if zero:
        best = zero[0]; a, b = best['start_s'], best['end_s']
        d = data[data.sim_time_s.between(max(0, a-5), b+5)]
        f = c[c.t.between(max(0, a-5), b+5)]
        fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True, layout='constrained')
        axes[0].step(f.t, f.vx, label='Forward m/s', color='#15756d')
        axes[0].step(f.t, f.omega, label='Yaw rad/s', color='#af6725')
        axes[0].set_ylabel('Command'); axes[0].legend()
        axes[1].plot(d.sim_time_s, d['root_position_w_m.z'], color='#334d80'); axes[1].set_ylabel('Body Z (m)')
        for wheel in [j for j in data.columns if '_foot_joint.velocity_rad_s' in j]:
            axes[2].plot(d.sim_time_s, d[wheel].abs(), lw=.6, alpha=.7, label=wheel.split('_')[0])
        axes[2].set_ylabel('|Wheel speed| (rad/s)'); axes[2].legend(ncol=4)
        axes[3].plot(d.sim_time_s, d.loaded_wheels, lw=.7, color='#555555')
        axes[3].set(ylabel='Loaded wheels\n(force >5 N)', xlabel='Simulated time (s)', ylim=(-.2, 4.2))
        for ax in axes:
            ax.axvspan(a, b, color='#5291dc', alpha=.15); ax.grid(alpha=.2)
        fig.suptitle(f'Largest measured zero-input climb: {a:.2f}–{b:.2f} s | +{best["dz_m"]:.2f} m Z\n'
                     '200 Hz physics; policy remains active. This does not establish wheel slip or a causal advantage.', fontsize=13)
        fig.savefig(out/'zero_input_detail.png', dpi=150); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    analyze(parser.parse_args().directory.resolve())
