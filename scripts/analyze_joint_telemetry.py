"""Reproducible offline sizing evidence, not motor qualification.

Requires numpy, pandas, matplotlib (available in env_isaacsim). Never changes raw
telemetry. Generates numerical exports and a self-contained HTML evidence section.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def longest_duration(mask, dt, episodes):
    """Do not bridge resets; caller must provide contiguous sampled intervals."""
    best = current = 0.0
    previous = None
    for active, duration, episode in zip(mask, dt, episodes):
        if episode != previous:
            current = 0.0
        current = current + duration if active else 0.0
        best = max(best, current)
        previous = episode
    return best


def metrics(frame, joint):
    a = frame[joint + '.torque_applied_est_Nm'].to_numpy()
    v = frame[joint + '.velocity_rad_s'].to_numpy()
    demand = frame[joint + '.torque_demand_est_Nm'].to_numpy()
    dt = frame.dt_s.to_numpy()
    clipped = frame[joint + '.torque_clipped_est'].to_numpy().astype(bool)
    k = int(np.argmax(np.abs(a)))
    rolling = []
    # Constant dt verified on input. Rolling windows never span episode resets.
    width = int(round(10 / dt[0]))
    for _, segment in frame.groupby('episode', sort=False):
        x = segment[joint + '.torque_applied_est_Nm'].to_numpy()
        if len(x) >= width:
            rolling.append(float(np.sqrt(pd.Series(x*x).rolling(width).mean().max())))
    return {
        'joint': joint, 'duration_s': float(dt.sum()),
        'rms_est_Nm': float(np.sqrt(np.average(a*a, weights=dt))),
        'p99_abs_est_Nm': float(np.quantile(abs(a), .99)),
        'peak_abs_est_Nm': float(abs(a[k])),
        'peak_first_time_s': float(frame.sim_time_s.iloc[k]),
        'peak_first_scenario': str(frame.scenario.iloc[k]),
        'speed_at_first_torque_peak_rad_s': float(v[k]),
        'max_abs_speed_rad_s': float(abs(v).max()),
        'max_abs_demand_est_Nm': float(abs(demand).max()),
        'clipped_time_pct': float(100*np.average(clipped, weights=dt)),
        'longest_clip_s': longest_duration(clipped, dt, frame.episode),
        'max_10s_rms_est_Nm': max(rolling) if rolling else None,
    }


def ratio_screen(rms, max_speed, ratio, nominal=7.5, module_max_rpm=320):
    efficiency = 1.0 if ratio == 1 else .9
    return {
        'additional_ratio': ratio, 'assumed_efficiency': efficiency,
        'joint_nominal_screen_Nm': nominal*ratio*efficiency,
        'joint_speed_endpoint_rad_s': module_max_rpm*2*np.pi/60/ratio,
        'module_rms_screen_Nm': rms/(ratio*efficiency),
        'module_max_speed_required_rpm': max_speed*ratio*60/(2*np.pi),
        'rms_endpoint_screen_pass': bool(rms <= nominal*ratio*efficiency),
        'speed_endpoint_screen_pass': bool(max_speed*ratio*60/(2*np.pi) <= module_max_rpm),
    }


def table(frame, columns, labels):
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return '—' if not np.isfinite(value) else f'{value:.2f}'
        return html.escape(str(value))
    head = ''.join(f'<th>{html.escape(x)}</th>' for x in labels)
    rows = ''.join('<tr>'+''.join(f'<td>{cell(row[c])}</td>' for c in columns)+'</tr>'
                   for _, row in frame.iterrows())
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>'


def image_html(fig, alt):
    import matplotlib.pyplot as plt
    stream = io.BytesIO()
    fig.savefig(stream, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    data = base64.b64encode(stream.getvalue()).decode('ascii')
    return f'<img style="width:100%;height:auto;display:block" alt="{html.escape(alt)}" src="data:image/png;base64,{data}">'


def build_section(frame, summary, joint_frame, scenarios, overloads, ratios):
    # The interpreted narrative below is a reviewed case study, not a generic
    # conclusion to stamp onto future captures. Generic exports remain usable.
    if (summary['source_hashes']['samples.csv'] != 'd5e7d9195589342a665e8be57260a9bfaef074c6acf6e041a510c53d5ba24f6b'
            or summary['analysis_sample_count'] != 31428):
        return '<section id="telemetry"><h2>New capture: interpretation pending</h2>'+table(
            joint_frame, ['joint','rms_est_Nm','peak_abs_est_Nm','max_abs_speed_rad_s'],
            ['Joint','RMS estimate Nm','Peak estimate Nm','Max speed rad/s'])+'</section>'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = ['#007b76', '#3159a5', '#b55423', '#913f84']
    names = summary['joint_names']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, kind, title in zip(axes.flat, ['hip', 'thigh', 'calf', 'foot'], ['Hip ab/ad', 'Thigh pitch', 'Knee pitch', 'Wheel drive']):
        for color, joint in zip(colors, [j for j in names if f'_{kind}_' in j]):
            ax.scatter(frame[joint+'.velocity_rad_s'], frame[joint+'.torque_applied_est_Nm'],
                       s=2, alpha=.15, color=color, rasterized=True)
            ax.plot([], [], '.', color=color, label=joint.split('_')[0])
        ax.set(title=title, xlabel='End-substep angular velocity (rad/s)', ylabel='Start-substep torque estimate (Nm)')
        ax.axhline(0, color='#aab4b6', lw=.5); ax.axvline(0, color='#aab4b6', lw=.5)
        ax.grid(alpha=.15); ax.legend(ncol=4, markerscale=1.2, fontsize=9)
    scatter = image_html(fig, 'All 16 joint torque estimates against angular velocity, grouped by joint type; all retained samples shown.')
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), constrained_layout=True, sharex=True)
    for color, joint in zip(colors, [j for j in names if '_calf_' in j]):
        axes[0].plot(frame.sim_time_s, abs(frame[joint+'.torque_applied_est_Nm']), lw=.55, color=color, label=joint.split('_')[0])
    axes[0].axhline(23.5, ls='--', color='black', lw=.8, label='23.5 Nm cap')
    axes[0].set_ylabel('|Knee torque estimate| (Nm)'); axes[0].legend(ncol=5, fontsize=9)
    axes[1].plot(frame.sim_time_s, frame['RR_calf_joint.torque_demand_est_Nm'], color=colors[3], lw=.7, label='RR demand estimate')
    axes[1].plot(frame.sim_time_s, frame['RR_calf_joint.torque_applied_est_Nm'], color=colors[0], lw=.7, label='RR clipped estimate')
    axes[1].set_ylabel('Rear-right knee (Nm)'); axes[1].legend(ncol=2, fontsize=9)
    axes[2].plot(frame.sim_time_s, frame.contact_count_reconstructed, lw=.5, color=colors[1])
    axes[2].set(yticks=range(5), ylabel='Contacts >5 N', xlabel='Capture time (s)')
    for ax in axes:
        ax.grid(alpha=.15)
        for item in summary['segments'][1:]:
            ax.axvline(item['start_s'], color='#888888', ls=':', lw=.7)
    timeline = image_html(fig, 'Knee torque histories, rear-right demand and applied estimates, reconstructed wheel-contact counts. Dotted lines mark label or policy transitions.')
    s = summary
    joint_table = table(joint_frame, ['joint','rms_est_Nm','max_10s_rms_est_Nm','peak_abs_est_Nm','max_abs_speed_rad_s','clipped_time_pct','longest_clip_s'],
                        ['Joint','Run RMS Nm','Max 10s RMS Nm','Peak Nm','Max speed rad/s','Clipped %','Longest clip s'])
    scenario_table = table(pd.DataFrame(s['segments']), ['label','policy','start_s','end_s','duration_s'], ['Operator label','Policy','Start s','End s','Duration s'])
    ratio_table = table(ratios, ['additional_ratio','joint_nominal_screen_Nm','module_rms_screen_Nm','module_max_speed_required_rpm','joint_speed_endpoint_rad_s','rms_endpoint_screen_pass','speed_endpoint_screen_pass'],
                        ['Added ratio','Joint nominal screen Nm','RR module RMS Nm','Required module rpm','Joint speed endpoint rad/s','RMS screen','Speed screen'])
    selected = overloads[overloads.joint == 'RR_calf_joint']
    duration_table = table(selected, ['threshold_Nm','total_above_s','time_above_pct','longest_above_s'], ['|Torque| threshold Nm','Total above s','Time above %','Longest continuous s'])
    contacts = ', '.join(f'{k} contacts: {v:.2f}%' for k,v in s['contact_pct'].items())
    return f'''<section id="telemetry"><div class="section-head"><span class="num">05A</span><h2>First policy-driven load evidence</h2></div>
<p class="lead">The knee is the governing actuator class in this run. The rear-right knee has <strong>15.80 Nm run RMS</strong>, a <strong>22.08 Nm highest 10-second RMS</strong>, and repeated clipping at 23.5 Nm. More reduction trades that torque problem for a speed problem.</p>
<div class="grid3"><div class="panel"><div class="kicker">Complete capture</div><div class="big">{s['duration_s']:.2f}<span class="unit"> s</span></div><p>{s['sample_count']:,} samples · 16 joints · 200 Hz · no gaps or automatic resets in this run.</p></div><div class="panel"><div class="kicker">Actual randomized mass</div><div class="big">{s['live_mass_kg']:.2f}<span class="unit"> kg</span></div><p>Not the 21.523 kg design reference, and not an explicitly mounted 2 kg payload.</p></div><div class="panel"><div class="kicker">Rear-right knee clipping</div><div class="big">12.70<span class="unit">%</span></div><p>21.26% during “climbing crater”; longest uninterrupted clipped interval 1.36 s in the analyzed capture.</p></div></div>
<div class="callout caution"><strong>Evidence class: simulated actuator-model estimates.</strong> Implicit PD effort is estimated at substep start; velocity is sampled at its end, 5 ms later. These are indicative paired trajectories, not exact synchronous motor measurements. Applied peaks are censored by simulation caps. Unclipped PD demand is not a procurement torque rating.</div>
<h3>What was driven</h3>{scenario_table}
<p class="caption">Labels are operator annotations, not automatic motion classifications. “initial_standing” includes driving and post-release settling. Full capture is retained. Primary statistics select timestamps ≥2 s ({s['analysis_duration_s']:.2f} s, {s['analysis_sample_count']:,} samples), excluding an early placement/landing transient by an explicit sensitivity cut—not claiming every later sample is steady or valid duty. The companion joint summary also includes all-capture statistics. No 35° terrain measurement or complete 60-minute duty was recorded. Runtime asset was the <code>rexmi_dog</code> reskin of Go2-W, not a replay directly loading the original <code>go2w</code> path.</p>
<h3>All joints: observed demand, not selected motor ratings</h3>{joint_table}
<p class="caption">FL/FR/RL/RR = front-left/front-right/rear-left/rear-right. Calf joints are knees; foot joints drive wheels. Torque and speed maxima need not coincide. P99, speed at first torque peak and uncapped-demand peaks are retained in the numerical exports. RMS is a heating proxy for this run; the worst 10-second RMS is a finite-duration load, not a continuous rating.</p>
<h3>Torque–speed trajectories</h3>{scatter}
<p class="caption">Every retained sample is plotted; color identifies the leg. No full motor torque–speed curve has been assumed. Clusters at ±23.5 Nm (legs) and ±23.7 Nm (wheels) are clipping boundaries, not proof that the policy would never require more.</p>
<h3>Where the knee load comes from</h3>{timeline}
<p>Rear-right knee RMS rises to <strong>16.96 Nm</strong> during the climbing label. Its longest clipping interval is at approximately 6.645–8.000 s, with speed −1.69 to +0.84 rad/s: a low-speed overload, not simply a fast-motion spike. Rear-right uncapped demand reaches 41.81 Nm after the cut; the largest uncapped knee demand is front-right at 46.55 Nm. Neither is a validated hardware requirement.</p>
<h3>Overload duration matters</h3>{duration_table}
<p class="caption">Rear-right knee, |applied estimate| strictly above threshold, using per-sample dt. Contiguous durations do not bridge resets. Full per-joint duration exports cover the same thresholds. Clipping flag uses demand/applied mismatch, not this threshold test.</p>
<div class="grid"><div><h3>Three-leg support is not guaranteed</h3><p>{contacts}.</p><p>The original wheel-count column was blank because bodies are named <code>*_foot</code>, not <code>*wheel*</code>. These counts are reconstructed from the saved world-frame foot force vectors, norm &gt;5 N. Raw logs are unchanged. Side/obstacle contact can count; a force threshold is not a proof of support or load sharing.</p></div><div><h3>Limits and coverage</h3><ul><li>No base force above 5 N and no auto-reset were detected; this does not prove every maneuver was stable.</li><li>Maximum body tilt was {s['max_body_tilt_deg']:.2f}° over the full capture. Tilt is not terrain slope.</li><li>Front wheels reached 59.85 and 56.53 rad/s at low estimated torque, above their recorded ~30.1 rad/s limit. Old <code>velocity_limit</code> is ignored for implicit actuators in this installed version; actual overshoot cause/enforcement remains to be diagnosed.</li><li>No winding temperatures, electrical current, battery voltage, bearing reactions or true ground slope were logged.</li></ul></div></div>
<h3>Can additional knee reduction rescue GIM8108-8?</h3>
<p>Screen against the manufacturer's GDS68 output endpoints: 7.5 Nm nominal and 320 rpm maximum, already after its internal 8:1 gearing. Use 90% assumed efficiency for an added stage; ratio 1 means no added stage and uses 100%. Apply the conservative forward-motoring transformation to torque magnitude for screening only; it is not a bidirectional loss or thermal model.</p>
<div class="eq">τ<sub>module,RMS</sub> ≈ τ<sub>joint,RMS</sub> / (ηN)<br>ω<sub>module,max</sub> = Nω<sub>joint,max</sub><br>N<sub>RMS screen</sub> ≥ 15.80 / (0.90 × 7.5) = 2.34<br>N<sub>speed screen</sub> ≤ (320 × 2π/60) / 20.05 = 1.67</div>
{ratio_table}
<div class="callout caution"><strong>No ratio in this screen satisfies both endpoints.</strong> A 2.5:1 stage brings RR module RMS to about 7.02 Nm but needs about 479 rpm; 3:1 needs about 574 rpm. Even a hypothetical true 8 Nm module with the same speed endpoint needs N≥2.19 on the RMS screen. Do not select 2.5:1 or 3:1 from torque alone. A finite 157 s RMS is not proof of thermal failure: the mismatch means this candidate is not yet supported for unchanged motion, not that every finite maneuver is impossible.</div>
<p>Vendor endpoint screens are necessary screening checks under the stated assumptions, not a feasible torque–speed envelope. Passing either check does not establish available torque at speed, low-voltage performance, winding temperature, allowable peak duration or transmission dynamics. Source: <a href="https://www.steadywin.cn/en/pd.jsp?fromColId=0&amp;id=133">SteadyWin GIM8108-8 manufacturer table</a>, rechecked for this revision.</p>
<h3>Revised design decisions</h3><ul><li><strong>Knees:</strong> shortlist a higher-capacity/faster actuator-plus-transmission assembly; retain GIM8108 only as an unqualified comparison or restricted-motion experiment. No purchase release.</li><li><strong>Hips and thighs:</strong> evaluate separately against the exported trajectories; maxima reaching the cap still need validation. Avoid selecting different left/right hardware from a single asymmetric course.</li><li><strong>Wheels:</strong> validate the 30 rad/s intended envelope and high-speed events before selecting hub gearing.</li><li><strong>Keep Go2-W link lengths fixed.</strong> Evaluate motor size/location, transmission ratio and stiffness first; feed added mass/inertia/friction/backlash into a separate replay configuration.</li><li><strong>Next replay:</strong> controlled payload/COM, measured 35° surface, explicit realistic actuator torque–speed limits, mirrored ascent/descent/turns and repeated seeds. Preserve this baseline; do not silently change frozen controller physics.</li><li><strong>Next bench:</strong> known-load effort calibration, full torque–speed and peak-duration curves, low-bus-voltage tests and thermal model in the intended mount.</li></ul>
<details><summary>Traceability and reproducibility</summary><p>Run: <code>{html.escape(s['run_id'])}</code>. Raw CSV SHA-256: <code style="overflow-wrap:anywhere">{s['source_hashes']['samples.csv']}</code>. Metadata and events hashes, retained/excluded counts, all-capture sensitivity, scenario-by-joint results, overload durations and reduction assumptions are in the accompanying analysis exports. The raw ~104 MB CSV remains local and was not embedded or uploaded to GitHub. Exported joint order and metadata checkpoint/root-asset hashes identify the captured configuration; referenced USD layers still need their own archive.</p><p>Clean capture_end event and final row count agree. Tk reported a shutdown error after log files were closed; no row loss is indicated. Full capture remains unchanged.</p></details>
</section>'''


def analyze(run, output, exclude_initial_s=2.0):
    output.mkdir(parents=True, exist_ok=True)
    m = json.loads((run/'metadata.json').read_text())
    events = [json.loads(line) for line in (run/'events.jsonl').read_text().splitlines()]
    raw = pd.read_csv(run/'samples.csv')
    names = m['joint_names']
    if not len(raw) or not np.all(np.diff(raw['sample']) == 1):
        raise ValueError('Empty or noncontiguous capture')
    dt = float(raw.dt_s.iloc[0])
    if not np.allclose(raw.dt_s, dt) or not np.allclose(np.diff(raw.sim_time_s), dt):
        raise ValueError('This analysis requires constant-dt, gap-free telemetry')
    for suffix in ['torque_applied_est_Nm', 'torque_demand_est_Nm', 'velocity_rad_s']:
        if not np.isfinite(raw[[j+'.'+suffix for j in names]].to_numpy()).all():
            raise ValueError('Nonfinite required joint signal')
    for _, control in raw.groupby('control_step'):
        if list(control.substep) != list(range(m['decimation'])):
            raise ValueError('Incomplete physics substeps')
    if events[-1]['event'] != 'capture_end' or events[-1]['samples'] != len(raw):
        raise ValueError('Capture end/sample count mismatch')
    frame = raw[raw.sim_time_s >= exclude_initial_s].copy()
    if frame.empty:
        raise ValueError('Initial exclusion removed all samples')
    # Explicit mapping is deliberate: this Go2-W asset calls its wheel bodies foot.
    feet = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
    forces = np.stack([frame[[f'contact.{name}.{axis}_N' for axis in ['fx_w','fy_w','fz_w']]].to_numpy() for name in feet], axis=1)
    frame['contact_count_reconstructed'] = (np.linalg.norm(forces, axis=2) > 5).sum(axis=1)
    joints = pd.DataFrame([metrics(frame, j) for j in names])
    all_joints = pd.DataFrame([metrics(raw, j) for j in names])
    rows = []
    for (scenario, policy), group in frame.groupby(['scenario','policy'], sort=False):
        for joint in names:
            # Aggregate RMS is valid; don't invent contiguous windows across gaps.
            stat = metrics(group, joint)
            stat.pop('longest_clip_s'); stat.pop('max_10s_rms_est_Nm')
            rows.append({'scenario':scenario, 'policy':policy, **stat})
    scenarios = pd.DataFrame(rows)
    rows = []
    for joint in names:
        a = abs(frame[joint+'.torque_applied_est_Nm'].to_numpy())
        for threshold in [5,7.5,8,12,16,20,23]:
            mask = a > threshold
            rows.append({'joint':joint, 'threshold_Nm':threshold,
                         'total_above_s':float(frame.dt_s.to_numpy()[mask].sum()),
                         'time_above_pct':float(np.average(mask, weights=frame.dt_s)*100),
                         'longest_above_s':longest_duration(mask, frame.dt_s, frame.episode)})
    overloads = pd.DataFrame(rows)
    rr = joints.set_index('joint').loc['RR_calf_joint']
    ratios = pd.DataFrame([ratio_screen(rr.rms_est_Nm, rr.max_abs_speed_rad_s, n) for n in [1,1.5,2,2.5,3]])
    segments = []
    transitions = raw[['scenario','policy']].ne(raw[['scenario','policy']].shift()).any(axis=1).cumsum()
    for _, g in raw.groupby(transitions):
        segments.append({'label':str(g.scenario.iloc[0]), 'policy':str(g.policy.iloc[0]),
                         'start_s':float(g.sim_time_s.iloc[0]), 'end_s':float(g.sim_time_s.iloc[-1]), 'duration_s':float(g.dt_s.sum())})
    up = 1 - 2*(raw['root_quaternion_w.x']**2 + raw['root_quaternion_w.y']**2)
    summary = {
        'schema_version':1, 'run_id':run.name, 'sample_count':len(raw), 'duration_s':float(raw.dt_s.sum()),
        'analysis_sample_count':len(frame), 'analysis_duration_s':float(frame.dt_s.sum()),
        'selection':f'sim_time_s >= {exclude_initial_s}; actual floating timestamps, no rounding',
        'live_mass_kg':sum(m['live_body_masses_kg']), 'joint_names':names, 'segments':segments,
        'max_body_tilt_deg':float(np.degrees(np.arccos(np.clip(up,-1,1))).max()),
        'contact_pct':{str(i):float(np.average(frame.contact_count_reconstructed==i, weights=frame.dt_s)*100) for i in range(5)},
        'contact_mapping':feet, 'contact_threshold_N':5,
        'source_hashes':{name:sha256(run/name) for name in ['samples.csv','metadata.json','events.jsonl']},
        'analyzer_sha256':sha256(__file__), 'checkpoints':m['checkpoints'], 'robot_asset':m['robot_asset'],
        'torque_semantics':m['torque_semantics'], 'auto_resets':sum(e['event']=='auto_reset' for e in events),
        'notes':'Finite, asymmetric teleop run; no thermal, electrical, exact ground slope or bearing-reaction validation. Endpoint reduction screening is not a motor envelope.',
        'joints':joints.to_dict('records'), 'all_capture_joints':all_joints.to_dict('records'),
        'reduction_screen':ratios.to_dict('records'),
    }
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    for filename, data in [('joint_summary',joints),('all_capture_joint_summary',all_joints),
                           ('scenario_joint_summary',scenarios),('overload_durations',overloads),('reduction_screen',ratios)]:
        data.to_csv(output/(filename+'.csv'), index=False)
    section = build_section(frame, summary, joints, scenarios, overloads, ratios)
    (output/'telemetry-section.html').write_text(section)
    print(json.dumps({'output':str(output), 'samples':len(raw), 'retained':len(frame), 'reduction_screen':ratios.to_dict('records')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--exclude_initial_s', type=float, default=2.)
    args = parser.parse_args()
    analyze(args.run, args.output, args.exclude_initial_s)
