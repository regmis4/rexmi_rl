"""Evidence-first hardware report from teleop logs. Replaces the legacy report.

No hypothetical lever arms, inferred terrain forces or selected motor ratings.
All plots/tables use saved actuator estimates and world-frame body contact forces.
"""
from __future__ import annotations
import argparse
import base64
import html
import io
import json
import re
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_joint_telemetry import metrics, sha256, longest_duration, table


STYLE = '''
:root{--ink:#172d39;--muted:#536874;--line:#d6e0e2;--teal:#007b76;--orange:#ae491f;--bg:#f2f5f4}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,sans-serif}main,header>div,nav>div{max-width:1120px;margin:auto}header{background:#102c39;color:white;padding:36px 22px}h1{font-size:clamp(30px,5vw,52px);line-height:1.08;letter-spacing:-.035em;margin:14px 0}header p{max-width:850px;color:#cbe0e1}.eyebrow{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:#92d2c8}.tags{display:flex;flex-wrap:wrap;gap:8px}.tags span{border:1px solid #69818b;border-radius:4px;padding:3px 8px;font-size:12px}nav{padding:14px 22px;background:white;border-bottom:1px solid var(--line)}nav>div{display:flex;flex-wrap:wrap;gap:12px 25px}a{color:var(--teal);text-underline-offset:3px}nav a{font-size:14px;font-weight:650;text-decoration:none}main{padding:0 22px 48px}section{padding:30px 0;border-bottom:1px solid var(--line);scroll-margin-top:20px}h2{font-size:27px;line-height:1.2;margin:0 0 18px}h3{font-size:18px;margin:22px 0 10px}p{margin:9px 0 15px}li{margin:6px 0}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:15px}.card{background:white;border:1px solid var(--line);border-radius:7px;padding:20px}.label{font-size:12px;text-transform:uppercase;color:var(--muted);letter-spacing:.06em}.value{font-size:38px;font-weight:680;line-height:1.3;letter-spacing:-.03em}.value small{font-size:15px;font-weight:500}.note{padding:16px 20px;background:#e2eeeb;border-left:4px solid var(--teal);margin:18px 0}.warn{background:#fff0e5;border-color:var(--orange)}.small,figcaption{font-size:13px;color:var(--muted)}.table-wrap{overflow:auto;background:white;border:1px solid var(--line);border-radius:6px}table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:10px 11px;text-align:left;border-bottom:1px solid #e2e8e8;vertical-align:top}th{font-size:12px;background:#eaf0f0;color:#3f5661;line-height:1.4}tbody tr:last-child td{border:0}td:first-child{font-weight:600}figure{margin:18px 0;padding:16px;background:white;border:1px solid var(--line);border-radius:6px}figure img{width:100%;height:auto;display:block;cursor:zoom-in}figcaption{margin-top:10px}button{font:inherit;padding:7px 12px;border:1px solid #79939c;background:transparent;color:inherit;border-radius:4px;cursor:pointer}header button{float:right}code{font-size:12px;overflow-wrap:anywhere}.wide{grid-template-columns:1fr 1fr}summary{font-weight:650;cursor:pointer}details{padding:13px 0}dialog{width:min(96vw,1600px);max-width:96vw;padding:15px;border:1px solid var(--line);border-radius:6px}dialog::backdrop{background:#102c39dd}dialog img{width:100%;height:auto}dialog button{margin-bottom:10px}footer{color:var(--muted);font-size:13px;padding:20px;text-align:center}a:focus-visible,button:focus-visible,summary:focus-visible{outline:3px solid #df984e;outline-offset:4px}
@media(max-width:700px){.grid,.wide{grid-template-columns:1fr}main{padding:0 15px 30px}figure{padding:6px}h2{font-size:23px}header button{float:none;margin-top:10px}.value{font-size:33px}}
@media print{header{background:white;color:#172d39;padding:20px}header p,.eyebrow{color:#536874}nav,button{display:none}main{padding:0}body{background:white;font-size:11px}h1{font-size:32px}h2{font-size:22px}section{padding:18px 0}.card,figure,.note{break-inside:avoid}th,td{padding:7px}.table-wrap{overflow:visible}table{font-size:10px}figure{padding:8px}*{-webkit-print-color-adjust:exact;print-color-adjust:exact}}
'''


def polish_text(source):
    """Space prose without touching image data, CSS, scripts or source hashes."""
    class Formatter(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=False)
            self.parts=[]; self.protected=[]
        def handle_starttag(self, tag, attrs):
            self.parts.append(self.get_starttag_text())
            if tag in ('style','script','code'):self.protected.append(tag)
        def handle_endtag(self, tag):
            self.parts.append(f'</{tag}>')
            if self.protected and self.protected[-1]==tag:self.protected.pop()
        def handle_data(self, data):
            if not self.protected:
                data=re.sub(r'(?<=[a-z])(?=\d)', ' ', data)
                data=re.sub(r'(?<=\d)(?=(?:Nm|rad/s|Hz|kg|ms|MB|s|N)\b)', ' ', data)
            self.parts.append(data)
        def handle_entityref(self, name):self.parts.append('&'+name+';')
        def handle_charref(self, name):self.parts.append('&#'+name+';')
        def handle_decl(self, decl):self.parts.append('<!'+decl+'>')
    parser=Formatter();parser.feed(source);parser.close()
    return ''.join(parser.parts)


def figure_html(fig, title, caption):
    import matplotlib.pyplot as plt
    stream=io.BytesIO()
    # A compact, high-quality scientific plot; embedded for offline viewing.
    for artist in fig.findobj():
        if hasattr(artist, 'get_text') and hasattr(artist, 'set_text'):
            value=artist.get_text()
            value=re.sub(r'(?<=[a-z])(?=\d)', ' ', value)
            value=re.sub(r'(?<=\d)(?=(?:Nm|rad/s|Hz|kg|ms|MB|s|N)\b)', ' ', value)
            artist.set_text(value)
    fig.savefig(stream, format='webp', dpi=140, bbox_inches='tight', pil_kwargs={'quality':92})
    plt.close(fig)
    encoded=base64.b64encode(stream.getvalue()).decode()
    return f'<figure><img tabindex="0" alt="{html.escape(title)}" src="data:image/webp;base64,{encoded}"><figcaption>{html.escape(caption)} Click a chart to enlarge.</figcaption></figure>'


def build(run, output, export, review='Pending independent review'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    raw=pd.read_csv(run/'samples.csv')
    if sha256(run/'samples.csv') != 'd5e7d9195589342a665e8be57260a9bfaef074c6acf6e041a510c53d5ba24f6b':
        raise ValueError('Revision C interpretation is specific to manual_20260909_01; use the generic analyzer for another capture.')
    meta=json.loads((run/'metadata.json').read_text())
    events=[json.loads(s) for s in (run/'events.jsonl').read_text().splitlines()]
    names=meta['joint_names']; dt=float(raw.dt_s.iloc[0])
    assert np.all(np.diff(raw['sample'])==1) and np.allclose(np.diff(raw.sim_time_s),dt)
    assert np.allclose(raw.dt_s,dt)
    assert events[-1]['event']=='capture_end' and events[-1]['samples']==len(raw)
    assert all(list(g.substep)==list(range(meta['decimation'])) for _,g in raw.groupby('control_step'))
    d=raw[raw.sim_time_s>=2].copy(); duration=float(d.dt_s.sum())
    assert len(d)>0
    stats=pd.DataFrame([metrics(d,j) for j in names]); keyed=stats.set_index('joint')
    assert np.isfinite(stats[['rms_est_Nm','peak_abs_est_Nm','max_abs_speed_rad_s']].to_numpy()).all()
    rr=keyed.loc['RR_calf_joint']; knee_names=[j for j in names if '_calf_' in j]
    feet=['FL_foot','FR_foot','RL_foot','RR_foot']
    f=np.stack([d[[f'contact.{b}.{a}_N' for a in ['fx_w','fy_w','fz_w']]].to_numpy() for b in feet],axis=1)
    assert np.isfinite(f).all()
    magnitude=np.linalg.norm(f,axis=2); counts=(magnitude>5).sum(axis=1)
    contacts=pd.DataFrame([{'contacts':n,'seconds':float(np.sum(counts==n)*dt),'percent':float(np.mean(counts==n)*100)} for n in range(5)])
    force_rows=[]
    for i,b in enumerate(feet):
        mask=magnitude[:,i]>5; k=int(np.argmax(magnitude[:,i]))
        force_rows.append({'body':b,'contact_pct':float(mask.mean()*100),
                           'mean_world_Fz_N':float(f[:,i,2].mean()),
                           'p95_force_norm_N':float(np.quantile(magnitude[:,i],.95)),
                           'max_force_norm_N':float(magnitude[k,i]),'peak_time_s':float(d.sim_time_s.iloc[k])})
    wheel_forces=pd.DataFrame(force_rows)
    nonwheel=[]
    for b in meta['contact_body_names']:
        if b in feet: continue
        xyz=d[[f'contact.{b}.{a}_N' for a in ['fx_w','fy_w','fz_w']]].to_numpy()
        mag=np.linalg.norm(xyz,axis=1); mask=mag>5
        if mask.any():
            nonwheel.append({'body':b,'contact_pct':float(mask.mean()*100),'samples_above_5N':int(mask.sum()),
                             'max_force_norm_N':float(mag.max()),'first_contact_s':float(d.sim_time_s[mask].iloc[0])})
    nonwheel=pd.DataFrame(nonwheel)
    sensitivity=pd.DataFrame([{'threshold_N':threshold,**{f'{n}_contacts_pct':float(np.mean((magnitude>threshold).sum(axis=1)==n)*100) for n in range(5)}} for threshold in [1,5,10,20]])
    contact_scenarios=[]
    for label,g in d.groupby('scenario',sort=False):
        n=counts[d.index.get_indexer(g.index)]
        contact_scenarios.append({'label':label,'duration_s':float(g.dt_s.sum()),**{f'{i}_contacts_pct':float(np.mean(n==i)*100) for i in range(5)}})
    contact_scenarios=pd.DataFrame(contact_scenarios)
    segments=[]
    seq=raw[['scenario','policy']].ne(raw[['scenario','policy']].shift()).any(axis=1).cumsum()
    for _,g in raw.groupby(seq):
        segments.append({'label':g.scenario.iloc[0],'policy':g.policy.iloc[0],'start_s':float(g.sim_time_s.iloc[0]),'duration_s':float(g.dt_s.sum())})
    segments=pd.DataFrame(segments)
    colors=['#007b76','#3159a5','#b55423','#913f84']
    groups=['hip','thigh','calf','foot']
    fig,ax=plt.subplots(figsize=(11,6),constrained_layout=True)
    y=np.arange(len(stats)); palette=[colors[groups.index(j.split('_')[1])] for j in names]
    ax.barh(y,stats.rms_est_Nm,color=palette,height=.65)
    ax.scatter(stats.max_10s_rms_est_Nm,y,c='#172d39',marker='|',s=160,label='Highest 10-second RMS')
    ax.set(yticks=y,yticklabels=[j.replace('_joint','') for j in names],xlabel='Torque estimate (Nm)',title='Per-joint run RMS and highest 10-second RMS')
    ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.legend(loc='lower right')
    rms_chart=figure_html(fig,'RMS torque comparison for all sixteen joints','Bars: RMS over the retained157.14s. Black ticks: highest contiguous10s RMS. Neither is a measured continuous motor rating.')
    fig,axes=plt.subplots(2,1,figsize=(11,6.5),constrained_layout=True)
    for ax,frame,title in [(axes[0],d,'Rear-right knee: entire retained capture'),(axes[1],d[(d.sim_time_s>=6)&(d.sim_time_s<=8.5)],'Zoom: sustained clipping at low speed, around6.645–8.000s')]:
        ax.plot(frame.sim_time_s,frame['RR_calf_joint.torque_demand_est_Nm'],lw=.7,color='#913f84',label='Demand estimate')
        ax.plot(frame.sim_time_s,frame['RR_calf_joint.torque_applied_est_Nm'],lw=.8,color='#007b76',label='Applied estimate')
        ax.axhline(23.5,ls='--',color='#b55423',lw=1,label='23.5Nm cap')
        ax.set(title=title,xlabel='Capture time (s)',ylabel='Torque estimate (Nm)');ax.grid(alpha=.15);ax.legend(ncol=3,fontsize=9)
    knee_chart=figure_html(fig,'Rear-right knee demand and applied torque estimates','The capped trace hides unmet PD demand. Unclipped demand is diagnostic—not a motor procurement specification. The zoom is an observed interval, not a hypothetical loading case.')
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for ax,kind,title in zip(axes.flat,groups,['Hip ab/ad','Thigh pitch','Knee pitch','Wheel drive']):
        for color,j in zip(colors,[j for j in names if f'_{kind}_' in j]):
            ax.scatter(d[j+'.velocity_rad_s'],d[j+'.torque_applied_est_Nm'],s=2,alpha=.18,color=color)
            ax.plot([],[],'.',color=color,label=j[:2])
        ax.set(title=title,xlabel='Angular velocity (rad/s)',ylabel='Torque estimate (Nm)');ax.grid(alpha=.15);ax.legend(ncol=4,fontsize=8)
    envelope_chart=figure_html(fig,'All saved torque-speed pairs for all joints','Every retained sample is shown. Effort is evaluated at substep start and velocity read at its end—a5ms offset. These are indicative torque-speed traces, not exactly synchronized shaft measurements.')
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    axes[0].bar(contacts.contacts,contacts.percent,color=['#b55423','#bd734e','#3159a5','#007b76','#51998e'])
    for _,row in contacts.iterrows():axes[0].text(row.contacts,row.percent+1,f'{row.percent:.1f}%',ha='center',fontsize=10)
    axes[0].set(xticks=range(5),xlabel='Wheel bodies with force norm >5N',ylabel='Time (%)',ylim=(0,60),title='How many wheels were contacting?')
    axes[1].bar(wheel_forces.body.str[:2],wheel_forces.contact_pct,color=colors)
    for i,row in wheel_forces.iterrows():axes[1].text(i,row.contact_pct+1,f'{row.contact_pct:.1f}%',ha='center',fontsize=10)
    axes[1].set(ylabel='Time above5N (%)',ylim=(0,100),title='Contact duty by wheel')
    contact_chart=figure_html(fig,'Wheel contact count percentages and contact duty by wheel','Counts are reconstructed from saved *_foot force vectors. Contact does not prove stable support, equal load sharing, or contact with the ground rather than an obstacle.')
    fig,axes=plt.subplots(2,1,figsize=(11,6),constrained_layout=True,sharex=True)
    for i,color in enumerate(colors):axes[0].plot(d.sim_time_s,magnitude[:,i],color=color,lw=.45,label=feet[i][:2])
    axes[0].set(ylabel='Net contact-force norm (N)',title='Directly logged wheel-body contact forces');axes[0].legend(ncol=4,fontsize=9)
    for b in nonwheel.body:
        xyz=d[[f'contact.{b}.{a}_N' for a in ['fx_w','fy_w','fz_w']]].to_numpy()
        axes[1].plot(d.sim_time_s,np.linalg.norm(xyz,axis=1),lw=.6,label=b)
    axes[1].set(xlabel='Capture time (s)',ylabel='Net normal-force norm (N)',title='Non-wheel contacts were also recorded');axes[1].legend(ncol=3,fontsize=8)
    for ax in axes:ax.grid(alpha=.15)
    force_chart=figure_html(fig,'Logged wheel, calf and lower-head normal contact forces','These are net normal contact-force vectors in world coordinates. Tangential/friction forces are NOT included in net_forces_w. Brief peaks are not sustained loads, bearing reactions, or individual contact-patch forces.')
    mass=sum(meta['live_body_masses_kg']); under3=float(np.mean(counts<3)*100)
    climb=d[d.scenario=='climbing crater']; climb_rms=metrics(climb,'RR_calf_joint')['rms_est_Nm']
    report=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>REXMI · Recorded hardware loads · Revision C</title><style>{STYLE}</style></head><body>
<header><div><div class="eyebrow">REXMI / Hardware evidence / Revision C</div><button onclick="window.print()">Print / save PDF</button><h1>What the robot actually asked<br>of its joints.</h1><p>One teleop run. All16 joints. Recorded torque estimates, angular velocities and contact forces. This report replaces the previous hypothetical sizing narrative.</p><div class="tags"><span>{html.escape(run.name)}</span><span>{len(raw):,} samples · {1/dt:.0f}Hz · {raw.dt_s.sum():.2f}s</span><span>{mass:.2f}kg runtime mass</span><span id="review">{html.escape(review)}</span></div></div></header>
<nav><div><a href="#findings">Findings</a><a href="#joints">All joints</a><a href="#knee">Limiting knee</a><a href="#contacts">Contacts &amp; forces</a><a href="#coverage">Run coverage</a><a href="#caveats">Caveats / next decisions</a></div></nav><main>
<section id="findings"><h2>The limiting joint in this run: rear-right knee</h2><div class="grid"><div class="card"><div class="label">Rear-right knee run RMS</div><div class="value">{rr.rms_est_Nm:.2f}<small> Nm</small></div><p>Highest10-second RMS: <strong>{rr.max_10s_rms_est_Nm:.2f}Nm</strong>.</p></div><div class="card"><div class="label">Rear-right knee clipped time</div><div class="value">{rr.clipped_time_pct:.2f}<small>%</small></div><p>Applied estimate repeatedly reaches the <strong>23.5Nm cap</strong>.</p></div><div class="card"><div class="label">Fewer than three wheel contacts</div><div class="value">{under3:.2f}<small>%</small></div><p>At a5N contact threshold. Guaranteed three-leg support is not an assumption this run supports.</p></div></div>
<div class="note"><strong>What changes:</strong> size and compare each joint against its recorded trajectory and overload duration. The knee is the priority here; hips, thighs and wheels have separate demands. A single asymmetric run does not justify different left/right hardware.</div>
<p class="small">All headline and table statistics use actual timestamps ≥2s: {len(d):,} samples / {duration:.2f}s. The full{raw.dt_s.sum():.2f}s capture is preserved; the excluded first2s contain the early placement/landing transient. This is a sensitivity cut, not proof that every remaining sample is normal operation.</p></section>
<section id="joints"><h2>All16 joints: loads and speeds</h2>{rms_chart}
{table(stats,['joint','rms_est_Nm','max_10s_rms_est_Nm','peak_abs_est_Nm','max_abs_speed_rad_s','clipped_time_pct','longest_clip_s'],['Joint','Run RMS Nm','Highest10s RMS Nm','Peak estimate Nm','Max |speed| rad/s','Clipped time %','Longest clip s'])}
<p class="small">FL/FR/RL/RR denote the four legs. Calf=knee; foot=wheel drive. Torque and speed maxima are generally not simultaneous. Clipped torque peaks are lower bounds on the controller’s uncapped request—not proof of sufficient actuator capacity. RMS is a finite-run thermal proxy, not a continuous motor rating.</p>
<h3>Recorded torque–speed trajectories</h3>{envelope_chart}</section>
<section id="knee"><h2>Why the rear-right knee is the priority</h2><ul><li>Run RMS <strong>{rr.rms_est_Nm:.2f}Nm</strong>; climbing-labelled RMS <strong>{climb_rms:.2f}Nm</strong>.</li><li>Longest uninterrupted clipping: <strong>{rr.longest_clip_s:.2f}s</strong>. Around6.645–8.000s, joint speed was −1.69 to +0.84rad/s: a low-speed overload, not only a fast impact spike.</li><li>Rear-right uncapped demand reached <strong>{rr.max_abs_demand_est_Nm:.2f}Nm</strong>; front-right knee demand reached <strong>{keyed.loc['FR_calf_joint','max_abs_demand_est_Nm']:.2f}Nm</strong>. Those are PD requests, not validated torque requirements.</li></ul>{knee_chart}
<h3>Observed rear-right overload durations</h3>'''
    overload=pd.DataFrame([{'threshold_Nm':x,'seconds_above':float(np.sum(abs(d['RR_calf_joint.torque_applied_est_Nm'])>x)*dt),'longest_s':longest_duration(abs(d['RR_calf_joint.torque_applied_est_Nm'])>x,d.dt_s,d.episode)} for x in [7.5,8,12,16,20,23]])
    report+=table(overload,['threshold_Nm','seconds_above','longest_s'],['Absolute torque threshold Nm','Total above threshold s','Longest continuous interval s'])
    report+=f'''<p class="small">Thresholds help compare future actuator curves; they are not selected motor ratings. Durations use applied estimates and therefore remain censored at the simulation cap.</p></section>
<section id="contacts"><h2>Actual contacts and contact forces</h2>{contact_chart}
{table(contacts,['contacts','seconds','percent'],['Wheel contacts >5N','Time s','Time %'])}
<h3>Which wheels carried contact force?</h3>{table(wheel_forces,['body','contact_pct','mean_world_Fz_N','p95_force_norm_N','max_force_norm_N','peak_time_s'],['Wheel body','Contact time %','Mean world Fz N','P95 force norm N','Peak force norm N','Peak time s'])}
<p class="small">Mean Fz includes all retained samples, including no-contact samples. It is the world-vertical component of the summed normal-contact vectors, not the full surface-normal magnitude on a slope. Percentile/peak columns use the3D norm of that vector. <strong>Tangential/friction forces are not included.</strong> Do not treat these as total reactions or bearing loads, or divide robot weight by contact count.</p>
{force_chart}<h3>Non-wheel collisions matter</h3>{table(nonwheel,['body','contact_pct','samples_above_5N','max_force_norm_N','first_contact_s'],['Body','Time >5N %','Samples >5N','Peak force norm N','First contact s'])}
<div class="note warn"><strong>No base contact did not mean no body collisions.</strong> All four calf links registered contact. Lower-head contact occurred for two samples, at105.725 and105.730s, with peak731.10N. The log does not identify the collision counterpart. These brief solver-force peaks need replay inspection before being assigned to structural load cases.</div>
<details><summary>Contact count by operator scenario</summary>{table(contact_scenarios,['label','duration_s','0_contacts_pct','1_contacts_pct','2_contacts_pct','3_contacts_pct','4_contacts_pct'],['Label','Duration s','0 contacts %','1 contact %','2 contacts %','3 contacts %','4 contacts %'])}</details>
<details><summary>How much does the contact threshold change the result?</summary>{table(sensitivity,['threshold_N','0_contacts_pct','1_contacts_pct','2_contacts_pct','3_contacts_pct','4_contacts_pct'],['Threshold N','0 contacts %','1 contact %','2 contacts %','3 contacts %','4 contacts %'])}<p class="small">The exact percentages depend on threshold; absence of guaranteed three-wheel contact persists across these thresholds. Side/obstacle contact can count, and a lightly touching wheel need not carry meaningful support load.</p></details>
<p class="small">The logger’s wheel-count column was blank because this asset calls wheel bodies *_foot. Counts here are reconstructed from saved FL_foot/FR_foot/RL_foot/RR_foot vectors, norm&gt;5N. Multiple contact normals can partially cancel on the same body, so this is a force-based activity indicator, not a contact-patch census. Raw logs are unchanged.</p></section>
<section id="coverage"><h2>What this run actually covered</h2>{table(segments,['label','policy','start_s','duration_s'],['Operator label','Policy','Start time s','Duration s'])}
<ul><li>Labels are annotations, not verified maneuver classes. “initial_standing” includes driving.</li><li>No automatic resets; sample indices and physics substeps are complete.</li><li>Runtime asset: <strong>rexmi_dog Go2-W reskin</strong>. Total mass <strong>{mass:.4f}kg</strong> came from existing startup randomization—not a mounted2kg payload.</li><li>This run does not prove35° terrain capability or the60-minute duty target. Body tilt is not terrain angle.</li></ul></section>
<section id="caveats"><h2>What these data establish—and what they do not</h2><div class="grid wide"><div class="card"><h3>Use now</h3><ul><li>Prioritize the knee actuator/transmission investigation.</li><li>Compare each joint’s torque–speed trace, RMS and overload duration.</li><li>Use logged body-contact forces to identify collision events for replay and structural analysis.</li><li>Reject guaranteed three-leg support and equal load-sharing assumptions for this policy.</li></ul></div><div class="card"><h3>Do not claim yet</h3><ul><li>Physical motor torque: implicit actuator outputs are PD estimates.</li><li>Continuous thermal capability, electrical power, bus current or battery sizing.</li><li>Qualified bearings, shafts or links from net body-contact forces alone.</li><li>Hardware-ready motion or a released motor/gearbox selection.</li></ul></div></div>
<div class="note warn"><strong>Speed-envelope issue:</strong> front wheels reached59.85 and56.53rad/s, above recorded~30.1rad/s limits, at low estimated torque. The installed implicit actuator ignores old velocity_limit requests; the cause of the observed overspeed still needs diagnosis. Do not size gearing around assumed30rad/s enforcement or silently alter the frozen baseline.</div>
<p><strong>Signal alignment:</strong> torque estimates are evaluated at substep start; velocities are sampled at substep end, about5ms later. The plotted pairs and mechanical power are indicative, not exact synchronous shaft measurements. Applied peaks are clipped, while unclipped requests include controller error and cannot simply become a motor nameplate requirement.</p>
<h3>Next decisions supported by this evidence</h3><ol><li>Validate effort estimates against a known-load case; inspect the clipping, overspeed and calf/head collision timestamps in replay.</li><li>Repeat mirrored ascent/descent/turns with controlled payload/COM and a measured ramp. Preserve this run as baseline.</li><li>Compare candidate motor/transmission curves against the full trajectories and durations. More knee reduction is not automatically sufficient: speed, losses, inertia and thermal behavior must close together.</li></ol>
<details><summary>Traceability, exports and review</summary><p>CSV SHA-256: <code>{sha256(run/'samples.csv')}</code>. Source metadata/checkpoints and numerical exports: <a href="https://github.com/regmis4/rexmi_hardware/tree/main/analysis/teleop_20260909_01">private hardware evidence folder</a>. Raw~104MB CSV remains local. All plots are embedded and use saved data; no hypothetical lever-arm, mission-force or component-sizing charts remain.</p><p>Report builder: <code>analysis/build_run_report.py</code>. Primary selection: actual timestamp≥2s. Full-capture sensitivity is in all_capture_joint_summary.csv. Contacts use3D norm of net normal force. Installed ContactSensorData documents that net_forces_w excludes tangential forces; this corrects the earlier generic “net contact force” description without modifying capture metadata. These are not joint reactions or individual contact patches.</p><p>Capture_end count agrees with the CSV; Tk teardown errored after the logger closed its files. This does not indicate lost rows.</p><p>Independent review: {html.escape(review)}. Review score concerns analysis/report quality, not hardware certification.</p></details></section>
</main><footer>REXMI · Revision C · Recorded simulation evidence, not fabrication release</footer>
<dialog id="zoom"><button id="closeZoom">Close chart</button><img alt="Enlarged chart"></dialog><script>
const zoom=document.getElementById('zoom');for(const img of document.querySelectorAll('figure img')){{const show=()=>{{zoom.querySelector('img').src=img.src;zoom.querySelector('img').alt=img.alt;zoom.showModal();}};img.addEventListener('click',show);img.addEventListener('keydown',e=>{{if(e.key==='Enter')show();}});}}document.getElementById('closeZoom').addEventListener('click',()=>zoom.close());
</script></body></html>'''
    report=polish_text(report)
    output.write_text(report)
    if output.name=='rexmi-hardware-report.html':
        output.with_name('rexmi-hardware-report-revC.html').write_text(report)
    export.mkdir(parents=True,exist_ok=True)
    for name,frame in [('wheel_contact_counts',contacts),('wheel_contact_forces',wheel_forces),('nonwheel_contacts',nonwheel),('contact_threshold_sensitivity',sensitivity),('scenario_contact_counts',contact_scenarios)]:
        frame.to_csv(export/(name+'.csv'),index=False)
    evidence={'run':run.name,'report_revision':'C','source_csv_sha256':sha256(run/'samples.csv'),
              'builder_sha256':sha256(__file__),'retained_samples':len(d),'duration_s':duration,'selection':'sim_time_s >= 2',
              'contact_threshold_N':5,'contact_semantics':'net_forces_w is the sum of normal contact forces in world frame; excludes tangential/friction forces. Norm can reflect cancellation of multiple normals.',
              'wheel_bodies':feet,'review':review,'contacts':contacts.to_dict('records'),
              'nonwheel_contacts':nonwheel.to_dict('records')}
    (export/'contact_review.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps({'report':str(output),'bytes':output.stat().st_size,'exports':str(export),'under_three_contacts_pct':under3}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--export',required=True,type=Path);p.add_argument('--review',default='Pending independent review')
    a=p.parse_args();build(a.run,a.output,a.export,a.review)
