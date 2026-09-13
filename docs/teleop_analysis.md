# First teleop capture — evidence-first hardware report, revision C

## Current report: revision C replaces the hypothetical narrative

- Open `reports/rexmi-hardware-report-revC.html` (same content as the canonical report). It contains five log-driven figures, all16 joint statistics, knee clipping/overload intervals, wheel-contact percentages/forces, non-wheel contacts, coverage and caveats.
- Removed the150mm support-moment calculator, mission-to-force section, illustrative pin/bushing sizing, battery/cost examples and other unrelated report sections. Older revisionB analysis below remains historical context, not the current report layout.
- Important correction: the installed `ContactSensorData.net_forces_w` is the sum of **normal** contact-force vectors in world coordinates, excluding tangential/friction forces. Old capture metadata used the broader phrase 'net contact force'; raw metadata is preserved, and revisionC corrects its interpretation.
- New exports: `wheel_contact_counts.csv`, `wheel_contact_forces.csv`, `nonwheel_contacts.csv`, `contact_threshold_sensitivity.csv`, `scenario_contact_counts.csv`, `contact_review.json`.
- Calf links registered contact; lower-head contact occurred at105.725/105.730s (peak731.10N). No base contact is not equivalent to no body collisions. Contact counterpart and impact classification require replay; netnormal vectors are not bearing reactions.
- Independent revisionC review:8.8/10, round1. Browser QA verified five embedded charts, zoom, no150mm/oldsections, no broken images/pageerrors/overflow at1440/736/390/320px. Source/prose regression tests pass.
- Current builder: `analysis/build_run_report.py` in the hardware repo (`scripts/build_run_report.py` in RL), using the raw capture directory and `--output <report.html> --export <analysis-directory> --review '8.8/10 · round 1'`. Interpretation is bound to this capture hash. Historical revisionB builder is not the current report generator.

- Run: `manual_20260909_01`, 159.14 simulated seconds, 31,828 rows, 16 joints, 200 Hz. No gaps or auto-resets; final capture_end/sample counts agree.
- Primary selection: actual timestamp >=2 s (31,428 samples, 157.14 s). All-capture statistics remain in a separate export. This explicit early-transient sensitivity cut does not classify every later sample as normal duty.
- Actual startup-randomized mass: 21.7695936 kg. Runtime root asset: rexmi_dog reskin, not directly the original Go2-W path. It is not a physically mounted 2 kg payload simulation.
- Raw `samples.csv`, `metadata.json`, `events.jsonl` remain untouched in the RL project. Do not put the ~104 MB raw CSV in ordinary Git. Summary includes all source SHA-256 hashes; retain original files/checkpoints/assets separately.

## What changed in the hardware decision

- Rear-right knee: 15.7978 Nm run RMS; 22.0848 Nm maximum 10-second RMS; 23.5 Nm clipped peak; 12.699% clipped time after the cut. Climbing-labelled subset RMS16.9569 Nm and clipping21.26%.
- With GDS68 nominal7.5Nm/max320rpm and assumed90% added-stage efficiency: RR knee RMS screen requires N>=2.34; speed endpoint requires N<=1.67. No fixed ratio closes both simplistic endpoints for unchanged recorded motion. Do not select2.5:1 or3:1 from torque alone.
- RMS is a finite-run thermal proxy, not a continuous rating. Nominal/stall/max-speed points do not define a motor curve. Low-voltage, cooling, permitted overload duration and transmission dynamics remain unqualified.
- Unclipped PD demands are not hardware requirements: after the cut, RR knee demand peaks41.81Nm; largest knee demand is FR46.55Nm. Applied torque is censored by caps.
- Directional asymmetry is evidence to repeat mirrored maneuvers, not a reason to select different left/right actuators now.
- 35° terrain capability, full60min duty, electrical/thermal behavior and bearing reactions remain unverified.

## Exports

- `summary.json`: provenance, integrity, metadata hashes, per-joint full/cut results, contacts and reduction assumptions.
- `joint_summary.csv` / `all_capture_joint_summary.csv`: RMS, P99, peak, first-peak time/scenario and paired end-substep speed, maximum speed, unclipped demand peak, clipping fraction/duration and highest10s RMS.
- `scenario_joint_summary.csv`: each joint by operator label and policy. Labels may include multiple motions; separated intervals are not used as contiguous overload/windows.
- `overload_durations.csv`: duration above absolute applied-estimate thresholds5/7.5/8/12/16/20/23Nm and longest contiguous time, not bridging resets.
- `reduction_screen.csv`: ratios1/1.5/2/2.5/3, eta1 for no added stage and eta0.9 otherwise. Forward-motoring magnitude screen only, not bidirectional loss/thermal simulation.
- Capture metadata/events are archived with these exports in the hardware repo. The standalone hardware report embeds plots and key numerical tables; no network is needed to view it.

## Limitations discovered in capture

- Implicit effort telemetry is a start-substep PD estimate; state is end-substep (~5ms offset). Torque-speed pairing and mechanical power are indicative, not measured synchronized motor effort or electrical energy.
- The wheel-count column was blank because body names use `*_foot`. Analysis reconstructs counts from all four saved foot force vectors with norm>5N; side/obstacle contact may count and equal sharing is not established. Original CSV is not repaired or relabelled.
- Old implicit `velocity_limit` requests are ignored by this installed Isaac Lab version; runtime recorded limits differ from source requests. Wheel speeds exceeded recorded~30.1rad/s, reaching59.85rad/s at low estimated torque. Diagnose enforcement/impact/external forces before changing frozen physics.
- Startup mass randomization and an unseeded terrain mean this is not yet a deterministic payload qualification run. Metadata records resolved event/actuator configurations.
- No base contact>5N or auto-reset was detected, but some fall terminations are disabled. Body tilt reached51.31°; this is not ground slope.
- Tk shutdown errored after telemetry files were closed. Files and sample counts were verified complete; this does not imply GUI teardown is fixed.

## Reproduce the numerical analysis

Use Python with numpy, pandas and matplotlib (available in the current `env_isaacsim`). From the RL repo:

```bash
python scripts/analyze_joint_telemetry.py logs/telemetry/manual_20260909_01 --output logs/telemetry/manual_20260909_01/analysis
python -m unittest discover -s tests -v
```

In the hardware repo the analyzer is at `analysis/analyze_joint_telemetry.py`; pass the raw run directory from your local RL checkout. HTML interpretation is deliberately bound to this CSV hash and cut; another capture gets numerical tables with interpretation pending, not copied conclusions. CSV exports remain reusable for compatible Go2-W captures. Report revision builder uses the preserved revision A report (hardware commit72842ca6) and generated `telemetry-section.html`.

## Review and next gates

- Independent systems/mechatronics review: **8.8/10, round1**, technical9.0 / narrative8.5 / traceability-coverage9.0. Independently recomputed knee, contact and reduction findings. No blocking technical corrections.
- Browser QA: no page errors, missing images/anchors or page-width overflow at1440/736/390/320px. Torque-speed and timeline charts visually inspected. Interactive slope/lever checks preserved.
- Next: validate effort estimates in a known-load fixture; diagnose velocity-limit behavior; freeze payload/COM and measured ramp; repeat mirrored/repeated scenarios with realistic separate actuator models; obtain measured motor torque-speed/overload/thermal curves. Preserve frozen baseline/checkpoints.
- Motor purchase, gearbox selection, CAD/PCB fabrication and battery sizing remain unreleased.
