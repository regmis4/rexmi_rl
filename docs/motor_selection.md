# Prototype motor selection — combined teleop evidence

> Superseded motor baseline: see [RobStride motor selection](reports/robstride-motor-selection.html) and [combined test report Rev D](reports/rexmi-hardware-report.html). Current leg choice is RS06 direct for hips/thighs and RS06 plus 2:1 belt for knees. This earlier CubeMars assessment is retained as history, not the current BOM.

Decision date: 2026-09-10. Status: leg design baseline selected for CAD and bench qualification; wheel SKU unresolved. No purchase release and no simulator settings changed.

## Accepted requirement

- Preserve the demonstrated **23.5 Nm leg command cap** and **23.7 Nm wheel cap**. Do not size actuators to the uncapped 65.86 Nm PD request.
- The near-vertical interpretation is plausible but unverified: terrain slope was not measured. The spike remains in the evidence, not the rated-torque requirement.
- Preserve all four quadrants of measured torque–speed trajectories, including braking. Torque cap acceptance does not establish a speed-dependent motor envelope or structural impact limit.

## Combined evidence

- Two independent captures, 79,788 original samples. Exclude the first 2 seconds of each from primary statistics: **78,988 samples, 394.94 seconds**, 200 Hz.
- Run masses differ: **21.77 kg and 18.89 kg**. Do not interpret their concatenation as a standardized mission or normalize loads by mass alone.
- Joint-level exports retain pooled RMS, worst individual-run RMS, maximum within-run 10-second RMS, clipping duration and paired peak-speed data. Windows never cross capture boundaries. Original raw files are unchanged.

| Joint family | Worst individual-run RMS, Nm | Worst 10-second RMS, Nm | Applied peak, Nm | Maximum speed, rad/s |
|---|---:|---:|---:|---:|
| Hip ab/ad | 8.43 | 13.73 | 23.50 | 18.66 |
| Thigh pitch | 7.03 | 12.76 | 23.50 | 30.05 |
| Knee | 15.80 | 22.08 | 23.50 | 20.09 |
| Wheel | 5.50 | 9.59 | 23.70 | 79.00 |

Each column is the family maximum, not necessarily the same joint, sample or operating point. RMS is a finite-run heating proxy, not a proven continuous rating.

- Rear-right knee: pooled RMS 14.18 Nm, but retain **15.80 Nm** for the worst-run screen. Longest clipped interval **1.36 seconds**; pooled clipping **7.75%**.
- Contact activity: **0 contacts 0.57%, 1 contact 4.44%, 2 contacts 25.50%, 3 contacts 52.54%, 4 contacts 16.94%**. Fewer than three occurs **30.51%** of retained time.
- Contact definition: saved foot normal-force-vector magnitude >5 N. Not proof of stable support, equal sharing, or full reaction force; tangential friction force is absent.

## Selected leg design baseline

Use identical hardware for all four joints of each family. Values below are manufacturer module-output specifications; internal gearing is already included.

| Position | Quantity | Selected actuator | Internal ratio | Additional ratio | Published rated / peak torque | Rated / no-load speed at 48 V |
|---|---:|---|---:|---:|---:|---:|
| Hip ab/ad | 4 | CubeMars AK80-8 KV60 | 8:1 | 1:1, no extra stage | 10 / 25 Nm | 25.45 / 37.70 rad/s |
| Thigh pitch | 4 | CubeMars AK10-9 V2.0 KV100 | 9:1 | 1:1, no extra stage | 15 / 38 Nm | 44.09 / 55.82 rad/s |
| Knee | 4 | CubeMars AK10-9 V3.0 KV60 | 9:1 | 1:1, no extra stage | 18 / 53 Nm | 24.61 / 33.51 rad/s |

- **Hips:** smaller 570 g module. 10 Nm nominal exceeds 8.43 Nm worst-run RMS, but 13.73 Nm / 10-second load requires overload qualification. Only 6.4% nameplate peak margin over the command cap; do not spend that margin on an unmodeled transmission loss.
- **Thighs:** choose the faster KV100 winding, not the knee's KV60 by default. Recorded RR thigh point is **19.83 Nm at +30.05 rad/s**, motoring. This warrants the faster, higher-power module; a low-load/no-load speed comparison alone would miss it. V2 KV100 stock, firmware support and full loaded curve must be confirmed before ordering.
- **Knees:** 18 Nm nominal exceeds the worst-run 15.80 Nm RMS. Worst 10-second RMS is 22.08 Nm, **1.23× nominal**, requiring thermal/overload testing. Keep command cap 23.5 Nm despite the module's higher nameplate peak.
- **No extra knee reduction now.** Direct module output retains speed and avoids additional reflected inertia, backlash and losses. A remote shaft or 1:1 bevel arrangement is a packaging alternative, not a proven stiffness improvement; rerun the loss/compliance calculation if adopted.
- Listed leg actuator mass totals **9.88 kg**, before mounts, shafts and cables. These replace modeled actuator/link mass, not add to the entire USD mass. CAD must reconcile total mass, each link inertia and motor placement. Do not assume Go2-W physics remains unchanged because link lengths match.
- 48 V is the catalog screening voltage, not an approved battery specification. Verify allowed bus range, low-charge performance and regenerative overvoltage before choosing series cell count.

## Why not the proposed GIM8108 with extra knee gearing?

For the manufacturer's 7.5 Nm, 320 rpm module variant and an assumed 90%-efficient added reduction N:

- Torque screen: N ≥ 15.80 / (0.90 × 7.5) = **2.34**.
- Speed screen: N ≤ (320 × 2π / 60) / 20.09 = **1.67**.
- No overlap. A true 8 Nm module with that speed still needs N ≥2.19. This conservative sustained screen does not prove every short maneuver impossible, but it does not support selecting it for unchanged motion.
- More gearbox reduction cannot solve both constraints. Retain the existing GIM6010 leg as a development fixture, not this robot's actuator baseline.

## Wheels: do not release an unsupported part

- Required observed output envelope includes 23.7 Nm peaks, 5.50 Nm worst-run RMS and 9.59 Nm worst 10-second RMS.
- FR wheel: **−21.02 Nm at −61.50 rad/s**, approximately **+1.29 kW mechanical motoring**, in run 2 at 154.255 s.
- RR wheel: **+18.62 Nm at −79.00 rad/s**, approximately **−1.47 kW mechanical braking**, in run 2 at 191.245 s. This is not electrical recovered power. RR also reaches **76.25 rad/s while motoring**.
- No shortlisted direct-output CubeMars module above supports this complete wheel speed range. AK80-6 has sufficient no-load speed but only 12 Nm peak, below the accepted wheel cap.
- A 0.7:1 external speed-increasing stage on AK10-9 V2 KV100 gives, at assumed 90% forward efficiency, 23.94 Nm peak and 79.74 rad/s no-load. Those barely passing separate endpoints **do not** provide the loaded wheel points. Reject this as a validated solution.
- **Wheel part and gearing remain open.** Next selection must use a faster motor plus custom reduction and a verified loaded torque–speed curve, or explicitly re-test reduced wheel dynamics. Do not silently discard these events or change the accepted simulation.

## Qualification before ordering the full set

- Overlay the actual paired trajectories on manufacturer loaded curves at minimum bus voltage. Published rated/peak/no-load endpoints are only preliminary screens; no claimed full-envelope pass yet.
- Replay measured waveforms on one hip and one knee/thigh candidate with winding temperature and calibrated output torque. Include 22.08 Nm knee 10-second RMS and the 1.36-second capped interval; test repeated duty in the intended enclosure.
- Test current/torque calibration, CAN timing, encoder direction, loss of communications, hard joint limits and independent emergency stop. Integrated CubeMars drive is the baseline; do not assume it is interchangeable with the existing ODrive interface.
- Structural qualification includes collision loads separately from command torque. Run 2 includes non-wheel contacts; the motor cap does not cap external shock loads on gears/bearings.
- Obtain a complete actuator quote before claiming the $5–10k robot budget closes. AK80-8 page lists $469.90 each; four hips alone total $1,879.60 before tax/shipping. Other selected configurations are not yet quoted.

## Sources and reproduction

- [AK80-8 manufacturer specification](https://www.cubemars.com/product/ak80-8-kv60-robotic-actuator.html), checked 2026-09-10.
- [AK10-9 V3 KV60 manufacturer specification](https://www.cubemars.com/product/ak10-9-v3-0-kv60-robotic-actuator.html), checked 2026-09-10.
- [Manufacturer comparison including AK10-9 V2 KV100](https://www.cubemars.com/de/goods-1175-AK10-9%2BV30.html), checked 2026-09-10; variant availability unconfirmed.
- [SteadyWin GIM8108-8 variant table](https://www.steadywin.cn/en/pd.jsp?fromColId=0&id=133).
- `motor_selection_evidence/manifest.json`: original capture identities, SHA-256 hashes, masses and aggregation method.
- `motor_selection_evidence/joint_envelope.csv`: all 16 combined joint records.
- `motor_selection_evidence/per_run_joint_metrics.csv`: 32 individual-run joint records.
- Reproduce from repository root: `python scripts/merge_joint_runs.py` with numpy and pandas installed.
- Torque fields are implicit-PD applied-effort estimates, not measured shaft torques. Velocity is sampled approximately 5 ms later; paired power/quadrant results are indicative. Neither capture proves the complete 60-minute mission, explicit payload COM, or measured 35° slope.
