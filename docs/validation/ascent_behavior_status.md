# Behavior-tree ascent implementation and characterization

The ascent controller now uses a reactive behavior tree and one exclusive
teleop-style movement state machine. Ascent qualification remains **not passed**.
The reachability model remains **unvalidated** for this execution protocol.
Descent and high-speed optimization have not started.

## Implemented behavior

- Rocky-slope uphill planning permits 33°, replacing the former 25° default
  and removing its additional one-degree reserve. Cross-slope capability stays
  independently configured at 25°; the overall terrain boundary remains 35°.
- Planning includes arrival heading and supported gentle staging. Every bend
  requires staging; steep sections use straight forward or reverse commands.
- STOP, faults and manual takeover have explicit priority. A running movement
  owns its direction, deadline, progress measurements and one terminal result.
  A new global plan cannot interrupt it.
- Alignment uses measured pose and the fixed endpoint. Remaining geometry is
  checked before alignment/drive. The checked heading remains latched during
  steep entry. Gentle corrections reset the travel-axis progress window.
- Pulse stopping uses measured heading error rather than an assumed coast
  response. The policy sometimes unwinds after release. Pulses retain their
  direction, have a two-second bound, and settle before reversal or driving.
- Useful supported displacement is a censored repositioning outcome. It can
  trigger fresh observation and replanning on gentle terrain, without falsely
  reporting checkpoint success or recording a physical failure.
- Command speed remains at most 0.40 m/s. Autonomous yaw is zero above 15°
  footprint terrain inclination. Turning resumes only on supported terrain at
  or below 12° with settled motion. Body posture has separate pause/resume gates.
- Body-heading deviation alone no longer creates a physical failure on steep
  terrain. Progress stays measured along the original committed axis. A checked
  straight action can resume after temporary braking without requesting a turn
  or resetting its direction, axis or deadline.
- Forward uphill movement can enter policy-active RELEASE. It retains zero
  command while measured progress remains useful and resumes the same checked
  drive when progress fades. Drive and release share their attempt, deadline
  and low-progress allowance. Prospective creep and actual drift are checked.
- A soft diagonal approach preference operates within the existing grade limits.
  Wider tracking bounds require observed support throughout the swept band;
  arrival and settling use consistent circular regions.
- Route search and adoption apply fresh-observation requirements to the same
  spatial near field. Compaction does not require fresh scans of distant valid
  survey geometry. Rejected handoffs now record the affected segments and ages.

The shared policy adapter, policy weights, observation layout, robot physics,
continuous costmap and Isaac Sim/LiDAR overlays are retained. The checkpoint
baseline sources match their preserved copies. The prior ascent supervisor is
selectable using `--ascent_architecture legacy --max_uphill_deg 25`.

## Evidence and limits

All runs use seed 42, the standard floor spawn, destination `(13, 0)`, the saved
full-crater development survey and the unchanged rocky-slope policy. There are
no supplied intermediate routes, manual interventions or policy switches.
Each run preserves its exact source snapshot, prior, scan-updated map, commands,
state transitions and outcomes in `outputs/ascent_behavior_tree/trialNN/`.

| Trial | Protocol | Result | Simulated time | Main finding |
| --- | --- | --- | ---: | --- |
| 01 | v1 | HOLD, destination not reached | 79.58 s | Body lean was incorrectly combined with terrain grade; weak yaw commands made little progress. |
| 02 | v2 | HOLD, destination not reached | 151.68 s | Original-bearing correction could not remove lateral offset; useful forward displacement incorrectly triggered reverse recovery. |
| 03 | v3 | HOLD, destination not reached | 156.44 s | Assumed positive yaw coast conflicted with measured policy unwind, causing repeated short pulses. |
| 04 | v4 | HOLD, destination not reached | 156.02 s | Two movement endpoints reached and settled; later alignment failed. Recovery also exposed a heading-only physical-failure rule, removed in v5. |
| 05 | v5 | Interrupted; no final mission result | 85.58 s recorded | Process exited 143 (SIGTERM, cause unknown). Two movement endpoints reached, followed by another alignment failure. The run ended before exercising the later steep continuation change. |
| 06 | v6-release, before handoff fix | Deliberately interrupted; no mission result | 36.16 s recorded | Repeated planning/adoption freshness mismatch prevented any movement primitive. Assistant stopped after reproducing the defect; no physical failure vote. |
| 07 | v6-release, shared freshness | HOLD, destination not reached | 163.46 s | Route adopted at 4 s. Second movement endpoint was physically reached during alignment, but arrival was checked only in DRIVE. Eight recovery actions followed. RELEASE was not reached. |
| 08 | v7-response | HOLD, destination not reached | 169.86 s | Three endpoints settled; uphill drive began. Repeated braking reset the continuous release timer, so RELEASE never executed. |
| 09 | v8-exposure | HOLD, destination not reached | 168.04 s | RELEASE ran 4.12 s with about 0.583 m displacement and 0.262 m body-height gain. Its actual-drift guard censored the approach; recovery later exhausted. |
| 10 | v9-response-contract | HOLD, destination not reached | 139.82 s | RELEASE ran 7.28 s, advancing 0.907 m net XY and +0.395 m body height. Backward overspeed triggered braking; the unconditional forward-drive resumption exposed missing suspended intent. |
| 11 | v10-suspended-release | HOLD, destination not reached | 147.88 s | Checked release resumed after braking; two intervals totaled 14.92 s, 1.779 m summed net XY and +0.727 m body height. A rapid-posture guard then ended the climb. |
| 12 | v11-posture-observation | HOLD, destination not reached | 151.64 s | First posture spike recovered in 0.28 s and release resumed. Later backslide ended the approach. All eight recovery requests remained in braking/observation. |
| 13 | v12-straight-admission | HOLD, destination not reached | 148.88 s | Reverse executed, including 1.62 m over about 9 s from terrain near 23°. Route selection exposed an unnecessary shelf detour; later recovery exhausted. |
| 14 | v13-route-shortcuts, before telemetry fix | Software interruption; no mission result | 3.98 s recorded | First plan diagnostics omitted logger-required fields. No movement primitive ran; reproduced and corrected as a software integration error. |
| 15 | v13-route-shortcuts | HOLD, destination not reached | 207.96 s | Shortcut removed 7.954 m detour. Circular footprint rejection and 20 unnecessary gentle-heading corrections interrupted progress. |
| 16 | v14-arrival-heading | HOLD, destination not reached | 139.48 s | RELEASE gained 0.613 m body height over 10.76 s; lateral corridor departure ended the climb. Reverse later traveled 1.377 m over 9.10 s. Source manifest dependency omission is documented. |
| 17 | v15-oriented-stopping | HOLD, destination not reached | 125.46 s | Same RELEASE climb and lateral-bound abort; body-sweep admissions changed subsequent recovery, which still exhausted eight maneuvers. |
| 18 | v16-terminal-connector | HOLD, destination not reached | 94.88 s | Planner selected a complete route to the exact exterior goal. RELEASE gained 0.893 m height over 17.92 s; a recovered posture event remained tied to braking and expired. Eight recovery maneuvers followed. |
| 19 | v17-posture-brake-ownership | HOLD, destination not reached | 102.86 s | Posture correctly transferred to braking; the original eight-second braking deadline then expired at 74.68 s. Recorded motion identified bounded uphill continuation windows that strict turn-style settling rejected. |
| 20 | v18-moving-release | HOLD after excessive tilt, destination not reached | 98.62 s | Moving RELEASE resumed at 68.92 s and gained 0.049 m height over 2.62 s. Forward correction ran only 0.16 s before a recovered posture interruption could not resume it. Shared progress expired; reverse recovery reached 1.998 m/s actual speed and 62.91° tilt. |

| 21 | v19-committed-continuation | HOLD, destination not reached | 105.04 s | Forward continuation emitted four times; only 2.26 s of deliberate forward command within the 12 s shared stall window. Eight recovery actions then exhausted. Active peak actual speed 0.782 m/s and body tilt 47.64°. |

| 22 | v20-exposure-budget | HOLD, destination not reached | 110.70 s | Main climb ended at 79.24 s by measured backslide, before either new budget expired. Exposure remained 4.88 s; no renewal, −0.372 m committed-axis progress in the final low-progress episode. Passive attitude logs show 8.66–14.72° terrain-relative tilt at the five earlier posture interruptions. |

| 23 | v21-terrain-relative-posture | HOLD after excessive tilt, destination not reached | 79.18 s | First command difference at 71.70 s; forward correction ran 2.68 s but lost height. A later two-second window measured real downhill backslide, ending ascent at 75.94 s. Reverse recovery ended at 63.67° tilt. The posture exemption did not solve ascent. |

| 24 | v22-release-patience | HOLD, destination not reached | 203.22 s | First command difference at 71.54 s. RELEASE stayed near the stalled patch until its 12 s exposure limit at 83.14 s. Reverse later traveled 2.36 m; recovery reached gentle terrain but kept correcting toward its endpoint until the episode deadline. |

| 25 | v23-functional-recovery | HOLD, destination not reached | 277.94 s | Recovery staged and settled at 163.50 s, then planned and attempted a distinct ascent at 167.52 s with exclusions intact. The new climb ended at 176.74 s by an obstacle stopping-envelope rejection; later recovery exhausted. |

| 26 | v24-measured-release | HOLD, destination not reached | 92.06 s | Productive DRIVE continued past the former timer switch at 48.46 s. No measured RELEASE admission occurred. Physical backslide ended ascent at 60.38 s; eight recovery actions exhausted. Active peak actual speed 0.560 m/s and body tilt 45.56°. |

| 27 | v25-route-exploration | HOLD, destination not reached | 94.08 s | New autonomous entrance aligned and drove 0.951 m, but passive yaw drifted 15.52° toward a known rock. Initial approach censored at 53 s before its sustained climb; one later physical recovery failure. No new climb capability was characterized. |

| 28 | v26-entrance-exploration | HOLD, destination not reached | 93.56 s | Northern entrance gained 1.32 m horizontally and 0.34 m vertically; stopping fan touched one stale cell at 57.94 s. Geometry otherwise clear. One later physical recovery failure; climb remains uncharacterized. |

| 29 | v27-checked-forward-speed | HOLD, destination not reached | 87.96 s | Cleared previous freshness stop with 0.375/0.35 m/s demand, then continued climbing. At 65.78 s exceeded 0.50 m original-path lateral bound while still gaining height. Eight recovery guards rejected drift through the resulting directional exclusion. Active peak speed 0.752 m/s, tilt 43.60°. |

| 30 | v28-directional-recovery-exit | HOLD, destination not reached | 116.00 s | Same main-climb lateral failure. Actual reverse runs of 7.66 s and 10.40 s ended in stalls before staging; exclusions retained. Active peak speed 0.611 m/s, tilt 49.55°. |

| 31 | v29-paired-launch | HOLD, destination not reached | 73.62 s | Same-class launch reduced preparatory bends3→1 and launch inclination10.88°→4.75°. Main climb gained1.09m body height but departed laterally at48.82s; body heading remained within1.6° of commitment. No RELEASE; eight recovery actions failed to stage. Active peak speed0.588m/s, tilt44.18°. |

| 32 | v30-uphill-speed-response | HOLD, destination not reached | 120.34 s | Zero response at 37.74 s decayed below 0.45 m/s next tick, then RELEASE gained 0.50 m height over 13.84 s. Ascent and productive reverse both ended at the lateral bound. Peak active speed 0.483 m/s, tilt 31.89°; eight recovery actions, no staging. |

| 33 | v31-functional-recovery | HOLD after physical tip, destination not reached | 77.28 s | Matched trial32 through 67.16 s; checked region continuation retained reverse at 67.18 s and drove another 9.48 s. Downhill speed then ran away during braking: peak 1.931 m/s, terminal tilt 64.50°. No entry-core arrival or staging. |

| 34 | v30 default-route revisit | HOLD, destination not reached | 130.44 s | Selected the earlier45° route autonomously. Matched trial26 through48.52s; zero response at48.54s sustained19.68s and gained0.827m body height. Later uphill speed surge triggered braking; backslide ended main approach77.02s. Two recovery endpoints settled, but none enabled mission completion. |

There are **zero completed ascent missions**, thirty-one completed failed trials, and
three interrupted trials. Trials 05, 06 and 14 are preserved separately and are not counted as
an autonomous physical failure, a completed mission, or qualification evidence.
Measured simulator intervals establish that the new RELEASE state can sustain
uphill movement. It does not establish full ascent reliability.

The controller evaluates arrival and displacement across movement phases, freezes
whole-leg predictions before primitive output, and accounts deliberate forward
exposure across temporary braking. It separates commanded approach grades from
physical drift support, supported endpoint departure from terrain failure, and
brake-only outcomes from executed directional traversal evidence.
It also retains suspended RELEASE intent across checked braking and tracks
traversal progress independently of the short release-response window.
Transient rapid posture change now enters a bounded observation state; it no
longer establishes a physically failed patch by itself.
Settled movement endpoints are still not whole-checkpoint or mission success.

The current deterministic regression suite passes **690 tests**. The reviewer
resumed after the earlier transient usage failure and independently passed
**203** focused tests for the v16 terminal connector and **209** for v17 posture /
braking ownership. Both implementations received **9/10**; the isolated trial
18 and trial 19 run gates each received **8.5/10**. V18 moving RELEASE received
**8.5/10** after **217** independent focused tests; one trial 20 was approved
at **8.5/10**. The new admission requires actual uphill displacement in addition
to progress along the committed axis and all existing live geometry checks.
V19 committed DRIVE continuation and its single trial-21 gate each received
**8.5/10**, following **225** independent focused tests. Approval covers bounded
simulator characterization; trial 20's uncontrolled reverse motion remains a
physical failure, despite the command ceiling and later hard-fault termination.
V20 exposure/wall budgeting and its single trial-22 gate received **8/10**, following **236** independent focused tests plus **11** final provenance checks.
V21 terrain-relative posture entry and the single trial-23 gate received **8/10**, following **246** independent focused tests.
V22 RELEASE patience and the single trial-24 gate received **8/10**, following **253** independent focused tests. Trial 24 retained all limits and exclusions but did not complete ascent.
V23 functional recovery staging and its single trial-25 gate received **8.5/10**, following **267** independent focused tests. Trial 25 demonstrated a measured staging handoff and a new autonomous ascent attempt; it did not finish the mission. A staged recovery is never counted as ascent success.
V24 measured DRIVE-to-RELEASE selection received **8.5/10** for implementation and **8/10** for one trial 26, after **275** independent focused tests. Trial 26 ended HOLD at 92.06 s; no RELEASE admission or ascent success.
Exact gate scopes and source hashes are in `ascent_review_gates.json`.
This is not human certification.
See the [review record](ascent_behavior_tree_review.md).

The [straight geometry evidence](straight_geometry_revision.md) records the
exact 62-second diagnostic replay, v14 arrival-heading change and its physical
result. V15 integrates continuously swept body geometry in straight stopping
checks. Trial 17 exercises the fallback but does not complete ascent. The exact
pre-departure capture and reviewed next-step rationale are in
[departure_capture.md](departure_capture.md). Successful physical characterization
and probability calibration remain outstanding.

The [command audit](ascent_behavior_audit.json) checks emitted commands against
terrain, body posture, simultaneous drive/yaw, speed ceilings and duplicate
outcomes. Its sampled committed-drive backslide metric excludes velocity damping;
the older `backslide_m` summary counter includes motion opposed to braking commands.
The [trial summary](ascent_behavior_summary.json) reports outcomes, recovery counts,
tilt, speed by terrain and runtime latency. Command invariants do not establish
physical controllability or ascent reliability.

All thirty-one completed trials pass the command audit: no autonomous yaw above 15°
footprint inclination, no combined translation/yaw command, no command-ceiling
violation, no duplicate terminal outcomes, and no false destination completion.
Trial33 exposes a missing terminal movement event when hard fault takes priority;
its separate approach outcome records the physical tip exactly once. A uniqueness
check does not establish complete terminal-event logging.
Completed-run active peak measured speed ranged from 0.453 to 1.998 m/s despite
the 0.40 m/s command ceiling. Controller p95 latency reached 15.35 ms in trial33; these
figures exclude mapping, route-search and rendering costs. This is not a claim
that physical speed is always at or below the command limit.

The bounded heading search can miss some feasible corridors. The offline
full-survey geometry check found a complete route, but that is not physical
traversal evidence. No ascent gate or calibration claim follows from it.

## Reproduction

The [ascent launch instructions](../ascent_navigation.md) open Isaac Sim and the
dashboard together, with costmap, LiDAR, candidate-route and checkpoint overlays.
Use a new log filename for every run. Add `--headless --no_dashboard` for the
same mission without rendering.

Three consecutive qualified floor-to-exterior successes are still required
before advancing to descent. Qualification additionally requires a frozen,
held-out calibrated reachability model matching the policy and execution source.

The optional v25 initial-corridor coverage selector is implemented and passes the regression suite. The reviewer rated implementation **8.5/10** after **287** independent focused tests and approved one trial 27 at **8/10**; it ended HOLD at 94.08 s without reaching the sustained climb. Frozen-survey diagnostic results are recorded in `outputs/ascent_route_diagnostics/`; neither they nor a selected new route establish physical ascent success.

V26 separates physically attempted, geometry-censored entrances from completed sustained-climb characterization. Both remain experiment scheduling metadata. The reviewer rated implementation **8.5/10** after independently passing all **588** tests and approved one trial 28 at **8/10**. It ended HOLD at 93.56 s with unchanged movement controls. The captured stopping fan failed freshness alone at cell (1.1, 4.9), last observed at 0.4 s; it is not an obstacle or grade failure and is not imported as entrance-failure coverage. Active peak measured speed was 0.469 m/s and tilt 26.35°. No ascent completion.

V27 bounded forward-speed selection received **8.5/10** implementation review after **603** independent tests. One trial 29 was approved at **8/10**. Trial29 passed the targeted freshness stop, then ended HOLD at 87.96 s after a lateral-tracking failure and eight censored recovery actions. The full mission remains incomplete.

V28 directional occupied-exit received **8.5/10** implementation review after **614** independent tests. One trial30 was approved at **8/10** and ended HOLD at 116 s. It performed actual reverse movement but did not reach staging or complete ascent. It preserves the lateral tracking boundary and failed-patch evidence while permitting only checked incidental outward motion during reverse recovery.

V29 paired-launch implementation received **8.5/10** after **627** independent tests, and one trial31 received **8/10**. It confirmed the comparison was selected and executed, but ascent still failed. Six measured-speed braking episodes interrupted the main powered-response windows. Five were initially uphill, one downhill; this observation does not establish that removing braking or using zero command would improve traction. Further changes and runs require review.

V30 is opt-in characterization only. Implementation received **8.5/10** after **641** independent tests; one trial32 received **8/10**. The run matched trial31 through 37.72 s and isolated the intended zero-command response. It reduced peak active speed and tilt in this paired run, but neither ascent nor recovery reached a settled checkpoint. This does not establish probability calibration or reliable traversal.

V31 received **8/10** for implementation and **8/10** for one trial33 after **679** independent tests. The fixed recovery region passed geometric admission, but reverse controllability failed before arrival. The original 51.58 s exclusion remains identical; the old point attempt was censored once and the separate region prediction remained null/unvalidated. The region's tip was counted once, without assigning a directional exclusion to the distant final tip position. No further V31 run is authorized by that consumed gate. Admission took 46.07 ms; the pre-core continuation bucket (including safety interruptions) had p95 15.63 ms, maximum 19.10 ms. Map updates averaged 57.74 ms. These measurements do not meet the intended lightweight goal; traversal remains the current gate.

The unchanged-source trial34 revisit received **8/10**. It applies V30 with the existing V27/V28 behavior to the earlier default-cost route; it is not a new corridor or an isolated comparison of V30 alone. Active peak measured speed was0.861m/s, tilt48.83°, command-opposed backslide metric2.196m, controller p952.311ms and map-update mean46.75ms. The 68.22s speed event was uphill (gradient-projected velocity+0.510m/s), and the later backslide was real. A passive diagnostic extension received9/10 proposal approval; code review and a new physics gate remain separate.
