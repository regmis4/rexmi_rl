# Teleop-informed ascent release and measured response, protocols v6–v13

The second demonstration completed another exterior–floor–exterior traversal
with the same rocky-slope policy and ±0.40 m/s translation commands. It strongly
supports treating zero desired velocity as an active motion primitive. It does
not establish autonomous mission completion or calibrated reachability.

## Evidence

Capture: `logs/telemetry/crater-demo-repeat-20260923-225750/`.
Analysis: `outputs/teleop_analysis/crater-demo-repeat-20260923-225750/`.
The 35,536 physics samples form 8,884 complete, gap-free control steps over
177.68 simulated seconds. Commands and poses agree with the separate navigation
CSV. There are no reset events; no clean-close marker was recorded. Source
hashes are retained in `summary.json`. Raw captures are unchanged.

| Interval (simulated seconds) | Input | Net XY displacement | Body height change |
|---|---|---:|---:|
| 92.84–104.10 | Forward, no commanded yaw | 3.353 m | +1.655 m |
| 130.08–135.92 | Zero translation and yaw | 0.114 m | −0.077 m |
| 139.16–155.80 | Zero translation and yaw | 1.713 m | +1.143 m |
| 140.16–155.80 | Same release, excluding its first second | 1.533 m | +0.993 m |
| 157.84–162.80 | Zero translation and yaw | 0.515 m | +0.079 m |

During the long productive release, mean absolute wheel targets remain about
6.646 rad/s: the policy is operating, not simply coasting. Progress fades in
the final seconds. The unsuccessful release reaches 59.53° body tilt and
45.64° absolute roll; useful release does not justify removing posture guards.

Supporting geometry from the labeled development survey puts the long useful
release at approximately +30° from steepest ascent on entry, with a median
offset of +12°. The unsuccessful release enters at approximately −30°. The
first demonstration's useful release enters near +1°. These are survey-based
context, not terrain measurements from the teleop logger. They show why angle
alone cannot determine success. The best direction also depends on rocks,
support, posture and the complete approach.

The two captures share policy and resolved configuration but have different
measured total masses (19.882 versus 18.695 kg), caused by base-mass
randomization. They are demonstrations, not matched-dynamics causal trials.
Wheel speeds and normal-force contacts do not measure frictional slip.
The northern exit also differs from the autonomous destination `(13, 0)`.

## Implemented response

The behavior tree retains exclusive movement ownership. Forward ascent can
transition to policy-active RELEASE and resume the same forward primitive
without reverse braking or a hillside alignment maneuver. The request,
deadline, committed axis, evidence and poor-progress episode survive that
transition. Productive release continues; declining progress triggers checked
resumption. Both prospective powered creep and actual drift need observed room.

The planner softly prefers a modest diagonal, with no new hillside turns.
Checked wider corridors allow natural drift, and circular arrival regions
replace inconsistent along/cross admission. All slope and speed limits remain.
Initial schedule settings are documented in `docs/ascent_behavior_tree.md`;
their optimality is unvalidated and the protocol invalidates old calibration.

## Independent control review and verification

An independent AI reviewer acting in the requested senior control engineering
role analyzed both captures and reviewed the implementation. This is not human
certification. Review identified and corrected:

- Forward resumption before verifying that heading still advances the corridor.
- Mismatched arrival/settling regions that could create a false physical failure.
- A gradient change retroactively shortening the shared stall budget.
- Omission of zero-command backslide from the reported metric.

Focused tests cover these cases, zero-command guards, sustained progress,
threshold jitter, reverse/descent exclusion, STOP, stale scans, overspeed,
posture, deadline, duplicate outcomes, checked width and directional limits.
Small source-hashed observation fixtures reproduce three measured releases.
The first useful release remains active throughout; the long repeat release
resumes forward at 15.82 seconds as progress fades; the bad release resumes at
0.52 seconds because it has already lost more than 8 cm along the committed axis.
These are open-loop observation replays with clear-corridor stubs, not a claim
that changed commands would have produced the recorded motion.

The reviewer independently passed 75 focused tests and cleared bounded physics
characterization after rechecking the fixes. Ascent qualification remains unpassed; descent and
optimization remain gated.

## Physics trial 06: planning handoff diagnosis

The standard seed-42 full-survey trial stayed in OBSERVE and never executed a
movement primitive. It was deliberately stopped at 36.16 simulated seconds;
`outputs/ascent_behavior_tree/trial06/interruption.json` records the interruption
separately from physical failures and complete mission trials.

Independent reconstruction showed that full geometry and staging were valid.
The search required freshness on nearby short edges, but adoption applied that
requirement to entire compacted segments extending beyond the 2 m near field.
This repeatedly rejected retained survey geometry and resubmitted the plan.
The defect also reproduced on flat terrain with a straight 4 m route.

Search and adoption now share an analytic circle-clipping freshness check.
Full-route support, grade, exclusions and geometry are retained, as are fresh
local execution checks. Regressions distinguish valid distant survey geometry
from stale nearby cells and distant hazards. Adoption rejections now record
their segment and observation ages rather than disappearing behind OBSERVE.

## Physics trial 07: arrival during alignment

With the shared freshness contract, the rerun adopted its route at 4.0 s and
settled at its first movement endpoint at 23.46 s. The second action entered
its 0.30 m arrival region at 58.90 s while braking, but continued commanding yaw
because arrival was checked only in DRIVE. At 65 s it was just 0.085 m from its
target. It eventually censored alignment and exhausted eight recovery actions,
ending in HOLD at 163.46 s. It never reached RELEASE or completed ascent.

The command audit passes. Peak commanded speed was 0.400 m/s, peak measured
active speed 0.470 m/s, controller p95 latency 0.556 ms, and average map update
46.86 ms. One physical approach failure was recorded during recovery. The full
source, map, events, commands and summary remain in `trial07`.

Protocol v7 evaluates arrival and displacement bounds before state-specific
BRAKE/ALIGN/DRIVE/RELEASE commands. Arrival still requires supported ground and
measured settling. It also freezes the complete-leg prediction at initial
request admission: trial07's first yaw command began at 4.38 s, before the old
prediction at 19.44 s, after about 0.22 m of translation. Later primitives retain
the original prediction and do not receive independent outcome votes.

## Physics trial 08: drive-time accounting

V7 settled at three primitive endpoints by 64.48 s and drove onto the uphill
segment. Its fourth action accumulated 7.62 seconds of deliberate forward
command in several bursts, but the longest burst was 3.54 seconds. Overspeed
braking reset the continuous five-second release timer each time, so RELEASE
never executed. The robot later left the tracking corridor while braking;
that was censored rather than marked as bad terrain. Recovery exhausted at
169.86 s, without reaching the exterior destination.

The command audit passes. Peak active body tilt was 33.98°, measured committed
drive backslide was approximately 0.010 m, and controller p95 latency was
0.546 ms. These are partial-traversal measurements, not ascent qualification.

V8 counts cumulative deliberately emitted forward-command exposure within the
same primitive, excluding braking even when damping emits positive velocity.
Braking pauses preserve exposure; release, actual alignment and a new request
reset it. The five-second threshold remains an unvalidated characterization
setting. This changes scheduling without changing speed or overspeed protection.

## Physics trial 09: release exercised

The cumulative exposure clock triggered RELEASE at 73.94 s. It emitted zero
translation and yaw for 4.12 seconds. Over the interval, the robot advanced
approximately 0.583 m horizontally and gained 0.262 m in body height, using
the same active policy. At 78.06 s the actual-drift corridor guard censored the
approach. This is a geometry rejection, not a physical failure vote.

The later recovery completed two supported primitive endpoints but then
encountered posture, settling and staging failures. The run ended in HOLD at
168.04 s with eight recovery actions and four physical approach failures during
recovery. The exterior destination was not reached. Peak active body tilt was
46.36°; command invariants pass, and controller p95 was 0.581 ms.

This trial supports the release primitive's ability to retain uphill motion.
It does not establish complete ascent or calibrated probability. Sources,
maps, outcomes and the failed mission remain preserved in `trial09`.

![Autonomous release response](../../outputs/ascent_behavior_tree/trial09/release_response.png)

## Review corrections after trial 09

The reviewer reconstructed the final RELEASE drift vector as approximately
`(+0.05676, -0.03912) m/s`. On the retained final geometry, its stopping envelope
passes physical support but fails the 25° planning cross-slope limit at 25.74°.
This is corroborating final-map evidence; the contemporaneous map was not frozen.
V9 checks incidental drift against physical support, observations, obstacles and
exclusions, while preserving directional grades for the prospective command.
It logs the actual velocity, checked endpoint and rejection category.

Recovery did reverse continuously for 22.58 seconds, moving about 2.10 m and
descending 0.585 m to gentle terrain. At 117.88 s, an endpoint distance of
0.300789 m exceeded the 0.30 m arrival radius and was incorrectly treated as
physical terrain failure. V9 distinguishes supported repositioning from lost
support and keeps the mission destination outstanding.

Later requests never emitted DRIVE, yet their planned directions supplied
failure votes. V9 separately logs physical attempt outcomes and actually
executed traversal evidence. Only held DRIVE/RELEASE commands count; damping
does not. Exclusions use the locally executed direction and the recorded failure
position/time. New requests and alignment clear traversal evidence. A brake-only
failure still records a failed attempt, but cannot close a new approach direction.

The reviewer independently passed 101 focused tests; the full suite passed
392 tests before the next simulator characterization. These checks do not
establish mission completion or probability calibration.

## Physics trial 10: release followed by overspeed braking

V9 retained RELEASE from 73.94–81.22 s: 7.28 s, 0.907 m net XY displacement
and +0.395 m body-height change, measured through the interrupting observation.
The previously rejecting incidental cross-slope check no longer interrupted it.
At 81.22 s, actual velocity was about `(-0.4494, -0.2130) m/s`, backward along
the ascent. Braking was appropriate. After 5.06 s of braking, the executor
unconditionally resumed DRIVE, despite useful release progress before interruption.
A rapid posture change ended that climb at 89.34 s. The trace cannot establish
whether continued release would have avoided it.

The run ended in HOLD at 139.82 s after eight recovery actions, six physical
attempt failures, and no exterior completion. Brake-only failures correctly
withheld directional exclusions. Peak active tilt was 42.80°, actual speed
peaked at 0.534 m/s, and controller p95 was 0.598 ms. The emitted-command audit
passes; actual speed is not guaranteed to equal its 0.40 m/s command ceiling.

V10 explicitly suspends RELEASE across overspeed braking and restores it only
after measured settling and renewed heading/geometry/uphill checks. It has
separate traversal-progress and short policy-response observers so repeatedly
interrupted release cannot renew a stall allowance indefinitely.
The reviewer independently passed 110 focused tests after two further fixes:
consistent world-displacement projection when a gentle resume refines heading,
and supported arrival taking precedence over low-progress expiry. All 401 tests
passed before trial 11. Shared traversal progress and remaining allowance are
now logged even during BRAKE.

![Trial 10 release response](../../outputs/ascent_behavior_tree/trial10/release_response.png)

## Physics trial 11: release restoration and posture evidence

The interrupted release restored at 86.28 s after measured braking and checks.
It continued another 7.64 s, with 0.872 m net XY displacement and +0.332 m body
height. Both release intervals totaled 14.92 s, 1.779 m summed net displacement
and +0.727 m body-height change. This is useful partial ascent, not completion.

At 93.92 s the rapid-posture rule ended RELEASE at 41.17° body tilt, despite
positive 0.067 m/s two-second committed-axis progress. Recovery exhausted at
147.88 s: eight recovery actions, nine physical attempt outcomes under v10,
but only one directional exclusion because later requests had no executed
traversal evidence. Peak active tilt was 43.34°, speed 0.534 m/s, and controller
p95 was 0.625 ms. Command invariants passed.

The reviewer applied the same posture rule to manual data. It would abort the
successful repeat zero-command climb at 148.52 s: a 37.96° control-sampled tilt
spike recovered below 36.87° by 148.56 s with zero command unchanged, then gained
approximately 0.206 m body height over two seconds. Six successful ascent events
met a continuous 0.20 s below-threshold recovery dwell in 0.22–0.36 s. A more
severe event took 1.34 s, including a human command change; it is not a controlled
counterfactual for an earlier autonomous pause. Disappearance of the rapid-rise
signal alone is insufficient: that happened while tilt still exceeded 52°.

V11 adds POSTURE_PAUSE with a provisional 1.5 s observation bound and 0.20 s
absolute-tilt recovery dwell. Overspeed braking, hard tip and geometry guards
remain binding. Stable/improving unresolved posture is censored; continuing
measured deterioration can still establish physical failure. Timing alone
does not establish bad terrain. The reviewer caught and corrected skipped
overspeed observations in the dwell, brake-settle bypass, late restoration
after the deadline, and route-admissibility being mistaken for physical support
loss. All 415 tests passed, including 124 independently reviewed focused tests.
The manual fixture is an open-loop observation replay, not a physical-safety
or counterfactual dynamics test.

![Trial 11 release response](../../outputs/ascent_behavior_tree/trial11/release_response.png)

## Physics trial 12: posture response and reverse admission

POSTURE_PAUSE recovered the first rapid tilt event at 94.20 s after 0.28 s of
policy-active zero input. RELEASE continued. A second event at 94.68 s was
followed by actual committed-axis backslide at 95.14 s, ending the approach.
Across RELEASE intervals, the robot gained 0.721 m in body height with 1.844 m
summed net XY displacement over 15.40 s. This excludes posture-pause intervals.

All eight recovery requests remained in braking or observation. The mission
ended in HOLD at 151.64 s, with six physical attempt outcomes and three censored
outcomes; only the executed uphill failure created a directional exclusion.
Peak active tilt was 47.85°, actual speed 0.534 m/s, controller p95 0.627 ms,
and average mapping time 47.12 ms. The command audit passes. Ascent did not finish.

The independent reviewer compared reverse entry in both manual recordings with
the autonomous admission rule. The three manual entries had only 0.675–1.031°
heading excursion over the preceding 0.30 s, despite yaw-rate or speed spikes
that failed the stationary admission gate. Later manual reverse included yaw,
so those demonstrations do not prove sustained autonomous straight reverse.

V12 distinguishes aligned straight takeover from stopped turning. It permits
only a bounded speed, opposing translation and complete heading excursion
window, with checked forecast stopping room. It retains the turn gate, zero
commanded yaw, capability limits and existing corridor checks. These provisional
settings require physical characterization and do not establish calibration.

![Trial 12 release response](../../outputs/ascent_behavior_tree/trial12/release_response.png)

## Physics trial 13: executed reverse and shelf detour

The straight takeover contract executed reverse at 60.88 s and reached a
settled recovery endpoint at 70.50 s (0.594 m net displacement on gentle ground).
Later RELEASE ran 95.92–97.80 s, with 0.180 m XY displacement and +0.025 m body
height. Its prospective footprint guard censored that approach. Reverse then
ran for about nine seconds, moving 1.618 m and descending 0.171 m from terrain
near 23° before a physical lateral-corridor departure. Several subsequent short
recovery endpoints were reached, but the mission ended in HOLD at 148.88 s.

Peak active tilt was 36.74°, measured speed reached 0.721 m/s, and controller
p95 was 0.609 ms. Mapping averaged 53.13 ms. Commands remained within 0.40 m/s;
the measured overshoot demonstrates that the command ceiling is not a physical
speed guarantee. Two physical approach failures were recorded. The command and
directional-evidence audits pass. No ascent mission completed.

An earlier replan at 38.68 s exposed an unnecessary shelf detour. Direct search
exhausted its 60,000-node budget and produced a frontier whose route later failed
execution geometry. The only eligible shelf alternative went 4.17 m backward,
then 4.4 m north, passing near the starting pose. The independent reviewer found
a 0.615 m connection to its onward staging point, removing 7.954 m. Connector,
suffix and staging pass on the retained final geometry. Later failure evidence
changes retreat validity, so this is not a reconstruction of the exact earlier
planning state or an autonomous success claim.

V13 adds bounded, checked shortcut candidates before admitting any checkpoint.
It retains the original and the same terminal/frontier status, then recomputes
terrain cost, whole-leg prediction and retreat. It does not impose a recorded
teleop route or change an active checkpoint. Full candidate routes and precise
rejected segments are now logged separately from checkpoint prefixes.

![Trial 13 release response](../../outputs/ascent_behavior_tree/trial13/release_response.png)

## Trial 14: software interruption before motion

The first plan selected an approach at 4.0 s, but no primitive ran. The last
CSV observation is 3.98 s. New rejected-shortcut diagnostics omitted fields
required by the logger (`length_m`/`max_uphill_deg` and `uphill_limit`). Both
missing-field failures reproduce outside Isaac; cleanup hid the original
traceback, so its exact first missing field was not captured. The process exited
zero without a mission summary. This is a preserved software interruption,
not a physical failure or completed mission.

The producer now supplies the established diagnostic schema. A regression
consumes the actual logger signature with rejected shortcuts and serializes
all candidate records. Future software exceptions are printed and saved to a
per-run `.failure.json` before simulator shutdown, independently of exit status.

## Trial 15 and straight-motion follow-up

The shortcut removed the unnecessary detour, but the mission ended in HOLD at
207.96 s. No RELEASE interval occurred. The contemporaneous diagnostic captured
a clear configured rectangular sweep rejected by the circular obstacle margin,
and the trace showed twenty heading corrections despite a straight direction
that already reached the arrival region. See the [evidence and v14 change](straight_geometry_revision.md).
These findings do not establish a completed ascent; descent remains gated.

## Reproduce the analysis

```bash
/home/susan/miniconda3/envs/env_isaacsim/bin/python scripts/analyze_teleop_navigation.py \
  logs/telemetry/crater-demo-repeat-20260923-225750 \
  --output outputs/teleop_analysis/crater-demo-repeat-20260923-225750
/home/susan/miniconda3/envs/env_isaacsim/bin/python -m unittest discover -s tests -p test_uphill_release.py
```
