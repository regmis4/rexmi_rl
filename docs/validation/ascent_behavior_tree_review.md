# Independent control review: staged teleop ascent

This revision was reviewed by a separate AI agent assigned the senior control
software reviewer role. It is an independent code/design review, not a human
engineering certification or physical mission qualification.

## Review iterations

1. **Architecture review.** Require a reactive action-selection tree and one
   exclusive movement executor; separate planning and motion ownership; retain
   STOP priority and directional evidence. A 33° uphill capability must not
   imply that a hillside pivot or the same cross-slope angle is controllable.
2. **Implementation review: changes requested.** The reviewer reproduced a
   continuous steep-start/lattice mismatch, stale turn-staging adoption, a
   near-goal completion gap and a tiny-segment action crash. They also identified
   incomplete braking authorization, missing preventive posture/backslide
   monitoring and the need to enforce execution-protocol calibration provenance.
3. **Re-review: cleared for bounded simulator characterization.** The seven
   change groups were addressed. The reviewer independently ran all 33 new
   deterministic tests; all passed. The clearance was explicitly limited to
   simulator characterization, not reliability or calibrated reachability.
4. **Empirical protocol review after trial 01.** The first run showed body lean
   triggering the combined 15° gate on gentle ground and negligible yaw response
   to 0.10–0.12 rad/s proportional commands. Protocol v2 keeps the 15° footprint
   terrain lock and treats body posture separately (25° pause / 20° resume).
   It uses full ±0.35 rad/s teleop pulses, each at most two seconds, with latched
   sign and measured yaw settling before an opposite pulse or drive. The reviewer
   independently ran the expanded 41-test suite and cleared bounded trial 02.
5. **Closed-loop feedback review after trial 02.** A useful displacement missed
   its endpoint laterally, then incorrectly entered reverse recovery. Protocol
   v3 aligns from the actual pose, checks the full remaining corridor, latches
   the new heading across steep entry, and measures progress along that heading.
   Overshoot is a typed, censored repositioning outcome; fresh supported gentle
   ground permits observation and replanning without a fabricated success or
   failure vote. The reviewer also reproduced an old angular-progress timer
   cancelling a new correction after a long drive. The timer now resets only
   when driving transitions into a new braking/correction episode, preserving
   deadlines and progress history across pulses and posture pauses. All **48**
   movement/planning integration tests passed independently. The reviewer found
   no remaining blocker in these deltas and cleared bounded trial 03 only.
6. **Policy-response review after trial 03.** The reviewer measured yaw unwind
   in 21 of 22 pulse/coast cycles. The old rule predicted a median +0.055 rad
   coast, while measured median coast was −0.0096 rad; the largest residual
   increase was 0.071 rad. Protocol v4 removes that unsupported coast prediction
   and stops pulses at signed measured residual ≤0.02 rad. It retains the
   0.10 rad settled drive tolerance, two-second pulse bound, and deadlines.
   A deterministic response fixture varies yaw rebound and lag across nine
   cases in the observed rebound range. Larger uncharacterized rebound cases
   explicitly test bounded censoring, not a false claim of convergence.
   The reviewer independently passed all **50** focused tests and compared both
   laws on the same fixture: v4 entered DRIVE at residuals 0.049, 0.069 and
   0.088 rad, while the earlier rule timed out. They cleared bounded trial 04;
   this does not establish physical convergence or mission completion.
7. **Commitment and resumption review.** A 0.16 rad body-heading deviation
   alone did not establish physical traversal failure. Protocol v5 continues
   zero-yaw steep travel while the checked geometry, original travel-axis
   progress, positive command projection, speed, posture and displacement
   constraints hold. It logs heading variation without a failure vote. The
   reviewer then reproduced a braking/resumption gap: temporary overspeed
   incorrectly sent the already committed action through initial alignment
   admission. Resumption now rechecks actual remaining/stopping geometry and
   retains the original heading, progress axis, request and deadline. Initial
   admission still requires alignment. Dedicated forward/reverse continuation,
   brake/resume and geometry-change tests cover these contracts.
   The reviewer independently passed all **55** focused tests, found no remaining
   blocker in this delta, and cleared bounded trial 05 characterization.
8. **Teleop-informed release review.** Two manual captures established that
   zero command can retain substantial uphill motion with the policy active.
   The reviewer required prospective movement guards at low speed, shared
   attempt/progress accounting, response-driven resumption and unchanged
   posture/STOP/speed protections. Review of v6 found three reproducible
   transition defects: opposing-heading resumption, inconsistent arrival and
   settling regions, and retroactively shortened patience after a gradient
   change. All were corrected with regressions, along with omitted release
   backslide accounting. Gentle-terrain correction brakes before alignment.
   The reviewer independently passed **75 focused tests** and cleared bounded
   trial 06. See [the measured evidence and limitations](teleop_release_revision.md).
9. **Spatial freshness contract after trial 06.** The diagnostic trial never
   entered a movement primitive: adoption incorrectly required fresh scans over
   entire compacted segments, including distant survey geometry. The reviewer
   reproduced this on both the frozen crater map and an independent flat map.
   Shared circle clipping now checks freshness within the same 2 m spatial
   region during search and adoption, with full geometry/support/grade and
   exclusions everywhere. Five previously rejected crater routes pass the new
   contract and still reproduce the old failure. The reviewer independently
   passed **79 focused tests** and cleared bounded trial 07. Trial 06 remains
   recorded as an assistant-requested diagnostic interruption, not a physical
   failure or completed mission.
10. **Measured arrival across phases after trial 07.** The frozen trace showed
    the second action inside its arrival region at 58.90 s while BRAKE/ALIGN
    continued and later censored the action. Arrival and displacement bounds
    now precede state-specific commands, retaining supported measured SETTLE
    before success. The first complete-leg prediction is frozen at request
    admission because yaw and braking can translate the policy; later primitives
    retain the same prediction and evidence unit. The reviewer independently
    passed **84 focused tests**, exercised continued rotation during settling,
    and cleared bounded trial 08 under `staged-teleop-v7-response`.
11. **Forward-exposure accounting after trial 08.** The fourth action emitted
    7.62 seconds of deliberate forward drive, but overspeed interruptions kept
    every continuous burst below 3.54 seconds. V8 credits elapsed time to the
    previously emitted forward intent across braking pauses. Positive damping
    and BRAKE-to-DRIVE handoff outputs are excluded; release, actual alignment
    and new requests reset exposure. The reviewer independently passed **87
    focused tests** and cleared bounded trial 09. A release immediately
    interrupted by braking remains inconclusive; suspended-release restoration
    is a separate state-ownership question, not permission to raise speed limits.
12. **Physical response and evidence after trial 09.** The reviewer found a
    planning cross-slope limit applied to incidental RELEASE sway, a supported
    0.789 mm endpoint-radius departure classified as terrain failure, and
    brake-only requests supplying distinct traversal directions. V9 separates
    the drift envelope from commanded grades, censors supported repositioning,
    and admits exclusions only from locally executed DRIVE/RELEASE evidence.
    Actual output intervals, committed travel headings and failure position/time
    are logged; braking alone contributes no direction. The reviewer tested
    corner exclusions and reconstructed trial09's drift on its final map,
    independently passed **101 focused tests**, and cleared bounded trial 10.
    The full suite passed **392 tests**. Final-map reconstruction is explicitly
    not proof of the contemporaneous map state.
13. **Suspended release after trial 10.** Productive RELEASE was interrupted by
    actual backward overspeed, then unconditionally replaced by DRIVE after
    braking. V10 preserves suspended intent and restores exact zero only after
    measured settling and renewed geometry/heading/uphill checks. The reviewer
    required an independent progress observer through braking, so short repeated
    interruptions cannot prevent a stall clock from starting. Review found and
    corrected an absolute-projection basis change that fabricated backslide,
    and a low-progress decision that incorrectly preempted safe arrival. The
    observer now projects world-space displacements consistently and admits
    supported settling before the deferred stall decision. The reviewer
    independently passed **110 focused tests** and cleared bounded trial 11;
    all **401 full-suite tests** passed. Dedicated telemetry records the shared
    traversal window/progress/remaining budget separately from release response.
14. **Posture observation after trial 11.** The reviewer reproduced the rapid
    posture abort on a successful manual zero-command climb, where the spike
    recovered within 40 ms and climbing continued. V11 adds a provisional
    1.5 s POSTURE_PAUSE, requiring absolute tilt recovery for 0.20 s and any
    outstanding measured braking dwell. Stable/improving timeout is censored.
    Review found and corrected recovery evidence skipped on overspeed samples,
    restoration without brake settling, restoration after the episode deadline,
    and map/exclusion inadmissibility being labeled physical support failure.
    The same attribution correction applies to SETTLE. The reviewer independently
    passed **124 focused tests** and cleared bounded trial 12. All **415 tests**
    passed; no ascent or probability qualification follows from unit/replay tests.

15. **Straight takeover after trial 12.** Manual reverse entries had small heading
    excursion but failed the old instantaneous yaw-rate settling criterion.
    V12 separates already-aligned straight admission from stationary turning.
    Review reproduced two blockers: net yaw trend omitted zero-net oscillation
    from the stopping fan, and a denied fan later became a physical brake failure.
    The fan now includes observed heading extrema and projected trend; geometry
    denial immediately censors even with prior executed traversal evidence.
    Sample gaps over 0.04 s invalidate the alternate admission window.
    The independent reviewer passed **136 focused tests** and cleared bounded
    trial 13; all **427 full-suite tests** passed before launch. The settings
    remain provisional pending physical characterization.

16. **Shelf detours after trial 13.** The reviewer found an onward-connected
    shelf route that unnecessarily backtracked through the floor. Retained-map
    geometry admits a 0.615 m connector removing 7.954 m of detour; later
    exclusions prevent treating this as exact planning-time replay. V13 adds
    bounded checked shortcut alternatives before checkpoint admission while
    preserving originals, terminal destinations and complete/frontier flags.
    Full-route geometry and heading staging are checked; leg predictions and
    retreat are recomputed. Different suffixes remain distinct, and full
    candidate paths plus precise rejected segments are logged. The independent
    reviewer passed **146 focused tests** and cleared bounded trial 14;
    all **437 full-suite tests** passed before launch.

17. **Diagnostic integration after trial 14.** New shortcut records lacked
    fields read by the runner's telemetry signature, stopping the run at its
    first plan. Both missing-field failures were reproduced; the original
    traceback was masked by simulator cleanup. Producer fields now match the
    existing schema, with an exact consumer/JSON regression. Run-body exceptions
    are printed and saved before cleanup, and reporting preserves these as
    incomplete software failures. Cleanup-only errors are outside that handler.
    The reviewer passed **148 focused tests** and cleared bounded trial 15;
    all **439 full-suite tests** passed before launch.

18. **Trial 15 geometry diagnosis; review interrupted.** The reviewer identified
    a circular-footprint rejection whose configured rectangular poses were clear
    on a retained map. Local instrumentation subsequently captured the exact
    rejection and reproduced the first 3,100 poses/commands without differences.
    It also identified 20 corrections whose current straight rays already
    reached the existing arrival region. The reviewer then hit a usage limit.
    The diagnostic logging, offline sweep reference and v14 arrival-heading
    change have not received independent review. Local full-suite regression
    caught early yaw-pulse stopping interacting with unwind; pulse stopping was
    restored before characterization. See [evidence and scope](straight_geometry_revision.md).

19. **Local verification while independent review is unavailable.** V14's
    physical trial 16 sustained 10.76 s of RELEASE (+0.613 m body height), then
    crossed the original lateral corridor bound. Later reverse traveled 1.377 m
    over 9.10 s; the mission still failed. V15 applies a continuous configured
    rectangular obstacle sweep plus passive-yaw stopping fan consistently to
    command, drift, braking and RELEASE checks. Global planning, ground support,
    turn staging and the lateral bound remain unchanged. All **485 local tests**
    pass. This iteration is not independently cleared: the review package is
    queued with the existing reviewer, whose usage limit has not been reset.
    Trial 16's omitted geometry dependency is documented explicitly; future
    source archives include it. Review must examine forecast/interpolation
    conservativeness, directional exclusions and the distinct question of
    safe continuation after useful uphill displacement leaves a planned corridor.

20. **Reviewer resumed; explicit score gate.** The original reviewer resumed
    normally after the transient usage failure. An immediate changed-destination
    continuation was rejected at **6/10**. An observational departure capture
    was approved at **9/10**, including independent verification of 197 focused
    tests. The run reproduced all 3,046 trial-17 pose/command samples through
    60.90 s and preserved exact pre-exclusion terrain. It received no mission
    or qualification credit.
21. **Terminal connector.** A generic continuous final straight segment from
    checked gentle staging removes a spurious terminal sub-cell turn. The
    exact destination, 33° uphill, 25° cross-slope, 35° footprint, exclusions,
    and original observer-relative freshness checks remain. The reviewer rated
    the implementation **9/10** after independently passing 203 focused tests.
    The full suite passed 498. One v16 ascent characterization run, trial 18,
    was authorized at **8.5/10**, with the movement executor unchanged.
22. **Response experiment remains separate.** A proposed 0.8 m/5 s probe was
    rejected at **7/10** because stopping room must fit inside fresh terrain.
    The reviewer approved implementing a narrower **2 s or 0.25 m** forward
    response probe at **8.5/10**, subject to fresh swept stopping/drift checks,
    inherited budgets, all protections, and a subsequent code/run gate. It
    is not part of trial 18. Full adaptive handoff remains rejected at 6/10.

23. **Posture and braking ownership.** Trial 18's posture recovered at 66.58 s
    and satisfied the existing 0.20 s dwell by 66.78 s. Braking remained
    unresolved; the old combined timeout misreported unresolved posture at
    67.88 s. V17 transfers recovered posture to BRAKE without direct motion
    resumption, retaining the original 66.66 s brake anchor, suspended intent,
    request deadline and shared uphill progress budget. Repeated posture
    interruptions cannot renew braking's eight-second bound. The reviewer
    rated the implementation **9/10**, independently passed **209** focused
    tests, verified the full **504-test** suite, and approved one trial 19 at
    **8.5/10**. Strict RELEASE settling remains unchanged and unproven.

24. **Measured moving RELEASE admission.** Trial 19 matched 3,394 trial-18
    pose/command samples through 67.86 s. It correctly retained BRAKE after
    posture recovery, but strict stationary settling never completed before
    the original eight-second brake deadline. The reviewer independently found
    70 bounded motion-window candidates, first at 68.92 s. V18 uses those
    kinematic conditions only as prerequisites: actual two-second displacement
    must also be uphill relative to the fresh terrain gradient, and the live
    stopping, remaining-corridor and RELEASE guards must pass. Exact zero is
    restored under the same request, destination, axis and shared budget.
    Ordinary drive heading and stationary turn admission remain separate.
    The implementation and single trial-20 gate received **8.5/10**, following
    **217** independent focused tests and the full **512-test** suite. Whether
    the physical policy continues climbing is the experiment's open question.

25. **Executed DRIVE continuation.** Trial 20 restored RELEASE at 68.92 s,
    gained 0.049 m body height over 2.62 s, then requested forward correction.
    Only 0.16 s of forward command was emitted before another posture pause.
    After posture recovered, original-heading admission prevented continuation
    despite 159 measured bounded-motion windows. V19 admits only previously
    observed DRIVE in the same request, matching direction and committed axis,
    with recovered posture and all geometry, backslide and budget checks.
    BRAKE/RELEASE-only evidence and previous requests do not qualify. Admission
    and actual output are logged separately. The reviewer passed **225** focused
    tests independently, verified **520** full-suite tests, and rated both the
    implementation and one trial-21 gate **8.5/10**. Trial 20's actual reverse
    speed of 1.998 m/s and tilt of 62.91° remain material failed outcomes; the
    rating does not establish braking capability or ascent reliability.

26. **Finite actuation exposure.** Trial 21 reproduced 3,658 trial-20 pose/command
    samples through 73.14 s and resumed forward four times. The original
    66.70–78.70 s stall epoch contained 2.26 s DRIVE, 2.62 s RELEASE, 6.28 s
    BRAKE/admission and 0.84 s posture observation. DRIVE/RELEASE intervals gained
    0.335 m along the committed axis; braking intervals lost 0.481 m. These are
    associations, not causal claims. V20 separately bounds deliberate exposure
    at 12 s and total elapsed time at 24 s, renewing only on 0.10 m net advance
    from the fixed anchor. Zero-exposure budget expiry is censored. Posture,
    speed, backslide and geometry protections are unchanged. Passive orientation,
    angular velocity and observed terrain-normal logs support later diagnosis.
    The reviewer approved the proposal, implementation and one trial 22 at
    **8/10**, independently passed **236** focused tests, then **11** after a
    provenance-field correction; the full suite passed **531**. The experiment
    asks whether finite extra actuation helps; it is not an ascent success.

27. **Terrain-relative posture entry.** Trial 22's climb ended at 79.24 s by
    measured backslide, before either new budget expired. Additional patience
    supplied no additional drive. Same-tick vector attitude analysis found
    8.66–14.72° tilt relative to the observed fitted normal at five earlier
    posture interruptions, with fast outward rotation. V21 introduces only the
    reviewed provisional forward-uphill entry exemption (relative ≤20°,
    absolute <50°), an independent 50° pause, and retains the 60° hard fault.
    Invalid/inconsistent inputs fall back to the original rapid trigger; active
    pauses and all other motion guards remain authoritative. Raw rapid-change
    signals remain available for timeout classification. Review corrected an
    old test helper that forced RELEASE after an already-valid pause, and an
    overflow-to-NaN normal calculation. Original adverse-test expectations were
    preserved. **246** independent focused tests and **541** full-suite tests
    passed. The proposal, final implementation and one trial 23 each received
    **8/10**. This is a provisional simulator experiment, not a validated
    stability boundary or a guarantee against tip onset.

28. **RELEASE patience.** V22 observes a complete two-second response before five continuous seconds with either axis or gradient progress below 0.01 m/s requests a checked correction. All outer budgets and protections remain. Proposal, implementation and one trial 24 each received **8/10**; **253** independent focused tests and **548** full tests passed. Trial 24 first diverged from trial 23 at 71.54 s, keeping zero command. The last 7.86 s RELEASE gained only 0.016 m along the axis; intermittent motion kept the continuous plateau under 1.86 s and no correction emitted. The shared 12 s exposure budget ended ascent at 83.14 s. A later reverse traveled 2.36 m, but recovery continued correcting toward its nominal endpoint after reaching gentle terrain and ended at 203.22 s. No ascent success.

29. **Functional recovery staging.** Trial 24 reached gentle ground but continued correcting toward a nominal reverse endpoint for roughly 41 s. V23 adds recovery-only `RECOVERY_STAGE`: actual drive and at least 0.4 m displacement, fresh complete gentle rotation and stopping footprint, fixed six-second observation and 0.5 m anchor, and contiguous 0.6 s measured settle. The result is censored `REPOSITIONED`; normal planning must still select a new ascent. The reviewer rated the proposal **9/10**. Code review caught and corrected posture-priority and physical-backslide admission bypasses, nonfinite observations, the valid infinite-clearance sentinel, and one-tick map changes before result consumption. **267** independent focused tests and **562** full tests passed. The implementation and one trial 25 each received **8.5/10**. No RELEASE, speed, route, posture, physics, policy, or lateral-limit changes were included. Trial 25 ended HOLD at 277.94 s. Commands/poses matched trial 24 through 160.80 s. Stage entry at 160.82 s, measured settle at 163.50 s, and next-tick censored handoff preserved all 87 prior directional exclusion bits. A distinct complete route was selected at 167.52 s and actually attempted. It was censored at 176.74 s after passive heading drift brought the stopping sweep near an observed hazard; two later physical recovery departures were retained. No ascent success.

30. **Measured DRIVE-to-RELEASE selection.** Both trial-25 ascent approaches switched to zero at five seconds of forward exposure while making about 0.30 m/s axis progress. V24 makes exposure a minimum: a complete continuous two-second window of deliberately emitted DRIVE and one continuous second with either axis or gradient progress below 0.10 m/s are also required. Short history resets on interruption, invalid support, or a gap above 0.04 s. All existing outer budgets and guards remain. The proposal received **8/10**, implementation **8.5/10**, and one trial-26 experiment **8/10**, following **275** independent focused tests and **570** full tests. Existing RELEASE-response fixtures are separated from the new 50 Hz admission tests; original failure expectations remain. Manual demonstrations do not establish this threshold as optimal. Trial 26 ended HOLD at 92.06 s with no mission completion. Its first changed command at 48.46 s continued productive drive; no measured RELEASE admission occurred. Physical backslide at 60.38 s ended the climb.

31. **Frozen survey route diversity.** All six initially admissible trial-26 routes shared an 8.77 m, 45° climb. The reviewer approved a read-only, eight-sector diagnostic at **9/10**, with up to sixteen existing 60,000-expansion searches, complete approach/onward/retreat geometry checks, separate surveyed geometry and fresh live admission, and unchanged mission evidence. No controller edit or physics trial is authorized by this diagnostic gate.

32. **Initial untested-corridor coverage experiment.** V25 adds optional full-survey sector candidates and picks the lowest original-cost eligible complete route from a previously unexecuted sustained climb class. Prior runs supply scheduling coverage only; mission probabilities, failure counts and exclusions remain fresh. The reviewer approved the proposal at **8/10**, implementation at **8.5/10**, and one trial 27 at **8/10**. Review corrected success bookkeeping: `physical` marks failure attribution, so completed successful attempts must collect their actual uphill DRIVE chunks across movement outcomes rather than requiring that flag. **287** independent focused tests and **582** full tests passed. Movement source is unchanged from V24; baseline/default route ranking and protections remain. Trial 27 ended HOLD at 94.08 s. The new route was selected and adopted, but its entrance drove only 0.951 m while yaw drifted 15.52°. It was correctly censored at 53 s before the sustained climb; a known rock lay along the emitted body heading. One later physical recovery failure remains recorded. No ascent success or new sustained-climb coverage.

33. **Separate entrance scheduling evidence.** V26 distinguishes actual entry into a climb-containing primitive followed by a completed execution-geometry censor from sustained-climb characterization. Import validates projected advance, direction and approach membership, associated terminal outcome, actual emitted command/immutable target/pose in CSV, and archived guard semantics. The entrance keeps zero probability/exclusion votes and its climb remains uncharacterized. The proposal received **8/10**, implementation **8.5/10**, and one trial 28 **8/10**. Both independent and primary full suites passed **588 tests**, including **18** ledger tests. The automatic unchanged-cost selector chooses the next northern entrance using trial 26 and 27 experiment metadata. Trial 28 ended HOLD at 93.56 s with the unchanged V24 movement executor. From 53.72–57.94 s uninterrupted forward drive gained 1.321 m horizontally and 0.340 m vertically. Exact captured-map replay found three stopping-fan rays touching the stale cell (1.1, 4.9), last observed at 0.4 s; all nine rays pass the remaining geometry checks. This is observation loss, not a physical entrance failure or sustained-climb characterization. One later recovery lateral departure remains a physical failure. No ascent success.

34. **Checked forward command selection.** Trial 28 stopped despite productive climbing because one stale cell lay in the nominal stopping fan. Captured-map replay rejects 0.40 m/s and accepts 0.375 m/s with all existing checks. V27 probes a bounded descending speed grid, retains actual velocity in stopping calculations, and validates/rechecks the output before emission. Review found and fixed mutable-command ceiling validation. Legacy trial26/27 records have explicit reviewed source hashes and scheduling-only labels. Proposal and implementation received **8.5/10**; **603** independent tests passed. One trial29 received **8/10**. It matched trial28 exactly through 57.92 s, emitted reduced speed at 57.94 s, and passed that freshness stop. The main climb ended at 65.78 s by the 0.50 m lateral tracking bound despite positive height/progress; final HOLD was at 87.96 s. No mission success. RELEASE zero, reverse, priorities and all deadlines remain unchanged.

35. **Occupied directional recovery exit.** Trial29 created a directional exclusion after its lateral tracking failure. All eight subsequent captured reverse-recovery rejections came from incidental drift through that mask. Full-mask replay allows six outward paths and continues rejecting a deeper diagonal crossing and outside-to-inside re-entry. V28 applies non-increasing depth checks only to incidental reverse-recovery stopping/braking; commanded reverse, forward movement, geometry and all evidence remain checked. The proposal and implementation received **8.5/10**, with **614** independent tests passing. One trial30 received **8/10**. It matched trial29 through 65.98 s, then performed 7.66 s and 10.40 s reverse traversals. Both stalled before staging; all exclusions remained. It ended HOLD at 116 s without completing ascent. No adaptive uphill continuation or lateral-failure reclassification is included.

## Resulting contract and verification

| Concern | Implementation / verification |
| --- | --- |
| 33° uphill | New ascent profile and planner use 33° without the former hidden 1° subtraction; cross-slope remains separately configured at 25°. |
| Steep turns | Straight heading states above gentle staging; zero autonomous yaw above 15° footprint terrain inclination; turning re-enabled below 12° after motion settles. Body lean has separate 25°/20° pause/resume thresholds in v2. |
| Continuous poses | Lattice retains measured sub-cell offset; straight 33° starts work at arbitrary tested offsets. |
| Updated terrain | Initial heading and every bend are rechecked at asynchronous plan adoption and primitive handoff. |
| Tiny/final legs | Nearby endpoints use supported SETTLE; short final legs are allowed and retain the actual mission destination. |
| Braking | Damping opposes measured longitudinal velocity; it has no position-restoration target. Unavailable stopping room requests zero desired velocity and records why. |
| Actual response | Slow 0.03 m/s progress continues; stalls, backslide, rapid posture change, lost support and settling drift have distinct measured outcomes. |
| Closed-loop correction | Measured-pose alignment, checked remaining corridor and latched drive heading; progress resets to the new axis; kinematic drift, overshoot and gentle-to-steep transitions tested. |
| Useful displacement | A supported alternative position is a censored reposition, not endpoint success or physical failure; observation and settling precede a new plan. |
| Cancellation | Immutable action IDs, consumed-once results, STOP/manual invalidation, no opportunistic plan preemption. |
| Calibration | Qualified mode requires the new execution protocol; the runner additionally requires exact policy, source and capability-profile provenance. |
| Recovery budget | A 120 simulated-second episode deadline interrupts even a still-active primitive; repeated ticks cannot extend it. |

The full-crater offline geometry check finds a complete route from `(0, 0)`
to `(13, 0)` with straight climbing sections connected by supported gentle
staging locations. It uses the labeled development survey and treats observed
geometry as fresh only for this offline route check. That is not sensor or
physical traversal evidence. See `outputs/ascent_behavior_tree/route_check_v2.json`.

## Remaining characterization questions

The bounded heading set can miss a feasible straight corridor at other entry
orientations. Velocity damping, the transition from slow slope creep into an
aligned straight primitive, and policy yaw drift require measured physical
outcomes. No trial may be treated as qualified without the complete ascent gate
and predictor calibration under this controller/protocol.

The later `corridor02` observational diagnostic captured the exact first lateral rejection at 65.78 s, with all 3290 prior pose/command rows identical to trial30. Its returned failure/brake command was not applied to physics. Forward local geometry still passes, but the current body ray misses the unchanged checkpoint by 2.61 m and has no confirmed gentle staging in the sampled corridor. Wider parallel paths also encounter directional grades above 33°. The diagnostic has no mission or qualification credit.

The reviewer approved the V29 initial paired-launch proposal at **8/10**: compare map-derived approaches within the same existing climb class, requiring both fewer preparatory bends and flatter full-footprint launch ground. Preserve the original eligible reference and costs; re-search unnecessary portal suffixes to the exact destination. No qualifying variant means diagnostic HOLD. The frozen-survey replay selected cost609.020/one bend/4.747° launch versus reference595.366/three bends/10.878°, with the same climb class and exact13,0 destination. Its synthetic3m freshness does not establish live admission. Review found and fixed repeated-coordinate launch-edge identification and copied the comparison into the actual adopted prediction event. All627 tests pass. Code and a single-run gate remain separate.

Physical logs must be checked for zero autonomous yaw above 15° footprint terrain, no combined
translation/yaw command, a 0.40 m/s ceiling, unique terminal action/attempt
outcomes, and correct destination accounting. Failed trials remain preserved.

V29 implementation was approved at **8.5/10** after **627** independent tests, with one trial31 at **8/10**. Trial31 adopted the paired route (live launch4.784° vs reference10.864°, one bend vs three), gained1.0896m body height during its main climb and ended HOLD at73.62s. Its body heading remained within1.584° of commitment at lateral failure. The last complete2s had only0.0213m/s axis and0.0079m/s gradient progress; shorter post-brake metrics were negative. Six speed brakes interrupted continuous DRIVE response, and no RELEASE was selected.

The unchanged-code corridor03 diagnostic received **8.5/10** and reproduced2442 pose/command rows through48.82s. Finer map sampling found possible body-heading staging with onward connectivity, but unchanged near-field freshness and robust settling/stopping-room checks do not establish an admissible handoff. The reviewer rated that handoff **7/10**; it was not implemented. Instead, the bounded one-episode uphill zero-response proposal received **8/10** for implementation:0.45m/s trigger, provisional0.50m/s damping boundary,0.20s response deadline, complete fresh2s powered history and all common guards/budgets. Code/run gates remain separate.

V30 implementation received **8.5/10** after **641** independent tests; one trial32 received **8/10**. The trial matched 1,887 samples through37.72s, then a zero command at37.74s reduced measured speed0.4833→0.3995m/s next tick. RELEASE sustained13.84s and gained0.502m height. Ascent remained incomplete: main climb departed the original lateral corridor at51.58s; subsequent uninterrupted reverse descended0.508m in15.22s before the same bound stopped it at67.18s. Its last2s still made0.0987m/s committed reverse progress. Active peak speed0.4833m/s and tilt31.893° were lower than trial31, but total main climb gain0.813m was also lower. No reliability or calibration claim. The0.50m/s response fallback and downhill overspeed branch remain test-covered, not live-proven by this trial.

The reviewer approved a purpose-filtered exact reverse-departure capture at **9/10** for implementation only. It preserves the first mission failure and its exclusion, skips nonmatching callbacks before any diagnostic mutation, retains the75s cap, and stops before applying the selected departure output. No recovery continuation or relaxed tracking bound has been approved.

The purpose-filtered recovery01 diagnostic received **9/10** for implementation and **9/10** for one run, after644 independent tests. It reproduced all3,360 pose/command samples through67.18s. The original51.58s mission exclusion remains; the reverse boundary outcome and+0.4m/s damping command were captured before being applied. Current reverse/stopping guards pass. A full surveyed floor region exists along the current reverse approach; distant freshness is not asserted. This diagnostic earns no mission or qualification credit.

The V31 functional recovery proposal received **8/10**, for implementation only: one fixed1m outer/.5m inner staging region per recovery episode, bounded checked approach, unchanged live fresh stopping and full region freshness before staging, original action/deadline/axis/progress/budget retained. The old point prediction is censored once and a separate unvalidated region phase begins before output. Region labels are excluded from the point-target calibration model. No new certificate can be chosen during the same episode. The existing measured6s/.6s staging contract yields only censored reposition.

V31 final implementation and one trial33 each received **8/10** after **679** independent tests and 45 post-telemetry checks. Default-disabled replay matched 1,741 archived observations exactly. The live run matched trial32 through67.16s; continuation at67.18s replaced the original lateral cutoff with reverse toward the fixed checked region. It physically reversed another9.48s, then entered overspeed braking at76.66s and tipped at77.28s (64.50°, peak actual1.931m/s). No core or staging arrival and no destination success. Whole-run controller p95 was15.35ms; admission46.07ms and continued pre-core p9515.63ms, including protective interruptions. Geometry was insufficient to establish reverse dynamic controllability. The independent audit is `outputs/control_review/trial33_independent_review.json`. The hard-fault path preserved one physical approach outcome but omitted its terminal movement event; it did not assign the distant tip position to the last locally executed reverse patch. Brake guard reasons after76.70s are not directly logged. Further implementation and simulator runs require separate review.

One unchanged-code climb01 diagnostic was approved at **9/10** and captured51.58s before applying its computed cutoff command; all2,580 pose/command rows matchtrial33. Independent fine passive-drift sweeps (`commanded=False`, actual body orientation) found no usable fixed landing region for the observed body/response directions. The current body ray stays geometrically clear10.45m but has no gentle full-footprint staging; the measured response ray becomes blocked after2.25m and exceeds35° support by3.35m. Original-heading gentle points would require a steep turn and lack robust expanded region support. Direct fixed-region continuation was rated **5/10**, rejected and not implemented. This capture is diagnostic, not another completed mission or qualification attempt.

The default-route V30 trial34 revisit received8/10 and ended HOLD130.44s. Its autonomous route exactly matchedtrial26 and2,427 rows matched through48.52s. The first changedzero48.54s reduced speed0.45794→0.37479m/s nexttick; RELEASE gained0.82678m bodyheight and1.65565m along-axis over19.68s. At68.22s the instantaneous overspeed was uphill, but the preceding2s netaxis/gradient progress was alreadynegative; this does not justify attributing failure solely to braking or relaxing speed limits. Real backslide ended the main approach77.02s. Two recovery endpoints settled, but all8actions exhausted before usable mission staging. Destination remainsunreached.

Optional passive event snapshots received9/10 proposal approval and8.5/10 independent implementation review. All690 tests passed independently;1,741 default-disabled archived-observation responses matched frozen movement exactly. The recorder captures first release-owned overspeed-brake entry, actual brake rejection, and failed/censored termination while leaving motion unchanged. It bounds detached captures to4 and64MiB cumulative array/JSON data, retains original underlying stoppingreason without a second geometry evaluation, and distinguishes returned/submitted/completed physics output. Submitted-but-failed steps have unknownapplication. Hardfault publishes the original pending movement result once without invoking recovery or adding an evidencevote. Further physics needs a separate gate.

The brake02 diagnostic received **9/10** and reproduced all 3,900 trial34 pose/command samples through 77.98 s. Two passive snapshots captured the RELEASE overspeed brake at 68.22 s and physical backslide termination at 77.02 s, with no drops or recorder errors. Average/maximum copy times were 2.69/3.14 ms. Exact contemporaneous braking, RELEASE, remaining-corridor and sampled command checks pass at both events: a hidden geometry veto did not cause this main failure. At 68.22 s instantaneous uphill speed was +0.510 m/s, but the preceding two-second net uphill progress was -0.0125 m/s. The later two-second original-axis rate was -0.1021 m/s, supporting physical backslide classification. No motion revision or further run is authorized by this diagnostic. See `outputs/control_review/brake02_independent_review.json`.

V32 response-memory implementation received **8.5/10**, with **708** independent full-suite tests and 29 final focused checks. One trial35 received **8/10**. All 3,494 pose/command samples before 69.88 s matched trial34; the independently guarded corrective forward command occurred once at 69.88 s. It did not restore sustained progress. The unchanged low-progress exposure budget ended the main approach at 80.58 s; final HOLD followed at 200.62 s, with no destination reached. All command/evidence invariants passed. Joint telemetry matched every command and next-step pose over 40,124 physics rows and 10,031 control steps, without reset or partial steps. Increased wheel motion during correction is an association, not direct slip measurement. Exact reverse snapshots show fresh physical stopping geometry blocked solely by incidental drift inside the occupied directional exclusion; recovery semantics remain unchanged. See `outputs/control_review/trial35_outcome_review.json` and `trial35_exclusion_analysis.json`.

The reviewer approved an isolated V33 corrective-return proposal at **8/10**: only after the single actually emitted response-memory correction, allow the existing complete two-second powered window and one-second deficient-response dwell to select guarded RELEASE after at least three seconds of actual forward exposure. Ordinary DRIVE retains five seconds. Productive correction continues; protective interruptions do not accrue exposure or renew outer budgets. The opportunity retires at first intentional RELEASE, alignment, STOP, result, or new request. Recovery-exclusion changes are outside this experiment.
