# Reported climb and recovery investigation

The reported run exposes navigation-supervisor errors. It does not establish
that the rocky-slope policy cannot climb or reverse.

## What the recording establishes

Source: `logs/nav/checkpoint_2026-09-23_20-06-34.csv` and its event/provenance files.
The pasted console excerpt matches this recording. The recording ends at 62.26
simulated seconds in RECOVER, without a final summary; a later final HOLD is
not established by this evidence.

| Time | Event | Meaning |
| --- | --- | --- |
| 11.78 s | First checkpoint reached and settled | The initial floor shelf worked. |
| 14.84 s | Complete route to `(13, 0)` selected | The whole-crater prior was available. |
| 28.92 s | Predicted drift stopping corridor intersects obstacle clearance | Geometric rejection; no physical stall or tip was recorded. |
| 30.92 s | Floor shelf `(-0.9, -1.3)` selected | The rejected exterior endpoint had been deferred for 20 seconds, also suppressing other routes sharing that destination. |
| 31.76–62.26 s | Repeated recovery/replanning | No committed local maneuver ever started. |

Across 32.52 seconds logged in RECOVER, the mean command was −0.142 m/s.
Commands at or below −0.35 m/s occupied only 0.78 seconds. Those commands were
braking outputs, not a sustained reverse attempt. No physical-failure exclusion
was recorded. The GPU acquisition warning in the pasted text is a performance
warning; the navigation event identifies the actual trigger separately.

The route search was also incorrectly carrying the direct search's completion
flag into shelf-connected routes. A valid shelf-to-exterior connection could
therefore be labeled incomplete, and the known exterior goal treated as an
exploration frontier. In recovery, global planning could repeatedly preempt
the local maneuver checks.

[Recorded-run analysis](../../outputs/ascent_recovery_review/reported_run_analysis.json)
preserves the measurements. A diagnostic using the saved 60-second map found
checked motions at some reported poses, including reverse at the 60-second
pose. Earlier-pose checks use later geometry and are diagnostic comparisons,
not exact reproductions of the earlier sensor state.

[Command comparison](../../outputs/ascent_recovery_review/reverse_command_comparison.png)
shows the reported recovery output beside trial 05. These are separate physical
runs, not a deterministic replay of the reported pose.

## Changes

- Recent geometric rejection adds a temporary cost to the local approach,
  rather than prohibiting every route with the same endpoint.
- Shelf-connected routes establish their own completion status.
- A checked local adjustment gets priority over another global route search.
  Pending planning cannot indefinitely preempt a checked maneuver; a queued
  stale plan cannot interrupt an active commitment. STOP still interrupts it.
- Local scoring values displacement and obstacle clearance before distance
  to the final mission goal. The selected forward/reverse command is retained
  while its path remains executable, with measured progress reported.
- Arrival tolerance for a bounded adjustment is 0.12 m, with current footprint
  support. The previous 0.25 m tolerance could declare a 0.40 m reverse complete
  after about 0.16 m of actual displacement.
- Reverse positioning is distinct from a downhill regroup leg. It uses the
  normal grade limits instead of inheriting the regroup leg's 8° uphill cap.
- Immediate geometry cancellations are censored, keep that command deferred,
  and do not exhaust the budget for meaningful recovery attempts. Failure
  counters remain tied to independent physical outcomes.
- Overspeed braking also applies during committed recovery. Every command
  remains within ±0.40 m/s; normal 25° directional limits, 35° overall terrain,
  footprint, tip/drop and directional exclusion checks remain in force.
- Committed maneuvers now use actual motion evidence, including reverse stalls,
  backslide and lateral drift. Slow steady progress remains commanded. A sustained
  stall or rapidly worsening body posture retires one physical attempt rather
  than holding the same command until the hard tilt threshold.
- Steady progress as slow as 0.02 m/s gets up to 20 seconds for the same bounded
  maneuver, rather than being judged against ideal commanded speed. Checked
  steering feedback corrects observed yaw drift. If actual displacement reaches
  its bound away from the predicted endpoint, the controller settles and replans
  from that supported pose and censors the original endpoint label.
- If no maneuver can execute, the log reports its rejection reason. Timeout
  without any maneuver is no longer described as exhausted physical approaches.

## Physical evidence

Runs and exact source snapshots are preserved under
`outputs/ascent_recovery_review/`. These are characterization runs with an
unvalidated probability model; none can establish the ascent qualification gate.

Trial 01 isolated recovery sequencing. It started a −0.40 m/s reverse at 33.04 s,
but exposed the premature endpoint acceptance: it ended at 35.42 s after only
about 0.16 m of longitudinal backout. Subsequent adjustments were rejected;
the run ended in HOLD at 52.08 s.

Trial 02 tested longer commitment, corrected endpoint acceptance, and separate
reverse-positioning limits. It exposed goal-distance bias in local scoring:
forward adjustments progressed toward the exterior but lost room around the
rock. It ended in HOLD at 122.06 s. These unsuccessful runs are retained rather
than presented as validation of the final recovery behavior.

Trial 03 tested clearance-based local scoring. It committed −0.40 m/s reverse
at 27.46 s but made little progress, then developed overspeed and excessive body
tilt. It stopped at 32.06 s. This is evidence against treating sustained reverse
alone as a sufficient fix. It motivated adding measured motion and posture-trend
monitoring to the committed maneuver, which previously relied primarily on
geometry, endpoint distance and timeout.

Trial 04 added motion/posture monitoring and stopped the ineffective reverse
attempt rather than reaching the excessive tilt seen in trial 03. It ended in
HOLD at 119.80 s, still unable to complete an escape. Its short progress window
also showed why a generic stall threshold was too strict for slow but moving
slope traversal. The final controller accepts steady progress down to 0.02 m/s,
extends the bounded maneuver time to at most 20 seconds, and adds checked
heading feedback.

An offline comparison with the same full prior illustrates the directional
tradeoff: the normal limits selected a 22.3 m route with peak planned uphill
23.7° and cross-slope 23.5°. A separate 35° characterization configuration found
a 24.8 m route with uphill 31.5° and cross-slope 12.1°, while its overall footprint
terrain remained below 32.4°. This is a geometry comparison, not a probability
estimate or proof that the steeper approach succeeds in physics.

Trial 05 used the final controller with the normal 25° uphill/cross-slope
limits. It held a −0.40 m/s reverse for about 6.8 seconds, moving about 0.49 m
before actual progress stalled. It ended in HOLD at 122.02 s, with one physical
failure, 28.0° peak active body tilt, 0.455 m/s peak actual speed and 0.273 m
accumulated backslide. This verifies a meaningful committed attempt, not a
successful escape or completed ascent.

[Trial 05 map and actual trajectory](../../outputs/ascent_recovery_review/trial05/observed_route.png)
and the [machine-readable results](ascent_recovery_summary.json) retain the
failed run and commanded versus actual speed measurements.

Trial 06 separately characterizes 35° uphill/cross-slope allowances while
retaining the 35° overall terrain boundary and ±0.40 m/s command ceiling.
Those wider directional allowances are not the normal defaults and are not
qualified capability. It completed bounded reverse maneuvers, selected another
ascent approach after a failed climb, and reached a lower regroup shelf at
208.4 s while retaining the exterior destination. It nevertheless ended in HOLD
at 359.84 s because a later committed regroup timed out. Across the run it
started 16 local maneuvers and recorded 10 physical failures. Peak active body
tilt was about 45.0°, peak actual speed 0.788 m/s despite the 0.40 m/s command
ceiling, and accumulated backslide 2.59 m. Widening directional grade limits did
not prove better mission reliability and produced more sliding in this trial.
The normal limits remain unchanged.

[Trial 06 map and actual trajectory](../../outputs/ascent_recovery_review/trial06_grade_characterization/observed_route.png)
shows both independently selected approaches, local recovery movement, and
the failed-direction exclusions. All six development runs ended without
completing ascent; intermediate local recovery success is not mission success.

## Regression verification and limits

All 291 navigation tests pass, including 72 ascent tests. The additional cases
exercise local rejection versus same-goal alternatives, independent onward
connectivity, planner/maneuver precedence, sustained reverse, meaningful
endpoint acceptance, slow progress, steering feedback, actual overspeed,
posture trends, directional physical failures, bounded displacement, and STOP.
The baseline controller and observed-map implementation still match their
preserved baseline copies. Policy weights, robot physics, observation layout,
and the shared teleop command path are unchanged by this investigation.

The original supervisor defects are corrected, but autonomous ascent is still
unproven. Unit tests do not establish physical traversal or probability
calibration. These runs have no calibrated predictor, so none counts toward
the three-consecutive-success ascent gate. Descent and speed optimization
remain gated.

## Commands

The [existing GUI command](../ascent_navigation.md) runs the updated controller,
dashboard and Isaac Sim overlays with the same saved full-crater prior. No
policy or physics change is required.
