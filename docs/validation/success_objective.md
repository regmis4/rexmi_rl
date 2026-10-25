# Success objective: implementation and evidence boundary

2026-10-17. Budget: at most 20 percentage points of weekly usage consumed.
No simulation was run in this revision. Autonomous crater ascent remains unsolved.

## Plan and execution

1. Audit existing probability data before changing the objective.
2. Compare the existing autonomous candidates with demonstrated useful uphill motion.
3. Isolate terrain rejection from the planner's straight-on-slopes restriction.
4. Implement only the probability ordering supported by the existing interfaces;
   preserve unknown estimates and unchanged fallback behavior.
5. Verify the exact implementation and stop before a simulator task without a
   new executable route hypothesis.

## What changed

`success_selection.py` ranks validated, comparable whole-leg approach-success
estimates before terrain cost. Cost breaks exact probability ties. The existing
completeness, eligibility-tier and physical-admission filters still precede it.
The compared prediction must cover the entire route from the current planning
start to the same destination; a partial checkpoint estimate cannot stand in for
the whole route. Cell probabilities are never multiplied.

The runner checks model qualification and exact policy hash, controller source
hash and capability profile before enabling probability ranking, including in
characterization mode. An unavailable, invalid, out-of-domain or partial-leg
estimate remains unavailable. If no comparable validated estimate exists, the
current terrain-cost winner is retained exactly and explicitly labeled as a
characterization fallback.

Route-comparison logs, the dashboard and Isaac Sim overlay report the selection
reason separately from terrain cost. Displayed estimates are labeled plan-time;
the normal attempt logger still records predictions updated at plan adoption.

**This is objective plumbing, not a trained success predictor.** The current
model's historical target is generic `approach_outcome=success`, including some
frontier arrivals without settling. It must not be described as a calibrated
settled-exit or complete-round-trip probability. That stronger target requires
its own correctly labeled, held-out evaluation.

No model was trained, no manual trace became a runtime waypoint, and no motion
primitive, speed/grade limit, policy, physics, observations, survey or recovery
timing changed. Remote protocol is `persistent-remote-v16-success-selection`.

## Findings

- The bounded ledger covers 17 completed remote runs and 36 approach outcomes:
  17 successes, 7 physical failures, 12 censored outcomes. Attempts whose planned
  net ascent exceeded 1 m contain 7 failures, 6 censors and **zero successes**.
  These are heterogeneous controller runs, not interchangeable training trials.
- Two manual recordings yielded 28 nonoverlapping two-second useful uphill
  intervals. Matching used actual displacement direction within 20 degrees and
  a lateral distance no greater than the robot's 0.274 m half-width. Candidate
  uphill gain was credited once, excluding grades below 10 degrees. These are
  diagnostic motion observations, not autonomous-success labels or probabilities.
- For the captured floor query on the **original survey**, the existing planner
  admitted one complete corridor, heading approximately 69 degrees. It had no
  matching demonstrated uphill support. Western screens overlapped as much as
  1.68 m of demonstrated height gain, but their full straight approaches violated
  existing grade/footprint checks. Reweighting could not select them.
- One offline-only diagnostic changed just the turn-staging inclination predicate
  from 15 to 35 degrees. It found a complete geometrically clear path, maximum
  terrain inclination 34.71 degrees, but needed a 26.57-degree heading change on
  26.04-degree terrain and a 30.18-degree change on 28.34-degree terrain. It had
  no matching manual support. The current steep DRIVE command emits zero yaw:
  accepting passive drift is not an executable steering primitive for these bends.
- The original survey is not the missing historical planning-time local map.
  The latest live run selected a different central path after observations;
  this offline result must not be presented as its exact planning replay.
- Manual recordings share task and policy but differ in operator command history
  and mass; they lack an independently certified recorded terrain geometry hash.
  They establish feasibility, not autonomous probability calibration.

## Verification and review

427 focused and compatibility tests passed in 30.22 seconds. Tests cover probability
ordering over terrain cost, equal-probability ties, mixed known/unknown estimates,
invalid numbers, mismatched starts/goals, partial legs, incomplete candidates,
unvalidated estimates, exact model-domain matching, physical rejection precedence,
shared overlay explanations, and the existing motion/protection regressions.

The captured original-survey query retained the identical four-point route and
cost 1048.9803124769658 with no calibrated model. This proves fallback preservation,
not improved physical completion.

Final controls acceptance: 8.7/10 (RB-SUCCESS-SELECTOR-EXACT-51). Final practicality
acceptance: 8.6/10 (P4-EXACT-68). Agent 2 accepted the exact implementation under
RB-A2-SUCCESS-SELECTOR-51. After the target-label wording correction, all 23 selector
tests passed again. Final exact source hashes and review acceptance are recorded
alongside the audit artifacts. Weekly usage was 7% consumed at completion,
below the authorized 20% ceiling.

## Next engineering step

Before spending another ascent simulation, establish an executable approach
alternative. Characterize bounded steering/positioning on the relevant terrain
and compare it with the manual command history, or find another complete
straight-drive connection within the existing limits. Then collect completed
attempts under one frozen controller with an explicit success target. Fit and
evaluate probabilities across separate complete runs; measure calibration as
well as completion. Easy height gain should count only when the approach retains
a checked onward connection; it is not itself evidence of successful exit.

## Artifacts

- `outputs/remote_autonomy/success-objective-audit/attempt_ledger.json`
- `outputs/remote_autonomy/success-objective-audit/audit.py` and `audit.json`
- `outputs/remote_autonomy/success-objective-audit/topology.py` and `topology.json`
- `outputs/remote_autonomy/success-objective-audit/route_gap.png`

The existing GitHub backup was not modified.
