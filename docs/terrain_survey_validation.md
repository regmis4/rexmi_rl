# Terrain survey validation — 2026-09-16

Status: implemented; full autonomous crater traversal is **not validated**.
The policy and demo contact materials have not been changed to improve results.

## Automated checks

76 tests pass: 70 map/navigation/command checks and 6 existing perception checks.
They cover cream exits without unnecessary settling, footprint exclusions,
unknown terrain, slope-detrended roughness and sample support, directional
failure memory, successful-corridor preference, gentler route selection, the
15% replacement threshold, survey-frontier selection, sideways tracking
failures, braking out of a failed direction, historical-map retention and
hazard-preserving distant display aggregation. Syntax and whitespace checks pass.

## What the controlled slope comparison found

Eight matched 30° conditions used the same policy, a flat approach, 2 seconds
settling, 10 seconds of forward command, then 3 seconds of zero velocity command.
Static robot friction was 0.8. Terrain friction was 1.0 with multiply combination.
The effective robot material values were read back from PhysX. All eight final
conditions initialized successfully and none tipped during the 15-second trial.

| Generated texture | Dynamic friction | Command m/s | Height gained m | Max lateral displacement m | Drift during stop m |
|---|---:|---:|---:|---:|---:|
| Smooth | 0.6 | 0.40 | 0.41 | 1.58 | 0.34 |
| Smooth | 0.6 | 0.80 | 0.96 | 2.54 | 0.40 |
| Smooth | 0.8 | 0.40 | 1.15 | 0.57 | 0.36 |
| Smooth | 0.8 | 0.80 | 1.84 | 1.63 | 0.36 |
| 5 cm texture | 0.6 | 0.40 | 0.67 | 1.85 | 0.42 |
| 5 cm texture | 0.6 | 0.80 | 1.28 | 3.07 | 0.43 |
| 5 cm texture | 0.8 | 0.40 | 0.99 | 1.65 | 0.48 |
| 5 cm texture | 0.8 | 0.80 | 1.71 | 2.77 | 0.46 |

These are fixed-command observations from one seed, not controlled-autonomy
success rates. The 5 cm figure is generated peak-to-peak height texture, not
measured roughness RMS. Robots can move sideways onto a different part of the
pyramid; height gain alone does not establish route tracking. The study suggests
that grip and geometry both matter, but does not identify the cause of every
crater failure. Smooth terrain is not universally impassable and more texture
is not universally better. Faster commands improved height gain but did not
establish acceptable tracking or stopping; the autonomous default stays 0.40 m/s.

An initial study placed robots directly on the slope; they tipped before the
speed comparison. Those initialization failures are excluded from the table.
The final harness starts on the flat approach and reports initialization validity.

The live demo's effective robot friction is static 0.8, dynamic 0.6; the terrain
is 1.0 with multiply combination. No per-patch friction difference was found in
the demo terrain generator. Visual whiteness alone is not a grip measurement.

Machine-readable results: [controlled study summary](validation/traction_study_30deg_summary.json).
The [survey guide](terrain_survey.md) contains the reproduction command.

## Autonomous development run

A 300-second seed-42 crater run reached 2/4 mission destinations and attempted
climb-out. It ended at the step limit, still following/replanning. Maximum
cross-track error was 0.762 m; four poor-progress episodes were recorded,
1.73 m cumulative body-backward travel under forward drive, 14 hold failures
and four rolling checkpoints. No manual intervention, policy switching or
automatic reset occurred. This development run preceded the final side-slip
recording and stopping-corridor refinements; it is not final-revision acceptance.

Its export retained 477.48 m² of directly surveyed terrain, 16,019 known cells
(including supported small gaps), observations from t=0, seven excluded
cell/direction pairs and 127 successful cell/direction pairs. Direct surveyed
area is distinct from the interpolated known area. The map remained intact.

A subsequent 90-second nearby-start climb trial exercised failure learning but
did not reach its uphill goal. It also exposed a renderer defect: defining new
USD chunks inside an attribute batch disabled the overlay. That defect was
fixed by creating chunks outside the batch and retested visibly. Four dashboard
layers were rendered from exported survey data and inspected.

## Final visible check

The corrected overlay remained enabled through a 60-second nearby-start trial;
the captured viewport was inspected and showed the route, hazards and survey
tiles. No new survey-chunk or opacity-index error appeared. Logged overlay
update time was median 25.4 ms and 95th percentile 125.2 ms;
map rebuild averaged 94.9 ms. These are processing times, not frame-rate claims.
The known velocity-arrow prototype warnings remain unrelated to the survey mesh.

**Navigation failed:** the robot tipped at 27.58 simulation seconds
and entered HOLD. It stayed stopped under policy control without an automatic
reset, and the test window closed at its 60-second limit. The uphill goal was
not reached. Maximum cross-track error before HOLD was 0.508 m. Three-repeat
full-crater acceptance is therefore unmet. A short successful unit or display
check must not be reported as reliable autonomous climb-out.

Local simulator artifacts were saved under `/tmp/rexmi-survey-change/` during
development. Summary records are preserved in `docs/validation/`; the survey
exports and corrected screenshot are retained in
`logs/nav/survey_validation_2026-09-16/` on this workstation.

## Acceptance still required

Three repeatable complete crater traversals on the final revision, with no
intervention, policy switching or automatic resets. Additional slope seeds and
closed-loop 0.80 m/s tracking/position-hold tests are required before raising
normal autonomous speed or claiming reliable crater exit capability.
