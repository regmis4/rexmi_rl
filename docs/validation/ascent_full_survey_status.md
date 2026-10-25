# Full-crater ascent development status

The subsequent [recovery investigation](ascent_recovery_status.md) addresses a
reported GUI run, local maneuver commitment, and measured response. The trials
below preserve the earlier full-survey implementation and its original results.

The full-crater information condition and Isaac Sim overlays are implemented
and verified. **Ascent has not passed its qualification gate.** Four completed
full-survey characterization missions ended in HOLD; none reached the exterior.
Descent, round trips, and performance/high-speed development remain gated.

## Terrain information

The user-selected development survey samples the crater mesh before the mission
with 409,600 downward rays at 5 cm spacing. It supplies 1,024 m² of elevation
evidence, with 100% known coverage inside the required 16 m radius. The planner
receives this geometry through the normal costmap and selects routes itself.
The survey is explicitly labeled **simulator development ground truth**, not
an onboard LiDAR survey. Live LiDAR continues updating the local map.

The reusable initial artifact is
[`run.prior.npz`](../../outputs/ascent_full_survey/trial01/run.prior.npz).
Its geometry SHA-256 is
`87552488ff3913451653dc60010d0bdbe3f7f063c53fcd6860289d83bfb25671`.
Loading validates identity, task, seed, grid and mission-area coverage,
recomputes terrain costs, and clears inherited scan freshness and outcomes.
Prior geometry never supplies success/failure votes or fresh robot scans.

The earlier [local-only trials](ascent_v2_status.md) remain a separate historical
baseline. They are not presented as evidence for the new information condition.

## Physical results

All runs use seed 42, the established floor spawn, the exterior goal `(13, 0)`,
the unchanged rocky-slope policy and a 0.40 m/s command ceiling. Controller
versions changed between development runs, so this is not a qualification
series. Peak tilt and actual speed below exclude the first three seconds of
spawn settling. Terrain inclination and robot body tilt are distinct measures.

| Trial | Change tested | Result / simulated seconds | Physical failures | Peak body tilt | Peak actual speed | Backslide |
| --- | --- | --- | --- | --- | --- | --- |
| 01 | Full survey, original search budget | HOLD / 24.40 s | 0 | 16.3° | 0.469 m/s | 0.311 m |
| 02 | Larger bounded global search | HOLD / 110.42 s | 2 | 30.8° | 0.506 m/s | 0.351 m |
| 03 | 1° uphill planning reserve; 25° execution limit | HOLD / 122.06 s | 0 | 35.9° | 0.470 m/s | 0.396 m |
| 04 | Separate 30° uphill characterization; side limit 25° | HOLD / 153.02 s | 1 | 35.6° | 0.540 m/s | 0.369 m |

Every run's peak command was 0.40 m/s. Measured speed can exceed the command
while slipping; command compliance does not establish slope controllability.
The normal uphill/side limits remain 25° and the footprint terrain boundary
remains 35°. The 30° trial does not establish a validated higher capability.

The first test exposed the original 5,000-expansion search limit. With the same
survey, a bounded 25,000-expansion search found a complete approximately 21.5 m
route; the initial live search used 6,353 expansions. Subsequent runs therefore
planned using a complete crater-to-exterior route, not unexplored frontiers.

The remaining failure is execution near hillside bends: a legal planned edge
does not ensure that the follower's current heading, stopping corridor and
under-turning arc remain legal. In trial 02 a 25.3° predicted uphill corridor
rejected a route planned below 25°. The planning reserve allowed a shelf stop
in trial 03, but downhill drift near a rock later invalidated its stopping
corridor. Trial 04 still reached a 30.1° rejection. Simply increasing the grade
limit is not a demonstrated fix. Further ascent work must reconcile route
curvature, tracking variation and physically executable approach adjustments.

No calibrated full-survey predictor exists yet. Predictions remain explicitly
unvalidated; neither mission completion nor the CPU search establishes an 80%
reach-and-settle probability. The model contract now includes the terrain
source and geometry identity, so local-only models cannot authorize these runs.

## Visible overlay check

`visible01` ran for 12.98 simulated seconds in visible Isaac Sim. Its summary
reports the overlay enabled with 15,699 rendered terrain tiles and no overlay
render error. The viewport capture was inspected:

![Isaac Sim costmap, route and LiDAR overlay](../../outputs/ascent_full_survey/visible01/viewport.png)

The panel offers costmap, live LiDAR, route and ray toggles; terrain, slope,
roughness and traversal layers; and a development-source label. Ascent costs
use a continuous gradient, with shelf outlines. Amber shows the complete
mission route, red the active leg, and green the checkpoint. This short display
check is recorded separately from mission trials and is not an ascent success.

See the [GUI and headless commands](../ascent_navigation.md).

## Evidence and checks

- **274 regression tests pass**, including prior identity, fresh local motion
  requirements, retained geometry beyond the fresh scan area, empty inherited
  mission evidence, insufficient survey coverage, and updated-survey reloads.
  The final targeted ascent rerun passed all 55 tests.
- Navigation modules and launch/report scripts compile; `git diff --check` passes.
- The baseline checkpoint controller and observed-map module still match their
  saved copies byte for byte. Policy weights, robot physics, 247-observation
  layout and shared teleop command injection were preserved.
- Controller p95 was 2.45–5.45 ms across mission trials. Mapping averaged
  48.67–76.28 ms per update. These are measurements, not a performance gate.
- [Machine-readable results](ascent_full_survey_summary.json) include outcomes,
  recovery counts, backslide, tilt, speed by terrain and runtime measurements.
  Raw CSV, events, provenance, source snapshots and surveys are preserved under
  `outputs/ascent_full_survey/`.

![Full survey and trial 03 trajectory](../../outputs/ascent_full_survey/trial03/observed_route.png)

Regenerate this report with:

```bash
/home/susan/miniconda3/envs/env_isaacsim/bin/python scripts/report_ascent_trials.py \
  --root outputs/ascent_full_survey \
  --output docs/validation/ascent_full_survey_summary.json \
  --visual-check visible01 --plot-run outputs/ascent_full_survey/trial03
```
