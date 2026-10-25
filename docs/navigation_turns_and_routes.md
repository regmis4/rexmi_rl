# Tight turns and gentler return routes

**Latest handoff (updated 24 October 2026):** [Current state, user concerns, preserved plots,
known failures and next-agent starting points](validation/crater_navigation_handoff_2026-10-23.md).
Autonomous crater traversal remains unproven. A new successful manual ascent,
measured path costs and the user's planning/recovery logic are now preserved in
the [manual ascent report](validation/manual_ascent_2026-10-24/report.md).

Checkpoints are green in both views. The dashboard shows the larger planned
route as a faint red dotted line, with the active checkpoint segment in stronger
red. Its log panel reports route length, maximum uphill/side slope and metres
of climbing above 25°. If the destination cannot yet be connected through
observed terrain, the panel says **Observed frontier route**.
See [ascent and retreat](ascent_retreat.md) for current grade limits and the
committed descent used to leave a failed climb. Normal flat turns remain unchanged.

## Normal turns first; reverse only at a dangerous slope

**Flat ground and clear, permitted slopes use ordinary route following.**
A heading change, nearby boulder, cream margin or unknown cell cannot by itself
start reverse-assisted turning. The controller tries the normal forward turn,
a checked slower forward arc, then ordinary alignment at up to 0.35 rad/s.

Reverse assistance requires **measured terrain above 35° inside the predicted
body sweep or drift corridor**, with no safe ordinary turn available. A flat
rim beside a cliff can still meet this condition; an unrelated steep patch
behind the robot does not. Unknown terrain still restricts motion, but is not
treated as evidence of a cliff that calls for reversing.

Rocky-slope can turn more slowly than commanded. Motion checks now consider the
requested curve, reduced yaw response and straight continuation. They check
the swept body footprint against unknown ground and slopes above 35°, alongside
the existing obstacle/stopping checks. A yaw-only request no longer skips the
measured drift check.

Only when that steep-edge condition holds and a clear rear corridor exists:

1. `TURN_BRAKE`: brake below 0.08 m/s for 0.3 s.
2. `TURN_REVERSE`: choose space 0.4–1.0 m behind and reverse at 0.15 m/s,
   with small heading corrections. Finish early once at least 0.15 m of
   repositioning opens a safe normal turn.
3. `TURN_STOP`: stop before changing drive direction.
4. Resume the normal follower as soon as its motion is clear. Only if it is
   still unsafe, `TURN_ALIGN` turns at up to 0.20 rad/s using a checked forward creep
   (0.12 m/s), reverse arc (0.15 m/s), or stationary yaw. Drive direction is
   selected after stopping and held for that manoeuvre, then the route is refreshed.

If reverse creep moves less than 0.08 m in two seconds, it can use the existing
teleop reverse command of 0.40 m/s, but only after checking that command's larger
stopping corridor and while more than 0.20 m remains. No reverse displacement
after six seconds causes an explicit hold instead of repeated ineffective cycles.

Rocky-slope remains active. The rear corridor must be observed and fresh and is
checked continuously. New obstacles, unsafe drift, poor heading tracking and
timeouts interrupt the manoeuvre. Three unsuccessful attempts near the same
location bound repeated reversals; a completed turn does not consume that budget.
STOP and manual takeover interrupt the sequence. If it
cannot stop safely, it requests assistance rather than swapping policies or
resetting the robot.

A 0.25 m turn-drift reserve applies to yaw-only turns beside a measured steep
edge and to selecting the reverse destination. It is not a general flat-ground
buffer around obstacles or unknown cells. This is an experimental
allowance, not a measured guarantee of stopping distance. Stored geometry at
the footprint perimeter remains usable when the robot occludes those cells;
freshness still gates the motion corridor. Unknown perimeter ground stays
excluded. Orange tiles receive no new blocked buffer.

## Choosing the larger route

A* already searched the retained observed map. Its directional costs now grow
nonlinearly for uphill travel above 15°, downhill travel above 20° and side
slope. Turning on inclined terrain costs more. Proximity to an excessive-slope
boundary adds a soft preference for body clearance. The 35° exclusion and tile
colours are unchanged; 15° and 25° are planning/reporting thresholds, not new
forbidden grades.
Route centres also require support for the robot's footprint, matching execution.
A clear centre cell alone cannot authorize a route with wheels over an orange
cliff or unknown ground. This adds no extra drift reserve to ordinary route planning.

At checkpoints, the selected route is compared with alternatives across the
retained map. The existing 15% improvement threshold limits route oscillation;
invalid routes are replaced immediately. Local checkpoints are prefixes of that
larger route, including when it initially heads away from the destination.
Cached routes also lose validity if newly observed steep terrain overlaps the
body footprint, even when the old centreline remains clear.

Poor forward progress is recorded even when the fitted slope looks easy.
The [slip tracker](slip_and_route_recovery.md) measures forward progress,
backsliding and sideways drift, including slow forward turns. A first confirmed
episode excludes that approach; a second overlapping episode marks a dark-red
no-go zone. Recovery checks a backoff or change of heading before selecting a
new route. Intentional stops, pure alignment turns, reversing and manual control
are excluded. The historical CSV name `climb_failures` also counts these
apparently easy-terrain failures for compatibility.

Uphill planning compares balanced, shorter and gentler approaches to the same
destination, including an observed frontier when the full goal is not connected.
Peak uphill and side slope, footprint terrain and distance all affect selection.
The recovery guide records the bounded detour rule and current trial results.

Roughness remains separate from traction: maximum roughness is not rewarded,
and white/smooth geometry is not automatically blocked.

## Manual-checkpoint return test

The default rolling 20 m mode deletes distant terrain. Use **full** retention to
reuse the entire current-run survey. No previous run is loaded. This command
keeps the opt-in flat-ground sidestep trial:

```bash
cd /home/susan/rexmi_rl
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgcc_s.so.1 ./run.sh scripts/navigate.py \
  --task RexmiRl-Go2w-Crater-Bowl-RockySlope-Play-v0 \
  --checkpoint logs/rsl_rl/go2w_velocity_rocky_slope/2026-06-30_09-31-48/model_13994.pt \
  --ckpt_rough logs/rsl_rl/go2w_velocity_rough/2026-06-14_20-03-41/model_8996.pt \
  --flat_sidesteps --perception_view --manual_checkpoints \
  --map_retention full --lidar_display_points 1000 --overlay_hz 30 \
  --log_file logs/nav/manual_return_route.csv --max_steps 30000
```

Click either dashboard map. Remove `--flat_sidesteps` to compare rocky-slope
alone. Use a different log filename for each run. Full retention uses more memory
than the rolling map; distant display tiles still use aggregation.

## Validation

In a completely observed two-corridor fixture, the previous costs chose an
8.26 m route with a 32° climb. The new costs chose a 13.85 m route with a 20°
maximum climb and no uphill exposure above 25°. This is a routing test, not a
measured robot-efficiency result. See [the fixture](validation/route_strategy_fixture.json).

Regression checks cover braking before reverse, fresh rear corridors, newly
blocked reverse paths, operator STOP, yaw drift, cliff footprints, cream exits,
directional failure learning and checkpoints following the larger route.
The initial route/reverse version passed 128 checks. The revised steep-edge-only
trigger also covers clear turns on 0°, 15° and 30° terrain, nearby unknown cells,
flat obstacle alignment, safe forward arcs before reverse, and ending a reversal
as soon as room is available. Current results are in
[the selective-turn check](validation/selective_turn_summary.json).
At that stage, all **137 checks passed**. Four sensor-fed simulator cases completed
using rocky-slope alone: flat left/right turns, textured flat ground and a
10° slope boundary. None entered a `TURN_*` state, tipped or switched policies.
Final early-release changes to cliff recovery have controller-test coverage;
full crater recovery has not been rerun with this revision.

In the earlier 100-second sensor-fed trial with reverse alignment arcs, the robot escaped
the earlier sticking point and reached approximately (7.65, −1.34) m without
tipping. It did **not** reach the (13, 0) m destination: it stopped beside steep
terrain. This exposed a planner/follower footprint mismatch. After correcting
the planner, a 30-second smoke run ended in FOLLOW without tipping; subsequent
cached-route footprint rejection is covered by regression tests. A full ascent
has not been rerun with all final corrections.

These runs started with an empty map. They do not establish performance on the
fully surveyed return trip. Policy, contact materials and terrain physics were
unchanged; the runs used rocky-slope only and no automatic resets.
Sensor-fed Isaac results and limitations are in
[the trial summary](validation/route_strategy_trials.json). Full crater navigation
remains unvalidated until three complete traversals succeed without intervention,
automatic resets or policy switching.

## Experimental ascent-first route objective

`--ascent_route_objective ascent_first` selects the remote controller's separate
ascent-first planner; `baseline` remains the default. It requires the validated
full-crater development survey. The option does not change policy commands,
physical limits, observations, recovery or STOP ownership.

The same bounded checkpoint graph supplies both searches. Backward search
certifies arrival-heading states with an entirely gentle, physically checked
continuation to the exact goal, including the joining turn. The forward search
ranks `(ascent effort, gentle continuation effort, total distance)` and stops
charging ascent effort at the first certified state. That state is a sampled
entry to goal-connected gentle terrain, not an independently detected rim.

Ascent effort integrates the displayed terrain cost multiplied by
`1 + clip((goal_height - local_height)/(goal_height - start_height), 0, 1)`.
The height reference stays fixed during each request; nonascending requests use
a unit multiplier. Exterior effort integrates displayed cost alone. The mode
uses no legacy uphill penalties, maximum-color ceiling or hard 0.20 m loss
veto. All edge costs are positive; contour travel and height losses remain
possible. Drawdown, cumulative descent and route length must be inspected.

No checked route means an explicit planning failure, not silent fallback to the
baseline. These are uncalibrated terrain-effort scores, not probabilities.
The reproducible saved-map comparison and review approvals are under
`outputs/remote_autonomy/ascent-first/`. Physical ascent qualification remains
separate from geometry-only planning.

The first matched-map trial passed endpoint, turn, footprint and objective-rank
checks but **failed the intended route-selection experiment**. The exact goal's
best sampled oriented footprint has 18.49° inclination, above the 15° gentle
classification threshold. Consequently the only exit was the goal itself; the
ascent/exterior split never activated. The selected central climb reached
31.99° uphill with 3.73 m above 25°, versus the stored baseline's 25.60° and
0.79 m. Reviewers rejected promotion and physical testing. This is a failed
phase-definition assumption, not evidence against the policy or the user's
route. An exterior-region definition independent of destination slope remains
necessary. Do not raise the classification threshold or move the goal merely
to fit this trial. Baseline remains the default.

## Climbing-burden objective

`--ascent_route_objective climb_burden` adds a separate experimental remote
planner. Its primary score is the integral of displayed terrain difficulty
against **positive elevation gained**. Total route distance breaks ties. Flat
and descending sections add no primary burden; they remain subject to all
physical, observation, directional-exclusion and heading checks. Every reclimb
is charged. There is no gentle-suffix classification or elevation-loss veto.

Height is a continuous bilinear field of fully observed circular-footprint
means. The integral splits at cost-cell boundaries, interpolation boundaries
and height stationary points. Consequently a hump between equal-height
endpoints is counted and splitting a straight leg does not alter its burden.
Unavailable footprint-height support is reported, not treated as zero cost or
physical failure. The two-ended search uses original directed heading-state
costs and a complete-path lower-bound stopping condition, not first meeting.

One-metre elevation bands are anchored at the request's starting height and
charge repeated upward crossings. The reported worst window spans one metre
of cumulative positive gain; intervening flat/descent travel remains in its
spatial interval. These reports do not alter route ranking. Geometry-only
optimality applies to the sampled checkpoint graph, not all continuous paths
or physical success. The baseline remains the default.

The first matched-map climbing-burden trial completed in 18.38 seconds and
passed independent route geometry, heading, endpoint and objective checks.
It reduced burden from 35.51 to 32.34 versus the baseline, but retained the
central corridor: peak uphill 32.78° and 4.04 m above 25°, versus 25.60° and
0.79 m. The worst one-metre positive-gain window was essentially unchanged
from the preceding ascent-first route (9.590 versus 9.584). Thus a lower
numerical objective is not evidence of easier physical ascent. 106 links
lacked the additional supported height interpolation required by this
objective; that is missing objective support, not physical failure evidence.
The trial and review evidence are under `outputs/remote_autonomy/climb-burden/`.
No physical trial or default-planner promotion followed this result.

## Wheelbase-scale uphill demand (experimental v25)

`--ascent_route_objective wheelbase_demand` minimizes the lexicographic pair
`(integral C * max((h_front-h_rear)/L, 0)^2 ds, horizontal distance)`.
`C` is the unchanged displayed terrain cost; `h` is the previous supported,
footprint-mean bilinear height field, with no extra smoothing. Each point uses
an overlapping centered window along its straight leg's heading. The window
extends onto observed terrain beyond leg endpoints; it never clamps height to
waypoints. Collinear subdivision leaves the score unchanged. Turns change the
window orientation and still require the original physical/turn checks.

Nominal `L=0.4505454148531053 m` comes from both robot URDFs and the default
stance in `assets/go2w.py`: hip x=±0.1934 m, thigh/calf offsets 0.213/0.2264 m,
front/rear thigh angles 0.8/1.0 rad and calf -1.5 rad. Wheel-center x positions
are 0.186454037 and -0.264091378 m. This is a fixed reference scale: actual
wheelbase changes with articulated posture. It is not footprint padding.

Integration splits at terrain-cost boundaries, both shifted interpolation
grids, and height-difference zeroes. Three-point Gaussian quadrature then
integrates each quartic exactly, within floating-point precision. Unknown
extended support makes the objective unavailable; it is not physical failure.
Flat/downhill primary demand is zero. Repeated climbs incur cost; distance
breaks ties. Same height gained over a shorter uniform ramp costs more.

This remains an uncalibrated geometry proxy: footprint averaging and net
front/rear height can hide short humps; existing roughness/step/footprint checks
remain necessary. It cannot prove traction, successful release climbing, or
that the sampled checkpoint graph contains the best physical route. Default
v22 and all control, speed, STOP, policy and physics behavior remain unchanged.

The single frozen-map v25 comparison completed in 25.71 s, independently
rechecked for endpoints, geometry, turns and score. Against v24, demand fell
15.1207→12.0280 and distance above 25° fell 4.04→2.15 m, but route length rose
15.13→28.64 m. Peak directional grade stayed 32.78°; peak nominal-wheelbase
height difference worsened 0.313→0.331 m. The 74-node graph accepted 1,776
links; 112 physically considered links lacked extended height support.
The approximate sketch, scored only AFTER search, scored 11.0496 but failed
turn eligibility and the 35° body/turn envelope (36.05°). This is not proof
that all routes through that corridor are infeasible. Candidate/turn coverage
remains unresolved. Exact reviews: controls 8.6, practicality 8.5; 204 tests.
Evidence: `outputs/remote_autonomy/wheelbase-demand/`. No physical simulation,
qualification or default promotion; no automatic second run or weight sweep.

## Stronger wheelbase penalty (experimental v26)

The user requested greater emphasis on high wheelbase elevation gains.
`wheelbase_demand` now selects objective `wheelbase-demand-v2-fourth`:
`(integral C * max((h_front-h_rear)/L, 0)^4 ds, horizontal distance)`.
Only the exponent, exact integration order and provenance/labels change.
Five-point Gaussian quadrature integrates degree eight after the same boundary
and zero splits. Doubling wheelbase height gain raises local cost sixteenfold,
compared with fourfold previously, at equal terrain cost and distance.
Absolute scores across exponents are incomparable; both candidate routes must
be rescored with the new objective. This remains an integrated objective, not
a guarantee of minimizing the worst section. Previous source is archived under
`outputs/remote_autonomy/wheelbase-demand-fourth/before/`; v25 results remain
unchanged. Baseline v22 remains the default. No physics approval is implied.

The one v26 frozen-map test passed independent route/score checks in25.88s.
Compared with the squared-demand route, peak wheelbase rise fell0.331→0.245m,
peak uphill32.78→25.60°, and distance above25° fell2.15→0.79m. However length
rose28.64→52.16m and cumulative footprint-mean height loss became1.34m.
The preserved21.97m baseline already has the same peak rise, peak grade and
steep exposure. Its quartic score2.3201 versus1.9747 for the new route shows
the extra30.19m buys14.9% less integrated score. Distance only breaks exact
score ties. Stronger local weighting works numerically but does not guarantee
mission-level efficiency or ascent completion. No physical/default promotion.
Evidence: `outputs/remote_autonomy/wheelbase-demand-fourth/`;205 tests pass,
controls/practicality8.6 each, A2 EXACT93. Same74nodes/1776accepted links and
112 unavailable extended-height links; no additional search or exponent sweep.

## Sustained longitudinal climb (experimental v27)

`--ascent_route_objective sustained_climb` ranks `(sum of bout effort squared,
horizontal distance)`. Bout effort is the integral of displayed terrain cost
C times positive nominal-wheelbase longitudinal grade squared. It persists
across all intermediate waypoints. The active bout is charged at the goal.

A new bout starts only after travelling one current support diameter
(2*footprint_radius,1.00565m on this map) with absolute nominal-wheelbase grade
at most15°. This is **longitudinal respite while moving**, not flat terrain,
traction recovery, a safe resting shelf, or permission to turn. Front/rear
window support, ordinary route/body/cross-slope limits and actual heading-turn
checks remain binding. The45.05cm demand window and1.00565m respite distance
are different lengths. The15° criterion is an unvalidated characterization
preference, not a newly established physical capability.

There is one reset per uninterrupted qualifying stretch; leaving it re-arms
the reset and clears respite distance only. All gentle travel effort is still
charged. No initial respite credit, waypoint reset, negative reward, or division
by flat travel length is used. Interval polynomials split at signed±tan15°
crossings; the exact partial integral at the respite-distance crossing is
transferred to completed bout cost. Adjacent intervals are grouped only to
reduce evaluation work without changing the integral/reset position.

The existing physical/heading graph is retained. Reverse reachability removes
heading states without a complete goal connection. Forward search preserves
exact labels for heading,completed bout squares,active effort,respite distance,
latch and distance. Dominance compares only identical heading/relief phase
and componentwise no-greater completed squares,active effort and distance.
Priority includes the unfinished bout and is monotone. No rounded states or
best-K pruning. Limits:60,000 TOTAL admitted labels,60,000 expansions,200,000
relaxations and the shared180s build/features/search deadline. Exhaustion is
explicitly censored and returns no certified route; no old-planner fallback.
Baseline v22 and preceding experimental options remain available unchanged.

Pre-search evidence is in `outputs/remote_autonomy/sustained-climb-study/`.
The full-footprint≤15° shelf hypothesis failed to recognize interior contour
relief; it is retained as a failed audit. The directional audit recognized an
interior respite on a physically eligible stored route near(0.878,7.465),
height2.842m. The sketch remains ineligible under unchanged turn checks.

v27 map outcome: INCONCLUSIVE. First planning call lost its report because a
NumPy boolean was not JSON serializable; its planning result was not preserved.
A native-boolean repair, strict serialization regression and pre-report planning
snapshot preceded one identical replay. Both calls are recorded; no physics.
The replay reached the200,000-relaxation work cap in41.89s (build28.64s,
search13.12s):7,053 admitted labels,6,020 expansions,47 dominated labels,
104,857 dominance rejections and88,091 reverse-unreachable transitions skipped.
No new route or optimality result was returned; this does not imply terrain
infeasibility. Stored physically legal routes still have finite scores. Default
unchanged;216 tests pass, repair reviews8.6 each. No cap increase or further
weight tuning was applied. Evidence: `outputs/remote_autonomy/sustained-climb/`.

## Offline guard ablation

`scripts/analyze_guard_ladder.py` separates route connectivity from constrained
objective optimization. It never changes the runtime controller or sends commands.
The earlier v27 heading graph already had a complete goal connection; the empty
result came from bounded objective optimization, not demonstrated infeasibility.

The study freezes one inventory: original terrain candidates and all their
directed connections, plus neutral 1.6 m grid anchors and local connections.
Neutral anchors are not screened by turn staging, slope or obstacle masks.
Finite, observed wheelbase-height/cost support remains a data prerequisite.
At most 512 nodes and 16,770 directed pairs are allowed. All displayed costs,
height fields, and sustained-climb scoring remain unchanged.

Cumulative stages are finite terrain only; oriented obstacle clearance; oriented
35-degree body slope; 33-degree uphill/32-degree cross slope; existing edge
contracts (including circular support, clearance, exclusions and observation
requirements); and initial/join heading turns. Original-only candidates are a
separate sampling control, not another safety requirement.

Each stage saves a complete reverse-reachability witness before optional cheap
minimum-effort and minimum-length searches. Their candidates are rescored using
the exact sustained objective. Only the strictest successful stage runs the
history-aware improvement search, capped at 200,000 extensions/60,000 labels.
A complete incumbent survives censorship. Neither a cheap witness nor a capped
incumbent is called optimal. Total planning computation is limited to 900 seconds;
validation/report export can finish afterward. Finite-graph optimality, static
guard admission, calibrated reachability, and physical mission completion are
separate claims.

Reproduce with a fresh output directory:

```bash
/home/susan/miniconda3/envs/env_isaacsim/bin/python scripts/analyze_guard_ladder.py \
  --output outputs/remote_autonomy/guard-ablation-new
/home/susan/miniconda3/envs/env_isaacsim/bin/python scripts/plot_guard_ladder.py \
  outputs/remote_autonomy/guard-ablation-new
```

Recorded batch `outputs/remote_autonomy/guard-ablation/`: all six stages found
complete witnesses in 155.2 seconds total (434 nodes, 8,908 proposed links).
The stage-4 witness scored 16.05 over 48.55 m and failed 14 turning checks;
no other static guard group rejected it. Adding turns produced a full-static-
guard witness scoring 91.90 over 75.48 m. The history-aware optimizer exhausted
200,000 extensions with no improvement, preserving that complete witness.
These are witness comparisons, not certified optimal guard-cost differences.

The original-candidate control also passes all static guards: score 92.58 over
28.64 m. Thus the expanded route buys only 0.73% score improvement for 164% more
travel. Both are reported; the shorter alternative is the practical comparison,
not silently substituted as the objective winner. Turn staging excludes the
reported low-score contour route, but does not establish that every neighboring
contour is impossible. Route ranking still tolerates excessive detours. No
physics, calibrated reachability, runtime promotion, or mission success follows
from this study. Verification: 223 tests; controls review 8.5, practicality 8.6.
The next search-engine question is avoiding repeated work on transitions already
proven unable to reach the goal, under the same objective and resource limits.
