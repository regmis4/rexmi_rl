# Slip tracking and uphill route selection

Normal travel stays at 0.40 m/s on rocky-slope. Clear flat-ground turns retain
ordinary steering; this change adds recovery when measured progress fails.
Existing launch commands still work. Keep `--map_retention full` to compare
return routes across the entire surveyed terrain.

**Ascent update:** failed climbs now use a larger route exclusion and a committed
downhill regroup point. Along-travel and side-slope limits also apply to planning
and steering. See [the current ascent behavior](ascent_retreat.md); the short
local recovery below remains available for other failures.

## What is measured

The tracker compares forward demand with world-frame displacement and measured
body velocity. Body movement, rather than wheel commands, establishes progress.
These are slip/stall indicators, not measurements of tire friction or wheel slip ratio.

| Evidence | Trigger |
| --- | --- |
| Backsliding | At least 1 s of forward demand, with mean forward velocity or measured progress below −0.10 m/s |
| Sideways slip | At least 1.5 s; mean lateral speed above 0.12 m/s and route error grows by more than 0.20 m |
| Forward stall | 2 s above 0.25 m/s commanded, with progress or mean forward velocity below 0.05 m/s |
| Slow-turn stall | 3 s of 0.10–0.25 m/s forward demand; progress threshold scales from 0.02 to 0.0375 m/s |
| Replan loop | Three motion/route rejections within 15 s and 0.75 m; handled separately from physical slip |

Manual operation, intentional stops, pure alignment turns, reversing and policy
handoffs do not count as forward slip. All timers use simulation time. A velocity
reading without matching displacement does not count as successful travel.

## Failed approaches and no-go zones

A confirmed physical failure excludes that heading and its neighbouring 45°
heading bins around the failed patch immediately. Other approach directions
remain available. A second independent confirmed episode in overlapping cells
marks a **learned no-go zone**, shown in **dark red**. This remains distinct from
red geometric obstacles and orange terrain above 35°.

The patch radius is the greater of 0.35 m or the robot footprint radius plus
0.10 m. A single episode counts once. A mere planner rejection does not label
the terrain slippery or create a learned no-go zone.

These exclusions persist for the mission, including when rolling retention
discards old raw terrain measurements. The exported map includes `no_go`,
`approach_blocked` and `difficulty_episodes`. No cross-run map is loaded automatically.

## Recovery

The robot brakes, then selects a checked reverse, turn or exit arc. It can use
the proven 0.40 m/s reverse command where stopping room permits; a tighter
corridor may require 0.20 m/s. Movement is limited to 3 s, 0.85 m or roughly
37° of turning before stopping and reassessing. An ineffective motion is not
repeated at the same position and heading. Recovery has a 15 s total budget and
at most three movement attempts in the same 1.2 m area. Briefly resuming route
following does not clear that history or restart the budget. Operator Resume
permits a new recovery attempt while preserving the learned terrain exclusions.

Recovery can begin after speed stays below 0.10 m/s for 0.2 s, or after net
displacement over 0.3 s falls below 0.08 m/s while instantaneous speed stays
below 0.20 m/s. The second check accommodates gait oscillation; it does not
declare the robot stopped. Actual velocity still determines the drift/stopping
corridor for every proposed motion. Sustained drift does not pass this check.
If neither condition is reached within 3 s, recovery holds for assistance.
If a settled gait still has a transient blocked drift corridor, recovery keeps
braking and checks again every 0.2 s within that same 3 s window. A single
rejected sample does not immediately end recovery; clearance rules still apply.

The mission destination is preserved. The local checkpoint and larger route
are replaced after changing the approach. A robot already inside a learned
no-go patch may leave it only without going deeper or re-entering; that exception
does not relax unknown-ground, obstacle, physical footprint or 35° exclusions.
If no checked escape exists, the controller holds for assistance. STOP and manual
takeover remain immediate.

## Route comparison

For an observed uphill route above 15°, the planner searches with balanced,
shorter-route and gentler-climb preferences. The gentler search also discourages
reusing the same steep section. Identical results are deduplicated.
If the final destination is not connected yet, these searches compare approaches
to the same observed frontier endpoint; they do not invent a connection through
unknown ground.

Every candidate is evaluated on travel cost, peak uphill grade, side slope,
steep terrain under the full footprint, turning demand and recorded difficulty.
The selected candidate must stay within 2× the shortest candidate, or 6 m
extra for short routes. A route within the preferred uphill grade wins among
these candidates. This bounds detours; it is a planning heuristic, not a
claim about the physically optimal exit. The existing 15% threshold still limits
switching between valid cached routes. Invalid or failed routes are discarded.

The planner now checks excluded directions at both ends of an edge, including
the robot's actual starting position. Previously it could propose a route out of
a failed cell that the follower immediately rejected.

## Dashboard and records

The right-hand status panel shows motion evidence, actual progress, lateral
velocity and the route choice. The traversal view shows failed approaches;
learned no-go zones appear in every terrain view and in Isaac Sim.

CSV records include motion status, forward and lateral speed, net progress,
no-go cell count and candidate count. The `.events.jsonl` file records confirmed
episodes, repeated command rejection and route comparisons with distances and
slope profiles. Escape decisions record each considered command, its score or
exact rejection reason, and the selected motion. The summary retains motion
events, escape decisions and final route options.

## Recorded results

169 regression checks pass. They cover motion evidence, failure memory,
planner/follower agreement, checked escape, manual STOP, repeated recovery
budgets and uphill alternatives.

The sensor-fed floor trial reached its destination in 28.8 s without intervention
or policy switching. It did not trigger a slip episode. An early uphill trial
detected real backward motion despite forward commands. The last uphill trial
resumed after a transient drift rejection and attempted heading changes, but
ended in HOLD at 79.62 s after repeated mapped footprint conflicts. It did not
reach the rim. The final change preserving recovery attempts and the time budget
across repeated episodes has controller-test coverage only.

Policy, friction, terrain physics and normal speed were unchanged. No automatic
resets were used to obtain progress. The saved full-survey comparison found
different routes, but both still required roughly 31° peak climbs; it does not
prove a gentle physical exit is available.

Detailed results are recorded in [the recovery report](validation/slip_recovery_summary.json).
Full crater navigation still requires three complete traversals without
intervention, policy switching or automatic resets before it is called validated.
