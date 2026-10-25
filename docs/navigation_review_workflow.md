# Navigation review workflow

Four agent roles; review scores are AI assessments, not human certification.

| Role | Responsibility | Progression authority |
| --- | --- | --- |
| Executor | Implement the authorized change, test, run, preserve evidence | Execute only the diagnosis agent's approved scope |
| Diagnosis | Compare current behavior with successful manual logs; identify the first causal failure; propose the smallest useful change | Authorize execution after both reviews pass |
| Controls reviewer | Check control ownership, state transitions, geometry, deadlines, evidence classification, STOP and failure cases | Require an honest score of at least 8/10 for exact implementation before simulation |
| Practicality reviewer | Challenge complexity, resource use, architecture and mission relevance | Approve, simplify, defer or stop the proposed work list |

The practicality review asks:

1. What observable mission failure does this work resolve?
2. What do the user's successful teleop logs actually demonstrate, and what remains unproven?
3. Is this a conventional engineering solution with a clear owner, bounded state and measurable behavior? If not, why is the departure justified?
4. Can an existing component solve it with fewer rules, searches or interfaces?
5. Does the detailed change fit the overall plan: reliable ascent, descent/round trips, then measured optimization?
6. What will this experiment distinguish? What result would stop or change the approach?
7. Are compute, wall time, usage and repeated reviews proportional to the information gained?
8. Are we treating planner rejection as evidence that the policy cannot perform the maneuver?

Review a concrete proposal and work list, not every control tick. Reviewers do not steer the robot during autonomous trials. One hypothesis and one bounded run at a time; no mid-run patches or automatic retries. Preserve failed trials and explain exclusions, uncertainty and deliberate diagnostic truncation.

Before each simulation, record current account usage and approvals in the run directory. The current user ceiling is **70% weekly used**; do not redeem reset credits. Stop earlier when the experiment no longer provides useful evidence. A stop is not mission completion or a new physical failure.

Current example: `outputs/autonomy_resume/route-diversity-oGYz9LxY` was approved by controls (8.4), practicality (8.2), then diagnosis. It reached the center autonomously but failed to exit. Practicality recommended ending repeated recovery after a stale planning-purpose bug was identified; diagnosis authorized truncation at 660.86 simulated seconds. See that run's `review.json`, `first_ascent_audit.json` and `diagnostic_summary.json`.

## 2026-10-23 checkpoint inventory repair

Remote v22 preserves original validated survey candidate locations, rechecks
them against live support/staging, then fills remaining capacity with live
candidates. All links, turns and learned exclusions use the current map.
The graph remains capped at 130 nodes and 16,770 directed checks; route
objectives and physical limits are unchanged. Planning archives include the
exact current map and immutable six-field survey candidate source.

The captured 3-second live map demonstrated a valid lower-cost route had been
lost through a small representative slope-ranking change. The repaired planner
independently restored it (25.60° peak uphill, 0.79 m above 25°, compared with
32.78° and 2.15 m). This is a map-only result, not ascent qualification.
The preceding v21 physical trial ended in HOLD at 62.12 seconds; no waypoint
was reached. Current validation: 189 tests; controls/practicality 8.6/10 each.
Evidence: `outputs/remote_autonomy/reachable-height-objective/decision_report.json`.
