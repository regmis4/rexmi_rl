# Retired northern-route tracking trials — 25 October 2026

**Status: retired at the user’s direction.** The user requested the previously validated
blue western corridor, not further adaptation to the newly selected northern route.
The active executor has been restored to its pre-trial v2 tracking behavior.
The third trial (`mission-OtTCTjdx`, v5) was interrupted for this change; it is
not a completed mission. Its [audit](crater_full_loop_2026-10-25/tracking_v3_interrupted/audit.json)
and trace are retained. The full v5 CPU suite passed 1,364 tests in 91.27 s.
The description below records the abandoned experiment, not active behavior.

This follows the [40° descent and dashboard correction](crater_full_loop_2026-10-25.md).
That revision descended successfully but failed its full 900 s mission budget:
134 normal turns completed and restarted within the same control tick, without
issuing forward input between them. Ascent spent 571.36 s turning and only
66.32 s in normal driving. The user authorized correcting tracking while
preserving the accepted cost model and unrestricted motion.

## Retired correction

The ascent tracker now aims **2.0 m** along the selected route instead of 0.35 m.
The shorter target was inside the scale of the robot and the translation caused
by turning on slopes; small grid bends could repeatedly move its desired heading.
The target still lies on the original route. Local projection, branch matching,
progress credit and endpoint detection are unchanged.

During normal turns, the tracker updates the desired heading from the current
measured position. Previously it finished turning to a stale world heading,
then immediately discovered that the target had moved relative to the robot
and requested another turn. Completion now uses the same current heading error
as the transition to driving. Normal tracking now steers while advancing:
forward speed is `0.4 * max(0, cos(heading_error))` m/s, and yaw command is the
heading error clipped to the existing ±0.35 rad/s range. A target behind the
robot uses yaw-only alignment until it comes ahead. Small corrections no longer
cut climbing drive to zero. Existing feedback recovery still owns its discrete
reverse/turn/push actions.

No terrain admission, motion veto, cost weight, route simplification, compass
preference or recovery action was added. The existing turn-response deadlines,
reverse escalation, terrain-ranked recovery headings, push/release behavior,
operator controls and 900 s mission budget remain. Descent is unchanged.
The protocol identifier is `saved-slope-path-v5-continuous-tracking`.

## Validation

- 57 focused tests passed, including lateral drift during normal turning,
  maintaining drive through small heading corrections, and preserving local
  progress when a future route branch is nearby.
- Full suite and full physical mission results will be recorded below.

The first tracking attempt (`mission-UFW7Klx3`, protocol v3) fixed the immediate
turn restart bug but retained yaw-only corrections for small heading errors.
It produced **zero immediate turn restarts**, progressed beyond the previous
bottleneck to Y≈6.72 m, then rolled backward at 197.84 s and remained overturned.
The run was interrupted at 269.32 s after that physical failure; no completed
mission summary exists. It is not a successful full-loop result. The trace
showed intermittent removal of forward drive during modest heading corrections,
motivating continuous steering in v4.

That first attempt's descent trace, exact frozen-map hash, selected route and
predicted route metrics match the preceding baseline. The change affected
execution only. [Audit and trace](crater_full_loop_2026-10-25/tracking_v1_failed/audit.json),
[baseline comparison](crater_full_loop_2026-10-25/tracking_v1_failed/baseline_comparison.json).
The v3 full CPU suite passed 1,363 tests in 90.80 s.

The next attempt (`mission-obidjVWx`, protocol v4) combined continuous steering
with a 0.8 m target. It again reached the second face, slid back and repeated
the approach. It was interrupted at 399.78 s; it did not complete the mission.
It had zero immediate turn restarts and no inversion (`minimum up_z=0.270`).
The ascent spent 35.16 s in normal turns, 79.48 s driving, 86.84 s releasing and
97.90 s recovering. [Audit](crater_full_loop_2026-10-25/tracking_v2_failed/audit.json),
[identical descent/map/route comparison](crater_full_loop_2026-10-25/tracking_v2_failed/baseline_comparison.json).
The full suite passed 1,364 tests in 90.99 s.

At the observed second-face pose `(0.49, 6.62)`, the frozen wheelbase-gradient
model predicts about **41.83°** along the 0.8 m target heading, **36.57°** along
a 1.5 m target heading and **32.11°** toward the 2.0 m target. The longer target
anticipates the route's upper contour instead of pointing almost straight up
the local face. This is a local directional comparison, not a prediction of
physical success. At the first face `(0.3, 5.7)`, all three headings remain
around 42°; the tuning does not remove that difficult section.

Launch the full mission with the unchanged command:

```bash
./crater_auto_nav.sh
```

Physical validation uses `./crater_auto_nav.sh --headless --no_dashboard`, retaining the
same robot, physics, policy, live mapping and both mission phases.
