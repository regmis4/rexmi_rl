# Ascent implementation and physical validation status

This is the preserved **local-only, cold-start baseline**. The information
assumption has since changed: return-ascent tests now start with a labeled
full-crater development survey. See [full-survey validation](ascent_full_survey_status.md)
for those results and the Isaac Sim overlay check. The numbers below have not
been relabeled as full-survey trials.

The ascent framework is implemented, but the ascent acceptance gate has **not
passed**. No calibrated model currently qualifies autonomous checkpoint
attempts. Descent, round trips, and high-speed/performance optimization have
not started.

## Preserved evidence

There are 20 completed characterization runs with the new ascent controller,
one baseline-controller run, and one recorded simulator infrastructure failure.
None of the completed ascent runs reached and settled at the exterior goal.
The controller changed between development trials, so these runs are not a
frozen-controller qualification series.

All runs used observed terrain for planning. The rocky-slope weights, robot
physics, observation layout, and shared command-injection pipeline were kept.
The original checkpoint controller and observed-map implementation match the
saved baseline copies byte for byte.

The most recent run, `trial21`, used the normal 25-degree uphill and side-slope
limits, seed 42, floor spawn `(0, 0, 1.15)`, and goal `(13, 0)`.

| Measurement | Result |
| --- | --- |
| Mission result | HOLD after 221.44 simulated seconds |
| Final position | `(5.058, -1.912)` m |
| Reached-and-settled checkpoint attempts | 2 |
| Failed reachability attempts / censored attempts | 4 / 10 |
| Confirmed physical-failure exclusions | 0 |
| Local recovery maneuvers | 6 |
| Peak commanded forward/reverse speed | 0.400 m/s |
| Peak active measured speed | 0.479 m/s |
| Measured slope overspeed above 0.45 m/s | 0.38 simulated seconds |
| Peak active body tilt | 30.10 degrees |
| Measured backslide | 0.371 m |
| Controller mean / p95 wall latency | 1.38 / 3.06 ms |
| Map rebuild mean wall latency | 39.99 ms |

Active statistics exclude the first three seconds of spawn settling. The raw
summary also preserves the larger startup transient. A command ceiling does
not establish a physical speed ceiling while slipping.

This run completed a retreat to the floor and resumed its original mission,
but repeated geometry rejection near the next boulder exhausted recovery.
Earlier 35-degree characterization runs reached farther up the wall and
explored alternative approaches, but also showed lateral drift during hillside
turns and unsuccessful regroups. Those runs do not establish demonstrated
reliability at the relaxed planning limits.

The final directional-exclusion check for partial-cliff escapes was added
after `trial21` started. It has regression coverage; the exact earlier source
used by that physical run is retained in its `run.sources` directory. The final
working tree is not represented as having passed a physical qualification run.

## Software verification and model status

All **268 regression tests pass**, including 49 ascent-specific checks.
They cover complete-leg feature extraction, supported shelves, alternative
approaches, turn support, settled arrival, probability thresholds and updates,
approach-specific evidence, directional exclusions, duplicate prevention,
checked exclusion exit, STOP precedence, and bounded command behavior.
They are not substitutes for physical mission or braking validation.

The diagnostic logistic model was fitted using 12 labeled training attempts,
9 calibration attempts, and 12 held-out attempts from separate runs. Only four
held-out attempts fall within its characterized feature range. The artifact is
explicitly unqualified because of insufficient evidence and mixed controller
versions, among other reported reasons. Its reliability bands, sample counts,
uncertainty, and Brier scores are in the machine-readable report; unsupported
predictions cannot authorize autonomy.

The remaining ascent work is to make hillside turns and recovery near newly
observed obstacles physically executable, then freeze the controller, gather
independent characterization runs, validate the predictor and its online
update, and achieve three consecutive qualified floor-to-exterior successes.
No descent gate can be credited before that sequence succeeds.

- [Run guide and contracts](../ascent_navigation.md)
- [All preserved trial metrics](ascent_v2_summary.json)
- [Latest observed route and terrain](../../outputs/ascent_v2/trial21/observed_route.png)
- [Latest raw summary](../../outputs/ascent_v2/trial21/run.summary.json)
