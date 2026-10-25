# Straight motion: arrival heading and footprint evidence

Ascent remains unproven. These changes use measured trial failures, with no
policy, physics, observation-layout, slope-limit or speed-ceiling changes.
The earlier reviewer usage failure was transient. The existing reviewer resumed
on September 24 and approved the observational departure diagnostic at 9/10,
after independently running 197 focused tests. Approval is scoped in
`ascent_review_gates.json`; it does not authorize an adaptive continuation.

## Trial 15 and exact diagnostic replay

Trial 15 selected the checked shelf shortcut, removing 7.954 m of detour, but
ended in HOLD at 207.96 simulated seconds. It logged twelve recovery maneuvers,
two physical approach failures and seven censored approaches. RELEASE was never
entered. Peak active tilt was 36.61°, peak measured speed 0.479 m/s, controller
p95 0.590 ms and average mapping cost 51.75 ms. Commands remained within 0.40 m/s.
The command/evidence audit passes; this is not ascent success or calibration.

Twenty gentle-ground corrections occurred at about 9.2° of heading error. Every
current straight ray intersected the inner 75% of the unchanged arrival disk.
Nineteen occurred during reverse. Fixed angular precision was therefore stopping
motion that already pointed into the required destination region. Fixtures retain
the source CSV hash, pose, heading, direction, destination and radius.

A separate 62-second observational run reproduced all 3,100 trial-15 poses and
commands exactly, including the footprint rejection at 52.60 s. This diagnostic
is not a completed mission trial. At rejection the circular obstacle check
required 0.603 m clearance and measured 0.566 m. The configured rectangle, which
includes the body reserve, passed every sampled pose and the continuous swept
rectangle reference. Terrain support and directional grade checks were clear.

![Contemporaneous footprint rejection](../../outputs/ascent_guard_diagnostics/corridor01/guard-00.png)

## V14: enabled arrival-heading acceptance

On gentle terrain, the executor may retain a heading whose straight body-axis
ray reaches the inner 75% of the existing arrival disk. Reverse changes the ray
direction without changing body orientation. This supplements the existing
small-angle tolerances during straight admission and continuation. Already
active alignment pulses retain their measured stopping contract: regression
testing found that an earlier ray-based pulse stop can create repeated
turn/coast cycles under policy yaw unwind. That early-stop change was removed.

The endpoint, arrival radius, action identity, direction, deadline and committed
progress evidence remain fixed. Geometry, stopping, freshness, opposing-motion,
posture, speed, stall and settling checks remain binding. A useful ray is neither
arrival nor motion authorization. Steep entry does not gain a heading exemption;
existing zero-yaw steep continuation remains intact. RELEASE resumption keeps
its prior heading contract. Acceptance is logged once per primitive.

Ten new executor regressions cover all recorded corrections, both signs, target
behind, the outer arrival margin, steep entry, opposing motion, geometry denial,
overspeed, pulse termination, settling, stale observations and STOP.

## V14 physical result

Trial 16 ended in HOLD at 139.48 simulated seconds. RELEASE ran 10.76 s,
moving 1.327 m net XY and gaining 0.613 m body height. At 60.90 s a lateral
departure from the original corridor ended the climb despite positive measured
uphill progress. This is a separate execution-contract issue, not proof that
continuing would succeed. Reverse later traveled 1.377 m continuously over
9.10 s. Eight recovery maneuvers and two physical approach failures were logged.
Peak active measured speed was 0.775 m/s, peak tilt 42.17°, controller p95
0.638 ms and average mapping cost 46.96 ms. Commands remained within 0.40 m/s;
the audit passes. No ascent completed.

V14 accepted two useful headings and performed two later gentle corrections.
Its trajectory and timing differ from trial 15, so those full-run counts are
not a controlled causal comparison. All twenty old correction samples pass
the deterministic executor replay. V14's source manifest omitted the new
geometry module. A retrospective snapshot of the unchanged executed arrival
helper and the limitation are preserved in `trial16/dependency_note.json`.
Its original source hash is not a complete dependency hash. Future archives
include the geometry module; trial 16 is not qualification evidence.

![Trial 16 release response](../../outputs/ascent_behavior_tree/trial16/release_response.png)

## V15: integrated oriented straight sweep

`swept_footprint.py` also implements continuous rectangle/cell intersection for
translation, with a conservative union over passive heading uncertainty. Tests
cover between-endpoint and cell-corner hazards, intermediate rotation, unknown
or stale terrain, 35° support, directional grades/exclusions and monotonic escape
from an occupied exclusion. V15 uses it when a circular straight-motion obstacle
margin rejects a path whose configured body footprint remains clear.

Commanded stopping motion, actual drift, braking and RELEASE use the same body
sweep fallback. A continuous 0.30 s heading history supplies observed extrema
and the net trend over the estimated stopping time. Missing history, gaps over
0.04 s or a forecast over a half-turn reject the fallback. The stopping fan
covers both body rotation and centre-path variation with explicit interpolation
reserves. Remaining-path admission checks static geometry; emitted motion must
also pass the live stopping check. Circular ground support, global route
clearance and turn staging remain, as do configured body reserve, terrain,
freshness, directional exclusions, immutable targets and measured settling.
No simulator ground truth is read for these checks.

The lateral corridor bound is intentionally unchanged in this comparison.
Before changing it, characterize whether displaced uphill motion has a complete
checked continuation and a useful reachable staging endpoint. Merely increasing
a numeric tolerance does not establish that the alternate approach will work.

The exact rejected sweep being geometrically clear does not prove collision-free
policy motion. Physical characterization and the outstanding independent review
are still required; no probability or ascent gate follows from these fixtures.

## V15 physical result and next review boundary

Trial 17 ended in HOLD at 125.46 s, with one physical approach failure and eight
censored approaches. Its first 3,055 pose/command samples exactly match trial 16
through 61.08 s, including the 10.76-second RELEASE climb and 60.90 s lateral
departure. The first changed output is at 61.10 s, after that departure. The
oriented fallback was admitted eight times across distinct action/intent pairs;
it does not bypass subsequent geometry rejection. Recovery exhausted eight
maneuvers. Peak active speed was 0.582 m/s and tilt 39.61°. Controller p95 was
0.956 ms; mapping averaged 47.37 ms. The command/evidence audit passes.

Neither trial completed ascent. Three settled movement endpoints in the initial
leg do not count as complete checkpoint or mission success. The changed recovery
trace does not establish a reliability improvement.

The next control revision needs a reviewed continuation contract, not a larger
lateral-distance constant. Proposed scope, **not implemented**:

- Request a continuation evaluation before a useful uphill traversal reaches
  the boundary of its originally checked corridor. Keep local emitted-command
  and stopping guards active while evaluation runs.
- From measured pose and current heading, seek a complete checked continuation
  to reachable gentle staging or a useful observation frontier with onward
  geometry and retreat. Use the observed full-crater map, not a supplied route.
  Do not assume that a flat patch is reachable, or require hillside yaw.
- Require the applicable whole-leg probability gate in qualified mode. Under
  characterization, log unsupported predictions explicitly. A predicted route
  is not physical success, and positive recent progress alone is insufficient.
- Adopt a changed approach atomically with a new immutable request and logged
  prediction, censoring the abandoned target without a false completion. Review
  whether the departure is a planning tolerance event or demonstrated failure
  of an approach before applying terrain exclusions.
- Carry the physical episode's deadline, progress budget and evidence through
  handoff; repeated replans must not renew patience, duplicate outcomes, walk
  the destination indefinitely or erase exclusions. Keep the mission target.
- If there is no admissible continuation, use an explicitly checked bounded
  retreat. Preserve STOP, actual overspeed, hard-tip/drop protections and the
  0.40 m/s command ceiling throughout.

The post-abort loss of height is observed, but a causal claim that braking alone
caused it requires a matched counterfactual. The trajectory does not prove that
continuing RELEASE would have reached a shelf or the exterior. Independent
review is still unavailable because the reviewer reached its usage limit; no
usage-reset credit has been consumed.
