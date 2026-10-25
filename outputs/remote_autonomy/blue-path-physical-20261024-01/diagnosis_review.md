# Diagnosis review: inspected blue-route execution

**Score: 8.3/10. Authorize exactly one seed-42, north-facing, headless characterization run of the recorded launch.** Controls and practicality each report 8.3/10 approval. The executor reports all 160 focused tests passed on the final implementation; this reviewer inspected source and performed the static route audit, not a simulator run or an independent repeat of that test suite.

## Reviewed implementation

The saved controller fixes the accepted `routes.any_exit` polyline and checks its planning-map checksum and terrain identity. Runner integration records and rechecks the result hash before archiving its bytes. It cannot substitute the fitted manual-endpoint route or silently replan after a refusal.

The follower uses bounded local projection, 0.35 m geometric lookahead, persistent measured-heading turns, active-zero climbing, and one forward re-initiation per credited progress increment. The shared 35-second no-progress deadline survives target/button changes. It stops on the first recovery request, preserving the original refusal rather than introducing another route planner.

Removing minimum/staging restrictions does not remove the final physical command gate. The subclass explicitly checks combined measured translation and requested/measured yaw with `checked_turn.physical_reason`, including full-swept-footprint cross-slope. Inherited STOP/manual, tip, overspeed, measured drift, anticipation, and 900-second mission limits remain. Arrival requires proximity to the selected endpoint, supported gentle exterior terrain, contiguous settling, and the final output gate.

The first terminal refusal archives the exact observed pose and planning-map snapshot. This directly addresses the previous trials' inability to reconstruct their first live-map rejection from a later archived map.

## Independent static audit

Audited the accepted result against `outputs/remote_autonomy/reachable-height-objective/stable_capture/request-d55yep2q.npz`, using the existing oriented footprint/grade checks without freshness or turn-location restrictions. The route has 27 points and length 17.247348980724645 m.

- All 26 straight segments pass the oriented physical footprint and commanded-grade checks.
- Endpoint `(11.1, 4.1)` passes the existing supported gentle exterior-region check.
- Exact stationary corner sweeps still encounter physical limits:

| Zero-based vertex | Position | Requested turn | Physical refusal |
| --- | --- | --- | --- |
| 12 | `(6.7, -3.1)` | 53.13 degrees | Cross-slope limit |
| 14 | `(7.3, -2.1)` | 45.00 degrees | Footprint terrain exceeds 35 degrees |
| 15 | `(7.3, -1.9)` | -18.43 degrees | Cross-slope limit |
| 17 | `(7.7, -1.1)` | 45.00 degrees | Footprint terrain exceeds 35 degrees |
| 19 | `(7.9, -0.3)` | 63.43 degrees | Cross-slope limit |

These corner checks are diagnostic, not a claim that lookahead will execute stationary rotations at those exact points. Actual drift and lookahead can change the swept geometry. Conversely, straight-segment admission does not prove turn executability or traction.

## Scope, stop criteria, and interpretation

Use the unchanged rocky-slope checkpoint, physics, seed 42, north-facing initial yaw, 0.40 m/s command ceiling, configured 32-degree side-slope limit, and 900 simulated-second ceiling. Freeze the recorded launch and source; no live edits, automatic retries, alternate routes, or reset credits.

Stop at settled supported endpoint completion or the first terminal refusal: physical output gate, greater than 1 m cross-track deviation, 35 seconds without 0.1 m new route progress, bounded angular-progress failure, tip/terminal overspeed, mission deadline, or operator takeover. Preserve the full trace and first-refusal snapshot. Do not continue an identical stopped experiment merely to reach the mission deadline.

This run distinguishes whether the accepted fitted corridor can be executed with a simple geometric follower and unrestricted turn staging while actual physical limits remain active. A refusal at a known corner identifies a follower/geometry conflict, not proof that the policy cannot traverse the broader corridor. Success establishes one saved-path execution; it is not independent validation of the fitted preference model or generalization to other terrain.

## Reviewed artifact hashes

- `source/rexmi_rl/nav/saved_path_control.py`: `2b7205016be8e23d0cca3d9ae39abb86d64048050320610bc47c252001912077`
- `outputs/remote_autonomy/blue-path-physical-20261024-01/launch.sh`: `63d05bf24d307fc5a9ecd4585831f454feb0ed5cce654cc3868c69dba1df3103`
- Accepted `outputs/remote_autonomy/slope-fit-20261024-04/result.json`: `3a56ca3ed18bff83bbe24d6457816fefa37e7e3e1baa52d293dc7802e76281b3`

This is an agent engineering review authorizing a bounded experiment, not physical qualification or certification.
