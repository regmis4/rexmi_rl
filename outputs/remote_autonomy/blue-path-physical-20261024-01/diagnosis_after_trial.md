# Exact first-refusal diagnosis

The trial stopped at **42.18 simulated seconds after 7.220 m of credited progress along the 17.247 m saved route**. This was a predictive commanded-motion envelope refusal, not an observed traction failure, tip, collision, or completed ascent. The event correctly records `physical_failure=false`; no learned exclusion was added.

The exact refusal map is `run.planning/request-aqt2nj_g.npz`. Reconstructing the command gates with that map and the final telemetry row identifies the rejecting layer unambiguously:

| Gate/envelope | Result | Maximum footprint terrain slope |
| --- | --- | --- |
| Actual current body footprint | Pass | 30.489 degrees |
| Superclass intended stationary yaw, next 0.5 s | Pass | 30.489 degrees |
| Saved-path combined translation/yaw forecast | **Reject: footprint exceeds 35 degrees** | **35.572 degrees** |
| Inherited actual stopping forecast | Pass | 32.686 degrees |
| Anticipation | No rejection; yaw-only command has no commanded translation |

At refusal the robot was at `(6.18724, -3.37723)`, yaw `0.16071 rad`, measured velocity `(0.27600, -0.17015) m/s`, speed `0.32423 m/s`, and body tilt `19.44 degrees`. Proposed command was zero forward velocity and `+0.35 rad/s` yaw. The lookahead target was `(6.48739, -3.17087)` and cross-track error was approximately `0.101 m`.

The saved-path forecast uses `0.5 + speed/0.4 = 1.31059 s`. Continuing the measured southeast velocity across that horizon predicts **0.42494 m of translation**, ending at `(6.54897, -3.60023)`. Its yaw envelope spans `0` to `0.45870 rad` relative to current yaw. The swept rectangle includes exactly one above-limit cell:

- Grid cell `(195, 140)`, center `(7.1, -3.9)`.
- Live-map terrain slope **35.572 degrees**, original planning-map slope **35.541 degrees**.
- Known, fresh, and not marked as an obstacle hazard.
- Surface elevation `2.25369 m`.

This is an existing steep terrain cell, not a newly discovered obstacle or a body-tilt measurement. Its center is approximately `0.894 m` from the nearest saved-path point `(6.7, -3.1)`. The nominal nearby segment runs northeast from `(6.1, -3.3)` to `(6.7, -3.1)`; the measured drift at refusal instead points southeast.

Separating the forecast components further narrows the cause: **the full translation forecast with no additional yaw already includes the offending cell**. Stationary rotation over the entire forecast yaw range remains clear. Therefore this refusal is primarily the projected southward drift of the body envelope near the bend, not the stationary turning prohibition identified in older trials. The shorter inherited actual-stopping forecast travels approximately `0.21247 m` and remains within the 35-degree boundary.

## Smallest next question

Can the follower enter this bend with less southward drift, by aligning earlier on the preceding route segment or reducing translational motion before the heading change, while retaining the same physical limits? Compare these bounded executor trajectories against the exact archived geometry before another trial. If they cannot retain clearance, inspect a local change to the path geometry that provides turning/drift room.

The longer forecast is provisional, not calibrated braking or yaw-response evidence. Its rejection does not establish that the robot would actually touch the offending cell. Equally, the passing shorter stopping envelope does not justify deleting the longer command forecast or raising the terrain limit. Actual turn-plus-drift response is the unresolved quantity.

Machine-readable envelope endpoints, offending cells and gate results are preserved in `diagnosis_after_trial.json`. No execution code was changed and no retry was performed for this diagnosis.
