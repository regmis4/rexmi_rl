# Independent completion audit

**Verified: one autonomous simulated ascent along the saved blue route completed at 102.30 simulated seconds without manual intervention, resets, policy switching, or recovery actions.** The machine-readable audit is `after_trial_review.json`.

## Completion evidence

Final XYZ is `(11.40265, 3.99617, 4.73446)` m, radius `12.08263 m`, and distance to the selected endpoint `(11.1, 4.1)` is `0.31997 m`. Radius 11 m was first crossed at 85.06 s. The full 17.24735 m saved-route progress was reached.

The final contiguous qualifying interval is **101.68–102.30 s**, 32 navigation observations spanning **0.62 s**. Throughout that interval:

- Endpoint distance stays at or below `0.35339 m`, inside the 0.4 m arrival disk.
- Speed stays below `0.11267 m/s`, within the 0.12 m/s settling criterion.
- Absolute yaw rate stays below `0.11128 rad/s`, within the 0.12 rad/s criterion.
- Upright component stays above `0.96733`, exceeding the completion requirement.

This verifies the configured upright low-speed settling criterion. It does not mean the policy became motionless or establish long-duration station keeping.

## Continuity and provenance

- All **5,116 navigation rows** have consecutive steps and 0.02 s timestamps from 0 through 102.30 s.
- All **20,460 physics rows** have consecutive sample indices, 0.005 s timestamps, and the expected four substeps for each of **5,115 executed control steps**.
- All 5,115 logged policy-action indices are consecutive. The extra navigation row is the terminal observation, not a missing physical step.
- Physics and events remain in episode 0 with the rocky-slope policy throughout. Capture has a normal end marker, and final physics XYZ exactly matches final navigation XYZ.
- All **34 archived source hashes** match provenance. The archived saved-route result hash also matches.
- Events contain no guard refusal, emergency, recovery action, or negative forward command. Summary records no manual intervention and no policy switches.

## Cross-track interpretation

The reported maximum `1.00158 m` occurred at **95.32 s**, at `(12.09926, 4.16823)`, after the first endpoint-disk arrival at 86.36 s. It equals distance from the final endpoint: the robot overshot outside, turned, and returned. Maximum cross-track error **before first endpoint arrival was 0.21103 m**. Thus the one-metre peak is endpoint overshoot, not a one-metre departure during the main climb.

## Claims supported and not supported

This establishes a successful autonomous saved-path execution on this seed-42 crater using the unchanged policy and the full-crater simulator development survey. Motion guardrails were explicitly disabled. The route preferences were fitted using same-crater evidence; this is not independent generalization, repeatability, onboard-perception, guarded-autonomy, or hardware qualification.

Feedback recovery was implemented but **never exercised**. No claim about reverse/push recovery effectiveness follows from this run. Normal heading tracking completed the maneuver sequence.

The legacy CSV contains `climb_failures=0`, but that classifier is disabled in this mode. The authoritative summary correctly reports `climb_failures=null` and classification unavailable. Do not translate the legacy zero into a validated claim of zero physical failures.

No execution code was changed and no additional run was performed for this audit. Reports should retain the original fitted magma/LogNorm map and accepted route colors, with measured motion overlaid.
