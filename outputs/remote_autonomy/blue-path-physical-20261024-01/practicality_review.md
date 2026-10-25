# Saved blue path: practicality review

Assessment: **8.3/10; practicality approves one diagnostic trial.** Controls approval and successful final verification remain launch prerequisites. This is an AI review, not physical qualification. No simulation was launched by this reviewer.

The user accepted the fitted slope-cost comparison and requested execution of its blue any-exit polyline. The proposed trial directly tests that corridor through the unchanged policy. The route is planner-generated using same-crater fitted preferences; it is not a manual command replay, and it is not independent validation of the preference model.

The implementation is proportionate: a fixed polyline, bounded local projection, 0.35 m lookahead, persistent yaw, one forward re-initiation per credited progress increment, and a shared 35 s no-progress stop. It inherits the command/output pipeline rather than creating another planner. First-refusal map capture and disabling replanning make an unsuccessful run useful for diagnosis. No further cost fitting, route search, recovery state machine or batch is justified before this trial.

## Prelaunch conditions

- **Resolved:** the combined translated-yaw sweep now calls existing `checked_turn.physical_reason`, preserving full-footprint cross-slope checking throughout yaw. The 32-degree boundary remains; no stage, local-minimum or `MAX_CORRECTION` restriction is reintroduced. Source inspection and the dedicated side-slope regression confirm the intended change.
- The executor reports all 160 focused tests passed after the cross-slope correction. Route/profile metadata and resume assertions were subsequently repaired; the final 160-test rerun is pending at this review update. Source inspection confirms `mission_route`, `route_profile` and resume metadata restoration. The final rerun must pass before launch.
- Archive the exact accepted result hash, launch configuration, source snapshot, usage and controls approval. Preserve the failed stationary-corner audit: straight-segment validity is not proof of turn executability.

## One-trial scope and interpretation

Run one seed-42, north-facing baseline ascent with the unchanged rocky-slope policy/physics, command cap 0.40 m/s and inherited 900 simulated-second mission ceiling. No live edits, alternate routes, automatic retries or reset credits. Stop at the first terminal refusal or completion; preserve the planned corridor and actual trace.

Measure route progress, cross-track error, commanded turns, active-zero climbing, first refusal and settled exterior arrival. If a stationary-corner violation becomes the first runtime refusal, the run establishes the follower/geometry conflict at that location; it does not establish that the policy cannot climb the corridor. If repeated short heading turns consume the no-progress budget, adjust execution only after inspecting that evidence. Successful arrival establishes this trial's execution, not generalization across crater geometry or repeated starts.

The route's short lookahead and known problematic exact corners make a refusal plausible, but this does not make the single trial wasteful: the exact observed state at refusal is the missing evidence. Repeated trials without a changed causal hypothesis would be wasteful.

Reviewed sources: `source/rexmi_rl/nav/saved_path_control.py`, its CLI/runner integration, `swept_footprint.py`, `remote_control.py`, `planning_snapshot.py`, the accepted-fit/execution sections of `docs/ascent_navigation.md`, and relevant side-slope/resume assertions in `tests/test_saved_path_control.py`. Test outcomes above are attributed to the executor; this read-only implementation review did not rerun the suite.
