# Unguarded blue-path simulation: practicality review

**Score: 8.2/10. Approve one diagnostic seed-42 trial after focused tests and controls review.** This is an AI assessment, not physical qualification. No simulation was launched by this reviewer.

## Authorization and mission relevance

The user explicitly authorized ignoring all motion guardrails in simulation and requested measured-progress reverse/turn/push recovery. That instruction supersedes the preceding guarded trial's slope, footprint, tilt, speed and cross-track admission requirements. This review does not require reinstating them. STOP/manual ownership, invalid-observation handling and the finite experiment budget remain execution controls.

The preceding blue-path run reached 7.22 m of route progress before stopping at 42.18 simulated seconds. Its combined turn/drift forecast touched one 35.572-degree cell; it did not establish a failed physical climb. A single trial without those vetoes can therefore reveal actual policy behavior beyond the prior refusal. The same saved fitted planner polyline and policy/physics preserve a useful comparison, although added recovery means this is not a clean single-variable ablation.

## Scope and complexity

The separate experimental controller is a reasonable containment boundary. It preserves the saved route and reuses existing delivery and logging rather than modifying the general controller. Recovery has three explicit actions: bounded yaw, a five-second push, and a reverse bounded by 0.5 m or four seconds. Five candidate headings use terrain and route preference plus a short failed-push history. These are conventional local feedback heuristics, not a learned success model or a global planner.

The eight-second plateau trigger, action response windows, capped 64-event recovery history and 900-simulation-second ceiling make one trial proportionate. No new search, weight sweep, extra recovery modes, automatic retry or live editing should be added during it. Retain the requested 0.40 m/s forward/reverse and 0.35 rad/s yaw commands; actual motion is measured rather than vetoed.

## Evidence limits to preserve

- Recovery score is route projection minus cross-track error. Reducing lateral error or regaining previously lost ground can count as useful local response without gaining new mission distance. Report furthest route progress, current route position, backslide and total travel separately from action gains.
- Normal yaw progress renews the eight-second response timer. Thus angular activity alone can postpone translational recovery; the overall run remains bounded. Report persistent turn cycles plainly rather than counting them as ascent progress.
- Heading preference samples one point 0.8 m ahead. It ranks exploratory attempts and does not establish complete-corridor support or predict traction. No such guarantee is needed for this expressly unguarded diagnostic.
- The policy can drift or fall while nonzero or zero commands are active. Record the measured result and end classification; do not label removed guard checks as physical failures or claim the experiment qualifies deployment.
- Recovery memory discourages unsuccessful nearby pushes but cannot guarantee a new productive maneuver. If a cycle repeats, preserve its event history and explain the lack of net progress in the report.

## Prelaunch verification and reporting

Focused tests should establish that the experimental tick does not invoke inherited geometry, tilt, overspeed or cross-track vetoes; STOP/manual still own commands; invalid observations stop; the 900-second autonomous budget stops; and plateau, turn/push/reverse transitions, failed-attempt memory and upright settled endpoint completion behave as documented. Tests must exercise actual observed progress and stalled responses, not only helper return values. No additional approval of the already authorized guardrail removal is needed.

Archive exact inputs, source hashes, command flags, policy identity, initial state, account usage and reviews. One run, no reset credits, no automatic retry or live changes. Success means the selected endpoint is reached upright and settled without intervention; report this as one simulated outcome.

Every user-facing report must retain the **original fitted cost-map field with magma and its original LogNorm scaling**, the accepted route colors and stable plot extent. Overlay actual motion and recovery events on that same background. Do not replace it with a newly normalized live map, linear scale or alternate palette; any live-map diagnostic must be additional and explicitly labeled. Include the planned blue route and actual trace so drift and recovery can be judged directly.

Reviewed: `source/rexmi_rl/nav/experimental_path_control.py`, CLI and runner integration, the inherited saved-path projection, fitted slope-cost implementation, and the preceding trial's `diagnostic_summary.json`. Tests were being added at review time. No implementation changes were made by this reviewer.
