# Diagnosis review: explicitly unguarded saved-path simulation

**Score: 8.2/10. Authorize exactly one recorded seed-42, north-facing, headless trial.** Controls and practicality are finalized at 8.2/10, and the executor reports the final 169 focused checks passed. The bound source and launch hashes remain unchanged. This reviewer inspected the concrete source/launch and prior telemetry; no simulation or code change was performed.

## Authorization and experiment

The user explicitly replaced the earlier simulation motion constraints: no predicted geometry, terrain inclination, tip, overspeed or cross-track motion vetoes. The experimental tick honors that request rather than invoking the inherited guarded tick. STOP/manual ownership, nonfinite observation handling and the 900-second experiment budget remain. Upright settled endpoint arrival is a completion criterion, not a motion veto.

The accepted blue route, policy, physics, seed and initial north-facing yaw are retained. The launch explicitly selects `--ascent_unguarded` and uses fresh output paths. The saved-route/map identity checks remain provenance checks; they do not reject motion along the path. No new failure exclusions are created by local recovery.

## Why this run is informative

The previous trial stopped at 42.18 simulated seconds because a provisional 1.31-second motion forecast included one 35.572-degree cell. Its current footprint, stationary yaw sweep and shorter actual-stopping forecast passed. There was no demonstrated physical climb failure at that stop. Over its last eight seconds the robot displaced 1.13 m horizontally and gained 0.59 m body height; the final 1.8-second turn changed heading by 0.183 rad. Thus continuing measured response beyond that refusal is a concrete experiment.

This is not a perfectly isolated ablation: motion vetoes are removed and feedback recovery is added. Record whether recovery activates before the former refusal location and distinguish continued nominal tracking from recovery-assisted advancement.

## Concrete recovery assessment

- An eight-second plateau uses route projection minus cross-track error. Normal yaw progress is also credited while a turn is committed.
- Recovery tests a selected heading, a five-second forward push, and a reverse bounded by 0.5 m measured displacement or four seconds. Response windows select the next maneuver; they do not reinstate physical motion vetoes.
- Heading selection ranks five finite candidates by route distance, the frozen fitted terrain preference and nearby unsuccessful push history. It is a preference, not a clearance or traction guarantee. Censored actions are excluded from failed-push evidence.
- The history is bounded to 64 records; the overall run is bounded to 900 simulated seconds. No trial restart, live patch, route search or new weight fitting is warranted during this run.
- STOP/manual and interrupted-action censoring retain clear ownership. Source inspection confirms the overridden tick does not call the inherited command geometry, actual-geometry, anticipation, tilt, overspeed or cross-track stop gates.

## Evidence limitations and stopping

Complete the run at upright, settled arrival at the selected endpoint, the 900-second autonomous budget, invalid data, process failure, or user STOP/takeover. Do not reintroduce the previous 35-degree, speed, tilt or cross-track thresholds as informal reasons to terminate this explicitly authorized experiment.

Local response score can improve by reducing cross-track error or regaining lost ground. Normal angular progress can defer translational recovery even without new route distance. Therefore report current route projection, furthest route progress, actual total travel, backslide, yaw activity and per-action response separately. Repeated turns or repeated recovery are observations, not successful ascent. Failure to reach the endpoint is not automatically a diagnosed traction failure.

Keep the original fitted cost field with **magma, its original LogNorm normalization and the accepted route colors** for every subsequent report. The shared `slope_cost_plot.py` supplies that normalization from the frozen fitted map; the run report overlays measured motion on it. Do not replace this with a differently normalized live map.

## Bound artifacts

- `source/rexmi_rl/nav/experimental_path_control.py`: `39b753b7897b12e3afbe319169a69b36713a18b1a8d137be34cb4c52286dce73`
- `outputs/remote_autonomy/blue-path-unguarded-20261024-01/launch.sh`: `0cf000d2df3594929fe71efc14fdca99d6b86eca677e0d31549b44cff42a662e`
- Accepted saved result: `3a56ca3ed18bff83bbe24d6457816fefa37e7e3e1baa52d293dc7802e76281b3`

Archive source, inputs, telemetry, recovery events and final reviews. One run, no automatic retry, no live modifications, no reset credit. This approval is an agent review of a user-authorized simulator experiment, not physical deployment qualification.
