# Controls review — inspected blue path

**Score: 8.3/10. Approved for one seed-42, north-start characterization trial, subject to the recorded practicality/diagnosis approvals and usage ceiling.** No remaining concrete controls blocker found in the reviewed implementation.

Reviewed `saved_path_control.py`, its `navigate.py`/`checkpoint_runner.py` integration, inherited remote final gates, and the execution contract in `docs/ascent_navigation.md`.

- The saved any-exit polyline remains fixed. Map checksum and terrain identity are checked; result checksum is recorded and rechecked before its exact bytes are archived. Source snapshots must accompany the run.
- Bounded projection prevents a distant future branch from advancing the route cursor. Cross-track and shared progress deadlines survive button changes; failure stops without replanning.
- Turning removes staging/minimum restrictions as authorized. Requested yaw plus observed translation is checked as one footprint envelope, including full-sweep cross-slope, hazards, unknown support, exclusions and physical terrain limits. Inherited measured-drift, anticipation, STOP, tilt, overspeed and mission deadline remain active.
- Completion requires the selected endpoint, supported gentle exterior terrain, contiguous settling and the final physical gate. Radius crossing alone cannot complete the mission.
- Mission-route/profile telemetry now initializes and survives resume. The archived result cannot silently differ from the controller-loaded result.

Independent validation: 142 tests passed across saved-path, remote-control and exit-region suites; after final telemetry/hash changes, all 11 saved-path tests passed again. No simulator run performed by this reviewer.

Scope: unchanged policy/physics, 0.40 m/s command ceiling, existing configured terrain limits, 900-second mission ceiling, no automatic retry or live edits. The translated-yaw forecast is provisional. Known exact-corner rotation refusals and the fitted route's unproven traction remain experimental uncertainties; approval permits the bounded test, not a claim of safe or successful ascent.
