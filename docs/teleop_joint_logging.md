# Teleop motor-sizing capture

Add these options to your normal `scripts/teleop.py` invocation (keep your task/checkpoints):

```bash
python scripts/teleop.py --joint_log_dir logs/telemetry/flat_run_01 --scenario flat --run_notes "Baseline robot; no added payload"
```

- Activate `env_isaacsim` first. The existing teleop defaults still apply.
- Each output directory must be new. Existing runs are never overwritten.
- Without `--joint_log_dir`, joint logging is disabled and existing navigation CSV behavior remains.
- Change labels using the remote's Scenario entry and Apply button, or type `mark uphill_35` in the terminal and press Enter. Stop before typing; terminal direction commands remain latched until `s`.
- Labels/notes describe the run: they do **not** add a 2 kg payload or create/measure a 35-degree terrain slope.

## Files

- `samples.csv`: one wide row per physics step, all articulation joints in runtime order, units in column names. At dt=0.005 this is 200 Hz, not the 50 Hz policy rate.
- `metadata.json`: resolved environment/actuator configuration, joint/body ordering, runtime properties and body masses, timestep, CLI arguments, checkpoint and root USD hashes, git commit/status and logger source hashes. USD hash covers only the root file, not referenced layers/assets; retain the asset tree and checkpoint configuration for reproduction.
- `events.jsonl`: capture start/end, scenario/policy changes and automatic resets, keyed to simulation time and episode. An `auto_reset` event at control step K flags that whole terminal step for review.

## Signals and interpretation

- Every joint: position, angular velocity, angular acceleration, position/velocity targets, pre-clipping torque demand estimate, post-clipping torque estimate, runtime effort/speed limits, signed mechanical power estimate and clipping flag.
- Body: world position, quaternion **wxyz**, world linear/angular velocities.
- Contacts: world-frame net force on each contact-sensor body and estimated wheel contact count (force norm >5 N). Side/obstacle contact also counts; this is **not** a guaranteed supporting-foot count. Sensor update period is in the resolved config; periods longer than physics dt produce held/stale contact readings.
- Missing optional values are blank, not zero. Core position/velocity/torque absence stops logging with an error. Targets may be inactive (e.g. wheel position targets under velocity control); consult actuator gains/config.
- Current implicit actuator torques are **PD/model estimates, not measured PhysX motor efforts**. Estimates correspond to substep-start control evaluation; state is read after that physics substep. Power is the estimate multiplied by end-of-substep speed, not exact electrical power or integrated mechanical work.
- Clipping means demand and applied model estimates differ by >1e-5 Nm. This does not detect every solver-side speed/effort constraint. A capped peak is not proof of sufficient motor capacity.
- Joint acceleration is the simulator's finite-difference state estimate; treat reset/impact spikes carefully. Contact force is not joint bearing reaction load.

## Sampling and safe capture

- The optional observer wraps this scene instance's `update` method, calls the original first, then samples during `env.step`. It neither changes actions nor replaces the environment stepping loop. The installed ManagerBasedRLEnv updates the scene once per physics substep, before automatic reset.
- Exactly `decimation` samples must occur per control step; an incompatible stepping path raises an error. The observer is detached on exit.
- Initial reset and forced settling are excluded. Terminal physics states are captured before auto-reset; the next control step gets a new episode ID. Capture time is monotonic simulated time since logging began, not wall time or time since app launch.
- CSV is streamed with bounded memory and flushed at least once per wall-clock second of sampling, and on clean quit/Ctrl-C. Sudden process/power loss can lose the unflushed tail. Logging synchronizes device data to CPU and can reduce real-time rendering speed; it never deliberately skips samples.
- Use short labelled runs first. Inspect the console for errors and ensure each control step has the expected sample count before a long session. Prefer `q`/remote close to killing the process.
- Start with level standing/rolling, acceleration/braking, turns, then climb/descent/hold/slow turns and rough terrain. Record actual slope and payload separately. Label falls/recovery and review them separately from normal operating duty—teleop currently disables some fall terminations.
- These traces support later torque-speed, RMS and peak-duration analysis; they are not yet a validated motor rating or thermal qualification. Do not change the policy/controller simply to obtain a larger torque number.

## CPU-only tests

```bash
python -m unittest discover -s tests -v
```

Tests cover physics substeps, power signs, clipping, contacts, missing data, scenario/reset boundaries, cleanup and overwrite protection using a fake environment. A real Isaac Sim capture is still required to validate runtime integration and effort-estimate accuracy against a known-load case.
