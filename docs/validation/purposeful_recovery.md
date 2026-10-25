# Purposeful recovery — 2026-09-28

Ascent remains unqualified. This revision preserves the existing policy-active
zero-command climb, slowdown thresholds, progress budgets, terrain limits,
physics, policy weights, and GUI overlays. The GitHub demo checkout remains at
`2e391f87de58766562a384ec78351d59c1e9ee9b`.

## Change

- Choose forward or reverse at recovery action boundaries. Rank checked mission
  connections before turn staging before purposeful escape/reposition. Remove
  the unconditional preference for descending terrain and the cross-action sign
  lock. Keep commitment during an action and the original episode/trial memory.
- Freeze occupied failed-patch identities, raster cells, and heading bins at
  episode entry. Only incidental drift accompanying independently checked drive
  or velocity-opposing damping can use this permission. Actual exit, STOP,
  manual takeover, fault, terminal state, and episode expiry retire it. Queries
  cannot renew permission, acquire new patches, or erase failure evidence.
- Allow checked straight mission continuation without requiring a turn area.
  Actual body heading, passive yaw, stopping room, and route geometry still apply.
- Archive the new permission state and source; use the execution-protocol suffix
  `+purposeful-recovery-v1` so old probability calibration cannot masquerade as
  validation of this revision.

## Evidence and limits

Trial35's exact 81.14/81.68-second snapshots now admit braking and reverse without
erasing its directional failure. This corrects a recovery veto; the preceding
80.58-second uphill stall remains a separate cause. The newer user GUI capture's
63.28-second reverse grade/yaw rejection remains binding. Opposing damping uses
the original physical stopping envelope, not a new planned traversal's grade
classification.

The historical exterior checkpoint regression does not prove boulder incapacity.
Initial footprint support was approximately 12.60 degrees in session03 and
13.36 degrees in current_ascent. No exact rejected-edge snapshots exist; final
maps cannot reconstruct those decisions. Planning rejection produced no physical
failure vote. Preserve the later genuinely executed reverse lateral failures.

The default movement comparison replay contains 1,741 recorded observations and
zero differences. Its geometry callbacks assume clearance: it establishes
executor equivalence, not alternative physical outcomes. Exact map snapshot
tests separately exercise the changed recovery guards.

Independent AI control review: design 8.4/10; exact implementation 8.3/10 after
fixing permission retirement before timeout braking. Review authorizes one
diagnostic ascent only after full tests and source/configuration freeze.

The test harness also needed its existing perception-view mock to initialize
`selected_path_only=False`, matching the constructor. Overlay runtime code was
unchanged by this revision.

## Diagnostic run

Output: `outputs/ascent_behavior_tree/trial36/`.
Seed42, floor to `(13, 0)`, labeled development survey, rocky-slope policy,
0.40 m/s command ceiling, 900 simulated seconds maximum, no intervention or
resets. V32 release memory and uphill-speed response remain enabled; V33
correction timing and V31 recovery continuation remain disabled. Stop on the
normal mission terminal condition and report separately: climbing response,
actual recovery travel, onward continuation, and mission completion.

No second simulator run is authorized for this revision. Failed runs remain
preserved. Reliability qualification and probability calibration are outstanding.

## Verification and initial startup blocker

- Full suite: **734 tests passed**, 32.924 seconds. Independent focused review:
  16 tests passed. Compile and diff-whitespace checks passed.
- Reviewer approved the exact implementation **8.3/10**. The single-launch gate
  freezes 23 source files, the policy, survey, and launch configuration in
  `outputs/control_review/purposeful_run_gate.json`.
- The authorized launch failed during environment creation with
  `RuntimeError: No CUDA GPUs are available`. **No robot mission started, no
  physical attempt occurred, and no ascent success/failure is inferred.**
- Hardware enumeration finds the NVIDIA GPU, but `nvidia-smi` cannot communicate
  with its driver, `/dev/nvidia*` is absent, and `modinfo nvidia` reports no module
  for the running kernel `7.0.0-34-generic`. No driver installation, reboot, CPU
  physics substitution, or second simulator launch was attempted.
- All frozen sources and policy/survey hashes still match after the failed
  startup. See `trial36/console.log` and `trial36/startup_failure.json`.

The user restored GPU availability and explicitly authorized a retry. The
reviewed sources, policy and survey hashes were unchanged. The completed retry
is recorded separately below; trial36 remains an infrastructure failure.

## Authorized retry: trial37

**Ascent failed.** The mission stopped at 112.34 simulated seconds on excessive
body tilt, with zero destinations reached. No intervention, policy switching,
second run, or post-run controller changes occurred.

| Measurement | Observed result |
|---|---:|
| Original uphill low-progress failure | 80.58 s, unchanged from trial35 |
| First changed command | 81.14 s: opposing brake −0.358 m/s instead of zero |
| Exit from original frozen exclusion | 85.08 s; no subsequent re-entry |
| Actual DRIVE_REVERSE command exposure | 26.56 s; longest continuous interval 10.32 s |
| Recovery maneuvers | 5 |
| Usable staging / onward mission handoff | Not achieved |
| Later reverse lateral-departure failure | 111.82 s |
| Peak active body tilt | 61.85° |
| Peak measured active speed / command ceiling | 0.935 / 0.400 m/s |
| Controller p95 / mean map update | 0.987 / 48.110 ms |

Measured poses and commands matched trial35 until the first changed command at
81.14 s. The same policy-active RELEASE interval from 48.54–68.22 s gained
approximately 0.827 m of body height; the subsequent uphill stall was not fixed.

The escape repair worked locally: reverse travel left the original failed patch.
It did not produce controlled recovery to a usable staging area. At 111.82 s,
lateral departure ended the final reverse action. Commanded velocity became
zero at 111.84 s, while measured motion subsequently accelerated and body tilt
increased until the hard fault. This sequence does not establish that a different
braking command would have prevented the fault; no alternative was simulated.

Command/evidence audit passed: no command above ±0.40 m/s, no commanded steep
turns, no simultaneous drive/yaw, no duplicate outcomes or censored failure votes.
These checks do not make the physical outcome successful. Two physical approach
failures remain recorded, and reachability probabilities remain unvalidated.

Artifacts: `outputs/ascent_behavior_tree/trial37/run.csv`, `run.events.jsonl`,
`run.summary.json`, `result.json`, `audit.json`, `comparison.json`, frozen source
archive, joint telemetry and decision snapshots. All frozen sources/inputs still
match. Ascent qualification remains unpassed; no downstream stage advanced.

## GUI launcher and output-path startup fix

An unset shell `RUN` variable expanded the user's logging paths to `/run.csv`
and `/joints`. The resulting permissions error occurred after simulator startup.
Use the self-contained launcher, which creates a fresh output directory:

```bash
bash /home/susan/rexmi_rl/scripts/launch_ascent_gui.sh
```

It enables the dashboard, LiDAR/costmap overlays, event snapshots and joint
logging. Output directory validation now precedes Isaac Sim startup; early-error
cleanup no longer calls an uninitialized survey exporter or masks the original
exception with a failed error-report write. Control behavior is unchanged.

Validation: 22 focused tests passed; launcher checked twice with `RUN` unset
using a simulator stub; the actual invalid-path CLI exits before `AppLauncher`.
Independent review: 8.8/10. No additional physics run. This later startup-only
change modifies launcher/runner source hashes; trial37 retains its exact frozen
source archive and the hashes verified at that run's completion.
