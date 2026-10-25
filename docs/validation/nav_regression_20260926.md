# Navigation regression comparison — 2026-09-26

## Result and operational decision

Use the exact GitHub baseline, `2e391f87de58766562a384ec78351d59c1e9ee9b`, for the manual-checkpoint demo. It reached the failing click destination in a fresh-start smoke test. Do not erase the experimental stack or rewrite from scratch based on this single comparison.

The clean baseline lives at `/home/susan/rexmi_rl_github_baseline`. Launch it using `bash /home/susan/rexmi_rl/scripts/launch_github_nav_baseline.sh`. It uses live LiDAR, exterior spawn, the existing rocky-slope weights, and manual checkpoints. No survey prior is loaded. No experimental ascent modules are imported by this launch.

## Evidence

- GitHub origin/main: `2e391f8`. Main workspace HEAD: `c243b5b`, with extensive additional uncommitted navigation changes. These are different baselines; switching only the controller flag in the dirty workspace does not restore GitHub behavior.
- Recorded user run: `outputs/downhill_video/session03`. Click accepted at 35.46 s, target `(10.091271818868542, 1.525175596292815)`. At 38.96 s both route searches returned no candidates after two expansions. Eight recovery maneuvers followed. The run reached HOLD at approximately 69.8 s without reaching the target.
- Exact GitHub baseline: `outputs/nav_regression/github_baseline/run.summary.json`. Same seed 42, environment, exterior spawn, policy, target, 0.40 m/s ceiling, and live sensing. Started navigation immediately after initial observation; completed at 15.6 simulated seconds, zero recovery attempts, zero policy switches, no intervention. Final position approximately `(10.093, 1.123)`; goal radius 0.8 m. Baseline checkpoint tests: 51 passed.
- Current experimental ascent controller: `outputs/nav_regression/current_ascent`. Same fresh-start launch and target as baseline. Entered recovery at 3.5 s; at 15 s was reversing away instead of following a mission route. Operator-interrupted diagnostic run, not a completed qualification or terminal-failure trial. CSV and events retain its exact final observation.
- The original user run waited before clicking, so its initial state differs from the two fresh-start tests. This comparison establishes a regression on this short leg, not crater-wide reliability.

## Code findings

1. The supplied filming command explicitly selected `--nav_controller ascent`. The GitHub default is `checkpoint`. The experimental ascent stack was incorrectly used as the general demo baseline.
2. Checkpoint admission uses the shared observed-route checker, while the behavior-tree planner applies a stricter staged heading search. A click can be accepted and subsequently have no executable candidates.
3. `staged_routes.py` permits heading changes only at staging locations passing the 12-degree support-slope and footprint checks. The bound uses footprint support terrain, not simply chassis tilt. Exact rejected edges were not captured in this run, so the precise cause of its two-expansion search remains unconfirmed.
4. Recovery scores turnability, maneuver length, and elevation change, without destination-distance or onward-route progress in that score. Successful recovery only hands back to mission planning when turning is permitted. This explains how repeated mechanically successful moves can produce mission regression.

## Preservation

Original workspace and outputs retained. Modified/untracked source and documentation archived in `outputs/nav_regression/pre-rollback-source.tar.gz` (119 files); tracked changes also saved in `pre-rollback.patch`. Existing experimental output directories remain in place. No weights, physics, observations, or main-workspace navigation code were reset.
