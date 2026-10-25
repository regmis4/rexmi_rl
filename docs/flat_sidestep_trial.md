# Manual-checkpoint trial: guarded flat-ground sidesteps

This opt-in experiment keeps rocky-slope for ordinary navigation and slope travel.
It can switch to rough for a short sideways correction along the planned route.
The earlier 35-degree route limit remains separate from the much stricter
5-degree policy-switch limit. No policy retraining or physics changes are involved.

## Launch

```bash
cd /home/susan/rexmi_rl
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgcc_s.so.1 ./run.sh scripts/navigate.py \
  --task RexmiRl-Go2w-Crater-Bowl-RockySlope-Play-v0 \
  --checkpoint logs/rsl_rl/go2w_velocity_rocky_slope/2026-06-30_09-31-48/model_13994.pt \
  --ckpt_rough logs/rsl_rl/go2w_velocity_rough/2026-06-14_20-03-41/model_8996.pt \
  --flat_sidesteps --perception_view --manual_checkpoints \
  --map_retention radius --map_keep_radius 20 \
  --lidar_display_points 1000 --overlay_hz 30 \
  --log_file logs/nav/manual_flat_sidestep.csv --max_steps 30000
```

Use a different log filename for each run to retain comparisons. Remove
`--flat_sidesteps` to return to the single-policy controller.

Click a destination in either map of the **REXMI Nav Dashboard**. First try a
short sideways destination in open, mapped, nearly level terrain. The normal
spawn is on an exterior incline: sidestepping should stay unavailable there.
Drive to a suitable flat patch before testing it. The new floor boulder and
nearby walls may exclude some sidesteps even when the robot's centre looks clear.

## Guardrails

- Switch only if the entire observed footprint is at or below **5 degrees**.
  Entering rough additionally checks the route capsule plus **0.65 m reserve**
  beyond the footprint for stopping drift. Unknown, stale, hazardous or blocked
  terrain rejects the manoeuvre. Measurements must be at most **5 s old**.
- Body tilt must be no more than **8 degrees**. Planar speed must stay below
  **0.05 m/s**, vertical speed below **0.08 m/s**, and angular speed below **0.15 rad/s** for **0.5 s** before either
  policy handoff. A failed initial stop cancels entry after 5 s.
- Each sidestep targets at most **0.8 m** of the already checked route. It is
  considered only for a route bearing 30–105 degrees to either side. Normal
  straight travel stays on rocky-slope. Cream exits do not trigger a switch.
- Commands are capped at **0.15 m/s sideways**, **0.15 m/s forward**,
  **0.12 m/s reverse**, with yaw feedback holding the entry heading. The map is
  checked continuously; measured speed above 0.35 m/s, excessive heading drift,
  inadequate progress, lost flat-ground evidence or operator input ends the
  sidestep. An eight-second limit bounds the driving phase.
- It brakes with rough, then switches back to rocky-slope only after another
  stable, flat-ground stop. A 5 s cooldown prevents rapid switching.
- If terrain under the robot is no longer flat or adequately observed, the
  return handoff is blocked. It holds with the currently active policy and
  displays why; it never swaps policies on a newly discovered slope. This is
  a conservative fallback, not a claim that unexpected slipping is impossible.
- STOP immediately cancels the sidestep and keeps autonomy paused. A new target,
  manual drive command or mode change also cancels it; lateral braking and a
  safe return handoff take precedence over executing a new drive request.

These thresholds are conservative experimental choices, not validated physical
limits. Larger body rotation may still be necessary when lateral space is poor.

## What to watch

The overlay, dashboard and control window show the **active policy** and switch
reason. `STOP_ROCKY` is the entry stop, `SIDESTEP` is rough moving laterally, and
`STOP_ROUGH` is braking/return handoff. A rejected sidestep leaves rocky-slope
navigation available. The CSV includes `vy_cmd`, `sidestep_phase`, `switch_reason`
and the actual policy. Both handoffs are recorded in `.events.jsonl`; the summary
reports actual switch counts. The selected checkpoint and mission order are kept.

## Automated checks

The pure controller tests cover slope/footprint margins, stale and unknown cells,
obstacles, body tilt, stable-stop timing, left/right commands, drift feedback,
operator STOP, a newly discovered slope, return handoffs and switch cooldown.
The navigation and perception regression suites remain required.

Reproduce the sensor-fed simulator comparison:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgcc_s.so.1 ./run.sh scripts/flat_sidestep_study.py \
  --rough logs/rsl_rl/go2w_velocity_rough/2026-06-14_20-03-41/model_8996.pt \
  --rocky logs/rsl_rl/go2w_velocity_rocky_slope/2026-06-30_09-31-48/model_13994.pt \
  --output logs/nav/flat_switch/seed42.json
```

This compares left/right flat goals, a textured flat goal and a 10-degree terrain
boundary with real simulated sensor observations. It is a development test, not
full crater-traversal validation.

For the rocky-slope-only comparison, add `--no_switches` and use a different
`--output` filename.

## Results — 20 September 2026

All **105 controller, navigation and perception tests passed**. A visible
100-step launch loaded both policies, the overlay and dashboard, then waited
for a manual checkpoint without switching.

The simulator comparison used seed 42, four environments and 25 simulated
seconds per run. Goals were selected at 2 s. Both modes completed all four
goals without tips or automatic resets, using the existing **0.6 m arrival
tolerance**. Times below run from goal selection to confirmed completion;
distance is measured robot travel over that interval.

| Case | Guarded sidestep: time / distance | Rocky-slope only: time / distance | Handoffs |
| --- | --- | --- | --- |
| Flat left | 9.36 s / 0.84 m | 9.84 s / 1.74 m | 2 |
| Flat right | 9.52 s / 0.78 m | 7.82 s / 1.14 m | 2 |
| Textured flat | 9.42 s / 0.77 m | 7.50 s / 1.22 m | 2 |
| 10° slope boundary | 8.06 s / 1.43 m | 8.04 s / 1.45 m | 0 |

Every handoff met the flat-ground and stable-stop gates. The largest measured
handoff speed was 0.0463 m/s, angular speed 0.0964 rad/s, vertical speed
0.0251 m/s and body tilt 4.14°. The slope-boundary case stayed on rocky-slope.

Sidestepping reduced travel distance in the three flat cases, but braking and
handoffs made two cases slower. Final goal errors were 0.35–0.60 m with switching
versus 0.36–0.41 m without it. This supports a manual trial, not a claim of faster
or more precise navigation. Only one seed and short goals were tested; longer
paths, crater operations and repeated switching still need evaluation.

The controlled test fixes robot friction at 0.8 static / 0.6 dynamic in both
modes; it does not change the demo's contact materials. Summary maxima over the
whole run include initial spawn settling and must not be read as handoff values.
Measurements and handoff events are saved in
[`validation/flat_sidestep_summary.json`](validation/flat_sidestep_summary.json).
