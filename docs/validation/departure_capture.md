# Exact departure evidence, September 24

The independent reviewer approved this one observational run at **9/10**. It
reproduced all **3,046** logged pose and command samples from trial 17 through
60.90 simulated seconds exactly. The original returned braking command was
recorded but not applied to physics on the final diagnostic tick. This is not a
mission trial, ascent success, or qualification run. No simulator is left running.

The complete observed map was copied before any new failure exclusion, together
with freshness, motion history, original request, prediction, and returned result.
There were no failure records or no-go cells in that snapshot. The actual lateral
bound was **0.8 m**, correcting earlier prose that described it as 1 m.

At rejection, the robot was at `(1.6255, 4.6322)`, facing 6.83°. The original
committed axis was 45.61° toward `(5.1401, 9.4041)`. Over the last two seconds,
measured displacement was `(0.2741, -0.0725)` m and body height increased 0.0884 m.
Its measured response heading was -14.82°, so continued zero input cannot be
assumed to follow its body axis or the old target.

Offline checks on the exact captured arrays show:

- The current forward body ray has a geometrically clear 10 m corridor, with
  maximum uphill grade 27.12°, cross grade 20.33°, and footprint inclination
  32.87°. The same complete reverse corridor passes geometry checks.
- Its endpoint `(11.5546, 5.8208)` is a gentle staging footprint at 6.84° and
  has a direct geometrically checked route to the actual destination `(13, 0)`.
- Current oriented freshness supports a 0.8 m sweep. A 1.6 m oriented sweep
  encounters stale perimeter observations; the ordinary 2 m planning near field
  also fails freshness. Survey geometry alone cannot authorize a long command.
- The recent measured response ray already exceeds the 25° directional cross
  grade limit. Forward body geometry does not establish zero-command response.
- The existing lattice search exhausted 60,000 expansions from an earlier
  gentle point. Its final subcell connector may demand an unnecessary turn at
  the destination, where footprint inclination is 12.81°, above its 12° turn
  threshold. A continuous terminal connector from gentle staging needs review.

These are geometric hypotheses, not measured traversal success or probabilities.
The reviewer rejected immediate adaptive handoff at 6/10 pending this evidence.
No control change or further simulator run is covered by the diagnostic approval.

Artifacts: `outputs/ascent_departure_diagnostics/capture01/`. Reproduce the
offline report and figure with:

```bash
/home/susan/miniconda3/envs/env_isaacsim/bin/python scripts/analyze_departure.py \
  outputs/ascent_departure_diagnostics/capture01/run.departure.npz
```

![Observed map and competing motion directions](../../outputs/ascent_departure_diagnostics/capture01/run.departure.png)
