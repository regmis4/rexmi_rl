# Autonomous crater loop

From the repository root:

```bash
./crater.sh
```

This opens Isaac Sim with the autonomous dashboard and terrain overlay. All
mission parameters are in the script. No map, path or checkpoint selections are
required. For unattended validation, use `./crater.sh --headless`.

The first complete seed-42 loop revision finished in **158.12 simulated seconds**, with
no intervention, policy switch or reset. One descent steering recovery was
needed; ascent completed without recovery. This establishes one working
continuous simulation, not reliability across terrain seeds or hardware. The
subsequent [conservative descent revision](validation/crater_descent_observation_reserve_2026-10-24.md)
added measured-data reserves and stronger reverse recovery. The current
[incremental descent revision](validation/crater_incremental_descent_2026-10-24.md)
replaces its slow viewpoint detours with short goalward approaches; each
revision's evidence is recorded separately.

The mission starts outside the east rim at `(13, 0)`, facing the crater. The
checkpoint controller chooses intermediate observed checkpoints toward
`(0, 0)`. Full map retention keeps the LiDAR, forward-scanner and height-scanner
observations gathered during descent. It does not load a map from another run
or request a simulator-wide terrain survey.

Descent now uses the same exponential slope objective as ascent, applied to
absolute travel grade so steep downhill faces are expensive too. It requires
0.10 m of data reserve along a route and 0.15 m at an easy-slope checkpoint,
beyond the body envelope, ordinary clearance and cell-square allowance.
Checkpoint reserve grows by 0.02 m per degree above 15°. The extrapolated edge
of the displayed plane does not count as measured coverage.

When the center is not yet connected, descent chooses steps up to 2 m toward
it, rather than rewarding unknown terrain area or touring lateral viewpoints.
Each frontier approach needs a measured 0.6 m straight retreat corridor whose
wheelbase-scale terrain inclination is at most 20°. Goal-distance reduction
must be at least half the straight-line travel distance. New observations roll
the checkpoint forward near arrival, without a mandatory stop at every old
data edge. If no recoverable goalward step exists, it rescans and reports that
limitation instead of starting a backwards or lateral-only information hunt.

For this simulation, descent bypasses predictive motion/turn-clearance vetoes and automatic
steep-edge repositioning. This follows the requested motion-veto ablation and
is scoped to this mission; existing standalone checkpoint runs are unchanged.
If a sustained turn fails to reduce heading error by 0.05 rad within 5 s, descent
backs out through observed terrain, then rescans and replans.
Measured heading progress renews that patience; it is not a fixed turn cutoff.

At the center, measured speed and yaw rate must remain below 0.12 for 0.6 s,
with an upright body. The mission freezes a copy of the collected map and plans
an exit on a worker thread while the balancing policy continues with zero input.
It uses the accepted exponential slope preference: 15° easy-grade knee, 6° scale,
wheelbase gradient, roughness interaction and positive distance costs. All
observed, connected, gentle exterior terminals compete on this objective. The
old blue route and manual-route coordinates are not loaded. An incomplete map
can produce a different exit or no connected exit; that result is reported.

The route starts at the measured center pose. If the robot drifts more than
0.3 m during planning, the route is recomputed from its new pose. The successful
ascent executor then follows the new route with its existing measured-progress
turn/reverse/push recovery. The robot, policy, environment and live map remain
in place across the transition. There is no respawn, teleport or environment
reset between descent and ascent. One 900 simulated-second budget covers the
whole mission.

The current local-map/dashboard layout and magma colors are retained. During
descent the map shows live terrain costs; after planning it shows the frozen
fitted cost field used to select the exit. Descent uses the same magma/log
renderer with its live exponential cost model. The title identifies autonomous,
manual, paused or assisted operation, and the mission label reports the active
stage. The elevation/grade chart spans the entire mission. Manual takeover or
STOP remains available and is recorded as intervention.

Reverse recovery in both phases starts at 0.4 m/s requested speed and can
escalate to 0.7, then 1.0 m/s when measured retreat stalls. Each attempt ends
after 0.5 m retreat or 6 s; productive motion avoids unnecessary escalation.
Forward cruise stays at 0.4 m/s. Missing checkpoint data causes a rescan/hold,
not a blind forward recovery.

Each launch creates a fresh directory under `outputs/crater_loop/`; the path is
printed, and `outputs/crater_loop/latest.txt` points to it. Outputs include:

- `run.csv`, `console.log`, `run.events.jsonl`: time-aligned motion and stage events.
- `run.map.npz`: retained live map, periodically and finally saved.
- `exit-plan-*/request-*.npz`: exact frozen planning inputs.
- `exit-plan-*/exit_plan.json`, `exit_route.csv`: selected exit, route, cost,
  observed area and planning duration.
- `run.loop.json`, `run.summary.json`: both-stage outcome and intervention status.
- `run.provenance.json`, `run.sources/`: policy fingerprint and runtime source snapshots.

The launcher checks that the trained policy exists, preserves previous runs,
and exits nonzero for an incomplete/intervened mission or a software failure.
Set `REXMI_CHECKPOINT` to relocate the same trained policy. Existing Isaac Lab
configuration in `.env` and the environment selection in `run.sh` still apply.

Implementation checks and physical validation are tracked in
[the crater-loop validation report](validation/crater_loop_2026-10-24.md).
