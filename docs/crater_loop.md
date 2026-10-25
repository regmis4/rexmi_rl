# Autonomous crater loop

From the repository root:

```bash
./crater_auto_nav.sh
```

The launcher was renamed from `crater.sh` to `crater_auto_nav.sh`; archived
run provenance retains the filename used at validation time.

This opens Isaac Sim with the autonomous dashboard and terrain overlay. All
mission parameters are in the script. No map, path or checkpoint selections are
required. For unattended validation, use `./crater_auto_nav.sh --headless`.

**Current selection, 25 October:** the launcher now uses the exact previously
validated blue western ascent route, with a short connector from the measured
center arrival. Its original 0.35 m target and turn/drive tracking are restored;
the northern-route tracking experiments are retired. No ascent terrain/slope
veto is added. **The full loop completed in 185.90 simulated seconds**, with zero recovery
actions, manual interventions or policy switches. Center arrival was 99.60 s.
See the [validated-route report and trajectory](validation/crater_west_loop_2026-10-25.md).
The 40° descent execution setting and dashboard colorbar crash fix remain.
The earlier [fresh northern-route trial](validation/crater_full_loop_2026-10-25.md)
failed despite successful descent; it is not evidence against the old blue route.

The first complete seed-42 loop revision finished in **158.12 simulated seconds**, with
no intervention, policy switch or reset. One descent steering recovery was
needed; ascent completed without recovery. This establishes one working
continuous simulation, not reliability across terrain seeds or hardware. The
subsequent [conservative descent revision](validation/crater_descent_observation_reserve_2026-10-24.md)
added measured-data reserves and stronger reverse recovery. The current
[incremental descent revision](validation/crater_incremental_descent_2026-10-24.md)
replaces its slow viewpoint detours with short goalward approaches; each
revision's evidence is recorded separately.

The preceding follow-up was deliberately limited to descent checkpoint selection
and recovery. The [25 October validation](validation/crater_descent_completion_2026-10-25.md)
tracks it separately from earlier successful runs. The larger experimental
controller rewrite has been retired. Those trials retained the ascent planner,
executor and mission handoff from checkpoint commit `dfc0ef7`. That revision reached the
center in 138.74 s unattended and 150.52 s with the original STOP/RESUME
timing. Both handed off to unchanged ascent; full-loop qualification was not
completed in these descent-focused trials.

Descent checkpoints require 0.25 m of extra measured-data clearance beyond
the robot envelope, ordinary clearance and cell allowance, increasing with
slope as before. Route reserve remains 0.10 m; the exponential objective and
goalward checkpoint selection remain. Shallow inward contours are allowed;
exponential cost prices endpoint slope instead of a separate hard cutoff.
Display extrapolation does not count as measured terrain. Waiting for scans uses the existing position
hold; drift beyond 0.20 m starts backout. Descent reverse retains the existing
0.4 → 0.7 → 1.0 m/s escalation and 0.5 m measured-retreat target, with up to
18 s to respond. An unsuccessful attempt reports HOLD instead of resuming
forward navigation. Ascent retains its original six-second reverse window.

The mission starts outside the east rim at `(13, 0)`, facing the crater. The
checkpoint controller chooses intermediate observed checkpoints toward
`(0, 0)`. Full map retention keeps the LiDAR, forward-scanner and height-scanner
observations gathered during descent. It does not load a map from another run
or request a simulator-wide terrain survey.

Descent now uses the same exponential slope objective as ascent, applied to
absolute travel grade so steep downhill faces are expensive too. It requires
0.10 m of data reserve along a route and 0.25 m at an easy-slope checkpoint,
beyond the body envelope, ordinary clearance and cell-square allowance.
Checkpoint reserve grows by 0.02 m per degree above 15°. The extrapolated edge
of the displayed plane does not count as measured coverage. The occupied
starting cell may omit the extra 0.10 m route reserve if its body, ordinary
clearance and cell allowance are already fully measured. All future route
cells and checkpoint margins remain required. This prevents a supported robot
from being stranded solely by the extra routing buffer.

When the center is not yet connected, descent chooses steps up to 2 m toward
it, rather than rewarding unknown terrain area or touring lateral viewpoints.
Each frontier approach needs a measured 0.6 m straight retreat corridor whose
wheelbase-scale grade along the reverse direction is at most 20°. The next
checkpoint must reduce goal distance by at least 0.05 m. New observations roll
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
with an upright body. The mission freezes the collected live map and connects
its measured pose to the first waypoint of
`source/rexmi_rl/nav/routes/validated_west_seed42.json`. All 27 original blue
waypoints, including the exterior endpoint `(11.1, 4.1)`, remain unchanged.
This artifact comes from the successful `blue-path-unguarded-20261024-01`
run, whose standalone ascent finished in 102.30 simulated seconds. Its task,
seed, crater origin and rim radius must match; this is a scenario-specific
validated route, not a general route for arbitrary terrain seeds.

The cost model remains the accepted 15° knee / 6° exponential scale with
wheelbase gradient, roughness interaction and positive distance costs. Live
costs describe the selected route but do not replace it with another exit or
apply a new gentle-terminal veto. The original route artifact and its hash,
the exact live map and the connector length are recorded with each plan.
No historical height map is loaded. Direct `navigate.py` crater-loop runs
without `--crater_ascent_route` still explicitly use fresh any-exit planning;
the simple launcher supplies the validated route by default.

If the robot drifts more than 0.3 m during preparation, the connector is rebuilt
from its new pose. The unguarded v2 executor follows the route using its original
turn/drive behavior, with the previously added stall detection and requested
reverse escalation retained. The experimental long lookahead and continuous
steering are removed. Robot, policy, environment and live map remain in place;
there is no respawn, teleport or reset. One 900 simulated-second budget covers
the full mission.

The current local-map/dashboard layout and magma colors are retained. During
descent the map shows live terrain costs; after planning it shows the frozen
fitted cost field measured during this descent, with the validated route overlaid. Descent uses the same magma/log
renderer with its live exponential cost model. The title identifies autonomous,
manual, paused or assisted operation, and the mission label reports the active
stage. The elevation/grade chart spans the entire mission. Manual takeover or
STOP remains available and is recorded as intervention.

Reverse recovery in both phases starts at 0.4 m/s requested speed and can
escalate to 0.7, then 1.0 m/s when measured retreat stalls. Each attempt ends
after 0.5 m retreat, or after 18 s for descent / 6 s for ascent. Productive
motion avoids unnecessary escalation; a timed-out descent retreat cannot
automatically resume forward driving.
Forward cruise stays at 0.4 m/s. Missing checkpoint data causes a rescan/hold,
not a blind forward recovery.

Each launch creates a fresh directory under `outputs/crater_loop/`; the path is
printed, and `outputs/crater_loop/latest.txt` points to it. Outputs include:

- `run.csv`, `console.log`, `run.events.jsonl`: time-aligned motion and stage events.
- `run.map.npz`: retained live map, periodically and finally saved.
- `exit-plan-*/request-*.npz`: exact frozen planning inputs.
- `exit-plan-*/validated_route.json`: copied route artifact and source fingerprint.
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
