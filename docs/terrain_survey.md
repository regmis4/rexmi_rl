# Terrain survey and climb-aware routing

Checkpoint navigation keeps the rocky-slope policy, teleop adapter, mission
order and 0.40 m/s default. The demo's contact materials are unchanged.

## Clearance and terrain

Red denotes a detected obstacle or the existing >50° slope limit. Cream is the
planning clearance buffer: ordinary routes exclude it. If the robot is already
inside cream, a short exit may continue at up to 0.20 m/s when fresh observations
confirm full body clearance and stopping room. An obstacle inside the body
envelope still blocks movement. Yellow and teal/blue are traversable cost bands,
not stop commands. Logs distinguish extra margin, actual footprint conflict,
unknown terrain, stale observations and excessive slope.

The map adds roughness RMS in metres, estimated from height samples after
subtracting local ground slope. At least six of the sixteen fine evidence bins
must be observed; insufficient support remains unknown. Roughness is not a
friction estimate. Excess relief above 6 cm RMS adds difficulty; greater
roughness never grants permission to cross a hazard.

Routing includes along-travel slope, cross-slope travel, turning, clearance and
directional traversal history. Successfully driven directions avoid a small
untested-travel penalty. One failed uphill episode adds cost; a second excludes
that heading bin in the affected area for the mission. There are eight heading
bins; the other travel directions remain separately assessed. Evidence is
recorded once per cell per episode, with a 0.4 m neighbourhood at 0.2 m grid
resolution for failures.

Uphill failure detection requires aligned forward commands above 0.25 m/s and
an along-heading grade above 0.15. A two-second window with mean measured forward
speed below 0.05 m/s records poor progress; a mean below -0.10 m/s records
backsliding. Stops, manual control, alignment turns and reverse commands are
excluded. These are diagnostic defaults, not a measured friction coefficient.
The existing 0.75 m cross-track limit also records a failed slope approach when
forward drive is active on terrain steeper than grade 0.25. This catches sideways
sliding even when measured forward speed stays positive. It is labelled a tracking
failure, not an inferred friction value. A failure triggers settling and replanning,
not another blind acceleration.

At checkpoints the planner compares the whole observed route with alternatives.
A valid cached route stays unless a comparable alternative is at least 15%
cheaper; invalid routes are replaced immediately. When no goal-directed approach
exists, it can target a reachable, observed survey boundary with unknown terrain
nearby. It never targets an unknown cell or skips a mission destination.

## Display and exports

The entire current mission survey is retained at 20 cm resolution, including
5 cm elevation evidence bins. Checkpoints, old timestamps and display distance
do not erase it. Freshness gates the immediate motion corridor separately.

Both displays offer Terrain, Slope, Roughness and Traversal views. The dashboard
shows the full mission area. Isaac Sim shows fine tiles within 12 m and combines
fully observed distant 2x2 groups into 40 cm tiles. Aggregation retains isolated
red hazards and does not fill unknown gaps. Stable mesh chunks avoid reallocating
the entire survey for each update. Layer colours are diagnostic; they never
change the planner's blocked cells.

Add `--survey_layer slope` (or `terrain`, `roughness`, `traversal`) to choose the
initial Isaac Sim layer. Buttons in the perception window and radio buttons in
the dashboard switch layers during a run. The red line is the active route;
gold marks the checkpoint. Purple in Traversal means a direction has failed twice,
not that every possible direction through that cell is forbidden.

A complete `.map.npz` is atomically replaced every 30 simulation seconds and on
shutdown. It includes heights, slope, roughness/support, hazards, clearance,
observation times, evidence counts, directional successes/failures, world origin
and cell size. The `.csv` adds slope, roughness, cumulative climb failures,
backslide distance, route-selection reason and overlay update time. Climb events
also appear in `.events.jsonl`; `.materials.json` records actual robot contact
properties and configured terrain properties. No cross-run map loading is added.

## Commands

From the repository root, use the existing checkpoint command:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgcc_s.so.1 ./run.sh scripts/navigate.py \
  --task RexmiRl-Go2w-Crater-Bowl-RockySlope-Play-v0 \
  --checkpoint logs/rsl_rl/go2w_velocity_rocky_slope/2026-06-30_09-31-48/model_13994.pt \
  --perception_view --survey_layer terrain
```

Controlled uphill comparison (separate experimental terrain, eight matched
conditions, one initial reset, no recovery resets):

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgcc_s.so.1 ./run.sh scripts/traction_study.py \
  --checkpoint logs/rsl_rl/go2w_velocity_rocky_slope/2026-06-30_09-31-48/model_13994.pt \
  --slope_deg 30 --output /tmp/rexmi-traction30.json
```

This compares 0/5 cm generated texture, dynamic friction 0.6/0.8 and commands
0.40/0.80 m/s. Static friction stays 0.8; terrain friction stays 1.0 with multiply
combination. Robots start on the flat approach, settle for two seconds, drive for
ten seconds and receive zero velocity commands for three seconds. Stops retain
policy balancing. This is a fixed-command diagnostic, not autonomous routing or
position-hold control. It records tipping, climb gain, backsliding, lateral error
and stop drift. The full time series is saved alongside summaries in the JSON.

See [validation results](terrain_survey_validation.md). Full navigation remains
unvalidated until three repeatable crater traversals meet the acceptance criteria.
