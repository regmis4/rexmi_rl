# Terrain survey and climb-aware routing

Checkpoint navigation keeps the rocky-slope policy, teleop adapter, mission
order and 0.40 m/s default. The demo's contact materials are unchanged.

## Clearance and terrain

Red denotes a detected obstacle. Orange marks slopes >35°, blocked without
additional slope inflation. Cream is the
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

The default retains a rolling **20 m radius** at 20 cm resolution, including
5 cm elevation evidence bins. Evidence and traversal outcomes outside that radius
are discarded; exported maps contain only retained terrain. Use
`--map_retention full` to keep the full current-run survey instead. In either
mode, freshness gates immediate motion separately from retained evidence.

Both displays offer Terrain, Slope, Roughness and Traversal views. The dashboard
shows the retained map area. Isaac Sim shows fine tiles within 12 m and combines
fully observed distant 2x2 groups into 40 cm tiles. Aggregation retains isolated
red hazards and does not fill unknown gaps. Stable mesh chunks avoid reallocating
the entire survey for each update. Layer colours are diagnostic; they never
change the planner's blocked cells.

Add `--survey_layer slope` (or `terrain`, `roughness`, `traversal`) to choose the
initial Isaac Sim layer. Buttons in the perception window and radio buttons in
the dashboard switch layers during a run. The red line is the active route;
gold marks the checkpoint. Purple in Traversal means a direction has failed twice,
not that every possible direction through that cell is forbidden.

A snapshot of the retained map, `.map.npz`, is atomically replaced every 30 simulation seconds and on
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


### Cream-buffer stop regression (2026-09-16)

The reported run repeatedly rejected its drift stopping corridor for the extra
planning margin, then alternated FOLLOW, SETTLE and OBSERVE. Execution checks had
used the route-planning mask even for slow corrective movement. Cream-exit checks
also required the starting obstacle distance rather than the actual body envelope.

Normal routes still exclude every cream cell. Commands at or below 0.20 m/s may
use the additional planning buffer if the sampled stopping corridor and commanded
arc preserve footprint clearance, fresh observations, the slope limit and allowed
traversal directions. Actual drift checks enforce body clearance rather than the
extra planning margin. Diagonal corner checks remain active. Motion above 0.20 m/s
retains the full margin. This does not authorize crossing the body-envelope part
of cream. Four added regression tests pass; simulator confirmation is pending.

### Terrain colours: exact current conditions

- **Orange:** known local ground slope greater than 35 degrees. No extra
  slope buffer is added. This is the user-reported ceiling with some roughness,
  not a guarantee of traction below 35 degrees.
- **Red:** a detected hazard (unless the same cell is orange for excessive slope).
  Hazard detection requires a measured cell and at least six observed cells in
  its 11-by-11 neighbourhood. It flags slope-corrected protrusion above the
  configured step threshold with supporting compactness, within-cell excess
  relief above that threshold, or an unexplained neighbouring height step above
  that threshold. The default threshold is 0.20 m (`--step_thresh`). The compact
  protrusion test additionally requires more than half that threshold in its
  evaluated opposite-direction comparisons.
- **Cream:** a non-hazard, non-excessive-slope cell within footprint radius plus
  0.10 m of a detected hazard. This includes BOTH the body envelope and the extra
  planning margin. Only the extra margin can be used for checked slow movement.
  The footprint radius is computed at startup from maximum horizontal body-origin
  distance from the base plus 0.10 m, with a 0.45 m minimum; read the startup
  `envelope=` value for the actual run.
- **Blue/teal:** known, unblocked terrain with cost below 2.
- **Yellow:** known, unblocked terrain with cost from 2 up to 6.
- **Magenta:** known, unblocked terrain with cost at least 6. It is traversable.
- **Grey/purple:** unknown terrain; not authorized for travel.

Unblocked cost is `1 + 4*slope + 4*proximity + roughness_penalty`, where slope is
rise/run, proximity is `max(0, 1 - obstacle_distance/(radius + 0.10 + 0.60))`, and
roughness penalty is `min(4, 10*max(0, roughness_metres - 0.06))` (zero when roughness
is unavailable). These colours are difficulty estimates, not measured grip.
Freshness gates immediate movement separately: an observation older than eight
simulation seconds does not erase the survey, but cannot authorize local movement.


### Off-centre crater-floor avoidance obstacle

The demo bowl now contains a deterministic boulder at tile-centred (1.5, 0.8) m,
0.85 by 0.65 m across and 0.45 m above its local terrain. It is added to the existing
heightfield, so render geometry, collision geometry and terrain ray sensors share
it. Its config fields are `floor_boulder_xy`, `floor_boulder_radii` and
`floor_boulder_height` in `LunarCraterDemoBowlCfg`. Set height to zero for a matched
comparison. No friction, policy, mission destinations or spawn parameters changed.
Restart the simulator to regenerate terrain; an already running stage is unchanged.

Validation: 78 navigation/map tests and 6 perception tests pass. New checks cover
34.9/35.1-degree slopes, adjacent unbuffered gentler cells, distant orange-tile
preservation, and sensed detection plus planned avoidance of the new boulder.
These are automated geometry/navigation tests; no new Isaac Sim traversal has
been performed for this change.
