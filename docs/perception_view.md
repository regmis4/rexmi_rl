# REXMI live perception view

Add `--perception_view` to your existing `scripts/navigate.py` command.
The default is now checkpoint navigation; see [the launch and control guide](checkpoint_navigation.md).
Keep your normal task, checkpoint and mission arguments. Add `--no_dashboard`
if you only want the Isaac Sim scene instead of the separate 2D plots.

## Launch command

Run from the repository root. This opens Isaac Sim with the existing rocky-slope
checkpoint and the live perception panel:

```bash
cd /home/susan/rexmi_rl
./run.sh scripts/navigate.py \
  --task RexmiRl-Go2w-Crater-Bowl-RockySlope-Play-v0 \
  --checkpoint logs/rsl_rl/go2w_velocity_rocky_slope/2026-06-30_09-31-48/model_13994.pt \
  --nav_controller checkpoint \
  --perception_view --no_dashboard \
  --max_steps 15000
```

Close the simulator window or press Ctrl+C in the launching terminal to stop.
Remove `--no_dashboard` to also display the existing 2D costmaps.

On this workstation, if an RTX extension reports a missing `GCC_12.0.0` symbol,
prefix the launch command with
`LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgcc_s.so.1 ./run.sh ...`.
This workaround applies only to that process; it does not change the environment.

The short live check verified LiDAR and cost rendering. Route geometry has CPU
test coverage; full autonomous mission validation is separate. Checkpoint navigation disconnects the debug-arrow callback before shutdown;
legacy mode may still report its previous CPU/CUDA callback warning.

## What appears

The **REXMI | Perception** window toggles four independent scene layers:

- **Live LiDAR returns:** cyan points from the current simulated LiDAR scan.
- **Observed terrain costs:** the complete observed mission survey, with detailed nearby tiles and aggregated distant tiles.
  Teal = low planner cost, amber = elevated cost, orange = high cost,
  red = detected obstacle / very steep terrain. Muted amber marks the
  clearance buffer, which the planner avoids but which is not itself an obstacle.
- **Route on observed terrain:** red route segments lifted above measured terrain.
  Gaps indicate that a path segment has no observed height; it is not drawn at
  an invented elevation. The 2D dashboard still shows the complete planned route.
- **Sampled LiDAR rays:** at most 32 rays from the sensor origin, off by default
  to keep the scene readable. These illustrate scan geometry, not every beam.

Use the normal Isaac Sim viewport camera to orbit/zoom. For a presentation,
start with costs and route enabled, then toggle points/rays to explain sensing.
The panel reports the actual number of displayed returns, tiles and route segments.
Checkpoint mode also separates mapped local coverage, fresh local coverage and
total surveyed area. Terrain colours do not change merely because a cell ages.
See [persistent scouting map checks](mapping_memory.md).

## Data and performance

This overlay visualizes the selected navigation controller.
It uses the Isaac Lab RayCaster LiDAR's world-frame hits, not an RTX LiDAR or
validated flight-sensor model. The navigation localizer remains
simulator-assisted. Checkpoint navigation uses live world-frame observations and does not feed the
accumulated SLAM cloud back into its map. Legacy navigation retains its old mapping.

Unknown cells are omitted. Planner cost colors are not safety guarantees or
resource-composition classifications. The height of each tile comes from the
map's accumulated upper cell elevation; it is a discrete cell visualization, not a
reconstructed continuous surface. Vertical walls and steep discontinuities may
show raised or partially occluded tiles.

Refresh is capped at 5 Hz and 6,000 live returns. Checkpoint mode retains the whole
survey display using stable chunks: 20 cm nearby tiles and 40 cm distant tiles.
Legacy mode retains its earlier 4,000-cell local view. Geometry updates
run on the simulation thread. The overlay adds no physics/collision components and changes no
trained checkpoint, sensor configuration or terrain asset.
USD geometry lives under `/RexmiPerception` in the session layer and is removed
on close; it is not saved to the robot or terrain source assets.

CPU geometry checks:

```bash
python -m unittest discover -s tests -p test_perception_view.py -v
```

Checkpoint mode also provides Terrain, Slope, Roughness and Traversal buttons.
Use `--survey_layer slope` to select an initial diagnostic layer. See the
[whole-survey guide](terrain_survey.md) for legends, route learning and exports.
