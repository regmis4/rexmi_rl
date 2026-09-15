# REXMI live perception view

Add `--perception_view` to your existing `scripts/navigate.py` command.
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
  --policy_mode rocky_slope --no_turn \
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
test coverage; full autonomous mission validation is separate. The existing
debug-arrow callback may report a CPU/CUDA mismatch during simulator shutdown.

## What appears

The **REXMI | Perception** window toggles four independent scene layers:

- **Live LiDAR returns:** cyan points from the current simulated LiDAR scan.
- **Observed terrain costs:** small terrain-following tiles within 12 m of the robot.
  Teal = low planner cost, amber = elevated cost, orange = high cost,
  red = blocked. Costs include the planner's obstacle inflation.
- **Route on observed terrain:** cyan route segments lifted above measured terrain.
  Gaps indicate that a path segment has no observed height; it is not drawn at
  an invented elevation. The 2D dashboard still shows the complete planned route.
- **Sampled LiDAR rays:** at most 32 rays from the sensor origin, off by default
  to keep the scene readable. These illustrate scan geometry, not every beam.

Use the normal Isaac Sim viewport camera to orbit/zoom. For a presentation,
start with costs and route enabled, then toggle points/rays to explain sensing.
The panel reports the actual number of displayed returns, tiles and route segments.

## Data and performance

This is a visualization of the existing stack, not a new perception algorithm.
It uses the Isaac Lab RayCaster LiDAR's world-frame hits, not an RTX LiDAR or
validated flight-sensor model. The existing navigation localizer remains
simulator-assisted. The accumulated SLAM map is deliberately not overlaid as if
it were identical to the live scan; those can differ due to localization drift.

Unknown cells are omitted. Planner cost colors are not safety guarantees or
resource-composition classifications. The height of each tile comes from the
map's observed maximum cell height; it is a discrete cell visualization, not a
reconstructed continuous surface. Vertical walls and steep discontinuities may
show raised or partially occluded tiles.

Refresh is capped at 5 Hz, 6,000 returns and 4,000 cost cells. Geometry updates
run on the simulation thread. No physics/collision components are added and no
controller, checkpoint, sensor configuration or terrain asset is changed.
USD geometry lives under `/RexmiPerception` in the session layer and is removed
on close; it is not saved to the robot or terrain source assets.

CPU geometry checks:

```bash
python -m unittest discover -s tests -p test_perception_view.py -v
```
