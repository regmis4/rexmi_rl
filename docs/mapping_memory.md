# Persistent scouting map

Previously, each scan replaced the height bounds of every touched 20 cm cell.
As the robot moved, rays hit different parts of the cell, changing its apparent
height, slope and obstacle classification. An age penalty also changed terrain
colours even when there were no new measurements.

The map now accumulates evidence in 5 cm bins within each planning cell. New
views add to the map instead of replacing the last view. Repeated compatible
returns are averaged. Higher returns enter immediately for obstacle avoidance;
clearing a previously observed top requires three consistent scans at that same
bin. Missing returns do not clear it. This also allows a false high return to be
corrected instead of preserving every maximum forever.

Terrain cost and observation freshness are separate. Age alone does not change
terrain colours, heights or the measured area. A short-range freshness check
still applies before driving, so stored knowledge is not treated as proof that
the immediate corridor has no new obstacle.

The display reports mapped local coverage, fresh local coverage and total
surveyed area. Coverage includes supported small sampling gaps; surveyed area
counts directly measured 20 cm cells. These are coverage measures, not calibrated
probabilities of safety. Viewport rendering remains local and budgeted, while
the full mission map stays in memory and is exported to `.map.npz` at normal exit.

## Checks performed

35 checkpoint/mapping tests and 4 perception geometry tests pass. New regressions
cover age-only stability, partial-cell viewpoints, retention of traversed terrain,
confirmation before obstacle clearing and retention with missing returns.

A 40-second Isaac Sim run recorded 200 world-frame sensor batches while the
robot moved from the rim toward the slope. The same batches were replayed through
the previous and new map implementations, using identical map geometry and
0.503 m footprint radius.

| Measurement | Previous map | Accumulated map |
|---|---:|---:|
| Terrain colour-band changes in cells known for at least 2 s | 10,934 | 1,294 |
| Fraction of eligible cell/frame comparisons that changed band | 0.782% | 0.093% |

This is **88% fewer colour-band changes** on that replay. It measures temporal
stability, not obstacle-detection accuracy. Some remaining changes are expected
when new views reveal terrain or refine the local ground estimate.

Directly measured area increased monotonically from 52.44 to 458.40 m². Stored
heights of cells receiving no new observations remained unchanged. A separate
age-only test verifies exact preservation of terrain costs and heights.

The recording run reached its 40-second limit without completing its navigation
destination. It recorded a hold failure; this mapping result does not establish
full crater navigation or braking reliability. Those remain separate validation
items in [the navigation record](checkpoint_validation.md).

To record another dataset, add `--scan_log /tmp/rexmi-scans` to the command in
[the launch guide](checkpoint_navigation.md). Each compressed batch records the
simulation timestamp, world-frame hits, sensor origins and range gates.

## Renderer follow-up

Terrain tiles now keep stable world-cell-to-face assignments, fixed topology,
opaque active faces and explicit colour/opacity indices. This addresses the
reported missing-opacity-indices warning and avoids reshuffling face identities
when nearest-cell order changes. See [the visible test record](spawn_stall_validation.md).

## Terrain versus clearance

Red now identifies detected hazard cells or terrain above the configured slope
limit. The surrounding footprint/clearance buffer is shown in muted amber, with
an explicit “planner avoids” legend in both views. This changes display
semantics, not the blocked mask or the physical capability assumptions.
The obstacle-height threshold remains configurable at 0.20 m and the slope
limit remains 50 degrees pending capability testing. In the last saved
spawn-checkpoint map, 344 of 375 observed blocked cells were clearance buffer;
only 31 were detected hazard/steep-terrain cells.
