# Rough policy: sideways-motion assessment

The existing rough checkpoint can sidestep and move diagonally on flat and lightly
textured terrain. It is a useful candidate for short lateral route corrections.
It is not yet qualified for automatic policy transitions or obstacle-constrained
sidestepping. The default checkpoint navigator remains rocky-slope only.

## What the saved training configurations actually contain

| Checkpoint | Forward command | Sideways command | Yaw command |
|---|---|---|---|
| rough `model_8996.pt` (2026-06-14_20-03-41) | -0.5 to +0.5 m/s | -0.5 to +0.5 m/s | -1 to +1 rad/s |
| rocky_slope `model_13994.pt` (2026-06-30_09-31-48) | +0.2 to +0.5 m/s | zero | -0.5 to +0.5 rad/s |

These are saved training ranges, not promises of achieved speed or safe operating
limits. Rocky-slope has yaw training, so its wide turns are not simply caused by
an absence of turn commands in training. It has no lateral-command training.
Successful reverse motion reported in teleop is outside its saved forward range.

Both checkpoints use 247 actor inputs, 16 outputs, and the same action ordering
and scales: wheels 10, thighs 0.5, hips 0.3, calves 0.5. This allows testing both
in the same observation/action environment; it does not validate transitions.
The shared command adapter previously forced lateral velocity to zero. It now
accepts an explicit keyword `vy`, defaulting to zero. Autonomous navigation and
existing teleop controls continue to send zero sideways velocity.

## Measured behaviour

72 cases: two policies × two surfaces × six commands × three seeds (42–44).
Each case settled for 2 s, drove for 6 s, then received zero commands for 4 s.
Surfaces were flat or a nominally flat heightfield with 4 cm peak-to-peak texture.
Commands covered left/right at 0.20 m/s, diagonal forward/sideways at 0.30/0.20 m/s,
forward at 0.30 m/s and yaw at 0.35 rad/s. The study used the navigation robot and
observation configuration, fixed robot friction 0.8 static / 0.6 dynamic, and
removed random pushes and mass variation. Demo physics and policies were not edited.

| Pure sideways cases | Sideways displacement in 6 s | Largest forward drift | Largest heading drift | Largest post-command drift |
|---|---:|---:|---:|---:|
| Rough, flat | 1.04–1.08 m | 0.32 m | 7.9° | 0.41 m |
| Rough, textured | 0.68–0.97 m | 0.34 m | 18.0° | 0.27 m |
| Rocky-slope, flat | 0.03–0.09 m | 0.22 m | 9.1° | 0.02 m |
| Rocky-slope, textured | 0.00–0.07 m | 0.40 m | 14.7° | 0.07 m |

No tip events (body up-vector below 0.5), interventions or automatic resets occurred.
Flat repeats were effectively deterministic; texture seeds provide the main terrain
variation. Post-command drift is endpoint displacement over four seconds, not a
certified stopping-distance envelope; peak excursion may be larger. Lateral motion
was not classified by contact mechanics, so these measurements alone do not prove
which contacts stepped or slid, lower energy use, or better obstacle clearance.

Results and checkpoint hashes: [machine-readable summary](validation/lateral_study_summary.json).
Full local samples: `logs/nav/lateral_study/seed42.json`, `seed43.json`, `seed44.json`.

## How to use this in navigation

Use short lateral corrections on verified near-level ground, especially when the
next route segment is slightly sideways and a large body rotation is unnecessary.
Keep rocky-slope throughout slopes. A future integration should:

1. Select a sidestep only when its full swept footprint, observed stopping region
   and destination are clear. Include measured forward creep, not just lateral travel.
2. Confirm the robot is stopped on observed near-level ground before switching.
   Validate a conservative slope gate in transition trials; these tests only cover
   nominally level terrain, so no nonzero switching-slope limit is established.
3. Apply heading feedback and forward-drift correction while following the route.
4. Keep the same STOP/manual override, map restrictions and mission order. Hold the
   chosen policy through the manoeuvre rather than switching every control tick.
5. Test braking, both switch directions and obstacle detours before enabling it.

The current measurements support a development experiment, not default autonomous
sidestepping. No automatic switching, slope switching or new navigation mode is
introduced by this assessment.

## Reproduce

```bash
cd /home/susan/rexmi_rl
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgcc_s.so.1 ./run.sh scripts/lateral_study.py \
  --rough logs/rsl_rl/go2w_velocity_rough/2026-06-14_20-03-41/model_8996.pt \
  --rocky logs/rsl_rl/go2w_velocity_rocky_slope/2026-06-30_09-31-48/model_13994.pt \
  --seed 42 --output logs/nav/lateral_study/seed42.json
```

Repeat with seeds 43 and 44 and distinct output filenames. This standalone headless
study does not start the mission navigator or change saved training configurations.
