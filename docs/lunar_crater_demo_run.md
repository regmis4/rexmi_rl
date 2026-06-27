# Lunar Crater Demo — Run Guide

> **Purpose:** Investor showcase — Go2W traversing LOLA-calibrated lunar south pole
> crater terrains in Isaac Sim.
>
> **Phase 7 policy (rough-terrain):** `go2w_velocity_rough/2026-06-14_20-03-41` — handles Type 3 wall cross-section.
>
> **Phase 8 policy (steep-slope):** `go2w_velocity_steep_slope/2026-06-20_15-37-32` — required for Bowl demo (25–35°) and best for Type 3 wall (31°).

---

## Quick Reference

| Terrain | Description | Slope Range | Recommended policy | Play env ID | Record env ID |
|---------|-------------|-------------|-------------------|-------------|---------------|
| Type 3 | Shackleton wall cross-section | 22.5–35° | ✅ Phase 8 (steep-slope) | `RexmiRl-Go2w-Crater-Type3-Play-v0` | `RexmiRl-Go2w-Crater-Type3-Record-v0` |
| **🌑 Bowl** | **Full Shackleton-class crater + mountain** | **25–35° + geology** | **✅ Phase 8 (steep-slope)** | `RexmiRl-Go2w-Crater-Bowl-SteepSlope-Play-v0` | `RexmiRl-Go2w-Crater-Bowl-SteepSlope-Record-v0` |

**Bowl also supports rough policy** via `RexmiRl-Go2w-Crater-Bowl-Play-v0` (same env, Go2wRoughPPORunnerCfg runner).  
**Flat / FastFlat bowl variants** registered for future use — see [Policy Selection](#policy-selection) below.

**Episode length:** 300 s (5 min) — 4–5 full traversals for cross-section tiles; ~5 complete bowl crossings.

**Policy checkpoints:**
```
# Phase 7 (Type 3 wall cross-section with rough policy):
logs/rsl_rl/go2w_velocity_rough/2026-06-14_20-03-41/model_8996.pt

# Phase 8 (Type 3 wall, Bowl demo — recommended):
logs/rsl_rl/go2w_velocity_steep_slope/2026-06-20_15-37-32/model_5998.pt
```

> **How `--load_run` works:** Isaac Lab automatically prepends
> `logs/rsl_rl/<experiment_name>/` to whatever you pass as `--load_run`.
> Pass **only the date folder** — Isaac Lab auto-selects the latest `.pt` in it.
>
> ⚠️ **Use `python scripts/play.py` directly — do NOT use `./run.sh`.**
> `run.sh` has a known PYTHONPATH bug that shadows the installed `rsl_rl` package
> and causes import errors.  All commands below use `python scripts/play.py`.

---

## Demo Commands

### 🌑 Bowl — Full Radial Crater (HEADLINE PHASE 8 DEMO)

The bowl is a **complete axisymmetric crater** at robot scale — 22 m diameter,
4.4 m deep — with geologically realistic surface features and an adjacent
mountain backdrop. The robot traverses straight through: spawns outside, descends
into the crater, reaches the floor, climbs out the opposite side, and exits.
**The 64 m × 64 m tile also shows a 27° mountain (NE corner, 20 boulders) as a visual backdrop.**

```bash
# PRIMARY: 10 robots — steep-slope policy (25–35° bowl + mountain backdrop)
python scripts/play.py \
    --task RexmiRl-Go2w-Crater-Bowl-SteepSlope-Play-v0 \
    --load_run 2026-06-20_15-37-32
```

```bash
# Single-robot recording (best for investor video)
python scripts/play.py \
    --task RexmiRl-Go2w-Crater-Bowl-SteepSlope-Record-v0 \
    --load_run 2026-06-20_15-37-32
```

```bash
# Alt: same bowl, rough policy runner (if loading rough checkpoint)
python scripts/play.py \
    --task RexmiRl-Go2w-Crater-Bowl-Play-v0 \
    --load_run 2026-06-14_20-03-41
```

**What to expect:**

Each 52-second crossing follows this sequence:

| Phase | Distance | Slope | What the robot does |
|-------|----------|-------|---------------------|
| Exterior approach | 2 m | −8° (downhill) | Descends toward crater rim |
| Rim crest entry | 1 m | 23° | Crosses into the crater |
| Upper wall descent | 2 m | **33°** | Steepest zone — Phase 8 working |
| Main wall descent | 4 m | **30°** | Sustained steep slope, boulders |
| Scarp (at r=7 m) | 0.3 m | step | 0.18 m ridge step — robot adapts |
| Lower wall / transition | 4 m | **20°** | Easing toward floor |
| Flat floor | 6 m | 0° | Brief flat, debris apron at r≈3.8 m |
| Opposite ascent | 7 m | 20°→33° | Mirrored challenge uphill |
| Rim exit | 1 m | 23° | Exits crater |
| Exterior exit | 2 m | 8° (uphill) | Decelerates, episode continues |

**Surface features (visible during traversal):**
- **10 crater-wall boulders** (0.15–0.50 m) on inner wall — robot steps around/over
- **8 exterior boulders** (0.15–0.65 m) on rim exterior — visible at spawn
- **2 fracture troughs** (concentric rings at r=5.5 m and r=8.0 m)
- **Mid-wall scarp ridge** (0.18 m at r=7.0 m) — visible as circumferential bump
- **Mountain backdrop** (27° slope, 4.1 m tall, 20 boulders) — NE corner of tile
- **Azimuthal variation** — each of the 10 parallel robots sees a different slope profile (±5° from nominal)

**Demo narrative:**
> "We've generated a 22-metre Shackleton-class crater with geologically accurate
> surface features — boulders, fractures, mass-wasting scarps — derived from LROC
> camera data of the actual Shackleton interior.  The Phase 8 policy navigates all of
> it in real time, from approach to exit, at 0.5 m/s.  The mountain in the background
> matches the same 25–30° slope regime — that's the next traversal target."

---

### Type 3 — Shackleton wall cross-section (Phase 8 policy ✅)

```bash
# UPSLOPE: 10 robots — floor → rim
python scripts/play.py \
    --task RexmiRl-Go2w-Crater-Type3-Play-v0 \
    --load_run 2026-06-20_15-37-32
```

```bash
# DOWNSLOPE: 10 robots — rim → floor
python scripts/play.py \
    --task RexmiRl-Go2w-Crater-Type3-Down-Play-v0 \
    --load_run 2026-06-20_15-37-32
```

**What to expect:**
Classic cross-section traversal on Shackleton-calibrated wall slopes (31° main wall).
Phase 8 policy handles the full traverse. Height gain ~17 m over 32 m.

---

## Policy Selection

The bowl demo is registered under four env IDs matching the four policy types.
Each pair (Play + Record) uses the **same env** (same terrain, same obs space),
only the `rsl_rl_cfg_entry_point` differs (which determines the network architecture).

| Env suffix | Runner cfg | Obs space | Use with |
|:-----------|:-----------|:---------:|----------|
| `Bowl-Play-v0` | `Go2wRoughPPORunnerCfg` | ~208-dim | rough checkpoint |
| `Bowl-SteepSlope-Play-v0` | `Go2wSteepSlopePPORunnerCfg` | ~208-dim | steep-slope checkpoint ✅ |
| `Bowl-Flat-Play-v0` | `Go2wFlatPPORunnerCfg` | ~48-dim | flat checkpoint† |
| `Bowl-FastFlat-Play-v0` | `Go2wFastFlatPPORunnerCfg` | ~48-dim | fast-flat checkpoint† |

> **† Obs-space note:** `Flat` and `FastFlat` use a different env class
> (`LunarCraterDemoBowlFlatEnvCfg` — no height scanner, ~48-dim obs).
> Loading a rough/steep checkpoint into these envs **will fail** with a
> network shape mismatch.  Conversely, you cannot load a flat checkpoint
> into the `Bowl-SteepSlope-Play-v0` env.  The split is intentional:
> flat/fastflat variants are ready stubs for when those policies are
> retrained with a larger observation space that includes the height scanner.
>
> **For the immediate demo: use `Bowl-SteepSlope-Play-v0` with the Phase 8
> steep-slope checkpoint — no obs-space issues.**

---

## Terrain Configuration Reference

| Parameter | Type3 tile | Bowl tile |
|-----------|:-----------:|:---------:|
| Tile size | 32 m × 32 m | 64 m × 64 m |
| Resolution | 10 cm/pixel | 10 cm/pixel |
| Border width | 20 m (flat) | 20 m (flat) |
| Num envs (Play) | 10 | 10 |
| Num envs (Record) | 1 | 1 |
| Env spacing | 34 m | 66 m |
| Episode length | 300 s | 300 s |
| Velocity command | 0.5 m/s forward | 0.5 m/s forward |
| Spawn yaw (Type3 up) | 0 rad | — |
| Spawn yaw (Bowl) | — | π rad (faces −x = into crater) |
| Spawn yaw (Type3 down) | π rad | — |

---

## Spawn Height Corrections

The terrain height at spawn differs from the tile centre (where `env_origin_z` is set).
Each config applies a `z` correction to `reset_base.params["pose_range"]`:

| Config | Spawn location | Terrain Δh vs. centre | z correction |
|--------|---------------|----------------------|-------------|
| Type 3 upslope | floor (x=3 m, h≈0.16 m) | −6.58 m | `(-6.5, -5.9)` |
| Type 3 downslope | rim (x=29 m, h≈14.17 m) | +7.43 m | `(+6.9, +7.7)` |
| **Bowl** | exterior ramp (r=13 m, h≈4.12 m) | **+4.12 m** | `(+4.0, +4.9)` |

> Bowl note: env_origin_z ≈ 0 m (crater floor = global minimum after height shift).
> Spawn z = h_exterior + body_clearance ≈ 4.12 + 0.35 = 4.47 m above env_origin.
> Range widened to (4.0, 4.9) to accommodate ±5° azimuthal slope variation and boulders.

---

## Bowl — Radial Zone Breakdown

The bowl is axisymmetric — zones are defined by radial distance (r) from crater centre.
Each crossing traverses the zones twice (inbound + outbound).

| Zone | Radial range | Base slope | Surface features | Height gain per half |
|------|:------------:|:----------:|------------------|:--------------------:|
| Exterior ramp | r = 11–32 m | −8° (descent) | 8 exterior boulders (0.15–0.65 m) | −0.70 m |
| Rim ease-off | r = 10–11 m | 23° | Rim blocks, coarser debris | +0.42 m |
| Upper wall | r = 8–10 m | **33°** | Boulders, scarp at r=7 m | +1.30 m |
| Main wall | r = 4–8 m | **30°** | 10 boulders, fracture at r=8 m | +2.31 m |
| Transition | r = 3–4 m | 20° | Fracture at r=5.5 m, debris apron | +0.36 m |
| Floor | r = 0–3 m | 0° | Loose debris, higher roughness | 0 m |
| **Total depth (floor→rim)** | | | | **~4.4 m** |

**Mountain (NE backdrop, +10 m east / +24 m north of crater centre):**

| Feature | Value |
|---------|-------|
| Base radius | 8 m |
| Peak height | 4.1 m |
| Slope | ~27° (arctan 4.1/8) |
| Boulder count | 20 (0.20–0.70 m, larger than crater-wall blocks) |
| Distance from crater rim | ~5 m clearance |

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Robots not visible (downslope) | z correction missing → spawning underground | Check `pose_range["z"]` in each Down config |
| Robots fall and tip on spawn | z correction too small → spawning too high | Decrease `z` range (more negative for upslope) |
| Robots reset every 30 s | `episode_length_s` too short | Set `self.episode_length_s = 300.0` in base class |
| All robots face random directions | `yaw` not fixed | Set `"yaw": (0.0, 0.0)` in `reset_base` |
| Robots scatter sideways | `lin_vel_y` not zeroed | Set `lin_vel_y = (0.0, 0.0)` in commands |
| `network shape mismatch` error | Wrong env ID for the checkpoint | See Policy Selection table above |
