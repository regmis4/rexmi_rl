# Lunar Crater Demo — Run Guide

> **Purpose:** Investor showcase — Go2W traversing LOLA-calibrated lunar south pole
> crater wall terrains in Isaac Sim.  No re-training required; uses the existing
> Phase 6-Optimized rough-terrain policy checkpoint.

---

## Quick Reference

| Crater Type | Archetype | Slope Range | Policy | Upslope env ID | Downslope env ID |
|-------------|-----------|-------------|--------|----------------|------------------|
| Type 1 | Haworth / Shoemaker | 7.5–14° | ✅ All zones | `RexmiRl-Go2w-Crater-Type1-Play-v0` | `RexmiRl-Go2w-Crater-Type1-Down-Play-v0` |
| **Type 2** | **Faustini** | **11.5–17.5°** | **✅ PRIMARY** | `RexmiRl-Go2w-Crater-Type2-Play-v0` | `RexmiRl-Go2w-Crater-Type2-Down-Play-v0` |
| Type 3 | Shackleton | 22.5–35° | ⚠️ Phase 8 | `RexmiRl-Go2w-Crater-Type3-Play-v0` | `RexmiRl-Go2w-Crater-Type3-Down-Play-v0` |

**Traversal directions:**
- **Upslope** (`-Play-v0`): robot spawns at crater floor (tile x ≈ 2–4 m), faces `+x` (floor→rim), `yaw=0`
- **Downslope** (`-Down-Play-v0`): robot spawns near rim (tile x ≈ 28–30 m), faces `-x` (rim→floor), `yaw=π`
- Both directions command `lin_vel_x = +0.5 m/s` in body frame (body forward = traverse direction)

**Episode length:** 300 s (5 min) — enough for 4–5 full traversals per episode.

**Latest policy checkpoint (Phase 6-Optimized):**
```
logs/rsl_rl/go2w_velocity_rough/2026-06-14_20-03-41/model_8996.pt
```

> **How `--load_run` works:** Isaac Lab automatically prepends
> `logs/rsl_rl/<experiment_name>/` to whatever you pass as `--load_run`.
> Because `experiment_name = "go2w_velocity_rough"`, use **only the date folder**
> (e.g. `2026-06-14_20-03-41`) — not the full path.
> Isaac Lab then auto-selects the latest `.pt` in that folder (`model_8996.pt`).
>
> If you need to pin a specific checkpoint, pass the **full absolute path** to
> `--checkpoint` — bare filenames (e.g. `--checkpoint model_8996.pt`) will fail.
>
> ⚠️ **Use `python scripts/play.py` directly — do NOT use `./run.sh`.**
> `run.sh` has a known PYTHONPATH bug that shadows the installed `rsl_rl` package
> and causes import errors.  All commands below use `python scripts/play.py`.

---

## Demo Commands

### Type 2 — Faustini (PRIMARY INVESTOR DEMO)

```bash
# UPSLOPE: 10 robots traverse from crater floor → rim
python scripts/play.py \
    --task RexmiRl-Go2w-Crater-Type2-Play-v0 \
    --load_run 2026-06-14_20-03-41
```

```bash
# DOWNSLOPE: 10 robots descend from rim → crater floor
python scripts/play.py \
    --task RexmiRl-Go2w-Crater-Type2-Down-Play-v0 \
    --load_run 2026-06-14_20-03-41
```

```bash
# Single-robot recording — upslope (best for video capture)
python scripts/play.py \
    --task RexmiRl-Go2w-Crater-Type2-Record-v0 \
    --load_run 2026-06-14_20-03-41
```

```bash
# Single-robot recording — downslope
python scripts/play.py \
    --task RexmiRl-Go2w-Crater-Type2-Down-Record-v0 \
    --load_run 2026-06-14_20-03-41
```

```bash
# Or with explicit full checkpoint path (if you need a specific model):
python scripts/play.py \
    --task RexmiRl-Go2w-Crater-Type2-Record-v0 \
    --checkpoint /home/susan/rexmi_rl/logs/rsl_rl/go2w_velocity_rough/2026-06-14_20-03-41/model_8996.pt
```

**What to expect (upslope):**  
Robot spawns at the flat PSR floor (~1.5°) and traverses upslope through lower
wall (11.5°), mid wall (15°), upper wall (17.5°), and over the rim crest (6.5°).
Total height gain: ~6.7 m over 32 m.  All zones handled cleanly.

**What to expect (downslope):**  
Robot spawns near the rim crest and descends — slope decreases progressively
from 17.5° → 15° → 11.5° → 1.5° as it approaches the PSR floor.
Total height drop: ~6.7 m over 32 m.

---

### Type 1 — Haworth / Shoemaker (Gentlest — good opener)

```bash
# UPSLOPE: 10 robots — floor → rim
python scripts/play.py \
    --task RexmiRl-Go2w-Crater-Type1-Play-v0 \
    --load_run 2026-06-14_20-03-41
```

```bash
# DOWNSLOPE: 10 robots — rim → floor
python scripts/play.py \
    --task RexmiRl-Go2w-Crater-Type1-Down-Play-v0 \
    --load_run 2026-06-14_20-03-41
```

**What to expect:**  
Gentle ancient crater wall at 7.5–14°.  Total height gain ~3.9 m over 32 m.
Best-case performance — smooth, confident traversal throughout.
Good choice as opening demo before showing steeper terrain.

---

### Type 3 — Shackleton (Future roadmap — shows capability limit)

```bash
# UPSLOPE: 10 robots — floor → rim (hits limit at 31.5° mid-wall)
python scripts/play.py \
    --task RexmiRl-Go2w-Crater-Type3-Play-v0 \
    --load_run 2026-06-14_20-03-41
```

```bash
# DOWNSLOPE: 10 robots — spawns at 35° rim, struggles immediately
python scripts/play.py \
    --task RexmiRl-Go2w-Crater-Type3-Down-Play-v0 \
    --load_run 2026-06-14_20-03-41
```

**What to expect (upslope):**  
Floor and lower wall (3–22.5°) traversed successfully.  
Robot struggles on mid wall (31.5°) and rim (35°) — this is intentional.  
Use this to demonstrate the Phase 8 roadmap: "here's where we're going next."

**What to expect (downslope):**  
Robot spawns at the 35° rim zone immediately — policy struggles right away.
More dramatic than upslope for showing the Phase 8 capability gap.

---

## Terrain Configuration Reference

| Parameter | Value |
|-----------|-------|
| Tile size | 32 m × 32 m |
| Resolution | 10 cm/pixel (320 × 320) |
| Border width | 20 m (flat) |
| Num envs (Play variant) | 10 |
| Num envs (Record variant) | 1 |
| Env spacing | 34 m |
| Episode length | 300 s (5 min) |
| Velocity command | 0.5 m/s forward in body frame (fixed) |
| Spawn yaw (upslope) | 0 rad — faces +x = floor→rim |
| Spawn yaw (downslope) | π rad — faces −x = rim→floor |
| Spawn x (upslope) | tile x ≈ 2–4 m (crater floor zone) |
| Spawn x (downslope) | tile x ≈ 28–30 m (near rim) |
| Sensor noise | Disabled |
| Random pushes | Disabled |
| Terrain curriculum | Disabled |

---

## Spawn Height Corrections

The terrain height at the floor and rim endpoints differs significantly from the
tile centre (which is where `env_origin_z` is set).  Without correction, robots
would spawn above the terrain on upslope (fall and tip) or underground on
downslope (immediately terminated, invisible).

Each config applies a `z` correction to `reset_base.params["pose_range"]`:

| Config | Spawn location | Terrain Δh vs. centre | z correction |
|--------|---------------|----------------------|-------------|
| Type 1 upslope | floor (x=3 m, h≈0.08 m) | −1.35 m | `(-1.6, -1.0)` |
| Type 2 upslope | floor (x=3 m, h≈0.08 m) | −2.33 m | `(-2.6, -2.0)` |
| Type 3 upslope | floor (x=3 m, h≈0.16 m) | −6.58 m | `(-6.9, -6.3)` |
| Type 1 downslope | rim (x=29 m, h≈4.12 m) | +2.69 m | `(+2.4, +3.2)` |
| Type 2 downslope | rim (x=29 m, h≈6.05 m) | +3.65 m | `(+3.4, +4.2)` |
| Type 3 downslope | rim (x=29 m, h≈14.17 m) | +7.43 m | `(+7.2, +8.2)` |

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Robots not visible (downslope) | z correction missing → spawning underground | Check `pose_range["z"]` in each Down config |
| Robots fall and tip on spawn | z correction too small → spawning too high | Decrease `z` range (more negative for upslope) |
| Robots reset every 30 s | `episode_length_s` too short | Set `self.episode_length_s = 300.0` in base class |
| All robots face random directions | `yaw` not fixed | Set `"yaw": (0.0, 0.0)` in `reset_base` |
| Robots scatter sideways | `lin_vel_y` not zeroed | Set `lin_vel_y = (0.0, 0.0)` in commands |

---

## Terrain Zone Breakdown

### Type 1 — Haworth archetype

| Zone | x range | Slope | Height gain |
|------|---------|-------|-------------|
| PSR floor | 0–6.4 m (20%) | 1.5° | +0.17 m |
| Lower wall | 6.4–17.6 m (35%) | 7.5° | +1.47 m |
| Mid wall | 17.6–25.6 m (25%) | 11.5° | +1.63 m |
| Upper wall | 25.6–30.4 m (15%) | 14° | +1.20 m |
| Rim crest | 30.4–32 m (5%) | 6.5° | +0.18 m |
| **Total** | **32 m** | | **~4.65 m** |

### Type 2 — Faustini archetype (PRIMARY)

| Zone | x range | Slope | Height gain |
|------|---------|-------|-------------|
| PSR floor | 0–4.8 m (15%) | 1.5° | +0.13 m |
| Lower wall | 4.8–16.0 m (35%) | 11.5° | +2.28 m |
| Mid wall | 16.0–25.6 m (30%) | 15° | +2.57 m |
| Upper wall | 25.6–30.4 m (15%) | 17.5° | +1.46 m |
| Rim crest | 30.4–32 m (5%) | 6.5° | +0.18 m |
| **Total** | **32 m** | | **~6.62 m** |

### Type 3 — Shackleton archetype

| Zone | x range | Slope | Height gain |
|------|---------|-------|-------------|
| PSR floor | 0–3.2 m (10%) | 3° | +0.17 m |
| Lower wall | 3.2–9.6 m (20%) | 22.5° | +2.65 m |
| Mid wall | 9.6–27.2 m (55%) | 31.5° | +10.80 m |
| Rim | 27.2–32 m (15%) | 35° | +3.36 m |
| **Total** | **32 m** | | **~16.98 m** |
