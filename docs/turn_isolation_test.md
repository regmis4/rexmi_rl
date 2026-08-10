# Turn isolation test (Option A Phase 1)

## Goal

Prove whether **model_13345** can plant-and-spin on the **crater** env with a
**clean start** (no rocky handoff, no nav FSM).

| Result | Meaning |
|--------|---------|
| **PASS** | Policy OK on crater → fix **nav handoff** (Phase 2A) |
| **FAIL** | Domain gap → finetune or path-only demo (Phase 2B) |

## Spawn presets

| Preset | Position | Terrain |
|--------|----------|---------|
| `floor` (default) | (0, 0, 1.15) | Crater floor, rough mesh |
| `mid_slope` | (6.5, 0, 2.40) | Bowl wall slope |
| `rim_out` | (13, 0, 4.50) | Exterior ramp |

## Run

```bash
conda activate env_isaacsim
cd /home/susan/rexmi_rl

python scripts/test_turn_crater.py \
  --task RexmiRl-Go2w-Crater-Bowl-RockySlope-Play-v0 \
  --checkpoint logs/rsl_rl/go2w_velocity_slope_turn/2026-07-27_20-54-25/model_13345.pt

# Headless:
python scripts/test_turn_crater.py --task ... --checkpoint ... --headless

# Mid wall:
python scripts/test_turn_crater.py ... --spawn_preset mid_slope

# Exterior:
python scripts/test_turn_crater.py ... --spawn_preset rim_out
```

## Schedule

| Phase | Duration | Command |
|-------|----------|---------|
| SETTLE | 2 s | vx=0.05 ω=0 |
| YAW+ | 6 s | vx=0.05 ω=+0.07 |
| PLANT | 1 s | vx=0.05 ω=0 |
| YAW− | 6 s | vx=0.05 ω=−0.07 |
| PLANT | 1 s | vx=0.05 ω=0 |

## Pass criteria (each YAW phase)

- |Δbody_yaw| ≥ 25°
- mean |ω_body| ≥ 0.03 rad/s
- up_z min ≥ 0.50
- peak |ω_body| < 2.5 (no thrash)

## After PASS — Phase 2A handoff (in navigate)

```
FOLLOW until |he|>150° AND v<0.08 AND up_z>0.85
  → BRAKE (rocky, vx=0) until stopped
  → settle actions 0.4 s (zeros)
  → force_turn + PLANT/YAW/SETTLE (13345, vx=0.05 ω=±0.07)
  → release_turn + short hold → FOLLOW
```

Enable with:

```bash
python scripts/navigate.py ... --enable_turn
```

## Logs

CSV: `logs/nav/turn_iso_<timestamp>.csv`

---

## After PASS — full nav command

```bash
conda activate env_isaacsim
cd /home/susan/rexmi_rl

python scripts/navigate.py \
    --task RexmiRl-Go2w-Crater-Bowl-RockySlope-Play-v0 \
    --ckpt_rough logs/rsl_rl/go2w_velocity_rough/2026-06-14_20-03-41/model_8996.pt \
    --ckpt_rocky logs/rsl_rl/go2w_velocity_rocky_slope/2026-06-30_09-31-48/model_13994.pt \
    --ckpt_turn  logs/rsl_rl/go2w_velocity_slope_turn/2026-07-27_20-54-25/model_13345.pt \
    --mission traverse
```

Turn is ON by default. Disable with `--no_turn`.
