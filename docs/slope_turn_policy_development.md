# Slope Turn Policy Development

**Robot:** Unitree Go2W (wheeled quadruped)  
**Goal:** Adapt the flat-terrain pivot-turn gait (model_10992.pt) to slopes up to 35°  
**Status:** 🔄 READY TO TRAIN — Phase SA first

---

## Prerequisite: Flat Turn Best Policy

Before training any slope turn, you need the flat Phase B checkpoint:

- **File:** `logs/rsl_rl/best_policies/go2w_turn_flat_v6_model_10992.pt`
- **Also at:** `logs/rsl_rl/go2w_velocity_turn_b/2026-07-25_13-51-28/model_10992.pt`
- **Capability:** Pivot/spin on flat terrain, micro-tap foot gait, CW and CCW

---

## Why Gradual Curriculum (NOT all-in-one to 35°)

Going straight from flat (0°) to 35° will fail:

1. **Gravity asymmetry** — at 35°, `g_perp = g × sin(35°) ≈ 5.6 m/s²` pulls feet downhill during swing. Uphill and downhill foot contacts have fundamentally different forces.
2. **Instant fall risk at spawn** — robot spawned upright on a 35° slope has its CoM outside the support polygon before the episode even starts.
3. **Gait collapse** — the micro-tap gait was tuned for zero-gravity-asymmetry. A 35° jump causes the policy to abandon it and search for exploits.
4. **No gradient** — if every episode terminates in <5 steps from falling, there's no learning signal.

The 5°→20°→35° curriculum gives the policy a continuous learning path, each step transferable from the last.

---

## Training Sequence

```
model_10992.pt  (flat Phase B — 0° terrain)
        ↓
Phase SA: 5°–20° slopes   →  go2w_velocity_slope_turn_a/  (~500 iters)
        ↓
Phase SB: 15°–35° slopes  →  go2w_velocity_slope_turn/    (~500 iters)
```

Total: ~1000 iterations (~2 hours on RTX 3090/4090).

---

## Phase SA: Gentle Slopes (5°–20°)

**Task:** `RexmiRl-Go2w-Velocity-SlopeTurnA-v0`  
**PPO runner:** `Go2wSlopeTurnAPPORunnerCfg`  
**Logs:** `logs/rsl_rl/go2w_velocity_slope_turn_a/`

### Train
```bash
conda activate env_isaacsim && cd /home/susan/rexmi_rl
python scripts/train.py --task RexmiRl-Go2w-Velocity-SlopeTurnA-v0 --headless \
    --load_run go2w_velocity_turn_b/2026-07-25_13-51-28 \
    --checkpoint model_10992.pt
```

### Visual eval
```bash
python scripts/play.py --task RexmiRl-Go2w-Velocity-SlopeTurnA-Play-v0 \
    --load_run go2w_velocity_slope_turn_a/<date> --checkpoint model_<N>.pt
```

### Terrain
- Pyramid slopes (ascending) + inverted pyramid slopes (descending)
- Angle range: 5°–20° (5 curriculum rows)
- Tile size: 8m × 8m, `platform_width=0.8m` (tiny flat top)

### Spawn setup
- `reset_base x/y: (-2.5, 2.5)` → robots spawn mid-slope, not on flat platform
- Stationary spawn (no velocity) — no slamming
- Joint noise `(0.9, 1.1)` — slight randomisation helps settle on tilted ground

### Expected TensorBoard
| Signal | Expected | Meaning |
|--------|----------|---------|
| `heading_progress_turn` | >0.15 from iter 0 | Flat gait transferred |
| `base_contact` | <5% | Not falling at spawn |
| `bad_orientation` | 5–15% initially, drops | Settling on slope |
| `terrain_levels` | rises to 3–4 | Curriculum advancing |
| `wheel_lock` | low (-1 to -3) | Wheels still locked |
| `pivot_step_coord` | >0.5 | Stepping coordination preserved |

---

## Phase SB: Steep Slopes (15°–35°)

**Task:** `RexmiRl-Go2w-Velocity-SlopeTurn-v0`  
**PPO runner:** `Go2wSlopeTurnPPORunnerCfg`  
**Logs:** `logs/rsl_rl/go2w_velocity_slope_turn/`

### Train (after Phase SA converges)
```bash
# Find best Phase SA checkpoint
ls logs/rsl_rl/go2w_velocity_slope_turn_a/ | sort | tail -1

conda activate env_isaacsim && cd /home/susan/rexmi_rl
python scripts/train.py --task RexmiRl-Go2w-Velocity-SlopeTurn-v0 --headless \
    --load_run go2w_velocity_slope_turn_a/<date> \
    --checkpoint model_<N>.pt
```

### Visual eval
```bash
python scripts/play.py --task RexmiRl-Go2w-Velocity-SlopeTurn-Play-v0 \
    --load_run go2w_velocity_slope_turn/<date> --checkpoint model_<N>.pt
```

### Terrain
- Same structure as Phase SA but 15°–35° (6 curriculum rows)
- Row 0: ~15° (gentle crater approach)
- Row 5: ~35° (Shackleton crater wall gradient)

---

## Key Reward Changes: Flat → Slope

| Reward term | Flat Phase B value | Slope value | Why |
|-------------|-------------------|-------------|-----|
| `flat_orientation_l2` | -3.0 | **0.0** | Body IS tilted on slope — don't penalise |
| `ang_vel_xy_l2` | -0.05 | **-0.05 (LOCKED)** | Lateral sway IS stepping motion — never increase |
| `lin_vel_z_l2` | -1.5 (flat) | **-0.3** | Body moves vertically during cross-slope turns |
| `is_alive` | 0.2 | **0.8** | Stronger survival on harder terrain |
| `position_drift` threshold | 0.25m | **1.5m** | Gravity drift is unavoidable on slope |
| `trunk_stability` max_tilt | 15° | **20°** | Gravity naturally tilts body on slope |

---

## Critical Rule: ang_vel_xy_l2 = -0.05 (DO NOT CHANGE)

This is the single most important constraint learned from flat turn development.

**Why:** During a pivot turn, the stepping motion creates lateral body angular velocity. The micro-tap gait shuffles feet quickly — each shuffle creates a small angular impulse in the lateral direction. `ang_vel_xy_l2` penalises this.

- At -0.05 (Unitree default): lateral motion costs `0.05 × 0.3² = 0.0045/step` — acceptable
- At -0.15 (flat v7 mistake): costs `0.15 × 0.3² = 0.0135/step` — policy suppresses micro-taps, gait collapses

On a slope, the body already has inherent lateral angular velocity from gravity. The penalty fires constantly at any value above -0.05, making the gait impossible.

**Flat v7 result at -0.15:** 85% bad_orientation within 200 iterations. The micro-tap gait completely disappeared.

---

## Spawn-on-Slope Design

### The problem
`HfPyramidSlopedTerrainCfg` has a flat platform at the center (default `platform_width=2.5m`). The base class `reset_base` spawns within `±0.5m` of tile center — ALL robots land on the flat platform. Zero slope training.

### The fix
1. **`platform_width=0.8m`** — tiny flat top, slope starts at 0.4m from center
2. **`reset_base x/y: (-2.5, 2.5)`** — >90% of spawns land on slope face
3. **`velocity_range` all zeros** — no initial velocity, robot settles gently
4. **`bad_orientation` termination at 57°** — enough headroom for 35° slope without instant termination

### Why this doesn't cause slamming
- Robot spawns with zero velocity (not dropped)
- Default joint angles place legs in a stable stance
- The 57° termination threshold means even a 35° slope spawn (body tilted 35° from vertical) has 22° of margin before termination
- Joint noise `(0.9, 1.1)` creates slight variation that helps physics settle

---

## What to Watch During Training

### Good signs (Phase SA, iter 0–100)
- `bad_orientation` starts <20% and falls — robot is surviving on gentle slopes
- `heading_progress_turn` positive from start — flat gait transferred
- `terrain_levels` starts moving after iter 100 — curriculum engaging

### Bad signs (action required)
| Signal | Bad value | Diagnosis | Fix |
|--------|-----------|-----------|-----|
| `bad_orientation` >50% | Any | Robot falling at spawn | Reduce `platform_width` further to 1.0m, or check `reset_base` params |
| `heading_progress_turn` near 0 | iter 0 | Gait not transferring | Check warm-start — wrong base run or wrong checkpoint |
| `ang_vel_xy_l2` penalty huge | Any | Someone changed the weight | Revert to -0.05 |
| `terrain_levels` stuck at 0 | iter 300+ | Curriculum not advancing | Check `curriculum.terrain_levels` — was it disabled by Turn base class? |
| `wheel_lock` penalty large | Any | Policy spinning wheels | `wheel_lock` reward weight may need increasing |

---

## Config Files

| File | Role |
|------|------|
| `source/rexmi_rl/tasks/locomotion/velocity/config/go2w/slope_turn_env_cfg.py` | Phase SA + SB environment configs |
| `source/rexmi_rl/tasks/locomotion/velocity/config/go2w/agents/rsl_rl_ppo_cfg.py` | `Go2wSlopeTurnAPPORunnerCfg`, `Go2wSlopeTurnPPORunnerCfg` |
| `source/rexmi_rl/tasks/locomotion/velocity/config/go2w/__init__.py` | Task registrations |

---

## Registered Tasks

| Task | Class | Use |
|------|-------|-----|
| `RexmiRl-Go2w-Velocity-SlopeTurnA-v0` | `Go2wSlopeTurnAEnvCfg` | Phase SA training |
| `RexmiRl-Go2w-Velocity-SlopeTurnA-Play-v0` | `Go2wSlopeTurnAEnvCfg_PLAY` | Phase SA visual eval |
| `RexmiRl-Go2w-Velocity-SlopeTurn-v0` | `Go2wSlopeTurnEnvCfg` | Phase SB training |
| `RexmiRl-Go2w-Velocity-SlopeTurn-Play-v0` | `Go2wSlopeTurnEnvCfg_PLAY` | Phase SB visual eval |

---

## Navigator Integration (after training)

The slope turn policy becomes the production turn policy for crater traversal:

```
obstacle/slip detected
    ↓
PolicySelector: select TURN policy
    → if flat terrain:  use go2w_turn_flat_v6_model_10992.pt
    → if slope terrain: use slope_turn checkpoint (Phase SB best)
    ↓
turn to new heading
    ↓
resume locomotion policy (rough/steep_rough)
```

The `policy_selector.py` already has a slope detection hook. After Phase SB training, point `--ckpt_slope_turn` to the best checkpoint in navigate.py.


---

## v11 — Systematic fix from model_12349 (2026-07-27)

**Baseline:** `logs/rsl_rl/go2w_velocity_slope_turn_a/2026-07-26_20-09-34/model_12349.pt`  
**Symptom:** rotates on 10° sometimes; intermittent somersault / sideways flip; always microstepping; mild position loss.

### Diagnosis (not vibes)

| Hypothesis | Verdict |
|---|---|
| Friction too low at 10° | **No** — mu=0.7 holds ~35° |
| Raise roughness 2→4 cm | **Wrong first move** — more pivot disturbance |
| Random / absolute heading | **Real** in training; good play diagnostic |
| Reward shaping | **Yes** — drift threshold 0.8 m, trunk 30° too loose |
| Microstepping | **Bought** by `pivot_step_coord` / `foot_alternation` — keep |

### Play diagnostics (run BEFORE trusting a retrain)

Always load model_12349:

```bash
# Baseline (v10e constant sampled yaw ±0.12)
./run.sh scripts/play.py --task RexmiRl-Go2w-Velocity-SlopeTurnA-Play-v0 \
  --load_run go2w_velocity_slope_turn_a/2026-07-26_20-09-34 --checkpoint model_12349.pt

# D1: one direction only (+ then -)
./run.sh scripts/play.py --task RexmiRl-Go2w-Velocity-SlopeTurnA-Play-D1-Pos-v0 \
  --load_run go2w_velocity_slope_turn_a/2026-07-26_20-09-34 --checkpoint model_12349.pt
./run.sh scripts/play.py --task RexmiRl-Go2w-Velocity-SlopeTurnA-Play-D1-Neg-v0 \
  --load_run go2w_velocity_slope_turn_a/2026-07-26_20-09-34 --checkpoint model_12349.pt

# D2: omega=0 station hold
./run.sh scripts/play.py --task RexmiRl-Go2w-Velocity-SlopeTurnA-Play-D2-Hold-v0 \
  --load_run go2w_velocity_slope_turn_a/2026-07-26_20-09-34 --checkpoint model_12349.pt
```

**How to read them**

- D1 flips drop a lot → command reversals / direction-vs-slope asymmetry matters  
- D1 flips stay → continuous-pivot competence / reward / posture  
- D2 holds station → grip OK; problem is turn skill  
- D2 creeps/slides → station-keeping weak even without turning  

Do **not** train on D1/D2 configs. One-sided yaw killed v9-a.

### SA-v11 train (station-keeping only)

One theme. No roughness/friction/omega-range change.

| Change | From → To |
|---|---|
| `position_drift` threshold | 0.8 → **0.35 m** |
| `position_drift` weight | -0.5 → **-1.0** |
| `trunk_stability` | 30° → **20°** |
| terminations | + `bad_pitch`, `bad_roll` (1.4 rad) alongside `bad_orientation` |

```bash
./run.sh scripts/train.py --task RexmiRl-Go2w-Velocity-SlopeTurnA-v0 --headless \
  --load_run go2w_velocity_slope_turn_a/2026-07-26_20-09-34 \
  --checkpoint model_12349.pt
```

**Success (read together):** episode length ↑, `bad_orientation` ↓,  
`Episode_Termination/bad_pitch` vs `bad_roll` tells somersault vs sideways,  
`track_ang_vel_z_exp` stays > 0, visual station tighter, microstep OK if controlled.

### Explicitly NOT in v11

- roughness 4 cm  
- asymmetric ω training  
- killing `pivot_step_coord`  
- relative-heading train (only after D1 says reversals matter, and with hold phases)

### Next only if v11 plateaus

1. Relative-heading **train** with mandatory hold gaps  
2. Wheel-lock gate/weight (careful)  
3. Roughness 3 cm then 4 cm as grip curriculum  
4. SB 20° → SC 30°

---

## D1/D2 results on model_12349 (2026-07-27) — DECISIVE

| Test | Command | Result |
|---|---|---|
| **D1+** | ω = **+0.12** constant | **PICTURE PERFECT** — continuous spin, acceptable station-keeping, clean slope turn |
| **D1−** | ω = **−0.12** constant | **MASSIVE FAILURE** — repeatedly falls **back-to-back** (backward pitch / somersault) |
| **D2** | ω = 0 hold | **Better than baseline** — holds position reasonably, falls sometimes |
| Baseline | ω ~ U(−0.12, +0.12) | Intermittent flips — now explained as mixture of good (+ω) and bad (−ω) windows |

### What this means

1. **Not friction, not roughness, not “can’t turn on slopes.”**  
   The policy already has a working slope pivot — **in one direction only**.

2. **Baseline intermittency was direction mixing.**  
   Random sign every 6 s sometimes drew the competent +ω skill, sometimes the broken −ω skill.

3. **Failure mode on −ω is pitch-backward** (“back to back”), not a generic tip.  
   TensorBoard `bad_pitch` vs `bad_roll` should confirm this under train.

4. **D2 OK ⇒ grip/physics at 10° are adequate** for standing.  
   The crisis is **CW / negative-yaw motor skill on slope**, not mu.

5. **v9-a lesson still holds for heading_command=True:**  
   asymmetric *clip* under heading P-control is unsatisfiable.  
   With **heading_command=False** (sampled ω), a negative-focused range is valid — ω is the command, not a clip on a signed error.

### v12 training plan (from 12349)

**Do not** ship a forever-+ω-only policy (nav needs both ways).  
**Do** repair −ω with a focused warm-start, then re-mix.

| Phase | Command | Purpose |
|---|---|---|
| **SA-v12a** | `heading_command=False`, ω ∈ **(−0.12, −0.08)** only | Teach the missing CW skill; station-keeping from v11 kept |
| **SA-v12b** | `heading_command=False`, ω ∈ **(−0.12, +0.12)** symmetric | Re-integrate both directions without absolute-heading teleports |
| Play demo (optional) | D1+ until v12b is good | Honest demo of current competence |

Also kept from v11: drift 0.35 m / weight −1.0, trunk 20°, bad_pitch/bad_roll.

**Not in v12:** roughness 4 cm, relative-heading play hacks, killing microstep rewards.

---

## Phase SB 20° (from balanced V12B)

**Warm-start:** `go2w_velocity_slope_turn_a/2026-07-27_19-49-02/model_12847.pt`  
(symmetric ω ±0.12, both directions good on 10°)

**Config fixes applied before this phase:**
- SB/SC use `_apply_slope_command_overrides_symmetric` (not negative-only v12a)
- SB `trunk_stability` 25° (SA stays 20°)
- SC `trunk_stability` 30°

```bash
./run.sh scripts/train.py --task RexmiRl-Go2w-Velocity-SlopeTurn-v0 --headless \
  --load_run go2w_velocity_slope_turn_a/2026-07-27_19-49-02 \
  --checkpoint model_12847.pt
```

Play:
```bash
./run.sh scripts/play.py --task RexmiRl-Go2w-Velocity-SlopeTurn-Play-v0 \
  --load_run go2w_velocity_slope_turn/<date> --checkpoint model_<N>.pt
```

One variable: slope 10° → 20°. Same sampled-ω interface, station-keeping, friction.

---

## SB-v2 — slower ω on 20° (balance pass)

**Context:** SB-v1 `model_13096` turns well on 20° but struggles to stabilize;
logs showed ~61% `bad_orientation`, ~24% `bad_roll`, pitch ~0. Visual confirmed
rotation OK, balance hard.

**One change:** ω (±0.12) → **(±0.08)** symmetric, still `heading_command=False`.
Warm-start from **13096** (already on 20°).

```bash
./run.sh scripts/train.py --task RexmiRl-Go2w-Velocity-SlopeTurn-v0 --headless \
  --load_run go2w_velocity_slope_turn/2026-07-27_20-15-35 \
  --checkpoint model_13096.pt
```

Play (matches ±0.08):
```bash
./run.sh scripts/play.py --task RexmiRl-Go2w-Velocity-SlopeTurn-Play-v0 \
  --load_run go2w_velocity_slope_turn/<date> --checkpoint model_<N>.pt
```

**Success:** episode length ↑ from ~500, `bad_roll` ↓, both directions still turn.
If stuck after a few hundred iters: try ±0.06 or a 15° bridge — not 30°, not roughness.

---

## SC-v1 — 30° with slower ω (±0.06)

**Warm-start:** `go2w_velocity_slope_turn/2026-07-27_20-54-25/model_13345.pt`  
(SB-v2: 20°, both directions, nav-acceptable finite turns)

**Why ±0.06:** 10→20 at ±0.12 caused roll deaths; ±0.08 fixed most. Another +10° of slope needs another ease of yaw rate. One variable besides slope: ω clip.

```bash
./run.sh scripts/train.py --task RexmiRl-Go2w-Velocity-SlopeTurnC-v0 --headless \
  --load_run go2w_velocity_slope_turn/2026-07-27_20-54-25 \
  --checkpoint model_13345.pt
```

Logs: `logs/rsl_rl/go2w_velocity_slope_turn_c/`

Play:
```bash
./run.sh scripts/play.py --task RexmiRl-Go2w-Velocity-SlopeTurnC-Play-v0 \
  --load_run go2w_velocity_slope_turn_c/<date> --checkpoint model_<N>.pt
```

**Nav note:** evaluate short reorients, not endless open-loop spin. Policy is for finite heading change then stop.

---

## SC-v1 gate fix (v13) — before 30° train

**Bug:** Turn-B gates used `omega_threshold` / `min_cmd` = **0.1**.  
At SB-v2 ω ±0.08 those rewards were already dead (`foot_alternation=0`, `wheel_lock=0` in logs).  
At SC ±0.06 they would stay dead.

**Fix (all slope phases via `_apply_slope_reward_overrides`):**
- `wheel_lock`, `foot_alternation`, `yaw_stagnation`, `heading_progress`,
  `pivot_step_coord`, `foot_air_time` gates → **0.04**
- `pivot_step_coordination` now takes `omega_threshold` (was hardcoded 0.05)

**Train SC 30° from SB-v2:**
```bash
./run.sh scripts/train.py --task RexmiRl-Go2w-Velocity-SlopeTurnC-v0 --headless \
  --load_run go2w_velocity_slope_turn/2026-07-27_20-54-25 \
  --checkpoint model_13345.pt
```

Logs: `logs/rsl_rl/go2w_velocity_slope_turn_c/`

**Watch:** ep_len, `bad_roll` / `bad_orientation`, and that
`foot_alternation` / `wheel_lock` are **non-zero** (gates alive).

---

## S25 mixed hold + turn @ 25° (after SC slip failure)

**Context:** SC 30° (13594) slips immediately — no static plant. Pure hold-only
risks erasing turn skill from SB-v2. Bridge at **25°** with **mixed** commands.

**Warm-start:** `go2w_velocity_slope_turn/2026-07-27_20-54-25/model_13345.pt`

| Knob | Value |
|---|---|
| Slope | 25° |
| Standing fraction | **35%** (ω=0, vx=0) |
| Turn envs | ω ±0.08, vx=0.05 |
| trunk | 28° |
| Gates | 0.04 (v13) |

```bash
./run.sh scripts/train.py --task RexmiRl-Go2w-Velocity-SlopeTurnS25-v0 --headless \
  --load_run go2w_velocity_slope_turn/2026-07-27_20-54-25 \
  --checkpoint model_13345.pt
```

Logs: `logs/rsl_rl/go2w_velocity_slope_turn_s25/`

Play:
```bash
# Mixed (like train)
./run.sh scripts/play.py --task RexmiRl-Go2w-Velocity-SlopeTurnS25-Play-v0 \
  --load_run go2w_velocity_slope_turn_s25/<date> --checkpoint model_<N>.pt

# Hold only
./run.sh scripts/play.py --task RexmiRl-Go2w-Velocity-SlopeTurnS25-Play-Hold-v0 \
  --load_run go2w_velocity_slope_turn_s25/<date> --checkpoint model_<N>.pt

# Turn only
./run.sh scripts/play.py --task RexmiRl-Go2w-Velocity-SlopeTurnS25-Play-Turn-v0 \
  --load_run go2w_velocity_slope_turn_s25/<date> --checkpoint model_<N>.pt
```

**Pass:** no instant slip; still CW/CCW turn. Then S30 mixed or slow turn on 30°.

---

## S25-v2 — slower turn, vx=0 (keep hold)

**S25-v1 result:** hold on 25° OK; turn loses position / rolls (bad_roll ~37%).

**One theme:** gentler pivot command while keeping 35% standing.

| Knob | v1 | v2 |
|---|---|---|
| ω | ±0.08 | **±0.05** |
| vx turn | 0.05 | **0.0** |
| standing | 35% | 35% |

**Warm-start:** `go2w_velocity_slope_turn_s25/2026-07-28_20-59-54/model_13594.pt`

```bash
./run.sh scripts/train.py --task RexmiRl-Go2w-Velocity-SlopeTurnS25-v0 --headless \
  --load_run go2w_velocity_slope_turn_s25/2026-07-28_20-59-54 \
  --checkpoint model_13594.pt
```

Play turn-only (vx=0, ω±0.05):
```bash
./run.sh scripts/play.py --task RexmiRl-Go2w-Velocity-SlopeTurnS25-Play-Turn-v0 \
  --load_run go2w_velocity_slope_turn_s25/<new_date> --checkpoint model_<N>.pt
```

---

## S25-v3 — micro-turn → rebalance (from 13345)

**Visual (user):** 13345 on 25° holds; rarely turns; when it turns sometimes
succeeds. Continuous spin is wrong. Need: **hold → small turn → rebalance →
small turn**.

**S25-v2 failed** by teaching freeze (Play-Turn also held; high track_ang,
near-zero heading_progress).

### Recipe
| Item | Value |
|---|---|
| Warm-start | `go2w_velocity_slope_turn/2026-07-27_20-54-25/model_13345.pt` |
| Slope | 25° |
| Command | **RelativeHeading** Δ **8–18°**, ω clip **0.08** |
| Settle | **2.0 s** ω≈0 after each burst (rebalance) |
| Standing | **10%** (not 35%) |
| vx | 0 |
| heading_progress | weight **120** |
| is_alive | **0.35** (was 0.8) |
| track_ang | **1.5** |

```bash
./run.sh scripts/train.py --task RexmiRl-Go2w-Velocity-SlopeTurnS25-v0 --headless \
  --load_run go2w_velocity_slope_turn/2026-07-27_20-54-25 \
  --checkpoint model_13345.pt
```

**Watch:** `heading_progress_turn` must rise; visual must show intermittent
yaw bursts with plant between them — not freeze, not continuous thrash.

Play:
```bash
./run.sh scripts/play.py --task RexmiRl-Go2w-Velocity-SlopeTurnS25-Play-Turn-v0 \
  --load_run go2w_velocity_slope_turn_s25/<date> --checkpoint model_<N>.pt
```

---

## S25-v3b — error-driven heading (not open-loop bursts)

**User design:** error accumulates until a step is worth it; when error is
small, rebalance can last as long as needed. Stability makes reckless spin
expensive. Prefer this over fixed ω bursts + fixed settle timers.

### Command
- Relative heading goal Δ **10–25°**
- ω clip **0.08** via P-control on error
- **settle_margin = 0** (rebalance emerges when |error| small)
- standing **8%**, vx **0**

### Rewards
| Term | Weight | Role |
|---|---|---|
| heading_error_l1 | -2.5 | pressure while |err| > deadzone |
| heading_error_reduction | +8 | pay for shrinking error |
| heading_progress_turn | +80 | signed real Δyaw |
| track_ang_vel_z_exp | +1.0 | weak (lied in freeze run) |
| is_alive | +0.30 | not freeze optimum |
| position_drift | -1.5 @ 0.28 m | stay planted |
| trunk_stability | -5 @ 28° | stability |

### Train (warm-start 13345 only)

```bash
./run.sh scripts/train.py --task RexmiRl-Go2w-Velocity-SlopeTurnS25-v0 --headless \
  --load_run go2w_velocity_slope_turn/2026-07-27_20-54-25 \
  --checkpoint /home/susan/rexmi_rl/logs/rsl_rl/go2w_velocity_slope_turn/2026-07-27_20-54-25/model_13345.pt
```

### Play (full checkpoint path required)

```bash
./run.sh scripts/play.py --task RexmiRl-Go2w-Velocity-SlopeTurnS25-Play-Turn-v0 \
  --load_run go2w_velocity_slope_turn/2026-07-27_20-54-25 \
  --checkpoint /home/susan/rexmi_rl/logs/rsl_rl/go2w_velocity_slope_turn/2026-07-27_20-54-25/model_13345.pt

# After train:
./run.sh scripts/play.py --task RexmiRl-Go2w-Velocity-SlopeTurnS25-Play-Turn-v0 \
  --load_run go2w_velocity_slope_turn_s25/<date> \
  --checkpoint /home/susan/rexmi_rl/logs/rsl_rl/go2w_velocity_slope_turn_s25/<date>/model_<N>.pt
```

**Watch:** `Episode_Reward/heading_error` (more negative = more residual error),
`heading_error_reduction` and `heading_progress_turn` up; visual intermittent
corrections with plant between — not freeze, not continuous thrash.

---

## S25 Phase A — Language A anti-freeze (not v2, not v3b)

**User chose Language A** (sampled ω). Nav can schedule ω bursts later.

| | v2 (froze) | **Phase A** |
|---|---|---|
| Language | A | A |
| standing | 35% | **8%** |
| ω | ±0.05 | **±0.08** |
| is_alive | 0.8 | **0.25** |
| heading_progress | weak | **100** |
| Warm-start | 13345 | **13345 only** |

```bash
# TRAIN
./run.sh scripts/train.py --task RexmiRl-Go2w-Velocity-SlopeTurnS25-v0 --headless \
  --load_run go2w_velocity_slope_turn/2026-07-27_20-54-25 \
  --checkpoint /home/susan/rexmi_rl/logs/rsl_rl/go2w_velocity_slope_turn/2026-07-27_20-54-25/model_13345.pt

# PLAY turn (full checkpoint path)
./run.sh scripts/play.py --task RexmiRl-Go2w-Velocity-SlopeTurnS25-Play-Turn-v0 \
  --load_run go2w_velocity_slope_turn_s25/<date> \
  --checkpoint /home/susan/rexmi_rl/logs/rsl_rl/go2w_velocity_slope_turn_s25/<date>/model_<N>.pt

# PLAY hold
./run.sh scripts/play.py --task RexmiRl-Go2w-Velocity-SlopeTurnS25-Play-Hold-v0 \
  --load_run go2w_velocity_slope_turn_s25/<date> \
  --checkpoint /home/susan/rexmi_rl/logs/rsl_rl/go2w_velocity_slope_turn_s25/<date>/model_<N>.pt
```

**Abort if:** `heading_progress_turn` → 0 while ep_len only rises (freeze attractor).
**Pass if:** Play-Turn shows visible CW/CCW rotation on 25°.

---

## S25 Phase A2 — slow + non-continuous (A1+A2)

**Visual Phase A:** hold OK; continuous spin → fall.

| Knob | Phase A | **A2** |
|---|---|---|
| ω | ±0.08 | **±0.04** |
| resample | 5 s | **2.0 s** |
| standing | 8% | **30%** (plant windows) |
| Language | A | A |
| Warm-start | 13345 | **13345** |

```bash
# TRAIN
./run.sh scripts/train.py --task RexmiRl-Go2w-Velocity-SlopeTurnS25-v0 --headless \
  --load_run go2w_velocity_slope_turn/2026-07-27_20-54-25 \
  --checkpoint /home/susan/rexmi_rl/logs/rsl_rl/go2w_velocity_slope_turn/2026-07-27_20-54-25/model_13345.pt

# PLAY turn (pulse + plant)
./run.sh scripts/play.py --task RexmiRl-Go2w-Velocity-SlopeTurnS25-Play-Turn-v0 \
  --load_run go2w_velocity_slope_turn_s25/<date> \
  --checkpoint /home/susan/rexmi_rl/logs/rsl_rl/go2w_velocity_slope_turn_s25/<date>/model_<N>.pt

# PLAY hold
./run.sh scripts/play.py --task RexmiRl-Go2w-Velocity-SlopeTurnS25-Play-Hold-v0 \
  --load_run go2w_velocity_slope_turn_s25/<date> \
  --checkpoint /home/susan/rexmi_rl/logs/rsl_rl/go2w_velocity_slope_turn_s25/<date>/model_<N>.pt
```

**Plant windows:** 30% (not 45%) — enough rebalance without v2-style freeze bias.

**Watch:** bad_roll down from ~45%; heading_progress non-zero on turn windows;
visual small yaw then plant — not freeze, not continuous thrash.
