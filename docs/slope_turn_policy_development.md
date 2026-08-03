# Slope Turn Policy Development

**Robot:** Unitree Go2W (wheeled quadruped)  
**Goal:** Reliable **stop → micro-turn → rebalance** on slopes up to **~33–35°**, then hand the same interface to the **nav layer**.  
**Last updated:** 2026-08-02  
**Status:** 🔄 **Pulse20-v2 skill confirmed** (hold→turn real); play envelope softened (ω±0.04, yaw 1s) — re-visual

---

## 0. Read this first (current truth)

### What works today

| Asset | Path | Capability |
|-------|------|------------|
| **Best slope-turn ckpt** | `logs/rsl_rl/go2w_velocity_slope_turn/2026-07-27_20-54-25/model_13345.pt` | **Turns both ways on ~20°** (Language A, continuous-ish sampled ω). Warm-start root for later work. |
| **Hold on steep** | Multiple S25 / SC visuals | Robot can **plant and hold** on **25°** (and hold was seen near **30°**). Stance is not the missing primitive. |
| **Forward locomotion** | `fast_flat`, `rough`, `rocky_slope` | Separate stack; do not retrain for turn. |

### What does **not** work yet

| Claim | Evidence |
|-------|----------|
| Reliable **pivot / heading change on ≥25°** | S25-v1 tip, v2 freeze, v3b death, Phase A roll, A2 no-spin |
| Pulse FSM that produces **visible multi-pulse yaw** | Pulse20-v1: metrics excellent, **visual = wiggle only** |
| Nav-ready steep reorient | Blocked on real yaw skill |

### Active bet (do this next)

**Pulse20-v2** — same HOLD→YAW→SETTLE FSM at **20°**, but **longer/stronger yaw pulses** and **tighter real-yaw pay**.  
Warm-start **only** from `model_13345.pt` — **not** Pulse20-v1 `model_13594.pt`.

```bash
./run.sh scripts/train.py --task RexmiRl-Go2w-Velocity-SlopeTurnPulse20-v0 --headless \
  --load_run go2w_velocity_slope_turn/2026-07-27_20-54-25 \
  --checkpoint /home/susan/rexmi_rl/logs/rsl_rl/go2w_velocity_slope_turn/2026-07-27_20-54-25/model_13345.pt
```

**Pass bar (visual, mandatory):** clear heading change each yaw pulse, both directions, settle plants between pulses.  
**Then:** same FSM at 22°→25°→…→33–35°, then nav emits the same pulse schedule.

### Full checkpoint paths (always)

Play/train must use **absolute** `--checkpoint /home/susan/rexmi_rl/logs/rsl_rl/.../model_N.pt`  
plus `--load_run <experiment>/<date>`. Relative names alone have caused wrong-ckpt loads.

---

## 1. Target skill (reframe)

### Wrong problem (what failed repeatedly)
Track a **continuous** yaw-rate ω for many seconds on a steep face.

### Right problem
On slope θ:

> **HOLD** (plant) → **YAW** (small heading increment) → **SETTLE** (rebalance) → repeat → **stop**.

That matches:

- Physical capability (hold works; continuous spin rolls)
- Nav need (finite Δheading, then drive or hold)
- Language A (policy sees piecewise-constant `[vx, vy, ωz]`)

Nav later: schedule the same pulses (`sign(Δψ)·ω` for `T_yaw`, then `ω=0` for `T_settle`).

---

## 2. Command languages

| | **Language A — sampled ω** | **Language B — relative heading** |
|--|----------------------------|-----------------------------------|
| What is sampled | Yaw **rate** ω (held for a window) | Heading **goal** θ*; ω = P(error) |
| Obs ω | Piecewise **constant** | **Time-varying** as error shrinks |
| 13345 trained on | **Yes** | No |
| Nav mapping | Finite ω burst then 0 | Closed-loop face target |
| Project choice | **Preferred / current** | Tried in S25-v3b; abandoned for now |

**Rationale for sticking with A:** 13345 already speaks it; nav can emit bursts without changing obs semantics; B + new slope + new rewards in one jump caused a death spiral.

Implementation of structured A: `HoldYawSettleVelocityCommand` in  
`source/rexmi_rl/tasks/locomotion/velocity/mdp/commands.py`.

---

## 3. Phase timeline, outcomes, pivots

| Phase | Slope | Command / recipe | Metrics / visual | Outcome | Pivot / rationale |
|-------|-------|------------------|------------------|---------|-------------------|
| Flat Turn B | 0° | Pivot gait | — | **Works** (`model_10992`) | Baseline |
| SA | 5–20° | Transfer + slope | — | Path to SB | Curriculum |
| **SB-v2** | **20°** | Sampled ω ~±0.08 | Turns CW/CCW | **KEEP — `model_13345.pt`** | Best slope-turn asset |
| SC-v1 | 30° | Continuous ω ±0.06 | Slip, no plant | **Fail** | 30° too hard before 25° bridge |
| S25-v1 | 25° | ω±0.08, stand ~35%, some vx | Hold OK; turn tips | Partial | Need slower / less continuous |
| S25-v2 | 25° | ω±0.05, stand 35%, vx=0 | ep↑, **hp→0**, Play freezes | **Freeze** | Survival > yaw; do not replay |
| S25-v3b | 25° | **Language B** + heading_error | ep~60, bad_o~72% | **Death spiral** | Too many changes vs 13345 prior |
| S25 Phase A | 25° | A anti-freeze: stand 8%, ω±0.08 | hp alive; bad_roll~45% | Turn then **roll** | Continuous aggressive spin is the killer |
| S25 A2 | 25° | ω±0.04, 2s, stand 30% | ep↑ roll↓; **visual no spin** | Soft freeze / lunge-fall | Leave 25° until pulse works at 20° |
| **Pulse20-v1** | **20°** | HOLD 1.5 / YAW 1.0@0.06 / SETTLE 2.0 | ep~960, timeout~95%, hp~0.15 | **Metrics lie** | Visual: **wiggle only** |
| **Pulse20-v2** | **20°** | HOLD 1.0 / YAW **2.0@0.08** / SETTLE 1.5; hp**180**, track_ang**1**, is_alive**0.22** | — | **Current train** | Force visible Δψ per pulse |

### Scoreboard (representative end-of-run)

| Run | ep_len | bad_orient | bad_roll | heading_progress | Visual |
|-----|--------|------------|----------|------------------|--------|
| 13345 @ 20° | ~850 | ~23% | ~6% | good | **Turns** |
| S25-v2 | ~350 | ~69% | ~23% | ~0 | **Freeze** |
| S25-v3b | ~60 | ~72% | ~28% | tiny | **Dies** |
| Phase A | ~80 | ~54% | ~45% | ~0.055 | Turn → roll |
| A2 | ~130–160 | ~66% | ~34% | ~0.01 | No spin |
| Pulse20-v1 | ~960 | ~3.4% | ~1.7% | ~0.15 | **Wiggle only** |

---

## 4. Hard learnings (rationales)

1. **Visual gates beat TensorBoard.**  
   `track_ang_vel_z_exp`, `is_alive`, and even moderate `heading_progress` can look “healthy” while the robot only wiggles (Pulse20-v1) or freezes (S25-v2).

2. **Hold is solved earlier than turn.**  
   Do not spend the next train proving plant on 25°; spend it on **yaw under stability**.

3. **Continuous ω on steep slopes → roll.**  
   Lateral load and support-polygon shift during long pivots. Skill must be **non-continuous**.

4. **High standing fraction + weak yaw pay → freeze.**  
   PPO learns “never pivot” when hold is long-lived and turn is punished by falls (S25-v2).

5. **Anti-freeze alone is not enough.**  
   Phase A kept yaw pressure and the robot **tried** to turn — then rolled. Need **structure** (pulses), not only reward weights.

6. **Language B is not free.**  
   Switching 13345 (Language A) onto relative-heading P-control + new rewards + 25° in one run destroyed the prior (v3b).

7. **Teach the FSM where turn already exists (20°), then climb angle.**  
   Inventing steepness and a new command schedule together failed. Pulse at 20° first was the correct strategic pivot after A2.

8. **Target skill is micro-reorient, not spin rate.**  
   Nav wants finite Δheading. Training should look like nav.

9. **Never warm-start from freeze/death/wiggle champions** when the goal is real yaw.  
   Prefer **13345** until a pulse ckpt **visually** turns.

10. **One theme per train.**  
    Slope **or** command language **or** reward family — not all three overnight.

---

## 5. Active plan & gates

```
model_13345.pt (20° continuous turn — KEEP)
        │
        ▼
Pulse20-v2  HOLD→YAW→SETTLE @ 20°   ← YOU ARE HERE (train + visual)
        │  pass: multi-pulse heading both ways, settle OK
        ▼
Pulse22 / Pulse25  (same FSM; slower ω / longer settle if needed)
        │
        ▼
Pulse28 → 30 → 33–35°
        │
        ▼
Nav: emit same pulse schedule for reorient; forward policies unchanged
```

### Pulse20-v2 recipe (wired)

| Knob | Value |
|------|--------|
| Task | `RexmiRl-Go2w-Velocity-SlopeTurnPulse20-v0` |
| Logs | `logs/rsl_rl/go2w_velocity_slope_turn_pulse20/` |
| hold / yaw / settle | **1.0 s / 2.0 s / 1.5 s** |
| ω during yaw | **±0.08** |
| heading_progress | **180** |
| track_ang | **1.0** |
| is_alive | **0.22** |
| yaw_stagnation | min **8°**, window ~80 steps, weight **-4** |
| Warm-start | **13345 only** |

Play after train:

```bash
./run.sh scripts/play.py --task RexmiRl-Go2w-Velocity-SlopeTurnPulse20-Play-Yaw-v0 \
  --load_run go2w_velocity_slope_turn_pulse20/<date> \
  --checkpoint /home/susan/rexmi_rl/logs/rsl_rl/go2w_velocity_slope_turn_pulse20/<date>/model_<N>.pt
```

### Play-only soft envelope (after v2 visual — real turn, tips)

**Visual v2 full cycle:** hold → spin works; long/fast yaw destabilizes → fall.
Sometimes recovers after aggressive turn → skill is real; envelope too hot.

**No retrain.** Play configs only (train remains 2.0s @ ±0.08):

| Knob | Train v2 | **Play now** |
|------|----------|--------------|
| ω | ±0.08 | **±0.04** |
| yaw time | 2.0 s | **1.0 s** |
| settle | 1.5 s | 1.5 s |
| Ckpt | — | `.../pulse20/2026-08-02_16-36-04/model_13594.pt` |

```bash
./run.sh scripts/play.py --task RexmiRl-Go2w-Velocity-SlopeTurnPulse20-Play-v0 \
  --load_run go2w_velocity_slope_turn_pulse20/2026-08-02_16-36-04 \
  --checkpoint /home/susan/rexmi_rl/logs/rsl_rl/go2w_velocity_slope_turn_pulse20/2026-08-02_16-36-04/model_13594.pt

./run.sh scripts/play.py --task RexmiRl-Go2w-Velocity-SlopeTurnPulse20-Play-Yaw-v0 \
  --load_run go2w_velocity_slope_turn_pulse20/2026-08-02_16-36-04 \
  --checkpoint /home/susan/rexmi_rl/logs/rsl_rl/go2w_velocity_slope_turn_pulse20/2026-08-02_16-36-04/model_13594.pt
```

**Watch:** multi-pulse heading creep without tip; settle plants.

### Explicitly do **not**

- Climb to 25°+ on Pulse20-v1 or any freeze/death ckpt  
- Replay S25-v2 (35% standing + timid ω + high is_alive)  
- Replay Language B death stack  
- Ship on metrics without visual multi-pulse yaw  
- Optimize turn **speed** before stability  

---

## 6. Key code map

| Piece | Location |
|-------|----------|
| Pulse FSM command | `mdp/commands.py` → `HoldYawSettleVelocityCommand` |
| Relative heading (Language B, legacy) | `mdp/commands.py` → `RelativeHeadingVelocityCommand` |
| Slope envs / S25 / Pulse20 | `config/go2w/slope_turn_env_cfg.py` |
| Task IDs | `config/go2w/__init__.py` (`SlopeTurnPulse20-*`, `SlopeTurnS25-*`, …) |
| PPO experiment names | `config/go2w/agents/rsl_rl_ppo_cfg.py` |
| Heading / yaw rewards | `mdp/rewards.py` (`heading_progress`, `yaw_stagnation`, …) |
| Command semantics (broader) | `docs/turn_command_semantics.md` |
| Nav plan | `docs/autonomous_nav_plan.md` |

---

## 7. Checkpoint index (do not lose)

| Ckpt | Use |
|------|-----|
| `.../go2w_velocity_turn_b/2026-07-25_13-51-28/model_10992.pt` | Flat turn root |
| `.../go2w_velocity_slope_turn/2026-07-27_20-54-25/model_13345.pt` | **Best 20° turn — primary warm-start** |
| `.../go2w_velocity_slope_turn_s25/2026-08-02_14-55-14/model_13594.pt` | Phase A (roll) — research only |
| `.../go2w_velocity_slope_turn_s25/2026-08-02_15-24-36/model_13594.pt` | A2 (no spin) — research only |
| `.../go2w_velocity_slope_turn_pulse20/2026-08-02_16-10-30/model_13594.pt` | Pulse20-v1 **wiggle** — **do not warm-start for yaw** |

---

# Archive — historical recipes & early curriculum notes

> Sections below are **append-era notes** (SA/SB/SC/S25/Pulse recipes).  
> Prefer **§0–§7 above** for decisions. Recipes remain useful for exact CLI and hyperparams.

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

---

## Pulse FSM @ 20° — HOLD → YAW → SETTLE (single policy)

**Decision:** single policy + pulse FSM; start at **20°** from **model_13345**.
Path to 25→35° only after pulse-turn is visually solid at each step.

### Command machine (Language A)
| Phase | Duration | ω |
|---|---|---|
| HOLD (episode start) | 1.5 s | 0 |
| YAW | 1.0 s | ±0.06 |
| SETTLE | 2.0 s | 0 |
| then YAW ↔ SETTLE | … | … |

### Train
```bash
./run.sh scripts/train.py --task RexmiRl-Go2w-Velocity-SlopeTurnPulse20-v0 --headless \
  --load_run go2w_velocity_slope_turn/2026-07-27_20-54-25 \
  --checkpoint /home/susan/rexmi_rl/logs/rsl_rl/go2w_velocity_slope_turn/2026-07-27_20-54-25/model_13345.pt
```

Logs: `logs/rsl_rl/go2w_velocity_slope_turn_pulse20/`

### Play (full checkpoint paths)
```bash
# Full cycle
./run.sh scripts/play.py --task RexmiRl-Go2w-Velocity-SlopeTurnPulse20-Play-v0 \
  --load_run go2w_velocity_slope_turn_pulse20/<date> \
  --checkpoint /home/susan/rexmi_rl/logs/rsl_rl/go2w_velocity_slope_turn_pulse20/<date>/model_<N>.pt

# Yaw/settle emphasis
./run.sh scripts/play.py --task RexmiRl-Go2w-Velocity-SlopeTurnPulse20-Play-Yaw-v0 \
  --load_run go2w_velocity_slope_turn_pulse20/<date> \
  --checkpoint /home/susan/rexmi_rl/logs/rsl_rl/go2w_velocity_slope_turn_pulse20/<date>/model_<N>.pt

# Hold only
./run.sh scripts/play.py --task RexmiRl-Go2w-Velocity-SlopeTurnPulse20-Play-Hold-v0 \
  --load_run go2w_velocity_slope_turn_pulse20/<date> \
  --checkpoint /home/susan/rexmi_rl/logs/rsl_rl/go2w_velocity_slope_turn_pulse20/<date>/model_<N>.pt
```

**Pass:** visible small yaw bursts both ways, stable settle between, no continuous thrash.
**Then:** Pulse22 / Pulse25 with same FSM, slower ω / longer settle as needed.

---

## Pulse20-v2 — longer/stronger yaw after wiggle-only fail

**v1 visual:** no heading change, wiggle only @ 20° despite great survival metrics.

| Knob | v1 | **v2** |
|---|---|---|
| hold | 1.5 s | **1.0 s** |
| yaw | 1.0 s @ ±0.06 | **2.0 s @ ±0.08** |
| settle | 2.0 s | **1.5 s** |
| heading_progress | 100 | **180** |
| track_ang | 2.0 | **1.0** |
| is_alive | 0.35 | **0.22** |
| yaw_stag | 4° / 50 steps | **8° / ~80 steps** |
| Warm-start | 13345 | **13345 only** (not v1 13594) |

```bash
./run.sh scripts/train.py --task RexmiRl-Go2w-Velocity-SlopeTurnPulse20-v0 --headless \
  --load_run go2w_velocity_slope_turn/2026-07-27_20-54-25 \
  --checkpoint /home/susan/rexmi_rl/logs/rsl_rl/go2w_velocity_slope_turn/2026-07-27_20-54-25/model_13345.pt

./run.sh scripts/play.py --task RexmiRl-Go2w-Velocity-SlopeTurnPulse20-Play-Yaw-v0 \
  --load_run go2w_velocity_slope_turn_pulse20/<date> \
  --checkpoint /home/susan/rexmi_rl/logs/rsl_rl/go2w_velocity_slope_turn_pulse20/<date>/model_<N>.pt
```

**Pass:** visible heading change each pulse, both directions, settle plants.
