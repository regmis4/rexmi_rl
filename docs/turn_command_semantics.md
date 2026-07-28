# Turn Command Semantics — what the turn policies are actually being asked to do

**Date:** 2026-07-26
**Applies to:** every turn policy in this project — spin, spin_static, Turn A/B/C, Slope Turn SA/SB/SC
**Status:** root cause confirmed by source inspection

---

## TL;DR

The turn policies are **not** being trained to spin. They are being trained to
**face a random compass heading and then hold it**, with a new target every 10 s.

Every "burst of rotation, then idle micro-tapping, then another burst" observation
is that command being executed *correctly*. It is not a policy defect.

This is also **the behaviour the nav layer wants** ("turn to face X, then drive"),
so the semantics are correct and should be kept. What was wrong was our *reading*
of them.

---

## The mechanism

Isaac Lab base config — `isaaclab_tasks/.../locomotion/velocity/velocity_env_cfg.py`:

```python
base_velocity = mdp.UniformVelocityCommandCfg(
    resampling_time_range=(10.0, 10.0),
    rel_standing_envs=0.02,
    rel_heading_envs=1.0,          # 100% of envs are heading envs
    heading_command=True,
    heading_control_stiffness=0.5,
    ranges=...Ranges(
        lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi),
    ),
)
```

`isaaclab/envs/mdp/commands/velocity_command.py::_update_command`:

```python
heading_error = wrap_to_pi(self.heading_target[env_ids] - self.robot.data.heading_w[env_ids])
self.vel_command_b[env_ids, 2] = torch.clip(
    self.cfg.heading_control_stiffness * heading_error,
    min=self.cfg.ranges.ang_vel_z[0],
    max=self.cfg.ranges.ang_vel_z[1],
)
```

### Consequence 1 — `ang_vel_z` is not a command, it is a clip bound

With `heading_command=True` and `rel_heading_envs=1.0`, the yaw command is
**never sampled from `ranges.ang_vel_z`**. It is a P-controller output:

```
omega_cmd = clip(0.5 * wrap_to_pi(heading_target - heading_now), ang_vel_z_min, ang_vel_z_max)
```

`ranges.ang_vel_z` only *saturates* that controller. Every phase of yaw-range
tuning in this project was adjusting clip bounds, not commanded yaw rates.

### Consequence 2 — the steady state is omega = 0, by design

As the heading error decays, `omega_cmd -> 0`. Sitting still once the heading is
reached is the command being satisfied. Combined with `resampling_time_range=(10, 10)`,
the robot spends most of each 10 s window already-arrived and idle:

```
spawn -> large heading error -> command saturates -> ROTATES
      -> target reached -> error -> 0 -> command -> 0 -> IDLE
      -> 10 s later resample -> new random target -> ROTATES again
```

### Consequence 3 — nothing in this project overrides any of it

Verified by grep across `source/rexmi_rl/`: `heading_command`, `rel_heading_envs`,
`resampling_time_range` and `ranges.heading` are **never** set anywhere. All turn
policies inherit the defaults above.

---

## Why this matters: it explains the v9-a catastrophe precisely

v9-a set `ang_vel_z = (0.25, 0.35)` — a **positive-only clip**. Because the clip is
applied to a *signed* P-controller output:

> whenever the heading target lay to the **right** (negative error), the command was
> still forced to **at least +0.25**, i.e. "turn left" when the robot physically had
> to turn right.

The heading error could then never converge — the robot would rotate left forever,
overshooting. The command was **unsatisfiable**, not merely out-of-distribution.
Result: 99.9% `bad_orientation`, `track_ang_vel_z_exp` = 0.008.

**Rule:** the `ang_vel_z` clip must stay **symmetric** about zero as long as
`heading_command=True`. An asymmetric clip is a sign-contradictory command.

---

## Hypothesis tested and rejected: "we trained it to turn one direction only"

Not supported. The training lineage is symmetric in both directions:

| where | value |
|---|---|
| `Go2wSlopeTurnAEnvCfg(Go2wTurnBEnvCfg)` | inherits `ang_vel_z = (-0.2, 0.2)` — symmetric |
| `_apply_slope_command_overrides` | deliberately does **not** set `ang_vel_z` |
| `ranges.heading` | `(-pi, pi)` base default, never overridden |
| `Go2wSlopeTurn*EnvCfg_PLAY` | overrode only `num_envs`, `env_spacing`, `enable_corruption`, `terrain_levels` — never `commands` |

Train and play distributions were **identical**. There was no direction mismatch to
fix, so restricting the play command to one direction would not have fixed a bug —
it would have hidden the ~67% fall rate behind a narrower demo.

### Still open (a different, legitimate hypothesis)

Turn competence may genuinely be **asymmetric on a slope**, because turning toward
uphill vs downhill is different physics even under symmetric commands. Test this by
comparing survival between a positive-only and a negative-only clip **as a
diagnostic run only** — do not bake the result into the configs.

---

## What was changed (v10)

**Play-only** (`_apply_slope_play_overrides`, applied to SA/SB/SC `_PLAY`), first pass:
`resampling_time_range` 10 s -> 4 s and `ang_vel_z` clip -> `(-0.12, 0.12)`.
Visuals improved a lot, but exposed a deeper defect — see v10b below.

---

## v10b — THE ACTUAL BUG: targets are unreachable, so the command reverses mid-turn

The heading target is an **absolute world yaw**, uniform over `(-pi, pi)`. The maximum
achievable turn rate is the clip `w`. So finishing a turn takes `|error| / w` seconds:

| clip `w` | worst case (pi) | mean case (pi/2) | resample | completes? |
|---|---|---|---|---|
| 0.20 | 15.7 s | 7.9 s | 10.0 s | **NO** |
| 0.12 | 26.2 s | 13.1 s | 4.0 s | **NO** |

**The resample period is shorter than the time needed to finish the turn.** The target
is therefore replaced *while the robot is still mid-turn*, by a fresh uniform draw that
reverses the required direction about half the time. The commanded yaw rate flips sign
discontinuously — `+w` to `-w` — while the legs are mid-pivot with the wheels
**anchored**. That is the "really weird command", and a sign flip against anchored
wheels is a very plausible way to produce a pitch-forward topple.

At `w=0.12 / 4 s` the robot covers only **28°** per window against a mean error of
**90°**. It never arrives — it is permanently saturated *and* randomly reversing.

**This is not a regression from the v10 change.** The inherited default has it too
(0.20 needs 15.7 s, gets 10 s). Tightening the clip made an existing defect worse.
Which also means **the falls are not a play-config artefact** — v8/v9b training logs
already showed 72–84% `bad_orientation` under this same command definition.

### Fix applied (play only)

```python
heading_command  = False      # omega is SAMPLED, not P-controlled
rel_heading_envs = 0.0
resampling_time_range = (6.0, 6.0)
ranges.ang_vel_z = (-0.12, 0.12)
```

`omega_cmd = uniform(-0.12, 0.12)`, held **constant** for 6 s. No heading target, so
nothing can be unreachable; no mid-turn sign reversal; still slow; still symmetric and
spanning zero, so in-distribution. 6 s x 0.12 = 41° of steady visible rotation.

**Diagnostic value:** this removes the command as a confound. If the robot still falls
under a clean constant slow yaw command, the fall is the *policy*, not the command.

---

## v10c — CONTINUOUS RELATIVE HEADING TARGETS (the actual fix)

Visual eval after v10b: rotation is now genuine and continuous ("rotating while
microstepping"), but the robot still occasionally rolls right or somersaults. The
remaining trigger is the **large delta between the current heading and a freshly drawn
absolute target**.

The fix is not to shape the clip harder — it is to stop drawing distant targets. New
command term `RelativeHeadingVelocityCommand` in
`source/rexmi_rl/tasks/locomotion/velocity/mdp/commands.py`:

```python
heading_target = wrap_to_pi(heading_now ± uniform(delta_min, delta_max))
```

| property | effect |
|---|---|
| **continuous** | each target continues from the previous heading — no step in heading error, so no step in `omega_cmd` |
| **reachable** | `delta_max` bounded, resample period **derived** from `delta_max / clip`; `__init__` raises if it is ever too short |
| **symmetric** | sign is a fair coin — never one-sided (an asymmetric yaw command is unsatisfiable; that was v9-a) |
| **non-trivial** | `\|delta\| >= delta_min`, so every window is a real turn, not an idle hold |

Because each turn now *completes* and `omega_cmd` decays to ~0 before the next target,
a direction reversal between windows **ramps from zero** instead of jumping `+w -> -w`.
That jump was the specific jolt throwing the robot onto its side.

### Derived play schedule

```
clip w           = 0.12 rad/s (6.9 deg/s)
heading delta    = 20-40 deg, random sign, relative
stiffness k      = 1.0   (raised from 0.5)
saturated turn   = 5.82 s
settling (3/k)   = 3.00 s
derived resample = 8.82 s     <- never hand-set
turning fraction = 66% of each window
```

`k` raised to 1.0 because higher stiffness keeps the command **saturated** for more of
the turn and shortens the exponential tail (settling `3/k`: 3 s instead of 6 s). That
long near-zero tail at `k=0.5` was what read as "idle". The policy observes `omega`,
not `k`, and `omega` is still bounded by the same clip — so this stays in-distribution.

**The resample period is derived, never hand-picked.** The original bug was the period
and the clip silently drifting apart; derived, they cannot.

---

## v10d — v10c KILLED THE ROTATION. THE ARITHMETIC I GOT WRONG.

Eval after v10c: **"it doesn't rotate at all."** Correct, and predictable:

| | command | rotation |
|---|---|---|
| v10b (worked) | ω = ±0.12 **constant** for 6 s | 41°/window, never idle, continuous |
| v10c (broke) | ≤40° turn, then **3 s dead time** | avg **4.5°/s**, and directions cancelled |

Three compounding mistakes:

1. **Dead time.** The settle margin is time with heading error ≈ 0, so ω ≈ 0 and the
   robot just sits. 3 s of hold on a 5.8 s turn is a third of the demo motionless.
2. **Direction cancellation.** Re-flipping the sign every window made consecutive small
   turns *undo* each other. Small deltas only look like rotation if they **chain**.
3. I deleted the large spawn turn — most of the visible rotation — and replaced it with
   nothing continuous.

### The correction: chain small deltas in a persistent direction, no dead time

New cfg field `direction_flip_prob` (default 0.15): the turn direction **persists**
across resamples and only reverses with that probability. New factory arg
`settle_margin_s` (default **0.0**): no dead time.

```
ang_vel_clip        = 0.12       6.9 deg/s, unchanged
heading_delta       = 15-25 deg  SMALL, gentle, always reachable
direction_flip_prob = 0.15       persists ~6.7 windows = 24 s = ~167 deg one way
settle_margin_s     = 0.0        omega never decays -> continuous rotation
stiffness k         = 2.0        saturated while error > 3.4 deg (almost all of it)
derived resample    = 3.64 s     = delta_max / rate
```

Average rotation is back to the full **6.9°/s continuous** of v10b, but now built from
gentle 15–25° always-reachable targets instead of ±π teleports.

**Why zero dead time is safe here** even though a too-short window was the original bug:
back then the replacement target was *uncorrelated*, so it reversed ω mid-turn. Now the
direction persists, so an arriving target **extends** the current turn instead of
fighting it. Reversals still happen on 15% of windows, but only at a boundary where the
turn just completed — so ω ramps through zero rather than jumping `+w → -w`.

### Status

Still **play-only**. Training keeps the inherited absolute-heading command for now,
since changing the trained task is a separate experiment with its own warm-start risk.
But this is the term training *should* move to — and it is the nav interface, since nav
emits relative course corrections ("turn 30° left around that boulder"), not absolute
compass bearings.

**If it still falls:** the command is no longer a plausible cause. Remaining falls are
policy competence on a slope, consistent with the 72–84% `bad_orientation` already in
the v8/v9b training logs.

### Lesson

Both v10b→v10c regressions came from fixing a *correctness* property (continuity,
reachability) without checking the *quantity* it controlled (degrees per second). Every
command change from here gets its average rotation rate computed before it ships.




**Training** (`_apply_slope_friction_event`) — one variable only:

| change | why |
|---|---|
| `static_friction_range` `(0.6, 0.8)` -> `(0.7, 0.8)` | a wheel holds station only while `mu >= tan(theta)`. `mu=0.6` caps a static hold at **31.0 deg**; Phase SC runs at **30 deg**, so envs sampling near the floor were within ~1 deg of sliding regardless of policy quality. `mu=0.7` gives 35.0 deg — a real margin. Critical because the pure-leg pivot uses the wheels as **anchors**. |

`dynamic_friction_range` left at `(0.5, 0.7)` deliberately — one variable per revision.

### Friction reference

| mu | max slope held statically |
|---|---|
| 0.6 | 31.0 deg |
| **0.7** | **35.0 deg** |
| 0.8 | 38.7 deg |
| 1.0 | 45.0 deg |
| 1.5 | 56.3 deg |

`tan(theta)` is **gravity-independent** — switching to lunar gravity will not relax
this. Holding 45 deg requires `mu >= 1.0`, which the current range does not provide;
38.7 deg is the hard ceiling at the top of `(0.7, 0.8)`.

---

## Design decision on record: pure-leg pivot

Confirmed 2026-07-26: turning is done by the **legs**, with the wheels as anchors.
`wheel_velocity_penalty` ("the wheels are ANCHORS, not actuators, during a spin")
and `pivot_step_coordination` both stay.

**Corollary to accept, not chase:** `pivot_step_coordination` only pays when a foot
has `contact_force < 1.0 N` — i.e. **off the ground** — and it is the largest
positive reward term in the log (+1.32). The foot micro-tapping is therefore
**bought on purpose**. It is not a defect to be trained away.

(The alternative — locking the legs rigid and driving the wheels differentially for
a skid-steer/tank turn — is mechanically available, since the wheels are
`JointVelocityActionCfg` on `.*_foot_joint`. It was considered and rejected.)

---

## Diagnostics still missing

Current logging cannot separate the remaining failure modes:

- `track_ang_vel_z_exp` cannot distinguish "idle because it arrived" from "idle because it failed"
- `bad_orientation` is a single number — it cannot tell **roll** from **pitch**, so the
  observed "kinda backflip" (a pitch failure) cannot be confirmed from logs

Worth adding: heading error, command-saturation fraction, actual-vs-commanded yaw
rate, and roll/pitch separately.

Note: adding these as **zero-weight reward terms will log zeros**, because
`RewardManager` accumulates `func() * weight * dt`. Use a custom command term's
`_update_metrics()` (surfaces under `Metrics/`) or split the termination term.


---

## v10e — REVERT TO v10b. Relative heading is NOT a play-only fix.

User report after v10d: still worse than pre-relative-heading. Less rotation,
not holding position, slipping. Falls reduced (good) but competence worse.

### Root cause of the regression

Training taught a **turn-then-hold** cycle under absolute heading:

```
large heading error -> saturated omega -> ROTATE
error decays -> omega -> 0 -> HOLD / stabilize on the slope
10 s later -> new target -> ROTATE again
```

The HOLD phase is load-bearing on a slope. That is when the policy re-plants
the wheels as anchors, kills residual lateral velocity, and stops downhill
creep. Without it the robot never recovers grip between pivots.

v10d removed that hold on purpose (`settle_margin_s=0`, persistent direction,
`k=2.0`) so `omega` stayed saturated almost forever. On a 10° slope:

- continuous pivot disturbance with no rest
- wheels never get a clean re-anchor window
- body drifts / slips downhill instead of rotating in place
- less visible rotation (energy goes into slip, not yaw)

That matches the visual exactly. **Not a friction bug. Not a policy-competence
bug exposed by a cleaner command.** A play command that deleted the stabilize
phase the checkpoint was trained to use.

### What play is now

Back to **v10b**:

```python
heading_command = False
rel_heading_envs = 0.0
resampling_time_range = (6.0, 6.0)
ranges.ang_vel_z = (-0.12, 0.12)   # sampled, held constant 6 s
```

Why v10b worked better:

- omega held CONSTANT for 6 s — clean tracking, no mid-window decay
- `uniform(-0.12, 0.12)` still samples near zero often enough for rest windows
- no unreachable absolute target → no mid-turn sign flip
- still inside the trained omega magnitude range

### What stays

`RelativeHeadingVelocityCommand` remains in `mdp/commands.py`. It is the right
interface for **nav** ("turn 30° left around that boulder") and the right term
to **train against** eventually. It is **not** wired into play until a
checkpoint is trained against it **with explicit hold phases**.

### Rule

Do not re-introduce relative heading into play without retraining. Play-only
command surgery cannot invent a stabilize skill the weights never learned under
continuous saturated yaw.
