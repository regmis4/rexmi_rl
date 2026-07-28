"""
Slope Turn Policy — v12: FIX NEGATIVE-YAW (D1− failure) from model_12349



===============================================================

HISTORY OF FAILED APPROACHES
-----------------------------
v1 (tilted gravity, vx=(0,0.2)):
  Exploited tilted gravity — walked downhill then snap-turned.

v2 (tilted gravity, vx=0):
  Phase B gait destabilised without forward creep. bad_orientation 99%.

v3 (tilted gravity, vx=0.05):
  Creep helped but bad_orientation still 98%.

v4 (tilted gravity, vx=0.05, bad_orientation=1.4rad):
  21% bad_orientation — robots survived but visual check showed
  intermittent pulse-turns with NO body alignment to slope.
  Root cause: flat terrain + tilted gravity = flat foot contact.

v5 (HfPyramidSlopedTerrainCfg, platform_width=0.0, spawn 1m from center):
  bad_orientation 100%, episode_length 32 steps.
  Root cause: spawn 1m down slope = 0.18m height drop, instantly fatal.

v6 (HfPyramidSlopedTerrainCfg, platform_width=1.0, spawn x=(0.3,0.7),
    friction (0.3,0.5), 20% random rough tiles):
  Visual check issues:
    1. Platform too large — robots spawning entirely on flat top, no slope contact
    2. Robot slides without wheel rotation — friction (0.3,0.5) too low to train
    3. Random rough tiles wrong — need uniform roughness on ALL pyramid tiles

CORRECT APPROACH (v7): RockyPyramidSlopeCfg + TIGHT STRADDLE SPAWN
--------------------------------------------------------------------
Use RockyPyramidSlopeCfg from crater_terrain.py:
  - Real pyramid slope geometry (same as HfPyramidSlopedTerrainCfg)
  - Built-in surface roughness: 2cm uniform noise → wheel bite without boulders
  - NO boulders: boulder_count_min=0, boulder_count_max=0
  - platform_width=0.5m → small flat top, edge at 0.25m from center

Spawn at x=(0.4, 0.8) from tile center:
  - Body center 0.4–0.8m from tile center, clearly on slope
  - Platform edge at 0.25m → robot is fully on slope at x=0.4
  - Front leg at x+0.2 = 0.6m, rear leg at x-0.2 = 0.2m (near/on edge)

Friction raised to trainable range:
  - static (0.7, 0.8): raised from (0.6, 0.8) in v10 — see
    _apply_slope_friction_event. mu=0.6 caps a static hold at 31.0°, which
    made Phase SC (30°) physically borderline regardless of policy quality.
  - dynamic (0.5, 0.7): 2cm roughness provides mechanical wheel bite
  - Still at/below rocky_slope (0.8–1.5) for lunar realism


WHY 2cm ROUGHNESS:
  The rocky_slope Phase 8g lesson: "The terrain now physically rewards the
  robot for proper gait without a posture-specific reward term."
  2cm bumps give wheels purchase on the slope without boulders that would
  destabilise an early-stage policy. The rough surface creates micro-contacts
  that allow wheel locking to actually work.

CURRICULUM
----------
Phase SA: 10°  slope, warm-start from flat Phase B (model_10992.pt)
Phase SB: 20°  slope, warm-start from Phase SA
Phase SC: 30°  slope, warm-start from Phase SB

SPAWN DESIGN
------------
  pose_range (see the Z-COMPENSATION block further down — v8 fix):
    x = (1.0, 1.2)       # NARROW band so Z compensation stays accurate
    y = (-0.4, 0.4)      # |y| < x so Chebyshev distance stays driven by x
    z = -dist*tan(slope) # spawn ON the surface, not in mid-air
    yaw = (-3.14, 3.14)  # all headings vs slope

v8 (Z-compensated spawn, ang_vel_z inherited as (-0.2, 0.2), vx reward = 0):
  Z fix WORKED — base_height -1.05 -> -0.28, mean reward -11 -> +8.4,
  foot_alternation flipped positive (real gait emerging).
  BUT bad_orientation only 75% -> 72%, then the run DIVERGED:
  peaked at iter ~12120 (+8.4 reward, pivot_step_coord 1.08) and collapsed
  monotonically to iter 12489 (-12.2 reward, pivot_step_coord 0.65,
  bad_orientation 84.5%, action std 1.07 -> 1.31, surrogate loss ~= 0).

v9-a (FAILED, REVERTED — ang_vel_z forced to (0.25, 0.35)):
  Hypothesised that inheriting ang_vel_z = (-0.2, 0.2) was the problem
  because it SPANS ZERO, and omega ~= 0 makes the orbit radius r = v/omega
  diverge into a straight-line downhill creep.
  RESULT: made everything dramatically worse — bad_orientation 99.9%,
  episode length collapsed to 52, track_ang_vel_z_exp fell to 0.008 (the
  robot stopped turning entirely).
  ROOT CAUSE OF THE FAILURE: (0.25, 0.35) is OUTSIDE the warm-start
  distribution. Go2wTurnBEnvCfg trains omega in (-0.2, 0.2), so every
  commanded yaw rate exceeded anything the checkpoint had ever seen, while
  simultaneously removing the near-stationary regime that was the bulk of
  its competence. Full analysis in `_apply_slope_command_overrides`.
  The orbit-radius maths is sound; the inference that omega ~= 0 caused the
  v8 falls was never tested, and the chosen range broke warm-start transfer.

CURRENT APPROACH (v9b): ONE VARIABLE — RESTORE THE FORWARD REWARD
------------------------------------------------------------------
After the v9-a revert, exactly ONE environment change remains relative to
v8, chosen because it is the only one backed by an already-proven value:

Root cause — track_lin_vel_xy_exp was zeroed (UNDOCUMENTED).

  The slope design doc lists six intended flat->slope reward changes; this
  term is NOT among them. Flat Phase B proved +0.5 ("match small forward
  velocity"). Zeroing it left lin_vel_x=0.05 in the observation with no
  reward backing — the policy was told to creep but never paid for it. That
  creep is what generates the front(+y)/rear(-y) lateral leg coordination the
  pivot gait depends on. Without it the gait degenerates into foot-tapping
  and drift, which is what the visual eval showed.
  FIX: track_lin_vel_xy_exp = 0.5 — restored.

Also changed (TRAINING side only, orthogonal to the environment):

  With most rollouts failing the advantage signal collapsed (surrogate ~= 0),
  so entropy_coef=0.01 became the dominant gradient and inflated action std
  1.07 -> 1.31, degrading the policy further. Self-reinforcing.
  FIX: entropy_coef = 0.002, max_iterations = 250 (see rsl_rl_ppo_cfg.py).

REWARD OVERRIDES (matching steep_slope / rocky_slope closely)
-------------------------------------------------------------
  flat_orientation_l2  = 0.0    — body MUST tilt on slope
  bad_orientation      = 1.4 rad (80°) — same as steep_slope
  is_alive             = 0.8    — stronger survival signal
  track_lin_vel_xy_exp = 0.5    — RESTORED (proven flat Phase B value)
  ang_vel_xy_l2        = -0.05  — LOCKED (v7 lesson)
  lin_vel_z_l2         = -0.3   — relaxed for pivot on slope
  position_drift       = 0.8m   — unchanged from v8
  trunk_stability      = 30°    — body tilts naturally on slope
  lin_vel_x = (0.05, 0.05)     — stability creep (now actually rewarded)
  ang_vel_z = INHERITED (-0.2, 0.2) from Go2wTurnBEnvCfg — do NOT override;
                                 see the v9-a failure note in
                                 _apply_slope_command_overrides

FRICTION
--------
  static  = (0.7, 0.8)   — v10: floor raised from 0.6 (mu=0.6 caps a static
                           hold at 31.0°, borderline for Phase SC at 30°)
  dynamic = (0.5, 0.7)   — unchanged


HEALTH SIGNALS (Phase SA — v9b)
-------------------------------------
Read these TOGETHER. v9-a looked like it was improving on mean reward alone
while actually collapsing, because shorter episodes accumulate less penalty.

  MEAN EPISODE LENGTH   → the primary signal. Must RISE (or hold ~150).
                          If it falls while reward rises, the policy is
                          dying faster, not behaving better.
  bad_orientation       → must fall below 72% (the v8 peak). 99% = collapse.
  track_ang_vel_z_exp   → must stay clearly > 0. Near 0 means the robot has
                          stopped turning at all (the v9-a failure mode).
  track_lin_vel_xy_exp  → MUST BE > 0. It was exactly 0.0000 in every v7/v8
                          log, the signature of the disabled coordination
                          mechanism. 0.0000 means the override did not apply.
  pivot_step_coord      → rising toward >= 1.0 (the v8 peak was 1.08)
  action noise std      → FLAT OR FALLING, never climbing (v8 diverged
                          1.07 -> 1.31 and never recovered)

TRAIN COMMANDS
--------------
  # Phase SA (10° slope)
  python scripts/train.py --task RexmiRl-Go2w-Velocity-SlopeTurnA-v0 --headless \\
      --load_run go2w_velocity_turn_b/2026-07-25_13-51-28 \\
      --checkpoint model_10992.pt

  # Phase SB (20°)
  python scripts/train.py --task RexmiRl-Go2w-Velocity-SlopeTurn-v0 --headless \\
      --load_run go2w_velocity_slope_turn_a/<date> --checkpoint model_<N>.pt

  # Phase SC (30°)
  python scripts/train.py --task RexmiRl-Go2w-Velocity-SlopeTurnC-v0 --headless \\
      --load_run go2w_velocity_slope_turn/<date> --checkpoint model_<N>.pt

PLAY COMMANDS
-------------
  python scripts/play.py --task RexmiRl-Go2w-Velocity-SlopeTurnA-Play-v0 \\
      --load_run go2w_velocity_slope_turn_a/<date> \\
      --checkpoint /path/to/model_<N>.pt
"""

from __future__ import annotations

import math

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EvtTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass

from rexmi_rl.tasks.locomotion.velocity.config.go2w.crater_terrain import (
    RockyPyramidSlopeDownCfg,
)
from rexmi_rl.tasks.locomotion.velocity.config.go2w.turn_env_cfg import (
    Go2wTurnBEnvCfg,
)
from rexmi_rl.tasks.locomotion.velocity.mdp import terminations as rexmi_term



def _make_slope_terrain(slope_deg: float) -> TerrainGeneratorCfg:
    """
    Create terrain: 100% RockyPyramidSlopeDownCfg with 2cm roughness, no boulders.

    RockyPyramidSlopeDownCfg (from crater_terrain.py):
      - HILL geometry: peak (flat 0.5m platform) at tile CENTER, slopes fall OUTWARD
      - h = max_h - tan(slope) * dist_from_platform → center is highest, edges lowest
      - This is the correct geometry: robot spawns on hillside below the summit
      - NOTE: RockyPyramidSlopeCfg is a BOWL (low center, high edges — crater wall).
        We use the Down variant which inverts this to a hill.
      - Built-in surface roughness (uniform noise ±roughness/2 per pixel)
      - platform_width=0.5m → small flat summit, edge at 0.25m from tile center
      - slope_min_deg = slope_max_deg = slope_deg → fixed slope, no curriculum
      - roughness_min_m = roughness_max_m = 0.02 → fixed 2cm on all tiles
      - boulder_count_min = boulder_count_max = 0 → NO boulders — pure rough hillside

    num_rows=1: single difficulty row (slope fixed, not curriculum-scaled)
    num_cols=20: 20 parallel tiles
    """
    return TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=0.0,
        num_rows=1,
        num_cols=20,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=None,
        sub_terrains={
            "slope": RockyPyramidSlopeDownCfg(
                proportion=1.0,
                slope_min_deg=slope_deg,
                slope_max_deg=slope_deg,   # fixed angle — no within-phase variation
                platform_width=0.5,        # small flat summit, edge at 0.25m from center
                roughness_min_m=0.02,      # fixed 2cm roughness (not difficulty-scaled)
                roughness_max_m=0.02,
                boulder_count_min=0,       # NO boulders — pure rough slope
                boulder_count_max=0,
                size=(8.0, 8.0),
                seed=99,                   # explicit — rocky_pyramid_slope_down uses seed+1000
            ),
        },
    )


# ---------------------------------------------------------------------------
# SPAWN GEOMETRY — Z COMPENSATION (v8 CRITICAL FIX)
# ---------------------------------------------------------------------------
# THE BUG THAT KILLED v7 (SA 75% / SB 100% bad_orientation):
#
#   RockyPyramidSlopeDownCfg is a PYRAMID — terrain height varies with
#   distance from the tile centre.  Isaac Lab's height_field_to_mesh places
#   the sub-terrain origin at the centre of the tile, with
#       origin_z = max height in the central 1 m × 1 m patch
#                = the flat platform (peak) height.
#
#   `pose_range["x"]` displaces the robot HORIZONTALLY from that origin but
#   does NOT adjust Z.  The terrain beneath the displaced robot is lower by
#       drop = dist_from_platform × tan(slope)
#   so the robot SPAWNS IN MID-AIR and free-falls onto the slope.
#
#   v7 used x=(0.8, 1.5) → dist_from_platform = 0.55–1.25 m:
#       10° slope → drop = 0.10–0.22 m  →  75 % bad_orientation
#       20° slope → drop = 0.20–0.45 m  → 100 % bad_orientation
#
#   This is why the yaw constraint made ZERO difference (75.1 → 75.5 %):
#   the robot was never falling because of its heading — it was falling
#   because it was dropped onto the slope from up to 45 cm.
#   It also explains the hard, un-learnable ceiling: no policy can train
#   its way out of being spawned in mid-air.
#
#   Reference: crater_env_cfg.py already does exactly this compensation
#   (lines 275 / 382 / 488) — slope_turn_env_cfg.py simply omitted it.
#
# THE FIX:
#   1. Tighten x to a NARROW band so dist_from_platform is nearly constant.
#   2. Set pose_range["z"] = −dist × tan(slope) so the robot spawns exactly
#      on the terrain surface.
#   3. Keep |y| < x so the pyramid's L∞ (Chebyshev) distance metric stays
#      driven by x — otherwise y would change dist_from_platform too.
#
#   x = (1.0, 1.2) → dist_from_platform = 0.75–0.95 m, mean 0.85 m
#   Residual spawn error: ±1.8 cm (was ±22 cm at 10°, ±45 cm at 20°)
# ---------------------------------------------------------------------------

_SPAWN_X_MIN = 1.0      # metres from tile centre
_SPAWN_X_MAX = 1.2
_PLATFORM_HALF = 0.25   # platform_width / 2 — flat summit half-width

# Mean distance from the platform EDGE (this is what drives terrain height)
_SPAWN_MEAN_DIST = 0.5 * (_SPAWN_X_MIN + _SPAWN_X_MAX) - _PLATFORM_HALF   # 0.85 m


def _apply_slope_spawn(cfg, slope_deg: float) -> None:
    """
    Set spawn pose with Z compensation for the pyramid slope height.

    Places the robot on a narrow band of the slope face and lowers it by
    exactly the terrain drop at that band, so it spawns ON the surface
    rather than in mid-air.

    Z offset for each phase:
        10° → −0.150 m
        20° → −0.309 m
        30° → −0.491 m

    Yaw is FULL RANGE (−π, π): we proved heading was never the failure
    cause, and the turn policy must handle every slope-relative heading.
    """
    z_offset = -_SPAWN_MEAN_DIST * math.tan(math.radians(slope_deg))

    cfg.events.reset_base.params["pose_range"]["x"] = (_SPAWN_X_MIN, _SPAWN_X_MAX)
    cfg.events.reset_base.params["pose_range"]["y"] = (-0.4, 0.4)   # |y| < x
    cfg.events.reset_base.params["pose_range"]["z"] = (z_offset, z_offset)
    cfg.events.reset_base.params["pose_range"]["yaw"] = (-3.14, 3.14)


def _apply_slope_reward_overrides(cfg) -> None:
    """
    Apply reward overrides common to ALL slope-turn phases.

    ---------------------------------------------------------------------------
    v11 — STATION-KEEPING (from model_12349 baseline)
    ---------------------------------------------------------------------------
    model_12349 rotates on 10° but: intermittent somersault/sideways flips,
    constant microstepping, mild position loss. Diagnosis (2026-07-27):

      * Friction is NOT the bottleneck at 10° (mu=0.7 holds ~35°).
      * Roughness 2→4 cm is the WRONG first move (more disturbance under pivot).
      * Microstepping is BOUGHT by pivot_step_coord / foot_alternation — keep it.
      * Position loss is ALLOWED: slope drift threshold was 0.8 m vs flat-B 0.25 m.
      * trunk_stability was loosened to 30° — permits lean-into-flip.
      * Random absolute heading in TRAINING is a real issue, but play diagnostics
        (D1 one-direction / D2 omega=0) must settle command vs reward first.

    v11 changes ONE theme only — station-keeping. No roughness/friction/omega
    range change. Warm-start from model_12349.

      position_drift threshold  0.8 → 0.35 m   (back toward flat-B discipline)
      position_drift weight    -0.5 → -1.0     (wander is expensive)
      trunk_stability          30° → 20°       (still above slope tilt; stops
                                                extreme lean-into-flip)

    Also wires bad_pitch / bad_roll terminations (same 1.4 rad limit as
    bad_orientation) so TensorBoard can separate somersault vs sideways falls.
    bad_orientation is KEPT as the primary combined tripwire.
    """
    if hasattr(cfg.rewards, "flat_orientation_l2"):
        cfg.rewards.flat_orientation_l2.weight = 0.0

    # LOCKED — lateral ang vel IS the micro-tap stepping gait
    if hasattr(cfg.rewards, "ang_vel_xy_l2"):
        cfg.rewards.ang_vel_xy_l2.weight = -0.05

    if hasattr(cfg.rewards, "lin_vel_z_l2"):
        cfg.rewards.lin_vel_z_l2.weight = -0.3

    if hasattr(cfg.rewards, "is_alive"):
        cfg.rewards.is_alive.weight = 0.8

    # v11 station-keeping: tighten drift (was 0.8 m — allowed visible wander)
    if hasattr(cfg.rewards, "position_drift"):
        cfg.rewards.position_drift.params["drift_threshold"] = 0.35
        cfg.rewards.position_drift.weight = -1.0

    # v11: 20° still above natural 10° body tilt on a 10° slope; stops extreme lean
    if hasattr(cfg.rewards, "trunk_stability"):
        cfg.rewards.trunk_stability.params["max_tilt_deg"] = 20.0

    # RESTORED forward tracking (v9b) — creep drives pivot leg coordination
    if hasattr(cfg.rewards, "track_lin_vel_xy_exp"):
        cfg.rewards.track_lin_vel_xy_exp.weight = 0.5

    # Combined orientation tripwire (unchanged limit)
    cfg.terminations.bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 1.4},   # 80°
    )
    # Split failure modes — same limit, separate counters in TensorBoard
    cfg.terminations.bad_pitch = DoneTerm(
        func=rexmi_term.bad_pitch,
        params={"limit_angle": 1.4},
    )
    cfg.terminations.bad_roll = DoneTerm(
        func=rexmi_term.bad_roll,
        params={"limit_angle": 1.4},
    )


def _apply_slope_command_overrides(cfg) -> None:
    """
    Command overrides for slope-turn TRAINING.

    ---------------------------------------------------------------------------
    v12 — D1 DIAGNOSTICS REWROTE THE COMMAND STRATEGY (2026-07-27)
    ---------------------------------------------------------------------------
    model_12349 on 10° slope:
      D1+  ω=+0.12 constant  → picture-perfect spin + station-keeping
      D1−  ω=−0.12 constant  → catastrophic back-to-back (pitch-backward) falls
      D2   ω=0 hold          → better than baseline; grip is fine at 10°

    So the policy already CAN slope-pivot — only in the +ω direction.
    Baseline intermittency was random sign mixing a working skill with a
    broken one. Friction/roughness are not the bottleneck at 10°.

    TRAINING COMMAND (not play):
      heading_command = False
        → ω is SAMPLED and held, matching the D1 interface that works.
        → Avoids absolute-heading mid-turn teleports (the old train default).
      lin_vel_x = 0.05 creep (unchanged — stability + pivot coordination)
      lin_vel_y = 0
      ang_vel_z = NEGATIVE WINDOW by default for SA-v12a repair runs
        → (-0.12, -0.08): focused CW skill, still inside prior magnitude
        → heading_command=False makes this safe (v9-a only applies when
          heading_command=True turns ang_vel_z into a signed clip)

    After a successful v12a checkpoint, flip to symmetric mix via
    `_apply_slope_command_overrides_symmetric` (SA-v12b) — do not stay
    forever one-sided; nav needs both turn directions.

    FAILED EXPERIMENT (v9-a) — still do not repeat under heading_command=True:
    asymmetric ang_vel_z clip with heading P-control is unsatisfiable.
    """
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.rel_heading_envs = 0.0
    cfg.commands.base_velocity.resampling_time_range = (6.0, 6.0)
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.05, 0.05)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    # SA-v12a default: repair the broken CW / negative-yaw direction.
    # Magnitudes inside what D1 used successfully on the other sign (0.12).
    cfg.commands.base_velocity.ranges.ang_vel_z = (-0.12, -0.08)


def _apply_slope_command_overrides_symmetric(cfg) -> None:
    """
    SA-v12b: both turn directions, still sampled ω (heading_command=False).

    Use only AFTER v12a has produced a checkpoint that survives D1−.
    """
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.rel_heading_envs = 0.0
    cfg.commands.base_velocity.resampling_time_range = (6.0, 6.0)
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.05, 0.05)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (-0.12, 0.12)


def _apply_slope_friction_event(cfg) -> None:
    """
    Friction randomization: trainable lunar-like range.

    WHY THE STATIC FLOOR WAS RAISED 0.6 → 0.7 (v10)
    -----------------------------------------------
    A wheel holds station on a slope only while  mu >= tan(theta).  With the
    pure-leg pivot strategy the wheels are ANCHORS (see wheel_velocity_penalty),
    so the whole turn depends on them NOT sliding — friction is not a detail
    here, it is the precondition.

        mu    max slope it can statically hold
        0.6   31.0 deg   <-- old floor
        0.7   35.0 deg   <-- new floor
        0.8   38.7 deg
        1.0   45.0 deg

    The old floor of 0.6 caps a static hold at 31.0 deg.  Phase SC runs at
    30 deg, so any env that sampled near mu=0.6 was within ~1 deg of sliding
    NO MATTER HOW GOOD THE POLICY IS — an unwinnable env the policy still
    gets penalised for.  Raising the floor to 0.7 (35.0 deg) restores a real
    margin at 30 deg and is the precondition for the higher-slope curriculum.

    Note tan(theta) is GRAVITY-INDEPENDENT, so switching to lunar gravity
    later will not relax this — mu must carry the slope on its own.
    Holding 45 deg would need mu >= 1.0, which this range still does NOT
    provide; 38.7 deg is the hard ceiling at the top of the range.

    static  = (0.7, 0.8): holds 35.0-38.7 deg — covers SA 10 / SB 20 / SC 30
    dynamic = (0.5, 0.7): UNCHANGED, and deliberately so. Only one variable
                          moves in this revision (the v9-a lesson: never
                          change several at once). dynamic <= static remains
                          physically valid.
    """
    cfg.events.randomize_robot_friction = EvtTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.7, 0.8),
            "dynamic_friction_range": (0.5, 0.7),

            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )


def _apply_slope_play_overrides(cfg) -> None:
    """
    PLAY-ONLY baseline command (v10e / v10b). Does NOT affect training.

    Constant sampled yaw — cleaner than training's absolute heading, still
    symmetric, still in-distribution magnitude. Use this for all baseline
    visual comparisons against model_12349.

    Diagnostic variants (one-direction / hold) live in dedicated PLAY classes
    below — do not bake those into this baseline.
    """
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.rel_heading_envs = 0.0
    cfg.commands.base_velocity.resampling_time_range = (6.0, 6.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (-0.12, 0.12)
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.05, 0.05)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)


def _apply_slope_play_common(cfg) -> None:
    """Shared PLAY scene settings (few envs, no noise, no curriculum)."""
    cfg.scene.num_envs = 16
    cfg.scene.env_spacing = 8.0
    cfg.observations.policy.enable_corruption = False
    cfg.curriculum.terrain_levels = None


def _apply_slope_play_diag_one_dir(cfg, omega: float) -> None:
    """
    D1 DIAGNOSTIC — fixed one-direction yaw. PLAY ONLY. Do not train on this.

    If flips drop a lot vs baseline → command reversals / slope-direction
    asymmetry is a major piece of the intermittent failure.
    If flips stay → continuous-pivot competence / reward / posture.
    """
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.rel_heading_envs = 0.0
    cfg.commands.base_velocity.resampling_time_range = (10.0, 10.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (omega, omega)
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.05, 0.05)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)


def _apply_slope_play_diag_hold(cfg) -> None:
    """
    D2 DIAGNOSTIC — omega=0 hold on the slope. PLAY ONLY.

    If the robot holds station → grip/physics OK; problem is turn skill.
    If it creeps/slides a lot → station-keeping / wheel lock weak even
    without turning (reward/physics, not heading command).
    """
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.rel_heading_envs = 0.0
    cfg.commands.base_velocity.resampling_time_range = (20.0, 20.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.05, 0.05)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)


# ===========================================================================
# PHASE SA: 10° rocky pyramid slope
# ===========================================================================


@configclass
class Go2wSlopeTurnAEnvCfg(Go2wTurnBEnvCfg):
    """
    Phase SA: Pivot turn on a 10° rocky pyramid slope.

    Terrain: RockyPyramidSlopeCfg, slope=10°, platform_width=0.5m, roughness=2cm.
    Spawn: x=(1.0, 1.2) — narrow band, Z-compensated (see _apply_slope_spawn).
           The old x=(0.8, 1.5) is the v7 value that spawned the robot in mid-air.
    Friction: static (0.7, 0.8) — see _apply_slope_friction_event.
    Warm-start: flat Phase B (model_10992.pt).
    """

    def __post_init__(self):
        super().__post_init__()

        # ==============================================================
        # TERRAIN: 10° rocky pyramid slope, 2cm roughness, no boulders
        # ==============================================================
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = _make_slope_terrain(10.0)
        self.scene.env_spacing = 8.0

        self.curriculum.terrain_levels = None
        self.sim.gravity = (0.0, 0.0, -9.81)

        # SPAWN: on-surface with Z compensation (z = -0.150 m at 10°)
        _apply_slope_spawn(self, 10.0)

        # ==============================================================
        # FRICTION + REWARDS + COMMANDS
        # ==============================================================
        _apply_slope_friction_event(self)
        _apply_slope_reward_overrides(self)
        _apply_slope_command_overrides(self)

        if hasattr(self.events, "push_robot"):
            self.events.push_robot = None


@configclass
class Go2wSlopeTurnAEnvCfg_PLAY(Go2wSlopeTurnAEnvCfg):
    """Phase SA play baseline (v10e constant sampled yaw)."""

    def __post_init__(self):
        super().__post_init__()
        _apply_slope_play_common(self)
        _apply_slope_play_overrides(self)


@configclass
class Go2wSlopeTurnAEnvCfg_PLAY_D1_POS(Go2wSlopeTurnAEnvCfg):
    """D1 diagnostic: constant +0.12 rad/s yaw (left). PLAY ONLY."""

    def __post_init__(self):
        super().__post_init__()
        _apply_slope_play_common(self)
        _apply_slope_play_diag_one_dir(self, +0.12)


@configclass
class Go2wSlopeTurnAEnvCfg_PLAY_D1_NEG(Go2wSlopeTurnAEnvCfg):
    """D1 diagnostic: constant -0.12 rad/s yaw (right). PLAY ONLY."""

    def __post_init__(self):
        super().__post_init__()
        _apply_slope_play_common(self)
        _apply_slope_play_diag_one_dir(self, -0.12)


@configclass
class Go2wSlopeTurnAEnvCfg_PLAY_D2_HOLD(Go2wSlopeTurnAEnvCfg):
    """D2 diagnostic: omega=0 station-hold on 10° slope. PLAY ONLY."""

    def __post_init__(self):
        super().__post_init__()
        _apply_slope_play_common(self)
        _apply_slope_play_diag_hold(self)


@configclass
class Go2wSlopeTurnAEnvCfg_V12B(Go2wSlopeTurnAEnvCfg):
    """
    SA-v12b: symmetric sampled ω after negative-yaw repair.

    Warm-start from a v12a checkpoint that survives D1−. Same rewards/terrain
    as SA; only the yaw command range returns to (−0.12, +0.12).
    """

    def __post_init__(self):
        super().__post_init__()
        # Parent already applied v12a negative-only via _apply_slope_command_overrides.
        # Replace with symmetric mix.
        _apply_slope_command_overrides_symmetric(self)


# ===========================================================================
# PHASE SB: 20° rocky pyramid slope
# ===========================================================================

@configclass
class Go2wSlopeTurnEnvCfg(Go2wTurnBEnvCfg):
    """
    Phase SB: Pivot turn on a 20° rocky pyramid slope.

    Warm-start from Phase SA. Slope 20° — visible body tilt required.
    """

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = _make_slope_terrain(20.0)
        self.scene.env_spacing = 8.0
        self.curriculum.terrain_levels = None
        self.sim.gravity = (0.0, 0.0, -9.81)

        # SPAWN: on-surface with Z compensation (z = -0.309 m at 20°)
        _apply_slope_spawn(self, 20.0)

        _apply_slope_friction_event(self)
        _apply_slope_reward_overrides(self)
        _apply_slope_command_overrides(self)

        if hasattr(self.events, "push_robot"):
            self.events.push_robot = None


@configclass
class Go2wSlopeTurnEnvCfg_PLAY(Go2wSlopeTurnEnvCfg):
    """Phase SB play baseline (v10e constant sampled yaw)."""

    def __post_init__(self):
        super().__post_init__()
        _apply_slope_play_common(self)
        _apply_slope_play_overrides(self)


# ===========================================================================
# PHASE SC: 30° rocky pyramid slope
# ===========================================================================

@configclass
class Go2wSlopeTurnCEnvCfg(Go2wTurnBEnvCfg):
    """
    Phase SC: Pivot turn on a 30° rocky pyramid slope.

    Warm-start from Phase SB. Slope 30° — approaching crater wall angles.
    """

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = _make_slope_terrain(30.0)
        self.scene.env_spacing = 8.0
        self.curriculum.terrain_levels = None
        self.sim.gravity = (0.0, 0.0, -9.81)

        # SPAWN: on-surface with Z compensation (z = -0.491 m at 30°)
        _apply_slope_spawn(self, 30.0)

        _apply_slope_friction_event(self)
        _apply_slope_reward_overrides(self)
        _apply_slope_command_overrides(self)

        if hasattr(self.events, "push_robot"):
            self.events.push_robot = None


@configclass
class Go2wSlopeTurnCEnvCfg_PLAY(Go2wSlopeTurnCEnvCfg):
    """Phase SC play baseline (v10e constant sampled yaw)."""

    def __post_init__(self):
        super().__post_init__()
        _apply_slope_play_common(self)
        _apply_slope_play_overrides(self)
