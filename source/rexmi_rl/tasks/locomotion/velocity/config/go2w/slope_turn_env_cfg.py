"""
Slope Turn Policy — v9b: Z-COMPENSATED SPAWN + RESTORED FORWARD REWARD



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
    Apply reward overrides common to ALL slope-turn phases (v7).

    Matches steep_slope / rocky_slope closely:
      1. flat_orientation_l2 = 0.0    — body IS tilted on slope
      2. bad_orientation     = 1.4 rad (80°) — same as steep_slope
      3. ang_vel_xy_l2       = -0.05  — LOCKED (higher kills gait)
      4. lin_vel_z_l2        = -0.3   — relaxed (body moves vertically on slope)
      5. is_alive            = 0.8    — stronger survival signal
      6. position_drift      = 0.8m   — unchanged from v8 (v9-a tried 0.5 and
                                         reverted; that value was derived from an
                                         omega range that no longer applies)
      7. trunk_stability     = 30°    — body naturally tilts on slope
      8. track_lin_vel_xy_exp = 0.5   — RESTORED to the proven flat Phase B value.
                                         Previously 0.0 ("pure pivot"), which was an
                                         UNDOCUMENTED deviation: the slope design doc's
                                         flat->slope reward table never listed this term.
                                         Zeroing it left lin_vel_x = 0.05 in the
                                         observation with no reward backing, so the
                                         policy was told to creep but never paid for it.
                                         That creep is what generates the front/rear
                                         lateral leg coordination the pivot gait needs
                                         (front feet +y, rear feet -y). Without it the
                                         gait degenerates into foot-tapping and drift.


    NOTE: No explicit lean reward (rocky_slope Phase 8g lesson).
    2cm roughness + friction=0.7–0.8 physically force the correct lean posture.

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

    # Kept at 0.8 (v8 value). v9-a briefly tried 0.5, derived from an orbit
    # radius that assumed omega in (0.25, 0.35) — but that omega range was
    # reverted (see _apply_slope_command_overrides), so the derivation no
    # longer applies. Not changed, to keep this run single-variable.
    if hasattr(cfg.rewards, "position_drift"):
        cfg.rewards.position_drift.params["drift_threshold"] = 0.8

    if hasattr(cfg.rewards, "trunk_stability"):
        cfg.rewards.trunk_stability.params["max_tilt_deg"] = 30.0

    # RESTORED to proven flat Phase B value. Was 0.0, which stripped the reward
    # backing from the lin_vel_x=0.05 creep that drives front/rear leg coordination.
    # A log showing track_lin_vel_xy_exp == 0.0000 means this mechanism is dead.
    if hasattr(cfg.rewards, "track_lin_vel_xy_exp"):
        cfg.rewards.track_lin_vel_xy_exp.weight = 0.5

    cfg.terminations.bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 1.4},   # 80° — headroom for slope-induced sway
    )


def _apply_slope_command_overrides(cfg) -> None:
    """
    Command overrides: minimal stability creep, no lateral motion.

    ang_vel_z is deliberately NOT set here — it inherits (-0.2, 0.2) from
    Go2wTurnBEnvCfg, which is the range the warm-start checkpoint was trained
    on.  See the FAILED EXPERIMENT note below before changing this.

    FAILED EXPERIMENT (v9-a, 2026-07-26) — DO NOT REPEAT
    ---------------------------------------------------
    Hypothesis: a body with forward velocity v and yaw rate omega orbits at
    radius r = v/omega, so sampling omega ~= 0 makes r -> infinity and
    degenerates the orbit into a straight-line downhill creep that topples the
    robot.  Proposed fix: bound omega away from zero, omega in (0.25, 0.35).

    RESULT: CATASTROPHIC. Ran from model_12100 for 50 iterations:

        metric                 v8 baseline     v9-a
        bad_orientation        72-84%          99.9%
        mean episode length    ~150            52 and falling
        pivot_step_coord       1.08 peak       0.127 and falling
        track_ang_vel_z_exp    small           0.008 (robot barely turns)

    WHY IT FAILED — the hypothesis was untested and the range was
    OUT OF DISTRIBUTION.  Go2wTurnBEnvCfg (the warm-start parent, line ~415)
    trains omega in (-0.2, 0.2).  Demanding (0.25, 0.35) put EVERY sample
    above the largest yaw rate the policy had ever experienced, while also
    removing omega ~= 0 — the near-stationary regime that was the stable
    majority of its competence.  The policy was asked to do something strictly
    harder than anything in its history, starting from weights that had never
    seen it.  It stopped turning altogether and fell essentially every episode.

    (The apparent precedent, Go2wTurnEnvCfg_PLAY at (0.25, 0.3), belongs to a
    DIFFERENT lineage — the pure-pivot branch with lin_vel_x = 0.0 — not to
    Phase B.  It does not license this range here.)

    LESSONS
      1. When warm-starting, keep the command distribution INSIDE the range
         the checkpoint was trained on.  Widening or shifting it discards the
         transfer you are warm-starting for.
      2. Mean reward alone is not progress.  v9-a's reward "improved"
         -30 -> -2 purely because episodes got SHORTER, so less penalty
         accumulated.  Always read reward together with episode length.
      3. Change ONE variable per run.  v9-a moved three at once and could not
         attribute the failure without a revert.

    If bounding omega away from zero is worth retesting later, do it INSIDE
    the trained range — e.g. (0.1, 0.2) — and as the only change in that run.
    """
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.05, 0.05)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)


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
    PLAY-ONLY command shaping.  Does NOT affect training.

    ---------------------------------------------------------------------------
    v10e — REVERT TO v10b CONSTANT SAMPLED YAW (2026-07-26)
    ---------------------------------------------------------------------------
    Relative-heading play (v10c/v10d) was a regression. User report after v10d:
      "still worse than before when heading targets were not continuous"
      "not maintaining its position as before and is not rotating as much"
      "I think its not rotating as it keeps on slipping"

    WHY RELATIVE HEADING FAILED AS A PLAY DEMO
    ------------------------------------------
    Training taught the policy a TURN-THEN-HOLD cycle under absolute heading:

        large heading error -> saturated omega -> ROTATE
        error decays -> omega -> 0 -> HOLD / stabilize on the slope
        10 s later -> new target -> ROTATE again

    The HOLD phase is load-bearing on a slope: it is when the policy re-plants
    the wheels as anchors, kills residual lateral velocity, and stops downhill
    creep. Without it the robot never recovers grip between pivots.

    v10d removed that hold on purpose (settle_margin_s=0, persistent direction,
    k=2.0) so omega stayed saturated almost forever. Result on a 10 deg slope:
      * continuous pivot disturbance with no rest
      * wheels never get a clean re-anchor window
      * body drifts / slips downhill instead of rotating in place
      * less visible rotation (energy goes into slip, not yaw)

    That matches the visual exactly. It is NOT a friction bug and NOT a
    policy-competence bug exposed by a cleaner command — it is a play command
    that deleted the stabilize phase the checkpoint was trained to use.

    v10b (heading_command=False, constant sampled omega) worked better because:
      * omega is held CONSTANT for 6 s — clean tracking, no mid-window decay
      * uniform(-0.12, 0.12) still samples near zero ~often enough to give
        natural rest windows
      * no unreachable absolute target, so no mid-turn sign flip
      * still inside the trained omega magnitude range

    So PLAY goes back to v10b. RelativeHeadingVelocityCommand stays in
    mdp/commands.py for a FUTURE training experiment (and for the nav
    interface), but it is not wired into play until a checkpoint is trained
    against it with explicit hold phases.

    DO NOT re-introduce relative heading into play without retraining.
    """
    # v10b: sample omega directly. No heading target => nothing unreachable,
    # no mid-turn sign reversal. Symmetric clip, spans zero, in-distribution.
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.rel_heading_envs = 0.0
    cfg.commands.base_velocity.resampling_time_range = (6.0, 6.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (-0.12, 0.12)
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
    """Phase SA play config: fewer envs, no noise, denser + slower turn command."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 8.0
        self.observations.policy.enable_corruption = False
        self.curriculum.terrain_levels = None
        _apply_slope_play_overrides(self)


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
    """Phase SB play config: denser + slower turn command."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 8.0
        self.observations.policy.enable_corruption = False
        self.curriculum.terrain_levels = None
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
    """Phase SC play config: denser + slower turn command."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 8.0
        self.observations.policy.enable_corruption = False
        self.curriculum.terrain_levels = None
        _apply_slope_play_overrides(self)
