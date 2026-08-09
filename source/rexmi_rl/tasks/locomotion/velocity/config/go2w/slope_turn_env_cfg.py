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

  # Phase SB (20°) — warm-start from balanced SA-v12b (symmetric ω), NOT v12a
  #   ./run.sh scripts/train.py --task RexmiRl-Go2w-Velocity-SlopeTurn-v0 --headless \
  #       --load_run go2w_velocity_slope_turn_a/2026-07-27_19-49-02 \
  #       --checkpoint model_12847.pt

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
from rexmi_rl.tasks.locomotion.velocity.mdp.commands import (
    make_relative_heading_command,
    make_hold_yaw_settle_command,
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
        25° → −0.396 m
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

    ---------------------------------------------------------------------------
    v13 — LOW-ω GATE FIX (SB-v2 / SC-v1, 2026-07-27)
    ---------------------------------------------------------------------------
    Turn-B inherited gates used omega_threshold / min_cmd = 0.1.  Once slope
    training slowed ω to ±0.08 (SB-v2) and ±0.06 (SC), those gates NEVER fire:

      wheel_lock, foot_alternation, yaw_stagnation → Episode_Reward = 0.000
      (confirmed in SB-v2 logs: foot_alt=0, wheel_lock=0)

    Lower all turn-shaping gates to 0.04 so they stay active under slow clips
    while still ignoring near-zero residual commands.  pivot_step_coord and
    foot_air_time also get 0.04 (was hardcoded / 0.05).
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

    # v13: keep turn-shaping rewards alive under slow ω (±0.06 / ±0.08)
    _LOW_OMEGA_GATE = 0.04
    if hasattr(cfg.rewards, "wheel_lock"):
        cfg.rewards.wheel_lock.params["omega_threshold"] = _LOW_OMEGA_GATE
    if hasattr(cfg.rewards, "foot_alternation"):
        cfg.rewards.foot_alternation.params["omega_threshold"] = _LOW_OMEGA_GATE
    if hasattr(cfg.rewards, "yaw_stagnation"):
        cfg.rewards.yaw_stagnation.params["min_cmd"] = _LOW_OMEGA_GATE
    if hasattr(cfg.rewards, "heading_progress_turn"):
        cfg.rewards.heading_progress_turn.params["min_cmd"] = _LOW_OMEGA_GATE
    if hasattr(cfg.rewards, "pivot_step_coord"):
        cfg.rewards.pivot_step_coord.params["omega_threshold"] = _LOW_OMEGA_GATE
    if hasattr(cfg.rewards, "foot_air_time"):
        cfg.rewards.foot_air_time.params["omega_threshold"] = _LOW_OMEGA_GATE

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


def _apply_slope_command_overrides_symmetric(cfg, ang_vel_clip: float = 0.12) -> None:
    """
    Both turn directions, sampled ω (heading_command=False).

    ang_vel_clip:
      0.12 — SA-v12b / default (matched D1 demo rate on 10°)
      0.08 — SB-v2 slower pivot on 20° so the policy can learn roll
             balance without as much lateral disturbance per step

    Use only AFTER v12a has produced a checkpoint that survives D1−
    (for the first jump to symmetric), or warm-start from a prior
    symmetric ckpt when only changing the clip.
    """
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.rel_heading_envs = 0.0
    cfg.commands.base_velocity.resampling_time_range = (6.0, 6.0)
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.05, 0.05)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    w = float(ang_vel_clip)
    cfg.commands.base_velocity.ranges.ang_vel_z = (-w, w)


def _apply_slope_command_overrides_mixed_hold_turn(
    cfg,
    ang_vel_clip: float = 0.08,
    rel_standing_envs: float = 0.35,
    lin_vel_x_when_turning: float = 0.05,
) -> None:
    """
    S25 mixed hold + turn (2026-07-28).

    Visual on SC 30° (model_13594): robot slips immediately — never plants.
    Pure hold-only risks erasing turn skill from SB-v2 (model_13345).
    So bridge at 25° with BOTH:
      - fraction rel_standing_envs → full stop (ω=0, vx=0) for stance/slip
      - remaining envs → slow symmetric ω ±ang_vel_clip
      - lin_vel_x_when_turning: 0.05 (v1) or 0.0 (v2 — no creep while pivoting)

    heading_command=False (sampled ω), same interface as D1/SB-v2.
    """
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.rel_heading_envs = 0.0
    cfg.commands.base_velocity.rel_standing_envs = float(rel_standing_envs)
    cfg.commands.base_velocity.resampling_time_range = (6.0, 6.0)
    # Turn envs: mild creep (standing envs zero all cmds via rel_standing)
    cfg.commands.base_velocity.ranges.lin_vel_x = (
        float(lin_vel_x_when_turning),
        float(lin_vel_x_when_turning),
    )
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    w = float(ang_vel_clip)
    cfg.commands.base_velocity.ranges.ang_vel_z = (-w, w)




def _apply_slope_command_overrides_lang_a_antifreeze(
    cfg,
    *,
    ang_vel_clip: float = 0.04,
    rel_standing_envs: float = 0.30,
    lin_vel_x: float = 0.0,
    resampling_s: float = 2.0,
) -> None:
    """
    S25 Phase A2 — Language A, SLOW + NON-CONTINUOUS (2026-08-02).

    Visual Phase A (ω±0.08, standing 8%, 5s holds): hold OK; spin → fall (roll).
    User: mix A1 (slower) + A2 (not continuous). Speed later.

    NOT v2 freeze (standing 35% + ω±0.05 + long windows + high is_alive).
    NOT Phase A continuous aggressive spin.

    Recipe A2:
      - heading_command=False (Language A / 13345 dialect)
      - ω ±0.04 when turning (slow)
      - resample every 2.0 s → short pulses, not 5–6 s continuous pivot
      - ~30% standing windows (ω=0 plant / rebalance) interleaved with turns
      - vx=0
      - is_alive moderate; heading_progress strong on turn windows
      - yaw_stag / gates tuned for SHORT slow pulses (min_yaw ~3–4° achievable)
      - Language B heading_error weights forced 0 if present

    Warm-start: model_13345.pt only.
    """
    # --- Language A commands: short slow pulses + plant windows ---
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.rel_heading_envs = 0.0
    cfg.commands.base_velocity.rel_standing_envs = float(rel_standing_envs)
    cfg.commands.base_velocity.resampling_time_range = (float(resampling_s), float(resampling_s))
    cfg.commands.base_velocity.ranges.lin_vel_x = (float(lin_vel_x), float(lin_vel_x))
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    w = float(ang_vel_clip)
    cfg.commands.base_velocity.ranges.ang_vel_z = (-w, w)

    # --- Kill Language B reward terms if a prior cfg added them ---
    for name in ("heading_error", "heading_error_reduction"):
        if hasattr(cfg.rewards, name):
            getattr(cfg.rewards, name).weight = 0.0

    # --- Rewards: pay for real yaw in turn pulses; don't make freeze free ---
    if hasattr(cfg.rewards, "is_alive"):
        cfg.rewards.is_alive.weight = 0.30
    if hasattr(cfg.rewards, "heading_progress_turn"):
        cfg.rewards.heading_progress_turn.weight = 100.0
        # Must fire at ω=0.04
        cfg.rewards.heading_progress_turn.params["min_cmd"] = 0.03
    if hasattr(cfg.rewards, "track_ang_vel_z_exp"):
        cfg.rewards.track_ang_vel_z_exp.weight = 2.0
    if hasattr(cfg.rewards, "yaw_stagnation"):
        # At ω=0.04, ~2s pulse ⇒ max ~4.6°; require modest real yaw, not 12°
        cfg.rewards.yaw_stagnation.weight = -2.5
        cfg.rewards.yaw_stagnation.params["min_cmd"] = 0.03
        cfg.rewards.yaw_stagnation.params["min_yaw_deg"] = 3.5
        cfg.rewards.yaw_stagnation.params["window_steps"] = 75
    if hasattr(cfg.rewards, "position_drift"):
        cfg.rewards.position_drift.params["drift_threshold"] = 0.28
        cfg.rewards.position_drift.weight = -1.2
    # Gates must be < ang_vel_clip so shaping fires during ±0.04 pulses
    _g = 0.03
    if hasattr(cfg.rewards, "wheel_lock"):
        cfg.rewards.wheel_lock.params["omega_threshold"] = _g
    if hasattr(cfg.rewards, "foot_alternation"):
        cfg.rewards.foot_alternation.params["omega_threshold"] = _g
    if hasattr(cfg.rewards, "foot_air_time"):
        cfg.rewards.foot_air_time.params["omega_threshold"] = _g
    if hasattr(cfg.rewards, "pivot_step_coord"):
        cfg.rewards.pivot_step_coord.params["omega_threshold"] = _g



def _apply_slope_command_overrides_micro_turn_rebalance(
    cfg,
    *,
    ang_vel_clip: float = 0.08,
    heading_delta_deg: tuple = (10.0, 25.0),
    settle_margin_s: float = 0.0,
    rel_standing_envs: float = 0.08,
    direction_flip_prob: float = 0.20,
    heading_control_stiffness: float = 1.0,
) -> None:
    """
    S25-v3b: ERROR-DRIVEN micro-turn / rebalance (2026-08-02).

    User design (preferred over open-loop bursts):
      - Command = relative heading GOAL (nav-like), not continuous ω.
      - Heading ERROR accumulates cost until stepping is worth it.
      - When error is small, penalty stops → natural rebalance (no fixed settle).
      - Stability (trunk, drift, falls) makes reckless continuous spin expensive.
      - Skill emerges: hold → small correction → plant → next correction.

    Command = RelativeHeadingVelocityCommand, settle_margin_s=0 so rebalance
    length is NOT hard-coded; P-control already drives ω→0 as error→0.

    Rewards:
      heading_error_l1          — pressure while |err| > deadzone
      heading_error_reduction   — pay for shrinking error (real yaw)
      heading_progress_turn     — keep (signed Δyaw)
      track_ang_vel_z_exp       — weak (lied in S25-v2 freeze)
      is_alive                  — moderate (not freeze optimum)
      position_drift / trunk    — station + stability
    """
    from isaaclab.managers import RewardTermCfg as RewTerm
    from rexmi_rl.tasks.locomotion.velocity.mdp.rewards import (
        heading_error_l1,
        heading_error_reduction,
    )

    cfg.commands.base_velocity = make_relative_heading_command(
        ang_vel_clip=float(ang_vel_clip),
        heading_delta_deg=heading_delta_deg,
        lin_vel_x=(0.0, 0.0),
        settle_margin_s=float(settle_margin_s),
        rel_standing_envs=float(rel_standing_envs),
        direction_flip_prob=float(direction_flip_prob),
        heading_control_stiffness=float(heading_control_stiffness),
        debug_vis=True,
    )

    # --- Error-driven turn pressure ---
    cfg.rewards.heading_error = RewTerm(
        func=heading_error_l1,
        weight=-2.5,
        params={"command_name": "base_velocity", "deadzone_rad": 0.06},
    )
    cfg.rewards.heading_error_reduction = RewTerm(
        func=heading_error_reduction,
        weight=8.0,
        params={"command_name": "base_velocity", "deadzone_rad": 0.06},
    )

    if hasattr(cfg.rewards, "heading_progress_turn"):
        cfg.rewards.heading_progress_turn.weight = 80.0
        # Gate on omega still useful during P-control saturation
        cfg.rewards.heading_progress_turn.params["min_cmd"] = 0.02
    if hasattr(cfg.rewards, "track_ang_vel_z_exp"):
        cfg.rewards.track_ang_vel_z_exp.weight = 1.0
    if hasattr(cfg.rewards, "is_alive"):
        cfg.rewards.is_alive.weight = 0.30
    if hasattr(cfg.rewards, "yaw_stagnation"):
        cfg.rewards.yaw_stagnation.params["min_cmd"] = 0.02
        cfg.rewards.yaw_stagnation.params["min_yaw_deg"] = 5.0
        cfg.rewards.yaw_stagnation.params["window_steps"] = 80
        cfg.rewards.yaw_stagnation.weight = -3.0
    if hasattr(cfg.rewards, "position_drift"):
        cfg.rewards.position_drift.params["drift_threshold"] = 0.28
        cfg.rewards.position_drift.weight = -1.5
    if hasattr(cfg.rewards, "trunk_stability"):
        # Keep plant quality while allowing step tilt
        cfg.rewards.trunk_stability.params["max_tilt_deg"] = 28.0



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

    Warm-start from balanced SA-v12b (model_12847.pt), NOT from negative-only v12a.

    SB-v2 (after model_13096 visual): ω = (-0.08, +0.08) symmetric sampled.
    Rotation on 20° was already good; roll/stabilization was the gap. Slower
    yaw reduces lateral disturbance so balance can be learned.

    Warm-start: go2w_velocity_slope_turn/2026-07-27_20-15-35/model_13096.pt
    (or SA-v12b model_12847.pt if restarting the 20° jump).

    Never use _apply_slope_command_overrides (negative-only v12a) on SB.
    trunk_stability 25° on SB: body already tilts ~slope angle on 20°.
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
        # SB-v2: slower symmetric ω (±0.08). Visual on model_13096: turns OK,
        # balance/roll hard. One knob only — ease lateral disturbance.
        # Never the v12a negative-only override on SB.
        _apply_slope_command_overrides_symmetric(self, ang_vel_clip=0.08)

        # SB: allow natural slope lean without fighting trunk_stability
        if hasattr(self.rewards, "trunk_stability"):
            self.rewards.trunk_stability.params["max_tilt_deg"] = 25.0

        if hasattr(self.events, "push_robot"):
            self.events.push_robot = None


@configclass
class Go2wSlopeTurnEnvCfg_PLAY(Go2wSlopeTurnEnvCfg):
    """Phase SB play: match SB-v2 train — sampled ω ±0.08."""

    def __post_init__(self):
        super().__post_init__()
        _apply_slope_play_common(self)
        # Match train clip (parent already set train commands; re-assert play schedule)
        cfg = self
        cfg.commands.base_velocity.heading_command = False
        cfg.commands.base_velocity.rel_heading_envs = 0.0
        cfg.commands.base_velocity.resampling_time_range = (6.0, 6.0)
        cfg.commands.base_velocity.ranges.ang_vel_z = (-0.08, 0.08)
        cfg.commands.base_velocity.ranges.lin_vel_x = (0.05, 0.05)
        cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)


# ===========================================================================
# PHASE SC: 30° rocky pyramid slope
# ===========================================================================

@configclass
class Go2wSlopeTurnCEnvCfg(Go2wTurnBEnvCfg):
    """
    Phase SC: Pivot turn on a 30° rocky pyramid slope.

    Warm-start from SB-v2 model_13345.pt (20°, ω±0.08, both directions OK for
    finite nav-style reorients). Slope 30° — crater-wall class angles.

    SC-v1 lesson from 10→20: jump slope AND keep the same ω → roll deaths.
    So SC uses SLOWER symmetric ω (±0.06) as the only new disturbance besides
    the slope angle. trunk_stability 30° matches natural body tilt on a 30° face.

    v13 gate fix (in _apply_slope_reward_overrides): turn-shaping omega gates
    lowered 0.1 → 0.04 so wheel_lock / foot_alt / yaw_stag still fire at ±0.06.

    Never use negative-only v12a command override on SC.
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
        # SC-v1: slower than SB-v2 (±0.08) — ease lateral load on steeper face
        _apply_slope_command_overrides_symmetric(self, ang_vel_clip=0.06)

        if hasattr(self.rewards, "trunk_stability"):
            self.rewards.trunk_stability.params["max_tilt_deg"] = 30.0

        if hasattr(self.events, "push_robot"):
            self.events.push_robot = None


@configclass
class Go2wSlopeTurnCEnvCfg_PLAY(Go2wSlopeTurnCEnvCfg):
    """Phase SC play: match SC-v1 train — sampled ω ±0.06."""

    def __post_init__(self):
        super().__post_init__()
        _apply_slope_play_common(self)
        cfg = self
        cfg.commands.base_velocity.heading_command = False
        cfg.commands.base_velocity.rel_heading_envs = 0.0
        cfg.commands.base_velocity.resampling_time_range = (6.0, 6.0)
        cfg.commands.base_velocity.ranges.ang_vel_z = (-0.06, 0.06)
        cfg.commands.base_velocity.ranges.lin_vel_x = (0.05, 0.05)
        cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)


# ===========================================================================
# PHASE S25: 25° Language A — Phase A2 SLOW + PULSE (after Phase A roll wall)
# ===========================================================================
# Phase A visual: hold OK; continuous ω±0.08 spin → fall (bad_roll ~45%)
# User: A1 slower + A2 non-continuous. Optimize speed later.
# Warm-start: go2w_velocity_slope_turn/2026-07-27_20-54-25/model_13345.pt
# ===========================================================================

@configclass
class Go2wSlopeTurnS25EnvCfg(Go2wTurnBEnvCfg):
    """
    S25 Phase A2: 25° Language A — slow ω pulses + plant windows.

    Warm-start ONLY from SB-v2 model_13345.pt.

    Commands:
      heading_command=False
      ang_vel_z ±0.04 (slow)
      resample 2.0 s (short turn pulses)
      rel_standing_envs=0.30 (plant / rebalance windows)
      lin_vel_x=0

    Not v2 freeze: short pulses + strong heading_progress when ω active.
    Not Phase A: no continuous ±0.08 for 5 s.

    Pass: small visible yaw bursts without instant roll; hold still OK.
    """

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = _make_slope_terrain(25.0)
        self.scene.env_spacing = 8.0
        self.curriculum.terrain_levels = None
        self.sim.gravity = (0.0, 0.0, -9.81)

        _apply_slope_spawn(self, 25.0)
        _apply_slope_friction_event(self)
        _apply_slope_reward_overrides(self)
        # A2 defaults inside helper: ω±0.04, standing 0.30, resample 2s
        _apply_slope_command_overrides_lang_a_antifreeze(self)

        if hasattr(self.rewards, "trunk_stability"):
            self.rewards.trunk_stability.params["max_tilt_deg"] = 28.0

        if hasattr(self.events, "push_robot"):
            self.events.push_robot = None


@configclass
class Go2wSlopeTurnS25EnvCfg_PLAY(Go2wSlopeTurnS25EnvCfg):
    """S25 Phase A2 play: same slow pulse + plant schedule as train."""

    def __post_init__(self):
        super().__post_init__()
        _apply_slope_play_common(self)
        _apply_slope_command_overrides_lang_a_antifreeze(self)


@configclass
class Go2wSlopeTurnS25EnvCfg_PLAY_HOLD(Go2wSlopeTurnS25EnvCfg):
    """S25 play: pure hold ω=0. PLAY ONLY."""

    def __post_init__(self):
        super().__post_init__()
        _apply_slope_play_common(self)
        cfg = self
        cfg.commands.base_velocity.heading_command = False
        cfg.commands.base_velocity.rel_heading_envs = 0.0
        cfg.commands.base_velocity.rel_standing_envs = 0.0
        cfg.commands.base_velocity.resampling_time_range = (20.0, 20.0)
        cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)


@configclass
class Go2wSlopeTurnS25EnvCfg_PLAY_TURN(Go2wSlopeTurnS25EnvCfg):
    """
    S25 Phase A2 play: pulse duty-cycle (not continuous spin).

    standing 0.30 + ω±0.04 + 2s resample so visual shows turn → plant → turn.
    """

    def __post_init__(self):
        super().__post_init__()
        _apply_slope_play_common(self)
        _apply_slope_command_overrides_lang_a_antifreeze(
            self,
            ang_vel_clip=0.04,
            rel_standing_envs=0.30,
            lin_vel_x=0.0,
            resampling_s=2.0,
        )

# ===========================================================================
# PULSE FSM @ 20° — HOLD → YAW → SETTLE (single policy, Language A)
# ===========================================================================
# After S25 failures: continuous spin rolls; freeze kills turn; hold works.
# Teach structured micro-reorient on 20° (where 13345 already turns), then
# climb slope. Nav will emit the same pulse pattern later.
# Warm-start: go2w_velocity_slope_turn/2026-07-27_20-54-25/model_13345.pt
# ===========================================================================

def _apply_pulse_fsm_rewards(cfg, omega_mag: float = 0.08, yaw_s: float = 2.0) -> None:
    """
    Pulse20-v2 rewards (Language A) — after v1 visual: wiggle only, no heading.

    v1 failure: high track_ang + is_alive + short weak pulses → survival without Δψ.
    v2: real heading_progress must dominate; track_ang weak; is_alive lower;
        yaw_stagnation requires visible degrees over a yaw-scale window.
    """
    # Disable Language B leftovers
    for name in ("heading_error", "heading_error_reduction"):
        if hasattr(cfg.rewards, name):
            getattr(cfg.rewards, name).weight = 0.0

    # Survival must NOT beat yaw
    if hasattr(cfg.rewards, "is_alive"):
        cfg.rewards.is_alive.weight = 0.22

    # Real net yaw — primary turn pay
    if hasattr(cfg.rewards, "heading_progress_turn"):
        cfg.rewards.heading_progress_turn.weight = 180.0
        cfg.rewards.heading_progress_turn.params["min_cmd"] = max(0.04, 0.5 * omega_mag)
    # track_ang lied in v1 (high while no heading change)
    if hasattr(cfg.rewards, "track_ang_vel_z_exp"):
        cfg.rewards.track_ang_vel_z_exp.weight = 1.0

    # Require real degrees during yaw-scale windows (ω=0.08 * 2s ≈ 9°)
    if hasattr(cfg.rewards, "yaw_stagnation"):
        cfg.rewards.yaw_stagnation.weight = -4.0
        cfg.rewards.yaw_stagnation.params["min_cmd"] = max(0.04, 0.5 * omega_mag)
        cfg.rewards.yaw_stagnation.params["min_yaw_deg"] = 8.0
        # ~ yaw_s / dt ; dt~0.02 → 2.0/0.02=100 steps; use 80
        cfg.rewards.yaw_stagnation.params["window_steps"] = max(60, int(yaw_s / 0.02 * 0.8))

    if hasattr(cfg.rewards, "position_drift"):
        cfg.rewards.position_drift.params["drift_threshold"] = 0.30
        cfg.rewards.position_drift.weight = -1.2

    # Gates below omega_mag so shaping fires during yaw pulses
    _g = max(0.03, 0.4 * omega_mag)
    for name, key in (
        ("wheel_lock", "omega_threshold"),
        ("foot_alternation", "omega_threshold"),
        ("foot_air_time", "omega_threshold"),
        ("pivot_step_coord", "omega_threshold"),
    ):
        if hasattr(cfg.rewards, name):
            getattr(cfg.rewards, name).params[key] = _g



def _apply_pulse_fsm_rewards_capped(
    cfg,
    omega_mag: float = 0.05,
    yaw_s: float = 0.9,
    max_rate_scale: float = 2.5,
) -> None:
    """
    Pulse25 microstep rewards — Pulse20 family + rate-capped heading_progress.

    14092 (uncapped plant-heavy): one-shot twist, forgot microstep.
    Cap per-step hp at |ω|*dt*max_rate_scale so many small steps beat one wrench.
    Same weights otherwise (hp 180, is_alive 0.22). Pulse20 stays uncapped.
    """
    _apply_pulse_fsm_rewards(cfg, omega_mag=omega_mag, yaw_s=yaw_s)
    if hasattr(cfg.rewards, "heading_progress_turn"):
        cfg.rewards.heading_progress_turn.params["max_rate_scale"] = float(max_rate_scale)
        cfg.rewards.heading_progress_turn.params["step_dt"] = 0.02
        cfg.rewards.heading_progress_turn.params["min_cmd"] = max(0.02, 0.4 * omega_mag)



@configclass
class Go2wSlopeTurnPulse20EnvCfg(Go2wTurnBEnvCfg):
    """
    Pulse20-v2 FSM on 20° — HOLD → YAW → SETTLE (after v1 wiggle-only fail).

    Warm-start: SB-v2 model_13345.pt ONLY (not Pulse20-v1 13594).

    v2 pulse box (tighter yaw pay + longer/stronger pulse):
      hold 1.0s | yaw 2.0s @ ±0.08 | settle 1.5s
      heading_progress 180, track_ang 1.0, is_alive 0.22, yaw_stag min 8°
    """

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = _make_slope_terrain(20.0)
        self.scene.env_spacing = 8.0
        self.curriculum.terrain_levels = None
        self.sim.gravity = (0.0, 0.0, -9.81)

        _apply_slope_spawn(self, 20.0)
        _apply_slope_friction_event(self)
        _apply_slope_reward_overrides(self)

        omega = 0.08
        yaw_s = 2.0
        self.commands.base_velocity = make_hold_yaw_settle_command(
            omega_mag=omega,
            hold_s=1.0,
            yaw_s=yaw_s,
            settle_s=1.5,
            direction_flip_prob=0.2,
            debug_vis=True,
        )
        _apply_pulse_fsm_rewards(self, omega_mag=omega, yaw_s=yaw_s)

        if hasattr(self.rewards, "trunk_stability"):
            self.rewards.trunk_stability.params["max_tilt_deg"] = 25.0
        if hasattr(self.events, "push_robot"):
            self.events.push_robot = None


@configclass
class Go2wSlopeTurnPulse20EnvCfg_PLAY(Go2wSlopeTurnPulse20EnvCfg):
    """
    Pulse20 play: full HOLD→YAW→SETTLE.

    PLAY-ONLY soft envelope (no retrain): ω±0.04, yaw 1.0s, settle 1.5s.
    Train v2 stays 2.0s @ ±0.08; visual showed real turn but tips when hot.
    """

    def __post_init__(self):
        super().__post_init__()
        _apply_slope_play_common(self)
        omega = 0.04
        yaw_s = 1.0
        self.commands.base_velocity = make_hold_yaw_settle_command(
            omega_mag=omega,
            hold_s=1.0,
            yaw_s=yaw_s,
            settle_s=1.5,
            direction_flip_prob=0.2,
            debug_vis=True,
        )
        _apply_pulse_fsm_rewards(self, omega_mag=omega, yaw_s=yaw_s)


@configclass
class Go2wSlopeTurnPulse20EnvCfg_PLAY_HOLD(Go2wSlopeTurnPulse20EnvCfg):
    """Pulse20 play diagnostic: permanent HOLD (ω=0)."""

    def __post_init__(self):
        super().__post_init__()
        _apply_slope_play_common(self)
        # Degenerate FSM: very long hold, zero yaw magnitude
        self.commands.base_velocity = make_hold_yaw_settle_command(
            omega_mag=0.0,
            hold_s=30.0,
            yaw_s=0.5,
            settle_s=0.5,
            direction_flip_prob=0.0,
            debug_vis=True,
        )


@configclass
class Go2wSlopeTurnPulse20EnvCfg_PLAY_YAW(Go2wSlopeTurnPulse20EnvCfg):
    """
    Pulse20 play YAW emphasis — PLAY-ONLY soft envelope.

    hold 0.5s | yaw 1.0s @ ±0.04 | settle 1.5s
    (half rate + half duration vs train v2; same policy ckpt)
    """

    def __post_init__(self):
        super().__post_init__()
        _apply_slope_play_common(self)
        omega = 0.04
        yaw_s = 1.0
        self.commands.base_velocity = make_hold_yaw_settle_command(
            omega_mag=omega,
            hold_s=0.5,
            yaw_s=yaw_s,
            settle_s=1.5,
            direction_flip_prob=0.15,
            debug_vis=True,
        )
        _apply_pulse_fsm_rewards(self, omega_mag=omega, yaw_s=yaw_s)


# ===========================================================================
# PULSE FSM @ 25° — plant-heavy pulse (2026-08-07)
# ===========================================================================
# After minimal 25° finetune (13843): plant OK, hot yaw tips / continuous spin feel.
# Train matches successful play language: long hold/settle, short moderate yaw.
# Cmd: hold 2.5 | yaw 0.9 @ ±0.05 | settle 3.0
# Rewards: same _apply_pulse_fsm_rewards (no new terms).
# Warm-start: pulse25/2026-08-07_19-50-19/model_13843.pt
# ===========================================================================


@configclass
class Go2wSlopeTurnPulse25EnvCfg(Go2wTurnBEnvCfg):
    """
    Pulse25 plant-heavy + rate-capped hp @ 25° (microstep, not one-shot).

    Warm-start: pulse25/2026-08-07_19-50-19/model_13843.pt
    (minimal 25° finetune from Pulse20 13594; turns but tippy on hot yaw)

    Schedule (play-proven language):
      hold 2.5s | yaw 0.9s @ ±0.05 | settle 3.0s
    Pulse20 rewards + heading_progress max_rate_scale=2.5 (anti one-shot).
    Warm-start model_13843.pt ONLY (not 14092 one-shot).
    """

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = _make_slope_terrain(25.0)
        self.scene.env_spacing = 8.0
        self.curriculum.terrain_levels = None
        self.sim.gravity = (0.0, 0.0, -9.81)

        _apply_slope_spawn(self, 25.0)
        _apply_slope_friction_event(self)
        _apply_slope_reward_overrides(self)

        # Plant-heavy pulse (play-aligned)
        omega = 0.05
        yaw_s = 0.9
        self.commands.base_velocity = make_hold_yaw_settle_command(
            omega_mag=omega,
            hold_s=2.5,
            yaw_s=yaw_s,
            settle_s=3.0,
            direction_flip_prob=0.1,
            debug_vis=True,
        )
        _apply_pulse_fsm_rewards_capped(self, omega_mag=omega, yaw_s=yaw_s, max_rate_scale=2.5)

        if hasattr(self.rewards, "trunk_stability"):
            # slightly above slope angle (Pulse20 used 25 on 20° terrain)
            self.rewards.trunk_stability.params["max_tilt_deg"] = 30.0
        if hasattr(self.events, "push_robot"):
            self.events.push_robot = None


@configclass
class Go2wSlopeTurnPulse25EnvCfg_PLAY(Go2wSlopeTurnPulse25EnvCfg):
    """
    Pulse25 play: plant-heavy HOLD→YAW→SETTLE (no retrain).

    User visual: stable plant; turn too fast / continuous spin → fall.
    Schedule: long hold, short slow yaw, long settle — stack small heading
    changes with ample stability between pulses.
      hold 2.5s | yaw 0.8s @ ±0.04 | settle 3.0s
    """

    def __post_init__(self):
        super().__post_init__()
        _apply_slope_play_common(self)
        omega = 0.04
        yaw_s = 0.8
        self.commands.base_velocity = make_hold_yaw_settle_command(
            omega_mag=omega,
            hold_s=2.5,
            yaw_s=yaw_s,
            settle_s=3.0,
            direction_flip_prob=0.1,
            debug_vis=True,
        )
        _apply_pulse_fsm_rewards_capped(self, omega_mag=omega, yaw_s=yaw_s, max_rate_scale=2.5)


class Go2wSlopeTurnPulse25EnvCfg_PLAY_HOLD(Go2wSlopeTurnPulse25EnvCfg):
    """Pulse25 play: pure hold ω=0."""

    def __post_init__(self):
        super().__post_init__()
        _apply_slope_play_common(self)
        self.commands.base_velocity = make_hold_yaw_settle_command(
            omega_mag=0.0,
            hold_s=30.0,
            yaw_s=0.5,
            settle_s=0.5,
            direction_flip_prob=0.0,
            debug_vis=True,
        )


@configclass
class Go2wSlopeTurnPulse25EnvCfg_PLAY_YAW(Go2wSlopeTurnPulse25EnvCfg):
    """
    Pulse25 play yaw emphasis — still plant-heavy.
      hold 1.5s | yaw 0.8s @ ±0.04 | settle 3.0s
    """

    def __post_init__(self):
        super().__post_init__()
        _apply_slope_play_common(self)
        omega = 0.04
        yaw_s = 0.8
        self.commands.base_velocity = make_hold_yaw_settle_command(
            omega_mag=omega,
            hold_s=1.5,
            yaw_s=yaw_s,
            settle_s=3.0,
            direction_flip_prob=0.1,
            debug_vis=True,
        )
        _apply_pulse_fsm_rewards_capped(self, omega_mag=omega, yaw_s=yaw_s, max_rate_scale=2.5)
