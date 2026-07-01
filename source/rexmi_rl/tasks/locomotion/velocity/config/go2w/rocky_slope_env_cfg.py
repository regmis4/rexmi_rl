# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Go2W rocky-slope training environment — Phase 8i.

Phase history
-------------
  Phase 8b: model_7497.pt — rocky slopes uphill baseline established.
  Phase 8c: model_12495.pt — 50% downhill, forward-only cmds.  Downhill solved (0.72–0.90).
  Phase 8d: model_13994.pt — 100% uphill. terrain_levels stalled at 0.19.
  Phase 8e: model_15993.pt — slope_max=25°, friction rand (0.5–1.5), lean reward (+0.5).
             terrain_levels advanced 0.19 → 0.64 (≈21°).
  Phase 8f: model_17992.pt (in rocky_slope run) — ZERO boulders, slope_max=35°.
             terrain_levels ended at 0.315 (= 21.3°). Same ceiling as 8e.
  Phase 8g: CRASHED — lean weight raised to 3.0. Thigh-salute exploit → value_loss→∞.
  Phase 8h: model_17992.pt (in rocky_slope run 2026-06-29_19-58-05) — physics fix:
             friction 0.5→0.8, boulders restored (max=12, h≤15cm), lean reward removed.
             terrain_levels ended at 0.385 (= 22.7° on 35° max terrain).
             SAME 22° ceiling as all prior phases.
             Root cause confirmed: the model_15993 lineage is DESCENT-DOMINANT.
             Phase 8c trained 50% downhill tiles → robot learned to BRAKE against gravity
             rather than push against it.  Phases 8d/e/f/g/h all inherited those descent
             weights and converged to the same ~22° equilibrium regardless of terrain or
             reward changes.
  Phase 8i (this file): RESTART from model_8996.pt — the rough-terrain generalist.
             CORRECTION from original Phase 8i design:
             model_5998.pt is NOT an uphill specialist.  HfPyramidSlopedTerrainCfg
             has a 2m flat CENTER PLATFORM; robots spawn on the platform and command
             FORWARD → they go DOWN the slope on all sides.  terrain_levels=4.77 means
             model_5998 descends 33°+ slopes — not climbs them.  The "going sideways"
             nav problem: on a pyramid with slopes on all sides, the robot finds the
             lowest-gradient lateral path rather than fighting straight uphill.
             model_5998 is a DESCENT specialist, same as model_12495.  Using it as a
             Phase 8i starting point would reproduce the same descent-dominant gait.

             Correct starting point: model_8996.pt (rough terrain generalist).
             model_8996 was trained on MIXED terrain (stairs up/down, slopes, boxes,
             rough) with no pure-descent exposure.  It has the most neutral gait of
             any checkpoint — not dominated by uphill OR downhill optimization.
             Starting rocky-slope uphill from model_8996 directly skips the entire
             steep-slope descent detour (model_8996 → model_5998 → Phase 8b–8h lineage)
             that contaminated all subsequent policies with descent-dominant weights.

  Gait basin analysis (corrected):
  ─────────────────────────────────
    model_8996.pt   — rough terrain, mixed up/down, NEUTRAL gait     ← best start
         ↓ Phase 8 steep-slope (HfPyramidSlopedTerrainCfg = DOWN):
    model_5998.pt   — pyramid slope, robots go DOWN from platform    ← ALSO DESCENT
         ↓ Phase 8b: rocky slope from model_5998 → model_7497
         ↓ Phase 8c: 50% downhill → model_12495 → DESCENT DOMINANT
         ↓ 8d/e/f/g/h: all inherit descent gait → 22° ceiling        ← confirmed
    model_8996 → rocky slope directly (Phase 8i) skips all contamination.
    This is the cleanest available starting point without training from scratch.

  NOTE: 22° ceiling may still apply — it may be a true physical limit for this
  robot and reward function, not just a gait basin issue.  model_8996 will reveal
  whether the neutral gait can break through 22°.  If it also stalls at 22°, then
  the ceiling is physical (friction/traction/robot geometry) and the demo should
  proceed with model_15993.pt (best descent, functional 22°+ uphill).
  Training command (Phase 8i — fresh start from model_8996.pt, 3000 iters)
  -------------------------------------------------------------------------
    conda activate env_isaacsim
    python scripts/train.py --task RexmiRl-Go2w-Velocity-RockySlope-v0 --headless \\
        --load_run go2w_velocity_rough/2026-06-14_20-03-41 \\
        --checkpoint model_8996.pt --max_iterations 3000

Command distribution (Phase 8i, unchanged)
------------------------------------------
  vx  ∈ (0.2, 0.5) m/s    — always forward
  vy  = 0.0
  ωz  ∈ (-0.5, 0.5) rad/s

Terrain (Phase 8i — same as Phase 8h: boulders max=12, friction 0.8–1.5)
------------------------------------------------------------------------
  100% uphill pyramid slope, slope_max_deg=35°.
  Boulder count 0→12 (max), height 5–15 cm.
  Roughness 1–6 cm provides geometric interlocking throughout all rows.

  row 0 → 15°,  0–1 boulders,  h_max=5cm,  roughness=1cm
  row 4 → 23°,  4–5 boulders,  h_max=10cm, roughness=3cm
  row 7 → 29°,  8–9 boulders,  h_max=13cm, roughness=5cm
  row 9 → 35°, 10–12 boulders, h_max=15cm, roughness=6cm

Active rewards (Phase 8i — identical to Phase 8h):
  • uphill_lean REMOVED (exploitable via thigh-salute at any useful weight)
  • friction μ_min = 0.8 (all episodes physically achievable at 35°)
  All other inherited rewards kept.

Phase 8i TensorBoard signals to watch:
  terrain_levels:   GATE at 500 iters: expect >0.5 (model_8996 → rocky slope is an
                    untested combination; model_8996 handled rough slopes to 23° but
                    never pure 35° rocky uphill — expect some reconvergence overhead)
                    If still < 0.3 at iter+500: physics limit is real, not gait issue
  stagnation:       will fire against boulders — target -0.05 to -0.08 (manageable)
                    if > -0.12 consistently → boulders wedging, needs more stagnation weight
  flat_orientation: will go more negative (body tilting into slope) — expected
  thigh_salute:     should stay near 0 (no lean reward bait)
  track_lin_vel:    model_8996 forward velocity range was ±0.5 m/s (same scale as rocky
                    slope env) — no mis-match expected, tracking should be good from iter 0

Phase 8j (planned, after Phase 8i reveals whether 22° is gait basin or physical limit):
  If terrain_levels > 1.0: gait basin was the problem, model_8996 can escape it.
    → Extend 2000 more iters, add 25% downhill tiles to begin rebuilding descent.
  If terrain_levels still ≈ 0.38 after 3000 iters: 22° is a physical limit.
    → Accept model_15993.pt as the production demo policy (excellent descent, 22°+ uphill).
"""

import math

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EvtTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass

from rexmi_rl.tasks.locomotion.velocity.config.go2w.steep_slope_env_cfg import (
    Go2wSteepSlopeEnvCfg,
)
from rexmi_rl.tasks.locomotion.velocity.config.go2w.crater_terrain import (
    RockyPyramidSlopeCfg,
    RockyPyramidSlopeDownCfg,
)


@configclass
class Go2wRockySlopeEnvCfg(Go2wSteepSlopeEnvCfg):
    """
    Go2W velocity-tracking environment — rocky slopes (15°–35°) with boulders.

    Phase 8i PRODUCTION: 100% uphill tiles; full slope range (15°→35°); modest boulders
    (max=12, h≤15cm) for geometric interlocking; friction floor 0.8–1.5;
    lean reward removed (exploitable). Started from model_8996.pt (neutral gait).
    Final checkpoint: model_13994.pt (2026-06-30_09-31-48) — PRODUCTION POLICY.
    Eval: up_25°=0.60, up_30°=0.57, up_35°=0.57 | down_30°=0.98, down_35°=0.91.

    Inherited settings of note:
      • Height scanner (160-dim) — same obs space as model_5998/model_7497, loads cleanly
      • flat_orientation_l2 = -0.1  — slopes require sustained body tilt
      • bad_orientation = 1.4 rad (80°) — headroom for boulder-induced tilts
      • hip_crossing_penalty, thigh_salute, calf_symmetry, hip_symmetry — all kept
      • stagnation penalty weight overridden to -2.5 (was -1.5 in parent)
      • climb_progress — always in obstacle_weight mode on slopes (>> 10 cm threshold)
      • terrain curriculum (terrain_levels_vel) — row 0→9 auto-advancement
    """

    def __post_init__(self):
        # ---------------------------------------------------------------
        # Apply all steep-slope overrides (orientation, anti-exploit rewards,
        # pure steep-slope terrain) then replace terrain below.
        # ---------------------------------------------------------------
        super().__post_init__()

        # ==============================================================
        # 1. TERRAIN — 100% uphill rocky slope, boulders restored
        # ==============================================================
        # Rocky uphill: robot spawns at centre LOW point, commands forward → ascends
        #
        # Path B physics rationale:
        #   Smooth slope (Phase 8f): traction = μ × N_perpendicular only.
        #   Rocky/rough slope (Phase 8h): traction = μ × N_perp + geometric_interlock.
        #   The surface roughness (1–6 cm bumps) provides wheel bite independent of μ.
        #   Low boulder count (max=12 vs Phase 8e's 25) keeps obstacles as traction
        #   helpers rather than curriculum blockers.
        #   Boulder height capped at 15 cm (was 20 cm in Phase 8e) — 15 cm = 3× wheel
        #   radius, enough to add interlocking without causing stuck-at-base-of-boulder.
        #
        # Curriculum interpretation (num_rows=10) — Phase 8h:
        #   row 0 → 15°,  boulder_max=1,  h_max=5cm,  roughness=1cm
        #   row 4 → 23°,  boulder_max=5,  h_max=10cm, roughness=3cm
        #   row 7 → 29°,  boulder_max=9,  h_max=13cm, roughness=5cm
        #   row 9 → 35°,  boulder_max=12, h_max=15cm, roughness=6cm
        rocky_slope_params = dict(
            slope_min_deg=15.0,
            slope_max_deg=35.0,
            boulder_count_min=0,
            boulder_count_max=12,
            boulder_height_min=0.05,
            boulder_height_max=0.15,
            boulder_radius_min=0.15,
            boulder_radius_max=0.40,
            roughness_min_m=0.010,
            roughness_max_m=0.060,
        )

        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            seed=0,
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=10,
            num_cols=20,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={
                # Phase 8h: 100% uphill, slope_max=35°, boulders restored (max=12).
                # Key change from Phase 8e (which stalled): boulder height capped at
                # 15 cm (was 20 cm) and count reduced (was max=25).  The Phase 8e
                # stall was caused by 20 cm boulders physically blocking wheel climbing
                # — the wheel radius is 5 cm so a 20 cm vertical face is 4× wheel
                # diameter, impassable by rolling alone.  At 15 cm (3× radius) the
                # robot can still leg-lift over with Phase 8c's stagnation escape.
                "rocky_slope_up": RockyPyramidSlopeCfg(
                    proportion=1.0,
                    **rocky_slope_params,
                ),
            },
        )

        # ==============================================================
        # 2. COMMANDS — forward-only, no lateral, halved yaw range
        # ==============================================================
        self.commands.base_velocity.ranges.lin_vel_x = (0.2, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)

        # ==============================================================
        # 3. STAGNATION — stronger escape signal against boulders
        # ==============================================================
        self.rewards.stagnation.weight = -2.5

        # ==============================================================
        # 4. FRICTION RANDOMISATION — Path C: floor raised 0.5 → 0.8
        # ==============================================================
        # Physical motivation:
        #   μ_min = 0.5 → maximum gripable slope (no slip) = arctan(0.5) = 26.6°
        #   At our target slope 35°: tan(35°) = 0.70 > 0.5 → GUARANTEED slip every
        #   episode that draws μ from the lower half of [0.5, 1.5].
        #
        #   μ_min = 0.8 → maximum gripable slope = arctan(0.8) = 38.7°
        #   All episodes are now physically achievable at 35° — slip is possible
        #   only at the very low end of [0.8, 1.0] due to dynamic friction lag.
        #
        #   μ_max = 1.5 unchanged — dry rubber on basalt rock (physically realistic
        #   upper bound for lunar crater rock surface, crater wall is bare anorthosite).
        #
        # Why not just set μ = 1.5 always?
        #   Sim-to-real requires the policy to work across friction values, not to
        #   be brittle to a single coefficient. Randomisation forces "weight-commit"
        #   strategy. The floor at 0.8 ensures physical achievability; the range
        #   [0.8, 1.5] still provides robustness training.
        self.events.randomize_robot_friction = EvtTerm(
            func=mdp.randomize_rigid_body_material,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (0.8, 1.5),
                "dynamic_friction_range": (0.8, 1.5),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 64,
            },
        )

        # NOTE: uphill_lean reward REMOVED (Phase 8g lesson).
        # The lean reward at weight=0.5 was too weak to reshape gait.
        # At weight=3.0 it was exploited: thigh-salute (raise front thighs)
        # generates nose-down pitch → +2.0 lean reward/step, far exceeding
        # the -1.0 thigh_salute penalty → value function diverged to infinity.
        # There is no safe weight between "too weak" and "exploitable" because
        # the thigh-salute exploit is always available as a lean generator.
        # The physics fix (friction floor + roughness) is the correct solution:
        # the terrain now physically rewards the robot for proper uphill gait
        # without needing a posture-specific reward term.


@configclass
class Go2wRockySlopeEnvCfg_PLAY(Go2wRockySlopeEnvCfg):
    """
    Play-mode variant for rocky-slope policy: fewer envs, no noise, no pushes.

    Spawns 50 robots on rocky uphill slope tiles to observe Phase 8h climbing gait.

        python scripts/play.py --task RexmiRl-Go2w-Velocity-RockySlope-Play-v0 \\
            --load_run go2w_velocity_rocky_slope/<run_date> --checkpoint model_<N>.pt
    """

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 8.0   # match 8 m × 8 m slope tile size
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
