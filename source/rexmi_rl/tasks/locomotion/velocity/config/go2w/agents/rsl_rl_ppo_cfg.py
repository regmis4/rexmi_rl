# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
PPO hyperparameter configuration for the Go2W velocity task using RSL-RL.

What is RSL-RL?
---------------
RSL-RL (Robotic Systems Lab RL) is a lightweight, fast PPO implementation
developed at ETH Zurich and used in Isaac Lab's official locomotion tutorials.
It is optimised for GPU-parallel environments (many envs running simultaneously).

PPO (Proximal Policy Optimisation) overview
--------------------------------------------
PPO is an on-policy actor-critic algorithm:
  1. Roll out the current policy for N steps across all envs → collect data
  2. Compute advantages (how much better/worse each action was than expected)
  3. Update the policy (actor) and value function (critic) for K epochs
  4. The "proximal" constraint clips the update ratio to prevent large steps

Network architecture choice
-----------------------------
Phase 1 (wheels only, no height scan):
  Observation size ≈ 3 + 3 + 3 + 3 + 16 + 16 + 4 = 48 dims
  Action size = 4 (one velocity target per wheel)
  A small network [128, 128, 128] is sufficient.
  Scaled up to [512, 256, 128] in Phase 4 when height scan adds ~160 dims.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class Go2wFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner configuration for the Go2W flat-terrain velocity task."""

    num_steps_per_env = 24
    max_iterations    = 1000
    save_interval     = 50
    experiment_name   = "go2w_velocity_flat"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


# ==============================================================================
# Fast flat-terrain PPO config — high-speed forward locomotion (up to 2 m/s)
# ==============================================================================

@configclass
class Go2wFastFlatPPORunnerCfg(Go2wFlatPPORunnerCfg):
    """
    PPO runner configuration for the Go2W fast flat-terrain task.

    Key differences from Go2wFlatPPORunnerCfg:
      1. More iterations (1500 vs 1000) — the policy must learn the full 0–2 m/s
         forward command range, which takes more exploration than 0–0.5 m/s.
      2. Separate experiment_name — logs go to logs/rsl_rl/go2w_velocity_fast_flat/
         so fast-flat and standard-flat runs don't overwrite each other.
    """

    max_iterations  = 1500
    save_interval   = 100
    experiment_name = "go2w_velocity_fast_flat"


# ==============================================================================
# Phase 4 — Rough terrain PPO config
# ==============================================================================

@configclass
class Go2wRoughPPORunnerCfg(Go2wFlatPPORunnerCfg):
    """
    PPO runner configuration for the Go2W rough-terrain velocity task.

    Key differences from the flat config:
      1. Larger network [512, 256, 128] — height scan adds ~160 dims to obs.
      2. More iterations (3000) — curriculum must advance through 10 difficulty levels.
      3. Separate experiment_name — logs/rsl_rl/go2w_velocity_rough/
      4. empirical_normalization=True — normalise height scan observations.

    Obs size breakdown (~220 dims):
      3   base linear velocity
      3   base angular velocity
      3   projected gravity vector
      3   velocity command (vx, vy, ωz)
      16  joint positions  (12 leg + 4 wheel)
      16  joint velocities (12 leg + 4 wheel)
      16  last actions
    +160  height scan (1.6 m × 1.0 m grid at 0.1 m resolution)
    -----
     220  total (approximate)
    """

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    max_iterations          = 3000
    save_interval           = 100
    experiment_name         = "go2w_velocity_rough"
    empirical_normalization = True


# ==============================================================================
# Phase 8 — Dedicated steep-slope PPO config (23°–45°)
# ==============================================================================

@configclass
class Go2wSteepSlopePPORunnerCfg(Go2wRoughPPORunnerCfg):
    """
    PPO runner configuration for the Go2W dedicated steep-slope policy.

    Inherits everything from Go2wRoughPPORunnerCfg (same network [512,256,128],
    same obs space ~220 dims with height scanner, same algorithm hyperparameters).

    Separate experiment_name keeps steep-slope logs away from rough-terrain runs
    so --resume cannot accidentally cross-contaminate the two experiments.
    """

    experiment_name = "go2w_velocity_steep_slope"
    max_iterations  = 3000
    save_interval   = 50


# ==============================================================================
# Phase 8b — Rocky slope PPO config (steep slopes WITH boulders, 15°–35°)
# ==============================================================================

@configclass
class Go2wRockySlopePPORunnerCfg(Go2wSteepSlopePPORunnerCfg):
    """
    PPO runner configuration for the Go2W rocky-slope policy (Phase 8b).

    Inherits everything from Go2wSteepSlopePPORunnerCfg:
      • Network: [512, 256, 128] — same architecture, loads cleanly from model_5998.pt
      • Obs space: ~220 dims with height scanner — identical to steep-slope policy
      • Algorithm: adaptive LR, empirical normalisation, same PPO hyperparameters
      • save_interval: 50 iters — fine-grained curriculum tracking

    Separate experiment_name → logs/rsl_rl/go2w_velocity_rocky_slope/

    Training command:
        python scripts/train.py --task RexmiRl-Go2w-Velocity-RockySlope-v0 --headless \\
            --load_run go2w_velocity_steep_slope/2026-06-20_15-37-32 \\
            --checkpoint model_5998.pt

    Phase 8d: 1500-iter diagnostic with 100% uphill tiles.
    Final production checkpoint: model_13994.pt (2026-06-30_09-31-48).
    """

    experiment_name = "go2w_velocity_rocky_slope"

    # Phase 8d: 1500-iter diagnostic, 100% uphill tiles.
    # Final checkpoint model_13994.pt used in all crater demo runs.
    max_iterations = 1500


# ==============================================================================
# Turn-in-place PPO config — pure yaw-rate tracking, blank-slate design
# ==============================================================================

@configclass
class Go2wTurnPPORunnerCfg(Go2wRoughPPORunnerCfg):
    """
    PPO runner configuration for the Go2W turn-in-place policy.

    INHERITS FROM: Go2wRoughPPORunnerCfg (NOT rocky slope)
    -------------------------------------------------------
    The rough policy (model_8996.pt) was trained with vx ∈ (-0.5, 0.5) and
    omega ∈ (-1.0, 1.0).  It has seen vx=0 before and will transfer cleanly
    to a purely rotational task.  Rocky slope (vx ∈ (0.2, 0.5) always forward)
    has never seen vx=0 — it stalls and falls when commanded to stop.

    Network: [512, 256, 128] — same architecture as rough, compatible with
    model_8996.pt checkpoint transfer.

    Purpose
    -------
    All existing policies produce ~1°/s actual yaw rate at vx=0 — far below
    the 57°/s commanded.  This is because they were trained only at vx > 0.
    This turn policy is trained exclusively at vx=0, vy=0, omega∈(-1, +1)
    on mixed flat+slope terrain.

    Key differences from Go2wRoughPPORunnerCfg:
      1. experiment_name → "go2w_velocity_turn"
         Logs → logs/rsl_rl/go2w_velocity_turn/ — completely separate experiment.
      2. max_iterations = 1500 — generous upper bound; expect convergence ~800.
      3. save_interval = 50 — fine-grained checkpoints to track convergence curve.

    Training command (start from rough model_8996.pt):
        conda activate env_isaacsim
        python scripts/train.py --task RexmiRl-Go2w-Velocity-Turn-v0 --headless \\
            --load_run go2w_velocity_rough/2026-06-14_20-03-41 \\
            --checkpoint model_8996.pt \\
            --max_iterations 1500

    TensorBoard health signals:
        track_ang_vel_z_exp  > 0.5 by iter 300  → spinning established
        base_contact         < 5%               → robot not falling
        track_lin_vel_xy_exp → small neg        → minimal drift
        terrain_levels       > 4.0 by iter 600  → slope generalisation working

    Nav integration (navigate.py --ckpt_turn):
        PolicySelector activates TURN mode when |heading_error| > 75°.
        The turn policy runs with vx_cmd=0, vy_cmd=0, omega_cmd=±1.0.
        LocalPlanner uses vx=0.0 during committed turns (pure rotation).
    """

    experiment_name = "go2w_velocity_turn"
    max_iterations  = 1500
    save_interval   = 50


# ==============================================================================
# Slope-turn PPO config — pivot turning on slopes up to 35°
# ==============================================================================

@configclass
class Go2wSlopeTurnPPORunnerCfg(Go2wTurnPPORunnerCfg):
    """
    PPO runner configuration for the Go2W slope-turn policy (Phase B turning).

    INHERITS FROM: Go2wTurnPPORunnerCfg
    ------------------------------------
    Identical network [512, 256, 128] and algorithm hyperparameters.
    Warm-start from the converged flat-turn checkpoint.

    Purpose
    -------
    Phase B of the two-phase turn training:
      Phase A (Go2wTurnPPORunnerCfg): flat terrain, learns basic pivot mechanics
      Phase B (this):                 slopes 15°–35°, adapts pivot to gravity

    The flat-turn policy already knows: lock wheels, step legs, rotate body.
    This training adapts that gait to tilted terrain where gravity creates
    asymmetric ground reaction forces during each leg step.

    Key differences from Go2wTurnPPORunnerCfg:
      1. experiment_name → "go2w_velocity_slope_turn"
         Logs → logs/rsl_rl/go2w_velocity_slope_turn/ — separate experiment.
      2. max_iterations = 1500 — slopes require more adaptation than flat.

    Training command (warm-start from flat-turn checkpoint):
        # Find latest flat-turn checkpoint:
        ls logs/rsl_rl/go2w_velocity_turn/ | sort | tail -1

        conda activate env_isaacsim
        python scripts/train.py --task RexmiRl-Go2w-Velocity-SlopeTurn-v0 --headless \\
            --load_run go2w_velocity_turn/<latest_run> \\
            --checkpoint model_<N>.pt \\
            --max_iterations 1500

    Nav integration:
        The slope-turn policy is the PRODUCTION turn policy for the crater demo.
        PolicySelector loads it instead of flat-turn when slopes are detected.
        Use --ckpt_turn pointing to a slope_turn checkpoint in navigate.py.
    """

    experiment_name = "go2w_velocity_slope_turn"
    max_iterations  = 1500
    save_interval   = 50


# ==============================================================================
# Curved-path curriculum turn PPO configs (runs 18-20)
# ==============================================================================

@configclass
class Go2wTurnAPPORunnerCfg(Go2wTurnPPORunnerCfg):
    """
    PPO runner for Turn Phase A — wide-arc turning (vx=0.3-0.5, omega=0.15-0.3).

    Logs to go2w_velocity_turn_a/ — separate from the pure-pivot experiment.
    Warm-start from model_8996.pt (rough walking policy).
    Expected convergence: ~500 iterations (robot already knows forward walking).

    max_iterations=500: this is a brand-new run starting from iter 0.
    The --load_run/--checkpoint flags transfer weights only — the iteration
    counter resets. 500 total iterations ≈ 25 minutes.
    """

    experiment_name = "go2w_velocity_turn_a"
    max_iterations  = 500
    save_interval   = 50


@configclass
class Go2wTurnBPPORunnerCfg(Go2wTurnPPORunnerCfg):
    """
    PPO runner for Turn Phase B — tight-arc to near-pivot (vx=0-0.2, omega=0.15-0.3).

    Logs to go2w_velocity_turn_b/ — separate from Phase A and pure-pivot.
    Warm-start from Phase A checkpoint (best model from go2w_velocity_turn_a/).
    Expected convergence: ~500 iterations.
    """

    experiment_name = "go2w_velocity_turn_b"
    max_iterations  = 500
    save_interval   = 50
