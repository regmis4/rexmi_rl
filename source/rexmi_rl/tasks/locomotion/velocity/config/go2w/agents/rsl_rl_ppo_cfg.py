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

Key hyperparameters explained below in the config.

Network architecture choice
-----------------------------
Phase 1 (wheels only, no height scan):
  Observation size ≈ 3 + 3 + 3 + 3 + 16 + 16 + 4 = 48 dims
    (lin_vel + ang_vel + proj_gravity + vel_cmd + joint_pos + joint_vel + actions)
  Action size = 4 (one velocity target per wheel)

  A small network [128, 128, 128] is sufficient — the task is relatively simple.
  We'll scale up to [512, 256, 128] in Phase 2 when height scan adds ~160 dims.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class Go2wFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """
    PPO runner configuration for the Go2W flat-terrain velocity task.

    The runner orchestrates the training loop:
      collect rollout → compute returns → update networks → log → repeat
    """

    # Number of physics steps collected per environment per update.
    # With 4096 envs and 24 steps: 4096 × 24 = 98,304 transitions per update.
    # More steps = better advantage estimation but slower updates.
    num_steps_per_env = 24

    # Total number of policy update iterations.
    # 300 iterations × 98k transitions = ~29M environment interactions.
    # This is sufficient for flat wheel locomotion (~20-30 min on a good GPU).
    # Phase 2: 8D action space (4 wheels + 4 thighs) needs more training.
    # ~1000 iterations × 98k transitions = ~98M environment interactions.
    max_iterations = 1000

    # Save a checkpoint every N iterations (useful for resuming training)
    save_interval = 50

    # Name used for the log folder under logs/rsl_rl/
    # Full path: logs/rsl_rl/go2w_velocity_flat/<date>/
    experiment_name = "go2w_velocity_flat"

    # empirical_normalization: normalise observations using running mean/std
    # computed from actual rollout data.  False = use fixed normalisation
    # defined in the env (simpler, sufficient for Phase 1).
    empirical_normalization = False

    # ------------------------------------------------------------------
    # Actor-Critic network architecture
    # ------------------------------------------------------------------
    policy = RslRlPpoActorCriticCfg(
        # Initial standard deviation of the action distribution.
        # 1.0 = high initial exploration; the policy will reduce this as it learns.
        init_noise_std=1.0,

        # Hidden layer sizes for the actor (policy) network.
        # [128, 128, 128] = three hidden layers of 128 units each with ELU activations.
        # Suitable for the ~48-dim observation space of Phase 1.
        actor_hidden_dims=[128, 128, 128],

        # Hidden layer sizes for the critic (value function) network.
        # Can be the same as the actor; it predicts scalar state values.
        critic_hidden_dims=[128, 128, 128],

        # Activation function between layers.
        # ELU (Exponential Linear Unit) is standard for locomotion RL:
        # it avoids dying neurons (unlike ReLU) and produces smooth gradients.
        activation="elu",
    )

    # ------------------------------------------------------------------
    # PPO algorithm hyperparameters
    # ------------------------------------------------------------------
    algorithm = RslRlPpoAlgorithmCfg(
        # Weight of the value function loss relative to the policy loss.
        # 1.0 = equal weighting.
        value_loss_coef=1.0,

        # Use clipped value loss (standard PPO improvement — prevents the
        # critic from changing too rapidly).
        use_clipped_value_loss=True,

        # Clip ratio ε: the policy update is clipped to [1-ε, 1+ε] × old_ratio.
        # 0.2 is the original PPO paper value and works well here.
        clip_param=0.2,

        # Entropy coefficient: adds entropy bonus to encourage exploration.
        # 0.01 is a small but non-zero value — keeps the policy from collapsing
        # to deterministic behaviour too early.
        entropy_coef=0.01,

        # Number of gradient update epochs per collected batch.
        # 5 is standard for locomotion tasks.
        num_learning_epochs=5,

        # Split the batch into N mini-batches for each update epoch.
        # More mini-batches = smaller gradient updates = more stable learning.
        num_mini_batches=4,

        # Learning rate (or initial LR if using adaptive schedule).
        learning_rate=1.0e-3,

        # Learning rate schedule.
        # "adaptive" adjusts the LR based on the KL divergence between old and
        # new policy — reduces LR if updates are too large, increases if too small.
        schedule="adaptive",

        # Discount factor γ: how much future rewards are worth.
        # 0.99 = strongly considers long-term rewards (good for locomotion).
        gamma=0.99,

        # GAE-λ: Generalised Advantage Estimation smoothing parameter.
        # 0.95 balances bias vs variance in advantage estimates.
        lam=0.95,

        # Target KL divergence for adaptive LR schedule.
        # If KL > desired_kl × 2, LR is halved; if KL < desired_kl / 2, LR doubles.
        desired_kl=0.01,

        # Maximum gradient norm for gradient clipping.
        # Prevents exploding gradients during early training.
        max_grad_norm=1.0,
    )
