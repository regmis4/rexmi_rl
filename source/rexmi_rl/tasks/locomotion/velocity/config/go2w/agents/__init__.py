# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
RL agent (algorithm) configurations for the Go2W velocity task.

Each file in this folder defines hyperparameters for a specific RL library:

  rsl_rl_ppo_cfg.py  — PPO via RSL-RL (recommended; fastest for Isaac Lab)
  (future) skrl_cfg.yaml  — SKRL library support
  (future) sb3_cfg.yaml   — Stable-Baselines3 support

RSL-RL is the default choice because:
  * It is the library used in Isaac Lab's own locomotion tutorials
  * It is highly optimised for GPU-parallel environments
  * The PPO implementation matches the AnymalC/Go2 baseline papers
"""
