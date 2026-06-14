# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Robot-specific environment configurations for velocity locomotion.

Each sub-folder contains:
  flat_env_cfg.py  — flat-terrain override of the base env
  rough_env_cfg.py — base env with robot-specific parameters
  agents/          — RL algorithm hyperparameter configs

Currently supported robots:
* go2w/ — Unitree Go2W wheeled-legged quadruped
"""

import rexmi_rl.tasks.locomotion.velocity.config.go2w  # noqa: F401
