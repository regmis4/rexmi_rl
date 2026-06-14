# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Velocity-tracking locomotion environments.

These environments train the robot to follow a commanded base velocity
(linear XY + angular Z) using a Markov Decision Process (MDP) with:
- Observations: IMU, joint state, commanded velocity
- Actions:      wheel velocity targets (Phase 1) or full joint targets (Phase 2)
- Rewards:      velocity tracking + energy efficiency + stability penalties
- Termination:  body contact with ground OR timeout

Registered environments (via gym.register in config/go2w/__init__.py)
----------------------------------------------------------------------
* RexmiRl-Go2w-Velocity-Flat-v0       — flat ground, training
* RexmiRl-Go2w-Velocity-Flat-Play-v0  — flat ground, fewer envs for visualization
"""

# Importing the config sub-package triggers gym.register() calls.
import rexmi_rl.tasks.locomotion.velocity.config  # noqa: F401
