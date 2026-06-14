# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Robot asset configurations for REXMI RL.

This sub-package holds ArticulationCfg objects — Isaac Lab's data structure that
describes a robot: where its USD file lives, physics properties, actuator models,
and the joint positions/velocities it should start with.

Current assets
--------------
* go2w.py  — Unitree Go2W (wheeled-legged quadruped, 16 controllable DOF)
"""

# Re-export the Go2W config so users can do:
#   from rexmi_rl.assets import GO2W_CFG
from rexmi_rl.assets.go2w import GO2W_CFG  # noqa: F401
