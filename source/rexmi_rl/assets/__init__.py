# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Robot asset configurations for REXMI RL.

This sub-package holds ArticulationCfg objects — Isaac Lab's data structure that
describes a robot: where its USD file lives, physics properties, actuator models,
and the joint positions/velocities it should start with.

Current assets
--------------
* go2w.py  — Go2W kinematics + actuators (16 DOF). Visual skin defaults to
             rexmi_dog; override with REXMI_ROBOT_VISUAL=go2w for Unitree look.

"""


# Re-export the Go2W config so users can do:
#   from rexmi_rl.assets import GO2W_CFG
from rexmi_rl.assets.go2w import GO2W_CFG  # noqa: F401
