# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
REXMI RL — Reinforcement learning for the REXMI wheeled-legged quadruped robot.

Built on top of NVIDIA Isaac Lab (https://github.com/isaac-sim/IsaacLab).

Package layout
--------------
rexmi_rl/
├── assets/           # Robot ArticulationCfg definitions (USD paths, actuators, init states)
└── tasks/
    └── locomotion/
        └── velocity/ # Velocity-tracking RL environments (flat + rough terrain)

Import chain (how Isaac Lab discovers our tasks)
-------------------------------------------------
When this package is installed (pip install -e .) and Isaac Lab boots, it scans
registered extensions. Our tasks/__init__.py imports each task sub-package, which
in turn calls gym.register() so the environment IDs become available engine-wide.
"""

import os

# Absolute path to the installed extension root (source/rexmi_rl/).
# Used internally to locate assets relative to the package without hard-coding paths.
REXMI_RL_EXT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Trigger task registration by importing the tasks sub-package.
# This must happen at import time so that gym.register() calls run before any
# training script tries to create an environment by ID.
from rexmi_rl import tasks  # noqa: E402, F401
