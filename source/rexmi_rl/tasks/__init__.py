# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
RL task definitions for REXMI.

Importing this package triggers gym.register() for all environments defined
inside sub-packages. Isaac Lab's training scripts look up environments by their
gym ID string (e.g. "RexmiRl-Go2w-Velocity-Flat-v0"), so registration must
happen before any Env is created.

Task hierarchy
--------------
tasks/
└── locomotion/          ← locomotion-category tasks
    └── velocity/        ← velocity-tracking sub-category
        └── config/
            └── go2w/   ← Go2W-specific env + agent configs
"""

# Import the locomotion sub-package; this cascades down and registers all envs.
import rexmi_rl.tasks.locomotion  # noqa: F401
