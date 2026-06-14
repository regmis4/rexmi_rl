# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Locomotion tasks for REXMI.

Currently contains:
* velocity/  — environments where the robot tracks a commanded base velocity

Future tasks (planned):
* stance/    — static balance and posture control
* climbing/  — rough terrain + crater traversal (Phase 3+)
"""

import rexmi_rl.tasks.locomotion.velocity  # noqa: F401
