# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
REXMI autonomous navigation layer.

A deterministic nav stack that sits above the RL policies and converts
high-level mission goals into per-step (vx, vy, ωz) velocity commands.

Architecture
------------
  MissionPlanner   — crater-relative waypoint sequences (size-agnostic)
  Localizer        — robot pose provider (SimLocalizer for sim, SLAM for real)
  OccupancyMap     — accumulates height-scan point cloud into traversability grid
  GlobalPlanner    — A* on occupancy map → next waypoint
  LocalPlanner     — height-scan traversability → (vx, vy, ωz) command
  RecoveryFSM      — stuck detection and escape behaviour
  Dashboard        — live 3D point cloud + 2D cost map (matplotlib, daemon thread)
  Navigator        — top-level loop tying all modules together
"""

from rexmi_rl.nav.localizer import SimLocalizer, Localizer
from rexmi_rl.nav.mission import MissionPlanner, Mission
from rexmi_rl.nav.occupancy_map import OccupancyMap
from rexmi_rl.nav.global_planner import GlobalPlanner
from rexmi_rl.nav.local_planner import LocalPlanner
from rexmi_rl.nav.recovery import RecoveryFSM
from rexmi_rl.nav.policy_selector import PolicySelector, PolicyMode
from rexmi_rl.nav.navigator import Navigator

__all__ = [
    "SimLocalizer", "Localizer",
    "MissionPlanner", "Mission",
    "OccupancyMap",
    "GlobalPlanner",
    "LocalPlanner",
    "RecoveryFSM",
    "PolicySelector", "PolicyMode",
    "Navigator",
]
