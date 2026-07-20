# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
REXMI autonomous navigation layer.

A deterministic nav stack that sits above the RL policies and converts
high-level mission goals into per-step (vx, vy, ωz) velocity commands.

Architecture
------------
  MissionPlanner   — crater-relative waypoint sequences (size-agnostic)
  Localizer        — robot pose provider (SimLocalizer / SLAMLocalizer)
  LidarSLAM        — 3D ICP SLAM with voxel map (phase N-2)
  OccupancyMap     — accumulates RayCaster + LiDAR cloud into 320×320 grid
  GlobalPlanner    — A* on occupancy map → next waypoint
  LocalPlanner     — height-scan traversability → (vx, vy, ωz) command
  RecoveryFSM      — stuck detection and escape behaviour
  PolicySelector   — terrain-aware RL policy switcher
  Dashboard        — live 3D SLAM cloud + 2D cost map (matplotlib, daemon thread)
  Navigator        — top-level loop tying all modules together
"""

from rexmi_rl.nav.localizer import SimLocalizer, SLAMLocalizer, Localizer, Pose
from rexmi_rl.nav.slam import LidarSLAM, SLAMPose
from rexmi_rl.nav.mission import MissionPlanner, Mission
from rexmi_rl.nav.occupancy_map import OccupancyMap
from rexmi_rl.nav.global_planner import GlobalPlanner
from rexmi_rl.nav.local_planner import LocalPlanner
from rexmi_rl.nav.recovery import RecoveryFSM
from rexmi_rl.nav.policy_selector import PolicySelector, PolicyMode
from rexmi_rl.nav.navigator import Navigator

__all__ = [
    "SimLocalizer", "SLAMLocalizer", "Localizer", "Pose",
    "LidarSLAM", "SLAMPose",
    "MissionPlanner", "Mission",
    "OccupancyMap",
    "GlobalPlanner",
    "LocalPlanner",
    "RecoveryFSM",
    "PolicySelector", "PolicyMode",
    "Navigator",
]
