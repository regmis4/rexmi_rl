# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Mission planner — crater-relative, size-agnostic waypoint sequences.

All missions are defined as fractions of ``r_rim`` and ``r_floor``, so they
work for any crater regardless of actual radius.

Available missions
------------------
  traverse     — Enter crater at rim, cross the floor, exit opposite side.
                 Best for showcasing the full uphill + downhill capability.
  survey       — Systematic lawnmower scan of the crater floor.
                 Simulates a resource-prospecting mission.
  rim_circuit  — Clockwise loop around the crater rim.
                 Simulates a perimeter survey before descent.

Waypoint format
---------------
  Each waypoint is a ``Waypoint`` dataclass:
    x, y            — world-frame target position (metres)
    arrival_radius  — waypoint reached when robot is within this distance (m)
    label           — human-readable name for dashboard display

Usage
-----
    planner = MissionPlanner(
        crater_centre=(0.0, 0.0),
        r_floor=3.0,
        r_rim=11.0,
        spawn_x=-18.0,    # robot starts this far from crater centre
    )
    waypoints = planner.get_waypoints(Mission.TRAVERSE)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import List


class Mission(Enum):
    TRAVERSE       = "traverse"
    SURVEY         = "survey"
    RIM_CIRCUIT    = "rim_circuit"
    VELOCITY_GOAL  = "velocity_goal"   # single user-specified waypoint — flat terrain debug


@dataclass
class Waypoint:
    x:              float
    y:              float
    arrival_radius: float  # metres — "close enough" distance
    label:          str    # shown on dashboard


class MissionPlanner:
    """
    Generates crater-relative waypoint sequences for each mission type.

    Parameters
    ----------
    crater_centre : (float, float)
        World-frame (x, y) of the crater centre.
    r_floor : float
        Radius of the flat crater floor (metres).
    r_rim : float
        Radius of the crater rim (metres).
    spawn_x : float
        Robot's initial x position relative to crater centre (negative = starts
        on the -x side of the crater, drives toward +x to enter).
        Default -18.0 matches the bowl demo spawn (robot spawns ~7 m outside rim).
    entry_azimuth_deg : float
        Azimuth angle (degrees, CCW from +x) at which the robot enters the crater.
        Default 180° = enters from the -x side heading toward +x.
    """

    def __init__(
        self,
        crater_centre: tuple[float, float] = (0.0, 0.0),
        r_floor: float = 3.0,
        r_rim: float = 11.0,
        spawn_x: float = -18.0,
        entry_azimuth_deg: float = 180.0,
    ):
        self.cx, self.cy = crater_centre
        self.r_floor = r_floor
        self.r_rim   = r_rim
        self.spawn_x = spawn_x
        self.entry_az = math.radians(entry_azimuth_deg)

        # Derived geometry
        self._entry_x = self.cx + self.r_rim * math.cos(self.entry_az)
        self._entry_y = self.cy + self.r_rim * math.sin(self.entry_az)
        self._exit_x  = self.cx - self.r_rim * math.cos(self.entry_az)
        self._exit_y  = self.cy - self.r_rim * math.sin(self.entry_az)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_waypoints(
        self,
        mission: Mission,
        goal_x: float = 0.0,
        goal_y: float = 0.0,
        goal_radius: float = 1.5,
    ) -> List[Waypoint]:
        """Return the ordered waypoint list for the given mission.

        Parameters
        ----------
        goal_x, goal_y : float
            Used only for Mission.VELOCITY_GOAL — the single target position.
        goal_radius : float
            Arrival radius for velocity_goal waypoint (m). Default 1.5 m.
        """
        if mission == Mission.TRAVERSE:
            return self._traverse()
        elif mission == Mission.SURVEY:
            return self._survey()
        elif mission == Mission.RIM_CIRCUIT:
            return self._rim_circuit()
        elif mission == Mission.VELOCITY_GOAL:
            return self._velocity_goal(goal_x, goal_y, goal_radius)
        else:
            raise ValueError(f"Unknown mission: {mission}")

    # ------------------------------------------------------------------
    # Mission implementations
    # ------------------------------------------------------------------

    def _traverse(self) -> List[Waypoint]:
        """
        Classic crater traverse: outside → rim → floor centre → opposite rim → outside.

        This showcases the full capability: downhill approach, floor navigation,
        uphill exit.  The most visually compelling demo for investors.
        """
        wps: List[Waypoint] = []

        # Approach corridor only if spawn is far enough from the rim that a
        # mid-point waypoint is useful.  Bowl demo defaults (spawn_x=+13,
        # rim at +11) put spawn only 2 m outside the rim — a halfway
        # "approach" at +12 is already inside arrival_radius=2 m at spawn
        # and would be skipped immediately (or worse, arm the closest-
        # approach gate and skip without driving).
        dist_spawn_to_entry = math.hypot(
            self.spawn_x - self._entry_x, self.cy - self._entry_y
        )
        if dist_spawn_to_entry > 4.0:
            approach_x = self.spawn_x + 0.5 * (self._entry_x - self.spawn_x)
            approach_y = self.cy
            wps.append(Waypoint(approach_x, approach_y, 2.0, "approach"))

        # rim_entry/rim_exit: 1.5 m — require genuine rim crossing
        wps.append(Waypoint(self._entry_x, self._entry_y, 1.5, "rim_entry"))
        wps.append(Waypoint(self.cx, self.cy, 2.0, "floor_centre"))
        wps.append(Waypoint(self._exit_x, self._exit_y, 1.5, "rim_exit"))
        # Continue 7 m beyond the far rim
        wps.append(
            Waypoint(
                self._exit_x - 7.0 * math.cos(self.entry_az),
                self._exit_y - 7.0 * math.sin(self.entry_az),
                3.0,
                "exit_clear",
            )
        )
        return wps


    def _survey(self) -> List[Waypoint]:
        """
        Lawnmower survey of the crater floor.

        Generates parallel east–west sweeps spaced 1.5 m apart, covering the
        full floor area (r_floor radius).  Simulates a resource-prospecting
        mission scanning for water-ice deposits.

        Pattern: enter at rim → descend to floor → lawnmower rows → exit.
        """
        wps = [
            Waypoint(self._entry_x, self._entry_y, 1.5, "rim_entry"),
        ]

        # Lawnmower rows within floor radius
        stripe_spacing = 1.5   # metres between rows
        n_stripes = max(1, int(2 * self.r_floor / stripe_spacing))
        y_start   = self.cy - (n_stripes // 2) * stripe_spacing

        for i in range(n_stripes):
            y_row = y_start + i * stripe_spacing
            if abs(y_row - self.cy) > self.r_floor:
                continue   # outside floor disk — skip
            # Clamp row endpoints to floor circle
            dx = math.sqrt(max(0.0, self.r_floor**2 - (y_row - self.cy)**2))
            x_left  = self.cx - dx * 0.9   # 0.9× to stay inside rim
            x_right = self.cx + dx * 0.9
            label_row = f"survey_row_{i}"
            if i % 2 == 0:   # alternate direction (lawnmower)
                wps.append(Waypoint(x_left,  y_row, 1.0, label_row + "_L"))
                wps.append(Waypoint(x_right, y_row, 1.0, label_row + "_R"))
            else:
                wps.append(Waypoint(x_right, y_row, 1.0, label_row + "_R"))
                wps.append(Waypoint(x_left,  y_row, 1.0, label_row + "_L"))

        wps.append(Waypoint(self._exit_x, self._exit_y, 1.5, "rim_exit"))
        return wps

    def _rim_circuit(self) -> List[Waypoint]:
        """
        Clockwise loop around the crater rim.

        8 waypoints at r = 1.1 × r_rim (just outside the rim crest), 45° apart.
        Simulates a perimeter survey before deciding on an entry point.
        """
        r_circuit = self.r_rim * 1.15   # slightly outside rim
        n_points  = 8
        wps = []
        for i in range(n_points + 1):   # +1 closes the loop
            angle = self.entry_az - i * (2 * math.pi / n_points)  # CW = subtract
            x = self.cx + r_circuit * math.cos(angle)
            y = self.cy + r_circuit * math.sin(angle)
            label = f"rim_{i % n_points}"
            wps.append(Waypoint(x, y, 2.0, label))
        return wps

    def _velocity_goal(
        self,
        goal_x: float,
        goal_y: float,
        arrival_radius: float = 1.5,
    ) -> List[Waypoint]:
        """
        Single-waypoint mission for flat-terrain motion debugging.

        The robot drives directly to (goal_x, goal_y) — no crater geometry,
        no intermediate waypoints.  Use this to:
          • Verify the turn-then-drive sequence on flat ground
          • Measure straight-line tracking accuracy
          • Isolate steering vs. locomotion issues before adding terrain complexity

        Usage (navigate.py):
            --mission velocity_goal --goal_x 5.0 --goal_y 0.0
        """
        return [
            Waypoint(goal_x, goal_y, arrival_radius, "velocity_goal"),
        ]
