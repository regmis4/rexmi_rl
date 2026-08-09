# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
PolicySelector — conservative terrain-aware switcher for crater nav.

Policies (demo):
  rough        — moderate terrain / gentle slopes
  rocky_slope  — steep crater walls / boulders
  turn         — Pulse slope-turn (Language A HOLD→YAW→SETTLE via ReorientController)

No fast_flat / turn_flat in the crater demo path.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Callable


class PolicyMode(Enum):
    ROUGH       = "rough"
    ROCKY_SLOPE = "rocky_slope"
    TURN        = "turn"
    # Kept for backward-compat imports; mapped to ROUGH / TURN if used.
    FAST_FLAT   = "fast_flat"
    TURN_FLAT   = "turn_flat"


STEEP_THRESH  = math.tan(math.radians(18))   # prefer rocky a bit earlier
ROUGH_THRESH  = 0.06
VERY_STEEP    = math.tan(math.radians(28))
HOLD_SAFE     = 15

# Large heading: force turn policy (ReorientController owns ω pulses)
TURN_ENTER_RAD = math.radians(60)   # turn policy for brake+pivot
TURN_EXIT_RAD  = math.radians(30)


class PolicySelector:
    """
    Rough / rocky_slope / turn only.

    Parameters
    ----------
    policies : dict[PolicyMode, Callable]
        Must include ROUGH and ROCKY_SLOPE. TURN optional (falls back to ROUGH).
    """

    POLICY_VX = {
        PolicyMode.ROUGH:       0.45,
        PolicyMode.ROCKY_SLOPE: 0.40,
        PolicyMode.TURN:        0.0,
        PolicyMode.FAST_FLAT:   0.45,  # alias → treat as rough speed if ever selected
        PolicyMode.TURN_FLAT:   0.0,
    }

    def __init__(
        self,
        policies: dict[PolicyMode, Callable],
        initial_mode: PolicyMode = PolicyMode.ROCKY_SLOPE,
    ):
        # Normalize aliases
        if PolicyMode.FAST_FLAT in policies and PolicyMode.ROUGH not in policies:
            policies[PolicyMode.ROUGH] = policies[PolicyMode.FAST_FLAT]
        if PolicyMode.TURN_FLAT in policies and PolicyMode.TURN not in policies:
            policies[PolicyMode.TURN] = policies[PolicyMode.TURN_FLAT]

        required = {PolicyMode.ROUGH, PolicyMode.ROCKY_SLOPE}
        missing = required - set(policies.keys())
        if missing:
            raise ValueError(
                f"Must provide rough + rocky_slope. Missing: {[m.value for m in missing]}"
            )

        if PolicyMode.TURN not in policies:
            policies[PolicyMode.TURN] = policies[PolicyMode.ROUGH]
            self._has_turn_policy = False
            print("[PolicySelector] WARNING: no turn ckpt — turn uses rough fallback")
        else:
            self._has_turn_policy = True

        # Aliases point at safe policies
        policies.setdefault(PolicyMode.FAST_FLAT, policies[PolicyMode.ROUGH])
        policies.setdefault(PolicyMode.TURN_FLAT, policies[PolicyMode.TURN])

        if initial_mode in (PolicyMode.FAST_FLAT,):
            initial_mode = PolicyMode.ROUGH
        if initial_mode == PolicyMode.TURN_FLAT:
            initial_mode = PolicyMode.TURN

        self._policies = policies
        self._current_mode = initial_mode
        self._candidate = initial_mode
        self._hold_count = 0
        self._in_turn_override = False
        self._turn_slope_ahead = 0.0
        self.switch_count = 0
        self.mode_history: list[str] = [initial_mode.value]
        self._has_turn_flat = False  # disabled for crater demo

    @property
    def current_mode(self) -> PolicyMode:
        return self._current_mode

    @property
    def current_policy(self) -> Callable:
        return self._policies[self._current_mode]

    def current_vx(self) -> float:
        return self.POLICY_VX.get(self._current_mode, 0.40)

    def force_turn(self, slope_ahead: float = 0.0) -> PolicyMode:
        """Immediate turn override (reorient / recovery). Always TURN (pulse)."""
        self._in_turn_override = True
        self._turn_slope_ahead = 0.0 if slope_ahead != slope_ahead else float(slope_ahead)
        turn_mode = PolicyMode.TURN
        if self._current_mode != turn_mode:
            old = self._current_mode
            self._current_mode = turn_mode
            self._candidate = turn_mode
            self._hold_count = 0
            self.switch_count += 1
            self.mode_history.append(turn_mode.value)
            label = "turn" if self._has_turn_policy else "rough(turn-fallback)"
            print(f"[PolicySelector] TURN FORCE: {old.value} → {label}")
        return self._current_mode

    def release_turn(self) -> None:
        self._in_turn_override = False

    def update(
        self,
        slope_ahead: float,
        max_step: float,
        traversable_fraction: float,
        heading_error_rad: float = 0.0,
    ) -> PolicyMode:
        he_abs = abs(heading_error_rad)

        # Rocky-primary: do NOT auto-switch to turn from heading alone.
        # Navigator calls force_turn() only after brake + about-face.
        if self._in_turn_override:
            return self._current_mode

        recommended = self._recommend(slope_ahead, max_step, traversable_fraction)
        # Never pick fast_flat
        if recommended == PolicyMode.FAST_FLAT:
            recommended = PolicyMode.ROUGH

        if recommended == self._current_mode:
            self._candidate = recommended
            self._hold_count = 0
        elif recommended == self._candidate:
            self._hold_count += 1
            if self._hold_count >= HOLD_SAFE:
                old = self._current_mode
                self._current_mode = recommended
                self._hold_count = 0
                self.switch_count += 1
                self.mode_history.append(recommended.value)
                print(
                    f"[PolicySelector] switch: {old.value} → {recommended.value}  "
                    f"(slope={math.degrees(math.atan(max(0.0, slope_ahead))):.1f}°, "
                    f"step={max_step*100:.1f}cm)"
                )
        else:
            self._candidate = recommended
            self._hold_count = 1

        return self._current_mode

    def status_str(self) -> str:
        return (
            f"{self._current_mode.value}"
            f"{'→' + self._candidate.value if self._candidate != self._current_mode else ''}"
            f" [{self._hold_count}/{HOLD_SAFE}]"
            f" (switched×{self.switch_count})"
        )

    @staticmethod
    def _recommend(slope: float, max_step: float, trav_frac: float) -> PolicyMode:
        if slope != slope:
            slope = 0.0
        if slope > STEEP_THRESH:
            return PolicyMode.ROCKY_SLOPE
        if max_step > ROUGH_THRESH:
            if slope > VERY_STEEP or max_step > 0.10:
                return PolicyMode.ROCKY_SLOPE
            return PolicyMode.ROUGH
        if trav_frac < 0.4:
            return PolicyMode.ROUGH
        # Clear-ish terrain still use rough (no fast_flat)
        return PolicyMode.ROUGH
