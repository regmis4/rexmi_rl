# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
PolicySelector — terrain-aware RL policy switcher.

Reads terrain metrics computed by the LocalPlanner and selects which of the
three trained RL policies to use for the current step:

  fast_flat    — flat/clear terrain, no obstacles, slope < 5°
                 Checkpoint: go2w_velocity_fast_flat   model_1499.pt
                 Speed: up to 2 m/s

  rough        — rough/moderate terrain, obstacles < 8 cm, slope < 20°
                 Checkpoint: go2w_velocity_rough       model_8996.pt
                 Speed: ~0.8 m/s

  rocky_slope  — steep terrain with boulders, slope 20–35°
                 Checkpoint: go2w_velocity_rocky_slope model_13994.pt
                 Speed: ~0.4 m/s

Selection logic (hysteresis to prevent rapid switching)
-------------------------------------------------------
  The selector uses a HOLD counter — a policy must be recommended for
  HOLD_STEPS consecutive steps before the switch is committed.  This prevents
  oscillation at terrain boundaries (e.g. robot straddling a slope edge).

  Terrain metrics used:
    max_step   : max vertical step in the forward columns (m)
    slope      : terrain slope ahead (tan θ)
    traversable_fraction : fraction of heading candidates that are traversable

  Decision tree (evaluated top-to-bottom, first match wins):
    1. slope > STEEP_THRESH (0.36 = tan 20°)    → rocky_slope
    2. max_step > ROUGH_THRESH (0.06 m)          → rough (or rocky_slope if also steep)
    3. else                                       → fast_flat
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Optional, Callable


class PolicyMode(Enum):
    FAST_FLAT   = "fast_flat"
    ROUGH       = "rough"
    ROCKY_SLOPE = "rocky_slope"


# Terrain thresholds
STEEP_THRESH   = math.tan(math.radians(20))   # 0.364 — above this → rocky_slope
ROUGH_THRESH   = 0.06                          # 6 cm max step → rough
VERY_STEEP     = math.tan(math.radians(28))   # 0.532 — above this → rocky_slope even if step ok
HOLD_STEPS     = 25                            # steps before committing a switch (~0.5 s at 50 Hz)


class PolicySelector:
    """
    Terrain-aware RL policy switcher.

    Loads all three policy checkpoints at startup (avoids latency during nav).
    Switches between them based on real-time terrain metrics from LocalPlanner.

    Parameters
    ----------
    policies : dict[PolicyMode, Callable]
        Mapping from PolicyMode → inference callable (output of runner.get_inference_policy()).
        All three must be provided.
    initial_mode : PolicyMode
        Which policy to start with. Default: ROCKY_SLOPE (safest for crater terrain).
    """

    def __init__(
        self,
        policies: dict[PolicyMode, Callable],
        initial_mode: PolicyMode = PolicyMode.ROCKY_SLOPE,
    ):
        if set(policies.keys()) != set(PolicyMode):
            raise ValueError(
                f"Must provide all 3 policies: {[m.value for m in PolicyMode]}. "
                f"Got: {[m.value for m in policies.keys()]}"
            )
        self._policies      = policies
        self._current_mode  = initial_mode
        self._candidate     = initial_mode
        self._hold_count    = 0

        # Counters for dashboard display
        self.switch_count   = 0
        self.mode_history:  list[str] = [initial_mode.value]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def current_mode(self) -> PolicyMode:
        return self._current_mode

    @property
    def current_policy(self) -> Callable:
        return self._policies[self._current_mode]

    def update(
        self,
        slope_ahead: float,        # tan(θ) of terrain slope in best forward column
        max_step: float,           # maximum vertical step in forward scan (m)
        traversable_fraction: float,  # 0–1: fraction of 5 candidates that are clear
    ) -> PolicyMode:
        """
        Evaluate terrain metrics and potentially switch the active policy.

        Parameters
        ----------
        slope_ahead : float
            Terrain slope (tan θ) in the best forward direction.
            0 = flat, 0.36 = 20°, 0.70 = 35°.
        max_step : float
            Largest vertical step detected in the forward scan (metres).
        traversable_fraction : float
            Fraction of the 5 heading candidates that are traversable (0.0–1.0).
            Low value → blocked/chaotic terrain.

        Returns
        -------
        PolicyMode
            The currently active policy mode (may be unchanged).
        """
        # Decide what mode is appropriate for current terrain
        recommended = self._recommend(slope_ahead, max_step, traversable_fraction)

        if recommended == self._current_mode:
            # Already on correct policy — reset hold counter
            self._candidate  = recommended
            self._hold_count = 0
        elif recommended == self._candidate:
            # Same recommendation building up — increment hold
            self._hold_count += 1
            if self._hold_count >= HOLD_STEPS:
                # Commit the switch
                old_mode = self._current_mode
                self._current_mode = recommended
                self._hold_count   = 0
                self._candidate    = recommended
                self.switch_count += 1
                self.mode_history.append(recommended.value)
                print(
                    f"[PolicySelector] switch: {old_mode.value} → {recommended.value}  "
                    f"(slope={math.degrees(math.atan(slope_ahead)):.1f}°, "
                    f"step={max_step*100:.1f}cm, "
                    f"trav={traversable_fraction:.0%})"
                )
        else:
            # Different candidate from before — restart hold
            self._candidate  = recommended
            self._hold_count = 1

        return self._current_mode

    def status_str(self) -> str:
        """Human-readable status for dashboard display."""
        return (
            f"{self._current_mode.value}"
            f"{'→' + self._candidate.value if self._candidate != self._current_mode else ''}"
            f" [{self._hold_count}/{HOLD_STEPS}]"
            f" (switched×{self.switch_count})"
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _recommend(
        slope: float,
        max_step: float,
        trav_frac: float,
    ) -> PolicyMode:
        """
        Terrain → policy recommendation (no hysteresis — pure terrain logic).

        Decision tree (top = highest priority):
          steep slope (> 20°)  → rocky_slope
          large step (> 6 cm)  → rough
          nearly blocked       → rough (conservative)
          else                 → fast_flat
        """
        # Rule 1: steep slope always needs rocky_slope
        if slope > STEEP_THRESH:
            return PolicyMode.ROCKY_SLOPE

        # Rule 2: significant obstacles → rough or rocky_slope
        if max_step > ROUGH_THRESH:
            if slope > VERY_STEEP or max_step > 0.10:
                return PolicyMode.ROCKY_SLOPE
            return PolicyMode.ROUGH

        # Rule 3: mostly blocked headings → conservative
        if trav_frac < 0.4:
            return PolicyMode.ROUGH

        # Rule 4: clear terrain
        return PolicyMode.FAST_FLAT
