# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
PolicySelector — terrain-aware RL policy switcher.

Reads terrain metrics computed by the LocalPlanner and selects which of the
four trained RL policies to use for the current step:

  fast_flat    — flat/clear terrain, no obstacles, slope < 5°
                 Checkpoint: go2w_velocity_fast_flat   model_1499.pt
                 Speed: up to 2 m/s

  rough        — rough/moderate terrain, obstacles < 8 cm, slope < 20°
                 Checkpoint: go2w_velocity_rough       model_8996.pt
                 Speed: ~0.8 m/s

  rocky_slope  — steep terrain with boulders, slope 20–35°
                 Checkpoint: go2w_velocity_rocky_slope model_13994.pt
                 Speed: ~0.4 m/s

  turn         — pure in-place rotation (any terrain, vx=0)
                 Checkpoint: go2w_velocity_turn        model_<N>.pt
                 vx=0, vy=0, omega=±1.0 rad/s
                 Activated when |heading_error| > 75° OR RecoveryFSM is rotating.
                 Train: python scripts/train.py --task RexmiRl-Go2w-Velocity-Turn-v0
                 If checkpoint not loaded, falls back to rough policy.

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
    0. RecoveryFSM is rotating OR |heading_error| > 75° → turn
    1. slope > STEEP_THRESH (0.36 = tan 20°)            → rocky_slope
    2. max_step > ROUGH_THRESH (0.06 m)                 → rough (or rocky_slope if steep)
    3. else                                              → fast_flat
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Optional, Callable


class PolicyMode(Enum):
    FAST_FLAT   = "fast_flat"
    ROUGH       = "rough"
    ROCKY_SLOPE = "rocky_slope"
    TURN        = "turn"


# Terrain thresholds
STEEP_THRESH   = math.tan(math.radians(20))   # 0.364 — above this → rocky_slope
ROUGH_THRESH   = 0.06                          # 6 cm max step → rough
VERY_STEEP     = math.tan(math.radians(28))   # 0.532 — above this → rocky_slope even if step ok

# Asymmetric hysteresis:
#   Switching TO a safer/slower policy (rough or rocky_slope) is committed quickly
#   (15 steps = 0.3 s) — we never want to stay on fast_flat on terrain that needs
#   careful handling.
#
#   Switching TO fast_flat is slow (50 steps = 1.0 s) — the terrain must be
#   consistently flat for 1 full second before we allow fast_flat.  This prevents
#   the observed oscillation where the robot flips to fast_flat for 2-3 steps
#   on a momentarily-flat patch of crater rim and then immediately back to rocky.
#
#   Switching between rough ↔ rocky_slope uses HOLD_SAFE (15 steps) in both
#   directions — both are in-distribution at slow speed, so rapid switching is OK.
HOLD_SAFE       = 15    # steps to commit switch to rough or rocky_slope (~0.3 s)
HOLD_FAST_FLAT  = 50    # steps to commit switch to fast_flat (~1.0 s) — very conservative

# Turn-override thresholds
# When |heading_error| > TURN_ENTER_RAD, force ROUGH policy (it was trained with
# vx ∈ (-0.5, 0.5) and omega ∈ (-1.0, 1.0) — handles vx=0 point turns correctly).
# rocky_slope was trained with vx ∈ (0.2, 0.5) only — vx=0 is OOD and causes stalling.
# fast_flat has no height scanner — unsafe on crater terrain.
#
# Threshold lowered 90° → 75° (2026-07-14):
# At 90°, heading drift on the crater slope was observed growing from 70° to 91°
# over ~50 steps before TURN OVERRIDE fired.  By then vx=0.40 had already driven
# the robot 1-2 m in the wrong direction.  At 75° we catch it earlier while the
# position error is still small.  The local_planner's committed-turn latch still
# uses 90° (independent of the policy switch) — these thresholds are decoupled.
TURN_ENTER_RAD = math.radians(75)   # enter turn-override when |he| > 75°
TURN_EXIT_RAD  = math.radians(40)   # release turn-override when |he| < 40° (hysteresis)


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
        # Allow either 3 policies (no turn — falls back to ROUGH for turns) or
        # 4 policies (with turn — uses dedicated turn policy for in-place rotation).
        # This keeps backward compatibility when --ckpt_turn is not provided.
        if not set(policies.keys()).issubset(set(PolicyMode)):
            raise ValueError(
                f"Unknown PolicyMode keys: {[m.value for m in policies.keys()]}"
            )
        required = {PolicyMode.FAST_FLAT, PolicyMode.ROUGH, PolicyMode.ROCKY_SLOPE}
        missing  = required - set(policies.keys())
        if missing:
            raise ValueError(
                f"Must provide at least fast_flat, rough, rocky_slope policies. "
                f"Missing: {[m.value for m in missing]}"
            )
        # If TURN not provided, fall back to ROUGH for turn-override.
        # rough has vx∈(-0.5,0.5) so it at least partially handles vx=0 turns.
        # Log a warning so the user knows turn quality will be degraded.
        if PolicyMode.TURN not in policies:
            policies[PolicyMode.TURN] = policies[PolicyMode.ROUGH]
            self._has_turn_policy = False
        else:
            self._has_turn_policy = True
        self._policies      = policies
        self._current_mode  = initial_mode
        self._candidate     = initial_mode
        self._hold_count    = 0

        # Turn-override state: True when |heading_error| > TURN_ENTER_RAD.
        # While in turn-override, ROUGH is forced regardless of terrain.
        # Released when |heading_error| < TURN_EXIT_RAD (hysteresis dead-band).
        self._in_turn_override: bool = False

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
        heading_error_rad: float = 0.0,  # current heading error to waypoint (rad)
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
        heading_error_rad : float
            Current heading error to the active waypoint (radians, wrapped to ±π).
            Used for turn-override: when |he| > 90°, force ROUGH policy which was
            trained with vx ∈ (-0.5, 0.5) and can handle vx=0 point turns.
            rocky_slope (vx ∈ (0.2, 0.5) always forward) stalls at vx=0.

        Returns
        -------
        PolicyMode
            The currently active policy mode (may be unchanged).
        """
        # ------------------------------------------------------------------
        # TURN OVERRIDE (highest priority — checked before terrain rules)
        #
        # When the goal is more than 90° away, the robot must spin in place.
        # rocky_slope was trained with vx ∈ (0.2, 0.5) only — it has never
        # seen vx=0 and produces a stalled idle gait when commanded to stop.
        # rough was trained with vx ∈ (-0.5, 0.5) and omega ∈ (-1.0, 1.0)
        # — it handles genuine zero-vx point turns correctly.
        #
        # Hysteresis: enter at |he| > 90°, release at |he| < 45°.
        # ------------------------------------------------------------------
        he_abs = abs(heading_error_rad)
        if not self._in_turn_override and he_abs > TURN_ENTER_RAD:
            # Enter turn-override — use TURN policy (or ROUGH fallback if no turn ckpt).
            # TURN is trained with vx=0, vy=0, omega∈(-1,+1) on mixed flat+slope terrain.
            # It produces clean in-place rotation without the ~1°/s stall seen when
            # rough/rocky_slope receives vx=0 commands out-of-distribution.
            self._in_turn_override = True
            turn_policy = PolicyMode.TURN   # always available (falls back to ROUGH in __init__)
            if self._current_mode != turn_policy:
                old_mode = self._current_mode
                self._current_mode = turn_policy
                self._candidate    = turn_policy
                self._hold_count   = 0
                self.switch_count += 1
                self.mode_history.append(turn_policy.value)
                turn_label = "turn" if self._has_turn_policy else "rough(turn-fallback)"
                print(
                    f"[PolicySelector] TURN OVERRIDE: {old_mode.value} → {turn_label} "
                    f"(|he|={math.degrees(he_abs):.0f}° > {math.degrees(TURN_ENTER_RAD):.0f}° "
                    f"— turning in place)"
                )
            return self._current_mode

        if self._in_turn_override:
            if he_abs < TURN_EXIT_RAD:
                # Heading error reduced to < 40° — release turn override
                self._in_turn_override = False
                print(
                    f"[PolicySelector] TURN OVERRIDE released "
                    f"(|he|={math.degrees(he_abs):.0f}° < {math.degrees(TURN_EXIT_RAD):.0f}° "
                    f"— resuming terrain selection)"
                )
                # Fall through to terrain-based selection below
            else:
                # Still spinning — stay in turn-override mode
                return self._current_mode

        # Decide what mode is appropriate for current terrain
        recommended = self._recommend(slope_ahead, max_step, traversable_fraction)

        if recommended == self._current_mode:
            # Already on correct policy — reset hold counter
            self._candidate  = recommended
            self._hold_count = 0
        elif recommended == self._candidate:
            # Same recommendation building up — increment hold.
            # Use asymmetric hold: fast_flat requires 50 steps, safe policies only 15.
            hold_needed = (
                HOLD_FAST_FLAT if recommended == PolicyMode.FAST_FLAT else HOLD_SAFE
            )
            self._hold_count += 1
            if self._hold_count >= hold_needed:
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
                    f"trav={traversable_fraction:.0%}, "
                    f"held={hold_needed} steps)"
                )
        else:
            # Different candidate from before — restart hold
            self._candidate  = recommended
            self._hold_count = 1

        return self._current_mode

    def status_str(self) -> str:
        """Human-readable status for dashboard display."""
        # Show the correct hold limit depending on which policy is pending
        _hold_limit = (
            HOLD_FAST_FLAT if self._candidate == PolicyMode.FAST_FLAT else HOLD_SAFE
        )
        return (
            f"{self._current_mode.value}"
            f"{'→' + self._candidate.value if self._candidate != self._current_mode else ''}"
            f" [{self._hold_count}/{_hold_limit}]"
            f" (switched×{self.switch_count})"
        )

    # Per-policy commanded forward speeds (set by navigate.py after loading ckpts).
    # These are the in-distribution vx values for each policy's training range:
    #   fast_flat  : trained vx ∈ (-0.5, 2.0) → command 1.5 m/s (confident mid-range)
    #   rough      : trained vx ∈ (-0.5, 0.5) → command 0.45 m/s (near training max)
    #   rocky_slope: trained vx ∈ (0.2, 0.5)  → command 0.40 m/s (near training max)
    #   turn       : trained vx = 0.0          → command 0.0 m/s (pure rotation)
    # Exposed as a property so LocalPlanner can query which speed to use this step.
    POLICY_VX: dict = {
        PolicyMode.FAST_FLAT:   1.5,
        PolicyMode.ROUGH:       0.45,
        PolicyMode.ROCKY_SLOPE: 0.40,
        PolicyMode.TURN:        0.0,    # turn-in-place: zero forward speed
    }

    def current_vx(self) -> float:
        """Return the in-distribution forward speed for the currently active policy."""
        return self.POLICY_VX.get(self._current_mode, 0.40)

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
