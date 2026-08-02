# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Custom command terms for REXMI velocity-tracking environments.

RelativeHeadingVelocityCommand — CONTINUOUS, BOUNDED HEADING TARGETS
===================================================================

WHY THIS EXISTS
---------------
Isaac Lab's `UniformVelocityCommand` samples the heading target as an
ABSOLUTE world yaw, uniform over `ranges.heading` (default `(-pi, pi)`):

    self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)

and then drives it with a clipped P-controller:

    omega_cmd = clip(k * wrap_to_pi(heading_target - heading_now), -w, +w)

Two things go wrong with that for a slow pivot turner:

1. THE TARGET IS UNREACHABLE WITHIN ONE RESAMPLE WINDOW.
   Turning takes |error| / w seconds. With w = 0.12 rad/s the mean error of
   pi/2 needs 13.1 s and the worst case pi needs 26.2 s — but the default
   resample period is 10 s. The target is replaced mid-turn.

2. THE REPLACEMENT IS UNCORRELATED WITH THE CURRENT HEADING.
   A fresh uniform draw over (-pi, pi) reverses the required turn direction
   roughly half the time, so `omega_cmd` flips sign DISCONTINUOUSLY (+w to -w)
   while the legs are mid-pivot and the wheels are ANCHORED. Observed result:
   the robot suddenly rolls onto its side or pitches into a somersault.

Both are fixed by sampling the target RELATIVE to where the robot is
already pointing, with a BOUNDED delta:

    heading_target = wrap_to_pi(heading_now + sign * uniform(delta_min, delta_max))

Properties:
  * CONTINUOUS   — each target continues from the previous heading; no jumps
  * REACHABLE    — delta_max / w seconds < resample period (assert enforced)
  * SYMMETRIC    — the sign is a fair coin, so there is no directional bias.
                   This matters: an asymmetric yaw command is unsatisfiable
                   and caused the v9-a 99.9% fall rate.
  * NON-TRIVIAL  — |delta| >= delta_min guarantees a real turn, so the robot
                   does not sit idle when a draw lands near zero.

The commanded `omega` magnitude is unchanged (still bounded by the same clip),
so this stays INSIDE the trained distribution — it only removes the
discontinuous reversals.

BONUS: THIS IS THE NAV INTERFACE
--------------------------------
The nav layer emits relative course corrections — "turn 30 degrees left to
get around that boulder" — not absolute world compass headings. A bounded
relative heading delta is exactly that, so training against this term also
means training against the interface nav will actually drive.
"""

from __future__ import annotations

import math
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class RelativeHeadingVelocityCommand(UniformVelocityCommand):
    """
    Velocity command whose heading target is relative to the CURRENT heading.

    Identical to `UniformVelocityCommand` in every respect except how
    `heading_target` is drawn on resample. See the module docstring.
    """

    cfg: RelativeHeadingVelocityCommandCfg

    def __init__(self, cfg: RelativeHeadingVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # ------------------------------------------------------------------
        # PERSISTENT TURN DIRECTION.
        #
        # A fresh coin flip on EVERY resample makes successive small turns
        # CANCEL instead of accumulate, so the robot ends up oscillating around
        # one heading and never visibly rotates. Holding the sign across
        # windows makes the small bounded deltas CHAIN into one long, smooth
        # rotation, which is what "keep turning" actually looks like.
        # ------------------------------------------------------------------
        self._turn_sign = torch.where(
            torch.rand(self.num_envs, device=self.device) < 0.5,
            -torch.ones(self.num_envs, device=self.device),
            torch.ones(self.num_envs, device=self.device),
        )


        # ------------------------------------------------------------------
        # Fail loudly if the turn cannot physically finish in the window.
        #
        # This is the ENTIRE point of the class — if the assert below can
        # trip, we are back to replacing targets mid-turn and the
        # discontinuous yaw reversal returns.
        #
        # Saturated turn time = delta_max / w, where w is the ang_vel_z clip.
        # The P-controller only saturates while |k * error| > w, after which
        # it decays exponentially with time constant 1/k, so allow a settling
        # margin on top of the saturated phase.
        # ------------------------------------------------------------------
        w = min(abs(cfg.ranges.ang_vel_z[0]), abs(cfg.ranges.ang_vel_z[1]))
        if w > 1.0e-6:
            saturated_s = cfg.heading_delta_max / w
            settle_s = 3.0 / max(cfg.heading_control_stiffness, 1.0e-6)
            need_s = saturated_s + settle_s
            have_s = min(cfg.resampling_time_range)
            if have_s < saturated_s:
                raise ValueError(
                    f"RelativeHeadingVelocityCommand: resampling_time_range min "
                    f"({have_s:.1f}s) is shorter than the time needed to turn "
                    f"heading_delta_max={cfg.heading_delta_max:.2f} rad at the "
                    f"ang_vel_z clip {w:.3f} rad/s ({saturated_s:.1f}s). The target "
                    f"would be replaced mid-turn, reintroducing the discontinuous "
                    f"yaw reversal this class exists to prevent. Use resampling "
                    f">= {need_s:.1f}s, or reduce heading_delta_max to "
                    f"<= {have_s * w:.2f} rad."
                )

    def __str__(self) -> str:
        return (
            f"RelativeHeadingVelocityCommand:\n"
            f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
            f"\tResampling time range: {self.cfg.resampling_time_range}\n"
            f"\tHeading delta: +/-({self.cfg.heading_delta_min:.2f}, "
            f"{self.cfg.heading_delta_max:.2f}) rad, relative to current heading\n"
            f"\tang_vel_z clip: {self.cfg.ranges.ang_vel_z}"
        )

    def _resample_command(self, env_ids):
        # Draw lin_vel_x / lin_vel_y / ang_vel_z / standing mask exactly as the
        # parent does, so nothing else changes.
        super()._resample_command(env_ids)

        if not self.cfg.heading_command:
            return

        # ------------------------------------------------------------------
        # Override ONLY the heading target: relative to the current heading,
        # bounded magnitude, PERSISTENT sign.
        # ------------------------------------------------------------------
        n = len(env_ids)
        magnitude = torch.empty(n, device=self.device).uniform_(
            self.cfg.heading_delta_min, self.cfg.heading_delta_max
        )

        # Keep the previous turn direction most of the time; flip only with
        # `direction_flip_prob`. This is what turns a sequence of small bounded
        # deltas into ONE LONG SMOOTH ROTATION instead of a jitter that
        # averages to zero. Expected run length = 1 / direction_flip_prob
        # windows before the direction changes.
        #
        # Still unbiased overall: the initial sign is a fair coin and flips are
        # symmetric, so neither direction is preferred. (A hard one-sided yaw
        # command would be unsatisfiable — that was the v9-a collapse.)
        flip = torch.rand(n, device=self.device) < self.cfg.direction_flip_prob
        self._turn_sign[env_ids] = torch.where(
            flip, -self._turn_sign[env_ids], self._turn_sign[env_ids]
        )

        current_heading = self.robot.data.heading_w[env_ids]
        self.heading_target[env_ids] = wrap_to_pi(
            current_heading + self._turn_sign[env_ids] * magnitude
        )



@configclass
class RelativeHeadingVelocityCommandCfg(UniformVelocityCommandCfg):
    """
    Configuration for `RelativeHeadingVelocityCommand`.

    Adds a bounded, relative heading delta. `ranges.heading` is IGNORED by
    this term (the target is derived from the current heading instead), but it
    is inherited from the parent config and left in place for compatibility.

    CONSTRAINT (asserted at construction):
        min(resampling_time_range) >= heading_delta_max / min|ang_vel_z|
    otherwise the target is replaced before the turn completes.
    """

    class_type: type = RelativeHeadingVelocityCommand

    heading_delta_min: float = MISSING
    """Minimum |heading change| per resample, radians. Keep > 0 so every
    resample produces an actual turn instead of an idle window."""

    heading_delta_max: float = MISSING
    """Maximum |heading change| per resample, radians. Bounded by the
    resample period and the ang_vel_z clip — see the constraint above."""

    direction_flip_prob: float = 0.15
    """Probability of REVERSING the turn direction at each resample.

    The direction otherwise PERSISTS, so consecutive bounded deltas chain into
    one long smooth rotation. Expected run length is 1 / direction_flip_prob
    windows.

    Setting this to 0.5 is a fresh coin flip every window, which makes the
    small turns cancel out and the robot barely rotate at all — that was the
    v10c mistake. Setting it to 0.0 would make the direction permanent, which
    is fine for a demo but removes reversal practice entirely.
    """



def make_relative_heading_command(
    *,
    ang_vel_clip: float,
    heading_delta_deg: tuple[float, float],
    lin_vel_x: tuple[float, float],
    resampling_time_range: tuple[float, float] | None = None,
    heading_control_stiffness: float = 1.0,
    direction_flip_prob: float = 0.15,
    settle_margin_s: float = 0.0,
    rel_standing_envs: float = 0.0,
    debug_vis: bool = True,
) -> RelativeHeadingVelocityCommandCfg:
    """
    Build a `RelativeHeadingVelocityCommandCfg` with a SAFE resample period.

    If `resampling_time_range` is omitted it is DERIVED from the turn physics:

        saturated phase = delta_max / ang_vel_clip
        resample        = saturated + settle_margin_s

    Deriving it is strongly preferred over hand-picking a number, because the
    original failure mode was a resample period that silently became too short
    relative to the clip. Derived, the two can never drift apart.

    WHY settle_margin_s DEFAULTS TO 0 (flat / continuous spin demos)
    ----------------------------------------------------------------
    A settle margin is DEAD TIME: the turn has finished, the heading error is
    ~0, so the yaw command is ~0 and the robot just sits there. A 3 s margin on
    a 3.6 s turn is 45% of the demo spent motionless, which reads as "it
    doesn't rotate at all".

    With margin 0 the next bounded target arrives exactly as the previous turn
    finishes saturating, so `omega_cmd` never fully decays and the rotation is
    effectively CONTINUOUS. This is safe here — unlike the original bug — because
    the direction PERSISTS (see `direction_flip_prob`), so an arriving target
    extends the current turn rather than reversing it.

    SLOPE MICRO-TURN / REBALANCE (S25-v3): use settle_margin_s > 0 deliberately.
    On steep slopes the viable skill is: small yaw burst → ω≈0 rebalance →
    another small burst. Continuous saturated ω is what caused roll deaths.

    Args:
        ang_vel_clip: symmetric yaw-rate clip w. Command becomes (-w, +w).
        heading_delta_deg: (min, max) |heading change| per resample, DEGREES.
            Keep these SMALL — the point is a gentle, always-reachable target.
        lin_vel_x: forward velocity command range.
        resampling_time_range: override the derived value. Must satisfy the
            constraint or construction raises.
        heading_control_stiffness: P gain on heading error. Higher keeps the
            command saturated for more of the turn and shortens the near-zero
            exponential tail that reads as idling.
        direction_flip_prob: chance of reversing turn direction per window.
        settle_margin_s: dead time added after the turn completes.
        debug_vis: draw the command arrow.
    """
    delta_min = math.radians(heading_delta_deg[0])
    delta_max = math.radians(heading_delta_deg[1])

    if resampling_time_range is None:
        saturated_s = delta_max / ang_vel_clip
        period = saturated_s + settle_margin_s
        resampling_time_range = (period, period)


    return RelativeHeadingVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=resampling_time_range,
        rel_standing_envs=float(rel_standing_envs),
        rel_heading_envs=1.0,       # every non-standing env is a heading env
        heading_command=True,
        heading_control_stiffness=heading_control_stiffness,
        heading_delta_min=delta_min,
        heading_delta_max=delta_max,
        direction_flip_prob=direction_flip_prob,
        debug_vis=debug_vis,

        ranges=RelativeHeadingVelocityCommandCfg.Ranges(
            lin_vel_x=lin_vel_x,
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-ang_vel_clip, ang_vel_clip),   # SYMMETRIC — never one-sided
            heading=(-math.pi, math.pi),               # unused by this term
        ),
    )
