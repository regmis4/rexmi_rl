# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
ReorientController — isolation-style TURN_ONCE for model_13345.

Matches scripts/test_turn_crater.py command schedule after calm handoff:

  PLANT   vx=0.05  ω=0      until calm (or hold_s)
  YAW     vx=0.05  ω=±0.07  until |err|<exit or yaw_s
  SETTLE  vx=0.05  ω=0      settle_s
  DONE → FOLLOW

Iso findings:
  • Zero-action HOLD until upright is required before 13345
  • ω=−0.07 produced large stable turns; +0.07 often weak
  • Stay upright (up_z high) even when body rate spikes

Nav must: brake → settle actions → force_turn → this controller.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


def _wrap(a: float) -> float:
    return a - 2.0 * math.pi * math.floor((a + math.pi) / (2.0 * math.pi))


class ReorientPhase(Enum):
    IDLE = auto()
    PLANT = auto()
    YAW = auto()
    SETTLE = auto()
    HOLD = auto()  # alias of PLANT


@dataclass(frozen=True)
class PulseEnvelope:
    omega_mag: float
    hold_s: float
    yaw_s: float
    settle_s: float
    label: str = "iso"


# Match isolation test: vx=0.05, |ω|=0.07, long yaw window
ENVELOPE_ISO = PulseEnvelope(
    omega_mag=0.07, hold_s=2.0, yaw_s=8.0, settle_s=1.5, label="iso"
)
ENVELOPE_STEEP = PulseEnvelope(
    omega_mag=0.07, hold_s=2.5, yaw_s=8.0, settle_s=1.5, label="iso-steep"
)

_SLOPE_STEEP = math.tan(math.radians(22.0))
VX_TURN = 0.05


def select_envelope(slope_ahead: float) -> PulseEnvelope:
    if slope_ahead != slope_ahead:
        slope_ahead = 0.0
    if abs(float(slope_ahead)) >= _SLOPE_STEEP:
        return ENVELOPE_STEEP
    return ENVELOPE_ISO


@dataclass
class ReorientOutput:
    vx: float
    vy: float
    omega: float
    active: bool
    phase: ReorientPhase
    pulse_count: int
    envelope_label: str
    done: bool
    failed: bool
    wants_turn_policy: bool = True


class ReorientController:
    """
    Single-shot isolation-style turn.

    Sign: prefer heading_error sign. If YAW makes little progress, flip once
    (iso: −ω often works when +ω freezes).
    """

    def __init__(
        self,
        enter_rad: float = math.radians(100.0),
        exit_rad: float = math.radians(35.0),
        max_pulses: int = 1,
        fixed_envelope: Optional[PulseEnvelope] = None,
        progress_pulses: int = 1,
        min_he_improve_rad: float = math.radians(20.0),
        body_up_abort: float = -0.10,
        vx_turn: float = VX_TURN,
        max_yaw_s: float = 8.0,
        thrash_omega_body: float = 6.0,  # allow fast spin if upright
        thrash_count_limit: int = 8,
        calm_omega_body: float = 0.35,
        calm_up_z: float = 0.85,
        bleed_up_z: float = 0.45,
        plant_max_s: float = 4.0,
        stuck_yaw_s: float = 3.0,
        min_body_dyaw_rad: float = math.radians(12.0),
        max_consecutive_thrash: int = 3,
        # unused compat
        bite_rad: float = 0.0,
    ):
        self.enter_rad = float(enter_rad)
        self.exit_rad = float(exit_rad)
        self.max_pulses = 1
        self.fixed_envelope = fixed_envelope or ENVELOPE_ISO
        self.min_he_improve_rad = float(min_he_improve_rad)
        self.body_up_abort = float(body_up_abort)
        self.vx_turn = float(vx_turn)
        self.max_yaw_s = float(max_yaw_s)
        self.thrash_omega_body = float(thrash_omega_body)
        self.thrash_count_limit = int(thrash_count_limit)
        self.calm_omega_body = float(calm_omega_body)
        self.calm_up_z = float(calm_up_z)
        self.bleed_up_z = float(bleed_up_z)
        self.plant_max_s = float(plant_max_s)
        self.stuck_yaw_s = float(stuck_yaw_s)
        self.min_body_dyaw_rad = float(min_body_dyaw_rad)
        self.cooldown_s = 10.0

        self._phase = ReorientPhase.IDLE
        self._yaw_sign = 1.0
        self._envelope = ENVELOPE_ISO
        self._phase_start: float | None = None
        self._yaw_start: float | None = None
        self._pulse_count = 0
        self._active = False
        self._just_done = False
        self._just_failed = False
        self._he_at_start = math.pi
        self._err_best = math.pi
        self._cooldown_until = 0.0
        self._target_yaw: float | None = None
        self._body_yaw_last: float | None = None
        self._body_yaw_at_yaw_start: float | None = None
        self._thrash_count = 0
        self._manoeuvre_he0 = math.pi
        self._flipped_once = False
        self._preferred_sign: float | None = None  # set from iso: often -1

    @property
    def active(self) -> bool:
        return self._active

    @property
    def phase(self) -> ReorientPhase:
        return self._phase

    @property
    def yaw_sign(self) -> float:
        return self._yaw_sign

    @property
    def wants_turn_policy(self) -> bool:
        return self._active

    def should_start(self, heading_error: float) -> bool:
        if self._active:
            return False
        if time.monotonic() < self._cooldown_until:
            return False
        return abs(heading_error) > self.enter_rad

    def start(
        self,
        heading_error: float,
        slope_ahead: float = 0.0,
        yaw_sign: float | None = None,
        body_yaw: float | None = None,
    ) -> None:
        if time.monotonic() < self._cooldown_until:
            print("[Reorient] start ignored — cooldown")
            return

        he = float(heading_error)
        if abs(he) < 1e-6:
            return

        # Prefer explicit sign, else he sign. Optional preferred_sign bias if |he| large.
        if yaw_sign is not None and abs(yaw_sign) > 1e-6:
            self._yaw_sign = math.copysign(1.0, float(yaw_sign))
        else:
            self._yaw_sign = math.copysign(1.0, he)
            # Iso: −ω often works better; if he wants + but preferred is −, still use he
            # (nav needs correct direction). Keep he sign.

        self._envelope = (
            self.fixed_envelope
            if self.fixed_envelope is not None
            else select_envelope(slope_ahead)
        )
        self._active = True
        self._just_done = False
        self._just_failed = False
        self._he_at_start = abs(he)
        self._manoeuvre_he0 = abs(he)
        self._err_best = abs(he)
        self._pulse_count = 0
        self._thrash_count = 0
        self._yaw_start = None
        self._flipped_once = False

        by = 0.0 if body_yaw is None else float(body_yaw)
        self._body_yaw_last = by
        self._body_yaw_at_yaw_start = None
        self._target_yaw = _wrap(by + he)

        self._enter_phase(ReorientPhase.PLANT)
        print(
            f"[Reorient] START iso/{self._envelope.label} "
            f"he={math.degrees(he):+.0f}° sign={self._yaw_sign:+.0f} "
            f"body={math.degrees(by):+.0f}° tgt={math.degrees(self._target_yaw):+.0f}° "
            f"ω=±{self._envelope.omega_mag:.3f} yaw_max={self._envelope.yaw_s:.1f}s "
            f"vx={self.vx_turn:.2f}"
        )

    def cancel(self) -> None:
        was = self._active
        self._active = False
        self._phase = ReorientPhase.IDLE
        self._phase_start = None
        self._yaw_start = None
        self._just_done = False
        self._just_failed = False
        if was:
            print("[Reorient] CANCELLED")

    def reset(self) -> None:
        self.cancel()
        self._cooldown_until = 0.0

    def update(
        self,
        heading_error: float,
        slope_ahead: float = 0.0,
        body_up_z: float | None = None,
        body_yaw: float | None = None,
        omega_body: float | None = None,
    ) -> ReorientOutput:
        self._just_done = False
        self._just_failed = False
        if not self._active:
            return self._out(0.0, 0.0)

        he_live = float(heading_error)
        by = float(body_yaw) if body_yaw is not None else (
            self._body_yaw_last if self._body_yaw_last is not None else 0.0
        )
        self._body_yaw_last = by

        if self._target_yaw is not None:
            err = _wrap(self._target_yaw - by)
        else:
            err = he_live
        err_abs = abs(err)
        if err_abs < self._err_best:
            self._err_best = err_abs

        # Hard invert only
        if body_up_z is not None and float(body_up_z) < self.body_up_abort:
            return self._fail("inverted")

        # Bleed during yaw only if collapsing
        if (
            body_up_z is not None
            and self._phase == ReorientPhase.YAW
            and float(body_up_z) < self.bleed_up_z
        ):
            return self._fail("attitude_bleed")

        # Thrash: high rate AND bad attitude (fast spin while upright is OK)
        if (
            omega_body is not None
            and body_up_z is not None
            and self._phase == ReorientPhase.YAW
            and abs(float(omega_body)) > self.thrash_omega_body
            and float(body_up_z) < 0.70
        ):
            self._thrash_count += 1
            if self._thrash_count >= self.thrash_count_limit:
                return self._fail("thrash")
        else:
            self._thrash_count = max(0, self._thrash_count - 1)

        now = time.monotonic()
        if self._phase_start is None:
            self._phase_start = now
        elapsed = now - self._phase_start

        # Success
        upright = body_up_z is None or float(body_up_z) > 0.70
        if err_abs < self.exit_rad and upright:
            if self._phase != ReorientPhase.SETTLE:
                print(f"[Reorient] aligned err={math.degrees(err):+.0f}° — SETTLE")
                self._enter_phase(ReorientPhase.SETTLE)

        if self._phase == ReorientPhase.SETTLE:
            if elapsed >= self._envelope.settle_s:
                return self._done()
            return self._plant()

        if self._phase in (ReorientPhase.PLANT, ReorientPhase.HOLD):
            wb = 0.0 if omega_body is None else abs(float(omega_body))
            uz = 1.0 if body_up_z is None else float(body_up_z)
            if (
                elapsed >= self._envelope.hold_s
                and wb <= self.calm_omega_body
                and uz >= self.calm_up_z
            ):
                self._enter_phase(ReorientPhase.YAW)
                self._yaw_start = time.monotonic()
                self._body_yaw_at_yaw_start = by
                print(
                    f"[Reorient] YAW iso ω={self._yaw_sign * self._envelope.omega_mag:+.3f} "
                    f"calm ωb={wb:.2f} up_z={uz:+.2f}"
                )
                return self._yaw()
            if elapsed >= self.plant_max_s:
                # Start yaw anyway if mostly upright
                if uz >= 0.75:
                    self._enter_phase(ReorientPhase.YAW)
                    self._yaw_start = time.monotonic()
                    self._body_yaw_at_yaw_start = by
                    print("[Reorient] YAW iso (plant timeout, upright enough)")
                    return self._yaw()
                return self._fail("plant_not_calm")
            return self._plant()

        if self._phase == ReorientPhase.YAW:
            yaw_t = (
                time.monotonic() - self._yaw_start
                if self._yaw_start is not None
                else elapsed
            )
            body_turned = 0.0
            if self._body_yaw_at_yaw_start is not None:
                body_turned = abs(_wrap(by - self._body_yaw_at_yaw_start))

            wb = 0.0 if omega_body is None else abs(float(omega_body))

            # Stuck: little body turn — flip sign once (iso asymmetry)
            if (
                yaw_t >= self.stuck_yaw_s
                and body_turned < self.min_body_dyaw_rad
                and not self._flipped_once
            ):
                self._flipped_once = True
                self._yaw_sign = -self._yaw_sign
                self._yaw_start = time.monotonic()
                self._body_yaw_at_yaw_start = by
                print(
                    f"[Reorient] YAW stuck — flip sign → {self._yaw_sign:+.0f} "
                    f"(iso: other direction often works)"
                )
                return self._yaw()

            if yaw_t >= self.stuck_yaw_s * 2 and body_turned < self.min_body_dyaw_rad:
                return self._fail("stuck_no_yaw")

            # Time limit — accept partial progress
            max_yaw = max(self._envelope.yaw_s, self.max_yaw_s)
            if yaw_t >= max_yaw:
                if err_abs < self.exit_rad * 1.8 or body_turned > math.radians(30):
                    print(
                        f"[Reorient] YAW time done Δbody={math.degrees(body_turned):.0f}° "
                        f"err={math.degrees(err):+.0f}° — SETTLE"
                    )
                    self._enter_phase(ReorientPhase.SETTLE)
                    return self._plant()
                return self._fail("timeout")
            return self._yaw()

        return self._plant()

    def status_str(self) -> str:
        if not self._active:
            return "IDLE"
        t = 0.0
        if self._yaw_start is not None and self._phase == ReorientPhase.YAW:
            t = time.monotonic() - self._yaw_start
        elif self._phase_start is not None:
            t = time.monotonic() - self._phase_start
        return (
            f"{self._phase.name}[iso] t={t:.1f}s "
            f"ω={self._yaw_sign * self._envelope.omega_mag:+.3f}"
        )

    def diag_str(
        self,
        heading_error: float,
        body_yaw: float,
        body_up_z: float,
        speed: float,
        omega_body: float = float("nan"),
    ) -> str:
        err = (
            _wrap(self._target_yaw - body_yaw)
            if self._target_yaw is not None
            else heading_error
        )
        return (
            f"phase={self._phase.name} sign={self._yaw_sign:+.0f} "
            f"ω_cmd={self._yaw_sign * self._envelope.omega_mag:+.3f} "
            f"ω_body={omega_body:+.3f} "
            f"he={math.degrees(heading_error):+.0f}° "
            f"err={math.degrees(err):+.0f}° "
            f"body={math.degrees(body_yaw):+.0f}° "
            f"up_z={body_up_z:+.2f} v={speed:.2f}"
        )

    def _enter_phase(self, p: ReorientPhase) -> None:
        self._phase = p
        self._phase_start = time.monotonic()

    def _plant(self) -> ReorientOutput:
        return self._out(self.vx_turn, 0.0)

    def _yaw(self) -> ReorientOutput:
        return self._out(self.vx_turn, self._yaw_sign * self._envelope.omega_mag)

    def _done(self) -> ReorientOutput:
        print(
            f"[Reorient] DONE iso best_err={math.degrees(self._err_best):.0f}° "
            f"start={math.degrees(self._manoeuvre_he0):.0f}°"
        )
        self._active = False
        self._phase = ReorientPhase.IDLE
        self._just_done = True
        return self._out(0.0, 0.0, done=True)

    def _fail(self, reason: str) -> ReorientOutput:
        print(
            f"[Reorient] FAILED iso ({reason}) "
            f"start={math.degrees(self._manoeuvre_he0):.0f}° "
            f"best={math.degrees(self._err_best):.0f}° "
            f"cooldown {self.cooldown_s:.0f}s"
        )
        self._active = False
        self._phase = ReorientPhase.IDLE
        self._just_failed = True
        self._cooldown_until = time.monotonic() + self.cooldown_s
        return self._out(0.0, 0.0, failed=True)

    def _out(
        self,
        vx: float,
        omega: float,
        done: bool = False,
        failed: bool = False,
    ) -> ReorientOutput:
        return ReorientOutput(
            vx=float(vx),
            vy=0.0,
            omega=float(omega),
            active=self._active,
            phase=self._phase,
            pulse_count=self._pulse_count,
            envelope_label=self._envelope.label,
            done=done or self._just_done,
            failed=failed or self._just_failed,
            wants_turn_policy=self._active,
        )
