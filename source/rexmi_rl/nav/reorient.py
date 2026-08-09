# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
ReorientController — in-distribution continuous pivot for model_13345.

Training:
  lin_vel_x = 0.05, ang_vel_z ∈ (−0.08, +0.08)

CRITICAL NAV RULES (from crater demo failures):
  1. NEVER re-latch yaw sign mid-turn (he wrap ±π caused ω flip → flail/tip).
  2. Success uses latched target heading vs body yaw — not live he to moving imm_wp.
  3. Plant with (vx=0.05, ω=0) not (0,0).
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
    HOLD = auto()


@dataclass(frozen=True)
class PulseEnvelope:
    omega_mag: float
    hold_s: float
    yaw_s: float
    settle_s: float
    label: str = ""


ENVELOPE_FLAT = PulseEnvelope(
    omega_mag=0.07, hold_s=1.00, yaw_s=0.0, settle_s=1.50, label="direct13345"
)
ENVELOPE_GENTLE = PulseEnvelope(
    omega_mag=0.07, hold_s=1.00, yaw_s=0.0, settle_s=1.50, label="direct13345"
)
ENVELOPE_STEEP = PulseEnvelope(
    omega_mag=0.05, hold_s=1.40, yaw_s=0.0, settle_s=2.00, label="direct13345-steep"
)

_SLOPE_STEEP = math.tan(math.radians(22.0))
VX_TURN = 0.05


def select_envelope(slope_ahead: float) -> PulseEnvelope:
    if slope_ahead != slope_ahead:
        slope_ahead = 0.0
    if abs(float(slope_ahead)) >= _SLOPE_STEEP:
        return ENVELOPE_STEEP
    return ENVELOPE_GENTLE


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


class ReorientController:
    """
    Continuous slow reorient. Yaw sign latched once at start — never flipped.
    Progress measured by body yaw toward latched target heading.
    """

    def __init__(
        self,
        enter_rad: float = math.radians(60.0),
        exit_rad: float = math.radians(30.0),
        max_pulses: int = 1,
        fixed_envelope: Optional[PulseEnvelope] = None,
        progress_pulses: int = 1,
        min_he_improve_rad: float = math.radians(15.0),
        body_up_abort: float = -0.15,
        vx_turn: float = VX_TURN,
        max_yaw_s: float = 35.0,
        stuck_yaw_s: float = 6.0,  # abort if body barely yaws for this long
        min_body_dyaw_rad: float = math.radians(8.0),
    ):
        self.enter_rad = float(enter_rad)
        self.exit_rad = float(exit_rad)
        self.max_pulses = int(max_pulses)
        self.fixed_envelope = fixed_envelope
        self.progress_pulses = int(progress_pulses)
        self.min_he_improve_rad = float(min_he_improve_rad)
        self.body_up_abort = float(body_up_abort)
        self.vx_turn = float(vx_turn)
        self.max_yaw_s = float(max_yaw_s)
        self.stuck_yaw_s = float(stuck_yaw_s)
        self.min_body_dyaw_rad = float(min_body_dyaw_rad)
        self.cooldown_s = 6.0

        self._phase = ReorientPhase.IDLE
        self._yaw_sign = 1.0
        self._envelope = ENVELOPE_GENTLE
        self._phase_start: float | None = None
        self._yaw_start: float | None = None
        self._pulse_count = 0
        self._active = False
        self._just_done = False
        self._just_failed = False
        self._he_at_start = math.pi
        self._err_best = math.pi
        self._cooldown_until = 0.0
        # Latched geometry (body frame at start)
        self._body_yaw0: float | None = None
        self._target_yaw: float | None = None
        self._body_yaw_last: float | None = None
        self._body_yaw_at_yaw_start: float | None = None
        self._last_progress_t: float | None = None
        self._diag_counter = 0

    @property
    def active(self) -> bool:
        return self._active

    @property
    def phase(self) -> ReorientPhase:
        return self._phase

    @property
    def yaw_sign(self) -> float:
        return self._yaw_sign

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
            print("[Reorient] start ignored — fail cooldown active")
            return

        he = float(heading_error)
        if yaw_sign is None:
            yaw_sign = 1.0 if abs(he) < 1e-6 else math.copysign(1.0, he)
        else:
            yaw_sign = 1.0 if yaw_sign >= 0.0 else -1.0

        # LATCH ONCE — never change until done/fail
        self._yaw_sign = float(yaw_sign)
        self._envelope = (
            self.fixed_envelope
            if self.fixed_envelope is not None
            else select_envelope(slope_ahead)
        )
        self._pulse_count = 0
        self._active = True
        self._just_done = False
        self._just_failed = False
        self._he_at_start = abs(he)
        self._err_best = abs(he)
        self._yaw_start = None
        self._diag_counter = 0

        by = 0.0 if body_yaw is None else float(body_yaw)
        self._body_yaw0 = by
        self._body_yaw_last = by
        self._body_yaw_at_yaw_start = None
        # Target world yaw = current body + heading_error (to waypoint at start)
        self._target_yaw = _wrap(by + he)
        self._last_progress_t = time.monotonic()

        self._enter_phase(ReorientPhase.PLANT)
        print(
            f"[Reorient] START 13345/{self._envelope.label} "
            f"he0={math.degrees(he):+.1f}° "
            f"sign={self._yaw_sign:+.0f} LATCHED "
            f"body={math.degrees(by):+.1f}° "
            f"tgt={math.degrees(self._target_yaw):+.1f}° "
            f"ω=±{self._envelope.omega_mag:.3f} vx={self.vx_turn:.2f} "
            f"plant={self._envelope.hold_s:.1f}s max_yaw={self.max_yaw_s:.0f}s"
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
    ) -> ReorientOutput:
        self._just_done = False
        self._just_failed = False
        if not self._active:
            return self._output(0.0, vx=0.0)

        # Live he only for logging — control uses latched target vs body yaw
        he_live = float(heading_error)
        by = float(body_yaw) if body_yaw is not None else (
            self._body_yaw_last if self._body_yaw_last is not None else 0.0
        )
        self._body_yaw_last = by

        # Error to latched target (stable during pivot)
        if self._target_yaw is not None:
            err = _wrap(self._target_yaw - by)
        else:
            err = he_live
        err_abs = abs(err)
        if err_abs < self._err_best:
            self._err_best = err_abs
            self._last_progress_t = time.monotonic()

        # NOTE: yaw_sign is NEVER updated here (anti ±π thrash)

        if body_up_z is not None and body_up_z < self.body_up_abort:
            if self._phase != ReorientPhase.SETTLE:
                print(f"[Reorient] ABORT inverted up_z={body_up_z:+.2f}")
                return self._finish_failed("inverted")

        now = time.monotonic()
        if self._phase_start is None:
            self._phase_start = now
        elapsed = now - self._phase_start

        # Periodic diagnostic (every ~1s wall if caller logs often)
        self._diag_counter += 1

        if self._phase == ReorientPhase.SETTLE:
            if elapsed >= self._envelope.settle_s:
                return self._finish_success()
            return self._plant_cmd()

        # Success: close enough to latched target
        if err_abs < self.exit_rad and self._phase in (
            ReorientPhase.PLANT, ReorientPhase.YAW, ReorientPhase.HOLD
        ):
            print(
                f"[Reorient] aligned err={math.degrees(err):+.0f}° "
                f"(live_he={math.degrees(he_live):+.0f}°) — SETTLE "
                f"{self._envelope.settle_s:.1f}s"
            )
            self._enter_phase(ReorientPhase.SETTLE)
            return self._plant_cmd()

        if self._phase in (ReorientPhase.PLANT, ReorientPhase.HOLD):
            if elapsed >= self._envelope.hold_s:
                self._enter_phase(ReorientPhase.YAW)
                self._yaw_start = time.monotonic()
                self._body_yaw_at_yaw_start = by
                self._last_progress_t = time.monotonic()
                print(
                    f"[Reorient] YAW continuous ω="
                    f"{self._yaw_sign * self._envelope.omega_mag:+.3f} "
                    f"vx={self.vx_turn:.2f} "
                    f"sign={self._yaw_sign:+.0f} FIXED "
                    f"err={math.degrees(err):+.0f}°"
                )
                return self._yaw_cmd()
            return self._plant_cmd()

        if self._phase == ReorientPhase.YAW:
            yaw_elapsed = (
                time.monotonic() - self._yaw_start
                if self._yaw_start is not None
                else elapsed
            )
            # Body yaw progress since YAW started
            if self._body_yaw_at_yaw_start is not None:
                body_turned = abs(_wrap(by - self._body_yaw_at_yaw_start))
            else:
                body_turned = 0.0

            # Stuck: little body rotation for stuck_yaw_s
            if (
                self._last_progress_t is not None
                and (now - self._last_progress_t) >= self.stuck_yaw_s
                and yaw_elapsed >= self.stuck_yaw_s
            ):
                if body_turned < self.min_body_dyaw_rad:
                    print(
                        f"[Reorient] STUCK body_turned={math.degrees(body_turned):.1f}° "
                        f"in {yaw_elapsed:.1f}s"
                    )
                    return self._finish_failed("stuck_no_yaw")

            if yaw_elapsed >= 0.70 * self.max_yaw_s:
                improved = (self._he_at_start - self._err_best) >= self.min_he_improve_rad
                if not improved and body_turned < self.min_he_improve_rad:
                    return self._finish_failed("no_progress")

            if yaw_elapsed >= self.max_yaw_s:
                if err_abs < self.exit_rad * 1.5:
                    self._enter_phase(ReorientPhase.SETTLE)
                    return self._plant_cmd()
                return self._finish_failed("timeout")

            return self._yaw_cmd()

        return self._plant_cmd()

    def status_str(self) -> str:
        if not self._active:
            return "IDLE"
        t = 0.0
        if self._phase == ReorientPhase.YAW and self._yaw_start is not None:
            t = time.monotonic() - self._yaw_start
        elif self._phase_start is not None:
            t = time.monotonic() - self._phase_start
        err_s = ""
        if self._target_yaw is not None and self._body_yaw_last is not None:
            e = _wrap(self._target_yaw - self._body_yaw_last)
            err_s = f" err={math.degrees(e):+.0f}°"
        return (
            f"{self._phase.name}[{self._envelope.label}] "
            f"t={t:.1f}s "
            f"ω={self._yaw_sign * self._envelope.omega_mag:+.3f} "
            f"sign={self._yaw_sign:+.0f}{err_s}"
        )

    def diag_str(
        self,
        heading_error: float,
        body_yaw: float,
        body_up_z: float,
        speed: float,
        omega_body: float = float("nan"),
    ) -> str:
        """One-line diagnostic for nav logs."""
        err = (
            _wrap(self._target_yaw - body_yaw)
            if self._target_yaw is not None
            else heading_error
        )
        return (
            f"phase={self._phase.name} sign={self._yaw_sign:+.0f} "
            f"ω_cmd={self._yaw_sign * self._envelope.omega_mag:+.3f} "
            f"ω_body={omega_body:+.3f} "
            f"vx={self.vx_turn:.2f} "
            f"he_live={math.degrees(heading_error):+.1f}° "
            f"err_tgt={math.degrees(err):+.1f}° "
            f"body={math.degrees(body_yaw):+.1f}° "
            f"tgt={math.degrees(self._target_yaw or 0):+.1f}° "
            f"up_z={body_up_z:+.2f} v={speed:.2f} "
            f"best_err={math.degrees(self._err_best):.0f}°"
        )

    def _enter_phase(self, phase: ReorientPhase) -> None:
        self._phase = phase
        self._phase_start = time.monotonic()

    def _plant_cmd(self) -> ReorientOutput:
        return self._output(0.0, vx=self.vx_turn)

    def _yaw_cmd(self) -> ReorientOutput:
        return self._output(
            self._yaw_sign * self._envelope.omega_mag,
            vx=self.vx_turn,
        )

    def _finish_success(self) -> ReorientOutput:
        print(
            f"[Reorient] DONE 13345/{self._envelope.label} "
            f"best_err={math.degrees(self._err_best):.0f}° "
            f"sign_held={self._yaw_sign:+.0f}"
        )
        self._active = False
        self._phase = ReorientPhase.IDLE
        self._phase_start = None
        self._yaw_start = None
        self._just_done = True
        return self._output(0.0, vx=0.0, done=True, failed=False)

    def _finish_failed(self, reason: str = "") -> ReorientOutput:
        print(
            f"[Reorient] FAILED 13345/{self._envelope.label} ({reason}) "
            f"start_he={math.degrees(self._he_at_start):.0f}° "
            f"best_err={math.degrees(self._err_best):.0f}° "
            f"sign_held={self._yaw_sign:+.0f} "
            f"— cooldown {self.cooldown_s:.0f}s"
        )
        self._active = False
        self._phase = ReorientPhase.IDLE
        self._phase_start = None
        self._yaw_start = None
        self._just_failed = True
        self._cooldown_until = time.monotonic() + self.cooldown_s
        return self._output(0.0, vx=0.0, done=False, failed=True)

    def _output(
        self,
        omega: float,
        vx: float | None = None,
        done: bool | None = None,
        failed: bool | None = None,
    ) -> ReorientOutput:
        return ReorientOutput(
            vx=self.vx_turn if vx is None else float(vx),
            vy=0.0,
            omega=float(omega),
            active=self._active,
            phase=self._phase,
            pulse_count=self._pulse_count,
            envelope_label=self._envelope.label,
            done=self._just_done if done is None else done,
            failed=self._just_failed if failed is None else failed,
        )
