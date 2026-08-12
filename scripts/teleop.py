#!/usr/bin/env python3
# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Manual teleop for crater policy characterization.

Spawn Go2W on crater terrain, pick a policy (rough / rocky_slope / turn),
and drive with a remote.  No autonomous nav — navigate.py is untouched.

Controls are MOMENTARY:
  hold Forward/Back/Left/Right  → command applied
  release                       → stop (vx=0, ω=0)

Same for keyboard WASD / arrows.  Space always stops.

Usage
-----
  conda activate env_isaacsim
  cd /home/susan/rexmi_rl

  python scripts/teleop.py \\
      --task RexmiRl-Go2w-Crater-Bowl-RockySlope-Play-v0 \\
      --ckpt_rough logs/rsl_rl/go2w_velocity_rough/2026-06-14_20-03-41/model_8996.pt \\
      --ckpt_rocky logs/rsl_rl/go2w_velocity_rocky_slope/2026-06-30_09-31-48/model_13994.pt \\
      --ckpt_turn  logs/rsl_rl/go2w_velocity_slope_turn/2026-07-27_20-54-25/model_13345.pt \\
      --spawn_preset floor

  # Start on mid wall / exterior rim
  python scripts/teleop.py ... --spawn_preset mid_slope
  python scripts/teleop.py ... --spawn_preset rim_out

  # Stdin-only (no Tk window)
  python scripts/teleop.py ... --no_gui
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SOURCE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "source"
)
if _SOURCE_DIR not in sys.path:
    sys.path.insert(0, _SOURCE_DIR)


# ---------------------------------------------------------------------------
# Policy defaults (train / nav matched)
# ---------------------------------------------------------------------------

POLICY_DEFAULTS = {
    "rough": {
        "vx": 0.45,
        "omega": 0.40,
        "ang_range": (-1.0, 1.0),
        "lin_x_range": (-0.5, 1.0),
    },
    "rocky_slope": {
        "vx": 0.40,
        "omega": 0.35,
        "ang_range": (-1.0, 1.0),
        "lin_x_range": (-0.5, 1.0),
    },
    "turn": {
        # model_13345 Language A band (isolation / reorient)
        "vx": 0.05,
        "omega": 0.07,
        "ang_range": (-0.10, 0.10),
        "lin_x_range": (0.0, 0.15),
    },
}

SPAWN_PRESETS = {
    "floor":     dict(x=0.0,  y=0.0, z=1.15),
    "mid_slope": dict(x=6.5,  y=0.0, z=2.40),
    "rim_out":   dict(x=13.0, y=0.0, z=4.50),
}


# ---------------------------------------------------------------------------
# Shared teleop state (sim thread reads; UI / stdin write)
# ---------------------------------------------------------------------------

@dataclass
class TeleopState:
    """Thread-safe command + policy selection."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    policy_name: str = "rocky_slope"
    # Magnitude fields (edited in UI; applied while a direction is held)
    vx_mag: float = 0.40
    omega_mag: float = 0.35
    # Active held directions (momentary)
    hold_forward: bool = False
    hold_back: bool = False
    hold_left: bool = False
    hold_right: bool = False
    # Live status for UI
    status: str = "boot"
    pose_str: str = ""
    quit: bool = False

    def set_policy(self, name: str) -> None:
        name = name.strip().lower().replace("-", "_")
        if name in ("rocky", "rocky_slope"):
            name = "rocky_slope"
        if name not in POLICY_DEFAULTS:
            return
        d = POLICY_DEFAULTS[name]
        with self.lock:
            self.policy_name = name
            self.vx_mag = float(d["vx"])
            self.omega_mag = float(d["omega"])
            # Drop holds on switch so we don't surprise-drive with new gains
            self.hold_forward = self.hold_back = False
            self.hold_left = self.hold_right = False

    def set_mags(self, vx: Optional[float] = None, omega: Optional[float] = None) -> None:
        with self.lock:
            if vx is not None:
                self.vx_mag = float(vx)
            if omega is not None:
                self.omega_mag = float(omega)

    def set_hold(self, direction: str, pressed: bool) -> None:
        """direction in forward|back|left|right."""
        with self.lock:
            if direction == "forward":
                self.hold_forward = pressed
                if pressed:
                    self.hold_back = False
            elif direction == "back":
                self.hold_back = pressed
                if pressed:
                    self.hold_forward = False
            elif direction == "left":
                self.hold_left = pressed
                if pressed:
                    self.hold_right = False
            elif direction == "right":
                self.hold_right = pressed
                if pressed:
                    self.hold_left = False

    def stop_all(self) -> None:
        with self.lock:
            self.hold_forward = self.hold_back = False
            self.hold_left = self.hold_right = False

    def command(self) -> tuple[str, float, float]:
        """Return (policy_name, vx, omega) for this sim step."""
        with self.lock:
            name = self.policy_name
            vx_m = self.vx_mag
            om_m = self.omega_mag
            fx = self.hold_forward
            bk = self.hold_back
            lf = self.hold_left
            rt = self.hold_right

        vx = 0.0
        omega = 0.0
        if fx:
            vx = +vx_m
        elif bk:
            vx = -vx_m
        if lf:
            omega = +om_m  # CCW / left
        elif rt:
            omega = -om_m  # CW / right

        # Turn policy (13345) was trained with plant vx≈0.05 during yaw.
        # If user only holds Left/Right, still apply vx_mag as plant so the
        # spin stays in-distribution. Explicit Forward/Back already set vx.
        if name == "turn" and omega != 0.0 and not fx and not bk:
            vx = +vx_m

        return name, vx, omega


    def snapshot_mags(self) -> tuple[str, float, float]:
        with self.lock:
            return self.policy_name, self.vx_mag, self.omega_mag


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="REXMI manual teleop (crater)")
    p.add_argument(
        "--task",
        default="RexmiRl-Go2w-Crater-Bowl-RockySlope-Play-v0",
        help="Isaac Lab gym task",
    )
    p.add_argument(
        "--ckpt_rough",
        default="logs/rsl_rl/go2w_velocity_rough/2026-06-14_20-03-41/model_8996.pt",
    )
    p.add_argument(
        "--ckpt_rocky",
        default="logs/rsl_rl/go2w_velocity_rocky_slope/2026-06-30_09-31-48/model_13994.pt",
    )
    p.add_argument(
        "--ckpt_turn",
        default=(
            "logs/rsl_rl/go2w_velocity_slope_turn/"
            "2026-07-27_20-54-25/model_13345.pt"
        ),
    )
    p.add_argument(
        "--policy",
        choices=["rough", "rocky_slope", "turn"],
        default="rocky_slope",
        help="Initial policy",
    )
    p.add_argument(
        "--spawn_preset",
        choices=list(SPAWN_PRESETS.keys()),
        default="rim_out",
        help="Spawn location on crater",
    )
    p.add_argument("--spawn_x", type=float, default=None)
    p.add_argument("--spawn_y", type=float, default=None)
    p.add_argument("--spawn_z", type=float, default=None)
    p.add_argument("--spawn_yaw", type=float, default=math.pi, help="radians")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max_steps", type=int, default=100000)
    p.add_argument("--settle_s", type=float, default=2.0,
                   help="Zero-action settle after reset (s)")
    p.add_argument("--no_gui", action="store_true",
                   help="Stdin remote only (no Tk window)")
    p.add_argument("--log_file", default=None,
                   help="CSV path (default logs/nav/teleop_<ts>.csv)")
    p.add_argument("--headless", action="store_true",
                   help="Isaac Sim headless (still need remote)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Policy loader (same pattern as navigate / test_turn_crater)
# ---------------------------------------------------------------------------

def _load_policy(checkpoint_path: str, device: str) -> Callable:
    import torch
    from rsl_rl.modules import ActorCritic

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    msd = ckpt["model_state_dict"]
    obs_dim = msd["actor.0.weight"].shape[1]
    num_actions = msd["actor.6.weight"].shape[0]
    hidden_dims = [
        msd["actor.0.bias"].shape[0],
        msd["actor.2.bias"].shape[0],
        msd["actor.4.bias"].shape[0],
    ]
    ac = ActorCritic(
        num_actor_obs=obs_dim,
        num_critic_obs=obs_dim,
        num_actions=num_actions,
        actor_hidden_dims=hidden_dims,
        critic_hidden_dims=hidden_dims,
    ).to(device)
    ac.load_state_dict(msd)
    ac.eval()

    obs_mean = obs_std = None
    if "obs_norm_state_dict" in ckpt:
        nd = ckpt["obs_norm_state_dict"]
        if "_mean" in nd and "_std" in nd:
            obs_mean = nd["_mean"].to(device)
            obs_std = nd["_std"].to(device).clamp(min=1e-6)
            print(f"[teleop] obs normalizer loaded shape={tuple(obs_mean.shape)}")

    def policy(obs):
        with torch.inference_mode():
            o = obs[..., :obs_dim]
            if obs_mean is not None:
                o = (o - obs_mean) / obs_std
            return ac.act_inference(o)

    print(
        f"[teleop] Loaded {os.path.basename(checkpoint_path)} "
        f"obs={obs_dim} act={num_actions} hidden={hidden_dims}"
    )
    return policy


def _inject(env, vx: float, omega: float, env_idx: int = 0) -> None:
    try:
        cmd = env.unwrapped.command_manager.get_command("base_velocity")
        cmd[env_idx, 0] = float(vx)
        cmd[env_idx, 1] = 0.0
        cmd[env_idx, 2] = float(omega)
    except Exception as e:
        print(f"[teleop] inject failed: {e}")


def _patch_command_ranges(env, policy_name: str) -> None:
    d = POLICY_DEFAULTS[policy_name]
    try:
        bv = env.unwrapped.command_manager.cfg.base_velocity
        bv.ranges.ang_vel_z = d["ang_range"]
        if hasattr(bv.ranges, "lin_vel_x"):
            bv.ranges.lin_vel_x = d["lin_x_range"]
        # Disable heading P-controller so injected ω is used as-is
        if hasattr(bv, "heading_command"):
            bv.heading_command = False
        if hasattr(bv, "rel_heading_envs"):
            bv.rel_heading_envs = 0.0
        print(
            f"[teleop] command ranges for {policy_name}: "
            f"ang={d['ang_range']} lin_x={d['lin_x_range']}"
        )
    except Exception as e:
        print(f"[teleop] WARNING range patch: {e}")


def _yaw(robot, env_idx: int = 0) -> float:
    try:
        quat = robot.data.root_quat_w[env_idx]
        w, qx, qy, qz = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
        return math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    except Exception:
        return float("nan")


def _body_up_z(robot, env_idx: int = 0) -> float:
    try:
        quat = robot.data.root_quat_w[env_idx]
        w, qx, qy, qz = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
        return 1.0 - 2.0 * (qx * qx + qy * qy)
    except Exception:
        return float("nan")


def _speed(robot, env_idx: int = 0) -> float:
    try:
        v = robot.data.root_lin_vel_w[env_idx]
        return float((v[0] ** 2 + v[1] ** 2) ** 0.5)
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Tk remote (momentary buttons)
# ---------------------------------------------------------------------------

def _start_tk_remote(state: TeleopState) -> Optional[threading.Thread]:
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception as e:
        print(f"[teleop] Tk unavailable ({e}) — use stdin remote")
        return None

    def run():
        root = tk.Tk()
        root.title("REXMI Teleop Remote")
        root.geometry("420x480")
        root.configure(bg="#1e1e1e")

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TLabel", background="#1e1e1e", foreground="#e0e0e0")
        style.configure("TFrame", background="#1e1e1e")
        style.configure("TButton", padding=8)
        style.configure("TCombobox", padding=4)

        frm = ttk.Frame(root, padding=12)
        frm.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frm, text="Policy", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        pol_var = tk.StringVar(value=state.policy_name)
        pol_box = ttk.Combobox(
            frm,
            textvariable=pol_var,
            values=["rough", "rocky_slope", "turn"],
            state="readonly",
            width=18,
        )
        pol_box.pack(anchor="w", pady=(0, 8))

        mag_frm = ttk.Frame(frm)
        mag_frm.pack(fill=tk.X, pady=4)
        ttk.Label(mag_frm, text="vx (m/s)").grid(row=0, column=0, sticky="w")
        ttk.Label(mag_frm, text="ω (rad/s)").grid(row=0, column=1, sticky="w", padx=(16, 0))
        vx_var = tk.StringVar(value=f"{state.vx_mag:.3f}")
        om_var = tk.StringVar(value=f"{state.omega_mag:.3f}")
        vx_entry = ttk.Entry(mag_frm, textvariable=vx_var, width=10)
        om_entry = ttk.Entry(mag_frm, textvariable=om_var, width=10)
        vx_entry.grid(row=1, column=0, sticky="w")
        om_entry.grid(row=1, column=1, sticky="w", padx=(16, 0))

        def apply_mags(*_a):
            try:
                vx = float(vx_var.get())
            except ValueError:
                vx = None
            try:
                om = float(om_var.get())
            except ValueError:
                om = None
            state.set_mags(vx, om)

        vx_entry.bind("<Return>", apply_mags)
        om_entry.bind("<Return>", apply_mags)
        vx_entry.bind("<FocusOut>", apply_mags)
        om_entry.bind("<FocusOut>", apply_mags)

        def on_policy(_evt=None):
            name = pol_var.get()
            state.set_policy(name)
            _, vx, om = state.snapshot_mags()
            vx_var.set(f"{vx:.3f}")
            om_var.set(f"{om:.3f}")

        pol_box.bind("<<ComboboxSelected>>", on_policy)

        ttk.Label(
            frm,
            text="Hold buttons to drive — release to stop",
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(12, 4))

        pad = ttk.Frame(frm)
        pad.pack(pady=8)

        def bind_momentary(btn, direction: str):
            btn.bind("<ButtonPress-1>", lambda e: (apply_mags(), state.set_hold(direction, True)))
            btn.bind("<ButtonRelease-1>", lambda e: state.set_hold(direction, False))
            # If pointer leaves while held, still stop
            btn.bind("<Leave>", lambda e: state.set_hold(direction, False))

        btn_f = ttk.Button(pad, text="▲  Forward (W)", width=18)
        btn_f.grid(row=0, column=1, padx=4, pady=4)
        bind_momentary(btn_f, "forward")

        btn_l = ttk.Button(pad, text="◀ Left (A)", width=14)
        btn_l.grid(row=1, column=0, padx=4, pady=4)
        bind_momentary(btn_l, "left")

        btn_s = ttk.Button(pad, text="■ STOP", width=10, command=state.stop_all)
        btn_s.grid(row=1, column=1, padx=4, pady=4)

        btn_r = ttk.Button(pad, text="Right (D) ▶", width=14)
        btn_r.grid(row=1, column=2, padx=4, pady=4)
        bind_momentary(btn_r, "right")

        btn_b = ttk.Button(pad, text="▼  Back (S)", width=18)
        btn_b.grid(row=2, column=1, padx=4, pady=4)
        bind_momentary(btn_b, "back")

        help_txt = (
            "Keyboard (focus this window):\n"
            "  W/S or ↑/↓  forward / back\n"
            "  A/D or ←/→  left / right turn\n"
            "  Space       stop\n"
            "  1/2/3       rough / rocky / turn\n"
            "\n"
            "Turn policy defaults: vx=0.05  |ω|=0.07\n"
            "Hold Left/Right to spin; release stops."
        )
        ttk.Label(frm, text=help_txt, justify=tk.LEFT).pack(anchor="w", pady=(8, 4))

        status_var = tk.StringVar(value="ready")
        ttk.Label(frm, textvariable=status_var, wraplength=390).pack(anchor="w", pady=8)

        # Keyboard momentary
        key_map = {
            "w": "forward", "Up": "forward",
            "s": "back", "Down": "back",
            "a": "left", "Left": "left",
            "d": "right", "Right": "right",
        }

        def on_key_press(event):
            k = event.keysym
            if k in ("space", "Space"):
                state.stop_all()
                return
            if k == "1":
                pol_var.set("rough")
                on_policy()
                return
            if k == "2":
                pol_var.set("rocky_slope")
                on_policy()
                return
            if k == "3":
                pol_var.set("turn")
                on_policy()
                return
            direction = key_map.get(k) or key_map.get(k.lower() if len(k) == 1 else k)
            if direction:
                apply_mags()
                state.set_hold(direction, True)

        def on_key_release(event):
            k = event.keysym
            direction = key_map.get(k) or key_map.get(k.lower() if len(k) == 1 else k)
            if direction:
                state.set_hold(direction, False)

        root.bind("<KeyPress>", on_key_press)
        root.bind("<KeyRelease>", on_key_release)

        def tick():
            with state.lock:
                if state.quit:
                    root.destroy()
                    return
                st = state.status
                pose = state.pose_str
                name = state.policy_name
                holds = []
                if state.hold_forward:
                    holds.append("F")
                if state.hold_back:
                    holds.append("B")
                if state.hold_left:
                    holds.append("L")
                if state.hold_right:
                    holds.append("R")
                hold_s = "+".join(holds) if holds else "idle"
            status_var.set(f"[{name}] {hold_s}  |  {st}\n{pose}")
            root.after(100, tick)

        def on_close():
            state.stop_all()
            with state.lock:
                state.quit = True
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", on_close)
        root.after(100, tick)
        # Keep remote on top initially
        try:
            root.attributes("-topmost", True)
            root.after(500, lambda: root.attributes("-topmost", False))
        except Exception:
            pass
        root.mainloop()

    t = threading.Thread(target=run, name="teleop-tk", daemon=True)
    t.start()
    print("[teleop] Tk remote launched (momentary hold-to-drive)")
    return t


# ---------------------------------------------------------------------------
# Stdin remote (always available as backup)
# ---------------------------------------------------------------------------

def _start_stdin_remote(state: TeleopState) -> threading.Thread:
    help_msg = (
        "\n[teleop stdin] commands:\n"
        "  p rough|rocky|turn   select policy (+ load defaults)\n"
        "  v <m/s>              set |vx|\n"
        "  w <rad/s>            set |ω|\n"
        "  f / b / l / r        HOLD direction (type s to stop)\n"
        "  s                    stop all\n"
        "  ?                    help\n"
        "  q                    quit\n"
        "Note: stdin is latch-until-stop (no key-release). Prefer Tk for momentary.\n"
    )
    print(help_msg)

    def run():
        while True:
            with state.lock:
                if state.quit:
                    return
            try:
                line = sys.stdin.readline()
            except Exception:
                return
            if not line:
                time.sleep(0.2)
                continue
            parts = line.strip().split()
            if not parts:
                continue
            cmd = parts[0].lower()
            if cmd in ("q", "quit", "exit"):
                state.stop_all()
                with state.lock:
                    state.quit = True
                return
            if cmd in ("?", "h", "help"):
                print(help_msg)
            elif cmd == "p" and len(parts) >= 2:
                state.set_policy(parts[1])
                n, vx, om = state.snapshot_mags()
                print(f"[teleop] policy={n} vx={vx:.3f} ω={om:.3f}")
            elif cmd == "v" and len(parts) >= 2:
                try:
                    state.set_mags(vx=float(parts[1]))
                    print(f"[teleop] vx_mag={float(parts[1]):.3f}")
                except ValueError:
                    print("[teleop] bad vx")
            elif cmd == "w" and len(parts) >= 2:
                try:
                    state.set_mags(omega=float(parts[1]))
                    print(f"[teleop] omega_mag={float(parts[1]):.3f}")
                except ValueError:
                    print("[teleop] bad omega")
            elif cmd == "f":
                state.stop_all()
                state.set_hold("forward", True)
            elif cmd == "b":
                state.stop_all()
                state.set_hold("back", True)
            elif cmd == "l":
                state.stop_all()
                state.set_hold("left", True)
            elif cmd == "r":
                state.stop_all()
                state.set_hold("right", True)
            elif cmd == "s":
                state.stop_all()
                print("[teleop] STOP")
            else:
                print(f"[teleop] unknown: {line.strip()}  (type ? for help)")

    t = threading.Thread(target=run, name="teleop-stdin", daemon=True)
    t.start()
    return t


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app

    import csv
    import torch
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab.envs import ManagerBasedRLEnvCfg
    import importlib
    import rexmi_rl  # noqa: F401

    # Spawn
    pre = SPAWN_PRESETS[args.spawn_preset]
    sx = args.spawn_x if args.spawn_x is not None else pre["x"]
    sy = args.spawn_y if args.spawn_y is not None else pre["y"]
    sz = args.spawn_z if args.spawn_z is not None else pre["z"]
    print(
        f"[teleop] spawn_preset={args.spawn_preset} → "
        f"x={sx:.2f} y={sy:.2f} z={sz:.2f} yaw={math.degrees(args.spawn_yaw):.0f}°"
    )

    # Env cfg
    task_spec = gym.spec(args.task)
    env_cfg_entry = task_spec.kwargs["env_cfg_entry_point"]
    agent_cfg_entry = task_spec.kwargs["rsl_rl_cfg_entry_point"]
    mod, cls = env_cfg_entry.rsplit(":", 1)
    env_cfg: ManagerBasedRLEnvCfg = getattr(importlib.import_module(mod), cls)()
    env_cfg.scene.num_envs = 1
    if hasattr(env_cfg.scene, "terrain") and hasattr(env_cfg.scene.terrain, "terrain_generator"):
        tg = env_cfg.scene.terrain.terrain_generator
        if tg is not None and hasattr(tg, "num_cols") and tg.num_cols != 1:
            print(f"[teleop] Patching terrain num_cols {tg.num_cols} → 1")
            tg.num_cols = 1
    env_cfg.sim.device = args.device

    try:
        env_cfg.events.reset_base.params = {
            "pose_range": {
                "x": (sx, sx),
                "y": (sy, sy),
                "yaw": (args.spawn_yaw, args.spawn_yaw),
                "z": (sz - 0.02, sz + 0.02),
            },
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        }
        print(f"[teleop] ✓ Spawn pinned")
    except Exception as e:
        print(f"[teleop] WARNING spawn pin: {e}")

    try:
        if hasattr(env_cfg, "terminations"):
            if hasattr(env_cfg.terminations, "base_contact"):
                env_cfg.terminations.base_contact = None
            if hasattr(env_cfg.terminations, "bad_orientation"):
                env_cfg.terminations.bad_orientation = None
            print("[teleop] ✓ Terminations base_contact/bad_orientation disabled")
    except Exception as e:
        print(f"[teleop] WARNING terminations: {e}")

    # Keep heading arrow if possible
    try:
        if hasattr(env_cfg.commands, "base_velocity"):
            env_cfg.commands.base_velocity.debug_vis = True
    except Exception:
        pass

    mod_a, cls_a = agent_cfg_entry.rsplit(":", 1)
    agent_cfg = getattr(importlib.import_module(mod_a), cls_a)()
    agent_cfg.device = args.device

    env = gym.make(
        args.task,
        cfg=env_cfg,
        render_mode=None if args.headless else "rgb_array",
    )
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Load policies
    policies: dict[str, Callable] = {}
    ckpt_map = {
        "rough": args.ckpt_rough,
        "rocky_slope": args.ckpt_rocky,
        "turn": args.ckpt_turn,
    }
    for name, path in ckpt_map.items():
        if path and os.path.isfile(path):
            print(f"[teleop] Loading {name}: {path}")
            policies[name] = _load_policy(path, args.device)
        else:
            print(f"[teleop] WARNING: missing ckpt for {name}: {path}")

    if not policies:
        raise SystemExit("[teleop] ERROR: no policies loaded")

    initial = args.policy if args.policy in policies else next(iter(policies))
    state = TeleopState()
    state.set_policy(initial)
    _patch_command_ranges(env, initial)
    last_policy = initial

    # Remotes
    tk_thread = None
    if not args.no_gui:
        tk_thread = _start_tk_remote(state)
    _start_stdin_remote(state)
    if tk_thread is None and not args.no_gui:
        print("[teleop] Falling back to stdin-only remote")

    # CSV
    os.makedirs("logs/nav", exist_ok=True)
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_path = args.log_file or os.path.join("logs/nav", f"teleop_{ts}.csv")
    log_f = open(log_path, "w", newline="", buffering=1)
    writer = csv.writer(log_f)
    writer.writerow([
        "step", "policy", "vx_cmd", "omega_cmd",
        "x", "y", "z", "yaw_deg", "speed", "up_z",
    ])
    print(f"[teleop] log → {log_path}")

    robot = env.unwrapped.scene["robot"]
    obs, _ = env.reset()
    if isinstance(obs, dict):
        obs_t = obs.get("policy", next(iter(obs.values())))
    else:
        obs_t = obs
    if not torch.is_tensor(obs_t):
        obs_t = torch.as_tensor(obs_t, device=args.device)

    # Settle
    with torch.inference_mode():
        zero_actions = torch.zeros_like(policies[initial](obs_t))
    n_settle = max(1, int(round(args.settle_s / 0.02)))
    print(f"[teleop] settle {args.settle_s:.1f}s zero-action…")
    for _ in range(n_settle):
        if not simulation_app.is_running():
            break
        _inject(env, 0.0, 0.0)
        obs_t, _, _, _ = env.step(zero_actions)
        if not torch.is_tensor(obs_t):
            obs_t = torch.as_tensor(obs_t, device=args.device)

    print(
        "[teleop] READY — hold Forward/Back/Left/Right (or WASD). "
        "Release = stop. Policy dropdown or 1/2/3."
    )

    step = 0
    try:
        while simulation_app.is_running() and step < args.max_steps:
            with state.lock:
                if state.quit:
                    break

            pol_name, vx, omega = state.command()

            # Policy switch → patch ranges once
            if pol_name != last_policy:
                if pol_name not in policies:
                    # fallback
                    pol_name = last_policy if last_policy in policies else next(iter(policies))
                    with state.lock:
                        state.policy_name = pol_name
                else:
                    _patch_command_ranges(env, pol_name)
                    print(f"[teleop] active policy → {pol_name}")
                    last_policy = pol_name

            if pol_name not in policies:
                pol_name = next(iter(policies))

            _inject(env, vx, omega)

            if not torch.is_tensor(obs_t):
                obs_t = torch.as_tensor(obs_t, device=args.device)
            actions = policies[pol_name](obs_t)
            # If fully stopped, still run policy (standing) — command is zero
            obs_t, _, dones, _ = env.step(actions)

            # Telemetry
            pos = robot.data.root_pos_w[0]
            x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
            yaw = _yaw(robot)
            spd = _speed(robot)
            uz = _body_up_z(robot)
            writer.writerow([
                step, pol_name, f"{vx:.4f}", f"{omega:.4f}",
                f"{x:.3f}", f"{y:.3f}", f"{z:.3f}",
                f"{math.degrees(yaw):.2f}", f"{spd:.3f}", f"{uz:.3f}",
            ])

            pose_s = (
                f"pos=({x:+.1f},{y:+.1f},{z:+.1f}) "
                f"yaw={math.degrees(yaw):+.0f}° v={spd:.2f} "
                f"cmd=({vx:+.2f},{omega:+.3f}) up_z={uz:+.2f}"
            )
            with state.lock:
                state.status = f"step={step}"
                state.pose_str = pose_s

            if step % 50 == 0:
                print(f"[teleop][{step:5d}] {pol_name:12s} {pose_s}")

            try:
                if bool(dones[0]):
                    print(f"[teleop] WARNING env done at step {step}")
            except Exception:
                pass

            step += 1

    except KeyboardInterrupt:
        print("\n[teleop] Interrupted.")
    finally:
        with state.lock:
            state.quit = True
        state.stop_all()
        log_f.close()
        print(f"[teleop] Log saved: {log_path}")
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
