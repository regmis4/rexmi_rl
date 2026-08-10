#!/usr/bin/env python3
# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Crater turn isolation test — Option A Phase 1.

Loads ONLY model_13345 on the crater bowl env and runs a train-matched
command schedule.  No rocky, no nav FSM, no policy switch.

Purpose
-------
Answer: does 13345 spin on crater terrain when given a clean start?

  PASS → domain OK; nav failure is handoff (Phase 2A)
  FAIL → domain gap; need finetune or demo without nav pivot (Phase 2B)

Schedule
--------
  HOLD   10.0 s   actions=0, cmd=(0,0,0)  # physics settle ONLY — no policy
  (optional soft-upright if still tipped)
  SETTLE  2.0 s   vx=0.05  ω=0            # then 13345
  YAW+    6.0 s   vx=0.05  ω=+0.07
  PLANT   1.0 s   vx=0.05  ω=0
  YAW-    6.0 s   vx=0.05  ω=-0.07
  PLANT   1.0 s   vx=0.05  ω=0

Pass criteria (per yaw phase)
-----------------------------
  • |Δbody_yaw| ≥ 25° over the 6 s window
  • mean |ω_body| ≥ 0.03 rad/s during yaw
  • up_z stayed ≥ 0.50 (no tip)
  • no thrash: peak |ω_body| < 2.5 rad/s while |ω_cmd|≤0.08

Usage
-----
  conda activate env_isaacsim
  cd /home/susan/rexmi_rl

  python scripts/test_turn_crater.py \\
      --task RexmiRl-Go2w-Crater-Bowl-RockySlope-Play-v0 \\
      --checkpoint logs/rsl_rl/go2w_velocity_slope_turn/2026-07-27_20-54-25/model_13345.pt

  # Headless metrics only:
  python scripts/test_turn_crater.py --task ... --checkpoint ... --headless

  # Spawn on floor (x=0) instead of rim:
  python scripts/test_turn_crater.py ... --spawn_x 0.0 --spawn_z 1.2
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

_SOURCE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "source"
)
if _SOURCE_DIR not in sys.path:
    sys.path.insert(0, _SOURCE_DIR)


def _parse_args():
    p = argparse.ArgumentParser(description="Crater turn isolation test (13345 only)")
    p.add_argument(
        "--task",
        default="RexmiRl-Go2w-Crater-Bowl-RockySlope-Play-v0",
        help="Crater gym task (same as navigate)",
    )
    p.add_argument(
        "--checkpoint",
        default=(
            "logs/rsl_rl/go2w_velocity_slope_turn/"
            "2026-07-27_20-54-25/model_13345.pt"
        ),
        help="Turn policy checkpoint",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--headless", action="store_true")
    p.add_argument(
        "--spawn_preset",
        choices=["floor", "mid_slope", "rim_out"],
        default="floor",
        help="Spawn location: floor (inside bowl), mid_slope (wall), rim_out (exterior). "
             "Overridden by explicit --spawn_x/y/z if those flags are passed.",
    )
    p.add_argument("--spawn_x", type=float, default=None,
                   help="Override spawn X (default from --spawn_preset)")
    p.add_argument("--spawn_y", type=float, default=None,
                   help="Override spawn Y")
    p.add_argument("--spawn_z", type=float, default=None,
                   help="Override spawn Z")
    p.add_argument("--spawn_yaw", type=float, default=math.pi, help="radians")
    p.add_argument("--vx", type=float, default=0.05, help="train lin_vel_x")
    p.add_argument("--omega", type=float, default=0.07, help="train |ang_vel_z|")
    p.add_argument("--hold_s", type=float, default=10.0,
                   help="Zero-action physics settle after spawn (s)")
    p.add_argument("--settle_s", type=float, default=2.0)
    p.add_argument("--yaw_s", type=float, default=6.0)
    p.add_argument("--plant_s", type=float, default=1.0)
    p.add_argument("--dt", type=float, default=0.02, help="policy dt guess (s)")
    p.add_argument(
        "--log_file",
        default=None,
        help="CSV path (default logs/nav/turn_iso_<ts>.csv)",
    )
    return p.parse_args()


def _load_policy(checkpoint_path: str, device: str):
    """Same architecture inference as navigate._load_policy."""
    import torch
    from rsl_rl.modules import ActorCritic

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
            print(f"[iso] obs normalizer loaded shape={tuple(obs_mean.shape)}")

    def policy(obs):
        with torch.inference_mode():
            o = obs[..., :obs_dim]
            if obs_mean is not None:
                o = (o - obs_mean) / obs_std
            return ac.act_inference(o)

    print(
        f"[iso] Loaded {os.path.basename(checkpoint_path)} "
        f"obs={obs_dim} act={num_actions} hidden={hidden_dims}"
    )
    return policy, obs_dim, num_actions


def _wrap(a: float) -> float:
    return a - 2.0 * math.pi * math.floor((a + math.pi) / (2.0 * math.pi))


def _body_up_z(robot, env_idx: int = 0) -> float:
    try:
        quat = robot.data.root_quat_w[env_idx]
        w, qx, qy, qz = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
        return 1.0 - 2.0 * (qx * qx + qy * qy)
    except Exception:
        return float("nan")


def _yaw(robot, env_idx: int = 0) -> float:
    try:
        quat = robot.data.root_quat_w[env_idx]
        w, qx, qy, qz = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
        return math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    except Exception:
        return float("nan")


def _omega_body(robot, env_idx: int = 0) -> float:
    try:
        return float(robot.data.root_ang_vel_b[env_idx][2])
    except Exception:
        return float("nan")



def _soft_upright(robot, env_idx: int = 0) -> None:
    """Force upright pose at current xy (for isolation only)."""
    import torch
    try:
        pos = robot.data.root_pos_w[env_idx]
        tx, ty, tz = float(pos[0]), float(pos[1]), float(pos[2]) + 0.35
        yaw = _yaw(robot, env_idx)
        if yaw != yaw:
            yaw = 0.0
        half = 0.5 * yaw
        qw, qx, qy, qz = math.cos(half), 0.0, 0.0, math.sin(half)
        device = robot.data.root_pos_w.device
        dtype = robot.data.root_pos_w.dtype
        env_ids = torch.tensor([env_idx], device=device, dtype=torch.long)
        root_pose = torch.tensor(
            [[tx, ty, tz, qw, qx, qy, qz]], device=device, dtype=dtype
        )
        root_vel = torch.zeros((1, 6), device=device, dtype=dtype)
        if hasattr(robot, "write_root_pose_to_sim"):
            robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        if hasattr(robot, "write_root_velocity_to_sim"):
            robot.write_root_velocity_to_sim(root_vel, env_ids=env_ids)
        if hasattr(robot, "write_joint_state_to_sim") and hasattr(robot.data, "default_joint_pos"):
            q = robot.data.default_joint_pos[env_idx].unsqueeze(0).clone()
            qd = torch.zeros_like(q)
            try:
                robot.write_joint_state_to_sim(q, qd, env_ids=env_ids)
            except TypeError:
                robot.write_joint_state_to_sim(q, qd, None, env_ids)
        print(f"[iso] soft-upright at ({tx:+.1f},{ty:+.1f},{tz:+.1f}) yaw={math.degrees(yaw):+.0f}°")
    except Exception as e:
        print(f"[iso] soft-upright failed: {e}")


def _inject(env, vx: float, omega: float, env_idx: int = 0) -> None:
    try:
        cmd = env.unwrapped.command_manager.get_command("base_velocity")
        cmd[env_idx, 0] = float(vx)
        cmd[env_idx, 1] = 0.0
        cmd[env_idx, 2] = float(omega)
    except Exception as e:
        print(f"[iso] inject failed: {e}")


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


    # Resolve spawn from preset (interior crater by default)
    _PRESETS = {
        # floor centre — rough bowl mesh, low slope
        "floor":     dict(x=0.0,  y=0.0, z=1.15),
        # mid wall on +x side — real crater slope
        "mid_slope": dict(x=6.5,  y=0.0, z=2.40),
        # exterior ramp (old default)
        "rim_out":   dict(x=13.0, y=0.0, z=4.50),
    }
    _pre = _PRESETS.get(getattr(args, "spawn_preset", "floor"), _PRESETS["floor"])
    if args.spawn_x is None:
        args.spawn_x = _pre["x"]
    if args.spawn_y is None:
        args.spawn_y = _pre["y"]
    if args.spawn_z is None:
        args.spawn_z = _pre["z"]
    print(
        f"[iso] spawn_preset={getattr(args, 'spawn_preset', '?')} → "
        f"x={args.spawn_x:.2f} y={args.spawn_y:.2f} z={args.spawn_z:.2f}"
    )

    if not os.path.isfile(args.checkpoint):
        raise SystemExit(f"[iso] checkpoint not found: {args.checkpoint}")

    # --- env ---
    task_spec = gym.spec(args.task)
    env_cfg_entry = task_spec.kwargs["env_cfg_entry_point"]
    agent_cfg_entry = task_spec.kwargs["rsl_rl_cfg_entry_point"]
    mod, cls = env_cfg_entry.rsplit(":", 1)
    env_cfg: ManagerBasedRLEnvCfg = getattr(importlib.import_module(mod), cls)()
    env_cfg.scene.num_envs = 1
    if hasattr(env_cfg.scene, "terrain") and hasattr(env_cfg.scene.terrain, "terrain_generator"):
        tg = env_cfg.scene.terrain.terrain_generator
        if tg is not None and hasattr(tg, "num_cols") and tg.num_cols != 1:
            tg.num_cols = 1
    env_cfg.sim.device = args.device

    try:
        env_cfg.events.reset_base.params = {
            "pose_range": {
                "x": (args.spawn_x, args.spawn_x),
                "y": (args.spawn_y, args.spawn_y),
                "yaw": (args.spawn_yaw, args.spawn_yaw),
                "z": (args.spawn_z - 0.02, args.spawn_z + 0.02),
            },
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        }
        print(
            f"[iso] spawn x={args.spawn_x} y={args.spawn_y} z={args.spawn_z} "
            f"yaw={math.degrees(args.spawn_yaw):.0f}°"
        )
    except Exception as e:
        print(f"[iso] WARNING spawn pin failed: {e}")

    try:
        if hasattr(env_cfg, "terminations"):
            if hasattr(env_cfg.terminations, "base_contact"):
                env_cfg.terminations.base_contact = None
            if hasattr(env_cfg.terminations, "bad_orientation"):
                env_cfg.terminations.bad_orientation = None
    except Exception:
        pass

    mod_a, cls_a = agent_cfg_entry.rsplit(":", 1)
    agent_cfg = getattr(importlib.import_module(mod_a), cls_a)()
    agent_cfg.device = args.device

    # Disable command debug_vis (avoids cuda/cpu arrow crash on inject)
    try:
        if hasattr(env_cfg.commands, "base_velocity"):
            env_cfg.commands.base_velocity.debug_vis = False
    except Exception:
        pass

    env = gym.make(args.task, cfg=env_cfg, render_mode=None if args.headless else "rgb_array")
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Command ranges: match 13345 train band (and allow inject)
    try:
        _bv = env.unwrapped.command_manager.cfg.base_velocity
        _bv.ranges.ang_vel_z = (-0.10, 0.10)
        if hasattr(_bv.ranges, "lin_vel_x"):
            _bv.ranges.lin_vel_x = (0.0, 0.15)
        print("[iso] command ranges: ang_vel_z=(-0.10,0.10) lin_vel_x=(0,0.15)")
    except Exception as e:
        print(f"[iso] WARNING range patch: {e}")

    policy, obs_dim, n_act = _load_policy(args.checkpoint, args.device)
    robot = env.unwrapped.scene["robot"]

    # CSV log
    os.makedirs("logs/nav", exist_ok=True)
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_path = args.log_file or os.path.join("logs/nav", f"turn_iso_{ts}.csv")
    log_f = open(log_path, "w", newline="")
    import csv as _csv

    w = _csv.writer(log_f)
    w.writerow(
        [
            "step", "phase", "t_phase",
            "vx_cmd", "omega_cmd",
            "yaw_deg", "omega_body", "up_z",
            "action_rms", "x", "y", "z",
        ]
    )
    print(f"[iso] log → {log_path}")

    obs, _ = env.reset()
    if isinstance(obs, dict):
        obs_t = obs.get("policy", next(iter(obs.values())))
    else:
        obs_t = obs

    # Infer action dim for zero actions
    if not torch.is_tensor(obs_t):
        obs_t = torch.as_tensor(obs_t, device=args.device)
    # Probe action shape once
    with torch.inference_mode():
        _probe = policy(obs_t)
    zero_actions = torch.zeros_like(_probe)

    yaw_stats: dict[str, dict] = {}
    step = 0
    dt = args.dt

    # ------------------------------------------------------------------
    # HOLD: physics settle — NO policy (actions = 0)
    # ------------------------------------------------------------------
    hold_s = float(args.hold_s)
    n_hold = max(1, int(round(hold_s / dt)))
    print(f"[iso] === HOLD {hold_s:.0f}s zero-action physics settle ===")
    calm_ok = False
    for i in range(n_hold):
        if not simulation_app.is_running():
            break
        _inject(env, 0.0, 0.0)
        obs_t, _, _, _ = env.step(zero_actions)
        if not torch.is_tensor(obs_t):
            obs_t = torch.as_tensor(obs_t, device=args.device)

        y = _yaw(robot)
        ob = _omega_body(robot)
        uz = _body_up_z(robot)
        pos = robot.data.root_pos_w[0]
        x, yy, z = float(pos[0]), float(pos[1]), float(pos[2])

        if step % 50 == 0:
            print(
                f"[iso][{step:5d}] HOLD     "
                f"yaw={math.degrees(y):+7.1f}° "
                f"ωb={ob:+.3f} up_z={uz:+.2f} "
                f"act=0 "
                f"t={i * dt:.1f}/{hold_s:.0f}s"
            )
        w.writerow(
            [
                step, "HOLD", f"{i * dt:.3f}",
                "0.0000", "0.0000",
                f"{math.degrees(y):.3f}", f"{ob:.4f}", f"{uz:.4f}",
                "0.0000", f"{x:.3f}", f"{yy:.3f}", f"{z:.3f}",
            ]
        )
        step += 1

        # Note calm once (keep full hold duration for visual settle)
        if (
            not calm_ok
            and i * dt >= 3.0
            and uz == uz and uz > 0.90
            and ob == ob and abs(ob) < 0.15
        ):
            calm_ok = True
            print(
                f"[iso] HOLD calm at t={i * dt:.1f}s "
                f"up_z={uz:+.2f} ωb={ob:+.3f} (continuing hold)"
            )

    uz_end = _body_up_z(robot)
    ob_end = _omega_body(robot)
    print(
        f"[iso] HOLD done: up_z={uz_end:+.2f} ωb={ob_end:+.3f} "
        f"(want up_z>0.9 |ωb|<0.15)"
    )

    # If still tipped after hold, soft-upright so we can test the policy
    if uz_end != uz_end or uz_end < 0.70:
        print("[iso] still not upright after HOLD — soft-upright + 2s plant")
        _soft_upright(robot)
        for i in range(int(round(2.0 / dt))):
            _inject(env, 0.0, 0.0)
            obs_t, _, _, _ = env.step(zero_actions)
            if not torch.is_tensor(obs_t):
                obs_t = torch.as_tensor(obs_t, device=args.device)
            step += 1
        uz_end = _body_up_z(robot)
        print(f"[iso] after soft-upright: up_z={uz_end:+.2f}")

    if uz_end != uz_end or uz_end < 0.70:
        print("[iso] WARNING: starting schedule while not upright — results may be invalid")

    # Schedule after calm
    phases = [
        ("SETTLE", args.settle_s, args.vx, 0.0),
        ("YAW_POS", args.yaw_s, args.vx, +args.omega),
        ("PLANT1", args.plant_s, args.vx, 0.0),
        ("YAW_NEG", args.yaw_s, args.vx, -args.omega),
        ("PLANT2", args.plant_s, args.vx, 0.0),
    ]

    print("[iso] === START isolation schedule (policy ON) ===")

    for phase_name, dur_s, vx_cmd, om_cmd in phases:
        n_steps = max(1, int(round(dur_s / dt)))
        yaw0 = _yaw(robot)
        up_min = 1.0
        om_abs_sum = 0.0
        om_abs_n = 0
        om_peak = 0.0
        act_rms_sum = 0.0
        print(
            f"[iso] PHASE {phase_name}  {dur_s:.1f}s  "
            f"vx={vx_cmd:+.3f} ω={om_cmd:+.3f}  yaw0={math.degrees(yaw0):+.1f}°"
        )

        for i in range(n_steps):
            if not simulation_app.is_running():
                break

            _inject(env, vx_cmd, om_cmd)

            # Policy step
            if not torch.is_tensor(obs_t):
                obs_t = torch.as_tensor(obs_t, device=args.device)
            actions = policy(obs_t)
            act_rms = float(actions.float().pow(2).mean().sqrt().item())
            act_rms_sum += act_rms

            obs_t, _, _, _ = env.step(actions)

            y = _yaw(robot)
            ob = _omega_body(robot)
            uz = _body_up_z(robot)
            pos = robot.data.root_pos_w[0]
            x, yy, z = float(pos[0]), float(pos[1]), float(pos[2])

            if uz == uz and uz < up_min:
                up_min = uz
            if ob == ob:
                om_abs_sum += abs(ob)
                om_abs_n += 1
                om_peak = max(om_peak, abs(ob))

            if step % 25 == 0:
                print(
                    f"[iso][{step:5d}] {phase_name:8s} "
                    f"yaw={math.degrees(y):+7.1f}° "
                    f"ωb={ob:+.3f} up_z={uz:+.2f} "
                    f"act_rms={act_rms:.3f} "
                    f"cmd=({vx_cmd:+.2f},{om_cmd:+.3f})"
                )

            w.writerow(
                [
                    step, phase_name, f"{i * dt:.3f}",
                    f"{vx_cmd:.4f}", f"{om_cmd:.4f}",
                    f"{math.degrees(y):.3f}", f"{ob:.4f}", f"{uz:.4f}",
                    f"{act_rms:.4f}", f"{x:.3f}", f"{yy:.3f}", f"{z:.3f}",
                ]
            )
            step += 1

        yaw1 = _yaw(robot)
        dyaw = abs(_wrap(yaw1 - yaw0))
        mean_om = om_abs_sum / max(1, om_abs_n)
        mean_act = act_rms_sum / max(1, n_steps)
        yaw_stats[phase_name] = {
            "dyaw_deg": math.degrees(dyaw),
            "mean_om": mean_om,
            "peak_om": om_peak,
            "up_min": up_min,
            "mean_act_rms": mean_act,
            "yaw0": math.degrees(yaw0),
            "yaw1": math.degrees(yaw1),
        }
        print(
            f"[iso] END {phase_name}: Δyaw={math.degrees(dyaw):.1f}° "
            f"mean|ωb|={mean_om:.3f} peak|ωb|={om_peak:.3f} "
            f"up_min={up_min:+.2f} act_rms={mean_act:.3f}"
        )

    log_f.close()

    # --- Verdict ---
    print("\n========== ISOLATION VERDICT ==========")
    # Fair criteria: big yaw + stayed upright = success.
    # Peak ω alone is NOT thrash if up_z stayed high (fast spin is OK).
    overall_pass = False
    any_good_turn = False
    for name in ("YAW_POS", "YAW_NEG"):
        s = yaw_stats.get(name)
        if s is None:
            print(f"  {name}: MISSING")
            continue
        ok_dyaw = s["dyaw_deg"] >= 40.0
        ok_up = s["up_min"] >= 0.85
        # thrash only if high rate AND attitude collapse
        ok_stable = ok_up and not (s["peak_om"] > 4.0 and s["up_min"] < 0.70)
        phase_pass = ok_dyaw and ok_stable
        if phase_pass:
            any_good_turn = True
        flag = "PASS" if phase_pass else "FAIL"
        print(
            f"  {name}: {flag}  "
            f"Δyaw={s['dyaw_deg']:.1f}°({'ok' if ok_dyaw else 'LOW'})  "
            f"mean|ωb|={s['mean_om']:.3f}  "
            f"up_min={s['up_min']:+.2f}({'ok' if ok_up else 'TIP'})  "
            f"peak|ωb|={s['peak_om']:.2f}"
        )

    overall_pass = any_good_turn
    if overall_pass:
        print("\n  >>> OVERALL: PASS — at least one sign spun ≥40° while upright.")
        print("  >>> Next: Phase 2A clean handoff in navigate (--enable_turn).")
        print("  >>> Note: if only one sign works, nav should prefer that sign / retry flip.")
    else:
        print("\n  >>> OVERALL: FAIL — no stable large turn on this spawn.")
        print("  >>> Try --spawn_preset floor|mid_slope or check cmd sign / finetune.")
    print(f"  log: {log_path}")
    print("========================================\n")

    env.close()
    simulation_app.close()
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
