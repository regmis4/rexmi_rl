#!/usr/bin/env python3
# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Autonomous navigation entry point for REXMI RL.

Loads ALL THREE trained RL policy checkpoints and runs a deterministic nav layer
on top.  The nav layer picks which policy to use each step based on real-time
terrain metrics from the height scanner:

  fast_flat    — slope < 5°, no obstacles (up to 2 m/s)
  rough        — rough terrain, step < 10 cm, slope < 20° (~0.8 m/s)
  rocky_slope  — steep with boulders, slope 20–35° (~0.4 m/s)

Policy switching uses a 25-step hysteresis (0.5 s) to avoid oscillation
at terrain boundaries.

Usage
-----
  # Full autonomous traverse (down into crater, floor, up opposite side)
  python scripts/navigate.py \\
      --task RexmiRl-Go2w-Crater-Bowl-RockySlope-Play-v0 \\
      --ckpt_fast_flat  logs/rsl_rl/go2w_velocity_fast_flat/2026-06-17_20-08-58/model_1499.pt \\
      --ckpt_rough      logs/rsl_rl/go2w_velocity_rough/2026-06-14_20-03-41/model_8996.pt \\
      --ckpt_rocky      logs/rsl_rl/go2w_velocity_rocky_slope/2026-06-30_09-31-48/model_13994.pt \\
      --mission traverse

  # Resource survey (lawnmower scan of crater floor)
  python scripts/navigate.py ... --mission survey

  # Perimeter rim circuit
  python scripts/navigate.py ... --mission rim_circuit

  # Use only one policy (no auto-switching)
  python scripts/navigate.py ... --policy_mode rocky_slope

  # Headless (no matplotlib dashboard)
  python scripts/navigate.py ... --no_dashboard

Missions
--------
  traverse    — enter crater → cross floor → exit opposite side
  survey      — systematic lawnmower scan of crater floor
  rim_circuit — clockwise loop around the crater rim

Crater geometry (defaults match LunarCraterDemoBowlCfg)
--------------------------------------------------------
  --crater_x 0.0 --crater_y 0.0   crater centre (world frame)
  --r_floor 3.0                    flat floor radius (m)
  --r_rim 11.0                     rim radius (m)
  --spawn_x -18.0                  robot spawn x (m from centre)

Policy switching
----------------
  The PolicySelector reads terrain metrics every step from the LocalPlanner:
    slope_ahead           : terrain slope (tan θ) in best forward column
    max_step              : largest vertical step in 16×10 height scan (m)
    traversable_fraction  : fraction of 5 heading candidates that are clear

  Decision rules (hysteresis: 25 steps before switching):
    slope > tan(20°) = 0.36        → rocky_slope
    step > 6 cm                    → rough (or rocky_slope if > 10 cm)
    traversable_fraction < 40%     → rough
    else                           → fast_flat
"""

import argparse
import math
import os
import sys
import time

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ISAACLAB_DIR = os.environ.get("ISAACLAB_DIR", os.path.expanduser("~/IsaacLab"))
_SOURCE_DIR  = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "source"
)
if _SOURCE_DIR not in sys.path:
    sys.path.insert(0, _SOURCE_DIR)

import rexmi_rl  # noqa: F401 — registers all environments


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="REXMI autonomous navigation demo")

    # Checkpoints — all three required for auto-switching
    p.add_argument("--task", required=True,
                   help="Isaac Lab gym task name")
    p.add_argument("--ckpt_fast_flat", required=False, default=None,
                   help="Checkpoint for fast_flat policy (.pt)")
    p.add_argument("--ckpt_rough",     required=False, default=None,
                   help="Checkpoint for rough policy (.pt)")
    p.add_argument("--ckpt_rocky",     required=False, default=None,
                   help="Checkpoint for rocky_slope policy (.pt)")

    # Task IDs used to build the network for each policy (must match training task)
    p.add_argument("--task_fast_flat",
                   default="RexmiRl-Go2w-Crater-Bowl-FastFlat-Play-v0",
                   help="Task used to build fast_flat network architecture")
    p.add_argument("--task_rough",
                   default="RexmiRl-Go2w-Crater-Bowl-Play-v0",
                   help="Task used to build rough network architecture")
    # rocky uses --task (the main nav env)

    # Convenience: single checkpoint + fixed mode (disables auto-switching)
    p.add_argument("--checkpoint", default=None,
                   help="Single checkpoint path (use with --policy_mode)")
    p.add_argument("--policy_mode",
                   choices=["auto", "fast_flat", "rough", "rocky_slope"],
                   default="auto",
                   help="'auto' uses PolicySelector; others use a fixed policy")

    # Mission
    p.add_argument("--mission",
                   choices=["traverse", "survey", "rim_circuit"],
                   default="traverse")

    # Crater geometry — defaults match LunarCraterDemoBowlEnvCfg:
    #   crater centre at tile centre = env_origin = (0, 0)
    #   robot spawns at x=+13 m (exterior ramp, 2 m past rim at r=11 m)
    #   yaw=π → robot faces -x (toward crater centre)
    #   So waypoints must be in -x direction: approach at x=+8, rim_entry at x=+11, etc.
    p.add_argument("--crater_x",  type=float, default=0.0)
    p.add_argument("--crater_y",  type=float, default=0.0)
    p.add_argument("--r_floor",   type=float, default=3.0)
    p.add_argument("--r_rim",     type=float, default=11.0)
    p.add_argument("--spawn_x",   type=float, default=13.0,
                   help="Robot spawn X (m). Default 13.0 matches LunarCraterDemoBowlEnvCfg "
                        "(exterior ramp, 2 m past rim). Robot faces -x (toward crater).")
    p.add_argument("--entry_azimuth_deg", type=float, default=0.0,
                   help="Azimuth (degrees, CCW from +x) at which robot enters crater. "
                        "Default 0° = enters from +x side heading -x (matches spawn_x=+13).")

    # Nav tuning
    p.add_argument("--replan_interval", type=float, default=2.0)
    p.add_argument("--vx_normal",       type=float, default=0.50)
    p.add_argument("--vx_steep",        type=float, default=0.40)

    # Sim settings
    p.add_argument("--num_envs",    type=int,   default=1)
    p.add_argument("--device",      default="cuda:0")
    p.add_argument("--max_steps",   type=int,   default=15000,
                   help="Max sim steps (~300 s at 50 Hz)")
    p.add_argument("--no_dashboard", action="store_true")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint loader helper
# ---------------------------------------------------------------------------

def _load_policy(checkpoint_path: str, nav_env, agent_cfg, device: str):
    """
    Load a single RSL-RL checkpoint and return its inference callable.

    Isaac Lab only supports one simulation context per process, so we cannot
    create a second env.  Instead we:
      1. Inspect the actor.0.weight shape in the checkpoint to get obs_dim
         and hidden_sizes.
      2. Build an ActorCritic with the correct architecture directly.
      3. Load the state_dict.
      4. Wrap in the OnPolicyRunner's obs-normaliser so the returned callable
         mirrors what runner.get_inference_policy() would return.

    Returns a callable:  policy(obs_tensor) → action_tensor
    """
    import torch
    from rsl_rl.modules import ActorCritic
    from rsl_rl.runners import OnPolicyRunner

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    msd  = ckpt["model_state_dict"]

    # Infer architecture from actor weight shapes
    obs_dim    = msd["actor.0.weight"].shape[1]  # input features
    num_actions = msd["actor.6.weight"].shape[0]  # output actions

    # Collect hidden layer widths (layers 0, 2, 4 → biases give hidden dims)
    hidden_dims = [
        msd["actor.0.bias"].shape[0],
        msd["actor.2.bias"].shape[0],
        msd["actor.4.bias"].shape[0],
    ]

    # Build ActorCritic with inferred architecture
    ac = ActorCritic(
        num_actor_obs=obs_dim,
        num_critic_obs=obs_dim,
        num_actions=num_actions,
        actor_hidden_dims=hidden_dims,
        critic_hidden_dims=hidden_dims,
    ).to(device)
    ac.load_state_dict(msd)
    ac.eval()

    # Load obs normalizer if present (newer RSL-RL checkpoints store it separately).
    # The state_dict has keys: _mean, _var, _std, count — apply normalisation
    # manually rather than importing a class that may not exist in this rsl_rl version.
    obs_mean = None
    obs_std  = None
    if "obs_norm_state_dict" in ckpt:
        nd = ckpt["obs_norm_state_dict"]
        if "_mean" in nd and "_std" in nd:
            obs_mean = nd["_mean"].to(device)   # shape (1, obs_dim) or (obs_dim,)
            obs_std  = nd["_std"].to(device)
            # Clamp std to avoid division by zero
            obs_std  = obs_std.clamp(min=1e-6)
            print(f"  obs normalizer loaded (mean/std shape: {obs_mean.shape})")

    # Build callable that mimics runner.get_inference_policy()
    # obs may come from a larger env (e.g. 247-dim rocky env feeding a 60-dim
    # fast_flat policy).  Slice to the policy's expected obs_dim.
    def policy(obs: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            o = obs[..., :obs_dim]   # slice to this policy's obs dim
            if obs_mean is not None:
                o = (o - obs_mean) / obs_std
            return ac.act_inference(o)

    print(f"[_load_policy] Loaded {os.path.basename(checkpoint_path)}: "
          f"obs_dim={obs_dim}, hidden={hidden_dims}, actions={num_actions}")
    return policy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()

    # ------------------------------------------------------------------
    # 1. Boot Isaac Sim
    # ------------------------------------------------------------------
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=False)
    simulation_app = app_launcher.app

    # ------------------------------------------------------------------
    # 2. Post-sim imports
    # ------------------------------------------------------------------
    import torch
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from rexmi_rl.nav.mission import Mission
    from rexmi_rl.nav.navigator import Navigator
    from rexmi_rl.nav.policy_selector import PolicySelector, PolicyMode

    # ------------------------------------------------------------------
    # 3. Load env cfg from task registry and create environment
    # ------------------------------------------------------------------
    from isaaclab.envs import ManagerBasedRLEnvCfg

    # Resolve the env cfg entry point registered with the task
    task_spec = gym.spec(args.task)
    env_cfg_entry = task_spec.kwargs["env_cfg_entry_point"]
    agent_cfg_entry = task_spec.kwargs["rsl_rl_cfg_entry_point"]

    # Import and instantiate env_cfg
    module_str, cls_str = env_cfg_entry.rsplit(":", 1)
    import importlib
    env_cfg: ManagerBasedRLEnvCfg = getattr(importlib.import_module(module_str), cls_str)()

    # Navigation always uses exactly 1 robot.
    # The training cfg may spawn 10 envs with 10 terrain columns — collapse to 1.
    env_cfg.scene.num_envs = 1
    if hasattr(env_cfg.scene, "terrain") and hasattr(env_cfg.scene.terrain, "terrain_generator"):
        tg = env_cfg.scene.terrain.terrain_generator
        if tg is not None and hasattr(tg, "num_cols") and tg.num_cols != 1:
            print(f"[navigate] Patching terrain: num_cols {tg.num_cols} → 1 "
                  f"(navigation uses single terrain tile)")
            tg.num_cols = 1
    env_cfg.sim.device = args.device

    # Import and instantiate agent_cfg
    mod_a, cls_a = agent_cfg_entry.rsplit(":", 1)
    agent_cfg = getattr(importlib.import_module(mod_a), cls_a)()
    agent_cfg.device = args.device

    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ------------------------------------------------------------------
    # 4. Load policies
    # ------------------------------------------------------------------
    use_auto = (args.policy_mode == "auto")

    if use_auto:
        # Require all three checkpoints
        missing = []
        if not args.ckpt_fast_flat: missing.append("--ckpt_fast_flat")
        if not args.ckpt_rough:     missing.append("--ckpt_rough")
        if not args.ckpt_rocky:     missing.append("--ckpt_rocky")
        if missing:
            # Graceful fallback: if only one checkpoint given, disable auto
            if args.checkpoint:
                print(f"[navigate] WARNING: --policy_mode auto requires all 3 "
                      f"checkpoints. Falling back to fixed rocky_slope.")
                use_auto = False
                args.policy_mode = "rocky_slope"
                args.ckpt_rocky = args.checkpoint
            else:
                raise SystemExit(
                    f"[navigate] ERROR: --policy_mode auto requires: {missing}\n"
                    f"Or use --checkpoint + --policy_mode rocky_slope for fixed mode."
                )

    if use_auto:
        print("[navigate] Loading all 3 policies (each from its own task cfg)...")
        # Each policy is loaded from its own task so the network architecture matches
        # the checkpoint (different obs dims: flat=60, rough=varies, rocky=247)
        policies = {
            PolicyMode.FAST_FLAT:   _load_policy(args.ckpt_fast_flat, env, agent_cfg, args.device),
            PolicyMode.ROUGH:       _load_policy(args.ckpt_rough,     env, agent_cfg, args.device),
            PolicyMode.ROCKY_SLOPE: _load_policy(args.ckpt_rocky,     env, agent_cfg, args.device),
        }
        selector = PolicySelector(policies, initial_mode=PolicyMode.ROCKY_SLOPE)
        print("[navigate] PolicySelector: auto mode (terrain-aware switching)")
        print(f"  fast_flat  : {args.ckpt_fast_flat}")
        print(f"  rough      : {args.ckpt_rough}")
        print(f"  rocky_slope: {args.ckpt_rocky}")
    else:
        # Single policy fixed mode
        mode_map = {
            "fast_flat":   (PolicyMode.FAST_FLAT,   args.ckpt_fast_flat or args.checkpoint),
            "rough":       (PolicyMode.ROUGH,        args.ckpt_rough     or args.checkpoint),
            "rocky_slope": (PolicyMode.ROCKY_SLOPE,  args.ckpt_rocky     or args.checkpoint),
        }
        fixed_mode, fixed_ckpt = mode_map[args.policy_mode]
        if not fixed_ckpt:
            raise SystemExit(f"[navigate] ERROR: No checkpoint provided for {args.policy_mode}.")
        print(f"[navigate] Fixed policy mode: {args.policy_mode}")
        print(f"  checkpoint: {fixed_ckpt}")
        fixed_policy = _load_policy(fixed_ckpt, env, agent_cfg, args.device)
        policies = {
            PolicyMode.FAST_FLAT:   fixed_policy,
            PolicyMode.ROUGH:       fixed_policy,
            PolicyMode.ROCKY_SLOPE: fixed_policy,
        }
        selector = PolicySelector(policies, initial_mode=fixed_mode)
        _fixed = fixed_mode
        selector.update = lambda slope, max_step, trav_frac: _fixed

    print(f"[navigate] Mission: {args.mission}")

    # ------------------------------------------------------------------
    # 5. Create Navigator and attach PolicySelector
    # ------------------------------------------------------------------
    mission_enum = Mission(args.mission)
    nav = Navigator(
        env,
        mission=mission_enum,
        crater_centre=(args.crater_x, args.crater_y),
        r_floor=args.r_floor,
        r_rim=args.r_rim,
        spawn_x=args.spawn_x,
        entry_azimuth_deg=args.entry_azimuth_deg,
        env_idx=0,
        replan_interval_s=args.replan_interval,
    )
    nav._local.vx_normal = args.vx_normal
    nav._local.vx_steep  = args.vx_steep
    nav.policy_selector  = selector

    # Expose policy_status key in shared dict for dashboard
    nav.shared["policy_status"] = selector.status_str()

    # ------------------------------------------------------------------
    # 6. Start dashboard
    # ------------------------------------------------------------------
    if not args.no_dashboard:
        nav.start_dashboard()
        print("[navigate] Dashboard window launched")

    # ------------------------------------------------------------------
    # 7. Main sim loop
    # ------------------------------------------------------------------
    obs, _ = env.get_observations()
    step   = 0

    try:
        while simulation_app.is_running() and step < args.max_steps:
            # Get active policy from selector
            active_policy = selector.current_policy

            # RL policy forward pass with currently active policy
            with torch.no_grad():
                actions = active_policy(obs)

            # Sim step
            obs, rewards, dones, infos = env.step(actions)

            # Nav tick — updates selector, injects velocity command
            nav.step()

            step += 1

            # Mission complete check
            if nav._wp_idx >= len(nav._waypoints):
                print(f"[navigate] Mission '{args.mission}' COMPLETE at step {step}!")
                time.sleep(3.0)
                break

            # Terminal status every 50 steps (~1 s)
            if step % 50 == 0:
                with nav._lock:
                    pose      = nav.shared["pose"]
                    wp_idx    = nav.shared["wp_idx"]
                    speed     = nav.shared["speed"]
                    rec       = nav.shared["recovery_status"]
                    pol_stat  = nav.shared.get("policy_status", "?")
                    wp_lbl    = (nav._waypoints[wp_idx].label
                                 if wp_idx < len(nav._waypoints) else "DONE")
                print(
                    f"[{step:5d}] "
                    f"pos=({pose.x:+6.1f},{pose.y:+6.1f},{pose.z:+5.1f}) "
                    f"yaw={math.degrees(pose.yaw):+5.1f}° "
                    f"v={speed:.2f}m/s "
                    f"WP[{wp_idx}]={wp_lbl} "
                    f"policy={pol_stat} "
                    f"state={rec}"
                )

    except KeyboardInterrupt:
        print("\n[navigate] Interrupted.")

    finally:
        nav.stop_dashboard()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
