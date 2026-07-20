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

    # Checkpoints — all three required for auto-switching; spin is optional
    p.add_argument("--task", required=True,
                   help="Isaac Lab gym task name")
    p.add_argument("--ckpt_fast_flat", required=False, default=None,
                   help="Checkpoint for fast_flat policy (.pt)")
    p.add_argument("--ckpt_rough",     required=False, default=None,
                   help="Checkpoint for rough policy (.pt)")
    p.add_argument("--ckpt_rocky",     required=False, default=None,
                   help="Checkpoint for rocky_slope policy (.pt)")
    p.add_argument("--ckpt_turn",      required=False, default=None,
                   help="Checkpoint for turn-in-place policy (.pt). "
                        "Optional — if omitted, turn-override falls back to rough policy. "
                        "Train with: python scripts/train.py "
                        "--task RexmiRl-Go2w-Velocity-Turn-v0 --headless "
                        "--load_run go2w_velocity_rough/2026-06-14_20-03-41 "
                        "--checkpoint model_8996.pt --max_iterations 1500")

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
                   choices=["traverse", "survey", "rim_circuit", "velocity_goal"],
                   default="traverse")

    # velocity_goal — single flat-terrain waypoint for motion debugging
    p.add_argument("--goal_x",      type=float, default=0.0,
                   help="Target X for --mission velocity_goal (world frame, metres)")
    p.add_argument("--goal_y",      type=float, default=0.0,
                   help="Target Y for --mission velocity_goal (world frame, metres)")
    p.add_argument("--goal_radius", type=float, default=1.5,
                   help="Arrival radius for velocity_goal waypoint (m). Default 1.5")

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
    p.add_argument("--slam_icp_iter",   type=int,   default=10,
                   help="ICP iterations per SLAM update (default 10; more = slower but "
                        "more accurate; 30 was the original default before perf fix)")
    p.add_argument("--step_thresh",     type=float, default=0.20,
                   help="Costmap step threshold (m) — cells with step > this are blocked. "
                        "Default 0.20 m; lower = more conservative, higher = more permissive.")

    # Sim settings
    p.add_argument("--num_envs",    type=int,   default=1)
    p.add_argument("--device",      default="cuda:0")
    p.add_argument("--max_steps",   type=int,   default=15000,
                   help="Max sim steps (~300 s at 50 Hz)")
    p.add_argument("--no_dashboard", action="store_true")
    p.add_argument("--log_file",    default=None,
                   help="Path to write nav log (CSV: step,x,y,z,yaw,speed,wp_idx,policy,state). "
                        "Default: auto-generated under logs/nav/TIMESTAMP.log")
    p.add_argument("--perf_log",    action="store_true",
                   help="Print per-step nav.step() timing every 50 steps")

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

    # Keep velocity command debug_vis ENABLED (True is the default).
    # This renders the green/blue heading arrow above the robot in Isaac Sim.
    # No need to touch env_cfg.commands.base_velocity.debug_vis here.

    # Import and instantiate agent_cfg
    mod_a, cls_a = agent_cfg_entry.rsplit(":", 1)
    agent_cfg = getattr(importlib.import_module(mod_a), cls_a)()
    agent_cfg.device = args.device

    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ------------------------------------------------------------------
    # Velocity command debug visualisation — keep ENABLED so Isaac Sim
    # renders the green/blue heading arrow above the robot.
    # The callback is safe as long as we do not call env.reset() mid-run;
    # navigation never resets (only the RL policy's episode reset fires,
    # which does not invalidate the command tensor view in the nav env).
    print("[navigate] ✓ Velocity command debug_vis ENABLED (Isaac Sim heading arrow active)")

    # ------------------------------------------------------------------
    # CRITICAL: patch command manager ranges to allow omega steering.
    #
    # The crater env cfg (crater_env_cfg.py) fixes ang_vel_z = (0.0, 0.0)
    # for training stability — the policy was trained with a fixed forward
    # command so the terrain curriculum could focus on balance.
    #
    # At inference time (navigation) the command manager still uses these
    # ranges to NORMALISE the command before inserting it into the policy
    # observation.  With range=(0,0) any injected omega is clamped to 0
    # before the policy sees it → policy always generates straight-ahead
    # actions regardless of what nav injects.
    #
    # Fix: widen the command ranges to match the rocky_slope TRAINING
    # distribution so the policy sees in-distribution omega values.
    # rocky_slope was trained with ang_vel_z ∈ (-0.5, 0.5).
    # rough was trained with ang_vel_z ∈ (-1.0, 1.0).
    # We use (-1.0, 1.0) so the rough policy can also turn properly.
    # ------------------------------------------------------------------
    try:
        _cmd_cfg = env.unwrapped.command_manager.cfg
        _bv = getattr(_cmd_cfg, "base_velocity", None)
        if _bv is not None and hasattr(_bv, "ranges"):
            old_az = (_bv.ranges.ang_vel_z if hasattr(_bv.ranges, "ang_vel_z")
                      else "?")
            _bv.ranges.ang_vel_z = (-1.0, 1.0)
            print(f"[navigate] ✓ Patched command ang_vel_z: {old_az} → (-1.0, 1.0)")
            print(f"[navigate]   (without this patch, omega is clamped to 0 "
                  f"and the policy ignores all steering commands)")
        else:
            print("[navigate] WARNING: Could not patch command ranges — "
                  "base_velocity not found in command manager cfg")
    except Exception as _e:
        print(f"[navigate] WARNING: command range patch failed: {_e}")

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

        # Optional turn policy — provides clean in-place rotation when
        # |heading_error| > 75° or RecoveryFSM is rotating.
        # Trained with vx=0, vy=0, omega∈(-1,+1) on mixed flat+slope terrain.
        # If --ckpt_turn is not provided, PolicySelector falls back to ROUGH
        # for turn-override (rough has vx∈(-0.5,0.5) — partial vx=0 support).
        if args.ckpt_turn:
            print(f"[navigate] Loading turn policy from {args.ckpt_turn}")
            policies[PolicyMode.TURN] = _load_policy(
                args.ckpt_turn, env, agent_cfg, args.device
            )
            print(f"[navigate] ✓ Turn policy loaded — turn-override will use dedicated "
                  f"turn policy (vx=0, omega=±1 rad/s) instead of rough fallback")
        else:
            print("[navigate] NOTE: --ckpt_turn not provided — turn-override falls back "
                  "to rough policy (degraded turn quality). Train turn policy with:")
            print("[navigate]   python scripts/train.py --task RexmiRl-Go2w-Velocity-Turn-v0 "
                  "--headless --load_run go2w_velocity_rough/2026-06-14_20-03-41 "
                  "--checkpoint model_8996.pt --max_iterations 1500")

        selector = PolicySelector(policies, initial_mode=PolicyMode.ROCKY_SLOPE)
        print("[navigate] PolicySelector: auto mode (terrain-aware switching)")
        print(f"  fast_flat  : {args.ckpt_fast_flat}")
        print(f"  rough      : {args.ckpt_rough}")
        print(f"  rocky_slope: {args.ckpt_rocky}")
        if args.ckpt_turn:
            print(f"  turn       : {args.ckpt_turn}")
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
    if args.mission == "velocity_goal":
        print(f"[navigate] velocity_goal target: ({args.goal_x:.2f}, {args.goal_y:.2f}) "
              f"radius={args.goal_radius:.1f} m")

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

    # For velocity_goal, regenerate waypoints with the user-specified target.
    # Navigator.__init__ calls get_waypoints(mission) with default goal (0,0);
    # we override here after construction so the user's --goal_x/--goal_y is used.
    if args.mission == "velocity_goal":
        nav._waypoints = nav._mission_planner.get_waypoints(
            mission_enum,
            goal_x=args.goal_x,
            goal_y=args.goal_y,
            goal_radius=args.goal_radius,
        )
        nav._wp_idx = 0
        nav._wp_closest_dist = [math.inf] * len(nav._waypoints)
        # Update global planner goal to the actual user-specified target
        if nav._waypoints:
            wp0 = nav._waypoints[0]
            nav._global.set_goal(wp0.x, wp0.y)
        # velocity_goal uses fast_flat policy — flat terrain, no crater
        # Override initial selector mode to fast_flat for responsiveness
        if hasattr(selector, "_current_mode"):
            from rexmi_rl.nav.policy_selector import PolicyMode
            selector._current_mode = PolicyMode.FAST_FLAT
            selector._candidate    = PolicyMode.FAST_FLAT
    nav._local.vx_normal = args.vx_normal
    nav._local.vx_steep  = args.vx_steep
    nav.policy_selector  = selector

    # Apply CLI tuning to SLAM and costmap
    nav._slam.max_icp_iter = args.slam_icp_iter
    nav._omap.STEP_THRESH  = args.step_thresh
    print(f"[navigate] SLAM: max_icp_iter={args.slam_icp_iter}  "
          f"costmap: step_thresh={args.step_thresh:.2f} m")

    # Expose policy_status key in shared dict for dashboard
    nav.shared["policy_status"] = selector.status_str()

    # ------------------------------------------------------------------
    # 6. Start dashboard
    # ------------------------------------------------------------------
    if not args.no_dashboard:
        nav.start_dashboard()
        print("[navigate] Dashboard window launched")

    # ------------------------------------------------------------------
    # 7. Set up log file
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # CSV telemetry is ALWAYS written (every step, not every 50).
    # Full step-by-step telemetry enables post-run analysis (speed profiles,
    # policy-switch timing, SLAM convergence history, etc.).
    # Use --log_file to specify a custom path; default is logs/nav/nav_<ts>.csv.
    # ------------------------------------------------------------------
    import csv
    _log_dir = "logs/nav"
    os.makedirs(_log_dir, exist_ok=True)
    _ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    _log_path = args.log_file or os.path.join(_log_dir, f"nav_{_ts}.csv")
    _log_f = open(_log_path, "w", newline="", buffering=1)  # line-buffered
    _csv = csv.writer(_log_f)
    _csv.writerow(["step", "x", "y", "z", "yaw_deg", "speed",
                   "wp_idx", "wp_label", "policy", "state",
                   "slam_converged", "slam_map_size", "slam_rms",
                   "nav_step_ms"])
    print(f"[navigate] Nav telemetry → {_log_path}  (every step)")

    # ------------------------------------------------------------------
    # 8. Main sim loop
    # ------------------------------------------------------------------
    # Explicit reset + warm-up before the main loop.
    #
    # Isaac Lab's gym.make() creates the env and runs one physics step for
    # initialization, but the PhysX tensor view may still be in a transitional
    # state after the contact sensor setup (which deletes/recreates collision
    # prims).  Running 5 warm-up steps at zero action lets the physics engine
    # fully stabilize before nav starts writing commands.
    # Without this, the very first env.step() hits "Failed to set DOF actuation
    # forces in backend" because the tensor view was flagged invalid during
    # the contact sensor prim replacement at startup.
    obs, _ = env.reset()
    step   = 0
    _nav_step_times: list = []   # ring buffer for perf reporting

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
            _t0 = time.perf_counter()
            nav.step()
            _nav_ms = (time.perf_counter() - _t0) * 1000.0
            _nav_step_times.append(_nav_ms)
            if len(_nav_step_times) > 50:
                _nav_step_times.pop(0)

            step += 1

            # Mission complete check
            if nav._wp_idx >= len(nav._waypoints):
                print(f"[navigate] Mission '{args.mission}' COMPLETE at step {step}!")
                time.sleep(3.0)
                break

            # ------------------------------------------------------------------
            # Per-step telemetry snapshot (shared dict read under lock)
            # ------------------------------------------------------------------
            with nav._lock:
                pose      = nav.shared["pose"]
                wp_idx    = nav.shared["wp_idx"]
                speed     = nav.shared["speed"]
                rec       = nav.shared["recovery_status"]
                pol_stat  = nav.shared.get("policy_status", "?")
                slam_conv = nav.shared.get("slam_converged", False)
                slam_sz   = nav.shared.get("slam_map_size", 0)
                slam_rms  = nav.shared.get("slam_rms", 0.0)
                wp_lbl    = (nav._waypoints[wp_idx].label
                             if wp_idx < len(nav._waypoints) else "DONE")

            # CSV: write EVERY step for full-resolution post-run analysis.
            # buffering=1 (line-buffered) means each row is flushed immediately
            # without the overhead of an explicit _log_f.flush() call.
            avg_nav_ms = sum(_nav_step_times) / max(1, len(_nav_step_times))
            _csv.writerow([
                step,
                f"{pose.x:.3f}", f"{pose.y:.3f}", f"{pose.z:.3f}",
                f"{math.degrees(pose.yaw):.1f}",
                f"{speed:.3f}",
                wp_idx, wp_lbl,
                pol_stat, rec,
                int(slam_conv), slam_sz, f"{slam_rms:.4f}",
                f"{_nav_ms:.2f}",
            ])

            # Console status every 50 steps (~1 s) — avoids flooding terminal
            if step % 50 == 0:
                perf_str = f"  nav_step={avg_nav_ms:.1f}ms(avg50)" if args.perf_log else ""
                print(
                    f"[{step:5d}] "
                    f"pos=({pose.x:+6.1f},{pose.y:+6.1f},{pose.z:+5.1f}) "
                    f"yaw={math.degrees(pose.yaw):+5.1f}° "
                    f"v={speed:.2f}m/s "
                    f"WP[{wp_idx}]={wp_lbl} "
                    f"policy={pol_stat} "
                    f"slam={'ON' if slam_conv else 'BOOT'}({slam_sz}vox) "
                    f"state={rec}"
                    f"{perf_str}"
                )

    except KeyboardInterrupt:
        print("\n[navigate] Interrupted.")

    finally:
        if _log_f is not None:
            _log_f.close()
            print(f"[navigate] Log saved: {_log_path}")
        nav.stop_dashboard()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
