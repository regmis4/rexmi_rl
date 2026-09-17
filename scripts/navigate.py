#!/usr/bin/env python3
# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""REXMI navigation entrypoint.

Default: observed-route checkpoint navigation using the teleop rocky_slope
policy and command adapter. See docs/checkpoint_navigation.md for launch,
manual takeover, validation status and legacy comparison.

Pass --nav_controller legacy for the original planner/recovery/policy-selector
implementation. Its configuration flags and behavior remain available below.
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

    p.add_argument("--nav_controller", choices=["checkpoint", "legacy"], default="checkpoint")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--capture_file", default=None, help="Optional viewport PNG for a visible test run")
    p.add_argument("--scan_log", default=None,
                   help="Optional directory for raw world-frame scans for repeatable mapping comparisons")
    p.add_argument("--capture_step", type=int, default=750)
    p.add_argument("--cruise_speed", type=float, default=0.4,
                   help="Forward ceiling, at most 0.8; above 0.4 is experimental")
    p.add_argument("--start_paused", action="store_true")
    p.add_argument("--spawn_y", type=float, default=0.0)
    p.add_argument("--spawn_z", type=float, default=4.50)
    p.add_argument("--spawn_yaw", type=float, default=math.pi)
    # Checkpoints — all three required for auto-switching; spin is optional
    p.add_argument("--task", required=True,
                   help="Isaac Lab gym task name")
    # Conservative crater demo: rough + rocky_slope + pulse turn only (no fast_flat).
    p.add_argument("--ckpt_rough",     required=False, default=None,
                   help="Checkpoint for rough policy (.pt)")
    p.add_argument("--ckpt_rocky",     required=False, default=None,
                   help="Checkpoint for rocky_slope policy (.pt)")
    p.add_argument("--enable_turn", action="store_true", default=True,
                   help="Enable iso-style 13345 turn (default ON)")
    p.add_argument("--no_turn", action="store_true",
                   help="Disable turn policy — path-only FOLLOW")
    p.add_argument("--ckpt_turn",      required=False, default=None,
                   help="Direct slope-turn ckpt (continuous w=+/-0.08). Default: "
                        "logs/rsl_rl/go2w_velocity_slope_turn/"
                        "2026-07-27_20-54-25/model_13345.pt")
    # Deprecated aliases (ignored with warning)
    p.add_argument("--ckpt_fast_flat", required=False, default=None,
                   help=argparse.SUPPRESS)
    p.add_argument("--ckpt_turn_flat", required=False, default=None,
                   help=argparse.SUPPRESS)


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
                   choices=["auto", "rough", "rocky_slope"],
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
    p.add_argument("--survey_layer",choices=["terrain","slope","roughness","traversal"],default="terrain")
    p.add_argument("--perception_view", action="store_true",
                   help="Live in-scene LiDAR, observed terrain costs and route overlay")
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
    if args.nav_controller == "checkpoint":
        checkpoint = args.ckpt_rocky or args.checkpoint
        if not checkpoint or not os.path.isfile(checkpoint):
            raise SystemExit("Checkpoint navigation requires an existing rocky_slope --checkpoint or --ckpt_rocky.")
        if args.policy_mode == "rough":
            raise SystemExit("Use --nav_controller legacy for the rough policy.")
        if args.max_steps < 1 or args.step_thresh <= 0 or not .1 <= args.cruise_speed <= .8:
            raise SystemExit("Require max_steps > 0, step_thresh > 0 and cruise_speed between 0.1 and 0.8.")

    # ------------------------------------------------------------------
    # 1. Boot Isaac Sim
    # ------------------------------------------------------------------
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app
    if args.nav_controller == "checkpoint":
        from rexmi_rl.nav.checkpoint_runner import run
        return run(args, simulation_app)

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

    # --- Nav-safe spawn + no fall terminations (always, even if task uses non-PLAY cfg) ---
    try:
        import math as _math
        env_cfg.events.reset_base.params = {
            "pose_range": {
                "x": (13.0, 13.0),
                "y": (0.0, 0.0),
                "yaw": (_math.pi, _math.pi),
                "z": (4.48, 4.52),  # just above exterior mesh ~4.12 + 0.35
            },
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        }
        print("[navigate] ✓ Spawn pinned: x=13 y=0 yaw=π z≈4.50 (no air-drop)")
    except Exception as _e:
        print(f"[navigate] WARNING: could not pin spawn: {_e}")
    try:
        if hasattr(env_cfg, "terminations"):
            if hasattr(env_cfg.terminations, "base_contact"):
                env_cfg.terminations.base_contact = None
            if hasattr(env_cfg.terminations, "bad_orientation"):
                env_cfg.terminations.bad_orientation = None
            print("[navigate] ✓ Terminations base_contact/bad_orientation disabled for nav")
    except Exception as _e:
        print(f"[navigate] WARNING: could not disable terminations: {_e}")

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
            # Wide enough for rocky/rough; turn path clamps inject to ±0.08.
            # Also open lin_vel_x so vx=0.05 is not clipped if train range was fixed.
            _bv.ranges.ang_vel_z = (-1.0, 1.0)
            if hasattr(_bv.ranges, "lin_vel_x"):
                old_lx = _bv.ranges.lin_vel_x
                # Ensure 0.05 is inside range for turn plant
                lo = min(float(old_lx[0]) if isinstance(old_lx, (list, tuple)) else -0.1, 0.0)
                hi = max(float(old_lx[1]) if isinstance(old_lx, (list, tuple)) else 1.0, 0.5)
                _bv.ranges.lin_vel_x = (lo, hi)
            print(f"[navigate] Patched ang_vel_z: {old_az} → (-1.0, 1.0)")
            print(f"[navigate]   turn injects ω≤0.08 vx=0.05 (13345 train band)")
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
        missing = []
        if not args.ckpt_rough: missing.append("--ckpt_rough")
        if not args.ckpt_rocky: missing.append("--ckpt_rocky")
        if missing:
            if args.checkpoint:
                print("[navigate] WARNING: auto needs rough+rocky. "
                      "Falling back to fixed rocky_slope.")
                use_auto = False
                args.policy_mode = "rocky_slope"
                args.ckpt_rocky = args.checkpoint
            else:
                raise SystemExit(
                    f"[navigate] ERROR: --policy_mode auto requires: {missing}\n"
                    f"Conservative demo: rough + rocky + optional turn (model_13345)."
                )
        if args.ckpt_fast_flat:
            print("[navigate] NOTE: --ckpt_fast_flat ignored (conservative demo: no flat)")
        if args.ckpt_turn_flat:
            print("[navigate] NOTE: --ckpt_turn_flat ignored (use --ckpt_turn model_13345)")

    if use_auto:
        print("[navigate] Loading conservative policies: rough + rocky + turn")
        policies = {
            PolicyMode.ROUGH:       _load_policy(args.ckpt_rough, env, agent_cfg, args.device),
            PolicyMode.ROCKY_SLOPE: _load_policy(args.ckpt_rocky, env, agent_cfg, args.device),
        }
        # Direct continuous slope-turn (model_13345) — not Pulse20
        turn_path = args.ckpt_turn or (
            "logs/rsl_rl/go2w_velocity_slope_turn/"
            "2026-07-27_20-54-25/model_13345.pt"
        )
        import os as _os
        if _os.path.isfile(turn_path):
            print(f"[navigate] Loading direct turn from {turn_path}")
            policies[PolicyMode.TURN] = _load_policy(
                turn_path, env, agent_cfg, args.device
            )
            print("[navigate] Turn loaded: direct model_13345 (continuous w=+/-0.08)")
        else:
            print(f"[navigate] WARNING: turn ckpt missing ({turn_path}) — rough fallback")

        selector = PolicySelector(policies, initial_mode=PolicyMode.ROCKY_SLOPE)
        print("[navigate] PolicySelector: rough | rocky_slope | turn (no fast_flat)")
        print(f"  rough      : {args.ckpt_rough}")
        print(f"  rocky_slope: {args.ckpt_rocky}")
        print(f"  turn       : {turn_path}")

    else:
        # Single policy fixed mode
        mode_map = {
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
            PolicyMode.ROUGH:       fixed_policy,
            PolicyMode.ROCKY_SLOPE: fixed_policy,
        }
        selector = PolicySelector(policies, initial_mode=fixed_mode)
        _fixed = fixed_mode

        # Must accept the same kwargs Navigator passes (heading_error_rad, etc.)
        def _fixed_update(*_a, **_k):
            return _fixed

        selector.update = _fixed_update  # type: ignore[method-assign]
        selector.force_turn = lambda slope_ahead=0.0: _fixed  # type: ignore[method-assign]


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

    # Isolation-style turn ON by default. Pass --no_turn for path-only.
    if getattr(args, "no_turn", False):
        nav.enable_turn = False
        print("[navigate] --no_turn: path-only FOLLOW (13345 disabled)")
    else:
        nav.enable_turn = True
        print("[navigate] TURN enabled (iso handoff: brake→settle→13345→FOLLOW)")

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
        nav._wp_first_dist = [math.inf] * len(nav._waypoints)

        # Update global planner goal to the actual user-specified target
        if nav._waypoints:
            wp0 = nav._waypoints[0]
            nav._global.set_goal(wp0.x, wp0.y)
        # velocity_goal uses fast_flat policy — flat terrain, no crater
        # Override initial selector mode to fast_flat for responsiveness
        if hasattr(selector, "_current_mode"):
            from rexmi_rl.nav.policy_selector import PolicyMode
            selector._current_mode = PolicyMode.ROUGH
            selector._candidate    = PolicyMode.ROUGH
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
    perception_view = None
    if args.perception_view:
        from rexmi_rl.nav.perception_view import PerceptionView
        try:
            perception_view = PerceptionView(nav)
        except Exception as exc:
            print(f"[navigate] Perception overlay unavailable: {exc}")
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

            # During BOOT hold still: zero actions (cmd alone is not enough —
            # residual policy torque was spinning the robot during SLAM warm-up).
            with torch.no_grad():
                # Zero residual torques during BOOT and post soft-reset plant
                # Zero actions: BOOT, post soft-reset, and BRAKE (stop before turn).
                # Turn policy runs only after brake when reorient.active.
                # Zero actions: BOOT / plant / brake / clean turn handoff settle
                hold_still = getattr(nav, "_in_boot", False) or (
                    time.monotonic() < getattr(nav, "_post_soft_hold_until", 0.0)
                ) or (
                    time.monotonic() < getattr(nav, "_brake_until", 0.0)
                ) or getattr(nav, "_pending_reorient", False) or (
                    time.monotonic() < getattr(nav, "_settle_actions_until", 0.0)
                )
                if hold_still:
                    actions = torch.zeros_like(active_policy(obs))
                else:
                    actions = active_policy(obs)

            # Sim step
            obs, rewards, dones, infos = env.step(actions)

            # If env terminated (should be rare with terminations off), log it
            try:
                if bool(dones[0]):
                    print(f"[navigate] WARNING: env done at step {step} "
                          f"(termination still active?)")
            except Exception:
                pass

            # Nav tick — updates selector, injects velocity command
            _t0 = time.perf_counter()
            nav.step()
            if perception_view is not None:
                perception_view.update()
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
        if perception_view is not None:
            perception_view.close()
        nav.stop_dashboard()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
