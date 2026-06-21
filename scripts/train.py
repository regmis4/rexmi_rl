#!/usr/bin/env python3
# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
REXMI RL Training Script — proper implementation with curriculum state save/restore.

This script replaces the thin wrapper (runpy.run_path to Isaac Lab's train.py) with a
full implementation that adds one critical feature: **curriculum state continuity**.

Problem with the old wrapper
-----------------------------
Isaac Lab's runner.load(checkpoint) restores ONLY the network weights.  The
``terrain_levels`` tensor — which tracks each robot's current curriculum difficulty
— lives in the environment and is ALWAYS initialised fresh (random 0→num_rows-1).
Every ``--load_run --checkpoint`` resume therefore:
  1. Starts from a random terrain_levels distribution
  2. Spends ~200 iterations re-converging to the policy's earned equilibrium
  3. Then has the SAME number of iterations at the hard levels as before

This makes iterative training ("make a change, resume, continue") wasteful and
prevents the curriculum from being a genuine progressive ladder.

Solution: curriculum state save/restore
-----------------------------------------
This script monkey-patches ``runner.save`` so that alongside every model_N.pt
checkpoint it also writes a ``model_N_curriculum.pt`` file containing the
terrain_levels tensor.  When loading a checkpoint, if the matching
``model_N_curriculum.pt`` exists, it is restored before training begins.

Usage
-----
All arguments are identical to Isaac Lab's train.py.  One new flag is added:

  # Start fresh (no checkpoint):
  python scripts/train.py --task RexmiRl-Go2w-Velocity-SteepSlope-v0 --headless

  # Resume with curriculum continuity (default):
  python scripts/train.py \\
      --task RexmiRl-Go2w-Velocity-SteepSlope-v0 --headless \\
      --load_run go2w_velocity_steep_slope/2026-06-19_22-37-58 \\
      --checkpoint model_1999.pt
  # → auto-loads model_1999_curriculum.pt from the SAME directory
  # → terrain_levels starts at ~4.77 (where the previous run ended)
  # → zero re-warming overhead

  # Resume weights only (skip curriculum restore):
  python scripts/train.py ... --no_curriculum_restore

Checkpoints saved per run
--------------------------
  logs/rsl_rl/<experiment>/<timestamp>/
    ├── model_0.pt              ← weights at iter 0
    ├── model_50.pt
    ├── ...
    ├── model_2999.pt
    ├── curriculum/             ← terrain_levels tensors (subdirectory avoids glob collision)
    │   ├── model_0.pt          ← terrain_levels at iter 0
    │   ├── model_50.pt
    │   └── model_2999.pt
    ├── params/
    └── events.out.tfevents

NOTE: curriculum files live in curriculum/ NOT next to model_*.pt.
This prevents Isaac Lab's get_checkpoint_path (glob model_*.pt) from
accidentally picking up a raw tensor instead of a weight dict.
"""

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# Step 1: Add rexmi_rl source to sys.path so gym.register() is triggered.
# This MUST happen before any gym.make() call or Isaac Lab's env lookup.
# ---------------------------------------------------------------------------
_SOURCE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "source")
if _SOURCE_DIR not in sys.path:
    sys.path.insert(0, _SOURCE_DIR)

# ---------------------------------------------------------------------------
# Step 2: Add Isaac Lab's RSL-RL scripts dir to sys.path.
# Isaac Lab's train.py does `import cli_args` — a local import from that dir.
# We need it importable before AppLauncher initialisation.
# ---------------------------------------------------------------------------
ISAACLAB_DIR = os.environ.get("ISAACLAB_DIR", os.path.expanduser("~/IsaacLab"))
_RSL_RL_SCRIPTS_DIR = os.path.join(ISAACLAB_DIR, "scripts", "reinforcement_learning", "rsl_rl")
if _RSL_RL_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _RSL_RL_SCRIPTS_DIR)

# ---------------------------------------------------------------------------
# Step 3: AppLauncher MUST be initialised before any other Isaac Lab imports.
# We import the class and cli_args first, parse args, then launch the app,
# then do all remaining imports inside the guarded block below.
# ---------------------------------------------------------------------------
from isaaclab.app import AppLauncher  # class import only — app not yet running
import cli_args  # noqa: E402  (from Isaac Lab's rsl_rl scripts dir)

# Build the argument parser, mirroring Isaac Lab's train.py exactly
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False,
                    help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200,
                    help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000,
                    help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None,
                    help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None,
                    help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point",
                    help="Name of the RL agent configuration entry point.")
parser.add_argument("--seed", type=int, default=None,
                    help="Seed used for the environment.")
parser.add_argument("--max_iterations", type=int, default=None,
                    help="RL Policy training iterations.")
parser.add_argument("--distributed", action="store_true", default=False,
                    help="Run training with multiple GPUs or nodes.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False,
                    help="Export IO descriptors.")
# REXMI addition: opt-out of curriculum state restore
parser.add_argument(
    "--no_curriculum_restore",
    action="store_true",
    default=False,
    help=(
        "Skip restoring curriculum (terrain_levels) state when loading a checkpoint. "
        "Use this when intentionally starting the curriculum from scratch with "
        "transferred weights (e.g. loading from a different terrain task)."
    ),
)
# Append RSL-RL CLI args (--resume, --load_run, --checkpoint, --logger, etc.)
cli_args.add_rsl_rl_args(parser)
# Append AppLauncher CLI args (--device, --headless, etc.)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()

# Video recording requires cameras to be enabled
if args_cli.video:
    args_cli.enable_cameras = True

# Clear sys.argv of our args — Hydra gets its own args from hydra_args
sys.argv = [sys.argv[0]] + hydra_args

# ---------------------------------------------------------------------------
# Launch Isaac Sim / Omniverse.  ALL subsequent IsaacLab imports must come
# AFTER this line.
# ---------------------------------------------------------------------------
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# All remaining imports — only valid after AppLauncher has initialised.
# ---------------------------------------------------------------------------
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from datetime import datetime  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import (  # noqa: E402
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.io import dump_pickle, dump_yaml  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402  (registers Isaac Lab built-in envs)
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import rexmi_rl  # noqa: F401, E402  (registers all REXMI gym environments)

# TF32 settings matching Isaac Lab's train.py
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


# ===========================================================================
# Curriculum state save / restore helpers
# ===========================================================================

def _get_terrain(env):
    """Return the TerrainImporter from the (unwrapped) environment, or None."""
    try:
        return env.unwrapped.scene["terrain"]
    except Exception:
        return None


def _curriculum_path(model_path: str) -> str:
    """Derive curriculum state path from a model checkpoint path.

    Curriculum files are stored in a ``curriculum/`` subdirectory of the run's
    log directory so they NEVER match the ``model_*.pt`` glob that Isaac Lab's
    ``get_checkpoint_path`` uses to auto-select the latest checkpoint.
    Without this separation, ``play.py --resume`` would pick up
    ``model_N_curriculum.pt`` (a raw 1-D tensor) as the "latest" checkpoint,
    causing ``load_state_dict`` to raise ``IndexError: too many indices``.

    ``logs/.../model_1999.pt`` → ``logs/.../curriculum/model_1999.pt``
    """
    dirpath = os.path.dirname(model_path)
    filename = os.path.basename(model_path)
    return os.path.join(dirpath, "curriculum", filename)


def _save_curriculum_state(terrain, model_path: str) -> None:
    """Save the terrain_levels tensor alongside a model checkpoint.

    Called by the monkey-patched runner.save() after every checkpoint write.
    Silently skips if the terrain importer does not expose terrain_levels.

    Saved file: ``<same-dir-as-model>/model_N_curriculum.pt``
    Content:    1-D int64 tensor of shape [num_envs,], dtype matching terrain_levels.
    """
    if terrain is None or not hasattr(terrain, "terrain_levels"):
        return
    cpath = _curriculum_path(model_path)
    os.makedirs(os.path.dirname(cpath), exist_ok=True)  # create curriculum/ dir
    torch.save(terrain.terrain_levels.cpu(), cpath)
    mean_lv = terrain.terrain_levels.float().mean().item()
    print(f"[REXMI] curriculum saved  mean_level={mean_lv:.3f}  → {os.path.basename(cpath)}")


def _restore_curriculum_state(terrain, checkpoint_path: str) -> bool:
    """Load and restore terrain_levels from a saved curriculum state file.

    Looks for ``model_N_curriculum.pt`` next to the loaded ``model_N.pt``.
    Returns True if the state was successfully restored, False otherwise.
    """
    if terrain is None or not hasattr(terrain, "terrain_levels"):
        return False
    cpath = _curriculum_path(checkpoint_path)
    if not os.path.isfile(cpath):
        print(
            f"[REXMI] No curriculum state at {os.path.basename(cpath)} — "
            "terrain_levels stays at random initial values."
        )
        return False
    saved = torch.load(cpath, map_location="cpu")
    terrain.terrain_levels.copy_(saved.to(terrain.terrain_levels.device))
    mean_lv = terrain.terrain_levels.float().mean().item()
    print(f"[REXMI] curriculum restored  mean_level={mean_lv:.3f}  ← {os.path.basename(cpath)}")
    return True


# ===========================================================================
# Main training function
# ===========================================================================

@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlOnPolicyRunnerCfg,
):
    """Train with RSL-RL agent, with curriculum state save/restore."""
    # Apply CLI overrides to agent and env configs
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)

    # Isaac Lab's update_rsl_rl_cfg only sets agent_cfg.resume when --resume is
    # explicitly passed.  Passing --load_run alone does NOT set resume=True, so
    # our checkpoint-loading block would be silently skipped.
    # We treat --load_run / --checkpoint as implicit --resume.
    if not agent_cfg.resume and (args_cli.load_run is not None or args_cli.checkpoint is not None):
        agent_cfg.resume = True

    env_cfg.scene.num_envs = (
        args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    )
    agent_cfg.max_iterations = (
        args_cli.max_iterations
        if args_cli.max_iterations is not None
        else agent_cfg.max_iterations
    )

    # Set environment seed and device
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )

    # Multi-GPU setup (mirrors Isaac Lab's train.py)
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # New timestamped log directory for this run
    log_root_path = os.path.abspath(
        os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    )
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # Resolve checkpoint path BEFORE creating the new log_dir.
    #
    # get_checkpoint_path(root, run_dir, checkpoint) scans `root` for child
    # directories whose NAME matches the `run_dir` regex.  If the user passes
    # --load_run go2w_velocity_rough/2026-06-14_20-03-41  (cross-experiment)
    # or  --load_run go2w_velocity_steep_slope/2026-06-20_11-32-10  (same but
    # fully-qualified), the slash means `run_dir` is "experiment/timestamp" —
    # which will never match a bare timestamp directory name.
    #
    # Fix: when load_run contains a '/', split off the experiment name and look
    # up logs/rsl_rl/<experiment> as the root instead of the current task's root.
    resume_path = None
    if agent_cfg.resume:
        load_run = agent_cfg.load_run  # may be None, ".*", "timestamp", or "exp/timestamp"
        load_ckpt = agent_cfg.load_checkpoint  # may be None or "model_N.pt"
        if load_ckpt is None:
            load_ckpt = ".*"
        if load_run and "/" in load_run:
            # Fully-qualified: "experiment_name/timestamp"
            exp_name, run_ts = load_run.split("/", 1)
            root = os.path.abspath(os.path.join("logs", "rsl_rl", exp_name))
        else:
            # Same-experiment: just a timestamp, a regex, or None → latest run
            root = log_root_path
            run_ts = load_run if load_run else ".*"
        resume_path = get_checkpoint_path(root, run_ts, load_ckpt)

    # Create and (optionally) wrap the environment
    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        import gymnasium as _gym
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        env = _gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Create the RSL-RL runner
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
    )
    runner.add_git_repo_to_log(__file__)

    # Load checkpoint (weights) and optionally restore curriculum state
    if agent_cfg.resume and resume_path is not None:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

        if not args_cli.no_curriculum_restore:
            terrain = _get_terrain(env)
            _restore_curriculum_state(terrain, resume_path)
        else:
            print("[REXMI] --no_curriculum_restore: terrain_levels left at random init.")

    # -----------------------------------------------------------------------
    # Monkey-patch runner.save to also write curriculum state alongside
    # every model checkpoint.  This hooks into BOTH the save_interval saves
    # (every 50 iters) and the final save at the end of runner.learn().
    # -----------------------------------------------------------------------
    terrain = _get_terrain(env)
    _orig_save = runner.save

    def _save_with_curriculum(path: str, infos=None) -> None:
        """Wrapper around runner.save that co-saves curriculum state."""
        # Call the original save (writes model_N.pt)
        if infos is not None:
            _orig_save(path, infos)
        else:
            _orig_save(path)
        # Write curriculum state alongside it
        _save_curriculum_state(terrain, path)

    runner.save = _save_with_curriculum

    # Dump full configs for reproducibility
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # Run training
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=True,
    )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
