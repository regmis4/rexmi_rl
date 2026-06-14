#!/usr/bin/env python3
# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Training entry point for REXMI RL.

This script is a thin wrapper around Isaac Lab's official RSL-RL training script.
It ensures our rexmi_rl package is importable (which triggers gym.register() for
all our custom environments) before handing control to Isaac Lab's trainer.

Usage
-----
Activate Isaac Lab's Python environment first, then run:

  # Basic training (flat terrain, wheels only)
  python scripts/train.py --task RexmiRl-Go2w-Velocity-Flat-v0

  # With more environments (requires more GPU memory)
  python scripts/train.py --task RexmiRl-Go2w-Velocity-Flat-v0 --num_envs 4096

  # Resume from a checkpoint
  python scripts/train.py --task RexmiRl-Go2w-Velocity-Flat-v0 --resume

  # Headless mode (no GUI window — faster, use for long training runs)
  python scripts/train.py --task RexmiRl-Go2w-Velocity-Flat-v0 --headless

Or use the convenience launcher (handles the IsaacLab Python path automatically):
  ./run.sh scripts/train.py --task RexmiRl-Go2w-Velocity-Flat-v0 --num_envs 4096

How training works
------------------
1. Isaac Sim launches (PhysX GPU pipeline initialises)
2. The environment is created: N copies of the robot + terrain + sensors
3. RSL-RL PPO runner starts the training loop:
   a. Collect rollout: run policy for 24 steps across all envs
   b. Compute returns and GAE advantages
   c. Update actor + critic networks (5 epochs, 4 mini-batches)
   d. Log metrics to TensorBoard
   e. Save checkpoint every 50 iterations
4. Final policy saved to:  logs/rsl_rl/go2w_velocity_flat/<timestamp>/

Checkpoints and logs
--------------------
  logs/rsl_rl/go2w_velocity_flat/<timestamp>/
    ├── model_<iter>.pt      # policy + value function weights
    ├── config.json          # full env + runner config snapshot
    └── events.out.tfevents  # TensorBoard log

View training curves:
  tensorboard --logdir logs/rsl_rl/go2w_velocity_flat
"""

import sys
import os

# ---------------------------------------------------------------------------
# Step 1: Ensure Isaac Lab's scripts directory is on the Python path so we
# can import its training utilities directly.
# ---------------------------------------------------------------------------
# We read ISAACLAB_DIR from the environment (set by run.sh) or fall back to
# the default install location.
ISAACLAB_DIR = os.environ.get("ISAACLAB_DIR", os.path.expanduser("~/IsaacLab"))
ISAACLAB_RSL_RL_SCRIPT = os.path.join(
    ISAACLAB_DIR, "scripts", "reinforcement_learning", "rsl_rl", "train.py"
)

# ---------------------------------------------------------------------------
# Step 2: Import rexmi_rl to trigger gym.register() for all our environments.
# This MUST happen before any gym.make() call or Isaac Lab's env lookup.
# ---------------------------------------------------------------------------
# Add the source directory to sys.path in case the package isn't installed yet.
_SOURCE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "source")
if _SOURCE_DIR not in sys.path:
    sys.path.insert(0, _SOURCE_DIR)

import rexmi_rl  # noqa: F401 — side effect: registers all gym environments

# ---------------------------------------------------------------------------
# Step 3: Delegate to Isaac Lab's official RSL-RL train script.
#
# We use runpy.run_path which executes the script in a fresh namespace but
# inherits the current sys.argv, so all command-line arguments (--task,
# --num_envs, --headless, etc.) are passed through transparently.
# ---------------------------------------------------------------------------
import runpy

if not os.path.isfile(ISAACLAB_RSL_RL_SCRIPT):
    raise FileNotFoundError(
        f"Isaac Lab RSL-RL train script not found at: {ISAACLAB_RSL_RL_SCRIPT}\n"
        f"Make sure ISAACLAB_DIR is set correctly (current: {ISAACLAB_DIR})\n"
        f"You can set it with: export ISAACLAB_DIR=/path/to/IsaacLab"
    )

# ---------------------------------------------------------------------------
# Step 4: Add Isaac Lab's rsl_rl scripts directory to sys.path.
#
# Isaac Lab's train.py does `import cli_args` — a LOCAL import that expects
# cli_args.py to be importable from the same folder as train.py itself:
#   /home/susan/IsaacLab/scripts/reinforcement_learning/rsl_rl/cli_args.py
#
# runpy.run_path() does NOT automatically add the script's directory to
# sys.path (unlike running a script directly), so we must do it manually.
# ---------------------------------------------------------------------------
_rsl_rl_dir = os.path.dirname(ISAACLAB_RSL_RL_SCRIPT)
if _rsl_rl_dir not in sys.path:
    sys.path.insert(0, _rsl_rl_dir)

# Run the Isaac Lab training script with our sys.argv intact
runpy.run_path(ISAACLAB_RSL_RL_SCRIPT, run_name="__main__")
