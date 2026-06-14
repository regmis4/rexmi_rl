#!/usr/bin/env python3
# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Policy playback / visualisation entry point for REXMI RL.

This script loads a trained policy checkpoint and runs it in Isaac Sim so you
can watch the robot perform the learned behaviour in real time.

Usage
-----
  # Play the latest checkpoint (interactive GUI)
  python scripts/play.py --task RexmiRl-Go2w-Velocity-Flat-Play-v0

  # Play a specific checkpoint
  python scripts/play.py --task RexmiRl-Go2w-Velocity-Flat-Play-v0 \
      --load_run go2w_velocity_flat/2026-06-13_12-00-00 \
      --checkpoint model_300.pt

Or via the convenience launcher:
  ./run.sh scripts/play.py --task RexmiRl-Go2w-Velocity-Flat-Play-v0

Difference from train.py
-------------------------
* Uses the *_PLAY variant of the env config:
    - 50 environments instead of 4096
    - No sensor noise (observations reflect true state)
    - No random external forces
* The Isaac Sim viewport is displayed (not headless)
* No policy updates happen — the checkpoint is loaded read-only

What to look for
----------------
After ~300 training iterations on a flat plane you should see:
  - All 4 wheels spinning to drive the robot in the commanded direction
  - The base staying roughly level (flat_orientation reward working)
  - Smooth acceleration/deceleration (action_rate penalty working)
  - Robot turning to match commanded angular velocity

The velocity command arrows (green/red vectors above each robot) show
what direction and speed is being commanded.  If the robot tracks them
well, training was successful.

Checkpoint discovery
--------------------
Isaac Lab's play script automatically finds the latest checkpoint in:
  logs/rsl_rl/go2w_velocity_flat/
If you have multiple runs, use --load_run to specify which one.
"""

import sys
import os

# ---------------------------------------------------------------------------
# Same setup as train.py: ensure Isaac Lab path is findable and trigger
# gym.register() by importing rexmi_rl.
# ---------------------------------------------------------------------------
ISAACLAB_DIR = os.environ.get("ISAACLAB_DIR", os.path.expanduser("~/IsaacLab"))
ISAACLAB_RSL_RL_SCRIPT = os.path.join(
    ISAACLAB_DIR, "scripts", "reinforcement_learning", "rsl_rl", "play.py"
)

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "source")
if _SOURCE_DIR not in sys.path:
    sys.path.insert(0, _SOURCE_DIR)

import rexmi_rl  # noqa: F401 — registers all environments

import runpy

if not os.path.isfile(ISAACLAB_RSL_RL_SCRIPT):
    raise FileNotFoundError(
        f"Isaac Lab RSL-RL play script not found at: {ISAACLAB_RSL_RL_SCRIPT}\n"
        f"Make sure ISAACLAB_DIR is set correctly (current: {ISAACLAB_DIR})\n"
        f"You can set it with: export ISAACLAB_DIR=/path/to/IsaacLab"
    )

# Isaac Lab's play.py also does `import cli_args` (same local import as train.py).
# Add the rsl_rl scripts directory so cli_args.py is importable.
_rsl_rl_dir = os.path.dirname(ISAACLAB_RSL_RL_SCRIPT)
if _rsl_rl_dir not in sys.path:
    sys.path.insert(0, _rsl_rl_dir)

runpy.run_path(ISAACLAB_RSL_RL_SCRIPT, run_name="__main__")
