#!/usr/bin/env python3
# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Go2W terrain capability evaluation script.

Loads a trained rough-terrain checkpoint and runs it on 36 individually
parameterised terrain variants (fixed step heights, slopes, box heights,
noise amplitudes).  Reports a table of metrics that shows exactly where
the policy succeeds and where it gives up.

Metrics per variant
--------------------
  tracking_ratio  — mean(actual forward vel) / commanded vel (1.0 = perfect)
  moving_frac     — fraction of steps where forward vel > 0.1 m/s
  survival_rate   — % of episodes that reach timeout (vs falling over)
  mean_ep_len     — mean episode length in steps
  mean_dist_m     — estimated distance traveled per episode (m)

A '←' flag marks any variant where tracking < 0.50 or moving < 50%.

Architecture
-------------
Each variant is evaluated in its own subprocess (fresh Isaac Sim session).
This avoids the known Isaac Sim limitation where destroying and recreating
physics scenes in the same process hangs indefinitely.

Usage
------
  # Headless sweep — all 36 variants
  python scripts/eval.py \\
      --checkpoint logs/rsl_rl/go2w_velocity_rough/YYYY-MM-DD_HH-MM-SS/model_2999.pt

  # Headless — one terrain group only
  python scripts/eval.py --checkpoint <path> --group stairs_up

  # Headless — single named variant
  python scripts/eval.py --checkpoint <path> --terrain stairs_up_15cm

  # Visual mode — watch robots on a specific terrain (GUI, no metrics saved)
  python scripts/eval.py --checkpoint <path> --visual --terrain stairs_up_15cm

  # Visual mode — default terrain (stairs_up_10cm) if --terrain not specified
  python scripts/eval.py --checkpoint <path> --visual

Available terrain groups: stairs_up, stairs_down, boxes, slope, rough

Notes
------
• Each variant runs args.steps physics-policy steps (default 1 000 ≈ 20 s at 50 Hz)
  over args.num_envs parallel robots (default 50).
• The CSV is saved to logs/eval_results/eval_TIMESTAMP.csv by default.
• Per-variant timeout defaults to 300 s (--timeout).  A hung variant is killed
  and recorded as TIMEOUT in the results table.
"""

# ---------------------------------------------------------------------------
# NOTE: argument parsing and source-path setup MUST happen before AppLauncher
# is imported, because AppLauncher parses sys.argv when it is first imported.
# ---------------------------------------------------------------------------

import argparse
import json
import os
import subprocess
import sys
import time

# Make the rexmi_rl package importable (needed here only for EVAL_VARIANTS list)
_SOURCE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "source"
)
if _SOURCE_DIR not in sys.path:
    sys.path.insert(0, _SOURCE_DIR)

# ---------------------------------------------------------------------------
# Argument parsing (before AppLauncher so we control the headless flag)
# ---------------------------------------------------------------------------
# We import AppLauncher lazily (only in --_single mode) so the orchestrator
# process (sweep mode) never starts Isaac Sim at all.

parser = argparse.ArgumentParser(
    description="Go2W terrain capability evaluation",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    metavar="PATH",
    help="Path to the trained .pt checkpoint (e.g. logs/rsl_rl/.../model_2999.pt).",
)
parser.add_argument(
    "--visual",
    action="store_true",
    default=False,
    help=(
        "Run in GUI mode (Isaac Sim window).  Shows robots in real time. "
        "Use --terrain to pick which variant to display. "
        "Runs until you close the window; no metrics are saved."
    ),
)
parser.add_argument(
    "--terrain",
    type=str,
    default=None,
    metavar="NAME",
    help=(
        "Terrain variant to run, e.g. 'stairs_up_15cm' or 'slope_20deg'. "
        "Required for --visual.  If omitted in headless mode, all variants run."
    ),
)
parser.add_argument(
    "--group",
    type=str,
    default=None,
    metavar="GROUP",
    help=(
        "Run only one terrain group (headless only). "
        "Choices: stairs_up | stairs_down | boxes | slope | rough"
    ),
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=50,
    metavar="N",
    help="Parallel environments per variant (default: 50).",
)
parser.add_argument(
    "--steps",
    type=int,
    default=1000,
    metavar="N",
    help=(
        "Simulation steps per variant in headless mode (default: 1000 ≈ 20 s). "
        "More steps → more complete episodes → more reliable survival_rate."
    ),
)
parser.add_argument(
    "--timeout",
    type=int,
    default=300,
    metavar="SECS",
    help="Per-variant subprocess timeout in seconds (default: 300).  "
         "A hung variant is killed and marked TIMEOUT.",
)
parser.add_argument(
    "--out",
    type=str,
    default=None,
    metavar="PATH",
    help="CSV output path.  Defaults to logs/eval_results/eval_TIMESTAMP.csv.",
)
# Hidden flag: run a single variant inside a fresh Isaac Sim process and emit
# a RESULT_JSON line, then exit.  Used internally by the sweep orchestrator.
parser.add_argument("--_single", type=str, default=None, help=argparse.SUPPRESS)

args, _unknown = parser.parse_known_args()

# ---------------------------------------------------------------------------
# ORCHESTRATOR MODE — no Isaac Sim, just spawn subprocesses
# ---------------------------------------------------------------------------
if args._single is None and not args.visual:

    import csv
    import datetime

    # Hard-coded variant name list — mirrors EVAL_VARIANTS in eval_env_cfg.py.
    # We do NOT import eval_env_cfg here because it imports isaaclab.terrains
    # at module level which requires omni.log (only available inside Isaac Sim).
    _ALL_VARIANT_NAMES: list[str] = (
        # stairs_up (9)
        [f"stairs_up_{s}cm"   for s in [3, 5, 8, 10, 12, 15, 18, 20, 23]] +
        # stairs_down (9)
        [f"stairs_down_{s}cm" for s in [3, 5, 8, 10, 12, 15, 18, 20, 23]] +
        # boxes (6)
        [f"boxes_{h}cm"       for h in [3, 5, 8, 10, 15, 20]] +
        # slope (7)
        [f"slope_{d}deg"      for d in [2, 5, 8, 10, 15, 20, 23]] +
        # rough (5)
        [f"rough_{n}cm"       for n in [2, 4, 6, 8, 10]]
    )
    _VALID_GROUPS = ["stairs_up", "stairs_down", "boxes", "slope", "rough"]

    if not os.path.isfile(args.checkpoint):
        print(f"[eval] ERROR: checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Select variants
    if args.terrain is not None:
        variants_to_run = [n for n in _ALL_VARIANT_NAMES if n == args.terrain]
        if not variants_to_run:
            print(f"[eval] ERROR: Unknown terrain variant '{args.terrain}'.")
            print("[eval] Available variants:")
            for name in _ALL_VARIANT_NAMES:
                print(f"         {name}")
            sys.exit(1)
    elif args.group is not None:
        variants_to_run = [n for n in _ALL_VARIANT_NAMES if n.startswith(args.group)]
        if not variants_to_run:
            print(f"[eval] ERROR: No variants found for group '{args.group}'.")
            print(f"[eval] Valid groups: {_VALID_GROUPS}")
            sys.exit(1)
    else:
        variants_to_run = list(_ALL_VARIANT_NAMES)

    n_total = len(variants_to_run)
    print(f"\n[eval] Evaluating {n_total} terrain variant(s) — one subprocess each")
    print(f"[eval] Checkpoint : {args.checkpoint}")
    print(f"[eval] Steps/var  : {args.steps}  |  Envs/var: {args.num_envs}  |  Timeout: {args.timeout}s")
    print()

    # Build base command (pass all relevant flags through to child)
    _this_script = os.path.abspath(__file__)
    _python = sys.executable
    _base_cmd = [
        _python, _this_script,
        "--checkpoint", args.checkpoint,
        "--num_envs", str(args.num_envs),
        "--steps", str(args.steps),
        "--headless",
    ]

    results = []
    sweep_t0 = time.time()

    for idx, var_name in enumerate(variants_to_run, start=1):
        cmd = _base_cmd + ["--_single", var_name]
        print(f"  [{idx:2d}/{n_total}] {var_name} ... ", end="", flush=True)
        t0 = time.time()

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=args.timeout,
            )
            elapsed = time.time() - t0

            # Find the RESULT_JSON line in stdout
            result_dict = None
            for line in proc.stdout.splitlines():
                if line.startswith("RESULT_JSON:"):
                    try:
                        result_dict = json.loads(line[len("RESULT_JSON:"):].strip())
                    except json.JSONDecodeError:
                        pass
                    break

            if result_dict is not None:
                r = result_dict
                flag = "  ←" if (r["tracking_ratio"] < 0.50 or r["moving_frac"] < 0.50) else ""
                print(
                    f"done ({elapsed:.0f}s)"
                    f"  tracking={r['tracking_ratio']:.2f}"
                    f"  moving={r['moving_frac']:.0%}"
                    f"  survival={r['survival_rate']:.0f}%"
                    f"{flag}"
                )
                results.append(r)
            else:
                print(f"ERROR (no result line) after {elapsed:.0f}s")
                # Print last few stderr lines for diagnosis
                stderr_tail = proc.stderr.strip().splitlines()[-5:]
                for ln in stderr_tail:
                    print(f"       stderr: {ln}")
                results.append({
                    "name": var_name,
                    "tracking_ratio": -1.0,
                    "moving_frac": -1.0,
                    "survival_rate": -1.0,
                    "mean_ep_len": -1.0,
                    "mean_dist_m": -1.0,
                    "note": "ERROR",
                })

        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            print(f"TIMEOUT after {elapsed:.0f}s — variant skipped")
            results.append({
                "name": var_name,
                "tracking_ratio": -1.0,
                "moving_frac": -1.0,
                "survival_rate": -1.0,
                "mean_ep_len": -1.0,
                "mean_dist_m": -1.0,
                "note": "TIMEOUT",
            })

        except Exception as exc:
            elapsed = time.time() - t0
            print(f"EXCEPTION after {elapsed:.0f}s: {exc}")
            results.append({
                "name": var_name,
                "tracking_ratio": -1.0,
                "moving_frac": -1.0,
                "survival_rate": -1.0,
                "mean_ep_len": -1.0,
                "mean_dist_m": -1.0,
                "note": f"EXCEPTION: {exc}",
            })

    sweep_elapsed = time.time() - sweep_t0
    print(f"\n[eval] Sweep complete in {sweep_elapsed/60:.1f} min\n")

    # -----------------------------------------------------------------------
    # Print results table
    # -----------------------------------------------------------------------
    GROUPS = ["stairs_up", "stairs_down", "boxes", "slope", "rough"]
    sep = "─" * 80

    print("=" * 80)
    print("  Go2W Terrain Capability Evaluation Results")
    print("=" * 80)

    current_group = None
    for r in results:
        group = next((g for g in GROUPS if r["name"].startswith(g)), "other")
        if group != current_group:
            current_group = group
            print(sep)
            print(f"  {group.upper().replace('_', ' ')}")
            print(sep)
            print(
                f"  {'Variant':<26} "
                f"{'Tracking':>9} "
                f"{'Moving':>8} "
                f"{'Survival':>9} "
                f"{'Ep.len':>7} "
                f"{'Dist(m)':>7}"
            )
            print(sep)

        note = r.get("note", "")
        if note in ("TIMEOUT", "ERROR") or note.startswith("EXCEPTION"):
            print(f"  {r['name']:<26}  {note}")
            continue

        flag = ""
        if r["tracking_ratio"] < 0.50 or r["moving_frac"] < 0.50:
            flag = "  ←"

        print(
            f"  {r['name']:<26} "
            f"{r['tracking_ratio']:>9.2f} "
            f"{r['moving_frac']:>7.0%} "
            f"{r['survival_rate']:>8.0f}% "
            f"{r['mean_ep_len']:>7.0f} "
            f"{r['mean_dist_m']:>7.1f}"
            f"{flag}"
        )

    print("=" * 80)
    print("  Tracking = mean fwd vel / 0.5 m/s   Moving = % steps with vel > 0.1 m/s")
    print("  ← tracking < 0.50 or moving < 50%\n")

    # -----------------------------------------------------------------------
    # Save CSV
    # -----------------------------------------------------------------------
    if results:
        if args.out:
            csv_path = args.out
        else:
            ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            out_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "logs", "eval_results",
            )
            os.makedirs(out_dir, exist_ok=True)
            csv_path = os.path.join(out_dir, f"eval_{ts}.csv")

        all_keys = list(results[0].keys())
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)

        print(f"[eval] Results saved → {csv_path}\n")

    sys.exit(0)


# ---------------------------------------------------------------------------
# SINGLE-VARIANT MODE (--_single) or VISUAL MODE
# Both of these start Isaac Sim inside this process.
# ---------------------------------------------------------------------------

# Now it's safe to import AppLauncher and boot Isaac Sim
from isaaclab.app import AppLauncher  # noqa: E402

# Re-parse with AppLauncher flags included so --headless etc. are honoured
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if not args.visual:
    args.headless = True

launcher = AppLauncher(args)
sim_app = launcher.app

# ---------------------------------------------------------------------------
# Isaac Sim is running — safe to import simulation modules
# ---------------------------------------------------------------------------
import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import rexmi_rl  # noqa: F401, E402

from rexmi_rl.tasks.locomotion.velocity.config.go2w.eval_env_cfg import EVAL_VARIANTS  # noqa: E402
from rexmi_rl.tasks.locomotion.velocity.config.go2w.agents.rsl_rl_ppo_cfg import (  # noqa: E402
    Go2wRoughPPORunnerCfg,
)

if not os.path.isfile(args.checkpoint):
    print(f"[eval] ERROR: checkpoint not found: {args.checkpoint}")
    sim_app.close()
    sys.exit(1)

_device = getattr(args, "device", "cuda") or "cuda"
if not torch.cuda.is_available():
    _device = "cpu"

_COMMANDED_VEL = 0.5
_POLICY_DT     = 0.02


def _build_env_and_policy(cfg_fn, checkpoint_path: str):
    """Create env + load policy. Returns (env, policy)."""
    env_cfg = cfg_fn()
    env_cfg.scene.num_envs = args.num_envs
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_cols = max(args.num_envs, 50)

    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    agent_cfg = Go2wRoughPPORunnerCfg()
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=_device)
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=_device)
    return env, policy


# ---------------------------------------------------------------------------
# VISUAL MODE
# ---------------------------------------------------------------------------
if args.visual:
    from rexmi_rl.tasks.locomotion.velocity.config.go2w.eval_env_cfg import EVAL_VARIANTS

    if args.terrain is not None:
        variants_to_show = [(n, f) for n, f in EVAL_VARIANTS if n == args.terrain]
        if not variants_to_show:
            print(f"[eval] ERROR: Unknown terrain variant '{args.terrain}'.")
            sim_app.close()
            sys.exit(1)
    else:
        default_name = "stairs_up_10cm"
        variants_to_show = [(n, f) for n, f in EVAL_VARIANTS if n == default_name]
        if not variants_to_show:
            variants_to_show = [EVAL_VARIANTS[0]]
        print(
            f"[eval] --visual without --terrain: showing '{variants_to_show[0][0]}'.\n"
            f"[eval] Use --terrain <name> to pick a specific variant."
        )

    vis_name, vis_fn = variants_to_show[0]
    print(f"\n[eval] VISUAL MODE — terrain: {vis_name}")
    print(f"[eval] Checkpoint: {args.checkpoint}")
    print("[eval] Close the Isaac Sim window to exit.\n")

    env, policy = _build_env_and_policy(vis_fn, args.checkpoint)
    obs, _ = env.get_observations()
    while sim_app.is_running():
        with torch.no_grad():
            actions = policy(obs)
        obs, _rew, _dones, _info = env.step(actions)

    env.close()
    sim_app.close()
    sys.exit(0)


# ---------------------------------------------------------------------------
# SINGLE-VARIANT HEADLESS MODE  (spawned by the orchestrator)
# ---------------------------------------------------------------------------
var_name = args._single
cfg_lookup = {n: f for n, f in EVAL_VARIANTS}
if var_name not in cfg_lookup:
    print(f"[eval] ERROR: unknown variant '{var_name}'")
    sim_app.close()
    sys.exit(1)

cfg_fn = cfg_lookup[var_name]

env, policy = _build_env_and_policy(cfg_fn, args.checkpoint)

num_e  = env.unwrapped.num_envs
robot  = env.unwrapped.scene["robot"]

_EP_MAX_STEPS    = env.unwrapped.max_episode_length
_SURVIVAL_THRESH = max(int(_EP_MAX_STEPS * 0.99), _EP_MAX_STEPS - 10)
_MOVING_THRESH   = 0.1

ep_count     = torch.zeros(num_e, dtype=torch.long,  device=_device)
ep_survived  = torch.zeros(num_e, dtype=torch.long,  device=_device)
vel_accum    = torch.zeros(num_e, dtype=torch.float, device=_device)
moving_steps = torch.zeros(num_e, dtype=torch.long,  device=_device)
step_count   = torch.zeros(num_e, dtype=torch.long,  device=_device)
ep_step_buf  = torch.zeros(num_e, dtype=torch.long,  device=_device)

obs, _ = env.get_observations()

for _step in range(args.steps):
    with torch.no_grad():
        actions = policy(obs)
    obs, _rew, dones, _info = env.step(actions)

    fwd_vel       = robot.data.root_lin_vel_b[:, 0].clamp(min=0.0)
    vel_accum    += fwd_vel
    moving_steps += (fwd_vel > _MOVING_THRESH).long()
    step_count   += 1
    ep_step_buf  += 1

    if dones.any():
        survived_mask = (ep_step_buf >= _SURVIVAL_THRESH) & dones
        ep_survived  += survived_mask.long()
        ep_count     += dones.long()
        ep_step_buf   = ep_step_buf * (~dones)

# Metrics
total_steps  = step_count.float()
mean_vel     = (vel_accum / total_steps.clamp(min=1)).mean().item()
tracking     = min(mean_vel / _COMMANDED_VEL, 1.0)
moving_frac  = (moving_steps.float() / total_steps.clamp(min=1)).mean().item()
total_ep     = ep_count.sum().item()
total_surv   = ep_survived.sum().item()
survival_pct = (total_surv / max(total_ep, 1)) * 100.0
mean_ep_len  = (total_steps.sum().item() / total_ep) if total_ep > 0 else float(args.steps)
mean_dist    = tracking * _COMMANDED_VEL * mean_ep_len * _POLICY_DT

result = {
    "name":           var_name,
    "tracking_ratio": round(tracking, 3),
    "moving_frac":    round(moving_frac, 3),
    "survival_rate":  round(survival_pct, 1),
    "mean_ep_len":    round(mean_ep_len, 1),
    "mean_dist_m":    round(mean_dist, 2),
}

# Emit the result on a dedicated line so the orchestrator can parse it cleanly
print(f"RESULT_JSON: {json.dumps(result)}", flush=True)

env.close()
if _device.startswith("cuda"):
    torch.cuda.empty_cache()

sim_app.close()
sys.exit(0)
