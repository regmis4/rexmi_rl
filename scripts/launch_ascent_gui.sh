#!/usr/bin/env bash
# Reviewed ascent characterization with dashboard, overlays and fresh telemetry.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
mkdir -p outputs/ascent_user_check
ascent_run_dir="$(mktemp -d "$repo_root/outputs/ascent_user_check/ascent-XXXXXXXX")"
echo "Ascent logs: $ascent_run_dir"
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libgcc_s.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
exec ./run.sh scripts/navigate.py \
  --task RexmiRl-Go2w-Crater-Bowl-RockySlope-Play-v0 \
  --checkpoint logs/rsl_rl/go2w_velocity_rocky_slope/2026-06-30_09-31-48/model_13994.pt \
  --nav_controller ascent --mission ascent --seed 42 \
  --ascent_architecture behavior_tree --ascent_mode characterize \
  --terrain_prior outputs/ascent_full_survey/trial01/run.prior.npz \
  --ascent_uphill_speed_response --ascent_release_memory \
  --ascent_event_snapshots --perception_view --max_steps 45000 \
  --joint_log_dir "$ascent_run_dir/joints" --log_file "$ascent_run_dir/run.csv"
