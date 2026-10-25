#!/usr/bin/env bash
# Experimental persistent teleop-button autonomy. No recorded commands.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
mkdir -p outputs/remote_autonomy
remote_run_dir="$(mktemp -d "$repo_root/outputs/remote_autonomy/round-trip-XXXXXXXX")"
echo "Remote round-trip logs: $remote_run_dir"
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6:/usr/lib/x86_64-linux-gnu/libgcc_s.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
exec ./run.sh scripts/navigate.py \
  --task RexmiRl-Go2w-Crater-Bowl-RockySlope-Play-v0 \
  --checkpoint logs/rsl_rl/go2w_velocity_rocky_slope/2026-06-30_09-31-48/model_13994.pt \
  --nav_controller ascent --ascent_architecture remote --ascent_mode characterize \
  --mission round_trip --seed 42 \
  --terrain_prior outputs/ascent_full_survey/trial01/run.prior.npz \
  --cruise_speed .40 --max_side_slope_deg 32 --max_steps 45000 \
  --joint_log_dir "$remote_run_dir/joints" --log_file "$remote_run_dir/run.csv" "$@"
