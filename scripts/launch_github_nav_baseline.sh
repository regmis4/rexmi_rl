#!/usr/bin/env bash
# Exact GitHub navigation baseline; live LiDAR and operator-selected checkpoints.
set -euo pipefail
baseline=/home/susan/rexmi_rl_github_baseline
checkpoint=/home/susan/rexmi_rl/logs/rsl_rl/go2w_velocity_rocky_slope/2026-06-30_09-31-48/model_13994.pt
cd "$baseline"
if [[ "$(git rev-parse HEAD)" != 2e391f87de58766562a384ec78351d59c1e9ee9b ]]; then
  echo 'Baseline checkout changed; expected GitHub commit 2e391f8.' >&2
  exit 1
fi
log_dir=$(mktemp -d /home/susan/rexmi_rl/outputs/nav_regression/manual-baseline-XXXXXX)
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libgcc_s.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
exec ./run.sh scripts/navigate.py \
  --task RexmiRl-Go2w-Crater-Bowl-RockySlope-Play-v0 \
  --checkpoint "$checkpoint" \
  --nav_controller checkpoint --mission velocity_goal \
  --spawn_x 13 --spawn_y 0 --spawn_z 4.5 --spawn_yaw 3.141592653589793 \
  --goal_x 0 --goal_y 0 --goal_radius 0.8 \
  --manual_checkpoints --perception_view \
  --lidar_display_points 2000 --overlay_hz 30 --max_steps 45000 \
  --log_file "$log_dir/run.csv" "$@"
