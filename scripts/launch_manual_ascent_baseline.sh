#!/usr/bin/env bash
# Manual ascent baseline: archived comparison weights + live LiDAR; no autonomy.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
map_file="$repo_root/outputs/remote_autonomy/reachable-height-objective/stable_capture/request-d55yep2q.npz"
policy_file="$repo_root/logs/rsl_rl/go2w_velocity_rocky_slope/2026-06-30_09-31-48/model_13994.pt"
[[ -f "$map_file" && -f "$policy_file" ]] || { echo 'Missing reference map or rocky-slope policy'; exit 1; }
mkdir -p logs/telemetry
manual_run_dir="$(mktemp -d "$repo_root/logs/telemetry/manual-ascent-XXXXXXXX")"
printf '%s\n' "$manual_run_dir" > logs/telemetry/latest_manual_ascent.txt
printf 'Manual ascent logs: %s\n' "$manual_run_dir"
cp -- scripts/launch_manual_ascent_baseline.sh "$manual_run_dir/launch.sh"
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6:/usr/lib/x86_64-linux-gnu/libgcc_s.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
./run.sh scripts/teleop.py \
  --task RexmiRl-Go2w-Crater-Bowl-RockySlope-Play-v0 \
  --seed 42 --policy rocky_slope --ckpt_rocky "$policy_file" \
  --spawn_preset floor --spawn_yaw 1.5707963267948966 \
  --perception_view --display_map "$map_file" --stop_on_reset \
  --max_steps 45000 --scenario human_ascent_baseline \
  --run_notes 'Human-chosen floor-to-exterior ascent; frozen comparison costmap; no autonomous planner. Initial heading north. Record actual exit for matched machine comparison.' \
  --joint_log_dir "$manual_run_dir/run" --log_file "$manual_run_dir/run.csv" \
  2>&1 | tee "$manual_run_dir/console.log"
