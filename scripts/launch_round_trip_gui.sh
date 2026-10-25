#!/usr/bin/env bash
# Autonomous two-goal characterization; no recorded button sequence.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
mkdir -p outputs/round_trip_user_check
round_trip_run_dir="$(mktemp -d "$repo_root/outputs/round_trip_user_check/round-trip-XXXXXXXX")"
echo "Round-trip logs: $round_trip_run_dir"
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6:/usr/lib/x86_64-linux-gnu/libgcc_s.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
exec ./run.sh scripts/navigate.py \
  --task RexmiRl-Go2w-Crater-Bowl-RockySlope-Play-v0 \
  --checkpoint logs/rsl_rl/go2w_velocity_rocky_slope/2026-06-30_09-31-48/model_13994.pt \
  --nav_controller ascent --mission round_trip --seed 42 \
  --ascent_architecture behavior_tree --ascent_mode characterize \
  --terrain_prior outputs/ascent_full_survey/trial01/run.prior.npz \
  --ascent_uphill_speed_response --ascent_release_memory --ascent_gentle_turn_entry \
  --ascent_active_zero_envelope \
  --ascent_event_snapshots --perception_view --cruise_speed .40 --max_steps 45000 \
  --joint_log_dir "$round_trip_run_dir/joints" --log_file "$round_trip_run_dir/run.csv"
