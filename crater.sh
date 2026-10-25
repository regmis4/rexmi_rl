#!/usr/bin/env bash
# One continuous outside → center → newly planned exit simulation.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"
if [[ "${1:-}" == --help ]]; then
    echo 'Usage: ./crater.sh [--headless] [additional navigate.py options]'
    echo 'Runs autonomous descent, retains live mapping, plans and executes ascent.'
    echo 'GUI dashboard and terrain overlay are enabled by default.'
    exit 0
fi
checkpoint="${REXMI_CHECKPOINT:-$repo_root/logs/rsl_rl/go2w_velocity_rocky_slope/2026-06-30_09-31-48/model_13994.pt}"
if [[ ! -f "$checkpoint" ]]; then
    echo "Missing rocky-slope checkpoint: $checkpoint" >&2
    echo 'Set REXMI_CHECKPOINT to the trained policy file.' >&2
    exit 2
fi
mkdir -p outputs/crater_loop
loop_run_dir="$(mktemp -d "$repo_root/outputs/crater_loop/mission-XXXXXXXX")"
printf '%s\n' "$loop_run_dir" > outputs/crater_loop/latest.txt
printf 'Crater loop logs: %s\n' "$loop_run_dir"
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6:/usr/lib/x86_64-linux-gnu/libgcc_s.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
./run.sh scripts/navigate.py \
    --task RexmiRl-Go2w-Crater-Bowl-RockySlope-Play-v0 \
    --checkpoint "$checkpoint" \
    --mission crater_loop --seed 42 \
    --spawn_x 13 --spawn_y 0 --spawn_z 4.5 --spawn_yaw 3.141592653589793 \
    --goal_x 0 --goal_y 0 --goal_radius .8 \
    --map_retention full --cruise_speed .40 --max_steps 45000 \
    --log_file "$loop_run_dir/run.csv" "$@" 2>&1 | tee "$loop_run_dir/console.log"
python3 - "$loop_run_dir/run.summary.json" <<'PY'
import json,sys
from pathlib import Path
path=Path(sys.argv[1])
if not path.exists():
    raise SystemExit('No mission summary was produced; inspect console.log')
report=json.loads(path.read_text())
print('Crater loop:',report['result'],'|',path)
raise SystemExit(0 if report.get('counts_as_mission') else 1)
PY
