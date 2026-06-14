#!/usr/bin/env bash
# =============================================================================
# run.sh — Convenience launcher for REXMI RL scripts
#
# Usage:
#   ./run.sh scripts/train.py --task RexmiRl-Go2w-Velocity-Flat-v0 --headless
#   ./run.sh scripts/play.py  --task RexmiRl-Go2w-Velocity-Flat-Play-v0
#
# What this script does
# ---------------------
# 1. Loads ISAACLAB_DIR from .env (if it exists) — so you don't need to
#    export the variable manually every session.
# 2. Activates Isaac Lab's Python virtual environment.
# 3. Sets PYTHONPATH so Isaac Lab can find its own modules.
# 4. Delegates to the given script with all remaining arguments forwarded.
#
# Setup (one-time)
# ----------------
#   echo "ISAACLAB_DIR=/home/susan/IsaacLab" > .env
#   chmod +x run.sh
# =============================================================================

set -euo pipefail  # exit on error, undefined vars, pipe failures

# ---------------------------------------------------------------------------
# 1. Load local configuration from .env
#    The .env file is gitignored and should contain:
#      ISAACLAB_DIR=/path/to/IsaacLab
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

if [[ -f "${ENV_FILE}" ]]; then
    # Source the .env file, exporting all variables
    set -a
    source "${ENV_FILE}"
    set +a
    echo "[run.sh] Loaded config from ${ENV_FILE}"
fi

# ---------------------------------------------------------------------------
# 2. Resolve Isaac Lab directory
# ---------------------------------------------------------------------------
ISAACLAB_DIR="${ISAACLAB_DIR:-${HOME}/IsaacLab}"

if [[ ! -d "${ISAACLAB_DIR}" ]]; then
    echo "ERROR: Isaac Lab directory not found: ${ISAACLAB_DIR}"
    echo "       Create a .env file with: ISAACLAB_DIR=/path/to/IsaacLab"
    exit 1
fi

echo "[run.sh] Isaac Lab: ${ISAACLAB_DIR}"

# ---------------------------------------------------------------------------
# 3. Resolve Python interpreter
#
#    Priority order:
#      1. CONDA_PYTHON env var (set it in .env to override)
#      2. Active conda environment (if `conda activate env_isaacsim` was run)
#      3. Known conda env path (hardcoded fallback for env_isaacsim)
#      4. Isaac Lab .venv (legacy fallback)
#
#    For this machine: conda activate env_isaacsim before running,
#    or set CONDA_PYTHON in .env:
#      CONDA_PYTHON=/home/susan/miniconda3/envs/env_isaacsim/bin/python
# ---------------------------------------------------------------------------

# If user explicitly set CONDA_PYTHON, use it.
if [[ -n "${CONDA_PYTHON:-}" ]] && [[ -f "${CONDA_PYTHON}" ]]; then
    VENV_PYTHON="${CONDA_PYTHON}"

# If a conda env is currently active (CONDA_PREFIX is set by `conda activate`),
# use its Python.
elif [[ -n "${CONDA_PREFIX:-}" ]] && [[ -f "${CONDA_PREFIX}/bin/python" ]]; then
    VENV_PYTHON="${CONDA_PREFIX}/bin/python"

# Known fallback: env_isaacsim on this machine
elif [[ -f "/home/susan/miniconda3/envs/env_isaacsim/bin/python" ]]; then
    VENV_PYTHON="/home/susan/miniconda3/envs/env_isaacsim/bin/python"

# Legacy fallback: Isaac Lab .venv
elif [[ -f "${ISAACLAB_DIR}/.venv/bin/python" ]]; then
    VENV_PYTHON="${ISAACLAB_DIR}/.venv/bin/python"

else
    echo "ERROR: No Python interpreter found."
    echo "       Run: conda activate env_isaacsim"
    echo "       Or set CONDA_PYTHON=/path/to/python in .env"
    exit 1
fi

echo "[run.sh] Python: ${VENV_PYTHON}"

# ---------------------------------------------------------------------------
# 4. Set PYTHONPATH so both Isaac Lab modules and rexmi_rl are importable
# ---------------------------------------------------------------------------
export PYTHONPATH="${ISAACLAB_DIR}/source/isaaclab/isaaclab:${PYTHONPATH:-}"
export PYTHONPATH="${ISAACLAB_DIR}/source/isaaclab_assets/isaaclab_assets:${PYTHONPATH}"
export PYTHONPATH="${ISAACLAB_DIR}/source/isaaclab_tasks/isaaclab_tasks:${PYTHONPATH}"
export PYTHONPATH="${ISAACLAB_DIR}/source/isaaclab_rl/isaaclab_rl:${PYTHONPATH}"
export PYTHONPATH="${SCRIPT_DIR}/source:${PYTHONPATH}"

# ---------------------------------------------------------------------------
# 5. Set logs directory (optional but keeps logs in the repo root)
# ---------------------------------------------------------------------------
export LOGS_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOGS_DIR}"

# ---------------------------------------------------------------------------
# 6. Forward to the requested script
# ---------------------------------------------------------------------------
if [[ $# -eq 0 ]]; then
    echo "Usage: ./run.sh <script> [args...]"
    echo "Example: ./run.sh scripts/train.py --task RexmiRl-Go2w-Velocity-Flat-v0 --headless"
    exit 1
fi

SCRIPT="$1"
shift  # remove script name, leave remaining args

echo "[run.sh] Running: ${VENV_PYTHON} ${SCRIPT} $*"
echo "─────────────────────────────────────────────────────────"

exec "${VENV_PYTHON}" "${SCRIPT_DIR}/${SCRIPT}" "$@"
