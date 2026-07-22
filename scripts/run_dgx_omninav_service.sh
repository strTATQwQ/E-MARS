#!/usr/bin/env bash
set -euo pipefail

ROOT="${DGX_UNITREE_ROOT:-$HOME/dgx-unitree}"
CONFIG="${1:-$ROOT/configs/isaac/models/cosmos_bf16.yaml}"
PYTHON_BIN="${OMNINAV_SERVICE_PYTHON:-$HOME/ai-stack/venvs/vlnav/bin/python}"
RESULTS_DIR="${OMNINAV_RESULTS_DIR:-$ROOT/results/omninav_cosmos/model_service}"

mkdir -p "$RESULTS_DIR"
export OMNINAV_RESULTS_DIR="$RESULTS_DIR"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" -m omninav_cosmos.serve --config "$CONFIG"

