#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OMNINAV_BACKTEST_PHASE=smoke
exec "$SCRIPT_DIR/run_isaac_backtest.sh" "$@"
