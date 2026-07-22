#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT/ros2_ws"
if [ -f install/setup.bash ]; then
  # shellcheck disable=SC1091
  source install/setup.bash
fi
ros2 launch isaac_vln_benchmark isaac_benchmark.launch.py "$@"
