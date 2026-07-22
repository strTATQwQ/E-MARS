#!/usr/bin/env bash
set -euo pipefail
: "${CONTROL_ROOT:?CONTROL_ROOT is required}"
: "${RESULT_DIR:?RESULT_DIR is required}"
: "${SENSOR_SOCKET:?SENSOR_SOCKET is required}"
: "${INTERNNAV_SESSION_PROFILE:?INTERNNAV_SESSION_PROFILE is required}"
export PYTHONPATH="$CONTROL_ROOT"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
exec python3 -m sensor_runtime.ros_inner_cleanup \
  --result-dir "$RESULT_DIR" --socket "$SENSOR_SOCKET"
