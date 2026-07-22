#!/usr/bin/env bash
set -euo pipefail

: "${INTERNNAV_SENSOR_SESSION_LEASE_ACK:?Isaac lease acknowledgement is required}"
test "$INTERNNAV_SENSOR_SESSION_LEASE_ACK" = 1
: "${CONTROL_ROOT:?CONTROL_ROOT is required}"
: "${RESULT_DIR:?RESULT_DIR is required}"
: "${SENSOR_SOCKET:?SENSOR_SOCKET is required}"

CONTAINER="${INTERNVLA_T4_CONTAINER_NAME:-internnav_t4_isaac_ros}"
test "$(docker inspect -f '{{.State.Running}}' "$CONTAINER")" = true
test -d "$RESULT_DIR"
test ! -e "$RESULT_DIR/inner_lifecycle"

# The control root and result directory are bind-mounted at identical paths.
# The inner Python supervisor, not this docker client, owns and proves cleanup
# of the actual bridge/sidecar PID and PGID values.
exec docker exec \
  --user admin \
  --workdir "$CONTROL_ROOT" \
  -e "CONTROL_ROOT=$CONTROL_ROOT" \
  -e "RESULT_DIR=$RESULT_DIR" \
  -e "SENSOR_SOCKET=$SENSOR_SOCKET" \
  -e "INTERNNAV_RUNTIME_POLICY=${INTERNNAV_RUNTIME_POLICY:-strict_evidence}" \
  -e "INTERNNAV_SESSION_PROFILE=${INTERNNAV_SESSION_PROFILE:-}" \
  -e "INTERNNAV_T4_MAP_COMPANION_MODULE=${INTERNNAV_T4_MAP_COMPANION_MODULE:-}" \
  -e "INTERNNAV_T4_MAP_CONFIG_DIR=${INTERNNAV_T4_MAP_CONFIG_DIR:-}" \
  -e "INTERNNAV_T4_MAP_NVBLOX_MODE=${INTERNNAV_T4_MAP_NVBLOX_MODE:-}" \
  -e "INTERNNAV_T4_MAP_NVBLOX_HEALTH=${INTERNNAV_T4_MAP_NVBLOX_HEALTH:-}" \
  -e "INTERNNAV_SIMULATION_TARGET=${INTERNNAV_SIMULATION_TARGET:-}" \
  -e "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}" \
  -e ROS_LOCALHOST_ONLY=0 \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONNOUSERSITE=1 \
  -e PYTHONDONTWRITEBYTECODE=1 \
  "$CONTAINER" \
  bash -lc 'unset PYTHONPATH
source /opt/ros/jazzy/setup.bash
source /workspaces/isaac/install/setup.bash
SETUP_PYTHONPATH="${PYTHONPATH:-}"
CONTROLLED_PYTHONPATH="$(
  /usr/bin/env -u PYTHONPATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    /usr/bin/python3 -I "$CONTROL_ROOT/sensor_runtime/pythonpath_policy.py" \
      --control-root "$CONTROL_ROOT" \
      --setup-pythonpath "$SETUP_PYTHONPATH"
)"
SETUP_PYTHONPATH_SHA256="$(printf '%s' "$SETUP_PYTHONPATH" | /usr/bin/sha256sum)"
SETUP_PYTHONPATH_SHA256="${SETUP_PYTHONPATH_SHA256%% *}"
export PYTHONPATH="$CONTROLLED_PYTHONPATH"
export INTERNNAV_FROZEN_SETUP_PYTHONPATH_SHA256="$SETUP_PYTHONPATH_SHA256"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
exec /usr/bin/setsid --wait /usr/bin/python3 -m sensor_runtime.ros_inner_supervisor \
  --control-root "$CONTROL_ROOT" \
  --result-dir "$RESULT_DIR" \
  --socket "$SENSOR_SOCKET"'
