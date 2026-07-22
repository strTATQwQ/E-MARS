#!/usr/bin/env bash
set -eo pipefail

# Run on the Isaac host under ROS-native Python 3.10. Isaac Sim connects only
# to the local 0600 Unix socket and never imports Humble's rclpy extension.
CONTROL_ROOT="${INTERNNAV_T1_CONTROL_ROOT:-$HOME/internnav-t1-t2}"
ROS_WS="${INTERNVLA_ROS_WS:-$CONTROL_ROOT/ros_ws}"
RESULT_DIR="${INTERNVLA_CLIENT_RESULT_DIR:-$CONTROL_ROOT/results/t1_1_client}"
ROS_DISTRO_NAME="${INTERNVLA_ROS_DISTRO:-humble}"
SOCKET_PATH="${INTERNVLA_CLIENT_SOCKET:-/tmp/internvla_client.sock}"

test -f "/opt/ros/$ROS_DISTRO_NAME/setup.bash"
test -f "$ROS_WS/install/setup.bash"
mkdir -p "$RESULT_DIR"

# shellcheck disable=SC1090
source "/opt/ros/$ROS_DISTRO_NAME/setup.bash"
# shellcheck disable=SC1090
source "$ROS_WS/install/setup.bash"
set -u

export INTERNVLA_CLIENT_SOCKET="$SOCKET_PATH"
export PYTHONUNBUFFERED=1
exec /usr/bin/python3 -m internvla_ros2.client_node \
  --ros-args \
  -p service_timeout_sec:="${INTERNVLA_SERVICE_TIMEOUT_SEC:-300.0}" \
  -p step_deadline_sec:="${INTERNVLA_STEP_DEADLINE_SEC:-30.0}" \
  -p step_validity_sec:="${INTERNVLA_STEP_VALIDITY_SEC:-35.0}" \
  -p ipc_idle_timeout_sec:="${INTERNVLA_IPC_IDLE_TIMEOUT_SEC:-300.0}" \
  "$@" \
  2>&1 | tee "$RESULT_DIR/client_node.log"
