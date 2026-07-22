#!/usr/bin/env bash
set -eo pipefail

# Run on the Isaac host with its ROS-native Python (not Isaac Sim Python).
CONTROL_ROOT="${INTERNNAV_T1_CONTROL_ROOT:-$HOME/internnav-t1-t2}"
ROS_WS="${INTERNVLA_ROS_WS:-$CONTROL_ROOT/ros_ws}"
REPLAY_ROOT="${INTERNVLA_REPLAY_ROOT:-$CONTROL_ROOT/replay/legacy_canary_120}"
RESULT_DIR="${INTERNVLA_REPLAY_RESULT_DIR:-$CONTROL_ROOT/results/t1_1_replay}"
ROS_DISTRO_NAME="${INTERNVLA_ROS_DISTRO:-humble}"
ROS_PYTHON="${INTERNVLA_ROS_PYTHON:-/usr/bin/python3}"

test -f "/opt/ros/$ROS_DISTRO_NAME/setup.bash"
test -f "$ROS_WS/install/setup.bash"
test -f "$REPLAY_ROOT/manifest.json"
mkdir -p "$RESULT_DIR"

# shellcheck disable=SC1090
source "/opt/ros/$ROS_DISTRO_NAME/setup.bash"
# shellcheck disable=SC1090
source "$ROS_WS/install/setup.bash"
set -u

exec "$ROS_PYTHON" -m internvla_ros2.replay_runner \
  --replay-root "$REPLAY_ROOT" \
  --output "$RESULT_DIR/replay_result.json" \
  --expected-steps 120 \
  --ros-args \
  -p service_timeout_sec:="${INTERNVLA_SERVICE_TIMEOUT_SEC:-300.0}" \
  -p step_deadline_sec:="${INTERNVLA_STEP_DEADLINE_SEC:-30.0}" \
  -p step_validity_sec:="${INTERNVLA_STEP_VALIDITY_SEC:-35.0}" \
  "$@" \
  2>&1 | tee "$RESULT_DIR/replay.log"
