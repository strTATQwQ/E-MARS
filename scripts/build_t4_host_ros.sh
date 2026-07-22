#!/usr/bin/env bash
set -euo pipefail

CONTROL_ROOT="${INTERNNAV_T1_CONTROL_ROOT:-$HOME/internnav-t1-t2}"
ROS_WS="${INTERNVLA_ROS_WS:-$CONTROL_ROOT/ros_ws}"
ROS_DISTRO_NAME="${INTERNVLA_ROS_DISTRO:-jazzy}"
case "${INTERNNAV_T4_RESOURCE_LEASE_ACK:-}" in
  dgx|dgx+isaac) ;;
  *) echo "DGX resource lease acknowledgement is required" >&2; exit 2 ;;
esac
test "${INTERNNAV_RUNTIME_POLICY:-}" = completion_sim
test "${INTERNNAV_SIMULATION_TARGET:-}" = isaac
test -d "$ROS_WS/src"
# DGX is the onboard-compute boundary: model inference, standard ROS sensor
# publication, static/global and LiDAR/local navigation, recovery, and the
# final bounded simulation-speed relay all execute here.  Isaac-only rendering
# and PhysX dependencies deliberately remain outside this workspace.
PACKAGES=(
  internvla_ros2_msgs
  internvla_ros2
  internvla_nav2_adapter
  internvla_go2_controller
  internvla_t4_sensors
  internvla_t4_recovery
  go2_sensor_bridge
)
for package in "${PACKAGES[@]}"; do
  test -f "$CONTROL_ROOT/$package/package.xml"
  ln -sfn "$CONTROL_ROOT/$package" "$ROS_WS/src/$package"
done
set +u
source "/opt/ros/$ROS_DISTRO_NAME/setup.bash"
test ! -f "$ROS_WS/install/setup.bash" || source "$ROS_WS/install/setup.bash"
set -u
ros2 pkg prefix nav2_bringup >/dev/null
ros2 pkg prefix nav2_collision_monitor >/dev/null
cd "$ROS_WS"
colcon build --symlink-install --event-handlers console_direct+ \
  --packages-up-to \
    internvla_t4_sensors \
    internvla_t4_recovery \
    go2_sensor_bridge
set +u
source "$ROS_WS/install/setup.bash"
set -u
ros2 interface show internvla_ros2_msgs/srv/RecoveryControl >/dev/null
for package in "${PACKAGES[@]}"; do
  ros2 pkg prefix "$package" >/dev/null
done
PYTHONPATH="$CONTROL_ROOT:${PYTHONPATH:-}" python3 - <<'PY'
from internvla_ros2_msgs.srv import RecoveryControl
from internvla_go2_controller.runtime import ControllerIPCClient
from internvla_t4_sensors.depth_codec import decode_depth_request
from internvla_t4_recovery import adapter_node
from go2_sensor_bridge import bridge_node

assert RecoveryControl is not None
assert ControllerIPCClient is not None
assert decode_depth_request is not None
assert adapter_node is not None
assert bridge_node is not None
PY
