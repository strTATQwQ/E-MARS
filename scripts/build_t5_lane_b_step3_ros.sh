#!/usr/bin/env bash
set -euo pipefail

control_root="${INTERNNAV_T1_CONTROL_ROOT:-$HOME/internnav-t1-t2}"
ros_ws="${INTERNVLA_ROS_WS:-$control_root/ros_ws}"
ros_distro_name="${INTERNVLA_ROS_DISTRO:-jazzy}"

test "${INTERNNAV_RUNTIME_POLICY:-}" = completion_sim
test "${INTERNNAV_SIMULATION_TARGET:-}" = isaac
test "${INTERNNAV_T5_LANE:-}" = b
test "${INTERNNAV_T5_LANE_NAMESPACE:-/t5/lane_b}" = /t5/lane_b
case "${INTERNNAV_T4_RESOURCE_LEASE_ACK:-${INTERNNAV_T5_RESOURCE_LEASE_ACK:-}}" in
  dgx|dgx-b|lane-b) ;;
  *) echo "Lane-B DGX lease acknowledgement is required" >&2; exit 2 ;;
esac
test -f "$control_root/internnav_t5_lane_b_msgs/package.xml"
test -d "$ros_ws/src"

ln -sfn "$control_root/internnav_t5_lane_b_msgs" \
  "$ros_ws/src/internnav_t5_lane_b_msgs"
set +u
source "/opt/ros/$ros_distro_name/setup.bash"
test ! -f "$ros_ws/install/setup.bash" || source "$ros_ws/install/setup.bash"
set -u
cd "$ros_ws"
colcon build --symlink-install --event-handlers console_direct+ \
  --packages-select internnav_t5_lane_b_msgs
set +u
source "$ros_ws/install/setup.bash"
set -u
ros2 interface show internnav_t5_lane_b_msgs/srv/PrepareFrontiers >/dev/null
ros2 interface show internnav_t5_lane_b_msgs/srv/CommitFrontier >/dev/null
PYTHONPATH="$control_root:${PYTHONPATH:-}" python3 - <<'PY'
from internnav_t5_lane_b_msgs.srv import CommitFrontier, PrepareFrontiers

assert CommitFrontier is not None
assert PrepareFrontiers is not None
PY
