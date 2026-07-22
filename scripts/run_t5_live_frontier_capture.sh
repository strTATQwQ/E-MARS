#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'usage: %s RESULT_DIR\n' "${0##*/}" >&2
  exit 64
}

[[ $# -eq 1 ]] || usage
result_dir="$1"
root="${INTERNNAV_T1_CONTROL_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)}"
ros_ws="${INTERNVLA_ROS_WS:-$root/ros_ws}"
namespace="${INTERNNAV_T5_LANE_NAMESPACE:-}"

# This process is deliberately a Lane-B, capture-only sidecar.  The resource
# lease is acquired by the outer lane coordinator; this wrapper cannot be used
# to enter an arbitrary ROS graph from an unleased shell.
test "${INTERNNAV_T5_RESOURCE_LEASE_ACK:-}" = lane-b
test "${INTERNNAV_T5_LANE:-}" = b
test "$namespace" = /t5/lane_b
test "${ROS_DOMAIN_ID:-}" = 76
test "${INTERNNAV_RUNTIME_POLICY:-}" = completion_sim
test "${INTERNNAV_SIMULATION_TARGET:-}" = isaac
test -f "$root/scripts/t5_live_frontier_snapshot_node.py"
test -f "$root/configs/internnav_t5/live_frontier_capture.json"
test -f "$ros_ws/install/setup.bash"
test ! -e "$result_dir"

mkdir -p "$result_dir"
result_dir="$(cd -- "$result_dir" && pwd -P)"
snapshot="$result_dir/current.json"
status="$result_dir/status.json"
dependency_status="$result_dir/dependency_status.json"

set +u
source /opt/ros/jazzy/setup.bash
source "$ros_ws/install/setup.bash"
set -u
export ROS2CLI_NO_DAEMON=1
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"

python3 "$root/scripts/t5_live_frontier_snapshot_node.py" \
  --config "$root/configs/internnav_t5/live_frontier_capture.json" \
  --namespace "$namespace" --status-output "$dependency_status" \
  --dependency-check-only

# The Python node owns atomic replacement and clears the current snapshot on
# reset, stale input, missing identity and graceful shutdown.  No ROS
# publisher, action client, goal, cmd_vel or STOP authority is introduced.
exec python3 -u "$root/scripts/t5_live_frontier_snapshot_node.py" \
  --config "$root/configs/internnav_t5/live_frontier_capture.json" \
  --namespace "$namespace" --output "$snapshot" --status-output "$status"
