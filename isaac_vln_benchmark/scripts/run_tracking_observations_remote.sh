#!/usr/bin/env bash
set -eo pipefail

RUN_DIR="${1:?run directory required}"
SEQUENCES="${2:?tracking sequence manifest required}"
shift 2
SCHEDULER_CONFIG="${SCHEDULER_CONFIG:-/home/song/dgx-unitree/ros2_ws/src/omninav_step_scheduler/config/scheduler_isaac_development.yaml}"
mkdir -p "${RUN_DIR}"
. /opt/ros/humble/setup.bash
. /home/song/dgx-unitree/ros2_ws/install/setup.bash
. /home/song/dgx-unitree/isaac_vln_benchmark/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export GO2_ENABLE_MOTION=1

pids=()
cleanup() {
  set +e
  for pid in "${pids[@]}"; do kill "${pid}" 2>/dev/null || true; done
  pkill -TERM -f "[/]lib/isaac_vln_benchmark/go2_benchmark_adapter_node" 2>/dev/null || true
  pkill -TERM -f "[/]lib/omninav_step_scheduler/" 2>/dev/null || true
}
trap cleanup EXIT
cleanup
set -e

ros2 launch omninav_step_scheduler scheduler_isaac.launch.py config_file:="${SCHEDULER_CONFIG}" enable_local_step_verifiers:=false > "${RUN_DIR}/scheduler.log" 2>&1 &
pids+=("$!")
ros2 run omninav_step_scheduler step_http_client_node --ros-args -p config_file:="${SCHEDULER_CONFIG}" > "${RUN_DIR}/step_http.log" 2>&1 &
pids+=("$!")
ros2 run isaac_vln_benchmark go2_benchmark_adapter_node --ros-args -p publish_synthetic_camera_fallback:=false > "${RUN_DIR}/adapter.log" 2>&1 &
pids+=("$!")
sleep 6

python3 /home/song/dgx-unitree/isaac_vln_benchmark/scripts/collect_tracking_observations_live.py \
  --sequences "${SEQUENCES}" --output "${RUN_DIR}/collection" "$@" | tee "${RUN_DIR}/collection.log"
python3 /home/song/dgx-unitree/isaac_vln_benchmark/scripts/run_tracking_sequence_replay.py \
  --sequences "${RUN_DIR}/collection/collected_sequences.json" \
  --observations "${RUN_DIR}/collection/tracking_observations.json" \
  --output "${RUN_DIR}/result" | tee "${RUN_DIR}/replay.log"
