#!/usr/bin/env bash
set -eo pipefail

RUN_DIR="${1:?run directory required}"
shift
SCHEDULER_CONFIG="${SCHEDULER_CONFIG:-/home/song/dgx-unitree/ros2_ws/src/omninav_step_scheduler/config/scheduler_isaac_development.yaml}"
mkdir -p "${RUN_DIR}"
. /opt/ros/humble/setup.bash
. /home/song/dgx-unitree/ros2_ws/install/setup.bash
. /home/song/dgx-unitree/isaac_vln_benchmark/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

pids=()
cleanup() {
  set +e
  for pid in "${pids[@]}"; do kill "${pid}" 2>/dev/null || true; done
  pkill -TERM -f "[/]lib/omninav_step_scheduler/" 2>/dev/null || true
}
trap cleanup EXIT
cleanup
set -e

ros2 launch omninav_step_scheduler scheduler_isaac.launch.py config_file:="${SCHEDULER_CONFIG}" enable_local_step_verifiers:=false > "${RUN_DIR}/scheduler.log" 2>&1 &
pids+=("$!")
ros2 bag record -o "${RUN_DIR}/decision_replay_bag" \
  /benchmark/mode_json /step/route_choice_json /step/request_json \
  /primitive/command_json /metrics/event_jsonl /cmd_vel_candidate /scheduler/state \
  > "${RUN_DIR}/rosbag.log" 2>&1 &
bag_pid="$!"
pids+=("${bag_pid}")
sleep 5
set +e
python3 /home/song/dgx-unitree/isaac_vln_benchmark/scripts/run_decision_robustness_live.py \
  --output "${RUN_DIR}/result" --scheduler-config "${SCHEDULER_CONFIG}" "$@" | tee "${RUN_DIR}/probe.log"
probe_rc="${PIPESTATUS[0]}"
set -e
kill -INT "${bag_pid}" 2>/dev/null || true
wait "${bag_pid}" 2>/dev/null || true
set +e
python3 /home/song/dgx-unitree/isaac_vln_benchmark/scripts/score_decision_robustness_bag.py \
  --bag "${RUN_DIR}/decision_replay_bag" \
  --expected "${RUN_DIR}/result/robustness_results.json" \
  --output "${RUN_DIR}/replay" | tee "${RUN_DIR}/replay.log"
replay_rc="${PIPESTATUS[0]}"
set -e
if [[ "${probe_rc}" -ne 0 || "${replay_rc}" -ne 0 ]]; then
  exit 2
fi
