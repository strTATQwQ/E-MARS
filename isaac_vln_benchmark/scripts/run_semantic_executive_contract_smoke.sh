#!/usr/bin/env bash
set -eo pipefail

ROOT="${ROOT:-/home/song/dgx-unitree}"
OUT="${1:-/tmp/semantic_executive_contract_smoke}"
SCHEDULER_CONFIG="${ROOT}/ros2_ws/src/omninav_step_scheduler/config/scheduler_isaac_v14_multimodal.yaml"
TASK_FILE="${ROOT}/isaac_vln_benchmark/configs/natural_language_navigation_v18.yaml"

mkdir -p "${OUT}"
source /opt/ros/humble/setup.bash
source "${ROOT}/ros2_ws/install/setup.bash"
source "${ROOT}/isaac_vln_benchmark/ros2_ws/install/setup.bash"
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-78}"

setsid ros2 run omninav_step_scheduler semantic_executive_node --ros-args \
  -p config_file:="${SCHEDULER_CONFIG}" >"${OUT}/semantic_executive.log" 2>&1 &
EXEC_PID=$!
setsid ros2 run omninav_step_scheduler omninav_scheduler_node --ros-args \
  -p config_file:="${SCHEDULER_CONFIG}" >"${OUT}/omninav_scheduler.log" 2>&1 &
OMNINAV_PID=$!
setsid ros2 run isaac_vln_benchmark forced_semantic_oracle_node --ros-args \
  -p task_file:="${TASK_FILE}" >"${OUT}/forced_oracle.log" 2>&1 &
ORACLE_PID=$!

cleanup() {
  local pids=("${EXEC_PID}" "${OMNINAV_PID}" "${ORACLE_PID}")
  for optional_pid in "${METRICS_PID:-}" "${ACCEPT_PID:-}" "${GOAL_PID:-}" "${REQUEST_PID:-}"; do
    if [[ -n "${optional_pid}" ]]; then
      pids+=("${optional_pid}")
    fi
  done
  for pid in "${EXEC_PID}" "${OMNINAV_PID}" "${ORACLE_PID}"; do
    kill -- "-${pid}" 2>/dev/null || true
  done
  kill "${pids[@]}" 2>/dev/null || true
  for _ in $(seq 1 20); do
    local alive=0
    for pid in "${pids[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        alive=1
      fi
    done
    [[ "${alive}" -eq 0 ]] && break
    sleep 0.1
  done
  for pid in "${EXEC_PID}" "${OMNINAV_PID}" "${ORACLE_PID}"; do
    kill -9 -- "-${pid}" 2>/dev/null || true
  done
  kill -9 "${pids[@]}" 2>/dev/null || true
  wait "${pids[@]}" 2>/dev/null || true
}
trap cleanup EXIT

sleep 2
timeout 20 ros2 topic echo --no-daemon --spin-time 2 --full-length /metrics/event_jsonl std_msgs/msg/String \
  >"${OUT}/metrics.txt" &
METRICS_PID=$!
timeout 15 ros2 topic echo --no-daemon --spin-time 2 --full-length --once /semantic_executive/accepted_subgoal_json \
  std_msgs/msg/String >"${OUT}/accepted_subgoal.txt" &
ACCEPT_PID=$!
timeout 15 ros2 topic echo --no-daemon --spin-time 2 --full-length --once /omninav/semantic_goal_json \
  std_msgs/msg/String >"${OUT}/semantic_goal.txt" &
GOAL_PID=$!
timeout 15 ros2 topic echo --no-daemon --spin-time 2 --full-length --once /omninav/request_json \
  std_msgs/msg/String >"${OUT}/omninav_request.txt" &
REQUEST_PID=$!

timeout 15 ros2 topic pub --once /benchmark/mode_json std_msgs/msg/String \
  "data: '{\"mode\":\"forced_semantic_oracle\",\"episode_id\":\"contract_ep_01\",\"task_id\":\"ref_01\"}'" \
  >"${OUT}/mode_publish.txt"
sleep 1
timeout 15 ros2 topic pub --once /user_instruction std_msgs/msg/String \
  "data: '{\"instruction\":\"Find the red fire extinguisher beside the exit sign and stop near it.\",\"mission_id\":\"contract_mission_01\"}'" \
  >"${OUT}/instruction_publish.txt"
sleep 1
timeout 15 ros2 topic pub --once /scheduler/state std_msgs/msg/String "data: 'RUN_FAST'" \
  >"${OUT}/state_publish.txt"

wait "${ACCEPT_PID}"
wait "${GOAL_PID}"
wait "${REQUEST_PID}"
kill "${METRICS_PID}" 2>/dev/null || true
wait "${METRICS_PID}" 2>/dev/null || true

grep -q 'red fire extinguisher' "${OUT}/accepted_subgoal.txt"
grep -q '"subgoal_type": "find"' "${OUT}/accepted_subgoal.txt"
grep -q 'red fire extinguisher' "${OUT}/semantic_goal.txt"
grep -q 'red fire extinguisher' "${OUT}/omninav_request.txt"
if grep -Eiq 'cmd_vel|waypoint|primitive|target_pose|trajectory' \
  "${OUT}/semantic_goal.txt" "${OUT}/omninav_request.txt"; then
  echo "semantic goal leaked a forbidden motion field" >&2
  exit 2
fi

cat >"${OUT}/result.json" <<'EOF'
{
  "pass": true,
  "chain": ["forced_semantic_oracle", "stale_gate", "semantic_executive", "omninav_semantic_goal", "omninav_request"],
  "motion_published": false,
  "qualification_evidence": false
}
EOF

cat "${OUT}/result.json"
