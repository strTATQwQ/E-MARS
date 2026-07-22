#!/usr/bin/env bash
set -eo pipefail

RUN_DIR="${1:-/home/song/dgx-unitree/isaac_vln_benchmark/runs/live_closed_loop_smoke_$(date +%Y%m%d_%H%M%S)}"
DURATION_SEC="${DURATION_SEC:-30}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"

mkdir -p "$RUN_DIR"

source /opt/ros/humble/setup.bash
source /home/song/dgx-unitree/ros2_ws/install/setup.bash
source /home/song/dgx-unitree/isaac_vln_benchmark/ros2_ws/install/setup.bash

export ROS_DOMAIN_ID
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export GO2_ENABLE_MOTION=1

pids=()

start_bg() {
  local name="$1"
  shift
  "$@" > "$RUN_DIR/${name}.log" 2>&1 &
  local pid="$!"
  pids+=("$pid")
  echo "$pid" > "$RUN_DIR/${name}.pid"
}

cleanup() {
  set +e
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  pkill -f "/lib/isaac_vln_benchmark/go2_benchmark_adapter_node" 2>/dev/null || true
  pkill -f "/lib/isaac_vln_benchmark/episode_manager_node" 2>/dev/null || true
  pkill -f "/lib/isaac_vln_benchmark/semantic_oracle_node" 2>/dev/null || true
  pkill -f "/lib/isaac_vln_benchmark/obstacle_controller_node" 2>/dev/null || true
  pkill -f "/lib/isaac_vln_benchmark/benchmark_logger_node" 2>/dev/null || true
  pkill -f "/lib/omninav_step_scheduler/mission_manager_node" 2>/dev/null || true
  pkill -f "/lib/omninav_step_scheduler/step_supervisor_node" 2>/dev/null || true
  pkill -f "/lib/omninav_step_scheduler/omninav_scheduler_node" 2>/dev/null || true
  pkill -f "/lib/omninav_step_scheduler/step_pending_policy_node" 2>/dev/null || true
  pkill -f "/lib/omninav_step_scheduler/primitive_executor_node" 2>/dev/null || true
  pkill -f "/lib/omninav_step_scheduler/safe_cmd_mux_node" 2>/dev/null || true
  pkill -f "/lib/omninav_step_scheduler/metrics_logger_node" 2>/dev/null || true
  pkill -f "/lib/omninav_step_scheduler/mock_step_client_node" 2>/dev/null || true
  pkill -f "/lib/omninav_step_scheduler/mock_omninav_client_node" 2>/dev/null || true
  timeout 4s ros2 topic pub --once /safe_cmd_vel geometry_msgs/msg/Twist \
    "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" \
    > "$RUN_DIR/final_zero_safe_cmd.log" 2>&1 || true
  timeout 4s ros2 topic pub --once /go2/cmd_vel geometry_msgs/msg/Twist \
    "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" \
    > "$RUN_DIR/final_zero_go2_cmd.log" 2>&1 || true
}
trap cleanup EXIT

start_bg scheduler_launch ros2 launch omninav_step_scheduler scheduler_isaac.launch.py
start_bg mock_step ros2 run omninav_step_scheduler mock_step_client_node
start_bg mock_omninav ros2 run omninav_step_scheduler mock_omninav_client_node

sleep 4

start_bg benchmark_launch ros2 launch isaac_vln_benchmark isaac_benchmark.launch.py \
  mode:=step_omninav_event \
  output_dir:="$RUN_DIR/logger"

sleep 10

ros2 topic list | sort > "$RUN_DIR/topic_list.txt" || true
ros2 node list | sort > "$RUN_DIR/node_list.txt" || true
ros2 topic info /safe_cmd_vel -v > "$RUN_DIR/safe_cmd_vel_info.txt" 2>&1 || true
ros2 topic info /go2/cmd_vel -v > "$RUN_DIR/go2_cmd_vel_info.txt" 2>&1 || true

timeout 8s ros2 topic echo --once /safe_cmd_vel > "$RUN_DIR/safe_cmd_vel_once.txt" 2>&1 || true
timeout 8s ros2 topic echo --once /go2/cmd_vel > "$RUN_DIR/go2_cmd_vel_once.txt" 2>&1 || true
timeout 8s ros2 topic echo --once /odom > "$RUN_DIR/odom_once.txt" 2>&1 || true
timeout 8s ros2 topic echo --once --field data /isaac/ground_truth_pose > "$RUN_DIR/ground_truth_pose_once_data.txt" 2>&1 || true
timeout 8s ros2 topic echo --once /isaac/episode_status > "$RUN_DIR/episode_status_once.txt" 2>&1 || true
timeout 8s ros2 topic echo --once --field data /isaac/episode_status > "$RUN_DIR/episode_status_once_data.txt" 2>&1 || true
timeout 8s ros2 topic echo --once /semantic_summary_json > "$RUN_DIR/semantic_summary_once.txt" 2>&1 || true
timeout 8s ros2 topic echo --once /scheduler/state > "$RUN_DIR/scheduler_state_once.txt" 2>&1 || true
timeout 8s ros2 topic hz /safe_cmd_vel > "$RUN_DIR/safe_cmd_vel_hz.txt" 2>&1 || true

if [ -f /home/song/isaac_projects/logs/go2_twist_bridge.log ]; then
  tail -80 /home/song/isaac_projects/logs/go2_twist_bridge.log > "$RUN_DIR/go2_twist_bridge_tail.txt"
fi

sleep "$DURATION_SEC"

timeout 8s ros2 topic echo --once /isaac/ground_truth_pose > "$RUN_DIR/ground_truth_pose_final.txt" 2>&1 || true
timeout 8s ros2 topic echo --once --field data /isaac/ground_truth_pose > "$RUN_DIR/ground_truth_pose_final_data.txt" 2>&1 || true
timeout 8s ros2 topic echo --once /isaac/episode_status > "$RUN_DIR/episode_status_final.txt" 2>&1 || true
timeout 8s ros2 topic echo --once --field data /isaac/episode_status > "$RUN_DIR/episode_status_final_data.txt" 2>&1 || true
if [ -f /home/song/isaac_projects/logs/go2_twist_bridge.log ]; then
  tail -120 /home/song/isaac_projects/logs/go2_twist_bridge.log > "$RUN_DIR/go2_twist_bridge_tail_final.txt"
fi

python3 - "$RUN_DIR" <<'PY' || true
import json
import pathlib
import sys

run_dir = pathlib.Path(sys.argv[1])
summary = {}
for name in ("ground_truth_pose_once_data", "ground_truth_pose_final_data", "episode_status_once_data", "episode_status_final_data"):
    path = run_dir / f"{name}.txt"
    if not path.exists():
        continue
    raw = path.read_text(encoding="utf-8", errors="replace").strip()
    raw = raw.split("\n---", 1)[0].strip()
    try:
        value = json.loads(raw)
    except Exception as exc:
        summary[name] = {"parse_error": repr(exc), "raw_prefix": raw[:400]}
        continue
    if "ground_truth_pose" in name:
        summary[name] = {
            "source": value.get("source"),
            "pose": value.get("pose") or value.get("robot_pose"),
            "telemetry_seq": value.get("telemetry_seq"),
            "telemetry_age_sec": value.get("telemetry_age_sec"),
            "task_id": value.get("task_id"),
            "scene_id": value.get("scene_id"),
        }
    else:
        summary[name] = {
            "success": value.get("success"),
            "done": value.get("done"),
            "reason": value.get("reason"),
            "distance_to_target": value.get("distance_to_target"),
            "target_visible": value.get("target_visible"),
            "time_sec": value.get("time_sec"),
        }
(run_dir / "live_smoke_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
PY

echo "$RUN_DIR"
