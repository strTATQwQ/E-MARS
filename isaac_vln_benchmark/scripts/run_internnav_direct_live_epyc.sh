#!/usr/bin/env bash
set -eo pipefail

RUN_DIR="${1:-/home/song/dgx-unitree/isaac_vln_benchmark/runs/internnav_direct_live_$(date +%Y%m%d_%H%M%S)}"
DURATION_SEC="${DURATION_SEC:-20}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"

mkdir -p "$RUN_DIR"

source /opt/ros/humble/setup.bash
source /home/song/dgx-unitree/ros2_ws/install/setup.bash
source /home/song/dgx-unitree/isaac_vln_benchmark/ros2_ws/install/setup.bash

export ROS_DOMAIN_ID
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export GO2_ENABLE_MOTION=1

pkill -f "/lib/isaac_vln_benchmark/go2_benchmark_adapter_node" 2>/dev/null || true
pkill -f "/lib/omninav_step_scheduler/internnav_go2_client_node" 2>/dev/null || true
sleep 1

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
  pkill -f "/lib/omninav_step_scheduler/internnav_go2_client_node" 2>/dev/null || true
  timeout 4s ros2 topic pub --once /go2/cmd_vel geometry_msgs/msg/Twist \
    "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" \
    > "$RUN_DIR/final_zero_go2_cmd.log" 2>&1 || true
}
trap cleanup EXIT

start_bg adapter ros2 run isaac_vln_benchmark go2_benchmark_adapter_node \
  --ros-args -p forward_to_isaac:=false

sleep 4

ros2 topic pub --once /isaac/reset_episode std_msgs/msg/String \
  "{data: '{\"task_id\":\"simple_001\",\"scene_id\":\"straight_001\"}'}" \
  > "$RUN_DIR/reset_episode.log" 2>&1 || true

sleep 2

start_bg internnav_direct ros2 launch omninav_step_scheduler internnav_go2_direct.launch.py \
  config_file:="${INTERNNAV_CONFIG_FILE:-/home/song/dgx-unitree/ros2_ws/install/omninav_step_scheduler/share/omninav_step_scheduler/config/internnav_go2_direct.yaml}"

sleep 4

timeout "$((DURATION_SEC + 25))s" ros2 topic echo --field data /metrics/event_jsonl std_msgs/msg/String \
  > "$RUN_DIR/metrics_event_stream.txt" 2>&1 &
pids+=("$!")

sleep 8

ros2 topic list | sort > "$RUN_DIR/topic_list.txt" || true
ros2 node list | sort > "$RUN_DIR/node_list.txt" || true
ros2 topic info /go2/cmd_vel -v > "$RUN_DIR/go2_cmd_vel_info.txt" 2>&1 || true
timeout 8s ros2 topic echo --once /go2/cmd_vel > "$RUN_DIR/go2_cmd_vel_once.txt" 2>&1 || true
timeout 8s ros2 topic hz /go2/cmd_vel > "$RUN_DIR/go2_cmd_vel_hz.txt" 2>&1 || true
timeout 8s ros2 topic echo --once /odom > "$RUN_DIR/odom_once.txt" 2>&1 || true
timeout 8s ros2 topic hz /camera/front/image > "$RUN_DIR/front_image_hz.txt" 2>&1 || true
timeout 8s ros2 topic hz /camera/front/depth > "$RUN_DIR/front_depth_hz.txt" 2>&1 || true
timeout 8s ros2 topic echo --once --field height /camera/front/image > "$RUN_DIR/front_image_height.txt" 2>&1 || true
timeout 8s ros2 topic echo --once --field width /camera/front/image > "$RUN_DIR/front_image_width.txt" 2>&1 || true
timeout 8s ros2 topic echo --once --field encoding /camera/front/image > "$RUN_DIR/front_image_encoding.txt" 2>&1 || true
timeout 8s ros2 topic echo --once --field height /camera/front/depth > "$RUN_DIR/front_depth_height.txt" 2>&1 || true
timeout 8s ros2 topic echo --once --field width /camera/front/depth > "$RUN_DIR/front_depth_width.txt" 2>&1 || true
timeout 8s ros2 topic echo --once --field encoding /camera/front/depth > "$RUN_DIR/front_depth_encoding.txt" 2>&1 || true
timeout 8s ros2 topic echo --once --field data /robot_state_json > "$RUN_DIR/robot_state_once_data.txt" 2>&1 || true
timeout 8s ros2 topic echo --once --field data /isaac/ground_truth_pose > "$RUN_DIR/ground_truth_pose_once_data.txt" 2>&1 || true

sleep "$DURATION_SEC"

timeout 8s ros2 topic echo --once /go2/cmd_vel > "$RUN_DIR/go2_cmd_vel_final.txt" 2>&1 || true
timeout 8s ros2 topic echo --once --field data /robot_state_json > "$RUN_DIR/robot_state_final_data.txt" 2>&1 || true
timeout 8s ros2 topic echo --once --field data /isaac/ground_truth_pose > "$RUN_DIR/ground_truth_pose_final_data.txt" 2>&1 || true

if [ -f /home/song/isaac_projects/logs/go2_twist_bridge.log ]; then
  tail -160 /home/song/isaac_projects/logs/go2_twist_bridge.log > "$RUN_DIR/go2_twist_bridge_tail_final.txt"
fi

latest_isaac_log="$(
  ls -t /home/song/isaac_projects/logs/go2_warehouse_teleop_desktop*.out \
        /home/song/isaac_projects/logs/go2_warehouse_teleop_desktop*.log 2>/dev/null | head -1 || true
)"
if [ -n "$latest_isaac_log" ] && [ -f "$latest_isaac_log" ]; then
  echo "$latest_isaac_log" > "$RUN_DIR/isaac_go2_log_path.txt"
  grep -E "TELEOP_STEP|PERF_STEP|BENCHMARK_TELEMETRY_READY|BENCHMARK_CONTROL_READY|BENCHMARK_RESET|CAMERA_READY|CAMERA_TELEMETRY_READY|CAMERA_TELEMETRY_EXCEPTION|CAMERA_UNAVAILABLE" "$latest_isaac_log" \
    | tail -180 > "$RUN_DIR/isaac_go2_tail_final.txt" || true
fi

python3 - "$RUN_DIR" <<'PY' || true
import json
import pathlib
import statistics
import sys

run_dir = pathlib.Path(sys.argv[1])
summary = {
    "internnav_init": [],
    "internnav_steps": [],
    "ground_truth": {},
    "robot_state": {},
    "camera_topics": {},
}

for line in (run_dir / "metrics_event_stream.txt").read_text(encoding="utf-8", errors="replace").splitlines():
    line = line.strip()
    if not line.startswith("{"):
        continue
    try:
        event = json.loads(line)
    except Exception:
        continue
    if event.get("node") != "internnav_go2_client":
        continue
    if event.get("event_type") == "internnav_go2_init":
        summary["internnav_init"].append(event)
    elif event.get("event_type") == "internnav_go2_step":
        summary["internnav_steps"].append(event)

for name in ("ground_truth_pose_once_data", "ground_truth_pose_final_data"):
    path = run_dir / f"{name}.txt"
    if not path.exists():
        continue
    raw = path.read_text(encoding="utf-8", errors="replace").strip().split("\n---", 1)[0].strip()
    try:
        value = json.loads(raw)
    except Exception as exc:
        summary["ground_truth"][name] = {"parse_error": repr(exc), "raw_prefix": raw[:400]}
        continue
    summary["ground_truth"][name] = {
        "source": value.get("source"),
        "pose": value.get("pose") or value.get("robot_pose"),
        "telemetry_seq": value.get("telemetry_seq"),
        "telemetry_age_sec": value.get("telemetry_age_sec"),
    }

for name in ("robot_state_once_data", "robot_state_final_data"):
    path = run_dir / f"{name}.txt"
    if not path.exists():
        continue
    raw = path.read_text(encoding="utf-8", errors="replace").strip().split("\n---", 1)[0].strip()
    try:
        value = json.loads(raw)
    except Exception as exc:
        summary["robot_state"][name] = {"parse_error": repr(exc), "raw_prefix": raw[:400]}
        continue
    summary["robot_state"][name] = {
        "pose": value.get("pose"),
        "camera": value.get("camera"),
    }

for topic_name, stem in (
    ("front_image", "front_image"),
    ("front_depth", "front_depth"),
):
    topic = {}
    for field in ("height", "width", "encoding"):
        path = run_dir / f"{stem}_{field}.txt"
        if path.exists():
            topic[field] = path.read_text(encoding="utf-8", errors="replace").strip().split("\n---", 1)[0].strip()
    hz_path = run_dir / f"{stem}_hz.txt"
    if hz_path.exists():
        topic["hz_raw_tail"] = "\n".join(hz_path.read_text(encoding="utf-8", errors="replace").splitlines()[-4:])
    summary["camera_topics"][topic_name] = topic

steps = summary["internnav_steps"]
latencies = [float(item.get("latency_s", 0.0)) for item in steps if item.get("latency_s") is not None]
summary["counts"] = {
    "init_events": len(summary["internnav_init"]),
    "step_events": len(steps),
    "results": {},
    "action_codes": {},
    "model_action_codes": {},
    "obs_sources": {},
    "override_reasons": {},
    "override_events": 0,
    "mean_step_latency_s": round(statistics.mean(latencies), 3) if latencies else None,
    "p50_step_latency_s": round(statistics.median(latencies), 3) if latencies else None,
}
for item in steps:
    for key, bucket in (
        ("result", "results"),
        ("action_code", "action_codes"),
        ("model_action_code", "model_action_codes"),
        ("obs_source", "obs_sources"),
    ):
        value = str(item.get(key))
        summary["counts"][bucket][value] = summary["counts"][bucket].get(value, 0) + 1
    if item.get("override_active"):
        summary["counts"]["override_events"] += 1
        value = str(item.get("override_reason") or item.get("override_detail", {}).get("override_reason"))
        summary["counts"]["override_reasons"][value] = summary["counts"]["override_reasons"].get(value, 0) + 1

(run_dir / "internnav_direct_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
PY

echo "$RUN_DIR"
