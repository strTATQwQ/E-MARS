#!/usr/bin/env bash
set -eo pipefail

RUN_DIR="${1:-/home/song/dgx-unitree/isaac_vln_benchmark/runs/live_benchmark_$(date +%Y%m%d_%H%M%S)}"
NUM_EPISODES="${NUM_EPISODES:-1}"
MODES="${MODES:-step_omninav_event}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
LIVE_EPISODE_TIMEOUT_SEC="${LIVE_EPISODE_TIMEOUT_SEC:-0}"
LIVE_STARTUP_TIMEOUT_SEC="${LIVE_STARTUP_TIMEOUT_SEC:-20}"
LIVE_SETTLE_SEC="${LIVE_SETTLE_SEC:-1.0}"
MOCK_MODELS="${MOCK_MODELS:-1}"
MOCK_STEP="${MOCK_STEP:-$MOCK_MODELS}"
MOCK_OMNINAV="${MOCK_OMNINAV:-$MOCK_MODELS}"
INTERNNAV_SERVER="${INTERNNAV_SERVER:-http://10.100.100.128:8087}"
TASKS_FILE="${TASKS_FILE:-}"
SCENES_FILE="${SCENES_FILE:-}"
BENCHMARK_CONFIG="${BENCHMARK_CONFIG:-}"
DECISION_STRESS_ENABLE="${DECISION_STRESS_ENABLE:-0}"
DECISION_STRESS_DELAY_SEC="${DECISION_STRESS_DELAY_SEC:-0.0}"
DECISION_STRESS_DROP_RATE="${DECISION_STRESS_DROP_RATE:-0.0}"
DECISION_STRESS_PROFILE="${DECISION_STRESS_PROFILE:-none}"
DECISION_STRESS_RESET_ON_RAW="${DECISION_STRESS_RESET_ON_RAW:-0}"
QUALIFICATION_BAG_ENABLE="${QUALIFICATION_BAG_ENABLE:-0}"
GROUNDED_SAM_ENDPOINT="${GROUNDED_SAM_ENDPOINT:-http://10.100.100.128:8097/detect_segment}"
BENCHMARK_SEED="${BENCHMARK_SEED:-42}"
ISAAC_HEADLESS_EVIDENCE="${ISAAC_HEADLESS_EVIDENCE:-0}"
ALLOW_SYNTHETIC_CAMERA_FALLBACK="${ALLOW_SYNTHETIC_CAMERA_FALLBACK:-1}"
ENABLE_GROUNDED_SAM="${ENABLE_GROUNDED_SAM:-1}"
OMNINAV_FORMAL_PREFLIGHT="${OMNINAV_FORMAL_PREFLIGHT:-0}"

mkdir -p "$RUN_DIR"

source /opt/ros/humble/setup.bash
source /home/song/dgx-unitree/ros2_ws/install/setup.bash
source /home/song/dgx-unitree/isaac_vln_benchmark/ros2_ws/install/setup.bash

export ROS_DOMAIN_ID
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export GO2_ENABLE_MOTION=1
SCHEDULER_CONFIG="${SCHEDULER_CONFIG:-/home/song/dgx-unitree/ros2_ws/install/omninav_step_scheduler/share/omninav_step_scheduler/config/scheduler_isaac_real_models.yaml}"

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
  pkill -f "[/]lib/isaac_vln_benchmark/go2_benchmark_adapter_node" 2>/dev/null || true
  pkill -f "[/]lib/isaac_vln_benchmark/grounded_sam_perception_node" 2>/dev/null || true
  pkill -f "[/]lib/omninav_step_scheduler/mission_manager_node" 2>/dev/null || true
  pkill -f "[/]lib/omninav_step_scheduler/step_supervisor_node" 2>/dev/null || true
  pkill -f "[/]lib/omninav_step_scheduler/step_role_router_node" 2>/dev/null || true
  pkill -f "[/]lib/omninav_step_scheduler/route_choice_verifier_node" 2>/dev/null || true
  pkill -f "[/]lib/omninav_step_scheduler/semantic_stop_verifier_node" 2>/dev/null || true
  pkill -f "[/]lib/omninav_step_scheduler/visible_to_stop_monitor_node" 2>/dev/null || true
  pkill -f "[/]lib/omninav_step_scheduler/route_stop_primitive_bridge_node" 2>/dev/null || true
  pkill -f "[/]lib/omninav_step_scheduler/omninav_scheduler_node" 2>/dev/null || true
  pkill -f "[/]lib/omninav_step_scheduler/internnav_scheduler_node" 2>/dev/null || true
  pkill -f "[/]lib/omninav_step_scheduler/internnav_go2_client_node" 2>/dev/null || true
  pkill -f "[/]lib/omninav_step_scheduler/step_pending_policy_node" 2>/dev/null || true
  pkill -f "[/]lib/omninav_step_scheduler/primitive_executor_node" 2>/dev/null || true
  pkill -f "[/]lib/omninav_step_scheduler/sensor_only_planner_node" 2>/dev/null || true
  pkill -f "[/]lib/omninav_step_scheduler/safe_cmd_mux_node" 2>/dev/null || true
  pkill -f "[/]lib/omninav_step_scheduler/metrics_logger_node" 2>/dev/null || true
  pkill -f "[/]lib/omninav_step_scheduler/mock_step_client_node" 2>/dev/null || true
  pkill -f "[/]lib/omninav_step_scheduler/step_http_client_node" 2>/dev/null || true
  pkill -f "[/]lib/omninav_step_scheduler/semantic_executive_node" 2>/dev/null || true
  pkill -f "[/]lib/omninav_step_scheduler/semantic_recovery_bridge_node" 2>/dev/null || true
  pkill -f "[/]lib/isaac_vln_benchmark/forced_semantic_oracle_node" 2>/dev/null || true
  pkill -f "[/]lib/isaac_vln_benchmark/public_semantic_heuristic_node" 2>/dev/null || true
  pkill -f "[/]lib/isaac_vln_benchmark/semantic_marker_judge_node" 2>/dev/null || true
  pkill -f "[/]lib/isaac_vln_benchmark/delay_injector_node" 2>/dev/null || true
  pkill -f "[/]lib/isaac_vln_benchmark/reset_on_decision_node" 2>/dev/null || true
  pkill -f "[/]lib/omninav_step_scheduler/mock_omninav_client_node" 2>/dev/null || true
  timeout 4s ros2 topic pub --once /safe_cmd_vel geometry_msgs/msg/Twist \
    "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" \
    > "$RUN_DIR/final_zero_safe_cmd.log" 2>&1 || true
  timeout 4s ros2 topic pub --once /go2/cmd_vel geometry_msgs/msg/Twist \
    "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" \
    > "$RUN_DIR/final_zero_go2_cmd.log" 2>&1 || true
}
trap cleanup EXIT

# Clear stale benchmark ROS nodes from previous probes before launching a new run.
cleanup
set -e

enable_local_step_verifiers=true
if [ "$MOCK_STEP" != "1" ] && [[ "$MODES" == *"step"* ]]; then
  enable_local_step_verifiers=false
fi
start_bg scheduler_launch ros2 launch omninav_step_scheduler scheduler_isaac.launch.py \
  config_file:="$SCHEDULER_CONFIG" enable_local_step_verifiers:="$enable_local_step_verifiers"
if [ "$MOCK_STEP" = "1" ]; then
  start_bg mock_step ros2 run omninav_step_scheduler mock_step_client_node
elif [[ "$MODES" == *"step"* ]]; then
  step_http_args=(ros2 run omninav_step_scheduler step_http_client_node --ros-args -p config_file:="$SCHEDULER_CONFIG")
  if [ "$DECISION_STRESS_ENABLE" = "1" ]; then
    step_http_args+=(
      -p response_topic:=/stress/raw/step/response_json
      -p route_choice_topic:=/stress/raw/step/route_choice_json
      -p semantic_stop_topic:=/stress/raw/step/semantic_stop_json)
  fi
  start_bg step_http "${step_http_args[@]}"
fi
if [ "$DECISION_STRESS_ENABLE" = "1" ]; then
  start_bg delay_step_response ros2 run isaac_vln_benchmark delay_injector_node --ros-args \
    -r __node:=delay_step_response -p input_topic:=/stress/raw/step/response_json -p output_topic:=/step/response_json \
    -p delay_sec:="$DECISION_STRESS_DELAY_SEC" -p drop_rate:="$DECISION_STRESS_DROP_RATE" -p profile:="$DECISION_STRESS_PROFILE" -p seed:=41
  start_bg delay_step_route ros2 run isaac_vln_benchmark delay_injector_node --ros-args \
    -r __node:=delay_step_route -p input_topic:=/stress/raw/step/route_choice_json -p output_topic:=/step/route_choice_json \
    -p delay_sec:="$DECISION_STRESS_DELAY_SEC" -p drop_rate:="$DECISION_STRESS_DROP_RATE" -p profile:="$DECISION_STRESS_PROFILE" -p seed:=42
  start_bg delay_step_stop ros2 run isaac_vln_benchmark delay_injector_node --ros-args \
    -r __node:=delay_step_stop -p input_topic:=/stress/raw/step/semantic_stop_json -p output_topic:=/step/semantic_stop_json \
    -p delay_sec:="$DECISION_STRESS_DELAY_SEC" -p drop_rate:="$DECISION_STRESS_DROP_RATE" -p profile:="$DECISION_STRESS_PROFILE" -p seed:=43
  if [ "$DECISION_STRESS_RESET_ON_RAW" = "1" ]; then
    start_bg reset_on_decision ros2 run isaac_vln_benchmark reset_on_decision_node --ros-args -p delay_sec:=0.10
  fi
fi
if [ "$MOCK_OMNINAV" = "1" ]; then
  start_bg mock_omninav ros2 run omninav_step_scheduler mock_omninav_client_node
fi
adapter_cmd=(ros2 run isaac_vln_benchmark go2_benchmark_adapter_node)
adapter_ros_args=()
adapter_ros_args+=(-p publish_synthetic_camera_fallback:="$([ "$ALLOW_SYNTHETIC_CAMERA_FALLBACK" = "1" ] && echo true || echo false)")
if [ -n "$TASKS_FILE" ]; then
  adapter_ros_args+=(-p tasks_path:="$TASKS_FILE")
fi
if [ -n "$SCENES_FILE" ]; then
  adapter_ros_args+=(-p scenes_path:="$SCENES_FILE")
fi
if [ "${PREFER_SYNTHETIC_CAMERA:-0}" = "1" ]; then
  adapter_ros_args+=(-p prefer_synthetic_camera_for_primary:=true)
fi
if [ "${#adapter_ros_args[@]}" -gt 0 ]; then
  adapter_cmd+=(--ros-args "${adapter_ros_args[@]}")
fi
start_bg adapter "${adapter_cmd[@]}"
if [[ "$MODES" == *"forced_semantic_oracle"* ]]; then
  if [ -z "$TASKS_FILE" ] || [ -z "$SCENES_FILE" ]; then
    echo "forced_semantic_oracle requires TASKS_FILE and SCENES_FILE" >&2
    exit 2
  fi
  start_bg forced_semantic_oracle ros2 run isaac_vln_benchmark forced_semantic_oracle_node --ros-args \
    -p task_file:="$TASKS_FILE"
fi
if [[ "$MODES" == *"public_heuristic"* ]]; then
  start_bg public_semantic_heuristic ros2 run isaac_vln_benchmark public_semantic_heuristic_node
fi
if [[ "$MODES" == *"semantic_screen"* || "$MODES" == *"forced_semantic_oracle"* || "$MODES" == *"semantic_executive"* ]]; then
  if [ -z "$TASKS_FILE" ] || [ -z "$SCENES_FILE" ]; then
    echo "semantic marker judge requires TASKS_FILE and SCENES_FILE" >&2
    exit 2
  fi
  start_bg semantic_marker_judge ros2 run isaac_vln_benchmark semantic_marker_judge_node --ros-args \
    -p task_file:="$TASKS_FILE" -p scenes_file:="$SCENES_FILE"
fi
if [ "$ENABLE_GROUNDED_SAM" = "1" ]; then
  start_bg grounded_sam_perception ros2 run isaac_vln_benchmark grounded_sam_perception_node --ros-args \
    -p endpoint:="$GROUNDED_SAM_ENDPOINT"
fi

if [ "$QUALIFICATION_BAG_ENABLE" = "1" ]; then
  start_bg qualification_bag ros2 bag record -o "$RUN_DIR/qualification_bag" \
    /benchmark/mode_json /isaac/reset_episode /isaac/reset_ack_json \
    /isaac/ground_truth_pose /isaac/episode_status /isaac/collision_event \
    /step/request_json /step/response_json /step/route_choice_json /step/semantic_stop_json \
    /stress/raw/step/response_json /stress/raw/step/route_choice_json /stress/raw/step/semantic_stop_json \
    /benchmark/delay_audit_json /benchmark/reset_stress_json \
    /camera/front/isaac_image /camera/front/depth /camera/front/camera_info \
    /perception/target_observation_json /perception/target_track_json /perception/target_mask \
    /planner/route_decision_json /primitive/command_json /safe_cmd_vel /metrics/event_jsonl
fi

sleep 6

if [ "$OMNINAV_FORMAL_PREFLIGHT" = "1" ]; then
  : "${OMNINAV_EXPECTED_MODEL_VARIANT:?required for formal preflight}"
  : "${OMNINAV_EXPECTED_PRECISION:?required for formal preflight}"
  : "${OMNINAV_SEED_MANIFEST:?required for formal preflight}"
  : "${OMNINAV_PREFLIGHT_OUTPUT:?required for formal preflight}"
  PYTHONPATH="/home/song/dgx-unitree:${PYTHONPATH:-}" python3 \
    /home/song/dgx-unitree/scripts/preflight_isaac_backtest.py \
    --endpoint "${OMNINAV_ENDPOINT:-tcp://10.100.100.128:8100}" \
    --expected-model-variant "$OMNINAV_EXPECTED_MODEL_VARIANT" \
    --expected-precision "$OMNINAV_EXPECTED_PRECISION" \
    --seed-manifest "$OMNINAV_SEED_MANIFEST" \
    --camera-topic /camera/front/isaac_image \
    --output "$OMNINAV_PREFLIGHT_OUTPUT" \
    > "$RUN_DIR/formal_preflight.log" 2>&1
fi

if [[ "$MODES" == *"internnav"* && "$MOCK_MODELS" != "1" ]]; then
  hostport="${INTERNNAV_SERVER#http://}"
  hostport="${hostport#https://}"
  internnav_host="${hostport%%:*}"
  internnav_port="${hostport##*:}"
  if [ "$internnav_port" = "$hostport" ]; then
    internnav_port=8087
  fi
  start_bg internnav_client ros2 run omninav_step_scheduler internnav_go2_client_node \
    --ros-args \
    --params-file /home/song/dgx-unitree/ros2_ws/install/omninav_step_scheduler/share/omninav_step_scheduler/config/internnav_go2_direct.yaml \
    -p publish_cmd_vel:=false \
    -p synthetic_fallback:=false \
    -p server_host:="$internnav_host" \
    -p server_port:="$internnav_port"
  sleep 4
fi

read -r -a mode_args <<< "$MODES"
cmd=(python3 /home/song/dgx-unitree/isaac_vln_benchmark/scripts/run_benchmark.py
  --use-isaac
  --modes "${mode_args[@]}"
  --num-episodes "$NUM_EPISODES"
  --output "$RUN_DIR/out"
  --ros-domain-id "$ROS_DOMAIN_ID"
  --live-startup-timeout-sec "$LIVE_STARTUP_TIMEOUT_SEC"
  --live-settle-sec "$LIVE_SETTLE_SEC")
cmd+=(--seed "$BENCHMARK_SEED")
if [ "$ISAAC_HEADLESS_EVIDENCE" = "1" ]; then
  cmd+=(--isaac-headless-evidence)
fi
if [ -n "$TASKS_FILE" ]; then
  cmd+=(--tasks "$TASKS_FILE")
fi
if [ -n "$SCENES_FILE" ]; then
  cmd+=(--scenes "$SCENES_FILE")
fi
if [ -n "$BENCHMARK_CONFIG" ]; then
  cmd+=(--config "$BENCHMARK_CONFIG")
fi
if [ "$LIVE_EPISODE_TIMEOUT_SEC" != "0" ]; then
  cmd+=(--live-episode-timeout-sec "$LIVE_EPISODE_TIMEOUT_SEC")
fi
if [ "$MOCK_MODELS" = "1" ]; then
  cmd+=(--mock-models)
fi

"${cmd[@]}" > "$RUN_DIR/runner.log" 2>&1

if [ -f /home/song/isaac_projects/logs/go2_twist_bridge.log ]; then
  tail -160 /home/song/isaac_projects/logs/go2_twist_bridge.log > "$RUN_DIR/go2_twist_bridge_tail_final.txt"
fi
latest_isaac_log="$(
  ls -t /home/song/isaac_projects/logs/go2_warehouse_teleop_desktop*.out \
        /home/song/isaac_projects/logs/go2_warehouse_teleop_desktop*.log 2>/dev/null | head -1 || true
)"
if [ -n "$latest_isaac_log" ] && [ -f "$latest_isaac_log" ]; then
  echo "$latest_isaac_log" > "$RUN_DIR/isaac_go2_log_path.txt"
  grep -E "TELEOP_STEP|PERF_STEP|BENCHMARK_TELEMETRY_READY|BENCHMARK_CONTROL_READY|BENCHMARK_RESET" "$latest_isaac_log" \
    | tail -160 > "$RUN_DIR/isaac_go2_tail_final.txt" || true
fi

echo "$RUN_DIR"
