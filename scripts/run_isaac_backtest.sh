#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/song/dgx-unitree}"
PHASE="${OMNINAV_BACKTEST_PHASE:-formal}"
EXPECTED_MODEL="${OMNINAV_EXPECTED_MODEL_VARIANT:?set OMNINAV_EXPECTED_MODEL_VARIANT}"
EXPECTED_PRECISION="${OMNINAV_EXPECTED_PRECISION:?set OMNINAV_EXPECTED_PRECISION}"
RUN_LABEL="${OMNINAV_RUN_LABEL:-${EXPECTED_MODEL//[^a-zA-Z0-9_.-]/_}_${EXPECTED_PRECISION}}"
RUN_ID="${OMNINAV_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${OMNINAV_RUN_DIR:-$PROJECT_ROOT/results/omninav_cosmos/${PHASE}_${RUN_LABEL}_${RUN_ID}}"
ENDPOINT="${OMNINAV_ENDPOINT:-tcp://10.100.100.128:8100}"
REMOTE_CONFIG="${OMNINAV_REMOTE_CONFIG:-$PROJECT_ROOT/configs/isaac/remote_client.yaml}"
SEEDS="$PROJECT_ROOT/configs/isaac/seeds.json"
SCENES="$PROJECT_ROOT/configs/isaac/scenes.yaml"
if [ "$PHASE" = "smoke" ]; then
  TASKS="$PROJECT_ROOT/configs/isaac/smoke_tasks.yaml"
  NUM_EPISODES=5
elif [ "$PHASE" = "formal" ]; then
  TASKS="$PROJECT_ROOT/configs/isaac/tasks.yaml"
  NUM_EPISODES=130
else
  echo "OMNINAV_BACKTEST_PHASE must be smoke or formal" >&2
  exit 2
fi

mkdir -p "$RUN_DIR/videos"
cp "$PROJECT_ROOT/configs/isaac/backtest.yaml" "$RUN_DIR/config.yaml"
cp "$SEEDS" "$RUN_DIR/seeds.json"
cp "$TASKS" "$RUN_DIR/tasks.yaml"
cp "$SCENES" "$RUN_DIR/scenes.yaml"

if pgrep -f '[g]o2_warehouse_waypoint_nav.py' >/dev/null; then
  echo "Refusing to reuse an existing Isaac process. Stop it, then rerun so the formal command and log are unambiguous." >&2
  exit 2
fi

source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

(
  cd "$PROJECT_ROOT/ros2_ws"
  colcon build --packages-select omninav_step_scheduler
) > "$RUN_DIR/build_omninav_bridge.log" 2>&1
(
  cd "$PROJECT_ROOT/isaac_vln_benchmark/ros2_ws"
  colcon build --packages-select isaac_vln_benchmark
) > "$RUN_DIR/build_isaac_benchmark.log" 2>&1
source "$PROJECT_ROOT/ros2_ws/install/setup.bash"
source "$PROJECT_ROOT/isaac_vln_benchmark/ros2_ws/install/setup.bash"

ISAAC_PYTHON="${ISAAC_PYTHON:-/home/song/env_isaacsim/bin/python3}"
ISAAC_SCRIPT="$PROJECT_ROOT/isaac_vln_benchmark/isaac_host/go2_warehouse_waypoint_nav.py"
isaac_command=(
  "$ISAAC_PYTHON" "$ISAAC_SCRIPT"
  --headless --enable_cameras --device cuda:0
  --control_mode ros_twist --ideal_kinematic_base --hold_open --real_time
  --cmd_udp_host 127.0.0.1 --cmd_udp_port 15002 --cmd_timeout 0.45
  --benchmark_udp_host 127.0.0.1 --benchmark_udp_port 15010 --benchmark_publish_hz 20
  --control_udp_host 127.0.0.1 --control_udp_port 15011
  --camera_udp_host 127.0.0.1 --camera_udp_port 15012 --camera_publish_hz 5
  --camera_width 640 --camera_height 360
  --linear_speed 0.20 --max_yaw_rate 0.35 --min_stable_linear_command 0
  --disable_base_contact_termination --max_steps 1000000000
  --trace_csv "$RUN_DIR/isaac_trace.csv" --trace_every 5 --perf_every 200
)
printf '%q ' "${isaac_command[@]}" > "$RUN_DIR/commands.txt"
printf '\n' >> "$RUN_DIR/commands.txt"
"${isaac_command[@]}" > "$RUN_DIR/isaac.log" 2>&1 &
isaac_pid=$!
echo "$isaac_pid" > "$RUN_DIR/isaac.pid"

remote_pid=""
video_pid=""
cleanup() {
  set +e
  if [ -n "$video_pid" ]; then kill -TERM "$video_pid" 2>/dev/null || true; wait "$video_pid" 2>/dev/null || true; fi
  if [ -n "$remote_pid" ]; then kill -TERM "$remote_pid" 2>/dev/null || true; wait "$remote_pid" 2>/dev/null || true; fi
  kill -TERM "$isaac_pid" 2>/dev/null || true
  wait "$isaac_pid" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 180); do
  if ! kill -0 "$isaac_pid" 2>/dev/null; then
    echo "Isaac exited during startup" >&2
    exit 1
  fi
  if grep -q 'CAMERA_TELEMETRY_READY' "$RUN_DIR/isaac.log" && grep -q 'BENCHMARK_CONTROL_READY' "$RUN_DIR/isaac.log"; then
    break
  fi
  sleep 1
done
grep -q 'CAMERA_TELEMETRY_READY' "$RUN_DIR/isaac.log"
grep -q 'BENCHMARK_CONTROL_READY' "$RUN_DIR/isaac.log"

ros2 run omninav_step_scheduler omninav_remote_model_client_node --ros-args \
  -p config_file:="$REMOTE_CONFIG" > "$RUN_DIR/remote_model_client.log" 2>&1 &
remote_pid=$!
echo "$remote_pid" > "$RUN_DIR/remote_model_client.pid"

python3 "$PROJECT_ROOT/scripts/record_ros_video.py" \
  --topic /camera/front/isaac_image \
  --output "$RUN_DIR/videos/representative.mp4" \
  --metadata "$RUN_DIR/videos/representative.json" \
  --fps 5 --max-duration-sec "${OMNINAV_VIDEO_MAX_DURATION_SEC:-300}" \
  > "$RUN_DIR/video_recorder.log" 2>&1 &
video_pid=$!
echo "$video_pid" > "$RUN_DIR/video_recorder.pid"

python3 "$PROJECT_ROOT/scripts/capture_runtime_versions.py" \
  --workspace "$PROJECT_ROOT" --output "$RUN_DIR/versions.json" \
  > "$RUN_DIR/versions.log" 2>&1

export NUM_EPISODES MODES=omninav_only MOCK_MODELS=0 MOCK_STEP=0 MOCK_OMNINAV=0
export TASKS_FILE="$TASKS" SCENES_FILE="$SCENES" BENCHMARK_SEED=2026071300
export ALLOW_SYNTHETIC_CAMERA_FALLBACK=0 ISAAC_HEADLESS_EVIDENCE=1 ENABLE_GROUNDED_SAM=0
export OMNINAV_FORMAL_PREFLIGHT=1 OMNINAV_EXPECTED_MODEL_VARIANT="$EXPECTED_MODEL"
export OMNINAV_EXPECTED_PRECISION="$EXPECTED_PRECISION" OMNINAV_SEED_MANIFEST="$SEEDS"
export OMNINAV_PREFLIGHT_OUTPUT="$RUN_DIR/model_health.json" OMNINAV_ENDPOINT="$ENDPOINT"
printf 'OMNINAV_BACKTEST_PHASE=%q NUM_EPISODES=%q TASKS_FILE=%q SCENES_FILE=%q %q %q\n' \
  "$PHASE" "$NUM_EPISODES" "$TASKS" "$SCENES" \
  "$PROJECT_ROOT/isaac_vln_benchmark/scripts/run_live_benchmark_epyc.sh" "$RUN_DIR/live" \
  >> "$RUN_DIR/commands.txt"

"$PROJECT_ROOT/isaac_vln_benchmark/scripts/run_live_benchmark_epyc.sh" "$RUN_DIR/live"

if kill -0 "$video_pid" 2>/dev/null; then
  kill -TERM "$video_pid"
  wait "$video_pid"
fi
video_pid=""
cp "$RUN_DIR/live/out/episodes.jsonl" "$RUN_DIR/episodes.jsonl"
cp "$RUN_DIR/live/out/config.yaml" "$RUN_DIR/runner_config.yaml"
cp "$RUN_DIR/live/runner.log" "$RUN_DIR/runner.log"

python3 "$PROJECT_ROOT/scripts/validate_isaac_backtest.py" \
  --run-dir "$RUN_DIR" --phase "$PHASE" \
  --expected-model-variant "$EXPECTED_MODEL" \
  --expected-precision "$EXPECTED_PRECISION" \
  > "$RUN_DIR/validation.log" 2>&1

echo "$RUN_DIR"
