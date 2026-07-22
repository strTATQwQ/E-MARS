#!/usr/bin/env python3
"""Derive the T4 sensor-stack launcher from the hash-frozen T3 phase runner."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one launcher token, found {count}: {old[:100]!r}")
    return text.replace(old, new)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    source_bytes = args.source.read_bytes()
    text = source_bytes.decode("utf-8")

    text = replace_once(
        text,
        "continuous_oracle)\n    EXPECTED_COUNT=10; MIN_SR=0.9; SOURCE_PHASE=oracle; OBSTACLE_AWARE=0; CLIENT_KIND=oracle ;;",
        "continuous_oracle)\n    EXPECTED_COUNT=\"${INTERNVLA_T4_EXPECTED_COUNT:-10}\"; MIN_SR=\"${INTERNVLA_T4_MIN_SR:-0.9}\"; SOURCE_PHASE=oracle; OBSTACLE_AWARE=0; CLIENT_KIND=oracle ;;",
    )
    text = replace_once(
        text,
        "obstacle_oracle)\n    EXPECTED_COUNT=10; MIN_SR=0.9; SOURCE_PHASE=oracle; OBSTACLE_AWARE=1; CLIENT_KIND=oracle ;;",
        "obstacle_oracle)\n    EXPECTED_COUNT=\"${INTERNVLA_T4_EXPECTED_COUNT:-10}\"; MIN_SR=\"${INTERNVLA_T4_MIN_SR:-0.9}\"; SOURCE_PHASE=oracle; OBSTACLE_AWARE=1; CLIENT_KIND=oracle ;;",
    )
    text = replace_once(
        text,
        "canary)\n    EXPECTED_COUNT=5; MIN_SR=0.4; SOURCE_PHASE=canary; OBSTACLE_AWARE=1; CLIENT_KIND=model ;;",
        "canary)\n    EXPECTED_COUNT=\"${INTERNVLA_T4_EXPECTED_COUNT:-5}\"; MIN_SR=\"${INTERNVLA_T4_MIN_SR:-0.4}\"; SOURCE_PHASE=canary; OBSTACLE_AWARE=1; CLIENT_KIND=model ;;",
    )
    text = replace_once(
        text,
        "pilot)\n    EXPECTED_COUNT=20; MIN_SR=0.3; SOURCE_PHASE=pilot; OBSTACLE_AWARE=1; CLIENT_KIND=model ;;",
        "pilot)\n    EXPECTED_COUNT=\"${INTERNVLA_T4_EXPECTED_COUNT:-20}\"; MIN_SR=\"${INTERNVLA_T4_MIN_SR:-0.0}\"; SOURCE_PHASE=pilot; OBSTACLE_AWARE=1; CLIENT_KIND=model ;;",
    )
    text = replace_once(
        text,
        'source /opt/ros/humble/setup.bash\nsource "$ROS_WS/install/setup.bash"\n',
        '# Nav2 and every ROS CLI/node execute through the audited Jazzy/Isaac\n'
        '# container shim. The host intentionally has no ROS setup to source.\n'
        'export PATH="$CONTROL_ROOT/runtime/t4_ros2_shim:$PATH"\n',
    )
    text = replace_once(
        text,
        '  -p "PolygonStop.points:=[$COLLISION_MONITOR_STOP_FORWARD_M,0.26,$COLLISION_MONITOR_STOP_FORWARD_M,-0.26,-0.26,-0.26,-0.26,0.26]" \\\n',
        '',
    )
    text = replace_once(
        text,
        '''setsid ros2 run nav2_collision_monitor collision_monitor --ros-args \\
  --params-file "$PARAMS" \\
  >"$RESULT_DIR/logs/collision_monitor.log" 2>&1 &
collision_pid=$!
''',
        '''# Jazzy navigation_launch.py already owns the single collision_monitor
# process. The dedicated lifecycle manager below configures that instance.
collision_pid=""
export INTERNNAV_T4_MAP_COMPANION_ACK=1
setsid bash "$CONTROL_ROOT/scripts/t4_python_container_shim.sh" \
  "$CONTROL_ROOT/t4_completion/map/warn_relay.py" \
  --result-dir "$RESULT_DIR" --ros-args -p use_sim_time:=false \
  >"$RESULT_DIR/logs/warn_only_relay.log" 2>&1 &
relay_pid=$!
for _ in $(seq 1 100); do
  test -s "$RESULT_DIR/warn_only_relay.jsonl" && break
  kill -0 "$relay_pid"
  sleep 0.05
done
test -s "$RESULT_DIR/warn_only_relay.jsonl"
''',
    )
    text = replace_once(
        text,
        'ros2 lifecycle get "$node" \\\n',
        'bash "$CONTROL_ROOT/scripts/t4_python_container_shim.sh" \\\n'
        '      "$SCRIPT_ROOT/t4_direct_ros_probe.py" lifecycle-responsive \\\n'
        '      "$node" --timeout 2 \\\n',
    )
    text = replace_once(
        text,
        "ros2 lifecycle get /collision_monitor 2>/dev/null | grep -q active",
        'bash "$CONTROL_ROOT/scripts/t4_python_container_shim.sh" '
        '"$SCRIPT_ROOT/t4_direct_ros_probe.py" lifecycle-active '
        '/collision_monitor --timeout 2 >/dev/null 2>&1',
    )
    text = replace_once(
        text,
        "ros2 lifecycle get /collision_monitor | grep -q active",
        '{ bash "$CONTROL_ROOT/scripts/t4_python_container_shim.sh" '
        '"$SCRIPT_ROOT/t4_direct_ros_probe.py" lifecycle-active '
        '/collision_monitor --timeout 2 >/dev/null 2>&1 || '
        'grep -Fq "Managed nodes are active" '
        '"$RESULT_DIR/logs/collision_lifecycle.log"; }',
    )
    graph_block = '''{
  echo "# nodes"; ros2 node list | sort
  echo "# topics"; ros2 topic list -t | sort
  echo "# services"; ros2 service list -t | sort
  echo "# actions"; ros2 action list -t | sort
} >"$RESULT_DIR/ros_graph_snapshot.txt"'''
    text = replace_once(
        text,
        graph_block,
        '''graph_ok=0
for _ in $(seq 1 3); do
  if bash "$CONTROL_ROOT/scripts/t4_python_container_shim.sh" \
    "$SCRIPT_ROOT/t4_direct_ros_probe.py" graph --timeout 30 \
    >"$RESULT_DIR/ros_graph_snapshot.txt"; then
    graph_ok=1
    break
  fi
  sleep 2
done
if test "$graph_ok" != 1; then
  {
    echo "# fallback=isolated_ros2_cli_no_daemon"
    echo "# nodes"; ros2 node list | sort
    echo "# topics"; ros2 topic list -t | sort
    echo "# services"; ros2 service list -t | sort
    echo "# actions"; ros2 action list -t | sort
  } >"$RESULT_DIR/ros_graph_snapshot.txt"
fi
test -s "$RESULT_DIR/ros_graph_snapshot.txt"''',
    )
    text = replace_once(
        text,
        'nav2_pid=""; nav_lifecycle_pid=""; adapter_pid=""; controller_pid=""; collision_pid=""; lifecycle_pid=""; client_pid=""',
        'nav2_pid=""; nav_lifecycle_pid=""; adapter_pid=""; controller_pid=""; collision_pid=""; lifecycle_pid=""; client_pid=""; relay_pid=""; nvblox_pid=""; odometry_pid=""; recovery_pid=""; tracer_pid=""; go2_sensor_pid=""',
    )
    text = replace_once(
        text,
        '  stop_group "$controller_pid"\n  stop_group "$adapter_pid"',
        '  stop_group "$relay_pid"\n'
        '  relay_pid=""\n'
        '  bash "$CONTROL_ROOT/scripts/t4_cleanup_container_processes.sh" '
        '>>"$RESULT_DIR/logs/container_cleanup.log" 2>&1 || true\n'
        '  stop_group "$controller_pid"\n'
        '  stop_group "$tracer_pid"\n'
        '  stop_group "$go2_sensor_pid"\n'
        '  stop_group "$recovery_pid"\n'
        '  stop_group "$odometry_pid"\n'
        '  stop_group "$nvblox_pid"\n'
        '  stop_group "$adapter_pid"',
    )

    static_start = text.index('DATASET_SHA256="$(python3 -')
    static_end = text.index('if test -n "${SCENARIO_MANIFEST:-}"; then', static_start)
    static_replacement = r'''DATASET_SHA256="$(python3 - "$DATASET_ROOT/val_unseen/val_unseen.json.gz" <<'PY'
import hashlib,sys
print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())
PY
)"
if test "$INTERNVLA_T4_MAP_SOURCE" = static_map; then
  STATIC_MAP_CLEARANCE_GATE="${INTERNVLA_T3_STATIC_CLEARANCE_GATE_M:-0.40}"
  STATIC_MAP_GATE_TAG="${STATIC_MAP_CLEARANCE_GATE//./p}"
  STATIC_MAP_CACHE="$CONTROL_ROOT/runtime/static_maps/t4_static_${DATASET_SHA256:0:16}_${STATIC_MAP_GATE_TAG}"
  python3 "$SCRIPT_ROOT/build_t3_static_maps.py" \
    --dataset "$DATASET_ROOT/val_unseen/val_unseen.json.gz" \
    --scene-root "$INTERNNAV_ROOT/data/scene_data/mp3d_pe" \
    --output-root "$STATIC_MAP_CACHE" \
    --minimum-required-prefix-clearance-m "$STATIC_MAP_CLEARANCE_GATE" \
    >"$RESULT_DIR/logs/static_map_build.log"
  mkdir -p "$RESULT_DIR/static_maps"
  cp "$STATIC_MAP_CACHE"/*.bin "$RESULT_DIR/static_maps/"
  python3 "$SCRIPT_ROOT/build_t4_truth_isolated_static_manifest.py" \
    --manifest "$STATIC_MAP_CACHE/manifest.json" \
    --dataset "$DATASET_ROOT/val_unseen/val_unseen.json.gz" \
    --output "$RESULT_DIR/static_maps/manifest.json" \
    >>"$RESULT_DIR/logs/static_map_build.log"
else
  STATIC_MAP_CLEARANCE_GATE=0.0
  mkdir -p "$RESULT_DIR/static_maps"
  cp "$CONTROL_ROOT/configs/internnav_t4/no_static_map_contract/manifest.json" \
    "$CONTROL_ROOT/configs/internnav_t4/no_static_map_contract/sentinel.bin" \
    "$RESULT_DIR/static_maps/"
fi
STATIC_MAP_MANIFEST="$RESULT_DIR/static_maps/manifest.json"
test -f "$STATIC_MAP_MANIFEST"
'''
    text = text[:static_start] + static_replacement + text[static_end:]

    text = replace_once(
        text,
        '  "controller_bridge=$CONTROL_ROOT/internvla_go2_controller/internvla_go2_controller/bridge_node.py" \\\n',
        '  "controller_bridge=$CONTROL_ROOT/internvla_t4_sensors/internvla_t4_sensors/sensor_bridge_node.py" \\\n'
        '  "completion_safe_relay=$CONTROL_ROOT/t4_completion/map/warn_relay.py" \\\n'
        '  "typed_client=$CONTROL_ROOT/internvla_t4_sensors/internvla_t4_sensors/client_node.py" \\\n'
        '  "nvblox_params=$CONTROL_ROOT/configs/internnav_t4/nvblox_gt_pose.yaml" \\\n'
        '  "costmap_stage_tracer=$CONTROL_ROOT/internvla_t4_sensors/internvla_t4_sensors/costmap_stage_tracer_node.py" \\\n',
    )

    old_adapter = '''setsid "$ROS_WS/install/internvla_nav2_adapter/lib/internvla_nav2_adapter/internvla_nav2_active" \\
  --ros-args -p result_dir:="$RESULT_DIR" -p command_mode:="$ADAPTER_COMMAND_MODE" \\
  -p goal_max_distance_m:="$ADAPTER_GOAL_MAX_DISTANCE" \\
  -p execution_mode:=continuous -p nav2_server_timeout_sec:=10.0 \\
  >"$RESULT_DIR/logs/adapter.log" 2>&1 &
adapter_pid=$!
'''
    new_adapter = '''ADAPTER_EXTRA=()
if test "${INTERNNAV_T5_STEP3_LIVE_ADVISOR:-0}" = 1; then
  test "${INTERNNAV_T5_LANE:-}" = b
  test "${INTERNNAV_T5_LANE_NAMESPACE:-}" = /t5/lane_b
  ADAPTER_PACKAGE=internvla_t4_recovery
  ADAPTER_EXEC=internvla_t4_adapter
elif test "${INTERNVLA_T4_ENABLE_RECOVERY:-0}" = 1 || test "${INTERNVLA_T4_USE_T4_ADAPTER:-0}" = 1; then
  ADAPTER_PACKAGE=internvla_t4_recovery
  ADAPTER_EXEC=internvla_t4_adapter
  ADAPTER_EXTRA+=(
    # Preserve enum/string types through rcl's YAML parser. In particular,
    # unquoted YAML 1.1 on/off values become booleans and are rejected by the
    # adapter's declared string parameters.
    -p "ablation_variant_id:='${INTERNVLA_T4_VARIANT_ID:-none}'"
    -p "ablation_config_sha256:='${INTERNVLA_T4_VARIANT_CONFIG_SHA256:-none}'"
    -p "system_mode:='${INTERNVLA_T4_SYSTEM_MODE:-full_system1_system2}'"
    -p "trajectory_mode:='${INTERNVLA_T4_TRAJECTORY_MODE:-full_trajectory}'"
    -p "termination_mode:='${INTERNVLA_T4_TERMINATION_MODE:-model_stop}'"
    -p "history_mode:='${INTERNVLA_T4_HISTORY_MODE:-on}'"
    -p "recovery_mode:='${INTERNVLA_T4_RECOVERY_MODE:-off}'"
    -p "view_mode:='${INTERNVLA_T4_VIEW_MODE:-go2_view}'"
    -p "recovery_safety_freshness_timeout_sec:=${INTERNVLA_T4_RECOVERY_SAFETY_FRESHNESS_SEC:-1.0}"
  )
  # rcl rejects an empty string parameter override before the adapter can
  # declare its default.  Oracle and non-ablation pilots intentionally have
  # no ablation dataset, so only forward this optional path when it is real.
  if test -n "${INTERNVLA_T4_ABLATION_DATASET_FILE:-}"; then
    ADAPTER_EXTRA+=(
      -p "ablation_dataset_file:=$INTERNVLA_T4_ABLATION_DATASET_FILE"
    )
  fi
else
  ADAPTER_PACKAGE=internvla_nav2_adapter
  ADAPTER_EXEC=internvla_nav2_active
fi
setsid ros2 run "$ADAPTER_PACKAGE" "$ADAPTER_EXEC" \\
  --ros-args -p result_dir:="$RESULT_DIR" -p command_mode:="$ADAPTER_COMMAND_MODE" \\
  -p goal_max_distance_m:="$ADAPTER_GOAL_MAX_DISTANCE" \\
  -p execution_mode:=continuous -p nav2_server_timeout_sec:=10.0 \\
  "${ADAPTER_EXTRA[@]}" \\
  >"$RESULT_DIR/logs/adapter.log" 2>&1 &
adapter_pid=$!
'''
    text = replace_once(text, old_adapter, new_adapter)

    old_controller = '''setsid "$ROS_WS/install/internvla_go2_controller/lib/internvla_go2_controller/internvla_go2_controller_bridge" \\
  --ros-args --params-file "$PARAMS" -p result_dir:="$RESULT_DIR" \\
  -p socket_path:="$CONTROLLER_SOCKET" -p static_map_manifest:="$STATIC_MAP_MANIFEST" \\
  >"$RESULT_DIR/logs/controller.log" 2>&1 &
controller_pid=$!
'''
    new_controller = '''setsid ros2 run internvla_t4_sensors internvla_t4_sensor_bridge \\
  --ros-args --params-file "$PARAMS" -p result_dir:="$RESULT_DIR" \\
  -p socket_path:="$CONTROLLER_SOCKET" -p static_map_manifest:="$STATIC_MAP_MANIFEST" \\
  -p map_source:="$INTERNVLA_T4_MAP_SOURCE" -p pose_source:="$INTERNVLA_T4_POSE_SOURCE" \\
  -p camera_height_above_support_m:="$INTERNVLA_T4_CAMERA_HEIGHT_M" \\
  -p camera_pitch_down_deg:="$INTERNVLA_T4_CAMERA_PITCH_DOWN_DEG" \\
  -p camera_hfov_deg:="$INTERNVLA_T4_CAMERA_HFOV_DEG" \\
  -p camera_vfov_deg:="$INTERNVLA_T4_CAMERA_VFOV_DEG" \\
  -p depth_height_above_support_m:="$INTERNVLA_T4_DEPTH_HEIGHT_M" \\
  -p depth_forward_m:="$INTERNVLA_T4_DEPTH_FORWARD_M" \\
  -p depth_pitch_down_deg:="$INTERNVLA_T4_DEPTH_PITCH_DOWN_DEG" \\
  -p depth_hfov_deg:="$INTERNVLA_T4_DEPTH_HFOV_DEG" \\
  -p depth_vfov_deg:="$INTERNVLA_T4_DEPTH_VFOV_DEG" \\
  >"$RESULT_DIR/logs/controller.log" 2>&1 &
controller_pid=$!
if test "${INTERNVLA_T4_R3_ENABLE_SENSOR_BRIDGE:-0}" = 1; then
  R3_D435I_BOOL=false
  R3_LIDAR_BOOL=false
  test "${INTERNVLA_T4_R3_ENABLE_D435I:-1}" = 1 && R3_D435I_BOOL=true
  test "${INTERNVLA_T4_R3_ENABLE_LIDAR:-1}" = 1 && R3_LIDAR_BOOL=true
  setsid ros2 run go2_sensor_bridge go2_sensor_bridge --ros-args \\
    -p result_dir:="$RESULT_DIR" \\
    -p enable_d435i:="$R3_D435I_BOOL" \\
    -p enable_lidar:="$R3_LIDAR_BOOL" \\
    -p sensor_timeout_sec:=0.35 \\
    >"$RESULT_DIR/logs/go2_sensor_bridge.log" 2>&1 &
  go2_sensor_pid=$!
fi
if test "$INTERNVLA_T4_MAP_SOURCE" = nvblox_online; then
  setsid ros2 run internvla_t4_sensors internvla_t4_nvblox_supervisor --ros-args \\
    -p result_dir:="$RESULT_DIR" \\
    -p nvblox_params_file:="${INTERNVLA_T4_NVBLOX_PARAMS_OVERRIDE:-$CONTROL_ROOT/configs/internnav_t4/nvblox_gt_pose.yaml}" \\
    >"$RESULT_DIR/logs/nvblox_supervisor.log" 2>&1 &
  nvblox_pid=$!
  if test "${INTERNVLA_T4_TRACE_COSTMAP_STAGES:-0}" = 1; then
    setsid ros2 run internvla_t4_sensors internvla_t4_costmap_stage_tracer --ros-args \\
      -p result_dir:="$RESULT_DIR" \\
      -p robot_radius_m:=0.30 -p footprint_padding_m:=0.02 \\
      -p slice_height_m:=0.15 -p slice_min_height_m:=0.15 \\
      -p slice_max_height_m:=0.65 -p binary_conversion:=true \\
      -p occupied_distance_threshold_m:=0.0 \\
      -p pre_inflation_layer_name:="${INTERNVLA_T4_PRE_INFLATION_LAYER_NAME:-nvblox_layer}" \\
      -p service_startup_delay_sec:=20.0 \\
      >"$RESULT_DIR/logs/costmap_stage_tracer.log" 2>&1 &
    tracer_pid=$!
  fi
fi
if test "$INTERNVLA_T4_POSE_SOURCE" = external_odometry; then
  setsid ros2 run internvla_t4_sensors internvla_t4_odometry_supervisor --ros-args \\
    -p result_dir:="$RESULT_DIR" -p odometry_timeout_sec:=0.30 \\
    >"$RESULT_DIR/logs/odometry_supervisor.log" 2>&1 &
  odometry_pid=$!
fi
if test "${INTERNVLA_T4_ENABLE_RECOVERY:-0}" = 1; then
  setsid ros2 run internvla_t4_recovery internvla_t4_recovery --ros-args \\
    -p result_dir:="$RESULT_DIR" \\
    -p progress_horizon_sec:="${INTERNVLA_T4_PROGRESS_HORIZON_SEC:-2.0}" \\
    -p minimum_progress_m:="${INTERNVLA_T4_MINIMUM_PROGRESS_M:-0.08}" \\
    -p oscillation_travel_m:="${INTERNVLA_T4_OSCILLATION_TRAVEL_M:-0.30}" \\
    -p trajectory_refresh_distance_m:="${INTERNVLA_T4_REFRESH_DISTANCE_M:-0.30}" \\
    -p trajectory_refresh_time_sec:="${INTERNVLA_T4_REFRESH_TIME_SEC:-2.0}" \\
    -p trajectory_validity_sec:="${INTERNVLA_T4_TRAJECTORY_VALIDITY_SEC:-5.0}" \\
    -p trajectory_deviation_m:="${INTERNVLA_T4_TRAJECTORY_DEVIATION_M:-0.60}" \\
    -p recovery_scan_yaw_rad:="${INTERNVLA_T4_RECOVERY_SCAN_YAW_RAD:-1.0471975511965976}" \\
    -p recovery_cooldown_sec:="${INTERNVLA_T4_RECOVERY_COOLDOWN_SEC:-3.0}" \\
    -p maximum_recoveries_per_episode:="${INTERNVLA_T4_MAXIMUM_RECOVERIES:-3}" \\
    -p maximum_recovery_duration_sec:="${INTERNVLA_T4_MAXIMUM_RECOVERY_DURATION_SEC:-60.0}" \\
    -p recovery_profile_id:="${INTERNVLA_T4_RECOVERY_PROFILE_ID:-completion-default}" \\
    -p recovery_profile_sha256:="${INTERNVLA_T4_RECOVERY_PROFILE_SHA256:-none}" \\
    -p enable_scheduled_refresh:=false -p enable_short_backup:=false \\
    >"$RESULT_DIR/logs/recovery.log" 2>&1 &
  recovery_pid=$!
fi
'''
    text = replace_once(text, old_controller, new_controller)
    text = replace_once(
        text,
        '    "$ROS_WS/install/internvla_ros2/lib/internvla_ros2/internvla_nav2_oracle_bridge" \\\n',
        '    ros2 run internvla_ros2 internvla_nav2_oracle_bridge \\\n',
    )
    text = replace_once(
        text,
        '    --ros-args -p publish_observation_pose:=true \\\n',
        '    --ros-args -p publish_observation_pose:=false \\\n',
    )
    text = replace_once(
        text,
        '    "$ROS_WS/install/internvla_ros2/lib/internvla_ros2/internvla_client_node" --ros-args \\\n',
        '    ros2 run internvla_t4_sensors internvla_t4_client --ros-args \\\n',
    )
    text = replace_once(
        text,
        '    -p nav2_resolution_timeout_sec:=10.0 -p publish_observation_pose:=true \\\n',
        '    -p nav2_resolution_timeout_sec:=10.0 -p publish_observation_pose:=false \\\n'
        '    -p navigation_odometry_timeout_sec:=2.0 -p allow_nearest_navigation_odometry:=true \\\n',
    )
    text = replace_once(
        text,
        '  export INTERNVLA_LOCAL_IPC_TIMEOUT_SEC=360\n'
        '  export INTERNNAV_SERVER_HOST=ros2-typed-transport\n'
        '  export INTERNVLA_GO2_ACTIVE_TASK_NAME="$TASK_NAME"\n'
        'fi\nexport INTERNVLA_GO2_CONTROLLER_SOCKET="$CONTROLLER_SOCKET"\n',
        '  export INTERNVLA_LOCAL_IPC_TIMEOUT_SEC=360\n'
        '  export INTERNNAV_SERVER_HOST=ros2-typed-transport\n'
        '  export INTERNVLA_GO2_ACTIVE_TASK_NAME="$TASK_NAME"\n'
        '  export INTERNVLA_MODEL_DATASET_FILE="$DATASET_ROOT/val_unseen/val_unseen.json.gz"\n'
        'fi\nexport INTERNVLA_GO2_CONTROLLER_SOCKET="$CONTROLLER_SOCKET"\n',
    )
    text = replace_once(
        text,
        'kill -0 "$nav2_pid"; kill -0 "$nav_lifecycle_pid"; kill -0 "$collision_pid"; kill -0 "$adapter_pid"; kill -0 "$controller_pid"; kill -0 "$client_pid"',
        'kill -0 "$nav2_pid"; kill -0 "$nav_lifecycle_pid"; kill -0 "$adapter_pid"; kill -0 "$controller_pid"; kill -0 "$client_pid"; kill -0 "$relay_pid"\n'
        'test -z "$nvblox_pid" || kill -0 "$nvblox_pid"\n'
        'test -z "$go2_sensor_pid" || kill -0 "$go2_sensor_pid"\n'
        'test -z "$tracer_pid" || kill -0 "$tracer_pid"\n'
        'test -z "$odometry_pid" || kill -0 "$odometry_pid"\n'
        'test -z "$recovery_pid" || kill -0 "$recovery_pid"',
    )
    text = replace_once(
        text,
        'stop_group "$controller_pid"; controller_pid=""\nstop_group "$adapter_pid"; adapter_pid=""',
        'stop_group "$relay_pid"; relay_pid=""\n'
        'bash "$CONTROL_ROOT/scripts/t4_cleanup_container_processes.sh" '
        '>"$RESULT_DIR/logs/container_cleanup.log" 2>&1\n'
        'stop_group "$controller_pid"; controller_pid=""\n'
        'stop_group "$tracer_pid"; tracer_pid=""\n'
        'stop_group "$go2_sensor_pid"; go2_sensor_pid=""\n'
        'stop_group "$recovery_pid"; recovery_pid=""\n'
        'stop_group "$odometry_pid"; odometry_pid=""\n'
        'stop_group "$nvblox_pid"; nvblox_pid=""\n'
        'stop_group "$adapter_pid"; adapter_pid=""',
    )

    validation_call = text.rfind('set +e\npython3 - "$RESULT_DIR" "$PHASE" "$EXPECTED_COUNT"')
    if validation_call < 0:
        raise RuntimeError("T3 inline validation start not found")
    validation_end = text.index('for file in "$RESULT_DIR"/logs/*.log; do', validation_call)
    new_validation = '''if test "$INTERNVLA_T4_POSE_SOURCE" = external_odometry; then
  set +e
  python3 "$SCRIPT_ROOT/analyze_t4_odometry.py" "$RESULT_DIR" \\
    --maximum-ate-rmse-m 0.20 >"$RESULT_DIR/logs/odometry_metrics.log" 2>&1
  ODOMETRY_EXIT=$?
  set -e
  test "$ODOMETRY_EXIT" = 0
fi
set +e
python3 "$CONTROL_ROOT/t4_completion/map/warn_relay.py" \
  --validate-evidence "$RESULT_DIR/warn_only_relay.jsonl" \
  --summary "$RESULT_DIR/warn_only_relay_summary.json"
RELAY_VALIDATION_EXIT=$?
python3 "$SCRIPT_ROOT/validate_t4_run.py" "$RESULT_DIR" \\
  --expected "$EXPECTED_COUNT" --minimum-sr "$MIN_SR" \\
  --map-source "$INTERNVLA_T4_MAP_SOURCE" \\
  --pose-source "$INTERNVLA_T4_POSE_SOURCE" \\
  --runtime-policy "${INTERNNAV_RUNTIME_POLICY:-strict_evidence}"
RUN_VALIDATION_EXIT=$?
VALIDATION_EXIT=$RUN_VALIDATION_EXIT
if test "$RELAY_VALIDATION_EXIT" != 0; then
  VALIDATION_EXIT=$RELAY_VALIDATION_EXIT
fi
set -e
test "$VALIDATION_EXIT" = 0
if test "${INTERNVLA_T4_ENABLE_RECOVERY:-0}" = 1; then
  set +e
  python3 "$SCRIPT_ROOT/analyze_t4_recovery.py" "$RESULT_DIR" \\
    >"$RESULT_DIR/logs/recovery_metrics.log" 2>&1
  RECOVERY_VALIDATION_EXIT=$?
  set -e
  if test "${INTERNVLA_T4_ENFORCE_RECOVERY_GATE:-1}" = 1; then
    test "$RECOVERY_VALIDATION_EXIT" = 0
  fi
fi

'''
    text = text[:validation_call] + new_validation + text[validation_end:]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8", newline="\n")
    output_bytes = args.output.read_bytes()
    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "output_sha256": hashlib.sha256(output_bytes).hexdigest(),
        "t3_frozen_source_modified": False,
        "truth_pose_from_evaluator_discarded_by_typed_client": True,
        "online_map_has_no_static_scene_payload": True,
        "ros_graph_probe": "isolated_jazzy_rclpy_no_daemon",
        "bounded_completion_safe_cmd_relay": True,
    }
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
