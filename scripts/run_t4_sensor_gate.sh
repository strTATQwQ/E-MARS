#!/usr/bin/env bash
set -euo pipefail

GATE="${1:-}"
ROLE="${2:-}"
ATTEMPT="${3:-001}"
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
case "$GATE" in t4_2|t4_3|t4_4) ;; *) echo "usage: $0 {t4_2|t4_3|t4_4} {oracle|model} [attempt]" >&2; exit 2;; esac
case "$ROLE" in oracle|model) ;; *) echo "usage: $0 {t4_2|t4_3|t4_4} {oracle|model} [attempt]" >&2; exit 2;; esac

CONTROL_ROOT="${INTERNNAV_T1_CONTROL_ROOT:-$HOME/internnav-t1-t2}"
ROS_WS="${INTERNVLA_ROS_WS:-$CONTROL_ROOT/ros_ws}"
CONTAINER="${INTERNVLA_T4_CONTAINER_NAME:-internnav_t4_isaac_ros}"
FROZEN_CAMERA="$CONTROL_ROOT/configs/internnav_t4/frozen_camera.json"
test -f "$FROZEN_CAMERA"
test -x "$ROS_WS/install/internvla_t4_sensors/lib/internvla_t4_sensors/internvla_t4_sensor_bridge"
docker inspect -f '{{.State.Running}}' "$CONTAINER" | grep -Fxq true

COMPLETION_FAST_PATH=0
if test "${INTERNVLA_T4_COMPLETION_FAST_PATH:-0}" = 1; then
  test "$GATE" = t4_4
  test "${INTERNNAV_RUNTIME_POLICY:-}" = completion_sim
  test "${INTERNNAV_SIMULATION_TARGET:-}" = isaac
  COMPLETION_FAST_PATH=1
fi

case "$GATE" in
  t4_2)
    export INTERNVLA_T4_MAP_SOURCE=nvblox_online
    export INTERNVLA_T4_POSE_SOURCE=ground_truth
    export INTERNVLA_NAV2_PARAMS="${INTERNVLA_T4_NAV2_PARAMS_OVERRIDE:-$CONTROL_ROOT/configs/internnav_t4/nav2_nvblox_gt_pose.yaml}"
    ORACLE_MIN_SR=0.9
    MODEL_MIN_SR=0.0
    ;;
  t4_3)
    export INTERNVLA_T4_MAP_SOURCE=static_map
    export INTERNVLA_T4_POSE_SOURCE=external_odometry
    export INTERNVLA_NAV2_PARAMS="$CONTROL_ROOT/configs/internnav_t4/nav2_odometry_static_map.yaml"
    ORACLE_MIN_SR=0.8
    MODEL_MIN_SR=0.0
    ;;
  t4_4)
    if test "$COMPLETION_FAST_PATH" = 1; then
      # Functional completion path: keep Nvblox and sensor odometry as strict
      # extensions.  The runnable core is static global + LiDAR local + Isaac
      # navigation odometry, exactly as allowed by completion_sim.
      export INTERNVLA_T4_MAP_SOURCE=static_map
      export INTERNVLA_T4_POSE_SOURCE=ground_truth
      export INTERNVLA_NAV2_PARAMS="$CONTROL_ROOT/configs/completion_sim/map/nav2_static_lidar.yaml"
    else
      export INTERNVLA_T4_MAP_SOURCE=nvblox_online
      export INTERNVLA_T4_POSE_SOURCE=external_odometry
      export INTERNVLA_NAV2_PARAMS="$CONTROL_ROOT/configs/internnav_t4/nav2_sensor_stack.yaml"
    fi
    ORACLE_MIN_SR=0.8
    MODEL_MIN_SR=0.2
    ;;
esac

readarray -t CAMERA_VALUES < <(python3 - "$FROZEN_CAMERA" <<'PY'
import json,sys
value=json.load(open(sys.argv[1],encoding="utf-8"))
camera=value["camera"]
print(camera["height_above_support_m"])
print(camera["pitch_down_deg"])
print(camera["hfov_deg"])
print(camera["vfov_deg"])
depth=value["depth_camera_contract"]
print(depth["hfov_deg"])
print(depth["vfov_deg"])
print(value["variant"])
PY
)
# Semantic RGB may be changed by the view ablation.  Depth and safety geometry
# are independently frozen unless an explicit non-ablation diagnostic sets a
# depth-specific override.
export INTERNVLA_T4_CAMERA_HEIGHT_M="${INTERNVLA_T4_CAMERA_HEIGHT_OVERRIDE_M:-${CAMERA_VALUES[0]}}"
export INTERNVLA_T4_CAMERA_PITCH_DOWN_DEG="${INTERNVLA_T4_CAMERA_PITCH_OVERRIDE_DEG:-${CAMERA_VALUES[1]}}"
export INTERNVLA_T4_CAMERA_HFOV_DEG="${INTERNVLA_T4_CAMERA_HFOV_OVERRIDE_DEG:-${CAMERA_VALUES[2]}}"
export INTERNVLA_T4_CAMERA_VFOV_DEG="${INTERNVLA_T4_CAMERA_VFOV_OVERRIDE_DEG:-${CAMERA_VALUES[3]}}"
export INTERNVLA_T4_CAMERA_FORWARD_M="${INTERNVLA_T4_CAMERA_FORWARD_OVERRIDE_M:-0.20}"
export INTERNVLA_T4_DEPTH_HEIGHT_M="${INTERNVLA_T4_DEPTH_HEIGHT_OVERRIDE_M:-${CAMERA_VALUES[0]}}"
export INTERNVLA_T4_DEPTH_FORWARD_M="${INTERNVLA_T4_DEPTH_FORWARD_OVERRIDE_M:-0.20}"
export INTERNVLA_T4_DEPTH_PITCH_DOWN_DEG="${INTERNVLA_T4_DEPTH_PITCH_OVERRIDE_DEG:-${CAMERA_VALUES[1]}}"
export INTERNVLA_T4_DEPTH_HFOV_DEG="${INTERNVLA_T4_DEPTH_HFOV_OVERRIDE_DEG:-${CAMERA_VALUES[4]}}"
export INTERNVLA_T4_DEPTH_VFOV_DEG="${INTERNVLA_T4_DEPTH_VFOV_OVERRIDE_DEG:-${CAMERA_VALUES[5]}}"
# The online safety/map stack consumes every simulated depth frame.  Sampling
# one frame in four made a healthy Isaac stream look stale to Collision Monitor.
export INTERNVLA_T4_DEPTH_STRIDE="${INTERNVLA_T4_DEPTH_STRIDE_OVERRIDE:-1}"
if test "${INTERNNAV_T5_STRICT_EXTENSION_PROFILE:-off}" = cuvslam_shadow; then
  test "${INTERNNAV_RUNTIME_POLICY:-}" = completion_sim
  test "${INTERNNAV_SIMULATION_TARGET:-}" = isaac
  test "${INTERNNAV_T5_LANE:-}" = a
  test "${INTERNNAV_T5_RESOURCE_LEASE_ACK:-}" = lane-a
  test "$INTERNVLA_T4_POSE_SOURCE" = ground_truth
  export INTERNVLA_T4_ENABLE_STEREO_ODOMETRY=1
elif test "$INTERNVLA_T4_POSE_SOURCE" = external_odometry; then
  export INTERNVLA_T4_ENABLE_STEREO_ODOMETRY=1
else
  export INTERNVLA_T4_ENABLE_STEREO_ODOMETRY=0
fi

if test "$ROLE" = oracle; then
  PHASE="${INTERNVLA_T4_PHASE_OVERRIDE:-obstacle_oracle}"
  case "$PHASE" in continuous_oracle|obstacle_oracle) ;; *) exit 2 ;; esac
  export INTERNVLA_T4_MIN_SR="${INTERNVLA_T4_MIN_SR_OVERRIDE:-$ORACLE_MIN_SR}"
  if test "$GATE" = t4_2 || test "$GATE" = t4_4; then
    # Reuse the immutable T3 obstacle-Oracle split that passed the T3 gate.
    # Both variables are frozen explicitly because the inherited phase runner
    # distinguishes the materialized obstacle set from its source split.
    export INTERNVLA_T3_OBSTACLE_ORACLE_DATASET_ROOT="${INTERNVLA_T3_OBSTACLE_ORACLE_DATASET_ROOT:-$CONTROL_ROOT/episodes/t3_obstacle_oracle_final_v2}"
    export INTERNVLA_ORACLE_DATASET_ROOT="${INTERNVLA_T4_ORACLE_DATASET_ROOT:-$CONTROL_ROOT/episodes/t3_obstacle_oracle_final_v2}"
  else
    export INTERNVLA_ORACLE_DATASET_ROOT="${INTERNVLA_T4_ORACLE_DATASET_ROOT:-$CONTROL_ROOT/episodes/t3_continuous_oracle_go2_clear_v2}"
  fi
else
  PHASE="${INTERNVLA_T4_PHASE_OVERRIDE:-pilot}"
  case "$PHASE" in
    pilot) ;;
    canary)
      # T4 remains pilot-only.  The five-episode phase is an additive T5
      # completion_sim path and is authorized only by a matching physical
      # Lane lease plus the dataset/profile contract established by the T5
      # Isaac entrypoint.
      test "${INTERNNAV_RUNTIME_POLICY:-}" = completion_sim
      test "${INTERNNAV_SIMULATION_TARGET:-}" = isaac
      # T5 candidate screening may use the deterministic first 1 or 3 rows of
      # the frozen five.  T4 behavior remains unchanged because every path is
      # still guarded by the T5 completion_sim Lane lease below.
      case "${INTERNVLA_T5_DATASET_EPISODE_COUNT:-}" in
        1|3|5) ;;
        *) exit 2 ;;
      esac
      case "${INTERNVLA_T5_EXPECTED_LEASE_ACK:-}:${INTERNNAV_T5_RESOURCE_LEASE_ACK:-}" in
        lane-a:lane-a|lane-b:lane-b) ;;
        *) exit 2 ;;
      esac
      case "${INTERNNAV_T5_EXECUTION_PROFILE:-}" in
        fixed_dataset) ;;
        engineering_canary)
          test "${INTERNNAV_T5_ENGINEERING_CANARY_ACK:-}" = fast-path
          ;;
        *) exit 2 ;;
      esac
      ;;
    *) exit 2 ;;
  esac
  export INTERNVLA_T4_MIN_SR="${INTERNVLA_T4_MIN_SR_OVERRIDE:-$MODEL_MIN_SR}"
  export INTERNVLA_GO2_DATASET_ROOT="${INTERNVLA_T4_MODEL_DATASET_ROOT:-$HOME/internnav-t0/episodes/pilot}"
fi

RUN_LABEL="${INTERNVLA_T4_RUN_LABEL:-$GATE}"
RESULT_ROOT="${INTERNVLA_T4_RESULT_ROOT:-$CONTROL_ROOT/results/internnav_t4}"
RESULT_DIR="$RESULT_ROOT/${RUN_LABEL}_${ROLE}_attempt_${ATTEMPT}"
test ! -e "$RESULT_DIR"
mkdir -p "$RESULT_DIR"
if test "${INTERNVLA_T4_ENABLE_RECOVERY:-0}" = 1; then
  test "${INTERNNAV_RUNTIME_POLICY:-}" = completion_sim
  test "${INTERNNAV_SIMULATION_TARGET:-}" = isaac
  RECOVERY_PROFILE="${INTERNVLA_T4_RECOVERY_PROFILE:-$CONTROL_ROOT/configs/completion_sim/recovery/profile_b.json}"
  test -f "$RECOVERY_PROFILE"
  RECOVERY_NAV2_PARAMS="$RESULT_DIR/nav2_recovery_params.yaml"
  RECOVERY_RUNTIME_MANIFEST="$RESULT_DIR/recovery_runtime_manifest.json"
  mapfile -t RECOVERY_FIELDS < <(
    python3 "$ROOT/t4_recovery_runtime.py" \
      --profile "$RECOVERY_PROFILE" \
      --nav2-input "$INTERNVLA_NAV2_PARAMS" \
      --nav2-output "$RECOVERY_NAV2_PARAMS" \
      --manifest "$RECOVERY_RUNTIME_MANIFEST" \
      --format fields
  )
  test "${#RECOVERY_FIELDS[@]}" -eq 12
  export INTERNVLA_T4_RECOVERY_PROFILE_ID="${RECOVERY_FIELDS[0]}"
  export INTERNVLA_T4_RECOVERY_PROFILE_SHA256="${RECOVERY_FIELDS[1]}"
  export INTERNVLA_T4_RECOVERY_NAV2_SHA256="${RECOVERY_FIELDS[2]}"
  export INTERNVLA_T4_PROGRESS_HORIZON_SEC="${RECOVERY_FIELDS[3]}"
  export INTERNVLA_T4_MINIMUM_PROGRESS_M="${RECOVERY_FIELDS[4]}"
  export INTERNVLA_T4_OSCILLATION_TRAVEL_M="${RECOVERY_FIELDS[5]}"
  export INTERNVLA_T4_RECOVERY_COOLDOWN_SEC="${RECOVERY_FIELDS[6]}"
  export INTERNVLA_T4_MAXIMUM_RECOVERIES="${RECOVERY_FIELDS[7]}"
  export INTERNVLA_T4_RECOVERY_SCAN_YAW_RAD="${RECOVERY_FIELDS[8]}"
  export INTERNVLA_T4_RECOVERY_SCAN_SPEED_RPS="${RECOVERY_FIELDS[9]}"
  export INTERNVLA_T4_RECOVERY_SAFETY_FRESHNESS_SEC="${RECOVERY_FIELDS[10]}"
  export INTERNVLA_T4_MAXIMUM_RECOVERY_DURATION_SEC="${RECOVERY_FIELDS[11]}"
  export INTERNVLA_T4_RECOVERY_PROFILE="$RECOVERY_PROFILE"
  export INTERNVLA_NAV2_PARAMS="$RECOVERY_NAV2_PARAMS"
  docker exec --user admin --workdir /workspaces/isaac "$CONTAINER" bash -lc '
    set -euo pipefail
    source /opt/ros/jazzy/setup.bash
    source /workspaces/isaac/install/setup.bash
    ros2 interface show internvla_ros2_msgs/srv/RecoveryControl >/dev/null
    python3 -c "from internvla_ros2_msgs.srv import RecoveryControl"
  '
fi
IPC_DIR="$CONTROL_ROOT/runtime/t4_ipc"
# The deployment root is deliberately verbose and can exceed Linux's
# 108-byte sockaddr_un.sun_path limit even after the filename is hashed.  Both
# the Isaac host and the long-lived ROS container therefore expose the same
# shared deployment directory through this short alias.  The socket inode
# remains under CONTROL_ROOT, so the existing residual scan still owns it.
readonly IPC_ALIAS="${INTERNVLA_T4_IPC_ALIAS_OVERRIDE:-/tmp/internnav_t4_ipc}"
case "$IPC_ALIAS" in
  /tmp/internnav_t4_ipc|/tmp/internnav_t5_a_ipc|/tmp/internnav_t5_b_ipc) ;;
  *) echo "unsafe IPC alias: $IPC_ALIAS" >&2; exit 2 ;;
esac
readonly IPC_PATH_MAX_BYTES=100
IPC_ALLOWED_ROOT="${INTERNNAV_T4_IPC_ALLOWED_ROOT:-$HOME/internnav-t1-t2}"
mkdir -p "$IPC_DIR"
if test -e "$IPC_ALIAS" || test -L "$IPC_ALIAS"; then
  test -L "$IPC_ALIAS"
  current_ipc_target="$(readlink "$IPC_ALIAS")"
  if test "$current_ipc_target" != "$IPC_DIR"; then
    case "$current_ipc_target" in
      "$IPC_ALLOWED_ROOT/runtime/t4_ipc"|"$IPC_ALLOWED_ROOT"/.t4-deployments/*/runtime/t4_ipc) ;;
      *) exit 1 ;;
    esac
    if test -d "$current_ipc_target"; then
      test -z "$(find "$current_ipc_target" -maxdepth 1 -type s -print -quit)"
    fi
    ln -sfn "$IPC_DIR" "$IPC_ALIAS"
  fi
else
  ln -s "$IPC_DIR" "$IPC_ALIAS"
fi
docker exec --user admin --workdir /workspaces/isaac \
  -e "IPC_DIR=$IPC_DIR" -e "IPC_ALIAS=$IPC_ALIAS" \
  -e "IPC_ALLOWED_ROOT=$IPC_ALLOWED_ROOT" "$CONTAINER" bash -lc '
    set -euo pipefail
    test -d "$IPC_DIR"
    if test -e "$IPC_ALIAS" || test -L "$IPC_ALIAS"; then
      test -L "$IPC_ALIAS"
      current_ipc_target="$(readlink "$IPC_ALIAS")"
      if test "$current_ipc_target" != "$IPC_DIR"; then
        case "$current_ipc_target" in
          "$IPC_ALLOWED_ROOT/runtime/t4_ipc"|"$IPC_ALLOWED_ROOT"/.t4-deployments/*/runtime/t4_ipc) ;;
          *) exit 1 ;;
        esac
        if test -d "$current_ipc_target"; then
          test -z "$(find "$current_ipc_target" -maxdepth 1 -type s -print -quit)"
        fi
        ln -sfn "$IPC_DIR" "$IPC_ALIAS"
      fi
    else
      ln -s "$IPC_DIR" "$IPC_ALIAS"
    fi
  '
IPC_TOKEN="$(printf '%s' "${RUN_LABEL}_${ROLE}_${ATTEMPT}" | sha256sum | cut -c1-16)"
export INTERNVLA_CLIENT_SOCKET="$IPC_ALIAS/${IPC_TOKEN}_agent.sock"
export INTERNVLA_ORACLE_SOCKET="$IPC_ALIAS/${IPC_TOKEN}_oracle.sock"
export INTERNVLA_GO2_CONTROLLER_SOCKET="$IPC_ALIAS/${IPC_TOKEN}_controller.sock"
export INTERNVLA_T4_IPC_ALIAS="$IPC_ALIAS"
export INTERNVLA_T4_IPC_TOKEN="$IPC_TOKEN"
for socket_path in \
  "$INTERNVLA_CLIENT_SOCKET" \
  "$INTERNVLA_ORACLE_SOCKET" \
  "$INTERNVLA_GO2_CONTROLLER_SOCKET"; do
  test "$(printf %s "$socket_path" | wc -c)" -le "$IPC_PATH_MAX_BYTES"
done
python3 - "$RESULT_DIR/t4_run_contract.json" <<'PY'
import json,os,sys
from pathlib import Path
payload={
    "schema_version":1,
    "runtime_policy":os.environ.get("INTERNNAV_RUNTIME_POLICY","strict_evidence"),
    "runtime_target":os.environ.get("INTERNNAV_SIMULATION_TARGET","unspecified"),
    "map_source":os.environ["INTERNVLA_T4_MAP_SOURCE"],
    "pose_source":os.environ["INTERNVLA_T4_POSE_SOURCE"],
    "camera":{
        "height_above_support_m":float(os.environ["INTERNVLA_T4_CAMERA_HEIGHT_M"]),
        "pitch_down_deg":float(os.environ["INTERNVLA_T4_CAMERA_PITCH_DOWN_DEG"]),
        "hfov_deg":float(os.environ["INTERNVLA_T4_CAMERA_HFOV_DEG"]),
        "vfov_deg":float(os.environ["INTERNVLA_T4_CAMERA_VFOV_DEG"]),
    },
    "depth_camera":{
        "camera_model":"Intel RealSense D435i depth imager",
        "height_above_support_m":float(os.environ["INTERNVLA_T4_DEPTH_HEIGHT_M"]),
        "forward_m":float(os.environ["INTERNVLA_T4_DEPTH_FORWARD_M"]),
        "pitch_down_deg":float(os.environ["INTERNVLA_T4_DEPTH_PITCH_DOWN_DEG"]),
        "hfov_deg":float(os.environ["INTERNVLA_T4_DEPTH_HFOV_DEG"]),
        "vfov_deg":float(os.environ["INTERNVLA_T4_DEPTH_VFOV_DEG"]),
        "minimum_depth_m":0.28,
    },
    "ablation_variant_id":os.environ.get("INTERNVLA_T4_VARIANT_ID","none"),
    "ablation_config_sha256":os.environ.get("INTERNVLA_T4_VARIANT_CONFIG_SHA256"),
    "ablation_matrix_sha256":os.environ.get("INTERNVLA_T4_MATRIX_SHA256"),
    "ablation_factors":{
        "system_mode":os.environ.get("INTERNVLA_T4_SYSTEM_MODE","full_system1_system2"),
        "trajectory_mode":os.environ.get("INTERNVLA_T4_TRAJECTORY_MODE","full_trajectory"),
        "termination_mode":os.environ.get("INTERNVLA_T4_TERMINATION_MODE","model_stop"),
        "history_mode":os.environ.get("INTERNVLA_T4_HISTORY_MODE","on"),
        "recovery_mode":os.environ.get("INTERNVLA_T4_RECOVERY_MODE","off"),
        "view_mode":os.environ.get("INTERNVLA_T4_VIEW_MODE","go2_view"),
    },
    "semantic_rgb_view_scope":"internvla_rgb_only",
    "mapping_and_safety_geometry":"go2_frozen",
    "recovery_enabled":os.environ.get("INTERNVLA_T4_ENABLE_RECOVERY","0")=="1",
    "recovery_profile":{
        "profile_id":os.environ.get("INTERNVLA_T4_RECOVERY_PROFILE_ID"),
        "profile_sha256":os.environ.get("INTERNVLA_T4_RECOVERY_PROFILE_SHA256"),
        "nav2_overlay_sha256":os.environ.get("INTERNVLA_T4_RECOVERY_NAV2_SHA256"),
        "short_backup_enabled":False,
    },
    "required_model_history_mode":os.environ.get("INTERNVLA_T4_REQUIRED_HISTORY_MODE","on"),
    "evaluator_ground_truth_pose_policy":"discard_in_typed_client",
    "ipc":{
        "shared_directory_owned_by_deployment":True,
        "short_alias":os.environ["INTERNVLA_T4_IPC_ALIAS"],
        "token":os.environ["INTERNVLA_T4_IPC_TOKEN"],
        "socket_paths":{
            "agent":os.environ["INTERNVLA_CLIENT_SOCKET"],
            "oracle":os.environ["INTERNVLA_ORACLE_SOCKET"],
            "controller":os.environ["INTERNVLA_GO2_CONTROLLER_SOCKET"],
        },
        "maximum_path_bytes":100,
    },
    "controller_transport":{
        "kind":"tcp" if os.environ.get("INTERNVLA_GO2_CONTROLLER_ENDPOINT") else "unix",
        "endpoint":os.environ.get("INTERNVLA_GO2_CONTROLLER_ENDPOINT"),
        "dgx_onboard":bool(os.environ.get("INTERNVLA_GO2_CONTROLLER_ENDPOINT")),
    },
}
Path(sys.argv[1]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
OVERLAY="$CONTROL_ROOT/runtime/${RUN_LABEL}_${ROLE}_${ATTEMPT}"
mkdir -p "$OVERLAY"
for source in "$CONTROL_ROOT"/scripts/*.py "$CONTROL_ROOT"/scripts/*.sh; do
  test -e "$source" || continue
  ln -sfn "$source" "$OVERLAY/$(basename "$source")"
done
# Python resolves an executed symlink's import root back to the source scripts
# directory.  Keep the entrypoint as an attempt-local regular file so its
# sibling generated internnav_go2_runtime.py is the module that is imported.
unlink "$OVERLAY/run_internnav_go2_entrypoint.py"
cp "$CONTROL_ROOT/scripts/run_internnav_go2_entrypoint.py" \
  "$OVERLAY/run_internnav_go2_entrypoint.py"
ln -sfn "${INTERNVLA_T4_GO2_USD_BUILDER:-$CONTROL_ROOT/scripts/build_t4_sensor_go2_usd.py}" \
  "$OVERLAY/build_go2_internvla_usd.py"
unlink "$OVERLAY/internnav_go2_runtime.py"
python3 "${INTERNVLA_T4_RUNTIME_OVERLAY_BUILDER:-$CONTROL_ROOT/scripts/build_t4_sensor_runtime_overlay.py}" \
  --source "$CONTROL_ROOT/scripts/internnav_go2_runtime.py" \
  --output "$OVERLAY/internnav_go2_runtime.py" \
  --manifest "$RESULT_DIR/runtime_overlay_manifest.json"
unlink "$OVERLAY/run_go2_continuous_phase.sh"
python3 "${INTERNVLA_T4_PHASE_OVERLAY_BUILDER:-$CONTROL_ROOT/scripts/build_t4_sensor_phase_overlay.py}" \
  --source "$CONTROL_ROOT/scripts/run_go2_continuous_phase.sh" \
  --output "$OVERLAY/run_go2_continuous_phase.sh" \
  --manifest "$RESULT_DIR/phase_overlay_manifest.json"

SHIM_DIR="$CONTROL_ROOT/runtime/t4_ros2_shim"
mkdir -p "$SHIM_DIR"
unlink "$SHIM_DIR/ros2" 2>/dev/null || true
install -m 700 "$CONTROL_ROOT/scripts/t4_ros2_container_shim.sh" "$SHIM_DIR/ros2"
export PATH="$SHIM_DIR:$PATH"
export INTERNVLA_T4_CONTAINER_NAME="$CONTAINER"
export INTERNVLA_SCRIPT_ROOT="$OVERLAY"
export INTERNVLA_T3_RESULT_DIR="$RESULT_DIR"
export INTERNVLA_T3_TASK_NAME="${RUN_LABEL}_${ROLE}_${ATTEMPT}"
export INTERNVLA_T3_STATIC_CLEARANCE_GATE_M=0.30
export INTERNVLA_T3_MAX_STEP="${INTERNVLA_T4_MAX_STEP:-16000}"
export ROS2CLI_NO_DAEMON=1

bash "$OVERLAY/run_go2_continuous_phase.sh" "$PHASE"
