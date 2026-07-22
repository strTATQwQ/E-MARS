#!/usr/bin/env bash
set -eo pipefail

CONTROL_ROOT="${INTERNNAV_T1_CONTROL_ROOT:-$HOME/internnav-t1-t2}"
T0_CONTROL_ROOT="${INTERNNAV_T0_CONTROL_ROOT:-$HOME/internnav-t0/control}"
ROS_WS="${INTERNVLA_ROS_WS:-$CONTROL_ROOT/ros_ws}"
INTERNNAV_ROOT="${INTERNNAV_ROOT:-$HOME/internnav-t0/InternNav}"
ISAAC_PYTHON="${INTERNNAV_ISAAC_PYTHON:-$HOME/env_isaacsim/bin/python}"
COMPAT_OVERLAY="${INTERNNAV_COMPAT_OVERLAY:-$HOME/internnav-t0/runtime/internutopia_2_2_0}"
DATASET_ROOT="${INTERNVLA_GO2_DATASET_ROOT:-$HOME/internnav-t0/episodes/pilot}"
RESULT_DIR="${INTERNVLA_T3_FLASH_RESULT_DIR:-$CONTROL_ROOT/results/internnav_t3/t3_0_go2_flash_nav2_sampled_baseline}"
PARAMS="${INTERNVLA_NAV2_PARAMS:-$CONTROL_ROOT/configs/internnav_t1_t2/nav2_active_rolling.yaml}"
CONFIG="${INTERNVLA_GO2_CONFIG:-$CONTROL_ROOT/configs/internnav_t1_t2/go2_nav2_active_cfg.py}"
SCRIPT_ROOT="${INTERNVLA_SCRIPT_ROOT:-$CONTROL_ROOT/scripts}"
SOCKET_PATH="${INTERNVLA_CLIENT_SOCKET:-/tmp/internvla_t3_flash_baseline.sock}"
TASK_NAME="${INTERNVLA_T3_FLASH_TASK_NAME:-internnav_t3_go2_flash_nav2_sampled_baseline}"
GO2_SOURCE="${INTERNVLA_GO2_SOURCE_USD:-$HOME/isaacsim_assets/Assets/Isaac/6.0/Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd}"
GO2_RUNTIME="$CONTROL_ROOT/runtime/go2_t3_flash"
GO2_WRAPPER="$GO2_RUNTIME/go2_internvla_t3_flash.usda"
GO2_MANIFEST="$RESULT_DIR/go2_asset_manifest.json"
FROZEN_T2_RESULT="$CONTROL_ROOT/results/t2_go2_nav2_pilot"

test -x "$ISAAC_PYTHON"; test -f "$ROS_WS/install/setup.bash"
test -f "$DATASET_ROOT/val_unseen/val_unseen.json.gz"; test -f "$GO2_SOURCE"
test -f "$PARAMS"; test -f "$CONFIG"; test ! -e "$RESULT_DIR/phase_status.json"
test -f "$SCRIPT_ROOT/extract_internnav_per_episode.py"
test -f "$SCRIPT_ROOT/summarize_internnav_progress.py"
test -f "$SCRIPT_ROOT/sanitize_internnav_log.py"
test -f "$SCRIPT_ROOT/postprocess_t3_flash.py"
PYTHONPATH="$SCRIPT_ROOT" python3 -c \
  'from summarize_internnav_progress import summarize; from extract_internnav_per_episode import main; import postprocess_t3_flash'
for frozen_file in \
  result.json validation.json per_episode.json active_records.jsonl active_summary.json; do
  test -f "$FROZEN_T2_RESULT/$frozen_file"
done
mkdir -p "$RESULT_DIR/logs" "$RESULT_DIR/logs_sanitized" "$GO2_RUNTIME"
rm -f "$SOCKET_PATH" "$GO2_WRAPPER" "$GO2_MANIFEST"

STARTED_UNIX="$(date +%s)"; EVAL_EXIT=125; PREFLIGHT_EXIT=125; VALIDATION_EXIT=125; FINAL_STATUS=FAIL
nav2_pid=""; adapter_pid=""; client_pid=""
stop_group() {
  local pid="$1"
  if test -n "$pid" && kill -0 "$pid" 2>/dev/null; then
    kill -INT -- "-$pid" 2>/dev/null || true
    for _ in $(seq 1 50); do kill -0 "$pid" 2>/dev/null || return 0; sleep 0.1; done
    kill -TERM -- "-$pid" 2>/dev/null || true
  fi
}
write_status() {
  local ended="$(date +%s)"
  python3 - "$RESULT_DIR/phase_status.json" "$STARTED_UNIX" "$ended" "$PREFLIGHT_EXIT" "$EVAL_EXIT" "$VALIDATION_EXIT" "$FINAL_STATUS" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); started=int(sys.argv[2]); ended=int(sys.argv[3])
value={"schema_version":1,"phase":"go2_flash_nav2_sampled_baseline","started_unix":started,"ended_unix":ended,"duration_sec":ended-started,"exit_codes":{"preflight":int(sys.argv[4]),"evaluator":int(sys.argv[5]),"validation":int(sys.argv[6])},"status":sys.argv[7]}
p.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
PY
}
cleanup() {
  stop_group "$client_pid"; stop_group "$adapter_pid"; stop_group "$nav2_pid"
  rm -f "$SOCKET_PATH"; write_status
}
trap cleanup EXIT INT TERM

"$ISAAC_PYTHON" "$SCRIPT_ROOT/build_go2_internvla_usd.py" \
  --source "$GO2_SOURCE" --output "$GO2_WRAPPER" --manifest "$GO2_MANIFEST"

export INTERNNAV_T1_CONTROL_ROOT="$CONTROL_ROOT"
export INTERNNAV_T0_CONTROL_ROOT="$T0_CONTROL_ROOT"
export INTERNNAV_ROOT
export INTERNNAV_T0_PHASE=pilot
export INTERNNAV_T0_DATASET_ROOT="$DATASET_ROOT"
export INTERNNAV_T0_RESULT_DIR="$RESULT_DIR"
export INTERNVLA_GO2_WRAPPER_USD="$GO2_WRAPPER"
export INTERNVLA_GO2_EXECUTION_MODE=flash
export INTERNVLA_GO2_RUNTIME_AUDIT="$RESULT_DIR/go2_runtime_audit.jsonl"
export INTERNVLA_GO2_ACTIVE_TASK_NAME="$TASK_NAME"
export INTERNNAV_SERVER_HOST=ros2-typed-transport
ISAAC_PYTHONPATH="$COMPAT_OVERLAY:$INTERNNAV_ROOT:$SCRIPT_ROOT"
set +e
PYTHONPATH="$ISAAC_PYTHONPATH" "$ISAAC_PYTHON" \
  "$SCRIPT_ROOT/preflight_go2_config.py" --config "$CONFIG" \
  >"$RESULT_DIR/preflight.json" 2>"$RESULT_DIR/logs/preflight.log"
PREFLIGHT_EXIT=$?
set -e
test "$PREFLIGHT_EXIT" = 0

source /opt/ros/humble/setup.bash
source "$ROS_WS/install/setup.bash"
set -u
setsid ros2 launch nav2_bringup navigation_launch.py \
  params_file:="$PARAMS" use_sim_time:=False autostart:=True \
  use_composition:=False use_respawn:=False >"$RESULT_DIR/logs/nav2.log" 2>&1 &
nav2_pid=$!
setsid "$ROS_WS/install/internvla_nav2_adapter/lib/internvla_nav2_adapter/internvla_nav2_active" \
  --ros-args -p result_dir:="$RESULT_DIR" -p command_mode:=follow_path \
  -p execution_mode:=sampled_flash >"$RESULT_DIR/logs/adapter.log" 2>&1 &
adapter_pid=$!
INTERNVLA_CLIENT_SOCKET="$SOCKET_PATH" setsid \
  "$ROS_WS/install/internvla_ros2/lib/internvla_ros2/internvla_client_node" --ros-args \
  -p service_timeout_sec:=300.0 -p step_deadline_sec:=30.0 -p step_validity_sec:=35.0 \
  -p ipc_idle_timeout_sec:=900.0 -p control_mode:=nav2 \
  -p nav2_resolution_timeout_sec:=10.0 >"$RESULT_DIR/logs/client.log" 2>&1 &
client_pid=$!
for _ in $(seq 1 600); do test -S "$SOCKET_PATH" && break; sleep 0.1; done
test -S "$SOCKET_PATH"; kill -0 "$nav2_pid"; kill -0 "$adapter_pid"; kill -0 "$client_pid"
{
  echo "# nodes"; ros2 node list | sort
  echo "# topics"; ros2 topic list -t | sort
  echo "# services"; ros2 service list -t | sort
  echo "# actions"; ros2 action list -t | sort
} >"$RESULT_DIR/ros_graph_snapshot.txt"

cd "$INTERNNAV_ROOT"
export INTERNVLA_CLIENT_SOCKET="$SOCKET_PATH"
export INTERNVLA_LOCAL_IPC_TIMEOUT_SEC=360
export INTERNNAV_SERVER_HOST=ros2-typed-transport
export INTERNVLA_GO2_CLIENT_MODE=model
set +e
env OMNI_KIT_ACCEPT_EULA=Y MESA_GL_VERSION_OVERRIDE=4.6 \
  PYTHONPATH="$ISAAC_PYTHONPATH" \
  "$ISAAC_PYTHON" "$SCRIPT_ROOT/run_internnav_go2_entrypoint.py" --config "$CONFIG" \
  2>&1 | tee "$RESULT_DIR/logs/eval.log"
EVAL_EXIT=${PIPESTATUS[0]}
set -e
test "$EVAL_EXIT" = 0
test -f "$INTERNNAV_ROOT/logs/$TASK_NAME/result.json"
cp "$INTERNNAV_ROOT/logs/$TASK_NAME/result.json" "$RESULT_DIR/result.json"

stop_group "$adapter_pid"; adapter_pid=""
python3 "$SCRIPT_ROOT/extract_internnav_per_episode.py" \
  "$INTERNNAV_ROOT/logs/$TASK_NAME" "$RESULT_DIR/per_episode.json" --expected 20 \
  >"$RESULT_DIR/logs/per_episode_extract.log"

set +e
python3 "$SCRIPT_ROOT/postprocess_t3_flash.py" "$CONTROL_ROOT" "$RESULT_DIR" --expected 20
VALIDATION_EXIT=$?
set -e
test "$VALIDATION_EXIT" = 0

for file in "$RESULT_DIR"/logs/*.log; do
  test -f "$file" || continue
  python3 "$SCRIPT_ROOT/sanitize_internnav_log.py" "$file" \
    "$RESULT_DIR/logs_sanitized/$(basename "$file")"
done
FINAL_STATUS=PASS
