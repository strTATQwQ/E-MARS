#!/usr/bin/env bash
set -eo pipefail

PHASE="${1:-}"
case "$PHASE" in
  diagnostics)
    EXPECTED_COUNT=5; MIN_SR=1.0; SOURCE_PHASE=oracle; OBSTACLE_AWARE=0; CLIENT_KIND=oracle ;;
  continuous_oracle)
    EXPECTED_COUNT=10; MIN_SR=0.9; SOURCE_PHASE=oracle; OBSTACLE_AWARE=0; CLIENT_KIND=oracle ;;
  obstacle_oracle)
    EXPECTED_COUNT=10; MIN_SR=0.9; SOURCE_PHASE=oracle; OBSTACLE_AWARE=1; CLIENT_KIND=oracle ;;
  canary_no_obstacle)
    EXPECTED_COUNT=5; MIN_SR=0.4; SOURCE_PHASE=canary; OBSTACLE_AWARE=0; CLIENT_KIND=model ;;
  pilot_no_obstacle)
    EXPECTED_COUNT=20; MIN_SR=0.3; SOURCE_PHASE=pilot; OBSTACLE_AWARE=0; CLIENT_KIND=model ;;
  canary)
    EXPECTED_COUNT=5; MIN_SR=0.4; SOURCE_PHASE=canary; OBSTACLE_AWARE=1; CLIENT_KIND=model ;;
  pilot)
    EXPECTED_COUNT=20; MIN_SR=0.3; SOURCE_PHASE=pilot; OBSTACLE_AWARE=1; CLIENT_KIND=model ;;
  stress)
    EXPECTED_COUNT=10; MIN_SR=0.01; SOURCE_PHASE=stress; OBSTACLE_AWARE=1; CLIENT_KIND=model ;;
  *) echo "usage: $0 {diagnostics|continuous_oracle|obstacle_oracle|canary_no_obstacle|pilot_no_obstacle|canary|pilot|stress}" >&2; exit 2 ;;
esac

ORACLE_SUCCESS_STOP_DISTANCE=2.5
if test "$PHASE" = diagnostics; then
  ORACLE_SUCCESS_STOP_DISTANCE=0.5
fi

CONTROL_ROOT="${INTERNNAV_T1_CONTROL_ROOT:-$HOME/internnav-t1-t2}"
T0_CONTROL_ROOT="${INTERNNAV_T0_CONTROL_ROOT:-$HOME/internnav-t0/control}"
ROS_WS="${INTERNVLA_ROS_WS:-$CONTROL_ROOT/ros_ws}"
INTERNNAV_ROOT="${INTERNNAV_ROOT:-$HOME/internnav-t0/InternNav}"
ISAAC_PYTHON="${INTERNNAV_ISAAC_PYTHON:-$HOME/env_isaacsim/bin/python}"
COMPAT_OVERLAY="${INTERNNAV_COMPAT_OVERLAY:-$HOME/internnav-t0/runtime/internutopia_2_2_0}"
SCRIPT_ROOT="${INTERNVLA_SCRIPT_ROOT:-$CONTROL_ROOT/scripts}"
RESULT_DIR="${INTERNVLA_T3_RESULT_DIR:-$CONTROL_ROOT/results/internnav_t3/t3_${PHASE}}"
# InternNav refuses to overwrite an existing evaluator task directory.  Bind
# the default task name to the immutable result-directory basename so retries
# cannot silently reuse or terminate on an earlier task's artifacts.
TASK_NAME="${INTERNVLA_T3_TASK_NAME:-$(basename "$RESULT_DIR")}"
GO2_SOURCE="${INTERNVLA_GO2_SOURCE_USD:-$HOME/isaacsim_assets/Assets/Isaac/6.0/Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd}"
GO2_RUNTIME="$CONTROL_ROOT/runtime/go2_t3"
GO2_WRAPPER="$GO2_RUNTIME/go2_internvla_t3.usda"
GO2_MANIFEST="$RESULT_DIR/go2_asset_manifest.json"
AGENT_SOCKET="${INTERNVLA_CLIENT_SOCKET:-/tmp/internvla_t3_${PHASE}_agent.sock}"
ORACLE_SOCKET="${INTERNVLA_ORACLE_SOCKET:-/tmp/internvla_t3_${PHASE}_oracle.sock}"
CONTROLLER_SOCKET="${INTERNVLA_GO2_CONTROLLER_SOCKET:-/tmp/internvla_t3_${PHASE}_controller.sock}"
if test "$OBSTACLE_AWARE" = 1; then
  PARAMS="${INTERNVLA_NAV2_PARAMS:-$CONTROL_ROOT/configs/internnav_t3/nav2_obstacle.yaml}"
  ADAPTER_COMMAND_MODE=navigate_to_pose
  # A raw 2 m model endpoint can lie behind mapped MP3D geometry even when the
  # earlier samples of the same System 1 trajectory are collision-free.  Keep
  # NavigateToPose as a rolling planner interface and advance to the furthest
  # 1 m sample; later observations extend the same metric trajectory.  This
  # avoids repeatedly aborting the BT on an occupied endpoint while preserving
  # the costmap/planner/Collision Monitor safety path.
  ADAPTER_GOAL_MAX_DISTANCE="${INTERNVLA_T3_NAVIGATE_GOAL_MAX_DISTANCE_M:-1.0}"
  LIFECYCLE_NODE_NAMES="['controller_server','smoother_server','planner_server','behavior_server','bt_navigator']"
  NAV2_LIFECYCLE_NODES="/controller_server /smoother_server /planner_server /behavior_server /bt_navigator"
else
  PARAMS="${INTERNVLA_NAV2_PARAMS:-$CONTROL_ROOT/configs/internnav_t3/nav2_continuous.yaml}"
  ADAPTER_COMMAND_MODE=follow_path
  ADAPTER_GOAL_MAX_DISTANCE="${INTERNVLA_T3_FOLLOW_PATH_MAX_DISTANCE_M:-2.0}"
  LIFECYCLE_NODE_NAMES="['controller_server']"
  NAV2_LIFECYCLE_NODES="/controller_server"
fi

# The formal stress challenge uses a conservative forward stop envelope so a
# newly appearing obstacle is handled by Collision Monitor before the local
# planner can reduce it to only a far-field replan.  Other phases retain the
# audited 0.38 m envelope from nav2_obstacle.yaml.
COLLISION_MONITOR_STOP_FORWARD_M=0.38
if test "$PHASE" = stress; then
  COLLISION_MONITOR_STOP_FORWARD_M=0.50
fi

if test "$CLIENT_KIND" = oracle; then
  if test "$PHASE" = diagnostics; then
    DATASET_ROOT="${INTERNVLA_T3_DIAGNOSTIC_DATASET_ROOT:-$CONTROL_ROOT/episodes/t3_continuous_diagnostics}"
    SOURCE_DATASET="${INTERNVLA_ORACLE_DATASET_ROOT:-$CONTROL_ROOT/episodes/t3_continuous_oracle_go2_clear_v2}/val_unseen/val_unseen.json.gz"
    python3 "$SCRIPT_ROOT/build_t3_continuous_diagnostics_dataset.py" \
      --source "$SOURCE_DATASET" --output-root "$DATASET_ROOT" \
      --manifest "$DATASET_ROOT/diagnostic_manifest.json"
  elif test "$PHASE" = obstacle_oracle; then
    DATASET_ROOT="${INTERNVLA_T3_OBSTACLE_ORACLE_DATASET_ROOT:-$CONTROL_ROOT/episodes/t3_obstacle_oracle}"
    SOURCE_DATASET="${INTERNVLA_ORACLE_DATASET_ROOT:-$CONTROL_ROOT/episodes/t3_continuous_oracle_go2_clear_v2}/val_unseen/val_unseen.json.gz"
    SCENARIO_MANIFEST="$DATASET_ROOT/obstacle_manifest.json"
    if test ! -f "$DATASET_ROOT/val_unseen/val_unseen.json.gz" || test ! -f "$SCENARIO_MANIFEST"; then
      SCENARIO_PLAN="${INTERNVLA_T3_OBSTACLE_SCENARIO_PLAN:-static_box,sudden_blockage,narrow_corridor,sudden_blockage,dynamic_crossing,static_box,dynamic_crossing,narrow_corridor,doorway,doorway}"
      python3 "$SCRIPT_ROOT/build_t3_obstacle_dataset.py" \
        --source "$SOURCE_DATASET" --output-root "$DATASET_ROOT" \
        --manifest "$SCENARIO_MANIFEST" --scenario-plan "$SCENARIO_PLAN" \
        --exclude-manifest "$CONTROL_ROOT/configs/internnav_t3/obstacle_route_exclusions.json"
    fi
  else
    DATASET_ROOT="${INTERNVLA_ORACLE_DATASET_ROOT:-$CONTROL_ROOT/episodes/t3_continuous_oracle_go2_clear_v2}"
  fi
  CONFIG="${INTERNVLA_T3_CONFIG:-$CONTROL_ROOT/configs/internnav_t3/go2_continuous_oracle_cfg.py}"
else
  if test "$PHASE" = stress; then
    DATASET_ROOT="${INTERNVLA_T3_STRESS_DATASET_ROOT:-$CONTROL_ROOT/episodes/t3_obstacle_stress}"
    SOURCE_DATASET="${INTERNVLA_GO2_PILOT_DATASET_ROOT:-$HOME/internnav-t0/episodes/pilot}/val_unseen/val_unseen.json.gz"
    SCENARIO_MANIFEST="$DATASET_ROOT/obstacle_manifest.json"
    if test ! -f "$DATASET_ROOT/val_unseen/val_unseen.json.gz" || test ! -f "$SCENARIO_MANIFEST"; then
      python3 "$SCRIPT_ROOT/build_t3_obstacle_dataset.py" \
        --source "$SOURCE_DATASET" --output-root "$DATASET_ROOT" \
        --manifest "$SCENARIO_MANIFEST" \
        --exclude-manifest "$CONTROL_ROOT/configs/internnav_t3/obstacle_route_exclusions.json"
    fi
  else
    DATASET_ROOT="${INTERNVLA_GO2_DATASET_ROOT:-$HOME/internnav-t0/episodes/$SOURCE_PHASE}"
  fi
  CONFIG="${INTERNVLA_T3_CONFIG:-$CONTROL_ROOT/configs/internnav_t3/go2_continuous_active_cfg.py}"
fi

test -x "$ISAAC_PYTHON"
test -f "$ROS_WS/install/setup.bash"
test -f "$DATASET_ROOT/val_unseen/val_unseen.json.gz"
test -f "$GO2_SOURCE"
test -f "$PARAMS"
test -f "$CONFIG"
test ! -e "$RESULT_DIR/phase_status.json"
mkdir -p "$RESULT_DIR/logs" "$GO2_RUNTIME"
python3 - "$RESULT_DIR/collision_monitor_contract.json" "$PHASE" \
  "$COLLISION_MONITOR_STOP_FORWARD_M" <<'PY'
import json, sys
from pathlib import Path

forward=float(sys.argv[3])
payload={
    "schema_version":1,
    "phase":sys.argv[2],
    "polygon_points":[forward,0.26,forward,-0.26,-0.26,-0.26,-0.26,0.26],
    "stop_forward_m":forward,
    "source":"command_line_parameter_override",
}
Path(sys.argv[1]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
DATASET_SHA256="$(python3 - "$DATASET_ROOT/val_unseen/val_unseen.json.gz" <<'PY'
import hashlib,sys
print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())
PY
)"
STATIC_MAP_CLEARANCE_GATE="${INTERNVLA_T3_STATIC_CLEARANCE_GATE_M:-0.40}"
STATIC_MAP_GATE_TAG="${STATIC_MAP_CLEARANCE_GATE//./p}"
STATIC_MAP_CACHE="$CONTROL_ROOT/runtime/static_maps/t3_static_map_v4_${DATASET_SHA256:0:16}_${STATIC_MAP_GATE_TAG}"
python3 "$SCRIPT_ROOT/build_t3_static_maps.py" \
  --dataset "$DATASET_ROOT/val_unseen/val_unseen.json.gz" \
  --scene-root "$INTERNNAV_ROOT/data/scene_data/mp3d_pe" \
  --output-root "$STATIC_MAP_CACHE" \
  --minimum-required-prefix-clearance-m "$STATIC_MAP_CLEARANCE_GATE" \
  >"$RESULT_DIR/logs/static_map_build.log"
mkdir -p "$RESULT_DIR/static_maps"
cp "$STATIC_MAP_CACHE"/*.bin "$STATIC_MAP_CACHE/manifest.json" "$RESULT_DIR/static_maps/"
STATIC_MAP_MANIFEST="$RESULT_DIR/static_maps/manifest.json"
test -f "$STATIC_MAP_MANIFEST"
if test -n "${SCENARIO_MANIFEST:-}"; then
  cp "$SCENARIO_MANIFEST" "$RESULT_DIR/obstacle_manifest.json"
fi
if test "$CLIENT_KIND" = model && test -n "${SCENARIO_MANIFEST:-}"; then
  python3 - "$DATASET_ROOT/val_unseen/val_unseen.json.gz" \
    "$SCENARIO_MANIFEST" "$RESULT_DIR/scenario_manifest_validation.json" <<'PY'
import gzip, hashlib, json, sys
from pathlib import Path

dataset_path, manifest_path, output_path = map(Path, sys.argv[1:])
with gzip.open(dataset_path, "rt", encoding="utf-8") as stream:
    episodes = json.load(stream)["episodes"]
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
by_digest = {}
for item in manifest.get("scenarios", []):
    digest = str(item.get("instruction_runtime_sha256", ""))
    scenario = str(item.get("scenario", ""))
    if len(digest) != 64 or not scenario:
        raise RuntimeError("stress manifest lacks a runtime instruction digest")
    previous = by_digest.setdefault(digest, scenario)
    if previous != scenario:
        raise RuntimeError("one runtime instruction maps to conflicting scenarios")
def evaluator_instruction_text(episode):
    instruction = episode.get("instruction", "")
    if isinstance(instruction, dict) and "instruction_text" in instruction:
        return str(instruction["instruction_text"])
    return str(instruction)

observed = [
    hashlib.sha256(evaluator_instruction_text(episode).encode("utf-8")).hexdigest()
    for episode in episodes
]
missing = [digest for digest in observed if digest not in by_digest]
payload = {
    "schema_version": 1,
    "status": "PASS" if not missing and len(observed) == 10 else "FAIL",
    "episode_count": len(observed),
    "covered_episode_count": len(observed) - len(missing),
    "unique_runtime_digest_count": len(set(observed)),
    "missing_runtime_digests": missing,
}
output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
if payload["status"] != "PASS":
    raise RuntimeError("stress scenario manifest does not cover the frozen dataset")
PY
fi
rm -f "$AGENT_SOCKET" "$ORACLE_SOCKET" "$CONTROLLER_SOCKET" "$GO2_WRAPPER" "$GO2_MANIFEST"

STARTED_UNIX="$(date +%s)"
EVAL_EXIT=125
PREFLIGHT_EXIT=125
VALIDATION_EXIT=125
FINAL_STATUS=FAIL
nav2_pid=""; nav_lifecycle_pid=""; adapter_pid=""; controller_pid=""; collision_pid=""; lifecycle_pid=""; client_pid=""

stop_group() {
  local pid="$1"
  test -n "$pid" || return 0
  # `ros2 run` can exit before its launched node.  Track the whole setsid
  # process group instead of treating disappearance of the launcher PID as
  # proof that cleanup completed.
  kill -0 -- "-$pid" 2>/dev/null || { wait "$pid" 2>/dev/null || true; return 0; }
  kill -INT -- "-$pid" 2>/dev/null || true
  for _ in $(seq 1 50); do
    kill -0 -- "-$pid" 2>/dev/null || { wait "$pid" 2>/dev/null || true; return 0; }
    sleep 0.1
  done
  kill -TERM -- "-$pid" 2>/dev/null || true
  for _ in $(seq 1 20); do
    kill -0 -- "-$pid" 2>/dev/null || { wait "$pid" 2>/dev/null || true; return 0; }
    sleep 0.1
  done
  kill -KILL -- "-$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

write_phase_status() {
  local ended
  ended="$(date +%s)"
  python3 - "$RESULT_DIR/phase_status.json" "$PHASE" "$STARTED_UNIX" "$ended" "$PREFLIGHT_EXIT" "$EVAL_EXIT" "$VALIDATION_EXIT" "$FINAL_STATUS" <<'PY'
import json, sys
from pathlib import Path
path=Path(sys.argv[1])
payload={
    "schema_version":1,
    "phase":sys.argv[2],
    "started_unix":int(sys.argv[3]),
    "ended_unix":int(sys.argv[4]),
    "duration_sec":int(sys.argv[4])-int(sys.argv[3]),
    "exit_codes":{"preflight":int(sys.argv[5]),"evaluator":int(sys.argv[6]),"validation":int(sys.argv[7])},
    "status":sys.argv[8],
}
path.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
}

cleanup() {
  stop_group "$client_pid"
  stop_group "$controller_pid"
  stop_group "$adapter_pid"
  stop_group "$collision_pid"
  stop_group "$lifecycle_pid"
  stop_group "$nav_lifecycle_pid"
  stop_group "$nav2_pid"
  rm -f "$AGENT_SOCKET" "$ORACLE_SOCKET" "$CONTROLLER_SOCKET"
  write_phase_status
}
trap cleanup EXIT INT TERM

"$ISAAC_PYTHON" "$SCRIPT_ROOT/build_go2_internvla_usd.py" \
  --source "$GO2_SOURCE" --output "$GO2_WRAPPER" --manifest "$GO2_MANIFEST"

python3 - "$RESULT_DIR/input_manifest.json" \
  "dataset=$DATASET_ROOT/val_unseen/val_unseen.json.gz" \
  "eval_config=$CONFIG" "nav2_params=$PARAMS" \
  "go2_runtime=$SCRIPT_ROOT/internnav_go2_runtime.py" \
  "static_map_manifest=$STATIC_MAP_MANIFEST" \
  "obstacle_exclusions=$CONTROL_ROOT/configs/internnav_t3/obstacle_route_exclusions.json" \
  "collision_monitor_contract=$RESULT_DIR/collision_monitor_contract.json" \
  "phase_launcher=$SCRIPT_ROOT/run_go2_continuous_phase.sh" \
  "continuous_agent=$SCRIPT_ROOT/internvla_go2_continuous_agent_client.py" \
  "controller_bridge=$CONTROL_ROOT/internvla_go2_controller/internvla_go2_controller/bridge_node.py" \
  "nav2_adapter=$ROS_WS/src/internvla_nav2_adapter/internvla_nav2_adapter/active_node.py" <<'PY'
import hashlib,json,sys
from pathlib import Path
output=Path(sys.argv[1]); entries=[]
for argument in sys.argv[2:]:
    label,raw=argument.split("=",1); path=Path(raw)
    if not path.is_file(): raise FileNotFoundError(path)
    entries.append({"label":label,"bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
output.write_text(json.dumps({"schema_version":1,"files":entries},indent=2,sort_keys=True)+"\n")
PY

export INTERNNAV_T1_CONTROL_ROOT="$CONTROL_ROOT"
export INTERNNAV_T0_CONTROL_ROOT="$T0_CONTROL_ROOT"
export INTERNNAV_ROOT
export INTERNNAV_T0_PHASE="$SOURCE_PHASE"
export INTERNNAV_T0_DATASET_ROOT="$DATASET_ROOT"
export INTERNNAV_T0_RESULT_DIR="$RESULT_DIR"
export INTERNVLA_ORACLE_DATASET_ROOT="$DATASET_ROOT"
export INTERNVLA_ORACLE_RESULT_DIR="$RESULT_DIR"
export INTERNVLA_ORACLE_TASK_NAME="$TASK_NAME"
export INTERNVLA_GO2_WRAPPER_USD="$GO2_WRAPPER"
export INTERNVLA_GO2_EXECUTION_MODE=continuous
export INTERNVLA_GO2_CONTROL_HZ="${INTERNVLA_GO2_CONTROL_HZ:-50}"
export INTERNVLA_T3_TASK_NAME="$TASK_NAME"
export INTERNVLA_T3_PHASE="$PHASE"
if test -z "${INTERNVLA_T3_MAX_STEP:-}"; then
  if test "$PHASE" = diagnostics; then
    INTERNVLA_T3_MAX_STEP=2500
  elif test "$CLIENT_KIND" = oracle; then
    INTERNVLA_T3_MAX_STEP=12000
  else
    INTERNVLA_T3_MAX_STEP=8000
  fi
fi
export INTERNVLA_T3_MAX_STEP
export INTERNVLA_T3_SCENARIO_MANIFEST="${SCENARIO_MANIFEST:-}"
export INTERNVLA_GO2_RUNTIME_AUDIT="$RESULT_DIR/go2_runtime_audit.jsonl"
ISAAC_PYTHONPATH="$COMPAT_OVERLAY:$INTERNNAV_ROOT:$SCRIPT_ROOT:$CONTROL_ROOT/internvla_go2_controller"
if test "$CLIENT_KIND" = model; then
  # The active model config is imported by preflight before the client process
  # is launched, so all of its model-only static settings must exist already.
  export INTERNNAV_SERVER_HOST=ros2-typed-transport
  export INTERNVLA_GO2_ACTIVE_TASK_NAME="$TASK_NAME"
fi

set +e
PYTHONPATH="$ISAAC_PYTHONPATH" "$ISAAC_PYTHON" \
  "$SCRIPT_ROOT/preflight_go2_continuous.py" --config "$CONFIG" \
  >"$RESULT_DIR/preflight.json" 2>"$RESULT_DIR/logs/preflight.log"
PREFLIGHT_EXIT=$?
set -e
test "$PREFLIGHT_EXIT" = 0

source /opt/ros/humble/setup.bash
source "$ROS_WS/install/setup.bash"
set -u

setsid ros2 launch nav2_bringup navigation_launch.py \
  params_file:="$PARAMS" use_sim_time:=False autostart:=False \
  use_composition:=False use_respawn:=False >"$RESULT_DIR/logs/nav2.log" 2>&1 &
nav2_pid=$!

# Do not race the lifecycle manager against node construction.  A change-state
# service that is merely present in discovery is not sufficient; query every
# managed node once so a transient configure response cannot strand bringup.
: >"$RESULT_DIR/logs/nav2_readiness.log"
for node in $NAV2_LIFECYCLE_NODES; do
  ready=0
  for _ in $(seq 1 60); do
    if timeout 2 ros2 lifecycle get "$node" \
      >>"$RESULT_DIR/logs/nav2_readiness.log" 2>&1; then
      ready=1
      break
    fi
    sleep 0.25
  done
  if test "$ready" != 1; then
    echo "Nav2 lifecycle service did not become responsive: $node" \
      >>"$RESULT_DIR/logs/nav2_readiness.log"
    exit 1
  fi
done

setsid ros2 run nav2_lifecycle_manager lifecycle_manager --ros-args \
  -r __node:=lifecycle_manager_navigation_t3 -p use_sim_time:=false -p autostart:=true \
  -p "node_names:=$LIFECYCLE_NODE_NAMES" >"$RESULT_DIR/logs/nav2_lifecycle.log" 2>&1 &
nav_lifecycle_pid=$!
setsid ros2 run nav2_collision_monitor collision_monitor --ros-args \
  --params-file "$PARAMS" \
  -p "PolygonStop.points:=[$COLLISION_MONITOR_STOP_FORWARD_M,0.26,$COLLISION_MONITOR_STOP_FORWARD_M,-0.26,-0.26,-0.26,-0.26,0.26]" \
  >"$RESULT_DIR/logs/collision_monitor.log" 2>&1 &
collision_pid=$!
setsid ros2 run nav2_lifecycle_manager lifecycle_manager --ros-args \
  -r __node:=lifecycle_manager_collision -p use_sim_time:=false -p autostart:=true \
  -p "node_names:=['collision_monitor']" >"$RESULT_DIR/logs/collision_lifecycle.log" 2>&1 &
lifecycle_pid=$!
setsid "$ROS_WS/install/internvla_nav2_adapter/lib/internvla_nav2_adapter/internvla_nav2_active" \
  --ros-args -p result_dir:="$RESULT_DIR" -p command_mode:="$ADAPTER_COMMAND_MODE" \
  -p goal_max_distance_m:="$ADAPTER_GOAL_MAX_DISTANCE" \
  -p execution_mode:=continuous -p nav2_server_timeout_sec:=10.0 \
  >"$RESULT_DIR/logs/adapter.log" 2>&1 &
adapter_pid=$!
setsid "$ROS_WS/install/internvla_go2_controller/lib/internvla_go2_controller/internvla_go2_controller_bridge" \
  --ros-args --params-file "$PARAMS" -p result_dir:="$RESULT_DIR" \
  -p socket_path:="$CONTROLLER_SOCKET" -p static_map_manifest:="$STATIC_MAP_MANIFEST" \
  >"$RESULT_DIR/logs/controller.log" 2>&1 &
controller_pid=$!

if test "$CLIENT_KIND" = oracle; then
  INTERNVLA_ORACLE_SOCKET="$ORACLE_SOCKET" setsid \
    "$ROS_WS/install/internvla_ros2/lib/internvla_ros2/internvla_nav2_oracle_bridge" \
    --ros-args -p publish_observation_pose:=true \
    -p success_stop_distance_m:="$ORACLE_SUCCESS_STOP_DISTANCE" \
    >"$RESULT_DIR/logs/oracle_bridge.log" 2>&1 &
  client_pid=$!
  IPC_SOCKET="$ORACLE_SOCKET"
else
  INTERNVLA_CLIENT_SOCKET="$AGENT_SOCKET" setsid \
    "$ROS_WS/install/internvla_ros2/lib/internvla_ros2/internvla_client_node" --ros-args \
    -p service_timeout_sec:=300.0 -p step_deadline_sec:=30.0 -p step_validity_sec:=35.0 \
    -p ipc_idle_timeout_sec:=900.0 -p control_mode:=nav2 \
    -p nav2_resolution_timeout_sec:=10.0 -p publish_observation_pose:=true \
    -p result_dir:="$RESULT_DIR" \
    >"$RESULT_DIR/logs/client.log" 2>&1 &
  client_pid=$!
  IPC_SOCKET="$AGENT_SOCKET"
fi

for _ in $(seq 1 600); do
  test -S "$IPC_SOCKET" && test -S "$CONTROLLER_SOCKET" && break
  sleep 0.1
done
test -S "$IPC_SOCKET"; test -S "$CONTROLLER_SOCKET"
kill -0 "$nav2_pid"; kill -0 "$nav_lifecycle_pid"; kill -0 "$collision_pid"; kill -0 "$adapter_pid"; kill -0 "$controller_pid"; kill -0 "$client_pid"
for _ in $(seq 1 100); do
  ros2 lifecycle get /collision_monitor 2>/dev/null | grep -q active && break
  sleep 0.1
done
ros2 lifecycle get /collision_monitor | grep -q active

# Static map/TF publication starts with the evaluator, so controller activation
# may legitimately block here.  Configuration must nevertheless have
# completed and the manager must have advanced to that activation request.
for _ in $(seq 1 120); do
  grep -Fq "Activating controller_server" "$RESULT_DIR/logs/nav2_lifecycle.log" && break
  kill -0 "$nav_lifecycle_pid"
  sleep 0.25
done
grep -Fq "Activating controller_server" "$RESULT_DIR/logs/nav2_lifecycle.log"
{
  echo "# nodes"; ros2 node list | sort
  echo "# topics"; ros2 topic list -t | sort
  echo "# services"; ros2 service list -t | sort
  echo "# actions"; ros2 action list -t | sort
} >"$RESULT_DIR/ros_graph_snapshot.txt"

cd "$INTERNNAV_ROOT"
if test "$CLIENT_KIND" = oracle; then
  export INTERNVLA_GO2_CLIENT_MODE=continuous_oracle
  export INTERNVLA_ORACLE_SOCKET="$ORACLE_SOCKET"
  export INTERNVLA_ORACLE_DATASET_FILE="$DATASET_ROOT/val_unseen/val_unseen.json.gz"
  export INTERNVLA_ORACLE_TASK_NAME="$TASK_NAME"
else
  export INTERNVLA_GO2_CLIENT_MODE=continuous_model
  export INTERNVLA_CLIENT_SOCKET="$AGENT_SOCKET"
  export INTERNVLA_LOCAL_IPC_TIMEOUT_SEC=360
  export INTERNNAV_SERVER_HOST=ros2-typed-transport
  export INTERNVLA_GO2_ACTIVE_TASK_NAME="$TASK_NAME"
fi
export INTERNVLA_GO2_CONTROLLER_SOCKET="$CONTROLLER_SOCKET"

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
python3 "$SCRIPT_ROOT/extract_internnav_per_episode.py" \
  "$INTERNNAV_ROOT/logs/$TASK_NAME" "$RESULT_DIR/per_episode.json" \
  --expected "$EXPECTED_COUNT"

stop_group "$controller_pid"; controller_pid=""
stop_group "$adapter_pid"; adapter_pid=""
stop_group "$client_pid"; client_pid=""
test -f "$RESULT_DIR/controller_summary.json"
test -f "$RESULT_DIR/active_summary.json"
if test "$CLIENT_KIND" = model; then
  test -f "$RESULT_DIR/client_summary.json"
fi
python3 "$SCRIPT_ROOT/summarize_go2_controller_records.py" \
  "$RESULT_DIR/controller_records.jsonl" "$RESULT_DIR/controller_per_episode.json"

set +e
python3 - "$RESULT_DIR" "$PHASE" "$EXPECTED_COUNT" "$MIN_SR" "$STATIC_MAP_CLEARANCE_GATE" <<'PY'
import json, math, sys
from pathlib import Path
root=Path(sys.argv[1]); phase=sys.argv[2]; expected=int(sys.argv[3]); minimum=float(sys.argv[4])
static_clearance_gate=float(sys.argv[5])
result=json.loads((root/"result.json").read_text())
metrics=result.get("val_unseen",result)
active=json.loads((root/"active_summary.json").read_text())
controller=json.loads((root/"controller_summary.json").read_text())
per_episode=json.loads((root/"per_episode.json").read_text())
controller_episodes=json.loads((root/"controller_per_episode.json").read_text())
static_map=json.loads((root/"static_maps"/"manifest.json").read_text())
collision_contract=json.loads((root/"collision_monitor_contract.json").read_text())
client=json.loads((root/"client_summary.json").read_text()) if (root/"client_summary.json").is_file() else None
count=int(metrics.get("Count",metrics.get("length",0)))
sr=float(metrics.get("SR",metrics.get("sr",0.0)))
raw_success_count=int(per_episode.get("success_count",0))
raw_sr=raw_success_count/expected if expected else 0.0
hz=float(controller.get("measured_control_hz",0.0))
obstacle_phase=phase in {"obstacle_oracle","stress"}
obstacle_manifest=(
    json.loads((root/"obstacle_manifest.json").read_text()) if obstacle_phase else None
)
recovery_count=sum(int(item.get("recovery_count",0)) for item in controller_episodes.get("episodes",[]))
obstacle_acquired_episode_count=sum(
    int(
        item.get("obstacle_expected_target_count",0)>0
        and item.get("obstacle_detected_target_count",0)>0
    )
    for item in controller_episodes.get("episodes",[])
)
obstacle_acquisition_rate=(
    obstacle_acquired_episode_count/expected if obstacle_phase and expected else 0.0
)
diagnostic_motion_ok=(
    phase!="diagnostics"
    or all(
        float(item.get("measured_path_length_m",0.0))>=0.5
        for item in controller_episodes.get("episodes",[])
    )
)
diagnostic_control_ok=(
    phase!="diagnostics"
    or active.get("nav2_control_count",0)>=expected
)
obstacle_ok=(
    not obstacle_phase
    or (
        controller.get("costmap_expected_target_count",0)>0
        and controller.get("obstacle_detection_rate",0.0)>=0.95
        and obstacle_acquisition_rate>=0.95
        and controller.get("collision_monitor_stop_count",0)>0
        and controller.get("collision_monitor_recovery_count",0)>0
    )
)
passing=(
    # The evaluator's official Success metric is recomputed from final NE.
    # A task may terminate through its stuck handler while its final pose is
    # inside the success radius, so the progress-log termination label is not
    # an independent Success metric and must not override official SR.
    count==expected and sr>=minimum
    and active.get("status")=="FINISHED"
    and active.get("failure_count")==0
    and active.get("goal_count",0)>0
    and diagnostic_control_ok
    and active.get("cmd_vel_quantization_count")==0
    and active.get("direct_motion_bypass_count")==0
    and controller.get("status")=="FINISHED"
    and controller.get("flash_call_count")==0
    and controller.get("cmd_vel_quantization_count")==0
    and controller.get("direct_motion_bypass_count")==0
    and controller.get("static_map_publish_count")==expected
    and controller.get("static_map_selection_count")==expected
    and controller.get("static_map_max_start_xy_error_m",math.inf)<=0.05
    and controller.get("static_map_generation")==expected-1
    and len(str(controller.get("static_map_manifest_sha256","")))==64
    and abs(
        float(collision_contract.get("stop_forward_m",-1.0))
        - (0.50 if phase=="stress" else 0.38)
    )<=1e-9
    and len(collision_contract.get("polygon_points",[]))==8
    and (
        not obstacle_phase
        or (
            obstacle_manifest.get("excluded_pair_violation_count")==0
            and len(obstacle_manifest.get("exclusion_manifest_sha256",[]))>0
            and all(len(str(value))==64 for value in obstacle_manifest.get("exclusion_manifest_sha256",[]))
        )
    )
    and abs(float(static_map.get("required_prefix_minimum_clearance_gate_m",-1.0))-static_clearance_gate)<=1e-9
    and min(
        float(item.get("required_prefix_minimum_static_clearance_m",-1.0))
        for item in static_map.get("generations",[])
    )>=static_clearance_gate
    and controller.get("nan_count")==0
    and controller.get("fall_count")==0
    and controller.get("physical_collision_count")==0
    and controller.get("stale_or_identity_reject_count")==0
    and controller.get("control_interval_sample_count",0)>=expected*10
    and per_episode.get("completed_episode_count")==expected
    and controller_episodes.get("episode_count")==expected
    and diagnostic_motion_ok
    and (client is None or client.get("status")=="FINISHED")
    and obstacle_ok
    and math.isfinite(hz) and 20.0<=hz<=50.0
)
payload={
    "schema_version":1,"status":"PASS" if passing else "FAIL","phase":phase,
    "required":{
        "episode_count":expected,"minimum_sr":minimum,"control_hz_range":[20.0,50.0],
        "obstacle_detection_rate":0.95 if obstacle_phase else None,
        "physical_collision_count":0,
        "static_map_publish_count":expected,
        "static_map_minimum_clearance_m":static_clearance_gate,
        "minimum_measured_path_length_m_per_episode":0.5 if phase=="diagnostics" else None,
    },
    "metrics":metrics,"termination_label_success_rate":raw_sr,
    "active":active,"controller":controller,
    "collision_monitor_contract":collision_contract,
    "client":client,
    "per_episode":per_episode,"controller_per_episode":controller_episodes,
    "obstacle_acquired_episode_count":obstacle_acquired_episode_count,
    "obstacle_acquisition_rate":obstacle_acquisition_rate,
}
(root/"validation.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps(payload,indent=2,sort_keys=True))
raise SystemExit(0 if passing else 1)
PY
VALIDATION_EXIT=$?
set -e
test "$VALIDATION_EXIT" = 0

for file in "$RESULT_DIR"/logs/*.log; do
  test -f "$file" || continue
  python3 "$SCRIPT_ROOT/sanitize_internnav_log.py" "$file" \
    "$RESULT_DIR/logs_sanitized/$(basename "$file")"
done

FINAL_STATUS=PASS
