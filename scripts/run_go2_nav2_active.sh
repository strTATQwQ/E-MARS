#!/usr/bin/env bash
set -eo pipefail

PHASE="${1:-}"
case "$PHASE" in
  canary) EXPECTED_COUNT=5; MIN_SR=0.4 ;;
  pilot) EXPECTED_COUNT=20; MIN_SR=0.01 ;;
  *) echo "usage: $0 {canary|pilot}" >&2; exit 2 ;;
esac
CONTROL_ROOT="${INTERNNAV_T1_CONTROL_ROOT:-$HOME/internnav-t1-t2}"
T0_CONTROL_ROOT="${INTERNNAV_T0_CONTROL_ROOT:-$HOME/internnav-t0/control}"
ROS_WS="${INTERNVLA_ROS_WS:-$CONTROL_ROOT/ros_ws}"
INTERNNAV_ROOT="${INTERNNAV_ROOT:-$HOME/internnav-t0/InternNav}"
ISAAC_PYTHON="${INTERNNAV_ISAAC_PYTHON:-$HOME/env_isaacsim/bin/python}"
COMPAT_OVERLAY="${INTERNNAV_COMPAT_OVERLAY:-$HOME/internnav-t0/runtime/internutopia_2_2_0}"
DATASET_ROOT="${INTERNVLA_GO2_DATASET_ROOT:-$HOME/internnav-t0/episodes/$PHASE}"
RESULT_DIR="${INTERNVLA_GO2_RESULT_DIR:-$CONTROL_ROOT/results/t2_go2_nav2_$PHASE}"
PARAMS="${INTERNVLA_NAV2_PARAMS:-$CONTROL_ROOT/configs/internnav_t1_t2/nav2_active_rolling.yaml}"
CONFIG="${INTERNVLA_GO2_CONFIG:-$CONTROL_ROOT/configs/internnav_t1_t2/go2_nav2_active_cfg.py}"
SCRIPT_ROOT="${INTERNVLA_SCRIPT_ROOT:-$CONTROL_ROOT/scripts}"
SOCKET_PATH="${INTERNVLA_CLIENT_SOCKET:-/tmp/internvla_go2_$PHASE.sock}"
TASK_NAME="${INTERNVLA_GO2_ACTIVE_TASK_NAME:-internnav_t2_go2_nav2_${PHASE}_follow_path}"
GO2_SOURCE="${INTERNVLA_GO2_SOURCE_USD:-$HOME/isaacsim_assets/Assets/Isaac/6.0/Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd}"
GO2_RUNTIME="$CONTROL_ROOT/runtime/go2"
GO2_WRAPPER="$GO2_RUNTIME/go2_internvla.usda"
GO2_MANIFEST="$RESULT_DIR/go2_asset_manifest.json"

test -x "$ISAAC_PYTHON"; test -f "$ROS_WS/install/setup.bash"
test -f "$DATASET_ROOT/val_unseen/val_unseen.json.gz"; test -f "$GO2_SOURCE"
test -f "$PARAMS"; test -f "$CONFIG"; test ! -e "$RESULT_DIR/active_records.jsonl"
mkdir -p "$RESULT_DIR/logs" "$GO2_RUNTIME"
rm -f "$SOCKET_PATH" "$GO2_WRAPPER" "$GO2_MANIFEST"
"$ISAAC_PYTHON" "$SCRIPT_ROOT/build_go2_internvla_usd.py" \
  --source "$GO2_SOURCE" --output "$GO2_WRAPPER" --manifest "$GO2_MANIFEST"
source /opt/ros/humble/setup.bash
source "$ROS_WS/install/setup.bash"
set -u
nav2_pid=""; adapter_pid=""; client_pid=""
cleanup() {
  if test -n "$client_pid"; then kill -- "-$client_pid" 2>/dev/null || true; fi
  if test -n "$adapter_pid"; then kill -- "-$adapter_pid" 2>/dev/null || true; fi
  if test -n "$nav2_pid"; then kill -- "-$nav2_pid" 2>/dev/null || true; fi
  rm -f "$SOCKET_PATH"
}
trap cleanup EXIT INT TERM
setsid ros2 launch nav2_bringup navigation_launch.py \
  params_file:="$PARAMS" use_sim_time:=False autostart:=True \
  use_composition:=False use_respawn:=False >"$RESULT_DIR/logs/nav2.log" 2>&1 &
nav2_pid=$!
setsid "$ROS_WS/install/internvla_nav2_adapter/lib/internvla_nav2_adapter/internvla_nav2_active" \
  --ros-args -p result_dir:="$RESULT_DIR" -p command_mode:=follow_path \
  >"$RESULT_DIR/logs/adapter.log" 2>&1 &
adapter_pid=$!
INTERNVLA_CLIENT_SOCKET="$SOCKET_PATH" setsid \
  "$ROS_WS/install/internvla_ros2/lib/internvla_ros2/internvla_client_node" --ros-args \
  -p service_timeout_sec:=300.0 -p step_deadline_sec:=30.0 -p step_validity_sec:=35.0 \
  -p ipc_idle_timeout_sec:=900.0 -p control_mode:=nav2 -p nav2_resolution_timeout_sec:=10.0 \
  >"$RESULT_DIR/logs/client.log" 2>&1 &
client_pid=$!
for _ in $(seq 1 600); do test -S "$SOCKET_PATH" && break; sleep 0.1; done
test -S "$SOCKET_PATH"; kill -0 "$nav2_pid"; kill -0 "$adapter_pid"; kill -0 "$client_pid"

cd "$INTERNNAV_ROOT"
env OMNI_KIT_ACCEPT_EULA=Y MESA_GL_VERSION_OVERRIDE=4.6 \
  PYTHONPATH="$COMPAT_OVERLAY:$INTERNNAV_ROOT:$SCRIPT_ROOT" \
  INTERNNAV_ROOT="$INTERNNAV_ROOT" INTERNNAV_T0_CONTROL_ROOT="$T0_CONTROL_ROOT" \
  INTERNNAV_T0_PHASE="$PHASE" INTERNNAV_T0_DATASET_ROOT="$DATASET_ROOT" \
  INTERNNAV_T0_RESULT_DIR="$RESULT_DIR" INTERNNAV_SERVER_HOST="ros2-typed-transport" \
  INTERNVLA_GO2_CLIENT_MODE=model INTERNVLA_GO2_WRAPPER_USD="$GO2_WRAPPER" \
  INTERNVLA_CLIENT_SOCKET="$SOCKET_PATH" INTERNVLA_LOCAL_IPC_TIMEOUT_SEC=360 \
  INTERNVLA_GO2_ACTIVE_TASK_NAME="$TASK_NAME" \
  "$ISAAC_PYTHON" "$SCRIPT_ROOT/run_internnav_go2_entrypoint.py" --config "$CONFIG" \
  2>&1 | tee "$RESULT_DIR/logs/eval.log"
test -f "$INTERNNAV_ROOT/logs/$TASK_NAME/result.json"
cp "$INTERNNAV_ROOT/logs/$TASK_NAME/result.json" "$RESULT_DIR/result.json"
/usr/bin/python3 - "$RESULT_DIR" "$PHASE" "$EXPECTED_COUNT" "$MIN_SR" <<'PY'
import json, sys
from collections import Counter
from pathlib import Path
root=Path(sys.argv[1]); phase=sys.argv[2]; expected=int(sys.argv[3]); minimum=float(sys.argv[4])
payload=json.loads((root/"result.json").read_text()); metrics=payload.get("val_unseen",payload)
active=json.loads((root/"active_summary.json").read_text())
records=[json.loads(x) for x in (root/"active_records.jsonl").read_text().splitlines()]
actions=Counter(int(x["selected_action"]) for x in records if "selected_action" in x)
sr=float(metrics.get("SR",metrics.get("sr",0.0))); count=int(metrics.get("Count",metrics.get("length",0)))
nondegenerate=len(actions)>=2 and actions.get(0,0)<max(1,sum(actions.values()))
passing=count==expected and sr>=minimum and active.get("failure_count")==0 and active.get("goal_count",0)>0 and active.get("nav2_control_count",0)>0 and nondegenerate
summary={"schema_version":1,"status":"PASS" if passing else "FAIL","phase":phase,"required":{"episode_count":expected,"minimum_sr":minimum},"metrics":metrics,"active":active,"selected_action_distribution":dict(sorted(actions.items())),"nondegenerate_control":nondegenerate}
(root/"validation.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
print(json.dumps(summary,indent=2,sort_keys=True)); raise SystemExit(0 if passing else 1)
PY
