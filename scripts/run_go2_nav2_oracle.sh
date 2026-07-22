#!/usr/bin/env bash
set -eo pipefail

CONTROL_ROOT="${INTERNNAV_T1_CONTROL_ROOT:-$HOME/internnav-t1-t2}"
ROS_WS="${INTERNVLA_ROS_WS:-$CONTROL_ROOT/ros_ws}"
INTERNNAV_ROOT="${INTERNNAV_ROOT:-$HOME/internnav-t0/InternNav}"
ISAAC_PYTHON="${INTERNNAV_ISAAC_PYTHON:-$HOME/env_isaacsim/bin/python}"
COMPAT_OVERLAY="${INTERNNAV_COMPAT_OVERLAY:-$HOME/internnav-t0/runtime/internutopia_2_2_0}"
DATASET_ROOT="${INTERNVLA_ORACLE_DATASET_ROOT:-$CONTROL_ROOT/episodes/h1_nav2_oracle}"
RESULT_DIR="${INTERNVLA_GO2_ORACLE_RESULT_DIR:-$CONTROL_ROOT/results/t2_go2_nav2_oracle}"
PARAMS="${INTERNVLA_NAV2_PARAMS:-$CONTROL_ROOT/configs/internnav_t1_t2/nav2_active_rolling.yaml}"
CONFIG="${INTERNVLA_GO2_ORACLE_CONFIG:-$CONTROL_ROOT/configs/internnav_t1_t2/go2_nav2_oracle_cfg.py}"
SCRIPT_ROOT="${INTERNVLA_SCRIPT_ROOT:-$CONTROL_ROOT/scripts}"
SOCKET_PATH="${INTERNVLA_GO2_ORACLE_SOCKET:-/tmp/internvla_go2_oracle.sock}"
TASK_NAME="${INTERNVLA_ORACLE_TASK_NAME:-internnav_t2_go2_nav2_oracle}"
GO2_SOURCE="${INTERNVLA_GO2_SOURCE_USD:-$HOME/isaacsim_assets/Assets/Isaac/6.0/Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd}"
GO2_RUNTIME="$CONTROL_ROOT/runtime/go2"
GO2_WRAPPER="$GO2_RUNTIME/go2_internvla.usda"
GO2_MANIFEST="$RESULT_DIR/go2_asset_manifest.json"

test -x "$ISAAC_PYTHON"
test -f "$ROS_WS/install/setup.bash"
test -f "$DATASET_ROOT/val_unseen/val_unseen.json.gz"
test -f "$GO2_SOURCE"
test -f "$PARAMS"
test -f "$CONFIG"
test ! -e "$RESULT_DIR/active_records.jsonl"
mkdir -p "$RESULT_DIR/logs" "$GO2_RUNTIME"
rm -f "$SOCKET_PATH" "$GO2_WRAPPER" "$GO2_MANIFEST"
"$ISAAC_PYTHON" "$SCRIPT_ROOT/build_go2_internvla_usd.py" \
  --source "$GO2_SOURCE" --output "$GO2_WRAPPER" --manifest "$GO2_MANIFEST"

source /opt/ros/humble/setup.bash
source "$ROS_WS/install/setup.bash"
set -u
nav2_pid=""; adapter_pid=""; bridge_pid=""
cleanup() {
  if test -n "$bridge_pid"; then kill -- "-$bridge_pid" 2>/dev/null || true; fi
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
  --ros-args -p result_dir:="$RESULT_DIR" >"$RESULT_DIR/logs/adapter.log" 2>&1 &
adapter_pid=$!
INTERNVLA_ORACLE_SOCKET="$SOCKET_PATH" setsid \
  "$ROS_WS/install/internvla_ros2/lib/internvla_ros2/internvla_nav2_oracle_bridge" \
  >"$RESULT_DIR/logs/oracle_bridge.log" 2>&1 &
bridge_pid=$!
for _ in $(seq 1 300); do test -S "$SOCKET_PATH" && break; sleep 0.1; done
test -S "$SOCKET_PATH"
kill -0 "$nav2_pid"; kill -0 "$adapter_pid"; kill -0 "$bridge_pid"

cd "$INTERNNAV_ROOT"
env OMNI_KIT_ACCEPT_EULA=Y MESA_GL_VERSION_OVERRIDE=4.6 \
  PYTHONPATH="$COMPAT_OVERLAY:$INTERNNAV_ROOT:$SCRIPT_ROOT" \
  INTERNNAV_ROOT="$INTERNNAV_ROOT" \
  INTERNVLA_GO2_CLIENT_MODE=oracle \
  INTERNVLA_GO2_WRAPPER_USD="$GO2_WRAPPER" \
  INTERNVLA_ORACLE_SOCKET="$SOCKET_PATH" \
  INTERNVLA_ORACLE_DATASET_FILE="$DATASET_ROOT/val_unseen/val_unseen.json.gz" \
  INTERNVLA_ORACLE_DATASET_ROOT="$DATASET_ROOT" \
  INTERNVLA_ORACLE_RESULT_DIR="$RESULT_DIR" \
  INTERNVLA_ORACLE_TASK_NAME="$TASK_NAME" \
  "$ISAAC_PYTHON" "$SCRIPT_ROOT/run_internnav_go2_entrypoint.py" --config "$CONFIG" \
  2>&1 | tee "$RESULT_DIR/logs/eval.log"

test -f "$INTERNNAV_ROOT/logs/$TASK_NAME/result.json"
cp "$INTERNNAV_ROOT/logs/$TASK_NAME/result.json" "$RESULT_DIR/result.json"
/usr/bin/python3 - "$RESULT_DIR" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
payload = json.loads((root / "result.json").read_text())
metrics = payload.get("val_unseen", payload)
active = json.loads((root / "active_summary.json").read_text())
sr = float(metrics.get("SR", metrics.get("sr", 0.0)))
count = int(metrics.get("Count", metrics.get("length", 0)))
passing = count == 10 and sr >= 0.9 and active.get("failure_count") == 0 and active.get("goal_count", 0) > 0
summary = {"schema_version": 1, "status": "PASS" if passing else "FAIL", "metrics": metrics, "active": active}
(root / "validation.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
raise SystemExit(0 if passing else 1)
PY
