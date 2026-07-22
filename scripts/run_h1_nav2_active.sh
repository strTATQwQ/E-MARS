#!/usr/bin/env bash
set -eo pipefail

PHASE="${1:-}"
case "$PHASE" in
  canary) EXPECTED_COUNT=5; MIN_SR=0.4 ;;
  pilot) EXPECTED_COUNT=20; MIN_SR=0.35 ;;
  *) echo "usage: $0 {canary|pilot}" >&2; exit 2 ;;
esac

CONTROL_ROOT="${INTERNNAV_T1_CONTROL_ROOT:-$HOME/internnav-t1-t2}"
T0_CONTROL_ROOT="${INTERNNAV_T0_CONTROL_ROOT:-$HOME/internnav-t0/control}"
ROS_WS="${INTERNVLA_ROS_WS:-$CONTROL_ROOT/ros_ws}"
INTERNNAV_ROOT="${INTERNNAV_ROOT:-$HOME/internnav-t0/InternNav}"
ISAAC_PYTHON="${INTERNNAV_ISAAC_PYTHON:-$HOME/env_isaacsim/bin/python}"
COMPAT_OVERLAY="${INTERNNAV_COMPAT_OVERLAY:-$HOME/internnav-t0/runtime/internutopia_2_2_0}"
DATASET_ROOT="${INTERNVLA_H1_DATASET_ROOT:-$HOME/internnav-t0/episodes/$PHASE}"
RESULT_DIR="${INTERNVLA_H1_RESULT_DIR:-$CONTROL_ROOT/results/t1_3_h1_nav2_$PHASE}"
PARAMS="${INTERNVLA_NAV2_PARAMS:-$CONTROL_ROOT/configs/internnav_t1_t2/nav2_active_rolling.yaml}"
CONFIG="${INTERNVLA_H1_CONFIG:-$CONTROL_ROOT/configs/internnav_t1_t2/h1_nav2_active_cfg.py}"
SCRIPT_ROOT="${INTERNVLA_SCRIPT_ROOT:-$CONTROL_ROOT/scripts}"
SOCKET_PATH="${INTERNVLA_CLIENT_SOCKET:-/tmp/internvla_h1_nav2_$PHASE.sock}"
TASK_NAME="${INTERNVLA_H1_ACTIVE_TASK_NAME:-internnav_t1_h1_nav2_${PHASE}_follow_path}"

test -x "$ISAAC_PYTHON"
test -f "$ROS_WS/install/setup.bash"
test -f "$DATASET_ROOT/val_unseen/val_unseen.json.gz"
test -f "$PARAMS"
test -f "$CONFIG"
test ! -e "$RESULT_DIR/active_records.jsonl"
test ! -e "$INTERNNAV_ROOT/data/sample_episodes/$TASK_NAME"
mkdir -p "$RESULT_DIR/logs"
rm -f "$SOCKET_PATH"

# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1090
source "$ROS_WS/install/setup.bash"
set -u

nav2_pid=""
adapter_pid=""
client_pid=""
cleanup() {
  if test -n "$client_pid"; then kill -- "-$client_pid" 2>/dev/null || true; fi
  if test -n "$adapter_pid"; then kill -- "-$adapter_pid" 2>/dev/null || true; fi
  if test -n "$nav2_pid"; then kill -- "-$nav2_pid" 2>/dev/null || true; fi
  if test -n "$client_pid"; then wait "$client_pid" 2>/dev/null || true; fi
  if test -n "$adapter_pid"; then wait "$adapter_pid" 2>/dev/null || true; fi
  if test -n "$nav2_pid"; then wait "$nav2_pid" 2>/dev/null || true; fi
  rm -f "$SOCKET_PATH"
}
trap cleanup EXIT INT TERM

setsid ros2 launch nav2_bringup navigation_launch.py \
  params_file:="$PARAMS" use_sim_time:=False autostart:=True \
  use_composition:=False use_respawn:=False \
  >"$RESULT_DIR/logs/nav2.log" 2>&1 &
nav2_pid=$!

setsid "$ROS_WS/install/internvla_nav2_adapter/lib/internvla_nav2_adapter/internvla_nav2_active" \
  --ros-args -p result_dir:="$RESULT_DIR" -p command_mode:=follow_path \
  >"$RESULT_DIR/logs/adapter.log" 2>&1 &
adapter_pid=$!

INTERNVLA_CLIENT_SOCKET="$SOCKET_PATH" setsid \
  "$ROS_WS/install/internvla_ros2/lib/internvla_ros2/internvla_client_node" \
  --ros-args \
  -p service_timeout_sec:=300.0 \
  -p step_deadline_sec:=30.0 \
  -p step_validity_sec:=35.0 \
  -p ipc_idle_timeout_sec:=900.0 \
  -p control_mode:=nav2 \
  -p nav2_resolution_timeout_sec:=10.0 \
  >"$RESULT_DIR/logs/client.log" 2>&1 &
client_pid=$!

for _ in $(seq 1 600); do
  test -S "$SOCKET_PATH" && break
  kill -0 "$nav2_pid"
  kill -0 "$adapter_pid"
  kill -0 "$client_pid"
  sleep 0.1
done
test -S "$SOCKET_PATH"
kill -0 "$nav2_pid"
kill -0 "$adapter_pid"
kill -0 "$client_pid"

cd "$INTERNNAV_ROOT"
env \
  OMNI_KIT_ACCEPT_EULA=Y \
  MESA_GL_VERSION_OVERRIDE=4.6 \
  PYTHONPATH="$COMPAT_OVERLAY:$INTERNNAV_ROOT:$SCRIPT_ROOT" \
  INTERNNAV_ROOT="$INTERNNAV_ROOT" \
  INTERNNAV_T0_CONTROL_ROOT="$T0_CONTROL_ROOT" \
  INTERNNAV_T1_CONTROL_ROOT="$CONTROL_ROOT" \
  INTERNNAV_T0_PHASE="$PHASE" \
  INTERNNAV_T0_DATASET_ROOT="$DATASET_ROOT" \
  INTERNNAV_T0_RESULT_DIR="$RESULT_DIR" \
  INTERNNAV_SERVER_HOST="ros2-typed-transport" \
  INTERNVLA_CLIENT_SOCKET="$SOCKET_PATH" \
  INTERNVLA_LOCAL_IPC_TIMEOUT_SEC=360 \
  INTERNVLA_H1_ACTIVE_TASK_NAME="$TASK_NAME" \
  "$ISAAC_PYTHON" "$SCRIPT_ROOT/run_internnav_t1_ros2_entrypoint.py" \
  --config "$CONFIG" 2>&1 | tee "$RESULT_DIR/logs/eval.log"

test -f "$INTERNNAV_ROOT/logs/$TASK_NAME/result.json"
cp "$INTERNNAV_ROOT/logs/$TASK_NAME/result.json" "$RESULT_DIR/result.json"

/usr/bin/python3 - "$RESULT_DIR" "$PHASE" "$EXPECTED_COUNT" "$MIN_SR" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
phase = sys.argv[2]
expected_count = int(sys.argv[3])
minimum_sr = float(sys.argv[4])
payload = json.loads((root / "result.json").read_text())
metrics = payload.get("val_unseen", payload)
active = json.loads((root / "active_summary.json").read_text())
records = [json.loads(line) for line in (root / "active_records.jsonl").read_text().splitlines()]
actions = Counter(int(record["selected_action"]) for record in records if "selected_action" in record)
sr = float(metrics.get("SR", metrics.get("sr", metrics.get("success", 0.0))))
count = int(metrics.get("length", metrics.get("Count", 0)))
nondegenerate = len(actions) >= 2 and actions.get(0, 0) < max(1, sum(actions.values()))
passing = (
    count == expected_count
    and sr >= minimum_sr
    and active.get("failure_count") == 0
    and active.get("goal_count", 0) > 0
    and active.get("nav2_control_count", 0) > 0
    and nondegenerate
)
summary = {
    "schema_version": 1,
    "status": "PASS" if passing else "FAIL",
    "phase": phase,
    "required": {"episode_count": expected_count, "minimum_sr": minimum_sr},
    "metrics": metrics,
    "active": active,
    "selected_action_distribution": dict(sorted(actions.items())),
    "nondegenerate_control": nondegenerate,
}
(root / "validation.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
raise SystemExit(0 if passing else 1)
PY
