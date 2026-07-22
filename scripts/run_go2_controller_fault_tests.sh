#!/usr/bin/env bash
set -eo pipefail

CONTROL_ROOT="${INTERNNAV_T1_CONTROL_ROOT:-$HOME/internnav-t1-t2}"
ROS_WS="${INTERNVLA_ROS_WS:-$CONTROL_ROOT/ros_ws}"
RESULT_DIR="${INTERNVLA_T3_FAULT_RESULT_DIR:-$CONTROL_ROOT/results/internnav_t3/t3_1_controller_faults}"
SOCKET_PATH="${INTERNVLA_GO2_CONTROLLER_SOCKET:-/tmp/internvla_t3_fault_controller.sock}"
PARAMS="${INTERNVLA_NAV2_PARAMS:-$CONTROL_ROOT/configs/internnav_t3/nav2_continuous.yaml}"
test -f "$ROS_WS/install/setup.bash"; test -f "$PARAMS"
test ! -e "$RESULT_DIR/controller_records.jsonl"
mkdir -p "$RESULT_DIR/logs"
rm -f "$SOCKET_PATH"
source /opt/ros/humble/setup.bash
source "$ROS_WS/install/setup.bash"
bridge_pid=""
cleanup() {
  if test -n "$bridge_pid" && kill -0 "$bridge_pid" 2>/dev/null; then
    kill -INT -- "-$bridge_pid" 2>/dev/null || true
    wait "$bridge_pid" 2>/dev/null || true
  fi
  rm -f "$SOCKET_PATH"
}
trap cleanup EXIT INT TERM
setsid "$ROS_WS/install/internvla_go2_controller/lib/internvla_go2_controller/internvla_go2_controller_bridge" \
  --ros-args --params-file "$PARAMS" -p result_dir:="$RESULT_DIR" \
  -p socket_path:="$SOCKET_PATH" >"$RESULT_DIR/logs/bridge.log" 2>&1 &
bridge_pid=$!
for _ in $(seq 1 100); do test -S "$SOCKET_PATH" && break; sleep 0.1; done
test -S "$SOCKET_PATH"; kill -0 "$bridge_pid"
python3 "$CONTROL_ROOT/scripts/run_go2_controller_fault_tests.py" \
  --socket "$SOCKET_PATH" --result-dir "$RESULT_DIR"
kill -INT -- "-$bridge_pid"; wait "$bridge_pid" || true; bridge_pid=""
python3 - "$RESULT_DIR" <<'PY'
import json, math, sys
from pathlib import Path
root=Path(sys.argv[1])
fault=json.loads((root/"fault_validation.json").read_text())
summary=json.loads((root/"controller_summary.json").read_text())
hz=float(summary["measured_control_hz"])
passing=(
    fault["status"]=="PASS" and summary["status"]=="FINISHED"
    and math.isfinite(hz) and 20.0<=hz<=50.0
    and summary["depth_frame_count"]>=20
    and summary["depth_nonempty_rate"]>=0.95
    and summary["flash_call_count"]==0
    and summary["cmd_vel_quantization_count"]==0
    and summary["direct_motion_bypass_count"]==0
    and summary["physical_collision_count"]==1
    and summary["fall_count"]==1
    and summary["nan_count"]==1
)
payload={"schema_version":1,"status":"PASS" if passing else "FAIL","faults":fault,"controller":summary}
(root/"validation.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps(payload,indent=2,sort_keys=True))
raise SystemExit(0 if passing else 1)
PY
