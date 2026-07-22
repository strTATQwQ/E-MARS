#!/usr/bin/env bash
set -eo pipefail

CONTROL_ROOT="${INTERNNAV_T1_CONTROL_ROOT:-$HOME/internnav-t1-t2}"
ROS_WS="${INTERNVLA_ROS_WS:-$CONTROL_ROOT/ros_ws}"
RESULT_DIR="${INTERNVLA_NAV2_ACTIVE_RESULT_DIR:-$CONTROL_ROOT/results/t1_3_active_replay}"
PARAMS="${INTERNVLA_NAV2_PARAMS:-$CONTROL_ROOT/configs/internnav_t1_t2/nav2_active_rolling.yaml}"
SCRIPT_ROOT="${INTERNVLA_SCRIPT_ROOT:-$CONTROL_ROOT/scripts}"
ROS_DISTRO_NAME="${INTERNVLA_ROS_DISTRO:-humble}"

test -f "/opt/ros/$ROS_DISTRO_NAME/setup.bash"
test -f "$ROS_WS/install/setup.bash"
test -f "$PARAMS"
test ! -e "$RESULT_DIR/active_records.jsonl"
mkdir -p "$RESULT_DIR"
source "/opt/ros/$ROS_DISTRO_NAME/setup.bash"
source "$ROS_WS/install/setup.bash"
set -u

nav2_pid=""
adapter_pid=""
cleanup() {
  if test -n "$adapter_pid"; then kill -- "-$adapter_pid" 2>/dev/null || true; fi
  if test -n "$nav2_pid"; then kill -- "-$nav2_pid" 2>/dev/null || true; fi
  if test -n "$adapter_pid"; then wait "$adapter_pid" 2>/dev/null || true; fi
  if test -n "$nav2_pid"; then wait "$nav2_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM

setsid ros2 launch nav2_bringup navigation_launch.py \
  params_file:="$PARAMS" use_sim_time:=False autostart:=True \
  use_composition:=False use_respawn:=False \
  >"$RESULT_DIR/nav2.log" 2>&1 &
nav2_pid=$!
for _ in $(seq 1 300); do
  ros2 action info /navigate_to_pose >/dev/null 2>&1 && break
  sleep 0.1
done

setsid ros2 run internvla_nav2_adapter internvla_nav2_active --ros-args \
  -p result_dir:="$RESULT_DIR" \
  -p command_mode:=follow_path \
  >"$RESULT_DIR/adapter.log" 2>&1 &
adapter_pid=$!
sleep 2
kill -0 "$nav2_pid"
kill -0 "$adapter_pid"

INTERNVLA_REPLAY_RESULT_DIR="$RESULT_DIR/replay" \
  "$SCRIPT_ROOT/run_internnav_ros2_replay.sh" --ros-args \
  -p control_mode:=nav2 \
  -p nav2_resolution_timeout_sec:=5.0

kill -- "-$adapter_pid" 2>/dev/null || true
wait "$adapter_pid" 2>/dev/null || true
adapter_pid=""
/usr/bin/python3 - "$RESULT_DIR" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
active = json.loads((root / "active_summary.json").read_text())
replay = json.loads((root / "replay" / "replay_result.json").read_text())
passing = (
    replay.get("status") == "PASS"
    and replay.get("completed_steps") == 120
    and replay.get("action_matches") == 120
    and replay.get("nav2_goal_sent_count", 0) > 0
    and replay.get("nav2_plan_valid_count", 0) > 0
    and active.get("failure_count") == 0
    and active.get("nav2_control_count", 0) > 0
)
summary = {"schema_version": 1, "status": "PASS" if passing else "FAIL", "active": active, "replay": replay}
(root / "validation.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
raise SystemExit(0 if passing else 1)
PY
