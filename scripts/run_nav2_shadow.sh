#!/usr/bin/env bash
set -eo pipefail

# T1.2 structural shadow only. This never commands the simulated robot and
# the deterministic free map is explicitly forbidden in active H1/Go2 gates.
CONTROL_ROOT="${INTERNNAV_T1_CONTROL_ROOT:-$HOME/internnav-t1-t2}"
ROS_WS="${INTERNVLA_ROS_WS:-$CONTROL_ROOT/ros_ws}"
RESULT_DIR="${INTERNVLA_NAV2_SHADOW_RESULT_DIR:-$CONTROL_ROOT/results/t1_2_nav2_shadow}"
CONFIG_ROOT="${INTERNVLA_CONFIG_ROOT:-$CONTROL_ROOT/configs/internnav_t1_t2}"
SCRIPT_ROOT="${INTERNVLA_SCRIPT_ROOT:-$CONTROL_ROOT/scripts}"
ROS_DISTRO_NAME="${INTERNVLA_ROS_DISTRO:-humble}"
EXPECTED_COMMANDS="${INTERNVLA_EXPECTED_COMMANDS:-120}"
MAP_YAML="$CONFIG_ROOT/nav2_shadow_map.yaml"
MAP_IMAGE="$CONFIG_ROOT/nav2_shadow_map.pgm"

test -f "/opt/ros/$ROS_DISTRO_NAME/setup.bash"
test -f "$ROS_WS/install/setup.bash"
test -f "$MAP_YAML"
test -f "$SCRIPT_ROOT/generate_nav2_shadow_map.py"
test -f "$SCRIPT_ROOT/run_internnav_ros2_replay.sh"
test -f "$SCRIPT_ROOT/validate_nav2_shadow.py"
if test -e "$RESULT_DIR/records.jsonl"; then
  echo "refusing to append existing Nav2 shadow result: $RESULT_DIR" >&2
  exit 2
fi
mkdir -p "$RESULT_DIR"
if ! test -f "$MAP_IMAGE"; then
  /usr/bin/python3 "$SCRIPT_ROOT/generate_nav2_shadow_map.py" --output "$MAP_IMAGE"
fi

# shellcheck disable=SC1090
source "/opt/ros/$ROS_DISTRO_NAME/setup.bash"
# shellcheck disable=SC1090
source "$ROS_WS/install/setup.bash"
set -u

map_pid=""
shadow_pid=""
cleanup() {
  if test -n "$shadow_pid"; then kill -- "-$shadow_pid" 2>/dev/null || true; fi
  if test -n "$map_pid"; then kill -- "-$map_pid" 2>/dev/null || true; fi
  if test -n "$shadow_pid"; then wait "$shadow_pid" 2>/dev/null || true; fi
  if test -n "$map_pid"; then wait "$map_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM

setsid ros2 run nav2_map_server map_server --ros-args \
  -p yaml_filename:="$MAP_YAML" \
  >"$RESULT_DIR/map_server.log" 2>&1 &
map_pid=$!
for _ in $(seq 1 100); do
  ros2 service type /map_server/change_state >/dev/null 2>&1 && break
  sleep 0.1
done
ros2 lifecycle set /map_server configure | tee "$RESULT_DIR/map_lifecycle.log"
ros2 lifecycle set /map_server activate | tee -a "$RESULT_DIR/map_lifecycle.log"

setsid ros2 run internvla_nav2_adapter internvla_nav2_shadow --ros-args \
  -p result_dir:="$RESULT_DIR" \
  >"$RESULT_DIR/shadow_node.log" 2>&1 &
shadow_pid=$!
sleep 1
kill -0 "$map_pid"
kill -0 "$shadow_pid"

INTERNVLA_REPLAY_RESULT_DIR="$RESULT_DIR/replay" \
  "$SCRIPT_ROOT/run_internnav_ros2_replay.sh"

SUMMARY="$RESULT_DIR/summary.json"
for _ in $(seq 1 300); do
  if test -f "$SUMMARY" && /usr/bin/python3 - "$SUMMARY" "$EXPECTED_COMMANDS" <<'PY'
import json
import sys
try:
    value = json.load(open(sys.argv[1], encoding="utf-8"))
except (OSError, ValueError):
    raise SystemExit(1)
raise SystemExit(0 if int(value.get("command_count", -1)) == int(sys.argv[2]) else 1)
PY
  then
    break
  fi
  sleep 0.1
done

kill -- "-$shadow_pid" 2>/dev/null || true
wait "$shadow_pid" 2>/dev/null || true
shadow_pid=""
/usr/bin/python3 "$SCRIPT_ROOT/validate_nav2_shadow.py" \
  --summary "$SUMMARY" \
  --expected-commands "$EXPECTED_COMMANDS" | tee "$RESULT_DIR/validation.log"
