#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: run_t4_dgx_onboard.sh --result-dir DIR --static-map-manifest FILE

Required environment:
  INTERNNAV_T4_RESOURCE_LEASE_ACK=dgx|dgx+isaac
  INTERNNAV_RUNTIME_POLICY=completion_sim
  INTERNNAV_SIMULATION_TARGET=isaac

Optional environment:
  INTERNVLA_T4_CONTROLLER_TCP_PORT       default 24137
  INTERNVLA_T4_DGX_BIND_IP               default 10.100.100.128
  INTERNVLA_T4_ISAAC_PEER_IP             default 10.100.120.111
  INTERNVLA_T4_ENABLE_RECOVERY           default 0
  INTERNVLA_T4_ALLOW_PRECLOCK_ZERO_TF_WARN_DROP default 0 (T5 only)
  INTERNVLA_T4_ALLOW_COMMAND_POSE_ANCHOR_FALLBACK default 0 (T5 completion_sim only)
EOF
  exit 64
}

RESULT_DIR=""
STATIC_MAP_MANIFEST=""
while (($#)); do
  case "$1" in
    --result-dir) (($# >= 2)) || usage; RESULT_DIR="$2"; shift 2 ;;
    --static-map-manifest) (($# >= 2)) || usage; STATIC_MAP_MANIFEST="$2"; shift 2 ;;
    *) usage ;;
  esac
done

case "${INTERNNAV_T4_RESOURCE_LEASE_ACK:-}" in
  dgx|dgx+isaac) ;;
  *) echo "DGX resource lease acknowledgement is required" >&2; exit 2 ;;
esac
test "${INTERNNAV_RUNTIME_POLICY:-}" = completion_sim
test "${INTERNNAV_SIMULATION_TARGET:-}" = isaac

CONTROL_ROOT="${INTERNNAV_T1_CONTROL_ROOT:-$HOME/internnav-t1-t2}"
ROS_WS="${INTERNVLA_ROS_WS:-$CONTROL_ROOT/ros_ws}"
ROS_DISTRO_NAME="${INTERNVLA_ROS_DISTRO:-jazzy}"
PARAMS="${INTERNVLA_NAV2_PARAMS:-$CONTROL_ROOT/configs/completion_sim/map/nav2_static_lidar.yaml}"
ENABLE_STEREO_FEED="${INTERNVLA_T4_ENABLE_STEREO_FEED:-0}"
case "$ENABLE_STEREO_FEED" in 0|1) ;; *) exit 64 ;; esac
DGX_BIND_IP="${INTERNVLA_T4_DGX_BIND_IP:-10.100.100.128}"
ISAAC_PEER_IP="${INTERNVLA_T4_ISAAC_PEER_IP:-10.100.120.111}"
TCP_PORT="${INTERNVLA_T4_CONTROLLER_TCP_PORT:-24137}"
ENABLE_RECOVERY="${INTERNVLA_T4_ENABLE_RECOVERY:-0}"
ENABLE_D435I="${INTERNVLA_T4_ENABLE_D435I:-1}"
ONBOARD_PROFILE="${INTERNVLA_T4_ONBOARD_PROFILE:-migration_smoke}"
RECOVERY_RUNTIME_MANIFEST="${INTERNVLA_T4_RECOVERY_RUNTIME_MANIFEST:-}"
ENABLE_SCHEDULED_REFRESH="${INTERNVLA_T4_ENABLE_SCHEDULED_REFRESH:-0}"
MAXIMUM_SCHEDULED_REFRESHES="${INTERNVLA_T4_MAXIMUM_SCHEDULED_REFRESHES:-1}"
USE_SIM_TIME="${INTERNVLA_ONBOARD_USE_SIM_TIME:-false}"
HOST_ROLE="${INTERNVLA_ONBOARD_HOST_ROLE:-dgx_onboard_compute}"
MODEL_OWNER="${INTERNVLA_ONBOARD_MODEL_OWNER:-local_dgx}"
NODE_NAMESPACE="${INTERNVLA_ONBOARD_NAMESPACE:-}"
ALLOW_PRECLOCK_ZERO_TF_WARN_DROP="${INTERNVLA_T4_ALLOW_PRECLOCK_ZERO_TF_WARN_DROP:-0}"
ALLOW_COMMAND_POSE_ANCHOR_FALLBACK="${INTERNVLA_T4_ALLOW_COMMAND_POSE_ANCHOR_FALLBACK:-0}"

test -n "$RESULT_DIR"
test -n "$STATIC_MAP_MANIFEST"
test -f "$STATIC_MAP_MANIFEST"
test -f "$PARAMS"
test -f "$ROS_WS/install/setup.bash"
test -f "$CONTROL_ROOT/t4_completion/map/warn_relay.py"
[[ "$DGX_BIND_IP" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]
[[ "$ISAAC_PEER_IP" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]
[[ "$TCP_PORT" =~ ^[0-9]+$ ]] && ((TCP_PORT >= 1024 && TCP_PORT <= 65535))
[[ "$ENABLE_RECOVERY" =~ ^[01]$ ]]
[[ "$ENABLE_D435I" =~ ^[01]$ ]]
[[ "$ENABLE_SCHEDULED_REFRESH" =~ ^[01]$ ]]
[[ "$MAXIMUM_SCHEDULED_REFRESHES" =~ ^[0-3]$ ]]
[[ "$ALLOW_PRECLOCK_ZERO_TF_WARN_DROP" =~ ^[01]$ ]]
[[ "$ALLOW_COMMAND_POSE_ANCHOR_FALLBACK" =~ ^[01]$ ]]
case "$USE_SIM_TIME" in true|false) ;; *) echo "invalid use_sim_time" >&2; exit 2 ;; esac
case "$HOST_ROLE" in dgx_onboard_compute|dgx_edge_compute) ;; *) echo "invalid host role" >&2; exit 2 ;; esac
case "$MODEL_OWNER" in local_dgx|remote_dgx_model) ;; *) echo "invalid model owner" >&2; exit 2 ;; esac
case "$NODE_NAMESPACE" in
  "") ;;
  /*) [[ "$NODE_NAMESPACE" =~ ^/[A-Za-z0-9_/-]+$ ]] ;;
  *) echo "invalid onboard namespace" >&2; exit 2 ;;
esac
ROS_REMAP_ARGS=()
NAV2_NAMESPACE_ARGS=()
NAV2_LAUNCH_FILE=navigation_launch.py
if test -n "$NODE_NAMESPACE"; then
  # tf2's default /tf and /tf_static names are absolute.  Nav2's namespaced
  # bringup remaps them to relative topics, so every directly launched T5 node
  # must receive the same remap.  The empty-namespace T4 path keeps no remaps.
  ROS_REMAP_ARGS=(-r "__ns:=$NODE_NAMESPACE" -r /tf:=tf -r /tf_static:=tf_static)
  NAV2_LAUNCH_FILE=bringup_launch.py
  NAV2_NAMESPACE_ARGS=("namespace:=$NODE_NAMESPACE" use_namespace:=True
    use_localization:=False slam:=False)
fi
case "$ONBOARD_PROFILE" in
  migration_smoke|migrated_oracle10|migrated_pilot20|migrated_recovery_a5|migrated_recovery_b5) ;;
  migrated_ablation20_full_system1_system2|migrated_ablation20_oracle_high_level_system1|migrated_ablation20_system2_oracle_local_path|migrated_ablation20_full_trajectory|migrated_ablation20_endpoint|migrated_ablation20_straight_line|migrated_ablation20_model_stop|migrated_ablation20_oracle_termination|migrated_ablation20_history_on|migrated_ablation20_history_off|migrated_ablation20_recovery_on|migrated_ablation20_recovery_off|migrated_ablation20_h1_view|migrated_ablation20_go2_view) ;;
  *) echo "invalid DGX onboard profile" >&2; exit 2 ;;
esac
if test "$ENABLE_RECOVERY" = 1; then
  test -n "$RECOVERY_RUNTIME_MANIFEST"
  test -f "$RECOVERY_RUNTIME_MANIFEST"
  test "${INTERNVLA_T4_RECOVERY_MODE:-}" = on
fi
ip -4 -o addr show scope global | awk '{sub(/\/.*/, "", $4); print $4}' | grep -Fxq "$DGX_BIND_IP"
test ! -e "$RESULT_DIR"
mkdir -p "$RESULT_DIR/logs"
RESULT_DIR="$(cd "$RESULT_DIR" && pwd -P)"
STATIC_MAP_MANIFEST="$(readlink -f "$STATIC_MAP_MANIFEST")"
PARAMS="$(readlink -f "$PARAMS")"

if test -n "$(ss -H -ltn "sport = :$TCP_PORT")"; then
  echo "controller TCP port is already in use: $TCP_PORT" >&2
  exit 73
fi

set +u
source "/opt/ros/$ROS_DISTRO_NAME/setup.bash"
source "$ROS_WS/install/setup.bash"
set -u
export PYTHONPATH="$CONTROL_ROOT:${PYTHONPATH:-}"
export ROS2CLI_NO_DAEMON=1
for package in \
  nav2_bringup nav2_collision_monitor internvla_t4_sensors \
  internvla_t4_recovery go2_sensor_bridge; do
  ros2 pkg prefix "$package" >/dev/null
done

nav2_pid=""
nav_lifecycle_pid=""
collision_lifecycle_pid=""
adapter_pid=""
controller_pid=""
go2_sensor_pid=""
relay_pid=""
recovery_pid=""
ONLINE_READY=0
SHUTDOWN_REASON="process_exit"
CURRENT_PHASE="startup"
LAST_READINESS_PROBE=0
FAILURE_DIAGNOSTIC_WRITTEN=0
LAST_RESOLVER_RC=unset
LAST_RESOLVER_ELAPSED_MS=0
LAST_RESOLVER_TYPE=""

tcp_ready() {
  local sockets
  sockets="$(ss -H -ltn "sport = :$TCP_PORT")" || return 1
  test -n "$sockets"
}

resolver_type_query() {
  # A ros2 CLI discovery subprocess may outlive its launcher or hang while the
  # daemon is changing state.  Bound every individual query, including a
  # TERM-ignoring child, so the supervisor can never mistake that hang for a
  # healthy startup.
  timeout --signal=TERM --kill-after=0.5s 3s \
    ros2 service type /internvla/nav2_resolve
}

resolver_ready() {
  local service_type="" query_rc=0 started_ms finished_ms
  started_ms="$(date +%s%3N)"
  service_type="$(resolver_type_query)" || query_rc=$?
  finished_ms="$(date +%s%3N)"
  LAST_RESOLVER_RC="$query_rc"
  LAST_RESOLVER_ELAPSED_MS=$((finished_ms - started_ms))
  LAST_RESOLVER_TYPE="$service_type"
  test "$query_rc" = 0 || return "$query_rc"
  test "$service_type" = internvla_ros2_msgs/srv/ResolveCommand
}

log_resolver_probe() {
  local scope="$1" attempt="$2" ready="$3" type_log
  type_log="${LAST_RESOLVER_TYPE//$'\n'/,}"
  printf 'scope=%s attempt=%s ready=%s rc=%s elapsed_ms=%s type=%q\n' \
    "$scope" "$attempt" "$ready" "$LAST_RESOLVER_RC" \
    "$LAST_RESOLVER_ELAPSED_MS" "$type_log" \
    >>"$RESULT_DIR/logs/onboard_readiness.log"
}

resolver_ready_bounded() {
  local attempt
  # The main readiness loop has already observed the exact service type.  Give
  # the final ROS CLI recheck a short bounded window so one discovery hiccup
  # cannot create a TOCTOU failure; a missing or wrong type still fails closed.
  for attempt in $(seq 1 2); do
    if resolver_ready 2>/dev/null; then
      log_resolver_probe final_recheck "$attempt" 1
      return 0
    fi
    log_resolver_probe final_recheck "$attempt" 0
    sleep 0.1
  done
  return 1
}

child_alive() {
  local pid="$1" state="" pgid=""
  test -n "$pid" || return 1
  read -r state pgid < <(ps -o stat=,pgid= -p "$pid") || return 1
  [[ "$state" != Z* && "$pgid" = "$pid" ]]
}

require_child_alive() {
  local component="$1" pid="$2"
  if ! child_alive "$pid"; then
    SHUTDOWN_REASON="required_child_exit:$component:$pid"
    CURRENT_PHASE="required_child_liveness:$component"
    return 1
  fi
}

write_failure_diagnostic() (
  set +e
  local rc="$1" line="$2"
  local output="$RESULT_DIR/onboard_failure_diagnostic.json"
  local temporary="$RESULT_DIR/.onboard_failure_diagnostic.$BASHPID.json"
  local children="$RESULT_DIR/.onboard_failure_children.$BASHPID.tsv"
  local tcp_state=0 relay_state=0 resolver_state=0 resolver_type=""
  : >"$children"

  record_child() {
    local component="$1" pid="$2" alive=0 group_alive=0 process_state=""
    if test -n "$pid"; then
      kill -0 "$pid" 2>/dev/null && alive=1
      kill -0 -- "-$pid" 2>/dev/null && group_alive=1
      process_state="$(ps -o stat= -p "$pid" 2>/dev/null | awk '{print $1}')"
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "$component" "$pid" "$alive" "$group_alive" "$process_state" >>"$children"
  }

  record_child nav2 "$nav2_pid"
  record_child nav_lifecycle "$nav_lifecycle_pid"
  record_child collision_lifecycle "$collision_lifecycle_pid"
  record_child adapter "$adapter_pid"
  record_child controller "$controller_pid"
  record_child go2_sensor_shadow "$go2_sensor_pid"
  record_child warn_only_relay "$relay_pid"
  record_child recovery "$recovery_pid"
  tcp_ready && tcp_state=1
  test -s "$RESULT_DIR/warn_only_relay.jsonl" && relay_state=1
  resolver_ready 2>"$RESULT_DIR/logs/failure_resolver_probe.log" \
    && resolver_state=1
  resolver_type="$LAST_RESOLVER_TYPE"
  printf 'rc=%s elapsed_ms=%s type=%q\n' "$LAST_RESOLVER_RC" \
    "$LAST_RESOLVER_ELAPSED_MS" "$resolver_type" \
    >>"$RESULT_DIR/logs/failure_resolver_probe.log"

  ONBOARD_FAILURE_PHASE="$CURRENT_PHASE" ONBOARD_FAILURE_SHUTDOWN_REASON="$SHUTDOWN_REASON" \
  ONBOARD_FAILURE_RESOLVER_TYPE="$resolver_type" \
  ONBOARD_FAILURE_RESOLVER_RC="$LAST_RESOLVER_RC" \
  ONBOARD_FAILURE_RESOLVER_ELAPSED_MS="$LAST_RESOLVER_ELAPSED_MS" \
  python3 - "$temporary" "$rc" "$line" "$ONLINE_READY" \
    "$LAST_READINESS_PROBE" "$tcp_state" "$relay_state" "$resolver_state" \
    "$children" "$RESULT_DIR" <<'PY'
import json
import os
import sys
import time
from pathlib import Path

output = Path(sys.argv[1])
result = Path(sys.argv[10])


def artifact(relative):
    path = result / relative
    item = {
        "relative_path": relative,
        "exists": path.is_file(),
        "nonempty": path.is_file() and path.stat().st_size > 0,
    }
    if path.is_file() and path.suffix == ".json":
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            item["json_error"] = str(exc)
        else:
            item["status"] = value.get("status") if isinstance(value, dict) else None
    return item


children = []
for raw in Path(sys.argv[9]).read_text(encoding="utf-8").splitlines():
    component, pid, alive, group_alive, process_state = raw.split("\t", 4)
    children.append(
        {
            "component": component,
            "pid": int(pid) if pid else None,
            "leader_alive": alive == "1",
            "process_group_alive": group_alive == "1",
            "process_state": process_state or None,
        }
    )

payload = {
    "schema_version": 1,
    "status": "FAIL",
    "exit_code": int(sys.argv[2]),
    "source_line": int(sys.argv[3]),
    "phase": os.environ.get("ONBOARD_FAILURE_PHASE", ""),
    "shutdown_reason": os.environ.get("ONBOARD_FAILURE_SHUTDOWN_REASON", ""),
    "online_ready": sys.argv[4] == "1",
    "readiness": {
        "last_probe": int(sys.argv[5]),
        "tcp_ready": sys.argv[6] == "1",
        "warn_only_relay_nonempty": sys.argv[7] == "1",
        "resolver_ready_diagnostic_reprobe": sys.argv[8] == "1",
        "resolver_type_diagnostic_reprobe": os.environ.get(
            "ONBOARD_FAILURE_RESOLVER_TYPE", ""
        ),
        "resolver_rc_diagnostic_reprobe": os.environ.get(
            "ONBOARD_FAILURE_RESOLVER_RC", "unset"
        ),
        "resolver_elapsed_ms_diagnostic_reprobe": int(
            os.environ.get("ONBOARD_FAILURE_RESOLVER_ELAPSED_MS", "0")
        ),
        "artifacts": [
            artifact("nav2_lifecycle_ready.json"),
            artifact("nav2_data_plane_ready.json"),
            artifact("bridge_role_ready.json"),
            artifact("warn_only_relay.jsonl"),
            artifact("ros_graph_snapshot.txt"),
            artifact("onboard_ready.json"),
        ],
    },
    "children": children,
    "recorded_unix": time.time(),
}
output.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
  test "$?" = 0 || { rm -f "$temporary" "$children"; return 1; }
  mv -f "$temporary" "$output" || { rm -f "$temporary" "$children"; return 1; }
  rm -f "$children"
)

on_error() {
  local rc="$1" line="$2"
  trap - ERR
  # Preserve a more specific cause already set by a fail-closed gate (for
  # example required_child_exit:controller:PID).  Only synthesize the generic
  # phase/line reason for commands that did not classify their own failure.
  if test "$SHUTDOWN_REASON" = process_exit; then
    if test "$ONLINE_READY" = 0; then
      SHUTDOWN_REASON="startup_error:$CURRENT_PHASE:line_$line"
    else
      SHUTDOWN_REASON="runtime_error:$CURRENT_PHASE:line_$line"
    fi
  fi
  if test "$FAILURE_DIAGNOSTIC_WRITTEN" = 0; then
    if write_failure_diagnostic "$rc" "$line"; then
      FAILURE_DIAGNOSTIC_WRITTEN=1
    fi
  fi
  return "$rc"
}

stop_group() {
  local pid="$1"
  test -n "$pid" || return 0
  if kill -0 -- "-$pid" 2>/dev/null; then
    kill -INT -- "-$pid" 2>/dev/null || true
    for _ in $(seq 1 40); do
      kill -0 -- "-$pid" 2>/dev/null || break
      sleep 0.1
    done
  fi
  if kill -0 -- "-$pid" 2>/dev/null; then
    kill -TERM -- "-$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 -- "-$pid" 2>/dev/null || break
      sleep 0.1
    done
  fi
  kill -KILL -- "-$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

write_status() {
  local status="$1" residual="$2"
  python3 - "$RESULT_DIR/onboard_status.json" "$status" "$ONLINE_READY" \
    "$SHUTDOWN_REASON" "$DGX_BIND_IP" "$ISAAC_PEER_IP" "$TCP_PORT" "$residual" \
    "$HOST_ROLE" "$MODEL_OWNER" "$USE_SIM_TIME" "$NODE_NAMESPACE" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "schema_version": 1,
            "status": sys.argv[2],
            "online_ready": bool(int(sys.argv[3])),
            "shutdown_reason": sys.argv[4],
            "host_role": sys.argv[9],
            "controller_transport": {
                "kind": "tcp",
                "bind_ip": sys.argv[5],
                "expected_peer_ip": sys.argv[6],
                "port": int(sys.argv[7]),
            },
            "residual_process_group_count": int(sys.argv[8]),
            "speed_control_owner": "dgx",
            "map_owner": "dgx",
            "navigation_owner": "dgx",
            "model_owner": sys.argv[10],
            "use_sim_time": sys.argv[11] == "true",
            "namespace": sys.argv[12],
            "go2_sensor_bridge_mode": "shadow_nonfatal",
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
}

cleanup() {
  local rc=$? residual=0 status=FAIL
  trap - ERR EXIT INT TERM HUP
  set +e
  stop_group "$recovery_pid"
  stop_group "$relay_pid"
  stop_group "$go2_sensor_pid"
  stop_group "$controller_pid"
  stop_group "$adapter_pid"
  stop_group "$collision_lifecycle_pid"
  stop_group "$nav_lifecycle_pid"
  stop_group "$nav2_pid"
  for pid in "$recovery_pid" "$relay_pid" "$go2_sensor_pid" "$controller_pid" \
    "$adapter_pid" "$collision_lifecycle_pid" "$nav_lifecycle_pid" "$nav2_pid"; do
    test -z "$pid" || ! kill -0 -- "-$pid" 2>/dev/null || residual=$((residual + 1))
  done
  if test -n "$(ss -H -ltn "sport = :$TCP_PORT")"; then
    residual=$((residual + 1))
  fi
  if test "$ONLINE_READY" = 1 && test "$residual" = 0 && \
      { test "$rc" = 0 || test "$rc" = 130 || test "$rc" = 143; }; then
    status=PASS
  fi
  write_status "$status" "$residual"
  exit "$rc"
}

on_signal() {
  SHUTDOWN_REASON="$1"
  case "$1" in
    INT) exit 130 ;;
    TERM|HUP) exit 143 ;;
  esac
}
trap cleanup EXIT
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM
trap 'on_signal HUP' HUP
trap 'on_error "$?" "$LINENO"' ERR

CURRENT_PHASE="input_manifest"
python3 - "$RESULT_DIR/input_manifest.json" "$PARAMS" "$STATIC_MAP_MANIFEST" "$0" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

items = []
for label, raw in zip(("nav2_params", "static_map_manifest", "launcher"), sys.argv[2:]):
    path = Path(raw).resolve()
    items.append(
        {
            "label": label,
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    )
Path(sys.argv[1]).write_text(
    json.dumps({"schema_version": 1, "files": items}, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
if test "$ENABLE_RECOVERY" = 1; then
  cp "$RECOVERY_RUNTIME_MANIFEST" "$RESULT_DIR/recovery_runtime_manifest.json"
  python3 - "$RESULT_DIR/recovery_runtime_manifest.json" \
    "$ENABLE_SCHEDULED_REFRESH" "$MAXIMUM_SCHEDULED_REFRESHES" <<'PY'
import json,sys
from pathlib import Path
path=Path(sys.argv[1]); value=json.loads(path.read_text(encoding="utf-8"))
value["scheduled_refresh_enabled"]=sys.argv[2]=="1"
value["maximum_scheduled_refreshes_per_episode"]=int(sys.argv[3])
path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="utf-8")
PY
fi

CURRENT_PHASE="child_startup"
if test "$USE_SIM_TIME" = false; then
  setsid ros2 launch nav2_bringup "$NAV2_LAUNCH_FILE" \
    params_file:="$PARAMS" use_sim_time:=False autostart:=False \
    use_composition:=False use_respawn:=False "${NAV2_NAMESPACE_ARGS[@]}" \
    >"$RESULT_DIR/logs/nav2.log" 2>&1 &
else
  setsid ros2 launch nav2_bringup "$NAV2_LAUNCH_FILE" \
    params_file:="$PARAMS" use_sim_time:=true autostart:=False \
    use_composition:=False use_respawn:=False "${NAV2_NAMESPACE_ARGS[@]}" \
    >"$RESULT_DIR/logs/nav2.log" 2>&1 &
fi
nav2_pid=$!

setsid ros2 run nav2_lifecycle_manager lifecycle_manager --ros-args \
  "${ROS_REMAP_ARGS[@]}" \
  -r __node:=lifecycle_manager_navigation_t4_dgx \
  -p use_sim_time:="$USE_SIM_TIME" -p autostart:=true \
  -p "node_names:=['controller_server','smoother_server','planner_server','behavior_server','bt_navigator']" \
  >"$RESULT_DIR/logs/nav2_lifecycle.log" 2>&1 &
nav_lifecycle_pid=$!
setsid ros2 run nav2_lifecycle_manager lifecycle_manager --ros-args \
  "${ROS_REMAP_ARGS[@]}" \
  -r __node:=lifecycle_manager_collision_t4_dgx \
  -p use_sim_time:="$USE_SIM_TIME" -p autostart:=true \
  -p "node_names:=['collision_monitor']" \
  >"$RESULT_DIR/logs/collision_lifecycle.log" 2>&1 &
collision_lifecycle_pid=$!

ADAPTER_EXTRA=(
  -p "ablation_variant_id:='${INTERNVLA_T4_VARIANT_ID:-none}'"
  -p "ablation_config_sha256:='${INTERNVLA_T4_VARIANT_CONFIG_SHA256:-none}'"
  -p "system_mode:='${INTERNVLA_T4_SYSTEM_MODE:-full_system1_system2}'"
  -p "trajectory_mode:='${INTERNVLA_T4_TRAJECTORY_MODE:-full_trajectory}'"
  -p "termination_mode:='${INTERNVLA_T4_TERMINATION_MODE:-model_stop}'"
  -p "history_mode:='${INTERNVLA_T4_HISTORY_MODE:-on}'"
  -p "recovery_mode:='${INTERNVLA_T4_RECOVERY_MODE:-off}'"
  -p "view_mode:='${INTERNVLA_T4_VIEW_MODE:-go2_view}'"
  -p "recovery_safety_freshness_timeout_sec:=${INTERNVLA_T4_RECOVERY_SAFETY_FRESHNESS_SEC:-2.0}"
)
if test -n "${INTERNVLA_T4_ABLATION_DATASET_FILE:-}"; then
  ADAPTER_EXTRA+=(-p "ablation_dataset_file:=$INTERNVLA_T4_ABLATION_DATASET_FILE")
fi
setsid ros2 run internvla_t4_recovery internvla_t4_adapter --ros-args \
  "${ROS_REMAP_ARGS[@]}" \
  -p result_dir:="$RESULT_DIR" -p use_sim_time:="$USE_SIM_TIME" \
  -p command_mode:=navigate_to_pose \
  -p goal_max_distance_m:="${INTERNVLA_T4_GOAL_MAX_DISTANCE_M:-1.0}" \
  -p execution_mode:=continuous -p nav2_server_timeout_sec:=10.0 \
  -p allow_command_pose_anchor_fallback:="$([[ $ALLOW_COMMAND_POSE_ANCHOR_FALLBACK = 1 ]] && printf true || printf false)" \
  "${ADAPTER_EXTRA[@]}" >"$RESULT_DIR/logs/adapter.log" 2>&1 &
adapter_pid=$!

if test "$USE_SIM_TIME" = false; then
  setsid ros2 run internvla_t4_sensors internvla_t4_sensor_bridge --ros-args \
    "${ROS_REMAP_ARGS[@]}" \
    --params-file "$PARAMS" -p use_sim_time:=false -p result_dir:="$RESULT_DIR" \
    -p ipc_transport:=tcp -p tcp_bind_host:="$DGX_BIND_IP" \
    -p tcp_port:="$TCP_PORT" -p tcp_expected_peer_ip:="$ISAAC_PEER_IP" \
    -p static_map_manifest:="$STATIC_MAP_MANIFEST" \
    -p map_source:=static_map -p pose_source:=ground_truth \
    -p enable_stereo_feed:="$([[ $ENABLE_STEREO_FEED = 1 ]] && printf true || printf false)" \
    >"$RESULT_DIR/logs/controller.log" 2>&1 &
else
  setsid ros2 run internvla_t4_sensors internvla_t4_sensor_bridge --ros-args \
    "${ROS_REMAP_ARGS[@]}" \
    --params-file "$PARAMS" -p use_sim_time:=true -p result_dir:="$RESULT_DIR" \
    -p ipc_transport:=tcp -p tcp_bind_host:="$DGX_BIND_IP" \
    -p tcp_port:="$TCP_PORT" -p tcp_expected_peer_ip:="$ISAAC_PEER_IP" \
    -p static_map_manifest:="$STATIC_MAP_MANIFEST" \
    -p map_source:=static_map -p pose_source:=ground_truth \
    -p enable_stereo_feed:="$([[ $ENABLE_STEREO_FEED = 1 ]] && printf true || printf false)" \
    >"$RESULT_DIR/logs/controller.log" 2>&1 &
fi
controller_pid=$!

setsid ros2 run go2_sensor_bridge go2_sensor_bridge --ros-args \
  "${ROS_REMAP_ARGS[@]}" \
  -p result_dir:="$RESULT_DIR/go2_sensor_bridge" -p use_sim_time:="$USE_SIM_TIME" \
  -p runtime_policy:=completion_sim \
  -p enable_d435i:="$([[ $ENABLE_D435I = 1 ]] && printf true || printf false)" \
  -p enable_lidar:=true -p enable_rgb:=false -p sensor_timeout_sec:=5.0 \
  -p allow_preclock_zero_tf_warn_drop:="$([[ $ALLOW_PRECLOCK_ZERO_TF_WARN_DROP = 1 ]] && printf true || printf false)" \
  >"$RESULT_DIR/logs/go2_sensor_bridge.log" 2>&1 &
go2_sensor_pid=$!

export INTERNNAV_T4_MAP_COMPANION_ACK=1
setsid python3 "$CONTROL_ROOT/t4_completion/map/warn_relay.py" \
  --result-dir "$RESULT_DIR" --ros-args "${ROS_REMAP_ARGS[@]}" \
  -p use_sim_time:="$USE_SIM_TIME" \
  >"$RESULT_DIR/logs/warn_only_relay.log" 2>&1 &
relay_pid=$!

if test "$ENABLE_RECOVERY" = 1; then
  setsid ros2 run internvla_t4_recovery internvla_t4_recovery --ros-args \
    "${ROS_REMAP_ARGS[@]}" \
    -p result_dir:="$RESULT_DIR" -p use_sim_time:="$USE_SIM_TIME" \
    -p progress_horizon_sec:="${INTERNVLA_T4_PROGRESS_HORIZON_SEC:-5.0}" \
    -p minimum_progress_m:="${INTERNVLA_T4_MINIMUM_PROGRESS_M:-0.08}" \
    -p oscillation_travel_m:="${INTERNVLA_T4_OSCILLATION_TRAVEL_M:-0.30}" \
    -p trajectory_refresh_distance_m:="${INTERNVLA_T4_REFRESH_DISTANCE_M:-0.30}" \
    -p trajectory_refresh_time_sec:="${INTERNVLA_T4_REFRESH_TIME_SEC:-2.0}" \
    -p trajectory_validity_sec:="${INTERNVLA_T4_TRAJECTORY_VALIDITY_SEC:-5.0}" \
    -p trajectory_deviation_m:="${INTERNVLA_T4_TRAJECTORY_DEVIATION_M:-0.60}" \
    -p recovery_scan_yaw_rad:="${INTERNVLA_T4_RECOVERY_SCAN_YAW_RAD:-1.0471975511965976}" \
    -p recovery_scan_speed_rps:="${INTERNVLA_T4_RECOVERY_SCAN_SPEED_RPS:-0.3}" \
    -p replan_deadline_sec:="${INTERNVLA_T4_REPLAN_DEADLINE_SEC:-30.0}" \
    -p recovery_cooldown_sec:="${INTERNVLA_T4_RECOVERY_COOLDOWN_SEC:-3.0}" \
    -p maximum_recoveries_per_episode:="${INTERNVLA_T4_MAXIMUM_RECOVERIES:-3}" \
    -p maximum_recovery_duration_sec:="${INTERNVLA_T4_MAXIMUM_RECOVERY_DURATION_SEC:-60.0}" \
    -p recovery_profile_id:="${INTERNVLA_T4_RECOVERY_PROFILE_ID:-completion-default}" \
    -p recovery_profile_sha256:="${INTERNVLA_T4_RECOVERY_PROFILE_SHA256:-none}" \
    -p enable_scheduled_refresh:="$([[ $ENABLE_SCHEDULED_REFRESH = 1 ]] && printf true || printf false)" \
    -p maximum_scheduled_refreshes_per_episode:="$MAXIMUM_SCHEDULED_REFRESHES" \
    -p enable_short_backup:=false \
    >"$RESULT_DIR/logs/recovery.log" 2>&1 &
  recovery_pid=$!
fi

LIFECYCLE_PROBE_ARGS=()
if [[ "$ONBOARD_PROFILE" == migrated_pilot20 || "$ONBOARD_PROFILE" == migrated_recovery_a5 || "$ONBOARD_PROFILE" == migrated_recovery_b5 || "$ONBOARD_PROFILE" == migrated_ablation20_* ]]; then
  # completion_sim pilot keeps bounded warn relay + simulation e-stop as the
  # motion boundary.  A lifecycle RPC race may leave Collision Monitor
  # inactive, which is recorded as a deviation instead of blocking function.
  LIFECYCLE_PROBE_ARGS+=(--allow-inactive /collision_monitor)
fi
CURRENT_PHASE="nav2_lifecycle_readiness_probe"
python3 "$CONTROL_ROOT/scripts/check_t4_nav2_lifecycle.py" \
  --output "$RESULT_DIR/nav2_lifecycle_ready.json" --timeout-sec 20 \
  --namespace "$NODE_NAMESPACE" \
  "${LIFECYCLE_PROBE_ARGS[@]}" \
  >"$RESULT_DIR/logs/nav2_active_probe.log" 2>&1
if test -n "$NODE_NAMESPACE"; then
  # The data-plane isolation proof is a T5 namespace gate.  Keep it out of the
  # frozen empty-namespace T4 readiness path.
  CURRENT_PHASE="nav2_data_plane_readiness_probe"
  python3 "$CONTROL_ROOT/scripts/check_t4_nav2_data_plane.py" \
    --output "$RESULT_DIR/nav2_data_plane_ready.json" --timeout-sec 20 \
    --namespace "$NODE_NAMESPACE" \
    >"$RESULT_DIR/logs/nav2_data_plane_probe.log" 2>&1
fi

CURRENT_PHASE="readiness_wait"
: >"$RESULT_DIR/logs/onboard_readiness.log"
for probe in $(seq 1 8); do
  LAST_READINESS_PROBE="$probe"
  tcp_state=0
  relay_state=0
  resolver_state=0
  tcp_ready && tcp_state=1
  test -s "$RESULT_DIR/warn_only_relay.jsonl" && relay_state=1
  resolver_ready 2>/dev/null && resolver_state=1
  log_resolver_probe readiness "$probe" "$resolver_state"
  if test "$tcp_state" = 1 && test "$relay_state" = 1 && \
      test "$resolver_state" = 1; then
    break
  fi
  require_child_alive nav2 "$nav2_pid"
  require_child_alive nav_lifecycle "$nav_lifecycle_pid"
  require_child_alive collision_lifecycle "$collision_lifecycle_pid"
  require_child_alive adapter "$adapter_pid"
  require_child_alive controller "$controller_pid"
  require_child_alive warn_only_relay "$relay_pid"
  printf 'scope=readiness_state attempt=%s tcp=%s relay=%s nav2=1\n' \
    "$probe" "$tcp_state" "$relay_state" \
    >>"$RESULT_DIR/logs/onboard_readiness.log"
  sleep 0.1
done
CURRENT_PHASE="readiness_tcp_recheck"
tcp_ready
CURRENT_PHASE="readiness_warn_only_relay_recheck"
test -s "$RESULT_DIR/warn_only_relay.jsonl"
CURRENT_PHASE="readiness_resolver_recheck"
resolver_ready_bounded
CURRENT_PHASE="readiness_lifecycle_receipt_recheck"
python3 - "$RESULT_DIR/nav2_lifecycle_ready.json" <<'PY'
import json,sys
value=json.load(open(sys.argv[1],encoding="utf-8"))
if value.get("status") != "PASS": raise SystemExit("Nav2 lifecycle readiness changed")
PY
if test -n "$NODE_NAMESPACE"; then
  CURRENT_PHASE="readiness_data_plane_receipt_recheck"
  python3 - "$RESULT_DIR/nav2_data_plane_ready.json" <<'PY'
import json,sys
value=json.load(open(sys.argv[1],encoding="utf-8"))
if value.get("status") != "PASS": raise SystemExit("Nav2 data-plane readiness changed")
PY
fi
CURRENT_PHASE="readiness_required_child_recheck"
require_child_alive nav2 "$nav2_pid"
require_child_alive nav_lifecycle "$nav_lifecycle_pid"
require_child_alive collision_lifecycle "$collision_lifecycle_pid"
require_child_alive adapter "$adapter_pid"
require_child_alive controller "$controller_pid"
require_child_alive warn_only_relay "$relay_pid"
test -z "$recovery_pid" || require_child_alive recovery "$recovery_pid"

CURRENT_PHASE="ros_graph_snapshot"
{
  echo "# nodes"; ros2 node list | sort
  echo "# topics"; ros2 topic list -t | sort
  echo "# actions"; ros2 action list -t | sort
} >"$RESULT_DIR/ros_graph_snapshot.txt"
CURRENT_PHASE="onboard_ready_receipt"
python3 - "$RESULT_DIR/onboard_ready.json" "$DGX_BIND_IP" "$ISAAC_PEER_IP" "$TCP_PORT" \
  "$HOST_ROLE" "$MODEL_OWNER" "$USE_SIM_TIME" "$NODE_NAMESPACE" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "schema_version": 1,
            "status": "READY",
            "host_role": sys.argv[5],
            "controller_endpoint": f"tcp://{sys.argv[2]}:{sys.argv[4]}",
            "expected_isaac_peer_ip": sys.argv[3],
            "owners": ["navigation", "map", "speed_control", "recovery"]
            + (["model"] if sys.argv[6] == "local_dgx" else []),
            "model_owner": sys.argv[6],
            "use_sim_time": sys.argv[7] == "true",
            "namespace": sys.argv[8],
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
ONLINE_READY=1
CURRENT_PHASE="online_supervision"

while :; do
  require_child_alive nav2 "$nav2_pid"
  require_child_alive nav_lifecycle "$nav_lifecycle_pid"
  require_child_alive collision_lifecycle "$collision_lifecycle_pid"
  require_child_alive adapter "$adapter_pid"
  require_child_alive controller "$controller_pid"
  require_child_alive warn_only_relay "$relay_pid"
  test -z "$recovery_pid" || require_child_alive recovery "$recovery_pid"
  sleep 1
done
