#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'usage: %s RESULT_DIR STATIC_MAP_MANIFEST\n' "${0##*/}" >&2
  exit 64
}

[[ $# -eq 2 ]] || usage
result_dir="$1"
static_map_manifest="$2"
root="${INTERNNAV_T1_CONTROL_ROOT:-$HOME/internnav-t1-t2}"
ros_ws="${INTERNVLA_ROS_WS:-$root/ros_ws}"
params="${INTERNVLA_T5_PARAMS:-$root/configs/internnav_t5/primary_completion_sim.yaml}"
nav2_params="${INTERNVLA_NAV2_PARAMS:-$root/configs/internnav_t5/nav2_static_lidar.yaml}"
lane_ip=10.100.120.122
isaac_ip=10.100.120.123
controller_port=25138
client_port=25239
namespace=/t5/lane_b
step3_venv="${INTERNNAV_T5_STEP3_VENV:-/home/rail/ai-stack/venvs/step3-vl-10b-tf4.57.6}"

test "$(id -un)" = rail
test "${INTERNNAV_T5_RESOURCE_LEASE_ACK:-}" = lane-b
test "${INTERNNAV_T5_STEP3_DIRECT_HIGH_LEVEL:-0}" = 1
test "${INTERNNAV_T5_STEP3_LIVE_ADVISOR:-0}" = 0
test "${INTERNNAV_T5_LANE:-}" = b
test "${INTERNNAV_T5_LANE_NAMESPACE:-}" = "$namespace"
test "${INTERNNAV_T5_ID_PREFIX:-}" = 'b::'
test "${INTERNNAV_RUNTIME_POLICY:-}" = completion_sim
test "${INTERNNAV_SIMULATION_TARGET:-}" = isaac
test "${ROS_DOMAIN_ID:-}" = 76
test "${CUDA_VISIBLE_DEVICES:-}" = 0
test -f "$static_map_manifest"
test -f "$params"
test -f "$nav2_params"
test -f "$ros_ws/install/setup.bash"
test -f "$root/scripts/run_t4_dgx_onboard.sh"
test -f "$root/scripts/run_t5_step3_live_services.sh"
test -d "${INTERNNAV_T5_STEP3_MODEL_PATH:-/home/rail/ai-stack/models/Step3-VL-10B}"
test -f "$step3_venv/T5_STEP3_RUNTIME_READY.json"
step3_site="$("$step3_venv/bin/python" -c 'import site; from pathlib import Path; values=[value for value in site.getsitepackages() if (Path(value) / "zmq").is_dir()]; assert len(values) == 1; print(values[0])')"
test -d "$step3_site/zmq"
"$step3_venv/bin/python" -c 'import zmq; assert zmq.__version__ == "27.1.0"'
[[ "$result_dir" = /home/rail/* ]]
test ! -e "$result_dir"
for port in "$controller_port" "$client_port" 8200 8300; do
  test -z "$(ss -H -lntp | grep -E "[.:]${port}[[:space:]]" || true)"
done

mkdir -p "$result_dir/logs" "$result_dir/health" "$result_dir/client"
result_dir="$(cd -- "$result_dir" && pwd -P)"
static_map_manifest="$(readlink -f "$static_map_manifest")"
pid_ledger="$result_dir/pid_ledger.jsonl"
: >"$pid_ledger"
onboard_pid=""
step3_pid=""
client_pid=""
shutdown_reason=startup_failure

record_pid() {
  local component="$1" pid="$2" event="$3" pgid=""
  test -z "$pid" || pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
  python3 - "$pid_ledger" "$component" "$pid" "$pgid" "$event" <<'PY'
import json, os, sys, time
from pathlib import Path
row = {
    "schema_version": 1, "component": sys.argv[2],
    "pid": int(sys.argv[3]) if sys.argv[3] else None,
    "pgid": int(sys.argv[4]) if sys.argv[4] else None,
    "event": sys.argv[5], "host": os.uname().nodename,
    "wall_unix": time.time(),
}
with Path(sys.argv[1]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(row, sort_keys=True) + "\n")
PY
}

stop_group() {
  local component="$1" pid="$2"
  test -n "$pid" || return 0
  record_pid "$component" "$pid" stop_requested
  kill -TERM -- "-$pid" 2>/dev/null || true
  for _ in $(seq 1 200); do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 0.1
  done
  kill -KILL -- "-$pid" 2>/dev/null || true
  for _ in $(seq 1 50); do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 0.1
  done
  return 75
}

audit_no_internvla() {
  local output="$1"
  python3 - "$output" "$root" <<'PY'
import json, os, sys, time
from pathlib import Path
output, root = Path(sys.argv[1]), str(Path(sys.argv[2]).resolve()).encode()
matches = []
for entry in Path("/proc").iterdir():
    if not entry.name.isdigit():
        continue
    try:
        if entry.stat().st_uid != os.getuid():
            continue
        argv = (entry / "cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    if root in argv and (
        b"internvla_t4_recovery.model_node" in argv
        or b"run_t4_model_server.sh" in argv
    ):
        matches.append(int(entry.name))
payload = {
    "schema_version": 1, "status": "PASS" if not matches else "FAIL",
    "internvla_model_loaded": False, "detected_pids": sorted(matches),
    "recorded_unix": time.time(),
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
raise SystemExit(0 if payload["status"] == "PASS" else 75)
PY
}

cleanup() {
  local rc=$? residual=0
  trap - EXIT INT TERM HUP
  set +e
  stop_group client "$client_pid" || residual=$((residual + 1))
  stop_group step3 "$step3_pid" || residual=$((residual + 1))
  stop_group onboard "$onboard_pid" || residual=$((residual + 1))
  audit_no_internvla "$result_dir/health/no_internvla_cleanup.json" || residual=$((residual + 1))
  for port in "$controller_port" "$client_port" 8200 8300; do
    test -z "$(ss -H -lntp | grep -E "[.:]${port}[[:space:]]" || true)" || residual=$((residual + 1))
  done
  python3 - "$result_dir/cleanup.json" "$rc" "$residual" "$shutdown_reason" <<'PY'
import json, sys, time
from pathlib import Path
rc, residual = map(int, sys.argv[2:4])
payload = {
    "schema_version": 1,
    "status": "PASS" if residual == 0 else "FAIL",
    "command_exit": rc, "residual_count": residual,
    "shutdown_reason": sys.argv[4], "dgx_b_released": residual == 0,
    "recorded_unix": time.time(),
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  test "$residual" = 0 || rc=75
  exit "$rc"
}
trap cleanup EXIT
trap 'shutdown_reason=signal_int; exit 130' INT
trap 'shutdown_reason=signal_term; exit 143' TERM HUP

audit_no_internvla "$result_dir/health/no_internvla_prestart.json"
set +u
source "/opt/ros/${INTERNVLA_ROS_DISTRO:-jazzy}/setup.bash"
source "$ros_ws/install/setup.bash"
set -u

common_env=(
  INTERNNAV_T1_CONTROL_ROOT="$root"
  INTERNVLA_ROS_WS="$ros_ws"
  INTERNNAV_RUNTIME_POLICY=completion_sim
  INTERNNAV_SIMULATION_TARGET=isaac
  INTERNNAV_T5_RESOURCE_LEASE_ACK=lane-b
  INTERNNAV_T4_RESOURCE_LEASE_ACK=dgx+isaac
  INTERNNAV_T5_LANE=b
  INTERNNAV_T5_LANE_NAMESPACE="$namespace"
  INTERNNAV_T5_ID_PREFIX='b::'
  INTERNVLA_EPISODE_ID_PREFIX='b::'
  INTERNVLA_RESET_ID_PREFIX='b::'
  INTERNVLA_REQUEST_ID_PREFIX='b::'
  INTERNNAV_T5_STEP3_LIVE_ADVISOR=0
  INTERNNAV_T5_STEP3_DIRECT_HIGH_LEVEL=1
  INTERNNAV_T5_STEP3_ENDPOINT=tcp://127.0.0.1:8200
  ROS_DOMAIN_ID=76
  ROS_NAMESPACE="$namespace"
  ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
  ROS_STATIC_PEERS="$isaac_ip"
  CUDA_VISIBLE_DEVICES=0
  T5_LANE_B_RESULTS="$result_dir"
  PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
)

setsid env "${common_env[@]}" \
  INTERNVLA_T4_DGX_BIND_IP="$lane_ip" \
  INTERNVLA_T4_ISAAC_PEER_IP="$isaac_ip" \
  INTERNVLA_T4_CONTROLLER_TCP_PORT="$controller_port" \
  INTERNVLA_T4_ENABLE_D435I=1 \
  INTERNVLA_T4_ONBOARD_PROFILE=migrated_pilot20 \
  INTERNVLA_ONBOARD_USE_SIM_TIME=true \
  INTERNVLA_ONBOARD_HOST_ROLE=dgx_onboard_compute \
  INTERNVLA_ONBOARD_MODEL_OWNER=remote_dgx_model \
  INTERNVLA_ONBOARD_NAMESPACE="$namespace" \
  INTERNVLA_T4_ALLOW_PRECLOCK_ZERO_TF_WARN_DROP=1 \
  INTERNVLA_T4_ALLOW_COMMAND_POSE_ANCHOR_FALLBACK=1 \
  INTERNVLA_NAV2_PARAMS="$nav2_params" \
  bash "$root/scripts/run_t4_dgx_onboard.sh" \
    --result-dir "$result_dir/onboard" \
    --static-map-manifest "$static_map_manifest" \
    >"$result_dir/logs/onboard.log" 2>&1 </dev/null &
onboard_pid=$!
record_pid onboard "$onboard_pid" started
for _ in $(seq 1 2400); do
  kill -0 "$onboard_pid" 2>/dev/null || exit 1
  test -f "$result_dir/onboard/onboard_ready.json" && break
  sleep 0.25
done
test -f "$result_dir/onboard/onboard_ready.json"

setsid env "${common_env[@]}" \
  bash "$root/scripts/run_t5_step3_live_services.sh" "$result_dir" \
    "${INTERNNAV_T5_STEP3_MODEL_PATH:-/home/rail/ai-stack/models/Step3-VL-10B}" \
    "$step3_venv" \
    >"$result_dir/logs/step3_services.log" 2>&1 </dev/null &
step3_pid=$!
record_pid step3 "$step3_pid" started
for _ in $(seq 1 4800); do
  kill -0 "$step3_pid" 2>/dev/null || exit 1
  test -f "$result_dir/step3/services_ready.json" && break
  sleep 0.25
done
test -f "$result_dir/step3/services_ready.json"

setsid env "${common_env[@]}" \
  INTERNVLA_CLIENT_TCP_BIND_HOST="$lane_ip" \
  INTERNVLA_CLIENT_TCP_PORT="$client_port" \
  INTERNVLA_CLIENT_TCP_EXPECTED_PEER="$isaac_ip" \
  PYTHONPATH="$root:$step3_site${PYTHONPATH:+:$PYTHONPATH}" \
  ros2 run internvla_t4_sensors internvla_t4_client --ros-args \
    --params-file "$params" -r __ns:="$namespace" \
    -r /tf:=tf -r /tf_static:=tf_static \
    -p use_sim_time:=true -p result_dir:="$result_dir/client" \
    -p control_mode:=nav2 -p publish_observation_pose:=false \
    -p navigation_odometry_timeout_sec:=2.0 \
    -p allow_nearest_navigation_odometry:=true \
    -p sensor_future_tolerance_sec:=0.55 \
    >"$result_dir/logs/client.log" 2>&1 </dev/null &
client_pid=$!
record_pid client "$client_pid" started
for _ in $(seq 1 1200); do
  kill -0 "$client_pid" 2>/dev/null || exit 1
  test -n "$(ss -H -ltn "sport = :$client_port")" && \
    test -f "$result_dir/client/client_summary.json" && break
  sleep 0.25
done
test -n "$(ss -H -ltn "sport = :$client_port")"
test -f "$result_dir/client/client_summary.json"
audit_no_internvla "$result_dir/health/no_internvla_ready.json"

python3 - "$result_dir/direct_ready.json" "$onboard_pid" "$step3_pid" "$client_pid" <<'PY'
import json, os, sys, time
from pathlib import Path
payload = {
    "schema_version": 1, "status": "READY", "lane": "b",
    "planner_mode": "direct_high_level", "protocol_version": 1,
    "internvla_model_loaded": False, "internvla_fallback_allowed": False,
    "motion_authority": "nav2_prevalidated_frontier_only",
    "step3_cmd_vel_authority": False, "step3_terminal_stop_authority": False,
    "pids": {"onboard": int(sys.argv[2]), "step3": int(sys.argv[3]), "client": int(sys.argv[4])},
    "stop_request": str(Path(sys.argv[1]).parent / "stop.request"),
    "recorded_unix": time.time(), "host": os.uname().nodename,
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

shutdown_reason=stop_request
while test ! -f "$result_dir/stop.request"; do
  kill -0 "$onboard_pid" 2>/dev/null || { shutdown_reason=onboard_exit; exit 1; }
  kill -0 "$step3_pid" 2>/dev/null || { shutdown_reason=step3_exit; exit 1; }
  kill -0 "$client_pid" 2>/dev/null || { shutdown_reason=client_exit; exit 1; }
  audit_no_internvla "$result_dir/health/no_internvla_live.json"
  sleep 2
done
