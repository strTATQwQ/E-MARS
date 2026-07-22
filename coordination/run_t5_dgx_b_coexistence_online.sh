#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'usage: %s CODE_SHA LANE_B_DEPLOYMENT_ROOT RUN_ID results/internnav_t5/dgx-b-coexistence-RUN_ID\n' "${0##*/}" >&2
  exit 64
}

[[ $# -eq 4 ]] || usage
code_sha="$1"
deployment_root="$2"
run_id="$3"
result_relative="$4"
[[ "$code_sha" =~ ^[0-9a-f]{40}$ ]] || usage
[[ "$deployment_root" =~ ^/home/rail/internnav-t1-t2/\.t5-deployments/[a-z0-9._-]+-${code_sha:0:12}-lane-b$ ]] || usage
[[ "$run_id" =~ ^[a-z0-9][a-z0-9._-]{7,63}$ ]] || usage
test "$result_relative" = "results/internnav_t5/dgx-b-coexistence-$run_id" || usage

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
result_dir="$root/$result_relative"
target=rail@10.100.120.116
remote_result="/home/rail/internnav-t1-t2/results/t5_dgx_b_coexistence/$run_id"
step3_model=/home/rail/ai-stack/models/Step3-VL-10B
step3_venv=/home/rail/ai-stack/venvs/step3-vl-10b-tf4.57.6
internvla_python=/home/rail/internnav-t0/venv-model/bin/python
ssh_options=(-T -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 -o StrictHostKeyChecking=yes)

if [[ "${INTERNNAV_T5_INSIDE_DGX_B_COEXISTENCE:-0}" != 1 ]]; then
  test ! -e "$result_dir"
  mkdir -p "$result_dir"
  env INTERNNAV_T5_INSIDE_DGX_B_COEXISTENCE=1 \
    bash "$root/scripts/with_resource_lease.sh" dgx-b \
      --owner codex-00 --task "t5-dgx-b-coexistence:$run_id:$code_sha" \
      --log-dir "$result_dir/lease" --acquire-timeout 30 \
      --cleanup-timeout 300 --kill-wait-timeout 30 -- \
    bash "$root/coordination/run_t5_dgx_b_coexistence_online.sh" \
      "$code_sha" "$deployment_root" "$run_id" "$result_relative"
  exit $?
fi

test -d "$result_dir"
mkdir -p "$result_dir/audits"
remote() {
  local host="$1"
  shift
  ssh "${ssh_options[@]}" "$host" "$@"
}
source "$root/scripts/t5_quarantine_common.sh"
quarantine_marker=/tmp/internnav_dgx.quarantine
quarantine_role=dgx_b
quarantine_tag="coexistence:$run_id"
t5_quarantine_arm "$target" "$quarantine_marker" "$quarantine_role" \
  "$quarantine_tag" "$deployment_root" \
  "$result_dir/audits/quarantine_arm.json"

remote_program="$(cat <<'REMOTE_PROGRAM'
set -euo pipefail
deployment="$1"
code_sha="$2"
result="$3"
step3_model="$4"
step3_venv="$5"
internvla_python="$6"
run_id="$7"
ports=(8200 8300 25138 25239 25240 25241)

test "$(id -un)" = rail
ip -4 -o addr show | grep -Fq ' 10.100.120.116/'
test ! -L "$deployment"
test "$(readlink -f -- "$deployment")" = "$deployment"
test "$(cat "$deployment/T5_DEPLOYMENT_REF")" = "$code_sha"
test -f "$deployment/scripts/run_t4_model_server.sh"
test -x "$deployment/scripts/run_t5_step3_shadow.sh"
test -f "$deployment/scripts/t4_model_health_probe.py"
test -f "$deployment/scripts/probe_t5_internvla_observation_latency.py"
test -f "$deployment/scripts/sample_t5_dgx_coexistence.py"
test -f "$deployment/configs/internnav_t5/primary_completion_sim.yaml"
test -f "$deployment/ros_ws/install/setup.bash"
test -x "$internvla_python"
test -x "$step3_venv/bin/python"
test -f "$step3_venv/T5_STEP3_RUNTIME_READY.json"
test -d "$step3_model"
test ! -e "$result"
for port in "${ports[@]}"; do
  test -z "$(ss -H -lntup | grep -E "[:.]${port}[[:space:]]" || true)"
done
umask 077
mkdir -p "$result/logs" "$result/health"

internvla_runner_pid=""
internvla_pid=""
step3_runner_pid=""
step3_pid=""
step3_frontend_pid=""
preload_sampler_pid=""
internvla_runner_identity=""
internvla_identity=""
step3_runner_identity=""
step3_identity=""
step3_frontend_identity=""
preload_sampler_identity=""
cleanup_done=0

capture_identity() {
  local pid="$1" scope="${2:-member}" identity="" attempt
  case "$scope" in leader|member) ;; *) return 64 ;; esac
  for attempt in $(seq 1 200); do
    if identity="$(python3 - "$pid" "$deployment" "$scope" <<'PY'
import os
import sys
from pathlib import Path

pid = int(sys.argv[1])
run_root = sys.argv[2]
scope = sys.argv[3]
entry = Path(f"/proc/{pid}")
if entry.stat().st_uid != os.getuid():
    raise SystemExit("owned process UID differs")
stat = (entry / "stat").read_text(encoding="utf-8")
fields = stat[stat.rfind(")") + 2:].split()
pgid, sid, starttime = int(fields[2]), int(fields[3]), int(fields[19])
argv = (entry / "cmdline").read_bytes()
if run_root.encode() not in argv.split(b"\0") and (run_root + "/").encode() not in argv:
    raise SystemExit("owned process argv is outside the deployment root")
if scope == "leader" and (pgid != pid or sid != pid):
    raise SystemExit("setsid process has not become its own group/session leader")
print(f"{starttime}:{pgid}:{sid}")
PY
)"; then
      printf '%s\n' "$identity"
      return 0
    fi
    kill -0 "$pid" 2>/dev/null || return 75
    sleep 0.05
  done
  return 75
}

identity_matches() {
  local pid="$1" identity="$2"
  test -n "$identity" || return 1
  python3 - "$pid" "$identity" "$deployment" <<'PY' >/dev/null 2>&1
import os
import sys
from pathlib import Path

pid = int(sys.argv[1])
expected_start, expected_pgid, expected_sid = map(int, sys.argv[2].split(":"))
run_root = sys.argv[3]
try:
    entry = Path(f"/proc/{pid}")
    if entry.stat().st_uid != os.getuid():
        raise RuntimeError
    stat = (entry / "stat").read_text(encoding="utf-8")
    fields = stat[stat.rfind(")") + 2:].split()
    current = (int(fields[19]), int(fields[2]), int(fields[3]))
    argv = (entry / "cmdline").read_bytes()
except (OSError, ValueError, RuntimeError):
    raise SystemExit(1)
root_bound = run_root.encode() in argv.split(b"\0") or (run_root + "/").encode() in argv
raise SystemExit(0 if current == (expected_start, expected_pgid, expected_sid) and root_bound else 1)
PY
}

group_has_runnable_member() {
  local pgid="$1"
  ps -eo pgid=,stat= | awk -v wanted="$pgid" '
    $1 == wanted && $2 !~ /^Z/ { found=1 }
    END { exit(found ? 0 : 1) }
  '
}

stop_group() {
  local pgid="$1" identity="$2" first_signal="${3:-TERM}"
  [[ -n "$pgid" ]] || return 0
  group_has_runnable_member "$pgid" || return 0
  identity_matches "$pgid" "$identity" || return 75
  kill -"$first_signal" -- "-$pgid" 2>/dev/null || true
  for _ in $(seq 1 200); do
    group_has_runnable_member "$pgid" || return 0
    sleep 0.1
  done
  identity_matches "$pgid" "$identity" || return 75
  kill -TERM -- "-$pgid" 2>/dev/null || true
  for _ in $(seq 1 100); do
    group_has_runnable_member "$pgid" || return 0
    sleep 0.1
  done
  identity_matches "$pgid" "$identity" || return 75
  kill -KILL -- "-$pgid" 2>/dev/null || true
  for _ in $(seq 1 50); do
    group_has_runnable_member "$pgid" || return 0
    sleep 0.1
  done
  return 75
}

cleanup_runtime() {
  local command_exit="$1" cleanup_rc=0
  set +e
  if test -n "$preload_sampler_pid"; then
    touch "$result/preload_memory.stop" 2>/dev/null
    for _ in $(seq 1 300); do
      identity_matches "$preload_sampler_pid" "$preload_sampler_identity" || break
      sleep 0.1
    done
    stop_group "$preload_sampler_pid" "$preload_sampler_identity" TERM || cleanup_rc=75
    wait "$preload_sampler_pid" 2>/dev/null
  fi
  if test -n "$step3_runner_pid"; then
    touch "$result/step3/stop.request" 2>/dev/null
    for _ in $(seq 1 900); do
      identity_matches "$step3_runner_pid" "$step3_runner_identity" || break
      sleep 0.2
    done
    if identity_matches "$step3_runner_pid" "$step3_runner_identity"; then
      kill -TERM "$step3_runner_pid" 2>/dev/null
      for _ in $(seq 1 150); do
        identity_matches "$step3_runner_pid" "$step3_runner_identity" || break
        sleep 0.2
      done
    fi
    if identity_matches "$step3_runner_pid" "$step3_runner_identity"; then
      kill -KILL "$step3_runner_pid" 2>/dev/null
    fi
    wait "$step3_runner_pid" 2>/dev/null
  fi
  stop_group "$step3_frontend_pid" "$step3_frontend_identity" TERM || cleanup_rc=75
  stop_group "$step3_pid" "$step3_identity" TERM || cleanup_rc=75
  stop_group "$internvla_runner_pid" "$internvla_runner_identity" INT || cleanup_rc=75
  if test -n "$internvla_runner_pid"; then
    wait "$internvla_runner_pid" 2>/dev/null
  fi

  local internvla_group_alive=false step3_group_alive=false frontend_group_alive=false
  local preload_group_alive=false
  test -z "$internvla_runner_pid" || ! group_has_runnable_member "$internvla_runner_pid" || internvla_group_alive=true
  test -z "$step3_pid" || ! group_has_runnable_member "$step3_pid" || step3_group_alive=true
  test -z "$step3_frontend_pid" || ! group_has_runnable_member "$step3_frontend_pid" || frontend_group_alive=true
  test -z "$preload_sampler_pid" || ! group_has_runnable_member "$preload_sampler_pid" || preload_group_alive=true
  local occupied_ports=""
  for port in "${ports[@]}"; do
    if test -n "$(ss -H -lntup | grep -E "[:.]${port}[[:space:]]" || true)"; then
      occupied_ports="${occupied_ports}${occupied_ports:+,}${port}"
    fi
  done
  python3 - "$result/cleanup.json" "$command_exit" "$internvla_runner_pid" \
    "$internvla_pid" "$step3_runner_pid" "$step3_pid" "$step3_frontend_pid" \
    "$preload_sampler_pid" "$internvla_runner_identity" "$internvla_identity" \
    "$step3_runner_identity" "$step3_identity" "$step3_frontend_identity" \
    "$preload_sampler_identity" "$deployment" \
    "$internvla_group_alive" "$step3_group_alive" "$frontend_group_alive" \
    "$preload_group_alive" "$occupied_ports" "$result/step3/cleanup.json" <<'PY'
import json
import os
import sys
import time
from pathlib import Path

(
    output,
    command_exit,
    internvla_runner_pid,
    internvla_pid,
    step3_runner_pid,
    step3_pid,
    step3_frontend_pid,
    preload_sampler_pid,
    internvla_runner_identity,
    internvla_identity,
    step3_runner_identity,
    step3_identity,
    step3_frontend_identity,
    preload_sampler_identity,
    deployment,
    internvla_group_alive,
    step3_group_alive,
    frontend_group_alive,
    preload_group_alive,
    occupied_ports,
    step3_cleanup_path,
) = sys.argv[1:]
named = {
    "internvla_runner": (internvla_runner_pid, internvla_runner_identity),
    "internvla_service": (internvla_pid, internvla_identity),
    "step3_runner": (step3_runner_pid, step3_runner_identity),
    "step3_service": (step3_pid, step3_identity),
    "step3_frontend": (step3_frontend_pid, step3_frontend_identity),
    "preload_sampler": (preload_sampler_pid, preload_sampler_identity),
}
owned = {name: int(value[0]) for name, value in named.items() if value[0]}

def same_identity(pid_text, identity):
    if not pid_text or not identity:
        return False
    pid = int(pid_text)
    try:
        entry = Path(f"/proc/{pid}")
        if entry.stat().st_uid != os.getuid():
            return False
        stat = (entry / "stat").read_text(encoding="utf-8")
        fields = stat[stat.rfind(")") + 2:].split()
        current = (int(fields[19]), int(fields[2]), int(fields[3]))
        expected = tuple(map(int, identity.split(":")))
        argv = (entry / "cmdline").read_bytes()
    except (OSError, ValueError):
        return False
    root_bound = deployment.encode() in argv.split(b"\0") or (deployment + "/").encode() in argv
    return current == expected and root_bound

residual = [
    int(pid) for pid, identity in named.values() if same_identity(pid, identity)
]
groups = {
    "internvla": internvla_group_alive == "true",
    "step3": step3_group_alive == "true",
    "step3_frontend": frontend_group_alive == "true",
    "preload_sampler": preload_group_alive == "true",
}
ports = [int(value) for value in occupied_ports.split(",") if value]
step3_cleanup = None
path = Path(step3_cleanup_path)
if path.is_file():
    step3_cleanup = json.loads(path.read_text(encoding="utf-8"))
step3_cleanup_valid = not step3_runner_pid or (
    isinstance(step3_cleanup, dict)
    and step3_cleanup.get("status") == "PASS"
    and step3_cleanup.get("residual_pids") == []
)
checks = {
    "owned_pids_absent": not residual,
    "owned_process_groups_absent": not any(groups.values()),
    "related_ports_free": not ports,
    "step3_runner_cleanup": step3_cleanup_valid,
}
payload = {
    "schema_version": 1,
    "status": "PASS" if all(checks.values()) else "FAIL",
    "command_exit": int(command_exit),
    "owned_pids": owned,
    "residual_pids": residual,
    "residual_process_groups": [name for name, alive in groups.items() if alive],
    "related_ports": [8200, 8300, 25138, 25239, 25240, 25241],
    "occupied_ports": ports,
    "checks": checks,
    "recorded_unix": time.time(),
}
Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
raise SystemExit(0 if payload["status"] == "PASS" else 75)
PY
  test $? = 0 || cleanup_rc=75
  set -e
  return "$cleanup_rc"
}

on_exit() {
  local rc=$?
  trap - EXIT INT TERM HUP
  if test "$cleanup_done" = 0; then
    cleanup_runtime "$rc" || rc=75
  fi
  exit "$rc"
}
trap on_exit EXIT
trap 'exit 130' INT TERM HUP

setsid "$step3_venv/bin/python" "$deployment/scripts/sample_t5_dgx_coexistence.py" \
  --preload-stop-file "$result/preload_memory.stop" \
  --duration-sec 2400 --interval-sec 1 \
  --samples "$result/preload_memory_samples.jsonl" \
  --summary "$result/preload_memory_summary.json" \
  >"$result/logs/preload_memory.log" 2>&1 &
preload_sampler_pid=$!
preload_sampler_identity="$(capture_identity "$preload_sampler_pid" leader)"

setsid env \
  -u INTERNNAV_RUNTIME_POLICY \
  -u INTERNNAV_SIMULATION_TARGET \
  -u INTERNNAV_T5_LANE \
  -u INTERNVLA_MODEL_CONFIG \
  -u INTERNVLA_T4_VARIANT_CONFIG \
  CUDA_VISIBLE_DEVICES=0 \
  ROS_DOMAIN_ID=76 \
  ROS_LOCALHOST_ONLY=1 \
  ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
  ROS_STATIC_PEERS= \
  INTERNNAV_T5_RESOURCE_LEASE_ACK=dgx-b \
  INTERNNAV_T1_CONTROL_ROOT="$deployment" \
  INTERNVLA_ROS_WS="$deployment/ros_ws" \
  INTERNVLA_MODEL_PYTHON="$internvla_python" \
  INTERNVLA_MODEL_RESULT_DIR="$result/model" \
  INTERNVLA_BACKEND=real \
  INTERNVLA_PRELOAD_MODEL=1 \
  INTERNVLA_T4_FUNCTIONAL_MODEL=0 \
  INTERNVLA_T4_HISTORY_MODE=on \
  bash "$deployment/scripts/run_t4_model_server.sh" --ros-args \
    --params-file "$deployment/configs/internnav_t5/primary_completion_sim.yaml" \
    -r __ns:=/t5/lane_b/coexistence -p use_sim_time:=false \
    >"$result/logs/internvla_runner.log" 2>&1 &
internvla_runner_pid=$!
internvla_runner_identity="$(capture_identity "$internvla_runner_pid" leader)"

set +u
source /opt/ros/jazzy/setup.bash
source "$deployment/ros_ws/install/setup.bash"
set -u
export ROS_DOMAIN_ID=76
export ROS_LOCALHOST_ONLY=1
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ROS_STATIC_PEERS=
deadline=$((SECONDS + 1200))
while (( SECONDS < deadline )); do
  kill -0 "$internvla_runner_pid" 2>/dev/null || break
  identity_matches "$preload_sampler_pid" "$preload_sampler_identity" || break
  if test -f "$result/model/model_weight_audit.json"; then
    rm -f -- "$result/health/model_health.candidate.json"
    if python3 "$deployment/scripts/t4_model_health_probe.py" \
      --output "$result/health/model_health.candidate.json" --timeout-sec 3 \
      >"$result/logs/internvla_health_probe.log" 2>&1; then
      mv -- "$result/health/model_health.candidate.json" "$result/health/model_health.json"
      break
    fi
  fi
  sleep 2
done
kill -0 "$internvla_runner_pid"
test -f "$result/model/model_weight_audit.json"
test -f "$result/health/model_health.json"

internvla_pid="$(python3 - "$internvla_runner_pid" <<'PY'
import os
import sys
from pathlib import Path

pgid = int(sys.argv[1])
matches = []
for entry in Path("/proc").iterdir():
    if not entry.name.isdigit():
        continue
    pid = int(entry.name)
    try:
        if os.getpgid(pid) != pgid:
            continue
        argv = (entry / "cmdline").read_bytes().split(b"\0")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    decoded = [value.decode("utf-8", "surrogateescape") for value in argv if value]
    if any(
        decoded[index:index + 2] == ["-m", "internvla_t4_recovery.model_node"]
        for index in range(len(decoded) - 1)
    ):
        matches.append(pid)
if len(matches) != 1:
    raise SystemExit(f"expected one real InternVLA Python PID in PGID {pgid}, got {matches}")
print(matches[0])
PY
)"
[[ "$internvla_pid" =~ ^[1-9][0-9]*$ ]]
kill -0 "$internvla_pid"
internvla_identity="$(capture_identity "$internvla_pid" member)"

python3 "$deployment/scripts/probe_t5_internvla_observation_latency.py" \
  --phase internvla_only --initialize --count 5 \
  --episode-id "b::coexistence-probe::$run_id::internvla-only" \
  --output "$result/internvla_only_latency.json" \
  >"$result/logs/internvla_only_latency.log" 2>&1
identity_matches "$internvla_runner_pid" "$internvla_runner_identity"
identity_matches "$internvla_pid" "$internvla_identity"

env INTERNNAV_T5_RESOURCE_LEASE_ACK=dgx-b INTERNNAV_T1_CONTROL_ROOT="$deployment" \
  bash "$deployment/scripts/run_t5_step3_shadow.sh" shadow \
    "$result/step3" "$step3_model" "$step3_venv" \
    >"$result/logs/step3_runner.log" 2>&1 &
step3_runner_pid=$!
step3_runner_identity="$(capture_identity "$step3_runner_pid" member)"
deadline=$((SECONDS + 900))
while (( SECONDS < deadline )); do
  kill -0 "$step3_runner_pid" 2>/dev/null || break
  identity_matches "$internvla_runner_pid" "$internvla_runner_identity" || break
  identity_matches "$internvla_pid" "$internvla_identity" || break
  identity_matches "$preload_sampler_pid" "$preload_sampler_identity" || break
  test -f "$result/step3/shadow_ready.json" && break
  sleep 2
done
kill -0 "$step3_runner_pid"
test -f "$result/step3/shadow_ready.json"

read -r step3_pid step3_frontend_pid < <(python3 - \
  "$result/step3/shadow_ready.json" "$result/step3/step3/health.json" \
  "$step3_runner_pid" <<'PY'
import json
import os
import sys
from pathlib import Path

ready = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
health = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
runner = int(sys.argv[3])
assert ready.get("status") == "READY" and ready.get("mode") == "shadow"
assert ready.get("motion_authority") == "none"
assert ready.get("terminal_stop_authority") == "none"
assert health.get("status") == "READY" and all(health.get("checks", {}).values())
raw = health.get("health", {})
assert raw.get("precision_mode") == "bf16"
assert raw.get("checkpoint_load_clean") is True
assert raw.get("parameter_count") == 10_171_750_144
assert set(raw.get("parameter_dtype_counts") or {}) == {"torch.bfloat16"}

def ancestors(pid):
    values = []
    while pid > 1:
        values.append(pid)
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        pid = int(text.rpartition(")")[2].strip().split()[1])
    return values

def validate(pid, module):
    assert pid > 1 and os.getpgid(pid) == pid
    argv = [
        value.decode("utf-8", "surrogateescape")
        for value in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        if value
    ]
    assert any(argv[index:index + 2] == ["-m", module] for index in range(len(argv) - 1))
    assert runner in ancestors(pid)

service = int(ready["service_pid"])
frontend = int(ready["frontend_pid"])
validate(service, "slow_planner.serve")
validate(frontend, "slow_planner_frontend")
print(service, frontend)
PY
)
[[ "$step3_pid" =~ ^[1-9][0-9]*$ ]]
[[ "$step3_frontend_pid" =~ ^[1-9][0-9]*$ ]]
test "$internvla_pid" != "$step3_pid"
kill -0 "$internvla_pid"
kill -0 "$step3_pid"
step3_identity="$(capture_identity "$step3_pid" leader)"
step3_frontend_identity="$(capture_identity "$step3_frontend_pid" leader)"

python3 "$deployment/scripts/probe_t5_internvla_observation_latency.py" \
  --phase co_loaded --reset --count 5 \
  --expected-reset-generation 0 --reset-barrier-sequence 4 \
  --episode-id "b::coexistence-probe::$run_id::co-loaded" \
  --output "$result/co_loaded_latency.json" \
  >"$result/logs/co_loaded_latency.log" 2>&1
identity_matches "$internvla_runner_pid" "$internvla_runner_identity"
identity_matches "$internvla_pid" "$internvla_identity"
identity_matches "$step3_pid" "$step3_identity"

python3 - "$result/internvla_only_latency.json" \
  "$result/co_loaded_latency.json" "$result/latency_comparison.json" <<'PY'
import json
import sys
import time
from pathlib import Path

baseline_path, co_loaded_path, output_path = map(Path, sys.argv[1:])
baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
co_loaded = json.loads(co_loaded_path.read_text(encoding="utf-8"))
assert baseline.get("status") == "PASS" and baseline.get("sample_count") == 5
assert co_loaded.get("status") == "PASS" and co_loaded.get("sample_count") == 5
assert baseline.get("probe_mode") == "observation_only_no_navigation_publishers"
assert co_loaded.get("probe_mode") == "observation_only_no_navigation_publishers"
for value in (baseline, co_loaded):
    assert not any(value.get("forbidden_output_publisher_counts_before", {}).values())
    assert not any(value.get("forbidden_output_publisher_counts_after", {}).values())
baseline_latency = baseline["inference_latency_sec"]
co_loaded_latency = co_loaded["inference_latency_sec"]
baseline_p95 = float(baseline_latency["p95"])
co_loaded_p95 = float(co_loaded_latency["p95"])
if baseline_p95 <= 0.0:
    raise RuntimeError("baseline p95 must be positive")
p50_degradation = (
    float(co_loaded_latency["p50"]) / float(baseline_latency["p50"]) - 1.0
) * 100.0
p95_degradation = (co_loaded_p95 / baseline_p95 - 1.0) * 100.0
checks = {
    "exactly_five_internvla_only_samples": baseline.get("sample_count") == 5,
    "exactly_five_co_loaded_samples": co_loaded.get("sample_count") == 5,
    "no_navigation_action_stop_publishers": True,
    "internvla_p95_degradation_lte_20_percent": p95_degradation <= 20.0,
}
payload = {
    "schema_version": 1,
    "status": "PASS" if all(checks.values()) else "FAIL",
    "checks": checks,
    "internvla_only_inference_latency_sec": baseline_latency,
    "co_loaded_inference_latency_sec": co_loaded_latency,
    "p50_degradation_percent": p50_degradation,
    "p95_degradation_percent": p95_degradation,
    "recorded_unix": time.time(),
}
output_path.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
raise SystemExit(0 if payload["status"] == "PASS" else 75)
PY

touch "$result/preload_memory.stop"
for _ in $(seq 1 300); do
  identity_matches "$preload_sampler_pid" "$preload_sampler_identity" || break
  sleep 0.1
done
if identity_matches "$preload_sampler_pid" "$preload_sampler_identity"; then
  echo "preload memory sampler did not stop at the two-model READY boundary" >&2
  exit 75
fi
wait "$preload_sampler_pid"
python3 - "$result/preload_memory_summary.json" <<'PY'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
required = {
    "minimum_samples",
    "explicit_stop_observed",
    "system_memory_observed",
    "system_swap_observed",
    "system_swap_did_not_increase",
    "oom_kill_counter_observed",
    "system_oom_kill_did_not_increase",
}
checks = value.get("checks", {})
assert value.get("status") == "PASS"
assert all(checks.get(name) is True for name in required)
PY

python3 - "$result/model/model_weight_audit.json" "$result/health/model_health.json" \
  "$result/ready.json" "$internvla_runner_pid" "$internvla_pid" \
  "$step3_runner_pid" "$step3_pid" "$step3_frontend_pid" "$run_id" <<'PY'
import json
import sys
import time
from pathlib import Path

audit = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
health = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
assert audit.get("status") == "PASS" and audit.get("backend") == "real"
assert audit.get("meta_parameter_count") == 0 and audit.get("meta_buffer_count") == 0
assert health.get("status") == "PASS" and health.get("step_action_ready") is True
payload = {
    "schema_version": 1,
    "status": "READY",
    "run_id": sys.argv[9],
    "services": {
        "internvla": {"runner_pid": int(sys.argv[4]), "service_pid": int(sys.argv[5]), "backend": "real"},
        "step3": {"runner_pid": int(sys.argv[6]), "service_pid": int(sys.argv[7]), "frontend_pid": int(sys.argv[8]), "precision": "bf16"},
    },
    "gpu_index": 0,
    "resource_lease": "dgx-b",
    "motion_authority": "none",
    "isaac_started": False,
    "control_components_started": False,
    "recorded_unix": time.time(),
}
Path(sys.argv[3]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

"$step3_venv/bin/python" "$deployment/scripts/sample_t5_dgx_coexistence.py" \
  --internvla-pid "$internvla_pid" --step3-pid "$step3_pid" --gpu-index 0 \
  --duration-sec 60 --interval-sec 1 \
  --samples "$result/memory_samples.jsonl" \
  --summary "$result/memory_summary.json"

python3 - "$result/memory_summary.json" "$result/preload_memory_summary.json" \
  "$internvla_pid" "$step3_pid" <<'PY'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
preload = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
assert value.get("status") == "PASS"
assert preload.get("status") == "PASS" and all(preload.get("checks", {}).values())
assert value.get("pids") == {"internvla": int(sys.argv[3]), "step3": int(sys.argv[4])}
assert value.get("motion_authority") == "none"
assert value.get("gpu_memory_accounting_status") in {"observed", "not_applicable_unified"}
required = {
    "internvla_alive_throughout",
    "step3_alive_throughout",
    "internvla_process_swap_zero",
    "step3_process_swap_zero",
    "system_swap_did_not_increase",
    "system_oom_kill_did_not_increase",
}
checks = value.get("checks", {})
assert all(checks.get(name) is True for name in required)
PY

cleanup_runtime 0 || exit 75
cleanup_done=1
python3 - "$result/memory_summary.json" "$result/cleanup.json" "$result/ready.json" <<'PY'
import json
import sys
from pathlib import Path

memory, cleanup, ready = [json.loads(Path(value).read_text(encoding="utf-8")) for value in sys.argv[1:]]
assert memory.get("status") == "PASS"
assert cleanup.get("status") == "PASS"
assert cleanup.get("residual_pids") == []
assert cleanup.get("residual_process_groups") == []
assert cleanup.get("occupied_ports") == []
assert ready.get("motion_authority") == "none"
assert ready.get("isaac_started") is False
assert ready.get("control_components_started") is False
PY
REMOTE_PROGRAM
)"

set +e
ssh "${ssh_options[@]}" "$target" bash -s -- \
  "$deployment_root" "$code_sha" "$remote_result" "$step3_model" \
  "$step3_venv" "$internvla_python" "$run_id" <<<"$remote_program"
remote_rc=$?
set -e

mkdir -p "$result_dir/remote"
set +e
scp -q -r -o BatchMode=yes -o StrictHostKeyChecking=yes \
  "$target:$remote_result/model" "$target:$remote_result/health" \
  "$target:$remote_result/step3" "$target:$remote_result/logs" \
  "$target:$remote_result/ready.json" \
  "$target:$remote_result/preload_memory_samples.jsonl" \
  "$target:$remote_result/preload_memory_summary.json" \
  "$target:$remote_result/internvla_only_latency.json" \
  "$target:$remote_result/co_loaded_latency.json" \
  "$target:$remote_result/latency_comparison.json" \
  "$target:$remote_result/memory_samples.jsonl" \
  "$target:$remote_result/memory_summary.json" \
  "$target:$remote_result/cleanup.json" "$result_dir/remote/"
collect_rc=$?
set -e

quarantine_clear_rc=125
if python3 - "$result_dir/remote/cleanup.json" <<'PY' >/dev/null 2>&1
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
value = json.loads(path.read_text(encoding="utf-8"))
checks = value.get("checks", {})
raise SystemExit(0 if value.get("status") == "PASS" and checks
                 and all(item is True for item in checks.values()) else 1)
PY
then
  set +e
  t5_quarantine_owned_clear "$target" "$quarantine_marker" \
    "$quarantine_role" "$quarantine_tag" "$deployment_root" \
    "$result_dir/remote/cleanup.json" \
    "$result_dir/audits/quarantine_clear.json"
  quarantine_clear_rc=$?
  set -e
fi

if test "$remote_rc" != 0 || test "$collect_rc" != 0 || \
    test "$quarantine_clear_rc" != 0; then
  python3 - "$result_dir" "$code_sha" "$deployment_root" "$remote_result" \
    "$remote_rc" "$collect_rc" "$quarantine_clear_rc" <<'PY'
import json
import sys
import time
from pathlib import Path

root = Path(sys.argv[1])
code_sha, deployment, remote_result = sys.argv[2:5]
remote_rc, collect_rc, quarantine_clear_rc = map(int, sys.argv[5:8])

def optional(relative):
    path = root / relative
    if not path.is_file() or path.is_symlink():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None

memory = optional("remote/memory_summary.json")
preload = optional("remote/preload_memory_summary.json")
cleanup = optional("remote/cleanup.json")
failure_stage = (
    "remote_coexistence_runtime" if remote_rc != 0
    else "evidence_collection" if collect_rc != 0
    else "resource_cleanup"
)
payload = {
    "schema_version": 1,
    "status": "FAIL",
    "failure_stage": failure_stage,
    "code_ref_sha": code_sha,
    "deployment_root": deployment,
    "remote_result_root": remote_result,
    "remote_exit_code": remote_rc,
    "partial_evidence_collect_exit_code": collect_rc,
    "quarantine_clear_exit_code": quarantine_clear_rc,
    "cleanup_status": cleanup.get("status") if isinstance(cleanup, dict) else None,
    "memory_status": memory.get("status") if isinstance(memory, dict) else None,
    "preload_memory_status": preload.get("status") if isinstance(preload, dict) else None,
    "process_rss_peak_kib": (
        memory.get("process_rss_peak_kib") if isinstance(memory, dict) else None
    ),
    "system_mem_available_min_kib": (
        memory.get("system_mem_available_min_kib") if isinstance(memory, dict) else None
    ),
    "preload_system_mem_available_min_kib": (
        preload.get("system_mem_available_min_kib")
        if isinstance(preload, dict) else None
    ),
    "recorded_unix": time.time(),
}
(root / "dgx_b_coexistence_summary.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(payload, sort_keys=True))
PY
  exit 75
fi

python3 - "$result_dir" "$code_sha" "$deployment_root" "$remote_result" <<'PY'
import json
import sys
import time
from pathlib import Path

root = Path(sys.argv[1])
code_sha, deployment, remote_result = sys.argv[2:]

def load(relative):
    return json.loads((root / relative).read_text(encoding="utf-8"))

ready = load("remote/ready.json")
memory = load("remote/memory_summary.json")
preload = load("remote/preload_memory_summary.json")
latency = load("remote/latency_comparison.json")
cleanup = load("remote/cleanup.json")
quarantine_clear = load("audits/quarantine_clear.json")
model_audit = load("remote/model/model_weight_audit.json")
model_health = load("remote/health/model_health.json")
step3_health = load("remote/step3/step3/health.json")
step3_cleanup = load("remote/step3/cleanup.json")
services = ready.get("services", {})
pids = memory.get("pids", {})
memory_checks = memory.get("checks", {})
raw_step3_health = step3_health.get("health", {})
checks = {
    "exact_deployment_ref": len(code_sha) == 40 and deployment.endswith(f"-{code_sha[:12]}-lane-b"),
    "real_internvla_ready": model_audit.get("status") == "PASS"
    and model_audit.get("backend") == "real"
    and model_health.get("status") == "PASS"
    and model_health.get("step_action_ready") is True,
    "step3_bf16_ready": step3_health.get("status") == "READY"
    and raw_step3_health.get("precision_mode") == "bf16"
    and raw_step3_health.get("checkpoint_load_clean") is True
    and set(raw_step3_health.get("parameter_dtype_counts") or {}) == {"torch.bfloat16"},
    "two_real_service_pids": isinstance(pids.get("internvla"), int)
    and isinstance(pids.get("step3"), int)
    and pids.get("internvla") != pids.get("step3")
    and services.get("internvla", {}).get("service_pid") == pids.get("internvla")
    and services.get("step3", {}).get("service_pid") == pids.get("step3"),
    "memory_sampling_pass": memory.get("status") == "PASS"
    and memory.get("sample_count", 0) >= 60
    and memory.get("gpu_memory_accounting_status") in {"observed", "not_applicable_unified"}
    and all(memory_checks.get(name) is True for name in (
        "internvla_alive_throughout",
        "step3_alive_throughout",
        "internvla_process_swap_zero",
        "step3_process_swap_zero",
        "system_swap_did_not_increase",
        "system_oom_kill_did_not_increase",
    )),
    "preload_memory_sampling_pass": preload.get("status") == "PASS"
    and preload.get("sample_count", 0) >= 2
    and all(preload.get("checks", {}).get(name) is True for name in (
        "minimum_samples",
        "explicit_stop_observed",
        "system_memory_observed",
        "system_swap_observed",
        "system_swap_did_not_increase",
        "oom_kill_counter_observed",
        "system_oom_kill_did_not_increase",
    )),
    "observation_only_latency_gate": latency.get("status") == "PASS"
    and latency.get("checks", {}).get("exactly_five_internvla_only_samples") is True
    and latency.get("checks", {}).get("exactly_five_co_loaded_samples") is True
    and latency.get("checks", {}).get("no_navigation_action_stop_publishers") is True
    and latency.get("checks", {}).get(
        "internvla_p95_degradation_lte_20_percent"
    ) is True,
    "no_motion_stack": ready.get("motion_authority") == "none"
    and ready.get("resource_lease") == "dgx-b"
    and ready.get("isaac_started") is False
    and ready.get("control_components_started") is False,
    "owned_cleanup": cleanup.get("status") == "PASS"
    and cleanup.get("residual_pids") == []
    and cleanup.get("residual_process_groups") == []
    and cleanup.get("occupied_ports") == []
    and step3_cleanup.get("status") == "PASS"
    and step3_cleanup.get("residual_pids") == []
    and quarantine_clear.get("status") == "PASS"
    and quarantine_clear.get("marker_absent") is True,
}
payload = {
    "schema_version": 1,
    "status": "PASS" if all(checks.values()) else "FAIL",
    "code_ref_sha": code_sha,
    "deployment_root": deployment,
    "remote_result_root": remote_result,
    "checks": checks,
    "pids": pids,
    "sample_count": memory.get("sample_count"),
    "gpu_memory_accounting_status": memory.get("gpu_memory_accounting_status"),
    "process_rss_peak_kib": memory.get("process_rss_peak_kib"),
    "system_mem_available_min_kib": memory.get("system_mem_available_min_kib"),
    "preload_system_mem_available_min_kib": preload.get(
        "system_mem_available_min_kib"
    ),
    "internvla_only_inference_latency_sec": latency.get(
        "internvla_only_inference_latency_sec"
    ),
    "co_loaded_inference_latency_sec": latency.get(
        "co_loaded_inference_latency_sec"
    ),
    "p50_degradation_percent": latency.get("p50_degradation_percent"),
    "p95_degradation_percent": latency.get("p95_degradation_percent"),
    "recorded_unix": time.time(),
}
(root / "dgx_b_coexistence_summary.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(payload, sort_keys=True))
raise SystemExit(0 if payload["status"] == "PASS" else 75)
PY
