#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'usage: %s RESULT_ROOT MODEL_PATH VENV_PATH\n' "${0##*/}" >&2
  exit 64
}

[[ $# -eq 3 ]] || usage
result_root="$1"
model_path="$2"
venv_path="$3"
root="${INTERNNAV_T1_CONTROL_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)}"
frontend_root="$root/frontend"
python="$venv_path/bin/python"
service_config="$root/configs/slow_models/step3_vl_10b_bf16.yaml"
frontend_config="$root/configs/internnav_t5/lane_b_step3.yaml"

test "$(id -un)" = rail
test "${INTERNNAV_T5_RESOURCE_LEASE_ACK:-}" = lane-b
step3_live_advisor="${INTERNNAV_T5_STEP3_LIVE_ADVISOR:-0}"
step3_direct_high_level="${INTERNNAV_T5_STEP3_DIRECT_HIGH_LEVEL:-0}"
case "$step3_live_advisor:$step3_direct_high_level" in
  1:0) planner_mode=bounded_advisor ;;
  0:1) planner_mode=direct_high_level ;;
  *) echo "exactly one Step3 runtime mode is required" >&2; exit 64 ;;
esac
test "${INTERNNAV_T5_LANE:-}" = b
test "${INTERNNAV_T5_LANE_NAMESPACE:-}" = /t5/lane_b
test "${ROS_DOMAIN_ID:-}" = 76
test "${CUDA_VISIBLE_DEVICES:-}" = 0
test "${INTERNNAV_RUNTIME_POLICY:-}" = completion_sim
test "${INTERNNAV_SIMULATION_TARGET:-}" = isaac
test -x "$python"
test -d "$model_path"
test -f "$venv_path/T5_STEP3_RUNTIME_READY.json"
test -f "$service_config"
test -f "$frontend_config"
test -f "$frontend_root/pyproject.toml"
[[ "$result_root" = /home/rail/* ]]
test -d "$result_root"
test ! -e "$result_root/step3/services_ready.json"
for port in 8200 8300; do
  test -z "$(ss -H -lntp | grep -E "127[.]0[.]0[.]1:${port}[[:space:]]" || true)"
done

mkdir -p "$result_root/step3/cameras" "$result_root/logs"
result_root="$(cd -- "$result_root" && pwd -P)"
model_path="$(cd -- "$model_path" && pwd -P)"
service_pid=""
frontend_pid=""
telemetry_pid=""

stop_group() {
  local pid="$1"
  [[ -n "$pid" ]] || return 0
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

cleanup() {
  local rc=$? residual=0
  trap - EXIT INT TERM HUP
  set +e
  stop_group "$frontend_pid" || residual=$((residual + 1))
  stop_group "$telemetry_pid" || residual=$((residual + 1))
  stop_group "$service_pid" || residual=$((residual + 1))
  python3 - "$result_root/step3/services_cleanup.json" "$rc" "$residual" \
    "$service_pid" "$frontend_pid" "$telemetry_pid" <<'PY'
import json, sys, time
from pathlib import Path

output = Path(sys.argv[1])
rc, residual_count = map(int, sys.argv[2:4])
pids = [int(value) for value in sys.argv[4:] if value]
residual = [pid for pid in pids if Path(f"/proc/{pid}").exists()]
payload = {
    "schema_version": 1,
    "status": "PASS" if residual_count == 0 and not residual else "FAIL",
    "command_exit": rc,
    "owned_pids": pids,
    "residual_pids": residual,
    "recorded_unix": time.time(),
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
raise SystemExit(0 if payload["status"] == "PASS" else 75)
PY
  test $? = 0 || rc=75
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

export STEP3_VL_10B_MODEL_PATH="$model_path"
export SLOW_BENCHMARK_RESULTS="$result_root/step3"
export T5_LANE_B_RESULTS="$result_root"
export PYTHONPATH="$frontend_root:$root${PYTHONPATH:+:$PYTHONPATH}"

setsid env CUDA_VISIBLE_DEVICES=0 "$python" -m slow_planner.serve \
  --config "$service_config" >"$result_root/logs/step3_service.log" 2>&1 &
service_pid=$!

deadline=$((SECONDS + ${INTERNNAV_T5_STEP3_READY_TIMEOUT_SEC:-900}))
while (( SECONDS < deadline )); do
  kill -0 "$service_pid" 2>/dev/null || {
    printf 'Step3 service exited before health readiness\n' >&2
    exit 1
  }
  if "$python" - "$result_root/step3/model_health.json" <<'PY' >/dev/null 2>&1
import json, os, sys, time
from pathlib import Path
import zmq

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.setsockopt(zmq.LINGER, 0)
socket.setsockopt(zmq.SNDTIMEO, 1500)
socket.setsockopt(zmq.RCVTIMEO, 1500)
try:
    socket.connect("tcp://127.0.0.1:8200")
    socket.send_json({"type": "health"})
    health = dict(socket.recv_json())
finally:
    socket.close(linger=0)
    context.term()
checks = {
    "ok": health.get("ok") is True,
    "model_variant": health.get("model_variant") == "step3_vl_10b_bf16",
    "precision": health.get("precision_mode") == "bf16",
    "revision": health.get("revision") == "5026053b0c2f5dfaa08fc2d149384162c3c8bca1",
    "transformers": health.get("runtime_transformers_version") == "4.57.6",
    "clean_load": health.get("checkpoint_load_clean") is True,
    "parameter_count": health.get("parameter_count") == 10_171_750_144,
    "all_bf16": set(health.get("parameter_dtype_counts") or {}) == {"torch.bfloat16"},
    "deadline_token_budget": health.get("max_new_tokens") == 96,
    "single_attempt": health.get("max_retries") == 0,
    "private_reasoning_skipped": health.get("skip_private_reasoning") is True,
    "complete_json_stop": health.get("stop_on_complete_json") is True,
    "generation_wall_budget": health.get("generation_wall_budget_s") == 10.5,
    "generation_join_grace": health.get("generation_join_grace_s") == 0.5,
    "raw_text_redacted": health.get("redact_raw_text") is True,
}
if not all(checks.values()):
    raise SystemExit(75)
payload = {
    "schema_version": 1,
    "status": "READY",
    "ready": True,
    "endpoint": "tcp://127.0.0.1:8200",
    "checks": checks,
    **health,
    "recorded_unix": time.time(),
}
output = Path(sys.argv[1])
temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, output)
PY
  then
    break
  fi
  sleep 2
done
test -f "$result_root/step3/model_health.json"
cp "$result_root/step3/model_health.json" "$result_root/step3/health.json"

internvla_pid="${INTERNNAV_T5_INTERNVLA_MODEL_PID:-}"
if test "$planner_mode" = bounded_advisor; then
  [[ "$internvla_pid" =~ ^[1-9][0-9]*$ ]]
  test -d "/proc/$internvla_pid"
else
  test -z "$internvla_pid"
  internvla_pid=0
fi
setsid "$python" -u - "$result_root/step3/gpu.jsonl" \
  "$result_root/step3/gpu.json" "$service_pid" "$internvla_pid" \
  "$planner_mode" "$root" <<'PY' \
  >"$result_root/logs/step3_telemetry.log" 2>&1 &
import json, os, signal, subprocess, sys, time
from pathlib import Path

rows_path, summary_path = map(Path, sys.argv[1:3])
step3_pid, internvla_pid = map(int, sys.argv[3:5])
planner_mode, deployment_root = sys.argv[5], Path(sys.argv[6]).resolve()
direct = planner_mode == "direct_high_level"
stopping = False

def stop(_signum, _frame):
    global stopping
    stopping = True

signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)
peaks = {"gpu_memory_used_mib": 0.0, "unified_memory_used_mib": 0.0,
         "step3_rss_mib": 0.0, "internvla_rss_mib": 0.0,
         "system_swap_used_mib": 0.0, "step3_swap_mib": 0.0,
         "internvla_swap_mib": 0.0}
sample_count = 0
both_models_alive_throughout = True
internvla_absent_throughout = True

def proc_rows(path):
    result = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        key, _, rest = line.partition(":")
        if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree", "VmRSS", "VmSwap"}:
            result[key] = float(rest.strip().split()[0]) / 1024.0
    return result

def oom_kill_count():
    for line in Path("/proc/vmstat").read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition(" ")
        if key == "oom_kill":
            return int(value.strip())
    raise RuntimeError("kernel oom_kill counter is unavailable")

def optional_float(value):
    normalized = value.strip()
    if normalized in {"", "N/A", "[N/A]", "Not Supported", "[Not Supported]"}:
        return None
    return float(normalized)

def internvla_processes():
    matches = []
    deployment = str(deployment_root).encode()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            argv = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if deployment not in argv:
            continue
        if b"internvla_t4_recovery.model_node" in argv or b"run_t4_model_server.sh" in argv:
            matches.append(int(entry.name))
    return sorted(matches)

oom_kill_baseline = oom_kill_count()
system_swap_baseline = None

while not stopping:
    try:
        gpu_raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"], text=True, timeout=3
        ).splitlines()[0].split(",")
        system = proc_rows("/proc/meminfo")
        step3_alive = Path(f"/proc/{step3_pid}/status").is_file()
        internvla_alive = (
            internvla_pid > 0 and Path(f"/proc/{internvla_pid}/status").is_file()
        )
        step3 = proc_rows(f"/proc/{step3_pid}/status") if step3_alive else {}
        internvla = proc_rows(f"/proc/{internvla_pid}/status") if internvla_alive else {}
        system_swap_used = system["SwapTotal"] - system["SwapFree"]
        if system_swap_baseline is None:
            system_swap_baseline = system_swap_used
        gpu_memory_used = optional_float(gpu_raw[4])
        gpu_memory_total = optional_float(gpu_raw[5])
        current_oom_kills = oom_kill_count()
        detected_internvla_pids = internvla_processes()
        row = {
            "timestamp": time.time(), "gpu_index": int(gpu_raw[0].strip()),
            "gpu_name": gpu_raw[1].strip(), "gpu_uuid": gpu_raw[2].strip(),
            "gpu_util_percent": optional_float(gpu_raw[3]),
            "memory_used_mib": gpu_memory_used,
            "memory_total_mib": gpu_memory_total,
            "gpu_memory_used_mib": gpu_memory_used,
            "unified_memory_used_mib": system["MemTotal"] - system["MemAvailable"],
            "unified_memory_total_mib": system["MemTotal"],
            "unified_memory_available_mib": system["MemAvailable"],
            "system_swap_used_mib": system_swap_used,
            "step3_rss_mib": step3.get("VmRSS", 0.0),
            "step3_swap_mib": step3.get("VmSwap", 0.0),
            "internvla_rss_mib": internvla.get("VmRSS", 0.0),
            "internvla_swap_mib": internvla.get("VmSwap", 0.0),
            "step3_process_rss_mib": step3.get("VmRSS", 0.0),
            "step3_process_swap_mib": step3.get("VmSwap", 0.0),
            "step3_alive": step3_alive,
            "internvla_alive": internvla_alive,
            "internvla_detected_pids": detected_internvla_pids,
            "oom_kill_count": current_oom_kills,
        }
        with rows_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        sample_count += 1
        both_models_alive_throughout = both_models_alive_throughout and (
            step3_alive and internvla_alive
        )
        internvla_absent_throughout = (
            internvla_absent_throughout and not detected_internvla_pids
        )
        for key in peaks:
            value = row.get(key)
            if isinstance(value, (int, float)) and value is not None:
                peaks[key] = max(peaks[key], float(value))
        oom_observed = current_oom_kills > oom_kill_baseline
        swap_increased = system_swap_used > system_swap_baseline
        checks = {
            "required_model_processes": (
                step3_alive and internvla_absent_throughout
                if direct else both_models_alive_throughout
            ),
            "step3_process_swap_zero": peaks["step3_swap_mib"] == 0.0,
            "internvla_process_contract": (
                peaks["internvla_swap_mib"] == 0.0
                if not direct else internvla_absent_throughout
            ),
            "system_swap_did_not_increase": not swap_increased,
            "system_oom_kill_did_not_increase": not oom_observed,
        }
        summary = {"schema_version": 1,
                   "status": "PASS" if all(checks.values()) else "FAIL",
                   "sample_count": sample_count, "latest": row, "peaks": peaks,
                   "oom_kill_baseline": oom_kill_baseline,
                   "oom_observed": oom_observed, "checks": checks,
                   "planner_mode": planner_mode,
                   "internvla_model_loaded": not direct,
                   "internvla_absent_throughout": internvla_absent_throughout,
                   "both_models_alive": both_models_alive_throughout}
        temporary = summary_path.with_name(f".{summary_path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, summary_path)
    except BaseException as exc:
        if not stopping:
            print(type(exc).__name__, flush=True)
    time.sleep(1.0)
PY
telemetry_pid=$!

setsid env CUDA_VISIBLE_DEVICES=0 "$python" -m slow_planner_frontend \
  --config "$frontend_config" >"$result_root/logs/frontend.log" 2>&1 &
frontend_pid=$!
deadline=$((SECONDS + 30))
while (( SECONDS < deadline )); do
  kill -0 "$frontend_pid" 2>/dev/null || exit 1
  if "$python" - "$result_root/step3/frontend_state.json" <<'PY' >/dev/null 2>&1
import json, sys, urllib.request
from pathlib import Path
with urllib.request.urlopen("http://127.0.0.1:8300/api/v1/state", timeout=2) as response:
    payload = json.load(response)
if payload.get("lane_id") != "b" or payload.get("readonly") is not True:
    raise SystemExit(75)
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  then
    break
  fi
  sleep 1
done
test -f "$result_root/step3/frontend_state.json"

python3 - "$result_root/step3/services_ready.json" "$service_pid" "$frontend_pid" \
  "$planner_mode" <<'PY'
import json, os, sys, time
from pathlib import Path
output = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "status": "READY",
    "service_pid": int(sys.argv[2]),
    "frontend_pid": int(sys.argv[3]),
    "service_endpoint": "tcp://127.0.0.1:8200",
    "frontend_endpoint": "http://127.0.0.1:8300",
    "planner_mode": sys.argv[4],
    "internvla_model_loaded": sys.argv[4] != "direct_high_level",
    "internvla_fallback_allowed": sys.argv[4] != "direct_high_level",
    "motion_authority": "none",
    "terminal_stop_authority": "none",
    "recorded_unix": time.time(),
}
temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, output)
PY

while :; do
  kill -0 "$service_pid"
  kill -0 "$frontend_pid"
  sleep 1
done
