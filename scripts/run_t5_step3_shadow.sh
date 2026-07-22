#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'usage: %s <health-only|shadow> RESULT_DIR MODEL_PATH VENV_PATH\n' \
    "${0##*/}" >&2
  exit 64
}

[[ $# -eq 4 ]] || usage
mode="$1"
result_dir="$2"
model_path="$3"
venv_path="$4"
case "$mode" in health-only|shadow) ;; *) usage ;; esac

root="${INTERNNAV_T1_CONTROL_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)}"
frontend_root="$root/frontend"
python="$venv_path/bin/python"
service_config="$root/configs/slow_models/step3_vl_10b_bf16.yaml"
frontend_config="$root/configs/internnav_t5/lane_b_step3.yaml"

resource_lease_ack="${INTERNNAV_T5_RESOURCE_LEASE_ACK:-}"
case "$resource_lease_ack" in
  dgx-a|lane-a) expected_user=railgun; expected_home=/home/railgun ;;
  dgx-b|lane-b) expected_user=rail; expected_home=/home/rail ;;
  *) exit 73 ;;
esac
test "$(id -un)" = "$expected_user"
test -x "$python"
test -d "$model_path"
test -f "$venv_path/T5_STEP3_RUNTIME_READY.json"
test -f "$service_config"
test -f "$frontend_config"
test -f "$frontend_root/pyproject.toml"
[[ "$result_dir" = "$expected_home"/* ]]
test ! -e "$result_dir"
for port in 8200 8300; do
  test -z "$(ss -H -lntp | grep -E "[:.]${port}[[:space:]]" || true)"
done

mkdir -p "$result_dir/logs" "$result_dir/step3/cameras"
result_dir="$(cd -- "$result_dir" && pwd -P)"
model_path="$(cd -- "$model_path" && pwd -P)"
service_pid=""
frontend_pid=""

stop_group() {
  local pid="$1"
  [[ -n "$pid" ]] || return 0
  kill -TERM -- "-$pid" 2>/dev/null || true
  for _ in $(seq 1 100); do
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
  local rc=$?
  trap - EXIT INT TERM HUP
  stop_group "$frontend_pid" || rc=75
  stop_group "$service_pid" || rc=75
  python3 - "$result_dir/cleanup.json" "$rc" "$service_pid" "$frontend_pid" <<'PY'
import json, os, sys, time
from pathlib import Path

output = Path(sys.argv[1])
rc = int(sys.argv[2])
pids = [int(value) for value in sys.argv[3:] if value]
residual = [pid for pid in pids if Path(f"/proc/{pid}").exists()]
payload = {
    "schema_version": 1,
    "status": "PASS" if not residual else "FAIL",
    "command_exit": rc,
    "owned_pids": pids,
    "residual_pids": residual,
    "recorded_unix": time.time(),
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
raise SystemExit(0 if not residual else 75)
PY
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP

export STEP3_VL_10B_MODEL_PATH="$model_path"
export SLOW_BENCHMARK_RESULTS="$result_dir/step3"
export T5_LANE_B_RESULTS="$result_dir"
export PYTHONPATH="$frontend_root:$root${PYTHONPATH:+:$PYTHONPATH}"

setsid env CUDA_VISIBLE_DEVICES=0 "$python" -m slow_planner.serve \
  --config "$service_config" >"$result_dir/logs/step3_service.log" 2>&1 &
service_pid=$!

deadline=$((SECONDS + ${INTERNNAV_T5_STEP3_READY_TIMEOUT_SEC:-900}))
while (( SECONDS < deadline )); do
  kill -0 "$service_pid" 2>/dev/null || {
    printf 'Step3 service exited before health readiness\n' >&2
    exit 1
  }
  if "$python" - "$result_dir/step3/health.json" <<'PY' >/dev/null 2>&1
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
    "endpoint": "tcp://127.0.0.1:8200",
    "checks": checks,
    "health": health,
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
test -f "$result_dir/step3/health.json"

if command -v nvidia-smi >/dev/null 2>&1; then
  set +e
  nvidia-smi --query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
    --format=csv,noheader,nounits | head -n 1 | \
    "$python" -c 'import json,sys,time
from pathlib import Path
p=[x.strip() for x in sys.stdin.read().split(",")]
def metric(value):
    if value in {"", "N/A", "[N/A]", "Not Supported", "[Not Supported]"}:
        return None
    return float(value)
def kib_rows(path):
    rows={}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        key,_,rest=line.partition(":")
        if key in {"MemTotal","MemAvailable","SwapTotal","SwapFree","VmRSS","VmSwap"}:
            rows[key]=float(rest.strip().split()[0])
    return rows
system=kib_rows("/proc/meminfo")
process=kib_rows(f"/proc/{int(sys.argv[1])}/status")
print(json.dumps({"timestamp":time.time(),"gpu_index":int(p[0]),"gpu_name":p[1],"gpu_uuid":p[2],"gpu_util_percent":metric(p[3]),"memory_used_mib":metric(p[4]),"memory_total_mib":metric(p[5]),"temperature_c":metric(p[6]),"power_w":metric(p[7]),"unified_memory_total_mib":system.get("MemTotal",0.0)/1024.0,"unified_memory_available_mib":system.get("MemAvailable",0.0)/1024.0,"system_swap_used_mib":(system.get("SwapTotal",0.0)-system.get("SwapFree",0.0))/1024.0,"step3_process_rss_mib":process.get("VmRSS",0.0)/1024.0,"step3_process_swap_mib":process.get("VmSwap",0.0)/1024.0}))' "$service_pid" \
    >"$result_dir/step3/gpu.json"
  telemetry_rc=$?
  set -e
  test "$telemetry_rc" = 0 || rm -f -- "$result_dir/step3/gpu.json"
fi

setsid env CUDA_VISIBLE_DEVICES=0 "$python" -m slow_planner_frontend \
  --config "$frontend_config" >"$result_dir/logs/frontend.log" 2>&1 &
frontend_pid=$!
deadline=$((SECONDS + 30))
while (( SECONDS < deadline )); do
  kill -0 "$frontend_pid" 2>/dev/null || exit 1
  if "$python" - "$result_dir/frontend_state.json" <<'PY' >/dev/null 2>&1
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
test -f "$result_dir/frontend_state.json"

python3 - "$result_dir/shadow_ready.json" "$mode" "$service_pid" "$frontend_pid" <<'PY'
import json, os, sys, time
from pathlib import Path
output = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "status": "READY",
    "mode": sys.argv[2],
    "service_pid": int(sys.argv[3]),
    "frontend_pid": int(sys.argv[4]),
    "service_endpoint": "tcp://127.0.0.1:8200",
    "frontend_endpoint": "http://127.0.0.1:8300",
    "motion_authority": "none",
    "terminal_stop_authority": "none",
    "recorded_unix": time.time(),
}
temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, output)
PY

if test "$mode" = health-only; then
  exit 0
fi
while test ! -e "$result_dir/stop.request"; do
  kill -0 "$service_pid"
  kill -0 "$frontend_pid"
  sleep 1
done
