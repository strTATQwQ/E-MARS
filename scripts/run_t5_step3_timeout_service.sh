#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: ${0##*/} RESULT_ROOT MODEL_PATH VENV_PATH" >&2
  exit 64
fi
result_root="$1"
model_path="$2"
venv_path="$3"
root="${INTERNNAV_T1_CONTROL_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)}"
python="$venv_path/bin/python"
config="$root/configs/slow_models/step3_vl_10b_timeout_advisor.yaml"

lane="${INTERNNAV_T5_LANE:-}"
case "$lane" in
  a) expected_user=railgun; lane_ip=10.100.100.128; expected_lease=lane-a ;;
  b) expected_user=rail; lane_ip=10.100.120.122; expected_lease=lane-b ;;
  *) exit 64 ;;
esac
test "$(id -un)" = "$expected_user"
test "${INTERNNAV_T5_RESOURCE_LEASE_ACK:-}" = "$expected_lease"
test "${INTERNNAV_RUNTIME_POLICY:-}" = completion_sim
test "${INTERNNAV_SIMULATION_TARGET:-}" = isaac
test "${CUDA_VISIBLE_DEVICES:-}" = 0
ip -4 -o addr show scope global | awk '{sub(/\/.*/, "", $4); print $4}' | \
  grep -Fxq "$lane_ip"
test -x "$python"
test -d "$model_path"
test -f "$venv_path/T5_STEP3_RUNTIME_READY.json"
test -f "$config"
[[ "$result_root" = "$HOME"/* ]]
test ! -e "$result_root"
test -z "$(ss -H -lntp | grep -F "$lane_ip:8200" || true)"

mkdir -p "$result_root/logs" "$result_root/step3"
result_root="$(cd -- "$result_root" && pwd -P)"
model_path="$(cd -- "$model_path" && pwd -P)"
service_pid=""

stop_service() {
  test -n "$service_pid" || return 0
  kill -TERM -- "-$service_pid" 2>/dev/null || true
  # The owning Lane gives this wrapper a 20-second TERM window.  Finish the
  # nested service's own TERM -> KILL sequence inside that window so its
  # setsid process cannot be reparented after the wrapper is reaped.
  for _ in $(seq 1 100); do
    kill -0 "$service_pid" 2>/dev/null || return 0
    sleep 0.1
  done
  kill -KILL -- "-$service_pid" 2>/dev/null || true
  for _ in $(seq 1 50); do
    kill -0 "$service_pid" 2>/dev/null || return 0
    sleep 0.1
  done
  return 75
}

cleanup() {
  local rc=$? residual=0
  trap - EXIT INT TERM HUP
  set +e
  stop_service || residual=1
  python3 - "$result_root/cleanup.json" "$rc" "$residual" "$service_pid" <<'PY'
import json, sys, time
from pathlib import Path
path = Path(sys.argv[1])
pid = int(sys.argv[4]) if sys.argv[4] else 0
residual = bool(pid and Path(f"/proc/{pid}").exists())
value = {
    "schema_version": 1,
    "status": "PASS" if int(sys.argv[3]) == 0 and not residual else "FAIL",
    "command_exit": int(sys.argv[2]),
    "service_pid": pid,
    "residual": residual,
    "finished_unix": time.time(),
}
path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
raise SystemExit(0 if value["status"] == "PASS" else 75)
PY
  test $? = 0 || rc=75
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

export STEP3_VL_10B_MODEL_PATH="$model_path"
export SLOW_BENCHMARK_RESULTS="$result_root/step3"
export STEP3_TIMEOUT_BIND="tcp://$lane_ip:8200"
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
setsid env CUDA_VISIBLE_DEVICES=0 "$python" -m slow_planner.serve \
  --config "$config" >"$result_root/logs/service.log" 2>&1 &
service_pid=$!

deadline=$((SECONDS + ${INTERNVLA_T5_STEP3_READY_TIMEOUT_SEC:-900}))
while ((SECONDS < deadline)); do
  kill -0 "$service_pid" 2>/dev/null || {
    echo "Step3 timeout service exited before READY" >&2
    exit 1
  }
if "$python" - "$result_root/health.json" "$lane_ip" <<'PY' >/dev/null 2>&1
import json, sys, time, zmq
from pathlib import Path
ctx = zmq.Context()
sock = ctx.socket(zmq.REQ)
sock.setsockopt(zmq.LINGER, 0)
sock.setsockopt(zmq.SNDTIMEO, 1500)
sock.setsockopt(zmq.RCVTIMEO, 1500)
try:
    sock.connect(f"tcp://{sys.argv[2]}:8200")
    sock.send_json({"type": "health"})
    value = dict(sock.recv_json())
finally:
    sock.close(linger=0)
    ctx.term()
checks = {
    "ok": value.get("ok") is True,
    "model": value.get("model_variant") == "step3_vl_10b_bf16",
    "revision": value.get("revision") == "5026053b0c2f5dfaa08fc2d149384162c3c8bca1",
    "transformers": value.get("runtime_transformers_version") == "4.57.6",
    "clean_load": value.get("checkpoint_load_clean") is True,
    "bf16": set(value.get("parameter_dtype_counts") or {}) == {"torch.bfloat16"},
    "redacted": value.get("redact_raw_text") is True,
    "task_state": (
        value.get("planner_mode") == "task_state_v1"
        and value.get("private_task_state_contract") == 1
    ),
}
if not all(checks.values()):
    raise SystemExit(75)
Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": 1, "status": "READY", "checks": checks,
    "endpoint": f"tcp://{sys.argv[2]}:8200", "recorded_unix": time.time(),
    **value,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  then
    break
  fi
  sleep 2
done
test -f "$result_root/health.json"
wait "$service_pid"
