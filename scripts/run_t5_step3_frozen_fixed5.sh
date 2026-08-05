#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'usage: %s <bounded_advisor|direct_high_level> RESULT_DIR MODEL_PATH VENV_PATH FROZEN_MANIFEST MANIFEST_SHA256\n' \
    "${0##*/}" >&2
  exit 64
}

[[ $# -eq 6 ]] || usage
mode="$1"
result_dir="$2"
model_path="$3"
venv_path="$4"
manifest="$5"
manifest_sha256="$6"
case "$mode" in bounded_advisor|direct_high_level) ;; *) usage ;; esac
[[ "$manifest_sha256" =~ ^[0-9a-f]{64}$ ]] || usage

root="${INTERNNAV_T1_CONTROL_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)}"
python="$venv_path/bin/python"

resource_lease_ack="${INTERNNAV_T5_RESOURCE_LEASE_ACK:-}"
case "$resource_lease_ack" in
  dgx-a|lane-a) expected_user=railgun; expected_home=/home/railgun ;;
  dgx-b|lane-b) expected_user=rail; expected_home=/home/rail ;;
  *) exit 73 ;;
esac
test "$(id -un)" = "$expected_user"
test -x "$python"
test -d "$model_path"
test -f "$manifest"
test "$($python - "$manifest" <<'PY'
import hashlib,sys
print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())
PY
)" = "$manifest_sha256"
test -x "$root/scripts/run_t5_step3_shadow.sh"
test -f "$root/scripts/run_t5_step3_frozen_fixed5.py"
[[ "$result_dir" = "$expected_home"/* ]]
test ! -e "$result_dir"
for port in 8200 8300; do
  test -z "$(ss -H -lntp | grep -E "[:.]${port}[[:space:]]" || true)"
done

mkdir -p "$result_dir/logs"
result_dir="$(cd -- "$result_dir" && pwd -P)"
manifest="$(cd -- "$(dirname -- "$manifest")" && pwd -P)/$(basename -- "$manifest")"
runner_pid=""

stop_runtime() {
  local pid="$1"
  [[ -n "$pid" ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    touch "$result_dir/runtime/stop.request" 2>/dev/null || true
  fi
  for _ in $(seq 1 180); do
    kill -0 "$pid" 2>/dev/null || {
      wait "$pid" 2>/dev/null || return $?
      return 0
    }
    sleep 1
  done
  kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 1 30); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  kill -KILL "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  return 75
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM HUP
  if [[ -n "$runner_pid" ]]; then
    stop_runtime "$runner_pid" || rc=75
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP

env INTERNNAV_T5_RESOURCE_LEASE_ACK="$resource_lease_ack" INTERNNAV_T1_CONTROL_ROOT="$root" \
  bash "$root/scripts/run_t5_step3_shadow.sh" shadow "$result_dir/runtime" \
    "$model_path" "$venv_path" >"$result_dir/logs/runtime_wrapper.log" 2>&1 &
runner_pid=$!

deadline=$((SECONDS + ${INTERNNAV_T5_STEP3_READY_TIMEOUT_SEC:-900}))
while (( SECONDS < deadline )); do
  kill -0 "$runner_pid" 2>/dev/null || {
    printf 'Step3 shadow runtime exited before readiness\n' >&2
    exit 1
  }
  test -f "$result_dir/runtime/shadow_ready.json" && break
  sleep 2
done
test -f "$result_dir/runtime/shadow_ready.json"

env PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}" \
  "$python" "$root/scripts/run_t5_step3_frozen_fixed5.py" \
    --manifest "$manifest" --manifest-sha256 "$manifest_sha256" \
    --endpoint tcp://127.0.0.1:8200 --mode "$mode" \
    --runtime-step3-dir "$result_dir/runtime/step3" \
    --output "$result_dir/fixed5" \
    >"$result_dir/logs/fixed5.log" 2>&1

"$python" - "$result_dir/frontend_final_state.json" \
  "$result_dir/fixed5/summary.json" <<'PY'
import json,sys,urllib.request
from pathlib import Path

output=Path(sys.argv[1])
summary=json.load(open(sys.argv[2], encoding="utf-8"))
with urllib.request.urlopen("http://127.0.0.1:8300/api/v1/health", timeout=2) as response:
    health=json.load(response)
with urllib.request.urlopen("http://127.0.0.1:8300/api/v1/state", timeout=2) as response:
    state=json.load(response)
assert summary["status"] == "PASS" and summary["case_count"] == 5
assert health["lane_id"] == "b" and health["readonly"] is True
assert state["lane_id"] == "b" and state["readonly"] is True
assert len(state["cameras"]) == 4 and all(row["available"] for row in state["cameras"])
assert state["decision"]["decision"]["motion_authority"] == "none"
encoded=json.dumps({"health":health,"state":state}, sort_keys=True)
for forbidden in ("raw_text","chain_of_thought","reasoning_content","cmd" + "_vel"):
    assert forbidden not in encoded
output.write_text(json.dumps({"health":health,"state":state},indent=2,sort_keys=True)+"\n",encoding="utf-8")
PY

touch "$result_dir/runtime/stop.request"
wait "$runner_pid"
runner_pid=""
test "$(python3 - "$result_dir/runtime/cleanup.json" <<'PY'
import json,sys
value=json.load(open(sys.argv[1],encoding="utf-8"))
print(value.get("status"))
PY
)" = PASS
for port in 8200 8300; do
  test -z "$(ss -H -lntp | grep -E "[:.]${port}[[:space:]]" || true)"
done

python3 - "$result_dir/entry_summary.json" "$mode" "$manifest_sha256" \
  "$result_dir/fixed5/summary.json" <<'PY'
import json,sys,time
from pathlib import Path
output=Path(sys.argv[1])
fixed=json.load(open(sys.argv[4],encoding="utf-8"))
value={
  "schema_version":1,
  "status":"PASS",
  "mode":sys.argv[2],
  "manifest_sha256":sys.argv[3],
  "fixed5_status":fixed["status"],
  "promotion_eligible":fixed["promotion_eligible"],
  "model_interface_eligible":fixed["model_interface_eligible"],
  "navigation_effect_claim_eligible":fixed["navigation_effect_claim_eligible"],
  "evaluation_scope":fixed["evaluation_scope"],
  "frontend_readonly":True,
  "motion_authority":"none",
  "terminal_stop_authority":"none",
  "owned_cleanup":True,
  "recorded_wall_time_s":time.time(),
}
output.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="utf-8")
print(json.dumps(value,sort_keys=True))
PY
