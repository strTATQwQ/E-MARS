#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'usage: %s CODE_SHA DEPLOYMENT_ROOT RUN_ID results/internnav_t5/step3-replay-RUN_ID\n' "${0##*/}" >&2
  exit 64
}

[[ $# -eq 4 ]] || usage
code_sha="$1"
deployment_root="$2"
run_id="$3"
result_relative="$4"
[[ "$code_sha" =~ ^[0-9a-f]{40}$ ]] || usage
[[ "$deployment_root" =~ ^/home/rail/internnav-t1-t2/\.t5-deployments/t5step3-[a-z0-9._-]+-${code_sha:0:12}$ ]] || usage
[[ "$run_id" =~ ^[a-z0-9][a-z0-9._-]{7,63}$ ]] || usage
test "$result_relative" = "results/internnav_t5/step3-replay-$run_id" || usage

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
result_dir="$root/$result_relative"
target=rail@10.100.120.116
remote_result="/home/rail/internnav-t1-t2/results/t5_step3_replay/$run_id"
model_path=/home/rail/ai-stack/models/Step3-VL-10B
venv_path=/home/rail/ai-stack/venvs/step3-vl-10b-tf4.57.6

if [[ "${INTERNNAV_T5_INSIDE_STEP3_REPLAY:-0}" != 1 ]]; then
  test ! -e "$result_dir"
  mkdir -p "$result_dir"
  env INTERNNAV_T5_INSIDE_STEP3_REPLAY=1 \
    bash "$root/scripts/with_resource_lease.sh" dgx-b \
      --owner codex-00 --task "t5-step3-advisor-replay:$run_id:$code_sha" \
      --log-dir "$result_dir/lease" --acquire-timeout 30 \
      --cleanup-timeout 180 --kill-wait-timeout 30 -- \
    bash "$root/coordination/run_t5_step3_replay_online.sh" \
      "$code_sha" "$deployment_root" "$run_id" "$result_relative"
  exit $?
fi

test -d "$result_dir"
remote_program='set -euo pipefail
deployment="$1"; code_sha="$2"; result="$3"; model="$4"; venv="$5"; run_id="$6"
test "$(id -un)" = rail
test "$(cat "$deployment/T5_DEPLOYMENT_REF")" = "$code_sha"
test -x "$deployment/scripts/run_t5_step3_shadow.sh"
test -f "$deployment/scripts/probe_t5_step3_lane_b_replay.py"
test ! -e "$result"
mkdir -p "$result/input"
runner_pid=""
cleanup() {
  rc=$?
  trap - EXIT INT TERM HUP
  if test -n "$runner_pid" && kill -0 "$runner_pid" 2>/dev/null; then
    touch "$result/runtime/stop.request" 2>/dev/null || true
    for _ in $(seq 1 180); do kill -0 "$runner_pid" 2>/dev/null || break; sleep 1; done
    if kill -0 "$runner_pid" 2>/dev/null; then kill -TERM "$runner_pid" 2>/dev/null || true; fi
    wait "$runner_pid" 2>/dev/null || rc=75
  fi
  exit "$rc"
}
trap cleanup EXIT
trap "exit 130" INT TERM HUP
source_image="$deployment/isaac_vln_benchmark/isaac_vln_benchmark/runs/v10_target_pose_audit_local/visual/viewport.png"
test -s "$source_image"
"$venv/bin/python" - "$source_image" "$result/input" <<"PY"
import sys
from pathlib import Path
from PIL import Image, ImageEnhance
source=Path(sys.argv[1]); output=Path(sys.argv[2])
views=("front_left","front","front_right","rear")
with Image.open(source) as original:
    base=original.convert("RGB").resize((640,480), Image.Resampling.LANCZOS)
for index,view in enumerate(views):
    frame=ImageEnhance.Brightness(base).enhance(0.92 + 0.04*index)
    frame.save(output/f"{view}.jpg", format="JPEG", quality=92, optimize=True)
PY
env INTERNNAV_T5_RESOURCE_LEASE_ACK=dgx-b INTERNNAV_T1_CONTROL_ROOT="$deployment" \
  bash "$deployment/scripts/run_t5_step3_shadow.sh" shadow "$result/runtime" "$model" "$venv" &
runner_pid=$!
deadline=$((SECONDS+900))
while ((SECONDS < deadline)); do
  kill -0 "$runner_pid" 2>/dev/null || break
  test -f "$result/runtime/shadow_ready.json" && break
  sleep 2
done
kill -0 "$runner_pid"
test -f "$result/runtime/shadow_ready.json"
env PYTHONPATH="$deployment${PYTHONPATH:+:$PYTHONPATH}" \
  "$venv/bin/python" "$deployment/scripts/probe_t5_step3_lane_b_replay.py" \
  --endpoint tcp://127.0.0.1:8200 --mode bounded_advisor \
  --episode-id "$run_id" --sequence-id 0 --request-class warmup \
  --image "$result/input/front_left.jpg" --image "$result/input/front.jpg" \
  --image "$result/input/front_right.jpg" --image "$result/input/rear.jpg" \
  --output "$result/warmup.json"
env PYTHONPATH="$deployment${PYTHONPATH:+:$PYTHONPATH}" \
  "$venv/bin/python" "$deployment/scripts/probe_t5_step3_lane_b_replay.py" \
  --endpoint tcp://127.0.0.1:8200 --mode bounded_advisor \
  --episode-id "$run_id" --sequence-id 1 --request-class production \
  --image "$result/input/front_left.jpg" --image "$result/input/front.jpg" \
  --image "$result/input/front_right.jpg" --image "$result/input/rear.jpg" \
  --output "$result/advisor.json"
touch "$result/runtime/stop.request"
wait "$runner_pid"
runner_pid=""
python3 - "$result/warmup.json" "$result/advisor.json" "$result/runtime/cleanup.json" <<"PY"
import json,sys
warmup=json.load(open(sys.argv[1],encoding="utf-8"))
advisor=json.load(open(sys.argv[2],encoding="utf-8"))
cleanup=json.load(open(sys.argv[3],encoding="utf-8"))
assert warmup["status"] == "PASS" and warmup["deadline_ms"] == 180000
assert advisor["status"] == "PASS"
assert advisor["deadline_ms"] == 12000
assert advisor["decision"]["mode"] == "bounded_advisor"
assert advisor["decision"]["motion_authority"] == "none"
assert cleanup["status"] == "PASS" and cleanup["residual_pids"] == []
assert "raw_text" not in json.dumps([warmup,advisor])
PY
for port in 8200 8300; do
  test -z "$(ss -H -lntp | grep -E "[:.]${port}[[:space:]]" || true)"
done'

ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes "$target" \
  bash -s -- "$deployment_root" "$code_sha" "$remote_result" \
  "$model_path" "$venv_path" "$run_id" <<<"$remote_program"

mkdir -p "$result_dir/remote"
scp -q -r -o BatchMode=yes -o StrictHostKeyChecking=yes \
  "$target:$remote_result/input" "$target:$remote_result/runtime" \
  "$target:$remote_result/warmup.json" "$target:$remote_result/advisor.json" \
  "$result_dir/remote/"

python3 - "$result_dir" "$code_sha" "$deployment_root" "$remote_result" <<'PY'
import hashlib,json,sys,time
from pathlib import Path
root=Path(sys.argv[1]); code_sha,deployment,remote=sys.argv[2:]
warmup=json.loads((root/"remote/warmup.json").read_text(encoding="utf-8"))
advisor=json.loads((root/"remote/advisor.json").read_text(encoding="utf-8"))
cleanup=json.loads((root/"remote/runtime/cleanup.json").read_text(encoding="utf-8"))
checks={
 "advisor_contract":advisor.get("status")=="PASS" and advisor.get("decision",{}).get("mode")=="bounded_advisor",
 "warmup_contract":warmup.get("status")=="PASS" and warmup.get("deadline_ms")==180000,
 "no_motion_authority":advisor.get("decision",{}).get("motion_authority")=="none",
 "deterministic_response_or_fallback":advisor.get("model_request_status") in {"RESPONSE","TIMEOUT","ERROR"},
 "no_cot":all(word not in json.dumps([warmup,advisor]) for word in ("raw_text","chain_of_thought","reasoning_content")),
 "owned_cleanup":cleanup.get("status")=="PASS" and cleanup.get("residual_pids")==[],
}
payload={"schema_version":1,"status":"PASS" if all(checks.values()) else "FAIL",
 "code_ref_sha":code_sha,"deployment_root":deployment,"remote_result_root":remote,
 "checks":checks,"advisor_model_request_status":advisor.get("model_request_status"),
 "advisor_deadline_met":advisor.get("deadline_met"),"advisor_wall_ms":advisor.get("wall_ms"),
 "warmup_model_request_status":warmup.get("model_request_status"),
 "warmup_wall_ms":warmup.get("wall_ms"),
 "recorded_unix":time.time()}
(root/"step3_replay_summary.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
print(json.dumps(payload,sort_keys=True))
raise SystemExit(0 if payload["status"]=="PASS" else 75)
PY
