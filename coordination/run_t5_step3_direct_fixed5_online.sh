#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'usage: %s PROFILE RUN_ID CODE_SHA PREP_RESULT_ROOT RESULT_ROOT\n' "${0##*/}" >&2
  printf 'PROFILE: screen1 | fixed5\n' >&2
  exit 64
}

[[ $# -eq 5 ]] || usage
profile="$1"
run_id="$2"
code_sha="$3"
prep_relative="$4"
result_relative="$5"
case "$profile" in screen1) execution_count=1 ;; fixed5) execution_count=5 ;; *) usage ;; esac
[[ "$run_id" =~ ^[a-z0-9][a-z0-9._-]{7,95}$ ]] || usage
[[ "$code_sha" =~ ^[0-9a-f]{40}$ ]] || usage
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
prep_dir="$root/$prep_relative"
result_dir="$root/$result_relative"
expected_result="results/internnav_t5/step3-direct-${profile}-${run_id}"
test "$result_relative" = "$expected_result" || usage
[[ "$prep_relative" =~ ^results/internnav_t5/step3-lane-b-prepare-[a-z0-9][a-z0-9._-]{7,95}$ ]] || usage
test -f "$prep_dir/fast_prepare_input.json"

git_command=(git -C "$root")
if [[ -f "$root/.git" ]] && grep -Eq '^gitdir: [A-Za-z]:/' "$root/.git"; then
  git_command=(git.exe -C "$(wslpath -w "$root")")
fi
test "$("${git_command[@]}" rev-parse HEAD | tr -d '\r')" = "$code_sha"
test -z "$("${git_command[@]}" status --porcelain --untracked-files=all | tr -d '\r')"

if [[ "${INTERNNAV_T5_INSIDE_STEP3_DIRECT:-0}" != 1 ]]; then
  test ! -e "$result_dir"
  mkdir -p "$root/results/internnav_t5"
  lease_tmp="$(mktemp -d "$root/results/internnav_t5/.step3-direct-${run_id}.XXXXXX")"
  set +e
  env ISAAC_HOST=10.100.120.123 bash "$root/scripts/with_resource_lease.sh" lane-b \
    --owner codex-t5-step3-direct \
    --task "t5-step3-direct:${profile}:${run_id}:${code_sha}" \
    --log-dir "$lease_tmp" --acquire-timeout 7200 \
    --cleanup-timeout 600 --kill-wait-timeout 60 -- \
    env INTERNNAV_T5_INSIDE_STEP3_DIRECT=1 \
      INTERNNAV_T5_RESOURCE_LEASE_ACK=lane-b \
      bash "$root/coordination/run_t5_step3_direct_fixed5_online.sh" \
        "$profile" "$run_id" "$code_sha" "$prep_relative" "$result_relative"
  rc=$?
  set -e
  if test -d "$result_dir"; then
    mkdir -p "$result_dir/lease"
    cp -a -- "$lease_tmp/." "$result_dir/lease/"
    python3 - "$result_dir/lease_release_summary.json" \
      "$result_dir/lease/lease_metadata.txt" \
      "$result_dir/lease/lease_cleanup_receipt.json" "$rc" <<'PY'
import json, sys, time
from pathlib import Path
output, metadata_path, cleanup_path = map(Path, sys.argv[1:4])
text = metadata_path.read_text(encoding="utf-8") if metadata_path.is_file() else ""
cleanup = json.loads(cleanup_path.read_text(encoding="utf-8")) if cleanup_path.is_file() else {}
checks = {
    "dgx_b_only": "dgx_b" in text and "dgx_a" not in text,
    "isaac_gpu1_only": "isaac_gpu1" in text and "isaac_gpu0" not in text,
    "released": "state=RELEASED" in text,
    "cleanup_pass": cleanup.get("status") == "PASS" and cleanup.get(
        "wrapped_process_group_absent_before_lock_release"
    ) is True,
}
payload = {
    "schema_version": 1, "status": "PASS" if all(checks.values()) else "FAIL",
    "command_exit": int(sys.argv[4]), "resource_profile": "lane-b",
    "dgx_b_released": all(checks.values()), "checks": checks,
    "recorded_unix": time.time(),
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  fi
  case "$lease_tmp" in
    "$root/results/internnav_t5/.step3-direct-${run_id}."*) rm -rf -- "$lease_tmp" ;;
    *) echo "refusing to remove unexpected lease temp path" >&2; exit 75 ;;
  esac
  exit "$rc"
fi

test "${INTERNNAV_T5_RESOURCE_LEASE_ACK:-}" = lane-b
test ! -e "$result_dir"
mkdir -p "$result_dir/logs" "$result_dir/remote/dgx_b" \
  "$result_dir/remote/x86" "$result_dir/audits"

mapfile -t binding < <(python3 - "$prep_dir/fast_prepare_input.json" <<'PY'
import json, sys
value=json.load(open(sys.argv[1],encoding="utf-8"))
print(value["code_ref_sha"])
print(value["deployment_roots"]["dgx_b"])
print(value["deployment_roots"]["x86_b"])
print(value["dataset_root"])
print(value["dataset_file_sha256"])
print(",".join(value["episode_keys"]))
PY
)
test "${binding[0]}" = "$code_sha"
dgx_root="${binding[1]}"
x86_root="${binding[2]}"
dataset_root="${binding[3]}"
dataset_sha="${binding[4]}"
episode_keys_csv="${binding[5]}"
map_manifest="$dgx_root/inputs/d0_fixed5_static_maps/manifest.json"
run_token="step3-direct-${profile}-${run_id}"
dgx_run="$dgx_root/results/$run_token"
x86_run="$x86_root/results/$run_token"
dgx_ledger="$dgx_root/results/.${run_token}.supervisor.json"
x86_ledger="$x86_root/results/.${run_token}.supervisor.json"
dgx_target=rail@10.100.120.122
x86_target=song@10.100.120.123
container=internnav_t5_isaac_b
ssh_options=(-T -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=2)
dgx_ssh_pid=""
x86_ssh_pid=""
container_started=0
completed=0

remote() { local target="$1"; shift; ssh "${ssh_options[@]}" "$target" "$@"; }

stop_remote_ledger() {
  local target="$1" ledger="$2"
  remote "$target" "python3 - '$ledger'" <<'PY' || true
import json, os, signal, sys, time
from pathlib import Path
path=Path(sys.argv[1])
if not path.is_file(): raise SystemExit(0)
value=json.loads(path.read_text(encoding="utf-8"))
pgid=int(value.get("pgid",0))
if pgid <= 1: raise SystemExit(0)
try: os.killpg(pgid, signal.SIGTERM)
except ProcessLookupError: raise SystemExit(0)
deadline=time.monotonic()+20
while time.monotonic()<deadline:
    try: os.killpg(pgid,0)
    except ProcessLookupError: raise SystemExit(0)
    time.sleep(.1)
try: os.killpg(pgid,signal.SIGKILL)
except ProcessLookupError: pass
PY
}

finish() {
  local rc=$?
  trap - EXIT INT TERM HUP
  set +e
  if test "$completed" != 1; then
    remote "$dgx_target" "test ! -d '$dgx_run' || touch '$dgx_run/stop.request'" >/dev/null 2>&1
    stop_remote_ledger "$x86_target" "$x86_ledger"
    stop_remote_ledger "$dgx_target" "$dgx_ledger"
  fi
  if test "$container_started" = 1; then
    remote "$x86_target" "docker stop -t 20 '$container' >/dev/null 2>&1 || true" >/dev/null 2>&1
  fi
  test -z "$x86_ssh_pid" || wait "$x86_ssh_pid" 2>/dev/null || true
  test -z "$dgx_ssh_pid" || wait "$dgx_ssh_pid" 2>/dev/null || true
  exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT TERM HUP

remote "$dgx_target" \
  "set -euo pipefail; test \"\$(id -un)\" = rail; test \"\$(cat '$dgx_root/T5_DEPLOYMENT_REF')\" = '$code_sha'; test -x '$dgx_root/scripts/run_t5_step3_direct_dgx_lane.sh'; test -f '$map_manifest'; test ! -e '$dgx_run'; test ! -e '$dgx_ledger'; test -z \"\$(nvidia-smi --query-compute-apps=pid --format=csv,noheader|sed '/^[[:space:]]*$/d')\"; for p in 25138 25239 8200 8300; do test -z \"\$(ss -H -lntup|grep -E \"[:.]\$p[[:space:]]\"||true)\"; done"
remote "$x86_target" \
  "set -euo pipefail; test \"\$(id -un)\" = song; test \"\$(cat '$x86_root/T5_DEPLOYMENT_REF')\" = '$code_sha'; test -x '$x86_root/scripts/run_t5_distributed_isaac.sh'; test -x '$x86_root/scripts/materialize_t5_screen_dataset.py'; test ! -e '$x86_run'; test ! -e '$x86_ledger'; test \"\$(sha256sum '$dataset_root/val_unseen/val_unseen.json.gz'|cut -d' ' -f1)\" = '$dataset_sha'; test \"\$(docker inspect -f '{{.State.Running}}' '$container')\" = false; test \"\$(docker inspect -f '{{.State.Pid}}' '$container')\" = 0; test \"\$(docker inspect -f '{{.HostConfig.CpusetCpus}}' '$container')\" = 1,3,5,7,9,11,13,15,17; test -z \"\$(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader|sed '/^[[:space:]]*$/d')\""

read -r -d '' dgx_program <<'REMOTE_DGX' || true
set -euo pipefail
deployment="$1"; result="$2"; map="$3"; ledger="$4"
pgid="$(ps -o pgid= -p $$|tr -d ' ')"; sid="$(ps -o sid= -p $$|tr -d ' ')"
python3 - "$ledger" "$result" "$$" "$pgid" "$sid" <<'PY'
import json,os,sys,time
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"schema_version":1,"run_root":sys.argv[2],
 "pid":int(sys.argv[3]),"pgid":int(sys.argv[4]),"sid":int(sys.argv[5]),
 "role":"dgx_b_step3_direct","wall_unix":time.time()},indent=2,sort_keys=True)+"\n",encoding="utf-8")
PY
exec env INTERNNAV_T5_RESOURCE_LEASE_ACK=lane-b \
 INTERNNAV_T5_STEP3_DIRECT_HIGH_LEVEL=1 INTERNNAV_T5_STEP3_LIVE_ADVISOR=0 \
 INTERNNAV_T5_LANE=b INTERNNAV_T5_LANE_NAMESPACE=/t5/lane_b \
 INTERNNAV_T5_ID_PREFIX='b::' INTERNNAV_RUNTIME_POLICY=completion_sim \
 INTERNNAV_SIMULATION_TARGET=isaac ROS_DOMAIN_ID=76 CUDA_VISIBLE_DEVICES=0 \
 INTERNNAV_T1_CONTROL_ROOT="$deployment" INTERNVLA_ROS_WS="$deployment/ros_ws" \
 bash "$deployment/scripts/run_t5_step3_direct_dgx_lane.sh" "$result" "$map"
REMOTE_DGX
dgx_b64="$(printf '%s' "$dgx_program"|base64|tr -d '\r\n')"
dgx_command="exec setsid --wait bash -c \"\$(printf '%s' '$dgx_b64'|base64 -d)\" direct-dgx '$dgx_root' '$dgx_run' '$map_manifest' '$dgx_ledger'"
remote "$dgx_target" "$dgx_command" >"$result_dir/logs/dgx_runtime_ssh.log" 2>&1 &
dgx_ssh_pid=$!
deadline=$((SECONDS + 3600))
while ((SECONDS < deadline)); do
  kill -0 "$dgx_ssh_pid" 2>/dev/null || break
  remote "$dgx_target" "test -f '$dgx_run/direct_ready.json'" >/dev/null 2>&1 && break
  sleep 2
done
kill -0 "$dgx_ssh_pid"
remote "$dgx_target" "python3 -c 'import json;v=json.load(open(\"$dgx_run/direct_ready.json\"));assert v[\"status\"]==\"READY\" and v[\"internvla_model_loaded\"] is False'"

remote "$x86_target" "docker start '$container' >/dev/null"
container_started=1
read -r -d '' x86_program <<'REMOTE_X86' || true
set -euo pipefail
deployment="$1"; result="$2"; dataset="$3"; ledger="$4"; count="$5"; source_sha="$6"; keys="$7"
pgid="$(ps -o pgid= -p $$|tr -d ' ')"; sid="$(ps -o sid= -p $$|tr -d ' ')"
python3 - "$ledger" "$result" "$$" "$pgid" "$sid" <<'PY'
import json,sys,time
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"schema_version":1,"run_root":sys.argv[2],
 "pid":int(sys.argv[3]),"pgid":int(sys.argv[4]),"sid":int(sys.argv[5]),
 "role":"x86_b_step3_direct","wall_unix":time.time()},indent=2,sort_keys=True)+"\n",encoding="utf-8")
PY
if test "$count" = 1; then
  screen="$deployment/inputs/screen_datasets/$(basename "$result")"
  mkdir -p "$(dirname "$screen")"
  python3 "$deployment/scripts/materialize_t5_screen_dataset.py" \
    --source-root "$dataset" --output-root "$screen" --count 1 \
    --expected-source-sha256 "$source_sha" --expected-episode-keys "$keys"
  dataset="$screen"
fi
exec env INTERNNAV_T5_RESOURCE_LEASE_ACK=lane-b \
 INTERNNAV_RUNTIME_POLICY=completion_sim INTERNNAV_SIMULATION_TARGET=isaac \
 INTERNNAV_T5_LANE=b INTERNNAV_T5_LANE_NAMESPACE=/t5/lane_b \
 INTERNNAV_T5_ID_PREFIX='b::' INTERNNAV_T5_STEP3_DIRECT_HIGH_LEVEL=1 \
 INTERNNAV_T5_STEP3_LIVE_ADVISOR=0 \
 INTERNNAV_T5_ISAAC_SENSOR_PROFILE=lane_b_step3_direct_fixed5 \
 INTERNNAV_T5_RTF_ABLATION_PROFILE=navigation_fast ROS_DOMAIN_ID=76 \
 CUDA_VISIBLE_DEVICES=1 INTERNVLA_T5_LANE_B_CPUSET=1,3,5,7,9,11,13,15,17 \
 INTERNNAV_T1_CONTROL_ROOT="$deployment" \
 INTERNVLA_ROS_WS=/home/song/internnav-t4/isaac_ros_ws_45 \
 INTERNVLA_T5_ISAAC_WORKER_ROOT=/home/song/internnav-t1-t2/runtime/t5_isaac_workers \
 bash "$deployment/scripts/run_t5_distributed_isaac.sh" b model "$result" "$dataset"
REMOTE_X86
x86_b64="$(printf '%s' "$x86_program"|base64|tr -d '\r\n')"
x86_command="exec setsid --wait bash -c \"\$(printf '%s' '$x86_b64'|base64 -d)\" direct-x86 '$x86_root' '$x86_run' '$dataset_root' '$x86_ledger' '$execution_count' '$dataset_sha' '$episode_keys_csv'"
remote "$x86_target" "$x86_command" >"$result_dir/logs/x86_runtime_ssh.log" 2>&1 &
x86_ssh_pid=$!
timeout_sec=10800; test "$profile" = fixed5 && timeout_sec=21600
deadline=$((SECONDS + timeout_sec))
while kill -0 "$x86_ssh_pid" 2>/dev/null; do
  kill -0 "$dgx_ssh_pid" 2>/dev/null || { echo "DGX direct runtime exited" >&2; exit 1; }
  ((SECONDS < deadline)) || exit 124
  sleep 2
done
set +e; wait "$x86_ssh_pid"; x86_rc=$?; set -e
x86_ssh_pid=""
test "$x86_rc" = 0
remote "$dgx_target" "touch '$dgx_run/stop.request'"
deadline=$((SECONDS + 300))
while kill -0 "$dgx_ssh_pid" 2>/dev/null && ((SECONDS < deadline)); do sleep 1; done
! kill -0 "$dgx_ssh_pid" 2>/dev/null
set +e; wait "$dgx_ssh_pid"; dgx_rc=$?; set -e
dgx_ssh_pid=""
test "$dgx_rc" = 0
remote "$x86_target" "docker stop -t 20 '$container' >/dev/null; test \"\$(docker inspect -f '{{.State.Pid}}' '$container')\" = 0"
container_started=0

remote "$dgx_target" "tar -C '$dgx_run' -czf - ." | tar -C "$result_dir/remote/dgx_b" -xzf -
remote "$x86_target" "tar -C '$x86_run' -czf - ." | tar -C "$result_dir/remote/x86" -xzf -
python3 - "$result_dir/run_summary.json" "$profile" "$code_sha" "$dgx_root" "$x86_root" <<'PY'
import json,sys,time
from pathlib import Path
root=Path(sys.argv[1]).parent
dgx=json.loads((root/"remote/dgx_b/cleanup.json").read_text(encoding="utf-8"))
x86=json.loads((root/"remote/x86/isaac_status.json").read_text(encoding="utf-8"))
checks={
 "dgx_cleanup":dgx.get("status")=="PASS" and dgx.get("dgx_b_released") is True,
 "x86_completed":x86.get("status")=="PASS",
 "internvla_absent":json.loads((root/"remote/dgx_b/health/no_internvla_cleanup.json").read_text()).get("status")=="PASS",
}
payload={"schema_version":1,"status":"PASS" if all(checks.values()) else "FAIL",
 "profile":sys.argv[2],"code_sha":sys.argv[3],"deployment_roots":{"dgx_b":sys.argv[4],"x86_b":sys.argv[5]},
 "planner_mode":"direct_high_level","internvla_model_loaded":False,
 "internvla_fallback_allowed":False,"checks":checks,"recorded_unix":time.time()}
Path(sys.argv[1]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
raise SystemExit(0 if payload["status"]=="PASS" else 75)
PY
completed=1
trap - EXIT INT TERM HUP
finish
