#!/usr/bin/env bash
set -euo pipefail

readonly COORDINATION_REF=refs/heads/codex/parallel-integration

resolve_git_dir() {
  local root="$1" raw
  if [[ -d "$root/.git" ]]; then readlink -f "$root/.git"; return; fi
  [[ -f "$root/.git" ]] || exit 64
  raw="$(sed -n 's/^gitdir: //p' "$root/.git")"
  case "$raw" in
    [A-Za-z]:/*|[A-Za-z]:\\*) wslpath -u "$raw" ;;
    /*) printf '%s\n' "$raw" ;;
    *) readlink -f "$root/$raw" ;;
  esac
}

if [[ "${1:-}" == _lane ]]; then
  shift
  [[ $# -ge 10 ]] || exit 64
  lane="$1"; root="$2"; authorization_ref="$3"; deployment_ref="$4"
  dgx_root="$5"; isaac_root="$6"; result_rel="$7"; prepare_rel="$8"
  prerequisite_rel="$9"; shift 9
  case "$lane" in a|b) ;; *) exit 64 ;; esac
  for spec in "$@"; do
    variant="${spec%%:*}"; index="${spec##*:}"
    case "$variant:$index" in
      recovery_on:11|recovery_off:12|h1_view:13|go2_view:14) ;;
      *) exit 64 ;;
    esac
    printf -v arm_grant '%s-a%02d' "${result_rel##*-00-}" "$index"
    arm_result="$result_rel/arms/$variant"
    INTERNNAV_T4_LANE_ID="$lane" bash "$root/coordination/run_t4_dgx_migration_online.sh" \
      _under_lease "$root" "$arm_grant" "$authorization_ref" "$deployment_ref" \
      "$dgx_root" "$isaac_root" "$arm_result" "$result_rel/leases/lane-${lane}" \
      "$prepare_rel" "$root/coordination/t4_functional_run_contract.py" \
      "migrated_ablation20_${variant}" "$prerequisite_rel"
  done
  exit 0
fi

[[ $# -eq 5 ]] || {
  echo "usage: run_t4_migrated_ablation_parallel_resume_online.sh GRANT AUTH_SHA PREPARE_RESULT_REL SELECTED_RECOVERY_RESULT_REL SOURCE_MATRIX_RESULT_REL" >&2
  exit 64
}
grant_id="$1"; authorization_ref="$2"; prepare_rel="$3"
prerequisite_rel="$4"; source_rel="$5"
[[ "$grant_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{7,95}$ ]] || exit 64
[[ "$authorization_ref" =~ ^[0-9a-f]{40}$ ]] || exit 64
[[ "$prepare_rel" =~ ^results/parallel/t4_functional/prepare-00-[A-Za-z0-9][A-Za-z0-9._-]{7,95}$ ]] || exit 64
[[ "$prerequisite_rel" =~ ^results/parallel/t4_functional/migrated-recovery-a5-00-[A-Za-z0-9][A-Za-z0-9._-]{7,95}$ ]] || exit 64
[[ "$source_rel" =~ ^results/parallel/t4_functional/migrated-ablation-matrix-00-[A-Za-z0-9][A-Za-z0-9._-]{7,95}$ ]] || exit 64

script_path="$(readlink -f "${BASH_SOURCE[0]}")"
root="$(readlink -f "$(dirname "$script_path")/..")"
git_dir="$(resolve_git_dir "$root")"
actual_sha="$(git -c core.autocrlf=false --git-dir="$git_dir" rev-parse "$COORDINATION_REF")"
test "$actual_sha" = "$authorization_ref"
test -z "$(git -c core.autocrlf=true --git-dir="$git_dir" --work-tree="$root" status --porcelain --untracked-files=no)"

result_rel="results/parallel/t4_functional/migrated-ablation-parallel-resume-00-${grant_id}"
result_dir="$root/$result_rel"
test ! -e "$result_dir"
test -d "$root/$source_rel"
test -d "$root/$prepare_rel"
python3 - "$root" "$git_dir" "$authorization_ref" "$grant_id" "$result_rel" <<'PY'
import json,re,subprocess,sys
root,git_dir,ref,grant,result=sys.argv[1:]
doc=subprocess.run(["git",f"--git-dir={git_dir}",f"--work-tree={root}","show",f"{ref}:coordination/TASK_BOARD.md"],check=True,capture_output=True,text=True).stdout
blocks=re.findall(r"<!--\s*INTERNAV_ONLINE_GRANT_V1\s*\r?\n(.*?)\r?\nINTERNAV_ONLINE_GRANT_V1\s*-->",doc,flags=re.DOTALL)
expected={"schema_version":1,"status":"GRANTED","worker":"00","resource":"dual-dgx+dual-isaac-gpu","profile":"dgx_onboard_ablation20_parallel_resume","result_dir":result,"grant_id":grant}
if len(blocks)!=1 or json.loads(blocks[0])!=expected: raise SystemExit("parallel resume grant mismatch")
PY

mapfile -t fields < <(python3 "$root/coordination/t4_functional_run_contract.py" prepare-fields \
  --result-dir "$root/$prepare_rel" --git-dir "$git_dir" \
  --authorization-ref "$authorization_ref" --format pilot-lines)
test "${#fields[@]}" -eq 6
deployment_ref="${fields[1]}"; dgx_a_root="${fields[2]}"; isaac_a_root="${fields[4]}"
dgx_b_root="/home/railgun/internnav-t1-t2/.t4-deployments/${grant_id}-${deployment_ref:0:12}-dgx-b"
isaac_b_root="/home/song/internnav-t1-t2/.t4-deployments/${grant_id}-${deployment_ref:0:12}-isaac-b"
runner_commit_sha="$(git --git-dir="$git_dir" --work-tree="$root" log -1 --format=%H "$authorization_ref" -- \
  scripts/build_t4_ablation_episode_records.py coordination/run_t4_dgx_migration_online.sh \
  coordination/remote_t4_isaac_migration_session.sh coordination/remote_t4_dgx_onboard_session.sh \
  coordination/remote_t4_model_session.sh)"

python3 "$root/scripts/prepare_t4_ablation_resume.py" \
  --source-result "$root/$source_rel" --destination-result "$result_dir" \
  --runner-commit-sha "$runner_commit_sha" >"$result_dir.resume_import.log" 2>&1

prepare_lease="$result_dir/leases/prepare"
bash "$root/scripts/with_resource_lease.sh" all-lanes --owner codex-00 \
  --task "00-t4-parallel-lane-prepare-${grant_id}" --log-dir "$prepare_lease" -- \
  bash "$root/scripts/prepare_t4_parallel_lanes.sh" "$root" "$grant_id" \
    "$deployment_ref" "$dgx_a_root" "$isaac_a_root" "$dgx_b_root" "$isaac_b_root" \
    "$result_dir/parallel_lane_prepare.json"

baseline_available_bytes="$(ssh -o BatchMode=yes song@10.100.120.111 \
  "awk '/MemAvailable:/ {print \$2*1024}' /proc/meminfo")"
[[ "$baseline_available_bytes" =~ ^[1-9][0-9]*$ ]] || exit 69
set +e
bash "$root/scripts/with_resource_lease.sh" lane-a --owner codex-00 \
  --task "00-t4-ablation-recovery-pair-${grant_id}" --log-dir "$result_dir/leases/lane-a" -- \
  bash "$script_path" _lane a "$root" "$authorization_ref" "$deployment_ref" \
    "$dgx_a_root" "$isaac_a_root" "$result_rel" "$prepare_rel" "$prerequisite_rel" \
    recovery_on:11 recovery_off:12 &
lane_a_pid=$!

memory_ready=0
memory_probe_ok=1
for _ in $(seq 1 150); do
  kill -0 "$lane_a_pid" 2>/dev/null || break
  if ssh -o BatchMode=yes song@10.100.120.111 \
      "test -s /home/song/.codex-internnav-stage/t4-dgx-migration-${grant_id}-a11/isaac_outer.log && test \"\$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n1)\" -ge 1000"; then
    memory_ready=1
    break
  fi
  sleep 2
done
current_available_bytes="$(ssh -o BatchMode=yes song@10.100.120.111 \
  "awk '/MemAvailable:/ {print \$2*1024}' /proc/meminfo" 2>/dev/null || true)"
if [[ ! "$current_available_bytes" =~ ^[1-9][0-9]*$ ]]; then
  memory_probe_ok=0
  current_available_bytes=0
fi
# MemAvailable can briefly rebound when the kernel drops cache.  Sample the
# running single-Isaac lane for 30 seconds and retain the lowest value so a
# second instance is never admitted from one optimistic snapshot.
minimum_available_bytes="$current_available_bytes"
if (( memory_ready == 1 )); then
  for _ in $(seq 1 15); do
    sample_available_bytes="$(ssh -o BatchMode=yes song@10.100.120.111 \
      "awk '/MemAvailable:/ {print \$2*1024}' /proc/meminfo" 2>/dev/null || true)"
    if [[ ! "$sample_available_bytes" =~ ^[1-9][0-9]*$ ]]; then
      memory_probe_ok=0
      break
    fi
    if (( sample_available_bytes < minimum_available_bytes )); then
      minimum_available_bytes="$sample_available_bytes"
    fi
    sleep 2
  done
fi
current_available_bytes="$minimum_available_bytes"
observed_increment_bytes=$(( baseline_available_bytes > current_available_bytes ? baseline_available_bytes - current_available_bytes : 0 ))
required_remaining_bytes=$(( observed_increment_bytes + 4294967296 ))
memory_admitted=0
if (( memory_ready == 1 && memory_probe_ok == 1 && current_available_bytes >= required_remaining_bytes )); then
  memory_admitted=1
fi
python3 - "$result_dir/memory_admission.json" "$memory_ready" "$memory_probe_ok" "$memory_admitted" \
  "$baseline_available_bytes" "$current_available_bytes" "$observed_increment_bytes" \
  "$required_remaining_bytes" <<'PY'
import json,sys,time
from pathlib import Path
ready,probe,admitted=map(int,sys.argv[2:5]); nums=list(map(int,sys.argv[5:9]))
status=("PASS" if admitted else "MEMORY_UPGRADE_REQUIRED" if ready and probe else
        "MEMORY_PROBE_FAILED" if ready else "ISAAC_GPU0_RUNTIME_NOT_OBSERVED")
payload={"schema_version":1,"status":status,
"gpu0_runtime_observed":bool(ready),"memory_probe_ok":bool(probe),
"second_isaac_admitted":bool(admitted),
"baseline_mem_available_bytes":nums[0],"gpu0_mem_available_bytes":nums[1],
"observed_single_instance_increment_bytes":nums[2],"required_remaining_bytes":nums[3],
"reserved_headroom_bytes":4294967296,"real_go2_targeted":False,"recorded_unix":time.time()}
Path(sys.argv[1]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY

lane_b_pid=""
if (( memory_admitted == 1 )); then
  bash "$root/scripts/with_resource_lease.sh" lane-b --owner codex-00 \
    --task "00-t4-ablation-view-pair-${grant_id}" --log-dir "$result_dir/leases/lane-b" -- \
    bash "$script_path" _lane b "$root" "$authorization_ref" "$deployment_ref" \
      "$dgx_b_root" "$isaac_b_root" "$result_rel" "$prepare_rel" "$prerequisite_rel" \
      h1_view:13 go2_view:14 &
  lane_b_pid=$!
elif (( memory_ready == 1 && memory_probe_ok == 1 )); then
  python3 - "$result_dir/future_work.json" "$grant_id" "$authorization_ref" \
    "$result_rel" "$prepare_rel" "$prerequisite_rel" "$source_rel" <<'PY'
import json,sys,time
from pathlib import Path
payload={"schema_version":1,"status":"MEMORY_UPGRADE_REQUIRED","grant_id":sys.argv[2],
"authorization_ref_sha":sys.argv[3],"completed_or_running_lane":"a",
"deferred_lane":"b","deferred_variants":["h1_view","go2_view"],
"current_result_dir":sys.argv[4],"prepare_result_dir":sys.argv[5],
"selected_recovery_result_dir":sys.argv[6],"source_matrix_result_dir":sys.argv[7],
"preserved_work":["imported_and_revalidated_arms_01_through_10",
"recovery_on_arm_11_running_or_complete","recovery_off_arm_12_pending_or_running"],
"future_plan":["finish_lane_a_without_starting_lane_b",
"power_off_x86_isaac_host_only_after_user_confirmation","add_memory_and_boot_host",
"verify_ssh_ros2_and_two_gpu_health","issue_a_fresh_single_online_grant",
"resume_h1_view_arm_13_and_go2_view_arm_14","finalize_frozen_t4_6_ablation",
"archive_sha_logs_deviations_and_verify_all_resource_locks_are_free"],
"resume_profile":"dgx_onboard_ablation20_parallel_resume",
"required_action":"power_off_x86_isaac_host_and_add_memory_then_issue_fresh_grant",
"resume_entrypoint":"coordination/run_t4_migrated_ablation_parallel_resume_online.sh",
"automatic_shutdown_performed":False,"strict_evidence_modified":False,
"real_go2_targeted":False,"recorded_unix":time.time()}
Path(sys.argv[1]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
  echo "MEMORY_UPGRADE_REQUIRED future_work=$result_dir/future_work.json"
else
  blocker_status=ISAAC_GPU0_RUNTIME_NOT_OBSERVED
  blocker_action=inspect_lane_a_startup_and_cleanup_evidence_before_any_memory_change
  if (( memory_ready == 1 )); then
    blocker_status=MEMORY_PROBE_FAILED
    blocker_action=restore_ssh_memory_probe_before_any_memory_change
  fi
  python3 - "$result_dir/online_blocker.json" "$grant_id" "$authorization_ref" \
    "$result_rel" "$blocker_status" "$blocker_action" <<'PY'
import json,sys,time
from pathlib import Path
payload={"schema_version":1,"status":sys.argv[5],
"grant_id":sys.argv[2],"authorization_ref_sha":sys.argv[3],
"current_result_dir":sys.argv[4],"deferred_lane":"b",
"deferred_variants":["h1_view","go2_view"],
"required_action":sys.argv[6],
"memory_upgrade_requested":False,"automatic_shutdown_performed":False,
"strict_evidence_modified":False,"real_go2_targeted":False,
"recorded_unix":time.time()}
Path(sys.argv[1]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
  echo "$blocker_status blocker=$result_dir/online_blocker.json"
fi
wait "$lane_a_pid"; lane_a_rc=$?
if [[ -n "$lane_b_pid" ]]; then
  wait "$lane_b_pid"; lane_b_rc=$?
else
  if (( memory_ready == 1 && memory_probe_ok == 1 )); then lane_b_rc=78; else lane_b_rc=79; fi
fi
set -e

python3 - "$result_dir/batch_status.json" "$grant_id" "$authorization_ref" "$lane_a_rc" "$lane_b_rc" <<'PY'
import json,sys,time
from pathlib import Path
path=Path(sys.argv[1]); a=int(sys.argv[4]); b=int(sys.argv[5]); root=path.parent; arms={}
for arm in sorted((root/"arms").glob("*")):
    summary=next(arm.glob("migrated_ablation20_*_summary.json"),None)
    arms[arm.name]=json.loads(summary.read_text()).get("status","UNKNOWN") if summary else "MISSING"
payload={"schema_version":1,"status":"ONLINE_PASS_EVIDENCE_PENDING" if a==0 and b==0 else "FAIL",
"grant_id":sys.argv[2],"authorization_ref_sha":sys.argv[3],"lane_exit_codes":{"a":a,"b":b},
"arms":arms,"failed_lane_count":int(a!=0)+int(b!=0),"strict_evidence_modified":False,
"real_go2_targeted":False,"recorded_unix":time.time()}
path.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
(( lane_a_rc == 0 && lane_b_rc == 0 )) || exit 1

set +e
python3 "$root/scripts/finalize_t4_migrated_ablation.py" --result-dir "$result_dir" \
  >"$result_dir/finalize.log" 2>&1
finalize_rc=$?
set -e
python3 - "$result_dir/batch_status.json" "$finalize_rc" <<'PY'
import json,sys,time
from pathlib import Path
p=Path(sys.argv[1]); rc=int(sys.argv[2]); v=json.loads(p.read_text()); v["finalize_exit_code"]=rc
v["status"]="PASS" if rc==0 else "ONLINE_PASS_EVIDENCE_PENDING"; v["recorded_unix"]=time.time()
p.write_text(json.dumps(v,indent=2,sort_keys=True)+"\n")
PY
exit "$finalize_rc"
