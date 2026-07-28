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

[[ $# -eq 6 ]] || {
  echo "usage: run_t4_migrated_ablation_view_resume_online.sh GRANT AUTH_SHA PREPARE_RESULT SELECTED_RECOVERY_RESULT SOURCE_MATRIX_RESULT PASSED_PARALLEL_RESULT" >&2
  exit 64
}
grant_id="$1"; authorization_ref="$2"; prepare_rel="$3"
prerequisite_rel="$4"; source_rel="$5"; passed_rel="$6"
[[ "$grant_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{7,95}$ ]] || exit 64
[[ "$authorization_ref" =~ ^[0-9a-f]{40}$ ]] || exit 64
[[ "$prepare_rel" =~ ^results/parallel/t4_functional/prepare-00-[A-Za-z0-9][A-Za-z0-9._-]{7,95}$ ]] || exit 64
[[ "$prerequisite_rel" =~ ^results/parallel/t4_functional/migrated-recovery-a5-00-[A-Za-z0-9][A-Za-z0-9._-]{7,95}$ ]] || exit 64
[[ "$source_rel" =~ ^results/parallel/t4_functional/migrated-ablation-matrix-00-[A-Za-z0-9][A-Za-z0-9._-]{7,95}$ ]] || exit 64
[[ "$passed_rel" =~ ^results/parallel/t4_functional/migrated-ablation-parallel-resume-00-[A-Za-z0-9][A-Za-z0-9._-]{7,95}$ ]] || exit 64

script_path="$(readlink -f "${BASH_SOURCE[0]}")"
root="$(readlink -f "$(dirname "$script_path")/..")"
git_dir="$(resolve_git_dir "$root")"
actual_sha="$(git -c core.autocrlf=false --git-dir="$git_dir" rev-parse "$COORDINATION_REF")"
test "$actual_sha" = "$authorization_ref"
test -z "$(git -c core.autocrlf=true --git-dir="$git_dir" --work-tree="$root" status --porcelain --untracked-files=no)"

result_rel="results/parallel/t4_functional/migrated-ablation-view-resume-00-${grant_id}"
result_dir="$root/$result_rel"
test ! -e "$result_dir"
for relative in "$prepare_rel" "$prerequisite_rel" "$source_rel" "$passed_rel"; do
  test -d "$root/$relative" && test ! -L "$root/$relative"
done
python3 - "$root" "$git_dir" "$authorization_ref" "$grant_id" "$result_rel" <<'PY'
import json,re,subprocess,sys
root,git_dir,ref,grant,result=sys.argv[1:]
doc=subprocess.run(["git",f"--git-dir={git_dir}",f"--work-tree={root}","show",f"{ref}:coordination/TASK_BOARD.md"],check=True,capture_output=True,text=True).stdout
blocks=re.findall(r"<!--\s*INTERNAV_ONLINE_GRANT_V1\s*\r?\n(.*?)\r?\nINTERNAV_ONLINE_GRANT_V1\s*-->",doc,flags=re.DOTALL)
expected={"schema_version":1,"status":"GRANTED","worker":"00","resource":"dual-dgx+dual-isaac-gpu","profile":"dgx_onboard_ablation20_view_resume","result_dir":result,"grant_id":grant}
if len(blocks)!=1 or json.loads(blocks[0])!=expected: raise SystemExit("view resume grant mismatch")
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
  --source-result "$root/$source_rel" --additional-source-result "$root/$passed_rel" \
  --destination-result "$result_dir" --runner-commit-sha "$runner_commit_sha" \
  >"$result_dir.resume_import.log" 2>&1

bash "$root/scripts/with_resource_lease.sh" all-lanes --owner codex-00 \
  --task "00-t4-view-resume-prepare-${grant_id}" --log-dir "$result_dir/leases/prepare" -- \
  bash "$root/scripts/prepare_t4_parallel_lanes.sh" "$root" "$grant_id" \
    "$deployment_ref" "$dgx_a_root" "$isaac_a_root" "$dgx_b_root" "$isaac_b_root" \
    "$result_dir/parallel_lane_prepare.json"

python3 - "$root/$passed_rel" "$result_dir/memory_admission_reuse.json" \
  "$passed_rel" "$authorization_ref" <<'PY'
import json,sys,time
from pathlib import Path
source=Path(sys.argv[1]); memory=json.loads((source/"memory_admission.json").read_text())
batch=json.loads((source/"batch_status.json").read_text())
arms=batch.get("arms") or {}; lanes=batch.get("lane_exit_codes") or {}
checks=(memory.get("status")=="PASS", memory.get("second_isaac_admitted") is True,
        memory.get("memory_probe_ok") is True, int(lanes.get("a",-1))==0,
        arms.get("recovery_on")=="PASS", arms.get("recovery_off")=="PASS",
        batch.get("strict_evidence_modified") is False, batch.get("real_go2_targeted") is False)
if not all(checks): raise SystemExit("passed parallel result is not reusable")
payload={"schema_version":1,"status":"PASS","source_result":sys.argv[3],
"source_memory_admission":memory,"reused_lane_a_variants":["recovery_on","recovery_off"],
"authorization_ref_sha":sys.argv[4],"second_isaac_runs_alone_in_resume":True,
"strict_evidence_modified":False,"real_go2_targeted":False,"recorded_unix":time.time()}
Path(sys.argv[2]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY

set +e
bash "$root/scripts/with_resource_lease.sh" lane-b --owner codex-00 \
  --task "00-t4-ablation-view-only-${grant_id}" --log-dir "$result_dir/leases/lane-b" -- \
  bash "$root/coordination/run_t4_migrated_ablation_parallel_resume_online.sh" \
    _lane b "$root" "$authorization_ref" "$deployment_ref" \
    "$dgx_b_root" "$isaac_b_root" "$result_rel" "$prepare_rel" "$prerequisite_rel" \
    h1_view:13 go2_view:14
lane_b_rc=$?
set -e

python3 - "$result_dir/batch_status.json" "$grant_id" "$authorization_ref" "$lane_b_rc" <<'PY'
import json,sys,time
from pathlib import Path
path=Path(sys.argv[1]); b=int(sys.argv[4]); root=path.parent; arms={}
for arm in sorted((root/"arms").glob("*")):
    summary=next(arm.glob("migrated_ablation20_*_summary.json"),None)
    arms[arm.name]=json.loads(summary.read_text()).get("status","UNKNOWN") if summary else "MISSING"
payload={"schema_version":1,"status":"ONLINE_PASS_EVIDENCE_PENDING" if b==0 else "FAIL",
"grant_id":sys.argv[2],"authorization_ref_sha":sys.argv[3],
"lane_exit_codes":{"a_reused":0,"b":b},"arms":arms,"failed_lane_count":int(b!=0),
"strict_evidence_modified":False,"real_go2_targeted":False,"recorded_unix":time.time()}
path.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
(( lane_b_rc == 0 )) || exit 1

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
