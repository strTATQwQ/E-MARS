#!/usr/bin/env bash
set -euo pipefail

readonly COORDINATION_REF=refs/heads/codex/parallel-integration
readonly VARIANTS=(
  full_system1_system2 oracle_high_level_system1 system2_oracle_local_path
  full_trajectory endpoint straight_line model_stop oracle_termination
  history_on history_off recovery_on recovery_off h1_view go2_view
)

die() { echo "T4 migrated ablation matrix: $*" >&2; exit 64; }
usage() {
  echo "usage: run_t4_migrated_ablation_matrix_online.sh GRANT AUTH_SHA PREPARE_RESULT_REL SELECTED_RECOVERY_RESULT_REL [RESUME_RESULT_REL]" >&2
  exit 64
}

resolve_git_dir() {
  local root="$1" raw
  if [[ -d "$root/.git" ]]; then readlink -f "$root/.git"; return; fi
  [[ -f "$root/.git" ]] || die "worktree has no git pointer"
  raw="$(sed -n 's/^gitdir: //p' "$root/.git")"
  case "$raw" in
    [A-Za-z]:/*|[A-Za-z]:\\*) wslpath -u "$raw" ;;
    /*) printf '%s\n' "$raw" ;;
    *) readlink -f "$root/$raw" ;;
  esac
}

run_batch() {
  [[ $# -eq 12 ]] || die "invalid internal batch invocation"
  local root="$1" grant_id="$2" authorization_ref="$3" deployment_ref="$4"
  local dgx_root="$5" isaac_root="$6" result_rel="$7" lease_rel="$8"
  local prepare_rel="$9" prerequisite_rel="${10}" migration_runner="${11}"
  local resume_rel="${12}"
  local failures=0 index=0 variant arm_grant arm_result profile rc
  if [[ -n "$resume_rel" ]]; then
    local runner_commit_sha
    runner_commit_sha="$(
      git --git-dir="$git_dir" --work-tree="$root" log -1 --format=%H "$authorization_ref" -- \
        scripts/build_t4_ablation_episode_records.py \
        coordination/run_t4_dgx_migration_online.sh \
        coordination/remote_t4_isaac_migration_session.sh \
        coordination/remote_t4_dgx_onboard_session.sh \
        coordination/remote_t4_model_session.sh
    )"
    python3 "$root/scripts/prepare_t4_ablation_resume.py" \
      --source-result "$root/$resume_rel" --destination-result "$root/$result_rel" \
      --runner-commit-sha "$runner_commit_sha" \
      >"$root/${result_rel}.resume_prepare.log" 2>&1
  fi
  for variant in "${VARIANTS[@]}"; do
    index=$((index + 1))
    printf -v arm_grant '%s-a%02d' "$grant_id" "$index"
    arm_result="$result_rel/arms/$variant"
    profile="migrated_ablation20_${variant}"
    if [[ -d "$root/$arm_result" && -n "$resume_rel" ]]; then
      continue
    fi
    test ! -e "$root/$arm_result" || die "arm result already exists: $arm_result"
    set +e
    bash "$migration_runner" _under_lease "$root" "$arm_grant" \
      "$authorization_ref" "$deployment_ref" "$dgx_root" "$isaac_root" \
      "$arm_result" "$lease_rel" "$prepare_rel" \
      "$root/coordination/t4_functional_run_contract.py" "$profile" "$prerequisite_rel"
    rc=$?
    set -e
    if (( rc != 0 )); then
      failures=$((failures + 1))
      # Every later arm depends on the same uncontaminated hosts.  Stop the
      # batch at the first failure so a cleanup or residual fault cannot leak
      # into nominally independent evidence.
      break
    fi
  done

  python3 - "$root/$result_rel" "$grant_id" "$authorization_ref" "$failures" <<'PY'
import json,sys,time
from pathlib import Path
root=Path(sys.argv[1]); failures=int(sys.argv[4]); arms={}
for path in sorted((root/"arms").glob("*")) if (root/"arms").is_dir() else []:
    summary=next(path.glob("migrated_ablation20_*_summary.json"),None)
    status="MISSING"
    if summary is not None:
        try: status=json.loads(summary.read_text(encoding="utf-8")).get("status","UNKNOWN")
        except (OSError,json.JSONDecodeError): status="INVALID"
    arms[path.name]=status
payload={"schema_version":1,"status":"PASS" if failures==0 else "FAIL",
"grant_id":sys.argv[2],"authorization_ref_sha":sys.argv[3],
"failed_arm_count":failures,"arms":arms,"recorded_unix":time.time(),
"strict_evidence_modified":False,"real_go2_targeted":False}
(root/"batch_status.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
PY
  (( failures == 0 )) || return 1
  set +e
  python3 "$root/scripts/finalize_t4_migrated_ablation.py" \
    --result-dir "$root/$result_rel" >"$root/$result_rel/finalize.log" 2>&1
  local finalize_rc=$?
  set -e
  python3 - "$root/$result_rel/batch_status.json" "$finalize_rc" <<'PY'
import json,sys,time
from pathlib import Path
path=Path(sys.argv[1]); rc=int(sys.argv[2]); value=json.loads(path.read_text(encoding="utf-8"))
value["finalize_exit_code"]=rc
value["status"]="PASS" if rc==0 else "ONLINE_PASS_EVIDENCE_PENDING"
value["recorded_unix"]=time.time()
path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="utf-8")
PY
  return "$finalize_rc"
}

if [[ "${1:-}" == _under_lease ]]; then shift; run_batch "$@"; exit $?; fi
[[ $# -ge 4 && $# -le 5 ]] || usage
grant_id="$1"
authorization_ref="$2"
prepare_rel="$3"
prerequisite_rel="$4"
resume_rel="${5:-}"
[[ "$grant_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{7,95}$ ]] || die "unsafe grant"
[[ "$authorization_ref" =~ ^[0-9a-f]{40}$ ]] || die "authorization SHA must be full"
[[ "$prepare_rel" =~ ^results/parallel/t4_functional/prepare-00-[A-Za-z0-9][A-Za-z0-9._-]{7,95}$ ]] || die "unsafe prepare path"
[[ "$prerequisite_rel" =~ ^results/parallel/t4_functional/migrated-recovery-a5-00-[A-Za-z0-9][A-Za-z0-9._-]{7,95}$ ]] || die "unsafe selected recovery prerequisite"
if [[ -n "$resume_rel" ]]; then
  [[ "$resume_rel" =~ ^results/parallel/t4_functional/migrated-ablation-matrix-00-[A-Za-z0-9][A-Za-z0-9._-]{7,95}$ ]] || die "unsafe resume result path"
fi

script_path="$(readlink -f "${BASH_SOURCE[0]}")"
root="$(readlink -f "$(dirname "$script_path")/..")"
git_dir="$(resolve_git_dir "$root")"
contract_tool="$root/coordination/t4_functional_run_contract.py"
migration_runner="$root/coordination/run_t4_dgx_migration_online.sh"
actual_sha="$(git -c core.autocrlf=false --git-dir="$git_dir" rev-parse "$COORDINATION_REF")"
test "$actual_sha" = "$authorization_ref"
test -z "$(git -c core.autocrlf=true --git-dir="$git_dir" --work-tree="$root" status --porcelain --untracked-files=no)"
mapfile -t fields < <(python3 "$contract_tool" prepare-fields \
  --result-dir "$root/$prepare_rel" --git-dir "$git_dir" \
  --authorization-ref "$authorization_ref" --format pilot-lines)
test "${#fields[@]}" -eq 6
deployment_ref="${fields[1]}"
dgx_root="${fields[2]}"
isaac_root="${fields[4]}"
python3 - "$root/$prerequisite_rel/migrated_recovery_a5_summary.json" <<'PY'
import json,sys
value=json.load(open(sys.argv[1],encoding="utf-8"))
checks=(value.get("status")=="PASS",value.get("run_profile")=="migrated_recovery_a5",
        int(value.get("expected_episode_count",-1))==5,
        bool((value.get("checks") or {}).get("recovery_functional")),
        value.get("real_go2_targeted") is False)
if not all(checks): raise SystemExit("selected migrated recovery A prerequisite is not PASS")
PY

result_rel="results/parallel/t4_functional/migrated-ablation-matrix-00-${grant_id}"
lease_rel="results/parallel/t4_functional/lease-migrated-ablation-matrix-00-${grant_id}"
test ! -e "$root/$result_rel"
test ! -e "$root/$lease_rel"
python3 - "$root" "$git_dir" "$authorization_ref" "$grant_id" "$result_rel" <<'PY'
import json,re,subprocess,sys
root,git_dir,ref,grant,result=sys.argv[1:]
doc=subprocess.run(["git",f"--git-dir={git_dir}",f"--work-tree={root}","show",f"{ref}:coordination/TASK_BOARD.md"],check=True,capture_output=True,text=True).stdout
blocks=re.findall(r"<!--\s*INTERNAV_ONLINE_GRANT_V1\s*\r?\n(.*?)\r?\nINTERNAV_ONLINE_GRANT_V1\s*-->",doc,flags=re.DOTALL)
expected={"schema_version":1,"status":"GRANTED","worker":"00","resource":"dgx+isaac","profile":"dgx_onboard_ablation20_matrix","result_dir":result,"grant_id":grant}
if len(blocks)!=1 or json.loads(blocks[0])!=expected: raise SystemExit("migrated ablation matrix grant mismatch")
PY
bash "$root/scripts/with_resource_lease.sh" both --owner codex-00 \
  --task "00-dgx-onboard-ablation20-matrix-${grant_id}" \
  --log-dir "$root/$lease_rel" -- \
  bash "$script_path" _under_lease "$root" "$grant_id" "$authorization_ref" \
    "$deployment_ref" "$dgx_root" "$isaac_root" "$result_rel" "$lease_rel" \
    "$prepare_rel" "$prerequisite_rel" "$migration_runner" "$resume_rel"
