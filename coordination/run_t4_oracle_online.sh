#!/usr/bin/env bash
set -euo pipefail

readonly COORDINATION_REF="refs/heads/codex/parallel-integration"
readonly ISAAC_ROOT=/home/song/internnav-t1-t2
readonly ISAAC_STAGE_PARENT=/home/song/.codex-internnav-stage

usage() {
  cat >&2 <<'EOF'
Usage:
  run_t4_oracle_online.sh <grant-id> <authorization-ref-sha> <functional-prepare-result-relative-path>

The functional prepare result must be PASS and its deployed functional code
must differ from the authorization ref only in coordination/TASK_BOARD.md.
The authoritative grant must be worker 00 / isaac / functional_oracle10.
EOF
  exit 64
}

die() {
  printf 'T4 Oracle coordinator: %s\n' "$*" >&2
  exit 64
}

# shellcheck source=remote_helper_transport.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/remote_helper_transport.sh"

resolve_git_dir() {
  local root="$1" raw
  if [[ -d "$root/.git" ]]; then
    readlink -f "$root/.git"
    return
  fi
  [[ -f "$root/.git" ]] || die "worktree has no .git pointer"
  raw="$(sed -n 's/^gitdir: //p' "$root/.git")"
  [[ -n "$raw" ]] || die "worktree .git pointer is malformed"
  case "$raw" in
    [A-Za-z]:/*|[A-Za-z]:\\*) wslpath -u "$raw" ;;
    /*) printf '%s\n' "$raw" ;;
    *) readlink -f "$root/$raw" ;;
  esac
}

configure_direct_transport() {
  ISAAC_HOST_VALUE="${ISAAC_HOST:-10.100.120.111}"
  ISAAC_USER_VALUE="${ISAAC_USER:-song}"
  ISAAC_PORT_VALUE="${ISAAC_PORT:-22}"
  [[ "$ISAAC_HOST_VALUE" =~ ^[A-Za-z0-9_.:-]+$ ]] || die "unsafe Isaac host"
  [[ "$ISAAC_USER_VALUE" =~ ^[A-Za-z0-9_.-]+$ ]] || die "unsafe Isaac user"
  [[ "$ISAAC_PORT_VALUE" =~ ^[0-9]+$ ]] || die "unsafe Isaac port"
  [[ "$ISAAC_HOST_VALUE" == 10.100.120.111 ]] || die "Oracle target is not the frozen Isaac host"
  [[ "$ISAAC_USER_VALUE" == song ]] || die "Oracle user is not the frozen Isaac user"
  ISAAC_TARGET="$ISAAC_USER_VALUE@$ISAAC_HOST_VALUE"
  SSH_AUTH=()
  if [[ -n "${ISAAC_PASSWORD_FILE:-}" ]]; then
    [[ -r "$ISAAC_PASSWORD_FILE" ]] || die "ISAAC_PASSWORD_FILE is unreadable"
    command -v sshpass >/dev/null 2>&1 || die "sshpass is required for password-file transport"
    SSH_AUTH=(sshpass -f "$ISAAC_PASSWORD_FILE")
  elif [[ -n "${ISAAC_PASSWORD:-}" ]]; then
    command -v sshpass >/dev/null 2>&1 || die "sshpass is required for password transport"
    export SSHPASS="$ISAAC_PASSWORD"
    SSH_AUTH=(sshpass -e)
  fi
  SSH_OPTIONS=(
    -o ConnectTimeout=8
    -o ServerAliveInterval=5
    -o ServerAliveCountMax=2
    -o StrictHostKeyChecking=accept-new
  )
  if [[ ${#SSH_AUTH[@]} -eq 0 ]]; then
    SSH_OPTIONS+=(-o BatchMode=yes)
  fi
}

run_under_lease() {
  [[ $# -eq 9 ]] || die "invalid internal invocation"
  local root="$1" grant_id="$2" authorization_ref="$3" deployment_ref="$4"
  local deployment_root="$5" result_rel="$6" lease_rel="$7"
  local prepare_rel="$8" contract_tool="$9"
  local remote_stage="$ISAAC_STAGE_PARENT/t4-oracle-${grant_id}"
  local result_dir="$root/$result_rel"
  local remote_rc=125 collect_rc=0 cleanup_rc=125 local_validation_rc=125
  local transport=ssh remote_command file

  mkdir -p "$result_dir/remote-artifacts" "$result_dir/run"

  if internnav_helper_requested; then
    internnav_helper_init || die "external helper transport initialization failed"
    transport=helper
  else
    configure_direct_transport
    command -v ssh >/dev/null 2>&1 || die "ssh is unavailable"
    command -v scp >/dev/null 2>&1 || die "scp is unavailable"
  fi

  remote_exec() {
    if [[ "$transport" == helper ]]; then
      internnav_helper_exec "$1"
    else
      "${SSH_AUTH[@]}" ssh -T -p "$ISAAC_PORT_VALUE" \
        "${SSH_OPTIONS[@]}" "$ISAAC_TARGET" "$1"
    fi
  }
  remote_get() {
    if [[ "$transport" == helper ]]; then
      internnav_helper_get "$1" "$2"
    else
      "${SSH_AUTH[@]}" scp -P "$ISAAC_PORT_VALUE" \
        "${SSH_OPTIONS[@]}" "$ISAAC_TARGET:$1" "$2"
    fi
  }

  printf -v remote_command \
    'test "$(id -un)" = song && ip -4 -o addr show | grep -Fq %q && test ! -e %q && install -d -m 700 %q' \
    " 10.100.120.111/" "$remote_stage" "$remote_stage"
  remote_exec "$remote_command"

  printf -v remote_command 'bash %q %q %q %q %q %q' \
    "$deployment_root/coordination/remote_t4_oracle_run.sh" \
    "$deployment_root" "$remote_stage" "$grant_id" \
    "$authorization_ref" "$deployment_ref"
  set +e
  remote_exec "$remote_command"
  remote_rc=$?
  set -e

  set +e
  remote_get "$remote_stage/oracle_run_receipt.json" \
    "$result_dir/remote-artifacts/oracle_run_receipt.json"
  collect_rc=$?
  for file in deployment_validation.json payload_reverification.json \
      overlay_source_validation.json oracle_invocation.json oracle_outer.log \
      container_cleanup_primary.log container_cleanup_probe.log residual_probe.json \
      residual_probe.log oracle_validation.log; do
    remote_get "$remote_stage/$file" "$result_dir/remote-artifacts/$file" \
      >/dev/null 2>&1 || true
  done
  if remote_exec "test -f '$remote_stage/oracle_result.tgz'"; then
    remote_get "$remote_stage/oracle_result.tgz" \
      "$result_dir/remote-artifacts/oracle_result.tgz" || collect_rc=$?
  elif (( remote_rc == 0 )); then
    collect_rc=74
  fi
  set -e

  if [[ -f "$result_dir/remote-artifacts/oracle_result.tgz" ]]; then
    set +e
    python3 "$contract_tool" extract-oracle-archive \
      --archive "$result_dir/remote-artifacts/oracle_result.tgz" \
      --output-dir "$result_dir/run" \
      >"$result_dir/archive_extraction.json" 2>"$result_dir/archive_extraction.stderr.log"
    extract_rc=$?
    set -e
    (( extract_rc == 0 )) || collect_rc=$extract_rc
  fi

  if (( collect_rc == 0 )); then
    printf -v remote_command \
      'test %q = %q && test -d %q && test ! -L %q && test "$(realpath %q)" = %q && rm -rf -- %q' \
      "$remote_stage" "$ISAAC_STAGE_PARENT/t4-oracle-${grant_id}" \
      "$remote_stage" "$remote_stage" "$remote_stage" "$remote_stage" "$remote_stage"
    set +e
    remote_exec "$remote_command"
    cleanup_rc=$?
    set -e
  fi

  set +e
  python3 "$contract_tool" validate-collected \
    --result-dir "$result_dir/run" \
    --receipt "$result_dir/remote-artifacts/oracle_run_receipt.json" \
    --output "$result_dir/oracle10_summary.json" \
    --grant-id "$grant_id" \
    --authorization-ref "$authorization_ref" \
    --deployment-ref "$deployment_ref" \
    --remote-stage-cleanup "$([[ $cleanup_rc == 0 ]] && printf PASS || printf FAIL)" \
    >"$result_dir/local_validation.log" 2>&1
  local_validation_rc=$?
  set -e
  if (( local_validation_rc != 0 )); then
    python3 - "$result_dir/oracle10_summary.json" "$grant_id" \
      "$authorization_ref" "$deployment_ref" "$prepare_rel" "$remote_rc" \
      "$collect_rc" "$cleanup_rc" "$local_validation_rc" <<'PY'
import json,sys,time
from pathlib import Path
output=Path(sys.argv[1])
if not output.exists():
    payload={
        "schema_version":1,
        "status":"FAIL",
        "grant_id":sys.argv[2],
        "authorization_ref_sha":sys.argv[3],
        "deployment_ref_sha":sys.argv[4],
        "functional_prepare_result":sys.argv[5],
        "exit_codes":{
            "remote":int(sys.argv[6]),
            "collection":int(sys.argv[7]),
            "remote_stage_cleanup":int(sys.argv[8]),
            "local_validation":int(sys.argv[9]),
        },
        "runtime_policy":"completion_sim",
        "runtime_target":"isaac_simulation",
        "model_host":"dgx_spark_only",
        "model_process_started":False,
        "strict_evidence_modified":False,
        "real_go2_targeted":False,
        "recorded_unix":time.time(),
    }
    output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
PY
  fi

  if (( collect_rc != 0 )); then
    return "$collect_rc"
  fi
  if (( cleanup_rc != 0 )); then
    return "$cleanup_rc"
  fi
  if (( local_validation_rc != 0 )); then
    return "$local_validation_rc"
  fi
  return "$remote_rc"
}

if [[ "${1:-}" == _under_lease ]]; then
  shift
  run_under_lease "$@"
  exit $?
fi

[[ $# -eq 3 ]] || usage
grant_id="$1"
authorization_ref="$2"
prepare_rel="$3"
[[ "$grant_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{7,95}$ ]] || die "unsafe grant ID"
[[ "$authorization_ref" =~ ^[0-9a-f]{40}$ ]] || die "authorization ref must be a full SHA"
[[ "$prepare_rel" =~ ^results/parallel/t4_functional/prepare-00-[A-Za-z0-9][A-Za-z0-9._-]{7,95}$ ]] || \
  die "unsafe functional prepare result path"

script_path="$(readlink -f "${BASH_SOURCE[0]}")"
root="$(readlink -f "$(dirname "$script_path")/..")"
git_dir="$(resolve_git_dir "$root")"
contract_tool="$root/coordination/t4_functional_run_contract.py"
GIT_STATUS=(git -c core.autocrlf=true --git-dir="$git_dir" --work-tree="$root")
GIT_OBJECTS=(git -c core.autocrlf=false --git-dir="$git_dir" --work-tree="$root")
actual_sha="$("${GIT_OBJECTS[@]}" rev-parse "$COORDINATION_REF")"
[[ "$actual_sha" == "$authorization_ref" ]] || die "coordination ref does not equal authorization SHA"
[[ -z "$("${GIT_STATUS[@]}" status --porcelain --untracked-files=no)" ]] || \
  die "tracked integration tree is dirty"

prepare_lines="$(python3 "$contract_tool" prepare-fields \
  --result-dir "$root/$prepare_rel" \
  --git-dir "$git_dir" \
  --authorization-ref "$authorization_ref" \
  --format lines)"
mapfile -t prepare_fields <<<"$prepare_lines"
[[ ${#prepare_fields[@]} -eq 4 ]] || die "functional prepare fields are incomplete"
prepare_grant="${prepare_fields[0]}"
deployment_ref="${prepare_fields[1]}"
deployment_root="${prepare_fields[2]}"
isaac_ros_workspace="${prepare_fields[3]}"
[[ "$deployment_root" == "$ISAAC_ROOT/.t4-deployments/${prepare_grant}-${deployment_ref:0:12}-isaac" ]] || \
  die "functional deployment root changed"
[[ "$isaac_ros_workspace" == /home/song/internnav-t4/isaac_ros_ws_45 ]] || \
  die "Isaac ROS workspace changed"

result_rel="results/parallel/t4_functional/oracle10-00-${grant_id}"
lease_rel="results/parallel/t4_functional/lease-oracle10-00-${grant_id}"
[[ ! -e "$root/$result_rel" && ! -e "$root/$lease_rel" ]] || \
  die "Oracle result paths must be fresh"

python3 - "$root" "$git_dir" "$authorization_ref" "$grant_id" "$result_rel" <<'PY'
import json,re,subprocess,sys
root,git_dir,ref_sha,grant_id,result_dir=sys.argv[1:]
document=subprocess.run(
    ["git",f"--git-dir={git_dir}",f"--work-tree={root}","show",f"{ref_sha}:coordination/TASK_BOARD.md"],
    check=True,capture_output=True,text=True,
).stdout
matches=re.findall(r"<!--\s*INTERNAV_ONLINE_GRANT_V1\s*\r?\n(.*?)\r?\nINTERNAV_ONLINE_GRANT_V1\s*-->",document,flags=re.DOTALL)
if len(matches) != 1:
    raise SystemExit("authoritative grant block count is not one")
expected={
    "schema_version":1,
    "status":"GRANTED",
    "worker":"00",
    "resource":"isaac",
    "profile":"functional_oracle10",
    "result_dir":result_dir,
    "grant_id":grant_id,
}
if json.loads(matches[0]) != expected:
    raise SystemExit("authoritative functional Oracle grant mismatch")
PY

# Authentication remains process-only or repository-external.  Never add a
# secret to argv, lease metadata, results, receipts, or the deployed tree.
if internnav_helper_requested; then
  internnav_helper_init || die "external helper transport initialization failed"
  export LEASE_SSH_BIN="$root/coordination/lease_paramiko_ssh.sh"
else
  configure_direct_transport
fi

bash "$root/scripts/with_resource_lease.sh" isaac \
  --owner codex-00 \
  --task "00-functional-oracle10-${grant_id}" \
  --log-dir "$root/$lease_rel" \
  -- bash "$script_path" _under_lease "$root" "$grant_id" \
    "$authorization_ref" "$deployment_ref" "$deployment_root" \
    "$result_rel" "$lease_rel" "$prepare_rel" "$contract_tool"
