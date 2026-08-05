#!/usr/bin/env bash
set -euo pipefail

readonly COORDINATION_REF="refs/heads/codex/parallel-integration"

usage() {
  cat >&2 <<'EOF'
Usage:
  run_t4_functional_prepare.sh <grant-id> <expected-ref-sha> <f1-result-relative-path>

The F1 result must be a completed completion_sim_map run.  The authoritative
TASK_BOARD grant must be worker 00 / dgx+isaac / functional_prepare.  Both
remote locks are held in fixed DGX-then-Isaac order for staging and builds.
EOF
  exit 64
}

die() {
  printf 'T4 functional prepare: %s\n' "$*" >&2
  exit 64
}

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

configure_role() {
  local role="$1"
  case "$role" in
    dgx)
      REMOTE_HOST="${DGX_HOST:-10.100.100.128}"
      REMOTE_USER="${DGX_USER:-railgun}"
      REMOTE_PORT="${DGX_PORT:-22}"
      REMOTE_PASSWORD_FILE="${DGX_PASSWORD_FILE:-}"
      REMOTE_PASSWORD="${DGX_PASSWORD:-}"
      REMOTE_ROOT=/home/railgun/internnav-t1-t2
      ;;
    isaac)
      REMOTE_HOST="${ISAAC_HOST:-10.100.120.111}"
      REMOTE_USER="${ISAAC_USER:-song}"
      REMOTE_PORT="${ISAAC_PORT:-22}"
      REMOTE_PASSWORD_FILE="${ISAAC_PASSWORD_FILE:-}"
      REMOTE_PASSWORD="${ISAAC_PASSWORD:-}"
      REMOTE_ROOT=/home/song/internnav-t1-t2
      ;;
    *) die "unknown remote role" ;;
  esac
  [[ "$REMOTE_HOST" =~ ^[A-Za-z0-9_.:-]+$ ]] || die "unsafe host"
  [[ "$REMOTE_USER" =~ ^[A-Za-z0-9_.-]+$ ]] || die "unsafe user"
  [[ "$REMOTE_PORT" =~ ^[0-9]+$ ]] || die "unsafe port"
  if [[ -n "$REMOTE_PASSWORD_FILE" ]]; then
    [[ -r "$REMOTE_PASSWORD_FILE" ]] || die "$role password file is unreadable"
  fi
  if [[ -n "$REMOTE_PASSWORD_FILE$REMOTE_PASSWORD" ]]; then
    command -v sshpass >/dev/null 2>&1 || die "sshpass is required for password transport"
  fi
  REMOTE_TARGET="$REMOTE_USER@$REMOTE_HOST"
  SSH_OPTIONS=(
    -o ConnectTimeout=8
    -o ServerAliveInterval=5
    -o ServerAliveCountMax=2
    -o StrictHostKeyChecking=accept-new
  )
  if [[ -z "$REMOTE_PASSWORD_FILE$REMOTE_PASSWORD" ]]; then
    SSH_OPTIONS+=(-o BatchMode=yes)
  fi
}

remote_exec() {
  local role="$1" command="$2"
  configure_role "$role"
  if [[ -n "$REMOTE_PASSWORD_FILE" ]]; then
    sshpass -f "$REMOTE_PASSWORD_FILE" ssh -T -p "$REMOTE_PORT" \
      "${SSH_OPTIONS[@]}" "$REMOTE_TARGET" "$command"
  elif [[ -n "$REMOTE_PASSWORD" ]]; then
    SSHPASS="$REMOTE_PASSWORD" sshpass -e ssh -T -p "$REMOTE_PORT" \
      "${SSH_OPTIONS[@]}" "$REMOTE_TARGET" "$command"
  else
    ssh -T -p "$REMOTE_PORT" "${SSH_OPTIONS[@]}" "$REMOTE_TARGET" "$command"
  fi
}

remote_put() {
  local role="$1" remote_dir="$2"
  shift 2
  configure_role "$role"
  if [[ -n "$REMOTE_PASSWORD_FILE" ]]; then
    sshpass -f "$REMOTE_PASSWORD_FILE" scp -P "$REMOTE_PORT" \
      "${SSH_OPTIONS[@]}" "$@" "$REMOTE_TARGET:$remote_dir/"
  elif [[ -n "$REMOTE_PASSWORD" ]]; then
    SSHPASS="$REMOTE_PASSWORD" sshpass -e scp -P "$REMOTE_PORT" \
      "${SSH_OPTIONS[@]}" "$@" "$REMOTE_TARGET:$remote_dir/"
  else
    scp -P "$REMOTE_PORT" "${SSH_OPTIONS[@]}" "$@" \
      "$REMOTE_TARGET:$remote_dir/"
  fi
}

remote_get() {
  local role="$1" remote_path="$2" local_path="$3"
  configure_role "$role"
  if [[ -n "$REMOTE_PASSWORD_FILE" ]]; then
    sshpass -f "$REMOTE_PASSWORD_FILE" scp -P "$REMOTE_PORT" \
      "${SSH_OPTIONS[@]}" "$REMOTE_TARGET:$remote_path" "$local_path"
  elif [[ -n "$REMOTE_PASSWORD" ]]; then
    SSHPASS="$REMOTE_PASSWORD" sshpass -e scp -P "$REMOTE_PORT" \
      "${SSH_OPTIONS[@]}" "$REMOTE_TARGET:$remote_path" "$local_path"
  else
    scp -P "$REMOTE_PORT" "${SSH_OPTIONS[@]}" \
      "$REMOTE_TARGET:$remote_path" "$local_path"
  fi
}

run_under_lease() {
  [[ $# -eq 7 ]] || die "invalid internal invocation"
  local root="$1" local_stage="$2" grant_id="$3" expected_sha="$4"
  local result_rel="$5" lease_log="$6" f1_rel="$7"
  local role remote_stage remote_root remote_command role_rc overall_rc=0
  local result_dir="$root/$result_rel"
  mkdir -p "$result_dir/remote-receipts"

  # Both locks are already held.  Deployment/build order remains DGX first so
  # Isaac is untouched if the model-host preflight cannot be made runnable.
  for role in dgx isaac; do
    configure_role "$role"
    remote_root="$REMOTE_ROOT"
    remote_stage="$(dirname "$remote_root")/.codex-internnav-stage/t4-functional-${grant_id}"
    printf -v remote_command \
      'test ! -e %q && install -d -m 700 %q' "$remote_stage" "$remote_stage"
    set +e
    remote_exec "$role" "$remote_command"
    role_rc=$?
    set -e
    if (( role_rc != 0 )); then
      overall_rc=$role_rc
      break
    fi
    set +e
    remote_put "$role" "$remote_stage" \
      "$local_stage/payload.tar" \
      "$local_stage/payload_manifest.json" \
      "$local_stage/SHA256SUMS"
    role_rc=$?
    set -e
    if (( role_rc != 0 )); then
      overall_rc=$role_rc
      break
    fi
    printf -v remote_command \
      'set -euo pipefail; cd %q; mkdir -m 700 bootstrap; tar -C bootstrap -xf payload.tar coordination/t4_functional_payload.py coordination/remote_t4_functional_stage.sh; bash bootstrap/coordination/remote_t4_functional_stage.sh %q %q %q %q %q' \
      "$remote_stage" "$role" "$remote_stage" "$remote_root" \
      "$grant_id" "$expected_sha"
    set +e
    remote_exec "$role" "$remote_command"
    role_rc=$?
    set -e
    set +e
    remote_get "$role" "$remote_stage/${role}_deployment_receipt.json" \
      "$result_dir/remote-receipts/${role}_deployment_receipt.json"
    receipt_rc=$?
    set -e
    if (( role_rc != 0 || receipt_rc != 0 )); then
      overall_rc=$(( role_rc != 0 ? role_rc : receipt_rc ))
      break
    fi
  done

  python3 - "$result_dir/functional_prepare_summary.json" "$overall_rc" \
    "$grant_id" "$expected_sha" "$f1_rel" \
    "$result_dir/remote-receipts/dgx_deployment_receipt.json" \
    "$result_dir/remote-receipts/isaac_deployment_receipt.json" <<'PY'
import json,sys,time
from pathlib import Path
output=Path(sys.argv[1])
rc=int(sys.argv[2])
receipts={}
for role,path_text in zip(("dgx","isaac"),sys.argv[6:8]):
    path=Path(path_text)
    receipts[role]=json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
passed=rc == 0 and all(
    value is not None and value.get("status") == "PASS"
    for value in receipts.values()
)
payload={
    "schema_version":1,
    "status":"PASS" if passed else "FAIL",
    "exit_code":rc,
    "grant_id":sys.argv[3],
    "ref_sha":sys.argv[4],
    "f1_prerequisite_result":sys.argv[5],
    "resource_order":["dgx","isaac"],
    "runtime_policy":"completion_sim",
    "runtime_target":"isaac_simulation",
    "model_host":"dgx_spark_only",
    "strict_evidence_modified":False,
    "real_go2_targeted":False,
    "receipts":receipts,
    "recorded_unix":time.time(),
}
output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
if not passed and rc == 0:
    raise SystemExit(1)
PY
  return "$overall_rc"
}

if [[ "${1:-}" == _under_lease ]]; then
  shift
  run_under_lease "$@"
  exit $?
fi

[[ $# -eq 3 ]] || usage
grant_id="$1"
expected_sha="$2"
f1_rel="$3"
[[ "$grant_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{7,95}$ ]] || usage
[[ "$expected_sha" =~ ^[0-9a-f]{40}$ ]] || usage
[[ "$f1_rel" =~ ^results/parallel/t4_map/online-smoke-10-[A-Za-z0-9][A-Za-z0-9._-]{7,95}$ ]] || usage

script_path="$(readlink -f "${BASH_SOURCE[0]}")"
root="$(readlink -f "$(dirname "$script_path")/..")"
git_dir="$(resolve_git_dir "$root")"
GIT_STATUS=(git -c core.autocrlf=true --git-dir="$git_dir" --work-tree="$root")
GIT_OBJECTS=(git -c core.autocrlf=false --git-dir="$git_dir" --work-tree="$root")
actual_sha="$("${GIT_OBJECTS[@]}" rev-parse "$COORDINATION_REF")"
[[ "$actual_sha" == "$expected_sha" ]] || die "coordination ref does not equal expected SHA"
[[ -z "$("${GIT_STATUS[@]}" status --porcelain --untracked-files=no)" ]] || \
  die "tracked integration tree is dirty"

f1_dir="$root/$f1_rel"
[[ -d "$f1_dir" && ! -L "$f1_dir" ]] || die "F1 evidence directory is missing or unsafe"
python3 - "$f1_dir" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
completion=json.loads((root/"session_completion.json").read_text(encoding="utf-8"))
validation=json.loads((root/"session_validation.json").read_text(encoding="utf-8"))
runtime=completion.get("map_runtime_validation") or {}
cleanup_bundle=completion.get("map_cleanup_validation") or {}
cleanup=cleanup_bundle.get("cleanup") or {}
map_validation=cleanup_bundle.get("validation") or {}
checks={
    "session_completion":completion.get("status") == "PASS",
    "session_validation":validation.get("status") == "PASS",
    "inner_cleanup":completion.get("inner_cleanup_confirmed") is True,
    "outer_cleanup":completion.get("outer_cleanup_confirmed") is True,
    "isaac_cleanup":completion.get("isaac_cleanup_confirmed") is True,
    "zero_process_residuals":all(int(completion.get(name,-1)) == 0 for name in ("pid_count","pgid_count","socket_count")),
    "map_runtime":runtime.get("status") == "PASS" and runtime.get("continues_until_shared_stop") is True,
    "map_cleanup":cleanup.get("status") == "PASS" and cleanup.get("child_roles_alive") == [] and cleanup.get("descendant_pids_alive") == [] and int(cleanup.get("owned_socket_count",-1)) == 0,
    "map_validation":map_validation.get("status") == "PASS" and map_validation.get("runtime_validation_written") is True,
}
if not all(checks.values()):
    raise SystemExit(f"F1 prerequisite is not PASS: {checks}")
print("F1_PREREQUISITE_PASS")
PY

result_rel="results/parallel/t4_functional/prepare-00-${grant_id}"
lease_rel="results/parallel/t4_functional/lease-prepare-00-${grant_id}"
[[ ! -e "$root/$result_rel" && ! -e "$root/$lease_rel" ]] || die "result paths must be fresh"

python3 - "$root" "$git_dir" "$expected_sha" "$grant_id" "$result_rel" <<'PY'
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
    "resource":"dgx+isaac",
    "profile":"functional_prepare",
    "result_dir":result_dir,
    "grant_id":grant_id,
}
if json.loads(matches[0]) != expected:
    raise SystemExit("authoritative functional prepare grant mismatch")
PY

local_stage="$(mktemp -d "${TMPDIR:-/tmp}/internnav-t4-functional.XXXXXX")"
cleanup_stage() { rm -rf -- "$local_stage"; }
trap cleanup_stage EXIT INT TERM HUP
python3 "$root/coordination/t4_functional_payload.py" build \
  --git-dir "$git_dir" \
  --ref-sha "$expected_sha" \
  --archive "$local_stage/payload.tar" \
  --manifest "$local_stage/payload_manifest.json"
(
  cd "$local_stage"
  sha256sum payload.tar payload_manifest.json >SHA256SUMS
)

# Passwords are consumed only from environment/password files by sshpass and
# never enter argv, the lease metadata, payload, receipts, or result paths.
for role in dgx isaac; do
  configure_role "$role"
done

bash "$root/scripts/with_resource_lease.sh" both \
  --owner codex-00 \
  --task "00-functional-prepare-${grant_id}" \
  --log-dir "$root/$lease_rel" \
  -- bash "$script_path" _under_lease "$root" "$local_stage" "$grant_id" \
    "$expected_sha" "$result_rel" "$root/$lease_rel" "$f1_rel"
