#!/usr/bin/env bash
set -euo pipefail

readonly COORDINATION_REF="refs/heads/codex/parallel-integration"
readonly REMOTE_ROOT="/home/song/internnav-t1-t2"
readonly REMOTE_STAGE_PARENT="/home/song/.codex-internnav-stage"

usage() {
  cat >&2 <<'EOF'
Usage:
  run_01r_online.sh <bootstrap|soak|completion_sim|completion_sim_map> <grant-id> <expected-ref-sha>

The authoritative TASK_BOARD grant must already match the derived result path.
Authentication is read only from ISAAC_PASSWORD_FILE/ISAAC_PASSWORD or an
explicit repo-external ISAAC_EXEC_HELPER/ISAAC_PUT_HELPER/ISAAC_GET_HELPER triplet.
EOF
  exit 64
}

die() {
  printf '01R coordinator runner: %s\n' "$*" >&2
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

sshpass_prefix() {
  SSH_AUTH=()
  if [[ -n "${ISAAC_PASSWORD_FILE:-}" ]]; then
    [[ -r "$ISAAC_PASSWORD_FILE" ]] || die "ISAAC_PASSWORD_FILE is unreadable"
    SSH_AUTH=(sshpass -f "$ISAAC_PASSWORD_FILE")
  elif [[ -n "${ISAAC_PASSWORD:-}" ]]; then
    export SSHPASS="$ISAAC_PASSWORD"
    SSH_AUTH=(sshpass -e)
  fi
}

run_under_lease() {
  [[ $# -eq 7 ]] || die "invalid internal invocation"
  local root="$1" local_stage="$2" profile="$3" grant_id="$4"
  local expected_sha="$5" result_rel="$6" lease_log="$7"
  local host="${ISAAC_HOST:-10.100.120.111}"
  local user="${ISAAC_USER:-song}"
  local port="${ISAAC_PORT:-22}"
  local target="${user}@${host}"
  local remote_stage="${REMOTE_STAGE_PARENT}/01r-${profile}-${grant_id}"
  local remote_result="${REMOTE_ROOT}/${result_rel}"
  local remote_rc=0 collect_rc=0
  local -a ssh_options

  local transport="ssh"
  if internnav_helper_requested; then
    internnav_helper_init || die "external helper transport initialization failed"
    transport="helper"
  else
    sshpass_prefix
    command -v ssh >/dev/null 2>&1 || die "ssh is unavailable"
    command -v scp >/dev/null 2>&1 || die "scp is unavailable"
    ssh_options=(-T -p "$port" -o ConnectTimeout=8 -o ServerAliveInterval=5
      -o ServerAliveCountMax=2 -o StrictHostKeyChecking=accept-new)
  fi

  remote_exec() {
    if [[ "$transport" == helper ]]; then
      internnav_helper_exec "$1"
    else
      "${SSH_AUTH[@]}" ssh "${ssh_options[@]}" "$target" "$1"
    fi
  }
  remote_put() {
    local remote_dir="$1"
    shift
    if [[ "$transport" == helper ]]; then
      internnav_helper_put "$remote_dir" "$@"
    else
      "${SSH_AUTH[@]}" scp -P "$port" -o ConnectTimeout=8 \
        -o StrictHostKeyChecking=accept-new "$@" "$target:$remote_dir/"
    fi
  }
  remote_get() {
    if [[ "$transport" == helper ]]; then
      internnav_helper_get "$1" "$2"
    else
      "${SSH_AUTH[@]}" scp -P "$port" -o ConnectTimeout=8 \
        -o StrictHostKeyChecking=accept-new "$target:$1" "$2"
    fi
  }

  printf -v remote_command \
    'test "$(id -un)" = song && ip -4 -o addr show | grep -Fq %q && test ! -e %q && install -d -m 700 %q' \
    " 10.100.120.111/" "$remote_stage" "$remote_stage"
  remote_exec "$remote_command"

  remote_put "$remote_stage" \
    "$local_stage/payload.tar" \
    "$local_stage/integration.bundle" \
    "$local_stage/SHA256SUMS" \
    "$root/coordination/remote_01r_deploy_and_run.sh"

  printf -v remote_command 'bash %q %q %q %q %q %q %q' \
    "$remote_stage/remote_01r_deploy_and_run.sh" \
    "$remote_stage" "$REMOTE_ROOT" "$profile" "$result_rel" \
    "$grant_id" "$expected_sha"
  set +e
  remote_exec "$remote_command"
  remote_rc=$?
  set -e

  # Collection remains inside the same lease and runs even after an online
  # failure.  The result directory is never created by this collector.
  printf -v remote_command \
    'set -e; if test -d %q; then tar -C %q -czf %q %q; fi; test -f %q' \
    "$remote_result" "$REMOTE_ROOT" "$remote_stage/result.tgz" \
    "$result_rel" "$remote_stage/deployment_receipt.txt"
  set +e
  remote_exec "$remote_command"
  collect_rc=$?
  mkdir -p -- "$lease_log/remote-artifacts"
  remote_get "$remote_stage/deployment_receipt.txt" \
    "$lease_log/remote-artifacts/deployment_receipt.txt" || collect_rc=$?
  if remote_exec "test -f '$remote_stage/result.tgz'"; then
    remote_get "$remote_stage/result.tgz" \
      "$lease_log/remote-artifacts/result.tgz" || collect_rc=$?
    if [[ -f "$lease_log/remote-artifacts/result.tgz" ]]; then
      [[ ! -e "$root/$result_rel" ]] || collect_rc=74
      if (( collect_rc == 0 )); then
        tar -C "$root" -xzf "$lease_log/remote-artifacts/result.tgz" || collect_rc=$?
      fi
    fi
  fi
  set -e

  (( collect_rc == 0 )) || {
    printf '01R coordinator runner: evidence collection failed rc=%s\n' "$collect_rc" >&2
    return "$collect_rc"
  }
  return "$remote_rc"
}

if [[ "${1:-}" == "_under_lease" ]]; then
  shift
  run_under_lease "$@"
  exit $?
fi

[[ $# -eq 3 ]] || usage
profile="$1"
grant_id="$2"
expected_sha="$3"
case "$profile" in bootstrap|soak|completion_sim|completion_sim_map) ;; *) usage ;; esac
[[ "$grant_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{7,95}$ ]] || die "unsafe grant-id"
[[ "$expected_sha" =~ ^[0-9a-f]{40}$ ]] || die "expected ref must be a 40-character SHA"

script_path="$(readlink -f "${BASH_SOURCE[0]}")"
root="$(readlink -f "$(dirname "$script_path")/..")"
git_dir="$(resolve_git_dir "$root")"
[[ -d "$git_dir" ]] || die "resolved Git dir does not exist"
# Status must match the Windows checkout filter, while every deployable archive
# byte must come directly from Git objects without CRLF conversion.
GIT_STATUS=(git -c core.autocrlf=true --git-dir="$git_dir" --work-tree="$root")
GIT_OBJECTS=(git -c core.autocrlf=false --git-dir="$git_dir" --work-tree="$root")
actual_sha="$("${GIT_OBJECTS[@]}" rev-parse "$COORDINATION_REF")"
[[ "$actual_sha" == "$expected_sha" ]] || die "coordination ref does not equal expected SHA"
[[ -z "$("${GIT_STATUS[@]}" status --porcelain --untracked-files=no)" ]] || \
  die "tracked integration tree is dirty"

if [[ "$profile" == "completion_sim_map" ]]; then
  result_rel="results/parallel/t4_map/online-smoke-10-${grant_id}"
  lease_rel="results/parallel/t4_map/lease-smoke-10-${grant_id}"
  grant_worker="10"
else
  result_rel="results/parallel/sensor_producer/online-${profile}-01r-${grant_id}"
  lease_rel="results/parallel/sensor_producer/lease-${profile}-01r-${grant_id}"
  grant_worker="01R"
fi
[[ ! -e "$root/$result_rel" ]] || die "local result path already exists"
[[ ! -e "$root/$lease_rel" ]] || die "local lease path already exists"

PYTHONDONTWRITEBYTECODE=1 python3 - "$root" "$git_dir" "$expected_sha" "$profile" \
  "$grant_id" "$result_rel" "$grant_worker" <<'PY'
import json
import re
import subprocess
import sys

root, git_dir, ref_sha, profile, grant_id, result_dir, grant_worker = sys.argv[1:]
document = subprocess.run(
    [
        "git",
        f"--git-dir={git_dir}",
        f"--work-tree={root}",
        "show",
        f"{ref_sha}:coordination/TASK_BOARD.md",
    ],
    cwd=root,
    check=True,
    capture_output=True,
    text=True,
).stdout
matches = re.findall(
    r"<!--\s*INTERNAV_ONLINE_GRANT_V1\s*\r?\n(.*?)\r?\nINTERNAV_ONLINE_GRANT_V1\s*-->",
    document,
    flags=re.DOTALL,
)
if len(matches) != 1:
    raise SystemExit("authoritative grant block count is not one")
payload = json.loads(matches[0])
expected = {
    "schema_version": 1,
    "status": "GRANTED",
    "worker": grant_worker,
    "resource": "isaac",
    "profile": profile,
    "result_dir": result_dir,
    "grant_id": grant_id,
}
if payload != expected:
    raise SystemExit(f"authoritative grant mismatch: {payload!r}")
PY

local_stage="$(mktemp -d "${TMPDIR:-/tmp}/internnav-01r-${profile}.XXXXXX")"
cleanup_stage() { rm -rf -- "$local_stage"; }
trap cleanup_stage EXIT INT TERM HUP

"${GIT_OBJECTS[@]}" archive --format=tar --output="$local_stage/payload.tar" \
  "$expected_sha" -- sensor_runtime scripts go2_sensor_bridge coordination \
  t4_completion configs
"${GIT_OBJECTS[@]}" bundle create "$local_stage/integration.bundle" "$COORDINATION_REF"

# Verify the complete payload against raw ref blob IDs before any resource
# lease is requested.  This catches checkout filters and archive transforms.
PYTHONDONTWRITEBYTECODE=1 python3 \
  "$root/coordination/validate_payload_archive.py" \
  --git-dir "$git_dir" \
  --ref-sha "$expected_sha" \
  --archive "$local_stage/payload.tar"

bundle_head="$("${GIT_OBJECTS[@]}" bundle list-heads "$local_stage/integration.bundle" \
  "$COORDINATION_REF" | awk 'NR == 1 {print $1}')"
[[ "$bundle_head" == "$expected_sha" ]] || die "bundle ref does not equal expected SHA"
(
  cd "$local_stage"
  sha256sum payload.tar integration.bundle > SHA256SUMS
  sha256sum "$root/coordination/remote_01r_deploy_and_run.sh" \
    | sed 's#  .*/#  #' >> SHA256SUMS
)

# The lease holder is itself a long-lived SSH session.  When the explicitly
# configured external helper transport is in use, route that holder through
# the same repository-external connect() implementation before asking for the
# remote flock.  No helper means the ordinary ssh/key/password path remains.
if internnav_helper_requested; then
  internnav_helper_init || die "external helper transport initialization failed"
  export LEASE_SSH_BIN="$root/coordination/lease_paramiko_ssh.sh"
fi

bash "$root/scripts/with_resource_lease.sh" isaac \
  --owner codex-00 \
  --task "${grant_worker}-model-free-${profile}-${grant_id}" \
  --log-dir "$root/$lease_rel" \
  -- bash "$script_path" _under_lease "$root" "$local_stage" "$profile" \
    "$grant_id" "$expected_sha" "$result_rel" "$root/$lease_rel"
