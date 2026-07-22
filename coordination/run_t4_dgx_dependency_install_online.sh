#!/usr/bin/env bash
set -euo pipefail

readonly COORDINATION_REF=refs/heads/codex/parallel-integration
usage() { echo "usage: run_t4_dgx_dependency_install_online.sh GRANT AUTH_SHA" >&2; exit 64; }
die() { echo "DGX dependency coordinator: $*" >&2; exit 64; }

resolve_git_dir() {
  local root="$1" raw
  if [[ -d "$root/.git" ]]; then readlink -f "$root/.git"; return; fi
  raw="$(sed -n 's/^gitdir: //p' "$root/.git")"
  case "$raw" in [A-Za-z]:/*|[A-Za-z]:\\*) wslpath -u "$raw";; /*) printf '%s\n' "$raw";; *) readlink -f "$root/$raw";; esac
}

run_under_lease() {
  [[ $# -eq 6 ]] || die "invalid internal invocation"
  local root="$1" grant="$2" auth="$3" result_rel="$4" lease_rel="$5" installer_sha="$6"
  local result_dir="$root/$result_rel"
  local stage="/home/railgun/.codex-internnav-stage/t4-dgx-dependencies-${grant}"
  local target=railgun@10.100.100.128
  local options=(-o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5
    -o ServerAliveCountMax=2 -o StrictHostKeyChecking=accept-new)
  mkdir -p "$result_dir/remote-artifacts"
  ssh -T "${options[@]}" "$target" \
    "ip -4 -o addr show | grep -Fq ' 10.100.100.128/' && test ! -e '$stage' && install -d -m 700 '$stage'"
  scp "${options[@]}" "$root/scripts/install_t4_dgx_nav2_dependencies.sh" \
    "$target:$stage/install.sh"
  ssh -T "${options[@]}" "$target" \
    "test \"\$(sha256sum '$stage/install.sh' | cut -d' ' -f1)\" = '$installer_sha' && chmod 700 '$stage/install.sh'"

  remote_command="INTERNNAV_T4_RESOURCE_LEASE_ACK=dgx INTERNNAV_RUNTIME_POLICY=completion_sim INTERNNAV_SIMULATION_TARGET=isaac INTERNVLA_T4_DGX_DEPENDENCY_RESULT_DIR='$stage/result' INTERNNAV_DGX_SUDO_STDIN_ACK=1 bash '$stage/install.sh'"
  set +e
  if ssh -T "${options[@]}" "$target" "sudo -n true" >/dev/null 2>&1; then
    ssh -T "${options[@]}" "$target" "$remote_command" </dev/null
    remote_rc=$?
  elif [[ -n "${DGX_SUDO_PASSWORD_FILE:-}" ]]; then
    test -r "$DGX_SUDO_PASSWORD_FILE" || die "DGX_SUDO_PASSWORD_FILE is unreadable"
    ssh -T "${options[@]}" "$target" "$remote_command" <"$DGX_SUDO_PASSWORD_FILE"
    remote_rc=$?
  elif [[ -n "${DGX_SUDO_PASSWORD:-}" ]]; then
    printf '%s\n' "$DGX_SUDO_PASSWORD" | ssh -T "${options[@]}" "$target" "$remote_command"
    remote_rc=$?
  else
    echo "DGX sudo credential is not cached and no process-only stdin source was provided" >&2
    remote_rc=77
  fi
  set -e

  set +e
  ssh -T "${options[@]}" "$target" "test -d '$stage/result' && tar -C '$stage/result' -czf '$stage/result.tgz' ." >/dev/null 2>&1
  archive_remote_rc=$?
  scp "${options[@]}" "$target:$stage/result.tgz" \
    "$result_dir/remote-artifacts/result.tgz" >/dev/null 2>&1
  collect_rc=$?
  set -e
  if test -f "$result_dir/remote-artifacts/result.tgz"; then
    mkdir "$result_dir/run"
    tar -C "$result_dir/run" -xzf "$result_dir/remote-artifacts/result.tgz"
  fi
  cleanup_rc=125
  if test "$collect_rc" = 0; then
    set +e
    ssh -T "${options[@]}" "$target" \
      "test '$stage' = '/home/railgun/.codex-internnav-stage/t4-dgx-dependencies-${grant}' && test -d '$stage' && test ! -L '$stage' && rm -rf -- '$stage'"
    cleanup_rc=$?
    set -e
  fi
  python3 - "$result_dir/dgx_dependency_summary.json" "$grant" "$auth" \
    "$remote_rc" "$archive_remote_rc" "$collect_rc" "$cleanup_rc" <<'PY'
import json,sys,time
from pathlib import Path
status={}
try: status=json.load(open(Path(sys.argv[1]).parent/"run/status.json",encoding="utf-8"))
except (OSError,json.JSONDecodeError): pass
codes=list(map(int,sys.argv[4:8]))
passed=all(code==0 for code in codes) and status.get("status")=="PASS"
payload={"schema_version":1,"status":"PASS" if passed else "FAIL",
"grant_id":sys.argv[2],"authorization_ref_sha":sys.argv[3],
"exit_codes":{"remote":codes[0],"remote_archive":codes[1],"collection":codes[2],"cleanup":codes[3]},
"host_role":"dgx_onboard_compute","installed":["navigation2","nav2_bringup"],
"credential_persisted":False,"strict_evidence_modified":False,
"real_go2_targeted":False,"recorded_unix":time.time()}
Path(sys.argv[1]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
if not passed: raise SystemExit(1)
PY
}

if [[ "${1:-}" == _under_lease ]]; then shift; run_under_lease "$@"; exit $?; fi
[[ $# -eq 2 ]] || usage
grant="$1"; auth="$2"
[[ "$grant" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{7,95}$ ]] || die "unsafe grant"
[[ "$auth" =~ ^[0-9a-f]{40}$ ]] || die "authorization SHA must be full"
script_path="$(readlink -f "${BASH_SOURCE[0]}")"
root="$(readlink -f "$(dirname "$script_path")/..")"
git_dir="$(resolve_git_dir "$root")"
test "$(git -c core.autocrlf=false --git-dir="$git_dir" rev-parse "$COORDINATION_REF")" = "$auth"
test -z "$(git -c core.autocrlf=true --git-dir="$git_dir" --work-tree="$root" status --porcelain --untracked-files=no)"
result_rel="results/parallel/t4_functional/dgx-dependencies-00-${grant}"
lease_rel="results/parallel/t4_functional/lease-dgx-dependencies-00-${grant}"
test ! -e "$root/$result_rel"; test ! -e "$root/$lease_rel"
python3 - "$root" "$git_dir" "$auth" "$grant" "$result_rel" <<'PY'
import json,re,subprocess,sys
root,git_dir,ref,grant,result=sys.argv[1:]
doc=subprocess.run(["git",f"--git-dir={git_dir}",f"--work-tree={root}","show",f"{ref}:coordination/TASK_BOARD.md"],check=True,capture_output=True,text=True).stdout
blocks=re.findall(r"<!--\s*INTERNAV_ONLINE_GRANT_V1\s*\r?\n(.*?)\r?\nINTERNAV_ONLINE_GRANT_V1\s*-->",doc,flags=re.DOTALL)
expected={"schema_version":1,"status":"GRANTED","worker":"00","resource":"dgx","profile":"dgx_onboard_dependencies","result_dir":result,"grant_id":grant}
if len(blocks)!=1 or json.loads(blocks[0])!=expected: raise SystemExit("DGX dependency grant mismatch")
PY
installer_sha="$(sha256sum "$root/scripts/install_t4_dgx_nav2_dependencies.sh" | cut -d' ' -f1)"
bash "$root/scripts/with_resource_lease.sh" dgx --owner codex-00 \
  --task "00-dgx-onboard-dependencies-${grant}" --log-dir "$root/$lease_rel" -- \
  bash "$script_path" _under_lease "$root" "$grant" "$auth" "$result_rel" \
    "$lease_rel" "$installer_sha"
