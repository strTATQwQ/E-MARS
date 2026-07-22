#!/usr/bin/env bash
set -euo pipefail

readonly COORDINATION_REF="refs/heads/codex/parallel-integration"

usage() {
  cat >&2 <<'EOF'
Usage:
  run_t4_pilot_online.sh <grant-id> <authorization-ref-sha> <functional-prepare-result-rel> <oracle10-result-rel>

Requires PASS functional staging and 10-episode Oracle evidence. The exact
grant is worker 00 / dgx+isaac / functional_pilot20. The combined lease starts
the real model on DGX first, then runs 20 completion_sim episodes on Isaac.
EOF
  exit 64
}

die() { printf 'T4 pilot coordinator: %s\n' "$*" >&2; exit 64; }

resolve_git_dir() {
  local root="$1" raw
  if [[ -d "$root/.git" ]]; then readlink -f "$root/.git"; return; fi
  [[ -f "$root/.git" ]] || die "worktree has no .git pointer"
  raw="$(sed -n 's/^gitdir: //p' "$root/.git")"
  [[ -n "$raw" ]] || die "malformed .git pointer"
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
  case "$role:$REMOTE_USER@$REMOTE_HOST" in
    dgx:railgun@10.100.100.128|isaac:song@10.100.120.111) ;;
    *) die "remote role identity changed" ;;
  esac
  if [[ -n "$REMOTE_PASSWORD_FILE" ]]; then
    [[ -r "$REMOTE_PASSWORD_FILE" ]] || die "$role password file is unreadable"
  fi
  if [[ -n "$REMOTE_PASSWORD_FILE$REMOTE_PASSWORD" ]]; then
    command -v sshpass >/dev/null 2>&1 || die "sshpass is required for password transport"
  fi
  REMOTE_TARGET="$REMOTE_USER@$REMOTE_HOST"
  SSH_OPTIONS=(
    -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=2
    -o StrictHostKeyChecking=accept-new
  )
  [[ -n "$REMOTE_PASSWORD_FILE$REMOTE_PASSWORD" ]] || SSH_OPTIONS+=(-o BatchMode=yes)
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
    scp -P "$REMOTE_PORT" "${SSH_OPTIONS[@]}" "$REMOTE_TARGET:$remote_path" "$local_path"
  fi
}

create_stage() {
  local role="$1" stage="$2" expected_ip
  expected_ip=" 10.100.100.128/"
  test "$role" != isaac || expected_ip=" 10.100.120.111/"
  printf -v command_text \
    'ip -4 -o addr show | grep -Fq %q && test ! -e %q && install -d -m 700 %q' \
    "$expected_ip" "$stage" "$stage"
  remote_exec "$role" "$command_text"
}

remove_stage() {
  local role="$1" stage="$2"
  printf -v command_text \
    'test -d %q && test ! -L %q && test "$(realpath %q)" = %q && rm -rf -- %q' \
    "$stage" "$stage" "$stage" "$stage" "$stage"
  remote_exec "$role" "$command_text"
}

run_under_lease() {
  [[ $# -eq 12 ]] || die "invalid internal invocation"
  local root="$1" grant_id="$2" authorization_ref="$3" deployment_ref="$4"
  local dgx_root="$5" isaac_root="$6" result_rel="$7" lease_rel="$8"
  local prepare_rel="$9" oracle_rel="${10}" contract_tool="${11}" output_root="${12}"
  local dgx_stage="/home/railgun/.codex-internnav-stage/t4-pilot-model-${grant_id}"
  local isaac_stage="/home/song/.codex-internnav-stage/t4-pilot-${grant_id}"
  local result_dir="$root/$result_rel"
  local model_started=0 model_stopped=0 model_start_rc=125 pilot_rc=125 model_stop_rc=125
  local collect_rc=0 dgx_cleanup_rc=125 isaac_cleanup_rc=125 validation_rc=125
  local command_text model_ready_sha

  mkdir -p "$result_dir/remote-artifacts/dgx" "$result_dir/remote-artifacts/isaac" \
    "$result_dir/model-run" "$result_dir/pilot-run"
  configure_role dgx
  configure_role isaac

  cleanup_model() {
    local incoming=$?
    trap - EXIT INT TERM HUP
    set +e
    if (( model_started == 1 && model_stopped == 0 )); then
      printf -v command_text 'bash %q stop %q %q %q %q %q' \
        "$dgx_root/coordination/remote_t4_model_session.sh" "$dgx_root" \
        "$dgx_stage" "$grant_id" "$authorization_ref" "$deployment_ref"
      remote_exec dgx "$command_text" >/dev/null 2>&1 || true
    fi
    exit "$incoming"
  }
  trap cleanup_model EXIT
  trap 'exit 130' INT TERM HUP

  create_stage dgx "$dgx_stage"
  printf -v command_text 'bash %q start %q %q %q %q %q' \
    "$dgx_root/coordination/remote_t4_model_session.sh" "$dgx_root" \
    "$dgx_stage" "$grant_id" "$authorization_ref" "$deployment_ref"
  set +e
  remote_exec dgx "$command_text"
  model_start_rc=$?
  set -e
  test "$model_start_rc" = 0
  model_started=1
  remote_get dgx "$dgx_stage/model_ready_receipt.json" \
    "$result_dir/remote-artifacts/dgx/model_ready_receipt.json"
  python3 - "$result_dir/remote-artifacts/dgx/model_ready_receipt.json" \
    "$grant_id" "$authorization_ref" "$deployment_ref" <<'PY'
import json,sys
value=json.load(open(sys.argv[1],encoding="utf-8"))
checks=(value.get("status")=="PASS",value.get("grant_id")==sys.argv[2],
        value.get("authorization_ref_sha")==sys.argv[3],value.get("deployment_ref_sha")==sys.argv[4],
        value.get("backend")=="real",value.get("resource_lease_ack")=="dgx+isaac")
if not all(checks): raise SystemExit("DGX model ready receipt mismatch")
PY
  model_ready_sha="$(sha256sum "$result_dir/remote-artifacts/dgx/model_ready_receipt.json" | cut -d' ' -f1)"

  create_stage isaac "$isaac_stage"
  printf -v command_text 'bash %q %q %q %q %q %q %q' \
    "$isaac_root/coordination/remote_t4_pilot_run.sh" "$isaac_root" \
    "$isaac_stage" "$grant_id" "$authorization_ref" "$deployment_ref" \
    "$model_ready_sha"
  set +e
  remote_exec isaac "$command_text"
  pilot_rc=$?
  set -e

  printf -v command_text 'bash %q stop %q %q %q %q %q' \
    "$dgx_root/coordination/remote_t4_model_session.sh" "$dgx_root" \
    "$dgx_stage" "$grant_id" "$authorization_ref" "$deployment_ref"
  set +e
  remote_exec dgx "$command_text"
  model_stop_rc=$?
  set -e
  model_stopped=1

  set +e
  for file in model_ready_receipt.json model_stop_receipt.json model_invocation.json \
      model_process.json model_health_ready.json deployment_validation.json \
      payload_reverification.json dgx_workspace_validation.json model_outer.log \
      model_health_probe.log model_residual_probe.json model_residual_probe.log \
      model_prestart_nodes.txt model_prestart_nodes.log \
      model_prestart_processes.txt model_prestart_processes.log; do
    remote_get dgx "$dgx_stage/$file" "$result_dir/remote-artifacts/dgx/$file" \
      >/dev/null 2>&1 || true
  done
  remote_get dgx "$dgx_stage/model_stop_receipt.json" \
    "$result_dir/remote-artifacts/dgx/model_stop_receipt.json" || collect_rc=$?
  remote_get dgx "$dgx_stage/model_result.tgz" \
    "$result_dir/remote-artifacts/dgx/model_result.tgz" || collect_rc=$?

  for file in pilot_run_receipt.json pilot_invocation.json deployment_validation.json \
      payload_reverification.json overlay_source_validation.json pilot_outer.log \
      container_cleanup_primary.log container_cleanup_probe.log residual_probe.json \
      residual_probe.log pilot_validation.log; do
    remote_get isaac "$isaac_stage/$file" "$result_dir/remote-artifacts/isaac/$file" \
      >/dev/null 2>&1 || true
  done
  remote_get isaac "$isaac_stage/pilot_run_receipt.json" \
    "$result_dir/remote-artifacts/isaac/pilot_run_receipt.json" || collect_rc=$?
  remote_get isaac "$isaac_stage/pilot_result.tgz" \
    "$result_dir/remote-artifacts/isaac/pilot_result.tgz" || collect_rc=$?
  set -e

  if [[ -f "$result_dir/remote-artifacts/dgx/model_result.tgz" ]]; then
    python3 "$contract_tool" extract-oracle-archive \
      --archive "$result_dir/remote-artifacts/dgx/model_result.tgz" \
      --output-dir "$result_dir/model-run" \
      >"$result_dir/model_archive_extraction.json" 2>&1 || collect_rc=$?
  fi
  if [[ -f "$result_dir/remote-artifacts/isaac/pilot_result.tgz" ]]; then
    python3 "$contract_tool" extract-oracle-archive \
      --archive "$result_dir/remote-artifacts/isaac/pilot_result.tgz" \
      --output-dir "$result_dir/pilot-run" \
      >"$result_dir/pilot_archive_extraction.json" 2>&1 || collect_rc=$?
  fi

  if (( collect_rc == 0 )); then
    set +e
    remove_stage isaac "$isaac_stage"; isaac_cleanup_rc=$?
    remove_stage dgx "$dgx_stage"; dgx_cleanup_rc=$?
    set -e
  fi

  set +e
  python3 "$contract_tool" validate-pilot-collected \
    --pilot-result "$result_dir/pilot-run" --model-result "$result_dir/model-run" \
    --model-ready-receipt "$result_dir/remote-artifacts/dgx/model_ready_receipt.json" \
    --model-stop-receipt "$result_dir/remote-artifacts/dgx/model_stop_receipt.json" \
    --pilot-receipt "$result_dir/remote-artifacts/isaac/pilot_run_receipt.json" \
    --output "$result_dir/pilot20_summary.json" --grant-id "$grant_id" \
    --authorization-ref "$authorization_ref" --deployment-ref "$deployment_ref" \
    --remote-stages-cleanup "$([[ $dgx_cleanup_rc == 0 && $isaac_cleanup_rc == 0 ]] && printf PASS || printf FAIL)" \
    >"$result_dir/local_validation.log" 2>&1
  validation_rc=$?
  set -e
  if (( validation_rc != 0 )) && [[ ! -e "$result_dir/pilot20_summary.json" ]]; then
    python3 - "$result_dir/pilot20_summary.json" "$grant_id" "$authorization_ref" \
      "$deployment_ref" "$prepare_rel" "$oracle_rel" "$model_start_rc" \
      "$pilot_rc" "$model_stop_rc" "$collect_rc" "$dgx_cleanup_rc" \
      "$isaac_cleanup_rc" "$validation_rc" <<'PY'
import json,sys,time
from pathlib import Path
payload={"schema_version":1,"status":"FAIL","grant_id":sys.argv[2],
"authorization_ref_sha":sys.argv[3],"deployment_ref_sha":sys.argv[4],
"functional_prepare_result":sys.argv[5],"oracle_prerequisite_result":sys.argv[6],
"exit_codes":{"model_start":int(sys.argv[7]),"pilot":int(sys.argv[8]),
"model_stop":int(sys.argv[9]),"collection":int(sys.argv[10]),
"dgx_stage_cleanup":int(sys.argv[11]),"isaac_stage_cleanup":int(sys.argv[12]),
"validation":int(sys.argv[13])},"runtime_policy":"completion_sim",
"runtime_target":"isaac_simulation","model_host":"dgx_spark_only",
"strict_evidence_modified":False,"real_go2_targeted":False,"recorded_unix":time.time()}
Path(sys.argv[1]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
PY
  fi

  trap - EXIT INT TERM HUP
  (( model_start_rc == 0 )) || return "$model_start_rc"
  (( pilot_rc == 0 )) || return "$pilot_rc"
  (( model_stop_rc == 0 )) || return "$model_stop_rc"
  (( collect_rc == 0 )) || return "$collect_rc"
  (( dgx_cleanup_rc == 0 && isaac_cleanup_rc == 0 )) || return 1
  return "$validation_rc"
}

if [[ "${1:-}" == _under_lease ]]; then shift; run_under_lease "$@"; exit $?; fi

[[ $# -eq 4 ]] || usage
grant_id="$1"; authorization_ref="$2"; prepare_rel="$3"; oracle_rel="$4"
[[ "$grant_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{7,95}$ ]] || die "unsafe grant ID"
[[ "$authorization_ref" =~ ^[0-9a-f]{40}$ ]] || die "authorization ref must be full SHA"
[[ "$prepare_rel" =~ ^results/parallel/t4_functional/prepare-00-[A-Za-z0-9][A-Za-z0-9._-]{7,95}$ ]] || die "unsafe prepare path"
[[ "$oracle_rel" =~ ^results/parallel/t4_functional/oracle10-00-[A-Za-z0-9][A-Za-z0-9._-]{7,95}$ ]] || die "unsafe Oracle path"

script_path="$(readlink -f "${BASH_SOURCE[0]}")"
root="$(readlink -f "$(dirname "$script_path")/..")"
git_dir="$(resolve_git_dir "$root")"
contract_tool="$root/coordination/t4_functional_run_contract.py"
GIT_STATUS=(git -c core.autocrlf=true --git-dir="$git_dir" --work-tree="$root")
GIT_OBJECTS=(git -c core.autocrlf=false --git-dir="$git_dir" --work-tree="$root")
actual_sha="$("${GIT_OBJECTS[@]}" rev-parse "$COORDINATION_REF")"
[[ "$actual_sha" == "$authorization_ref" ]] || die "coordination ref differs from authorization"
[[ -z "$("${GIT_STATUS[@]}" status --porcelain --untracked-files=no)" ]] || die "tracked tree is dirty"

prepare_lines="$(python3 "$contract_tool" prepare-fields --result-dir "$root/$prepare_rel" \
  --git-dir "$git_dir" --authorization-ref "$authorization_ref" --format pilot-lines)"
mapfile -t fields <<<"$prepare_lines"
[[ ${#fields[@]} -eq 6 ]] || die "prepare fields are incomplete"
prepare_grant="${fields[0]}"; deployment_ref="${fields[1]}"
dgx_root="${fields[2]}"; dgx_ws="${fields[3]}"; isaac_root="${fields[4]}"; isaac_ws="${fields[5]}"
[[ "$dgx_root" == "/home/railgun/internnav-t1-t2/.t4-deployments/${prepare_grant}-${deployment_ref:0:12}-dgx" ]] || die "DGX deployment changed"
[[ "$dgx_ws" == "$dgx_root/ros_ws" ]] || die "DGX workspace changed"
[[ "$isaac_root" == "/home/song/internnav-t1-t2/.t4-deployments/${prepare_grant}-${deployment_ref:0:12}-isaac" ]] || die "Isaac deployment changed"
[[ "$isaac_ws" == /home/song/internnav-t4/isaac_ros_ws_45 ]] || die "Isaac workspace changed"

python3 - "$root/$oracle_rel/oracle10_summary.json" "$deployment_ref" <<'PY'
import json,sys
value=json.load(open(sys.argv[1],encoding="utf-8"))
checks=(value.get("status")=="PASS",value.get("deployment_ref_sha")==sys.argv[2],
        value.get("runtime_policy")=="completion_sim",value.get("model_process_started") is False,
        (value.get("oracle_validation") or {}).get("status")=="PASS")
if not all(checks): raise SystemExit("10-episode Oracle prerequisite is not PASS")
PY

result_rel="results/parallel/t4_functional/pilot20-00-${grant_id}"
lease_rel="results/parallel/t4_functional/lease-pilot20-00-${grant_id}"
[[ ! -e "$root/$result_rel" && ! -e "$root/$lease_rel" ]] || die "pilot paths must be fresh"
python3 - "$root" "$git_dir" "$authorization_ref" "$grant_id" "$result_rel" <<'PY'
import json,re,subprocess,sys
root,git_dir,ref_sha,grant_id,result_dir=sys.argv[1:]
doc=subprocess.run(["git",f"--git-dir={git_dir}",f"--work-tree={root}","show",f"{ref_sha}:coordination/TASK_BOARD.md"],check=True,capture_output=True,text=True).stdout
matches=re.findall(r"<!--\s*INTERNAV_ONLINE_GRANT_V1\s*\r?\n(.*?)\r?\nINTERNAV_ONLINE_GRANT_V1\s*-->",doc,flags=re.DOTALL)
expected={"schema_version":1,"status":"GRANTED","worker":"00","resource":"dgx+isaac",
"profile":"functional_pilot20","result_dir":result_dir,"grant_id":grant_id}
if len(matches)!=1 or json.loads(matches[0])!=expected: raise SystemExit("functional pilot grant mismatch")
PY

for role in dgx isaac; do configure_role "$role"; done
bash "$root/scripts/with_resource_lease.sh" both --owner codex-00 \
  --task "00-functional-pilot20-${grant_id}" --log-dir "$root/$lease_rel" -- \
  bash "$script_path" _under_lease "$root" "$grant_id" "$authorization_ref" \
    "$deployment_ref" "$dgx_root" "$isaac_root" "$result_rel" "$lease_rel" \
    "$prepare_rel" "$oracle_rel" "$contract_tool" "$root/$result_rel"
