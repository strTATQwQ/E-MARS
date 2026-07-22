#!/usr/bin/env bash
set -euo pipefail

# Hold a fail-closed flock on the shared resource host while a local command
# runs. The wrapped command may itself use SSH, ROS 2 launchers, or other
# orchestration; losing any holder terminates it.

readonly EXIT_USAGE=64
readonly EXIT_UNAVAILABLE=69
readonly EXIT_BUSY=73
readonly EXIT_LEASE_LOST=75

usage() {
  cat >&2 <<'EOF'
Usage:
  with_resource_lease.sh <isaac|dgx|dgx-a|dgx-b|both|lane-a|lane-b|all-lanes|t5-network|t5-primary|t5-dual> --task TASK --log-dir DIR
      [--owner OWNER] [--acquire-timeout SEC] [--cleanup-timeout SEC]
      [--kill-wait-timeout SEC] -- COMMAND [ARG ...]

Recovery-only modes (the wrapped command is fixed and cannot be substituted):
  recover-dgx-a | recover-dgx-b | recover-isaac-gpu0 | recover-isaac-gpu1
  recover-isaac | recover-all

Authentication (secrets are never written to lease metadata):
  ISAAC_PASSWORD_FILE / DGX_PASSWORD_FILE  preferred password-file inputs
  ISAAC_PASSWORD      / DGX_PASSWORD       process-environment fallback
  Otherwise ssh uses the configured key, agent, or interactive authentication.

Host overrides:
  ISAAC_HOST, ISAAC_USER, ISAAC_PORT
  DGX_HOST, DGX_USER, DGX_PORT
  DGX_B_HOST, DGX_B_USER, DGX_B_PORT
  LEASE_SSH_BIN, LEASE_SSH_EXTRA_OPTS
EOF
  exit "$EXIT_USAGE"
}

die() {
  local code="$1"
  shift
  printf 'resource lease: %s\n' "$*" >&2
  exit "$code"
}

base64_one_line() {
  printf '%s' "$1" | base64 | tr -d '\r\n'
}

sanitize_local_field() {
  printf '%s' "$1" | tr '\t\r\n' '   '
}

[[ $# -ge 1 ]] || usage
mode="$1"
shift
recovery_only=0
recovery_marker=''

case "$mode" in
  # Legacy/global Isaac work must exclude both T5 GPU lanes.  Acquiring both
  # GPU locks before the historical global lock preserves compatibility while
  # preventing the independent lock files from admitting conflicting jobs.
  isaac) requested_resources=(isaac_gpu0 isaac_gpu1 isaac) ;;
  dgx) requested_resources=(dgx) ;;
  dgx-a) requested_resources=(dgx_a) ;;
  dgx-b) requested_resources=(dgx_b) ;;
  both|dgx+isaac) requested_resources=(dgx isaac_gpu0 isaac_gpu1 isaac) ;;
  lane-a) requested_resources=(dgx_a isaac_gpu0) ;;
  lane-b) requested_resources=(dgx_b isaac_gpu1) ;;
  all-lanes) requested_resources=(dgx_a dgx_b isaac_gpu0 isaac_gpu1) ;;
  t5-network) requested_resources=(dgx_a dgx_b isaac_gpu0) ;;
  t5-primary) requested_resources=(dgx_a isaac_gpu0) ;;
  t5-dual) requested_resources=(dgx_a dgx_b isaac_gpu0 isaac_gpu1) ;;
  recover-dgx-a) requested_resources=(dgx_a); recovery_only=1; recovery_marker=/tmp/internnav_dgx.quarantine ;;
  recover-dgx-b) requested_resources=(dgx_b); recovery_only=1; recovery_marker=/tmp/internnav_dgx.quarantine ;;
  recover-isaac-gpu0) requested_resources=(isaac_gpu0); recovery_only=1; recovery_marker=/tmp/internnav_isaac_gpu0.quarantine ;;
  recover-isaac-gpu1) requested_resources=(isaac_gpu1); recovery_only=1; recovery_marker=/tmp/internnav_isaac_gpu1.quarantine ;;
  # The Isaac quarantine is host-wide, so recovery must exclude the legacy
  # global holder and both GPU lanes before auditing or changing the marker.
  recover-isaac) requested_resources=(isaac_gpu0 isaac_gpu1 isaac); recovery_only=1; recovery_marker='*' ;;
  recover-all) requested_resources=(dgx_a dgx_b isaac_gpu0 isaac_gpu1 isaac); recovery_only=1; recovery_marker='*' ;;
  *) usage ;;
esac

owner="${USER:-unknown}@$(hostname 2>/dev/null || printf unknown)"
task=""
log_dir=""
acquire_timeout=15
cleanup_timeout="${LEASE_CLEANUP_TIMEOUT_SEC:-30}"
kill_wait_timeout="${LEASE_KILL_WAIT_TIMEOUT_SEC:-10}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --owner)
      [[ $# -ge 2 ]] || usage
      owner="$2"
      shift 2
      ;;
    --task)
      [[ $# -ge 2 ]] || usage
      task="$2"
      shift 2
      ;;
    --log-dir)
      [[ $# -ge 2 ]] || usage
      log_dir="$2"
      shift 2
      ;;
    --acquire-timeout)
      [[ $# -ge 2 ]] || usage
      acquire_timeout="$2"
      shift 2
      ;;
    --cleanup-timeout)
      [[ $# -ge 2 ]] || usage
      cleanup_timeout="$2"
      shift 2
      ;;
    --kill-wait-timeout)
      [[ $# -ge 2 ]] || usage
      kill_wait_timeout="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *) usage ;;
  esac
done

[[ -n "$task" ]] || die "$EXIT_USAGE" '--task is required'
[[ -n "$log_dir" ]] || die "$EXIT_USAGE" '--log-dir is required'
[[ $# -gt 0 ]] || die "$EXIT_USAGE" 'a command is required after --'
[[ "$acquire_timeout" =~ ^[1-9][0-9]*$ ]] || die "$EXIT_USAGE" '--acquire-timeout must be a positive integer'
[[ "$cleanup_timeout" =~ ^[1-9][0-9]*$ ]] || die "$EXIT_USAGE" '--cleanup-timeout must be a positive integer'
[[ "$kill_wait_timeout" =~ ^[1-9][0-9]*$ ]] || die "$EXIT_USAGE" '--kill-wait-timeout must be a positive integer'

command_argv=("$@")
if (( recovery_only == 1 )); then
  lease_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
  recovery_entry="$lease_script_dir/recover_t5_resource_quarantine.sh"
  stale_nav2_entry="$lease_script_dir/recover_t5_stale_dgx_nav2.sh"
  lease_root="$(cd -- "$lease_script_dir/.." && pwd -P)"
  if [[ ${#command_argv[@]} -eq 5 && \
        "${command_argv[0]}" == bash && \
        "${command_argv[1]}" == "$recovery_entry" && \
        "${command_argv[2]}" == --inside ]]; then
    case "$mode:${command_argv[3]}" in
      recover-dgx-a:dgx-a|recover-dgx-b:dgx-b|recover-isaac-gpu0:isaac-gpu0|recover-isaac-gpu1:isaac-gpu1|recover-isaac:isaac|recover-all:all) ;;
      *) die "$EXIT_USAGE" 'recovery-only mode/scope mismatch' ;;
    esac
  elif [[ ${#command_argv[@]} -eq 8 && \
          "$mode" == recover-dgx-a && \
          "${command_argv[0]}" == bash && \
          "${command_argv[1]}" == "$stale_nav2_entry" && \
          "${command_argv[2]}" == --inside && \
          "${command_argv[3]}" == dgx-a && \
          "${command_argv[4]}" == /home/railgun/internnav-t1-t2/.t5-deployments/t5d0020260719t135307-78cc4b1e7b1b-lane-a && \
          "${command_argv[6],,}" == 60e193f5ce3ca1060d72c65de74505b2ba5790cc10aa12386d0dfb8938232b85 && \
          "$(dirname -- "${command_argv[7]}")" == "$lease_root/results/internnav_t5" && \
          "$(basename -- "${command_argv[7]}")" =~ ^stale-dgx-recovery-[A-Za-z0-9][A-Za-z0-9._-]*$ && \
          "${command_argv[5]}" == "${command_argv[7]}/evidence/stale_nav2_evidence.json" ]]; then
    : # The fixed stale-Nav2 entry performs the deeper evidence/root checks.
  else
    die "$EXIT_USAGE" \
      'recovery-only mode requires the fixed recovery entry; recovery-only mode cannot run an arbitrary command'
  fi
  command_argv=(env "INTERNNAV_T5_RESOURCE_LEASE_ACK=$mode" "${command_argv[@]}")
fi
printf -v command_text '%q ' "${command_argv[@]}"
command_text="${command_text% }"
started_at="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
caller_host="$(hostname 2>/dev/null || printf unknown)"

mkdir -p -- "$log_dir"
log_dir="$(cd -- "$log_dir" && pwd -P)"
lease_state_dir="$(mktemp -d "${TMPDIR:-/tmp}/internnav-lease.XXXXXX")"

declare -a held_resources=()
declare -A holder_pid=()
declare -A holder_fd=()
declare -A holder_fifo=()
declare -A holder_status=()
declare -A holder_stderr=()
wrapped_pid=""
wrapped_pgid=""
cleanup_started=0
cleanup_receipt_written=0

terminate_wrapped_group() {
  local signal="${1:-TERM}"
  if [[ "$wrapped_pgid" =~ ^[1-9][0-9]*$ ]]; then
    kill -"$signal" -- "-$wrapped_pgid" 2>/dev/null || true
  elif [[ "$wrapped_pid" =~ ^[1-9][0-9]*$ ]]; then
    kill -"$signal" "$wrapped_pid" 2>/dev/null || true
  fi
}

wrapped_leader_is_alive() {
  [[ "$wrapped_pid" =~ ^[1-9][0-9]*$ ]] && kill -0 "$wrapped_pid" 2>/dev/null
}

wrapped_group_is_alive() {
  [[ "$wrapped_pgid" =~ ^[1-9][0-9]*$ ]] && \
    kill -0 -- "-$wrapped_pgid" 2>/dev/null
}

wrapped_is_alive() {
  wrapped_leader_is_alive || wrapped_group_is_alive
}

wait_wrapped_absent() {
  local timeout="$1" deadline
  deadline=$((SECONDS + timeout))
  while wrapped_is_alive && (( SECONDS < deadline )); do
    sleep 0.2
  done
  ! wrapped_is_alive
}

write_cleanup_receipt() {
  local reason="$1" term_sent="$2" graceful="$3" escalated="$4" absent="$5"
  local resource live_resources=""
  (( cleanup_receipt_written == 0 )) || return 0
  cleanup_receipt_written=1
  for resource in "${held_resources[@]}"; do
    if kill -0 "${holder_pid[$resource]:-}" 2>/dev/null; then
      live_resources+="${live_resources:+ }$resource"
    fi
  done
  python3 - "$log_dir/lease_cleanup_receipt.json" "$reason" "$term_sent" \
    "$graceful" "$escalated" "$absent" "$cleanup_timeout" \
    "$kill_wait_timeout" "${requested_resources[*]}" "$live_resources" <<'PY'
import json, sys, time
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "status": "PASS" if sys.argv[6] == "true" else "FAIL",
    "reason": sys.argv[2],
    "term_sent": sys.argv[3] == "true",
    "graceful_cleanup_completed": sys.argv[4] == "true",
    "kill_escalated": sys.argv[5] == "true",
    "wrapped_process_group_absent_before_lock_release": sys.argv[6] == "true",
    "cleanup_timeout_sec": int(sys.argv[7]),
    "kill_wait_timeout_sec": int(sys.argv[8]),
    "requested_resources": sys.argv[9].split(),
    "resources_still_held_before_release": sys.argv[10].split(),
    "recorded_unix": time.time(),
}
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(path)
PY
}

# Drain the entire local coordinator process group while every holder that is
# still connected remains locked.  Coordinators use their TERM trap to stop
# remote supervisors and write residual evidence.  KILL is only the bounded
# last resort after that remote cleanup budget has expired.
drain_wrapped_group() {
  local reason="$1" term_sent=false graceful=false escalated=false absent=false
  if wrapped_is_alive; then
    term_sent=true
    terminate_wrapped_group TERM
    if wait_wrapped_absent "$cleanup_timeout"; then
      graceful=true
    else
      escalated=true
      terminate_wrapped_group KILL
      wait_wrapped_absent "$kill_wait_timeout" || true
    fi
  else
    graceful=true
  fi
  wrapped_is_alive || absent=true
  if [[ "$absent" == true && "$wrapped_pid" =~ ^[1-9][0-9]*$ ]]; then
    wait "$wrapped_pid" 2>/dev/null || true
  fi
  write_cleanup_receipt "$reason" "$term_sent" "$graceful" "$escalated" "$absent"
  [[ "$absent" == true ]]
}

close_fd() {
  local fd="$1"
  [[ "$fd" =~ ^[0-9]+$ ]] || return 0
  eval "exec ${fd}>&-" 2>/dev/null || true
}

release_all() {
  local resource fd pid index
  (( cleanup_started == 0 )) || return 0
  cleanup_started=1

  for (( index=${#held_resources[@]}-1; index>=0; index-- )); do
    resource="${held_resources[$index]}"
    fd="${holder_fd[$resource]:-}"
    pid="${holder_pid[$resource]:-}"
    if [[ -n "$fd" ]]; then
      printf 'RELEASE\n' >&"$fd" 2>/dev/null || true
      close_fd "$fd"
    fi
    if [[ -n "$pid" ]]; then
      wait "$pid" 2>/dev/null || true
    fi
  done
}

persist_holder_diagnostics() {
  local resource status_file stderr_file
  for resource in "${held_resources[@]}"; do
    status_file="${holder_status[$resource]:-}"
    stderr_file="${holder_stderr[$resource]:-}"
    if [[ -n "$status_file" && -f "$status_file" ]]; then
      cp -- "$status_file" "$log_dir/holder_${resource}.stdout.log" 2>/dev/null || true
    fi
    if [[ -n "$stderr_file" && -f "$stderr_file" ]]; then
      cp -- "$stderr_file" "$log_dir/holder_${resource}.stderr.log" 2>/dev/null || true
    fi
  done
}

on_exit() {
  local rc=$?
  trap - EXIT INT TERM HUP
  set +e
  if [[ -n "$wrapped_pid" ]]; then
    drain_wrapped_group exit_trap || rc="$EXIT_LEASE_LOST"
  fi
  release_all
  persist_holder_diagnostics
  rm -rf -- "$lease_state_dir"
  exit "$rc"
}

on_signal() {
  local signal="$1"
  printf 'resource lease: received %s; terminating wrapped command\n' "$signal" >&2
  exit 130
}

trap on_exit EXIT
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM
trap 'on_signal HUP' HUP

read -r -d '' remote_holder_program <<'REMOTE_HOLDER' || true
set -euo pipefail
lock_file="$1"
resource="$2"
owner_b64="$3"
task_b64="$4"
started_b64="$5"
command_b64="$6"
log_dir_b64="$7"
caller_b64="$8"
quarantine_files="$9"
recovery_only="${10}"
recovery_marker="${11}"

command -v flock >/dev/null 2>&1 || {
  printf 'NO_FLOCK resource=%s host=%s\n' "$resource" "$(hostname)"
  exit 69
}

# Opening with append preserves the current holder metadata for a losing
# contender. The winner rewrites metadata only after flock succeeds.
exec 9>>"$lock_file"
if ! flock -n 9; then
  printf 'BUSY resource=%s host=%s\n' "$resource" "$(hostname)"
  cat "$lock_file" 2>/dev/null || true
  exit 73
fi

# A flock only proves that no current holder exists. It cannot prove that a
# previous remote workload cleaned up its PGID, sockets, or containers. D0
# preparation leaves this host-persistent marker whenever absence could not be
# proved, and only an audited cleanup may remove it.
IFS='|' read -r -a quarantine_candidates <<<"$quarantine_files"
for quarantine_file in "${quarantine_candidates[@]}"; do
  [[ -n "$quarantine_file" ]] || exit 75
  # `-e` follows symlinks; include `-L` so a dangling quarantine symlink is
  # still a fail-closed marker instead of being treated as absent.
  if [[ -e "$quarantine_file" || -L "$quarantine_file" ]]; then
    if [[ "$recovery_only" == 1 && \
          ( "$recovery_marker" == '*' || "$recovery_marker" == "$quarantine_file" ) ]]; then
      printf 'RECOVERY_ONLY resource=%s host=%s marker=%s marker_present=true\n' \
        "$resource" "$(hostname)" "$quarantine_file"
      continue
    fi
    printf 'QUARANTINED resource=%s host=%s marker=%s\n' \
      "$resource" "$(hostname)" "$quarantine_file"
    cat "$quarantine_file" 2>/dev/null || true
    exit 75
  fi
done
if [[ "$recovery_only" == 1 ]]; then
  printf 'RECOVERY_ONLY resource=%s host=%s recoverable_marker=%s\n' \
    "$resource" "$(hostname)" "$recovery_marker"
fi

decode() { printf '%s' "$1" | base64 -d; }
clean() { printf '%s' "$1" | tr '\t\r\n' '   '; }

owner="$(clean "$(decode "$owner_b64")")"
task="$(clean "$(decode "$task_b64")")"
requested_at="$(clean "$(decode "$started_b64")")"
command_text="$(clean "$(decode "$command_b64")")"
log_dir="$(clean "$(decode "$log_dir_b64")")"
caller_host="$(clean "$(decode "$caller_b64")")"
acquired_at="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"

write_metadata() {
  local state="$1"
  local ended_at="${2:-}"
  {
    printf 'state=%s\n' "$state"
    printf 'resource=%s\n' "$resource"
    printf 'owner=%s\n' "$owner"
    printf 'task=%s\n' "$task"
    printf 'holder_pid=%s\n' "$$"
    printf 'requested_at=%s\n' "$requested_at"
    printf 'acquired_at=%s\n' "$acquired_at"
    printf 'command=%s\n' "$command_text"
    printf 'log_dir=%s\n' "$log_dir"
    printf 'caller_host=%s\n' "$caller_host"
    printf 'resource_host=%s\n' "$(hostname)"
    [[ -z "$ended_at" ]] || printf 'ended_at=%s\n' "$ended_at"
  } >"$lock_file"
}

released=0
mark_released() {
  (( released == 0 )) || return 0
  released=1
  write_metadata RELEASED "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" || true
}

finish_holder() {
  local rc=$?
  trap - EXIT HUP INT TERM
  mark_released
  exit "$rc"
}

trap finish_holder EXIT
# A signal must exit instead of merely rewriting metadata and returning to the
# blocking read.  The previous trap could leave a RELEASED-but-locked holder.
trap 'exit 130' HUP INT TERM

write_metadata HELD
printf 'ACQUIRED resource=%s host=%s pid=%s\n' "$resource" "$(hostname)" "$$"

while true; do
  control=""
  if IFS= read -r -t 10 control; then
    [[ "$control" == RELEASE ]] && break
  else
    read_rc=$?
    # Bash returns 1 for EOF and >128 for a timeout.  EOF releases normally;
    # a timeout writes application data so an idle SSH transport cannot fail
    # silently while leaving the remote flock alive.
    (( read_rc == 1 )) && break
    (( read_rc > 128 )) || exit "$read_rc"
    printf 'HEARTBEAT resource=%s host=%s holder_pid=%s at=%s\n' \
      "$resource" "$(hostname)" "$$" "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" || exit 75
  fi
done
REMOTE_HOLDER

remote_program_b64="$(base64_one_line "$remote_holder_program")"

start_ssh_holder() {
  local resource="$1"
  local host user port lock_file quarantine_files password password_file target
  local status_file stderr_file fifo control_fd pid remote_command
  local askpass_script
  local ssh_bin="${LEASE_SSH_BIN:-ssh}"
  local -a ssh_options

  case "$resource" in
    dgx)
      host="${DGX_HOST:-10.100.100.128}"
      user="${DGX_USER:-railgun}"
      port="${DGX_PORT:-22}"
      lock_file='/tmp/internnav_dgx.lock'
      quarantine_files='/tmp/internnav_dgx.quarantine'
      password="${DGX_PASSWORD:-}"
      password_file="${DGX_PASSWORD_FILE:-}"
      ;;
    dgx_a)
      host="${DGX_HOST:-10.100.100.128}"
      user="${DGX_USER:-railgun}"
      port="${DGX_PORT:-22}"
      lock_file='/tmp/internnav_dgx.lock'
      quarantine_files='/tmp/internnav_dgx.quarantine'
      password="${DGX_PASSWORD:-}"
      password_file="${DGX_PASSWORD_FILE:-}"
      ;;
    dgx_b)
      host="${DGX_B_HOST:-10.100.120.116}"
      user="${DGX_B_USER:-rail}"
      port="${DGX_B_PORT:-22}"
      lock_file='/tmp/internnav_dgx.lock'
      quarantine_files='/tmp/internnav_dgx.quarantine'
      password="${DGX_B_PASSWORD:-}"
      password_file="${DGX_B_PASSWORD_FILE:-}"
      ;;
    isaac)
      host="${ISAAC_HOST:-10.100.120.111}"
      user="${ISAAC_USER:-song}"
      port="${ISAAC_PORT:-22}"
      lock_file='/tmp/internnav_isaac.lock'
      quarantine_files='/tmp/internnav_isaac.quarantine|/tmp/internnav_isaac_gpu0.quarantine|/tmp/internnav_isaac_gpu1.quarantine'
      password="${ISAAC_PASSWORD:-}"
      password_file="${ISAAC_PASSWORD_FILE:-}"
      ;;
    isaac_gpu0)
      host="${ISAAC_HOST:-10.100.120.123}"
      user="${ISAAC_USER:-song}"
      port="${ISAAC_PORT:-22}"
      lock_file='/tmp/internnav_isaac_gpu0.lock'
      quarantine_files='/tmp/internnav_isaac.quarantine|/tmp/internnav_isaac_gpu0.quarantine'
      password="${ISAAC_PASSWORD:-}"
      password_file="${ISAAC_PASSWORD_FILE:-}"
      ;;
    isaac_gpu1)
      host="${ISAAC_HOST:-10.100.120.123}"
      user="${ISAAC_USER:-song}"
      port="${ISAAC_PORT:-22}"
      lock_file='/tmp/internnav_isaac_gpu1.lock'
      quarantine_files='/tmp/internnav_isaac.quarantine|/tmp/internnav_isaac_gpu1.quarantine'
      password="${ISAAC_PASSWORD:-}"
      password_file="${ISAAC_PASSWORD_FILE:-}"
      ;;
    *) die "$EXIT_USAGE" "unknown resource: $resource" ;;
  esac

  [[ "$host" =~ ^[A-Za-z0-9_.:-]+$ ]] || die "$EXIT_USAGE" "unsafe host value for $resource"
  [[ "$user" =~ ^[A-Za-z0-9_.-]+$ ]] || die "$EXIT_USAGE" "unsafe user value for $resource"
  [[ "$port" =~ ^[0-9]+$ ]] || die "$EXIT_USAGE" "invalid SSH port for $resource"
  command -v "$ssh_bin" >/dev/null 2>&1 || die "$EXIT_UNAVAILABLE" "SSH executable not found: $ssh_bin"

  ssh_options=(
    -T
    -p "$port"
    -o ConnectTimeout=8
    -o ServerAliveInterval=5
    -o ServerAliveCountMax=2
    -o ExitOnForwardFailure=yes
  )
  if [[ -n "${LEASE_SSH_EXTRA_OPTS:-}" ]]; then
    read -r -a extra_options <<<"$LEASE_SSH_EXTRA_OPTS"
    ssh_options+=("${extra_options[@]}")
  fi

  target="${user}@${host}"
  status_file="$lease_state_dir/${resource}.status"
  stderr_file="$lease_state_dir/${resource}.stderr"
  fifo="$lease_state_dir/${resource}.control"
  mkfifo "$fifo"
  exec {control_fd}<>"$fifo"

  remote_command="bash -c \"\$(printf '%s' '$remote_program_b64' | base64 -d)\" lease-holder '$lock_file' '$resource' '$(base64_one_line "$owner")' '$(base64_one_line "$task")' '$(base64_one_line "$started_at")' '$(base64_one_line "$command_text")' '$(base64_one_line "$log_dir")' '$(base64_one_line "$caller_host")' '$quarantine_files' '$recovery_only' '$recovery_marker'"

  if [[ -n "$password_file" ]]; then
    [[ -r "$password_file" ]] || die "$EXIT_UNAVAILABLE" "$resource password file is not readable"
    if command -v sshpass >/dev/null 2>&1; then
      sshpass -f "$password_file" "$ssh_bin" "${ssh_options[@]}" "$target" "$remote_command" \
        <"$fifo" >"$status_file" 2>"$stderr_file" &
    else
      askpass_script="$lease_state_dir/${resource}.askpass.sh"
      printf '%s\n' '#!/usr/bin/env bash' 'IFS= read -r secret <"$LEASE_ASKPASS_FILE"' 'printf "%s\n" "$secret"' >"$askpass_script"
      chmod 700 "$askpass_script"
      DISPLAY="${DISPLAY:-internnav-lease}" SSH_ASKPASS_REQUIRE=force \
        SSH_ASKPASS="$askpass_script" LEASE_ASKPASS_FILE="$password_file" \
        "$ssh_bin" "${ssh_options[@]}" "$target" "$remote_command" \
        <"$fifo" >"$status_file" 2>"$stderr_file" &
    fi
  elif [[ -n "$password" ]]; then
    if command -v sshpass >/dev/null 2>&1; then
      SSHPASS="$password" sshpass -e "$ssh_bin" "${ssh_options[@]}" "$target" "$remote_command" \
        <"$fifo" >"$status_file" 2>"$stderr_file" &
    else
      askpass_script="$lease_state_dir/${resource}.askpass.sh"
      printf '%s\n' '#!/usr/bin/env bash' 'printf "%s\n" "$LEASE_ASKPASS_SECRET"' >"$askpass_script"
      chmod 700 "$askpass_script"
      DISPLAY="${DISPLAY:-internnav-lease}" SSH_ASKPASS_REQUIRE=force \
        SSH_ASKPASS="$askpass_script" LEASE_ASKPASS_SECRET="$password" \
        "$ssh_bin" "${ssh_options[@]}" "$target" "$remote_command" \
        <"$fifo" >"$status_file" 2>"$stderr_file" &
    fi
  else
    "$ssh_bin" "${ssh_options[@]}" "$target" "$remote_command" \
      <"$fifo" >"$status_file" 2>"$stderr_file" &
  fi
  pid=$!

  holder_pid["$resource"]="$pid"
  holder_fd["$resource"]="$control_fd"
  holder_fifo["$resource"]="$fifo"
  holder_status["$resource"]="$status_file"
  holder_stderr["$resource"]="$stderr_file"

  local waited=0
  while (( waited < acquire_timeout * 10 )); do
    if grep -q '^ACQUIRED ' "$status_file" 2>/dev/null; then
      held_resources+=("$resource")
      cat "$status_file"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" 2>/dev/null || true
      cat "$status_file" >&2 2>/dev/null || true
      cat "$stderr_file" >&2 2>/dev/null || true
      close_fd "$control_fd"
      return 1
    fi
    sleep 0.1
    (( waited += 1 ))
  done

  printf 'resource lease: timed out acquiring %s at %s\n' "$resource" "$target" >&2
  kill -TERM "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  cat "$status_file" >&2 2>/dev/null || true
  cat "$stderr_file" >&2 2>/dev/null || true
  close_fd "$control_fd"
  return 1
}

for resource in "${requested_resources[@]}"; do
  if ! start_ssh_holder "$resource"; then
    # EXIT trap releases an earlier DGX holder if Isaac acquisition failed.
    die "$EXIT_BUSY" "failed to acquire $resource; wrapped command was not started"
  fi
done

command -v setsid >/dev/null 2>&1 || \
  die "$EXIT_UNAVAILABLE" 'setsid is required to contain the wrapped workload'

{
  printf 'state=HELD\n'
  printf 'owner=%s\n' "$(sanitize_local_field "$owner")"
  printf 'task=%s\n' "$(sanitize_local_field "$task")"
  printf 'resources=%s\n' "${requested_resources[*]}"
  printf 'started_at=%s\n' "$started_at"
  printf 'command=%s\n' "$(sanitize_local_field "$command_text")"
  printf 'log_dir=%s\n' "$(sanitize_local_field "$log_dir")"
  printf 'cleanup_timeout_sec=%s\n' "$cleanup_timeout"
  printf 'kill_wait_timeout_sec=%s\n' "$kill_wait_timeout"
  printf 'recovery_only=%s\n' "$recovery_only"
  printf 'recovery_marker=%s\n' "$recovery_marker"
} >"$log_dir/lease_metadata.txt"

set +e
wrapped_pgid_file="$lease_state_dir/wrapped.pgid"
setsid --wait bash -c \
  'pgid_file="$1"; shift; printf "%s\n" "$$" >"$pgid_file"; exec "$@"' \
  lease-command "$wrapped_pgid_file" "${command_argv[@]}" \
  > >(tee "$log_dir/command.stdout.log") \
  2> >(tee "$log_dir/command.stderr.log" >&2) &
wrapped_pid=$!
set -e

# The session leader records its own PID before exec. Do not proceed without a
# confirmed process group: killing only the direct wrapper could leave remote
# or background descendants running after the lease is lost.
for _ in 1 2 3 4 5 6 7 8 9 10; do
  [[ -s "$wrapped_pgid_file" ]] && break
  kill -0 "$wrapped_pid" 2>/dev/null || break
  sleep 0.1
done
if [[ -s "$wrapped_pgid_file" ]]; then
  read -r wrapped_pgid <"$wrapped_pgid_file"
fi
if [[ ! "$wrapped_pgid" =~ ^[1-9][0-9]*$ ]]; then
  terminate_wrapped_group TERM
  wait "$wrapped_pid" 2>/dev/null || true
  wrapped_pid=""
  die "$EXIT_UNAVAILABLE" 'wrapped workload process group could not be confirmed'
fi

lease_lost=""
while kill -0 "$wrapped_pid" 2>/dev/null; do
  for resource in "${held_resources[@]}"; do
    if ! kill -0 "${holder_pid[$resource]}" 2>/dev/null; then
      lease_lost="$resource"
      break 2
    fi
  done
  sleep 0.5
done

if [[ -n "$lease_lost" ]]; then
  printf 'resource lease: lost %s holder; terminating wrapped command\n' "$lease_lost" >&2
  drain_wrapped_group "lease_lost:$lease_lost" || true
  printf 'state=LEASE_LOST\nresource=%s\nended_at=%s\n' \
    "$lease_lost" "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" >>"$log_dir/lease_metadata.txt"
  exit "$EXIT_LEASE_LOST"
fi

set +e
wait "$wrapped_pid"
command_rc=$?
set -e
if wrapped_group_is_alive; then
  # A coordinator that exits while leaving a background descendant must not
  # cause lock release.  Drain that PGID under the still-live holders and make
  # the invocation fail even if the session leader reported success.
  drain_wrapped_group command_left_descendants || true
  command_rc="$EXIT_LEASE_LOST"
else
  write_cleanup_receipt command_completed false true false true
fi
wrapped_pid=""
wrapped_pgid=""
printf 'state=RELEASED\ncommand_exit=%s\nended_at=%s\n' \
  "$command_rc" "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" >>"$log_dir/lease_metadata.txt"
exit "$command_rc"
