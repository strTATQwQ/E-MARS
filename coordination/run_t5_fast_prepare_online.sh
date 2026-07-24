#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: run_t5_fast_prepare_online.sh CODE_SHA RUN_ID RESULT_ROOT

Engineering-only exact-ref T5 preparation.  It does not read or consume the
formal board grant.  RUN_ID must be t5d00YYYYMMDDtHHMMSS and RESULT_ROOT must
be results/internnav_t5/d0-0-prepare-RUN_ID.  DGX-A, DGX-B, and the shared x86
prepare task run concurrently under dgx-a, dgx-b, and isaac leases.
EOF
  exit 64
}

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
prepare_scope="${INTERNNAV_T5_FAST_PREPARE_SCOPE:-dual}"
case "$prepare_scope" in
  dual) ;;
  lane-a)
    test "${INTERNNAV_T5_LANE_A_PREPARE_ENTRY:-}" = 1 || {
      echo "Lane-A-only prepare must use run_t5_fast_prepare_lane_a_online.sh" >&2
      exit 64
    }
    ;;
  *) usage ;;
esac
ssh_options=(-T -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=2)
dgx_a_target=railgun@10.100.100.128
dgx_b_target=rail@10.100.120.122
x86_ip="${ISAAC_HOST:-10.100.120.123}"
[[ "$x86_ip" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]] || {
  echo "ISAAC_HOST must be an IPv4 address" >&2
  exit 64
}
test "$x86_ip" = 10.100.120.123 || {
  echo "active T5 Isaac host is fixed at 10.100.120.123" >&2
  exit 64
}
x86_target="song@$x86_ip"
lane_a_cpuset="${INTERNVLA_T5_LANE_A_CPUSET:-0,2,4,6,8,10,12,14,16}"
lane_b_cpuset="${INTERNVLA_T5_LANE_B_CPUSET:-1,3,5,7,9,11,13,15,17}"
[[ "$lane_a_cpuset" =~ ^[0-9,-]+$ && "$lane_b_cpuset" =~ ^[0-9,-]+$ ]] || {
  echo "Isaac Lane CPU sets contain unsafe characters" >&2
  exit 64
}
test "$lane_a_cpuset" = 0,2,4,6,8,10,12,14,16
test "$lane_b_cpuset" = 1,3,5,7,9,11,13,15,17

remote() {
  local target="$1"; shift
  ssh "${ssh_options[@]}" "$target" "$@"
}

source "$root/scripts/t5_quarantine_common.sh"
source "$root/scripts/t5_remote_compute_audit_common.sh"

git_command=(git -C "$root")
if [[ -f "$root/.git" ]] && grep -Eq '^gitdir: [A-Za-z]:/' "$root/.git"; then
  command -v git.exe >/dev/null
  command -v wslpath >/dev/null
  git_command=(git.exe -C "$(wslpath -w "$root")")
fi

validate_inputs() {
  local code_sha="$1" run_id="$2" result_relative="$3"
  [[ "$code_sha" =~ ^[0-9a-f]{40}$ ]] || usage
  [[ "$run_id" =~ ^t5d00[0-9]{8}t[0-9]{6}$ ]] || usage
  test "$result_relative" = "results/internnav_t5/d0-0-prepare-$run_id" || usage
}

ensure_exact_clean_ref() {
  local code_sha="$1"
  "${git_command[@]}" cat-file -e "$code_sha^{commit}"
  test "$("${git_command[@]}" rev-parse HEAD | tr -d '\r')" = "$code_sha"
  test -z "$("${git_command[@]}" status --porcelain --untracked-files=all | tr -d '\r')"
}

deployment_values() {
  local code_sha="$1" run_id="$2"
  tag="${run_id}-${code_sha:0:12}"
  dgx_a_root="/home/railgun/internnav-t1-t2/.t5-deployments/${tag}-lane-a"
  dgx_b_root="/home/rail/internnav-t1-t2/.t5-deployments/${tag}-lane-b"
  x86_prepare_root="/home/song/internnav-t1-t2/.t5-deployments/${tag}-isaac-prepare"
  x86_a_root="/home/song/internnav-t1-t2/.t5-deployments/${tag}-isaac-a"
  x86_b_root="/home/song/internnav-t1-t2/.t5-deployments/${tag}-isaac-b"
}

deploy_archive() {
  local target="$1" stage_archive="$2"; shift 2
  local destination archive="$result_dir/deployment.tar.gz"
  test "$(sha256sum "$archive" | cut -d' ' -f1)" = "$archive_sha256"
  remote "$target" "set -euo pipefail; umask 077; test ! -e '$stage_archive'; : >'$stage_archive'"
  ssh "${ssh_options[@]}" "$target" "cat >'$stage_archive'" <"$archive"
  remote "$target" \
    "test \"\$(sha256sum '$stage_archive' | cut -d' ' -f1)\" = '$archive_sha256'"
  for destination in "$@"; do
    remote "$target" \
      "set -euo pipefail; test ! -e '$destination'; install -d -m 700 '$destination'; gzip -dc '$stage_archive' | tar -C '$destination' -xf -; printf '%s\n' '$code_sha' >'$destination/T5_DEPLOYMENT_REF'; printf '%s\n' '$archive_sha256' >'$destination/T5_DEPLOYMENT_ARCHIVE_SHA256'; test \"\$(cat '$destination/T5_DEPLOYMENT_REF')\" = '$code_sha'; test \"\$(cat '$destination/T5_DEPLOYMENT_ARCHIVE_SHA256')\" = '$archive_sha256'"
  done
  remote "$target" "rm -f -- '$stage_archive'"
}

record_deployment_bindings() {
  local target="$1" output="$2"; shift 2
  local destination evidence="${output%.json}.txt"
  : >"$evidence"
  for destination in "$@"; do
    printf 'root=%s\n' "$destination" >>"$evidence"
    remote "$target" "cat '$destination/T5_DEPLOYMENT_REF'" >>"$evidence"
    remote "$target" "cat '$destination/T5_DEPLOYMENT_ARCHIVE_SHA256'" >>"$evidence"
  done
  python3 - "$evidence" "$output" "$code_sha" "$archive_sha256" <<'PY'
import json,sys,time
from pathlib import Path
lines=Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
records=[]
for index in range(0,len(lines),3):
    if index + 2 >= len(lines) or not lines[index].startswith("root="):
        raise SystemExit("malformed deployment binding evidence")
    records.append({"root":lines[index][5:],"code_ref_sha":lines[index+1],
                    "deployment_archive_sha256":lines[index+2]})
checks={"records_present":bool(records),
        "code_ref_exact":all(value["code_ref_sha"]==sys.argv[3] for value in records),
        "archive_sha_exact":all(value["deployment_archive_sha256"]==sys.argv[4] for value in records),
        "archive_bytes_verified_before_extract":True}
payload={"schema_version":1,"status":"PASS" if all(checks.values()) else "FAIL",
         "records":records,"checks":checks,"recorded_unix":time.time()}
Path(sys.argv[2]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
raise SystemExit(0 if payload["status"]=="PASS" else 1)
PY
}

read -r -d '' residual_audit_program <<'REMOTE_RESIDUAL_AUDIT' || true
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

role = sys.argv[1]
roots = tuple(value for value in sys.argv[2].split("|") if value)
ownership_dir = None if sys.argv[3] == "-" else Path(sys.argv[3])
if role not in {"dgx_a", "dgx_b", "x86", "x86_a"} or not roots:
    raise SystemExit(75)
if any(not value.startswith("/") for value in roots):
    raise SystemExit(75)

errors = []
ancestors = set()
cursor = os.getpid()
while cursor > 1 and cursor not in ancestors:
    ancestors.add(cursor)
    try:
        lines = Path(f"/proc/{cursor}/status").read_text(encoding="utf-8").splitlines()
        cursor = int(next(line.split()[1] for line in lines if line.startswith("PPid:")))
    except (OSError, StopIteration, ValueError) as error:
        errors.append({"pid": cursor, "error": f"ancestor:{type(error).__name__}"})
        break

def scoped_processes():
    found = []
    try:
        entries = sorted(Path("/proc").iterdir(), key=lambda item: item.name)
    except OSError as error:
        errors.append({"pid": None, "error": f"proc_root:{type(error).__name__}"})
        return found
    for entry in entries:
        if not entry.name.isdigit() or int(entry.name) in ancestors:
            continue
        process_id = int(entry.name)
        try:
            raw = (entry / "cmdline").read_bytes()
        except FileNotFoundError:
            continue
        except OSError as error:
            errors.append({"pid": process_id, "error": type(error).__name__})
            continue
        arguments = [
            item.decode("utf-8", errors="replace")
            for item in raw.split(b"\0") if item
        ]
        command = " ".join(arguments)
        root_bound = any(
            argument == root or argument.startswith(root + "/")
            for argument in arguments for root in roots
        )
        if command and root_bound:
            found.append({"pid": process_id, "command": command})
    return found

containers = {}
x86_role = role in {"x86", "x86_a"}
if x86_role:
    if role == "x86":
        expected_roots = {"a": roots[1], "b": roots[2]} if len(roots) == 3 else {}
        selected_containers = (("a", "internnav_t5_isaac_a"),
                               ("b", "internnav_t5_isaac_b"))
        if set(expected_roots) != {"a", "b"}:
            errors.append({"pid": None, "error": "x86_root_scope_mismatch"})
    else:
        expected_roots = {"a": roots[1]} if len(roots) == 2 else {}
        selected_containers = (("a", "internnav_t5_isaac_a"),)
        if set(expected_roots) != {"a"}:
            errors.append({"pid": None, "error": "x86_a_root_scope_mismatch"})
    for lane, name in selected_containers:
        marker = ownership_dir / name if ownership_dir else None
        owned = bool(marker and marker.is_file())
        stop_exit = None
        before = subprocess.run(
            ["docker", "inspect", name], text=True, capture_output=True, timeout=10
        )
        if owned and before.returncode == 0:
            try:
                value = json.loads(before.stdout)[0]
                labels = value.get("Config", {}).get("Labels") or {}
                if labels.get("internnav.t5.deployment_root") != expected_roots.get(lane):
                    raise ValueError("owned_container_root_mismatch")
                stopped = subprocess.run(
                    ["docker", "stop", "-t", "10", name],
                    text=True, capture_output=True, timeout=15,
                )
                stop_exit = stopped.returncode
            except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                errors.append({"pid": None, "error": f"{name}:{error}"})
        after = subprocess.run(
            ["docker", "inspect", name], text=True, capture_output=True, timeout=10
        )
        if after.returncode == 0:
            try:
                value = json.loads(after.stdout)[0]
                running = bool(value["State"]["Running"])
                pid = int(value["State"]["Pid"])
                absent_or_stopped = not running and pid == 0
            except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                errors.append({"pid": None, "error": f"{name}:{type(error).__name__}"})
                running, pid, absent_or_stopped = None, None, False
        else:
            running, pid, absent_or_stopped = False, 0, True
        containers[lane] = {
            "name": name,
            "owned_by_this_run": owned,
            "stop_exit": stop_exit,
            "running": running,
            "pid": pid,
            "absent_or_stopped": absent_or_stopped,
        }

processes = scoped_processes()
try:
    sockets_result = subprocess.run(
        ["ss", "-H", "-lntup"], text=True, capture_output=True, timeout=10
    )
except (OSError, subprocess.TimeoutExpired) as error:
    errors.append({"pid": None, "error": f"socket_audit:{type(error).__name__}"})
    sockets_result = None
ports = ((25137, 25139, 25140, 25141) if role == "x86_a" else
         (25137, 25138, 25139, 25140, 25141, 25239, 25240, 25241))
sockets = []
if sockets_result is not None:
    if sockets_result.returncode != 0:
        errors.append({"pid": None, "error": "socket_audit_nonzero"})
    sockets = [
        line for line in sockets_result.stdout.splitlines()
        if any(re.search(rf":{port}(?:\s|$)", line) for port in ports)
    ]

health_sockets_absent = True
runtime_locks_available = True
if x86_role:
    health_paths = (("/tmp/internnav_t5_a_ipc/isaac_health.sock",)
                    if role == "x86_a" else
                    ("/tmp/internnav_t5_a_ipc/isaac_health.sock",
                     "/tmp/internnav_t5_b_ipc/isaac_health.sock"))
    health_sockets_absent = all(not os.path.lexists(path) for path in health_paths)
    runtime_locks = (("/tmp/internnav_t5_isaac_a_runtime.lock",)
                     if role == "x86_a" else
                     ("/tmp/internnav_t5_isaac_a_runtime.lock",
                      "/tmp/internnav_t5_isaac_b_runtime.lock",
                      "/tmp/internnav_t5_isaac_shared_assets.lock"))
    for lock in runtime_locks:
        probe = subprocess.run(["flock", "-n", lock, "true"], capture_output=True)
        runtime_locks_available = runtime_locks_available and probe.returncode == 0

checks = {
    "deployment_processes_absent": not processes,
    "t5_sockets_absent": not sockets,
    "owned_containers_stopped_or_absent": not x86_role
        or all(value["absent_or_stopped"] for value in containers.values()),
    "x86_health_sockets_absent": not x86_role or health_sockets_absent,
    "x86_runtime_locks_available": not x86_role or runtime_locks_available,
    "errors_empty": not errors,
}
payload = {
    "schema_version": 1,
    "status": "PASS" if all(checks.values()) else ("ERROR" if errors else "FAIL"),
    "role": role,
    "deployment_roots": list(roots),
    "processes": processes,
    "sockets": sockets,
    "containers": containers,
    "errors": errors,
    "checks": checks,
    "recorded_unix": time.time(),
}
print(json.dumps(payload, sort_keys=True))
raise SystemExit(0 if payload["status"] == "PASS" else 75)
REMOTE_RESIDUAL_AUDIT
residual_audit_b64="$(printf '%s' "$residual_audit_program" | base64 | tr -d '\r\n')"

# The prepare/build command is a remote setsid session.  Its immutable ledger
# binds cleanup to PID/PGID/SID/starttime/argv and deployment root.  If the
# leader has already disappeared, only current groups with a distinct argv
# element rooted at the deployment path are eligible for the fallback signal.
# A same-host SSH coordinator may contain that path inside its remote command
# string; substring matching would signal the coordinator itself.
read -r -d '' prepare_supervisor_program <<'REMOTE_PREP_SUPERVISOR_CONTROL' || true
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path

ledger_path, expected_root, action = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
value = json.loads(ledger_path.read_text(encoding="utf-8"))
if value.get("run_root") != expected_root:
    raise SystemExit("prepare supervisor run_root mismatch")
pid, pgid, sid, starttime = (
    value.get("pid"), value.get("pgid"), value.get("sid"), value.get("starttime")
)
argv_sha256 = value.get("argv_sha256")
if not all(isinstance(item, int) and item > 1 for item in (pid, pgid, sid, starttime)):
    raise SystemExit("invalid prepare supervisor identity")
if pgid != sid or not isinstance(argv_sha256, str) or len(argv_sha256) != 64:
    raise SystemExit("unsafe prepare supervisor session identity")
if action not in {"AUDIT", "TERM", "KILL"}:
    raise SystemExit("unsupported prepare supervisor action")

ancestors = set()
cursor = os.getpid()
while cursor > 1 and cursor not in ancestors:
    ancestors.add(cursor)
    try:
        status = Path(f"/proc/{cursor}/status").read_text(encoding="utf-8")
        cursor = int(next(line.split()[1] for line in status.splitlines()
                          if line.startswith("PPid:")))
    except (OSError, StopIteration, ValueError):
        break

table = []
for entry in Path("/proc").iterdir():
    if not entry.name.isdigit() or int(entry.name) in ancestors:
        continue
    candidate = int(entry.name)
    try:
        tail = (entry / "stat").read_text(encoding="utf-8").rsplit(")", 1)[1].strip().split()
        raw_command = (entry / "cmdline").read_bytes()
        arguments = [
            item.decode(errors="replace")
            for item in raw_command.split(b"\0") if item
        ]
        table.append({
            "pid": candidate,
            "pgid": int(tail[2]),
            "sid": int(tail[3]),
            "starttime": int(tail[19]),
            "command": raw_command.replace(b"\0", b" ").decode(errors="replace").strip(),
            "root_bound": any(
                argument == expected_root
                or argument.startswith(expected_root + "/")
                for argument in arguments
            ),
            "argv_sha256": hashlib.sha256(raw_command).hexdigest(),
        })
    except (OSError, ValueError, IndexError):
        continue

leader = next((row for row in table if row["pid"] == pid), None)
if leader is not None and (
    leader["pgid"] != pgid
    or leader["sid"] != sid
    or leader["starttime"] != starttime
    or leader["argv_sha256"] != argv_sha256
    or leader["root_bound"] is not True
):
    raise SystemExit("prepare supervisor identity drift")
members = [row for row in table if row["pgid"] == pgid]
run_processes = [row for row in table if row["root_bound"] is True]
groups = {row["pgid"] for row in run_processes}
if members:
    groups.add(pgid)
signalled_groups = []
if action in {"TERM", "KILL"}:
    requested_signal = signal.SIGTERM if action == "TERM" else signal.SIGKILL
    for candidate in sorted(groups):
        rows = [row for row in table if row["pgid"] == candidate]
        if not rows:
            continue
        if not any(row["root_bound"] is True for row in rows):
            raise SystemExit(f"refusing unassociated prepare PGID {candidate}")
        os.killpg(candidate, requested_signal)
        signalled_groups.append(candidate)
payload = {
    "schema_version": 1,
    "ledger": str(ledger_path),
    "run_root": expected_root,
    "action": action,
    "supervisor": {"pid": pid, "pgid": pgid, "sid": sid,
                   "starttime": starttime, "argv_sha256": argv_sha256},
    "leader": leader,
    "members": members,
    "run_processes": run_processes,
    "signalled_groups": signalled_groups,
    "absent": not members and not run_processes,
    "recorded_unix": time.time(),
}
print(json.dumps(payload, sort_keys=True))
REMOTE_PREP_SUPERVISOR_CONTROL
prepare_supervisor_b64="$(printf '%s' "$prepare_supervisor_program" | base64 | tr -d '\r\n')"

remote_prepare_supervisor_action() {
  local target="$1" ledger="$2" run_root="$3" action="$4" output rc
  output="$(remote "$target" \
    "python3 -c \"\$(printf '%s' '$prepare_supervisor_b64' | base64 -d)\" '$ledger' '$run_root' '$action'" 2>&1)" && rc=0 || rc=$?
  printf '%s\n' "$output" >>"$audit_dir/supervisor_cleanup_events.jsonl"
  printf '%s\n' "$output"
  return "$rc"
}

remote_prepare_supervisor_absent() {
  local output
  output="$(remote_prepare_supervisor_action "$1" "$2" "$3" AUDIT)" || return 1
  python3 -c 'import json,sys;raise SystemExit(0 if json.loads(sys.argv[1]).get("absent") else 1)' "$output"
}

local_prepare_supervisor_action() {
  local ledger="$1" run_root="$2" action="$3" output rc
  output="$(python3 -c "$prepare_supervisor_program" "$ledger" "$run_root" "$action" 2>&1)" && rc=0 || rc=$?
  printf '%s\n' "$output" >>"$result_dir/audits/outer_cleanup_events.jsonl"
  printf '%s\n' "$output"
  return "$rc"
}

local_prepare_supervisor_absent() {
  local output
  output="$(local_prepare_supervisor_action "$1" "$2" AUDIT)" || return 1
  python3 -c 'import json,sys;raise SystemExit(0 if json.loads(sys.argv[1]).get("absent") else 1)' "$output"
}

run_residual_audit() {
  local target="$1" role="$2" roots="$3" ownership="$4" output="$5"
  local raw="${output%.json}.remote.json" stderr_log="${output%.json}.stderr.log" remote_exit
  mkdir -p -- "$(dirname -- "$output")"
  set +e
  remote "$target" \
    "python3 -c \"\$(printf '%s' '$residual_audit_b64' | base64 -d)\" '$role' '$roots' '$ownership'" \
    >"$raw" 2>"$stderr_log"
  remote_exit=$?
  python3 - "$raw" "$output" "$role" "$remote_exit" <<'PY'
import json, os, sys, time
from pathlib import Path

raw, output = Path(sys.argv[1]), Path(sys.argv[2])
role, remote_exit = sys.argv[3], int(sys.argv[4])
audit = None
parse_error = None
try:
    audit = json.loads(raw.read_text(encoding="utf-8"))
except (OSError, UnicodeError, json.JSONDecodeError) as error:
    parse_error = type(error).__name__
checks = {
    "remote_exit_zero": remote_exit == 0,
    "audit_object": isinstance(audit, dict),
    "audit_status_pass": isinstance(audit, dict) and audit.get("status") == "PASS",
    "role_exact": isinstance(audit, dict) and audit.get("role") == role,
    "remote_checks_pass": isinstance(audit, dict) and bool(audit.get("checks"))
        and all(value is True for value in audit["checks"].values()),
    "remote_errors_empty": isinstance(audit, dict) and audit.get("errors") == [],
}
payload = {
    "schema_version": 1,
    "status": "PASS" if all(checks.values()) else "FAIL",
    "role": role,
    "remote_exit": remote_exit,
    "remote_audit": audit,
    "parse_error": parse_error,
    "checks": checks,
    "recorded_unix": time.time(),
}
temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, output)
raise SystemExit(0 if payload["status"] == "PASS" else 75)
PY
}

write_cleanup_receipt() {
  local role="$1" supervisor="$2" residual="$3" global="$4" output="$5"
  python3 - "$role" "$supervisor" "$residual" "$global" "$output" <<'PY'
import json, os, sys, time
from pathlib import Path

role = sys.argv[1]
supervisor_path, residual_path, global_path, output = map(Path, sys.argv[2:6])
def load(path):
    try: return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError): return None
supervisor = load(supervisor_path)
residual = load(residual_path)
global_audit = load(global_path)
checks = {
    "evidence_bound_supervisor_cleanup_pass": isinstance(supervisor, dict)
        and supervisor.get("status") == "PASS"
        and bool(supervisor.get("checks"))
        and all(value is True for value in supervisor["checks"].values()),
    "structured_residual_audit_pass": isinstance(residual, dict)
        and residual.get("status") == "PASS"
        and bool(residual.get("checks"))
        and all(value is True for value in residual["checks"].values()),
    "global_forbidden_compute_audit_pass": isinstance(global_audit, dict)
        and global_audit.get("status") == "PASS"
        and bool(global_audit.get("checks"))
        and all(value is True for value in global_audit["checks"].values())
}
payload = {
    "schema_version": 1,
    "status": "PASS" if all(checks.values()) else "FAIL",
    "role": role,
    "supervisor_cleanup_evidence": supervisor,
    "residual_evidence": residual,
    "global_compute_evidence": global_audit,
    "checks": checks,
    "recorded_unix": time.time(),
}
temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, output)
raise SystemExit(0 if payload["status"] == "PASS" else 75)
PY
}

collect_remote_directory() {
  local target="$1" source="$2" destination="$3"
  mkdir -p "$destination"
  remote "$target" "tar -C '$source' -czf - ." >"$destination.tar.gz"
  tar -C "$destination" -xzf "$destination.tar.gz"
}

read_frozen_values() {
  mapfile -t frozen_values < <(python3 - "$result_dir/fast_prepare_input.json" <<'PY'
import json, sys
value=json.load(open(sys.argv[1],encoding="utf-8"))
for key in ("model_revision", "checkpoint_revision", "model_weight_inventory_sha256",
            "checkpoint_content_manifest_canonical_sha256", "dataset_root",
            "dataset_file_sha256"):
    print(value[key])
PY
  )
  model_revision="${frozen_values[0]}"
  checkpoint_revision="${frozen_values[1]}"
  model_weight_inventory_sha256="${frozen_values[2]}"
  checkpoint_content_manifest_sha256="${frozen_values[3]}"
  dataset_root="${frozen_values[4]}"
  dataset_sha256="${frozen_values[5]}"
}

read -r -d '' dgx_build_program <<'REMOTE_DGX_BUILD' || true
set -euo pipefail
deployment_root="$1"
expected_model_revision="$2"
expected_checkpoint_revision="$3"
expected_weight_inventory_sha256="$4"
expected_checkpoint_manifest_sha256="$5"
supervisor_ledger="$6"
build_role="$7"
case "$build_role" in dgx_a|dgx_b) ;; *) exit 64 ;; esac
audit="$deployment_root/results/fast_prepare"
workspace="$deployment_root/ros_ws"
mkdir -p "$audit" "$workspace/src"
if test "$build_role" = dgx_b; then
  ln -sfn "$deployment_root/internnav_t5_lane_b_msgs" \
    "$workspace/src/internnav_t5_lane_b_msgs"
fi
pgid="$(ps -o pgid= -p "$$" | tr -d ' ')"
sid="$(ps -o sid= -p "$$" | tr -d ' ')"
starttime="$(awk '{print $22}' "/proc/$$/stat")"
argv_sha256="$(sha256sum "/proc/$$/cmdline" | cut -d' ' -f1)"
python3 - "$supervisor_ledger" "$deployment_root" "$$" "$pgid" "$sid" \
  "$starttime" "$argv_sha256" dgx <<'PY'
import json,os,sys,time
from pathlib import Path
output=Path(sys.argv[1])
payload={"schema_version":2,"run_root":sys.argv[2],"pid":int(sys.argv[3]),
 "pgid":int(sys.argv[4]),"sid":int(sys.argv[5]),"starttime":int(sys.argv[6]),
 "argv_sha256":sys.argv[7],"role":sys.argv[8],"started_unix":time.time()}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
started="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
write_ledger() {
  local state="$1" rc="$2"
  printf 'state=%s\npid=%s\npgid=%s\nstarted_at=%s\nended_at=%s\nexit_code=%s\ncommand=build_t4_host_ros.sh\n' \
    "$state" "$$" "$pgid" "$started" "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$rc" \
    >"$audit/build_ledger.txt"
}
finish_build() {
  local rc=$?
  trap - EXIT HUP INT TERM
  if test "$rc" = 0; then write_ledger PASS 0; else write_ledger FAIL "$rc"; fi
  exit "$rc"
}
trap finish_build EXIT
trap 'exit 130' HUP INT TERM
write_ledger RUNNING -1

"$HOME/internnav-t0/venv-model/bin/python" - "$audit/model_runtime_environment.json" <<'PY'
import importlib.metadata, json, sys, time
from pathlib import Path
import torch
packages={}
for name in ("torch","transformers","safetensors","numpy"):
    try: packages[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: packages[name]=None
checks={"required_packages_present":all(packages.values()),
        "cuda_available":torch.cuda.is_available(),
        "cuda_device_count_nonzero":torch.cuda.device_count() >= 1}
payload={"schema_version":1,"status":"PASS" if all(checks.values()) else "FAIL",
         "python_executable":sys.executable,"packages":packages,
         "cuda_device_count":torch.cuda.device_count(),"checks":checks,
         "recorded_unix":time.time()}
Path(sys.argv[1]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
raise SystemExit(0 if payload["status"]=="PASS" else 1)
PY
python3 "$deployment_root/scripts/validate_t5_frozen_assets.py" checkpoint \
  --golden "$deployment_root/configs/internnav_t5/golden_bundle_manifest.json" \
  --content-manifest "$deployment_root/configs/internnav_t5/checkpoint_content_manifest.json" \
  --internnav-root "$HOME/internnav-t0/InternNav" \
  --output "$audit/model_inventory.json"
python3 - "$audit/model_inventory.json" "$expected_model_revision" \
  "$expected_checkpoint_revision" "$expected_weight_inventory_sha256" \
  "$expected_checkpoint_manifest_sha256" <<'PY'
import json,sys
value=json.load(open(sys.argv[1],encoding="utf-8"))
checks={"model_revision":value.get("model_revision")==sys.argv[2],
        "checkpoint_revision":value.get("checkpoint_revision")==sys.argv[3],
        "runtime_inventory":value.get("expected_runtime_weight_inventory_sha256")==sys.argv[4],
        "content_manifest":value.get("checkpoint_content_manifest_canonical_sha256")==sys.argv[5]}
raise SystemExit(0 if value.get("status")=="PASS" and all(checks.values()) else 1)
PY
export PYTHONDONTWRITEBYTECODE=1
set +u
source /opt/ros/jazzy/setup.bash
set -u
INTERNNAV_T1_CONTROL_ROOT="$deployment_root" \
INTERNVLA_ROS_WS="$workspace" \
INTERNNAV_RUNTIME_POLICY=completion_sim \
INTERNNAV_SIMULATION_TARGET=isaac \
INTERNNAV_T4_RESOURCE_LEASE_ACK=dgx \
  bash "$deployment_root/scripts/build_t4_host_ros.sh" \
  >"$audit/ros_build.log" 2>&1
if test "$build_role" = dgx_b; then
  INTERNNAV_T1_CONTROL_ROOT="$deployment_root" \
  INTERNVLA_ROS_WS="$workspace" \
  INTERNNAV_RUNTIME_POLICY=completion_sim \
  INTERNNAV_SIMULATION_TARGET=isaac \
  INTERNNAV_T5_LANE=b \
  INTERNNAV_T5_LANE_NAMESPACE=/t5/lane_b \
  INTERNNAV_T4_RESOURCE_LEASE_ACK=dgx \
    bash "$deployment_root/scripts/build_t5_lane_b_step3_ros.sh" \
    >"$audit/step3_lane_b_ros_build.log" 2>&1
fi
python3 - "$audit/build_summary.json" "$deployment_root" \
  "$audit/model_inventory.json" "$audit/model_runtime_environment.json" <<'PY'
import json,sys,time
from pathlib import Path
output,root,inventory_path,runtime_path=Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]),Path(sys.argv[4])
inventory=json.loads(inventory_path.read_text(encoding="utf-8"))
runtime=json.loads(runtime_path.read_text(encoding="utf-8"))
checks={"deployment_ref_present":(root/"T5_DEPLOYMENT_REF").is_file(),
        "archive_sha_present":(root/"T5_DEPLOYMENT_ARCHIVE_SHA256").is_file(),
        "model_inventory_pass":inventory.get("status")=="PASS",
        "model_runtime_pass":runtime.get("status")=="PASS" and all(runtime.get("checks",{}).values()),
        "ros_workspace_built":(root/"ros_ws/install/setup.bash").is_file()}
payload={"schema_version":1,"status":"PASS" if all(checks.values()) else "FAIL",
         "deployment_root":str(root),"checks":checks,"recorded_unix":time.time()}
output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
raise SystemExit(0 if payload["status"]=="PASS" else 1)
PY
REMOTE_DGX_BUILD
dgx_build_b64="$(printf '%s' "$dgx_build_program" | base64 | tr -d '\r\n')"

read -r -d '' x86_prepare_program <<'REMOTE_X86_PREPARE' || true
set -euo pipefail
deployment_root="$1"
dataset_root="$2"
expected_dataset_sha="$3"
lane_a_deployment_root="$4"
lane_b_deployment_root="$5"
supervisor_ledger="$6"
expected_isaac_ip="$7"
lane_a_cpuset="$8"
lane_b_cpuset="$9"
prepare_scope="${10:-dual}"
case "$prepare_scope" in dual|lane-a) ;; *) exit 64 ;; esac
audit="$deployment_root/results/fast_prepare"
mkdir -p "$audit" "$deployment_root/inputs"
pgid="$(ps -o pgid= -p "$$" | tr -d ' ')"
sid="$(ps -o sid= -p "$$" | tr -d ' ')"
starttime="$(awk '{print $22}' "/proc/$$/stat")"
argv_sha256="$(sha256sum "/proc/$$/cmdline" | cut -d' ' -f1)"
python3 - "$supervisor_ledger" "$deployment_root" "$$" "$pgid" "$sid" \
  "$starttime" "$argv_sha256" x86 <<'PY'
import json,os,sys,time
from pathlib import Path
output=Path(sys.argv[1])
payload={"schema_version":2,"run_root":sys.argv[2],"pid":int(sys.argv[3]),
 "pgid":int(sys.argv[4]),"sid":int(sys.argv[5]),"starttime":int(sys.argv[6]),
 "argv_sha256":sys.argv[7],"role":sys.argv[8],"started_unix":time.time()}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
started="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
containers_owned="$audit/containers_owned"
write_ledger() {
  local state="$1" rc="$2"
  printf 'state=%s\npid=%s\npgid=%s\nstarted_at=%s\nended_at=%s\nexit_code=%s\ncommand=dataset_audit_then_static_maps_then_worker_prepare\n' \
    "$state" "$$" "$pgid" "$started" "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$rc" \
    >"$audit/prepare_ledger.txt"
}
stop_workers() {
  local container
  local containers=(internnav_t5_isaac_a internnav_t5_isaac_b)
  test "$prepare_scope" = dual || containers=(internnav_t5_isaac_a)
  for container in "${containers[@]}"; do
    test ! -f "$containers_owned/$container" || docker stop -t 10 "$container" >/dev/null 2>&1 || true
  done
}
finish_prepare() {
  local rc=$?
  trap - EXIT HUP INT TERM
  stop_workers
  if test "$rc" = 0; then write_ledger PASS 0; else write_ledger FAIL "$rc"; fi
  exit "$rc"
}
trap finish_prepare EXIT
trap 'exit 130' HUP INT TERM
write_ledger RUNNING -1

dataset="$dataset_root/val_unseen/val_unseen.json.gz"
actual_dataset_sha="$(sha256sum "$dataset" | cut -d' ' -f1)"
test "$actual_dataset_sha" = "$expected_dataset_sha"
python3 - "$deployment_root/configs/internnav_t5/d0_run_manifest.json" \
  "$dataset" "$audit/dataset_audit.json" "$actual_dataset_sha" <<'PY'
import gzip,json,sys
from pathlib import Path
manifest=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
with gzip.open(sys.argv[2],"rt",encoding="utf-8") as stream: payload=json.load(stream)
episodes=payload.get("episodes") if isinstance(payload,dict) else None
if not isinstance(episodes,list): raise SystemExit("dataset has no episode list")
keys=[f"{item['trajectory_id']}_{item['episode_id']}" for item in episodes]
fixed=manifest["fixed_input"]
checks={"sha_match":sys.argv[4]==fixed["dataset_file_sha256"],
        "episode_count_match":len(episodes)==fixed["episode_count"],
        "episode_keys_match":keys==fixed["episode_keys"]}
result={"schema_version":1,"status":"PASS" if all(checks.values()) else "FAIL",
        "dataset_file":sys.argv[2],"dataset_sha256":sys.argv[4],
        "episode_count":len(episodes),"episode_keys":keys,"checks":checks}
Path(sys.argv[3]).write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
raise SystemExit(0 if result["status"]=="PASS" else 1)
PY
static_output="$deployment_root/inputs/d0_fixed5_static_maps"
internnav_root="${INTERNNAV_ROOT:-$HOME/internnav-t0/InternNav}"
clearance=0.30
cache_tag="${clearance//./p}"
cache_dir="$deployment_root/runtime/static_maps/dgx_onboard_${actual_dataset_sha:0:16}_${cache_tag}"
{
  test -d "$internnav_root/data/scene_data/mp3d_pe"
  test ! -e "$static_output"
  mkdir -p "$(dirname "$static_output")" "$deployment_root/runtime/static_maps"
  if test ! -f "$cache_dir/manifest.json"; then
    test ! -e "$cache_dir"
    python3 "$deployment_root/scripts/build_t3_static_maps.py" \
      --dataset "$dataset" \
      --scene-root "$internnav_root/data/scene_data/mp3d_pe" \
      --output-root "$cache_dir" \
      --minimum-required-prefix-clearance-m "$clearance"
  fi
  mkdir "$static_output"
  cp "$cache_dir"/*.bin "$static_output/"
  python3 "$deployment_root/scripts/build_t4_truth_isolated_static_manifest.py" \
    --manifest "$cache_dir/manifest.json" \
    --dataset "$dataset" \
    --output "$static_output/manifest.json"
  python3 - "$static_output/map_prepare_status.json" "$dataset" \
    "$actual_dataset_sha" "$clearance" "$static_output/manifest.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
manifest=Path(sys.argv[5])
payload={"schema_version":1,"status":"PASS",
         "host_role":"isaac_offline_scene_map_builder",
         "dataset_file":sys.argv[2],"dataset_sha256":sys.argv[3],
         "minimum_clearance_m":float(sys.argv[4]),
         "manifest_sha256":hashlib.sha256(manifest.read_bytes()).hexdigest(),
         "destination_role":"dgx_onboard_compute"}
Path(sys.argv[1]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
PY
} >"$audit/static_map_prepare.log" 2>&1
python3 "$deployment_root/scripts/validate_t5_frozen_assets.py" static-map \
  --golden "$deployment_root/configs/internnav_t5/golden_bundle_manifest.json" \
  --manifest "$deployment_root/inputs/d0_fixed5_static_maps/manifest.json" \
  --output "$audit/static_map_source_audit.json"
install -d -m 700 "$containers_owned"
INTERNNAV_T1_CONTROL_ROOT=/home/song/internnav-t1-t2 \
INTERNVLA_ROS_WS=/home/song/internnav-t4/isaac_ros_ws_45 \
INTERNVLA_T5_ISAAC_WORKER_ROOT=/home/song/internnav-t1-t2/runtime/t5_isaac_workers \
INTERNVLA_T5_X86_LANE_A_ROOT="$lane_a_deployment_root" \
INTERNVLA_T5_X86_LANE_B_ROOT="$lane_b_deployment_root" \
INTERNVLA_T5_ISAAC_PREPARE_SCOPE="$(test "$prepare_scope" = lane-a && printf a || printf dual)" \
INTERNVLA_T5_CONTAINER_OWNERSHIP_DIR="$containers_owned" \
INTERNNAV_T5_RESOURCE_LEASE_ACK="$(test "$prepare_scope" = lane-a && printf lane-a || printf isaac)" \
INTERNNAV_RUNTIME_POLICY=completion_sim \
INTERNNAV_SIMULATION_TARGET=isaac \
INTERNVLA_T5_ISAAC_IP="$expected_isaac_ip" \
INTERNVLA_T5_LANE_A_CPUSET="$lane_a_cpuset" \
INTERNVLA_T5_LANE_B_CPUSET="$lane_b_cpuset" \
  bash "$deployment_root/scripts/prepare_t5_isaac_workers.sh" \
    "$audit/isaac_worker_prepare.json" \
    >"$audit/isaac_worker_prepare.log" 2>&1
stop_workers
lanes=(a b)
test "$prepare_scope" = dual || lanes=(a)
for lane in "${lanes[@]}"; do
  container="internnav_t5_isaac_$lane"
  docker inspect "$container" >"$audit/container_${lane}_inspect.json"
  python3 "$deployment_root/scripts/validate_t5_isaac_worker_spec.py" \
    "/home/song/internnav-t1-t2/runtime/t5_isaac_workers/$lane/expected_container_spec.json" \
    "$audit/container_${lane}_inspect.json" "$audit/container_${lane}_spec_audit.json"
done
if test "$prepare_scope" = dual; then
  python3 - "$audit/container_cleanup.json" "$audit/container_a_inspect.json" \
    "$audit/container_a_spec_audit.json" "$audit/container_b_inspect.json" \
    "$audit/container_b_spec_audit.json" <<'PY'
import json,sys,time
from pathlib import Path
lanes={}
for lane,inspect_path,audit_path in (("a",sys.argv[2],sys.argv[3]),("b",sys.argv[4],sys.argv[5])):
    value=json.loads(Path(inspect_path).read_text())[0]
    spec=json.loads(Path(audit_path).read_text())
    lanes[lane]={"container":value["Name"].lstrip("/"),"container_id":value["Id"],
                 "running":value["State"]["Running"],"pid":value["State"]["Pid"],
                 "image_id":value["Image"],"spec_audit":spec}
checks={"both_stopped":all(not value["running"] and value["pid"]==0 for value in lanes.values()),
        "exact_specs_after_stop":all(value["spec_audit"]["status"]=="PASS" for value in lanes.values()),
        "gpu_device_requests_match":all(value["spec_audit"]["checks"]["gpu_device_request"] for value in lanes.values())}
payload={"schema_version":1,"status":"PASS" if all(checks.values()) else "FAIL",
         "lanes":lanes,"checks":checks,"recorded_unix":time.time()}
Path(sys.argv[1]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
raise SystemExit(0 if payload["status"]=="PASS" else 1)
PY
else
  python3 - "$audit/container_cleanup.json" "$audit/container_a_inspect.json" \
    "$audit/container_a_spec_audit.json" <<'PY'
import json,sys,time
from pathlib import Path
value=json.loads(Path(sys.argv[2]).read_text())[0]
spec=json.loads(Path(sys.argv[3]).read_text())
lane={"container":value["Name"].lstrip("/"),"container_id":value["Id"],
      "running":value["State"]["Running"],"pid":value["State"]["Pid"],
      "image_id":value["Image"],"spec_audit":spec}
checks={"lane_a_stopped":not lane["running"] and lane["pid"]==0,
        "exact_spec_after_stop":spec["status"]=="PASS",
        "gpu0_device_request_match":spec["checks"]["gpu_device_request"]}
payload={"schema_version":1,"status":"PASS" if all(checks.values()) else "FAIL",
         "prepared_lanes":["a"],"lanes":{"a":lane},"checks":checks,
         "recorded_unix":time.time()}
Path(sys.argv[1]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
raise SystemExit(0 if payload["status"]=="PASS" else 1)
PY
fi
python3 - "$audit/x86_prepare_summary.json" "$audit/dataset_audit.json" \
  "$audit/static_map_source_audit.json" "$audit/isaac_worker_prepare.json" \
  "$audit/container_cleanup.json" <<'PY'
import json,sys,time
from pathlib import Path
values=[json.loads(Path(path).read_text(encoding="utf-8")) for path in sys.argv[2:]]
checks={"dataset_audit":values[0].get("status")=="PASS",
        "static_map_audit":values[1].get("status")=="PASS",
        "worker_prepare":values[2].get("status")=="PASS",
        "containers_stopped":values[3].get("status")=="PASS"}
payload={"schema_version":1,"status":"PASS" if all(checks.values()) else "FAIL",
         "checks":checks,"recorded_unix":time.time()}
Path(sys.argv[1]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
raise SystemExit(0 if payload["status"]=="PASS" else 1)
PY
REMOTE_X86_PREPARE
x86_prepare_b64="$(printf '%s' "$x86_prepare_program" | base64 | tr -d '\r\n')"

wait_for_map_ready() {
  local timeout="${INTERNNAV_T5_FAST_MAP_TIMEOUT_SEC:-3600}" deadline
  [[ "$timeout" =~ ^[1-9][0-9]*$ ]]
  deadline=$((SECONDS + timeout))
  while (( SECONDS < deadline )); do
    test ! -e "$result_dir/maps/x86_failed" || return 75
    test ! -f "$result_dir/maps/map_ready.json" || return 0
    sleep 1
  done
  return 75
}

copy_map_archive() {
  local target="$1" destination="$2" archive="$result_dir/maps/d0_fixed5_static_maps.tar.gz"
  local expected_archive expected_manifest
  mapfile -t values < <(python3 - "$result_dir/maps/map_ready.json" <<'PY'
import json,sys
value=json.load(open(sys.argv[1],encoding="utf-8"))
assert value.get("status")=="PASS"
print(value["archive_sha256"]); print(value["manifest_sha256"])
PY
  )
  expected_archive="${values[0]}"; expected_manifest="${values[1]}"
  test "$(sha256sum "$archive" | cut -d' ' -f1)" = "$expected_archive"
  remote "$target" "install -d -m 700 '$destination/inputs/d0_fixed5_static_maps'"
  gzip -dc "$archive" | ssh "${ssh_options[@]}" "$target" \
    "tar -C '$destination/inputs/d0_fixed5_static_maps' -xf -"
  remote "$target" \
    "test \"\$(sha256sum '$destination/inputs/d0_fixed5_static_maps/manifest.json' | cut -d' ' -f1)\" = '$expected_manifest'"
}

stop_remote_prepare_supervisor() {
  local term_timeout="${INTERNNAV_T5_FAST_PREP_REMOTE_TERM_TIMEOUT_SEC:-60}"
  local kill_timeout="${INTERNNAV_T5_FAST_PREP_REMOTE_KILL_TIMEOUT_SEC:-15}"
  local term_deadline kill_deadline final_output="" final_rc=75
  [[ "$term_timeout" =~ ^[1-9][0-9]*$ ]] || return 75
  [[ "$kill_timeout" =~ ^[1-9][0-9]*$ ]] || return 75
  (( term_timeout + kill_timeout <= 120 )) || return 75

  if test "$remote_operation_launch_attempted" != true; then
    python3 - "$audit_dir/supervisor_cleanup.json" <<'PY'
import json,time,sys
from pathlib import Path
checks={"operation_not_launched":True,"supervisor_absent":True,
        "identity_bound_if_launched":True,"term_then_kill_bounded":True}
Path(sys.argv[1]).write_text(json.dumps({"schema_version":1,"status":"PASS",
 "launch_attempted":False,"checks":checks,"recorded_unix":time.time()},
 indent=2,sort_keys=True)+"\n",encoding="utf-8")
PY
    return 0
  fi

  remote "$target" "test -f '$supervisor_ledger'" >/dev/null 2>&1 || {
    python3 - "$audit_dir/supervisor_cleanup.json" <<'PY'
import json,time,sys
from pathlib import Path
checks={"ledger_present":False,"supervisor_absent":False,
        "identity_bound_if_launched":False,"term_then_kill_bounded":False}
Path(sys.argv[1]).write_text(json.dumps({"schema_version":1,"status":"FAIL",
 "launch_attempted":True,"checks":checks,"recorded_unix":time.time()},
 indent=2,sort_keys=True)+"\n",encoding="utf-8")
PY
    return 75
  }
  remote "$target" "cat '$supervisor_ledger'" \
    >"$audit_dir/supervisor_ledger.json" 2>/dev/null || return 75
  remote_prepare_supervisor_action "$target" "$supervisor_ledger" \
    "$destination" TERM >/dev/null 2>&1 || true
  term_deadline=$((SECONDS + term_timeout))
  while (( SECONDS < term_deadline )); do
    remote_prepare_supervisor_absent "$target" "$supervisor_ledger" \
      "$destination" >/dev/null 2>&1 && break
    sleep 1
  done
  if ! remote_prepare_supervisor_absent "$target" "$supervisor_ledger" \
      "$destination" >/dev/null 2>&1; then
    remote_prepare_supervisor_action "$target" "$supervisor_ledger" \
      "$destination" KILL >/dev/null 2>&1 || true
  fi
  kill_deadline=$((SECONDS + kill_timeout))
  while (( SECONDS < kill_deadline )); do
    remote_prepare_supervisor_absent "$target" "$supervisor_ledger" \
      "$destination" >/dev/null 2>&1 && break
    sleep 1
  done
  final_output="$(remote_prepare_supervisor_action "$target" "$supervisor_ledger" \
    "$destination" AUDIT)" && final_rc=0 || final_rc=$?
  printf '%s\n' "$final_output" >"$audit_dir/supervisor_final_audit.json"
  python3 - "$audit_dir/supervisor_cleanup.json" \
    "$audit_dir/supervisor_ledger.json" "$audit_dir/supervisor_final_audit.json" \
    "$final_rc" "$term_timeout" "$kill_timeout" <<'PY'
import json,sys,time
from pathlib import Path
output,ledger_path,final_path=map(Path,sys.argv[1:4])
def load(path):
    try:return json.loads(path.read_text(encoding="utf-8"))
    except (OSError,UnicodeError,json.JSONDecodeError):return None
ledger,final=load(ledger_path),load(final_path)
identity_keys=("pid","pgid","sid","starttime","argv_sha256","run_root")
checks={
 "ledger_present":isinstance(ledger,dict),
 "identity_bound_if_launched":isinstance(ledger,dict)
    and all(ledger.get(key) not in (None,"") for key in identity_keys),
 "final_audit_exit_zero":int(sys.argv[4])==0,
 "supervisor_absent":isinstance(final,dict) and final.get("absent") is True,
 "term_then_kill_bounded":int(sys.argv[5])+int(sys.argv[6])<=120,
}
payload={"schema_version":1,"status":"PASS" if all(checks.values()) else "FAIL",
 "launch_attempted":True,"ledger":ledger,"final_audit":final,
 "term_timeout_sec":int(sys.argv[5]),"kill_timeout_sec":int(sys.argv[6]),
 "checks":checks,"recorded_unix":time.time()}
output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
raise SystemExit(0 if payload["status"]=="PASS" else 75)
PY
}

inside_finish() {
  local incoming=$? supervisor_rc=75 cleanup_rc=75 clear_rc=75 final_exit
  trap - EXIT INT TERM HUP
  set +e
  stop_remote_prepare_supervisor
  supervisor_rc=$?
  remote "$target" "rm -f -- '$stage_archive'" >/dev/null 2>&1 || true
  run_residual_audit "$target" "$role" "$scope_roots" "$ownership_dir" \
    "$audit_dir/residual_cleanup.json"
  residual_rc=$?
  global_path="$audit_dir/global_compute_final.json"
  if test "$role" = x86_a; then
    cp "$audit_dir/residual_cleanup.json" "$global_path"
    global_rc=$?
  else
    t5_remote_compute_absent "$target" "$global_path"
    global_rc=$?
  fi
  write_cleanup_receipt "$role" "$audit_dir/supervisor_cleanup.json" \
    "$audit_dir/residual_cleanup.json" \
    "$global_path" "$audit_dir/cleanup_receipt.json"
  cleanup_rc=$?
  if test "$quarantine_armed" = true && test "$cleanup_rc" = 0; then
    t5_quarantine_owned_clear "$target" "$quarantine_file" "$quarantine_role" \
      "$tag" "$scope_roots" "$audit_dir/cleanup_receipt.json" \
      "$audit_dir/quarantine_clear.json"
    clear_rc=$?
  fi
  python3 - "$task_dir/task_summary.json" "$role" "$code_sha" "$archive_sha256" \
    "$operation_complete" "$incoming" "$supervisor_rc" "$residual_rc" "$global_rc" \
    "$cleanup_rc" "$clear_rc" "$scope_roots" \
    "$task_dir/deployment_binding.json" <<'PY'
import json,sys,time
from pathlib import Path
output=Path(sys.argv[1])
try: binding=json.loads(Path(sys.argv[13]).read_text(encoding="utf-8"))
except (OSError,UnicodeError,json.JSONDecodeError): binding=None
checks={"operation_completed":sys.argv[5]=="true" and int(sys.argv[6])==0,
        "supervisor_cleanup_pass":int(sys.argv[7])==0,
        "residual_audit_pass":int(sys.argv[8])==0,
        "global_audit_pass":int(sys.argv[9])==0,
        "cleanup_receipt_pass":int(sys.argv[10])==0,
        "quarantine_clear_pass":int(sys.argv[11])==0,
        "deployment_binding_pass":isinstance(binding,dict) and binding.get("status")=="PASS"
            and all(binding.get("checks",{}).values())}
payload={"schema_version":1,"status":"PASS" if all(checks.values()) else "FAIL",
         "role":sys.argv[2],"code_ref_sha":sys.argv[3],
         "deployment_archive_sha256":sys.argv[4],
         "deployment_roots":sys.argv[12].split("|"),"checks":checks,
         "recorded_unix":time.time()}
output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
PY
  set -e
  final_exit="$incoming"
  test "$supervisor_rc" = 0 || final_exit=75
  test "$cleanup_rc" = 0 && test "$clear_rc" = 0 || final_exit=75
  test "$operation_complete" = true || test "$final_exit" != 0 || final_exit=1
  exit "$final_exit"
}

inside_main() {
  [[ $# -eq 6 ]] || usage
  role="$2"; code_sha="$3"; run_id="$4"; result_relative="$5"
  validate_inputs "$code_sha" "$run_id" "$result_relative"
  case "$role" in dgx_a|dgx_b|x86|x86_a) ;; *) usage ;; esac
  ensure_exact_clean_ref "$code_sha"
  deployment_values "$code_sha" "$run_id"
  result_dir="$root/$result_relative"
  test "$result_dir" = "$6"
  archive_sha256="$(tr -d '\r\n' <"$result_dir/deployment_archive_sha256.txt")"
  [[ "$archive_sha256" =~ ^[0-9a-f]{64}$ ]]
  test "$(sha256sum "$result_dir/deployment.tar.gz" | cut -d' ' -f1)" = "$archive_sha256"
  read_frozen_values
  task_dir="$result_dir/remote/$role"
  audit_dir="$result_dir/audits/$role"
  mkdir -p "$task_dir" "$audit_dir"
  quarantine_armed=false
  operation_complete=false
  remote_operation_launch_attempted=false
  ownership_dir=-
  quarantine_file=/tmp/internnav_dgx.quarantine
  quarantine_role="$role"
  case "$role" in
    dgx_a)
      if test "$prepare_scope" = lane-a; then
        test "${INTERNNAV_T5_RESOURCE_LEASE_ACK:-}" = lane-a
      else
        test "${INTERNNAV_T5_RESOURCE_LEASE_ACK:-}" = dgx-a
      fi
      target="$dgx_a_target"; destination="$dgx_a_root"
      scope_roots="$dgx_a_root"; expected_user=railgun; expected_ip=10.100.100.128
      ;;
    dgx_b)
      test "${INTERNNAV_T5_RESOURCE_LEASE_ACK:-}" = dgx-b
      target="$dgx_b_target"; destination="$dgx_b_root"
      scope_roots="$dgx_b_root"; expected_user=rail; expected_ip=10.100.120.122
      ;;
    x86)
      test "${INTERNNAV_T5_RESOURCE_LEASE_ACK:-}" = isaac
      target="$x86_target"; destination="$x86_prepare_root"
      scope_roots="$x86_prepare_root|$x86_a_root|$x86_b_root"
      expected_user=song; expected_ip="$x86_ip"
      quarantine_file=/tmp/internnav_isaac.quarantine
      ownership_dir="$x86_prepare_root/results/fast_prepare/containers_owned"
      ;;
    x86_a)
      test "$prepare_scope" = lane-a
      test "${INTERNNAV_T5_RESOURCE_LEASE_ACK:-}" = lane-a
      target="$x86_target"; destination="$x86_prepare_root"
      scope_roots="$x86_prepare_root|$x86_a_root"
      expected_user=song; expected_ip="$x86_ip"
      quarantine_file=/tmp/internnav_isaac_gpu0.quarantine
      quarantine_role=x86_gpu0
      ownership_dir="$x86_prepare_root/results/fast_prepare/containers_owned"
      ;;
  esac
  stage_archive="/tmp/internnav-t5-fast-${tag}-${role}.tar.gz"
  supervisor_ledger="$destination/results/fast_prepare/supervisor.json"
  trap inside_finish EXIT
  trap 'exit 130' INT TERM HUP

  if test "$role" = x86; then
    remote "$target" \
      "set -euo pipefail; test \"\$(id -un)\" = '$expected_user'; ip -4 -o addr show | grep -Fq ' $expected_ip/'; command -v docker >/dev/null; command -v flock >/dev/null; test ! -e '$x86_prepare_root'; test ! -e '$x86_a_root'; test ! -e '$x86_b_root'; test ! -e '$stage_archive'; test -f '$dataset_root/val_unseen/val_unseen.json.gz'; test \"\$(sha256sum '$dataset_root/val_unseen/val_unseen.json.gz' | cut -d' ' -f1)\" = '$dataset_sha256'; for c in internnav_t5_isaac_a internnav_t5_isaac_b; do if docker inspect \"\$c\" >/dev/null 2>&1; then test \"\$(docker inspect -f '{{.State.Running}}' \"\$c\")\" = false; fi; done; flock -n /tmp/internnav_t5_isaac_shared_assets.lock true"
  elif test "$role" = x86_a; then
    remote "$target" \
      "set -euo pipefail; test \"\$(id -un)\" = '$expected_user'; ip -4 -o addr show | grep -Fq ' $expected_ip/'; command -v docker >/dev/null; command -v flock >/dev/null; test ! -e '$x86_prepare_root'; test ! -e '$x86_a_root'; test ! -e '$stage_archive'; test -f '$dataset_root/val_unseen/val_unseen.json.gz'; test \"\$(sha256sum '$dataset_root/val_unseen/val_unseen.json.gz' | cut -d' ' -f1)\" = '$dataset_sha256'; if docker inspect internnav_t5_isaac_a >/dev/null 2>&1; then test \"\$(docker inspect -f '{{.State.Running}}' internnav_t5_isaac_a)\" = false; fi; flock -n /tmp/internnav_t5_isaac_shared_assets.lock true"
  else
    remote "$target" \
      "set -euo pipefail; test \"\$(id -un)\" = '$expected_user'; ip -4 -o addr show | grep -Fq ' $expected_ip/'; command -v setsid >/dev/null; command -v nvidia-smi >/dev/null; test ! -e '$destination'; test ! -e '$stage_archive'; test -d \"\$HOME/internnav-t0/InternNav/checkpoints/InternVLA-N1-DualVLN\"; test -x \"\$HOME/internnav-t0/venv-model/bin/python\""
  fi

  t5_quarantine_arm "$target" "$quarantine_file" "$quarantine_role" "$tag" \
    "$scope_roots" "$audit_dir/quarantine_arm.json"
  quarantine_armed=true

  if test "$role" = x86_a; then
    run_residual_audit "$target" "$role" "$scope_roots" "$ownership_dir" \
      "$audit_dir/global_compute_predeploy.json"
  else
    t5_remote_compute_absent "$target" "$audit_dir/global_compute_predeploy.json"
  fi
  if [[ "$role" != x86 && "$role" != x86_a ]]; then
    deploy_archive "$target" "$stage_archive" "$destination"
    record_deployment_bindings "$target" "$task_dir/deployment_binding.json" \
      "$destination"
    remote_operation_launch_attempted=true
    remote "$target" \
      "exec setsid --wait bash -c \"\$(printf '%s' '$dgx_build_b64' | base64 -d)\" fast-dgx-build '$destination' '$model_revision' '$checkpoint_revision' '$model_weight_inventory_sha256' '$checkpoint_content_manifest_sha256' '$supervisor_ledger' '$role'" \
      >"$result_dir/logs/${role}_build_ssh.log" 2>&1
    collect_remote_directory "$target" "$destination/results/fast_prepare" "$task_dir/build"
    cp "$task_dir/build/build_summary.json" "$task_dir/build_summary.json"
    wait_for_map_ready
    copy_map_archive "$target" "$destination"
    python3 - "$task_dir/map_install.json" "$destination" \
      "$result_dir/maps/map_ready.json" <<'PY'
import json,sys,time
from pathlib import Path
ready=json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
payload={"schema_version":1,"status":"PASS","deployment_root":sys.argv[2],
         "manifest_sha256":ready["manifest_sha256"],"checks":{"manifest_verified":True},
         "recorded_unix":time.time()}
Path(sys.argv[1]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
  elif test "$role" = x86; then
    deploy_archive "$target" "$stage_archive" \
      "$x86_prepare_root" "$x86_a_root" "$x86_b_root"
    record_deployment_bindings "$target" "$task_dir/deployment_binding.json" \
      "$x86_prepare_root" "$x86_a_root" "$x86_b_root"
    remote_operation_launch_attempted=true
    remote "$target" \
      "exec setsid --wait bash -c \"\$(printf '%s' '$x86_prepare_b64' | base64 -d)\" fast-x86-prepare '$x86_prepare_root' '$dataset_root' '$dataset_sha256' '$x86_a_root' '$x86_b_root' '$supervisor_ledger' '$x86_ip' '$lane_a_cpuset' '$lane_b_cpuset'" \
      >"$result_dir/logs/x86_prepare_ssh.log" 2>&1
    collect_remote_directory "$target" "$x86_prepare_root/results/fast_prepare" "$task_dir/prepare"
    cp "$task_dir/prepare/dataset_audit.json" "$task_dir/dataset_audit.json"
    cp "$task_dir/prepare/container_cleanup.json" "$task_dir/container_cleanup.json"
    cp "$task_dir/prepare/x86_prepare_summary.json" "$task_dir/x86_prepare_summary.json"
    remote "$target" \
      "tar -C '$x86_prepare_root/inputs/d0_fixed5_static_maps' -czf - ." \
      >"$result_dir/maps/d0_fixed5_static_maps.tar.gz"
    map_archive_sha256="$(sha256sum "$result_dir/maps/d0_fixed5_static_maps.tar.gz" | cut -d' ' -f1)"
    map_manifest_sha256="$(remote "$target" "sha256sum '$x86_prepare_root/inputs/d0_fixed5_static_maps/manifest.json' | cut -d' ' -f1")"
    copy_map_x86() {
      local destination="$1"
      remote "$target" "install -d -m 700 '$destination/inputs/d0_fixed5_static_maps'"
      gzip -dc "$result_dir/maps/d0_fixed5_static_maps.tar.gz" | \
        ssh "${ssh_options[@]}" "$target" \
          "tar -C '$destination/inputs/d0_fixed5_static_maps' -xf -"
      remote "$target" \
        "test \"\$(sha256sum '$destination/inputs/d0_fixed5_static_maps/manifest.json' | cut -d' ' -f1)\" = '$map_manifest_sha256'"
    }
    copy_map_x86 "$x86_a_root"
    copy_map_x86 "$x86_b_root"
    python3 - "$result_dir/maps/map_ready.json" "$map_archive_sha256" \
      "$map_manifest_sha256" <<'PY'
import json,os,sys,time
from pathlib import Path
output=Path(sys.argv[1])
payload={"schema_version":1,"status":"PASS","archive_sha256":sys.argv[2],
         "manifest_sha256":sys.argv[3],"x86_lane_roots_verified":True,
         "recorded_unix":time.time()}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
  else
    deploy_archive "$target" "$stage_archive" \
      "$x86_prepare_root" "$x86_a_root"
    record_deployment_bindings "$target" "$task_dir/deployment_binding.json" \
      "$x86_prepare_root" "$x86_a_root"
    remote_operation_launch_attempted=true
    remote "$target" \
      "exec setsid --wait bash -c \"\$(printf '%s' '$x86_prepare_b64' | base64 -d)\" fast-x86-prepare '$x86_prepare_root' '$dataset_root' '$dataset_sha256' '$x86_a_root' '-' '$supervisor_ledger' '$x86_ip' '$lane_a_cpuset' '-' lane-a" \
      >"$result_dir/logs/x86_a_prepare_ssh.log" 2>&1
    collect_remote_directory "$target" "$x86_prepare_root/results/fast_prepare" "$task_dir/prepare"
    cp "$task_dir/prepare/dataset_audit.json" "$task_dir/dataset_audit.json"
    cp "$task_dir/prepare/container_cleanup.json" "$task_dir/container_cleanup.json"
    cp "$task_dir/prepare/x86_prepare_summary.json" "$task_dir/x86_prepare_summary.json"
    remote "$target" \
      "tar -C '$x86_prepare_root/inputs/d0_fixed5_static_maps' -czf - ." \
      >"$result_dir/maps/d0_fixed5_static_maps.tar.gz"
    map_archive_sha256="$(sha256sum "$result_dir/maps/d0_fixed5_static_maps.tar.gz" | cut -d' ' -f1)"
    map_manifest_sha256="$(remote "$target" "sha256sum '$x86_prepare_root/inputs/d0_fixed5_static_maps/manifest.json' | cut -d' ' -f1")"
    copy_map_x86() {
      local destination="$1"
      remote "$target" "install -d -m 700 '$destination/inputs/d0_fixed5_static_maps'"
      gzip -dc "$result_dir/maps/d0_fixed5_static_maps.tar.gz" | \
        ssh "${ssh_options[@]}" "$target" \
          "tar -C '$destination/inputs/d0_fixed5_static_maps' -xf -"
      remote "$target" \
        "test \"\$(sha256sum '$destination/inputs/d0_fixed5_static_maps/manifest.json' | cut -d' ' -f1)\" = '$map_manifest_sha256'"
    }
    copy_map_x86 "$x86_a_root"
    python3 - "$result_dir/maps/map_ready.json" "$map_archive_sha256" \
      "$map_manifest_sha256" <<'PY'
import json,os,sys,time
from pathlib import Path
output=Path(sys.argv[1])
payload={"schema_version":1,"status":"PASS","archive_sha256":sys.argv[2],
         "manifest_sha256":sys.argv[3],"x86_lane_roots_verified":True,
         "prepared_lanes":["a"],"recorded_unix":time.time()}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
  fi
  operation_complete=true
}

write_lease_summary() {
  if test "$prepare_scope" = lane-a; then
    python3 - "$result_dir/lease_release_summary.json" \
      "$result_dir/lease/lane_a_pair/lease_metadata.txt" "$lane_a_pair_rc" <<'PY'
import json,sys,time
from pathlib import Path
output,metadata=map(Path,sys.argv[1:3]); command_rc=int(sys.argv[3])
text=metadata.read_text(encoding="utf-8") if metadata.is_file() else ""
cleanup_path=metadata.parent/"lease_cleanup_receipt.json"
try: cleanup=json.loads(cleanup_path.read_text(encoding="utf-8"))
except (OSError,UnicodeError,json.JSONDecodeError): cleanup=None
checks={
 "command_exit_zero":command_rc==0,
 "exact_lane_a_resources":"dgx_a" in text and "isaac_gpu0" in text
    and "dgx_b" not in text and "isaac_gpu1" not in text and "resource=isaac " not in text,
 "wrapped_command_released":"state=RELEASED" in text,
 "wrapped_command_exit_recorded":f"command_exit={command_rc}" in text,
 "cleanup_pass":isinstance(cleanup,dict) and cleanup.get("status")=="PASS"
    and cleanup.get("requested_resources")==["dgx_a","isaac_gpu0"]
    and cleanup.get("wrapped_process_group_absent_before_lock_release") is True,
}
payload={"schema_version":1,"status":"PASS" if all(checks.values()) else "FAIL",
 "tasks":{"lane_a_pair":{"status":"PASS" if all(checks.values()) else "FAIL",
 "resource_profile":"lane-a","command_exit":command_rc,"checks":checks,
 "cleanup_receipt":cleanup}},"checks":{"lane_a_pair_lease_pass":all(checks.values())},
 "recorded_unix":time.time()}
output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
PY
    return
  fi
  python3 - "$result_dir/lease_release_summary.json" "$result_dir/lease" \
    "$dgx_a_rc" "$dgx_b_rc" "$x86_rc" <<'PY'
import json,sys,time
from pathlib import Path
output,root=Path(sys.argv[1]),Path(sys.argv[2])
command_rcs={"dgx_a":int(sys.argv[3]),"dgx_b":int(sys.argv[4]),"x86":int(sys.argv[5])}
expected={"dgx_a":["dgx_a"],"dgx_b":["dgx_b"],
          "x86":["isaac_gpu0","isaac_gpu1","isaac"]}
tasks={}; checks={}
for role in ("dgx_a","dgx_b","x86"):
    directory=root/role
    metadata=directory/"lease_metadata.txt"
    cleanup_path=directory/"lease_cleanup_receipt.json"
    text=metadata.read_text(encoding="utf-8") if metadata.is_file() else ""
    try: cleanup=json.loads(cleanup_path.read_text(encoding="utf-8"))
    except (OSError,UnicodeError,json.JSONDecodeError): cleanup=None
    task_checks={
        "command_exit_zero":command_rcs[role]==0,
        "wrapped_command_released":"state=RELEASED" in text,
        "wrapped_command_exit_recorded":f"command_exit={command_rcs[role]}" in text,
        "cleanup_pass":isinstance(cleanup,dict) and cleanup.get("status")=="PASS"
            and cleanup.get("requested_resources")==expected[role]
            and cleanup.get("wrapped_process_group_absent_before_lock_release") is True,
    }
    tasks[role]={"status":"PASS" if all(task_checks.values()) else "FAIL",
                 "resource_profile":{"dgx_a":"dgx-a","dgx_b":"dgx-b","x86":"isaac"}[role],
                 "command_exit":command_rcs[role],"checks":task_checks,
                 "cleanup_receipt":cleanup}
    checks[f"{role}_lease_pass"]=tasks[role]["status"]=="PASS"
payload={"schema_version":1,"status":"PASS" if all(checks.values()) else "FAIL",
         "tasks":tasks,"checks":checks,"recorded_unix":time.time()}
output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
PY
}

write_final_receipts() {
  if test "$prepare_scope" = lane-a; then
    python3 - "$result_dir" "$code_sha" "$run_id" "$archive_sha256" <<'PY'
import hashlib,json,sys,time
from pathlib import Path
result=Path(sys.argv[1]); code_sha,run_id,archive_sha=sys.argv[2:5]
def load(relative):
    try:return json.loads((result/relative).read_text(encoding="utf-8"))
    except (OSError,UnicodeError,json.JSONDecodeError):return None
inputs=load("fast_prepare_input.json"); input_path=result/"fast_prepare_input.json"
lease=load("lease_release_summary.json")
tasks={role:load(f"remote/{role}/task_summary.json") for role in ("dgx_a","x86_a")}
bindings={role:load(f"remote/{role}/deployment_binding.json") for role in ("dgx_a","x86_a")}
build=load("remote/dgx_a/build_summary.json")
x86=load("remote/x86_a/x86_prepare_summary.json")
dataset=load("remote/x86_a/dataset_audit.json"); dataset_path=result/"remote/x86_a/dataset_audit.json"
dataset_sha=hashlib.sha256(dataset_path.read_bytes()).hexdigest() if dataset_path.is_file() else None
containers=load("remote/x86_a/container_cleanup.json"); maps=load("maps/map_ready.json")
cleanups={role:load(f"audits/{role}/cleanup_receipt.json") for role in ("dgx_a","x86_a")}
clears={role:load(f"audits/{role}/quarantine_clear.json") for role in ("dgx_a","x86_a")}
roots=inputs.get("deployment_roots",{}) if isinstance(inputs,dict) else {}
checks={
 "exact_ref_input":isinstance(inputs,dict) and inputs.get("code_ref_sha")==code_sha
    and inputs.get("deployment_archive_sha256")==archive_sha and inputs.get("prepared_lanes")==["a"],
 "two_lane_a_tasks_pass":all(isinstance(v,dict) and v.get("status")=="PASS"
    and all(v.get("checks",{}).values()) for v in tasks.values()),
 "lane_a_bindings_exact":all(isinstance(v,dict) and v.get("status")=="PASS"
    and all(v.get("checks",{}).values()) for v in bindings.values()),
 "dgx_a_build_pass":isinstance(build,dict) and build.get("status")=="PASS" and all(build.get("checks",{}).values()),
 "x86_lane_a_prepare_pass":isinstance(x86,dict) and x86.get("status")=="PASS" and all(x86.get("checks",{}).values()),
 "dataset_audit_pass":isinstance(dataset,dict) and dataset.get("status")=="PASS" and all(dataset.get("checks",{}).values()),
 "dataset_audit_frozen_binding":isinstance(dataset,dict) and dataset.get("dataset_sha256")==inputs.get("dataset_file_sha256")
    and dataset.get("episode_count")==inputs.get("episode_count")==5 and dataset.get("episode_keys")==inputs.get("episode_keys")
    and isinstance(dataset_sha,str),
 "lane_a_container_stopped":isinstance(containers,dict) and containers.get("status")=="PASS"
    and containers.get("prepared_lanes")==["a"] and set((containers.get("lanes") or {}))=={"a"}
    and all(containers.get("checks",{}).values()),
 "static_maps_lane_a_roots":isinstance(maps,dict) and maps.get("status")=="PASS"
    and maps.get("prepared_lanes")==["a"] and maps.get("x86_lane_roots_verified") is True,
 "lane_a_cleanup_pass":all(isinstance(v,dict) and v.get("status")=="PASS" and all(v.get("checks",{}).values()) for v in cleanups.values()),
 "lane_a_quarantines_cleared":all(isinstance(v,dict) and v.get("status")=="PASS" for v in clears.values()),
 "lane_a_lease_released":isinstance(lease,dict) and lease.get("status")=="PASS" and all(lease.get("checks",{}).values()),
}
input_sha=hashlib.sha256(input_path.read_bytes()).hexdigest()
summary={"schema_version":1,"status":"PASS" if all(checks.values()) else "FAIL",
 "stage":"d0_0_online_prepare","grant_id":run_id,"authorization_mode":"FAST_EXACT_REF_NO_BOARD",
 "board_grant_used":False,"authorization_ref_sha":code_sha,"code_ref_sha":code_sha,
 "prepare_scope":"lane-a","prepared_lanes":["a"],"deployment_archive_sha256":archive_sha,
 "deployment_roots":roots,"fast_prepare_input_sha256":input_sha,
 "dataset_file_sha256":inputs.get("dataset_file_sha256"),"dataset_audit_sha256":dataset_sha,
 "episode_count":inputs.get("episode_count"),"episode_keys":inputs.get("episode_keys"),
 "static_map_archive_sha256":maps.get("archive_sha256") if isinstance(maps,dict) else None,
 "static_map_manifest_sha256":maps.get("manifest_sha256") if isinstance(maps,dict) else None,
 "checks":checks,"containers":containers.get("lanes") if isinstance(containers,dict) else None,
 "locks":{"profiles":["lane-a"],"all_lanes_used":False,"release_evidence":"lease_release_summary.json"},
 "execution":{"lane_a_pair_concurrent":True,"internvla_model_loaded":False,
 "isaac_sim_started":False,"episode_or_evaluator_started":False},"recorded_unix":time.time()}
summary_path=result/"d0_prepare_summary.json"
summary_path.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8")
final_checks={"preparation_pass":summary["status"]=="PASS" and all(checks.values()),
 "lease_release_pass":checks["lane_a_lease_released"],"cleanup_pass":checks["lane_a_cleanup_pass"] and checks["lane_a_quarantines_cleared"],
 "exact_code_and_archive":checks["exact_ref_input"],"dataset_audit_frozen_binding":checks["dataset_audit_frozen_binding"]}
final={"schema_version":1,"status":"PASS" if all(final_checks.values()) else "FAIL",
 "stage":"d0_0_online_prepare","authorization_mode":"FAST_EXACT_REF_NO_BOARD","board_grant_used":False,
 "authorization_ref_sha":code_sha,"code_ref_sha":code_sha,"prepare_scope":"lane-a","prepared_lanes":["a"],
 "deployment_archive_sha256":archive_sha,"deployment_roots":roots,"fast_prepare_input_sha256":input_sha,
 "dataset_file_sha256":summary.get("dataset_file_sha256"),"dataset_audit_sha256":dataset_sha,
 "episode_count":summary.get("episode_count"),"episode_keys":summary.get("episode_keys"),
 "preparation_summary":"d0_prepare_summary.json","preparation_summary_sha256":hashlib.sha256(summary_path.read_bytes()).hexdigest(),
 "lease_release_summary":"lease_release_summary.json","lease_release_summary_sha256":hashlib.sha256((result/"lease_release_summary.json").read_bytes()).hexdigest(),
 "checks":final_checks,"recorded_unix":time.time()}
(result/"d0_prepare_final_summary.json").write_text(json.dumps(final,indent=2,sort_keys=True)+"\n",encoding="utf-8")
raise SystemExit(0 if final["status"]=="PASS" else 1)
PY
    return
  fi
  python3 - "$result_dir" "$code_sha" "$run_id" "$archive_sha256" <<'PY'
import hashlib,json,sys,time
from pathlib import Path
result=Path(sys.argv[1]); code_sha,run_id,archive_sha=sys.argv[2:5]
def load(relative):
    path=result/relative
    try:return json.loads(path.read_text(encoding="utf-8"))
    except (OSError,UnicodeError,json.JSONDecodeError):return None
inputs=load("fast_prepare_input.json")
input_path=result/"fast_prepare_input.json"
fast_prepare_input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest() \
    if input_path.is_file() else None
lease=load("lease_release_summary.json")
tasks={role:load(f"remote/{role}/task_summary.json") for role in ("dgx_a","dgx_b","x86")}
bindings={role:load(f"remote/{role}/deployment_binding.json") for role in ("dgx_a","dgx_b","x86")}
builds={role:load(f"remote/{role}/build_summary.json") for role in ("dgx_a","dgx_b")}
x86=load("remote/x86/x86_prepare_summary.json")
dataset=load("remote/x86/dataset_audit.json")
dataset_path=result/"remote/x86/dataset_audit.json"
dataset_audit_sha256=hashlib.sha256(dataset_path.read_bytes()).hexdigest() \
    if dataset_path.is_file() else None
containers=load("remote/x86/container_cleanup.json")
maps=load("maps/map_ready.json")
cleanups={role:load(f"audits/{role}/cleanup_receipt.json") for role in ("dgx_a","dgx_b","x86")}
clears={role:load(f"audits/{role}/quarantine_clear.json") for role in ("dgx_a","dgx_b","x86")}
roots=inputs.get("deployment_roots",{}) if isinstance(inputs,dict) else {}
checks={
    "exact_ref_input":isinstance(inputs,dict) and inputs.get("code_ref_sha")==code_sha
        and inputs.get("deployment_archive_sha256")==archive_sha,
    "three_parallel_tasks_pass":all(isinstance(value,dict) and value.get("status")=="PASS"
        and all(value.get("checks",{}).values()) for value in tasks.values()),
    "all_remote_deployment_bindings_exact":all(isinstance(value,dict)
        and value.get("status")=="PASS" and all(value.get("checks",{}).values())
        for value in bindings.values()),
    "both_dgx_builds_pass":all(isinstance(value,dict) and value.get("status")=="PASS"
        and all(value.get("checks",{}).values()) for value in builds.values()),
    "x86_serial_prepare_pass":isinstance(x86,dict) and x86.get("status")=="PASS"
        and all(x86.get("checks",{}).values()),
    "dataset_audit_pass":isinstance(dataset,dict) and dataset.get("status")=="PASS"
        and all(dataset.get("checks",{}).values()),
    "dataset_audit_frozen_binding":isinstance(inputs,dict) and isinstance(dataset,dict)
        and dataset.get("dataset_sha256")==inputs.get("dataset_file_sha256")
        and dataset.get("episode_count")==inputs.get("episode_count")==5
        and dataset.get("episode_keys")==inputs.get("episode_keys")
        and isinstance(inputs.get("episode_keys"),list)
        and len(set(inputs["episode_keys"]))==5
        and isinstance(dataset_audit_sha256,str),
    "worker_containers_stopped":isinstance(containers,dict) and containers.get("status")=="PASS"
        and all(containers.get("checks",{}).values()),
    "static_maps_all_roots":isinstance(maps,dict) and maps.get("status")=="PASS"
        and maps.get("x86_lane_roots_verified") is True,
    "all_cleanup_receipts_pass":all(isinstance(value,dict) and value.get("status")=="PASS"
        and all(value.get("checks",{}).values()) for value in cleanups.values()),
    "all_owned_quarantines_cleared":all(isinstance(value,dict) and value.get("status")=="PASS"
        for value in clears.values()),
    "all_leases_released":isinstance(lease,dict) and lease.get("status")=="PASS"
        and all(lease.get("checks",{}).values()),
}
summary={
    "schema_version":1,"status":"PASS" if all(checks.values()) else "FAIL",
    "stage":"d0_0_online_prepare","grant_id":run_id,
    "authorization_mode":"FAST_EXACT_REF_NO_BOARD","board_grant_used":False,
    "authorization_ref_sha":code_sha,"code_ref_sha":code_sha,
    "golden_bundle_id":inputs.get("golden_bundle_id") if isinstance(inputs,dict) else None,
    "golden_bundle_canonical_sha256":inputs.get("golden_bundle_canonical_sha256") if isinstance(inputs,dict) else None,
    "run_manifest_canonical_sha256":inputs.get("run_manifest_canonical_sha256") if isinstance(inputs,dict) else None,
    "deployment_archive_sha256":archive_sha,"deployment_roots":roots,
    "fast_prepare_input_sha256":fast_prepare_input_sha256,
    "dataset_file_sha256":inputs.get("dataset_file_sha256") if isinstance(inputs,dict) else None,
    "dataset_audit_sha256":dataset_audit_sha256,
    "episode_count":inputs.get("episode_count") if isinstance(inputs,dict) else None,
    "episode_keys":inputs.get("episode_keys") if isinstance(inputs,dict) else None,
    "static_map_archive_sha256":maps.get("archive_sha256") if isinstance(maps,dict) else None,
    "static_map_manifest_sha256":maps.get("manifest_sha256") if isinstance(maps,dict) else None,
    "checks":checks,"containers":containers.get("lanes") if isinstance(containers,dict) else None,
    "locks":{"profiles":["dgx-a","dgx-b","isaac"],"all_lanes_used":False,
             "release_evidence":"lease_release_summary.json"},
    "execution":{"dgx_ros_builds_concurrent":True,
                 "x86_dataset_maps_and_container_prepare_serial":True,
                 "three_physical_lease_tasks_concurrent":True,
                 "internvla_model_loaded":False,"isaac_sim_started":False,
                 "episode_or_evaluator_started":False},
    "recorded_unix":time.time(),
}
summary_path=result/"d0_prepare_summary.json"
summary_path.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8")
final_checks={
    "preparation_pass":summary["status"]=="PASS" and all(summary["checks"].values()),
    "lease_release_pass":isinstance(lease,dict) and lease.get("status")=="PASS",
    "cleanup_pass":checks["all_cleanup_receipts_pass"] and checks["all_owned_quarantines_cleared"],
    "exact_code_and_archive":checks["exact_ref_input"],
    "dataset_audit_frozen_binding":checks["dataset_audit_frozen_binding"],
}
final={"schema_version":1,"status":"PASS" if all(final_checks.values()) else "FAIL",
       "stage":"d0_0_online_prepare","authorization_mode":"FAST_EXACT_REF_NO_BOARD",
       "board_grant_used":False,"authorization_ref_sha":code_sha,"code_ref_sha":code_sha,
       "deployment_archive_sha256":archive_sha,"deployment_roots":roots,
       "fast_prepare_input_sha256":summary.get("fast_prepare_input_sha256"),
       "dataset_file_sha256":summary.get("dataset_file_sha256"),
       "dataset_audit_sha256":summary.get("dataset_audit_sha256"),
       "episode_count":summary.get("episode_count"),
       "episode_keys":summary.get("episode_keys"),
       "preparation_summary":"d0_prepare_summary.json",
       "preparation_summary_sha256":hashlib.sha256(summary_path.read_bytes()).hexdigest(),
       "lease_release_summary":"lease_release_summary.json",
       "lease_release_summary_sha256":hashlib.sha256((result/"lease_release_summary.json").read_bytes()).hexdigest(),
       "checks":final_checks,"recorded_unix":time.time()}
(result/"d0_prepare_final_summary.json").write_text(
    json.dumps(final,indent=2,sort_keys=True)+"\n",encoding="utf-8")
raise SystemExit(0 if final["status"]=="PASS" else 1)
PY
}

lane_a_pair_main() {
  [[ $# -eq 5 ]] || usage
  local code_sha="$2" run_id="$3" result_relative="$4" expected_result_dir="$5"
  test "$prepare_scope" = lane-a
  test "${INTERNNAV_T5_RESOURCE_LEASE_ACK:-}" = lane-a
  validate_inputs "$code_sha" "$run_id" "$result_relative"
  ensure_exact_clean_ref "$code_sha"
  result_dir="$root/$result_relative"
  test "$result_dir" = "$expected_result_dir"
  set +e
  bash "$root/coordination/run_t5_fast_prepare_online.sh" --inside x86_a \
    "$code_sha" "$run_id" "$result_relative" "$result_dir" \
    >"$result_dir/logs/x86_a_task.log" 2>&1 &
  local x86_a_pid=$!
  bash "$root/coordination/run_t5_fast_prepare_online.sh" --inside dgx_a \
    "$code_sha" "$run_id" "$result_relative" "$result_dir" \
    >"$result_dir/logs/dgx_a_task.log" 2>&1 &
  local dgx_a_pid=$!
  wait "$x86_a_pid"; local x86_a_rc=$?
  test "$x86_a_rc" = 0 || : >"$result_dir/maps/x86_failed"
  wait "$dgx_a_pid"; local dgx_a_rc=$?
  set -e
  printf '%s\n' "$x86_a_rc" >"$result_dir/audits/x86_a_command.rc"
  printf '%s\n' "$dgx_a_rc" >"$result_dir/audits/dgx_a_command.rc"
  test "$x86_a_rc" = 0 && test "$dgx_a_rc" = 0
}

lease_task_main() {
  [[ $# -eq 7 ]] || usage
  local role="$2" profile="$3" code_sha="$4" run_id="$5"
  local result_relative="$6" expected_result_dir="$7"
  validate_inputs "$code_sha" "$run_id" "$result_relative"
  case "$role:$profile" in
    dgx_a:dgx-a|dgx_b:dgx-b|x86:isaac|lane_a_pair:lane-a) ;;
    *) usage ;;
  esac
  ensure_exact_clean_ref "$code_sha"
  result_dir="$root/$result_relative"
  test "$result_dir" = "$expected_result_dir"
  test -d "$result_dir/audits"
  wrapper_ledger="$result_dir/audits/${role}_lease_wrapper_supervisor.json"
  pgid="$(ps -o pgid= -p "$$" | tr -d ' ')"
  sid="$(ps -o sid= -p "$$" | tr -d ' ')"
  starttime="$(awk '{print $22}' "/proc/$$/stat")"
  argv_sha256="$(sha256sum "/proc/$$/cmdline" | cut -d' ' -f1)"
  python3 - "$wrapper_ledger" "$result_dir" "$$" "$pgid" "$sid" \
    "$starttime" "$argv_sha256" "$role" "$run_id" "$code_sha" <<'PY'
import json,os,sys,time
from pathlib import Path
output=Path(sys.argv[1])
payload={"schema_version":2,"run_root":sys.argv[2],"pid":int(sys.argv[3]),
 "pgid":int(sys.argv[4]),"sid":int(sys.argv[5]),"starttime":int(sys.argv[6]),
 "argv_sha256":sys.argv[7],"role":sys.argv[8],"run_id":sys.argv[9],
 "code_ref_sha":sys.argv[10],"started_unix":time.time()}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
  test "$pgid" = "$sid"
  set +e
  if test "$role" = lane_a_pair; then
    test "$prepare_scope" = lane-a
    env ISAAC_HOST="$x86_ip" bash "$root/scripts/with_resource_lease.sh" "$profile" \
      --owner codex-00 --task "t5-fast-prepare:$role:$run_id:$code_sha" \
      --log-dir "$result_dir/lease/$role" --acquire-timeout 30 \
      --cleanup-timeout 60 --kill-wait-timeout 30 -- \
      env INTERNNAV_T5_RESOURCE_LEASE_ACK="$profile" \
        bash "$root/coordination/run_t5_fast_prepare_online.sh" \
          --lane-a-pair "$code_sha" "$run_id" "$result_relative" "$result_dir"
  else
    env ISAAC_HOST="$x86_ip" bash "$root/scripts/with_resource_lease.sh" "$profile" \
      --owner codex-00 --task "t5-fast-prepare:$role:$run_id:$code_sha" \
      --log-dir "$result_dir/lease/$role" --acquire-timeout 30 \
      --cleanup-timeout 60 --kill-wait-timeout 30 -- \
      env INTERNNAV_T5_RESOURCE_LEASE_ACK="$profile" \
        bash "$root/coordination/run_t5_fast_prepare_online.sh" \
          --inside "$role" "$code_sha" "$run_id" "$result_relative" "$result_dir"
  fi
  local rc=$?
  printf '%s\n' "$rc" >"$result_dir/audits/${role}_command.rc"
  if test "$role" = x86 && test "$rc" != 0; then
    : >"$result_dir/maps/x86_failed"
  fi
  exit "$rc"
}

outer_main() {
  [[ $# -eq 3 ]] || usage
  code_sha="$1"; run_id="$2"; result_relative="$3"
  validate_inputs "$code_sha" "$run_id" "$result_relative"
  ensure_exact_clean_ref "$code_sha"
  deployment_values "$code_sha" "$run_id"
  result_dir="$root/$result_relative"
  test ! -e "$result_dir"
  umask 077
  mkdir -p "$result_dir" "$result_dir/logs" "$result_dir/remote" \
    "$result_dir/audits" "$result_dir/lease" "$result_dir/maps"
  bash "$root/scripts/create_source_bundle.sh" "$code_sha" \
    "$result_dir/deployment.tar.gz" 1
  archive_sha256="$(sha256sum "$result_dir/deployment.tar.gz" | cut -d' ' -f1)"
  printf '%s\n' "$archive_sha256" >"$result_dir/deployment_archive_sha256.txt"
  python3 - "$result_dir/fast_prepare_input.json" "$code_sha" "$run_id" \
    "$archive_sha256" "$dgx_a_root" "$dgx_b_root" "$x86_prepare_root" \
    "$x86_a_root" "$x86_b_root" \
    "$root/configs/internnav_t5/golden_bundle_manifest.json" \
    "$root/configs/internnav_t5/d0_run_manifest.json" \
    "$root/configs/internnav_t5/checkpoint_content_manifest.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
output=Path(sys.argv[1]); code_sha,run_id,archive_sha=sys.argv[2:5]
roots={"dgx_a":sys.argv[5],"dgx_b":sys.argv[6],"x86_prepare":sys.argv[7],
       "x86_a":sys.argv[8],"x86_b":sys.argv[9]}
golden_path,d0_path,checkpoint_path=map(Path,sys.argv[10:13])
golden=json.loads(golden_path.read_text(encoding="utf-8"))
d0=json.loads(d0_path.read_text(encoding="utf-8"))
checkpoint=json.loads(checkpoint_path.read_text(encoding="utf-8"))
fixed=d0.get("fixed_input",{})
episode_keys=fixed.get("episode_keys")
if fixed.get("episode_count") != 5 or not isinstance(episode_keys,list) \
        or len(episode_keys) != 5 or len(set(episode_keys)) != 5:
    raise SystemExit("fast preparation requires the exact frozen five episode keys")
def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),
        ensure_ascii=True,allow_nan=False).encode("ascii")).hexdigest()
payload={"schema_version":1,"status":"PASS","authorization_mode":"FAST_EXACT_REF_NO_BOARD",
         "code_ref_sha":code_sha,"run_id":run_id,"deployment_archive_sha256":archive_sha,
         "deployment_roots":roots,"golden_bundle_id":golden["bundle_id"],
         "golden_bundle_canonical_sha256":canonical(golden),
         "run_manifest_canonical_sha256":canonical(d0),
         "model_revision":golden["model"]["internnav_revision"],
         "checkpoint_revision":golden["model"]["checkpoint_revision"],
         "model_weight_inventory_sha256":golden["model"]["weight_inventory_sha256"],
         "checkpoint_content_manifest_canonical_sha256":canonical(checkpoint),
         "dataset_root":fixed["dataset_remote_path"],
         "dataset_file_sha256":fixed["dataset_file_sha256"],
         "episode_count":fixed["episode_count"],"episode_keys":episode_keys}
output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
PY
  if test "$prepare_scope" = lane-a; then
    python3 - "$result_dir/fast_prepare_input.json" <<'PY'
import json,os,sys
from pathlib import Path
path=Path(sys.argv[1]); value=json.loads(path.read_text(encoding="utf-8"))
roots=value["deployment_roots"]
value["deployment_roots"]={key:roots[key] for key in ("dgx_a","x86_prepare","x86_a")}
value["prepare_scope"]="lane-a"; value["prepared_lanes"]=["a"]
temporary=path.with_name(f".{path.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="utf-8")
os.replace(temporary,path)
PY
  fi

  child_pids=()
  child_roles=()
  child_ledgers=()
  run_task() {
    local role="$1" profile="$2" log="$3" output_variable="$4"
    setsid --wait bash "$root/coordination/run_t5_fast_prepare_online.sh" \
      --lease-task "$role" "$profile" "$code_sha" "$run_id" \
      "$result_relative" "$result_dir" >"$log" 2>&1 &
    printf -v "$output_variable" '%s' "$!"
  }
  stop_children() {
    trap - INT TERM HUP
    local term_timeout="${INTERNNAV_T5_FAST_PREP_OUTER_TERM_TIMEOUT_SEC:-240}"
    local kill_timeout="${INTERNNAV_T5_FAST_PREP_OUTER_KILL_TIMEOUT_SEC:-30}"
    local deadline kill_deadline residual index pid ledger role
    set +e
    [[ "$term_timeout" =~ ^[1-9][0-9]*$ ]] || term_timeout=240
    [[ "$kill_timeout" =~ ^[1-9][0-9]*$ ]] || kill_timeout=30
    (( term_timeout + kill_timeout <= 300 )) || { term_timeout=240; kill_timeout=30; }

    # Ledgers are written before lease acquisition.  Give a just-forked task a
    # short bounded chance to publish its exact session identity.
    for index in "${!child_pids[@]}"; do
      pid="${child_pids[$index]}"; ledger="${child_ledgers[$index]}"
      ready_deadline=$((SECONDS + 5))
      while test ! -f "$ledger" && kill -0 "$pid" 2>/dev/null && \
          (( SECONDS < ready_deadline )); do sleep 0.1; done
      test ! -f "$ledger" || local_prepare_supervisor_action "$ledger" \
        "$result_dir" TERM >/dev/null 2>&1 || true
    done
    deadline=$((SECONDS + term_timeout))
    while (( SECONDS < deadline )); do
      residual=0
      for ledger in "${child_ledgers[@]}"; do
        test -f "$ledger" || { residual=$((residual + 1)); continue; }
        local_prepare_supervisor_absent "$ledger" "$result_dir" \
          >/dev/null 2>&1 || residual=$((residual + 1))
      done
      test "$residual" = 0 && break
      sleep 1
    done
    for ledger in "${child_ledgers[@]}"; do
      test ! -f "$ledger" || local_prepare_supervisor_absent "$ledger" \
        "$result_dir" >/dev/null 2>&1 || \
        local_prepare_supervisor_action "$ledger" "$result_dir" KILL \
          >/dev/null 2>&1 || true
    done
    kill_deadline=$((SECONDS + kill_timeout))
    while (( SECONDS < kill_deadline )); do
      residual=0
      for ledger in "${child_ledgers[@]}"; do
        test -f "$ledger" || { residual=$((residual + 1)); continue; }
        local_prepare_supervisor_absent "$ledger" "$result_dir" \
          >/dev/null 2>&1 || residual=$((residual + 1))
      done
      test "$residual" = 0 && break
      sleep 1
    done
    # Reap the setsid --wait launchers only after the lease wrappers have had
    # their shared cleanup window; with_resource_lease writes holder cleanup
    # receipts before these launchers return.
    for pid in "${child_pids[@]}"; do
      kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    done
    python3 - "$result_dir/audits/outer_interrupt_cleanup.json" \
      "$result_dir/lease" "$term_timeout" "$kill_timeout" "$prepare_scope" <<'PY'
import json,sys,time
from pathlib import Path
output,lease_root=map(Path,sys.argv[1:3])
receipts={}
roles=("lane_a_pair",) if sys.argv[5]=="lane-a" else ("dgx_a","dgx_b","x86")
for role in roles:
    path=lease_root/role/"lease_cleanup_receipt.json"
    try:value=json.loads(path.read_text(encoding="utf-8"))
    except (OSError,UnicodeError,json.JSONDecodeError):value=None
    receipts[role]={"path":str(path),"status":value.get("status") if isinstance(value,dict) else None}
checks={"term_then_kill_shared_budget":int(sys.argv[3])+int(sys.argv[4])<=300,
        "all_holder_cleanup_receipts_pass":all(value["status"]=="PASS" for value in receipts.values())}
payload={"schema_version":1,"status":"PASS" if all(checks.values()) else "FAIL",
 "lease_cleanup_receipts":receipts,"checks":checks,"recorded_unix":time.time()}
output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
PY
    exit 130
  }
  trap stop_children INT TERM HUP

  if test "$prepare_scope" = lane-a; then
    run_task lane_a_pair lane-a "$result_dir/logs/lane_a_pair_lease.log" lane_a_pair_pid
    child_pids+=("$lane_a_pair_pid"); child_roles+=(lane_a_pair)
    child_ledgers+=("$result_dir/audits/lane_a_pair_lease_wrapper_supervisor.json")
    set +e
    wait "$lane_a_pair_pid"; lane_a_pair_rc=$?
    set -e
  else
    run_task dgx_a dgx-a "$result_dir/logs/dgx_a_lease.log" dgx_a_pid
    child_pids+=("$dgx_a_pid"); child_roles+=(dgx_a)
    child_ledgers+=("$result_dir/audits/dgx_a_lease_wrapper_supervisor.json")
    run_task dgx_b dgx-b "$result_dir/logs/dgx_b_lease.log" dgx_b_pid
    child_pids+=("$dgx_b_pid"); child_roles+=(dgx_b)
    child_ledgers+=("$result_dir/audits/dgx_b_lease_wrapper_supervisor.json")
    run_task x86 isaac "$result_dir/logs/x86_lease.log" x86_pid
    child_pids+=("$x86_pid"); child_roles+=(x86)
    child_ledgers+=("$result_dir/audits/x86_lease_wrapper_supervisor.json")

    set +e
    wait "$dgx_a_pid"; dgx_a_rc=$?
    wait "$dgx_b_pid"; dgx_b_rc=$?
    wait "$x86_pid"; x86_rc=$?
    set -e
  fi
  child_pids=()
  trap - INT TERM HUP
  write_lease_summary
  set +e
  write_final_receipts
  final_rc=$?
  set -e
  exit "$final_rc"
}

if test "${1:-}" = --inside; then
  inside_main "$@"
elif test "${1:-}" = --lane-a-pair; then
  lane_a_pair_main "$@"
elif test "${1:-}" = --lease-task; then
  lease_task_main "$@"
else
  outer_main "$@"
fi
