#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: run_t5_d0_prepare_online.sh GRANT_ID AUTHORIZATION_REF RESULT_ROOT

This is the D0.0 preparation-only entrypoint.  It deploys one immutable ref to
both DGX lane hosts and the Isaac x86 host, builds the two DGX ROS workspaces,
audits the frozen five-episode input, prepares static maps and the two stopped
Isaac ROS worker containers, and records capacity.  It never loads InternVLA,
starts Isaac Sim, or runs an episode.
EOF
  exit 64
}

[[ $# -eq 3 ]] || usage
grant_id="$1"
authorization_ref="$2"
result_relative="$3"
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "$root/scripts/t5_quarantine_common.sh"
source "$root/scripts/t5_remote_compute_audit_common.sh"
# Preserve the exact frozen auditor snapshot used by the pre-deploy helper so
# EXIT cleanup can apply the same process-identity contract without trusting a
# mutable remote checkout.
remote_compute_auditor_b64="$_t5_compute_auditor_b64"
remote_compute_auditor_sha256="$_t5_compute_auditor_sha256"
readonly remote_compute_auditor_b64 remote_compute_auditor_sha256
board_relative=coordination/T5_DUAL_LANE_BOARD.md
board="$root/$board_relative"
golden_relative=configs/internnav_t5/golden_bundle_manifest.json
d0_manifest_relative=configs/internnav_t5/d0_run_manifest.json
checkpoint_content_relative=configs/internnav_t5/checkpoint_content_manifest.json

[[ "$grant_id" =~ ^t5d00[0-9]{8}t[0-9]{6}$ ]] || usage
[[ "$authorization_ref" =~ ^[0-9a-f]{40}$ ]] || usage
case "$result_relative" in
  results/internnav_t5/d0-0-prepare-"$grant_id") ;;
  *) echo "result root does not match the D0.0 grant id" >&2; exit 64 ;;
esac

git_command=(git -C "$root")
if [[ -f "$root/.git" ]] && grep -Eq '^gitdir: [A-Za-z]:/' "$root/.git"; then
  command -v git.exe >/dev/null
  command -v wslpath >/dev/null
  git_command=(git.exe -C "$(wslpath -w "$root")")
fi

# The grant is a one-commit, board-only signature over an immutable code ref.
# Untracked files are included in the cleanliness check so an online run can
# never silently depend on an uncommitted launcher or config.
"${git_command[@]}" cat-file -e "$authorization_ref^{commit}"
"${git_command[@]}" merge-base --is-ancestor "$authorization_ref" HEAD
[[ "$("${git_command[@]}" rev-list --count "$authorization_ref..HEAD" | tr -d '\r')" = 1 ]]
mapfile -t post_ref_changes < <(
  "${git_command[@]}" diff --name-only "$authorization_ref" HEAD | tr -d '\r'
)
[[ "${#post_ref_changes[@]}" -eq 1 ]]
[[ "${post_ref_changes[0]}" = "$board_relative" ]]
[[ -z "$("${git_command[@]}" status --porcelain --untracked-files=all | tr -d '\r')" ]]

validation_tmp="$(mktemp -d "${TMPDIR:-/tmp}/internnav-t5-d0-validate.XXXXXX")"
cleanup_validation_tmp() { rm -rf -- "$validation_tmp"; }
trap cleanup_validation_tmp EXIT
"${git_command[@]}" show "$authorization_ref:$board_relative" >"$validation_tmp/board.authorization.md"
"${git_command[@]}" show "HEAD:$board_relative" >"$validation_tmp/board.granted.md"
"${git_command[@]}" show "$authorization_ref:$golden_relative" >"$validation_tmp/golden.json"
"${git_command[@]}" show "$authorization_ref:$d0_manifest_relative" >"$validation_tmp/d0.json"
"${git_command[@]}" show "$authorization_ref:$checkpoint_content_relative" \
  >"$validation_tmp/checkpoint_content.json"

python3 - "$validation_tmp" "$grant_id" "$authorization_ref" "$result_relative" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

directory = Path(sys.argv[1])
grant_id, authorization_ref, result_root = sys.argv[2:5]
marker = "INTERNAV_T5_DUAL_LANE_ONLINE_GRANT_V1"
pattern = re.compile(
    rf"<!-- {marker}\s*(\{{.*?\}})\s*{marker} -->", re.DOTALL
)

old_text = (directory / "board.authorization.md").read_text(encoding="utf-8")
new_text = (directory / "board.granted.md").read_text(encoding="utf-8")
old_matches = pattern.findall(old_text)
new_matches = pattern.findall(new_text)
if len(old_matches) != 1 or len(new_matches) != 1:
    raise SystemExit("the authority board must contain exactly one grant block")
if pattern.sub(f"<!-- {marker} GRANT_BLOCK {marker} -->", old_text) != pattern.sub(
    f"<!-- {marker} GRANT_BLOCK {marker} -->", new_text
):
    raise SystemExit("authorization commit changed board content outside the grant block")

old_grant = json.loads(old_matches[0])
expected_old_keys = {
    "schema_version", "status", "grant_id", "authorization_ref_sha", "code_ref_sha", "stage",
    "lane_scope", "resource_profile", "candidate_id", "golden_bundle_sha256",
    "run_manifest_sha256", "predecessor_stage", "predecessor_result_root",
    "predecessor_receipt_sha256", "result_root",
}
if (
    set(old_grant) != expected_old_keys
    or old_grant.get("schema_version") != 2
    or old_grant.get("status") != "NO_GRANT"
    or any(old_grant.get(key) is not None for key in expected_old_keys - {"schema_version", "status"})
):
    raise SystemExit("authorization ref was not a clean NO_GRANT state")

def canonical(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()

golden = json.loads((directory / "golden.json").read_text(encoding="utf-8"))
d0 = json.loads((directory / "d0.json").read_text(encoding="utf-8"))
checkpoint_content = json.loads(
    (directory / "checkpoint_content.json").read_text(encoding="utf-8")
)
golden_sha = canonical(golden)
d0_sha = canonical(d0)
checkpoint_content_sha = canonical(checkpoint_content)
if d0.get("golden_bundle_canonical_sha256") != golden_sha:
    raise SystemExit("D0 manifest is not bound to the authorization-ref Golden Bundle")
if golden.get("status") != "FROZEN_FOR_D0_UNVALIDATED":
    raise SystemExit("unexpected Golden Bundle state for D0 preparation")
model = golden.get("model", {})
if (
    model.get("checkpoint_content_manifest")
    != "configs/internnav_t5/checkpoint_content_manifest.json"
    or model.get("checkpoint_content_manifest_canonical_sha256")
    != checkpoint_content_sha
    or checkpoint_content.get("checkpoint_revision")
    != model.get("checkpoint_revision")
):
    raise SystemExit("checkpoint content manifest is not bound to the Golden Bundle")

expected = {
    "schema_version": 2,
    "status": "GRANTED",
    "grant_id": grant_id,
    "authorization_ref_sha": authorization_ref,
    "code_ref_sha": authorization_ref,
    "stage": "d0_0_online_prepare",
    "lane_scope": "all_lanes",
    "resource_profile": "all-lanes",
    "candidate_id": golden["bundle_id"],
    "golden_bundle_sha256": golden_sha,
    "run_manifest_sha256": d0_sha,
    "predecessor_stage": None,
    "predecessor_result_root": None,
    "predecessor_receipt_sha256": None,
    "result_root": result_root,
}
grant = json.loads(new_matches[0])
if grant != expected:
    raise SystemExit(f"D0.0 grant mismatch: {grant!r}")

fixed = d0.get("fixed_input", {})
lanes = d0.get("lanes", {})
if fixed.get("episode_count") != 5 or len(fixed.get("episode_keys", [])) != 5:
    raise SystemExit("D0.0 requires the frozen five-episode input")
if set(lanes) != {"a", "b"}:
    raise SystemExit("D0.0 requires both symmetric lanes")

validation = {
    "schema_version": 1,
    "status": "PASS",
    "grant": grant,
    "golden_bundle_canonical_sha256": golden_sha,
    "run_manifest_canonical_sha256": d0_sha,
    "golden_bundle_id": golden["bundle_id"],
    "model_revision": golden["model"]["internnav_revision"],
    "checkpoint_revision": golden["model"]["checkpoint_revision"],
    "model_weight_inventory_sha256": golden["model"]["weight_inventory_sha256"],
    "checkpoint_content_manifest": golden["model"]["checkpoint_content_manifest"],
    "checkpoint_content_manifest_canonical_sha256": golden["model"][
        "checkpoint_content_manifest_canonical_sha256"
    ],
    "d0_selected_scene_geometry": golden["scenes"]["d0_selected_scene_geometry"],
    "dataset_root": fixed["dataset_remote_path"],
    "dataset_file_sha256": fixed["dataset_file_sha256"],
    "episode_keys": fixed["episode_keys"],
    "dual_isaac_minimum_available_memory_bytes": d0["x86_host_inventory"][
        "prior_t4_second_instance_required_remaining_bytes"
    ],
}
(directory / "validation.json").write_text(
    json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

mapfile -t frozen_values < <(python3 - "$validation_tmp/validation.json" <<'PY'
import json, sys
v=json.load(open(sys.argv[1], encoding="utf-8"))
for key in ("golden_bundle_canonical_sha256", "run_manifest_canonical_sha256",
            "golden_bundle_id", "model_revision", "checkpoint_revision",
            "model_weight_inventory_sha256",
            "checkpoint_content_manifest_canonical_sha256",
            "dataset_root", "dataset_file_sha256"):
    print(v[key])
PY
)
golden_sha256="${frozen_values[0]}"
run_manifest_sha256="${frozen_values[1]}"
golden_bundle_id="${frozen_values[2]}"
model_revision="${frozen_values[3]}"
checkpoint_revision="${frozen_values[4]}"
model_weight_inventory_sha256="${frozen_values[5]}"
checkpoint_content_manifest_sha256="${frozen_values[6]}"
dataset_root="${frozen_values[7]}"
dataset_sha256="${frozen_values[8]}"
[[ "$dataset_root" =~ ^/home/song/internnav-t1-t2/episodes/[A-Za-z0-9._/-]+$ ]]
[[ "$dataset_sha256" =~ ^[0-9a-f]{64}$ ]]
[[ "$model_weight_inventory_sha256" =~ ^[0-9a-f]{64}$ ]]
[[ "$checkpoint_content_manifest_sha256" =~ ^[0-9a-f]{64}$ ]]

result_dir="$root/$result_relative"

# Acquire all four physical resources before atomically consuming result_root.
# Lease logs are staged beside it because with_resource_lease needs a log path
# before the wrapped command begins; after release they are copied into the
# immutable result root without overwriting an existing file.
if [[ "${INTERNNAV_T5_INSIDE_D0_PREPARE:-0}" != 1 ]]; then
  test ! -e "$result_dir"
  mkdir -p "$root/results/internnav_t5"
  lease_bootstrap="$(mktemp -d "$root/results/internnav_t5/.d0-0-lease-$grant_id.XXXXXX")"
  trap - EXIT
  cleanup_validation_tmp
  set +e
  bash "$root/scripts/with_resource_lease.sh" all-lanes \
    --owner codex-00 \
    --task "t5_d0_0_prepare:$grant_id:$authorization_ref" \
    --log-dir "$lease_bootstrap" \
    --acquire-timeout 30 \
    --cleanup-timeout 180 --kill-wait-timeout 30 \
    -- env INTERNNAV_T5_INSIDE_D0_PREPARE=1 \
      INTERNNAV_T5_RESOURCE_LEASE_ACK=all-lanes \
      bash "$root/coordination/run_t5_d0_prepare_online.sh" \
        "$grant_id" "$authorization_ref" "$result_relative"
  lease_rc=$?
  set -e
  if test -d "$result_dir"; then
    install -d -m 700 "$result_dir/lease"
    cp -a -- "$lease_bootstrap/." "$result_dir/lease/"
    python3 - "$result_dir/lease_release_summary.json" \
      "$result_dir/lease/lease_metadata.txt" \
      "$result_dir/lease/lease_cleanup_receipt.json" "$lease_rc" <<'PY'
import hashlib, json, sys, time
from pathlib import Path
output, metadata_path, cleanup_path = map(Path, sys.argv[1:4])
command_rc = int(sys.argv[4])
text = metadata_path.read_text(encoding="utf-8") if metadata_path.is_file() else ""
holders = sorted(metadata_path.parent.glob("holder_*.stdout.log"))
cleanup = json.loads(cleanup_path.read_text(encoding="utf-8")) if cleanup_path.is_file() else None
checks = {
    "four_holders_recorded": len(holders) == 4,
    "all_resources_named": all(name in text for name in ("dgx_a", "dgx_b", "isaac_gpu0", "isaac_gpu1")),
    "wrapped_command_released": "state=RELEASED" in text,
    "wrapped_command_exit_recorded": f"command_exit={command_rc}" in text,
    "cleanup_before_release": isinstance(cleanup, dict)
        and cleanup.get("status") == "PASS"
        and cleanup.get("wrapped_process_group_absent_before_lock_release") is True,
}
payload = {
    "schema_version": 1,
    "status": "PASS" if all(checks.values()) else "FAIL",
    "command_exit": command_rc,
    "checks": checks,
    "holder_logs": [path.name for path in holders],
    "lease_cleanup_receipt": cleanup,
    "recorded_unix": time.time(),
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
    python3 - "$result_dir/d0_prepare_final_summary.json" \
      "$result_dir/d0_prepare_summary.json" \
      "$result_dir/lease_release_summary.json" \
      "$result_dir/remote_cleanup_receipt.json" "$lease_rc" <<'PY'
import hashlib, json, sys, time
from pathlib import Path
output, preparation_path, lease_path, remote_cleanup_path = map(Path, sys.argv[1:5])
command_rc = int(sys.argv[5])
preparation = json.loads(preparation_path.read_text(encoding="utf-8")) if preparation_path.is_file() else None
lease = json.loads(lease_path.read_text(encoding="utf-8"))
remote_cleanup = json.loads(remote_cleanup_path.read_text(encoding="utf-8")) \
    if remote_cleanup_path.is_file() else None
checks = {
    "wrapped_command_exit_zero": command_rc == 0,
    "preparation_pass": preparation is not None and preparation.get("status") == "PASS",
    "formal_board_authorization": preparation is not None
        and preparation.get("authorization_mode") == "FORMAL_BOARD_GRANT_V2"
        and preparation.get("board_grant_used") is True,
    "remote_cleanup_pass": remote_cleanup is not None
        and remote_cleanup.get("status") == "PASS",
    "lease_release_pass": lease.get("status") == "PASS",
}
payload = {
    "schema_version": 1,
    "status": "PASS" if all(checks.values()) else "FAIL",
    "stage": "d0_0_online_prepare",
    "authorization_mode": preparation.get("authorization_mode") if preparation else None,
    "board_grant_used": preparation.get("board_grant_used") if preparation else None,
    "grant_id": preparation.get("grant_id") if preparation else None,
    "checks": checks,
    "preparation_summary": preparation_path.name,
    "preparation_summary_sha256": hashlib.sha256(preparation_path.read_bytes()).hexdigest()
        if preparation_path.is_file() else None,
    "lease_release_summary": lease_path.name,
    "lease_release_summary_sha256": hashlib.sha256(lease_path.read_bytes()).hexdigest(),
    "remote_cleanup_receipt": remote_cleanup_path.name,
    "remote_cleanup_receipt_sha256": hashlib.sha256(remote_cleanup_path.read_bytes()).hexdigest()
        if remote_cleanup_path.is_file() else None,
    "authorization_ref_sha": preparation.get("authorization_ref_sha") if preparation else None,
    "code_ref_sha": preparation.get("code_ref_sha") if preparation else None,
    "golden_bundle_canonical_sha256": preparation.get("golden_bundle_canonical_sha256")
        if preparation else None,
    "run_manifest_canonical_sha256": preparation.get("run_manifest_canonical_sha256")
        if preparation else None,
    "deployment_roots": preparation.get("deployment_roots") if preparation else None,
    "recorded_unix": time.time(),
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
    final_status="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$result_dir/d0_prepare_final_summary.json")"
    test "$final_status" = PASS || lease_rc=1
    rm -rf -- "$lease_bootstrap"
  else
    printf 'D0.0 did not consume result_root; lease diagnostics remain at %s\n' \
      "$lease_bootstrap" >&2
  fi
  exit "$lease_rc"
fi

test "${INTERNNAV_T5_RESOURCE_LEASE_ACK:-}" = all-lanes
test ! -e "$result_dir"
umask 077
mkdir "$result_dir"
mkdir "$result_dir/logs" "$result_dir/remote" "$result_dir/maps"
cp "$validation_tmp/validation.json" "$result_dir/grant_validation.json"
trap - EXIT
cleanup_validation_tmp

ssh_options=(-T -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=2)
dgx_a_target=railgun@10.100.100.128
dgx_b_target=rail@10.100.120.116
x86_target=song@10.100.120.111
tag="${grant_id}-${authorization_ref:0:12}"
dgx_a_root="/home/railgun/internnav-t1-t2/.t5-deployments/${tag}-lane-a"
dgx_b_root="/home/rail/internnav-t1-t2/.t5-deployments/${tag}-lane-b"
x86_prepare_root="/home/song/internnav-t1-t2/.t5-deployments/${tag}-isaac-prepare"
x86_a_root="/home/song/internnav-t1-t2/.t5-deployments/${tag}-isaac-a"
x86_b_root="/home/song/internnav-t1-t2/.t5-deployments/${tag}-isaac-b"
archive="$result_dir/deployment.tar.gz"

remote() {
  local target="$1"; shift
  ssh "${ssh_options[@]}" "$target" "$@"
}

dgx_a_ssh_pid=""
dgx_b_ssh_pid=""
containers_owned_marker="$x86_prepare_root/results/d0_prepare/containers_owned"
dgx_a_quarantine_armed=false
dgx_b_quarantine_armed=false
x86_quarantine_armed=false
dgx_quarantine_file=/tmp/internnav_dgx.quarantine
isaac_quarantine_file=/tmp/internnav_isaac.quarantine

# This program is sent over stdin/base64 rather than read from a mutable remote
# checkout.  It only signals a process group after a non-ancestor member has
# been associated with this run's unique deployment root.  A reused PGID with
# no such association is never signalled and makes cleanup fail closed.
read -r -d '' remote_cleanup_program <<'REMOTE_CLEANUP' || true
import base64
import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

ledger_path = Path(sys.argv[1])
deployment_roots = tuple(root for root in sys.argv[2].split("|") if root)
if not deployment_roots or any(not root.startswith("/") for root in deployment_roots):
    raise SystemExit("invalid deployment root scope")
role = sys.argv[3]
ownership_dir = None if sys.argv[4] == "-" else Path(sys.argv[4])
term_timeout = float(sys.argv[5])
kill_timeout = float(sys.argv[6])
quarantine_armed = sys.argv[7] == "true"
quarantine_path = Path(sys.argv[8])
expected_quarantine_tag = sys.argv[9]
expected_compute_auditor_sha256 = sys.argv[10]
compute_auditor_b64 = sys.argv[11]
if role not in {"dgx_a", "dgx_b", "x86"}:
    raise SystemExit("invalid cleanup role")

ancestors = set()
pid = os.getpid()
while pid > 1 and pid not in ancestors:
    ancestors.add(pid)
    try:
        lines = Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines()
        pid = int(next(line.split()[1] for line in lines if line.startswith("PPid:")))
    except (FileNotFoundError, PermissionError, StopIteration, ValueError):
        break
own_pgid = os.getpgrp()

def process_table():
    table = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        process_id = int(entry.name)
        if process_id in ancestors:
            continue
        try:
            group_id = os.getpgid(process_id)
        except (ProcessLookupError, PermissionError):
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            ).strip()
            command_readable = True
        except OSError:
            command = ""
            command_readable = False
        table.append({
            "pid": process_id, "pgid": group_id, "command": command,
            "command_readable": command_readable,
        })
    return table

def scoped_processes(table):
    return [
        item for item in table
        if any(root in item["command"] for root in deployment_roots)
    ]

def group_members(table, group_id):
    return [item for item in table if item["pgid"] == group_id]

def run_embedded_forbidden_compute_audit(
    auditor_b64, expected_source_sha256, proc_root="/proc"
):
    """Run the frozen identity rules against every process on a DGX.

    The auditor is executed as a library so its exact process_identities() and
    forbidden_reason() implementation is shared with the pre-deploy audit.  A
    malformed payload, missing API, or unreadable /proc entry is evidence of an
    audit failure rather than evidence of process absence.
    """
    proc_path = Path(proc_root)
    matches = []
    errors = []
    scanned = 0
    source_sha256 = None
    namespace = {}
    try:
        source_bytes = base64.b64decode(auditor_b64, validate=True)
        source_sha256 = hashlib.sha256(source_bytes).hexdigest()
        if source_sha256 != expected_source_sha256:
            raise ValueError("auditor_sha256_mismatch")
        source = source_bytes.decode("utf-8", errors="strict")
        namespace = {
            "__name__": "t5_process_identity_audit_embedded",
            "__file__": "t5_process_identity_audit.py",
        }
        exec(compile(source, namespace["__file__"], "exec"), namespace)
        for api_name in ("_basename", "process_identities", "forbidden_reason"):
            if not callable(namespace.get(api_name)):
                raise ValueError(f"missing_auditor_api:{api_name}")
    except Exception as error:
        errors.append({
            "pid": None,
            "error": f"auditor_load:{type(error).__name__}",
            "detail": str(error),
        })

    if not errors:
        try:
            if not proc_path.is_dir():
                errors.append({"pid": None, "error": "proc_root_missing"})
                entries = []
            else:
                entries = sorted(proc_path.iterdir(), key=lambda item: item.name)
        except OSError as error:
            errors.append({
                "pid": None,
                "error": f"proc_root_read:{type(error).__name__}",
            })
            entries = []
        for entry in entries:
            if not entry.name.isdigit():
                continue
            process_id = int(entry.name)
            try:
                raw = (entry / "cmdline").read_bytes()
            except FileNotFoundError:
                continue
            except OSError as error:
                errors.append({"pid": process_id, "error": type(error).__name__})
                continue
            if not raw:
                continue
            try:
                argv = [
                    part.decode("utf-8", errors="replace")
                    for part in raw.split(b"\0") if part
                ]
                identities = namespace["process_identities"](argv)
                reason = namespace["forbidden_reason"](identities)
                scanned += 1
            except Exception as error:
                errors.append({
                    "pid": process_id,
                    "error": f"process_parse:{type(error).__name__}",
                })
                continue
            if reason:
                matches.append({
                    "pid": process_id,
                    "executable": namespace["_basename"](argv[0]),
                    "identities": sorted(identities),
                    "reason": reason,
                })

    status = "ERROR" if errors else ("FAIL" if matches else "PASS")
    evidence = {
        "schema_version": 1,
        "status": status,
        "mode": "forbidden-compute",
        "proc_root": str(proc_path),
        "deployment_root": None,
        "scanned_process_count": scanned,
        "match_count": len(matches),
        "matches": matches,
        "errors": errors,
    }
    check = (
        source_sha256 == expected_source_sha256
        and evidence["status"] == "PASS"
        and evidence["match_count"] == 0
        and evidence["errors"] == []
    )
    return {
        "schema_version": 1,
        "status": "PASS" if check else "FAIL",
        "auditor_sha256": source_sha256,
        "expected_auditor_sha256": expected_source_sha256,
        "evidence": evidence,
        "matches": matches,
        "errors": errors,
        "check": check,
    }

ledger = {}
ledger_error = None
if ledger_path.is_file():
    try:
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                ledger[key] = value
    except (OSError, UnicodeError) as error:
        ledger_error = type(error).__name__
claimed_pgid = None
try:
    candidate = int(ledger.get("pgid", ""))
    if candidate > 1:
        claimed_pgid = candidate
except ValueError:
    if ledger:
        ledger_error = "invalid_pgid"

initial_table = process_table()
initial_scoped = scoped_processes(initial_table)
scoped_groups = sorted(
    {item["pgid"] for item in initial_scoped if item["pgid"] != own_pgid}
)
claimed_members = group_members(initial_table, claimed_pgid) if claimed_pgid else []
claimed_associated = claimed_pgid is None or not claimed_members or all(
    any(root in item["command"] for root in deployment_roots)
    for item in claimed_members
)
target_groups = list(scoped_groups)
signals = []

def signal_groups(groups, requested_signal):
    for group_id in sorted(set(groups)):
        if group_id <= 1 or group_id == own_pgid:
            continue
        current = process_table()
        # Re-prove association immediately before every group signal.
        members = group_members(current, group_id)
        associated_members = [
            item for item in members
            if any(root in item["command"] for root in deployment_roots)
        ]
        # killpg() affects every member.  If an unrelated process has joined
        # the deployment group, preserve quarantine instead of signalling the
        # mixed group.
        associated = bool(members) and len(associated_members) == len(members)
        if not associated:
            signals.append(
                {"pgid": group_id, "signal": requested_signal.name, "sent": False,
                 "reason": "mixed_or_unassociated_process_group"}
            )
            continue
        try:
            os.killpg(group_id, requested_signal)
            signals.append(
                {"pgid": group_id, "signal": requested_signal.name, "sent": True}
            )
        except ProcessLookupError:
            signals.append(
                {"pgid": group_id, "signal": requested_signal.name, "sent": False,
                 "reason": "already_absent"}
            )
        except PermissionError:
            signals.append(
                {"pgid": group_id, "signal": requested_signal.name, "sent": False,
                 "reason": "permission_denied"}
            )

def wait_for_absence(timeout):
    deadline = time.monotonic() + timeout
    while True:
        table = process_table()
        scoped = scoped_processes(table)
        claimed = group_members(table, claimed_pgid) if claimed_pgid else []
        if not scoped and not claimed:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.25)

if target_groups:
    signal_groups(target_groups, signal.SIGTERM)
term_absent = wait_for_absence(term_timeout)
if not term_absent:
    remaining_table = process_table()
    remaining_groups = {
        item["pgid"] for item in scoped_processes(remaining_table)
        if item["pgid"] != own_pgid
    }
    signal_groups(remaining_groups, signal.SIGKILL)
    kill_absent = wait_for_absence(kill_timeout)
else:
    kill_absent = True

containers = {}
container_cleanup_ok = True
if role == "x86":
    for lane, name in (("a", "internnav_t5_isaac_a"), ("b", "internnav_t5_isaac_b")):
        marker = ownership_dir / name if ownership_dir else None
        owned = bool(marker and marker.is_file())
        stop_rc = None
        if owned:
            try:
                stopped = subprocess.run(
                    ["docker", "stop", "-t", "10", name],
                    text=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=15,
                )
                stop_rc = stopped.returncode
            except (OSError, subprocess.TimeoutExpired):
                stop_rc = 124
        try:
            inspected = subprocess.run(
                ["docker", "inspect", name], text=True, capture_output=True, timeout=10
            )
        except (OSError, subprocess.TimeoutExpired):
            inspected = None
        if inspected is not None and inspected.returncode == 0:
            try:
                value = json.loads(inspected.stdout)[0]
                running = bool(value["State"]["Running"])
                container_pid = int(value["State"]["Pid"])
                absent_or_stopped = not running and container_pid == 0
            except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
                running = None
                container_pid = None
                absent_or_stopped = False
        elif inspected is not None:
            running = False
            container_pid = 0
            absent_or_stopped = True
        else:
            running = None
            container_pid = None
            absent_or_stopped = False
        containers[lane] = {
            "name": name,
            "owned_by_this_run": owned,
            "stop_exit": stop_rc,
            "running": running,
            "pid": container_pid,
            "absent_or_stopped": absent_or_stopped,
        }
        container_cleanup_ok = container_cleanup_ok and absent_or_stopped

final_table = process_table()
final_scoped = scoped_processes(final_table)
final_claimed = group_members(final_table, claimed_pgid) if claimed_pgid else []
try:
    socket_check = subprocess.run(
        ["ss", "-H", "-lntup"], text=True, capture_output=True, timeout=10
    )
except (OSError, subprocess.TimeoutExpired):
    socket_check = None
t5_ports = {25137, 25138, 25139, 25140, 25141, 25239, 25240, 25241}
socket_lines = [
    line for line in (socket_check.stdout.splitlines() if socket_check is not None else [])
    if any(f":{port} " in line for port in t5_ports)
]
checks = {
    "ledger_readable_or_absent": ledger_error is None,
    "claimed_pgid_associated_or_absent": claimed_associated,
    "deployment_processes_absent": not final_scoped,
    "claimed_pgid_absent": not final_claimed,
    "term_then_kill_completed": term_absent or kill_absent,
    "t5_sockets_absent": socket_check is not None
        and socket_check.returncode == 0 and not socket_lines,
    "owned_containers_absent_or_stopped": container_cleanup_ok,
}
global_compute_audit = None
if role in {"dgx_a", "dgx_b"}:
    global_compute_audit = run_embedded_forbidden_compute_audit(
        compute_auditor_b64, expected_compute_auditor_sha256
    )
    checks["dgx_global_forbidden_compute_absent"] = global_compute_audit["check"]
quarantine_present_at_start = os.path.lexists(quarantine_path)
quarantine_values = {}
if quarantine_present_at_start:
    try:
        marker_metadata = quarantine_path.lstat()
        if (
            stat.S_ISLNK(marker_metadata.st_mode)
            or not stat.S_ISREG(marker_metadata.st_mode)
            or marker_metadata.st_mode & 0o077
            or marker_metadata.st_uid != os.geteuid()
            or marker_metadata.st_nlink != 1
        ):
            raise ValueError("unsafe_marker_metadata")
        for line in quarantine_path.read_text(encoding="utf-8").splitlines():
            if not line or "=" not in line:
                raise ValueError("malformed_marker_line")
            key, value = line.split("=", 1)
            if not key or key in quarantine_values:
                raise ValueError("duplicate_marker_key")
            quarantine_values[key] = value
        marker_keys = {
            "schema_version", "state", "reason", "role", "run_tag",
            "scope_roots", "armed_at",
        }
        if set(quarantine_values) != marker_keys:
            raise ValueError("marker_key_set_mismatch")
        armed_at = quarantine_values["armed_at"]
        parsed_armed_at = time.strptime(armed_at, "%Y-%m-%dT%H:%M:%SZ")
        if time.strftime("%Y-%m-%dT%H:%M:%SZ", parsed_armed_at) != armed_at:
            raise ValueError("invalid_armed_at")
    except (OSError, UnicodeError, ValueError):
        quarantine_values = {}
quarantine_owned = (
    quarantine_armed
    and quarantine_present_at_start
    and quarantine_values.get("schema_version") == "1"
    and quarantine_values.get("state") == "DIRTY"
    and quarantine_values.get("reason") == "d0_prepare_in_progress"
    and quarantine_values.get("run_tag") == expected_quarantine_tag
    and quarantine_values.get("role") == role
    and tuple(quarantine_values.get("scope_roots", "").split("|"))
        == deployment_roots
)
quarantine_cleared = False
quarantine_error = None
if quarantine_armed and not quarantine_present_at_start:
    # The caller received confirmation that the marker was armed, but it has
    # disappeared. Restore fail-closed state and refuse to certify cleanup.
    try:
        restored = {
            "schema_version": "1",
            "state": "DIRTY",
            "reason": "d0_prepare_in_progress",
            "role": role,
            "run_tag": expected_quarantine_tag,
            "scope_roots": "|".join(deployment_roots),
            "armed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(quarantine_path, flags, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", closefd=True) as stream:
            stream.write("".join(f"{key}={value}\n" for key, value in restored.items()))
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as error:
        quarantine_error = type(error).__name__
elif quarantine_owned and all(checks.values()):
    try:
        quarantine_path.unlink()
        quarantine_cleared = not os.path.lexists(quarantine_path)
    except OSError as error:
        quarantine_error = type(error).__name__
checks["quarantine_marker_present_if_armed"] = not quarantine_armed or (
    quarantine_present_at_start and quarantine_owned
)
checks["unowned_quarantine_never_removed"] = quarantine_armed or (
    not quarantine_present_at_start or os.path.lexists(quarantine_path)
)
checks["quarantine_cleared_only_after_audit_pass"] = (
    quarantine_cleared if quarantine_armed else not quarantine_present_at_start
)
payload = {
    "schema_version": 1,
    "status": "PASS" if all(checks.values()) else "FAIL",
    "role": role,
    "deployment_roots": list(deployment_roots),
    "ledger": {
        "path": str(ledger_path),
        "present": ledger_path.is_file(),
        "state": ledger.get("state"),
        "claimed_pgid": claimed_pgid,
        "error": ledger_error,
    },
    "initial": {
        "scoped_pids": sorted(item["pid"] for item in initial_scoped),
        "scoped_pgids": scoped_groups,
        "claimed_group_member_pids": sorted(item["pid"] for item in claimed_members),
    },
    "signals": signals,
    "final": {
        "scoped_pids": sorted(item["pid"] for item in final_scoped),
        "claimed_group_member_pids": sorted(item["pid"] for item in final_claimed),
        "socket_count": len(socket_lines),
    },
    "containers": containers,
    "global_compute_audit": global_compute_audit,
    "quarantine": {
        "path": str(quarantine_path),
        "armed_by_caller": quarantine_armed,
        "present_at_start": quarantine_present_at_start,
        "owned_by_this_run": quarantine_owned,
        "expected_run_tag": expected_quarantine_tag,
        "cleared": quarantine_cleared,
        "error": quarantine_error,
    },
    "checks": checks,
    "recorded_unix": time.time(),
}
print(json.dumps(payload, sort_keys=True))
raise SystemExit(0 if payload["status"] == "PASS" else 75)
REMOTE_CLEANUP
remote_cleanup_b64="$(printf '%s' "$remote_cleanup_program" | base64 | tr -d '\r\n')"

run_remote_cleanup() {
  local target="$1" ledger="$2" deployment="$3" role="$4" ownership="$5"
  local quarantine_armed="$6" quarantine_file="$7"
  local output="$8" rc_file="$9" error_log="${10}"
  local compute_auditor_sha256=- compute_auditor_b64=-
  case "$role" in
    dgx_a|dgx_b)
      compute_auditor_sha256="$remote_compute_auditor_sha256"
      compute_auditor_b64="$remote_compute_auditor_b64"
      ;;
    x86) ;;
    *) return 64 ;;
  esac
  set +e
  remote "$target" \
    "python3 -c \"\$(printf '%s' '$remote_cleanup_b64' | base64 -d)\" '$ledger' '$deployment' '$role' '$ownership' 20 10 '$quarantine_armed' '$quarantine_file' '$tag' '$compute_auditor_sha256' '$compute_auditor_b64'" \
    >"$output" 2>>"$error_log"
  printf '%s\n' "$?" >"$rc_file"
}

cleanup_inside() {
  local incoming=$?
  local cleanup_status cleanup_exit="$incoming"
  trap - EXIT INT TERM HUP
  set +e
  for pid in "$dgx_a_ssh_pid" "$dgx_b_ssh_pid"; do
    test -z "$pid" || ! kill -0 "$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
  done
  cleanup_tmp="$result_dir/.remote-cleanup"
  mkdir -p "$cleanup_tmp"
  run_remote_cleanup "$dgx_a_target" \
    "$dgx_a_root/results/d0_prepare/build_ledger.txt" "$dgx_a_root" dgx_a - \
    "$dgx_a_quarantine_armed" "$dgx_quarantine_file" \
    "$cleanup_tmp/dgx_a.json" "$cleanup_tmp/dgx_a.rc" \
    "$result_dir/logs/dgx_a_cleanup_ssh.log" &
  cleanup_a_pid=$!
  run_remote_cleanup "$dgx_b_target" \
    "$dgx_b_root/results/d0_prepare/build_ledger.txt" "$dgx_b_root" dgx_b - \
    "$dgx_b_quarantine_armed" "$dgx_quarantine_file" \
    "$cleanup_tmp/dgx_b.json" "$cleanup_tmp/dgx_b.rc" \
    "$result_dir/logs/dgx_b_cleanup_ssh.log" &
  cleanup_b_pid=$!
  run_remote_cleanup "$x86_target" \
    "$x86_prepare_root/results/d0_prepare/prepare_ledger.txt" \
    "$x86_prepare_root|$x86_a_root|$x86_b_root" x86 \
    "$containers_owned_marker" "$x86_quarantine_armed" "$isaac_quarantine_file" \
    "$cleanup_tmp/x86.json" "$cleanup_tmp/x86.rc" \
    "$result_dir/logs/x86_cleanup_ssh.log" &
  cleanup_x86_pid=$!
  wait "$cleanup_a_pid" "$cleanup_b_pid" "$cleanup_x86_pid" 2>/dev/null || true
  wait 2>/dev/null || true
  python3 - "$result_dir/remote_cleanup_receipt.json" "$cleanup_tmp" "$incoming" <<'PY'
import json, sys, time
from pathlib import Path
output, source = Path(sys.argv[1]), Path(sys.argv[2])
incoming = int(sys.argv[3])
hosts = {}
checks = {}
for host in ("dgx_a", "dgx_b", "x86"):
    rc_path, value_path = source / f"{host}.rc", source / f"{host}.json"
    try:
        rc = int(rc_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        rc = None
    try:
        value = json.loads(value_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        value = None
    hosts[host] = {"ssh_exit": rc, "audit": value}
    checks[f"{host}_cleanup_pass"] = (
        rc == 0 and isinstance(value, dict) and value.get("status") == "PASS"
    )
payload = {
    "schema_version": 1,
    "status": "PASS" if all(checks.values()) else "FAIL",
    "incoming_exit": incoming,
    "hosts": hosts,
    "checks": checks,
    "recorded_unix": time.time(),
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  cleanup_status="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$result_dir/remote_cleanup_receipt.json" 2>/dev/null)"
  rm -rf -- "$cleanup_tmp"
  if test "$cleanup_status" != PASS; then
    cleanup_exit=75
  else
    cleanup_exit="$incoming"
  fi
  exit "$cleanup_exit"
}
trap cleanup_inside EXIT
trap 'exit 130' INT TERM HUP

# Fail closed before mutating any host.  In particular, a preparation grant
# never takes over an already-running Isaac worker container.
remote "$dgx_a_target" \
  "set -euo pipefail; test \"\$(id -un)\" = railgun; ip -4 -o addr show | grep -Fq ' 10.100.100.128/'; command -v setsid >/dev/null; command -v nvidia-smi >/dev/null; test ! -e '$dgx_a_root'; test -d /home/railgun/internnav-t0/InternNav/checkpoints/InternVLA-N1-DualVLN; test -x /home/railgun/internnav-t0/venv-model/bin/python"
remote "$dgx_b_target" \
  "set -euo pipefail; test \"\$(id -un)\" = rail; ip -4 -o addr show | grep -Fq ' 10.100.120.116/'; command -v setsid >/dev/null; command -v nvidia-smi >/dev/null; test ! -e '$dgx_b_root'; test -d /home/rail/internnav-t0/InternNav/checkpoints/InternVLA-N1-DualVLN; test -x /home/rail/internnav-t0/venv-model/bin/python"
remote "$x86_target" \
  "set -euo pipefail; test \"\$(id -un)\" = song; ip -4 -o addr show | grep -Fq ' 10.100.120.111/'; command -v flock >/dev/null; command -v docker >/dev/null; command -v nvidia-smi >/dev/null; test ! -e '$x86_prepare_root'; test ! -e '$x86_a_root'; test ! -e '$x86_b_root'; test -f '$dataset_root/val_unseen/val_unseen.json.gz'; test \"\$(sha256sum '$dataset_root/val_unseen/val_unseen.json.gz' | cut -d' ' -f1)\" = '$dataset_sha256'; for c in internnav_t5_isaac_a internnav_t5_isaac_b; do if docker container inspect \"\$c\" >/dev/null 2>&1; then test \"\$(docker inspect -f '{{.State.Running}}' \"\$c\")\" != true; fi; done"

# Arm host-persistent quarantine before the first deployment mutation. The
# EXIT cleanup removes each marker only after that host proves process, PGID,
# socket, and (for x86) owned-container absence. If SSH disappears, the marker
# remains on the shared resource host and the next lease is denied.
t5_quarantine_arm "$dgx_a_target" "$dgx_quarantine_file" dgx_a "$tag" \
  "$dgx_a_root" "$result_dir/remote/dgx_a_quarantine_arm.json" \
  d0_prepare_in_progress
dgx_a_quarantine_armed=true
t5_quarantine_arm "$dgx_b_target" "$dgx_quarantine_file" dgx_b "$tag" \
  "$dgx_b_root" "$result_dir/remote/dgx_b_quarantine_arm.json" \
  d0_prepare_in_progress
dgx_b_quarantine_armed=true
t5_quarantine_arm "$x86_target" "$isaac_quarantine_file" x86 "$tag" \
  "$x86_prepare_root|$x86_a_root|$x86_b_root" \
  "$result_dir/remote/x86_quarantine_arm.json" d0_prepare_in_progress
x86_quarantine_armed=true

# Quarantine is armed before observing the whole DGX process table.  These
# audits are global rather than deployment-scoped: any production compute
# identity left by another run blocks every deployment mutation fail closed.
t5_remote_compute_absent "$dgx_a_target" \
  "$result_dir/remote/dgx_a_global_compute_predeploy.json"
t5_remote_compute_absent "$dgx_b_target" \
  "$result_dir/remote/dgx_b_global_compute_predeploy.json"

# Exactly the authorization ref is archived.  Credentials, ignored files,
# mutable results and the grant-only board commit cannot enter a deployment.
bash "$root/scripts/create_source_bundle.sh" "$authorization_ref" "$archive" 1
archive_sha256="$(sha256sum "$archive" | cut -d' ' -f1)"

deploy() {
  local target="$1" destination="$2"
  remote "$target" "install -d -m 700 '$destination'"
  gzip -dc "$archive" | ssh "${ssh_options[@]}" "$target" "tar -C '$destination' -xf -"
  remote "$target" \
    "printf '%s\n' '$authorization_ref' >'$destination/T5_DEPLOYMENT_REF'; printf '%s\n' '$archive_sha256' >'$destination/T5_DEPLOYMENT_ARCHIVE_SHA256'"
}
deploy "$dgx_a_target" "$dgx_a_root"
deploy "$dgx_b_target" "$dgx_b_root"
deploy "$x86_target" "$x86_prepare_root"
deploy "$x86_target" "$x86_a_root"
deploy "$x86_target" "$x86_b_root"

read -r -d '' dgx_build_program <<'REMOTE_DGX_BUILD' || true
set -euo pipefail
deployment_root="$1"
expected_model_revision="$2"
expected_checkpoint_revision="$3"
expected_weight_inventory_sha256="$4"
expected_checkpoint_manifest_sha256="$5"
audit="$deployment_root/results/d0_prepare"
workspace="$deployment_root/ros_ws"
mkdir -p "$audit" "$workspace/src"
started="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
pgid="$(ps -o pgid= -p $$ | tr -d ' ')"
write_ledger() {
  local state="$1" rc="$2"
  {
    printf 'state=%s\n' "$state"
    printf 'pid=%s\n' "$$"
    printf 'pgid=%s\n' "$pgid"
    printf 'started_at=%s\n' "$started"
    printf 'ended_at=%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    printf 'exit_code=%s\n' "$rc"
    printf 'command=build_t4_host_ros.sh\n'
  } >"$audit/build_ledger.txt"
}
finish() {
  local rc=$?
  trap - EXIT HUP INT TERM
  if test "$rc" = 0; then write_ledger PASS 0; else write_ledger FAIL "$rc"; fi
  exit "$rc"
}
trap finish EXIT
trap 'exit 130' HUP INT TERM
write_ledger RUNNING -1

capture_capacity() {
  python3 - "$1" <<'PY'
import json, os, shutil, subprocess, sys, time
from pathlib import Path
memory={}
for line in Path('/proc/meminfo').read_text().splitlines():
    key, value = line.split(':', 1)
    if key in {'MemTotal','MemAvailable','SwapTotal','SwapFree'}:
        memory[key + '_kib'] = int(value.split()[0])
disk=shutil.disk_usage(str(Path(sys.argv[1]).parent))
gpu=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.total,memory.free,driver_version',
                    '--format=csv,noheader,nounits'],text=True,capture_output=True,check=True)
compute=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory',
                        '--format=csv,noheader,nounits'],text=True,capture_output=True)
payload={'schema_version':1,'memory':memory,
         'disk':{'total_bytes':disk.total,'used_bytes':disk.used,'free_bytes':disk.free},
         'gpu_csv':gpu.stdout.splitlines(),'compute_process_csv':compute.stdout.splitlines(),
         'cpu_count':os.cpu_count(),'recorded_unix':time.time()}
Path(sys.argv[1]).write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
PY
}
capture_capacity "$audit/capacity_before.json"
"$HOME/internnav-t0/venv-model/bin/python" - "$audit/model_runtime_environment.json" <<'PY'
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import torch

packages = {}
for name in ("torch", "transformers", "safetensors", "numpy"):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None
checks = {
    "required_packages_present": all(packages.values()),
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count_nonzero": torch.cuda.device_count() >= 1,
}
payload = {
    "schema_version": 1,
    "status": "PASS" if all(checks.values()) else "FAIL",
    "python_executable": sys.executable,
    "packages": packages,
    "cuda_device_count": torch.cuda.device_count(),
    "cuda_device_names": [
        torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
    ],
    "checks": checks,
    "recorded_unix": time.time(),
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
raise SystemExit(0 if payload["status"] == "PASS" else 1)
PY
python3 "$deployment_root/scripts/validate_t5_frozen_assets.py" checkpoint \
  --golden "$deployment_root/configs/internnav_t5/golden_bundle_manifest.json" \
  --content-manifest \
    "$deployment_root/configs/internnav_t5/checkpoint_content_manifest.json" \
  --internnav-root "$HOME/internnav-t0/InternNav" \
  --output "$audit/model_inventory.json"
python3 - "$audit/model_inventory.json" "$expected_model_revision" \
  "$expected_checkpoint_revision" "$expected_weight_inventory_sha256" \
  "$expected_checkpoint_manifest_sha256" <<'PY'
import json, sys
value=json.load(open(sys.argv[1], encoding='utf-8'))
checks={
  'model_revision':value.get('model_revision')==sys.argv[2],
  'checkpoint_revision':value.get('checkpoint_revision')==sys.argv[3],
  'runtime_inventory':value.get('expected_runtime_weight_inventory_sha256')==sys.argv[4],
  'content_manifest':value.get('checkpoint_content_manifest_canonical_sha256')==sys.argv[5],
}
if value.get('status')!='PASS' or not all(checks.values()):
    raise SystemExit(f'checkpoint/Golden binding failed: {checks}')
PY
export PYTHONDONTWRITEBYTECODE=1
set +u
source /opt/ros/jazzy/setup.bash
set -u
INTERNNAV_T1_CONTROL_ROOT="$deployment_root" \
INTERNVLA_ROS_WS="$workspace" \
INTERNNAV_RUNTIME_POLICY=completion_sim \
INTERNNAV_SIMULATION_TARGET=isaac \
INTERNNAV_T4_RESOURCE_LEASE_ACK=dgx+isaac \
INTERNNAV_T5_RESOURCE_LEASE_ACK=all-lanes \
  bash "$deployment_root/scripts/build_t4_host_ros.sh" \
  >"$audit/ros_build.log" 2>&1
capture_capacity "$audit/capacity_after.json"
REMOTE_DGX_BUILD
dgx_build_b64="$(printf '%s' "$dgx_build_program" | base64 | tr -d '\r\n')"

start_dgx_build() {
  local target="$1" destination="$2" local_log="$3" output_variable="$4"
  remote "$target" \
    "exec setsid bash -c \"\$(printf '%s' '$dgx_build_b64' | base64 -d)\" d0-dgx-build '$destination' '$model_revision' '$checkpoint_revision' '$model_weight_inventory_sha256' '$checkpoint_content_manifest_sha256'" \
    >"$local_log" 2>&1 &
  printf -v "$output_variable" '%s' "$!"
}
start_dgx_build "$dgx_a_target" "$dgx_a_root" \
  "$result_dir/logs/dgx_a_build_ssh.log" dgx_a_ssh_pid
start_dgx_build "$dgx_b_target" "$dgx_b_root" \
  "$result_dir/logs/dgx_b_build_ssh.log" dgx_b_ssh_pid

# x86 preparation is deliberately one serial program: dataset audit -> static
# maps -> worker-container preparation -> stopped-container cleanup.  It does
# not invoke either lane runtime or any evaluator.
read -r -d '' x86_prepare_program <<'REMOTE_X86_PREPARE' || true
set -euo pipefail
deployment_root="$1"
dataset_root="$2"
expected_dataset_sha="$3"
lane_a_deployment_root="$4"
lane_b_deployment_root="$5"
audit="$deployment_root/results/d0_prepare"
mkdir -p "$audit" "$deployment_root/inputs"
started="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
pgid="$(ps -o pgid= -p $$ | tr -d ' ')"
containers_owned="$audit/containers_owned"
write_ledger() {
  local state="$1" rc="$2"
  {
    printf 'state=%s\n' "$state"
    printf 'pid=%s\n' "$$"
    printf 'pgid=%s\n' "$pgid"
    printf 'started_at=%s\n' "$started"
    printf 'ended_at=%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    printf 'exit_code=%s\n' "$rc"
    printf 'command=dataset_audit_then_static_maps_then_worker_prepare\n'
  } >"$audit/prepare_ledger.txt"
}
stop_workers() {
  local container
  for container in internnav_t5_isaac_a internnav_t5_isaac_b; do
    test ! -f "$containers_owned/$container" || \
      docker stop -t 10 "$container" >/dev/null 2>&1 || true
  done
}
finish() {
  local rc=$?
  trap - EXIT HUP INT TERM
  stop_workers
  if test "$rc" = 0; then write_ledger PASS 0; else write_ledger FAIL "$rc"; fi
  exit "$rc"
}
trap finish EXIT
trap 'exit 130' HUP INT TERM
write_ledger RUNNING -1

capture_capacity() {
  python3 - "$1" <<'PY'
import json, os, shutil, subprocess, sys, time
from pathlib import Path
memory={}
for line in Path('/proc/meminfo').read_text().splitlines():
    key, value = line.split(':', 1)
    if key in {'MemTotal','MemAvailable','SwapTotal','SwapFree'}:
        memory[key + '_kib'] = int(value.split()[0])
disk=shutil.disk_usage(str(Path(sys.argv[1]).parent))
gpu=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.total,memory.free,driver_version',
                    '--format=csv,noheader,nounits'],text=True,capture_output=True,check=True)
compute=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory',
                        '--format=csv,noheader,nounits'],text=True,capture_output=True)
payload={'schema_version':1,'memory':memory,
         'disk':{'total_bytes':disk.total,'used_bytes':disk.used,'free_bytes':disk.free},
         'gpu_csv':gpu.stdout.splitlines(),'compute_process_csv':compute.stdout.splitlines(),
         'cpu_count':os.cpu_count(),'recorded_unix':time.time()}
Path(sys.argv[1]).write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
PY
}
capture_capacity "$audit/capacity_before.json"
dataset="$dataset_root/val_unseen/val_unseen.json.gz"
actual_dataset_sha="$(sha256sum "$dataset" | cut -d' ' -f1)"
test "$actual_dataset_sha" = "$expected_dataset_sha"
python3 - "$deployment_root/configs/internnav_t5/d0_run_manifest.json" \
  "$dataset" "$audit/dataset_audit.json" "$actual_dataset_sha" <<'PY'
import gzip, json, sys
from pathlib import Path
manifest=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
with gzip.open(sys.argv[2], 'rt', encoding='utf-8') as stream:
    payload=json.load(stream)
episodes=payload.get('episodes') if isinstance(payload,dict) else None
if not isinstance(episodes,list): raise SystemExit('dataset has no episode list')
keys=[f"{item['trajectory_id']}_{item['episode_id']}" for item in episodes]
expected=manifest['fixed_input']['episode_keys']
checks={'sha_match':sys.argv[4]==manifest['fixed_input']['dataset_file_sha256'],
        'episode_count_match':len(episodes)==manifest['fixed_input']['episode_count'],
        'episode_keys_match':keys==expected}
result={'schema_version':1,'status':'PASS' if all(checks.values()) else 'FAIL',
        'dataset_file':sys.argv[2],'dataset_sha256':sys.argv[4],
        'episode_count':len(episodes),'episode_keys':keys,'checks':checks}
Path(sys.argv[3]).write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
if result['status'] != 'PASS': raise SystemExit('frozen dataset audit failed')
PY
INTERNNAV_T1_CONTROL_ROOT="$deployment_root" \
INTERNNAV_RUNTIME_POLICY=completion_sim \
INTERNNAV_SIMULATION_TARGET=isaac \
INTERNNAV_T4_RESOURCE_LEASE_ACK=dgx+isaac \
  bash "$deployment_root/scripts/prepare_t4_isaac_static_maps.sh" \
    --dataset-root "$dataset_root" \
    --output-dir "$deployment_root/inputs/d0_fixed5_static_maps" \
    >"$audit/static_map_prepare.log" 2>&1
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
INTERNVLA_T5_CONTAINER_OWNERSHIP_DIR="$containers_owned" \
INTERNNAV_T5_RESOURCE_LEASE_ACK=all-lanes \
INTERNNAV_RUNTIME_POLICY=completion_sim \
INTERNNAV_SIMULATION_TARGET=isaac \
  bash "$deployment_root/scripts/prepare_t5_isaac_workers.sh" \
    "$audit/isaac_worker_prepare.json" \
    >"$audit/isaac_worker_prepare.log" 2>&1
stop_workers
for lane in a b; do
  container="internnav_t5_isaac_$lane"
  docker inspect "$container" >"$audit/container_${lane}_inspect.json"
  python3 "$deployment_root/scripts/validate_t5_isaac_worker_spec.py" \
    "/home/song/internnav-t1-t2/runtime/t5_isaac_workers/$lane/expected_container_spec.json" \
    "$audit/container_${lane}_inspect.json" \
    "$audit/container_${lane}_spec_audit.json"
done
python3 - "$audit/container_cleanup.json" "$audit/container_a_inspect.json" \
  "$audit/container_a_spec_audit.json" "$audit/container_b_inspect.json" \
  "$audit/container_b_spec_audit.json" <<'PY'
import json, sys, time
from pathlib import Path
lanes={}
for lane, inspect_path, audit_path in (
    ('a',sys.argv[2],sys.argv[3]),('b',sys.argv[4],sys.argv[5])):
    value=json.loads(Path(inspect_path).read_text())[0]
    spec=json.loads(Path(audit_path).read_text())
    mounts={item['Destination']:{'source':item['Source'],'rw':item['RW']}
            for item in value.get('Mounts',[])}
    lanes[lane]={'container':value['Name'].lstrip('/'),'container_id':value['Id'],
                 'running':value['State']['Running'],'pid':value['State']['Pid'],
                 'image_id':value['Image'],'mounts':mounts,'spec_audit':spec}
checks={'both_stopped':all(not value['running'] and value['pid']==0 for value in lanes.values()),
        'exact_specs_after_stop':all(value['spec_audit']['status']=='PASS' for value in lanes.values()),
        'gpu_device_requests_match':all(value['spec_audit']['checks']['gpu_device_request'] for value in lanes.values()),
        'stable_control_root_mounted':all('/home/song/internnav-t1-t2' in value['mounts'] for value in lanes.values()),
        'control_root_read_only':all(not value['mounts']['/home/song/internnav-t1-t2']['rw'] for value in lanes.values())}
# The validator's exact mount contract proves the writable-overlay property;
# retain an explicit aggregate check without reconstructing deployment paths.
checks['only_expected_writable_overlays']=all(
    value['spec_audit']['checks']['mounts_exact']
    and value['spec_audit']['checks']['control_root_read_only']
    and value['spec_audit']['checks']['other_lane_deployment_not_writable']
    and value['spec_audit']['checks']['other_lane_profile_not_writable']
    for value in lanes.values())
payload={'schema_version':1,'status':'PASS' if all(checks.values()) else 'FAIL',
         'lanes':lanes,'checks':checks,'recorded_unix':time.time()}
Path(sys.argv[1]).write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
if payload['status'] != 'PASS': raise SystemExit('worker cleanup audit failed')
PY
capture_capacity "$audit/capacity_after.json"
REMOTE_X86_PREPARE
x86_prepare_b64="$(printf '%s' "$x86_prepare_program" | base64 | tr -d '\r\n')"
remote "$x86_target" \
  "exec setsid bash -c \"\$(printf '%s' '$x86_prepare_b64' | base64 -d)\" d0-x86-prepare '$x86_prepare_root' '$dataset_root' '$dataset_sha256' '$x86_a_root' '$x86_b_root'" \
  >"$result_dir/logs/x86_prepare_ssh.log" 2>&1

build_rc=0
set +e
wait "$dgx_a_ssh_pid"; dgx_a_rc=$?
wait "$dgx_b_ssh_pid"; dgx_b_rc=$?
set -e
dgx_a_ssh_pid=""
dgx_b_ssh_pid=""
test "$dgx_a_rc" = 0 || build_rc=1
test "$dgx_b_rc" = 0 || build_rc=1
test "$build_rc" = 0

# The immutable static-map output is copied to both same-ref DGX deployments.
remote "$x86_target" "tar -C '$x86_prepare_root/inputs/d0_fixed5_static_maps' -czf - ." \
  >"$result_dir/maps/d0_fixed5_static_maps.tar.gz"
map_archive_sha256="$(sha256sum "$result_dir/maps/d0_fixed5_static_maps.tar.gz" | cut -d' ' -f1)"
copy_map() {
  local target="$1" destination="$2"
  remote "$target" "install -d -m 700 '$destination/inputs/d0_fixed5_static_maps'"
  gzip -dc "$result_dir/maps/d0_fixed5_static_maps.tar.gz" | \
    ssh "${ssh_options[@]}" "$target" "tar -C '$destination/inputs/d0_fixed5_static_maps' -xf -"
  remote "$target" \
    "sha256sum '$destination/inputs/d0_fixed5_static_maps/manifest.json' | cut -d' ' -f1"
}
map_a_manifest_sha="$(copy_map "$dgx_a_target" "$dgx_a_root")"
map_b_manifest_sha="$(copy_map "$dgx_b_target" "$dgx_b_root")"
map_x86_a_manifest_sha="$(copy_map "$x86_target" "$x86_a_root")"
map_x86_b_manifest_sha="$(copy_map "$x86_target" "$x86_b_root")"
map_x86_manifest_sha="$(remote "$x86_target" "sha256sum '$x86_prepare_root/inputs/d0_fixed5_static_maps/manifest.json' | cut -d' ' -f1")"
test "$map_a_manifest_sha" = "$map_x86_manifest_sha"
test "$map_b_manifest_sha" = "$map_x86_manifest_sha"
test "$map_x86_a_manifest_sha" = "$map_x86_manifest_sha"
test "$map_x86_b_manifest_sha" = "$map_x86_manifest_sha"

read -r -d '' residual_audit_program <<'REMOTE_RESIDUAL_AUDIT' || true
import json, os, subprocess, sys, time
from pathlib import Path
output, deployment_root, role = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
ports={25137,25138,25139,25140,25141,25239,25240,25241}
ancestors=set()
pid=os.getpid()
while pid > 1 and pid not in ancestors:
    ancestors.add(pid)
    try:
        status=Path(f'/proc/{pid}/status').read_text()
        pid=int(next(line.split()[1] for line in status.splitlines() if line.startswith('PPid:')))
    except Exception:
        break
processes=[]
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit() or int(entry.name) in ancestors: continue
    try: command=(entry/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace').strip()
    except (FileNotFoundError,PermissionError): continue
    if deployment_root in command: processes.append({'pid':int(entry.name),'command':command})
socket_result=subprocess.run(['ss','-H','-lntup'],text=True,capture_output=True)
sockets=[]
for line in socket_result.stdout.splitlines():
    if any(f':{port} ' in line or f':{port}\n' in line for port in ports): sockets.append(line)
containers={}
if role == 'x86':
    for lane,name in (('a','internnav_t5_isaac_a'),('b','internnav_t5_isaac_b')):
        value=json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0]
        containers[lane]={'name':name,'running':value['State']['Running'],'pid':value['State']['Pid']}
checks={'deployment_process_count_zero':not processes,'t5_socket_count_zero':not sockets,
        'containers_stopped':role!='x86' or all(not v['running'] and v['pid']==0 for v in containers.values())}
payload={'schema_version':1,'status':'PASS' if all(checks.values()) else 'FAIL','role':role,
         'deployment_root':deployment_root,'processes':processes,'sockets':sockets,
         'containers':containers,'checks':checks,'recorded_unix':time.time()}
output.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
if payload['status'] != 'PASS': raise SystemExit('residual audit failed')
REMOTE_RESIDUAL_AUDIT
residual_audit_b64="$(printf '%s' "$residual_audit_program" | base64 | tr -d '\r\n')"
for specification in \
  "$dgx_a_target|$dgx_a_root|dgx_a" \
  "$dgx_b_target|$dgx_b_root|dgx_b" \
  "$x86_target|$x86_prepare_root|x86" \
  "$x86_target|$x86_a_root|x86_lane" \
  "$x86_target|$x86_b_root|x86_lane"; do
  IFS='|' read -r target destination role <<<"$specification"
  remote "$target" "install -d -m 700 '$destination/results/d0_prepare'"
  remote "$target" \
    "python3 -c \"\$(printf '%s' '$residual_audit_b64' | base64 -d)\" '$destination/results/d0_prepare/residual_audit.json' '$destination' '$role'"
done

collect_audit() {
  local target="$1" source="$2" destination="$3"
  mkdir "$destination"
  remote "$target" "tar -C '$source/results/d0_prepare' -czf - ." \
    >"$destination.tar.gz"
  tar -C "$destination" -xzf "$destination.tar.gz"
}
collect_audit "$dgx_a_target" "$dgx_a_root" "$result_dir/remote/dgx_a"
collect_audit "$dgx_b_target" "$dgx_b_root" "$result_dir/remote/dgx_b"
collect_audit "$x86_target" "$x86_prepare_root" "$result_dir/remote/x86"
collect_audit "$x86_target" "$x86_a_root" "$result_dir/remote/x86_a"
collect_audit "$x86_target" "$x86_b_root" "$result_dir/remote/x86_b"

python3 - "$result_dir/d0_prepare_summary.json" "$authorization_ref" \
  "$grant_id" "$golden_bundle_id" "$golden_sha256" "$run_manifest_sha256" \
  "$archive_sha256" "$map_archive_sha256" "$map_x86_manifest_sha" \
  "$dgx_a_root" "$dgx_b_root" "$x86_prepare_root" "$x86_a_root" \
  "$x86_b_root" <<'PY'
import json, sys, time
from pathlib import Path
output=Path(sys.argv[1]); result=output.parent
def load(relative): return json.loads((result/relative).read_text(encoding='utf-8'))
a_inventory=load('remote/dgx_a/model_inventory.json')
b_inventory=load('remote/dgx_b/model_inventory.json')
a_runtime=load('remote/dgx_a/model_runtime_environment.json')
b_runtime=load('remote/dgx_b/model_runtime_environment.json')
dataset=load('remote/x86/dataset_audit.json')
scene_geometry=load('remote/x86/static_map_source_audit.json')
workers=load('remote/x86/isaac_worker_prepare.json')
container_cleanup=load('remote/x86/container_cleanup.json')
grant_validation=load('grant_validation.json')
audits={role:load(f'remote/{role}/residual_audit.json')
        for role in ('dgx_a','dgx_b','x86','x86_a','x86_b')}
ledgers={role:(result/f'remote/{role}'/('prepare_ledger.txt' if role=='x86' else 'build_ledger.txt')).read_text()
         for role in ('dgx_a','dgx_b','x86')}
x86_available_bytes=load('remote/x86/capacity_after.json')['memory']['MemAvailable_kib'] * 1024
minimum_available_bytes=grant_validation['dual_isaac_minimum_available_memory_bytes']
expected_checkpoint_revision=grant_validation['checkpoint_revision']
expected_checkpoint_manifest=grant_validation['checkpoint_content_manifest_canonical_sha256']
expected_weight_inventory=grant_validation['model_weight_inventory_sha256']
def inventory_matches(value):
    return (
        value.get('status') == 'PASS'
        and value.get('checkpoint_revision') == expected_checkpoint_revision
        and value.get('checkpoint_content_manifest_canonical_sha256')
            == expected_checkpoint_manifest
        and value.get('expected_runtime_weight_inventory_sha256')
            == expected_weight_inventory
        and bool(value.get('checks'))
        and all(value['checks'].values())
    )
checks={
  'same_model_revision':a_inventory['model_revision']==b_inventory['model_revision'],
  'lane_a_model_runtime_environment':a_runtime.get('status')=='PASS'
      and bool(a_runtime.get('checks')) and all(a_runtime['checks'].values()),
  'lane_b_model_runtime_environment':b_runtime.get('status')=='PASS'
      and bool(b_runtime.get('checks')) and all(b_runtime['checks'].values()),
  'lane_a_checkpoint_content':inventory_matches(a_inventory),
  'lane_b_checkpoint_content':inventory_matches(b_inventory),
  'same_checkpoint_content_manifest':a_inventory.get('checkpoint_content_manifest_canonical_sha256')
      == b_inventory.get('checkpoint_content_manifest_canonical_sha256'),
  'same_expected_runtime_weight_inventory':a_inventory.get('expected_runtime_weight_inventory_sha256')
      == b_inventory.get('expected_runtime_weight_inventory_sha256'),
  'models_not_loaded':not a_inventory['model_loaded'] and not b_inventory['model_loaded'],
  'dataset_audit':dataset['status']=='PASS',
  'frozen_scene_geometry':scene_geometry['status']=='PASS',
  'worker_isolation_prepared':workers['status']=='PASS',
  'worker_containers_stopped':container_cleanup['status']=='PASS',
  'remote_steps_passed':all('state=PASS' in ledger for ledger in ledgers.values()),
  'zero_remote_residuals':all(value['status']=='PASS' for value in audits.values()),
  'x86_memory_admission':x86_available_bytes >= minimum_available_bytes,
}
payload={
  'schema_version':1,'status':'PASS' if all(checks.values()) else 'FAIL',
  'stage':'d0_0_online_prepare','grant_id':sys.argv[3],
  'authorization_mode':'FORMAL_BOARD_GRANT_V2','board_grant_used':True,
  'authorization_ref_sha':sys.argv[2],'code_ref_sha':sys.argv[2],
  'golden_bundle_id':sys.argv[4],
  'golden_bundle_canonical_sha256':sys.argv[5],
  'run_manifest_canonical_sha256':sys.argv[6],
  'deployment_archive_sha256':sys.argv[7],
  'static_map_archive_sha256':sys.argv[8],
  'static_map_manifest_sha256':sys.argv[9],
  'deployment_roots':{'dgx_a':sys.argv[10],'dgx_b':sys.argv[11],
                      'x86_prepare':sys.argv[12],'x86_a':sys.argv[13],
                      'x86_b':sys.argv[14]},
  'checks':checks,
  'capacity':{
    role:{'before':load(f'remote/{role}/capacity_before.json'),
          'after':load(f'remote/{role}/capacity_after.json')}
    for role in ('dgx_a','dgx_b','x86')
  },
  'memory_admission':{
    'available_bytes':x86_available_bytes,
    'minimum_available_bytes':minimum_available_bytes,
    'status':'PASS' if x86_available_bytes >= minimum_available_bytes else 'FAIL',
    'on_failure':'PRESERVE_DEPLOYMENTS_AND_RESUME_PLAN_NOTIFY_USER_NO_AUTO_REBOOT',
    'dual_runtime_recheck_required':True,
  },
  'pid_ledgers':ledgers,
  'containers':container_cleanup['lanes'],
  'locks':{'profile':'all-lanes','state_at_summary':'HELD_BY_OUTER_FAIL_CLOSED_LEASE',
           'release_evidence':'lease_release_summary.json'},
  'execution':{'dgx_ros_builds_concurrent':True,
               'x86_dataset_maps_and_container_prepare_serial':True,
               'internvla_model_loaded':False,'isaac_sim_started':False,
               'episode_or_evaluator_started':False},
  'recorded_unix':time.time(),
}
output.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8')
if payload['status'] != 'PASS': raise SystemExit('D0.0 preparation failed')
PY

trap - EXIT INT TERM HUP
cleanup_inside
