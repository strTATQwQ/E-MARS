#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: recover_t5_stale_dgx_nav2.sh dgx-a OLD_DEPLOYMENT_ROOT EVIDENCE_FILE EVIDENCE_SHA256 RESULT_DIR

Recovery-only DGX_A entry for the three stale Lane-A Nav2 session leaders.
EVIDENCE_FILE, its SHA, OLD_DEPLOYMENT_ROOT, conflict evidence, and all three
leader identities must equal the immutable attempt-10 recovery manifest in
this code ref. The command archives that manifest as a read-only file,
acquires recover-dgx-a, atomically arms the DGX quarantine to the exact old
deployment root, and signals no process until live identity, namespace,
PID=PGID=SID, starttime, and argv all match it.
EOF
  exit 64
}

if [[ "${1:-}" == --inside ]]; then
  [[ $# -eq 6 ]] || usage
  inside=1
  role="$2"
  old_root="$3"
  evidence_file="$4"
  evidence_sha256="$5"
  result_dir="$6"
else
  [[ $# -eq 5 ]] || usage
  inside=0
  role="$1"
  old_root="$2"
  evidence_file="$3"
  evidence_sha256="$4"
  result_dir="$5"
fi

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
fixed_old_root=/home/railgun/internnav-t1-t2/.t5-deployments/t5d0020260719t135307-78cc4b1e7b1b-lane-a
fixed_evidence_sha256=60e193f5ce3ca1060d72c65de74505b2ba5790cc10aa12386d0dfb8938232b85
fixed_evidence_file="$root/configs/internnav_t5/attempt10_stale_nav2_recovery_manifest.json"
test "$role" = dgx-a
case "$old_root" in
  /home/railgun/internnav-t1-t2/.t5-deployments/*) ;;
  *) echo "OLD_DEPLOYMENT_ROOT must be an absolute DGX_A deployment root" >&2; exit 64 ;;
esac
[[ "$old_root" != *$'\n'* && "$old_root" != *$'\r'* && "$old_root" != *"|"* ]]
[[ "$evidence_sha256" =~ ^[0-9A-Fa-f]{64}$ ]]
evidence_sha256="${evidence_sha256,,}"
test "$old_root" = "$fixed_old_root"
test "$evidence_sha256" = "$fixed_evidence_sha256"
if (( inside == 0 )); then
  test "$evidence_file" = "$fixed_evidence_file"
fi
[[ "$result_dir" = /* ]]
test "$(dirname -- "$result_dir")" = "$root/results/internnav_t5"
result_name="$(basename -- "$result_dir")"
[[ "$result_name" =~ ^stale-dgx-recovery-[A-Za-z0-9][A-Za-z0-9._-]*$ ]]
(( ${#result_name} <= 200 ))

engine="$root/scripts/t5_stale_nav2_recovery.py"
auditor="$root/scripts/t5_process_identity_audit.py"
test -f "$engine"
test -f "$auditor"

if (( inside == 0 )); then
  [[ "$evidence_file" = /* ]]
  test -f "$evidence_file"
  test ! -L "$evidence_file"
  test ! -e "$result_dir"
  test ! -L "$result_dir"
  umask 077
  mkdir -p "$result_dir/evidence" "$result_dir/audits" "$result_dir/lease"
  archived_evidence="$result_dir/evidence/stale_nav2_evidence.json"
  python3 "$engine" stage-evidence "$evidence_file" "$archived_evidence" \
    "$evidence_sha256" "$old_root" \
    >"$result_dir/evidence/stage_receipt.json"
  exec bash "$root/scripts/with_resource_lease.sh" recover-dgx-a \
    --owner codex-00 --task "t5-stale-nav2-recovery:dgx-a" \
    --log-dir "$result_dir/lease" --acquire-timeout 30 \
    --cleanup-timeout 120 --kill-wait-timeout 30 -- \
    bash "$root/scripts/recover_t5_stale_dgx_nav2.sh" --inside dgx-a \
      "$old_root" "$archived_evidence" "$evidence_sha256" "$result_dir"
fi

test "${INTERNNAV_T5_RESOURCE_LEASE_ACK:-}" = recover-dgx-a
test "$evidence_file" = "$result_dir/evidence/stale_nav2_evidence.json"
test -d "$result_dir/lease"
test -d "$result_dir/audits"
test -f "$result_dir/evidence/stage_receipt.json"
python3 "$engine" verify-evidence "$evidence_file" "$evidence_sha256" \
  "$old_root" >"$result_dir/evidence/pre_recovery_verify.json"

source "$root/scripts/t5_quarantine_common.sh"
ssh_options=(-T -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=2)
remote() { local target="$1"; shift; ssh "${ssh_options[@]}" "$target" "$@"; }

target=railgun@10.100.100.128
marker=/tmp/internnav_dgx.quarantine
marker_role=dgx_a
run_tag="stale_nav2:${result_name}"
resume_run_tag=stale_nav2:stale-dgx-recovery-attempt10-c3f2d1f
marker_run_tag="$run_tag"
arm_receipt="$result_dir/audits/dgx_a_quarantine_arm.json"
recovery_receipt="$result_dir/audits/stale_nav2_cleanup_receipt.json"
clear_receipt="$result_dir/audits/dgx_a_quarantine_clear.json"
post_evidence_receipt="$result_dir/evidence/pre_clear_verify.json"

arm_rc=75
remote_rc=75
receipt_rc=75
evidence_rc=75
clear_rc=75
marker_state="$(remote "$target" \
  "if test -e '$marker'; then printf PRESENT; else printf ABSENT; fi" \
  2>/dev/null)" || marker_state=ERROR
if test "$marker_state" = PRESENT; then
  marker_run_tag="$resume_run_tag"
  arm_receipt="$result_dir/audits/dgx_a_quarantine_observe.json"
  if t5_quarantine_owned_observe "$target" "$marker" "$marker_role" \
      "$marker_run_tag" "$old_root" "$arm_receipt"; then
    arm_rc=0
  fi
elif test "$marker_state" = ABSENT && \
    t5_quarantine_arm "$target" "$marker" "$marker_role" "$run_tag" \
      "$old_root" "$arm_receipt"; then
  arm_rc=0
fi

if (( arm_rc == 0 )); then
  engine_b64="$(base64 <"$engine" | tr -d '\r\n')"
  auditor_b64="$(base64 <"$auditor" | tr -d '\r\n')"
  old_root_b64="$(printf '%s' "$old_root" | base64 | tr -d '\r\n')"
  evidence_b64="$(base64 <"$evidence_file" | tr -d '\r\n')"
  set +e
  remote "$target" \
    "python3 -c \"\$(printf '%s' '$engine_b64' | base64 -d)\" recover '$old_root_b64' '$evidence_b64' '$evidence_sha256' '$auditor_b64' 20 10" \
    >"$recovery_receipt"
  remote_rc=$?
  set -e
  if python3 - "$recovery_receipt" "$evidence_file" "$old_root" <<'PY'
import json, sys
from pathlib import Path

receipt=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
evidence=json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
expected_pgids={row["pgid"] for row in evidence["leaders"]}
checks=receipt.get("checks")
signals=receipt.get("signals")
term_pgids={row.get("pgid") for row in signals or [] if row.get("signal")=="SIGTERM"}
kill_pgids={row.get("pgid") for row in signals or [] if row.get("signal")=="SIGKILL"}
valid=(receipt.get("status")=="PASS"
       and receipt.get("action")=="RECOVER_STALE_DGX_A_NAV2"
       and receipt.get("role")=="dgx-a"
       and receipt.get("deployment_root")==sys.argv[3]
       and receipt.get("lane_namespace")=="/t5/lane_a"
       and receipt.get("workload_started") is False
       and isinstance(checks,dict) and checks
       and all(value is True for value in checks.values())
       and isinstance(signals,list)
       and term_pgids==expected_pgids
       and kill_pgids.issubset(expected_pgids))
raise SystemExit(0 if valid else 75)
PY
  then
    receipt_rc=0
  fi
  if python3 "$engine" verify-evidence "$evidence_file" "$evidence_sha256" \
      "$old_root" >"$post_evidence_receipt"; then
    evidence_rc=0
  fi
  if (( remote_rc == 0 && receipt_rc == 0 && evidence_rc == 0 )); then
    if t5_quarantine_owned_clear "$target" "$marker" "$marker_role" \
        "$marker_run_tag" "$old_root" "$recovery_receipt" "$clear_receipt"; then
      clear_rc=0
    fi
  fi
fi

python3 - "$result_dir/recovery_summary.json" "$role" "$old_root" \
  "$evidence_sha256" "$arm_rc" "$remote_rc" "$receipt_rc" "$evidence_rc" \
  "$clear_rc" "$result_dir/evidence/stage_receipt.json" "$arm_receipt" \
  "$recovery_receipt" "$post_evidence_receipt" "$clear_receipt" <<'PY'
import json, sys, time
from pathlib import Path

output=Path(sys.argv[1])
names=("stage_evidence","quarantine_arm","remote_cleanup","evidence_pre_clear","owned_clear")
paths=[Path(value) for value in sys.argv[10:]]
documents={}
for name,path in zip(names,paths):
    try:
        documents[name]=json.loads(path.read_text(encoding="utf-8"))
    except (OSError,UnicodeError,json.JSONDecodeError):
        documents[name]=None
checks={
    "role_is_dgx_a":sys.argv[2]=="dgx-a",
    "old_root_exact":sys.argv[3].startswith("/home/railgun/internnav-t1-t2/.t5-deployments/"),
    "evidence_sha_bound":len(sys.argv[4])==64,
    "evidence_staged":isinstance(documents["stage_evidence"],dict) and documents["stage_evidence"].get("status")=="PASS",
    "quarantine_armed_before_cleanup":int(sys.argv[5])==0 and isinstance(documents["quarantine_arm"],dict) and documents["quarantine_arm"].get("status")=="PASS",
    "remote_receipt_all_true":int(sys.argv[6])==0 and int(sys.argv[7])==0,
    "evidence_immutable_before_clear":int(sys.argv[8])==0 and isinstance(documents["evidence_pre_clear"],dict) and documents["evidence_pre_clear"].get("status")=="PASS",
    "owned_clear_after_pass_receipt":int(sys.argv[9])==0 and isinstance(documents["owned_clear"],dict) and documents["owned_clear"].get("status")=="PASS",
}
payload={"schema_version":1,"status":"PASS" if all(checks.values()) else "FAIL",
 "action":"RECOVER_STALE_DGX_A_NAV2","role":sys.argv[2],
 "deployment_root":sys.argv[3],"evidence_sha256":sys.argv[4],
 "recovery_only":True,"arbitrary_workload_allowed":False,"checks":checks,
 "artifacts":{name:str(path) for name,path in zip(names,paths)},
 "recorded_unix":time.time()}
temporary=output.with_name(f".{output.name}.tmp")
temporary.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
temporary.replace(output)
raise SystemExit(0 if payload["status"]=="PASS" else 75)
PY
