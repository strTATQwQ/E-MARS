#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: run_t5_fault_injection.sh LANE RUN_ID CODE_SHA PREP_RESULT_ROOT RESULT_ROOT [CANDIDATE_PROFILE]

Run the exact six-event completion_sim profile during one 600-sim-second soak.
The lane fast runner owns the selected DGX and Isaac GPU leases; this wrapper
adds no lock or quarantine layer.
EOF
  exit 64
}

[[ $# -ge 5 && $# -le 6 ]] || usage
lane="$1"
run_id="$2"
code_sha="$3"
prep_relative="$4"
result_relative="$5"
candidate_profile="${6:-baseline}"
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"

case "$lane" in a|b) ;; *) usage ;; esac
case "$candidate_profile" in baseline|recovery_a) ;; a0|a1|a0+b0|a0+b1|a1+b0|a1+b1|a1+b0+c0|a1+b0+c1|a1+b0+c2|a1+b1+c0|a1+b1+c1|a1+b1+c2) ;; *) usage ;; esac
[[ "$run_id" =~ ^[a-z0-9][a-z0-9._-]{7,95}$ ]] || usage
[[ "$code_sha" =~ ^[0-9a-f]{40}$ ]] || usage
test "$result_relative" = \
  "results/internnav_t5/fast-lane-${lane}-soak600-${run_id}" || usage

python3 "$root/scripts/run_t5_fault_injection.py" validate \
  --config "$root/configs/internnav_t5/fault_injection_minimal_v1.json" \
  >/dev/null

set +e
INTERNNAV_T5_CANDIDATE_PROFILE="$candidate_profile" \
INTERNNAV_T5_FAULT_INJECTION_PROFILE=completion_sim_minimal_v1 \
  bash "$root/coordination/run_t5_fast_lane_online.sh" \
    "$lane" soak600 "$run_id" "$code_sha" \
    "$prep_relative" "$result_relative"
runner_rc=$?
set -e

result_dir="$root/$result_relative"
test -d "$result_dir" || exit "$runner_rc"

set +e
python3 - "$result_dir/t5_fault_injection_run_summary.json" "$lane" \
  "$run_id" "$code_sha" "$candidate_profile" "$runner_rc" <<'PY'
import json, sys, time
from pathlib import Path

output = Path(sys.argv[1])
result = output.parent
lane, run_id, code_sha, candidate_profile = sys.argv[2:6]
runner_rc = int(sys.argv[6])


def load(relative):
    try:
        return json.loads((result / relative).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


runtime = load("fast_lane_summary.json")
faults = load("fault_injection/fault_injection_summary.json")
engineering = load("remote/x86/engineering_canary.json")
dgx = load("remote/dgx/lane_status.json")
x86 = load("remote/x86/isaac_status.json")
cleanup = load("audits/coordinator_cleanup_receipt.json")
lease = load("lease_release_summary.json")
final = load("fast_lane_final_summary.json")
checks = {
    "runner_exit_zero": runner_rc == 0,
    "fast_runtime_pass": isinstance(runtime, dict)
    and runtime.get("status") == "PASS"
    and runtime.get("lane") == lane
    and runtime.get("profile") == "soak600"
    and runtime.get("run_id") == run_id
    and runtime.get("code_ref_sha") == code_sha
    and runtime.get("candidate_profile") == candidate_profile
    and runtime.get("fault_injection_profile")
    == "completion_sim_minimal_v1",
    "six_faults_recovered": isinstance(faults, dict)
    and faults.get("status") == "PASS"
    and faults.get("profile") == "completion_sim_minimal_v1"
    and len(faults.get("event_results", [])) == 6
    and bool(faults.get("checks"))
    and all(faults["checks"].values()),
    "soak_uses_600_sim_seconds": isinstance(engineering, dict)
    and engineering.get("status") == "PASS"
    and engineering.get("configured_seconds") == 600
    and engineering.get("duration_timebase") == "sim"
    and engineering.get("observed_sim_seconds", 0.0) >= 600.0,
    "zero_runtime_residual": isinstance(dgx, dict)
    and dgx.get("status") == "PASS"
    and dgx.get("residual_count") == 0
    and isinstance(x86, dict)
    and x86.get("status") == "PASS"
    and x86.get("residual_count") == 0
    and x86.get("socket_residual_count") == 0,
    "coordinator_cleanup_pass": isinstance(cleanup, dict)
    and cleanup.get("status") == "PASS"
    and bool(cleanup.get("checks"))
    and all(cleanup["checks"].values()),
    "lease_release_pass": isinstance(lease, dict)
    and lease.get("status") == "PASS"
    and bool(lease.get("checks"))
    and all(lease["checks"].values()),
    "final_summary_pass": isinstance(final, dict)
    and final.get("status") == "PASS"
    and bool(final.get("checks"))
    and all(final["checks"].values()),
}
payload = {
    "schema_version": 1,
    "status": "PASS" if all(checks.values()) else "FAIL",
    "profile": "completion_sim_minimal_v1",
    "lane": lane,
    "run_id": run_id,
    "code_ref_sha": code_sha,
    "candidate_profile": candidate_profile,
    "configured_seconds": 600,
    "duration_timebase": "sim",
    "wall_time_scope": "actuator_and_process_liveness_only",
    "real_go2_eligible": False,
    "checks": checks,
    "recorded_unix": time.time(),
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
raise SystemExit(0 if payload["status"] == "PASS" else 75)
PY
summary_rc=$?
set -e

test "$runner_rc" = 0 || exit "$runner_rc"
exit "$summary_rc"
