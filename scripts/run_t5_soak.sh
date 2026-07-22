#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: run_t5_soak.sh LANE RUN_ID CODE_SHA PREP_RESULT_ROOT RESULT_ROOT [CANDIDATE_PROFILE]

  LANE:              a | b
  CANDIDATE_PROFILE: baseline | recovery_a (default baseline)

Run one passive 600-simulation-second READY-window soak through the lane-scoped T5 fast
runner.  This wrapper adds no fault injection and acquires no resources of its
own; the fast runner holds only the selected Lane's DGX and Isaac GPU leases.
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
case "$candidate_profile" in baseline|recovery_a) ;; *) usage ;; esac
[[ "$run_id" =~ ^[a-z0-9][a-z0-9._-]{7,95}$ ]] || usage
[[ "$code_sha" =~ ^[0-9a-f]{40}$ ]] || usage
expected_result="results/internnav_t5/fast-lane-${lane}-soak600-${run_id}"
test "$result_relative" = "$expected_result" || usage

set +e
INTERNNAV_T5_CANDIDATE_PROFILE="$candidate_profile" \
  bash "$root/coordination/run_t5_fast_lane_online.sh" \
    "$lane" soak600 "$run_id" "$code_sha" \
    "$prep_relative" "$result_relative"
runner_rc=$?
set -e

result_dir="$root/$result_relative"
if test ! -d "$result_dir"; then
  test "$runner_rc" != 0 && exit "$runner_rc"
  exit 75
fi

set +e
python3 - "$result_dir/t5_soak_summary.json" "$lane" "$run_id" \
  "$code_sha" "$candidate_profile" "$runner_rc" <<'PY'
import json
import sys
import time
from pathlib import Path

output = Path(sys.argv[1])
result = output.parent
lane, run_id, code_sha, candidate_profile = sys.argv[2:6]
runner_rc = int(sys.argv[6])


def load(relative):
    path = result / relative
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


runtime = load("fast_lane_summary.json")
cleanup = load("audits/coordinator_cleanup_receipt.json")
lease = load("lease_release_summary.json")
final = load("fast_lane_final_summary.json")
engineering = load("remote/x86/engineering_canary.json")
x86_status = load("remote/x86/isaac_status.json")
dgx_status = load("remote/dgx/lane_status.json")

checks = {
    "runner_exit_zero": runner_rc == 0,
    "runtime_pass": isinstance(runtime, dict)
    and runtime.get("status") == "PASS"
    and runtime.get("lane") == lane
    and runtime.get("profile") == "soak600"
    and runtime.get("run_id") == run_id
    and runtime.get("code_ref_sha") == code_sha
    and runtime.get("candidate_profile") == candidate_profile,
    "ready_window_600_pass": isinstance(engineering, dict)
    and engineering.get("status") == "PASS"
    and engineering.get("configured_seconds") == 600
    and engineering.get("duration_timebase") == "sim"
    and engineering.get("observed_seconds", 0) >= 600
    and engineering.get("observed_sim_seconds", 0.0) >= 600.0
    and engineering.get("episode_acceptance_claimed") is False
    and engineering.get("evaluation_completed_naturally") is False,
    "runtime_cleanup_pass": isinstance(cleanup, dict)
    and cleanup.get("status") == "PASS"
    and bool(cleanup.get("checks"))
    and all(cleanup["checks"].values())
    and isinstance(dgx_status, dict)
    and dgx_status.get("status") == "PASS"
    and dgx_status.get("residual_count") == 0
    and isinstance(x86_status, dict)
    and x86_status.get("status") == "PASS"
    and x86_status.get("residual_count") == 0
    and x86_status.get("socket_residual_count") == 0,
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
    "profile": "soak600",
    "lane": lane,
    "run_id": run_id,
    "code_ref_sha": code_sha,
    "candidate_profile": candidate_profile,
    "configured_seconds": 600,
    "duration_timebase": "sim",
    "observed_sim_seconds": (
        engineering.get("observed_sim_seconds")
        if isinstance(engineering, dict)
        else None
    ),
    "episode_acceptance_claimed": False,
    "fault_injection": "not_requested",
    "checks": checks,
    "runtime_summary_path": "fast_lane_summary.json",
    "cleanup_receipt_path": "audits/coordinator_cleanup_receipt.json",
    "lease_release_summary_path": "lease_release_summary.json",
    "recorded_unix": time.time(),
}
output.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
raise SystemExit(0 if payload["status"] == "PASS" else 75)
PY
summary_rc=$?
set -e

if test "$runner_rc" != 0; then
  exit "$runner_rc"
fi
exit "$summary_rc"
