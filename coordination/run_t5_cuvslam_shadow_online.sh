#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: run_t5_cuvslam_shadow_online.sh RUN_ID CODE_SHA PREP_RESULT_ROOT RESULT_ROOT [CANDIDATE_PROFILE]

T5-only Lane-A strict-extension entrypoint. It reuses the exact fast-lane
Lane-A + Isaac-GPU0/even-CPU leases, executes the deterministic screen3 set,
keeps GT /odom as the sole navigation authority, and accepts cuVSLAM only when
the collected shadow evidence covers at least 120 simulated seconds and two
episode resets. It does not perform odometry takeover.
EOF
  exit 64
}

[[ $# -ge 4 && $# -le 5 ]] || usage
run_id="$1"
code_sha="$2"
prep_relative="$3"
result_relative="$4"
candidate_profile="${5:-baseline}"
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"

[[ "$run_id" =~ ^[a-z0-9][a-z0-9._-]{7,95}$ ]] || usage
[[ "$code_sha" =~ ^[0-9a-f]{40}$ ]] || usage
expected_result="results/internnav_t5/fast-lane-a-screen3-${run_id}"
test "$result_relative" = "$expected_result" || {
  echo "RESULT_ROOT must equal $expected_result" >&2
  exit 64
}
test ! -e "$root/$result_relative"

set +e
INTERNNAV_T5_CANDIDATE_PROFILE="$candidate_profile" \
INTERNNAV_T5_STRICT_EXTENSION_PROFILE=cuvslam_shadow \
  bash "$root/coordination/run_t5_fast_lane_online.sh" \
    a screen3 "$run_id" "$code_sha" "$prep_relative" "$result_relative"
runner_rc=$?
set -e

metrics_rc=2
if test -d "$root/$result_relative"; then
  set +e
  python3 "$root/scripts/analyze_t5_cuvslam_shadow.py" \
    --result-root "$root/$result_relative" \
    --output "$root/$result_relative/cuvslam_shadow_metrics.json"
  metrics_rc=$?
  set -e
  set +e
  python3 - "$root/$result_relative/cuvslam_shadow_final_summary.json" \
    "$root/$result_relative/fast_lane_final_summary.json" \
    "$root/$result_relative/cuvslam_shadow_metrics.json" \
    "$runner_rc" "$metrics_rc" "$run_id" "$code_sha" \
    "$candidate_profile" <<'PY'
import json, sys, time
from pathlib import Path

output, fast_path, metrics_path = map(Path, sys.argv[1:4])

def load(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None

runner_rc, metrics_rc = int(sys.argv[4]), int(sys.argv[5])
fast, metrics = load(fast_path), load(metrics_path)
checks = {
    "fast_lane_screen3_pass": runner_rc == 0
        and isinstance(fast, dict) and fast.get("status") == "PASS",
    "component_smoke_pass": metrics_rc == 0
        and isinstance(metrics, dict)
        and metrics.get("status") == "COMPONENT_SMOKE_PASS",
    "shadow_only_not_nav_takeover": isinstance(metrics, dict)
        and metrics.get("nav_candidate_pass") is False
        and metrics.get("navigation_pose_authority") == "ground_truth",
}
payload = {
    "schema_version": 1,
    "status": "PASS" if all(checks.values()) else "FAIL",
    "run_id": sys.argv[6],
    "code_ref_sha": sys.argv[7],
    "lane": "a",
    "fast_profile": "screen3",
    "strict_extension_profile": "cuvslam_shadow",
    "candidate_profile": sys.argv[8],
    "resource_profile": "lane-a",
    "isaac_gpu": 0,
    "isaac_cpuset": "0,2,4,6,8,10,12,14,16",
    "checks": checks,
    "fast_lane_final_summary": "fast_lane_final_summary.json",
    "component_metrics": "cuvslam_shadow_metrics.json",
    "online_navigation_acceptance": "NOT_EVALUATED",
    "nav_candidate_pass": False,
    "recorded_unix": time.time(),
}
output.write_text(
    json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    encoding="utf-8",
)
raise SystemExit(0 if payload["status"] == "PASS" else 2)
PY
  final_rc=$?
  set -e
else
  final_rc=2
fi

test "$runner_rc" = 0 && test "$metrics_rc" = 0 && test "$final_rc" = 0
