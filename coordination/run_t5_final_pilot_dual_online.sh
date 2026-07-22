#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: run_t5_final_pilot_dual_online.sh RUN_ID CODE_SHA PREP_RESULT_ROOT RESULT_ROOT

Launch the frozen disjoint A10 and B10 through the ordinary per-Lane fast
runner.  Each child acquires only DGX_A+GPU0 or DGX_B+GPU1.  No all-lanes grant
is used.  The default is parallel; INTERNNAV_T5_FINAL_PILOT_LAUNCH_MODE=serial
is the explicit simulator-contention fallback.
EOF
  exit 64
}

[[ $# -eq 4 ]] || usage
run_id="$1"
code_sha="$2"
prepare_relative="$3"
result_relative="$4"
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
launch_mode="${INTERNNAV_T5_FINAL_PILOT_LAUNCH_MODE:-parallel}"
[[ "$run_id" =~ ^[a-z0-9][a-z0-9._-]{7,75}$ ]] || usage
[[ "$code_sha" =~ ^[0-9a-f]{40}$ ]] || usage
case "$launch_mode" in parallel|serial) ;; *) usage ;; esac
[[ "$prepare_relative" =~ ^results/internnav_t5/final-pilot-prepare-[a-z0-9][a-z0-9._-]{7,95}$ ]] || usage
expected_result="results/internnav_t5/final-pilot-dual-${run_id}"
test "$result_relative" = "$expected_result" || {
  echo "RESULT_ROOT must equal $expected_result" >&2
  exit 64
}

prepare_root="$root/$prepare_relative"
result_root="$root/$result_relative"
receipt="$prepare_root/final_pilot_prepare_receipt.json"
test -f "$receipt"
test ! -L "$receipt"
python3 - "$receipt" "$code_sha" <<'PY'
import json,sys
value=json.load(open(sys.argv[1],encoding="utf-8"))
assert value.get("status")=="PASS"
assert value.get("code_ref_sha")==sys.argv[2]
assert value.get("final_profile")=={
 "candidate_profile":"a1+b1+c1","rtf_ablation_profile":"navigation_fast",
 "isaac_sensor_profile":"baseline","strict_extension_profile":"off",
 "nvblox_mode":"off","run_mode":"model"}
assert bool(value.get("checks")) and all(value["checks"].values())
PY
test -f "$root/scripts/finalize_t5_final_pilot.py"
test -f "$root/scripts/finalize_t5_final_pilot_bundle.py"
test -f "$root/scripts/analyze_t5_episode_runtime.py"

lane_a_run_id="${run_id}-a10"
lane_b_run_id="${run_id}-b10"
lane_a_relative="results/internnav_t5/fast-lane-a-final10-${lane_a_run_id}"
lane_b_relative="results/internnav_t5/fast-lane-b-final10-${lane_b_run_id}"
lane_a_root="$root/$lane_a_relative"
lane_b_root="$root/$lane_b_relative"
test ! -e "$lane_a_root"
test ! -e "$lane_b_root"
test ! -e "$result_root"
umask 077
mkdir -p "$result_root/logs" "$result_root/analysis"

run_lane() {
  local lane="$1" lane_run_id="$2" lane_result="$3" log="$4"
  env INTERNNAV_T5_CANDIDATE_PROFILE=a1+b1+c1 \
    INTERNNAV_T5_RTF_ABLATION_PROFILE=navigation_fast \
    INTERNNAV_T5_ISAAC_SENSOR_PROFILE=baseline \
    INTERNNAV_T5_STRICT_EXTENSION_PROFILE=off \
    INTERNNAV_T5_NVBLOX_MODE=off \
    INTERNNAV_T5_RUN_MODE=model \
    bash "$root/coordination/run_t5_fast_lane_online.sh" \
      "$lane" final10 "$lane_run_id" "$code_sha" \
      "$prepare_relative" "$lane_result" >"$log" 2>&1
}

lane_a_rc=0
lane_b_rc=0
if test "$launch_mode" = parallel; then
  set +e
  run_lane a "$lane_a_run_id" "$lane_a_relative" "$result_root/logs/lane_a.log" &
  lane_a_pid=$!
  run_lane b "$lane_b_run_id" "$lane_b_relative" "$result_root/logs/lane_b.log" &
  lane_b_pid=$!
  wait "$lane_a_pid"; lane_a_rc=$?
  wait "$lane_b_pid"; lane_b_rc=$?
  set -e
else
  set +e
  run_lane a "$lane_a_run_id" "$lane_a_relative" "$result_root/logs/lane_a.log"
  lane_a_rc=$?
  run_lane b "$lane_b_run_id" "$lane_b_relative" "$result_root/logs/lane_b.log"
  lane_b_rc=$?
  set -e
fi

analysis_a_rc=1
analysis_b_rc=1
finalizer_rc=1
if test "$lane_a_rc" = 0; then
  set +e
  python3 "$root/scripts/analyze_t5_episode_runtime.py" \
    --result-dir "$lane_a_root" \
    --output "$lane_a_root/analysis/episode_runtime_metrics.json" \
    >"$result_root/logs/lane_a_analysis.json" 2>&1
  analysis_a_rc=$?
  set -e
fi
if test "$lane_b_rc" = 0; then
  set +e
  python3 "$root/scripts/analyze_t5_episode_runtime.py" \
    --result-dir "$lane_b_root" \
    --output "$lane_b_root/analysis/episode_runtime_metrics.json" \
    >"$result_root/logs/lane_b_analysis.json" 2>&1
  analysis_b_rc=$?
  set -e
fi
if test "$lane_a_rc" = 0 && test "$lane_b_rc" = 0 \
    && test "$analysis_a_rc" = 0 && test "$analysis_b_rc" = 0; then
  set +e
  python3 "$root/scripts/finalize_t5_final_pilot_bundle.py" \
    --lane-a-root "$lane_a_root" --lane-b-root "$lane_b_root" \
    --prepare-root "$prepare_root" \
    --output "$result_root/analysis/final_pilot_report.json" \
    >"$result_root/logs/finalizer.json" 2>&1
  finalizer_rc=$?
  set -e
fi

python3 - "$result_root/final_pilot_dual_summary.json" "$run_id" "$code_sha" \
  "$launch_mode" "$lane_a_rc" "$lane_b_rc" "$analysis_a_rc" \
  "$analysis_b_rc" "$finalizer_rc" "$lane_a_relative" "$lane_b_relative" \
  "$prepare_relative" "$result_relative/analysis/final_pilot_report.json" <<'PY'
import hashlib,json,sys,time
from pathlib import Path
output=Path(sys.argv[1])
lane_a_rc,lane_b_rc,analysis_a_rc,analysis_b_rc,finalizer_rc=map(int,sys.argv[5:10])
report=output.parent/"analysis/final_pilot_report.json"
report_value=json.loads(report.read_text(encoding="utf-8")) if report.is_file() else None
checks={"lane_a_pass":lane_a_rc==0,"lane_b_pass":lane_b_rc==0,
 "lane_a_runtime_analysis_pass":analysis_a_rc==0,
 "lane_b_runtime_analysis_pass":analysis_b_rc==0,
 "portable_finalizer_pass":finalizer_rc==0 and isinstance(report_value,dict)
    and report_value.get("status")=="PASS"
    and report_value.get("path_contract",{}).get("all_local_paths_relative") is True}
payload={"schema_version":1,"status":"PASS" if all(checks.values()) else "FAIL",
 "stage":"t5_final_pilot_disjoint_a10_b10","run_id":sys.argv[2],
 "code_ref_sha":sys.argv[3],"launch_mode":sys.argv[4],
 "board_grant_used":False,"all_lanes_lock_used":False,
 "lane_results":{"a":sys.argv[10],"b":sys.argv[11]},
 "prepare_result":sys.argv[12],"final_report":sys.argv[13] if report.is_file() else None,
 "final_report_sha256":hashlib.sha256(report.read_bytes()).hexdigest() if report.is_file() else None,
 "checks":checks,"recorded_unix":time.time()}
output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
raise SystemExit(0 if payload["status"]=="PASS" else 75)
PY
