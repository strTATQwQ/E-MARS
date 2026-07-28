#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: run_t5_paired10_online.sh RUN_ID CODE_SHA PREP_RESULT_ROOT RESULT_ROOT

Run the frozen retrospective paired-10 comparison in two non-overlapping
rounds.  Round 1 launches Lane A InternVLA-only and Lane B InternVLA+Step3;
Round 2 reverses the arms while retaining each lane's frozen ten episodes.
Both children in Round 1 must finish successfully and prove lease cleanup
before Round 2 can start.  This wrapper starts no shared-I/O work.

PREP_RESULT_ROOT must be a dual-Lane final-pilot preparation generated for
CODE_SHA.  RESULT_ROOT must equal:
  results/internnav_t5/paired10-retrospective-RUN_ID
EOF
  exit 64
}

[[ $# -eq 4 ]] || usage
run_id="$1"
code_sha="$2"
prepare_relative="$3"
result_relative="$4"
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
manifest_relative=configs/internnav_t5/paired10_retrospective_manifest.json
manifest="$root/$manifest_relative"

[[ "$run_id" =~ ^[a-z0-9][a-z0-9._-]{7,63}$ ]] || usage
[[ "$code_sha" =~ ^[0-9a-f]{40}$ ]] || usage
[[ "$prepare_relative" =~ ^results/internnav_t5/final-pilot-prepare-[a-z0-9][a-z0-9._-]{7,95}$ ]] || usage
expected_result="results/internnav_t5/paired10-retrospective-${run_id}"
test "$result_relative" = "$expected_result" || {
  echo "RESULT_ROOT must equal $expected_result" >&2
  exit 64
}

git_command=(git -C "$root")
if [[ -f "$root/.git" ]] && grep -Eq '^gitdir: [A-Za-z]:/' "$root/.git"; then
  command -v git.exe >/dev/null
  command -v wslpath >/dev/null
  git_command=(git.exe -C "$(wslpath -w "$root")")
fi
"${git_command[@]}" cat-file -e "$code_sha^{commit}"
"${git_command[@]}" cat-file -e "$code_sha:$manifest_relative"
test "$("${git_command[@]}" rev-parse HEAD | tr -d '\r')" = "$code_sha"
test -z "$("${git_command[@]}" status --porcelain --untracked-files=all | tr -d '\r')"

prepare_root="$root/$prepare_relative"
result_root="$root/$result_relative"
receipt="$prepare_root/final_pilot_prepare_receipt.json"
test -f "$manifest"
test ! -L "$manifest"
test -f "$receipt"
test ! -L "$receipt"
test ! -e "$result_root"
test -f "$root/coordination/run_t5_fast_lane_online.sh"

# Fail closed if either the authored protocol or the independently prepared
# final10 assets drift.  The preparation's old final_profile is an asset
# binding; the child runner applies the explicit WP-03 runtime overlay.
python3 - "$manifest" "$receipt" "$code_sha" <<'PY'
import json
import sys
from pathlib import Path

manifest_path, receipt_path = map(Path, sys.argv[1:3])
code_sha = sys.argv[3]
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
expected_a = [
    "6898_1741", "6292_1573", "5840_1474", "6842_1720", "1420_364",
    "4084_1003", "583_145", "1803_448", "6623_1657", "2613_625",
]
expected_b = [
    "5627_1417", "4182_1027", "4009_976", "654_157", "4943_1255",
    "6982_1765", "3542_877", "6157_1561", "2564_610", "2853_703",
]
rounds = manifest.get("rounds")
sets = manifest.get("episode_sets") or {}
assert manifest.get("schema_version") == 1
assert manifest.get("status") == "FROZEN_FOR_EXECUTION"
assert manifest.get("evidence_classification", {}).get("held_out") is False
assert sets.get("paired10_a", {}).get("lane") == "a"
assert sets.get("paired10_a", {}).get("episode_keys") == expected_a
assert sets.get("paired10_b", {}).get("lane") == "b"
assert sets.get("paired10_b", {}).get("episode_keys") == expected_b
assert isinstance(rounds, list) and len(rounds) == 2
assert rounds[0].get("round_id") == "round1"
assert rounds[0].get("lane_a", {}).get("evaluation_arm") == "internvla_only"
assert rounds[0].get("lane_b", {}).get("evaluation_arm") == "internvla_step3"
assert rounds[1].get("round_id") == "round2"
assert rounds[1].get("requires_round1_clean_release") is True
assert rounds[1].get("lane_a", {}).get("evaluation_arm") == "internvla_step3"
assert rounds[1].get("lane_b", {}).get("evaluation_arm") == "internvla_only"
profile = manifest.get("runtime_profile") or {}
assert profile == {
    "candidate_profile": "recovery_a",
    "rtf_ablation_profile": "navigation_fast",
    "isaac_sensor_profile": "dual_lane_wp03_stop_shadow",
    "termination_mode": "oracle_termination",
    "system2_replan_policy": "observation_bound",
    "strict_extension_profile": "off",
    "nvblox_mode": "off",
    "run_mode": "model",
    "system1_queue_horizon": 0,
    "system2_queue_horizon": 0,
    "fault_injection_profile": "off",
}
capture = manifest.get("capture_contract") or {}
assert capture.get("model_observations", {}).get("enabled") is True
assert capture.get("d435_rgb", {}).get("sim_hz") == 5
assert capture.get("step3_snapshot", {}).get("ordered_views") == [
    "front_left", "front", "front_right", "rear"
]
latency = manifest.get("latency_contract") or {}
assert "wall_response_duration_ms" in latency.get("step3", {}).get(
    "required_fields", []
)
assert "inference_latency_sec" in latency.get("internvla", {}).get(
    "required_fields", []
)
assert receipt.get("status") == "PASS"
assert receipt.get("code_ref_sha") == code_sha
assert receipt.get("prepare_scope") == "dual"
assert receipt.get("prepared_lanes") == ["a", "b"]
assert receipt.get("final_profile") == {
    "candidate_profile": "a1+b1+c1",
    "rtf_ablation_profile": "navigation_fast",
    "isaac_sensor_profile": "baseline",
    "strict_extension_profile": "off",
    "nvblox_mode": "off",
    "run_mode": "model",
}
split = receipt.get("split") or {}
lanes = split.get("lanes") or {}
assert lanes.get("a", {}).get("episode_keys") == expected_a
assert lanes.get("b", {}).get("episode_keys") == expected_b
assert receipt.get("checks") and all(receipt["checks"].values())
PY

umask 077
mkdir -p "$result_root/logs"

round1_a_id="${run_id}-r1a"
round1_b_id="${run_id}-r1b"
round2_a_id="${run_id}-r2a"
round2_b_id="${run_id}-r2b"
round1_a_relative="results/internnav_t5/fast-lane-a-final10-${round1_a_id}"
round1_b_relative="results/internnav_t5/fast-lane-b-final10-${round1_b_id}"
round2_a_relative="results/internnav_t5/fast-lane-a-final10-${round2_a_id}"
round2_b_relative="results/internnav_t5/fast-lane-b-final10-${round2_b_id}"
for relative in "$round1_a_relative" "$round1_b_relative" \
    "$round2_a_relative" "$round2_b_relative"; do
  test ! -e "$root/$relative"
done

active_pids=()
terminate_active_children() {
  local deadline pid
  trap - EXIT INT TERM HUP
  for pid in "${active_pids[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  deadline=$((SECONDS + 60))
  while (( SECONDS < deadline )); do
    local any_alive=0
    for pid in "${active_pids[@]}"; do
      kill -0 "$pid" 2>/dev/null && any_alive=1
    done
    test "$any_alive" = 1 || break
    sleep 1
  done
  for pid in "${active_pids[@]}"; do
    kill -KILL "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  done
  exit 130
}
trap terminate_active_children INT TERM HUP

run_lane() {
  local lane="$1" advisor="$2" lane_run_id="$3" lane_result="$4" log="$5"
  (
    unset INTERNVLA_T4_VIEW_MODE INTERNVLA_T4_HISTORY_MODE
    unset INTERNVLA_T5_TRAJECTORY_RERANK INTERNVLA_T4_PROGRESS_HORIZON_SEC
    unset INTERNVLA_T4_REFRESH_DISTANCE_M INTERNVLA_T4_REFRESH_TIME_SEC
    unset INTERNVLA_T4_TRAJECTORY_VALIDITY_SEC INTERNNAV_T5_SCREEN_EPISODE_KEY
    export INTERNNAV_T5_CANDIDATE_PROFILE=recovery_a
    export INTERNNAV_T5_RTF_ABLATION_PROFILE=navigation_fast
    export INTERNNAV_T5_ISAAC_SENSOR_PROFILE=dual_lane_wp03_stop_shadow
    export INTERNNAV_T5_STEP3_LIVE_ADVISOR=0
    export INTERNVLA_T5_STEP3_TIMEOUT_ADVISOR="$advisor"
    export INTERNVLA_T5_FULL_RGB_CAPTURE=1
    export INTERNVLA_T5_D435_5HZ_CAPTURE=1
    export INTERNVLA_T5_TERMINATION_MODE=oracle_termination
    export INTERNVLA_T5_SYSTEM2_REPLAN_POLICY=observation_bound
    export INTERNVLA_T5_SYSTEM2_QUEUE_HORIZON=0
    export INTERNVLA_T5_SYSTEM1_QUEUE_HORIZON=0
    export INTERNNAV_T5_LIVE_FRONTIER_CAPTURE=0
    export INTERNNAV_T5_STRICT_EXTENSION_PROFILE=off
    export INTERNNAV_T5_NVBLOX_MODE=off
    export INTERNNAV_T5_RUN_MODE=model
    export INTERNNAV_T5_FAULT_INJECTION_PROFILE=off
    exec bash "$root/coordination/run_t5_fast_lane_online.sh" \
      "$lane" final10 "$lane_run_id" "$code_sha" \
      "$prepare_relative" "$lane_result"
  ) >"$log" 2>&1
}

verify_clean_release() {
  local result_relative="$1"
  python3 - "$root/$result_relative/fast_lane_final_summary.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
value = json.loads(path.read_text(encoding="utf-8"))
lease = value.get("lease_release") or {}
checks = lease.get("checks") or {}
assert value.get("status") == "PASS"
assert value.get("checks") and all(value["checks"].values())
assert lease.get("status") == "PASS"
assert checks.get("wrapped_command_released") is True
assert checks.get("wrapped_group_absent_before_release") is True
assert checks.get("exact_lane_resources") is True
PY
}

run_round() {
  local round_id="$1" advisor_a="$2" advisor_b="$3"
  local a_id="$4" b_id="$5" a_result="$6" b_result="$7"
  local a_log="$result_root/logs/${round_id}_lane_a.log"
  local b_log="$result_root/logs/${round_id}_lane_b.log"
  local a_pid b_pid a_rc b_rc verify_a_rc verify_b_rc
  set +e
  run_lane a "$advisor_a" "$a_id" "$a_result" "$a_log" &
  a_pid=$!
  run_lane b "$advisor_b" "$b_id" "$b_result" "$b_log" &
  b_pid=$!
  active_pids=("$a_pid" "$b_pid")
  wait "$a_pid"; a_rc=$?
  wait "$b_pid"; b_rc=$?
  active_pids=()
  set -e
  printf '%s\n' "$a_rc" >"$result_root/${round_id}_lane_a.rc"
  printf '%s\n' "$b_rc" >"$result_root/${round_id}_lane_b.rc"
  test "$a_rc" = 0 && test "$b_rc" = 0 || return 75
  verify_a_rc=0
  verify_b_rc=0
  verify_clean_release "$a_result" || verify_a_rc=$?
  verify_clean_release "$b_result" || verify_b_rc=$?
  test "$verify_a_rc" = 0 && test "$verify_b_rc" = 0 || return 75
}

round1_rc=0
run_round round1 0 1 "$round1_a_id" "$round1_b_id" \
  "$round1_a_relative" "$round1_b_relative" || round1_rc=$?
if test "$round1_rc" != 0; then
  echo "Round 1 failed or did not prove clean lease release; Round 2 was not started" >&2
fi

round2_rc=125
if test "$round1_rc" = 0; then
  run_round round2 1 0 "$round2_a_id" "$round2_b_id" \
    "$round2_a_relative" "$round2_b_relative" || round2_rc=$?
fi

trap - INT TERM HUP
python3 - "$result_root/paired10_execution_summary.json" "$manifest" \
  "$run_id" "$code_sha" "$prepare_relative" "$round1_rc" "$round2_rc" \
  "$round1_a_relative" "$round1_b_relative" \
  "$round2_a_relative" "$round2_b_relative" <<'PY'
import hashlib
import json
import sys
import time
from pathlib import Path

output, manifest_path = map(Path, sys.argv[1:3])
round1_rc, round2_rc = map(int, sys.argv[6:8])
lane_results = {
    "round1": {
        "lane_a": sys.argv[8],
        "lane_b": sys.argv[9],
        "status": "PASS" if round1_rc == 0 else "FAIL",
    },
    "round2": {
        "lane_a": sys.argv[10] if round2_rc != 125 else None,
        "lane_b": sys.argv[11] if round2_rc != 125 else None,
        "status": "NOT_RUN" if round2_rc == 125 else (
            "PASS" if round2_rc == 0 else "FAIL"
        ),
    },
}
checks = {
    "round1_pass_and_clean_release": round1_rc == 0,
    "round2_started_only_after_round1_release": round2_rc != 125
        if round1_rc == 0 else round2_rc == 125,
    "round2_pass_and_clean_release": round2_rc == 0,
}
payload = {
    "schema_version": 1,
    "status": "PASS" if all(checks.values()) else "FAIL",
    "stage": "t5_paired10_retrospective_two_rounds",
    "run_id": sys.argv[3],
    "code_ref_sha": sys.argv[4],
    "prepare_result": sys.argv[5],
    "manifest": "configs/internnav_t5/paired10_retrospective_manifest.json",
    "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    "lane_results": lane_results,
    "checks": checks,
    "evidence_classification": "RETROSPECTIVE_DEVELOPMENT_NOT_HELD_OUT",
    "unique_episode_count": 20,
    "execution_count": 40 if all(checks.values()) else None,
    "recorded_unix": time.time(),
}
output.write_text(
    json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    encoding="utf-8",
)
raise SystemExit(0 if payload["status"] == "PASS" else 75)
PY
