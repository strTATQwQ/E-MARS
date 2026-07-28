#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: run_t5_fast_lane_online.sh LANE PROFILE RUN_ID CODE_SHA PREP_RESULT_ROOT RESULT_ROOT

  LANE:    a | b
  PROFILE: canary60 | soak600 | screen1 | screen3 | fixed5 | pilot-screen1 | final10

  INTERNNAV_T5_CANDIDATE_PROFILE: baseline | recovery_a | a[01][+b[01][+c[012]]]
      (default baseline; composite IDs are resolved only from the preregistered
      Lane-A manifest, in dependency order)
  INTERNNAV_T5_RTF_ABLATION_PROFILE: navigation_fast | off | baseline |
      lidar_off_probe | lidar_720 | depth_stride8 | sensor_2p5hz
      (default navigation_fast; only diagnostic profiles require canary60)
  INTERNNAV_T5_ISAAC_SENSOR_PROFILE: baseline | lane_b_revc_smoke |
      lane_b_revc_fixed5_capture | lane_b_step3_live_canary |
      lane_a_step3_timeout_advisor | dual_lane_wp03_stop_shadow
  INTERNNAV_T5_STRICT_EXTENSION_PROFILE: off | cuvslam_shadow
      (default off; cuVSLAM shadow is Lane A screen3 only)
  INTERNNAV_T5_NVBLOX_MODE: off | shadow | active_local_gt
      (default off; shadow is Lane A screen3/fixed5, active-local is screen3,
      and neither mode can overlap cuVSLAM)
  INTERNNAV_T5_LIVE_FRONTIER_CAPTURE: 0 | 1
      (default follows the Lane-B Step3 advisor; capture-only ROS sidecar)
  INTERNNAV_T5_RUN_MODE: model | oracle
      (default model; oracle is reserved for Lane-A active-local acceptance)
  INTERNVLA_T5_SYSTEM2_REPLAN_POLICY: strict | observation_bound | raw_wire_warn
      (default observation_bound; strict/raw-wire are Lane-A Recovery-A
      completion_sim diagnostics only)
  INTERNVLA_T5_SYSTEM2_QUEUE_HORIZON: 0 | 1
      (default 0; 1 is a Lane-A Recovery-A completion_sim receding-horizon
      diagnostic that discards unexecuted actions from an old observation)
  INTERNVLA_T5_SYSTEM1_QUEUE_HORIZON: 0 | 1
      (default 0; 1 is a Lane-A Recovery-A completion_sim receding-horizon
      diagnostic that replans a new System1 trajectory after each observation)

Engineering-only T5 lane runner.  It deliberately does not consume a board
grant or a D0 predecessor.  A run is bound to one clean code commit and one
successful D0.0 preparation receipt, and holds only the selected Lane's DGX
and Isaac GPU leases.  canary60 and soak600 prove READY for 60 or 600 seconds
respectively and then perform a supervised completion_sim shutdown. screen1
and screen3 materialize the deterministic first N rows of the prepared frozen
five; pilot-screen1 selects one explicit episode from a sealed pilot lane, and
final10 consumes that lane's full half of the independently sealed pilot20.
All fixed-dataset profiles let their evaluator finish normally.
EOF
  exit 64
}

[[ $# -eq 6 ]] || usage
lane="$1"
profile="$2"
run_id="$3"
code_sha="$4"
prep_relative="$5"
result_relative="$6"
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
x86_ip="${ISAAC_HOST:-10.100.120.123}"
[[ "$x86_ip" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]] || {
  echo "ISAAC_HOST must be an IPv4 address" >&2
  exit 64
}
test "$x86_ip" = 10.100.120.123 || {
  echo "active T5 Isaac host is fixed at 10.100.120.123" >&2
  exit 64
}

case "$lane" in
  a)
    resource_profile=lane-a
    dgx_target=railgun@10.100.100.128
    dgx_user=railgun
    dgx_ip=10.100.100.128
    dgx_root_key=dgx_a
    x86_root_key=x86_a
    ros_domain_id=75
    gpu=0
    cpuset="${INTERNVLA_T5_LANE_A_CPUSET:-0,2,4,6,8,10,12,14,16}"
    container=internnav_t5_isaac_a
    own_ports="25137 25139 25140 25141"
    ipc_alias=/tmp/internnav_t5_a_ipc
    lane_runtime_lock=/tmp/internnav_t5_isaac_a_runtime.lock
    dgx_quarantine_role=dgx_a
    x86_quarantine_role=x86_gpu0
    ;;
  b)
    resource_profile=lane-b
    dgx_target=rail@10.100.120.122
    dgx_user=rail
    dgx_ip=10.100.120.122
    dgx_root_key=dgx_b
    x86_root_key=x86_b
    ros_domain_id=76
    gpu=1
    cpuset="${INTERNVLA_T5_LANE_B_CPUSET:-1,3,5,7,9,11,13,15,17}"
    container=internnav_t5_isaac_b
    own_ports="25138 25239 25240 25241"
    ipc_alias=/tmp/internnav_t5_b_ipc
    lane_runtime_lock=/tmp/internnav_t5_isaac_b_runtime.lock
    dgx_quarantine_role=dgx_b
    x86_quarantine_role=x86_gpu1
    ;;
  *) usage ;;
esac
[[ "$cpuset" =~ ^[0-9,-]+$ ]] || {
  echo "Isaac Lane CPU set contains unsafe characters" >&2
  exit 64
}
if test "$lane" = a; then
  test "$cpuset" = 0,2,4,6,8,10,12,14,16
else
  test "$cpuset" = 1,3,5,7,9,11,13,15,17
fi
case "$profile" in canary60|soak600|screen1|screen3|fixed5|pilot-screen1|final10) ;; *) usage ;; esac
case "$profile" in
  screen1|pilot-screen1) screen_episode_count=1 ;;
  screen3) screen_episode_count=3 ;;
  *) screen_episode_count=0 ;;
esac
screen_episode_key="${INTERNNAV_T5_SCREEN_EPISODE_KEY:-}"
if test -n "$screen_episode_key"; then
  case "$profile" in screen1|pilot-screen1) ;; *) usage ;; esac
  [[ "$screen_episode_key" =~ ^[A-Za-z0-9_.-]+$ ]] || usage
fi
if test "$profile" = pilot-screen1; then
  test -n "$screen_episode_key" || usage
fi
if [[ -v INTERNNAV_T5_CANDIDATE_PROFILE ]]; then
  candidate_profile="$INTERNNAV_T5_CANDIDATE_PROFILE"
else
  candidate_profile=baseline
fi
case "$candidate_profile" in baseline|recovery_a) ;; a0|a1|a0+b0|a0+b1|a1+b0|a1+b1|a1+b0+c0|a1+b0+c1|a1+b0+c2|a1+b1+c0|a1+b1+c1|a1+b1+c2) ;; *) usage ;; esac
system2_replan_policy="${INTERNVLA_T5_SYSTEM2_REPLAN_POLICY:-observation_bound}"
case "$system2_replan_policy" in
  strict|observation_bound|raw_wire_warn) ;;
  *) usage ;;
esac
system2_queue_horizon="${INTERNVLA_T5_SYSTEM2_QUEUE_HORIZON:-0}"
case "$system2_queue_horizon" in 0|1) ;; *) usage ;; esac
system1_queue_horizon="${INTERNVLA_T5_SYSTEM1_QUEUE_HORIZON:-0}"
case "$system1_queue_horizon" in 0|1) ;; *) usage ;; esac
if test "$system2_replan_policy" != observation_bound; then
  test "$lane" = a || usage
  test "$candidate_profile" != baseline || usage
fi
candidate_runtime_names=(
  INTERNVLA_T4_VIEW_MODE
  INTERNVLA_T4_HISTORY_MODE
  INTERNVLA_T5_TRAJECTORY_RERANK
  INTERNVLA_T4_PROGRESS_HORIZON_SEC
  INTERNVLA_T4_REFRESH_DISTANCE_M
  INTERNVLA_T4_REFRESH_TIME_SEC
  INTERNVLA_T4_TRAJECTORY_VALIDITY_SEC
)
for candidate_runtime_name in "${candidate_runtime_names[@]}"; do
  if [[ -v $candidate_runtime_name ]]; then
    echo "candidate runtime overrides must come from the preregistered manifest: $candidate_runtime_name" >&2
    exit 64
  fi
done
test -f "$root/scripts/resolve_t5_lane_a_candidate.py"
python3 "$root/scripts/resolve_t5_lane_a_candidate.py" \
  --root "$root" --selector "$candidate_profile" --format none
rtf_ablation_profile="${INTERNNAV_T5_RTF_ABLATION_PROFILE:-navigation_fast}"
case "$rtf_ablation_profile" in
  navigation_fast|off|baseline|lidar_off_probe|lidar_720|depth_stride8|sensor_2p5hz) ;;
  *) usage ;;
esac
case "$rtf_ablation_profile" in
  navigation_fast|off) ;;
  *) test "$profile" = canary60 || usage ;;
esac
if [[ -v INTERNVLA_T5_REVC_ENABLE ]]; then
  echo "Rev-C must be selected only through INTERNNAV_T5_ISAAC_SENSOR_PROFILE" >&2
  exit 64
fi
isaac_sensor_profile="${INTERNNAV_T5_ISAAC_SENSOR_PROFILE:-baseline}"
step3_live_advisor="${INTERNNAV_T5_STEP3_LIVE_ADVISOR:-0}"
case "$step3_live_advisor" in 0|1) ;; *) usage ;; esac
step3_timeout_advisor="${INTERNVLA_T5_STEP3_TIMEOUT_ADVISOR:-0}"
case "$step3_timeout_advisor" in 0|1) ;; *) usage ;; esac
evaluation_arm=internvla_only
test "$step3_timeout_advisor" != 1 || evaluation_arm=internvla_step3
full_rgb_capture="${INTERNVLA_T5_FULL_RGB_CAPTURE:-0}"
case "$full_rgb_capture" in 0|1) ;; *) usage ;; esac
d435_5hz_capture="${INTERNVLA_T5_D435_5HZ_CAPTURE:-0}"
case "$d435_5hz_capture" in 0|1) ;; *) usage ;; esac
termination_mode="${INTERNVLA_T5_TERMINATION_MODE:-model_stop}"
case "$termination_mode" in model_stop|oracle_termination) ;; *) usage ;; esac
live_frontier_capture="${INTERNNAV_T5_LIVE_FRONTIER_CAPTURE:-$step3_live_advisor}"
case "$live_frontier_capture" in 0|1) ;; *) usage ;; esac
test "$live_frontier_capture" != 1 || test "$lane" = b || usage
step3_dgx_ports=""
test "$step3_live_advisor" != 1 || step3_dgx_ports="8200 8300"
test "$step3_timeout_advisor" != 1 || step3_dgx_ports="8200"
case "$isaac_sensor_profile" in
  baseline)
    test "$step3_live_advisor" = 0 || usage
    test "$step3_timeout_advisor" = 0 || usage
    ;;
  lane_b_revc_smoke)
    test "$step3_live_advisor" = 0 || usage
    test "$step3_timeout_advisor" = 0 || usage
    test "$lane" = b || usage
    test "$profile" = canary60 || usage
    case "$rtf_ablation_profile" in navigation_fast|off) ;; *) usage ;; esac
    ;;
  lane_b_revc_fixed5_capture)
    test "$step3_live_advisor" = 0 || usage
    test "$step3_timeout_advisor" = 0 || usage
    test "$lane" = b || usage
    test "$profile" = fixed5 || usage
    case "$rtf_ablation_profile" in navigation_fast|off) ;; *) usage ;; esac
    ;;
  lane_b_step3_live_canary)
    test "$step3_live_advisor" = 1 || usage
    test "$step3_timeout_advisor" = 0 || usage
    test "$lane" = b || usage
    case "$profile" in screen1|screen3) ;; *) usage ;; esac
    test "$candidate_profile" = a1+b1+c1 || usage
    case "$rtf_ablation_profile" in navigation_fast|off) ;; *) usage ;; esac
    ;;
  lane_a_step3_timeout_advisor)
    test "$step3_live_advisor" = 0 || usage
    test "$step3_timeout_advisor" = 1 || usage
    test "$lane" = a || usage
    case "$profile" in screen1|screen3|fixed5) ;; *) usage ;; esac
    test "$candidate_profile" = recovery_a || usage
    case "$rtf_ablation_profile" in navigation_fast|off) ;; *) usage ;; esac
    ;;
  dual_lane_wp03_stop_shadow)
    test "$step3_live_advisor" = 0 || usage
    case "$profile" in pilot-screen1|final10) ;; *) usage ;; esac
    test "$candidate_profile" = recovery_a || usage
    test "$termination_mode" = oracle_termination || usage
    case "$rtf_ablation_profile" in navigation_fast|off) ;; *) usage ;; esac
    ;;
  *) usage ;;
esac
live_frontier_ready_required=0
if test "$lane" = b && test "$profile" = canary60 && \
    test "$isaac_sensor_profile" = baseline && \
    test "$step3_live_advisor" = 0 && test "$live_frontier_capture" = 1; then
  live_frontier_ready_required=1
fi
strict_extension_profile="${INTERNNAV_T5_STRICT_EXTENSION_PROFILE:-off}"
case "$strict_extension_profile" in
  off) ;;
  cuvslam_shadow)
    test "$lane" = a || usage
    test "$profile" = screen3 || usage
    case "$rtf_ablation_profile" in navigation_fast|off) ;; *) usage ;; esac
    test "$isaac_sensor_profile" = baseline || usage
    ;;
  *) usage ;;
esac
nvblox_mode="${INTERNNAV_T5_NVBLOX_MODE:-off}"
case "$nvblox_mode" in
  off) ;;
  shadow)
    test "$lane" = a || usage
    case "$profile" in screen3|fixed5) ;; *) usage ;; esac
    case "$rtf_ablation_profile" in navigation_fast|off) ;; *) usage ;; esac
    test "$isaac_sensor_profile" = baseline || usage
    test "$strict_extension_profile" = off || usage
    ;;
  active_local_gt)
    test "$lane" = a || usage
    test "$profile" = screen3 || usage
    case "$rtf_ablation_profile" in navigation_fast|off) ;; *) usage ;; esac
    test "$isaac_sensor_profile" = baseline || usage
    test "$strict_extension_profile" = off || usage
    ;;
  *) usage ;;
esac
run_mode="${INTERNNAV_T5_RUN_MODE:-model}"
case "$run_mode" in
  model) ;;
  oracle)
    test "$lane" = a || usage
    test "$profile" = screen3 || usage
    test "$candidate_profile" = baseline || usage
    test "$nvblox_mode" = active_local_gt || usage
    ;;
  *) usage ;;
esac
if test "$system2_queue_horizon" = 1; then
  test "$lane" = a || usage
  test "$candidate_profile" = recovery_a || usage
  test "$run_mode" = model || usage
fi
if test "$system1_queue_horizon" = 1; then
  test "$lane" = a || usage
  test "$candidate_profile" = recovery_a || usage
  test "$run_mode" = model || usage
fi
if test "$termination_mode" = oracle_termination; then
  test "$run_mode" = model || usage
  if test "$isaac_sensor_profile" = dual_lane_wp03_stop_shadow; then
    case "$profile" in pilot-screen1|final10) ;; *) usage ;; esac
    test "$candidate_profile" = recovery_a || usage
  else
    test "$lane" = a || usage
    case "$profile" in screen1|screen3|fixed5) ;; *) usage ;; esac
    test "$isaac_sensor_profile" = lane_a_step3_timeout_advisor || usage
  fi
fi
if [[ "$profile" == final10 || "$profile" == pilot-screen1 ]]; then
  test "$system2_replan_policy" = observation_bound || usage
  test "$rtf_ablation_profile" = navigation_fast || usage
  test "$strict_extension_profile" = off || usage
  test "$nvblox_mode" = off || usage
  test "$run_mode" = model || usage
  test "$live_frontier_capture" = 0 || usage
  if test "$termination_mode" = oracle_termination; then
    test "$candidate_profile" = recovery_a || usage
    test "$isaac_sensor_profile" = dual_lane_wp03_stop_shadow || usage
  else
    test "$candidate_profile" = a1+b1+c1 || usage
    test "$isaac_sensor_profile" = baseline || usage
  fi
fi
if test "$step3_live_advisor" = 1; then
  test "$strict_extension_profile" = off || usage
  test "$nvblox_mode" = off || usage
  test "$run_mode" = model || usage
fi
if test "$step3_timeout_advisor" = 1; then
  test "$strict_extension_profile" = off || usage
  test "$nvblox_mode" = off || usage
  test "$run_mode" = model || usage
  test "$system2_replan_policy" = observation_bound || usage
fi
fault_injection_profile="${INTERNNAV_T5_FAULT_INJECTION_PROFILE:-off}"
case "$fault_injection_profile" in
  off) ;;
  completion_sim_minimal_v1)
    test "$profile" = soak600 || usage
    test "$run_mode" = model || usage
    test "$isaac_sensor_profile" = baseline || usage
    test "$strict_extension_profile" = off || usage
    test "$nvblox_mode" = off || usage
    case "$rtf_ablation_profile" in navigation_fast|off) ;; *) usage ;; esac
    ;;
  *) usage ;;
esac
[[ "$run_id" =~ ^[a-z0-9][a-z0-9._-]{7,95}$ ]] || usage
[[ "$code_sha" =~ ^[0-9a-f]{40}$ ]] || usage
if [[ "$profile" == final10 || "$profile" == pilot-screen1 ]]; then
  [[ "$prep_relative" =~ ^results/internnav_t5/final-pilot-prepare-[a-z0-9][a-z0-9._-]{7,95}$ ]] || usage
else
  [[ "$prep_relative" =~ ^results/internnav_t5/(d0-0-prepare-t5d00[0-9]{8}t[0-9]{6}|step3-lane-b-prepare-[a-z0-9][a-z0-9._-]{7,95})$ ]] || usage
fi
expected_result="results/internnav_t5/fast-lane-${lane}-${profile}-${run_id}"
test "$result_relative" = "$expected_result" || {
  echo "RESULT_ROOT must equal $expected_result" >&2
  exit 64
}

result_dir="$root/$result_relative"
prep_dir="$root/$prep_relative"
test -d "$prep_dir"
test ! -L "$prep_dir"
if [[ "$profile" == final10 || "$profile" == pilot-screen1 ]]; then
  required_receipts=(final_pilot_prepare_receipt.json assets/final_pilot_split_audit.json)
else
  required_receipts=(fast_prepare_input.json d0_prepare_summary.json \
    d0_prepare_final_summary.json)
fi
for receipt in "${required_receipts[@]}"; do
  test -f "$prep_dir/$receipt"
  test ! -L "$prep_dir/$receipt"
done

# Each invocation owns exactly one disjoint hardware pair.  Lease diagnostics
# live beside the immutable result until the remote holders have released.
if [[ "${INTERNNAV_T5_INSIDE_FAST_LANE:-0}" != 1 ]]; then
  test ! -e "$result_dir"
  mkdir -p "$root/results/internnav_t5"
  lease_bootstrap="$(mktemp -d "$root/results/internnav_t5/.fast-lane-${lane}-${run_id}.XXXXXX")"
  set +e
  env ISAAC_HOST="$x86_ip" bash "$root/scripts/with_resource_lease.sh" "$resource_profile" \
    --owner codex-00 --task "t5-fast:${lane}:${profile}:${run_id}:${code_sha}" \
    --log-dir "$lease_bootstrap" --acquire-timeout 30 \
    --cleanup-timeout 60 --kill-wait-timeout 10 -- \
    env INTERNNAV_T5_INSIDE_FAST_LANE=1 \
      INTERNNAV_T5_RESOURCE_LEASE_ACK="$resource_profile" \
      INTERNNAV_T5_CANDIDATE_PROFILE="$candidate_profile" \
      INTERNNAV_T5_RTF_ABLATION_PROFILE="$rtf_ablation_profile" \
      INTERNNAV_T5_ISAAC_SENSOR_PROFILE="$isaac_sensor_profile" \
      INTERNNAV_T5_STEP3_LIVE_ADVISOR="$step3_live_advisor" \
      INTERNVLA_T5_STEP3_TIMEOUT_ADVISOR="$step3_timeout_advisor" \
      INTERNVLA_T5_FULL_RGB_CAPTURE="$full_rgb_capture" \
      INTERNVLA_T5_D435_5HZ_CAPTURE="$d435_5hz_capture" \
      INTERNVLA_T5_TERMINATION_MODE="$termination_mode" \
      INTERNNAV_T5_LIVE_FRONTIER_CAPTURE="$live_frontier_capture" \
      INTERNNAV_T5_STRICT_EXTENSION_PROFILE="$strict_extension_profile" \
      INTERNNAV_T5_NVBLOX_MODE="$nvblox_mode" \
      INTERNNAV_T5_RUN_MODE="$run_mode" \
      INTERNNAV_T5_FAULT_INJECTION_PROFILE="$fault_injection_profile" \
      INTERNNAV_T5_SCREEN_EPISODE_KEY="$screen_episode_key" \
      INTERNVLA_T5_SYSTEM2_REPLAN_POLICY="$system2_replan_policy" \
      INTERNVLA_T5_SYSTEM2_QUEUE_HORIZON="$system2_queue_horizon" \
      INTERNVLA_T5_SYSTEM1_QUEUE_HORIZON="$system1_queue_horizon" \
      bash "$root/coordination/run_t5_fast_lane_online.sh" \
        "$lane" "$profile" "$run_id" "$code_sha" \
        "$prep_relative" "$result_relative"
  command_rc=$?
  set -e
  if test -d "$result_dir"; then
    install -d -m 700 "$result_dir/lease"
    cp -a -- "$lease_bootstrap/." "$result_dir/lease/"
    python3 - "$result_dir/lease_release_summary.json" \
      "$result_dir/lease/lease_metadata.txt" "$command_rc" \
      "$resource_profile" <<'PY'
import hashlib, json, sys, time
from pathlib import Path

output, metadata = Path(sys.argv[1]), Path(sys.argv[2])
command_rc, profile = int(sys.argv[3]), sys.argv[4]
text = metadata.read_text(encoding="utf-8") if metadata.is_file() else ""
holders = sorted(metadata.parent.glob("holder_*.stdout.log"))
cleanup_path = metadata.parent / "lease_cleanup_receipt.json"
cleanup = json.loads(cleanup_path.read_text(encoding="utf-8")) \
    if cleanup_path.is_file() else None
expected = ("dgx_a", "isaac_gpu0") if profile == "lane-a" \
    else ("dgx_b", "isaac_gpu1")
other = {"dgx_a", "dgx_b", "isaac_gpu0", "isaac_gpu1"} - set(expected)
checks = {
    "two_holders_recorded": len(holders) == 2,
    "exact_lane_resources": all(value in text for value in expected)
        and all(value not in text for value in other),
    "wrapped_command_released": "state=RELEASED" in text,
    "wrapped_command_exit_recorded": f"command_exit={command_rc}" in text,
    "wrapped_group_absent_before_release": isinstance(cleanup, dict)
        and cleanup.get("status") == "PASS"
        and cleanup.get("wrapped_process_group_absent_before_lock_release") is True,
}
payload = {
    "schema_version": 1,
    "status": "PASS" if all(checks.values()) else "FAIL",
    "resource_profile": profile,
    "command_exit": command_rc,
    "checks": checks,
    "holder_logs": [path.name for path in holders],
    "recorded_unix": time.time(),
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
    python3 - "$result_dir/fast_lane_summary.json" \
      "$result_dir/lease_release_summary.json" \
      "$result_dir/fast_lane_final_summary.json" "$command_rc" <<'PY'
import hashlib, json, sys, time
from pathlib import Path

runtime_path, lease_path, output = map(Path, sys.argv[1:4])
command_rc = int(sys.argv[4])
runtime = json.loads(runtime_path.read_text(encoding="utf-8")) \
    if runtime_path.is_file() else None
lease = json.loads(lease_path.read_text(encoding="utf-8"))
checks = {
    "runtime_pass": isinstance(runtime, dict) and runtime.get("status") == "PASS",
    "lease_release_pass": lease.get("status") == "PASS"
        and bool(lease.get("checks")) and all(lease["checks"].values()),
    "command_exit_zero": command_rc == 0,
}
payload = {
    "schema_version": 1,
    "status": "PASS" if all(checks.values()) else "FAIL",
    "checks": checks,
    "runtime_summary": runtime,
    "lease_release": lease,
    "recorded_unix": time.time(),
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
    final_status="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "$result_dir/fast_lane_final_summary.json")"
    test "$final_status" = PASS || command_rc=1
    rm -rf -- "$lease_bootstrap"
  else
    printf 'fast Lane did not consume result root; lease logs remain at %s\n' \
      "$lease_bootstrap" >&2
  fi
  exit "$command_rc"
fi

test "${INTERNNAV_T5_RESOURCE_LEASE_ACK:-}" = "$resource_profile"

git_command=(git -C "$root")
if [[ -f "$root/.git" ]] && grep -Eq '^gitdir: [A-Za-z]:/' "$root/.git"; then
  command -v git.exe >/dev/null
  command -v wslpath >/dev/null
  git_command=(git.exe -C "$(wslpath -w "$root")")
fi
"${git_command[@]}" cat-file -e "$code_sha^{commit}"
test "$("${git_command[@]}" rev-parse HEAD | tr -d '\r')" = "$code_sha"
test -z "$("${git_command[@]}" status --porcelain --untracked-files=all | tr -d '\r')"

test ! -e "$result_dir"
validation_tmp="$(mktemp -d "${TMPDIR:-/tmp}/internnav-t5-fast-lane.XXXXXX")"
trap 'rm -rf -- "$validation_tmp"' EXIT
python3 "$root/scripts/resolve_t5_lane_a_candidate.py" \
  --root "$root" --selector "$candidate_profile" \
  --output "$validation_tmp/candidate_resolution.json" --format none
if [[ "$profile" == final10 || "$profile" == pilot-screen1 ]]; then
  final_pilot_selection_args=(--execution-profile "$profile")
  if test "$profile" = pilot-screen1; then
    final_pilot_selection_args+=(--episode-key "$screen_episode_key")
  fi
  python3 "$root/scripts/resolve_t5_final_pilot_lane_binding.py" \
    --prepare-root "$prep_dir" --lane "$lane" --code-sha "$code_sha" \
    --candidate-resolution "$validation_tmp/candidate_resolution.json" \
    --candidate-profile "$candidate_profile" \
    --rtf-ablation-profile "$rtf_ablation_profile" \
    --isaac-sensor-profile "$isaac_sensor_profile" \
    --strict-extension-profile "$strict_extension_profile" \
    --nvblox-mode "$nvblox_mode" --run-mode "$run_mode" \
    --evaluation-arm "$evaluation_arm" \
    "${final_pilot_selection_args[@]}" \
    --output "$validation_tmp/input_binding.json" >/dev/null
else
  python3 - "$prep_dir/fast_prepare_input.json" \
  "$prep_dir/d0_prepare_summary.json" \
  "$prep_dir/d0_prepare_final_summary.json" \
  "$prep_dir" "$code_sha" "$lane" \
  "$candidate_profile" "$rtf_ablation_profile" "$isaac_sensor_profile" \
  "$nvblox_mode" "$run_mode" "$profile" "$screen_episode_count" \
  "$fault_injection_profile" "$live_frontier_capture" \
  "$live_frontier_ready_required" \
  "$system2_replan_policy" \
  "$validation_tmp/candidate_resolution.json" \
  "$validation_tmp/input_binding.json" \
  "$screen_episode_key" <<'PY'
import hashlib, json, re, sys
from pathlib import Path, PurePosixPath

input_path, prep_path, final_path, prepare_root = map(Path, sys.argv[1:5])
code_sha, lane, candidate_profile, rtf_ablation_profile, isaac_sensor_profile, \
    nvblox_mode, run_mode, profile, screen_count, fault_injection_profile, \
    live_frontier_capture, live_frontier_ready_required, system2_replan_policy = (
    sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9],
    sys.argv[10], sys.argv[11], sys.argv[12], int(sys.argv[13]), sys.argv[14],
    sys.argv[15] == "1", sys.argv[16] == "1", sys.argv[17]
)
candidate_path, output = map(Path, sys.argv[18:20])
screen_episode_key = sys.argv[20]
frozen_input = json.loads(input_path.read_text(encoding="utf-8"))
prep = json.loads(prep_path.read_text(encoding="utf-8"))
final = json.loads(final_path.read_text(encoding="utf-8"))
records = (frozen_input, prep, final)
legacy_dual = all(
    "prepare_scope" not in value and "prepared_lanes" not in value
    for value in records
)
explicit_dual = all(
    value.get("prepare_scope") == "dual"
    and value.get("prepared_lanes") == ["a", "b"]
    for value in records
)
lane_a_only = all(
    value.get("prepare_scope") == "lane-a"
    and value.get("prepared_lanes") == ["a"]
    for value in records
)
lane_b_only = all(
    value.get("prepare_scope") in {"lane-b", "lane_b"}
    and value.get("prepared_lanes") == ["b"]
    for value in records
)
if lane_a_only:
    if lane != "a":
        raise SystemExit("Lane-A prepare cannot authorize Lane B")
    prepare_scope = "lane-a"
    prepared_lanes = ["a"]
    dataset_relative = Path("remote/x86_a/dataset_audit.json")
elif lane_b_only:
    if lane != "b":
        raise SystemExit("Lane-B prepare cannot authorize Lane A")
    prepare_scope = "lane-b"
    prepared_lanes = ["b"]
    dataset_relative = Path("remote/x86/dataset_audit.json")
elif legacy_dual or explicit_dual:
    prepare_scope = "dual"
    prepared_lanes = ["a", "b"]
    dataset_relative = Path("remote/x86/dataset_audit.json")
else:
    raise SystemExit("prepare scope receipts disagree")
dataset_path = prepare_root / dataset_relative
if not dataset_path.is_file() or dataset_path.is_symlink():
    raise SystemExit("dataset audit missing or unsafe")
dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
roots = prep.get("deployment_roots", {})
dgx_key, x86_key = ("dgx_a", "x86_a") if lane == "a" else ("dgx_b", "x86_b")
dgx_prefix = "/home/railgun/internnav-t1-t2/.t5-deployments/" if lane == "a" \
    else "/home/rail/internnav-t1-t2/.t5-deployments/"
x86_prefix = "/home/song/internnav-t1-t2/.t5-deployments/"
dgx_root, x86_root = roots.get(dgx_key, ""), roots.get(x86_key, "")
dataset_file = str(dataset.get("dataset_file", ""))
dataset_root = str(PurePosixPath(dataset_file).parent.parent) if dataset_file else ""
dataset_audit_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
fast_prepare_input_sha256 = hashlib.sha256(input_path.read_bytes()).hexdigest()
expected_keys = frozen_input.get("episode_keys")
safe_sha = re.compile(r"^[0-9a-f]{64}$")
def canonical_sha256(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()
unsigned_candidate = dict(candidate)
declared_resolution_sha256 = unsigned_candidate.pop("resolution_sha256", None)
candidate_binding = candidate.get("canonical_binding")
candidate_is_legacy = candidate.get("selector_kind") == "legacy"
candidate_is_composite = candidate.get("selector_kind") == "preregistered_composite"
selected_ids = candidate.get("selected_candidate_ids")
selected_configs = candidate.get("selected_configs", [])
registered_configs = candidate.get("registered_config_sha256_by_family")
candidate_manifest = candidate.get("candidate_manifest")
raw_provenance = candidate.get("provenance")
provenance = raw_provenance if isinstance(raw_provenance, dict) else {}
composite_profile_allowed = re.fullmatch(
    r"(?:a0(?:\+b[01])?|a1(?:\+b[01](?:\+c[012])?)?)",
    candidate_profile,
) is not None
checks = {
    "candidate_profile_allowed": candidate_profile in {"baseline", "recovery_a"}
        or composite_profile_allowed,
    "candidate_resolution_pass": candidate.get("schema_version") == 2
        and candidate.get("status") == "PASS"
        and candidate.get("candidate_selector") == candidate_profile,
    "candidate_resolution_self_hash": safe_sha.fullmatch(
        str(declared_resolution_sha256 or "")) is not None
        and declared_resolution_sha256 == canonical_sha256(unsigned_candidate),
    "candidate_binding_exact": isinstance(candidate_binding, dict)
        and candidate_binding.get("candidate_selector") == candidate_profile
        and candidate_binding.get("effective_candidate_profile")
            == candidate.get("effective_candidate_profile")
        and candidate_binding.get("candidate_manifest") == candidate_manifest
        and candidate_binding.get("selected_configs") == selected_configs
        and candidate_binding.get("registered_config_sha256_by_family")
            == registered_configs
        and isinstance(raw_provenance, dict)
        and candidate_binding.get("code_bundle_sha256")
            == provenance.get("code_bundle_sha256")
        and candidate_binding.get("fixed_episode_count")
            == provenance.get("fixed_episode_count")
        and candidate_binding.get("fixed_episode_keys")
            == provenance.get("fixed_episode_keys")
        and candidate_binding.get("predecessor_candidate_ids")
            == provenance.get("predecessor_candidate_ids"),
    "candidate_legacy_binding": (not candidate_is_legacy) or (
        candidate_profile in {"baseline", "recovery_a"}
        and candidate_manifest is None
        and selected_ids == [] and selected_configs == []
        and registered_configs == {}
        and provenance == {
            "code_bundle_sha256": None,
            "fixed_episode_count": None,
            "fixed_episode_keys": None,
            "predecessor_candidate_ids": [],
        }
    ),
    "candidate_manifest_config_code_binding": (not candidate_is_composite) or (
        isinstance(candidate_manifest, dict)
        and safe_sha.fullmatch(str(candidate_manifest.get("canonical_sha256", "")))
            is not None
        and isinstance(selected_ids, list) and selected_ids
        and isinstance(selected_configs, list)
        and len(selected_configs) == len(selected_ids)
        and all(isinstance(item, dict)
                and safe_sha.fullmatch(str(item.get("canonical_sha256", "")))
                    is not None for item in selected_configs)
        and isinstance(registered_configs, dict)
        and set(registered_configs) == {
            "action_observation_recovery", "camera_history_alignment",
            "trajectory_horizon_refresh",
        }
        and all(safe_sha.fullmatch(str(value)) is not None
                for value in registered_configs.values())
        and safe_sha.fullmatch(str(provenance.get("code_bundle_sha256", "")))
            is not None
    ),
    "candidate_fixed_episode_binding": (not candidate_is_composite) or (
        provenance.get("fixed_episode_count") == len(expected_keys or []) == 5
        and provenance.get("fixed_episode_keys") == expected_keys
    ),
    "candidate_predecessor_binding": (not candidate_is_composite) or (
        provenance.get("predecessor_candidate_ids") == selected_ids[:-1]
        and len(selected_ids) == len(candidate_profile.split("+"))
    ),
    "rtf_ablation_profile_allowed": rtf_ablation_profile in {
        "navigation_fast", "off", "baseline", "lidar_off_probe", "lidar_720", "depth_stride8",
        "sensor_2p5hz",
    },
    "isaac_sensor_profile_allowed": isaac_sensor_profile in {
        "baseline", "lane_b_revc_smoke", "lane_b_revc_fixed5_capture",
        "lane_b_step3_live_canary", "lane_a_step3_timeout_advisor",
        "dual_lane_wp03_stop_shadow",
    },
    "fault_injection_exact_scope": fault_injection_profile == "off" or (
        fault_injection_profile == "completion_sim_minimal_v1"
        and profile == "soak600" and run_mode == "model"
        and isaac_sensor_profile == "baseline"
        and nvblox_mode == "off"
        and rtf_ablation_profile in {"navigation_fast", "off"}
    ),
    "live_frontier_capture_exact_scope": (not live_frontier_capture) or lane == "b",
    "live_frontier_ready_exact_scope": live_frontier_ready_required == (
        live_frontier_capture and lane == "b" and profile == "canary60"
        and isaac_sensor_profile == "baseline"
    ),
    "revc_profile_exact_scope": isaac_sensor_profile == "baseline" or (
        isaac_sensor_profile == "lane_b_revc_smoke"
        and lane == "b" and profile == "canary60"
        and rtf_ablation_profile in {"navigation_fast", "off"}
    ) or (
        isaac_sensor_profile == "lane_b_revc_fixed5_capture"
        and lane == "b" and profile == "fixed5"
        and rtf_ablation_profile in {"navigation_fast", "off"}
    ) or (
        isaac_sensor_profile == "lane_b_step3_live_canary"
        and lane == "b" and profile in {"screen1", "screen3"}
        and candidate_profile == "a1+b1+c1"
        and run_mode == "model" and nvblox_mode == "off"
        and rtf_ablation_profile in {"navigation_fast", "off"}
    ) or (
        isaac_sensor_profile == "lane_a_step3_timeout_advisor"
        and lane == "a" and profile in {"screen1", "screen3", "fixed5"}
        and candidate_profile == "recovery_a"
        and run_mode == "model" and nvblox_mode == "off"
        and system2_replan_policy == "observation_bound"
        and rtf_ablation_profile in {"navigation_fast", "off"}
    ) or (
        isaac_sensor_profile == "dual_lane_wp03_stop_shadow"
        and lane in {"a", "b"} and profile == "final10"
        and candidate_profile == "recovery_a"
        and termination_mode == "oracle_termination"
        and run_mode == "model"
        and nvblox_mode == "off"
        and system2_replan_policy == "observation_bound"
        and rtf_ablation_profile == "navigation_fast"
    ),
    "nvblox_mode_allowed": nvblox_mode in {
        "off", "shadow", "active_local_gt",
    },
    "nvblox_exact_scope": nvblox_mode == "off" or (
        lane == "a"
        and (
            (nvblox_mode == "shadow" and profile in {"screen3", "fixed5"})
            or (nvblox_mode == "active_local_gt" and profile == "screen3")
        )
        and rtf_ablation_profile in {"navigation_fast", "off"}
        and isaac_sensor_profile == "baseline"
    ),
    "run_mode_allowed": run_mode in {"model", "oracle"},
    "system2_replan_policy_allowed": system2_replan_policy in {
        "strict", "observation_bound", "raw_wire_warn",
    },
    "system2_replan_policy_exact_scope": (
        system2_replan_policy == "observation_bound"
        or (
            lane == "a" and run_mode == "model"
            and candidate.get("effective_candidate_profile") == "recovery_a"
        )
    ),
    "oracle_exact_scope": run_mode == "model" or (
        lane == "a" and profile == "screen3"
        and candidate_profile == "baseline"
        and nvblox_mode == "active_local_gt"
    ),
    "execution_profile_allowed": profile in {
        "canary60", "soak600", "screen1", "screen3", "fixed5"
    },
    "screen_count_matches_profile": screen_count == {
        "screen1": 1, "screen3": 3
    }.get(profile, 0),
    "screen_episode_key_exact_scope": (
        not screen_episode_key
        or (
            profile == "screen1"
            and isinstance(expected_keys, list)
            and screen_episode_key in expected_keys
            and re.fullmatch(r"[A-Za-z0-9_.-]+", screen_episode_key)
                is not None
        )
    ),
    "preparation_pass": prep.get("status") == "PASS"
        and bool(prep.get("checks")) and all(prep["checks"].values()),
    "final_pass": final.get("status") == "PASS"
        and bool(final.get("checks")) and all(final["checks"].values()),
    "prepare_scope_compatible": prepare_scope in {"dual", "lane-a", "lane-b"}
        and lane in prepared_lanes,
    "exact_code_ref": prep.get("code_ref_sha") == code_sha
        and final.get("code_ref_sha") == code_sha
        and frozen_input.get("code_ref_sha") == code_sha,
    "summary_binding": final.get("preparation_summary_sha256")
        == hashlib.sha256(prep_path.read_bytes()).hexdigest(),
    "fast_prepare_input_binding": prep.get("fast_prepare_input_sha256")
        == fast_prepare_input_sha256
        and final.get("fast_prepare_input_sha256") == fast_prepare_input_sha256,
    "deployment_roots_binding": final.get("deployment_roots") == roots,
    "dataset_pass": dataset.get("status") == "PASS"
        and bool(dataset.get("checks")) and all(dataset["checks"].values())
        and dataset.get("episode_count") == 5,
    "dataset_audit_sha_binding": safe_sha.fullmatch(dataset_audit_sha256) is not None
        and prep.get("dataset_audit_sha256") == dataset_audit_sha256
        and final.get("dataset_audit_sha256") == dataset_audit_sha256,
    "dataset_sha_frozen_binding": safe_sha.fullmatch(
        str(frozen_input.get("dataset_file_sha256", ""))) is not None
        and dataset.get("dataset_sha256") == frozen_input.get("dataset_file_sha256")
        and prep.get("dataset_file_sha256") == frozen_input.get("dataset_file_sha256")
        and final.get("dataset_file_sha256") == frozen_input.get("dataset_file_sha256"),
    "episode_keys_frozen_binding": isinstance(expected_keys, list)
        and len(expected_keys) == frozen_input.get("episode_count") == 5
        and len(set(expected_keys)) == 5
        and dataset.get("episode_keys") == expected_keys
        and prep.get("episode_keys") == expected_keys
        and final.get("episode_keys") == expected_keys
        and prep.get("episode_count") == final.get("episode_count") == 5,
    "map_sha": safe_sha.fullmatch(str(prep.get("static_map_manifest_sha256", ""))) is not None,
    "dgx_root_safe": isinstance(dgx_root, str) and dgx_root.startswith(dgx_prefix)
        and "/../" not in dgx_root + "/",
    "x86_root_safe": isinstance(x86_root, str) and x86_root.startswith(x86_prefix)
        and "/../" not in x86_root + "/",
    "dataset_root_exact": dataset_root == frozen_input.get("dataset_root"),
    "dataset_root_safe": dataset_root.startswith("/home/song/internnav-t1-t2/episodes/")
        and "/../" not in dataset_root + "/",
}
payload = {
    "schema_version": 1,
    "status": "PASS" if all(checks.values()) else "FAIL",
    "lane": lane,
    "prepare_scope": prepare_scope,
    "prepared_lanes": prepared_lanes,
    "dataset_audit_relative_path": dataset_relative.as_posix(),
    "code_ref_sha": code_sha,
    "candidate_profile": candidate_profile,
    "candidate_resolution_sha256": declared_resolution_sha256,
    "candidate_resolution_file_sha256": hashlib.sha256(
        candidate_path.read_bytes()).hexdigest(),
    "candidate_binding": candidate_binding,
    "rtf_ablation_profile": rtf_ablation_profile,
    "isaac_sensor_profile": isaac_sensor_profile,
    "nvblox_mode": nvblox_mode,
    "run_mode": run_mode,
    "system2_replan_policy": system2_replan_policy,
    "fault_injection_profile": fault_injection_profile,
    "live_frontier_capture": live_frontier_capture,
    "live_frontier_ready_required": live_frontier_ready_required,
    "execution_profile": profile,
    "execution_episode_count": screen_count or dataset.get("episode_count"),
    "execution_episode_keys": (
        [screen_episode_key] if screen_episode_key
        else expected_keys[:screen_count] if screen_count
        else expected_keys
    ),
    "screen_episode_key": screen_episode_key or None,
    "deployment_roots": {"dgx": dgx_root, "x86": x86_root},
    "dataset_root": dataset_root,
    "dataset_sha256": dataset.get("dataset_sha256"),
    "static_map_manifest_sha256": prep.get("static_map_manifest_sha256"),
    "episode_count": dataset.get("episode_count"),
    "episode_keys": dataset.get("episode_keys"),
    "source_receipts": {
        "prepare_summary_sha256": hashlib.sha256(prep_path.read_bytes()).hexdigest(),
        "prepare_final_sha256": hashlib.sha256(final_path.read_bytes()).hexdigest(),
        "fast_prepare_input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "dataset_audit_sha256": dataset_audit_sha256,
    },
    "checks": checks,
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if payload["status"] != "PASS":
    raise SystemExit(f"fast-lane preparation binding failed: {checks}")
PY
fi

mapfile -t binding < <(python3 - "$validation_tmp/input_binding.json" <<'PY'
import json, sys
value=json.load(open(sys.argv[1],encoding="utf-8"))
for item in (value["deployment_roots"]["dgx"], value["deployment_roots"]["x86"],
             value["dataset_root"], value["dataset_sha256"],
             value["static_map_manifest_sha256"],
             value["candidate_resolution_sha256"],
             value.get("static_map_manifest_path", ""),
             value["execution_episode_count"]): print(item)
PY
)
dgx_root="${binding[0]}"
x86_root="${binding[1]}"
dataset_root="${binding[2]}"
dataset_sha256="${binding[3]}"
map_manifest_sha256="${binding[4]}"
candidate_resolution_sha256="${binding[5]}"
[[ "$candidate_resolution_sha256" =~ ^[0-9a-f]{64}$ ]]
map_manifest="${binding[6]}"
if test -z "$map_manifest"; then
  map_manifest="$dgx_root/inputs/d0_fixed5_static_maps/manifest.json"
fi
execution_episode_count="${binding[7]}"
[[ "$execution_episode_count" =~ ^[1-9][0-9]*$ ]]
episode_keys_csv="$(python3 - "$validation_tmp/input_binding.json" <<'PY'
import json,sys
value=json.load(open(sys.argv[1],encoding="utf-8"))
print(",".join(value["episode_keys"]))
PY
)"
execution_episode_keys_csv="$(python3 - "$validation_tmp/input_binding.json" <<'PY'
import json,sys
value=json.load(open(sys.argv[1],encoding="utf-8"))
print(",".join(value["execution_episode_keys"]))
PY
)"
python3 - "$execution_episode_keys_csv" "$execution_episode_count" <<'PY'
import re,sys
keys=sys.argv[1].split(",")
assert len(keys)==int(sys.argv[2])
assert len(set(keys))==len(keys)
assert all(re.fullmatch(r"[A-Za-z0-9_.-]+",key) for key in keys)
PY
remote_run_name="t5_fast_${profile}_${run_id}"
dgx_run="$dgx_root/results/$remote_run_name"
x86_run="$x86_root/results/$remote_run_name"
dgx_supervisor_ledger="${dgx_run}.supervisor.json"
x86_supervisor_ledger="${x86_run}.supervisor.json"

umask 077
mkdir "$result_dir"
mkdir "$result_dir/logs" "$result_dir/audits" "$result_dir/remote"
cp "$validation_tmp/input_binding.json" "$result_dir/input_binding.json"
cp "$validation_tmp/candidate_resolution.json" \
  "$result_dir/candidate_resolution.json"
rm -rf -- "$validation_tmp"
trap - EXIT

credentials="$root/.env.local"
test -f "$credentials"
set +a
source "$credentials"
: "${HF_TOKEN:?HF_TOKEN is required in .env.local}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
[[ "$HF_TOKEN" =~ ^hf_[A-Za-z0-9]{20,}$ ]]
[[ "$HF_ENDPOINT" =~ ^https://hf-mirror\.com/?$ ]]
export -n HF_TOKEN HUGGING_FACE_HUB_TOKEN DGX_A_PASSWORD DGX_B_PASSWORD \
  ISAAC_X86_PASSWORD 2>/dev/null || true

source "$root/scripts/t5_quarantine_common.sh"
source "$root/scripts/t5_remote_compute_audit_common.sh"
ssh_options=(-T -i "${INTERNNAV_T5_SSH_IDENTITY_FILE:-$HOME/.ssh/id_ed25519_internnav_runtime}" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=2)
x86_target="song@$x86_ip"
remote() { local target="$1"; shift; ssh "${ssh_options[@]}" "$target" "$@"; }

dgx_quarantine_file=/tmp/internnav_dgx.quarantine
x86_quarantine_file="/tmp/internnav_isaac_gpu${gpu}.quarantine"
quarantine_run_tag="fast:${profile}:${run_id}"
dgx_quarantine_armed=false
x86_quarantine_armed=false
dgx_launch_attempted=0
x86_launch_attempted=0
container_started=0
dgx_ssh_pid=""
x86_ssh_pid=""
fault_director_pid=""
run_completed=0
canary_window_completed=0
cleanup_timeout="${INTERNNAV_T5_FAST_CLEANUP_TIMEOUT_SEC:-30}"
cleanup_kill_timeout="${INTERNNAV_T5_FAST_CLEANUP_KILL_TIMEOUT_SEC:-10}"
[[ "$cleanup_timeout" =~ ^[1-9][0-9]*$ ]]
[[ "$cleanup_kill_timeout" =~ ^[1-9][0-9]*$ ]]
(( cleanup_timeout <= 60 && cleanup_kill_timeout <= 60 )) || {
  echo "fast-lane TERM and KILL waits must each be <= 60 seconds" >&2
  exit 64
}

read -r -d '' supervisor_program <<'REMOTE_SUPERVISOR_CONTROL' || true
import hashlib, json, os, signal, sys, time
from pathlib import Path

ledger_path, expected_root, action = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
value = json.loads(ledger_path.read_text(encoding="utf-8"))
if value.get("run_root") != expected_root:
    raise SystemExit("supervisor run_root mismatch")
pid, pgid, sid, starttime = (
    value.get("pid"), value.get("pgid"), value.get("sid"), value.get("starttime")
)
argv_sha256 = value.get("argv_sha256")
if not all(isinstance(item, int) and item > 1 for item in (pid, pgid, sid, starttime)):
    raise SystemExit("invalid supervisor identity")
if pgid != sid or not isinstance(argv_sha256, str) or len(argv_sha256) != 64:
    raise SystemExit("unsafe supervisor session identity")

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

process_table = []
for entry in Path("/proc").iterdir():
    if not entry.name.isdigit():
        continue
    candidate = int(entry.name)
    if candidate in ancestors:
        continue
    try:
        raw_stat = (entry / "stat").read_text(encoding="utf-8")
        stat_tail = raw_stat.rsplit(")", 1)[1].strip().split()
        raw_command = (entry / "cmdline").read_bytes()
        process_table.append({
            "pid": candidate,
            "pgid": int(stat_tail[2]),
            "sid": int(stat_tail[3]),
            "starttime": int(stat_tail[19]),
            "command": raw_command.replace(b"\0", b" ").decode(errors="replace").strip(),
            "argv_sha256": hashlib.sha256(raw_command).hexdigest(),
        })
    except (OSError, ValueError, IndexError):
        continue

leaders = [row for row in process_table if row["pid"] == pid]
leader = leaders[0] if leaders else None
if leader is not None and (
    leader["pgid"] != pgid
    or leader["sid"] != sid
    or leader["starttime"] != starttime
    or expected_root not in leader["command"]
):
    raise SystemExit("supervisor identity drift")
# Bash may legally exec its final runtime command in place.  PID, PGID, SID
# and /proc starttime still bind the same process instance, while the current
# command must remain scoped to the immutable run root.  Preserve both hashes
# as evidence, but do not mistake that expected exec transition for PID reuse.
leader_argv_matches_ledger = (
    leader is None or leader["argv_sha256"] == argv_sha256
)

members = [row for row in process_table if row["pgid"] == pgid]
run_processes = [row for row in process_table if expected_root in row["command"]]
if members and not any(expected_root in row["command"] for row in members):
    raise SystemExit("refusing reused/unassociated supervisor PGID")

ledger_pgids = set()
runtime_ledger = Path(expected_root) / "pid_ledger.jsonl"
runtime_ledger_error = None
if runtime_ledger.is_file():
    try:
        rows = [json.loads(line) for line in runtime_ledger.read_text(
            encoding="utf-8").splitlines() if line.strip()]
        absent_ids = {
            (row.get("scope", "host"), row.get("component"), row.get("pid"))
            for row in rows if row.get("event") == "verified_absent"
        }
        for row in rows:
            key = (row.get("scope", "host"), row.get("component"), row.get("pid"))
            candidate_pgid = row.get("pgid")
            if (row.get("event") == "started" and key not in absent_ids
                    and row.get("scope", "host") == "host"
                    and isinstance(candidate_pgid, int) and candidate_pgid > 1):
                ledger_pgids.add(candidate_pgid)
    except (OSError, ValueError, TypeError) as error:
        runtime_ledger_error = type(error).__name__
managed_groups = {
    candidate: [row for row in process_table if row["pgid"] == candidate]
    for candidate in sorted(ledger_pgids)
}

def read_process_identity(candidate):
    entry = Path(f"/proc/{candidate}")
    try:
        raw_stat = (entry / "stat").read_text(encoding="utf-8")
        stat_tail = raw_stat.rsplit(")", 1)[1].strip().split()
        raw_command = (entry / "cmdline").read_bytes()
        return {
            "pid": candidate,
            "pgid": int(stat_tail[2]),
            "sid": int(stat_tail[3]),
            "starttime": int(stat_tail[19]),
            "command": raw_command.replace(b"\0", b" ").decode(
                errors="replace").strip(),
            "argv_sha256": hashlib.sha256(raw_command).hexdigest(),
        }
    except (OSError, ValueError, IndexError):
        return None

def read_group_members(candidate_pgid):
    current = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        row = read_process_identity(int(entry.name))
        if row is not None and row["pgid"] == candidate_pgid:
            current.append(row)
    return current

signal_identity_checks = []

def signal_verified_group(candidate_pgid, expected_anchor, anchor_source, label,
                          requested_signal):
    # The earlier process table is discovery evidence only.  Re-read both the
    # anchored process instance and its process group immediately before
    # killpg so PID/PGID reuse or an exec outside this run fails closed.
    current_anchor = read_process_identity(expected_anchor["pid"])
    current_members = read_group_members(candidate_pgid)
    if not current_members:
        signal_identity_checks.append({
            "label": label, "pgid": candidate_pgid,
            "anchor_source": anchor_source, "status": "already_absent",
        })
        return False
    identity_fields = ("pid", "pgid", "sid", "starttime")
    if current_anchor is None or any(
        current_anchor[field] != expected_anchor[field]
        for field in identity_fields
    ):
        raise SystemExit(f"{label} identity drift before signal")
    if current_anchor["pgid"] != candidate_pgid:
        raise SystemExit(f"{label} PGID drift before signal")
    if expected_root not in current_anchor["command"]:
        raise SystemExit(f"{label} run_root drift before signal")
    if any(row["sid"] != current_anchor["sid"] for row in current_members):
        raise SystemExit(f"{label} SID drift before signal")
    if not any(expected_root in row["command"] for row in current_members):
        raise SystemExit(f"{label} group run_root drift before signal")
    signal_identity_checks.append({
        "label": label,
        "pgid": candidate_pgid,
        "anchor_source": anchor_source,
        "status": "verified",
        "anchor": current_anchor,
        "member_pids": sorted(row["pid"] for row in current_members),
    })
    os.killpg(candidate_pgid, requested_signal)
    return True

if action not in {"AUDIT", "TERM", "KILL"}:
    raise SystemExit("unsupported supervisor action")
requested_signal = signal.SIGTERM if action == "TERM" else signal.SIGKILL
signalled_groups = []
if action in {"TERM", "KILL"} and members:
    if leader is not None:
        supervisor_anchor = {
            "pid": pid, "pgid": pgid, "sid": sid, "starttime": starttime,
        }
        supervisor_anchor_source = "immutable_supervisor_ledger"
    else:
        fallback_anchors = [
            row for row in members
            if row["pgid"] == pgid and row["sid"] == sid
            and expected_root in row["command"]
        ]
        if not fallback_anchors:
            raise SystemExit("supervisor leader absent without associated member anchor")
        supervisor_anchor = fallback_anchors[0]
        supervisor_anchor_source = "discovered_live_group_member"
    if signal_verified_group(
        pgid, supervisor_anchor, supervisor_anchor_source, "supervisor",
        requested_signal
    ):
        signalled_groups.append(pgid)
if action in {"TERM", "KILL"} and (action == "KILL" or not members):
    # A runtime may create independent setsid children.  Once the supervisor
    # leader is gone, the run-root and pid ledger are the only admissible
    # fallback; every live group must still contain this immutable run root.
    direct_groups = set(managed_groups)
    direct_groups.update(row["pgid"] for row in run_processes)
    for candidate in sorted(direct_groups):
        rows = [row for row in process_table if row["pgid"] == candidate]
        if not rows or candidate == pgid:
            continue
        if not any(expected_root in row["command"] for row in rows):
            raise SystemExit(f"refusing unassociated managed PGID {candidate}")
        anchors = [
            row for row in rows
            if row["pid"] == candidate and expected_root in row["command"]
        ]
        if not anchors:
            anchors = [row for row in rows if expected_root in row["command"]]
        if not anchors:
            raise SystemExit(f"refusing unanchored managed PGID {candidate}")
        if signal_verified_group(
            candidate, anchors[0], "discovered_managed_group_member",
            f"managed PGID {candidate}", requested_signal
        ):
            signalled_groups.append(candidate)

payload = {
    "schema_version": 1,
    "ledger": str(ledger_path),
    "run_root": expected_root,
    "action": action,
    "supervisor": {"pid": pid, "pgid": pgid, "sid": sid,
                   "starttime": starttime, "argv_sha256": argv_sha256},
    "leader_argv_matches_ledger": leader_argv_matches_ledger,
    "leader": leader,
    "members": members,
    "run_processes": run_processes,
    "managed_groups": managed_groups,
    "runtime_ledger_error": runtime_ledger_error,
    "signal_identity_checks": signal_identity_checks,
    "signalled_groups": signalled_groups,
    "absent": runtime_ledger_error is None and not members and not run_processes
        and not any(managed_groups.values()),
    "recorded_unix": time.time(),
}
print(json.dumps(payload, sort_keys=True))
REMOTE_SUPERVISOR_CONTROL
supervisor_program_b64="$(printf '%s' "$supervisor_program" | base64 | tr -d '\r\n')"

supervisor_action() {
  local target="$1" ledger="$2" run_root="$3" action="$4" output rc
  output="$(remote "$target" \
    "python3 -c \"\$(printf '%s' '$supervisor_program_b64' | base64 -d)\" '$ledger' '$run_root' '$action'" 2>&1)" && rc=0 || rc=$?
  printf '%s\n' "$output" >>"$result_dir/audits/coordinator_cleanup_events.jsonl"
  printf '%s\n' "$output"
  return "$rc"
}

supervisor_absent() {
  local output
  output="$(supervisor_action "$1" "$2" "$3" AUDIT)" || return 1
  python3 -c 'import json,sys;raise SystemExit(0 if json.loads(sys.argv[1]).get("absent") else 1)' "$output"
}

run_scope_absent() {
  local target="$1" scopes="$2"
  remote "$target" "python3 - '$scopes'" <<'PY'
import os, sys
from pathlib import Path
scopes=tuple(value for value in sys.argv[1].split("|") if value)
if not scopes: raise SystemExit(75)
ancestors=set(); cursor=os.getpid()
while cursor > 1 and cursor not in ancestors:
    ancestors.add(cursor)
    try:
        lines=Path(f"/proc/{cursor}/status").read_text().splitlines()
        cursor=int(next(line.split()[1] for line in lines if line.startswith("PPid:")))
    except (OSError, StopIteration, ValueError): break
found=[]
for entry in Path("/proc").iterdir():
    if not entry.name.isdigit() or int(entry.name) in ancestors: continue
    try: command=(entry/"cmdline").read_bytes().replace(b"\0",b" ").decode(errors="replace")
    except OSError: continue
    if any(scope in command for scope in scopes): found.append(int(entry.name))
raise SystemExit(0 if not found else 75)
PY
}

collect_remote_tree() {
  local target="$1" source="$2" destination="$3"
  test ! -e "$destination" || return 0
  remote "$target" "test -d '$source'" >/dev/null 2>&1 || return 0
  mkdir "$destination"
  if remote "$target" "tar -C '$source' -czf - ." >"$destination.tar.gz" 2>/dev/null; then
    tar -C "$destination" -xzf "$destination.tar.gz"
  fi
}

# The two Isaac workers share one SSD.  During concurrent engineering runs do
# not tar an entire x86 result tree: retain the complete raw tree remotely and
# transfer only machine-readable evidence/records needed for live decisions.
# Full log/video archival is a later serialized shared-I/O operation.
collect_x86_machine_records() {
  local source="$1" destination="$2"
  test ! -e "$destination" || return 0
  remote "$x86_target" "test -d '$source'" >/dev/null 2>&1 || return 0
  mkdir "$destination"
  if remote "$x86_target" \
      "set -euo pipefail; cd '$source'; find . -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.txt' \) -print0 | tar --null -T - -czf -" \
      >"$destination.machine-records.tar.gz" 2>/dev/null; then
    tar -C "$destination" -xzf "$destination.machine-records.tar.gz"
  fi
}

# The fixed-five Rev-C bundle is 20 Step3 keyframes plus up to five optional
# review-only third-person frames, not a video/full-result archive.
# Pull only the five frozen snapshot directories so the local materializer can
# consume them without a later shared-SSD archive pass.
collect_revc_fixed5_snapshots() {
  local source="$x86_run/evaluator/revc_snapshots"
  local destination="$result_dir/remote/x86/evaluator/revc_snapshots"
  local archive="$result_dir/revc_fixed5_snapshots.tar.gz"
  test "$isaac_sensor_profile" = lane_b_revc_fixed5_capture || return 0
  remote "$x86_target" \
    "set -euo pipefail; test -d '$source'; test ! -L '$source'; test -z \"\$(find '$source' -type l -print -quit)\"; test \"\$(find '$source' -mindepth 1 -maxdepth 1 -type d | wc -l)\" = 5; test \"\$(find '$source' -mindepth 2 -maxdepth 2 -type f -name snapshot.json | wc -l)\" = 5; png_count=\$(find '$source' -mindepth 2 -maxdepth 2 -type f -name '*.png' | wc -l); test \"\$png_count\" -ge 20; test \"\$png_count\" -le 25; cd '$source'; find . -mindepth 2 -maxdepth 2 -type f \( -name snapshot.json -o -name '*.png' \) -print0 | sort -z | tar --null -T - -czf -" \
    >"$archive"
  mkdir -p "$destination"
  tar -C "$destination" -xzf "$archive"
  python3 - "$destination" \
    "$result_dir/audits/revc_fixed5_scoped_pull.json" \
    "$result_dir/remote/x86/evaluator/revc_fixed5_capture.json" <<'PY'
import hashlib, json, os, sys, time
from pathlib import Path

root, output, capture_path = map(Path, sys.argv[1:4])
root = root.resolve()
if capture_path.is_symlink() or not capture_path.is_file():
    raise SystemExit("fixed-five capture summary is not a regular file")
capture = json.loads(capture_path.read_text(encoding="utf-8"))
capture_rows = capture.get("snapshots")
if capture.get("status") != "PASS" or not isinstance(capture_rows, list):
    raise SystemExit("fixed-five capture summary is not PASS")
capture_by_sidecar = {
    row.get("sidecar"): row for row in capture_rows if isinstance(row, dict)
}
sidecars = sorted(root.glob("*/snapshot.json"))
pngs = sorted(root.glob("*/*.png"))
identities = []
referenced = []
summary_bindings = []
observer_count = 0
for sidecar_path in sidecars:
    if sidecar_path.is_symlink() or not sidecar_path.is_file():
        raise SystemExit("scoped pull contains a non-regular sidecar")
    value = json.loads(sidecar_path.read_text(encoding="utf-8"))
    identity = (
        value.get("episode_id"), value.get("reset_generation"),
        value.get("sequence_id"),
    )
    identities.append(identity)
    relative_sidecar = sidecar_path.relative_to(root.parent).as_posix()
    capture_row = capture_by_sidecar.get(relative_sidecar)
    if not isinstance(capture_row, dict):
        raise SystemExit("scoped pull sidecar is absent from capture summary")
    if capture_row.get("sidecar_sha256") != hashlib.sha256(
        sidecar_path.read_bytes()
    ).hexdigest():
        raise SystemExit("scoped pull sidecar SHA differs from capture summary")
    request = capture_row.get("request")
    if not isinstance(request, dict) or identity != (
        request.get("episode_id"), request.get("reset_generation"),
        request.get("sequence_id"),
    ):
        raise SystemExit("scoped pull identity differs from capture summary")
    cameras = value.get("cameras")
    if not isinstance(cameras, list) or len(cameras) != 4:
        raise SystemExit("scoped pull sidecar does not bind four cameras")
    image_bindings = []
    for camera in cameras:
        relative = camera.get("path") if isinstance(camera, dict) else None
        if not isinstance(relative, str) or not relative.startswith("revc_snapshots/"):
            raise SystemExit("scoped pull camera path is invalid")
        image = (root.parent / relative).resolve()
        try:
            image.relative_to(root)
        except ValueError:
            raise SystemExit("scoped pull camera path escapes or is not regular")
        if image.is_symlink() or not image.is_file():
            raise SystemExit("scoped pull camera path escapes or is not regular")
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        if camera.get("sha256") != digest:
            raise SystemExit("scoped pull camera SHA differs from its sidecar")
        referenced.append(image)
        image_bindings.append({
            "identity": camera.get("identity"),
            "path": relative,
            "sha256": digest,
            "bytes": image.stat().st_size,
        })
    if capture_row.get("images") != image_bindings:
        raise SystemExit("scoped pull images differ from capture summary")
    observer = value.get("observer")
    if isinstance(observer, dict) and observer.get("status") == "CAPTURED":
        relative = observer.get("path")
        if not isinstance(relative, str) or not relative.startswith(
            "revc_snapshots/"
        ):
            raise SystemExit("scoped pull observer path is invalid")
        image = (root.parent / relative).resolve()
        try:
            image.relative_to(root)
        except ValueError:
            raise SystemExit("scoped pull observer path escapes")
        if image.is_symlink() or not image.is_file():
            raise SystemExit("scoped pull observer is not a regular file")
        if observer.get("sha256") != hashlib.sha256(
            image.read_bytes()
        ).hexdigest():
            raise SystemExit("scoped pull observer SHA differs from sidecar")
        referenced.append(image)
        observer_count += 1
    summary_bindings.append(relative_sidecar)
checks = {
    "snapshot_directory_count_five": len([p for p in root.iterdir() if p.is_dir()]) == 5,
    "snapshot_sidecar_count_five": len(sidecars) == 5,
    "png_count_four_views_plus_optional_observer": (
        len(pngs) == 20 + observer_count and 0 <= observer_count <= 5
    ),
    "execution_identities_unique": len(identities) == len(set(identities)) == 5,
    "sidecar_references_exact_png_set": set(referenced) == set(pngs),
    "capture_summary_exact_binding": (
        len(capture_by_sidecar) == 5
        and set(summary_bindings) == set(capture_by_sidecar)
        and [row.get("ordered_episode_id") for row in capture_rows]
            == capture.get("ordered_episode_ids")
    ),
    "no_symlinks": not any(path.is_symlink() for path in root.rglob("*")),
}
payload = {
    "schema_version": 1,
    "status": "PASS" if all(checks.values()) else "FAIL",
    "profile": "lane_b_revc_fixed5_capture",
    "local_snapshot_root": str(root),
    "snapshot_count": len(sidecars),
    "png_count": len(pngs),
    "capture_summary_sha256": hashlib.sha256(capture_path.read_bytes()).hexdigest(),
    "checks": checks,
    "recorded_unix": time.time(),
}
temporary = output.with_name("." + output.name + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, output)
raise SystemExit(0 if payload["status"] == "PASS" else 75)
PY
}

request_stop_and_audit() {
  local dgx_absent=false x86_absent=false dgx_scoped=false x86_scoped=false
  local container_clean=false dgx_ports_clean=false x86_lane_clean=false
  local dgx_compute_absent=false cleanup_rc=0
  set +e
  remote "$dgx_target" "cat '$dgx_supervisor_ledger'" \
    >"$result_dir/audits/dgx_supervisor_ledger.json" 2>/dev/null || true
  remote "$x86_target" "cat '$x86_supervisor_ledger'" \
    >"$result_dir/audits/x86_supervisor_ledger.json" 2>/dev/null || true
  # Signal both linked remote sessions once before waiting.  Do not combine a
  # stop.request with TERM here: the request can move the child into cleanup
  # before TERM arrives and make the second signal interrupt that cleanup.
  # One shared TERM window and one shared KILL evidence window stay inside the
  # outer 60-second lease TERM budget; one unresponsive host cannot consume a
  # second full per-host timeout.
  for specification in \
    "$dgx_target|$dgx_supervisor_ledger|$dgx_run" \
    "$x86_target|$x86_supervisor_ledger|$x86_run"; do
    IFS='|' read -r target ledger run_root <<<"$specification"
    remote "$target" "test -f '$ledger'" >/dev/null 2>&1 || continue
    supervisor_action "$target" "$ledger" "$run_root" TERM >/dev/null 2>&1 || true
  done
  cleanup_deadline=$((SECONDS + cleanup_timeout))
  while (( SECONDS < cleanup_deadline )); do
    residual=0
    for specification in \
      "$dgx_target|$dgx_supervisor_ledger|$dgx_run" \
      "$x86_target|$x86_supervisor_ledger|$x86_run"; do
      IFS='|' read -r target ledger run_root <<<"$specification"
      remote "$target" "test -f '$ledger'" >/dev/null 2>&1 || continue
      supervisor_absent "$target" "$ledger" "$run_root" >/dev/null 2>&1 || \
        residual=$((residual + 1))
    done
    test "$residual" = 0 && break
    sleep 1
  done
  for specification in \
    "$dgx_target|$dgx_supervisor_ledger|$dgx_run" \
    "$x86_target|$x86_supervisor_ledger|$x86_run"; do
    IFS='|' read -r target ledger run_root <<<"$specification"
    remote "$target" "test -f '$ledger'" >/dev/null 2>&1 || continue
    supervisor_absent "$target" "$ledger" "$run_root" >/dev/null 2>&1 || \
      supervisor_action "$target" "$ledger" "$run_root" KILL >/dev/null 2>&1 || true
  done
  kill_deadline=$((SECONDS + cleanup_kill_timeout))
  while (( SECONDS < kill_deadline )); do
    residual=0
    for specification in \
      "$dgx_target|$dgx_supervisor_ledger|$dgx_run" \
      "$x86_target|$x86_supervisor_ledger|$x86_run"; do
      IFS='|' read -r target ledger run_root <<<"$specification"
      remote "$target" "test -f '$ledger'" >/dev/null 2>&1 || continue
      supervisor_absent "$target" "$ledger" "$run_root" >/dev/null 2>&1 || \
        residual=$((residual + 1))
    done
    test "$residual" = 0 && break
    sleep 1
  done
  for pid in "$x86_ssh_pid" "$dgx_ssh_pid"; do
    test -z "$pid" || ! kill -0 "$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  done
  if test "$container_started" != 0; then
    remote "$x86_target" \
      "set +e; test \"\$(docker inspect -f '{{index .Config.Labels \"internnav.t5.deployment_root\"}}' '$container')\" = '$x86_root' || exit 75; docker stop -t 20 '$container' >/dev/null 2>&1; test \"\$(docker inspect -f '{{.State.Running}}' '$container')\" = false; test \"\$(docker inspect -f '{{.State.Pid}}' '$container')\" = 0" \
      >/dev/null 2>&1 || true
    container_started=0
  fi
  if remote "$dgx_target" "test -f '$dgx_supervisor_ledger'" >/dev/null 2>&1; then
    supervisor_absent "$dgx_target" "$dgx_supervisor_ledger" "$dgx_run" >/dev/null 2>&1 && dgx_absent=true
  elif test "$dgx_launch_attempted" = 0; then dgx_absent=true; fi
  if remote "$x86_target" "test -f '$x86_supervisor_ledger'" >/dev/null 2>&1; then
    supervisor_absent "$x86_target" "$x86_supervisor_ledger" "$x86_run" >/dev/null 2>&1 && x86_absent=true
  elif test "$x86_launch_attempted" = 0; then x86_absent=true; fi
  run_scope_absent "$dgx_target" "$dgx_run|$dgx_root" >/dev/null 2>&1 && dgx_scoped=true
  run_scope_absent "$x86_target" "$x86_run|$x86_root" >/dev/null 2>&1 && x86_scoped=true
  remote "$x86_target" \
    "test \"\$(docker inspect -f '{{index .Config.Labels \"internnav.t5.deployment_root\"}}' '$container')\" = '$x86_root'; test \"\$(docker inspect -f '{{.State.Running}}' '$container')\" = false; test \"\$(docker inspect -f '{{.State.Pid}}' '$container')\" = 0" \
    >/dev/null 2>&1 && container_clean=true
  remote "$dgx_target" \
    "for p in $own_ports $step3_dgx_ports; do test -z \"\$(ss -H -lntup|grep -E \"[:.]\$p[[:space:]]\"||true)\" || exit 1; done" \
    >/dev/null 2>&1 && dgx_ports_clean=true
  remote "$x86_target" \
    "set -e; test ! -e '$ipc_alias' && test ! -L '$ipc_alias'; test ! -e '$lane_runtime_lock' || flock -n '$lane_runtime_lock' true; test ! -d '$x86_root/runtime/t4_ipc' || test -z \"\$(find '$x86_root/runtime/t4_ipc' -maxdepth 1 -type s -print -quit)\"; for p in $own_ports; do test -z \"\$(ss -H -lntup|grep -E \"[:.]\$p[[:space:]]\"||true)\" || exit 1; done" \
    >/dev/null 2>&1 && x86_lane_clean=true
  t5_remote_compute_absent "$dgx_target" \
    "$result_dir/audits/dgx_structured_compute_poststop.json" \
    >/dev/null 2>&1 && dgx_compute_absent=true
  python3 - "$result_dir/audits/coordinator_cleanup_receipt.json" \
    "$dgx_absent" "$x86_absent" "$dgx_scoped" "$x86_scoped" \
    "$container_clean" "$dgx_ports_clean" "$x86_lane_clean" \
    "$dgx_compute_absent" <<'PY'
import json, sys, time
from pathlib import Path
checks = {
    "dgx_supervisor_pgid_absent": sys.argv[2] == "true",
    "x86_supervisor_pgid_absent": sys.argv[3] == "true",
    "dgx_deployment_processes_absent": sys.argv[4] == "true",
    "x86_deployment_processes_absent": sys.argv[5] == "true",
    "own_isaac_container_stopped_pid_zero": sys.argv[6] == "true",
    "own_dgx_ports_zero": sys.argv[7] == "true",
    "own_x86_ports_socket_runtime_lock_zero": sys.argv[8] == "true",
    "dgx_structured_compute_absent": sys.argv[9] == "true",
}
payload={"schema_version":1,"status":"PASS" if all(checks.values()) else "FAIL",
         "checks":checks,"lane_scoped":True,"other_lane_quiet_required":False,
         "recorded_unix":time.time()}
Path(sys.argv[1]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
raise SystemExit(0 if payload["status"] == "PASS" else 75)
PY
  cleanup_rc=$?
  set -e
  return "$cleanup_rc"
}

finish() {
  local incoming=$?
  trap - EXIT INT TERM HUP
  if test -n "$fault_director_pid" && kill -0 "$fault_director_pid" 2>/dev/null; then
    kill -TERM -- "-$fault_director_pid" 2>/dev/null || true
    for _ in $(seq 1 100); do
      kill -0 "$fault_director_pid" 2>/dev/null || break
      sleep 0.1
    done
    kill -0 "$fault_director_pid" 2>/dev/null && \
      kill -KILL -- "-$fault_director_pid" 2>/dev/null || true
    wait "$fault_director_pid" 2>/dev/null || true
    fault_director_pid=""
    incoming=1
  fi
  request_stop_and_audit || incoming=1
  set +e
  cleanup_receipt="$result_dir/audits/coordinator_cleanup_receipt.json"
  if test "$dgx_quarantine_armed" = true; then
    t5_quarantine_owned_clear "$dgx_target" "$dgx_quarantine_file" \
      "$dgx_quarantine_role" "$quarantine_run_tag" "$dgx_root" \
      "$cleanup_receipt" "$result_dir/audits/dgx_quarantine_clear.json" || incoming=1
  fi
  if test "$x86_quarantine_armed" = true; then
    t5_quarantine_owned_clear "$x86_target" "$x86_quarantine_file" \
      "$x86_quarantine_role" "$quarantine_run_tag" "$x86_root" \
      "$cleanup_receipt" "$result_dir/audits/x86_quarantine_clear.json" || incoming=1
  fi
  collect_remote_tree "$dgx_target" "$dgx_run" "$result_dir/remote/dgx"
  collect_x86_machine_records "$x86_run" "$result_dir/remote/x86"
  collect_revc_fixed5_snapshots || incoming=1
  case "$rtf_ablation_profile" in
    navigation_fast|off) ;;
    *)
      python3 "$root/scripts/summarize_t5_rtf_ablation.py" \
        --result-root "$result_dir" \
        --output "$result_dir/rtf_ablation_summary.json" || incoming=1
      ;;
  esac
  python3 - "$result_dir/fast_lane_summary.json" "$lane" "$profile" \
    "$run_id" "$code_sha" "$resource_profile" "$run_completed" \
    "$canary_window_completed" "$incoming" "$candidate_profile" \
    "$rtf_ablation_profile" "$isaac_sensor_profile" "$nvblox_mode" \
    "$run_mode" "$fault_injection_profile" "$live_frontier_capture" \
    "$live_frontier_ready_required" "$evaluation_arm" \
    "$full_rgb_capture" "$d435_5hz_capture" <<'PY'
import hashlib, json, sys, time
from pathlib import Path

output=Path(sys.argv[1]); result=output.parent
lane, profile, run_id, code_sha, resource_profile = sys.argv[2:7]
run_completed, canary_completed, incoming = map(int, sys.argv[7:10])
candidate_profile = sys.argv[10]
rtf_ablation_profile = sys.argv[11]
isaac_sensor_profile = sys.argv[12]
nvblox_mode = sys.argv[13]
run_mode = sys.argv[14]
fault_injection_profile = sys.argv[15]
live_frontier_capture = sys.argv[16] == "1"
live_frontier_ready_required = sys.argv[17] == "1"
evaluation_arm = sys.argv[18]
full_rgb_capture = sys.argv[19] == "1"
d435_5hz_capture = sys.argv[20] == "1"
def load(relative):
    path=result/relative
    try: return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError): return None
def load_jsonl(relative):
    path=result/relative
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, UnicodeError, json.JSONDecodeError):
        return []
binding=load("input_binding.json")
candidate_preflight=load("audits/dgx_candidate_resolution.json")
dgx_candidate=load("remote/dgx/candidate_resolution.json")
dgx_contract=load("remote/dgx/lane_contract.json")
dgx_ready=load("remote/dgx/lane_ready.json")
x86_ready=load("remote/x86/health/ready_probe.json")
dgx_status=load("remote/dgx/lane_status.json")
live_frontier_status=load("remote/dgx/live_frontier/status.json")
x86_status=load("remote/x86/isaac_status.json")
x86_contract=load("remote/x86/isaac_contract.json")
revc_snapshot_smoke=load("remote/x86/evaluator/revc_snapshot_smoke.json")
revc_fixed5_capture=load("remote/x86/evaluator/revc_fixed5_capture.json")
revc_fixed5_scoped_pull=load("audits/revc_fixed5_scoped_pull.json")
screen_dataset_audit=load("remote/x86/screen_dataset_audit.json")
ordered_episode_manifest=load("remote/x86/ordered_episode_manifest.json")
fault_summary=load("fault_injection/fault_injection_summary.json")
step3_health=load("remote/dgx/step3/health.json")
step3_model_health=load("remote/dgx/step3/model_health.json")
step3_services_ready=load("remote/dgx/step3/services_ready.json")
step3_gpu=load("remote/dgx/step3/gpu.json")
step3_decisions=load_jsonl("remote/dgx/step3/frontend_decisions.jsonl")
step3_snapshots=load_jsonl("remote/dgx/step3/snapshots.jsonl")
step3_transactions=load_jsonl("remote/dgx/onboard/step3_advisor_records.jsonl")
step3_profile=isaac_sensor_profile=="lane_b_step3_live_canary"
step3_committed=any(item.get("phase")=="commit" and item.get("status")=="COMMITTED"
                    for item in step3_transactions)
step3_semantic_abstain=any(
    isinstance(item.get("decision"),dict)
    and item["decision"].get("source_decision")=="abstain"
    and item["decision"].get("fallback_used") is True
    for item in step3_decisions
)
step3_released=any(item.get("phase")=="commit"
                   and item.get("status")=="RELEASED_FOR_FALLBACK"
                   for item in step3_transactions)
fixed_dataset_profiles={
    "canary60":5,
    "soak600":5,
    "screen1":1,
    "screen3":3,
    "fixed5":5,
    "pilot-screen1":1,
    "final10":10,
}
final_pilot_profiles={"pilot-screen1","final10"}
expected_execution_count=(binding or {}).get("execution_episode_count")
expected_execution_keys=(binding or {}).get("execution_episode_keys")
expected_execution_episode_ids_ordered=(
    ordered_episode_manifest.get("ordered_episode_ids", [])
    if isinstance(ordered_episode_manifest,dict) else []
)
expected_execution_episode_ids={str(key).rsplit("_",1)[-1]
    for key in expected_execution_keys} if isinstance(expected_execution_keys,list) else set()
expected_candidate_resolution_sha256=(binding or {}).get(
    "candidate_resolution_sha256")
expected_candidate_binding=(binding or {}).get("candidate_binding")
expected_isaac_sensor_profile=(binding or {}).get("isaac_sensor_profile")
expected_remote_run_token="t5_fast_" + profile + "_" + run_id
engineering_canary_path=result/"remote/x86/engineering_canary.json"
engineering_canary=load("remote/x86/engineering_canary.json")
expected_engineering_seconds={"canary60":60,"soak600":600}.get(profile)
expected_engineering_timebase={"canary60":"wall","soak600":"sim"}.get(profile)
engineering_profile=expected_engineering_seconds is not None
engineering_canary_sha256=(
    hashlib.sha256(engineering_canary_path.read_bytes()).hexdigest()
    if engineering_profile and engineering_canary_path.is_file()
       and not engineering_canary_path.is_symlink()
    else None
)
engineering_canary_seal={
    "applicability":"required" if engineering_profile else "not_applicable",
    "relative_path":"remote/x86/engineering_canary.json"
        if engineering_profile else None,
    "sha256":engineering_canary_sha256,
}
revc_snapshot_path=result/"remote/x86/evaluator/revc_snapshot_smoke.json"
revc_snapshot_sha256=(
    hashlib.sha256(revc_snapshot_path.read_bytes()).hexdigest()
    if isaac_sensor_profile=="lane_b_revc_smoke"
       and revc_snapshot_path.is_file() and not revc_snapshot_path.is_symlink()
    else None
)
revc_snapshot_seal={
    "applicability":"required"
        if isaac_sensor_profile=="lane_b_revc_smoke" else "not_applicable",
    "relative_path":"remote/x86/evaluator/revc_snapshot_smoke.json"
        if isaac_sensor_profile=="lane_b_revc_smoke" else None,
    "sha256":revc_snapshot_sha256,
}
revc_fixed5_path=result/"remote/x86/evaluator/revc_fixed5_capture.json"
revc_fixed5_sha256=(
    hashlib.sha256(revc_fixed5_path.read_bytes()).hexdigest()
    if isaac_sensor_profile=="lane_b_revc_fixed5_capture"
       and revc_fixed5_path.is_file() and not revc_fixed5_path.is_symlink()
    else None
)
revc_fixed5_seal={
    "applicability":"required"
        if isaac_sensor_profile=="lane_b_revc_fixed5_capture" else "not_applicable",
    "relative_path":"remote/x86/evaluator/revc_fixed5_capture.json"
        if isaac_sensor_profile=="lane_b_revc_fixed5_capture" else None,
    "sha256":revc_fixed5_sha256,
}
cleanup=load("audits/coordinator_cleanup_receipt.json")
dgx_clear=load("audits/dgx_quarantine_clear.json")
x86_clear=load("audits/x86_quarantine_clear.json")
checks={
    "command_path_completed": incoming == 0 and run_completed == 1,
    "input_binding": isinstance(binding,dict) and binding.get("status")=="PASS"
        and binding.get("code_ref_sha")==code_sha
        and binding.get("execution_profile")==profile
        and binding.get("candidate_profile")==candidate_profile
        and binding.get("rtf_ablation_profile")==rtf_ablation_profile
        and binding.get("isaac_sensor_profile")==isaac_sensor_profile
        and binding.get("nvblox_mode")==nvblox_mode
        and binding.get("run_mode")==run_mode
        and (
            profile not in final_pilot_profiles
            or (
                binding.get("evaluation_arm")==evaluation_arm
                and binding.get("pair_set")==(
                    "paired10_a" if lane=="a" else "paired10_b"
                )
            )
        )
        and (
            binding.get("fault_injection_profile")==fault_injection_profile
            or (
                profile in final_pilot_profiles and fault_injection_profile=="off"
                and "fault_injection_profile" not in binding
            )
        )
        and (
            binding.get("live_frontier_capture") is live_frontier_capture
            and binding.get("live_frontier_ready_required")
                is live_frontier_ready_required
            or (
                profile in final_pilot_profiles and not live_frontier_capture
                and not live_frontier_ready_required
                and "live_frontier_capture" not in binding
                and "live_frontier_ready_required" not in binding
            )
        ),
    "capture_contract": (
        isinstance(x86_contract,dict)
        and isinstance(x86_status,dict)
        and x86_contract.get("model_observation_capture_enabled")
            is full_rgb_capture
        and x86_status.get("model_observation_capture_enabled")
            is full_rgb_capture
        and x86_contract.get("independent_d435_5hz_capture_enabled")
            is d435_5hz_capture
        and x86_status.get("independent_d435_5hz_capture_enabled")
            is d435_5hz_capture
    ),
    "fault_injection_binding": (
        fault_injection_profile=="completion_sim_minimal_v1"
        and isinstance(fault_summary,dict)
        and fault_summary.get("status")=="PASS"
        and fault_summary.get("profile")==fault_injection_profile
        and len(fault_summary.get("event_results",[]))==6
        and isinstance(dgx_status,dict)
        and dgx_status.get("fault_injection_profile")==fault_injection_profile
        and isinstance(x86_status,dict)
        and x86_status.get("fault_injection_profile")==fault_injection_profile
    ) or (
        fault_injection_profile=="off" and fault_summary is None
    ),
    "candidate_cross_host_preflight": isinstance(candidate_preflight,dict)
        and candidate_preflight.get("status")=="PASS"
        and candidate_preflight.get("resolution_sha256")
            == expected_candidate_resolution_sha256
        and candidate_preflight.get("canonical_binding")
            == expected_candidate_binding,
    "candidate_resolution_remote": isinstance(dgx_candidate,dict)
        and dgx_candidate.get("status")=="PASS"
        and dgx_candidate.get("candidate_selector")==candidate_profile
        and dgx_candidate.get("resolution_sha256")
            == expected_candidate_resolution_sha256
        and dgx_candidate.get("canonical_binding")==expected_candidate_binding,
    "candidate_lane_contract": isinstance(dgx_contract,dict)
        and dgx_contract.get("candidate_resolution_sha256")
            == expected_candidate_resolution_sha256
        and dgx_contract.get("candidate_binding")==expected_candidate_binding,
    "run_mode_binding": isinstance(dgx_contract,dict)
        and dgx_contract.get("mode")==run_mode
        and isinstance(dgx_ready,dict) and dgx_ready.get("mode")==run_mode
        and isinstance(dgx_status,dict) and dgx_status.get("mode")==run_mode
        and isinstance(x86_status,dict) and x86_status.get("mode")==run_mode,
    "nvblox_mode_binding": isinstance(dgx_contract,dict)
        and isinstance(dgx_contract.get("nvblox"),dict)
        and dgx_contract["nvblox"].get("mode")==nvblox_mode
        and isinstance(dgx_status,dict)
        and dgx_status.get("nvblox_mode")==nvblox_mode,
    "live_frontier_capture_binding": (
        isinstance(dgx_contract,dict)
        and isinstance(dgx_contract.get("live_frontier_capture"),dict)
        and dgx_contract["live_frontier_capture"].get("enabled")
            is live_frontier_capture
        and isinstance(dgx_ready,dict)
        and isinstance(dgx_ready.get("live_frontier_capture"),dict)
        and dgx_ready["live_frontier_capture"].get("enabled")
            is live_frontier_capture
        and isinstance(dgx_status,dict)
        and isinstance(dgx_status.get("live_frontier_capture"),dict)
        and dgx_status["live_frontier_capture"].get("enabled")
            is live_frontier_capture
    ),
    "live_frontier_ready_gate": (not live_frontier_ready_required) or (
        live_frontier_capture
        and isinstance(live_frontier_status,dict)
        and int(live_frontier_status.get("snapshot_ready_count",0)) >= 1
    ),
    "dgx_ready": isinstance(dgx_ready,dict) and dgx_ready.get("status")=="READY"
        and dgx_ready.get("lane")==lane
        and dgx_ready.get("candidate_profile")==candidate_profile
        and dgx_ready.get("candidate_resolution_sha256")
            == expected_candidate_resolution_sha256
        and dgx_ready.get("candidate_binding")==expected_candidate_binding,
    "x86_ready": isinstance(x86_ready,dict) and x86_ready.get("status")=="PASS"
        and x86_ready.get("lane")==lane,
    "rtf_ablation_binding": isinstance(x86_contract,dict)
        and isinstance(x86_contract.get("rtf_ablation"),dict)
        and x86_contract["rtf_ablation"].get("profile")==rtf_ablation_profile,
    "isaac_sensor_profile_binding": expected_isaac_sensor_profile
        == isaac_sensor_profile
        and isinstance(x86_contract,dict)
        and isinstance(x86_contract.get("isaac_sensor_profile"),dict)
        and x86_contract["isaac_sensor_profile"].get("profile")
            == isaac_sensor_profile
        and x86_contract["isaac_sensor_profile"].get("revc_enabled")
            == (isaac_sensor_profile!="baseline")
        and isinstance(x86_status,dict)
        and x86_status.get("isaac_sensor_profile")==isaac_sensor_profile
        and x86_status.get("revc_enabled")
            == (isaac_sensor_profile!="baseline"),
    "step3_live_canary": (
        step3_profile
        and isinstance(step3_services_ready,dict)
        and step3_services_ready.get("status")=="READY"
        and step3_services_ready.get("motion_authority")=="none"
        and step3_services_ready.get("terminal_stop_authority")=="none"
        and isinstance(step3_model_health,dict)
        and step3_model_health.get("ready") is True
        and step3_model_health.get("revision")
            == "5026053b0c2f5dfaa08fc2d149384162c3c8bca1"
        and step3_model_health.get("runtime_transformers_version")=="4.57.6"
        and step3_model_health.get("precision_mode")=="bf16"
        and step3_model_health.get("checkpoint_load_clean") is True
        and isinstance(step3_health,dict)
        and step3_health.get("raw_text_exposed") is False
        and len(step3_snapshots)>=1 and len(step3_decisions)>=1
        and all(item.get("view_order")==["front_left","front","front_right","rear"]
                for item in step3_snapshots)
        and all(str(item.get("snapshot_id","")).startswith("b::")
                for item in step3_snapshots)
        and (step3_committed or (step3_semantic_abstain and step3_released))
        and isinstance(step3_gpu,dict) and step3_gpu.get("status")=="PASS"
        and step3_gpu.get("both_models_alive") is True
        and step3_gpu.get("oom_observed") is False
        and isinstance(step3_gpu.get("peaks"),dict)
        and float(step3_gpu["peaks"].get("step3_rss_mib",0))>0
        and float(step3_gpu["peaks"].get("internvla_rss_mib",0))>0
        and float(step3_gpu["peaks"].get("step3_swap_mib",-1))==0
        and float(step3_gpu["peaks"].get("internvla_swap_mib",-1))==0
        and step3_gpu.get("checks",{}).get("system_swap_did_not_increase") is True
        and step3_gpu.get("checks",{}).get("system_oom_kill_did_not_increase") is True
        and "raw_text" not in json.dumps(step3_decisions,sort_keys=True)
    ) or (
        not step3_profile and step3_services_ready is None
        and step3_model_health is None and step3_decisions==[]
    ),
    "revc_snapshot_smoke": (
        isaac_sensor_profile=="lane_b_revc_smoke"
        and isinstance(revc_snapshot_smoke,dict)
        and revc_snapshot_smoke.get("status")=="PASS"
        and revc_snapshot_smoke.get("profile")=="lane_b_revc_smoke"
        and revc_snapshot_smoke.get("lane")=="b"
        and revc_snapshot_smoke.get("same_render_tick") is True
        and revc_snapshot_smoke.get("camera_order")
            == ["front_left","front","front_right","rear"]
        and isinstance(revc_snapshot_smoke.get("request"),dict)
        and str(revc_snapshot_smoke["request"].get("request_id","")).startswith(
            "b::revc-smoke::" + expected_remote_run_token + "::")
        and str(revc_snapshot_smoke["request"].get("episode_id","")).startswith("b::")
        and str(revc_snapshot_smoke["request"].get("episode_id",""))[3:]
            in expected_execution_episode_ids
        and revc_snapshot_smoke["request"].get("reset_generation")==0
        and isinstance(revc_snapshot_smoke.get("identity_sync_attempt_count"),int)
        and 0 <= revc_snapshot_smoke["identity_sync_attempt_count"] <= 2
        and isinstance(revc_snapshot_smoke.get("images"),list)
        and len(revc_snapshot_smoke["images"])==4
        and isinstance(revc_snapshot_smoke.get("contract_sha256"),str)
        and len(revc_snapshot_smoke["contract_sha256"])==64
        and isinstance(revc_snapshot_sha256,str)
        and len(revc_snapshot_sha256)==64
    ) or (
        isaac_sensor_profile!="lane_b_revc_smoke" and revc_snapshot_smoke is None
        and revc_snapshot_seal["applicability"]=="not_applicable"
    ),
    "revc_fixed5_capture": (
        isaac_sensor_profile=="lane_b_revc_fixed5_capture"
        and isinstance(revc_fixed5_capture,dict)
        and revc_fixed5_capture.get("status")=="PASS"
        and revc_fixed5_capture.get("profile")=="lane_b_revc_fixed5_capture"
        and revc_fixed5_capture.get("lane")=="b"
        and revc_fixed5_capture.get("capture_count")==5
        and revc_fixed5_capture.get("ordered_episode_ids")
            == expected_execution_episode_ids_ordered
        and revc_fixed5_capture.get("unique_execution_identities") is True
        and revc_fixed5_capture.get("same_render_tick_all") is True
        and revc_fixed5_capture.get("camera_order")
            == ["front_left","front","front_right","rear"]
        and isinstance(revc_fixed5_capture.get("snapshots"),list)
        and len(revc_fixed5_capture["snapshots"])==5
        and [item.get("capture_index")
             for item in revc_fixed5_capture["snapshots"]]==list(range(5))
        and [item.get("request",{}).get("episode_id")
             for item in revc_fixed5_capture["snapshots"]]
            == ["b::"+item for item in expected_execution_episode_ids_ordered]
        and len({(
            item.get("request",{}).get("episode_id"),
            item.get("request",{}).get("reset_generation"),
            item.get("request",{}).get("sequence_id"),
        ) for item in revc_fixed5_capture["snapshots"]})==5
        and all(
            isinstance(item,dict)
            and item.get("profile")=="lane_b_revc_fixed5_capture"
            and item.get("same_render_tick") is True
            and item.get("camera_order")==["front_left","front","front_right","rear"]
            and isinstance(item.get("images"),list) and len(item["images"])==4
            and str(item.get("request",{}).get("request_id","")).startswith(
                "b::revc-fixed5::"+expected_remote_run_token+"::"
            )
            for item in revc_fixed5_capture["snapshots"]
        )
        and isinstance(revc_fixed5_sha256,str) and len(revc_fixed5_sha256)==64
    ) or (
        isaac_sensor_profile!="lane_b_revc_fixed5_capture"
        and revc_fixed5_capture is None
        and revc_fixed5_seal["applicability"]=="not_applicable"
    ),
    "revc_fixed5_local_materialization_ready": (
        isaac_sensor_profile=="lane_b_revc_fixed5_capture"
        and isinstance(revc_fixed5_scoped_pull,dict)
        and revc_fixed5_scoped_pull.get("status")=="PASS"
        and revc_fixed5_scoped_pull.get("profile")==isaac_sensor_profile
        and revc_fixed5_scoped_pull.get("snapshot_count")==5
        and revc_fixed5_scoped_pull.get("png_count")==20
        and bool(revc_fixed5_scoped_pull.get("checks"))
        and all(revc_fixed5_scoped_pull["checks"].values())
    ) or (
        isaac_sensor_profile!="lane_b_revc_fixed5_capture"
        and revc_fixed5_scoped_pull is None
    ),
    "execution_episode_binding": isinstance(x86_contract,dict)
        and x86_contract.get("expected_episode_count")==expected_execution_count
        and expected_execution_count==fixed_dataset_profiles.get(profile),
    "natural_episode_order_binding": (
        # The ordered manifest is emitted by the model client and therefore
        # does not exist in the isolated active-Nvblox Oracle profile.  Oracle
        # binds its fixed set through the screen-dataset audit and separately
        # proves the completed episode count in isaac_remote_validation.json.
        run_mode=="oracle"
        and ordered_episode_manifest is None
        and profile=="screen3"
        and nvblox_mode=="active_local_gt"
        and isinstance(screen_dataset_audit,dict)
        and screen_dataset_audit.get("status")=="PASS"
        and screen_dataset_audit.get("selected_episode_count")
            == expected_execution_count
        and screen_dataset_audit.get("selected_episode_keys")
            == expected_execution_keys
    ) or (
        run_mode=="model"
        and isinstance(ordered_episode_manifest,dict)
        and ordered_episode_manifest.get("status")=="PASS"
        and ordered_episode_manifest.get("dataset_episode_count")
            == expected_execution_count
        and isinstance(expected_execution_episode_ids_ordered,list)
        and len(expected_execution_episode_ids_ordered)==expected_execution_count
        and len(set(expected_execution_episode_ids_ordered))
            == expected_execution_count
        and set(expected_execution_episode_ids_ordered)
            == expected_execution_episode_ids
    ),
    "screen_dataset_binding": (
        profile in {"screen1","screen3","pilot-screen1"}
        and isinstance(screen_dataset_audit,dict)
        and screen_dataset_audit.get("status")=="PASS"
        and screen_dataset_audit.get("source_sha256")==(
            binding or {}).get("dataset_sha256")
        and screen_dataset_audit.get("source_episode_keys")==(
            binding or {}).get("episode_keys")
        and screen_dataset_audit.get("selected_episode_count")==expected_execution_count
        and screen_dataset_audit.get("selected_episode_keys")==expected_execution_keys
    ) or (
        profile not in {"screen1","screen3","pilot-screen1"}
        and screen_dataset_audit is None
    ),
    "final_pilot_binding": (
        profile in final_pilot_profiles
        and isinstance(x86_contract,dict)
        and isinstance(x86_status,dict)
        and (binding or {}).get("split_audit_sha256")
            == (binding or {}).get("source_receipts",{}).get("split_audit_sha256")
        and isinstance((binding or {}).get("split_audit_sha256"),str)
        and len((binding or {}).get("split_audit_sha256"))==64
        and x86_contract.get("final_pilot_lane")==lane
        and x86_status.get("final_pilot_lane")==lane
    ) or (
        profile not in final_pilot_profiles
        and isinstance(x86_contract,dict)
        and isinstance(x86_status,dict)
        and x86_contract.get("final_pilot_lane") in {None,"off"}
        and x86_status.get("final_pilot_lane") in {None,"off"}
    ),
    "dgx_clean_stop": isinstance(dgx_status,dict) and dgx_status.get("status")=="PASS"
        and dgx_status.get("residual_count")==0
        and dgx_status.get("candidate_profile")==candidate_profile
        and dgx_status.get("candidate_resolution_sha256")
            == expected_candidate_resolution_sha256
        and dgx_status.get("candidate_binding")==expected_candidate_binding,
    "x86_clean_stop": isinstance(x86_status,dict) and x86_status.get("residual_count")==0
        and x86_status.get("socket_residual_count")==0
        and x86_status.get("clock_publishers_after_stop")==0
        and x86_status.get("lane_lock_released") is True
        and x86_status.get("shared_asset_lock_fd_released") is True,
    "profile_exit_contract": isinstance(x86_status,dict)
        and x86_status.get("status")=="PASS"
        and x86_status.get("evaluator_exit_code")==0
        and ((profile in fixed_dataset_profiles
              and x86_status.get("execution_profile")=="fixed_dataset"
              and x86_status.get("episode_acceptance_claimed") is True
              and x86_status.get("evaluation_completed_naturally") is True
              and x86_status.get("termination_reason")=="evaluator_natural_exit") or (
            engineering_profile and canary_completed==1
            and isinstance(engineering_canary,dict)
            and engineering_canary.get("status")=="PASS"
            and engineering_canary.get("configured_seconds")==expected_engineering_seconds
            and engineering_canary.get("duration_timebase")==expected_engineering_timebase
            and engineering_canary.get("episode_acceptance_claimed") is False
            and x86_status.get("execution_profile")=="engineering_canary"
            and x86_status.get("episode_acceptance_claimed") is False
            and x86_status.get("evaluation_completed_naturally") is False)),
    "engineering_canary_sealed": (
        engineering_profile
        and isinstance(engineering_canary,dict)
        and isinstance(engineering_canary_sha256,str)
        and len(engineering_canary_sha256)==64
    ) or (
        profile in fixed_dataset_profiles
        and engineering_canary_seal["applicability"]=="not_applicable"
        and engineering_canary_seal["relative_path"] is None
        and engineering_canary_seal["sha256"] is None
    ),
    "owned_cleanup": isinstance(cleanup,dict) and cleanup.get("status")=="PASS"
        and bool(cleanup.get("checks")) and all(cleanup["checks"].values()),
    "dgx_quarantine_cleared": isinstance(dgx_clear,dict) and dgx_clear.get("status")=="PASS",
    "x86_quarantine_cleared": isinstance(x86_clear,dict) and x86_clear.get("status")=="PASS",
}
payload={"schema_version":1,"status":"PASS" if all(checks.values()) else "FAIL",
 "lane":lane,"profile":profile,"run_id":run_id,"code_ref_sha":code_sha,
 "candidate_profile":candidate_profile,
 "evaluation_arm":evaluation_arm,
 "pair_set":"paired10_a" if lane=="a" else "paired10_b",
 "capture":{
   "model_observations":full_rgb_capture,
   "independent_d435_5hz":d435_5hz_capture,
 },
 "candidate_resolution_sha256":expected_candidate_resolution_sha256,
 "candidate_binding":expected_candidate_binding,
 "rtf_ablation_profile":rtf_ablation_profile,
 "isaac_sensor_profile":isaac_sensor_profile,
 "nvblox_mode":nvblox_mode,"run_mode":run_mode,
 "fault_injection_profile":fault_injection_profile,
 "live_frontier_capture":live_frontier_capture,
 "live_frontier_ready_required":live_frontier_ready_required,
 "live_frontier_status":live_frontier_status,
 "resource_profile":resource_profile,"board_grant_used":False,
 "predecessor_gate_used":False,"other_lane_quiet_required":False,
 "raw_log_locations":{
   "local_dgx_ssh":"logs/dgx_runtime_ssh.log",
   "local_x86_ssh":"logs/x86_runtime_ssh.log",
   "remote_x86_result_root":(binding or {}).get("deployment_roots",{}).get("x86","")
       + "/results/t5_fast_" + profile + "_" + run_id,
   "x86_full_archive":"DEFERRED_TO_SERIAL_SHARED_IO",
 },
 "checks":checks,"input_binding":binding,"dgx_status":dgx_status,
 "x86_status":x86_status,"screen_dataset_audit":screen_dataset_audit,
  "sealed_evidence":{"engineering_canary":engineering_canary_seal,
                     "revc_snapshot_smoke":revc_snapshot_seal,
                     "revc_fixed5_capture":revc_fixed5_seal},
 "recorded_unix":time.time()}
output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
raise SystemExit(0 if payload["status"]=="PASS" else 75)
PY
  summary_rc=$?
  set -e
  test "$summary_rc" = 0 || incoming=1
  unset HF_TOKEN
  exit "$incoming"
}
trap finish EXIT
trap 'exit 130' INT TERM HUP

# Only this Lane's deployment, container, ports and runtime lock are inspected.
remote "$dgx_target" \
  "set -euo pipefail; test \"\$(id -un)\" = '$dgx_user'; ip -4 -o addr show | grep -Fq ' $dgx_ip/'; test \"\$(cat '$dgx_root/T5_DEPLOYMENT_REF')\" = '$code_sha'; test -x '$dgx_root/scripts/run_t5_dgx_lane.sh'; test -f '$dgx_root/scripts/resolve_t5_lane_a_candidate.py'; test -f '$dgx_root/scripts/verify_t5_oracle_termination_dataset.py'; test -f '$dgx_root/configs/internnav_t5/completion_sim_geometric_termination.json'; test -f '$dgx_root/ros_ws/install/setup.bash'; test ! -e '$dgx_run'; test ! -e '$dgx_supervisor_ledger'; test \"\$(sha256sum '$map_manifest'|cut -d' ' -f1)\" = '$map_manifest_sha256'; for p in $own_ports $step3_dgx_ports; do test -z \"\$(ss -H -lntup|grep -E \"[:.]\$p[[:space:]]\"||true)\"; done"
remote "$x86_target" \
  "set -euo pipefail; test \"\$(id -un)\" = song; ip -4 -o addr show | grep -Fq ' $x86_ip/'; test \"\$(cat '$x86_root/T5_DEPLOYMENT_REF')\" = '$code_sha'; test -x '$x86_root/scripts/run_t5_distributed_isaac.sh'; test -x '$x86_root/scripts/materialize_t5_screen_dataset.py'; test -f '$x86_root/scripts/probe_t5_revc_snapshot_smoke.py'; test -f '$x86_root/scripts/probe_t5_revc_fixed5_capture.py'; test -f '$x86_root/configs/internnav_t5/revc_four_camera_snapshot.json'; test ! -e '$x86_run'; test ! -e '$x86_supervisor_ledger'; test \"\$(sha256sum '$dataset_root/val_unseen/val_unseen.json.gz'|cut -d' ' -f1)\" = '$dataset_sha256'; test \"\$(docker inspect -f '{{.State.Running}}' '$container')\" = false; test \"\$(docker inspect -f '{{.State.Pid}}' '$container')\" = 0; test \"\$(docker inspect -f '{{index .Config.Labels \"internnav.t5.deployment_root\"}}' '$container')\" = '$x86_root'; test \"\$(docker inspect -f '{{.HostConfig.CpusetCpus}}' '$container')\" = '$cpuset'; test ! -e '$lane_runtime_lock' || flock -n '$lane_runtime_lock' true; for p in $own_ports; do test -z \"\$(ss -H -lntup|grep -E \"[:.]\$p[[:space:]]\"||true)\"; done"

python3 "$root/scripts/resolve_t5_lane_a_candidate.py" --root "$root" \
  --selector "$candidate_profile" --format json \
  >"$result_dir/audits/local_candidate_resolution.json"
remote "$dgx_target" \
  "python3 '$dgx_root/scripts/resolve_t5_lane_a_candidate.py' --root '$dgx_root' --selector '$candidate_profile' --format json" \
  >"$result_dir/audits/dgx_candidate_resolution.json"
cmp -s "$result_dir/audits/local_candidate_resolution.json" \
  "$result_dir/audits/dgx_candidate_resolution.json"

t5_quarantine_arm "$dgx_target" "$dgx_quarantine_file" \
  "$dgx_quarantine_role" "$quarantine_run_tag" "$dgx_root" \
  "$result_dir/audits/dgx_quarantine_arm.json"
dgx_quarantine_armed=true
t5_quarantine_arm "$x86_target" "$x86_quarantine_file" \
  "$x86_quarantine_role" "$quarantine_run_tag" "$x86_root" \
  "$result_dir/audits/x86_quarantine_arm.json"
x86_quarantine_armed=true
t5_remote_compute_absent "$dgx_target" \
  "$result_dir/audits/dgx_structured_compute_prestart.json"

oracle_dataset=""
if test "$termination_mode" = oracle_termination; then
  x86_oracle_root="$x86_root/inputs/oracle_termination_datasets/$remote_run_name"
  dgx_oracle_root="$dgx_root/inputs/oracle_termination_datasets/$remote_run_name"
  source_oracle_dataset="$dataset_root/val_unseen/val_unseen.json.gz"
  source_oracle_audit=""
  if test "$screen_episode_count" != 0; then
    screen_key_args=()
    if test -n "$screen_episode_key"; then
      screen_key_args=(--episode-key "$screen_episode_key")
    fi
    remote "$x86_target" \
      "python3 '$x86_root/scripts/materialize_t5_screen_dataset.py' --source-root '$dataset_root' --output-root '$x86_oracle_root' --count '$screen_episode_count' ${screen_key_args[*]} --expected-source-sha256 '$dataset_sha256' --expected-episode-keys '$episode_keys_csv'" \
      >"$result_dir/audits/oracle_dataset_materialize.json"
    source_oracle_dataset="$x86_oracle_root/val_unseen/val_unseen.json.gz"
    source_oracle_audit="$x86_oracle_root/screen_dataset_audit.json"
  fi
  oracle_dataset="$dgx_oracle_root/val_unseen/val_unseen.json.gz"
  remote "$dgx_target" \
    "set -euo pipefail; test ! -e '$dgx_oracle_root'; mkdir -p '$dgx_oracle_root/val_unseen'"
  scp -q -3 -o BatchMode=yes -o ConnectTimeout=8 \
    "$x86_target:$source_oracle_dataset" \
    "$dgx_target:$oracle_dataset"
  if test -n "$source_oracle_audit"; then
    scp -q -3 -o BatchMode=yes -o ConnectTimeout=8 \
      "$x86_target:$source_oracle_audit" \
      "$dgx_target:$dgx_oracle_root/source_screen_dataset_audit.json"
  fi
  remote "$dgx_target" \
    "python3 '$dgx_root/scripts/verify_t5_oracle_termination_dataset.py' --dataset '$oracle_dataset' --episode-keys '$execution_episode_keys_csv' --output '$dgx_oracle_root/oracle_dataset_binding.json'"
  remote "$dgx_target" "cat '$dgx_oracle_root/oracle_dataset_binding.json'" \
    >"$result_dir/audits/oracle_dataset_binding.json"
fi

read -r -d '' dgx_runtime_program <<'REMOTE_DGX' || true
set -euo pipefail
IFS= read -r HF_TOKEN
IFS= read -r HF_ENDPOINT
[[ "$HF_TOKEN" =~ ^hf_[A-Za-z0-9]{20,}$ ]]
exec {hf_token_fd}<<<"$HF_TOKEN"
unset HF_TOKEN
deployment="$1"; lane="$2"; result="$3"; map="$4"; lease="$5"; domain="$6"; ledger="$7"; candidate="$8"; isaac_ip="$9"; candidate_resolution_sha256="${10}"; strict_extension_profile="${11}"; nvblox_mode="${12}"; run_mode="${13}"; fault_injection_profile="${14}"; step3_live_advisor="${15}"; live_frontier_capture="${16}"; system2_replan_policy="${17}"; step3_timeout_advisor="${18}"; termination_mode="${19}"; oracle_dataset="${20}"; system2_queue_horizon="${21}"; system1_queue_horizon="${22}"
[[ "$candidate_resolution_sha256" =~ ^[0-9a-f]{64}$ ]]
case "$strict_extension_profile" in off|cuvslam_shadow) ;; *) exit 64 ;; esac
case "$nvblox_mode" in off|shadow|active_local_gt) ;; *) exit 64 ;; esac
case "$run_mode" in model|oracle) ;; *) exit 64 ;; esac
case "$fault_injection_profile" in off|completion_sim_minimal_v1) ;; *) exit 64 ;; esac
case "$step3_live_advisor" in 0|1) ;; *) exit 64 ;; esac
case "$live_frontier_capture" in 0|1) ;; *) exit 64 ;; esac
case "$system2_replan_policy" in strict|observation_bound|raw_wire_warn) ;; *) exit 64 ;; esac
case "$system2_queue_horizon" in 0|1) ;; *) exit 64 ;; esac
case "$system1_queue_horizon" in 0|1) ;; *) exit 64 ;; esac
case "$step3_timeout_advisor" in 0|1) ;; *) exit 64 ;; esac
case "$termination_mode" in model_stop|oracle_termination) ;; *) exit 64 ;; esac
test "$termination_mode" != oracle_termination || test -f "$oracle_dataset"
test "$live_frontier_capture" != 1 || test "$lane" = b
pgid="$(ps -o pgid= -p "$$"|tr -d ' ')"
sid="$(ps -o sid= -p "$$"|tr -d ' ')"
starttime="$(awk '{print $22}' /proc/$$/stat)"
argv_sha256="$(sha256sum "/proc/$$/cmdline"|cut -d' ' -f1)"
python3 - "$ledger" "$result" "$$" "$pgid" "$sid" "$starttime" \
  "$argv_sha256" "$lane" <<'PY'
import json,os,sys,time
from pathlib import Path
output=Path(sys.argv[1])
payload={"schema_version":2,"run_root":sys.argv[2],"pid":int(sys.argv[3]),
 "pgid":int(sys.argv[4]),"sid":int(sys.argv[5]),"starttime":int(sys.argv[6]),
 "argv_sha256":sys.argv[7],"lane":sys.argv[8],"role":"dgx",
 "started_unix":time.time()}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
env INTERNNAV_T5_RESOURCE_LEASE_ACK="$lease" INTERNVLA_HF_TOKEN_FD="$hf_token_fd" \
 INTERNNAV_T5_CANDIDATE_PROFILE="$candidate" \
 INTERNNAV_T5_EXPECTED_CANDIDATE_RESOLUTION_SHA256="$candidate_resolution_sha256" \
 INTERNNAV_T5_STRICT_EXTENSION_PROFILE="$strict_extension_profile" \
 INTERNNAV_T5_NVBLOX_MODE="$nvblox_mode" \
 INTERNNAV_T5_FAULT_INJECTION_PROFILE="$fault_injection_profile" \
 INTERNNAV_T5_STEP3_LIVE_ADVISOR="$step3_live_advisor" \
 INTERNVLA_T5_STEP3_TIMEOUT_ADVISOR="$step3_timeout_advisor" \
 INTERNVLA_T5_TERMINATION_MODE="$termination_mode" \
 INTERNVLA_T5_ORACLE_DATASET_FILE="$oracle_dataset" \
 INTERNNAV_T5_LIVE_FRONTIER_CAPTURE="$live_frontier_capture" \
 INTERNVLA_T5_SYSTEM2_REPLAN_POLICY="$system2_replan_policy" \
 INTERNVLA_T5_SYSTEM2_QUEUE_HORIZON="$system2_queue_horizon" \
 INTERNVLA_T5_SYSTEM1_QUEUE_HORIZON="$system1_queue_horizon" \
 INTERNVLA_T5_ISAAC_IP="$isaac_ip" \
 HF_ENDPOINT="$HF_ENDPOINT" INTERNNAV_RUNTIME_POLICY=completion_sim INTERNNAV_SIMULATION_TARGET=isaac \
 ROS_DOMAIN_ID="$domain" CUDA_VISIBLE_DEVICES=0 INTERNNAV_T1_CONTROL_ROOT="$deployment" \
 INTERNVLA_ROS_WS="$deployment/ros_ws" bash "$deployment/scripts/run_t5_dgx_lane.sh" \
 "$lane" "$run_mode" "$result" "$map"
REMOTE_DGX
dgx_runtime_b64="$(printf '%s' "$dgx_runtime_program"|base64|tr -d '\r\n')"
dgx_command="exec setsid --wait bash -c \"\$(printf '%s' '$dgx_runtime_b64'|base64 -d)\" fast-dgx '$dgx_root' '$lane' '$dgx_run' '$map_manifest' '$resource_profile' '$ros_domain_id' '$dgx_supervisor_ledger' '$candidate_profile' '$x86_ip' '$candidate_resolution_sha256' '$strict_extension_profile' '$nvblox_mode' '$run_mode' '$fault_injection_profile' '$step3_live_advisor' '$live_frontier_capture' '$system2_replan_policy' '$step3_timeout_advisor' '$termination_mode' '$oracle_dataset' '$system2_queue_horizon' '$system1_queue_horizon'"
dgx_launch_attempted=1
printf '%s\n%s\n' "$HF_TOKEN" "$HF_ENDPOINT" | \
  ssh "${ssh_options[@]}" "$dgx_target" "$dgx_command" \
    >"$result_dir/logs/dgx_runtime_ssh.log" 2>&1 &
dgx_ssh_pid=$!

dgx_ready_timeout="${INTERNNAV_T5_FAST_DGX_READY_TIMEOUT_SEC:-7200}"
[[ "$dgx_ready_timeout" =~ ^[1-9][0-9]*$ ]]
deadline=$((SECONDS + dgx_ready_timeout))
while (( SECONDS < deadline )); do
  kill -0 "$dgx_ssh_pid" 2>/dev/null || break
  if remote "$dgx_target" \
      "test -f '$dgx_run/lane_ready.json' && python3 -c 'import json;v=json.load(open(\"$dgx_run/lane_ready.json\"));assert v[\"status\"]==\"READY\" and v[\"lane\"]==\"$lane\"'" \
      >/dev/null 2>&1; then break; fi
  sleep 2
done
kill -0 "$dgx_ssh_pid" 2>/dev/null
remote "$dgx_target" "test -f '$dgx_run/lane_ready.json'"

remote "$x86_target" \
  "docker start '$container' >/dev/null; test \"\$(docker inspect -f '{{.State.Running}}' '$container')\" = true"
container_started=1
read -r -d '' x86_runtime_program <<'REMOTE_X86' || true
set -euo pipefail
deployment="$1"; lane="$2"; result="$3"; dataset="$4"; lease="$5"; domain="$6"; gpu="$7"; ledger="$8"; canary_sec="$9"; canary_ack="${10}"; rtf_ablation_profile="${11}"; cpuset="${12}"; screen_count="${13}"; source_dataset_sha256="${14}"; frozen_episode_keys_csv="${15}"; isaac_sensor_profile="${16}"; strict_extension_profile="${17}"; run_mode="${18}"; canary_timebase="${19}"; execution_count="${20}"; final_pilot_lane="${21}"; fault_injection_profile="${22}"; nvblox_mode="${23}"; step3_live_advisor="${24}"; step3_timeout_advisor="${25}"; screen_episode_key="${26}"; full_rgb_capture="${27}"; d435_5hz_capture="${28}"
[[ "$cpuset" =~ ^[0-9,-]+$ ]]
case "$canary_timebase" in wall|sim) ;; *) exit 64 ;; esac
if test "$canary_timebase" = sim; then test "$canary_sec" = 600; fi
case "$screen_count" in 0|1|3) ;; *) exit 64 ;; esac
case "$isaac_sensor_profile" in baseline|lane_b_revc_smoke|lane_b_revc_fixed5_capture|lane_b_step3_live_canary|lane_a_step3_timeout_advisor|dual_lane_wp03_stop_shadow) ;; *) exit 64 ;; esac
case "$step3_live_advisor" in 0|1) ;; *) exit 64 ;; esac
case "$step3_timeout_advisor" in 0|1) ;; *) exit 64 ;; esac
case "$full_rgb_capture" in 0|1) ;; *) exit 64 ;; esac
case "$d435_5hz_capture" in 0|1) ;; *) exit 64 ;; esac
case "$strict_extension_profile" in off|cuvslam_shadow) ;; *) exit 64 ;; esac
case "$run_mode" in model|oracle) ;; *) exit 64 ;; esac
case "$fault_injection_profile" in off|completion_sim_minimal_v1) ;; *) exit 64 ;; esac
case "$nvblox_mode" in off|shadow|active_local_gt) ;; *) exit 64 ;; esac
if test "$nvblox_mode" = active_local_gt; then
  test "$lane" = a
  test "$run_mode" = oracle
  test "$screen_count" = 3
  test "$canary_sec" = 0
fi
if test "$strict_extension_profile" = cuvslam_shadow; then
  test "$lane" = a
  test "$screen_count" = 3
  test "$canary_sec" = 0
fi
if test "$isaac_sensor_profile" = lane_b_revc_smoke; then
  test "$lane" = b
  test "$canary_sec" = 60
  case "$rtf_ablation_profile" in navigation_fast|off) ;; *) exit 64 ;; esac
elif test "$isaac_sensor_profile" = lane_b_revc_fixed5_capture; then
  test "$lane" = b
  test "$canary_sec" = 0
  test "$screen_count" = 0
  case "$rtf_ablation_profile" in navigation_fast|off) ;; *) exit 64 ;; esac
elif test "$isaac_sensor_profile" = lane_b_step3_live_canary; then
  test "$step3_live_advisor" = 1
  test "$lane" = b
  case "$screen_count" in 1|3) ;; *) exit 64 ;; esac
  case "$rtf_ablation_profile" in navigation_fast|off) ;; *) exit 64 ;; esac
elif test "$isaac_sensor_profile" = lane_a_step3_timeout_advisor; then
  test "$step3_live_advisor" = 0
  test "$step3_timeout_advisor" = 1
  test "$lane" = a
  case "$screen_count" in 0|1|3) ;; *) exit 64 ;; esac
  case "$rtf_ablation_profile" in navigation_fast|off) ;; *) exit 64 ;; esac
elif test "$isaac_sensor_profile" = dual_lane_wp03_stop_shadow; then
  test "$step3_live_advisor" = 0
  case "$screen_count:$execution_count" in 0:10|1:1) ;; *) exit 64 ;; esac
  test "$final_pilot_lane" = "$lane"
  test "$rtf_ablation_profile" = navigation_fast
else
  test "$step3_live_advisor" = 0
  test "$step3_timeout_advisor" = 0
fi
pilot_max_step=16000
# WP-03's frozen 5/5 successes completed between 2231 and 5831 evaluator
# steps.  Do not truncate the STOP-shadow pilot below the navigation contract:
# process liveness remains bounded by the outer wall-time timeout, while episode
# completion is governed by the same 16000-step budget as the WP-03 baseline.
[[ "$source_dataset_sha256" =~ ^[0-9a-f]{64}$ ]]
[[ "$execution_count" =~ ^[1-9][0-9]*$ ]]
[[ "$pilot_max_step" =~ ^[1-9][0-9]*$ ]]
python3 - "$frozen_episode_keys_csv" "$execution_count" "$screen_count" <<'PY'
import re,sys
keys=sys.argv[1].split(",")
execution_count,screen_count=int(sys.argv[2]),int(sys.argv[3])
assert len(keys)==len(set(keys))
if screen_count:
    assert screen_count==execution_count and len(keys)>=execution_count
else:
    assert len(keys)==execution_count
assert all(re.fullmatch(r"[A-Za-z0-9_.-]+",key) for key in keys)
PY
if test -n "$screen_episode_key"; then
  test "$screen_count" = 1
  [[ "$screen_episode_key" =~ ^[A-Za-z0-9_.-]+$ ]]
  case ",$frozen_episode_keys_csv," in
    *",$screen_episode_key,"*) ;;
    *) exit 64 ;;
  esac
fi
case "$final_pilot_lane" in
  off) ;;
  a|b)
    test "$final_pilot_lane" = "$lane"
    case "$screen_count:$execution_count" in
      0:10) test -z "$screen_episode_key" ;;
      1:1) test -n "$screen_episode_key" ;;
      *) exit 64 ;;
    esac
    ;;
  *) exit 64 ;;
esac
case "$lane" in
  a) cpuset_env=INTERNVLA_T5_LANE_A_CPUSET ;;
  b) cpuset_env=INTERNVLA_T5_LANE_B_CPUSET ;;
  *) exit 64 ;;
esac
ledger_parent="$(dirname -- "$ledger")"
result_parent="$(dirname -- "$result")"
test "$ledger_parent" = "$result_parent"
test "$ledger_parent" = "$deployment/results"
mkdir -p "$ledger_parent"
test -d "$ledger_parent"
test ! -L "$ledger_parent"
pgid="$(ps -o pgid= -p "$$"|tr -d ' ')"
sid="$(ps -o sid= -p "$$"|tr -d ' ')"
starttime="$(awk '{print $22}' /proc/$$/stat)"
argv_sha256="$(sha256sum "/proc/$$/cmdline"|cut -d' ' -f1)"
python3 - "$ledger" "$result" "$$" "$pgid" "$sid" "$starttime" \
  "$argv_sha256" "$lane" <<'PY'
import json,os,sys,time
from pathlib import Path
output=Path(sys.argv[1])
payload={"schema_version":2,"run_root":sys.argv[2],"pid":int(sys.argv[3]),
 "pgid":int(sys.argv[4]),"sid":int(sys.argv[5]),"starttime":int(sys.argv[6]),
 "argv_sha256":sys.argv[7],"lane":sys.argv[8],"role":"x86",
 "started_unix":time.time()}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
screen_audit=""
if test "$screen_count" != 0; then
  screen_parent="$deployment/inputs/screen_datasets"
  mkdir -p "$screen_parent"
  screen_root="$screen_parent/$(basename "$result")"
  test ! -e "$screen_root"
  screen_key_args=()
  if test -n "$screen_episode_key"; then
    screen_key_args=(--episode-key "$screen_episode_key")
  fi
  python3 "$deployment/scripts/materialize_t5_screen_dataset.py" \
    --source-root "$dataset" --output-root "$screen_root" \
    --count "$screen_count" \
    "${screen_key_args[@]}" \
    --expected-source-sha256 "$source_dataset_sha256" \
    --expected-episode-keys "$frozen_episode_keys_csv" \
    >"${screen_root}.materialize.json"
  screen_audit="$screen_root/screen_dataset_audit.json"
  test -s "$screen_audit"
  dataset="$screen_root"
fi
set +e
env INTERNNAV_T5_RESOURCE_LEASE_ACK="$lease" INTERNNAV_RUNTIME_POLICY=completion_sim \
 INTERNNAV_SIMULATION_TARGET=isaac ROS_DOMAIN_ID="$domain" CUDA_VISIBLE_DEVICES="$gpu" \
 "$cpuset_env=$cpuset" \
 INTERNNAV_T5_ENGINEERING_CANARY_SEC="$canary_sec" \
 INTERNNAV_T5_ENGINEERING_CANARY_ACK="$canary_ack" \
 INTERNNAV_T5_ENGINEERING_CANARY_TIMEBASE="$canary_timebase" \
 INTERNNAV_T5_RTF_ABLATION_PROFILE="$rtf_ablation_profile" \
 INTERNNAV_T5_ISAAC_SENSOR_PROFILE="$isaac_sensor_profile" \
 INTERNNAV_T5_STEP3_LIVE_ADVISOR="$step3_live_advisor" \
 INTERNVLA_T5_STEP3_TIMEOUT_ADVISOR="$step3_timeout_advisor" \
 INTERNVLA_T5_FULL_RGB_CAPTURE="$full_rgb_capture" \
 INTERNVLA_T5_D435_5HZ_CAPTURE="$d435_5hz_capture" \
 INTERNNAV_T5_STRICT_EXTENSION_PROFILE="$strict_extension_profile" \
 INTERNNAV_T5_NVBLOX_MODE="$nvblox_mode" \
 INTERNNAV_T5_FINAL_PILOT_LANE="$final_pilot_lane" \
 INTERNNAV_T5_FAULT_INJECTION_PROFILE="$fault_injection_profile" \
 INTERNVLA_T4_MAX_STEP="$pilot_max_step" \
 INTERNNAV_T1_CONTROL_ROOT="$deployment" INTERNVLA_ROS_WS=/home/song/internnav-t4/isaac_ros_ws_45 \
 INTERNVLA_T5_ISAAC_WORKER_ROOT=/home/song/internnav-t1-t2/runtime/t5_isaac_workers \
 bash "$deployment/scripts/run_t5_distributed_isaac.sh" "$lane" "$run_mode" "$result" "$dataset"
runtime_rc=$?
set -e
if test -n "$screen_audit" && test -d "$result"; then
  cp "$screen_audit" "$result/screen_dataset_audit.json"
fi
exit "$runtime_rc"
REMOTE_X86
x86_runtime_b64="$(printf '%s' "$x86_runtime_program"|base64|tr -d '\r\n')"
engineering_canary_sec=0
engineering_canary_ack=none
engineering_canary_timebase=wall
final_pilot_lane=off
case "$profile" in
  canary60) engineering_canary_sec=60; engineering_canary_ack=fast-path ;;
  soak600)
    engineering_canary_sec=600
    engineering_canary_ack=fast-path
    engineering_canary_timebase=sim
    ;;
  pilot-screen1|final10) final_pilot_lane="$lane" ;;
esac
x86_command="exec setsid --wait bash -c \"\$(printf '%s' '$x86_runtime_b64'|base64 -d)\" fast-x86 '$x86_root' '$lane' '$x86_run' '$dataset_root' '$resource_profile' '$ros_domain_id' '$gpu' '$x86_supervisor_ledger' '$engineering_canary_sec' '$engineering_canary_ack' '$rtf_ablation_profile' '$cpuset' '$screen_episode_count' '$dataset_sha256' '$episode_keys_csv' '$isaac_sensor_profile' '$strict_extension_profile' '$run_mode' '$engineering_canary_timebase' '$execution_episode_count' '$final_pilot_lane' '$fault_injection_profile' '$nvblox_mode' '$step3_live_advisor' '$step3_timeout_advisor' '$screen_episode_key' '$full_rgb_capture' '$d435_5hz_capture'"
x86_launch_attempted=1
remote "$x86_target" "$x86_command" >"$result_dir/logs/x86_runtime_ssh.log" 2>&1 &
x86_ssh_pid=$!

x86_ready_timeout="${INTERNNAV_T5_FAST_X86_READY_TIMEOUT_SEC:-1200}"
[[ "$x86_ready_timeout" =~ ^[1-9][0-9]*$ ]]
deadline=$((SECONDS + x86_ready_timeout))
while (( SECONDS < deadline )); do
  kill -0 "$dgx_ssh_pid" 2>/dev/null || { echo "DGX exited before Isaac READY" >&2; exit 1; }
  kill -0 "$x86_ssh_pid" 2>/dev/null || break
  if remote "$x86_target" \
      "test -f '$x86_run/health/ready_probe.json' && python3 -c 'import json;v=json.load(open(\"$x86_run/health/ready_probe.json\"));assert v[\"status\"]==\"PASS\" and v[\"lane\"]==\"$lane\"'" \
      >/dev/null 2>&1; then break; fi
  sleep 2
done
kill -0 "$x86_ssh_pid" 2>/dev/null
remote "$x86_target" "test -f '$x86_run/health/ready_probe.json'"

if test "$fault_injection_profile" = completion_sim_minimal_v1; then
  fault_wall_liveness_timeout="${INTERNNAV_T5_FAULT_WALL_LIVENESS_TIMEOUT_SEC:-7200}"
  [[ "$fault_wall_liveness_timeout" =~ ^[1-9][0-9]*$ ]]
  ((fault_wall_liveness_timeout >= 300 && fault_wall_liveness_timeout <= 8400))
  setsid python3 -u "$root/scripts/run_t5_fault_injection.py" run \
    --lane "$lane" \
    --config "$root/configs/internnav_t5/fault_injection_minimal_v1.json" \
    --output-dir "$result_dir/fault_injection" \
    --dgx-run "$dgx_run" --x86-run "$x86_run" \
    --wall-liveness-timeout-sec "$fault_wall_liveness_timeout" \
    >"$result_dir/logs/fault_injection_director.log" 2>&1 &
  fault_director_pid=$!
fi

case "$profile" in
  canary60) run_timeout="${INTERNNAV_T5_FAST_CANARY_TIMEOUT_SEC:-420}" ;;
  soak600) run_timeout="${INTERNNAV_T5_FAST_SOAK_TIMEOUT_SEC:-7800}" ;;
  screen1|screen3|pilot-screen1) run_timeout="${INTERNNAV_T5_FAST_SCREEN_TIMEOUT_SEC:-10800}" ;;
  fixed5) run_timeout="${INTERNNAV_T5_FAST_RUN_TIMEOUT_SEC:-21600}" ;;
  final10) run_timeout="${INTERNNAV_T5_FINAL_PILOT_TIMEOUT_SEC:-43200}" ;;
esac
[[ "$run_timeout" =~ ^[1-9][0-9]*$ ]]
if test "$profile" = soak600; then
  ((run_timeout >= 3000 && run_timeout <= 9000))
fi
deadline=$((SECONDS + run_timeout))
while kill -0 "$x86_ssh_pid" 2>/dev/null; do
  kill -0 "$dgx_ssh_pid" 2>/dev/null || { echo "DGX failed during $profile" >&2; exit 1; }
  (( SECONDS < deadline )) || { echo "$profile timed out" >&2; exit 124; }
  sleep 2
done
set +e; wait "$x86_ssh_pid"; x86_rc=$?; set -e
x86_ssh_pid=""
test "$x86_rc" = 0
if test "$fault_injection_profile" = completion_sim_minimal_v1; then
  set +e
  wait "$fault_director_pid"
  fault_director_rc=$?
  set -e
  fault_director_pid=""
  test "$fault_director_rc" = 0
  python3 -c 'import json,sys;v=json.load(open(sys.argv[1],encoding="utf-8"));assert v.get("profile")=="completion_sim_minimal_v1" and v.get("status")=="PASS" and len(v.get("event_results",[]))==6 and v.get("checks") and all(v["checks"].values())' \
    "$result_dir/fault_injection/fault_injection_summary.json"
fi
if test "$profile" = canary60 || test "$profile" = soak600; then
  remote "$x86_target" \
    "python3 -c 'import json;v=json.load(open(\"$x86_run/engineering_canary.json\"));assert v[\"status\"]==\"PASS\" and v[\"configured_seconds\"]==$engineering_canary_sec and v[\"duration_timebase\"]==\"$engineering_canary_timebase\" and v[\"episode_acceptance_claimed\"] is False'" \
    >/dev/null
  canary_window_completed=1
fi
if test "$isaac_sensor_profile" = lane_b_revc_smoke; then
  remote "$x86_target" \
    "python3 -c 'import json;v=json.load(open(\"$x86_run/evaluator/revc_snapshot_smoke.json\"));assert v[\"status\"]==\"PASS\" and v[\"profile\"]==\"lane_b_revc_smoke\" and v[\"lane\"]==\"b\" and v[\"same_render_tick\"] is True and v[\"camera_order\"]==[\"front_left\",\"front\",\"front_right\",\"rear\"] and 0 < v[\"external_preview_max_hz\"] <= 1 and len(v[\"images\"])==4'" \
    >/dev/null
elif test "$isaac_sensor_profile" = lane_b_revc_fixed5_capture; then
  remote "$x86_target" \
    "python3 -c 'import json;v=json.load(open(\"$x86_run/evaluator/revc_fixed5_capture.json\"));s=v.get(\"snapshots\",[]);ids=[(x.get(\"request\",{}).get(\"episode_id\"),x.get(\"request\",{}).get(\"reset_generation\"),x.get(\"request\",{}).get(\"sequence_id\")) for x in s];assert v[\"status\"]==\"PASS\" and v[\"profile\"]==\"lane_b_revc_fixed5_capture\" and v[\"lane\"]==\"b\" and v[\"capture_count\"]==5 and v[\"same_render_tick_all\"] is True and len(ids)==len(set(ids))==5 and all(len(x.get(\"images\",[]))==4 for x in s)'" \
    >/dev/null
fi

remote "$dgx_target" "touch '$dgx_run/stop.request'"
deadline=$((SECONDS + 300))
while kill -0 "$dgx_ssh_pid" 2>/dev/null && (( SECONDS < deadline )); do sleep 1; done
! kill -0 "$dgx_ssh_pid" 2>/dev/null
set +e; wait "$dgx_ssh_pid"; dgx_rc=$?; set -e
dgx_ssh_pid=""
test "$dgx_rc" = 0
remote "$x86_target" \
  "docker stop -t 20 '$container' >/dev/null; test \"\$(docker inspect -f '{{.State.Running}}' '$container')\" = false; test \"\$(docker inspect -f '{{.State.Pid}}' '$container')\" = 0"
container_started=0
run_completed=1

trap - EXIT INT TERM HUP
finish
