#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: run_t5_dgx_lane.sh LANE MODE RESULT_DIR STATIC_MAP_MANIFEST

  LANE: a | b
  MODE: oracle | model

This entrypoint runs one complete, non-Isaac T5 compute lane on its assigned
DGX.  It must be invoked while the matching lane-a/lane-b resource lease is
held.  Credentials are intentionally not accepted by this script.
EOF
  exit 64
}

[[ $# -eq 4 ]] || usage
lane="$1"
mode="$2"
result_dir="$3"
static_map_manifest="$4"
case "$mode" in oracle|model) ;; *) usage ;; esac

case "$lane" in
  a)
    expected_user=railgun
    lane_ip="${INTERNVLA_T5_LANE_IP:-10.100.100.128}"
    ros_domain_id=75
    lane_namespace=/t5/lane_a
    controller_port=25137
    model_client_port=25139
    oracle_port=25140
    identity_prefix='a::'
    expected_lease=lane-a
    ;;
  b)
    expected_user=rail
    lane_ip="${INTERNVLA_T5_LANE_IP:-10.100.120.122}"
    ros_domain_id=76
    lane_namespace=/t5/lane_b
    controller_port=25138
    model_client_port=25239
    oracle_port=25240
    identity_prefix='b::'
    expected_lease=lane-b
    ;;
  *) usage ;;
esac

if [[ -v INTERNNAV_T5_CANDIDATE_PROFILE ]]; then
  candidate_profile="$INTERNNAV_T5_CANDIDATE_PROFILE"
else
  candidate_profile=baseline
fi
case "$candidate_profile" in
  baseline|recovery_a) ;;
  a0|a1|a0+b0|a0+b1|a1+b0|a1+b1|a1+b0+c0|a1+b0+c1|a1+b0+c2|a1+b1+c0|a1+b1+c1|a1+b1+c2) ;;
  *) echo "invalid T5 candidate profile: $candidate_profile" >&2; exit 64 ;;
esac
system2_replan_policy="${INTERNVLA_T5_SYSTEM2_REPLAN_POLICY:-observation_bound}"
case "$system2_replan_policy" in
  strict|observation_bound|raw_wire_warn) ;;
  *) echo "invalid System2 replan policy: $system2_replan_policy" >&2; exit 64 ;;
esac
system2_queue_horizon="${INTERNVLA_T5_SYSTEM2_QUEUE_HORIZON:-0}"
case "$system2_queue_horizon" in 0|1) ;;
  *) echo "invalid System2 queue horizon: $system2_queue_horizon" >&2; exit 64 ;;
esac
system1_queue_horizon="${INTERNVLA_T5_SYSTEM1_QUEUE_HORIZON:-0}"
case "$system1_queue_horizon" in 0|1) ;;
  *) echo "invalid System1 queue horizon: $system1_queue_horizon" >&2; exit 64 ;;
esac
if test "$system2_queue_horizon" = 1; then
  test "$lane" = a
  test "$mode" = model
  test "$candidate_profile" = recovery_a
fi
if test "$system1_queue_horizon" = 1; then
  test "$lane" = a
  test "$mode" = model
  test "$candidate_profile" = recovery_a
fi

root="${INTERNNAV_T1_CONTROL_ROOT:-$HOME/internnav-t1-t2}"
ros_ws="${INTERNVLA_ROS_WS:-$root/ros_ws}"
model_python="${INTERNVLA_MODEL_PYTHON:-$HOME/internnav-t0/venv-model/bin/python}"
params="${INTERNVLA_T5_PARAMS:-$root/configs/internnav_t5/primary_completion_sim.yaml}"
nav2_params="${INTERNVLA_NAV2_PARAMS:-$root/configs/internnav_t5/nav2_static_lidar.yaml}"
golden_bundle="${INTERNVLA_T5_GOLDEN_BUNDLE:-$root/configs/internnav_t5/golden_bundle_manifest.json}"
isaac_ip="${INTERNVLA_T5_ISAAC_IP:-10.100.120.123}"
test "$isaac_ip" = 10.100.120.123
nvblox_mode="${INTERNNAV_T5_NVBLOX_MODE:-off}"
case "$nvblox_mode" in off|shadow|active_local_gt) ;; *) usage ;; esac
strict_extension_profile="${INTERNNAV_T5_STRICT_EXTENSION_PROFILE:-off}"
case "$strict_extension_profile" in
  off) cuvslam_mode=off ;;
  cuvslam_shadow) cuvslam_mode=shadow ;;
  *) usage ;;
esac
fault_injection_profile="${INTERNNAV_T5_FAULT_INJECTION_PROFILE:-off}"
case "$fault_injection_profile" in
  off) ;;
  completion_sim_minimal_v1)
    test "$mode" = model
    test "$strict_extension_profile" = off
    test "$nvblox_mode" = off
    ;;
  *) echo "unsupported T5 fault injection profile: $fault_injection_profile" >&2; exit 64 ;;
esac
export INTERNNAV_T5_FAULT_INJECTION_PROFILE="$fault_injection_profile"
termination_mode="${INTERNVLA_T5_TERMINATION_MODE:-model_stop}"
case "$termination_mode" in model_stop|oracle_termination) ;; *) usage ;; esac
termination_overlay="$root/configs/internnav_t5/completion_sim_geometric_termination.json"
termination_dataset="${INTERNVLA_T5_ORACLE_DATASET_FILE:-}"
termination_variant_id=none
termination_variant_sha256=none
if test "$termination_mode" = oracle_termination; then
  test "$mode" = model
  test -f "$termination_overlay"
  test -f "$termination_dataset"
  termination_variant_id=t5_completion_sim_geometric_termination
  termination_variant_sha256="$(sha256sum "$termination_overlay"|cut -d' ' -f1)"
  [[ "$termination_variant_sha256" =~ ^[0-9a-f]{64}$ ]]
else
  test -z "$termination_dataset"
fi
step3_live_advisor="${INTERNNAV_T5_STEP3_LIVE_ADVISOR:-0}"
case "$step3_live_advisor" in 0|1) ;; *) usage ;; esac
step3_timeout_advisor="${INTERNVLA_T5_STEP3_TIMEOUT_ADVISOR:-0}"
case "$step3_timeout_advisor" in 0|1) ;; *) usage ;; esac
live_frontier_capture="${INTERNNAV_T5_LIVE_FRONTIER_CAPTURE:-$step3_live_advisor}"
case "$live_frontier_capture" in 0|1) ;; *) usage ;; esac
if test "$live_frontier_capture" = 1; then
  test "$lane" = b
fi
if test "$step3_live_advisor" = 1; then
  test "$lane" = b
  test "$mode" = model
  test "$candidate_profile" = a1+b1+c1
  test "$strict_extension_profile" = off
  test "$nvblox_mode" = off
  test "$fault_injection_profile" = off
fi
if test "$step3_timeout_advisor" = 1; then
  test "$mode" = model
  test "$candidate_profile" = recovery_a
  test "$strict_extension_profile" = off
  test "$nvblox_mode" = off
  test "$fault_injection_profile" = off
  test "$system2_replan_policy" = observation_bound
  test "$step3_live_advisor" = 0
  test "$live_frontier_capture" = 0
fi
if test "$cuvslam_mode" = shadow; then
  test "$lane" = a
  test "$nvblox_mode" = off
fi

# The real lease is acquired outside this process.  The legacy T4 launchers
# receive their historical acknowledgement only after this lane-specific
# acknowledgement has been checked.
test "${INTERNNAV_T5_RESOURCE_LEASE_ACK:-}" = "$expected_lease"
test "${INTERNNAV_RUNTIME_POLICY:-}" = completion_sim
test "${INTERNNAV_SIMULATION_TARGET:-}" = isaac
test "$(id -un)" = "$expected_user"
test "${ROS_DOMAIN_ID:-}" = "$ros_domain_id"
test "${CUDA_VISIBLE_DEVICES:-}" = 0
ip -4 -o addr show scope global | awk '{sub(/\/.*/, "", $4); print $4}' | \
  grep -Fxq "$lane_ip"
test -f "$static_map_manifest"
test -f "$params"
test -f "$nav2_params"
test -f "$golden_bundle"
test -f "$ros_ws/install/setup.bash"
test -x "$model_python"
test -f "$root/scripts/run_t4_model_server.sh"
test -f "$root/scripts/run_t4_dgx_onboard.sh"
test -f "$root/scripts/t4_model_health_probe.py"
test -f "$root/scripts/t5_fault_restart_session_probe.py"
test -f "$root/scripts/check_t4_nav2_data_plane.py"
test -f "$root/scripts/audit_t5_hf_token_process_scope.py"
test -f "$root/scripts/resolve_t5_lane_a_candidate.py"
test -f "$root/scripts/materialize_t5_nvblox_profile.py"
test -f "$root/configs/internnav_t5/nvblox_runtime_contract.json"
if test "$live_frontier_capture" = 1; then
  test -f "$root/scripts/run_t5_live_frontier_capture.sh"
  test -f "$root/scripts/t5_live_frontier_snapshot_node.py"
  test -f "$root/configs/internnav_t5/live_frontier_capture.json"
fi
if test "$step3_live_advisor" = 1; then
  test -f "$root/scripts/run_t5_step3_live_services.sh"
  test -d "${INTERNNAV_T5_STEP3_MODEL_PATH:-/home/rail/ai-stack/models/Step3-VL-10B}"
  test -f "${INTERNNAV_T5_STEP3_VENV:-/home/rail/ai-stack/venvs/step3-vl-10b-tf4.57.6}/T5_STEP3_RUNTIME_READY.json"
fi
if test "$step3_timeout_advisor" = 1; then
  test -f "$root/scripts/run_t5_step3_timeout_service.sh"
  test -d "${INTERNNAV_T5_STEP3_MODEL_PATH:-$HOME/ai-stack/models/Step3-VL-10B}"
  test -f "${INTERNNAV_T5_STEP3_VENV:-$HOME/ai-stack/venvs/step3-vl-10b-tf4.57.6}/T5_STEP3_RUNTIME_READY.json"
fi
if test "$cuvslam_mode" = shadow; then
  test -f "$root/configs/internnav_t5/cuvslam_shadow.yaml"
  test -f "$root/internvla_t4_sensors/launch/t4_cuvslam.launch.py"
  test -f "$root/scripts/materialize_t5_cuvslam_shadow_params.py"
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
mapfile -t candidate_env < <(
  python3 "$root/scripts/resolve_t5_lane_a_candidate.py" \
    --root "$root" --selector "$candidate_profile" --format env
)
test "${#candidate_env[@]}" -ge 1
declare -A candidate_env_seen=()
effective_candidate_profile=""
for candidate_assignment in "${candidate_env[@]}"; do
  candidate_name="${candidate_assignment%%=*}"
  candidate_value="${candidate_assignment#*=}"
  case "$candidate_name" in
    INTERNNAV_T5_CANDIDATE_PROFILE|INTERNVLA_T4_VIEW_MODE|INTERNVLA_T4_HISTORY_MODE|INTERNVLA_T5_TRAJECTORY_RERANK|INTERNVLA_T4_PROGRESS_HORIZON_SEC|INTERNVLA_T4_REFRESH_DISTANCE_M|INTERNVLA_T4_REFRESH_TIME_SEC|INTERNVLA_T4_TRAJECTORY_VALIDITY_SEC) ;;
    *) echo "resolver emitted an unapproved candidate key: $candidate_name" >&2; exit 64 ;;
  esac
  test -z "${candidate_env_seen[$candidate_name]:-}"
  candidate_env_seen[$candidate_name]=1
  if test "$candidate_name" = INTERNNAV_T5_CANDIDATE_PROFILE; then
    effective_candidate_profile="$candidate_value"
  fi
done
case "$effective_candidate_profile" in baseline|recovery_a) ;; *) exit 64 ;; esac
if test "$system2_replan_policy" != observation_bound; then
  test "$lane" = a
  test "$mode" = model
  test "$effective_candidate_profile" = recovery_a
fi
if test "$effective_candidate_profile" = recovery_a; then
  test -f "$root/scripts/t4_recovery_runtime.py"
  test -f "$root/configs/completion_sim/recovery/profile_a.json"
  test -f "$root/configs/internnav_t5/candidates/recovery_a.json"
fi
test ! -e "$result_dir"

# D0 may hand the token to this coordinator through one inherited anonymous
# descriptor.  Consume and close it before any child exists; the value remains
# a non-exported shell variable and is exported only inside the model subshell.
hf_token=""
hf_token_supplied=0
if test -n "${INTERNVLA_HF_TOKEN_FD:-}"; then
  [[ "$INTERNVLA_HF_TOKEN_FD" =~ ^[0-9]+$ ]]
  IFS= read -r hf_token <&"$INTERNVLA_HF_TOKEN_FD"
  eval "exec ${INTERNVLA_HF_TOKEN_FD}<&-"
  [[ "$hf_token" =~ ^hf_[A-Za-z0-9]{20,}$ ]]
  hf_token_supplied=1
fi
unset INTERNVLA_HF_TOKEN_FD HF_TOKEN HUGGING_FACE_HUB_TOKEN

for port in "$controller_port" "$model_client_port" "$oracle_port"; do
  [[ "$port" =~ ^[0-9]+$ ]] && ((port >= 1024 && port <= 65535))
done
if test "$step3_live_advisor" = 1; then
  for port in 8200 8300; do
    ((port >= 1024 && port <= 65535))
  done
elif test "$step3_timeout_advisor" = 1; then
  ((8200 >= 1024 && 8200 <= 65535))
fi
test "$controller_port" != "$model_client_port"
test "$controller_port" != "$oracle_port"
test "$model_client_port" != "$oracle_port"

mkdir -p "$result_dir/logs" "$result_dir/health"
result_dir="$(cd "$result_dir" && pwd -P)"
if test "$fault_injection_profile" != off; then
  mkdir -p "$result_dir/fault_control/requests" \
    "$result_dir/fault_control/acks"
  fault_control_path="$result_dir/fault_control/state.json"
  fault_event_path="$result_dir/fault_control/events.jsonl"
  python3 - "$fault_control_path" "$lane" "$fault_injection_profile" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
temporary.write_text(json.dumps({
    "schema_version": 1,
    "profile": sys.argv[3],
    "lane": sys.argv[2],
    "revision": 0,
    "observed_sim_ns": 0,
    "active": [],
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
else
  fault_control_path=""
  fault_event_path=""
fi
static_map_manifest="$(readlink -f "$static_map_manifest")"
params="$(readlink -f "$params")"
nav2_params="$(readlink -f "$nav2_params")"
candidate_config=""
recovery_runtime_manifest=""
recovery_env=()
onboard_profile="$([[ $mode = oracle ]] && printf migrated_oracle10 || printf migrated_pilot20)"
python3 "$root/scripts/resolve_t5_lane_a_candidate.py" \
  --root "$root" --selector "$candidate_profile" \
  --output "$result_dir/candidate_resolution.json" --format none
candidate_resolution_sha256="$(python3 - "$result_dir/candidate_resolution.json" \
  "$candidate_profile" "$effective_candidate_profile" <<'PY'
import hashlib, json, sys
from pathlib import Path

path = Path(sys.argv[1])
value = json.loads(path.read_text(encoding="utf-8"))
declared = value.pop("resolution_sha256", None)
canonical = json.dumps(
    value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    allow_nan=False,
).encode("utf-8")
observed = hashlib.sha256(canonical).hexdigest()
if declared != observed:
    raise SystemExit("candidate resolution self-hash mismatch")
if value.get("candidate_selector") != sys.argv[2]:
    raise SystemExit("candidate resolution selector mismatch")
if value.get("effective_candidate_profile") != sys.argv[3]:
    raise SystemExit("candidate resolution effective profile mismatch")
binding = value.get("canonical_binding")
if not isinstance(binding, dict):
    raise SystemExit("candidate resolution canonical binding is missing")
print(observed)
PY
)"
[[ "$candidate_resolution_sha256" =~ ^[0-9a-f]{64}$ ]]
if test -n "${INTERNNAV_T5_EXPECTED_CANDIDATE_RESOLUTION_SHA256:-}"; then
  [[ "$INTERNNAV_T5_EXPECTED_CANDIDATE_RESOLUTION_SHA256" =~ ^[0-9a-f]{64}$ ]]
  test "$candidate_resolution_sha256" = \
    "$INTERNNAV_T5_EXPECTED_CANDIDATE_RESOLUTION_SHA256"
fi
unset INTERNNAV_T5_EXPECTED_CANDIDATE_RESOLUTION_SHA256
if test "$effective_candidate_profile" = recovery_a; then
  candidate_config="$root/configs/internnav_t5/candidates/recovery_a.json"
  recovery_profile="$root/configs/completion_sim/recovery/profile_a.json"
  recovery_nav2_params="$result_dir/recovery_nav2_params.yaml"
  recovery_runtime_manifest="$result_dir/recovery_runtime_manifest.json"
  mapfile -t recovery_fields < <(
    python3 "$root/scripts/t4_recovery_runtime.py" \
      --profile "$recovery_profile" --nav2-input "$nav2_params" \
      --nav2-output "$recovery_nav2_params" \
      --manifest "$recovery_runtime_manifest" --format fields
  )
  test "${#recovery_fields[@]}" -eq 12
  test "${recovery_fields[0]}" = A
  test "${recovery_fields[3]}" = 4.0
  test "${recovery_fields[7]}" = 2
  test "${recovery_fields[11]}" = 35.0
  mapfile -t recovery_override_fields < <(
    python3 - "$candidate_config" "$recovery_runtime_manifest" <<'PY'
import json
import sys
from pathlib import Path

candidate_path = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
overrides = candidate.get("t5_completion_sim_overrides")
expected = {
    "recovery_scan_yaw_rad": 0.3,
    "replan_deadline_sec": 65.0,
}
if not isinstance(overrides, dict) or set(overrides) != set(expected):
    raise SystemExit("invalid T5 recovery override keys")
for key, expected_value in expected.items():
    value = overrides[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"T5 recovery override {key} must be numeric")
    if float(value) != expected_value:
        raise SystemExit(f"unexpected T5 recovery override {key}: {value!r}")
if candidate.get("baseline_contract", {}).get(
    "t4_profile_and_canonical_sha256_unchanged"
) is not True:
    raise SystemExit("candidate must preserve the frozen T4 profile and hash")
if manifest.get("profile_sha256") != candidate.get("runtime", {}).get(
    "t4_profile_canonical_sha256"
):
    raise SystemExit("materialized T4 profile hash differs from candidate")
manifest["effective_t5_completion_sim_overrides"] = expected
manifest["frozen_t4_profile_unchanged"] = True
manifest_path.write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
    newline="\n",
)
print("0.3")
print("65.0")
PY
  )
  test "${#recovery_override_fields[@]}" -eq 2
  test "${recovery_override_fields[0]}" = 0.3
  test "${recovery_override_fields[1]}" = 65.0
  nav2_params="$recovery_nav2_params"
  onboard_profile=migrated_recovery_a5
  recovery_env=(
    INTERNVLA_T4_ENABLE_RECOVERY=1
    INTERNVLA_T4_RECOVERY_MODE=on
    INTERNVLA_T4_RECOVERY_PROFILE="$recovery_profile"
    INTERNVLA_T4_RECOVERY_RUNTIME_MANIFEST="$recovery_runtime_manifest"
    INTERNVLA_T4_RECOVERY_PROFILE_ID="${recovery_fields[0]}"
    INTERNVLA_T4_RECOVERY_PROFILE_SHA256="${recovery_fields[1]}"
    INTERNVLA_T4_RECOVERY_NAV2_SHA256="${recovery_fields[2]}"
    INTERNVLA_T4_PROGRESS_HORIZON_SEC="${recovery_fields[3]}"
    INTERNVLA_T4_MINIMUM_PROGRESS_M="${recovery_fields[4]}"
    INTERNVLA_T4_OSCILLATION_TRAVEL_M="${recovery_fields[5]}"
    INTERNVLA_T4_RECOVERY_COOLDOWN_SEC="${recovery_fields[6]}"
    INTERNVLA_T4_MAXIMUM_RECOVERIES="${recovery_fields[7]}"
    INTERNVLA_T4_RECOVERY_SCAN_YAW_RAD="${recovery_override_fields[0]}"
    INTERNVLA_T4_REPLAN_DEADLINE_SEC="${recovery_override_fields[1]}"
    INTERNVLA_T4_RECOVERY_SCAN_SPEED_RPS="${recovery_fields[9]}"
    INTERNVLA_T4_RECOVERY_SAFETY_FRESHNESS_SEC="${recovery_fields[10]}"
    INTERNVLA_T4_MAXIMUM_RECOVERY_DURATION_SEC="${recovery_fields[11]}"
    INTERNVLA_T4_ENABLE_SCHEDULED_REFRESH=1
    INTERNVLA_T4_MAXIMUM_SCHEDULED_REFRESHES=1
  )
fi
nvblox_bundle=""
nvblox_contract=""
nvblox_params=""
enable_d435i=0
if test "$nvblox_mode" != off; then
  nvblox_bundle="$result_dir/nvblox_profile"
  python3 "$root/scripts/materialize_t5_nvblox_profile.py" \
    --mode "$nvblox_mode" --nav2-base "$nav2_params" \
    --lane-namespace "$lane_namespace" \
    --output-dir "$nvblox_bundle" \
    >"$result_dir/logs/nvblox_materialize.json"
  nvblox_contract="$nvblox_bundle/runtime_contract.json"
  nvblox_params="$nvblox_bundle/nvblox_params.yaml"
  nav2_params="$nvblox_bundle/nav2_params.yaml"
  test -f "$nvblox_contract"
  test -f "$nvblox_params"
  test -f "$nav2_params"
  enable_d435i=1
fi
if test "$cuvslam_mode" = shadow; then
  cuvslam_profile_dir="$result_dir/cuvslam_profile"
  mkdir "$cuvslam_profile_dir"
  python3 "$root/scripts/materialize_t5_cuvslam_shadow_params.py" \
    --base "$nav2_params" \
    --overlay "$root/configs/internnav_t5/cuvslam_shadow.yaml" \
    --output "$cuvslam_profile_dir/nav2_params.yaml" \
    --receipt "$cuvslam_profile_dir/materialization.json" \
    >"$result_dir/logs/cuvslam_materialize.json"
  nav2_params="$cuvslam_profile_dir/nav2_params.yaml"
  test -f "$nav2_params"
fi
pid_ledger="$result_dir/pid_ledger.jsonl"
: >"$pid_ledger"

set +u
source /opt/ros/jazzy/setup.bash
source "$ros_ws/install/setup.bash"
set -u
export ROS2CLI_NO_DAEMON=1
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ROS_STATIC_PEERS="$isaac_ip"
if test "$nvblox_mode" != off; then
  ros2 pkg prefix nvblox_ros >/dev/null
  ros2 pkg prefix nvblox_nav2 >/dev/null
  test -x "$ros_ws/install/internvla_t4_sensors/lib/internvla_t4_sensors/internvla_t5_nvblox_supervisor"
fi
if test "$cuvslam_mode" = shadow; then
  ros2 pkg prefix isaac_ros_visual_slam >/dev/null
  ros2 pkg prefix internvla_t4_sensors >/dev/null
  test -x "$ros_ws/install/internvla_t4_sensors/lib/internvla_t4_sensors/internvla_t4_odometry_supervisor"
fi

port_is_free() {
  test -z "$(ss -H -ltn "sport = :$1")"
}

for port in "$controller_port" "$model_client_port" "$oracle_port"; do
  if ! port_is_free "$port"; then
    echo "lane $lane TCP port is already in use: $port" >&2
    exit 73
  fi
done
if test "$step3_live_advisor" = 1; then
  for port in 8200 8300; do
    if ! port_is_free "$port"; then
      echo "Lane-B Step3 loopback port is already in use: $port" >&2
      exit 73
    fi
  done
elif test "$step3_timeout_advisor" = 1; then
  if ! port_is_free 8200; then
    echo "Lane $lane Step3 timeout service port is already in use: 8200" >&2
    exit 73
  fi
fi

if pgrep -af '[i]saac-sim|[k]it/kit' >"$result_dir/forbidden_isaac_prestart.txt"; then
  echo "Isaac/Kit is forbidden on a T5 DGX lane" >&2
  exit 73
fi
if pgrep -af '[i]nternvla_t4_recovery.model_node|[i]nternvla_ros2.model_node|[n]av2_|[c]ontroller_server|[i]nternvla_t4_sensor_bridge' \
    >"$result_dir/conflicting_lane_prestart.txt"; then
  echo "a model or navigation stack already exists on this leased DGX" >&2
  exit 73
fi

python3 - "$golden_bundle" "$HOME/internnav-t0/InternNav" \
  "$result_dir/internnav_source_identity_audit.json" <<'PY'
import json, subprocess, sys
from pathlib import Path

golden_path, root, output = map(Path, sys.argv[1:4])
model = json.loads(golden_path.read_text(encoding='utf-8'))['model']
revision = subprocess.check_output(
    ['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True
).strip()
tree = subprocess.check_output(
    ['git', '-C', str(root), 'rev-parse', 'HEAD^{tree}'], text=True
).strip()
status = subprocess.check_output(
    ['git', '-C', str(root), 'status', '--porcelain=v1',
     '--untracked-files=all'], text=True
)
raw_submodules = subprocess.check_output(
    ['git', '-C', str(root), 'submodule', 'status', '--recursive'], text=True
)
submodules = {}
markers_clean = True
for line in raw_submodules.splitlines():
    if not line:
        continue
    markers_clean &= line[0] == ' '
    fields = line[1:].split()
    if len(fields) < 2:
        markers_clean = False
        continue
    submodules[fields[1]] = fields[0]
checks = {
    'revision': revision == model['internnav_revision'],
    'tree': tree == model['internnav_tree_sha'],
    'worktree_clean': model['internnav_worktree_clean_required'] and not status,
    'submodule_markers_clean': markers_clean,
    'submodule_revisions': submodules == model['internnav_submodules'],
}
payload = {
    'schema_version': 1, 'status': 'PASS' if all(checks.values()) else 'FAIL',
    'revision': revision, 'tree': tree, 'submodules': submodules,
    'worktree_status_entry_count': len(status.splitlines()), 'checks': checks,
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
if payload['status'] != 'PASS':
    raise SystemExit(f'InternNav source identity mismatch: {checks}')
PY

record_pid_event() {
  local component="$1" pid="$2" event="$3" log_path="$4" pgid=""
  if test -n "$pid"; then
    pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
  fi
  python3 - "$pid_ledger" "$lane" "$component" "$pid" "$pgid" "$event" \
    "$log_path" "$ros_domain_id" "$lane_namespace" <<'PY'
import json, os, sys, time
from pathlib import Path

row = {
    "schema_version": 1,
    "lane": sys.argv[2],
    "component": sys.argv[3],
    "pid": int(sys.argv[4]) if sys.argv[4] else None,
    "pgid": int(sys.argv[5]) if sys.argv[5] else None,
    "event": sys.argv[6],
    "log_path": sys.argv[7],
    "ros_domain_id": int(sys.argv[8]),
    "namespace": sys.argv[9],
    "host": os.uname().nodename,
    "wall_unix": time.time(),
}
with Path(sys.argv[1]).open("a", encoding="utf-8", newline="\n") as stream:
    stream.write(json.dumps(row, sort_keys=True) + "\n")
PY
}

model_pid=""
internvla_service_pid=""
onboard_pid=""
step3_pid=""
step3_log_path="$result_dir/logs/step3_services.log"
client_pid=""
nvblox_pid=""
cuvslam_pid=""
live_frontier_pid=""
model_result_dir="$result_dir/model"
model_log_path="$result_dir/logs/model_outer.log"
onboard_result_dir="$result_dir/onboard"
onboard_log_path="$result_dir/logs/onboard_outer.log"
online_ready=0
shutdown_reason=process_exit

group_is_alive() {
  local pid="$1"
  test -n "$pid" || return 1
  # Final cleanup is literal: even a zombie-only PGID remains a residual until
  # its reaper removes it.  This predicate therefore includes every state.
  ps -eo pgid= | awk -v expected="$pid" \
    '$1 == expected { found=1 } END { exit(found ? 0 : 1) }'
}

group_has_runnable_member() {
  local pid="$1"
  test -n "$pid" || return 1
  # Zombies cannot react to a signal.  Use the runnable subset only while
  # deciding whether another escalation signal is useful.
  ps -eo stat=,pgid= | awk -v expected="$pid" \
    '$2 == expected && $1 !~ /^Z/ { found=1 } END { exit(found ? 0 : 1) }'
}

# Runtime health belongs to the direct supervised leader, not merely to any
# process that still shares its process group.  A ROS 2 daemon can outlive a
# failed launcher in that PGID; treating the group as healthy would hide the
# launcher exit indefinitely.  The group check remains separate for cleanup so
# every orphaned descendant is still drained before the Lane lease is released.
leader_is_alive() {
  local pid="$1" state="" pgid=""
  test -n "$pid" || return 1
  read -r state pgid < <(ps -o stat=,pgid= -p "$pid") || return 1
  [[ "$state" != Z* && "$pgid" = "$pid" ]]
}

bounded_wait_child() {
  local pid="$1" timeout_sec="${2:-5}" timer wait_rc=0 parent_starttime
  parent_starttime="$(awk '{print $22}' "/proc/$$/stat")" || return 1
  # Bash wait is normally immediate after the process group is gone, but it is
  # still a builtin with no timeout.  A trapped private signal keeps abnormal
  # child-reaping behavior from pinning the whole lane finalizer indefinitely.
  (
    local current_starttime
    sleep "$timeout_sec"
    current_starttime="$(awk '{print $22}' "/proc/$$/stat" 2>/dev/null)" || exit 0
    test "$current_starttime" = "$parent_starttime" || exit 0
    kill -USR1 "$$" 2>/dev/null || true
  ) &
  timer=$!
  trap ':' USR1
  wait "$pid" 2>/dev/null || wait_rc=$?
  kill "$timer" 2>/dev/null || true
  wait "$timer" 2>/dev/null || true
  trap - USR1
  return 0
}

handle_evaluator_exit() {
  local client_rc=0
  wait "$client_pid" || client_rc=$?
  record_pid_event evaluator "$client_pid" exited "$result_dir/logs/evaluator.log"
  # The model client is a persistent server.  Its only successful D0 stop is
  # the coordinator's stop.request after the x86 fixed-five process returns 0.
  # Even rc=0 (and even a FINISHED summary written from Python's finally block)
  # is unexpected before that barrier.  The oracle bridge has a distinct
  # protocol: it may naturally exit after its remote shutdown operation.
  if test "$mode" = model; then
    shutdown_reason="evaluator_unexpected_exit:rc_$client_rc"
    return 1
  fi
  if test "$client_rc" != 0; then
    shutdown_reason="evaluator_exit:rc_$client_rc"
    return 1
  fi
  shutdown_reason=evaluator_completed
}

stop_group() {
  local component="$1" pid="$2" log_path="$3"
  local graceful_already_sent="${4:-0}"
  test -n "$pid" || return 0
  if test "$graceful_already_sent" != 1; then
    record_pid_event "$component" "$pid" stop_requested "$log_path"
  fi
  if test "$component" = onboard; then
    # This shell is launched as an asynchronous non-interactive job.  Bash can
    # inherit SIGINT as ignored in that position, so do not burn the graceful
    # window on an INT that cannot enter its cleanup trap.  TERM maps to the
    # accepted 143 shutdown and the onboard trap may drain eight independent
    # child sessions serially (a bounded worst case of roughly 48 seconds).
    if group_has_runnable_member "$pid"; then
      if test "$graceful_already_sent" != 1; then
        kill -TERM -- "-$pid" 2>/dev/null || true
      fi
      for _ in $(seq 1 700); do
        group_has_runnable_member "$pid" || break
        sleep 0.1
      done
    fi
  elif test "$component" = step3; then
    # The timeout-advisor wrapper owns a nested setsid model service.  Enter
    # its TERM trap directly; an asynchronous bash may ignore INT, which
    # would otherwise consume the whole outer minute before the child is
    # killed and allow it to be reparented outside the wrapper PGID.
    if group_has_runnable_member "$pid"; then
      if test "$graceful_already_sent" != 1; then
        kill -TERM -- "-$pid" 2>/dev/null || true
      fi
      for _ in $(seq 1 200); do
        group_has_runnable_member "$pid" || break
        sleep 0.1
      done
    fi
  else
    if test "$graceful_already_sent" = 1; then
      for _ in $(seq 1 200); do
        group_has_runnable_member "$pid" || break
        sleep 0.1
      done
    elif group_has_runnable_member "$pid"; then
      kill -INT -- "-$pid" 2>/dev/null || true
      for _ in $(seq 1 600); do
        group_has_runnable_member "$pid" || break
        sleep 0.1
      done
    fi
    if test "$graceful_already_sent" != 1 && \
        group_has_runnable_member "$pid"; then
      kill -TERM -- "-$pid" 2>/dev/null || true
      for _ in $(seq 1 200); do
        group_has_runnable_member "$pid" || break
        sleep 0.1
      done
    fi
  fi
  group_has_runnable_member "$pid" && kill -KILL -- "-$pid" 2>/dev/null || true
  bounded_wait_child "$pid" 5
  # Reaping a killed orphan can lag signal delivery.  Give the host init a
  # bounded window, then fail closed if the literal PGID still exists.
  for _ in $(seq 1 50); do
    group_is_alive "$pid" || break
    sleep 0.1
  done
  if group_is_alive "$pid"; then
    record_pid_event "$component" "$pid" residual "$log_path"
    return 1
  fi
  record_pid_event "$component" "$pid" verified_absent "$log_path"
}

write_status() {
  local status="$1" observed_rc="$2" normalized_rc="$3" residual="$4"
  python3 - "$result_dir/lane_status.json" "$status" "$lane" "$mode" \
    "$online_ready" "$shutdown_reason" "$observed_rc" "$normalized_rc" \
    "$residual" "$ros_domain_id" "$lane_namespace" "$lane_ip" "$isaac_ip" \
    "$controller_port" "$model_client_port" "$oracle_port" "$identity_prefix" \
    "$candidate_profile" "$effective_candidate_profile" \
    "$result_dir/candidate_resolution.json" "$nvblox_mode" \
    "$live_frontier_capture" <<'PY'
import json, os, sys, time
from pathlib import Path

candidate = json.loads(Path(sys.argv[20]).read_text(encoding="utf-8"))
candidate_resolution_sha256 = candidate.get("resolution_sha256")
candidate_binding = candidate.get("canonical_binding")
live_frontier_enabled = sys.argv[22] == "1"
live_frontier_status_path = Path(sys.argv[1]).parent / "live_frontier/status.json"
try:
    live_frontier_status = json.loads(
        live_frontier_status_path.read_text(encoding="utf-8")
    ) if live_frontier_enabled else None
except (OSError, UnicodeError, json.JSONDecodeError):
    live_frontier_status = None
owners = [
    "internvla_n1_model", "ros2_sensor_ingress", "static_global_map",
    "ground_truth_localization", "lidar_local_costmap", "nav2",
    "watchdog", "command_adapter", "bounded_continuous_velocity_control",
]
if sys.argv[21] != "off":
    owners.append("nvblox_sensor_fed_{}".format(sys.argv[21]))
if live_frontier_enabled:
    owners.append("capture_only_live_frontier")
Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": 1,
    "status": sys.argv[2],
    "lane": sys.argv[3],
    "mode": sys.argv[4],
    "online_ready": sys.argv[5] == "1",
    "shutdown_reason": sys.argv[6],
    "observed_exit_code": int(sys.argv[7]),
    "normalized_exit_code": int(sys.argv[8]),
    "residual_count": int(sys.argv[9]),
    "host": os.uname().nodename,
    "host_role": f"dgx_lane_{sys.argv[3]}",
    "ros_domain_id": int(sys.argv[10]),
    "namespace": sys.argv[11],
    "bind_ip": sys.argv[12],
    "expected_isaac_peer_ip": sys.argv[13],
    "ports": {
        "controller": int(sys.argv[14]),
        "model_client": int(sys.argv[15]),
        "oracle": int(sys.argv[16]),
    },
    "identity_prefix": sys.argv[17],
    "candidate_profile": sys.argv[18],
    "effective_candidate_profile": sys.argv[19],
    "candidate_resolution_sha256": candidate_resolution_sha256,
    "candidate_binding": candidate_binding,
    "nvblox_mode": sys.argv[21],
    "live_frontier_capture": {
        "enabled": live_frontier_enabled,
        "runtime_status": (live_frontier_status or {}).get("status"),
        "snapshot_ready_count": int(
            (live_frontier_status or {}).get("snapshot_ready_count", 0)
        ),
        "status_path": "live_frontier/status.json",
        "snapshot_path": "live_frontier/current.json",
    },
    "fault_injection_profile": os.environ["INTERNNAV_T5_FAULT_INJECTION_PROFILE"],
    "use_sim_time": True,
    "cuda_visible_devices": "0",
    "owners": owners,
    "forbidden_owners": ["isaac_sim", "go2_physics"],
    "finished_unix": time.time(),
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

cleanup() {
  local rc=$? residual=0 status=FAIL normalized_rc
  trap - EXIT INT TERM HUP
  set +e
  hf_token=""
  unset hf_token
  # Open both graceful-stop paths immediately at the coordinator barrier.
  # The client must not outlive the 60-second canary cutoff, while onboard
  # must begin draining Nav2 before any later child-reaping anomaly.
  test -z "$onboard_pid" || \
    record_pid_event onboard "$onboard_pid" stop_requested "$onboard_log_path"
  test -z "$nvblox_pid" || \
    record_pid_event nvblox "$nvblox_pid" stop_requested "$result_dir/logs/nvblox_supervisor.log"
  test -z "$cuvslam_pid" || \
    record_pid_event cuvslam "$cuvslam_pid" stop_requested "$result_dir/logs/cuvslam_supervisor.log"
  test -z "$live_frontier_pid" || \
    record_pid_event live_frontier "$live_frontier_pid" stop_requested "$result_dir/logs/live_frontier_capture.log"
  test -z "$client_pid" || \
    record_pid_event evaluator "$client_pid" stop_requested "$result_dir/logs/evaluator.log"
  test -z "$nvblox_pid" || ! group_has_runnable_member "$nvblox_pid" || \
    kill -TERM -- "-$nvblox_pid" 2>/dev/null || true
  test -z "$cuvslam_pid" || ! group_has_runnable_member "$cuvslam_pid" || \
    kill -TERM -- "-$cuvslam_pid" 2>/dev/null || true
  test -z "$live_frontier_pid" || ! group_has_runnable_member "$live_frontier_pid" || \
    kill -INT -- "-$live_frontier_pid" 2>/dev/null || true
  for pid in "$onboard_pid" "$client_pid"; do
    test -z "$pid" || ! group_has_runnable_member "$pid" || \
      kill -TERM -- "-$pid" 2>/dev/null || true
  done
  # Let Nvblox disable its optional local layer before stopping the
  # motion-producing onboard stack. A ROS client reap delay must not postpone
  # Nav2's own bounded child-session cleanup.
  stop_group nvblox "$nvblox_pid" "$result_dir/logs/nvblox_supervisor.log" 1 || residual=$((residual + 1))
  stop_group cuvslam "$cuvslam_pid" "$result_dir/logs/cuvslam_supervisor.log" 1 || residual=$((residual + 1))
  stop_group live_frontier "$live_frontier_pid" "$result_dir/logs/live_frontier_capture.log" 1 || residual=$((residual + 1))
  if test "$live_frontier_capture" = 1; then
    rm -f "$result_dir/live_frontier/current.json"
    test ! -e "$result_dir/live_frontier/current.json" || residual=$((residual + 1))
  fi
  stop_group onboard "$onboard_pid" "$onboard_log_path" 1 || residual=$((residual + 1))
  stop_group evaluator "$client_pid" "$result_dir/logs/evaluator.log" 1 || residual=$((residual + 1))
  stop_group step3 "$step3_pid" "$step3_log_path" || residual=$((residual + 1))
  stop_group model "$model_pid" "$model_log_path" || residual=$((residual + 1))
  if test -n "$onboard_pid"; then
    python3 - "$onboard_result_dir/onboard_status.json" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit("onboard cleanup status is missing")
value = json.loads(path.read_text(encoding="utf-8"))
if int(value.get("residual_process_group_count", -1)) != 0:
    raise SystemExit("onboard child process group survived cleanup")
PY
    test $? = 0 || residual=$((residual + 1))
  fi
  if test "$fault_injection_profile" != off; then
    python3 -c 'import json,sys;v=json.load(open(sys.argv[1],encoding="utf-8"));assert v.get("profile")=="completion_sim_minimal_v1" and v.get("active")==[], "fault control remained active during cleanup"' \
      "$fault_control_path"
    test $? = 0 || residual=$((residual + 1))
  fi
  for port in "$controller_port" "$model_client_port" "$oracle_port"; do
    port_is_free "$port" || residual=$((residual + 1))
  done
  if test "$step3_live_advisor" = 1; then
    for port in 8200 8300; do
      port_is_free "$port" || residual=$((residual + 1))
    done
  elif test "$step3_timeout_advisor" = 1; then
    port_is_free 8200 || residual=$((residual + 1))
  fi
  pgrep -af '[i]saac-sim|[k]it/kit' >"$result_dir/forbidden_isaac_poststop.txt"
  test $? = 1 || residual=$((residual + 1))
  normalized_rc="$rc"
  if test "$online_ready" = 1 && test "$residual" = 0 && \
      { test "$rc" = 0 || test "$rc" = 130 || test "$rc" = 143; }; then
    status=PASS
    # A clean SIGINT/SIGTERM is a normal supervised lane shutdown.
    normalized_rc=0
  fi
  write_status "$status" "$rc" "$normalized_rc" "$residual"
  if test "$status" != PASS && test "$normalized_rc" = 0; then
    normalized_rc=1
  fi
  exit "$normalized_rc"
}
trap cleanup EXIT
trap 'shutdown_reason=signal_int; exit 130' INT
trap 'shutdown_reason=signal_term; exit 143' TERM HUP

python3 - "$result_dir/lane_contract.json" "$lane" "$mode" "$ros_domain_id" \
  "$lane_namespace" "$lane_ip" "$isaac_ip" "$controller_port" \
  "$model_client_port" "$oracle_port" "$identity_prefix" "$static_map_manifest" \
  "$params" "$nav2_params" "$candidate_profile" "$effective_candidate_profile" \
  "$result_dir/candidate_resolution.json" "$recovery_runtime_manifest" \
  "$candidate_config" "$nvblox_mode" "$nvblox_contract" \
  "$live_frontier_capture" "$system2_replan_policy" "$termination_mode" \
  "$termination_overlay" "$termination_dataset" \
  "$system2_queue_horizon" "$system1_queue_horizon" <<'PY'
import hashlib, json, os, sys, time
from pathlib import Path

def evidence(raw):
    path = Path(raw).resolve()
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

candidate_path = Path(sys.argv[17])
candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
candidate_resolution_sha256 = candidate.pop("resolution_sha256", None)
candidate_canonical = json.dumps(
    candidate, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    allow_nan=False,
).encode("utf-8")
if candidate_resolution_sha256 != hashlib.sha256(candidate_canonical).hexdigest():
    raise SystemExit("candidate resolution self-hash mismatch in lane contract")
if candidate.get("candidate_selector") != sys.argv[15]:
    raise SystemExit("candidate selector mismatch in lane contract")
if candidate.get("effective_candidate_profile") != sys.argv[16]:
    raise SystemExit("candidate effective profile mismatch in lane contract")
candidate_binding = candidate.get("canonical_binding")
if not isinstance(candidate_binding, dict):
    raise SystemExit("candidate canonical binding missing in lane contract")
expected_binding_keys = {
    "candidate_selector", "effective_candidate_profile", "candidate_manifest",
    "selected_configs", "registered_config_sha256_by_family",
    "code_bundle_sha256", "fixed_episode_count", "fixed_episode_keys",
    "predecessor_candidate_ids",
}
if set(candidate_binding) != expected_binding_keys:
    raise SystemExit("candidate canonical binding key set drift")

Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": 1,
    "status": "READY_TO_START",
    "lane": sys.argv[2],
    "mode": sys.argv[3],
    "host": os.uname().nodename,
    "ros_domain_id": int(sys.argv[4]),
    "namespace": sys.argv[5],
    "bind_ip": sys.argv[6],
    "expected_isaac_peer_ip": sys.argv[7],
    "dds": {
        "automatic_discovery_range": "LOCALHOST",
        "static_peers": [sys.argv[7]],
    },
    "ports": {"controller": int(sys.argv[8]),
              "model_client": int(sys.argv[9]), "oracle": int(sys.argv[10])},
    "identity_prefix": sys.argv[11],
    "episode_prefix": sys.argv[11],
    "reset_prefix": sys.argv[11],
    "request_prefix": sys.argv[11],
    "use_sim_time": True,
    "cuda_visible_devices": "0",
    "colocation_required": True,
    "model_transport": "same_host_ros2",
    "isaac_allowed_on_this_host": False,
    "candidate_profile": sys.argv[15],
    "effective_candidate_profile": sys.argv[16],
    "candidate_resolution_sha256": candidate_resolution_sha256,
    "candidate_binding": candidate_binding,
    "inputs": {
        "static_map_manifest": evidence(sys.argv[12]),
        "model_params": evidence(sys.argv[13]),
        "nav2_params": evidence(sys.argv[14]),
        "candidate_resolution": evidence(sys.argv[17]),
    },
    "recovery": {
        "enabled": sys.argv[16] == "recovery_a",
        "runtime_manifest": evidence(sys.argv[18]) if sys.argv[18] else None,
        "candidate_config": evidence(sys.argv[19]) if sys.argv[19] else None,
        "system2_replan_policy": sys.argv[23],
        "raw_wire_warn_only": sys.argv[23] == "raw_wire_warn",
        "system2_queue_horizon": int(sys.argv[27]),
        "system1_queue_horizon": int(sys.argv[28]),
    },
    "nvblox": {
        "mode": sys.argv[20],
        "opt_in": sys.argv[20] != "off",
        "runtime_contract": evidence(sys.argv[21]) if sys.argv[21] else None,
        "default_navigation": "static_global_plus_lidar_voxel_local",
    },
    "live_frontier_capture": {
        "enabled": sys.argv[22] == "1",
        "status_path": "live_frontier/status.json",
        "snapshot_path": "live_frontier/current.json",
        "authority": "none",
    },
    "termination": {
        "mode": sys.argv[24],
        "geometric_success_only": sys.argv[24] == "oracle_termination",
        "credits_model_stop": False if sys.argv[24] == "oracle_termination" else None,
        "overlay": evidence(sys.argv[25])
            if sys.argv[24] == "oracle_termination" else None,
        "dataset": evidence(sys.argv[26])
            if sys.argv[24] == "oracle_termination" else None,
    },
    "started_unix": time.time(),
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

common_env=(
  INTERNNAV_T1_CONTROL_ROOT="$root"
  INTERNVLA_ROS_WS="$ros_ws"
  INTERNNAV_RUNTIME_POLICY=completion_sim
  INTERNNAV_SIMULATION_TARGET=isaac
  INTERNNAV_T4_RESOURCE_LEASE_ACK=dgx+isaac
  INTERNNAV_T5_LANE="$lane"
  INTERNNAV_T5_LANE_NAMESPACE="$lane_namespace"
  INTERNNAV_T5_ID_PREFIX="$identity_prefix"
  INTERNVLA_EPISODE_ID_PREFIX="$identity_prefix"
  INTERNVLA_RESET_ID_PREFIX="$identity_prefix"
  INTERNVLA_REQUEST_ID_PREFIX="$identity_prefix"
  ROS_DOMAIN_ID="$ros_domain_id"
  ROS_NAMESPACE="$lane_namespace"
  ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
  ROS_STATIC_PEERS="$isaac_ip"
  CUDA_VISIBLE_DEVICES=0
  INTERNNAV_T5_FAULT_INJECTION_PROFILE="$fault_injection_profile"
  INTERNNAV_T5_FAULT_CONTROL_PATH="$fault_control_path"
  INTERNNAV_T5_FAULT_EVENT_PATH="$fault_event_path"
  INTERNNAV_T5_STEP3_LIVE_ADVISOR="$step3_live_advisor"
  INTERNVLA_T5_STEP3_TIMEOUT_ADVISOR="$step3_timeout_advisor"
  INTERNNAV_T5_LIVE_FRONTIER_CAPTURE="$live_frontier_capture"
  INTERNVLA_T5_SYSTEM2_REPLAN_POLICY="$system2_replan_policy"
  INTERNVLA_T5_SYSTEM2_QUEUE_HORIZON="$system2_queue_horizon"
  INTERNVLA_T5_SYSTEM1_QUEUE_HORIZON="$system1_queue_horizon"
  INTERNVLA_T4_TERMINATION_MODE="$termination_mode"
  INTERNVLA_T4_VARIANT_ID="$termination_variant_id"
  INTERNVLA_T4_VARIANT_CONFIG_SHA256="$termination_variant_sha256"
  INTERNVLA_T4_ABLATION_DATASET_FILE="$termination_dataset"
  INTERNNAV_T5_STEP3_ENDPOINT=tcp://127.0.0.1:8200
  T5_LANE_B_LIVE_FRONTIER_PATH="$result_dir/live_frontier/current.json"
  T5_LANE_B_RESULTS="$result_dir"
  PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
)

model_restart_count=0
onboard_restart_count=0
model_restart_session_file=""

start_model() {
  local launch_result_dir="$1" launch_log_path="$2"
  local -a restart_session_env=(INTERNVLA_T5_MODEL_RESTART_SESSION_FILE=)
  if test -n "$model_restart_session_file"; then
    restart_session_env=(
      INTERNVLA_T5_MODEL_RESTART_SESSION_FILE="$model_restart_session_file"
    )
  fi
  test ! -e "$launch_result_dir"
  (
    if test -n "${hf_token:-}"; then
      export HF_TOKEN="$hf_token"
    fi
    exec setsid env "${common_env[@]}" "${recovery_env[@]}" "${candidate_env[@]}" \
      "${restart_session_env[@]}" \
      INTERNVLA_MODEL_RESULT_DIR="$launch_result_dir" \
      INTERNVLA_MODEL_PYTHON="$model_python" \
      INTERNVLA_BACKEND=real INTERNVLA_PRELOAD_MODEL=1 \
      INTERNVLA_T4_FUNCTIONAL_MODEL=1 \
      HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
      bash "$root/scripts/run_t4_model_server.sh" --ros-args \
        --params-file "$params" -r __ns:="$lane_namespace" \
        -p use_sim_time:=true
  ) >"$launch_log_path" 2>&1 </dev/null &
  model_pid=$!
  record_pid_event model "$model_pid" started "$launch_log_path"
}

start_onboard() {
  local launch_result_dir="$1" launch_log_path="$2"
  local restart_session_file="${3:-}"
  local -a restart_session_env=(INTERNVLA_T5_ONBOARD_RESTART_SESSION_FILE=)
  if test -n "$restart_session_file"; then
    test -f "$restart_session_file"
    restart_session_env=(
      INTERNVLA_T5_ONBOARD_RESTART_SESSION_FILE="$restart_session_file"
    )
  fi
  test ! -e "$launch_result_dir"
  setsid env "${common_env[@]}" "${recovery_env[@]}" "${candidate_env[@]}" \
    "${restart_session_env[@]}" \
    INTERNVLA_T4_DGX_BIND_IP="$lane_ip" \
    INTERNVLA_T4_ISAAC_PEER_IP="$isaac_ip" \
    INTERNVLA_T4_CONTROLLER_TCP_PORT="$controller_port" \
    INTERNVLA_T4_ENABLE_D435I="$enable_d435i" \
    INTERNVLA_T4_ENABLE_STEREO_FEED="$([[ $cuvslam_mode = shadow ]] && printf 1 || printf 0)" \
    INTERNVLA_T4_ONBOARD_PROFILE="$onboard_profile" \
    INTERNVLA_ONBOARD_USE_SIM_TIME=true \
    INTERNVLA_ONBOARD_HOST_ROLE=dgx_onboard_compute \
    INTERNVLA_ONBOARD_MODEL_OWNER=local_dgx \
    INTERNVLA_ONBOARD_NAMESPACE="$lane_namespace" \
    INTERNVLA_T4_ALLOW_PRECLOCK_ZERO_TF_WARN_DROP=1 \
    INTERNVLA_T4_ALLOW_COMMAND_POSE_ANCHOR_FALLBACK=1 \
    INTERNVLA_NAV2_PARAMS="$nav2_params" \
    bash "$root/scripts/run_t4_dgx_onboard.sh" \
      --result-dir "$launch_result_dir" \
      --static-map-manifest "$static_map_manifest" \
      >"$launch_log_path" 2>&1 </dev/null &
  onboard_pid=$!
  record_pid_event onboard "$onboard_pid" started "$launch_log_path"
}

start_model "$model_result_dir" "$model_log_path"
start_onboard "$onboard_result_dir" "$onboard_log_path"

if test "$live_frontier_capture" = 1; then
  setsid env "${common_env[@]}" INTERNNAV_T5_RESOURCE_LEASE_ACK=lane-b \
    bash "$root/scripts/run_t5_live_frontier_capture.sh" \
      "$result_dir/live_frontier" \
      >"$result_dir/logs/live_frontier_capture.log" 2>&1 </dev/null &
  live_frontier_pid=$!
  record_pid_event live_frontier "$live_frontier_pid" started \
    "$result_dir/logs/live_frontier_capture.log"
fi

if test "$cuvslam_mode" = shadow; then
  setsid env "${common_env[@]}" \
    ros2 run internvla_t4_sensors internvla_t4_odometry_supervisor --ros-args \
      -r __ns:="$lane_namespace" -r /tf:=tf -r /tf_static:=tf_static \
      -p use_sim_time:=true -p launch_use_sim_time:=true \
      -p shadow_only:=true -p odometry_timeout_sec:=0.30 \
      -p result_dir:="$result_dir/cuvslam" \
      >"$result_dir/logs/cuvslam_supervisor.log" 2>&1 </dev/null &
  cuvslam_pid=$!
  record_pid_event cuvslam "$cuvslam_pid" started "$result_dir/logs/cuvslam_supervisor.log"
fi

if test "$nvblox_mode" != off; then
  setsid env "${common_env[@]}" \
    ros2 run internvla_t4_sensors internvla_t5_nvblox_supervisor --ros-args \
      -r __ns:="$lane_namespace" -r /tf:=tf -r /tf_static:=tf_static \
      -p use_sim_time:=true -p mode:="$nvblox_mode" \
      -p pose_source:=isaac_ground_truth \
      -p result_dir:="$result_dir/nvblox" \
      -p nvblox_params_file:="$nvblox_params" \
      >"$result_dir/logs/nvblox_supervisor.log" 2>&1 </dev/null &
  nvblox_pid=$!
  record_pid_event nvblox "$nvblox_pid" started "$result_dir/logs/nvblox_supervisor.log"
fi

# Model loading and Nav2 bring-up intentionally overlap on the same DGX.  Both
# must become healthy before the x86 evaluator endpoint is exposed.
for _ in $(seq 1 7200); do
  leader_is_alive "$model_pid" || { shutdown_reason=model_startup_exit; exit 1; }
  leader_is_alive "$onboard_pid" || { shutdown_reason=onboard_startup_exit; exit 1; }
  test -z "$nvblox_pid" || leader_is_alive "$nvblox_pid" || \
    { shutdown_reason=nvblox_supervisor_startup_exit; exit 1; }
  test -z "$cuvslam_pid" || leader_is_alive "$cuvslam_pid" || \
    { shutdown_reason=cuvslam_supervisor_startup_exit; exit 1; }
  test -z "$live_frontier_pid" || leader_is_alive "$live_frontier_pid" || \
    { shutdown_reason=live_frontier_startup_exit; exit 1; }
  if test -f "$result_dir/onboard/onboard_ready.json" && \
      test -f "$result_dir/model/model_weight_audit.json"; then
    set +e
    python3 "$root/scripts/t4_model_health_probe.py" \
      --output "$result_dir/health/model_health.tmp.json" --timeout-sec 3 \
      >"$result_dir/logs/model_health_probe.log" 2>&1
    probe_rc=$?
    set -e
    if test "$probe_rc" = 0; then
      mv "$result_dir/health/model_health.tmp.json" \
        "$result_dir/health/model_health.json"
      break
    fi
    rm -f "$result_dir/health/model_health.tmp.json"
  fi
  sleep 1
done
if test "$live_frontier_capture" = 1; then
  leader_is_alive "$live_frontier_pid"
  python3 - "$result_dir/live_frontier/dependency_status.json" \
    "$result_dir/live_frontier/status.json" <<'PY'
import json, sys
from pathlib import Path

dependency = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
runtime = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if dependency.get("status") != "PASS":
    raise SystemExit("live frontier ROS dependency check did not pass")
if runtime.get("status") not in {
    "WAITING_FOR_IDENTITY", "WAITING_FOR_INPUT", "SNAPSHOT_READY", "BLOCKED"
}:
    raise SystemExit("live frontier runtime did not enter a bounded state")
if any(runtime.get(key) != "none" for key in (
    "motion_authority", "terminal_stop_authority", "goal_authority",
    "model_request_authority",
)):
    raise SystemExit("live frontier sidecar acquired forbidden authority")
PY
fi
test -f "$result_dir/onboard/onboard_ready.json"
test -f "$result_dir/model/model_weight_audit.json"
test -f "$result_dir/health/model_health.json"
leader_is_alive "$model_pid"
leader_is_alive "$onboard_pid"
test -z "$nvblox_pid" || leader_is_alive "$nvblox_pid"
test -z "$cuvslam_pid" || leader_is_alive "$cuvslam_pid"

if test "$step3_live_advisor" = 1; then
  internvla_service_pid="$(python3 - "$model_pid" "$root" <<'PY'
import os
import sys
from pathlib import Path

pgid = int(sys.argv[1])
deployment = sys.argv[2].encode()
matches = []
for entry in Path("/proc").iterdir():
    if not entry.name.isdigit():
        continue
    pid = int(entry.name)
    try:
        if os.getpgid(pid) != pgid or entry.stat().st_uid != os.getuid():
            continue
        argv = (entry / "cmdline").read_bytes().split(b"\0")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    if deployment not in b"\0".join(argv):
        continue
    decoded = [value.decode("utf-8", "surrogateescape") for value in argv if value]
    if any(
        decoded[index:index + 2] == ["-m", "internvla_t4_recovery.model_node"]
        for index in range(len(decoded) - 1)
    ):
        matches.append(pid)
if len(matches) != 1:
    raise SystemExit(
        f"expected one InternVLA service PID in model PGID {pgid}, got {matches}"
    )
print(matches[0])
PY
)"
  [[ "$internvla_service_pid" =~ ^[1-9][0-9]*$ ]]
  kill -0 "$internvla_service_pid"
  step3_model_path="${INTERNNAV_T5_STEP3_MODEL_PATH:-/home/rail/ai-stack/models/Step3-VL-10B}"
  step3_venv="${INTERNNAV_T5_STEP3_VENV:-/home/rail/ai-stack/venvs/step3-vl-10b-tf4.57.6}"
  setsid env "${common_env[@]}" \
    INTERNNAV_T5_RESOURCE_LEASE_ACK=lane-b \
    INTERNNAV_T5_INTERNVLA_MODEL_PID="$internvla_service_pid" \
    bash "$root/scripts/run_t5_step3_live_services.sh" \
      "$result_dir" "$step3_model_path" "$step3_venv" \
      >"$result_dir/logs/step3_services.log" 2>&1 </dev/null &
  step3_pid=$!
  record_pid_event step3 "$step3_pid" started "$result_dir/logs/step3_services.log"
  for _ in $(seq 1 4800); do
    leader_is_alive "$step3_pid" || { shutdown_reason=step3_startup_exit; exit 1; }
    test -f "$result_dir/step3/services_ready.json" && break
    sleep 0.2
  done
  test -f "$result_dir/step3/services_ready.json"
  leader_is_alive "$step3_pid"
elif test "$step3_timeout_advisor" = 1; then
  step3_model_path="${INTERNNAV_T5_STEP3_MODEL_PATH:-$HOME/ai-stack/models/Step3-VL-10B}"
  step3_venv="${INTERNNAV_T5_STEP3_VENV:-$HOME/ai-stack/venvs/step3-vl-10b-tf4.57.6}"
  step3_log_path="$result_dir/logs/step3_timeout_service.log"
  setsid env "${common_env[@]}" \
    INTERNNAV_T5_RESOURCE_LEASE_ACK="$expected_lease" \
    bash "$root/scripts/run_t5_step3_timeout_service.sh" \
      "$result_dir/step3_timeout_service" "$step3_model_path" "$step3_venv" \
      >"$step3_log_path" 2>&1 </dev/null &
  step3_pid=$!
  record_pid_event step3 "$step3_pid" started "$step3_log_path"
  for _ in $(seq 1 4800); do
    leader_is_alive "$step3_pid" || { shutdown_reason=step3_startup_exit; exit 1; }
    test -f "$result_dir/step3_timeout_service/health.json" && break
    sleep 0.2
  done
  test -f "$result_dir/step3_timeout_service/health.json"
  leader_is_alive "$step3_pid"
fi

if test "$mode" = model; then
  evaluator_port="$model_client_port"
  setsid env "${common_env[@]}" "${candidate_env[@]}" \
    INTERNVLA_CLIENT_TCP_BIND_HOST="$lane_ip" \
    INTERNVLA_CLIENT_TCP_PORT="$model_client_port" \
    INTERNVLA_CLIENT_TCP_EXPECTED_PEER="$isaac_ip" \
    ros2 run internvla_t4_sensors internvla_t4_client --ros-args \
      --params-file "$params" -r __ns:="$lane_namespace" \
      -r /tf:=tf -r /tf_static:=tf_static \
      -p use_sim_time:=true \
      -p result_dir:="$result_dir/client" \
      -p control_mode:=nav2 -p publish_observation_pose:=false \
      -p navigation_odometry_timeout_sec:=2.0 \
      -p allow_nearest_navigation_odometry:=true \
      -p sensor_future_tolerance_sec:=0.55 \
      >"$result_dir/logs/evaluator.log" 2>&1 </dev/null &
else
  evaluator_port="$oracle_port"
  setsid env "${common_env[@]}" "${candidate_env[@]}" \
    INTERNVLA_ORACLE_TCP_BIND_HOST="$lane_ip" \
    INTERNVLA_ORACLE_TCP_PORT="$oracle_port" \
    INTERNVLA_ORACLE_TCP_EXPECTED_PEER="$isaac_ip" \
    ros2 run internvla_ros2 internvla_nav2_oracle_bridge --ros-args \
      -r __ns:="$lane_namespace" -r /tf:=tf -r /tf_static:=tf_static \
      -p use_sim_time:=true \
      -p publish_observation_pose:=false \
      >"$result_dir/logs/evaluator.log" 2>&1 </dev/null &
fi
client_pid=$!
record_pid_event evaluator "$client_pid" started "$result_dir/logs/evaluator.log"

for _ in $(seq 1 1500); do
  leader_is_alive "$client_pid" || { shutdown_reason=evaluator_startup_exit; exit 1; }
  port_is_free "$evaluator_port" || break
  sleep 0.2
done
leader_is_alive "$client_pid"
test -n "$(ss -H -ltn "sport = :$evaluator_port")"
# The inactive evaluator port is reserved and must remain unused by this lane.
if test "$mode" = model; then
  port_is_free "$oracle_port"
else
  port_is_free "$model_client_port"
fi

# D0 supplies the token by anonymous descriptor.  While the exact value is
# still only a non-exported variable in this coordinator, scan every readable
# /proc/*/environ through a second anonymous descriptor.  The audit contains
# PIDs and process-group classification only—never the token or its digest.
if test "$hf_token_supplied" = 1; then
  model_pgid="$(ps -o pgid= -p "$model_pid" | tr -d ' ')"
  [[ "$model_pgid" =~ ^[1-9][0-9]*$ ]]
  exec {hf_process_audit_fd}<<<"$hf_token"
  python3 "$root/scripts/audit_t5_hf_token_process_scope.py" \
    --output "$result_dir/hf_token_process_audit.json" \
    --secret-fd "$hf_process_audit_fd" \
    --model-pid "$model_pid" --model-pgid "$model_pgid" \
    --parent-pid "$$" --onboard-pid "$onboard_pid" \
    --evaluator-pid "$client_pid"
fi
hf_token=""
unset hf_token

# Re-run the fail-closed data-plane probe after the evaluator joins the graph.
# Besides the transform lookup, this proves the client/oracle introduced no
# root /tf or /tf_static endpoint into the lane domain.
env ROS_DOMAIN_ID="$ros_domain_id" ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
  ROS_STATIC_PEERS="$isaac_ip" \
  python3 "$root/scripts/check_t4_nav2_data_plane.py" \
  --output "$result_dir/nav2_data_plane_evaluator_ready.json" \
  --timeout-sec 20 --namespace "$lane_namespace" \
  >"$result_dir/logs/nav2_data_plane_evaluator_probe.log" 2>&1

graph_discovery_log="$result_dir/logs/ros_graph_discovery.log"
: >"$graph_discovery_log"
query_graph_until_present() {
  local output="$1" label="$2" required_one="$3" required_two="$4"
  local candidate="${output}.candidate" attempt
  shift 4
  for attempt in $(seq 1 15); do
    if "$@" >"$candidate" 2>>"$result_dir/logs/ros_graph_stderr.log"; then
      LC_ALL=C sort -o "$candidate" "$candidate"
      if { test -z "$required_one" || grep -Fxq "$required_one" "$candidate"; } &&
         { test -z "$required_two" || grep -Fxq "$required_two" "$candidate"; }; then
        mv "$candidate" "$output"
        printf '%s\tattempt=%s\tPASS\n' "$label" "$attempt" >>"$graph_discovery_log"
        return 0
      fi
    fi
    printf '%s\tattempt=%s\tRETRY\n' "$label" "$attempt" >>"$graph_discovery_log"
    leader_is_alive "$model_pid"
    leader_is_alive "$onboard_pid"
    leader_is_alive "$client_pid"
    sleep 1
  done
  test ! -f "$candidate" || mv "$candidate" "$output"
  printf '%s\tattempts=15\tFAIL\n' "$label" >>"$graph_discovery_log"
  return 1
}

# A daemon-free ROS CLI process has only a 0.5 second discovery window by
# default.  On both DGX hosts that single window intermittently returned an
# empty or model-only node list even after the independent data-plane probe
# had proved Nav2, TF, services and actions live.  Give every frozen graph
# section a bounded direct-discovery window and retry without weakening the
# required node/service/action identities.
graph_nodes="$result_dir/.ros_graph_nodes.txt"
graph_topics="$result_dir/.ros_graph_topics.txt"
graph_services="$result_dir/.ros_graph_services.txt"
graph_actions="$result_dir/.ros_graph_actions.txt"
query_graph_until_present "$graph_nodes" nodes \
  "$lane_namespace/internvla_model_node" "$lane_namespace/controller_server" \
  ros2 node list --no-daemon --spin-time 2.0
query_graph_until_present "$graph_topics" topics '' '' \
  ros2 topic list -t --no-daemon --spin-time 2.0
query_graph_until_present "$graph_services" services \
  '/internvla/health [internvla_ros2_msgs/srv/Health]' '' \
  ros2 service list -t --no-daemon --spin-time 2.0
query_graph_until_present "$graph_actions" actions \
  '/internvla/step [internvla_ros2_msgs/action/Step]' '' \
  ros2 action list -t
{
  echo "# nodes"
  cat "$graph_nodes"
  echo "# topics"
  cat "$graph_topics"
  echo "# services"
  cat "$graph_services"
  echo "# actions"
  cat "$graph_actions"
} >"$result_dir/ros_graph_snapshot.txt"
rm -f "$graph_nodes" "$graph_topics" "$graph_services" "$graph_actions"
grep -Fxq "$lane_namespace/internvla_model_node" "$result_dir/ros_graph_snapshot.txt"
grep -Fxq "$lane_namespace/controller_server" "$result_dir/ros_graph_snapshot.txt"
grep -Fxq '/internvla/health [internvla_ros2_msgs/srv/Health]' \
  "$result_dir/ros_graph_snapshot.txt"
grep -Fxq '/internvla/step [internvla_ros2_msgs/action/Step]' \
  "$result_dir/ros_graph_snapshot.txt"

parameter_tsv="$result_dir/parameter_audit.tsv"
: >"$parameter_tsv"
audit_parameter() {
  local node="$1" parameter="$2" expected="$3" output="" actual attempt
  for attempt in $(seq 1 5); do
    if output="$(timeout --signal=TERM --kill-after=0.5s 6s \
        ros2 param get "$node" "$parameter" \
        2>>"$result_dir/logs/parameter_audit_stderr.log")"; then
      actual="${output#*: }"
      printf '%s\t%s\t%s\t%s\n' \
        "$node" "$parameter" "$expected" "$actual" >>"$parameter_tsv"
      test "$actual" = "$expected"
      return
    fi
    printf 'parameter probe retry: node=%s parameter=%s attempt=%s/5\n' \
      "$node" "$parameter" "$attempt" \
      >>"$result_dir/logs/parameter_audit_stderr.log"
    test "$attempt" = 5 || sleep 1
  done
  printf '%s\t%s\t%s\t%s\n' \
    "$node" "$parameter" "$expected" UNAVAILABLE >>"$parameter_tsv"
  return 1
}
audit_parameter "$lane_namespace/internvla_model_node" observation_transport raw
audit_parameter "$lane_namespace/internvla_model_node" observation_wait_sec 5.0
if test "$mode" = model; then
  audit_parameter "$lane_namespace/internvla_client_node" observation_transport raw
  audit_parameter "$lane_namespace/internvla_client_node" step_deadline_sec 30.0
  audit_parameter "$lane_namespace/internvla_client_node" discovery_timeout_sec 1200.0
else
  audit_parameter "$lane_namespace/internvla_nav2_oracle_bridge" \
    resolution_timeout_sec 10.0
  audit_parameter "$lane_namespace/internvla_nav2_oracle_bridge" \
    publish_observation_pose False
fi
python3 - "$parameter_tsv" "$result_dir/parameter_audit.json" <<'PY'
import json, sys
from pathlib import Path

rows=[]
for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    node, parameter, expected, actual = line.split("\t")
    rows.append({"node": node, "parameter": parameter, "expected": expected,
                 "actual": actual, "match": actual == expected})
payload={"schema_version":1,"status":"PASS" if rows and all(r["match"] for r in rows) else "FAIL",
         "parameters":rows}
Path(sys.argv[2]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
if payload["status"] != "PASS": raise SystemExit("T5 parameter audit failed")
PY

python3 - "$golden_bundle" "$result_dir/model/model_weight_audit.json" \
  "$result_dir/internnav_source_identity_audit.json" \
  "$result_dir/model_identity_audit.json" <<'PY'
import hashlib, json, sys
from pathlib import Path

golden_path, weight_path, source_path, output = map(Path, sys.argv[1:5])
golden = json.loads(golden_path.read_text(encoding='utf-8'))
weight = json.loads(weight_path.read_text(encoding='utf-8'))
source = json.loads(source_path.read_text(encoding='utf-8'))
model = golden['model']
canonical = json.dumps(
    golden, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
    allow_nan=False,
).encode('ascii')
checks = {
    'weight_status': weight.get('status') == 'PASS',
    'source_identity': source.get('status') == 'PASS'
        and bool(source.get('checks')) and all(source['checks'].values()),
    'real_backend': weight.get('backend') == 'real',
    'model_revision': weight.get('model_revision') == model['internnav_revision'],
    'checkpoint_revision': weight.get('checkpoint_revision') == model['checkpoint_revision'],
    'runtime_weight_inventory': weight.get('inventory_sha256')
        == model['weight_inventory_sha256'],
    'parameter_tensor_count': weight.get('parameter_tensor_count')
        == model['strict_load_contract']['parameter_tensor_count'],
    'buffer_tensor_count': weight.get('buffer_tensor_count')
        == model['strict_load_contract']['buffer_tensor_count'],
    'no_meta_parameters': weight.get('meta_parameter_count')
        == model['strict_load_contract']['meta_parameter_count'],
    'no_meta_buffers': weight.get('meta_buffer_count')
        == model['strict_load_contract']['meta_buffer_count'],
}
payload = dict(weight)
payload.update({
    'schema_version': 2,
    'status': 'PASS' if all(checks.values()) else 'FAIL',
    'golden_bundle_canonical_sha256': hashlib.sha256(canonical).hexdigest(),
    'expected_runtime_weight_inventory_sha256': model['weight_inventory_sha256'],
    'checks': checks,
    'internnav_source_identity': source,
})
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
if payload['status'] != 'PASS':
    raise SystemExit(f'Golden runtime model identity mismatch: {checks}')
PY

python3 - "$result_dir/lane_ready.json" "$result_dir/lane_contract.json" \
  "$result_dir/onboard/onboard_ready.json" "$result_dir/health/model_health.json" \
  "$result_dir/model_identity_audit.json" "$result_dir/parameter_audit.json" \
  "$result_dir/nav2_data_plane_evaluator_ready.json" \
  "$lane" "$mode" "$lane_ip" \
  "$evaluator_port" "$ros_domain_id" "$lane_namespace" "$identity_prefix" \
  "$candidate_profile" "$effective_candidate_profile" \
  "$result_dir/candidate_resolution.json" "$live_frontier_capture" <<'PY'
import hashlib, json, os, sys, time
from pathlib import Path

contract, onboard, health, weight, parameters, data_plane = [
    json.loads(Path(value).read_text(encoding="utf-8")) for value in sys.argv[2:8]
]
candidate = json.loads(Path(sys.argv[17]).read_text(encoding="utf-8"))
candidate_resolution_sha256 = candidate.pop("resolution_sha256", None)
candidate_canonical = json.dumps(
    candidate, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    allow_nan=False,
).encode("utf-8")
candidate_binding = candidate.get("canonical_binding")
live_frontier_enabled = sys.argv[18] == "1"
live_frontier_status_path = Path(sys.argv[1]).parent / "live_frontier/status.json"
try:
    live_frontier_status = json.loads(
        live_frontier_status_path.read_text(encoding="utf-8")
    ) if live_frontier_enabled else None
except (OSError, UnicodeError, json.JSONDecodeError):
    live_frontier_status = None
checks = {
    "contract_lane": contract.get("lane") == sys.argv[8],
    "contract_mode": contract.get("mode") == sys.argv[9],
    "onboard_ready": onboard.get("status") == "READY",
    "model_local": onboard.get("model_owner") == "local_dgx",
    "sim_time": onboard.get("use_sim_time") is True,
    "model_health": health.get("status") == "PASS",
    "model_action": health.get("step_action_ready") is True,
    "weight_audit": weight.get("status") == "PASS",
    "weight_materialized": weight.get("meta_parameter_count") == 0
        and weight.get("meta_buffer_count") == 0,
    "parameter_audit": parameters.get("status") == "PASS",
    "nav2_data_plane": data_plane.get("status") == "PASS",
    "tf_chain": data_plane.get("transforms_pass") is True,
    "root_tf_isolated": data_plane.get("root_tf_isolated") is True,
    "onboard_namespace": onboard.get("namespace") == sys.argv[13],
    "contract_candidate_profile": contract.get("candidate_profile") == sys.argv[15],
    "contract_effective_candidate_profile":
        contract.get("effective_candidate_profile") == sys.argv[16],
    "candidate_resolution_self_hash": candidate_resolution_sha256
        == hashlib.sha256(candidate_canonical).hexdigest(),
    "candidate_resolution_selector": candidate.get("candidate_selector")
        == sys.argv[15],
    "candidate_resolution_effective_profile":
        candidate.get("effective_candidate_profile") == sys.argv[16],
    "contract_candidate_resolution_sha":
        contract.get("candidate_resolution_sha256") == candidate_resolution_sha256,
    "contract_candidate_binding": isinstance(candidate_binding, dict)
        and contract.get("candidate_binding") == candidate_binding,
    "contract_live_frontier_capture": isinstance(
        contract.get("live_frontier_capture"), dict
    ) and contract["live_frontier_capture"].get("enabled")
        is live_frontier_enabled,
    "live_frontier_bounded_zero_authority": (not live_frontier_enabled) or (
        isinstance(live_frontier_status, dict)
        and live_frontier_status.get("status") in {
            "WAITING_FOR_IDENTITY", "WAITING_FOR_INPUT", "SNAPSHOT_READY", "BLOCKED"
        }
        and all(live_frontier_status.get(key) == "none" for key in (
            "motion_authority", "terminal_stop_authority", "goal_authority",
            "model_request_authority",
        ))
    ),
}
if not all(checks.values()):
    raise SystemExit(f"lane readiness failed: {checks}")
Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": 1,
    "status": "READY",
    "lane": sys.argv[8],
    "mode": sys.argv[9],
    "host": os.uname().nodename,
    "bind_ip": sys.argv[10],
    "evaluator_endpoint": f"tcp://{sys.argv[10]}:{sys.argv[11]}",
    "health_endpoint": f"ros2://domain-{sys.argv[12]}/internvla/health",
    "stop_request": str(Path(sys.argv[1]).parent / "stop.request"),
    "ros_domain_id": int(sys.argv[12]),
    "namespace": sys.argv[13],
    "identity_prefix": sys.argv[14],
    "candidate_profile": sys.argv[15],
    "effective_candidate_profile": sys.argv[16],
    "candidate_resolution_sha256": candidate_resolution_sha256,
    "candidate_binding": candidate_binding,
    "dds": {
        "automatic_discovery_range": "LOCALHOST",
        "static_peers": [contract["expected_isaac_peer_ip"]],
    },
    "use_sim_time": True,
    "model_and_nav2_colocated": True,
    "golden_bundle_canonical_sha256": weight.get(
        "golden_bundle_canonical_sha256"
    ),
    "runtime_weight_inventory_sha256": weight.get("inventory_sha256"),
    "live_frontier_capture": {
        "enabled": live_frontier_enabled,
        "runtime_status": (live_frontier_status or {}).get("status"),
        "snapshot_ready_count": int(
            (live_frontier_status or {}).get("snapshot_ready_count", 0)
        ),
        "status_path": "live_frontier/status.json",
        "snapshot_path": "live_frontier/current.json",
    },
    "checks": checks,
    "ready_unix": time.time(),
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
online_ready=1

write_fault_ack() {
  local output="$1" status="$2" event_id="$3" action="$4" \
    old_pid="$5" new_pid="$6" safe_stop="$7" residual="$8" message="$9"
  local session_continuity="${10:-0}" session_sha256="${11:-}"
  python3 - "$output" "$status" "$lane" "$event_id" "$action" \
    "$old_pid" "$new_pid" "$safe_stop" "$residual" "$message" \
    "$session_continuity" "$session_sha256" <<'PY'
import json, os, sys, time
from pathlib import Path

output = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "profile": "completion_sim_minimal_v1",
    "status": sys.argv[2],
    "lane": sys.argv[3],
    "event_id": sys.argv[4],
    "action": sys.argv[5],
    "old_pid": int(sys.argv[6]),
    "new_pid": int(sys.argv[7]) if sys.argv[7] else None,
    "safe_stop_confirmed": sys.argv[8] == "1",
    "residual_after_stop": int(sys.argv[9]),
    "message": sys.argv[10],
    "session_continuity_confirmed": sys.argv[11] == "1",
    "session_sha256": sys.argv[12] or None,
    "wall_unix": time.time(),
}
temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, output)
PY
}

capture_fault_restart_session() {
  local event_id="$1" action="$2" session_path="$3" log_path="$4"
  env "${common_env[@]}" \
    python3 "$root/scripts/t5_fault_restart_session_probe.py" capture \
      --output "$session_path" --lane "$lane" --event-id "$event_id" \
      --action "$action" --client-summary "$result_dir/client/client_summary.json" \
      --timeout-sec 10 \
      >>"$log_path" 2>&1
}

verify_fault_restart_session() {
  local session_path="$1" receipt_path="$2" log_path="$3"
  env "${common_env[@]}" \
    python3 "$root/scripts/t5_fault_restart_session_probe.py" verify \
      --session "$session_path" --output "$receipt_path" --lane "$lane" \
      --client-summary "$result_dir/client/client_summary.json" \
      --timeout-sec 10 \
      >>"$log_path" 2>&1
}

fault_safe_stop() {
  local event_id="$1"
  timeout 15 env "${common_env[@]}" \
    ros2 topic pub --once /internvla/stop std_msgs/msg/Bool '{data: true}' \
    >"$result_dir/logs/fault_safe_stop_${event_id}.log" 2>&1
}

parse_fault_request() {
  python3 - "$1" "$lane" <<'PY'
import json, re, sys
from pathlib import Path

path = Path(sys.argv[1])
value = json.loads(path.read_text(encoding="utf-8"))
event_id = value.get("event_id")
action = value.get("action")
if (
    value.get("schema_version") != 1
    or value.get("profile") != "completion_sim_minimal_v1"
    or value.get("lane") != sys.argv[2]
    or action not in {"model_service_restart", "dgx_ros_node_restart"}
    or not isinstance(event_id, str)
    or not re.fullmatch(r"fi-[a-z0-9-]{3,90}", event_id)
    or path.name != event_id + ".json"
    or isinstance(value.get("requested_sim_ns"), bool)
    or not isinstance(value.get("requested_sim_ns"), int)
    or value["requested_sim_ns"] <= 0
):
    raise SystemExit("invalid T5 fault restart request")
print(event_id)
print(action)
PY
}

restart_model_for_fault() {
  local event_id="$1" old_pid="$model_pid" session_path receipt_path session_sha256
  fault_safe_stop "$event_id" || return 1
  session_path="$result_dir/fault_control/sessions/$event_id.json"
  receipt_path="$result_dir/fault_control/sessions/$event_id.restore.json"
  capture_fault_restart_session "$event_id" model_service_restart \
    "$session_path" "$model_log_path" || return 1
  stop_group model "$old_pid" "$model_log_path" || return 1
  model_restart_count=$((model_restart_count + 1))
  model_result_dir="$result_dir/fault_control/model_restart_$(printf '%03d' "$model_restart_count")"
  model_log_path="$result_dir/logs/model_restart_$(printf '%03d' "$model_restart_count").log"
  model_restart_session_file="$session_path"
  start_model "$model_result_dir" "$model_log_path"
  for _ in $(seq 1 7200); do
    leader_is_alive "$model_pid" || return 1
    if test -f "$model_result_dir/model_weight_audit.json"; then
      set +e
      verify_fault_restart_session "$session_path" \
        "$receipt_path" "$model_log_path"
      local probe_rc=$?
      set -e
      if test "$probe_rc" = 0; then
        cp "$receipt_path" "$model_result_dir/model_health.json"
        session_sha256="$(sha256sum "$session_path" | awk '{print $1}')"
        write_fault_ack "$result_dir/fault_control/acks/$event_id.json" \
          PASS "$event_id" model_service_restart "$old_pid" "$model_pid" 1 0 \
          "model service restarted with its quiesced session" 1 "$session_sha256"
        return 0
      fi
    fi
    sleep 1
  done
  return 1
}

restart_onboard_for_fault() {
  local event_id="$1" old_pid="$onboard_pid" old_result_dir="$onboard_result_dir"
  local session_path receipt_path session_sha256
  fault_safe_stop "$event_id" || return 1
  session_path="$result_dir/fault_control/sessions/$event_id.json"
  receipt_path="$result_dir/fault_control/sessions/$event_id.restore.json"
  capture_fault_restart_session "$event_id" dgx_ros_node_restart \
    "$session_path" "$onboard_log_path" || return 1
  stop_group onboard "$old_pid" "$onboard_log_path" || return 1
  python3 -c 'import json,sys;v=json.load(open(sys.argv[1],encoding="utf-8"));assert int(v.get("residual_process_group_count",-1))==0' \
    "$old_result_dir/onboard_status.json" || return 1
  onboard_restart_count=$((onboard_restart_count + 1))
  onboard_result_dir="$result_dir/fault_control/onboard_restart_$(printf '%03d' "$onboard_restart_count")"
  onboard_log_path="$result_dir/logs/onboard_restart_$(printf '%03d' "$onboard_restart_count").log"
  start_onboard "$onboard_result_dir" "$onboard_log_path" "$session_path"
  for _ in $(seq 1 1800); do
    leader_is_alive "$onboard_pid" || return 1
    if test -f "$onboard_result_dir/onboard_ready.json" && \
        test -n "$(ss -H -ltn "sport = :$controller_port")"; then
      set +e
      env ROS_DOMAIN_ID="$ros_domain_id" ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
        ROS_STATIC_PEERS="$isaac_ip" \
        python3 "$root/scripts/check_t4_nav2_data_plane.py" \
          --output "$onboard_result_dir/nav2_data_plane_restart.json" \
          --timeout-sec 20 --namespace "$lane_namespace" \
          >>"$onboard_log_path" 2>&1
      local probe_rc=$?
      set -e
      if test "$probe_rc" = 0; then
        set +e
        verify_fault_restart_session "$session_path" "$receipt_path" \
          "$onboard_log_path"
        local session_probe_rc=$?
        set -e
        test "$session_probe_rc" = 0 || return 1
        session_sha256="$(sha256sum "$session_path" | awk '{print $1}')"
        write_fault_ack "$result_dir/fault_control/acks/$event_id.json" \
          PASS "$event_id" dgx_ros_node_restart "$old_pid" "$onboard_pid" 1 0 \
          "DGX onboard ROS stack restarted with model session continuity" \
          1 "$session_sha256"
        return 0
      fi
    fi
    sleep 1
  done
  return 1
}

process_fault_requests() {
  test "$fault_injection_profile" != off || return 0
  local request event_id action ack old_pid
  local -a parsed=()
  shopt -s nullglob
  for request in "$result_dir"/fault_control/requests/*.json; do
    parsed=()
    mapfile -t parsed < <(parse_fault_request "$request") || return 1
    test "${#parsed[@]}" = 2 || return 1
    event_id="${parsed[0]}"
    action="${parsed[1]}"
    ack="$result_dir/fault_control/acks/$event_id.json"
    test ! -e "$ack" || continue
    old_pid="$([[ $action = model_service_restart ]] && printf '%s' "$model_pid" || printf '%s' "$onboard_pid")"
    if test "$action" = model_service_restart; then
      restart_model_for_fault "$event_id" || {
        write_fault_ack "$ack" FAIL "$event_id" "$action" "$old_pid" "" 0 1 \
          "bounded model restart failed"
        return 1
      }
    else
      restart_onboard_for_fault "$event_id" || {
        write_fault_ack "$ack" FAIL "$event_id" "$action" "$old_pid" "" 0 1 \
          "bounded onboard restart failed"
        return 1
      }
    fi
  done
  shopt -u nullglob
}

while :; do
  process_fault_requests || { shutdown_reason=fault_restart_failed; exit 1; }
  leader_is_alive "$model_pid" || { shutdown_reason=model_exit; exit 1; }
  leader_is_alive "$onboard_pid" || { shutdown_reason=onboard_exit; exit 1; }
  test -z "$step3_pid" || leader_is_alive "$step3_pid" || \
    { shutdown_reason=step3_exit; exit 1; }
  test -z "$nvblox_pid" || leader_is_alive "$nvblox_pid" || \
    { shutdown_reason=nvblox_supervisor_exit; exit 1; }
  test -z "$cuvslam_pid" || leader_is_alive "$cuvslam_pid" || \
    { shutdown_reason=cuvslam_supervisor_exit; exit 1; }
  test -z "$live_frontier_pid" || leader_is_alive "$live_frontier_pid" || \
    { shutdown_reason=live_frontier_exit; exit 1; }
  if ! leader_is_alive "$client_pid"; then
    handle_evaluator_exit || exit 1
    exit 0
  fi
  test ! -f "$result_dir/stop.request" || {
    shutdown_reason=coordinator_stop
    exit 0
  }
  sleep 1
done
