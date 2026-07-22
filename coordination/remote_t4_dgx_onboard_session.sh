#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

[[ $# -eq 6 || $# -eq 7 ]] || { echo "remote DGX onboard session: invalid arguments" >&2; exit 64; }
action="$1"
deployment_root="$2"
stage="$3"
grant_id="$4"
authorization_ref="$5"
deployment_ref="$6"
run_profile="${7:-migration_smoke}"
ablation_variant=none
case "$action" in start|stop) ;; *) exit 64 ;; esac
case "$run_profile" in
  migration_smoke|migrated_oracle10|migrated_pilot20|migrated_recovery_a5|migrated_recovery_b5) ;;
  migrated_ablation20_*)
    ablation_variant="${run_profile#migrated_ablation20_}"
    case "$ablation_variant" in
      full_system1_system2|oracle_high_level_system1|system2_oracle_local_path|full_trajectory|endpoint|straight_line|model_stop|oracle_termination|history_on|history_off|recovery_on|recovery_off|h1_view|go2_view) ;;
      *) exit 64 ;;
    esac
    ;;
  *) exit 64 ;;
esac
# Metric RGB-D is owned by internvla_t4_sensor_bridge.  The independent Go2
# bridge remains a non-fatal LiDAR/IMU shadow and must not duplicate D435 input.
enable_d435i=0
[[ "$grant_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{7,95}$ ]] || exit 64
[[ "$authorization_ref" =~ ^[0-9a-f]{40}$ ]] || exit 64
[[ "$deployment_ref" =~ ^[0-9a-f]{40}$ ]] || exit 64

readonly EXPECTED_USER="${INTERNNAV_T4_EXPECTED_USER:-railgun}"
readonly EXPECTED_IP="${INTERNNAV_T4_EXPECTED_IP:-10.100.100.128}"
readonly STABLE_ROOT="${INTERNNAV_T4_STABLE_ROOT:-/home/railgun/internnav-t1-t2}"
readonly STAGE_BASE="${INTERNNAV_T4_STAGE_BASE:-/home/railgun/.codex-internnav-stage}"
readonly DGX_BIND_IP="${INTERNNAV_T4_DGX_BIND_IP:-10.100.100.128}"
readonly ISAAC_PEER_IP="${INTERNNAV_T4_ISAAC_PEER_IP:-10.100.120.111}"
readonly CONTROLLER_PORT="${INTERNVLA_T4_CONTROLLER_TCP_PORT:-24137}"
readonly LANE_ROS_DOMAIN_ID="${INTERNNAV_T4_ROS_DOMAIN_ID:-71}"
readonly EXPECTED_STAGE="$STAGE_BASE/t4-dgx-onboard-${grant_id}"
readonly ROS_WORKSPACE="$deployment_root/ros_ws"
readonly RESULT_DIR="$deployment_root/results/internnav_t4/dgx-onboard-${grant_id}"
readonly CONTRACT_TOOL="$deployment_root/coordination/t4_functional_run_contract.py"
readonly PROCESS_RECORD="$stage/onboard_process.json"
readonly MAP_DIR="$stage/static_maps"
readonly ABLATION_CONFIG="$stage/ablation_configs/variants/${ablation_variant}.json"
readonly ABLATION_DATASET="$stage/smoke_dataset/val_unseen/val_unseen.json.gz"

test "$(id -un)" = "$EXPECTED_USER"
ip -4 -o addr show | grep -Fq " $EXPECTED_IP/"
[[ "$CONTROLLER_PORT" =~ ^[1-9][0-9]{3,4}$ ]]
[[ "$LANE_ROS_DOMAIN_ID" =~ ^[0-9]{1,3}$ ]]
test "$stage" = "$EXPECTED_STAGE"
test -d "$stage" && test ! -L "$stage"
test "$(realpath "$stage")" = "$EXPECTED_STAGE"
test -d "$deployment_root" && test ! -L "$deployment_root"
case "$deployment_root" in "$STABLE_ROOT/.t4-deployments/"*) ;; *) exit 64 ;; esac
test "$(realpath "$deployment_root")" = "$deployment_root"
test -x "$CONTRACT_TOOL"
test -x "$deployment_root/scripts/run_t4_dgx_onboard.sh"

if test "$action" = start; then
  test -f "$MAP_DIR/manifest.json"
  python3 "$CONTRACT_TOOL" validate-deployment \
    --role dgx --deployment-root "$deployment_root" --deployment-ref "$deployment_ref" \
    >"$stage/deployment_validation.json"
  python3 "$CONTRACT_TOOL" validate-dgx-workspace \
    --deployment-root "$deployment_root" --workspace "$ROS_WORKSPACE" \
    >"$stage/dgx_workspace_validation.json"
  test ! -e "$RESULT_DIR"
  mkdir -p "$(dirname "$RESULT_DIR")"
  enable_recovery=0
  recovery_mode=off
  nav2_params="$deployment_root/configs/completion_sim/map/nav2_static_lidar.yaml"
  recovery_runtime_manifest=""
  recovery_profile_id=completion-default
  recovery_profile_sha256=none
  recovery_nav2_sha256=none
  recovery_progress_horizon=5.0
  recovery_minimum_progress=0.08
  recovery_oscillation_travel=0.30
  recovery_cooldown=3.0
  recovery_maximum_count=3
  recovery_scan_yaw=1.0471975511965976
  recovery_scan_speed=0.35
  recovery_safety_freshness=1.0
  recovery_maximum_duration=60.0
  recovery_enable_scheduled_refresh=0
  recovery_maximum_scheduled_refreshes=1
  variant_id=none
  variant_config_sha256=none
  matrix_sha256=none
  system_mode=full_system1_system2
  trajectory_mode=full_trajectory
  termination_mode=model_stop
  history_mode=on
  recovery_mode=off
  view_mode=go2_view
  ablation_dataset=""
  if test "$ablation_variant" != none; then
    test -f "$ABLATION_CONFIG"
    test -f "$ABLATION_DATASET"
    mapfile -t ablation_fields < <(
      python3 "$deployment_root/scripts/t4_ablation_runtime.py" fields \
        --config "$ABLATION_CONFIG" \
        --matrix "$deployment_root/configs/completion_sim/ablation/frozen_matrix_v1.json"
    )
    test "${#ablation_fields[@]}" -eq 11
    test "${ablation_fields[0]}" = "$ablation_variant"
    variant_id="${ablation_fields[0]}"
    variant_config_sha256="${ablation_fields[1]}"
    matrix_sha256="${ablation_fields[2]}"
    system_mode="${ablation_fields[3]}"
    trajectory_mode="${ablation_fields[4]}"
    termination_mode="${ablation_fields[5]}"
    history_mode="${ablation_fields[6]}"
    recovery_mode="${ablation_fields[7]}"
    view_mode="${ablation_fields[8]}"
    ablation_dataset="$ABLATION_DATASET"
  fi
  case "$run_profile" in
    migrated_recovery_a5) recovery_profile="$deployment_root/configs/completion_sim/recovery/profile_a.json" ;;
    migrated_recovery_b5) recovery_profile="$deployment_root/configs/completion_sim/recovery/profile_b.json" ;;
    *) recovery_profile="" ;;
  esac
  if test "$ablation_variant" != none && test "$recovery_mode" = on; then
    # The frozen recovery factor uses the profile selected by the completed
    # A/B gate; completion_sim profile A is the sole selected implementation.
    recovery_profile="$deployment_root/configs/completion_sim/recovery/profile_a.json"
  fi
  if test -n "$recovery_profile"; then
    enable_recovery=1
    recovery_mode=on
    nav2_params="$stage/recovery_nav2_params.yaml"
    recovery_runtime_manifest="$stage/recovery_runtime_manifest.json"
    mapfile -t recovery_fields < <(
      python3 "$deployment_root/scripts/t4_recovery_runtime.py" \
        --profile "$recovery_profile" \
        --nav2-input "$deployment_root/configs/completion_sim/map/nav2_static_lidar.yaml" \
        --nav2-output "$nav2_params" \
        --manifest "$recovery_runtime_manifest" --format fields
    )
    test "${#recovery_fields[@]}" -eq 12
    recovery_profile_id="${recovery_fields[0]}"
    recovery_profile_sha256="${recovery_fields[1]}"
    recovery_nav2_sha256="${recovery_fields[2]}"
    recovery_progress_horizon="${recovery_fields[3]}"
    recovery_minimum_progress="${recovery_fields[4]}"
    recovery_oscillation_travel="${recovery_fields[5]}"
    recovery_cooldown="${recovery_fields[6]}"
    recovery_maximum_count="${recovery_fields[7]}"
    recovery_scan_yaw="${recovery_fields[8]}"
    recovery_scan_speed="${recovery_fields[9]}"
    recovery_safety_freshness="${recovery_fields[10]}"
    recovery_maximum_duration="${recovery_fields[11]}"
    recovery_enable_scheduled_refresh=1
  fi
  onboard_pid=""
  cleanup_failed_start() {
    local rc=$?
    trap - EXIT INT TERM HUP
    set +e
    if [[ "$onboard_pid" =~ ^[1-9][0-9]*$ ]]; then
      kill -TERM -- "-$onboard_pid" 2>/dev/null || true
      for _ in $(seq 1 600); do
        kill -0 "$onboard_pid" 2>/dev/null || break
        sleep 0.1
      done
      kill -0 "$onboard_pid" 2>/dev/null && \
        kill -KILL -- "-$onboard_pid" 2>/dev/null || true
      wait "$onboard_pid" 2>/dev/null || true
    fi
    exit "$rc"
  }
  trap cleanup_failed_start EXIT
  trap 'exit 130' INT TERM HUP

  setsid env \
    INTERNNAV_T1_CONTROL_ROOT="$deployment_root" \
    INTERNVLA_ROS_WS="$ROS_WORKSPACE" \
    INTERNNAV_RUNTIME_POLICY=completion_sim \
    INTERNNAV_SIMULATION_TARGET=isaac \
    INTERNNAV_T4_RESOURCE_LEASE_ACK=dgx+isaac \
    ROS_DOMAIN_ID="$LANE_ROS_DOMAIN_ID" \
    INTERNVLA_T4_DGX_BIND_IP="$DGX_BIND_IP" \
    INTERNVLA_T4_ISAAC_PEER_IP="$ISAAC_PEER_IP" \
    INTERNVLA_T4_CONTROLLER_TCP_PORT="$CONTROLLER_PORT" \
    INTERNVLA_T4_ENABLE_D435I="$enable_d435i" \
    INTERNVLA_T4_ENABLE_RECOVERY="$enable_recovery" \
    INTERNVLA_T4_RECOVERY_MODE="$recovery_mode" \
    INTERNVLA_NAV2_PARAMS="$nav2_params" \
    INTERNVLA_T4_RECOVERY_RUNTIME_MANIFEST="$recovery_runtime_manifest" \
    INTERNVLA_T4_RECOVERY_PROFILE_ID="$recovery_profile_id" \
    INTERNVLA_T4_RECOVERY_PROFILE_SHA256="$recovery_profile_sha256" \
    INTERNVLA_T4_RECOVERY_NAV2_SHA256="$recovery_nav2_sha256" \
    INTERNVLA_T4_PROGRESS_HORIZON_SEC="$recovery_progress_horizon" \
    INTERNVLA_T4_MINIMUM_PROGRESS_M="$recovery_minimum_progress" \
    INTERNVLA_T4_OSCILLATION_TRAVEL_M="$recovery_oscillation_travel" \
    INTERNVLA_T4_RECOVERY_COOLDOWN_SEC="$recovery_cooldown" \
    INTERNVLA_T4_MAXIMUM_RECOVERIES="$recovery_maximum_count" \
    INTERNVLA_T4_RECOVERY_SCAN_YAW_RAD="$recovery_scan_yaw" \
    INTERNVLA_T4_RECOVERY_SCAN_SPEED_RPS="$recovery_scan_speed" \
    INTERNVLA_T4_RECOVERY_SAFETY_FRESHNESS_SEC="$recovery_safety_freshness" \
    INTERNVLA_T4_MAXIMUM_RECOVERY_DURATION_SEC="$recovery_maximum_duration" \
    INTERNVLA_T4_ENABLE_SCHEDULED_REFRESH="$recovery_enable_scheduled_refresh" \
    INTERNVLA_T4_MAXIMUM_SCHEDULED_REFRESHES="$recovery_maximum_scheduled_refreshes" \
    INTERNVLA_T4_VARIANT_ID="$variant_id" \
    INTERNVLA_T4_VARIANT_CONFIG_SHA256="$variant_config_sha256" \
    INTERNVLA_T4_MATRIX_SHA256="$matrix_sha256" \
    INTERNVLA_T4_SYSTEM_MODE="$system_mode" \
    INTERNVLA_T4_TRAJECTORY_MODE="$trajectory_mode" \
    INTERNVLA_T4_TERMINATION_MODE="$termination_mode" \
    INTERNVLA_T4_HISTORY_MODE="$history_mode" \
    INTERNVLA_T4_RECOVERY_MODE="$recovery_mode" \
    INTERNVLA_T4_VIEW_MODE="$view_mode" \
    INTERNVLA_T4_ABLATION_DATASET_FILE="$ablation_dataset" \
    INTERNVLA_T4_ONBOARD_PROFILE="$run_profile" \
    bash "$deployment_root/scripts/run_t4_dgx_onboard.sh" \
      --result-dir "$RESULT_DIR" \
      --static-map-manifest "$MAP_DIR/manifest.json" \
      >"$stage/onboard_outer.log" 2>&1 </dev/null &
  onboard_pid=$!
  onboard_pgid="$(ps -o pgid= -p "$onboard_pid" | tr -d ' ')"
  test "$onboard_pgid" = "$onboard_pid"
  kill -0 "$onboard_pid"
  python3 - "$PROCESS_RECORD" "$onboard_pid" "$onboard_pgid" \
    "$deployment_root" "$RESULT_DIR" "$grant_id" <<'PY'
import json,sys,time
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "schema_version":1,"status":"RUNNING","pid":int(sys.argv[2]),
    "pgid":int(sys.argv[3]),"deployment_root":sys.argv[4],
    "result_dir":sys.argv[5],"grant_id":sys.argv[6],"started_unix":time.time(),
},indent=2,sort_keys=True)+"\n",encoding="utf-8")
PY
  ready=0
  for _ in $(seq 1 600); do
    kill -0 "$onboard_pid" 2>/dev/null || break
    if test -f "$RESULT_DIR/onboard_ready.json"; then ready=1; break; fi
    sleep 0.1
  done
  test "$ready" = 1
  kill -0 "$onboard_pid"
  python3 - "$RESULT_DIR/onboard_ready.json" "$stage/onboard_ready_receipt.json" \
    "$grant_id" "$authorization_ref" "$deployment_ref" "$deployment_root" \
    "$run_profile" <<'PY'
import json,sys,time
from pathlib import Path
ready=json.load(open(sys.argv[1],encoding="utf-8"))
if ready.get("status") != "READY" or ready.get("host_role") != "dgx_onboard_compute":
    raise SystemExit("DGX onboard readiness mismatch")
payload={"schema_version":1,"status":"PASS","grant_id":sys.argv[3],
"authorization_ref_sha":sys.argv[4],"deployment_ref_sha":sys.argv[5],
"deployment_root":sys.argv[6],"resource_lease_ack":"dgx+isaac",
"run_profile":sys.argv[7],
"navigation_owner":"dgx","map_owner":"dgx","speed_control_owner":"dgx",
"strict_evidence_modified":False,"real_go2_targeted":False,"recorded_unix":time.time()}
Path(sys.argv[2]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
PY
  trap - EXIT INT TERM HUP
  exit 0
fi

test -f "$PROCESS_RECORD" && test ! -L "$PROCESS_RECORD"
mapfile -t fields < <(python3 - "$PROCESS_RECORD" "$deployment_root" "$grant_id" <<'PY'
import json,sys
value=json.load(open(sys.argv[1],encoding="utf-8"))
if value.get("deployment_root") != sys.argv[2] or value.get("grant_id") != sys.argv[3]:
    raise SystemExit("onboard process identity changed")
pid=int(value.get("pid",0)); pgid=int(value.get("pgid",0))
if pid <= 1 or pgid != pid: raise SystemExit("unsafe onboard process group")
print(pid); print(pgid); print(value.get("result_dir",""))
PY
)
test "${#fields[@]}" -eq 3
onboard_pid="${fields[0]}"; onboard_pgid="${fields[1]}"
test "${fields[2]}" = "$RESULT_DIR"
was_alive=0
kill -0 "$onboard_pid" 2>/dev/null && was_alive=1
if test "$was_alive" = 1; then
  command_text="$(tr '\0' ' ' <"/proc/$onboard_pid/cmdline")"
  [[ "$command_text" == *"run_t4_dgx_onboard.sh"* ]]
  kill -TERM -- "-$onboard_pgid" 2>/dev/null || true
  for _ in $(seq 1 300); do
    kill -0 -- "-$onboard_pgid" 2>/dev/null || break
    sleep 0.1
  done
  kill -0 -- "-$onboard_pgid" 2>/dev/null && kill -KILL -- "-$onboard_pgid" 2>/dev/null || true
fi
group_remaining=0
kill -0 -- "-$onboard_pgid" 2>/dev/null && group_remaining=1
set +e
if [[ "$run_profile" == migrated_pilot20 || "$run_profile" == migrated_recovery_a5 || "$run_profile" == migrated_recovery_b5 || "$run_profile" == migrated_ablation20_* ]]; then
  # The real model is an intentional co-resident DGX process and remains live
  # until the onboard group has stopped.  Its subsequent stop action performs
  # the authoritative whole-DGX residual probe.
  residual_rc=0
  python3 - "$stage/onboard_residual_probe.json" <<'PY'
import json,time,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "schema_version":1,"status":"DEFERRED_TO_MODEL_STOP",
    "reason":"expected_cohosted_real_model","recorded_unix":time.time(),
},indent=2,sort_keys=True)+"\n",encoding="utf-8")
PY
  : >"$stage/onboard_residual_probe.log"
else
  python3 "$CONTRACT_TOOL" residual-probe --role dgx \
    --deployment-root "$deployment_root" --output "$stage/onboard_residual_probe.json" \
    >"$stage/onboard_residual_probe.log" 2>&1
  residual_rc=$?
fi
relay_rc=125
if test -f "$RESULT_DIR/warn_only_relay.jsonl"; then
  python3 "$deployment_root/t4_completion/map/warn_relay.py" \
    --validate-evidence "$RESULT_DIR/warn_only_relay.jsonl" \
    --summary "$RESULT_DIR/warn_only_relay_summary.json"
  relay_rc=$?
fi
archive_rc=125
if test -d "$RESULT_DIR" && test ! -L "$RESULT_DIR"; then
  cp "$stage/onboard_outer.log" "$RESULT_DIR/onboard_outer.log"
  cp "$stage/onboard_residual_probe.json" "$RESULT_DIR/onboard_residual_probe.json" 2>/dev/null || true
  tar -C "$RESULT_DIR" -czf "$stage/onboard_result.tgz" .
  archive_rc=$?
fi
set -e
python3 - "$stage/onboard_stop_receipt.json" "$was_alive" "$group_remaining" \
  "$residual_rc" "$relay_rc" "$archive_rc" "$grant_id" "$authorization_ref" \
  "$deployment_ref" "$deployment_root" "$RESULT_DIR/onboard_status.json" \
  "$run_profile" <<'PY'
import json,sys,time
from pathlib import Path
alive,remaining,residual,relay,archive=map(int,sys.argv[2:7])
status={}
try: status=json.load(open(sys.argv[11],encoding="utf-8"))
except (OSError,json.JSONDecodeError): pass
passed=(alive==1 and remaining==0 and residual==0 and relay==0 and archive==0
        and status.get("status")=="PASS" and status.get("residual_process_group_count")==0)
payload={"schema_version":1,"status":"PASS" if passed else "FAIL",
"onboard_was_alive":bool(alive),"process_group_remaining":remaining,
"residual_probe_exit_code":residual,"relay_validation_exit_code":relay,
"archive_exit_code":archive,"grant_id":sys.argv[7],
"authorization_ref_sha":sys.argv[8],"deployment_ref_sha":sys.argv[9],
"deployment_root":sys.argv[10],"navigation_owner":"dgx","map_owner":"dgx",
"run_profile":sys.argv[12],
"residual_probe_deferred_to_model_stop":sys.argv[12] in {"migrated_pilot20","migrated_recovery_a5","migrated_recovery_b5"} or sys.argv[12].startswith("migrated_ablation20_"),
"speed_control_owner":"dgx","resource_lease_ack":"dgx+isaac",
"strict_evidence_modified":False,"real_go2_targeted":False,"recorded_unix":time.time()}
Path(sys.argv[1]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
if not passed: raise SystemExit(1)
PY
