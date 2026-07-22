#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

[[ $# -eq 6 || $# -eq 7 ]] || { echo "remote Isaac migration session: invalid arguments" >&2; exit 64; }
action="$1"
deployment_root="$2"
stage="$3"
grant_id="$4"
authorization_ref="$5"
deployment_ref="$6"
run_profile="${7:-migration_smoke}"
ablation_variant=none
case "$action" in prepare|run) ;; *) exit 64 ;; esac
case "$run_profile" in
  migration_smoke)
    episode_count=1
    minimum_sr=0.0
    enable_d435i=0
    result_kind=dgx-migration
    run_label_prefix=dgx_migration
    source_dataset_name=t3_continuous_oracle_go2_clear_v2
    evaluator_role=oracle
    ;;
  migrated_oracle10)
    episode_count=10
    minimum_sr=0.8
    enable_d435i=1
    result_kind=migrated-oracle10
    run_label_prefix=migrated_oracle10
    source_dataset_name=t3_continuous_oracle_go2_clear_v2
    evaluator_role=oracle
    ;;
  migrated_pilot20)
    episode_count=20
    minimum_sr=0.01
    # The typed model client already carries real simulated RGB/RGB-D.  Keep
    # the independent mapping D435 renderer off so the pilot pays for one
    # semantic observation path rather than two camera products.
    enable_d435i=0
    result_kind=migrated-pilot20
    run_label_prefix=migrated_pilot20
    source_dataset_name=t3_model_pilot_go2_clear_v2
    evaluator_role=model
    ;;
  migrated_recovery_a5)
    episode_count=5
    minimum_sr=0.0
    enable_d435i=0
    result_kind=migrated-recovery-a5
    run_label_prefix=migrated_recovery_a5
    source_dataset_name=t3_model_pilot_go2_clear_v2
    evaluator_role=model
    ;;
  migrated_recovery_b5)
    episode_count=5
    minimum_sr=0.0
    enable_d435i=0
    result_kind=migrated-recovery-b5
    run_label_prefix=migrated_recovery_b5
    source_dataset_name=t3_model_pilot_go2_clear_v2
    evaluator_role=model
    ;;
  migrated_ablation20_*)
    ablation_variant="${run_profile#migrated_ablation20_}"
    case "$ablation_variant" in
      full_system1_system2|oracle_high_level_system1|system2_oracle_local_path|full_trajectory|endpoint|straight_line|model_stop|oracle_termination|history_on|history_off|recovery_on|recovery_off|h1_view|go2_view) ;;
      *) exit 64 ;;
    esac
    episode_count=20
    minimum_sr=0.0
    enable_d435i=0
    result_kind="migrated-ablation20-${ablation_variant}"
    run_label_prefix="migrated_ablation20_${ablation_variant}"
    source_dataset_name=t3_model_pilot_go2_clear_v2
    evaluator_role=model
    ;;
  *) exit 64 ;;
esac
[[ "$grant_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{7,95}$ ]] || exit 64
[[ "$authorization_ref" =~ ^[0-9a-f]{40}$ ]] || exit 64
[[ "$deployment_ref" =~ ^[0-9a-f]{40}$ ]] || exit 64

readonly EXPECTED_USER="${INTERNNAV_T4_EXPECTED_USER:-song}"
readonly EXPECTED_IP="${INTERNNAV_T4_EXPECTED_IP:-10.100.120.111}"
readonly STABLE_ROOT="${INTERNNAV_T4_STABLE_ROOT:-/home/song/internnav-t1-t2}"
readonly STAGE_BASE="${INTERNNAV_T4_STAGE_BASE:-/home/song/.codex-internnav-stage}"
readonly DGX_BIND_IP="${INTERNNAV_T4_DGX_BIND_IP:-10.100.100.128}"
readonly CONTROLLER_PORT="${INTERNVLA_T4_CONTROLLER_TCP_PORT:-24137}"
readonly LANE_ROS_DOMAIN_ID="${INTERNNAV_T4_ROS_DOMAIN_ID:-71}"
readonly ISAAC_GPU_INDEX="${INTERNNAV_T4_ISAAC_GPU_INDEX:-0}"
readonly EXPECTED_STAGE="$STAGE_BASE/t4-dgx-migration-${grant_id}"
readonly ROS_WORKSPACE=/home/song/internnav-t4/isaac_ros_ws_45
readonly CONTRACT_TOOL="$deployment_root/coordination/t4_functional_run_contract.py"
readonly SOURCE_DATASET="$deployment_root/episodes/${source_dataset_name}/val_unseen/val_unseen.json.gz"
readonly SMOKE_DATASET="$stage/smoke_dataset"
readonly STATIC_MAPS="$stage/static_maps"
readonly ABLATION_CONFIGS="$stage/ablation_configs"
readonly RESULT_ROOT="$deployment_root/results/internnav_t4/${result_kind}-${grant_id}"
readonly RUN_LABEL="${run_label_prefix}_${grant_id}"
readonly RESULT_DIR="$RESULT_ROOT/${RUN_LABEL}_${evaluator_role}_attempt_001"

test "$(id -un)" = "$EXPECTED_USER"
ip -4 -o addr show | grep -Fq " $EXPECTED_IP/"
[[ "$CONTROLLER_PORT" =~ ^[1-9][0-9]{3,4}$ ]]
[[ "$LANE_ROS_DOMAIN_ID" =~ ^[0-9]{1,3}$ ]]
[[ "$ISAAC_GPU_INDEX" =~ ^[01]$ ]]
test "$stage" = "$EXPECTED_STAGE"
test -d "$stage" && test ! -L "$stage"
test "$(realpath "$stage")" = "$EXPECTED_STAGE"
test -d "$deployment_root" && test ! -L "$deployment_root"
case "$deployment_root" in "$STABLE_ROOT/.t4-deployments/"*) ;; *) exit 64 ;; esac
test "$(realpath "$deployment_root")" = "$deployment_root"
test -x "$CONTRACT_TOOL"

if test "$action" = prepare; then
  test -f "$SOURCE_DATASET"
  test ! -e "$SMOKE_DATASET"
  test ! -e "$STATIC_MAPS"
  python3 "$CONTRACT_TOOL" validate-deployment \
    --role isaac --deployment-root "$deployment_root" --deployment-ref "$deployment_ref" \
    >"$stage/deployment_validation.json"
  python3 "$deployment_root/scripts/build_t4_migration_smoke_dataset.py" \
    --source "$SOURCE_DATASET" --output-root "$SMOKE_DATASET" \
    --episode-count "$episode_count" \
    >"$stage/smoke_dataset_build.log"
  INTERNNAV_T1_CONTROL_ROOT="$deployment_root" \
  INTERNNAV_RUNTIME_POLICY=completion_sim \
  INTERNNAV_SIMULATION_TARGET=isaac \
  INTERNNAV_T4_RESOURCE_LEASE_ACK=dgx+isaac \
  INTERNVLA_T3_STATIC_CLEARANCE_GATE_M=0.30 \
    bash "$deployment_root/scripts/prepare_t4_isaac_static_maps.sh" \
      --dataset-root "$SMOKE_DATASET" --output-dir "$STATIC_MAPS" \
      >"$stage/static_map_build.log" 2>&1
  archive_members=(static_maps smoke_dataset)
  if test "$ablation_variant" != none; then
    test ! -e "$ABLATION_CONFIGS"
    python3 "$deployment_root/scripts/t4_ablation_generate.py" \
      --output-dir "$ABLATION_CONFIGS" >"$stage/ablation_config_build.log"
    python3 "$deployment_root/scripts/t4_ablation_validate.py" configs \
      --config-dir "$ABLATION_CONFIGS" >"$stage/ablation_config_validation.json"
    python3 "$deployment_root/scripts/t4_ablation_runtime.py" validate \
      --config "$ABLATION_CONFIGS/variants/${ablation_variant}.json" \
      --matrix "$deployment_root/configs/completion_sim/ablation/frozen_matrix_v1.json" \
      >"$stage/ablation_variant_validation.json"
    archive_members+=(ablation_configs)
  fi
  tar -C "$stage" -czf "$stage/migration_inputs.tgz" "${archive_members[@]}"
  python3 - "$stage/migration_prepare_receipt.json" "$grant_id" \
    "$authorization_ref" "$deployment_ref" "$deployment_root" \
    "$episode_count" "$run_profile" "$ablation_variant" <<'PY'
import hashlib,json,sys,time
from pathlib import Path
archive=Path(sys.argv[1]).with_name("migration_inputs.tgz")
payload={"schema_version":1,"status":"PASS","grant_id":sys.argv[2],
"authorization_ref_sha":sys.argv[3],"deployment_ref_sha":sys.argv[4],
"deployment_root":sys.argv[5],"episode_count":int(sys.argv[6]),
"run_profile":sys.argv[7],
"ablation_variant_id":None if sys.argv[8]=="none" else sys.argv[8],
"ablation_step_horizon":1200 if sys.argv[8]!="none" else None,
"migration_inputs_sha256":hashlib.sha256(archive.read_bytes()).hexdigest(),
"resource_lease_ack":"dgx+isaac","strict_evidence_modified":False,
"real_go2_targeted":False,"recorded_unix":time.time()}
Path(sys.argv[1]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
PY
  exit 0
fi

test -f "$SMOKE_DATASET/val_unseen/val_unseen.json.gz"
test -f "$STATIC_MAPS/manifest.json"
if test "$ablation_variant" != none; then
  test -f "$ABLATION_CONFIGS/variants/${ablation_variant}.json"
  mapfile -t ablation_fields < <(
    python3 "$deployment_root/scripts/t4_ablation_runtime.py" fields \
      --config "$ABLATION_CONFIGS/variants/${ablation_variant}.json" \
      --matrix "$deployment_root/configs/completion_sim/ablation/frozen_matrix_v1.json"
  )
  test "${#ablation_fields[@]}" -eq 11
  test "${ablation_fields[0]}" = "$ablation_variant"
  export INTERNVLA_T4_VARIANT_ID="${ablation_fields[0]}"
  export INTERNVLA_T4_VARIANT_CONFIG_SHA256="${ablation_fields[1]}"
  export INTERNVLA_T4_MATRIX_SHA256="${ablation_fields[2]}"
  export INTERNVLA_T4_SYSTEM_MODE="${ablation_fields[3]}"
  export INTERNVLA_T4_TRAJECTORY_MODE="${ablation_fields[4]}"
  export INTERNVLA_T4_TERMINATION_MODE="${ablation_fields[5]}"
  export INTERNVLA_T4_HISTORY_MODE="${ablation_fields[6]}"
  export INTERNVLA_T4_RECOVERY_MODE="${ablation_fields[7]}"
  export INTERNVLA_T4_VIEW_MODE="${ablation_fields[8]}"
  export INTERNVLA_T4_REQUIRED_HISTORY_MODE="${ablation_fields[6]}"
  export INTERNVLA_T4_MAX_STEP=1200
  if test "${ablation_fields[8]}" = h1_view; then
    # The view ablation changes only semantic RGB.  Depth, LiDAR, map and
    # collision geometry remain on the frozen Go2 configuration.
    export INTERNVLA_T4_CAMERA_HEIGHT_OVERRIDE_M=1.25
    export INTERNVLA_T4_CAMERA_PITCH_OVERRIDE_DEG=30.0
    export INTERNVLA_T4_CAMERA_HFOV_OVERRIDE_DEG=90.0
    export INTERNVLA_T4_CAMERA_VFOV_OVERRIDE_DEG=67.5
  fi
fi
test ! -e "$RESULT_DIR"
python3 "$CONTRACT_TOOL" validate-overlay-source \
  --deployment-root "$deployment_root" --workspace "$ROS_WORKSPACE" \
  >"$stage/overlay_source_validation.json"

raw_run_rc=125
cleanup_primary_rc=125
cleanup_probe_rc=125
residual_rc=125
archive_rc=125
finalize() {
  local incoming=$? final_rc=1
  trap - EXIT INT TERM HUP
  set +e
  INTERNNAV_T1_CONTROL_ROOT="$deployment_root" \
    bash "$deployment_root/scripts/t4_cleanup_container_processes.sh" \
      >"$stage/container_cleanup_primary.log" 2>&1
  cleanup_primary_rc=$?
  INTERNNAV_T1_CONTROL_ROOT="$deployment_root" \
    bash "$deployment_root/scripts/t4_cleanup_container_processes.sh" \
      >"$stage/container_cleanup_probe.log" 2>&1
  cleanup_probe_rc=$?
  python3 "$CONTRACT_TOOL" residual-probe \
    --deployment-root "$deployment_root" \
    --socket-root "$deployment_root/runtime/t4_ipc" \
    --output "$stage/isaac_residual_probe.json" \
    >"$stage/isaac_residual_probe.log" 2>&1
  residual_rc=$?
  if test -d "$RESULT_DIR" && test ! -L "$RESULT_DIR"; then
    cp "$stage/isaac_residual_probe.json" "$RESULT_DIR/isaac_residual_probe.json" 2>/dev/null || true
    tar -C "$RESULT_DIR" -czf "$stage/isaac_result.tgz" .
    archive_rc=$?
  fi
  if { test "$raw_run_rc" = 0 || test "$raw_run_rc" = 143; } \
      && test "$cleanup_primary_rc" = 0 && test "$cleanup_probe_rc" = 0 \
      && test "$residual_rc" = 0 && test "$archive_rc" = 0 \
      && test -f "$RESULT_DIR/isaac_remote_validation.json" \
      && grep -Fq '"status": "PASS"' "$RESULT_DIR/isaac_remote_validation.json"; then
    final_rc=0
  elif test "$incoming" -ne 0; then final_rc=$incoming
  elif test "$raw_run_rc" -ne 0; then final_rc=$raw_run_rc
  fi
  python3 - "$stage/isaac_run_receipt.json" "$final_rc" "$raw_run_rc" \
    "$cleanup_primary_rc" "$cleanup_probe_rc" "$residual_rc" "$archive_rc" \
    "$grant_id" "$authorization_ref" "$deployment_ref" "$deployment_root" \
    "$run_profile" "$episode_count" <<'PY'
import json,sys,time
from pathlib import Path
codes=list(map(int,sys.argv[2:8])); passed=codes[0]==0
payload={"schema_version":1,"status":"PASS" if passed else "FAIL",
"exit_codes":{"final":codes[0],"run":codes[1],"cleanup_primary":codes[2],
"cleanup_probe":codes[3],"residual":codes[4],"archive":codes[5]},
"grant_id":sys.argv[8],"authorization_ref_sha":sys.argv[9],
"deployment_ref_sha":sys.argv[10],"deployment_root":sys.argv[11],
"run_profile":sys.argv[12],"expected_episode_count":int(sys.argv[13]),
"ablation_step_horizon":1200 if sys.argv[12].startswith("migrated_ablation20_") else None,
"host_role":"isaac_sim_sensor_client","local_navigation_processes_started":False,
"local_speed_control_processes_started":False,"resource_lease_ack":"dgx+isaac",
"strict_evidence_modified":False,"real_go2_targeted":False,"recorded_unix":time.time()}
Path(sys.argv[1]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
PY
  exit "$final_rc"
}
trap finalize EXIT
trap 'exit 130' INT TERM HUP

export INTERNNAV_T1_CONTROL_ROOT="$deployment_root"
export INTERNNAV_T0_CONTROL_ROOT="$deployment_root"
export INTERNVLA_ROS_WS="$ROS_WORKSPACE"
export INTERNNAV_RUNTIME_POLICY=completion_sim
export INTERNNAV_SIMULATION_TARGET=isaac
export INTERNNAV_T4_RESOURCE_LEASE_ACK=dgx+isaac
export ROS_DOMAIN_ID="$LANE_ROS_DOMAIN_ID"
export CUDA_VISIBLE_DEVICES="$ISAAC_GPU_INDEX"
export INTERNVLA_T4_COMPLETION_FAST_PATH=1
export INTERNVLA_T4_EXPECTED_COUNT="$episode_count"
export INTERNVLA_T4_MIN_SR_OVERRIDE="$minimum_sr"
export INTERNVLA_T4_RESULT_ROOT="$RESULT_ROOT"
export INTERNVLA_T4_RUN_LABEL="$RUN_LABEL"
export INTERNVLA_T4_PHASE_OVERLAY_BUILDER="$deployment_root/scripts/build_t4_isaac_remote_phase_overlay.py"
export INTERNVLA_T4_RUNTIME_OVERLAY_BUILDER="$deployment_root/scripts/build_t4_r3_sensor_runtime_overlay.py"
export INTERNVLA_T4_GO2_USD_BUILDER="$deployment_root/scripts/build_t4_r3_go2_usd.py"
export INTERNVLA_T4_R3_ENABLE_SENSOR_BRIDGE=1
export INTERNVLA_T4_R3_ENABLE_D435I="$enable_d435i"
export INTERNVLA_T4_R3_ENABLE_LIDAR=1
export INTERNVLA_T4_R3_ENABLE_RGB_IPC=0
export INTERNVLA_T4_DEPTH_STRIDE_OVERRIDE=4
export INTERNVLA_T4_DGX_BIND_IP="$DGX_BIND_IP"
export INTERNVLA_T4_CONTROLLER_TCP_PORT="$CONTROLLER_PORT"
export INTERNVLA_GO2_CONTROLLER_ENDPOINT="tcp://$DGX_BIND_IP:$CONTROLLER_PORT"

if test "$evaluator_role" = model; then
  export INTERNVLA_T4_PHASE_OVERRIDE=pilot
  export INTERNVLA_T4_MODEL_DATASET_ROOT="$SMOKE_DATASET"
else
  export INTERNVLA_T4_PHASE_OVERRIDE=continuous_oracle
  export INTERNVLA_T4_ORACLE_DATASET_ROOT="$SMOKE_DATASET"
  export INTERNVLA_ORACLE_DATASET_ROOT="$SMOKE_DATASET"
fi

set +e
bash "$deployment_root/scripts/run_t4_sensor_gate.sh" t4_4 "$evaluator_role" 001 \
  >"$stage/isaac_outer.log" 2>&1
raw_run_rc=$?
set -e
exit "$raw_run_rc"
