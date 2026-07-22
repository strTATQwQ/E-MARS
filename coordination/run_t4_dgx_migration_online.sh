#!/usr/bin/env bash
set -euo pipefail

readonly COORDINATION_REF=refs/heads/codex/parallel-integration

lane_id="${INTERNNAV_T4_LANE_ID:-a}"
case "$lane_id" in
  a)
    readonly DGX_REMOTE_USER=railgun DGX_REMOTE_HOST=10.100.100.128
    readonly DGX_EXPECTED_IP=10.100.100.128 ISAAC_GPU_INDEX=0
    readonly LANE_ROS_DOMAIN_ID=71 LANE_TCP_PORT=24137
    ;;
  b)
    readonly DGX_REMOTE_USER=rail DGX_REMOTE_HOST=10.100.120.116
    readonly DGX_EXPECTED_IP=10.100.120.116 ISAAC_GPU_INDEX=1
    readonly LANE_ROS_DOMAIN_ID=72 LANE_TCP_PORT=24138
    ;;
  *) echo "T4 DGX migration coordinator: invalid lane" >&2; exit 64 ;;
esac
readonly LANE_ID="$lane_id"
readonly ISAAC_REMOTE_USER=song ISAAC_REMOTE_HOST=10.100.120.111
readonly DGX_STABLE_ROOT=/home/railgun/internnav-t1-t2
readonly DGX_STAGE_BASE=/home/railgun/.codex-internnav-stage
readonly ISAAC_STABLE_ROOT=/home/song/internnav-t1-t2
readonly ISAAC_STAGE_BASE=/home/song/.codex-internnav-stage

usage() {
  echo "usage: run_t4_dgx_migration_online.sh GRANT AUTH_SHA PREPARE_RESULT_REL [migration_smoke|migrated_oracle10|migrated_pilot20|migrated_recovery_a5|migrated_recovery_b5|migrated_ablation20_VARIANT] [PREREQUISITE_RESULT_REL]" >&2
  exit 64
}
die() { echo "T4 DGX migration coordinator: $*" >&2; exit 64; }

resolve_git_dir() {
  local root="$1" raw
  if [[ -d "$root/.git" ]]; then readlink -f "$root/.git"; return; fi
  [[ -f "$root/.git" ]] || die "worktree has no git pointer"
  raw="$(sed -n 's/^gitdir: //p' "$root/.git")"
  case "$raw" in
    [A-Za-z]:/*|[A-Za-z]:\\*) wslpath -u "$raw" ;;
    /*) printf '%s\n' "$raw" ;;
    *) readlink -f "$root/$raw" ;;
  esac
}

configure_role() {
  case "$1" in
    dgx) REMOTE_USER="$DGX_REMOTE_USER"; REMOTE_HOST="$DGX_REMOTE_HOST" ;;
    isaac) REMOTE_USER="$ISAAC_REMOTE_USER"; REMOTE_HOST="$ISAAC_REMOTE_HOST" ;;
    *) die "unknown role" ;;
  esac
  REMOTE_TARGET="$REMOTE_USER@$REMOTE_HOST"
  SSH_OPTIONS=(-o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5
    -o ServerAliveCountMax=2 -o StrictHostKeyChecking=accept-new)
}
remote_exec() {
  local role="$1" command="$2" prefix
  configure_role "$role"
  if [[ "$role" == dgx ]]; then
    printf -v prefix 'export INTERNNAV_T4_EXPECTED_USER=%q INTERNNAV_T4_EXPECTED_IP=%q INTERNNAV_T4_STABLE_ROOT=%q INTERNNAV_T4_STAGE_BASE=%q INTERNNAV_T4_DGX_BIND_IP=%q INTERNNAV_T4_ISAAC_PEER_IP=%q INTERNVLA_T4_CONTROLLER_TCP_PORT=%q INTERNNAV_T4_ROS_DOMAIN_ID=%q ROS_DOMAIN_ID=%q INTERNNAV_T4_LANE_ID=%q; ' \
      "$DGX_REMOTE_USER" "$DGX_EXPECTED_IP" "$DGX_STABLE_ROOT" "$DGX_STAGE_BASE" \
      "$DGX_EXPECTED_IP" "$ISAAC_REMOTE_HOST" "$LANE_TCP_PORT" \
      "$LANE_ROS_DOMAIN_ID" "$LANE_ROS_DOMAIN_ID" "$LANE_ID"
  else
    printf -v prefix 'export INTERNNAV_T4_EXPECTED_USER=%q INTERNNAV_T4_EXPECTED_IP=%q INTERNNAV_T4_STABLE_ROOT=%q INTERNNAV_T4_STAGE_BASE=%q INTERNNAV_T4_DGX_BIND_IP=%q INTERNVLA_T4_CONTROLLER_TCP_PORT=%q INTERNNAV_T4_ROS_DOMAIN_ID=%q ROS_DOMAIN_ID=%q CUDA_VISIBLE_DEVICES=%q INTERNNAV_T4_ISAAC_GPU_INDEX=%q INTERNNAV_T4_LANE_ID=%q; ' \
      "$ISAAC_REMOTE_USER" "$ISAAC_REMOTE_HOST" "$ISAAC_STABLE_ROOT" "$ISAAC_STAGE_BASE" \
      "$DGX_EXPECTED_IP" "$LANE_TCP_PORT" "$LANE_ROS_DOMAIN_ID" \
      "$LANE_ROS_DOMAIN_ID" "$ISAAC_GPU_INDEX" "$ISAAC_GPU_INDEX" "$LANE_ID"
  fi
  ssh -T "${SSH_OPTIONS[@]}" "$REMOTE_TARGET" "$prefix$command"
}
remote_get() { local role="$1" source="$2" target="$3"; configure_role "$role"; scp "${SSH_OPTIONS[@]}" "$REMOTE_TARGET:$source" "$target"; }
remote_put() { local role="$1" source="$2" target="$3"; configure_role "$role"; scp "${SSH_OPTIONS[@]}" "$source" "$REMOTE_TARGET:$target"; }
create_stage() {
  local role="$1" stage="$2" expected
  expected=" $DGX_EXPECTED_IP/"; test "$role" != isaac || expected=" $ISAAC_REMOTE_HOST/"
  printf -v command_text 'ip -4 -o addr show | grep -Fq %q && test ! -e %q && install -d -m 700 %q' "$expected" "$stage" "$stage"
  remote_exec "$role" "$command_text"
}
remove_stage() {
  local role="$1" stage="$2"
  printf -v command_text 'test -d %q && test ! -L %q && test "$(realpath %q)" = %q && rm -rf -- %q' "$stage" "$stage" "$stage" "$stage" "$stage"
  remote_exec "$role" "$command_text"
}

run_under_lease() {
  [[ $# -eq 12 ]] || die "invalid internal invocation"
  local root="$1" grant_id="$2" authorization_ref="$3" deployment_ref="$4"
  local dgx_root="$5" isaac_root="$6" result_rel="$7" lease_rel="$8"
  local prepare_rel="$9" contract_tool="${10}"
  local run_profile="${11}"
  local prerequisite_rel="${12}"
  local git_dir
  git_dir="$(resolve_git_dir "$root")"
  local model_profile=0 recovery_profile=0
  case "$run_profile" in
    migrated_pilot20) model_profile=1 ;;
    migrated_recovery_a5|migrated_recovery_b5) model_profile=1; recovery_profile=1 ;;
    migrated_ablation20_*) model_profile=1 ;;
  esac
  local result_dir="$root/$result_rel"
  local dgx_stage="$DGX_STAGE_BASE/t4-dgx-onboard-${grant_id}"
  local model_stage="$DGX_STAGE_BASE/t4-pilot-model-${grant_id}"
  local isaac_stage="$ISAAC_STAGE_BASE/t4-dgx-migration-${grant_id}"
  local onboard_started=0 onboard_stopped=0 prepare_rc=125 start_rc=125 isaac_rc=125 stop_rc=125
  local model_started=0 model_stopped=0 model_start_rc=0 model_stop_rc=0
  local isaac_stage_created=0 dgx_stage_created=0 model_stage_created=0
  local collect_rc=0 dgx_cleanup_rc=125 isaac_cleanup_rc=125 validation_rc=125
  mkdir -p "$result_dir/remote-artifacts/dgx" "$result_dir/remote-artifacts/isaac" \
    "$result_dir/remote-artifacts/dgx/model" "$result_dir/dgx-run" \
    "$result_dir/isaac-run" "$result_dir/model-run"

  cleanup_services() {
    local incoming=$?
    trap - EXIT INT TERM HUP
    set +e
    if (( onboard_started == 1 && onboard_stopped == 0 )); then
      printf -v command_text 'bash %q stop %q %q %q %q %q %q' \
        "$dgx_root/coordination/remote_t4_dgx_onboard_session.sh" \
        "$dgx_root" "$dgx_stage" "$grant_id" "$authorization_ref" "$deployment_ref" \
        "$run_profile"
      remote_exec dgx "$command_text" >/dev/null 2>&1 || true
    fi
    if (( model_started == 1 && model_stopped == 0 )); then
      printf -v command_text 'bash %q stop %q %q %q %q %q %q' \
        "$dgx_root/coordination/remote_t4_model_session.sh" \
        "$dgx_root" "$model_stage" "$grant_id" "$authorization_ref" "$deployment_ref" \
        "$run_profile"
      remote_exec dgx "$command_text" >/dev/null 2>&1 || true
    fi
    if (( model_stage_created == 1 )); then
      for file in model_outer.log deployment_validation.json payload_reverification.json \
          dgx_workspace_validation.json model_health_probe.log model_process.json; do
        remote_get dgx "$model_stage/$file" \
          "$result_dir/remote-artifacts/dgx/model/$file" >/dev/null 2>&1 || true
      done
    fi
    if (( dgx_stage_created == 1 )); then
      for file in onboard_outer.log deployment_validation.json \
          dgx_workspace_validation.json onboard_process.json \
          onboard_ready_receipt.json onboard_stop_receipt.json \
          onboard_residual_probe.json onboard_residual_probe.log; do
        remote_get dgx "$dgx_stage/$file" \
          "$result_dir/remote-artifacts/dgx/$file" >/dev/null 2>&1 || true
      done
      remote_get dgx "$dgx_stage/onboard_result.tgz" \
        "$result_dir/remote-artifacts/dgx/onboard_result.tgz" >/dev/null 2>&1 || true
    fi
    if (( isaac_stage_created == 1 )); then
      for file in deployment_validation.json smoke_dataset_build.log static_map_build.log \
          migration_prepare_receipt.json isaac_outer.log; do
        remote_get isaac "$isaac_stage/$file" \
          "$result_dir/remote-artifacts/isaac/$file" >/dev/null 2>&1 || true
      done
    fi
    (( isaac_stage_created == 0 )) || remove_stage isaac "$isaac_stage" >/dev/null 2>&1 || true
    (( model_stage_created == 0 )) || remove_stage dgx "$model_stage" >/dev/null 2>&1 || true
    (( dgx_stage_created == 0 )) || remove_stage dgx "$dgx_stage" >/dev/null 2>&1 || true
    exit "$incoming"
  }
  trap cleanup_services EXIT
  trap 'exit 130' INT TERM HUP

  create_stage isaac "$isaac_stage"
  isaac_stage_created=1
  printf -v command_text 'bash %q prepare %q %q %q %q %q %q' \
    "$isaac_root/coordination/remote_t4_isaac_migration_session.sh" \
    "$isaac_root" "$isaac_stage" "$grant_id" "$authorization_ref" "$deployment_ref" \
    "$run_profile"
  set +e; remote_exec isaac "$command_text"; prepare_rc=$?; set -e
  test "$prepare_rc" = 0
  remote_get isaac "$isaac_stage/migration_inputs.tgz" "$result_dir/migration_inputs.tgz"
  remote_get isaac "$isaac_stage/migration_prepare_receipt.json" \
    "$result_dir/remote-artifacts/isaac/migration_prepare_receipt.json"

  create_stage dgx "$dgx_stage"
  dgx_stage_created=1
  remote_put dgx "$result_dir/migration_inputs.tgz" "$dgx_stage/migration_inputs.tgz"
  printf -v command_text 'tar -C %q -xzf %q && test -f %q && test -f %q' \
    "$dgx_stage" "$dgx_stage/migration_inputs.tgz" "$dgx_stage/static_maps/manifest.json" \
    "$dgx_stage/smoke_dataset/val_unseen/val_unseen.json.gz"
  remote_exec dgx "$command_text"

  if (( model_profile == 1 )); then
    create_stage dgx "$model_stage"
    model_stage_created=1
    if [[ "$run_profile" == migrated_ablation20_* ]]; then
      ablation_variant="${run_profile#migrated_ablation20_}"
      printf -v command_text 'test -f %q && cp %q %q && test -f %q' \
        "$dgx_stage/ablation_configs/variants/${ablation_variant}.json" \
        "$dgx_stage/ablation_configs/variants/${ablation_variant}.json" \
        "$model_stage/variant_config.json" "$model_stage/variant_config.json"
      remote_exec dgx "$command_text"
    fi
    printf -v command_text 'bash %q start %q %q %q %q %q %q' \
      "$dgx_root/coordination/remote_t4_model_session.sh" \
      "$dgx_root" "$model_stage" "$grant_id" "$authorization_ref" "$deployment_ref" \
      "$run_profile"
    set +e; remote_exec dgx "$command_text"; model_start_rc=$?; set -e
    test "$model_start_rc" = 0
    model_started=1
    remote_get dgx "$model_stage/model_ready_receipt.json" \
      "$result_dir/remote-artifacts/dgx/model_ready_receipt.json"
  fi

  onboard_started=1
  printf -v command_text 'bash %q start %q %q %q %q %q %q' \
    "$dgx_root/coordination/remote_t4_dgx_onboard_session.sh" \
    "$dgx_root" "$dgx_stage" "$grant_id" "$authorization_ref" "$deployment_ref" \
    "$run_profile"
  set +e; remote_exec dgx "$command_text"; start_rc=$?; set -e
  test "$start_rc" = 0
  remote_get dgx "$dgx_stage/onboard_ready_receipt.json" \
    "$result_dir/remote-artifacts/dgx/onboard_ready_receipt.json"

  printf -v command_text 'bash %q run %q %q %q %q %q %q' \
    "$isaac_root/coordination/remote_t4_isaac_migration_session.sh" \
    "$isaac_root" "$isaac_stage" "$grant_id" "$authorization_ref" "$deployment_ref" \
    "$run_profile"
  set +e; remote_exec isaac "$command_text"; isaac_rc=$?; set -e

  printf -v command_text 'bash %q stop %q %q %q %q %q %q' \
    "$dgx_root/coordination/remote_t4_dgx_onboard_session.sh" \
    "$dgx_root" "$dgx_stage" "$grant_id" "$authorization_ref" "$deployment_ref" \
    "$run_profile"
  set +e; remote_exec dgx "$command_text"; stop_rc=$?; set -e
  onboard_stopped=1

  if (( model_profile == 1 )); then
    printf -v command_text 'bash %q stop %q %q %q %q %q %q' \
      "$dgx_root/coordination/remote_t4_model_session.sh" \
      "$dgx_root" "$model_stage" "$grant_id" "$authorization_ref" "$deployment_ref" \
      "$run_profile"
    set +e; remote_exec dgx "$command_text"; model_stop_rc=$?; set -e
    model_stopped=1
  fi

  set +e
  for file in onboard_ready_receipt.json onboard_stop_receipt.json \
      onboard_process.json onboard_residual_probe.json onboard_residual_probe.log \
      onboard_outer.log deployment_validation.json dgx_workspace_validation.json; do
    remote_get dgx "$dgx_stage/$file" "$result_dir/remote-artifacts/dgx/$file" >/dev/null 2>&1 || true
  done
  remote_get dgx "$dgx_stage/onboard_result.tgz" \
    "$result_dir/remote-artifacts/dgx/onboard_result.tgz" || collect_rc=$?
  if (( model_profile == 1 )); then
    remote_get dgx "$model_stage/model_stop_receipt.json" \
      "$result_dir/remote-artifacts/dgx/model_stop_receipt.json" || collect_rc=$?
    for file in model_ready_receipt.json model_stop_receipt.json model_invocation.json \
        model_process.json model_health_ready.json deployment_validation.json \
        payload_reverification.json dgx_workspace_validation.json model_outer.log \
        model_health_probe.log model_residual_probe.json model_residual_probe.log; do
      remote_get dgx "$model_stage/$file" \
        "$result_dir/remote-artifacts/dgx/model/$file" >/dev/null 2>&1 || true
    done
    remote_get dgx "$model_stage/model_result.tgz" \
      "$result_dir/remote-artifacts/dgx/model_result.tgz" || collect_rc=$?
  fi
  for file in migration_prepare_receipt.json isaac_run_receipt.json \
      isaac_outer.log isaac_residual_probe.json isaac_residual_probe.log \
      container_cleanup_primary.log container_cleanup_probe.log \
      deployment_validation.json overlay_source_validation.json; do
    remote_get isaac "$isaac_stage/$file" "$result_dir/remote-artifacts/isaac/$file" >/dev/null 2>&1 || true
  done
  remote_get isaac "$isaac_stage/isaac_result.tgz" \
    "$result_dir/remote-artifacts/isaac/isaac_result.tgz" || collect_rc=$?
  set -e

  if test -f "$result_dir/remote-artifacts/dgx/onboard_result.tgz"; then
    python3 "$contract_tool" extract-oracle-archive \
      --archive "$result_dir/remote-artifacts/dgx/onboard_result.tgz" \
      --output-dir "$result_dir/dgx-run" >"$result_dir/dgx_extract.log" 2>&1 || collect_rc=$?
  fi
  if test -f "$result_dir/remote-artifacts/isaac/isaac_result.tgz"; then
    python3 "$contract_tool" extract-oracle-archive \
      --archive "$result_dir/remote-artifacts/isaac/isaac_result.tgz" \
      --output-dir "$result_dir/isaac-run" >"$result_dir/isaac_extract.log" 2>&1 || collect_rc=$?
  fi
  if (( model_profile == 1 )) \
      && test -f "$result_dir/remote-artifacts/dgx/model_result.tgz"; then
    python3 "$contract_tool" extract-oracle-archive \
      --archive "$result_dir/remote-artifacts/dgx/model_result.tgz" \
      --output-dir "$result_dir/model-run" >"$result_dir/model_extract.log" 2>&1 || collect_rc=$?
  fi
  if (( collect_rc == 0 )); then
    set +e
    remove_stage isaac "$isaac_stage"; isaac_cleanup_rc=$?
    remove_stage dgx "$dgx_stage"; dgx_cleanup_rc=$?
    if (( model_profile == 1 )); then
      remove_stage dgx "$model_stage" || dgx_cleanup_rc=$?
    fi
    set -e
  fi

  if (( collect_rc == 0 && recovery_profile == 1 )); then
    set +e
    python3 "$root/scripts/analyze_t4_recovery.py" "$result_dir/dgx-run" \
      --per-episode "$result_dir/isaac-run/per_episode.json" \
      --records "$result_dir/dgx-run/recovery_records.jsonl" \
      --replan-records "$result_dir/isaac-run/replan_request_records.jsonl" \
      --adapter-records "$result_dir/dgx-run/active_records.jsonl" \
      --output "$result_dir/recovery_metrics.json" \
      >"$result_dir/recovery_metrics.log" 2>&1
    set -e
  fi

  set +e
  python3 - "$result_dir" "$grant_id" "$authorization_ref" "$deployment_ref" \
    "$prepare_rel" "$prepare_rc" "$start_rc" "$isaac_rc" "$stop_rc" \
    "$collect_rc" "$dgx_cleanup_rc" "$isaac_cleanup_rc" "$run_profile" \
    "$model_start_rc" "$model_stop_rc" "$prerequisite_rel" <<'PY'
import json,sys,time
from pathlib import Path
root=Path(sys.argv[1])
missing=[]
load_errors={}
def load(relative):
    path=root/relative
    if not path.is_file():
        missing.append(relative)
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        load_errors[relative]=f"{type(exc).__name__}: {exc}"[:512]
        return {}
controller=load("dgx-run/controller_summary.json")
go2_bridge=load("dgx-run/go2_sensor_bridge/go2_sensor_bridge_summary.json")
relay=load("dgx-run/warn_only_relay_summary.json")
onboard=load("dgx-run/onboard_status.json")
isaac=load("isaac-run/isaac_remote_validation.json")
ready=load("remote-artifacts/dgx/onboard_ready_receipt.json")
stopped=load("remote-artifacts/dgx/onboard_stop_receipt.json")
prepared=load("remote-artifacts/isaac/migration_prepare_receipt.json")
isaac_receipt=load("remote-artifacts/isaac/isaac_run_receipt.json")
profile=sys.argv[13]
ablation_prefix="migrated_ablation20_"
ablation=profile.startswith(ablation_prefix)
variant=profile[len(ablation_prefix):] if ablation else None
variant_ids={"full_system1_system2","oracle_high_level_system1","system2_oracle_local_path","full_trajectory","endpoint","straight_line","model_stop","oracle_termination","history_on","history_off","recovery_on","recovery_off","h1_view","go2_view"}
if profile not in {"migration_smoke","migrated_oracle10","migrated_pilot20","migrated_recovery_a5","migrated_recovery_b5"} and (not ablation or variant not in variant_ids):
    raise SystemExit("unknown DGX-onboard run profile")
expected_count=20 if ablation else {"migration_smoke":1,"migrated_oracle10":10,"migrated_pilot20":20,"migrated_recovery_a5":5,"migrated_recovery_b5":5}[profile]
minimum_sr=0.0 if ablation else {"migration_smoke":0.0,"migrated_oracle10":0.8,"migrated_pilot20":0.01,"migrated_recovery_a5":0.0,"migrated_recovery_b5":0.0}[profile]
model_run=ablation or profile in {"migrated_pilot20","migrated_recovery_a5","migrated_recovery_b5"}
recovery=profile in {"migrated_recovery_a5","migrated_recovery_b5"}
expected_recovery_id={"migrated_recovery_a5":"A","migrated_recovery_b5":"B"}.get(profile)
model_ready=load("remote-artifacts/dgx/model_ready_receipt.json") if model_run else {}
model_stopped=load("remote-artifacts/dgx/model_stop_receipt.json") if model_run else {}
model_weight=load("model-run/model_weight_audit.json") if model_run else {}
client=load("isaac-run/client_summary.json") if model_run else {}
recovery_metrics=load("recovery_metrics.json") if recovery else {}
recovery_runtime=load("dgx-run/recovery_runtime_manifest.json") if recovery else {}
ablation_contract=load("isaac-run/t4_run_contract.json") if ablation else {}
variant_claim=load("model-run/model_variant_claim.json") if ablation else {}
model_lifecycle=load("model-run/model_lifecycle_manifest.json") if ablation else {}
sensor_records=[]
sensor_path=root/"dgx-run/sensor_frames.jsonl"
if sensor_path.is_file():
    try:
        for line in sensor_path.read_text(encoding="utf-8").splitlines():
            if line.strip(): sensor_records.append(json.loads(line))
    except Exception as exc:
        load_errors["dgx-run/sensor_frames.jsonl"]=f"{type(exc).__name__}: {exc}"[:512]
checks={
"ready":ready.get("status")=="PASS","stopped":stopped.get("status")=="PASS",
"isaac_receipt":isaac_receipt.get("status")=="PASS","onboard":onboard.get("status")=="PASS",
"isaac":isaac.get("status")=="PASS","tcp":controller.get("ipc_transport")=="tcp",
"accepted":int(controller.get("ipc_accepted_connection_count",0))>=1,
"lidar":int(controller.get("r3_lidar_frame_count",0))>=1 and int((go2_bridge.get("counts") or {}).get("completion_lidar_groups",0))>=1 and int((go2_bridge.get("counts") or {}).get("lidar_output_points",0))>=1,
"bounded_motion":ablation or int(relay.get("nonzero_output_count",0))>=1,
"bounded_limits":float(relay.get("maximum_abs_linear_mps",999.0))<=0.25+1e-9 and float(relay.get("maximum_abs_angular_rps",999.0))<=1.0+1e-9,
"fatal_safety_zero":all(int(controller.get(name,-1))==0 for name in ("nan_count","fall_count","stale_motion_execution_count")),
"no_direct_bypass":int(controller.get("direct_motion_bypass_count",-1))==0,
"map_and_pose":controller.get("map_source")=="static_map" and controller.get("pose_source")=="ground_truth" and controller.get("ground_truth_pose_used_for_nav") is True,
"owners":onboard.get("navigation_owner")==onboard.get("map_owner")==onboard.get("speed_control_owner")=="dgx",
"isaac_isolated":isaac.get("local_navigation_processes_started") is False and isaac.get("local_speed_control_processes_started") is False and isaac_receipt.get("local_navigation_processes_started") is False and isaac_receipt.get("local_speed_control_processes_started") is False,
"cleanup":int(sys.argv[11])==0 and int(sys.argv[12])==0,
"profile":ready.get("run_profile")==stopped.get("run_profile")==prepared.get("run_profile")==isaac_receipt.get("run_profile")==profile,
"prepared_episodes":int(prepared.get("episode_count",-1))==expected_count,
"episodes":int(isaac.get("episode_count",-1))==expected_count and int(isaac.get("expected_episode_count",-1))==expected_count,
"success_rate":float(isaac.get("success_rate",-1.0)) >= minimum_sr,
"metric_depth":profile != "migrated_oracle10" or (int(controller.get("metric_depth_frame_count",0)) > 0 and int(controller.get("metric_depth_valid_count",0)) > 0),
"depth_stride":profile != "migrated_oracle10" or (bool(sensor_records) and all(int(item.get("stride",-1))==4 for item in sensor_records)),
"depth_latency":profile != "migrated_oracle10" or (controller.get("metric_depth_transport_latency_max_ms") is not None and float(controller["metric_depth_transport_latency_max_ms"]) <= 5000.0),
"pilot_real_model":not model_run or (model_ready.get("status")=="PASS" and model_ready.get("backend")=="real" and model_ready.get("model_host")=="dgx_spark_only"),
"pilot_model_stopped":not model_run or (model_stopped.get("status")=="PASS" and int(model_stopped.get("process_group_remaining",-1))==0 and int(model_stopped.get("residual_probe_exit_code",-1))==0),
"pilot_weight_materialized":not model_run or (model_weight.get("status")=="PASS" and int(model_weight.get("meta_parameter_count",-1))==0 and int(model_weight.get("meta_buffer_count",-1))==0),
"pilot_client":not model_run or (client.get("status")=="FINISHED" and int(client.get("step_count",0))>0 and (client.get("model_health_start") or {}).get("status")=="PASS"),
"pilot_model_identity":not model_run or (model_weight.get("model_revision")==((client.get("model_health_start") or {}).get("model_revision")) and model_weight.get("checkpoint_revision")==((client.get("model_health_start") or {}).get("checkpoint_revision"))),
"pilot_map_resets":not model_run or (int(controller.get("static_map_selection_count",-1))==expected_count and int(controller.get("static_map_generation",-1))==expected_count-1),
"pilot_duplicate_d435_off":not model_run or int(controller.get("metric_depth_frame_count",-1))==0,
"pilot_exit_codes":not model_run or (int(sys.argv[14])==0 and int(sys.argv[15])==0),
"pilot_prerequisite":not model_run or (sys.argv[16].startswith("results/parallel/t4_functional/migrated-recovery-a5-00-") if ablation else (sys.argv[16].startswith("results/parallel/t4_functional/migrated-oracle10-00-") if profile=="migrated_pilot20" else sys.argv[16].startswith("results/parallel/t4_functional/migrated-pilot20-00-"))),
"recovery_profile":not recovery or (recovery_runtime.get("status")=="RECOVERY_RUNTIME_READY" and recovery_runtime.get("profile_id")==expected_recovery_id and recovery_runtime.get("scheduled_refresh_enabled") is True and int(recovery_runtime.get("maximum_scheduled_refreshes_per_episode",-1))==1 and recovery_metrics.get("recovery_profile_ids")==[expected_recovery_id] and recovery_metrics.get("recovery_profile_sha256")==[recovery_runtime.get("profile_sha256")]),
"recovery_functional":not recovery or (recovery_metrics.get("status")=="PASS" and int(recovery_metrics.get("full_recovery_count",0))+int(recovery_metrics.get("routine_refresh_count",0))>=1 and int(recovery_metrics.get("old_trajectory_execution_count",-1))==0 and recovery_metrics.get("typed_transaction_ok") is True and recovery_metrics.get("replan_consumption_ok") is True),
"ablation_identity":not ablation or (prepared.get("ablation_variant_id")==variant and ready.get("run_profile")==profile and model_ready.get("variant_id")==variant and model_stopped.get("variant_id")==variant and variant_claim.get("variant_id")==variant and ablation_contract.get("ablation_variant_id")==variant),
"ablation_history":not ablation or (model_ready.get("history_mode")==((ablation_contract.get("ablation_factors") or {}).get("history_mode")) and model_lifecycle.get("history_mode")==((ablation_contract.get("ablation_factors") or {}).get("history_mode"))),
"ablation_horizon":not ablation or (prepared.get("ablation_step_horizon")==1200 and isaac_receipt.get("ablation_step_horizon")==1200),
}
payload={"schema_version":1,"status":"PASS" if all(checks.values()) else "FAIL",
"checks":checks,"grant_id":sys.argv[2],"authorization_ref_sha":sys.argv[3],
"deployment_ref_sha":sys.argv[4],"functional_prepare_result":sys.argv[5],
"exit_codes":{"prepare":int(sys.argv[6]),"onboard_start":int(sys.argv[7]),
"isaac":int(sys.argv[8]),"onboard_stop":int(sys.argv[9]),"collection":int(sys.argv[10]),
"dgx_stage_cleanup":int(sys.argv[11]),"isaac_stage_cleanup":int(sys.argv[12]),
"model_start":int(sys.argv[14]),"model_stop":int(sys.argv[15])},
"architecture":{"isaac":["simulator","sensor_rendering","evaluator"],
"dgx":["navigation","map","speed_control","recovery","model"]},
"diagnostics":{"missing_files":missing,"load_errors":load_errors,
"observed_nonzero_bounded_motion":int(relay.get("nonzero_output_count",0))>=1,
"relay_warnings":relay.get("warnings") or []},
"oracle_prerequisite_result":sys.argv[16] or None,
"run_profile":profile,"expected_episode_count":expected_count,"minimum_success_rate":minimum_sr,
"ablation_variant_id":variant,"ablation_step_horizon":1200 if ablation else None,
"strict_evidence_modified":False,"real_go2_targeted":False,"recorded_unix":time.time()}
summary_name={"migration_smoke":"dgx_migration_summary.json",
"migrated_oracle10":"migrated_oracle10_summary.json",
"migrated_pilot20":"migrated_pilot20_summary.json",
"migrated_recovery_a5":"migrated_recovery_a5_summary.json",
"migrated_recovery_b5":"migrated_recovery_b5_summary.json"}.get(profile,f"migrated_ablation20_{variant}_summary.json")
(root/summary_name).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
if payload["status"]!="PASS": raise SystemExit(1)
PY
  validation_rc=$?
  set -e
  if (( validation_rc == 0 )) && [[ "$run_profile" == migrated_ablation20_* ]]; then
    ablation_variant="${run_profile#migrated_ablation20_}"
    runner_commit_sha="$(
      git --git-dir="$git_dir" --work-tree="$root" log -1 --format=%H "$authorization_ref" -- \
        scripts/build_t4_ablation_episode_records.py \
        coordination/run_t4_dgx_migration_online.sh \
        coordination/remote_t4_isaac_migration_session.sh \
        coordination/remote_t4_dgx_onboard_session.sh \
        coordination/remote_t4_model_session.sh
    )"
    set +e
    python3 "$root/scripts/build_t4_ablation_episode_records.py" \
      --result-dir "$result_dir" --variant-id "$ablation_variant" \
      --runner-commit-sha "$runner_commit_sha" \
      --output "$result_dir/episode_records.jsonl" \
      >"$result_dir/episode_records_build.log" 2>&1
    records_rc=$?
    set -e
    python3 - "$result_dir/episode_records_shadow_status.json" "$records_rc" <<'PY'
import json,sys,time
from pathlib import Path
path=Path(sys.argv[1]); rc=int(sys.argv[2])
payload={"schema_version":1,"status":"PASS" if rc==0 else "WARN",
"builder_exit_code":rc,"nonfatal_shadow":True,
"deviation":None if rc==0 else "episode_records_build_failed",
"recorded_unix":time.time()}
path.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
PY
  fi
  trap - EXIT INT TERM HUP
  (( prepare_rc == 0 )) || return "$prepare_rc"
  (( start_rc == 0 )) || return "$start_rc"
  (( isaac_rc == 0 )) || return "$isaac_rc"
  (( stop_rc == 0 )) || return "$stop_rc"
  (( model_start_rc == 0 )) || return "$model_start_rc"
  (( model_stop_rc == 0 )) || return "$model_stop_rc"
  (( collect_rc == 0 )) || return "$collect_rc"
  (( dgx_cleanup_rc == 0 && isaac_cleanup_rc == 0 )) || return 1
  return "$validation_rc"
}

if [[ "${1:-}" == _under_lease ]]; then shift; run_under_lease "$@"; exit $?; fi
[[ $# -ge 3 && $# -le 5 ]] || usage
grant_id="$1"; authorization_ref="$2"; prepare_rel="$3"
run_profile="${4:-migration_smoke}"
prerequisite_rel="${5:-}"
case "$run_profile" in
  migration_smoke)
    test -z "$prerequisite_rel" || usage
    grant_profile=dgx_onboard_migration_smoke
    result_kind=dgx-migration
    lease_kind=dgx-migration
    task_kind=dgx-onboard-migration
    ;;
  migrated_oracle10)
    test -z "$prerequisite_rel" || usage
    grant_profile=dgx_onboard_oracle10
    result_kind=migrated-oracle10
    lease_kind=migrated-oracle10
    task_kind=dgx-onboard-oracle10
    ;;
  migrated_pilot20)
    [[ "$prerequisite_rel" =~ ^results/parallel/t4_functional/migrated-oracle10-00-[A-Za-z0-9][A-Za-z0-9._-]{7,95}$ ]] || die "unsafe migrated Oracle prerequisite"
    grant_profile=dgx_onboard_pilot20
    result_kind=migrated-pilot20
    lease_kind=migrated-pilot20
    task_kind=dgx-onboard-pilot20
    ;;
  migrated_recovery_a5)
    [[ "$prerequisite_rel" =~ ^results/parallel/t4_functional/migrated-pilot20-00-[A-Za-z0-9][A-Za-z0-9._-]{7,95}$ ]] || die "unsafe migrated pilot prerequisite"
    grant_profile=dgx_onboard_recovery_a5
    result_kind=migrated-recovery-a5
    lease_kind=migrated-recovery-a5
    task_kind=dgx-onboard-recovery-a5
    ;;
  migrated_recovery_b5)
    [[ "$prerequisite_rel" =~ ^results/parallel/t4_functional/migrated-pilot20-00-[A-Za-z0-9][A-Za-z0-9._-]{7,95}$ ]] || die "unsafe migrated pilot prerequisite"
    grant_profile=dgx_onboard_recovery_b5
    result_kind=migrated-recovery-b5
    lease_kind=migrated-recovery-b5
    task_kind=dgx-onboard-recovery-b5
    ;;
  migrated_ablation20_*)
    ablation_variant="${run_profile#migrated_ablation20_}"
    case "$ablation_variant" in
      full_system1_system2|oracle_high_level_system1|system2_oracle_local_path|full_trajectory|endpoint|straight_line|model_stop|oracle_termination|history_on|history_off|recovery_on|recovery_off|h1_view|go2_view) ;;
      *) die "unsafe ablation variant" ;;
    esac
    [[ "$prerequisite_rel" =~ ^results/parallel/t4_functional/migrated-recovery-a5-00-[A-Za-z0-9][A-Za-z0-9._-]{7,95}$ ]] || die "unsafe selected recovery prerequisite"
    grant_profile="dgx_onboard_ablation20_${ablation_variant}"
    result_kind="migrated-ablation20-${ablation_variant}"
    lease_kind="migrated-ablation20-${ablation_variant}"
    task_kind="dgx-onboard-ablation20-${ablation_variant}"
    ;;
  *) die "unknown run profile" ;;
esac
[[ "$grant_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{7,95}$ ]] || die "unsafe grant"
[[ "$authorization_ref" =~ ^[0-9a-f]{40}$ ]] || die "authorization SHA must be full"
[[ "$prepare_rel" =~ ^results/parallel/t4_functional/prepare-00-[A-Za-z0-9][A-Za-z0-9._-]{7,95}$ ]] || die "unsafe prepare path"
script_path="$(readlink -f "${BASH_SOURCE[0]}")"
root="$(readlink -f "$(dirname "$script_path")/..")"
git_dir="$(resolve_git_dir "$root")"
contract_tool="$root/coordination/t4_functional_run_contract.py"
actual_sha="$(git -c core.autocrlf=false --git-dir="$git_dir" rev-parse "$COORDINATION_REF")"
test "$actual_sha" = "$authorization_ref"
test -z "$(git -c core.autocrlf=true --git-dir="$git_dir" --work-tree="$root" status --porcelain --untracked-files=no)"
mapfile -t fields < <(python3 "$contract_tool" prepare-fields \
  --result-dir "$root/$prepare_rel" --git-dir "$git_dir" \
  --authorization-ref "$authorization_ref" --format pilot-lines)
test "${#fields[@]}" -eq 6
deployment_ref="${fields[1]}"; dgx_root="${fields[2]}"; isaac_root="${fields[4]}"
if test "$run_profile" = migrated_pilot20; then
  python3 - "$root/$prerequisite_rel/migrated_oracle10_summary.json" <<'PY'
import json,sys
value=json.load(open(sys.argv[1],encoding="utf-8"))
checks=(value.get("status")=="PASS",value.get("run_profile")=="migrated_oracle10",
        int(value.get("expected_episode_count",-1))==10,
        float(value.get("minimum_success_rate",-1.0))==0.8,
        value.get("real_go2_targeted") is False)
if not all(checks): raise SystemExit("migrated Oracle 10 prerequisite is not PASS")
PY
elif [[ "$run_profile" == migrated_recovery_a5 || "$run_profile" == migrated_recovery_b5 ]]; then
  python3 - "$root/$prerequisite_rel/migrated_pilot20_summary.json" <<'PY'
import json,sys
value=json.load(open(sys.argv[1],encoding="utf-8"))
checks=(value.get("status")=="PASS",value.get("run_profile")=="migrated_pilot20",
        int(value.get("expected_episode_count",-1))==20,
        float(value.get("minimum_success_rate",-1.0))==0.01,
        bool((value.get("checks") or {}).get("success_rate")),
        value.get("real_go2_targeted") is False)
if not all(checks): raise SystemExit("migrated pilot 20 prerequisite is not PASS")
PY
elif [[ "$run_profile" == migrated_ablation20_* ]]; then
  python3 - "$root/$prerequisite_rel/migrated_recovery_a5_summary.json" <<'PY'
import json,sys
value=json.load(open(sys.argv[1],encoding="utf-8"))
checks=(value.get("status")=="PASS",value.get("run_profile")=="migrated_recovery_a5",
        int(value.get("expected_episode_count",-1))==5,
        bool((value.get("checks") or {}).get("recovery_functional")),
        value.get("real_go2_targeted") is False)
if not all(checks): raise SystemExit("selected migrated recovery A prerequisite is not PASS")
PY
fi
result_rel="results/parallel/t4_functional/${result_kind}-00-${grant_id}"
lease_rel="results/parallel/t4_functional/lease-${lease_kind}-00-${grant_id}"
test ! -e "$root/$result_rel"; test ! -e "$root/$lease_rel"
python3 - "$root" "$git_dir" "$authorization_ref" "$grant_id" "$result_rel" "$grant_profile" <<'PY'
import json,re,subprocess,sys
root,git_dir,ref,grant,result,grant_profile=sys.argv[1:]
doc=subprocess.run(["git",f"--git-dir={git_dir}",f"--work-tree={root}","show",f"{ref}:coordination/TASK_BOARD.md"],check=True,capture_output=True,text=True).stdout
blocks=re.findall(r"<!--\s*INTERNAV_ONLINE_GRANT_V1\s*\r?\n(.*?)\r?\nINTERNAV_ONLINE_GRANT_V1\s*-->",doc,flags=re.DOTALL)
expected={"schema_version":1,"status":"GRANTED","worker":"00","resource":"dgx+isaac","profile":grant_profile,"result_dir":result,"grant_id":grant}
if len(blocks)!=1 or json.loads(blocks[0])!=expected: raise SystemExit("DGX migration grant mismatch")
PY
bash "$root/scripts/with_resource_lease.sh" both --owner codex-00 \
  --task "00-${task_kind}-${grant_id}" --log-dir "$root/$lease_rel" -- \
  bash "$script_path" _under_lease "$root" "$grant_id" "$authorization_ref" \
    "$deployment_ref" "$dgx_root" "$isaac_root" "$result_rel" "$lease_rel" \
    "$prepare_rel" "$contract_tool" "$run_profile" "$prerequisite_rel"
