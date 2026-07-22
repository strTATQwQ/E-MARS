#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

[[ $# -eq 6 ]] || { echo "remote T4 pilot: invalid arguments" >&2; exit 64; }
deployment_root="$1"
stage="$2"
grant_id="$3"
authorization_ref="$4"
deployment_ref="$5"
model_ready_sha256="$6"
[[ "$grant_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{7,95}$ ]] || exit 64
[[ "$authorization_ref" =~ ^[0-9a-f]{40}$ ]] || exit 64
[[ "$deployment_ref" =~ ^[0-9a-f]{40}$ ]] || exit 64
[[ "$model_ready_sha256" =~ ^[0-9a-f]{64}$ ]] || exit 64

readonly STABLE_ROOT=/home/song/internnav-t1-t2
readonly EXPECTED_STAGE="/home/song/.codex-internnav-stage/t4-pilot-${grant_id}"
readonly ROS_WORKSPACE=/home/song/internnav-t4/isaac_ros_ws_45
readonly RUN_LABEL="functional_pilot20_${grant_id}"
readonly RESULT_ROOT="$deployment_root/results/internnav_t4"
readonly RESULT_DIR="$RESULT_ROOT/${RUN_LABEL}_model_attempt_001"
readonly RECEIPT="$stage/pilot_run_receipt.json"
readonly CONTRACT_TOOL="$deployment_root/coordination/t4_functional_run_contract.py"

test "$(id -un)" = song
ip -4 -o addr show | grep -Fq " 10.100.120.111/"
test "$stage" = "$EXPECTED_STAGE"
test -d "$stage" && test ! -L "$stage"
test "$(realpath "$stage")" = "$EXPECTED_STAGE"
test -d "$deployment_root" && test ! -L "$deployment_root"
case "$deployment_root" in "$STABLE_ROOT/.t4-deployments/"*) ;; *) exit 64 ;; esac
test "$(realpath "$deployment_root")" = "$deployment_root"
test -x "$CONTRACT_TOOL"
test -f "$deployment_root/scripts/run_t4_sensor_gate.sh"
test -f "$deployment_root/scripts/t4_cleanup_container_processes.sh"
test -f "$ROS_WORKSPACE/install/setup.bash"
python3 "$CONTRACT_TOOL" validate-deployment \
  --deployment-root "$deployment_root" --deployment-ref "$deployment_ref" \
  >"$stage/deployment_validation.json"
python3 "$deployment_root/coordination/t4_functional_payload.py" verify-tree \
  --root "$deployment_root" --manifest "$deployment_root/payload_manifest.json" \
  --expected-ref "$deployment_ref" >"$stage/payload_reverification.json"
python3 "$CONTRACT_TOOL" validate-overlay-source \
  --deployment-root "$deployment_root" --workspace "$ROS_WORKSPACE" \
  >"$stage/overlay_source_validation.json"

test ! -e "$RESULT_DIR"
mkdir -p "$RESULT_ROOT"
python3 - "$stage/pilot_invocation.json" "$grant_id" "$authorization_ref" \
  "$deployment_ref" "$deployment_root" "$model_ready_sha256" <<'PY'
import json,sys,time
from pathlib import Path
payload={
    "schema_version":1,"status":"READY","grant_id":sys.argv[2],
    "authorization_ref_sha":sys.argv[3],"deployment_ref_sha":sys.argv[4],
    "deployment_root":sys.argv[5],"model_ready_receipt_sha256":sys.argv[6],
    "runtime_policy":"completion_sim","runtime_target":"isaac_simulation",
    "resource_lease_ack":"dgx+isaac","map_source":"static_map",
    "pose_source":"ground_truth","expected_episode_count":20,
    "minimum_sr":0.01,"use_t4_adapter":True,"recovery_enabled":False,
    "model_host":"dgx_spark_only","model_process_started":True,
    "strict_evidence_modified":False,"real_go2_targeted":False,
    "recorded_unix":time.time(),
}
Path(sys.argv[1]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
PY

raw_run_rc=125
run_rc=125
validation_rc=125
cleanup_primary_rc=125
cleanup_probe_rc=125
residual_probe_rc=125
archive_rc=125

finalize() {
  local incoming_rc=$? final_rc=1
  trap - EXIT INT TERM HUP
  set +e
  INTERNNAV_T1_CONTROL_ROOT="$deployment_root" \
  INTERNVLA_T4_CONTAINER_NAME=internnav_t4_isaac_ros \
    bash "$deployment_root/scripts/t4_cleanup_container_processes.sh" \
      >"$stage/container_cleanup_primary.log" 2>&1
  cleanup_primary_rc=$?
  INTERNNAV_T1_CONTROL_ROOT="$deployment_root" \
  INTERNVLA_T4_CONTAINER_NAME=internnav_t4_isaac_ros \
    bash "$deployment_root/scripts/t4_cleanup_container_processes.sh" \
      >"$stage/container_cleanup_probe.log" 2>&1
  cleanup_probe_rc=$?
  grep -Fxq 'trial_processes_initial=0 forced_kill=0' \
    "$stage/container_cleanup_probe.log" || cleanup_probe_rc=1
  python3 "$CONTRACT_TOOL" residual-probe \
    --deployment-root "$deployment_root" \
    --socket-root "$deployment_root/runtime/t4_ipc" \
    --output "$stage/residual_probe.json" \
    >"$stage/residual_probe.log" 2>&1
  residual_probe_rc=$?

  if test -d "$RESULT_DIR" && test ! -L "$RESULT_DIR"; then
    cp "$stage/pilot_invocation.json" "$RESULT_DIR/pilot_invocation.json"
    cp "$stage/pilot_outer.log" "$RESULT_DIR/pilot_outer.log" 2>/dev/null || true
    cp "$stage/container_cleanup_primary.log" "$RESULT_DIR/container_cleanup_primary.log"
    cp "$stage/container_cleanup_probe.log" "$RESULT_DIR/container_cleanup_probe.log"
    cp "$stage/residual_probe.json" "$RESULT_DIR/residual_probe.json" 2>/dev/null || true
    python3 "$CONTRACT_TOOL" validate-pilot-result \
      --result-dir "$RESULT_DIR" --grant-id "$grant_id" \
      --authorization-ref "$authorization_ref" --deployment-ref "$deployment_ref" \
      --output "$RESULT_DIR/coordinator_pilot_validation.json" \
      >"$stage/pilot_validation.log" 2>&1
    validation_rc=$?
    cp "$stage/pilot_validation.log" "$RESULT_DIR/pilot_validation.log"
    tar -C "$RESULT_DIR" -czf "$stage/pilot_result.tgz" .
    archive_rc=$?
  fi
  run_rc=$raw_run_rc
  if test "$raw_run_rc" = 143 && test "$validation_rc" = 0 \
      && test "$cleanup_primary_rc" = 0 && test "$cleanup_probe_rc" = 0 \
      && test "$residual_probe_rc" = 0; then
    run_rc=0
  fi
  if test "$run_rc" = 0 && test "$validation_rc" = 0 \
      && test "$cleanup_primary_rc" = 0 && test "$cleanup_probe_rc" = 0 \
      && test "$residual_probe_rc" = 0 && test "$archive_rc" = 0; then
    final_rc=0
  elif test "$incoming_rc" -ne 0; then
    final_rc=$incoming_rc
  elif test "$raw_run_rc" -ne 0; then
    final_rc=$raw_run_rc
  fi
  python3 - "$RECEIPT" "$final_rc" "$raw_run_rc" "$run_rc" "$validation_rc" \
    "$cleanup_primary_rc" "$cleanup_probe_rc" "$residual_probe_rc" "$archive_rc" \
    "$grant_id" "$authorization_ref" "$deployment_ref" "$deployment_root" \
    "$RESULT_DIR" "$model_ready_sha256" <<'PY'
import json,sys,time
from pathlib import Path
codes=list(map(int,sys.argv[2:10])); final,raw,run,validation,primary,probe,residual,archive=codes
payload={
    "schema_version":1,"status":"PASS" if final==0 else "FAIL","exit_code":final,
    "raw_run_exit_code":raw,"run_exit_code":run,"validation_exit_code":validation,
    "cleanup_primary_exit_code":primary,"cleanup_probe_exit_code":probe,
    "residual_probe_exit_code":residual,"archive_exit_code":archive,
    "grant_id":sys.argv[10],"authorization_ref_sha":sys.argv[11],
    "deployment_ref_sha":sys.argv[12],"deployment_root":sys.argv[13],
    "result_dir":sys.argv[14],"model_ready_receipt_sha256":sys.argv[15],
    "runtime_policy":"completion_sim","runtime_target":"isaac_simulation",
    "resource_lease_ack":"dgx+isaac","map_source":"static_map",
    "pose_source":"ground_truth","expected_episode_count":20,"minimum_sr":0.01,
    "model_host":"dgx_spark_only","model_process_started":True,
    "strict_evidence_modified":False,"real_go2_targeted":False,
    "recorded_unix":time.time(),
}
temporary=Path(sys.argv[1]).with_suffix(".tmp")
temporary.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
temporary.replace(Path(sys.argv[1]))
PY
  exit "$final_rc"
}
trap finalize EXIT
trap 'exit 130' INT TERM HUP

export INTERNNAV_T1_CONTROL_ROOT="$deployment_root"
export INTERNVLA_ROS_WS="$ROS_WORKSPACE"
export INTERNVLA_T4_CONTAINER_NAME=internnav_t4_isaac_ros
export INTERNVLA_T4_COMPLETION_FAST_PATH=1
export INTERNNAV_RUNTIME_POLICY=completion_sim
export INTERNNAV_SIMULATION_TARGET=isaac
export INTERNNAV_T4_RESOURCE_LEASE_ACK=dgx+isaac
export INTERNVLA_T4_USE_T4_ADAPTER=1
export INTERNVLA_T4_ENABLE_RECOVERY=0
export INTERNVLA_T4_EXPECTED_COUNT=20
export INTERNVLA_T4_MIN_SR_OVERRIDE=0.01
export INTERNVLA_T4_RUN_LABEL="$RUN_LABEL"
export INTERNVLA_T4_RESULT_ROOT="$RESULT_ROOT"

set +e
bash "$deployment_root/scripts/run_t4_sensor_gate.sh" t4_4 model 001 \
  >"$stage/pilot_outer.log" 2>&1
raw_run_rc=$?
set -e
exit "$raw_run_rc"
