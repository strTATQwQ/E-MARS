#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

[[ $# -eq 6 || $# -eq 7 ]] || { echo "remote T4 model session: invalid arguments" >&2; exit 64; }
action="$1"
deployment_root="$2"
stage="$3"
grant_id="$4"
authorization_ref="$5"
deployment_ref="$6"
run_profile="${7:-migrated_pilot20}"
ablation_variant=none
case "$action" in start|stop) ;; *) echo "remote T4 model session: invalid action" >&2; exit 64 ;; esac
case "$run_profile" in
  migrated_pilot20|migrated_recovery_a5|migrated_recovery_b5) ;;
  migrated_ablation20_*)
    ablation_variant="${run_profile#migrated_ablation20_}"
    case "$ablation_variant" in
      full_system1_system2|oracle_high_level_system1|system2_oracle_local_path|full_trajectory|endpoint|straight_line|model_stop|oracle_termination|history_on|history_off|recovery_on|recovery_off|h1_view|go2_view) ;;
      *) exit 64 ;;
    esac
    ;;
  *) exit 64 ;;
esac
[[ "$grant_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{7,95}$ ]] || exit 64
[[ "$authorization_ref" =~ ^[0-9a-f]{40}$ ]] || exit 64
[[ "$deployment_ref" =~ ^[0-9a-f]{40}$ ]] || exit 64

readonly EXPECTED_USER="${INTERNNAV_T4_EXPECTED_USER:-railgun}"
readonly EXPECTED_IP="${INTERNNAV_T4_EXPECTED_IP:-10.100.100.128}"
readonly STABLE_ROOT="${INTERNNAV_T4_STABLE_ROOT:-/home/railgun/internnav-t1-t2}"
readonly STAGE_BASE="${INTERNNAV_T4_STAGE_BASE:-/home/railgun/.codex-internnav-stage}"
readonly LANE_ROS_DOMAIN_ID="${INTERNNAV_T4_ROS_DOMAIN_ID:-71}"
readonly EXPECTED_STAGE="$STAGE_BASE/t4-pilot-model-${grant_id}"
readonly ROS_WORKSPACE="$deployment_root/ros_ws"
if test "$ablation_variant" = none; then
  result_name="model-pilot-${grant_id}"
else
  result_name="model-ablation-${ablation_variant}-${grant_id}"
fi
readonly RESULT_DIR="$deployment_root/results/internnav_t4/$result_name"
readonly CONTRACT_TOOL="$deployment_root/coordination/t4_functional_run_contract.py"
readonly PROCESS_RECORD="$stage/model_process.json"

test "$(id -un)" = "$EXPECTED_USER"
ip -4 -o addr show | grep -Fq " $EXPECTED_IP/"
[[ "$LANE_ROS_DOMAIN_ID" =~ ^[0-9]{1,3}$ ]]
test "$stage" = "$EXPECTED_STAGE"
test -d "$stage" && test ! -L "$stage"
test "$(realpath "$stage")" = "$EXPECTED_STAGE"
test -d "$deployment_root" && test ! -L "$deployment_root"
case "$deployment_root" in "$STABLE_ROOT/.t4-deployments/"*) ;; *) exit 64 ;; esac
test "$(realpath "$deployment_root")" = "$deployment_root"
test -x "$CONTRACT_TOOL"
test -f "$deployment_root/scripts/run_t4_model_server.sh"
test -x "$deployment_root/scripts/t4_model_health_probe.py"

if test "$action" = start; then
  history_mode=on
  if test "$ablation_variant" != none; then
    test -f "$stage/variant_config.json"
    mapfile -t ablation_fields < <(
      python3 "$deployment_root/scripts/t4_ablation_runtime.py" fields \
        --config "$stage/variant_config.json" \
        --matrix "$deployment_root/configs/completion_sim/ablation/frozen_matrix_v1.json"
    )
    test "${#ablation_fields[@]}" -eq 11
    test "${ablation_fields[0]}" = "$ablation_variant"
    history_mode="${ablation_fields[6]}"
  fi
  python3 "$CONTRACT_TOOL" validate-deployment \
    --role dgx --deployment-root "$deployment_root" \
    --deployment-ref "$deployment_ref" \
    >"$stage/deployment_validation.json"
  python3 "$deployment_root/coordination/t4_functional_payload.py" verify-tree \
    --root "$deployment_root" \
    --manifest "$deployment_root/payload_manifest.json" \
    --expected-ref "$deployment_ref" \
    >"$stage/payload_reverification.json"
  python3 "$CONTRACT_TOOL" validate-dgx-workspace \
    --deployment-root "$deployment_root" --workspace "$ROS_WORKSPACE" \
    >"$stage/dgx_workspace_validation.json"
  set +u
  export ROS_DOMAIN_ID="$LANE_ROS_DOMAIN_ID"
  source /opt/ros/jazzy/setup.bash
  source "$ROS_WORKSPACE/install/setup.bash"
  set -u
  export ROS2CLI_NO_DAEMON=1
  set +e
  timeout 10s ros2 node list >"$stage/model_prestart_nodes.txt" \
    2>"$stage/model_prestart_nodes.log"
  graph_rc=$?
  pgrep -af '[i]nternvla_t4_recovery.model_node|[i]nternvla_ros2.model_node' \
    >"$stage/model_prestart_processes.txt" 2>"$stage/model_prestart_processes.log"
  process_rc=$?
  set -e
  test "$graph_rc" = 0
  ! grep -Eq '(^|/)internvla_model_node$' "$stage/model_prestart_nodes.txt"
  test "$process_rc" = 1
  test ! -e "$RESULT_DIR"
  mkdir -p "$(dirname "$RESULT_DIR")"
  python3 - "$stage/model_invocation.json" "$grant_id" "$authorization_ref" \
    "$deployment_ref" "$deployment_root" "$RESULT_DIR" "$run_profile" \
    "$ablation_variant" "$history_mode" <<'PY'
import json,sys,time
from pathlib import Path
payload={
    "schema_version":1,
    "status":"READY_TO_START",
    "grant_id":sys.argv[2],
    "authorization_ref_sha":sys.argv[3],
    "deployment_ref_sha":sys.argv[4],
    "deployment_root":sys.argv[5],
    "result_dir":sys.argv[6],
    "runtime_policy":"completion_sim",
    "runtime_target":"isaac_simulation",
    "resource_lease_ack":"dgx+isaac",
    "backend":"real",
    "preload_model":True,
    "history_mode":sys.argv[9],
    "run_profile":sys.argv[7],
    "variant_id":None if sys.argv[8]=="none" else sys.argv[8],
    "model_host":"dgx_spark_only",
    "strict_evidence_modified":False,
    "real_go2_targeted":False,
    "recorded_unix":time.time(),
}
Path(sys.argv[1]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
PY

  model_pid=""
  cleanup_failed_start() {
    local rc=$?
    trap - EXIT INT TERM HUP
    set +e
    if [[ "$model_pid" =~ ^[1-9][0-9]*$ ]]; then
      kill -TERM -- "-$model_pid" 2>/dev/null || true
      sleep 1
      kill -KILL -- "-$model_pid" 2>/dev/null || true
    fi
    exit "$rc"
  }
  trap cleanup_failed_start EXIT
  trap 'exit 130' INT TERM HUP

  model_mode_env=(-u INTERNVLA_MODEL_CONFIG)
  if test "$ablation_variant" = none; then
    model_mode_env+=(
      -u INTERNVLA_T4_VARIANT_CONFIG
      INTERNVLA_T4_FUNCTIONAL_MODEL=1
      INTERNVLA_T4_HISTORY_MODE=on
    )
  else
    model_mode_env+=(
      INTERNVLA_T4_VARIANT_CONFIG="$stage/variant_config.json"
      INTERNVLA_T4_FUNCTIONAL_MODEL=0
    )
  fi
  setsid env \
    "${model_mode_env[@]}" \
    INTERNNAV_T1_CONTROL_ROOT="$deployment_root" \
    INTERNVLA_ROS_WS="$ROS_WORKSPACE" \
    INTERNVLA_MODEL_RESULT_DIR="$RESULT_DIR" \
    INTERNNAV_T4_RESOURCE_LEASE_ACK=dgx+isaac \
    INTERNNAV_RUNTIME_POLICY=completion_sim \
    INTERNNAV_SIMULATION_TARGET=isaac \
    INTERNVLA_BACKEND=real \
    INTERNVLA_PRELOAD_MODEL=1 \
    HF_ENDPOINT=https://hf-mirror.com \
    timeout --signal=TERM --kill-after=30s 14400s \
      bash "$deployment_root/scripts/run_t4_model_server.sh" \
      >"$stage/model_outer.log" 2>&1 </dev/null &
  model_pid=$!
  sleep 0.2
  model_pgid="$(ps -o pgid= -p "$model_pid" | tr -d ' ')"
  test "$model_pgid" = "$model_pid"
  kill -0 "$model_pid"
  python3 - "$PROCESS_RECORD" "$model_pid" "$model_pgid" "$deployment_root" \
    "$RESULT_DIR" "$grant_id" <<'PY'
import json,sys,time
from pathlib import Path
payload={
    "schema_version":1,"status":"RUNNING","pid":int(sys.argv[2]),
    "pgid":int(sys.argv[3]),"deployment_root":sys.argv[4],
    "result_dir":sys.argv[5],"grant_id":sys.argv[6],"started_unix":time.time(),
}
Path(sys.argv[1]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
PY

  ready=0
  for _ in $(seq 1 240); do
    kill -0 "$model_pid" 2>/dev/null || break
    set +e
    python3 "$deployment_root/scripts/t4_model_health_probe.py" \
      --output "$stage/model_health_ready.json" --timeout-sec 5 \
      >"$stage/model_health_probe.log" 2>&1
    probe_rc=$?
    set -e
    if test "$probe_rc" = 0; then
      ready=1
      break
    fi
  done
  test "$ready" = 1
  kill -0 "$model_pid"
  python3 - "$RESULT_DIR/model_weight_audit.json" "$stage/model_health_ready.json" \
    "$stage/model_ready_receipt.json" "$grant_id" "$authorization_ref" \
    "$deployment_ref" "$deployment_root" "$model_pid" "$model_pgid" \
    "$run_profile" "$ablation_variant" "$history_mode" <<'PY'
import json,sys,time
from pathlib import Path
weight=json.load(open(sys.argv[1],encoding="utf-8"))
health=json.load(open(sys.argv[2],encoding="utf-8"))
checks={
    "weight_status":weight.get("status")=="PASS",
    "weight_backend":weight.get("backend")=="real",
    "weight_meta":weight.get("meta_parameter_count")==0 and weight.get("meta_buffer_count")==0,
    "health_status":health.get("status")=="PASS",
    "health_host":health.get("model_host")=="dgx_spark",
    "health_action":health.get("step_action_ready") is True,
    "revision_match":weight.get("model_revision")==health.get("model_revision"),
    "checkpoint_match":weight.get("checkpoint_revision")==health.get("checkpoint_revision"),
}
if not all(checks.values()):
    raise SystemExit(f"model ready contract failed: {checks}")
payload={
    "schema_version":1,"status":"PASS","checks":checks,
    "grant_id":sys.argv[4],"authorization_ref_sha":sys.argv[5],
    "deployment_ref_sha":sys.argv[6],"deployment_root":sys.argv[7],
    "model_pid":int(sys.argv[8]),"model_pgid":int(sys.argv[9]),
    "run_profile":sys.argv[10],
    "variant_id":None if sys.argv[11]=="none" else sys.argv[11],
    "history_mode":sys.argv[12],
    "runtime_policy":"completion_sim","runtime_target":"isaac_simulation",
    "resource_lease_ack":"dgx+isaac","backend":"real",
    "model_host":"dgx_spark_only","strict_evidence_modified":False,
    "real_go2_targeted":False,"recorded_unix":time.time(),
}
Path(sys.argv[3]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
PY
  cp "$stage/model_invocation.json" "$RESULT_DIR/model_invocation.json"
  cp "$stage/model_health_ready.json" "$RESULT_DIR/model_health_ready.json"
  cp "$stage/model_ready_receipt.json" "$RESULT_DIR/model_ready_receipt.json"
  trap - EXIT INT TERM HUP
  exit 0
fi

# stop
test -f "$PROCESS_RECORD" && test ! -L "$PROCESS_RECORD"
mapfile -t PROCESS_FIELDS < <(python3 - "$PROCESS_RECORD" "$deployment_root" "$grant_id" <<'PY'
import json,sys
value=json.load(open(sys.argv[1],encoding="utf-8"))
if value.get("deployment_root") != sys.argv[2] or value.get("grant_id") != sys.argv[3]:
    raise SystemExit("model process identity changed")
pid=int(value.get("pid",0)); pgid=int(value.get("pgid",0))
if pid <= 1 or pgid != pid:
    raise SystemExit("model process group is unsafe")
print(pid); print(pgid); print(value.get("result_dir",""))
PY
)
test "${#PROCESS_FIELDS[@]}" -eq 3
model_pid="${PROCESS_FIELDS[0]}"
model_pgid="${PROCESS_FIELDS[1]}"
test "${PROCESS_FIELDS[2]}" = "$RESULT_DIR"
was_alive=0
kill -0 "$model_pid" 2>/dev/null && was_alive=1
if test "$was_alive" = 1; then
  command_text="$(tr '\0' ' ' <"/proc/$model_pid/cmdline")"
  [[ "$command_text" == *"$deployment_root/scripts/run_t4_model_server.sh"* ]]
  kill -INT -- "-$model_pgid" 2>/dev/null || true
  for _ in $(seq 1 300); do
    kill -0 -- "-$model_pgid" 2>/dev/null || break
    sleep 0.1
  done
  kill -0 -- "-$model_pgid" 2>/dev/null && kill -TERM -- "-$model_pgid" 2>/dev/null || true
  for _ in $(seq 1 100); do
    kill -0 -- "-$model_pgid" 2>/dev/null || break
    sleep 0.1
  done
  kill -0 -- "-$model_pgid" 2>/dev/null && kill -KILL -- "-$model_pgid" 2>/dev/null || true
fi
group_remaining=0
kill -0 -- "-$model_pgid" 2>/dev/null && group_remaining=1
set +e
python3 "$CONTRACT_TOOL" residual-probe --role dgx \
  --deployment-root "$deployment_root" --output "$stage/model_residual_probe.json" \
  >"$stage/model_residual_probe.log" 2>&1
residual_rc=$?
archive_rc=125
if test -d "$RESULT_DIR" && test ! -L "$RESULT_DIR"; then
  cp "$stage/model_outer.log" "$RESULT_DIR/model_outer.log"
  cp "$stage/model_residual_probe.json" "$RESULT_DIR/model_residual_probe.json" 2>/dev/null || true
  tar -C "$RESULT_DIR" -czf "$stage/model_result.tgz" .
  archive_rc=$?
fi
set -e
python3 - "$stage/model_stop_receipt.json" "$was_alive" "$group_remaining" \
  "$residual_rc" "$archive_rc" "$grant_id" "$authorization_ref" \
  "$deployment_ref" "$deployment_root" "$run_profile" "$ablation_variant" <<'PY'
import json,sys,time
from pathlib import Path
alive,remaining,residual,archive=map(int,sys.argv[2:6])
passed=alive==1 and remaining==0 and residual==0 and archive==0
payload={
    "schema_version":1,"status":"PASS" if passed else "FAIL",
    "model_was_alive":bool(alive),"process_group_remaining":remaining,
    "residual_probe_exit_code":residual,"archive_exit_code":archive,
    "grant_id":sys.argv[6],"authorization_ref_sha":sys.argv[7],
    "deployment_ref_sha":sys.argv[8],"deployment_root":sys.argv[9],
    "run_profile":sys.argv[10],
    "variant_id":None if sys.argv[11]=="none" else sys.argv[11],
    "runtime_policy":"completion_sim","runtime_target":"isaac_simulation",
    "resource_lease_ack":"dgx+isaac","model_host":"dgx_spark_only",
    "strict_evidence_modified":False,"real_go2_targeted":False,
    "recorded_unix":time.time(),
}
Path(sys.argv[1]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
if not passed:
    raise SystemExit(1)
PY
