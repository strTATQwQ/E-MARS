from __future__ import annotations

import gzip
import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dgx_onboard_launcher_owns_navigation_map_and_speed() -> None:
    launcher = (ROOT / "scripts" / "run_t4_dgx_onboard.sh").read_text(
        encoding="utf-8"
    )
    assert "ipc_transport:=tcp" in launcher
    assert "tcp_expected_peer_ip" in launcher
    assert "internvla_t4_sensor_bridge" in launcher
    assert "NAV2_LAUNCH_FILE=navigation_launch.py" in launcher
    assert 'ros2 launch nav2_bringup "$NAV2_LAUNCH_FILE"' in launcher
    assert "internvla_t4_adapter" in launcher
    assert '"$CONTROL_ROOT/t4_completion/map/warn_relay.py"' in launcher
    assert 'speed_control_owner": "dgx"' in launcher
    assert "INTERNVLA_GO2_CONTROLLER_SOCKET" not in launcher
    assert "sensor_timeout_sec:=5.0" in launcher
    assert "INTERNVLA_T4_RECOVERY_SAFETY_FRESHNESS_SEC:-2.0" in launcher
    assert "INTERNVLA_T4_RECOVERY_SAFETY_FRESHNESS_SEC:-5.0" not in launcher
    assert "INTERNVLA_T4_ALLOW_COMMAND_POSE_ANCHOR_FALLBACK:-0" in launcher
    assert "allow_command_pose_anchor_fallback:" in launcher
    assert "ros2 service type /internvla/nav2_resolve" in launcher
    assert "params_file:=\"$PARAMS\" use_sim_time:=False" in launcher
    assert "params_file:=\"$PARAMS\" use_sim_time:=True" not in launcher
    assert "check_t4_nav2_lifecycle.py" in launcher
    assert "nav2_lifecycle_ready.json" in launcher
    assert "check_t4_nav2_data_plane.py" in launcher
    assert "nav2_data_plane_ready.json" in launcher
    assert (
        'if test -n "$NODE_NAMESPACE"; then\n'
        "  # The data-plane isolation proof is a T5 namespace gate."
    ) in launcher
    assert '--params-file "$PARAMS" -p use_sim_time:=false' in launcher
    assert "tcp_ready()" in launcher
    assert "resolver_ready()" in launcher
    assert "onboard_readiness.log" in launcher
    assert "ss -H -ltn | awk" not in launcher
    assert "ros2 lifecycle get \"$node\"" not in launcher
    lifecycle_probe = (ROOT / "scripts" / "check_t4_nav2_lifecycle.py").read_text(
        encoding="utf-8"
    )
    assert '"/controller_server"' in lifecycle_probe
    assert '"/collision_monitor"' in lifecycle_probe
    assert "state_id == 3" in lifecycle_probe
    assert 'choices=NODES' in lifecycle_probe
    assert 'state_id == 2' in lifecycle_probe
    assert '"inactive_deviations"' in lifecycle_probe
    assert "GetState" in lifecycle_probe
    data_plane_probe = (
        ROOT / "scripts" / "check_t4_nav2_data_plane.py"
    ).read_text(encoding="utf-8")
    assert "get_action_server_names_and_types" in data_plane_probe
    assert "get_action_client_names_and_types_by_node" in data_plane_probe
    assert "get_publishers_info_by_topic" in data_plane_probe
    assert "get_subscriptions_info_by_topic" in data_plane_probe
    assert "INTERNVLA_T4_ONBOARD_PROFILE" in launcher
    assert "LIFECYCLE_PROBE_ARGS+=(--allow-inactive /collision_monitor)" in launcher
    assert "LIFECYCLE_PROBE_ARGS+=(--allow-inactive /bt_navigator)" in launcher
    assert 'result_dir:="$RESULT_DIR/go2_sensor_bridge"' in launcher
    assert "enable_rgb:=false" in launcher
    bridge = (
        ROOT / "go2_sensor_bridge" / "go2_sensor_bridge" / "bridge_node.py"
    ).read_text(encoding="utf-8")
    assert "_ingest_completion_locked" in bridge
    assert '"consumer_group": "nav2_lidar_nearest_tf"' in bridge
    assert '"completion_nearest_tf_matches"' in bridge
    client = (
        ROOT / "internvla_t4_sensors" / "internvla_t4_sensors" / "client_node.py"
    ).read_text(encoding="utf-8")
    sensor_bridge = (
        ROOT
        / "internvla_t4_sensors"
        / "internvla_t4_sensors"
        / "sensor_bridge_node.py"
    ).read_text(encoding="utf-8")
    assert '"/internvla_t4/episode_prime"' in client
    assert '"/internvla_t4/episode_prime"' in sensor_bridge
    assert "navigation_odometry_timeout_sec:=2.0" in (
        ROOT / "scripts" / "build_t4_sensor_phase_overlay.py"
    ).read_text(encoding="utf-8")
    gate = (ROOT / "scripts" / "run_t4_sensor_gate.sh").read_text(encoding="utf-8")
    assert "INTERNVLA_T4_PHASE_OVERLAY_BUILDER" in gate
    assert '"dgx_onboard":bool(os.environ.get("INTERNVLA_GO2_CONTROLLER_ENDPOINT"))' in gate


def test_command_pose_anchor_fallback_is_completion_sim_only_and_audited() -> None:
    adapter = (
        ROOT
        / "internvla_nav2_adapter"
        / "internvla_nav2_adapter"
        / "active_node.py"
    ).read_text(encoding="utf-8")
    assert 'declare_parameter("allow_command_pose_anchor_fallback", False)' in adapter
    assert 'os.environ.get("INTERNNAV_RUNTIME_POLICY", "") == "completion_sim"' in adapter
    assert 'os.environ.get("INTERNNAV_SIMULATION_TARGET", "") == "isaac"' in adapter
    assert "command_pose_anchor_fallback_count" in adapter
    assert '"path_anchor_odom_age_sec"' in adapter
    assert '"path_anchor_source"' in adapter


def test_dgx_onboard_startup_diagnostics_preserve_fail_closed_readiness() -> None:
    launcher = (ROOT / "scripts" / "run_t4_dgx_onboard.sh").read_text(
        encoding="utf-8"
    )

    assert "set -euo pipefail" in launcher
    assert "set -Eeuo pipefail" not in launcher
    assert "export ROS2CLI_NO_DAEMON=1" in launcher
    assert "write_failure_diagnostic() (" in launcher
    assert "onboard_failure_diagnostic.json" in launcher
    assert '"resolver_ready_diagnostic_reprobe"' in launcher
    assert 'resolver_ready 2>"$RESULT_DIR/logs/failure_resolver_probe.log"' in launcher
    assert '"required_child_exit:$component:$pid"' in launcher
    assert 'trap \'on_error "$?" "$LINENO"\' ERR' in launcher
    assert "BASH_COMMAND" not in launcher
    assert "ONBOARD_FAILURE_COMMAND" not in launcher
    assert ".onboard_failure_diagnostic.$BASHPID.json" in launcher
    assert 'if write_failure_diagnostic "$rc" "$line"; then' in launcher
    assert "FAILURE_DIAGNOSTIC_WRITTEN=1" in launcher
    assert 'if test "$SHUTDOWN_REASON" = process_exit; then' in launcher
    assert 'test "$?" = 0 || { rm -f "$temporary" "$children"; return 1; }' in launcher
    assert 'mv -f "$temporary" "$output" || {' in launcher
    assert 'return "$rc"' in launcher
    assert "trap - ERR EXIT INT TERM HUP" in launcher
    assert 'ps -o stat=,pgid= -p "$pid"' in launcher
    assert '[[ "$state" != Z* && "$pgid" = "$pid" ]]' in launcher

    readiness = launcher.split('CURRENT_PHASE="readiness_wait"', 1)[1]
    ready_receipt = readiness.index('CURRENT_PHASE="onboard_ready_receipt"')
    online_ready = readiness.index("ONLINE_READY=1")
    required_gates = (
        'CURRENT_PHASE="readiness_tcp_recheck"\ntcp_ready',
        'CURRENT_PHASE="readiness_warn_only_relay_recheck"\n'
        'test -s "$RESULT_DIR/warn_only_relay.jsonl"',
        'CURRENT_PHASE="readiness_resolver_recheck"\nresolver_ready_bounded',
        'CURRENT_PHASE="readiness_lifecycle_receipt_recheck"',
        'CURRENT_PHASE="readiness_data_plane_receipt_recheck"',
        'CURRENT_PHASE="readiness_required_child_recheck"',
    )
    for gate in required_gates:
        assert gate in readiness
        assert readiness.index(gate) < ready_receipt
    assert ready_receipt < online_ready
    assert 'resolver_ready || true' not in readiness[:online_ready]
    assert "for attempt in $(seq 1 2)" in launcher
    assert "timeout --signal=TERM --kill-after=0.5s 3s" in launcher
    assert "timeout 3 ros2 service type" not in launcher
    assert 'resolver_type="$LAST_RESOLVER_TYPE"' in launcher
    assert 'test "$service_type" = internvla_ros2_msgs/srv/ResolveCommand' in launcher
    assert 'require_child_alive controller "$controller_pid"' in readiness[:online_ready]
    assert "for probe in $(seq 1 8)" in readiness[:online_ready]
    assert 'resolver_ready 2>/dev/null && resolver_state=1' in readiness[:online_ready]
    assert 'log_resolver_probe readiness "$probe" "$resolver_state"' in readiness[:online_ready]
    assert '$(resolver_ready' not in readiness[:online_ready]
    resolver_budget_sec = 8 * (3.0 + 0.5 + 0.1) + 2 * (3.0 + 0.5 + 0.1) + 3.5
    assert resolver_budget_sec <= 45.0


def test_resolver_recheck_tolerates_only_bounded_exact_type_discovery() -> None:
    launcher = (ROOT / "scripts" / "run_t4_dgx_onboard.sh").read_text(
        encoding="utf-8"
    )
    resolver = re.search(
        r"^resolver_ready\(\) \{\n.*?^\}", launcher, re.MULTILINE | re.DOTALL
    )
    query = re.search(
        r"^resolver_type_query\(\) \{\n.*?^\}",
        launcher,
        re.MULTILINE | re.DOTALL,
    )
    bounded = re.search(
        r"^resolver_ready_bounded\(\) \{\n.*?^\}",
        launcher,
        re.MULTILINE | re.DOTALL,
    )
    logger = re.search(
        r"^log_resolver_probe\(\) \{\n.*?^\}",
        launcher,
        re.MULTILINE | re.DOTALL,
    )
    assert query is not None
    assert resolver is not None
    assert bounded is not None
    assert logger is not None

    probe = subprocess.run(
        ["bash"],
        input=f"""\
set -euo pipefail
{query.group(0)}
{resolver.group(0)}
{logger.group(0)}
{bounded.group(0)}
tmpdir="$(mktemp -d)"
counter="$tmpdir/counter"
cleanup() {{ rm -rf "$tmpdir"; }}
trap cleanup EXIT
sleep() {{ :; }}
RESULT_DIR="$tmpdir"
mkdir -p "$RESULT_DIR/logs"
cat >"$tmpdir/ros2" <<'MOCK_ROS2'
#!/usr/bin/env bash
set -euo pipefail
value="$(cat "$ROS2_TEST_COUNTER")"
value=$((value + 1))
printf '%s\n' "$value" >"$ROS2_TEST_COUNTER"
case "$ROS2_TEST_MODE" in
  transient) ((value >= 2)) || exit 1; printf '%s\n' internvla_ros2_msgs/srv/ResolveCommand ;;
  wrong) printf '%s\n' wrong_msgs/srv/Wrong ;;
  missing) exit 1 ;;
  slow_exact) command sleep 1; printf '%s\n' internvla_ros2_msgs/srv/ResolveCommand ;;
  slow_wrong) command sleep 1; printf '%s\n' wrong_msgs/srv/Wrong ;;
  hang) trap '' TERM; command sleep 30 ;;
esac
MOCK_ROS2
chmod +x "$tmpdir/ros2"
export PATH="$tmpdir:$PATH"
export ROS2_TEST_COUNTER="$counter"
printf '0\n' >"$counter"
export ROS2_TEST_MODE=transient
resolver_ready_bounded
test "$(cat "$counter")" = 2
printf '0\n' >"$counter"
export ROS2_TEST_MODE=wrong
! resolver_ready_bounded
test "$(cat "$counter")" = 2
printf '0\n' >"$counter"
export ROS2_TEST_MODE=missing
! resolver_ready_bounded
test "$(cat "$counter")" = 2
printf '0\n' >"$counter"
export ROS2_TEST_MODE=slow_exact
resolver_ready
test "$LAST_RESOLVER_RC" = 0
test "$LAST_RESOLVER_ELAPSED_MS" -ge 800
printf '0\n' >"$counter"
export ROS2_TEST_MODE=slow_wrong
! resolver_ready
test "$LAST_RESOLVER_RC" = 0
test "$LAST_RESOLVER_ELAPSED_MS" -ge 800
printf '0\n' >"$counter"
export ROS2_TEST_MODE=hang
! resolver_ready
test "$(cat "$counter")" = 1
""".encode(),
        capture_output=True,
        check=False,
        timeout=12,
    )
    assert probe.returncode == 0, probe.stderr.decode(errors="replace")


def test_failure_diagnostic_resolver_probe_kills_term_ignoring_cli() -> None:
    launcher = (ROOT / "scripts" / "run_t4_dgx_onboard.sh").read_text(
        encoding="utf-8"
    )
    query = re.search(
        r"^resolver_type_query\(\) \{\n.*?^\}",
        launcher,
        re.MULTILINE | re.DOTALL,
    )
    resolver = re.search(
        r"^resolver_ready\(\) \{\n.*?^\}", launcher, re.MULTILINE | re.DOTALL
    )
    diagnostic_body = launcher.split("write_failure_diagnostic() (", 1)[1].split(
        "\n)\n\non_error()", 1
    )[0]
    diagnostic = "write_failure_diagnostic() (" + diagnostic_body + "\n)"
    child = re.search(
        r"^child_alive\(\) \{\n.*?^\}", launcher, re.MULTILINE | re.DOTALL
    )
    required = re.search(
        r"^require_child_alive\(\) \{\n.*?^\}",
        launcher,
        re.MULTILINE | re.DOTALL,
    )
    on_error = re.search(
        r"^on_error\(\) \{\n.*?^\}", launcher, re.MULTILINE | re.DOTALL
    )
    assert query is not None
    assert resolver is not None
    assert child is not None
    assert required is not None
    assert on_error is not None

    probe = subprocess.run(
        ["bash"],
        input=f"""\
set -euo pipefail
{query.group(0)}
{resolver.group(0)}
{diagnostic}
{child.group(0)}
{required.group(0)}
{on_error.group(0)}
tmpdir="$(mktemp -d)"
cleanup() {{ rm -rf "$tmpdir"; }}
trap cleanup EXIT
mkdir -p "$tmpdir/result/logs" "$tmpdir/bin"
cat >"$tmpdir/bin/ros2" <<'MOCK_ROS2'
#!/usr/bin/env bash
case "${{ROS2_DIAGNOSTIC_TEST_MODE:-hang}}" in
  hang) trap '' TERM; sleep 30 ;;
  missing) exit 1 ;;
esac
MOCK_ROS2
chmod +x "$tmpdir/bin/ros2"
export PATH="$tmpdir/bin:$PATH"
export ROS2_DIAGNOSTIC_TEST_MODE=hang
RESULT_DIR="$tmpdir/result"
CURRENT_PHASE=diagnostic_hang_test
SHUTDOWN_REASON=test_failure
ONLINE_READY=0
LAST_READINESS_PROBE=7
FAILURE_DIAGNOSTIC_WRITTEN=0
nav2_pid=""
nav_lifecycle_pid=""
collision_lifecycle_pid=""
adapter_pid=""
controller_pid=""
go2_sensor_pid=""
relay_pid=""
recovery_pid=""
tcp_ready() {{ return 1; }}
write_failure_diagnostic 17 321
python3 - "$RESULT_DIR/onboard_failure_diagnostic.json" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
assert value["exit_code"] == 17
assert value["source_line"] == 321
assert value["phase"] == "diagnostic_hang_test"
assert value["readiness"]["resolver_ready_diagnostic_reprobe"] is False
assert value["readiness"]["resolver_rc_diagnostic_reprobe"] in {{"124", "137"}}
assert value["readiness"]["resolver_elapsed_ms_diagnostic_reprobe"] >= 3000
PY
export ROS2_DIAGNOSTIC_TEST_MODE=missing
SHUTDOWN_REASON=process_exit
CURRENT_PHASE=before_child_probe
! require_child_alive controller 99999999
error_rc=0
on_error 1 654 || error_rc=$?
test "$error_rc" = 1
python3 - "$RESULT_DIR/onboard_failure_diagnostic.json" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
assert value["source_line"] == 654
assert value["phase"] == "required_child_liveness:controller"
assert value["shutdown_reason"] == "required_child_exit:controller:99999999"
PY
""".encode(),
        capture_output=True,
        check=False,
        timeout=8,
    )
    assert probe.returncode == 0, probe.stderr.decode(errors="replace")


def test_migration_coordinator_is_joint_lease_ref_bound_and_ordered() -> None:
    coordinator = (ROOT / "coordination" / "run_t4_dgx_migration_online.sh").read_text(
        encoding="utf-8"
    )
    dgx = (ROOT / "coordination" / "remote_t4_dgx_onboard_session.sh").read_text(
        encoding="utf-8"
    )
    isaac = (
        ROOT / "coordination" / "remote_t4_isaac_migration_session.sh"
    ).read_text(encoding="utf-8")
    launcher = (ROOT / "scripts" / "run_t4_dgx_onboard.sh").read_text(
        encoding="utf-8"
    )
    assert '"resource":"dgx+isaac"' in coordinator
    assert "grant_profile=dgx_onboard_migration_smoke" in coordinator
    assert 'with_resource_lease.sh" both' in coordinator
    migration_inputs = coordinator.index("migration_inputs.tgz")
    onboard_ready = coordinator.index("onboard_ready_receipt.json", migration_inputs)
    isaac_receipt = coordinator.index("isaac_run_receipt.json", onboard_ready)
    assert migration_inputs < onboard_ready < isaac_receipt
    assert "run_t4_dgx_onboard.sh" in dgx
    assert "for _ in $(seq 1 600)" in dgx
    assert "migrated_recovery_a5|migrated_recovery_b5" in dgx
    assert "enable_d435i=0" in dgx
    assert 'INTERNVLA_T4_ONBOARD_PROFILE="$run_profile"' in dgx
    assert '"go2_sensor_bridge_mode": "shadow_nonfatal"' in launcher
    fatal_loop = launcher.split("while :; do", 1)[1]
    assert '"$go2_sensor_pid"' not in fatal_loop.split("sleep 1", 1)[0]
    assert "build_t4_isaac_remote_phase_overlay.py" in isaac
    assert "INTERNVLA_T4_R3_ENABLE_LIDAR=1" in isaac
    assert "INTERNVLA_T4_R3_ENABLE_RGB_IPC=0" in isaac
    assert 'INTERNVLA_T4_R3_ENABLE_D435I="$enable_d435i"' in isaac
    assert '--episode-count "$episode_count"' in isaac
    assert 'INTERNVLA_T4_EXPECTED_COUNT="$episode_count"' in isaac
    assert "INTERNVLA_T4_PHASE_OVERRIDE=continuous_oracle" in isaac
    assert 'DGX_BIND_IP="${INTERNNAV_T4_DGX_BIND_IP:-10.100.100.128}"' in isaac
    assert 'CONTROLLER_PORT="${INTERNVLA_T4_CONTROLLER_TCP_PORT:-24137}"' in isaac
    assert 'INTERNVLA_GO2_CONTROLLER_ENDPOINT="tcp://$DGX_BIND_IP:$CONTROLLER_PORT"' in isaac
    assert "dgx_onboard_oracle10" in coordinator
    assert "migrated_oracle10_summary.json" in coordinator
    assert "migrated_pilot20_summary.json" in coordinator
    assert "migrated_recovery_a5_summary.json" in coordinator
    assert "migrated_recovery_b5_summary.json" in coordinator
    assert "grant_profile=dgx_onboard_pilot20" in coordinator
    assert "grant_profile=dgx_onboard_recovery_a5" in coordinator
    assert "grant_profile=dgx_onboard_recovery_b5" in coordinator
    assert "remote_t4_model_session.sh" in coordinator
    assert 'source_dataset_name=t3_model_pilot_go2_clear_v2' in isaac
    assert 'evaluator_role=model' in isaac
    assert 'export INTERNNAV_T0_CONTROL_ROOT="$deployment_root"' in isaac
    assert 'INTERNVLA_T4_MODEL_DATASET_ROOT="$SMOKE_DATASET"' in isaac
    assert 'run_t4_sensor_gate.sh" t4_4 "$evaluator_role" 001' in isaac
    assert '"pilot_real_model"' in coordinator
    assert '"pilot_model_stopped"' in coordinator
    assert '"pilot_weight_materialized"' in coordinator
    assert '"pilot_client"' in coordinator
    assert "DEFERRED_TO_MODEL_STOP" in dgx
    assert '"residual_probe_deferred_to_model_stop"' in dgx
    assert "model_stage_created=1" in coordinator
    assert 'remove_stage dgx "$model_stage"' in coordinator
    assert 'remote_get dgx "$dgx_stage/onboard_result.tgz"' in coordinator
    model_start = coordinator.index(
        'if (( model_profile == 1 )); then\n    create_stage dgx "$model_stage"'
    )
    onboard_start = coordinator.index("onboard_started=1", model_start)
    isaac_done = coordinator.index("isaac_rc=$?", onboard_start)
    onboard_stop = coordinator.index(
        'set +e; remote_exec dgx "$command_text"; stop_rc=$?; set -e',
        isaac_done,
    )
    model_stop = coordinator.index("model_stop_rc=$?", onboard_stop)
    assert model_start < onboard_start < onboard_stop < model_stop
    assert 'int(item.get("stride",-1))==4' in coordinator
    assert '"diagnostics":{"missing_files":missing,"load_errors":load_errors,' in coordinator
    assert 'onboard_started=1\n  printf -v command_text \'bash %q start' in coordinator


def test_migrated_recovery_profiles_are_dgx_owned_and_cross_host_analyzed() -> None:
    coordinator = (ROOT / "coordination" / "run_t4_dgx_migration_online.sh").read_text(
        encoding="utf-8"
    )
    dgx = (ROOT / "coordination" / "remote_t4_dgx_onboard_session.sh").read_text(
        encoding="utf-8"
    )
    isaac = (
        ROOT / "coordination" / "remote_t4_isaac_migration_session.sh"
    ).read_text(encoding="utf-8")
    launcher = (ROOT / "scripts" / "run_t4_dgx_onboard.sh").read_text(
        encoding="utf-8"
    )
    analyzer = (ROOT / "scripts" / "analyze_t4_recovery.py").read_text(
        encoding="utf-8"
    )

    assert "migrated_recovery_a5" in coordinator
    assert "migrated_recovery_b5" in coordinator
    assert 'episode_count=5' in isaac
    assert 'source_dataset_name=t3_model_pilot_go2_clear_v2' in isaac
    assert 'recovery_profile="$deployment_root/configs/completion_sim/recovery/profile_a.json"' in dgx
    assert 'recovery_profile="$deployment_root/configs/completion_sim/recovery/profile_b.json"' in dgx
    assert 'INTERNVLA_T4_ENABLE_RECOVERY="$enable_recovery"' in dgx
    assert 'INTERNVLA_T4_RECOVERY_MODE="$recovery_mode"' in dgx
    assert 'INTERNVLA_NAV2_PARAMS="$nav2_params"' in dgx
    assert "t4_recovery_runtime.py" in dgx
    assert 'recovery_profile_id:="${INTERNVLA_T4_RECOVERY_PROFILE_ID' in launcher
    assert 'maximum_scheduled_refreshes_per_episode:="$MAXIMUM_SCHEDULED_REFRESHES"' in launcher
    assert 'INTERNVLA_T4_ENABLE_SCHEDULED_REFRESH="$recovery_enable_scheduled_refresh"' in dgx
    assert 'recovery_enable_scheduled_refresh=1' in dgx
    assert 'recovery_runtime.get("scheduled_refresh_enabled") is True' in coordinator
    assert "--per-episode" in coordinator
    assert "--replan-records" in coordinator
    assert "--adapter-records" in coordinator
    assert '"recovery_functional"' in coordinator
    assert '"recovery_profile"' in coordinator
    assert 'parser.add_argument("--per-episode"' in analyzer


def test_migrated_ablation_matrix_is_serial_joint_lease_and_dgx_owned() -> None:
    batch = (
        ROOT / "coordination" / "run_t4_migrated_ablation_matrix_online.sh"
    ).read_text(encoding="utf-8")
    coordinator = (
        ROOT / "coordination" / "run_t4_dgx_migration_online.sh"
    ).read_text(encoding="utf-8")
    isaac = (
        ROOT / "coordination" / "remote_t4_isaac_migration_session.sh"
    ).read_text(encoding="utf-8")
    dgx = (
        ROOT / "coordination" / "remote_t4_dgx_onboard_session.sh"
    ).read_text(encoding="utf-8")
    model = (ROOT / "coordination" / "remote_t4_model_session.sh").read_text(
        encoding="utf-8"
    )

    assert 'with_resource_lease.sh" both' in batch
    assert 'readonly VARIANTS=(' in batch
    assert 'profile="migrated_ablation20_${variant}"' in batch
    assert batch.count('with_resource_lease.sh" both') == 1
    assert "break" in batch
    assert '"profile":"dgx_onboard_ablation20_matrix"' in batch
    assert "finalize_t4_migrated_ablation.py" in batch
    assert "ONLINE_PASS_EVIDENCE_PENDING" in batch
    assert "build_t4_ablation_episode_records.py" in coordinator
    assert "episode_records_shadow_status.json" in coordinator
    assert '"nonfatal_shadow":True' in coordinator
    assert "ablation_step_horizon" in coordinator
    assert 'variant=profile[len(ablation_prefix):] if ablation else None' in coordinator
    assert ".removeprefix(" not in coordinator
    assert 'git --git-dir="$git_dir" --work-tree="$root" log -1' in coordinator
    assert 'git -C "$root" log -1' not in coordinator
    assert "INTERNVLA_T4_MAX_STEP=1200" in isaac
    assert "archive_members=(static_maps smoke_dataset)" in isaac
    assert "t4_ablation_generate.py" in isaac
    assert 'recovery_profile="$deployment_root/configs/completion_sim/recovery/profile_a.json"' in dgx
    assert 'INTERNVLA_T4_ABLATION_DATASET_FILE="$ablation_dataset"' in dgx
    assert 'INTERNVLA_T4_VARIANT_CONFIG="$stage/variant_config.json"' in model
    assert 'INTERNVLA_T4_FUNCTIONAL_MODEL=0' in model


def test_dgx_dependency_installer_takes_no_secret_argument() -> None:
    installer = (
        ROOT / "scripts" / "install_t4_dgx_nav2_dependencies.sh"
    ).read_text(encoding="utf-8")
    build = (ROOT / "scripts" / "build_t4_host_ros.sh").read_text(encoding="utf-8")
    launcher = (ROOT / "scripts" / "run_t4_dgx_onboard.sh").read_text(
        encoding="utf-8"
    )
    assert "ros-$ROS_DISTRO_NAME-navigation2" in installer
    assert "ros-$ROS_DISTRO_NAME-nav2-bringup" in installer
    assert "sudo -S -p ''" in installer
    assert "PASSWORD=" not in installer
    for package in (
        "internvla_nav2_adapter",
        "internvla_go2_controller",
        "internvla_t4_sensors",
        "internvla_t4_recovery",
        "go2_sensor_bridge",
    ):
        assert package in build
    assert 'PYTHONPATH="$CONTROL_ROOT:${PYTHONPATH:-}" python3' in build
    assert 'export PYTHONPATH="$CONTROL_ROOT:${PYTHONPATH:-}"' in launcher
    coordinator = (
        ROOT / "coordination" / "run_t4_dgx_dependency_install_online.sh"
    ).read_text(encoding="utf-8")
    assert '"resource":"dgx"' in coordinator
    assert '"profile":"dgx_onboard_dependencies"' in coordinator
    assert 'with_resource_lease.sh" dgx' in coordinator
    assert "DGX_SUDO_PASSWORD_FILE" in coordinator
    assert 'printf \'%s\\n\' "$DGX_SUDO_PASSWORD" | ssh' in coordinator
    assert "DGX_SUDO_PASSWORD" not in coordinator.split("--task", 1)[1]


def test_migration_smoke_dataset_is_one_episode_and_deterministic(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json.gz"
    payload = {
        "episodes": [
            {"episode_id": "e0", "scene_id": "s0"},
            {"episode_id": "e1", "scene_id": "s1"},
        ]
    }
    with gzip.open(source, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream)
    outputs = []
    for index in range(2):
        output_root = tmp_path / f"out-{index}"
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "build_t4_migration_smoke_dataset.py"),
                "--source",
                str(source),
                "--output-root",
                str(output_root),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        outputs.append(output_root / "val_unseen" / "val_unseen.json.gz")
    assert outputs[0].read_bytes() == outputs[1].read_bytes()
    with gzip.open(outputs[0], "rt", encoding="utf-8") as stream:
        selected = json.load(stream)
    assert selected["episodes"] == [{"episode_id": "e0", "scene_id": "s0"}]

    two_root = tmp_path / "out-two"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_t4_migration_smoke_dataset.py"),
            "--source",
            str(source),
            "--output-root",
            str(two_root),
            "--episode-count",
            "2",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    with gzip.open(
        two_root / "val_unseen" / "val_unseen.json.gz", "rt", encoding="utf-8"
    ) as stream:
        selected = json.load(stream)
    assert selected["episodes"] == payload["episodes"]


def test_bounded_t4_pilot_count_override_is_completion_sim_only(
    tmp_path: Path,
) -> None:
    internnav_root = tmp_path / "InternNav"
    upstream = (
        internnav_root
        / "scripts"
        / "eval"
        / "configs"
        / "h1_internvla_n1_async_cfg.py"
    )
    upstream.parent.mkdir(parents=True)
    upstream.write_text(
        "from types import SimpleNamespace as S\n"
        "eval_cfg=S(agent=S(server_host='',model_settings={}),eval_settings={},"
        "dataset=S(dataset_settings={}),task=S(task_name=''))\n",
        encoding="utf-8",
    )
    dataset_root = tmp_path / "dataset"
    episode_file = dataset_root / "val_unseen" / "val_unseen.json.gz"
    episode_file.parent.mkdir(parents=True)
    with gzip.open(episode_file, "wt", encoding="utf-8") as stream:
        json.dump({"episodes": [{"episode_id": str(i)} for i in range(5)]}, stream)
    env = os.environ.copy()
    env.update(
        {
            "INTERNNAV_T0_PHASE": "pilot",
            "INTERNNAV_ROOT": str(internnav_root),
            "INTERNNAV_T0_DATASET_ROOT": str(dataset_root),
            "INTERNNAV_T0_RESULT_DIR": str(tmp_path / "result"),
            "INTERNNAV_SERVER_HOST": "127.0.0.1",
            "INTERNVLA_T4_EXPECTED_COUNT": "5",
            "INTERNNAV_RUNTIME_POLICY": "completion_sim",
            "INTERNNAV_SIMULATION_TARGET": "isaac",
            "INTERNNAV_T4_RESOURCE_LEASE_ACK": "dgx+isaac",
        }
    )
    command = [
        sys.executable,
        "-c",
        "import runpy,sys; runpy.run_path(sys.argv[1])",
        str(ROOT / "configs" / "internnav_t0" / "official_agent_server_cfg.py"),
    ]
    completed = subprocess.run(
        command, env=env, check=False, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    env["INTERNNAV_RUNTIME_POLICY"] = "strict_evidence"
    completed = subprocess.run(
        command, env=env, check=False, capture_output=True, text=True
    )
    assert completed.returncode != 0
    assert "bounded T4 pilot episode-count override" in completed.stderr

    # T5 fixed-five uses the official five-episode canary contract, not a
    # shortened pilot override.  Leaving the pilot-only override in the
    # environment would be rejected by the frozen config.
    env["INTERNNAV_T0_PHASE"] = "canary"
    env["INTERNNAV_RUNTIME_POLICY"] = "completion_sim"
    env.pop("INTERNVLA_T4_EXPECTED_COUNT")
    completed = subprocess.run(
        command, env=env, check=False, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    env["INTERNVLA_T4_EXPECTED_COUNT"] = "5"
    completed = subprocess.run(
        command, env=env, check=False, capture_output=True, text=True
    )
    assert completed.returncode != 0
    assert "bounded T4 pilot episode-count override" in completed.stderr


def test_isaac_remote_phase_starts_no_local_navigation_or_speed_processes(
    tmp_path: Path,
) -> None:
    output = tmp_path / "isaac_remote_phase.sh"
    manifest = tmp_path / "manifest.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_t4_isaac_remote_phase_overlay.py"),
            "--source",
            str(ROOT / "scripts" / "run_go2_continuous_phase.sh"),
            "--output",
            str(output),
            "--manifest",
            str(manifest),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    generated = output.read_text(encoding="utf-8")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert "INTERNVLA_GO2_CONTROLLER_ENDPOINT" in generated
    assert "setsid ros2 launch nav2_bringup" not in generated
    assert "setsid ros2 run internvla_t4_sensors internvla_t4_sensor_bridge" not in generated
    assert "setsid python3" not in generated
    assert 'test "${INTERNNAV_T4_RESOURCE_LEASE_ACK:-}" = dgx+isaac' in generated
    assert payload["local_navigation_processes_started"] is False
    assert payload["local_speed_control_processes_started"] is False


def test_r3_runtime_can_freeze_review_rgb_off_without_disabling_lidar(
    tmp_path: Path,
) -> None:
    output = tmp_path / "r3_runtime.py"
    manifest = tmp_path / "r3_runtime_manifest.json"
    env = os.environ.copy()
    env["INTERNVLA_T4_R3_ENABLE_RGB_IPC"] = "0"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_t4_r3_sensor_runtime_overlay.py"),
            "--source",
            str(ROOT / "scripts" / "internnav_go2_runtime.py"),
            "--output",
            str(output),
            "--manifest",
            str(manifest),
        ],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    generated = output.read_text(encoding="utf-8")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert "'0' != \"1\"" in generated
    assert payload["frozen_geometry_sources"] == {
        "d435i": True,
        "lidar": True,
        "lidar_ray_count": 1440,
        "lidar_width": 180,
        "rgb_ipc": False,
    }


def test_remote_deployment_cleanup_is_role_and_path_bounded() -> None:
    cleanup = (ROOT / "scripts" / "cleanup_t4_remote_deployment.sh").read_text(
        encoding="utf-8"
    )
    assert 'case "$role" in' in cleanup
    assert '/home/railgun/internnav-t1-t2/.t4-deployments/' in cleanup
    assert '/home/song/internnav-t1-t2/.t4-deployments/' in cleanup
    assert 'dgx_b)' in cleanup
    assert 'remote="rail@10.100.120.116"' in cleanup
    assert 'isaac_b)' in cleanup
    assert 'test ! -L "$target"' in cleanup
    assert 'test "$(realpath "$target")" = "$target"' in cleanup
    assert 'case "$target" in "$allowed_prefix"?*)' in cleanup
    assert 'rm -rf -- "$target"' in cleanup
    assert 'test ! -e "$target"' in cleanup


def test_parallel_resume_freezes_two_dgx_two_gpu_isolation_and_memory_gate() -> None:
    coordinator = (
        ROOT / "coordination" / "run_t4_dgx_migration_online.sh"
    ).read_text(encoding="utf-8")
    parallel = (
        ROOT / "coordination" / "run_t4_migrated_ablation_parallel_resume_online.sh"
    ).read_text(encoding="utf-8")
    importer = (ROOT / "scripts" / "prepare_t4_ablation_resume.py").read_text(
        encoding="utf-8"
    )
    lease = (ROOT / "scripts" / "with_resource_lease.sh").read_text(
        encoding="utf-8"
    )
    policy = (ROOT / "coordination" / "RESOURCE_POLICY.md").read_text(
        encoding="utf-8"
    )

    assert "DGX_REMOTE_USER=railgun DGX_REMOTE_HOST=10.100.100.128" in coordinator
    assert "DGX_REMOTE_USER=rail DGX_REMOTE_HOST=10.100.120.116" in coordinator
    assert "LANE_ROS_DOMAIN_ID=71 LANE_TCP_PORT=24137" in coordinator
    assert "LANE_ROS_DOMAIN_ID=72 LANE_TCP_PORT=24138" in coordinator
    assert "ISAAC_GPU_INDEX=0" in coordinator
    assert "ISAAC_GPU_INDEX=1" in coordinator
    assert "CUDA_VISIBLE_DEVICES=%q" in coordinator

    assert "lane-a) requested_resources=(dgx_a isaac_gpu0)" in lease
    assert "lane-b) requested_resources=(dgx_b isaac_gpu1)" in lease
    assert "all-lanes) requested_resources=(dgx_a dgx_b isaac_gpu0 isaac_gpu1)" in lease
    assert "lock_file='/tmp/internnav_isaac_gpu0.lock'" in lease
    assert "lock_file='/tmp/internnav_isaac_gpu1.lock'" in lease
    assert "recovery_on:11 recovery_off:12" in parallel
    assert "h1_view:13 go2_view:14" in parallel
    assert "observed_single_instance_increment_bytes" in parallel
    assert "required_remaining_bytes" in parallel
    assert "memory_probe_ok" in parallel
    assert "MEMORY_PROBE_FAILED" in parallel
    assert 'for _ in $(seq 1 15)' in parallel
    assert 'minimum_available_bytes="$sample_available_bytes"' in parallel
    assert "MEMORY_UPGRADE_REQUIRED" in parallel
    assert "ISAAC_GPU0_RUNTIME_NOT_OBSERVED" in parallel
    assert '"memory_upgrade_requested":False' in parallel
    assert '"future_plan"' in parallel
    assert '"resume_entrypoint"' in parallel
    assert "power_off_x86_isaac_host_only_after_user_confirmation" in parallel
    assert "automatic_shutdown_performed" in parallel
    assert "HEARTBEAT resource=%s host=%s holder_pid=%s at=%s" in lease
    assert "trap 'exit 130' HUP INT TERM" in lease
    assert 'read -r -t 10 control' in lease
    assert "persist_holder_diagnostics" in lease
    assert 'holder_${resource}.stderr.log' in lease
    assert 'EXTERNAL_READ_ONLY_EVIDENCE = Path("dgx-run/logs/nav2.log")' in importer
    assert '"sha256": sha256(external_path)' in importer
    assert '"copied_into_resume": False' in importer
    assert '"source_result_must_be_archived_with_resume"' in importer
    assert "DGX-A → DGX-B → Isaac-GPU0 → Isaac-GPU1" in policy
    assert '"set -euo pipefail; install -d -m 700 \'$dgx_b_root\'; tar' in (
        ROOT / "scripts" / "prepare_t4_parallel_lanes.sh"
    ).read_text(encoding="utf-8")
    assert "'$dgx_b_root/ros_ws/src'" in (
        ROOT / "scripts" / "prepare_t4_parallel_lanes.sh"
    ).read_text(encoding="utf-8")
    lane_prepare = (ROOT / "scripts" / "prepare_t4_parallel_lanes.sh").read_text(
        encoding="utf-8"
    )
    assert "--exclude=/results --exclude=/runtime" in lane_prepare
    assert "--exclude results --exclude runtime" not in lane_prepare
    assert (
        lane_prepare.count(
            '"set -euo pipefail; export PYTHONDONTWRITEBYTECODE=1; python3'
        )
        == 2
    )
    assert (
        '"set -eo pipefail; export PYTHONDONTWRITEBYTECODE=1; set +u; source '
        in lane_prepare
    )

    assert 'ADDITIONAL_VARIANTS = ("recovery_on", "recovery_off")' in importer
    assert "--additional-source-result" in importer
    view_resume = (
        ROOT / "coordination" / "run_t4_migrated_ablation_view_resume_online.sh"
    ).read_text(encoding="utf-8")
    assert 'profile":"dgx_onboard_ablation20_view_resume"' in view_resume
    assert 'with_resource_lease.sh" lane-b' in view_resume
    assert "h1_view:13 go2_view:14" in view_resume
    assert "recovery_on:11 recovery_off:12" not in view_resume
    assert '"second_isaac_runs_alone_in_resume":True' in view_resume
