from __future__ import annotations

import gzip
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_t5_distributed_isaac.sh"


def source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def shell_function(name: str) -> str:
    match = re.search(
        rf"^{name}\(\) \{{\n.*?^\}}", source(), re.MULTILINE | re.DOTALL
    )
    assert match is not None, name
    return match.group(0)


def test_embedded_python_blocks_compile() -> None:
    lines = source().splitlines()
    compiled = 0
    index = 0
    while index < len(lines):
        if "<<'PY'" not in lines[index]:
            index += 1
            continue
        end = index + 1
        while end < len(lines) and lines[end] != "PY":
            end += 1
        assert end < len(lines), f"unclosed Python heredoc at line {index + 1}"
        compile(
            "\n".join(lines[index + 1 : end]) + "\n",
            f"{SCRIPT}:heredoc:{index + 1}",
            "exec",
        )
        compiled += 1
        index = end + 1
    assert compiled == 26


def test_entrypoint_is_lane_parameterized_and_requires_exact_lane_lease() -> None:
    text = source()
    assert "run_t5_distributed_isaac.sh LANE MODE RESULT_ROOT DATASET_ROOT" in text
    assert '[[ $# -eq 4 ]] || usage' in text
    assert "expected_lease=lane-a" in text
    assert "expected_lease=lane-b" in text
    assert 'INTERNNAV_T5_RESOURCE_LEASE_ACK:-}" = "$expected_lease"' in text
    assert "INTERNNAV_T5_RESOURCE_LEASE_ACK=t5-primary" not in text
    assert 'test "${INTERNNAV_RUNTIME_POLICY:-}" = completion_sim' in text


def test_run_token_uniquely_binds_upstream_resume_and_result_identity() -> None:
    text = source()
    token = text.index('run_token="$(basename "$result_root")"')
    label = text.index('task_run_label="t5_${lane_name}_${mode}_${run_token}"')
    export = text.index('export INTERNVLA_T4_RUN_LABEL="$task_run_label"')
    assert token < label < export
    assert '[[ "$run_token" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}$ ]]' in text
    assert 'upstream_task_name="${task_run_label}_${mode}_001"' in text
    assert 'data/sample_episodes/$upstream_task_name' in text
    assert 'logs/$upstream_task_name' in text


def test_t5_model_identity_uses_exact_materialized_evaluator_order() -> None:
    text = source()
    assert 'export INTERNVLA_MODEL_EPISODE_ORDER_MANIFEST=' in text
    assert "materialized_upstream_manifest" in text
    assert "reverse_gzip" not in text
    assert '"model_identity_order": (' in text
    assert '"model_episode_order_manifest": os.environ.get(' in text


def test_lane_a_has_frozen_gpu_domain_ports_container_and_cpuset() -> None:
    text = source()
    for fragment in (
        "gpu=0",
        "ros_domain_id=75",
        "edge_ip=10.100.100.128",
        "controller_port=25137",
        "model_client_port=25139",
        "oracle_port=25140",
        "clock_port=25141",
        "container=internnav_t5_isaac_a",
        'cpuset="${INTERNVLA_T5_LANE_A_CPUSET:-0,2,4,6,8,10,12,14,16}"',
        "identity_prefix='a::'",
        "ipc_alias=/tmp/internnav_t5_a_ipc",
    ):
        assert fragment in text


def test_lane_b_has_frozen_gpu_domain_ports_container_and_cpuset() -> None:
    text = source()
    for fragment in (
        "gpu=1",
        "ros_domain_id=76",
        "edge_ip=10.100.120.116",
        "controller_port=25138",
        "model_client_port=25239",
        "oracle_port=25240",
        "clock_port=25241",
        "container=internnav_t5_isaac_b",
        'cpuset="${INTERNVLA_T5_LANE_B_CPUSET:-1,3,5,7,9,11,13,15,17}"',
        "identity_prefix='b::'",
        "ipc_alias=/tmp/internnav_t5_b_ipc",
    ):
        assert fragment in text


def test_runtime_locks_one_lane_and_reads_shared_assets_concurrently() -> None:
    text = source()
    assert "lane_lock=/tmp/internnav_t5_isaac_a_runtime.lock" in text
    assert "lane_lock=/tmp/internnav_t5_isaac_b_runtime.lock" in text
    assert "shared_asset_lock=/tmp/internnav_t5_isaac_shared_assets.lock" in text
    assert 'exec 8>"$lane_lock"' in text
    assert "flock -n 8" in text
    assert 'exec 9>"$shared_asset_lock"' in text
    assert "flock -s -n 9" in text
    assert "flock -u 9" in text
    assert "flock -u 8" in text
    assert "lane_lock_released=true" in text


def test_host_and_container_cpu_gpu_isolation_are_fail_closed() -> None:
    text = source()
    assert '[[ "$cpuset" =~ ^[0-9,-]+$ ]]' in text
    assert 'taskset -pc "$cpuset" "$$"' in text
    assert "docker-exec does not inherit the client's affinity mask" in text
    assert "{{.HostConfig.CpusetCpus}}" in text
    assert "{{.HostConfig.PidMode}}" in text
    assert "{{.HostConfig.IpcMode}}" in text
    assert 'container_pid_mode="$(docker inspect' in text
    assert 'container_ipc_mode="$(docker inspect' in text
    assert 'test -z "$container_pid_mode"' in text
    assert 'test "$container_ipc_mode" = private' in text
    assert "container_pid_mode\" != host" not in text
    assert "container_ipc_mode\" != host" not in text
    assert 'export CUDA_VISIBLE_DEVICES="$gpu"' in text
    assert 'INTERNVLA_ISAAC_RENDER_GPU="$gpu"' in text
    assert "INTERNVLA_ISAAC_PHYSICS_GPU=0" in text
    assert "isaac_render_gpu_physical_index" in text
    assert "isaac_physics_gpu_visible_index" in text
    assert "kit_active_gpu_log_audit_required" in text
    assert 'INTERNVLA_T4_CONTAINER_NAME="$container"' in text
    assert 'nvidia-smi --query-gpu=index' in text
    assert "container_cuda_visible_devices=0" in text
    assert "host_gpu_uuid" in text
    assert "container_gpu_uuids" in text
    assert 'test "${container_gpu_uuids[0]}" = "$host_gpu_uuid"' in text
    assert "gpu_mapping.json" in text
    assert '"host_cuda_visible_devices": sys.argv[3]' in text
    assert '"host_cuda_logical_zero_gpu_uuid": sys.argv[5]' in text
    assert '"isaac_physics_logical_gpu_index": 0' in text
    assert "audit_t5_kit_gpu_log.py" in text
    assert 'kit_gpu_audit.json"' in text


def test_docker_label_templates_do_not_escape_quotes_inside_shell_single_quotes() -> None:
    text = source()
    for label in ("internnav.t5.lane", "internnav.t5.gpu"):
        assert f"'{{{{index .Config.Labels \"{label}\"}}}}'" in text
        assert f"'{{{{index .Config.Labels \\\"{label}\\\"}}}}'" not in text


def test_forbidden_compute_scan_uses_structured_process_identity_audit() -> None:
    text = source()
    scan = re.search(
        r"^scan_forbidden_compute\(\) \{\n.*?^\}", text, re.MULTILINE | re.DOTALL
    )
    assert scan is not None
    assert "t5_process_identity_audit.py" in scan.group(0)
    assert "--mode forbidden-compute" in scan.group(0)
    assert "pgrep" not in scan.group(0)
    assert 'test -f "$root/scripts/t5_process_identity_audit.py"' in text


def test_lane_has_private_profile_cache_tmp_log_result_and_root_claim() -> None:
    text = source()
    for variable in (
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "OV_CACHE_ROOT",
        "NVIDIA_SHADER_CACHE_PATH",
        "CUDA_CACHE_PATH",
        "TMPDIR",
        "OMNI_KIT_USER_CONFIG",
        "OMNI_KIT_LOG_PATH",
    ):
        assert variable in text
    assert "t5_isaac_lane_owner" in text
    assert "result root must be inside the lane deployment root" in text
    assert 'pid_ledger="$result_root/pid_ledger.jsonl"' in text


def test_namespace_identity_and_ipc_are_propagated_per_lane() -> None:
    text = source()
    assert "lane_namespace=/t5/lane_a" in text
    assert "lane_namespace=/t5/lane_b" in text
    assert 'export ROS_NAMESPACE="$lane_namespace"' in text
    assert "ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST" in text
    assert 'ROS_STATIC_PEERS="$edge_ip"' in text
    assert 'INTERNNAV_T5_ID_PREFIX="$identity_prefix"' in text
    assert 'INTERNVLA_EPISODE_ID_PREFIX="$identity_prefix"' in text
    assert 'INTERNVLA_RESET_ID_PREFIX="$identity_prefix"' in text
    assert 'INTERNVLA_REQUEST_ID_PREFIX="$identity_prefix"' in text
    assert 'INTERNVLA_T4_IPC_ALIAS_OVERRIDE="$ipc_alias"' in text
    assert 'INTERNVLA_T5_EXPECTED_LEASE_ACK="$expected_lease"' in text
    assert 'INTERNVLA_T5_LANE_IP="$edge_ip"' in text
    assert 'INTERNVLA_GO2_CONTROLLER_ENDPOINT="tcp://$edge_ip:$controller_port"' in text


def test_stale_ipc_alias_reconciliation_is_shared_and_fail_closed() -> None:
    text = source()
    names = (
        "directory_has_no_sockets",
        "ipc_target_is_allowed_t5_runtime",
        "ipc_kernel_has_no_live_socket",
        "ipc_old_target_has_no_usage",
        "replace_ipc_alias_atomically",
        "reconcile_ipc_alias",
    )
    for name in names:
        assert f"declare -f {name}" in text
    assert "docker exec -i --user admin" in text
    assert 'reconcile_ipc_alias "$1" "$2" "$3"' in text

    assert "test -r /proc/net/unix || return 1" in shell_function(
        "ipc_kernel_has_no_live_socket"
    )
    usage_guard = shell_function("ipc_old_target_has_no_usage")
    assert 'current_uid="$(id -u)" || return 1' in usage_guard
    assert 'test "$same_uid" = false || test ! -d "$process_dir" || return 1' in usage_guard
    assert 'for metadata in "$process_dir/cmdline" "$process_dir/environ"' not in usage_guard
    assert 'metadata="$process_dir/cmdline"' in usage_guard
    assert 'if test -r "$process_dir/environ"; then' in usage_guard
    assert "2>/dev/null <\"$process_dir/environ\"" in usage_guard
    functions = "\n".join(shell_function(name) for name in names)
    probe = subprocess.run(
        ["bash"],
        input=f"""\
set -euo pipefail
{functions}
# WSL1/MSYS used by this offline unit probe lacks /proc/net/unix.  The
# filesystem socket check remains real; the kernel-table helper is separately
# asserted above and runs fail-closed on the Linux deployment hosts.
ipc_kernel_has_no_live_socket() {{ return 0; }}
scratch="$(mktemp -d)"
socket_pid=""
cleanup() {{
  if test -n "$socket_pid"; then
    kill "$socket_pid" 2>/dev/null || true
    wait "$socket_pid" 2>/dev/null || true
  fi
  rm -rf "$scratch"
}}
trap cleanup EXIT
parent="$scratch/.t5-deployments"
old="$parent/old-isaac-b/runtime/t4_ipc"
current="$parent/current-isaac-b/runtime/t4_ipc"
alias_path="$scratch/internnav_t5_b_ipc"
mkdir -p "$old" "$current"

# A direct, inactive old T5 deployment is replaced atomically.
ln -s "$old" "$alias_path"
old_inode="$(stat -c '%d:%i' -- "$alias_path")"
reconcile_ipc_alias "$alias_path" "$current" "$parent"
test "$(readlink -- "$alias_path")" = "$current"
test "$(stat -c '%d:%i' -- "$alias_path")" != "$old_inode"
reconcile_ipc_alias "$alias_path" "$current" "$parent"

# A non-symlink at the fixed alias is never removed or replaced.
rm -f "$alias_path"
printf 'owned-by-someone-else\n' >"$alias_path"
set +e
reconcile_ipc_alias "$alias_path" "$current" "$parent"
rc=$?
set -e
test "$rc" != 0
grep -Fxq owned-by-someone-else "$alias_path"

# An absolute symlink outside the one-component T5 deployment boundary fails.
rm -f "$alias_path"
outside="$scratch/outside/runtime/t4_ipc"
mkdir -p "$outside"
ln -s "$outside" "$alias_path"
set +e
reconcile_ipc_alias "$alias_path" "$current" "$parent"
rc=$?
set -e
test "$rc" != 0
test "$(readlink -- "$alias_path")" = "$outside"

# A socket in an otherwise allowed old target is treated as active use.
rm -f "$alias_path"
busy_socket="$parent/socket-busy/runtime/t4_ipc"
mkdir -p "$busy_socket"
ln -s "$busy_socket" "$alias_path"
/usr/bin/python3 - "$busy_socket/live.sock" "$scratch/socket-ready" <<'PY' &
import pathlib, socket, sys, time
s = socket.socket(socket.AF_UNIX)
s.bind(sys.argv[1])
pathlib.Path(sys.argv[2]).touch()
time.sleep(30)
PY
socket_pid=$!
for _ in $(seq 1 100); do test -f "$scratch/socket-ready" && break; sleep 0.01; done
test -f "$scratch/socket-ready"
set +e
reconcile_ipc_alias "$alias_path" "$current" "$parent"
rc=$?
set -e
test "$rc" != 0
test "$(readlink -- "$alias_path")" = "$busy_socket"
kill "$socket_pid"
wait "$socket_pid" 2>/dev/null || true
socket_pid=""

# An open file under the old deployment is also an explicit use indicator.
rm -f "$alias_path"
open_busy="$parent/open-busy/runtime/t4_ipc"
mkdir -p "$open_busy"
ln -s "$open_busy" "$alias_path"
exec 7>"$open_busy/in-use"
set +e
reconcile_ipc_alias "$alias_path" "$current" "$parent"
rc=$?
set -e
test "$rc" != 0
test "$(readlink -- "$alias_path")" = "$open_busy"
exec 7>&-
""".encode(),
        capture_output=True,
        check=False,
        timeout=20,
    )
    assert probe.returncode == 0, probe.stderr.decode(errors="replace")


def test_isaac_is_the_sole_clock_source_and_has_socket_health() -> None:
    text = source()
    assert "clock_publisher_count()" in text
    assert "clock_publisher_count_is()" in text
    assert "clock_publisher_count_is 0" in text
    assert "clock_publisher_count_is 1" in text
    assert "runtime_clock_authority_ready()" in text
    assert 'test "$(clock_publisher_count)"' not in text
    assert "t5_clock_publisher.py" in text
    assert "isaac_health.sock" in text
    assert "socket.AF_UNIX" in text
    assert "write_health_state READY" in text
    assert "ready_probe.json" in text
    assert "lane health endpoint or runtime evidence is not READY/PASS" in text
    assert "write_health_state RUNNING" in text
    assert "write_health_state STOPPED" in text


def test_clock_count_uses_bounded_daemon_free_graph_probe_and_propagates_errors() -> None:
    text = source()
    clock = re.search(
        r"^clock_publisher_count\(\) \{\n.*?^\}",
        text,
        re.MULTILINE | re.DOTALL,
    )
    exact = re.search(
        r"^clock_publisher_count_is\(\) \{\n.*?^\}",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert clock is not None
    assert exact is not None
    body = clock.group(0)
    assert "t5_clock_graph_probe.py" in body
    assert "--kill-after=0.5s 8s" in body
    assert "--kill-after=0.5s 6s" in body
    assert body.index("--kill-after=0.5s 8s") < body.index("docker exec")
    assert body.index("docker exec") < body.index("--kill-after=0.5s 6s")
    assert "ros2 topic info" not in body
    assert "|| true" not in body
    graph_probe = (ROOT / "scripts/t5_clock_graph_probe.py").read_text(
        encoding="utf-8"
    )
    compile(graph_probe, "t5_clock_graph_probe.py", "exec")
    assert "get_publishers_info_by_topic" in graph_probe
    assert "rclpy.spin_once" in graph_probe
    assert graph_probe.index("rclpy.shutdown()") < graph_probe.index("print(count")

    probe = subprocess.run(
        ["bash"],
        input=f"""\
set -euo pipefail
{body}
{exact.group(0)}
ros_domain_id=75
lane_namespace=/t5/lane_a
edge_ip=10.100.100.128
container=synthetic
root=/synthetic/root
timeout() {{
  while test "$#" -gt 0; do
    case "$1" in --signal=*|--kill-after=*|[0-9]*s) shift ;; *) break ;; esac
  done
  "$@"
}}
docker() {{ printf '%s\n' "$DOCKER_CLOCK_OUTPUT"; return "$DOCKER_CLOCK_RC"; }}
DOCKER_CLOCK_RC=0
DOCKER_CLOCK_OUTPUT=0
clock_publisher_count_is 0
DOCKER_CLOCK_OUTPUT=1
clock_publisher_count_is 1
DOCKER_CLOCK_RC=17
DOCKER_CLOCK_OUTPUT=0
set +e
clock_publisher_count_is 0
clock_rc=$?
set -e
test "$clock_rc" = 17
""".encode(),
        capture_output=True,
        check=False,
        timeout=10,
    )
    assert probe.returncode == 0, probe.stderr.decode(errors="replace")

    bounded_body = body.replace("8s", "0.3s", 1).replace("6s", "0.2s", 1)
    bounded = subprocess.run(
        ["bash"],
        input=f"""\
set -euo pipefail
{bounded_body}
ros_domain_id=75
lane_namespace=/t5/lane_a
edge_ip=10.100.100.128
container=synthetic
root=/synthetic/root
fake_bin="$(mktemp -d)"
cleanup() {{ rm -rf "$fake_bin"; }}
trap cleanup EXIT
printf '#!/usr/bin/env bash\nsleep 30\n' >"$fake_bin/docker"
chmod +x "$fake_bin/docker"
PATH="$fake_bin:$PATH"
set +e
clock_publisher_count >/dev/null
probe_rc=$?
set -e
test "$probe_rc" = 124 || test "$probe_rc" = 137
""".encode(),
        capture_output=True,
        check=False,
        timeout=5,
    )
    assert bounded.returncode == 0, bounded.stderr.decode(errors="replace")


def test_ready_is_deferred_until_real_runtime_and_physical_gpu_evidence() -> None:
    text = source()
    gate_start = text.index('setsid bash "$root/scripts/run_t4_sensor_gate.sh"')
    ready = text.index("write_health_state READY")
    assert "write_health_state INFRA_READY" in text[:gate_start]
    assert gate_start < ready
    assert "kit_gpu_ready_audit.json" in text[gate_start:ready]
    assert "sensor_frames.jsonl" in text[gate_start:ready]
    assert "runtime_clock_authority_ready" in text[gate_start:ready]
    assert "test \"$model_action_count\" -gt 0" in text[gate_start:ready]
    assert "agent_application_round_trip_after_gate" in text[gate_start:ready]
    assert "controller_application_round_trip_after_gate" in text[gate_start:ready]
    assert "agent_readiness_errors_accepted_by_profile" in text[gate_start:ready]
    assert "engineering_canary_consecutive_actions_at_ready" in text[
        gate_start:ready
    ]
    assert "fresh_evaluator_dataset_count" in text[gate_start:ready]
    assert 'test "$evaluator_total_path_count" = "$dataset_episode_count"' in text[
        gate_start:ready
    ]
    assert "tcp_port_reachable" not in text
    assert "runtime_readiness_evidence.json" in text[gate_start:ready]
    assert "capture_runtime_gpu_evidence" in text[gate_start:ready]
    assert "physics_logical_zero_resolves_to_expected_uuid" in text
    assert "nvidia_compute_apps.csv" in text
    assert "nvidia_pmon.txt" in text


def test_cleanup_proves_pid_pgid_socket_clock_and_lock_zero() -> None:
    text = source()
    assert "stop_host_group evaluator" in text
    assert "stop_clock" in text
    assert "stop_host_group health" in text
    assert "group_has_runnable_member" in text
    assert "leader_is_alive" in text
    assert 'ps -eo pgid=' in text
    assert "verified_absent" in text
    assert 'find "$ipc_dir" -maxdepth 1 -type s' in text
    assert 'test ! -e "$ipc_alias" && test ! -L "$ipc_alias"' in text
    assert 'clock_publishers_after_stop' in text
    assert "lane_host_process_residuals.txt" in text
    assert "lane_container_process_residuals.txt" in text
    assert "container_top_poststop.txt" in text
    assert "clock_udp_residuals.txt" in text
    assert "mp4_residuals.txt" in text
    assert 'ss -H -lun "sport = :$clock_port"' in text
    assert '"residual_count": int(sys.argv[7])' in text
    assert 'test "$run_rc" = 0 && test "$residual" = 0 && status=PASS' in text
    assert "if ! write_health_state STOPPED" in text
    assert "if ! write_status" in text


def test_cleanup_distinguishes_leader_runnable_group_and_literal_pgid() -> None:
    text = source()
    functions: dict[str, re.Match[str]] = {}
    for name in (
        "group_is_alive",
        "group_has_runnable_member",
        "leader_is_alive",
        "stop_host_group",
    ):
        match = re.search(
            rf"^{name}\(\) \{{\n.*?^\}}", text, re.MULTILINE | re.DOTALL
        )
        assert match is not None
        functions[name] = match
    assert "!~ /^Z/" in functions["group_has_runnable_member"].group(0)
    assert "!~ /^Z/" not in functions["group_is_alive"].group(0)

    probe = subprocess.run(
        ["bash"],
        input=f"""\
set -euo pipefail
{functions['group_is_alive'].group(0)}
{functions['group_has_runnable_member'].group(0)}
{functions['leader_is_alive'].group(0)}
{functions['stop_host_group'].group(0)}
marker="$(mktemp)"
launcher_pid=""
leader_pid=""
child_pid=""
record_pid_event() {{ :; }}
sleep() {{ command sleep 0.001; }}
cleanup() {{
  test -z "$leader_pid" || kill -KILL -- "-$leader_pid" 2>/dev/null || true
  test -z "$child_pid" || kill -KILL "$child_pid" 2>/dev/null || true
  rm -f "$marker"
}}
trap cleanup EXIT
/usr/bin/python3 - "$marker" </dev/null >/dev/null 2>&1 <<'PY' &
import os
import signal
import sys
import time

session_child = os.fork()
if session_child > 0:
    os._exit(0)
os.setsid()
signal.signal(signal.SIGINT, signal.SIG_DFL)
signal.signal(signal.SIGTERM, signal.SIG_DFL)
leader = os.getpid()
child = os.fork()
if child == 0:
    time.sleep(30)
    os._exit(0)
with open(sys.argv[1], "w", encoding="utf-8") as stream:
    stream.write(f"{{leader}} {{child}}\\n")
os._exit(0)
PY
launcher_pid=$!
for _ in $(seq 1 100); do
  test -s "$marker" && break
  sleep 0.01
done
read -r leader_pid child_pid <"$marker"
wait "$launcher_pid"
! leader_is_alive "$leader_pid"
group_is_alive "$leader_pid"
group_has_runnable_member "$leader_pid"
stop_host_group synthetic "$leader_pid" /dev/null
! group_is_alive "$leader_pid"
""".encode(),
        capture_output=True,
        check=False,
        timeout=10,
    )
    assert probe.returncode == 0, probe.stderr.decode(errors="replace")

    ps_failure = subprocess.run(
        ["bash"],
        input=f"""\
set -euo pipefail
{functions['group_is_alive'].group(0)}
{functions['group_has_runnable_member'].group(0)}
{functions['stop_host_group'].group(0)}
ps() {{ return 42; }}
record_pid_event() {{ test "$3" = audit_error && printf '%s\n' "$3"; }}
set +e
event="$(stop_host_group synthetic 12345 /dev/null)"
stop_rc=$?
set -e
test "$stop_rc" = 1
test "$event" = audit_error
""".encode(),
        capture_output=True,
        check=False,
        timeout=5,
    )
    assert ps_failure.returncode == 0, ps_failure.stderr.decode(errors="replace")


def test_cleanup_normalizes_failed_clock_probe_before_writing_status() -> None:
    text = source()
    assert 'clock_count=-1' in text
    assert '"clock_probe_exit_code": int(sys.argv[10])' in text
    assert re.search(r'"\$clock_count"\s*\\?\s*"\$clock_probe_rc"', text)


def test_x86_never_owns_compute_stack_and_full_mp4_is_disabled() -> None:
    text = source()
    assert "run_t4_model_server.sh" not in text
    assert "run_t4_dgx_onboard.sh" not in text
    assert "model/Nav2/localization/map/velocity processes are forbidden on x86" in text
    assert "INTERNVLA_T5_FULL_MP4_ENCODING=0" in text
    assert "INTERNVLA_T5_VIDEO_POLICY=jsonl_keyframes_only" in text
    assert "INTERNVLA_SAVE_VIDEO=0" in text
    assert "full_mp4_encoding_allowed" in text
    assert "HF_TOKEN" not in text
    assert "hf_" not in text


def test_existing_t5_overlay_and_sensor_gate_are_reused() -> None:
    text = source()
    assert 'build_t5_isaac_remote_phase_overlay.py' in text
    assert 'run_t4_sensor_gate.sh" t4_4 "$mode" "$attempt"' in text
    assert "start_evaluator_cycle()" in text
    assert "evaluator_cycle_index=1" in text
    assert "INTERNVLA_T4_COMPLETION_FAST_PATH=1" in text
    assert "INTERNVLA_T4_R3_ENABLE_D435I=1" in text
    assert 'INTERNVLA_T4_R3_ENABLE_LIDAR="$rtf_lidar_enabled"' in text
    assert "rtf_lidar_enabled=1" in text
    assert "rtf_lidar_ray_count=1440" in text


def test_expected_count_comes_from_frozen_dataset_or_deterministic_screen() -> None:
    text = source()
    assert 'dataset_episode_count="$(python3 -' in text
    assert 'export INTERNVLA_T5_DATASET_EPISODE_COUNT="$dataset_episode_count"' in text
    assert 'export INTERNVLA_T4_EXPECTED_COUNT="$dataset_episode_count"' in text
    assert 'elif test "$dataset_episode_count" = 20' in text
    assert "D0 fixed-five is a hardware-reproduction gate" in text
    screen = text.index('    1|3)')
    fixed_five = text.index('    5)', screen)
    pilot_twenty = text.index('    10|20)', fixed_five)
    assert text.index('export INTERNVLA_T4_PHASE_OVERRIDE=pilot', screen) < fixed_five
    assert text.index('export INTERNVLA_T4_PHASE_OVERRIDE=canary', fixed_five) < pilot_twenty
    assert text.index('unset INTERNVLA_T4_EXPECTED_COUNT', fixed_five) < pilot_twenty
    assert 'export INTERNVLA_T4_PHASE_OVERRIDE=canary' in text
    assert 'unset INTERNVLA_T4_EXPECTED_COUNT' in text
    assert 'export INTERNVLA_T4_PHASE_OVERRIDE=pilot' in text
    assert (
        "T5 model evaluation requires 1/3 bounded pilot, 5 canary, or 10/20 pilot episodes"
        in text
    )


@pytest.mark.parametrize("episode_count", (1, 3))
def test_screen_counts_use_existing_completion_sim_pilot_override(
    tmp_path: Path, episode_count: int
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
        json.dump(
            {
                "episodes": [
                    {"episode_id": str(index)} for index in range(episode_count)
                ]
            },
            stream,
        )
    env = os.environ.copy()
    env.update(
        {
            "INTERNNAV_T0_PHASE": "pilot",
            "INTERNNAV_ROOT": str(internnav_root),
            "INTERNNAV_T0_DATASET_ROOT": str(dataset_root),
            "INTERNNAV_T0_RESULT_DIR": str(tmp_path / "result"),
            "INTERNNAV_SERVER_HOST": "127.0.0.1",
            "INTERNVLA_T4_EXPECTED_COUNT": str(episode_count),
            "INTERNNAV_RUNTIME_POLICY": "completion_sim",
            "INTERNNAV_SIMULATION_TARGET": "isaac",
            "INTERNNAV_T4_RESOURCE_LEASE_ACK": "dgx+isaac",
        }
    )
    command = [
        sys.executable,
        "-c",
        "import runpy,sys; runpy.run_path(sys.argv[1])",
        str(ROOT / "configs/internnav_t0/official_agent_server_cfg.py"),
    ]
    completed = subprocess.run(
        command, env=env, check=False, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    env["INTERNNAV_T0_PHASE"] = "canary"
    completed = subprocess.run(
        command, env=env, check=False, capture_output=True, text=True
    )
    assert completed.returncode != 0
    assert "bounded T4 pilot episode-count override" in completed.stderr


def test_t5_canary_phase_is_guarded_without_relaxing_t4_pilot(
    tmp_path: Path,
) -> None:
    gate = (ROOT / "scripts/run_t4_sensor_gate.sh").read_text(encoding="utf-8")
    assert 'PHASE="${INTERNVLA_T4_PHASE_OVERRIDE:-pilot}"' in gate
    assert 'case "${INTERNVLA_T5_DATASET_EPISODE_COUNT:-}" in' in gate
    assert '1|3|5)' in gate
    assert 'lane-a:lane-a|lane-b:lane-b' in gate
    assert 'fixed_dataset) ;;' in gate
    assert 'engineering_canary)' in gate
    assert 'INTERNNAV_T5_ENGINEERING_CANARY_ACK:-}" = fast-path' in gate
    output = tmp_path / "phase.sh"
    manifest = tmp_path / "manifest.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build_t4_sensor_phase_overlay.py"),
            "--source",
            str(ROOT / "scripts/run_go2_continuous_phase.sh"),
            "--output",
            str(output),
            "--manifest",
            str(manifest),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    generated = output.read_text(encoding="utf-8")
    assert 'EXPECTED_COUNT="${INTERNVLA_T4_EXPECTED_COUNT:-5}";' in generated


def test_engineering_canary_is_bounded_ready_only_and_exits_via_finalizer() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    canary = text[
        text.index('if test "$engineering_canary_sec" != 0; then') :
        text.index('set +e\nwait "$gate_pid"')
    ]
    assert 'engineering_canary_sec="${INTERNNAV_T5_ENGINEERING_CANARY_SEC:-0}"' in text
    assert "engineering_canary_sec >= 30 && engineering_canary_sec <= 600" in text
    assert 'INTERNNAV_T5_ENGINEERING_CANARY_ACK:-}" = fast-path' in text
    assert 'test "$mode" = model' in text[: text.index("execution_profile=engineering_canary")]
    assert 'leader_is_alive "$gate_pid"' in text
    assert 'runtime_clock_authority_ready' in text
    assert "sampled_model_action_count" in canary
    assert "read_agent_application_log_counts" in canary
    assert "INTERNVLA_MODEL_ACTION_OK" in text
    assert "INTERNVLA_LOCAL_IPC_STEP_ERROR" in text
    assert "INTERNVLA_ORACLE_ACTION_OK" in text
    assert "INTERNVLA_ORACLE_STEP_ERROR" in text
    assert 'INTERNNAV_T5_PRE_READY_NAV2_TIMEOUT_LIMIT:-5' in text
    assert 'safe_stop = value.get("safe_stop") is True' in text
    assert "error_text in safe_stop_timeout_errors" in text
    assert "local InternVLA step failed: 4 Nav2 goal/cmd_vel timeout" in text
    assert 'test "$canary_consecutive_action_count" -ge 2' in text
    assert "pre_ready_warnings.json" in text
    assert '"status": "WARN" if warning_count or nonwarning_count else "PASS"' in text
    assert '"engineering_canary_pre_ready_fatal_errors_absent"' in text
    assert 'test "$canary_nonwarning_error_count" != 0' in text
    assert '"clock_state_age_seconds": clock_state_age_seconds' in text
    assert '"model_action_count": int(sys.argv[6])' in text
    assert '"model_step_error_count": int(sys.argv[7])' in text
    assert '"evaluator_total_path_count": int(sys.argv[8])' in text
    assert '"new_model_step_error_count": int(sys.argv[7]) - int(sys.argv[10])' in text
    assert 'test "$((SECONDS - canary_last_action_progress_seconds))" -le 15' in text
    assert '"sensor_freshness_basis": "recent_success_action_consumed_rgbd_observation"' in text
    assert '"sensor_action_progress_age_seconds_at_ready"' in text
    tail_guard = text[
        text.index("finalize_engineering_canary_tail_guard()") :
        text.index("finalize() {")
    ]
    finalizer = text[text.index("finalize() {") : text.index("trap finalize EXIT")]
    assert "tail_model_step_error_count" in tail_guard
    assert "post_cutoff_model_step_errors_are_audit_only" in tail_guard
    assert "model_step_errors_absent_at_tail" not in tail_guard
    assert "evaluator_total_path_stable_at_tail" in tail_guard
    assert '"after_evaluator_stop": True' in tail_guard
    assert finalizer.index('stop_host_group evaluator') < finalizer.index(
        "finalize_engineering_canary_tail_guard"
    )
    assert "import json, math, socket, sys, time" in canary
    assert 'not math.isfinite(clock_state_age_seconds)' in text
    assert 'clock_state_age_seconds < 0.0' in text
    assert 'clock_state_age_seconds > 15.0' in text
    cutoff = canary.index("engineering_canary_cutoff.json")
    stop = canary.index('stop_host_group evaluator "$gate_pid"', cutoff)
    analyze = canary.index("analyze_t5_engineering_canary.py", stop)
    observed = canary.index("canary_selected_observed_seconds=")
    cutoff_parse = canary.index('canary_cutoff_state="$(read_engineering_canary', observed)
    assert canary.index('leader_is_alive "$gate_pid"', observed) < cutoff_parse
    assert canary.rindex('leader_is_alive "$gate_pid"', cutoff, stop) > cutoff
    assert cutoff < stop < analyze
    assert 'engineering_canary_timebase="${INTERNNAV_T5_ENGINEERING_CANARY_TIMEBASE:-wall}"' in text
    assert "canary_target_clock_ns" in canary
    assert '"duration_timebase": sys.argv[11]' in canary
    assert '--timebase "$engineering_canary_timebase"' in canary
    assert 'wait "$gate_pid"' not in canary[cutoff:analyze]
    assert '"episode_acceptance_claimed": False' in text
    assert "engineering_canary_samples.jsonl" in text
    assert "analyze_t5_engineering_canary.py" in text
    assert 'execution_profile = os.environ["INTERNNAV_T5_EXECUTION_PROFILE"]' in text
    assert '"evaluation_completed_naturally": fixed_dataset_completed' in text
    assert "run_rc=0\n  exit 0" in text


def test_runtime_clock_authority_uses_live_fresh_state_for_every_profile() -> None:
    helper = shell_function("runtime_clock_authority_ready")
    completed = subprocess.run(
        ["bash"],
        input=f"""\
set -euo pipefail
{helper}
leader_is_alive() {{ test "$1" = 123; }}
engineering_canary_fresh_sensor_clock_count() {{ return "$FRESH_STATE_RC"; }}
clock_publisher_count_is() {{ return 97; }}
clock_pid=123
engineering_canary_sec=60
FRESH_STATE_RC=0
runtime_clock_authority_ready
engineering_canary_sec=0
runtime_clock_authority_ready
FRESH_STATE_RC=17
set +e
runtime_clock_authority_ready
fresh_rc=$?
set -e
test "$fresh_rc" = 17
FRESH_STATE_RC=0
clock_pid=456
set +e
runtime_clock_authority_ready
leader_rc=$?
set -e
test "$leader_rc" != 0
""".encode(),
        capture_output=True,
        check=False,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr.decode(errors="replace")


def _run_canary_marker_parser(tmp_path: Path, markers: list[tuple[str, dict]]) -> list[int]:
    text = source()
    function = text.index("read_engineering_canary_application_log_state()")
    heredoc = text.index("<<'PY'", function)
    code_start = text.index("\n", heredoc) + 1
    code_end = text.index("\nPY\n", code_start)
    code = text[code_start:code_end]
    log = tmp_path / "evaluator.log"
    log.write_text(
        "total_path: 5\n"
        + "".join(
            f"prefix {marker} {json.dumps(payload, sort_keys=True)}\n"
            for marker, payload in markers
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, "-c", code, str(log), "model"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [int(value) for value in completed.stdout.split()]


def test_engineering_canary_parser_allows_only_exact_safe_stop_nav2_timeout(
    tmp_path: Path,
) -> None:
    success = (
        "INTERNVLA_MODEL_ACTION_OK",
        {"schema_version": 1, "episode_ordinal": 0, "discrete_action": 1},
    )
    exact_warning = (
        "INTERNVLA_LOCAL_IPC_STEP_ERROR",
        {
            "schema_version": 1,
            "episode_ordinal": 0,
            "safe_stop": True,
            "error": (
                "RuntimeError('local InternVLA step failed: 4 "
                "Nav2 goal/cmd_vel timeout')"
            ),
        },
    )

    assert _run_canary_marker_parser(
        tmp_path, [success, exact_warning, success, success]
    ) == [3, 1, 5, 2, 1, 0]

    goal_acceptance_warning = (
        "INTERNVLA_LOCAL_IPC_STEP_ERROR",
        {
            "schema_version": 1,
            "episode_ordinal": 0,
            "safe_stop": True,
            "error": (
                "RuntimeError('local InternVLA step failed: 4 "
                "timeout waiting for step goal acceptance')"
            ),
        },
    )
    assert _run_canary_marker_parser(
        tmp_path, [success, goal_acceptance_warning, success, success]
    ) == [3, 1, 5, 2, 1, 0]


def test_engineering_canary_parser_rejects_unstopped_or_lookalike_timeout(
    tmp_path: Path,
) -> None:
    exact_text = (
        "RuntimeError('local InternVLA step failed: 4 Nav2 goal/cmd_vel timeout')"
    )
    markers = [
        (
            "INTERNVLA_LOCAL_IPC_STEP_ERROR",
            {"schema_version": 1, "safe_stop": False, "error": exact_text},
        ),
        (
            "INTERNVLA_LOCAL_IPC_STEP_ERROR",
            {
                "schema_version": 1,
                "safe_stop": True,
                "error": exact_text + " extra",
            },
        ),
    ]

    assert _run_canary_marker_parser(tmp_path, markers) == [0, 2, 5, 0, 0, 2]


def _run_canary_freshness_probe(
    tmp_path: Path, *, sensor_age_seconds: float
) -> subprocess.CompletedProcess[str]:
    text = source()
    function = text.index("engineering_canary_fresh_sensor_clock_count()")
    heredoc = text.index("<<'PY'", function)
    code_start = text.index("\n", heredoc) + 1
    code_end = text.index("\nPY\n", code_start)
    code = text[code_start:code_end]
    now = time.time()
    sensor = tmp_path / "sensor.json"
    clock = tmp_path / "clock.json"
    sensor.write_text(
        json.dumps(
            {
                "status": "PASS",
                "evidence": "real_sim_depth_frame_acknowledged_by_dgx_controller",
                "wall_unix": now - sensor_age_seconds,
            }
        ),
        encoding="utf-8",
    )
    clock.write_text(
        json.dumps(
            {
                "status": "RUNNING",
                "last_clock_ns": 10,
                "received_step_count": 4,
                "publish_count": 4,
                "regression_count": 0,
                "invalid_count": 0,
                "updated_unix": now,
            }
        ),
        encoding="utf-8",
    )
    return subprocess.run(
        [sys.executable, "-c", code, str(sensor), str(clock)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_engineering_canary_first_only_sensor_ack_proves_source_not_live_age(
    tmp_path: Path,
) -> None:
    fresh = _run_canary_freshness_probe(tmp_path, sensor_age_seconds=1.0)
    assert fresh.returncode == 0
    assert fresh.stdout.strip() == "4"

    stale = _run_canary_freshness_probe(tmp_path, sensor_age_seconds=15.1)
    assert stale.returncode == 0
    assert stale.stdout.strip() == "4"


def test_engineering_canary_ready_rewait_recovers_after_allowed_warning(
    tmp_path: Path,
) -> None:
    text = source()
    helper_start = text.index("wait_for_engineering_canary_ready_snapshot()")
    helper_end = text.index("\ndirectory_has_no_sockets()", helper_start)
    helper = text[helper_start:helper_end]
    stubs = r'''
leader_is_alive() { return 0; }
read_engineering_canary_application_log_state() {
  case "${model_action_count:-0}" in
    0) printf '%s\n' '2 1 5 0 1 0' ;;
    2) printf '%s\n' '3 1 5 1 1 0' ;;
    *) printf '%s\n' '4 1 5 2 1 0' ;;
  esac
}
engineering_canary_fresh_sensor_clock_count() { printf '%s\n' 101; }
sleep() { :; }
export -f read_engineering_canary_application_log_state
export -f engineering_canary_fresh_sensor_clock_count
'''
    invocation = r'''
runtime_ready_deadline=100
gate_pid=1
result_root=/tmp/canary-rewait-test
mode=model
dataset_episode_count=5
engineering_canary_pre_ready_nav2_timeout_limit=5
canary_previous_action_count=0
canary_last_action_progress_seconds=-1
canary_previous_clock_received=100
canary_clock_progress_observed=1
model_action_count=0
wait_for_engineering_canary_ready_snapshot
test "$model_action_count" = 4
test "$model_step_error_count" = 1
test "$canary_consecutive_action_count" = 2
test "$canary_nav2_timeout_warning_count" = 1
test "$canary_nonwarning_error_count" = 0
test "$ready_model_action_count" = 4
test "$ready_model_step_error_count" = 1
    test "$canary_sensor_action_progress_age_sec" -le 15
'''
    harness = stubs + helper + invocation
    harness_path = tmp_path / "canary_rewait.sh"
    with harness_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(harness)
    if sys.platform == "win32":
        wsl_path = subprocess.check_output(
            ["wsl", "wslpath", "-a", str(harness_path)], text=True
        ).strip()
        command = ["wsl", "bash", wsl_path]
    else:
        command = ["bash", str(harness_path)]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr

    capture = text.index("capture_runtime_gpu_evidence", text.index("STARTING_SIM"))
    evidence = text.index("runtime_readiness_evidence.json", capture)
    ready = text.index("write_health_state READY", evidence)
    assert text.count(
        "wait_for_engineering_canary_ready_snapshot", capture, ready
    ) == 2
