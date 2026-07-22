from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from scripts.audit_t5_hf_token_process_scope import audit


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_t5_dgx_lane.sh"
CANDIDATE = ROOT / "configs/internnav_t5/candidates/recovery_a.json"


def source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_lane_entrypoint_has_exact_symmetric_assignments() -> None:
    text = source()
    assert "expected_user=railgun" in text
    assert "10.100.100.128" in text
    assert "ros_domain_id=75" in text
    assert "controller_port=25137" in text
    assert "model_client_port=25139" in text
    assert "oracle_port=25140" in text
    assert "identity_prefix='a::'" in text
    assert "expected_user=rail" in text
    assert "10.100.120.116" in text
    assert "ros_domain_id=76" in text
    assert "controller_port=25138" in text
    assert "model_client_port=25239" in text
    assert "oracle_port=25240" in text
    assert "identity_prefix='b::'" in text


def test_lane_runs_model_navigation_and_evaluator_on_same_dgx() -> None:
    text = source()
    assert 'bash "$root/scripts/run_t4_model_server.sh"' in text
    assert 'bash "$root/scripts/run_t4_dgx_onboard.sh"' in text
    assert "INTERNVLA_ONBOARD_MODEL_OWNER=local_dgx" in text
    assert "INTERNVLA_ONBOARD_USE_SIM_TIME=true" in text
    assert "INTERNVLA_T4_ALLOW_COMMAND_POSE_ANCHOR_FALLBACK=1" in text
    assert "internvla_t4_client" in text
    assert "-p sensor_future_tolerance_sec:=0.55" in text
    assert "internvla_nav2_oracle_bridge" in text
    assert 'grep -Fxq "$lane_namespace/internvla_model_node"' in text
    assert 'grep -Fxq "$lane_namespace/controller_server"' in text
    assert '"model_and_nav2_colocated": True' in text


def test_recovery_a_is_opt_in_and_reuses_frozen_t4_materializer(
    tmp_path: Path,
) -> None:
    text = source()
    assert "if [[ -v INTERNNAV_T5_CANDIDATE_PROFILE ]]" in text
    assert "candidate_profile=baseline" in text
    assert "baseline|recovery_a" in text
    assert "recovery_env=()" in text
    common_start = text.index("common_env=(")
    common_end = text.index("\n)", common_start)
    assert "INTERNVLA_T4_ENABLE_RECOVERY" not in text[common_start:common_end]
    assert text.count('env "${common_env[@]}" "${recovery_env[@]}"') == 2
    assert 'INTERNVLA_T4_ONBOARD_PROFILE="$onboard_profile"' in text
    assert 'python3 "$root/scripts/t4_recovery_runtime.py"' in text
    assert 'INTERNVLA_T4_ENABLE_SCHEDULED_REFRESH=1' in text
    assert 'INTERNVLA_T4_MAXIMUM_SCHEDULED_REFRESHES=1' in text
    assert (
        'INTERNVLA_T4_RECOVERY_SCAN_YAW_RAD="${recovery_override_fields[0]}"'
        in text
    )
    assert (
        'INTERNVLA_T4_REPLAN_DEADLINE_SEC="${recovery_override_fields[1]}"'
        in text
    )

    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    assert candidate["candidate_profile"] == "recovery_a"
    assert candidate["runtime"]["t4_profile"] == (
        "configs/completion_sim/recovery/profile_a.json"
    )
    assert candidate["baseline_contract"]["recovery_enabled_when_unset"] is False
    assert candidate["t5_completion_sim_overrides"] == {
        "recovery_scan_yaw_rad": 0.3,
        "replan_deadline_sec": 65.0,
    }
    assert (
        candidate["baseline_contract"][
            "t4_profile_and_canonical_sha256_unchanged"
        ]
        is True
    )
    assert 'manifest["effective_t5_completion_sim_overrides"] = expected' in text

    output = tmp_path / "nav2_recovery.yaml"
    manifest = tmp_path / "recovery_runtime_manifest.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/t4_recovery_runtime.py"),
            "--profile",
            str(ROOT / candidate["runtime"]["t4_profile"]),
            "--nav2-input",
            str(ROOT / "configs/internnav_t5/nav2_static_lidar.yaml"),
            "--nav2-output",
            str(output),
            "--manifest",
            str(manifest),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    materialized = json.loads(manifest.read_text(encoding="utf-8"))
    assert materialized["status"] == "RECOVERY_RUNTIME_READY"
    assert materialized["profile_id"] == "A"
    assert materialized["parameters"]["progress_horizon_sec"] == 4.0
    assert materialized["parameters"]["maximum_recoveries_per_episode"] == 2
    assert materialized["parameters"]["maximum_recovery_duration_sec"] == 35.0
    assert candidate["runtime"]["maximum_scheduled_refreshes_per_episode"] == 1


def test_lane_is_fail_closed_and_never_launches_isaac() -> None:
    text = source()
    assert 'INTERNNAV_T5_RESOURCE_LEASE_ACK:-}" = "$expected_lease"' in text
    assert "Isaac/Kit is forbidden on a T5 DGX lane" in text
    assert 'test "${CUDA_VISIBLE_DEVICES:-}" = 0' in text
    assert "isaac-sim" not in "\n".join(
        line for line in text.splitlines() if line.lstrip().startswith(("setsid", "exec"))
    )
    assert '"isaac_allowed_on_this_host": False' in text
    assert re.search(r"hf_[A-Za-z0-9]{20,}", text) is None
    assert "INTERNVLA_HF_TOKEN_FD" in text
    assert 'export HF_TOKEN="$hf_token"' in text
    assert "unset INTERNVLA_HF_TOKEN_FD HF_TOKEN HUGGING_FACE_HUB_TOKEN" in text
    assert 'exec {hf_process_audit_fd}<<<"$hf_token"' in text
    assert "audit_t5_hf_token_process_scope.py" in text
    assert '--secret-fd "$hf_process_audit_fd"' in text
    assert '--model-pid "$model_pid" --model-pgid "$model_pgid"' in text
    assert '--parent-pid "$$" --onboard-pid "$onboard_pid"' in text
    assert "hf_token_process_audit.json" in text


def test_lane_has_independent_pid_port_health_and_stop_contracts() -> None:
    text = source()
    assert 'pid_ledger="$result_dir/pid_ledger.jsonl"' in text
    assert "record_pid_event model" in text
    assert "record_pid_event onboard" in text
    assert "record_pid_event evaluator" in text
    assert 'health_endpoint": f"ros2://domain-' in text
    assert 'test ! -f "$result_dir/stop.request"' in text
    assert 'for port in "$controller_port" "$model_client_port" "$oracle_port"' in text
    assert "verified_absent" in text
    assert "residual_count" in text
    assert (
        "grep -Fxq '/internvla/health [internvla_ros2_msgs/srv/Health]'" in text
    )
    assert (
        "grep -Fxq '/internvla/step [internvla_ros2_msgs/action/Step]'" in text
    )
    assert 'grep -Fxq /internvla/health ' not in text
    assert 'grep -Fxq /internvla/step ' not in text


def test_lane_retries_daemon_free_ros_graph_discovery_before_freezing_snapshot() -> None:
    text = source()
    helper = re.search(
        r"^query_graph_until_present\(\) \{\n(?P<body>.*?)^\}",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert helper is not None
    assert "for attempt in $(seq 1 15)" in helper.group("body")
    assert 'grep -Fxq "$required_one" "$candidate"' in helper.group("body")
    assert 'grep -Fxq "$required_two" "$candidate"' in helper.group("body")
    assert 'leader_is_alive "$model_pid"' in helper.group("body")
    assert 'leader_is_alive "$onboard_pid"' in helper.group("body")
    assert 'leader_is_alive "$client_pid"' in helper.group("body")
    assert "ros2 node list --no-daemon --spin-time 2.0" in text
    assert "ros2 service list -t --no-daemon --spin-time 2.0" in text
    # Jazzy's ros2action list verb does not register NodeStrategy CLI options;
    # its existing command is still protected by the outer bounded retry.
    assert "ros2 action list -t\n" in text
    assert "ros2 action list -t --no-daemon" not in text
    assert 'ros_graph_discovery.log' in text
    assert "export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST" in text
    assert 'export ROS_STATIC_PEERS="$isaac_ip"' in text

    probe = subprocess.run(
        ["bash"],
        input=f"""\
set -euo pipefail
{helper.group(0)}
result_dir="$(mktemp -d)"
mkdir "$result_dir/logs"
graph_discovery_log="$result_dir/logs/ros_graph_discovery.log"
: >"$graph_discovery_log"
model_pid=$$
onboard_pid=$$
client_pid=$$
trap 'rm -rf "$result_dir"' EXIT
leader_is_alive() {{ return 0; }}
sleep() {{ :; }}
fake_graph() {{
  count_file="$result_dir/count"
  count=0
  test ! -f "$count_file" || count="$(cat "$count_file")"
  count=$((count + 1))
  printf '%s\n' "$count" >"$count_file"
  printf '%s\n' /t5/lane_a/internvla_model_node
  test "$count" -lt 2 || printf '%s\n' /t5/lane_a/controller_server
}}
query_graph_until_present "$result_dir/nodes" nodes \
  /t5/lane_a/internvla_model_node /t5/lane_a/controller_server fake_graph
grep -Fxq /t5/lane_a/internvla_model_node "$result_dir/nodes"
grep -Fxq /t5/lane_a/controller_server "$result_dir/nodes"
grep -Fq $'nodes\tattempt=1\tRETRY' "$result_dir/logs/ros_graph_discovery.log"
grep -Fq $'nodes\tattempt=2\tPASS' "$result_dir/logs/ros_graph_discovery.log"
""".encode(),
        check=False,
        capture_output=True,
        timeout=10,
    )
    assert probe.returncode == 0, probe.stderr.decode(errors="replace")


def test_lane_liveness_follows_leader_but_cleanup_drains_process_group() -> None:
    text = source()
    leader = re.search(
        r"^leader_is_alive\(\) \{\n(?P<body>.*?)^\}", text, re.MULTILINE | re.DOTALL
    )
    group = re.search(
        r"^group_is_alive\(\) \{\n(?P<body>.*?)^\}", text, re.MULTILINE | re.DOTALL
    )
    runnable_group = re.search(
        r"^group_has_runnable_member\(\) \{\n(?P<body>.*?)^\}",
        text,
        re.MULTILINE | re.DOTALL,
    )
    stop = re.search(
        r"^stop_group\(\) \{\n(?P<body>.*?)^\}", text, re.MULTILINE | re.DOTALL
    )
    bounded_wait = re.search(
        r"^bounded_wait_child\(\) \{\n(?P<body>.*?)^\}",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert leader is not None
    assert group is not None
    assert runnable_group is not None
    assert stop is not None
    assert bounded_wait is not None
    assert 'ps -o stat=,pgid= -p "$pid"' in leader.group("body")
    assert '[[ "$state" != Z* && "$pgid" = "$pid" ]]' in leader.group("body")
    assert 'kill -0 -- "-$pid"' not in leader.group("body")
    assert 'ps -eo pgid=' in group.group("body")
    assert '$1 == expected' in group.group("body")
    assert '!~ /^Z/' not in group.group("body")
    assert 'ps -eo stat=,pgid=' in runnable_group.group("body")
    assert '$2 == expected && $1 !~ /^Z/' in runnable_group.group("body")
    assert "group_is_alive" in stop.group("body")
    assert "group_has_runnable_member" in stop.group("body")
    assert 'for _ in $(seq 1 50)' in stop.group("body")
    assert 'bounded_wait_child "$pid" 5' in stop.group("body")

    # A surviving daemon in the old PGID must not satisfy any runtime gate.
    # Group liveness is reserved exclusively for bounded cleanup.
    without_helpers_or_cleanup = text
    for match in (leader, group, runnable_group, stop):
        without_helpers_or_cleanup = without_helpers_or_cleanup.replace(match.group(0), "")
    assert "group_is_alive" not in without_helpers_or_cleanup

    assert (
        'leader_is_alive "$onboard_pid" || '
        '{ shutdown_reason=onboard_startup_exit; exit 1; }' in text
    )
    assert (
        'leader_is_alive "$onboard_pid" || '
        '{ shutdown_reason=onboard_exit; exit 1; }' in text
    )
    assert (
        'leader_is_alive "$model_pid" || '
        '{ shutdown_reason=model_exit; exit 1; }' in text
    )
    assert 'if ! leader_is_alive "$client_pid"; then' in text
    assert 'wait "$client_pid" || client_rc=$?' in text
    assert 'shutdown_reason="evaluator_unexpected_exit:rc_$client_rc"' in text
    assert 'shutdown_reason="evaluator_exit:rc_$client_rc"' in text
    completion = text.split('if ! leader_is_alive "$client_pid"; then', 1)[1]
    assert "handle_evaluator_exit || exit 1" in completion
    assert completion.index("handle_evaluator_exit || exit 1") < completion.index(
        "exit 0"
    )

    # Reproduce the incident: the session leader exits after leaving an
    # orphaned child in its PGID.  The runtime predicate must fail immediately
    # while the cleanup predicate must still find the group to drain.
    probe = subprocess.run(
        ["bash"],
        input=f"""\
set -euo pipefail
{group.group(0)}
{runnable_group.group(0)}
    {leader.group(0)}
    {bounded_wait.group(0)}
    {stop.group(0)}
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
printf 'leader_absent_group_present\n'
stop_group onboard "$leader_pid" /dev/null
! group_is_alive "$leader_pid"
printf 'orphan_group_drained\n'
        """.encode(),
        check=False,
        capture_output=True,
        timeout=10,
    )
    assert probe.returncode == 0, probe.stderr.decode(errors="replace")


def test_onboard_shutdown_uses_term_with_full_child_cleanup_budget() -> None:
    text = source()
    stop = re.search(
        r"^stop_group\(\) \{\n(?P<body>.*?)^\}", text, re.MULTILINE | re.DOTALL
    )
    assert stop is not None
    body = stop.group("body")
    onboard = body.split('if test "$component" = onboard; then', 1)[1].split(
        "  else", 1
    )[0]
    assert 'kill -TERM -- "-$pid"' in onboard
    assert 'for _ in $(seq 1 700)' in onboard
    assert 'kill -INT -- "-$pid"' not in onboard
    non_onboard = body.split("  else", 1)[1]
    assert 'kill -INT -- "-$pid"' in non_onboard
    assert 'for _ in $(seq 1 600)' in non_onboard
    assert 'for _ in $(seq 1 200)' in non_onboard


def test_child_wait_is_bounded_and_onboard_cleanup_precedes_evaluator() -> None:
    text = source()
    helper = re.search(
        r"^bounded_wait_child\(\) \{\n(?P<body>.*?)^\}",
        text,
        re.MULTILINE | re.DOTALL,
    )
    cleanup = re.search(
        r"^cleanup\(\) \{\n(?P<body>.*?)^\}", text, re.MULTILINE | re.DOTALL
    )
    assert helper is not None
    assert cleanup is not None
    body = cleanup.group("body")
    immediate = body.index('for pid in "$onboard_pid" "$client_pid"; do')
    assert body.index("stop_group onboard") < body.index("stop_group evaluator")
    assert body.index("stop_group evaluator") < body.index("stop_group model")
    assert immediate < body.index("stop_group onboard")
    assert 'kill -TERM -- "-$pid"' in body[immediate : body.index("stop_group onboard")]

    probe = subprocess.run(
        ["bash"],
        input=f"""\
set -euo pipefail
{helper.group(0)}
sleep 30 & child=$!
started=$SECONDS
bounded_wait_child "$child" 1
elapsed=$((SECONDS - started))
test "$elapsed" -le 3
kill -0 "$child"
kill -KILL "$child" 2>/dev/null || true
wait "$child" 2>/dev/null || true
""".encode(),
        capture_output=True,
        check=False,
        timeout=8,
    )
    assert probe.returncode == 0, probe.stderr.decode(errors="replace")


def test_presignalled_onboard_gets_no_second_term_during_cleanup() -> None:
    text = source()
    names = ("group_is_alive", "group_has_runnable_member", "bounded_wait_child", "stop_group")
    helpers = []
    for name in names:
        match = re.search(
            rf"^{name}\(\) \{{\n.*?^\}}", text, re.MULTILINE | re.DOTALL
        )
        assert match is not None, name
        helpers.append(match.group(0))

    probe = subprocess.run(
        ["bash"],
        input=f"""\
set -euo pipefail
{chr(10).join(helpers)}
scratch="$(mktemp -d)"
leader=""
cleanup_probe() {{
  test -z "$leader" || kill -KILL -- "-$leader" 2>/dev/null || true
  rm -rf "$scratch"
}}
trap cleanup_probe EXIT
record_pid_event() {{ :; }}
sleep() {{ command sleep 0.005; }}
setsid bash -c '
  ready="$1"; entered="$2"; completed="$3"
  cleanup_child() {{
    trap - TERM
    touch "$entered"
    command sleep 0.2
    touch "$completed"
    exit 143
  }}
  trap cleanup_child TERM
  touch "$ready"
  while :; do command sleep 0.05; done
' presignalled "$scratch/ready" "$scratch/entered" "$scratch/completed" &
leader=$!
for _ in $(seq 1 200); do test -f "$scratch/ready" && break; command sleep 0.01; done
test -f "$scratch/ready"
kill -TERM -- "-$leader"
stop_group onboard "$leader" /dev/null 1
test -f "$scratch/entered"
test -f "$scratch/completed"
! group_is_alive "$leader"
""".encode(),
        capture_output=True,
        check=False,
        timeout=10,
    )
    assert probe.returncode == 0, probe.stderr.decode(errors="replace")


def test_model_evaluator_cannot_finish_before_coordinator_stop() -> None:
    text = source()
    handler = re.search(
        r"^handle_evaluator_exit\(\) \{\n(?P<body>.*?)^\}",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert handler is not None

    probe = subprocess.run(
        ["bash"],
        input=f"""\
set -euo pipefail
{handler.group(0)}
record_pid_event() {{ :; }}
result_dir=/tmp/not-used
shutdown_reason=unset
mode=model
(exit 0) & client_pid=$!
! handle_evaluator_exit
test "$shutdown_reason" = evaluator_unexpected_exit:rc_0
(exit 17) & client_pid=$!
! handle_evaluator_exit
test "$shutdown_reason" = evaluator_unexpected_exit:rc_17
mode=oracle
(exit 0) & client_pid=$!
handle_evaluator_exit
test "$shutdown_reason" = evaluator_completed
(exit 17) & client_pid=$!
! handle_evaluator_exit
test "$shutdown_reason" = evaluator_exit:rc_17
""".encode(),
        check=False,
        capture_output=True,
        timeout=10,
    )
    assert probe.returncode == 0, probe.stderr.decode(errors="replace")


def test_d0_coordinator_authorizes_stop_only_after_clean_x86_exit() -> None:
    coordinator = (
        ROOT / "coordination" / "run_t5_d0_lane_online.sh"
    ).read_text(encoding="utf-8")
    wait_x86 = coordinator.index('wait "$x86_ssh_pid"; x86_rc=$?')
    require_clean = coordinator.index('test "$x86_rc" = 0', wait_x86)
    request_stop = coordinator.index(
        'remote "$dgx_target" "touch \'$dgx_run/stop.request\'"', wait_x86
    )
    assert wait_x86 < require_clean < request_stop
    assert "fixed_five_episode_coverage.json" in coordinator


def test_clean_sigterm_143_is_normalized_only_after_zero_residuals() -> None:
    text = source()
    pass_condition = (
        'test "$online_ready" = 1 && test "$residual" = 0 && \\\n'
        '      { test "$rc" = 0 || test "$rc" = 130 || test "$rc" = 143; }'
    )
    assert pass_condition in text
    assert "normalized_rc=0" in text
    assert 'trap \'shutdown_reason=signal_term; exit 143\' TERM HUP' in text


def test_lane_identity_and_namespace_are_propagated_to_children() -> None:
    text = source()
    assert "lane_namespace=/t5/lane_a" in text
    assert "lane_namespace=/t5/lane_b" in text
    assert 'INTERNNAV_T5_ID_PREFIX="$identity_prefix"' in text
    assert 'INTERNVLA_EPISODE_ID_PREFIX="$identity_prefix"' in text
    assert 'INTERNVLA_RESET_ID_PREFIX="$identity_prefix"' in text
    assert 'INTERNVLA_REQUEST_ID_PREFIX="$identity_prefix"' in text
    assert 'ROS_NAMESPACE="$lane_namespace"' in text
    assert "ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST" in text
    assert 'INTERNVLA_ONBOARD_NAMESPACE="$lane_namespace"' in text
    onboard = (ROOT / "scripts/run_t4_dgx_onboard.sh").read_text(encoding="utf-8")
    lifecycle = (ROOT / "scripts/check_t4_nav2_lifecycle.py").read_text(
        encoding="utf-8"
    )
    assert "NAV2_LAUNCH_FILE=bringup_launch.py" in onboard
    assert 'NAV2_NAMESPACE_ARGS=("namespace:=$NODE_NAMESPACE" use_namespace:=True' in onboard
    assert "use_localization:=False slam:=False" in onboard
    assert '--namespace "$NODE_NAMESPACE"' in onboard
    assert 'resolved_nodes = {name: f"{namespace}{name}" for name in NODES}' in lifecycle
    assert "parameter_audit.json" in text
    assert "discovery_timeout_sec 1200.0" in text
    assert 'if test "$mode" = model; then' in text
    assert '"$lane_namespace/internvla_nav2_oracle_bridge"' in text
    assert "resolution_timeout_sec 10.0" in text
    assert "publish_observation_pose False" in text
    assert 'ROS_STATIC_PEERS="$isaac_ip"' in text
    assert '-r __ns:="$lane_namespace"' in text
    assert text.count("-r /tf:=tf -r /tf_static:=tf_static") >= 2
    assert "nav2_data_plane_evaluator_ready.json" in text
    assert '"tf_chain": data_plane.get("transforms_pass") is True' in text
    assert '"root_tf_isolated": data_plane.get("root_tf_isolated") is True' in text


def test_parameter_audit_retries_transient_discovery_but_stays_exact() -> None:
    text = source()
    helper = re.search(
        r"^audit_parameter\(\) \{\n(?P<body>.*?)^\}",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert helper is not None
    body = helper.group("body")
    assert "for attempt in $(seq 1 5)" in body
    assert "timeout --signal=TERM --kill-after=0.5s 6s" in body
    assert 'test "$actual" = "$expected"' in body
    assert "UNAVAILABLE" in body


def test_lane_records_static_dds_peer_contract() -> None:
    text = source()
    assert text.count('"automatic_discovery_range": "LOCALHOST"') == 2
    assert '"static_peers": [sys.argv[7]]' in text
    assert '"static_peers": [contract["expected_isaac_peer_ip"]]' in text


def _fake_process(
    root: Path,
    pid: int,
    ppid: int,
    pgid: int,
    environ: bytes,
    *,
    uid: int = 1000,
) -> None:
    process = root / str(pid)
    process.mkdir()
    (process / "status").write_text(
        f"Name:\tp{pid}\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\n",
        encoding="utf-8",
    )
    (process / "stat").write_text(
        f"{pid} (p{pid}) S {ppid} {pgid} 0 0 0\n", encoding="utf-8"
    )
    (process / "comm").write_text(f"p{pid}\n", encoding="utf-8")
    (process / "environ").write_bytes(environ)


def _make_environ_unreadable(root: Path, pid: int) -> None:
    path = root / str(pid) / "environ"
    path.unlink()
    # read_bytes() on this directory deterministically raises OSError on the
    # offline test platforms, standing in for a Linux non-dumpable /proc file.
    path.mkdir()


def test_process_secret_audit_allows_only_model_group_and_subtree(
    tmp_path: Path,
) -> None:
    secret = b"hf_test_process_scope_secret_123456789"
    _fake_process(tmp_path, 100, 1, 100, b"ROLE=parent\0")
    _fake_process(tmp_path, 200, 100, 200, b"HF_TOKEN=" + secret + b"\0")
    _fake_process(
        tmp_path, 201, 200, 200, b"HUGGING_FACE_HUB_TOKEN=" + secret + b"\0"
    )
    _fake_process(tmp_path, 300, 100, 300, b"ROLE=onboard\0")
    _fake_process(tmp_path, 400, 100, 400, b"ROLE=evaluator\0")

    result = audit(
        proc_root=tmp_path,
        secret=secret,
        model_pid=200,
        model_pgid=200,
        parent_pid=100,
        onboard_pid=300,
        evaluator_pid=400,
        effective_uid=1000,
    )
    assert result["status"] == "PASS"
    assert result["exact_secret_match_count"] == 2
    assert result["allowed_model_match_count"] == 2
    assert result["disallowed_match_count"] == 0
    assert secret.decode() not in json.dumps(result)

    _fake_process(tmp_path, 500, 100, 500, b"LEAK=" + secret + b"\0")
    leaked = audit(
        proc_root=tmp_path,
        secret=secret,
        model_pid=200,
        model_pgid=200,
        parent_pid=100,
        onboard_pid=300,
        evaluator_pid=400,
        effective_uid=1000,
    )
    assert leaked["status"] == "FAIL"
    assert leaked["disallowed_match_count"] == 1
    assert leaked["checks"]["other_processes_have_zero_matches"] is False


def test_process_secret_audit_bounds_non_dumpable_proc_coverage(
    tmp_path: Path,
) -> None:
    secret = b"hf_test_process_scope_secret_123456789"
    _fake_process(tmp_path, 100, 1, 100, b"ROLE=parent\0")
    _fake_process(tmp_path, 200, 100, 200, b"HF_TOKEN=" + secret + b"\0")
    _fake_process(tmp_path, 201, 200, 200, b"ROLE=model-child\0")
    _make_environ_unreadable(tmp_path, 201)
    _fake_process(tmp_path, 300, 100, 300, b"ROLE=onboard\0")
    _fake_process(tmp_path, 400, 100, 400, b"ROLE=evaluator\0")
    _fake_process(tmp_path, 600, 1, 600, b"ROLE=unrelated-infrastructure\0")
    _make_environ_unreadable(tmp_path, 600)

    bounded = audit(
        proc_root=tmp_path,
        secret=secret,
        model_pid=200,
        model_pgid=200,
        parent_pid=100,
        onboard_pid=300,
        evaluator_pid=400,
        effective_uid=1000,
    )
    assert bounded["status"] == "PASS"
    assert bounded["unreadable_allowed_model_scope_count"] == 1
    assert bounded["unreadable_outside_secret_inheritance_tree_count"] == 1
    assert bounded["unreadable_unapproved_secret_inheriting_scope_count"] == 0

    _fake_process(
        tmp_path,
        700,
        100,
        700,
        b"ROLE=unapproved-descendant\0",
        uid=2000,
    )
    _make_environ_unreadable(tmp_path, 700)
    unprovable = audit(
        proc_root=tmp_path,
        secret=secret,
        model_pid=200,
        model_pgid=200,
        parent_pid=100,
        onboard_pid=300,
        evaluator_pid=400,
        effective_uid=1000,
    )
    assert unprovable["status"] == "FAIL"
    assert unprovable["unreadable_unapproved_secret_inheriting_scope_count"] == 1
    assert (
        unprovable["checks"][
            "all_secret_inheriting_environs_readable_or_allowed_model_scope"
        ]
        is False
    )


def test_process_secret_audit_fails_closed_on_unknown_or_wrong_group_ancestry(
    tmp_path: Path,
) -> None:
    secret = b"hf_test_process_scope_secret_123456789"
    _fake_process(tmp_path, 100, 1, 100, b"ROLE=parent\0")
    _fake_process(tmp_path, 200, 100, 200, b"HF_TOKEN=" + secret + b"\0")
    _fake_process(tmp_path, 300, 100, 300, b"ROLE=onboard\0")
    _fake_process(tmp_path, 400, 100, 400, b"ROLE=evaluator\0")
    _fake_process(tmp_path, 201, 200, 201, b"ROLE=wrong-model-pgid\0")
    _make_environ_unreadable(tmp_path, 201)
    _fake_process(tmp_path, 800, 799, 800, b"ROLE=missing-ancestor\0")
    _make_environ_unreadable(tmp_path, 800)

    result = audit(
        proc_root=tmp_path,
        secret=secret,
        model_pid=200,
        model_pgid=200,
        parent_pid=100,
        onboard_pid=300,
        evaluator_pid=400,
        effective_uid=1000,
    )
    assert result["status"] == "FAIL"
    assert result["unreadable_unapproved_secret_inheriting_scope_count"] == 2
    by_pid = {row["pid"]: row for row in result["unreadable_processes"]}
    assert by_pid[201]["model_subtree"] is True
    assert by_pid[201]["model_process_group"] is False
    assert by_pid[800]["parent_subtree"] is None
