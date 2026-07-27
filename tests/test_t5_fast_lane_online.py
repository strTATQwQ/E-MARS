import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "coordination" / "run_t5_fast_lane_online.sh"


def source() -> str:
    return RUNNER.read_text(encoding="utf-8")


def named_heredoc(marker: str) -> str:
    lines = source().splitlines()
    start = next(index for index, line in enumerate(lines) if f"<<'{marker}'" in line)
    end = next(index for index in range(start + 1, len(lines)) if lines[index] == marker)
    return "\n".join(lines[start + 1 : end]) + "\n"


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
            f"{RUNNER}:heredoc:{index + 1}",
            "exec",
        )
        compiled += 1
        index = end + 1
    assert compiled == 14
    compile(
        named_heredoc("REMOTE_SUPERVISOR_CONTROL"),
        f"{RUNNER}:REMOTE_SUPERVISOR_CONTROL",
        "exec",
    )


@pytest.mark.parametrize(
    ("lane", "profile", "run_id", "code_sha", "result_root"),
    (
        ("c", "canary60", "t5fast01", "a" * 40,
         "results/internnav_t5/fast-lane-c-canary60-t5fast01"),
        ("a", "unknown", "t5fast01", "a" * 40,
         "results/internnav_t5/fast-lane-a-unknown-t5fast01"),
        ("a", "fixed5", "short", "a" * 40,
         "results/internnav_t5/fast-lane-a-fixed5-short"),
        ("a", "fixed5", "t5fast01", "not-a-commit",
         "results/internnav_t5/fast-lane-a-fixed5-t5fast01"),
        ("a", "fixed5", "t5fast01", "a" * 40,
         "results/internnav_t5/not-the-bound-result"),
    ),
)
def test_invalid_identity_contract_exits_before_any_resource_action(
    lane: str, profile: str, run_id: str, code_sha: str, result_root: str
) -> None:
    completed = subprocess.run(
        [
            "bash", RUNNER.relative_to(ROOT).as_posix(), lane, profile, run_id, code_sha,
            "results/internnav_t5/d0-0-prepare-t5d0020260719t000000",
            result_root,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 64
    assert "ACQUIRED" not in completed.stdout + completed.stderr


@pytest.mark.parametrize("candidate_profile", ("", "recovery-A", "strict"))
def test_invalid_candidate_profile_exits_before_any_resource_action(
    candidate_profile: str,
) -> None:
    env = dict(os.environ)
    env["INTERNNAV_T5_CANDIDATE_PROFILE"] = candidate_profile
    if os.name == "nt":
        forwarded = env.get("WSLENV", "")
        env["WSLENV"] = ":".join(
            item for item in (forwarded, "INTERNNAV_T5_CANDIDATE_PROFILE") if item
        )
    completed = subprocess.run(
        [
            "bash",
            RUNNER.relative_to(ROOT).as_posix(),
            "a",
            "canary60",
            "t5fast01",
            "a" * 40,
            "results/internnav_t5/d0-0-prepare-t5d0020260719t000000",
            "results/internnav_t5/fast-lane-a-canary60-t5fast01",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 64
    assert "ACQUIRED" not in completed.stdout + completed.stderr


def test_final10_rejects_raw_wire_before_any_resource_action() -> None:
    env = dict(os.environ)
    env["INTERNNAV_T5_CANDIDATE_PROFILE"] = "recovery_a"
    env["INTERNVLA_T5_SYSTEM2_REPLAN_POLICY"] = "raw_wire_warn"
    env["INTERNNAV_T5_ISAAC_SENSOR_PROFILE"] = "dual_lane_wp03_stop_shadow"
    env["INTERNVLA_T5_STEP3_TIMEOUT_ADVISOR"] = "1"
    env["INTERNVLA_T5_TERMINATION_MODE"] = "oracle_termination"
    if os.name == "nt":
        forwarded = env.get("WSLENV", "")
        env["WSLENV"] = ":".join(
            item
            for item in (
                forwarded,
                "INTERNNAV_T5_CANDIDATE_PROFILE",
                "INTERNVLA_T5_SYSTEM2_REPLAN_POLICY",
                "INTERNNAV_T5_ISAAC_SENSOR_PROFILE",
                "INTERNVLA_T5_STEP3_TIMEOUT_ADVISOR",
                "INTERNVLA_T5_TERMINATION_MODE",
            )
            if item
        )
    completed = subprocess.run(
        [
            "bash",
            RUNNER.relative_to(ROOT).as_posix(),
            "a",
            "final10",
            "t5fast01",
            "a" * 40,
            "results/internnav_t5/d0-0-prepare-t5d0020260719t000000",
            "results/internnav_t5/fast-lane-a-final10-t5fast01",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 64
    assert "ACQUIRED" not in completed.stdout + completed.stderr


def test_candidate_profile_is_bound_forwarded_and_reported() -> None:
    text = source()
    assert "if [[ -v INTERNNAV_T5_CANDIDATE_PROFILE ]]" in text
    assert "candidate_profile=baseline" in text
    assert 'case "$candidate_profile" in baseline|recovery_a)' in text
    assert text.count('INTERNNAV_T5_CANDIDATE_PROFILE="$candidate_profile"') == 1
    assert 'INTERNNAV_T5_CANDIDATE_PROFILE="$candidate"' in text
    assert "candidate_profile in {\"baseline\", \"recovery_a\"}" in text
    assert '"candidate_profile": candidate_profile' in text
    assert 'binding.get("candidate_profile")==candidate_profile' in text
    assert 'dgx_ready.get("candidate_profile")==candidate_profile' in text
    assert 'dgx_status.get("candidate_profile")==candidate_profile' in text


def test_oracle_does_not_require_model_only_episode_order_manifest() -> None:
    text = source()
    assert 'run_mode=="oracle"' in text
    assert "and ordered_episode_manifest is None" in text
    assert 'and nvblox_mode=="active_local_gt"' in text
    assert 'screen_dataset_audit.get("selected_episode_keys")' in text
    assert 'run_mode=="model"' in text


def test_fast_lane_has_exact_disjoint_physical_profiles() -> None:
    text = source()
    assert "resource_profile=lane-a" in text
    assert "resource_profile=lane-b" in text
    assert 'with_resource_lease.sh" "$resource_profile"' in text
    assert 'INTERNNAV_T5_RESOURCE_LEASE_ACK="$resource_profile"' in text
    assert "DGX_A+GPU0" not in text  # documentation is not used as an acknowledgement
    assert 'cpuset="${INTERNVLA_T5_LANE_A_CPUSET:-0,2,4,6,8,10,12,14,16}"' in text
    assert 'cpuset="${INTERNVLA_T5_LANE_B_CPUSET:-1,3,5,7,9,11,13,15,17}"' in text


def test_fast_lane_does_not_consume_formal_authority_or_cross_lane_quiet() -> None:
    text = source()
    forbidden = (
        "T5_DUAL_LANE_BOARD",
        "ONLINE_GRANT",
        "predecessor_receipt",
        "assert_other_lane_quiet",
        "resource_profile=all-lanes",
        'with_resource_lease.sh" all-lanes',
    )
    for fragment in forbidden:
        assert fragment not in text
    assert '"board_grant_used":False' in text
    assert '"predecessor_gate_used":False' in text
    assert '"other_lane_quiet_required":False' in text


def test_run_identity_and_result_root_are_fail_closed() -> None:
    text = source()
    assert ".removeprefix(" not in text  # coordinator still supports Python 3.8
    assert '[[ "$run_id" =~ ^[a-z0-9][a-z0-9._-]{7,95}$ ]]' in text
    assert '[[ "$code_sha" =~ ^[0-9a-f]{40}$ ]]' in text
    assert 'expected_result="results/internnav_t5/fast-lane-${lane}-${profile}-${run_id}"' in text
    assert 'test "$result_relative" = "$expected_result"' in text
    assert 'rev-parse HEAD' in text
    assert 'status --porcelain --untracked-files=all' in text
    assert 'test ! -L "$prep_dir/$receipt"' in text
    assert '"deployment_roots_binding": final.get("deployment_roots") == roots' in text
    assert 'test ! -e "$result_dir"' in text
    assert "test ! -e '$dgx_run'" in text
    assert "test ! -e '$x86_run'" in text


def test_fast_lane_reuses_low_level_colocated_and_isaac_only_runners() -> None:
    text = source()
    assert 'run_t5_dgx_lane.sh"' in text
    assert 'run_t5_distributed_isaac.sh"' in text
    assert '"$lane" "$run_mode" "$result" "$map"' in text
    assert '"$lane" "$run_mode" "$result" "$dataset"' in text
    assert "INTERNNAV_RUNTIME_POLICY=completion_sim" in text
    assert "INTERNNAV_SIMULATION_TARGET=isaac" in text
    assert 'cpuset="${12}"' in text
    assert '"$cpuset_env=$cpuset"' in text
    assert "'$rtf_ablation_profile' '$cpuset'" in text


def test_nvblox_shadow_fixed5_and_active_oracle_remain_lane_a_scoped() -> None:
    text = source()
    assert 'nvblox_mode="${INTERNNAV_T5_NVBLOX_MODE:-off}"' in text
    assert 'case "$nvblox_mode" in' in text
    assert 'case "$profile" in screen3|fixed5)' in text
    assert '(nvblox_mode == "shadow" and profile in {"screen3", "fixed5"})' in text
    assert '(nvblox_mode == "active_local_gt" and profile == "screen3")' in text
    assert 'run_mode="${INTERNNAV_T5_RUN_MODE:-model}"' in text
    assert 'test "$nvblox_mode" = active_local_gt || usage' in text
    assert text.count('INTERNNAV_T5_NVBLOX_MODE="$nvblox_mode"') == 3
    assert 'nvblox_mode="${23}"' in text
    assert "'$fault_injection_profile' '$nvblox_mode'" in text
    assert 'binding.get("nvblox_mode")==nvblox_mode' in text
    assert 'dgx_contract["nvblox"].get("mode")==nvblox_mode' in text
    assert 'dgx_status.get("nvblox_mode")==nvblox_mode' in text
    assert 'x86_status.get("mode")==run_mode' in text


def test_fixed_active_isaac_host_targets_x86_and_reaches_dgx_peer_config() -> None:
    text = source()
    assert 'x86_ip="${ISAAC_HOST:-10.100.120.123}"' in text
    assert 'test "$x86_ip" = 10.100.120.123' in text
    assert 'x86_target="song@$x86_ip"' in text
    assert 'env ISAAC_HOST="$x86_ip" bash "$root/scripts/with_resource_lease.sh"' in text
    assert "grep -Fq ' $x86_ip/'" in text
    assert 'candidate="$8"; isaac_ip="$9"' in text
    assert 'INTERNVLA_T5_ISAAC_IP="$isaac_ip"' in text
    assert "'$candidate_profile' '$x86_ip'" in text


def test_engineering_and_fixed_dataset_profiles_have_distinct_exit_contracts() -> None:
    text = source()
    assert "PROFILE: canary60 | soak600 | screen1 | screen3 | fixed5" in text
    assert 'engineering_canary_sec=60' in text
    assert 'engineering_canary_sec=600' in text
    assert 'INTERNNAV_T5_ENGINEERING_CANARY_SEC="$canary_sec"' in text
    assert 'engineering_canary_ack=fast-path' in text
    assert 'INTERNNAV_T5_ENGINEERING_CANARY_ACK="$canary_ack"' in text
    assert 'INTERNNAV_T5_FAST_CANARY_TIMEOUT_SEC:-420' in text
    assert 'canary_window_completed=1' in text
    assert 'test "$x86_rc" = 0' in text
    assert 'expected_engineering_seconds={"canary60":60,"soak600":600}.get(profile)' in text
    assert 'engineering_profile and canary_completed==1' in text
    assert 'engineering_canary.get("configured_seconds")==expected_engineering_seconds' in text
    assert 'engineering_canary.get("episode_acceptance_claimed") is False' in text
    assert 'x86_status.get("execution_profile")=="engineering_canary"' in text
    assert 'x86_status.get("episode_acceptance_claimed") is False' in text
    assert 'x86_status.get("execution_profile")=="fixed_dataset"' in text
    assert 'x86_status.get("episode_acceptance_claimed") is True' in text
    assert 'x86_status.get("evaluation_completed_naturally") is False' in text


def test_screen_profiles_materialize_a_frozen_first_n_dataset() -> None:
    text = source()
    assert 'screen1|pilot-screen1) screen_episode_count=1' in text
    assert 'screen3) screen_episode_count=3' in text
    assert 'print(",".join(value["episode_keys"]))' in text
    assert 'print(",".join(value["execution_episode_keys"]))' in text
    assert 'python3 - "$execution_episode_keys_csv" "$execution_episode_count"' in text
    assert 'assert screen_count==execution_count and len(keys)>=execution_count' in text
    assert 'materialize_t5_screen_dataset.py' in text
    assert '--expected-source-sha256 "$source_dataset_sha256"' in text
    assert '--expected-episode-keys "$frozen_episode_keys_csv"' in text
    assert 'screen_episode_key="${INTERNNAV_T5_SCREEN_EPISODE_KEY:-}"' in text
    assert 'if test "$profile" = pilot-screen1; then' in text
    assert 'final_pilot_selection_args=(--execution-profile "$profile")' in text
    assert 'final_pilot_selection_args+=(--episode-key "$screen_episode_key")' in text
    assert 'screen_key_args=(--episode-key "$screen_episode_key")' in text
    assert '"screen_episode_key_exact_scope"' in text
    assert 'print(",".join(value["episode_keys"]))' in text
    assert 'print(",".join(value["execution_episode_keys"]))' in text
    assert (
        'python3 - "$execution_episode_keys_csv" "$execution_episode_count"'
        in text
    )
    assert (
        'python3 - "$frozen_episode_keys_csv" "$execution_count" "$screen_count"'
        in text
    )
    assert "assert screen_count==execution_count and len(keys)>=execution_count" in text
    assert '"canary60":5' in text
    assert '"soak600":5' in text
    assert '"screen1":1' in text
    assert '"screen3":3' in text
    assert '"fixed5":5' in text
    assert '"pilot-screen1":1' in text
    assert '"final10":10' in text
    assert '"screen_dataset_binding"' in text


def test_completion_sim_oracle_termination_supports_exact_dual_lane_shadow_profile() -> None:
    text = source()
    assert 'termination_mode="${INTERNVLA_T5_TERMINATION_MODE:-model_stop}"' in text
    assert "model_stop|oracle_termination" in text
    assert 'test "$isaac_sensor_profile" = dual_lane_wp03_stop_shadow' in text
    assert 'and lane in {"a", "b"} and profile == "final10"' in text
    assert 'test "$run_mode" = model || usage' in text
    assert 'test "$isaac_sensor_profile" = lane_a_step3_timeout_advisor || usage' in text
    assert "verify_t5_oracle_termination_dataset.py" in text
    assert 'INTERNVLA_T5_ORACLE_DATASET_FILE="$oracle_dataset"' in text
    assert text.count('INTERNVLA_T5_TERMINATION_MODE="$termination_mode"') == 2
    assert 'pilot_max_step=16000' in text
    assert 'pilot_max_step=1200' not in text
    assert 'INTERNVLA_T4_MAX_STEP="$pilot_max_step"' in text


def test_screen_episode_key_crosses_the_resource_lease_boundary() -> None:
    text = source()
    assert (
        'INTERNNAV_T5_SCREEN_EPISODE_KEY="$screen_episode_key"' in text
    )


def test_summary_preserves_machine_readable_failure_before_x86_receipts_exist() -> None:
    text = source()
    assert 'and isinstance(x86_contract,dict)' in text
    assert 'and isinstance(x86_status,dict)' in text
    assert 'and x86_contract.get("final_pilot_lane")==lane' in text


def test_engineering_sha_is_sealed_into_runtime_and_fixed5_is_not_applicable() -> None:
    text = source()
    assert "hashlib.sha256(engineering_canary_path.read_bytes()).hexdigest()" in text
    assert '"engineering_canary_sealed"' in text
    assert '"sealed_evidence":{"engineering_canary":engineering_canary_seal,' in text
    assert '"revc_snapshot_smoke":revc_snapshot_seal,' in text
    assert '"revc_fixed5_capture":revc_fixed5_seal}' in text
    assert '"applicability":"required" if engineering_profile' in text
    assert 'else "not_applicable"' in text
    assert '"relative_path":"remote/x86/engineering_canary.json"' in text
    assert 'and engineering_canary_seal["sha256"] is None' in text


def test_cleanup_is_owned_and_strictly_lane_scoped() -> None:
    text = source()
    assert "t5_quarantine_arm" in text
    assert "t5_quarantine_owned_clear" in text
    assert "coordinator_cleanup_receipt.json" in text
    assert "dgx_supervisor_ledger.json" in text
    assert "x86_supervisor_ledger.json" in text
    assert '"own_isaac_container_stopped_pid_zero"' in text
    assert '"own_dgx_ports_zero"' in text
    assert '"own_x86_ports_socket_runtime_lock_zero"' in text
    assert "t5_remote_compute_absent" in text
    assert "run_scope_absent" in text
    assert "internnav_t5_isaac_shared_assets.lock" not in text
    assert "other_container" not in text
    assert "other_ports" not in text
    assert "assert_other_lane_quiet" not in text


def test_cleanup_has_child_ledger_fallback_and_one_shared_budget() -> None:
    text = source()
    cleanup_start = text.index("request_stop_and_audit()")
    cleanup = text[cleanup_start : text.index("\nfinish()", cleanup_start)]
    assert 'runtime_ledger = Path(expected_root) / "pid_ledger.jsonl"' in text
    assert 'row.get("event") == "verified_absent"' in text
    assert 'row.get("scope", "host") == "host"' in text
    assert 'expected_root in row["command"]' in text
    assert '"argv_sha256":sys.argv[7]' in text
    assert '"sid":int(sys.argv[5])' in text
    assert 'cleanup_deadline=$((SECONDS + cleanup_timeout))' in text
    assert 'kill_deadline=$((SECONDS + cleanup_kill_timeout))' in text
    assert '--cleanup-timeout 60 --kill-wait-timeout 10' in text
    assert 'INTERNNAV_T5_FAST_CLEANUP_TIMEOUT_SEC:-30' in text
    assert 'INTERNNAV_T5_FAST_CLEANUP_KILL_TIMEOUT_SEC:-10' in text
    assert 'cleanup_timeout <= 60 && cleanup_kill_timeout <= 60' in text
    assert "wait_supervisor_absent" not in text
    assert "touch '$dgx_run/stop.request'" not in cleanup
    assert "touch '$x86_run/stop.request'" not in cleanup
    assert 'supervisor_action "$target" "$ledger" "$run_root" TERM' in cleanup
    assert 'supervisor_action "$target" "$ledger" "$run_root" KILL' in cleanup


def test_every_killpg_is_guarded_by_fresh_proc_identity_check() -> None:
    program = named_heredoc("REMOTE_SUPERVISOR_CONTROL")
    assert program.count("os.killpg(") == 1
    assert "def signal_verified_group(" in program
    assert 'current_anchor = read_process_identity(expected_anchor["pid"])' in program
    assert "current_members = read_group_members(candidate_pgid)" in program
    assert 'identity_fields = ("pid", "pgid", "sid", "starttime")' in program
    assert 'expected_root not in current_anchor["command"]' in program
    assert program.index("current_anchor = read_process_identity") < program.index(
        "os.killpg("
    )


@pytest.mark.skipif(os.name != "posix", reason="requires Linux /proc and process groups")
def test_missing_supervisor_terminates_setsid_child_from_runtime_ledger(
    tmp_path: Path,
) -> None:
    program = named_heredoc("REMOTE_SUPERVISOR_CONTROL")
    run_root = tmp_path / "runtime-root"
    run_root.mkdir()
    child = subprocess.Popen(
        ["bash", "-c", 'exec -a "$0/managed-child" sleep 60', str(run_root)],
        preexec_fn=os.setsid,
    )
    try:
        (run_root / "pid_ledger.jsonl").write_text(
            json.dumps({
                "event": "started", "scope": "host", "component": "nav2",
                "pid": child.pid, "pgid": child.pid,
            }) + "\n",
            encoding="utf-8",
        )
        ledger = tmp_path / "supervisor.json"
        ledger.write_text(json.dumps({
            "run_root": str(run_root), "pid": 99_999_991,
            "pgid": 99_999_991, "sid": 99_999_991, "starttime": 2,
            "argv_sha256": "0" * 64,
        }), encoding="utf-8")
        completed = subprocess.run(
            ["python3", "-c", program, str(ledger), str(run_root), "TERM"],
            capture_output=True, text=True, check=False,
        )
        assert completed.returncode == 0, completed.stderr
        child.wait(timeout=5)
        audit = subprocess.run(
            ["python3", "-c", program, str(ledger), str(run_root), "AUDIT"],
            capture_output=True, text=True, check=False,
        )
        assert audit.returncode == 0, audit.stderr
        assert json.loads(audit.stdout)["absent"] is True
    finally:
        if child.poll() is None:
            os.killpg(child.pid, 9)
            child.wait(timeout=5)


@pytest.mark.skipif(os.name != "posix", reason="requires Linux /proc and process groups")
def test_dead_supervisor_leader_uses_live_member_anchor_and_becomes_absent(
    tmp_path: Path,
) -> None:
    program = named_heredoc("REMOTE_SUPERVISOR_CONTROL")
    run_root = tmp_path / "runtime-root"
    run_root.mkdir()
    info_path = tmp_path / "leader-info.json"
    release_path = tmp_path / "release-leader"
    launcher = """
import json, os, sys, time
from pathlib import Path

child = os.fork()
if child == 0:
    os.execlp("bash", "bash", "-c", 'exec -a "$0/member" sleep 60', sys.argv[3])
stat = Path(f"/proc/{os.getpid()}/stat").read_text(encoding="utf-8")
tail = stat.rsplit(")", 1)[1].strip().split()
Path(sys.argv[1]).write_text(json.dumps({
    "pid": os.getpid(), "child_pid": child, "pgid": int(tail[2]),
    "sid": int(tail[3]), "starttime": int(tail[19]),
}), encoding="utf-8")
while not Path(sys.argv[2]).exists():
    time.sleep(0.01)
os._exit(0)
"""
    leader = subprocess.Popen(
        ["python3", "-c", launcher, str(info_path), str(release_path), str(run_root)],
        preexec_fn=os.setsid,
    )
    child_pid = None
    try:
        for _ in range(500):
            if info_path.exists():
                break
            time.sleep(0.01)
        else:
            pytest.fail("leader did not publish its process identity")
        identity = json.loads(info_path.read_text(encoding="utf-8"))
        child_pid = identity.pop("child_pid")
        ledger = tmp_path / "supervisor.json"
        ledger.write_text(json.dumps({
            "run_root": str(run_root), **identity, "argv_sha256": "0" * 64,
        }), encoding="utf-8")
        release_path.touch()
        leader.wait(timeout=5)
        completed = subprocess.run(
            ["python3", "-c", program, str(ledger), str(run_root), "TERM"],
            capture_output=True, text=True, check=False,
        )
        assert completed.returncode == 0, completed.stderr
        receipt = json.loads(completed.stdout)
        assert receipt["signal_identity_checks"][0]["anchor_source"] == (
            "discovered_live_group_member"
        )
        assert identity["pgid"] in receipt["signalled_groups"]
        for _ in range(100):
            audit = subprocess.run(
                ["python3", "-c", program, str(ledger), str(run_root), "AUDIT"],
                capture_output=True, text=True, check=False,
            )
            assert audit.returncode == 0, audit.stderr
            if json.loads(audit.stdout)["absent"] is True:
                break
            time.sleep(0.05)
        else:
            pytest.fail("leader-dead supervisor group did not become absent")
    finally:
        if leader.poll() is None:
            os.killpg(leader.pid, 9)
            leader.wait(timeout=5)
        if child_pid is not None:
            try:
                os.killpg(identity["pgid"], 9)
            except ProcessLookupError:
                pass


@pytest.mark.skipif(os.name != "posix", reason="requires Linux /proc and process groups")
def test_supervisor_cleanup_accepts_same_process_bash_exec_transition(
    tmp_path: Path,
) -> None:
    program = named_heredoc("REMOTE_SUPERVISOR_CONTROL")
    run_root = tmp_path / "runtime-root"
    run_root.mkdir()
    child = subprocess.Popen(
        [
            "bash",
            "-c",
            'read -r _; exec -a "$0/runtime-supervisor" sleep 60',
            str(run_root),
        ],
        stdin=subprocess.PIPE,
        preexec_fn=os.setsid,
    )
    try:
        raw = Path(f"/proc/{child.pid}/cmdline").read_bytes()
        stat = Path(f"/proc/{child.pid}/stat").read_text(encoding="utf-8")
        tail = stat.rsplit(")", 1)[1].strip().split()
        ledger = tmp_path / "supervisor.json"
        ledger.write_text(
            json.dumps(
                {
                    "run_root": str(run_root),
                    "pid": child.pid,
                    "pgid": int(tail[2]),
                    "sid": int(tail[3]),
                    "starttime": int(tail[19]),
                    "argv_sha256": hashlib.sha256(raw).hexdigest(),
                }
            ),
            encoding="utf-8",
        )
        assert child.stdin is not None
        child.stdin.write(b"go\n")
        child.stdin.flush()
        for _ in range(100):
            current = Path(f"/proc/{child.pid}/cmdline").read_bytes()
            if hashlib.sha256(current).hexdigest() != hashlib.sha256(raw).hexdigest():
                break
            time.sleep(0.01)
        else:
            pytest.fail("supervisor did not exec its final runtime command")
        completed = subprocess.run(
            ["python3", "-c", program, str(ledger), str(run_root), "TERM"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert json.loads(completed.stdout)["leader_argv_matches_ledger"] is False
        child.wait(timeout=5)
    finally:
        if child.poll() is None:
            os.killpg(child.pid, 9)
            child.wait(timeout=5)


@pytest.mark.skipif(os.name != "posix", reason="requires Linux /proc and process groups")
@pytest.mark.parametrize("drift", ("pgid", "sid", "starttime", "run_root"))
def test_supervisor_cleanup_refuses_identity_drift_before_signal(
    tmp_path: Path, drift: str,
) -> None:
    program = named_heredoc("REMOTE_SUPERVISOR_CONTROL")
    run_root = tmp_path / "runtime-root"
    run_root.mkdir()
    child = subprocess.Popen(
        ["bash", "-c", 'exec -a "$0/runtime-supervisor" sleep 60', str(run_root)],
        preexec_fn=os.setsid,
    )
    try:
        raw = Path(f"/proc/{child.pid}/cmdline").read_bytes()
        stat = Path(f"/proc/{child.pid}/stat").read_text(encoding="utf-8")
        tail = stat.rsplit(")", 1)[1].strip().split()
        payload = {
            "run_root": str(run_root),
            "pid": child.pid,
            "pgid": int(tail[2]),
            "sid": int(tail[3]),
            "starttime": int(tail[19]),
            "argv_sha256": hashlib.sha256(raw).hexdigest(),
        }
        expected_root = str(run_root)
        if drift == "pgid":
            payload["pgid"] += 1
        elif drift == "sid":
            payload["sid"] += 1
        elif drift == "starttime":
            # Inject the change only after discovery so this case exercises
            # the fresh pre-signal identity comparison, not the initial audit.
            program = program.replace(
                '    current_anchor = read_process_identity(expected_anchor["pid"])',
                '    expected_anchor = dict(expected_anchor)\n'
                '    expected_anchor["starttime"] += 1\n'
                '    current_anchor = read_process_identity(expected_anchor["pid"])',
                1,
            )
        else:
            other_root = tmp_path / "other-runtime-root"
            other_root.mkdir()
            payload["run_root"] = str(other_root)
            expected_root = str(other_root)
        ledger = tmp_path / "supervisor.json"
        ledger.write_text(json.dumps(payload), encoding="utf-8")
        completed = subprocess.run(
            ["python3", "-c", program, str(ledger), expected_root, "TERM"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode != 0
        assert child.poll() is None, f"{drift} drift must fail before killpg"
        if drift == "starttime":
            assert "identity drift before signal" in completed.stderr
    finally:
        if child.poll() is None:
            os.killpg(child.pid, 9)
            child.wait(timeout=5)


def test_dataset_is_cryptographically_bound_to_frozen_input() -> None:
    text = source()
    assert '"dataset_audit_sha_binding"' in text
    assert '"dataset_sha_frozen_binding"' in text
    assert '"episode_keys_frozen_binding"' in text
    assert 'dataset.get("episode_keys") == expected_keys' in text
    assert 'prep.get("episode_keys") == expected_keys' in text
    assert 'final.get("episode_keys") == expected_keys' in text
    assert 'dataset_root == frozen_input.get("dataset_root")' in text


def test_raw_remote_and_lease_logs_are_preserved() -> None:
    text = source()
    assert '"$result_dir/logs/dgx_runtime_ssh.log"' in text
    assert '"$result_dir/logs/x86_runtime_ssh.log"' in text
    assert '"$result_dir/remote/dgx"' in text
    assert '"$result_dir/remote/x86"' in text
    assert "collect_x86_machine_records" in text
    assert "find . -type f" in text
    assert "-name '*.jsonl'" in text
    assert 'collect_remote_tree "$x86_target"' not in text
    assert '"x86_full_archive":"DEFERRED_TO_SERIAL_SHARED_IO"' in text
    assert '"remote_x86_result_root"' in text
    assert 'cp -a -- "$lease_bootstrap/." "$result_dir/lease/"' in text
    assert "lease_release_summary.json" in text
    assert "fast_lane_final_summary.json" in text
