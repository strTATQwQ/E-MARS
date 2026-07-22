from __future__ import annotations

import os
import json
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "coordination" / "run_t5_fast_prepare_online.sh"


def source() -> str:
    return RUNNER.read_text(encoding="utf-8")


def heredoc(text: str, marker: str) -> str:
    lines = text.splitlines()
    start = next(index for index, line in enumerate(lines) if f"<<'{marker}'" in line)
    end = next(index for index in range(start + 1, len(lines)) if lines[index] == marker)
    return "\n".join(lines[start + 1 : end]) + "\n"


def test_shell_and_embedded_programs_parse() -> None:
    completed = subprocess.run(
        ["bash", "-n", RUNNER.relative_to(ROOT).as_posix()],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    text = source()
    for marker in ("REMOTE_DGX_BUILD", "REMOTE_X86_PREPARE"):
        completed = subprocess.run(
            ["bash", "-n"],
            input=heredoc(text, marker).encode("utf-8"),
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, (
            f"{marker}: {completed.stderr.decode('utf-8', errors='replace')}"
        )

    compile(heredoc(text, "REMOTE_RESIDUAL_AUDIT"), str(RUNNER), "exec")
    compile(heredoc(text, "REMOTE_PREP_SUPERVISOR_CONTROL"), str(RUNNER), "exec")
    lines = text.splitlines()
    compiled = 0
    index = 0
    while index < len(lines):
        if "<<'PY'" not in lines[index]:
            index += 1
            continue
        end = next(
            candidate
            for candidate in range(index + 1, len(lines))
            if lines[candidate] == "PY"
        )
        compile("\n".join(lines[index + 1 : end]) + "\n", str(RUNNER), "exec")
        compiled += 1
        index = end + 1
    assert compiled >= 12


@pytest.mark.parametrize(
    "arguments",
    (
        (),
        ("not-a-sha", "t5d0020260719t000000", "results/internnav_t5/d0-0-prepare-t5d0020260719t000000"),
        ("a" * 40, "bad-run", "results/internnav_t5/d0-0-prepare-bad-run"),
        ("a" * 40, "t5d0020260719t000000", "results/internnav_t5/wrong"),
    ),
)
def test_invalid_identity_exits_before_resource_or_network_action(
    arguments: tuple[str, ...],
) -> None:
    completed = subprocess.run(
        ["bash", RUNNER.relative_to(ROOT).as_posix(), *arguments],
        cwd=ROOT,
        env={**os.environ, "INTERNNAV_T5_RESOURCE_LEASE_ACK": ""},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 64
    assert "ACQUIRED" not in completed.stdout + completed.stderr


def test_exact_clean_ref_unique_result_and_deployment_roots() -> None:
    text = source()
    assert '[[ "$code_sha" =~ ^[0-9a-f]{40}$ ]]' in text
    assert '[[ "$run_id" =~ ^t5d00[0-9]{8}t[0-9]{6}$ ]]' in text
    assert 'test "$result_relative" = "results/internnav_t5/d0-0-prepare-$run_id"' in text
    assert 'rev-parse HEAD' in text
    assert 'status --porcelain --untracked-files=all' in text
    assert 'test ! -e "$result_dir"' in text
    for suffix in ("lane-a", "lane-b", "isaac-prepare", "isaac-a", "isaac-b"):
        assert suffix in text
    assert "deployment_roots" in text


def test_active_isaac_host_and_cpusets_are_fixed_for_prepare_and_workers() -> None:
    text = source()
    remote_prepare = heredoc(text, "REMOTE_X86_PREPARE")
    assert 'x86_ip="${ISAAC_HOST:-10.100.120.123}"' in text
    assert 'test "$x86_ip" = 10.100.120.123' in text
    assert 'x86_target="song@$x86_ip"' in text
    assert 'lane_a_cpuset="${INTERNVLA_T5_LANE_A_CPUSET:-0,2,4,6,8,10,12,14,16}"' in text
    assert 'lane_b_cpuset="${INTERNVLA_T5_LANE_B_CPUSET:-1,3,5,7,9,11,13,15,17}"' in text
    assert 'env ISAAC_HOST="$x86_ip" bash "$root/scripts/with_resource_lease.sh"' in text
    assert 'expected_user=song; expected_ip="$x86_ip"' in text
    assert "'$supervisor_ledger' '$x86_ip' '$lane_a_cpuset' '$lane_b_cpuset'" in text
    assert 'expected_isaac_ip="$7"' in remote_prepare
    assert 'INTERNVLA_T5_ISAAC_IP="$expected_isaac_ip"' in remote_prepare
    assert 'lane_a_cpuset="$8"' in remote_prepare
    assert 'lane_b_cpuset="$9"' in remote_prepare
    assert 'INTERNVLA_T5_LANE_A_CPUSET="$lane_a_cpuset"' in remote_prepare
    assert 'INTERNVLA_T5_LANE_B_CPUSET="$lane_b_cpuset"' in remote_prepare


def test_one_exact_archive_is_verified_and_bound_in_every_root() -> None:
    text = source()
    assert text.count('archive --format=tar "$code_sha"') == 1
    assert 'test "$(sha256sum "$archive" | cut -d\' \' -f1)" = "$archive_sha256"' in text
    assert "sha256sum '$stage_archive'" in text
    assert "gzip -dc '$stage_archive' | tar" in text
    assert "T5_DEPLOYMENT_REF" in text
    assert "T5_DEPLOYMENT_ARCHIVE_SHA256" in text
    assert "record_deployment_bindings" in text
    assert '"code_ref_exact"' in text
    assert '"archive_sha_exact"' in text


def test_three_real_disjoint_lease_tasks_start_in_parallel() -> None:
    text = source()
    assert 'run_task dgx_a dgx-a' in text
    assert 'run_task dgx_b dgx-b' in text
    assert 'run_task x86 isaac' in text
    assert text.count('run_task dgx_') == 2
    assert text.count('run_task x86 isaac') == 1
    assert 'run_task lane_a_pair lane-a' in text
    assert text.count("child_pids+=(") == 4
    assert 'INTERNNAV_T5_RESOURCE_LEASE_ACK="$profile"' in text
    assert 'with_resource_lease.sh" all-lanes' not in text
    assert "INTERNNAV_T5_RESOURCE_LEASE_ACK=all-lanes" not in text
    assert "dgx+isaac" not in text
    assert '"all_lanes_used":False' in text


def test_x86_dataset_map_and_container_work_is_strictly_serial() -> None:
    text = heredoc(source(), "REMOTE_X86_PREPARE")
    dataset = text.index('"$audit/dataset_audit.json"')
    map_build = text.index("build_t3_static_maps.py")
    manifest = text.index("build_t4_truth_isolated_static_manifest.py")
    validate = text.index("validate_t5_frozen_assets.py")
    worker = text.index("prepare_t5_isaac_workers.sh")
    stop = text.index("stop_workers\n", worker)
    assert dataset < map_build < manifest < validate < worker < stop
    assert "prepare_t4_isaac_static_maps.sh" not in text
    assert "INTERNNAV_T4_RESOURCE_LEASE_ACK" not in text
    assert 'INTERNNAV_T5_RESOURCE_LEASE_ACK="$(test "$prepare_scope" = lane-a && printf lane-a || printf isaac)"' in text
    assert "map_prepare_status.json" in text
    assert '"isaac_offline_scene_map_builder"' in text


def test_quarantine_has_predeploy_and_final_global_forbidden_compute_audits() -> None:
    text = source()
    armed = text.index("t5_quarantine_arm")
    predeploy = text.index('t5_remote_compute_absent "$target" "$audit_dir/global_compute_predeploy.json"')
    deploy = text.index('deploy_archive "$target"', predeploy)
    assert armed < predeploy < deploy

    finish = text.index("inside_finish()")
    residual = text.index("run_residual_audit", finish)
    final_global = text.index('global_path="$audit_dir/global_compute_final.json"', finish)
    receipt = text.index("write_cleanup_receipt", final_global)
    clear = text.index("t5_quarantine_owned_clear", receipt)
    assert finish < residual < final_global < receipt < clear
    assert '"global_forbidden_compute_audit_pass"' in text
    assert 'if test "$role" != x86; then\n    t5_remote_compute_absent' not in text


def test_cleanup_is_fail_closed_and_never_clears_without_pass_receipt() -> None:
    text = source()
    assert '"errors_empty": not errors' in text
    assert '"remote_errors_empty"' in text
    assert 'if test "$quarantine_armed" = true && test "$cleanup_rc" = 0; then' in text
    assert 'test "$cleanup_rc" = 0 && test "$clear_rc" = 0 || final_exit=75' in text
    assert '"supervisor_cleanup_pass":int(sys.argv[7])==0' in text
    assert '"cleanup_receipt_pass":int(sys.argv[10])==0' in text
    assert '"quarantine_clear_pass":int(sys.argv[11])==0' in text


def test_prepare_supervisors_and_outer_lease_wrappers_are_identity_bound() -> None:
    text = source()
    assert 'supervisor_ledger="$destination/results/fast_prepare/supervisor.json"' in text
    assert '"pgid":int(sys.argv[4]),"sid":int(sys.argv[5])' in text
    assert '"starttime":int(sys.argv[6])' in text
    assert '"argv_sha256":sys.argv[7]' in text
    assert 'remote_operation_launch_attempted=true' in text
    assert text.index("stop_remote_prepare_supervisor", text.index("inside_finish()")) \
        < text.index("run_residual_audit", text.index("inside_finish()"))
    assert 'setsid --wait bash "$root/coordination/run_t5_fast_prepare_online.sh"' in text
    assert "--lease-task" in text
    assert "_lease_wrapper_supervisor.json" in text
    assert 'deadline=$((SECONDS + term_timeout))' in text
    assert 'kill_deadline=$((SECONDS + kill_timeout))' in text
    assert 'all_holder_cleanup_receipts_pass' in text


@pytest.mark.skipif(os.name != "posix", reason="requires Linux /proc and process groups")
def test_prepare_interrupt_fallback_terminates_root_bound_orphan(
    tmp_path: Path,
) -> None:
    program = heredoc(source(), "REMOTE_PREP_SUPERVISOR_CONTROL")
    run_root = tmp_path / "prepare-root"
    run_root.mkdir()
    child = subprocess.Popen(
        ["bash", "-c", 'exec -a "$0/prepare-child" sleep 60', str(run_root)],
        preexec_fn=os.setsid,
    )
    try:
        ledger = tmp_path / "supervisor.json"
        ledger.write_text(json.dumps({
            "run_root": str(run_root), "pid": 99_999_993,
            "pgid": 99_999_993, "sid": 99_999_993, "starttime": 2,
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
def test_prepare_cleanup_does_not_match_root_inside_remote_command_argument(
    tmp_path: Path,
) -> None:
    program = heredoc(source(), "REMOTE_PREP_SUPERVISOR_CONTROL")
    run_root = tmp_path / "prepare-root"
    run_root.mkdir()
    decoy = subprocess.Popen(
        [
            "python3",
            "-c",
            "import time; time.sleep(60)",
            f"ssh-remote-command=run --root {run_root}",
        ],
        preexec_fn=os.setsid,
    )
    try:
        ledger = tmp_path / "supervisor.json"
        ledger.write_text(json.dumps({
            "run_root": str(run_root), "pid": 99_999_992,
            "pgid": 99_999_992, "sid": 99_999_992, "starttime": 2,
            "argv_sha256": "0" * 64,
        }), encoding="utf-8")
        completed = subprocess.run(
            ["python3", "-c", program, str(ledger), str(run_root), "TERM"],
            capture_output=True, text=True, check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert decoy.poll() is None
        payload = json.loads(completed.stdout)
        assert payload["signalled_groups"] == []
        assert payload["run_processes"] == []
        assert payload["absent"] is True
    finally:
        if decoy.poll() is None:
            os.killpg(decoy.pid, 9)
            decoy.wait(timeout=5)


def test_dataset_binding_is_carried_by_prepare_summary_and_final() -> None:
    text = source()
    assert '"dataset_audit_frozen_binding"' in text
    assert '"dataset_audit_sha256":dataset_audit_sha256' in text
    assert '"dataset_file_sha256":inputs.get("dataset_file_sha256")' in text
    assert '"episode_keys":inputs.get("episode_keys")' in text
    assert '"dataset_file_sha256":summary.get("dataset_file_sha256")' in text
    assert '"dataset_audit_sha256":summary.get("dataset_audit_sha256")' in text
    assert '"episode_keys":summary.get("episode_keys")' in text


def test_compatible_preparation_and_final_receipts_cover_all_artifacts() -> None:
    text = source()
    for artifact in (
        "d0_prepare_summary.json",
        "d0_prepare_final_summary.json",
        "remote/x86/dataset_audit.json",
        "lease_release_summary.json",
        "deployment_binding.json",
        "build_summary.json",
        "container_cleanup.json",
        "map_ready.json",
        "cleanup_receipt.json",
    ):
        assert artifact in text
    for check in (
        "exact_ref_input",
        "all_remote_deployment_bindings_exact",
        "both_dgx_builds_pass",
        "x86_serial_prepare_pass",
        "dataset_audit_pass",
        "worker_containers_stopped",
        "static_maps_all_roots",
        "all_cleanup_receipts_pass",
        "all_owned_quarantines_cleared",
        "all_leases_released",
    ):
        assert f'"{check}"' in text
    assert '"preparation_summary_sha256"' in text
    assert '"deployment_archive_sha256":archive_sha' in text
    assert '"deployment_roots":roots' in text
    assert '"profiles":["dgx-a","dgx-b","isaac"]' in text
