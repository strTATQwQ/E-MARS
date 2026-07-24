from __future__ import annotations

import ast
import base64
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "coordination/run_t5_d0_prepare_online.sh"


def runner_text() -> str:
    return RUNNER.read_text(encoding="utf-8")


def embedded_cleanup_function(name: str):
    text = runner_text()
    marker = "read -r -d '' remote_cleanup_program <<'REMOTE_CLEANUP' || true\n"
    program = text.split(marker, 1)[1].split("\nREMOTE_CLEANUP\n", 1)[0]
    tree = ast.parse(program)
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        or isinstance(node, ast.FunctionDef) and node.name == name
    ]
    namespace: dict[str, object] = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(RUNNER), "exec"), namespace)
    return namespace[name]


def write_process(proc_root: Path, pid: int, argv: list[str]) -> None:
    process = proc_root / str(pid)
    process.mkdir()
    (process / "cmdline").write_bytes(b"\0".join(value.encode() for value in argv) + b"\0")


def test_d0_prepare_consumes_exact_board_only_grant_and_frozen_hashes() -> None:
    text = runner_text()
    assert "INTERNAV_T5_DUAL_LANE_ONLINE_GRANT_V1" in text
    assert 'rev-list --count "$authorization_ref..HEAD"' in text
    assert 'diff --name-only "$authorization_ref" HEAD' in text
    assert 'status --porcelain --untracked-files=all' in text
    assert "changed board content outside the grant block" in text
    assert '"stage": "d0_0_online_prepare"' in text
    assert '"schema_version": 2' in text
    assert '"code_ref_sha": authorization_ref' in text
    assert '"predecessor_receipt_sha256": None' in text
    assert '"lane_scope": "all_lanes"' in text
    assert '"resource_profile": "all-lanes"' in text
    assert '"golden_bundle_sha256": golden_sha' in text
    assert '"run_manifest_sha256": d0_sha' in text
    assert 'show "$authorization_ref:$golden_relative"' in text
    assert 'show "$authorization_ref:$d0_manifest_relative"' in text
    assert "'authorization_mode':'FORMAL_BOARD_GRANT_V2'" in text
    assert "'board_grant_used':True" in text
    assert 'preparation.get("authorization_mode") == "FORMAL_BOARD_GRANT_V2"' in text
    assert 'preparation.get("board_grant_used") is True' in text


def test_d0_prepare_uses_all_four_fail_closed_leases_and_no_overwrite() -> None:
    text = runner_text()
    assert 'with_resource_lease.sh" all-lanes' in text
    assert "INTERNNAV_T5_RESOURCE_LEASE_ACK=all-lanes" in text
    assert 'test ! -e "$result_dir"' in text
    assert 'mkdir "$result_dir"' in text
    assert "four_holders_recorded" in text
    for resource in ("dgx_a", "dgx_b", "isaac_gpu0", "isaac_gpu1"):
        assert resource in text


def test_d0_prepare_deploys_same_authorization_ref_to_three_hosts() -> None:
    text = runner_text()
    assert "railgun@10.100.100.128" in text
    assert "rail@10.100.120.122" in text
    assert "song@10.100.120.123" in text
    assert 'archive --format=tar "$authorization_ref"' in text
    assert 'deploy "$dgx_a_target" "$dgx_a_root"' in text
    assert 'deploy "$dgx_b_target" "$dgx_b_root"' in text
    assert 'deploy "$x86_target" "$x86_prepare_root"' in text
    assert 'deploy "$x86_target" "$x86_a_root"' in text
    assert 'deploy "$x86_target" "$x86_b_root"' in text
    assert "T5_DEPLOYMENT_REF" in text
    assert "T5_DEPLOYMENT_ARCHIVE_SHA256" in text


def test_dgx_builds_can_overlap_but_x86_mutations_are_serial() -> None:
    text = runner_text()
    assert 'dgx_a_build_ssh.log" dgx_a_ssh_pid' in text
    assert 'dgx_b_build_ssh.log" dgx_b_ssh_pid' in text
    assert 'wait "$dgx_a_ssh_pid"' in text
    assert 'wait "$dgx_b_ssh_pid"' in text
    dataset_position = text.index("actual_dataset_sha=")
    maps_position = text.index("prepare_t4_isaac_static_maps.sh")
    worker_position = text.index("prepare_t5_isaac_workers.sh")
    assert dataset_position < maps_position < worker_position
    assert "x86_dataset_maps_and_container_prepare_serial" in text


def test_prepare_only_never_invokes_model_isaac_or_episode_runtime() -> None:
    text = runner_text()
    for forbidden in (
        "run_t4_model_server.sh",
        "run_t5_dgx_lane.sh",
        "run_t5_distributed_isaac.sh",
        "run_go2_continuous_phase.sh",
        "run_t4_functional_pilot.sh",
    ):
        assert forbidden not in text
    assert "internvla_model_loaded':False" in text
    assert "isaac_sim_started':False" in text
    assert "episode_or_evaluator_started':False" in text
    assert ".env.local" not in text
    assert "HF_TOKEN" not in text


def test_prepare_records_dataset_maps_capacity_pid_container_and_cleanup() -> None:
    text = runner_text()
    assert "dataset_audit.json" in text
    assert "episode_keys_match" in text
    assert "capacity_before.json" in text
    assert "capacity_after.json" in text
    assert "model_runtime_environment.json" in text
    assert '"required_packages_present"' in text
    assert '"cuda_available"' in text
    assert "lane_a_model_runtime_environment" in text
    assert "lane_b_model_runtime_environment" in text
    assert "dual_isaac_minimum_available_memory_bytes" in text
    assert "x86_memory_admission" in text
    assert "PRESERVE_DEPLOYMENTS_AND_RESUME_PLAN_NOTIFY_USER_NO_AUTO_REBOOT" in text
    assert "build_ledger.txt" in text
    assert "prepare_ledger.txt" in text
    assert '"$x86_prepare_root|$x86_a_root|$x86_b_root" x86' in text
    assert "container_cleanup.json" in text
    assert "residual_audit.json" in text
    assert "remote_cleanup_receipt.json" in text
    assert '"remote_cleanup_pass"' in text
    assert '"remote_cleanup_receipt_sha256"' in text
    assert "deployment_roots" in text
    assert "'x86_prepare':sys.argv[12]" in text
    assert "'x86_a':sys.argv[13]" in text
    assert "'x86_b':sys.argv[14]" in text
    assert "static_map_manifest_sha256" in text
    assert "lease_release_summary.json" in text
    assert "d0_prepare_final_summary.json" in text
    assert "preparation_summary_sha256" in text
    assert "lease_release_summary_sha256" in text
    assert '"code_ref_sha": preparation.get("code_ref_sha")' in text
    assert 'test ! -f "$containers_owned/$container"' in text
    assert 'docker stop -t 10 "$container"' in text
    assert "INTERNVLA_T5_CONTAINER_OWNERSHIP_DIR" in text
    assert "gpu_device_requests_match" in text
    assert "stable_control_root_mounted" in text


def test_remote_cleanup_is_fail_closed_and_precedes_lease_pass() -> None:
    text = runner_text()
    assert "remote_cleanup_program" in text
    assert "claimed_pgid_associated_or_absent" in text
    assert "deployment_processes_absent" in text
    assert "claimed_pgid_absent" in text
    assert "t5_sockets_absent" in text
    assert "owned_containers_absent_or_stopped" in text
    assert "os.killpg(group_id, requested_signal)" in text
    assert "signal.SIGTERM" in text
    assert "signal.SIGKILL" in text
    assert 'cleanup_exit=75' in text
    assert text.index('"remote_cleanup_pass"') < text.index('"lease_release_pass"')


def test_prepare_arms_persistent_quarantine_and_only_audited_cleanup_clears_it() -> None:
    text = runner_text()
    assert "dgx_quarantine_file=/tmp/internnav_dgx.quarantine" in text
    assert "isaac_quarantine_file=/tmp/internnav_isaac.quarantine" in text
    assert 'source "$root/scripts/t5_quarantine_common.sh"' in text
    assert "t5_quarantine_arm" in text
    assert "d0_prepare_in_progress" in text
    assert 'set(quarantine_values) != marker_keys' in text
    assert 'quarantine_values.get("schema_version") == "1"' in text
    assert 'os.path.lexists(quarantine_path)' in text
    assert "set -C" not in text  # arm uses O_EXCL/O_NOFOLLOW in the frozen guard
    assert "quarantine_marker_present_if_armed" in text
    assert "quarantine_cleared_only_after_audit_pass" in text
    assert "unowned_quarantine_never_removed" in text
    assert "elif quarantine_owned and all(checks.values()):" in text
    assert 'quarantine_values.get("run_tag") == expected_quarantine_tag' in text
    assert text.index('t5_quarantine_arm "$dgx_a_target"') < text.index(
        '"${git_command[@]}" archive'
    )
    assert text.index("quarantine_path.unlink()") > text.index(
        '"owned_containers_absent_or_stopped": container_cleanup_ok'
    )


def test_prepare_globally_audits_both_quarantined_dgx_hosts_before_deploy() -> None:
    text = runner_text()
    assert 'source "$root/scripts/t5_remote_compute_audit_common.sh"' in text
    assert 'remote_compute_auditor_b64="$_t5_compute_auditor_b64"' in text
    assert 'remote_compute_auditor_sha256="$_t5_compute_auditor_sha256"' in text
    audit_a = 't5_remote_compute_absent "$dgx_a_target"'
    audit_b = 't5_remote_compute_absent "$dgx_b_target"'
    assert "dgx_a_global_compute_predeploy.json" in text
    assert "dgx_b_global_compute_predeploy.json" in text
    assert text.index('t5_quarantine_arm "$dgx_a_target"') < text.index(audit_a)
    assert text.index('t5_quarantine_arm "$dgx_b_target"') < text.index(audit_a)
    assert text.index(audit_a) < text.index(audit_b)
    assert text.index(audit_b) < text.index('"${git_command[@]}" archive')
    assert text.index(audit_b) < text.index('deploy "$dgx_a_target"')
    assert 't5_remote_compute_absent "$x86_target"' not in text


def test_cleanup_repeats_global_dgx_audit_before_quarantine_clear_only() -> None:
    text = runner_text()
    cleanup = text.split("<<'REMOTE_CLEANUP' || true\n", 1)[1].split(
        "\nREMOTE_CLEANUP\n", 1
    )[0]
    assert 'if role in {"dgx_a", "dgx_b"}:' in cleanup
    assert 'checks["dgx_global_forbidden_compute_absent"]' in cleanup
    assert '"global_compute_audit": global_compute_audit' in cleanup
    assert '"evidence": evidence' in cleanup
    assert '"matches": matches' in cleanup
    assert '"errors": errors' in cleanup
    assert '"check": check' in cleanup
    assert cleanup.index("run_embedded_forbidden_compute_audit(") < cleanup.index(
        "quarantine_path.unlink()"
    )
    assert 'elif quarantine_owned and all(checks.values()):' in cleanup
    runner_cleanup = text.split("run_remote_cleanup() {", 1)[1].split("\n}\n", 1)[0]
    assert "dgx_a|dgx_b)" in runner_cleanup
    assert "x86) ;;" in runner_cleanup
    assert 'compute_auditor_sha256="$remote_compute_auditor_sha256"' in runner_cleanup
    assert 'compute_auditor_b64="$remote_compute_auditor_b64"' in runner_cleanup


def test_cleanup_embedded_global_audit_passes_fails_and_errors_closed(
    tmp_path: Path,
) -> None:
    audit = embedded_cleanup_function("run_embedded_forbidden_compute_audit")
    auditor_source = (ROOT / "scripts/t5_process_identity_audit.py").read_bytes()
    auditor_b64 = base64.b64encode(auditor_source).decode("ascii")
    auditor_sha256 = hashlib.sha256(auditor_source).hexdigest()

    clean_proc = tmp_path / "clean-proc"
    clean_proc.mkdir()
    write_process(clean_proc, 101, ["python3", "-c", "print('run_t5_dgx_lane.sh')"])
    clean = audit(auditor_b64, auditor_sha256, str(clean_proc))
    assert clean["status"] == "PASS"
    assert clean["check"] is True
    assert clean["evidence"]["status"] == "PASS"
    assert clean["matches"] == []
    assert clean["errors"] == []

    dirty_proc = tmp_path / "dirty-proc"
    dirty_proc.mkdir()
    native_nav2 = "/opt/ros/jazzy/lib/nav2_lifecycle_manager/lifecycle_manager"
    write_process(dirty_proc, 202, [native_nav2, "--ros-args"])
    dirty = audit(auditor_b64, auditor_sha256, str(dirty_proc))
    assert dirty["status"] == "FAIL"
    assert dirty["check"] is False
    assert dirty["evidence"]["status"] == "FAIL"
    assert dirty["matches"][0]["pid"] == 202
    assert dirty["matches"][0]["reason"].startswith("ros-native-package:nav2_")

    unreadable = audit(auditor_b64, auditor_sha256, str(tmp_path / "missing-proc"))
    assert unreadable["status"] == "FAIL"
    assert unreadable["check"] is False
    assert unreadable["evidence"]["status"] == "ERROR"
    assert unreadable["errors"] == [{"pid": None, "error": "proc_root_missing"}]

    malformed = audit("not-base64!", auditor_sha256, str(clean_proc))
    assert malformed["status"] == "FAIL"
    assert malformed["check"] is False
    assert malformed["evidence"]["status"] == "ERROR"
    assert malformed["errors"][0]["error"].startswith("auditor_load:")


def test_prepare_cleanup_never_signals_a_mixed_scope_process_group() -> None:
    text = runner_text()
    assert "len(associated_members) == len(members)" in text
    assert "mixed_or_unassociated_process_group" in text
    assert text.index("len(associated_members) == len(members)") < text.index(
        "os.killpg(group_id, requested_signal)"
    )


def test_three_host_remote_cleanup_audits_run_concurrently() -> None:
    text = runner_text()
    for variable in ("cleanup_a_pid", "cleanup_b_pid", "cleanup_x86_pid"):
        assert f"{variable}=$!" in text
    assert 'wait "$cleanup_a_pid" "$cleanup_b_pid" "$cleanup_x86_pid"' in text
    assert '"dgx_a_cleanup_pass"' not in text  # keys are generated, not trusted constants
    assert 'for host in ("dgx_a", "dgx_b", "x86")' in text
