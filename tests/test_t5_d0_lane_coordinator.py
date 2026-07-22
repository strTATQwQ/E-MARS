from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "coordination" / "run_t5_d0_lane_online.sh"


def source() -> str:
    return RUNNER.read_text(encoding="utf-8")


def embedded_python(containing: str) -> str:
    lines = source().splitlines()
    for index, line in enumerate(lines):
        if "<<'PY'" not in line:
            continue
        end = index + 1
        while end < len(lines) and lines[end] != "PY":
            end += 1
        assert end < len(lines), f"unclosed PY heredoc at {index + 1}"
        program = "\n".join(lines[index + 1 : end]) + "\n"
        if containing in program:
            return program
    raise AssertionError(f"no embedded Python block contains {containing!r}")


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_embedded_python_blocks_compile() -> None:
    lines = source().splitlines()
    compiled = 0
    for marker in ("PY", "REMOTE_AUDIT", "REMOTE_SUPERVISOR_CONTROL"):
        index = 0
        while index < len(lines):
            if f"<<'{marker}'" not in lines[index]:
                index += 1
                continue
            end = index + 1
            while end < len(lines) and lines[end] != marker:
                end += 1
            assert end < len(lines), f"unclosed {marker} heredoc at {index + 1}"
            compile(
                "\n".join(lines[index + 1 : end]) + "\n",
                f"{RUNNER}:{index + 1}",
                "exec",
            )
            compiled += 1
            index = end + 1
    assert compiled >= 15


def test_exact_board_only_grant_and_frozen_hashes_are_required() -> None:
    text = source()
    assert "INTERNAV_T5_DUAL_LANE_ONLINE_GRANT_V1" in text
    assert 'rev-list --count "$authorization_ref..HEAD"' in text
    assert 'diff --name-only "$authorization_ref" HEAD' in text
    assert "status --porcelain --untracked-files=all" in text
    assert "grant commit changed board content outside the grant block" in text
    assert '"schema_version": 2' in text
    assert '"code_ref_sha": code_ref' in text
    assert 'show "$code_ref:$golden_relative"' in text
    assert 'show "$code_ref:$d0_manifest_relative"' in text
    assert 'merge-base --is-ancestor "$code_ref" "$authorization_ref"' in text
    assert 'diff --name-only "$code_ref" "$authorization_ref"' in text
    assert '[[ "$changed_path" = "$board_relative" ]]' in text
    assert '"candidate_id": golden["bundle_id"]' in text
    assert '"golden_bundle_sha256": golden_sha' in text
    assert '"run_manifest_sha256": d0_sha' in text
    assert 'show "$authorization_ref:$golden_relative"' not in text
    assert 'show "$authorization_ref:$d0_manifest_relative"' not in text


def test_stage_a_and_b_bind_exact_scope_profile_and_result_root() -> None:
    text = source()
    for fragment in (
        "stage=d0_1_lane_a_fixed_5",
        "lane_scope=lane_a",
        "resource_profile=lane-a",
        "result_prefix=d0-1-lane-a",
        "stage=d0_2_lane_b_same_fixed_5",
        "lane_scope=lane_b",
        "resource_profile=lane-b",
        "result_prefix=d0-2-lane-b",
    ):
        assert fragment in text
    assert 'with_resource_lease.sh" "$resource_profile"' in text
    assert 'INTERNNAV_T5_RESOURCE_LEASE_ACK="$resource_profile"' in text
    assert "--cleanup-timeout 600 --kill-wait-timeout 60" in text
    assert "dgx_supervisor_ledger" in text
    assert "x86_supervisor_ledger" in text
    assert "exec setsid --wait bash" in text
    assert "coordinator_cleanup_receipt.json" in text
    assert "supervisor_action" in text
    assert "supervisor_absent" in text


def test_prepare_receipt_is_same_ref_golden_manifest_and_lane_roots() -> None:
    text = source()
    assert "d0_prepare_summary.json" in text
    assert "lease_release_summary.json" in text
    assert '"same_code_ref"' in text
    assert '"same_golden"' in text
    assert '"same_run_manifest"' in text
    assert 'required = {"dgx_a", "dgx_b", "x86_a", "x86_b"}' in text
    assert 'roots.get("x86_a") != roots.get("x86_b")' in text
    assert "static_map_manifest_sha256" in text
    assert "dataset_keys" in text
    assert "prepare_only" in text
    assert "d0_prepare_final_summary.json" in text
    assert "predecessor_binding.json" in text
    assert "predecessor_receipt_sha256" in text
    assert "d0_1_lane_a_fixed_5" in text
    assert "d0_lane_summary.json" in text
    assert 'prep.get("authorization_mode")' in text
    assert '== "FORMAL_BOARD_GRANT_V2"' in text
    assert 'prep_final.get("authorization_mode") == "FORMAL_BOARD_GRANT_V2"' in text
    assert 'prep.get("board_grant_used") is True' in text
    assert 'prep_final.get("board_grant_used") is True' in text
    assert 'prep_final.get("grant_id") == prep.get("grant_id")' in text
    assert 'prep_final.get("authorization_ref_sha")' in text


def test_formal_lane_rejects_fast_prepare_authorization_receipts() -> None:
    # Execute the exact validator embedded in the formal coordinator.  A
    # structurally valid fast-path receipt must fail solely on authorization.
    program = embedded_python("formal_authorization_mode")
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as directory_name:
        directory = Path(directory_name)
        code_sha, golden_sha, run_sha = "a" * 40, "b" * 64, "c" * 64
        dataset_sha, map_sha = "d" * 64, "e" * 64
        episode_keys = [f"episode-{index}" for index in range(5)]
        grant = directory / "grant.json"
        prep = directory / "prep.json"
        release = directory / "release.json"
        dataset = directory / "dataset.json"
        prep_final = directory / "prep-final.json"
        output = directory / "output.json"
        write_json(
            grant,
            {
                "golden_bundle_id": "golden-v0",
                "dataset_root": "/datasets/internnav",
                "dataset_file_sha256": dataset_sha,
                "episode_keys": episode_keys,
                "grant": {
                    "authorization_ref_sha": code_sha,
                    "code_ref_sha": code_sha,
                    "golden_bundle_sha256": golden_sha,
                    "run_manifest_sha256": run_sha,
                },
            },
        )
        formal_prep = {
            "status": "PASS",
            "stage": "d0_0_online_prepare",
            "authorization_mode": "FORMAL_BOARD_GRANT_V2",
            "board_grant_used": True,
            "grant_id": "t5d0020260719t120000",
            "authorization_ref_sha": code_sha,
            "code_ref_sha": code_sha,
            "golden_bundle_canonical_sha256": golden_sha,
            "run_manifest_canonical_sha256": run_sha,
            "golden_bundle_id": "golden-v0",
            "checks": {"prepared": True},
            "execution": {
                "internvla_model_loaded": False,
                "isaac_sim_started": False,
                "episode_or_evaluator_started": False,
            },
            "deployment_roots": {
                "dgx_a": "/home/railgun/internnav-t1-t2/.t5-deployments/formal-a",
                "dgx_b": "/home/rail/internnav-t1-t2/.t5-deployments/formal-b",
                "x86_a": "/home/song/internnav-t1-t2/.t5-deployments/formal-a",
                "x86_b": "/home/song/internnav-t1-t2/.t5-deployments/formal-b",
            },
            "static_map_manifest_sha256": map_sha,
        }
        write_json(prep, formal_prep)
        write_json(
            release,
            {"status": "PASS", "command_exit": 0, "checks": {"released": True}},
        )
        write_json(
            dataset,
            {
                "status": "PASS",
                "dataset_sha256": dataset_sha,
                "episode_count": 5,
                "episode_keys": episode_keys,
                "dataset_file": "/datasets/internnav/val_unseen/val_unseen.json.gz",
            },
        )
        formal_final = {
            "status": "PASS",
            "authorization_mode": "FORMAL_BOARD_GRANT_V2",
            "board_grant_used": True,
            "grant_id": formal_prep["grant_id"],
            "authorization_ref_sha": code_sha,
            "code_ref_sha": code_sha,
            "checks": {"finalized": True},
        }
        write_json(prep_final, formal_final)
        command = [
            sys.executable,
            "-c",
            program,
            str(grant),
            str(prep),
            str(release),
            str(dataset),
            str(prep_final),
            "a",
            str(output),
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr
        assert json.loads(output.read_text(encoding="utf-8"))["status"] == "PASS"

        formal_prep["authorization_mode"] = "FAST_EXACT_REF_NO_BOARD"
        formal_prep["board_grant_used"] = False
        formal_final["authorization_mode"] = "FAST_EXACT_REF_NO_BOARD"
        formal_final["board_grant_used"] = False
        write_json(prep, formal_prep)
        write_json(prep_final, formal_final)
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        assert completed.returncode != 0
        rejected = json.loads(output.read_text(encoding="utf-8"))
        assert rejected["status"] == "FAIL"
        assert rejected["checks"]["formal_authorization_mode"] is False
        assert rejected["checks"]["formal_board_grant_used"] is False


def test_result_root_is_atomically_consumed_and_never_overwritten() -> None:
    text = source()
    assert 'test ! -e "$result_dir"' in text
    assert 'mkdir "$result_dir"' in text
    assert 'with output.open("x"' in text
    assert "D0 Lane did not consume result_root" in text
    assert "lease_release_summary.json" in text
    assert "two_holders_recorded" in text
    assert "exact_resources_named" in text


def test_hf_token_only_enters_dgx_over_stdin() -> None:
    text = source()
    assert 'source "$credentials"' in text
    assert 'HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"' in text
    assert "IFS= read -r HF_TOKEN" in text
    assert "IFS= read -r HF_ENDPOINT" in text
    assert 'printf \'%s\\n%s\\n\' "$HF_TOKEN" "$HF_ENDPOINT" |' in text
    assert 'exec {hf_token_fd}<<<"$HF_TOKEN"' in text
    assert "unset HF_TOKEN" in text
    assert 'INTERNVLA_HF_TOKEN_FD="$hf_token_fd"' in text
    # Token must not be interpolated into the remote command string or lease argv.
    command_line = next(line for line in text.splitlines() if line.startswith("dgx_command="))
    assert "HF_TOKEN" not in command_line
    assert "HF_TOKEN=hf_" not in text
    assert "hf_token_process_audit.json" in text
    assert 'a[\\"status\\"]==\\"PASS\\"' in text
    assert 'a[\\"exact_secret_match_count\\"]>=1' in text
    assert 'a[\\"disallowed_match_count\\"]==0' in text


def test_d0_summary_requires_fail_closed_token_process_scope() -> None:
    text = source()
    assert "token_process_audit=load('remote/dgx/hf_token_process_audit.json')" in text
    assert "'credential_process_scope':token_process_audit.get('status')=='PASS'" in text
    assert "token_process_audit.get('allowed_model_match_count')" in text
    assert "token_process_audit.get('parent_match_count')==0" in text
    assert "token_process_audit.get('onboard_match_count')==0" in text
    assert "token_process_audit.get('evaluator_match_count')==0" in text
    assert "token_process_audit.get('secret_value_recorded') is False" in text
    assert "token_process_audit.get('secret_digest_recorded') is False" in text
    assert "result/'remote/dgx/hf_token_process_audit.json'" in text
    # Existing collected-artifact exact-secret scanning remains mandatory.
    assert "credential_exposure_audit.json" in text
    assert "secret in path.read_bytes()" in text


def test_complete_dgx_and_isaac_processes_overlap_and_are_linked() -> None:
    text = source()
    assert 'run_t5_dgx_lane.sh"' in text
    assert 'run_t5_distributed_isaac.sh"' in text
    assert '"$lane" model "$result_root"' in text
    assert 'dgx_ssh_pid=$!' in text
    assert 'x86_ssh_pid=$!' in text
    assert "DGX lane failed while Isaac was running" in text
    assert "D0 fixed-five runtime timed out" in text
    assert "request_linked_stop" in text
    assert 'touch \'$dgx_run/stop.request\'' in text
    assert "remote_failure_dgx" in text
    assert "remote_failure_x86" in text
    assert "d0_lane_failure.json" in text
    assert "INTERNNAV_T5_ENGINEERING_CANARY_SEC=0" in text
    assert "'fixed_dataset_execution':" in text
    assert "x86_status.get('execution_profile')=='fixed_dataset'" in text


def test_readiness_cleanup_and_residuals_are_fail_closed() -> None:
    text = source()
    assert (
        'local phase="$1"\n'
        '  local raw="$result_dir/audits/container_${phase}_inspect.json"'
    ) in text
    assert 'local phase="$1" raw=' not in text
    for readiness in ("dgx_ready", "x86_ready"):
        assert f"{readiness}=0" in text
        assert f"{readiness}=1" in text
        assert f'test "${readiness}" = 1' in text
    assert "lane_ready.json" in text
    assert "health/ready_probe.json" in text
    assert "docker stop -t 20" in text
    assert "capture_container_inspect prestart" in text
    assert "capture_container_inspect poststop" in text
    assert "capture_container_inspect failure_cleanup" in text
    assert "container_id" in text
    assert "container_lifecycle" in text
    assert "residual_host_audit.json" in text
    assert "lane_runtime_lock_free" in text
    assert "both_containers_stopped" in text
    assert "run_process_count_zero" in text
    assert "lane_socket_count_zero" in text
    assert "pid_pgid_ledgers_clean" in text
    assert "verified_absent" in text
    assert "clock_publishers_after_stop" in text
    assert "socket_residual_count" in text
    assert "shared_asset_lock_fd_released" in text


def test_lane_persistent_quarantine_brackets_every_online_workload() -> None:
    text = source()
    assert 'source "$root/scripts/t5_quarantine_common.sh"' in text
    assert 'source "$root/scripts/t5_remote_compute_audit_common.sh"' in text
    assert "dgx_quarantine_armed=false" in text
    assert "x86_quarantine_armed=false" in text
    assert 'quarantine_run_tag="${stage}:${grant_id}"' in text
    assert 'isaac_lane_quarantine_file="/tmp/internnav_isaac_gpu${gpu}.quarantine"' in text
    assert 'x86_quarantine_role="x86_gpu${gpu}"' in text
    assert "t5_quarantine_arm \"$dgx_target\"" in text
    assert "t5_quarantine_arm \"$x86_target\"" in text
    assert text.index('t5_quarantine_arm "$x86_target"') < text.index(
        'dgx_command="exec setsid'
    )
    prestart = 'dgx_structured_compute_prestart.json'
    assert prestart in text
    assert text.index('t5_quarantine_arm "$x86_target"') < text.index(prestart)
    assert text.index(prestart) < text.index('dgx_command="exec setsid')
    assert text.index('t5_quarantine_arm "$x86_target"') < text.index(
        'docker start \'$container\''
    )
    assert "t5_quarantine_owned_clear" in text
    assert text.index("request_linked_stop || incoming=1") < text.index(
        't5_quarantine_owned_clear "$dgx_target"'
    )
    assert "x86_containers_sockets_ports_runtime_locks_clean" in text
    assert "dgx_t5_ports_absent" in text
    assert "dgx_structured_compute_absent" in text
    assert "dgx_structured_compute_poststop.json" in text
    assert "run_root_processes_absent" in text
    assert "runtime_ledger_error is None" in text
    assert "isaac_container_stopped_and_owned" in text
    assert "/tmp/internnav_isaac.quarantine" not in text


def test_fixed_five_reports_metrics_without_success_threshold() -> None:
    text = source()
    assert "fixed_five_completed" in text
    for metric in ("SR", "OS", "SPL", "NE"):
        assert f"'{metric}'" in text
    assert "sr_report_only" in text
    assert "minimum_sr':None" in text
    assert "D0 has intentionally no minimum-SR gate" in text
    assert "required['SR']>=" not in text


def test_gpu_kit_and_cross_lane_isolation_are_audited() -> None:
    text = source()
    assert "gpu_mapping.json" in text
    assert "kit_gpu_audit.json" in text
    assert "kit_gpu_audit.get('status')=='PASS'" in text
    assert "kit_active_gpu_log_audit_required" in text
    assert "isaac_render_gpu_physical_index" in text
    assert "isaac_physics_gpu_visible_index" in text
    assert "kit_log_files" in text
    assert "no_other_lane_identity" in text
    assert "assert_other_lane_quiet" in text
    assert "next_cross_lane_audit" in text
    assert "other_container" in text
    assert "cross_lane_observation_count" in text
