from __future__ import annotations

import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "coordination"))

import t4_functional_run_contract as run_contract  # noqa: E402

from t4_functional_run_contract import (  # noqa: E402
    ContractError,
    extract_oracle_archive,
    validate_deployment,
    validate_oracle_result,
    validate_prepare_result,
)


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


@pytest.mark.parametrize("role", ["dgx", "isaac"])
def test_parallel_lane_b_deployment_receipt_is_explicit_and_role_correct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: str
) -> None:
    stable = tmp_path / role
    workspace = (tmp_path / "isaac_ros_ws_45").as_posix()
    monkeypatch.setattr(run_contract, "DGX_STABLE_ROOT", stable.as_posix())
    monkeypatch.setattr(run_contract, "ISAAC_STABLE_ROOT", stable.as_posix())
    monkeypatch.setattr(run_contract, "ISAAC_ROS_WORKSPACE", workspace)
    grant = "apg20260719t000033"
    ref = "a" * 40
    root = stable / ".t4-deployments" / f"{grant}-{ref[:12]}-{role}-b"
    root.mkdir(parents=True)
    _write(
        root / "deployment_receipt.json",
        {
            "schema_version": 1,
            "status": "PASS",
            "phase": "ready",
            "exit_code": 0,
            "role": role,
            "grant_id": grant,
            "ref_sha": ref,
            "deployment_root": root.as_posix(),
            "ros_workspace": workspace if role == "isaac" else (root / "ros_ws").as_posix(),
            "parallel_lane_clone": "b",
            "runtime_policy": "completion_sim",
            "runtime_target": "isaac_simulation",
            "resource_lease_ack": "dgx+isaac",
            "strict_evidence_modified": False,
            "real_go2_targeted": False,
        },
    )
    _write(
        root / "payload_manifest.json",
        {
            "status": "FUNCTIONAL_PAYLOAD_READY",
            "ref_sha": ref,
            "runtime_policy": "completion_sim",
            "runtime_target": "isaac_simulation",
            "model_host": "dgx_spark_only",
            "strict_evidence_modified": False,
            "real_go2_targeted": False,
        },
    )
    result = validate_deployment(root, ref, role)
    assert result["status"] == "PASS"
    assert result["parallel_lane_clone"] == "b"


def _oracle_result(tmp_path: Path, *, collisions: int = 2) -> Path:
    result = tmp_path / "run"
    result.mkdir()
    quality = "WARN" if collisions else "PASS"
    _write(
        result / "validation.json",
        {
            "schema_version": 2,
            "status": "PASS",
            "quality_status": quality,
            "runtime_policy": "completion_sim",
            "required": {
                "episode_count": 10,
                "minimum_sr": 0.8,
                "map_source": "static_map",
                "pose_source": "ground_truth",
                "fatal_safety_zero": [
                    "fall_count",
                    "nan_count",
                    "stale_motion_execution_count",
                ],
                "physical_collision_policy": "metric_only_warn",
            },
            "safety": {
                "fall_count": 0,
                "nan_count": 0,
                "stale_motion_execution_count": 0,
                "physical_collision_count": collisions,
                "identity_safe_stop_count": 0,
            },
        },
    )
    _write(
        result / "phase_status.json",
        {
            "status": "PASS",
            "phase": "obstacle_oracle",
            "exit_codes": {"preflight": 0, "evaluator": 0, "validation": 0},
        },
    )
    _write(
        result / "t4_run_contract.json",
        {
            "runtime_policy": "completion_sim",
            "runtime_target": "isaac",
            "map_source": "static_map",
            "pose_source": "ground_truth",
            "recovery_enabled": False,
            "mapping_and_safety_geometry": "go2_frozen",
        },
    )
    _write(result / "result.json", {"val_unseen": {"Count": 10, "SR": 0.9}})
    _write(result / "active_summary.json", {"status": "FINISHED", "failure_count": 0})
    _write(
        result / "controller_summary.json",
        {
            "status": "FINISHED",
            "map_source": "static_map",
            "pose_source": "ground_truth",
            "ground_truth_pose_used_for_nav": True,
        },
    )
    _write(result / "per_episode.json", {"completed_episode_count": 10})
    _write(
        result / "oracle_invocation.json",
        {
            "grant_id": "oracle123",
            "authorization_ref_sha": "b" * 40,
            "deployment_ref_sha": "a" * 40,
            "runtime_policy": "completion_sim",
            "runtime_target": "isaac_simulation",
            "resource_lease_ack": "isaac",
            "model_host": "dgx_spark_only",
            "model_process_started": False,
            "use_t4_adapter": True,
            "expected_episode_count": 10,
            "minimum_sr": 0.8,
            "strict_evidence_modified": False,
            "real_go2_targeted": False,
        },
    )
    (result / "active_records.jsonl").write_text(
        json.dumps(
            {
                "event": "ablation_runtime_summary",
                "variant_id": "none",
                "factors": {
                    "system_mode": "full_system1_system2",
                    "trajectory_mode": "full_trajectory",
                    "termination_mode": "model_stop",
                    "history_mode": "on",
                    "recovery_mode": "off",
                    "view_mode": "go2_view",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def test_oracle_contract_allows_collision_warn_without_weakening_fatal_safety(
    tmp_path: Path,
) -> None:
    result = _oracle_result(tmp_path)

    payload = validate_oracle_result(result, "oracle123", "b" * 40, "a" * 40)

    assert payload["status"] == "PASS"
    assert payload["quality_status"] == "WARN"
    assert payload["physical_collision_count"] == 2

    validation = json.loads((result / "validation.json").read_text(encoding="utf-8"))
    validation["safety"]["fall_count"] = 1
    _write(result / "validation.json", validation)
    with pytest.raises(ContractError, match="Oracle result contract failed"):
        validate_oracle_result(result, "oracle123", "b" * 40, "a" * 40)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _prepare_fixture(tmp_path: Path) -> tuple[Path, Path, str, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Contract Test")
    (repo / "coordination").mkdir()
    (repo / "scripts").mkdir()
    (repo / "coordination" / "TASK_BOARD.md").write_text("NO_GRANT\n", encoding="utf-8")
    (repo / "scripts" / "runtime.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "deployment")
    deployment_ref = _git(repo, "rev-parse", "HEAD")
    (repo / "coordination" / "TASK_BOARD.md").write_text("GRANTED\n", encoding="utf-8")
    _git(repo, "add", "coordination/TASK_BOARD.md")
    _git(repo, "commit", "-m", "authorization")
    authorization_ref = _git(repo, "rev-parse", "HEAD")

    grant = "prepare123"
    result = tmp_path / f"prepare-00-{grant}"
    receipts: dict[str, dict[str, object]] = {}
    for role, stable_root in (
        ("dgx", "/home/railgun/internnav-t1-t2"),
        ("isaac", "/home/song/internnav-t1-t2"),
    ):
        receipt: dict[str, object] = {
            "schema_version": 1,
            "role": role,
            "status": "PASS",
            "phase": "ready",
            "exit_code": 0,
            "grant_id": grant,
            "ref_sha": deployment_ref,
            "deployment_root": (
                f"{stable_root}/.t4-deployments/"
                f"{grant}-{deployment_ref[:12]}-{role}"
            ),
            "ros_workspace": (
                "/home/song/internnav-t4/isaac_ros_ws_45"
                if role == "isaac"
                else f"{stable_root}/.t4-deployments/{grant}-{deployment_ref[:12]}-{role}/ros_ws"
            ),
            "runtime_policy": "completion_sim",
            "runtime_target": "isaac_simulation",
            "resource_lease_ack": "dgx+isaac",
            "strict_evidence_modified": False,
            "real_go2_targeted": False,
        }
        receipts[role] = receipt
        _write(result / "remote-receipts" / f"{role}_deployment_receipt.json", receipt)
    _write(
        result / "functional_prepare_summary.json",
        {
            "status": "PASS",
            "exit_code": 0,
            "grant_id": grant,
            "ref_sha": deployment_ref,
            "resource_order": ["dgx", "isaac"],
            "runtime_policy": "completion_sim",
            "runtime_target": "isaac_simulation",
            "model_host": "dgx_spark_only",
            "strict_evidence_modified": False,
            "real_go2_targeted": False,
            "receipts": receipts,
        },
    )
    return result, repo / ".git", deployment_ref, authorization_ref


def test_prepare_contract_allows_only_task_board_change_after_deployment(
    tmp_path: Path,
) -> None:
    result, git_dir, deployment_ref, authorization_ref = _prepare_fixture(tmp_path)

    payload = validate_prepare_result(result, git_dir, authorization_ref)

    assert payload["deployment_ref_sha"] == deployment_ref
    assert payload["allowed_post_deployment_changes"] == [
        "coordination/TASK_BOARD.md"
    ]

    repo = git_dir.parent
    (repo / "scripts" / "runtime.sh").write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    _git(repo, "add", "scripts/runtime.sh")
    _git(repo, "commit", "-m", "production drift")
    drift_ref = _git(repo, "rev-parse", "HEAD")
    with pytest.raises(ContractError, match="restaging is required"):
        validate_prepare_result(result, git_dir, drift_ref)


def test_oracle_archive_extraction_rejects_links(tmp_path: Path) -> None:
    good = tmp_path / "good.tgz"
    data = b"{}\n"
    with tarfile.open(good, "w:gz") as archive:
        root = tarfile.TarInfo(".")
        root.type = tarfile.DIRTYPE
        archive.addfile(root)
        info = tarfile.TarInfo("./validation.json")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    output = tmp_path / "output"
    output.mkdir()
    payload = extract_oracle_archive(good, output)
    assert payload["status"] == "PASS"
    assert (output / "validation.json").read_bytes() == data

    bad = tmp_path / "bad.tgz"
    with tarfile.open(bad, "w:gz") as archive:
        info = tarfile.TarInfo("escape")
        info.type = tarfile.SYMTYPE
        info.linkname = "../outside"
        archive.addfile(info)
    bad_output = tmp_path / "bad-output"
    bad_output.mkdir()
    with pytest.raises(ContractError, match="unsafe Oracle archive member"):
        extract_oracle_archive(bad, bad_output)


def test_oracle_launchers_are_single_isaac_lease_and_model_free() -> None:
    coordinator = (ROOT / "coordination" / "run_t4_oracle_online.sh").read_text(
        encoding="utf-8"
    )
    remote = (ROOT / "coordination" / "remote_t4_oracle_run.sh").read_text(
        encoding="utf-8"
    )
    assert '"profile":"functional_oracle10"' in coordinator
    assert 'with_resource_lease.sh" isaac' in coordinator
    assert 'with_resource_lease.sh" both' not in coordinator
    assert "functional code changed after deployment" in (
        ROOT / "coordination" / "t4_functional_run_contract.py"
    ).read_text(encoding="utf-8")
    assert "INTERNVLA_T4_EXPECTED_COUNT=10" in remote
    assert "INTERNVLA_T4_MIN_SR_OVERRIDE=0.8" in remote
    assert "INTERNVLA_T4_USE_T4_ADAPTER=1" in remote
    assert "export PYTHONDONTWRITEBYTECODE=1" in remote
    assert "INTERNNAV_T4_RESOURCE_LEASE_ACK=isaac" in remote
    assert '"model_process_started":False' in remote
    assert "run_t4_model_server" not in remote
    assert "DGX_PASSWORD" not in coordinator
    assert "raw_run_exit_code" in remote
    assert "residual-probe" in remote
    assert 'test -f "$deployment_root/scripts/run_t4_sensor_gate.sh"' in remote
    assert 'test -f "$deployment_root/scripts/t4_cleanup_container_processes.sh"' in remote
    assert 'test -x "$deployment_root/scripts/run_t4_sensor_gate.sh"' not in remote
    assert "t4_functional_payload.py\" verify-tree" in remote
    assert "validate-overlay-source" in remote
    assert "oracle_result.tgz" in coordinator
