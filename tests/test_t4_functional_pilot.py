from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "coordination"))
sys.path.insert(0, str(ROOT / "internvla_ros2"))

from t4_functional_run_contract import (  # noqa: E402
    ContractError,
    validate_pilot_collected,
    validate_pilot_result,
)
from internvla_ros2.identity import CHECKPOINT_REVISION, MODEL_REVISION  # noqa: E402


GRANT = "pilot1234"
AUTHORIZATION_REF = "b" * 40
DEPLOYMENT_REF = "a" * 40


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _pilot_result(
    root: Path, *, success_rate: float = 0.05, ready_sha256: str = "c" * 64
) -> Path:
    result = root / "pilot"
    result.mkdir()
    _write(
        result / "validation.json",
        {
            "schema_version": 2,
            "status": "PASS",
            "quality_status": "WARN",
            "runtime_policy": "completion_sim",
            "client_pose_ok": True,
            "required": {
                "episode_count": 20,
                "minimum_sr": 0.01,
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
                "physical_collision_count": 1,
            },
        },
    )
    _write(
        result / "phase_status.json",
        {
            "status": "PASS",
            "phase": "pilot",
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
    _write(result / "result.json", {"val_unseen": {"Count": 20, "SR": success_rate}})
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
    _write(result / "per_episode.json", {"completed_episode_count": 20})
    _write(
        result / "client_summary.json",
        {
            "status": "FINISHED",
            "step_count": 4,
            "model_health_start": {
                "status": "PASS",
                "status_code": 0,
                "initialized": False,
                "lifecycle_state": 0,
                "model_revision": MODEL_REVISION,
                "checkpoint_revision": CHECKPOINT_REVISION,
            },
        },
    )
    _write(
        result / "pilot_invocation.json",
        {
            "grant_id": GRANT,
            "authorization_ref_sha": AUTHORIZATION_REF,
            "deployment_ref_sha": DEPLOYMENT_REF,
            "model_ready_receipt_sha256": ready_sha256,
            "runtime_policy": "completion_sim",
            "runtime_target": "isaac_simulation",
            "resource_lease_ack": "dgx+isaac",
            "model_host": "dgx_spark_only",
            "model_process_started": True,
            "use_t4_adapter": True,
            "expected_episode_count": 20,
            "minimum_sr": 0.01,
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


def test_pilot_requires_nonzero_sr_and_keeps_fatal_safety_hard(tmp_path: Path) -> None:
    result = _pilot_result(tmp_path)
    payload = validate_pilot_result(result, GRANT, AUTHORIZATION_REF, DEPLOYMENT_REF)
    assert payload["status"] == "PASS"
    assert payload["success_rate"] == 0.05
    assert payload["quality_status"] == "WARN"

    _write(result / "result.json", {"val_unseen": {"Count": 20, "SR": 0.0}})
    with pytest.raises(ContractError, match="pilot result contract failed"):
        validate_pilot_result(result, GRANT, AUTHORIZATION_REF, DEPLOYMENT_REF)

    _write(result / "result.json", {"val_unseen": {"Count": 20, "SR": 0.05}})
    validation = json.loads((result / "validation.json").read_text(encoding="utf-8"))
    validation["safety"]["nan_count"] = 1
    _write(result / "validation.json", validation)
    with pytest.raises(ContractError, match="pilot result contract failed"):
        validate_pilot_result(result, GRANT, AUTHORIZATION_REF, DEPLOYMENT_REF)


def test_collected_pilot_binds_dgx_model_receipt_and_zero_residuals(
    tmp_path: Path,
) -> None:
    ready_path = tmp_path / "model_ready_receipt.json"
    _write(
        ready_path,
        {
            "status": "PASS",
            "grant_id": GRANT,
            "authorization_ref_sha": AUTHORIZATION_REF,
            "deployment_ref_sha": DEPLOYMENT_REF,
            "resource_lease_ack": "dgx+isaac",
            "backend": "real",
        },
    )
    ready_sha = hashlib.sha256(ready_path.read_bytes()).hexdigest()
    pilot = _pilot_result(tmp_path, ready_sha256=ready_sha)
    model = tmp_path / "model"
    model.mkdir()
    identity = {
        "model_revision": MODEL_REVISION,
        "checkpoint_revision": CHECKPOINT_REVISION,
    }
    _write(
        model / "model_weight_audit.json",
        {
            "status": "PASS",
            "backend": "real",
            "meta_parameter_count": 0,
            "meta_buffer_count": 0,
            **identity,
        },
    )
    _write(
        model / "model_health_ready.json",
        {"status": "PASS", "model_host": "dgx_spark", "step_action_ready": True, **identity},
    )
    _write(
        model / "model_invocation.json",
        {"grant_id": GRANT, "backend": "real", "preload_model": True},
    )
    _write(
        model / "model_residual_probe.json",
        {"status": "PASS", "pid_count": 0},
    )
    stop_path = tmp_path / "model_stop_receipt.json"
    _write(
        stop_path,
        {
            "status": "PASS",
            "model_was_alive": True,
            "process_group_remaining": 0,
            "residual_probe_exit_code": 0,
            "archive_exit_code": 0,
            "authorization_ref_sha": AUTHORIZATION_REF,
            "deployment_ref_sha": DEPLOYMENT_REF,
        },
    )
    pilot_receipt_path = tmp_path / "pilot_run_receipt.json"
    _write(
        pilot_receipt_path,
        {
            "status": "PASS",
            "authorization_ref_sha": AUTHORIZATION_REF,
            "deployment_ref_sha": DEPLOYMENT_REF,
            "run_exit_code": 0,
            "validation_exit_code": 0,
            "cleanup_probe_exit_code": 0,
            "residual_probe_exit_code": 0,
            "archive_exit_code": 0,
            "model_ready_receipt_sha256": ready_sha,
        },
    )

    output = tmp_path / "pilot20_summary.json"
    payload = validate_pilot_collected(
        pilot,
        model,
        ready_path,
        stop_path,
        pilot_receipt_path,
        output,
        GRANT,
        AUTHORIZATION_REF,
        DEPLOYMENT_REF,
        True,
    )
    assert payload["status"] == "PASS"
    assert payload["model_host"] == "dgx_spark_only"
    assert json.loads(output.read_text(encoding="utf-8"))["success_rate"] == 0.05

    stop = json.loads(stop_path.read_text(encoding="utf-8"))
    stop["process_group_remaining"] = 1
    _write(stop_path, stop)
    with pytest.raises(ContractError, match="collected pilot contract failed"):
        validate_pilot_collected(
            pilot,
            model,
            ready_path,
            stop_path,
            pilot_receipt_path,
            tmp_path / "must-not-exist.json",
            GRANT,
            AUTHORIZATION_REF,
            DEPLOYMENT_REF,
            True,
        )


def test_pilot_launchers_hold_combined_lease_and_start_dgx_first() -> None:
    coordinator = (ROOT / "coordination" / "run_t4_pilot_online.sh").read_text(
        encoding="utf-8"
    )
    model = (ROOT / "coordination" / "remote_t4_model_session.sh").read_text(
        encoding="utf-8"
    )
    pilot = (ROOT / "coordination" / "remote_t4_pilot_run.sh").read_text(
        encoding="utf-8"
    )
    assert '"profile":"functional_pilot20"' in coordinator
    assert 'with_resource_lease.sh" both' in coordinator
    assert "for role in dgx isaac" in coordinator
    assert coordinator.index("create_stage dgx") < coordinator.index("create_stage isaac")
    assert coordinator.index("remote_exec dgx \"$command_text\"") < coordinator.index(
        "remote_exec isaac \"$command_text\""
    )
    assert "cleanup_model" in coordinator
    assert "validate-pilot-collected" in coordinator
    assert "INTERNVLA_BACKEND=real" in model
    assert "INTERNVLA_PRELOAD_MODEL=1" in model
    assert "HF_ENDPOINT=https://hf-mirror.com" in model
    assert 'test -f "$deployment_root/scripts/run_t4_model_server.sh"' in model
    assert 'test -x "$deployment_root/scripts/run_t4_model_server.sh"' not in model
    assert "export PYTHONDONTWRITEBYTECODE=1" in model
    assert "model_prestart_nodes.txt" in model
    assert "model_prestart_processes.txt" in model
    assert "residual-probe --role dgx" in model
    assert "timeout --signal=TERM --kill-after=30s 14400s" in model
    assert "INTERNVLA_T4_EXPECTED_COUNT=20" in pilot
    assert "export PYTHONDONTWRITEBYTECODE=1" in pilot
    assert "INTERNVLA_T4_MIN_SR_OVERRIDE=0.01" in pilot
    assert "INTERNNAV_T4_RESOURCE_LEASE_ACK=dgx+isaac" in pilot
    assert 'test -f "$deployment_root/scripts/run_t4_sensor_gate.sh"' in pilot
    assert 'test -f "$deployment_root/scripts/t4_cleanup_container_processes.sh"' in pilot
    assert 'test -x "$deployment_root/scripts/run_t4_sensor_gate.sh"' not in pilot
    assert "validate-pilot-result" in pilot
    assert "residual-probe" in pilot
