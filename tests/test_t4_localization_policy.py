from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from t4_completion.localization.config import SelectorPolicy, load_config
from t4_completion.localization.contracts import (
    CANONICAL_ODOMETRY_TOPIC,
    CANONICAL_TF_TOPIC,
    ISAAC_GT,
)
from t4_completion.localization.ros_runtime import _health_attestation
from t4_completion.localization import cli
from t4_completion.localization.cli import (
    _normalize_oracle_result_dir,
    _parse_authoritative_grant,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/completion_sim/localization/selector.json"


def test_completion_config_is_explicit_isaac_only_and_frozen() -> None:
    config = load_config(
        CONFIG,
        expected_runtime_policy="completion_sim",
        expected_runtime_target="isaac_simulation",
    )
    assert config.policy.source_timeout_ns == 5_000_000_000
    assert config.policy.tf_tolerance_ns == 2_500_000_000
    assert config.policy.allow_isaac_gt_fallback is True
    assert config.allowed_targets == ("isaac_simulation",)
    assert set(config.forbidden_targets) == {"real_go2", "hardware_motion"}
    assert config.use_sim_time is True
    assert config.sources[ISAAC_GT].role == "fallback"
    assert all(
        source.odometry_topic != CANONICAL_ODOMETRY_TOPIC
        for source in config.sources.values()
    )
    assert CANONICAL_TF_TOPIC == "/tf"


@pytest.mark.parametrize("target", ["real_go2", "hardware_motion", "unknown"])
def test_completion_config_rejects_non_isaac_targets(target: str) -> None:
    with pytest.raises(ValueError):
        load_config(
            CONFIG,
            expected_runtime_policy="completion_sim",
            expected_runtime_target=target,
        )


def test_completion_config_requires_explicit_profile_not_legacy_default() -> None:
    with pytest.raises(ValueError, match="runtime_policy_mismatch"):
        load_config(
            CONFIG,
            expected_runtime_policy="strict_evidence",
            expected_runtime_target="isaac_simulation",
        )


def test_policy_constructor_enforces_strict_and_hardware_boundaries() -> None:
    strict = SelectorPolicy.strict_evidence()
    assert strict.source_timeout_ns == 350_000_000
    assert strict.tf_tolerance_ns == 0
    assert strict.allow_isaac_gt_fallback is False
    with pytest.raises(ValueError, match="requires_isaac"):
        SelectorPolicy(
            runtime_policy="completion_sim",
            runtime_target="real_go2",
            source_timeout_ns=5_000_000_000,
            tf_tolerance_ns=2_500_000_000,
            maximum_future_stamp_ns=1_000_000,
            preferred_recovery_hold_ns=1_000_000_000,
            minimum_dwell_ns=500_000_000,
            quaternion_norm_tolerance=0.001,
            allow_isaac_gt_fallback=True,
        )
    with pytest.raises(ValueError, match="forbids_gt_fallback"):
        SelectorPolicy(
            runtime_policy="strict_evidence",
            runtime_target="isaac_simulation",
            source_timeout_ns=350_000_000,
            tf_tolerance_ns=0,
            maximum_future_stamp_ns=0,
            preferred_recovery_hold_ns=1_000_000_000,
            minimum_dwell_ns=500_000_000,
            quaternion_norm_tolerance=0.001,
            allow_isaac_gt_fallback=True,
        )


def test_existing_strict_runtime_contract_remains_exact() -> None:
    text = (ROOT / "configs/runtime/strict_evidence.yaml").read_text(encoding="utf-8")
    for frozen in (
        "runtime_policy: strict_evidence",
        "mode: fatal_exact_batch",
        "watchdog_sec: 0.35",
        "lookup: exact",
        "transform_tolerance_sec: 0.0",
        "collision_monitor: enforce",
        "nvblox: active_required",
    ):
        assert frozen in text
    assert "isaac_ground_truth_pose" not in text


def test_health_attestation_is_exact_and_generation_bound() -> None:
    payload = {
        "schema_version": 1,
        "source": "cuvslam",
        "reset_generation": 2,
        "sequence_id": 7,
        "sample_stamp_ns": 123,
        "backend_ready": True,
        "tracking": True,
        "reason": "tracking",
    }
    assert _health_attestation(json.dumps(payload), "cuvslam") == payload
    payload["extra"] = "not allowed"
    with pytest.raises(ValueError, match="keys_mismatch"):
        _health_attestation(json.dumps(payload), "cuvslam")
    payload.pop("extra")
    payload["schema_version"] = True
    with pytest.raises(ValueError, match="identity_mismatch"):
        _health_attestation(json.dumps(payload), "cuvslam")


def test_optional_ros_boundary_does_not_import_ros_at_module_import_time() -> None:
    # Merely loading the core and adapter helpers works on the offline worker,
    # where rclpy/Isaac packages are intentionally absent.
    import t4_completion.localization as localization

    assert localization.LocalizationSelector is not None


def test_ros_runtime_injects_frozen_sim_time_without_cli_ros_args() -> None:
    source = (ROOT / "t4_completion/localization/ros_runtime.py").read_text(
        encoding="utf-8"
    )
    assert 'Parameter("use_sim_time", value=config.use_sim_time)' in source
    assert "rclpy.init(args=[])" in source


def test_oracle_result_path_is_exactly_grant_bound_and_repo_relative(
    tmp_path: Path,
) -> None:
    relative, resolved = _normalize_oracle_result_dir(
        tmp_path,
        Path("results/parallel/localization/oracle-grant_123"),
        "grant_123",
    )
    assert relative == "results/parallel/localization/oracle-grant_123"
    assert resolved == (
        tmp_path / "results/parallel/localization/oracle-grant_123"
    ).resolve()
    for invalid in (
        Path("results/parallel/localization/not-the-grant"),
        Path("results/parallel/localization/oracle-other"),
        tmp_path / "results/parallel/localization/oracle-grant_123",
        Path("../oracle-grant_123"),
    ):
        with pytest.raises(ValueError):
            _normalize_oracle_result_dir(tmp_path, invalid, "grant_123")


def test_authoritative_grant_parser_requires_one_exact_block() -> None:
    grant = {
        "schema_version": 1,
        "status": "GRANTED",
        "worker": "20",
        "resource": "isaac",
        "profile": "completion_sim_localization_oracle",
        "result_dir": "results/parallel/localization/oracle-g",
        "grant_id": "g",
    }
    board = (
        "before\n<!-- INTERNAV_ONLINE_GRANT_V1\n"
        + json.dumps(grant)
        + "\nINTERNAV_ONLINE_GRANT_V1 -->\nafter\n"
    )
    assert _parse_authoritative_grant(board) == grant
    with pytest.raises(RuntimeError, match="unique"):
        _parse_authoritative_grant(board + board)
    duplicate = (
        "<!-- INTERNAV_ONLINE_GRANT_V1\n"
        '{"status":"NO_GRANT","status":"GRANTED"}'
        "\nINTERNAV_ONLINE_GRANT_V1 -->\n"
    )
    with pytest.raises(RuntimeError, match="duplicate_key"):
        _parse_authoritative_grant(duplicate)


def test_oracle_preflight_claim_and_ref_check_are_grant_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    grant_id = "offline_grant_20"
    relative = Path(f"results/parallel/localization/oracle-{grant_id}")
    sha = "a" * 40
    grant = {
        "schema_version": 1,
        "status": "GRANTED",
        "worker": "20",
        "resource": "isaac",
        "profile": "completion_sim_localization_oracle",
        "result_dir": relative.as_posix(),
        "grant_id": grant_id,
    }
    board = (
        "<!-- INTERNAV_ONLINE_GRANT_V1\n"
        + json.dumps(grant)
        + "\nINTERNAV_ONLINE_GRANT_V1 -->\n"
    )

    def fake_git(_root: Path, *arguments: str) -> str:
        if arguments == ("rev-parse", "--show-toplevel"):
            return str(tmp_path)
        if arguments == ("rev-parse", "--verify", cli.INTEGRATION_REF):
            return sha
        if arguments == ("rev-parse", "HEAD"):
            return sha
        if arguments == ("status", "--porcelain", "--untracked-files=no"):
            return ""
        if arguments[:2] == ("cat-file", "-e"):
            return ""
        if arguments == ("show", f"{sha}:coordination/TASK_BOARD.md"):
            return board
        raise AssertionError(arguments)

    monkeypatch.setattr(cli, "_git", fake_git)
    arguments = argparse.Namespace(
        repository_root=tmp_path,
        result_dir=relative,
        grant_id=grant_id,
    )
    assert cli._oracle_preflight(arguments) == 0
    claim_path = tmp_path / relative / "localization_grant_claim.json"
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    assert claim["status"] == "PASS"
    assert claim["integration_ref_sha"] == sha
    assert claim["execution_head_sha"] == sha
    assert claim["tracked_runtime_paths"] == list(cli.ORACLE_TRACKED_PATHS)
    assert cli._oracle_ref_check(
        argparse.Namespace(repository_root=tmp_path, result_dir=relative)
    ) == 0
    with pytest.raises(FileExistsError, match="fresh"):
        cli._oracle_preflight(arguments)


def test_oracle_preflight_retains_fail_claim_when_ref_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    grant_id = "changing_ref"
    relative = Path(f"results/parallel/localization/oracle-{grant_id}")
    first_sha = "b" * 40
    second_sha = "c" * 40
    grant = {
        "schema_version": 1,
        "status": "GRANTED",
        "worker": "20",
        "resource": "isaac",
        "profile": "completion_sim_localization_oracle",
        "result_dir": relative.as_posix(),
        "grant_id": grant_id,
    }
    ref_reads = 0

    def fake_git(_root: Path, *arguments: str) -> str:
        nonlocal ref_reads
        if arguments == ("rev-parse", "--show-toplevel"):
            return str(tmp_path)
        if arguments == ("rev-parse", "--verify", cli.INTEGRATION_REF):
            ref_reads += 1
            return first_sha if ref_reads == 1 else second_sha
        if arguments == ("rev-parse", "HEAD"):
            return first_sha
        if arguments == ("status", "--porcelain", "--untracked-files=no"):
            return ""
        if arguments[:2] == ("cat-file", "-e"):
            return ""
        if arguments == ("show", f"{first_sha}:coordination/TASK_BOARD.md"):
            return (
                "<!-- INTERNAV_ONLINE_GRANT_V1\n"
                + json.dumps(grant)
                + "\nINTERNAV_ONLINE_GRANT_V1 -->\n"
            )
        raise AssertionError(arguments)

    monkeypatch.setattr(cli, "_git", fake_git)
    arguments = argparse.Namespace(
        repository_root=tmp_path,
        result_dir=relative,
        grant_id=grant_id,
    )
    with pytest.raises(RuntimeError, match="changed_during_result_claim"):
        cli._oracle_preflight(arguments)
    claim = json.loads(
        (tmp_path / relative / "localization_grant_claim.json").read_text(
            encoding="utf-8"
        )
    )
    assert claim["status"] == "FAIL"
    assert claim["ref_unchanged_after_claim"] is False


def test_oracle_preflight_rejects_boolean_grant_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    grant_id = "bad_schema"
    relative = Path(f"results/parallel/localization/oracle-{grant_id}")
    sha = "d" * 40
    grant = {
        "schema_version": True,
        "status": "GRANTED",
        "worker": "20",
        "resource": "isaac",
        "profile": "completion_sim_localization_oracle",
        "result_dir": relative.as_posix(),
        "grant_id": grant_id,
    }

    def fake_git(_root: Path, *arguments: str) -> str:
        if arguments == ("rev-parse", "--show-toplevel"):
            return str(tmp_path)
        if arguments == ("rev-parse", "--verify", cli.INTEGRATION_REF):
            return sha
        if arguments == ("show", f"{sha}:coordination/TASK_BOARD.md"):
            return (
                "<!-- INTERNAV_ONLINE_GRANT_V1\n"
                + json.dumps(grant)
                + "\nINTERNAV_ONLINE_GRANT_V1 -->\n"
            )
        raise AssertionError(arguments)

    monkeypatch.setattr(cli, "_git", fake_git)
    with pytest.raises(RuntimeError, match="field_types_invalid"):
        cli._oracle_preflight(
            argparse.Namespace(
                repository_root=tmp_path,
                result_dir=relative,
                grant_id=grant_id,
            )
        )
    assert not (tmp_path / relative).exists()
