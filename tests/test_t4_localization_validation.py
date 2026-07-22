from __future__ import annotations

import json
from pathlib import Path

import pytest

from t4_completion.localization.contracts import CUVSLAM, ISAAC_GT, LIDAR_IMU
from t4_completion.localization.validation import (
    _record_aggregates,
    validate_oracle_result,
    write_validation,
)


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "configs/completion_sim/localization/oracle_gate.json"


def write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def fixture_result(root: Path, *, with_valid_deviation: bool = True) -> None:
    deviation = {
        "schema_version": 1,
        "severity": "WARN",
        "code": "ODOMETRY_DEFERRED_ISAAC_GT_FALLBACK",
        "runtime_policy": "completion_sim",
        "runtime_target": "isaac_simulation",
        "fallback_source": ISAAC_GT,
        "sensor_odometry_status": "STRICT_EXTENSION_PENDING",
        "sensor_sources": {
            CUVSLAM: {
                "state": "DEFERRED",
                "reason": "stereo_feed_and_backend_pending_runtime_integration",
            },
            LIDAR_IMU: {
                "state": "DEFERRED",
                "reason": "lio_backend_and_linear_acceleration_pending_runtime_integration",
            },
        },
    }
    write(
        root / "localization_selector/localization_summary.json",
        {
            "schema_version": 1,
            "status": "PASS_WITH_DEVIATION",
            "decision_count": 50,
            "output_count": 49,
            "switch_count": 1,
            "selection_counts": {ISAAC_GT: 50},
            "odometry_deferred_decision_count": 50,
            "maximum_switch_translation_jump_m": 0.0,
            "maximum_switch_rotation_jump_rad": 0.0,
            "selector": {
                "schema_version": 1,
                "runtime_policy": "completion_sim",
                "runtime_target": "isaac_simulation",
                "generation": 9,
                "selected_source": ISAAC_GT,
                "current_output_available": True,
                "switch_count": 1,
                "deviation_count": 49,
                "no_output_count": 1,
                "odometry_deferred": True,
                "sensor_odometry_status": "STRICT_EXTENSION_PENDING",
            },
            "deviation_artifact": "localization_deviations.json",
        },
    )
    write(
        root / "localization_selector/localization_deviations.json",
        {
            "schema_version": 1,
            "status": "WARN" if with_valid_deviation else "NONE",
            "odometry_deferred": with_valid_deviation,
            "deviation_count": 50 if with_valid_deviation else 0,
            "deviations": [deviation] if with_valid_deviation else [],
        },
    )
    records: list[dict[str, object]] = []
    for index in range(50):
        output = None
        if index != 1:
            output = {
                "source": ISAAC_GT,
                "generation": 9,
                "sequence_id": index,
                "stamp_ns": 100 + index,
                "parent_frame": "odom",
                "child_frame": "base_link",
                "pose": {
                    "translation_xyz": [float(index), 0.0, 0.0],
                    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "linear_velocity_xyz": [0.0, 0.0, 0.0],
                "angular_velocity_xyz": [0.0, 0.0, 0.0],
                "odometry_topic": "/odom",
                "tf_topic": "/tf",
            }
        health: dict[str, object] = {}
        for source, state, reason in (
            (
                CUVSLAM,
                "DEFERRED",
                "stereo_feed_and_backend_pending_runtime_integration",
            ),
            (
                LIDAR_IMU,
                "DEFERRED",
                "lio_backend_and_linear_acceleration_pending_runtime_integration",
            ),
            (ISAAC_GT, "HEALTHY", "tracking"),
        ):
            health[source] = {
                "source": source,
                "configured": True,
                "state": state,
                "reason": reason,
                "generation": 9,
                "last_stamp_ns": 100 if index == 1 else 100 + index,
                "age_sec": None if source != ISAAC_GT else 0.0,
                "stamp_age_sec": None if source != ISAAC_GT else 0.0,
                "valid_count": 0 if source != ISAAC_GT else index + 1,
                "invalid_count": 0,
                "ordered_count": 0 if source != ISAAC_GT else index + 1,
                "backend_ready": source == ISAAC_GT,
                "consecutive_healthy_decisions": 0 if source != ISAAC_GT else index + 1,
            }
        records.append(
            {
                "schema_version": 1,
                "event": "localization_selection",
                "runtime_policy": "completion_sim",
                "runtime_target": "isaac_simulation",
                "generation": 9,
                "decision_monotonic_ns": 1_000 + index,
                "previous_source": None if index == 0 else ISAAC_GT,
                "selected_source": ISAAC_GT,
                "switch_reason": (
                    "ALL_SENSOR_ODOMETRY_DEFERRED"
                    if index == 0
                    else (
                        "CANDIDATE_TIMESTAMP_NOT_NEWER"
                        if index == 1
                        else "ACTIVE_HEALTHY"
                    )
                ),
                "switch_event": index == 0,
                "source_health": health,
                "output": output,
                "odometry_deferred": True,
                "deviation": deviation if with_valid_deviation else None,
                "switch_translation_jump_m": None,
                "switch_rotation_jump_rad": None,
            }
        )
    record_path = root / "localization_selector/localization_records.jsonl"
    record_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    write(
        root / "oracle_episode_summary.json",
        {
            "schema_version": 1,
            "status": "PASS",
            "episode_count": 10,
            "success_count": 8,
            "collision_count": 0,
            "fall_count": 0,
            "generation_contamination_count": 0,
            "stale_motion_count": 0,
        },
    )
    write(
        root / "localization_cleanup.json",
        {
            "schema_version": 1,
            "status": "PASS",
            "pid_count": 0,
            "pgid_count": 0,
            "socket_count": 0,
        },
    )


def test_completion_oracle_accepts_gt_only_with_explicit_deviation(
    tmp_path: Path,
) -> None:
    fixture_result(tmp_path)
    payload = validate_oracle_result(tmp_path, GATE)
    assert payload["status"] == "PASS"
    assert payload["functional_disposition"] == "PASS_WITH_DEVIATION"
    assert payload["sensor_odometry_status"] == "STRICT_EXTENSION_PENDING"
    assert payload["gt_fallback_selection_count"] == 50


def test_completion_oracle_rejects_silent_gt_fallback(tmp_path: Path) -> None:
    fixture_result(tmp_path, with_valid_deviation=False)
    payload = validate_oracle_result(tmp_path, GATE)
    assert payload["status"] == "FAIL"
    assert "gt_fallback_deviation" in payload["errors"]


def test_validation_is_exclusive_and_cleanup_is_required(tmp_path: Path) -> None:
    fixture_result(tmp_path)
    payload = write_validation(tmp_path, GATE)
    assert payload["status"] == "PASS"
    with pytest.raises(FileExistsError):
        write_validation(tmp_path, GATE)

    second = tmp_path / "bad-cleanup"
    fixture_result(second)
    cleanup = second / "localization_cleanup.json"
    value = json.loads(cleanup.read_text())
    value["pgid_count"] = 1
    write(cleanup, value)
    failed = validate_oracle_result(second, GATE)
    assert failed["status"] == "FAIL"
    assert "cleanup" in failed["errors"]


def test_validation_rejects_negative_jump_and_counter_mismatch(
    tmp_path: Path,
) -> None:
    fixture_result(tmp_path)
    summary = tmp_path / "localization_selector/localization_summary.json"
    value = json.loads(summary.read_text(encoding="utf-8"))
    value["maximum_switch_translation_jump_m"] = -1.0
    value["selector"]["switch_count"] = 0
    write(summary, value)
    payload = validate_oracle_result(tmp_path, GATE)
    assert payload["status"] == "FAIL"
    assert "translation_continuity" in payload["errors"]
    assert "counter_bounds" in payload["errors"]


def test_failed_structural_validation_does_not_leave_reserved_output(
    tmp_path: Path,
) -> None:
    fixture_result(tmp_path)
    summary = tmp_path / "localization_selector/localization_summary.json"
    value = json.loads(summary.read_text(encoding="utf-8"))
    value["unexpected"] = True
    write(summary, value)
    with pytest.raises(ValueError, match="keys_mismatch"):
        write_validation(tmp_path, GATE)
    assert not (tmp_path / "localization_validation.json").exists()
    assert not (tmp_path / "localization_validation.json.tmp").exists()


def test_raw_records_restart_source_chain_and_stamp_at_generation_boundary(
    tmp_path: Path,
) -> None:
    fixture_result(tmp_path)
    source_path = tmp_path / "localization_selector/localization_records.jsonl"
    first = json.loads(source_path.read_text(encoding="utf-8").splitlines()[0])
    second = json.loads(json.dumps(first))
    second["generation"] = 10
    second["decision_monotonic_ns"] = 2_000
    second["previous_source"] = None
    second["switch_event"] = True
    second["output"]["generation"] = 10
    second["output"]["sequence_id"] = 0
    second["output"]["stamp_ns"] = 1
    for health in second["source_health"].values():
        health["generation"] = 10
        health["last_stamp_ns"] = 1
    two_generation = tmp_path / "two-generation-records.jsonl"
    two_generation.write_text(
        json.dumps(first) + "\n" + json.dumps(second) + "\n",
        encoding="utf-8",
    )
    aggregates = _record_aggregates(two_generation)
    assert aggregates["decision_count"] == 2
    assert aggregates["output_count"] == 2
    assert aggregates["switch_count"] == 2


def test_validator_accepts_gt_history_followed_by_sensor_recovery(
    tmp_path: Path,
) -> None:
    fixture_result(tmp_path)
    selector_path = tmp_path / "localization_selector/localization_summary.json"
    selector = json.loads(selector_path.read_text(encoding="utf-8"))
    selector["switch_count"] = 2
    selector["selection_counts"] = {ISAAC_GT: 49, CUVSLAM: 1}
    selector["odometry_deferred_decision_count"] = 49
    selector["selector"].update(
        {
            "selected_source": CUVSLAM,
            "switch_count": 2,
            "deviation_count": 48,
            "odometry_deferred": False,
            "sensor_odometry_status": "STRICT_EXTENSION_PENDING",
        }
    )
    write(selector_path, selector)
    deviations_path = tmp_path / "localization_selector/localization_deviations.json"
    deviations = json.loads(deviations_path.read_text(encoding="utf-8"))
    deviations["deviation_count"] = 49
    write(deviations_path, deviations)
    records_path = tmp_path / "localization_selector/localization_records.jsonl"
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
    ]
    final = records[-1]
    final.update(
        {
            "previous_source": ISAAC_GT,
            "selected_source": CUVSLAM,
            "switch_reason": "RECOVERED_AFTER_HYSTERESIS",
            "switch_event": True,
            "odometry_deferred": False,
            "deviation": None,
            "switch_translation_jump_m": 0.0,
            "switch_rotation_jump_rad": 0.0,
        }
    )
    final["output"]["source"] = CUVSLAM
    final["output"]["pose"]["translation_xyz"] = [48.0, 0.0, 0.0]
    final_health = final["source_health"][CUVSLAM]
    final_health.update(
        {
            "state": "HEALTHY",
            "reason": "tracking",
            "last_stamp_ns": final["output"]["stamp_ns"],
            "age_sec": 0.0,
            "stamp_age_sec": 0.0,
            "valid_count": 1,
            "ordered_count": 1,
            "backend_ready": True,
            "consecutive_healthy_decisions": 1,
        }
    )
    records_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    payload = validate_oracle_result(tmp_path, GATE)
    assert payload["status"] == "PASS"
    assert payload["functional_disposition"] == "PASS_WITH_DEVIATION"


def test_raw_records_recompute_switch_jump_from_canonical_poses(
    tmp_path: Path,
) -> None:
    fixture_result(tmp_path)
    records_path = tmp_path / "localization_selector/localization_records.jsonl"
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
    ]
    final = records[-1]
    final.update(
        {
            "previous_source": ISAAC_GT,
            "selected_source": CUVSLAM,
            "switch_reason": "RECOVERED_AFTER_HYSTERESIS",
            "switch_event": True,
            "odometry_deferred": False,
            "deviation": None,
            "switch_translation_jump_m": 0.0,
            "switch_rotation_jump_rad": 0.0,
        }
    )
    final["output"]["source"] = CUVSLAM
    records_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="switch_jump_measurement_mismatch"):
        _record_aggregates(records_path)
